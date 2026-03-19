#!/usr/bin/env python3
"""
Convert HLOC data to VGGT format

Based on analysis results:
- VGGT format:
  * extrinsic: (N, 3, 4) - 3x4 projection matrix [R|t] (w2c format)
  * intrinsic: (N, 3, 3) - 3x3 intrinsic matrix
- HLOC format: separate camera and image data

This script converts HLOC camera parameters to VGGT sequence format
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add hloc module to Python path
current_dir = Path(__file__).parent
hloc_path = current_dir / "hloc"
if hloc_path.exists():
    sys.path.insert(0, str(current_dir))

from hloc.utils.read_write_model import read_cameras_binary, read_images_binary


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def convert_hloc_to_vggt_sequence(sfm_dir="outputs/0df0f621-205e-4b48-8832-fdccddc5509c/sfm", output_prefix="hloc_sequence_corrected"):
    """Convert HLOC data to VGGT sequence format
    
    Correction explanation:
    1. Position correction: t vector is negated
    2. Orientation correction: xyz axis of rotation matrix is negated (correct orientation direction)
    """
    
    cameras_path = os.path.join(sfm_dir, "cameras.bin")
    images_path = os.path.join(sfm_dir, "images.bin")
    
    # Load HLOC data
    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)
    
    # Sort by image ID, ensure sequence consistency
    sorted_images = sorted(images.items(), key=lambda x: x[0])
    
    num_frames = len(sorted_images)
    # import ipdb; ipdb.set_trace()
    # Initialize sequence array
    extrinsics_sequence = np.zeros((num_frames, 3, 4), dtype=np.float32)
    intrinsics_sequence = np.zeros((num_frames, 3, 3), dtype=np.float32)
    
    for idx, (image_id, image) in enumerate(sorted_images):
        camera = cameras[image.camera_id]
        
        # Convert extrinsics: qvec, tvec -> [R|t] (3x4)
        R = qvec2rotmat(image.qvec)  # 3x3
        t = image.tvec.reshape(3, 1)  # 3x1
        
        # Coordinate system conversion: HLOC -> VGGT 
        # Based on visualization verification results:
        
        # Build 3x4 projection matrix [R|t]
        extrinsic_3x4 = np.hstack([R, t])  # (3, 4)
        extrinsics_sequence[idx] = extrinsic_3x4
        
        # Convert intrinsics: SIMPLE_RADIAL -> K matrix
        if camera.model == "SIMPLE_RADIAL":
            f = camera.params[0]   # focal length
            cx = camera.params[1]  # principal point x
            cy = camera.params[2]  # principal point y
            # k1 = camera.params[3]  # distortion coefficient (not included in K matrix)
            
            K = np.array([
                [f,  0,  cx],
                [0,  f,  cy],
                [0,  0,  1 ]
            ], dtype=np.float32)
            
            intrinsics_sequence[idx] = K
        else:
            print(f"Warning: camera {camera.camera_id} uses unsupported model {camera.model}")
    
    print("extrinsic_sequence shape: ", extrinsics_sequence.shape)
    # Save as VGGT format
    extrinsic_file = f"{output_prefix}_extrinsic.npy"
    intrinsic_file = f"{output_prefix}_intrinsic.npy"
    
    np.save(extrinsic_file, extrinsics_sequence)
    np.save(intrinsic_file, intrinsics_sequence)
        
    return extrinsics_sequence, intrinsics_sequence





def process_all_outputs(outputs_dir="outputs", camera_output_dir="camera", target_cases=None):
    """
    Traverse all outputs folders, generate camera parameters file for each UUID
    
    Args:
        outputs_dir: output directory containing all UUID folders
        camera_output_dir: output directory for camera parameters file
        target_cases: comma-separated target case ID list string (optional)
    """
    print("="*80)
    print("HLOC to VGGT format batch conversion tool")
    print("="*80)
    
    # Create camera output directory
    os.makedirs(camera_output_dir, exist_ok=True)
    print(f"📁 Output directory: {camera_output_dir}/")
    
    # Get all UUID folders
    if not os.path.exists(outputs_dir):
        print(f"❌ Error: output directory {outputs_dir} does not exist!")
        return
    
    # If target_cases is specified, only process these cases
    if target_cases:
        target_case_ids = set(target_cases.split(','))
        uuid_folders = []
        for case_id in target_case_ids:
            uuid_path = os.path.join(outputs_dir, case_id)
            sfm_path = os.path.join(uuid_path, "sfm")
            if os.path.isdir(uuid_path) and os.path.exists(sfm_path):
                uuid_folders.append(case_id)
        print(f"🔍 Target processing {len(target_case_ids)} specified cases, found {len(uuid_folders)} valid UUID folders")
    else:
        # Process all folders
        uuid_folders = []
        for item in os.listdir(outputs_dir):
            uuid_path = os.path.join(outputs_dir, item)
            sfm_path = os.path.join(uuid_path, "sfm")
            if os.path.isdir(uuid_path) and os.path.exists(sfm_path): #  and os.path.exists(os.path.join('camera.train_fixed.tighter.v1', item+"_extrinsic.npy")) and os.path.exists(os.path.join('camera.train_fixed', item+"_extrinsic.npy"))
                uuid_folders.append(item)
        print(f"🔍 Found {len(uuid_folders)} valid UUID folders")
    
    if not uuid_folders:
        print(f"❌ No valid UUID folders found in {outputs_dir}!")
        return
    
    # Statistics
    success_count = 0
    failed_count = 0
    # uuid_folders = ['0df0f621-205e-4b48-8832-fdccddc5509c']
    
    # Process each UUID folder
    for i, uuid in enumerate(uuid_folders, 1):
        print(f"\n" + "="*60)
        print(f"Processing [{i}/{len(uuid_folders)}]: {uuid}")
        print("="*60)
        
        try:
            # Build path
            sfm_dir = os.path.join(outputs_dir, uuid, "sfm")
            
            # Convert sequence
            hloc_ext, hloc_int = convert_hloc_to_vggt_sequence(
                sfm_dir=sfm_dir, 
                output_prefix=f"{camera_output_dir}/{uuid}"
            )
            
            success_count += 1
            
        except Exception as e:
            failed_count += 1
            print(f"❌ {uuid} processing failed: {str(e)}")
            continue
    
    # Final statistics
    print(f"\n" + "="*80)
    print("Batch processing completed")
    print("="*80)
    print(f"📊 Processing statistics:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📁 Total: {len(uuid_folders)}")
    print(f"💾 All files saved to: {camera_output_dir}/")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HLOC to VGGT format conversion - support batch and incremental processing"
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="./data-frames_4fps-hloc",
        help="HLOC output directory"
    )
    parser.add_argument(
        "--camera-dir",
        type=str,
        default="./data-frames_4fps-camera",
        help="camera parameters output directory"
    )
    parser.add_argument(
        "--target-cases",
        type=str,
        default=None,
        help="comma-separated target case ID list (if specified, only process these cases)"
    )
    
    args = parser.parse_args()
    
    # Process all outputs folders or specified cases
    process_all_outputs(args.outputs_dir, args.camera_dir, args.target_cases)

if __name__ == "__main__":
    main()
