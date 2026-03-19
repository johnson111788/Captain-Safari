#!/usr/bin/env python3
"""
Convert camera_fixed data from global coordinate system to local coordinate system (first frame as origin)

Target: make the first frame camera position to (0, 0, 0), and the orientation to the identity matrix
So that it can be used normally in wan2.2
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def convert_to_local_coordinates(extrinsics):
    """
    Convert extrinsics from global coordinate system to local coordinate system (first frame as origin)
    
    Args:
        extrinsics: (N, 3, 4) extrinsics in global coordinate system (w2c format)
    
    Returns:
        extrinsics_local: (N, 3, 4) extrinsics in local coordinate system
    """
    N = extrinsics.shape[0]
    
    # Extrinsics of the first frame
    R0 = extrinsics[0, :, :3]  # Rotation matrix of the first frame
    t0 = extrinsics[0, :, 3]    # Translation vector of the first frame
    
    # Calculate the camera center of the first frame (world coordinate system)
    C0 = -(R0.T @ t0)
    
    # c2w matrix of the first frame
    R0_c2w = R0.T
    t0_c2w = C0
    
    # Build the c2w 4x4 matrix of the first frame
    T0_c2w = np.eye(4)
    T0_c2w[:3, :3] = R0_c2w
    T0_c2w[:3, 3] = t0_c2w
    
    # The inverse matrix (w2c) is the reference transformation
    T0_w2c = np.linalg.inv(T0_c2w)
    
    # Convert all frames to local coordinate system
    extrinsics_local = np.zeros_like(extrinsics)
    
    for i in range(N):
        # Extrinsics of the current frame
        Ri = extrinsics[i, :, :3]
        ti = extrinsics[i, :, 3]
        
        # Build the 4x4 w2c matrix
        Ti_w2c = np.eye(4)
        Ti_w2c[:3, :3] = Ri
        Ti_w2c[:3, 3] = ti
        
        # Convert to local coordinate system: T_local = T0_w2c @ T_global_c2w @ T_i_w2c
        # Simplify: T_local = T0_w2c @ inv(T_i_w2c)
        Ti_c2w = np.linalg.inv(Ti_w2c)
        Ti_local_c2w = T0_w2c @ Ti_c2w
        Ti_local_w2c = np.linalg.inv(Ti_local_c2w)
        
        # Extract the 3x4 extrinsics
        extrinsics_local[i] = Ti_local_w2c[:3, :]
    
    return extrinsics_local


def verify_conversion(extrinsics_original, extrinsics_local):
    """Verify the conversion result"""
    print("\nVerify the conversion result")
    print("=" * 80)
    
    # Check the first frame
    R0_local = extrinsics_local[0, :, :3]
    t0_local = extrinsics_local[0, :, 3]
    C0_local = -(R0_local.T @ t0_local)
    
    print(f"\nOriginal first frame:")
    print(f"  R0:\n{extrinsics_original[0, :, :3]}")
    print(f"  t0: {extrinsics_original[0, :, 3]}")
    R0_orig = extrinsics_original[0, :, :3]
    t0_orig = extrinsics_original[0, :, 3]
    C0_orig = -(R0_orig.T @ t0_orig)
    print(f"  Camera center: {C0_orig}")
    print(f"  Distance from origin: {np.linalg.norm(C0_orig):.6f}")
    
    print(f"\nConverted first frame:")
    print(f"  R0:\n{R0_local}")
    print(f"  t0: {t0_local}")
    print(f"  Camera center: {C0_local}")
    print(f"  Distance from origin: {np.linalg.norm(C0_local):.6f}")
    
    print(f"\nCheck the first frame:")
    is_identity = np.allclose(R0_local, np.eye(3), atol=1e-6)
    is_zero_t = np.allclose(t0_local, 0, atol=1e-6)
    is_zero_C = np.allclose(C0_local, 0, atol=1e-6)
    
    print(f"  R0 is an identity matrix? {'✓ Yes' if is_identity else '✗ No'}")
    print(f"  t0 is a zero vector? {'✓ Yes' if is_zero_t else '✗ No'}")
    print(f"  Camera center is at the origin? {'✓ Yes' if is_zero_C else '✗ No'}")
    
    if is_identity and is_zero_C:
        print("\n✅ Conversion successful! The first frame camera position has been set to the origin")
    else:
        print("\n⚠️ Conversion may have problems, please check")
    
    # Check if the relative relationship is maintained
    print(f"\nRelative relationship verification (check the change of the second frame relative to the first frame):")
    if extrinsics_original.shape[0] > 1:
        # Relative transformation in the original coordinate system
        R0_orig = extrinsics_original[0, :, :3]
        t0_orig = extrinsics_original[0, :, 3]
        C0_orig = -(R0_orig.T @ t0_orig)
        
        R1_orig = extrinsics_original[1, :, :3]
        t1_orig = extrinsics_original[1, :, 3]
        C1_orig = -(R1_orig.T @ t1_orig)
        
        relative_pos_orig = C1_orig - C0_orig
        
        # Relative transformation in the local coordinate system
        R1_local = extrinsics_local[1, :, :3]
        t1_local = extrinsics_local[1, :, 3]
        C1_local = -(R1_local.T @ t1_local)
        
        relative_pos_local = C1_local  # Because C0_local = 0
        
        print(f"  Original relative position: {relative_pos_orig}")
        print(f"  Local relative position: {relative_pos_local}")
        print(f"  Difference: {np.linalg.norm(relative_pos_orig - relative_pos_local):.6e}")
        
        # Rotation difference
        R_rel_orig = R0_orig.T @ R1_orig
        R_rel_local = R1_local  # Because R0_local = I
        
        print(f"  Rotation difference: {np.linalg.norm(R_rel_orig - R_rel_local, 'fro'):.6e}")


def main():
    """Main function: convert the entire directory"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert camera parameters from global coordinate system to local coordinate system (first frame as origin)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data-frames_4fps-camera-fixed-split",
        help="Input directory (global coordinate system)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data-frames_4fps-camera-fixed-split-local",
        help="Output directory (local coordinate system)"
    )
    parser.add_argument(
        "--test-video",
        type=str,
        default=None,
        help="Test single video ID (optional)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 100)
    print("Convert camera parameters to local coordinate system (first frame as origin)")
    print("=" * 100)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all extrinsic files
    extrinsic_files = sorted(input_dir.glob("*_extrinsic.npy"))
    
    if args.test_video:
        # Only process test video
        extrinsic_files = [f for f in extrinsic_files if args.test_video in f.name]
        print(f"\n🔍 Test mode: only process video {args.test_video}")
    
    print(f"\nFound {len(extrinsic_files)} videos")
    
    success_count = 0
    failed_count = 0
    
    for ext_file in tqdm(extrinsic_files, desc="Conversion progress"):
        try:
            video_id = ext_file.stem.replace("_extrinsic", "")
            int_file = input_dir / f"{video_id}_intrinsic.npy"
            
            # Load extrinsics and intrinsics
            extrinsics = np.load(ext_file)#.squeeze(1)
            intrinsics = np.load(int_file)
            
            # Convert extrinsics to local coordinate system
            extrinsics_local = convert_to_local_coordinates(extrinsics)
            
            # Save (intrinsics不变)
            output_ext_file = output_dir / f"{video_id}_extrinsic.npy"
            output_int_file = output_dir / f"{video_id}_intrinsic.npy"
            
            np.save(output_ext_file, extrinsics_local[:, np.newaxis, :, :])
            np.save(output_int_file, intrinsics)
            
            # If it is test mode, display detailed information
            if args.test_video:
                print(f"\n{'='*80}")
                print(f"Video ID: {video_id}")
                print(f"{'='*80}")
                verify_conversion(extrinsics, extrinsics_local)
            
            success_count += 1
            
        except Exception as e:
            failed_count += 1
            print(f"\n❌ Process {ext_file.name} failed: {e}")
            continue
    
    # Statistics
    print(f"\n{'='*100}")
    print("Conversion completed")
    print(f"{'='*100}")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📁 Output directory: {output_dir}/")
    
    print(f"\n{'='*100}")
    print("Next step")
    print(f"{'='*100}")
    print(f"""
1. Check the conversion result:
   python {__file__} --test-video <video_id>

2. Use the converted data in wan2.2:
   - Use the data in {output_dir}/ directory
   - It should not crash anymore!

3. Compare the effect:
   - streamvggt: the first frame camera distance from origin ~0.008
   - camera_local: the first frame camera distance from origin ~0.000
   - Both should work normally in wan2.2
""")


if __name__ == "__main__":
    main()

