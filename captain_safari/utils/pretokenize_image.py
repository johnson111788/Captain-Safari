"""
Pre-processing script: Pre-encode the input_image (first frame) using VAE
for the camera control task in WanVideoUnit_FunCameraControl

Usage:
    python utils/pretokenize_image.py \
        --dataset_base_path ./data/ \
        --dataset_metadata_path ./data/metadata.csv \
        --output_dir ./data/ \
        --num_frames 121 \
        --height 704 \
        --width 1280 \
        --gpu_id 0 \
        --num_gpus 8
"""

import torch
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import imageio
import torchvision.transforms.functional as TF

# Import WanVideo pipeline
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


def load_first_frame_from_video(video_path):
    """Load the first frame from the video"""
    reader = imageio.get_reader(video_path)
    frame = reader.get_data(0)
    frame = Image.fromarray(frame).convert("RGB")
    reader.close()
    return frame


def load_image(image_path):
    """Load the image"""
    return Image.open(image_path).convert("RGB")


def is_video_file(file_path):
    """Check if the file is a video file"""
    video_extensions = ('.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv', '.webm')
    return file_path.lower().endswith(video_extensions)


def is_image_file(file_path):
    """Check if the file is an image file"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    return file_path.lower().endswith(image_extensions)


def crop_and_resize(image, target_height, target_width):
    """
    Crop and resize the image
    Keep the same as VideoDataset.crop_and_resize
    """
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = TF.resize(
        image,
        (round(height * scale), round(width * scale)),
        interpolation=TF.InterpolationMode.BILINEAR
    )
    image = TF.center_crop(image, (target_height, target_width))
    return image


@torch.no_grad()
def encode_image_for_camera_control(pipe, image, height, width, num_frames, device='cuda', tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
    """
    Encode the image in the same way as WanVideoUnit_FunCameraControl
    
    This function simulates the processing of input_image in WanVideoUnit_FunCameraControl:
    1. Resize the image
    2. Pre-process the image as a video format (single frame)
    3. VAE encode
    4. Create mask and concatenate
    
    Returns:
        y: The complete tensor containing mask and latent, shape [1, 52, T, H, W]
        input_latents_single: The single frame latent, shape [1, 48, 1, H, W]
    """
    pipe.load_models_to_device(['vae'])
    
    # Step 1: Resize the image
    image = crop_and_resize(image, height, width)
    
    # Step 2: Pre-process the image as a video format (参考 WanVideoUnit_FunCameraControl 第 940 行）
    image_tensor = pipe.preprocess_image(image).to(pipe.device)
    
    # Method A: Encode the single frame (for y[:, :, :1] = input_latents)
    input_video_single = pipe.preprocess_video([image])  # [1, 3, 1, H, W]
    input_latents_single = pipe.vae.encode(input_video_single, device=device).to(dtype=pipe.torch_dtype, device='cpu')  # [1, 48, 1, H, W]
    
    # Method B: Create the complete y (first frame + zeros, with mask)
    # Reference WanVideoUnit_FunCameraControl line 948-958
    vae_input = torch.concat([
        image_tensor.transpose(0, 1),  # [3, 1, H, W]
        torch.zeros(3, num_frames-1, height, width, device=device)  # [3, num_frames-1, H, W]
    ], dim=1)  # [3, num_frames, H, W]
    
    # VAE encode
    y_latent = pipe.vae.encode(
        [vae_input.to(dtype=pipe.torch_dtype, device=device)], 
        device=device, 
        tiled=tiled, 
        tile_size=tile_size, 
        tile_stride=tile_stride
    )[0]  # [48, T, H/16, W/16]
    
    # Create mask
    msk = torch.ones(1, num_frames, height//pipe.vae.upsampling_factor, width//pipe.vae.upsampling_factor, device=device)
    msk[:, 1:] = 0  # Only the first frame is 1
    
    # Adjust mask to match time compression
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
        msk[:, 1:]
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, height//pipe.vae.upsampling_factor, width//pipe.vae.upsampling_factor)
    msk = msk.transpose(1, 2)[0]  # [4, T, H/16, W/16]
    
    # Concatenate mask and latent
    y = torch.cat([msk.cpu(), y_latent.cpu()])  # [52, T, H/16, W/16]
    y = y.unsqueeze(0)  # [1, 52, T, H/16, W/16]
    y = y.to(dtype=pipe.torch_dtype)
    
    return y, input_latents_single


def main():
    parser = argparse.ArgumentParser(description='Pre-process input_image for camera control')
    
    # Dataset parameters
    parser.add_argument('--dataset_base_path', type=str, default='./data/', help='Dataset base path')
    parser.add_argument('--dataset_metadata_path', type=str, default='./data/metadata.csv', help='Dataset metadata CSV file path')
    parser.add_argument('--output_dir', type=str, default='./data/', help='Output directory')
    
    # Model parameters
    parser.add_argument('--model_paths', type=str, default=None, help='Model path list (JSON format)')
    parser.add_argument('--model_id_with_origin_paths', type=str, default="PAI/Wan2.2-Fun-5B-Control-Camera:Wan2.2_VAE.pth", help='Model ID and file matching pattern (only VAE)')
    
    # Video parameters
    parser.add_argument('--num_frames', type=int, default=121, help='Number of frames')
    parser.add_argument('--height', type=int, default=704, help='Height')
    parser.add_argument('--width', type=int, default=1280, help='Width')
    
    # VAE parameters
    parser.add_argument('--tiled', action='store_true', help='Whether to use tiled VAE encoding')
    parser.add_argument('--tile_size_h', type=int, default=34, help='Tile height')
    parser.add_argument('--tile_size_w', type=int, default=34, help='Tile width')
    parser.add_argument('--tile_stride_h', type=int, default=18, help='Tile stride height')
    parser.add_argument('--tile_stride_w', type=int, default=16, help='Tile stride width')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--start_idx', type=int, default=None, help='Start index (manual specified, for resume)')
    parser.add_argument('--end_idx', type=int, default=None, help='End index (manual specified, for processing partial data)')
    
    # Multi-GPU parallel parameters
    parser.add_argument('--gpu_id', type=int, default=0, help='Current GPU ID (0-based)')
    parser.add_argument('--num_gpus', type=int, default=1, help='Total number of GPUs')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    image_y_dir = output_dir / 'image_y_latents'  # The complete y (contains mask)
    image_single_dir = output_dir / 'image_single_latents'  # The single frame latent
    image_y_dir.mkdir(parents=True, exist_ok=True)
    image_single_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print(f"Loading metadata: {args.dataset_metadata_path}")
    metadata = pd.read_csv(args.dataset_metadata_path)
    total_samples = len(metadata)
    print(f"Total {total_samples} samples")
    
    # Determine the processing range (support multi-GPU data splitting)
    if args.start_idx is not None and args.end_idx is not None:
        start_idx = args.start_idx
        end_idx = args.end_idx
        print(f"[Manual specified] Processing range: [{start_idx}, {end_idx})")
    else:
        samples_per_gpu = (total_samples + args.num_gpus - 1) // args.num_gpus
        start_idx = args.gpu_id * samples_per_gpu
        end_idx = min((args.gpu_id + 1) * samples_per_gpu, total_samples)
        
        if args.num_gpus > 1:
            print(f"[Multi-GPU mode] GPU {args.gpu_id}/{args.num_gpus - 1}: Processing range [{start_idx}, {end_idx}) / {total_samples}")
        else:
            print(f"[Single-GPU mode] Processing range: [{start_idx}, {end_idx})")
    
    end_idx = min(end_idx, total_samples)
    
    # Set CUDA device
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(0)
        print(f"Using device: {device} (Physical GPU ID: {args.gpu_id})")
    else:
        device = 'cpu'
        print("Warning: CUDA is not available, using CPU")
    
    # Load model (only load VAE)
    print("Loading VAE model...")
    model_configs = []
    if args.model_paths is not None:
        model_paths = json.loads(args.model_paths)
        model_configs += [ModelConfig(path=path) for path in model_paths]
    if args.model_id_with_origin_paths is not None:
        model_id_with_origin_paths = args.model_id_with_origin_paths.split(",")
        model_configs += [
            ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) 
            for i in model_id_with_origin_paths
        ]
    
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, 
        device=device, 
        model_configs=model_configs
    )
    print("VAE model loaded")
    
    # Process data
    tile_size = (args.tile_size_h, args.tile_size_w)
    tile_stride = (args.tile_stride_h, args.tile_stride_w)
    
    # Statistics
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    progress_desc = f"GPU {args.gpu_id}" if args.num_gpus > 1 else "Processing data"
    
    for idx in tqdm(range(start_idx, end_idx), desc=progress_desc):
        row = metadata.iloc[idx]
        
        # Generate output file name (using index)
        image_y_output_path = image_y_dir / f"{idx:08d}.pt"
        image_single_output_path = image_single_dir / f"{idx:08d}.pt"
        
        # Skip already processed data
        if image_y_output_path.exists() and image_single_output_path.exists():
            skipped_count += 1
            continue
        
        try:
            # Get video/image path
            video_path = os.path.join(args.dataset_base_path, row['video'])
            
            # Load the first frame
            if is_video_file(video_path):
                image = load_first_frame_from_video(video_path)
            elif is_image_file(video_path):
                image = load_image(video_path)
            else:
                print(f"\nWarning: Unrecognized file type: {video_path}")
                error_count += 1
                continue
            
            # Encode image
            y, input_latents_single = encode_image_for_camera_control(
                pipe, image,
                args.height, args.width, args.num_frames,
                device=device,
                tiled=args.tiled,
                tile_size=tile_size,
                tile_stride=tile_stride
            )
            
            # Save results
            torch.save(y, image_y_output_path)
            torch.save(input_latents_single, image_single_output_path)
            
            processed_count += 1
            
        except Exception as e:
            print(f"\nError processing {idx}th data: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"GPU {args.gpu_id} processing completed:")
    print(f"  - Processing range: [{start_idx}, {end_idx})")
    print(f"  - New processed: {processed_count}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  - Error count: {error_count}")
    print(f"{'='*60}\n")
    
    # Only create complete metadata file on GPU 0 or single GPU mode
    if args.gpu_id == 0 or args.num_gpus == 1:
        print("\nUpdating metadata file...")
        
        # If the input metadata already has pre-encoded fields, add them on top
        new_metadata = metadata.copy()
        new_metadata['image_y_latent'] = [f"image_y_latents/{i:08d}.pt" for i in range(len(metadata))]
        new_metadata['image_single_latent'] = [f"image_single_latents/{i:08d}.pt" for i in range(len(metadata))]
        
        # Determine output file name
        input_filename = os.path.basename(args.dataset_metadata_path)
        if '.encoded' in input_filename:
            # If already contains .encoded, replace with .encoded_with_image
            output_filename = input_filename.replace('.encoded.csv', '.encoded_with_image.csv')
        else:
            # Otherwise add .encoded_with_image
            output_filename = input_filename.replace('.csv', '.encoded_with_image.csv')
        
        new_metadata_path = output_dir / output_filename
        new_metadata.to_csv(new_metadata_path, index=False)
        print(f"New metadata saved to: {new_metadata_path}")
        
        print("\nPre-processing completed!")
        print(f"- Image Y latents: {image_y_dir}")
        print(f"- Image single latents: {image_single_dir}")
        print(f"- New metadata: {new_metadata_path}")
    else:
        print(f"\nGPU {args.gpu_id} processing completed! Waiting for other GPUs...")
        print(f"- The output of this GPU has been saved to {output_dir}")


if __name__ == '__main__':
    main()

