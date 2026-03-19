"""
Pre-processing script: Pre-encode prompts and videos to improve training speed

Single GPU usage method:
    python utils/pretokenize.py \
        --dataset_base_path /path/to/dataset \
        --dataset_metadata_path /path/to/metadata.csv \
        --output_dir /path/to/output \
        --model_paths '["path/to/model1", "path/to/model2"]' \
        --num_frames 121 \
        --height 704 \
        --width 1280

Multi-GPU parallel usage method (recommended):
    # Run on N GPUs in parallel
    for i in {0..7}; do
        CUDA_VISIBLE_DEVICES=$i python utils/pretokenize.py \
            --dataset_base_path /path/to/dataset \
            --dataset_metadata_path /path/to/metadata.csv \
            --output_dir /path/to/output \
            --gpu_id $i \
            --num_gpus 8 \
            ... &
    done
    wait
    
    # Or use the provided startup script
    bash utils/run_pretokenize_multi_gpu.sh
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


def load_video(file_path, num_frames):
    """Load video frames"""
    reader = imageio.get_reader(file_path)
    frames = []
    for i, frame in enumerate(reader):
        if i >= num_frames:
            break
        frame = Image.fromarray(frame).convert("RGB")
        frames.append(frame)
    reader.close()
    
    # If the number of video frames is less than the required number, repeat the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    return frames[:num_frames]


def load_image_as_video(file_path, num_frames):
    """Load a single image as a video (repeat multiple frames)"""
    image = Image.open(file_path).convert("RGB")
    return [image] * num_frames


def is_video_file(file_path):
    """Check if it is a video file"""
    video_extensions = ('.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv', '.webm')
    return file_path.lower().endswith(video_extensions)


def is_image_file(file_path):
    """Check if it is an image file"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    return file_path.lower().endswith(image_extensions)


def crop_and_resize(image, target_height, target_width):
    """
    Crop and resize the image
    与 VideoDataset.crop_and_resize 保持完全一致
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
def encode_prompt(pipe, prompt, device='cuda'):
    """Encode prompt"""
    pipe.load_models_to_device(['text_encoder'])
    prompt_emb = pipe.prompter.encode_prompt(prompt, positive=True, device=device)
    return prompt_emb.cpu()


@torch.no_grad()
def encode_video(pipe, video_frames, height, width, device='cuda', tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
    """Encode video to latents"""
    # Crop and resize all frames
    processed_frames = [crop_and_resize(frame, height, width) for frame in video_frames]
    
    # Pre-process video
    pipe.load_models_to_device(['vae'])
    input_video = pipe.preprocess_video(processed_frames)
    
    # VAE encode
    input_latents = pipe.vae.encode(
        input_video, 
        device=device, 
        tiled=tiled, 
        tile_size=tile_size, 
        tile_stride=tile_stride
    ).to(dtype=pipe.torch_dtype, device='cpu')
    
    return input_latents


def main():
    parser = argparse.ArgumentParser(description='Pre-process dataset: encode prompt and video')
    
    # Dataset parameters
    parser.add_argument('--dataset_base_path', type=str, default='./data', help='Dataset base path')
    parser.add_argument('--dataset_metadata_path', type=str, default='./data/metadata.csv', help='Dataset metadata CSV file path')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory')
    
    # Model parameters
    parser.add_argument('--model_paths', type=str, default=None, help='Model path list (JSON format)')
    parser.add_argument('--model_id_with_origin_paths', type=str, default="PAI/Wan2.2-Fun-5B-Control-Camera:diffusion_pytorch_model*.safetensors,PAI/Wan2.2-Fun-5B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-Fun-5B-Control-Camera:Wan2.2_VAE.pth", help='Model ID and file matching pattern')
    
    # Video parameters
    parser.add_argument('--num_frames', type=int, default=121, help='Number of video frames')
    parser.add_argument('--height', type=int, default=704, help='Height')
    parser.add_argument('--width', type=int, default=1280, help='Width')
    
    # VAE parameters
    parser.add_argument('--tiled', action='store_true', help='Whether to use tiled VAE encoding')
    parser.add_argument('--tile_size_h', type=int, default=34, help='Tile height')
    parser.add_argument('--tile_size_w', type=int, default=34, help='Tile width')
    parser.add_argument('--tile_stride_h', type=int, default=18, help='Tile stride height')
    parser.add_argument('--tile_stride_w', type=int, default=16, help='Tile stride width')
    
    # Other parameters
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (recommended to be 1)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--start_idx', type=int, default=None, help='Start index (manual specified, for resume)')
    parser.add_argument('--end_idx', type=int, default=None, help='End index (manual specified, for processing partial data)')
    
    # Multi-GPU parallel parameters
    parser.add_argument('--gpu_id', type=int, default=0, help='Current GPU ID (0-based)')
    parser.add_argument('--num_gpus', type=int, default=1, help='Total number of GPUs')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    prompt_dir = output_dir / 'prompt_embeddings'
    video_dir = output_dir / 'video_latents'
    prompt_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print(f"Loading metadata: {args.dataset_metadata_path}")
    metadata = pd.read_csv(args.dataset_metadata_path)
    total_samples = len(metadata)
    print(f"Total {total_samples} samples")
    
    # Determine processing range (support multi-GPU data splitting)
    if args.start_idx is not None and args.end_idx is not None:
        # Manually specify range (for resume)
        start_idx = args.start_idx
        end_idx = args.end_idx
        print(f"[Manual specified] Processing range: [{start_idx}, {end_idx})")
    else:
        # Multi-GPU automatic data splitting
        # Calculate the data range that the current GPU should process
        samples_per_gpu = (total_samples + args.num_gpus - 1) // args.num_gpus
        start_idx = args.gpu_id * samples_per_gpu
        end_idx = min((args.gpu_id + 1) * samples_per_gpu, total_samples)
        
        if args.num_gpus > 1:
            print(f"[Multi-GPU mode] GPU {args.gpu_id}/{args.num_gpus - 1}: Processing range [{start_idx}, {end_idx}) / {total_samples}")
        else:
            print(f"[Single-GPU mode] Processing range: [{start_idx}, {end_idx})")
    
    end_idx = min(end_idx, total_samples)
    
    # Set CUDA device (ensure the correct GPU is used)
    if torch.cuda.is_available():
        # If CUDA_VISIBLE_DEVICES is used, cuda:0 is the correct device
        device = 'cuda:0'
        torch.cuda.set_device(0)
        print(f"Using device: {device} (Physical GPU ID: {args.gpu_id})")
    else:
        device = 'cpu'
        print("Warning: CUDA is not available, using CPU")
    
    # Load model
    print("Loading model...")
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
    print("Model loaded")
    
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
        prompt_output_path = prompt_dir / f"{idx:08d}.pt"
        video_output_path = video_dir / f"{idx:08d}.pt"
        
        # Skip already processed data
        if prompt_output_path.exists() and video_output_path.exists():
            skipped_count += 1
            continue
        
        try:
            # Encode prompt
            if not prompt_output_path.exists():
                prompt = row['prompt']
                prompt_emb = encode_prompt(pipe, prompt, device=device)
                torch.save(prompt_emb, prompt_output_path)
            
            # Encode video
            if not video_output_path.exists():
                video_path = os.path.join(args.dataset_base_path, row['video'])
                
                # Load video
                if is_video_file(video_path):
                    video_frames = load_video(video_path, args.num_frames)
                elif is_image_file(video_path):
                    video_frames = load_image_as_video(video_path, args.num_frames)
                else:
                    print(f"\nWarning: Unrecognized file type: {video_path}")
                    error_count += 1
                    continue
                
                # Encode video
                video_latents = encode_video(
                    pipe, video_frames, 
                    args.height, args.width,
                    device=device,
                    tiled=args.tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride
                )
                torch.save(video_latents, video_output_path)
            
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
        print("\nCreating new metadata file...")
        
        # Check if all data has been processed
        if args.num_gpus > 1:
            print("Warning: In multi-GPU mode, please wait for all GPUs to complete before using metadata_encoded.csv")
        
        # Create complete metadata (contains references to all samples)
        new_metadata = metadata.copy()
        new_metadata['prompt_embedding'] = [f"prompt_embeddings/{i:08d}.pt" for i in range(len(metadata))]
        new_metadata['video_latent'] = [f"video_latents/{i:08d}.pt" for i in range(len(metadata))]
        
        new_metadata_path = output_dir / 'metadata_encoded.csv'
        new_metadata.to_csv(new_metadata_path, index=False)
        print(f"New metadata saved to: {new_metadata_path}")
        
        print("\nPre-processing completed!")
        print(f"- Prompt embeddings: {prompt_dir}")
        print(f"- Video latents: {video_dir}")
        print(f"- New metadata: {new_metadata_path}")
    else:
        print(f"\nGPU {args.gpu_id} processing completed! Waiting for other GPUs...")
        print(f"- The output of this GPU has been saved to {output_dir}")


if __name__ == '__main__':
    main()

