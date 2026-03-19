#!/usr/bin/env python3
"""
Process camera parameters (extrinsic and intrinsic) from full trajectories (60 frames).
This script performs two main tasks, outputting to the specified directory:
1. Extracts 11 5-second camera clips (and converts extrinsics to local coordinates).
2. Extracts the last frame of each 5-second clip as a query frame.
3. Generates all possible key camera slices for local world memory conditioning.

Logic overview:
- Original data: 60 frames, corresponding to 15 seconds at 4fps.
- Extrinsic shape: (60, 3, 4)
- Intrinsic shape: (60, 3, 3)
"""

import argparse
import os
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

# Constants
FPS = 4                  # 4 frames per second
STEP_SIZE = 4            # Step size: 1 second = 4 frames
CLIP_FRAMES = 20         # Frames per 5-second clip (5s * 4fps)
NUM_CLIPS = 11           # Total 11 overlapping clips (0-10)
MAX_FRAMES = 60          # Maximum frames in the original data
KEY_LENGTH_LIMIT = 5     # Maximum length for key slice in seconds

def convert_to_local_coordinates(extrinsics):
    """
    Convert extrinsic matrices from the global coordinate system to a local 
    coordinate system where the first frame becomes the origin.
    """
    N = extrinsics.shape[0]
    
    R0 = extrinsics[0, :, :3]
    t0 = extrinsics[0, :, 3]    
    
    # Camera center of the first frame
    C0 = -(R0.T @ t0)
    
    R0_c2w = R0.T
    t0_c2w = C0
    
    T0_c2w = np.eye(4)
    T0_c2w[:3, :3] = R0_c2w
    T0_c2w[:3, 3] = t0_c2w
    
    T0_w2c = np.linalg.inv(T0_c2w)
    
    extrinsics_local = np.zeros_like(extrinsics)
    
    for i in range(N):
        Ri = extrinsics[i, :, :3]
        ti = extrinsics[i, :, 3]
        
        Ti_w2c = np.eye(4)
        Ti_w2c[:3, :3] = Ri
        Ti_w2c[:3, 3] = ti
        
        Ti_c2w = np.linalg.inv(Ti_w2c)
        Ti_local_c2w = T0_w2c @ Ti_c2w
        Ti_local_w2c = np.linalg.inv(Ti_local_c2w)
        
        extrinsics_local[i] = Ti_local_w2c[:3, :]
    
    return extrinsics_local

def generate_all_possible_keys():
    """
    Generate all possible (key_start, key_end) frame combinations for local world memory.
    local world memory logic (in seconds):
    - key_start: up to 4 seconds back from clip start, step size 1 second.
    - key_length: 1-5 seconds, step size 1 second.
    """
    possible_keys = set()
    for clip_idx in range(NUM_CLIPS):
        clip_start = clip_idx * STEP_SIZE
        clip_end = clip_start + CLIP_FRAMES
        
        min_key_start = max(0, clip_start - 4 * STEP_SIZE)
        max_key_start = clip_start
        
        for key_start in range(min_key_start, max_key_start + 1, STEP_SIZE):
            for length_sec in range(1, KEY_LENGTH_LIMIT + 1):
                key_length = length_sec * STEP_SIZE
                key_end = key_start + key_length
                
                if key_end > clip_start and key_end <= min(clip_end, MAX_FRAMES):
                    possible_keys.add((key_start, key_end))
    
    return sorted(list(possible_keys))

def process_single_file(args):
    """Process a single camera file to generate clips, queries, and local keys."""
    file_path, output_dir = args
    filename = file_path.name
    
    if '_extrinsic.npy' in filename:
        uuid = filename.replace('_extrinsic.npy', '')
        cam_type = 'extrinsic'
    elif '_intrinsic.npy' in filename:
        uuid = filename.replace('_intrinsic.npy', '')
        cam_type = 'intrinsic'
    else:
        return f"Unknown file type: {filename}"
    
    try:
        data = np.load(file_path)
        # Squeeze out extra dimensions if necessary
        data = data.squeeze(1) if data.shape[1] == 1 else data
        
        if data.shape[0] < MAX_FRAMES:
            return f"Warning: {uuid} {cam_type} frame count is {data.shape[0]} < {MAX_FRAMES}, skipping."
        
        # 1. Generate 5-second clips and individual query frames
        for clip_idx in range(NUM_CLIPS):
            start_frame = clip_idx * STEP_SIZE
            end_frame = start_frame + CLIP_FRAMES
            
            # Slice clip data
            clip_data = data[start_frame:end_frame]
            
            # Only apply local coordinate transformation to extrinsics
            if cam_type == 'extrinsic':
                clip_data = convert_to_local_coordinates(clip_data)
            
            clip_path = output_dir / f"{uuid}_clip_{start_frame}_{end_frame}_{cam_type}.npy"
            np.save(clip_path, clip_data)
            
            # Extract query frame (the last frame of the current clip)
            last_frame_idx = end_frame - 1
            last_frame = data[last_frame_idx]
            
            query_path = output_dir / f"{uuid}_query_{end_frame}_{cam_type}.npy"
            np.save(query_path, last_frame)
            
        # 2. Generate local keys for local world memory
        possible_keys = generate_all_possible_keys()
        for key_start, key_end in possible_keys:
            key_path = output_dir / f"{uuid}_key_{key_start}_{key_end}_{cam_type}.npy"
            
            # Skip if the file already exists to save time
            if not key_path.exists():
                sliced_data = data[key_start:key_end]
                np.save(key_path, sliced_data)
                
        return None
        
    except Exception as e:
        return f"Error processing {uuid} {cam_type}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Process camera data to generate clips, queries, and local keys")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing camera parameter files (.npy)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--num-processes", type=int, default=30, help="Number of parallel processes")
    args = parser.parse_args()

    source_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(source_dir / "*.npy")
    files = sorted(glob(pattern))
    
    print("=" * 80)
    print("Camera Data Processing (Clips, Queries, Keys)")
    print("=" * 80)
    print(f"Found {len(files)} camera files")
    print(f"Output directory: {output_dir}")
    print(f"Using {args.num_processes} processes")
    print()
    
    tasks = [(Path(f), output_dir) for f in files]
    
    with Pool(processes=args.num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, tasks),
            total=len(tasks),
            desc="Processing camera files"
        ))
    
    errors = [r for r in results if r is not None]
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  {error}")
    
    print(f"\nCompleted! Files saved to: {output_dir}")

if __name__ == "__main__":
    main()
