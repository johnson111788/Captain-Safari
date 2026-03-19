#!/usr/bin/env python3
"""
Process StreamVGGT memory features (aggregated_tokens) from full trajectories.
This script extracts required slices for local world memory conditioning:
1. Extracts the last frame of each 5-second clip as a query target frame.
2. Generates all possible key memory slices.

Logic overview:
- Original data: 60 frames, corresponding to 15 seconds at 4fps.
"""

import argparse
import os
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

# Constants
STEP_SIZE = 4            # Step size: 1 second = 4 frames
CLIP_FRAMES = 20         # Frames per 5-second clip (5s * 4fps)
NUM_CLIPS = 11           # Total 11 overlapping clips (0-10)
MAX_FRAMES = 60          # Maximum frames in the original data
KEY_LENGTH_LIMIT = 5     # Maximum length for key slice in seconds

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
    """Process a single memory file to extract query frames and local keys."""
    file_path, output_dir = args
    filename = os.path.basename(file_path)
    uuid = filename.replace("_aggregated_tokens.npy", "")
    
    try:
        data = np.load(file_path)
        
        if data.shape[0] < MAX_FRAMES:
            return f"Warning: {uuid} frame count is {data.shape[0]} < {MAX_FRAMES}, skipping."
        
        # 1. Extract target frames (queries)
        for clip_idx in range(NUM_CLIPS):
            start_frame = clip_idx * STEP_SIZE
            end_frame = start_frame + CLIP_FRAMES
            last_frame_idx = end_frame - 1
            
            target_frame = data[last_frame_idx]
            query_path = output_dir / f"{uuid}_{end_frame}.npy"
            
            np.save(query_path, target_frame)
            
        # 2. Generate local keys for local world memory
        possible_keys = generate_all_possible_keys()
        for key_start, key_end in possible_keys:
            key_path = output_dir / f"{uuid}_{key_start}_{key_end}.npy"
            
            # Skip if the file already exists to save time
            if not key_path.exists():
                sliced_data = data[key_start:key_end]
                np.save(key_path, sliced_data)
                
        return None
        
    except Exception as e:
        return f"Error processing {uuid}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Process memory tokens to generate queries and local keys")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing memory tokens (*_aggregated_tokens.npy)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for processed slices")
    parser.add_argument("--num-processes", type=int, default=20, help="Number of parallel processes")
    args = parser.parse_args()

    source_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(source_dir / "*_aggregated_tokens.npy")
    files = sorted(glob(pattern))
    
    print("=" * 80)
    print("Memory Data Processing (Queries, Keys)")
    print("=" * 80)
    print(f"Found {len(files)} aggregated_tokens files")
    print(f"Output directory: {output_dir}")
    print(f"Using {args.num_processes} processes")
    print()
    
    tasks = [(Path(f), output_dir) for f in files]
    
    with Pool(processes=args.num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, tasks),
            total=len(tasks),
            desc="Processing memory files"
        ))
    
    errors = [r for r in results if r is not None]
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  {error}")
    
    print(f"\nCompleted! Files saved to: {output_dir}")

if __name__ == "__main__":
    main()
