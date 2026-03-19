#!/usr/bin/env python3
"""
Generate metadata.csv for training.

Features:
- Flexible key sampling: key length and start position are randomized (instead of discrete options)
- Adds extrinsic_clip and intrinsic_clip for the intermediate 5s clip
- Adds corresponding video path
- Mixes short and long captions randomly

Sampling Logic (in seconds/4 frames):
- key_start_time = Rand(Max(0, clip_starting_time - 5), clip_starting_time)
- key_end_time = Min(key_start_time + Rand(1, 5), clip_end_time)
- Constraint: key_end_time >= clip_starting_time (resample if not met)

Sampling scheme:
- Completely random: randomly select a clip (ensure no video duplication), randomly select a valid key
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from tqdm import tqdm

# Constants
STEP_SIZE = 4  # Step size: 1 second = 4 frames
CLIP_FRAMES = 20  # Frames per 5-second clip
NUM_CLIPS = 11  # Total 11 clips of 5 seconds (0-10)

def get_clip_info(clip_idx):
    """Get the start and end frame of a clip"""
    start_frame = clip_idx * STEP_SIZE
    end_frame = start_frame + CLIP_FRAMES
    return start_frame, end_frame

def sample_flexible_key(clip_start_frame, clip_end_frame, max_attempts=50):
    """
    Flexibly sample the start and end frames for a key context.
    
    Logic:
    - key_start_time = Rand(Max(0, clip_starting_time - 5), clip_starting_time)
    - key_end_time = Min(key_start_time + Rand(1, 5), clip_end_time)
    - Requires key_end_time >= clip_starting_time
    - Unit: every 4 frames (1 second)
    
    Returns:
        (key_start_frame, key_end_frame) or None (if conditions are not met)
    """
    # Convert frames to seconds (time)
    clip_start_time = clip_start_frame // STEP_SIZE  # in seconds
    clip_end_time = clip_end_frame // STEP_SIZE  # in seconds
    
    for _ in range(max_attempts):
        # 1. Sample key start time (seconds)
        key_start_time_min = max(0, clip_start_time - 5)
        key_start_time_max = clip_start_time
        
        if key_start_time_min >= key_start_time_max:
            # Boundary case: clip is at the very beginning, key_start can only be 0
            key_start_time = 0
        else:
            key_start_time = random.randint(key_start_time_min, key_start_time_max - 1)
        
        # 2. Sample key length (1-5 seconds)
        key_length = random.randint(1, 5)
        
        # 3. Calculate key end time
        key_end_time = min(key_start_time + key_length, clip_end_time)
        
        # 4. Check constraint: key_end_time >= clip_starting_time
        if key_end_time >= clip_start_time:
            # Convert time back to frames
            key_start_frame = key_start_time * STEP_SIZE
            key_end_frame = key_end_time * STEP_SIZE
            return (key_start_frame, key_end_frame)
    
    # Return None if multiple attempts fail
    return None

def generate_samples_for_video(uuid, samples_per_video=11):
    """Generate samples for a single video (ensure unique clips, use flexible key sampling)"""
    samples = []
    used_clip_indices = set()  # Track used clip_idx to prevent duplication per video
    max_attempts = samples_per_video * 10  # Maximum attempts to prevent infinite loop
    
    # Scheme 1: Completely random sampling
    attempts = 0
    while len(samples) < samples_per_video and attempts < max_attempts:
        attempts += 1
        
        # Randomly select a clip
        clip_idx = random.randint(0, NUM_CLIPS - 1)
        
        # Check if clip_idx has already been used
        if clip_idx in used_clip_indices:
            continue
        
        clip_start_frame, clip_end_frame = get_clip_info(clip_idx)
        
        # Get key using flexible sampling
        key_result = sample_flexible_key(clip_start_frame, clip_end_frame)
        
        if key_result is None:
            # Skip this clip if valid key cannot be sampled
            continue
        
        key_start_frame, key_end_frame = key_result
        
        # Mark this clip_idx as used
        used_clip_indices.add(clip_idx)
        
        # Build sample
        sample = {
            'uuid': uuid,
            'clip_idx': clip_idx,
            'clip_start': clip_start_frame,
            'clip_end': clip_end_frame,
            'key_start': key_start_frame,
            'key_end': key_end_frame
        }
        samples.append(sample)
    
    return samples

def build_csv_entry(sample, video_dir, memory_dir, camera_dir):
    """Build a CSV entry from a sample"""
    uuid = sample['uuid']
    clip_idx = sample['clip_idx']
    clip_start = sample['clip_start']
    clip_end = sample['clip_end']
    key_start = sample['key_start']
    key_end = sample['key_end']
    
    # Construct file paths (relative paths for CSV)
    # Get the directory names without the path
    mem_dir_name = memory_dir.name
    cam_dir_name = camera_dir.name
    vid_dir_name = video_dir.name
    
    # memory_target: {uuid}_{end_frame}.npy
    memory_target = f"{mem_dir_name}/{uuid}_{clip_end}.npy"
    
    # memory_key: {uuid}_{start}_{end}.npy
    memory_key = f"{mem_dir_name}/{uuid}_{key_start}_{key_end}.npy"
    
    # extrinsic_query, intrinsic_query: {uuid}_query_{end_frame}_{cam_type}.npy
    extrinsic_query = f"{cam_dir_name}/{uuid}_query_{clip_end}_extrinsic.npy"
    intrinsic_query = f"{cam_dir_name}/{uuid}_query_{clip_end}_intrinsic.npy"
    
    # extrinsic_key, intrinsic_key: {uuid}_key_{start}_{end}_{cam_type}.npy
    extrinsic_key = f"{cam_dir_name}/{uuid}_key_{key_start}_{key_end}_extrinsic.npy"
    intrinsic_key = f"{cam_dir_name}/{uuid}_key_{key_start}_{key_end}_intrinsic.npy"
    
    # === Added: camera data for the intermediate 5s clip ===
    # extrinsic_clip, intrinsic_clip: {uuid}_clip_{start}_{end}_{cam_type}.npy
    extrinsic_clip = f"{cam_dir_name}/{uuid}_clip_{clip_start}_{clip_end}_extrinsic.npy"
    intrinsic_clip = f"{cam_dir_name}/{uuid}_clip_{clip_start}_{clip_end}_intrinsic.npy"
    
    # === Added: corresponding video ===
    # video: {uuid}_{clip_idx}.mp4
    video = f"{vid_dir_name}/{uuid}_{clip_idx}.mp4"
    
    # Check if files exist (using absolute paths based on original dirs)
    files_to_check = {
        'memory_target': memory_dir / f"{uuid}_{clip_end}.npy",
        'memory_key': memory_dir / f"{uuid}_{key_start}_{key_end}.npy",
        'extrinsic_query': camera_dir / f"{uuid}_query_{clip_end}_extrinsic.npy",
        'intrinsic_query': camera_dir / f"{uuid}_query_{clip_end}_intrinsic.npy",
        'extrinsic_key': camera_dir / f"{uuid}_key_{key_start}_{key_end}_extrinsic.npy",
        'intrinsic_key': camera_dir / f"{uuid}_key_{key_start}_{key_end}_intrinsic.npy",
        'extrinsic_clip': camera_dir / f"{uuid}_clip_{clip_start}_{clip_end}_extrinsic.npy",
        'intrinsic_clip': camera_dir / f"{uuid}_clip_{clip_start}_{clip_end}_intrinsic.npy",
        'videos': video_dir / f"{uuid}_{clip_idx}.mp4"
    }
    
    missing_files = []
    for file_type, file_path in files_to_check.items():
        if not file_path.exists():
            missing_files.append(f"{file_type}: {file_path}")
    
    if missing_files:
        print(f"Warning: The following files are missing, skipping sample:")
        for missing in missing_files:
            print(f"  - {missing}")
        
    return {
        'memory_target': memory_target,
        'memory': memory_key,
        'extrinsic_query': extrinsic_query,
        'intrinsic_query': intrinsic_query,
        'extrinsic_key': extrinsic_key,
        'intrinsic_key': intrinsic_key,
        'extrinsic_clip': extrinsic_clip,
        'intrinsic_clip': intrinsic_clip,
        'videos': video,
        'prompt': "" # Will be filled later by mix_prompts
    }

def extract_video_filename(video_path):
    """
    Extract filename from video path
    Example: videos/003349f4-92cf-4fd1-bb22-774d163e8d8c_2.mp4 -> 003349f4-92cf-4fd1-bb22-774d163e8d8c_2.mp4
    """
    return os.path.basename(video_path)

def mix_prompts(df, caption_dict_short, caption_dict_long, mix_ratio=0.5):
    """
    Mix long and short captions into the metadata
    """
    print(f"Mixing prompts... Long captions ratio {mix_ratio*100:.1f}%, Short captions ratio {(1-mix_ratio)*100:.1f}%...")
    
    # Extract video filename for matching
    df['video_filename'] = df['videos'].apply(extract_video_filename)
    
    replaced_long = 0
    replaced_short = 0
    missing = 0
    
    for idx, row in df.iterrows():
        video_filename = row['video_filename']
        
        # Randomly decide whether to use long or short caption
        use_long = random.random() < mix_ratio
        
        if use_long and video_filename in caption_dict_long:
            df.at[idx, 'prompt'] = caption_dict_long[video_filename]
            replaced_long += 1
        elif not use_long and video_filename in caption_dict_short:
            df.at[idx, 'prompt'] = caption_dict_short[video_filename]
            replaced_short += 1
        else:
            # Fallback strategy
            if video_filename in caption_dict_short:
                df.at[idx, 'prompt'] = caption_dict_short[video_filename]
                replaced_short += 1
            elif video_filename in caption_dict_long:
                df.at[idx, 'prompt'] = caption_dict_long[video_filename]
                replaced_long += 1
            else:
                missing += 1
                
    df = df.drop(columns=['video_filename'])
    
    print(f"  - Using Long Caption: {replaced_long} records")
    print(f"  - Using Short Caption: {replaced_short} records")
    if missing > 0:
        print(f"  - Warning: {missing} records missing any caption")
        
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate metadata.csv for training")
    parser.add_argument("--output-csv", type=str, default="metadata.csv", help="Output metadata CSV filename")
    parser.add_argument("--camera-dir", type=str, default="./data-frames_4fps-camera-fixed-local-interpolated-slices", help="Camera directory")
    parser.add_argument("--video-dir", type=str, default="./data-slices", help="Video directory")
    parser.add_argument("--memory-dir", type=str, default="./data-streamvggt-slices", help="Memory directory")

    parser.add_argument("--short-captions", type=str, default="short_captions.csv", help="Short captions CSV filename")
    parser.add_argument("--long-captions", type=str, default="long_captions.csv", help="Long captions CSV filename")
    parser.add_argument("--mix-ratio", type=float, default=0.5, help="Ratio of long captions to use (0.0-1.0), e.g., 0.5 for half long, half short")
    parser.add_argument("--samples-per-video", type=int, default=11, help="Number of samples per 15s video")
    parser.add_argument("--key-length-limit", type=int, default=5, help="Maximum key length limit (seconds)")
    args = parser.parse_args()

    # Validate ratio
    if not 0.0 <= args.mix_ratio <= 1.0:
        parser.error("Mix ratio must be between 0.0 and 1.0")

    # Setup paths
    camera_dir = Path(args.camera_dir)
    video_dir = Path(args.video_dir)
    memory_dir = Path(args.memory_dir)
    output_csv = Path(args.output_csv)
    short_captions_csv = Path(args.short_captions)
    long_captions_csv = Path(args.long_captions)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # === Load captions.csv ===
    print(f"Loading captions...")
    
    # Load short captions
    if not short_captions_csv.exists():
        print(f"Warning: Short caption file {short_captions_csv} not found")
        caption_dict_short = {}
    else:
        caption_df_short = pd.read_csv(short_captions_csv)
        caption_dict_short = dict(zip(caption_df_short['video'], caption_df_short['short_caption']))
        print(f"Loaded {len(caption_dict_short)} short captions")
        
    # Load long captions
    if not long_captions_csv.exists():
        print(f"Warning: Long caption file {long_captions_csv} not found")
        caption_dict_long = {}
    else:
        caption_df_long = pd.read_csv(long_captions_csv)
        col_name = 'long_caption' if 'long_caption' in caption_df_long.columns else 'short_caption'
        caption_dict_long = dict(zip(caption_df_long['video'], caption_df_long[col_name]))
        print(f"Loaded {len(caption_dict_long)} long captions")
    print()
    
    # Get all UUIDs (from query files in the camera directory)
    pattern = str(camera_dir / "*_query_20_extrinsic.npy")  # Use query files of clip 0 to get all UUIDs
    query_files = glob(pattern)
    
    uuids = []
    for f in query_files:
        filename = os.path.basename(f)
        uuid = filename.replace('_query_20_extrinsic.npy', '')
        uuids.append(uuid)
    
    uuids = sorted(uuids)
    
    print("=" * 80)
    print("Generating Training CSV (includes clip camera, video, and prompt)")
    print("=" * 80)
    print(f"Output CSV: {output_csv}")
    print(f"Found {len(uuids)} videos")
    print(f"\nSampling parameters:")
    print(f"  Samples per video: {args.samples_per_video}")
    print(f"  Key length limit: {args.key_length_limit} seconds")
    print()
    
    # Generate all samples
    all_samples = []
    for uuid in tqdm(uuids, desc="Generating samples"):
        samples = generate_samples_for_video(uuid, samples_per_video=args.samples_per_video)
        all_samples.extend(samples)
    
    print(f"\nGenerated a total of {len(all_samples)} samples")
    
    # Build CSV entries
    csv_entries = []
    for sample in tqdm(all_samples, desc="Building CSV entries"):
        entry = build_csv_entry(sample, video_dir, memory_dir, camera_dir)
        csv_entries.append(entry)
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_entries)
    
    # === Mix Long / Short Captions ===
    df = mix_prompts(df, caption_dict_short, caption_dict_long, mix_ratio=args.mix_ratio)
    
    df.to_csv(output_csv, index=False)
    
    print(f"\nComplete! CSV file saved to: {output_csv}")
    print(f"Total rows: {len(df)}")
    
    # Print column info
    print(f"\nCSV Columns:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Print some statistics
    print("\nStatistics:")
    
    # Analyze the distribution of key lengths
    key_lengths = []
    for sample in all_samples:
        key_length = (sample['key_end'] - sample['key_start']) // STEP_SIZE
        key_lengths.append(key_length)
    
    from collections import Counter
    length_counts = Counter(key_lengths)
    print("\nKey Length Distribution (seconds):")
    for length in sorted(length_counts.keys()):
        print(f"  {length}s: {length_counts[length]} samples")
    
    # Analyze the distribution of clip positions
    clip_counts = Counter([s['clip_idx'] for s in all_samples])
    print("\nClip Position Distribution:")
    for clip_idx in sorted(clip_counts.keys()):
        print(f"  Clip {clip_idx}: {clip_counts[clip_idx]} samples")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
