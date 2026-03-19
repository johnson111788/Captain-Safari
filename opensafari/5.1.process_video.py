#!/usr/bin/env python3
"""
Read 15-second videos from the input directory.
Split each video into multiple overlapping 5-second clips and save them to the output directory.

Logic overview:
- Original video duration: 15 seconds
- Split into 11 clips: 0-5s, 1-6s, 2-7s, 3-8s, 4-9s, 5-10s, 6-11s, 7-12s, 8-13s, 9-14s, 10-15s
- Each clip duration: 5 seconds
"""

import argparse
import os
import subprocess
from pathlib import Path
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

# Constants
CLIP_DURATION = 5  # Duration of each clip in seconds
NUM_CLIPS = 11     # Total number of clips (0 to 10)
QUALITY = 23       # Video quality parameter (CRF value, lower is better)
FPS = 24           # Video frame rate

def split_single_video(args):
    """Process the splitting task for a single video."""
    video_path, output_dir = args
    
    # Get the video filename (without extension) as UUID
    uuid = video_path.stem
    
    try:
        # Cut the video at each starting offset
        for start_idx in range(NUM_CLIPS):
            # Calculate frame offset and time
            start_offset = start_idx * FPS  # Starting frame offset
            clip_frames = CLIP_DURATION * FPS  # Number of frames in a clip (5s * 24fps = 120 frames)
            start_time = start_offset / FPS  # Starting time in seconds
            end_time = (start_offset + clip_frames) / FPS  # Ending time in seconds
            
            # Output file path
            output_filename = f"{uuid}_{start_idx}.mp4"
            output_path = output_dir / output_filename
            
            # Build video and audio filters
            if start_idx < NUM_CLIPS - 1:
                v_filter = f"select='between(n,{start_offset},{start_offset + clip_frames})',setpts=PTS-STARTPTS"
            else:
                padding_duration = 1 / FPS
                v_filter = f"select='between(n,{start_offset},{start_offset + clip_frames})',setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={padding_duration}"
            a_filter = f"aselect='between(t,{start_time},{end_time})',asetpts=PTS-STARTPTS"
            
            # Build ffmpeg command
            command = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vf', v_filter,
                '-af', a_filter,
                '-c:v', 'libx264',     # Use CPU encoding
                '-crf', str(QUALITY),  # Quality parameter (CRF for libx264)
                '-preset', 'medium',   # Encoding speed preset
                '-c:a', 'aac',
                '-b:a', '192k',
                str(output_path)
            ]
            
            # Execute command
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        return None  # Successfully processed
        
    except subprocess.CalledProcessError as e:
        return f"Error processing {uuid}: {e}"
    except Exception as e:
        return f"Error processing {uuid}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Split 15-second videos into multiple overlapping 5-second clips")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing original videos")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for video clips")
    parser.add_argument("--num-processes", type=int, default=20, help="Number of parallel processes")
    args = parser.parse_args()

    source_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all video files
    pattern = str(source_dir / "*.mp4")
    videos = sorted(glob(pattern))
    
    print(f"Found {len(videos)} video files")
    print(f"Output directory: {output_dir}")
    print(f"Each video will be split into {NUM_CLIPS} clips (5 seconds each)")
    print()
    
    # Prepare task list
    tasks = [(Path(video_path), output_dir) for video_path in videos]
    
    print(f"Using {args.num_processes} parallel processes")
    print()

    with Pool(processes=args.num_processes) as pool:
        results = list(tqdm(
            pool.imap(split_single_video, tasks),
            total=len(tasks),
            desc="Splitting videos"
        ))
    
    # Print error messages if any
    errors = [r for r in results if r is not None]
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  {error}")
    
    print(f"\nCompleted! Files saved to: {output_dir}")

if __name__ == "__main__":
    main()
