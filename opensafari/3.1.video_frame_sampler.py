#!/usr/bin/env python3
"""
A script to sample frames from videos
Support sampling frames from videos by specified frame rate
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


class VideoFrameSampler:
    """A video frame sampler"""
    
    def __init__(self, video_path: str, output_dir: str = None):
        """
        Initialize the sampler
        
        Args:
            video_path: the path to the video file or the video folder
            output_dir: the output directory, default is video_path/frames
        """
        self.video_path = Path(video_path)
        
        if output_dir is None:
            if self.video_path.is_file():
                # If it is a single file, create a frames folder in the directory of the file
                self.base_output_dir = self.video_path.parent / "frames"
            else:
                # If it is a folder, create a frames folder in the folder
                self.base_output_dir = self.video_path / "frames"
        else:
            self.base_output_dir = Path(output_dir)
        
        # Create the base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # A lock for thread-safe printing
        self.print_lock = Lock()
        
    def get_video_info(self, video_path: str) -> Tuple[int, float, int, int]:
        """
        Get the video information
        
        Args:
            video_path: the path to the video file
            
        Returns:
            (total frames, frame rate, width, height)
        """
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return total_frames, fps, width, height
    
    def sample_frames_by_fps(self, video_path: str, target_fps: float, 
                           max_frames: Optional[int] = None) -> List[str]:
        """
        Sample frames from videos by specified frame rate
        
        Args:
            video_path: the path to the video file
            target_fps: the target frame rate
            max_frames: the maximum number of frames to sample, None means no limit
            
        Returns:
            the list of the paths to the sampled frames
        """
        # Get the video information
        total_frames, original_fps, width, height = self.get_video_info(video_path)
        
        print(f"Processing video: {Path(video_path).name}")
        print(f"Original frame rate: {original_fps:.2f} fps")
        print(f"Total frames: {total_frames}")
        print(f"Resolution: {width}x{height}")
        print(f"Target frame rate: {target_fps} fps")
        
        # Calculate the sampling interval
        if target_fps >= original_fps:
            # If the target frame rate is greater than or equal to the original frame rate, sample all frames
            frame_interval = 1
            print("The target frame rate is greater than or equal to the original frame rate, will sample all frames")
        else:
            # Calculate how many frames to sample per interval
            frame_interval = int(original_fps / target_fps)
            print(f"Sample every {frame_interval} frames")
        
        # Calculate the expected number of sampled frames
        expected_samples = total_frames // frame_interval
        if max_frames is not None and expected_samples > max_frames:
            # If it exceeds the maximum limit, recalculate the interval
            frame_interval = total_frames // max_frames
            expected_samples = max_frames
            print(f"Limit the maximum number of frames to {max_frames}, adjust the sampling interval to {frame_interval}")
        
        print(f"Expected to sample {expected_samples} frames")
        
        # Start sampling
        cap = cv2.VideoCapture(video_path)
        video_name = Path(video_path).stem
        
        # Create a dedicated output folder for the current video
        video_output_dir = self.base_output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_frames = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sample by interval
            if frame_count % frame_interval == 0:
                if max_frames is not None and saved_count >= max_frames:
                    break
                    
                # Save the frame
                frame_filename = f"frame_{saved_count:06d}.png"
                frame_path = video_output_dir / frame_filename
                
                success = cv2.imwrite(str(frame_path), frame)
                if success:
                    saved_frames.append(str(frame_path))
                    saved_count += 1
                    
                    if saved_count % 100 == 0:
                        print(f"Saved {saved_count} frames...")
                        
            frame_count += 1
            
        cap.release()
        
        print(f"Done! Saved {saved_count} frames to {video_output_dir}")
        return saved_frames
    
    def sample_frames_by_interval(self, video_path: str, interval_seconds: float,
                                max_frames: Optional[int] = None) -> List[str]:
        """
        Sample frames from videos by specified time interval
        
        Args:
            video_path: the path to the video file
            interval_seconds: the time interval to sample (seconds)
            max_frames: the maximum number of frames to sample, None means no limit
            
        Returns:
            the list of the paths to the sampled frames
        """
        # Get the video information
        total_frames, fps, width, height = self.get_video_info(video_path)
        duration = total_frames / fps
        
        print(f"Processing video: {Path(video_path).name}")
        print(f"Video duration: {duration:.2f} seconds")
        print(f"Sampling interval: {interval_seconds} seconds")
        
        # Calculate the sampling time points
        sample_times = []
        current_time = 0
        while current_time < duration:
            sample_times.append(current_time)
            current_time += interval_seconds
            if max_frames is not None and len(sample_times) >= max_frames:
                break
        
        print(f"Expected to sample {len(sample_times)} frames")
        
        # Start sampling
        cap = cv2.VideoCapture(video_path)
        video_name = Path(video_path).stem
        
        # Create a dedicated output folder for the current video
        video_output_dir = self.base_output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_frames = []
        
        for i, sample_time in enumerate(sample_times):
            # Jump to the specified time
            cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
            
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save the frame
            frame_filename = f"time_{sample_time:.2f}s_frame_{i:06d}.png"
            frame_path = video_output_dir / frame_filename
            
            success = cv2.imwrite(str(frame_path), frame)
            if success:
                saved_frames.append(str(frame_path))
                
                if (i + 1) % 50 == 0:
                    print(f"Saved {i + 1} frames...")
                    
        cap.release()
        
        print(f"Done! Saved {len(saved_frames)} frames to {video_output_dir}")
        return saved_frames
    
    def _process_single_video(self, video_file: Path, video_idx: int, total_videos: int,
                             target_fps: float = None, interval_seconds: float = None,
                             max_frames: Optional[int] = None) -> Tuple[str, List[str]]:
        """
        Process a single video file (for multi-threading)
        
        Args:
            video_file: the path to the video file
            video_idx: the index of the video
            total_videos: the total number of videos
            target_fps: the target frame rate
            interval_seconds: the time interval to sample (seconds)
            max_frames: the maximum number of frames to sample, None means no limit
            
        Returns:
            (the path to the video, the list of the paths to the sampled frames)
        """
        with self.print_lock:
            print(f"\n--- Processing the {video_idx}/{total_videos}th video ---")
        
        try:
            if target_fps is not None:
                frames = self.sample_frames_by_fps(str(video_file), target_fps, max_frames)
            elif interval_seconds is not None:
                frames = self.sample_frames_by_interval(str(video_file), interval_seconds, max_frames)
            else:
                raise ValueError("Must specify target_fps or interval_seconds")
            
            return (str(video_file), frames)
            
        except Exception as e:
            with self.print_lock:
                print(f"Error processing video {video_file.name}: {e}")
            return (str(video_file), [])
    
    def process_directory(self, target_fps: float = None, interval_seconds: float = None,
                         max_frames: Optional[int] = None, video_extensions: List[str] = None,
                         max_workers: int = 4) -> dict:
        """
        Process all video files in the directory (multi-threading)
        
        Args:
            target_fps: the target frame rate (one of target_fps or interval_seconds)
            interval_seconds: the time interval to sample (one of target_fps or interval_seconds)
            max_frames: the maximum number of frames to sample per video
            video_extensions: the supported video file extensions
            max_workers: the maximum number of worker threads, default is 4
            
        Returns:
            the result dictionary {video_path: [frame_paths]}
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        if not self.video_path.is_dir():
            raise ValueError(f"{self.video_path} is not a valid directory")
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(self.video_path.glob(f"*{ext}"))
        
        print(f"Found {len(video_files)} video files in {self.video_path}")
        print(f"Using {max_workers} threads to process")
        
        if not video_files:
            print("No video files found")
            return {}
        
        results = {}
        
        # Use thread pool to process videos
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_video = {}
            for i, video_file in enumerate(video_files):
                future = executor.submit(
                    self._process_single_video,
                    video_file,
                    i + 1,
                    len(video_files),
                    target_fps,
                    interval_seconds,
                    max_frames
                )
                future_to_video[future] = video_file
            
            # Collect results
            for future in as_completed(future_to_video):
                video_path, frames = future.result()
                results[video_path] = frames
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Video frame sampling tool")
    parser.add_argument("--input_path", default='./data', help="the path to the video file or the video folder")
    parser.add_argument("--output",     default='./data-frames_4fps', help="the output directory")
    parser.add_argument("--fps", type=float, default=4, help="the target frame rate")
    parser.add_argument("--interval", type=float, help="the time interval to sample (seconds)")
    parser.add_argument("--max-frames", type=int, help="the maximum number of frames to sample per video")
    parser.add_argument("--max-workers", type=int, default=32, help="the maximum number of threads to process (default: 4)")
    parser.add_argument("--extensions", nargs="+", 
                       default=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'],
                       help="the supported video file extensions")
    
    args = parser.parse_args()
    
    if args.fps is None and args.interval is None:
        parser.error("Must specify --fps or --interval parameter")
    
    if args.fps is not None and args.interval is not None:
        parser.error("--fps and --interval parameters cannot be used together")
    
    # Create sampler
    sampler = VideoFrameSampler(args.input_path, args.output)
    
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        # Process single video file
        print("Processing single video file...")
        if args.fps is not None:
            frames = sampler.sample_frames_by_fps(args.input_path, args.fps, args.max_frames)
        else:
            frames = sampler.sample_frames_by_interval(args.input_path, args.interval, args.max_frames)
        
        print(f"\nSummary: successfully sampled {len(frames)} frames")
        
    elif input_path.is_dir():
        # Process directory
        print("Batch processing video files...")
        results = sampler.process_directory(
            target_fps=args.fps,
            interval_seconds=args.interval,
            max_frames=args.max_frames,
            video_extensions=args.extensions,
            max_workers=args.max_workers
        )
        
        # Print summary
        total_frames = sum(len(frames) for frames in results.values())
        successful_videos = sum(1 for frames in results.values() if frames)
        
        print(f"\nSummary:")
        print(f"Number of processed videos: {len(results)}")
        print(f"Successfully processed: {successful_videos}")
        print(f"Total number of sampled frames: {total_frames}")
        
    else:
        print(f"Error: {args.input_path} is not a valid file or directory")


if __name__ == "__main__":
    main()
