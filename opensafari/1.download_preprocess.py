import os
import csv
import re
import shutil
import time
import logging
import argparse
import warnings
import requests
import subprocess
from pathlib import Path
import pandas as pd
import uuid
import torch
import concurrent.futures as cf
from tqdm import tqdm

# Required for scene detection, ensure it's installed: pip install scenedetect
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector

from utils.raft import raft
from utils.watermark import watermark

# --- Setup Enhanced Logging ---
# This will log to both a file and the console for comprehensive tracking.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log"),
        logging.StreamHandler()
    ]
)

class Timer:
    """Context manager to time and log a block of code."""
    def __init__(self, stage_name):
        self.stage_name = stage_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logging.info(f"--- Starting stage: {self.stage_name} ---")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        logging.info(f"--- Finished stage: {self.stage_name} in {elapsed:.2f} seconds ---")

class VideoPipeline:
    """
    An encapsulated, robust, and resumable pipeline for video processing.
    """
    def __init__(self, args):
        self.args = args
        self.gpu_ids = [int(gid.strip()) for gid in args.gpu_ids.split(',')]
        
        warnings.filterwarnings("ignore")
        Path(self.args.dataset_dir).mkdir(parents=True, exist_ok=True)
        # Create a central place for error logs
        self.error_log_path = Path("./error_logs")
        self.error_log_path.mkdir(exist_ok=True)


    def run(self):
        """Main execution flow of the entire pipeline."""
        video_csv = self._read_csv(self.args.video_csv)
        if not video_csv:
            return

        self.process_all_videos(video_csv)

        logging.info("--- Pipeline finished successfully! ---")

    def process_all_videos(self, video_data):
        """Processes all videos through all pipeline stages."""
        logging.info(f"========== Processing all videos ==========")

        # Define all paths
        base_path = Path("./processing_data")
        download_path = base_path / self.args.download_dir
        resized_path = base_path / self.args.resized_dir
        scene_path = base_path / self.args.scene_dir
        clip_path = base_path / self.args.clip_dir
        uniform_clip_path = base_path / self.args.uniform_clip_dir
        frame_path = base_path / self.args.frame_dir
        watermark_path = base_path / self.args.watermark_dir
        raft_path = base_path / self.args.raft_dir
        
        # STAGE 1: Download
        self._download_videos(video_data, download_path)

        # STAGE 2: Resize
        self._crop_and_resize_videos(download_path, resized_path)
        if self.args.cleanup_intermediate: shutil.rmtree(download_path, ignore_errors=True)

        # STAGE 3: Detect Scenes
        self._detect_scenes(resized_path, scene_path)

        # STAGE 4: Split into Scene Clips
        self._split_scenes(resized_path, scene_path, clip_path)
        if self.args.cleanup_intermediate: shutil.rmtree(resized_path, ignore_errors=True)
            
        # STAGE 5: Uniform Split
        self._uniform_split(clip_path, uniform_clip_path)
        if self.args.cleanup_intermediate: shutil.rmtree(clip_path, ignore_errors=True)
            
        # STAGE 6: Extract Frames (Optimized with ffmpeg)
        self._extract_frames(uniform_clip_path, frame_path)
        
        # STAGE 7: Watermark Detection
        with Timer(f"Watermark detection to {str(watermark_path)}"):
            watermark(str(frame_path), str(watermark_path), self.args)

        # STAGE 8: RAFT Motion Analysis
        with Timer(f"RAFT motion analysis to {str(raft_path)}"):
            raft(str(frame_path), str(watermark_path), str(raft_path), self.args)
        if self.args.cleanup_intermediate: shutil.rmtree(frame_path, ignore_errors=True)

        # STAGE 9: Finalize - Filter, Move, and create Metadata
        self._finalize_assets(uniform_clip_path, watermark_path, raft_path)
        if self.args.cleanup_intermediate:
            # Clean up the entire processing folder
            shutil.rmtree(base_path, ignore_errors=True)

    def _read_csv(self, file_path):
        """Safely reads a CSV file."""
        try:
            with open(file_path, mode='r', newline='', encoding='utf-8') as f:
                return list(csv.DictReader(f))
        except FileNotFoundError:
            logging.error(f"Metadata CSV not found at {file_path}. Aborting.")
            with open(self.error_log_path / 'file_not_found_errors.log', 'a') as f:
                f.write(f"Attempted to read non-existent file: {file_path}\n")
            return None

    def _get_files_to_process(self, input_dir, output_dir, input_ext, output_ext=None, output_is_dir=False):
        """
        Helper to determine which files need processing for resumability.
        Can now check for output files (default) or output directories.
        """
        if not Path(input_dir).is_dir(): return []
        
        inputs = {p.stem for p in Path(input_dir).glob(f"*.{input_ext}")}
        
        if output_is_dir:
            # New logic: Check for existing directories whose names match the input stems.
            outputs = {p.name for p in Path(output_dir).iterdir() if p.is_dir()}
        else:
            # Original logic: Check for existing files with a specific extension.
            output_ext = output_ext or input_ext
            outputs = {p.stem for p in Path(output_dir).glob(f"*.{output_ext}")}
        
        to_process_stems = sorted(list(inputs - outputs))
        to_process_paths = [Path(input_dir) / f"{stem}.{input_ext}" for stem in to_process_stems]

        if to_process_paths:
            logging.info(f"Found {len(inputs)} inputs, {len(outputs)} already processed. Need to process {len(to_process_paths)} items.")
        else:
            logging.info(f"All {len(inputs)} items already processed for this stage. Skipping.")
        
        return to_process_paths

    # --- STAGE 1: DOWNLOAD ---
    def _download_videos(self, data, output_path):
        with Timer(f"Download to {output_path}"):
            output_path.mkdir(parents=True, exist_ok=True)
            existing = {p.stem for p in output_path.glob("*.mp4")}
            tasks = [row for row in data if row['mp4'].split('/')[-1].split('-')[0].split('.mp4')[0] not in existing]
            if not tasks: return

            for row in tqdm(tasks, desc="Downloading"):
                self._download_single_video(row['mp4'], output_path)

    def _download_single_video(self, url, path):
        
        try:
            # The filename is based on the URL before resolution changes.
            filename = url.split('/')[-1].split('-')[0].split('.mp4')[0] + '.mp4'
            output_file = path / filename

            # Try resolutions from highest to lowest.
            preferred_resolutions = ['-2160p', '-1080p', '-720p', '-480p']
            # Remove any existing resolution specifier to create a base URL.
            base_url = re.sub(r'-\d+p', '', url)

            for res in preferred_resolutions:
                # Construct the URL for the preferred resolution.
                try_url = base_url.replace('.mp4', f'{res}.mp4')
                response = requests.get(try_url, timeout=30)
                if response.status_code == 200:
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    logging.info(f"Downloaded {filename} at {res} resolution.")
                    return # Exit after successful high-res download

            # --- MODIFICATION STARTS HERE ---
            # Fallback to the original URL if no preferred resolutions work.
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(output_file, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded {filename} using original URL. Verifying resolution...")

            # After downloading, check the actual resolution using ffprobe.
            try:
                ffprobe_command = [
                    'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=height', '-of', 'csv=s=x:p=0',
                    str(output_file)
                ]
                result = subprocess.run(ffprobe_command, capture_output=True, text=True, check=True, timeout=60)
                actual_height = int(result.stdout.strip())

                # If the actual height is self.args.height or less, log it as an issue.
                if actual_height < self.args.height:
                    os.remove(output_file)
                    logging.warning(f"No version >{self.args.height}p found for {filename}. Fallback version is {actual_height}p.")
                    with open(self.error_log_path / 'download_errors.log', 'a') as f:
                        f.write(f"No version >{self.args.height}p available for {url} (actual height: {actual_height}p)\n")

            except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
                logging.warning(f"Could not verify resolution for {filename} using ffprobe. Error: {e}")
            # --- MODIFICATION ENDS HERE ---

        except requests.RequestException as e:
            logging.error(f"Failed to download {url}. Error: {e}")
            with open(self.error_log_path / 'download_errors.log', 'a') as f:
                f.write(f"{url}. Error: {e}\n")
    
    # --- GENERIC FFMPEG WORKER ---
    def _run_command(self, task_info):
        """Generic worker for running a command, logs errors."""
        command, log_name, error_log_path = task_info
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return (log_name, True, "Success")
        except subprocess.CalledProcessError as e:
            error_message = f"Error processing {log_name}:\n{e.stderr.strip()}"
            logging.error(error_message)
            with open(error_log_path / 'command_errors.log', 'a') as f:
                f.write(f"{error_message}\n{'-'*20}\n")
            return (log_name, False, error_message)

    # --- STAGE 2: CROP AND RESIZE ---
    def _crop_and_resize_videos(self, input_dir, output_dir):
        with Timer(f"Crop and Resize to {output_dir}"):
            output_dir.mkdir(parents=True, exist_ok=True)
            videos_to_process = self._get_files_to_process(input_dir, output_dir, 'mp4')
            if not videos_to_process:
                return

            tasks = []
            for i, video_path in enumerate(videos_to_process):
                gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
                out_path = output_dir / video_path.name
                tasks.append((video_path, out_path, gpu_id, self.args.height, self.args.width, self.args.quality, self.args.fps, self.error_log_path))
            
            with cf.ProcessPoolExecutor(max_workers=self.args.workers) as executor:
                list(tqdm(executor.map(self._crop_resize_worker, tasks), total=len(tasks), desc="Cropping and Resizing"))

    @staticmethod
    def _crop_resize_worker(task_args):
        video_path, out_path, gpu_id, target_h, target_w, quality, fps, error_log_path = task_args
        crop_params = None

        try:
            detect_command = [
                'ffmpeg', '-i', str(video_path),
                '-vf', "select='between(t,20,30)',cropdetect",
                '-an', '-f', 'null', '-'
            ]
            result = subprocess.run(detect_command, capture_output=True, text=True, timeout=600)
            for line in result.stderr.split('\n'):
                if 'crop=' in line:
                    match = re.search(r'crop=(\d+:\d+:\d+:\d+)', line)
                    if match:
                        crop_params = match.group(1)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logging.warning(f"Crop detection failed for {video_path.name}. Will use standard resize. Error: {e}")
            crop_params = None

        filters = [f'fps={fps}']
        if crop_params:
            filters.append(f"crop={crop_params}")
            logging.info(f"Detected crop={crop_params} for {video_path.name}")
        filters.extend([
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase",
            f"crop={target_w}:{target_h}",
            "format=yuv420p"
        ])
        final_filters = ",".join(filters)

        final_command = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-i', str(video_path),
            '-vf', final_filters,
            '-c:v', 'h264_nvenc', '-gpu', str(gpu_id),
            '-tune', 'hq', '-cq', str(quality),
            '-c:a', 'copy', str(out_path)
        ]

        try:
            subprocess.run(final_command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            error_message = f"Failed to process {video_path.name}. FFmpeg error:\n{e.stderr.decode('utf-8')}"
            logging.error(error_message)
            with open(error_log_path / 'ffmpeg_errors.log', 'a') as f:
                f.write(f"{error_message}\n{'-'*20}\n")


    # --- STAGE 3: SCENE DETECTION ---
    def _detect_scenes(self, input_dir, output_dir):
        with Timer(f"Detect scenes to {output_dir}"):
            output_dir.mkdir(parents=True, exist_ok=True)
            videos_to_process = self._get_files_to_process(input_dir, output_dir, 'mp4', 'txt')
            if not videos_to_process: return

            tasks = [(p, output_dir, self.args.scene_threshold, self.args.video_frames, self.error_log_path) for p in videos_to_process]
            with cf.ProcessPoolExecutor(max_workers=self.args.workers) as executor:
                list(tqdm(executor.map(self._detect_one_scene_worker, tasks), total=len(tasks), desc="Detecting Scenes"))

    @staticmethod
    def _detect_one_scene_worker(task_args):
        video_path, out_dir, threshold, min_frames, error_log_path = task_args
        try:
            video = open_video(str(video_path))
            sm = SceneManager()
            sm.add_detector(AdaptiveDetector(adaptive_threshold=threshold))
            sm.detect_scenes(video)
            scenes = sm.get_scene_list()
            txt_path = out_dir / f"{video_path.stem}.txt"
            
            with open(txt_path, "w") as f:
                if scenes:
                    for s, e in scenes:
                        if e.get_frames() - s.get_frames() > min_frames:
                            f.write(f"{s.get_frames()} {e.get_frames()}\n")
                else:
                    total_frames = video.duration.get_frames()
                    f.write(f"0 {total_frames}\n")
        except Exception as e:
            error_message = f"Scene detection failed for {video_path.name}: {e}"
            logging.error(error_message)
            with open(error_log_path / 'scenedetect_errors.log', 'a') as f:
                f.write(f"{error_message}\n")

    # --- STAGE 4: SPLIT BY SCENE ---
    def _split_scenes(self, input_dir, scene_dir, output_dir):
        with Timer(f"Split scenes to {output_dir}"):
            output_dir.mkdir(parents=True, exist_ok=True)
            scene_files = list(scene_dir.glob("*.txt"))
            if not scene_files: return
            
            tasks = []
            for scene_file in scene_files:
                video_name = scene_file.stem
                if not any(output_dir.glob(f"{video_name}_clip_*.mp4")):
                    video_path = input_dir / f"{video_name}.mp4"
                    if video_path.exists():
                        tasks.append(video_path)
            
            if not tasks:
                logging.info("All scenes already split. Skipping.")
                return

            p_tasks = []
            for i, video_path in enumerate(tasks):
                gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
                p_tasks.append((video_path, scene_dir, output_dir, gpu_id, self.args.quality, self.args.fps, self.error_log_path))
            
            with cf.ProcessPoolExecutor(max_workers=self.args.workers) as executor:
                list(tqdm(executor.map(self._split_one_video_by_scene_worker, p_tasks), total=len(p_tasks), desc="Splitting Scenes"))

    @staticmethod
    def _split_one_video_by_scene_worker(task_args):
        video_path, scene_dir, out_dir, gpu_id, quality, fps, error_log_path = task_args
        scene_file = scene_dir / f"{video_path.stem}.txt"
        if not scene_file.exists(): return

        with open(scene_file, 'r') as f:
            for i, line in enumerate(f, 1):
                try:
                    start_frame, end_frame = map(int, line.strip().split())
                    start_time, end_time = start_frame / fps, end_frame / fps
                    out_path = out_dir / f"{video_path.stem}_clip_{i:03d}_({start_frame}-{end_frame}).mp4"
                    
                    v_filter = f"select='between(n,{start_frame},{end_frame-1})',setpts=PTS-STARTPTS"
                    a_filter = f"aselect='between(t,{start_time},{end_time})',asetpts=PTS-STARTPTS"
                    
                    command = ['ffmpeg', '-y', '-loglevel', 'error',
                               '-i', str(video_path),
                               '-vf', v_filter, '-af', a_filter,
                               '-c:v', 'h264_nvenc', '-gpu', str(gpu_id),
                               '-tune', 'hq', '-cq', str(quality), '-c:a', 'aac', '-b:a', '192k',
                               str(out_path)]
                    subprocess.run(command, check=True, capture_output=True)
                except (ValueError, subprocess.CalledProcessError) as e:
                    error_message = f"Failed splitting scene for {video_path.name} line {i}: {e}"
                    logging.error(error_message)
                    with open(error_log_path / 'ffmpeg_errors.log', 'a') as f:
                        f.write(f"{error_message}\n")

    # --- STAGE 5: UNIFORM SPLIT ---
    def _uniform_split(self, input_dir, output_dir):
        with Timer(f"Uniform split to {output_dir}"):
            output_dir.mkdir(parents=True, exist_ok=True)
            videos_to_process = list(input_dir.glob("*.mp4"))
            if not videos_to_process: return

            if any(output_dir.iterdir()): # TODO: 
                logging.info("Output directory for uniform split is not empty, assuming completion. Skipping.")
                return

            tasks = []
            for i, video_path in enumerate(videos_to_process):
                gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
                tasks.append((video_path, output_dir, self.args.video_frames, gpu_id, self.args.quality, self.args.fps, self.error_log_path))
            
            with cf.ProcessPoolExecutor(max_workers=self.args.workers) as executor:
                list(tqdm(executor.map(self._uniform_split_worker, tasks), total=len(tasks), desc="Uniform Splitting"))

    @staticmethod
    def _uniform_split_worker(task_args):
        video_path, out_dir, video_frames, gpu_id, quality, fps, error_log_path = task_args
        match = re.search(r'\((\d+)-(\d+)\)', video_path.name)
        if not match: return
        start_frame_src, end_frame_src = int(match.group(1)), int(match.group(2))
        total_frames = end_frame_src - start_frame_src

        if total_frames < video_frames: return

        num_clips = total_frames // video_frames
        for i in range(num_clips):
            try:
                start_offset = i * video_frames
                actual_start = start_frame_src + start_offset
                actual_end = actual_start + video_frames
                out_path = out_dir / f"{video_path.stem}_clip_{i+1:04d}_({actual_start}-{actual_end}).mp4"
                
                start_time, end_time = start_offset / fps, (start_offset + video_frames) / fps
                v_filter = f"select='between(n,{start_offset},{start_offset + video_frames - 1})',setpts=PTS-STARTPTS"
                a_filter = f"aselect='between(t,{start_time},{end_time})',asetpts=PTS-STARTPTS"

                command = ['ffmpeg', '-y', '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
                           '-i', str(video_path), '-vf', v_filter, '-af', a_filter,
                           '-c:v', 'h264_nvenc', '-tune', 'hq', '-cq', str(quality),
                           '-c:a', 'aac', '-b:a', '192k', str(out_path)]
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                error_message = f"Uniform split failed for {video_path.name} clip {i}: {e}"
                logging.error(error_message)
                with open(error_log_path / 'ffmpeg_errors.log', 'a') as f:
                    f.write(f"{error_message}\n")

    # --- STAGE 6: EXTRACT FRAMES ---
    def _extract_frames(self, input_dir, output_dir):
        with Timer(f"Extract frames to {output_dir}"):
            output_dir.mkdir(parents=True, exist_ok=True)
            videos_to_process = self._get_files_to_process(input_dir, output_dir, 'mp4', output_is_dir=True)
            if not videos_to_process: return

            tasks = []
            for i, video_path in enumerate(videos_to_process):
                gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
                video_out_dir = output_dir / video_path.stem
                video_out_dir.mkdir(exist_ok=True)
                command = ['ffmpeg', '-y', '-loglevel', 'error',
                           '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
                           '-i', str(video_path),
                           '-q:v', '2',
                           str(video_out_dir / 'frame_%05d.jpg')]
                tasks.append((command, video_path.name, self.error_log_path))

            with cf.ProcessPoolExecutor(max_workers=self.args.workers) as executor:
                list(tqdm(executor.map(self._run_command, tasks), total=len(tasks), desc="Extracting Frames"))
    
    # --- STAGE 9: FINALIZE ASSETS ---
    def _finalize_assets(self, uniform_clip_path, watermark_path, raft_path):
        with Timer(f"Finalizing assets"):
            watermark_file = watermark_path / 'watermark.csv'
            raft_file = raft_path / 'raft.csv'
            meta_file = Path(self.args.meta_file)

            if not watermark_file.exists() or not raft_file.exists():
                logging.warning(f"Skipping finalization: missing raft or watermark CSV.")
                return

            processed_basenames = set()
            if meta_file.exists():
                try:
                    existing_meta_df = pd.read_csv(meta_file)
                    if 'basename' in existing_meta_df.columns:
                        processed_basenames = set(existing_meta_df['basename'])
                        logging.info(f"Found {len(processed_basenames)} previously finalized clips in {meta_file}.")
                except pd.errors.EmptyDataError:
                    logging.warning(f"Metadata file {meta_file} is empty. No prior clips found.")
                except Exception as e:
                    logging.error(f"Could not read metadata file {meta_file} for resume check. Error: {e}")
            
            try:
                w_df = pd.read_csv(watermark_file)
                r_df = pd.read_csv(raft_file)
                merged_df = pd.merge(w_df, r_df, on='video')

                eligible_clips_df = merged_df[
                    (merged_df['watermark_ratio'] < self.args.watermark_threshold) & 
                    (merged_df['average_motion'] > self.args.raft_threshold)
                ]
                logging.info(f"Found {len(eligible_clips_df)} clips meeting thresholds.")
                # Filter out clips that have already been finalized
                videos_to_keep_df = eligible_clips_df[~eligible_clips_df['video'].isin(processed_basenames)]

                if videos_to_keep_df.empty:
                    logging.info("All eligible clips from this dataset have already been finalized. Skipping.")
                    return

                logging.info(f"After filtering, {len(videos_to_keep_df)} new clips will be finalized and moved.")
                
                filename_pattern = re.compile(r'([a-f0-9]+)_clip_(\d+)_\((\d+)-(\d+)\)_clip_(\d+)_\((\d+)-(\d+)\)')

                all_metadata = []
                for _, row in tqdm(videos_to_keep_df.iterrows(), total=len(videos_to_keep_df), desc="Moving Final Clips"):
                    basename = row['video']
                    source_path = uniform_clip_path / f"{basename}.mp4"
                    if not source_path.exists():
                        logging.warning(f"Source clip not found, skipping: {source_path}")
                        continue

                    match = filename_pattern.match(basename)
                    if not match:
                        logging.warning(f"Could not parse basename to extract full metadata, skipping: {basename}")
                        continue

                    original_video, scene_clip_id_str, scene_start_frame_str, \
                    scene_end_frame_str, uniform_clip_id_str, uniform_start_frame_str, \
                    uniform_end_frame_str = match.groups()

                    new_uuid = str(uuid.uuid4())
                    dest_path = Path(self.args.dataset_dir) / f"{new_uuid}.mp4"
                    shutil.copy2(str(source_path), str(dest_path))
                    
                    all_metadata.append({
                        'basename': basename, # Add basename to prevent future duplicates
                        'original_video': original_video,
                        'scene_clip_id': int(scene_clip_id_str),
                        'scene_start_frame': int(scene_start_frame_str),
                        'scene_end_frame': int(scene_end_frame_str),
                        'uniform_clip_id': int(uniform_clip_id_str),
                        'uniform_start_frame': int(uniform_start_frame_str),
                        'uniform_end_frame': int(uniform_end_frame_str),
                        'watermark_ratio': row['watermark_ratio'],
                        'average_motion': row['average_motion'],
                        'uuid': new_uuid
                    })

                if all_metadata:
                    meta_df = pd.DataFrame(all_metadata)
                    meta_file = Path(self.args.meta_file)
                    is_new = not meta_file.exists()
                    meta_df.to_csv(meta_file, mode='a', header=is_new, index=False)
                    logging.info(f"Appended {len(all_metadata)} records to {meta_file}")

            except Exception as e:
                error_message = f"Failed during asset finalization. Error: {e}"
                logging.error(error_message)
                with open(self.error_log_path / 'finalization_errors.log', 'a') as f:
                    f.write(f"{error_message}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Video Processing Pipeline")
    parser.add_argument("--cleanup_intermediate", action='store_true', help="Delete intermediate files after each stage.")
    parser.add_argument("--dataset_dir", type=str, default='./data')
    parser.add_argument("--meta_file",   type=str, default='meta.csv')
    parser.add_argument("--workers",    type=int, default=96)
    parser.add_argument("--gpu_ids",    type=str,default="0,1,2,3,4,5,6,7")
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    parser.add_argument("--video_csv",  type=str, default='videos.csv')
    parser.add_argument("--download_dir",       type=str, default='0_downloads')
    parser.add_argument("--resized_dir",        type=str, default='1_resized')
    parser.add_argument("--scene_dir",          type=str, default='2_scenes')
    parser.add_argument("--clip_dir",           type=str, default='3_scene_clips')
    parser.add_argument("--uniform_clip_dir",   type=str, default='4_uniform_clips')
    parser.add_argument("--frame_dir",          type=str, default='5_frames')
    parser.add_argument("--watermark_dir",      type=str, default='6_watermark_results')
    parser.add_argument("--raft_dir",           type=str, default='7_raft_results')
    parser.add_argument("--video_frames", type=int, default=360)
    parser.add_argument("--fps",          type=int, default=24)
    parser.add_argument("--height",       type=int, default=720)
    parser.add_argument("--width",        type=int, default=1280)
    parser.add_argument("--quality",      type=int, default=23)
    parser.add_argument("--scene_threshold",     type=float, default=2.0, help='')
    parser.add_argument('--watermark_threshold', type=float, default=0.1, help='')
    parser.add_argument('--raft_threshold',      type=float, default=5.0, help='')
    parser.add_argument('--watermark_model',        type=str, default="prithivMLmods/Watermark-Detection-SigLIP2")
    parser.add_argument('--raft_model',             type=str, default='models/raft-things.pth')
    parser.add_argument('--raft_batch_size',        type=int, default=8)
    parser.add_argument('--watermark_batch_size',   type=int, default=32)
    parser.add_argument('--watermark_index',        type=int, default=1)
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')

    args = parser.parse_args()
    
    pipeline = VideoPipeline(args)
    pipeline.run()
