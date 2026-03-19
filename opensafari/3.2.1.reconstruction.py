import tqdm
import os
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import time
import random
import fcntl
import h5py
import subprocess

from pathlib import Path
import numpy as np

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d


def safe_hloc_operation(func, *args, max_retries=5, **kwargs):
    """Safely execute hloc operation, with retry mechanism"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (BlockingIOError, OSError, PermissionError) as e:
            err_msg = str(e).lower()
            
            # Handle h5py file lock problem
            if ("file is already open for write" in err_msg or "h5clear" in err_msg) and attempt < max_retries - 1:
                print(f"Detected HDF5 file lock残留 (attempt {attempt + 1}/{max_retries})")
                
                # Try to find h5 file path from kwargs and args
                h5_files = []
                for k, v in kwargs.items():
                    if isinstance(v, (str, Path)) and str(v).endswith('.h5'):
                        h5_files.append(Path(v))
                for arg in args:
                     if isinstance(arg, (str, Path)) and str(arg).endswith('.h5'):
                        h5_files.append(Path(arg))

                if h5_files:
                    for h5_file in h5_files:
                        if h5_file.exists():
                            print(f"Try to fix file lock: {h5_file}")
                            try:
                                # Use h5clear -s to clear status flag
                                subprocess.run(["h5clear", "-s", str(h5_file)], check=False, capture_output=True)
                            except Exception as clear_err:
                                print(f"Running h5clear failed: {clear_err}")
                
                # Wait for a while
                time.sleep(random.uniform(1, 3))
                continue

            if any(msg in err_msg for msg in ["resource temporarily unavailable", "unable to lock file", "errno = 11"]) and attempt < max_retries - 1:
                # Randomly wait to avoid simultaneous retries
                wait_time = random.uniform(2, 8)
                print(f"File lock conflict, waiting {wait_time:.1f}s before retrying (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                raise e
        except Exception as e:
            # Other exceptions directly raise
            raise e
    

def process_uuid_folder_on_gpu(args):
    """Function to process a single UUID folder on a specified GPU"""
    uuid_folder_path, uuid_name, gpu_id, skip_geometric_verification, outputs_base_dir = args
    
    # Set the GPU device used by the current process
    process_id = os.getpid()
    if torch.cuda.is_available():
        gpu_device = gpu_id % torch.cuda.device_count()
        torch.cuda.set_device(gpu_device)
        # Ensure the current device is set correctly
        current_device = torch.cuda.current_device()
        print(f"[GPU {current_device}|PID {process_id}] CUDA device is set to GPU {gpu_device}")
    else:
        gpu_device = 0
        print(f"[CPU|PID {process_id}] CUDA is not available, using CPU")
    
    print(f"[GPU {gpu_device}|PID {process_id}] Processing folder: {uuid_name}")
    
    # Set the path
    images = Path(uuid_folder_path)
    outputs = Path(outputs_base_dir) / uuid_name
    
    # Create output directory
    outputs.mkdir(parents=True, exist_ok=True)
    
    sfm_pairs = outputs / "pairs-sfm.txt"
    sfm_dir = outputs / "sfm"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"
    
    # Ensure the output directory exists and is independent
    sfm_dir.mkdir(parents=True, exist_ok=True)

    feature_conf = extract_features.confs["aliked-n16"]
    matcher_conf = match_features.confs["aliked+lightglue"]

    # Get all image files in the folder
    image_files = [f for f in images.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    if not image_files:
        print(f"[GPU {gpu_device}|PID {process_id}] Warning: No image files found in {uuid_name} folder")
        return False, uuid_name, f"No image files found in {uuid_name} folder"
    
    # Generate relative path image list
    references = [p.relative_to(images).as_posix() for p in image_files]
    print(f"[GPU {gpu_device}|PID {process_id}] Found {len(references)} images")

    try:
        start_time = time.time()
        
        # Feature extraction
        print(f"[GPU {gpu_device}|PID {process_id}] Starting feature extraction: {uuid_name}")
        safe_hloc_operation(
            extract_features.main,
            feature_conf, images, image_list=references, feature_path=features
        )
        
        # Generate image pairs
        print(f"[GPU {gpu_device}|PID {process_id}] Generating image pairs: {uuid_name}")
        safe_hloc_operation(
            pairs_from_exhaustive.main, sfm_pairs, image_list=references
        )
        
        # Feature matching
        print(f"[GPU {gpu_device}|PID {process_id}] Starting feature matching: {uuid_name}")
        safe_hloc_operation(
            match_features.main, matcher_conf, sfm_pairs, features=features, matches=matches
        )

        # 3D reconstruction
        if skip_geometric_verification:
            print(f"[GPU {gpu_device}|PID {process_id}] Starting 3D reconstruction (skip geometric verification): {uuid_name}")
        else:
            print(f"[GPU {gpu_device}|PID {process_id}] Starting 3D reconstruction (include geometric verification): {uuid_name}")
            
        model = safe_hloc_operation(
            reconstruction.main,
            sfm_dir, images, sfm_pairs, features, matches,
            image_list=references,
            skip_geometric_verification=skip_geometric_verification,
            # min_reg_images=60
        )
        
        # Generate 3D visualization
        # print(f"[GPU {gpu_device}|PID {process_id}] Generate 3D visualization: {uuid_name}")
        # fig = viz_3d.init_figure()
        # viz_3d.plot_reconstruction(
        #     fig, model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
        # )
        
        # Save HTML file
        # html_path = f"html/{uuid_name}.html"
        # fig.write_html(html_path)
        
        elapsed = time.time() - start_time
        print(f"[GPU {gpu_device}|PID {process_id}] Completed processing {uuid_name}, time: {elapsed:.2f} seconds")
        
        return True, uuid_name, f"Successfully processed, time: {elapsed:.2f} seconds"
        
    except Exception as e:
        error_msg = f"Error processing {uuid_name}: {str(e)}"
        print(f"[GPU {gpu_device}|PID {process_id}] {error_msg}")
        # Print detailed stack information, for debugging
        import traceback
        traceback.print_exc()
        return False, uuid_name, error_msg


def process_multiple_folders_on_gpu(folder_batch, gpu_id, skip_geometric_verification=True, outputs_base_dir="outputs.train"):
    """Function to process multiple folders on a single GPU"""
    results = []
    for uuid_folder_path, uuid_name in folder_batch:
        args = (uuid_folder_path, uuid_name, gpu_id, skip_geometric_verification, outputs_base_dir)
        result = process_uuid_folder_on_gpu(args)
        results.append(result)
        
        # Release GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def distribute_folders_to_gpus(uuid_folders, num_gpus=8, processes_per_gpu=4):
    """Distribute folders to multiple GPUs and processes"""
    total_processes = num_gpus * processes_per_gpu
    folder_chunks = [[] for _ in range(total_processes)]
    
    # Distribute folders evenly to all processes
    for i, folder in enumerate(uuid_folders):
        chunk_idx = i % total_processes
        folder_chunks[chunk_idx].append((folder, folder.name))
    
    # Create GPU assignment information
    gpu_assignments = []
    for gpu_id in range(num_gpus):
        for proc_id in range(processes_per_gpu):
            chunk_idx = gpu_id * processes_per_gpu + proc_id
            if folder_chunks[chunk_idx]:  # Only process non-empty chunks
                gpu_assignments.append((folder_chunks[chunk_idx], gpu_id))
    
    return gpu_assignments

# python demo.py >> demo.log 2>&1
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HLOC video reconstruction - support batch and incremental processing"
    )
    parser.add_argument(
        "--data-base-path",
        type=Path,
        default=Path("./data-frames_4fps"),
        help="Original data directory"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("./data-frames_4fps-hloc"),
        help="HLOC output directory"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs"
    )
    parser.add_argument(
        "--processes-per-gpu",
        type=int,
        default=4,
        help="Number of processes per GPU"
    )
    parser.add_argument(
        "--skip-geometric-verification",
        action="store_true",
        help="Skip geometric verification to speed up"
    )
    parser.add_argument(
        "--target-cases",
        type=str,
        default=None,
        help="Comma-separated list of target case IDs (if specified, only process these cases)"
    )
    
    args = parser.parse_args()
    
    # Configure parameters
    NUM_GPUS = args.num_gpus
    PROCESSES_PER_GPU = args.processes_per_gpu
    SKIP_GEOMETRIC_VERIFICATION = args.skip_geometric_verification
    
    # Specify data directory
    data_base_path = args.data_base_path
    
    # Get all UUID folders
    if args.target_cases:
        # Only process specified cases
        target_case_ids = set(args.target_cases.split(','))
        uuid_folders = [data_base_path / case_id for case_id in target_case_ids if (data_base_path / case_id).is_dir()]
        print(f"Targeting to process {len(target_case_ids)} specified cases, found {len(uuid_folders)} valid folders")
    else:
        # Process all folders
        uuid_folders = [f for f in data_base_path.iterdir() if f.is_dir()]
        print(f"Found {len(uuid_folders)} UUID folders")
    
    print(f"Found {len(uuid_folders)} UUID folders")
    print(f"Using {NUM_GPUS} GPUs, each running {PROCESSES_PER_GPU} processes")
    print(f"Total parallelism: {NUM_GPUS * PROCESSES_PER_GPU}")
    print(f"Geometric verification setting: {'skip' if SKIP_GEOMETRIC_VERIFICATION else 'enable'}")
    if SKIP_GEOMETRIC_VERIFICATION:
        print("⚠️  Skipping geometric verification can significantly speed up, but may affect 3D reconstruction quality")
    
    # Assign tasks to GPUs
    gpu_assignments = distribute_folders_to_gpus(uuid_folders, NUM_GPUS, PROCESSES_PER_GPU)
    print(f"Created {len(gpu_assignments)} work tasks")
    
    # Use multiprocessing pool to process
    total_start_time = time.time()
    all_results = []
    
    # If total processes is 1, don't use multiprocessing pool, run directly for debugging
    total_workers = len(gpu_assignments)
    if total_workers == 1 and NUM_GPUS * PROCESSES_PER_GPU == 1:
        print("Running in single process mode (for debugging)...")
        for folder_batch, gpu_id in gpu_assignments:
            results = process_multiple_folders_on_gpu(folder_batch, gpu_id, SKIP_GEOMETRIC_VERIFICATION, str(args.outputs_dir))
            all_results.extend(results)
    else:
        # Force set multiprocessing start method to spawn
        # PyTorch and pycolmap often deadlock or crash in fork mode
        try:
            mp.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to: spawn")
        except RuntimeError:
            pass

        with ProcessPoolExecutor(max_workers=len(gpu_assignments)) as executor:
            # Submit all tasks
            future_to_gpu = {
                executor.submit(process_multiple_folders_on_gpu, folder_batch, gpu_id, SKIP_GEOMETRIC_VERIFICATION, str(args.outputs_dir)): gpu_id 
                for folder_batch, gpu_id in gpu_assignments
            }
            
            # Collect results (with progress bar)
            with tqdm.tqdm(total=len(uuid_folders), desc="Processing progress") as pbar:
                for future in as_completed(future_to_gpu):
                    gpu_id = future_to_gpu[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        pbar.update(len(results))
                    except Exception as exc:
                        print(f'GPU {gpu_id} has an exception: {exc}')
    
    # Statistics
    total_elapsed = time.time() - total_start_time
    successful = sum(1 for success, _, _ in all_results if success)
    failed = len(all_results) - successful
    
    print(f"\nProcessing completed")
    print(f"Total time: {total_elapsed:.2f} seconds")
    print(f"Successfully processed: {successful} folders")
    print(f"Failed to process: {failed} folders")
    if len(uuid_folders) > 0:
        print(f"Average time per folder: {total_elapsed/len(uuid_folders):.2f} seconds")
    
    # Display failed folders
    if failed > 0:
        print(f"\nFailed folders:")
        for success, uuid_name, message in all_results:
            if not success:
                print(f"  - {uuid_name}: {message}")
    
    print(f"\nHTML files are saved in ./html/ directory")
