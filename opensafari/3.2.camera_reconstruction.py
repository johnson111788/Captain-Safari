#!/usr/bin/env python3
"""
Camera Reconstruction Pipeline
Progressive Rebuild Pipeline, automatically process the missing samples in camera_fixed

Workflow:
1. Scan camera_fixed, find the missing samples
2. Call 3.2.1.reconstruction.py to reconstruct these samples
3. Call 3.2.2.fix_model_selection.py to fix the model selection problem
4. Call 3.2.3.convert_hloc_to_vggt_format.py to convert the format
5. Call 3.2.4.verify_and_repair.py to verify and repair
6. Loop until there are no new samples to be repaired, or the maximum number of iterations is reached

Usage:
    python 3.2.camera_reconstruction.py --max-iterations 3 --num-gpus 8 --processes-per-gpu 10
"""

import argparse
import subprocess
import logging
from pathlib import Path
from typing import Set, List
import time

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_existing_cases_in_camera_fixed(camera_fixed_dir: Path) -> Set[str]:
    """Get the existing case IDs in camera_fixed"""
    if not camera_fixed_dir.exists():
        return set()
    
    existing = set()
    for file in camera_fixed_dir.glob("*_extrinsic.npy"):
        case_id = file.stem.replace("_extrinsic", "")
        existing.add(case_id)
    
    logger.info(f"There are {len(existing)} samples in camera_fixed")
    return existing


def get_all_cases_in_data(data_base_path: Path) -> Set[str]:
    """Get all the case IDs in the data directory"""
    all_cases = set()
    for folder in data_base_path.iterdir():
        if folder.is_dir():
            all_cases.add(folder.name)
    
    logger.info(f"There are {len(all_cases)} samples in the data directory")
    return all_cases


def get_missing_cases(all_cases: Set[str], existing_cases: Set[str]) -> List[str]:
    """Get the missing samples that need to be processed"""
    missing = sorted(list(all_cases - existing_cases))
    logger.info(f"Found {len(missing)} missing samples that need to be processed")
    return missing


def run_reconstruction(
    missing_cases: List[str],
    data_base_path: Path,
    outputs_dir: Path,
    num_gpus: int,
    processes_per_gpu: int,
    skip_geometric_verification: bool,
    batch_size: int = 100
) -> bool:
    """Run Step 1: Reconstruction (batch processing to avoid parameter过长)"""
    logger.info("=" * 80)
    logger.info(f"Step 1: Reconstruction {len(missing_cases)} samples (batch processing, each batch {batch_size} samples)")
    logger.info("=" * 80)
    
    # Batch processing
    total_batches = (len(missing_cases) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(missing_cases), batch_size):
        batch = missing_cases[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches}, containing {len(batch)} samples")
        
        cmd = [
            "python", "3.2.1.reconstruction.py",
            "--data-base-path", str(data_base_path),
            "--outputs-dir", str(outputs_dir),
            "--num-gpus", str(num_gpus),
            "--processes-per-gpu", str(processes_per_gpu),
            "--target-cases", ",".join(batch)
        ]
        
        if skip_geometric_verification:
            cmd.append("--skip-geometric-verification")
        
        logger.info(f"Executing command: {' '.join(cmd[:8])}... (containing {len(batch)} cases)")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            logger.info(f"✅ Batch {batch_num}/{total_batches} reconstruction completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Batch {batch_num}/{total_batches} reconstruction failed: {e}")
            return False
    
    logger.info("✅ All batches reconstruction completed")
    return True


def run_model_selection_fix(
    outputs_dir: Path,
    target_cases: List[str],
    batch_size: int = 100
) -> bool:
    """Run Step 1.5: Model selection repair (batch processing to avoid parameter过长)"""
    logger.info("=" * 80)
    logger.info(f"Step 1.5: Model selection repair {len(target_cases)} samples (batch processing, each batch {batch_size} samples)")
    logger.info("=" * 80)
    
    # Batch processing
    total_batches = (len(target_cases) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(target_cases), batch_size):
        batch = target_cases[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches}, containing {len(batch)} samples")
        
        cmd = [
            "python", "3.2.2.fix_model_selection.py",
            "--outputs-dir", str(outputs_dir),
            "--target-cases", ",".join(batch)
        ]
        
        logger.info(f"Executing command: {' '.join(cmd[:4])}... (containing {len(batch)} cases)")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            logger.info(f"✅ Batch {batch_num}/{total_batches} model selection repair completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Batch {batch_num}/{total_batches} model selection repair failed: {e}")
            return False
    
    logger.info("✅ All batches model selection repair completed")
    return True


def run_conversion(
    outputs_dir: Path,
    camera_dir: Path,
    target_cases: List[str],
    batch_size: int = 100
) -> bool:
    """Run Step 2: Format conversion (batch processing to avoid parameter过长)"""
    logger.info("=" * 80)
    logger.info(f"Step 2: Format conversion {len(target_cases)} samples (batch processing, each batch {batch_size} samples)")
    logger.info("=" * 80)
    
    # Batch processing
    total_batches = (len(target_cases) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(target_cases), batch_size):
        batch = target_cases[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches}, containing {len(batch)} samples")
        
        cmd = [
            "python", "3.2.3.convert_hloc_to_vggt_format.py",
            "--outputs-dir", str(outputs_dir),
            "--camera-dir", str(camera_dir),
            "--target-cases", ",".join(batch)
        ]
        
        logger.info(f"Executing command: {' '.join(cmd[:6])}... (containing {len(batch)} cases)")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            logger.info(f"✅ Batch {batch_num}/{total_batches} conversion completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Batch {batch_num}/{total_batches} conversion failed: {e}")
            return False
    
    logger.info("✅ All batches conversion completed")
    return True


def run_verification_and_repair(
    camera_dir: Path,
    camera_fixed_dir: Path,
    outputs_dir: Path,
    target_cases: List[str],
    batch_size: int = 100
) -> bool:
    """Run Step 3: Verification and repair (batch processing to avoid parameter过长)"""
    logger.info("=" * 80)
    logger.info(f"Step 3: Verification and repair {len(target_cases)} samples (batch processing, each batch {batch_size} samples)")
    logger.info("=" * 80)
    
    # Batch processing
    total_batches = (len(target_cases) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(target_cases), batch_size):
        batch = target_cases[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches}, containing {len(batch)} samples")
        
        cmd = [
            "python", "3.2.4.verify_and_repair.py",
            "--camera-dir", str(camera_dir),
            "--outputs-dir", str(outputs_dir),
            "--target-cases", ",".join(batch)
        ]
        
        logger.info(f"Executing command: {' '.join(cmd[:6])}... (containing {len(batch)} cases)")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            logger.info(f"✅ Batch {batch_num}/{total_batches} verification and repair completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Batch {batch_num}/{total_batches} verification and repair failed: {e}")
            return False
    
    logger.info("✅ All batches verification and repair completed")
    return True


def progressive_rebuild(
    data_base_path: Path,
    outputs_dir: Path,
    camera_dir: Path,
    camera_fixed_dir: Path,
    max_iterations: int,
    num_gpus: int,
    processes_per_gpu: int,
    skip_geometric_verification: bool,
    batch_size: int = 100
):
    """Progressive Rebuild Main Flow"""
    
    logger.info("=" * 80)
    logger.info("Progressive Rebuild Pipeline started")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_base_path}")
    logger.info(f"Output directory: {outputs_dir}")
    logger.info(f"Camera directory: {camera_dir}")
    logger.info(f"Fixed camera directory: {camera_fixed_dir}")
    logger.info(f"Maximum iterations: {max_iterations}")
    logger.info(f"Number of GPUs: {num_gpus}")
    logger.info(f"Number of processes per GPU: {processes_per_gpu}")
    logger.info("=" * 80)
    
    # Create necessary directories
    camera_fixed_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all samples
    all_cases = get_all_cases_in_data(data_base_path)
    
    for iteration in range(1, max_iterations + 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"Iteration {iteration}/{max_iterations}")
        logger.info("=" * 80)
        
        # Get existing and missing samples
        existing_cases = get_existing_cases_in_camera_fixed(camera_fixed_dir)
        missing_cases = get_missing_cases(all_cases, existing_cases)
        
        if not missing_cases:
            logger.info("✅ All samples have been processed!")
            break
        
        logger.info(f"This iteration needs to process {len(missing_cases)} missing samples")
        
        # Step 1: Reconstruction
        if not run_reconstruction(
            missing_cases,
            data_base_path,
            outputs_dir,
            num_gpus,
            processes_per_gpu,
            skip_geometric_verification,
            batch_size
        ):
            logger.error(f"❌ Iteration {iteration} reconstruction failed, skipping this iteration")
            continue
        
        # Step 1.5: Model selection repair
        if not run_model_selection_fix(outputs_dir, missing_cases, batch_size):
            logger.error(f"❌ Iteration {iteration} model selection repair failed, skipping this iteration")
            continue
        
        # Step 2: Format conversion
        if not run_conversion(outputs_dir, camera_dir, missing_cases, batch_size):
            logger.error(f"❌ Iteration {iteration} format conversion failed, skipping this iteration")
            continue
        
        # Step 3: Verification and repair
        if not run_verification_and_repair(
            camera_dir, camera_fixed_dir, outputs_dir, missing_cases, batch_size
        ):
            logger.error(f"❌ Iteration {iteration} verification and repair failed, skipping this iteration")
            continue
        
        # Check the newly added samples in this iteration
        new_existing = get_existing_cases_in_camera_fixed(camera_fixed_dir)
        newly_added = new_existing - existing_cases
        
        logger.info(f"This iteration added {len(newly_added)} successfully processed samples")
        
        if len(newly_added) == 0:
            logger.warning("⚠️ This iteration added no successfully processed samples, stopping the iteration")
            break
    
    # Final statistics
    final_existing = get_existing_cases_in_camera_fixed(camera_fixed_dir)
    final_missing = all_cases - final_existing
    
    logger.info("\n" + "=" * 80)
    logger.info("Progressive Rebuild Pipeline completed")
    logger.info("=" * 80)
    logger.info(f"Total number of samples: {len(all_cases)}")
    logger.info(f"Successfully processed samples: {len(final_existing)} ({len(final_existing)/len(all_cases)*100:.1f}%)")
    logger.info(f"Failed samples: {len(final_missing)} ({len(final_missing)/len(all_cases)*100:.1f}%)")
    
    if final_missing:
        logger.info(f"\nStill {len(final_missing)} samples not successfully processed")
        # Optional: save the failed list
        failed_list_path = Path("failed_cases.txt")
        with open(failed_list_path, 'w') as f:
            for case in sorted(final_missing):
                f.write(f"{case}\n")
        logger.info(f"Failed samples list saved to: {failed_list_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Progressive Rebuild Pipeline"
    )
    
    parser.add_argument(
        "--data-base-path",
        type=Path,
        default=Path("./data-frames_4fps"),
        help="Original data directory (default: data-frames_4fps)"
    )
    
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("./data-frames_4fps-hloc"),
        help="HLOC output directory (default: data-frames_4fps-hloc)"
    )
    
    parser.add_argument(
        "--camera-dir",
        type=Path,
        default=Path("./data-frames_4fps-camera"),
        help="Camera parameter directory (default: data-frames_4fps-camera)"
    )
    
    parser.add_argument(
        "--camera-fixed-dir",
        type=Path,
        default=Path("./data-frames_4fps-camera-fixed"),
        help="Fixed camera parameter directory (default: data-frames_4fps-camera-fixed)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum iterations (default: 3)"
    )
    
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs (default: 8)"
    )
    
    parser.add_argument(
        "--processes-per-gpu",
        type=int,
        default=5,
        help="Number of processes per GPU (default: 10)"
    )
    
    parser.add_argument(
        "--skip-geometric-verification",
        action="store_true",
        help="Skip geometric verification to speed up"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for batch processing, to avoid command line parameter过长 (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Run progressive rebuild
    progressive_rebuild(
        data_base_path=args.data_base_path,
        outputs_dir=args.outputs_dir,
        camera_dir=args.camera_dir,
        camera_fixed_dir=args.camera_fixed_dir,
        max_iterations=args.max_iterations,
        num_gpus=args.num_gpus,
        processes_per_gpu=args.processes_per_gpu,
        skip_geometric_verification=args.skip_geometric_verification,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

