#!/usr/bin/env python3
"""
Fix model selection problem script

This script will:
1. Scan all reconstruction results in the outputs/ directory
2. Check all models in each output (sfm/ and models/ subdirectories)
3. Find the model with the most registered images
4. Move the correct model file to the sfm/ directory
"""

import pycolmap
import shutil
from pathlib import Path
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def read_reconstruction_safe(model_path: Path) -> Optional[pycolmap.Reconstruction]:
    """Safely read the reconstruction model"""
    try:
        reconstruction = pycolmap.Reconstruction()
        reconstruction.read(str(model_path))
        return reconstruction
    except Exception as e:
        print(f"    Failed to read {model_path}: {e}")
        return None


def analyze_output_folder(output_dir: Path) -> Tuple[Optional[str], int, Dict[str, int]]:
    """
    Analyze an output folder, find the largest model
    
    Returns:
        (best_model_path, max_images, all_models_info)
    """
    
    models_info = {}
    max_images = 0
    best_model = None
    
    sfm_dir = output_dir / "sfm"
    if not sfm_dir.exists():
        print(f"  Skip {output_dir.name}: no sfm directory")
        return None, 0, {}
    
    # 1. Check sfm/ directory itself
    sfm_reconstruction = read_reconstruction_safe(sfm_dir)
    if sfm_reconstruction:
        num_images = sfm_reconstruction.num_reg_images()
        models_info["sfm"] = num_images
        if num_images > max_images:
            max_images = num_images
            best_model = "sfm"
    
    # 2. Check models/ subdirectories
    models_path = sfm_dir / "models"
    if models_path.exists():
        for model_dir in models_path.iterdir():
            if model_dir.is_dir() and model_dir.name.isdigit():
                model_name = f"models/{model_dir.name}"
                
                # Check if there are reconstruction files, avoid reading empty folders
                filenames = ["images.bin", "cameras.bin", "points3D.bin", "frames.bin", "rigs.bin"]
                has_files = any((model_dir / filename).exists() for filename in filenames)
                
                if has_files:
                    reconstruction = read_reconstruction_safe(model_dir)
                    if reconstruction:
                        num_images = reconstruction.num_reg_images()
                        models_info[model_name] = num_images
                        if num_images > max_images:
                            max_images = num_images
                            best_model = model_name
    
    return best_model, max_images, models_info


def find_empty_model_folder(models_dir: Path) -> Optional[Path]:
    """Find empty models folder"""
    if not models_dir.exists():
        return None
    
    filenames = ["images.bin", "cameras.bin", "points3D.bin", "frames.bin", "rigs.bin"]
    
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            # Check if this folder is empty (no reconstruction files)
            has_files = any((model_dir / filename).exists() for filename in filenames)
            if not has_files:
                return model_dir
    
    return None


def backup_sfm_to_models(sfm_dir: Path) -> Optional[Path]:
    """Backup the current sfm file to an empty models folder"""
    models_dir = sfm_dir / "models"
    empty_folder = find_empty_model_folder(models_dir)
    
    if not empty_folder:
        print("  Warning: no empty models folder found, cannot backup original sfm file")
        return None
    
    filenames = ["images.bin", "cameras.bin", "points3D.bin", "frames.bin", "rigs.bin"]
    
    for filename in filenames:
        source_file = sfm_dir / filename
        backup_file = empty_folder / filename
        if source_file.exists():
            shutil.copy2(str(source_file), str(backup_file))
    
    return empty_folder


def move_model_to_sfm(output_dir: Path, best_model: str, dry_run: bool = False) -> bool:
    """Move the best model to the sfm directory"""
    sfm_dir = output_dir / "sfm"
    
    if best_model == "sfm":
        return True
    
    # Parse model path
    if best_model.startswith("models/"):
        model_index = best_model.split("/")[1]
        source_dir = sfm_dir / "models" / model_index
    else:
        print(f"  Unknown model path format: {best_model}")
        return False
    
    if not source_dir.exists():
        print(f"  Source directory does not exist: {source_dir}")
        return False
    
    
    if not dry_run:
        # Backup the original sfm file to an empty models folder
        backup_folder = backup_sfm_to_models(sfm_dir)
        
        # Move files
        filenames = ["images.bin", "cameras.bin", "points3D.bin", "frames.bin", "rigs.bin"]
        success_count = 0
        
        for filename in filenames:
            source_file = source_dir / filename
            target_file = sfm_dir / filename
            
            if source_file.exists():
                try:
                    # Delete existing file
                    if target_file.exists():
                        target_file.unlink()
                    
                    # Copy file (use copy instead of move, preserve original file)
                    shutil.copy2(str(source_file), str(target_file))
                    print(f"    ✓ Copy {filename}")
                    success_count += 1
                except Exception as e:
                    print(f"    ✗ Copy {filename} failed: {e}")
            else:
                print(f"    ⚠ Source file does not exist: {filename}")
        
        # Verify the move result
        final_reconstruction = read_reconstruction_safe(sfm_dir)
        if final_reconstruction:
            final_images = final_reconstruction.num_reg_images()
            print(f"  ✓ Move completed, final sfm/ directory has {final_images} registered images")
            return success_count == len(filenames)
        else:
            print(f"  ✗ Move failed, cannot read sfm/ directory")
            return False
    else:
        return True


def main():
    parser = argparse.ArgumentParser(description="Fix model selection problem in HLOC reconstruction")
    parser.add_argument("--outputs-dir", type=Path, default="./data-frames_4fps-hloc", 
                       help="Output directory path (default: data-frames_4fps-hloc)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Only analyze, do not actually move files")
    parser.add_argument("--specific-uuid", type=str, 
                       help="Only process specific UUID folders (deprecated, please use --target-cases)")
    parser.add_argument("--target-cases", type=str, default=None,
                       help="Comma-separated target case ID list (if specified, only process these cases)")
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"Error: output directory does not exist: {outputs_dir}")
        sys.exit(1)
    
    print(f"Scan output directory: {outputs_dir}")
    print(f"Mode: {'dry run' if args.dry_run else 'actual execution'}")
    
    # Get all output folders
    if args.target_cases:
        # Only process specified cases
        target_case_ids = set(args.target_cases.split(','))
        output_folders = [outputs_dir / case_id for case_id in target_case_ids if (outputs_dir / case_id).exists()]
        print(f"Process {len(target_case_ids)} specified cases, found {len(output_folders)} valid folders")
    elif args.specific_uuid:
        # Compatible with old parameters
        print("Warning: --specific-uuid is deprecated, please use --target-cases")
        output_folders = [outputs_dir / args.specific_uuid]
        if not output_folders[0].exists():
            print(f"Error: specified UUID folder does not exist: {output_folders[0]}")
            sys.exit(1)
    else:
        output_folders = [f for f in outputs_dir.iterdir() if f.is_dir()]
    
    print(f"Found {len(output_folders)} output folders")
    
    # Statistics
    total_processed = 0
    fixed_count = 0
    error_count = 0
    skip_count = 0
    
    # Process each folder
    for output_dir in tqdm(sorted(output_folders)):
        try:
            best_model, max_images, models_info = analyze_output_folder(output_dir)
            
            if not best_model:
                print(f"  Skip: cannot analyze or no valid model")
                print(f"    Skipped folder: {output_dir.name}")
                skip_count += 1
                continue
            
            total_processed += 1
            
            # Check if repair is needed - as long as there is a better model, repair
            current_sfm_images = models_info.get("sfm", 0)
            if best_model == "sfm":
                pass
            elif current_sfm_images >= max_images:
                pass
            else:
                pass
                
                if move_model_to_sfm(output_dir, best_model, args.dry_run):
                    fixed_count += 1
                else:
                    error_count += 1
        
        except Exception as e:
            print(f"  ✗ Process error: {e}")
            error_count += 1
    
    # Summary
    print(f"\n=== Processing completed ===")
    print(f"Total folders: {len(output_folders)}")
    print(f"Skip: {skip_count}")
    print(f"Processed: {total_processed}")
    print(f"Fixed: {fixed_count}")
    print(f"Failed: {error_count}")
    
    if args.dry_run:
        print(f"\nThis is a dry run. To actually execute the repair, please remove the --dry_run parameter")


if __name__ == "__main__":
    main()
