#!/usr/bin/env python3
"""
Video SfM Trajectory Post-Verification & Repair Tool

Usage:
    python 3.verify_and_repair.py --case <case_id> [options]
    python 3.verify_and_repair.py --case_glob "*" [options]  # Batch processing
"""

import argparse
import logging
import yaml
import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.db_utils import get_transition_metrics_from_db, read_matches_from_h5, read_keypoints_from_h5
from utils.geometry import stage_b_geometric_check
from utils.kinematics import kinematic_check
from utils.repair import interpolate_poses, decide_action, validate_repair
from utils.visualization import plot_trajectory, plot_metrics

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_extrinsics(camera_dir: Path, case_id: str) -> np.ndarray:
    """
    Load extrinsics, support (N, 3, 4) and (N, 1, 3, 4)
    
    Returns:
        (N, 3, 4) extrinsics
    """
    ext_path = camera_dir / f"{case_id}_extrinsic.npy"
    
    if not ext_path.exists():
        raise FileNotFoundError(f"Extrinsics file not found: {ext_path}")
    
    extrinsics = np.load(ext_path)
    
    # Process dimensions
    if extrinsics.ndim == 4 and extrinsics.shape[1] == 1:
        extrinsics = extrinsics.squeeze(1)  # (N, 1, 3, 4) -> (N, 3, 4)
    
    if extrinsics.shape[1:] != (3, 4):
        raise ValueError(f"Extrinsic shape error: {extrinsics.shape}, expected (N, 3, 4)")
    
    # logger.info(f"Load extrinsics: {ext_path}, shape {extrinsics.shape}")
    return extrinsics


def load_intrinsics(camera_dir: Path, case_id: str) -> np.ndarray:
    """
    Load intrinsics, support (N, 3, 3) and (N, 1, 3, 3)
    
    Returns:
        (N, 3, 3) intrinsics
    """
    int_path = camera_dir / f"{case_id}_intrinsic.npy"
    
    if not int_path.exists():
        raise FileNotFoundError(f"Intrinsic file not found: {int_path}")
    
    intrinsics = np.load(int_path)
    
    if intrinsics.ndim == 4 and intrinsics.shape[1] == 1:
        intrinsics = intrinsics.squeeze(1)
    
    if intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Intrinsic shape error: {intrinsics.shape}, expected (N, 3, 3)")
    
    # logger.info(f"Load intrinsics: {int_path}, shape {intrinsics.shape}")
    return intrinsics


def apply_strict_mode(config: Dict):
    """Apply strict mode adjustment"""
    if not config['strict_mode']['enabled']:
        return
    
    logger.info("Enable strict mode, adjust thresholds...")
    
    # Stage A: increase min_inlier requirement
    mult = config['strict_mode']['stage_a_multiplier']
    original_min_inlier = config['stage_a']['min_inlier_count']
    config['stage_a']['min_inlier_count'] = int(original_min_inlier * mult)
    logger.info(f"  Stage A min_inlier: {original_min_inlier} → {config['stage_a']['min_inlier_count']}")
    
    # Stage B: decrease epipolar error tolerance
    div = config['strict_mode']['stage_b_epi_divider']
    config['stage_b']['epi_median_max'] /= div
    config['stage_b']['epi_p95_max'] /= div
    logger.info(f"  Stage B epi_median_max: → {config['stage_b']['epi_median_max']:.2f}px")
    
    # Kinematics: translation jump change more sensitive
    div_z = config['strict_mode']['kinematics_z_divider']
    original_trans_z = config['kinematics']['translation_mad_z_threshold']
    config['kinematics']['translation_mad_z_threshold'] /= div_z
    logger.info(f"  Kinematics translation_z: {original_trans_z} → {config['kinematics']['translation_mad_z_threshold']:.2f}")
    
    # Kinematics: rotation jump change more sensitive (newly added separate control)
    if 'kinematics_rotation_z_divider' in config['strict_mode']:
        div_rot_z = config['strict_mode']['kinematics_rotation_z_divider']
        original_rot_z = config['kinematics']['rotation_mad_z_threshold']
        config['kinematics']['rotation_mad_z_threshold'] /= div_rot_z
        logger.info(f"  Kinematics rotation_z: {original_rot_z} → {config['kinematics']['rotation_mad_z_threshold']:.2f}")
    else:
        # compatible with old configuration
        config['kinematics']['rotation_mad_z_threshold'] /= div_z
    
    # Repair: decrease maximum SLERP angle
    config['repair']['max_slerp_angle'] = config['strict_mode']['repair_max_slerp_angle']
    logger.info(f"  Repair max_slerp_angle: → {config['repair']['max_slerp_angle']}°")
    
    # Decision: decrease maximum bad point ratio (newly added)
    if 'max_bad_ratio_override' in config['strict_mode']:
        original_bad_ratio = config['decision']['max_bad_ratio']
        config['decision']['max_bad_ratio'] = config['strict_mode']['max_bad_ratio_override']
        logger.info(f"  Decision max_bad_ratio: {original_bad_ratio} → {config['decision']['max_bad_ratio']}")


def run_stage_a(
    db_path: Path,
    matches_h5_path: Path,
    features_h5_path: Path,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    config: Dict
) -> List[Dict]:
    """
    Stage A: quick DB check
    
    Returns:
        transition metrics list
    """
    # logger.info("Stage A: quick DB check...")
    
    transitions = get_transition_metrics_from_db(
        db_path,
        matches_h5_path,
        features_h5_path,
        extrinsics,
        intrinsics,
        frame_order=None  # automatically sort by file name
    )
    
    # apply thresholds
    min_inlier_count = config['stage_a']['min_inlier_count']
    min_inlier_ratio = config['stage_a']['min_inlier_ratio']
    
    for trans in transitions:
        if trans.get('missing', False):
            trans['stage_a_pass'] = False
            trans['stage_a_suspicious'] = True
            continue
        
        pass_count = trans['inlier_count'] >= min_inlier_count
        pass_ratio = trans['inlier_ratio'] >= min_inlier_ratio
        
        trans['stage_a_pass'] = bool(pass_count and pass_ratio)
        trans['stage_a_suspicious'] = bool(not trans['stage_a_pass'])
    
    suspicious_count = sum(1 for t in transitions if t.get('stage_a_suspicious', False))
    # logger.info(f"Stage A completed, suspicious transitions: {suspicious_count}/{len(transitions)}")
    
    return transitions


def run_stage_b(
    transitions: List[Dict],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    features_h5_path: Path,
    matches_h5_path: Path,
    config: Dict
):
    """
    Stage B: precise geometric check for suspicious transitions
    """
    # logger.info("Stage B: precise geometric check...")
    
    checked_count = 0
    
    for trans in transitions:
        if not trans.get('stage_a_suspicious', False):
            trans['stage_b_pass'] = True
            continue
        
        idx = trans['idx']
        
        # read matches
        matches = read_matches_from_h5(
            matches_h5_path,
            trans['image_name1'],
            trans['image_name2']
        )
        
        if matches is None or len(matches) < 10:
            trans['stage_b_pass'] = False
            trans['stage_b_reason'] = 'insufficient_matches'
            continue
        
        # read keypoints
        kpts1 = read_keypoints_from_h5(features_h5_path, trans['image_name1'])
        kpts2 = read_keypoints_from_h5(features_h5_path, trans['image_name2'])
        
        if kpts1 is None or kpts2 is None:
            trans['stage_b_pass'] = False
            trans['stage_b_reason'] = 'missing_keypoints'
            continue
        
        # extract matches coordinates
        pts1 = kpts1[matches[:, 0].astype(int)]
        pts2 = kpts2[matches[:, 1].astype(int)]
        
        # geometric check
        R1 = extrinsics[idx, :, :3]
        t1 = extrinsics[idx, :, 3]
        R2 = extrinsics[idx + 1, :, :3]
        t2 = extrinsics[idx + 1, :, 3]
        K1 = intrinsics[idx]
        K2 = intrinsics[idx + 1]
        
        geom_results = stage_b_geometric_check(
            pts1, pts2, K1, K2, R1, t1, R2, t2,
            config['stage_b']
        )
        
        trans.update(geom_results)
        trans['stage_b_pass'] = geom_results['pass_overall']
        
        checked_count += 1
    
    # logger.info(f"Stage B completed, precise check: {checked_count} transitions")


def run_kinematics(
    extrinsics: np.ndarray,
    transitions: List[Dict],
    config: Dict
):
    """kinematic check"""
    # logger.info("kinematic continuity check...")
    
    kin_results = kinematic_check(extrinsics, config['kinematics'])
    
    # merge to transitions
    for i, trans in enumerate(transitions):
        trans['translation_distance'] = float(kin_results['translation_distances'][i])
        trans['translation_z_score'] = float(kin_results['translation_z_scores'][i])
        trans['rotation_angle'] = float(kin_results['rotation_angles'][i])
        trans['rotation_z_score'] = float(kin_results['rotation_z_scores'][i])
        trans['forward_flip'] = bool(kin_results['forward_flip'][i])
        trans['kinematic_bad'] = bool(kin_results['overall_bad'][i])
    
    bad_count = kin_results['overall_bad'].sum()
    # logger.info(f"kinematic check completed, bad transitions: {bad_count}/{len(transitions)}")


def consolidate_bad_mask(transitions: List[Dict]) -> np.ndarray:
    """
    consolidate Stage A/B + kinematic, generate final bad_mask
    
    Returns:
        (N-1,) bool array
    """
    N = len(transitions)
    bad_mask = np.zeros(N, dtype=bool)
    
    for i, trans in enumerate(transitions):
        # any check failed -> bad
        stage_a_fail = not trans.get('stage_a_pass', True)
        stage_b_fail = not trans.get('stage_b_pass', True)
        kinematic_fail = trans.get('kinematic_bad', False)
        
        bad_mask[i] = stage_a_fail or stage_b_fail or kinematic_fail
        trans['is_bad'] = bool(bad_mask[i])
    
    return bad_mask


def process_case(
    case_id: str,
    outputs_dir: Path,
    camera_dir: Path,
    config: Dict,
    output_suffix: str = '_fixed.tight'
) -> Dict:
    """
    process single case
    
    Returns:
        result summary dictionary
    """
    # logger.info(f"\n{'='*60}")
    # logger.info(f"process case: {case_id}")
    # logger.info(f"{'='*60}")
    
    # path
    case_dir = outputs_dir / case_id
    db_path = case_dir / "sfm" / "database.db"
    matches_h5_path = case_dir / "matches.h5"
    features_h5_path = case_dir / "features.h5"
    
    # check files
    if not db_path.exists():
        logger.error(f"database.db not found: {db_path}")
        return {'case_id': case_id, 'status': 'error', 'reason': 'missing_db'}
    
    if not matches_h5_path.exists():
        logger.warning(f"matches.h5 not found: {matches_h5_path}")
    
    if not features_h5_path.exists():
        logger.warning(f"features.h5 not found: {features_h5_path}")
    
    # load data
    try:
        extrinsics = load_extrinsics(camera_dir, case_id)
        intrinsics = load_intrinsics(camera_dir, case_id)
    except Exception as e:
        logger.error(f"load extrinsics/intrinsics failed: {e}")
        return {
            'case_id': case_id,
            'num_frames': 0,
            'bad_count': 0,
            'bad_ratio': 0.0,
            'action': 'error',
            'final_status': 'error',
            'repaired_count': 0,
            'error_reason': str(e)
        }
    
    N = len(extrinsics)
    # logger.info(f"total frames: {N}")
    
    # check if frames are enough
    if N < 5:
        logger.warning(f"frames are not enough ({N} < 5), skip processing")
        return {
            'case_id': case_id,
            'num_frames': N,
            'bad_count': 0,
            'bad_ratio': 0.0,
            'action': 'skip',
            'final_status': 'skipped',
            'repaired_count': 0,
            'error_reason': 'insufficient_frames'
        }
    
    # Stage A
    transitions = run_stage_a(
        db_path, matches_h5_path, features_h5_path,
        extrinsics, intrinsics,
        config
    )
    
    # Stage B
    if features_h5_path.exists() and matches_h5_path.exists():
        run_stage_b(
            transitions, extrinsics, intrinsics,
            features_h5_path, matches_h5_path,
            config
        )
    
    # Kinematics
    run_kinematics(extrinsics, transitions, config)
    
    # consolidate decision
    bad_mask = consolidate_bad_mask(transitions)
    
    bad_count = bad_mask.sum()
    bad_ratio = bad_count / len(bad_mask) if len(bad_mask) > 0 else 0
    
    # logger.info(f"bad transitions: {bad_count}/{len(bad_mask)} ({bad_ratio:.1%})")
    
    # decision
    decision = decide_action(
        bad_mask,
        config['decision']['max_bad_run'],
        config['decision']['max_bad_ratio'],
        config['decision']['min_healthy_frames'],
        transitions=transitions,  # pass transition data for distance multiplier check
        distance_outlier_ratio_threshold=config['decision'].get('distance_outlier_ratio_threshold', 0.15),
        distance_multiplier_threshold=config['decision'].get('distance_multiplier_threshold', 2.0)
    )
    
    action = decision['action']
    # logger.info(f"decision: {action}")
    
    # prepare image names list for validation
    image_names = []
    for trans in transitions:
        if trans.get('image_name1') and trans['image_name1'] not in image_names:
            image_names.append(trans['image_name1'])
    # add last frame
    if transitions and transitions[-1].get('image_name2'):
        image_names.append(transitions[-1]['image_name2'])
    
    # execute action
    repaired_extrinsics = extrinsics
    repaired_indices = []
    final_status = action
    
    if action == 'repair':
        # logger.info("try to repair...")
        repaired_extrinsics, failed = interpolate_poses(
            extrinsics,
            decision['repairable_indices'],
            method=config['repair']['position_interp_method'],
            max_slerp_angle=config['repair']['max_slerp_angle']
        )
        
        if failed:
            logger.warning(f"partial repair failed: {failed}")
            repaired_indices = [idx for idx in decision['repairable_indices'] if idx not in failed]
            
            # check if all repairs failed
            if not repaired_indices:
                # all repairs failed, bad points cannot be repaired, should be dropped
                logger.error("all repair attempts failed, mark case as drop")
                final_status = 'drop'
                action = 'drop'  # update action to skip subsequent validation and saving
            else:
                final_status = 'partial_repair'
        else:
            # logger.info("repair successful")
            repaired_indices = decision['repairable_indices']
            final_status = 'repaired'
        
        # validate repaired result (if enabled)
        if config['repair']['post_repair_validation'] and repaired_indices:
            
            all_pass, validation_failed = validate_repair(
                repaired_extrinsics,
                intrinsics,
                repaired_indices,
                db_path,
                matches_h5_path,
                features_h5_path,
                image_names,
                config['stage_a'],
                config['stage_b'],
                config['kinematics']
            )
            
            if not all_pass:
                # logger.warning(f"validation failed frames: {validation_failed}")
                # any repair failed validation → directly drop the entire case
                # logger.error("validation failed, mark case as drop")
                final_status = 'drop'
                action = 'drop'  # update action to skip subsequent saving
                repaired_indices = []
    
    elif action == 'drop':
        # logger.warning("drop the entire case")
        final_status = 'drop'
    
    # output directory
    output_subdir = case_dir / config['output']['subdir']
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # save report.json
    report = {
        'case_id': case_id,
        'num_frames': N,
        'num_transitions': len(transitions),
        'bad_count': int(bad_count),
        'bad_ratio': float(bad_ratio),
        'action': action,
        'final_status': final_status,
        'repaired_indices': repaired_indices,
        'keep_range': decision.get('keep_range'),
        'transitions': transitions,
        'config': config
    }
    
    report_path = output_subdir / "report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    # logger.info(f"report saved: {report_path}")
    
    # visualization
    if config['output']['generate_visualization']:
        # logger.info("生成可视化...")
        
        # trajectory plot - Before
        plot_trajectory(
            extrinsics,
            bad_mask=bad_mask,
            repaired_indices=None,
            output_path=output_subdir / "trajectory_before.png",
            title=f"{case_id} - Before Repair",
            dpi=config['output']['viz_dpi'],
            figsize=config['output']['viz_figsize']
        )
        
        # trajectory plot - After
        if action in ['repair', 'repaired', 'partial_repair']:
            plot_trajectory(
                repaired_extrinsics,
                bad_mask=None,
                repaired_indices=repaired_indices,
                output_path=output_subdir / "trajectory_after.png",
                title=f"{case_id} - After Repair",
                dpi=config['output']['viz_dpi'],
                figsize=config['output']['viz_figsize']
            )
        
        # metrics plot
        plot_metrics(
            transitions,
            output_path=output_subdir / "metrics.png",
            dpi=config['output']['viz_dpi'],
            figsize=(14, 10)
        )
    
    # save to camera_fixed folder (only save successful cases: repair and pass)
    if config['output']['save_repaired'] and final_status not in ['drop', 'repair_failed']:
        # create camera_fixed folder
        camera_fixed_dir = camera_dir.parent / (camera_dir.name + output_suffix)
        camera_fixed_dir.mkdir(exist_ok=True)
        
        # save extrinsics and intrinsics (without _repaired suffix)
        extrinsic_path = camera_fixed_dir / f"{case_id}_extrinsic.npy"
        intrinsic_path = camera_fixed_dir / f"{case_id}_intrinsic.npy"
        
        if final_status in ['repaired', 'partial_repair']:
            # repair case: save repaired extrinsics
            np.save(extrinsic_path, repaired_extrinsics)
            np.save(intrinsic_path, intrinsics)  # intrinsics unchanged
            # logger.info(f"repaired extrinsics and intrinsics saved to {camera_fixed_dir}: {case_id}")
        elif final_status == 'pass':
            # pass case: save original extrinsics and intrinsics
            np.save(extrinsic_path, extrinsics)
            np.save(intrinsic_path, intrinsics)
            # logger.info(f"original extrinsics and intrinsics saved to {camera_fixed_dir}: {case_id}")
    
    # return summary
    summary = {
        'case_id': case_id,
        'num_frames': N,
        'bad_count': int(bad_count),
        'bad_ratio': float(bad_ratio),
        'action': action,
        'final_status': final_status,
        'repaired_count': len(repaired_indices),
        'error_reason': ''  # no error in normal case
    }
    
    return summary


def process_case_wrapper(case_id: str, outputs_dir: Path, camera_dir: Path, 
                         config: Dict, output_suffix: str) -> Dict:
    """
    wrapper function for multiprocessing, add exception handling
    """
    try:
        return process_case(case_id, outputs_dir, camera_dir, config, output_suffix)
    except Exception as e:
        logger.error(f"error processing {case_id}: {e}", exc_info=True)
        return {
            'case_id': case_id,
            'num_frames': 0,
            'bad_count': 0,
            'bad_ratio': 0.0,
            'action': 'error',
            'final_status': 'error',
            'repaired_count': 0,
            'error_reason': str(e)
        }


# python verify_and_repair.py --case_glob "*" > log.txt 2>&1
def main():
    parser = argparse.ArgumentParser(
        description='Video SfM Trajectory Post-Verification & Repair'
    )
    parser.add_argument('--case', type=str, help='Case ID (single case)')
    parser.add_argument('--case-glob', type=str, help='Case glob pattern (e.g., "*")')
    parser.add_argument('--target-cases', type=str, help='Comma-separated target case ID list')
    parser.add_argument('--outputs-dir', type=str, default='./data-frames_4fps-hloc',
                        help='Outputs directory')
    parser.add_argument('--camera-dir', type=str, default='./data-frames_4fps-camera',
                        help='Camera directory')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Config file path')
    parser.add_argument('--strict', type=str, choices=['true', 'false'],
                        help='Enable strict mode')
    parser.add_argument('--save_repaired', type=str, choices=['true', 'false'],
                        help='Save repaired extrinsics')
    parser.add_argument('--viz', type=str, choices=['true', 'false'],
                        help='Generate visualization')
    parser.add_argument('--output-suffix', type=str, default='_fixed',
                        help='Suffix for camera_fixed output directory (default: _fixed.tight)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count)')
    
    args = parser.parse_args()
    
    # load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # command line override
    if args.strict:
        config['strict_mode']['enabled'] = (args.strict == 'true')
    
    if args.save_repaired:
        config['output']['save_repaired'] = (args.save_repaired == 'true')
    
    if args.viz:
        config['output']['generate_visualization'] = (args.viz == 'true')
    
    # Apply strict mode
    apply_strict_mode(config)
    
    outputs_dir = Path(args.outputs_dir)
    camera_dir = Path(args.camera_dir)
    
    # get case list
    if args.target_cases:
        # use comma-separated case list
        case_ids = [c.strip() for c in args.target_cases.split(',') if c.strip()]
        logger.info(f"target to process {len(case_ids)} specified cases")
    elif args.case:
        case_ids = [args.case]
    elif args.case_glob:
        # find matching cases
        case_ids = [d.name for d in outputs_dir.glob(args.case_glob) if d.is_dir()]
        logger.info(f"found {len(case_ids)} cases")
    else:
        logger.error("please specify --case or --case-glob or --target-cases")
        sys.exit(1)
    
    # process all cases - use multiprocessing
    num_workers = args.num_workers if args.num_workers else cpu_count()
    logger.info(f"use {num_workers} parallel processes")
    
    # create partial function, fixed parameters
    process_func = partial(
        process_case_wrapper,
        outputs_dir=outputs_dir,
        camera_dir=camera_dir,
        config=config,
        output_suffix=args.output_suffix
    )
    
    # use multiprocessing pool to process
    summaries = []
    with Pool(processes=num_workers) as pool:
        # use imap_unordered with tqdm to show progress
        results = list(tqdm(
            pool.imap_unordered(process_func, case_ids),
            total=len(case_ids),
            desc="processing progress"
        ))
        summaries = results
    
    # save summary CSV
    if len(summaries) > 1:
        summary_csv_path = outputs_dir / "summary.csv"
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
            if summaries:
                # Collect all unique field names across summaries
                fieldnames = set()
                for s in summaries:
                    fieldnames.update(s.keys())
                fieldnames = list(fieldnames)
                # Optionally sort fieldnames for consistent order, or keep as list(fieldnames)
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for row in summaries:
                    writer.writerow(row)
        logger.info(f"\nsummary saved: {summary_csv_path}")
    
    logger.info("\nprocessing completed!")


if __name__ == '__main__':
    main()

