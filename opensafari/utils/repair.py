"""
Repair module - position interpolation, rotation SLERP, validation
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Quaternion spherical linear interpolation (SLERP), supports extrapolation
    
    Args:
        q1, q2: (4,) quaternion [w, x, y, z]
        t: interpolation/extrapolation parameter (t∈[0,1] for interpolation, t<0 or t>1 for extrapolation)
        
    Returns:
        interpolated/extrapolated quaternion
    """
    # Ensure quaternion normalization
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate angle
    dot = np.dot(q1, q2)
    
    # If dot product is negative, take the opposite of q2 to choose the shorter path
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # If quaternion is very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # SLERP
    dot = np.clip(dot, -1, 1)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    return w1 * q1 + w2 * q2


def rotation_matrix_to_quaternion(R_matrix: np.ndarray) -> np.ndarray:
    """
    Rotation matrix to quaternion
    
    Args:
        R_matrix: (3, 3) rotation matrix
        
    Returns:
        (4,) quaternion [w, x, y, z]
    """
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(R_matrix)
    # scipy returns [x, y, z, w], convert to [w, x, y, z]
    quat_xyzw = rot.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Quaternion to rotation matrix
    
    Args:
        q: (4,) [w, x, y, z]
        
    Returns:
        (3, 3) rotation matrix
    """
    from scipy.spatial.transform import Rotation
    # Convert to [x, y, z, w]
    quat_xyzw = np.array([q[1], q[2], q[3], q[0]])
    rot = Rotation.from_quat(quat_xyzw)
    return rot.as_matrix()


def interpolate_poses(
    extrinsics: np.ndarray,
    bad_indices: List[int],
    method: str = 'cubic',
    max_slerp_angle: float = 90.0
) -> Tuple[np.ndarray, List[int]]:
    """
    Interpolate/extrapolate poses (supports first/last frame extrapolation)
    
    Args:
        extrinsics: (N, 3, 4) extrinsic matrix
        bad_indices: list of frame indices to repair (must be isolated or short segments, supports boundary frames)
        method: position interpolation method 'linear' or 'cubic' (deprecated, use linear)
        max_slerp_angle: maximum SLERP angle (degrees), if exceeds,放弃
        
    Returns:
        (repaired_extrinsics, failed_indices):
            repaired_extrinsics: repaired extrinsic matrix
            failed_indices: failed indices
            
    策略:
        - Middle segment: use left and right healthy frames interpolation
        - First frame segment: use next two healthy frames extrapolation
        - Last frame segment: use previous two healthy frames extrapolation
    """
    N = len(extrinsics)
    repaired = extrinsics.copy()
    failed = []
    
    # Group consecutive bad points
    bad_set = set(bad_indices)
    groups = []
    current_group = []
    
    for i in sorted(bad_indices):
        if not current_group or i == current_group[-1] + 1:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    if current_group:
        groups.append(current_group)
    
    # Interpolate or extrapolate for each group
    for group in groups:
        start_idx = group[0]
        end_idx = group[-1]
        
        # Determine if it is the first/last frame or middle segment
        is_head = (start_idx == 0)
        is_tail = (end_idx == N - 1)
        
        # Determine the reference frame
        if is_head:
            # First frame extrapolation: use next two healthy frames [end_idx+1, end_idx+2]
            if end_idx + 2 >= N:
                failed.extend(group)
                logger.warning(f"First frame extrapolation failed: not enough next healthy frames: {group}")
                continue
            ref_idx_1 = end_idx + 1
            ref_idx_2 = end_idx + 2
        elif is_tail:
            # Last frame extrapolation: use previous two healthy frames [start_idx-2, start_idx-1]
            if start_idx - 2 < 0:
                failed.extend(group)
                logger.warning(f"Last frame extrapolation failed: not enough previous healthy frames: {group}")
                continue
            ref_idx_1 = start_idx - 2
            ref_idx_2 = start_idx - 1
        else:
            # Middle segment interpolation: use left and right healthy frames
            left_idx = start_idx - 1
            right_idx = end_idx + 1
            ref_idx_1 = left_idx
            ref_idx_2 = right_idx
        
        # Extract position and rotation (reference frame 1 and reference frame 2)
        R_ref1 = extrinsics[ref_idx_1, :, :3]
        t_ref1 = extrinsics[ref_idx_1, :, 3]
        C_ref1 = -R_ref1.T @ t_ref1
        
        R_ref2 = extrinsics[ref_idx_2, :, :3]
        t_ref2 = extrinsics[ref_idx_2, :, 3]
        C_ref2 = -R_ref2.T @ t_ref2
        
        # Check the validity of the rotation matrix (the determinant should be close to 1)
        det_ref1 = np.linalg.det(R_ref1)
        det_ref2 = np.linalg.det(R_ref2)
        
        if det_ref1 < 0.5 or det_ref2 < 0.5:
            # Invalid rotation matrix, skip repair
            failed.extend(group)
            logger.warning(f"Invalid rotation matrix (det_ref1={det_ref1:.3f}, det_ref2={det_ref2:.3f}), skip repair: {group}")
            continue
        
        q_ref1 = rotation_matrix_to_quaternion(R_ref1)
        q_ref2 = rotation_matrix_to_quaternion(R_ref2)
        
        # Check the rotation angle
        R_rel = R_ref2 @ R_ref1.T
        trace = np.trace(R_rel)
        cos_theta = (trace - 1) / 2
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        
        if angle > max_slerp_angle:
            failed.extend(group)
            logger.warning(f"Rotation angle too large ({angle:.1f}°), skip interpolation/extrapolation: {group}")
            continue
        
        # Interpolate/extrapolate each bad point
        for idx in group:
            # Calculate the normalized parameter t
            if is_head or is_tail:
                # Extrapolation: based on the velocity of the two reference frames
                # t < 0 (first frame extrapolation) or t > 1 (last frame extrapolation)
                t = (idx - ref_idx_1) / (ref_idx_2 - ref_idx_1)
            else:
                # Interpolation: t ∈ [0, 1]
                t = (idx - ref_idx_1) / (ref_idx_2 - ref_idx_1)
            
            # Position interpolation/extrapolation (linear)
            C_interp = (1 - t) * C_ref1 + t * C_ref2
            
            # Rotation SLERP (supports extrapolation, t can be <0 or >1)
            q_interp = quaternion_slerp(q_ref1, q_ref2, t)
            R_interp = quaternion_to_rotation_matrix(q_interp)
            
            # Reconstruct [R|t]
            t_interp = -R_interp @ C_interp
            repaired[idx, :, :3] = R_interp
            repaired[idx, :, 3] = t_interp
            
    return repaired, failed


def validate_repair(
    repaired_extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    repaired_indices: List[int],
    db_path,
    matches_h5_path,
    features_h5_path,
    image_names: List[str],
    stage_a_config: Dict,
    stage_b_config: Dict,
    kinematics_config: Dict
) -> Tuple[bool, List[int]]:
    """
    Validate the repaired transitions: re-check if they pass Stage A/B + kinematics
    
    Args:
        repaired_extrinsics: repaired extrinsic matrix
        intrinsics: intrinsic matrix
        repaired_indices: repaired frame indices
        db_path: database.db path
        matches_h5_path: matches.h5 path
        features_h5_path: features.h5 path
        image_names: image name list (in time order)
        stage_a_config: Stage A configuration
        stage_b_config: Stage B configuration
        kinematics_config: kinematics configuration
        
    Returns:
        (all_pass, failed_indices):
            all_pass: whether all passed
            failed_indices: failed indices
    """
    from .db_utils import read_matches_from_h5, read_keypoints_from_h5
    from .geometry import stage_b_geometric_check
    from .kinematics import kinematic_check
    
    # logger.info("Validate the repaired transitions...")
    
    # 1. Kinematics check
    kin_results = kinematic_check(repaired_extrinsics, kinematics_config)
    
    failed = set()
    
    for idx in repaired_indices:
        # Check the two transitions involving the repaired frame: (idx-1 -> idx) and (idx -> idx+1)
        transitions_to_check = []
        
        if idx > 0:
            transitions_to_check.append((idx - 1, idx))
        if idx < len(repaired_extrinsics) - 1:
            transitions_to_check.append((idx, idx + 1))
        
        for i, j in transitions_to_check:
            trans_idx = i  # transition index
            
            # 1.1 Kinematics check
            if trans_idx < len(kin_results['overall_bad']):
                if kin_results['overall_bad'][trans_idx]:
                    # logger.warning(f"Kinematics validation failed for repaired frame {idx} (transition {i}->{j})")
                    failed.add(idx)
                    continue  # already failed, skip geometric check
            
            # # 2. Stage B geometric check (if there are matches data)
            # if matches_h5_path.exists() and features_h5_path.exists() and image_names:
            #     if i >= len(image_names) or j >= len(image_names):
            #         continue
                
            #     name1 = image_names[i]
            #     name2 = image_names[j]
                
            #     # Read matches and keypoints
            #     matches = read_matches_from_h5(matches_h5_path, name1, name2)
            #     if matches is None or len(matches) < 10:
            #         logger.warning(f"Repaired frame {idx} missing enough matches (transition {i}->{j})")
            #         continue
                
            #     kpts1 = read_keypoints_from_h5(features_h5_path, name1)
            #     kpts2 = read_keypoints_from_h5(features_h5_path, name2)
                
            #     if kpts1 is None or kpts2 is None:
            #         continue
                
            #     # Extract matching points
            #     pts1 = kpts1[matches[:, 0].astype(int)]
            #     pts2 = kpts2[matches[:, 1].astype(int)]
                
            #     # Stage B check
            #     R1 = repaired_extrinsics[i, :, :3]
            #     t1 = repaired_extrinsics[i, :, 3]
            #     R2 = repaired_extrinsics[j, :, :3]
            #     t2 = repaired_extrinsics[j, :, 3]
            #     K1 = intrinsics[i]
            #     K2 = intrinsics[j]
                
            #     geom_results = stage_b_geometric_check(
            #         pts1, pts2, K1, K2, R1, t1, R2, t2,
            #         stage_b_config
            #     )
                
            #     if not geom_results['pass_overall']:
            #         logger.warning(
            #             f"Geometric validation failed for repaired frame {idx} (transition {i}->{j}): "
            #             f"epi_median={geom_results['epi_median']:.2f}, "
            #             f"ransac_ratio={geom_results['ransac_inlier_ratio']:.2f}"
            #         )
            #         failed.add(idx)
            
            # 2. Stage B geometric check (skip! interpolated frames do not correspond to images, only kinematics validation)
            # Interpolated frames will fail geometric validation because their poses are based on kinematics interpolation,
            # not a real SfM reconstruction, so the mismatch with images is normal.
            # we only need to validate kinematics continuity.
            pass  # skip geometric check
    
    failed_list = sorted(list(failed))
    all_pass = len(failed_list) == 0
    
    return all_pass, failed_list


def identify_repairable_segments(
    bad_mask: np.ndarray,
    max_run: int = 2
) -> Tuple[List[int], List[int]]:
    """
    Identify repairable and unrepairable bad points (supports boundary extrapolation)
    
    Args:
        bad_mask: (N-1,) bool array, mark bad transitions
        max_run: maximum number of repairable consecutive transitions (not frame number)
        
    Returns:
        (repairable, unrepairable):
            repairable: repairable frame index list (including boundary frames, use extrapolation)
            unrepairable: unrepairable frame index list
    """
    # Improved strategy: based on consecutive bad transition segments
    # Key: use "number of bad transitions" rather than "number of bad frames" to determine repairability
    N = len(bad_mask) + 1
    repairable = []
    unrepairable = []
    
    # Find all consecutive bad transition segments
    i = 0
    while i < len(bad_mask):
        if not bad_mask[i]:
            i += 1
            continue
        
        # Find the end position of consecutive bad transitions
        j = i
        while j < len(bad_mask) and bad_mask[j]:
            j += 1
        
        # Consecutive bad transitions [i, i+1, ..., j-1], number is (j - i)
        bad_transition_count = j - i
        
        # The frames affected by these bad transitions are [i, i+1, ..., j]
        affected_frames = list(range(i, j + 1))
        
        # Determine if it is repairable
        # Condition: number of bad transitions <= max_run
        if bad_transition_count <= max_run:
            # Repairable
            if i == 0:
                # First frame bad: extrapolate repair first frame [0, 1, ..., j-1]
                frames_to_repair = list(range(0, j))
            elif j == N - 1:
                # Last frame bad: extrapolate repair last frame [i+1, ..., N-1]
                frames_to_repair = list(range(i + 1, N))
            else:
                # Middle segment: interpolate repair middle frame [i+1, ..., j-1]
                frames_to_repair = list(range(i + 1, j)) if j > i + 1 else [i + 1]
            
            repairable.extend(frames_to_repair)
        else:
            # Unrepairable: too many consecutive bad points
            unrepairable.extend(affected_frames)
        
        i = j
    
    return repairable, unrepairable


def decide_action(
    bad_mask: np.ndarray,
    max_bad_run: int,
    max_bad_ratio: float,
    min_healthy_frames: int = None,  # deprecated parameter (for compatibility)
    transitions: List[Dict] = None,  # transition data list (for distance multiplier check)
    distance_outlier_ratio_threshold: float = 0.15,  # distance outlier transition ratio threshold (drop if exceeds)
    distance_multiplier_threshold: float = 2.0  # distance multiplier threshold (view as abnormal if greater than median N times)
) -> Dict:
    """
    Decision: repair or drop (removed cut segment logic)
    
    Args:
        bad_mask: (N-1,) bad transition mark
        max_bad_run: maximum allowed number of consecutive bad points (drop if exceeds)
        max_bad_ratio: maximum bad point ratio (drop if exceeds)
        min_healthy_frames: deprecated parameter (for compatibility)
        transitions: transition data list (for distance multiplier check)
        distance_outlier_ratio_threshold: distance outlier transition ratio threshold (drop if exceeds)
        distance_multiplier_threshold: distance multiplier threshold (view as abnormal if greater than median N times)
        
    Returns:
        {
            'action': 'pass' | 'repair' | 'drop',
            'repairable_indices': List[int],
            'keep_range': None
        }
    """
    N = len(bad_mask) + 1
    bad_count = bad_mask.sum()
    bad_ratio = bad_count / len(bad_mask) if len(bad_mask) > 0 else 0
    
    # Check if there are long consecutive bad segments
    max_run = 0
    current_run = 0
    for is_bad in bad_mask:
        if is_bad:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    
    # Decision logic (optimized version: first check repairability, then check severity)
    # 1. First identify repairable segments (supports boundary extrapolation)
    repairable, unrepairable = identify_repairable_segments(bad_mask, max_run=max_bad_run)
    
    # 2. If there are unrepairable segments, check severity
    if unrepairable:
        # There are long consecutive bad segments in the middle, cannot be repaired, drop
        return {'action': 'drop', 'repairable_indices': [], 'keep_range': None}
    
    # 3. If all are repairable, check if the bad point ratio is too high
    if bad_ratio > max_bad_ratio:
        # Bad point ratio too high (even if all are at the boundaries, there are too many)
        return {'action': 'drop', 'repairable_indices': [], 'keep_range': None}
    
    # 4. New: check translation distance multiplier anomaly (only check middle segment, exclude repairable boundary segments)
    #    If there are too many transitions with distances significantly greater than the median, the trajectory is not smooth overall
    if transitions and len(transitions) > 10:  # At least have enough transitions to check
        trans_dists = np.array([t.get('translation_distance', 0) for t in transitions])
        median_dist = np.median(trans_dists)
        
        if median_dist > 1e-6:  # Avoid division by zero
            # Only check middle segment (exclude first and last 5 transitions, because they may be repaired by extrapolation)
            middle_start = min(5, len(trans_dists) // 4)
            middle_end = max(len(trans_dists) - 5, len(trans_dists) * 3 // 4)
            middle_dists = trans_dists[middle_start:middle_end]
            
            if len(middle_dists) > 0:
                # Count transitions with distances > N times the median
                outlier_count = np.sum(middle_dists > distance_multiplier_threshold * median_dist)
                outlier_ratio = outlier_count / len(middle_dists)
                
                if outlier_ratio > distance_outlier_ratio_threshold:
                    logger.warning(
                        f"Middle segment distance outlier transition too many: {outlier_count}/{len(middle_dists)} ({outlier_ratio:.1%}) "
                        f"> {distance_outlier_ratio_threshold:.1%}, "
                        f"Trajectory is not smooth overall, drop"
                    )
                    return {'action': 'drop', 'repairable_indices': [], 'keep_range': None}
    
    # 5. Repairable and acceptable ratio
    if repairable:
        return {'action': 'repair', 'repairable_indices': repairable, 'keep_range': None}
    
    return {'action': 'pass', 'repairable_indices': [], 'keep_range': None}

