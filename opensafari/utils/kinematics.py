"""
Kinematic continuity check module - translation/rotation jumps, forward flip, second-order smoothness
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    """
    Compute camera centers from w2c extrinsic
    
    Args:
        extrinsics: (N, 3, 4) [R|t]
        
    Returns:
        (N, 3) camera centers C = -R^T @ t
    """
    N = len(extrinsics)
    centers = np.zeros((N, 3))
    
    for i in range(N):
        R = extrinsics[i, :, :3]
        t = extrinsics[i, :, 3]
        centers[i] = -R.T @ t
    
    return centers


def compute_forward_vectors(extrinsics: np.ndarray) -> np.ndarray:
    """
    Compute camera forward vectors (+Z direction)
    
    Args:
        extrinsics: (N, 3, 4)
        
    Returns:
        (N, 3) forward vectors f = R^T @ [0, 0, 1]^T
    """
    N = len(extrinsics)
    forward_vecs = np.zeros((N, 3))
    
    for i in range(N):
        R = extrinsics[i, :, :3]
        forward_vecs[i] = R.T @ np.array([0, 0, 1])
    
    return forward_vecs


def compute_rotation_angles(extrinsics: np.ndarray) -> np.ndarray:
    """
    Compute rotation angles between adjacent frames (degrees)
    
    Args:
        extrinsics: (N, 3, 4)
        
    Returns:
        (N-1,) rotation angles array
    """
    N = len(extrinsics)
    angles = np.zeros(N - 1)
    
    for i in range(N - 1):
        R1 = extrinsics[i, :, :3]
        R2 = extrinsics[i + 1, :, :3]
        R_rel = R2 @ R1.T
        
        trace = np.trace(R_rel)
        cos_theta = (trace - 1) / 2
        cos_theta = np.clip(cos_theta, -1, 1)
        angles[i] = np.degrees(np.arccos(cos_theta))
    
    return angles


def compute_mad_zscore(values: np.ndarray) -> np.ndarray:
    """
    Compute MAD (Median Absolute Deviation) z-score
    
    MAD z-score = |x - median(x)| / MAD
    where MAD = median(|x - median(x)|)
    
    Args:
        values: (N,) array of values
        
    Returns:
        (N,) MAD z-score array
    """
    if len(values) == 0:
        return np.array([])
    
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    
    if mad < 1e-10:
        # MAD is close to 0, all values are almost the same
        return np.zeros_like(values)
    
    # Constant factor 1.4826 makes MAD consistent with standard deviation under normal distribution
    # Here we use the original MAD
    z_scores = np.abs(values - median) / mad
    
    return z_scores


def check_translation_jumps(
    centers: np.ndarray,
    threshold: float = 4.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Check translation jumps
    
    Args:
        centers: (N, 3) camera centers
        threshold: MAD z-score threshold
        
    Returns:
        (distances, z_scores, is_bad):
            distances: (N-1,) distances between adjacent frames
            z_scores: (N-1,) MAD z-score
            is_bad: (N-1,) bool array, True means abnormal
    """
    N = len(centers)
    distances = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
    
    z_scores = compute_mad_zscore(distances)
    is_bad = z_scores > threshold
    
    return distances, z_scores, is_bad


def check_rotation_jumps(
    extrinsics: np.ndarray,
    abs_threshold: float = 45.0,
    mad_z_threshold: float = 4.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Check rotation jumps
    
    Args:
        extrinsics: (N, 3, 4)
        abs_threshold: absolute angle threshold (degrees)
        mad_z_threshold: MAD z-score threshold
        
    Returns:
        (angles, z_scores, is_bad)
    """
    angles = compute_rotation_angles(extrinsics)
    z_scores = compute_mad_zscore(angles)
    
    # Bad: either absolute angle or z-score exceeds threshold
    is_bad = (angles > abs_threshold) | (z_scores > mad_z_threshold)
    
    return angles, z_scores, is_bad


def check_forward_flips(
    forward_vecs: np.ndarray,
    angle_threshold: float = 120.0
) -> np.ndarray:
    """
    Check forward flips
    
    Args:
        forward_vecs: (N, 3)
        angle_threshold: angle threshold (degrees), if exceeds, it is considered a flip
        
    Returns:
        (N-1,) bool array, True means a flip
    """
    N = len(forward_vecs)
    cos_threshold = np.cos(np.radians(angle_threshold))
    
    is_flipped = np.zeros(N - 1, dtype=bool)
    
    for i in range(N - 1):
        dot_product = np.dot(forward_vecs[i], forward_vecs[i + 1])
        # Normalize
        norm_product = np.linalg.norm(forward_vecs[i]) * np.linalg.norm(forward_vecs[i + 1])
        if norm_product > 1e-10:
            cos_angle = dot_product / norm_product
            is_flipped[i] = cos_angle < cos_threshold
    
    return is_flipped


def check_second_order_smoothness(
    centers: np.ndarray,
    extrinsics: np.ndarray,
    translation_threshold: float = 4.5,
    rotation_threshold: float = 4.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check second-order smoothness (acceleration anomaly)
    
    Args:
        centers: (N, 3)
        extrinsics: (N, 3, 4)
        translation_threshold: translation acceleration MAD z-score threshold
        rotation_threshold: rotation acceleration MAD z-score threshold
        
    Returns:
        (trans_is_bad, rot_is_bad):
            trans_is_bad: (N-2,) translation acceleration anomaly
            rot_is_bad: (N-2,) rotation acceleration anomaly
    """
    N = len(centers)
    
    # Translation second-order difference: |C_{i+1} - 2*C_i + C_{i-1}|
    trans_accel = np.linalg.norm(
        centers[2:] - 2 * centers[1:-1] + centers[:-2],
        axis=1
    )
    
    trans_z = compute_mad_zscore(trans_accel)
    trans_is_bad = trans_z > translation_threshold
    
    # Rotation angle first-order difference: |θ_{i+1} - θ_i|
    angles = compute_rotation_angles(extrinsics)
    if len(angles) >= 2:
        rot_accel = np.abs(angles[1:] - angles[:-1])
        rot_z = compute_mad_zscore(rot_accel)
        rot_is_bad = rot_z > rotation_threshold
    else:
        rot_is_bad = np.array([], dtype=bool)
    
    return trans_is_bad, rot_is_bad


def kinematic_check(
    extrinsics: np.ndarray,
    config: Dict
) -> Dict:
    """
    Complete kinematic continuity check
    
    Args:
        extrinsics: (N, 3, 4)
        config: configuration dictionary (from config.yaml of kinematics)
        
    Returns:
        {
            'translation_distances': (N-1,),
            'translation_z_scores': (N-1,),
            'translation_bad': (N-1,) bool,
            'rotation_angles': (N-1,),
            'rotation_z_scores': (N-1,),
            'rotation_bad': (N-1,) bool,
            'forward_flip': (N-1,) bool,
            'second_order_trans_bad': (N-2,) bool,
            'second_order_rot_bad': (N-2,) bool,
            'overall_bad': (N-1,) bool  # any check failed
        }
    """
    centers = compute_camera_centers(extrinsics)
    forward_vecs = compute_forward_vectors(extrinsics)
    
    # First-order check
    trans_dist, trans_z, trans_bad = check_translation_jumps(
        centers,
        threshold=config['translation_mad_z_threshold']
    )
    
    rot_angles, rot_z, rot_bad = check_rotation_jumps(
        extrinsics,
        abs_threshold=config['rotation_angle_max'],
        mad_z_threshold=config['rotation_mad_z_threshold']
    )
    
    forward_flip = check_forward_flips(
        forward_vecs,
        angle_threshold=config['forward_flip_angle_max']
    )
    
    # Second-order check
    N = len(extrinsics)
    if config['enable_second_order'] and N >= 3:
        trans_accel_bad, rot_accel_bad = check_second_order_smoothness(
            centers,
            extrinsics,
            translation_threshold=config['translation_accel_mad_z_threshold'],
            rotation_threshold=config['rotation_accel_mad_z_threshold']
        )
        
        # Align to (N-1,) length: second-order anomalies affect two transitions i and i+1
        # trans_accel_bad[j] corresponds to centers[j:j+3], affecting transitions j and j+1
        trans_2nd_aligned = np.zeros(N - 1, dtype=bool)
        rot_2nd_aligned = np.zeros(N - 1, dtype=bool)
        
        for j in range(len(trans_accel_bad)):
            if trans_accel_bad[j]:
                trans_2nd_aligned[j] = True
                if j + 1 < N - 1:
                    trans_2nd_aligned[j + 1] = True
        
        for j in range(len(rot_accel_bad)):
            if rot_accel_bad[j]:
                rot_2nd_aligned[j] = True
                if j + 1 < N - 1:
                    rot_2nd_aligned[j + 1] = True
    else:
        trans_accel_bad = np.array([], dtype=bool)
        rot_accel_bad = np.array([], dtype=bool)
        trans_2nd_aligned = np.zeros(N - 1, dtype=bool)
        rot_2nd_aligned = np.zeros(N - 1, dtype=bool)
    
    # Overall judgment
    overall_bad = trans_bad | rot_bad | forward_flip | trans_2nd_aligned | rot_2nd_aligned
    
    return {
        'translation_distances': trans_dist,
        'translation_z_scores': trans_z,
        'translation_bad': trans_bad,
        'rotation_angles': rot_angles,
        'rotation_z_scores': rot_z,
        'rotation_bad': rot_bad,
        'forward_flip': forward_flip,
        'second_order_trans_bad': trans_accel_bad,
        'second_order_rot_bad': rot_accel_bad,
        'overall_bad': overall_bad
    }

