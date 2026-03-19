"""
Geometric check module - symmetric epipolar error, RANSAC-E, triangulation angle
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def compute_essential_matrix_from_pose(
    K1: np.ndarray,
    K2: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray
) -> np.ndarray:
    """
    Compute essential matrix E from the w2c pose of two cameras
    
    Args:
        K1, K2: (3, 3) intrinsic matrices
        R1, t1: w2c extrinsic of the first camera [R|t]
        R2, t2: w2c extrinsic of the second camera [R|t]
        
    Returns:
        E: (3, 3) essential matrix (normalized coordinates)
    """
    # Compute relative pose: from camera1 to camera2
    # C1 = -R1^T @ t1, C2 = -R2^T @ t2
    # R_rel = R2 @ R1^T
    # t_rel = C1 - R_rel^T @ C2 = C1 - R1 @ R2^T @ C2
    
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    R_rel = R2 @ R1.T
    t_rel = C1 - R_rel.T @ C2
    
    # E = [t]_x @ R
    t_skew = skew_symmetric(t_rel.flatten())
    E = t_skew @ R_rel
    
    return E


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Construct the skew symmetric matrix of a vector
    
    Args:
        v: (3,) vector
        
    Returns:
        (3, 3) skew symmetric matrix
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def compute_symmetric_epipolar_error(
    pts1: np.ndarray,
    pts2: np.ndarray,
    E: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray
) -> np.ndarray:
    """
    Compute symmetric epipolar error (pixel units)
    
    For each match (p1, p2):
        err = 0.5 * (d(p2, l2) + d(p1, l1))
        其中 l2 = E @ K1^{-1} @ p1, l1 = E^T @ K2^{-1} @ p2
    
    Args:
        pts1: (N, 2) pixel points of the first image
        pts2: (N, 2) pixel points of the second image
        E: (3, 3) essential matrix
        K1, K2: (3, 3) intrinsic matrices
        
    Returns:
        (N,) symmetric epipolar error (pixels)
    """
    # Convert to homogeneous coordinates
    N = len(pts1)
    pts1_h = np.hstack([pts1, np.ones((N, 1))])  # (N, 3)
    pts2_h = np.hstack([pts2, np.ones((N, 1))])
    
    # Normalize to camera coordinates
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    
    pts1_norm = (K1_inv @ pts1_h.T).T  # (N, 3)
    pts2_norm = (K2_inv @ pts2_h.T).T
    
    # Compute epipolar lines
    # l2 = E @ pts1_norm^T -> (3, N)
    # l1 = E^T @ pts2_norm^T
    l2 = (E @ pts1_norm.T).T  # (N, 3)
    l1 = (E.T @ pts2_norm.T).T
    
    # Distance from point to epipolar line (normalized coordinates)
    # d(p, l) = |l^T @ p| / sqrt(l[0]^2 + l[1]^2)
    num2 = np.abs(np.sum(l2 * pts2_norm, axis=1))
    den2 = np.sqrt(l2[:, 0]**2 + l2[:, 1]**2)
    dist2_norm = num2 / (den2 + 1e-10)
    
    num1 = np.abs(np.sum(l1 * pts1_norm, axis=1))
    den1 = np.sqrt(l1[:, 0]**2 + l1[:, 1]**2)
    dist1_norm = num1 / (den1 + 1e-10)
    
    # Convert to pixel units (approximate: using the average focal length)
    f1 = (K1[0, 0] + K1[1, 1]) / 2
    f2 = (K2[0, 0] + K2[1, 1]) / 2
    
    dist1_px = dist1_norm * f1
    dist2_px = dist2_norm * f2
    
    # Symmetric error
    symmetric_err = 0.5 * (dist1_px + dist2_px)
    
    return symmetric_err


def compute_epipolar_metrics(
    pts1: np.ndarray,
    pts2: np.ndarray,
    E: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray
) -> Dict[str, float]:
    """
    Compute statistical metrics of epipolar error
    
    Returns:
        {
            'median': float,
            'mean': float,
            'p95': float,
            'max': float
        }
    """
    errors = compute_symmetric_epipolar_error(pts1, pts2, E, K1, K2)
    
    return {
        'median': float(np.median(errors)),
        'mean': float(np.mean(errors)),
        'p95': float(np.percentile(errors, 95)),
        'max': float(np.max(errors))
    }


def ransac_essential_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    threshold: float = 1.0,
    confidence: float = 0.999,
    max_iters: int = 2000
) -> Tuple[Optional[np.ndarray], float, np.ndarray]:
    """
    Re-estimate essential matrix using RANSAC
    
    Args:
        pts1, pts2: (N, 2) matching points (pixel coordinates)
        K1, K2: (3, 3) intrinsic matrices
        threshold: RANSAC threshold (normalized coordinates)
        confidence: RANSAC confidence
        max_iters: maximum number of iterations
        
    Returns:
        (E, inlier_ratio, inlier_mask):
            E: (3, 3) essential matrix (may be None)
            inlier_ratio: inlier ratio
            inlier_mask: (N,) bool array
    """
    # Normalize points
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    
    N = len(pts1)
    pts1_h = np.hstack([pts1, np.ones((N, 1))])
    pts2_h = np.hstack([pts2, np.ones((N, 1))])
    
    pts1_norm = (K1_inv @ pts1_h.T).T[:, :2]  # (N, 2)
    pts2_norm = (K2_inv @ pts2_h.T).T[:, :2]
    
    # RANSAC
    E, mask = cv2.findEssentialMat(
        pts1_norm,
        pts2_norm,
        np.eye(3),  # already normalized, using identity matrix
        method=cv2.RANSAC,
        prob=confidence,
        threshold=threshold,
        maxIters=max_iters
    )
    
    if E is None or mask is None:
        return None, 0.0, np.zeros(N, dtype=bool)
    
    mask = mask.flatten().astype(bool)
    inlier_ratio = mask.sum() / N
    
    return E, inlier_ratio, mask


def compute_relative_rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute the relative angle between two rotation matrices (degrees)
    
    Args:
        R1, R2: (3, 3) rotation matrices
        
    Returns:
        angle: relative angle between two rotation matrices (degrees)
    """
    R_rel = R2 @ R1.T
    # trace(R) = 1 + 2*cos(theta)
    trace = np.trace(R_rel)
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_rad = np.arccos(cos_theta)
    return np.degrees(theta_rad)


def check_triangulation_angle(
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    min_angle: float = 1.0
) -> bool:
    """
    Check if the triangulation angle is sufficient (approximated by the relative rotation angle)
    
    Args:
        R1, t1, R2, t2: w2c extrinsic
        min_angle: minimum angle (degrees)
        
    Returns:
        True if the angle is sufficient, False if degenerate
    """
    angle = compute_relative_rotation_angle(R1, R2)
    return angle >= min_angle


def stage_b_geometric_check(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    config: Dict
) -> Dict:
    """
    Stage B precise geometric check
    
    Args:
        pts1, pts2: (N, 2) matching points
        K1, K2: intrinsic matrices
        R1, t1, R2, t2: w2c extrinsic
        config: configuration dictionary (from config.yaml of stage_b)
        
    Returns:
        {
            'epi_median': float,
            'epi_p95': float,
            'ransac_inlier_ratio': float,
            'pass_epipolar': bool,
            'pass_ransac': bool,
            'pass_triangulation': bool,
            'pass_overall': bool
        }
    """
    # 1. Symmetric epipolar error based on pose
    E_pose = compute_essential_matrix_from_pose(K1, K2, R1, t1, R2, t2)
    epi_metrics = compute_epipolar_metrics(pts1, pts2, E_pose, K1, K2)
    
    pass_epipolar = (
        epi_metrics['median'] < config['epi_median_max'] and
        epi_metrics['p95'] < config['epi_p95_max']
    )
    
    # 2. RANSAC-E re-estimation
    _, ransac_inlier_ratio, _ = ransac_essential_matrix(
        pts1, pts2, K1, K2,
        threshold=config['ransac_threshold'],
        confidence=config['ransac_confidence'],
        max_iters=config['ransac_max_iters']
    )
    
    pass_ransac = ransac_inlier_ratio >= config['ransac_e_min_inlier_ratio']
    
    # 3. Triangulation angle check
    pass_triangulation = check_triangulation_angle(
        R1, t1, R2, t2,
        min_angle=config['triangulation_angle_min']
    )
    
    pass_overall = pass_epipolar and pass_ransac and pass_triangulation
    
    return {
        'epi_median': float(epi_metrics['median']),
        'epi_mean': float(epi_metrics['mean']),
        'epi_p95': float(epi_metrics['p95']),
        'epi_max': float(epi_metrics['max']),
        'ransac_inlier_ratio': float(ransac_inlier_ratio),
        'pass_epipolar': bool(pass_epipolar),
        'pass_ransac': bool(pass_ransac),
        'pass_triangulation': bool(pass_triangulation),
        'pass_overall': bool(pass_overall)
    }

