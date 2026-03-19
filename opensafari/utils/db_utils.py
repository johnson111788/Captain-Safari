"""
Database read utilities - read data from COLMAP database.db and HDF5 files
"""

import sqlite3
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def read_images_from_db(db_path: Path) -> Dict[int, str]:
    """
    Read images table from database.db, return {image_id: name} mapping
    
    Args:
        db_path: path to database.db
        
    Returns:
        {image_id: image_name} dictionary
    """
    conn = sqlite3.connect(f'file:{db_path}?mode=ro&immutable=1', uri=True)
    cursor = conn.cursor()
    
    cursor.execute("SELECT image_id, name FROM images")
    images = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    return images


def read_cameras_from_db(db_path: Path) -> Dict[int, Tuple]:
    """
    Read cameras table from database.db
    
    Returns:
        {camera_id: (model, width, height, params)} dictionary
    """
    conn = sqlite3.connect(f'file:{db_path}?mode=ro&immutable=1', uri=True)
    cursor = conn.cursor()
    
    cursor.execute("SELECT camera_id, model, width, height, params FROM cameras")
    cameras = {}
    for row in cursor.fetchall():
        camera_id, model, width, height, params_blob = row
        params = np.frombuffer(params_blob, dtype=np.float64)
        cameras[camera_id] = (model, width, height, params)
    
    conn.close()
    return cameras


def read_two_view_geometries(db_path: Path) -> Dict[Tuple[int, int], Dict]:
    """
    Read two_view_geometries table from database.db
    
    Returns:
        {(image_id1, image_id2): {
            'rows': int,  # inlier count
            'F': np.ndarray,  # fundamental matrix (9,)
            'E': np.ndarray,  # essential matrix (9,)
            'H': np.ndarray,  # homography matrix (9,)
            'config': int
        }}
    """
    conn = sqlite3.connect(f'file:{db_path}?mode=ro&immutable=1', uri=True)
    cursor = conn.cursor()
    
    # Note: The structure of the two_view_geometries table in COLMAP may vary between versions
    # Standard columns: pair_id, rows, data, config
    # rows = inlier count
    cursor.execute("SELECT pair_id, rows, data, config FROM two_view_geometries")
    
    geometries = {}
    for row in cursor.fetchall():
        pair_id, inlier_count, data_blob, config = row
        
        # pair_id encoding: image_id1 * 2147483647 + image_id2 (assuming)
        # Actual COLMAP uses different encoding, needs to be adjusted based on version
        # Here we use the standard COLMAP pair_id decoding
        image_id2 = pair_id % 2147483647
        image_id1 = (pair_id - image_id2) // 2147483647
        
        # data contains F/E/H matrices (each 9 doubles)
        if data_blob:
            data = np.frombuffer(data_blob, dtype=np.float64)
            # Usually the first 9 are F, then 9 are E, then 9 are H
            F = data[0:9] if len(data) >= 9 else None
            E = data[9:18] if len(data) >= 18 else None
            H = data[18:27] if len(data) >= 27 else None
        else:
            F = E = H = None
        
        geometries[(image_id1, image_id2)] = {
            'inlier_count': inlier_count,
            'F': F,
            'E': E,
            'H': H,
            'config': config
        }
    
    conn.close()
    return geometries


def read_matches_from_h5(matches_h5_path: Path, image_name1: str, image_name2: str) -> Optional[np.ndarray]:
    """
    Read matches from matches.h5 for two images
    
    Args:
        matches_h5_path: path to matches.h5
        image_name1, image_name2: image file names
        
    Returns:
        matches array (N, 2) or None (if not exists)
    """
    if not matches_h5_path.exists():
        return None
    
    try:
        with h5py.File(str(matches_h5_path), 'r') as f:
            # HLOC matches.h5 structure: f[image1][image2]['matches0']
            # matches0 is a one-dimensional array, the value at index i is the index of the matching keypoint in image 2 (-1 means no match)
            matches_nn = None
            swap = False
            
            if image_name1 in f and image_name2 in f[image_name1]:
                group = f[image_name1][image_name2]
                if 'matches0' in group:
                    matches_nn = group['matches0'][:]
                    swap = False
            elif image_name2 in f and image_name1 in f[image_name2]:
                group = f[image_name2][image_name1]
                if 'matches0' in group:
                    matches_nn = group['matches0'][:]
                    swap = True
            
            if matches_nn is None:
                return None
            
            # Convert NN format to (N, 2) format
            # Find all valid matches (!= -1)
            valid = matches_nn >= 0
            indices0 = np.where(valid)[0]
            indices1 = matches_nn[valid].astype(int)
            
            if len(indices0) == 0:
                return None
            
            if swap:
                matches = np.stack([indices1, indices0], axis=1)
            else:
                matches = np.stack([indices0, indices1], axis=1)
            
            return matches
    except Exception as e:
        logger.warning(f"Failed to read matches for {image_name1}/{image_name2}: {e}")
        return None


def read_keypoints_from_h5(features_h5_path: Path, image_name: str) -> Optional[np.ndarray]:
    """
    Read keypoints from features.h5 for an image
    
    Returns:
        keypoints array (N, 2) [x, y] or None
    """
    if not features_h5_path.exists():
        return None
    
    try:
        with h5py.File(str(features_h5_path), 'r') as f:
            if image_name not in f:
                return None
            # HLOC features.h5 usually stored as {image_name: {'keypoints': (N, 2)}}
            if 'keypoints' in f[image_name]:
                kpts = f[image_name]['keypoints'][...]
            else:
                # Some versions directly store (N, 2)
                kpts = f[image_name][...]
        return kpts
    except Exception as e:
        logger.warning(f"Failed to read keypoints for {image_name}: {e}")
        return None


def build_image_id_mapping(db_path: Path, frame_order: Optional[List[str]] = None) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Build mapping from database image_id to time-ordered index
    
    Args:
        db_path: path to database.db
        frame_order: optional list of frame names (in time order); if None, sort by file name
        
    Returns:
        (id_to_index, id_to_name):
            id_to_index: {image_id: time index}
            id_to_name: {image_id: image_name}
    """
    images = read_images_from_db(db_path)
    id_to_name = images
    
    if frame_order is None:
        # Sort by file name (assuming frame names contain timestamps or increasing numbers)
        sorted_names = sorted(images.values())
    else:
        sorted_names = frame_order
    
    name_to_index = {name: idx for idx, name in enumerate(sorted_names)}
    
    id_to_index = {}
    for img_id, img_name in images.items():
        if img_name in name_to_index:
            id_to_index[img_id] = name_to_index[img_name]
    
    return id_to_index, id_to_name


def get_transition_metrics_from_db(
    db_path: Path,
    matches_h5_path: Path,
    features_h5_path: Path,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    frame_order: Optional[List[str]] = None
) -> List[Dict]:
    """
    Get quick transition metrics for all adjacent frames (Stage A) from DB and HDF5
    
    Args:
        db_path: path to database.db
        matches_h5_path: matches.h5
        features_h5_path: features.h5
        extrinsics: (N, 3, 4) w2c extrinsics
        intrinsics: (N, 3, 3) intrinsics
        frame_order: optional list of frame names (in time order)
        
    Returns:
        List of dictionaries, each corresponding to the transition metrics for adjacent frames i -> i+1:
        {
            'idx': i,
            'image_id1': int,
            'image_id2': int,
            'image_name1': str,
            'image_name2': str,
            'inlier_count': int,
            'match_count': int,
            'inlier_ratio': float,
            'has_geometry': bool
        }
    """
    id_to_index, id_to_name = build_image_id_mapping(db_path, frame_order)
    index_to_id = {v: k for k, v in id_to_index.items()}
    
    # Read two_view_geometries
    geometries = read_two_view_geometries(db_path)
    
    N = len(extrinsics)
    transitions = []
    
    for i in range(N - 1):
        if i not in index_to_id or (i + 1) not in index_to_id:
            # Missing frame
            transitions.append({
                'idx': i,
                'image_id1': None,
                'image_id2': None,
                'image_name1': None,
                'image_name2': None,
                'inlier_count': 0,
                'match_count': 0,
                'inlier_ratio': 0.0,
                'has_geometry': False,
                'missing': True
            })
            continue
        
        id1 = index_to_id[i]
        id2 = index_to_id[i + 1]
        name1 = id_to_name[id1]
        name2 = id_to_name[id2]
        
        # Check two_view_geometries
        geom = geometries.get((id1, id2)) or geometries.get((id2, id1))
        
        if geom:
            inlier_count = geom['inlier_count']
            has_geometry = True
        else:
            inlier_count = 0
            has_geometry = False
        
        # Read match count
        matches = read_matches_from_h5(matches_h5_path, name1, name2)
        match_count = len(matches) if matches is not None else 0
        
        inlier_ratio = inlier_count / match_count if match_count > 0 else 0.0
        
        transitions.append({
            'idx': int(i),
            'image_id1': int(id1),
            'image_id2': int(id2),
            'image_name1': name1,
            'image_name2': name2,
            'inlier_count': int(inlier_count),
            'match_count': int(match_count),
            'inlier_ratio': float(inlier_ratio),
            'has_geometry': bool(has_geometry),
            'missing': False
        })
    
    return transitions

