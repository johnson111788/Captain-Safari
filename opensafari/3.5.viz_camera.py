#!/usr/bin/env python3
"""
Visualize camera trajectories from extrinsic numpy files.
Saves 3D plots showing camera positions and orientations.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Visualize camera trajectories from extrinsic files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing extrinsic.npy files")
    args = parser.parse_args()

    data_path = args.input_dir
    output_dir = os.path.join(data_path, 'camera_trajectory')
    os.makedirs(output_dir, exist_ok=True)
    
    extrinsic_clips = [file for file in os.listdir(data_path) if file.endswith('extrinsic.npy')]
    
    if not extrinsic_clips:
        print(f"No extrinsic.npy files found in {data_path}")
        return

    print(f"Found {len(extrinsic_clips)} extrinsic files to visualize.")
    
    for extrinsic_clip_path in tqdm(extrinsic_clips, desc="Rendering trajectories"):
        extrinsic_clip = np.load(os.path.join(data_path, extrinsic_clip_path))
        
        # Remove extra dimensions to facilitate processing
        extrinsic_clip_squeezed = extrinsic_clip.squeeze(1) if len(extrinsic_clip.shape) == 4 else extrinsic_clip
        
        # Original camera center position (w2c format, needs to be converted to camera center)
        # Convert w2c to camera center: camera_center = -R^T * t
        R_w2c = extrinsic_clip_squeezed[:, :3, :3]  # [N, 3, 3]
        t_w2c = extrinsic_clip_squeezed[:, :3, 3]   # [N, 3]
        original_cam_centers_opencv = -(R_w2c.transpose(0, 2, 1) @ t_w2c[..., np.newaxis]).squeeze(-1)  # [N, 3]
        
        # Coordinate system transformation: 
        # From OpenCV (x-right, y-down, z-forward) to Intuitive (x-right, y-forward, z-up)
        # Transform matrix: [1, 0, 0; 0, 0, 1; 0, -1, 0]
        original_cam_centers = original_cam_centers_opencv.copy()
        original_cam_centers[:, 1] = original_cam_centers_opencv[:, 2]   # z_opencv -> y_intuitive  
        original_cam_centers[:, 2] = -original_cam_centers_opencv[:, 1]  # -y_opencv -> z_intuitive
        
        # Draw camera trajectory
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Draw original data points (opaque)
        ax.scatter(original_cam_centers[:, 0], original_cam_centers[:, 1], original_cam_centers[:, 2], 
                c='blue', s=40, alpha=1.0, label='Original Points', edgecolors='black', linewidth=0.5)

        # Add camera direction arrows
        for i in range(len(original_cam_centers)):
            R_w2c_i = extrinsic_clip_squeezed[i, :3, :3]  # w2c rotation matrix
            # Convert w2c to c2w to calculate camera direction
            R_c2w_opencv = R_w2c_i.T  # c2w rotation matrix (OpenCV system)
            cam_center = original_cam_centers[i]  # Camera center position (intuitive system)
            
            # In OpenCV system, camera forward is +Z direction
            forward_opencv = R_c2w_opencv @ np.array([0, 0, 1])
            
            # Convert forward vector to intuitive system
            forward = np.array([forward_opencv[0], forward_opencv[2], -forward_opencv[1]])
            
            ax.quiver(cam_center[0], cam_center[1], cam_center[2],
                    forward[0], forward[1], forward[2],
                    length=0.05, color='red', normalize=True, alpha=0.8, arrow_length_ratio=0.2)

        # Mark start and end points
        ax.scatter(original_cam_centers[0, 0], original_cam_centers[0, 1], original_cam_centers[0, 2], 
                c='green', s=100, alpha=1.0, label='Start', edgecolors='black', linewidth=1)
        ax.scatter(original_cam_centers[-1, 0], original_cam_centers[-1, 1], original_cam_centers[-1, 2], 
                c='red', s=100, alpha=1.0, label='End', edgecolors='black', linewidth=1)

        ax.set_xlabel('X (Right)')
        ax.set_ylabel('Y (Forward)')  
        ax.set_zlabel('Z (Up)')
        ax.set_title(f'Camera Trajectory Orientation')
        ax.legend()
        plt.tight_layout()
        
        save_filename = extrinsic_clip_path.replace("/", "_").replace(".npy", ".png")
        plt.savefig(os.path.join(output_dir, save_filename), dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
