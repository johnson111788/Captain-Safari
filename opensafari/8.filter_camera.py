#!/usr/bin/env python3
"""
Trajectory filtering script.
Calculates and filters camera direction changes for each sample in metadata.csv.
Ensures a more uniform data distribution and prevents an overrepresentation of straight flight trajectories.
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

def compute_camera_forward(extrinsic):
    """
    Compute the forward direction vector of the camera
    
    Args:
        extrinsic: (3, 4) or (4, 4) extrinsic matrix
        
    Returns:
        forward: (3,) forward direction vector (unit vector)
    """
    # Extract rotation matrix (world to camera)
    R_w2c = extrinsic[:3, :3]
    
    # Convert to camera to world
    R_c2w_opencv = R_w2c.T
    
    # In OpenCV coordinate system, camera forward is +Z direction
    forward_opencv = R_c2w_opencv @ np.array([0, 0, 1])
    
    # Normalize
    forward = forward_opencv / np.linalg.norm(forward_opencv)
    
    return forward

def compute_angle_between_vectors(v1, v2):
    """
    Compute the angle between two vectors (in degrees)
    
    Args:
        v1, v2: (3,) vectors
        
    Returns:
        angle: angle (degrees)
    """
    # Ensure vectors are normalized
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate dot product
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    
    # Calculate angle
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def compute_trajectory_angle_change(extrinsic_clip):
    """
    Compute the average angle change of a trajectory
    
    Args:
        extrinsic_clip: (20, 3, 4) or (20, 4, 4) extrinsic sequence
        
    Returns:
        avg_angle: average angle change between frame 1-10 and frame 10-20
    """
    # Compute forward direction of the 1st frame (index 0)
    forward_1 = compute_camera_forward(extrinsic_clip[0])
    
    # Compute forward direction of the 10th frame (index 9)
    forward_10 = compute_camera_forward(extrinsic_clip[9])
    
    # Compute forward direction of the 20th frame (index 19)
    forward_20 = compute_camera_forward(extrinsic_clip[19])
    
    # Calculate angle differences
    angle_1_10 = compute_angle_between_vectors(forward_1, forward_10)
    angle_10_20 = compute_angle_between_vectors(forward_10, forward_20)
    
    # Calculate average
    avg_angle = (angle_1_10 + angle_10_20) / 2.0
    
    return avg_angle

def main():
    parser = argparse.ArgumentParser(description="Filter trajectories based on angle changes to ensure diversity.")
    parser.add_argument("--input-csv", type=str, default="metadata.csv", help="Input metadata CSV file path")
    parser.add_argument("--output-csv", type=str, default="metadata.filtered-trajectory.csv", help="Output metadata CSV file path")
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    output_csv_path = Path(args.output_csv)
    
    print(f"Reading CSV file: {csv_path}")
    if not csv_path.exists():
        print(f"Error: Input CSV {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    
    # Store angle change for each sample
    angle_changes = []
    valid_indices = []
    
    print("\nCalculating angle changes for each sample...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Get extrinsic_clip file path
            extrinsic_clip_path = row['extrinsic_clip']
            
            # Check if file exists
            if not os.path.exists(extrinsic_clip_path):
                print(f"Warning: File not found {extrinsic_clip_path}")
                continue
            
            # Load extrinsic_clip
            extrinsic_clip = np.load(extrinsic_clip_path)
            
            # Check shape
            if extrinsic_clip.shape[0] != 20:
                print(f"Warning: Incorrect shape {extrinsic_clip.shape} at index {idx}")
                continue
            
            # Compute angle change
            avg_angle = compute_trajectory_angle_change(extrinsic_clip)
            
            angle_changes.append(avg_angle)
            valid_indices.append(idx)
            
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue
    
    angle_changes = np.array(angle_changes)
    print(f"\nSuccessfully processed samples: {len(angle_changes)}")
    print(f"Angle change statistics:")
    print(f"  Min: {angle_changes.min():.2f} degrees")
    print(f"  Max: {angle_changes.max():.2f} degrees")
    print(f"  Mean: {angle_changes.mean():.2f} degrees")
    print(f"  Median: {np.median(angle_changes):.2f} degrees")
    
    # Distribution statistics (every 10 degrees)
    print("\n" + "="*60)
    print("Angle change distribution statistics (per 10 degrees):")
    print("="*60)
    
    bins = range(0, 130, 10)  # 0-10, 10-20, ..., 110-120, 120+
    bin_counts = defaultdict(int)
    bin_indices = defaultdict(list)
    
    for i, angle in enumerate(angle_changes):
        bin_idx = int(angle // 10) * 10
        if bin_idx > 120:
            bin_idx = 120
        bin_counts[bin_idx] += 1
        bin_indices[bin_idx].append(valid_indices[i])
    
    for bin_start in sorted(bin_counts.keys()):
        bin_end = bin_start + 10 if bin_start < 120 else 180
        count = bin_counts[bin_start]
        percentage = count / len(angle_changes) * 100
        print(f"{bin_start:3d}-{bin_end:3d} degrees: {count:6d} samples ({percentage:5.2f}%)")
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    plt.hist(angle_changes, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Average Angle Change (degrees)')
    plt.ylabel('Number of Samples')
    plt.title('Trajectory Angle Change Distribution')
    plt.grid(True, alpha=0.3)
    out_img = 'trajectory_angle_distribution_before.png'
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    print(f"\nDistribution plot saved to: {out_img}")
    
    # Uniform sampling
    print("\n" + "="*60)
    print("Starting uniform sampling...")
    print("="*60)
    
    # Find bins with minimum threshold (ignore bins with very few samples)
    valid_bins = {k: v for k, v in bin_counts.items() if v >= 10}  # Consider bins with at least 10 samples
    
    if len(valid_bins) == 0:
        print("Error: Not enough valid bins")
        return
    
    # Calculate target number of samples (use median or smaller value to keep more data)
    target_samples_per_bin = int(np.median(list(valid_bins.values())))
    print(f"Target number of samples per bin: {target_samples_per_bin}")
    
    # Sample from each bin
    sampled_indices = []
    
    for bin_start in sorted(bin_counts.keys()):
        bin_data_indices = bin_indices[bin_start]
        count = len(bin_data_indices)
        
        if count == 0:
            continue
        
        # If samples are fewer than target, keep all
        if count <= target_samples_per_bin:
            sampled_indices.extend(bin_data_indices)
            sampled_count = count
        else:
            # Random sampling
            sampled = np.random.choice(bin_data_indices, size=target_samples_per_bin, replace=False)
            sampled_indices.extend(sampled.tolist())
            sampled_count = target_samples_per_bin
        
        bin_end = bin_start + 10 if bin_start < 120 else 180
        print(f"{bin_start:3d}-{bin_end:3d} degrees: {count:6d} -> {sampled_count:6d} samples")
    
    # Create filtered dataset
    sampled_indices = sorted(sampled_indices)
    df_filtered = df.iloc[sampled_indices].copy()
    
    # Add angle change column
    angle_map = dict(zip(valid_indices, angle_changes))
    df_filtered['trajectory_angle_change'] = df_filtered.index.map(angle_map)
    
    # Reset index
    df_filtered = df_filtered.reset_index(drop=True)
    
    print(f"\nSamples after filtering: {len(df_filtered)} (Original: {len(df)})")
    print(f"Retention ratio: {len(df_filtered)/len(df)*100:.2f}%")
    
    # Verify distribution after filtering
    print("\n" + "="*60)
    print("Angle change distribution after filtering:")
    print("="*60)
    
    filtered_angles = df_filtered['trajectory_angle_change'].values
    bin_counts_after = defaultdict(int)
    
    for angle in filtered_angles:
        bin_idx = int(angle // 10) * 10
        if bin_idx > 120:
            bin_idx = 120
        bin_counts_after[bin_idx] += 1
    
    for bin_start in sorted(bin_counts_after.keys()):
        bin_end = bin_start + 10 if bin_start < 120 else 180
        count = bin_counts_after[bin_start]
        percentage = count / len(filtered_angles) * 100
        print(f"{bin_start:3d}-{bin_end:3d} degrees: {count:6d} samples ({percentage:5.2f}%)")
    
    # Plot distribution after filtering
    plt.figure(figsize=(12, 6))
    plt.hist(filtered_angles, bins=50, edgecolor='black', alpha=0.7, color='green')
    plt.xlabel('Average Angle Change (degrees)')
    plt.ylabel('Number of Samples')
    plt.title('Trajectory Angle Change Distribution (Filtered)')
    plt.grid(True, alpha=0.3)
    out_img_after = 'trajectory_angle_distribution_after.png'
    plt.savefig(out_img_after, dpi=150, bbox_inches='tight')
    print(f"\nFiltered distribution plot saved to: {out_img_after}")
    
    # Save filtered CSV
    df_filtered.to_csv(output_csv_path, index=False)
    print(f"\nFiltered data saved to: {output_csv_path}")
    
    print("\nComplete!")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
