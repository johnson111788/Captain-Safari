"""
可视化模块 - 轨迹绘制、坏点标注
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非GUI后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def plot_trajectory(
    extrinsics: np.ndarray,
    bad_mask: Optional[np.ndarray] = None,
    repaired_indices: Optional[List[int]] = None,
    output_path: Path = None,
    title: str = "Camera Trajectory",
    dpi: int = 150,
    figsize: tuple = (12, 8)
):
    """
    绘制相机轨迹，标注坏点和修复点
    
    Args:
        extrinsics: (N, 3, 4) 外参
        bad_mask: (N-1,) bool 数组，标记坏过渡
        repaired_indices: 被修复的帧索引列表
        output_path: 输出图片路径
        title: 图标题
        dpi: 分辨率
        figsize: 图大小
    """
    N = len(extrinsics)
    
    # 计算相机中心和前向向量（OpenCV坐标系）
    R_w2c = extrinsics[:, :3, :3]
    t_w2c = extrinsics[:, :3, 3]
    
    # 相机中心：C = -R^T @ t
    centers_opencv = -(R_w2c.transpose(0, 2, 1) @ t_w2c[..., np.newaxis]).squeeze(-1)
    
    # 前向向量：f = R^T @ [0, 0, 1]（OpenCV中相机+Z是前向）
    forward_vecs_opencv = np.array([R_w2c[i].T @ np.array([0, 0, 1]) for i in range(N)])
    
    # 坐标系转换：OpenCV (x-right, y-down, z-forward) -> 直观坐标系 (x-right, y-forward, z-up)
    # 参考 5.viz_camera_trajectory.py 的转换方式
    centers = centers_opencv.copy()
    centers[:, 1] = centers_opencv[:, 2]   # z_opencv -> y_intuitive
    centers[:, 2] = -centers_opencv[:, 1]  # -y_opencv -> z_intuitive
    
    forward_vecs = forward_vecs_opencv.copy()
    forward_vecs[:, 1] = forward_vecs_opencv[:, 2]   # z -> y
    forward_vecs[:, 2] = -forward_vecs_opencv[:, 1]  # -y -> z
    
    # 创建3D图
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹线
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
            'b-', alpha=0.6, linewidth=1, label='Trajectory')
    
    # 标记帧位置
    healthy_mask = np.ones(N, dtype=bool)
    
    if bad_mask is not None:
        # 将过渡bad_mask转换为帧标记
        for i in range(len(bad_mask)):
            if bad_mask[i]:
                healthy_mask[i] = False
                healthy_mask[i + 1] = False
    
    if repaired_indices is not None:
        for idx in repaired_indices:
            healthy_mask[idx] = False  # 单独标记
    
    # 健康帧（蓝色）
    healthy_idx = np.where(healthy_mask)[0]
    if len(healthy_idx) > 0:
        ax.scatter(centers[healthy_idx, 0],
                   centers[healthy_idx, 1],
                   centers[healthy_idx, 2],
                   c='blue', s=20, alpha=0.6, label='Healthy')
    
    # 坏帧（红色）
    bad_idx = np.where(~healthy_mask)[0]
    if repaired_indices is not None:
        bad_idx = np.setdiff1d(bad_idx, repaired_indices)
    
    if len(bad_idx) > 0:
        ax.scatter(centers[bad_idx, 0],
                   centers[bad_idx, 1],
                   centers[bad_idx, 2],
                   c='red', s=50, marker='x', alpha=0.8, label='Bad')
    
    # 修复帧（橙色）
    if repaired_indices is not None and len(repaired_indices) > 0:
        rep_idx = np.array(repaired_indices)
        ax.scatter(centers[rep_idx, 0],
                   centers[rep_idx, 1],
                   centers[rep_idx, 2],
                   c='orange', s=60, marker='o', alpha=0.8, label='Repaired')
    
    # 绘制前向向量（相机朝向箭头）- 参考 5.viz_camera_trajectory.py
    # 每隔几帧绘制，避免过密
    step = max(1, N // 15)
    for i in range(0, N, step):
        # 根据健康状态选择颜色
        if repaired_indices is not None and i in repaired_indices:
            color = 'orange'  # 修复的帧
            alpha = 0.9
        elif healthy_mask[i]:
            color = 'green'  # 健康帧
            alpha = 0.7
        else:
            color = 'red'  # 坏帧
            alpha = 0.8
        
        # 计算前向向量的长度（基于场景规模自适应）
        # 使用中位数而非平均值，避免极端跳变影响箭头大小
        median_dist = np.median(np.linalg.norm(centers[1:] - centers[:-1], axis=1))
        arrow_length = median_dist * 3  # 箭头长度为中位数移动距离的3倍
        
        # 绘制朝向箭头
        ax.quiver(centers[i, 0], centers[i, 1], centers[i, 2],
                  forward_vecs[i, 0] * arrow_length,
                  forward_vecs[i, 1] * arrow_length,
                  forward_vecs[i, 2] * arrow_length,
                  color=color, alpha=alpha, 
                  arrow_length_ratio=0.3,
                  linewidth=1.5)
    
    # 标记起点和终点（参考 5.viz_camera_trajectory.py）
    ax.scatter(centers[0, 0], centers[0, 1], centers[0, 2],
               c='lime', s=150, alpha=1.0, 
               edgecolors='black', linewidth=2, label='Start', zorder=10)
    ax.scatter(centers[-1, 0], centers[-1, 1], centers[-1, 2],
               c='darkred', s=150, alpha=1.0,
               edgecolors='black', linewidth=2, label='End', zorder=10)
    
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Forward)')
    ax.set_zlabel('Z (Up)')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=9)
    
    # 设置相同的坐标轴比例（以最大轴范围为基准）
    x_range = centers[:, 0].max() - centers[:, 0].min()
    y_range = centers[:, 1].max() - centers[:, 1].min()
    z_range = centers[:, 2].max() - centers[:, 2].min()
    max_range = max(x_range, y_range, z_range) / 2.0
    
    x_mid = (centers[:, 0].max() + centers[:, 0].min()) * 0.5
    y_mid = (centers[:, 1].max() + centers[:, 1].min()) * 0.5
    z_mid = (centers[:, 2].max() + centers[:, 2].min()) * 0.5
    
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        # logger.info(f"轨迹图已保存至: {output_path}")
    
    plt.close(fig)


def plot_metrics(
    transition_metrics: List[dict],
    output_path: Path,
    dpi: int = 150,
    figsize: tuple = (14, 10)
):
    """
    绘制过渡指标图表
    
    Args:
        transition_metrics: 过渡指标列表
        output_path: 输出路径
        dpi: 分辨率
        figsize: 图大小
    """
    N = len(transition_metrics)
    if N == 0:
        return
    
    indices = np.arange(N)
    
    # 提取指标
    inlier_counts = [m.get('inlier_count', 0) for m in transition_metrics]
    inlier_ratios = [m.get('inlier_ratio', 0) for m in transition_metrics]
    epi_medians = [m.get('epi_median', np.nan) for m in transition_metrics]
    trans_dists = [m.get('translation_distance', np.nan) for m in transition_metrics]
    rot_angles = [m.get('rotation_angle', np.nan) for m in transition_metrics]
    
    fig, axes = plt.subplots(3, 2, figsize=figsize, dpi=dpi)
    
    # 1. Inlier Count
    ax = axes[0, 0]
    ax.plot(indices, inlier_counts, 'b-', linewidth=1)
    ax.set_ylabel('Inlier Count')
    ax.set_title('RANSAC Inlier Count')
    ax.grid(True, alpha=0.3)
    
    # 2. Inlier Ratio
    ax = axes[0, 1]
    ax.plot(indices, inlier_ratios, 'g-', linewidth=1)
    ax.set_ylabel('Inlier Ratio')
    ax.set_title('Inlier Ratio')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 3. Epipolar Error Median
    ax = axes[1, 0]
    valid = ~np.isnan(epi_medians)
    if valid.any():
        ax.plot(indices[valid], np.array(epi_medians)[valid], 'r-', linewidth=1)
    ax.set_ylabel('Median Epi Error (px)')
    ax.set_title('Epipolar Error Median')
    ax.grid(True, alpha=0.3)
    
    # 4. Translation Distance
    ax = axes[1, 1]
    valid = ~np.isnan(trans_dists)
    if valid.any():
        ax.plot(indices[valid], np.array(trans_dists)[valid], 'm-', linewidth=1)
    ax.set_ylabel('Distance')
    ax.set_title('Translation Distance')
    ax.grid(True, alpha=0.3)
    
    # 5. Rotation Angle
    ax = axes[2, 0]
    valid = ~np.isnan(rot_angles)
    if valid.any():
        ax.plot(indices[valid], np.array(rot_angles)[valid], 'c-', linewidth=1)
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Rotation Angle')
    ax.set_xlabel('Transition Index')
    ax.grid(True, alpha=0.3)
    
    # 6. Bad Transitions
    ax = axes[2, 1]
    bad_flags = [m.get('is_bad', False) for m in transition_metrics]
    ax.scatter(indices, bad_flags, c=['red' if b else 'green' for b in bad_flags],
               s=10, alpha=0.5)
    ax.set_ylabel('Is Bad')
    ax.set_title('Bad Transition Flags')
    ax.set_xlabel('Transition Index')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    # logger.info(f"指标图已保存至: {output_path}")
    plt.close(fig)

