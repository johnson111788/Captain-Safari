#!/usr/bin/env python3
"""
插值相机参数脚本
将 hloc 目录中不到60帧的 extrinsic/intrinsic 插值到60帧
如果已经是60帧，则直接复制
"""

import numpy as np
import torch
import os
import sys
from pathlib import Path
from tqdm import tqdm
import shutil

# 导入插值函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffsynth.models.wan_video_camera_controller import (
    build_time_axes,
    interpolate_extrinsics_w2c_cuda,
    interpolate_intrinsics_cuda
)


def interpolate_camera_params(extrinsic_path, intrinsic_path, target_frames=20, fps_in=4.0, fps_out=24.0):
    """
    插值相机参数到目标帧数
    
    Args:
        extrinsic_path: 外参数文件路径
        intrinsic_path: 内参数文件路径
        target_frames: 目标帧数（默认20）
        fps_in: 输入帧率（默认4.0）
        fps_out: 输出帧率（默认24.0）
    
    Returns:
        interpolated_extrinsic: 插值后的外参数 (target_frames, 1, 3, 4)
        interpolated_intrinsic: 插值后的内参数 (target_frames, 1, 3, 3)
    """
    # 加载数据
    extrinsic = np.load(extrinsic_path)  # (n_frames, 1, 3, 4)
    intrinsic = np.load(intrinsic_path)[:, np.newaxis, :, :]  # (n_frames, 1, 3, 3)
    
    n_frames = extrinsic.shape[0]
    
    # 如果已经是目标帧数，直接返回
    if n_frames == target_frames:
        return extrinsic, intrinsic
    
    # 转换为 torch tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    extrinsic_tensor = torch.from_numpy(extrinsic).to(device=device, dtype=dtype)
    intrinsic_tensor = torch.from_numpy(intrinsic).to(device=device, dtype=dtype)
    
    # 转换为4x4矩阵
    if extrinsic_tensor.shape[-2:] == (3, 4):
        B, N, _, _ = extrinsic_tensor.shape
        extrinsic_4x4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
        extrinsic_4x4[:, :, :3, :4] = extrinsic_tensor
        extrinsic_tensor = extrinsic_4x4
    
    # 构建时间轴
    t_in, t_out = build_time_axes(n_frames, fps_in, target_frames, fps_out, device=device, dtype=dtype)
    
    # 插值
    E4 = interpolate_extrinsics_w2c_cuda(extrinsic_tensor, t_in, t_out)
    Kq = interpolate_intrinsics_cuda(intrinsic_tensor, t_in, t_out)
    
    # 转换回 (target_frames, 1, 3, 4) 和 (target_frames, 1, 3, 3) 格式
    E4_out = E4[0, :, :3, :4].cpu().numpy()  # (target_frames, 1, 3, 4) .unsqueeze(1)
    Kq_out = Kq[0, :, :, :].cpu().numpy()    # (target_frames, 1, 3, 3) .unsqueeze(1)
    
    return E4_out, Kq_out


def process_all_files(input_dir, output_dir, target_frames=20):
    """
    处理所有相机参数文件
    
    Args:
        input_dir: 输入目录 (hloc)
        output_dir: 输出目录 (hloc20)
        target_frames: 目标帧数（默认20）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 extrinsic 文件
    extrinsic_files = sorted(input_path.glob('*_extrinsic.npy'))
    
    print(f"找到 {len(extrinsic_files)} 个 extrinsic 文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标帧数: {target_frames}")
    print()
    
    stats = {
        'copied': 0,
        'interpolated': 0,
        'errors': 0
    }
    
    # 处理每个文件
    for extrinsic_file in tqdm(extrinsic_files, desc="处理中"):
        try:
            # 构建对应的 intrinsic 文件路径
            base_name = extrinsic_file.stem.replace('_extrinsic', '')
            intrinsic_file = input_path / f"{base_name}_intrinsic.npy"
            
            if not intrinsic_file.exists():
                print(f"警告: 找不到对应的 intrinsic 文件: {intrinsic_file}")
                stats['errors'] += 1
                continue
            
            # 加载并检查帧数
            extrinsic = np.load(extrinsic_file)
            n_frames = extrinsic.shape[0]
            
            # 输出文件路径
            out_extrinsic = output_path / extrinsic_file.name
            out_intrinsic = output_path / intrinsic_file.name
            
            if n_frames == target_frames:
                # 直接复制
                shutil.copy2(extrinsic_file, out_extrinsic)
                shutil.copy2(intrinsic_file, out_intrinsic)
                stats['copied'] += 1
            else:
                # 插值
                extrinsic_interp, intrinsic_interp = interpolate_camera_params(
                    extrinsic_file, intrinsic_file, target_frames=target_frames
                )
                print(extrinsic_interp.shape, intrinsic_interp.shape)
                np.save(out_extrinsic, extrinsic_interp)
                np.save(out_intrinsic, intrinsic_interp)
                stats['interpolated'] += 1
                
        except Exception as e:
            print(f"\n错误处理文件 {extrinsic_file.name}: {e}")
            stats['errors'] += 1
            continue
    
    print("\n" + "="*60)
    print("处理完成!")
    print(f"直接复制: {stats['copied']} 个文件对")
    print(f"插值处理: {stats['interpolated']} 个文件对")
    print(f"错误: {stats['errors']} 个文件对")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interpolate camera parameters to a target number of frames"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing camera parameters"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for interpolated camera parameters"
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=60,
        help="Target number of frames (default: 60)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Interpolating camera parameters")
    print("=" * 80)
    
    # 处理所有文件
    process_all_files(args.input_dir, args.output_dir, target_frames=args.target_frames)

