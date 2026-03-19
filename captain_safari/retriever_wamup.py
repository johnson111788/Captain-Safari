#!/usr/bin/env python3
"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="localhost" \
         --master_port=12345 \
         retriever_wamup.py
"""

import os
import wandb
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffsynth.models.wan_video_camera_controller import extri_intri_to_pose_encoding
from diffsynth.models.wan_video_dit import (
    precompute_freqs_cis, 
    precompute_freqs_cis_3d,
    SelfAttention,
    CrossAttention
)


# ===================== hyperparameters (top-level constants) =====================
DEFAULT_MEM_ENC_LAYERS = 1      # Memory encoder layers
DEFAULT_JOINT_LAYERS = 1        # Joint attention layers  
DEFAULT_RETRIEVAL_BLOCKS = 1    # Total retrieval blocks number
JOINT_DROPOUT = 0.10
JOINT_FFN_MULT = 4


def set_seed(seed: int):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NPROCS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        print("No distributed environment detected, using single GPU training")
        return False, 0, 1, 0

    # check CUDA availability
    if not torch.cuda.is_available():
        print(f"Error: CUDA not available in distributed environment!")
        print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        print("Please ensure:")
        print("  1. System has available GPUs")
        print("  2. CUDA driver is correctly installed")
        print("  3. If distributed training is not needed, do not use torchrun")
        return False, 0, 1, 0
    
    # check local_rank validity
    if local_rank >= torch.cuda.device_count():
        print(f"Error: local_rank {local_rank} exceeds available GPU count {torch.cuda.device_count()}")
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", 
        rank=rank, 
        world_size=world_size
    )
    torch.distributed.barrier()
    
    return True, rank, world_size, local_rank


def cleanup_distributed():
    """clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """check if is main process"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size():
    """get world size"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_metric(tensor, world_size):
    """average metrics across all processes"""
    if world_size == 1:
        return tensor
    
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor


def compute_hit_metrics(logits_agg, targets):
    """
    calculate Hit@1, Hit@±1, Hit@±2 metrics
    
    Args:
        logits_agg: [B, T] aggregated logits
        targets: [B] true labels
    
    Returns:
        dict: dictionary containing various hit metrics
    """
    pred_idx = logits_agg.argmax(dim=-1)  # [B]
    
    # Hit@1: exact match
    hit_at_1 = (pred_idx == targets).float().mean()
    
    # Hit@±1: |pred - target| <= 1
    diff_abs = torch.abs(pred_idx - targets)
    hit_at_pm1 = (diff_abs <= 1).float().mean()
    
    # Hit@±2: |pred - target| <= 2  
    hit_at_pm2 = (diff_abs <= 2).float().mean()
    
    return {
        'hit_at_1': hit_at_1.item(),
        'hit_at_pm1': hit_at_pm1.item(), 
        'hit_at_pm2': hit_at_pm2.item()
    }


def save_checkpoint(model, optimizer, scheduler, step, args, input_dim, save_dir, keep_checkpoints=3):
    """save checkpoint and manage file number"""
    import glob
    import re
    
    os.makedirs(save_dir, exist_ok=True)
    
    # get actual model (if DDP, then get module)
    model_to_save = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'args': vars(args),
        'model_config': {
            'input_dim': input_dim,
            'model_dim': args.model_dim,
            'num_heads': args.num_heads,
            'retrieval_blocks': args.retrieval_blocks,
            'mem_enc_layers': args.mem_enc_layers,
            'joint_layers': args.joint_layers,
            'mse_weight': args.mse_weight,
            'rope_mode': args.rope_mode
        },
        'training_config': {
            'use_bf16': getattr(args, 'use_bf16', False),
            'bf16_full_eval': getattr(args, 'bf16_full_eval', False),
            'model_dtype': str(next(model_to_save.parameters()).dtype)
        }
    }
    
    # save new checkpoint
    checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # clean up old checkpoints (keep recent N)
    if keep_checkpoints > 0:
        checkpoint_pattern = os.path.join(save_dir, "checkpoint_step_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) > keep_checkpoints:
            # sort by step
            def extract_step(filename):
                match = re.search(r'checkpoint_step_(\d+)\.pt', filename)
                return int(match.group(1)) if match else 0
            
            checkpoint_files.sort(key=extract_step)
            
            # delete old checkpoints
            for old_checkpoint in checkpoint_files[:-keep_checkpoints]:
                try:
                    os.remove(old_checkpoint)
                    print(f"🗑️  Deleted old checkpoint: {os.path.basename(old_checkpoint)}")
                except Exception as e:
                    print(f"⚠️  Cannot delete {old_checkpoint}: {e}")
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device=None, strict=True):
    """load checkpoint and correctly handle BF16 model"""
    checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu')
    
    # check saved model precision
    training_config = checkpoint.get('training_config', {})
    saved_use_bf16 = training_config.get('use_bf16', False)
    saved_dtype = training_config.get('model_dtype', 'torch.float32')
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Saved model dtype: {saved_dtype}")
    print(f"Saved BF16 setting: {saved_use_bf16}")
    
    # get actual model (if DDP, then get module)
    model_to_load = model.module if hasattr(model, 'module') else model
    
    # load model state
    try:
        model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print("✅ Model weights loaded successfully")
    except Exception as e:
        print(f"❌ Model weights loaded failed: {e}")
        raise
    
    # load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state loaded successfully")
        except Exception as e:
            print(f"⚠️  Optimizer state loaded failed: {e}")
    
    # load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ Learning rate scheduler state loaded successfully")
        except Exception as e:
            print(f"⚠️  Learning rate scheduler state loaded failed: {e}")
    
    # return step and configuration information
    return {
        'step': checkpoint.get('step', 0),
        'model_config': checkpoint.get('model_config', {}),
        'training_config': training_config,
        'args': checkpoint.get('args', {})
    }

def convert_to_local_coordinates(extrinsics):
    """
    convert extrinsics from global coordinate system to local coordinate system (first frame is origin)
    
    Args:
        extrinsics: (N, 3, 4) extrinsics in global coordinate system (w2c format)
    
    Returns:
        extrinsics_local: (N, 3, 4) extrinsics in local coordinate system
    """
    N = extrinsics.shape[0]
    
    # extrinsics of first frame
    R0 = extrinsics[0, :, :3]  # rotation matrix of first frame
    t0 = extrinsics[0, :, 3]    # translation vector of first frame
    
    # calculate camera center of first frame (world coordinate system)
    C0 = -(R0.T @ t0)
    
    # c2w matrix of first frame
    R0_c2w = R0.T
    t0_c2w = C0
    
    # build c2w 4x4 matrix of first frame
    T0_c2w = np.eye(4)
    T0_c2w[:3, :3] = R0_c2w
    T0_c2w[:3, 3] = t0_c2w
    
    # its inverse matrix (w2c) is the reference transformation
    T0_w2c = np.linalg.inv(T0_c2w)
    
    # convert all frames to local coordinate system
    extrinsics_local = np.zeros_like(extrinsics)
    
    for i in range(N):
        # extrinsics of current frame
        Ri = extrinsics[i, :, :3]
        ti = extrinsics[i, :, 3]
        
        # build 4x4 w2c matrix
        Ti_w2c = np.eye(4)
        Ti_w2c[:3, :3] = Ri
        Ti_w2c[:3, 3] = ti
        
        # convert to local coordinate system: T_local = T0_w2c @ T_global_c2w @ T_i_w2c
        # simplify: T_local = T0_w2c @ inv(T_i_w2c)
        Ti_c2w = np.linalg.inv(Ti_w2c)
        Ti_local_c2w = T0_w2c @ Ti_c2w
        Ti_local_w2c = np.linalg.inv(Ti_local_c2w)
        
        # extract 3x4 extrinsics
        extrinsics_local[i] = Ti_local_w2c[:3, :]
    
    return extrinsics_local

class JointSelfAttention(SelfAttention):
    """
    Joint Self-Attention based on DiffSynth SelfAttention, supporting RoPE
    
    Features:
    - directly inherit DiffSynth SelfAttention module
    - support joint attention with Q without RoPE, K with 3D RoPE
    - keep all optimizations and standard implementations of DiffSynth
    """
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        # directly use parent class initialization
        super().__init__(dim, num_heads, eps)

    def forward(self, x_joint, freqs_cis=None, attn_mask=None):
        """
        override parent class forward method to support Joint RoPE logic
        
        x_joint: [B, L, D] where L = 1 + memory_tokens (Q at index 0, memory at indices 1..L)
        freqs_cis: [L, 1, head_dim] for 3D RoPE, first row is identity for Q
        attn_mask: [B, L] attention mask
        """
        if freqs_cis is not None:
            # call parent class forward method, using provided freqs
            return super().forward(x_joint, freqs_cis, attn_mask=attn_mask)
        else:
            # no RoPE, perform standard attention
            B, L, D = x_joint.shape
            # FIX: last dimension of freqs_identity should be head_dim // 2 (complex)
            freqs_identity = torch.ones((L, 1, self.head_dim // 2), device=x_joint.device, dtype=torch.complex64)
            return super().forward(x_joint, freqs_identity, attn_mask=attn_mask)


class UnifiedJointBlock(nn.Module):
    """
    Unified Joint Transformer Block, memEnc and jointAttn share the same implementation
    
    Features:
    - use JointSelfAttention to support 3D RoPE
    - do not include CrossAttention (removed as requested)
    - use FFN structure of DiffSynth
    """
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, ffn_mult: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_mult = ffn_mult
        
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        
        self.joint_self_attn = JointSelfAttention(dim, num_heads, eps)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_mult * dim, dim)
        )
    
    def forward(self, x_joint, freqs_cis=None, attn_mask=None):
        """
        standard Pre-Norm Transformer Block forward pass
        """
        x_joint = x_joint + self.joint_self_attn(self.norm1(x_joint), freqs_cis, attn_mask=attn_mask)
        x_joint = x_joint + self.ffn(self.norm2(x_joint))
        return x_joint


class RetrievalBlock(nn.Module):
    """
    Complete Retrieval Block, wrapping memEnc + jointAttn + crossAttn
    
    This block contains:
    1. Memory Encoder: perform joint self-attention on (pose_t, memory_t) for each time step
    2. Query-Output Joint Attention: perform joint self-attention on (q_tok, Q_out)
    3. Cross Attention: jointAttn output cross-attend to memEnc output
    """
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, 
                 mem_enc_layers: int = 2, joint_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mem_enc_layers = mem_enc_layers
        self.joint_layers = joint_layers
        
        # Memory Encoder: multiple UnifiedJointBlock
        self.mem_enc_blocks = nn.ModuleList([
            UnifiedJointBlock(dim, num_heads, eps, JOINT_FFN_MULT) 
            for _ in range(mem_enc_layers)
        ])
        
        # Query-Output Joint Attention: multiple UnifiedJointBlock  
        self.joint_attn_blocks = nn.ModuleList([
            UnifiedJointBlock(dim, num_heads, eps, JOINT_FFN_MULT)
            for _ in range(joint_layers)
        ])
        
        # Cross Attention: reuse implementation of wan_video_dit
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=False)
        self.norm_cross = nn.LayerNorm(dim, eps=eps)
        
    def forward(self, q_tok, pose_tokens, memory_tokens, 
                query_freqs=None, memory_freqs=None, learnable_query=None, key_mask=None):
        """
        Args:
            q_tok: [B, 1, dim] current pose token (query)
            pose_tokens: [B, T, dim] T time steps of pose tokens
            memory_tokens: [B, T, 4*782, dim] T time steps of memory tokens
            query_freqs: [1+4*782, 1, head_dim] RoPE freqs for query+learnable_query
            memory_freqs: [T*(1+4*782), 1, head_dim] RoPE freqs for memory sequence
            learnable_query: [B, 4*782, dim] learnable output queries
            key_mask: [B, T] mask for valid key frames (1=valid, 0=padding) (1=valid, 0=padding)
            
        Returns:
            joint_output: [B, 1+4*782, dim] contains q_out and memory_pred
        """
        B, T = pose_tokens.shape[:2]
        
        # 1. Memory Encoding: perform joint self-attention on (pose_t, memory_t) for each time step
        encoded_memory_list = []
        for t in range(T):
            # process each time step separately (including padding positions, cross attention will use mask to mask)
            pose_t = pose_tokens[:, t:t+1, :]  # [B, 1, dim]
            memory_t = memory_tokens[:, t, :, :]  # [B, 4*782, dim]
            joint_t = torch.cat([pose_t, memory_t], dim=1)  # [B, 1+4*782, dim]
            
            # build freqs for this time step (if memory_freqs is provided)
            if memory_freqs is not None:
                # extract freqs for this time step: [1+4*782, 1, head_dim]
                start_idx = t * (1 + 4*782)
                end_idx = (t + 1) * (1 + 4*782)
                freqs_t = memory_freqs[start_idx:end_idx]  # [1+4*782, 1, head_dim]
            else:
                freqs_t = None
            
            # pass memory encoder blocks
            # for self-attention, tokens within the same time step do not need mask (all are valid)
            for block in self.mem_enc_blocks:
                joint_t = block(joint_t, freqs_t, attn_mask=None)
                
            encoded_memory_list.append(joint_t)
        
        # merge all time steps: [B, T*(1+4*782), dim]
        encoded_memory = torch.cat(encoded_memory_list, dim=1)
        
        # build attention mask: [B, T*(1+4*782)]
        # each time step has (1+4*782) tokens
        if key_mask is not None:
            # expand mask to all tokens within each time step
            attn_mask = key_mask.unsqueeze(-1).expand(-1, -1, 1 + 4*782)  # [B, T, 1+4*782]
            attn_mask = attn_mask.reshape(B, -1)  # [B, T*(1+4*782)]
        else:
            attn_mask = None
        
        # 2. Query-Output Joint Attention: perform joint self-attention on (q_tok, Q_out)
        if learnable_query is None:
            raise ValueError("learnable_query is required for RetrievalBlock")
            
        joint_query = torch.cat([q_tok, learnable_query], dim=1)  # [B, 1+4*782, dim]
        
        # pass joint attention blocks
        for block in self.joint_attn_blocks:
            joint_query = block(joint_query, query_freqs)

        # 3. Cross Attention: joint_query attend to encoded_memory
        # use attn_mask to mask padding key frames
        joint_output = joint_query + self.cross_attn(
            self.norm_cross(joint_query), encoded_memory, attn_mask=attn_mask
        )
        
        # 4. extract encoded_pose for CE loss: K
        # encoded_memory: [B, T*(1+4*782), dim] -> [B, T, 1+4*782, dim]
        encoded_memory_reshaped = encoded_memory.view(B, T, 1+4*782, self.dim)
        encoded_pose = encoded_memory_reshaped[:, :, 0, :]  # [B, T, dim] - pose token for each time step
        
        return joint_output, encoded_pose


def build_3d_freqs_unified(T, L, H, W, f_freqs, h_freqs, w_freqs, include_pose_token=True, use_time_encoding=True):
    """
    Unified 3D RoPE build function, handling 3D position encoding of memory tokens
    
    Reference implementation of train_retriever.py, supporting:
    - T=20 time dimension
    - L=4 layer dimension (4 layers of streamvggt)
    - H=21, W=37 spatial dimension
    - 1+4 camera/register special tokens (no spatial coordinates)
    
    Args:
        T: time dimension (20, for query freqs set to 1)
        L: layer dimension (4)  
        H, W: spatial dimension (21, 37)
        f_freqs, h_freqs, w_freqs: 3D RoPE three dimensions frequencies
        include_pose_token: whether to include pose token at the beginning of each time step
        use_time_encoding: whether to use time encoding (False for query freqs, following pose_retrieval_rope1d_joint.py)
    
    Returns:
        freqs: [T*(1+L*782), 1, head_dim] if include_pose_token else [T*L*782, 1, head_dim]
    """
    freqs_list = []
    
    for t in range(T):
        if include_pose_token:
            # pose token at the beginning of each time step
            if use_time_encoding:
                # Memory tokens: use actual time encoding
                f_idx = min(t, len(f_freqs) - 1)
                f_freq = f_freqs[f_idx:f_idx+1]  # use frequency of corresponding time step
            else:
                # Query token: no time encoding (following pose_retrieval_rope1d_joint.py)
                f_freq = f_freqs[0:1]  # time encoding set to 0
            
            h_freq = h_freqs[0:1]  # spatial position set to 0 (special token)
            w_freq = w_freqs[0:1]  # spatial position set to 0 (special token)
            combined_freq = torch.cat([f_freq, h_freq, w_freq], dim=-1)
            freqs_list.append(combined_freq)
        
        # next is the L*782 tokens of this time step
        for l in range(L):
            # Special tokens (camera + register): time encoding according to use_time_encoding, spatial position set to 0
            for _ in range(5):  # 1 camera + 4 register
                if use_time_encoding:
                    # Memory tokens: use actual time encoding
                    f_idx = min(t, len(f_freqs) - 1)
                    f_freq = f_freqs[f_idx:f_idx+1]  # same time encoding as pose token
                else:
                    # Query tokens: no time encoding
                    f_freq = f_freqs[0:1]  # time encoding set to 0
                
                h_freq = h_freqs[0:1]  # special token spatial position set to 0
                w_freq = w_freqs[0:1]  # special token spatial position set to 0  
                combined_freq = torch.cat([f_freq, h_freq, w_freq], dim=-1)
                freqs_list.append(combined_freq)
            
            # Image tokens: time encoding according to use_time_encoding, using actual spatial position
            for h in range(H):
                for w in range(W):
                    if use_time_encoding:
                        # Memory tokens: use actual time encoding
                        f_idx = min(t, len(f_freqs) - 1)
                        f_freq = f_freqs[f_idx:f_idx+1]  # same time encoding as pose token
                    else:
                        # Query tokens: no time encoding, but keep spatial encoding
                        f_freq = f_freqs[0:1]  # time encoding set to 0
                    
                    h_idx = min(h, len(h_freqs) - 1) 
                    w_idx = min(w, len(w_freqs) - 1)
                    h_freq = h_freqs[h_idx:h_idx+1]
                    w_freq = w_freqs[w_idx:w_idx+1]
                    combined_freq = torch.cat([f_freq, h_freq, w_freq], dim=-1)
                    freqs_list.append(combined_freq)
    
    return torch.cat(freqs_list, dim=0)


# ===================== Dataset 類 =====================
class MemoryRetrievalDataset(Dataset):
    """Memory Retrieval dataset, handling memory_clip data, supporting pre-loading"""
    
    def __init__(self, metadata_csv: str, data_dir: str):
        import pandas as pd
        
        self.data_dir = data_dir
        
        # read sample list from CSV file
        try:
            df = pd.read_csv(metadata_csv)
        except FileNotFoundError:
            raise FileNotFoundError(f"metadata file not found: {metadata_csv}")
        except Exception as e:
            raise ValueError(f"failed to read metadata file: {e}")
        
        # verify required columns exist
        required_columns = ['memory_target', 'memory', 'extrinsic_query', 'intrinsic_query', 'extrinsic_key', 'intrinsic_key']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV file missing required column: {col}")
        
        # build sample list and verify file existence
        self.samples = []
        for _, row in df.iterrows():
            memory_target_file = os.path.join(data_dir, row['memory_target'])
            memory_key_file = os.path.join(data_dir, row['memory'])
            ext_query_file = os.path.join(data_dir, row['extrinsic_query'])
            int_query_file = os.path.join(data_dir, row['intrinsic_query'])
            ext_key_file = os.path.join(data_dir, row['extrinsic_key'])
            int_key_file = os.path.join(data_dir, row['intrinsic_key'])
            
            self.samples.append((memory_target_file, memory_key_file, ext_query_file, int_query_file, ext_key_file, int_key_file))
            
        if len(self.samples) == 0:
            raise ValueError(f"no valid data samples found")
        
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        memory_file, memory_key_file, ext_query_file, int_query_file, ext_key_file, int_key_file = self.samples[idx]
        
        # load data
        key_memory_clip = torch.from_numpy(np.load(memory_key_file)).float()  # [T_key, 4, 782, 1024]
        key_extrinsics = np.load(ext_key_file)  # [T_key, 3, 4]
        key_intrinsics = torch.from_numpy(np.load(int_key_file)).float()  # [T_key, 3, 3]
        
        query_memory_clip = torch.from_numpy(np.load(memory_file)).float()  # [4, 782, 1024]
        query_extrinsics = np.load(ext_query_file)  # [3, 4]
        query_intrinsics = torch.from_numpy(np.load(int_query_file)).float()  # [3, 3]
        
        # === convert extrinsics to local coordinate system (with key's first frame as origin) ===
        # 1. merge key and query extrinsics
        combined_extrinsics_np = np.concatenate([key_extrinsics, query_extrinsics[np.newaxis, :, :]], axis=0)  # [T_key+1, 3, 4]
        
        # 2. convert to local coordinate system (with first frame as origin)
        combined_extrinsics_local = convert_to_local_coordinates(combined_extrinsics_np)  # [T_key+1, 3, 4]
        
        # 3. convert back to torch tensor
        combined_extrinsics = torch.from_numpy(combined_extrinsics_local).float().unsqueeze(1)  # [T_key+1, 1, 3, 4]
        
        # merge key and query data
        combined_memory_clip = torch.cat([key_memory_clip, query_memory_clip.unsqueeze(0)], dim=0)  # [T_key+1, 4, 782, 1024]
        combined_intrinsics = torch.cat([key_intrinsics, query_intrinsics.unsqueeze(0)], dim=0).unsqueeze(1)     # [T_key+1, 1, 3, 3]
        
        # generate pose tokens
        pose_tokens = extri_intri_to_pose_encoding(
            combined_extrinsics, combined_intrinsics, 
            image_size_hw=(256, 512)
        )  # [T_key+1, 1, Din]
        
        # remove extra dimensions
        if pose_tokens.dim() == 3:
            pose_tokens = pose_tokens.squeeze(1)  # [T_key+1, Din]
        
        # process memory tokens: [T_key+1, 4, 782, 1024] -> [T_key+1, 4*782, 1024]
        memory_tokens = combined_memory_clip.view(combined_memory_clip.shape[0], -1, combined_memory_clip.shape[-1])  # [T_key+1, 4*782, 1024]
        
        return {
            'pose_tokens': pose_tokens,      # [T_key+1, Din] - first T_key frames are key, last frame is query
            'memory_tokens': memory_tokens,  # [T_key+1, 4*782, 1024] - first T_key frames are key, last frame is query
        }


# ===================== Model 類 =====================
class MemoryRetrievalModel(nn.Module):
    """Memory Retrieval model (including 3D RoPE + Joint Self-Attention)"""
    
    def __init__(
        self, 
        input_dim: int,
        model_dim: int = 1024,
        num_heads: int = 8,
        theta: float = 10000.0,
        retrieval_blocks: int = 2,
        mem_enc_layers: int = 2, 
        joint_layers: int = 3,
        mse_weight: float = 1.0,
        eps: float = 1e-6,
        rope_mode: str = '3d'  # '3d', '1d', 'none'
    ):
        super().__init__()
        
        assert model_dim % num_heads == 0, f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.retrieval_blocks = retrieval_blocks
        self.mse_weight = mse_weight
        self.rope_mode = rope_mode
        
        assert self.head_dim % 2 == 0, f"head_dim ({self.head_dim}) must be even (RoPE requirement)"
        
        # Pose embedding: small MLP
        self.embed = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )
        
        # Memory embedding: project 1024 dim memory to model_dim
        self.memory_embed = nn.Sequential(
            nn.Linear(1024, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )
        
        # Learnable Query: 4*782 learnable output queries (class variables)
        self.learnable_query = nn.Parameter(torch.randn(1, 4*782, model_dim) * 0.02)
        
        # Multiple Retrieval Blocks
        self.retrieval_blocks_list = nn.ModuleList([
            RetrievalBlock(
                dim=model_dim, 
                num_heads=num_heads, 
                mem_enc_layers=mem_enc_layers,
                joint_layers=joint_layers
            ) for _ in range(retrieval_blocks)
        ])
        
        # remove CE loss related components, only keep memory reconstruction
        self.memory_proj = nn.Linear(model_dim, 1024)       # 用於 MSE loss (memory reconstruction)
        
        # No PE no extra initialization
        if self.rope_mode == '1d':
            # precompute enough long 1D RoPE
            # estimate maximum length: T=20, tokens_per_step=1+4*782=3129 -> ~62580
            self.freqs_cis_1d_long = precompute_freqs_cis(self.head_dim, end=65709, theta=theta)
            
        # precompute 3D RoPE frequencies (for joint self-attention)
        # support T=20, L=4, H=21, W=37 3D memory structure
        if self.rope_mode == '3d':
            self.f_freqs, self.h_freqs, self.w_freqs = precompute_freqs_cis_3d(
                self.head_dim, end=2048, theta=theta
            )
        
        # precompute 1D RoPE frequencies (old one, possibly referenced by other places, but mainly logic uses the ones defined above)
        self.freqs_cis_1d = precompute_freqs_cis(self.head_dim, end=1024, theta=theta)
        
    def _ensure_freqs_device(self):
        '''ensure freqs on the correct device'''
        device = next(self.parameters()).device
        if hasattr(self, 'f_freqs') and self.f_freqs.device != device:
            self.f_freqs = self.f_freqs.to(device)
            self.h_freqs = self.h_freqs.to(device)
            self.w_freqs = self.w_freqs.to(device)
        if hasattr(self, 'freqs_cis_1d_long') and self.freqs_cis_1d_long.device != device:
            self.freqs_cis_1d_long = self.freqs_cis_1d_long.to(device)
        if self.freqs_cis_1d.device != device:
            self.freqs_cis_1d = self.freqs_cis_1d.to(device)
    
    def forward(self, pose_tokens: torch.Tensor, memory_tokens: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            pose_tokens: [B, T, Din] T time steps of pose tokens (first T-1 frames are key, last frame is query)
            memory_tokens: [B, T, 4*782, 1024] T time steps of memory tokens
            mask: [B, T] valid position mask (1 for valid, 0 for padding)
            
        Returns:
            memory_pred: [B, 4*782, 1024] memory prediction for MSE loss
        """
        # ensure freqs on the correct device
        self._ensure_freqs_device()
        
        B, T = pose_tokens.shape[:2]
        
        # use mask to find query position for each sample (last valid position)
        # find index of last valid position for each sample [B]
        query_indices = mask.sum(dim=1) - 1  # [B]
        
        # extract query pose and memory for each sample
        batch_indices = torch.arange(B, device=pose_tokens.device)
        query_pose_token = pose_tokens[batch_indices, query_indices].unsqueeze(1)  # [B, 1, Din]
        query_memory_target = memory_tokens[batch_indices, query_indices]  # [B, 4*782, 1024] - for loss
        
        # create key_mask: exclude query position
        key_mask = mask.clone()  # [B, T]
        key_mask[batch_indices, query_indices] = False
        
        # key frames are all frames (including padding), but marked with mask which are valid
        key_pose_tokens = pose_tokens  # [B, T, Din]
        key_memory_tokens = memory_tokens  # [B, T, 4*782, 1024]
        
        # Embed inputs
        embedded_key_pose = self.embed(key_pose_tokens)  # [B, T, model_dim] or [B, T-1, model_dim]
        embedded_key_memory = self.memory_embed(key_memory_tokens)  # [B, T, 4*782, model_dim] or [B, T-1, 4*782, model_dim]
        
        embedded_query_pose = self.embed(query_pose_token)  # [B, 1, model_dim]
        
        q_tok = embedded_query_pose.squeeze(1).unsqueeze(1)  # [B, 1, model_dim]
        
        # expand learnable_query to batch size
        learnable_query = self.learnable_query.expand(B, -1, -1)  # [B, 4*782, model_dim]
        
        query_freqs = None
        memory_freqs = None
        
        if self.rope_mode == '3d':
            # build 3D RoPE frequencies
            # Query freqs: 對 [q_tok ; learnable_query] = [1 + 4*782] tokens
            # important: query token should not have time position encoding
            query_freqs = build_3d_freqs_unified(
                T=1, L=4, H=21, W=37, 
                f_freqs=self.f_freqs, h_freqs=self.h_freqs, w_freqs=self.w_freqs,
                include_pose_token=True, use_time_encoding=False
            ).unsqueeze(1)  # [1+4*782, 1, head_dim]
            
            # Memory freqs: use actual length of key_pose_tokens to build position encoding
            # with mask: T (including padding and query position)
            memory_freqs = build_3d_freqs_unified(
                T=key_pose_tokens.shape[1], L=4, H=21, W=37,
                f_freqs=self.f_freqs, h_freqs=self.h_freqs, w_freqs=self.w_freqs,
                include_pose_token=True, use_time_encoding=True
            ).unsqueeze(1)  # [T_key*(1+4*782), 1, head_dim]
            
        elif self.rope_mode == '1d':
            # 1D RoPE: use global linear index
            # Query: [q_tok, learnable_query] -> length 1 + 4*782
            L_q = 1 + 4 * 782
            query_freqs = self.freqs_cis_1d_long[:L_q].unsqueeze(1) # [L_q, 1, head_dim/2]
            
            # Memory: each time step has (1+4*782) tokens
            # to make model aware of global time sequence, we use global index
            # total length = T_key * (1 + 4*782)
            # Memory Encoder will slice freqs when looping
            tokens_per_step = 1 + 4 * 782
            T_key = key_pose_tokens.shape[1]
            total_len = T_key * tokens_per_step
            
            if total_len > self.freqs_cis_1d_long.shape[0]:
                print(f"Warning: Sequence length {total_len} exceeds precomputed {self.freqs_cis_1d_long.shape[0]}")
                # dynamic expansion (slightly less efficient but safer)
                self.freqs_cis_1d_long = precompute_freqs_cis(self.head_dim, end=total_len + 1024, theta=10000.0).to(pose_tokens.device)
                
            memory_freqs = self.freqs_cis_1d_long[:total_len].unsqueeze(1) # [Total_L, 1, head_dim/2]
            
        # rope_mode == 'none': query_freqs, memory_freqs remain None
        
        # through multiple Retrieval Blocks
        # use key_mask to mask padding key frames
        joint_output = None
        encoded_key_pose = None
        for block in self.retrieval_blocks_list:
            joint_output, encoded_key_pose = block(
                q_tok=q_tok,
                pose_tokens=embedded_key_pose,
                memory_tokens=embedded_key_memory,
                query_freqs=query_freqs,
                memory_freqs=memory_freqs,
                learnable_query=learnable_query,
                key_mask=key_mask  # pass mask to attention
            )
            # subsequent blocks use output of previous block to update learnable_query
            learnable_query = joint_output[:, 1:, :]  # [B, 4*782, model_dim]
        
        # Split outputs
        memory_out = joint_output[:, 1:, :] # [B, 4*782, model_dim] - for MSE loss
        
        # Memory projection: restore original 1024 dim for MSE loss
        memory_pred = self.memory_proj(memory_out)  # [B, 4*782, 1024]
        
        return memory_pred, query_memory_target


# ===================== training function =====================
def train_one_step(model, batch, optimizer, device, world_size=1, max_grad_norm=1.0, mse_weight=1.0, 
                  use_bf16=False, bf16_full_eval=False):
    """train one step"""
    model.train()
    
    # move data to device and convert data type
    pose_tokens = batch['pose_tokens'].to(device)      # [B, T, Din]
    memory_tokens = batch['memory_tokens'].to(device)  # [B, T, 4*782, 1024]
    mask = batch.get('mask', None)
    if mask is not None:
        mask = mask.to(device)  # [B, T]
    
    # if using BF16, convert input data type
    if use_bf16:
        pose_tokens = pose_tokens.to(torch.bfloat16)
        memory_tokens = memory_tokens.to(torch.bfloat16)
    
    B, T = pose_tokens.shape[0], pose_tokens.shape[1]
    
    # Forward pass
    memory_pred, memory_target = model(pose_tokens, memory_tokens, mask)  # [B, 4*782, 1024], [B, 4*782, 1024]
    
    # Loss calculation: determine precision based on bf16_full_eval
    if use_bf16 and not bf16_full_eval:
        # BF16 training but FP32 loss calculation (more stable)
        memory_pred_loss = memory_pred.float()
        memory_target_loss = memory_target.float()
    else:
        # keep original precision for loss calculation
        memory_pred_loss = memory_pred
        memory_target_loss = memory_target
    
    # only keep MSE Loss: memory reconstruction
    mse_loss = F.mse_loss(memory_pred_loss, memory_target_loss)
    
    # total loss (only MSE)
    total_loss = mse_weight * mse_loss
    
    # Backward
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    
    # gradient clipping (Gradient Clipping)
    grad_norm_before = 0.0
    if max_grad_norm > 0:
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
    else:
        grad_norm_before = torch.sqrt(sum(
            torch.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None
        )).item()
    
    optimizer.step()
    
    # calculate metrics
    with torch.no_grad():
        # DDP sync metrics
        if world_size > 1:
            # only sync MSE loss and total loss
            metrics_to_sync = torch.stack([
                torch.tensor(mse_loss.item(), device=device),
                torch.tensor(total_loss.item(), device=device)
            ])
            metrics_to_sync = reduce_metric(metrics_to_sync, world_size)
            mse_sync, total_sync = metrics_to_sync
            mse_loss_item = mse_sync.item()
            total_loss_item = total_sync.item()
        else:
            mse_loss_item = mse_loss.item()
            total_loss_item = total_loss.item()
    
    return {
        'loss': total_loss_item,
        'mse_loss': mse_loss_item,
        'grad_norm': grad_norm_before,
    }


def main():
    parser = argparse.ArgumentParser(description="Memory Retrieval 3D RoPE + Joint Self-Attention training script")
    parser.add_argument("--data_dir", type=str, 
                       default="./data/",
                       help="base folder containing metadata.csv and *.npy files")
    parser.add_argument("--train_metadata", type=str, 
                       default="metadata.csv",
                       help="training set metadata file name (relative to data_dir)")
    parser.add_argument("--steps", type=int, default=None, help="training steps (if epochs is specified, this parameter will be ignored)")
    parser.add_argument("--epochs", type=int, default=1, help="training epochs (if specified, will override steps parameter)")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="maximum gradient norm for gradient clipping (0 means no clipping)")
    parser.add_argument("--model_dim", type=int, default=1024, help="model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="attention head number")
    
    # Memory retrieval specific parameters
    parser.add_argument("--retrieval_blocks", type=int, default=DEFAULT_RETRIEVAL_BLOCKS, help="retrieval blocks number")
    parser.add_argument("--mem_enc_layers", type=int, default=DEFAULT_MEM_ENC_LAYERS, help="memory encoder layers")
    parser.add_argument("--joint_layers", type=int, default=DEFAULT_JOINT_LAYERS, help="joint attention layers") 
    parser.add_argument("--mse_weight", type=float, default=1.0, help="MSE loss weight")
    parser.add_argument("--rope_mode", type=str, default="3d", choices=["3d", "1d", "none"], help="RoPE mode: 3d (default), 1d, none")
    
    # data and memory optimization parameters
    parser.add_argument("--analyze_model", action="store_true", help="analyze model parameters and memory usage")
    parser.add_argument("--use_bf16", action="store_true", help="use BFloat16 precision for training (requires supported GPU)")
    parser.add_argument("--bf16_full_eval", action="store_true", help="use BF16 for loss calculation in BF16 mode (default uses FP32)")
    
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--log_interval", type=int, default=20, help="log interval (steps)")
    parser.add_argument("--save_interval", type=int, default=500, help="save interval (steps)")
    parser.add_argument("--keep_checkpoints", type=int, default=3, help="number of recent checkpoints to keep")
    parser.add_argument("--resume_from", type=str, default=None, help="path to resume training from checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="device (auto/cpu/cuda)")
    parser.add_argument("--wandb_project", type=str, default="Captain-Safari", help="wandb project name")
    parser.add_argument("--wandb_name", type=str, default='memory-retriever-warmup', help="wandb run name")
    parser.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    
    args = parser.parse_args()
    
    # initialize distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    # set random seed (each process uses a different seed)
    set_seed(args.seed + rank)
    
    # select device
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # BF16 setup and check
    use_bf16 = args.use_bf16
    if use_bf16:
        if device.type == "cpu":
            print("Warning: CPU does not support BF16, automatically downgraded to FP32")
            use_bf16 = False
        elif device.type == "cuda":
            # check if GPU supports BF16
            major, minor = torch.cuda.get_device_capability(device)
            if major < 8:  # Ampere (A100, RTX 30/40 series) and above support BF16
                print(f"Warning: GPU compute capability {major}.{minor} does not support BF16, automatically downgraded to FP32")
                use_bf16 = False
    
    if is_main_process():
        print(f"Distributed training: {is_distributed}, Rank: {rank}/{world_size}, Device: {device}")
        print(f"Precision mode: {'BFloat16' if use_bf16 else 'Float32'}")
        if use_bf16:
            print(f"BF16 loss calculation: {'BF16' if args.bf16_full_eval else 'FP32'}")
        print(f"Retrieval blocks: {args.retrieval_blocks}")
        print(f"Memory encoder layers: {args.mem_enc_layers}")
        print(f"Joint attention layers: {args.joint_layers}")
        print(f"MSE weight: {args.mse_weight}")
        print(f"RoPE mode: {args.rope_mode}")
    
    # create dataset and DataLoader
    train_metadata_path = os.path.join(args.data_dir, args.train_metadata)
    
    if is_main_process():
        print(f"Loading training data from {train_metadata_path}...")
        print(f"Data file directory: {args.data_dir}")
        
    dataset = MemoryRetrievalDataset(
        train_metadata_path, args.data_dir, 
    )
    if is_main_process():
        print(f"Valid sample number: {len(dataset)}")
        
    # distributed sampler
    if is_distributed:
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # custom collate function to handle variable length sequences
    def custom_collate_fn(batch):
        """custom collate function to handle variable length key sequences"""
        # find the longest sequence length
        max_len = max(item['pose_tokens'].shape[0] for item in batch)
        
        # prepare batch data
        pose_tokens_list = []
        memory_tokens_list = []
        mask_list = []
        
        for item in batch:
            seq_len = item['pose_tokens'].shape[0]
            
            # padding pose_tokens
            pose_tokens = item['pose_tokens']
            if seq_len < max_len:
                pad_len = max_len - seq_len
                pose_tokens = torch.cat([
                    pose_tokens,
                    torch.zeros(pad_len, pose_tokens.shape[1], dtype=pose_tokens.dtype)
                ], dim=0)
            
            # padding memory_tokens
            memory_tokens = item['memory_tokens']
            if seq_len < max_len:
                pad_len = max_len - seq_len
                memory_tokens = torch.cat([
                    memory_tokens,
                    torch.zeros(pad_len, memory_tokens.shape[1], memory_tokens.shape[2], dtype=memory_tokens.dtype)
                ], dim=0)
            
            # create mask (1 for valid, 0 for padding)
            mask = torch.ones(max_len, dtype=torch.bool)
            if seq_len < max_len:
                mask[seq_len:] = 0
            
            pose_tokens_list.append(pose_tokens)
            memory_tokens_list.append(memory_tokens)
            mask_list.append(mask)
        
        # stack into a batch
        return {
            'pose_tokens': torch.stack(pose_tokens_list, dim=0),      # [B, max_len, Din]
            'memory_tokens': torch.stack(memory_tokens_list, dim=0),  # [B, max_len, 4*782, 1024]
            'mask': torch.stack(mask_list, dim=0)                     # [B, max_len]
        }
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=shuffle,
        sampler=sampler,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # check input_dim
    sample_batch = next(iter(dataloader))
    input_dim = sample_batch['pose_tokens'].shape[-1]
    if is_main_process():
        print(f"Pose encoding dimension: {input_dim}")
        print(f"Memory tokens shape: {sample_batch['memory_tokens'].shape}")
    
    # calculate total training steps (needs to have dataloader)
    if args.epochs is not None:
        steps_per_epoch = len(dataloader)
        total_steps = args.epochs * steps_per_epoch
    else:
        total_steps = args.steps
    
    # initialize wandb (only on main process)
    use_wandb = not args.no_wandb and is_main_process()
    if use_wandb:
        wandb_config = {
            'steps': args.steps,
            'epochs': args.epochs,
            'total_steps': total_steps,
            'batch_size': args.batch_size,
            'total_batch_size': args.batch_size * world_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'max_grad_norm': args.max_grad_norm,
            'model_dim': args.model_dim,
            'num_heads': args.num_heads,
            'retrieval_blocks': args.retrieval_blocks,
            'mem_enc_layers': args.mem_enc_layers,
            'joint_layers': args.joint_layers,
            'mse_weight': args.mse_weight,
            'rope_mode': args.rope_mode,
            'seed': args.seed,
            'world_size': world_size,
            'data_dir': args.data_dir,
            'use_bf16': use_bf16,
            'bf16_full_eval': args.bf16_full_eval if use_bf16 else False,
        }
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=wandb_config,
            tags=['memory-retrieval', 'rope', '3d-temporal', 'joint-attention', 'ddp'],
        )
        if is_main_process():
            print(f"Initialized wandb project: {args.wandb_project}")
    elif is_main_process():
        print("Skipping wandb logging")
    
    # create model
    model = MemoryRetrievalModel(
        input_dim=input_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        retrieval_blocks=args.retrieval_blocks,
        mem_enc_layers=args.mem_enc_layers,
        joint_layers=args.joint_layers,
        mse_weight=args.mse_weight,
        rope_mode=args.rope_mode
    ).to(device)
    
    # convert to BF16 (before DDP wrapping)
    if use_bf16:
        model = model.to(torch.bfloat16)
        if is_main_process():
            print("✅ Model converted to BFloat16 precision")
    
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        dtype_str = "BFloat16" if use_bf16 else "Float32"
        print(f"Model parameters: {total_params:,} ({dtype_str})")
    
    # wrap as DDP model
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # create optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # resume training from checkpoint
    start_step = 0
    if args.resume_from is not None:
        if is_main_process():
            print(f"Resuming training from checkpoint: {args.resume_from}")
        
        try:
            resume_info = load_checkpoint(
                args.resume_from, model, optimizer, scheduler, device
            )
            start_step = resume_info['step']
            
            if is_main_process():
                print(f"✅ Successfully resumed training, starting from step {start_step}")
                training_cfg = resume_info.get('training_config', {})
                if training_cfg.get('use_bf16') != use_bf16:
                    print(f"⚠️  Checkpoint BF16 setting ({training_cfg.get('use_bf16')}) does not match current setting ({use_bf16})")
        except Exception as e:
            if is_main_process():
                print(f"❌ Checkpoint recovery failed: {e}")
                print("Will start training from scratch")
    
    # 訓練循環
    if is_main_process():
        action = "Resuming training" if start_step > 0 else "Starting training"
        if args.epochs is not None:
            print(f"{action} (starting from step {start_step})...")
            print(f"Training mode: Epoch-based")
            print(f"  Epochs: {args.epochs}")
            print(f"  Steps per epoch: {len(dataloader)}")
            print(f"  Total steps: {total_steps}")
        else:
            print(f"{action} {total_steps} steps (starting from step {start_step})...")
            print(f"Training mode: Step-based")
        print(f"Training target: Memory Reconstruction (only MSE loss)")
        print(f"Total batch size: {args.batch_size * world_size}")
        print(f"Learning rate scheduler: CosineAnnealingLR (T_max={total_steps})")
        print(f"Initial learning rate: {args.lr:.2e}, Gradient clipping: {args.max_grad_norm}")
        print()
    
    model.train()
    step = start_step
    data_iter = iter(dataloader)
    current_epoch = start_step // len(dataloader) if args.epochs is not None else 0
    
    while step < total_steps:
        # set epoch for DistributedSampler
        if is_distributed and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(step // len(dataloader))
        start_time = time.time()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        end_time = time.time()
        
        # train one step
        metrics = train_one_step(
            model, batch, optimizer, device, world_size, 
            args.max_grad_norm, args.mse_weight, use_bf16, args.bf16_full_eval
        )
        
        # log to wandb
        if use_wandb:
            wandb_metrics = {
                'train/loss': metrics['loss'],
                'train/mse_loss': metrics['mse_loss'],
                'gradients/total_norm': metrics['grad_norm'],
                'optimizer/lr': optimizer.param_groups[0]['lr'],
                'step': step
            }
            if args.epochs is not None:
                wandb_metrics['epoch'] = step // len(dataloader)
            wandb.log(wandb_metrics, step=step)
        
        # console logging (only on main process)
        if is_main_process() and (step % args.log_interval == 0 or step == total_steps - 1):
            current_lr = optimizer.param_groups[0]['lr']
            if args.epochs is not None:
                current_epoch = step // len(dataloader)
                step_in_epoch = step % len(dataloader)
                print(f"Epoch {current_epoch}/{args.epochs} | Step {step_in_epoch}/{len(dataloader)} (Global: {step}) | "
                      f"LR: {current_lr:.2e} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"MSE: {metrics['mse_loss']:.4f} | "
                      f"GradNorm: {metrics['grad_norm']:.3f}")
            else:
                print(f"Step {step:5d} | "
                      f"LR: {current_lr:.2e} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"MSE: {metrics['mse_loss']:.4f} | "
                      f"GradNorm: {metrics['grad_norm']:.3f}")
        
        # periodically save checkpoint (only on main process)
        if is_main_process() and step > 0 and step % args.save_interval == 0:
            save_dir = "checkpoints/memory-retriever-warmup"
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, step, args, input_dim, 
                save_dir, args.keep_checkpoints
            )
            print(f"💾 Checkpoint saved to: {checkpoint_path}")
        
        # update learning rate
        scheduler.step()
        
        step += 1
    
    # save final model checkpoint
    if is_main_process():
        print("\nTraining completed! Saving model...")
        
        save_dir = "checkpoints/memory-retriever-warmup"
        os.makedirs(save_dir, exist_ok=True)
        
        # get actual model (if DDP, then get module)
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # save full checkpoint
        final_path = save_checkpoint(
            model, optimizer, scheduler, step, args, input_dim, 
            save_dir, keep_checkpoints=0  # no limit on final model number
        )
        # rename to final
        final_renamed = os.path.join(save_dir, f"final_step_{step}.pt")
        if final_path != final_renamed:
            os.rename(final_path, final_renamed)
        
        # save only model weights (for inference)
        model_only_path = os.path.join(save_dir, f"model_weights_step_{step}.pt")
        torch.save(model_to_save.state_dict(), model_only_path)
        
        print(f"✅ Final model saved to: {final_renamed}")
        print(f"✅ Model weights saved to: {model_only_path}")
        print(f"✅ Model saved! Total training steps: {step}")
    
    if is_main_process():
        print("\nTraining and saving completed!")
    
    # clean up distributed training
    if use_wandb:
        wandb.finish()
        if is_main_process():
            print("Finished wandb logging")
    
    if is_distributed:
        cleanup_distributed()
        if is_main_process():
            print("Cleaned up distributed training")


if __name__ == "__main__":
    main()
