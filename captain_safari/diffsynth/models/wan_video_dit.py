import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys
from .wan_video_camera_controller import SimpleAdapter
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = False
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, 
                   compatibility_mode=False, attn_mask: torch.Tensor = None):
    """
    Flash attention with optional attention mask support
    
    Args:
        attn_mask: [B, seq_k] boolean mask where True/1 means valid, False/0 means masked
    """
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        # Convert boolean mask to attention mask for scaled_dot_product_attention
        # scaled_dot_product_attention expects [B, num_heads, seq_q, seq_k]
        if attn_mask is not None:
            # attn_mask: [B, seq_k] -> [B, 1, 1, seq_k]
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            # Convert boolean to float mask (0 for valid, -inf for masked)
            attn_mask = torch.where(attn_mask, 0.0, float('-inf'))
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        # Flash attention 3 doesn't support mask directly, use compatibility mode if mask is provided
        if attn_mask is not None:
            q_compat = rearrange(q, "b s n d -> b n s d", n=num_heads)
            k_compat = rearrange(k, "b s n d -> b n s d", n=num_heads)
            v_compat = rearrange(v, "b s n d -> b n s d", n=num_heads)
            attn_mask_compat = attn_mask.unsqueeze(1).unsqueeze(2)
            attn_mask_compat = torch.where(attn_mask_compat, 0.0, float('-inf'))
            x = F.scaled_dot_product_attention(q_compat, k_compat, v_compat, attn_mask=attn_mask_compat)
            x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        else:
            x = flash_attn_interface.flash_attn_func(q, k, v)
            if isinstance(x,tuple):
                x = x[0]
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        # Flash attention 2 doesn't support mask directly, use compatibility mode if mask is provided
        if attn_mask is not None:
            q_compat = rearrange(q, "b s n d -> b n s d", n=num_heads)
            k_compat = rearrange(k, "b s n d -> b n s d", n=num_heads)
            v_compat = rearrange(v, "b s n d -> b n s d", n=num_heads)
            attn_mask_compat = attn_mask.unsqueeze(1).unsqueeze(2)
            attn_mask_compat = torch.where(attn_mask_compat, 0.0, float('-inf'))
            x = F.scaled_dot_product_attention(q_compat, k_compat, v_compat, attn_mask=attn_mask_compat)
            x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        else:
            x = flash_attn.flash_attn_func(q, k, v)
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        # SageAttention doesn't support mask, fall back to scaled_dot_product_attention if mask is provided
        if attn_mask is not None:
            # Convert boolean mask to attention mask
            attn_mask_compat = attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_k]
            attn_mask_compat = torch.where(attn_mask_compat, 0.0, float('-inf'))
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_compat)
        else:
            x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        # Fallback: use PyTorch's scaled_dot_product_attention
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        if attn_mask is not None:
            # Convert boolean mask to attention mask for scaled_dot_product_attention
            # attn_mask: [B, seq_k] -> [B, 1, 1, seq_k]
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            # Convert boolean to float mask (0 for valid, -inf for masked)
            attn_mask = torch.where(attn_mask, 0.0, float('-inf'))
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, attn_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, attn_mask=attn_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, attn_mask=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v, attn_mask=attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor, attn_mask: torch.Tensor = None):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v, attn_mask=attn_mask)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.memory_cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=False)
        self.norm_memory = nn.LayerNorm(dim, eps=eps)
        
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, memory_context=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)

        if memory_context is not None:
            x = x + self.memory_cross_attn(self.norm_memory(x), memory_context)
        
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


# ===================== Memory Retrieval Components =====================

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

    def forward(self, x_joint, freqs_cis=None):
        """
        override parent class forward method to support Joint RoPE logic
        
        x_joint: [B, L, D] where L = 1 + memory_tokens (Q at index 0, memory at indices 1..L)
        freqs_cis: [L, 1, head_dim] for 3D RoPE, first row is identity for Q
        """
        if freqs_cis is not None:
            # call parent class forward method, using provided freqs
            return super().forward(x_joint, freqs_cis)
        else:
            # no RoPE, perform standard attention
            B, L, D = x_joint.shape
            freqs_identity = torch.ones((L, 1, self.head_dim // 2), device=x_joint.device, dtype=torch.complex64)
            return super().forward(x_joint, freqs_identity)


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
    
    def forward(self, x_joint, freqs_cis=None):
        """
        standard Pre-Norm Transformer Block forward pass
        """
        x_joint = x_joint + self.joint_self_attn(self.norm1(x_joint), freqs_cis)
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
            UnifiedJointBlock(dim, num_heads, eps, 4) 
            for _ in range(mem_enc_layers)
        ])
        
        # Query-Output Joint Attention: multiple UnifiedJointBlock  
        self.joint_attn_blocks = nn.ModuleList([
            UnifiedJointBlock(dim, num_heads, eps, 4)
            for _ in range(joint_layers)
        ])
        
        # Cross Attention: reuse implementation of wan_video_dit
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=False)
        self.norm_cross = nn.LayerNorm(dim, eps=eps)
        
    def forward(self, q_tok, pose_tokens, memory_tokens, 
                query_freqs=None, memory_freqs=None, learnable_query=None):
        """
        Args:
            q_tok: [B, 1, dim] current pose token (query)
            pose_tokens: [B, T, dim] T time steps of pose tokens
            memory_tokens: [B, T, 4*782, dim] T time steps of memory tokens
            query_freqs: [1+4*782, 1, head_dim] RoPE freqs for query+learnable_query
            memory_freqs: [T*(1+4*782), 1, head_dim] RoPE freqs for memory sequence
            learnable_query: [B, 4*782, dim] learnable output queries
            
        Returns:
            joint_output: [B, 1+4*782, dim] contains q_out and memory_pred
        """
        B, T = pose_tokens.shape[:2]
        
        # 1. Memory Encoding: perform joint self-attention on (pose_t, memory_t) for each time step
        encoded_memory_list = []
        for t in range(T):
            # process each time step separately
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
            for block in self.mem_enc_blocks:
                joint_t = block(joint_t, freqs_t)
                
            encoded_memory_list.append(joint_t)
        
        # merge all time steps: [B, T*(1+4*782), dim]
        encoded_memory = torch.cat(encoded_memory_list, dim=1)
        
        # 2. Query-Output Joint Attention
        if learnable_query is None:
            raise ValueError("learnable_query is required for RetrievalBlock")
            
        joint_query = torch.cat([q_tok, learnable_query], dim=1)  # [B, 1+4*782, dim]
        
        # pass joint attention blocks
        for block in self.joint_attn_blocks:
            joint_query = block(joint_query, query_freqs)

        # 3. Cross Attention: joint_query attend to encoded_memory
        joint_output = joint_query + self.cross_attn(
            self.norm_cross(joint_query), encoded_memory
        )
        
        # 4. extract encoded_pose for CE loss key
        # encoded_memory: [B, T*(1+4*782), dim] -> [B, T, 1+4*782, dim]
        encoded_memory_reshaped = encoded_memory.view(B, T, 1+4*782, self.dim)
        encoded_pose = encoded_memory_reshaped[:, :, 0, :]  # [B, T, dim] - each time step's pose token
        
        return joint_output, encoded_pose


def build_3d_freqs_unified(T, L, H, W, f_freqs, h_freqs, w_freqs, include_pose_token=True, use_time_encoding=True):
    """
    Unified 3D RoPE build function, handling 3D position encoding of memory tokens
    
    Refer to train_retriever.py for implementation, supporting:
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
                f_freq = f_freqs[f_idx:f_idx+1]  # use freqs for this time step
            else:
                # Query token: no time encoding (following pose_retrieval_rope1d_joint.py)
                f_freq = f_freqs[0:1]  # time encoding set to 0
            
            h_freq = h_freqs[0:1]  # spatial position set to 0 (special identifier)
            w_freq = w_freqs[0:1]  # spatial position set to 0 (special identifier)
            combined_freq = torch.cat([f_freq, h_freq, w_freq], dim=-1)
            freqs_list.append(combined_freq)
        
        # next are L*782 tokens for this time step
        for l in range(L):
            # Special tokens (camera + register): time encoding according to use_time_encoding, spatial position set to 0
            for _ in range(5):  # 1 camera + 4 registers
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
            
            # Image tokens: time encoding according to use_time_encoding, use actual spatial position
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


class MemoryRetriever(nn.Module):
    """Memory Retrieval model (containing 3D RoPE + Joint Self-Attention)"""
    
    def __init__(
        self, 
        dim: int = 1024,
        num_heads: int = 8,
        theta: float = 10000.0,
        num_blocks: int = 2,
        mem_enc_layers: int = 2, 
        joint_layers: int = 3,
        mse_weight: float = 1.0,
        eps: float = 1e-6,
        rope_mode: str = '3d'  # '3d', '1d', 'none'
    ):
        super().__init__()
        
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_blocks = num_blocks
        self.mse_weight = mse_weight
        self.rope_mode = rope_mode
        
        assert self.head_dim % 2 == 0, f"head_dim ({self.head_dim}) must be even (RoPE requirement)"
        
        # Pose embedding: small MLP (matching original checkpoint)
        self.embed = nn.Sequential(
            nn.Linear(9, dim),  # input_dim=9 according to checkpoint
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # Memory embedding: project 1024 dim memory to model_dim
        self.memory_embed = nn.Sequential(
            nn.Linear(1024, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # Learnable Query: 4*782 learnable output queries (class variables)
        self.learnable_query = nn.Parameter(torch.randn(1, 4*782, dim) * 0.02)
        
        # Multiple Retrieval Blocks (matching checkpoint: mem_enc_layers=1, joint_layers=1)
        self.retrieval_blocks_list = nn.ModuleList([
            RetrievalBlock(
                dim=dim, 
                num_heads=num_heads, 
                mem_enc_layers=1,  # only 1 layer in checkpoint
                joint_layers=1     # only 1 layer in checkpoint
            ) for _ in range(num_blocks)
        ])
        
        # remove CE loss related components, only keep memory reconstruction
        self.memory_proj = nn.Linear(dim, 1024)       # for MSE loss (memory reconstruction)
        
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
        
        # do not register as buffer, avoid being destroyed by bfloat16 conversion
        # position encoding needs to keep complex128 precision
        # self.register_buffer('_f_freqs', self.f_freqs)
        # self.register_buffer('_h_freqs', self.h_freqs) 
        # self.register_buffer('_w_freqs', self.w_freqs)
        # self.register_buffer('_freqs_cis_1d', self.freqs_cis_1d)
    
    def _ensure_freqs_device(self):
        '''ensure freqs are on the correct device'''
        device = next(self.parameters()).device
        if hasattr(self, 'f_freqs') and self.f_freqs.device != device:
            self.f_freqs = self.f_freqs.to(device)
            self.h_freqs = self.h_freqs.to(device)
            self.w_freqs = self.w_freqs.to(device)
        if hasattr(self, 'freqs_cis_1d_long') and self.freqs_cis_1d_long.device != device:
            self.freqs_cis_1d_long = self.freqs_cis_1d_long.to(device)
        if self.freqs_cis_1d.device != device:
            self.freqs_cis_1d = self.freqs_cis_1d.to(device)
    
    def forward(self, pose_token, key_pose_token, memory_context):
        """
        Args:
            pose_token: [B, 1, 9] current pose token (query)
            key_pose_token: [B, 4, 9] previous 4 frames' pose tokens (keys)
            memory_context: [B, 4*4*782, 1024] previous 4 frames' memory tokens (need reshape)
            
        Returns:
            memory_pred: [B, 4*782, 1024] predicted memory
        """
        # ensure freqs are on the correct device
        self._ensure_freqs_device()
        
        B, T = key_pose_token.shape[:2]
        
        # Reshape memory from [B, 4*4*782, 1024] to [B, 4, 4*782, 1024]
        memory_tokens = memory_context.view(B, T, 4*782, 1024)  # [B, 4, 4*782, 1024]
        
        # Embed inputs according to original pose_predict_mem.py logic
        embedded_key_pose = self.embed(key_pose_token)  # [B, 4, model_dim]
        embedded_key_memory = self.memory_embed(memory_tokens)  # [B, 4, 4*782, model_dim]
        
        # process query pose token
        embedded_query_pose = self.embed(pose_token)  # [B, 1, model_dim]
        q_tok = embedded_query_pose  # [B, 1, model_dim]
        
        # expand learnable_query to batch size
        learnable_query = self.learnable_query.expand(B, -1, -1)  # [B, 4*782, dim]
        
        query_freqs = None
        memory_freqs = None
        
        if self.rope_mode == '3d':
            # build 3D RoPE frequencies
            # Query freqs: for [q_tok ; learnable_query] = [1 + 4*782] tokens
            # important: query token should not have time position encoding, following pose_retrieval_rope1d_joint.py design
            query_freqs = build_3d_freqs_unified(
                T=1, L=4, H=21, W=37, 
                f_freqs=self.f_freqs, h_freqs=self.h_freqs, w_freqs=self.w_freqs,
                include_pose_token=True, use_time_encoding=False
            ).unsqueeze(1)  # [1+4*782, 1, head_dim]
            
            # Memory freqs: use actual length of key_pose_tokens to build position encoding
            # with mask: T (including padding and query position)
            memory_freqs = build_3d_freqs_unified(
                T=T, L=4, H=21, W=37,
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
            # total length = T * (1 + 4*782)
            # Memory Encoder will slice freqs when looping
            tokens_per_step = 1 + 4 * 782
            total_len = T * tokens_per_step
            
            if total_len > self.freqs_cis_1d_long.shape[0]:
                print(f"Warning: Sequence length {total_len} exceeds precomputed {self.freqs_cis_1d_long.shape[0]}")
                # dynamic expansion (slightly less efficient but safer)
                self.freqs_cis_1d_long = precompute_freqs_cis(self.head_dim, end=total_len + 1024, theta=10000.0).to(key_pose_token.device)
                
            memory_freqs = self.freqs_cis_1d_long[:total_len].unsqueeze(1) # [Total_L, 1, head_dim/2]
            
        # rope_mode == 'none': query_freqs, memory_freqs remain None
        
        
        # through multiple Retrieval Blocks
        # important: only use previous 4 frames as memory context, avoid information leakage
        joint_output = None
        encoded_key_pose = None
        for block in self.retrieval_blocks_list:
            joint_output, encoded_key_pose = block(
                q_tok=q_tok,
                pose_tokens=embedded_key_pose,      # [B, 4, model_dim] - previous 4 frames' pose tokens
                memory_tokens=embedded_key_memory,  # [B, 4, 4*782, model_dim] - previous 4 frames' memory tokens
                query_freqs=query_freqs,
                memory_freqs=memory_freqs,
                learnable_query=learnable_query
            )
            # subsequent blocks use the output of the previous block to update learnable_query
            learnable_query = joint_output[:, 1:, :]  # [B, 4*782, dim]
        
        # Split outputs
        memory_out = joint_output[:, 1:, :] # [B, 4*782, dim] - for MSE loss
        
        # Memory projection: restore original 1024 dimension for MSE loss
        memory_pred = self.memory_proj(memory_out)  # [B, 4*782, 1024]
        
        return memory_pred


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        memory_dim: int = 1024,
        use_memory_cross_attn: bool = False,
        use_memory_retrieval: bool = False,
        num_retriever_blocks: int = 1,
        downscale_factor_control_adapter: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.use_memory_cross_attn = use_memory_cross_attn
        self.use_memory_retrieval = use_memory_retrieval

        self.patch_embedding = nn.Conv3d( # Conv3d(48, 3072, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.memory_emb = nn.Sequential(
            nn.Linear(memory_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.memory_retriever = MemoryRetriever(
            dim=memory_dim, 
            num_heads=num_heads//3, 
            num_blocks=num_retriever_blocks,
            rope_mode='3d',
            eps=eps,
        )
        
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:], downscale_factor=downscale_factor_control_adapter)
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor,control_camera_latents_input: torch.Tensor = None):
        x = self.patch_embedding(x) # torch.Size([1, 3072, 1, 22, 40])
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                memory_context: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context) # TODO: Here to add 3d memory
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, memory_context,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, memory_context,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs, memory_context)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            # 1.3B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            # 14B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": True
            }
        elif hash_state_dict_keys(state_dict) == "70ddad9d3a133785da5ea371aae09504":
            # 1.3B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": True
            }
        elif hash_state_dict_keys(state_dict) == "26bde73488a92e64cc20b0a7485b9e5b":
            # 14B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": True
            }
        elif hash_state_dict_keys(state_dict) == "ac6a5aa74f4a0aab6f64eb9a72f19901":
            # 1.3B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "b61c605c2adbd23124d152ed28e049ae":
            # 14B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "1f5ab7703c6fc803fdded85ff040c316":
            # Wan-AI/Wan2.2-TI2V-5B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
                "eps": 1e-6,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
            }
        elif hash_state_dict_keys(state_dict) == "5b013604280dd715f8457c6ed6d6a626":
            # Wan-AI/Wan2.2-I2V-A14B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "require_clip_embedding": False,
            }
        elif hash_state_dict_keys(state_dict) == "2267d489f0ceb9f21836532952852ee5":
            # Wan2.2-Fun-A14B-Control
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 52,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": True,
                "require_clip_embedding": False,
            }
        elif hash_state_dict_keys(state_dict) == "47dbeab5e560db3180adf51dc0232fb1":
            # Wan2.2-Fun-A14B-Control-Camera
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
                "require_clip_embedding": False,
            }
        elif hash_state_dict_keys(state_dict) == "0e2ab7dec4711919374f3d7ffdea90be":
            # Wan2.2-Fun-5B-Control-Camera
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 100,
                "dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
                "require_clip_embedding": False,
                "downscale_factor_control_adapter": 16,
            }
        else:
            config = {}
        return state_dict, config
