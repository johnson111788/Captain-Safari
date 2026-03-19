import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import os
from typing_extensions import Literal
import torch.nn.functional as F
from typing import Tuple

class SimpleAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, downscale_factor, num_residual_blocks=1):
        super(SimpleAdapter, self).__init__()

        # Pixel Unshuffle: reduce spatial dimensions by a factor of 8
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)

        # Convolution: reduce spatial dimensions by a factor
        #  of 2 (without overlap)
        self.conv = nn.Conv2d(in_dim * (downscale_factor ** 2), out_dim, kernel_size=kernel_size, stride=stride, padding=0)

        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        # Reshape to merge the frame dimension into batch
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)

        # Pixel Unshuffle operation
        x_unshuffled = self.pixel_unshuffle(x)

        # Convolution operation
        x_conv = self.conv(x_unshuffled)

        # Feature extraction with residual blocks
        out = self.residual_blocks(x_conv)

        # Reshape to restore original bf dimension
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))

        # Permute dimensions to reorder (if needed), e.g., swap channels and feature frames
        out = out.permute(0, 2, 1, 3, 4)

        return out
    
    def process_camera_coordinates(
        self,
        direction: Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown", "In", "Out"],
        length: int, # num_frames
        height: int,
        width: int,
        speed: float = 1/54,
        origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    ):
        if origin is None:
            origin = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
        coordinates = generate_camera_coordinates(direction, length, speed, origin)
        plucker_embedding = process_pose_file(coordinates, width, height)
        return plucker_embedding
        
    
    def process_vggt_coordinates(
        self,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        length: int, # num_frames
        height: int,
        width: int,
    ):
        
        plucker_embedding = interpolate_and_generate_plucker_cuda(extrinsic, intrinsic, length, height, width)
        
        return plucker_embedding

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out
    
class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

def get_relative_pose(cam_params):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def custom_meshgrid(*args):
    # torch>=2.0.0 only
    return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def process_pose_file(cam_params, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu', return_poses=False):
    if return_poses:
        return cam_params
    else:
        cam_params = [Camera(cam_param) for cam_param in cam_params]

        sample_wh_ratio = width / height
        pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = height * pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / width
        else:
            resized_ori_h = width / pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / height

        intrinsic = np.asarray([[cam_param.fx * width,
                                cam_param.fy * height,
                                cam_param.cx * width,
                                cam_param.cy * height]
                                for cam_param in cam_params], dtype=np.float32)

        K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
        c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
        c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
        plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
        plucker_embedding = plucker_embedding[None]
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
        return plucker_embedding



def generate_camera_coordinates(
    direction: Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown", "In", "Out"],
    length: int,
    speed: float = 1/54,
    origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
):
    coordinates = [list(origin)]
    while len(coordinates) < length:
        coor = coordinates[-1].copy()
        if "Left" in direction:
            coor[9] += speed
        if "Right" in direction:
            coor[9] -= speed
        if "Up" in direction:
            coor[13] += speed
        if "Down" in direction:
            coor[13] -= speed
        if "In" in direction:
            coor[18] -= speed
        if "Out" in direction:
            coor[18] += speed
        coordinates.append(coor)
    return coordinates


def generate_plucker_embedding(extrinsic: torch.Tensor, intrinsic: torch.Tensor, height: int = 672, width: int = 384):
    """
    Generate Plucker embedding from extrinsic and intrinsic parameters
    
    Args:
        extrinsic: world-to-camera parameters (B, F, 3, 4) or (B, 3, 4) or (3, 4)
        intrinsic: intrinsic parameters (B, F, 3, 3) or (B, 3, 3) or (3, 3)
        height: image height
        width: image width
    
    Returns:
        plucker_embedding: Plucker embedding (B, 6, F, H, W)
    """
    # process dimensions
    if extrinsic.dim() == 2:  # (3, 4)
        extrinsic = extrinsic.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 4)
    elif extrinsic.dim() == 3:  # (B, 3, 4)
        extrinsic = extrinsic.unsqueeze(1)  # (B, 1, 3, 4)
    
    if intrinsic.dim() == 2:  # (3, 3)
        intrinsic = intrinsic.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
    elif intrinsic.dim() == 3:  # (B, 3, 3)
        intrinsic = intrinsic.unsqueeze(1)  # (B, 1, 3, 3)
    
    B, F = extrinsic.shape[:2]
    device = extrinsic.device
    dtype = extrinsic.dtype
    
    # create pixel grid (only create once)
    y_coords = torch.linspace(0, height - 1, height, device=device, dtype=dtype) + 0.5
    x_coords = torch.linspace(0, width - 1, width, device=device, dtype=dtype) + 0.5
    v, u = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)
    
    # construct homogeneous coordinates
    ones = torch.ones_like(u)
    homog = torch.stack([u, v, ones], dim=-1)  # (H, W, 3)
    
    # calculate plucker embedding for all frames
    plucker_frames = []
    
    for f in range(F):
        # extract current frame's extrinsic parameters (world-to-camera)
        R_w2c = extrinsic[:, f, :3, :3]  # (B, 3, 3) rotation
        t_w2c = extrinsic[:, f, :3, 3]   # (B, 3) translation
        K = intrinsic[:, f]  # (B, 3, 3)
        
        # convert to camera-to-world
        # for world-to-camera: P_cam = R_w2c * P_world + t_w2c
        # for camera-to-world: P_world = R_c2w * P_cam + t_c2w
        # where R_c2w = R_w2c^T, t_c2w = -R_w2c^T * t_w2c
        R_c2w = R_w2c.transpose(-1, -2)  # (B, 3, 3)
        t_c2w = -torch.bmm(R_c2w, t_w2c.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        
        # camera center in world coordinate system is t_c2w
        o = t_c2w  # (B, 3)
        
        # expand homogeneous coordinates to batch
        homog_batch = homog.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 3)
        
        # calculate inverse of intrinsic parameters (need to convert precision because inverse does not support bfloat16)
        K_inv = torch.inverse(K.float()).to(dtype)  # (B, 3, 3)
        
        # camera system direction vector
        r_cam = torch.einsum('bij,bhwj->bhwi', K_inv, homog_batch)  # (B, H, W, 3)
        
        # convert to world coordinate system (using camera-to-world rotation)
        d = torch.einsum('bij,bhwj->bhwi', R_c2w, r_cam)  # (B, H, W, 3)
        
        # normalize direction vector
        d = torch.nn.functional.normalize(d, dim=-1)  # (B, H, W, 3)
        
        # calculate Plücker moment: m = o × d
        o_expanded = o.unsqueeze(1).unsqueeze(1).expand(-1, height, width, -1)  # (B, H, W, 3)
        m = torch.cross(o_expanded, d, dim=-1)  # (B, H, W, 3)
        
        # combine Plücker coordinates [m, d]
        plucker = torch.cat([m, d], dim=-1)  # (B, H, W, 6)
        
        # transpose to (B, 6, H, W)
        plucker = plucker.permute(0, 3, 1, 2)  # (B, 6, H, W)
        
        plucker_frames.append(plucker)
    
    # Stack all frames: (B, 6, F, H, W)
    plucker_embedding = torch.stack(plucker_frames, dim=2)
    
    return plucker_embedding

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out

def extri_intri_to_pose_encoding(
    extrinsics,
    intrinsics,
    image_size_hw=None,  # e.g., (256, 512)
    pose_encoding_type="absT_quaR_FoV",
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding.

    This function transforms camera parameters into a unified pose encoding format,
    which can be used for various downstream tasks like pose prediction or representation.

    Args:
        extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4,
            where B is batch size and S is sequence length.
            In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world transformation.
            The format is [R|t] where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
        intrinsics (torch.Tensor): Camera intrinsic parameters with shape BxSx3x3.
            Defined in pixels, with format:
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
            where fx, fy are focal lengths and (cx, cy) is the principal point
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for computing field of view values. For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding to use. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).

    Returns:
        torch.Tensor: Encoded camera pose parameters with shape BxSx9.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
    """

    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3

    if pose_encoding_type == "absT_quaR_FoV":
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3

        quat = mat_to_quat(R)
        # Note the order of h and w here
        H, W = image_size_hw
        fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
        pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    else:
        raise NotImplementedError

    return pose_encoding

# === Minimal functions (copied from the previous cell, without docstrings to avoid parsing issues) ===
def skew(w: torch.Tensor) -> torch.Tensor:
    wx, wy, wz = w.unbind(-1)
    O = torch.zeros_like(wx)
    return torch.stack([
        torch.stack([O,     -wz,   wy ], dim=-1),
        torch.stack([wz,    O,    -wx ], dim=-1),
        torch.stack([-wy,   wx,    O  ], dim=-1),
    ], dim=-2)

def so3_log(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    tr = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0).clamp(-2.0, 2.0) * 0.5
    tr = tr.clamp(-1.0, 1.0)
    theta = torch.acos(tr)
    small = theta < eps
    w = torch.zeros(R.shape[:-2] + (3,), device=R.device, dtype=R.dtype)
    denom = 2.0 * torch.sin(theta)
    denom = torch.where(small, torch.ones_like(denom), denom)
    W_hat = (R - R.transpose(-1, -2)) / denom.unsqueeze(-1).unsqueeze(-1)
    w_gen = theta.unsqueeze(-1) * torch.stack([W_hat[..., 2, 1], W_hat[..., 0, 2], W_hat[..., 1, 0]], dim=-1)
    if small.any():
        w_small = torch.stack([
            (R[..., 2, 1] - R[..., 1, 2]) * 0.5,
            (R[..., 0, 2] - R[..., 2, 0]) * 0.5,
            (R[..., 1, 0] - R[..., 0, 1]) * 0.5,
        ], dim=-1)
        w = torch.where(small.unsqueeze(-1), w_small, w_gen)
    else:
        w = w_gen
    return w

def so3_exp(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    theta = torch.linalg.norm(w, dim=-1) + eps
    W = skew(w)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(W.shape[:-2] + (3, 3))
    a = torch.sin(theta) / theta
    b = (1.0 - torch.cos(theta)) / (theta * theta)
    a = a.unsqueeze(-1).unsqueeze(-1)
    b = b.unsqueeze(-1).unsqueeze(-1)
    WW = W @ W
    return I + a * W + b * WW

def se3_exp_left(omega: torch.Tensor, v_body: torch.Tensor, tau: torch.Tensor, eps: float = 1e-8):
    wt = omega * tau.unsqueeze(-1)
    R = so3_exp(wt, eps=eps)
    theta = torch.linalg.norm(omega, dim=-1) + eps
    tht = theta * tau
    W = skew(omega)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(W.shape[:-2] + (3, 3))
    A = (1.0 - torch.cos(tht)) / (theta * theta)
    B = (tht - torch.sin(tht)) / (theta * theta * theta)
    A = A.unsqueeze(-1).unsqueeze(-1)
    B = B.unsqueeze(-1).unsqueeze(-1)
    V = I * tau.unsqueeze(-1).unsqueeze(-1) + A * W + B * (W @ W)
    p = (V @ v_body.unsqueeze(-1)).squeeze(-1)
    return R, p

def build_time_axes(n_in: int, fps_in: float, length: int, fps_out: float, device, dtype):
    t_in = torch.arange(n_in, device=device, dtype=dtype) / float(fps_in)
    t_out = torch.arange(length, device=device, dtype=dtype) / float(fps_out)
    return t_in, t_out

def interp_with_extrap_1d(t_in: torch.Tensor, y: torch.Tensor, t_out: torch.Tensor) -> torch.Tensor:
    if y.dim() == 1:
        y = y.unsqueeze(0)
    *lead, N = y.shape
    y = y.reshape(-1, N)
    dt0 = (t_in[1] - t_in[0])
    dt1 = (t_in[-1] - t_in[-2])
    m0 = (y[:, 1] - y[:, 0]) / dt0
    m1 = (y[:, -1] - y[:, -2]) / dt1
    M = t_out.numel()
    out = y.new_empty((y.shape[0], M))
    idx = torch.searchsorted(t_in, t_out, right=True) - 1
    idx = idx.clamp(min=0, max=t_in.numel() - 2)
    t0 = t_in[idx]
    t1 = t_in[idx + 1]
    denom = (t1 - t0)
    denom[denom == 0] = 1.0
    alpha = (t_out - t0) / denom
    y0 = y[:, idx]
    y1 = y[:, idx + 1]
    interp_vals = (1.0 - alpha) * y0 + alpha * y1
    head_mask = t_out <= t_in[0]
    tail_mask = t_out >= t_in[-1]
    out[:] = interp_vals
    if head_mask.any():
        out[:, head_mask] = y[:, :1] + m0.unsqueeze(-1) * (t_out[head_mask] - t_in[0])
    if tail_mask.any():
        out[:, tail_mask] = y[:, -1:] + m1.unsqueeze(-1) * (t_out[tail_mask] - t_in[-1])
    return out.reshape(*lead, M)

def interpolate_intrinsics_cuda(K20: torch.Tensor, t_in: torch.Tensor, t_out: torch.Tensor) -> torch.Tensor:
    if K20.dim() == 4 and K20.shape[1] == 1:
        K20 = K20[:,0]
    if K20.dim() == 3:
        K20 = K20.unsqueeze(0)
    if K20.dim() != 4:
        raise ValueError(f"Unexpected K20 shape: {K20.shape}")
    B = K20.shape[0]; L = t_out.numel()
    fx = K20[..., 0, 0]; fy = K20[..., 1, 1]; cx = K20[..., 0, 2]; cy = K20[..., 1, 2]
    # for intrinsic parameters, use constant extrapolation (clamping) instead of linear extrapolation to avoid numerical instability
    # use torch.clamp to limit t_out within t_in range, use boundary value (nearest input value) for out of range
    t_out_clamped = torch.clamp(t_out, min=t_in[0], max=t_in[-1])
    fx_q = interp_with_extrap_1d(t_in, fx, t_out_clamped)
    fy_q = interp_with_extrap_1d(t_in, fy, t_out_clamped)
    cx_q = interp_with_extrap_1d(t_in, cx, t_out_clamped)
    cy_q = interp_with_extrap_1d(t_in, cy, t_out_clamped)
    K_out = torch.zeros((B, L, 3, 3), device=K20.device, dtype=K20.dtype)
    K_out[..., 0, 0] = fx_q; K_out[..., 1, 1] = fy_q
    K_out[..., 0, 2] = cx_q; K_out[..., 1, 2] = cy_q
    K_out[..., 2, 2] = 1.0
    return K_out

def interpolate_extrinsics_w2c_cuda(E20: torch.Tensor, t_in: torch.Tensor, t_out: torch.Tensor) -> torch.Tensor:
    if E20.dim() == 4 and E20.shape[1] == 1:
        E20 = E20[:,0]
    if E20.dim() == 3:
        E20 = E20.unsqueeze(0)
    if E20.shape[-2:] == (3, 4):
        I = torch.eye(4, device=E20.device, dtype=E20.dtype).view(1, 1, 4, 4).repeat(E20.shape[0], E20.shape[1], 1, 1)
        I[:, :, :3, :4] = E20
        E20 = I
    elif E20.shape[-2:] == (4, 4):
        pass
    else:
        raise ValueError(f"Unexpected E20 shape: {E20.shape}")
    B, N = E20.shape[:2]
    Rw2c = E20[..., :3, :3]; tw2c = E20[..., :3, 3]
    Rc2w = Rw2c.transpose(-1, -2)
    tc2w = -(Rc2w @ tw2c.unsqueeze(-1)).squeeze(-1)
    dt = (t_in[1:] - t_in[:-1]).view(1, -1, 1)
    R0 = Rc2w[:, :-1]; R1 = Rc2w[:, 1:]
    dR = R0.transpose(-1, -2) @ R1
    omega = so3_log(dR) / dt
    v_body = (R0.transpose(-1, -2) @ (tc2w[:, 1:] - tc2w[:, :-1]).unsqueeze(-1)).squeeze(-1) / dt
    idx = torch.searchsorted(t_in, t_out, right=True) - 1
    idx = idx.clamp(min=0, max=N-2)
    tau = t_out - t_in[idx]
    gather_idx = idx.view(1, -1, 1, 1).expand(B, -1, 3, 3)
    R0_q = R0.gather(1, gather_idx)
    gather_idx2 = idx.view(1, -1, 1).expand(B, -1, 3)
    t0_q = tc2w[:, :-1].gather(1, gather_idx2)
    omega_q = omega.gather(1, gather_idx2)
    v_body_q = v_body.gather(1, gather_idx2)
    R_delta, p_delta = se3_exp_left(omega_q, v_body_q, tau.to(E20.dtype))
    Rc = R0_q @ R_delta
    tc = t0_q + (R0_q @ p_delta.unsqueeze(-1)).squeeze(-1)
    Rw = Rc.transpose(-1, -2)
    tw = -(Rw @ tc.unsqueeze(-1)).squeeze(-1)
    E_out = torch.eye(4, device=E20.device, dtype=E20.dtype).view(1, 1, 4, 4).repeat(B, t_out.numel(), 1, 1)
    E_out[..., :3, :3] = Rw
    E_out[..., :3, 3] = tw
    return E_out

def interpolate_extrinsics_c2w_cuda(E20: torch.Tensor, t_in: torch.Tensor, t_out: torch.Tensor) -> torch.Tensor:
    """
    Interpolate camera-to-world extrinsics using SE(3) interpolation.
    
    Args:
        E20: Camera-to-world extrinsic matrices, shape [..., N, 3, 4] or [..., N, 4, 4]
        t_in: Input time stamps, shape [N]
        t_out: Output time stamps, shape [L]
    
    Returns:
        Interpolated camera-to-world extrinsic matrices, shape [B, L, 4, 4]
    """
    if E20.dim() == 4 and E20.shape[1] == 1:
        E20 = E20[:,0]
    if E20.dim() == 3:
        E20 = E20.unsqueeze(0)
    if E20.shape[-2:] == (3, 4):
        I = torch.eye(4, device=E20.device, dtype=E20.dtype).view(1, 1, 4, 4).repeat(E20.shape[0], E20.shape[1], 1, 1)
        I[:, :, :3, :4] = E20
        E20 = I
    elif E20.shape[-2:] == (4, 4):
        pass
    else:
        raise ValueError(f"Unexpected E20 shape: {E20.shape}")
    B, N = E20.shape[:2]
    # For c2w input, directly use the rotation and translation
    Rc2w = E20[..., :3, :3]; tc2w = E20[..., :3, 3]
    
    dt = (t_in[1:] - t_in[:-1]).view(1, -1, 1)
    R0 = Rc2w[:, :-1]; R1 = Rc2w[:, 1:]
    dR = R0.transpose(-1, -2) @ R1
    omega = so3_log(dR) / dt
    v_body = (R0.transpose(-1, -2) @ (tc2w[:, 1:] - tc2w[:, :-1]).unsqueeze(-1)).squeeze(-1) / dt
    
    idx = torch.searchsorted(t_in, t_out, right=True) - 1
    idx = idx.clamp(min=0, max=N-2)
    tau = t_out - t_in[idx]
    
    gather_idx = idx.view(1, -1, 1, 1).expand(B, -1, 3, 3)
    R0_q = R0.gather(1, gather_idx)
    gather_idx2 = idx.view(1, -1, 1).expand(B, -1, 3)
    t0_q = tc2w[:, :-1].gather(1, gather_idx2)
    omega_q = omega.gather(1, gather_idx2)
    v_body_q = v_body.gather(1, gather_idx2)
    
    R_delta, p_delta = se3_exp_left(omega_q, v_body_q, tau.to(E20.dtype))
    Rc2w_out = R0_q @ R_delta
    tc2w_out = t0_q + (R0_q @ p_delta.unsqueeze(-1)).squeeze(-1)
    
    # Return c2w matrices directly
    E_out = torch.eye(4, device=E20.device, dtype=E20.dtype).view(1, 1, 4, 4).repeat(B, t_out.numel(), 1, 1)
    E_out[..., :3, :3] = Rc2w_out
    E_out[..., :3, 3] = tc2w_out
    return E_out

def generate_plucker_embedding_cuda(extrinsic: torch.Tensor, intrinsic: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Generate Plucker embedding from camera-to-world extrinsics.
    
    Args:
        extrinsic: Camera-to-world extrinsic matrices, shape [..., L, 3, 4] or [..., L, 4, 4]
        intrinsic: Camera intrinsic matrices, shape [..., L, 3, 3]
        height: Image height
        width: Image width
    
    Returns:
        Plucker embedding tensor, shape [B, L, height, width, 6]
    """
    if extrinsic.dim() == 2: extrinsic = extrinsic.unsqueeze(0).unsqueeze(0)
    elif extrinsic.dim() == 3: extrinsic = extrinsic.unsqueeze(0)
    if intrinsic.dim() == 2: intrinsic = intrinsic.unsqueeze(0).unsqueeze(0)
    elif intrinsic.dim() == 3: intrinsic = intrinsic.unsqueeze(0)
    B, L = extrinsic.shape[:2]; device = extrinsic.device; dtype = extrinsic.dtype
    y = torch.linspace(0, height - 1, height, device=device, dtype=dtype) + 0.5
    x = torch.linspace(0, width - 1, width, device=device, dtype=dtype) + 0.5
    v, u = torch.meshgrid(y, x, indexing='ij'); one = torch.ones_like(u)
    homog = torch.stack([u, v, one], dim=-1)
    frames = []
    for f in range(L):
        R_w2c = extrinsic[:, f, :3, :3]; t_w2c = extrinsic[:, f, :3, 3]; K = intrinsic[:, f]
        R_c2w = R_w2c.transpose(-1, -2); t_c2w = -(R_c2w @ t_w2c.unsqueeze(-1)).squeeze(-1)
        # For c2w input, directly use the rotation and translation
        # R_c2w = extrinsic[:, f, :3, :3]; t_c2w = extrinsic[:, f, :3, 3]; K = intrinsic[:, f]
        K_inv = torch.inverse(K.float()).to(dtype)
        homog_b = homog.unsqueeze(0).expand(B, -1, -1, -1)
        r_cam = torch.einsum('bij,bhwj->bhwi', K_inv, homog_b)
        d = torch.einsum('bij,bhwj->bhwi', R_c2w, r_cam)
        d = torch.nn.functional.normalize(d, dim=-1)
        o = t_c2w; o_e = o.unsqueeze(1).unsqueeze(1).expand(-1, height, width, -1)
        m = torch.cross(o_e, d, dim=-1)
        pl = torch.cat([m, d], dim=-1)
        frames.append(pl)
    return torch.stack(frames, dim=1)

def interpolate_and_generate_plucker_cuda(extrinsic_20: torch.Tensor, intrinsic_20: torch.Tensor, length: int, height: int, width: int, fps_in: float = 4.0, fps_out: float = 24.0):
    device = extrinsic_20.device; dtype = extrinsic_20.dtype
    if extrinsic_20.dim() == 4 and extrinsic_20.shape[1] == 1: extrinsic_20 = extrinsic_20[:,0]
    if intrinsic_20.dim() == 4 and intrinsic_20.shape[1] == 1: intrinsic_20 = intrinsic_20[:,0]
    if extrinsic_20.dim() == 3: extrinsic_20 = extrinsic_20.unsqueeze(0)
    if intrinsic_20.dim() == 3: intrinsic_20 = intrinsic_20.unsqueeze(0)
    t_in, t_out = build_time_axes(extrinsic_20.shape[1], fps_in, length, fps_out, device=device, dtype=dtype)
    E4 = interpolate_extrinsics_w2c_cuda(extrinsic_20, t_in, t_out)
    Kq = interpolate_intrinsics_cuda(intrinsic_20, t_in, t_out)
    E3 = E4[..., :3, :4]
    pl = generate_plucker_embedding_cuda(E3, Kq, height=height, width=width)
    return pl

def convert_to_local_coordinates(extrinsics):
    """
    convert extrinsic parameters from global coordinate system to local coordinate system (first frame as origin)
    
    Args:
        extrinsics: (N, 3, 4) extrinsic parameters in global coordinate system (w2c format), support numpy array or torch tensor
    
    Returns:
        extrinsics_local: (N, 3, 4) extrinsic parameters in local coordinate system (with the same type as input)
    """
    import torch
    
    # check input type
    is_torch = isinstance(extrinsics, torch.Tensor)
    
    if is_torch:
        # Torch tensor version
        device = extrinsics.device
        dtype = extrinsics.dtype
        N = extrinsics.shape[0]
        
        # if low precision type (BFloat16/Float16), temporarily convert to Float32 for calculation
        need_conversion = dtype in (torch.bfloat16, torch.float16)
        if need_conversion:
            extrinsics_compute = extrinsics.float()
        else:
            extrinsics_compute = extrinsics
        
        # extrinsic parameters of first frame
        R0 = extrinsics_compute[0, :, :3]  # [3, 3] 第一帧旋转矩阵
        t0 = extrinsics_compute[0, :, 3]    # [3] 第一帧平移向量
        
        # calculate camera center of first frame (world coordinate system)
        C0 = -(R0.T @ t0)
        
        # c2w matrix of first frame
        R0_c2w = R0.T
        t0_c2w = C0
        
        # construct c2w 4x4 matrix of first frame
        T0_c2w = torch.eye(4, device=device, dtype=torch.float32 if need_conversion else dtype)
        T0_c2w[:3, :3] = R0_c2w
        T0_c2w[:3, 3] = t0_c2w
        
        # its inverse (w2c) is the reference transformation
        T0_w2c = torch.linalg.inv(T0_c2w)
        
        # convert all frames to local coordinate system
        extrinsics_local = torch.zeros_like(extrinsics_compute)
        
        for i in range(N):
            # extrinsic parameters of current frame
            Ri = extrinsics_compute[i, :, :3]
            ti = extrinsics_compute[i, :, 3]
            
            # construct 4x4 w2c matrix
            Ti_w2c = torch.eye(4, device=device, dtype=torch.float32 if need_conversion else dtype)
            Ti_w2c[:3, :3] = Ri
            Ti_w2c[:3, 3] = ti
            
            # convert to local coordinate system: T_local = T0_w2c @ inv(T_i_w2c)
            Ti_c2w = torch.linalg.inv(Ti_w2c)
            Ti_local_c2w = T0_w2c @ Ti_c2w
            Ti_local_w2c = torch.linalg.inv(Ti_local_c2w)
            
            # extract 3x4 extrinsic parameters
            extrinsics_local[i] = Ti_local_w2c[:3, :]
        
        # if original input is low precision type, convert back
        if need_conversion:
            extrinsics_local = extrinsics_local.to(dtype)
        
        return extrinsics_local
    
    else:
        # Numpy array version (backward compatibility)
        N = extrinsics.shape[0]
        
        # extrinsic parameters of first frame
        R0 = extrinsics[0, :, :3]  # rotation matrix of first frame
        t0 = extrinsics[0, :, 3]    # translation vector of first frame
        
        # calculate camera center of first frame (world coordinate system)
        C0 = -(R0.T @ t0)
        
        # c2w matrix of first frame
        R0_c2w = R0.T
        t0_c2w = C0
        
        # construct c2w 4x4 matrix of first frame
        T0_c2w = np.eye(4)
        T0_c2w[:3, :3] = R0_c2w
        T0_c2w[:3, 3] = t0_c2w
        
        # its inverse (w2c) is the reference transformation
        T0_w2c = np.linalg.inv(T0_c2w)
        
        # convert all frames to local coordinate system
        extrinsics_local = np.zeros_like(extrinsics)
        
        for i in range(N):
            # extrinsic parameters of current frame
            Ri = extrinsics[i, :, :3]
            ti = extrinsics[i, :, 3]
            
            # construct 4x4 w2c matrix
            Ti_w2c = np.eye(4)
            Ti_w2c[:3, :3] = Ri
            Ti_w2c[:3, 3] = ti
            
            # convert to local coordinate system: T_local = T0_w2c @ inv(T_i_w2c)
            Ti_c2w = np.linalg.inv(Ti_w2c)
            Ti_local_c2w = T0_w2c @ Ti_c2w
            Ti_local_w2c = np.linalg.inv(Ti_local_c2w)
            
            # extract 3x4 extrinsic parameters
            extrinsics_local[i] = Ti_local_w2c[:3, :]
        
        return extrinsics_local