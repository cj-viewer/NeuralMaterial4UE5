"""Neural material model: 4 independently trained BC1 latent mip pyramids +
a shared 12->32->9 FMA MLP, with continuous-LOD trilinear sampling that
matches GPU behavior exactly (decode BC1 per level -> bilinear -> lerp by
frac(lod)). Spec sections 5.2 / 5.5.

UV convention: [0,1]^2, texel centers at (i+0.5)/N, clamp addressing —
grid_sample(align_corners=False, padding_mode="border") reproduces this.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bc1 import BC1LatentTexture

NUM_LATENTS = 4
LATENT_CHANNELS = 3  # RGB, BC1
MIN_MIP_RES = 4  # BC1 blocks are 4x4; power-of-two latents only


def mip_resolutions(resolution: int) -> list[int]:
    if resolution & (resolution - 1):
        raise ValueError(f"latent resolution must be a power of two, got {resolution}")
    out = []
    r = resolution
    while r >= MIN_MIP_RES:
        out.append(r)
        r //= 2
    return out


class BC1LatentPyramid(nn.Module):
    """One latent texture as a chain of independently trainable BC1 mip levels."""

    def __init__(self, resolution: int):
        super().__init__()
        self.resolutions = mip_resolutions(resolution)
        self.levels = nn.ModuleList(BC1LatentTexture(r) for r in self.resolutions)

    def decode_all(self) -> list[torch.Tensor]:
        return [lvl.decode() for lvl in self.levels]


def sample_levels(decoded: list[torch.Tensor], uv: torch.Tensor, lod: torch.Tensor) -> torch.Tensor:
    """Trilinear sample of a decoded pyramid.

    decoded: list of [1,C,H,W] (mip 0 first); uv [B,2] in [0,1]; lod [B].
    Returns [B,C]. lod is clamped to the pyramid's own range.
    """
    grid = (uv * 2.0 - 1.0).view(1, 1, -1, 2)
    per_level = [
        F.grid_sample(d, grid, mode="bilinear", padding_mode="border",
                      align_corners=False)[0, :, 0, :].t()
        for d in decoded
    ]
    stack = torch.stack(per_level)  # [L, B, C]
    L, _, C = stack.shape
    lodc = lod.clamp(0.0, float(L - 1))
    m0 = lodc.floor().long()
    m1 = (m0 + 1).clamp(max=L - 1)
    a = (lodc - m0.to(lodc.dtype)).unsqueeze(-1)
    f0 = stack.gather(0, m0.view(1, -1, 1).expand(1, -1, C))[0]
    f1 = stack.gather(0, m1.view(1, -1, 1).expand(1, -1, C))[0]
    return f0 * (1.0 - a) + f1 * a


def build_target_pyramid(target: torch.Tensor, min_res: int = MIN_MIP_RES) -> list[torch.Tensor]:
    """Reference mip chain by 2x2 box filter (the mipgen convention the runtime
    asset must match, spec section 9)."""
    levels = [target]
    while min(levels[-1].shape[-2:]) > min_res:
        levels.append(F.avg_pool2d(levels[-1], 2))
    return levels


class DecoderMLP(nn.Module):
    def __init__(self, in_dim: int = 12, hidden_dim: int = 32, out_dim: int = 9):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(F.relu(self.fc0(x)))


class MipMaterialModel(nn.Module):
    """4 BC1 latent pyramids + MLP; latents may sit at a different resolution
    than the source, so each gets a fixed lod offset log2(res_k / source_res)."""

    def __init__(self, latent_res: list[int], source_res: int, hidden_dim: int = 32,
                 out_dim: int = 9):
        super().__init__()
        self.pyramids = nn.ModuleList(BC1LatentPyramid(r) for r in latent_res)
        self.lod_offsets = [math.log2(r / source_res) for r in latent_res]
        self.mlp = DecoderMLP(len(latent_res) * LATENT_CHANNELS, hidden_dim, out_dim)

    def sample(self, uv: torch.Tensor, lod: torch.Tensor | None = None) -> torch.Tensor:
        if lod is None:
            lod = torch.zeros(uv.shape[0], device=uv.device)
        feats = [
            sample_levels(pyr.decode_all(), uv, lod + off)
            for pyr, off in zip(self.pyramids, self.lod_offsets)
        ]
        return torch.cat(feats, dim=-1)

    def forward(self, uv: torch.Tensor, lod: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(self.sample(uv, lod))


def texel_center_uv(height: int, width: int, device) -> torch.Tensor:
    """Full-resolution UV grid at texel centers, [H*W, 2] with uv = (i+0.5)/N."""
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).view(-1, 2)
