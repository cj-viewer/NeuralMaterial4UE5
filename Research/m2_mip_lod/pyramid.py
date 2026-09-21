"""M2: independently trained BC1 mip pyramids + continuous-LOD sampling
(spec 5.5; Weinreich BCF / Belcour BCF1 mip approach).

Trilinear here means exactly what the GPU does with BC1: decode the two mip
levels, bilinear-filter each, then lerp by frac(lod).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "m1_differentiable_bc1"))
sys.path.insert(0, str(HERE.parent / "m0_float_latent_baseline"))

from bc1 import BC1LatentTexture, endpoints_to_565, rgb565_to_float  # noqa: E402
from fixtures import bilinear_clamp  # noqa: E402  (m0 numpy gold helper)

MIN_MIP_RES = 4  # BC1 blocks are 4x4; power-of-two latents only in M2


def mip_resolutions(resolution: int) -> list[int]:
    if resolution & (resolution - 1):
        raise ValueError(f"M2 requires power-of-two latent resolution, got {resolution}")
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


class MipMaterialModel(nn.Module):
    """4 BC1 latent pyramids + MLP; latents may sit at a different resolution
    than the source, so each gets a fixed lod offset log2(res_k / source_res)."""

    def __init__(self, latent_res: list[int], source_res: int, hidden_dim: int = 32,
                 out_dim: int = 9):
        super().__init__()
        from model import DecoderMLP  # m0
        self.pyramids = nn.ModuleList(BC1LatentPyramid(r) for r in latent_res)
        self.lod_offsets = [math.log2(r / source_res) for r in latent_res]
        self.mlp = DecoderMLP(len(latent_res) * 3, hidden_dim, out_dim)

    @property
    def latents(self):  # keep the m0/m1 export interface shape
        return self

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


# ---------------------------------------------------------------------------
# Naive-mip baseline: decode trained base, box-downsample, box-fit BC1 encode.
# This is what a "train base only, let the exporter make mips" pipeline gives.
# ---------------------------------------------------------------------------

@torch.no_grad()
def boxfit_bc1_params(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """image [1,3,H,W] in [0,1] -> (e0, e1, w) block params via per-block PCA:
    endpoints are the extreme projections onto the principal color axis.
    (Per-channel min/max endpoints are wrong for anti-correlated channels —
    the AABB corners are not on the color segment.)"""
    _, c, h, w = image.shape
    bh, bw = h // 4, w // 4
    blocks = image[0].reshape(c, bh, 4, bw, 4).permute(1, 3, 2, 4, 0)  # [bh,bw,4,4,3]
    flat = blocks.reshape(bh, bw, 16, 3)
    mean = flat.mean(dim=2, keepdim=True)
    x = (flat - mean).double()
    cov = x.transpose(-1, -2) @ x  # [bh,bw,3,3]
    _, vecs = torch.linalg.eigh(cov)
    d = vecs[..., -1]  # principal axis [bh,bw,3]
    t = (x @ d.unsqueeze(-1))[..., 0]  # [bh,bw,16]
    tmin = t.min(dim=2, keepdim=True).values
    tmax = t.max(dim=2, keepdim=True).values
    e0 = (mean[..., 0, :] + d * tmin).clamp(0.0, 1.0).to(image.dtype)
    e1 = (mean[..., 0, :] + d * tmax).clamp(0.0, 1.0).to(image.dtype)
    wmap = ((t - tmin) / (tmax - tmin).clamp_min(1e-12)).clamp(0.0, 1.0).to(image.dtype)
    wmap = wmap.reshape(bh, bw, 4, 4).permute(0, 2, 1, 3).reshape(h, w)
    return e0, e1, wmap


@torch.no_grad()
def fill_naive_mips(model: MipMaterialModel) -> None:
    """Overwrite every non-base mip level with box-downsampled + box-fit
    re-encoded BC1 of the trained base level."""
    for pyr in model.pyramids:
        base = pyr.levels[0].decode()
        for i in range(1, len(pyr.levels)):
            img = F.avg_pool2d(base, 2 ** i)
            e0, e1, wmap = boxfit_bc1_params(img)
            pyr.levels[i].e0.copy_(e0)
            pyr.levels[i].e1.copy_(e1)
            pyr.levels[i].w.copy_(wmap)


# ---------------------------------------------------------------------------
# Numpy gold reference (torch-free trilinear, mirrors sample_levels exactly)
# ---------------------------------------------------------------------------

def numpy_sample_levels(decoded: list[np.ndarray], uv: np.ndarray, lod: np.ndarray) -> np.ndarray:
    """decoded: list of [C,H,W]; uv [B,2]; lod [B] -> [B,C]."""
    stack = np.stack([bilinear_clamp(d, uv) for d in decoded])  # [L,B,C]
    L = stack.shape[0]
    lodc = np.clip(lod, 0.0, float(L - 1)).astype(np.float32)
    m0 = np.floor(lodc).astype(np.int64)
    m1 = np.minimum(m0 + 1, L - 1)
    a = (lodc - m0)[:, None]
    idx = np.arange(uv.shape[0])
    return (stack[m0, idx] * (1.0 - a) + stack[m1, idx] * a).astype(np.float32)
