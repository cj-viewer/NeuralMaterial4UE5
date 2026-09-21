"""Float latent textures + small MLP (M0 representation, spec 5.2).

UV convention: [0,1]^2, texel centers at (i+0.5)/N, clamp addressing —
grid_sample(align_corners=False, padding_mode="border") reproduces exactly this.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_LATENTS = 4
LATENT_CHANNELS = 3  # RGB, BC1-compatible for M1


class LatentTextures(nn.Module):
    def __init__(self, resolutions: list[int], clamp01: bool = False):
        super().__init__()
        if len(resolutions) != NUM_LATENTS:
            raise ValueError(f"need {NUM_LATENTS} latent resolutions, got {resolutions}")
        self.clamp01 = clamp01
        self.textures = nn.ParameterList(
            nn.Parameter(0.5 + 0.05 * torch.randn(1, LATENT_CHANNELS, r, r))
            for r in resolutions
        )

    def sample(self, uv: torch.Tensor) -> torch.Tensor:
        """uv: [B, 2] in [0,1] -> features [B, NUM_LATENTS * LATENT_CHANNELS]."""
        grid = (uv * 2.0 - 1.0).view(1, 1, -1, 2)
        feats = []
        for tex in self.textures:
            t = tex.clamp(0.0, 1.0) if self.clamp01 else tex
            s = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=False)
            feats.append(s[0, :, 0, :].t())  # [B, C]
        return torch.cat(feats, dim=-1)


class DecoderMLP(nn.Module):
    def __init__(self, in_dim: int = 12, hidden_dim: int = 32, out_dim: int = 9):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(F.relu(self.fc0(x)))


class NeuralMaterialModel(nn.Module):
    def __init__(self, latent_resolutions: list[int], hidden_dim: int = 32,
                 out_dim: int = 9, clamp01: bool = False):
        super().__init__()
        self.latents = LatentTextures(latent_resolutions, clamp01=clamp01)
        self.mlp = DecoderMLP(NUM_LATENTS * LATENT_CHANNELS, hidden_dim, out_dim)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.latents.sample(uv))


def texel_center_uv(height: int, width: int, device) -> torch.Tensor:
    """Full-resolution UV grid at texel centers, [H*W, 2] with uv = (i+0.5)/N."""
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).view(-1, 2)
