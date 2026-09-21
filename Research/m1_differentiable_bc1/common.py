"""Shared plumbing for M1: imports the M0 modules (single source of truth for
dataset/model/eval code) and defines the BC1-latent model variant."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
M0_DIR = HERE.parent / "m0_float_latent_baseline"
sys.path.insert(0, str(M0_DIR))

from dataset import CHANNEL_GROUPS, load_texture_set  # noqa: E402
from model import DecoderMLP, NeuralMaterialModel, texel_center_uv  # noqa: E402
from synth_data import generate  # noqa: E402

from bc1 import BC1LatentTexture  # noqa: E402


def _load_m0_train():
    spec = importlib.util.spec_from_file_location("m0_train", M0_DIR / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m0_train = _load_m0_train()
sample_texture = m0_train.sample_texture
evaluate = m0_train.evaluate
save_images = m0_train.save_images
psnr = m0_train.psnr


class BC1LatentSet(nn.Module):
    """Four BC1-parameterized latent textures with the same sample() interface
    as M0's LatentTextures."""

    def __init__(self, resolutions: list[int]):
        super().__init__()
        self.textures = nn.ModuleList(BC1LatentTexture(r) for r in resolutions)

    def sample(self, uv: torch.Tensor) -> torch.Tensor:
        grid = (uv * 2.0 - 1.0).view(1, 1, -1, 2)
        feats = []
        for tex in self.textures:
            decoded = tex.decode()
            s = F.grid_sample(decoded, grid, mode="bilinear",
                              padding_mode="border", align_corners=False)
            feats.append(s[0, :, 0, :].t())
        return torch.cat(feats, dim=-1)


class BC1MaterialModel(nn.Module):
    def __init__(self, latent_resolutions: list[int], hidden_dim: int = 32, out_dim: int = 9):
        super().__init__()
        self.latents = BC1LatentSet(latent_resolutions)
        self.mlp = DecoderMLP(len(latent_resolutions) * 3, hidden_dim, out_dim)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.latents.sample(uv))


@torch.no_grad()
def project_latents(model: nn.Module) -> None:
    """Projected-gradient step: keep all latent parameters inside [0,1]."""
    for p in model.latents.parameters():
        p.clamp_(0.0, 1.0)


def build_model(mode: str, latent_res: list[int], hidden_dim: int = 32) -> nn.Module:
    if mode == "bc1":
        return BC1MaterialModel(latent_res, hidden_dim)
    if mode == "float":
        return NeuralMaterialModel(latent_res, hidden_dim, clamp01=True)
    raise ValueError(f"unknown mode {mode}")
