"""Differentiable BC1 (opaque 4-color mode only, spec 5.3-5.4).

Torch side: STE quantization + simulated decode used during training.
Numpy side: bit-exact pack/unpack/decode of real 64-bit BC1 blocks + DDS writer.

Palette convention: per-texel interpolation weight w in {0, 1/3, 2/3, 1} of
endpoint1, i.e. texel = lerp(C0, C1, w). Real BC1 index encoding maps
w=0 -> index 0, w=1 -> index 1, w=1/3 -> index 2, w=2/3 -> index 3.
"""

from __future__ import annotations

import struct

import numpy as np
import torch
import torch.nn as nn

RGB565_SCALE = (31.0, 63.0, 31.0)
WEIGHT_LEVELS = 3.0  # w quantized to {0, 1/3, 2/3, 1}
W_TO_INDEX = (0, 2, 3, 1)  # w level 0..3 -> BC1 2-bit index
INDEX_TO_W = (0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0)  # BC1 index -> weight of C1


def _ste(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return x + (q - x).detach()


def _clamp01_passthrough(x: torch.Tensor) -> torch.Tensor:
    # Identity gradient: a plain clamp() zeroes gradients outside [0,1] and
    # permanently kills parameters that drift out of range.
    return _ste(x, x.clamp(0.0, 1.0))


def quantize_endpoints(e: torch.Tensor) -> torch.Tensor:
    """e [..., 3] -> RGB565-quantized values in [0,1], STE gradients."""
    scale = torch.tensor(RGB565_SCALE, device=e.device, dtype=e.dtype)
    e = _clamp01_passthrough(e)
    return _ste(e, torch.round(e * scale) / scale)


def quantize_weights(w: torch.Tensor) -> torch.Tensor:
    """w [...] -> {0,1/3,2/3,1}, STE gradients."""
    w = _clamp01_passthrough(w)
    return _ste(w, torch.round(w * WEIGHT_LEVELS) / WEIGHT_LEVELS)


class BC1LatentTexture(nn.Module):
    """One latent texture stored as trainable BC1 block parameters.

    Continuous parameters: endpoints e0/e1 per 4x4 block, weight w per texel.
    decode() returns the simulated-BC1 texel image [1, 3, H, W].
    """

    def __init__(self, resolution: int):
        super().__init__()
        if resolution % 4 != 0:
            raise ValueError(f"BC1 resolution must be a multiple of 4, got {resolution}")
        self.resolution = resolution
        blocks = resolution // 4
        # Endpoints start apart: equal endpoints give w a zero gradient through
        # lerp(C0, C1, w), stalling index training.
        self.e0 = nn.Parameter(0.35 + 0.05 * torch.randn(blocks, blocks, 3))
        self.e1 = nn.Parameter(0.65 + 0.05 * torch.randn(blocks, blocks, 3))
        self.w = nn.Parameter(0.5 + 0.05 * torch.randn(resolution, resolution))

    def decode(self) -> torch.Tensor:
        c0 = quantize_endpoints(self.e0)  # [B, B, 3]
        c1 = quantize_endpoints(self.e1)
        wq = quantize_weights(self.w)  # [H, W]
        c0 = c0.repeat_interleave(4, dim=0).repeat_interleave(4, dim=1)  # [H, W, 3]
        c1 = c1.repeat_interleave(4, dim=0).repeat_interleave(4, dim=1)
        tex = c0 + (c1 - c0) * wq[..., None]
        return tex.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]


# ---------------------------------------------------------------------------
# Numpy bitstream side (export / gold reference)
# ---------------------------------------------------------------------------

def endpoints_to_565(e: np.ndarray) -> np.ndarray:
    """e [..., 3] float in [0,1] -> packed uint16 RGB565."""
    e = np.clip(e, 0.0, 1.0)
    r = np.round(e[..., 0] * 31.0).astype(np.uint16)
    g = np.round(e[..., 1] * 63.0).astype(np.uint16)
    b = np.round(e[..., 2] * 31.0).astype(np.uint16)
    return (r << 11) | (g << 5) | b


def rgb565_to_float(c: np.ndarray) -> np.ndarray:
    """packed uint16 -> [..., 3] float32 (r/31, g/63, b/31), matching training decode."""
    r = ((c >> 11) & 31).astype(np.float32) / 31.0
    g = ((c >> 5) & 63).astype(np.float32) / 63.0
    b = (c & 31).astype(np.float32) / 31.0
    return np.stack([r, g, b], axis=-1)


def pack_blocks(e0: np.ndarray, e1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pack quantized params into real BC1 blocks.

    e0/e1: [By, Bx, 3] float in [0,1]; w: [H, W] float (values on the 1/3 grid).
    Returns uint8 [By*Bx*8] in BC1 memory order (little-endian c0, c1, indices).

    Enforces the opaque-mode constraint c0 > c1: swaps endpoints (w -> 1-w) when
    needed; equal endpoints force all indices to 0 so the 3-color/alpha mode of
    c0 <= c1 can never produce transparency.
    """
    by, bx, _ = e0.shape
    c0 = endpoints_to_565(e0)
    c1 = endpoints_to_565(e1)
    wl = np.round(np.clip(w, 0.0, 1.0) * 3.0).astype(np.uint32)  # weight level 0..3
    wl = wl.reshape(by, 4, bx, 4).transpose(0, 2, 1, 3)  # [By, Bx, 4, 4]

    swap = c0 < c1
    c0s = np.where(swap, c1, c0)
    c1s = np.where(swap, c0, c1)
    wl = np.where(swap[..., None, None], 3 - wl, wl)

    w_to_idx = np.array(W_TO_INDEX, dtype=np.uint32)
    idx = w_to_idx[wl]  # [By, Bx, 4, 4]
    idx = np.where((c0s == c1s)[..., None, None], np.uint32(0), idx)

    texel = np.arange(16, dtype=np.uint32)
    shifts = (texel * 2).reshape(4, 4)
    indices = (idx << shifts[None, None]).sum(axis=(2, 3), dtype=np.uint64).astype(np.uint32)

    out = np.empty((by, bx, 8), dtype=np.uint8)
    out[..., 0] = c0s & 0xFF
    out[..., 1] = c0s >> 8
    out[..., 2] = c1s & 0xFF
    out[..., 3] = c1s >> 8
    for i in range(4):
        out[..., 4 + i] = (indices >> (8 * i)) & 0xFF
    return out.reshape(-1)


def decode_blocks(blocks: np.ndarray, height: int, width: int) -> np.ndarray:
    """Reference decode of packed BC1 bytes -> [3, H, W] float32.

    Ideal float palette (identical arithmetic to BC1LatentTexture.decode),
    honoring the c0 <= c1 3-color mode the way our exporter avoids it.
    """
    by, bx = height // 4, width // 4
    b = blocks.reshape(by, bx, 8).astype(np.uint32)
    c0 = b[..., 0] | (b[..., 1] << 8)
    c1 = b[..., 2] | (b[..., 3] << 8)
    indices = b[..., 4] | (b[..., 5] << 8) | (b[..., 6] << 16) | (b[..., 7] << 24)

    shifts = (np.arange(16, dtype=np.uint32) * 2).reshape(4, 4)
    idx = (indices[..., None, None] >> shifts[None, None]) & 3  # [By, Bx, 4, 4]

    idx_to_w = np.array(INDEX_TO_W, dtype=np.float32)
    w = idx_to_w[idx]
    # c0 <= c1 selects 3-color mode: index 2 -> midpoint, index 3 -> black.
    three_color = (c0 <= c1)[..., None, None]
    w = np.where(three_color & (idx == 2), np.float32(0.5), w)
    black = three_color & (idx == 3)

    p0 = rgb565_to_float(c0)[..., None, None, :]
    p1 = rgb565_to_float(c1)[..., None, None, :]
    tex = p0 + (p1 - p0) * w[..., None]
    tex = np.where(black[..., None], np.float32(0.0), tex)  # [By, Bx, 4, 4, 3]
    tex = tex.transpose(0, 2, 1, 3, 4).reshape(height, width, 3)
    return np.ascontiguousarray(tex.transpose(2, 0, 1)).astype(np.float32)


def write_dds(path, blocks: np.ndarray, height: int, width: int) -> None:
    """Minimal DDS header (DXT1 FourCC, no mips) + BC1 payload."""
    write_dds_mips(path, [blocks], height, width)


def write_dds_mips(path, levels: list[np.ndarray], height: int, width: int) -> None:
    """DDS with a BC1 mip chain: levels[i] holds the packed blocks of mip i
    (dimensions height>>i x width>>i, all multiples of 4)."""
    DDSD_FLAGS = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000  # caps|height|width|pixelformat|linearsize
    caps = 0x1000  # DDSCAPS_TEXTURE
    mipcount = 0
    if len(levels) > 1:
        DDSD_FLAGS |= 0x20000  # DDSD_MIPMAPCOUNT
        caps |= 0x8 | 0x400000  # DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
        mipcount = len(levels)
    pitch = max(1, (width + 3) // 4) * 8 * max(1, (height + 3) // 4)
    header = struct.pack(
        "<4s7I44x2I4s5I2I12x",
        b"DDS ", 124, DDSD_FLAGS, height, width, pitch, 0, mipcount,
        32, 0x4, b"DXT1", 0, 0, 0, 0, 0,
        caps, 0,
    )
    assert len(header) == 128
    payload = b"".join(lvl.tobytes() for lvl in levels)
    if hasattr(path, "write"):
        path.write(header)
        path.write(payload)
    else:
        with open(path, "wb") as f:
            f.write(header)
            f.write(payload)


# ---------------------------------------------------------------------------
# Offline PCA-fit BC1 encoder: encode an arbitrary image (e.g. a classic map
# for the renderer baseline) as BC1 block parameters. Not used in training.
# ---------------------------------------------------------------------------

@torch.no_grad()
def boxfit_bc1_params(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """image [1,3,H,W] in [0,1] -> (e0, e1, w) block params via per-block PCA:
    endpoints are the extreme projections onto the principal color axis.
    (Per-channel min/max endpoints are wrong for anti-correlated channels --
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
