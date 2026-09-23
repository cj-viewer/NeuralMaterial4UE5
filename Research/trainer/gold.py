"""Torch-free numpy gold reference (spec 11.1): bilinear/trilinear sampling and
the MLP forward, mirroring the GPU/torch arithmetic exactly. Consumed by
validate.py and by the D3D12/UE comparison tests."""

from __future__ import annotations

import numpy as np


def bilinear_clamp(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """tex [C,H,W], uv [B,2] in [0,1] -> [B,C]. Texel centers at (i+0.5)/N, clamp."""
    c, h, w = tex.shape
    x = uv[:, 0] * w - 0.5
    y = uv[:, 1] * h - 0.5
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0i = np.clip(x0.astype(np.int64), 0, w - 1)
    x1i = np.clip(x0.astype(np.int64) + 1, 0, w - 1)
    y0i = np.clip(y0.astype(np.int64), 0, h - 1)
    y1i = np.clip(y0.astype(np.int64) + 1, 0, h - 1)
    t00 = tex[:, y0i, x0i].T
    t10 = tex[:, y0i, x1i].T
    t01 = tex[:, y1i, x0i].T
    t11 = tex[:, y1i, x1i].T
    top = t00 * (1.0 - fx) + t10 * fx
    bot = t01 * (1.0 - fx) + t11 * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def numpy_sample_levels(decoded: list[np.ndarray], uv: np.ndarray, lod: np.ndarray) -> np.ndarray:
    """Trilinear over a decoded pyramid, mirroring model.sample_levels exactly.
    decoded: list of [C,H,W]; uv [B,2]; lod [B] -> [B,C]."""
    stack = np.stack([bilinear_clamp(d, uv) for d in decoded])  # [L,B,C]
    L = stack.shape[0]
    lodc = np.clip(lod, 0.0, float(L - 1)).astype(np.float32)
    m0 = np.floor(lodc).astype(np.int64)
    m1 = np.minimum(m0 + 1, L - 1)
    a = (lodc - m0)[:, None]
    idx = np.arange(uv.shape[0])
    return (stack[m0, idx] * (1.0 - a) + stack[m1, idx] * a).astype(np.float32)


def mlp_forward(x: np.ndarray, w0, b0, w1, b1) -> np.ndarray:
    """y = w1 @ relu(w0 @ x + b0) + b1, row-major [out, in] weights."""
    hidden = np.maximum(x @ w0.T + b0, 0.0)
    return (hidden @ w1.T + b1).astype(np.float32)
