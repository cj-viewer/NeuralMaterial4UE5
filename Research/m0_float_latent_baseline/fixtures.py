"""Torch-free gold reference (spec 11.1): verify export.npz reproduces the
training-time outputs using only numpy, then write fixtures for M3/UE tests.

    .venv/Scripts/python.exe m0_float_latent_baseline/fixtures.py <run_dir>/export.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TOLERANCE = 1e-5


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


def mlp_forward(x: np.ndarray, w0, b0, w1, b1) -> np.ndarray:
    hidden = np.maximum(x @ w0.T + b0, 0.0)
    return (hidden @ w1.T + b1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=Path, help="path to export.npz")
    a = ap.parse_args()

    run_dir = a.export.parent
    meta = json.loads((run_dir / "meta.json").read_text())
    data = np.load(a.export)

    latents = [data[f"latent{i}"] for i in range(meta["latent_count"])]
    if meta["latent_clamp01"]:
        latents = [np.clip(l, 0.0, 1.0) for l in latents]

    uv = data["fixture_uv"]
    features = np.concatenate([bilinear_clamp(l, uv) for l in latents], axis=-1)
    output = mlp_forward(features, data["w0"], data["b0"], data["w1"], data["b1"])

    feat_err = np.abs(features - data["fixture_features"]).max()
    out_err = np.abs(output - data["fixture_output"]).max()
    print(f"latent sample max abs err vs torch: {feat_err:.3e}")
    print(f"mlp output    max abs err vs torch: {out_err:.3e}")
    ok = feat_err < TOLERANCE and out_err < TOLERANCE
    print("PASS" if ok else f"FAIL (tolerance {TOLERANCE})")

    np.savez(run_dir / "fixtures.npz", uv=uv, features=features, output=output,
             w0=data["w0"], b0=data["b0"], w1=data["w1"], b1=data["b1"])
    preview = [
        {
            "uv": uv[i].tolist(),
            "features": np.round(features[i], 6).tolist(),
            "output": np.round(output[i], 6).tolist(),
        }
        for i in range(4)
    ]
    (run_dir / "fixtures_preview.json").write_text(json.dumps(preview, indent=2))
    print(f"fixtures -> {run_dir / 'fixtures.npz'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
