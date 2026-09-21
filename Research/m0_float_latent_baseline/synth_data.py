"""Deterministic procedural PBR texture set so M0 runs without external assets.

Writes basecolor.png, normal.png, arm.png (R=AO, G=Roughness, B=Metallic).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _value_noise(rng: np.random.Generator, res: int, cells: int) -> np.ndarray:
    grid = rng.random((cells, cells), dtype=np.float32)
    img = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    img = img.resize((res, res), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _fbm(rng: np.random.Generator, res: int, octaves: int = 4, base_cells: int = 4) -> np.ndarray:
    out = np.zeros((res, res), dtype=np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        out += amp * _value_noise(rng, res, base_cells * (2**o))
        total += amp
        amp *= 0.5
    return out / total


def generate(out_dir: Path, res: int = 512, seed: int = 1234) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    height = _fbm(rng, res, octaves=5)
    blotch = _fbm(rng, res, octaves=3, base_cells=3)

    # Sharp features so reconstruction quality on edges is visible: tile grooves.
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32) / res
    grooves = (np.abs(((xx * 8) % 1.0) - 0.5) < 0.04) | (np.abs(((yy * 8) % 1.0) - 0.5) < 0.04)
    height = np.where(grooves, height * 0.35, height)

    # Normal from height gradient, tangent space, stored in [0,1].
    scale = 6.0
    dhdx = np.gradient(height, axis=1) * res / 64.0
    dhdy = np.gradient(height, axis=0) * res / 64.0
    n = np.stack([-dhdx * scale, -dhdy * scale, np.ones_like(height)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    normal = (n * 0.5 + 0.5).clip(0.0, 1.0)

    color_a = np.array([0.62, 0.34, 0.20], dtype=np.float32)
    color_b = np.array([0.24, 0.26, 0.30], dtype=np.float32)
    t = np.clip((blotch - 0.35) / 0.3, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    basecolor = color_a[None, None, :] * (1.0 - t[..., None]) + color_b[None, None, :] * t[..., None]
    basecolor *= (0.55 + 0.45 * height)[..., None]
    basecolor = np.where(grooves[..., None], basecolor * 0.4, basecolor).clip(0.0, 1.0)

    metallic_mask = np.clip((blotch - 0.55) / 0.08, 0.0, 1.0)
    metallic = (metallic_mask * metallic_mask * (3.0 - 2.0 * metallic_mask)).clip(0.0, 1.0)
    roughness = (0.85 - 0.5 * height + 0.2 * _value_noise(rng, res, 16) - 0.4 * metallic).clip(0.05, 1.0)
    h01 = (height - height.min()) / (np.ptp(height) + 1e-8)
    ao = (0.55 + 0.45 * h01).clip(0.0, 1.0)
    ao = np.where(grooves, ao * 0.7, ao)

    def save(name: str, arr: np.ndarray) -> None:
        Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(out_dir / name)

    save("basecolor.png", basecolor)
    save("normal.png", normal)
    save("arm.png", np.stack([ao, roughness, metallic], axis=-1))
    print(f"synthetic PBR set ({res}x{res}, seed {seed}) -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    generate(a.out or Path(__file__).parent / "data" / f"synthetic_{a.res}", a.res, a.seed)
