"""Load a PBR texture-set folder into the fixed 9-channel target layout.

Channel semantics (spec 5.1, pinned): 0-2 BaseColor RGB, 3-5 Normal XYZ,
6 AO, 7 Roughness, 8 Metallic. All values are raw storage-space [0,1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

CHANNEL_GROUPS = {
    "basecolor": slice(0, 3),
    "normal": slice(3, 6),
    "ao": slice(6, 7),
    "roughness": slice(7, 8),
    "metallic": slice(8, 9),
}
NUM_CHANNELS = 9


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)[..., None] / 255.0


def load_texture_set(folder: Path) -> np.ndarray:
    """Returns [H, W, 9] float32."""
    folder = Path(folder)
    basecolor = _load_rgb(folder / "basecolor.png")
    normal = _load_rgb(folder / "normal.png")
    if (folder / "arm.png").exists():
        arm = _load_rgb(folder / "arm.png")
        ao, rough, metal = arm[..., 0:1], arm[..., 1:2], arm[..., 2:3]
    else:
        ao = _load_gray(folder / "ao.png")
        rough = _load_gray(folder / "roughness.png")
        metal = _load_gray(folder / "metallic.png")

    parts = [basecolor, normal, ao, rough, metal]
    shapes = {p.shape[:2] for p in parts}
    if len(shapes) != 1:
        raise ValueError(f"texture resolutions differ in {folder}: {shapes}")
    target = np.concatenate(parts, axis=-1)
    assert target.shape[-1] == NUM_CHANNELS
    return target
