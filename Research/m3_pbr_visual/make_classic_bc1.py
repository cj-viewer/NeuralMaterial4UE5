"""Encode the classic Cerberus maps as direct-BC1 DDS mip chains, so the
renderer's classic path is a 'direct BC1' baseline (same block format as the
neural latents).

    .venv/Scripts/python.exe m3_pbr_visual/make_classic_bc1.py --res 4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "m2_mip_lod"))
from pyramid import boxfit_bc1_params  # noqa: E402  (adds m1 to sys.path too)

import bc1  # noqa: E402

DEST = HERE.parent / "PBR" / "data" / "textures_bc1"
MIN_RES = 4
MAPS = [  # (source png, mode, output dds)
    ("basecolor.png", "RGB", "albedo.dds"),
    ("normal.png", "RGB", "normal.dds"),
    ("metallic.png", "L", "metalness.dds"),
    ("roughness.png", "L", "roughness.dds"),
]


def encode_map(src: Path, mode: str, dest: Path) -> None:
    img = np.asarray(Image.open(src).convert(mode), dtype=np.float32) / 255.0
    if mode == "L":
        img = np.repeat(img[..., None], 3, axis=-1)
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

    levels, level = [], t
    while level.shape[-1] >= MIN_RES:
        e0, e1, w = boxfit_bc1_params(level)
        levels.append(bc1.pack_blocks(e0.numpy(), e1.numpy(), w.numpy()))
        level = F.avg_pool2d(level, 2)

    res = t.shape[-1]
    bc1.write_dds_mips(dest, levels, res, res)

    decoded = bc1.decode_blocks(levels[0], res, res).transpose(1, 2, 0)
    mse = float(((decoded - img) ** 2).mean())
    psnr = -10.0 * np.log10(mse) if mse > 0 else float("inf")
    print(f"{dest.name:16s} {len(levels)} mips  base PSNR vs source {psnr:6.2f} dB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=4096, help="source set resolution (data/cerberus_<res>)")
    a = ap.parse_args()
    src = HERE / "data" / f"cerberus_{a.res}"
    DEST.mkdir(parents=True, exist_ok=True)
    for src_name, mode, dest_name in MAPS:
        encode_map(src / src_name, mode, DEST / dest_name)
    print(f"-> {DEST}")


if __name__ == "__main__":
    main()
