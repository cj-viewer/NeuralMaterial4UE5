"""Report images for M2: per-LOD reconstructions (all 5 output maps) computed
with the numpy gold path only (packed BC1 bytes -> trilinear -> MLP), plus the
box-filtered reference mips.

    .venv/Scripts/python.exe m2_mip_lod/report_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pyramid import numpy_sample_levels

import bc1
from dataset import load_texture_set  # m0, via pyramid sys.path
from fixtures import mlp_forward

HERE = Path(__file__).parent
OUT = HERE / "output" / "report"
SIZES = [512, 256, 128]  # latent resolutions with trained mip + naive runs
LODS = [0, 1, 2, 3, 4]
SOURCE_RES = 512
DISPLAY = 256  # every image is presented at this size
MAPS = {  # name -> (slice, PIL mode)
    "basecolor": (slice(0, 3), "RGB"),
    "normal": (slice(3, 6), "RGB"),
    "ao": (slice(6, 7), "L"),
    "roughness": (slice(7, 8), "L"),
    "metallic": (slice(8, 9), "L"),
}


def texel_center_uv(res: int) -> np.ndarray:
    c = (np.arange(res, dtype=np.float32) + 0.5) / res
    gx, gy = np.meshgrid(c, c)
    return np.stack([gx, gy], axis=-1).reshape(-1, 2)


def recon_at_lod(run_dir: Path, lod: float, out_res: int) -> np.ndarray:
    meta = json.loads((run_dir / "meta.json").read_text())
    data = np.load(run_dir / "export.npz")
    uv = texel_center_uv(out_res)
    lods = np.full(uv.shape[0], np.float32(lod))
    feats = []
    for i, mips in enumerate(meta["latent_mip_resolutions"]):
        decoded = [bc1.decode_blocks(data[f"latent{i}_mip{j}_blocks"], r, r)
                   for j, r in enumerate(mips)]
        feats.append(numpy_sample_levels(decoded, uv, lods + np.float32(meta["lod_offsets"][i])))
    out = mlp_forward(np.concatenate(feats, axis=-1),
                      data["w0"], data["b0"], data["w1"], data["b1"])
    return out.reshape(out_res, out_res, -1)


def save_map(arr: np.ndarray, mode: str, name: str) -> None:
    a = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    img = Image.fromarray(a[..., 0] if mode == "L" else a, mode=mode)
    resample = Image.Resampling.LANCZOS if img.width > DISPLAY else Image.Resampling.NEAREST
    if img.width != DISPLAY:
        img = img.resize((DISPLAY, DISPLAY), resample)
    img.save(OUT / name, optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()

    target = load_texture_set(HERE.parent / "m0_float_latent_baseline" / "data" / f"synthetic_{SOURCE_RES}")
    for lod in LODS:
        res = SOURCE_RES >> lod
        f = SOURCE_RES // res
        ref = target.reshape(res, f, res, f, 9).mean(axis=(1, 3))
        recons = {"ref": ref}
        for size in SIZES:
            recons[f"mip{size}"] = recon_at_lod(HERE / "output" / f"mip_{size}", float(lod), res)
            recons[f"naive{size}"] = recon_at_lod(HERE / "output" / f"naive_{size}", float(lod), res)
        for key, img in recons.items():
            for map_name, (sl, mode) in MAPS.items():
                save_map(img[..., sl], mode, f"{key}_lod{lod}_{map_name}.png")
    n = len(list(OUT.glob("*.png")))
    print(f"{n} report images -> {OUT}")


if __name__ == "__main__":
    main()
