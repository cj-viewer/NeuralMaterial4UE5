"""Install an M2 training run into the PBR renderer's data/neural/ folder:
copies the 4 mipped latent DDS files and packs the MLP weights into the
716-float layout the shader expects (w0 | b0 | w1 | b1, float32 LE).

    .venv/Scripts/python.exe m3_pbr_visual/export_to_renderer.py m2_mip_lod/output/cerberus_1024
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DEST = HERE.parent / "PBR" / "data" / "neural"
TOTAL_FLOATS = 716  # 12*32 + 32 + 9*32 + 9 = 713, padded to a float4 multiple


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="M2 run directory (with export.npz and latent*.dds)")
    a = ap.parse_args()
    run = a.run if a.run.is_absolute() else HERE.parent / a.run

    meta = json.loads((run / "meta.json").read_text())
    data = np.load(run / "export.npz")
    DEST.mkdir(parents=True, exist_ok=True)

    for i in range(4):
        shutil.copyfile(run / f"latent{i}.dds", DEST / f"latent{i}.dds")

    w0, b0, w1, b1 = data["w0"], data["b0"], data["w1"], data["b1"]
    assert w0.shape == (32, 12) and w1.shape == (9, 32), (w0.shape, w1.shape)
    packed = np.zeros(TOTAL_FLOATS, dtype="<f4")
    stream = np.concatenate([w0.reshape(-1), b0, w1.reshape(-1), b1]).astype("<f4")
    packed[: stream.size] = stream
    packed.tofile(DEST / "weights.bin")

    (DEST / "source_run.json").write_text(json.dumps(
        {"run": str(run), "milestone": meta.get("milestone"),
         "latent_resolutions": meta.get("latent_resolutions"),
         "lod_offsets": meta.get("lod_offsets")}, indent=2))
    print(f"installed {run.name} -> {DEST}")


if __name__ == "__main__":
    main()
