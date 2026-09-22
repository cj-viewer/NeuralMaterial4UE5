"""Convert the PBR renderer's Cerberus texture set into our trainer layout
(basecolor/normal/ao/roughness/metallic PNGs). AO is constant white — the
Cerberus set has no AO map and neither does the renderer's shading model.

    .venv/Scripts/python.exe m3_pbr_visual/prep_cerberus.py --res 1024
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
SRC = HERE.parent / "PBR" / "data" / "textures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=1024)
    a = ap.parse_args()
    out = HERE / "data" / f"cerberus_{a.res}"
    out.mkdir(parents=True, exist_ok=True)

    def load(name: str, mode: str) -> Image.Image:
        img = Image.open(SRC / name).convert(mode)
        if img.width != a.res:
            img = img.resize((a.res, a.res), Image.Resampling.LANCZOS)
        return img

    load("cerberus_A.png", "RGB").save(out / "basecolor.png")
    load("cerberus_N.png", "RGB").save(out / "normal.png")
    load("cerberus_R.png", "L").save(out / "roughness.png")
    load("cerberus_M.png", "L").save(out / "metallic.png")
    Image.new("L", (a.res, a.res), 255).save(out / "ao.png")
    print(f"cerberus set ({a.res}^2) -> {out}")


if __name__ == "__main__":
    main()
