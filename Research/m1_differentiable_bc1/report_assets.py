"""Prepare report images from sweep outputs: downscaled full views + 2x zoom
crops of a fixed region so visual degradation across bpp is comparable.

    .venv/Scripts/python.exe m1_differentiable_bc1/report_assets.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
OUT = HERE / "output" / "sweep" / "report"
CROP = (176, 176, 336, 336)  # 160^2 region with tile grooves + blotch edge
RESOLUTIONS = [512, 384, 256, 192, 128]
VIEWS = ["basecolor", "normal"]


def full(src: Path, dst: Path, size: int = 384) -> None:
    Image.open(src).resize((size, size), Image.Resampling.LANCZOS).save(dst)


def zoom(src: Path, dst: Path, scale: int = 2) -> None:
    img = Image.open(src).crop(CROP)
    img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    img.save(dst)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref_dir = HERE / "output" / "bc1_512"  # ref_* identical across runs (same source)
    for view in VIEWS:
        full(ref_dir / f"ref_{view}.png", OUT / f"ref_{view}_full.png")
        zoom(ref_dir / f"ref_{view}.png", OUT / f"ref_{view}_zoom.png")
    for res in RESOLUTIONS:
        run = HERE / "output" / f"bc1_{res}"
        for view in VIEWS:
            full(run / f"recon_{view}.png", OUT / f"bc1_{res}_{view}_full.png")
            zoom(run / f"recon_{view}.png", OUT / f"bc1_{res}_{view}_zoom.png")
            zoom(run / f"diff_{view}.png", OUT / f"bc1_{res}_{view}_diff.png")
    n = len(list(OUT.glob("*.png")))
    print(f"{n} report images -> {OUT}")


if __name__ == "__main__":
    main()
