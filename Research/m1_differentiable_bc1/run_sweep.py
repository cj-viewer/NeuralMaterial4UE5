"""Latent-resolution sweep: float control vs simulated BC1 at each resolution.

Produces output/sweep/results.json + sweep_summary.md with compression ratios,
per-group PSNR, and the quantization cost (float PSNR - bc1 PSNR).

    .venv/Scripts/python.exe m1_differentiable_bc1/run_sweep.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SOURCE_RES = 512
RESOLUTIONS = [512, 384, 256, 192, 128]
MODES = ["float", "bc1"]

# Baselines, bits per source texel for the 9-channel set:
BPP_UNCOMPRESSED = 72.0  # 3x RGB8
BPP_TRADITIONAL_BC = 24.0  # BaseColor BC7 + Normal BC5 + ARM BC7, 8 bpp each
BPP_DIRECT_BC1 = 12.0  # the 3 source textures compressed directly as BC1, 4 bpp each
MLP_PARAM_COUNT = 12 * 32 + 32 + 32 * 9 + 9  # fp32 in the MVP contract


def kib(bpp: float) -> float:
    """VRAM of one 512^2 set at this rate, KiB (base level only, no mips)."""
    return bpp / 8.0 * SOURCE_RES * SOURCE_RES / 1024.0


def bpp_neural(latent_res: int) -> float:
    latent_bits = 4 * 4.0 * (latent_res / SOURCE_RES) ** 2  # 4 BC1 textures, 4 bpp each
    mlp_bits = MLP_PARAM_COUNT * 32 / (SOURCE_RES * SOURCE_RES)
    return latent_bits + mlp_bits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--force", action="store_true", help="re-run even if metrics.json exists")
    a = ap.parse_args()

    results = []
    for res in RESOLUTIONS:
        for mode in MODES:
            name = f"{mode}_{res}"
            run_dir = HERE / "output" / name
            if a.force or not (run_dir / "metrics.json").exists():
                cmd = [sys.executable, str(HERE / "train.py"), "--mode", mode,
                       "--latent-res", str(res), "--iters", str(a.iters), "--name", name]
                print(f"=== {name} ===")
                subprocess.run(cmd, check=True)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            metrics["name"] = name
            results.append(metrics)

    by_name = {r["name"]: r for r in results}
    rows = []
    for res in RESOLUTIONS:
        f, b = by_name[f"float_{res}"], by_name[f"bc1_{res}"]
        bpp = bpp_neural(res)
        rows.append({
            "latent_res": res,
            "bpp": round(bpp, 3),
            "kib": round(kib(bpp), 1),
            "ratio_vs_uncompressed": round(BPP_UNCOMPRESSED / bpp, 2),
            "ratio_vs_bc7": round(BPP_TRADITIONAL_BC / bpp, 2),
            "pct_vs_direct_bc1": round(bpp / BPP_DIRECT_BC1 * 100.0, 1),
            "pct_vs_bc7": round(bpp / BPP_TRADITIONAL_BC * 100.0, 1),
            "psnr_float": round(f["psnr_overall"], 2),
            "psnr_bc1": round(b["psnr_overall"], 2),
            "quantization_cost_db": round(f["psnr_overall"] - b["psnr_overall"], 2),
            "bc1_psnr_basecolor": round(b["psnr_basecolor"], 2),
            "bc1_psnr_normal": round(b["psnr_normal"], 2),
            "bc1_psnr_ao": round(b["psnr_ao"], 2),
            "bc1_psnr_roughness": round(b["psnr_roughness"], 2),
            "bc1_psnr_metallic": round(b["psnr_metallic"], 2),
        })

    out = HERE / "output" / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    baselines = {
        "uncompressed_rgb8": {"bpp": BPP_UNCOMPRESSED, "kib": round(kib(BPP_UNCOMPRESSED), 1)},
        "bc7_bc5_set": {"bpp": BPP_TRADITIONAL_BC, "kib": round(kib(BPP_TRADITIONAL_BC), 1)},
        "direct_bc1": {"bpp": BPP_DIRECT_BC1, "kib": round(kib(BPP_DIRECT_BC1), 1)},
    }
    (out / "results.json").write_text(
        json.dumps({"runs": results, "summary": rows, "baselines": baselines}, indent=2))

    hdr = ("| latent res | bpp | vs raw 72bpp | vs BC 24bpp | PSNR float | PSNR BC1 | "
           "quant cost (dB) | BC1 basecolor | normal | ao | rough | metal |")
    sep = "|" + "---|" * 12
    lines = ["# M1 sweep summary", "",
             f"Source set: {SOURCE_RES}x{SOURCE_RES}, 9 channels. 4 uniform latents, "
             f"MLP 12-32-9 fp32 included in bpp.", "", hdr, sep]
    for r in rows:
        lines.append(
            f"| {r['latent_res']} | {r['bpp']} | {r['ratio_vs_uncompressed']}x | "
            f"{r['ratio_vs_bc7']}x | {r['psnr_float']} | {r['psnr_bc1']} | "
            f"{r['quantization_cost_db']} | {r['bc1_psnr_basecolor']} | {r['bc1_psnr_normal']} | "
            f"{r['bc1_psnr_ao']} | {r['bc1_psnr_roughness']} | {r['bc1_psnr_metallic']} |")
    (out / "sweep_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
