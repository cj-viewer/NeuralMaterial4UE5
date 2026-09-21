"""M2 training: BC1 mip pyramids with continuous-LOD sampling.

Run from Research/:
    .venv/Scripts/python.exe m2_mip_lod/train.py --latent-res 256
    .venv/Scripts/python.exe m2_mip_lod/train.py --latent-res 256 --base-only   # naive-mip baseline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pyramid import (MipMaterialModel, build_target_pyramid, fill_naive_mips,
                     sample_levels)

import bc1  # via pyramid's sys.path
from common import (CHANNEL_GROUPS, evaluate, generate, load_texture_set,
                    save_images)

HERE = Path(__file__).parent
FIXTURE_SEED = 4242
FIXTURE_COUNT = 64
EXPORT_VERSION = 3
EVAL_LODS = [0, 1, 2, 3, 4]


def psnr(mse: float) -> float:
    return float("inf") if mse <= 0 else float(-10.0 * np.log10(mse))


@torch.no_grad()
def evaluate_lods(model: MipMaterialModel, target_pyr: list[torch.Tensor], tile: int = 65536):
    from model import texel_center_uv  # m0
    out = {}
    for lod in EVAL_LODS:
        if lod >= len(target_pyr):
            break
        tgt = target_pyr[lod]
        _, c, h, w = tgt.shape
        uv = texel_center_uv(h, w, tgt.device)
        preds = []
        for i in range(0, uv.shape[0], tile):
            batch = uv[i : i + tile]
            preds.append(model(batch, torch.full((batch.shape[0],), float(lod), device=tgt.device)))
        recon = torch.cat(preds).t().reshape(1, c, h, w)
        out[f"psnr_lod{lod}"] = psnr(((recon - tgt) ** 2).mean().item())
    return out


def export(model: MipMaterialModel, args, out_dir: Path) -> None:
    model.eval()
    device = next(model.parameters()).device
    rng = np.random.default_rng(FIXTURE_SEED)
    fixture_uv = rng.random((FIXTURE_COUNT, 2), dtype=np.float32)
    fixture_lod = (rng.random(FIXTURE_COUNT, dtype=np.float32) * args.max_lod).astype(np.float32)
    with torch.no_grad():
        uv_t = torch.from_numpy(fixture_uv).to(device)
        lod_t = torch.from_numpy(fixture_lod).to(device)
        feats = model.sample(uv_t, lod_t)
        outs = model.mlp(feats)

    arrays = {
        "w0": model.mlp.fc0.weight.detach().cpu().numpy(),
        "b0": model.mlp.fc0.bias.detach().cpu().numpy(),
        "w1": model.mlp.fc1.weight.detach().cpu().numpy(),
        "b1": model.mlp.fc1.bias.detach().cpu().numpy(),
        "fixture_uv": fixture_uv,
        "fixture_lod": fixture_lod,
        "fixture_features": feats.cpu().numpy(),
        "fixture_output": outs.cpu().numpy(),
    }
    for i, pyr in enumerate(model.pyramids):
        level_blocks = []
        for j, lvl in enumerate(pyr.levels):
            blocks = bc1.pack_blocks(
                np.clip(lvl.e0.detach().cpu().numpy(), 0, 1),
                np.clip(lvl.e1.detach().cpu().numpy(), 0, 1),
                lvl.w.detach().cpu().numpy())
            arrays[f"latent{i}_mip{j}_blocks"] = blocks
            level_blocks.append(blocks)
        bc1.write_dds_mips(out_dir / f"latent{i}.dds", level_blocks,
                           pyr.resolutions[0], pyr.resolutions[0])
    np.savez(out_dir / "export.npz", **arrays)

    meta = {
        "export_version": EXPORT_VERSION,
        "milestone": "M2-mip",
        "mode": "bc1-mip" + ("-baseonly-naive" if args.base_only else ""),
        "channels": {k: [v.start, v.stop] for k, v in CHANNEL_GROUPS.items()},
        "latent_count": len(model.pyramids),
        "latent_resolutions": [p.resolutions[0] for p in model.pyramids],
        "latent_mip_resolutions": [p.resolutions for p in model.pyramids],
        "lod_offsets": model.lod_offsets,
        "max_train_lod": args.max_lod,
        "mlp": {"layout": "row-major [out,in]; y = w1 @ relu(w0 @ x + b0) + b1",
                "in_dim": 12, "hidden_dim": args.hidden, "out_dim": 9, "activation": "relu"},
        "uv_convention": "uv in [0,1], texel center (i+0.5)/N, bilinear, clamp addressing",
        "lod_convention": ("continuous lod in source-mip units; per-latent lod_k = "
                           "clamp(lod + log2(res_k/source_res), 0, levels_k-1); trilinear = "
                           "decode BC1 -> bilinear per level -> lerp by frac(lod)"),
        "lod_sampling": f"lod = U^{args.lod_pow} * max_lod",
        "target_mip_filter": "2x2 box average (runtime mipgen must match)",
        "bc1": "opaque 4-color mode only; texel = lerp(C0, C1, w), w in {0,1/3,2/3,1}",
        "fixture_seed": FIXTURE_SEED,
        "data": str(args.data),
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--latent-res", type=int, default=256, help="power-of-two uniform latent resolution")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr-latent", type=float, default=1e-2)
    ap.add_argument("--lr-mlp", type=float, default=2e-3)
    ap.add_argument("--base-only", action="store_true",
                    help="train lod=0 only, then fill mips naively (baseline)")
    ap.add_argument("--lod-pow", type=float, default=2.0,
                    help="lod = U^p * max_lod; p>1 biases sampling toward lod 0 "
                         "(1.0 = uniform). Low LODs dominate rendering, and a latent "
                         "coarser than the source shares its base level between lod 0 "
                         "and 1 targets, so uniform sampling starves lod 0.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.data is None:
        args.data = HERE.parent / "m0_float_latent_baseline" / "data" / f"synthetic_{args.res}"
        if not (args.data / "basecolor.png").exists():
            generate(args.data, args.res)

    target_np = load_texture_set(args.data)
    h, w, _ = target_np.shape
    target = torch.from_numpy(target_np).permute(2, 0, 1).unsqueeze(0).to(args.device)
    target_pyr = build_target_pyramid(target)
    args.max_lod = float(len(target_pyr) - 1)

    model = MipMaterialModel([args.latent_res] * 4, h, args.hidden).to(args.device)
    opt = torch.optim.Adam([
        {"params": [p for pyr in model.pyramids for p in pyr.parameters()], "lr": args.lr_latent},
        {"params": model.mlp.parameters(), "lr": args.lr_mlp},
    ])
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(args.iters * 0.6), int(args.iters * 0.85)], gamma=0.3)

    run_name = args.name or (("naive_" if args.base_only else "mip_") + str(args.latent_res))
    out_dir = HERE / "output" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"mode={'base-only(naive mips)' if args.base_only else 'mip-trained'} "
          f"device={args.device} target={w}x{h} latent={args.latent_res} "
          f"levels={len(model.pyramids[0].levels)} max_lod={args.max_lod} out={out_dir}")

    t0 = time.time()
    for it in range(1, args.iters + 1):
        uv = torch.rand(args.batch, 2, device=args.device)
        if args.base_only:
            lod = torch.zeros(args.batch, device=args.device)
        else:
            lod = torch.rand(args.batch, device=args.device) ** args.lod_pow * args.max_lod
        tgt = sample_levels(target_pyr, uv, lod)
        loss = (model(uv, lod) - tgt).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p in model.pyramids.parameters():
                p.clamp_(0.0, 1.0)
        sched.step()
        if it % 1000 == 0 or it == 1:
            print(f"iter {it:5d}  L1 {loss.item():.5f}  ({time.time() - t0:.1f}s)")

    if args.base_only:
        fill_naive_mips(model)

    recon, metrics = evaluate(model, target)
    metrics.update(evaluate_lods(model, target_pyr))
    metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics["iters"] = args.iters
    metrics["mode"] = "naive" if args.base_only else "mip"
    metrics["latent_res"] = args.latent_res
    print(json.dumps(metrics, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_images(recon, target, out_dir)
    export(model, args, out_dir)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
