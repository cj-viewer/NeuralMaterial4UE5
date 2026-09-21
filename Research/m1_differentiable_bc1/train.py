"""M1 training: BC1-simulated latents (or float control) + MLP.

Run from Research/:
    .venv/Scripts/python.exe m1_differentiable_bc1/train.py --mode bc1 --latent-res 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (CHANNEL_GROUPS, build_model, evaluate, generate,
                    load_texture_set, project_latents, sample_texture, save_images)
import bc1

HERE = Path(__file__).parent
FIXTURE_SEED = 4242
FIXTURE_COUNT = 64
EXPORT_VERSION = 2


def export(model, args, latent_res: list[int], out_dir: Path) -> None:
    model.eval()
    device = next(model.parameters()).device
    rng = np.random.default_rng(FIXTURE_SEED)
    fixture_uv = rng.random((FIXTURE_COUNT, 2), dtype=np.float32)
    with torch.no_grad():
        uv_t = torch.from_numpy(fixture_uv).to(device)
        feats = model.latents.sample(uv_t)
        outs = model.mlp(feats)

    arrays = {
        "w0": model.mlp.fc0.weight.detach().cpu().numpy(),
        "b0": model.mlp.fc0.bias.detach().cpu().numpy(),
        "w1": model.mlp.fc1.weight.detach().cpu().numpy(),
        "b1": model.mlp.fc1.bias.detach().cpu().numpy(),
        "fixture_uv": fixture_uv,
        "fixture_features": feats.cpu().numpy(),
        "fixture_output": outs.cpu().numpy(),
    }
    if args.mode == "bc1":
        for i, tex in enumerate(model.latents.textures):
            e0 = tex.e0.detach().cpu().numpy()
            e1 = tex.e1.detach().cpu().numpy()
            w = tex.w.detach().cpu().numpy()
            blocks = bc1.pack_blocks(np.clip(e0, 0, 1), np.clip(e1, 0, 1), w)
            arrays[f"latent{i}_blocks"] = blocks
            arrays[f"latent{i}_decoded"] = bc1.decode_blocks(blocks, tex.resolution, tex.resolution)
            bc1.write_dds(out_dir / f"latent{i}.dds", blocks, tex.resolution, tex.resolution)
    else:
        for i, tex in enumerate(model.latents.textures):
            arrays[f"latent{i}"] = tex.detach().cpu().numpy()[0]
    np.savez(out_dir / "export.npz", **arrays)

    meta = {
        "export_version": EXPORT_VERSION,
        "milestone": "M1-bc1" if args.mode == "bc1" else "M1-float-control",
        "mode": args.mode,
        "channels": {k: [v.start, v.stop] for k, v in CHANNEL_GROUPS.items()},
        "latent_count": len(latent_res),
        "latent_resolutions": latent_res,
        "mlp": {
            "layout": "row-major [out,in]; y = w1 @ relu(w0 @ x + b0) + b1",
            "in_dim": 12, "hidden_dim": args.hidden, "out_dim": 9,
            "activation": "relu",
        },
        "uv_convention": "uv in [0,1], texel center (i+0.5)/N, bilinear, clamp addressing",
        "bc1": "opaque 4-color mode only; texel = lerp(C0, C1, w), w in {0,1/3,2/3,1}",
        "fixture_seed": FIXTURE_SEED,
        "data": str(args.data),
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bc1", "float"], default="bc1")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--latent-res", type=int, default=512, help="uniform resolution of the 4 latents")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr-latent", type=float, default=1e-2)
    ap.add_argument("--lr-mlp", type=float, default=2e-3)
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

    latent_res = [args.latent_res] * 4
    model = build_model(args.mode, latent_res, args.hidden).to(args.device)

    opt = torch.optim.Adam([
        {"params": model.latents.parameters(), "lr": args.lr_latent},
        {"params": model.mlp.parameters(), "lr": args.lr_mlp},
    ])
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(args.iters * 0.6), int(args.iters * 0.85)], gamma=0.3)

    run_name = args.name or f"{args.mode}_{args.latent_res}"
    out_dir = HERE / "output" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"mode={args.mode} device={args.device} target={w}x{h} latents={latent_res} out={out_dir}")

    t0 = time.time()
    for it in range(1, args.iters + 1):
        uv = torch.rand(args.batch, 2, device=args.device)
        loss = (model(uv) - sample_texture(target, uv)).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        project_latents(model)
        sched.step()
        if it % 500 == 0 or it == 1:
            print(f"iter {it:5d}  L1 {loss.item():.5f}  ({time.time() - t0:.1f}s)")

    recon, metrics = evaluate(model, target)
    metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics["iters"] = args.iters
    metrics["mode"] = args.mode
    metrics["latent_res"] = args.latent_res
    print(json.dumps(metrics, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_images(recon, target, out_dir)
    export(model, args, latent_res, out_dir)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
