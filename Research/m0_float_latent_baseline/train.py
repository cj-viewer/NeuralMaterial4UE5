"""M0 training: float latents + MLP reconstruct one PBR texture set.

Run from Research/:
    .venv/Scripts/python.exe m0_float_latent_baseline/train.py --res 512 --iters 4000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import CHANNEL_GROUPS, load_texture_set
from model import NeuralMaterialModel, texel_center_uv
from synth_data import generate

HERE = Path(__file__).parent
FIXTURE_SEED = 4242
FIXTURE_COUNT = 64
EXPORT_VERSION = 1


def sample_texture(tex: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """tex [1,C,H,W], uv [B,2] in [0,1] -> [B,C]. Same conventions as the model."""
    grid = (uv * 2.0 - 1.0).view(1, 1, -1, 2)
    s = F.grid_sample(tex, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return s[0, :, 0, :].t()


def psnr(mse: float) -> float:
    return float("inf") if mse <= 0 else -10.0 * np.log10(mse)


@torch.no_grad()
def evaluate(model: NeuralMaterialModel, target: torch.Tensor, tile: int = 65536):
    _, c, h, w = target.shape
    uv = texel_center_uv(h, w, target.device)
    preds = []
    for i in range(0, uv.shape[0], tile):
        preds.append(model(uv[i : i + tile]))
    recon = torch.cat(preds).t().reshape(1, c, h, w)
    err = (recon - target) ** 2
    metrics = {"psnr_overall": psnr(err.mean().item())}
    for name, sl in CHANNEL_GROUPS.items():
        metrics[f"psnr_{name}"] = psnr(err[:, sl].mean().item())
    metrics["l1_overall"] = (recon - target).abs().mean().item()
    return recon, metrics


def save_images(recon: torch.Tensor, target: torch.Tensor, out_dir: Path) -> None:
    def to_img(t: torch.Tensor) -> Image.Image:
        arr = t.clamp(0, 1).mul(255).add(0.5).byte().cpu().numpy().transpose(1, 2, 0)
        return Image.fromarray(arr)

    views = {"basecolor": slice(0, 3), "normal": slice(3, 6), "arm": slice(6, 9)}
    for name, sl in views.items():
        to_img(target[0, sl]).save(out_dir / f"ref_{name}.png")
        to_img(recon[0, sl]).save(out_dir / f"recon_{name}.png")
        to_img((recon[0, sl] - target[0, sl]).abs() * 4).save(out_dir / f"diff_{name}.png")


def export(model: NeuralMaterialModel, args, latent_res: list[int], out_dir: Path) -> None:
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
    for i, tex in enumerate(model.latents.textures):
        arrays[f"latent{i}"] = tex.detach().cpu().numpy()[0]  # [3, H, W]
    np.savez(out_dir / "export.npz", **arrays)

    meta = {
        "export_version": EXPORT_VERSION,
        "milestone": "M0-float",
        "channels": {k: [v.start, v.stop] for k, v in CHANNEL_GROUPS.items()},
        "latent_count": len(latent_res),
        "latent_resolutions": latent_res,
        "latent_clamp01": args.clamp01,
        "mlp": {
            "layout": "row-major [out,in]; y = w1 @ relu(w0 @ x + b0) + b1",
            "in_dim": 12,
            "hidden_dim": args.hidden,
            "out_dim": 9,
            "activation": "relu",
        },
        "uv_convention": "uv in [0,1], texel center (i+0.5)/N, bilinear, clamp addressing",
        "fixture_seed": FIXTURE_SEED,
        "data": str(args.data),
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None, help="texture-set folder; default: synthetic")
    ap.add_argument("--res", type=int, default=512, help="synthetic set resolution")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--latent-res", type=str, default=None,
                    help="comma list of 4 resolutions; default: source resolution for all")
    ap.add_argument("--clamp01", action="store_true", help="constrain latents to [0,1] (M1 preview)")
    ap.add_argument("--lr-latent", type=float, default=1e-2)
    ap.add_argument("--lr-mlp", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.data is None:
        args.data = HERE / "data" / f"synthetic_{args.res}"
        if not (args.data / "basecolor.png").exists():
            generate(args.data, args.res)

    target_np = load_texture_set(args.data)
    h, w, _ = target_np.shape
    target = torch.from_numpy(target_np).permute(2, 0, 1).unsqueeze(0).to(args.device)

    latent_res = [int(r) for r in args.latent_res.split(",")] if args.latent_res else [h] * 4
    model = NeuralMaterialModel(latent_res, hidden_dim=args.hidden, clamp01=args.clamp01).to(args.device)

    opt = torch.optim.Adam([
        {"params": model.latents.parameters(), "lr": args.lr_latent},
        {"params": model.mlp.parameters(), "lr": args.lr_mlp},
    ])
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(args.iters * 0.6), int(args.iters * 0.85)], gamma=0.3)

    run_name = args.name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = HERE / "output" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={args.device} target={w}x{h} latents={latent_res} out={out_dir}")

    t0 = time.time()
    for it in range(1, args.iters + 1):
        uv = torch.rand(args.batch, 2, device=args.device)
        loss = (model(uv) - sample_texture(target, uv)).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 200 == 0 or it == 1:
            print(f"iter {it:5d}  L1 {loss.item():.5f}  ({time.time() - t0:.1f}s)")

    recon, metrics = evaluate(model, target)
    metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics["iters"] = args.iters
    print(json.dumps(metrics, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_images(recon, target, out_dir)
    export(model, args, latent_res, out_dir)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
