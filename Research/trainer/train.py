"""Neural material training: 4 BC1 latent mip pyramids + 12->32->9 MLP,
continuous-LOD sampling against a box-filtered reference pyramid.

Run from Research/:
    .venv/Scripts/python.exe trainer/train.py --data <texture-set dir> --latent-res 512
With no --data a deterministic synthetic set is generated (see data.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import bc1
from data import CHANNEL_GROUPS, generate, load_texture_set
from model import MipMaterialModel, build_target_pyramid, sample_levels, texel_center_uv

HERE = Path(__file__).parent
FIXTURE_SEED = 4242
FIXTURE_COUNT = 64
EXPORT_VERSION = 3
EVAL_LODS = [0, 1, 2, 3, 4]


def psnr(mse: float) -> float:
    return float("inf") if mse <= 0 else float(-10.0 * np.log10(mse))


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Progress:
    """Single-line progress bar on a TTY; periodic full lines when redirected."""

    def __init__(self, total: int, refresh: float = 0.25, line_every: float = 5.0):
        self.total = total
        self.t0 = time.time()
        self.last = 0.0
        self.tty = sys.stdout.isatty()
        self.refresh = refresh if self.tty else line_every
        self.loss_ema = None

    def update(self, it: int, loss_fn) -> None:
        # Time-gate BEFORE touching the loss: .item() forces a GPU sync, so it
        # must only happen on refresh ticks, not every iteration.
        now = time.time()
        if now - self.last < self.refresh and it < self.total:
            return
        self.last = now
        loss = float(loss_fn())
        self.loss_ema = loss if self.loss_ema is None else self.loss_ema * 0.8 + loss * 0.2
        frac = it / self.total
        elapsed = now - self.t0
        ips = it / elapsed if elapsed > 0 else 0.0
        eta = (self.total - it) / ips if ips > 0 else 0.0
        filled = int(frac * 30)
        bar = "#" * filled + "-" * (30 - filled)
        msg = (f"[{bar}] {it}/{self.total} {frac * 100:5.1f}% | L1 {self.loss_ema:.5f} | "
               f"{ips:6.1f} it/s | elapsed {_fmt_hms(elapsed)} | ETA {_fmt_hms(eta)}")
        if self.tty:
            print("\r" + msg, end="", flush=True)
        else:
            print(msg, flush=True)

    def close(self) -> None:
        if self.tty:
            print()


@torch.no_grad()
def evaluate(model: MipMaterialModel, target: torch.Tensor, tile: int = 65536):
    """Full-grid reconstruction at lod 0 + PSNR per channel group."""
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


@torch.no_grad()
def evaluate_lods(model: MipMaterialModel, target_pyr: list[torch.Tensor], tile: int = 65536):
    """PSNR against each reference mip on that mip's full texel grid."""
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


def save_images(recon: torch.Tensor, target: torch.Tensor, out_dir: Path) -> None:
    def to_img(t: torch.Tensor) -> Image.Image:
        arr = t.clamp(0, 1).mul(255).add(0.5).byte().cpu().numpy().transpose(1, 2, 0)
        return Image.fromarray(arr)

    views = {"basecolor": slice(0, 3), "normal": slice(3, 6), "arm": slice(6, 9)}
    for name, sl in views.items():
        to_img(target[0, sl]).save(out_dir / f"ref_{name}.png")
        to_img(recon[0, sl]).save(out_dir / f"recon_{name}.png")
        to_img((recon[0, sl] - target[0, sl]).abs() * 4).save(out_dir / f"diff_{name}.png")


CLASSIC_MAPS = {  # package file -> channel slice of the 9-channel source
    "albedo.dds": slice(0, 3),
    "normal.dds": slice(3, 6),
    "metalness.dds": slice(8, 9),
    "roughness.dds": slice(7, 8),
}
WEIGHTS_BIN_FLOATS = 716  # 12*32 + 32 + 9*32 + 9 = 713, padded to a float4 multiple


def export_classic_maps(target: torch.Tensor, out_dir: Path) -> None:
    """Encode the source textures as direct-BC1 mip chains into the package,
    so the renderer's classic path shows the same source the neural model was
    trained on. Grayscale maps are replicated to RGB."""
    res = target.shape[-1]
    if res & (res - 1) or res < 4:
        print(f"classic maps skipped: source resolution {res} is not a power of two")
        return
    for name, sl in CLASSIC_MAPS.items():
        img = target[:, sl]
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        levels = []
        for level in build_target_pyramid(img):
            e0, e1, wmap = bc1.boxfit_bc1_params(level.cpu())
            levels.append(bc1.pack_blocks(e0.numpy(), e1.numpy(), wmap.numpy()))
        bc1.write_dds_mips(out_dir / name, levels, res, res)
    print(f"classic direct-BC1 maps -> {out_dir}")


def export(model: MipMaterialModel, args, out_dir: Path) -> None:
    """Pack every mip level to real BC1, write mipped DDS files, the weights
    and (uv, lod) gold fixtures reproducible from the bitstream alone."""
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

    # Weight stream for the renderer (w0 | b0 | w1 | b1, float32 LE, padded to
    # a float4 multiple); the renderer derives its fp16 LinAlg copy from it.
    packed = np.zeros(WEIGHTS_BIN_FLOATS, dtype="<f4")
    stream = np.concatenate([
        arrays["w0"].reshape(-1), arrays["b0"],
        arrays["w1"].reshape(-1), arrays["b1"],
    ]).astype("<f4")
    packed[: stream.size] = stream
    packed.tofile(out_dir / "weights.bin")

    meta = {
        "export_version": EXPORT_VERSION,
        "milestone": "trainer",
        "mode": "bc1-mip",
        "channels": {k: [v.start, v.stop] for k, v in CHANNEL_GROUPS.items()},
        "latent_count": len(model.pyramids),
        "latent_resolutions": [p.resolutions[0] for p in model.pyramids],
        "latent_mip_resolutions": [p.resolutions for p in model.pyramids],
        "lod_offsets": model.lod_offsets,
        "max_train_lod": args.max_lod,
        "lod_sampling": f"lod = U^{args.lod_pow} * max_lod",
        "mlp": {"layout": "row-major [out,in]; y = w1 @ relu(w0 @ x + b0) + b1",
                "in_dim": 12, "hidden_dim": args.hidden, "out_dim": 9, "activation": "relu"},
        "uv_convention": "uv in [0,1], texel center (i+0.5)/N, bilinear, clamp addressing",
        "lod_convention": ("continuous lod in source-mip units; per-latent lod_k = "
                           "clamp(lod + log2(res_k/source_res), 0, levels_k-1); trilinear = "
                           "decode BC1 -> bilinear per level -> lerp by frac(lod)"),
        "target_mip_filter": "2x2 box average (runtime mipgen must match)",
        "bc1": "opaque 4-color mode only; texel = lerp(C0, C1, w), w in {0,1/3,2/3,1}",
        "fixture_seed": FIXTURE_SEED,
        "data": str(args.data),
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None, help="texture-set folder; default: synthetic")
    ap.add_argument("--res", type=int, default=512, help="synthetic set resolution")
    ap.add_argument("--latent-res", type=int, default=512, help="power-of-two uniform latent resolution")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr-latent", type=float, default=1e-2)
    ap.add_argument("--lr-mlp", type=float, default=2e-3)
    ap.add_argument("--lod-pow", type=float, default=2.0,
                    help="lod = U^p * max_lod; p>1 biases sampling toward lod 0 "
                         "(low LODs dominate rendering, and a latent coarser than "
                         "the source shares its base level between lod targets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="material package output folder (default: trainer/output/<name>); "
                         "point the renderer at it with -material <folder>")
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
    target_pyr = build_target_pyramid(target)
    args.max_lod = float(len(target_pyr) - 1)

    model = MipMaterialModel([args.latent_res] * 4, h, args.hidden).to(args.device)
    opt = torch.optim.Adam([
        {"params": [p for pyr in model.pyramids for p in pyr.parameters()], "lr": args.lr_latent},
        {"params": model.mlp.parameters(), "lr": args.lr_mlp},
    ])
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(args.iters * 0.6), int(args.iters * 0.85)], gamma=0.3)

    run_name = args.name or f"run_{args.latent_res}"
    out_dir = args.out if args.out else HERE / "output" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={args.device} target={w}x{h} latent={args.latent_res} "
          f"levels={len(model.pyramids[0].levels)} max_lod={args.max_lod}")
    print(f"training {args.iters} iterations, batch {args.batch} "
          f"(~{args.iters * args.batch / 1e6:.0f}M samples) -> {out_dir}")

    progress = Progress(args.iters)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        uv = torch.rand(args.batch, 2, device=args.device)
        lod = torch.rand(args.batch, device=args.device) ** args.lod_pow * args.max_lod
        tgt = sample_levels(target_pyr, uv, lod)
        loss = (model(uv, lod) - tgt).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # Projected-gradient step: a plain clamp inside the forward would zero
        # gradients outside [0,1] and permanently kill saturated parameters.
        with torch.no_grad():
            for p in model.pyramids.parameters():
                p.clamp_(0.0, 1.0)
        sched.step()
        progress.update(it, loss.item)
    progress.close()

    recon, metrics = evaluate(model, target)
    metrics.update(evaluate_lods(model, target_pyr))
    metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics["iters"] = args.iters
    metrics["latent_res"] = args.latent_res
    print(json.dumps(metrics, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_images(recon, target, out_dir)
    export(model, args, out_dir)
    export_classic_maps(target, out_dir)
    print(f"material package -> {out_dir}")


if __name__ == "__main__":
    main()
