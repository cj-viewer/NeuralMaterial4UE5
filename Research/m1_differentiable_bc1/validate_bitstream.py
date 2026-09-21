"""M1 bitstream validation (spec 10/M1 + 11.2 test A prerequisites).

Self-tests (no trained run needed):
    .venv/Scripts/python.exe m1_differentiable_bc1/validate_bitstream.py --selftest
Validate a trained export:
    .venv/Scripts/python.exe m1_differentiable_bc1/validate_bitstream.py <run>/export.npz
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import M0_DIR  # noqa: F401  (installs sys.path for m0 imports)
from fixtures import bilinear_clamp, mlp_forward  # m0 gold-reference helpers
import bc1

PILLOW_TOLERANCE = 2  # 8-bit LSBs; BC1 interpolation rounding is implementation-defined


def _parse_endpoints(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = blocks.reshape(-1, 8).astype(np.uint32)
    return b[:, 0] | (b[:, 1] << 8), b[:, 2] | (b[:, 3] << 8)


def _pillow_decode_dds(path_or_bytes) -> np.ndarray:
    img = Image.open(path_or_bytes)
    return np.asarray(img.convert("RGB"), dtype=np.int32)  # [H, W, 3]


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def selftest() -> bool:
    ok = True
    rng = np.random.default_rng(7)
    res = 64
    blocks_n = res // 4

    # 1. torch simulated decode == numpy pack -> reference decode (bit-level path)
    tex = bc1.BC1LatentTexture(res)
    with torch.no_grad():
        tex.e0.copy_(torch.from_numpy(rng.random((blocks_n, blocks_n, 3), dtype=np.float32)))
        tex.e1.copy_(torch.from_numpy(rng.random((blocks_n, blocks_n, 3), dtype=np.float32)))
        tex.w.copy_(torch.from_numpy(rng.random((res, res), dtype=np.float32)))
        sim = tex.decode()[0].numpy()
    packed = bc1.pack_blocks(tex.e0.detach().numpy(), tex.e1.detach().numpy(), tex.w.detach().numpy())
    ref = bc1.decode_blocks(packed, res, res)
    err = np.abs(sim - ref).max()
    ok &= check("simulated decode == packed reference decode", err < 1e-6, f"max err {err:.2e}")

    # 2. opaque-mode constraint: every packed block has c0 >= c1
    c0, c1 = _parse_endpoints(packed)
    ok &= check("all packed blocks satisfy c0 >= c1", bool((c0 >= c1).all()),
                f"{int((c0 < c1).sum())} violations, {int((c0 == c1).sum())} equal-endpoint blocks")

    # 3. degenerate blocks: identical endpoints must decode to that color (no 3-color black)
    e = np.full((1, 1, 3), 0.5, dtype=np.float32)
    w = rng.random((4, 4), dtype=np.float32)
    packed_d = bc1.pack_blocks(e, e, w)
    dec_d = bc1.decode_blocks(packed_d, 4, 4)
    expect = bc1.rgb565_to_float(bc1.endpoints_to_565(e))[0, 0][:, None, None]
    err = np.abs(dec_d - expect).max()
    ok &= check("degenerate (equal-endpoint) block decodes to endpoint color", err < 1e-6,
                f"max err {err:.2e}")

    # 4. endpoint boundary values 0.0 / 1.0 survive the round trip
    e0 = np.zeros((1, 1, 3), dtype=np.float32)
    e1 = np.ones((1, 1, 3), dtype=np.float32)
    packed_b = bc1.pack_blocks(e0, e1, np.zeros((4, 4), dtype=np.float32))
    dec_b = bc1.decode_blocks(packed_b, 4, 4)
    err = np.abs(dec_b - 0.0).max()  # w=0 -> endpoint0 = black
    ok &= check("boundary endpoints (0 and 1)", err < 1e-6, f"max err {err:.2e}")

    # 5. independent decoder: Pillow reads our DDS and agrees within tolerance
    buf = io.BytesIO()
    bc1.write_dds(buf, packed, res, res)
    buf.seek(0)
    pil = _pillow_decode_dds(buf)  # [H, W, 3]
    ours = np.round(ref.transpose(1, 2, 0) * 255.0).astype(np.int32)
    diff = np.abs(pil - ours)
    ok &= check(f"Pillow DDS decode within {PILLOW_TOLERANCE} LSB", int(diff.max()) <= PILLOW_TOLERANCE,
                f"max diff {int(diff.max())}, mean {diff.mean():.4f}")
    return ok


def validate_export(export_path: Path) -> bool:
    ok = True
    run_dir = export_path.parent
    meta = json.loads((run_dir / "meta.json").read_text())
    data = np.load(export_path)
    if meta.get("mode") != "bc1":
        print("export is not a bc1 run; nothing to validate")
        return False

    latents = []
    for i in range(meta["latent_count"]):
        res = meta["latent_resolutions"][i]
        blocks = data[f"latent{i}_blocks"]
        decoded = bc1.decode_blocks(blocks, res, res)
        err = np.abs(decoded - data[f"latent{i}_decoded"]).max()
        ok &= check(f"latent{i} blocks -> decode matches stored decode", err < 1e-6, f"max err {err:.2e}")

        pil = _pillow_decode_dds(run_dir / f"latent{i}.dds")
        ours = np.round(decoded.transpose(1, 2, 0) * 255.0).astype(np.int32)
        diff = int(np.abs(pil - ours).max())
        ok &= check(f"latent{i}.dds Pillow cross-check", diff <= PILLOW_TOLERANCE, f"max diff {diff} LSB")
        latents.append(decoded)

    features = np.concatenate([bilinear_clamp(l, data["fixture_uv"]) for l in latents], axis=-1)
    output = mlp_forward(features, data["w0"], data["b0"], data["w1"], data["b1"])
    feat_err = np.abs(features - data["fixture_features"]).max()
    out_err = np.abs(output - data["fixture_output"]).max()
    ok &= check("fixture latent samples (numpy from packed bits vs torch)", feat_err < 1e-5,
                f"max err {feat_err:.2e}")
    ok &= check("fixture MLP outputs", out_err < 1e-5, f"max err {out_err:.2e}")

    np.savez(run_dir / "fixtures.npz", uv=data["fixture_uv"], features=features, output=output,
             w0=data["w0"], b0=data["b0"], w1=data["w1"], b1=data["b1"])
    print(f"fixtures -> {run_dir / 'fixtures.npz'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=Path, nargs="?")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    ok = True
    if a.selftest or a.export is None:
        print("== BC1 bitstream self-tests ==")
        ok &= selftest()
    if a.export is not None:
        print(f"== validate {a.export} ==")
        ok &= validate_export(a.export)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
