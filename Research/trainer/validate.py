"""Validation for the training pipeline: BC1 bitstream and trilinear
self-tests (no trained run needed), plus trained-export verification that
recomputes the recorded fixtures from the packed bitstream alone.

    .venv/Scripts/python.exe trainer/validate.py --selftest
    .venv/Scripts/python.exe trainer/validate.py <run>/export.npz
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import bc1
from gold import bilinear_clamp, mlp_forward, numpy_sample_levels
from model import BC1LatentPyramid, sample_levels

PILLOW_TOLERANCE = 2  # 8-bit LSBs; BC1 interpolation rounding is implementation-defined


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def _parse_endpoints(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = blocks.reshape(-1, 8).astype(np.uint32)
    return b[:, 0] | (b[:, 1] << 8), b[:, 2] | (b[:, 3] << 8)


def _pillow_decode_dds(path_or_bytes) -> np.ndarray:
    img = Image.open(path_or_bytes)
    return np.asarray(img.convert("RGB"), dtype=np.int32)  # [H, W, 3]


def selftest() -> bool:
    ok = True
    torch.manual_seed(3)
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
    dec_d = bc1.decode_blocks(bc1.pack_blocks(e, e, w), 4, 4)
    expect = bc1.rgb565_to_float(bc1.endpoints_to_565(e))[0, 0][:, None, None]
    err = np.abs(dec_d - expect).max()
    ok &= check("degenerate (equal-endpoint) block decodes to endpoint color", err < 1e-6,
                f"max err {err:.2e}")

    # 4. endpoint boundary values 0.0 / 1.0 survive the round trip
    e0 = np.zeros((1, 1, 3), dtype=np.float32)
    e1 = np.ones((1, 1, 3), dtype=np.float32)
    dec_b = bc1.decode_blocks(bc1.pack_blocks(e0, e1, np.zeros((4, 4), dtype=np.float32)), 4, 4)
    err = np.abs(dec_b - 0.0).max()  # w=0 -> endpoint0 = black
    ok &= check("boundary endpoints (0 and 1)", err < 1e-6, f"max err {err:.2e}")

    # 5. independent decoder: Pillow reads our single-level DDS within tolerance
    buf = io.BytesIO()
    bc1.write_dds(buf, packed, res, res)
    buf.seek(0)
    pil = _pillow_decode_dds(buf)
    ours = np.round(ref.transpose(1, 2, 0) * 255.0).astype(np.int32)
    diff = int(np.abs(pil - ours).max())
    ok &= check(f"Pillow DDS decode within {PILLOW_TOLERANCE} LSB", diff <= PILLOW_TOLERANCE,
                f"max diff {diff}")

    # 6. torch trilinear == numpy trilinear (incl. lod clamp/boundaries)
    pyr = BC1LatentPyramid(16)
    decoded_t = [d.detach() for d in pyr.decode_all()]
    decoded_n = [d[0].numpy() for d in decoded_t]
    uv = rng.random((256, 2), dtype=np.float32)
    lod = np.concatenate([np.zeros(64, np.float32),
                          np.full(64, 2.0, np.float32),
                          (rng.random(128, dtype=np.float32) * 3.0 - 0.5).astype(np.float32)])
    t = sample_levels(decoded_t, torch.from_numpy(uv), torch.from_numpy(lod)).numpy()
    n = numpy_sample_levels(decoded_n, uv, lod)
    err = np.abs(t - n).max()
    ok &= check("torch trilinear == numpy trilinear", err < 1e-6, f"max err {err:.2e}")

    # 7. DDS mip chain: header fields + per-level payload round trip + Pillow base
    levels = []
    for r in (16, 8, 4):
        le0 = rng.random((r // 4, r // 4, 3), dtype=np.float32)
        le1 = rng.random((r // 4, r // 4, 3), dtype=np.float32)
        lw = rng.random((r, r), dtype=np.float32)
        levels.append(bc1.pack_blocks(le0, le1, lw))
    buf = io.BytesIO()
    bc1.write_dds_mips(buf, levels, 16, 16)
    raw = buf.getvalue()
    flags, height, width, _, _, mipcount = struct.unpack_from("<6I", raw, 8)
    hdr_ok = (flags & 0x20000) and mipcount == 3 and height == width == 16
    payload = raw[128:]
    sizes = [lvl.nbytes for lvl in levels]
    slices_ok = all(
        payload[sum(sizes[:i]):sum(sizes[:i + 1])] == levels[i].tobytes()
        for i in range(3)) and len(payload) == sum(sizes)
    pil = _pillow_decode_dds(io.BytesIO(raw))
    ours = np.round(bc1.decode_blocks(levels[0], 16, 16).transpose(1, 2, 0) * 255).astype(np.int32)
    diff = int(np.abs(pil - ours).max())
    ok &= check("DDS mip header + payload layout + Pillow base level",
                bool(hdr_ok and slices_ok) and diff <= PILLOW_TOLERANCE,
                f"mipcount={mipcount}, base diff {diff}")

    # 8. PCA-fit encoder sanity. Re-encoding BC1 content is NOT lossless by
    # design: when a block only uses interior palette entries the recovered
    # endpoints are interpolated colors and the original weights land between
    # the {0,1/3,2/3,1} steps (error up to |C1-C0|/9 on such texels).
    src = bc1.decode_blocks(levels[0], 16, 16)
    e0f, e1f, wf = bc1.boxfit_bc1_params(torch.from_numpy(src).unsqueeze(0))
    redec = bc1.decode_blocks(bc1.pack_blocks(e0f.numpy(), e1f.numpy(), wf.numpy()), 16, 16)
    max_err, mean_err = np.abs(redec - src).max(), np.abs(redec - src).mean()
    ok &= check("PCA-fit encoder re-encode error within inherent bound",
                max_err < 0.12 and mean_err < 5e-3,
                f"max {max_err:.4f} (bound 0.12), mean {mean_err:.5f}")
    return ok


def validate_export(export_path: Path) -> bool:
    ok = True
    run_dir = export_path.parent
    meta = json.loads((run_dir / "meta.json").read_text())
    data = np.load(export_path)

    decoded_per_latent = []
    for i, mips in enumerate(meta["latent_mip_resolutions"]):
        decoded = []
        for j, r in enumerate(mips):
            decoded.append(bc1.decode_blocks(data[f"latent{i}_mip{j}_blocks"], r, r))
        decoded_per_latent.append(decoded)
        pil = _pillow_decode_dds(run_dir / f"latent{i}.dds")
        ours = np.round(decoded[0].transpose(1, 2, 0) * 255).astype(np.int32)
        diff = int(np.abs(pil - ours).max())
        ok &= check(f"latent{i}.dds ({len(mips)} mips) Pillow base-level cross-check",
                    diff <= PILLOW_TOLERANCE, f"max diff {diff} LSB")

    uv, lod = data["fixture_uv"], data["fixture_lod"]
    feats = np.concatenate([
        numpy_sample_levels(dec, uv, lod + np.float32(off))
        for dec, off in zip(decoded_per_latent, meta["lod_offsets"])
    ], axis=-1)
    output = mlp_forward(feats, data["w0"], data["b0"], data["w1"], data["b1"])
    feat_err = np.abs(feats - data["fixture_features"]).max()
    out_err = np.abs(output - data["fixture_output"]).max()
    ok &= check("fixture trilinear latent samples (numpy from packed bits vs torch)",
                feat_err < 1e-5, f"max err {feat_err:.2e}")
    ok &= check("fixture MLP outputs", out_err < 1e-5, f"max err {out_err:.2e}")

    np.savez(run_dir / "fixtures.npz", uv=uv, lod=lod, features=feats, output=output,
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
        print("== trainer self-tests ==")
        ok &= selftest()
    if a.export is not None:
        print(f"== validate {a.export} ==")
        ok &= validate_export(a.export)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
