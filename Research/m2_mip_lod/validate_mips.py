"""M2 validation: trilinear gold reference + mip-chain bitstream checks.

    .venv/Scripts/python.exe m2_mip_lod/validate_mips.py --selftest
    .venv/Scripts/python.exe m2_mip_lod/validate_mips.py <run>/export.npz
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

from pyramid import (BC1LatentPyramid, boxfit_bc1_params, numpy_sample_levels,
                     sample_levels)

import bc1
from fixtures import mlp_forward  # m0 gold helper (sys.path via pyramid)

PILLOW_TOLERANCE = 2


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def selftest() -> bool:
    ok = True
    torch.manual_seed(3)
    rng = np.random.default_rng(3)

    # 1. torch trilinear == numpy trilinear on a random 3-level pyramid
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
    ok &= check("torch trilinear == numpy trilinear (incl. lod clamp/boundaries)",
                err < 1e-6, f"max err {err:.2e}")

    # 2. DDS mip chain: header fields + per-level payload round trip
    levels, res = [], 16
    for r in (16, 8, 4):
        e0 = rng.random((r // 4, r // 4, 3), dtype=np.float32)
        e1 = rng.random((r // 4, r // 4, 3), dtype=np.float32)
        w = rng.random((r, r), dtype=np.float32)
        levels.append(bc1.pack_blocks(e0, e1, w))
    buf = io.BytesIO()
    bc1.write_dds_mips(buf, levels, res, res)
    raw = buf.getvalue()
    flags, height, width, _, _, mipcount = struct.unpack_from("<6I", raw, 8)
    hdr_ok = (flags & 0x20000) and mipcount == 3 and height == width == 16
    payload = raw[128:]
    sizes = [lvl.nbytes for lvl in levels]
    slices_ok = all(
        payload[sum(sizes[:i]):sum(sizes[:i + 1])] == levels[i].tobytes()
        for i in range(3)) and len(payload) == sum(sizes)
    ok &= check("DDS mip header + level payload layout", bool(hdr_ok and slices_ok),
                f"mipcount={mipcount}")
    pil = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.int32)
    ours = np.round(bc1.decode_blocks(levels[0], res, res).transpose(1, 2, 0) * 255).astype(np.int32)
    diff = int(np.abs(pil - ours).max())
    ok &= check("Pillow reads mipped DDS base level", diff <= PILLOW_TOLERANCE, f"max diff {diff} LSB")

    # 3. PCA-fit BC1 encoder sanity. Re-encoding BC1 content is NOT lossless by
    # design: when a block only uses interior palette entries, the recovered
    # endpoints are interpolated colors and the original weights land between
    # the {0,1/3,2/3,1} steps (error up to |C1-C0|/9 on such texels). We assert
    # the error stays within that inherent bound — this weakness of naive
    # re-encoding is exactly what the naive-mip baseline demonstrates.
    src = bc1.decode_blocks(levels[0], res, res)
    e0f, e1f, wf = boxfit_bc1_params(torch.from_numpy(src).unsqueeze(0))
    repacked = bc1.pack_blocks(e0f.numpy(), e1f.numpy(), wf.numpy())
    redec = bc1.decode_blocks(repacked, res, res)
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
            blocks = data[f"latent{i}_mip{j}_blocks"]
            decoded.append(bc1.decode_blocks(blocks, r, r))
        decoded_per_latent.append(decoded)
        pil = np.asarray(Image.open(run_dir / f"latent{i}.dds").convert("RGB"), dtype=np.int32)
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
        print("== M2 mip/trilinear self-tests ==")
        ok &= selftest()
    if a.export is not None:
        print(f"== validate {a.export} ==")
        ok &= validate_export(a.export)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
