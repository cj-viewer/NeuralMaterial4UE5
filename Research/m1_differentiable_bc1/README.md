# M1 — Differentiable BC1 latents + bitstream exporter

Milestone M1 from the spec (§10): store/train the latents as **simulated BC1**
(STE-quantized endpoints + indices, spec §5.3–5.4) so the MLP trains against exactly
what hardware BC1 decode will produce, and export real BC1/DDS bitstreams that
independent decoders read correctly. Method follows AMD NTBC (`Docs/2407.09543v3.pdf` §3)
with the spec's STE-round parameterization.

## Representation

Per 4×4 block: continuous endpoints `e0, e1 ∈ [0,1]³`, per-texel weight `w ∈ [0,1]`.
Quantization (STE): endpoints → RGB565 (`round(x·31)/31`, `round(x·63)/63`), weight →
`{0, 1/3, 2/3, 1}`; decode `texel = lerp(C0, C1, w)` — mathematically identical to the
BC1 opaque 4-color palette. Export maps weight levels to real 2-bit indices
(`w=0→0, 1→1, 1/3→2, 2/3→3`), enforces `c0 > c1` (swap + `w→1−w`), and forces
index 0 in equal-endpoint blocks so the 3-color/alpha mode is never selected.

## Files

| File | Purpose |
|---|---|
| `bc1.py` | Torch STE quantize + simulated decode; numpy pack/unpack/reference decode; DDS writer. |
| `common.py` | Imports the M0 modules (dataset/model/eval stay single-source) and defines the BC1 latent model. |
| `train.py` | `--mode bc1` (simulated BC1) or `--mode float` (clamp01 control for the A/B), uniform `--latent-res`. |
| `validate_bitstream.py` | Self-tests + trained-export validation (see below). |
| `run_sweep.py` | Resolution sweep float vs bc1, compression/quality summary table. |

## Validation (all must PASS)

Self-tests (`--selftest`):
1. Torch simulated decode == numpy pack→reference decode (bit-level path), max err < 1e-6.
2. Every packed block satisfies `c0 >= c1` (opaque mode).
3. Equal-endpoint blocks decode to the endpoint color (no 3-color black).
4. Boundary endpoint values 0/1 survive the round trip.
5. **Pillow's independent DDS/DXT1 decoder** agrees with ours within 2 LSB (8-bit).

Trained-export validation (`validate_bitstream.py <run>/export.npz`):
packed bytes → numpy decode == stored decode (bit-exact); `latent*.dds` re-read via
Pillow; fixture latent samples + MLP outputs recomputed from the **packed bitstream
alone** match the torch-recorded values < 1e-5. `fixtures.npz` from a bc1 run is the
gold reference for M3's GPU-hardware-decode comparison (spec 11.2 test B).

## Sweep

`run_sweep.py` trains float + bc1 at latent resolutions 512/384/256/192/128
(source 512², 6000 iters) and writes `output/sweep/results.json` +
`sweep_summary.md`: bits per source texel (4×BC1 + fp32 MLP), compression ratio
vs raw RGB8 (72 bpp) and vs a traditional BC7/BC5 set (24 bpp), PSNR per mode,
and the **quantization cost** = PSNR(float) − PSNR(BC1) at equal resolution
(spec 11.2 test A).

Results: see `output/sweep/sweep_summary.md` (regenerate with `run_sweep.py`) and the
visual report: https://claude.ai/artifact/Fo8GLZCBWG2kfTM5fzKyLA

Reference sweep (20k iters, RTX 5060 Ti, projected-gradient fix applied):

| latent res | bpp | vs BC 24bpp | PSNR float | PSNR BC1 | quant cost (dB) |
|---|---|---|---|---|---|
| 4×512² | 16.09 | 1.5x | 61.86 | 39.03 | 22.83 |
| 4×384² | 9.09 | 2.6x | 43.78 | 37.86 | 5.93 |
| 4×256² | 4.09 | 5.9x | 37.73 | 33.03 | 4.70 |
| 4×192² | 2.34 | 10.3x | 36.59 | 32.19 | 4.40 |
| 4×128² | 1.09 | 22.1x | 32.29 | 28.97 | 3.32 |

Training-stability note: a plain `clamp(0,1)` on latent parameters zeroes gradients
outside the range and permanently kills saturated parameters (non-monotonic sweep,
bc1_512 stuck at ~31.8 dB). Fixed with pass-through clamp + per-step projection to
[0,1] and spread endpoint init (0.35/0.65); bc1_512 recovered to 39.0 dB.

## Out of scope (later milestones)

- Latent mip pyramids, continuous LOD, filtered targets → M2.
- HLSL/D3D12, hardware BC1 decode comparison → M3 (consumes `fixtures.npz` + `latent*.dds`).
- BC1 3-color/alpha mode — excluded by design (spec §5.4).
