# M0 — Float latent + MLP baseline (pure Python correctness)

Milestone M0 from the spec (`Docs/UE5_Neural_Material_Plugin_Development_Spec.docx`, §10):
train **unquantized float** latent textures plus a small MLP to reconstruct a single
PBR material set, and prove the pipeline is trainable and reconstructs correctly.
No BC1 simulation (M1), no mip/LOD sampling (M2), no D3D12 (M3).

## Representation (spec §5.2)

- 4 RGB latent textures (float, per-texture resolution configurable) → bilinear sample
  at UV → concat = 12-dim feature.
- MLP `12 -> 32 -> 9`, ReLU hidden, linear output.
- 9 target channels, fixed semantics: BaseColor RGB (0-2), Normal XYZ (3-5),
  AO (6), Roughness (7), Metallic (8). All targets are raw storage-space values in [0,1]
  (BaseColor stays sRGB-encoded; no color-space conversion in M0 — spec §12).
- Loss: L1 on randomly sampled UVs (base mip only in M0).

## Sampling conventions (pinned for the data contract, spec §9)

- UV in [0,1]², texel center at `(i + 0.5) / N` (`grid_sample(align_corners=False)`,
  matching D3D texel addressing).
- Address mode: **clamp** (recorded in export metadata).
- Weight layout: row-major `[out_dim, in_dim]`, `y = W1 @ relu(W0 @ x + b0) + b1`.

## Files

| File | Purpose |
|---|---|
| `synth_data.py` | Deterministic procedural PBR set (basecolor/normal/arm PNGs) so the experiment is self-contained. |
| `dataset.py` | Load a texture-set folder into the 9-channel target tensor. |
| `model.py` | Latent textures + MLP. |
| `train.py` | Training loop, eval (PSNR per channel group), image dumps, export. |
| `fixtures.py` | **Torch-free** numpy reimplementation of sample+MLP; cross-checks the exported `.npz` against outputs recorded at training time and writes gold fixtures for M3/UE (spec §11.1). |

## Run

```bash
# from Research/ (venv with torch, numpy, pillow)
.venv/Scripts/python.exe m0_float_latent_baseline/train.py --res 512 --iters 4000
.venv/Scripts/python.exe m0_float_latent_baseline/fixtures.py m0_float_latent_baseline/output/<run>/export.npz
```

`train.py` with no `--data` generates `data/synthetic_<res>/` on first use. To train on a real
material, pass `--data <folder>` containing `basecolor.png`, `normal.png` and either `arm.png`
(R=AO, G=Roughness, B=Metallic) or separate `ao.png` / `roughness.png` / `metallic.png`.

## Outputs (`output/<run>/`, gitignored)

- `metrics.json` — L1 + PSNR (overall and per channel group) at texel centers.
- `recon_*.png` / `ref_*.png` / `diff_*.png` — visual comparison.
- `export.npz` + `meta.json` — latents, MLP weights, channel map, sampling conventions,
  and 64 seeded fixture UVs with the model outputs recorded by torch.
- `fixtures.npz` (from `fixtures.py`) — numpy-verified gold reference values.

## Acceptance for M0

1. Loss decreases and training is stable at default settings. ✅
2. Full-resolution reconstruction is visually faithful; PSNR reported per channel group. ✅
3. `fixtures.py` reproduces the recorded torch outputs from the exported arrays alone
   (max abs error < 1e-5) — proves the export layout is complete and unambiguous. ✅

Reference run (`--res 512 --iters 4000`, defaults, RTX 5060 Ti, torch 2.11.0+cu128, 4.9 s):

| Metric | Value |
|---|---|
| L1 overall | 4.69e-4 |
| PSNR overall | 60.1 dB |
| PSNR basecolor / normal | 63.7 / 58.8 dB |
| PSNR ao / roughness / metallic | 60.3 / 58.5 / 59.3 dB |
| fixtures.py max abs err | 2.4e-7 (PASS) |

Note: full-resolution float latents have no compression pressure, so near-lossless PSNR is
the expected M0 outcome; the numbers become meaningful comparisons once M1 adds BC1
quantization and lower-resolution latent layouts.

## Deliberately out of scope

- Latent range constraint / BC1 quantization (STE), exporter bitstream → **M1**
  (`--clamp01` exists as a forward-compatibility toggle, default off).
- Latent mip pyramids, continuous LOD, filtered reference targets → **M2**.
- HLSL/D3D12 comparison → **M3** (consumes `fixtures.npz`).
