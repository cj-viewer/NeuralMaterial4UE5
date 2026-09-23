# Neural material trainer

The consolidated pipeline (successor of the m0/m1/m2/m3 milestone experiments,
which live on in git history): **4 independently trained BC1 latent mip
pyramids + a shared 12→32→9 FMA MLP**, trained with continuous-LOD sampling
against a box-filtered reference pyramid, packaged together with the source
textures' direct-BC1 encodings so the D3D12 test renderer can load the result
straight from the training output.

## Files

| File | Purpose |
|---|---|
| `data.py` | Texture-set loading (fixed 9-channel layout) + deterministic synthetic test set. |
| `bc1.py` | STE quantization + simulated decode (training), bit-exact pack/unpack/reference decode, mipped DDS writer, offline PCA-fit encoder. |
| `model.py` | `BC1LatentPyramid`, GPU-exact trilinear sampling, `MipMaterialModel`, reference target pyramid. |
| `gold.py` | Torch-free numpy reference (bilinear/trilinear + MLP) for the data contract. |
| `train.py` | Training entry point: train, evaluate, **emit the material package**. |
| `validate.py` | Self-tests + trained-export verification; writes `fixtures.npz`. |
| `prep_cerberus.py` | Renderer's Cerberus PNGs → trainer source layout (constant-white AO). |
| `build_pbr.bat` | D3D12-only build of the test renderer (`../PBR`). |

## The material package (trainer output == renderer input)

`train.py --data <source dir> --out <package dir>` writes one folder that the
renderer loads directly with `-material <package dir>`:

| File | Consumer |
|---|---|
| `albedo/normal/metalness/roughness.dds` | classic path — the **source textures** as direct-BC1 mip chains (PCA-fit encoder; grayscale maps replicated to RGB; needs a power-of-two source) |
| `latent0..3.dds` | neural path — trained BC1 latent mip chains |
| `weights.bin` | MLP weights, 716 packed fp32 (`w0|b0|w1|b1`) |
| `export.npz`, `meta.json`, `fixtures.npz`, `metrics.json`, PNGs | training artifacts / data contract (renderer ignores) |

## Entry scripts (Research/train.bat, validate.bat, test.bat — run from anywhere)

`train.bat` / `validate.bat` forward all arguments to `trainer/train.py` /
`trainer/validate.py`; `test.bat` launches the renderer with
`-datadir <Research>/PBR/data` (the renderer chdirs there itself) and a default
`-material` of `PBR/data/material`. One-time setup: `trainer/prep_cerberus.py`
(`--res`, default 4096) and `trainer/build_pbr.bat`.

### train.bat arguments

| Argument | Default | Meaning |
|---|---|---|
| `--data <dir>` | *(none)* | Source texture-set folder (`basecolor.png`, `normal.png`, plus `arm.png` or `ao/roughness/metallic.png`). Omitted → deterministic synthetic set under `trainer/data/synthetic_<res>`. |
| `--res <n>` | `512` | Synthetic-set resolution (only when `--data` omitted). |
| `--latent-res <n>` | `512` | Uniform resolution of the 4 BC1 latents; power of two. Drives compression (mem ≈ 4 × 4 bpp × 4/3 mips). |
| `--iters <n>` | `20000` | Training iterations. |
| `--batch <n>` | `65536` | UV samples per iteration. |
| `--hidden <n>` | `32` | MLP hidden width; the renderer's shader is fixed to 32 — change both or neither. |
| `--lr-latent <f>` | `1e-2` | Adam LR for latent block parameters. |
| `--lr-mlp <f>` | `2e-3` | Adam LR for the MLP. |
| `--lod-pow <f>` | `2.0` | `lod = U^p · max_lod`; p>1 weights low LODs. Raise when a latent much coarser than the source starves lod0. |
| `--seed <n>` | `0` | torch RNG seed. |
| `--name <s>` | `run_<latent-res>` | Run name (default output folder name). |
| `--out <dir>` | `trainer/output/<name>` | Material package output — the renderer's `-material` target. |
| `--device <s>` | `cuda` if available | torch device. |

Progress: single-line bar on a terminal (periodic lines when redirected) with
iteration count/total, L1 (EMA), it/s, elapsed and ETA.

### test.bat / renderer arguments

| Argument | Default | Meaning |
|---|---|---|
| `-material <dir>` | `PBR\data\material` (from test.bat) | Material package to load; relative paths resolve against your launch directory. |
| `-datadir <dir>` | set by test.bat | Renderer asset root (`PBR/data`); the process chdirs there. |

In-app: `N` classic ↔ neural · `D` B/W ×8 difference · `C` FMA ↔
cooperative-vector backend · LMB/RMB drag, scroll zoom, `F1–F3` lights.
Title bar live-updates mode, MLP backend, material VRAM and PBR-pass GPU time.

### validate.bat arguments

| Argument | Meaning |
|---|---|
| `--selftest` *(default with no args)* | 8 checks: BC1 round-trip, opaque-mode constraint, degenerate/boundary blocks, Pillow independent decode, torch-vs-numpy trilinear, DDS mip layout, PCA-encoder bound. |
| `<package>/export.npz` | Verify a trained package; fixtures recomputed from the packed bitstream alone must match torch (<1e-5). Writes `fixtures.npz`. |

**Renderer controls**: `N` classic ↔ neural, `D` B/W ×8 difference view,
`C` FMA ↔ cooperative-vector MLP backend (needs the preview Agility SDK + DXC
under `PBR/external/` — NuGet `Microsoft.Direct3D.D3D12` 1.717.1-preview and
`Microsoft.Direct3D.DXC` 1.8.2505.32 — Windows Developer Mode, and NVIDIA's
SM 6.9 preview driver; measured 2.4× on the PBR pass at tier 1.1, renders
matching within fp16 rounding). Window title shows mode, MLP backend, material
VRAM (classic vs neural) and the PBR pass GPU time from timestamp queries.

## Pinned conventions (the training-to-runtime data contract, spec §9)

- **Channels**: 0-2 BaseColor RGB (sRGB storage values), 3-5 Normal XYZ, 6 AO,
  7 Roughness, 8 Metallic; all raw storage-space [0,1].
- **UV**: [0,1]², texel centers at `(i+0.5)/N`, bilinear, clamp addressing.
- **LOD**: source-mip units; per-latent `lod_k = clamp(lod + log2(res_k/source_res),
  0, levels_k−1)`; trilinear = decode BC1 per level → bilinear → lerp by `frac(lod)`
  (exactly what GPU hardware does). Training samples `lod = U^p · max_lod`
  (`--lod-pow`, default 2 — low LODs dominate rendering, and a latent coarser than
  the source shares its base level between lod targets).
- **Reference mips**: 2×2 box average; the runtime asset's mipgen must match.
- **BC1**: opaque 4-color mode only; `texel = lerp(C0, C1, w)`, `w ∈ {0,⅓,⅔,1}`;
  exporter enforces `c0 > c1` (endpoint swap + index flip) and pins equal-endpoint
  blocks to index 0.
- **MLP weights**: row-major `[out, in]`; `y = w1 @ relu(w0 @ x + b0) + b1`.
- **Latents**: power-of-two resolutions (mip chains stop at 4²).
- **Training stability**: pass-through clamp + per-step projection to [0,1]
  (a plain clamp permanently kills saturated parameters); endpoints initialized
  apart (0.35/0.65) so index weights get gradients from the start.

## Export (`output/<name>/`, gitignored)

`export.npz` (per-level packed BC1 blocks, fp32 weights, seeded (uv, lod)
fixtures with torch-recorded features/outputs), `meta.json` (the full contract),
`latent0..3.dds` (real mipped BC1, loadable by any DDS consumer including the
test renderer), `metrics.json`, recon/diff PNGs. `validate.py <run>/export.npz`
must PASS before an export is used downstream; it also writes `fixtures.npz`,
the gold reference for D3D12/UE comparisons (spec §11).

## Reference results

See git history for the milestone studies these defaults came from
(M0 float baseline, M1 float-vs-BC1 sweep, M2 trained-vs-naive mips) and
`m3_pbr_visual/` for the renderer-side validation (visual A/B, VRAM/GPU-time
stats, FMA-vs-cooperative-vector MLP backends).
