# M2 — Trained BC1 mip pyramids + continuous LOD

Milestone M2 from the spec (§5.5, §10): every latent mip level is an
**independently trainable** set of BC1 block parameters; training samples a
continuous LOD and matches a filtered reference target, so runtime mip/trilinear
filtering behaves exactly as trained. Method follows the Weinreich BCF /
Belcour BCF1 mip approach.

## Conventions (pinned for the data contract, spec §9)

- **Trilinear = what the GPU does with BC1**: decode the two mip levels,
  bilinear-filter each (texel centers `(i+0.5)/N`, clamp), lerp by `frac(lod)`.
- **LOD domain**: source-mip units, `lod ~ U(0, max_lod)` during training with
  `max_lod = log2(source_res / 4)` (both pyramids stop at 4², the BC1 block size).
- **Per-latent LOD mapping**: a latent at a different resolution than the source
  uses `lod_k = clamp(lod + log2(res_k / source_res), 0, levels_k − 1)`.
- **Reference mip filter**: 2×2 box average. The runtime asset's mipgen must match
  (generate mips in the exporter rather than trusting engine mipgen defaults).
- **Power-of-two latents only**: non-pow2 resolutions (e.g. 384) hit partial BC1
  blocks in deep mips; deferred until the UE asset path needs them.

## Files

| File | Purpose |
|---|---|
| `pyramid.py` | `BC1LatentPyramid`, torch trilinear sampling, numpy gold trilinear, PCA-fit BC1 encoder for the naive baseline. |
| `train.py` | Continuous-LOD training; `--base-only` trains lod 0 only and then fills mips naively (decode → box downsample → PCA re-encode) as the A/B baseline. |
| `validate_mips.py` | Self-tests + trained-export validation; writes `fixtures.npz` (uv, lod, features, outputs) for M3. |

## Validation (all must PASS)

Self-tests: torch trilinear == numpy trilinear bit-level (incl. LOD clamp
boundaries); DDS mip-chain header/payload layout; Pillow reads the mipped DDS;
PCA-fit re-encode error stays within the inherent naive-re-encode bound
(interior-palette blocks cannot round-trip — that loss is the point of the baseline).

Export validation: every mip level's packed blocks decode-match; fixture
trilinear samples + MLP outputs reproduce from the packed bitstream alone (<1e-5).

## Results

Reference runs (source 512², 20k iters, RTX 5060 Ti). `psnr_lodN` = full-grid PSNR (dB)
against reference mip N. `naive` = base-only training + decode/box-downsample/PCA
re-encode mips; `mip` = trained mips with `lod = U^p · max_lod` sampling.

| run | lod0 | lod1 | lod2 | lod3 | lod4 |
|---|---|---|---|---|---|
| naive_512 | **39.28** | 31.33 | 31.24 | 30.17 | 29.20 |
| mip_512 (p=2) | 38.66 | **36.67** | **35.88** | 34.54 | **33.23** |
| mip_512 (p=1, uniform) | 37.20 | 36.33 | 35.63 | **35.01** | 32.95 |
| naive_256 | **33.42** | 33.17 | 26.56 | 26.51 | 27.16 |
| mip_256 (p=2) | 29.23 | **35.83** | **35.57** | **34.64** | **33.03** |
| mip_256 (p=4) | 31.11 | 33.98 | 34.87 | 34.55 | 32.97 |
| mip_256 (p=1, uniform) | 27.59 | 36.85 | 35.80 | 34.73 | 33.00 |

Findings:

1. **Same-resolution latents (512, offset 0): trained mips are a near-Pareto win.**
   −0.6 dB at lod0 buys +4–5 dB at every other LOD versus naive exporter mips.
2. **Coarser latents (256, offset −1) share their base level between the lod 0
   and lod 1 targets**, so lod0 quality is capped below the naive baseline and the
   `--lod-pow` bias only trades lod0 against lod1 (p=1→4: lod0 27.6→31.1, lod1
   36.9→34.0). When lod0 fidelity is the priority, keep latent res == source res;
   otherwise pick p per asset. This is a training-profile parameter (spec §7.3).
3. Naive re-encoded mips collapse from lod2 on (26–31 dB) — deep mips of decoded
   BC1 content do not survive re-quantization, which is the paper's argument for
   trained mip pyramids in the first place.

## Out of scope

- Anisotropic filtering, SampleGrad LOD matching → M3/M7.
- Non-pow2 latent mip chains (partial blocks) → UE asset path.
