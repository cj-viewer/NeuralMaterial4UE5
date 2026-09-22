# M3a — Visual validation in a real PBR renderer

Renders our neural material (4 BC1 latent mip chains + 12→32→9 FMA MLP) inside
[Nadrin/PBR](https://github.com/Nadrin/PBR) (MIT), a single-file-per-backend
PBR+IBL renderer, using its **D3D12 backend** — plain HLSL SM 5.0, hardware BC1
decode + trilinear/aniso filtering, no Cooperative Vectors. This is the first
GPU consumer of the M1/M2 data contract and doubles as groundwork for M3
(fixtures vs GPU comparison).

The Intel TSNC sample stays a reference only: it requires Cooperative Vector
drivers, a preview Agility SDK and Developer Mode.

## Layout

| Piece | Where |
|---|---|
| Renderer source, **vendored with our modifications** | `Research/PBR/` — based on [Nadrin/PBR](https://github.com/Nadrin/PBR) @ `cd61a5d5` (MIT, `COPYING.txt`; Cerberus by Andrew Maximov; `environment.hdr` from HDRLabs sIBL, CC-BY-NC-SA 3.0) |
| D3D12-only build script (no Vulkan SDK) | `build_pbr.bat` |
| Cerberus → trainer data prep | `prep_cerberus.py` |
| Direct-BC1 classic baseline (regenerable, gitignored) | `make_classic_bc1.py` → `PBR/data/textures_bc1/` |
| Trained run → renderer install | `export_to_renderer.py` |

`PBR/build/` and `PBR/data/textures_bc1/` are gitignored (compiled output and
regenerable data); everything else in `Research/PBR/`, including our renderer
modifications and the installed neural model in `PBR/data/neural/`, is committed.

## Pipeline

```bash
# 1. prepare data + train (from Research/)
.venv/Scripts/python.exe m3_pbr_visual/prep_cerberus.py --res 1024
.venv/Scripts/python.exe m2_mip_lod/train.py --data m3_pbr_visual/data/cerberus_1024 \
    --latent-res 512 --iters 20000 --name cerberus_512
# 2. install into the renderer
.venv/Scripts/python.exe m3_pbr_visual/export_to_renderer.py m2_mip_lod/output/cerberus_512
# 3. build + run
m3_pbr_visual/build_pbr.bat
cd PBR/data && ../build/pbr_d3d12.exe -d3d12
```

**Controls**: `N` toggles neural material ↔ classic per-map textures; `D` toggles the
**B/W difference view** (per-pixel |neural − classic| shaded color, luminance ×8,
computed in one pixel-shader invocation — skybox pass is unaffected and stays as
context). LMB-drag orbits, RMB-drag rotates the model, scroll zooms, F1–F3 lights.

**Stats (window title, live)**: material texture VRAM — classic maps
(GetResourceAllocationInfo incl. mips) vs neural (4 BC1 latent mip chains +
weights CBV) — and the PBR draw's GPU time from a per-frame timestamp-query pair
(EMA-smoothed; indicative, short-window).

**Classic baseline**: `make_classic_bc1.py --res <N>` encodes the four Cerberus
maps as direct-BC1 mip chains (our PCA-fit encoder) into `PBR/data/textures_bc1/`;
the renderer prefers those and falls back to the native PNGs when absent. At the
current 4096² setting: base-level PSNR vs source albedo 40.1 / normal 39.0 /
metalness 37.3 / roughness 34.6 dB, VRAM `classic 42.8 MB / neural 0.75 MB`
(4×512² latents trained against the 4K reference = **1.75%**). The PBR-pass
delta between modes is the pure MLP cost (≈0.1 ms at full-frame coverage; the
title's EMA is short-window and view-dependent). Note the report's "direct BC1
×3" baseline assumes ARM packing — this renderer keeps metal/rough as two
separate BC1 textures, hence 4 maps.

## Renderer modifications (all in the `Research/PBR` clone)

- `src/d3d12.cpp/.hpp`: minimal BC1/DDS mip-chain loader (reads the files
  `bc1.py:write_dds_mips` produces), 4 latent SRVs (t7–t10, own descriptor
  table), MLP weights in a b1 constant buffer (716 packed floats), `N`-key
  toggle carried in `eyePosition.w`; missing `data/neural/` falls back to the
  classic path.
- `data/shaders/hlsl/pbr.hlsl`: `neuralMaterial()` — sample 4 latents with the
  ordinary sampler (hardware BC1 decode + filtering, exactly as trained),
  unrolled FMA MLP, storage-space outputs converted like the classic path
  (manual sRGB→linear for basecolor, `*2-1` for normal); ambient term now
  multiplied by the AO output (classic path uses ao=1: Cerberus has no AO map).
- `src/common/mesh.cpp`: `#include <stdexcept>` (MSVC 2022 compat).

## Notes / limitations

- Cerberus has no AO map; the trainer gets a constant-white `ao.png`, so the
  MLP's AO output ≈ 1 and the comparison stays fair.
- The renderer flips V (`1-texcoord.y`) for all textures alike, so latents and
  classic maps sample identically; anisotropic filtering on latents is what the
  BCF papers do at runtime, though our training only models trilinear (M2).
- Weight layout, sRGB handling and sampler conventions must match
  `m2_mip_lod/train.py` meta.json — this folder is a data-contract consumer.
