# Research and proof-of-concept workspace

Place isolated experiments here. Each experiment should be self-contained, document its inputs and expected output, and avoid becoming a runtime plugin dependency until it has a reproducible validation result.

Shared Python environment: `Research/.venv` (torch cu128 + numpy + pillow), created per
`m0_float_latent_baseline/requirements.txt`.

## Experiments

- `m0_float_latent_baseline/` — M0: float latent + MLP reconstruction of one PBR set,
  with numpy gold fixtures for the later D3D12/UE comparisons. See its README.
- `m1_differentiable_bc1/` — M1: STE-quantized simulated-BC1 latents, real BC1/DDS
  exporter with independent-decoder validation, float-vs-BC1 resolution sweep. See its README.
- `m2_mip_lod/` — M2: independently trained BC1 mip pyramids, continuous-LOD training
  against box-filtered reference targets, mipped DDS export, naive-mip A/B baseline. See its README.
- `m3_pbr_visual/` — M3a: the neural material rendered inside a real D3D12 PBR+IBL
  renderer. The renderer source (based on Nadrin/PBR, MIT) is vendored with our
  modifications at `Research/PBR/`: hardware BC1 decode + FMA MLP in HLSL, live
  VRAM/GPU-time stats, N/D toggles vs a direct-BC1 classic baseline. See its README.

Suggested first experiments:

1. Python BC1 quantization/decode and exporter fixtures.
2. 12 -> 32 -> PBR MLP reconstruction using four latent textures.
3. D3D12/HLSL FMA inference compared against the Python reference.

## Reference checkouts (read-only, not our code)

- `TextureSetNeuralCompressionSample/` — Intel's official demo for the Belcour & Benyoub BCF1 paper (MIT). Inference-only: pretrained BC1 latents + MLP models and the D3D12/HLSL inference shaders. Use as the gold reference for the M3 GPU comparison and the training-to-runtime data contract (weight layout, latent binding).
- `neural-compression-textures/` — unofficial PyTorch reimplementation of NVIDIA's Random-Access Neural Texture Compression (MIT). Different architecture; reference for PyTorch training-loop structure only.

Both keep their own `.git` and upstream license; do not commit them into this repository or copy code from them without checking the license.
