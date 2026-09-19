# Neural Material documentation

`UE5_Neural_Material_Plugin_Development_Spec.docx` is the authoritative architecture and agent handoff document.

## Primary references

- `2311.16121v2.pdf` — Weinreich et al., *Real-Time Neural Materials using Block-Compressed Features* (Ubisoft BCF direction).
- `2506.06040v1.pdf` — Belcour & Benyoub, *Hardware Accelerated Neural Block Texture Compression with Cooperative Vectors* (BCF1 training direction).
- [Ubisoft La Forge: Shipping Neural Texture Compression in Assassin's Creed Mirage](https://www.ubisoft.com/en-us/studio/laforge/news/415slmB3ZGzv8d2LhO9QT/shipping-neural-texture-compression-in-assassins-creed-mirage) — production implementation reference.

Key initial decisions:

- UE5.7, D3D12-first; Runtime must not require Python or Cooperative Vectors.
- Runtime representation: four BC1 latent textures with mip chains plus a small FMA MLP.
- Editor-side training is an external Python/PyTorch backend for the MVP.
- Keep the training-to-runtime data contract verifiable with fixed BC1 and MLP fixtures.

Update the specification before changing the latent layout, MLP profile, asset format, material-expression interface, or training backend.
