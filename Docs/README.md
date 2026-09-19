# Neural Material documentation

`UE5_Neural_Material_Plugin_Development_Spec.docx` is the authoritative architecture and agent handoff document.

Key initial decisions:

- UE5.7, D3D12-first; Runtime must not require Python or Cooperative Vectors.
- Runtime representation: four BC1 latent textures with mip chains plus a small FMA MLP.
- Editor-side training is an external Python/PyTorch backend for the MVP.
- Keep the training-to-runtime data contract verifiable with fixed BC1 and MLP fixtures.

Update the specification before changing the latent layout, MLP profile, asset format, material-expression interface, or training backend.
