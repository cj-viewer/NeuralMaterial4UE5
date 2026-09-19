# Neural Material

An Unreal Engine 5.7 experimental plugin for training and using neural PBR materials. The MVP targets BC1-compressed latent textures and an ordinary HLSL FMA decoder.

## Layout

| Path | Purpose |
| --- | --- |
| `Source/NeuralMaterialRuntime` | Runtime asset, material integration, resource binding, and inference. |
| `Source/NeuralMaterialEditor` | Editor training workflow, import/export, preview, and validation UI. |
| `Docs` | Architecture specifications and long-lived developer documentation. |
| `Research` | Standalone experiments and proof-of-concept work. |

## Initial module graph

```text
NeuralMaterialEditor  -->  NeuralMaterialRuntime  -->  Engine / CoreUObject / Core
```

Keep the Runtime module independent of the external training environment. See `Docs/README.md` and the included development specification for the data contract and milestones.
