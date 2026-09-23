# Research workspace

神经材质训练与测试。详细参数、数据契约见 `trainer/README.md`。

## 一次性准备

```bat
trainer\build_pbr.bat                                 :: 构建 D3D12 测试渲染器
.venv\Scripts\python.exe trainer\prep_cerberus.py     :: 生成 4K Cerberus 训练集
```

Python 环境：`Research\.venv`（torch cu128 + numpy + pillow）。

## 日常三步（任意目录可运行）

```bat
train.bat --data trainer\data\cerberus_4096 --latent-res 512 --out PBR\data\material
validate.bat PBR\data\material\export.npz
test.bat
```

- **train.bat** — 训练并输出材质包（经典 BC1 四图 + 神经 latent/权重 + 训练工件）。
  常用参数：`--data` 源纹理目录（省略则用合成集）、`--latent-res` latent 分辨率
  （默认 512，2 的幂）、`--out` 材质包输出目录、`--iters`（默认 20000）。
  带进度条 / 总 iter / ETA。全部参数：`train.bat --help`。
- **validate.bat** — `--selftest` 跑 8 项管线自测；传入 `<包>\export.npz` 校验
  训练导出（含从位流独立复现 fixtures）。
- **test.bat** — 启动渲染器，默认加载 `PBR\data\material`；
  `test.bat -material <目录>` 换包（相对路径按你的启动目录解析）。
  按键：`N` 经典↔神经 · `D` 黑白差异 · `C` FMA↔CoopVec · 滚轮/左右键拖动控制相机。

## 第三方参考（只读，勿提交、勿抄代码未查许可证）

`TextureSetNeuralCompressionSample/`（Intel BCF1 官方 demo）、
`neural-compression-textures/`（NVIDIA NTC 非官方复现）。

里程碑实验（m0–m3）已整合进 `trainer/`，历史代码见 git（最后存在于 `f58b2ca`）。
