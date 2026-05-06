# One-Step Diffusion Super-Resolution

单步 diffusion / flow 超分辨率：在保持生成先验感知质量的同时解决推理效率、保真-真实感权衡和可控性。

## 论文列表

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [RCOD-SR](./%5BAAAI%202026%5D%20RCOD-SR/) | AAAI 2026 | RCOD-SR 用 latent domain grouping、退化感知蒸馏和视觉 prompt 注入，让 one-step diffusion SR 具备推理时 realism 控制能力。 |
| [CODSR](./%5BArxiv%202025%5D%20CODSR/) | Arxiv 2025 | CODSR 通过 LQ-guided feature modulation、区域自适应生成先验激活和 text-matching guidance，在单步 SR 中兼顾局部保真与感知质量。 |
| [TinySR](./%5BArxiv%202025%5D%20TinySR/) | Arxiv 2025 | TinySR 面向实时 Real-ISR，把 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存压成轻量单步模型。 |
| [OFTSR](./%5BICLR%202026%5D%20OFTSR/) | ICLR 2026 | OFTSR 用 conditional flow teacher 和 ODE trajectory alignment distillation 构建 one-step SR，并保留可调 fidelity-realism trade-off。 |
| [OSEDiff](./%5BNeurIPS%202024%5D%20OSEDiff/) | NeurIPS 2024 | OSEDiff 将低质量图像直接作为扩散起点，用 latent-space VSD 把 Stable Diffusion 先验压缩到单步 Real-ISR 推理。 |

## 推荐阅读顺序

1. **OSEDiff**：OSEDiff 将低质量图像直接作为扩散起点，用 latent-space VSD 把 Stable Diffusion 先验压缩到单步 Real-ISR 推理。
2. **OFTSR**：OFTSR 用 conditional flow teacher 和 ODE trajectory alignment distillation 构建 one-step SR，并保留可调 fidelity-realism trade-off。
3. **TinySR**：TinySR 面向实时 Real-ISR，把 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存压成轻量单步模型。
4. **RCOD-SR**：RCOD-SR 用 latent domain grouping、退化感知蒸馏和视觉 prompt 注入，让 one-step diffusion SR 具备推理时 realism 控制能力。
5. **CODSR**：CODSR 通过 LQ-guided feature modulation、区域自适应生成先验激活和 text-matching guidance，在单步 SR 中兼顾局部保真与感知质量。

## 方法对比

| 论文 | 课题角色 | 关键词 |
|------|----------|--------|
| RCOD-SR | RCOD-SR 用 latent domain grouping、退化感知蒸馏和视觉 prompt 注入，让 one-s | Inference steps: 1; 控制变量: latent-domain group / denoising degree |
| CODSR | CODSR 通过 LQ-guided feature modulation、区域自适应生成先验激活和 text-matc | Inference steps: 1; 核心模块: LQ modulation + region prior + text matching |
| TinySR | TinySR 面向实时 Real-ISR，把 one-step diffusion teacher 通过深度剪枝、VAE | Speedup: up to 5.68×; Parameter reduction: 83% |
| OFTSR | OFTSR 用 conditional flow teacher 和 ODE trajectory alignment  | Inference steps: 1; Teacher: conditional flow-based SR model |
| OSEDiff | OSEDiff 将低质量图像直接作为扩散起点，用 latent-space VSD 把 Stable Diffusion | Inference steps: 1; Inference time: 0.11s on A100 for 512×512 |
