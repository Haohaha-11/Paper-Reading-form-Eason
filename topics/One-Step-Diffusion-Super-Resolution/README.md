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

## 横向数据流与研究机会

| 论文 | 数据流抓手 | 主要优势 | 主要待做 |
|------|------------|----------|----------|
| RCOD-SR | LDG / DAS / VPIM | 控制变量直接落在 diffusion timestep/latent domain 上，和多步方法的可调采样逻辑一致。 | 学习一个可校准的 degradation/realism estimator，把用户 slider 映射到稳定的感知效果。 |
| CODSR | RGPA / LQFM / TMG / VSD | 把“哪里该生成更多细节”做成空间自适应，而不是全图统一释放生成先验。 | 把 RGPA 的区域权重与退化估计或不确定性估计相连，自动决定局部生成强度。 |
| TinySR | DIA / Expansion-Corrosion / VAE compression / prompt-time removal | 问题定义非常工程化：不是提出更复杂先验，而是让 one-step diffusion SR 真正可部署。 | 做硬件感知剪枝/量化联合搜索，直接优化端侧 latency 和显存峰值。 |
| OFTSR | conditional flow teacher / ODE trajectory alignment / one-step student | 从 flow/ODE 角度重新定义 one-step SR，比固定 timestep diffusion 更自然地表达连续轨迹。 | 把 real-world degradation estimator 引入 flow condition，让 trajectory 同时感知退化类型。 |
| OSEDiff | LQ latent initialization / LoRA tuning / latent VSD / data loss | 把 LQ latent 作为起点，天然保留输入结构，避免随机噪声起点带来的不确定性。 | 加入 RCOD 式 timestep/latent group 控制，把 OSEDiff 从固定映射变成可调映射。 |

## 统一阅读问题

- 单步 SR 的生成自由度来自哪里：fixed timestep、latent noise、flow trajectory，还是 teacher score?
- fidelity-realism trade-off 是全图控制、局部控制，还是隐含在训练数据/teacher 中?
- 方法是否真的 one-step 端到端部署，还是把成本转移到 VAE、prompt extractor、teacher 或额外控制分支?
- 视觉质量提升是否伴随结构 hallucination，尤其在文字、人脸身份、规则纹理和平坦区域上?
