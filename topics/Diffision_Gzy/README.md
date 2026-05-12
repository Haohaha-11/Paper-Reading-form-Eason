# Diffision_Gzy

单步 diffusion Real-ISR 批读专题：围绕 score distillation、timestep control 和 fidelity-realism trade-off，比较如何把多步生成先验压缩到一次前向推理中。

## 论文列表

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [TSD-SR](./%5BCVPR%202025%5D%20TSD-SR/) | CVPR 2025 | TSD-SR 用 Target Score Distillation 和 DASM 修正 one-step VSD 的 score direction 与细节梯度，在 0.1362s 单步推理中提升 Real-ISR 感知质量。 |
| [TADSR](./%5BCVPR%202026%5D%20TADSR/) | CVPR 2026 | TADSR 用 Time-Aware VAE Encoder 和 Time-Aware VSD 把 timestep 变成 one-step SR 的 fidelity-realism 控制变量。 |

## 推荐阅读顺序

1. **TSD-SR**：先理解普通 VSD 在 Real-ISR 中为什么会给出不可靠 score direction，以及 target score / DASM 如何把 teacher prior 对齐到真实 HQ reference。
2. **TADSR**：再看 timestep 如何从训练噪声变量变成推理控制变量，并理解 TAE 与 TAVSD 如何共同调节保真和真实感。

## 方法对比

| 论文 | 课题角色 | 关键词 |
|------|----------|--------|
| TSD-SR | 解决 one-step VSD 梯度方向不准的问题 | Target Score Distillation; Target Score Matching; Distribution-Aware Sampling Module; SD3 teacher; 1-step inference |
| TADSR | 解决 one-step SR timestep 固定且不可控的问题 | Time-Aware VAE Encoder; Time-Aware VSD; $t_s \rightarrow t_v$ mapping; controllable fidelity-realism trade-off |

## 横向数据流与研究机会

| 论文 | 数据流抓手 | 主要优势 | 主要待做 |
|------|------------|----------|----------|
| TSD-SR | LQ image -> student latent -> DASM noisy latent -> teacher / LoRA score -> TSD gradient | 把 score distillation 的“方向是否可信”显式问题化，用 HQ target score 和多步 DASM 采样强化真实细节。 | 推理时缺少可控参数，可以结合 timestep slider 或退化估计器自动调节真实感强度。 |
| TADSR | LQ image + $t_s$ -> time-aware latent -> one-step UNet -> mapped $t_v$ teacher guidance -> TAVSD gradient | 把 SD timestep 的原生语义引入 one-step SR，使同一模型能在 fidelity 和 realism 间连续调节。 | 大 timestep 可能引入 hallucination，需要结构一致性、OCR、人脸身份或视频时序约束。 |

## 统一阅读问题

- score distillation 给 student 的梯度到底来自哪里：teacher score、LoRA fake score、target HQ score，还是 noisy latent 采样轨迹?
- fidelity-realism trade-off 是训练时固定进模型，还是可以在推理时由 timestep、latent domain 或损失权重控制?
- one-step 推理真正省掉了哪些成本，哪些成本仍留在 VAE、DAPE、teacher distillation 或训练阶段?
- 感知指标提升是否伴随结构 hallucination，尤其在文字、人脸身份、规则纹理和平坦区域上?
- 能否把 TSD-SR 的 target score 可靠性和 TADSR 的 timestep 可控性合并成一个可校准的 one-step Real-ISR framework?
