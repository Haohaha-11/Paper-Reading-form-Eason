# VisMem for Med Image

本课题聚焦 **latent vision memory / latent reasoning / medical image VLM**：核心问题是医学影像 VLM 如何在长推理或诊断过程中持续保持视觉证据、调用隐式临床经验，并避免被语言先验或离散 token 瓶颈带偏。

## 论文列表

| 论文 | 年份 | 核心问题 | 方法特点 |
|------|------|----------|----------|
| [The Latent Space Survey](./%5BArxiv%202026%5D%20Latent-Space-Survey/) | 2026 | latent space 概念、机制和能力谱系 | 从 foundation/evolution/mechanism/ability/outlook 梳理 latent computation |
| [Visual Enhanced Depth Scaling](./%5BArxiv%202026%5D%20Visual-Enhanced-Depth-Scaling/) | 2026 | latent reasoning 中视觉 token 不稳定、语言偏置 | Spatially-Coherent Finer Visual Replay + Routing Depth Scaling |
| [MedSynapse-V](./%5BArxiv%202026%5D%20MedSynapse-V/) | 2026 | 医学 VLM 缺少 case-adaptive clinical intuition | latent diagnostic memory evolution + counterfactual refinement |
| [AlignVLM](./%5BArxiv%202025%5D%20AlignVLM/) | 2025 | 视觉特征与语言 embedding latent space 错位 | connector/latent alignment for multimodal document understanding |
| [VisMem](./%5BArxiv%202025%5D%20VisMem/) | 2025 | VLM 长生成中视觉 grounding 丢失 | latent vision memory + memory invocation + memory formation |

## 推荐阅读顺序

1. **Latent Space Survey**：先建立 latent representation / computation / memory 的概念地图。
2. **AlignVLM**：理解视觉特征进入语言 latent space 前的 alignment 问题。
3. **VisMem**：读通用视觉记忆机制，理解 memory invocation / formation。
4. **Visual Enhanced Depth Scaling**：读 latent reasoning 中如何保持视觉证据并动态扩展推理深度。
5. **MedSynapse-V**：最后读医学影像核心论文，看 latent memory 如何变成 clinical intuition / diagnostic memory evolution。

## 方法谱系

```text
Latent Space as substrate
        │
        ├── AlignVLM: vision-language latent alignment before reasoning/memory
        │
        ├── VisMem: latent vision memory for visual grounding in long generation
        │       └── gap: general VLM memory, not specialized for clinical diagnosis
        │
        ├── Visual Enhanced Depth Scaling: visual evidence preservation during latent reasoning
        │
        └── MedSynapse-V: latent diagnostic memory evolution for medical VLMs
```

## 横向比较

| 维度 | Latent Space Survey | AlignVLM | VisMem | Visual Enhanced Depth Scaling | MedSynapse-V |
|------|---------------------|----------|--------|-------------------------------|--------------|
| 课题角色 | 理论地图 | latent alignment 基础 | 通用视觉记忆 | latent reasoning 视觉增强 | 医学影像核心方法 |
| 主要对象 | latent space | connector / embedding | latent vision memory | visual tokens / depth routing | diagnostic memory |
| 解决痛点 | 概念分散 | 跨模态表示错位 | 长生成视觉遗忘 | 视觉证据传播衰减 | 医学临床经验缺失 |
| 医学相关性 | 间接 | 医学报告/文档潜在相关 | 可迁移到医学 VQA | 可迁移到医学推理 | 直接医学影像/诊断 |
| 风险点 | 概念过宽 | 细粒度异常可能被压缩 | 错误记忆形成 | 伪影视觉重放 | 可解释性与临床安全 |

## 可复用 Insight

- **先 alignment，再 memory/reasoning**：视觉特征如果没有进入 LLM 可用 latent space，后续记忆和推理都会错位。
- **医学场景需要动态经验**：MedSynapse-V 的关键不是静态 feature，而是 case-adaptive diagnostic memory evolution。
- **视觉证据会衰减**：VisMem 和 Visual Enhanced Depth Scaling 都在处理长生成/深层 latent reasoning 中视觉证据被语言先验覆盖的问题。
- **隐式 latent 能力有可解释风险**：医学影像需要额外关注 latent memory 的可审查性、失败案例和 hallucination 风险。
- **评估要分模态/分场景**：不能只看总体 benchmark 分数，应看 modality breakdown、case study、robustness、latency 和 failure cases。

## 待读问题

- latent diagnostic memory 能否被可视化为病灶证据或临床经验片段？
- 错误视觉记忆形成后，模型是否会在后续回答中持续放大错误？
- AlignVLM 式 connector 对医学小病灶/低对比异常是否会过度平滑？
- latent reasoning 深度动态扩展是否能适配不同医学模态，如 X-ray、CT、MRI、病理切片？
