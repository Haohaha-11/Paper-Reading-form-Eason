# VisMem for Med Image

latent vision memory / latent reasoning / medical image VLM：让医学影像模型持续保持视觉证据并调用隐式临床经验。

## 论文列表

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [AlignVLM](./%5BArxiv%202025%5D%20AlignVLM/) | Arxiv 2025 | AlignVLM 把 connector 视为 vision-language latent alignment 模块，用更强归纳偏置改善多模态文档理解。 |
| [VisMem](./%5BArxiv%202025%5D%20VisMem/) | Arxiv 2025 | VisMem 提出 latent vision memory，通过 memory invocation 和 formation 缓解 VLM 长生成中的视觉 grounding 丢失。 |
| [Latent-Space-Survey](./%5BArxiv%202026%5D%20Latent-Space-Survey/) | Arxiv 2026 | 这篇综述系统梳理 latent space 的基础、演进、机制、能力和展望，是理解 latent memory/reasoning 的概念地图。 |
| [MedSynapse-V](./%5BArxiv%202026%5D%20MedSynapse-V/) | Arxiv 2026 | MedSynapse-V 面向医学 VLM，提出 latent diagnostic memory evolution，模拟临床专家在诊断时调用和演化隐式经验。 |
| [Visual-Enhanced-Depth-Scaling](./%5BArxiv%202026%5D%20Visual-Enhanced-Depth-Scaling/) | Arxiv 2026 | Visual Enhanced Depth Scaling 针对 multimodal latent reasoning 中视觉 token 梯度不稳定与语言偏置，提出视觉重放和动态深度扩展。 |

## 推荐阅读顺序

1. **Latent-Space-Survey**：这篇综述系统梳理 latent space 的基础、演进、机制、能力和展望，是理解 latent memory/reasoning 的概念地图。
2. **AlignVLM**：AlignVLM 把 connector 视为 vision-language latent alignment 模块，用更强归纳偏置改善多模态文档理解。
3. **VisMem**：VisMem 提出 latent vision memory，通过 memory invocation 和 formation 缓解 VLM 长生成中的视觉 grounding 丢失。
4. **Visual-Enhanced-Depth-Scaling**：Visual Enhanced Depth Scaling 针对 multimodal latent reasoning 中视觉 token 梯度不稳定与语言偏置，提出视觉重放和动态深度扩展。
5. **MedSynapse-V**：MedSynapse-V 面向医学 VLM，提出 latent diagnostic memory evolution，模拟临床专家在诊断时调用和演化隐式经验。

## 方法对比

| 论文 | 课题角色 | 关键词 |
|------|----------|--------|
| AlignVLM | AlignVLM 把 connector 视为 vision-language latent alignment 模块， | 问题: vision-language latent alignment; 领域: document understanding |
| VisMem | VisMem 提出 latent vision memory，通过 memory invocation 和 format | 机制: latent vision memory; 阶段: memory invocation + formation |
| Latent-Space-Survey | 这篇综述系统梳理 latent space 的基础、演进、机制、能力和展望，是理解 latent memory/reas | 类型: Survey; 主线: foundation/evolution/mechanism/ability/outlook |
| MedSynapse-V | MedSynapse-V 面向医学 VLM，提出 latent diagnostic memory evolution， | 机制: latent diagnostic memory evolution; 模块: Meta Query / Counterfactual / Transition |
| Visual-Enhanced-Depth-Scaling | Visual Enhanced Depth Scaling 针对 multimodal latent reasoning | 机制: Visual Replay + Routing Depth Scaling; 问题: visual token instability |
