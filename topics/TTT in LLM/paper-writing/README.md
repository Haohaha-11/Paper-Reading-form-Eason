# Paper Writing（TTT 论文写作参考）

这是 **TTT in LLM** 下的子专题，聚焦为撰写 TTT（Test-Time Training）相关论文而精读的核心参考文献。集中收录长上下文 TTT、next-token prediction 目标、证据对齐/自引导选择性训练等最相关工作，方便写作时快速对比动机、方法和实验设置。

## 论文列表

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [TTT-NTP](./%5BArxiv%202026%5D%20TTT-NTP/) | Arxiv 2026 | Test-Time Training with Next-Token Prediction：用 next-token prediction 作为测试时训练目标。 |
| [Self-Guided-TTT](./%5BArxiv%202026%5D%20Self-Guided-TTT/) | Arxiv 2026 | 自引导长上下文 TTT：无需外部标签，模型自生成引导信号做测试时训练。 |
| [EASE-TTT](./%5BArxiv%202026%5D%20EASE-TTT/) | Arxiv 2026 | 证据对齐的选择性 TTT，面向长上下文问答，只在相关证据上做测试时训练。 |
| [TTT4LC](./%5BArxiv%202025%5D%20TTT4LC/) | Arxiv 2025 | 揭示长上下文 static attention 的 score dilution 导致 thinking tokens 失效，提出 Query-Only TTT。 |
| [In-Place-TTT](./%5BICLR%202026%5D%20In-Place-TTT/) | ICLR 2026 Oral | 把 MLP block 的 final projection matrix 当 fast weights，LM-aligned objective + chunk-wise update 的可插拔 TTT。 |
| [TTT-E2E](./%5BArxiv%202025%5D%20TTT-E2E/) | Arxiv 2025 | 将 long-context LM 视作 continual learning，测试时 next-token 更新权重 + 训练时 meta-learning 学初始化。 |

## 写作对比维度

| 维度 | 可对比问题 |
|------|-----------|
| 测试时更新目标 | next-token prediction / 自引导信号 / 证据对齐损失 各有何取舍？ |
| 更新对象 | fast weights / LoRA / 部分权重 / final projection matrix |
| 是否需要训练时配合 | TTT-E2E 需 meta-learning 初始化；其余偏纯测试时 |
| 长上下文适配 | Query-Only、证据选择、自引导——如何避免在无关上下文上过拟合 |
| 效率 | 每次更新的计算/显存开销，是否 chunk-wise、是否可插拔 |

## 阅读建议

1. **TTT4LC**：先理解长上下文下为什么需要 TTT（score dilution 诊断）。
2. **TTT-E2E / In-Place-TTT**：理解测试时更新的两种典型实现（continual learning 视角 vs fast weights 视角）。
3. **TTT-NTP**：聚焦 next-token prediction 作为测试时目标的设计。
4. **Self-Guided-TTT / EASE-TTT**：理解如何在无标签、长上下文场景下选择"在什么上训练"——自引导信号 vs 证据对齐。
