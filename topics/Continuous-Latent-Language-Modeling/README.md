# Continuous Latent Language Modeling

这个 topic 关注**连续潜空间语言建模**：核心问题不是"如何让 token-level AR 模型更强"，而是用 diffusion/flow 在连续嵌入空间中建模语言，用全局语义先验替代逐 token 的左到右生成，为文本与连续模态的统一建模提供路径。

## 论文列表

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [Cola DLM](./%5BArxiv%202026%5D%20Cola-DLM/) | Arxiv 2026 | 层级化连续潜扩散语言模型：Text VAE → block-causal DiT prior → conditional decoding。从 Markov-path 视角将 diffusion 定义为 latent prior transport 而非 token 恢复。 |
| [ELF](./%5BArxiv%202026%5D%20ELF/) | Arxiv 2026 | 连续嵌入空间 Flow Matching 语言模型：全程在连续空间建模，最后一步共享权重映射到离散 token。天然支持 CFG，fewer steps 超过现有 DLM。 |

## 推荐阅读顺序

1. **ELF**：先理解"连续嵌入空间 diffusion LM"的基本范式——Flow Matching + 最后一步 unembedding + CFG，方法相对简洁直观。
2. **Cola DLM**：再深入层级化设计——为什么需要 Text VAE 先做 text→latent 映射，再用 DiT 在潜空间建模 prior，以及如何从 Markov-path 统一视角理解各类语言模型。

## 方法对比

| 论文 | 解决的瓶颈 | 核心中间表示 | 输出形态 |
|------|------------|--------------|----------|
| Cola DLM | AR 固定左到右顺序限制灵活性，discrete diffusion 缺乏全局语义建模 | Text VAE latent + block-causal DiT prior | 层级化连续潜空间 → 条件解码 → 文本 |
| ELF | 现有 DLM 在离散空间操作或连续空间效果不佳 | 连续嵌入空间 + Flow Matching trajectory + 共享 unembedding 权重 | 嵌入空间扩散轨迹 → 最后一步映射 → 文本 |

## 横向数据流与研究机会

| 层级 | 输入 | 中间变化 | 输出 | 可追问的问题 |
|------|------|----------|------|--------------|
| 编码 | 离散文本 | Text VAE / embedding lookup 将文本映射到连续潜空间 | continuous latent / embedding | 潜空间的结构化程度是否影响后续 diffusion 建模？ |
| 先验建模 | 连续潜表示 + noise schedule | diffusion/flow 在潜空间中 transport 噪声到语义先验 | clean latent prior | latent prior 是否真正学到了全局语义结构，还是只做表面统计？ |
| 解码 | 连续潜 prior | 条件解码 / shared-weight unembedding 将连续表示映射回 token | 离散文本 | 解码步骤是否会引入信息瓶颈，限制 scaling？ |

## 统一阅读问题

- **Q1: 为什么连续潜空间比离散 token 空间更适合 diffusion？**
  A: 离散空间 diffusion 需要设计复杂的 corrupt/mask 策略和 categorical distribution，连续空间可以直接复用图像 diffusion 的成熟技术（Gaussian noise、CFG 等）。

- **Q2: Cola DLM 的层级设计相比 ELF 的平层设计有哪些优劣势？**
  A: Cola 的 Text VAE + DiT 分离了表示学习和先验建模，理论更清晰但工程更复杂；ELF 端到端在嵌入空间操作更简洁但可能缺乏潜空间的显式结构化约束。

- **Q3: 最大风险在哪里？**
  A: 连续潜空间语言建模的 scaling law 尚未充分验证；如果 scaling 行为不理想，可能在更大规模上被 AR 模型反超。

- **Q4: 最值得继续做的方向是什么？**
  A: 将连续潜语言建模与多模态（图像、视频、语音）的连续 latent 统一到一个 diffusion prior 框架中，实现真正的 unified continuous-modal foundation model。
