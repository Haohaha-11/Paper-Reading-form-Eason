# VisMem for Med Image

这个 topic 关注 latent vision memory / latent reasoning / medical image VLM：核心问题不是“让模型多说推理过程”，而是让视觉证据、隐式经验和诊断逻辑在连续 latent stream 中被保留、调用、筛选和内化。

## 论文列表

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [AlignVLM](./%5BArxiv%202025%5D%20AlignVLM/) | Arxiv 2025 | 将视觉特征映射到 LLM text embedding 的加权组合中，用语言嵌入先验约束 connector，减少 vision-language latent misalignment。 |
| [VisMem](./%5BArxiv%202025%5D%20VisMem/) | Arxiv 2025 | 构造动态 latent vision memory，用短期细粒度感知记忆和长期语义记忆缓解长生成中的 visual grounding 丢失。 |
| [Latent-Space-Survey](./%5BArxiv%202026%5D%20Latent-Space-Survey/) | Arxiv 2026 | 从 foundation、evolution、mechanism、ability、outlook 五个视角组织 latent space 研究，是本 topic 的概念地图。 |
| [Visual-Enhanced-Depth-Scaling](./%5BArxiv%202026%5D%20Visual-Enhanced-Depth-Scaling/) | Arxiv 2026 | 分析 latent reasoning 训练中的 visual token 梯度不稳定，提出 visual replay 和 routing depth scaling。 |
| [MedSynapse-V](./%5BArxiv%202026%5D%20MedSynapse-V/) | Arxiv 2026 | 面向医学 VLM 的 latent diagnostic memory evolution：先取解剖先验，再用反事实奖励筛选记忆，最后把专家分支内化到学生模型。 |
| [DMLR](./%5BArxiv%202025%5D%20DMLR/) | Arxiv 2025 | 训练自由的测试时动态多模态潜空间推理：置信度引导的潜策略梯度优化 latent think tokens，动态视觉注入实现推理-感知自适应交错。 |
| [MedVLThinker](./%5BArxiv%202025%5D%20MedVLThinker/) | Arxiv 2025 | 首个全开源医学多模态推理模型配方：RLVR > SFT，text-only > image-text，32B 匹配 GPT-4o。揭示医学 VLM 推理训练的数据质量瓶颈。 |

## 推荐阅读顺序

1. **Latent-Space-Survey**：先建立 latent space 与 explicit CoT、memory、planning、perception 的共同语言。
2. **AlignVLM**：理解视觉 token 进入 LLM latent space 的入口问题；如果 connector 已经错位，后续 memory/reasoning 会把错误证据放大。
3. **VisMem**：学习“视觉证据如何在生成过程中持续存在”的通用 memory 设计。
4. **Visual-Enhanced-Depth-Scaling**：理解 latent reasoning 训练为什么会偏语言、弱视觉，以及如何按 token 难度分配推理深度。
5. **DMLR**：学习 test-time 潜空间推理的核心机制——置信度引导优化 + 动态视觉注入，是 training-free latent reasoning 的代表方案。
6. **MedSynapse-V**：把前面几条线迁移到医学诊断：先验调用、因果筛选、教师经验内化。
7. **MedVLThinker**：验证医学 VLM 推理训练的配方选择——RLVR vs SFT、text-only vs multimodal，建立开源基线和系统性对比。

## 方法对比

| 论文 | 解决的瓶颈 | 核心中间表示 | 输出形态 |
|------|------------|--------------|----------|
| AlignVLM | 视觉特征直接投到 LLM embedding space 时缺少语言结构约束 | LLM text embedding weighted average / aligned visual token | 更易被 LLM 解释的文档视觉表示 |
| VisMem | 长推理/长生成中视觉 grounding 逐步丢失 | short-term visual memory + long-term semantic memory | 更稳定的 VLM 回答、推理和生成 |
| Latent-Space-Survey | latent space 研究分散，机制与能力边界不清 | foundation/evolution/mechanism/ability/outlook 分类框架 | 可复用的综述地图和研究问题清单 |
| Visual-Enhanced-Depth-Scaling | latent training 中 visual token 梯度大且不稳，复杂 token 受固定深度限制 | visual replay saliency + routing depth scaling | 更快、更强的 multimodal latent reasoning |
| MedSynapse-V | 医学 VLM 离散 token 化导致量化损失、长程信息衰减和缺少 case-adaptive expertise | meta query memory、counterfactual refined memory、intrinsic memory transition | 更符合诊断逻辑的医学 VQA / diagnosis |
| DMLR | 显式 CoT 视觉 grounding 弱，think-with-image 工具调用不稳定，固定位置的 latent reasoning 缺乏自适应 | confidence-guided latent think tokens + dynamic visual injection via policy gradient | training-free 的测试时多模态 reasoning，95%+ benchmark 达最优 |
| MedVLThinker | 医学多模态推理缺乏开源配方、SFT vs RLVR 系统性对比缺失 | difficulty-filtered QA → SFT/RLVR trained on text-only or image-text data | 开源医学 VLM reasoning 模型，32B 匹配 GPT-4o |

## 横向数据流与研究机会

| 层级 | 输入 | 中间变化 | 输出 | 可追问的问题 |
|------|------|----------|------|--------------|
| 对齐入口 | 图像、文档、医学影像 patch/token | connector 把视觉表示放入 LLM 可解释的 latent 区域 | aligned visual tokens | 视觉 token 是否真的保留可诊断细节，还是只贴近语言先验？ |
| 记忆保持 | 当前视觉证据、历史生成状态、任务上下文 | short-term memory 保存局部感知，long-term memory 抽象语义经验 | 可被反复调用的 latent memory | memory 调用会不会引入过时或无关证据？ |
| 隐式推理 | latent token、复杂问题、可选 CoT teacher | visual replay 强化视觉 token，routing depth 给复杂 token 更多迭代 | compact latent reasoning trace | 如何判定哪些 token 需要更深推理？ |
| 测试时优化 | latent think tokens、图像 patch、置信度信号 | confidence reward → policy gradient 更新 latent tokens，reward-gated 动态注入视觉 patch | 优化后的 latent tokens 解码为最终答案 | 测试时优化是否能在医学影像推理中替代 fine-tuning？ |
| 医学推理训练 | 医学 QA 数据 (text-only / image-text)、教师 CoT 蒸馏 / 答案校验 reward | 难度过滤 → SFT/RLVR 训练，GRPO 二值 reward 引导推理策略 | 推理链 + 最终答案 | text-only RLVR 为何优于 image-text？数据质量 vs 视觉信号如何权衡？ |
| 医学诊断 | 医学影像、解剖先验、病例问题 | meta query 取先验，CCR 用反事实遮挡筛 causal memory，IMT 内化教师分布 | 诊断答案与隐式临床经验 | latent memory 是否能给出可审计的临床证据链？ |

## 统一阅读问题

- **Q1: latent memory 与 explicit CoT 的关系是什么？**
  A: 这组论文倾向把 CoT 看作可被蒸馏或替代的显式轨迹；真正要保留的是连续 latent 中的视觉证据、任务状态和经验模式。
- **Q2: 为什么医学场景更需要 latent memory？**
  A: 医学影像判断依赖细粒度局部证据和专家先验，普通 VLM 的离散文本解释容易丢掉影像细节；latent memory 可以在回答前后持续承载隐式诊断线索。
- **Q3: 最大风险在哪里？**
  A: latent 表示更难审计。若 memory invocation、counterfactual reward 或 routing depth 的选择机制不可靠，模型可能更自信地固化错误先验。
- **Q4: 最值得继续做的方向是什么？**
  A: 把 latent memory 的可解释性、跨病例检索、影像区域级因果验证和临床下游校验连起来，避免只在 VQA accuracy 上证明有效。
