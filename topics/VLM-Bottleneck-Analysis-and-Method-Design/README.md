# VLM Bottleneck Analysis and Method Design

这个 topic 关注 VLM 的**瓶颈分析与方法设计**：核心问题不是"堆更多数据和参数让 VLM 更强"，而是系统性地诊断 VLM 在视觉编码、推理过程、奖励设计、多图理解等环节的 failure mode，并针对性地设计新方法。

## 子类别

### Encoding（视觉编码）
视觉信息如何进入模型、如何自适应地选择分辨率/压缩率/编码策略。

| 论文 | 子类别 | 方法特点 |
|------|--------|----------|
| [Q-Zoom](./encoding/%5BArxiv%202025%5D%20Q-Zoom/) | encoding | Query-aware 自适应视觉感知，根据问题动态调整图像编码粒度。 |
| [CARES](./encoding/%5BArxiv%202025%5D%20CARES/) | encoding | 上下文感知的分辨率选择器，让 VLM 自适应选择最优图像分辨率。 |
| [iGVLM](./encoding/%5BArxiv%202026%5D%20iGVLM/) | encoding | 动态指令引导视觉编码，实现问题感知的多模态理解。 |
| [Perceptual-Bandwidth-Bottleneck](./encoding/%5BArxiv%202026%5D%20Perceptual-Bandwidth-Bottleneck/) | encoding | 感知带宽瓶颈分析，提出主动视觉推理作为 sequential experimental design。 |

### Grounding（视觉定位）
视觉信息与语言推理的空间对齐。

| 论文 | 子类别 | 方法特点 |
|------|--------|----------|
| [Vision-aligned-Latent-Reasoning](./grounding/%5BArxiv%202026%5D%20Vision-aligned-Latent-Reasoning/) | grounding | 视觉对齐的潜空间推理，将视觉特征对齐到 LLM 推理的 latent space。 |

### Hierarchical Design（分层设计）
多层级视觉信息注入与融合。

| 论文 | 子类别 | 方法特点 |
|------|--------|----------|
| [Hierarchical-Visual-Cues](./hierarchical-design/%5BArxiv%202026%5D%20Hierarchical-Visual-Cues/) | hierarchical | 层级化视觉线索注入，多粒度视觉信息逐步融合到推理过程。 |

### Invoke（视觉调用）
推理过程中何时、如何主动调用视觉信息。

| 论文 | 子类别 | 方法特点 |
|------|--------|----------|
| [Iterative-Evidence-Refinement](./invoke/%5BArxiv%202026%5D%20Iterative-Evidence-Refinement/) | invoke | 迭代证据精炼，多轮调用视觉信息逐步增强推理质量。 |
| [Thinking-with-Visual-Grounding](./invoke/%5BArxiv%202026%5D%20Thinking-with-Visual-Grounding/) | invoke | 视觉 grounding 驱动的思维过程，让推理步骤与图像区域直接关联。 |

### Reward Design（奖励设计）
如何为 VLM 推理过程设计有效的奖励信号。

| 论文 | 子类别 | 方法特点 |
|------|--------|----------|
| [RegionReasoner](./reward/%5BArxiv%202026%5D%20RegionReasoner/) | reward | 区域定位的多轮视觉推理，用 region grounding 作为推理的中间监督。 |
| [VisualPRM](./reward/%5BArxiv%202025%5D%20VisualPRM/) | reward | 面向多模态推理的 Process Reward Model，对推理步骤而非最终答案打分。 |
| [Perception-centric-PRM](./reward/%5BArxiv%202026%5D%20Perception-centric-PRM/) | reward | 以感知为中心的 PRM，强化对视觉感知步骤的奖励信号。 |

### Long Reasoning（长程推理）
长时间/多步推理中的视觉信息保持。

| 论文 | 子类别 | 方法特点 |
|------|--------|----------|
| [Imagine-Before-Predict](./long-reasoning/%5BArxiv%202026%5D%20Imagine-Before-Predict/) | long-reasoning | 预测前先想象——交错潜空间视觉推理用于视频事件预测。 |
| [VisMem](./existing/%5BArxiv%202025%5D%20VisMem/) | existing | 动态 latent vision memory，短期感知+长期语义记忆缓解视觉 grounding 丢失。 |
| [DMLR](./existing/%5BArxiv%202025%5D%20DMLR/) | existing | 测试时动态多模态潜推理，置信度引导 latent policy gradient + 动态视觉注入。 |

### Bottleneck Analysis（瓶颈分析）
诊断 VLM 的 failure mode。

| 论文 | 子类别 | 方法特点 |
|------|--------|----------|
| [More-Images-More-Problems](./bottleneck-analysis/%5BArxiv%202026%5D%20More-Images-More-Problems/) | bottleneck | 受控分析 VLM 在多图场景下的 failure mode——更多图像是否带来更多问题？ |

### Multi-Image（多图理解）
多图像场景下的空间与语义绑定。

| 论文 | 子类别 | 方法特点 |
|------|--------|----------|
| [Dual-Mechanisms-Spatial-Binding](./multi-image/%5BArxiv%202026%5D%20Dual-Mechanisms-Spatial-Binding/) | multi-image | VLM 中空间变量绑定的双重机制。 |

### Medical（医学应用）
VLM 瓶颈分析在医学影像中的延伸。

| 论文 | 子类别 | 方法特点 |
|------|--------|----------|
| [MedSynapse-V-v2](./medical/%5BArxiv%202026%5D%20MedSynapse-V-v2/) | medical | 视觉感知与临床直觉的 latent memory 桥接，latent diagnostic memory evolution。 |

---

## 推荐阅读路线

### 路线 1：诊断先行
1. **More-Images-More-Problems**：先理解 VLM 在哪些场景下会失败
2. **Perceptual-Bandwidth-Bottleneck**：理解视觉编码的带宽瓶颈
3. **Dual-Mechanisms-Spatial-Binding**：理解多图场景的空间绑定问题
4. 然后按瓶颈类型选择对应方法论文

### 路线 2：方法驱动
1. **encoding 子类**（Q-Zoom/CARES/iGVLM）：视觉编码效率
2. **grounding + invoke**：视觉信息何时被调用
3. **reward**：如何用奖励信号引导更好的视觉推理
4. **long-reasoning**：长程推理中的视觉保持

### 路线 3：医学落地
1. **bottleneck analysis** 论文 → 理解通用 VLM 瓶颈
2. **MedSynapse-V-v2** → 看瓶颈分析在医学场景的迁移

---

## 横向数据流

| 层级 | 瓶颈 | 方法方向 | 代表论文 |
|------|------|----------|----------|
| 视觉编码 | 分辨率/压缩率选择不当，信息丢失 | adaptive encoding, query-aware perception | Q-Zoom, CARES, iGVLM |
| 视觉调用 | 推理过程中不知道何时重新查看图像 | iterative evidence refinement, visual grounding | Iterative-Evidence-Refinement, Thinking-with-Visual-Grounding |
| 奖励信号 | 缺少对视觉推理步骤的细粒度反馈 | process reward model, region grounding reward | VisualPRM, RegionReasoner, Perception-centric-PRM |
| 长程保持 | 长推理链中视觉 grounding 逐步丢失 | latent memory, dynamic visual injection | VisMem, DMLR, Imagine-Before-Predict |
| 多图绑定 | 多图场景下对象-属性-关系混淆 | spatial variable binding analysis | Dual-Mechanisms-Spatial-Binding, More-Images-More-Problems |
