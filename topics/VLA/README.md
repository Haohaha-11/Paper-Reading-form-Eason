# VLA · Vision-Language-Action

> 从“如何稳定地产生机器人训练数据”到“如何把不同 backbone、action head、训练与评测拼成可比较系统”，这个 Topic 关注 VLA 研究中最容易被模型论文掩盖的两层基础设施：**数据/环境层**与**模型/系统层**。

## 主题简介

Vision-Language-Action（VLA）的目标，是让机器人把视觉观测和语言指令转化为可执行动作。一个完整系统至少包含两条相互依赖的链：

1. **数据链**：真实物体与场景如何数字化、如何自动产生高质量演示、仿真与真机如何对齐、困难任务是否真正覆盖双臂交互；
2. **模型链**：多模态 backbone 如何连接 action head、离散/回归/flow/双系统动作解码如何公平比较、训练和 benchmark 接口如何统一。

本 Topic 的两篇论文恰好位于不同层级。RoboTwin 从真实 RGB 物体出发生成带空间语义的数字孪生和专家演示，是 **VLA 上游的数据与评测基础设施**；StarVLA 将 VLM/world-model backbone 与四类 action head 放进统一训练、评测和部署管线，是 **VLA 中游的模型与系统基础设施**。二者组合起来，形成一条从“世界和演示”到“策略和部署”的研究主线。

## 收录论文

| 论文 | 层级 | 真正解决的问题 | 关键证据 |
|---|---|---|---|
| **[RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins](./%5BCVPR%202025%5D%20RoboTwin-%20Dual-Arm%20Robot%20Benchmark%20with%20Generative%20Digital%20Twins/README.md)** · CVPR 2025 Highlight | 数据、环境、benchmark | 单张 RGB 如何变成多样 3D 资产，如何用空间标注 + LLM + 规划器自动生成双臂专家演示，以及 sim 数据是否减少 real 数据需求 | 300 sim + 20 real 对比 20 real：单臂 1.2%→72%（+70.8 百分点），双臂 20%→62%（+42 百分点） |
| **[StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing](./%5BArxiv%202026%5D%20StarVLA-%20A%20Lego-like%20Codebase%20for%20Vision-Language-Action%20Model%20Developing/README.md)** · arXiv 2026 Technical Report | 模型、训练、评测、部署 | 如何通过标准化 hidden-state 与 I/O 契约，将 VLM/world-model backbone 和 FAST/OFT/π/GR00T action head 双向解耦并公平比较 | LIBERO StarVLA-OFT 30K steps 达 96.6%，接近 OpenVLA-OFT 175K steps 的 97.1%；RoboTwin 2.0 random 上 StarVLA-π 达 88.8% |

## 两篇论文如何接起来

~~~mermaid
flowchart LR
    A[真实物体 RGB 图] --> B[RoboTwin 生成多样 3D 数字孪生]
    B --> C[关键点 + 功能/接近/侧向轴]
    C --> D[LLM 任务分解与受限 API 代码]
    D --> E[规划器生成无碰专家轨迹]
    E --> F[RGB-D / 点云 / 双臂状态 / 动作]
    F --> G[StarVLA 统一数据与训练接口]
    G --> H{可插拔 backbone}
    H --> H1[Qwen3-VL 等 VLM]
    H --> H2[Cosmos-Predict2 等 world model]
    H1 --> I[标准化 hidden state]
    H2 --> I
    I --> J{可插拔 action head}
    J --> J1[FAST 离散 token]
    J --> J2[OFT 并行回归]
    J --> J3[π flow matching]
    J --> J4[GR00T 双系统]
    J1 --> K[动作 chunk]
    J2 --> K
    J3 --> K
    J4 --> K
    K --> L[仿真 / 真机 server-client 评测]
~~~

这条连接目前仍有一处重要空白：RoboTwin 原论文只用 DP/DP3 验证数据价值，没有直接训练语言条件 VLA；StarVLA 虽整合 RoboTwin 2.0 benchmark，但没有证明它在 RoboTwin 2025 的生成式数字孪生数据上能够获得同样的 sim-to-real 收益。因此“RoboTwin 数据 → StarVLA generalist”是自然但尚未被两篇论文共同实证的下一步。

## 横向比较

| 维度 | RoboTwin | StarVLA |
|---|---|---|
| 核心产物 | 生成式数字孪生、双臂演示、真仿真对齐 benchmark | 模块化 VLA 代码库、统一训练/评测/部署接口 |
| 输入 | 真实物体 RGB 图、任务描述、空间标注 | 多视角视觉观测、语言指令、机器人动作数据 |
| 关键中间表示 | function/contact points + function/approach/lateral axes | backbone 输出的标准化 hidden state |
| 生成/预测目标 | 可执行代码、无碰轨迹、专家演示 | 未来动作 chunk |
| 模型角色 | GPT-4V/LLM 做资产语义与高层程序综合，规划器做轨迹 | VLM/world model 做表征，FAST/OFT/π/GR00T 做动作解码 |
| 评测重点 | 数据量、2D/3D 观测、困难双臂任务、sim-to-real | action head、backbone、specialist/generalist、系统吞吐 |
| 最强价值 | 降低真实数据采集成本并暴露双臂协调难点 | 消除框架差异造成的混杂变量，提供可复现对比平台 |
| 主要边界 | 标注半自动、力学保真有限、只测 DP/DP3、14/15 任务口径冲突 | RL 尚未交付、部分 benchmark 缺结果、FAST 系统性偏弱、部分表格口径不一致 |

## 推荐阅读路线

### 路线 A：先理解“VLA 数据从哪里来”

1. RoboTwin [Abstract](./%5BCVPR%202025%5D%20RoboTwin-%20Dual-Arm%20Robot%20Benchmark%20with%20Generative%20Digital%20Twins/sections/00-abstract.md) 与 [Introduction](./%5BCVPR%202025%5D%20RoboTwin-%20Dual-Arm%20Robot%20Benchmark%20with%20Generative%20Digital%20Twins/sections/01-introduction.md)：建立数据 + benchmark 的问题框架；
2. [Methodology](./%5BCVPR%202025%5D%20RoboTwin-%20Dual-Arm%20Robot%20Benchmark%20with%20Generative%20Digital%20Twins/sections/03-methodology.md)：看数字孪生、空间中间表示和 LLM/规划器分工；
3. [Experiments](./%5BCVPR%202025%5D%20RoboTwin-%20Dual-Arm%20Robot%20Benchmark%20with%20Generative%20Digital%20Twins/sections/05-experiments.md)：核对 +70.8/+42 绝对百分点以及困难双臂任务；
4. [Appendix](./%5BCVPR%202025%5D%20RoboTwin-%20Dual-Arm%20Robot%20Benchmark%20with%20Generative%20Digital%20Twins/sections/07-appendix.md)：复现时看观测规格、训练参数、prompt 与 API。

### 路线 B：再理解“VLA 系统如何公平比较”

1. StarVLA [Introduction](./%5BArxiv%202026%5D%20StarVLA-%20A%20Lego-like%20Codebase%20for%20Vision-Language-Action%20Model%20Developing/sections/01-introduction.md)：理解框架碎片化和统一契约；
2. [Unified Framework](./%5BArxiv%202026%5D%20StarVLA-%20A%20Lego-like%20Codebase%20for%20Vision-Language-Action%20Model%20Developing/sections/02-unified-framework.md)：看 backbone–head 解耦与广义 VLA 形式；
3. [Single-Benchmark Results](./%5BArxiv%202026%5D%20StarVLA-%20A%20Lego-like%20Codebase%20for%20Vision-Language-Action%20Model%20Developing/sections/05-single-benchmark-results.md)：比较四种动作范式，注意 96.6% 是“接近”而非超过 97.1%；
4. [Cross-Benchmark](./%5BArxiv%202026%5D%20StarVLA-%20A%20Lego-like%20Codebase%20for%20Vision-Language-Action%20Model%20Developing/sections/07-cross-benchmark.md) 与 [Computation Efficiency](./%5BArxiv%202026%5D%20StarVLA-%20A%20Lego-like%20Codebase%20for%20Vision-Language-Action%20Model%20Developing/sections/08-computation-efficiency.md)：看 generalist 正迁移与系统扩展边界。

## 统一研究问题

### 1. 数据多样性和模型多样性如何解耦？

RoboTwin 同时改变外观、形状、物理参数和轨迹，StarVLA 同时允许替换 backbone、action head 和训练策略。干净的实验需要固定一侧、只改变另一侧：先用同一 StarVLA 配置比较 RoboTwin 的不同数据因子，再用同一 RoboTwin 数据比较不同 action head。

### 2. 空间语义应该放在数据层还是模型层？

RoboTwin 把 function/contact points 与方向轴显式写进资产和规划约束；StarVLA 把多模态信息压到 hidden state，再由 action head 解码。值得验证的是：显式空间标注是只用于生成演示，还是也应作为 VLA 的结构化输入或 auxiliary target，以提升复杂交接和避碰任务泛化。

### 3. 为什么复杂双臂协调仍然失败？

RoboTwin 中 Dual Shoes Place 全部设置低于 15%，说明“有更多仿真数据”并不足以解决动态交互、臂间信用分配和长时序互避。StarVLA 的 action chunk 和 generalist 训练可提供新建模手段，但需要专门报告交接、同步和碰撞失败类型，不能只看任务平均成功率。

### 4. sim-to-real 增益能否扩展到通用 VLA？

最直接的下一组实验是：用 RoboTwin 生成数据预训练 StarVLA-OFT/π，固定 20 条真实微调数据，分别比较 specialist、跨任务 generalist 和语言重述泛化；同时测试纯 RGB、RGB-D/点云辅助与结构化空间标注三种输入。

### 5. 模块化是否真的带来可归因的科学结论？

模块化本身是工程能力，不自动等于科学因果。需要完整的“资产生成因子 × backbone × action head × 训练策略 × benchmark 难度”因子设计，并控制训练步数、epoch、GPU 小时、数据量和随机种子，才能把性能差异归因到具体选择。

## 当前结论

- VLA 的瓶颈不只在模型结构；数据生成、空间语义、真仿真对齐、训练接口和评测协议共同决定结论是否可信。
- RoboTwin 证明生成式仿真数据可以显著减少真机样本，但尚未证明通用语言条件 VLA 的收益。
- StarVLA 证明统一平台能用更少训练量得到很强的可复现 baseline，但 96.6% 应表述为接近而非超过 OpenVLA-OFT 的 97.1%，且 RL 仍未交付。
- 两篇论文最有价值的交叉方向，是把 RoboTwin 的显式空间语义与生成演示接入 StarVLA 的统一 backbone–head 系统，针对复杂双臂协调做可归因的 sim-to-real 研究。

## 开放问题

- 如何量化数字孪生的视觉保真、力学保真与功能语义保真各自对 VLA 的贡献？
- 关键点/方向轴能否作为辅助监督，改善 VLM backbone 的空间 grounding 而不损害动作学习？
- flow-matching action head 是否比离散 token 或 L1 回归更适合接触丰富、双臂耦合任务？
- generalist 联合训练的正迁移来自共享视觉语义、共享动作模式，还是数据量本身？
- 何种评测能够把“选错手臂、交接失败、碰撞、末态偏差、语言误解”分开统计？
- 如何在保持模块化的同时，对数据量、训练步数、GPU 时间和推理延迟做真正公平的预算化比较？

---

**建议起点**：如果关心数据与机器人场景，先读 RoboTwin；如果关心 VLA 架构与实验平台，先读 StarVLA；如果要设计新项目，直接从“RoboTwin 数据/空间标注 → StarVLA-OFT/π → 双臂 sim-to-real”这条交叉路线开始。
