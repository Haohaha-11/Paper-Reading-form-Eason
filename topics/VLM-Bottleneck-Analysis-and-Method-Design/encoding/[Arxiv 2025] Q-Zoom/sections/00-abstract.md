# 00 - Abstract & Overview

## 原文 Abstract

Multimodal Large Language Models (MLLMs) require high-resolution visual inputs for fine-grained tasks like document understanding and dense scene perception. However, current global resolution scaling paradigms indiscriminately flood the quadratic self-attention mechanism with visually redundant tokens, severely bottlenecking inference throughput while ignoring spatial sparsity and query intent.

To overcome this, we propose Q-Zoom, a query-aware adaptive high-resolution perception framework that operates in an efficient coarse-to-fine manner. First, a lightweight Dynamic Gating Network safely bypasses high-resolution processing when coarse global features suffice. Second, for queries demanding fine-grained perception, a Self-Distilled Region Proposal Network (SD-RPN) precisely localizes the task-relevant Region-of-Interest (RoI) directly from intermediate feature spaces.

To optimize these modules efficiently, the gating network uses a consistency-aware generation strategy to derive deterministic routing labels, while the SD-RPN employs a fully self-supervised distillation paradigm. A continuous spatio-temporal alignment scheme and targeted finetuning then seamlessly fuse the dense local RoI with the coarse global layout.

Extensive experiments demonstrate that Q-Zoom establishes a dominant Pareto frontier. Using Qwen2.5-VL-7B as a primary testbed, Q-Zoom accelerates inference by 2.52x on Document & OCR benchmarks and 4.39x in High-Resolution scenarios while matching the baseline's peak accuracy. Furthermore, when configured for maximum perceptual fidelity, Q-Zoom surpasses the baseline's peak performance by 1.1% and 8.1% on these respective benchmarks. These robust improvements transfer seamlessly to Qwen3-VL, LLaVA, and emerging RL-based thinking-with-image models, setting a new state-of-the-art for efficient, fine-grained visual perception.

---

## 核心数字速览

| 场景 | 加速比 | Token 减少 | 精度 vs Baseline Peak |
|------|--------|-----------|----------------------|
| Document & OCR | 2.52x | 53.0% | +1.1% |
| High-Resolution | 4.39x | 73.2% | +8.1% |

> **Hao 批注**: 这两个数字是整篇论文最核心的卖点。注意这里的加速是相对于 brute-force 4096-token baseline，且 Q-Zoom 自己只用 max 1024 tokens 就超越了 4096-token 的峰值。这是一种典型的"少即是多"——通过精准的空间注意力聚焦，用更少的 token 获得更好的效果。

---

## Figure 1: 三种 Adaptive High-Resolution Perception 范式对比

![Figure 1](../images/60277866c869d1ea142a1d0927f831c9e5f84d7db9e0be5672316edfe7cd56d9.jpg)

**Fig. 1**: Comparison of adaptive high-resolution perception paradigms.

三种范式的核心对比：

| 范式 | 代表方法 | 工作机制 | 核心瓶颈 |
|------|---------|---------|---------|
| **Training-Free** | ViCrop [23] | 手工设计的对比规则提取 RoI | 需要多次冗余 prefill pass |
| **RL-Based** | Thyme [5], DeepEyes [4] | LLM 自回归生成 code/坐标定位 RoI | 依赖冗长 CoT 解码，训练昂贵不稳定 |
| **Q-Zoom (Ours)** | - | 在中间特征空间单次 prefill 直接操作 | - |

> **Hao 批注**: 这是论文的核心定位——Q-Zoom 不是 training-free 也不依赖 RL，而是在中间特征空间上做轻量级可学习模块。这个定位让它同时避开了 training-free 的多次前传开销和 RL 的 CoT 解码延迟。Fig.1 清晰地展示了三种范式在 prefill/decode 流程上的差异。

---

## 三大核心贡献

### 贡献 1: Q-Zoom 两阶段自适应框架

解耦了感知保真度与二次计算代价：
- **阶段一**: Dynamic Gating Network 判断粗粒度特征是否足够
- **阶段二**: SD-RPN 精确定位任务相关 RoI
- **关键优势**: 两个模块均在单次 prefill pass 中完成

### 贡献 2: 数据高效优化策略

- **Consistency-Aware Sample Generation**: 沿分辨率轨迹评估 → 干净二元路由标签
- **Tri-State Self-Distillation**: 从内部交叉注意力挖掘伪标签 → sink token 过滤 + 三态标签分配

### 贡献 3: 时空对齐与定向微调

- 连续时空 MRoPE 编码方案
- LLM-as-a-Judge 硬样本挖掘 → 定向 Post-SFT
- 消除裁剪 RoI 与全局布局的空间失准，恢复空间推理能力

> **Hao 批注**: 这三个贡献层层递进：贡献 1 定义了"做什么"，贡献 2 解决了"怎么训练（不用标注数据）"，贡献 3 修复了"做完之后的副作用（空间失准）"。这是一个完整的端到端解决方案。

---

## 与前置工作 SD-RPN 的关系

本文是 SD-RPN (ICLR 2026) 的扩展版本，核心增量包括：

1. **Dynamic Gating Network**: 原 SD-RPN 对所有 query 都触发 RoI 分支（刚性 pipeline），Q-Zoom 增加了 query-level 的路由判断
2. **Spatio-Temporal Alignment + Post-SFT**: 原 SD-RPN 未解决裁剪 RoI 后的空间失准问题
3. **更丰富的训练数据和评估**: 增加了高分辨率 DocVQA 样本、更多的 backbone 模型验证

---

## Index Terms

Multimodal large language models, Region of interest, High-resolution perception.

> **Hao 批注**: 这三个关键词精准地概括了论文的技术主线——在大模型中做自适应感知，核心手段是 RoI 定位和高分辨率重编码。
