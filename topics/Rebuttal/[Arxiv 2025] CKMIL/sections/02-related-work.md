[← 返回 README](../README.md)

# 02 - Related Work

## 原文 Section: RELATED WORK

### Multiple Instance Learning for WSI Analysis

The MIL paradigm addresses the challenge of gigapixel-scale WSIs by treating each slide as a bag of instances (Maron and Lozano-Perez 1997), effectively leveraging bag-level labels. Under this paradigm, the primary objective of MIL becomes learning the relationships among instances within a bag. A typical two-stage MIL approach involves two steps (Lu et al. 2021). First, a feature encoder (e.g., Resnet50 (He et al. 2016)), often pre-trained on large-scale image datasets (Deng et al. 2009), transforms instances into low-dimensional feature vectors. Second, an aggregation module is designed to aggregate instance-level features into a bag-level representation for downstream tasks.

> **Hao 批注, 机制拆解**: MIL 的两阶段范式是本文所有方法的前提。关键设定：第一阶段特征提取器冻结，第二阶段聚合器可训练。CKMIL 的所有创新都在第二阶段。这种两阶段设定的优点是模块化（换特征提取器不影响聚合器设计），缺点是特征提取和聚合解耦可能导致信息瓶颈。

---

### Attention as Independent Instance Weighting

To overcome the limitations of simple pooling aggregators such as Mean-pooling and Max-pooling, attention mechanisms were introduced to assign discriminative weights to instances based on their importance (Ilse, Tomczak, and Welling 2018). Foundational methods in this category, including ABMIL (Ilse, Tomczak, and Welling 2018), CLAM (Lu et al. 2021), and DSMIL (Li, Li, and Eliceiri 2021), typically employ a shared attention network to score each instance independently. However, these methods are fundamentally built on the independent and identically distributed (i.i.d.) assumption, thereby neglecting the correlations between instances. This premise contradicts core pathology principles, where interactions within the tumor microenvironment are often crucial for diagnosis. Consequently, by treating each instance in isolation, these models cannot fully model the broader tissue context and may over-focus on cytologically salient but diagnostically redundant areas.

> **Hao 批注, 问题动机**: 这一段的批评非常精准——独立注意力方法有两个根本问题：
> 1. **忽略肿瘤微环境中的细胞间交互**: 病理诊断中，肿瘤细胞的空间分布（如浸润前沿、免疫细胞聚集）本身就是重要诊断线索
> 2. **可能过度关注"细胞学突出但诊断冗余"的区域**: 比如大片的坏死区域在 HE 染色下可能十分显眼，但无诊断价值
>
> 这两个批评是后文 CKMIL 建模实例间相关性的正当性基础。但注意 DSMIL 其实尝试了通过 max-instance 的特征级联引入一定程度的实例间交互，作者的分类可能过于简化。

---

### Global Interaction in MIL with Linear Complexity

To address the context-agnostic nature of independent weighting, methods incorporating global self-attention were explored. However, the standard self-attention mechanism, with its prohibitive $O(n^2)$ computational complexity, is ill-suited for the massive number of instances in a WSI. This challenge motivated the development of several global interaction methods with linear complexity. Prominent examples include TransMIL (Shao et al. 2021), which uses Nystrom's method to approximate the attention matrix; MambaMIL (Yang, Wang, and Chen 2024), which leverages the Mamba state space model (Gu and Dao 2023); and RRTMIL (Tang et al. 2024), which adapts the Swin Transformer (Liu et al. 2021). While computationally efficient, each carries its own limitations. TransMIL's Nystrom approximation with pooling-based landmarks risks diluting key signals. MambaMIL is constrained by the fixed 1D sequential processing of the Mamba architecture, and RRTMIL's performance is sensitive to its window configuration and parameter count. Critically, these methods share a common flaw: they are key-instance agnostic. By treating all instances uniformly during interaction, they risk overlooking the sparse yet crucial diagnostic signals present in WSIs, leading to suboptimal outcomes.

> **Hao 批注, 问题动机**: 这是 Related Work 中最重要的段落，因为它直接为 CKMIL 的 core claim 建立对比基线。三类方法及其局限：
>
> | 方法 | 技术路线 | 核心局限（按论文说法） |
> |------|---------|---------------------|
> | TransMIL | Nystrom attention + pooling landmarks | Pooling-based landmarks 稀释关键信号 |
> | MambaMIL | 1D 序列建模 (Mamba SSM) | 固定 1D 序列处理，忽略空间结构 |
> | RRTMIL | Swin Transformer window attention | 对 window 配置和参数敏感 |
>
> 三者共享的缺陷："key-instance agnostic"。CKMIL 的差异化策略是针对 TransMIL 的直接改进——保留 Nystrom attention 的效率优势，但将 pooling-based landmarks 替换为 SDA 筛选的关键实例。这个定位非常聪明：CKMIL 不是凭空创造新机制，而是在现有最高效机制（Nystrom）上做一个关键的输入改进。
>
> 值得注意的细节：作者没有说 MambaMIL 和 RRTMIL 也存在"pooling-based landmark 稀释"的问题——它们根本没有 landmarks 的概念。作者批判的是一个更底层的共性问题（所有实例等同对待），而 CKMIL 的具体改进（替换 landmarks）实际只针对 TransMIL 这类使用显式 landmarks 的方法最直接。从 MambaMIL/RRTMIL 的角度看，CKMIL 的改进能否直接移植过去是存疑的（但 Table 4 中 TransMIL+KGGA 实验给出了正向信号）。

---

## 🔖 Section 总结

### 相关工作分类框架

```
WSI MIL 方法演化
├── 简单池化 (Mean/Max Pooling)
├── 独立注意力加权 (ABMIL, CLAM, DSMIL)
│   └── 问题: i.i.d. 假设，忽略实例间关联
├── 线性复杂度全局交互
│   ├── TransMIL (Nystrom + pooling landmarks)
│   ├── MambaMIL (SSM 1D 序列建模)
│   └── RRTMIL (Swin Transformer window attention)
│   └── 共同问题: key-instance agnostic
└── CKMIL (本文): 关键实例引导的全局交互
    └── 核心改进: Nystrom landmarks 由 SDA 筛选的关键实例替代 pooling
```

### 核心洞察

1. Related Work 的组织逻辑清晰：先介绍 MIL 范式，再按"独立注意力 → 线性全局交互"递进，最后指出两类方法的共同缺陷，自然引出 CKMIL 的定位。

2. 论文的差异化策略是"改进而非革命"——在 TransMIL 的 Nystrom attention 框架基础上，将 landmark 选择策略从 pooling 替换为关键实例引导。这种方法创新（改进现有机制的输入而非发明新机制）比凭空创造更容易被审稿人接受。

3. 对 MambaMIL 和 RRTMIL 的批判力度有限——只说了"各自有局限"和"共享 key-instance agnostic 缺陷"，但没有分析 CKMIL 的思想能否移植到这两种架构上。这是一个可能的 rebuttal 需要补齐的点。

### 可追问点

- CKMIL 与 TransMIL 除了 landmark 选择策略不同，还有哪些实现差异？Gate fusion 机制是 CKMIL 独有的吗？
- MambaMIL 的 1D 序列约束是否本质性地阻碍了"关键实例引导"思想的应用？
- 作者是否有必要更详细地分析为什么 pooling-based landmarks 会"稀释"信号（机制的量化分析而非仅 final performance 对比）？
