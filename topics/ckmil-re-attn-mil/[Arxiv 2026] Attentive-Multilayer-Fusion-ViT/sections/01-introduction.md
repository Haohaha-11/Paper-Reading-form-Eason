[← 返回 README](../README.md)

# 01 - 引言与相关工作

> 💡 **Hao 批注 - 论文定位**: 这是一篇 ICML 2026 的 probing 方法论文。核心问题非常明确——ViT 的标准用法是取最后一层的 CLS token 做分类，但中间层的信息被完全丢弃了。作者的核心洞察：任务相关信息分布在网络全深度，不同任务偏好不同深度，只用最后一层是次优的。

> 💡 **Hao 批注 - 与 MIL 的关联**: 这个思路与 MIL 中"bag 内不同 instance 包含不同的部分信息"惊人相似——只是这里 bag = ViT 的所有层，instances = 每层的 CLS/AP token。本质上 ALF 就是一种 layer-level MIL：attention over layers 来聚合对当前任务有用的表征。

> 💡 **Hao 批注 - 为什么不是 fine-tuning**: 作者明确把 ALF 定位为 probing（backbone 冻结），而非 PEFT（LoRA/Adapter）。价值主张是"比 fine-tuning 便宜，但比 linear probe 好很多"。对于 WSI 场景（一个 slide 上千 patches，每个都要过 ViT），freezing backbone 是巨大优势。

## 1. Introduction & Problem Setting

Foundation models have transformed machine learning across various domains, ranging from language to vision. The standard approach for adapting ViTs to downstream tasks is **linear probing**: extract the final-layer CLS token, train a linear classifier on top. This is computationally efficient (backbone frozen), but potentially suboptimal — it relies on the idea that all task-relevant information has been compressed into a single final-layer representation.

> 💡 **Hao 批注 - 核心矛盾**: ViTs 处理图像时通过逐层 refinement：早期层捕获低层视觉模式（边缘、纹理），深层编码高层语义。但 linear probing 只看最后一层——相当于强迫 ViT 把所有信息都"压缩"到 CLS token。这在预训练任务上或许有效（因为训练目标就是优化 CLS 表示），但在分布外的下游任务上，CLS token 可能丢弃了有用信息。

> The standard linear probing approach operates exclusively on the final-layer representation, which in Vision Transformers is typically represented by the CLS token. This design implicitly assumes that the CLS token encodes all task-relevant information.

> However, recent work challenges this assumption: Chen et al. (2024) show that attentive probing over final-layer patch tokens outperforms CLS-only approaches by facilitating task-dependent spatial information fusion. Similarly, DINOv2 demonstrates that concatenating CLS tokens from several of the last layers can surpass single-layer methods by exploiting some hierarchical information fusion. Together, these results suggest that information crucial for downstream tasks is distributed across layers and tokens rather than exclusively captured by the final CLS token representation.

> ViTs process information across multiple layers: early layers capture low-level visual patterns and structural cues (e.g., edges, textures), whereas later layers encode high-level semantic concepts aligned with the pre-training objective. When downstream tasks differ from the pre-training domain, the final layer likely discards structural or textural information that remains crucial for the target application, yet this information often persists in intermediate layers.

A potential solution is to fuse information distributed across the different levels of model layers. Recent work has begun to exploit intermediate representations for transfer learning (Tu et al., 2023; Evci et al., 2022), but existing approaches either concatenate features naively (leading to high variance and sometimes degraded performance) or are limited to specific model families.

> 💡 **Hao 批注 - 为什么 naive concatenation 不行**: 把所有层的特征拼起来做分类，维度是 d×2|L|，参数爆炸并且容易过拟合。更根本的问题是：不同层对任务的 relevance 不同，拼接时给所有层同等权重 → 无关层引入噪声 → 甚至比只用最后一层更差。ALF 用 attention 解决了这个"选择性利用"问题。

**Problem Statement.** Consider a ViT encoder with L attention layers processing an input image x. For each layer ℓ, we extract two complementary representations:

- **CLS token**: h_CLS^(ℓ) = z_0^(ℓ) — the learned global summary token
- **AP token**: h_AP^(ℓ) = (1/P) Σ_i z_i^(ℓ) — average-pooled patch tokens capturing spatial feature statistics

For the full set of layers L = {1,...,L}, we form:

H_L = [h_CLS^(1),..., h_CLS^(L), h_AP^(1),..., h_AP^(L)]^T ∈ R^{2L×d}

The goal is to learn an attention-based fusion function f_θ that maps H_L to a single task-optimized representation for classification.

> 💡 **Hao 批注 - Token 选择**: CLS+AP 是两个互补的 summary statistic。CLS 是"模型自己觉得重要的全局信息"，AP 是"空间上的平均特征统计量"。直觉上，CLS 可能丢失了不用于预训练任务的空间细节，AP 作为"无偏"的空间平均可能保留更多中间层信息。

![Figure 1: ALF 架构示意图](../images/ba370e52d00342aad6c39742e240468c9b6e8e415931715d91b0f12805408072.jpg)

> 💡 **Hao 批注 - Figure 1**: ALF 架构总览。左侧：ViT 处理输入图像，逐层提取 CLS 和 AP token。中间：将所有层的 token stack 成 H_L。右侧：Shared learnable query Q 通过 multi-head cross-attention 对 H_L 做注意力聚合，产生 fused representation，再进 linear classifier。这张图清晰地展示了"从所有层读取 → 注意力选择 → 融合 → 分类"的管道。

### Contributions

论文列出三项核心贡献：

1. **Attentive probing using CLS and AP tokens** from all intermediate layers, achieving consistent gains across 20 datasets with an average accuracy improvement of +5.54pp over standard linear probing.

2. **Intermediate layer fusion provides consistent improvements** across small, base, and large models, indicating the approach generalizes across model scales without diminishing returns.

3. **Performance gains are largest for tasks different from the pre-training domain**. Interpretable attention patterns show that natural image tasks rely more on later layers, whereas structural or specialized datasets benefit more from intermediate representations, particularly AP tokens.

## 2. Related Work

### 2.1 Probing and Lightweight Adaptation

Parameter-efficient fine-tuning (PEFT) aims to adapt large neural networks without updating all backbone weights. Popular methods include **LoRA** (low-rank adaptation), **adapters** (bottleneck modules inserted between layers), **prompt tuning**, and **BitFit** (bias-only tuning). These methods modify the model's internal representations, while probing methods only read from frozen representations.

> 💡 **Hao 批注 - Probing vs PEFT**: 这是两个方向：(1) PEFT 修改 backbone 前向传播（LoRA 加低秩矩阵，adapter 插 bottleneck），可以改变中间层表征；(2) Probing 只读不改，backbone 完全冻结。ALF 属于后者——代价小（不用存梯度通过 backbone），但也受限于"表征本身不变"。LoRA 可能更好但更贵，ALF 的价值在某个精度-效率 trade-off 区间。

**Attentive probing** extends standard probing by using attention mechanisms. Chen et al. (2024) proposed **AAT** (Attentive probe on All Tokens of the last layer), which uses cross-attention over all patch tokens of the last layer. ALF extends this idea to cross-attention over CLS/AP tokens from ALL intermediate layers.

### 2.2 The Value of Intermediate Representations

The principle that hierarchical features are crucial for robust recognition is fundamental to deep learning. In CNNs, representations progress from low-level patterns in early layers to high-level semantics in later ones (Zeiler & Fergus, 2014). Transferability varies with depth, with earlier layers being more general and later layers more specialized (Yosinski et al., 2014). This led to iconic architectures such as U-Net and Feature Pyramid Networks, which explicitly fuse features from shallow and deep layers.

For ViTs, Raghu et al. (2021) showed that their representations are more uniform across layers than CNNs, but still encode different levels of abstraction:

- **Early layers**: Local texture, edges — share features across token positions
- **Middle layers**: Part-level features, spatial arrangements
- **Later layers**: High-level semantics, become more CLS-token-dominated

Recent lightweight methods such as Head2Toe (Evci et al., 2022), Visual Query Tuning (Tu et al., 2023), and Perception Encoder (Bolya et al., 2025) have shown that explicitly exploiting intermediate ViT layers can enhance transfer performance. Similar findings have emerged in language models, where intermediate layers can even outperform the final layer depending on the task (Liu et al., 2019; Skean et al., 2025).

Building on these insights, ALF learns to dynamically fuse representations from all layers, determining which features are most relevant for a given downstream task.

> 💡 **Hao 批注 - ViT vs CNN 层级差异**: Raghu et al. (2021) 的核心发现是 ViT 的表示比 CNN 更"均匀"分布在层间（CNN 有明显的高低层分工），但这不意味着中间层没用——只是信息的变化更缓和。也意味着只看最后一层丢的信息可能比 CNN 更多（CNN 最后一层已经聚合了多尺度信息）。

> 💡 **Hao 批注 - 与 NLP probing 的联系**: NLP 中的 probing 研究发现 BERT 的中间层在某些语言学任务上优于最后一层——这与 ALF 的发现高度一致。跨模态的 convergence（vision 和 language 都有"中间层有用"的现象）说明这是一般性原则，而非 ViT 的特异性。
