[← 返回 README](../README.md)

# 02 — Methodology

## 2.1 Weakly supervised learning on fine-tuning slide-level tasks

> **原文**:

Before the large-scale development of pathology foundation models, visual models pretrained on natural images struggled to extract effective features from pathology images due to their limited pathology domain understanding. Consequently, weakly supervised learning has been necessary to obtain slide representations from patch features (Fig.1a). Specifically, given a WSI with patch feature set denoted as P = {p_1, p_2, ..., p_n}, feature transformation F(·) and aggregation G(·) are proposed:

(p̃_1, p̃_2, ..., p̃_n) = F(p_1, p_2, ..., p_n),  s = G({p̃_1, p̃_2, ..., p̃_n}),

where F and G respectively denote vector and scalar-valued functions. MIL-based fine-tuning typically follows the composition of these two functions. For instance, in the classical ABMIL [8], F is the identity mapping and G is a gated attention mechanism; whereas in TransMIL [21], F applies a nonlinear self-attention transformation and G outputs a class token. Regardless of the specific method, the composite function invariably contains learnable parameters that must be optimized using the slide-level labels from downstream tasks:

ŷ = Softmax(Ws),  L(s, y) = -Σ_{k=1}^K y_k ln(ŷ_k),

where W is a linear classifier and L represents the cross-entropy loss. While effective, this approach yields task-dependent slide representations, limiting generalizability and robustness to distributional shifts.

> 💡 **F·G 分解框架**: Hao 批注 — 作者将 MIL 方法统一为 F(特征变换) ∘ G(特征聚合) 的组合，这是一个很好的抽象。不同 MIL 方法的差异在于 F 和 G 的具体实现：(1) ABMIL: F=identity, G=gated attention；(2) TransMIL: F=self-attention, G=class token；(3) DTFD-MIL: F=pseudo-bag generation + distillation, G=attention。但作者暗示这种分解本身可能是有问题的——F 和 G 中的可学习参数需要在每个 task 上重新训练，导致 slide representation 是 task-specific 的。

> 💡 **Task-dependent representation 的代价**: Hao 批注 — 当 slide representation 依赖于下游任务标签时，模型学习的是"对这个任务有用的特征组合"而非"对 WSI 形态学的通用表示"。这有两个后果：(1) 泛化性差——换一个任务需要重新训练整个 F∘G；(2) 对分布偏移敏感——如果测试数据的采集协议不同，task-specific 特征可能失效。SiMLP 通过 task-agnostic mean pooling 避免这个问题。

## 2.2 Slide representation with task-agnostic pooling

> **原文**:

Pathology foundation models pretrained over millions of histopathology images provide the possibility of obtaining task-agnostic slide representation. For instance, by clustering patch features extracted from the foundation model, WSI features can be represented as a combination of morphological prototypes [22]. Additionally, further training a slide encoder with proxy tasks based on large-scale patch features has been shown to be an effective aggregation strategy for generating generic slide-level features, both in visual [33,30] and multimodal [9,6,27] settings. Although these approaches have demonstrated promising results, they often rely on additional signals for guidance. In contrast, a more straightforward approach is to leverage pooling layers, which represent one of the simplest feature aggregation methods. Pooling has been widely adopted in fine-tuning modules across various vision tasks and requires no additional learnable parameters. Therefore, the aggregation capability of pooling-based methods is worth exploring as a baseline, providing a simplified solution for slide-level fine-tuning and validating its transferability across diverse tasks.

> 💡 **Task-agnostic 方法的谱系**: Hao 批注 — 作者将 task-agnostic slide representation 方法分为几个层次：(1) 最复杂：morphological prototypes [22]（无监督聚类 + prototype 编码）；(2) 中等：slide-level self-supervised pretraining [33,30]（用 proxy task 预训练 slide encoder）；(3) 最简单：pooling（无参数聚合）。SiMLP 选择了最简单的一端——这既是 strength（极简）也是 limitation（可能在某些任务上不够 expressive）。

> 💡 **Pooling 的三个优势**: Hao 批注 — (1) 无额外可学习参数——不引入过拟合风险；(2) 计算高效——O(n*d) 的简单聚合；(3) 天然 task-agnostic——pooling 操作本身不依赖任务标签。但 mean pooling 的致命弱点是"对每个 patch 等权"——如果 WSI 中只有 5% 的区域包含诊断信息，mean pooling 相当于用 95% 的噪声稀释了 5% 的信号。

## 2.3 Non-linear classification head

> **原文**:

Using linear probe, a simple linear transformation, general-purpose slide representations can be widely adapted to various WSI-based clinical tasks. However, its linear nature limits its ability to effectively align representations with the lower-dimensional space of downstream tasks. To enhance the transferability of slide representations, we adopt a non-linear classifier based on a two-layer MLP. Notably, modern deep learning frameworks can efficiently optimize matrix multiplications and the additional activation layer, this adjustment strikes a balance between improving representation flexibility and maintaining efficiency. The overall of SiMLP is shown in Fig.1b.

> 💡 **线性 probe vs MLP 的选择**: Hao 批注 — 这里的论述比较弱。作者说线性 probe "限制了将表示对齐到下游任务低维空间的能力"，但没有量化解释为什么。实际上，如果 PFM 特征已经线性可分（高维空间的线性分类器 ≈ 低维空间的复杂决策边界），线性 probe 就够了。MLP 的优势在于：(1) 可以学习非线性决策边界；(2) 额外的隐层可以做隐式的特征选择/降维。但从实验结果看，MLP vs 线性 probe 的收益通常只有 1-3pp——说明 PFM 特征在大多数任务上确实接近线性可分。

> 💡 **SiMLP 的命名**: Hao 批注 — SiMLP = Simple MLP。名称准确地传达了方法的核心——就是把复杂的 F∘G 替换为最简单的 mean pooling + MLP。这个名字也暗示了作者的立场：在 PFM 时代，"simple isn't just simpler, it's better"。

![Figure 1a: Traditional fine-tuning vs SiMLP](../images/802262e2b01d3e5f091a30d051c28ea43d21a1a7199806a96a7f744e816134b4.jpg)
![Figure 1b: SiMLP structure](../images/802262e2b01d3e5f091a30d051c28ea43d21a1a7199806a96a7f744e816134b4.jpg)
![Figure 1c: Performance comparison](../images/802262e2b01d3e5f091a30d051c28ea43d21a1a7199806a96a7f744e816134b4.jpg)
