[← 返回 README](../README.md)

# 02 Method

> 💡 **Hao 批注 - 方法架构**: 这篇论文的方法部分主要是(1)统一现有方法的形式化框架，(2)引入4种新的实例级池化算子，(3)描述6种SSL预训练方法。论文本身的创新在于实验设计和实例级算子的引入，而非提出新的MIL或SSL方法。

---

## 2.1 MIL Problem Formulation

> 💡 **Hao 批注 - 统一框架**: 通用形式 $S(X_i) = g(\sigma(h(f(x))))$。实例级: h(.)是patch分类器, g(.)=identity。嵌入级: h(.)=identity, g(.)是bag分类器。这个统一框架很好地区分了两类方法的本质差异。

Let $X_i = \{x_{ij}\}_{j=1...K_i}$ be the WSI of subject i and $x_{ij} \in \mathbb{R}^d$ its j-th patch. The final bag-score is:

$$
S(X_i) = g(\sigma(h(f(x))))
$$

where $g(\cdot)$ and $h(\cdot)$ are bag-level and instance-level classifiers respectively, $\sigma$ is a symmetric permutation-invariant pooling operator, and $f(\cdot)$ is the encoder.

- **Instance-level approach**: only $h(\cdot)$, computing a score for each instance, $g(\cdot)$ = identity.
- **Embedding-level approach**: only $g(\cdot)$, computing score at bag level, $h(\cdot)$ = identity.

![](../images/674a66f95b1c367fcad6975867c1f89ca3a00f762001bd0600dab72a2e3c4333.jpg)
![](../images/ff6dee3a1a6491809caef41704f60acce46d9e01f816586344d5208a8032056e.jpg)

> 💡 **Hao 批注 - Figure 2**: 清晰的pipeline对比。(a) SSL预训练；(b.1) 嵌入级MIL: 聚合特征→bag分类；(b.2) 实例级MIL: 实例分类→分数聚合。

---

## 2.2 Instance-based MIL Pooling Operators

> 💡 **Hao 批注 - Max/Mean的局限**: Max-pooling可能遗漏关键信息(只看一个patch)，Mean-pooling可能引入过多噪声(所有patch平均)。本文引入的4种算子通过可学习参数在max和mean之间自适应调整。

### Existing: MaxMIL and MeanMIL

- MaxMIL: $\sigma = \max_k y_k$
- MeanMIL: $\sigma = \frac{1}{K}\sum_k y_k$

### 2.3 Proposed Instance-based MIL Pooling Operators

> 💡 **Hao 批注 - 4种新算子**: 都来自声音事件检测领域。关键是参数极少(MixMIL 1个参数α, AutoMIL 1个参数α, LNPMIL 1个参数p, AttenMIL K个权重w_k)，但提供了自适应能力。AttenMIL的K个权重使其成为实例级MIL中最"重"的方法(每个patch一个可学习权重)，但仍在sub-4000参数级别。

**MixMIL (Mixed-Pooling)**: $\sigma = \alpha \max_k y_k + (1-\alpha) \sum_k y_k / K$, where $\alpha \in [0,1]$ is trainable.

**AutoMIL (Auto-pooling)**: $\sigma = \sum_k y_k \cdot (\exp(\alpha y_k) / \sum_j \exp(\alpha y_j))$, where $\alpha \in [0,\infty)$ is trainable. When $\alpha=0$, equals MeanMIL; when $\alpha \to \infty$, approaches MaxMIL.

**LNPMIL (Learned-Norm Pooling / Softmax-pooling)**: $\sigma = (\frac{1}{K}\sum_k |y_k|^p)^{1/p}$, where $p = 1 + \log(1+e^{\tilde{p}})$, $\tilde{p} \in \mathbb{R}$ is trainable.

**AttenMIL (Attention-pooling)**: $\sigma = \sum_k y_k \hat{w}_k$ with $\hat{w} = softmax(w_k)$, where weights $w_k$ (one per patch) are learned.

### 2.3.1 Multi-class Instance-based MIL

> 💡 **Hao 批注 - 多类MaxMIL**: 朴素的多类MaxMIL的问题是——癌症slide中有大量正常组织区域，正常类别(如benign)的分数可能更高，导致max选到正常patch。解决方案：先在病理类别(如class 2,3)中找max，再在该patch的所有类别中取argmax。这个设计简单但有效。

For BRACS (3 classes: Benign, Atypical, Malignant), the proposed Multi-class MaxMIL:
1. First select instance with highest score among pathological classes: $m = \arg\max_k(s_{ik}^{(2)}, s_{ik}^{(3)})$
2. Then predict by taking max among all scores of instance m: $y_i = \arg\max(s_{im}^{(1)}, s_{im}^{(2)}, s_{im}^{(3)})$

---

## 2.4 Embedding-based MIL Pooling Operators

> 💡 **Hao 批注 - 嵌入级baseline**: 4种主流方法 + 2种额外方法(DTFDMIL, DAMIL)。注意TransMIL包含了position encoding，理论上能建模空间信息——这恰好是Paper 1论证可能未被有效利用的。

**ABMIL**: Attention-based, $\sigma = \sum_k a_k f_k$ where $a_k = \text{softmax}(w^T \tanh(V f_k^T))$.

**CLAM**: Attention-based + instance-level clustering to improve feature space.

**DSMIL**: Dual-stream, hybrid instance + bag classifier. Detects critical patch via MaxMIL, then computes query/information vectors, bag aggregator uses feature similarity to critical patch.

**TransMIL**: Transformer-based, TPT module with 2 Transformer layers + PPEG (Pyramid Position Encoding Generator) for spatial information.

**DTFDMIL**: Double-Tier Feature Distillation, uses "pseudo-bags" (subsets of patches) with two-tier MIL.

**DAMIL**: Deep Attention Multiple Instance Survival Learning, K-Means clustering on each slide.

---

## 2.5 Self Supervised Pre-training of f

> 💡 **Hao 批注 - 6种SSL方法**: 覆盖了SSL的主要范式——对比学习(SimCLR)、动量对比(MoCoV3)、掩码自编码(MAE)、知识蒸馏(DINO)、自举式(BYOL)、冗余消除(Barlow Twins)。注意MAE只兼容ViT，其余方法兼容CNN和ViT。

| Method | Paradigm | Key Mechanism |
|--------|----------|---------------|
| SimCLR | Contrastive | InfoNCE loss, positive/negative pairs |
| BYOL | Bootstrap | Online + target networks, no negatives |
| Barlow Twins | Redundancy Reduction | Cross-correlation matrix → identity |
| DINO | Knowledge Distillation | Teacher-student, ViT, centering+sharpening |
| MoCo v3 | Hybrid | BYOL arch + SimCLR contrastive + DINO ViT |
| MAE | Masked Autoencoder | Reconstruct masked 75% patches, ViT only |

---

## 2.6 Pathology Adapted SSL Methods

> 💡 **Hao 批注 - 病理适应SSL**: (1) PathAug: 病理特定增强(垂直翻转+染色增强)，简单但有效；(2) SRCL: 语义相关对比学习，额外选择语义相似的正样本对；(3) CluBYOL: 将聚类集成到BYOL中。这些方法的改进相对较小，主要是增强策略的调整。

- **PathAug**: Vertical flipping + stain augmentation (RandStainNA)
- **SRCL**: Semantically-relevant contrastive learning, additional positive pairs via cosine similarity
- **CluBYOL**: Cluster Bootstrap BYOL, integrates clustering into SSL process

---

## 2.7 Foundation Models

> 💡 **Hao 批注 - 基础模型作为零样本特征提取器**: 不做任何fine-tuning，直接下载预训练权重提取特征。UNI效果最好(ViT Large, 307M参数, 100M+ patches预训练)，但比本文训练的backbone大得多。DINOv2效果最差(非病理领域，自然图像域差异大)。

| Model | Architecture | Training Data | Notable |
|-------|-------------|---------------|---------|
| DINOv2 | ViT-Small | 142M natural images | General vision, domain gap |
| PathAugFM | ViT-Small | 19M pathology patches | DINO + pathology augmentations |
| CTransPath | Hybrid (CNN+Swin) | 15M pathology patches | SRCL framework |
| UNI | ViT-Large (307M) | 100M+ pathology patches | DINOv2, 20 tissue types |
