[← 返回 README](../README.md)

# 00 - 摘要

**Authors**: Laure Ciernik\*, Marco Morik\*, Lukas Thede, Luca Eyring, Shinichi Nakajima, Zeynep Akata, Lukas Muttenthaler (\*equal contribution)

**Venue**: ICML 2026 (PMLR 306)

## Abstract

> With the rise of large-scale foundation models, efficiently adapting them to downstream tasks remains a central challenge. Standard linear probing, which uses only the final-layer representation, often falls short, leaving substantial task-relevant information distributed across earlier layers untapped. In this work, we propose **Attentive Layer Fusion (ALF)**, a lightweight probing method that dynamically fuses representations from all intermediate layers of a frozen Vision Transformer (ViT) using multi-head cross-attention over CLS and average-pooled (AP) tokens.

> Evaluated on 20 diverse datasets across 9 ViT models from three families, ALF consistently outperforms standard linear probing, achieving an average gain of **+5.54 percentage points**. Our analysis reveals that task-relevant information is widely distributed across a model's hierarchy, with the importance of different layers varying substantially across tasks. Notably, performance improvements correlate with the task's distance from the pre-training domain, suggesting that ALF recovers latent features particularly relevant for tasks outside the model's original training objective. The approach provides an interpretable, computationally efficient way to extract richer representations from frozen foundation models without fine-tuning backbone parameters.

> 💡 **Hao 批注 - 论文定位**: 这是一篇 ICML 2026 的 probing 方法论文。核心问题非常明确——ViT 的标准用法是取最后一层的 CLS token 做分类，但中间层的信息被完全丢弃了。作者的核心洞察：任务相关信息分布在网络全深度，不同任务偏好不同深度，只用最后一层是次优的。

> 💡 **Hao 批注 - Abstract 核心**: 三个关键点：(1) 中间层有任务相关信息但被 discard；(2) layer importance 是 task-dependent；(3) gains 与"任务-预训练域距离"正相关。第三点尤其重要——它暗示 ALF 的价值主要在于 recovering "预训练模型知道但没用 CLS 表达出来"的信息。

> 💡 **Hao 批注 - 与 MIL 的关联**: 这个思路与 MIL 中"bag 内不同 instance 包含不同的部分信息"惊人相似——只是这里 bag = ViT 的所有层，instances = 每层的 CLS/AP token。本质上 ALF 就是一种 layer-level MIL：attention over layers 来聚合对当前任务有用的表征。

> 💡 **Hao 批注 - 为什么不是 fine-tuning**: 作者明确把 ALF 定位为 probing（backbone 冻结），而非 PEFT（LoRA/Adapter）。价值主张是"比 fine-tuning 便宜，但比 linear probe 好很多"。对于 WSI 场景（一个 slide 上千 patches，每个都要过 ViT），freezing backbone 是巨大优势。
