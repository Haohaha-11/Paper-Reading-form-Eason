[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览

Related Work 分两条线：一条是 WSI classification 的 MIL 发展，从 instance-based 到 embedding-based，再到 attention/Transformer/contrastive/hierarchical 方法；另一条是 redundancy reduction 在视觉任务中的 token pruning/merging。WSIR2 的定位是把第二条线迁移到 gigapixel WSI 的 MIL 聚合瓶颈里。

## 2.1 Multiple instance learning for WSI classification

MIL-based pathological diagnosis utilizes readily available slide-level diagnostic results as labels for model training and has gained widespread attention in the computational pathology community.

> 💡 **背景批注**: slide-level label 容易获得，patch-level label 稀缺，这是 WSI 分类普遍采用 MIL 的根源。压缩模块必须适配这种弱监督环境，不能依赖精确 lesion mask。

Early MIL-based methods are rooted in an instance-based approach, in which a model is first trained for patch-level prediction and then produces the slide-level prediction by aggregating the patch-level predicted logits using mean or max pooling. As this scheme introduces erroneous pseudo-labels for patches and fails to capture robust relationships between contextual information and slide-level predictions, it has gradually been superseded by embedding-based methods.

> 💡 **方法演进**: instance-based 先给 patch 伪标签，再聚合 logits，容易错；embedding-based 直接聚合 patch embeddings，保留更多信息，也成为 WSIR2 的输入设定。

The embedding-based approach takes patch-level embeddings as input and directly predicts the slide-level label. Because patch-level embeddings, rather than only their labels, are available, this branch of methods generally achieves better results.

> 💡 **WSIR2 接口**: WSIR2 并不重新训练 patch-level classifier，而是接收 patch embeddings X，再压缩成更小的 token set。因此它天然和 embedding-based MIL 对接。

The paper lists representative MIL methods including MIL-RNN, CLAM, TransMIL, HIPT, HAG-MIL, DTFD-MIL, Remix, MHIM and IBMIL.

> 💡 **baseline 批读**: Table 1/2 中真正参与对照的是 Mean/Max、ABMIL、DSMIL、TransMIL、CLAM、DTFD-MIL、MHIM。Related Work 中列更多方法，是为了说明 WSI MIL 已经很成熟，但效率仍被忽略。

In this work, the authors specifically explore redundancy reduction within these mechanisms and introduce cross-attention with learnable queries. This design enhances scalability and generalization, enabling the method to integrate into a variety of attention-based architectures while preserving efficiency.

> 💡 **差异点**: 不是替代所有 MIL 架构，而是在已有 MIL token bag 前做 compression，并用 learnable query 做 O(n) 聚合。这解释了 Table 3 为什么要把 WSIR2 接入 ABMIL、DSMIL、TransMIL。

## 2.2 Redundancy reduction for vision tasks

Redundancy reduction was first proposed in neuroscience and has been used to explain visual system organization, supervised/unsupervised learning, vision model pre-training and efficient model construction.

> 💡 **理论背景**: 引 Barlow、Barlow Twins、image compression 等工作，是为了把“减少重复信息”放在更宽的 representation learning 语境中。WSIR2 的独特之处在 WSI 场景和 MIL token bag。

For efficient vision transformers, related work includes sparse attention, dynamic token filtering, class-token-to-patch reorganization, and lightweight token merging.

> 💡 **迁移关系**: DynamicViT/A-ViT/ToMe 等自然图像方法通常面对固定尺寸 ViT token 序列。WSI 的 token 数量更大、slide label 更弱、病灶更稀疏，不能直接照搬逐层 ViT compression。

Nevertheless, these studies primarily focus on subject-specific natural images with fixed sizes. Applying redundancy reduction to gigapixel WSIs is rarely explored, and the prevalence of monotonous biological patterns within WSIs motivates WSIR2.

> 💡 **研究缺口**: 自然图像 token pruning 常依赖分类 token 或逐层 self-attention；WSI 不能直接用 vanilla ViT 处理全部 tokens，所以 WSIR2 用单个 learnable query 在 feature bag 上做 selection/merging。

## 与 Medical-Compression 主题的横向位置

| 方法 | 压缩对象 | 主要监督/信号 | 输出目标 |
|---|---|---|---|
| WISE | WSI 文件 byte stream | lossless coding rules | bit-exact storage |
| AdaSlide | patch image content | RL decision + FIE reconstruction | diagnosis-preserving storage |
| FOCUS | WSI visual tokens | foundation model + language prompt relevance | few-shot WSI classification |
| DTC-WSI | multi-stage WSI tokens | importance + similarity | efficient MIL classification |
| WSIR2 | patch token number + feature dimension | cross-attention score | efficient MIL classification |

> 💡 **横向比较批注**: WSIR2 和 DTC-WSI 最接近，但 WSIR2 更强调 cross-attention 的 O(n) 聚合与 lightweight classifier；DTC-WSI 更强调动态多阶段 token management。二者都不是图像文件压缩，而是诊断 pipeline 内部表征压缩。

## 🔖 Section 总结

### 核心洞察

1. WSIR2 站在 embedding-based MIL 之上，改的是 token bag 的大小和聚合复杂度。
2. 视觉 token pruning/merging 提供技术灵感，但 WSI 的 gigapixel、弱监督、小病灶风险让问题更难。
3. 论文的 Related Work 主要服务于一个论点：WSI MIL 已经追求 accuracy 很久，但冗余和效率仍缺少系统处理。

### 可追问点

- 是否存在 WSI-specific token compression baseline 未充分纳入?
- natural image token merging 的 spatial assumptions 在 WSI 中是否仍成立?
- cross-attention score 与 CLAM/ABMIL attention map 的可解释性差别是什么?
