[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览

本节建立研究动机：VLM 普遍使用高分辨率导致 visual token 爆炸（可占 99%），但大多数 query 不需要这么多视觉细节。现有效率方法多在 tokenization 后操作 visual tokens（pruning/merging），而不碰输入分辨率这个更根本的杠杆。CARES 的创新在于："能不能在预处理阶段就决定用多少像素？"

---

Large vision–language models (VLMs) are increasingly used as general-purpose systems that solve a broad variety of visual tasks using a single model. Since the complexity and nature of each task are not known in advance, these models typically process images at very high resolutions to preserve the visual detail necessary for any potential query. This leads to a sharp increase in the number of visual tokens, as modern architectures map higher resolutions to proportionally more tokens. Strategies like AnyRes and tiling further increase token counts in order to capture fine-grained information (Liu et al., 2024a; Wang et al., 2024). In practical settings, visual tokens make up to 99% of all tokens processed per request, which significantly impacts latency and memory consumption (Fig 1), even when the actual query may only require a coarse understanding of the scene.

> 💡 **批注**: 第一段建立"问题"。关键论据：visual tokens 占 99%（在 Qwen2.5-VL 4096x4096 下）。这个数字本质上是 token 组成分析：text tokens 相对固定（约 20-100），而 visual tokens 随分辨率平方增长。所以瓶颈不在 LLM 推理，而在视觉编码阶段。

> 💡 **Figure 1 批读**: Fig.1 展示 visual token 占比随分辨率的变化，对比了 Qwen2.5-VL（quadratic scaling）、AnyRes（tile-saturated scaling）和 InternVL3（stepwise tile-based scaling）三种 tokenization 策略。关键结论：无论哪种策略，中高分辨率下 visual tokens 都 dominate context window。

A key observation is that not all queries require the same visual granularity. Coarse queries (e.g., "What is the breed of the dog?") are typically answerable from a small image; fine-grained queries (e.g., "What is the name on the collar?") benefit from higher resolution. Existing efficiency methods typically operate after tokenization, on the output of the vision encoder -pruning, pooling, merging, or compressing with Q-former style architecture (Arif et al., 2025; Zhang et al., 2025c; Xing et al., 2025; Lin et al., 2025; Rao et al., 2021; Liang et al., 2022; Bolya et al., 2023; Hu et al., 2025; Cai et al., 2025). While complementary, these methods typically operate on the output of the visual encoder alone and are unaware of the text input or the current query. Yet a more fundamental lever remains untouched: Can we choose the input granularity as a pre-processing step?

> 💡 **批注**: 第二段区分了 CARES 和现有方法的层次差异。关键是"text-unaware"——token pruning 只看视觉特征，不知道用户问什么，可能把 query 关心的区域 pruned 掉。CARES 的 query-conditioned 设计正是针对这一点。

> 💡 **批注**: "a more fundamental lever" 这个表述很重要，它暗示 CARES 不是在和 token pruning 竞争，而是在一个更上游、更根本的维度上工作——pixel allocation before tokenization。

We propose a Context-Aware Resolution Selector (CARES), a lightweight model that, for a given image-query pair, selects the minimal sufficient resolution to answer the query (Fig. 2). CARES is model-agnostic, placed in front of an arbitrary VLM. While our main instantiation uses a compact frozen VLM with a lightweight discriminative classifier, the CARES formulation is not tied to a specific predictor architecture. We also study a closely related autoregressive instantiation based on Granite-Docling, fine-tuned with LoRA, and report it separately on document-centric benchmarks.

> 💡 **批注**: 这里提到 CARES 的 formulation 不限于特定架构，但主要 instantiation 用 VLM 做特征提取 + 分类器头。AR 变体的存在说明 CARES 是一个"范式"而不仅仅是"一个模型"。

> 💡 **Figure 2 批读**: Fig.2 是 CARES 的 overview 图。核心信息流是：低分辨率 pass → CARES predictor → 分辨率 r → image resize → target VLM。右侧用两个例子说明不同 query 会导致不同分辨率选择（"What animal?" → 低分辨率；"License plate?" → 高分辨率）。

It operates in three steps:

* A cheap low-resolution pass (e.g., <= 384^2) extracts a joint image–query representation using a small proxy VLM.
* Given this representation, a lightweight classifier predicts the minimal resolution required for the task.
* The image is resized to the predicted resolution and passed to the target VLM. No changes to the VLM's architecture, weights, or training are required.

> 💡 **批注**: 三步 pipeline 体现了 CARES 的核心设计哲学：preprocessing（不改 VLM）、cheap（低分辨率 pass 便宜）、query-conditioned（同时看图和问题）。

A central challenge is supervision: what resolution is truly sufficient for each example? We introduce a simple labeling procedure based on a discrete set of resolutions R and a task performance metric. For each image, query, and GT response, we evaluate a pretrained VLM with increasingly higher resolution up to convergence in terms of the task metric (or reaching the native resolution). The lowest resolution at which the convergence occurs is selected as the ground-truth optimal resolution for training CARES. Using a discrete resolution set avoids the cost of exhaustively searching over continuous values. Since the labels are discrete, the model is trained as a classifier. At inference time, however, we interpolate between the predicted class probabilities to recover a continuous resolution estimate.

> 💡 **批注**: 标注策略是本文最核心的 contribution 之一。关键设计点：用目标 VLM 自己来定义"充分性"，而不是人工判断——这保证了标签和下游部署的一致性。收敛规则（ANLS >= tau AND max improvement at higher resolutions <= delta）避免了为微不足道的性能提升而升级分辨率。

> 💡 **批注**: "discrete labels → classifier training → continuous interpolation at inference" 这个 pipeline 设计得相当优雅。它解决了两个矛盾：(1) 连续标注太贵 vs 连续推理需要细粒度控制；(2) label smoothing 为连续插值提供更好的概率分布。

Across 9 multimodal benchmarks, varying from natural images to document understanding (Section 4) and different open and api-based model, CARES reduces average visual tokens and GFLOPS by 70-80%, with minimal to no accuracy drop compared to always using the highest (native) resolution.

> 💡 **批注**: 70-80% 的节省是平均值。但要注意：(1) 节省主要集中在哪些 benchmark？文档任务可能节省更多（因为很多 query 不需要高分辨率），但 OCR-heavy 任务可能节省较少；(2)"minimal to no accuracy drop"需要仔细检查每一行 Table 2。

## Our contributions are as follows:

1. We define the task of query- and image-conditioned resolution selection for vision-language models, aimed at reducing input size without sacrificing accuracy.

2. We propose a simple yet effective supervision strategy based on multi-resolution rollouts and a convergence rule, yielding per-example sufficient resolution ground-truth, enabling training and evaluation.

3. We introduce CARES, a lightweight, model-agnostic module that selects resolution as a pre-processing step, requiring no changes to the target VLM.

4. We demonstrate that many visual tokens are unnecessary: CARES preserves performance across tasks while reducing visual compute by up to 78% on average across 9 benchmarks, and is orthogonal with post-tokenization token compression.

> 💡 **贡献批读**: 四个贡献分层：任务定义层（contribution 1）、标注方法论层（contribution 2）、模块设计层（contribution 3）、实证验证层（contribution 4）。contribution 4 特别提到 "orthogonal with post-tokenization token compression"——这是 alignment 声明，说明 CARES 不和其他效率方法冲突。

---

## 🔖 Section 总结

### 核心洞察
1. CARES 的核心动机不是技术推动（"我们可以做 adaptive resolution"），而是需求驱动（"visual tokens 占 99% 但大多数 query 不需要"）。
2. 三个设计原则（compactness, preprocessing role, VLM-agnosticism）贯穿全文，每个设计选择都需要回到这三个原则来审视。
3. 标注策略是整个 pipeline 的基础——如果"充分分辨率"定义不合理，CARES 预测出的分辨率也会出问题。后文需要检查 tau 和 delta 的敏感性。
4. "离散训练 + 连续推理"的设计是一个泛用的训练范式，不仅限于 CARES 本身。
