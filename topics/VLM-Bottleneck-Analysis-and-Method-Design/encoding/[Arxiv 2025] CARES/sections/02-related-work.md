[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览

本节建立 CARES 在文献中的位置。相关工作分为五条线：(1) visual token sparsification at inference（token 后处理）、(2) training for flexible token budgets（训练时适应多预算）、(3) Any-resolution inputs and tiling（多分辨率架构）、(4) dynamic computation（动态计算）、(5) adaptive input resolution selection（自适应分辨率选择，最接近 CARES 的工作）。CARES 的定位是与 (1)(2)(3)(4) 互补、且在 (5) 的基础上加了 query-conditioning 和多模态 QA 场景。

---

## Visual-token sparsification at inference

A growing line of work trims visual tokens after tokenization inside the VLM stack. HiRED uses [CLS] attention to allocate a per-partition token budget and drop the least-informative vision tokens under a fixed budget, yielding large speedups on high-resolution inputs without retraining (Arif et al., 2025). SparseVLM proposes a training-free, text-guided strategy: self-attention matrices rank visual tokens with an adaptive layer-wise sparsification ratio and a token-recycling mechanism to preserve information (Zhang et al., 2025c). PyramidDrop stages the model and progressively reduces tokens at stage boundaries, motivated by the observation that redundancy increases with depth; it accelerates both training and inference and can also be used in a plug-and-play inference mode (Xing et al., 2025). Complementary to these, Visual Tokens Withdrawal (VTW) argues that visual information migrates to text tokens in early layers and thus withdraws vision tokens beyond a learned layer, cutting compute while maintaining quality (Lin et al., 2025). In contrast, CARES decides before tokenization which input resolution to use and leaves all VLM's components frozen.

> 💡 **批注**: 这条线是 CARES 最直接的相关工作，也是 CARES 刻意要区分开来的。核心差异点：HiRED/SparseVLM/PyramidDrop/VTW 都在 VLM 内部、tokenization 之后操作；CARES 在 VLM 外部、tokenization 之前。这个差异意味着 CARES 可以和上述方法叠加。

> 💡 **批注**: 值得注意 SparseVLM 也用了 "text-guided"——它用 attention matrix 来 rank visual tokens，所以它至少是 text-aware 的。但它的操作对象仍是已 tokenized 的 visual tokens，所以还是和 CARES 处于不同层次。

## Training for flexible token budgets

Token-FLEX trains VLMs to operate across a range of visual–token counts by stochastically modulating tokens during training and adding a lightweight projector with adaptive pooling (Hu et al., 2025). Matryoshka Multimodal Models (MMM) further pursue elastic compute, training nested representations that remain useful under progressively smaller token/feature budgets (Cai et al., 2025). LLaVA-Mini pushes efficiency to the extreme by compressing visual information into (nearly) a single vision token while retaining competitive performance for both images and videos (Zhang et al., 2025b). CARES targets the complementary axis of adaptive pixel allocation before tokenization: it selects the minimal input resolution needed for a target utility and can front-end TokenFLEX/Matryoshka/LLaVA-Mini–style models to reduce pixels (and thus tokens) further.

> 💡 **批注**: 这条线是"训练时让模型适应不同 token 预算"。CARES 的思路不同：不是让模型学会在各种 budget 下工作，而是预测哪种 budget 对当前 query 是充分的。两者互补——CARES 可以在前面选分辨率，TokenFLEX/MMM 在后面处理选定的分辨率。

> 💡 **批注**: LLaVA-Mini 的 "one vision token" 思路和 CARES 有交集但也有冲突：如果 CARES 已经选了极低分辨率，LLaVA-Mini 可能就不再需要进一步压缩了。两者的最优组合策略值得探索。

## Any-resolution inputs and tiling

Many modern ViTs (Dehghani et al., 2023; Beyer et al., 2023) and VLMs boost fine-grained perception with AnyRes/dynamic-high-resolution tiling (e.g., LLaVA-NeXT) or native dynamic resolution that maps larger images to more tokens (e.g., Qwen2-VL) (Liu et al., 2024a; Wang et al., 2024). While effective, these strategies often increase visual tokens substantially. CARES explicitly avoids unnecessary tiling by routing easy cases to low resolutions and only escalating when the query and low-res cues predict a benefit.

> 💡 **批注**: AnyRes/tiling 是 VLM 社区解决"图像细节需求"的主流方案，但它的问题是"无差别放大"——每个输入都会触发 tiling，不管 query 是否需要。CARES 的贡献正是在 tiling 前面加一个 query-conditioned gate，避免不必要的 tiles。

## Dynamic computation

Vision-only methods reduce computation via token pruning/merging inside ViTs-e.g., DynamicViT prunes tokens hierarchically with learned importance (Rao et al., 2021), EViT reorganizes/discards inattentive tokens (Liang et al., 2022), and ToMe merges similar tokens on the fly (Bolya et al., 2023). WAVE-CLIP replaces patch tokenization with a multi-level wavelet tokenizer and performs coarse-to-fine inference in a single ViT (Kimhi et al., 2025b). For VLMs, SGL routes easy cases via a small 'stitch' model and defers hard ones to a larger counterpart, akin to early-exit routing (Zhao et al., 2024). These operate within the encoder after tokenization; CARES is complementary, deciding how many pixels to tokenize in the first place.

> 💡 **批注**: SGL 的思路和 CARES 有些相似——都是用一个小模型做 routing，不同的是 SGL 做的是 model routing（easy→small model, hard→large model），而 CARES 做的是 resolution routing（easy→low-res, hard→high-res）。两者可以组合：CARES 选分辨率，SGL 选模型。

## Adaptive input resolution selection

Outside VLMs, dynamic-resolution networks learn a per-image resolution predictor that trades accuracy for cost in classification (Zhu et al., 2021). CARES brings this idea to multimodal QA, conditions the policy on the query text, and supervises it with per-example multi-resolution rollouts of the target VLM using a sufficiency rule, which yields unambiguous labels at deployment resolutions.

> 💡 **批注**: Dynamic Resolution Network (Zhu et al., 2021) 是最直接的前驱——但它是在 image classification 场景下，且只 condition on image（没有 text query）。CARES 的三个拓展：multimodal QA 场景、query-conditioned、用 VLM rollout 做监督（而不是人工定义 cost-accuracy trade-off）。

## Extreme compression and design insights

Recent analyses argue that, under fixed inference budgets, compute-optimal VLMs may prefer very few visual tokens and a larger LLM (Li et al., 2024). Such results support approaches that minimize visual tokens when possible; methods like LLaVA-Mini instantiate the "one-token vision" regime in practice (Zhang et al., 2025b). CARES provides a query-conditioned mechanism to reduce pixels upstream, complementing these token-minimal designs.

> 💡 **批注**: Li et al. (2024) 的结论"compute-optimal VLMs prefer very few visual tokens + larger LLM"是 CARES 的重要理论支撑——它从 scaling law 角度论证了"减少 visual tokens 是正确方向"。CARES 提供的是实现这个方向的 query-conditioned 手段。

---

## 🔖 Section 总结

### 核心洞察
1. CARES 的文献定位是"与其他效率方法正交互补"。这个定位在整篇论文中被反复强调，目的是避免审稿人认为"这不就是另一种 token pruning 吗"。
2. 最直接的前驱是 Dynamic Resolution Network（image classification），CARES 把它拓展到了 multimodal QA + query-conditioned + VLM rollout supervision。
3. 五条相关工作线可以按"介入层次"排列：pixel allocation (CARES) → tokenization → token sparsification (HiRED 等) → dynamic computation (DynamicViT 等)。
4. "compute-optimal VLMs prefer few visual tokens"这一结论给了 CARES 来自 scaling law 的理论支持。
