[← 返回 README](../README.md)

# 4. Results & Analysis

## 📌 预览

本节是实验评估的核心，包含实验设置（4.1）、主结果（4.2）、Cross-Teacher Agreement 分析（4.3）和消融实验（4.4）。核心结论：CARES 在 9 个 benchmark、4 个 VLM 上平均降低 63-78% prefill FLOPs，准确率基本持平。消融实验验证了中间层特征、三元分类、连续推理和 label smoothing 的设计选择。

---

## 4.1 Experimental Setup

**Training Data.** To train the resolution selector, we construct a dataset of images and queries (x, q) we automatically annotated with the minimal sufficient resolution r*. We construct an 80K-sample training set by randomly sampling 20K instances from each of four datasets: TextVQA (Singh et al., 2019), ChartQA (Masry et al., 2022), DocVQA (Mathew et al., 2021), and LLaVA-Multi (Jiang et al., 2024), covering documents and natural images domains.

> 💡 **批注**: 训练数据覆盖了两个域（documents + natural images），但只有 4 个源数据集。这引发两个问题：(1) 训练没见过的 benchmark 类型（如 MathVista 的数学推理、MMMU 的多学科理解）能否被充分泛化？(2) 80K 样本对于训练一个 350M 参数的模型似乎偏少，但考虑到只有分类器头是可训练的而 backbone 冻结，这个量级也算合理。

**Training details.** We train CARES on the curated data described in 3.2 for 6 epochs using a learning rate of 1e-3 and a batch size of 32. We optimize the standard cross-entropy loss over the fixed resolution labels:

L(theta) = CE(f_theta(z), r*)

Where f_theta(z) is CARES composed of a frozen VLM and the lightweight classifier. In addition, we apply label smoothing of 0.05 to support continuous resolutions at inference time.

> 💡 **批注**: label smoothing = 0.05 的作用是在离散训练的背景下软化类别边界，为连续推理提供概率基础。Table 6 的消融验证了它的效果（OCRBench: 0.821 vs 0.811 without smoothing）。

**VLM variant training details.** For the autoregressive (AR) Granite-Docling instantiation, we use the same training set and the same discrete supervision labels. The model is fine-tuned with LoRA of rank 8, while the base model remains frozen. Training is performed with next-token supervision over the resolution tokens, and for efficiency, generation length is set to 1. Learning rate is set to 1e − 5 and a batch size of 64 for 3 epochs.

> 💡 **批注**: AR 变体的 generation length = 1 意味着只预测一个 token（分辨率标签），所以本质上退化为一个特殊的分类——用 next-token prediction 做 "classification"。

**Evaluation.** We evaluate on nine public benchmarks varying from documents to natural images: Ai2D (Kembhavi et al., 2016), ChartQA (Masry et al., 2022), DocVQA (Mathew et al., 2021), OCR-Bench (Liu et al., 2024b), and SeedBench-2 (Li et al., 2023), MMMU (Yue et al., 2024), Real-WorldQA (xAI, 2024), InfoVQA (Mathew et al., 2022) and MathVista (Lu et al., 2024). For Ai2D, ChartQA, and SeedBench-2 we report exact-match accuracy. For DocVQA and OCRBench we report Average Normalized Levenshtein Similarity (ANLS). All evaluations were performed with the standard lmms-eval (Zhang et al., 2024) setup. We also report a macro-averaged Performance (%) across all datasets.

> 💡 **批注**: 9 个 benchmark 按任务类型可以分组：(1) 文档理解：DocVQA, OCRBench, InfoVQA；(2) 图表/数据：ChartQA, Ai2D；(3) 自然图像：SeedBench-2, RealWorldQA；(4) 综合推理：MMMU, MathVista。不同类型对分辨率的需求差异很大——文档类通常需要高分辨率做 OCR，自然图像类很多 query 用低分辨率就够了。

> 💡 **Figure 3 批读**: Fig.3 展示 DocVQA 上 Qwen2.5-VL-72B 的 Accuracy vs. TTFT 曲线。CARES 在 2.58 TFLOPs 处达到接近 native accuracy，而 native 需要 7.5 TFLOPs。固定 1024^2 分辨率虽然 accuracy 高但 FLOPs 开销大；固定 384^2 虽然 TTFT 低但 accuracy 差。CARES 取得的是 Pareto-optimal 点。

> 💡 **Figure 4 批读**: Fig.4 展示 OCRBench 上 CARES 预测的连续分辨率分布直方图。分布是连续的（而非三个离散峰值），说明 continuous inference 确实产生了比离散分类更丰富的输出。

## 4.2 Main Results

We evaluate CARES across Granite-Vision 3.3-2B (Team et al., 2025), InternVL3-8B (Zhu et al., 2025), Qwen2.5-VL-72B (Bai et al., 2025), and GPT-4o (Achiam et al., 2023). We also report prefill-stage FLOPS savings for locally run models, and estimated dollar savings in API usage for GPT-4o. As summarized in Table 2, CARES maintains accuracy while cutting prefill compute: averaged over models and datasets, prefill FLOPs drop by 65–85% with at most a sub-point change in macro performance relative to always using the highest/native resolution. The effect is consistent for compact (Granite-Vision 3.3-2B) and large (Qwen2.5-VL-72B) backbones, and holds for GPT-4o accessed via API (accuracy parity at comparable quality).

> 💡 **Table 2 批读**: Table 2 是整篇论文最重要的结果表。关键观察：
> - Granite-Vision-2B + CARES: 平均 Score 0.60 vs native 0.59（还略有提升！），FLOPs -63%。
> - InternVL3-8B + CARES: 平均 Score 0.77 vs native 0.77（完全持平），FLOPs -64%。
> - Qwen2.5-VL-72B + CARES: 平均 Score 0.80 vs native 0.79（略有提升），FLOPs -70%。
> - GPT-4o + CARES: 平均 Score 0.68 vs native 0.69（几乎持平），Cost -55%。
>
> 💡 **批注**: 有几个 benchmark 上 CARES 的 cost savings 特别低：

> - MathVista: 只节省 7-31%（取决于模型）。可能原因：数学题目通常包含图表中的小文字/公式，确实需要高分辨率才能正确读取。
> - SeedBench-2 在部分模型上节省较少（44% for Granite-Vision-2B），但 Qwen2.5-VL-72B 上节省 77%。这可能与不同 VLM 的视觉能力差异有关。
> - OCRBench 在 GPT-4o 上只节省 33%——说明 OCR 任务在 GPT-4o 上确实需要较高分辨率。
>
> 💡 **批注**: CARES 在某些 benchmark 上 Score 甚至略高于 native，这看起来 counter-intuitive。可能的原因：(1) 更高分辨率并不总是帮助——有时会引入噪声/干扰，降低分辨率反而让模型更关注全局信息；(2) 统计噪声，benchmark score 本身有方差。

Fig. 3 shows the accuracy–latency frontier: CARES matches near-native accuracy while using far fewer TFLOPs (e.g., 2.58 vs. 7.5) and achieving ~1 second lower time-to-first-token (TTFT); static high-res inputs (e.g., 1024^2) incur substantial compute with limited TTFT gains, whereas fixed low-res (384^2) improves TTFT at the cost of quality. The query-aware routing yields a superior Pareto point.

> 💡 **批注**: Fig.3 的 Pareto 分析是评估 CARES 质量最直观的方式——如果 CARES 不能在相同的 accuracy 下提供更低的 compute，那它就只是 naive downscaling。而 Fig.3 清楚地显示 CARES 处于固定分辨率的 Pareto frontier 之上。

Finally, the distribution of predicted continuous resolutions r_tilde (Fig. 4) and the comparison in Table 5 indicate that continuous routing adapts per instance, matches or slightly improves accuracy over a discrete menu, and saves additional compute without quality loss.

## 4.3 Cross-Teacher Agreement for Resolution Labels

Because our supervision is generated by rolling out a pretrained VLM at multiple resolutions, one natural question is whether the resulting labels depend strongly on the specific annotating model. To test this, we compare labels generated by two substantially different teachers: Granite-Vision-2B and Qwen3-VL-235B, on a shared subset of 1000 examples.

We find a high degree of agreement between the two annotators. The two teachers predict the same sufficient resolution for more than 95% of examples, with Pearson correlation 0.908 and mutual information 1.116 between their predicted sufficiency levels. The confusion matrix is shown in Table 5 (Figure 5). These results suggest that the notion of sufficient resolution is largely shared across architectures and scales, and is not tied to a single model family.

> 💡 **批注**: Cross-teacher agreement >95% 是本文最强有力的发现之一。它说明 "充分分辨率" 不是某个特定模型的 idiosyncrasy，而是一个跨架构、跨尺度的通用概念。这为 CARES 的 cross-model transfer（用一个 teacher 训练，应用到多个 target VLM）提供了理论支撑。

> 💡 **Figure 5 批读**: Fig.5 的 confusion matrix 显示 Granite-Vision-2B 和 Qwen3-VL-235B 在分辨率标签上的高度一致（对角线占据绝大多数 mass）。两个模型之间的主要不一致出现在 384 vs 768 之间（而非 384 vs 1024），说明它们对"中等难度"样本的边界判断有细微差异。

This result complements the downstream transfer results in the main paper, where a selector trained using labels derived from one setup transfers well across multiple target VLMs. Together, these findings support the view that CARES captures a broadly shared notion of resolution adequacy, rather than overfitting to one teacher's idiosyncrasies.

> 💡 **批注**: 这一段是 connection paragraph——把 cross-teacher agreement (4.3) 和 cross-model transfer (4.2) 联系起来。结论是 CARES 学习的是 resolution adequacy 本身，而非某个 teacher 的特殊偏好。

## 4.4 Ablation Study

We conduct a series of ablations to isolate the effect of key training design choices on resolution selection accuracy and downstream benchmark performance.

**Feature extractor.** We ablate several frozen backbones used for feature extraction in CARES, varying both model type and layer depth. As shown in Table 3, both Qwen2.5-3B and SmolVLM achieve higher accuracy when using intermediate-layer features, outperforming their own final-layer variants. This aligns with prior findings suggesting that intermediate representations in VLMs often encode richer signals than final outputs.

> 💡 **Table 3 批读**:
> - SigLIP v2 (0.8B): 56.1% → 最差，因为它是 dual-encoder 架构（vision 和 language 分开编码），缺乏 joint image-query modeling。
> - SmolVLM Mid (0.35B): 63.3% → 默认选择，performance/size 最优。
> - SmolVLM Last (0.5B): 62.3% → 中间层优于最后层（+1.0%），同时少 150M 参数。
> - Qwen2.5-3B Mid (2.3B): 67.2% → 最好但参数量大（6.5x of SmolVLM Mid）。
> - Qwen2.5-3B Last (3.75B): 66.2% → 同样中间层优于最后层（+1.0%）。

Qwen2.5-3B and SmolVLM both process the image and query jointly within a unified transformer, in contrast to SigLIP v2's dual-encoder architecture, where vision and language are encoded separately. For SigLIP, we follow the original design by pooling the outputs of each tower, concatenating them, and passing the result to the classifier head. While this setup is architecturally simple, it underperforms joint encoding by a considerable margin (56.1% accuracy), and it requires more parameters than the lightweight SmolVLM.

> 💡 **批注**: SigLIP 的表现证实了 joint image-query encoding 的重要性。分辨率需求不是单纯的"图像复杂度"问题，而是"图像+query"的联合属性——同一张图对不同 query 可能需要不同分辨率。Dual-encoder 无法捕获这种交互。

Although Qwen2.5-3B achieves the best overall accuracy, we adopt SmolVLM as our default backbone due to its favorable trade-off between performance, size, and efficiency, making it a more practical choice for real-world pre-processing.

**Resolution menu size.** We compare training with binary Rd = {384, 1024} (|Rd| = 2) vs. ternary Rd = {384, 768, 1024} (|Rd| = 3) resolution choices. Table 4 reports both the classification accuracy and the downstream performance of Granite Vision, averaged over 5 benchmarks. As expected, the two-way classification yields higher validation accuracy in the resolution classification task compared to the more challenging three-way classification. But the ternary setup leads to better downstream benchmark performance due to the finer-grained control.

> 💡 **Table 4 批读**:
> - Binary (|Rd|=2): Resolution Accuracy 96.2%, Downstream Accuracy 0.76。
> - Ternary (|Rd|=3): Resolution Accuracy 67.2%, Downstream Accuracy 0.80。
>
> 二元分类准确率高（任务更简单），但下游性能低——因为缺少 768 这个中间选项，很多样本被迫在 384 和 1024 之间二选一，导致要么分辨率不足要么过多的像素浪费。三元分类虽准确率低（67.2%），但加上 continuous inference（softmax 期望插值）后，实际分辨率选择更接近需求。

**Discrete vs. continuous.** CARES is trained as a discrete resolution classifier, but at inference time, it can produce either discrete predictions or a continuous estimate via interpolation. In Table 5, we compare the impact of discrete versus continuous inference across three VLM backbones. All scores and FLOPS deltas are averaged over nine benchmarks. We find that continuous resolution selection achieves comparable accuracy to both discrete and native strategies, while significantly reducing compute. For example, with Granite-Vision 3.3-2B and InternVL3-8B, FLOPS are reduced by 63% using continuous prediction, compared to 46% with discrete. These results suggest that continuous inference allows finer control over input resolution and leads to more efficient inference without compromising performance.

> 💡 **Table 5 批读**:
> - Granite-Vision-2B: Native 0.803, Discrete -46% FLOPs, Continuous -63% FLOPs（Score 0.804，甚至略高于 native）。
> - InternVL3-8B: Native 0.851, Discrete -46%, Continuous -63%（Score 不变）。
> - Qwen2.5-VL-72B: Native 0.851, Discrete -74%, Continuous -80%（Score 略有下降 0.839 vs 0.851）。
>
> Continuous inference 相比 discrete inference 的额外 FLOPs 节省来自更细粒度的分辨率选择——中间分辨率（如 500-700）比 768 更省计算，但比 384 提供更多视觉细节。

**Label smoothing.** To bridge the mismatch between discrete supervision and our continuous inference policy, we apply label smoothing when training the classifier over Rd. Smoothing softens class boundaries and discourages over-confident logits, yielding better-calibrated probability distributions p that are subsequently mapped to a scalar resolution via expectation (Eq. 3). This improves the stability of the continuous selector, reduces spurious hard escalations near decision thresholds, and translates to higher downstream utility at similar—or lower—compute. Empirically, Table 6 shows that adding label smoothing improves OCR-Bench performance for Qwen2.5-VL-7B (0.821 vs. 0.811) while slightly reducing expected FLOPS, supporting its role as a simple but effective regularizer for continuous-resolution deployment.

> 💡 **Table 6 批读**: 在 OCRBench + Qwen2.5-VL-7B 上：
> - Native: 0.824 Score。
> - CARES w/o label smoothing: 0.811 Score, -60.5% FLOPs。
> - CARES w/ label smoothing: 0.821 Score, -63.8% FLOPs。
>
> Label smoothing 同时提升了 accuracy（+1.0%）和 compute savings（+3.3%）。这是因为平滑后的概率分布更"适度"，避免了在 384 和 768 之间的硬切换导致的 over/under-allocation。

---

## 🔖 Section 总结

### 核心洞察
1. 主结果的核心信息不是 "CARES 做得很好"，而是 "大多数视觉 token 是多余的"——在不用最高分辨率的情况下，9 个 benchmark 的准确率几乎不变这个事实本身就是强有力的发现。
2. Cross-teacher agreement >95% 是本文最有原创性的实验结果之一，它为 "充分分辨率是通用概念"提供了实证支持。
3. 消融实验的结论分层：(a) 联合编码优于双塔；(b) 中间层优于最后层；(c) 三元优于二元；(d) 连续优于离散；(e) label smoothing 是必需的。
4. MathVista 的低节省率（7-31%）提示了一个边界：当 query 本身涉及需要精确读取的视觉符号（如数学公式），高分辨率可能是不可省略的。
