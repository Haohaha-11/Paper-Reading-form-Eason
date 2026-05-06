[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
本节建立论文的核心缺口：deep fusion 太重，shallow fusion 虽轻但缺少把视觉特征限制在 LLM text latent space 内的归纳偏置。ALIGN 的动机就是在轻量 connector 中显式复用 LLM 词表 embedding。

---

> 💡 **Q&A 批注记录**:
> - Q: 为什么论文把 deep fusion 和 shallow fusion 分开讲？
> - A: deep fusion 通过改 LLM 层结构获得强交互，但成本高；shallow fusion 保持 LLM 不变，真正的问题集中到 connector 是否能产生 LLM 熟悉的 input embeddings。

Vision-Language Models (VLMs) have gained significant traction in recent years as a powerful framework for multimodal document understanding tasks that involve interpreting both the visual and textual contents of scanned documents [Kim et al., 2022, Lee et al., 2023, Liu et al., 2023a, 2024, Hu et al., 2024, Wang et al., 2023a, Rodriguez et al., 2024b]. Such tasks are common in real-world commercial applications, including invoice parsing [Park et al., 2019], form reading [Jaume et al., 2019], and document question answering [Mathew et al., 2021b]. VLM architectures typically consist of three components: $( i )$ a vision encoder to process raw images, (ii) a Large Language Model (LLM) pre-trained on text, and (iii) a connector module that maps the visual features from the vision encoder into the LLM’s semantic space.

> 💡 **任务定位**: 文档理解不是一般图片问答，图像里直接包含 OCR 字符、表格结构、票据字段和图表坐标。视觉证据与语言 token 的相关性极高，所以 connector 的 latent 对齐质量会直接影响答案是否能读对字段和值。

A central challenge in this pipeline is to effectively map the continuous feature embeddings of the vision encoder into the latent space of the LLM while preserving the semantic properties of visual concepts. Existing approaches can be broadly categorized into deep fusion and shallow fusion methods. Deep fusion methods, such as NVLM [Dai et al., 2024], Flamingo [Alayrac et al., 2022],

> 💡 **问题抽象**: 视觉 encoder 输出是连续高维 patch features；LLM 的输入空间则是文本预训练塑造出来的 embedding manifold。论文关心的是从前者到后者的映射是否保持语义，而不是简单维度匹配。

CogVLM [Wang et al., 2023b], and LLama 3.2-Vision [Grattafiori et al., 2024], integrate visual and textual features by introducing additional cross-attention and feed-forward layers at each layer of the LLM. While effective at enhancing cross-modal interaction, these methods substantially increase the parameter count of the VLM compared to the base LLM, resulting in high computational overhead and reduced efficiency.

> 💡 **deep fusion 取舍**: Flamingo/CogVLM/NVLM 这类方法在 LLM 内部多层融合视觉信息，表达力强但改动大、参数和训练成本高。AlignVLM 要证明的是：不改 LLM 主体，只改 connector 也能显著改善文档理解。

In contrast, shallow fusion methods project visual features from the vision encoder into the LLM input embedding space using either multilayer perceptrons (MLPs) [Liu et al., 2023b, 2024], convolution mappings such as Honey-Bee [Cha et al., 2024] and H-Reducer [Hu et al., 2024], or attention-based mechanisms such as the Perceiver Resampler [Li et al., 2023b, Laurençon et al., 2024, Alayrac et al., 2022]. This approach is more parameter-efficient and computationally lighter than deep fusion method However, these connectors lack inductive bias to ensure that the projected features remain within the region spanned by the LLM’s pretrained text embeddings. Consequently, the projected visual features may fall outside the distribution the LLM was trained on, leading to noisy or misaligned representations. Moreover, these mappings are typically learned from scratch, making them data-inefficient and less effective under low-resource conditions.

> 💡 **shallow fusion 缺口**: MLP/conv/Perceiver 都可以把维度变成 $D$，但没有约束输出必须像 LLM 的 token embeddings。作者的关键批评是：这些 connector 学到的视觉 token 可能是 LLM 预训练从未见过的输入分布，低资源下尤其难纠正。

Recent methods like Ovis [Lu et al., 2024] attempt to alleviate these issues by introducing separate visual embeddings indexed from the vision encoder outputs and combined together

> 💡 **Ovis 对照意义**: Ovis 用视觉 embedding table 给视觉 token 一个离散/结构化表示，但那是一套新学的 visual embedding space。AlignVLM 的差别是直接复用 LLM 已有 text embedding space。

![Figure 1](../images/15cb8c8bfd63b69a2c71aa2651f41aaea8661e208a8ddd60b9264e94a7288832.jpg)
*Figure 1: Figure 1: Performance of Different VLM Connectors. The proposed ALIGN connector outperforms other methods across benchmarks using the same training configuration. Radial distance is proportion of maximal score, truncated at 0.7 (black dot).*

> 💡 **Figure 1 批读**: 雷达图把不同 connector 的平均优势可视化：ALIGN 在相同训练配置下整体半径更大。它是引言中的“现象图”，真正的受控数字在 Table 2：ALIGN 58.81，高于 Ovis 54.72、MLP 53.06、Perceiver 50.68。

to construct the visual inputs to the LLM. However, this approach significantly increases parameter count due to the massive embedding matrix and requires extensive training to learn a new embedding space without guaranteeing alignment with the LLM’s input latent space.

> 💡 **为什么新 embedding table 不够**: 新增 visual embeddings 可以正则化视觉输入，但它仍要从头学“哪些向量对 LLM 有意义”。ALIGN 直接拿 text embeddings 做基底，少学一整套语义空间。

To address these limitations, this paper introduces ALIGNVLM, a novel framework that sidesteps direct projection of visual features into the LLM embedding space. Instead, our proposed connector, ALIGN, maps visual features into probability distributions over the LLM’s existing pretrained vocabulary embeddings, which are then combined into a weighted representation of the text embeddings. By constraining each visual feature as a convex combination of the LLM text embeddings, our approach leverages the linguistic priors already encoded in the LLM’s text space. This ensures that the resulting visual features lie within the convex hull of the LLM’s embedding space, reducing the risk of noisy or out-of-distribution inputs and improving alignment between modalities. The connector thus enables faster convergence and stronger performance, particularly in low-resource scenarios.

> 💡 **ALIGN 核心机制**: “probability distribution over vocabulary embeddings” 是全文关键。每个视觉 patch 不被硬映射成某个词，而是被表示为许多 text embeddings 的软组合；这保留连续性，同时把输出限制在 text embedding convex hull 内。

Our experimental results show that ALIGN improves performance on various document understanding tasks, outperforming prior connector methods, with especially large gains in low-data regimes. We summarize our main contributions as follows:

> 💡 **证据预告**: 低数据收益是这篇很重要的 claim，因为“更强归纳偏置”如果成立，最应该在样本不足时拉开差距。Table 3 后面正是用 LLaVA-NeXT 低资源设置验证这一点。

• We propose a novel connector, ALIGN, to bridge the representation gap between vision and text modalities.   
• We introduce a family of Vision-Language Models, ALIGNVLM, that achieves state-of-theart performance on multimodal document understanding tasks by leveraging ALIGN.   
• We conduct extensive experiments demonstrating the robustness and effectiveness of ALIGN across different LLM sizes and training data setups.

> 💡 **贡献批读**: 三条贡献分别对应方法、模型族和实验。真正新意集中在 connector；ALIGNVLM family 和多 LLM size 实验主要用于证明该 connector 不是只对单一 backbone 有效。

We release our code and research artifacts at alignvlm.github.io.

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| VLM 组成 | vision encoder + LLM + connector |
| 现有路线 | deep fusion vs shallow fusion |
| ALIGN 约束 | visual token = text embeddings 的凸组合 |

### 核心洞察
1. 论文不是在追求更大 VLM，而是在修 shallow fusion 的 latent mismatch。
2. deep fusion 提供强交互但成本高；MLP/Perceiver 等 shallow fusion 轻但无 text-space 约束；Ovis 有新 visual embedding space 但参数和重新学习语义空间成本高。
3. ALIGN 的理论直觉是用 LLM 已有词表 embedding 做视觉 token 的基底，从而提升低资源学习效率和抗噪性。

### 可追问点
- convex hull 约束会不会限制视觉表达上限？
- 为什么 document understanding 比 general VQA 更适合这种设计？
- Ovis 的 visual embedding table 和 ALIGN 的 text embedding reuse 在参数量与泛化上怎么取舍？
