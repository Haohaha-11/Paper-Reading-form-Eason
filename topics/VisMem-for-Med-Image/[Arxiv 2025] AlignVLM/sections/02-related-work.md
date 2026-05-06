[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
本节把已有 VLM 对齐方法分成 deep fusion、shallow fusion、visual embedding 和 encoder-free 几类。读法重点是：哪些方法改 LLM 主体，哪些只改输入层 connector，以及 ALIGN 如何复用 text embedding matrix 避免新语义空间。

---

> 💡 **Q&A 批注记录**:
> - Q: ALIGN 属于 shallow fusion 吗？
> - A: 是。它不在 LLM 每层加 cross-attention，也不改变自回归解码结构；它只把视觉特征变成 LLM input embeddings。新意在这个 input-level connector 有 text embedding convex-hull 约束。

# 2.1 Vision-Language Models

Over the past few years, Vision-Language Models (VLMs) have achieved remarkable progress, largely due to advances in Large Language Models (LLMs). Initially demonstrating breakthroughs in text understanding and generation [Brown et al., 2020, Raffel et al., 2023, Achiam et al., 2023, Grattafiori et al., 2024, Qwen et al., 2025, Team, 2024], LLMs are now increasingly used to effectively interpret visual inputs [Liu et al., 2023b, Li et al., 2024, Wang et al., 2024, Chen et al., 2024b, Dai et al., 2024, Drouin et al., 2024, Rodriguez et al., 2022]. This progress has enabled real-world applications across diverse domains, particularly in multimodal document understanding for tasks like form reading [Svetlichnaya, 2020], document question answering [Mathew et al., 2021b], and chart question answering [Masry et al., 2022]. VLMs commonly adopt a three-component architecture: a pretrained vision encoder [Zhai et al., 2023, Radford et al., 2021], a LLM, and a connector module. A key challenge for VLMs is effectively aligning visual features with the LLM’s semantic space to enable accurate and meaningful multimodal interpretation.

> 💡 **谱系定位**: 这段把 VLM 拆成三个可替换部件。AlignVLM 固定强 vision encoder/LLM，把论文贡献压到 connector；这让实验中的 MLP/Ovis/Perceiver 对照更有针对性。

# 2.2 Vision-Language Alignment for Multimodal Models

Existing vision-language alignment approaches can be classified into deep fusion and shallow fusion. Deep fusion methods integrate visual and textual features by modifying the LLM’s architecture, adding cross-attention and feed-forward layers. For example, Flamingo [Alayrac et al., 2022] employs the Perceiver Resampler, which uses fixed latent embeddings to attend to vision features and fuses them into the LLM via gated cross-attention layers. Similarly, NVLM [Dai et al., 2024] adopts cross-gated attention while replacing the Perceiver Resampler with a simpler MLP. CogVLM [Wang et al., 2023b] extends this approach by incorporating new feed-forward (FFN) and QKV layers for the vision modality within every layer of the LLM. While these methods improve cross-modal alignment, they significantly increase parameter counts and computational overhead, making them less efficient.

> 💡 **deep fusion 读法**: deep fusion 的优势是多层 cross-modal interaction，缺点是要修改 LLM 内部。对于商业文档场景，这意味着更高训练/部署成本，也更难快速替换不同 Llama/Qwen backbone。

On the other hand, shallow fusion methods are more computationally efficient, mapping visual features into the LLM’s embedding space without altering its architecture. These methods can be categorized into three main types: (1) MLP-based mapping, such as LLaVA [Liu et al., 2023b] and PaliGemma [Beyer et al., 2024], which use multilayer perceptrons (MLP) to project visual features but often produce misaligned or noisy features due to a lack of constraints and inductive bias [Rodriguez et al., 2024b]; (2) cross-attention mechanisms, BLIP-2 [Li et al., 2023b] uses Q-Former, which utilizes a fixed set of latent embeddings to cross-attend to visual features, but that may still produce noisy or OOD visual features; (3) convolution-based mechanisms, such as HoneyBee [Cha et al., 2024] and H-Reducer [Hu et al., 2024], which leverage convolutional or ResNet [He et al., 2015] layers to preserve spatial locality while reducing dimensionality; and (4) visual embeddings, such as those introduced by Ovis [Lu et al., 2024], which use embeddings indexed by the vision encoder’s outputs to produce the visual inputs. While this regularizes feature mapping, it adds substantial parameter overhead and creates a new vision embedding space, risking misalignment with the LLM’s text embedding space. Encoder-free VLMs, like Fuyu-8B 1 and EVE [Diao et al., 2024], eliminate dedicated vision encoders but show degraded performance [Beyer et al., 2024].

> 💡 **shallow fusion 细分**: MLP 是无约束连续投影；Q-Former/Perceiver 用 latent queries 汇聚视觉特征；HoneyBee/H-Reducer 加入局部性和降维；Ovis 新建 visual embeddings。它们共同问题是：即使输出维度对了，也不保证输出分布贴近 LLM text embeddings。

In contrast, ALIGNVLM maps visual features from the vision encoder into probability distributions over the LLM’s text embeddings, using them to compute a convex combination. By leveraging the linguistic priors encoded in the LLM’s vocabulary, ALIGNVLM ensures that visual features remain within the convex hull of the text embedding. This design mitigates noisy or out-of-distribution projections and achieves stronger multimodal alignment, particularly in tasks that require joint modalities representation like multimodal document understanding and in low-resource settings.

> 💡 **与已有工作的差别**: ALIGN 的输出不是“学出来的一组新 visual tokens”，而是 LLM 词表 embedding 的软组合。这个设计把 connector 学习难度从“发明一套 LLM 能懂的视觉 embedding 空间”降为“学会在已有词表语义基底上配权重”。

---

## 🔖 Section 总结

### 关键数字速查
| 类别 | 代表方法 | ALIGN 的相对位置 |
|------|----------|------------------|
| deep fusion | Flamingo, NVLM, CogVLM | 不改 LLM 层结构，成本更低 |
| shallow MLP/attention/conv | LLaVA, BLIP-2, HoneyBee, H-Reducer | 保持浅融合，但增加 text-space 约束 |
| visual embedding table | Ovis | 不新建 visual embedding space，复用 text embeddings |

### 核心洞察
1. 相关工作的分界线不是“是否多模态”，而是视觉信息在哪里进入 LLM：层内交互还是输入 embedding 层。
2. ALIGN 保持 shallow fusion 的部署友好性，同时给输出 latent 加上 LLM text embedding prior。
3. 对 VisMem/Med Image 方向，可复用的点是：外部视觉记忆或医疗图像证据最终也需要一个稳定的 LLM latent 入口。

### 可追问点
- 对非文档视觉任务，text embedding convex hull 是否仍是合理约束？
- 是否可以把 Ovis 的离散 visual embedding 和 ALIGN 的 text embedding 加权结合？
- connector 的公平对照需要控制哪些变量：vision encoder、LLM、训练数据、指令格式还是图像分辨率？
