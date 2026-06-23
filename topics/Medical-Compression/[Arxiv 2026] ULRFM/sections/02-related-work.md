# 02 — Related Work

[← 返回 README](../README.md)

---

## 📌 Preview

本章从三个维度梳理相关工作：（1）JPEG 压缩原理及其低效来源；（2）JPEG 无损重压缩的传统方法与 CNN 方法；（3）病理基础模型的现状及在压缩任务上的空白。理解 JPEG 压缩管道（RGB → YCbCr 4:2:0 → 8×8 DCT → 量化 → Huffman 编码）是理解 ULRFM 设计动机的前提。

---

## 2.1. Overview of JPEG Compression

### 原文

The JPEG compression pipeline begins by converting an RGB image into the YCbCr color space, which comprises one luma component (Y) and two chroma components (Cb and Cr). Because chromatic information is generally less perceptually critical than luma detail, the chroma channels are commonly subsampled. The most widely used configuration is YCbCr 4:2:0, in which the Y component is preserved at full resolution, whereas the Cb and Cr components are downsampled to one quarter of their original spatial resolution (Wallace, 2002). This study focuses on this most commonly used YCbCr 4:2:0 subsampling configuration.

Following subsampling, each component is partitioned into nonoverlapping 8 x 8 blocks, and each block is transformed into the frequency domain via the DCT, yielding an 8 x 8 matrix of DCT coefficients. These coefficients are subsequently quantized using specific quantization tables. Finally, the quantized DCT coefficients are coded by lossless Huffman coding (Huffman, 2006) to produce the final JPEG bitstream.

> 💡 **机制拆解**：这是 JPEG 标准的经典流程。关键数字：8×8 DCT 块产生 64 个系数（1 个 DC + 63 个 AC），在 4:2:0 子采样下色度空间分辨率仅为亮度空间的 1/4。注意量化表的差异性——亮度和色度的量化表通常不同（色度可以更激进地量化），这也是为什么本文对 Y 和 CbCr 分别建模。

---

## 2.2. Lossless JPEG Recompression

### 原文

To address the issues of quantization redundancy and the limited efficiency of Huffman coding (Huffman, 2006) in the JPEG compression pipeline, researchers have developed several traditional JPEG lossless recompression methods, most notably Lepton (Horn et al., 2017) and JPEG XL (Alakuijala et al., 2019, 2020). Lepton replaces JPEG's Huffman coding (Huffman, 2006) with more efficient arithmetic coding (Witten et al., 1987) and employs a context adaptive probability model that is dynamically updated in real time based on the structure of the JPEG bitstream and previously coded data. This enables more accurate probability estimation and achieves an additional compression ratio of about 20%. JPEG XL substitutes the fixed 8 x 8 DCT blocks in JPEG with a variable-block DCT representation, thereby further reducing quantization redundancy, and integrates asymmetric numeral systems (ANS) (Duda, 2013; Duda et al., 2015) in place of Huffman coding. In addition, the general-purpose compressor CMIX can also perform secondary lossless recompression of JPEG images. CMIX is characterized by the use of a large number of contextual features for prediction. However, compared with Lepton and JPEG XL, it does not yield a substantial gain in compression ratio, while incurring significantly higher computational complexity and much slower running speed.

> 💡 **机制拆解**：三种传统方法的对比揭示了压缩的 tradeoff：
> - **Lepton**（~20% saving）：仅替换熵编码（Huffman → Arithmetic），不改变 DCT 表示，所以收益有限。
> - **JPEG XL**（更高 saving）：同时替换 DCT 表示（variable-block）和熵编码（ANS），从根源上减少了量化冗余，理论收益更大。
> - **CMIX**：通用压缩器，上下文建模能力强但计算开销巨大，速度极慢，不实用。
>
> ULRFM 的思路更接近 JPEG XL——同时改进表示（侧信息 hyper-prior）和熵编码（ANS + 学习得到的熵模型），但用数据驱动的方式替代手工设计。

In recent years, CNN-based JPEG lossless recompression has attracted increasing attention, leveraging the powerful modeling capacity of deep neural networks. Guo et al. (2022) were the first to propose a CNN-based entropy model directly in the DCT domain, learning the probability distribution of quantized DCT coefficients, thereby narrowing the gap between the estimated and true data distributions. Their work demonstrated that directly compressing in the DCT domain has clear advantages over operating in the pixel domain, yielding an improvement in compression rate of approximately 30%. Fan et al. (2022) introduced an end-to-end learnable lossy transform that maps DCT coefficients to a more compact representation, effectively eliminating redundancy introduced by DCT quantization. They then jointly encode the transformed representation and the residual between the lossy reconstruction and the original coefficients, achieving an average compression ratio improvement of about 21.49% over standard JPEG. In Guo et al. (2023), the authors proposed a multi-level parallel conditional modeling framework that enables parallel coding of the luma and chroma components, significantly reducing coding latency while maintaining a compression ratio gain of roughly 30%. Nonetheless, their evaluation metrics for compression rates are often compromised by their development on limited natural image datasets.

> 💡 **问题动机**：这是一个非常关键的批注——CNN 方法的性能数字（~30%）是在小规模自然图像数据集上得到的。本文在 Table 2 中清楚地展示了这些方法在病理大规模数据集上的性能大幅下降（Guo 2022 从 ~30% 降到 ~17–20%，Eff-Net 从 ~30% 降到 ~23–25%）。这说明此前方法存在严重的过拟合和泛化不足问题，也间接证明了大规模异构训练数据对鲁棒压缩模型的必要性。

Furthermore, even though Convolutional Neural Networks (CNNs) have been leveraged for modeling DCT coefficient probability distributions, they predominantly capture local relationships. As a result, their representational capacity is generally inferior to that of transformer architectures, which are designed to model long-range dependencies. Moreover, these JPEG lossless recompression methods have not been thoroughly investigated on large-scale and heterogeneous datasets, making it difficult to rigorously validate their generalization ability and effectiveness, thereby impeding their deployment in real-world applications. In this study, we present a foundation model designed for lossless recompression of pathological images. Its performance is rigorously evaluated on a large-scale dataset to demonstrate its efficacy in real diagnostic workflows.

> 💡 **Q&A 批注记录**：
>
> **Q4: CNN 不能建模长程依赖，但在 DCT 域中 8×8 块之间真的有"长程"依赖吗？**
> A: 是的。虽然每个 DCT 块只有 8×8=64 个系数，但"长程"指的是跨块的依赖——相邻块的 DC 分量高度相关（图像局部亮度一致性），同一频率的 AC 分量在不同块之间也呈现空间相关性（纹理的连续性）。这也是为什么 ULRFM 的 luma context model 要在 spatial 和 frequency 两个方向同时做 partitioning（s=4, f=9）。Transformer 的自注意力可以在一个大的 spatial-frequency context 中学习这些跨块/跨频率的统计依赖，而 CNN 的有限感受野难以覆盖。

---

## 2.3. Pathology Foundation Model

### 原文

The rapid advancement in pathology digitization has led to an exponential increase in the availability of Whole Slide Images (WSIs), thereby establishing a robust foundation for developing powerful and effective models to support computer-aided diagnosis and analysis. The current landscape of pathology foundation models is largely defined by two prominent paradigms: vision-centric models and multimodal models.

Vision-centric foundation models are predominantly built upon Vision Transformers (ViTs) combined with self-supervised learning (SSL) on WSIs. Representative works such as Virchow (Vorontsov et al., 2024) and its successors Virchow2 and Virchow2G (Zimmermann et al., 2024) employ large ViT backbones with DINOv2 (Oquab et al., 2023) training and morphology-preserving augmentations. This approach effectively captures rich spatial dependencies crucial for pathological analysis. Subsequent models further refine this paradigm along several key dimensions, including: (1) Enhancements at the SSL level, where methods like UNI (Chen et al., 2024), BROW (Wu et al., 2023), Pathorchestra (Yan et al., 2025) and Hibou (Nechaev et al., 2024) extend the DINO/DINOv2 framework with techniques such as self-distillation, masked image modeling (MAE) (He et al., 2022), and stain-specific augmentations (e.g., RandStainNA (Shen et al., 2022)) to improve robustness against staining variability and generalize across tissues and stains; (2) Contributions at the data and domain knowledge level, exemplified by RudolfV (Dippel et al., 2024), which leverages expert-curated, heterogeneous datasets for tumor microenvironment analysis and biomarker prediction; and (3) Architectural and task-level innovations, such as Prov-GigaPath (H. Xu et al., 2024)'s two-stage design with tile-level pretraining and long-range slide-level encoding, and Kaiko-ai (Aben et al., 2024)'s online patching for scalability. Furthermore, aggregation modules like OmniScreen (Y.K. Wang et al., 2024) and COBRA (Lenz et al., 2025) utilize attention-based or Mamba-based (Gu and Dao, 2024; Rahman et al., 2024) mechanisms for effective slide-level representations.

Complementing these visual models, multimodal foundation models integrate visual and textual data, ushering in an era of more profound and comprehensive pathological analysis. Key developments include PathoDuet (Hua et al., 2024) and Madeleine (Vaidya et al., 2025), which are founded on SSL frameworks specifically engineered for H&E and IHC images. mSTAR (Y. Xu et al., 2024) seamlessly integrates visual and textual information to enrich analytical depth. While HistGen (Guo et al., 2024) focuses on report generation via Multiple Instance Learning, TITAN (Ding et al., 2025) employs a three-stage strategy leveraging iBOT (Zhou et al., 2021) for visual learning and CoCa (Yu et al., 2022) for multimodal vision-language alignment. The integration of large language models with vision encoders is further exemplified by PathChat (M.Y. Lu et al., 2024) and PRISM (Shaikovski et al., 2024).

Despite these remarkable advancements across various downstream pathological analysis tasks, a critical research gap persists. None of the existing foundation models have been proposed specifically for lossless compression of pathological images. The ever-growing storage demands in digital pathology infrastructure necessitate efficient and lossless compression strategies to manage the enormous data volume. In this work, we introduce a novel foundation model explicitly designed for lossless compression of pathological images, trained on a large-scale dataset, thereby laying a crucial foundation for practical deployment in real-world clinical settings and directly addressing this unmet need for storage efficiency.

> 💡 **问题动机**：这是一段非常全面的 Pathology Foundation Model 综述。作者覆盖了两大范式（Vision-centric vs. Multimodal）、多个关键维度（SSL 方法、数据知识、架构任务创新），最终精准地指出了研究空白（research gap）：**没有基础模型是针对病理图像压缩设计的**。注意作者的定位策略——ULRFM 不是与其他病理基础模型竞争（它们在诊断/预后任务上），而是在一个全新但同样关键的任务（压缩/存储）上开辟赛道。这种"大家都忽略了 X"的定位非常有说服力。

> 💡 **Q&A 批注记录**：
>
> **Q5: 为什么病理基础模型那么多，却没有一个做压缩的？**
> A: 两个原因：（1）学术惯性——CV 病理社区主要关注诊断和预后（分类、分割、生存预测等有医学意义的任务），压缩被视为"工程问题"而非"科学问题"；（2）技术门槛——JPEG 重压缩涉及 DCT 域操作、熵编码、速率-失真理论等，需要跨领域知识。本文恰恰是在学科交叉点上找到了机会。

---

## 🔖 Related Work 批读小结

Related Work 写得很专业且全面。三条线索清晰：（1）JPEG 压缩原理 → 揭示压缩低效的根源；（2）重压缩方法的演进——从手工启发式（Lepton/JPEG XL）到 CNN 学习式（Guo/Eff-Net），再到本文 Transformer 方法；（3）病理基础模型的生态全景 → 精准定位研究空白。一个亮点是作者坦诚地指出 CNN 方法在大规模数据上的性能退化，这既是本文的实验发现也是核心论据。
