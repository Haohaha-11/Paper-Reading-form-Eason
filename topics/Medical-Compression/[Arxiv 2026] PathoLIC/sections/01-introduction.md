# 01 Introduction

[← 返回 README](../README.md)

## 📌 Preview

> 从数字病理的 WSI 数据爆炸讲起，逐步收敛到 PathoLIC 的创新点——传统 JPEG/JPEG2000 在 WSI 上的局限 → LIC 方法虽好但未考虑 WSI 特性（跨 patch 冗余 + 内容差异） → 现有可变码率方法不能自动分配 patch 级压缩率 → PathoLIC 的 content-aware + attention 双核心解决方案。

---

## 原文

Digital pathology is an emerging field in modern medicine that converts histology slides into high-resolution whole slide images (WSIs) for computational analysis (Lu et al., 2021). With the rapid development of digital scanning technologies, clinical centers are now producing vast collections of WSIs for diagnostics, research, and long-term archiving. However, the enormous volume of digitized tissue slides presents major challenges in data storage and management. A single WSI at 40x magnification can reach resolutions of up to 80,000 x 80,000 pixels, with corresponding file sizes ranging from 1 to 4 GB (Van der Laak et al., 2021). For institutions that archive tens of thousands of slides, total data storage requirements can scale to multiple petabytes, posing significant challenges in both data storage and transfer (Association, 2019).

> 💡 **问题动机**：这组数字很有冲击力——单张 40x WSI 高达 80000x80000 像素（64 亿像素！），1-4GB 一张，积累到上万张就是 PB 级。这不是学术界的"toy problem"，而是医院真实的存储痛点。注意这里引用的是 DPA (Digital Pathology Association) 2019 年白皮书，说明业界早已关注这个问题。

The Aperio ScanScope Virtual Slide (SVS) format is the most widely adopted standard for WSI storage. It represents WSIs as multi-resolution pyramids, compressed with either JPEG or JPEG2000. In particular, JPEG relies on block-wise discrete cosine transforms (DCT) with quantization, while JPEG2000 employs wavelet transforms that support both lossy and lossless compression modes. JPEG enables fast compression but often introduces visible artifacts (Wallace, 2002), whereas JPEG2000 achieves higher fidelity at the cost of increased computational complexity (Taubman et al., 2002). Consequently, SVS files remain large and could benefit from more advanced compression techniques.

> 💡 **机制拆解**：SVS 格式的本质是"金字塔 + JPEG/JPEG2000"——多分辨率金字塔是为了交互式浏览，JPEG/JPEG2000 是传统的块内压缩。关键洞察：JPEG 快但有块效应，JPEG2000 好但慢，且两者都没有利用 WSI 的结构化先验（patch 之间的空间关系）。这是 PathoLIC 的突破口。

Given these limitations, there has been a growing research interest in learned image compression (LIC) (Balle et al., 2018; Minnen et al., 2018; Cheng et al., 2020), which involves training artificial neural networks on large-scale image datasets. Specifically, LIC systems comprise two main components: an autoencoder and an entropy model (Balle et al., 2018). The autoencoder compresses the input image into a compact latent representation and then reconstructs it from this representation, while the entropy model estimates the probability distribution of the latent codes to facilitate efficient compression. Values with higher probabilities are encoded with fewer bits. The framework is optimized to achieve a balance between compression rate and image fidelity. Extensive evaluations on natural image benchmarks have shown that LIC methods consistently outperform conventional approaches in both perceptual quality and rate-distortion performance (Cheng et al., 2020).

> 💡 **公式批读**：LIC 的 RD 优化本质是 L = R + λ·D，其中 R 是码率（熵模型的负对数似然），D 是失真（MSE/MS-SSIM），λ 是拉格朗日乘子控制 trade-off。这是所有 LIC 方法共同的数学框架，PathoLIC 的创新在于让 λ 变成 patch-adaptive 的——λ(Q)，由 content score Q 决定。

Despite recent progress, existing image compression approaches remain suboptimal for WSI compression as they overlook key properties of WSIs. Given their enormous resolution, WSIs are generally divided into non-overlapping patches for data preprocessing and downstream analysis (Xu et al., 2024; Wang et al., 2024). However, compressing patches independently using existing LIC methods ignores spatial redundancy across neighboring patches with similar morphology, resulting in insufficient compression.

More importantly, most existing LIC models are optimized for a fixed rate-distortion trade-off (Minnen et al., 2018; Cheng et al., 2020). Therefore, they yield similar compression levels across diverse inputs, regardless of their content variability. Nonetheless, diagnostic importance varies across different regions within WSIs. For example, tumor regions are of higher diagnostic importance than normal or fatty tissue (Angell et al., 2013). Therefore, diagnostically relevant regions should be compressed at higher fidelity, while less informative areas can be more aggressively compressed to improve overall efficiency. This characteristic of WSIs calls for content-aware compression frameworks with variable compression ratios. While recent variable-rate LIC models (Song et al., 2021; Cai et al., 2024) support flexible compression levels via introducing a global rate-distortion hyperparameter, they cannot automatically assign compression levels across WSI patches based on their contents.

> 💡 **问题动机**：这里是整篇论文最核心的"gap"论述。三层递进：Layer 1——现有 LIC 独立压缩每个 patch，浪费了跨 patch 的共享信息；Layer 2——现有 LIC 是固定码率（一个 λ 打天下），而 WSI 不同区域诊断价值不同；Layer 3——即使 QmapCompression 和 I2C 支持可变码率，也需要"外部手动指定"质量图，不能自动根据内容决定压缩级别。PathoLIC 同时解决这三层问题。

To address these challenges, we propose PathoLIC, a content-aware, variable-rate compression framework tailored for WSIs. As illustrated in Fig. 1, our framework simultaneously processes 16 patches as input, considering (1) the diagnostic relevance of each patch and (2) the spatial correlations among neighboring or similar patches. PathoLIC produces a highly compressed binary representation, significantly reducing storage requirements while preserving fine-grained histological details. The key contributions of PathoLIC can be summarized as follows:

1. We present a learning-based variable-rate framework for WSI-level image compression, namely PathoLIC. It computes patch-level content scores to guide adaptive compression, preserving critical content at higher fidelity while allocating fewer bits to less informative areas.
2. We leverage attention mechanisms to model correlations among neighboring or similar patches, thereby reducing redundancy through the compression of shared features.
3. We propose a region-wise WSI bitstream format to combine latent bitstreams with content-score metadata, for enabling efficient coordinate- and score-based decoding.

To the best of our knowledge, PathoLIC is the first framework that leverages content-aware strategies in whole slide image compression. PathoLIC is validated using both compression metrics and clinically relevant downstream tasks. Specifically, we benchmark its rate-distortion performance against conventional compression approaches and state-of-the-art LIC methods. To assess its practical utility, we further test its robustness across a diverse set of diagnostic tasks using public and in-house datasets covering multiple cancer types.

Extensive experiments demonstrate that PathoLIC achieves an average compression ratio 8x higher than the SVS format, while maintaining comparable performance on various downstream tasks. These results highlight the potential of PathoLIC as a practical and scalable solution for efficient WSI data management in digital pathology.

---

![Figure 1](../images/page1_img1.jpeg)

*Figure 1. The workflow of PathoLIC. Histology slides are digitized into WSIs in SVS format, which are partitioned into non-overlapping regions, each containing 16 patches. By exploiting patch-level correlations, the network reduces redundancy and yields a binary file significantly smaller than the original SVS. During decompression, PathoLIC restores the WSI with high fidelity, ensuring that fine-grained histological details are preserved.*

> 💡 **Figure 1 批读**：这张图是 PathoLIC 的"门面"，展示了完整的压缩-解压缩流水线。注意几个关键设计：(1) 输入不是单张 patch 而是 16 个 patch 组成的 region（4x4 排列），这是为了利用 patch 间的相关性；(2) 输出是二进制文件（不是中间特征），暗示了完整的编解码 pipeline 包含了熵编码；(3) "Content-Aware Variable-Rate"的概念体现在流程中的"Content Score → QCM Modulation"环节。注意图上的文字是红蓝双色标注，红色可能标识高 content score 的肿瘤区域、蓝色标识低 score 的基质/背景区域。

---

## 🔖 引言批注总结

- **从宏观到微观**：数字病理 → WSI 体积爆炸 → SVS/JPEG 局限 → LIC 优势 → LIC 在 WSI 上的两个盲区 → PathoLIC 三项创新
- **核心矛盾**：WSI 的"结构性"（patch 间高度相关 + 内容诊断价值不均）与现有压缩方法的"独立性假设"（独立压缩 + 均匀码率）之间的矛盾
- **创新定位**：PathoLIC = 首个 content-aware WSI 压缩框架，填补了 LIC 在病理领域的空白
- **一个观察**：Introduction 的风格非常"工程驱动"——问题定义清楚 → 现有方法不足（有具体引用和对比） → 方案清晰 → 贡献点编号列出。这种写法在 MedIA 级别的期刊中是高分范本
