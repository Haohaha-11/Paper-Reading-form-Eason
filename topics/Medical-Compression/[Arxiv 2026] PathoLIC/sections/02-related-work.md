# 02 Related Work

[← 返回 README](../README.md)

## 📌 Preview

> 三条技术脉络的梳理：(1) 数字病理与计算任务——从 cell/patch/WSI 三级任务建立评估基准；(2) 传统图像压缩——JPEG/JPEG2000 的机理与在 WSI 中的局限；(3) 学习图像压缩——从 Hyperprior 到 Transformer LIC 再到 Variable-Rate LIC，逐层递进到 PathoLIC 的差异化定位。

---

## 原文

### 2.1. Digital pathology and computational tasks

WSI has reshaped computational pathology by facilitating automated analysis of histological specimens for applications such as cancer diagnosis, tumor grading, and cellular characterization (Madabhushi and Lee, 2016; Litjens et al., 2017; Komura and Ishikawa, 2018). These diagnostic tasks operate across multiple spatial resolutions, ranging from individual patches to entire WSIs. At the cellular level, fine-grained analyses such as nucleus segmentation and phenotype classification are essential for quantifying the tumor microenvironment (Kumar et al., 2017; Graham et al., 2019; Gamper et al., 2019). Patch-level methods focus on performing mitosis detection, histological grading, and tissue classification (Aresta et al., 2019; Hou et al., 2016) with localized tissue regions as inputs. At the WSI level, models leverage global contextual information to perform cancer classification and tumor subtyping (Campanella et al., 2019; Lu et al., 2021) without requiring fine-grained manual annotations.

> 💡 **机制拆解**：这段看似平淡的分类综述，实际上铺垫了 PathoLIC 评估体系的三级层次——Cell-level（核分割）→ Patch-level（组织分类、ROI 检索）→ WSI-level（亚型分型、生存预测）。每一级对压缩的敏感度不同，因此在实验部分采用了不同的推理策略（cell/patch 用 fixed high-fidelity，WSI 用 content-aware variable-rate）。注意这里引用的数据集（PanNuke, BACH, TCGA）在后文实验中被直接使用。

### 2.2. Traditional image compression

Traditional image compression algorithms such as JPEG (Wallace, 2002) and JPEG2000 (Taubman et al., 2002) employ hand-crafted transform coding pipelines based on the discrete cosine transform (DCT) or wavelet transforms, followed by quantization and entropy coding. In digital pathology, WSIs are typically stored using TIFF-based container formats (e.g., SVS, NDPI) that organize large images as pyramids. Lossy JPEG compression, which divides images into blocks and applies the DCT followed by quantization and entropy coding, is commonly used to balance file size and visual quality (Farahani et al., 2015). However, WSI files still occupy substantial storage space and can be further compressed to reduce storage and transmission costs.

> 💡 **问题动机**：JPEG/JPEG2000 是"手工设计的变换 + 量化 + 熵编码"范式，本质是信号处理而非数据驱动。作者点出了一个关键：即使使用了 JPEG 压缩，SVS 文件仍然很大（实验数据显示约 5x 压缩比）——这意味着传统方法在 WSI 场景下已经逼近了其编码效率的天花板。

### 2.3. Learned image compression

Learned image compression (LIC) frameworks employ end-to-end trainable autoencoder architectures, where the encoder, decoder, and entropy model are jointly optimized to balance compression rate and image fidelity (Balle et al., 2017; Toderici et al., 2017). The introduction of hyperprior models (Balle et al., 2018), typically implemented with convolutional neural networks (CNNs), further enhances compression efficiency by enabling more accurate entropy estimation. Recently, transformer-based architectures have been investigated to further improve compression performance (Zhu et al., 2022; Liu et al., 2023).

> 💡 **机制拆解**：LIC 的技术演进线：Autoencoder (Balle 2017) → Hyperprior (Balle 2018) → Autoregressive + Hyperprior (Minnen 2018) → GMM + Attention (Cheng 2020) → Transformer-CNN (Liu 2023, Zhu 2022)。PathoLIC 的 backbone 继承自 Liu et al. 2023 的 TCM 架构（Swin Transformer + CNN 混合），这是一条路径依赖——TCM 在自然图像 LIC 中已经是 SOTA，PathoLIC 加上 WSI-specific 的 content-aware 机制。

However, early LIC methods are trained for a fixed compression level, which limits their flexibility. To address this limitation, variable-rate frameworks have been proposed to adapt bitrates based on external control signals. For instance, QmapCompression (Song et al., 2021) and I2C (Cai et al., 2024) employ Spatially Feature Transforms (SFT) or Invertible Activation Transformations (IAT) using convolutional networks to perform pixel-wise modulation based on spatial quality maps.

> 💡 **公式批读**：QmapCompression 的 SFT 本质是 Fi = (α(q)+1) * Fi + β(q)，其中 α(q) 和 β(q) 是 quality map q 的函数。注意这里 PathoLIC 的 QCM（Eq. 3-4）采用了几乎一样的仿射变换形式 [α, β] = MLP(Q), F_mod = (α+1)*F + β。这不是巧合——QCM 可以看作 SFT 的"content score 条件化"版本，核心差异在于 q 的来源：QmapCompression 的 q 是人为指定的 quality map，PathoLIC 的 Q 是自动计算的 diagnostic content score。

Unlike existing methods, our framework accounts for the unique characteristics of whole slide images (WSIs) to improve compression efficiency. First, it can automatically assign patch-level compression rates according to the diagnostic importance of each patch. Second, the model leverages a Transformer to capture spatial correlations among neighboring or similar patches, thereby further reducing redundancy.

> 💡 **机制拆解**：这是 Related Work 章节最关键的差异化声明。两层区分：(1) vs. 传统 LIC——增加 content-aware 和跨 patch 关联；(2) vs. 现有 variable-rate LIC（QmapCompression, I2C）——自动生成而非手动指定质量图 + Transformer 跨 patch 建模而非逐 patch 独立处理。注意这里没有提到的是，I2C 的 inverse neural network 架构在解码时速度极慢（21.8s/patch vs PathoLIC 的 0.31s/patch），这也是 PathoLIC 的工程优势。

---

## 🔖 相关工作批注总结

- **三条脉络交织**：数字病理（定义了什么任务是重要的）→ 传统压缩（定义了 baseline 和方法局限）→ LIC（定义了技术路径和改进方向）
- **PathoLIC 的独特定位**：在 LIC 和 Variable-Rate LIC 的交叉点上，加入了 WSI 领域的 domain knowledge（patch 间相关性和内容诊断重要性）
- **一个技术细节**：QCM 的仿射变换形式与 QmapCompression 的 SFT 几乎相同，说明 PathoLIC 的技术模块不是凭空发明，而是在现有 LIC 积木上的领域适配
- **Related Work 写得比较标准但稍显简短**，缺少对 Variable-Rate LIC 更深入的技术对比（如 quality map 的生成方式、调制机制的区别），这可能是因为论文篇幅受限
