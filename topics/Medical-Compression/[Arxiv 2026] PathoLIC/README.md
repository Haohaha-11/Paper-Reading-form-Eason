# PathoLIC: A Content-Aware Variable-Rate Framework for Pathology Learned Image Compression

## 论文信息

| 项目 | 内容 |
|------|------|
| **标题** | A content-aware variable-rate framework for pathology learned image compression (PathoLIC) |
| **作者** | Weiqi Li, Yonghao Li, Haoyuan Chen, Long Yang, Lin Wu, Zhenhui Li, Jing Ke\*, Dinggang Shen\* |
| **单位** | ShanghaiTech University（生物医学工程学院）, Kunming Medical University（云南省肿瘤医院）, Shanghai Jiao Tong University, United Imaging Intelligence |
| **发表** | Medical Image Analysis 111 (2026) 104018 |
| **DOI** | https://doi.org/10.1016/j.media.2026.104018 |
| **接收** | Received 4 Nov 2025, Revised 12 Feb 2026, Accepted 2 Mar 2026, Available online 6 Mar 2026 |
| **代码** | https://github.com/wqli498/PathoLIC |
| **关键词** | Learned image compression, Whole slide image (WSI), Variable-rate compression |

## 一句话总结

> 提出 PathoLIC，首个面向全切片图像（WSI）的**内容感知可变码率**学习压缩框架：根据 patch 的诊断重要性自动分配差异化压缩率（肿瘤区域高保真、基质/背景高压缩），并利用 Attention 机制消除跨 patch 冗余，实现超过 Aperio SVS 格式 8 倍的压缩比，同时在多癌种下游任务中保持与原图相当的诊断性能。

## 核心贡献

1. **内容感知可变码率 WSI 压缩**：首次将内容感知策略引入 WSI 压缩，利用预训练病理基础模型（CHIEF/TITAN）自动估计每个 patch 的诊断重要性分数，指导差异化压缩——高诊断价值区域（肿瘤、炎性区域）保留更多细节，低诊断价值区域（脂肪、基质、背景）更激进压缩。

2. **跨 Patch 冗余消除**：采用 Transformer-CNN 混合架构（Swin Transformer + Residual Blocks with Stride），通过注意力机制捕获相邻及相似 patch 之间的空间相关性，压缩共享特征以减少冗余。

3. **区域级 WSI 比特流格式**：设计了按 4x4 patch 区域分组的二进制存储格式，支持两种解码模式：(a) 全图解码为 SVS 金字塔以兼容现有读片软件；(b) 按坐标或诊断分数选择性解码特定区域，大幅降低 I/O 延迟。

4. **连续可变码率训练策略**：训练时从 [0,1] 均匀分布采样模拟内容分数，映射到 λ∈[0.0025, 0.04] 的连续压缩范围，使模型泛化到任意压缩水平；推理时切换为真实内容分数，无需二次训练。

## 📖 批读导航

| Section | 内容 | 说明 |
|---------|------|------|
| [00 Abstract](sections/00-abstract.md) | 摘要 | 全文概述，问题、方法、结果 |
| [01 Introduction](sections/01-introduction.md) | 引言 | WSI 存储挑战 → 现有方法局限 → PathoLIC 动机与贡献 |
| [02 Related Work](sections/02-related-work.md) | 相关工作 | 数字病理、传统压缩、学习压缩三条线 |
| [03 Methodology](sections/03-methodology.md) | 方法 | 内容评分 → 压缩架构 → 解压缩 → 训练策略 → 比特流格式 |
| [04 Experiments](sections/04-experiments.md) | 实验设置与结果 | 多层级下游任务评估 + 消融实验 |
| [05 Conclusion](sections/05-conclusion.md) | 结论与局限 | 总结贡献，指出现有局限与未来方向 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 单张 WSI 分辨率（40x） | 最大 80,000 x 80,000 像素 |
| 单张 WSI 原始大小 | 1-4 GB |
| 训练 patch 数量 | 73,730（1024x1024, 40x） |
| 输入区域大小 | L x L = 4x4 patches = 16 patches/region |
| 压缩比 vs. Aperio SVS | >8x |
| PathoLIC(TITAN) 平均压缩比 | 8.99x（TCGA-NSCLC 全部） |
| 模型参数量 | 879 MB |
| 编码时间/patc | 0.293 s |
| 解码时间/patc | 0.310 s |
| λ 范围 | [0.0025, 0.04]（指数映射） |
| PSNR @ 0.28 BPP | 40.6 dB |
| MS-SSIM @ 0.28 BPP | 0.990 |
| 潜在维度（y / z） | N=320 / M=192 |
| 训练迭代 | 80,000 steps, lr=4e-5 |
| 下游任务种类 | 5 类（WSI 亚型、生存预测、patch 分类、ROI 检索、细胞核分割） |
| 下游评估数据集 | 10+ 个（TCGA-BRCA/NSCLC/RCC, BACH, In-house, NCT-CRC-HE-100K, PanNuke, MNS） |

## 数据流：输入 → 中间表示 → 输出

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                               │
│                                                                     │
│  WSI → 256×256 patches → Foreground Filter (CLAM pipeline)         │
│      → Group into 1024×1024 regions (16 patches each)              │
│      → Content score Q ~ U[0,1] (uniform sampling)                 │
│      → λ(Q) = 0.0025 × (0.04/0.0025)^Q  [exponential mapping]     │
│                                                                     │
│  Region x ──→ gₐ(Encoder: RBS + TCM) ──→ y                         │
│           ──→ QCM_y(y, Q) ──→ y_control                             │
│           ──→ hₐ(Hyper-encoder) ──→ QCM_z(z, Q) ──→ ẑ              │
│                                                                     │
│  Loss: L = E[R(x;Q) + λ(Q)·D(x, x̂(Q))]                             │
│         R = -log p(ŷ|ẑ) - log p(ẑ)   [entropy estimation]         │
│         D = MSE(x, x̂)                 [distortion]                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       INFERENCE PHASE                               │
│                                                                     │
│  1. Content Score Generation:                                       │
│     WSI → Patches → Foundation Model (CHIEF/TITAN) → Attention     │
│     Map → MLP + Tanh-Sigmoid gate → Content Score Q per patch       │
│                                                                     │
│  2. Compression (per 4×4 region):                                   │
│     Region x + Q ──→ gₐ ──→ y ──→ QCM_y(y,Q) ──→ y_control         │
│          ──→ hₐ ──→ QCM_z(z,Q) ──→ Quantize ──→ ẑ                  │
│          ──→ AE(y_control, ẑ, Q) ──→ binary bitstream               │
│                                                                     │
│  3. Decompression:                                                  │
│     Bitstream ──→ AD ──→ ẑ ──→ hₛ(ẑ) ──→ (μ,σ)                    │
│              ──→ QCM_μσ(μ,σ,Q) ──→ (μ_control, σ_control)          │
│              ──→ CCM(Quantize(y_control); μ_control, σ_control)    │
│              ──→ QCM_ŷ(ŷ,Q) ──→ gₛ(Decoder: RBS + TCM) ──→ x̂      │
│                                                                     │
│  4. Output: Reconstructed WSI or SVS pyramid                        │
└─────────────────────────────────────────────────────────────────────┘
```

> 💡 **数据流批注**：关键设计在于训练/推理的 content score 来源不对称——训练用随机采样（迫使模型学习连续压缩映射），推理用真实诊断分数（无需微调）。QCM 作为内容分数的"注入点"分布在编码器的 y、z、μ/σ、ŷ 四个位置，形成多级调制。

## 优缺点与还能做什么

### 优点

| 方面 | 说明 |
|------|------|
| **压缩比显著** | >8x vs SVS，远超 JPEG baseline (5.04x) |
| **任务保持好** | 5 类下游任务性能与原图几乎持平，部分场景（compressed→original training）甚至略有提升 |
| **内容感知** | 自动识别诊断重要区域，差异化分配码率，不是"一刀切" |
| **两种推理模式** | Fixed high-fidelity（patch/cell 级任务）vs Content-aware variable-rate（WSI 级任务），灵活适配 |
| **区域级随机访问** | 支持按坐标/分数选择性解码，在带宽受限环境下大幅降低 I/O |
| **Foundation Model 鲁棒** | CHIEF 和 TITAN 两种基础模型引导均稳定，TITAN 压缩比更高 |

### 缺点

| 方面 | 说明 |
|------|------|
| **无 GUI 软件** | 缺少图形化压缩/解压/可视化平台 |
| **不支持标注集成** | 无法在压缩文件中嵌入或修改病理学家的标注（肿瘤边界、分型标签） |
| **模型较大** | 879 MB，比 QmapCompression (316MB) 和 I2C (576MB) 更大 |
| **编解码速度** | 0.29s/0.31s per patch，比 JPEG (0.035s/0.004s) 慢一个数量级 |
| **依赖基础模型** | 内容评分阶段依赖预训练病理基础模型，引入额外计算开销 |

### 还能做什么（未来方向）

1. **开发端到端 GUI 平台**：集成压缩、解压、可视化、标注管理于一体
2. **标注压缩与集成**：在比特流中支持病理标注（ROI、分级标签）的存储与修改
3. **模型轻量化**：通过知识蒸馏或模型剪枝减小 879MB 模型体积，降低部署门槛
4. **实时/近实时编码优化**：减少编码时间以接近 JPEG 的速度
5. **多模态扩展**：同时压缩 WSI 图像与基因组学/蛋白质组学等多模态数据
6. **渐进式解码**：支持先解压低分辨率概览图，再按需加载高分辨率细节区域
7. **跨机构联邦训练**：在分布式病理数据上训练压缩模型，保护数据隐私

## 阅读 Q&A 记录

### Q1: 为什么不直接用 JPEG2000 的 ROI 编码而非要重新设计一个框架？
> 💡 **A**: JPEG2000 的 ROI 编码依赖人工指定的感兴趣区域，而 PathoLIC 的核心创新在于**自动化**——利用病理基础模型（CHIEF/TITAN）自动识别诊断重要区域，无需人工标注。同时，JPEG2000 不能消除跨 patch 冗余（独立压缩每个 patch），而 PathoLIC 的 Attention 机制将 16 个相关 patch 联合压缩，进一步减少冗余。

### Q2: 训练时 Q~U[0,1] 随机采样，推理时用真实 content score，domain gap 怎么解决？
> 💡 **A**: 这是论文的一个巧妙设计。训练时让模型见过整个 [0,1] 区间的 λ 取值，迫使它学会一个从 content score 到压缩强度的连续映射，类似于 variable-rate 压缩中的"增益单元"思路。U[0,1] 覆盖了所有可能的压缩水平，真实 content score 只是这个连续区间的特定采样，因此不存在 domain gap。这类似于在训练期间使用均匀的 quality map，推理时换为真实的 quality map。

### Q3: TITAN 和 CHIEF 的 content score 有什么差异？为什么 TITAN 压缩比更高？
> 💡 **A**: 二者产生的 attention heatmap 语义一致（都正确高亮肿瘤区域），但 TITAN 的 attention map 更稀疏（sparser），即背景区域的 attention 值更低。PathoLIC 的压缩框架具有"自适应校准"能力：稀疏的 attention map 意味着更多 patch 获得低 content score，从而被更激进压缩，整体文件大小自动减小，但不损失下游性能。

### Q4: QCM 的 residual-style scaling (α+1) 为什么重要？
> 💡 **A**: 消融实验（Fig. 11）表明，去掉残差连接直接使用 FM = α·F + β 会导致 RD 性能显著下降，尤其在低码率区域。原因：(α+1) 保证了 identity 路径的存在——当 QCM 不需要大幅调制时，α→0 即可退化为恒等映射，保留了原始特征的稳定性。去掉残差连接后，网络必须学习从零开始重建特征分布，训练不稳定。

### Q5: Compressed→Original 训练为何有时性能比 Original→Original 更好？
> 💡 **A**: 这是一个"压缩作为数据增强"的有趣现象。在 TCGA-BRCA 的子类型分类中，PathoLIC(compressed) → 训练 + Original → 测试，BACC 从 0.907 提升到 0.937。这是因为压缩过程类似正则化——它滤除了与诊断无关的高频噪声和纹理细节，迫使下游模型关注真正有判别力的形态学特征。

### Q6: WSI 级和 patch/cell 级任务为什么要用不同的推理模式？
> 💡 **A**: WSI 级任务（亚型分类、生存预测）的输入是整个 WSI 的上千个 patch，单个 patch 的细节损失可以被全局上下文补偿，因此适合 content-aware variable-rate 以获得最大压缩。而 patch/cell 级任务（分类、分割）的输入本身就是单个或少数 patch，任何细节损失都直接影响预测，因此必须用 fixed high-fidelity (Q=1)。

### Q7: 这个方法能否扩展到 3D 病理（如多层切片重建）？
> 💡 **A**: 论文未涉及，但框架具有扩展潜力。Attention 机制可以自然地扩展到空间+深度维度，将相邻切片中的相似组织区域联合压缩，有望实现更高的压缩比。但需要解决 3D 配准和跨切片 content score 对齐等问题。

## 📊 Citation Landscape

| 类别 | 代表性工作 | 与本工作的关系 |
|------|-----------|---------------|
| **LIC 基础框架** | Ballé et al. 2018 (Hyperprior), Minnen et al. 2018 (Joint AR+HP), Cheng et al. 2020 (GMM+Attention) | PathoLIC 的熵模型和基础架构 backbone |
| **LIC Transformer** | Liu et al. 2023 (TCM), Zhu et al. 2022 | 采用 Swin Transformer + CNN 混合编码器 |
| **Variable-Rate LIC** | Song et al. 2021 (QmapCompression), Cai et al. 2024 (I2C) | 可变率思路来源，但 PathoLIC 的码率由内容自动决定而非外部参数 |
| **病理基础模型** | Wang et al. 2024 (CHIEF), Ding et al. 2025 (TITAN), Xu et al. 2024 (Prov-GigaPath) | CHIEF/TITAN 用于 content score 生成，Prov-GigaPath 用于下游评估 |
| **WSI 分析框架** | Lu et al. 2021 (CLAM), Campanella et al. 2019 | CLAM 提供前景检测 pipeline，MIL 框架提供下游评估范式 |
| **下游任务基准** | Chen et al. 2024 (UNI), Graham et al. 2019 (Hover-Net), Isensee et al. 2021 (nnU-Net) | ROI retrieval / 核分割评估工具 |
| **传统压缩** | JPEG (Wallace 2002), JPEG2000 (Taubman et al. 2002) | Baseline 对比方法 |

---

*Last updated: 2026-06-23*
