[← 返回 README](../README.md)

# 01 - Introduction

## 预览

Introduction 的逻辑是：WSI 存储成本已经成为 CPath 扩展瓶颈；既有 NIC/SR 能压缩但多是 uniform 或 task-specific；AdaSlide 的切入点是让压缩决策随 patch 信息量变化。

# Introduction

Recently, the transformation from glass slide pathology to the digital pathology system (DPS) has led to the emergence of machine learning-integrated DPS, known as computational pathology (CPath) [? ? ? ? ]. CPath has demonstrated strong performance in various pathological tasks, including cancer diagnosis [? ? ? ? ? ? ? ], survival prediction [? ], lymph node invasion detection [? ? ], and cell segmentation [? ? ]. CPath performance has significantly improved through the use of large-scale digitized slides, with robust foundation models recently proposed [? ? ? ? ? ? ? ]. However, unlike natural images, whole slide images (WSIs) are considerably large (3 or 4 GB), resulting in significant storage burdens for data archiving. For example, Link¨oping Hospital collects digital slides totaling 130 TB every six months [? ]. Due to storage limitations, slides older than six months are deleted, which is a temporary, impractical, and invasive solution. Simply expanding storage is also impractical, as it requires physical space and increases maintenance costs. Additionally, cloud servers may not be an optimal solution, as storage costs escalate cumulatively (see Supplementary Note 1 and Supplementary Figure 1). However, most CPath studies remain focused on prediction accuracy, thus leaving the issue of efficient data storage unresolved.

> 💡 **问题动机**: 作者把 WSI 压缩定义为 CPath 基础设施问题，而不是单纯图像质量问题。3-4 GB/slide 和 130 TB/6 months 说明“多存硬盘”不是长期方案；如果旧 slide 被删，后续 foundation model 训练、复查、质控都会受影响。

To tackle this large-scale storage issue, several methods have been proposed, including neural image compression (NIC) that effectively compresses the original high-dimensional image into a compressed low-dimensional vector using neural networks [? ? ? ? ? ]. These encoder-focused NIC methods utilize a supervised learning approach to maximize the task-specific prediction performance. However, the encoderfocused NICs are unsuitable for image data storage because, without a decoder, the compressed vectors cannot be reconstructed into the original images. Encoder-decoderbased NICs are more suitable for efficient storage, but limited methods are available for digital pathology applications [? ? ? ? ]. A notable effort by Tellez et al. [? ] involved training Variational Autoencoders (VAE) [? ] and Bidirectional Generative Adversarial Networks (BiGAN) [? ] to encode and decode digital pathology images. They demonstrated the effectiveness of the NIC for digital pathology image compression and restoration, with minimal performance degradation compared to the original image-based methods. Nasr et al. [? ] focused on the compression ratio (CR) using the VAE model, and they reported a state-of-the-art (SOTA) CR (1/512) on cancer imaging data. Previous studies tessellated WSIs into 32x32 or 64x64 pixels. However, Keighley et al. [? ] employed patches of size 512x512 pixels and demonstrated successful compression and restoration using a Vector Quantized VAE (VQVAE)-2 [? ] model. With the matched CR (1/19.2), the outputs of VQVAE-2 (namely, VQVAE) showed high structural similarity (SSIM) to the JPEG-compressed images (0.8); however, the peak signal-to-noise ratio (PSNR) degraded (20 and 29, for VQVAE, and JPEG, respectively). In contrast to NICs, Afshari et al. [? ] applied super-resolution (SR) methods for digital pathology images. The difference between NICs and SR is that the NICs use compressed vectors, whereas SR relies on human-interpretable lowresolution images. In their empirical study, the SR models successfully reconstructed low-resolution images into their original high-resolution forms.

> 💡 **相关方法定位**: 这里把先前方法分成三类：encoder-only NIC 适合任务特征但不适合归档；encoder-decoder NIC 可复原但病理场景有限；SR 用低分辨率图像作为可解释中间表示。AdaSlide 后续的 FIE 更接近“低分辨率/latent 到高分辨率复原”，但论文的新增点不在某个复原 backbone，而在 patch-wise 决策。

Previous studies [? ? ? ? ] have focused on compressing digital pathology images using specific models to achieve a higher CR with minimal information loss. However, even promising models suffer information loss during encoding and decoding steps due to structural limitations, which worsen at higher compression levels. Consequently, there is a risk of losing clinically significant information while aiming to maximize the CR. To address this concern, we focused on the inter-slide information variance in gigapixel WSIs, which we refer to as information disequilibrium (See Section Information Disequilibrium for more details). Each WSI comprises numerous regions of interest (ROIs), some containing diagnostic information while others do not. By enabling the model to distinguish between informative and non-informative ROIs, we can optimize compression by heavily compressing only clinically uninformative ROIs, thereby minimizing critical information loss.

> 💡 **information disequilibrium 的入口**: 关键不是“压缩模型不够强”，而是 uniform compression 的目标函数错了。即使 FIE 很强，高压缩也必然丢信息；所以策略应该把压缩预算更多花在背景、脂肪、低诊断价值 ROI 上，把高风险 ROI 保留。

One recent work [? ] introduced a multiple instance learning (MIL)-based method for identifying informative regions. MIL learns the relationship between the labels provided during training and the instances within a bag, enabling the model to distinguish between more and less important instances. Based on this capability, the authors proposed applying different compression rates to important and less important regions, as determined by a trained MIL classifier. However, this approach has several limitations. Since the MIL classifier is trained in a supervised manner tailored to a specific task, (1) it learns to distinguish instances only in the context of that task, (2) making it difficult to generalize to other tasks with different objectives. Moreover, (3) the method is highly dependent on the performance of the MIL classifier itself and requires an appropriate dataset for MIL training, which may not always be available. Consquently, while this approach can be effective in certain controlled settings, its general applicability is limited, and its scalability is constrained.

> 💡 **和 MIL adaptive 方法的差异**: MIL attention 的“重要区域”来自某个 slide label 和特定任务，换任务就可能换 attention。AdaSlide 选择细胞信息和复原难度作为更通用的代理，并用 RL 学决策，因此更像归档层的通用 compression policy。

To address this issues, we propose an Adaptive compression framework for giga-pixel whole Slide images, namely AdaSlide, to account for information disequilibrium. AdaSlide is designed to meet the following constraints: (1) the lost information should be minimized, (2) each patch image must be maximally compressed, (3) the compression decision should be object-dependent, and (4) the compression decision and image enhancement should be agnostic to backbone-model architecture, organ, magnification, and downstream tasks.

AdaSlide consists of two core modules: a Compression Decision Agent (CDA) and a Foundational Image Enhancer (FIE). The CDA autonomously decides whether to compress an image based on its informational content. The images chosen for compression are downscaled from the D-dimension to the d-dimension ( $D > d$ ), and later restored to the D-dimension using the FIE. The CDA is trained using reinforcement learning, eliminating the need for human annotation and allowing it to consider key factors in compression decisions based on the design of the reward function. The FIE is trained to handle multi-organ, multi-magnification scenarios and can be easily substituted with multiple backbones. In summary, our main contributions are as follows:

• AdaSlide is an organ- and task-agnostic compression method.   
• AdaSlide provides flexible control over compression tendency and quantifies clinically relevant information.   
• AdaSlide apply information disequilibrium in a scalable, task-agnostic compression framework.   
• AdaSlide offers an optimal solution for mid- and long-term archiving, achieving over 65–90% storage reduction with minimal impact on clinical diagnosis.

> 💡 **贡献拆解**: CDA 是“是否压缩”的 policy，FIE 是“压缩后怎么复原”的 model。这个分离很重要：FIE 可以换骨干，CDA reward 可以换信息定义，lambda 可以调保守程度，因此 AdaSlide 更像一套可配置归档框架，而不是单一压缩网络。

## Section 总结

| 线索 | 结论 |
|---|---|
| 存储瓶颈 | WSI 规模让长期归档成本成为 CPath 扩展瓶颈 |
| 现有压缩问题 | uniform 或 task-specific，难以兼顾归档复原和跨任务泛化 |
| AdaSlide 假设 | WSI patch 诊断信息不均衡，应按信息量自适应压缩 |
| 两个模块 | CDA 决策，FIE 复原 |
