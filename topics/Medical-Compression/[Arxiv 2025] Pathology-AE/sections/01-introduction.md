[← 返回 README](../README.md)

# 1 Introduction

## 预览

Introduction 把问题从“WSI 太大”推进到“传统压缩会破坏 AI 消费的病理细节”，再说明为什么 LDM autoencoder 的 latent bottleneck 是一个可复用的压缩结构。

With the increasing digitization of histopathology, large repositories of Whole Slide Images (WSIs), such as TCGA [3], have been invaluable for the development of large-scale machine learning models [5,20,8]. However, the size of high-resolution pathology images presents a major bottleneck in storage, transmission, and computational efficiency. A large pathology center produces over 1 million digital slides per year [22], which translates to petabytes of storage. Long-term retention of these slides, even in deep archival storage, could cost up to $\$ 100,000$ per year. As the volume of pathology data grows, developing efficient compression techniques that reduce storage requirements while preserving information relevant for AI model ‘consumption’ remains a key issue.

> 💡 **需求不是纯归档**: 作者明确把“AI model consumption”写进问题定义，说明压缩目标与传统医院归档不同：压缩后图像要能继续喂给 foundation model、segmentation model、MIL classifier，而不是只供人眼粗看。

Looking for ways to reduce image size, generative diffusion models [14] have employed autoencoders (AEs) for efficient high-resolution image synthesis. The

![](../images/b85283c75c0483e320b794c72631074f7512fff7c85b238ab046e61662c05d6e.jpg)  
Fig. 1. Examples of image reconstruction using JPEG, the vanilla DC-AE [4] and fine-tuned DC-AE. JPEG at quality 10, with a comparable file size to DC-AE, introduces severe compression artifacts, including deformed nuclei and blocky artifacts (highlighted in yellow and teal). Vanilla DC-AE fails to retain certain cell structures (green), which are largely recovered through our fine-tuning strategy.

> 💡 **Fig. 1 的阅读方式**: 这张图不是只做视觉展示，而是在定义 failure mode：JPEG-10 的块效应和核形变是“压缩伪影”；vanilla DC-AE 的细胞结构缺失是“跨域重建偏差”；fine-tuning 要解决的是后者。

Latent Diffusion Model (LDM) framework [21] introduced a trained autoencoder that first compresses images into a lower-dimensional latent representation before applying the diffusion model. These autoencoders are designed to maintain high reconstruction fidelity while preserving spatial locality and ensuring generalizability. While originally optimized for generative modeling, they are equally well-suited for image compression [23,11], making pre-trained LDM autoencoders a promising choice across diverse image domains and sizes.

> 💡 **为什么是 LDM AE**: LDM autoencoder 的 2D latent 仍保留空间局部性，所以压缩后能 tile-wise/patch-wise 解码；同时它在 web-scale 图像上训练，比只在小型病理数据上训练的 learned compressor 更可能跨组织、染色和扫描条件泛化。

In this work, we repurpose LDM autoencoders as efficient image compression models for histopathology. Conventional compression methods, such as JPEG, can significantly reduce image sizes but struggle to preserve the fine-grained pathology features critical for downstream tasks when aggressively compressing [6]. By leveraging the learned representations of LDM autoencoders, we achieve high compression rates while maintaining the essential pathology details. As shown in Fig. 1, heavy JPEG compression leads to substantial artifacts, whereas comparable compression with the DC-AE autoencoder model gives coherent reconstructions.

> 💡 **压缩范式变化**: JPEG 通过固定频域变换和量化压掉信息；AE compression 是学习一个 decoder 可解释的 latent。病理场景里，关键比较不是“同样 KB 下像素误差谁小”，而是“同样 KB 下细胞/组织表征谁更接近原图”。

We systematically benchmark three AEs — Stable Diffusion 1.5 (SD-1.5) [21], Stable Diffusion 3 (SD-3) [7], and Deep Compression Autoencoder (DC-AE-f32) [4] – each offering different compression rates. We assess the perceptual similarity between original and reconstructed images using pathology foundation models [5,9,25], finding that existing AEs perform surprisingly well (Fig.2 left). To further improve the AE reconstructions, we propose a fine-tuning strategy that optimizes the decoder for a pathology-specific learned perceptual metric. This fine-tuning strategy further aligns pathology-specific features between the reconstructions and the original without altering the desirable properties (generalizability, locality) of the autoencoder.

> 💡 **decoder-only fine-tuning 伏笔**: “without altering generalizability/locality” 对应后文只微调 decoder、不动 encoder。这样 latent 空间仍是原预训练 AE 的空间，避免小规模 TCGA patch fine-tuning 把 encoder 学窄。

![](../images/15f6d6b55624d2e4afb2e664b4e3684ee52e0fc8fa2fac8e9813a7d6588eb6bf.jpg)  
Fig. 2. Left: Pre-trained autoencoders outperform JPEG in reconstruction fidelity, further improved by fine-tuning with a pathology-specific perceptual loss. Right: Using fine-tuned AE-compressed reconstructions results in minimal performance degradation. The width of each bar denotes the relative sizes of the compressed representation.

> 💡 **Fig. 2 的双轴证据**: 左边证明 embedding fidelity，右边证明 downstream viability。bar 宽表示压缩表示大小，这一点很重要：DC-AE-f32 不是 fidelity 最高，但在极小 latent size 下仍优于重压缩 JPEG，是“高压缩率区间”的关键模型。

Beyond perceptual similarity, we validate our compression pipeline on multiple downstream tasks, including segmentation, patch and multiple instance learning classification. We show that replacing raw images with AE-compressed reconstructions results in minimal performance degradation, demonstrating the practical viability of our approach (Fig.2 right). Finally, we introduce a K-means clustering-based quantization for the AE latents that considers the unique characteristics of the latent distribution. By mapping the continuous latent representations to a fixed set of discrete values, we further reduce storage while preserving high reconstruction fidelity, outperforming static 8-bit (int8) rounding-based quantization, which introduces artifacts and degrades image quality.

> 💡 **三段式方法闭环**: 先用预训练 AE 得到可用 compression；再用 UNI perceptual loss 做 pathology alignment；最后用 K-means quantization 把 float latent 变成 8-bit index。没有最后一步，latent 存储成本会被浮点表示放大。

Our contributions are as follows

– We repurpose LDM autoencoders for pathology image compression, demonstrating their ability to achieve high compression rates while preserving essential phenotypical details.   
– We benchmark three AEs across multiple compression rates and evaluate their performance using pathology foundation models.   
We introduce a fine-tuning strategy to enhance reconstruction fidelity by aligning AE latents with a pathology-specific learned perceptual metric. We enhance storage efficiency with a K-means clustering-based quantization for AE latents, outperforming static int8 quantization, while preserving reconstruction fidelity.

> 💡 **贡献拆分**: 这篇论文的贡献不是提出全新 AE 架构，而是把“预训练 LDM AE + pathology foundation evaluation/fine-tuning + latent quantization + downstream validation”组织成一个病理压缩方案。

## Section 总结

| 主张 | 证据入口 |
|------|----------|
| WSI 存储/传输压力巨大 | 年产 >1M slides、petabyte storage、长期归档成本 |
| LDM AE 可作为压缩器 | SD-1.5 / SD-3 / DC-AE latent bottleneck |
| 病理压缩要看语义/任务 | UNI/Phikon-v2/Gigapath + downstream tasks |
| fine-tuning 只改重建解释 | decoder-only + pathology perceptual loss |
| 进一步省存储靠 latent quantization | K-means 256 centers 替代 static int8 |
