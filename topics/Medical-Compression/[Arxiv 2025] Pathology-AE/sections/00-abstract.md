[← 返回 README](../README.md)

# Abstract

## 预览

这篇论文把给 latent diffusion model 服务的 autoencoder 反过来当作病理图像压缩器：编码端产生低维 latent，解码端重建图像，再用 pathology foundation model 的 embedding similarity 判断压缩是否保住了对诊断/AI 任务有用的信息。

# Pathology Image Compression with Pre-trained Autoencoders

Srikar Yellapragada $\cdot ^ { 1 }$ , Alexandros Graikos $^ { 1 }$ , Kostas Triaridis $^ { 1 }$ , Zilinghan Li $^ 2$ , Tarak Nath Nandi $^ 2$ , Ravi K Madduri $^ 2$ , Prateek Prasanna $^ { 1 }$ , Joel Saltz $^ { 1 }$ , Dimitris Samaras1

$^ { 1 }$ Stony Brook University, 2Argonne National Laboratory myellapragad@cs.stonybrook.edu

Abstract. The growing volume of high-resolution Whole Slide Images in digital histopathology poses significant storage, transmission, and computational efficiency challenges. Standard compression methods, such as JPEG, reduce file sizes but often fail to preserve fine-grained phenotypic details critical for downstream tasks. In this work, we repurpose autoencoders (AEs) designed for Latent Diffusion Models as an efficient learned compression framework for pathology images. We systematically benchmark three AE models with varying compression levels and evaluate their reconstruction ability using pathology foundation models. We introduce a fine-tuning strategy to further enhance reconstruction fidelity that optimizes a pathology-specific learned perceptual metric. We validate our approach on downstream tasks, including segmentation, patch classification, and multiple instance learning, showing that replacing images with AE-compressed reconstructions leads to minimal performance degradation. Additionally, we propose a K-means clustering-based quantization method for AE latents, improving storage efficiency while maintaining reconstruction quality. We provide the weights of the fine-tuned autoencoders at this link.

> 💡 **问题设定**: 这里的目标不是视觉上“看起来还行”的自然图像压缩，而是高倍率病理 patch/WSI 在下游模型中仍保留细胞形态、组织结构和表型线索。JPEG 在高压缩率下的 block artifacts 会直接干扰这些细粒度模式。

> 💡 **核心转用**: LDM autoencoder 原本服务于生成模型，把像素图像压到结构化 latent，再由 diffusion 在 latent 空间建模。本文抓住的是“高保真编码/解码器”本身：不训练新 compressor，而是复用 SD-1.5、SD-3、DC-AE 的预训练压缩瓶颈。

> 💡 **评价链条**: 论文没有只停在 PSNR/SSIM，而是用 UNI/Phikon-v2/Gigapath embedding similarity，再接 segmentation、patch classification 和 MIL。这个证据链的逻辑是：foundation-model embedding 接近只是代理指标，最终还要看任务性能是否掉。

> 💡 **latent quantization 重点**: AE latent 是连续浮点数，直接存会抵消一部分压缩收益。作者发现 latent 分布不均匀、集中在 0 附近，因此用 K-means 学 256 个中心，比静态 int8 等宽 bin 更贴合真实 latent 分布。

Keywords: Image compression · Histopathology · Autoencoders.

## Section 总结

| 组件 | 论文中的角色 | 阅读时要抓住的点 |
|------|--------------|------------------|
| LDM autoencoder | learned lossy compressor | latent 空间低维、保持 locality 与重建质量 |
| Pathology perceptual metric | fine-tuning 信号 | 用 UNI embedding L1 距离补足像素损失 |
| K-means latent quantization | 存储优化 | 以 learned centroid 替代 uniform int8 bins |
| Downstream tasks | 功能验证 | segmentation / classification / MIL 检查是否真的可用 |
