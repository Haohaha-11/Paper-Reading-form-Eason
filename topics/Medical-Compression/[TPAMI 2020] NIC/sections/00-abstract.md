[← 返回 README](../README.md)

# 00 - Abstract

## 📌 预览

摘要给出 NIC 的两阶段结构：先无监督压缩 gigapixel image，再在压缩表示上用 image-level labels 训练 CNN；核心评估点是 encoder 选择、弱监督预测、以及 Grad-CAM 定位是否和专家标注一致。

---

# Neural Image Compression for Gigapixel Histopathology Image Analysis

David Tellez\*, Geert Litjens, Jeroen van der Laak, Francesco Ciompi

Abstract—We propose Neural Image Compression (NIC), a two-step method to build convolutional neural networks for gigapixel image analysis solely using weak image-level labels. First, gigapixel images are compressed using a neural network trained in an unsupervised fashion, retaining high-level information while suppressing pixel-level noise. Second, a convolutional neural network (CNN) is trained on these compressed image representations to predict image-level labels, avoiding the need for fine-grained manual annotations. We compared several encoding strategies, namely reconstruction error minimization, contrastive training and adversarial feature learning, and evaluated NIC on a synthetic task and two public histopathology datasets. We found that NIC can exploit visual cues associated with image-level labels successfully, integrating both global and local visual information. Furthermore, we visualized the regions of the input gigapixel images where the CNN attended to, and confirmed that they overlapped with annotations from human experts.

> 💡 **摘要机制拆解**: “compress then classify” 是 NIC 的核心。第一步 encoder 只看 patch、无监督学习，把像素噪声压下去；第二步 CNN 看的是 embedding grid 而不是原图像素，因此可以用 WSI 级弱标签训练。

Index Terms—Gigapixel image analysis, computational pathology, convolutional neural networks, representation learning.

> 💡 **任务边界**: 关键词里没有 segmentation/MIL，是因为本文不是先检测病灶再聚合，而是把 WSI 改写成 CNN 可处理的 latent image。

## 🔖 Section 总结

| 要点 | 内容 |
|---|---|
| 输入 | Gigapixel histopathology WSI + image-level labels |
| 中间表示 | 无监督 encoder 生成的 compressed embedding grid |
| 输出 | WSI 分类/回归预测 + Grad-CAM saliency |
| 关键判断 | encoder 是否保留高层语义并压制 pixel-level noise |
