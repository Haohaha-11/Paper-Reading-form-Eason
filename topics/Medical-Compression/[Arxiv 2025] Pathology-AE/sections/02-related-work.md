[← 返回 README](../README.md)

# 2 Related Work

## 预览

Related Work 很短，但它划清了三条线：传统 autoencoder compression、LDM autoencoder 的结构优势、以及数字病理中 JPEG/learned compression 的泛化问题。

Autoencoders for compression. Using autoencoders for data compression has been extensively studied with previous works showing that learned compression schemes can outperform standard compression algorithms [24,2]. However, only recently have autoencoders trained on large-scale datasets, such as those in LDMs, been explored for image compression.

> 💡 **定位**: 作者不声称 AE compression 是新概念；新点是拿大规模生成模型训练出的 AE 当通用压缩器。这个区别很重要，因为病理数据规模和多样性通常不足以从头训练泛化性强的 compressor。

LDM autoencoders. The autoencoders used in LDMs embed the images into a structured, 2D latent space that downscales the image size. The AEs are trained with reconstruction, perceptual [28] and patch-based adversarial losses [16] to ensure the high fidelity of the outputs. To constrain the learned latent space, Stable Diffusion [21] employs KL regularization [18] that centers the latent embeddings around zero and imposes unit variance. In DC-AE [4] there is no explicit regularization on the learned latent space, but it still maintains similar properties to the latent spaces learned by the SD AEs.

> 💡 **latent 分布伏笔**: KL regularization 让 SD 系 latent 接近零均值单位方差，DC-AE 虽无显式 KL 但也有类似集中性。后文 K-means quantization 正是利用“latent value 非均匀、接近 0 更密集”这个事实。

Image compression in digital pathology. The current go-to approach for image compression in digital pathology is JPEG [12]. Previous attempts at learned compression for pathology images have also used autoencoder models [23,11] with recent works proposing domain-specific decompositions for increased efficiency [10]. However, all existing learning-based compression schemes fail to generalize outside the training data distribution [12]. In contrast, the LDM autoencoders have been designed and trained to work with web images, allowing them to be used in a wider array of settings.

> 💡 **泛化论点**: 这里有一个可讨论的假设：web-image LDM AE 没见过病理图像细节，却可能比病理小数据训练的 compressor 更泛化。作者后文用 TCGA patches、NCT-CRC、BCSS、CRAG 和 BRCA MIL 来支撑这个跨数据任务链条。

## Section 总结

| 相关方向 | 本文继承什么 | 本文避开什么 |
|----------|--------------|--------------|
| Classic learned image compression | AE bottleneck 可压缩图像 | 从头训练病理专用 compressor |
| LDM autoencoders | 2D latent、perceptual/GAN 训练、高保真重建 | diffusion denoiser 本体 |
| Digital pathology compression | 关注 AI 下游任务和病理细节 | 只用 JPEG/PSNR 评价 |
