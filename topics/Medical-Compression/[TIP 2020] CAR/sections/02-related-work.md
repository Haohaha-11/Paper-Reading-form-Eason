[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
Related Work 把下采样方法分成 task independent 和 task specific 两类，CAR 站在第二类里，但用 resampling 避免直接生成 LR 所需的强监督约束。

---

# 2 Related Work

This section presents a review about image downscaling techniques aiming to maintain the visual quality of the LR image. Image downscaling algorithms can be categorized into two groups as follows.

# 2.1 Task independent image downscaling

> 💡 **传统下采样的局限**: Bicubic/Lanczos 这类滤波-抽样方法的缺点不是不可用，而是全局同一个滤波器无法区分纹理、边缘和平坦区域；对 SR 来说，被固定滤波器抹掉或混叠的高频信息之后很难恢复。

Earlier work of image downscaling primarily tends to prevent aliasing [5] which arises during sampling rate reduction. Those methods are based on linear filters [10], where the HR image is firstly convolved with a low-pass kernel to push frequency componenets of the image below the Nyquist frequency [11], then being sub-sampled into target size. Many frequency-based filters are developed, e.g., the box, bilinear and bicubic filter [12]. However, the downscaled images tend to be blurred because the high-frequency details are suppressed. Filters that are designed to model the ideal sinc filter, e.g., the Lanczos filter [13], tend to produce ringing artifacts near strong image edges. All of these filters are predetermined with some of them having tuning parameters. The same filter is applied globally to the input HR image, ignoring characteristics of image content with varying details.

Recently, many researchers begin to focus on the aspects of detail preserving and human perception when developing image downscaling algorithm. Kopf et al.firstly proposed a novel content adaptive image downscaling method based on a joint bilateral filter [14]. The key idea is to optimize the shape and locations of the downsampling kernels to better align with local image features by considering both spatial and color variances of the local region. Öztireli and Gross [15] proposed a method to downscale HR images without filtering. They consider image downscaling as an optimization problem and use the structural similarity index (SSIM) [16] as objective to directly optimize the downscaled image against its original image. This approach helps to capture most of the perceptually important details. Weber et al.[17] proposed an image downscaling algorithm aiming to preserve small details of the input image, which are often crucial for a faithful visual impression. The intuition is that small details transport more information than bigger areas with similar colors. To that end, an inverse bilateral filter is used to emphasize differences rather than punishing them. Gastal and Oliveira [18] introduced the spectral remapping algorithm to control aliasing during image downscaling. Instead of discarding high-frequency information, it remaps such information into the representable range of the downsampled spectrum. Recently, Liu et al.[19] proposed a L0-regularized optimization framework for image downscaling, which is composed of a gradient-ratio prior and reconstruction prior. The downscaling problem is solved by iteratively optimize the two priors in an alternative way.

# 2.2 Task specific image downscaling

> 💡 **与任务驱动方法的差异**: CNN-CR/TAD 等任务驱动方法会直接合成 LR，因此需要 bicubic guidance 或 feature consistency 来约束输出像真实图像。CAR 的设计选择是只学习“如何从 HR 采样”，把合法图像约束放进操作本身，而不是额外 loss。

Most image downscaling algorithms only care about the visual quality of the downscaled image, so that the downscaled image may not be optimal to other computer vision tasks. To tackle this problem, task guided image downscaling has emerged. Zhang et al.[3] took the quality of the interpolated image from the downscaled counterpart into consideration. They proposed an interpolation-dependent image downscaling algorithm by modeling the downscaling operation as the inverse operator of upsampling. Benefiting from the well established deep learning frameworks, Hou et al.[20] proposed a deep feature consistency network that is applicable to image mapping problems. One of the applications illustrated in the paper is image downscaling. The image downscaling network is trained by keeping the deep features of the input HR image and resulting LR image consistent through another pre-trained deep CNN. Kim et al.[21] presented a task aware image downscaling model based on the auto-encoder and the bottleneck layer outputs the downscaled image. In their framework, the encoder acts as the image downscaling network and the decoder is the SR network. The task aware downscaled image is obtained by jointly training the encoder and decoder to maximize the SR performance. Similar to the framework presented by [21], Li et al.[4] proposed a convolutional neural network for image compact resolution named CNN-CR, which is composed of a CNN to estimate the LR image and a learned or specified upscaling model to reconstruct the HR image. The generative nature of the encoder like networks implicitly require additional information to constrain the output to be a valid image whose content resembles the HR image. In [20], in order to compute feature consistency loss against the HR image, they upsample the downscaled image back to the same size as the HR input using nearest neighbor interpolation. In [21, 4], guidance images, obtained using bicubic downsampling, are employed to constrain the output space of the LR image generating networks.

## 🔖 Section 总结

- Task independent：Bicubic/Lanczos/感知增强/L0 等主要优化 LR 视觉质量或抗 aliasing。
- Task specific：CNN-CR/TAD 等为 SR 任务优化 LR，但通常需要 bicubic guidance 或 feature consistency 约束输出空间。
- CAR 的差异：输出由 HR 重采样得到，天然像图像；学习目标来自 SR reconstruction。
