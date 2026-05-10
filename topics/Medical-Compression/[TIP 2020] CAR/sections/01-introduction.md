[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
Introduction 建立 CAR 的应用动机：图像预览、存储、传输需要先下采样再上采样；固定下采样对 SR 不最优，因此下采样器应由 SR 任务反向指导。

---

# 1 Introduction

> 💡 **问题动机**: 论文从移动端预览、存储和传输切入：如果只存一份低分辨率图像，浏览时显示 LR、需要细节时再 SR 到 HR，就要求下采样过程主动为后续上采样保留信息。这个动机与医学压缩里的“低码率传输 + 后端重建/增强”很接近，但本文实验是自然图像 SR。

As the smartphone cameras are starting to rival or beat DSLR cameras, a large number of ultra high resolution images are produced everyday. However, it is always reduced from its original resolution to smaller sizes that are fit to the screen of different mobile devices and web applications. Thus, it is desirable to develop efficient image downscaling and upscaling method to make such application more practical and resources saving by only generating, storing and transmitting a single downscaled version for preview and upscaling it to high resolution when details are going to be viewed. Besides, the pre-downscaling and post-upscaling operation also helps to save storage and bandwidth for image or video compression and communication [1, 2, 3, 4].

Image downscaling is one of the most common image processing operations, aiming to reduce the resolution of the high-resolution (HR) image while keeping its visual appearance. According to the Nyquist-Shannon sampling theorem [5], it is inevitable that high-frequency content will get lost during the samplerate conversion. Contrary to the image downscaling task is the image upscaling, also known as resolution enhancement or super-resolution (SR), trying to recover the underlying HR image of the LR input. Image SR is essentially an ill-posed problem because an undersampled image could refer to numerous HR images. The quality of the SR result is very limited due to the ill-posed nature of the problem and the lost frequency components cannot be well-recovered [6, 7]. Previous work regards image downscaling and SR as independent tasks. Image downscaling techniques pay much attention to enhance details, such as edges, which helps to improve human visual perception [3]. On the other hand, recent state-of-the-art deep SR models have witnessed their capability to restore HR image from the

LR version downscaled using traditional filtering-decimation based methods with great performance gain [8, 9]. However, the predetermined downscaling operations may be sub-optimal to the SR task and state-of-the-art deep SR models still cannot well recover fine details from distorted textures caused by the fixed downscaling operations.

> 💡 **本文方案定位**: 关键转折是“不用 LR 监督”。CAR 不直接生成一个任意 latent-like LR，而是用可微重采样从 HR 像素取值，因此输出天然仍是图像；SRNet 只提供重建目标，告诉下采样器哪些细节对恢复 HR 有用。

In this paper, we proposed a learned content adaptive image downscaling model in which a SR model tries to best recover HR images while adaptively adjusting the downscaling model to produce LR images with potential detailed information that are key to the optimal SR performance. The downscaling model is trained without any LR image supervision. To make sure that the LR image produced by our downscaling model is a valid image, we propose to employ the resampling method where content adaptive non-uniform resampling kernels predicted by a convolutional neural network (CNN) are applied to the original HR image to generate pixels of the LR output. Quantitative and qualitative experimental results illustrate that LR images produced by the proposed model can maintain comparable visual quality as the widely used bicubic interpolation based image downscaling method while advanced SR image quality is obtained using state-of-the-art deep SR models trained with LR images generated from the proposed CAR model.

Our contributions are concluded as follows:

> 💡 **贡献拆解**: 三个贡献分别对应目标、形式和自由度：目标是 SR-guided learned downscaling；形式是 resampling 保证 LR 合法；自由度是 kernel weights + horizontal/vertical offsets，使每个 LR 像素可按局部结构选择采样形状。

• A learned image downscaling model is proposed which is trained under the guidance of the SR model. The proposed image downscaling model produces images that can be well super-resolved while comparable visual quality can be maintained. Experimental results indicate a new state-of-the-art SR performance with the proposed end-to-end image downscaling and upscaling framework. • The resampling method is employed to downscale image by applying content adaptive non-uniform resampling kernels on the original HR input, which can effectively maintain the structure of the HR input in an unsupervised manner. Because directly predicting the LR image by combining low and high-level abstract image features can not guarantee that the generated result is a genuine image without any LR image supervision.

• The learned content adaptive non-uniform resampling kernels perform non-uniform sampling and also make the size of resampling kernels to be more effective. The generated kernels produced by the proposed CAR model are composed of weights and sampling position offsets in both horizontal and vertical directions, making the learned resampling kernels adaptively change their weights and shape according to its corresponding resampling contents.

The rest of the paper is organized as follows. Section 2 reviews task independent and task driven image downscaling algorithms. Section 3 introduces the proposed SR guided content adaptive image downscaling framework, and computing process of each component in the framework is explained. Section 4 evaluates and analyzes experimental results for the SR images and downscaled images quantitatively and qualitatively. Finally, Section 5 summarizes our work.

## 🔖 Section 总结

| 要点 | 内容 |
|---|---|
| 动机 | 只存/传一份 LR，细节需要时再 SR 到 HR |
| 问题 | 固定 bicubic 等滤波器忽略内容差异，损失 SR 所需细节 |
| CAR 解法 | 用 CNN 预测每个 LR 像素的 kernel weights 与 offsets，通过 SR loss 训练 |

> 💡 **Q&A 批注记录**:
> - Q: CAR 为什么强调 LR 是“valid image”？
> - A: 因为它面向存储/传输/预览，LR 不能只是隐藏特征图；正文说明 CAR 通过重采样 HR 像素生成 LR，而不是任意合成像素。
