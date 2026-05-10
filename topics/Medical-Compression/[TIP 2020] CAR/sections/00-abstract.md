[← 返回 README](../README.md)

# 0 Abstract

## 📌 预览
CAR 在摘要中提出一个 SR-guided learned downscaling 框架：ResamplerNet 预测内容自适应 kernel 与 offsets，生成 LR；SRNet/EDSR 将 LR 复原为 HR，并用 L1 重建误差反向训练下采样器。

---

# Abstract

Deep convolutional neural network based image super-resolution (SR) models have shown superior performance in recovering the underlying high resolution (HR) images from low resolution (LR) images obtained from the predefined downscaling methods. In this paper we propose a learned image downscaling method based on content adaptive resampler (CAR) with consideration on the upscaling process. The proposed resampler network generates content adaptive image resampling kernels that are applied to the original HR input to generate pixels on the downscaled image. Moreover, a differentiable upscaling (SR) module is employed to upscale the LR result into its underlying HR counterpart. By back-propagating the reconstruction error down to the original HR input across the entire framework to adjust model parameters, the proposed framework achieves a new state-of-the-art SR performance through upscaling guided image resamplers which adaptively preserve detailed information that is essential to the upscaling. Experimental results indicate that the quality of the generated LR image is comparable to that of the traditional interpolation based method and the significant SR performance gain is achieved by deep SR models trained jointly with the CAR model. The code is publicly available on: https://github.com/sunwj/CAR.

> 💡 **摘要主线**: CAR 把下采样从固定滤波器改成由 HR 内容预测的 resampling kernels 和 offsets，并把 SRNet 的 L1 重建误差反传到下采样器。它解决的不是“LR 本身最锐利”，而是“LR 在存储/传输后能被 SR 网络恢复得最好，同时视觉质量接近 bicubic”。


## 🔖 Section 总结

- 核心变量：content adaptive resampling kernels、horizontal/vertical offsets、differentiable SR module。
- 核心洞察：LR 图像质量“接近 bicubic”只是约束之一，真正目标是让后续 SR 取得更高 PSNR/SSIM 和视觉恢复质量。
- 可追问点：如果 SRNet 换成医学图像重建网络，CAR 学到的 LR 是否仍可被医生直接查看？
