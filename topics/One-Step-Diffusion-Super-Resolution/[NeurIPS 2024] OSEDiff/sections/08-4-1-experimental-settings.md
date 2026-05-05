[← 返回 README](../README.md)

# 4.1 Experimental Settings

## 📌 预览
Experiments 验证方法是否真的改善质量、速度和可控性；注意 full-reference/no-reference 指标之间的张力。

> 💡 **与 OSEDiff 主线的关系**: OSEDiff 将低质量图像直接作为扩散起点，用少量 LoRA/可训练层和 latent-space VSD 正则把 Stable Diffusion 的生成先验压缩到单步 Real-ISR 推理中。

---

Training and Testing Datasets. Prior works [42, 57, 31, 52] employed different training datasets, making unified training standards for Real-ISR difficult to establish. For simplicity, we adopt SeeSR’s setup [52] and train OSEDiff using the LSDIR [26] dataset and the first 10K face images from FFHQ [19]. The degradation pipeline of Real-ESRGAN [45] is used to synthesize LQ-HQ training pairs. We evaluate OSEDiff and compare it with competing methods using the test set provided by StableSR [42], including both synthetic and real-world data. The synthetic data includes 3000 images of size $5 1 2 \times 5 1 2$ , whose GT are randomly cropped from DIV2K-Val [2] and degraded using the Real-ESRGAN pipeline [45]. The real-world data include LQ-HQ pairs from RealSR [3] and DRealSR [51], with sizes of $1 2 8 \times 1 2 8$ and $5 1 2 \times 5 1 2$ , respectively.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

Compared Methods. We compare OSEDiff with state-of-the-art DM-based Real-ISR methods, including StableSR [42], ResShift [60], PASD [57], DiffBIR [31], SeeSR [52] and SinSR [48]. Among them, StableSR, PASD, DiffBIR, and SeeSR are all based on the pre-trained SD model. ResShift trains a DM from scratch in the pixel domain, while SinSR is a one-step model distilled from ResShift. Note that we do not compare with the recent method SUPIR [59] because it tends to generate rich yet excessive details, which are however unfaithful to the input image.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

Table 1: Quantitative comparison with state-of-the-art methods on both synthetic and real-world benchmarks. ‘s’ denotes the number of diffusion reverse steps in the method. The best and second best results of each metric are highlighted in red and blue, respectively.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标，避免被单一数字误导。

![Table 1](../images/a236a580e95d4b33c1e2db66f2985b919d218318e50ffc850d0e4c768ce9bf42.jpg)
*Table 1: Table 1: Quantitative comparison with state-of-the-art methods on both synthetic and real-world benchmarks. ‘s’ denotes the number of diffusion reverse steps in the method. The best and second best results of each metric are highlighted in red and blue, respectively.*

> 💡 **Table 1 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

Table 2: Complexity comparison among different methods. All methods are tested with an input image of size $5 1 2 \times 5 1 2$ , and the inference time is measured on an A100 GPU.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标，避免被单一数字误导。

![Table 2](../images/d194b2d67bcbeb864681ee086090e39d129a94a01ec496f7b7fa2ffaff4f4eaa.jpg)
*Table 2: Table 2: Complexity comparison among different methods. All methods are tested with an input image of size $5 1 2 \times 5 1 2$ , and the inference time is measured on an A100 GPU.*

> 💡 **Table 2 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

For those GAN-based Real-ISR methods, including BSRGAN [61], Real-ESRGAN [45], LDL [27], and FeMaSR [4], we put their results into the Appendix.

Evaluation Metrics. To provide a comprehensive and holistic assessment on the performance of different methods, we employ a range of full-reference and no-reference metrics. PSNR and SSIM [50] (calculated on the Y channel in YCbCr space) are reference-based fidelity measures, while LPIPS [64], DISTS [12] are reference-based perceptual quality measures. FID [15] evaluates the distance of distributions between GT and restored images. NIQE [62], MANIQA-pipal [55], MUSIQ [22], and CLIPIQA [41] are no-reference image quality measures. We also conduct a user study, which is presented in the Appendix.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

Implementation Details. We train OSEDiff with the AdamW optimizer [33] at a learning rate of 5e-5. The entire training process took approximately 1 day on 4 NVIDIA A100 GPUs with a batch size of 16. The rank of LoRA in the VAE Encoder, diffusion network, and finetuned regularizer is set to 4. For the text prompt extractor, although advanced multimodal language models [32] can provide detailed text descriptions, they come at a high inference cost. We adopt the degradation-aware prompt extraction (DAPE) module in SeeSR [52] to extract text prompts. The SD 2.1-base is used as the pre-trained T2I model. The weighting scalars $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are set to 2 and 1, respectively.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

---

## 🔖 Section 总结

### 核心洞察

1. 同时看 PSNR/SSIM 与 LPIPS/FID/NIQE/MUSIQ 等指标。
2. 检查 one-step 是否真的在 latency/参数量上有优势。
3. 优先关注 ablation 是否支持论文的主模块设计。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Inference time | 0.11s on A100 for 512×512 input |
| Trainable parameters | 8.5M |
| Speedup claim | over 100× faster than StableSR in paper comparison |
