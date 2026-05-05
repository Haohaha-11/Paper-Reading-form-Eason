[← 返回 README](../README.md)

# 4.1. Experimental Settings

## 📌 预览
Experiments 验证方法是否真的改善质量、速度和可控性；注意 full-reference/no-reference 指标之间的张力。

> 💡 **与 TinySR 主线的关系**: TinySR 面向 Real-ISR 的实时落地，把已有 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存策略压成轻量单步模型。

---

Datasets. We utilize DIV2K [1], Flickr2K [39], LSDIR [29], and FFHQ [23] for training. To synthesize lowresolution and high-resolution image pairs, we employ the same degradation pipeline as described in Real-ESRGAN [42]. We evaluate the performance of our model on the synthetic DIV2K-Val [1] dataset, alongside two real-world datasets, RealSR [3] and DRealSR [46]. The datasets consist of paired images with $1 2 8 \mathrm { x } 1 2 8$ low-quality and 512x512 high-quality resolutions.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

Evaluation Metrics. For evaluating our method, we apply both full-reference and no-reference metrics. Full-reference metrics include PSNR and SSIM [44] (calculated on the Y channel in YCbCr space) for fidelity, LPIPS [56] and DISTS [12] for perceptual quality, and FID [18] for distribution comparison. No-reference metrics include NIQE [55], MUSIQ [24], MANIQA [51], CLIPIQA [40], TOPIQ [7] and Q-Align [47].

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

---

## 🔖 Section 总结

### 核心洞察

1. 同时看 PSNR/SSIM 与 LPIPS/FID/NIQE/MUSIQ 等指标。
2. 检查 one-step 是否真的在 latency/参数量上有优势。
3. 优先关注 ablation 是否支持论文的主模块设计。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Speedup | up to 5.68× over TSD-SR teacher |
| Parameter reduction | 83% reduction claimed in abstract |
| Inference regime | one-step Real-ISR |
| Compression targets | Diffusion backbone + VAE + prompt/time modules |
