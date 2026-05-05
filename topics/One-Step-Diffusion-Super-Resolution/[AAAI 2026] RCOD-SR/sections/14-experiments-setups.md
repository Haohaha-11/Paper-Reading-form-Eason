[← 返回 README](../README.md)

# Experiments Setups

## 📌 预览
Experiments 验证方法是否真的改善质量、速度和可控性；注意 full-reference/no-reference 指标之间的张力。

> 💡 **与 RCOD-SR 主线的关系**: RCOD-SR 在 one-step diffusion Real-ISR 中引入 latent domain grouping、退化感知采样蒸馏和视觉 prompt 注入，让单步模型在推理时可控地调节 realism。

---

Datasets: Following the training and testing settings of prior works(Wu et al. 2024b,a; Zhang et al. 2024), we employ LSDIR (Li et al. 2023) and the first 10K face images from FFHQ (Karras, Laine, and Aila 2019) for training and degradation pipeline of Real-ESRGAN (Wang et al. 2021) for LR image synthesizing. The synthetic test data use cropped $5 1 2 \times 5 1 2$ synthetic data from DIV2K-Val (Agustsson and Timofte 2017) and degraded using the Real-ESRGAN pipeline (Wang et al. 2021). The real-world data include LR-HR pairs from RealSR (Cai et al. 2019) and DRealSR (Wei et al. 2020), both with sizes of $1 2 8 \times 1 2 8 .$ - $5 1 2 \times 5 1 2$ for LR-HR pairs.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

Evaluation Metrics: We employ widely used FR and NR metrics. FR metrics include PSNR, SSIM (Wang et al. 2004), LPIPS (Zhang et al. 2018a), and DISTS (Ding et al. 2020). NR metrics include NIQE (Zhang, Zhang, and Bovik

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标，避免被单一数字误导。

Table S2: Quantitative comparison with state-of-the-art methods on both synthetic and real-world benchmarks. The best and second best results are highlighted in bold and underline, respectively.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标，避免被单一数字误导。

![Table S2](../images/18e7075787af7a46cf69ef1bb61230009e0da77b644bb7e2ea833535d8f1ecc5.jpg)
*Table S2: Table S2: Quantitative comparison with state-of-the-art methods on both synthetic and real-world benchmarks. The best and second best results are highlighted in bold and underline, respectively.*

> 💡 **Table S2 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

2015), MANIQA-pipal (Yang et al. 2022), MUSIQ (Ke et al.   
2021), and CLIPIQA (Wang, Chan, and Loy 2023).

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标，避免被单一数字误导。

Method Comparison: We compare our method with stateof-the-art methods (SOTA) in three categories: multi-step diffusion Real-ISR methods, including StableSR (Wang et al. 2024a), ResShift (Yue, Wang, and Loy 2023), Diff-BIR (Lin et al. 2023), and SeeSR (Wu et al. 2024b); onestep diffusion Real-ISR methods, such as SinSR (Wang et al. 2024b), OSEDiff (Wu et al. 2024a), S3Diff (Zhang et al. 2024), TSD-SR (Dong et al. 2025), PiSA-SR (default version) (Sun et al. 2025), and InvSR (Yue, Liao, and Loy 2025); and GAN-based methods, including BSR-GAN (Zhang et al. 2021) and Real-ESRGAN (Wang et al. 2021). We quantitatively compare recent SOTA OSD methods in Table S2 on real-world data. More qualitative comparisons and the full table including synthetic data, multi-step diffusion, and GAN-based methods are in SM.

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
| Control variable | latent-domain group / denoising degree |
| Accepted venue | AAAI 2026 Oral according to arXiv comment |
| Training data change | minimal paradigm modification and original training data claimed |
