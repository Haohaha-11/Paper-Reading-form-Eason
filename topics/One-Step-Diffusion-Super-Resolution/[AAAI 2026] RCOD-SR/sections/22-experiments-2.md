[← 返回 README](../README.md)

# Experiments

## 📌 预览
Experiments 验证方法是否真的改善质量、速度和可控性；注意 full-reference/no-reference 指标之间的张力。

> 💡 **与 RCOD-SR 主线的关系**: RCOD-SR 在 one-step diffusion Real-ISR 中引入 latent domain grouping、退化感知采样蒸馏和视觉 prompt 注入，让单步模型在推理时可控地调节 realism。

---

Quantitative Comparisons: We quantitatively compare recent SOTA SR methods in Table S4 including synthetic and real-world data, multi-step diffusion, and GAN-based methods. Compared with multi-step diffusion methods, our methods demonstrate better NR metrics and perceptual FR metrics across three datasets. Furthermore, their FR fidelity metrics are more competitive with those of multi-step diffusion methods when compared to other OSD methods. On the DrealSR dataset, $\mathrm { R C O D _ { O } }$ -Fid. can even surpass some multistep diffusion methods in many FR metrics (PSNR, SSIM, and LPIPS) and NR metrics (MUSIQ, MANIQA, and CLIP-IQA).

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

Qualitative Comparisons: We conduct more qualitative comparisons in Fig. S7. On Sony 0049 (DRealSR), $\mathrm { R C O D _ { O } }$ -Fid. shows fewer artefacts of handrail than other OSD methods. On Nikon 047 (RealSR) and 0802 pch 00007 (DIV2K), $\mathrm { R C O D _ { O } }$ -Real. and $\mathrm { R C O D } _ { \mathrm { S } }$ - Real. both generate more detailed textures than other OSD methods.

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
