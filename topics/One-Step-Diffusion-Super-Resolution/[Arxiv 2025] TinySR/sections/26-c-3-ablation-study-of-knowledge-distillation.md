[← 返回 README](../README.md)

# C.3. Ablation Study of Knowledge Distillation

## 📌 预览
Analysis/Ablation 用来判断每个模块是否必要，特别要看控制变量、loss 和轻量化组件是否各自贡献明确。

---

Table C.3 demonstrates the effectiveness of our knowledge distillation method under stage 1. We can draw the following conclusions: (1) Distillation employing GT (High-Quality) data consistently resulted in unsatisfactory performance, whether applied in the image space or the latent space. As illustrated in Figure C.2, distillation using GT data yields smooth, blurred results, whereas using the teacher produces clearer textures. (2) Distillation performed in the image space achieves better scores on fullreference metrics such as SSIM, LPIPS, and DISTS. Distillation in the latent space yields superior no-reference metrics (MUSIQ, CLIPIQA, TOPIQ and Q-Align), with most even matching those of the teacher model. However, a potential compromise in reference metrics necessitates a second stage of training, which we perform in the image space.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

Table A.1. Quantitative comparison among different GAN-based and diffusion-based Real-ISR approaches on both synthetic and realworld benchmarks. “s” denotes the required number of sampling steps in the diffusion-based method. The best and second-best results are highlighted in bold, italic, respectively

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标，避免被单一数字误导。

![Table A.1.](../images/3098c39c6d3f5ae99d37e02992f8742b875714d4270bbba7182238a32bac3ae2.jpg)
*Table A.1.: Table A.1. Quantitative comparison among different GAN-based and diffusion-based Real-ISR approaches on both synthetic and realworld benchmarks. “s” denotes the required number of sampling steps in the diffusion-based method. The best and second-best results are highlighted in bold, italic, respectively*

> 💡 **Table A.1. 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

Table C.1. Ablation study of prompt token pruning (TP) on DrealSR dataset. The best is highlighted in bold.

> 💡 **批注**: 这里涉及条件信号：prompt 是否准确、是否退化感知，会直接影响生成细节与语义一致性。

![Table C.1.](../images/c83e0112f58991ee86910b90ef066dc1a81a998b6d61ec2323b3fad96853da8a.jpg)
*Table C.1.: Table C.1. Ablation study of prompt token pruning (TP) on DrealSR dataset. The best is highlighted in bold.*

> 💡 **Table C.1. 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

---

## 🔖 Section 总结

### 核心洞察

1. 本节帮助定位论文贡献边界。
2. 读完后回到 README 的方法对比表中归纳。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Speedup | up to 5.68× over TSD-SR teacher |
| Parameter reduction | 83% reduction claimed in abstract |
| Inference regime | one-step Real-ISR |
| Compression targets | Diffusion backbone + VAE + prompt/time modules |
