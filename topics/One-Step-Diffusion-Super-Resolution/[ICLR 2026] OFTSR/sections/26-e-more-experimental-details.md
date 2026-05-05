[← 返回 README](../README.md)

# E MORE EXPERIMENTAL DETAILS

## 📌 预览
Experiments 验证方法是否真的改善质量、速度和可控性；注意 full-reference/no-reference 指标之间的张力。

> 💡 **与 OFTSR 主线的关系**: OFTSR 用 conditional flow teacher 和 ODE-trajectory alignment distillation 构建 one-step SR，并保留可调 fidelity-realism trade-off。

---

The training of all networks across both stages is smoothed using Exponential Moving Average (EMA) with a ratio of 0.9999. For FFHQ and ImageNet datasets, images are resized to 256 pixels with center cropping, while DIV2K training employs random crops of $2 5 6 \times 2 5 6$ patches. Data augmentation consists of horizontal flips with $50 \%$ probability and vertical flips with $6 \%$ probability throughout all experiments. For FFHQ noiseless experiment, we use default perturbation std $\sigma _ { p } =$ 0.1; for FFHQ noisy experiment, we use a higher perturbation std $\sigma _ { p } = 0 . 5$ to cover the resized noise from LR images, as suggested in Tab. 11; for both DIV2K and ImageNet we use $\sigma _ { p } = 0 . 2$ . For training, we employed three widely-used datasets: the standard ImageNet training set (1.28M images), the DIV2K training set (800 2K resolution images), and a subset of FFHQ consisting of the first 60,000 images from the dataset. All models are trained until convergence or up to $3 0 0 \mathrm { k }$ training iterations and we select the model based on best metrics. We train the model with uniform loss weight on $t$ . In the distillation stage, we sample the timestep $t$ using $t \sim \mathcal { U } \left[ t _ { \mathrm { m i n } } , t _ { \mathrm { m a x } } \right]$ with $t _ { \mathrm { m i n } } = 0 . 0 1$ and $t _ { \mathrm { m a x } } = 0 . 9 9$ in practice.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

For DIV2K evaluation, we first segment the large 2K resolution images into $2 5 6 \times 2 5 6$ patches for model inference, then reconstruct the final image by combining the restored patches. To ensure fair

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐，而不只是匹配终点。

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
| Teacher type | conditional flow-based SR model |
| Distillation target | same sampling ODE trajectory alignment |
| Datasets | FFHQ 256×256, DIV2K, ImageNet 256×256 |
