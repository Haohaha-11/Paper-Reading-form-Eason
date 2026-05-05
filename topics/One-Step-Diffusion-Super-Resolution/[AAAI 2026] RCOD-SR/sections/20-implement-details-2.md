[← 返回 README](../README.md)

# Implement Details

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 RCOD-SR 主线的关系**: RCOD-SR 在 one-step diffusion Real-ISR 中引入 latent domain grouping、退化感知采样蒸馏和视觉 prompt 注入，让单步模型在推理时可控地调节 realism。

---

Training: The training takes over $3 0 \mathrm { k }$ iterations, with a batch size of 4 and a learning rate of $\mathrm { 2 e ^ { - 5 } }$ . We adopted LoRA for fine-tuning. Follow the original setting $\mathrm { W u }$ et al. 2024a; Zhang et al. 2024), for $\mathrm { R C O D } _ { \mathrm { S } }$ , the rank parameter in LoRA is set as 16 for the VAE encoder and 32 for the diffusion UNet, respectively. For $\mathrm { R C O D _ { O } }$ , all rank is set to 4.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

Hardware and Software Environments: We train and test our model on NVIDIA A100 GPU. The amount of training memory is about $2 8 \ \mathrm { G B }$ with batch size 1 using $\mathrm { R C O D _ { O } }$ . We use a Linux system with pytorch version 2.4.

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐，而不只是匹配终点。

---

## 🔖 Section 总结

### 核心洞察

1. 明确输入、输出、teacher/student 或控制变量。
2. 把每个 loss/模块对应到 fidelity、realism、speed 或 controllability。
3. 关注哪些组件是训练时使用，哪些是推理时仍有成本。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Control variable | latent-domain group / denoising degree |
| Accepted venue | AAAI 2026 Oral according to arXiv comment |
| Training data change | minimal paradigm modification and original training data claimed |
