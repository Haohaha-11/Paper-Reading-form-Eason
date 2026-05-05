[← 返回 README](../README.md)

# A.1 Comparison with GAN-based Methods

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 OSEDiff 主线的关系**: OSEDiff 将低质量图像直接作为扩散起点，用少量 LoRA/可训练层和 latent-space VSD 正则把 Stable Diffusion 的生成先验压缩到单步 Real-ISR 推理中。

---

We compare OSEDiff with four representative GAN-based Real-ISR methods, including BSRGAN [61], Real-ESRGAN [45], LDL [27] and FeMaSR [4]. The results are shown in Table 8. It is not a surprise that GAN-based methods have better fidelity measures such as PSNR and SSIM than OSEDiff. However, OSEDiff has much better perceptual qualify metrics. We also provide visual comparisons in Figure 5. Compared to GAN-based methods, OSEDiff is able to generate realistic and reasonable details, such as squirrel hair, textures of petals, buildings, and leaves.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

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
| Inference time | 0.11s on A100 for 512×512 input |
| Trainable parameters | 8.5M |
| Speedup claim | over 100× faster than StableSR in paper comparison |
