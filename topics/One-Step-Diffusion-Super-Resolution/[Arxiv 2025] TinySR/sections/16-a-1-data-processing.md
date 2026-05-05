[← 返回 README](../README.md)

# A.1. Data Processing.

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 TinySR 主线的关系**: TinySR 面向 Real-ISR 的实时落地，把已有 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存策略压成轻量单步模型。

---

Following [5, 13, 48, 49], the super-resolution process is conducted with a scale factor of 4, upsampling images from $1 2 8 \times 1 2 8$ to $5 1 2 \times 5 1 2$ . To create the degraded dataset, Ground Truth (GT) images are first randomly cropped from their original sources. Subsequently, these GT images are synthesized into $1 2 8 \mathrm { x } 1 2 8$ degraded data via the well-established Real-ESRGAN [6] degradation pipeline, involving various corruptions such as noise, blurring, and compression. This data processing method is widely adopted and well-established. [41, 43, 50, 53]. Moreover, to minimize memory overhead and accelerate training, we preencode the low-quality, high-quality, and teacher-generated references into the VAE’s latent representations. These latent representations are then cached, enabling their swift retrieval during the training phase.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

---

## 🔖 Section 总结

### 核心洞察

1. 明确输入、输出、teacher/student 或控制变量。
2. 把每个 loss/模块对应到 fidelity、realism、speed 或 controllability。
3. 关注哪些组件是训练时使用，哪些是推理时仍有成本。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Speedup | up to 5.68× over TSD-SR teacher |
| Parameter reduction | 83% reduction claimed in abstract |
| Inference regime | one-step Real-ISR |
| Compression targets | Diffusion backbone + VAE + prompt/time modules |
