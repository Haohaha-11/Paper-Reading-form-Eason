[← 返回 README](../README.md)

# 3. Methodology

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 TinySR 主线的关系**: TinySR 面向 Real-ISR 的实时落地，把已有 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存策略压成轻量单步模型。

---

We aim to learn a compact but effective diffusion model for real-world image super-resolution (Real-ISR). To mitigate the computational load and inference latency caused by the excessively deep architecture of the TSD-SR network, we prioritize depth pruning (Sec. 3.1) due to its superior pruning efficiency, as shown in Fig. 3. In Sec. 3.2, we introduce several key innovations to improve computational efficiency, including a lightweight VAE, a conditional information removal strategy, and a pre-caching technology. Last, the complete training recipes are given in Sec. 3.3.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

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
