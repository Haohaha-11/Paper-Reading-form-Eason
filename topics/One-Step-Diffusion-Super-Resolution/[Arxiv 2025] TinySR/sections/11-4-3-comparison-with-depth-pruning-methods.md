[← 返回 README](../README.md)

# 4.3. Comparison with Depth Pruning Methods

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 TinySR 主线的关系**: TinySR 面向 Real-ISR 的实时落地，把已有 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存策略压成轻量单步模型。

---

We evaluate the depth pruning methods following these baselines: (1) Perturbation-based – We randomly prune models and select one with minimal task loss (LPIPS) for training; (2) Similarity-based – Typically, these methods base their decisions on an analysis of the similarity between each layer’s input and output, such as Flux-Lite [10] and ShortGPT [33]; (3) Metric-based – Decision-making through metric in general, such as Sensitivity Analysis [16] and SnapFusion [28]. (4) Experience-based - We follow the design of BK-SDM [25] for corresponding pruning; (5) Probability-based – Decision by optimized probability parameters, such as TinyFusion [15] and our method. Randomly generating and then selecting the minimum loss yields a low initial loss, but exhibits extremely weak recovery ability after training. Similarity-based methods demonstrate suboptimal fidelity (poor DISTS) and pronounced inconsistencies across various no-reference metrics. For example, ShortGPT performs well on CLIPIQA but struggles considerably on MANIQA. Metric-based methods often exhibit a bias towards the metrics they optimize. For instance, while Sensitivity Analysis performs well on reference metrics, and SnapFusion excels on CLIPIQA due to its pruning scheme’s relation to it, both approaches demonstrate shortcomings in other metrics. Although BK-SDM and Tiny-Fusion show some effectiveness, our method exhibits enhanced recoverability over all other approaches, performing favorably across both full-reference and no-reference evaluation metrics.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

![Figure 8.](../images/d58ff4644c0fd5de45ad57b085de82bf6306d2d3c0d620419ceffb111fec6d5d.jpg)
*Figure 8.: Figure 8. Qualitative comparisons of different DMs-based Real-ISR methods. Please zoom in for a better view.*

> 💡 **Figure 8. 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

Table 3. Ablation study of VAE compression on DrealSR.

![Table 3.](../images/3a7fc9ac0df56fed68b7149205dbab53fb3016093831482b3e44877a21232a19.jpg)
*Table 3.: Table 3. Ablation study of VAE compression on DrealSR.*

> 💡 **Table 3. 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

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
