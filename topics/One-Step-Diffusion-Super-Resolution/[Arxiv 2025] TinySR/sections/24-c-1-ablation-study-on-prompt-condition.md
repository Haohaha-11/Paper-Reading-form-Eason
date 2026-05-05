[← 返回 README](../README.md)

# C.1. Ablation Study on Prompt Condition

## 📌 预览
Analysis/Ablation 用来判断每个模块是否必要，特别要看控制变量、loss 和轻量化组件是否各自贡献明确。

---

Table C.1 presents the results of token pruning of the teacher model. We found that pruning prompt information does not negatively impact certain full-referenced metrics. In fact, some metrics, such as LPIPS and DISTS, even show improvement at specific pruning rates. Token pruning primarily affects no-referenced metrics. However, we observe no significant performance degradation even at a $50 \%$ pruning ratio. Furthermore, performance degrades gracefully at higher pruning percentages without a sharp decline, which suggests that the contribution of textual information to the final image synthesis is limited. As shown in Figure C.1, although TP $90 \%$ token’s output contains less fine-grained detail than the baseline, it effectively removes the noise from the low-quality input, resulting in an image with high visual quality.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

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
