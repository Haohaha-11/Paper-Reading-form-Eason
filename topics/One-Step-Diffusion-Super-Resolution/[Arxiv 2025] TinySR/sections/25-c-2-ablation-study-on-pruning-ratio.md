[← 返回 README](../README.md)

# C.2. Ablation Study on Pruning Ratio

## 📌 预览
Analysis/Ablation 用来判断每个模块是否必要，特别要看控制变量、loss 和轻量化组件是否各自贡献明确。

---

Table C.2 compares the performance of our method against ShortGPT and TinyFusion across various metrics at token pruning ratios of $33 \%$ , $50 \%$ , and $67 \%$ . The results show our approach surpassing TinyFusion [15] and ShotGPT [33] at every pruning ratio, which demonstrates its robust ability to recover performance. Furthermore, the model exhibits only a slight degradation in performance as the pruning ratio is increased from $33 \%$ to $50 \%$ , indicating the continued presence of parameter redundancy at the $33 \%$ level. However, as the pruning rate increases from $50 \%$ to $67 \%$ , the model’s performance on metrics such as DISTS and MANIQA declines sharply, indicating that excessive pruning leads to irreversible performance degradation.

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

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
