[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览
Conclusion 总结贡献和限制；适合回看本文真正解决了 one-step SR 的哪一个子问题。

---

In this paper, we propose TinySR, a highly efficient model that incorporates several novel contributions. For depth pruning, we introduce a Dynamic Inter-block Activation mechanism and an Expansion-Corrosion Strategy to facilitate mask learning optimization. These proposed techniques involve a strategic trade-off between optimization complexity and exploratory potential. To further reduce computational load, we compress the VAE via channel pruning, attention module removal, and the use of lightweight SepConv. Finally, we accelerate inference by eliminating time- and prompt-conditioning modules and implementing pre-caching techniques. Consequently, TinySR achieves up to a $\pmb { 5 . 6 8 } \times$ speedup and a $83 \%$ parameter reduction compared to its teacher, TSD-SR, while maintaining high-quality results.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

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
