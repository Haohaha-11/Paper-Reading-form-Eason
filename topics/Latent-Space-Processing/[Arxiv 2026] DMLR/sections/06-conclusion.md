[← 返回 README](../README.md)

# 6. Conclusion

## 📌 预览
结论回到两个核心变量：视觉信息使用和 confidence。DMLR 的最终 claim 是 training-free、test-time、同时增强 reasoning 与 perception。

---

# 6. Conclusion

In this work, we analyze how MLLMs utilize visual information and confidence during reasoning. Based on these observations, we introduce DMLR, a test-time multimodal latent reasoning framework that integrates confidence-guided latent optimization with dynamic visual injection.This method enables models to refine their reasoning, retrieve visual evidence only when need without training. Extensive experiments across various tasks show that DMLR consistently boosts both reasoning and perception tasks, offering a stable and training-free alternative to other methods.

> 💡 **结论批读**: 结论强调“without training”，这是 DMLR 的部署吸引力。但它并不是无成本：需要 eager attention、test-time latent update 和超参控制。真正的边界问题是：confidence 在高风险域或 OOD 图像上是否仍可靠。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 结论 |
|------|------|
| 训练需求 | 不训练模型参数 |
| 核心机制 | confidence-guided latent optimization + dynamic visual injection |
| 效果范围 | reasoning 与 perception 均提升 |

### 核心洞察
1. DMLR 把“多看图”和“多想几步”都变成 latent-space 中的可控推理过程。
2. 论文没有充分展开失败案例和 OOD 风险，后续值得补实验。
3. 对工程落地，最需要验证的是 attention access、batch latency 和 confidence calibration。
