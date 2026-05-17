[← 返回 README](../README.md)

# 6. Conclusion

## 一、Preview

本文的结论简洁地回顾了两个核心发现（视觉信息使用模式 + 置信度的作用）以及 DMLR 框架的贡献。

---

## 二、原始文本

In this work, we analyze how MLLMs utilize visual information and confidence during reasoning. Based on these observations, we introduce DMLR, a test-time multimodal latent reasoning framework that integrates confidence-guided latent optimization with dynamic visual injection. This method enables models to refine their reasoning, retrieve visual evidence only when needed without training. Extensive experiments across various tasks show that DMLR consistently boosts both reasoning and perception tasks, offering a stable and training-free alternative to other methods.

> 💡 **全文回顾**:
>
> **起点**: 两个实证观察 —— (1) 视觉信息只在少数步骤关键，(2) 置信度是视觉需求和推理质量的天然信号
>
> **方法**: DMLR = 置信度引导的潜空间策略梯度优化 + 动态视觉注入
>
> **关键特性**: Training-free、测试时优化、推理与感知在潜空间中动态交错
>
> **验证**: 6 个模型 x 7 个 benchmark，95%+ 任务达到最优，同时保持高效率
>
> **定位**: 训练自由的潜空间多模态推理的新范式

---

## 三、Summary

- DMLR 为多模态推理提供了一个 training-free、高效、稳定的替代方案
- 核心观点：推理应该是动态、自适应的——在潜空间中按需交错推理与感知
- 局限/未来：论文未讨论（需从上下文推断），可能的局限包括：(1) 优化过程需要 eager attention mode，限制了一些高效 attention 后端的应用；(2) 15 步迭代仍有一定的额外计算成本；(3) 仅在开源模型上验证，未在闭源 API 模型上测试
