[← 返回 README](../README.md)

## 📌 预览
Limitations 点出 latent token 的核心风险：性能提升可见，但可解释性弱，未来需要更大模型、更大数据和更高 latent capacity 验证。

---

# 6. Limitations and Future Work

We introduce a simple and effective method for improving visual reasoning in LMMs using latent tokens. A key limitation is that latent tokens may be less interpretable than textual explanations. Future work includes scaling to larger models, increasing latent capacity, and training on larger datasets. Overall, this work highlights a new avenue for enhancing visual reasoning via architectural inductive biases rather than explicit supervision, paving the way towards more capable LMMs.

> 💡 **局限批读**: 最大风险是 latent token 不可读。性能和 attention 能说明它们有用，但不能像文本 CoT 那样直接审计推理是否忠实；未来需要 latent-to-text/latent intervention 类分析。

---

## 🔖 Section 总结

> 💡 **Section 小结**:
> - 局限: latent token 不可读，且实验还需要更大模型、更大数据和更高 latent capacity 验证。
> - 后续方向: latent 可解释性、动态 K、跨任务共享 latent、因果干预分析。
