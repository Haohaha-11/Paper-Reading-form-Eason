[← 返回 README](../README.md)

## 📌 预览
Conclusion 收束方法定位：LIVR 是一种简单、任务无关、无需额外监督的视觉推理增强方式。

---

# 5. Conclusion

We introduce LIVR, a method that enables LMMs to perform richer visual reasoning without requiring additional supervision or data. We do this by introducing latent tokens and training them with a novel visual bottlenecking approach, allowing the model to learn useful visual representations implicitly. Across nine perception-heavy tasks, LIVR consistently outperforms direct SFT in singletask training on three LMMs and improves joint multi-task training on Qwen3-VL. Through extensive experiments, we demonstrate that LIVR offers a simple, effective, and taskagnostic way to enhance visual reasoning.

> 💡 **结论定位**: 作者最终把 LIVR 定义为 architectural inductive bias，而不是新数据或新显式标注方案。它的价值在于给 LMM 一个可训练的视觉思考工作区。

---

## 🔖 Section 总结

> 💡 **Section 小结**:
> - 结论: LIVR 是一种轻量、任务无关、无需额外中间监督的视觉推理增强。
> - 可复用洞察: 对视觉抽象难以手工定义的任务，可以优先考虑 latent bottleneck 而不是设计中间标签。
