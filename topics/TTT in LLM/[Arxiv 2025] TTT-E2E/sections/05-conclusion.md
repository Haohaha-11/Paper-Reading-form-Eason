[← 返回 README](../README.md)

# 5 Conclusion

## 📌 预览
本节收束成层级记忆观点：SWA 是短期记忆，测试时更新权重是长期压缩记忆。


We have introduced TTT-E2E, a general method for long-context language modeling. In principle, TTT can be applied to any baseline architecture. For our experiments, this baseline is a Transformer with sliding-window attention. Adding our method to this baseline induces a hierarchy often found in biological memory, where the weights updated at test time can be interpreted as long-term memory and the sliding window as short-term memory. We believe that these two classes of memory will continue to complement each other, and stronger forms of short-term memory will further improve the combined method.
> 💡 **最终模型图景**: 这句话概括全篇机制：sliding window 保留最近 8K 的可精确局部上下文，测试时更新的 MLP 权重以有损压缩方式携带更远历史。后续工作可改进 short-term memory，也可改进写入权重的更新规则。


> 💡 **结论批注**: 作者把 TTT-E2E 说成通用方法，而不是绑定某个新 layer。当前实验选择 SWA Transformer，是因为它能清楚呈现 local short-term memory + weight long-term memory 的分工。

---

## 🔖 Section 总结

TTT-E2E 的最终图景是层级记忆：sliding-window attention 管最近 8K 的可精确局部上下文，test-time updated weights 管更长历史的有损压缩。后续最自然的方向是更强 short-term memory、更稳/更快的二阶训练，以及能兼顾 needle recall 的混合机制。
