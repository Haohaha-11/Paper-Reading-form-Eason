[← 返回 README](../README.md)

# 5. Conclusion

In summary, our proposed dynamic framework enables stable and effective unsupervised, sample-specific TTA by learning step- and layer-wise update magnitudes. It consistently outperforms the fixed-rate baseline and the step-wise variant, which represents common hand-crafted learningrate schedules, suggesting that fine-grained, learned control is necessary.

> 💡 **结论批注**: 论文最强结论不是“prompt NLL 一定有用”，而是“当你已经要做 prompt-only LoRA TTA 时，固定 LR 很危险，layer-wise dynamic control 明显更稳”。它解决的是 TTA update allocation，而不是无监督目标本身的全部可靠性问题。

> 💡 **后续判断**: 下一步值得看跨任务泛化、延迟成本和更强 baseline。尤其需要比较 trust-region/clipping/gradient normalization 等无需训练控制器的方法，才能判断 SCALENET 的复杂度是否值得。
