[← 返回 README](../README.md)

# Abstract

Test-time adaptation (TTA) for large language models (LLMs) updates model parameters at inference time using signals available at deployment. This paper focuses on a common yet underexplored regime: unsupervised, sample-specific TTA, where the model adapts independently for each prompt using only the prompt itself, without gold answers or external supervision. Although appealing, na¨ıve unsupervised TTA with a fixed, handcrafted learning rate can be unstable: updates may overfit to prompt-specific statistics, drift from the desired answer distribution, and ultimately degrade generation quality. This failure mode is not surprising, as in this case TTA must adapt to a single prompt within only a few gradient steps, unlike standard training that averages updates over large datasets and long optimization horizons. Therefore, we propose layer-wise dynamic test-time adaptation, a framework which explicitly modulates TTA strength as a function of prompt representation, LLM structure and adaptation step. In our setting, TTA updates only LoRA parameters, and a lightweight hypernetwork predicts per-layer, per-step learning-rate multipliers, enabling fine-grained control. Experiments across various datasets and LLMs consistently show that our method substantially strengthens TTA by learning effective scaling patterns over adaptation steps and transformer layer projections, improving stability while delivering better performance.

> 💡 **批注**: 摘要的关键词不是普通 TTA，而是 **unsupervised + sample-specific + adapt-and-reset**。这意味着每个 prompt 只有自己的 token 可作为 input perplexity / NLL 信号，梯度高方差、步数很少；固定 LR 失稳不是工程偶然，而是这个设定的核心矛盾。

> 💡 **方法判断**: 本文没有改变无监督 TTA 的 proxy loss，仍然用 prompt 自身 NLL；真正新增的是对 LoRA update 的控制层。SCALENET 学的是“这一步、这一层、这个 Q/V LoRA 该更新多猛”，不是直接预测答案或 LoRA 权重。

> 💡 **读法提示**: 后文实验要看两个问题：一是 fixed LR 是否真的多步崩，二是 layer-wise 是否显著强于 step-wise。只有同时成立，才能证明“层级动态控制”不是普通 LR schedule 的包装。
