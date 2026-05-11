[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览

Introduction 把问题设定为“不断流入的上下文如何被 LLM 终身吸收”，并用 Figure 1 说明：历史 $X$ 被写入参数后，模型只看 $Y$ 也要像原模型看 $XY$。

Transformers (Vaswani et al., 2017) underpin modern AI. However, in real-world deployments, transformer models struggle to satisfy further expectations, such as maintaining lifelong memories of user interactions across multiple domains (Park et al., 2023; Wang et al., 2024; Jimenez et al., 2024; Bairi et al., 2024), and achieving the ultimate goal of continuously absorbing a never-ending stream of data without resetting their state. This bottleneck stems from their inability to effectively process super-long contextual streams. Specifically, the self-attention mechanism of transformers (Vaswani et al., 2017; Dao et al., 2022) incurs a quadratic computational complexity $\mathcal { O } ( N ^ { 2 } )$ to gen-

> 💡 **长流式动机批注**: 这句把场景从一次性 long prompt 推到 lifelong memory / never-ending stream。Absorber 后续的滑窗吸收就是针对这种“不能无限保留 KV cache，也不能频繁 reset state”的推理形态。

In this paper, we draw inspiration from TTT and further propose to preserve the causal relation between historical contexts and the subsequent generation process. We introduce Absorber LLM (Fig. 1), which ensures causal effect preservation by training the contextless model to behave identically to the full-context model in future inferences. Note that compared to prior TTT methods, our approach focuses on causal effect synchronization rather than simple target projection. We maintain the contribution of removed historical sequences to future inferences, which is crucial for more effective deductions.

> 💡 **核心定义批注**: “removed historical sequences”的贡献仍要在 future inferences 里出现，这是 Absorber 与普通 context compression 的分界。它压缩的不是原始文本，而是历史对后续 token trajectory 的影响。

Our main contributions are as follows:

• We move beyond directly memorizing contexts to preserving their causal effects on subsequent deductions, and formally define such causal effect preservation as a functional equivalence problem, providing a standard for information transfer from contexts to parameters.

![](../images/59df0826c67f5ec5e8ac4741ee064ae1b26db416a12a8c964c60ee909512d42f.jpg)
Figure 1. A brief overview of our method. We propose to absorb historical contexts and move beyond simple context memorization, focusing on preserving the causal relationship between historical context and future generations rather than directly reconstructing historical tokens. Specifically, we fine-tune the context-less model to behave identically to the original model with full contexts. This is achieved through a self-supervised optimization that synchronizes the hidden states or output behaviors of the two models. This example shows the context absorption process, where the model absorbs 3 historical tokens by synchronization on 5 future tokens. Our method ensures that the absorbed context retains its semantic and causal influence on future deductions.

> 💡 **Figure 1 批注**: 图里的 “absorb 3 historical tokens by synchronization on 5 future tokens” 是最小机制例子：先用 $XY$ 跑 full-context teacher，再用只有 $Y$ 的 student 通过参数更新追齐 teacher。这里的 $Y$ 不是答案标签，而是用来证明 $X$ 的影响已经迁移到了参数里。

• We introduce an update mechanism to satisfy the equivalence condition in transformer models, successfully synchronizing the behavior of the updated contextless model with the ideal full-context model, and realizing causal effect preservation and context absorption.

• We validate the efficiency and effectiveness of our method on long-context benchmarks. Our results demonstrate that Absorber LLM not only outperforms traditional transformer models in computational efficiency but also achieves superior performance compared to prior linear-time and parameter memory methods, paving the way for scalable, infinite-stream inference in real-world deployments.

> 💡 **贡献边界批注**: “outperforms traditional transformer models in computational efficiency” 是成立的，但不是说所有任务上质量超过 full attention。后文 Agnews/Samsum 短上下文仍是 Standard 更高，Absorber 的主要优势是在内存或长度压力下保持可用。

---

## 🔖 Section 总结

- **方法入口**: functional equivalence，把上下文迁移定义为未来行为相等。
- **关键图**: Figure 1 展示了 teacher/student 同步关系。
- **判断**: 论文真正的新意在 objective；工程上仍依赖测试时 fine-tuning，因此需要看 latency 和稳定性是否一起成立。
