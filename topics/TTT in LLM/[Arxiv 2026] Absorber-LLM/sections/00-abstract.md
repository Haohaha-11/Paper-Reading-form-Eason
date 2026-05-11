[← 返回 README](../README.md)

# Abstract

## 📌 预览

本节给出总 claim：Absorber LLM 将 long-context retention 写成自监督的 causal synchronization，要求吸收历史后的 contextless model 在未来生成上匹配 full-context model。

Transformers suffer from a high computational cost that grows with sequence length for selfattention, making inference in long streams prohibited by memory consumption. Constantmemory alternatives such as RNNs and SSMs compress history into states with fixed size and thus lose long-tail dependencies, while methods that memorize contexts into parameters, such as Test-Time Training (TTT), are prone to overfitting token-level projection and fail to preserve the causal effect of context in pretrained LLMs. We propose Absorber LLM, which formulates longcontext retention as a self-supervised causal synchronization: after absorbing historical contexts into parameters, a contextless model should match the original model with full context on future generations. We optimize this objective by synchronizing internal behaviors of the updated model with the original one, ensuring context absorption and generalization. Experiments on long-context and streaming benchmarks show that Absorber LLM reduces inference memory and improves accuracy over prior parameter-as-memory baselines.

> 💡 **摘要批注**: 这里最关键的转向是“参数记忆写什么”。Absorber 不让模型重构 $X$，而是让 $W^*$ 在 $Y$ 上复现 $X$ 对 full-context 推理的影响；这就是后文 hidden-state synchronization 和 streaming absorption 的共同目标。

erate texts. Although variants like sparse (Beltagy et al., 2020) and linearized attention (Katharopoulos et al., 2020) mitigate this overhead, they merely delay the inevitable exhaustion of cache and computing resources in long-stream scenarios.

To resolve these memory constraints, prior works have proposed linear-time models, including Recurrent Neural Networks (RNNs) (Peng et al., 2023) and State Space Models (SSMs) (Gu et al., 2021; Gu & Dao, 2024). These models achieve efficiency by compressing history into fixed-size hidden states. However, recent theoretical and empirical analyses reveal that this efficiency comes at the cost of model expressive power, leading to significant information loss compared to full-attention transformers (Merrill et al., 2024; Hsieh et al., 2024).

Alternatively, Test-Time Training (TTT) (Sun et al., 2025; Feng et al., 2026) methods utilize model parameters for context storage to overcome state capacity limits. Specifically, given a sequence $X$ to be stored in the parameters of a model, TTT methods train the model to project $X$ into a specifically constructed target $V$ . Such methods optimize the model’s performance on $X$ by writing it directly into the model parameters. Yet, when learning $X$ , they ignore the model’s performance on subsequent sequences, failing to help the model absorb $X$ in a way that contributes to future inferences. Crucially, such reconstruction methods overlook the causal mechanisms between historical contexts and subsequent inferences, which are essential for effective model deduction (Geiger et al., 2024).

> 💡 **抽取边界批注**: 以上三段在 `full.md` 中位于 Abstract 和 Introduction 标题之间，语义上实际接续 Introduction 的长上下文动机。保留它们能看清论文的三段式定位：full attention 太贵，RNN/SSM 状态容量有限，现有 TTT 又过度依赖 token projection。

---

## 🔖 Section 总结

- **核心 claim**: 用 full-context oracle 的未来行为监督 contextless updated model。
- **关键对比**: full attention 保真但贵；RNN/SSM 高效但压缩为固定状态；传统 TTT 用参数记忆但目标未约束未来推理。
- **读法提醒**: 论文把“causal effect”落实为未来行为/隐藏状态同步，后面要看实验是否支撑这个目标优于 reconstruction。
