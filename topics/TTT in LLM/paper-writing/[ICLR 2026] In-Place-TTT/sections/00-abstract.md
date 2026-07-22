[← 返回 README](../README.md)

## 📌 预览
摘要定位 In-Place TTT 的核心机制与实验主张。

---

# Abstract

The static “train then deploy" paradigm fundamentally limits Large Language Models (LLMs) from dynamically adapting their weights in response to continuous streams of new information inherent in real-world tasks. Test-Time Training (TTT) offers a compelling alternative by updating a subset of model parameters (fast weights) at inference time, yet its potential in the current LLM ecosystem is hindered by critical barriers including architectural incompatibility, computational inefficiency and misaligned fast weight objectives for language modeling. In this work, we introduce In-Place Test-Time Training (In-Place TTT), a framework that seamlessly endows LLMs with Test-Time Training ability. In-Place TTT treats the final projection matrix of the ubiquitous MLP blocks as its adaptable fast weights, enabling a “drop-in" enhancement for LLMs without costly retraining from scratch. Furthermore, we replace TTT’s generic reconstruction objective with a tailored, theoretically-grounded objective explicitly aligned with the Next-Token-Prediction task governing autoregressive language modeling. This principled objective, combined with an efficient chunk-wise update mechanism, results in a highly scalable algorithm compatible with context parallelism. Extensive experiments validate our framework’s effectiveness: as an in-place enhancement, it enables a 4B-parameter model to achieve superior performance on tasks with contexts up to 128k, and when pretrained from scratch, it consistently outperforms competitive TTT-related approaches. Ablation study results further provide deeper insights on our design choices. Collectively, our results establish In-Place TTT as a promising step towards a paradigm of continual learning in LLMs.

> 💡 **摘要抓手**: 这篇论文的核心不是“再发明一个 TTT 层”，而是把现有 Transformer MLP 的 final projection matrix 当成 fast weights，在推理时原位更新，因此能作为预训练 LLM 的 drop-in enhancement。

Correspondence: Di He at dihe@pku.edu.cn and Wenhao Huang at huang.wenhao@bytedance.com Code: https://github.com/ByteDance-Seed/In-Place-TTT

---

## 🔖 Section 总结

> 💡 **Section 小结**: In-Place TTT 是把现有 MLP final projection 变成推理时可更新 fast weights 的 drop-in TTT 框架，目标和实现都围绕 autoregressive NTP 与可并行 chunk update 设计。
