[← 返回 README](../README.md)

# Abstract

> 💡 **📌 一句话预览**: 本文揭示长上下文LLM中静态自注意力的"分数稀释"现象，并从理论证明thinking tokens无法解决该问题，提出query-only test-time training (qTTT) 以少量梯度更新替代大量think tokens，在FLOP匹配下取得12.6/14.1 pp的显著提升。

Progress on training and architecture strategies have enabled LLMs with millions of tokens in context length. However, empirical evidence suggests that such long-context LLMs can consume far more text than they can reliably use. On the other hand, it has been shown that inference-time compute can be used to scale performance of LLMs, often by generating thinking tokens, on challenging tasks involving multi-step reasoning. Through controlled experiments on sandbox long-context tasks, we find that such inference-time strategies show rapid diminishing returns, and fail at long context. We attribute these failures to score dilution, a phenomenon inherent to static self-attention. Further, we show that current inference-time strategies cannot retrieve relevant long-context signals under certain conditions. We propose query-only test-time-training (qTTT) that, through targeted gradients updates on the given context, provably overcomes limitations of static self-attention. We find that this simple shift in how inference-time compute is spent leads to consistently large performance improvements across models and long-context benchmarks. qTTT leads to massive 12.6% and 14.1% points improvements for Qwen3-4B on average across subsets of LongBench-v2 and ZeroScrolls benchmarks. The takeaway is practical: for long context, a small amount of context-specific training is a better use of inference compute than current inference-time scaling strategies like producing more thinking tokens.

> 💡 **摘要批读**: 一句话抓住核心冲突——模型能"吃进"远超能"消化"的上下文。Score dilution是他们诊断的根因：随着distractor增多，target信号在softmax分母中被淹没。关键洞察是"thinking tokens用同样的静态attention生成更多token，本质上是在同一个失效机制上重复劳动"，而qTTT通过微调query投影直接改变了attention分配机制本身。

Correspondence: rachitbansal@g.harvard.edu, az@astonzhang.com

> 💡 **🔖 摘要小结**: 全文的核心论点已经在摘要中完整呈现：(1)静态注意力导致分数稀释，(2)thinking tokens无法根本解决，(3)qTTT通过query-only梯度更新直接提升margin，(4)实验结果支撑理论分析。这篇摘要本身就是一篇高质量mini-paper。

---
