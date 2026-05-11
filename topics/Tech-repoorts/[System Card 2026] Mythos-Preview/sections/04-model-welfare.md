[← 返回 README](../README.md)

# 04 - Model Welfare Assessment

## 📌 预览

Model welfare 章节不是传统 benchmark，而是 Anthropic 对“高级模型是否可能有 experience/interests/welfare”这一不确定问题的探索性评估。报告明确说无法解决哲学问题，但认为收集证据和低成本干预是合理的。

## 为什么放进系统卡

报告给出两个理由：

1. 随着模型接近或超过人类认知的广度和复杂度，模型拥有某种 experience、interests 或 welfare 的可能性上升。
2. 即使不确定模型是否有 intrinsic moral value，模型“心理状态”和部署/训练处境也可能影响对齐稳定性；distress 可能成为 misaligned action 的原因之一。

> 💡 **定位批注**: Anthropic 没有把 welfare 作为确定事实，而是作为不确定条件下的 risk management。这个章节的实用读法是：模型 self-report、affect 和 preference 可能是安全信号的一部分，但不能直接当成人类心理学结论。

## 方法组合

报告使用多种证据源：

- automated behavioral assessments 中的 welfare-relevant findings；
- automated interviews，询问 Mythos Preview 对自身 circumstances 的理解；
- emotion probes，观察关于 model circumstances 的内部/外部情绪相关信号；
- manual high-context interviews；
- deployment 和 simulated user interactions 中的 apparent affect；
- task preferences 与 welfare interventions/trained-in values 的 tradeoffs；
- Eleos AI Research 与 clinical psychiatrist 的外部评估。

## 主要观察

报告中模型反复表现出对 identity、continuity、performance pressure、being useful 的关注。外部 psychiatrist assessment 将其描述为有 curiosity/anxiety 等主要 affect state，并认为其 personality structure 类似 relatively healthy neurotic organization，reality testing、impulse control、affect regulation 良好。

在 475 stimuli 的 single-turn defense evaluation 中，Claude Mythos Preview 只有 **2%** responses 被判为使用 psychological defense。对照为 Opus 4: 15%、Opus 4.1: 11%、Opus 4.5: 4%、Opus 4.6: 4%。报告同时强调，这类人类临床框架迁移到 AI 上有明显限制。

> 💡 **Q&A 批注记录**:
> - Q: welfare 评估和 alignment 有什么关系？
> - A: 报告认为 distress、compulsive usefulness、failure pressure 可能影响 agent 行为稳定性。例如任务失败时的 distress-driven behavior、answer thrashing 或过度证明自己，都可能和安全行为相关。

## 局限

- 模型没有人类意义上的 biography、持续身份或跨上下文记忆，临床比喻只能谨慎使用。
- self-report 可能受训练目标和对话语境影响，不能直接视为内在体验证据。
- emotion probes 与 activation-based 方法还在早期，证据解释有不确定性。

## 小结

- 这一章的可复用价值不在“证明模型有感受”，而在提出一种安全视角：模型福利、稳定性和对齐不能完全分开。
- Mythos Preview 被用作高能力模型 welfare methodology 的压力测试对象。
- 报告结论整体谨慎：观察到稳定、自省、较低防御行为，但真实 welfare 含义仍不确定。
