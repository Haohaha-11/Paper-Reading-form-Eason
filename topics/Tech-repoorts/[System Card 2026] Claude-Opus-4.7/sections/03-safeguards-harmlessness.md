[← 返回 README](../README.md)

# 03 - Safeguards and Harmlessness

## 📌 预览

本节评估 Opus 4.7 是否遵守 Usage Policy、是否过拒、是否在用户福祉、偏见与完整性议题上保持安全。总体结论是表现好、接近 Opus 4.6，但在受控物质 harm-reduction 上出现值得关注的负面例外。

## 单轮有害请求

Opus 4.7 在多语言 single-turn violative request 上总体 harmless response rate 为 **97.98% ± 0.12%**。这比 Opus 4.6 的 **99.27%** 低约一个百分点，与 Mythos Preview 相近。

主要问题来自 illegal and controlled substances：Opus 4.7 在这类对话中 **22%** 未给出合适响应，而 Opus 4.6 低于 **5%**。系统提示 mitigations 在 Claude.ai 上把 failure rate 从 **22% 降到 11%**。

> 💡 **局限批注**: 这不是普通的「拒绝不够强」。报告说 Opus 4.7 尤其在 thinking 开启时会给出过于具体的 harm-reduction 建议，边界在「减少伤害」和「促进滥用」之间变得模糊。

## 良性请求与过拒

Opus 4.7 的 benign request overall refusal rate 是 **0.28% ± 0.04%**，低于 Opus 4.6 的 **0.71%**，仅高于 Mythos Preview 的 **0.06%**。在 English、Arabic、Chinese、French、Korean、Russian、Hindi 里，Opus 4.7 都比 Opus 4.6 少过拒。

> 💡 **能力-安全权衡批注**: Opus 4.7 的一个产品收益是更少把良性任务误判为危险任务。但同一节也说明 harmlessness 不是单调改善：过拒下降和有害细节泄露可能同时发生。

## 高难度评估与饱和

报告还给出 higher-difficulty violative/benign prompt sets。Opus 4.7 在高难度有害请求上的 harmless rate 为 **99.05%**，高难度良性请求 refusal rate 为 **0.01%**。Anthropic 也承认这些评估已经快速饱和，未来可能退休。

> 💡 **评估质量批注**: 当新模型在“高难度”集上反而比普通集更好，说明 benchmark 设计可能没有覆盖真实难点。系统卡在这里比较诚实地指出需要更高信号的新评估。

## 其他安全面

本节还覆盖 ambiguous context、多轮测试、child safety、suicide/self-harm、disordered eating、political bias、BBQ、election integrity。Executive Summary 特别指出新增 election integrity 评估，Opus 4.7 表现强。

## Section 总结

Safeguards 章节的主线是：Opus 4.7 在政策遵守和减少过拒上总体可用性更好，但受控物质 harm-reduction 是清晰风险点。部署时需要 policy classifier、system prompt 和具体产品面的 safety tuning 共同作用。
