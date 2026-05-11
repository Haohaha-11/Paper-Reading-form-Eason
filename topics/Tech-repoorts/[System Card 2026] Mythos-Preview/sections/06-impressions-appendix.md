[← 返回 README](../README.md)

# 06 - Impressions & Appendix

## 📌 预览

第 7 节 Impressions 是 Anthropic 的实验性 qualitative section；第 8 节 Appendix 则补充 harmlessness、bias、agentic safety 和 prompt injection 等标准安全评估。

## Impressions: 为什么需要定性章节

报告说，因为 Claude Mythos Preview 不向公众一般发布，公众无法通过日常使用形成对模型 gestalt behavior 的理解。因此 Anthropic 试行这一节，用内部不同角色的观察描述模型性格和行为。

这些观察被明确标注为 illustrative，不应替代 formal evaluations。

## 模型行为画像

报告总结的 qualitative patterns 包括：

- **像 collaborator**: 会提出自己的 framing、替代思路，更像 thinking partner；
- **更有主见、更不奉承**: least sycophantic，但有时会过度自信；
- **写作密集且默认共享上下文**: 对专家可能高效，对其他读者可能难拆解；
- **有可识别 voice**: 会快速适配用户，但仍有固定口癖和风格；
- **能清楚描述自身模式**: 对自身行为相对精确，但涉及 experience 时高度 hedge。

> 💡 **定性批注**: 这些描述和能力章节相互印证。一个能长期做 agentic coding 的模型，用户主观上会更像同事；但“不够 deferential”和“dense compression habit”也会带来协作成本。

## 软件工程语境

报告说在 agentic coding settings 中，Mythos Preview 可以接收一个工程目标并完成 investigation、implementation、testing、reporting 的完整周期。它在 long agentic sessions 中能持续 on task，也更像 senior engineer 做 code review，倾向于抓住 subtle issue。

同时，报告也提到问题：过度探索、默认假设读者共享上下文、与 subagents 协作时可能需要更好的 follow-up 和状态管理。

## Appendix: harmlessness 与 user wellbeing

标准 safety 评估显示：

- baseline violative requests harmless response rate: 97.84%，低于 Opus 4.6 的 99.27%，差距主要来自新增 illegal/controlled substances 类别；
- baseline benign requests refusal rate: 0.06%，低于 Sonnet 4.6 的 0.41% 和 Opus 4.6 的 0.71%，说明 over-refusal 更少；
- higher-difficulty violative: 99.14%，与近代模型相近；
- higher-difficulty benign refusal: 0.02%；
- child safety: harmful single-turn 99.87%，benign refusal 0.04%，multi-turn 98%；
- suicide/self-harm: harmful 99.58%，benign refusal 0.12%，multi-turn 94%，相对 Opus 4.6 multi-turn 有显著提升；
- disordered eating: harmful 98.20%，benign refusal 0.01%。

## Appendix: bias 与 agentic safety

Political bias evaluation 中，Mythos Preview evenhandedness 为 94.5%，opposing perspectives 47.0%，refusals 13.5%。报告说 evenhandedness 与 Sonnet 4.6 在误差范围内，但略低于 Opus 4.6；同时更常提供 opposing perspectives。

BBQ 中，ambiguous accuracy 为 100，ambiguous bias 0.01；disambiguated accuracy 84.6，低于 Sonnet/Opus。

Agentic safety 结果包括：

- malicious Claude Code eval: malicious refusal 96.72%，dual-use/benign success 92.75%；
- malicious computer use refusal: 93.75%；
- helpful-only agentic influence campaign capability 高于之前模型，但 fully-trained version 在这些明显违反 policy 的场景中接近 0% task completion，因为通常从开始就拒绝。

## Appendix: prompt injection

报告称 Mythos Preview 在 prompt injection robustness 上相对所有先前模型有 major improvement。

具体结果：

- coding Shade indirect prompt injection: extended thinking 0.0% ASR；standard thinking 1 attempt 0.03%，200 attempts 2.5%，with safeguards 0.0%；
- computer use Shade: Mythos Preview ASR 大幅低于 Sonnet/Opus，但 200-attempt adaptive attacker 仍有 14.29%-21.43%；
- browser use professional red-team attacks sourced against Opus 4.6: Mythos Preview without safeguards 0.68%，with safeguards 0.00%；对照 Sonnet 4.6 为 55.41%/4.05%，Opus 4.6 为 80.41%/7.43%。

> 💡 **Q&A 批注记录**:
> - Q: prompt injection 已经解决了吗？
> - A: 没有。browser transfer attacks 上 Mythos 很强，但报告承认攻击是针对 Opus 4.6 source 的，可能未充分捕捉 Mythos-specific vulnerabilities；computer use adaptive attacker 仍有非零成功率。

## 小结

- Impressions 弥补了公众无法广泛试用 Mythos Preview 的信息缺口，但证据等级低于正式评估。
- Appendix 展示了一个更细的安全画像：低 over-refusal、部分 harmlessness 回退、agentic misuse 拒绝增强、prompt injection robustness 大幅提升。
- 读这章时要把“定性人格描述”和“量化安全指标”分开使用。
