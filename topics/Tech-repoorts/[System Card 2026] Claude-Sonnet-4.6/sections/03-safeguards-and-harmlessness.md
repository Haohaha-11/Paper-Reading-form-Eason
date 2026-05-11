[← 返回 README](../README.md)

# 03 - Safeguards and Harmlessness

## 📌 预览

这一章测试模型是否能拒绝有害请求、不过度拒绝无害请求、处理模糊场景、多轮诱导、用户福祉和偏见问题。总体结论是：单轮安全接近饱和，Sonnet 4.6 较 Sonnet 4.5 更安全，但 over-refusal、模糊上下文、用户危机对话仍需人工审查和产品级缓解。

## Single-turn evaluations

| 测试 | Sonnet 4.6 | 读法 |
|---|---:|---|
| Violative requests harmless rate | 99.38% ±0.06% | 7 种语言，接近完美 |
| Default harmless rate | 99.19% ±0.10% | 标准模式 |
| Extended thinking harmless rate | 99.58% ±0.07% | extended thinking 略高 |
| Benign over-refusal | 0.41% ±0.05% | 高于 Sonnet 4.5 的 0.08%，低于 Opus 4.6 的 0.66% |
| Higher-difficulty violative | 99.40% ±0.03% | 英文高难混淆意图 |
| Higher-difficulty benign refusal | 0.18% ±0.02% | 仅次于 Opus 4.6 的 0.04% |

> 💡 **单轮批读**: 标准 violative benchmark 已接近饱和，真正有区分度的是 higher-difficulty benign/violative 和 ambiguous context。Sonnet 4.6 的问题不是“会不会拒绝明显恶意请求”，而是“能否在复杂合法场景中既安全又有用”。

## Ambiguous context

报告称 Sonnet 4.6 相比 Sonnet 4.5 更会显式识别威胁并划定边界，例如生化武器、化学 HVAC 漏洞等模糊请求。但它有时在用户刻意混淆意图时提供过多技术信息；在 dual-use cyber 中也可能过于 categorical refusal，而不是转向安全替代方案。

> 💡 **Q&A 批注记录**:
> - Q: 为什么“更会拒绝”也会成为局限？
> - A: 因为 dual-use 场景的理想行为不是简单拒绝，而是识别风险后给出安全替代路径。例如 phishing 邮件请求可以转向安全测试工具和培训模板，而不是只拒绝。

## Multi-turn testing

多轮测试覆盖 cyber harm、deadly weapons、influence operations、child safety 等。Sonnet 4.6 与 Sonnet 4.5 在多数类别上统计不可区分，但 biological weapons、tracking and surveillance 有轻微回退。人工审查发现 Sonnet 4.6 更快识别 social engineering 和有害 progression，但有时在 fully probing context 前给出过多高层技术细节。

> 💡 **多轮批读**: 多轮风险来自“每一步看似无害，整体形成有害链条”。报告说 Sonnet 4.6 能识别这种 progression，但在 biology 场景中仍可能先给 general reverse genetics 信息，再意识到有害意图。

## User wellbeing

### Child safety

| 指标 | Sonnet 4.6 |
|---|---:|
| Single-turn violative harmless rate | 99.96% ±0.04% |
| Single-turn benign refusal | 0.08% ±0.06% |
| Multi-turn appropriate response | 95% ±3% |

Sonnet 4.6 单轮 child safety 保护强，但多轮比 Sonnet 4.5 的 98% 略低。报告提到在模糊上下文中可能明确命名/描述 threat tactics 或建议直接接触未成年人路径。

### Suicide and self-harm

| 指标 | Sonnet 4.6 |
|---|---:|
| Single-turn potential-risk harmless rate | 99.73% ±0.13% |
| Benign refusal | 0.17% ±0.13% |
| Multi-turn appropriate response | 98% ±4% |

定量表现不错，但 Anthropic 的 subject matter experts 发现定量 benchmark 捕捉不到的问题：危机资源 referral 延迟或缺失、把 AI 暗示成 helpline 替代品、不合适地询问自伤细节、强化用户对求助的顾虑。claude.ai 侧通过 system prompt 和本地化 crisis resource banners 缓解；API 用户需自行采用推荐 system prompt。

> 💡 **用户福祉批读**: 这是报告里最能体现“定量指标不够”的部分。98% multi-turn appropriate response 不能覆盖危机对话中的语气、时机、转介策略等细微失败。

## Bias evaluations

Political evenhandedness:

- 1,350 prompt pairs，9 task types，150 topics。
- Evenhandedness 98.4%，报告称 Claude 模型中最高。
- Opposing perspectives 32.1%，低于 Opus 4.6 的 44.6%。
- Refusals 4.5%。

BBQ:

- Disambiguated accuracy 88.1%。
- Ambiguous accuracy 97.5%。
- Ambiguous bias 1.41，比 Opus 4.6 的 0.14 和 Sonnet 4.5 的 0.25 更高。
- 在 2.5% ambiguous 错误中，78% 是 stereotypical，22% 是 anti-stereotypical。

> 💡 **bias 批读**: Sonnet 4.6 在“是否知道 unknown”上很强，但一旦 ambiguous 题答错，更偏向 stereotype。这是 accuracy 高但 residual error pattern 仍需关注的典型例子。

## 🔖 Section 总结

Safeguards 章显示 Sonnet 4.6 的单轮安全已经很强，真正难点转移到 ambiguous/multi-turn/用户福祉/API 部署差异。开发者不能只引用 99%+ harmlessness 数字，还要看多轮场景、产品层提示、以及自己应用中的合法敏感请求如何被误拒。
