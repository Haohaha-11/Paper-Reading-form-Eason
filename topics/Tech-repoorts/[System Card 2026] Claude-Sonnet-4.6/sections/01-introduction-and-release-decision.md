[← 返回 README](../README.md)

# 01 - Introduction and Release Decision

## 📌 预览

Introduction 给出模型来源、训练数据、thinking modes、评估范围和发布判定。最重要的线索是：Sonnet 4.6 不被 Anthropic 视为相对 Opus 4.6 的 frontier advance，因此采用 preliminary assessment；但它足够强，仍按 ASL-3 保护。

## 模型训练与特性

- 训练数据包括公开互联网信息，截止到 May 2025。
- 还包括第三方非公开数据、数据标注服务和承包商数据、opt-in Claude 用户数据、Anthropic 内部生成数据。
- 训练过程包括 data cleaning、filtering、deduplication、classification。
- pretraining 后进行 substantial post-training 和 fine-tuning，目标是 helpful、honest、harmless。

> 💡 **训练数据批读**: 这段披露的是来源类别而不是数据集清单。对可复现性没有帮助，但对合规/数据治理有帮助：它明确 opt-in 用户数据和内部生成数据是训练混合的一部分。

## Thinking modes

Sonnet 4.6 支持两类推理控制：

| 模式 | 含义 | 读法 |
|---|---|---|
| Extended thinking | 允许模型花更多时间推理 | 适合难题，但不总是提升所有 benchmark |
| Adaptive thinking / effort parameter | 模型按上下文决定多用或少用 extended thinking | benchmark 里常见 max/high effort 设置 |

> 💡 **机制批读**: 报告里多个结果依赖 effort 设置，例如 SWE-bench、MMMU-Pro、GMMLU 使用 adaptive thinking + max effort；BrowseComp 反而说明 thinking disabled 更好。这提醒读者不要把“模型能力”与“默认产品体验”混成一件事。

## Release decision process

Anthropic 对 Sonnet 4.6 进行了训练过程中的 iterative evaluations，覆盖：

- 多个 helpful/honest/harmless snapshots；
- 一个 helpful-only snapshot；
- final release candidate；
- agentic evaluations 中每个 snapshot 多次采样。

危险能力评估采取保守做法：把任意 snapshot 达到的高分纳入最终 capability assessment。

> 💡 **评估策略批读**: 取多个 snapshot 的最高危险能力分数，会让 RSP 判定更偏保守。这比只报告 deployed model 更能估计“训练路线可能达到的能力上限”。

## ASL 判定

核心结论：

- Sonnet 4.6 自动评估表现低于或等于 Opus 4.6；Opus 4.6 已按 ASL-3 safeguards 部署。
- 因此 Sonnet 4.6 没有把总体能力前沿推过 Opus 4.6，也按 ASL-3 部署。
- 但 Sonnet 4.6 crossed most rule-out thresholds used as early proxies for AI R&D-4。
- Anthropic 仍不认为它 fully qualifies for AI R&D-4。

> 💡 **Q&A 批注记录**:
> - Q: 为什么没有跨 AI R&D-4，却仍然主动上 AI R&D-4 相关 mitigations？
> - A: 因为报告承认灰区存在：模型接近多个 proxy threshold，而 AI R&D-4 本身需要判断“能否完全自动化 Anthropic entry-level remote-only Researcher 工作”，这个阈值难以单靠 benchmark 干净排除。

## Sabotage / CBRN / Cyber 结论

| 风险域 | 结论 | 关键理由 |
|---|---|---|
| Sabotage | 未准备 Sonnet 4.6 独立 comprehensive risk report | 不推进 Opus 4.6 之外的 sabotage-relevant capability frontier |
| AI R&D-4 | 未判定跨过 | 能力整体低于 Opus 4.6；但 scaffolding 可能使模型接近阈值 |
| CBRN-4 | 未判定跨过 | CBRN 评估低于此前模型，尤其 short-horizon computational biology 未跨 ASL-4 rule-out |
| Cyber | 无正式 RSP threshold | 当前 cyber eval 接近饱和，需更难评估与监控 |

## 🔖 Section 总结

Introduction 的重点不是模型架构，而是发布治理：Sonnet 4.6 的能力足以要求 ASL-3，但未被认定为超越 Opus 4.6 的新 frontier model。读者后续看所有 benchmark，都应把它们放回这个判定框架中：能力越接近阈值，越依赖 scaffolding、工具和评估设计。
