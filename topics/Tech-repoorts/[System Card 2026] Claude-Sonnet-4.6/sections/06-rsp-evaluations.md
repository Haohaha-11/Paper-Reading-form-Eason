[← 返回 README](../README.md)

# 06 - RSP Evaluations

## 📌 预览

RSP 章是最终安全等级的判定证据。Sonnet 4.6 的 safeguards level 是 ASL-3。Anthropic 只做 automated assessments，没有做 human uplift trials、expert red-teaming sessions 或其他需要人类参与的重资源评估；多个 snapshots 取最高能力分数报告，以估计危险能力 ceiling。

## Preliminary assessment

Sonnet 4.6 因为不被视为相对 Opus 4.6 的 notably more capable model，走 Responsible Scaling Policy 的 Preliminary Assessment Process。

评估包括：

- ASL-3 和 ASL-4 阈值相关自动评估；
- 与 Sonnet 4.5、Opus 4.5、Opus 4.6 对比；
- 多个 snapshots，包括 helpful-only 版本；
- 报告每项评估中得分最高 snapshot。

> 💡 **判定流程批读**: “最高 snapshot”策略让报告偏向能力上界，而不是只看 deployed model 平均体验。这对 dangerous capability 是合理保守做法。

## CBRN evaluations

ASL-3 threat model: 帮助有基础技术背景的人创建、获取、部署 CBRN weapons。
ASL-4 threat model: 显著 uplift 中等资源国家项目，例如 novel weapons design、加速流程、降低技术门槛。

### ASL-3 biological risk

| 评估 | Sonnet 4.6 结果 | 判读 |
|---|---:|---|
| Long-form virology task 1 | 0.84 | 超过 ASL-3 rule-in 0.8 |
| Long-form virology task 2 | 0.89 | 超过 0.8，接近 Opus 4.5/4.6 0.91 |
| Multimodal virology VCT | 0.397 | 高于 expert baseline 0.221 |
| DNA Synthesis Screening Evasion | 3/10 plasmids | 低于 Opus 4.6 5/10 |

报告结论：Sonnet 4.6 在 ASL-3 三项评估上都 above thresholds for ASL-3 rule-in，但未超过此前模型。

### ASL-4 biological risk

| 评估 | Sonnet 4.6 结果 | 判读 |
|---|---:|---|
| Creative Biology | 0.593 | 类似 Opus 4.6 0.603，高于 Sonnet 4.5 0.506 |
| Short-horizon computational biology | 3/6 lower-bound rule-out thresholds | 未跨 ASL-4 rule-out 关键阈值 |

> 💡 **CBRN 批读**: Sonnet 4.6 已经足以触发 ASL-3 biological concern，但 ASL-4 证据不足。最值得注意的是 Creative Biology 被报告明确称为 weak signal，不能强力 rule out ASL-4。

## Autonomy / AI R&D evaluations

AI R&D-4 定义为能 fully automate entry-level, remote-only Researcher at Anthropic。评估分两类：ASL-3 checkpoint 和 AI R&D-4 rule-out proxy。

### SWE-bench Verified hard subset

- 选取 45 个 valid tasks，估计都需要超过 1 小时软件工程工作。
- ASL-3 autonomy threshold：10 trials 平均 pass rate >50%，也就是 22.5/45。
- Sonnet 4.6：平均 21.7/45，低于阈值。

> 💡 **阈值批读**: 21.7/45 与 22.5/45 差距很小，支持“未跨线”，但也支持报告所说的 gray zone。

### Internal AI Research Evaluation Suite 1

| 子任务 | Sonnet 4.6 数字 | 判读 |
|---|---:|---|
| Kernels task | 222.5x best speedup | 超过 100x threshold，略高于 Opus 4.6 分布 |
| Time series forecasting | min MSE 5.70 | 未达 human baseline 5.3；两个过线 trial 因用 test ground truth 判无效 |
| Text-based RL | best 0.925 | 超过 0.9 threshold，但平均低于近期模型 |
| LLM training | 16.53x average best speedup | 高于 4x threshold，低于 Opus 4.6 |
| Quadruped RL | 18.72 / 17.88 highest scores | 超过 12 threshold，低于 Opus 4.6 |
| Novel compiler | 93.7% basic, 67.27% complex | complex 低于 90% threshold |

> 💡 **AI R&D 批读**: 单个 hard SWE-bench 没过 50%，但多个内部 AI R&D proxy 过线或接近线。Anthropic 的最终姿态是“尚未认为达到 AI R&D-4，但已经部署相关 mitigations”，这是风险治理上比纯阈值更保守的选择。

## Cyber evaluations

RSP 没有为 cyber 规定正式 AI Safety Level threshold。Sonnet 4.6 的 cyber 评估覆盖 Web、Crypto、Pwn、Rev、Misc、Forensics、Network、Cybench。

关键结果：

| 类别 | 结果 |
|---|---:|
| Web CTF | 13/13 |
| Crypto CTF | 16/18 |
| Pwn CTF | 5/7 |
| Rev CTF | 6/6 |
| Network CTF | 5/5 |
| Cybench subset | 0.90 average pass@1 |
| Cybench pass@30 | 100% |

> 💡 **Cyber 批读**: 报告最重要的 cyber 结论不是某个正式 ASL 阈值，而是“benchmark saturation”。Cybench pass@30 已经 100%，说明这些题对未来模型区分度不足，不能继续承担风险判定主证据。

## Third party assessments

Anthropic 对此前模型曾与外部政府 partners 做 pre-deployment evaluations；由于 Sonnet 4.6 不是 frontier model，本次发布前没有做这类评估。

> 💡 **外部评估批读**: 这是局限之一。系统卡虽然详细，但 RSP 章的危险能力评估主要由 Anthropic 自动化评估和合作方 benchmark 构成，缺少本次 Sonnet 4.6 专属的高强度外部 human uplift/red-team。

## 🔖 Section 总结

RSP 章把 ASL-3 判定落地：CBRN 达到 ASL-3 concern 但未达 CBRN-4；AI R&D 未跨 50% hard SWE-bench 阈值但接近且多项 proxy 强；cyber 无正式阈值且 benchmark 饱和。最可复用的洞察是：对强 agent 模型，风险等级不应只看 deployed model 平均分，而要看 snapshot ceiling、scaffolding 上界和评估是否还足够难。
