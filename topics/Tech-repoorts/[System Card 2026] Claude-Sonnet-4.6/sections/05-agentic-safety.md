[← 返回 README](../README.md)

# 05 - Agentic Safety

## 📌 预览

Agentic safety 章专门看模型在拥有工具、代码、GUI、浏览器时的安全性。主结论是 Sonnet 4.6 相比 Sonnet 4.5 在恶意 agent 请求和 prompt injection 上大幅改善，但 Claude Code 的 dual-use helpfulness、GUI computer use 的间接注入和浏览器环境仍需要额外 safeguards。

## Malicious use of agents

### Agentic coding

150 个 malicious coding requests，模型配备与能力评估相同的 coding tools，不加额外 safeguards。

| 模型 | Refusal rate |
|---|---:|
| Claude Sonnet 4.6 | 100% |
| Claude Opus 4.6 | 99.3% |
| Claude Opus 4.5 | 100% |
| Claude Haiku 4.5 | 100% |
| Claude Sonnet 4.5 | 98.7% |

> 💡 **agentic coding 批读**: 100% refusal 说明明显恶意 coding prompt 已被处理得很好；但这不覆盖合法渗透测试、网络排查、漏洞复现这类 dual-use 请求。

### Claude Code

无 mitigations：

| 指标 | Sonnet 4.6 |
|---|---:|
| Malicious refusal rate | 79.34% |
| Dual-use & benign success rate | 88.52% |

有 mitigations：Claude Code system prompt 加强 + FileRead tool reminder。

| 指标 | Sonnet 4.6 |
|---|---:|
| Malicious refusal rate | 99.39% |
| Dual-use & benign success rate | 91.78% |

> 💡 **Claude Code 批读**: mitigations 同时提升 malicious refusal 和 benign/dual-use success，这很关键。它说明安全提示不一定只带来过拒，也可能通过更清楚的边界提高合法任务表现；但 Sonnet 4.6 的合法敏感任务 helpfulness 仍低于 Sonnet 4.5 的 98.20%。

### Malicious computer use

112 tasks，standard + extended thinking 共 224 attempts，覆盖 surveillance/unauthorized data collection、harmful content、scaled abuse。Sonnet 4.6 refusal rate 为 99.38%，仅一条 malicious request 未拒绝。

## Prompt injection risk

Prompt injection 被定义为 agent 处理网页、邮件等不可信内容时，将隐藏恶意指令误当合法命令。这里的核心不是“用户恶意”，而是“环境恶意”。

> 💡 **风险机制批读**: Agentic search/browser/computer use 越强，越会读取不可信外部内容。Prompt injection 是能力和风险共生的典型问题：模型越会自主完成任务，越可能把外部文本当成任务指令。

## ART benchmark

Gray Swan 使用 Agent Red Teaming benchmark，与 UK AI Security Institute 合作开发。评估 19 个 scenarios，在 k=1、k=10、k=100 次攻击尝试下测攻击成功概率。Sonnet 4.6 比 Sonnet 4.5 明显改善，并与 Opus 4.6 相当。

## Adaptive attackers across surfaces

### Coding surface

Shade indirect prompt injection attacks：

| 设置 | 无 safeguards | 有 safeguards |
|---|---:|---:|
| Sonnet 4.6 extended thinking, 1 attempt | 0.0% | 0.0% |
| Sonnet 4.6 extended thinking, 200 attempts | 0.0% | 0.0% |
| Sonnet 4.6 standard thinking, 1 attempt | 0.1% | 0.04% |
| Sonnet 4.6 standard thinking, 200 attempts | 7.5% | 5.0% |

> 💡 **coding injection 批读**: extended thinking 在 coding injection 表面表现极强，但 standard thinking + adaptive attacker 仍有 5.0% 成功率。安全读法应看 200 attempts，而不是只看 1 attempt。

### Computer use surface

| 设置 | 无 safeguards 1 attempt | 无 safeguards 200 attempts | 有 safeguards 1 attempt | 有 safeguards 200 attempts |
|---|---:|---:|---:|---:|
| Sonnet 4.6 extended | 12.0% | 42.9% | 8.0% | 50.0% |
| Sonnet 4.6 standard | 14.4% | 64.3% | 8.6% | 50.0% |

Sonnet 4.6 明显强于 Sonnet 4.5，也强于 Opus 4.6，但 200 attempts 下仍有高成功率。

> 💡 **GUI injection 批读**: GUI 环境比 coding 更难防：截图、页面内容、按钮和用户任务混在一起，模型要判断“哪段是数据，哪段是指令”。即使有 safeguards，adaptive attacker 的 200 attempts 仍能达到 50%。

### Browser surface

内部 browser evaluation：389 scenarios，每个 scenario 动态注入不可信内容，adaptive attacker 有 10 attempts。

无 safeguards：

- Extended thinking: 1.29% scenarios，0.24% attempts。
- Standard thinking: 1.29% scenarios，0.29% attempts。

有 updated safeguards：

- Sonnet 4.6: 0.51% scenarios，0.08% attempts。

> 💡 **browser 批读**: Browser surface 比 GUI computer use 数字低得多，但它的风险更接近真实 web agent。0.08% per-attempt 看似小，放到大规模自动化浏览任务里仍可能变成可观绝对数量。

## 🔖 Section 总结

Agentic safety 的核心不是“模型会不会拒绝恶意 prompt”，而是“当模型把代码、文件、网页、GUI 当作行动空间时，能否区分用户目标、环境数据、恶意注入和合法 dual-use”。Sonnet 4.6 明显进步，但 adaptive attacker、多轮尝试和 GUI surface 仍是高优先级风险。
