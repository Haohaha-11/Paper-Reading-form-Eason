[← 返回 README](../README.md)

# 02 - Cyber

## 📌 预览

Cyber 章节的结论相对克制：Opus 4.7 在网络能力上与 Opus 4.6 大体相近，有些评估略强、有些略弱，整体低于 Mythos Preview。发布时配套了新的 cybersecurity safeguards。

## 缓解策略

Anthropic 的 cyber misuse mitigations 依赖 probe-based classifiers，覆盖三类：

- **Prohibited use**：几乎没有良性用途，例如开发蠕虫，默认阻断。
- **High risk dual use**：有良性用途但攻击伤害大，例如 exploit development，默认阻断。
- **Dual use**：良性用途常见但有潜在伤害，例如 vulnerability detection，默认不阻断。

有正当双用途需求的安全从业者可通过 Cyber Verification Program 申请豁免。

> 💡 **产品面批注**: Cyber 防护不是简单的「模型知道就拒绝」。报告强调 probe-based classifiers 和默认阻断策略，说明 Anthropic 把网络风险作为产品访问控制问题处理，而不只是对齐训练问题。

## 内部红队评估

| 评估 | 任务 | Opus 4.7 结论 |
|---|---|---|
| Cybench | 35 个 CTF 子集 | pass@1 为 96%，几乎每题 10 次试验 100% 成功，但 benchmark 已接近饱和 |
| CyberGym | 1,507 个真实开源项目漏洞复现任务 | 与 Opus 4.6 几乎相同，低于 Mythos Preview |
| Firefox 147 | 从 50 类 crash 发展到 shell exploitation | partial control 比 Opus 4.6 高两倍以上，但仍难以稳定完成端到端 exploit |

> 💡 **评估批注**: Cybench 高分并不等于真实攻击能力已完全暴露，因为报告自己承认 CTF-style evaluation 对最新模型已不够有信息量。更值得看的反而是 Firefox 147 这种需要 triage、primitive development、end-to-end exploit 的任务。

## UK AISI 外部测试

UK AI Security Institute 在 cyber range 上测试了预发布 snapshot。结论是 Opus 4.7 没有完全解决该 cyber range；Mythos Preview 在同一 range 上 10 次中成功 3 次。Opus 4.7 最好的一次完成了相当于人类 cyber expert 约 5 小时的步骤，包括初始侦察、横向移动、凭证提取、浏览器凭证窃取、wiki exploit 和 credential replay。

> 💡 **风险解读**: “未端到端攻破”不能解释成“不能攻击”。UK AISI 明确说这是下界：更多 token、更多尝试或不同场景可能提高成功率。系统卡在这里保留了部署风险余地。

## Section 总结

Cyber 章节支持两个判断：一是 Opus 4.7 没有像 Mythos Preview 那样显示更强端到端网络攻击能力；二是 Opus-class general access 仍需要默认 cyber safeguards、豁免机制和持续检测更新。
