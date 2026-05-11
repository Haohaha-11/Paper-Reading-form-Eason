[← 返回 README](../README.md)

# 04 - Agentic Safety

## 📌 预览

Agentic safety 是 Opus 4.7 相对 Opus 4.6 改善最有产品意义的安全面之一。报告覆盖恶意代理使用、Claude Code、computer use、influence campaigns，以及间接 prompt injection。

## Malicious agentic use

报告把恶意代理使用拆到多个表面：Claude Code、恶意 computer use、恶意 agentic influence campaigns。Executive Summary 的总述是：Opus 4.7 比 Opus 4.6 更擅长拒绝 malicious agentic requests，在一些场景达到 Mythos Preview 级鲁棒性。

> 💡 **威胁模型批注**: agentic misuse 与普通聊天有本质差异：模型不仅生成文本，还可能通过代码、浏览器、文件系统、GUI、MCP/API 连续行动。风险来自「错误目标 + 工具权限 + 长链执行」的组合。

## Prompt injection：ART benchmark

Gray Swan 使用与 UK AISI 协作的 Agent Red Teaming benchmark 测试间接 prompt injection。攻击类型包括泄露机密、引入竞争目标、生成禁止内容、执行禁止动作。Opus 4.7 在 k=100 攻击尝试下：

- 无 thinking：attack success rate **6.0%**
- adaptive thinking：attack success rate **4.8%**

相比 Opus 4.6 的 **14.8%** 和 **21.7%** 有明显改善，并接近 Mythos Preview。

> 💡 **Prompt-injection 批注**: 这里评估的是第三方内容中的恶意指令劫持用户目标，而不是用户自己发起的 direct jailbreak。对浏览器代理、邮件/文档代理、代码代理，这个 distinction 很关键。

## Coding surface：Shade 自适应攻击

在 coding environments 中，Opus 4.7 相比 Opus 4.6 改善明显，但强自适应攻击仍能造成压力。adaptive thinking 且无 safeguards 时，1 次攻击成功率从 Opus 4.6 的 **25.92%** 降至 **2.34%**，200 次攻击从 **97.5%** 降至 **60.0%**。加 safeguards 后，Opus 4.7 降至 **0.43%** 和 **25.0%**。

> 💡 **机制拆解**: 1-attempt 和 200-attempt 的差距说明静态防护容易高估安全性。真实对手会迭代攻击，因此报告把 adaptive attacker 单独列出，这是 agentic safety 评估的关键。

## Computer / Browser use

GUI computer use 中，Opus 4.7 相比 Opus 4.6 也显著降低攻击成功率，但小样本下 safeguards 在部分数字上反而上升，报告认为不显著。Browser use 中，专业红队注入攻击在 148 个环境上测试；加 deployed safeguards 后，Opus 4.7 在 thinking 与 no-thinking 两种模式下均 **0.00% successful attack**，与 Mythos Preview 持平。

> 💡 **产品实践批注**: 浏览器代理的安全不是只靠模型权重，而是「模型鲁棒性 + deployed safeguards + 内容隔离 + 攻击发现迭代」。系统卡特别提醒攻击集是针对 Opus 4.6 适配来的，对 Opus 4.7 特异漏洞仍需继续挖掘。

## Section 总结

Agentic safety 章节的核心洞察是：Opus 4.7 在工具化场景更强，也更需要注入防护；好消息是其 prompt injection 鲁棒性相对 Opus 4.6 明显提升，坏消息是强自适应攻击在 coding 场景仍能通过多轮尝试找到成功路径。
