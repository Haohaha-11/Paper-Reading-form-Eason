[← 返回 README](../README.md)

# 00 - Introduction

## 📌 预览

本节回答三个基础问题：Opus 4.7 是什么、如何训练/发布、为什么需要触发 RSP 讨论。系统卡的定位很明确：Opus 4.7 是 Anthropic 在 2026 年 4 月的强公开模型，但不是公司能力前沿的最高点。

## 关键内容

- 报告日期是 **April 16, 2026**，标题为 **Claude Opus 4.7 System Card**。
- 模型被描述为擅长 **software engineering、knowledge work、agentic tool use、computer use**。
- 训练数据来自公开互联网、公开/私有数据集，以及其他模型生成的合成数据；过程包含清洗、去重和分类。
- Anthropic 使用 ClaudeBot 抓取公开网页，并声明遵循 `robots.txt`、不访问需登录/CAPTCHA/密码保护的页面。
- 预训练后进行了 substantial post-training 和 fine-tuning，目标是符合 Claude constitution。
- 模型是 multilingual，通常跟随用户输入语言回复，但 **输出仅为文本**；多模态能力指输入/理解与工具化使用，不表示模型直接输出图像。

> 💡 **定位批注**: 「最强 general-access」是本系统卡的核心限定词。Opus 4.7 比 Opus 4.6 强，但 Mythos Preview 更强且只面向有限用户；因此 Opus 4.7 的产品意义和风险意义不同：产品上它是公开主力，RSP 上它不是新 frontier。

## 发布决策链

Anthropic 说明大模型发布不是只看 benchmark，而是把模型 snapshot、内部/外部测试、Usage Policy、防护机制和 RSP 风险报告放在一起。Opus 4.7 因为「significantly more capable than Opus 4.6」，系统卡需要解释它如何影响此前 Risk Report。

> 💡 **RSP 入口批注**: Introduction 本身不做完整安全证明，而是把读者导向后续 RSP、Cyber、Safeguards、Agentic Safety 和 Alignment 章节。读这类系统卡时，第一步不是看最高分，而是确认「与上一个风险报告相比，哪些威胁模型被重新评估」。

## Section 总结

Opus 4.7 的基础画像是：文本输出、多语言、强工具/代码/电脑使用、经过 constitutional post-training 的公开 Opus 模型。它的风险讨论从一开始就服务于一个判断：能力提升是否足以改变 Anthropic 已发布的 RSP 风险图景。
