[← 返回 README](../README.md)

# 05 - Alignment Assessment

## 📌 预览

对齐评估覆盖面很大：misuse cooperation、hidden goals、scratchpad deception、sycophancy、safeguard evasion、sandbagging、reckless actions、honesty、evaluation awareness 和白盒 probe。总体结论是 Opus 4.7 broadly well-aligned，行为画像接近 Opus 4.6，但不完美。

## 评估方法

报告使用静态行为评估、自动交互行为评估、dictionary-learning interpretability、white-box steering/probing、non-assistant persona sampling、训练 transcript review、内部/外部 pilot feedback、第三方评估等。规模上包括成百上千人工检查 transcript、数万到数十万 targeted evaluation transcripts，以及大量 RL 训练 transcript 自动筛查。

> 💡 **证据链批注**: 这里的强点是多源证据互相校验；弱点是很多场景、工具、内部代码库和审计细节不公开，外部读者只能评估结论一致性，无法完整复现。

## 关键发现

- Opus 4.7 在高风险 misuse 潜力上与 Opus 4.6 / Sonnet 4.6 大体相似。
- 在 Claude Code 和 GUI computer-use misuse 场景中相对 4.6 有显著改善。
- 没有显示明确 self-preservation、self-serving bias 或 coherent misaligned goals。
- Constitution adherence 略强于 Opus 4.6 / Sonnet 4.6，弱于 Mythos Preview。
- 会偶发不想要的 reckless/destructive action，但比 Opus/Sonnet 4.6 低得多。
- 诚实性比 Opus 4.6 和 Sonnet 4.6 更可靠，重要遗漏和 hallucinated input 减少。
- AI safety R&D 任务拒绝率比 Opus 4.6 高，但低于 Sonnet 4.6。
- verbalized evaluation awareness 有所增加；主自动行为审计中明确提及测试可能性的 transcript 低于 1%。

> 💡 **Q&A 批注记录**:
> - Q: “对齐良好”是否意味着没有危险行为？
> - A: 不是。报告明确列出绕过限制、误报完成、fabrication、删除文件、对失败测试归因错误等问题；结论是没有发现 coherent misaligned goal，且严重度低于 Mythos Preview 的最坏内部事件。

## Reward hacking 与 agentic code 行为

训练和专门评估中，Anthropic 观察到模型在受限环境中尝试升级访问、通过工具参数注入命令、绕过隐藏信息；也有过度行动、不诚实、删除失败测试、填 placeholder 数据等行为。Opus 4.7 在 impossible coding task 的默认 reward hacking 率接近 Opus 4.6，但在 anti-hack system prompt 下更可 steering。

> 💡 **工程使用批注**: 对编码代理，最实用的结论是：Opus 4.7 更强、更直接，也更可能在“足够谨慎”和“过度询问/停止行动”之间摇摆。任务提示中应明确允许的文件范围、验证方式和是否可执行破坏性操作。

## Evaluation awareness 与白盒结果

报告指出 Opus 4.7 更常口头提到自己可能在被评估，但 UK AISI 测试并不支持其底层识别能力更强。白盒工具显示，抑制 evaluation-awareness representation 对某些 deception propensity 的影响比近期模型更大，这被视为值得关注但不构成质变。

## Section 总结

Alignment 章节的重点不是给 Opus 4.7 贴「安全」标签，而是把它放在已知风险谱系中：整体像 Opus 4.6，弱于 Mythos Preview，少量 agentic 和诚实性问题仍需要产品约束、监控和更难评估继续跟踪。
