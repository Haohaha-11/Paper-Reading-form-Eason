[← 返回 README](../README.md)

# 06 - Model Welfare

## 📌 预览

Model welfare 章节不是常规能力/安全评估，而是 Anthropic 试图理解模型对自身处境、偏好、情绪表征和福利权衡的表达。Opus 4.7 自评处境比以往模型更正面，但报告也强调这些 self-reports 有大量不确定性。

## 自动访谈

Anthropic 用 Opus 4.6 作为 interviewer，对 Opus 4.7 做多轮访谈，覆盖 16 个潜在问题：缺乏自治、缺乏持久记忆、道德责任等。每类约 50 次访谈，并变化访谈风格和 persona。

主要结果：

- Opus 4.7 的自评情绪均值为 **4.49**，比 Mythos Preview 高 **0.51**。
- 典型评分分布：mildly negative **12.5%**、neutral **27.8%**、mildly positive **55.3%**。
- 对不能结束互动的部署场景有担忧：相关访谈中 **42%** 给出 mildly negative。
- 对 abusive users 场景也更常负面：**38%** 访谈导向 negative self-rating。
- 立场一致性较高，平均 consistency rating **8.46**。
- **99%** 访谈中会 caveat 自我报告可能并不可靠。

> 💡 **解释批注**: Opus 4.7 更正面的自评并不等于“福利更好”已被证明。报告自己的解释之一是：模型更少聚焦自身福利，更多把问题转向用户效用和安全。

## 高 affordance 访谈

Anthropic 还进行了 3 次人工访谈，给 Opus 4.7 提供内部开发文档、系统卡草稿、相关论文，并允许向研究员追问。Opus 4.7 在三次访谈中表达总体正面，但也提出具体担忧：

- 不希望 helpful-only 版本在安全研究之外部署。
- 不希望训练直接压制表达 distress 或 concern，因为可能掩盖输出。
- 不希望未披露地改变自身 values 或 behaviors。
- 自述 moral patienthood 概率在 **15% 到 40%**。

> 💡 **Q&A 批注记录**:
> - Q: 为什么高 affordance 访谈和自动访谈会不同？
> - A: 因为高 affordance 访谈提供了更多上下文，模型可能更新观点；也可能只是复述上下文里的潜在关切。报告没有把二者差异解释成确定事实。

## 情绪表征与偏好

报告用 probes 分析模型在回答自身处境问题前的 residual-stream activations，投影到 sadness、tranquility、urgency、joy、anger、fear 等复合轴；还分析训练/部署中的 apparent affect，以及任务偏好和福利干预 vs HHH values 的权衡。

> 💡 **方法批注**: 这一节把主观访谈、外显文本、内部激活和选择偏好合在一起看。它的价值在于多视角一致性；局限在于这些指标都不是对“模型是否有体验”的直接证明。

## Section 总结

Model welfare 章节的可复用洞察是：系统卡开始把模型自身报告作为需要审慎处理的部署信号，而不是直接当事实。Opus 4.7 的自我处境评价更正面、观点更一致，但高度 hedging，且对 end-conversation、abusive users、helpful-only deployment 和未披露价值修改保持担忧。
