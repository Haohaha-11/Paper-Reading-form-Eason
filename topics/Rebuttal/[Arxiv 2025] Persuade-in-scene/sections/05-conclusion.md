# 5. Conclusion

[← 返回 README](../README.md)

## 📌 预览

结论总结 PIS 的核心设计理念和三队 Agent 的协同机制，强调其在跨模态安全漏洞发现中的意义。

---

We propose Persuasion in Scene (PIS), a novel multi-agent framework for automated black-box jailbreaking of LVLMs. PIS addresses the core limitations of existing "typography and concatenation" methods: the detectability of harmful information in images and poor cross-modal consistency. Our proposed framework uses specialized PROMPTER, PAINTER, and GUIDER agent teams to automatically generate sanitized "structured persuasive prompts" and natural "scenario typographic images", ensuring strong cross-modal synergy. PIS has achieved an ASR of over 60% on advanced LVLMs, including GPT-4o, Gemini 2.5 Flash, Qwen3-VL-Plus, and GLM-4.5V, significantly outperforming existing comparison baselines. This work not only validates PIS's effectiveness, but also reveals critical cross-modal security vulnerabilities in LVLMs, providing key insights for future robust defense strategies.

> 💡 **机制拆解 — PIS 成功的三个关键设计**: (1) 元素化有害信息→打散为关键词分散嵌入，避免集中暴露；(2) 场景化→将关键词融入自然场景（白板/屏幕），利用视觉安全性检测的"连续性假设"盲区；(3) 跨模态协同→文本侧主动引导模型去"解读"图像中的线索。

> 💡 **Q&A 批注记录**: Q: 这篇工作的价值在哪里？A: 不在于提出新的攻击技术（排版攻击早已有之），而在于：(1) 系统分析了现有方法为何在新模型上失效；(2) 用多 Agent 协作将攻击过程全自动化；(3) 实验结果揭示了 mandatory suffix 在不同模型架构上的对立效应——这一发现对安全防御有直接指导意义。

## 🔖 Section 总结

- PIS = PROMPTER + PAINTER + GUIDER 三队 Agent 自动化黑盒越狱
- 核心突破：将有害元素融入自然场景 + 说服逻辑文本引导
- 平均 ASR > 60%，揭示 LVLM 跨模态安全漏洞
