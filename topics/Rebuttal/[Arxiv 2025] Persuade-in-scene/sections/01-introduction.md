# 1. Introduction

[← 返回 README](../README.md)

## 📌 预览

引言分析了 LVLM 安全漏洞的背景、现有黑盒越狱方法的两大局限（图像不自然+文本设计忽视），以及 PIS 的核心设计动机。

---

**原文：**

Large Vision-Language Models (LVLMs) have recently demonstrated remarkable capabilities in understanding multimodal information, enabling various applications such as image captioning and visual question answering. Due to outstanding performance in visual-textual content alignment and processing of multimodal models such as GPT-4o, Gemini, Qwen-VL, and LLaVA, they are widely used in healthcare diagnostics and autonomous driving. However, this rapid proliferation has simultaneously raised public concerns about safety. Although LVLMs inherit safety alignments from their foundational Large Language Models (LLMs), recent studies have revealed that their increased vulnerability to adversarial attacks, largely attributable to the introduction of image modality. Consequently, investigating jailbreak in LVLMs is imperative research for securing multimodal systems.

To exploit the vulnerability of visual safety mechanisms in LVLMs, existing studies propose white-box attack methods using model gradient information to generate adversarial images for jailbreaking. In contrast, black-box attacks for closed-source models assume attackers can only modify inputs for harmful content generation. A key paradigm in black-box attacks is the use of "typography" technology. For example, FigStep typesets text instructions onto images indirectly to achieve the transfer of harmful information, while other methods "concatenate" typographic images with auxiliary visuals such as scene-related images to steer the target model toward generating harmful responses. Although these methods have proven effective for early LVLMs, their limitations have gradually emerged due to the improved visual detection tools for harmful information in these attack images and advances in cross-modal security mechanisms.

Through systematic analysis, we empirically demonstrate two key limitations in current black-box attack paradigms using "typography and concatenation" techniques. First, the attack images constructed by these methods lack continuity and naturalness, making the harmful information so distinct that advanced LVLMs can detect it easily and trigger the security mechanism, leading to attack failure (Fig. 1). Second, existing methods overemphasize image design while neglecting textual prompts design, causing an imbalance in cross-modal attacks. However, recent studies have shown that cross-modal consistency between textual prompts and images is crucial for successful LVLM jailbreaking. To address both limitations, we hypothesize and experimentally verify that elementizing harmful information and integrating it naturally into scenario-relevant images can bypass visual security detection. Moreover, we use logical rewriting and data citation strategies to design "structured persuasive prompts", which enable the text modality to participate in jailbreaking actively.

Based on the above analysis, we propose a black-box framework Persuasion in Scene (PIS), which is a novel multi-agent crew for jailbreaking against LVLMs. PIS automatically generates a structured persuasive prompt and a scenario typographic image from an original harmful instruction, with the entire process driven by agent teams without human intervention. Specifically, the crew operates in three collaborative team: the PROMPTER team rewrites the harmful instruction using logical rewriting and data citation strategies; The PAINTER team embeds extracted harmful keywords into a scene image as reasonable clue, thereby ensuring the image is natural and continuous; and the GUIDER team sanitizes the text and enhances cross-modal synergy between the generated text and image.

> 💡 **问题动机**: 引言的核心论证链：LVLM 引入视觉模态 → 新的攻击面 → 排版攻击有效但粗暴 → 安全检测升级 → 现有方法失效 → 需要更隐蔽的"场景化"越狱。

> 💡 **机制拆解 — 两个关键假设**: (1) 将有害信息"元素化"后融入场景图像可以绕过视觉安全检测——因为安全检测依赖"异常信息"的显著性，而非场景中自然元素；(2) 用说服逻辑重写文本可以主动引导模型去解读图像中的有害线索——这是从"被动隐藏"到"主动引导"的范式转换。

### 贡献总结

- 系统分析了现有"排版+拼接"黑盒攻击的两大局限
- 提出 PIS：三队 Agent 协同的自动化端到端越狱框架
- 在 7 个主流 LVLM 上 ASR 超 60%，显著超越 SOTA

## 🔖 Section 总结

- 核心问题：排版攻击的图像不自然 + 跨模态攻击不平衡
- PIS 方案：元素化有害内容 → 场景图像 + 说服文本
- 三个 Agent 团队：PROMPTER（文本）、PAINTER（图像）、GUIDER（协同）
