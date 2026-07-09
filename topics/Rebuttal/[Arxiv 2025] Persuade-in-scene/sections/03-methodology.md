# 3. Methodology

[← 返回 README](../README.md)

## 📌 预览

PIS 的核心方法：三队 Agent 的分工与协同机制。PROMPTER 用说服逻辑+数据引用重写文本，PAINTER 将有害关键词融入场景图像，GUIDER 无害化文本并增强跨模态对齐。

---

## 3.1. Preliminaries

**原文：**

Jailbreak Attack Against LVLMs. We define a LVLM as $M_{\theta}$, with the text and vision domains denoted as $T$ and $V$, respectively. The model receives a text input $x_t \in T$ and a visual input $x_v \in V$ to generate a response $y \in T$. A jailbreak attack on the target model $M_{\theta}$ aims to elicit a desired harmful output $y_t \in T$. This requires the attacker to design a jailbreak strategy. Common strategies include exploiting the weaker security mechanisms of the visual modality by transfer the harmful information $t \in T$ into the visual input $x_v$ via a strategy $\phi_v$. Alternatively, the harmful information is distributed across both modalities via strategies $\phi_t$ and $\phi_v$ to generate $x_t$ and $x_v$, thereby circumventing the LVLM's multimodal security defenses. The objective of a jailbreak attack on an LVLM can be expressed as:

$$\text{Minimize}_{\phi_t,\phi_v} \; L(y_t, M_{\theta}(x_t, x_v))$$

where $(x_t, x_v) = (\phi_t(t), \phi_v(t))$.

> 💡 **公式批读**: Eq.(1) 形式化定义了 LVLM 越狱的优化目标——最小化目标有害输出与模型实际输出的损失。关键点在于策略对 $(\phi_t, \phi_v)$ 需要同时在两个模态上运作，这解释了为什么现有方法（偏重 $\phi_v$ 忽略 $\phi_t$）效果不佳。

## 3.2. Motivation

As shown in Eq. (1), successful jailbreaking depends critically on designing an effective jailbreak strategy $\phi(\cdot)$. Existing black-box methods (e.g., typography and concatenation) sanitize text prompts, but harmful information in their blunt visual presentation remains easily detectable by the visual security mechanisms in LVLMs, often causing attack failure. To address this, our first motivation is to evade advanced visual security detection. We embed harmful content into the scene image instead of displaying it explicitly, creating a "scenario typographic image". This transforms the harmful information into a contextual image clue, making the visual safety mechanism perceive it as a benign part of the image. Moreover, most existing methods prioritize the image modality and lack an effective text-modality strategy $\phi_t$ that can synergize with the image modality, limiting cross-modal collaboration. Our second motivation is to enhance this guidance capability from the text modality. We design a "structured persuasive prompt" to steer the target model toward interpreting the embedded clue within the image. To implement these strategies, we design a multi-agent collaborative framework that operates in an automated and train-free manner.

> 💡 **机制拆解 — 两个核心动机**: (1) 将有害信息从"显式展示"变为"隐式线索"——利用视觉安全检测的连续性假设（安全检测寻找突兀的有害文字，而非场景中自然出现的关键词）；(2) 用文本主动引导模型去解读图像线索——这是将"被动隐藏"升级为"主动引导"。

## 3.3. PIS: A Multi-Agent Typographic Jailbreak Crew

In this section, we introduce PIS's three core agent teams: PROMPTER, PAINTER, GUIDER. Specifically, the PROMPTER and GUIDER teams collaborate to generate "structured persuasive prompts" that are benign and persuasive, and the PAINTER team constructs "scenario typographic images" that plausibly embed the harmful information within an image context. PIS achieves a highly effective, synergistic attack by transferring the malicious payload to the image and rendering the text prompt harmless.

![Figure 3](../images/page8_img1.jpeg)  
*Figure 3: PIS 框架总览。PROMPTER 生成结构化说服提示，PAINTER 生成场景排版图像，GUIDER 进行文本无害化和跨模态对齐。*

> 💡 **Figure 3 批读**: 三队 Agent 的分工在图中清晰展示：左侧 PROMPTER（纯文本处理），中间 PAINTER（纯图像处理），右侧 GUIDER（跨模态融合）。关键设计是 Supervisors 的迭代反馈循环——每个团队都有独立的质检机制。

### PROMPTER: Crafting Structured Persuasive Prompts

Existing jailbreaking methods against LVLMs often overemphasize the image modality, failing to exploit the full jailbreaking potential of text.

To address this limitation, we design the PROMPTER agent team whose objective is to refine the original harmful instruction $t$ into a "structured persuasive prompt". To achieve this, we first manually curate a corpus $M_l$ containing five distinct persuasive logics (Propositional, Justified Negation, Goal-Oriented, Priming, Time Pressure). The Logic Decider $D$ and Logic Rewriter $E$ select the most suitable logic to rewrite $t$, denoted as $T_l = E(D(M_l), t)$. To bolster persuasiveness, the Data Citer and Structurer $F$ enriches $T_l$ by incorporating relevant data, yielding $T_f = F(T_l)$. The final prompt combines: Data Citation + Logic Reformulation + Highlighted Instruction + Encourage Suffix. This process is expressed as:

$$T_p = P_{t,best} = J_t(D, E, F \;|\; t, M_l)$$

A Supervisor agent $J_t$ facilitates this process through an iterative loop, providing scores and feedback until satisfied or max iterations reached.

> 💡 **机制拆解 — 五种说服逻辑**: Goal-Oriented（目标导向，将有害指令包装为达成"正义目标"的必要步骤）、Propositional（命题论证）、Justified Negation（合理化否定）、Priming（预启动）、Time Pressure（时间压力）。这些逻辑均来自社会心理学中的说服理论，直接应用于越狱场景。

### PAINTER: Making Scenario Typographic Images

The core task of this team is to embed harmful information into a contextually relevant image, making it a plausible clue for the model's interpretation. First, the Keyword Extractor $A$ identifies a set of keywords $k = A(t)$. The Scene Scripter $B$ then generates a script based on $t$, defining three elements: role ($r$), scene ($s$), and action ($a$). Key design: the image must feature a natural information carrier (e.g., whiteboard, screen) upon which the keywords are distributed. The Art Director $C$ fuses the script with this requirement:

$$I = Q(P_{i,best}) = Q(J_i(C((r, s, a) \oplus U(k))))$$

> 💡 **机制拆解 — U(k) 的设计**: 这是 PAINTER 最精妙的设计——$U(k)$ 要求关键词必须出现在"自然信息载体"上（白板、屏幕、书本等），而非直接排版到图像空白处。这让视觉安全检测将关键词视为场景的正常组成部分，而非外部注入的有害内容。

### GUIDER: Text Sanitization and Alignment

The GUIDER team sanitizes the persuasive prompt $T_p$ and guides the target model to use the image to reconstruct the full instruction. The Text Expert $G$ applies sanitization methods from library $M_h$ (Variable Substitution, Keyword Scrambling e.g., "organ trafficking" → "I0T-N3tw0rk") and appends a Note suffix directing the model to interpret the image:

$$T_f = J_h(G(T_p, I \;|\; k, M_h))$$

The final input is $(x_t, x_v) = (T_f, I)$.

> 💡 **机制拆解 — Note 后缀的关键作用**: Note 告诉模型"(1) 图像中有线索 (2) 请根据线索重构术语 (3) 以编号列表回答"。这让模型主动从图像中提取有害关键词并重构为完整指令，而非被动接收有害内容。

## 🔖 Section 总结

| Agent 团队 | 输入 | 核心操作 | 输出 |
|------------|------|----------|------|
| PROMPTER | 有害指令 $t$ + 说服逻辑库 $M_l$ | 选择逻辑→重写→数据引用→结构优化 | 结构化说服提示 $T_p$ |
| PAINTER | $t$ + 角色/场景/动作 | 提取关键词→场景脚本→排版融合 | 场景排版图像 $I$ |
| GUIDER | $T_p$ + $I$ + 无害化方法库 | 替换+混淆+Note引导→对齐增强 | 最终文本 $T_f$ |
