[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
Introduction 把现有多模态推理分成 text-only CoT、think-with-images、latent reasoning 三条线，然后指出它们共同缺少“动态决定何时看图、看哪里、想多久”的能力。

---

# 1. Introduction

Multimodal Large Language Models (MLLMs) [1, 2, 3, 4] have achieved remarkable breakthroughs in integrating visual and linguistic information. This progress has facilitated the incorporation of Chain-of-Thought (CoT) reasoning into multimodal tasks, enabling models to construct structured reasoning paths across visual and textual modalities. Current multimodal reasoning approaches can be broadly categorized into three types: (1) Textual-only Reasoning [5, 6, 7], which generates intermediate reasoning steps in the sematic space. Such methods explicitly express reasoning logic through language generation but often suffer from language bias and insufficient visual grounding, as shown in Figure. 1(a). (2) Think with Images attempts to directly manipulate or augment images during reasoning, such as local zooming [8, 9], region highlighting [10, 11], or generating intermediate reasoning steps via diffusion models [12, 13] to enhance visual alignment. Despite their effectiveness in improving reasoning to a certain extent, they still face challenges such as unstable tool invocation and high inference overhead, as reflected in Figure 1(b). Recently, latent-space reasoning has emerged as a promising paradigm for enhancing reasoning capabilities in large language models, as exemplified by approaches such as CoCoNut [14] and LatentSeek [15]. Its core idea is to perform implicit reasoning in the latent space, replacing explicit textual steps with latent vectors to reduce redundant generation and capture more compact information. However, recent studies [16, 17, 18, 19] still rely on extra training to enforce latent reasoning triggered at fixed positions (via special tokens). This rigidity prevents the model from adaptively allocating reasoning effort.

> 💡 **问题谱系**: 这段不是泛泛综述，而是在为 DMLR 设定坐标。Text-only CoT 可解释但 grounding 弱；Think-with-Images 看得更多但工具调用慢；latent reasoning 省 token 但常要训练并固定触发位置。DMLR 想拿 latent reasoning 的效率，同时补上动态视觉交互。

Inspired by human cognition, we argue that reasoning is not fixed. Instead, humans dynamically revisit visual information, specifically when they encounter uncertainty. Drawing on this intuition, we empirically analyze the interplay between the model’s visual reliance and its internal confidence. Our analysis reveals two key phenomena: (i) Visual information is used only at a few specific stages of the reasoning process rather than at fixed positions, and (ii) Internal confidence serves as a natural indicator for the need of visual grounding as it strongly correlates with reasoning correctness. These findings suggest that effective multimodal reasoning relies on dynamic visual usage guided by internal confidence.

> 💡 **核心假设**: 作者把“人类不确定时回看图像”翻译成两个可测变量：visual reliance 和 internal confidence。后文第 3 节要证明视觉依赖确实是稀疏的，confidence 确实和正确性/faithfulness/grounding 有相关性；否则 DMLR 的 reward 设计就站不住。

In light of these observations, we propose DMLR, a Test-time Dynamic Multimodal Latent Reasoning Framework, as shown in Figure 1(c). Specifically, it introduces optimizable latent think tokens to serve as a mental draft, which are iteratively refined through confidence-guided policy gradient updates. Crucially, we design a confidence-driven dynamic visual injection strategy. At each step, the model autonomously determines whether to revisit visual information and which contents to select (ranging from none to a few specific patches). This mechanism allows the model to naturally skip visual injection when internal confidence is sufficient, or actively integrate targeted visual clues when necessary, all driven by the objective of maximizing reasoning confidence, effectively mimicking the human cognitive process of checking visual clues to build confidence. After several iterations, the optimized latent tokens are decoded with the input without extra inference cost. Extensive experiments demonstrate that DMLR consistently outperforms existing methods across diverse architectures and tasks while maintaining high efficiency. The main contributions can be summarized as follows:

> 💡 **方法总览**: 这里给出完整数据流：图文输入 + latent think tokens → confidence-guided policy gradient 更新 latent → 动态选 patch 注入 → 迭代后解码答案。注意 “without extra inference cost” 更准确应理解为不额外生成长 CoT token；实际仍有 test-time optimization 的 forward/attention 成本，后面效率图要具体看。

$\bullet$ We reveal two key phenomena: Visual information contributes only at specific reasoning steps; and confidence reflects both reasoning quality and visual grounding.
$\pmb { \varrho }$ We propose DMLR, a test-time framework for multimodal latent reasoning that integrates confidenceguided latent optimization with dynamic visual injection.
$\pmb { \otimes }$ Extensive evaluations show that DMLR consistently outperforms other methods across diverse architectures and multimodal tasks, while maintaining high efficiency.

> 💡 **贡献批读**: 三点贡献刚好对应论文三段证据：第 3 节做 motivation measurement，第 4 节给 test-time latent/DVI 方法，第 5 节用多模型多任务证明收益。最值得追问的是 confidence 是否只是相关 proxy，而不是可靠 causal signal。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 现有范式 | Textual-only Reasoning / Think with Images / Latent-space reasoning |
| DMLR 类型 | Test-time Dynamic Multimodal Latent Reasoning |
| 视觉注入策略 | none 到少量 specific patches 动态选择 |

### 核心洞察
1. DMLR 的切入点是“动态性”：动态分配 reasoning effort，动态选择视觉 patch。
2. 它不是替代 MLLM backbone，而是插在 inference pipeline 内。
3. Introduction 已经埋下两个验证点：confidence 能否作为信号，动态注入是否比固定/全量注入更稳。
