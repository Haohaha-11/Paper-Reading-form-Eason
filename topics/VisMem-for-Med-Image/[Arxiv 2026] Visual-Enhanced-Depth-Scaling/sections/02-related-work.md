[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
本节定位相关方法谱系，重点看作者如何区分自己与已有工作。

---

# 2.1. Explicit Multimodal Reasoning

Multimodal reasoning enables model to reason over information from different modalities to solve complex tasks. There are many prior works [15–17, 20, 36, 39, 40, 40, 41, 72, 73, 79] focusing extensively on enhancing reasoning capabilities. Earlier work rely on the CoT prompting to perform explicit thinking steps in the text space before generating the final answer. However, this paradigm generally lack sufficient visual grounding capability and leads to unsatisfactory misalignment and hallucination [3, 23]. To address these limitations, recent studies [20, 40, 79] have explored converting visual information into textual formats prior to reasoning, leveraging external tools or specialized visual experts to generate descriptive representations that guide LLMs. For instance, Hu et al. [20] pioneered the integration of visual captions, extracting semantic content as text and concatenating it with input prompts to bolster reasoning capabilities. To further enhance fine-grained reasoning, subsequent works have focused on regional understanding, aiming to improve textual expressiveness by describing specific image regions. Others [39–41] identified entities and their relationships within images, facilitating fine-grained reasoning through explicit modeling of inter-entity connections.

> 💡 **相关工作批读**: 显式 multimodal CoT 的优势是可解释，缺点是常把图像先转成文本或对象关系再推理，细粒度视觉证据会在文本化过程中丢失。本文选择 latent route，本质是在避免长文本链条和视觉证据文本化之间找折中。

A line of concurrent works advocates using vision-text interleaved format during the rationale generation and reasoning process. The model draws auxiliary lines or marks based on original image to record thinking path, zoom or crop regions, or perform code editing, etc. Building on these paradigms, Zhang et al. [77] first proposed decoupling rationale generation from answer generation in the Vision-Text Reasoning field. Subsequently, Shao et al. [45] annotated key regions of the original image in intermediate steps, training models to focus on image regions relevant to the answer. While some works [13, 75] further extract key image regions progressively during reasoning, combining visual information with textual reasoning to generate the final answer. Moreover, new methods [21, 35] emulated human thought by sketching images during reasoning, focusing on core concepts, structures, and relationships while ignoring redundant details. Other works [7, 32] generated new images with auxiliary markers during reasoning, combining them with text to improve reasoning in complex scenarios. To fully shift reasoning from the linguistic domain to the visual modality, Xu et al. [66] proposes to reasoning exclusively with dynamic generated images, demonstrating substantial performance gains in visual navigation tasks.

> 💡 **工具增强路线**: 这些方法通过画线、裁剪、缩放、标注或生成中间图像来“边看边想”。它们与 SCF-VR 的共同点是都承认静态一次性视觉输入不够；区别是 SCF-VR 把区域选择和 replay 放进 latent/attention 机制，目标是减少外部工具调用成本。

More recently, Vision-R1 [24] and VL-Rethinker [54] leveraged Group Relative Policy Optimization [9] (GRPO) to refine reasoning trajectories through rollout-based sampling and reward scoring. Complementing these policy-driven approaches, concurrent works further enhance reasoning capabilities via novel cognitive paradigms, including selfcritiquing cycles [8, 44], iterative rethinking [68], and onpolicy distillation [78].

> 💡 **RL/反思路线**: Vision-R1、VL-Rethinker 等通过 rollout 和 reward 改善显式推理轨迹，通常训练/采样成本更高。本文没有走 RL，而是用 curriculum + auxiliary self-distillation 让 latent token 自己吸收 CoT 结构。

# 2.2. Latent Reasoning

Different from explicit reasoning in the discrete token space, latent reasoning refers to internal computation performed in a hidden space before answer generation. Hao et al. [18] pioneers continuous latent space reasoning by feeding the last hidden states as input embeddings for the next step without generating intermediate discrete tokens, substantially reducing reasoning latency. However, subsequent studies have indicated that such paradigm may suffer from feature homogenization without explicit supervision on intermediate latent states. To address this limitation, a series of works attempt to enhance the quality of intermediate representations using diverse strategies. Cheng and Van Durme [6] introduced variable-length contemplation tokens for latent reasoning, mitigating quality degradation caused by fixedlength constraints. Similarly, Shen et al. [48] leveraged distillation tactic to align student and teacher hidden activations along with explicit supervision, thereby constraining latent reasoning paths. Beyond these efforts, Wei et al. [64] adopted step-level supervision to further stabilize the reasoning space.

> 💡 **Latent reasoning 背景**: 文本 latent reasoning 已经证明可以用 hidden state 替代部分 CoT token，但问题是中间 latent 容易同质化、缺少路径约束。本文在多模态场景里进一步指出：视觉 token 的优化稳定性比文本 token 更差，所以直接套文本 latent reasoning 不够。

Recently, latent reasoning has been extended to Multimodal Large Language Models (MLLMs). Distinct from text-only LLMs, MLLMs necessitate the effective integration of visual features within the latent reasoning space. Several efforts [30, 34, 43, 62, 69] have been dedicated to injecting visual cues into the latent space to facilitate visualgrounded reasoning. For instance, Wang et al. [62] emphasize image details by constructing a structured cognitive hierarchy, albeit relying on annotation-intensive multimodal reasoning data. Similarly, Liu et al. [34] progressively select visual patches to inject into latent thinking tokens via confidence-guided policy gradient optimization. In this work, we systematically analyze gradient dynamics during latent reasoning training and reveal two critical bottlenecks towards token-wise optimization behavior.

> 💡 **本文差异**: 现有 multimodal latent reasoning 也会注入视觉 cue 或选择 patch，但本文的差异在于先用梯度行为定位瓶颈，再用 SCF-VR 做空间一致视觉重放、用 RDS 做复杂 token 的深度扩展。它更像一套优化与计算预算分配机制，而不是单一 visual token injection。

---

## 🔖 Section 总结

### 核心洞察
1. 显式 CoT、工具增强、RL 推理都能增强多模态 reasoning，但普遍带来长链条、外部调用或训练成本。
2. Latent reasoning 的短板是中间表示质量不可见、可能同质化；多模态场景还额外面临视觉 token 欠优化。
3. 本文的位置是：把视觉证据重放和 token-wise depth scaling 加到 latent reasoning 框架中。
