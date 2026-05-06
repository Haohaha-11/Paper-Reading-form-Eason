[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
本节定位相关方法谱系，重点看作者如何区分自己与已有工作。

---

# 2.1. Explicit Multimodal Reasoning

Multimodal reasoning enables model to reason over information from different modalities to solve complex tasks. There are many prior works [15–17, 20, 36, 39, 40, 40, 41, 72, 73, 79] focusing extensively on enhancing reasoning capabilities. Earlier work rely on the CoT prompting to perform explicit thinking steps in the text space before generating the final answer. However, this paradigm generally lack sufficient visual grounding capability and leads to unsatisfactory misalignment and hallucination [3, 23]. To address these limitations, recent studies [20, 40, 79] have explored converting visual information into textual formats prior to reasoning, leveraging external tools or specialized visual experts to generate descriptive representations that guide LLMs. For instance, Hu et al. [20] pioneered the integration of visual captions, extracting semantic content as text and concatenating it with input prompts to bolster reasoning capabilities. To further enhance fine-grained reasoning, subsequent works have focused on regional understanding, aiming to improve textual expressiveness by describing specific image regions. Others [39–41] identified entities and their relationships within images, facilitating fine-grained reasoning through explicit modeling of inter-entity connections.

> 💡 **批注**: 这段是 latent memory / medical VLM 主线：关注视觉证据如何进入 latent space、如何被记忆/更新/调用，以及是否能支撑可靠诊断。

A line of concurrent works advocates using vision-text interleaved format during the rationale generation and reasoning process. The model draws auxiliary lines or marks based on original image to record thinking path, zoom or crop regions, or perform code editing, etc. Building on these paradigms, Zhang et al. [77] first proposed decoupling rationale generation from answer generation in the Vision-Text Reasoning field. Subsequently, Shao et al. [45] annotated key regions of the original image in intermediate steps, training models to focus on image regions relevant to the answer. While some works [13, 75] further extract key image regions progressively during reasoning, combining visual information with textual reasoning to generate the final answer. Moreover, new methods [21, 35] emulated human thought by sketching images during reasoning, focusing on core concepts, structures, and relationships while ignoring redundant details. Other works [7, 32] generated new images with auxiliary markers during reasoning, combining them with text to improve reasoning in complex scenarios. To fully shift reasoning from the linguistic domain to the visual modality, Xu et al. [66] proposes to reasoning exclusively with dynamic generated images, demonstrating substantial performance gains in visual navigation tasks.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

More recently, Vision-R1 [24] and VL-Rethinker [54] leveraged Group Relative Policy Optimization [9] (GRPO) to refine reasoning trajectories through rollout-based sampling and reward scoring. Complementing these policy-driven approaches, concurrent works further enhance reasoning capabilities via novel cognitive paradigms, including selfcritiquing cycles [8, 44], iterative rethinking [68], and onpolicy distillation [78].

> 💡 **批注**: 这是蒸馏逻辑：重点看 teacher 的什么能力被迁移给 student，以及训练/推理阶段各自保留哪些模块。

# 2.2. Latent Reasoning

Different from explicit reasoning in the discrete token space, latent reasoning refers to internal computation performed in a hidden space before answer generation. Hao et al. [18] pioneers continuous latent space reasoning by feeding the last hidden states as input embeddings for the next step without generating intermediate discrete tokens, substantially reducing reasoning latency. However, subsequent studies have indicated that such paradigm may suffer from feature homogenization without explicit supervision on intermediate latent states. To address this limitation, a series of works attempt to enhance the quality of intermediate representations using diverse strategies. Cheng and Van Durme [6] introduced variable-length contemplation tokens for latent reasoning, mitigating quality degradation caused by fixedlength constraints. Similarly, Shen et al. [48] leveraged distillation tactic to align student and teacher hidden activations along with explicit supervision, thereby constraining latent reasoning paths. Beyond these efforts, Wei et al. [64] adopted step-level supervision to further stabilize the reasoning space.

> 💡 **批注**: 这段是 latent memory / medical VLM 主线：关注视觉证据如何进入 latent space、如何被记忆/更新/调用，以及是否能支撑可靠诊断。

Recently, latent reasoning has been extended to Multimodal Large Language Models (MLLMs). Distinct from text-only LLMs, MLLMs necessitate the effective integration of visual features within the latent reasoning space. Several efforts [30, 34, 43, 62, 69] have been dedicated to injecting visual cues into the latent space to facilitate visualgrounded reasoning. For instance, Wang et al. [62] emphasize image details by constructing a structured cognitive hierarchy, albeit relying on annotation-intensive multimodal reasoning data. Similarly, Liu et al. [34] progressively select visual patches to inject into latent thinking tokens via confidence-guided policy gradient optimization. In this work, we systematically analyze gradient dynamics during latent reasoning training and reveal two critical bottlenecks towards token-wise optimization behavior.

> 💡 **批注**: 这段是 latent memory / medical VLM 主线：关注视觉证据如何进入 latent space、如何被记忆/更新/调用，以及是否能支撑可靠诊断。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
