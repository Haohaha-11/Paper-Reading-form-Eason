[← 返回 README](../README.md)

# Method

## 📌 预览
本文件合并 Method/Background 等核心技术段落，重点看 latent memory/reasoning/alignment 的构造、读写和训练目标。

---

# 2.1. Explicit Multimodal Reasoning

Multimodal reasoning enables model to reason over information from different modalities to solve complex tasks. There are many prior works [15–17, 20, 36, 39, 40, 40, 41, 72, 73, 79] focusing extensively on enhancing reasoning capabilities. Earlier work rely on the CoT prompting to perform explicit thinking steps in the text space before generating the final answer. However, this paradigm generally lack sufficient visual grounding capability and leads to unsatisfactory misalignment and hallucination [3, 23]. To address these limitations, recent studies [20, 40, 79] have explored converting visual information into textual formats prior to reasoning, leveraging external tools or specialized visual experts to generate descriptive representations that guide LLMs. For instance, Hu et al. [20] pioneered the integration of visual captions, extracting semantic content as text and concatenating it with input prompts to bolster reasoning capabilities. To further enhance fine-grained reasoning, subsequent works have focused on regional understanding, aiming to improve textual expressiveness by describing specific image regions. Others [39–41] identified entities and their relationships within images, facilitating fine-grained reasoning through explicit modeling of inter-entity connections.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

A line of concurrent works advocates using vision-text interleaved format during the rationale generation and reasoning process. The model draws auxiliary lines or marks based on original image to record thinking path, zoom or crop regions, or perform code editing, etc. Building on these paradigms, Zhang et al. [77] first proposed decoupling rationale generation from answer generation in the Vision-Text Reasoning field. Subsequently, Shao et al. [45] annotated key regions of the original image in intermediate steps, training models to focus on image regions relevant to the answer. While some works [13, 75] further extract key image regions progressively during reasoning, combining visual information with textual reasoning to generate the final answer. Moreover, new methods [21, 35] emulated human thought by sketching images during reasoning, focusing on core concepts, structures, and relationships while ignoring redundant details. Other works [7, 32] generated new images with auxiliary markers during reasoning, combining them with text to improve reasoning in complex scenarios. To fully shift reasoning from the linguistic domain to the visual modality, Xu et al. [66] proposes to reasoning exclusively with dynamic generated images, demonstrating substantial performance gains in visual navigation tasks.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

More recently, Vision-R1 [24] and VL-Rethinker [54] leveraged Group Relative Policy Optimization [9] (GRPO) to refine reasoning trajectories through rollout-based sampling and reward scoring. Complementing these policy-driven approaches, concurrent works further enhance reasoning capabilities via novel cognitive paradigms, including selfcritiquing cycles [8, 44], iterative rethinking [68], and onpolicy distillation [78].

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

# 3. Method

# 3.1. Preliminary: Implicit and Explicit Decoding

Given a multimodal input comprising a question $\mathcal { Q }$ and an image $\nu$ , we first tokenize them into a sequence of text embeddings $\mathbf { Q } = \{ \mathbf { q } _ { i } \} _ { i = 1 } ^ { N _ { q } }$ and visual features $\mathbf { V } = \{ \mathbf { v } _ { i } \} _ { i = 1 } ^ { N _ { v } }$ via a word embedding matrix $\mathbf { E } \in \mathbb { R } ^ { | \mathcal { W } | \times d }$ and a pretrained visual encoder, respectively, where $| \mathcal { W } |$ denotes the vocabulary size and $d$ is the hidden dimension. Subsequently, we employ an autoregressive MLLM $\mathcal { F } _ { \theta }$ to encode the concatenated input into an initial hidden state ${ \bf H } ^ { ( 0 ) } \in \mathbb { R } ^ { P \times d }$ , where $P = N _ { q } + N _ { v }$ represents the total number of tokens in the prefilling phase. During the decoding phase, we predetermine the number of implicit reasoning steps $T _ { r }$ and explicit answer tokens $T _ { a }$ . The generation process consists of two following distinct stages.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Implicit Reasoning Phase. In the $t$ -th implicit reasoning step, the model generates a continuous latent representation $\mathbf { z } _ { t }$ conditioned on the original input sequence $\mathbf { X } = \left[ \mathbf { Q } \parallel \mathbf { V } \right]$ and all preceding latent states. This iterative process is formulated as:

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

![Equation 1](../images/c15c8a7dbd41f40455bedb7178d8167bf503c615bc227df5b6f36ee823d65f40.jpg)
*Equation 1: Equation extracted by MinerU.*

> 💡 **Equation 1 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\mathbf Z _ { < t } = \{ \mathbf z ^ { ( 1 ) } , \dots , \mathbf z ^ { ( t - 1 ) } \}$ denotes the sequence of latent tokens generated in previous steps, and $\tau ( \cdot )$ extracts the final vector representation from the model output (i.e., the hidden state corresponding to the last generated token). Unlike vanilla explicit reasoning, each step yields a informative continuous latent vector rather than a discrete token. The resulting latent sequence $\mathbf { Z } _ { 1 : T _ { r } } = \{ \mathbf { z } ^ { ( t ) } \} _ { t = 1 } ^ { T _ { r } }$ is concatenated to the context before transitioning to the explicit answer decoding phase.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Explicit Answer Decoding. In the explicit phase, the model generates discrete answer tokens by sampling from the vocabulary distribution. The conditional probability of the $t$ -th answer token $a _ { t }$ is given by:

![Equation 2](../images/0708040c17ebd8db627ccf41eeb42f3cecc1d136d01ffa072426e7a84b5df021.jpg)
*Equation 2: Equation extracted by MinerU.*

> 💡 **Equation 2 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\mathbf { h } _ { t } ^ { \mathrm { d e c } }$ is the decoder hidden state at step $t$ , and ${ \bf W } _ { o } \in $ $\mathbb { R } ^ { | \mathcal { W } | \times d }$ is the output projection matrix that maps hidden states to vocabulary logits.

The likelihood of the complete answer sequence $\mathbf { a } _ { 1 : T _ { a } }$ is factorized as an autoregressive product:

![Equation 3](../images/1c1c93a314999a51efd277fc242211e5eb27d69e55bf2405812d34798f9f3da3.jpg)
*Equation 3: Equation extracted by MinerU.*

> 💡 **Equation 3 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

Here, the full context input for the decoder is defined as $\mathbf { U } _ { t } = \left[ \mathbf { X } \parallel \mathbf { Z } _ { 1 : T _ { r } } \parallel \mathbf { a } _ { < t } \right]$ , where $\parallel$ denotes sequence concatenation along the token dimension.

# 3.2. Spatially-Coherent Finer Visual Replay

As mentioned earlier, we empirically reveal the gradient disparities between visual and textual tokens throughout the learning dynamics. Specifically, visual tokens consistently exhibit substantially larger gradient norms and fluctuations compared to textual tokens, indicating that visual representations remain under-optimized despite their critical role in multimodal reasoning. Motivated by these insights, we introduce the visual replay module to reinforce the engagement of visual cues via salient region detection, while enhancing fine-grained spatially-coherent perception capabilities via self-distillation supervision at each reasoning step.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Attention-Guided Region Focus. Several works [33, 76] have demonstrated that LLMs exhibit fundamental visual grounding capabilities. To identify visually salient regions, we aggregate attention weights across all transformer layers and attention heads to obtain a consolidated spatial focus map. Specifically, given the $l$ -th layer and the $h$ -th attention head, we compute the mean attention map $\bar { \mathbf { A } } ^ { ( t ) }$ at reasoning step $t$ :

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

![Equation 4](../images/27322e749e4c48792a7b1f0c5a29e1e7dadf7f5ba1fc2e63054c8fc4632dd474.jpg)
*Equation 4: Equation extracted by MinerU.*

> 💡 **Equation 4 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where A(l,h,t) ∈ RP (t)×P (t) denotes the attention matrix for layer $l$ and head $h$ at iteration $t$ $\begin{array} { r } { 1 \leq t \leq T _ { r } } \end{array}$ ), with $P ^ { ( t ) }$ representing the number of input tokens at iteration $t$ . Here, $L$ and $H$ represent the total number of layers and heads, respectively. To obtain token-level attention scores, we extract the attention distribution from the most recently generated token to all preceding tokens via column-wise summation, i.e., a(t)all = colsum(A¯ (t)) ∈ RP (t) . Subsequently, we extract only the visual token attention scores from $\mathbf { a } _ { a l l } ^ { ( t ) }$ using the image mask, denoted as $\mathbf { a } ^ { ( t ) } \in \mathbb { R } ^ { N _ { v } }$ , where $N _ { v }$ is the number of visual tokens. This visual attention vector effectively captures which visual tokens are most relevant to the current reasoning context.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

As visual focus evolves across reasoning steps, we iteratively select the top- $K$ attended visual tokens $\{ \mathbf { v } _ { i } ^ { ( t ) } \} _ { i = 1 } ^ { K }$ as visual latents, which are integrated with hidden states $\mathbf { z } _ { t }$ . To prevent redundant re-selection and promote diverse exploration, we maintain a visited token set V (t)visite d, ensuring comprehensive visual coverage:

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

![Equation 5](../images/071216ad9b39ab0c49e7d492ef650708de87d8838831640e0addd50c4125df5b.jpg)
*Equation 5: Equation extracted by MinerU.*

> 💡 **Equation 5 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\boldsymbol { \mathcal { T } ^ { ( t ) } }$ represents the indices of the $K$ visual tokens with the highest attention scores at step t, and V (t)visite d is updated after each selection. The original embeddings of the selected tokens $\mathbf { V } _ { \mathcal { T } ^ { ( t ) } } = \{ \mathbf { v } _ { i } \ | \ i \in \mathcal { T } ^ { ( t ) } \}$ are further weighted by normalized attention scores:

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

![Equation 6](../images/bc557e70a2a327e90d609893c05fc925023dc3a99f848cb27881876e4aabaaf1.jpg)
*Equation 6: Equation extracted by MinerU.*

> 💡 **Equation 6 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\mathrm { D i a g ( \cdot ) }$ constructs a diagonal matrix from a vector.

Spatially-Coherent Regularization. Although leveraging learned attention within Transformers provides explainable visual locations, it often suffers from limited spatial continuity due to the scattered nature of selected tokens and introduces noise associated with the attention sink phenomenon [65]. To mitigate these issues and enhance finegrained perception without external annotations, we introduce self-distillation supervision. This mechanism involves cropping the visual regions exhibiting spatial coherence, reencoding them, and supervising the visual latents with these high-fidelity features.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Specifically, we first search for a $W \times W$ sub-grid patch that maximizes the density of attended visual tokens. Formally, we find the optimal top-left corner $( r ^ { * } , c ^ { * } )$ within the valid grid bounds:

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

![Equation 7](../images/e64b5669f5fd965070d2bfbd2e0ed38c391b5fc1ab89b2c7181979fb41bc91b0.jpg)
*Equation 7: Equation extracted by MinerU.*

> 💡 **Equation 7 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

![Figure 2.](../images/b9a52a98a28fabf6368e5ab3db96524beb8ac11db360df696dca730e523b14a9.jpg)
*Figure 2.: Figure 2. Left Panel: Schematic illustration of our framework. Input images are encoded into visual tokens by a pretrained visual encoder and projected into a text-centric semantic space aligned with the LLM, while questions are tokenized by the text tokenizer. Our method enhances a standard latent MLLM with two synergistic components during the implicit reasoning phase: (1) Spatially-Coherent Finer Visual Replay (SCF-VR): Attention-guided selection identifies salient visual regions at each implicit step, generating fine-grained visual latents enhanced by spatially-coherent constraints. (2) Routing Depth Scaling (RDS): A lightweight learnable router adaptively allocates additional reasoning steps for high-difficulty tokens or latents. Refined hidden states and visual latents are interleaved and propagated to subsequent reasoning steps. The overall objective is jointly optimized via standard cross-entropy loss and self-distillation loss. Right Panel: Detailed architecture of the SCF-VR module and token-wise depth scaling mechanism. TokenSR denotes the token super-resolution module, and Pool is the average pooling.*

> 💡 **Figure 2. 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

where the density function $\mathcal { N } ( r , c )$ counts the visited tokens falling within the window $\begin{array} { r } { \mathcal { N } ( r , c ) = \sum _ { i \in \mathcal { T } ^ { ( t ) } } \mathbb { I } \big [ r \leq r _ { i } < } \end{array}$ $r + W , c \leq c _ { i } < c + W \big ]$ . Here, $( r _ { i } , c _ { i } )$ denotes the rowcolumn position of token $i$ on the $G \times G$ grid, and $\mathbb { I } [ \cdot ]$ is the indicator function. This greedy selection ensures the cropped region captures a coherent visual context rather than scattered details. Then, the selected window is projected to pixel coordinates in the original image via mathematical transformation, cropped, and resized to the standard encoder input resolution using bilinear interpolation. We then reencode this refined patch through the same visual encoder to obtain fine-grained representations $\{ \mathbf { f } _ { i } \} _ { i = 1 } ^ { N _ { v } ^ { \prime } }$ . A global pooling operation $\operatorname { P o o l } ( { \mathord { \cdot } } )$ is applied to yield a robust reference token uref:

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Equation 8](../images/22601ebbf93e51350c9ccb21ae09d33c628642cf67c916cf3527a6517238601a.jpg)
*Equation 8: Equation extracted by MinerU.*

> 💡 **Equation 8 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

Finally, we align the coarse-grained global token $\mathbf { b } ^ { ( t ) } =$ $\mathrm { P o o l } ( \mathbf { B } ^ { ( t ) } )$ with this high-fidelity reference using a lightweight token super-resolution module $\mathcal { F } _ { \mathrm { S R } } : \mathbb { R } ^ { D } \to \mathbb { R } ^ { D }$ We minimize the reconstruction error as follows:

![Equation 9](../images/2913b7fb28dda5128bfd4df2a4efbc57c62b4a4468b3713efe0188f2fd23badd.jpg)
*Equation 9: Equation extracted by MinerU.*

> 💡 **Equation 9 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

This supervisory signal enables the model to prioritize spatially coherent and semantically intact visual contexts during latent generation through self-distillation.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

visual latents, existing methods typically allocate uniform computational budgets across all tokens during contextual refinement. Our empirical analysis reveals a fixed-depth optimization dilemma, demonstrating that token representations exhibit heterogeneous optimization complexities during latent training, which necessitates adaptive reasoning depths. To address this limitation without modifying the pretrained VLM architecture, while effectively leveraging its inherent knowledge, we introduce a lightweight router that dynamically allocates additional reasoning steps exclusively to critical tokens at each iteration. Router Network. Let the input token sequence at the $t { \cdot }$ -th reasoning step be denoted as ${ \bf U } ^ { ( t ) } = [ { \bf X } \| { \bf B } ^ { ( 1 ) } \| { \bf z } ^ { ( 1 ) } \| \cdot \cdot \cdot \| { \bf B } ^ { ( t ) } \| { \bf z } ^ { ( t ) } ] \in \mathbb { R } ^ { P ^ { ( t ) } \times d }$ , where $P ^ { ( t ) }$ represents the total sequence length and $d$ is the hidden dimension. To facilitate the illustration of our depth scaling mechanism, we decompose the forward pass into layer-wise computations. Specifically, within the $l$ -th transformer layer, we first compute the intermediate feature $\mathbf { H } ^ { ( l , t ) } = f ( \mathbf { U } ^ { ( t ) } ) \in \mathbb { R } ^ { P ^ { ( t ) } \times d }$ , where $f ( \cdot )$ denotes the transformation of the $l$ -th transformer layer in LLM. Subsequently, a lightweight router network computes a scalar importance score for each token based on its corresponding hidden representation:

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

While the visual replay mechanism significantly enhances fine-grained grounding by introducing spatially-coherent

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

![Equation 10](../images/1e628710764009bb713787da12b93eb1643520607f527a90061b69d9d047adff.jpg)
*Equation 10: Equation extracted by MinerU.*

> 💡 **Equation 10 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

# 3.3. Routing Depth Scaling

where the $i$ -th element $s _ { i } ^ { ( l , t ) }$ quantifies the importance of the i-th token in the l-th layer, and F (l)route $\mathcal { F } _ { \mathrm { r o u t e r } } ^ { ( l ) } ( \cdot )$ denotes the independent router network in the $l$ -th layer. We define $T _ { \alpha } ( \mathbf { s } ^ { ( l , t ) } )$

as the index set of the top- $\alpha$ tokens with the highest scores, where $\alpha$ serves as a predefined hyper-parameter. In practice, $\alpha$ can be formulated as a layer-dependent function $\alpha ( l )$ to enable adaptive resource allocation at different layers.

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

Depth Scaling Computation. Obtaining the router weight, we select the top- $\alpha$ tokens for depth scaling, i.e., repeat iteration in one transformer block. For a given depth scaling step $d \in \{ 1 , \ldots , D \}$ , the token-wise representation update rule within a transformer layer can be generally formulated as follows:

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

![Equation 11](../images/298844f0824d4459ec2a84ed163843f0cd2f2b415acf0f5e11c10de043d56f7f.jpg)
*Equation 11: Equation extracted by MinerU.*

> 💡 **Equation 11 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

For notational brevity, we omit the layer index $l$ and reasoning step $t$ in the following formulation, where $d$ denotes the refinement depth. For instance, the score vector $\mathbf { s } ^ { ( l , t ) }$ is simplified to $\mathbf { s } ^ { ( d ) }$ . The function $f _ { i \in T _ { \alpha } ( \mathbf { s } ^ { ( d ) } ) } ( \cdot )$ indicates that the attention operation is restricted to the token subset $T _ { \alpha } ( \mathbf { s } ^ { ( d ) } )$ Here, ${ \bf h } _ { i } ^ { ( 0 ) }$ represents the initial hidden state of the $i$ -th token after the first forward pass, and m(d) ∈ RP (t)×P (t) denotes the attention mask corresponding to refinement step $d$ . The condition $i \in T _ { \alpha } ( \mathbf { s } ^ { ( d ) } )$ functions as a binary gating mechanism, ensuring that only tokens with importance scores exceeding the threshold undergo additional computation.

> 💡 **批注**: 这是因果/反事实修正：关注它如何剔除伪相关、突出关键病灶或视觉证据。

The router network dynamically determines whether to apply depth scaling based on the current contextual representations in a data-driven manner. Tokens identified as critical are iteratively processed through the transformation function for $d$ additional refinement steps, while non-critical tokens retain their prior representations to preserve computational efficiency. Finally, we aggregate the depth-wise refined representation with step-aware positional encoding to form the final hidden state incrementally:

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Equation 12](../images/9e95891a06271606b493993c16b3472658fe10ae26875be9c6ecefa912c0896a.jpg)
*Equation 12: Equation extracted by MinerU.*

> 💡 **Equation 12 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

This adaptive scaling strategy effectively allocates deeper reasoning pathways to critical tokens, enabling the model to capture complex visual contexts and facilitate more profound reasoning capabilities.

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

# 3.4. Training Procedure

Curriculum Latent Training. To mitigate annotation overhead and avoid the risk of human priors constraining model learning, we depart from prior approaches that rely on intermediate supervision for latent representations. Instead, we design a curriculum that facilitates the contextual and logical dependencies between implicit latents and explicit reasoning chains. In the initial stage, the model is trained with standard Chain-of-Thought (CoT) supervision, generating all reasoning steps explicitly to establish foundational reasoning capabilities. Subsequently, as training progresses, latent tokens are incrementally introduced. Specifically, one explicit reasoning step is progressively encapsulated into an informative ⟨latent⟩ token. Through this curriculum, each latent token progressively learns to ground the relevant contextual cues, effectively internalizing explicit reasoning chains into compact latent representations.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Training Objective. The overall training objective combines the standard language modeling loss with the self-distillation loss in the VR-SCF module,

![Equation 13](../images/dc8350622edd27670032fd6f475d80db3c740bb56fbb55cd20a161561060d99b.jpg)
*Equation 13: Equation extracted by MinerU.*

> 💡 **Equation 13 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\mathcal { L } _ { \mathrm { C E } }$ is the standard cross-entropy loss over the language modeling task,

![Equation 14](../images/08bd42b3c6459beb91593d5ce8ca9604a0822acd0648dcdda2469fbe436a3c4c.jpg)
*Equation 14: Equation extracted by MinerU.*

> 💡 **Equation 14 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\lambda$ is a hyperparameter balancing the primary task and the auxiliary reconstruction objective. During inference, only the LLM component is active, and the visual play module is disabled, ensuring no additional computational overhead at test time.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

---

## 🔖 Section 总结

### 核心洞察
1. 明确 latent/memory/alignment 的读写路径。
2. 区分训练时组件和推理时组件。
3. 关注是否能迁移到医学影像。
