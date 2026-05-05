[← 返回 README](../README.md)

# Method

## 📌 预览
本文件合并 Method/Background 等核心技术段落，重点看 latent memory/reasoning/alignment 的构造、读写和训练目标。

---

# 2 Foundation: What is Latent Space? 5

2.1 Concept 5   
2.2 Comparison with Explicit Space 6   
2.2.1 Representational Properties 6   
2.2.2 Functional Capabilities 7   
2.3 Comparison with Generative Visual Models 8

# 3 Evolution: How Did Latent Space Develop? 9

3.1 Prototype 9   
3.2 Formation 10   
3.3 Expansion 11   
3.4 Outbreak 12

# Mechanism: How Does Latent Space Work? 13

# 4.1 Architecture 15

4.1.1 Backbone 15   
4.1.2 Component 16   
4.1.3 Auxiliary Model 19

# .2 Representation 20

4.2.1 Internal 21   
4.2.2 External . 24   
4.2.3 Learnable 25   
4.2.4 Hybrid . 26

# .3 Computation 27

4.3.1 Compressed 27   
4.3.2 Expanded 29   
4.3.3 Adaptive 30   
4.3.4 Interleaved 32

# .4 Optimization 33

4.4.1 Pre-training . 33   
4.4.2 Post-training 34   
4.4.3 Inference 36

# Ability: What Does Latent Space Enable? 37

5.1 Reasoning 37   
5.2 Planning . 39   
5.3 Modeling 40   
5.4 Perception 41   
5.5 Memory 42   
5.6 Collaboration 42   
5.7 Embodiment 43

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

# 44

6.1 Perspective 45   
6.2 Challenge 46   
6.3 Future 46

![Figure 1](../images/f9229a060cdfd418c53e3c2a43ba32673af5d33928ac8e4b0bf74a692cfb343b.jpg)
*Figure 1: Figure 1 Overview of the latent space methods classified by two axes: four main Mechanisms (Section 4) and seven key Abilities (Section 5). Within our classification system, a single method may be affiliated with one or more mechanisms and capabilities. For the visualization in this figure, we adopt the most appropriate classification for each method; a comprehensive elaboration of these categories will be presented in the main text.*

> 💡 **Figure 1 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Figure 1 Overview of the latent space methods classified by two axes: four main Mechanisms (Section 4) and seven key Abilities (Section 5). Within our classification system, a single method may be affiliated with one or more mechanisms and capabilities. For the visualization in this figure, we adopt the most appropriate classification for each method; a comprehensive elaboration of these categories will be presented in the main text.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# Contributions

• We clarify the conceptual scope of latent space in language-based models, distinguishing it from explicit or verbal space and from the latent spaces commonly studied in generative visual models. • We provide a unified review of how latent space has evolved from early latent reasoning into a broader multimodal and systems-level research paradigm. • We introduce a two-dimensional taxonomy across Mechanism and Ability, offering a common framework for organizing otherwise fragmented methods and applications. • We provide a comprehensive collection of resources, including illustrative figures, structured tables, accessible links, and repositories, to facilitate further research and community engagement.

> 💡 **列表批读**: 这组条目通常对应贡献、设置、发现或模块；建议逐条映射到 visual memory / latent reasoning / medical diagnosis 的主线。

# 2 Foundation: What is Latent Space?

As an emerging paradigm, the exploration of the latent space within language models has exhibited immense potential and vast room for further development. Accordingly, this section provides a preliminary elaboration covering its formal definition and comparisons with existing paradigms.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 2.1 Concept

In language-based models, i.e., classic autoregressive models, encompassing LLMs, VLMs, VLAs, and derivative systems with language backbone, the entire operational process unfolds explicitly within the explicit space, or the verbal space [202]. The most immediate and externally observable domain is: the discrete space of linguistic symbols in which inputs and outputs are expressed. Formally, this space is defined by a vocabulary that specifies the language model’s next-token predictions. In this space, language is represented as overt verbalized units, i.e., subword-level tokens [168], that are directly interpretable by humans. The training objective of a language model is typically formulated in this explicit space, since the model learns to assign probabilities to symbol sequences and to predict the next token given a textual prefix.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

However, a language-based model does not operate solely on discrete symbols. To compute over language, it first maps tokens into internal continuous representations and transforms them through multiple layers of nonlinear computation. This gives rise to what is commonly called the latent space: the continuous, learned representational space in which the model encodes and manipulates information that is not explicitly verbalized at the token level. More precisely, the latent space of a language model can be understood as the family of hidden-state spaces, within which contextual, semantic, syntactic, and relational features of an input are jointly represented in this space. A token sequence in the explicit space is thus mapped to a trajectory of latent space, and these latent states are subsequently projected back into the verbal space to yield a probability distribution over possible next tokens. Furthermore, this latent space could be expanded into a

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

![Figure 3](../images/d55eb0f90fba9b15f3a11eadf6011d9e7346b69709858f22582458e9a328fa0e.jpg)
*Figure 3: Figure 3 Comparison of the explicit space and latent space of the language models, including their representational properties and functional capabilities.*

> 💡 **Figure 3 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Figure 3 Comparison of the explicit space and latent space of the language models, including their representational properties and functional capabilities.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

unified space beyond language that maps modality-specific inputs into continuous internal representations.   
We further provide a formal definition and formulation in Section 4.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

# 2.2 Comparison with Explicit Space

To further elucidate the unique characteristics of the latent space, we conduct a comparative analysis with the traditional explicit space, or verbal space, across several critical dimensions [141, 231, 232]. As demonstrated in Figure 3, this comparison highlights a paradigm shift in the representational properties and functional capabilities of the two language models.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 2.2.1 Representational Properties

The two distinct paradigms of language models, i.e., explicit space and latent space, exhibit fundamental differences in their representational forms, information processing modes, and practical performances. The latent space uses a Machine-native vectorized representation that is Continuous, Flexible, Efficient, and capable of preserving High-fidelity semantic information.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Human-readable v.s. Machine-native. In the explicit space, every state is expressed as a sequence of natural language tokens drawn from a finite vocabulary. This endows the generated trajectories with direct human readability and verifiability [219, 254].

By contrast, the latent space is a machine-native representational paradigm that lacks direct human legibility. Its core representations are high-dimensional real-valued vectors, where individual dimensions do not have a straightforward correspondence to human-interpretable semantic, structural, or perceptual features [148, 239]. Tailored for the intrinsic operational logic of language models, the latent representation and operations can be processed by the model, without the need for transformations of human-readable signals, reducing computational redundancy from additional encoding/decoding overhead [58, 167, 273].

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Discrete & Symbolic v.s. Continuous $\&$ Flexible. Explicit-space representations are redundant, discrete, and symbolic. For instance, a chain-of-thought trace for a complex reasoning problem may span thousands of tokens [219]. Each token is a discrete symbol whose meaning is grounded in linguistic convention, and the relationship to other tokens is expressed through positional and grammatical structure [92, 238]. This decoding pattern is, in part, a redundancy and inflexibility: the majority of tokens serve textual coherence rather than substantive semantic contents [40, 234]. They ensure grammaticality, maintain topic continuity, and satisfy the autoregressive constraint that each token be a plausible continuation of the preceding sequence and be mapped into a finite discrete token, obligations that have no intrinsic connection to the logical structure of the autoregressive generation [35, 138, 239].

> 💡 **批注**: 这是一段信息密度较高的论述：建议拆成“现象/问题 → 机制 → 证据/影响”三层读。

Conversely, latent space exhibits inherent continuity and flexibility. It encapsulates core semantic information in a continuous form, discarding superficial linguistic redundancies and symbolic mapping [58, 174, 294]. Liberated from the constraints of discrete tokenization and autoregressive linguistic conventions, this nature confers upon latent space distinct advantages in bolstering representational capacity: in reasoning models, it elevates explicit reasoning trajectories onto continuous manifolds [181, 298]; in visual scenarios, it enables smoother multimodal operations (e.g., fusion, alignment, and interaction) within latent spaces marked by a narrower modality gap [183, 227, 264]; in collaboration, it inherently serves as a more fitting medium for information storage and transmission [48, 300]. In essence, the continuous and flexible properties enable language-based models to prioritize the intrinsic semantic essence, unlocking potential across diverse tasks.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Inefficient v.s. Efficient. Conventional autoregressive generation operates entirely within an explicit discrete space, and this paradigm inherently introduces at least three fundamental inefficiencies [233, 277]: First, linguistic redundancy: as the main medium of explicit space, natural language introduces unavoidable structural and semantic redundancy, further reducing the efficiency [35, 92]; Second, representational transformation inefficiency: the model is forced to transfer representations through a narrow, explicit channel at every generation step, including intermediate steps [107, 153]. This mandatory conversion introduces unnecessary and high representational conversion costs. Third, sequential decoding overhead: discrete tokenization locks generation into a strictly sequential pipeline: each token requires a full model forward pass and vocabulary-wide probability calculation. This sequential paradigm not only has low computational efficiency but also increases computational burden [4, 94]; In contrast, latent space methods bypass these inefficiencies, e.g., removing mandatory representation conversion in inter-agent communication [48, 290, 300], and efficient recurrent or looped computation patterns [50, 167].

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Semantically-lossy v.s. High-fidelity. Explicit-space representations are prone to semantic loss. When a model externalizes its internal continuous state as a token sequence, the mapping from latent activations to discrete symbols imposes a quantization bottleneck: a finite vocabulary and the combinatorial constraints of natural language delimit what can be expressed [159, 221]. As a result, fine-grained uncertainty, intermediate computational traces, cross-modal alignments, and other structures that are difficult to render in language may be compressed, distorted, or discarded. In this sense, natural language constitutes a semantically lossy transformation of the underlying computational state.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Latent-space representations, by contrast, preserve information with higher fidelity. By avoiding discretization and linguistic rendering, latent variables can carry rich, continuous information between computational steps, including content that is inexpressible in natural language and representations that naturally support multimodal structure. This perspective motivates a growing line of work directly in latent space, such as continuous thoughts [58, 174, 294], latent memory [65, 264, 273], and latent visual reasoning [95, 251], etc.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 2.2.2 Functional Capabilities

The latent space, as a machine-native representational space, possesses multiple key functional capabilities that distinguish it from the explicit space, including Operability, Expressiveness, Scalability, Generalization, as well as Evaluability, controllability, and interpretability, which collectively underpin its utility in various advanced computational and representational tasks.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Operability. As a machine-native representational space, the operability of latent space characterizes its utility as a structured, differentiable manifold that serves as several internal computational substrates. This operability seamlessly enables direct calculations (such as concatenation, linear combination, etc.) and also advanced operations, e.g., controllable semantic steering [84, 250], active intervention [49, 98, 295], iterative interleaving [20, 111], and visual latent thinking [183, 211, 276]. On the contrary, the discrete tokens in the explicit space are inherently non-differentiable and lack the support of a continuous, structured manifold, rendering the aforementioned fine-grained, advanced semantic operations infeasible and permitting only limited, indirect token-level operations.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Expressiveness. It serves as a core capacity for internalizing and manipulating complex, high-dimensional, and even non-linguistic information. In contrast to natural language, which is constrained by a discrete vocabulary and grammatical conventions, representations within latent space provide a substantially richer representational substrate. In principle, it can express the representation including but not limited to the whole explicit space, enabling efficient communication [252, 290, 300], visual perception [11, 28, 251], embodied action planning [14, 69], latent memory formation [264, 273], etc.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Scalability. It follows naturally from the compactness and parallelizability of vectorized representations. Latent-space approaches are therefore well-positioned to benefit from continued scaling of longer reasoning trajectories [117, 164, 188, 220], deeper agent interaction turns [265, 290], and test-time compute [50, 262, 274].

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

Generalization. It further bolsters the ability to generalize effectively to inputs distinct from those encountered during training, which capture abstract semantic structures rather than superficial linguistic patterns. By embedding abstract semantic concepts into a latent space, models gain improved cross-domain transfer and zero-shot generalization, enabling the transfer of learned abstractions to previously unseen tasks and domains. Transfer learning [195, 270], and cross-domain robustness [37, 177, 264, 273] have all been empirically shown to benefit from such informative latent representations.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Evaluability $\&$ Controllability & Interpretability. This denotes the ability of humans or automated systems to evaluate, control, observe, interpret, and audit the autoregressive generation process. For explicit generation, the resulting generation traces are evaluable, controllable, and interpretable, as each intermediate step is represented in a human-readable format [6, 89, 108]. In principle, systems built upon explicit reasoning mechanisms can integrate human-aligned verification or automated consistency checks. In contrast, machine-native latent space representations make it inherently difficult for humans to perform granular, direct evaluation, control, and interpretation of the generation process, posing potential challenges to model evaluability, controllability, and interpretability (Section 6.2).

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 2.3 Comparison with Generative Visual Models

The latent space of generative visual models is derived from a VAE-style framework [85], which learns a probabilistic mapping from high-dimensional observations to a compact continuous representation. Subsequent work introduced discrete latent codes via VQ-VAE [201], and the decisive step toward scalable synthesis was taken by latent diffusion models [162], which demonstrated that diffusion processes operating in a perceptually compressed latent space could achieve both computational efficiency and high sample fidelity. Extensions to video generation, including VideoLDM [12] and related architectures, further organized this space along a spatiotemporal axis, encoding appearance and motion as jointly structured representations.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Despite the shared reliance on learned continuous representations, the latent spaces of generative visual models and large language models differ fundamentally in their geometric structure, representational organization, and conditioning regimes.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Training Objective. The latent space of generative visual models is explicitly shaped by a reconstruction objective, which anchors the learned geometry to the statistical structure of the visual signal [12, 200]. This produces a relatively smooth, locally Euclidean manifold in which linear interpolation between encoded points yields perceptually coherent intermediates, and in which distances carry interpretable perceptual meaning. Rather than a reconstruction objective, language model hidden states are organized by a predictive criterion, next-token prediction in autoregressive models [13], with no explicit constraint on the geometry of the space.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Structure. Latent space of visual generative models maintains an explicit spatiotemporal structure: the latent tensor of an image is a spatial grid of patch-level codes [43, 150], and that of a video extends this grid along the temporal axis, with motion treated as a first-class representational element. This structural regularity reflects the continuous, compositional nature of visual scenes, in which spatial proximity and temporal coherence are strong inductive priors. On the contrary, the latent space of a language model focuses on the linguistic semantics and is devoid of spatial topology or physical dynamics.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Controllability. Precise control over visual generation is exercised through dedicated architectural pathways integrated into the model itself [278]. These signals include pose sequences, depth maps, segmentation masks, and reference images, affording fine-grained, spatially localized control over the generated output by conditioning applied directly within the representational substrate.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

![Figure 4](../images/544aebe14bb9363d4ed1474f1b3309212638dfe34c1a4ff00230110f581645bc.jpg)
*Figure 4: Figure 4 Timeline of representative works in the evolution of latent space research, organized into four developmental stages: Prototype (Section 3.1), Formation (Section 3.2), Expansion (Section 3.3), and Outbreak (Section 3.4) stages, where the horizontal axis denotes the month, and vertical axis indicates the number of the latent-level works.*

> 💡 **Figure 4 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Figure 4 Timeline of representative works in the evolution of latent space research, organized into four developmental stages: Prototype (Section 3.1), Formation (Section 3.2), Expansion (Section 3.3), and Outbreak (Section 3.4) stages, where the horizontal axis denotes the month, and vertical axis indicates the number of the latent-level works.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 3 Evolution: How Did Latent Space Develop?

As large language models have demonstrated outstanding performance across a range of natural language understanding and reasoning tasks, the exploration of their latent spaces has undergone a remarkably rapid and transformative evolution. In this section, we trace the developmental trajectory of this emerging paradigm, organizing the literature into four chronologically and thematically coherent stages (Figure 4): (1) Prototype (Previous – Mar 2025), where pioneering works first demonstrated the feasibility of moving reasoning from discrete token sequences into continuous latent representations; (2) Formation $\left( \mathrm { A p r \mathrm { ~ - ~ } J u l ~ 2 0 2 5 } \right)$ , where theoretical foundations were established and systematic evaluations emerged, with research primarily centered on textual latent reasoning and initial explorations into multimodal settings; (3) Expansion (Aug – Nov 2025), where the field rapidly diversified into visual tasks, embodied action, multi-agent collaboration, and other application paths; and (4) Outbreak (Dec 2025 – Present), where explosive growth drove architectural innovation, advanced optimization, rigorous theoretical analysis, and cross-modal integration toward maturity.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 3.1 Prototype

The prototype stage marks the genesis of latent space reasoning, where researchers first question whether large language models must articulate every intermediate reasoning step in natural language. In this stage, Theory Validation and Early Exploration begin to use continuous representations as an alternative substrate.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Theory Validation. Before the first latent systems appeared, several works laid essential groundwork by revealing that reasoning-like behavior is already encoded in the internal representations of language models. Represented by HCoT [122], early precursors show that a full CoT trace can be compressed into a compact special-token representation via contrastive semantic alignment, suggesting that much of the information in explicit CoT is redundant for the model itself. Meanwhile, Zhang and Viteri [277] discovers latent thinking vectors by extracting steering vectors from model activations, and demonstrates that injecting these vectors at inference time can elicit CoT-like reasoning without any fine-tuning or explicit prompting. From a theoretical perspective, Hu et al. [66] offers a Hopfieldian interpretation, framing reasoning as transitions across representation spaces with grounding in cognitive neuroscience. Furthermore, Latent Space Chainof-Embedding [217] demonstrates that LLMs can perform self-evaluation through latent embeddings rather than explicit verbal outputs. Together, these precursors establish a critical insight: the capacities of language models are not fundamentally bound to discrete token sequences, but are already substantially encoded in their latent spaces.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Early Exploration. Building on these insights, the initial stage produced a series of high-impact works that gradually shaped the initial direction of latent space reasoning. COCONUT [58] proposes the first complete framework for reasoning in continuous latent space, feeding the last hidden state back as the next input embedding to form a loop of continuous thoughts that bypasses the discrete vocabulary bottleneck. Its key finding is that continuous thought vectors can encode superpositions of multiple potential next steps, enabling emergent breadth-first search in latent space. Along a similar compression-based direction, CCoT [31] introduces contemplation tokens that compress explicit reasoning chains into dense latent form. At the same time, Liu et al. [117] augments the KV cache with latent embeddings via an offline coprocessor, keeping the decoder entirely frozen. Both demonstrate that latent representation can be injected into existing models through parameter-efficient adaptation without inference latency overhead.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Subsequently, Huginn [50] proposes using recurrent depth to scale test-time compute in latent space, iterating a shared transformer block a variable number of times to perform all reasoning implicitly without specialized reasoning data. SoftCoT [243] provides the first plug-in approach to latent space, projecting instance-specific soft thought tokens into the frozen backbone’s representation space to avoid catastrophic forgetting.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Taken together, the prototype stage establishes the feasibility of latent-space reasoning, but it also exposes its first major bottleneck: the field still lacks a systematic account of why latent reasoning works, when it outperforms explicit CoT, and how it should be evaluated beyond isolated proof-of-concept demonstrations. In other words, the central challenge at this stage is not capability alone, but interpretability, formalization, and comparability. These limitations directly motivate the next stage, where the community moves from scattered prototypes toward theoretical systematization, benchmark construction, and more principled technical design.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 3.2 Formation

The formation stage advances from the prototype-stage insights toward building a more practically meaningful research program. This period is primarily centered on textual latent reasoning and is characterized by two advances: Theoretical Systematization that explains why it works, Technical Formation that works on how well it works, and initial explorations of extension into multimodal and embodied settings.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Theory Systematization. The most transformative contribution of this stage is the theoretical understanding of continuous thought mechanisms. These theoretical works provide formal guarantees for the expressiveness and computational advantages of latent space, grounding the empirical successes of the prototype stage in rigorous mathematical frameworks. Zhu et al. [294] (Reasoning by Superposition) provide the first formal complexity analysis, proving that continuous thought vectors acting as superposition states can encode multiple search frontiers simultaneously, offering a rigorous explanation for COCONUT’s empirically observed behavior. CoT2 [52] further quantifies the relationship between parallelism and embedding dimension and introduces continuous supervision and reinforcement learning for continuous thought optimization. Saunshi et al. [167] prove that looped transformers with latent iterations can express strictly more complex computations than standard transformers, establishing theoretical separation results for recurrent-depth architectures.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Technical Formation. The formation stage also witnesses important methodological innovations that translate the prototypes into more practical and versatile latent techniques. These advances span both representation design and optimization strategies, progressively expanding the toolkit available. On the representation front, a series of works explored latent representation design: Assorted [181] proposes mixing latent discrete tokens with text tokens to achieve shorter traces with improved accuracy; CODI [174] formalizes a self-distillation procedure where the same model serves as both teacher and student to generate reasoning chains in explicit and latent spaces respectively; and BoLT [164] shifts attention to pretraining, treating web text as compressed outcomes of thought processes and using bootstrapping for data-efficient pretraining.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

On the optimization front, represented by HRPO [266], System-1.5 [214], and CoLaR [188], a family of methods respectively introduced latent reinforcement methods, adaptive computation allocation between language and latent spaces, and dynamic inference-time controllable compression. These works demonstrate that latent optimization can be optimized end-to-end and adaptively allocated across different modalities.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Another characteristic of this stage is that latent-level models begin extending beyond text-only LLMs. While the primary focus remains on textual reasoning, several pioneering efforts demonstrate the broader applicability of the latent space paradigm across modalities. Mirage [251] enables VLMs to think visually by recasting hidden states as latent visual tokens interleaved with text, implementing internal visual manipulation analogous to human mental imagery. UniVLA [14] introduces task-centric latent actions for cross-embodiment robot policies, learning latent action representations from internet-scale video. These early multimodal explorations demonstrate that the latent space paradigm is not confined to textual reasoning but may constitute a general framework applicable across modalities and embodiment types.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Despite these advances, the formation stage remains constrained by a second bottleneck: most methods are still developed and evaluated within relatively narrow settings, with text-centric assumptions, limited downstream diversity, and weak integration across memory, planning, communication, and action. As a result, the field has learned how to make latent reasoning more principled, but not yet how to make it broadly useful across heterogeneous scenarios. This gap naturally drives the next stage, where the main question shifts from “can latent reasoning be formalized?” to “how far can the latent-space paradigm generalize across domains, modalities, and system settings? ”

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 3.3 Expansion

The expansion stage transforms latent space research from a text-centric landscape into a multi-modal, multi-domain ecosystem. This stage witnesses rapid diversification along concurrent dimensions. The period is marked by the Technical Maturation of domain-specific innovations and Paradigm/Scenario Expansion of cross-cutting themes such as latent memory, test-time scaling, and RL-based optimization.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Technical Maturation. In the LLM domain, latent methods evolve from proof-of-concept to more sophisticated systems that address challenges in memory, scalability, optimization, and domain-specific applications. The research landscape broadens considerably, with multiple sub-directions developing in parallel. Represented by MemGen [273], a line of works pioneers latent memory for agents, interweaving reasoning and memory so that planning, procedural, and working memory types emerge without explicit supervision. On the test-time scaling front, LTPO [258] treats latent thought vectors as optimizable parameters with online policy gradient; Ouro [296] moves optimization into pretraining via looped language models; and You et al. [262] enables parallel test-time scaling through stochastic sampling strategies with a latent reward model for trajectory selection. For RL-driven optimization, SofT-GRPO [291] solves the differentiability challenge of applying RL to continuous latent reasoning through Gumbel-reparameterized policy optimization. In the interleaving and hybrid direction, SpiralThinker [155] and CLaRa [59] propose text-latent iterative interleaving and unified retrieval-augmented generation in shared continuous spaces, respectively. Additionally, domain-specific applications emerge, including search-and-recommendation unification [177], code language model interpretability [172], and System $1 / 2$ dual-architecture communication [33].

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Paradigm/Scenario Expansion. The expansion stage sees an explosion of visual latent-level methods, establishing the paradigm of thinking in visual space as a complement to textual reasoning. These methods enable VLMs to perform internal visual manipulation and reasoning directly in latent representations, bypassing the information loss incurred by converting visual content into discrete text tokens. Represented by LVR [95] and Monet [211], a family of works introduces autoregressive reasoning in the visual embedding space, generating latent states interleaved with text for fine-grained visual understanding. 3DThinker [28] further extends this to 3D mental simulation from limited 2D views by aligning latent representations with 3D foundation models. Another line address latent visual tasks: VisMem [264] proposes cognitively inspired short-term and long-term latent vision memory modules, while CoMEM [230] scales continuous memory for visual agents. Works such as Latent Sketchpad [276] and LaCoT [183] further explore visual scratchpads for planning and variational inference for visual reasoning, respectively.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

The expansion stage gives birth to latent communication as a new paradigm for multi-agent systems. Unlike traditional text-based inter-agent communication, latent communication enables direct exchange of continuous representations, offering higher bandwidth and lower latency. C2C [48] introduces direct semantic communication between LLMs via KV-cache projection and fusion. [290] develops a theoretical framework for mind-to-mind latent thought communication, while LatentMAS [300] demonstrates latent collaboration through shared latent working memory, significantly reducing output tokens while improving accuracy.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

The embodied domain also expands significantly during this stage, with latent representations becoming a central tool for learning and transferring robotic manipulation and navigation skills. Self-supervised pretraining from unlabeled video emerged as a key methodology. Represented by LAPA [256] and LAWM [196], a line of self-supervised methods pioneers latent action pretraining from unlabeled video data through world modeling. OccVLA [118] integrates implicit 3D occupancy supervision for interpretable trajectory planning, while

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

SRPO [45] applies RL with latent world representations for VLA training. ATE [281] enables data-efficient VLA adaptation through unified latent guidance, achieving substantial real-world cross-embodiment gains.

However, rapid expansion also introduces a new bottleneck: the field becomes increasingly fragmented. Once latent-space methods spread across language, vision, multi-agent systems, and embodiment, the main challenge is no longer lack of applications, but lack of unification. Different works adopt different architectural assumptions, optimization objectives, evaluation criteria, and latent interfaces, making it difficult to compare methods or identify stable design principles. This fragmentation sets the stage for the outbreak period, in which the research frontier shifts toward architectural specialization, optimization sophistication, and more mature attempts to consolidate latent space as a first-class computational paradigm.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 3.4 Outbreak

The outbreak stage represents an explosive acceleration of the field, characterized by the All-roundOutbreak of all research threads. The maturity of this stage is reflected in many hallmarks: architectural and representational specialization, with purpose-built models designed specifically for latent representation; computation and optimization sophistication, with methods addressing fine-grained challenges such as exploration collapse and latent reward encoding; Moreover, multi-scenario surge, with unified frameworks spanning language, vision, action, and multi-agent systems.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

All-round Outbreak. A defining feature of this stage is the increasing specialization of model architectures. Rather than merely adapting standard transformer backbones via shallow recurrence or iterative decoding, recent work has developed architectures explicitly designed to support latent computation as a first-class mechanism. These models generally aim to improve the controllability, efficiency, and expressiveness of reasoning in latent space.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Representative examples include Dreamer [86] and LoopFormer [72], which introduce depth-recurrent designs that combine sequence attention with depth-wise computation and elastic looping, thereby enabling budgetaware reasoning and more flexible compute allocation. MLRA [121] further revises the attention mechanism through low-rank projections, while DLCM [158] shifts the granularity of latent computation from token-level operations toward concept-level reasoning with adaptive conceptual boundaries. Collectively, these studies suggest that architectural development is moving beyond incremental modifications to sequence models toward dedicated latent-space systems.

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

Alongside architectural specialization, optimization strategies for latent space also become substantially more sophisticated. For example, ReLaX [280] and Active Latent Planning [292] advance latent-space exploration by moving beyond imitation-based learning toward more explicit reinforcement-learning-based planning. At the same time, Özeren and Aßenmacher [149] demonstrate, through a systematic analysis, that reinforcement learning remains sensitive to design choices and continues to face persistent optimization challenges. LED [189] addresses post-training exploration collapse by leveraging entropy variation across recurrent depth. In contrast, Latent Thinking Optimization [41] shows that latent thoughts can themselves encode reward-relevant information, thereby opening the possibility of directly optimizing latent trajectories without relying exclusively on external reward models. Overall, these developments indicate that optimization is becoming an independent axis of progress rather than a secondary concern.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

In visual tasks, recent studies demonstrate that latent space supports increasingly complex forms of multi-step and interleaved inference. ILVR [39] and CrystaL [283], for instance, explore visual-text interleaved reasoning and report the emergence of visual latent representations during the reasoning process. LIVR [100] and Mull-Tokens [160] further extend this line of work by pushing visual reasoning more fully into latent space and by proposing modality-agnostic latent thinking mechanisms. Related efforts such as VL-JEPA [21] and DMLR [111] further expand the design space of visual reasoning and highlight the growing diversity of methodological formulations in this area.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

A similar expansion is also evident in multi-agent systems, where latent communication evolves from preliminary demonstrations into more structured frameworks for coordination and representation sharing. Dery et al. [38] propose latent-space communication via K-V cache alignment with lightweight adapters, enabling translation between heterogeneous internal states. L2-VMAS [265] and Wormhole [124] extend this perspective to visual and heterogeneous multi-agent settings, while LatentMem [47] introduces shared latent memory mechanisms for multi-agent experience accumulation. These works suggest that latent communication is becoming an important mechanism for scalable inter-agent coordination.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Table 1 General notations used throughout the paper.

![Table 1](../images/69d7877f0c39d762a51d71c83fca7591bb0e6c234c9213d4b72c456af538a1f8.jpg)
*Table 1: Table 1 General notations used throughout the paper.*

> 💡 **Table 1 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

The embodied VLA setting provides another major area of expansion. Here, latent representations increasingly serve as a unifying interface for perception, generation, and action, and latent action modeling is emerging as a central design paradigm. Motus [10] and VLA-JEPA [184] exemplify this trend by developing unified latent action world models that integrate action generation and environment understanding within a shared latent space. Villa-X [26] and JALA [131] further improve the expressiveness and scalability of latent action modeling, while CoWVLA [249] introduces world-model reasoning in latent motion space. Additional work, including WholeBodyVLA [77], SwiftVLA [142], and LoLA [213], reinforces the view that latent action representations are becoming increasingly central to VLA pretraining and deployment.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

At the same time, the outbreak stage surfaces the next-order bottleneck for the field as a whole: once latentspace methods become specialized, powerful, and widespread, the hardest problem is no longer demonstration but consolidation. The open questions now concern standardization of latent interfaces, principled evaluation across modalities, alignment between latent efficiency and interpretability, and the integration of latent computation into broader agentic systems. In this sense, the outbreak stage does not mark the end of the story; rather, it reveals that the next phase of progress will likely depend on turning a rapidly growing collection of techniques into a coherent science and harness engineering for latent computation.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

# 4 Mechanism: How Does Latent Space Work?

Mechanism concerns how latent space is instantiated, structured, and operationalized within a model, beyond the high-level question of why it is useful. Existing methods differ not only in where latent variables are introduced, but also in how they are represented, how computation is carried out through them, and at which stage they are shaped or exploited. To organize this design space, we categorize latent-space mechanisms along four complementary axes: Architecture (Section 4.1) characterizes the structural role of latent space in the model, including whether it is embedded in the backbone, realized as a dedicated component, or supported by an auxiliary model; Representation (Section 4.2) describes the form of latent variables, distinguishing internal, external, learnable, and hybrid representations. Computation (Section 4.3) captures how the latent space participates in information processing, including compressed, expanded, adaptive, or interleaved computation. Optimization (Section 4.4) focuses on when and how latent space is induced, aligned, or refined, spanning pre-training, post-training, and inference-time mechanisms. These four mechanistic types provide a unified lens for comparing diverse approaches and clarify the major design choices that govern how latent spaces are constructed and used in modern language-based systems.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

![Figure 5](../images/c0716b48f5958a3ab2ba37c17f957ecac30d108a2d7c649f1b54b1d561f2d294.jpg)
*Figure 5: Figure 5 Representative works operate in accordance with latent space mechanisms. We classify all methods into four lines based on diverse ways of utilizing the latent space, including: Architecture (Section 4.1), Representation (Section 4.2), Computation (Section 4.3), and Optimization (Section 4.4).*

> 💡 **Figure 5 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Figure 5 Representative works operate in accordance with latent space mechanisms. We classify all methods into four lines based on diverse ways of utilizing the latent space, including: Architecture (Section 4.1), Representation (Section 4.2), Computation (Section 4.3), and Optimization (Section 4.4).

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

General Notation and Formalization. Based on the notations in Table 1, we first formalize the standard autoregressive generation paradigm. Given an input sequence $\textbf { x } \in { \mathcal { V } }$ , a model $\Phi _ { \theta }$ defines a conditional distribution over the output sequence $\mathbf y \in \mathcal V$ :

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

![Equation 1](../images/b84c71a76bfc574c32978fd5675dcea01a774c42a2c297d848658f0bab48bdd0.jpg)
*Equation 1: Equation extracted by MinerU.*

> 💡 **Equation 1 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where generation is performed purely in token space: the model predicts each output token conditioned on the input and the previously generated context. Although the computation is internally carried out through continuous hidden states $\mathbf { h } \in \mathcal { H }$ , the generation interface itself remains token-to-token.

Latent-space methods extend this formulation by introducing an additional latent representation $\mathbf { z } \in \mathcal { H }$ , such that generation is conditioned not only on the observable input $\mathbf { x }$ but also on a continuous latent variable:

![Equation 2](../images/e66fe1be2ad7c5fccbfaab9f73e0f8af31bb5683873375c2a4a6eb22d4ab7b61.jpg)
*Equation 2: Equation extracted by MinerU.*

> 💡 **Equation 2 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\mathbf { z }$ denotes a latent representation in the latent space $\mathcal { H }$ . Compared with standard autoregressive generation, the latent variable $\mathbf { z }$ provides an additional channel for encoding information that may be difficult to express directly in token space, such as global semantics, multimodal features, intermediate reasoning states, structural constraints, or other task-relevant factors.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Table 2 Overview of the Backbone (Section 4.1.1) based architecture. We compare the hidden dimension, layer, size, and architectural feature of these backbones.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

![Table 2](../images/d415b1c532c5c578133e1a7c1ad7912389031f45917606ecb1c241bf699c24d8.jpg)
*Table 2: Table 2 Overview of the Backbone (Section 4.1.1) based architecture. We compare the hidden dimension, layer, size, and architectural feature of these backbones.*

> 💡 **Table 2 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

Under this perspective, the central question is not merely whether a method uses latent variables, but how latent space is instantiated and integrated into the generation process. This motivates the mechanism-oriented taxonomy in this survey.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 4.1 Architecture

Recent advances in latent-level methods have catalyzed a profound rethinking of the established architectural design paradigms for models operating in explicit representational spaces [66]. Departing from the exclusive reliance on conventional autoregressive frameworks, an increasing number of studies have pioneered innovative mechanisms that enable core computations within latent spaces, where high-level cognitive processes [104], e.g., reasoning, perception, and planning, can be conducted with substantially enhanced efficiency and expressiveness. This transformative shift extends far beyond a mere change of representational domain; it encapsulates a fundamental evolution in the architectural philosophy of modern neural models [294].

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

This emerging paradigm reveals that latent space is evolving from an isolated technical heuristic into a general, foundational architectural principle. To answer the core question: what architectures are utilized to learn the latent space?, we distinguish these methods by the location where latent space $\mathcal { H }$ is integrated into $\Phi$ , with a focus on the structure of the latent system within model space $\Phi = \left\{ \Phi ^ { b a c k } , \Phi ^ { c o m p } , \Phi ^ { a u x } \right\}$ . As reported in Table 2 and Table 3, we classify all existing architecture-driven methods into three categories:

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# Mechanism: Architecture

• Backbone (Section 4.1.1): endows the main model with native latent capacity through recurrent, looping, recursive structures, thereby making an architectural primitive. • Component (Section 4.1.2): employs generation, projection heads, alignment, control, storage, or other components, which allow latent functions while preserving the main model skeleton. • Auxiliary Model (Section 4.1.3): utilizes an extra model to provide supervision signals or intermediate features to guide or supplement the host model.

> 💡 **列表批读**: 这组条目通常对应贡献、设置、发现或模块；建议逐条映射到 visual memory / latent reasoning / medical diagnosis 的主线。

# 4.1.1 Backbone

In this category, latent computation is intrinsically embedded in the primary generative architecture rather than introduced via an external auxiliary module. Formally, the backbone itself carries out an iterative or structured transition over latent states:

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

![Equation 3](../images/189c0748207548fa1512869fb23baa18824f26839edc159e7551b8c1af086352.jpg)
*Equation 3: Equation extracted by MinerU.*

> 💡 **Equation 3 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where each subsequent output token is produced based on the updated hidden state, under this formulation, the latent operation constitutes a native operational mechanism of $\Phi ^ { b a c k } ( \cdot )$ itself, meaning that the reasoning or intermediate transformation process is realized internally within the backbone, without requiring any additional component.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Existing backbone-based methods can be broadly understood from three complementary perspectives: Parameter-shared, Iterative Refinement, and Augmented. Rather than simply following the explicit-level counterparts, these works revisit the backbone design itself to offer a promising path.

> 💡 **批注**: 这是因果/反事实修正：关注它如何剔除伪相关、突出关键病灶或视觉证据。

Parameter-shared Backbone. From the perspective of parameter sharing, a representative line of work replaces a deep stack of distinct layers with a smaller set of reusable modules applied repeatedly. However, in this paradigm, the recurrent depth or times is typically fixed. Huginn [50], for example, adopts a decoder-only architecture, but compensates for the shallow explicit depth by introducing a shared recurrent block that is reused across multiple depth steps. This design substantially reduces the number of unique parameters while enabling test-time scaling, where additional recurrent steps can be applied during inference to trade compute for performance. Looped Trans. [167] further develops this idea by enforcing an explicit layer-looping mechanism. A key technical feature is its looping-based regularization, which stabilizes the hidden-state dynamics under repeated application of the same transformation and encourages convergence of iterative representations. In a related efficiency-oriented setting, PHD-Trans. [223] explores how such recurrent computation can remain practical under long-context decoding. It integrates cache management with sliding-window attention, reducing memory overhead while preserving the benefits of repeated updates.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Iterative Backbone. This paradigm highlights the iterative process and dynamic updating, with variable or learnable depth allocation. For instance, Ouro [296] introduces a recursive inference framework whose iterations can serve as an alternative scaling axis, analogous to increasing architectural depth, showing that repeated application of parameter-shared transformations can yield consistent gains. LoopFormer [72] introduces an elastic-depth looped transformer, where the number of loop iterations is not fixed but can vary across inputs or computational budgets. This makes recurrent-depth execution more flexible and better aligned with the complexity of individual examples. PonderLM2 [267] is a representative method in this category. Built as a decoder-only model, it employs iterative refinement via Jacobi-style parallel updates. Instead of strictly following the standard autoregressive regime, where each forward pass advances the prediction by one token, it also performs multi-step hidden-state evolution in parallel, enabling richer internal computation while retaining decoding efficiency.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Augmented Backbone. Beyond these two designs, several works explore augmented architectures at different granularities. For example, Heima [223] and DLCM [158], unlike most encoder-only designs, employ hierarchical encoder-decoder architectures to organize latent computation. The latter one also shifts computation from tokens to a compressed concept space, with more efficient semantic operations. Dreamer [86] and MLRA [121] design sequence-depth sparse attention mixtures and low-rank attention, respectively, making multi-step latent transitions computationally tractable and efficient. In addition, MoLAE [127] designs an architecture of reformulating the mixture of latent experts in lower dimension.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Summary. Overall, backbone-oriented methods represent the most intrinsic form of architecture-level latent modeling. Parameter-sharing approaches improve efficiency by reusing subsets of layers or models; iterative-refinement methods further enhance flexibility by enabling dynamic, adaptive iterations; and augmentation-based designs provide a broader view of architectural shifts. This shift provides the foundation for more flexible, computation-aware, and cognitively expressive generative systems.

> 💡 **批注**: 这是因果/反事实修正：关注它如何剔除伪相关、突出关键病灶或视觉证据。

# 4.1.2 Component

This paradigm preserves the original backbone architecture but augments it with functional modules that construct, transform, store, or retrieve latent representations. Formally, a component produces and is then injected into the backbone decoding process:

![Equation 4](../images/bc19c12900002a46058c48f2d09b466ddeae6af4d486866983541ff2afd274fb.jpg)
*Equation 4: Equation extracted by MinerU.*

> 💡 **Equation 4 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where the backbone $\Phi ^ { b a c k } ( \cdot )$ remains the principal generator, while the extra component $\Phi ^ { \mathrm { c o m p } } ( \cdot )$ acts as a plug-in operator over latent space, and the output $\mathbf { z }$ will be used in Equation 2.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Such a design preserves the backbone architecture while equipping it with a latent mechanism to facilitate or guide downstream generation. It covers a broad family of modules that operate on latent spaces while leaving the backbone largely frozen. We categorize existing approaches into five component families: Generation, Projection, Alignment, Control, and Storage, in view of their functional characteristics.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Table 3 Overview of the Component (Section 4.1.2) and Auxiliary Model (Section 4.1.3) based architecture, respectively. We compare the modality, backbone, the type of the component module or external model, function, and scenario. Here, (VQ)-AE, SAE, MLP, Q-Former, LoRA, and JEPA denote (vector quantized) variational autoencoder, multilayer perceptron, querying transformer, low-rank adaptation, and joint embedding predictive architecture, respectively.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

![Table 3](../images/4793f664eb7847c88ee3901221a269fd45574b15f759159b4c834e4910d937b2.jpg)
*Table 3: Table 3 Overview of the Component (Section 4.1.2) and Auxiliary Model (Section 4.1.3) based architecture, respectively. We compare the modality, backbone, the type of the component module or external model, function, and scenario. Here, (VQ)-AE, SAE, MLP, Q-Former, LoRA, and JEPA denote (vector quantized) variational autoencoder, multilayer perceptron, querying transformer, low-rank adaptation, and joint embedding predictive architecture, respectively.*

> 💡 **Table 3 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

Generation Component. A major line of this part aims to construct intermediate latent states in hidden space, allowing the model to synthesize new implicit objectives, subgoals, or reasoning states that are not previously available as explicit symbols. A representative paradigm is realized as discrete tokens or a soft chain evolving rather than through explicit verbalized chains. For instance, ETD [87] introduces an encode–think–decode mechanism that shifts part of the process into latent computation without requiring explicit verbalization. At the same time, Palette [129], iCLP [25], ATP-Latent [292], and ReGuLaR [205] employ VAE-style components to modulate high-level contexts and encourage diverse exploration. At a coarser granularity, JEPA-Reasoner [110] formulates reasoning as a chain of latent predictions under a JEPA-style framework, thereby decoupling latent reasoning from surface token generation.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Beyond latent trajectory generation, another line of work leverages non-trajectory latent signals, e.g., compressed embeddings, activation-space features, or learned steering vectors. CoLaR [188], for example, enables more efficient silent reasoning through embedding prediction and control. More broadly, AR [64] and RB-CoT [63] both employ SAE to generate interpretable feature directions in activation space, while PILOT [289] synthesizes latent guidance vectors via an MLP to steer decoding. Additionally, MemGen [273], VisMem [264], and LatentMem [47] also adopt LoRA attached to the backbone to weave latent memories as an injection.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

This idea has also been extended to the multimodal setting, where latent variables serve not only as semantic states but also as vision carriers. Recent works, such as AURORA [11], introduce a variational autoencoder to synthesize perception tokens, enabling richer visual understanding. CoMEM [229] and its agent variant CoMEM-Agent [230] deploy querying transformers to produce compact visual memory tokens, showing potential in long-horizon tasks. In addition, SteerVAD [18] generates related latent frame-level vision embeddings from mixed layers for video-based tasks, and Latent Sketchpad [276] uses a VAE decoder to generate intermediate visual imaginations during multi-step visual reasoning.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Projection Component. This category does not primarily synthesize new latent content, but instead improves reasoning or control by projecting existing internal representations into a different target space. Several works attach lightweight linear layers, e.g., SoftCoT++ [244] and PLaT [207], or MLPs, such as SpiralThinker [155] and LiteReason [55], that project hidden states into a target semantic space. Besides, LF-Steering [250] trains an SAE to project activations into a semantic subspace to ensure semantic consistency.

In another typical path, projection serves as a bridge across modalities or agents. Wormhole [124] connects heterogeneous visual agents through an encoder–decoder bridge that projects each agent’s visual representation into a shared latent space. OccVLA [118] projects 3D spatial features via a transformer adapter, improving unified multimodal understanding. Furthermore, as embodied scenarios advance, more researchers are focusing on action projection. For instance, LCLA [182] and LaST0 [128] attach MLP projectors in embodied manipulation. ThinkAct [69] and Fast-ThinkAct [70] both attach Q-former and MLP heads to project visual reasoning traces into robot action spaces. Combining both, VITA [136] adopts an encoder–decoder architecture to encode visual-action context into one latent representation, unifying perception and control in manipulation, while ColaVLA [152] introduces a forward layer that projects visual features into the action-planning space.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Alignment Component. While generation creates and projection reshapes, alignment ensures the transformed latent is anchored to something meaningful. These components enforce correspondence between latent representations and external grounding signals, whether visual features, task semantics, or cross-domain knowledge. AlignVLM [137] redesigns the cross-modal fusion layers, introducing mixed alignment layers that more faithfully bind visual tokens to textual semantics, while PREGEN [169] aligns generated video embeddings to retrieval-relevant textual semantics. Aligned with generative priors, LaDiR [83] and LoLA [213] align hidden states into diffusion-compatible or transformer-based semantic spaces, ensuring consistency with desirable grounding signals. Extending to cross-domain scenes, Interpreter [82] uses LoRA to transfer latent abilities across domains, aligning source-domain competencies with target-domain representations.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Control Component. It determines when and how the model enters, exits, or delegates latent modes. Rather than generating content, they act as meta-level switches or routers that adaptively modulate the generation process. To provide switching signals, FR-Ponder [62] trains a small MLP gating head to predict whether a given input requires extended latent pondering, and TaH [49] positions a small MLP-style decider at each layer that votes on whether to enter a latent deliberation phase, enabling dynamic budget allocation in latent manifolds. MemGen [273] incorporates a trigger, implemented via LoRA or entropy signals, that determines when memory should be invoked, while Kelp [101] similarly uses an MLP-based module for risk-sensitive inputs in safety scenarios.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Storage Component. Storage components maintain persistent latent states across steps, turns, or episodes, enabling models to accumulate, compress, and retrieve information without relying on the finite context window. IMM [148] and L2-VMAS [265] introduce a differentiable vector library that acts as a latent memory bank, and G-MemLLM [242] scales this with a gated write–read logic that selectively updates memory. Further, PolarMem [30] replaces the flat vector store with a graph-topology structure that clusters and links visual memory across episodes.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Summary. Component-based methods preserve the backbone architecture while augmenting it with plug-in latent modules that construct, transform, align, control, or store internal representations for downstream generation. Rather than replacing the backbone, these components operate as functional operators over latent space, enhancing reasoning, grounding, controllability, and memory with minimal architectural disruption. Existing approaches can be broadly organized into five families: generation components that synthesize latent reasoning states; projection components that map hidden representations into task-relevant spaces; alignment components that anchor latents to external semantics or priors; control components that regulate latent computation adaptively; and storage components that maintain persistent latent storage.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 4.1.3 Auxiliary Model

Here, the latent guidance signal is introduced by an external auxiliary model, rather than being natively induced within the backbone or instantiated by an internal functional component. Concretely, an auxiliary model first produces a latent representation and then is used to condition, guide, or refine the generation process of the host model:

![Equation 5](../images/600bbea7134d9110e2e0400df82f303fa307ee9b9d7879ec3429a5401f5bf2ba.jpg)
*Equation 5: Equation extracted by MinerU.*

> 💡 **Equation 5 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where the latent introduction is outsourced to auxiliary model $\Phi ^ { a u x } ( \cdot )$ to replenish the host model $\Phi ^ { b a c k } ( \cdot )$ , then $\mathbf { z }$ will be injected into the autoregressive process in Equation 2.

This paradigm introduces a functional division of labour: the host model retains responsibility for downstream prediction, while the auxiliary model either shapes its learning objective or enriches its internal representations. The resulting works bifurcate cleanly into two families according to what the auxiliary model contributes: Supervision-oriented approaches that provide training signals, and Feature-oriented approaches that provide intermediate representations.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Supervision-oriented Auxiliary Model. For the methods in this paradigm, the auxiliary models supply signals that guide the host model with semantic constraints, regularization, and supervision. The most direct strategy is to leverage another language model serving as a teacher to generate expressive traces. For instance, HCoT [122] and LaViT [227] instantiate assistant models to generate explicit reasoning chains, whose internal representations are then distilled into the host hidden states. SoftCoT [243] extends this paradigm by projecting the auxiliary chain into a continuous latent embedding rather than discrete token sequences, yielding soft supervision compatible with gradient-based fine-tuning. For finer-grained supervision, some methods focus on aligning the host latent representations with those of an external reference model. SIM-CoT [220] and SemCoT [61] couple addition models whose hidden states function as semantic anchors, and train the host to reproduce these representations within its own latent space via a contrastive objective.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Another group treats the auxiliary model as a state generator whose outputs define structured training targets over latent trajectories. CTRLS [226] uses a small LLM to synthesise intermediate reasoning states that serve as exploration waypoints within the host latent space. LatentEvolve [274] extends this idea to evolutionary optimisation, employing an auxiliary model to generate candidate token sequences that drive iterative refinement of memory representations. CoLT [293] takes a more task-specific stance, assigning a tinyscale auxiliary model to decompose latent tool-calling states into interpretable intermediate representations that supervise the internal planning phase.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Feature-oriented Auxiliary Model. These auxiliary models shift the locus to features available for the host model, generating and injecting intermediate representations directly into the computation graph. This paradigm is especially prevalent in multimodal and embodied settings, where the heterogeneity between modalities, i.e., language, vision, and action, creates representational bottlenecks. A sizeable cohort of methods employs dedicated vision models whose outputs supplement or supplant the host’s own visual representations. CoVT [156] frames this injection as a chain of visual thought, wherein auxiliary vision models iteratively construct intermediate visual representations analogous to the token-level reasoning steps of textual CoT. 3DThinker [28] pairs a specialised 3D foundation model that provides spatially-grounded geometric priors. Further works, including LaRe [135], MM-CoT [171], and VaLR [73], leverage diffusion-architected generative models. Given a visual input, they reconstruct or imagine visual states whose latent codes are fed to the host model as an enriched perceptual context, enabling forms of visual imagination. OneLatent [133] and RoT [216] use a vision encoder to render textual reasoning into latent visual spaces. In addition, SkiLa [197] and VL-JEPA [21] further explore vision projection in generative and self-supervised regimes, respectively.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

In embodied and autonomous systems, the role of the auxiliary model shifts to policy grounding and environment perception. UniVLA [14] incorporates a dedicated vision encoder to generate task-specific feature sequences, and SSM-VLA [19] introduces a VLM-based auxiliary that bridges visual perception and motor action within a state-space framework. SwiftVLA [142] and LaST-VLA [132] introduce spatial-temporal information through vision auxiliary, proving particularly beneficial for long-horizon manipulation tasks that demand coherent temporal reasoning. In autonomous driving, LCDrive [187] and DW-VLA [75] incorporate world models and extra vision encoders, respectively, as auxiliary feature sources, grounding consistent representations of scene dynamics. Future-VLA [44] takes this further by conditioning action generation on auxiliary-predicted future visual features, effectively rendering the auxiliary model as a predictive look-ahead mechanism. In addition, WholeBodyVLA [77] and LatentVLA [235] extend this to whole-body humanoid control by employing a latent action model as a feature encoder whose representations capture joint-level coordination structure.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Summary. The auxiliary-model paradigm introduces latent guidance through an external model rather than the backbone itself. Such auxiliary models either provide supervision signals to shape the backbone’s latent space or supply intermediate features that enrich its internal computation, making this paradigm particularly effective for complex reasoning, multimodal understanding, and embodied decision-making.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 4.2 Representation

The transition from discrete token sequences to the continuous latent space $\mathcal { H }$ necessitates a precise definition of the core information carrier: the latent representation $\textbf { z } \in \ \mathcal { H }$ . Unlike discrete tokens $\textbf { x } \in { \mathcal { V } }$ , which are constrained to a predefined vocabulary, latent representations reside on a continuous, high-dimensional manifold, enabling substantially richer semantic expressivity [8, 58, 243].

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Two central questions motivate the study of these methods: what information does a latent representation encode, and how is it integrated into the generative pipeline? The answers fundamentally determine the representational capacity, training dynamics, and generalization behavior of the resulting system. To provide a coherent organizational framework, we propose a taxonomy that classifies existing methods along two orthogonal axes: the subject of the representation (how $\mathbf { z }$ is structurally constructed, i.e., whether it is computed natively within the backbone or generated by a structurally independent module) and its parameterization (whether the construction process relies on fixed model states or incorporates dedicated trainable modules). As illustrated in Figure 6 and Table 4, the intersection of these axes yields a comprehensive taxonomy comprising four distinct paradigms, detailed as follows:

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Figure 6](../images/32c27c21d89081356052b8165abc7a4ceacb860d1c98067ed55e7bea8bf81622.jpg)
*Figure 6: Figure 6 The schematic diagram of Representation mechanism, including four sub-types: Internal (Section 4.2.1), External (Section 4.2.2), Learnable (Section 4.2.3), and Hybrid (Section 4.2.4).*

> 💡 **Figure 6 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Figure 6 The schematic diagram of Representation mechanism, including four sub-types: Internal (Section 4.2.1), External (Section 4.2.2), Learnable (Section 4.2.3), and Hybrid (Section 4.2.4).

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

# Mechanism: Representation

• Internal (Section 4.2.1): operates directly on activations produced during the backbone’s forward pass, including token embeddings, intermediate hidden states, and KV caches. • External (Section 4.2.2): derived from a structurally independent auxiliary system (e.g., a pre-trained encoder), and injects these exogenous signals into the backbone as conditioning inputs or supervision targets while the auxiliary source remains frozen. • Learnable (Section 4.2.3): constructed by dedicated trainable modules (e.g., continuous virtual tokens or lightweight adapters) that are embedded directly into the backbone and optimized end-to-end under specific task objectives. • Hybrid (Section 4.2.4): combines the Learnable and External paradigms sequentially by first using trainable modules to create specialized representations, then injecting these states as exogenous signals into the backbone for conditioning or latent supervision.

> 💡 **列表批读**: 这组条目通常对应贡献、设置、发现或模块；建议逐条映射到 visual memory / latent reasoning / medical diagnosis 的主线。

# 4.2.1 Internal

In the internal paradigm, $\mathbf { z }$ is derived exclusively from endogenous activations generated during the backbone’s standard forward pass, without introducing any additional parameters. Let $\Phi ^ { \mathrm { b a c k } }$ consist of $L$ transformer blocks decomposed as:

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

![Equation 6](../images/7ea4cda2f4fd7ff32cdd2df2b1c036e940578a5b6629312ee1847a8b5a47a701.jpg)
*Equation 6: Equation extracted by MinerU.*

> 💡 **Equation 6 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\phi _ { \mathrm { e m b } } : \mathcal { V }  \mathbb { R } ^ { d }$ maps discrete tokens to continuous embeddings via the embedding matrix $\mathbf { E } \in \mathbb { R } ^ { | \mathcal { V } | \times d }$ each transformer block $\phi _ { l } : \mathbb { R } ^ { d \times T ^ { \prime } }  \mathbb { R } ^ { d \times T ^ { \prime } }$ processes the full sequence of $T$ tokens, and $\phi _ { \mathrm { h e a d } }$ projects the final hidden states back to the vocabulary space. Let $\mathbf { h } _ { l } ^ { t } \in \mathbb { R } ^ { d }$ denote the hidden state of token $t$ at layer $\it l$ , and let $\mathbf { H } _ { l } = \left| \mathbf { h } _ { l } ^ { 1 } , \ldots , \mathbf { h } _ { l } ^ { T } \right| \in \mathbb { R } ^ { T \times d }$ collect all token states at layer $\it l$ . The latent representation is then a deterministic readout:

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

![Equation 7](../images/8744d25ad531670803856aa63ca5c7b5d9bd02c6fdae5ea59219c88e9b8886dc.jpg)
*Equation 7: Equation extracted by MinerU.*

> 💡 **Equation 7 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $g ( \cdot )$ is a parameter-free aggregation function applied over the set of layer activations $S$ (e.g., index selection, mean pooling across tokens, or a fixed linear combination over layers). Depending on which

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

Table 4 Overview of the Internal (Section 4.2.1), External (Section 4.2.2), Learnable (Section 4.2.3), and Hybrid (Section 4.2.4) representation. We compare the modality, backbone, representation subject, and scenario.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

![Table 4](../images/15a3889604b34588c46176a451425a1463b4b160bcd2a3080a9b336a7902df9c.jpg)
*Table 4: Table 4 Overview of the Internal (Section 4.2.1), External (Section 4.2.2), Learnable (Section 4.2.3), and Hybrid (Section 4.2.4) representation. We compare the modality, backbone, representation subject, and scenario.*

> 💡 **Table 4 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

activation type is extracted, internal representations manifest in three principal forms: Hidden State, Weighted Embedding, and Cache.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

Internal Hidden State. As the most prevalent form, intermediate activations are extracted to provide a continuous summary of the model’s evolving computation. Common instantiations include taking the last hidden state of position $T$ , or computing a mean-pooled representation across all tokens at a particular layer $\it { \Delta } l$

![Equation 8](../images/d236dde2920635d3d2900ef5d5a6539253f01a5bfb054de4570478243c3401f5.jpg)
*Equation 8: Equation extracted by MinerU.*

> 💡 **Equation 8 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

For example, COCONUT [58] establishes a foundational pattern for continuous generation by feeding the last hidden state directly back as the next input embedding, bypassing discrete vocabulary projection and forming a recurrent loop of continuous “thoughts”. SIM-CoT [220] and LatentMAS [300] adopt this recurrent paradigm. Beyond direct generation, the rich semantic geometry of hidden states proves valuable in downstream applications. For instance, CoE [217] constructs chains of embeddings from pooled hidden states to perform label-free self-evaluation. In multimodal settings, internal activations frequently serve as diagnostic and corrective signals: VTI [119] and CGC-VTD [212] dynamically probe intermediate hidden states to detect and mitigate visual hallucinations. Final hidden states have also been repurposed as compressed visual-semantic summaries: IVT-LR [20] fuses them with image embeddings for efficient visual reasoning, while CausalEmbed [71] projects them into dense representations for visual document retrieval. Because these states natively encode pre-trained knowledge, CLReg [191] further leverages them as a regularizer to mitigate catastrophic forgetting. The high expressivity of endogenous states is, however, a double-edged sword from a security perspective. On the offensive side, LFJ [237] and LatentBreak [140] exploit hidden-state vulnerabilities to mount latent jailbreak attacks that bypass discrete text-level filters. Conversely, JLT [81] monitors these same signals defensively, using them to detect malicious intent and improve jailbreak robustness.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Internal Weighted Embedding. This form replaces hard token sampling with a soft, probability-weighted combination over the vocabulary embedding matrix $\mathbf { E } \in \mathbb { R } ^ { | \mathcal { V } | \times d }$ . Let $\pmb { \alpha } \in \mathbb { R } ^ { | \nu | }$ denote a weight vector derived from the model’s output logits (i.e., via softmax); the latent representation is:

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

![Equation 9](../images/08b0e3b442a3b51afc9320a9d443d7151790f74fde9154aea2c79a304de5b087.jpg)
*Equation 9: Equation extracted by MinerU.*

> 💡 **Equation 9 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\mathbf { o } \in \mathbb { R } ^ { | \nu | }$ denotes the pre-softmax logits. Because $\pmb { \alpha }$ lies in the probability simplex, $\mathbf { z }$ is constrained to the convex hull of the vocabulary embedding vectors.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

This differentiable relaxation forms a superposition of candidate token embeddings [52, 287], enabling gradient flow through the discrete generation step and facilitating parallel exploration of the reasoning space. LT-Tuning [123] builds on this property with a context-prediction fusion mechanism that exploits predictive semantic guidance from the vocabulary embedding space to construct robust latent thoughts. ThinkRouter [241] further leverages the continuous nature of $\mathbf { z }$ for confidence-aware routing, dynamically switching between continuous latent thinking and discrete token generation based on model uncertainty.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

Internal Cache. The third form treats accumulated key-value pairs as structured latent memory, enabling efficient context reuse without recomputation. Given the hidden states $\mathbf { H } _ { l } \in \mathbb { R } ^ { T \times d }$ , the cache representations are computed via learned projection matrices:

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Equation 10](../images/1fea04168a78d1a13d1a6078ce58683b626b7f23ced0cf90985882a69a0f8c0b.jpg)
*Equation 10: Equation extracted by MinerU.*

> 💡 **Equation 10 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\mathbf { W } _ { l } ^ { k } , \mathbf { W } _ { l } ^ { v } \in \mathbb { R } ^ { d \times d _ { k } }$ project the sequence into the key and value spaces of the attention heads.

By treating these compressed tensors as operational memory, SALS [139] exploits sparse attention patterns over the KV cache to accelerate inter-step computation and improve inference efficiency. LatentMAS [300] extends this to multi-agent collaboration, using the KV cache as a shared, continuous working memory that enables agents to coordinate without explicit textual communication. In the multimodal domain, LRP [253] combines attention weights and hidden states to capture richer visual semantics, demonstrating improved abstraction and generalization on complex visual tasks.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Summary. The internal paradigm converts endogenous model activations into parameter-free latent representations, entirely bypassing the discrete vocabulary bottleneck. By treating standard computational byproducts as versatile semantic assets, it demonstrates that intrinsic model states carry sufficient representational capacity to support continuous reasoning, accelerate inference, and enable robust downstream analysis.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

# 4.2.2 External

In the external paradigm, $\mathbf { z }$ originates from an auxiliary encoder $\Phi ^ { \mathrm { a u x } }$ that is structurally independent of the backbone. Given an auxiliary input $\bf { x } _ { a u x }$ , which may coincide with $\mathbf { x }$ or belong to a different modality:

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

![Equation 11](../images/231b5e551b0c5635406909c46dbf61f940aa109d1e74e5467e65ff073d7efcf9.jpg)
*Equation 11: Equation extracted by MinerU.*

> 💡 **Equation 11 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\Phi ^ { \mathrm { a u x } }$ is kept frozen during backbone training. Because $\Phi ^ { \mathrm { a u x } }$ and $\Phi ^ { \mathrm { b a c k } }$ operate in distinct latent manifolds with potentially mismatched dimensionalities ( $d _ { \mathrm { a u x } } \neq d _ { \mathrm { b a c k } }$ ), $\mathbf { z }$ requires explicit structural alignment before integration. Depending on its functional role, this paradigm operates in two modes.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

First, as a conditioning input to guide backbone generation, $\mathbf { z }$ is first aligned to the backbone’s native latent space via a learned projection $\psi : \mathbb { R } ^ { d _ { \mathrm { a u x } } }  \mathbb { R } ^ { d _ { \mathrm { b a c k } } }$ (e.g., a linear map or MLP), and the backbone conditions its output on the projected signal:

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

![Equation 12](../images/2b94cf4af8b3879cd91fc7e35e0749d2729b94ec1b2e6ce3135fe3cf476f4618.jpg)
*Equation 12: Equation extracted by MinerU.*

> 💡 **Equation 12 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where [zˆ; x] denotes a latent integration mechanism such as prefix concatenation or cross-attention injection.

Second, as a supervision target for representation alignment or knowledge distillation, $\mathbf { z }$ serves as a dense, non-verbalized supervision target. The backbone’s internal state $\mathbf { h }$ is trained to match the projected external signal. This continuous supervision transfers rich semantic priors from $\Phi ^ { \mathrm { a u x } }$ to the backbone without the information bottleneck imposed by discrete token generation. Based on the modality and functional role of $\Phi ^ { \mathrm { a u x } }$ , we identify three directions: Reasoning Priors, Perceptual Priors, and Embodied Priors.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

External Reasoning Priors. Here, the auxiliary source operates in the semantic domain to supply the backbone with structured logic and cognitive traces. A prevalent strategy is knowledge distillation from an expert reasoning model. CODI [174] formalizes this via a self-distillation loop in which the frozen teacher’s hidden states serve as continuous supervision targets, guiding the implicit reasoning of the student. SoftCoT [243] extends this concept to explicit injection: a lightweight assistant model generates speculative reasoning chains that are projected into soft-token embeddings and prepended to the backbone input, creating a gradient-compatible conditioning signal that substantially improves zero-shot generalization. KaVa [91] bypasses intermediate output tokens entirely by distilling the teacher’s compressed KV cache, directly transferring structured attention states as compact latent priors.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

External Perceptual Priors. Here, pre-trained vision encoders or specialized multimodal auxiliary models supply spatially, temporally, or structurally rich feature maps to augment a primarily textual backbone. 3DThinker [28] injects spatially grounded 3D tokens from a specialized auxiliary network to supply geometric priors that the base model cannot derive from two-dimensional inputs alone. SkiLa [197] leverages pre-trained sketch tokens to interleave intermediate visual thoughts with textual reasoning, explicitly grounding the model’s spatial understanding. VL-JEPA [21] employs pre-trained embeddings from a predictive-coding model in an abstract representation space to improve cross-modal classification and retrieval. COVT [156] uses an auxiliary vision model to iteratively extract visual tokens and fuse them into the backbone to construct continuous visual thought chains. OneLatent [133] demonstrates that hidden states from a strong visionlanguage model can condense rich perceptual and OCR context into a single latent token for highly efficient visual reasoning. Mull-Tokens [160] generalizes this injection with modality-agnostic latent tokens derived from an auxiliary system, enabling uniform latent reasoning across both linguistic and visual substrates.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

External Embodied Priors. This direction, prevalent in embodied models and autonomous driving, relies on auxiliary models to generate structured latent representations of environmental dynamics, 3D geometry, and future actions. OccVLA [118] applies pre-trained 3D occupancy tokens as a supervisory signal to impart fine-grained spatial understanding without requiring explicit 3D inputs at inference time. LCDrive [187] incorporates an external world model to generate action and scene tokens that inject temporally consistent representations of scene dynamics, grounding the host model’s navigational reasoning. LaRA-VLA [5] bridges continuous perception and motor control by utilizing pre-trained visual and action tokens, enabling a smooth transition from explicit multimodal supervision to internalized latent reasoning for complex embodied manipulation.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Summary. The external paradigm provides a principled approach to bridging modality gaps and circumventing the discrete token bottleneck by injecting structured knowledge from independent auxiliary systems. By aligning reasoning, perceptual, and embodied priors as either dynamic conditioning signals or dense supervision targets, it endows the backbone with continuous reasoning capabilities and complex structural constraints — without modifying the backbone’s parameters. This demonstrates that a foundation model’s representational scope is not strictly bounded by its native architecture: leveraging specialized latent priors offers a scalable pathway to expand semantic reach while keeping the backbone largely frozen.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

# 4.2.3 Learnable

In the learnable paradigm, $\mathbf { z }$ is actively constructed by a parameterized module $\Phi ^ { \mathrm { c o m p } }$ with learnable parameters $\theta$ that is directly embedded within the backbone architecture (e.g., continuous virtual tokens or lightweight adapters). Let c denote an optional conditioning context, such as the input $\mathbf { x }$ , intermediate hidden states $\mathbf { h }$ , or the empty set ( $\mathbf c = \emptyset$ ). The latent representation is formulated as:

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

![Equation 13](../images/1fc104b808ef749140c90e044fcc6df705290fab708cd16ff1bf67e34fbb1809.jpg)
*Equation 13: Equation extracted by MinerU.*

> 💡 **Equation 13 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

Unlike the external paradigm, $\Phi ^ { \mathrm { c o m p } }$ is structurally coupled with the backbone. The parameters $\theta$ are optimized end-to-end driven by specific task objectives (either independently with a frozen backbone or jointly with $\theta ^ { \mathrm { b a c k } }$ ). Depending on the targeted optimization objective, learnable representations manifest in three primary forms: Compression Learning, Distribution Learning, and Alignment Learning.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

Compression Learning. The first class is driven by information-bottleneck principles, optimizing $\Phi ^ { \mathrm { c o m p } }$ to compress explicit data into dense, continuous vectors. CoLaR [188] learns to aggregate consecutive reasoning tokens into compressed embeddings using a variance-preserving scaling factor. CoLT [293] employs supervised learning to condense long reasoning trajectories into continuous seed tokens for parametric tool calls. In the multimodal domain, LIVR [100] imposes a visual bottleneck during training, requiring the model to learn implicit spatial compressions without relying on explicit textual descriptions. At the memory level, DeltaKV [57] encodes residual differences between successive cache states to substantially compress long-term reasoning overhead.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Distribution Learning. The second class abandons deterministic point mappings, instead optimizing $\Phi ^ { \mathrm { c o m p } }$ to capture the underlying stochastic distributions and structural manifolds of reasoning. CTRLS [226] formulates reasoning as a Markov decision process and models state transitions via Dirichlet distributions to capture epistemic uncertainty. MARCOS [115] employs a conditional hidden Markov model with step-level latent variables, relying on variational training to learn stochastic continuous representations. UniCog [116] formulates cognitive distributions as a latent variable model, optimizing an evidence lower bound to project activations into a high-dimensional sparse space. LatentGuard [179] learns to model the latent space using a VAE, manipulating the resulting semantic distributions to enable robust refusal of adversarial inputs.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Alignment Learning. The third class optimizes $\Phi ^ { \mathrm { c o m p } }$ to construct cross-space projections, bridging rigid boundaries such as distinct modalities or heterogeneous agent architectures. KVCA [38] learns a globally shared latent manifold $\Sigma$ via cross-attention to translate KV caches between architecturally heterogeneous models. C2C [48] trains a neural MLP to directly project KV caches between specific models by aligning their terminal-layer representations. Interlat [42] optimizes communication adapters via a weighted Jensen-Shannon divergence to align transmitted hidden states with the receiving agent’s internal representations. This paradigm also bridges modalities: MCOUT [154] learns cross-modal attention fusion while penalizing entropy collapse; MAS4TS [163] learns to reconstruct predictive numerical trajectories from visual time-series plots; and TCLA [144] aligns noisy observational signals with clean, VLM-prompted latent action targets for embodied manipulation.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

Summary. The learnable paradigm excels at acquiring latent structures explicitly optimized for specific downstream objectives, breaking free from the rigid semantic boundaries imposed by textual pre-training. This flexibility allows models to encode non-verbal modalities (e.g., complex spatial layouts and embodied actions) and support high-bandwidth inter-agent collaboration. However, granting unconstrained optimization access to continuous spaces introduces the risk of manifold overfitting: modules may memorize the noise in supervision signals, yielding highly specialized representations that sacrifice zero-shot generalization. Rigorous regularization, including enforced structural sparsity, variance-preserving normalizations, and controlled belief-shift tuning, which is therefore essential to maintain broad utility.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

# 4.2.4 Hybrid

The hybrid paradigm utilizes a dedicated, structurally independent module $\Phi ^ { \mathrm { c o m p } }$ with learnable parameters $\theta$ to construct a structured latent representation. Let $\mathbf { c }$ denote a conditioning context drawn from the input $\mathbf { x }$ or auxiliary features. The hybrid representation is formulated as:

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

![Equation 14](../images/78a09109f12a96b33a405b0476378907d7ae80cec0a8f5e361346f9a7c1a8e2f.jpg)
*Equation 14: Equation extracted by MinerU.*

> 💡 **Equation 14 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

Crucially, unlike the Learnable paradigm, $\Phi ^ { \mathrm { c o m p } }$ remains architecturally disjoint from the backbone $\Phi ^ { \mathrm { b a c k } }$ Once constructed, $\mathbf { z }$ is deployed in the same manner as the External paradigm: it acts either as an exogenous conditioning signal injected into a typically frozen backbone (serving as a learned alignment bridge), or as a rich, optimized supervision target for latent distillation. This approach can be summarized into three functional categories: Traces, Grounding, and Augmentation.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Traces. The first branch distills discrete reasoning trajectories into compact continuous vectors that guide the backbone without the latency of explicit chain-of-thought generation. HCoT [122] compresses multi-step reasoning into a specialized thought token injected into a frozen decoder to accelerate inference. Assorted [181] combines latent and text tokens by compressing reasoning segments via a VQ-VAE codebook. Latent-SFT [37] restricts the latent space to the column space of the pre-trained vocabulary matrix via induction-supervision masking. EBM-CoT [29] refines thought trajectories toward lower-energy regions via Langevin dynamics calibration. GainRouter [288] learns a codebook of discrete strategy priors to condition single-pass decoding on continuous thinking vectors. ThoughtComm [290] employs a sparsity-regularized autoencoder to transmit latent thoughts between agents. For specialized domains, LatentChem [259] aligns projected molecular representations with linguistic spaces, while iCLP [25] and RB-CoT [63] generate prompt-sensitive latent plans to guide multi-step reasoning.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Grounding. The second branch translates continuous sensory or control signals into structured latent tokens, grounding the frozen backbone in physical reality. In the visual domain, AURORA [11] trains a VQ-VAE to produce discrete visual latent codes that enhance spatial perception, while Monet [211] generates continuous embeddings as intermediate visual thoughts via multi-stage distillation. In embodied robotics, UniVLA [14] derives task-centric latent actions from heterogeneous videos and projects them into a unified action space. LatBot [105] decomposes latent actions into discrete motion and scene tokens; Motus [10] extracts pixel-level delta actions via optical flow; UniLACT [51] constructs depth-aware latent action representations from RGB-D frames; and VITA [136] maps visual-action dynamics into unified codebooks.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Augmentation. The third branch condenses large contextual histories into compressed soft prompts or dynamically augmented cache states, extending effective context beyond standard window limits. DCA [117] generates latent embeddings via an offline coprocessor and appends them directly to the KV cache, enriching context without altering the decoding architecture. CLaRa [59] encodes lengthy documents into compact memory tokens via a salient compressor trained on synthetic data, enabling end-to-end optimization of the retrieval representation space. DEP [157] isolates user-specific interaction patterns via a sparse autoencoder and injects them as soft prompts for personalized generation. In multimodal settings, VisMem [264] and CoMEM [229] construct dedicated visual memory tokens that distill episodic experiences for long-horizon planning, while MemGen [273] integrates episodic representations into computation for self-evolving behavior.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Summary. Hybrid representations are particularly suited to complex, multimodal, or domain-specialized settings in which the gap between raw inputs and the backbone’s textual manifold creates bottlenecks that neither purely internal nor purely learnable approaches can adequately resolve. By first constructing a task-specific latent representations via targeted learnable acquisition and subsequently deploying it as a structured external conditioning signal, hybrid approaches achieve a dual advantage: semantic fidelity, grounding representations in the raw input modality; and computational efficiency, sparing the backbone from processing raw, high-dimensional inputs at inference time.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

![Figure 7](../images/a19daaf9d5c70e0802054d5f007eb766cc79aa20255a284d88ca9ec003e613aa.jpg)
*Figure 7: Figure 7 The schematic diagram of Computation mechanism, including four sub-types: Compressed (Section 4.3.1), Expanded (Section 4.3.2), Adaptive (Section 4.3.3), and Interleaved (Section 4.3.4).*

> 💡 **Figure 7 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Figure 7 The schematic diagram of Computation mechanism, including four sub-types: Compressed (Section 4.3.1), Expanded (Section 4.3.2), Adaptive (Section 4.3.3), and Interleaved (Section 4.3.4).

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

# 4.3 Computation

In recent years, research on latent-level computation has advanced rapidly, reflecting a broader departure from the conventional paradigm in which language models perform inference through fixed-depth, token-by-token generation [46, 297]. Instead, a growing body of work aims to equip models with more flexible, efficient, and scalable computational mechanisms by shifting part of the reasoning process to the latent space.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

To enable a systematic analysis, a central question to consider is: what kinds of computational operations are performed in the latent space? From this perspective, existing methods can be organized along the dimension of operation, yielding four major categories. Specifically, based on their underlying computational operations, as illustrated in Figure 7 and Table 5, we classify existing approaches into four representative types:

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# Mechanism: Computation

• Compressed (Section 4.3.1): reduces the volume of explicit traces, internal states and crossmodal features, enhancing the efficiency while preserving expressiveness.   
• Expanded (Section 4.3.2): increases the effective computation by expanding computation along depth or width by recurrent, looped, parallel, superposition, and structural designs, enabling higher information bandwidth.   
• Adaptive (Section 4.3.3): allocates computation adaptively instead of a fixed budget by dynamic depth, width, shortcuts, halting, semantic-unit boundaries adaptation, and control adaptation, balancing capacity and efficiency flexibly.   
• Interleaved (Section 4.3.4): interleave heterogeneous generation medias to bridge explicit-latent, language-vision, reasoning-memory, or planning-perception.

> 💡 **列表批读**: 这组条目通常对应贡献、设置、发现或模块；建议逐条映射到 visual memory / latent reasoning / medical diagnosis 的主线。

# 4.3.1 Compressed

This paradigm subsumes approaches that project an explicit trajectory $\mathbf { r } \in \mathcal { V }$ onto a semantically dense latent representation $\mathbf { z }$ . A compression operator $\Phi ( \cdot )$ acts upon the intermediate hidden states $\mathbf { h }$ to yield:

![Equation 15](../images/34c4c825a3207114d91096891a7db20c53dfb4eb90b388c5f4b9a5b92e4202f8.jpg)
*Equation 15: Equation extracted by MinerU.*

> 💡 **Equation 15 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\Phi ( \cdot )$ may be realized as both a functional component $\Phi ^ { c o m p } ( \cdot )$ or the backbone $\Phi ^ { \mathrm { b a c k } }$ itself, reducing autoregressive overhead while preserving the semantic fidelity requisite for faithful downstream decoding.

Table 5 Overview of the Compressed (Section 4.3.1), Expanded (Section 4.3.2), Adaptive (Section 4.3.3), and Interleaved

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

![Table 5](../images/e708d136f0ceac65244460016e5860f1bfcc5fed8231709c9a5e15bcf9e1ab70.jpg)
*Table 5: Table 5 Overview of the Compressed (Section 4.3.1), Expanded (Section 4.3.2), Adaptive (Section 4.3.3), and Interleaved*

> 💡 **Table 5 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

Compressed computation aims to reduce the cost of explicit intermediate computation by mapping verbose reasoning trajectories or high-dimensional hidden states into compact latent representations, while preserving the semantic information necessary for accurate downstream decoding. Rather than eliminating reasoning altogether, this paradigm seeks to retain inferential content in a more information-dense form, thereby improving efficiency across textual, internal, and cross-modal reasoning processes in three forms: Traces Compression, States Compression, and Features Compression.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Traces Compression. Recent work on this direction can be organized around a shared goal: preserving inferential fidelity while reducing explicit trace lengths. HCoT [122] and SoftCoT [243] emphasize semantic alignment across abstraction levels, with HCoT [122] using hierarchical compression to skip redundant intermediate tokens and SoftCoT [243] projecting reasoning into a continuous soft-token space that decouples reasoning depth from discrete trace length, thereby supporting zero-shot transfer without task-specific retraining. In parallel, CCoT [31] models compression as variable-length latent allocation, assigning shorter representations to simpler inferences and richer ones to more complex deductions, while CODI [174] and CoLaR [188] focus on adaptive compaction through self-distillation and dynamic token-level control, respectively.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

States Compression. A complementary and potentially deeper form of compression targets not the decoded token sequence but the internal cache, whose linear growth with sequence length has become a major bottleneck for long-context and reasoning-intensive inference. Recent work explores three main approaches. KaVa [91] formulates KV-cache compression as a knowledge distillation problem, training a student model to match the output distribution of a teacher operating over the full cache. SALS [139] instead adopts a lightweight, training-free strategy by projecting the cache into a low-rank principal subspace, arguing that the dominant directions are sufficient to preserve attention patterns with bounded error. DeltaKV [57] exploits the high correlation between KV-caches across successive reasoning steps, storing semantically compressed residuals rather than full cache states, and reports particularly strong gains on multi-step reasoning tasks where inter-step redundancy is greatest.

> 💡 **批注**: 这是一段信息密度较高的论述：建议拆成“现象/问题 → 机制 → 证据/影响”三层读。

Features Compression. Recent work on latent-space compression has moved beyond text to multimodal and embodied settings, where the main bottleneck is the high-dimensional visual and action representations used in perception–action loops such as autonomous driving, robot manipulation, and embodied navigation. In visual reasoning, RoT [216] renders intermediate reasoning states as low-resolution image patches rather than token sequences, enabling subsequent steps to operate over compressed visual structure, OneLatent [133] learns a single latent visual token that distills the image context needed for downstream reasoning, achieving competitive performance with a drastically reduced visual token budget. In embodied control, LatentVLA [235] projects visual inputs into compact action latents for real-time planning, showing that low-dimensional representations can retain the information necessary for closed-loop decision making, while Future-VLA [44] extends this idea by learning future-conditioned latents that incorporate anticipated states without increasing dimensionality.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Summary. Overall, compressed reasoning can be understood as a unifying paradigm for trading explicit length for latent density. Across explicit traces, internal states, and cross-modal representations, existing methods share the same central objective: to preserve reasoning fidelity while reducing the computational and memory overhead associated with verbose intermediate representations. This suggests that effective reasoning need not always remain fully tokenized or fully materialized, and that semantically compact representations may provide a scalable foundation for efficient inference in increasingly complex reasoning systems.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

# 4.3.2 Expanded

In this category, it augments the effective computational capacity of the model by extending latent computation along the depth or width dimensions. Formally, the model iterates over a step- $T$ , width- $K$ :

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

![Equation 16](../images/77d73919b1c7d75db1c0538826dbcb08d42e3429b37b512a7e437e398d019cd0.jpg)
*Equation 16: Equation extracted by MinerU.*

> 💡 **Equation 16 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\mathbf { h } _ { t } ^ { ( k ) }$ denotes the $k$ -th latent trajectory at step $t$ , and all $K$ paths share a common initialization $\mathbf { h } ^ { ( 0 ) }$

Expanded methods increase the effective computational capacity by enlarging the latent process along different dimensions. Rather than relying on a single fixed forward pass, they enable the model to trade extra latent computation for stronger ability, improved faithfulness, and better adaptability across tasks, which could be classified into three types of expansion: Depth Expansion, Width Expansion, and Structural Expansion.

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

Depth Expansion. Depth-expanding approaches increase effective compute by reusing the same or structurally similar layers across multiple recurrent passes, allowing the model to iteratively refine a latent representation before producing an output. In recurrent pretraining, Huginn [50] introduces a lightweight model trained from scratch with recurrent depth, where a fixed transformer block is applied for a variable number of thinking steps, thereby decoupling parameter count from inference-time compute and enabling test-time scaling without adding parameters. RD-VLA [198] extends this paradigm to embodied models and shows that recurrent latent refinement improves long-horizon manipulation planning. In looped transformers, Loop [167] demonstrates that repeatedly applying the full decoder stack can induce genuinely multi-step reasoning rather than simply approximating a deeper fixed network , while LoopFormer [72] adds inputadaptive looping to allocate depth dynamically according to task complexity . From a pretraining perspective, Ouro [296] shows favorable scaling behavior along the looping dimension, with gains in faithfulness and efficiency . Relatedly, ETD [87] proposes an encode-think-decode framework in which a latent thought state is iteratively refined before decoding, yielding strong zero-shot generalization on arithmetic and commonsense tasks without explicit chain-of-thought supervision.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Width Expansion. These methods contrast with other approaches by allocating compute across multiple parallel hypotheses or latent trajectories, thereby prioritizing broad exploration over sequential refinement. This paradigm appears in several recent frameworks, SoftCoT++ [244] instantiates multiple parallel reasoning paths in continuous embedding space and aggregates them before decoding, achieving zero-shot performance gains that scale monotonically with the number of paths without requiring path-specific supervision. A similar principle underlies LatentTTS [262], which performs concurrent latent tree search and prunes partial hypotheses with a learned value function. PCCoT [224] further extends this idea by enabling multiple latent chains to run in parallel and exchange information at synchronization points, improving efficiency while preserving solution diversity; likewise, CoT2 [52] shows that parallel sampling in continuous thought space substantially benefits even GPT2-scale models on compositional reasoning tasks. In specialized settings, Bubbles [113] formulates width expansion as parallel “bubbles" whose pooled representations support zeroshot and unsupervised learning at scale, while PLR [193] applies parallel latent reasoning to sequential recommendation, where diverse latent representations better capture user preference uncertainty than a single deterministic pass.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Structural Expansion. A smaller but conceptually distinct line of work departs from simply increasing the depth or width of the computation graph and instead introduces new topological primitives for composing latent information across positions, modalities, or levels of control. Latent-SFT [37] supervises models on superposed latent chains—continuous compressions of multi-step reasoning trajectories—rather than their token-level realizations, thereby reducing decoding cost while preserving reasoning performance. Extending a similar superposition principle to visual reasoning, Laser [218] fuses multi-scale visual features into a shared latent representation instead of processing each resolution independently. At the system level, ColaVLA [152] decomposes VLA modeling into parallel but interactive high-level semantic reasoning and low-level motor planning streams, linked through cross-attention across temporal scales, which better aligns linguistic abstraction with continuous control.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Summary. In this part, approaches enrich latent reasoning by scaling computation along three complementary dimensions: depth, width, and structure. Depth expansion emphasizes iterative refinement through recurrent or looped computation, width expansion promotes parallel exploration over multiple latent hypotheses, and structural expansion introduces richer topologies for composing information across trajectories, positions, or modalities. Together, these methods share a common goal of improving reasoning performance by increasing effective calculation in latent space, while avoiding a proportional increase in model parameters.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 4.3.3 Adaptive

In this part, methods generalize the Expanded framework by conditioning both recurrence depth $T$ and trajectory width $K$ on the input complexity of $\mathbf { x }$ , rather than prescribing them as fixed hyperparameters. A learned halting functional $\tau$ governs instance-specific termination, yielding a variable-horizon computation:

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

![Equation 17](../images/7ddf86800b519b48c7c6bc21c9a18af6720b938997ae91dda40f21409fd4d7db.jpg)
*Equation 17: Equation extracted by MinerU.*

> 💡 **Equation 17 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $t \in [ 1 , T ]$ and $k \in \left[ 1 , K \right]$ may be determined at the granularity of tokens boundaries, or other adaptive strategies based on the current hidden states by a halting function $\tau ( \cdot )$ :

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

![Equation 18](../images/a9df00d6478ad8ca24168152de35c92eecf17041d42d53a1765fde8b13f36fce.jpg)
*Equation 18: Equation extracted by MinerU.*

> 💡 **Equation 18 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where the halting function could be a explicit component, or be internalized the model. This input-conditioned allocation concentrates reasoning depth where warranted by task difficulty, realizing a principled trade-off between computational frugality and expressive capacity.

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

Adaptive computation allows the model to allocate computational resources dynamically according to the complexity of the input, rather than relying on a fixed recurrence depth or trajectory width. In this setting, the amount, structure, and locus of computation become input-dependent, enabling the model to balance efficiency and capacity more flexibly. Existing methods can be broadly understood from three perspectives: Depth/Width Adaptation, Semantic Adaptation, and Control Adaptation.

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

Depth/Width Adaptation. A natural axis for adaptive computation is depth, that is, the number of steps applied. Building on this principle, TaH [49] proposes a think-and-halt mechanism that allows a shared-weight iterative model to exit early on easy tokens while allocating additional refinement steps to harder ones. LWS [146] similarly treats halting as a learned policy, jointly optimizing an auxiliary stopping variable with the language modeling objective. Moving from depth-wise iteration to latent-chain guidance, PLaT [207] learns when to terminate latent token generation before decoding, explicitly trading off depth against efficiency. At the architectural level, Dreamer [86] and SpiralFormer [263] replace fixed-depth stacks with weight-tied recurrent transformers, demonstrating that depth adaptivity can be achieved not only through learned halting policies but also through recurrent designs.

> 💡 **批注**: 这是因果/反事实修正：关注它如何剔除伪相关、突出关键病灶或视觉证据。

While depth adaptation modulates the length of the computational chain, width adaptation modulates its branching structure, i.e., the parallel hypotheses or policy trajectories explored at each step. I2B-LPO [36] learns to expand its branching factor in regions of high uncertainty and to collapse branches where confidence is sufficient, demonstrating that width adaptation is particularly impactful in long-horizon reasoning tasks where premature commitment to a single reasoning path leads to systematic failure.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Semantic Adaptation. This paradigm allocates computation at the finest granularity by assigning different budgets to individual tokens or semantic units within a single forward pass. For example, AL-CoT [267] applies token-level adaptation within the semantic, assigning more refinement steps to difficult tokens. A similar idea is developed in PonderLM-3 [97] and AdaPonderLM [180], which endow individual tokens with learned depth or halting decisions, enabling heterogeneous per-token computation and stronger semantic consistency. In contrast, DLCM [158] shifts adaptation from tokens to semantically coherent concepts that may span multiple tokens.

> 💡 **批注**: 这是因果/反事实修正：关注它如何剔除伪相关、突出关键病灶或视觉证据。

Control Adaptation. This part refers to methods that regulate computation through control signals, e.g., learned activation vectors, routing policies, or extraction mechanisms. System-1.5 [214] introduces cognitive shortcuts that allow to bypass intermediate transformer blocks according to estimated input difficulty. FR-Ponder [62] extends this idea through instance-adaptive activation steering, routing each input through a dynamically selected subset of representational subspaces based on early-layer features. RISER [257] similarly treats adaptation as the composition of latent reasoning skills encoded as steering directions in activation space, enabling zero-shot generalization to unseen tasks without gradient-based updating. Complementarily, PRE [125] shows that selective extraction of intermediate-layer representations as reversed signals, outperforming reliance on final-layer outputs.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Summary. In summary, adaptive methods generalize fixed-budget recurrent reasoning into an inputconditioned computation paradigm. Across depth/width adaptation, semantic adaptation, and control adaptation, the common goal is to allocate computation more selectively: spending more resources on difficult inputs, uncertain reasoning branches, or critical semantic units, while preserving efficiency on simpler cases. Taken together, these approaches suggest that the future of adaptive methods lie not merely in increasing computation, but in learning where, when, and how to use it most effectively.

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

# 4.3.4 Interleaved

This paradigm constructs a heterogeneous generation sequence by alternating discrete token embeddings in $\nu$ with continuous latent in $\mathcal { H }$ , yielding interleaved generation:

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

![Equation 19](../images/2548b3bbdf483af6ce2195e91ee97fdea5c38ad9d76f445a34407be3d1578fd7.jpg)
*Equation 19: Equation extracted by MinerU.*

> 💡 **Equation 19 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where the $\mathbf { r }$ generation trajectory, achieving a synergistic coupling of explicit symbolic reasoning and implicit neural computation.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Interleaved computation views reasoning as a heterogeneous sequential process in which discrete tokens and continuous latents are alternated within a unified trajectory. Compared with purely token-based or purely latent reasoning, this paradigm provides a more flexible interface for allocating computation across explicit symbolic steps and implicit neural operations. Existing work can be broadly grouped into three categories: Explicit-latent Interleaving, Modality Interleaving, and Task Interleaving.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Explicit-latent Interleaving. Hybrid generation interleaves natural-language tokens with latent internal states within a single trajectory, motivated by the observation that many intermediate steps need not be fully verbalised and can therefore be executed more efficiently without degrading performance. Early empirical support comes from Assorted [181], which shows that replacing selected verbal chain-of-thought steps with latent activations preserves downstream accuracy while substantially reducing token usage. Subsequent work extends this idea in two main directions. One line studies fixed or curriculum-based interleaving schemes: SpiralThinker [155] progressively internalises explicit reasoning steps through a spiral curriculum, and LiteReason [55] demonstrates that even a lightweight model can learn a form of mixed explicit–latent reasoning. A second line makes the explicit–latent boundary itself adaptive: SwiReasoning [175] learns a step-level switching policy via reinforcement learning, LT-Tuning [123] incorporates latent-thought positions into instruction tuning, and ThinkRouter [241] routes queries to different reasoning depths at the system level to balance efficiency and accuracy.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Modality Interleaving. When generation over non-textual inputs such as visual imagination, or perceptual feature maps, latent representations are not merely an efficiency mechanism but a representational requirement, since the underlying information cannot be fully captured in discrete text. Modality-interleaving methods therefore integrate continuous perceptual latents with linguistic tokens, allowing models to perform intermediate reasoning in the latent domain. Early work such as AURORA [11] established this paradigm by inserting explicit perceptual reasoning steps. Subsequent studies, including Mirage [251], LVR [95], VisMem [264], and Monet [211], showed that directly interleaving vision latents into architectures improves visual tasks without relying on textual scene descriptions.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

A related line of work extends interleaving to multiple latent types: IVT-LR [20] couples text and vision activations through cross-attention, while SkiLa [197], LS [276], and MM-CoT [171] use visual latents as an internal sketchpad that supports subsequent language reasoning. Beyond two-dimensional perception, 3DThinker [28] incorporates spatial latents for geometric and spatial reasoning, and DMLR [111] together with ILVR [39] shows that sparse latent interleaving can preserve strong grounding performance while reducing computational cost.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Task Interleaving. A third family views interleaving not as a mechanism for mixing token types or modalities, but as a means of coordinating heterogeneous functional modules such as memory stores, retrieval components, generative heads, and agent sub-processes. Its defining feature is that the interleaved latent stream carries task-specific structured signals, e.g., retrieved evidence, episodic memory traces, or inter-agent communications, rather than general-purpose reasoning states. In single-model settings, this idea primarily appears as memory–reasoning integration: LCR-SER explicitly interleaves a compressed history buffer with ongoing inference for sequential recommendation [177]; MemGen [273], VisMem [264] and FlashMem [65] equip models with persistent working memory through, respectively, generated memory tokens; and CLaRa [59] interleaves retrieval and generation in a jointly trained latent lookup framework, rather than treating retrieval as an external pipeline. In multi-agent settings, interleaving further serves as a coordination protocol across asynchronously operating agents: LatentMem [47] introduces a shared latent memory that supports implicit communication during collaborative problem solving, while L2-VMAS [265] alternates perceptual and planning streams across agents to maintain a coherent joint world model in long-horizon embodied tasks.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Summary. Interleaved computation generalises the generation process beyond a homogeneous token stream by allowing models to alternate between symbolic outputs and latent computations. This design improves efficiency when some intermediate steps need not be verbalised, expands representational capacity when generation must operate over non-textual modalities, and enables tighter coordination when multiple functional modules or agents interact within a shared trajectory. Taken together, these studies suggest that interleaving is not merely a decoding strategy, but a general principle for organising hybrid computation systems that combine explicit interpretability with implicit computational power.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 4.4 Optimization

Latent space optimization generally happens at three stages: pre-training, post-training, and inference. The three stages differ mainly in what is optimized: during pre-training and post-training, the optimized variable is typically model parameters $\theta \in \mathcal { W }$ (or a subset of them); during inference, the model parameters are mostly fixed (sometimes also be trained), and the optimized variable is instead an inference-time state, such as a latent representation $\mathbf { z } \in \mathcal { H }$ or a generation trajectory $\mathbf { r } \in \mathcal { V }$ . For each stage, we categorize methods based on two aspects: supervision, that specifies what provides the learning signal; objective, that captures the purpose behind each loss component for latent optimization. Based on underlying computational operations, we categorize existing methods into three representative types:

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# Mechanism: Optimization

• Pre-training (Section 4.4.1): starts with a randomly initialized model and trains it from the scratch, enabling native latent-level abilities.   
• Post-training (Section 4.4.2): enhances the ability of pre-trained models, with diverse supervision signals and objectives, learning the latent space.   
• Inference (Section 4.4.3): focuses on inference manipulation of latent states, allowing dynamic adjustment.

> 💡 **列表批读**: 这组条目通常对应贡献、设置、发现或模块；建议逐条映射到 visual memory / latent reasoning / medical diagnosis 的主线。

# 4.4.1 Pre-training

During the pre-training stage, $\mathbf { z }$ is trained jointly with the base model from scratch, the objective is to learn model parameters from large-scale pre-training data $\mathcal { D }$ . Formally, the optimization variable is $\theta \in \Phi$ , and the objective can be written as:

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

![Equation 20](../images/5f157112e44f9b8d4c043eeddb326fbf5770720fab206c60faf6b1668775803c.jpg)
*Equation 20: Equation extracted by MinerU.*

> 💡 **Equation 20 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\theta \in \Phi \subseteq \mathcal { W }$ denotes the set of trainable parameters in pre-training, and $\mathcal { L }$ trains the whole system end-to-end.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

Optimization at this stage mostly relies on simple, scalable supervision already well-established for explicitspace training, rather than more sophisticated supervision that requires human annotation or elaborate processing. To stabilize training, optional auxiliary losses are designed to regulate the latent space. In total, these methods fall into three types: Autoregressive Supervision, Auxiliary Supervision, and Reinforcement.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Autoregressive Supervision Pre-Training. This category focuses on internalizing reasoning as a natural byproduct of next-token prediction or recurrent looping, without explicitly matching intermediate representations. Implementing Jacobi-iteration-based parallel training, PonderLM-2 [267] efficiently computes recursive latent thoughts without strict causal blocking. Exploring recurrent depth scaling through looped transformer layers, Looped Trans. [167] natively simulates complex reasoning via continuous latent updates. Scaling looped language models up to 2.6 billion parameters, Ouro [296] utilizes an entropy-regularized objective for consistent latent reasoning. To address scaling efficiency, PHD-Trans. [223] optimizes predictions via dynamic exploration of continuous states. To maintain stability in dynamic scenarios, AL-CoT [268] employs continuous state prediction objectives.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Auxiliary Supervision Pre-Training. This direction explicitly guides the formation of the latent space geometry through auxiliary semantics, contrastive learning, or reconstruction objectives. Optimizing continuous semantic concepts via cross-entropy and reconstruction losses, CoCoMix [186] implicitly guides multi-task textual predictions. Incorporating InfoNCE and cross-entropy, LARES [112] aligns latent representations for sequential recommendation. Introducing latent action pretraining via VQ-VAE, LAPA [255] learns discretized latent actions directly from unannotated videos. Aligning visual latent spaces from videos with proprioceptive action spaces, CLAP [272] utilizes contrastive pretraining for robust embodied manipulation. Bypassing full visual dynamic reconstruction, JALA [131] derives predictive action embeddings jointly aligned with inverse dynamics and sparse physical actions. Focusing on mean squared error and task losses, CARE [176] scales multi-task prediction for embodied manipulation. Utilizing cross-entropy and mean squared error, ConceptLM [126] learns latent representations natively for efficient multi-task prediction.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Reinforcement Pre-Training. This category integrates reward signals directly into the foundational pretraining phase to actively shape latent thought trajectories. Integrating reinforcement signals, LoopRPT [190] reframes next-token prediction as a reasoning task via noisy latent rollouts, optimizing intermediate representations to compress effective reasoning into fewer iterations from the very beginning of the lifecycle.

Summary. Methods at this stage embed latent reasoning capacity directly into model parameters during large-scale training, foregoing the need for human annotation or elaborate supervision. The dominant approach is autoregressive prediction over continuous states, where looped or recurrent architectures naturally develop latent reasoning through standard next-token objectives. Auxiliary losses serve a regularizing role, shaping the geometry of the latent space without disrupting scalability. A nascent thread introduces reinforcement signals at pre-training time, aiming to instill efficient reasoning trajectories before any fine-tuning occurs.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 4.4.2 Post-training

During the post-training stage, $\mathbf { z }$ is further optimized via fine-tuning on a pre-trained model. Thanks to the reduced data requirements compared to pre-training, post-training supervision can involve more sophisticated data augmentation or specialized loss designs. Many methods focus on defining an additional signal to supervise the latent variable, as supervision on explicit variables alone may not be sufficient. Again, the optimized variable is still model parameters, but the optimization is now restricted to a post-training objective:

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

![Equation 21](../images/5b3ab36ec95bc7019a811e3f755debf617669ac854ebe5b39ef2644ddb896e8f.jpg)
*Equation 21: Equation extracted by MinerU.*

> 💡 **Equation 21 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where the formulation subsumes supervised fine-tuning, preference optimization, and reinforcement-learningbased alignment. In latent-space methods, post-training often specializes in refining how latent variables are induced, controlled, or aligned with downstream objectives. Based on the supervision signals, there are three types, including: Explicit Supervision, Implicit Supervision, and Reinforcement Learning.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Explicit Supervision Fine-Tuning. This category optimizes latent reasoning by applying loss functions solely to the final human-readable output, without providing step-by-step targets for the continuous representations. Fine-tuning continuous variables via specific task losses, LATPC [260] enhances safety and robustness against jailbreak attacks. Using task loss alignment, GainRouter [288] dynamically routes features to enable fast and adaptive latent reasoning. Steering hidden states on a learned latent manifold, GeoSteer [84] ensures faithful and logically consistent reasoning trajectories. Fine-tuning models with cross-entropy and regularization, PILOT [289] internalizes the strategic oversight of large models into intrinsic latent guidance. Focusing on next-token prediction and alignment, TS [2] utilizes cross-entropy to streamline latent reasoning.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Implicit Supervision Fine-Tuning. This approach provides explicit gold targets for latent representations using knowledge distillation, contrastive alignment, or step-wise reconstruction signals.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Distillation-based methods anchor latent states to teacher-provided signals. SPOT [32] compresses explicit reasoning trajectories into compact latent tokens by anchoring them to corresponding teacher spans, while SemCoT [61] distills ground-truth reasoning into semantically aligned implicit tokens to accelerate computation. Latent-SFT [37] takes a different angle, redefining latent tokens as vocabulary-space superpositions and training them with KL divergence and cross-entropy losses.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Contrastive learning constitutes another prominent technique. LTA-Thinker [206] and EPR-Latent [215] employ contrastive objectives to optimize latent thought distributions and refine intermediate representations, respectively, while ReaRec [192] applies the same principle to sequential recommendation. EBM-CoT [29] further combines contrastive signals with regularization to improve overall latent reasoning efficiency.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Table 6 Overview of the Pre-training (Section 4.4.1), Post-training (Section 4.4.2), and Inference (Section 4.4.3) optimization. We compare the modality, backbone, computational operation, and scenario. Here, Recon., CE, InfoNCE, KL, MSE, and CL denote reconstruction, cross-entropy, information noise-contrastive estimation, Kullback–Leibler divergence, mean squared error, and contrastive learning, respectively.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Table 6](../images/369a8f1f407b1c696a4df9bf84a574f9cc25a6221362833f0eece0cde536b676.jpg)
*Table 6: Table 6 Overview of the Pre-training (Section 4.4.1), Post-training (Section 4.4.2), and Inference (Section 4.4.3) optimization. We compare the modality, backbone, computational operation, and scenario. Here, Recon., CE, InfoNCE, KL, MSE, and CL denote reconstruction, cross-entropy, information noise-contrastive estimation, Kullback–Leibler divergence, mean squared error, and contrastive learning, respectively.*

> 💡 **Table 6 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

Reconstruction objectives are especially prevalent in multimodal settings, where aligning across modalities demands explicit feature-level supervision. LaViT [227] jointly predicts next tokens and reconstructs visual features to align cross-modal reasoning, and RoT [216] takes a distinctive approach by rendering chain-ofthought as images and applying MSE together with cross-entropy. BNPO [102] uses reconstruction and task losses to align embodied action spaces. Methods such as Mirage [251], 3DThinker [28], VisMem [264],

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Monet [211], and CogSense [96] further demonstrate the breadth of this paradigm across visual imagination, spatial reasoning, and generalization tasks.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Reinforcement Learning. To mitigate geometric drift, this sub-category leverages policy gradients, rewards, and preference signals to autonomously discover efficient latent trajectories.

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

Self-rewarding mechanisms form one important thread, where the model itself provides the training signal without external annotation. LaTRO [22] formulates reasoning as a variational sampling process optimized via the model’s own probability estimates, and I2B-LPO [36] employs an iterative information bottleneck with self-rewards for thorough latent policy exploration. MemGen [273] similarly relies on targeted self-rewards to enhance generalization.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

A further set of methods introduces specialized regularizers to stabilize or enrich the optimization landscape. SofT-GRPO [291] applies Gumbel reparameterization for stable group relative policy optimization, GANPO [76] introduces an adversarial regularizer to robustify preference optimization, and DLR [170] combines contrastive stability constraints with reward signals for directed latent exploration. HRPO [266] takes a hybrid approach, using a learnable gate to progressively integrate continuous hidden states with discrete tokens. CoLaR [188], LWS [146], LatentR3 [282], and Reason-IAD [24] round out the landscape by applying reward-driven optimization to dynamic trajectory prediction, sequential recommendation, and visual spatial understanding. KL divergence regularization also serves as a recurring stabilization mechanism across many methods. LARES [112], RL-Latent [149], SCM [208], and ATP-Latent [292] all incorporate KL penalties alongside reward objectives to prevent excessive deviation from the reference policy during latent space optimization. ReLaX [280] further addresses premature convergence through strict reward regularization, and RLTT [222] distributes rewards across the full trajectory to improve alignment.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Summary. Compared to pre-training, post-training affords greater flexibility in supervision design, enabling richer signals such as distillation, contrastive alignment, and reward-based feedback to refine latent representations. A central tension in this stage is whether to supervise latent variables explicitly through gold targets or implicitly through output-level task losses alone. Reinforcement learning methods go further by treating latent trajectory discovery as a policy optimization problem, allowing models to autonomously identify compact and efficient reasoning paths without relying on predetermined supervision.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 4.4.3 Inference

For inference-time latent optimization, the model weights $\theta$ are usually frozen (also could be trained) and the latent states $\mathbf { z }$ are directly manipulated at test time. Unlike pre-training and post-training, inference-time methods treat $\mathbf { z }$ itself as the optimization variable. Formally, let $\theta$ denote the trained parameters, then the optimized variable becomes $\omega$ , with:

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

![Equation 22](../images/075cb96141ed81512ea264393e634ae82c49d635bbb042a5c42505332b1a762d.jpg)
*Equation 22: Equation extracted by MinerU.*

> 💡 **Equation 22 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\Omega$ is the feasible region of the variable, and $\mathcal { I }$ is the inference-time objective, final output is then generated conditioned on the optimized inference-time state. The part includes Scaling, Tuning, and Guidance.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Inference Scaling. This category focuses on exploring the latent space by generating multiple candidate trajectories and employing reward-based heuristics to identify the optimal reasoning path. Employing a continuous-space classifier as a latent reward model, LTO [41] aggressively prunes incorrect thinking patterns during inference. Performing manifold-informed latent foresight search, TGR [298] scores candidate latent anchors to encourage smooth trajectories and diverse long-horizon exploration. Reformulating latent exploration as conditional sampling over continuous thought representations, GTS [209] utilizes a Gaussian Thought Sampler to predict context-dependent perturbation distributions. Applying self-reward sampling, LatentSeek [98] effectively alleviates catastrophic forgetting during continuous generation. Employing selfrewards, SR [295] samples and aligns soft reasoning explorations to maximize efficiency. Leveraging the latent semantic space, LatentPrompt [17] automatically evaluates and optimizes prompts via intrinsic self-rewards. Utilizing KL divergence and task losses, ITR [88] guides inference-time sampling to greatly improve complex reasoning efficiency. Enhancing visual understanding and grounding, DMLR [111] implements continuous self-reward sampling and alignment.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Inference Tuning. This approach shifts from trial-and-error stochastic search to continuous, gradientbased optimization executed directly on the latent variables or policy during the forward pass. Optimizing intermediate latent thought vectors on the fly, LTPO [258] utilizes an online policy gradient method guided by an intrinsic confidence-based reward. Integrating differentiable optimization over token logits, $\nabla$ -Reasoner [210] refines the policy directly within the decoding loop via gradient descent in the continuous sample space. Combining self-reward and reconstruction losses, LatentEvolve [274] dynamically samples and fine-tunes experimental memory states through test-time scaling.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Inference Guidance. This category applies targeted structural constraints, contrastive logic, or sparse interventions to dynamically guide latent representations and prevent hallucinations. Steering internal activations via sparse interventions, REVIS [225] mitigates object hallucination in large vision-language models by extracting pure visual information vectors. Internalizing contrastive learning and self-rewards, STIR [178] introduces a value-modulated trajectory intervention that dynamically injects context-specific impulses via anchor-based gating. Perturbing latent thoughts via specialized initial tokens, SoftCoT $^ { + + }$ [244] uses contrastive learning to enforce diversity among soft representations. Utilizing contrastive learning at inference time, VTI [120] successfully mitigates hallucinations in massive visual models. Contrasting visual features at test time, L2V-CoT [270] tightly aligns transferring and visual reasoning capabilities through latent intervention. Preventing visual hallucinations dynamically, Control++ [228] applies highly targeted task losses during the alignment generation process.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Summary. Unlike parameter-level optimization, inference-time methods treat latent states themselves as the optimization variable while keeping model weights fixed. The key distinction among approaches lies in the search strategy: scaling methods explore the latent space stochastically through reward-guided trajectory selection, optimization methods apply gradient updates directly to latent variables during decoding, and guidance methods impose structural or contrastive constraints to correct latent representations on the fly, particularly to suppress hallucinations.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 5 Ability: What Does Latent Space Enable?

The latent space, as a machine-native representational substrate within large models, unlocks a spectrum of emergent capabilities that transcend the boundaries of explicit token-level processing. In this section, we systematically examine these capabilities along seven dimensions: Reasoning (Section 5.1) concerns the ability to carry out deduction and relational computation through continuous internal states; Planning (Section 5.2) emphasizes the prospective organization of solution trajectories, resource allocation, and multi-step decisionmaking; Modeling (Section 5.3) focuses on the characterization, interpretability, controllability, and scalable depth of latent representations themselves; Perception (Section 5.4) enables models to preserve and manipulate rich, spatially structured information for more faithful visual understanding; Memory (Section 5.5) supports compact, persistent, and adaptive knowledge retention across contexts; Collaboration (Section 5.6) allows multiple agents to exchange semantic content directly through latent channels rather than discrete language alone; and Embodiment (Section 5.7) extends latent computation into physical interaction, supporting grounded action, predictive foresight, spatial imagination, and transfer across heterogeneous bodies. As depicted in Figure 8, each dimension reflects a distinct facet of intelligence that latent representations uniquely empower, ranging from internal logical deduction to physical interaction with the environment.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 5.1 Reasoning

Reasoning in latent space refers to the capacity of large models to perform logical deduction, relational computation, and conclusion generation through internal continuous representations rather than through explicit token-by-token verbalization. The shift from explicit CoT reasoning [219] to latent reasoning represents a fundamental paradigm change: instead of articulating every intermediate step in natural language, models learn to think within a continuous, high-dimensional latent manifold [58, 294].

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

This paradigm offers substantial advantages in computational efficiency, representational capacity, and the ability to encode superpositions of multiple reasoning paths simultaneously. We organize this rapidly expanding landscape along six abilities: Implicit Inference without full verbalization, Compact Trace that condenses long chains into compact states, Continuous Refinement that sustains and revises thought in latent form, Branching Path over multiple candidates, and Modal Generalization beyond text-only settings.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

![Figure 8](../images/b799cf24a1c0dac16b6ce1c6e92cd920cdb0436244a474e26c298547b5de8924.jpg)
*Figure 8: Figure 8 Core abilities brought by the latent space, including: Reasoning (Section 5.1), Planning (Section 5.2), Modeling (Section 5.3), Perception (Section 5.4), Memory (Section 5.5), Collaboration (Section 5.6), and Embodiment (Section 5.7).*

> 💡 **Figure 8 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Figure 8 Core abilities brought by the latent space, including: Reasoning (Section 5.1), Planning (Section 5.2), Modeling (Section 5.3), Perception (Section 5.4), Memory (Section 5.5), Collaboration (Section 5.6), and Embodiment (Section 5.7).

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Implicit Inference. A central motivation for moving reasoning into latent space is the growing evidence that explicit CoT, while interpretable, is often redundant and fundamentally constrained by the discrete, sequential nature of language. Converging evidence from CoT compression [122, 277], implicit capability elicitation [22, 99], and latent self-evaluation [217] shows that reasoning-like behavior is already substantially encoded in the continuous activation spaces of pretrained models. Further analyses establish latent reasoning as a distinct computational mode encompassing richer, non-sequential inference [27, 63], collectively challenging the assumption that reasoning must be externalized in tokens. Building on these insights, COCONUT [58] demonstrated that continuous thought vectors can encode superpositions of multiple reasoning paths, enabling emergent breadth-first search and showing that models can infer internally before committing to language.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Compact Trace. One major ability unlocked by latent space is compressing explicit CoT into far more compact internal states without losing its problem-solving value. A broad range of supervision and post-training studies [31, 61, 173, 174, 215, 220, 288] shows that long reasoning traces can be absorbed into compact latent states while preserving much of their functional value. This evidence suggests that already-trained models can acquire compressed reasoning ability with only modest additional overhead. The collective evidence indicates that reasoning chains can be preserved in representations orders of magnitude more compact, revealing fundamental redundancy in token-level reasoning.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Continuous Refinement. Beyond compressing existing CoT, latent space also supports the ability to sustain, blend, and iteratively revise thought as a continuous state. Soft token methods [208, 243, 287] replace discrete sampling with probability-weighted embedding mixtures or learned concept-level blending, while stochastic refinement through diffusion [83] and Markov dynamics [115] enables revision of earlier reasoning decisions. Energy-based consistency enforcement [29] and theoretical analyses of vocabulary-space superposition [37, 52] further show that such latent states can preserve coherence while remaining amenable to simultaneous optimization. Thought augmentation and contextualization [123, 129, 164, 185, 206] enrich this refinement ability by fusing task-relevant and background knowledge and modulating reasoning through targeted interventions.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Branching Path. The continuous nature of latent space enables fundamentally new reasoning topologies that let models explore several candidate trajectories at once. Parallel latent reasoning [113, 224, 244, 262] demonstrates simultaneous search through soft path sampling, stochastic width, and Jacobi iteration, reducing wall-clock latency while maintaining quality. Hybrid latent-explicit systems [155, 175, 181, 241] further suggest that models can flexibly alternate between compact internal search and selective externalization when beneficial. Studies of dual-system orchestration [33] reinforce this view, indicating that strong reasoners benefit from coordinating multiple reasoning paths and modes rather than committing to a single linear trace.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Modal Generalization. A key indicator of the maturity of latent reasoning is its ability to generalize beyond text-only settings. Modality-agnostic continuous thought [154, 160] demonstrates that the latent reasoning paradigm applies across linguistic, visual, and heterogeneous substrates, while cross-modal transfer and efficiency techniques [133, 171, 216, 270] show that this capability can move across modalities with increasing compactness. We defer detailed treatment of visual latent reasoning to the Perception subsection (Section 5.4).

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Beyond vision, domain-specific applications spanning chemical synthesis [3], narrative generation [55], novelty discovery [16], and joint-embedding prediction [110] demonstrate broad applicability. Structured spatialtemporal reasoning [203, 247] extends this generalization to geometric and temporal domains, while faithfulness and controllability studies [64, 205, 257, 279] show that latent reasoning can remain reliable as it moves across settings.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 5.2 Planning

Planning concerns the search for optimal trajectories through the solution landscape, where the continuous, differentiable nature of the latent manifold admits gradient-based policy optimization and iterative trajectory refinement. Unlike reasoning, which focuses on logical deduction within a given context, planning emphasizes the prospective organization of computation, determining where to allocate resources, how to explore the solution space, and when to terminate search [207, 240]. We examine latent planning through four abilities: Controllable Exploration over internal solution paths, Search Efficiency in navigating the latent manifold, Adaptive Budget allocation that matches compute to difficulty, and Sequential Decision in downstream interactive tasks.

> 💡 **批注**: 这是因果/反事实修正：关注它如何剔除伪相关、突出关键病灶或视觉证据。

Controllable Exploration. A central ability in latent planning is controlling internal solution trajectories rather than merely generating the next token greedily. RL-based trajectory optimization [41, 258, 266, 291] shows that continuous thought representations can be directly improved via policy gradients, Gumbel reparameterization, and test-time refinement. Training stability and diversity remain active challenges, addressed through exploration collapse prevention [36], systematic design analysis [149], and contrastive reward shaping [170]. These works collectively establish that latent geometry enables deliberate trajectory improvement that is fundamentally difficult in discrete token space.

> 💡 **批注**: 这是因果/反事实修正：关注它如何剔除伪相关、突出关键病灶或视觉证据。

Efficient Search. The latent manifold provides a natural substrate where geometric smoothness and continuity can be exploited for efficient navigation. Exploration restoration and trajectory diversification [189, 280, 295] maintain reasoning variety through entropy exploitation, latent decoding, and controlled embedding exploration, while geometry-guided search [98, 178, 298] leverages intrinsic manifold properties for instance-level targeting and long-context foresight.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

Adaptive Budget. A defining characteristic of latent planning is dynamic, input-dependent resource allocation. Adaptive depth and horizon determination [62, 146, 207, 292] adjusts reasoning depth through instance-level steering, RL-based stopping, dynamic termination, and active budget determination, investing computation proportionally to problem complexity.

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

Sequential Decision. Latent planning has been productively deployed in sequential decision-making domains, where the temporal structure of user behavior or system states naturally maps onto trajectory optimization in latent space.In recommendation, retrieval,and cross-domain adaptation [54, 112, 177, 188, 193, 204, 271, 282], latent planning improves sequential prediction, re-ranking, and transfer by maintaining and refining internal trajectories over time. In multi-step planning and tool use [25, 102, 226, 269, 289, 293], it supports sustained state tracking, optimization over intermediate decisions, and tighter control of multimodal agents, and further extends to reparameterizing the action space itself—compressing recurrent low-entropy scaffolds into compact latent action units to directly reduce the effective decision horizon while preserving executability. The breadth of these applications, spanning recommendation systems, information retrieval, tool use, action representation learning, and conversational AI, establishes latent planning as a versatile paradigm that extends well beyond the scope of traditional reasoning benchmarks.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 5.3 Modeling

Modeling encompasses the ability to characterize, inspect, and shape latent representations within large language models. While reasoning and planning concern what models compute in latent space, modeling focuses on what latent representations let us understand and control about the computation itself. We structure this dimension into four abilities: Rich Expression to encode complex computation, Self Inspection that makes internal states analyzable, Robust Control over risky or unstable behavior, and Scalable Computation that expands capacity through latent recurrence.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Rich Expression. Rigorous analysis increasingly shows that latent space supports a richer computational capacity than explicit token-only reasoning. Expressiveness analyses [239, 294] prove that continuous thought vectors encode multiple search frontiers simultaneously and achieve provably greater expressiveness than CoT, while monitorability analysis [89] reveals the associated trade-offs between efficiency and interpretability. Fundamental limitation results [166, 301] show that exploration and execution cannot be simultaneously optimized within fixed budgets, and that reasoning depth correlates weakly with correctness under certain conditions. Cognitive and domain-specific frameworks [66, 172] frame reasoning as transitions across representation spaces, extending insights from neuroscience to program understanding, together clarifying both the power and the boundaries of latent reasoning.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Self Inspection. Understanding the internal dynamics of latent reasoning is critical because latent space increasingly supports direct inspection of what the model is representing and how those states evolve. Validity probing [107, 285] examines whether latent tokens encode genuine reasoning or exploit artifacts and whether models truly reason step-by-step or develop qualitatively different strategies. Transparency and visualization methods [23, 116, 145] make internal representations interpretable through latent debate, geometry visualization, polarity-aware probing, and cognitive analysis. Representation-level analysis [143, 161] demonstrates information preservation and reveals rich semantic dimensions beyond task performance, while information flow studies [125, 195] uncover multi-hop reasoning paths and cross-lingual transition mechanisms. Practical steering [82, 84] further suggests that inspection is not merely descriptive, but can directly support targeted improvements.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Robust Control. Latent space provides a powerful but double-edged lever for model safety, because the same representations that enable strong performance can also be manipulated for both attack and defense. On the one hand, attack vectors [140, 237] exploit latent fusion and feedback-based gradients, alongside backdoor triggers embedded in latent CoT. On the other hand, layered defense mechanisms [101, 179, 250, 260] provide controllable steering, adversarial training, feature activation steering, and streaming risk detection. Training-phase safety [6, 76, 191] addresses adversarial regularization, contrastive unlearning, and human preference modeling, underscoring that robust latent control is essential for building safe systems.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Scalable Computation. Modeling also highlights the ability of latent systems to expand effective depth and capacity without proportionally expanding explicit token generation. Formal expressiveness results [167] confirm that looped transformers with latent iterations express strictly more complex computations than feedforward counterparts. Recurrent-block scaling [1, 50, 87, 296] iterates shared blocks for test-time compute, scales looped pretraining, introduces encode-think-decode architectures, and demonstrates out-of-distribution generalization. Progressive refinement and architectural variants [72, 86, 130, 263] extend this flexibility through depth-recurrent attention, elastic depth, and multi-resolution recursion. Adaptive depth allocation [49, 97, 180, 214] further enables dynamic computation through selective iteration, dual-process routing, and token-wise pondering.

> 💡 **批注**: 这是因果/反事实修正：关注它如何剔除伪相关、突出关键病灶或视觉证据。

Beyond recurrence, concept-level and efficiency innovations [121, 158] operate on adaptive semantic boundaries and achieve KV cache efficiency, while representation-level design [17, 157, 204, 223] bypasses discrete bottlenecks in prompt optimization, pretraining scaling, and deployed system efficiency.

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

# 5.4 Perception

Perception in latent space addresses the fundamental challenge of enabling large models, particularly VLMs, to understand, represent, and process visual information in continuous, high-fidelity latent spaces. Current VLMs still face a critical bottleneck: converting rich visual content into discrete text tokens inevitably discards spatial structure, fine-grained detail, and relational geometry [11, 95]. Latent perception overcomes this limitation by preserving dense, spatially-structured information that discrete tokenization necessarily destroys, enabling models to reason about visual content with the richness and nuance of human perception. We organize latent perception into three progressively deeper abilities: Multimodal Inference over internal visual representations, Heuristic Imagination for generative manipulation and 3D understanding, and Faithful Grounding that improves output faithfulness through representation-level intervention.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Multimodal Inference. A primary thrust is enabling VLMs to reason about visual content through internal latent representations rather than text-mediated descriptions. Foundational latent visual reasoning [95, 211] demonstrated that generating and updating latent visual states alongside text enables fine-grained visual understanding unattainable through text-only reasoning. Follow-up work [21, 39, 100, 183, 218] shows that this visual inference ability can remain both accurate and efficient through selective computation, coarse-to-fine processing, and joint embedding prediction without pixel-level reconstruction. Multimodal coordination and alignment studies [20, 73, 96, 111, 169, 225, 283, 284] further show that structured visual latents can be enriched, stabilized, and generalized to temporal domains.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

Heuristic Imagination. Latent perception also enables VLMs to perform visual heuristic imagination, generating and manipulating internal visual representations as part of the reasoning process, analogous to human mental imagery. This capability is valuable for tasks requiring spatial reasoning, 3D understanding, or visual planning that cannot be adequately expressed in text. Internal visual imagination [28, 251] empower latent manipulation and align VLM features with 3D foundation models, while visual scratchpads [156, 197, 276] introduce sketching mechanisms that capture dense spatial and geometric information through continuous visual tokens. Perceptual fidelity preservation [135, 227] ensures that latent visual representations maintain fidelity during distillation and dynamically refocuses attention across modality gaps.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Faithful Grounding. A critical application of latent perception is improving the faithfulness of VLM outputs by intervening at the representation level, addressing the pervasive problem of hallucination. Hallucination mitigation via latent steering and architectural alignment [120, 137] corrects visual-textual misalignments and addresses root causes, while perceptual grounding tokens [11] provide auxiliary signals through depth maps and detection outputs. Representation analysis and diagnostics [9, 90, 246, 253] reveal when perceptual failures occur and support calibrated uncertainty estimation. Domain-specific deployment in industrial and video anomaly detection [18, 24] demonstrates practical reliability improvements. These methods collectively establish latent perception as a powerful mechanism for reducing the gap between what models see and what they report, with direct implications for the reliability of deployed vision-language systems.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

# 5.5 Memory

Memory has emerged as a necessary complement to LLMs, whose stateless architecture needs external mechanisms to retain knowledge across inference steps [68]. Yet token-based memory introduces its own bottleneck: representing accumulated context as discrete sequences inflates prompt length, degrades retrieval fidelity, and prevents the gradient-based optimization needed for adaptive memory consolidation. Latent memory resolves this by encoding persistent knowledge as continuous vectors, enabling compact cross-context retention with superior fidelity and adaptability. We organize latent memory into three progressively broader abilities: Working Retention for cache intervention, Persistent Mind evolution for self-evolving knowledge stores, and Multimodal Recall grounding across visual and embodied modalities.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Working Retention. Continuous latent representations transform the KV cache from a passive record into an actively managed working memory that can be augmented, compressed, and consolidated far beyond what discrete token sequences allow. Differentiable cache injection and selective token retention schemes [117, 223] demonstrate that models can deliberate asynchronously and scale to longer sequences without proportional memory growth, while low-rank and cross-layer compression [57, 139] confirm that latent key-vector structure can be exploited to reduce cache footprint without sacrificing representational fidelity. Intrinsic consolidation [65] further shows that working memory can be synthesized directly from the model’s own transient reasoning states, where uncertainty-driven triggering eliminates the need for auxiliary encoders.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Persistent Mind. Latent representations unlock a qualitatively different memory regime where knowledge stores persist across context resets, update selectively, and differentiate into specialized functions through experience alone. Gated and generative approaches [242, 273, 286] establish that latent slots can be durably maintained through differentiable selective writing while being dynamically synthesized on demand, with planning, procedural, and working memory faculties emerging without explicit cognitive supervision. Selfevolving and retrieval-unified methods [59, 274] further demonstrate that these stores improve across queries through dual-phase episodic and procedural consolidation, and can collapse external document retrieval into the same continuous space as generation, enabling end-to-end optimization; this persistent memory regime extends naturally to multi-agent settings [47], where role-conditioned composition resolves homogenization and token-overhead bottlenecks inherent to shared context.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Multimodal Recall. Latent space imposes a structural memory barrier for visual and embodied agents: spatial layout, widget details, and temporal continuity are lost in conversion, rendering token-based memory incapable of sustaining perceptual grounding during extended generation [229, 264]. Continuous encoding methods [229, 230] prove that VLMs can compress multimodal knowledge into fixed-length embeddings compatible with frozen backbones. Unlike text-based memory under long prompts, these methods scale performance monotonically with memory depth. Cognitively structured variants [264] further reveal that organizing latent stores into complementary perceptual and semantic modules prevents systematic drift, establishing continuous latent memory as the essential substrate for anchoring complex tasks.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 5.6 Collaboration

Collective intelligence in agent systems has traditionally been mediated by natural language [60]. Yet language constitutes an inherent bottleneck: compressing internal representations into discrete tokens discards semantic nuance, increases communication latency, and breaks the gradient pathways required for joint optimization [48, 300]. Latent collaboration addresses these limitations by enabling agents to exchange continuous representations, preserving richer internal states and supporting a more expressive form of collective collaboration. We organize latent collaboration into three progressively broader abilities: Semantic Fidelity for lossless inter-agent state transfer via latent channels, Shared Cognition for identifying and evolving shared latent thought structures across agents, and Heterogeneous Interoperability for extending latent collaboration across diverse model families and modalities without architectural coupling.

> 💡 **批注**: 这里涉及动态计算或训练信号：重点看是否自适应分配推理深度。

Semantic Fidelity. Continuous representations unlock the most structurally immediate advance in multiagent collaboration: replacing token-based message passing with direct latent state transfer that preserves the full semantic content of each agent’s internal representations. Work on KV-cache and hidden-state communication [42, 48, 300] demonstrates that agents can exchange internal states without intermediate decoding, with theoretical analyses confirming that it has strictly higher expressiveness and lower complexity than text-based counterparts. Complementary alignment approaches [38] further show that a shared latent space can be learned across heterogeneous models, establishing a unified high-bandwidth collaboration channel without modifying any pre-trained parameters.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Shared Cognition. Latent representations further make the structure of shared cognition between agents identifiable and continuously evolvable, a capability that text communication fundamentally lacks. Formal latent variable analyses [47, 265, 290] prove that shared and private thought components between agents can be identified from observable outputs nonparametrically, with the global topology of thought-sharing relationships also theoretically recoverable. Strategy evolution methods [194] demonstrate that agents can update collaborative strategies by reflecting on text embeddings and propagating these reflections into external latent vectors, where stable, disentangled strategic styles emerge over long horizons without model fine-tuning.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Heterogeneous Interoperability. Latent space also enables coordination across agents of different architectures, specializations, and modalities through a shared continuous substrate rather than task-specific natural language protocols. Agent Primitives [79] show that recurring MAS interaction patterns can be abstracted into reusable latent building blocks that generalize across tasks without manual role engineering. Visual latent frameworks [124, 265] further demonstrate that perceptual and reasoning trajectories in multimodal systems can be decoupled into complementary latent memories to overcome the performance-degrading scaling wall of text-centric communication, and that the visual interface of VLMs can serve as a model-agnostic port for injecting heterogeneous reasoning traces directly into a receiver’s perceptual pathway, enabling training-free cross-family collaboration without pair-specific translators.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 5.7 Embodiment

Embodied agents confront a data bottleneck that no purely linguistic domain faces as acutely: every increment in physical diversity, e.g., new hardware morphologies, viewpoints, and task environments, invalidates existing labeled demonstrations and forces platform-specific re-collection that does not transfer [14, 114, 142]. Action in latent space compounds this by severing the continuous geometric and causal structure that manipulation requires, collapsing spatially rich dynamics into a symbolic bottleneck that discards depth, temporal continuity, and cross-embodiment correspondence. Latent representations dissolve all three failure modes simultaneously, enabling action semantics to emerge from unlabeled video, deliberate reasoning to be internalized as continuous state trajectories, and spatial priors to be distilled directly into policy backbones without instrumentation or re-annotation. We organize latent embodiment into five progressively broader abilities: Unsupervised Grounding for deriving transferable action representations from unlabeled video without embodiment-specific labels, Implicit Thinking for internalizing multi-step planning as continuous latent computation without explicit chain-of-thought generation, Predictive Foresight for simulating future states in latent space to generate dense training signals and guide real-time decision-making, Spatial Cognition for reconstructing 3D/4D geometric structure from 2D observations within the policy latent space, and Generalized Transfer for bridging heterogeneous hardware morphologies through a shared body-agnostic latent substrate.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Unsupervised Grounding. The most consequential advantage latent representations confer on embodied AI is the ability to ground action semantics from internet-scale video without any teleoperation labels, converting the scarcity of robot demonstrations from a structural ceiling into a surmountable bottleneck. Quantization and continuous codebook approaches establish that inter-frame visual transitions can be compressed into latent action tokens that generalize across embodiments and outperform label-supervised models in low-data regimes [14, 196, 255], while grounding fidelity improves further when latent objectives are constrained by physical trajectory priors, contrastive proprioceptive alignment, or disentangled structure-and-motion decomposition [10, 34, 105, 131, 249, 272]. Spatial and temporal structure serve as auxiliary grounding signals that sharpen action-relevance and suppress task-irrelevant distractors, as confirmed by frameworks that jointly address geometric and temporal bottlenecks [19] and by pretraining pipelines that interleave cross-viewpoint alignment with latent action learning [74, 77], collectively demonstrating that the quality of the latent action space, not the quantity of labeled data, is the primary determinant of downstream manipulation generalization.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Implicit Thinking. Continuous representations enable a qualitatively different mode of embodied cognition: replacing the latency-dominated, linguistically bottlenecked chain-of-thought with compact latent trajectories that carry multi-step deliberation directly into the action generation pathway. Reinforcement-learned visual plan latents and their distilled successors [69, 70] show that embodied reasoning can be grounded in actionaligned visual rewards and compressed into a handful of continuous tokens, achieving long-horizon planning and few-shot adaptation at a fraction of the inference cost of textual alternatives. Curriculum-based approaches that progressively internalize explicit chain-of-thought supervision into pure latent computation [5], architectures that iterate action predictions through recurrent latent refinement at constant memory [198], and frameworks that unify visual dynamics, spatial priors, and proprioceptive states within a single token-efficient reasoning space [128, 132, 136, 152, 187, 235] collectively establish that latent reasoning is not merely more efficient than symbolic alternatives but strictly more expressive for the continuous, spatial domain of physical control.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Predictive Foresight. Latent representations uniquely enable embodied agents to simulate future states without generating pixels, allowing imagined outcomes to serve as training supervision and real-time decision guidance rather than expensive auxiliary outputs. JEPA-style pretraining [184] demonstrates that predicting target-encoder latents of future frames in a leakage-free regime yields dynamics abstractions robust to camera motion and background variation, while world-model-derived latent distances [45] provide dense progress rewards that resolve the sparsity bottleneck of VLA reinforcement learning, achieving near-complete task mastery within two hundred RL steps from a sparse-reward baseline. Frameworks that generate foresight and action within the same latent autoregressive pass [44, 69] further demonstrate that action-aligned visual rewards and reviewable visual look-aheads can be produced in a single forward pass, establishing latent-space future simulation as a training and deployment mechanism that discrete token prediction fundamentally cannot replicate.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Spatial Cognition. Physical manipulation imposes a 3D geometric demand on policies trained from 2D observations, and latent representations are uniquely positioned to reconstruct spatial structure without requiring explicit depth sensors or 3D annotations. Knowledge distillation of frozen geometry-aware encoders into LLM visual token representations [51, 53, 142] demonstrates that aligning policy latents with 3D and 4D structural features injects spatial priors into the backbone without architectural modification, enabling precise contact-rich manipulation where appearance-only latents systematically fail. Treating dense 3D occupancy as both a latent predictive output and a supervisory signal [118] shows that volumetric spatial awareness can be developed from auto-annotated 2D data alone, while jointly addressing geometric and temporal grounding bottlenecks [19, 74] confirms that geometry-aware encoding and cross-viewpoint alignment are complementary and mutually necessary, establishing latent spatial imagination as the representational prerequisite for precise physical interaction under realistic sensor constraints.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Generalized Transfer. Cross-hardware deployment is the structural bottleneck that prevents generalist embodied intelligence from scaling: every morphological change invalidates action spaces and demands platformspecific retraining that cannot amortize across the growing diversity of robotic hardware. Latent action spaces resolve this by functioning as body-agnostic abstraction layers where semantically equivalent motions from heterogeneous embodiments converge, enabling zero-shot cross-platform deployment and data-efficient adaptation without access to task rewards or target-platform demonstrations [77, 176, 281]. Language-action disentanglement via Bayesian decomposition [106] and state-aware latent re-representation [213] further demonstrate that the same latent substrate simultaneously resolves instruction-following collapse and supports multi-modal cross-task generalization, while navigation-oriented latent alignment [182] confirms that the body-agnostic latent principle extends from manipulation to embodied navigation, establishing continuous latent action spaces as the necessary foundation for embodied intelligence that must generalize across hardware morphology, task distribution, and physical environment simultaneously.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

# 6.1 Perspective

The rise of latent space signals a fundamental reorientation in the study of language-based intelligence. Rather than treating latent space as a incidental byproduct of computation, recent research increasingly positions them as a primary substrate. In this perspective, we formulate this survey in four sequential lenses: Foundation (Section 2) intrinsically clarifies what latent space is; Evolution (Section 3) reveals how it has grown from an emerging idea into a broad research paradigm; Mechanism (Section 4) explains through what technical designs is realized; and Ability (Section 5) shows what it enables within the latent space. Viewed as a whole, these dimensions suggest that latent space is a promising candidate basis for the next generation of general-purpose intelligent systems.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Foundation. From a foundational standpoint, latent space should be understood not merely as an auxiliary hidden representation, but as a machine-native space that redefines where language-based autoregressive models compute with the semantic information [66, 239, 301]. Relative to explicit space (or verbal space) [67, 141, 231, 232], its central value lies in four representational properties: machine-native, continuous, flexible & efficient, and high-fidelity, and four functional capabilities: operable, expressive, scalability, and generalized, which reduces the redundancy, discretization bottleneck, inefficiency, and semantic loss inherent in verbalized computation. In this sense, the latent paradigm marks a shift from human-aligned generation to machineoptimal computation. At the same time, this shift introduces a constitutive tension: the gains in efficiency, expressiveness. Thus, the foundational significance of latent space is not only technical but epistemic: it changes both what models compute and how that computation can be understood, laying the groundwork for next-generation models to transcend token-centric operation.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Evolution. The evolutionary trajectory of latent-space research suggests that the field is moving from sprout toward all-round outbreak, in terms of not only the number of the works but also the diversity of paradigms. The early stage established feasibility by showing that reasoning-relevant structure already resides in internal activations and can, in part, bypass explicit verbalization. The subsequent foundation stage supplied theoretical justification, and initial multimodal extensions, thereby converting isolated demonstrations into a coherent research agenda. The later expansion and outbreak stages reveal a further transition: latent space is no longer treated as a compression trick for textual reasoning alone [27, 99, 297], but as a general framework spanning visual cognition, memory, collaboration, and embodied action. This progression indicates that latent space is best viewed not as a transient optimization of autoregressive models, but as an emerging systems principle for next-generation general intelligence.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Mechanism. From a mechanistic perspective, the survey implies that progress in latent space is driven by the co-design of four interdependent dimensions: architecture, representation, computation, and optimization. The key issue is not simply whether a model contains latent variables, but what architectures are utilized, what representations are instantiated, how computation is operated through them, and at how they are optimized. This taxonomy reveals an important trend: the field is gradually moving from heuristic usage toward systematic principle, from externally integrated to internally enabled, from static and fixed to dynamic and adaptive, and from conventional designs to multiple paradigms. Accordingly, future research is less about adding latent space to existing models than about designing models whose primary core is intrinsically latent.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Ability. In terms of ability, the most consequential contribution of latent space is that it broadens the functional scope of intelligence beyond explicit-space models. The survey shows that latent space of language-based models supports not only enhanced reasoning, but also planning, modeling, perception, memory, communication, and embodiment. What unifies these capacities is that they all require structures that are difficult, costly, or fundamentally lossy to externalize in natural language. Latent space therefore acts as a common substrate for general integration across modalities, timescales, and agents. Under this view, its ultimate promise lies not in replacing explicit textual tokens, but in enabling models to coordinate heterogeneous forms of information within a shared continuous latent space. The strongest long-term implication is that latent space may become the principal medium of general-purpose models.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Overall, the perspective presented in this survey about latent space suggests that from its foundational properties to its historical expansion, from its underlying mechanisms to its emerging abilities, latent space consistently points toward a common conclusion: future intelligent systems may rely increasingly on latent rather than purely verbal computation as their primary operating principle.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 6.2 Challenge

Despite its growing promise as a machine-native substrate for computation, latent space still faces several fundamental obstacles before it can serve as a reliable foundation for general-purpose intelligent systems. The same properties that make latent representations powerful, their continuity, compression, flexibility, and expressive capacity, also make them difficult to inspect, assess, and govern, resulting relatively low Evaluability, Controllability, and Interpretability. Together, these issues reveal that progress in latent space depends not only on improving capability, but also on making latent computation more observable, steerable, and understandable.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Evaluability. A central challenge for latent-space reasoning methods lies in their limited evaluability. Unlike explicit reasoning traces [93, 199], latent trajectories are not directly accessible to human inspection, which makes it inherently difficult to determine whether an intermediate computation is correct, complete, or even relevant to the task at hand [41, 104, 301]. This opacity substantially constrains process-level verification: researchers are often unable to distinguish between genuinely structured intermediate reasoning and representations that merely correlate with the correct output. Consequently, the evaluation of latent reasoning systems still relies predominantly on final-answer accuracy or on post hoc verbalization, both of which offer only indirect and potentially incomplete evidence about the actual reasoning process [99, 285].

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Although some recent studies have begun to propose benchmarking strategies for latent-space reasoning [56], the field still lacks mature and widely accepted protocols for supervision and evaluation [166, 301]. Existing approaches remain fragmented across tasks, datasets, and measurement criteria, and there is as yet no standardized framework for assessing the faithfulness, robustness, or internal consistency of latent reasoning trajectories. Such benchmark fragmentation and metric inconsistency make fair comparison across methods difficult, hinder cumulative progress, and complicate the identification of genuine methodological advances. As a result, improving evaluability remains one of the most pressing open problems for the development of latent reasoning models.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Controllability. Although latent space is, in principle, a highly operable substrate for computation and control, achieving reliable and generalizable manipulation of latent representations remains a substantial challenge in practice [130, 275, 297]. Fine-grained interventions on hidden states can indeed reshape model behavior in useful and sometimes remarkably precise ways; however, such interventions often suffer from relatively low controllability. The central difficulty lies not merely in identifying where and how to intervene, but in determining how high-level semantic intentions should be specified so that they are simultaneously machineactionable, sufficiently precise, and intelligible to human operators [147, 165, 299]. This tension exposes a deeper gap between continuous internal dynamics and discrete, interpretable objectives. Consequently, the development of truly controllable latent systems requires more than local steering techniques alone: it calls for principled mechanisms that can map explicit goals, safety requirements, and resource constraints onto internal computational processes in a robust and adaptive manner.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Interpretability. The difficulty lies in the very nature of latent representations: they are high-dimensional, distributed, and often deeply entangled, so that neither individual dimensions nor their induced trajectories map neatly onto stable semantic concepts [201, 252, 299]. What emerges is a representational space of immense expressive power, but one whose internal organization is resistant to human understanding. This opacity makes it challenging to explain why a model reaches a particular conclusion, to trace how information is transformed across successive stages of computation, or to determine where error and misalignment first arise [9, 103, 166, 301]. More importantly, the problem is not simply epistemic but institutional: systems that reason through inscrutable latent operations may become more powerful while simultaneously becoming less auditable, less diagnosable, and less accountable. Thus, there is still rooms for the interpretability study.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

---

## 🔖 Section 总结

### 核心洞察
1. 明确 latent/memory/alignment 的读写路径。
2. 区分训练时组件和推理时组件。
3. 关注是否能迁移到医学影像。
