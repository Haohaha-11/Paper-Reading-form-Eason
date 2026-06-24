# MedSynapse-V: Bridging Visual Perception and Clinical Intuition via Latent Memory Evolution

Chunzheng Zhu<sup>1</sup>, Jiaqi Zeng<sup>1</sup>, Junyu Jiang<sup>1</sup>, Jianxin Lin<sup>1⋆</sup>, and Yijun Wang<sup>1</sup>

Hunan University, Changsha, China {zhuchzh, zjqxxl, jiangjy, linjianxin, wyjun}@hnu.edu.cn

Abstract. High-precision medical diagnosis relies not only on static imaging features but also on the implicit diagnostic memory experts instantly invoke during image interpretation. We pinpoint a fundamental cognitive misalignment in medical VLMs caused by discrete tokenization, leading to quantization loss, long-range information dissipation, and missing case-adaptive expertise. To bridge this gap, we propose MedSynapse-V, a framework for latent diagnostic memory evolution that simulates the experiential invocation of clinicians by dynamically synthesizing implicit diagnostic memories within the model’s hidden stream. Specifically, it begins with a Meta Query for Prior Memorization mechanism, where learnable probes retrieve structured priors from an anatomical prior encoder to generate condensed implicit memories. To ensure clinical fidelity, we introduce Causal Counterfactual Refinement (CCR) which leverages reinforcement learning and counterfactual rewards derived from region-level feature masking to quantify the causal contribution of each memory, thereby pruning redundancies and aligning latent representations with diagnostic logic. This evolutionary process culminates in Intrinsic Memory Transition (IMT), a privilegedautonomous dual-branch paradigm that internalizes teacher-branch diagnostic patterns into the student-branch via full-vocabulary divergence alignment. Comprehensive empirical evaluations across multiple datasets demonstrate that MedSynapse-V, by transferring external expertise into endogenous parameters, significantly outperforms existing state-of-theart methods, particularly Chain-of-Thought (CoT) paradigms, in diagnostic accuracy and multi-dataset generalization without compromising the inference eficiency of standard VLMs.

Keywords: VLMs · Implicit Diagnostic Memory · Latent Space Memory · Causal Counterfactual · Memory Distillation

## 1 Introduction

Shaoed diagnostic experts do not rely on stepwise logical reasoning when making clinical diagnoses; instead, they activate Implicit Diagnostic Memory,

⋆ Corresponding author.

![](images/38265a02629d8413a4f7022ef1aafe8f2320fedc0ac407edfa41c5fbd72835c9.jpg)  
Fig. 1: Existing medical VLMs sufer from coarse symbolic granularity and long-range information dissipation in discrete reasoning. MedSynapse-V addresses this by evolving diagnostic implicit memory in latent space via anatomical prior condensation, causal counterfactual refinement, and autonomous latent memory internalization.

that enables near-instantaneous pattern recognition against accumulated case knowledge [3, 82, 106]. Although medical vision-language models (VLMs) have made substantial progress in diagnostic assistance [7, 53, 79, 94, 117], with reinforcement learning from verifiable rewards [47, 84, 99, 100] and chain-of-thought (CoT) [8, 21, 49, 111, 113, 118] further advancing reasoning capabilities. However, their intrinsic reliance on discrete tokens engenders a profound Cognitive Misalignment with the inherently continuous nature of clinical expertise. As illustrated in Fig. 1, the limited granularity of a fixed vocabulary is inadequate for representing continuous pathological features such as gradual transitions in lesion density or textural heterogeneity, and the autoregressive decoding mechanism is prone to progressive attenuation of visual evidence over extended reasoning chains. Moreover, discrete symbols tend to encode generic linguistic priors rather than dynamic anatomical context, readily giving rise to “pseudo-logical” hallucinations that lack grounding in physical evidence.

An intuitive remedy is to supplement models with external diagnostic knowledge. Retrieval-augmented generation (RAG) prepends retrieved text fragments or similar cases to the input context [1, 119, 133, 136, 150], while soft-prompt and prefix-tuning methods concatenate learnable vectors to the input sequence to inject domain-specific cues [24, 51, 105]. However, both strategies inject information that remains static and causally unverified: it has undergone neither validation of causal relevance to the current diagnostic decision nor evolution into an intrinsic model capability through gradient-based optimization, persisting as a brittle external dependency prone to context saturation and information redundancy as the diferential diagnosis space expands.

Recent latent computation paradigms [28, 54, 97, 124] ofer a principled alternative by performing reasoning in continuous hidden state spaces, circumventing the expressiveness bottleneck of discrete symbols. However, their direct application to medical scenarios encounters two domain-specific obstacles. First, without structured anatomical priors, latent representations degenerate into abstract vectors decoupled from clinical semantics: they capture statistical regularities of the training distribution but fail to encode the structured spatial relationships (organ topology, lesion morphology, tissue boundaries) essential for diagnostic grounding. Second, without causal calibration, the coupling between latent representations and diagnostically critical visual features remains weak, as the model may produce correct answers by exploiting spurious correlations (e.g., dataset-specific formatting cues) rather than attending to pathologically relevant regions, undermining reliability in clinical deployment.

These observations converge on a fundamental question: Can a VLM progressively evolve its latent memory to simulate clinical intuition, enabling the rapid synthesis of case-adaptive diagnostic patterns, while ensuring this autonomous internal reasoning stream and its continuous refinement efectively steer the model toward clinically reliable decisions?

This paper proposes MedSynapse-V, a framework for latent diagnostic memory evolution that addresses this question through three synergistic mechanisms operating in a progressive training paradigm. First, Meta Query for Prior Memorization (§2.2) deploys learnable meta-query probes to retrieve multi-scale spatially aware features from a frozen anatomical encoder pre-trained on large-scale segmentation tasks, condensing them into compact diagnostic implicit memory vectors that are injected into the VLM’s hidden stream. This mechanism bridges the representation gap between the encoder’s anatomical feature space and the VLM’s generation space. Second, Causal Counterfactual Refinement (CCR; §2.3) performs reinforcement learning-driven memory optimization, introducing a novel causal counterfactual reward that quantifies the causal diagnostic contribution of each memory element through region-level feature masking interventions. By contrasting model behavior under original versus intervened memory conditions, CCR systematically prunes causally irrelevant components while reinforcing those with genuine diagnostic utility. Third, Intrinsic Memory Transition (IMT; §2.4) employs a privileged-autonomous dual-branch paradigm to distill the refined diagnostic patterns from a teacher branch (with the anatomical encoder) into a lightweight student branch via full-vocabulary Jensen-Shannon divergence alignment. At inference, the anatomical encoder is entirely removed, and the model generates diagnostic memory autonomously with computational overhead nearly identical to a standard VLM.

Comprehensive evaluations across seven medical multimodal benchmarks demonstrate that MedSynapse-V consistently outperforms a broad spectrum of state-of-the-art approaches, spanning medical-specific VLMs, RL-enhanced CoT paradigms, and general-purpose latent reasoning methods, in both diagnostic accuracy and cross-domain generalization, while introducing negligible additional inference cost compared with standard VLMs. Our main contributions are:

(1) We propose MedSynapse-V, the first framework that evolves diagnostic implicit memory in latent space for medical diagnosis, shifting from static external knowledge injection to progressive, autonomous memory internalization.

(2) We design Meta Query–based Prior Memorization coupled with Causal Counterfactual Refinement (CCR), which distills anatomical priors into compact latent memory and calibrates it via counterfactual interventions to retain only causally grounded diagnostic components.

(3) We introduce Intrinsic Memory Transition (IMT), a privileged–autonomous dual-branch distillation paradigm that internalizes encoder-dependent memory into autonomously generated intrinsic memory via full-vocabulary divergence alignment, eliminating all auxiliary modules at inference.

(4) In multimodal benchmarks, MedSynapse-V consistently surpasses mainstream CoT paradigms in diagnostic accuracy while maintaining inference efficiency on par with standard VLMs, validating latent memory evolution as a principled alternative to discrete token reasoning.

## 2 Methodology

## 2.1 Problem Formulation and Architecture Overview

Problem Formulation. Given an image $X \in \mathbb { R } ^ { H \times W \times 3 }$ and a clinical query $q ,$ the objective is to generate an output sequence y containing diagnostic analysis and final conclusions. Let $\pi _ { \theta }$ denote the VLM policy, ${ \mathcal { E } } _ { a n a }$ the frozen pretrained anatomical encoder, and $\mathcal { P } _ { \phi }$ the parameterized memory synthesis module. MedSynapse-V dynamically generates a set of diagnostic implicit memory $\mathcal { M } = \{ m _ { 1 } , . . . , m _ { N } \} \in \mathbb { R } ^ { N \times d _ { h } }$ for injection into the VLM hidden stream, where $d _ { h }$ is the hidden state dimensionality. The overall policy is formalized as:

$$
\pi _ { \boldsymbol { \theta } } ( y \mid X , q ) = \pi _ { \boldsymbol { \theta } } ( y \mid X , q , \mathcal { P } _ { \boldsymbol { \phi } } ( \mathcal { E } _ { a n a } ( X ) ) ) .\tag{1}
$$

Unlike explicit CoT, which concatenates reasoning tokens to the input sequence, MedSynapse-V constructs a diagnostic experience invocation mechanism based on implicit memory within the representation space.

Architecture Overview. As illustrated in Figure 2, MedSynapse-V follows a three-stage progressive memory evolution training paradigm. Stage I: Meta Query for Prior Memorization (§2.2) extracts priors from the frozen anatomical encoder, condenses them into diagnostic implicit memory M through a learnable memory synthesis module and injects them into the VLM hidden stream, while simultaneously completing semantic alignment warmup. Stage II: Causal Counterfactual Refinement (CCR; §2.3) freezes the synthesis module and performs policy optimization within the M conditioned latent space based on GRPO, incorporating causal counterfactual rewards for memory refinement. Stage III: Intrinsic Memory Transition (IMT; §2.4) writes the refined diagnostic memory into the model’s autonomous pathway through privileged autonomous dual branch full vocabulary divergence distillation, completely removing the anatomical encoder at inference. The entire process is represented as a progressive evolution chain of diagnostic memory: ${ \bf F } _ { a n a } \xrightarrow { \mathrm { M e t a ~ Q u e r y } } \mathcal { M } \xrightarrow { \mathrm { C C R } } \mathcal { M } ^ { \star } \xrightarrow { \mathrm { I M T } } \mathcal { M } _ { a u t o 2 }$ where $\mathbf { F } _ { a n a }$ denotes the anatomical encoder output features, M is the initial memory after semantic alignment, $\mathcal { M } ^ { \star }$ is the causally refined memory, and $\mathcal { M } _ { a u t o }$ is the intrinsic memory generated by the autonomous module.

![](images/fc5b05cf1fd20d405fa34a1095f16317a476de960f5fd06f22e2e2dceb793ae0.jpg)

![](images/cb6bd175dd95506c9f8b391f54517b8b775f978f3c02f1ce12ec4506b3f62d4f.jpg)  
Fig. 2: Stages I and II of MedSynapse-V. The hook features from an encoder are condensed into diagnostic implicit memory via learnable meta-query probes and injected into the VLM hidden stream. The memory is then refined through RL with composite rewards, ensuring causal alignment between memory and clinical decision logic.

## 2.2 Meta Query for Prior Memorization

Clinical cognition research has shown that experienced physicians rapidly activate long accumulated anatomical knowledge during image interpretation, forming compressed contextualized expert memory to guide diagnosis [106]. Inspired by this finding, this stage models this process as a structured prior retrieval and memorization mechanism: the anatomical encoder extracts spatially aware features, which are then condensed into diagnostic implicit memory through a learnable synthesis module and injected into the VLM hidden states.

Structured Prior Elicitation. Given an input image X, the frozen anatomical encoder ${ \mathcal { E } } _ { a n a }$ outputs spatial features from the final layer: ${ \mathbf F } = \mathcal { E } _ { a n a } ( X ) \in$ $\underbracket { \partial } { H _ { f } } \times W _ { f } \times d _ { f }$ , where $H _ { f } \times W _ { f }$ is the spatial resolution and $d _ { f }$ is the feature dimensionality. This feature map encodes structured spatial priors learned by the anatomical encoder from large scale medical image segmentation tasks, encompassing multi-granularity information such as lesion boundaries, organ topology, and tissue textures. It is flattened into a sequence $\mathbf { S } ~ = ~ \mathbf { f l a t } ( \mathbf { \bar { F } } ) ~ \in ~ \bar { \mathbb { R } } ^ { M \times \bar { d } _ { f } }$ $M = H _ { f } \times W _ { f }$ , forming the feature pool for candidate memory.

Diagnostic Memory Synthesis. To condense the high-dimensional feature pool into compact memory compatible with the VLM hidden state space, we design a lightweight Diagnostic Memory Sampler $\mathcal { P } _ { \phi }$ that maintains N learnable meta-query probes $\mathbf { Q } _ { 0 } \in \mathbb { R } ^ { N \times d _ { f } }$ , each of which learns to attend to specific pathological semantic patterns $( e . g .$ , boundary irregularity, density heterogeneity, or vascular-tissue spatial relationships). Using $\mathbf { Q } _ { 0 }$ as queries and S as key-value pairs, $\mathcal { P } _ { \phi }$ performs selective aggregation and dimensional alignment: $\mathcal { M } = \mathcal { P } _ { \phi } ( \mathbf { Q } _ { 0 } , \mathbf { S } ) \ \in \ \mathbb { R } ^ { N \times d _ { h } }$ , where $d _ { h }$ is the VLM hidden state dimensionality. The resulting N diagnostic implicit memory elements $\mathcal { M } = \{ m _ { 1 } , . . . , m _ { N } \}$ collectively span the feature subspace required for diagnosis. $\mathcal { P } _ { \phi }$ extracts the most diagnostically relevant compact representations from the encoder’s high dimensional feature pool according to the task context, bridging them from the anatomical encoder’s representation space to the VLM’s hidden state space.

Diagnostic Memory Injection. The diagnostic implicit memory M is injected into the VLM generation sequence as continuous vectors, positioned after the question encoding and before the answer generation: $\mathbf { x } _ { f u l l } = \lbrack \mathbf { E n c } ( X ) ; \mathbf { E n c } ( q )$ ; m<sub>1</sub>; $\ldots ; m _ { N } ; y _ { 1 } ; \ldots ; y _ { T } ]$ , where Enc(·) denotes the encoding operation and $\{ y _ { t } \} _ { t = 1 } ^ { T }$ are the answer tokens to be generated. Since $\mathcal { M }$ shares the $d _ { h }$ dimensional space with VLM hidden states, the self-attention mechanism enables the generation sequence to dynamically aggregate diagnostic evidence from ${ \mathcal { M } } ,$ allowing the VLM to modulate its predictive distribution based on latent priors without additional adaptation layers or architectural modifications.

Semantic Alignment Warmup. Directly injecting outputs from a randomly initialized synthesis module into the VLM would cause semantic mismatch. To address this, a warmup stage is established before formal refinement: the VLM backbone θ and the anatomical encoder ${ \mathcal { E } } _ { a n a }$ are frozen, and only ϕ is optimized with the standard next token prediction loss:

$$
\mathcal { L } _ { w a r m } ( \phi ) = - \sum _ { t = 1 } ^ { T } \log \pi _ { \theta } ( y _ { t } ^ { \star } \mid X , q , \mathcal { M } , y _ { < t } ^ { \star } ) ,\tag{2}
$$

where $y ^ { \star }$ denotes the reference answer sequence. This stage establishes a wellconditioned initial semantic mapping between the anatomical encoder feature space and the VLM hidden state space, providing a semantically coherent and stable initialization upon which the subsequent RL stage can reliably build.

## 2.3 Causal Counterfactual Refinement

After warmup, M possesses basic semantic alignment capability, but directly applying standard SFT has two limitations [129]: binding to fixed reference trajectories restricts out-of-distribution generalization; the model may bypass M and directly map from $( X , q )$ to answers, causing memory to degenerate into redundant placeholders. In this stage, we perform policy optimization within the M-conditioned latent space, incorporating causal counterfactual intervention to guide memory refinement toward clinical alignment.

Conditioned Policy Modeling. First, we freeze $\mathcal { P } _ { \phi }$ and the VLM backbone, optimizing only lightweight LoRA adapters (collectively denoted θ below). Based on GRPO [95], for each sample $( X , q , y ^ { \star } )$ , G candidate trajectories $\{ \mathbf { o } _ { 1 } , \hdots , \mathbf { o } _ { G } \}$ are sampled. M is explicitly incorporated into the conditioned context, and the policy gradient indirectly modulates the VLM’s utilization pattern of M through the attention pathway. The policy model optimization objective is:

$$
\mathcal { I } _ { C C R } ( \theta ) = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | \mathbf { o } _ { i } | } \sum _ { t = 1 } ^ { | \mathbf { o } _ { i } | } \operatorname* { m i n } \Bigl ( \rho _ { i , t } \hat { A } _ { i } , ~ \mathrm { c } \mathrm { { 1 } } \mathbf { i p } \bigl ( \rho _ { i , t } , { 1 - \varepsilon } , { 1 + \varepsilon } \bigr ) \hat { A } _ { i } \Bigr ) ,\tag{3}
$$

where $\begin{array} { r } { \rho _ { i , t } = { \frac { \pi _ { \theta } ( \mathbf { o } _ { i , t } | X , q , \mathcal { M } , \mathbf { o } _ { i , < t } ) } { \pi _ { \theta _ { o l d } } ( \mathbf { o } _ { i , t } | X , q , \mathcal { M } , \mathbf { o } _ { i , < t } ) } } } \end{array}$ and $\begin{array} { r } { \hat { A } _ { i } = \frac { R ( \mathbf o _ { i } ) - \mu _ { G } } { \sigma _ { G } + \varepsilon _ { 0 } } } \end{array}$ , with $\mu _ { G }$ and $\sigma _ { G }$ denoting the group reward mean and standard deviation, and $\varepsilon _ { 0 }$ a stability constant. Note that both $\rho _ { i , t }$ and ${ \hat { A } } _ { i }$ are conditioned on ${ \mathcal { M } } .$ allowing gradients to propagate through the attention pathway linking answer tokens with memory, thereby shaping how the VLM attends to the injected diagnostic cues during generation. Interventional Reward Design. The composite reward $R ( \mathbf { o } ) = \lambda _ { a c c } \cdot r _ { a c c } +$ $\lambda _ { c a u s a l } \cdot r _ { c a u s a l }$ consists of two components. The diagnostic accuracy reward is:

$$
r _ { a c c } ( \mathbf { o } ) = \mathbb { I } \big [ \mathbf { a n s w e r } ( \mathbf { o } ) = y ^ { \star } \big ] .\tag{4}
$$

The causal counterfactual reward quantifies the causal contribution of M through intervention. The frozen anatomical encoder ${ \mathcal { E } } _ { a n a }$ additionally provides highestconfidence region masks for diagnostically relevant areas; zeroing out the corresponding features yields the interventional memory ${ \mathcal { M } } ^ { \prime } { : \mathcal { M } } ^ { \prime } = { \mathcal { P } } _ { \phi } { \big ( } { \mathcal { E } } _ { a n a } ( X ) { \odot } { \overline { { \mathbf { B } } } } { \big ) }$ where B is the inverted binary mask. The causal reward measures the performance discrepancy between the original and interventional conditions:

$$
r _ { c a u s a l } ( \mathbf { o } ) = \sum _ { t = 1 } ^ { | \mathbf { o } | } \log \frac { \pi _ { \theta } ( \mathbf { o } _ { t } \mid X , q , \mathcal { M } , \mathbf { o } _ { < t } ) } { \pi _ { \theta } ( \mathbf { o } _ { t } \mid X , q , \mathcal { M } ^ { \prime } , \mathbf { o } _ { < t } ) } .\tag{5}
$$

$r _ { c a u s a l } > 0$ indicates that the original memory encodes information with causal contribution to diagnosis; $r _ { c a u s a l } \le 0$ indicates that the corresponding prior is causally irrelevant to the current decision. This design follows the principle of interventional efect estimation in causal inference, ensuring that the memory retained after refinement is causally consistent with clinical decision logic.

## 2.4 Intrinsic Memory Transition

After the causal $\mathrm { R L } ,$ the model can generate high quality diagnostic outputs under the guidance of externally derived M, but the persistent dependence on the encoder at inference introduces additional computational overhead. As shown in Fig. 3, IMT reformulates this problem as learning to autonomously generate equivalent latent space embeddings under the condition of removing the auxiliary encoder, employing a privileged autonomous dual-branch paradigm to accomplish memory transition from extrinsic to intrinsic.

![](images/7e7ce3118d49cacb9f8a0d0113b8bd5a4030f951fc52d7f9acf1cc0b6e122864.jpg)  
Fig. 3: Intrinsic Memory Transition (IMT) is achieved via Jensen–Shannon divergence alignment between the teacher $( \pi ^ { + }$ , conditioned on encoder-derived $\mathcal { M } _ { p r i } )$ and student $( \pi ^ { - }$ , conditioned on $\mathcal { M } _ { a u t o } )$ branches. Gradients propagate solely to $\mathcal { A } _ { \psi } \ : .$ , enabling complete removal of the anatomical encoder at inference with negligible overhead.

Privileged Branch and Autonomous Branch. The teacher branch (privileged) retains the complete pipeline with the anatomical encoder, generating reference memory $\mathcal { M } _ { p r i } \ : = \ : \mathcal { P } _ { \phi } ( \mathcal { E } _ { a n a } ( X ) ) \ : \in \ : \mathbb { R } ^ { N \times d _ { h } }$ . The student branch (autonomous) removes the encoder and predicts equivalent memory solely through a lightweight Autonomous Memory Module $\mathcal { A } _ { \psi }$ mounted on the VLM, using the VLM’s own visual encoding features:

$$
\mathcal { M } _ { a u t o } = \mathcal { A } _ { \psi } \big ( \mathtt { E n c } _ { V L M } ( X , q ) \big ) \in \mathbb { R } ^ { N \times d _ { h } } .\tag{6}
$$

Both branches share the VLM backbone parameters $\theta ,$ ensuring that alignment is completed within the same function space.

Training Objective: Unified Full Vocabulary Divergence. The core signal of IMT is the full vocabulary divergence objective covering all generation positions. Given a trajectory $\hat { y } \sim \pi ^ { - } ( \cdot \mid X , q , \mathcal { M } _ { a u t o } )$ sampled from the student branch, the trajectory averaged per position divergence is defined as:

$$
\mathcal { L } _ { I M T } ( \psi ) = \mathbb { E } _ { ( X , q , y ^ { \star } ) } \left[ \mathbb { E } _ { \hat { y } \sim \pi ^ { - } } \left[ \frac { 1 } { \vert \hat { y } \vert } \sum _ { n = 1 } ^ { \vert \hat { y } \vert } D \big ( \pi ^ { + } ( \cdot \vert \hat { y } _ { < n } ) \big \Vert \pi ^ { - } ( \cdot \vert \hat { y } _ { < n } ) \big ) \right] \right] ,\tag{7}
$$

where $\pi ^ { + } ( \cdot  { | \hat { y } _ { < n } } ) \triangleq \pi _ { \theta } ( \cdot  { | X , q , \mathcal { M } _ { p r i } , \hat { y } _ { < n } } )$ and $\pi ^ { - } ( \cdot  { | \hat { y } _ { < n } } ) \triangleq \pi _ { \theta } ( \cdot  { | \ X , q , \mathcal { M } _ { a u t o } , \hat { y } _ { < n } } )$ are the full vocabulary next-token distributions conditioned on privileged and autonomous memory, respectively. The divergence D adopts the generalized Jensen–Shannon divergence $\mathrm { J S D } _ { \beta } \ [ 6 7 ]$ :

$$
\mathrm { J S D } _ { \beta } \left( \pi ^ { + } \| \pi ^ { - } \right) = \beta D _ { K L } \left( \pi ^ { + } \| \bar { m } \right) + \left( 1 - \beta \right) D _ { K L } \left( \pi ^ { - } \| \bar { m } \right) ,\tag{8}
$$

where $\bar { m } = \beta \pi ^ { + } + \left( 1 { - } \beta \right) \pi ^ { - }$ . The teacher branch leverages privileged memory to expose the complete next token probability landscape to the student, driving $\mathcal { A } _ { \psi }$ to learn to generate latent space embeddings behaviorally equivalent to privileged memory. Gradients backpropagate only through the student branch to $\mathcal { A } _ { \psi }$ , while the teacher branch serves as a fixed distributional target.

At inference, $\mathcal { A } _ { \psi }$ directly generates $\mathcal { M } _ { a u t o }$ from VLM visual encoding features and injects them into the hidden stream. The entire Meta Query pipeline and the anatomical encoder are completely removed, rendering the computational overhead of MedSynapse-V nearly identical to that of a standard VLM.

## 3 Experiments

## 3.1 Experimental Setup

Datasets Training data: We conduct comprehensive experiments on seven medical multimodal benchmarks spanning diverse task types and dificulty levels. Stage I (MQPM warmup) uses 50K image–text pairs from PubMedVision [7] covering radiology and pathology. Stage II (CCR) constructs a mixed-modality RL set: 3K closed-ended VQA samples from OmniMedVQA [35] training split (8 modalities: CT, MRI, X-ray, dermoscopy, fundus, OCT, pathology, ultrasound) plus 1K open-ended samples from SLAKE [71] and PathVQA [29] training sets, totaling ∼4K samples. Region masks for $r _ { c a u s a l }$ are provided by Med-SAM3 [70]. Stage III (IMT) reuses the Stage II data. Evaluation benchmarks: (i) Closed-ended VQA: VQA-RAD [48], SLAKE [71], PathVQA [29], PMC-VQA [137]; (ii) Clinical reasoning: MMMU Health & Medicine [132] (denoted MMMU\*); (iii) Expert-level reasoning: MedXpertQA-MM [160] (Total score); (iv) Multi-granularity: GMAI-MMBench [127].

Baselines We compare against four categories of methods: (1) General VLMs: Qwen3-VL-8B [2] (our base model), InternVL3-8B [157]; (2) Medical-specific VLMs: RadFM [117], LLaVA-Med [53], GMAI-VL [59], HuatuoGPT-Vision [7], BiMediX2-8B [81], MedMO-8B [14]; (3) RL-enhanced medical reasoning: MedVLM-R1-2B [84], Med-R1-3B [47], MediX-R1-8B [80], MMedExpert-R1-7B [15]; (4) Latent-space reasoning: Coconut<sup>†</sup> [28], MCOUT-Multi<sup>†</sup> [85], IVT-LR<sup>†</sup> [4] (<sup>†</sup>: adapted with identical Qwen3-VL-8B backbone and training data). We additionally report MedSynapse-V-4B on the Qwen3-VL-4B backbone to assess scalability.

Implementation Details Our framework builds upon Qwen3-VL-8B-Instruct [2]. The frozen anatomical encoder ${ \mathcal { E } } _ { a n a }$ employs MedSAM3 [70] pre-trained on largescale multi-organ segmentation datasets. Stage I freezes both VLM and $\mathcal { E } _ { a n a } ,$ training only the Diagnostic Memory Sampler $\mathcal { P } _ { \phi }$ with lr $= 2 \times 1 0 ^ { - 4 }$ for 3 epochs. The diagnostic probe count is $N { = } 1 6 ;$ $\mathcal { P } _ { \phi }$ is a 2-layer cross-attention Transformer with output dimension $d _ { m } { = } 4 0 9 6$ (matching Qwen3-VL-8B). Images are processed at native dynamic resolution following Qwen3-VL’s default configuration. Stage II freezes $\mathcal { P } _ { \phi }$ and adapts VLM via LoRA [31] (rank=64, applied to all attention layers). GRPO generates $G { = } 4$ candidate trajectories per sample, with clipping coeficient ε=0.2, reward weights $\lambda _ { a c c } { = } 1 . 0$ and $\lambda _ { c a u s a l } { = } 0 . 5$ training for 200 steps with a rollout batch size of 32. Max generation length is 1024 tokens. Stage III introduces the Autonomous Memory Module $\mathcal { A } _ { \psi }$ (2-layer MLP + LayerNorm, input from VLM’s visual encoder features), with JSD coefficient $\beta { = } 0 . 5$ , l $\mathrm { r } = 1 \times 1 0 ^ { - 4 }$ , 3 epochs. For each sample we draw one on-policy trajectory $\hat { y } \sim \pi ^ { - }$ per gradient step; the Stage II data is reused with identical preprocessing. Closed-ended VQA tasks report overall accuracy (%). For GMAI-MMBench and MedXpertQA-MM, we follow their respective oficial evaluation protocols. Inference eficiency is measured quantitatively as ms/sample and peak GPU memory (GB). More details are provided in the supplementary material.

Table 1: Comprehensive comparison on seven medical benchmarks. Base model: Qwen3-VL-8B (unless otherwise noted). MMMU\* = Health & Medicine track. <sup>†</sup>: general latent-space methods adapted to medical VQA with identical backbone and training data. ∆: absolute gap (pp) to MedSynapse-V $\left( \mathbf { w } / \ \mathcal { E } _ { a n a } \right)$ average. All results are averaged over five independent runs. Bold: best; underline: second best.
<table><tr><td>Method</td><td>Size</td><td>VQA-RAD</td><td></td><td colspan="5">SLAKE PathVQA PMC-VQA MMMU*|MedXpert</td><td>GMAI Average</td><td>∆ (pp)</td></tr><tr><td colspan="9">General Vision-Language Models</td></tr><tr><td>Qwen3-VL-8B [2]</td><td>8B</td><td>58.6</td><td>66.2</td><td>55.4</td><td>42.5</td><td>48.3</td><td>22.1</td><td>47.2</td><td>48.6</td><td>-12.8</td></tr><tr><td>InternVL3-8B [157]</td><td>8B</td><td>57.3</td><td>64.8</td><td>50.6</td><td>41.2</td><td>50.1</td><td>21.5</td><td>49.6</td><td>47.9</td><td>-13.5</td></tr><tr><td colspan="9">Medical-Specific VLMs</td></tr><tr><td>RadFM [117]</td><td>14B</td><td>50.6</td><td>34.6</td><td>38.7</td><td>25.9</td><td>27.0</td><td>17.3</td><td>28.5</td><td>31.8</td><td>-29.6</td></tr><tr><td>LLaVA-Med [53]</td><td>7B</td><td>51.4</td><td>48.6</td><td>56.8</td><td>30.1</td><td>36.9</td><td>19.7</td><td>31.2</td><td>39.2</td><td>-22.2</td></tr><tr><td>GMAI-VL [59]</td><td>7B</td><td>64.6</td><td>71.9</td><td>47.2</td><td>52.3</td><td>51.2</td><td>23.8</td><td>45.2</td><td>50.9</td><td>-10.5</td></tr><tr><td>HuatuoGPT-V [7]</td><td>7B</td><td>63.8</td><td>74.5</td><td>59.9</td><td>53.4</td><td>49.1</td><td>22.7</td><td>51.3</td><td>53.5</td><td>-7.9</td></tr><tr><td>BiMediX2 [81]</td><td>8B</td><td>62.4</td><td>68.3</td><td>52.7</td><td>42.8</td><td>48.6</td><td>22.2</td><td>34.6</td><td>47.4</td><td>-14.0</td></tr><tr><td>MedMO-8B [14]</td><td>8B</td><td>59.3</td><td>66.8</td><td>48.5</td><td>36.0</td><td>46.2</td><td>20.8</td><td>38.2</td><td>45.1</td><td>-16.3</td></tr><tr><td colspan="9">RL-Enhanced Medical CoT / Latent Reasoning</td></tr><tr><td>MedVLM-R1 [84]</td><td>2B</td><td>58.6</td><td>63.2</td><td>42.5</td><td>35.8</td><td>31.2</td><td>16.8</td><td>33.5</td><td>40.2</td><td>-21.2</td></tr><tr><td>Med-R1 [47]</td><td>3B</td><td>53.2</td><td>52.8</td><td>44.1</td><td>38.5</td><td>30.4</td><td>18.5</td><td>35.2</td><td>38.9</td><td>-22.5</td></tr><tr><td>MediX-R1 [80]</td><td>8B</td><td>56.4</td><td>65.8</td><td>44.2</td><td>56.2</td><td>53.5</td><td>24.9</td><td>48.2</td><td>49.9</td><td>-11.5</td></tr><tr><td>MMedExpert-R1 [15]</td><td>7B</td><td>65.2</td><td>72.8</td><td>58.1</td><td>56.8</td><td>57.3</td><td>27.5</td><td>52.1</td><td>55.7</td><td>-5.7</td></tr><tr><td>Coconut† [28]</td><td>8B</td><td>55.8</td><td>63.4</td><td>50.1</td><td>41.6</td><td>43.2</td><td>19.8</td><td>37.6</td><td>44.5</td><td>-16.9</td></tr><tr><td>MCOUT-Multi† [85]</td><td>8B</td><td>59.2</td><td>67.4</td><td>53.8</td><td>44.8</td><td>47.5</td><td>22.0</td><td>40.9</td><td>47.9</td><td>-13.5</td></tr><tr><td>IVT-LR† [4]</td><td>8B</td><td>62.3</td><td>70.1</td><td>56.2</td><td>47.8</td><td>50.4</td><td>23.5</td><td>43.1</td><td>50.5</td><td>-10.9</td></tr><tr><td>MedSynapse-V-4B (w/ Eana)|</td><td>4B</td><td>67.8</td><td>75.2</td><td>60.5</td><td>53.1</td><td>54.5</td><td>24.2</td><td>47.3</td><td>54.7</td><td>-6.7</td></tr><tr><td>MedSynapse-V-4B (IMT)</td><td>4B</td><td>66.0</td><td>73.1</td><td>58.6</td><td>51.2</td><td>52.4</td><td>22.6</td><td>45.0</td><td>52.7</td><td>-8.7</td></tr><tr><td>MedSynapse-V (w/ εana)</td><td>8B</td><td>75.6</td><td>81.4</td><td>66.2</td><td>59.8</td><td>62.7</td><td>29.4</td><td>54.8</td><td>61.4</td><td></td></tr><tr><td>MedSynapse-V (IMT)</td><td>8B</td><td>74.2</td><td>79.8</td><td>64.8</td><td>58.5</td><td>61.4</td><td>26.8</td><td>51.6</td><td>59.6</td><td>-1.8</td></tr></table>

## 3.2 Main Results

As shown in Table 1, MedSynapse-V $\left( \mathrm { w } / \mathcal { E } _ { a n a } \right)$ achieves the highest average of 61.4%, and the encoder-free MedSynapse-V (IMT) retains 59.6%, surpassing all baselines. Compared to the strongest RL baseline MMedExpert-R1 (55.7%), MedSynapse-V (IMT) leads by +3.9 pp without any auxiliary module at inference, with the largest margins on visual-grounding benchmarks (VQA-RAD +9.0, SLAKE +7.0, PathVQA +6.7), where discrete CoT tokens are prone to attenuating early visual evidence across long reasoning chains. On GMAI-MMBench spanning 38 modalities, MedSynapse-V scores 54.8%, confirming that the anatomical priors generalize beyond the training distribution.

RL baselines reveal a specialization dilemma. MediX-R1 benefits from multilingual pretraining and leads on PMC-VQA (56.2%), yet this breadth dilutes radiology-specific precision (VQA-RAD: 56.4%); MMedExpert-R1 achieves the most balanced profile by leveraging guideline-based reward. Small-scale models (MedVLM-R1 2B, Med-R1 3B) collapse on out-of-domain tasks (MedXpert below 19%), confirming that parameter capacity sets a hard ceiling RL alone cannot raise. n contrast, MedSynapse-V sidesteps this dilemma by injecting latent priors that benefit all task types uniformly, achieving the top performance on every benchmark without task-specific tuning.

Latent methods require domain priors. Among adapted latent baselines, the hierarchy Coconut (44.5%) < MCOUT-Multi $( 4 7 . 9 \% ) < \mathrm { I V T - L R }$ (50.5%) tracks optimization sophistication, yet even IVT-LR barely exceeds zero-shot Qwen3-VL-8B (48.6%). This inversion reveals that latent compression without clinical grounding encodes statistical shortcuts rather than diagnostic logic; the

![](images/491934d34c21af62fd01e48e2bb580c052802ccb7d2ad83578c181931694ac5f.jpg)

![](images/e975ec00260700cf07f559e05c6f23de4366e9c2710ea4511c56aab7857f4b3e.jpg)

![](images/257fb56f2e7794875994df3d7029d5c1866a99c215958d765c409a7430d6a6b5.jpg)

![](images/e9c80051dd97bb87278bd0511410b9832d86794ff51c79ae3197cfd486f750ae.jpg)  
Fig. 4: Efect of diagnostic probe count N. Performance peaks around N=16 across benchmarks; further increasing N dilutes diagnostically relevant signals.

Table 2: Ablation study on the 8B backbone. All variants undergo the full threestage pipeline including IMT (§2.4); results reflect inference without the anatomical encoder unless stated otherwise. MQPM: Meta Query for Prior Memorization (§2.2); CCR: Causal Counterfactual Refinement (§2.3); M: diagnostic implicit memory. Best per group in bold.
<table><tr><td>Configuration</td><td colspan="7">|VQA-RAD SLAKE PathVQA PMC-VQA MMMU*|ms/token↓ Mem (GB)↓|</td><td>Avg.</td></tr><tr><td>Qwen3-VL-8B (zero-shot, no M)</td><td>58.6</td><td>66.2</td><td>55.4</td><td>42.5</td><td>48.3</td><td>126</td><td>16.2</td><td>54.2</td></tr><tr><td>(a) Progressive Training Stages</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MQPM → IMT (Stage I only, no RL)</td><td>59.5</td><td>67.4</td><td>57.1</td><td>45.6</td><td>47.2</td><td>104</td><td>16.5</td><td>55.4</td></tr><tr><td>MQPM → SFT → IMT (no RL)</td><td>63.2</td><td>71.2</td><td>60.4</td><td>49.8</td><td>51.3</td><td>104</td><td>16.5</td><td>59.2</td></tr><tr><td>Direct RL → IMT (skip MQPM)</td><td>57.4</td><td>64.3</td><td>54.8</td><td>43.1</td><td>44.7</td><td>105</td><td>16.5</td><td>52.9</td></tr><tr><td>MQPM → CCR → IMT (full)</td><td>74.2</td><td>79.8</td><td>64.8</td><td>58.5</td><td>61.4</td><td>102</td><td>16.5</td><td>67.7</td></tr><tr><td>(b) Reward Design</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>racc only</td><td>70.1</td><td>75.8</td><td>61.2</td><td>54.3</td><td>56.6</td><td>102</td><td>16.5</td><td>63.6</td></tr><tr><td>racc + rcausal (full)</td><td>74.2</td><td>79.8</td><td>64.8</td><td>58.5</td><td>61.4</td><td>102</td><td>16.5</td><td>67.7</td></tr><tr><td>(c) Encoder Retention vs. Removal</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/ / Eana (no privileged branch)</td><td>75.6</td><td>81.4</td><td>66.2</td><td>59.8</td><td>62.7</td><td>168</td><td>22.8</td><td>69.1</td></tr><tr><td>w/o Eana (IMT, default)</td><td>74.2</td><td>79.8</td><td>64.8</td><td>58.5</td><td>61.4</td><td>102</td><td>16.5</td><td>67.7</td></tr><tr><td>(d) Encoder Choice</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SÁM-Med2D [11]</td><td>69.0</td><td>75.4</td><td>60.8</td><td>54.1</td><td>57.0</td><td>102</td><td>16.5</td><td>63.3</td></tr><tr><td>MedSAM3 [70] (default)</td><td>74.2</td><td>79.8</td><td>64.8</td><td>58.5</td><td>61.4</td><td>102</td><td>16.5</td><td>67.7</td></tr><tr><td>Random init. → IMT</td><td>56.4</td><td>64.0</td><td>55.1</td><td>41.2</td><td>43.3</td><td>102</td><td>16.5</td><td>52.0</td></tr></table>

10.9 pp gap to MedSynapse-V confirms that prior injection and causal calibration are prerequisites for efective latent reasoning in medicine.

Scaling eficiency. MedSynapse-V-4B $\left( \mathrm { w } / \mathcal { E } _ { a n a } \right)$ reaches 54.7% with roughly half the parameters of 7B baselines; after encoder removal the IMT variant still achieves 52.7% at 85 ms and 10.8 GB, surpassing MediX-R1-8B (49.9%). This eficiency stems from a structural advantage: diagnostic expertise is distilled into 16 compact memory vectors consumed in a single forward pass, rather than spread across 150+ verbose reasoning tokens.

## 3.3 Ablation Study

Table 2 reports comprehensive ablation study results. Specifically, (i) Progressive training stages. MQPM warmup is indispensable: skipping it collapses Avg to 52.9, barely above zero-shot (54.2), because randomly initialized memory destabilizes early RL training. Replacing CCR with SFT reaches 59.2 but lags by 8.5 pp due to limited out-of-distribution generalization. The full pipeline (Avg 67.7) confirms non-redundant contributions: MQPM grounds semantics, CCR refines via exploration, IMT compresses into an autonomous pathway. (ii) Reward design. r<sub>causal</sub> is the dominant reward component (+4.1 pp, 63.6 → 67.7). Without causal pressure the model bypasses M via direct shortcuts, treating memory as inert padding; the counterfactual intervention penalizes trajectories insensitive to diagnostic regions. The efect concentrates on radiology benchmarks and persists after IMT, indicating stronger memory utilization transfers more faithfully through distillation. (iii) Encoder retention vs. removal. IMT achieves near-lossless removal: only 1.4 pp degradation (69.1 → 67.7) while latency drops 39% and memory decreases 6.3 GB. The gap is not uniform: core VQA metrics degrade minimally, whereas MedXpert and GMAI sufer more, suggesting complex reasoning depends more on encoder-derived priors than closed-ended recognition. (iv) Anatomical encoder choice. MedSAM3 outperforms SAM-Med2D by 4.4 pp (67.7 vs. 63.3), reflecting richer spatial representations from multi-organ segmentation pretraining. Random initialization yields only 52.0, confirming that gains originate from what the encoder knows, rather than how memory is aggregated. (v) Probe count N. As shown in Fig. 4, N=16 balances expressiveness against redundancy. The CCR to SFT gap widens with N (3.5 pp at N=4 vs. 7.2 pp at N=16), revealing that larger memory pools amplify bypass shortcuts and therefore benefit disproportionately from causal refinement.

![](images/95e90427940859b317545f42e02d737c4d248f804764a230e4b436e4bcf87853.jpg)  
Fig. 5: Qualitative comparison across CT, MRI, and Ultrasound cases. MedSynapse-V produces concise, correct diagnoses, while Med-R1 and MMedExpert-R1 generate verbose CoT with hallucinated findings (red) leading to misdiagnoses.

## 3.4 In-Depth Case Analysis

As illustrated in Fig. 5, we compare MedSynapse-V with Med-R1 [47] and MMedExpert-R1 [15] across three distinct imaging modalities. Both baselines produce verbose CoT reasoning (∼185–238 tokens) yet arrive at incorrect diagnoses due to hallucinated observations erroneously propagating through the chain. In the CT case, Med-R1 fabricates pleural thickening in the left upper lobe, while MMedExpert-R1 hallucinates a laminated calcification pattern and mischaracterizes the nodule as a benign granuloma. In the MRI case, Med-R1 misidentifies the extra-axial mass as intra-axial and concludes glioblastoma, whereas MMedExpert-R1 fabricates ring enhancement with central necrosis, both missing the classic meningioma presentation. In the ultrasound case, Med-R1 hallucinates gallbladder wall thickening to over-diagnose acute cholecystitis, while MMedExpert-R1 denies posterior acoustic shadowing and misdiagnoses a gallbladder polyp. In contrast, MedSynapse-V generates concise, correct answers (∼34–44 tokens) without explicit CoT, demonstrating that diagnostic implicit memory provides suficient latent guidance while avoiding the hallucination cascades inherent in token-level CoT.

## 3.5 Eficiency, RL Dynamics, and Latent Space

Performance–eficiency trade-of. As shown in Fig. 6, MedSynapse-V (IMT) achieves 59.6% at 2.6 s/sample, comparable to zero-shot Qwen3-VL-8B (48.6%, 2.8 s) since both share the same backbone and the 16 memory vectors add negligible overhead. Full-scale 7–8B CoT methods (MediX-R1, MMedExpert-R1) require 5.8 s each due to 300–400 autoregressive reasoning tokens, while smaller CoT models (MedVLM-R1 2B, Med-R1 3B) ofset verbosity with faster per-token speed yet remain 18–21 pp below MedSynapse-V. This confirms that compact latent memory provides diagnostic grounding without the token-generation overhead of full-scale CoT.

![](images/493b106681a412805397b7c4331d812e309be7c0958da4f640770e0d1c83d7aa.jpg)  
Fig. 6: Accuracy–latency trade-of across compared VLM categories.

Training dynamics. Fig. 7 shows the full model $\textit { \textbf { ( w / } \textit { r } _ { c a u s a l } ) }$ improving steadily to ${ \sim } 0 . 8 8$ with a transient exploration dip near step 900, where the policy sacrifices reward to explore memory-reliant generation strategies, while the $w / o \ r _ { c a u s a l }$ ablation plateaus at ∼0.48 throughout the training. This confirms that accuracy-only reward cannot distinguish memory-dependent from shortcut trajectories; without causal pressure the model bypasses M entirely, treating injected memory as inert padding.

![](images/5645229aee22f0c4ded48894327f0ca17f8313a2869e09c8b420efd0be8c7da4.jpg)  
Fig. 7: The RL training reward dynamics with and without $r _ { c a u s a l } .$

![](images/667e1b408ebe33c23efbbb447e9e61e65788043ba5de8d095b1106331ada6428.jpg)  
Fig. 8: t-SNE visualization of implicit memory $\mathcal { M } _ { a u t o }$ after CCR. (a) Eight imaging modalities form well-separated clusters with clinically coherent proximity. (b, c) Within CT and Pathology, disease subtypes further segregate into distinct regions.

Latent space structure. Fig. 8 visualizes the evolved memory $\mathcal { M } _ { a u t o }$ via t-SNE across three granularities. At the cross-modality level (a), eight imaging types form compact clusters with clinically coherent proximity (e.g., CT and MRI lie adjacent; dermoscopy and fundus form a nearby pair). Within individual modalities (b, c), disease subtypes further segregate: CT memory separates lung nodules, liver lesions, kidney cysts, pneumonia, and aortic aneurysms, while pathology memory distinguishes adenocarcinoma, squamous cell carcinoma, normal tissue, lymphoma, and melanoma. This hierarchical organization confirms that $r _ { c a u s a l }$ reshapes the latent space into a diagnostically meaningful manifold rather than merely boosting task accuracy.

Why latent memory evolution works. Our ablations pinpoint two necessary conditions that general latent methods lack. First, structured priors are indispensable: replacing MedSAM3 with a random encoder collapses Avg from 67.7% to 52.0% (Table 2d). Second, causal calibration activates the priors: r<sub>causal</sub> lifts accuracy by 4.1 pp (Table 2b) and reorganizes memory into the hierarchical diagnostic manifold shown in Fig. 8. Neither condition alone sufices, and their synergy is precisely what general latent methods lack.

## 4 Conclusion and Future Work

We propose MedSynapse-V, a medical vision-language model that performs clinical reasoning through compact latent tokens rather than explicit chain-ofthought generation. By combining causal counterfactual rewards with progressive memory evolution, our approach efectively internalizes diagnostic reasoning within a low-latency framework. Experiments across multiple medical benchmarks show that MedSynapse-V outperforms existing medical VLMs, generalpurpose VLMs, and RL-based CoT methods in both accuracy and eficiency, confirming that latent cognitive processes guided by well-designed rewards can efectively replace verbose explicit reasoning in the medical domain.

Table 3: Complete hyperparameter configuration for all three training stages.
<table><tr><td>Hyperparameter</td><td>Value</td><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Base Architecture</td><td colspan="2">|Stage II: CCR</td></tr><tr><td>VLM Backbone</td><td>Qwen3-VL-8B-Instruct</td><td>|Trainable Module</td><td>LoRA adapters on VLM</td></tr><tr><td>Anatomical Encoder  $\mathcal { E } _ { a n a }$  Hidden Dimension dh</td><td>MedSAM3 (ViT-B, frozen)</td><td colspan="2">LoRA Rank r / Alpha α 64 / 128</td></tr><tr><td>Diagnostic Probe Count N 16</td><td>4096</td><td>[LoRA Dropout LoRA Parameters</td><td>0.05 ~83.9M</td></tr><tr><td></td><td></td><td>RL Algorithm</td><td></td></tr><tr><td colspan="2">Stage I: MQPM Warmup</td><td>|Group Size G</td><td>GRPO</td></tr><tr><td>Trainable Module</td><td>Pφ only</td><td></td><td>4</td></tr><tr><td>Pφ Attention Heads</td><td>8</td><td>Clipping Coefficient ε</td><td>0.2</td></tr><tr><td> $\mathcal { P } _ { \phi }$  FFN Dimension</td><td>4096</td><td>λacc / λcausal Max Generation Length 1024 tokens</td><td>1.0 / 0.5</td></tr><tr><td> $\mathcal { P } _ { \phi }$  Parameters Learning Rate</td><td>~12.6M  $2 \times 1 0 ^ { - 4 }$ </td><td>Rollout Batch Size</td><td></td></tr><tr><td>Optimizer</td><td></td><td></td><td>32</td></tr><tr><td>Weight Decay</td><td>AdamW (β1=0.9, β2=0.999)</td><td>Training Steps</td><td>200</td></tr><tr><td></td><td> $1 \times 1 0 ^ { - 2 }$ </td><td>Learning Rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Training Epochs</td><td>3</td><td>Temperature (sampling) 0.7</td><td></td></tr><tr><td>Batch Size</td><td>32</td><td colspan="2"></td></tr><tr><td>Warmup Ratio</td><td>0.03</td><td>Stage III: IMT</td><td></td></tr><tr><td>LR Schedule</td><td>Cosine Annealing</td><td>Trainable Module</td><td>Aψ only</td></tr><tr><td colspan="2">Infrastructure</td><td>|Aψ Architecture</td><td>2-layer MLP + LayerNorm</td></tr><tr><td>GPUs</td><td>4× A100 (80GB)</td><td>|Aψ Hidden Dim</td><td>4096</td></tr><tr><td>Gradient Accumulation</td><td>2 steps</td><td>Aψ Parameters</td><td>~33.6M</td></tr><tr><td>Mixed Precision</td><td>bf16 + FlashAttention-2</td><td>JSD Coefficient β</td><td>0.5</td></tr><tr><td>Total Training Time</td><td>~38 hours</td><td>Learning Rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Cross-validation</td><td>5-fold random seed</td><td>Training Epochs</td><td>3</td></tr></table>

Looking ahead, we aim to extend latent memory evolution to longitudinal analysis and multi-modal report generation by integrating heterogeneous clinical evidence sources. Our research will further investigate scaling implicit memory to accommodate broader diferential diagnosis spaces with hundreds of competing hypotheses, validating the generalizability of latent cognitive architectures for complex clinical decision-making in high-stakes diagnostic environments.

## 5 Implementation Details

Training Configuration Table 3 provides the hyperparameter configuration across all three training stages for reproducibility. We employ standard data augmentation techniques to improve training robustness, including random rotation (±15), horizontal flipping (probability 0.5), brightness/contrast adjustment (±10%), and color jittering, while preserving critical diagnostic features and anatomical orientations. Images are processed at native dynamic resolution following Qwen3- VL’s default configuration (min pixels=256×28×28, max pixels=1280×28×28). All experiments are conducted five times and we report the mean.

Architectural Details The architecture comprises several integrated components: the Diagnostic Memory Sampler $\mathcal { P } _ { \phi }$ is implemented as a 2- layer (L=2) Transformer featuring 8 heads (head dimension 128) and 16 meta-query probes $\dot { \bf Q } _ { 0 } \in \mathbb { R } ^ { 1 6 \times 1 0 2 4 }$ initialized via a truncated normal distribution $( \sigma ~ = ~ 0 . 0 2 )$ , followed by a final linear projection to the 4096-dimensional hidden space; concurrently, the Autonomous Memory Mod ule $\mathcal { A } _ { \psi }$ processes pooled visual features through two 4096-dimensional linear layers with GELU activation and LayerNorm to produce an $N \times d _ { h }$ representation. For anatomical encoding, we uti-

![](images/6f6184d09be49f579daf375b3e9849fb8e6ff8db905a6579cfe073b33e77d50d.jpg)  
Fig. 9: Detailed architecture of the Diagnostic Memory Sampler $\mathcal { P } _ { \phi }$ . The frozen anatomical encoder $\mathcal { E } _ { a n a }$ extracts spatial features ${ \textbf { F } } \in$ $\mathbb { R } ^ { H _ { f } \times W _ { f } \times d _ { f } }$ , which are flattened into a token sequence and used as key–value pairs for the learnable meta-query probes $\mathbf { Q } _ { 0 }$ . Through L layers of selfattention, feed-forward processing, cross-attention, and a final linear projection $( d _ { f }  d _ { h } )$ , the module produces N compact implicit memory $\mathcal { M } \in \mathbb { R } ^ { N \times d _ { h } }$ that are injected into the VLM hidden stream between the question encoding and answer positions.

lize the MedSAM3 ViT-B backbone pretrained on 11 imaging modalities, which extracts $6 4 \times 6 4 \times 1 0 2 4$ spatial features (flattened to M = 4096 tokens) and provides highest-confidence region masks B via its segmentation head (threshold 0.7) to guide the causal counterfactual reward. Finally, the model is optimized in Stage II using LoRA adapters $( r = 6 4 , \alpha = 1 2 8 )$ applied to all attention projection matrices across the 32 layers of Qwen3-VL-8B, resulting in approximately 83.9M trainable parameters (∼1.0% of the backbone) to ensure eficient RL-driven adaptation while preserving the integrity of the pretrained knowledge.

Evaluation Details For quantitative evaluation, VQA-RAD, PMC-VQA, MMMU\* MedXpertQA-MM, and GMAI-MMBench are evaluated exclusively with the closed-ended template (Fig. 15). SLAKE and PathVQA contain both closedended and open-ended subsets: the corresponding template is applied to each subset respectively, and overall accuracy is reported by aggregating both. For closed-ended VQA tasks, we extract the predicted answer by matching the first occurrence of option letters $\mathrm { ( A / B / C / D / E ) }$ in the generated response. If no explicit option is found, we perform fuzzy string matching against candidate answers. For GMAI-MMBench and MedXpertQA-MM, we follow their respective oficial evaluation scripts to ensure cross-study comparability. All evaluations use greedy decoding (temperature=0, top-p=1.0) with a maximum generation length of 512 tokens. The 16 diagnostic memory vectors are injected at positions immediately following the question encoding, as described in §3.2 of the main paper.

![](images/23d42c707a9e90cd2d8165b4c8d99f27621af89b8080f68222e81c552c52d99b.jpg)  
Fig. 10: Training dynamics across three stages: (a-c) Stage II reward optimization and gradient stabilization via causal refinement; (d) Stage I NTP loss convergence; (e) Stage II policy-KL evolution; (f) Stage III distillation fidelity and output agreement.

## 6 Training Dynamics Analysis

Figure 10 extends the Stage II reward summary in Fig. 7 of the main text with additional monitoring metrics across all three stages.

Stage I (panel d): the NTP loss of $\mathcal { P } _ { \phi }$ drops sharply from ${ \sim } 2 . 6$ to ∼0.45 within the first epoch and converges smoothly across three epochs, with minor jumps at epoch boundaries due to learning-rate scheduling. This confirms that the semantic alignment warmup provides a well-conditioned initialization for subsequent RL. Stage II (panels a–c, e): the full model reward (panel a) climbs to ∼0.88 with a transient exploration dip near step 150, while the ablation $\left( \mathrm { w } / \mathrm { o } \ r _ { c a u s a l } \right)$ stalls at ∼0.48. Panel (b) shows $r _ { c a u s a l }$ rising from near-zero to ∼0.35, confirming progressive memory utilization. Panel (c) reveals that $r _ { c a u s a l }$ stabilizes gradient norms within [0.2, 0.6], whereas removing it produces frequent spikes exceeding the clip threshold, indicating an unstable optimization surface. Panel (e) shows monotonically decreasing policy loss with well-controlled KL divergence $\left( < 0 . 0 2 \right)$ Stage III (panel f): the JSD between teacher and student branches drops from ∼0.42 to ∼0.035, while output agreement rises from 72% to ${ \sim } 9 7 \%$ , consistent with the near-lossless encoder removal (∆=1.4 pp) reported in Table 2c.

## 7 Benchmark Dataset Statistics

We evaluate our model across below medical benchmarks, using oficial test splits where available. Our evaluation suite covers both closed-ended (CE) and multi-choice (MC) formats: (1) CE tasks include VQA-RAD [48] (451 radiology questions), SLAKE [71] (1,061 mixed-modality samples), and PathVQA [29] (6,719 pathology samples); (2) MC tasks comprise PMC-VQA [137] (10,000 samples), the Health & Medicine track of MMMU [132] (150 samples), and the expert-level MedXpertQA-MM [160] (960 samples). Additionally, we evaluate on GMAI-MMBench [127], a multi-granularity benchmark spanning 38 distinct modalities with 2,847 questions. Accuracy is the primary metric across all benchmarks, except for MedXpertQA-MM which uses a Total Score.

Table 4: Per-modality accuracy (%) on OmniMedVQA. ∆: improvement over Qwen3- VL-8B zero-shot baseline.
<table><tr><td>Method</td><td>CT</td><td>MRI</td><td>X-ray</td><td>Derm.</td><td>Fundus</td><td>OCT</td><td>Patho.</td><td>US</td><td>Avg.</td></tr><tr><td>Qwen3-VL-8B (zero-shot)</td><td>52.4</td><td>48.6</td><td>55.1</td><td>61.3</td><td>46.8</td><td>44.2</td><td>50.7</td><td>47.5</td><td>50.8</td></tr><tr><td>MedExpert-R1 [15]</td><td>58.6</td><td>55.8</td><td>60.2</td><td>66.5</td><td>52.4</td><td>49.8</td><td>57.2</td><td>54.1</td><td>56.8</td></tr><tr><td>MedSynapse-V (IMT)</td><td>66.8</td><td>63.5</td><td>68.2</td><td>72.4</td><td>58.6</td><td>56.1</td><td>62.9</td><td>60.7</td><td>63.7</td></tr><tr><td>Δ</td><td>+14.4</td><td>+14.9</td><td>+13.1</td><td>+11.1</td><td>+11.8</td><td>+11.9</td><td>+12.2</td><td>+13.2</td><td>+12.9</td></tr></table>

The training pipeline follows three progressive phases to enhance the medical grounding of the model. Stage I involves large scale pre training with 50K image text pairs from PubMedVision [7]. These samples were rigorously curated by medical experts to ensure accurate alignment across radiology modalities like CT, MRI, and X ray along with pathology. Stage II constructs a specialized mixed modality reinforcement learning set of 4K samples. These instances were also selected by clinical professionals to prioritize high diagnostic value, including 3K closed ended VQA samples from expert annotated OmniMedVQA [35] and 1K open ended samples from SLAKE and PathVQA. Stage III refines the model by reusing the Stage II data with identical preprocessing to ensure consistent optimization. Importantly, rigorous filtering was applied across all dataset splits to ensure that no evaluation test samples overlap with any training data, maintaining the absolute integrity of our zero shot assessment.

## 8 Additional Analysis

## 8.1 Per-Modality Breakdown on OmniMedVQA

Table 4 summarizes performance across the eight imaging modalities in OmniMedVQA. MedSynapse-V achieves consistent performance gains across all eight OmniMedVQA modalities, with the largest gains on radiology-centric modalities (CT: +14.4, MRI: +14.9, X-ray: +13.1) where structured anatomical priors are most informative. Notably, although MMedExpert-R1 improves over the Qwen3-VL-8B zero-shot baseline across all modalities through guideline-based RL rewards, the margin remains modest on challenging modalities such as OCT (+5.6) and Fundus (+5.6), where explicit CoT reasoning struggles to capture subtle spatial patterns. In contrast, MedSynapse-V’s latent memory mechanism yields substantially larger gains on these same modalities (+11.9 and +11.8), confirming that continuous diagnostic memory encodes fine-grained anatomical features more efectively than discrete token reasoning.

![](images/318872df63a764a9ef5f0d344cf81502fb37745d207bb4e7b4836d9d558ea12b.jpg)  
Fig. 11: Causal intervention visualization on fundus (left group) and dermoscopy (right group). Each group: original image, MedSAM3 region mask B, and post-CCR memory attention map. After refinement, memory attention concentrates on diagnostically critical structures while suppressing background.

## 8.2 Visualization of Causal Counterfactual Intervention

Figure 11 illustrates how CCR reshapes the spatial distribution of memory attention through counterfactual intervention. In the fundus case, MedSAM3 identifies retinal lesions including microaneurysms and hard exudates; after CCR, the memory attention map aligns tightly with these foci while the optic disc and healthy vasculature receive minimal activation. In the dermoscopy case, post-CCR attention concentrates at the lesion periphery where asymmetry, border irregularity, and color heterogeneity are most pronounced, consistent with the ABCD criteria used in clinical dermoscopic assessment. The corresponding intervention map again demonstrates substantial attention redistribution upon masking, with activation scattering to non-diagnostic background regions. These visualizations provide direct evidence that $r _ { c a u s a l }$ successfully enforces a causal dependency between diagnostic memory and pathologically relevant image regions, complementing the quantitative mask robustness analysis in §8.6.

## 8.3 Inference Latency and Parameter Count

The prefill latency of MedSynapse-V (IMT) is 102 ms, identical to the vanilla baseline; $\mathcal { A } _ { \psi }$ contributes only 4 ms. Although prefill overhead is negligible, the 16 memory vectors injected during this stage play a pivotal role in overall eficiency: they become part of the KV cache built at prefill time, so every subsequent decoding step can attend to condensed diagnostic priors at no extra cost. This latent conditioning steers the model toward shorter, more decisive outputs (∼34–44 answer tokens vs. ∼50–80 for the zero-shot baseline), reducing endto-end sample latency from ∼2.8 s to

Table 5: Inference latency (single A100, batch = 1) and parameter count. Prefill: time to first token. End-to-end : full autoregressive decoding (max 128 tokens, greedy).
<table><tr><td>Component</td><td colspan="2">|MedSynapse-V (IMT)|Qwen3-VL-8B</td></tr><tr><td>Visual Encoding</td><td>38 ms</td><td>38ms</td></tr><tr><td>Memory Generation (Aψ)</td><td>4ms</td><td></td></tr><tr><td>First Decoding Step</td><td>60ms</td><td>64ms</td></tr><tr><td>Prefill (to first token) End-to-end / sample</td><td>102 ms ~2.6 s</td><td>102 ms ~2.8 s</td></tr><tr><td></td><td></td><td></td></tr><tr><td colspan="3"></td></tr><tr><td>Module</td><td></td><td>[Parameters|Trainable Stage</td></tr><tr><td>Qwen3-VL-8B backbone</td><td>8.29B</td><td>Frozen</td></tr><tr><td>LoRA adapters</td><td>83.9M</td><td>Stage II</td></tr><tr><td>Pφ (Memory Sampler)</td><td>12.6M</td><td>Stage I</td></tr><tr><td>Aψ (Autonomous Module)</td><td>33.6M</td><td>Stage III</td></tr><tr><td>MedSAM3 encoder</td><td>93.7M</td><td>Frozen (removed)</td></tr><tr><td>Inference total</td><td>8.41B</td><td></td></tr></table>

∼2.6 s. CoT baselines require 5.2–5.8 s due to 300–400 autoregressive reasoning tokens. The inference footprint is 8.41B parameters (1.4% over the bare backbone), as the ${ \mathcal { E } } _ { a n a }$ is entirely removed.

![](images/b04b4dc545d52a94e4b115e1afa9d1694c642563ef56361584a49faa66d84434.jpg)  
Fig. 12: Memory evolution across training stages. Visualization of diagnostic memory vectors colored by modality. Contours denote kernel density estimates per modality to highlight the clustering density across diferent clinical domains.

Note on the “ms/token” column in Table 2 of the main text. The ms/token values in the main paper’s ablation table measure average per-token decoding latency (total decode wall time / number of generated tokens). The zeroshot Qwen3-VL-8B baseline shows a higher value (126 ms/token) than MedSynapse-V (∼102 ms/token) because, without compact memory conditioning in the KV cache, the model’s attention must scatter across the full visual token sequence at each decode step, yielding broader attention patterns and slower per-step computation. These per-token values should not be confused with the prefill latency (102 ms, Table 5) or the end-to-end sample latency (∼2.6 s, Fig. 5), which cover the complete inference pipeline.

## 8.4 Memory Evolution Across Training Stages

Figure 12 illustrates the progressive structuralization of the diagnostic memory space. Initially, the Before MQPM (panel a) snapshot reveals a chaotic distribution where raw VLM features lack modality-discriminative organization, intermixing all eight imaging types. Stage I (MQPM) (panel b) introduces a shared representational basis through semantic alignment; however, the persistent overlap between CT/MRI and the poor delineation of OCT suggest that warmup alone cannot achieve fine-grained discrimination. This bottleneck is resolved in Stage II (CCR) (panel c), where the causal counterfactual reward $r _ { c a u s a l }$ reshapes the manifold into compact, well-separated clusters with clinically coherent proximity—grouping radiological modalities and surface imaging into distinct neighborhoods. Finally, Stage III (IMT) (panel d) confirms that the autonomous memory $\mathcal { M } _ { a u t o }$ faithfully internalizes this refined structure; the near-identical topology to panel (c) corroborates the near-lossless distillation $( \varDelta = 1 . 4 \mathrm { p p } )$ after removing the anatomical encoder. These visualizations provide a structural rationale for the performance gains observed in Table 2, specifically the 52.9% collapse without MQPM and the 4.1 pp improvement driven by $r _ { c a u s a l } .$

Table 6: Memory synthesis design ablations. $L e f t { \mathrm { : } }$ meta-query sampling vs. simpler injection (full MQPM→CCR→IMT pipeline, encoder-free inference). Right: efect of including question tokens q in $\mathcal { A } _ { \psi }$ input. Default configurations are highlighted.
<table><tr><td>Aggregation Strategy</td><td>VQA- RAD</td><td>SLAKEPathVQA PMC-</td><td></td><td>VQA</td><td>MMMU|*</td><td> $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td>No injection (zero-shot)</td><td>58.6</td><td>66.2</td><td>55.4</td><td>42.5</td><td>48.3</td><td>54.2</td></tr><tr><td>Avg-pool concat</td><td>63.4</td><td>70.8</td><td>58.2</td><td>48.1</td><td>51.6</td><td>58.4</td></tr><tr><td>Linear projector</td><td>65.1</td><td>72.3</td><td>59.8</td><td>50.4</td><td>53.2</td><td>60.2</td></tr><tr><td>Meta-query Pφ (ours)</td><td>74.2</td><td>79.8</td><td>64.8</td><td>58.5</td><td>61.4</td><td>67.7</td></tr></table>

<table><tr><td>Aψ Input</td><td>VQA- RAD</td><td>SLAKE</td><td>Avg.</td></tr><tr><td>Visual only</td><td>72.8</td><td>78.1</td><td>67.0</td></tr><tr><td>Visual + q (default)</td><td>74.2</td><td>79.8</td><td>67.7</td></tr></table>

Table 7: Efect of mask confidence threshold τ. Full MQPM→CCR→IMT pipeline; encoder-free inference. |B|/HW : average masked fraction.
<table><tr><td>Threshold τ</td><td> $| \mathbf { B } | / H W \ ( \% )$ </td><td>VQA- RAD</td><td>SLAKE</td><td>PathVQA</td><td>PMC- VQA</td><td>MMMU*</td><td> $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td>0.3</td><td>42.6</td><td>71.4</td><td>77.0</td><td>62.5</td><td>55.8</td><td>58.2</td><td>65.0</td></tr><tr><td>0.5</td><td>28.3</td><td>73.1</td><td>78.6</td><td>63.8</td><td>57.4</td><td>60.1</td><td>66.6</td></tr><tr><td>0.7 (default)</td><td>15.8</td><td>74.2</td><td>79.8</td><td>64.8</td><td>58.5</td><td>61.4</td><td>67.7</td></tr><tr><td>0.8</td><td>10.1</td><td>73.8</td><td>79.2</td><td>64.2</td><td>58.0</td><td>60.8</td><td>67.2</td></tr><tr><td>0.9</td><td>5.4</td><td>72.5</td><td>77.8</td><td>63.0</td><td>56.5</td><td>59.0</td><td>65.8</td></tr><tr><td>No mask  $( r _ { a c c } ~ \mathrm { o n l y } )$ </td><td></td><td>70.1</td><td>75.8</td><td>61.2</td><td>54.3</td><td>56.6</td><td>63.6</td></tr></table>

## 8.5 Memory Synthesis Design Choices

We investigate two design dimensions of the memory synthesis module: (i) the aggregation strategy for converting anatomical encoder features into compact memory, and (ii) whether question tokens should condition the autonomous module $\mathcal { A } _ { \psi }$

Aggregation strategy (Table 6, left). Average-pooling MedSAM features into 16 tokens and concatenating them to the input provides +4.2 pp over zero-shot, confirming that the anatomical encoder supplies useful priors. A learnable linear projector $( { \mathrm { M e d S A M } } \to d _ { h } )$ further improves to 60.2%. The meta-query sampler $\mathcal { P } _ { \phi }$ outperforms both by a substantial margin (+7.5 pp over linear projector, +13.5 pp over zero-shot), demonstrating that selective, input-conditioned aggregation of spatial features via cross-attention is critical: static compression discards fine-grained spatial cues that the learnable probes can selectively retain. Query conditioning in $\mathcal { A } _ { \psi }$ (Table 6, right). Including question tokens as input to $\mathcal { A } _ { \psi }$ yields a consistent +0.7 pp improvement, as query context enables $\mathcal { A } _ { \psi }$ to generate task-relevant memory rather than generic anatomical summaries. The gain is modest because the VLM’s self-attention already conditions answer generation on ${ \bf q } ;$ the additional query signal in $\mathcal { A } _ { \psi }$ primarily helps disambiguate cases where multiple diagnostic hypotheses compete for the same visual features.

## 8.6 Robustness of Causal Intervention to Mask Quality

The causal counterfactual reward $r _ { c a u s a l }$ relies on region masks B from Med-SAM3. We verify robustness via two ablations: (i) varying the binarization threshold τ , and (ii) replacing the top-1 mask with lower-ranked candidates.

Table 8: Efect of mask rank selection. Rank-1: highest confidence (default); Rank-2: second-highest; Random: uniformly sampled candidate.
<table><tr><td>Mask Selection</td><td> $\mathbf { V Q A - }$  RAD</td><td>SLAKE</td><td>PathVQA</td><td>PMC- VQA</td><td>MMMU*</td><td> $\mathbf { A v } \mathbf { g } .$ </td><td>Δ</td></tr><tr><td>Rank-1 (default)</td><td>74.2</td><td>79.8</td><td>64.8</td><td>58.5</td><td>61.4</td><td>67.7</td><td>一</td></tr><tr><td>Rank-2</td><td>73.0</td><td>78.6</td><td>63.6</td><td>57.2</td><td>60.0</td><td>66.5</td><td>-1.2</td></tr><tr><td>Random</td><td>71.8</td><td>77.2</td><td>62.4</td><td>56.0</td><td>58.4</td><td>65.2</td><td>-2.5</td></tr><tr><td>No mask  $\left( r _ { a c c } \mathrm { \ o n l y } \right) $ </td><td>70.1</td><td>75.8</td><td>61.2</td><td>54.3</td><td>56.6</td><td>63.6</td><td>-4.1</td></tr></table>

(i) Confidence threshold τ. Table 7 reports accuracy under five thresholds (default τ =0.7). All thresholds outperform the no-mask baseline (Avg 63.6%). Performance is stable across $\tau \in [ 0 . 5 , 0 . 8 ]$ (spread only 1.1 pp), confirming that $r _ { c a u s a l }$ does not require pixel-perfect boundaries. Extreme values degrade: τ =0.3 masks 42.6% of the image (intervention too destructive); τ =0.9 masks only 5.4% (intervention too weak). (ii) Mask rank selection. We replace MedSAM3’s top-1 mask with its second-highest confidence candidate (Rank-2) or a random candidate. Rank-2 retains most of the gain (−1.2 pp vs. Rank-1), and even random masks outperform the no-mask baseline by 1.6 pp. The monotonic ordering Rank-1 > Rank-2 > Random > None confirms that mask quality helps but is not critical: the causal reward exploits the relative contrast between masked and unmasked conditions rather than relying on pixel-precise delineation.

## 9 Additional Qualitative Results

## 9.1 Additional Representative Cases

Figure 13 illustrates comparative evaluations between MedSynapse-V and two competitive RL-CoT baselines across diverse modalities. While Med-R1 and MMedExpert-R1 generate extensive reasoning chains, they frequently yield erroneous diagnoses as a result of hallucinatory observations that propagate and amplify throughout the inference process. In the chest X-ray case, Med-R1 fabricates bilateral interstitial opacities and claims sharp costophrenic angles, missing the obvious pleural efusion; MMedExpert-R1 hallucinates a convex border with cavitation and misdiagnoses a lung abscess. In the pathology case, Med-R1 incorrectly describes preserved polarity and intact basement membranes to conclude fibroadenoma, while MMedExpert-R1 fabricates lymphovascular invasion and comedonecrosis to misclassify as invasive lobular carcinoma. In the head CT case, Med-R1 denies the presence of a hyperdense lesion and diagnoses ischemic infarct, whereas MMedExpert-R1 hallucinates ring enhancement with central necrosis and concludes cerebral abscess. In contrast, MedSynapse-V directly identifies the correct findings in 38–43 tokens without explicit CoT, demonstrating that latent diagnostic memory provides suficient guidance while avoiding hallucination cascades.

![](images/0e8a389d2913561a42f3944b289843e794f152ff146dbf1ce52a9be9d7018dea.jpg)  
Fig. 13: Qualitative comparison across Chest X-ray, Pathology, and Head CT cases. MedSynapse-V produces concise, correct diagnoses (∼38–43 tokens), while other methods generate verbose CoT (∼195–215 tokens) with hallucinated findings (red).

## 9.2 Failure Case Analysis

Figure 14 illustrates three primary failure modes. (a) Rare modality under-representation: OCT, constituting the smallest training proportion (∼25%), exhibits the lowest per-modality accuracy, indicating that memory quality degrades when prior exposure is insuficient. (b) Multi-lesion ambiguity: accuracy drops from 78% on single-lesion images to 52% on multi-lesion cases, as the

![](images/88a047b2ff210ae68886dd5451693d7e7bf90879750987cd26d35201c7f33d51.jpg)  
Fig. 14: Three representative challenging modes.

fixed N=16 memory pool becomes saturated when multiple co-occurring pathologies compete for representational capacity. (c) Subtle feature discrimination: each scatter point represents one evaluation sample from the dermoscopy subset, where the x-axis is the model’s confidence defined as the mean token-level generation probability co $\begin{array} { r } { \mathrm { { n f } } ( \mathbf { o } ) = \frac { 1 } { \left| \mathbf { o } \right| } \sum _ { t } \pi _ { \theta } ( \mathbf { o } _ { t } \mid X , q , \mathcal { M } , \mathbf { o } _ { < t } ) } \end{array}$ , and the y-axis is binary diagnostic correctness (1=correct, 0=incorrect; vertical jitter applied for visibility). While high-confidence predictions are predominantly correct, a notable cluster at conf < 0.3 with correctness= 0 reveals that borderline cases (e.g., benign vs. dysplastic nevi) fall below the memory’s discriminative granularity. These modes point to future directions including balanced modality sampling, adaptive memory pool sizing, and calibrated uncertainty estimation.

System: You are a helpful medical assistant. Answer the question based on   
the image.   
User: <image>   
{question}   
Options:   
(A) {option\_a}   
(B) {option\_b}   
(C) {option\_c}   
(D) {option\_d}   
Please answer with the option letter only.   
Assistant:

Fig. 15: Prompt template for closed-ended multi-choice VQA (VQA-RAD, SLAKE, PathVQA, PMC-VQA, MMMU\*, MedXpertQA-MM, GMAI-MMBench). The number of options varies by dataset (2–5); the template adapts accordingly.  
System: You are a helpful medical assistant. Provide a concise answer to   
the question.   
User: <image>   
{question}   
Answer the question using a single word or phrase.   
Assistant:  
Fig. 16: Prompt template. Notably, $\mathcal { M } _ { a u t o }$ is autonomously generated and injected in the hidden stream without altering the text prompt.

## 10 Evaluation Prompt Templates

We adopt minimal, zero-shot prompt templates for all evaluations to avoid biasing the model through elaborate instructions and to ensure fair comparison across methods. Following prior medical VLM evaluation practices [7,47,80,84], we use a brief system instruction paired with the clinical query and image, without few-shot exemplars or chain-of-thought elicitation. This design isolates the efect of each model’s intrinsic capabilities (or, in our case, latent diagnostic memory) from prompt engineering.

The qualitative case analyses in the main paper and this supplement uniformly use the open-ended template to reveal each model’s complete diagnostic reasoning. Fig. 15 and 16 present the exact prompt templates used for closedended and open-ended evaluation, respectively. For MedSynapse-V, the Autonomous Memory Module $\mathcal { A } _ { \psi }$ generates diagnostic implicit memory $\mathcal { M } _ { a u t o } =$ $\{ m _ { 1 } , \hdots , m _ { 1 6 } \}$ directly from the VLM’s own visual encoding features and injects them into the hidden stream between the question encoding and the answer generation position (see §2.4 in the main text). The entire process is transparent to the surface-level prompt: no additional text tokens, special markers, or reasoning elicitation instructions are required, distinguishing MedSynapse-V from both explicit CoT methods (which append reasoning instructions such as “Let’s think step by step”) and other latent reasoning methods that require special delimiters (e.g., Coconut’s <bot>/<eot> markers [28] or Heima’s <CoT> tokens [97]).

Table 9: $L e f t { \mathrm { : } }$ efect of memory injection position on diagnostic accuracy (%), where V and q denote visual and question tokens. Right: efect of GRPO group size G on accuracy (%) and training cost. All results use the full three-stage pipeline with IMT inference. Default configurations are highlighted.
<table><tr><td>Injection Position</td><td>VQA- RAD</td><td></td><td>SLAKE PathVQA</td><td>PMC- VQA</td><td> $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td>Before V</td><td>70.2</td><td>75.6</td><td>61.3</td><td>54.8</td><td>65.5</td></tr><tr><td> $\mathbf { V }  { \mathcal { M } }  \mathbf { q }$ </td><td>72.5</td><td>77.8</td><td>63.1</td><td>56.9</td><td>67.6</td></tr><tr><td>After q (default)</td><td>74.2</td><td>79.8</td><td>64.8</td><td>58.5</td><td>69.3</td></tr><tr><td>Interleaved w/ q</td><td>73.1</td><td>78.4</td><td>63.8</td><td>57.6</td><td>68.2</td></tr></table>

<table><tr><td>G</td><td>VQA- RAD</td><td></td><td>SLAKE PathVQA</td><td>PMC- VQA</td><td> $\mathbf { A v } \mathbf { g } .$ </td><td>GPU-h</td></tr><tr><td>2</td><td>72.4</td><td>77.6</td><td>62.9</td><td>56.2</td><td>67.3</td><td>14</td></tr><tr><td>4</td><td>74.2</td><td>79.8</td><td>64.8</td><td>58.5</td><td>69.3</td><td>28</td></tr><tr><td>6</td><td>74.3</td><td>80.0</td><td>64.9</td><td>58.6</td><td>69.5</td><td>40</td></tr><tr><td>8</td><td>74.5</td><td>80.1</td><td>65.0</td><td>58.7</td><td>69.6</td><td>55</td></tr></table>

Answer extraction. For closed-ended tasks, we extract the first valid option letter $\mathrm { ( A / B / C / D / E ) }$ from the generated output using regex matching. For CoT baselines that produce structured tags $( e . g . ,$ , <answer>B</answer>), we parse the content within the answer tags. If no valid option is detected, the response is marked as incorrect. For open-ended tasks, we follow prior work [29, 71] and perform exact string matching after lowercasing and stripping punctuation.

Decoding configuration. All models are evaluated with greedy decoding (temperature = 0, top-p = 1.0) to ensure deterministic and reproducible outputs. The maximum generation length is set to 128 tokens for MedSynapse-V and other direct-answer models, and 1024 tokens for CoT baselines to accommodate their verbose reasoning traces. Note that the 16 diagnostic memory vectors (N=16) are injected into the hidden stream as continuous embeddings and do not count toward the generated token budget; the model’s actual text output for closedended tasks is typically 1–3 tokens and open-ended for 20-40 tokens.

## 11 Extended Ablation Studies

Table 9 and Table 10 present four complementary design analyses that further validate the key design choices of MedSynapse-V. All results use the full threestage pipeline with IMT inference; default configurations are highlighted.

(i) Memory injection position. Table 9 (left) examines the efect of injecting diagnostic memory M at diferent positions in the input sequence, where V denotes visual tokens and q denotes question tokens. Placing M after q and before answer generation (our default) yields the best average of 69.3%, as answer tokens can attend to both visual features and diagnostic memory simultaneously. Injection before V degrades performance to 65.5% because self-attention cannot condition memory on the question context; interleaving with q (68.2%) partially recovers but still disrupts the natural query encoding flow.

Table 10: Left: comparison of divergence measures for IMT distillation (β controls JSD interpolation weight). Right: sensitivity analysis of causal reward weight λ<sub>causal</sub>. Default configurations are highlighted.
<table><tr><td>Divergence</td><td>VQA- RAD</td><td colspan="3">SLAKEPathVQA PMC- VQA</td></tr><tr><td>Forward KL</td><td>72.1</td><td>77.5 62.8</td><td>56.4</td><td>67.2</td></tr><tr><td>Reverse KL</td><td>71.8</td><td>77.1 62.3</td><td>55.9</td><td>66.8</td></tr><tr><td>JSD (β=0.3)</td><td>73.5</td><td>79.0 64.0</td><td>57.8</td><td>68.6</td></tr><tr><td>JSD (β=0.5)</td><td>74.2</td><td>79.8</td><td>64.8 58.5</td><td>69.3</td></tr><tr><td>JSD (β=0.7)</td><td>73.8</td><td>79.3</td><td>64.3 58.0</td><td>68.9</td></tr></table>

<table><tr><td> $\lambda _ { c a u s a l }$ </td><td>VQA- RAD</td><td colspan="3">SLAKEPathVQA PMC- VQA</td><td> $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td>0.0</td><td>70.1</td><td>75.8</td><td>61.2</td><td>54.3</td><td>65.4</td></tr><tr><td>0.3</td><td>73.4</td><td>79.0</td><td>64.1</td><td>57.8</td><td>68.6</td></tr><tr><td>0.5</td><td>74.2</td><td>79.8</td><td>64.8</td><td>58.5</td><td>69.3</td></tr><tr><td>0.7</td><td>73.8</td><td>79.4</td><td>64.5</td><td>58.1</td><td>69.0</td></tr><tr><td>1.0</td><td>72.6</td><td>78.1</td><td>63.2</td><td>56.8</td><td>67.7</td></tr></table>

(ii) GRPO group size G. Table 9 (right) shows that G=4 achieves the optimal accuracy–cost balance: smaller groups (G=2) yield noisy advantage estimates (67.3%), while $G { = } 6$ and G=8 provide only marginal gains (+0.1– 0.3 pp) at 1.4–2× additional GPU hours. The diminishing returns beyond G=4 confirm that four trajectories sufice for stable advantage estimation under our composite reward.

(iii) IMT divergence function. Table 10 (left) compares divergence measures for the IMT distillation objective. Jensen–Shannon divergence with $\beta { = } 0 . 5$ outperforms both forward KL (67.2%) and reverse KL (66.8%). Forward KL causes mode-covering behavior that dilutes diagnostic specificity; reverse KL leads to mode-seeking collapse. The symmetric JSD provides a balanced learning signal, and performance remains stable across $\beta \in [ 0 . 3 , 0 . 7 ]$

(iv) Causal reward weight λ<sub>causal</sub>. Table 10 (right) reveals that performance is robust within $\lambda _ { c a u s a l } \in [ 0 . 3 , 0 . 7 ]$ , peaking at 0.5. Setting $\lambda _ { c a u s a l } { = } 0$ causes the model to bypass memory via direct shortcuts (65.4%); excessively high values (≥ 1.0) over-penalize trajectories and destabilize training (67.7%).

## 12 Related Works

Latent Computation and Memory Augmented Reasoning. Our method is related to latent computation, which leverages continuous latent states to reshape generation in large language models [13, 23, 130, 134]. Existing works include native latent reasoning [28, 97, 101, 138] and latent regulated generation [54, 72, 124–126]. Memory evolution and adaptive query mechanisms have also been explored, including progressive experiential compression [103], prompt based continual learning [104, 128, 159], spectral coverage analysis for in context learning stability [109], rendering aware visual generation with RL driven self feedback [62, 63], trajectory anchored memory for long horizon agents [98], hierarchical abductive reasoning [90], adaptive state fusion in vision state space models [45, 46], multimodal temporal point process modeling [43], knowledge graph re scoring and query graph alignment for structured reasoning [57, 139], and logic consistent structured knowledge reasoning [58]. Counterfactual explanation methods for deep RL agents [10] and causal dynamics analysis of modality arbitration [142] further motivate our use of interventional reasoning within latent spaces. Our diagnostic implicit memory condenses domain priors into continuous vectors that undergo progressive refinement from external dependency to intrinsic capability.

Reinforcement Learning for Vision Language Models. RL has become a powerful paradigm for aligning VLMs beyond supervised fine tuning [83,91,93], with medical applications [47,84,158] and broader tasks including spatial reasoning [96], cognitive supersensing [52], geometry reasoning [64], expression recognition [65], visual generation [61], optimization modeling [73], chart reasoning via programmatic synthesis [76], satirical image comprehension [44], and UI in the loop multimodal GUI reasoning [56]. Post training pipelines combining SFT and preference optimization [36,38,135,146], reasoning termination control [37], variation aware entropy scheduling for non stationary RL [110], multi agent collaboration [145], agent benchmarks [25, 131], LLM applications in financial analysis [12, 74], and codebook rebalancing for bias mitigation in generative systems [20] further advance alignment. High quality synthetic data for VLM training [66, 75], vision language representation fusion [77], automated phenotype recognition via prompting [102], PEFT methods for medical multimodal models [5], and LLM capabilities for spatial transcriptomics [112] demonstrate the importance of data centric and parameter eficient approaches. Hallucination mitigation in multimodal systems through snowball evaluation [42], medical hallucination detection [6], phase wise self reward [141], and modality preference evaluation [140] address critical reliability concerns. VLM robustness under adversarial attacks [121], backdoor defense [122, 123], and logical self reflection based safeguards [69] ensure deployment reliability. For ofline RL, federated methods [86, 87], conservative estimation [88], collapse suppressed optimization [89], and parameter eficient model merging [17–19] address training stability. Our CCR builds upon GRPO [95] with a causal counterfactual reward distinguishing memory utilization from shortcuts.

Medical Image Understanding and Eficient Deployment. The anatomical encoder in MedSynapse-V derives spatial priors from large scale segmentation pretraining. Annotation eficient medical segmentation [148, 149], semi supervised retinal cell identification [153, 154], hypergraph based pathological detection [55], prototype guided interactive pathology segmentation [22], curated pathology datasets [107], similarity aware medical event prediction [60], federated medical segmentation [147], pathology aware prototype evolution for multicenter diagnosis [156], dynamic visual focus learning for progressive diagnosis [155], adaptive causal reasoning for trustworthy medical VLMs [68], vision to text chain of thought for medical reasoning [111], chain of medical thought for hallucination reduction in report generation [40], multi agent diagnostic collaboration for hallucination resistant medical VQA [41], holistic medical vision language understanding [39], knowledge enhanced diagnostic reasoning [27, 34], domain adaptive predictive healthcare [32, 33], adaptive temporal mixture of experts for clinical prediction [144], graph representation learning for heterogeneous medical data [30, 143], bias mitigation in synthetic medical data [92], and clinical workflow analysis [50, 78, 120] collectively advance medical image and health informatics understanding. On the deployment side, eficient inference methods including speculative decoding [116], native parallel reading [108], diffusion model acceleration [114, 115], ultra low bit quantization [151, 152], neural parameter search [16], VLM architecture co design for NPU inference [9], and quantized multimodal split learning [26] validate that not all computation demands equal investment, a principle our 16 memory vectors embody by replacing hundreds of reasoning tokens.

## References

1. Arasteh, S.T., Lotfinia, M., Bressem, K., Siepmann, R., Adams, L., Ferber, D., Kuhl, C., Kather, J.N., Nebelung, S., Truhn, D.: Radiorag: factual large language models for enhanced diagnostics in radiology using online retrieval augmented generation 2024. arXiv preprint arXiv.2407.15621

2. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)

3. Brunyé, T.T., Drew, T., Weaver, D.L., Elmore, J.G.: A review of eye tracking for understanding and improving diagnostic interpretation. Cognitive research: principles and implications 4(1), 7 (2019)

4. Chen, C., Ma, Z., Li, Y., Hu, Y., Wei, Y., Li, W., Nie, L.: Reasoning in the dark: Interleaved vision-text reasoning in latent space. arXiv preprint arXiv:2510.12603 (2025)

5. Chen, J., Jiang, Y., Yang, D., Li, M., Wei, J., Qian, Z., Zhang, L.: Can llms tuning methods work in medical multimodal domain? In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 112–122. Springer (2024)

6. Chen, J., Yang, D., Wu, T., Jiang, Y., Hou, X., Li, M., Wang, S., Xiao, D., Li, K., Zhang, L.: Detecting and evaluating medical hallucinations in large vision language models. arXiv preprint arXiv:2406.10185 (2024)

7. Chen, J., Gui, C., Ouyang, R., Gao, A., Chen, S., Chen, G.H., Wang, X., Cai, Z., Ji, K., Wan, X., et al.: Towards injecting medical visual knowledge into multimodal llms at scale. In: Proceedings of the 2024 conference on empirical methods in natural language processing. pp. 7346–7370 (2024)

8. Chen, K., Rui, S., Jiang, Y., Wu, J., Zheng, Q., Song, C., Wang, X., Zhou, M., Liu, M.: Think twice to see more: Iterative visual reasoning in medical vlms. arXiv preprint arXiv:2510.10052 (2025)

9. Chen, W., Wu, L., Hu, Y., Li, Z., Cheng, Z., Qian, Y., Zhu, L., Hu, Z., Liang, L., Tang, Q., Liu, Z., Yang, H.: Autoneural: Co-designing vision-language models for npu inference (2025), https://arxiv.org/abs/2512.02924

10. Chen, Z., Silvestri, F., Tolomei, G., Wang, J., Zhu, H., Ahn, H.: Explain the explainer: Interpreting model-agnostic counterfactual explanations of a deep reinforcement learning agent. IEEE Transactions on Artificial Intelligence 5(4), 1443– 1457 (2022)

11. Cheng, J., Ye, J., Deng, Z., Chen, J., Li, T., Wang, H., Su, Y., Huang, Z., Chen, J., Jiang, L., et al.: Sam-med2d. arXiv preprint arXiv:2308.16184 (2023)

12. Cheng, K., Qi, X., Cheng, Z., Lai, L., Liu, X.: Regime-dependent volatility dynamics: Evidence from time-series analysis. In: Proceedings of the 2026 3rd International Conference on Applied Economics, Management Science and Social Development (AEMSS 2026). pp. 179–189. Atlantis Press (2026)

13. Deng, Y., Choi, Y., Shieber, S.: From explicit cot to implicit cot: Learning to internalize cot step by step. arXiv preprint arXiv:2405.14838 (2024)

14. Deria, A., Kumar, K., Dukre, A.M., Segal, E., Khan, S., Razzak, I.: Medmo: Grounding and understanding multimodal large language model for medical images. arXiv preprint arXiv:2602.06965 (2026)

15. Ding, M., Zhang, J., Wang, W., Zhong, H., Luo, X., Chen, W., Shen, L.: Mmedexpert-r1: Strengthening multimodal medical reasoning via domain-specific adaptation and clinical guideline reinforcement. arXiv preprint arXiv:2601.10949 (2026)

16. Du, G., Fang, Z., Li, J., Li, J., Jiang, R., Yu, S., Guo, Y., Chen, Y., Goh, S.K., Tang, H.K., He, D., Liu, H., Zhang, M.: Neural parameter search for slimmer finetuned models and better transfer. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (2025). https://doi.org/10.18653/v1/2025.acl-long.1570, https://aclanthology. org/2025.acl-long.1570/

17. Du, G., Lee, J., Li, J., Jiang, R., Guo, Y., Yu, S., Liu, H., Goh, S.K., Tang, H.K., He, D., Zhang, M.: Parameter competition balancing for model merging. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS) (2024)

18. Du, G., Li, Z., Zhou, X., Li, J., Shi, Z., Lin, W., Tang, H.K., Li, X., Liu, F., Wang, W., Zhang, M., Li, J.: Knowledge fusion of large language models via modular skillpacks. In: Proceedings of the International Conference on Learning Representations (ICLR) (2026)

19. Du, G., Lin, W.: Dynamic model merging made slim. arXiv preprint arXiv:2605.18904 (2026)

20. Fan, Z., Chen, Z., Ma, L., Huang, J., Morishetti, L., Nag, K., Kumar, S., Achan, K.: Crab: Codebook rebalancing for bias mitigation in generative recommendation. arXiv preprint arXiv:2604.05113 (2026)

21. Gai, X., Zhou, C., Liu, J., Feng, Y., Wu, J., Liu, Z.: Medthink: Explaining medical visual question answering via multimodal decision-making rationale. arXiv preprint arXiv:2404.12372 (2024)

22. Ge, J., Zhang, D., Zhan, Y., Liu, J., Gong, T., Wu, J., Crispin, M., Li, C., Gao, Z.: Progis: Prototype-guided interactive segmentation for pathological images. IEEE Transactions on Medical Imaging (2025)

23. Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Singh, S., Bartoldson, B.R., Kailkhura, B., Bhatele, A., Goldstein, T.: Scaling up test-time compute with latent reasoning: A recurrent depth approach. arXiv preprint arXiv:2502.05171 (2025)

24. Gu, T., Yang, K., Liu, D., Cai, W.: Lapa: Latent prompt assist model for medical visual question answering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4971–4980 (2024)

25. Guo, H., Xie, Z., Cao, S., Wang, B., Liu, W., Ye, Z., Li, Z., Liu, Z., Lu, W.: Petbench: Benchmarking the abilities of large language models as e-pets in social network services. In: Proceedings of the 34th ACM International Conference on Information and Knowledge Management. pp. 6402–6407 (2025)

26. Guo, J., Luo, X., Zheng, J., Wang, Y., Chang, K.W., Wang, W., Liu, J.: Quantized-tinyllava: A new multimodal foundation model enables eficient split learning. In: arXiv preprint arXiv:2511.23402 (2025)

27. Han, X., Hu, P., Lu, C., Ding, J.E., Liu, F., Ning, Y.: No black boxes: Interpretable and interactable predictive healthcare with knowledge-enhanced agentic causal discovery. In: Findings of the Association for Computational Linguistics: EMNLP 2025. pp. 23415–23427 (2025)

28. Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., Tian, Y.: Training large language models to reaso in a continuous latent space. arXiv preprint arXiv:2412.06769 (2024)

29. He, X., Zhang, Y., Mou, L., Xing, E., Xie, P.: Pathvqa: 30000+ questions for medical visual question answering. arXiv preprint arXiv:2003.10286 (2020)

30. He, Y., Zhang, Y., Gurukar, S., Parthasarathy, S.: Webmile: democratizing network representation learning at scale. Proceedings of the VLDB Endowment 15(12) (2022)

31. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. Iclr 1(2), 3 (2022)

32. Hu, P., Han, X., Wang, F., Ning, Y.: Udoncare: Hierarchy pruning for unseen domain discovery in predictive healthcare. arXiv preprint arXiv:2506.06977 (2025)

33. Hu, P., Lu, C., Liu, F., Ning, Y.: Exploring accurate and transparent domain adaptation in predictive healthcare via concept-grounded orthogonal inference. arXiv preprint arXiv:2602.12542 (2026)

34. Hu, P., Lu, C., Wang, F., Ning, Y.: Bridging stepwise lab-informed pretraining and knowledge-guided learning for diagnostic reasoning. IEEE Journal of Biomedical and Health Informatics (2026)

35. Hu, Y., Li, T., Lu, Q., Shao, W., He, J., Qiao, Y., Luo, P.: Omnimedvqa: A new large-scale comprehensive evaluation benchmark for medical lvlm. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22170–22183 (2024)

36. Huang, Z., Ban, Y., Fu, L., Li, X., Dai, Z., Li, J., Wang, D.: Adaptive sample scheduling for direct preference optimization. arXiv preprint arXiv:2506.17252 (2025)

37. Huang, Z., Xia, X., Ren, Y., Zheng, J., Wang, X., Zhang, Z., Xie, H., Liang, S., Chen, Z., Xiao, X., et al.: Does your reasoning model implicitly know when to stop thinking? arXiv preprint arXiv:2602.08354 (2026)

38. Huang, Z., Xia, X., Ren, Y., Zheng, J., Xiao, X., Xie, H., Li, H., Liang, S., Dai, Z., Zhuang, F., Li, J., Ban, Y., Wang, D.: Real-time aligned reward model beyond semantics (2026), https://api.semanticscholar.org/CorpusID:285240754

39. Jiang, S., Wang, Y., Song, S., Hu, T., Zhou, C., Pu, B., Zhang, Y., Yang, Z., Feng, Y., Zhou, J.T., et al.: Hulu-med: A transparent generalist model towards holistic medical vision-language understanding. arXiv preprint arXiv:2510.08668 (2025)

40. Jiang, Y., Chen, J., Yang, D., Li, M., Wang, S., Wu, T., Li, K., Zhang, L.: Comt: Chain-of-medical-thought reduces hallucination in medical report generation. In: ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1–5. IEEE (2025)

41. Jiang, Y., Han, M., Li, M., Hou, X., Zhang, H., Zhu, W., Li, H., He, Y., Wu, G., Yang, D., et al.: Multi-agent diagnostic collaboration and segmentationaware residual decoding for hallucination-resistant medical vqa. In: ICASSP 2026- 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 11122–11126. IEEE (2026)

42. Jiang, Y., Jiang, X., Zhang, L., Wang, Z., Lu, Y., Wang, P., Han, B., Zheng, F., Yang, D.: Mm-snowball: Evaluating and mitigating hallucination snowballing in multimodal multi-turn dialogue. In: ICML (2026)

43. Jiang, Y., Li, J., Liu, Y., Yang, D., Zhou, F., Kong, Q.: Danmakutppbench: A multi-modal benchmark for temporal point process modeling and understanding. Advances in Neural Information Processing Systems 38 (2026)

44. Jiang, Y., Xue, H., Han, M., Li, M., Hou, X., Yang, D., Zhang, L., Zheng, X.: Satiredecoder: Visual cascaded decoupling for enhancing satirical image comprehension. In: AAAI (2026)

45. Ke, H., Morris, J., Liu, Y., Kitai, S., Oguchi, K., Ding, Y., Wang, H.: Deformba: Vision state space model with adaptive state fusion (2026), https://arxiv.org/ abs/2605.21308

46. Ke, H., Morris, J., Oguchi, K., Cao, X., Liu, Y., Wang, H., Ding, Y.: Mambev: Enabling state space models to learn birds-eye-view representations. In: The Thirteenth International Conference on Learning Representations (2025)

47. Lai, Y., Zhong, J., Li, M., Zhao, S., Li, Y., Psounis, K., Yang, X.: Med-r1: Reinforcement learning for generalizable medical reasoning in vision-language models. IEEE Transactions on Medical Imaging (2026)

48. Lau, J.J., Gayen, S., Ben Abacha, A., Demner-Fushman, D.: A dataset of clinically generated visual questions and answers about radiology images. Scientific data 5(1), 1–10 (2018)

49. Le-Duc, K., Nguyen, D.M., Trinh, P.T., Nguyen, T.P., Diep, N.T., Ngo, A., Vu, T., Vuong, T., Nguyen, A.T., Nguyen, M., et al.: S-chain: Structured visual chainof-thought for medicine. arXiv preprint arXiv:2510.22728 (2025)

50. Lew, D., Baratta, L.R., Xia, L., Eiden, E., Sinsky, C.A., Kannampallil, T., Lou, S.S.: Association of ehr-integrated secure messaging use with clinician workload and attention switching. Journal of general internal medicine 40(10), 2240–2247 (2025)

51. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems 33, 9459–9474 (2020)

52. Li, B., Shen, Y., Liu, Y., Xu, Y., Liu, J., Li, X., Li, Z., Zhu, J., Zhong, Y., Lan, F., et al.: Toward cognitive supersensing in multimodal large language model. arXiv preprint arXiv:2602.01541 (2026)

53. Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J., Naumann, T., Poon, H., Gao, J.: Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems 36, 28541–28564 (2023)

54. Li, H., Li, C., Wu, T., Zhu, X., Wang, Y., Yu, Z., Jiang, E.H., Zhu, S.C., Jia, Z., Wu, Y.N., et al.: Seek in the dark: Reasoning via test-time instance-level policy gradient in latent space. arXiv preprint arXiv:2505.13308 (2025)

55. Li, J., Dong, D., Zheng, M., Zhang, J., Hang, Y., Zhang, L., Zhao, L.: Highprecision mixed feature fusion network using hypergraph computation for cervical abnormal cell detection. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 250–259. Springer (2025)

56. Li, S., Guo, X., Liu, T., Yi, B., Gong, Z., Liu, Z., Chen, H., Zhang, W.: What’s missing in screen-to-action? towards a ui-in-the-loop paradigm for multimodal gui reasoning (2026), https://arxiv.org/abs/2604.06995

57. Li, S., Liu, Z., Gui, Z., Chen, H., Zhang, W.: Enrich-on-graph: Query-graph alignment for complex reasoning with llm enriching. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 7683–7703. Association for Computational Linguistics (2025). https://doi.org/10.18653/ v1/2025.emnlp- main.390, http://dx.doi.org/10.18653/v1/2025.emnlpmain.390

58. Li, S., Liu, Z., Zhu, Y., Chen, H., Zhang, W.: Last layer logits to logic: Empowering llms with logic-consistent structured knowledge reasoning (2025), https: //arxiv.org/abs/2511.07910

59. Li, T., Su, Y., Li, W., Fu, B., Chen, Z., Huang, Z., Wang, G., Ma, C., Chen, Y., Hu, M., et al.: Gmai-vl & gmai-vl-5.5 m: A large vision-language model and a comprehensive multimodal dataset towards general medical ai. arXiv preprint arXiv:2411.14522 (2024)

60. Li, Y., Ge, J., Cao, S., Zhan, Y., Gong, Z., Niu, J., Liu, J., Zhang, D., Li, C.: Similarity-aware dual-perspective learning for medical event prediction. In: 2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). pp. 6263–6270. IEEE (2025)

61. Liang, G., Hu, J., Xing, X., Zhang, J., Yu, Q.: Multi-object sketch animation with grouping and motion trajectory priors. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 9237–9246 (2025)

62. Liang, G., Wang, Z., Hu, J., Zhou, H., Xue, Z., Zhang, J., Xu, D., Yu, Q.: Renderin-the-loop: Vector graphics generation via visual self-feedback. arXiv preprint arXiv:2604.20730 (2026)

63. Liang, G., Wang, Z., Wang, C., Hu, J., Zhou, H., Liu, J., Zhang, J., Xu, D., Yu, Q.: Vanim: Rendering-aware sparse state modeling for structure-preserving vector animation. arXiv preprint arXiv:2605.01517 (2026)

64. Lin, H., Bai, T., Chen, C., Zhang, J., Zeng, B., Zhang, W., Yuan, B.: Synthesizing multimodal geometry datasets from scratch and enabling visual alignment via plotting code. arXiv preprint arXiv:2602.18745 (2026)

65. Lin, H., Bai, T., Zhang, J., Chang, X., Lu, S., Gu, F., Hu, Z., Zhang, W.: Tag: Thinking with action unit grounding for facial expression recognition. arXiv preprint arXiv:2602.18763 (2026)

66. Lin, H., Liu, Z., Zhu, Y., Qin, C., Lin, J., Shang, X., He, C., Zhang, W., Wu, L.: Mmfinereaso: Closing the multimodal reasoning gap via open data-centric methods. arXiv preprint arXiv:2601.21821 (2026)

67. Lin, J.: Divergence measures based on the shannon entropy. IEEE Transactions on Information theory 37(1), 145–151 (2002)

68. Lin, J., Zhu, C., Kneuertz, P.J., Bai, Y., Xue, Y.: Medcausalx: Adaptive causal reasoning with self-reflection for trustworthy medical vision-language models. arXiv preprint arXiv:2603.23085 (2026)

69. Lin, L., You, J., Li, Y., Lin, L., Wang, Y., Zhang, Z., Zheng, M.: Reflect-guard: Enhancing llm safeguards against adversarial prompts via logical self-reflection (2026), https://arxiv.org/abs/2605.24834

70. Liu, A., Xue, R., Cao, X.R., Shen, Y., Lu, Y., Li, X., Chen, Q., Chen, J.: Medsam3: Delving into segment anything with medical concepts. arXiv preprint arXiv:2511.19046 (2025)

71. Liu, B., Zhan, L.M., Xu, L., Ma, L., Yang, Y., Wu, X.M.: Slake: A semanticallylabeled knowledge-enhanced dataset for medical visual question answering. In: 2021 IEEE 18th international symposium on biomedical imaging (ISBI). pp. 1650– 1654. IEEE (2021)

72. Liu, L., Pfeifer, J., Wu, J., Xie, J., Szlam, A.: Deliberation in latent space via diferentiable cache augmentation. arXiv preprint arXiv:2412.17747 (2024)

73. Liu, W., Wu, H., Kuang, Y., Han, X., Zhong, T., Feng, J., Lu, W.: Automated optimization modeling via a localizable error-driven perspective. arXiv preprint arXiv:2602.11164 (2026)

74. Liu, Y., Cheng, Z., Lai, L.: Improving the completeness and comparability of segment disclosures: A large language model approach. Available at SSRN 6720239 (2026)

75. Liu, Z., Liang, H., Huang, X., Xiong, W., Yu, Q., Sun, L., Chen, C., He, C., Cui, B., Zhang, W.: Synthvlm: High-eficiency and high-quality synthetic data for vision language models. arXiv preprint arXiv:2407.20756 3 (2024)

76. Liu, Z., Lin, H., Qin, C., Wang, X., Gao, X., Li, Y., Cai, M., Zhu, Y., Zhong, Z., Pei, Q., et al.: Chartverse: Scaling chart reasoning via reliable programmatic synthesis from scratch. arXiv preprint arXiv:2601.13606 (2026)

77. Liu, Z., Liu, M., Chen, J., Xu, J., Cui, B., He, C., Zhang, W.: Fusion: Fully integration of vision-language representations for deep cross-modal understanding. arXiv preprint arXiv:2504.09925 (2025)

78. Lou, S.S., Lew, D., Xia, L., Baratta, L., Eiden, E., Kannampallil, T.: Secure messaging use and wrong-patient ordering errors among inpatient clinicians. JAMA Network Open 7(12), e2447797 (2024)

79. Moor, M., Huang, Q., Wu, S., Yasunaga, M., Dalmia, Y., Leskovec, J., Zakka, C., Reis, E.P., Rajpurkar, P.: Med-flamingo: a multimodal medical few-shot learner. In: Machine learning for health (ML4H). pp. 353–367. PMLR (2023)

80. Mullappilly, S.S., Kurpath, M.I., Mohamed, O., Zidan, M., Khan, F., Khan, S., Anwer, R., Cholakkal, H.: Medix-r1: Open ended medical reinforcement learning. arXiv preprint arXiv:2602.23363 (2026)

81. Mullappilly, S.S., Kurpath, M.I., Pieri, S., Alseiari, S.Y., Cholakkal, S., Aldahmani, K., Khan, F., Anwer, R., Khan, S., Baldwin, T., et al.: Bimedix2: Bio-medical expert lmm for diverse medical modalities. arXiv preprint arXiv:2412.07769 (2024)

82. Norman, G.: Dual processing and diagnostic errors. Advances in Health Sciences Education 14(Suppl 1), 37–49 (2009)

83. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. Advances in neural information processing systems 35, 27730–27744 (2022)

84. Pan, J., Liu, C., Wu, J., Liu, F., Zhu, J., Li, H.B., Chen, C., Ouyang, C., Rueckert, D.: Medvlm-r1: Incentivizing medical reasoning capability of vision-language models (vlms) via reinforcement learning. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 337–347. Springer (2025)

85. Pham, T.H., Ngo, C.: Multimodal chain of continuous thought for latent-space reasoning in vision-language models. arXiv preprint arXiv:2508.12587 (2025)

86. Qiao, N., Yue, S.: Forler: Federated ofline reinforcement learning with q-ensemble and actor rectification (2026)

87. Qiao, N., Yue, S., Ren, J., Zhang, Y.: Fova: Ofline federated reinforcement learning with mixed-quality data. IEEE Transactions on Networking 34, 2031–2046 (2026). https://doi.org/10.1109/TON.2025.3637043

88. Qiao, N., Yue, S., Wang, S., Deng, Y., Ren, J.: Less is more: Clustered crosscovariance control for ofline RL. In: The Fourteenth International Conference on Learning Representations (2026), https://openreview.net/forum?id= drOy5wi6Qq

89. Qiao, N., Yue, S., Wang, S., Ren, J.: Adamo: A collapse-suppressed optimizer for ofline rl (2026)

90. Qiu, W., Luo, G., Jian, Z., Gao, J., Wang, M., Wu, Q.: Anchor: Abductive network construction with hierarchical orchestration for reliable probability inference in large language models (2026), https://arxiv.org/abs/2605.10328

91. Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems 36, 53728–53741 (2023)

92. Salarian, S., Zhang, Y., Padhee, S., Parthasarathy, S.: Medequalizer: A framework investigating bias in synthetic medical data and mitigation via augmentation. arXiv preprint arXiv:2511.01054 (2025)

93. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017)

94. Sellergren, A., Kazemzadeh, S., Jaroensri, T., Kiraly, A., Traverse, M., Kohlberger, T., Xu, S., Jamil, F., Hughes, C., Lau, C., et al.: Medgemma technical report. arXiv preprint arXiv:2507.05201 (2025)

95. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024)

96. Shen, Y., Liu, Y., Zhu, J., Cao, X., Zhang, X., He, Y., Ye, W., Rehg, J.M., Lourentzou, I.: Fine-grained preference optimization improves spatial reasoning in vlms. arXiv preprint arXiv:2506.21656 (2025)

97. Shen, Z., Yan, H., Zhang, L., Hu, Z., Du, Y., He, Y.: Codi: Compressing chainof-thought into continuous space via self-distillation. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 677–693 (2025)

98. Shi, Y., Li, J., Zhang, L., Dongfang, Z., Wu, B., Tao, S., Yan, Y., Qin, C., Liu, W., Lin, Z., et al.: Androtmem: From interaction trajectories to anchored memory in long-horizon gui agents. arXiv preprint arXiv:2603.18429 (2026)

99. Su, Y., Li, T., Liu, J., Ma, C., Ning, J., Tang, C., Ju, S., Ye, J., Chen, P., Hu, M., et al.: Gmai-vl-r1: Harnessing reinforcement learning for multimodal medical reasoning. arXiv preprint arXiv:2504.01886 (2025)

100. Sun, H., Jiang, Y., Lou, W., Zhang, Y., Li, W., Wang, L., Liu, M., Liu, L., Wang, X.: Chiron-o1: Igniting multimodal large language models towards generalizable medical reasoning via mentor-intern collaborative search. arXiv preprint arXiv:2506.16962 (2025)

101. Tan, W., Li, J., Ju, J., Luo, Z., Song, R., Luan, J.: Think silently, think fast: Dynamic latent compression of llm reasoning chains. arXiv preprint arXiv:2505.16552 (2025)

102. Tao, Y., Huang, Y., Wang, Y., Luo, X., Liu, J.: Autopcr: Automated phenotype concept recognition by prompting. In: arXiv preprint arXiv:2507.19315 (2025)

103. Tian, A., Lu, Y., Fan, X., Wang, C., Zhou, L., Zhang, Y., Liu, Y.: Rgmem: Renormalization group-based memory evolution for language agent user profile. arXiv preprint arXiv:2510.16392 (2025)

104. Tu, D., Yi, H., Wang, Y., Xu, B., Zhao, J., Shen, F.: Multiple queries with multiple keys: A precise prompt matching paradigm for prompt-based continual learning. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 372–381 (2025)

105. Van Sonsbeek, T., Derakhshani, M.M., Najdenkoska, I., Snoek, C.G., Worring, M.: Open-ended medical visual question answering through prefix tuning of language models. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 726–736. Springer (2023)

106. Waite, S., Scott, J., Gale, B., Fuchs, T., Kolla, S., Reede, D.: Interpretive error in radiology. American Journal of Roentgenology 208(4), 739–749 (2017)

107. Wang, C., Ge, J., Niu, Y., Ding, C., Fan, Y., Chang, H., Yang, Z., Ran, C., Teng, X., Wang, X., et al.: A fully annotated pathology slide dataset for early gastric cancer and precancerous lesions. Scientific Data 12(1), 1326 (2025)

108. Wang, T.: Fbs: Modeling native parallel reading inside a transformer (2026), https://arxiv.org/abs/2601.21708

109. Wang, T., Xia, Z.: Stability of in-context learning: A spectral coverage perspective (2026), https://arxiv.org/abs/2509.20677

110. Wang, T., Xia, Z., Chen, X., Liu, S.: Tracking drift: Variation-aware entropy scheduling for non-stationary reinforcement learning (2026), https://arxiv.org/ abs/2601.19624

111. Wang, Y., Liu, J., Gao, S., Feng, B., Tang, Z., Gai, X., Wu, J., Liu, Z.: V2tcot: From vision to text chain-of-thought for medical reasoning and diagnosis. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 658–668. Springer (2025)

112. Wei, H., Luo, X., Yu, H., Liang, J., Yang, L., Lin, L., Popa, A., Yan, X.: Identifying cellular niches in spatial transcriptomics: An investigation into the capabilities of large language models. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 9275– 9289 (2025)

113. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35, 24824–24837 (2022)

114. Wei, L., Chen, W., Tang, P., Guo, X., Ye, L., Wang, R., Li, M.: Orchestrating dualboundaries: An arithmetic intensity inspired acceleration framework for difusion language models. arXiv preprint arXiv:2511.21759 (2025)

115. Wei, L., Luo, Z., Tang, P., Li, M.: Team: Temporal-spatial consistency guided expert activation for moe difusion language model acceleration. arXiv preprint arXiv:2602.08404 (2026)

116. Wei, L., Zhong, S., Xu, S., Wang, R., Huang, R., Li, M.: Specasr: Accelerating llm-based automatic speech recognition via speculative decoding. In: 2025 62nd ACM/IEEE Design Automation Conference (DAC). pp. 1–7. IEEE (2025)

117. Wu, C., Zhang, X., Zhang, Y., Hui, H., Wang, Y., Xie, W.: Towards generalist foundation model for radiology by leveraging web-scale 2d&3d medical data. Nature Communications 16(1), 7866 (2025)

118. Wu, J., Deng, W., Li, X., Liu, S., Mi, T., Peng, Y., Xu, Z., Liu, Y., Cho, H., Choi, C.I., et al.: Medreaso: Eliciting factual medical reasoning steps in llms via knowledge graphs. arXiv preprint arXiv:2504.00993 (2025)

119. Wu, J., Zhu, J., Qi, Y., Chen, J., Xu, M., Menolascina, F., Grau, V.: Medical graph rag: Towards safe medical large language model via graph retrieval-augmented generation. arXiv preprint arXiv:2408.04187 (2024)

120. Xia, L., Lew, D., Baratta, L., Eiden, E., Lou, S., Kannampallil, T.: Association between conversational multitasking and clinician work behaviors at a large us health care system: Cohort study. Journal of medical Internet research 27, e72768 (2025)

121. Xu, B., Dai, X., Tang, D., Zhang, K.: One surrogate to fool them all: Universal, transferable, and targeted adversarial attacks with clip. In: Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security. pp. 3087–3101 (2025)

122. Xu, B., Yang, F., Dai, X., Tang, D., Zhang, K.: Clip-guided backdoor defense through entropy-based poisoned dataset separation. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 7415–7423 (2025)

123. Xu, B., Yang, F., Dai, X., Tang, D., Zhang, K.: From internal diagnosis to external auditing: A vlm-driven paradigm for online test-time backdoor defense. arXiv preprint arXiv:2601.19448 (2026)

124. Xu, Y., Guo, X., Zeng, Z., Miao, C.: Softcot: Soft chain-of-thought for eficient reasoning with llms. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 23336–23351 (2025)

125. Xu, Y., Guo, X., Zeng, Z., Miao, C.: Softcot++: Test-time scaling with soft chainof-thought reasoning. arXiv preprint arXiv:2505.11484 (2025)

126. Xu, Z., Wang, H., Bespalov, D., Wu, X., Stone, P., Qi, Y.: Lars: Latent reasoning skills for chain-of-thought reasoning. In: Findings of the Association for Computational Linguistics: EMNLP 2024. pp. 3624–3643 (2024)

127. Ye, J., Wang, G., Li, Y., Deng, Z., Li, W., Li, T., Duan, H., Huang, Z., Su, Y., Wang, B., et al.: Gmai-mmbench: A comprehensive multimodal evaluation benchmark towards general medical ai. Advances in Neural Information Processing Systems 37, 94327–94427 (2024)

128. Yi, H.: Few-shot class-incremental learning with class centers and contrastive learning for incremental vehicle recognition. In: 2024 International Joint Conference on Neural Networks (IJCNN). pp. 1–8 (2024)

129. Yu, H., Cheng, T., Cheng, Y., Feng, R.: Finemedlm-o1: Enhancing the medical reasoning ability of llm from supervised fine-tuning to test-time training. arXiv e-prints pp. arXiv–2501 (2025)

130. Yu, X., Xu, C., Zhang, G., Chen, Z., Zhang, Y., He, Y., Jiang, P.T., Zhang, J., Hu, X., Yan, S.: Vismem: Latent vision memory unlocks potential of vision-language models. arXiv preprint arXiv:2511.11007 (2025)

131. Yu, Y., Hu, G., Shen, C., Liu, X., Gu, J., Sun, H., Ma, J., Liu, W., Liu, J., Pu, M., et al.: Cirrusbench: Evaluating llm-based agents beyond correctness in real-world cloud service environments. arXiv preprint arXiv:2603.28569 (2026)

132. Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al.: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9556– 9567 (2024)

133. Zakka, C., Shad, R., Chaurasia, A., Dalal, A.R., Kim, J.L., Moor, M., Fong, R., Phillips, C., Alexander, K., Ashley, E., et al.: Almanac—retrieval-augmented language models for clinical medicine. Nejm ai 1(2), AIoa2300068 (2024)

134. Zhang, G., Fu, M., Yan, S.: Memgen: Weaving generative latent memory for selfevolving agents. arXiv preprint arXiv:2509.24704 (2025)

135. Zhang, L., Li, J., Hei, Y., Tao, S., Dai, S., Yan, Y., Dongfang, Z., Liu, W., Qin, C., Li, H., et al.: Temporal gains, spatial costs: Revisiting video fine-tuning in multimodal large language models. arXiv preprint arXiv:2603.17541 (2026)

136. Zhang, W., Guo, J., Zhang, H., Zhang, P., Chen, J., Zhang, S., Zhang, Z., Yi, Y., Bu, H.: Patho-agenticrag: towards multimodal agentic retrievalaugmented generation for pathology vlms via reinforcement learning. arXiv preprint arXiv:2508.02258 (2025)

137. Zhang, X., Wu, C., Zhao, Z., Lin, W., Zhang, Y., Wang, Y., Xie, W.: Pmc-vqa: Visual instruction tuning for medical visual question answering. arXiv preprint arXiv:2305.10415 (2023)

138. Zhang, Y., Xu, W., Zhao, X., Wang, W., Feng, F., He, X., Chua, T.S.: Reinforced latent reasoning for llm-based recommendation. arXiv preprint arXiv:2505.19092 (2025)

139. Zhang, Y., Chen, K., Bai, X., Kang, Z., Guo, Q., Zhang, M.: Question-guided knowledge graph re-scoring and injection for knowledge graph question answering. In: Findings of the Association for Computational Linguistics: EMNLP 2024. pp. 8972–8985 (2024)

140. Zhang, Y., Ma, J., Hou, Y., Bai, X., Chen, K., Xiang, Y., Yu, J., Zhang, M.: Evaluating and steering modality preferences in multimodal large language model. arXiv preprint arXiv:2505.20977 (2025)

141. Zhang, Y., Sun, C., Chen, K., Bai, X., Xiang, Y., Zhang, M.: Mitigating multimodal hallucination via phase-wise self-reward. arXiv preprint arXiv:2604.17982 (2026)

142. Zhang, Y., Xu, M., Bai, X., Zhang, P., Xiang, Y., Zhang, M., et al.: Instruction anchors: Dissecting the causal dynamics of modality arbitration. arXiv preprint arXiv:2602.03677 (2026)

143. Zhang, Y., He, Y., Gurukar, S., Parthasarathy, S.: Heteromile: a multi-level graph representation learning framework for heterogeneous graphs. In: Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining. pp. 63–72 (2026)

144. Zhang, Y., Padhee, S., Yuhas, P.T., Roberts, C.J., Parthasarathy, S.: Adaptive temporal mixture of experts for predicting stifness metrics from the ocular response analyzer and identifying keratoconus. American Journal of Ophthalmology (2026)

145. Zhang, Z., Huang, Z., Xia, X., Wang, D., Zhuang, F., Ma, S., Ding, N., Yang, Y., Li, J., Ban, Y.: Heterogeneous agent collaborative reinforcement learning. arXiv preprint arXiv:2603.02604 (2026)

146. Zhao, F., Lu, C., Xie, Z., Liu, Z., Qian, H., Huang, J., Shi, F., Meng, Z., Guo, H., He, M., et al.: Redone: Revealing domain-specific llm post-training in social networking services. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track. pp. 2648–2674 (2025)

147. Zhao, X., Huang, W., Wang, X., Zhao, H., Zhuang, L., Jiang, A., Wan, G., Ye, M.: Divide, conquer and unite: Hierarchical style-recalibrated prototype alignment for federated medical segmentation. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 28760–28768 (2026)

148. Zhao, X., Li, P., Luo, X., Yang, M., Chang, S., Li, Z.: Sam-driven weakly supervised nodule segmentation with uncertainty-aware cross teaching. In: 2024 IEEE International Symposium on Biomedical Imaging (ISBI). pp. 1–5. IEEE (2024)

149. Zhao, X., Li, Z., Luo, X., Li, P., Huang, P., Zhu, J., Liu, Y., Zhu, J., Yang, M., Chang, S., et al.: Ultrasound nodule segmentation using asymmetric learning with

simple clinical annotation. IEEE Transactions on Circuits and Systems for Video Technology 34(10), 9010–9023 (2024)

150. Zhao, X., Liu, S., Yang, S.Y., Miao, C.: Medrag: Enhancing retrieval-augmented generation with knowledge graph-elicited reasoning for healthcare copilot. In: Proceedings of the ACM on Web Conference 2025. pp. 4442–4457 (2025)

151. Zhao, Z., Liu, F., Wang, J., Guan, C., Wang, Z., Jiang, L., Guan, H.: Specquant: Spectral decomposition and adaptive truncation for ultra-low-bit llms quantization. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 28786–28794 (2026)

152. Zhao, Z., Xu, Z., Yang, D.: Bwla: Breaking the barrier of w1ax post-training quantization for llms (2026), https://arxiv.org/abs/2605.00422

153. Zhou, M., Zhang, Y., Karimi Monsefi, A., Choi, S.S., Doble, N., Parthasarathy, S., Ramnath, R.: Reducing manual labeling requirements and improved retinal ganglion cell identification in 3d ao-oct volumes using semi-supervised learning. Biomedical Optics Express 15(8), 4540–4556 (2024)

154. Zhou, M., Zhang, Y., Kirkendall, E., Karimi Monsefi, A., Wolfe, M., Chitkara, K.A., Choi, S.S., Doble, N., Parthasarathy, S., Ramnath, R.: Isosnet: a unified framework for cone photoreceptor detection and inner segment and outer segment length measurement from ao-oct b-scans. Biomedical Optics Express 16(8), 3237– 3254 (2025)

155. Zhu, C., Lin, Y., Chen, S., Wang, Y., Lin, J.: Medeyes: Learning dynamic visual focus for medical progressive diagnosis. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 13916–13924 (2026)

156. Zhu, C., Lin, Y., Shao, J., Lin, J., Wang, Y.: Pathology-aware prototype evolution via llm-driven semantic disambiguation for multicenter diabetic retinopathy diagnosis. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 9196–9205 (2025)

157. Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., et al.: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479 (2025)

158. Zhu, K., Xia, P., Li, Y., Zhu, H., Wang, S., Yao, H.: Mmedpo: Aligning medical vision-language models with clinical-aware multimodal preference optimization. arXiv preprint arXiv:2412.06141 (2024)

159. Zhu, L., Lan, Q., Tian, Q., Sun, W., Yang, L., Xia, L., Xie, Y., Xiao, X., Duan, T., Tao, C., et al.: Ett-ckge: Eficient task-driven tokens for continual knowledge graph embedding. In: Joint European Conference on Machine Learning and Knowledge Discovery in Databases. pp. 481–496. Springer (2025)

160. Zuo, Y., Qu, S., Li, Y., Chen, Z., Zhu, X., Hua, E., Zhang, K., Ding, N., Zhou, B.: Medxpertqa: Benchmarking expert-level medical reasoning and understanding. arXiv preprint arXiv:2501.18362 (2025)