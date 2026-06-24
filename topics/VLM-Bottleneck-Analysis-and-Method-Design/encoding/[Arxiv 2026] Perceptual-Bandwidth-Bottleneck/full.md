# The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoing via Sequential Experimental Design

Anjie Liu <sup>\*</sup> <sup>1</sup> Ziqin Gong <sup>\*</sup> <sup>1</sup> Yan Song <sup>2</sup> Yuxiang Chen <sup>2</sup> Xiaolong Liu <sup>3</sup> <sup>4</sup> Hengtong Lu <sup>5</sup> Kaike Zhang <sup>5</sup> Chen Wei <sup>5</sup> Jun Wang <sup>2</sup>

## Abstract

Visual perception in modern Vision-Language Models (VLMs) is constrained by a perceptual bandwidth bottleneck: a broad field of view preserves global context but sacrifices the finegrained details required for complex reasoing. We argue that high-resolution visual reasoing is therefore not only semantic reasoing but also task-relevant evidence acquisition under limited perceptual bandwidth. Inspired by active vision and information foraging, we formalise this process as sequential Bayesian optimal experimental design (S-BOED), where an agent decides which visual evidence to acquire before answering. Since exact Bayesian inference is intractable in continuous gigapixel spaces, we derive a tractable coverage–resolution objective as a proxy for task-relevant information gain. We instantiate this framework with FOVEA, a trainingfree procedure that refines VLM crop proposals through evidence-oriented probing. Experiments on high-resolution benchmarks show consistent gains over direct and ReAct-style baselines, with particularly strong improvements in searchdominated remote-sensing settings.

## 1. Introduction

Vision-Language Models (VLMs) have significantly advanced general visual understanding, demonstrating a remarkable ability to reaso about holistic scene context (Bai et al., 2025; Comanici et al., 2025). However, a critical performance gap remains: despite their high-level reasoing capabilities, these models often exhibit “perceptual blindness” in tasks requiring fine-grained resolution (Campbell et al., 2024; Li et al., 2025b). Current state-of-the-art models frequently struggle with small-scale object counting, optical character recognition (OCR), and precise spatial localisation, failing even when the underlying logic of the task is straightforward (Zhang et al., 2024; Tong et al., 2024). We argue that such failures are not only failures of semantic reasoing but also failures of evidence acquisition under limited perceptual bandwidth.

The Perceptual Bandwidth Bottleneck. We identify this limitation as a perceptual bandwidth bottleneck. Most standard vision encoders, such as ViT-based models, project an input image into a fixed number of visual tokens regardless of its original resolution (Dosovitskiy, 2020; Liu et al., 2023). This fixed budget induces an unavoidable field-ofview–resolution trade-off: a global view preserves broad spatial context but compresses fine-grained details, while a local crop preserves details but sacrifices coverage. When processing a high-resolution scene globally, each token must aggregate a large spatial area, causing small objects, text, and local spatial relations to vanish before reasoing begins. Consequently, the model cannot reaso about evidence that is absent from its visual representation.

The Need for an Active Strategy. Alleviating this bottleneck requires the model to act, not merely to perceive. Instead of passively encoding a single downsampled image, the agent must engage in information foraging (Pirolli & Card, 1999): it must decide where to allocate highresolution visual bandwidth in order to acquire task-relevant evidence. Passive scanning strategies, such as sliding windows, are computationally prohibitive and introduce large amounts of distractor evidence. Recent latent Chain-of-Thought (Li et al., 2025a; Sun et al., 2025) and tool-based methods (Ma et al., 2025; Zhang et al., 2025b; Su et al., 2025a; Gao et al., 2025) show that visual agents can benefit from iterative perception, but their crop or tool-selection policies often remain heuristic. They lack a decisiontheoretic objective for deciding which observation is most valuable when the target is not immediately visible.

![](images/6cf1d798eec2fff88dd1768ad8da8b83d492a5a7dcc2ecbc1b9c7ec248ff175e.jpg)  
Figure 1. S-BOED-guided active visual reasoing. Under the perceptual bandwidth bottleneck, FOVEA iteratively refines VLM crop proposals to acquire task-relevant evidence. Candidate crops are scored by a coverage–resolution utility estimated through resolvability probing, and selected views update the interaction history for subsequent search.

Our Approach: Active Visual Reasoing as S-BOED. We formalise active visual information acquisition as a sequential Bayesian optimal experimental design (S-BOED) problem (Lindley, 1956; Chaloner & Verdinelli, 1995; Rainforth et al., 2024). Analogous to a scientist choosing experiments to reduce uncertainty about hidden hypotheses, a VLM agent selects foveation actions to reduce uncertainty about the user’s query, as illustrated in Figure 1.

This formulation exposes a key challenge overlooked in prior work: active visual reasoing is not just discrete visual tool selection, but continuous visual foraging under a bandwidth constraint. While BOED has recently been applied to discrete information-gathering tasks such as question selection (Kobalczyk et al., 2025; Choudhury et al., 2025), highresolution visual reasoing requires selecting continuous foveation actions over large image spaces. The perceptual bandwidth bottleneck creates an Information Cliff : a wide view offers context but too little resolution, while a random zoom offers resolution but may miss the target. As a result, individual observations can have near-zero value until a critical coverage–resolution threshold is reached, motivating non-myopic planning.

Since exact Bayesian inference and exact expected information gain are intractable in continuous gigapixel spaces, we derive a tractable Coverage–Resolution Objective as a proxy for task-relevant information gain. We then instantiate the framework with FOVEA, a training-free inference-time procedure for Foveated Observation and Visual Evidence Acquisition. FOVEA treats the VLM’s initial crop proposal as a noisy spatial prior, generates candidate foveations, probes their query-relevant resolvability, and selects the design that maximises the coverage–resolution objective. Different optimisation strategies, including greedy sampling, MCMCstyle refinement, and look-ahead planning, can be plugged into the same S-BOED-guided template.

The main contributions are: (1) Problem formulation. We identify the perceptual bandwidth bottleneck as a central obstacle in high-resolution VLM reasoing and formulate active visual reasoing as an S-BOED problem. (2) Objective and instantiation. We derive a tractable Coverage– Resolution Objective as a proxy for task-relevant information gain, and instantiate it with FOVEA, a training-free crop-refinement procedure. (3) Empirical validation. We show consistent gains over direct and ReAct-style baselines on high-resolution benchmarks, with further analysis of remote-sensing search, oracle gaps, proposal-limited failures, and compute–accuracy trade-offs.

## 2. Problem Formulation: Active Vision as Experimental Design

We ground our approach in the rigorous framework of Bayesian optimal experimental design (BOED). We consider a VLM agent performing active visual reasoing over a high-resolution image I and a query Q. A comprehensive summary of notations is provided in Appendix B.

To bridge the gap between continuous visual signals and discrete token-based reasoing, we structure this formulation into three layers. First, we model the physical constraints of the VLM sensor, introducing the concept of perceptual bandwidth (Sec. 2.2). Second, we define the generative process, detailing how latent semantic states produce observable tokens through a resolution-gated mechanism (Sec. 2.3). Finally, we unify these components into a probabilistic graphical model (Fig. 2) that governs the agent’s belief updates.

![](images/f941b42b595ab62b58de57c02c05c7c55ea9960b675fe834a2ad99978a3b965e.jpg)  
Figure 2. Influence diagram of active visual reasoing. The foveation design d and latent target location ℓ jointly determine the visibility event S. This latent gate S modulates whether the observation z conveys information about the semantic target y. The agent’s objective is to maximise the utility U, defined as the expected information gain over y, by actively managing the sensing design d.

## 2.1. The Probabilistic System

Formally, we define the system as a tuple $\langle \pmb \theta , \mathcal { D } , \mathcal { Z } \rangle$ . Here, $\theta \in \Theta$ represents the latent parameters (unknown world state), d $\in \mathcal { D }$ denotes the design (action), and $\mathbf { z } \in { \mathcal { Z } }$ is the observation governed by a likelihood model $p ( \mathbf { z } \mid \pmb { \theta } , \mathbf { d } )$ .

## 2.2. Physical Constraints of Active Vision

To instantiate this framework, we first model the VLM as a stochastic sensor subject to rigorous resource limitations.

Definition 2.1 (Perceptual Bandwidth $B )$ . The fundamental bottleneck of VLM perception is the fixed encoder capacity (e.g., restricted token count (Dosovitskiy, 2020)), termed perceptual bandwidth B. This capacity induces a density-area trade-off (Najemnik & Geisler, 2005), where the information density ρ is defined as the ratio of the total bandwidth to the area $A ( \mathbf { d } )$ of a foveation crop:

$$
\rho ( \mathbf { d } ) \triangleq { \frac { B } { A ( \mathbf { d } ) } }
$$

Definition 2.2 (Resolution Probability ϕ). The probability that fine-grained features are resolved is governed by a saturation function $f _ { \mathrm { s a t } }$ (e.g., sigmoid) of information density:

$$
\phi ( \mathbf { d } ) \triangleq P ( \mathrm { R e s o l v e d } \mid \mathbf { d } ) = f _ { \mathrm { s a t } } ( \rho ( \mathbf { d } ) )\tag{1}
$$

This creates a physical trade-off: larger crops (high A) suffer from low density $( \phi  0 )$ , while smaller crops (low A) achieve high density (ϕ → 1).

Remark 2.3 (Analogy: The Semantic Nyquist Rate). The saturation behavior of $f _ { \mathrm { s a t } }$ mirrors the classical Nyquist-Shannon Sampling Theorem (Shannon, 1949). We posit the existence of a critical density threshold $\tau _ { \mathrm { n y q } } ,$ termed the semantic Nyquist Rate. When $\rho ( \mathbf { d } ) < \tau _ { \mathrm { n y q } }$ , the encoder fails to distinguish between distinct local features, rendering fine-grained features indistinguishable. Conversely, once the density exceeds this threshold, the features become recoverable. In our framework, the sigmoid function serves as a differentiable approximation of this critical transition.

## 2.3. The Generative Process

Visual reasoing is not a static task but an interactive loop initiated by the agent’s decisions. The generative process unfolds in three stages: action selection, physical interaction, and observation generation.

Design Space: Foveation Actions (D). Foveation actions are parameterised as spatial crops d $= [ u , v , w , h ] \in [ 0 , 1 ] ^ { 4 }$ Crucially, d acts as a control variable for bandwidth allocation: by selecting a smaller region $( w \cdot h \ll 1 )$ , the agent concentrates the fixed token budget onto a limited area, thereby boosting the local resolution density $\rho ( \mathbf { d } )$ and increasing the resolution probability ϕ(d).

Latent Parameters: Semantic & Spatial State (θ). We define the unknown state space as $\pmb { \theta } \triangleq \{ \ell , y \}$ , which factorises into two components: the spatial location ℓ of the relevant object and the semantic target y (e.g., the class label or text answer).

Agent’s Belief State. At any time step t, the agent’s knowledge about the latent parameters θ is captured by the joint posterior $p _ { t } ( \ell , y )$ . In real-world visual reasoing, spatial location ℓ and semantic identity y are often coupled (e.g., context implies location). However, maintaining a full highdimensional joint posterior is computationally intractable for real-time inference.

Assumption 2.4 (Factorised Belief Approximation). To ensure tractability during the sequential design process, we adopt a mean-field approximation (Blei et al., 2017), assuming that the spatial search and semantic identification are momentarily decoupled during planning:

$$
p _ { t } ( \ell , y ) \approx p _ { t } ( \ell ) \cdot p _ { t } ( y ) .
$$

Discussion. This approximation reduces complexity from the joint product space $\mathcal { L } \times \mathcal { V }$ to separate spatial and semantic factors, at the cost of ignoring higher-order spatial– semantic correlations. We use this factorisation only as a planning approximation, not as a claim that the true posterior is independent. The sequential feedback loop can partially mitigate this bias, since new observations reshape both the spatial and semantic beliefs through the context.

Under this assumption, we maintain two distinct belief maps: (1) A spatial belief $p _ { t } ( \ell )$ over the image coordinate space Ω, representing the agent’s uncertainty regarding the object’s location. (2) A semantic belief $p _ { t } ( y )$ , representing the uncertainty regarding the target’s identity (e.g., class distribution), initialised by the linguistic priors in Q. This separation allows the agent to explicitly reaso about “where to look” (spatial uncertainty reduction) as a distinct objective from “what it $\mathrm { i } \mathrm { s } ^ { \flat }$ (semantic identification), enabling the tractable EIG derivation in Section 3.

The core physical constraint is that semantic information is inaccessible unless the target is physically captured. This interaction is modelled by the visibility event S, which acts as a latent bottleneck between the world state and the sensor.

Definition 2.5 (The Visibility Event). To bridge physical actions and semantic observations, we define a binary latent indicator $S \in \{ 0 , 1 \}$ . This event represents whether the queried object is successfully captured by the encoder. Visibility occurs if and only if the object is both spatially encompassed and perceptually resolved:

$$
P ( \mathcal S = 1 \mid \ell , \mathbf d ) \triangleq \underbrace { \mathbb 1 \left[ \ell \in \mathbf d \right] } _ { \mathrm { s p a t i a l ~ c o v e r a g e } } \times \underbrace { \phi ( \mathbf d ) } _ { \mathrm { p e r c e p t u a l ~ r e s o l u t i o n } } ,\tag{2}
$$

where $\mathbb { 1 } [ \cdot ]$ is the indicator function, ℓ is the latent spatial location, and $\phi ( \mathbf { d } )$ is the resolution probability (Eq. 1).

Observation Generation (z). Finally, the visibility state S gates the information flow to the VLM. The generative process concludes with the emission of the observation z, which is a mixture of signal and noise modulated by S.

Definition 2.6 (Observation Model: Resolution-Modulated Likelihood). The visual observation z is governed by a mixture model conditioned on the latent state of S. By the Law of Total Probability over the visibility event, the likelihood $p ( \mathbf { z } \mid \pmb { \theta } , \mathbf { d } )$ is defined as:

$$
\begin{array} { r l } & { p ( \mathbf { z } \mid \boldsymbol { \theta } , \mathbf { d } ) = \displaystyle \sum _ { s \in \{ 0 , 1 \} } p ( \mathbf { z } \mid \boldsymbol { y } , \mathcal { S } = s , \boldsymbol { \ell } , \mathbf { d } ) P ( \mathcal { S } = s \mid \boldsymbol { \ell } , \mathbf { d } ) } \\ & { \quad \quad \quad = \underbrace { P ( \mathcal { S } = 1 \mid \boldsymbol { \ell } , \mathbf { d } ) } _ { \mathrm { G a t e o p e n } } \cdot p ( \mathbf { z } \mid \boldsymbol { y } , \mathbf { d } ) } \\ & { \quad \quad \quad + \underbrace { ( 1 - P ( \mathcal { S } = 1 \mid \boldsymbol { \ell } , \mathbf { d } ) ) } _ { \mathrm { G a t e C l o s e d } } \cdot p _ { 0 } ( \mathbf { z } \mid \mathbf { d } ) , } \end{array}\tag{3}
$$

where $p ( \mathbf { z } ~ \mid ~ y , \mathbf { d } ) ~ \triangleq ~ p ( \mathbf { z } ~ \mid ~ y , S ~ = ~ 1 , \ell , \mathbf { d } )$ denotes the informative signal distribution when resolved, which we treat as conditionally independent of ℓ given $s = 1$ . And $p _ { 0 } ( \mathbf { z } ) \triangleq p ( \mathbf { z } \mid S = 0 )$ denotes the background noise distribution. Note that $p _ { 0 } ( \mathbf { z } )$ is independent of $y \left( y \perp \mathbf { z } \mid S = 0 \right)$ representing the fact that an unresolved observation contains no semantic information about the target.

The complete generative process and the resulting decisiontheoretic structure are summarised in the influence diagram in Figure 2, which serves as the basis for our sequential strategy derivation in Section 3.

## 3. Active Visual Reasoing as S-BOED

Building on the generative process established in Section 2, we now formulate the S-BOED for active information foraging through a three-stage derivation. We first define the theoretical sequential objective and identify the “Information $\mathrm { C l i f f } ^ { \mathrm { , } \mathrm { , } }$ that renders standard greedy strategies insufficient (Sec. 3.1). To overcome computational intractability, we then derive a closed-form Coverage-Resolution utility under specific assumptions (Sec. 3.2). Finally, we present the idealised Bayesian belief update, which clarifies how positive and negative visual evidence should reshape the search distribution.

Throughout this section, all beliefs and information quantities at step t are conditioned on the interaction history $\mathcal { H } _ { t - 1 }$ For compactness, we write $p _ { t } ( \cdot )$ and $H _ { t } ( \cdot )$ for historyconditioned beliefs and entropies, and omit the subscript t in mutual-information terms when the conditioning is clear.

## 3.1. The Sequential Objective

The agent’s goal is to select a sequence of designs $\mathbf { d } _ { 1 : T }$ to reduce uncertainty about the latent state $\pmb { \theta } = \{ \ell , y \}$ . We quantify uncertainty using the Shannon entropy $H ( \pmb \theta )$

Expected Information Gain (EIG). For a single step, the utility of a design d is the expected reduction in entropy or, equivalently, the mutual information between the observation and the parameters:

$$
\operatorname { E I G } ( \mathbf { d } ) \triangleq { \mathcal { Z } } ( \mathbf { z } ; \pmb { \theta } \mid \mathbf { d } ) = H ( \pmb { \theta } ) - \mathbb { E } _ { \mathbf { z } \sim p ( \mathbf { z } \mid \mathbf { d } ) } [ H ( \pmb { \theta } \mid \mathbf { z } , \mathbf { d } ) ] .
$$

Sequential Planning via Bellman Equation. In the sequential setting, the agent maintains a history $\mathcal { H } _ { t - 1 }$ . The optimal strategy $\pi ^ { * }$ maximises the cumulative information gain over a horizon T . This is formally characterised by the value function $V ^ { * }$ , which satisfies the Bellman equation:

$$
\begin{array} { r l } & { V ^ { * } ( \mathcal { H } _ { t - 1 } ) = \underset { \mathbf { d } _ { t } } { \mathrm { m a x } } \left( \underset { \mathrm { i m m e d i a t e ~ } \mathbf { g } \mathrm { i n } } { \mathrm { E I G } \left( \mathbf { d } _ { t } \mid \mathcal { H } _ { t - 1 } \right) } \right. } \\ & { \quad \quad \quad \quad \left. + \underset { \mathrm { \Gamma } \mathbf { g } \in p \left( \mathbf { z } \mid \mathcal { H } _ { t - 1 } , \mathbf { d } _ { t } \right) } { \mathrm { E } _ { \mathbf { z } _ { t } \sim p \left( \mathbf { z } \mid \mathcal { H } _ { t - 1 } , \mathbf { d } _ { t } \right) } \left[ V ^ { * } \big ( \mathcal { H } _ { t - 1 } \cup \big \{ ( \mathbf { d } _ { t } , \mathbf { z } _ { t } ) \big \} \big ) \right] } \right) . } \end{array}\tag{4}
$$

Computing Eq. 4 requires solving a nested expectation over high-dimensional observations z, which is computationally intractable. Furthermore, the structure of visual information poses a unique theoretical challenge:

Remark 3.1 (The Information Cliff). Standard active learning often assumes submodularity (diminishing returns) to justify greedy strategies. However, constrained active vision is often super-additive. Consider reading a small text: A wide view $\mathbf { \Pi } ( \mathbf { d } _ { \mathrm { w i d e } } )$ locates the text but cannot read it $( \phi  0 )$ ; a random zoom $\mathbf { \Pi } ( \mathbf { d } _ { \mathrm { z o o m } } )$ can read but misses the location $( \ell \notin \mathbf { d } )$ . Both yield zero gain individually. Only their sequence yields high information:

$$
\begin{array} { r } { \mathcal { T } ( y ; \mathbf { z } _ { \mathrm { w i d e } } , \mathbf { z } _ { \mathrm { z o o m } } ) \gg \mathcal { T } ( y ; \mathbf { z } _ { \mathrm { w i d e } } ) + \mathcal { T } ( y ; \mathbf { z } _ { \mathrm { z o o m } } ) . } \end{array}
$$

This “information cliff” requires look-ahead planning.

## 3.2. Derivation of the Tractable Coverage-Resolution Objective

While the ideal agent optimises the sequential Bellman equation, the nested expectations over high-dimensional observations z render it computationally intractable. In this section, we derive a closed-form approximation for the immediate task-relevant information gain that drives our practical cropselection strategy.

The Joint Information Objective. The ultimate goal of the agent is to resolve the user’s query y. However, due to the physical coupling between “seeing” and “understanding”, the agent must jointly reaso about the full latent state $\pmb \theta =$ $\{ \ell , y \}$ . Theoretically, the total information gain decomposes into spatial and semantic components:

$$
\begin{array} { r } { \mathbf { \mathcal { T } } ( \mathbf { z } ; \ell , y \mid \mathbf { d } ) = \underbrace { \mathbf { \mathcal { T } } ( \mathbf { z } ; \ell \mid \mathbf { d } ) } _ { \mathrm { L o c a l i z a t i o n G a i n } } + \underbrace { \mathcal { T } ( \mathbf { z } ; y \mid \ell , \mathbf { d } ) } _ { \mathrm { S e m a n t i c G a i n } } . } \end{array}
$$

In our active vision setting, resolving y strictly necessitates localising ℓ. Rather than optimising these terms separately, we focus on maximising the marginal mutual information regarding the semantic target y.

Decomposition via the Visibility Event. Directly computing $\mathcal { T } ( \mathbf { z } ; y \mid \mathbf { d } )$ is intractable. To simplify, we introduce the auxiliary visibility variable S. We first introduce a crucial assumption regarding the VLM’s self-calibration:

Assumption 3.2 (Calibrated Visibility). We assume the observation z encodes sufficient statistics to determine the visibility state S (e.g., the model can distinguish between “blurry/empty” and “resolved” content). Mathematically, this implies $H ( S \mid \mathbf { z } , \mathbf { d } ) \approx 0 .$ , which allows us to approximate $\mathcal { T } ( y ; \mathbf { z } \mid \mathbf { d } ) \approx \mathcal { T } ( y ; \mathbf { z } , S \mid \mathbf { d } )$ . This assumption is empirically supported by recent findings that large-scale foundation models exhibit high calibration regarding their own predictive uncertainty (Kadavath et al., 2022).

By the chain rule,

$$
{ \cal T } _ { t } ( y ; { \mathbf { z } } _ { t } , S \mid { \mathbf { d } } ) = { \cal T } _ { t } ( y ; { \mathbf { z } } _ { t } \mid { \mathbf { d } } ) + { \cal T } _ { t } ( y ; S \mid { \mathbf { z } } _ { t } , { \mathbf { d } } ) .
$$

Since

$$
\begin{array} { r } { \mathcal { T } _ { t } ( y ; S \mid \mathbf { z } _ { t } , \mathbf { d } ) \le H _ { t } ( S \mid \mathbf { z } _ { t } , \mathbf { d } ) \approx 0 , } \end{array}
$$

Assumption 3.2 gives

$$
\begin{array} { r } { \mathcal { T } _ { t } ( y ; \mathbf { z } _ { t } \mid \mathbf { d } ) \approx \mathcal { T } _ { t } ( y ; \mathbf { z } _ { t } , S \mid \mathbf { d } ) . } \end{array}
$$

We then decompose the right-hand side as

$$
\begin{array} { r } { \mathcal { T } _ { t } ( y ; \mathbf { z } _ { t } , S \mid \mathbf { d } ) = \underbrace { \mathcal { T } _ { t } ( y ; S \mid \mathbf { d } ) } _ { \mathrm { T e r m ~ 1 } } + \underbrace { \mathcal { T } _ { t } ( y ; \mathbf { z } _ { t } \mid S , \mathbf { d } ) } _ { \mathrm { T e r m ~ 2 } } . } \end{array}
$$

We analyse these two terms based on the conditional independence properties established in Section 2:

Term 1. As illustrated in Figure 2, the visibility event S is structurally determined by the spatial parameters (ℓ, d) and sensor physics. Under the Factorised Belief Assumption (Assumption $2 . 4 )$ , the semantic identity y is independent of the spatial location ℓ during the planning phase $( \ell \perp y )$ Consequently, since $s$ is a function of ℓ, it follows that $y \perp S \mid \mathbf { d } .$ . Thus, $\mathcal { T } ( y ; \boldsymbol { S } \mid \mathbf { d } ) = 0$

Term 2. We expand the second term using the definition of conditional mutual information:

$$
\begin{array} { r l } & { { \mathscr { T } } _ { t } ( y ; { \mathbf { z } } _ { t } \mid { \boldsymbol { \mathcal { S } } } , { \mathbf { d } } ) = P _ { t } ( { \boldsymbol { \mathcal { S } } } = 1 \mid { \mathbf { d } } ) \cdot { \mathscr { T } } _ { t } ( y ; { \mathbf { z } } _ { t } \mid { \boldsymbol { \mathcal { S } } } = 1 , { \mathbf { d } } ) } \\ & { \qquad + P _ { t } ( { \boldsymbol { S } } = 0 \mid { \mathbf { d } } ) \cdot \underbrace { { \mathscr { T } } _ { t } ( y ; { \mathbf { z } } _ { t } \mid { \boldsymbol { \mathcal { S } } } = 0 , { \mathbf { d } } ) } _ { = 0 } . } \end{array}
$$

The second part vanishes because an unresolved observation $( S = 0 )$ yields only background noise independent of y $( y \perp \mathbf { z } \mid S = 0 )$

Combining these results, we define the Semantic Information Gain objective:

$$
\tilde { \mathcal { T } } _ { t } ( \mathbf { d } ) \triangleq P _ { t } ( S = 1 \mid \mathbf { d } ) \cdot \mathcal { T } _ { t } ( y ; \mathbf { z } _ { t } \mid \mathcal { S } = 1 , \mathbf { d } ) .\tag{5}
$$

The Perfect Perception Approximation. Eq. 5 remains difficult to compute. To proceed, we rely on the strong semantic extraction capabilities of modern VLMs.

Assumption 3.3 (Ideal Observer / Entropy Collapse). For planning tractability, we model the VLM as an ideal observer. We assume that if the target is successfully foveated and resolved $( S = 1 )$ , the VLM extracts semantic information with high fidelity, causing the conditional entropy of y to collapse to zero:

$$
H ( y \mid \mathbf { z } , S = 1 , \mathbf { d } ) \approx 0 .
$$

This implies that the information gain from a successful foveation is approximately equal to the prior uncertainty:

$$
{ \mathcal { T } } ( y ; \mathbf { z } \mid S = 1 , \mathbf { d } ) = H ( y ) - H ( y \mid \mathbf { z } , S = 1 , \mathbf { d } ) \approx H ( y ) .
$$

Remark 3.4. This assumption reduces the complex objective of semantic disambiguation to a geometric objective of visibility maximisation. It implies that the search strategy is responsible for acquiring high-fidelity evidence, while the interpretation of that evidence is delegated to the backbone VLM. While this ignores cases of hallucination on clear images, it is a necessary condition for tractable planning in open-ended spaces.

Under Assumption 3.3, the successful-foveation information term satisfies $\mathcal { T } _ { t } ( y ; \mathbf { z } _ { t } \mid S = 1 , \mathbf { d } ) \approx H _ { t } ( y )$ . Substituting this into Eq. 5 gives

$$
\tilde { \mathcal { T } } _ { t } ( \mathbf { d } ) \approx H _ { t } ( y ) P _ { t } ( S = 1 \mid \mathbf { d } ) .\tag{6}
$$

The remaining term is the probability that the latent target location is both covered by the crop and resolved under the fixed perceptual bandwidth. Marginalising over the current spatial belief $p _ { t } ( \ell )$ gives

$$
P _ { t } ( S = 1 \mid \mathbf { d } ) = \left( \int _ { \mathbf { x } \in \mathbf { d } } p _ { t } ( \mathbf { x } ) d \mathbf { x } \right) \phi ( \mathbf { d } ) \triangleq \mathcal { I } _ { t } ( \mathbf { d } ) .\tag{7}
$$

Thus, $\tilde { \mathcal { T } } _ { t } ( \mathbf { d } ) \approx H _ { t } ( y ) \mathcal { T } _ { t } ( \mathbf { d } )$ . Since $H _ { t } ( y )$ is independent of the current design d, maximising the task-relevant information gain reduces to maximising the coverage–resolution objective $\mathcal { T } _ { t } ( \mathbf { d } )$

Proposition 3.5 (Task-Relevant EIG Approximation). Under the Factorised Belief Approximation (Assump. $2 . 4 ) ,$ the Calibrated Visibility Assumption (Assump. 3.2), and the Ideal Observer Approximation (Assump. 3.3), the taskrelevant EIG about the answer variable y satisfies

$$
U _ { t } ( \mathbf { d } ) \triangleq { \mathcal { T } } _ { t } ( y ; \mathbf { z } _ { t } \mid \mathbf { d } ) \approx H _ { t } ( y ) { \mathcal { T } } _ { t } ( \mathbf { d } ) ,
$$

where $\mathcal { T } _ { t } ( \mathbf { d } )$ is the coverage–resolution objective defined in Eq. 7. Since $H _ { t } ( y )$ is independent of the current design d, maximising $U _ { t } ( \mathbf { d } )$ reduces to maximising $\mathcal { T } _ { t } ( \mathbf { d } )$

The Coverage–Resolution Product. The objective $\mathcal { T } _ { t } ( \mathbf { d } )$ has a simple interpretation. Visibility requires the latent target location ℓ to be both spatially covered by the crop and perceptually resolved under the fixed visual-token budget:

$$
\mathcal { T } _ { t } ( \mathbf { d } ) = \underbrace { \left( \int _ { \mathbf { x } \in \mathbf { d } } p _ { t } ( \mathbf { x } ) d \mathbf { x } \right) } _ { \mathrm { C o v e r a g e } } \times \underbrace { \phi ( \mathbf { d } ) } _ { \mathrm { R e s o l u t i o n } } .\tag{8}
$$

The greedy design is therefore selected as $\begin{array} { r l } { \mathbf { d } _ { t } ^ { * } } & { { } = } \end{array}$ $\mathrm { a r g m a x } _ { \mathbf { d } } \mathcal { T } _ { t } ( \mathbf { d } )$ . This objective makes the coverage– resolution trade-off explicit: larger crops cover more posterior mass but reduce effective perceptual resolution, while smaller crops increase resolution but risk missing the target.

## 3.3. Formal Bayesian Belief Update

The coverage–resolution objective depends on the current spatial belief $p _ { t } ( \ell )$ . In the idealised Bayesian model, this belief would be updated explicitly after each observation. Although our practical implementation approximates this update implicitly through the interaction history, the formal update clarifies how positive and negative visual evidence should reshape the search distribution.

Upon executing the optimal design $\mathbf { d } _ { t } ^ { * }$ and receiving observation $\mathbf { z } _ { t } .$ , the agent updates its spatial belief map $p _ { t } ( \ell )$ using Bayes’ rule:

$$
p _ { t + 1 } ( \ell ) = \frac { p ( \mathbf { z } _ { t } \mid \ell , \mathbf { d } _ { t } ^ { * } ) \cdot p _ { t } ( \ell ) } { \mathcal { Z } _ { t } } ,\tag{9}
$$

where $\mathcal { Z } _ { t }$ is the normalisation constant.

The core of this update is the spatial likelihood function $p ( \mathbf { z } _ { t } \mid \ell , \mathbf { d } _ { t } ^ { * } )$ . To derive this from the joint observation model (Definition 2.6), we marginalise over the semantic target y. Relying on the Factorised Belief Assumption (Assumption 2.4), which treats y and ℓ as independent during the inference step, the likelihood simplifies to:

$$
p ( \mathbf { z } _ { t } \mid \ell , \mathbf { d } _ { t } ^ { * } ) \approx \mathbb { E } _ { y \sim p _ { t } ( y ) } \left[ p ( \mathbf { z } _ { t } \mid \ell , y , \mathbf { d } _ { t } ^ { * } ) \right] .
$$

Substituting the mixture model from Eq. 3 into this expectation, the likelihood bifurcates based on whether the latent location ℓ falls within the crop region $\mathbf { d } _ { t } ^ { * }$

$$
p ( \mathbf { z } _ { t } \mid \ell , \mathbf { d } _ { t } ^ { * } ) = \left\{ \begin{array} { l l } { \phi ( \mathbf { d } _ { t } ^ { * } ) \mathbb { E } _ { y \sim p _ { t } ( y ) } [ p ( \mathbf { z } _ { t } \mid y , \mathbf { d } _ { t } ^ { * } ) ] } & { \mathrm { i f ~ } \ell \in \mathbf { d } _ { t } ^ { * } , } \\ { + \big ( 1 - \phi ( \mathbf { d } _ { t } ^ { * } ) \big ) p _ { 0 } ( \mathbf { z } _ { t } \mid \mathbf { d } _ { t } ^ { * } ) } & { \mathrm { i f ~ } \ell \in \mathbf { d } _ { t } ^ { * } , } \\ { p _ { 0 } ( \mathbf { z } _ { t } \mid \mathbf { d } _ { t } ^ { * } ) } & { \mathrm { i f ~ } \ell \not \in \mathbf { d } _ { t } ^ { * } . } \end{array} \right.
$$

Interpretation and Negative Evidence. The term $\mathbb { E } _ { y \sim p _ { t } ( y ) } [ p ( \mathbf { z } _ { t } \mid y , \mathbf { d } _ { t } ^ { * } ) ]$ represents the marginal likelihood of the observation given that the target is resolved, averaged over the agent’s current semantic belief. It quantifies how well the visual observation $\mathbf { z } _ { t }$ supports the hypothesis that any valid target y is present in the crop.

Crucially, this structure enables updates via negative evidence. Consider the scenario where the agent scans a candidate region with high effective perceptual resolution $( \phi ( \mathbf { d } _ { t } ^ { * } ) \approx 1 )$ but receives an uninformative observation $\left( \mathrm { i } . \mathrm { e } . , \mathrm { z } _ { t } \right.$ matches the background noise $p _ { 0 } )$ . For locations inside the crop $( \ell \in \mathbf { d } _ { t } ^ { * } )$ , the likelihood collapses to the signal probability, which is vanishingly small for noise inputs $( \mathbb { E } _ { y } [ p ( \mathbf { z } _ { t } \mid y ) ] \ll p _ { 0 } ( \mathbf { z } _ { t } ) )$ . Conversely, for unvisited locations $( \ell \notin \mathbf { d } _ { t } ^ { * } )$ ), the likelihood remains high at the baseline noise level $p _ { 0 } ( \mathbf { z } _ { t } \mid \mathbf { d } _ { t } ^ { * } )$ , reflecting consistency with the “not seen” state. Through normalisation, this discrepancy suppresses the probability mass within the visited area $( \mathbf { d } _ { t } ^ { * } )$ and effectively “pushes” the belief mass to the unvisited regions, driving exploration.

## 4. Algorithmic Realisation

The S-BOED formulation specifies a decision-theoretic objective, but exact inference is intractable in gigapixel image spaces. In particular, Eq. 8 depends on a spatial belief $p _ { t } ( \ell )$ over a continuous domain, an unknown resolution function $\phi ( \mathbf { d } )$ , and, for non-myopic planning, expectations over future observations. We therefore instantiate the framework with FOVEA, a training-free procedure for Foveated Observation and Visual Evidence Acquisition. FOVEA should be understood as a practical surrogate instantiation of the S-BOED view rather than an exact solver with explicit posterior maps or exact EIG computation.

FOVEA uses the interaction history $\mathcal { H } _ { t }$ as a historyconditioned search state, so later crop proposals can depend on both positive and negative evidence from earlier views. Appendix E provides empirical evidence for this history-based calibration.

Operationally, FOVEA has two main components: it estimates crop utility with a resolvability probe, and it optimises this utility with greedy sampling, MCMC-style refinement, or look-ahead planning.

## 4.1. Resolvability Probing

Zero-shot visual grounding in high-resolution regimes remains prone to spatial inaccuracies and hallucinations (Xiao et al., 2025; Su et al., 2025a). We therefore treat the VLM’s initial crop proposal as a noisy spatial prior rather than ground truth. Around this proposal, FOVEA samples candidate foveations and scores each crop independently.

We introduce a binary resolvability signal $r \in \{ 0 , 1 \}$ , where $r ~ = ~ 1$ denotes that the crop contains sufficient queryrelevant visual evidence for the VLM to answer. This signal is not an exact estimator of information gain; rather, it is an empirical surrogate for crop utility under the S-BOED view. We define

$$
\hat { \mathcal { I } } ( \mathbf { d } ) \triangleq P ( r = 1 | I _ { \mathbf { d } } , Q ) \approx P ( \mathrm { V L M } ( I _ { \mathbf { d } } , Q ) = ^ { * } \mathrm { Y e s } ^ { \prime , \prime } ) ,\tag{10}
$$

which estimates whether a candidate achieves a favourable coverage–resolution trade-off for the current query.

## 4.2. Optimisation Strategies

Given $\hat { \mathcal { I } } ( \mathbf { d } )$ , FOVEA supports different optimisers. The greedy variant selects the candidate with the largest immediate resolvability score and is used as the efficient default. MCMC-style refinement improves local search by iteratively perturbing the crop proposal. For tasks with an information cliff, where the value of a view depends on what it enables next, we use a FOVEA-Lookahead that scores a candidate by the estimated resolvability of its simulated next state:

$$
\mathbf { d } _ { t } ^ { * } = \underset { \mathbf { d } \in \mathcal { D } _ { \mathrm { c a n d } } } { \operatorname { a r g m a x } } \hat { V } ( \mathbf { d } , \mathcal { H } _ { t - 1 } ) .
$$

This keeps the objective fixed while allowing the optimiser to vary with the compute budget and task difficulty.

Algorithm 1 FOVEA: S-BOED-Guided Local Perceptua   
Refinement   
1: Require: Global image $I _ { \mathrm { g l o b a l } } ,$ , query $Q$   
2: Input: Initial crop proposal $\mathbf { d } _ { \mathrm { s e e d } }$   
3: Generate a candidate pool $\mathcal { D } _ { \mathrm { c a n d } }$ around $\mathbf { d } _ { \mathrm { s e e d } } ,$ includ  
ing the seed crop and local perturbations   
4: for each $\mathbf { d } _ { i } \in \mathcal { D } _ { \mathrm { c a n d } }$ do   
5: Extract crop $I _ { \mathbf { d } _ { i } }$   
6: Estimate utility $\hat { \mathcal { I } } ( \mathbf { d } _ { i } ) \gets P ( r = 1 | I _ { \mathbf { d } _ { i } } , Q )$   
7: end for   
8: if strategy is LOOKAHEAD then   
9: $\mathbf { d } _ { t } ^ { * } \gets \mathrm { a r g m a x } _ { \mathbf { d } \in \mathcal { D } _ { \mathrm { c a n d } } } \hat { V } ( \mathbf { d } , \mathcal { H } _ { t - 1 } )$   
10: else   
11: $\mathbf { d } _ { t } ^ { * } \gets \mathrm { a r g m a x } _ { \mathbf { d } \in \mathcal { D } _ { \mathrm { c a n d } } } \hat { \mathcal { I } } ( \mathbf { d } )$   
12: end if   
13: $\mathbf { z } _ { t } \gets \mathrm { V L M } ( I _ { \mathbf { d } _ { t } ^ { * } } , Q )$   
14: $\mathcal { H } _ { t }  \mathcal { H } _ { t - 1 } \cup \{ ( \mathbf { d } _ { t } ^ { * } , \hat { \mathcal { I } } ( \mathbf { d } _ { t } ^ { * } ) , \mathbf { z } _ { t } ) \}$   
15: return $\mathcal { H } _ { t }$

## 5. Experiments: The Tool-Integrated Agent

We instantiate the S-BOED framework with FOVEA, a plugin inference-time module that intercepts the VLM’s crop commands and refines them before tool execution. This refinement is essential because external vision experts (e.g., OCR, Detection, and Segmentation) are equally subject to the perceptual bandwidth bottleneck. Without highresolution inputs, these tools struggle to resolve dense or minute features in down-sampled global views (Akyon et al., 2022; Singh et al., 2019). Consequently, the cropping operation serves as a fundamental bridge to deliver high-fidelity signals to both the reasoing VLM and downstream tools. FOVEA optimises this critical interface by refining the crop’s spatial parameters to maximise its informative utility.

Benchmarks. We assess FOVEA on four benchmarks: HR-Bench (Team, 2024), MME-RealWorld-Lite (Zhang et al., 2024), V\*Bench (Wu & Xie, 2023), and CV-Bench (Tong et al., 2024). These datasets cover fine-grained recognition, small-object search, and 3D reasoing, all of which require high-fidelity local information. Together, these benchmarks test whether active crop refinement can alleviate the perceptual bandwidth bottleneck across recognition, search, and spatial reasoing tasks.

Baselines. We compare FOVEA against three groups of baselines. Proprietary models such as GPT-5 and Gemini establish the performance frontier for multi-modal reasoing, while Thyme (Zhang et al., 2025b) represents SOTA RL-based methods for integrated tool-use. Qwen3-VL-30B-A3B-Instruct serves as our controlled foundation model, processing only down-sampled global views without an active loop. The ReAct agent uses the same backbone and tool interface, but directly executes the VLM-proposed crop commands without S-BOED-guided refinement.

Table 1. Main results on multimodal benchmarks. FOVEA is compared with proprietary models, prior visual agents, and controlled direct/ReAct baselines under matched backbone settings where applicable. Bold indicates the best result within each comparison block.
<table><tr><td>Method</td><td>Backbone</td><td>MME-RealW</td><td>CV-Bench</td><td>V*</td><td>HR-Bench (4K)</td><td>HR-Bench (8K)</td><td>Mean</td></tr><tr><td colspan="8">State-of-the-Art &amp; Prior Agents</td></tr><tr><td>Thyme (Zhang et al., 2025b)</td><td>Qwen-2.5-VL-7B</td><td>55.2%</td><td>78.4%</td><td>82.2%</td><td>77.0%</td><td>72.0%</td><td>73.0%</td></tr><tr><td>GPT-5</td><td>Proprietary</td><td>55.0%</td><td>84.9%</td><td>77.0%</td><td>78.1%</td><td>75.5%</td><td>74.1%</td></tr><tr><td>Gemini 2.5 Flash</td><td>Proprietary</td><td>58.5%</td><td>87.3%</td><td>80.1%</td><td>83.4%</td><td>80.9%</td><td>78.0%</td></tr><tr><td colspan="8">Controlled Comparison (30B Backbone)</td></tr><tr><td>Direct</td><td>Qwen3-VL-30B-A3B-Instruct</td><td>48.2%</td><td>81.2%</td><td>81.2%</td><td>80.0%</td><td>75.9%</td><td>73.3%</td></tr><tr><td>ReAct Agent</td><td>Qwen3-VL-30B-A3B-Instruct</td><td>51.1%</td><td>81.3%</td><td>83.8%</td><td>80.8%</td><td>78.3%</td><td>75.1%</td></tr><tr><td>RAP (Wang et al., 2025)</td><td>Qwen3-VL-30B-A3B-Instruct</td><td>40.8%</td><td>72.2%</td><td>86.4%</td><td>79.6%</td><td>80.6%</td><td>71.9%</td></tr><tr><td>FOVEA (ours)</td><td>Qwen3-VL-30B-A3B-Instruct</td><td>54.6%</td><td>84.8%</td><td>85.3%</td><td>84.5%</td><td>79.2%</td><td>77.7%</td></tr><tr><td colspan="8">Controlled Comparison (8B Backbone)</td></tr><tr><td>Direct</td><td>Qwen3-VL-8B-Instruct</td><td>47.6%</td><td>84.5%</td><td>76.9%</td><td>74.5%</td><td>70.9%</td><td>70.9%</td></tr><tr><td>ReAct Agent</td><td>Qwen3-VL-8B-Instruct</td><td>48.1%</td><td>83.9%</td><td>78.8%</td><td>77.7%</td><td>73.8%</td><td>72.5%</td></tr><tr><td>FOVEA (ours)</td><td>Qwen3-VL-8B-Instruct</td><td>49.9%</td><td>84.7%</td><td>83.6%</td><td>80.9%</td><td>75.4%</td><td>74.9%</td></tr></table>

Implementation Details. Our primary results in Table 1 use the efficient greedy instantiation of FOVEA. To maintain computational tractability across the full benchmark, we prioritise inference-time efficiency over the more exhaustive look-ahead planning. For each initial crop proposal $\mathbf { d } _ { \mathrm { s e e d } } ,$ we generate two local perturbations, yielding a threecandidate pool $\{ { \bf d } _ { \mathrm { s e e d } } , { \bf d } _ { \mathrm { s m a l l } } , { \bf d } _ { \mathrm { l a r g e } } \}$ , and perform $K = 3$ stochastic probes per candidate to estimate $\hat { \mathcal { I } } ( \mathbf { d } )$ . The final action $\mathbf { d } _ { t } ^ { * }$ is selected greedily (Algorithm 1). More computationally intensive strategies (MCMC, look-ahead) are reserved for challenging subsets (Section 5.2).

![](images/0458f4e3c1cfc6cd905a322e10ba61c324717c18f0b526b1ef7ccf66b520ef58.jpg)  
Figure 3. Search efficacy in the gigapixel regime. We compare Direct, ReAct, and FOVEA variants on the Remote Sensing subset against an oracle-crop baseline. FOVEA-Lookahead yields the largest gain, while the remaining oracle gap reflects residual backbone recognition and reasoing errors.

## 5.1. Main Results

Table 1 presents a comparative analysis across all evaluated benchmarks. On the 30B backbone, FOVEA achieves a mean score of 77.7%, improving over the ReAct baseline (75.1%) and the Qwen3-VL-30B-Instruct baseline (73.3%), and also surpassing the related active-perception baseline RAP (Wang et al., 2025) (71.9%). These results remain competitive with proprietary frontiers like Gemini 2.5 Flash (78.0%). To assess whether the framework transfers beyond a single large backbone, we additionally evaluate it with the substantially smaller Qwen3-VL-8B-Instruct. The same overall trend holds: FOVEA improves the mean score from 70.9% / 72.5% (Direct / ReAct) to 74.9%, indicating that the proposed strategy is not specific to a single backbone scale. These gains indicate that better evidence acquisition can complement model scaling and heuristic tool use.

## 5.2. Strategy Efficacy in the Gigapixel Regime

To isolate the contribution of our search strategy from the VLM’s semantic reasoing capabilities, we conduct a focused ablation on the Remote Sensing subset of MME-RealWorld-lite (Zhang et al., 2024). This setting is searchdominated: images are extremely large, targets are sparse, and task-relevant regions are often nearly invisible in the downsampled global view. We compare Direct, ReAct, and FOVEA variants against an oracle-crop baseline, where the VLM is given a human-annotated crop. The oracle does not represent perfect answering; rather, it separates search failures from recognition failures.

Analysis. As visualised in Figure 3, ReAct improves over the base model by enabling active tool use, but remains limited by noisy crop proposals. FOVEA-Greedy and FOVEA-MCMC further improve accuracy by refining local foveations, while FOVEA-Lookahead reaches 54.7%, compared with 45.1% for ReAct. The remaining gap to the oracle-crop baseline shows that evidence acquisition and backbone recognition are distinct bottlenecks: even with task-relevant crops, the VLM can still misrecognise or misreaso. We analyse token costs in Appendix D.5, recognition failures in Appendix H.3, and qualitative cases in Appendix I.

## 5.3. Compute–Accuracy Scaling of Search Strategies

The previous section compares search strategies at a fixed budget on the full 150-question Remote Sensing set. To assess each strategy’s scaling potential, we also vary the search budget within each policy family on a 50-example subset and plot accuracy versus average tokens per question (Figure 4). For FOVEA-Greedy, the budget is the number of sampled branches; for FOVEA-MCMC, the number of refinement iterations; and for FOVEA-Lookahead, the number of search branches.

![](images/9150e728c590657e9fa926654553b0fb856c098094b479da3c46827d9b7c0295.jpg)  
Figure 4. Accuracy–compute scaling of FOVEA variants on a 50-example remote-sensing subset. We vary the search budget within each policy family and report accuracy against average total tokens per question.

Analysis. The trend is monotonic within each family: a higher search budget yields higher accuracy, but at increasing token cost. This indicates that FOVEA should be viewed as a family of compute–accuracy operating points rather than a single fixed policy. Lower-budget variants such as FOVEA-Greedy provide cheaper moderate gains and are suitable when latency is constrained, while higher-budget FOVEA-Lookahead yields larger improvements in searchdominated settings where additional search budget translates into meaningful gains.

This suggests that active perception provides a complementary axis of inference-time scaling: additional compute can be spent on acquiring higher-value visual evidence, not only on generating longer textual reasoing traces.

## 6. Conclusion

Summary. We formalise active visual reasoing under the perceptual bandwidth bottleneck as a sequential Bayesian optimal experimental design (S-BOED) problem. This perspective treats foveation not as heuristic preprocessing, but as principled acquisition of task-relevant visual evidence under a limited visual-token budget. We instantiate it with FOVEA, a training-free inference-time procedure that refines VLM crop proposals using a coverage–resolution utility estimated by resolvability probing. Experiments on high resolution and gigapixel benchmarks show consistent gains over direct and ReAct-style baselines, especially in searchdominated remote-sensing settings. Overall, our results suggest that high-resolution VLM reasoing should be viewed as both semantic reasoing and active evidence acquisition.

Limitations. This work has three main limitations. First, the derivation relies on the Ideal Observer approximation (Assump. 3.3); in practice, even oracle-level crops may still cause backbone recognition or reasoing failures. Second, FOVEA is proposal-limited: if the relevant region never enters the candidate pool, local refinement and look-ahead cannot recover it. We analyse this cold-start bottleneck in Appendix H.2 and discuss multi-seed proposals as a mitigation. Third, resolvability probing and search add inference-time overhead, so FOVEA is best viewed as a compute–accuracy trade-off frontier rather than a fixed-cost policy.

Future Work. We identify three directions: (1) Uncertainty Calibration. Improving estimators of the VLM’s epistemic uncertainty to sharpen task-relevant informationgain estimates (Choudhury et al., 2025; Feng et al., 2025). (2) Amortised Inference. Training a lightweight policy to predict useful foveation actions directly, using the coverage–resolution objective or downstream utility as supervision (Foster et al., 2019; Gershman & Goodman, 2014), thereby reducing the cost of iterative search. (3) Adaptive Invocation. The current implementation refines every crop call. A natural extension is a meta-policy that decides when active foveation is worth invoking, based on expected value of information (Howard, 1966) or causal necessity (Yu et al., 2026), connecting to the broader principle of targeted intervention (Liu et al., 2025; 2024a) that selective intervention is often more cost-effective than uniform guidance.

## Impact Statement

This work advances active perception and decision-making for multimodal agents. By formulating high-resolution visual reasoing as sequential experimental design, it provides a principled way for agents to decide what visual evidence to acquire before answering. This perspective may benefit applications where task-relevant information is sparse, local, or fine-grained, including remote sensing, document understanding, robotics, industrial inspection, medical imaging assistance, and embodied AI.

At the same time, more capable active perception systems may increase the reliability and scalability of automated visual monitoring and surveillance. Such systems should therefore be deployed with task-specific validation, transparency about acquired visual evidence, and appropriate human oversight in high-stakes settings. More broadly, our results suggest that improving multimodal agents requires not only stronger reasoing models, but also principled control over how models allocate perceptual resources.

## References

Akyon, F. C., Altinuc, S. O., and Temizel, A. Slicing aided hyper inference and fine-tuning for small object detection. In 2022 IEEE international conference on image processing (ICIP), pp. 966–970. IEEE, 2022.

Aloimonos, J., Weiss, I., and Bandyopadhyay, A. Active vision. International journal of computer vision, 1(4): 333–356, 1988.

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Bajcsy, R. Active perception. Proceedings of the IEEE, 76 (8):966–1005, 1988.

Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. Variational inference: A review for statisticians. Journal of the American statistical Association, 112(518):859–877, 2017.

Campbell, D. I., Rane, S., Giallanza, T., Sabbata, C. N. D., Ghods, K., Joshi, A., Ku, A., Frankland, S. M., Griffiths, T. L., Cohen, J. D., and Webb, T. W. Understanding

the limits of vision language models through the lens of the binding problem. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum? id=Q5RYn6jagC.

Chaloner, K. and Verdinelli, I. Bayesian experimental design: A review. Statistical science, pp. 273–304, 1995.

Choudhury, D., Williamson, S., Golinski, A., Miao, N.,´ Smith, F. B., Kirchhof, M., Zhang, Y., and Rainforth, T. Bed-llm: Intelligent information gathering with llms and bayesian experimental design. arXiv preprint arXiv:2508.21184, 2025.

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoing, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Connolly, C. The determination of next best views. In Proceedings. 1985 IEEE international conference on robotics and automation, volume 2, pp. 432–435. IEEE, 1985.

Dosovitskiy, A. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Feng, Y., Zhou, B., Lin, W., and Roth, D. BIRD: A trustworthy bayesian inference framework for large language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=fAAaT826Vv.

Foster, A., Jankowiak, M., Bingham, E., Horsfall, P., Teh, Y. W., Rainforth, T., and Goodman, N. Variational bayesian optimal experimental design. Advances in neural information processing systems, 32, 2019.

Gao, Z., Zhang, B., Li, P., Ma, X., Yuan, T., Fan, Y., Wu, Y., Jia, Y., Zhu, S.-C., and Li, Q. Multi-modal agent tuning: Building a VLM-driven agent for efficient tool usage. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=0bmGL4q7vJ.

Gershman, S. and Goodman, N. Amortized inference in probabilistic reasoing. In Proceedings of the annual meeting of the cognitive science society, volume 36, 2014.

Golovin, D. and Krause, A. Adaptive submodularity: Theory and applications in active learning and stochastic optimization. Journal of Artificial Intelligence Research, 42:427–486, 2011.

Gupta, T. and Kembhavi, A. Visual programming: Compositional visual reasoing without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14953–14962, 2023.

Houlsby, N., Huszar, F., Ghahramani, Z., and Lengyel, M.´ Bayesian active learning for classification and preference learning. arXiv preprint arXiv:1112.5745, 2011.

Howard, R. A. Information value theory. IEEE Transactions on systems science and cybernetics, 2(1):22–26, 1966.

Hu, Z., Liu, C., Feng, X., Zhao, Y., Ng, S.-K., Luu, A. T., He, J., Koh, P. W., and Hooi, B. Uncertainty of thoughts: Uncertainty-aware planning enhances information seeking in large language models. arXiv preprint arXiv:2402.03271, 2024.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Kobalczyk, K., Astorga, N., Liu, T., and van der Schaar, M. Active task disambiguation with llms. arXiv preprint arXiv:2502.04485, 2025.

Li, B., Sun, X., Liu, J., Wang, Z., Wu, J., Yu, X., Chen, H., Barsoum, E., Chen, M., and Liu, Z. Latent visual reasoing. arXiv preprint arXiv:2509.24251, 2025a.

Li, Y., Tian, M., Lin, Z., Zhu, J., Zhu, D., Liu, H., Zhang, Y., Xiong, Z., and Zhao, X. Fine-grained evaluation of large vision-language models in autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9431–9442, 2025b.

Lindley, D. V. On a measure of the information provided by an experiment. The Annals of Mathematical Statistics, 27 (4):986–1005, 1956.

Liu, A., Wang, J., Li, H., Chen, X., Wang, J., Kaski, S., and Yang, M. Attaining humans desirable outcomes in human-ai interaction via structural causal games. arXiv preprint arXiv:2405.16588, 2024a.

Liu, A., Wang, J., Kaski, S., Wang, J., and Yang, M. A principle of targeted intervention for multi-agent reinforcement learning. arXiv preprint arXiv:2510.17697, 2025.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., Zhu, J., and Zhang,

L. Grounding dino: Marrying dino with grounded pretraining for open-set object detection, 2024b. URL https://arxiv.org/abs/2303.05499.

Ma, Z., Zhang, J., Liu, Z., Zhang, J., Tan, J., Shu, M., Niebles, J. C., Heinecke, S., Wang, H., Xiong, C., Krishna, R., and Savarese, S. LATTE: Learning to think with vision specialists. In Christodoulopoulos, C., Chakraborty, T., Rose, C., and Peng, V. (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 11203–11240, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979- 8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main. 564. URL https://aclanthology.org/2025. emnlp-main.564/.

MacKay, D. J. Information-based objective functions for active data selection. Neural computation, 4(4):590–604, 1992.

Najemnik, J. and Geisler, W. S. Optimal eye movement strategies in visual search. Nature, 434(7031):387–391, 2005.

Niu, J., Liu, Z., Gu, Z., Wang, B., Ouyang, L., Zhao, Z., Chu, T., He, T., Wu, F., Zhang, Q., Jin, Z., Liang, G., Zhang, R., Zhang, W., Qu, Y., Ren, Z., Sun, Y., Zheng, Y., Ma, D., Tang, Z., Niu, B., Miao, Z., Dong, H., Qian, S., Zhang, J., Chen, J., Wang, F., Zhao, X., Wei, L., Li, W., Wang, S., Xu, R., Cao, Y., Chen, L., Wu, Q., Gu, H., Lu, L., Wang, K., Lin, D., Shen, G., Zhou, X., Zhang, L., Zang, Y., Dong, X., Wang, J., Zhang, B., Bai, L., Chu, P., Li, W., Wu, J., Wu, L., Li, Z., Wang, G., Tu, Z., Xu, C., Chen, K., Qiao, Y., Zhou, B., Lin, D., Zhang, W., and He, C. Mineru2.5: A decoupled vision-language model for efficient high-resolution document parsing, 2025. URL https://arxiv.org/abs/2509.22186.

Pirolli, P. and Card, S. Information foraging. Psychological review, 106(4):643, 1999.

Rainforth, T., Foster, A., Ivanova, D. R., and Bickford Smith, F. Modern bayesian experimental design. Statistical Science, 39(1):100–114, 2024.

Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Radle, R., Rolland, C., Gustafson, L., Mintun,¨ E., Pan, J., Alwala, K. V., Carion, N., Wu, C.-Y., Girshick, R., Dollar, P., and Feichtenhofer, C. Sam 2: Segment´ anything in images and videos, 2024. URL https: //arxiv.org/abs/2408.00714.

Shannon, C. E. Communication in the presence of noise. Proceedings of the IRE, 37:10–21, 1949. URL https://api.semanticscholar. org/CorpusID:12037187.

Singh, A., Natarjan, V., Shah, M., Jiang, Y., Chen, X., Parikh, D., and Rohrbach, M. Towards vqa models that can read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8317–8326, 2019.

Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm testtime compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Su, Z., Li, L., Song, M., Hao, Y., Yang, Z., Zhang, J., Chen, G., Gu, J., Li, J., Qu, X., et al. Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617, 2025a.

Su, Z., Xia, P., Guo, H., Liu, Z., Ma, Y., Qu, X., Liu, J., Li, Y., Zeng, K., Yang, Z., et al. Thinking with images for multimodal reasoing: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025b.

Sun, G., Hua, H., Wang, J., Luo, J., Dianat, S., Rabbani, M., Rao, R., and Tao, Z. Latent chain-of-thought for visual reasoing. arXiv preprint arXiv:2510.23925, 2025.

Team, D. Hr-bench: The high-resolution benchmark for multimodal llms. https://github.com/DreamMr/ HR-Bench, 2024. GitHub Repository.

Tong, S., Brown, E., Wu, P., Woo, S., Middepogu, M., Akula, S. C., Yang, J., Yang, S., Iyer, A., Pan, X., Wang, A., Fergus, R., LeCun, Y., and Xie, S. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024.

Wang, J., Fang, M., Wan, Z., Wen, M., Zhu, J., Liu, A., Gong, Z., Song, Y., Chen, L., Ni, L. M., et al. Openr: An open source framework for advanced reasoing with large language models. arXiv preprint arXiv:2410.09671, 2024.

Wang, W., Jing, Y., Ding, L., Wang, Y., Shen, L., Luo, Y., Du, B., and Tao, D. Retrieval-augmented perception: High-resolution image perception meets visual rag. arXiv preprint arXiv:2503.01222, 2025.

Wu, P. and Xie, S. V\*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 2023.

Xiao, L., Yang, X., Lan, X., Wang, Y., and Xu, C. Towards visual grounding: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., and Zhao, H. Depth anything: Unleashing the power of large-scale unlabeled data, 2024. URL https://arxiv.org/ abs/2401.10891.

Yu, X., Wang, Z., Yang, L., Li, H., Liu, A., Xue, X., Wang, J., and Yang, M. Causal sufficiency and necessity improves chain-of-thought reasoing. Advances in Neural Information Processing Systems, 38:126109–126141, 2026.

Zhang, Q., Lyu, F., Sun, Z., Wang, L., Zhang, W., Guo, Z., Wang, Y., Muennighoff, N., King, I., Liu, X., and Ma, C. What, how, where, and how well? a survey on test-time scaling in large language models, 2025a. URL https://arxiv.org/abs/2503.24235.

Zhang, Y.-F., Zhang, H., Liang, H., Wang, M., et al. Mmerealworld: Could your multimodal llm challenge the real world? arXiv preprint arXiv:2408.13257, 2024.

Zhang, Y.-F., Lu, X., Yin, S., Fu, C., Chen, W., Hu, X., Wen, B., Jiang, K., Liu, C., Zhang, T., et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025b.

## A. Related Work

Active Vision. The paradigm of controlling sensor parameters to maximise information utility originates from the semina definition of active perception by Bajcsy (1988) and Aloimonos et al. (1988). Classical approaches focused primarily on geometric tasks, such as Next-Best-View (NBV) planning for 3D reconstruction (Connolly, 1985). In the era of foundation models, while the scope of active vision has expanded to semantic reasoing (Su et al., 2025b), modern VLMs largely remain passive observers, processing fixed-resolution images provided by humans (Dosovitskiy, 2020; Liu et al., 2023). We contend that for fine-grained reasoing, VLMs must reclaim the agency of active vision. We treat “zooming” or “cropping” not merely as image pre-processing, but as an active sensory policy analogous to biological foveation (Pirolli & Card, 1999), allowing the model to alleviate the effective-resolution limitations imposed by its fixed visual-token budget.

Bayesian Experimental Design in Foundation Models. Our framework is grounded in Bayesian optimal experimental design (BOED), a field established by Lindley (1956) to quantify the value of data via Expected Information Gain (EIG). While BOED has long been a staple in active learning for data selection (MacKay, 1992; Houlsby et al., 2011), its application to the inference process of foundation models is nascent. Recent works in NLP have begun to leverage BOED principles to address task ambiguity. Approaches such as Uncertainty of Thoughts (UoT) (Hu et al., 2024), Active Task Disambiguation (Kobalczyk et al., 2025), and BED-LLM (Choudhury et al., 2025) frame question-asking as an experimental design problem, shifting the cognitive load from implicit heuristics to explicit inference about the solution space. In this work, we extend this information-theoretic paradigm to the visual modality. We identify that for VLM agents, a primary source of epistemic uncertainty stems not from ambiguous prompts, but from the perceptual bandwidth bottleneck. Consequently, we adapt BOED to model the spatio-semantic search process, deriving a tractable strategy that optimises foveation actions—rather than textual questions—to resolve fine-grained visual ambiguities.

Visual Agents and Inference-Time Scaling. Early visual agents, such as VisProg (Gupta & Kembhavi, 2023) and LATTE (Ma et al., 2025), treated tool use as modular programming with static execution plans, lacking adaptability to dynamic perceptual uncertainty. More recent works focus on internalising policies via fine-tuning or Reinforcement Learning (RL), such as Thyme (Zhang et al., 2025b). While effective within their training distribution, these methods often function as black-box heuristics—they learn where to look via pattern matching, but lack an explicit model of why a region is informative. Our approach represents a shift towards a principled inference-time framework. By formalising active vision as an S-BOED problem, we decouple the decision-theoretic objective from the optimiser, using the coverage–resolution utility as a practical proxy for task-relevant information gain. This aligns with the emerging trend of inference-time scaling (Snel et al., 2024; Zhang et al., 2025a; Wang et al., 2024), and allows future research to plug in diverse optimisers—from greedy selection (Golovin & Krause, 2011) to MCMC sampling (Foster et al., 2019)—for active visual evidence acquisition.

## A.1. Comparison with Discrete BOED for LLMs

Our work is theoretically grounded in Bayesian optimal experimental design (BOED), similar to recent advances in LLMs such as Kobalczyk et al. (2025). However, applying BOED to visual reasoing introduces unique challenges that distinguish our S-BOED framework from text-based active disambiguation.

Kobalczyk et al. (2025) address semantic ambiguity in user prompts (e.g., “Write a code for X” where X is vague). Their agent selects discrete questions to partition the hypothesis space. In contrast, our work addresses perceptual ambiguity caused by sensor bandwidth. Our agent must select continuous spatial parameters to physically resolve the target. This shift from discrete question selection to continuous spatial foraging necessitates fundamentally different assumptions and objectives, summarised in Table 2.

Table 2. Comparison between discrete active task disambiguation and our S-BOED-guided visual evidence acquisition.
<table><tr><td>Aspect</td><td>Kobalczyk et al. (2025)</td><td>FOVEA (Ours)</td></tr><tr><td rowspan="2">Problem Domain</td><td>Semantic Disambiguation</td><td>Perceptual Foraging</td></tr><tr><td>Resolving vague user intent in text genera- tion (e.g., code, 20 Questions).</td><td>Resolving fine-grained visual features un- der bandwidth constraints.</td></tr><tr><td rowspan="2">Action Space</td><td>Discrete &amp; Enumerable</td><td>Continuous &amp; High-Dimensional</td></tr><tr><td>Selection from a finite set of generated can- didate questions  $( q \in \mathcal { Q } )$ </td><td>Optimisation of spatial crop parameters in continuous coordinates  $( \mathbf { d } \in [ 0 , 1 ] ^ { 4 } )$ </td></tr><tr><td rowspan="2">Belief State</td><td>Explicit Posterior</td><td>Implicit Context</td></tr><tr><td>Estimated via explicit sampling of N hypo- thetical solutions  $\{ h _ { i } \}$ </td><td>Managed via the VLM&#x27;s attention mecha- nism over the interaction history  $\mathcal { H } _ { t } .$ </td></tr><tr><td>Optimal Design</td><td>Space Partitioning The optimal action bisects the solution space (e.g., a binary question with 50/50 split).</td><td>Search &amp; Resolution The optimal action maximises the joint probability of coverage and resolution.</td></tr><tr><td>Information Dynamics</td><td>Submodular (Diminishing Returns) Every relevant question reduces entropy. Greedy strategies usually suffice.</td><td>Super-additive (Information Cliff) A crop that misses the target yields zero se- mantic info. Requires look-ahead planning.</td></tr><tr><td>Constraint</td><td>Oracle Cost Minimising the number of questions asked to the user.</td><td>Perceptual Bandwidth Minimising the loss of visual information due to token limits.</td></tr></table>

## B. Notation

We align our notation with the standard Bayesian optimal experimental design (BOED) literature while introducing specific terms for the active visual reasoing setting. Table 3 summarises the primary symbols used throughout the paper. FOVEA-Greedy and FOVEA-MCMC are both myopic variants that optimise the immediate crop-utility estimator $\hat { \mathcal { I } } ( \mathbf { d } )$ ; they differ only in how candidate crops are generated. FOVEA-Lookahead is the non-myopic variant, using an estimated future value V<sup>ˆ</sup> .

Active Visual Reasoing under Perceptual Bandwidth
<table><tr><td colspan="3">Table 3. Summary of notation.</td></tr><tr><td>Symbol</td><td>Description</td><td>Domain / Definition</td></tr><tr><td colspan="3">Probabilistic Model &amp; Generative Process</td></tr><tr><td>θ</td><td>Latent world state parameters</td><td> $\pmb { \theta } = \{ \ell , y \} \in \Theta$ </td></tr><tr><td> $\ell$ </td><td>Latent spatial location of the target</td><td> $\boldsymbol { \ell } \in \Omega \subset \mathbb { R } ^ { 2 }$ </td></tr><tr><td> $y$ </td><td>Semantic target / answer variable</td><td> $y \in \mathcal { V }$ </td></tr><tr><td> $s$ </td><td>Visibility event / latent gating variable</td><td> $S \in \{ 0 , 1 \} ( \mathrm { E q . } 2 )$ </td></tr><tr><td> $\mathbf { z } _ { t }$ </td><td>Observation at step t</td><td> $\mathbf { z } _ { t } \in { \mathcal { Z } }$ </td></tr><tr><td> $p _ { t } ( \ell )$ </td><td>Spatial belief state at step t</td><td> $p ( \ell \mid \mathcal { H } _ { t - 1 } )$ </td></tr><tr><td> $p _ { t } ( y )$ </td><td>Semantic belief state at step t</td><td> $p ( y \mid \mathcal { H } _ { t - 1 } )$ </td></tr><tr><td colspan="3">Active Vision Setup</td></tr><tr><td> $\mathcal { D }$ </td><td>Design space / foveation action space</td><td> $[ 0 , 1 ] ^ { 4 }$  crop parameters</td></tr><tr><td> $\mathbf { d } _ { t }$ </td><td>Selected foveation design at step t</td><td> $\mathbf { d } _ { t } = [ u , v , w , h ] \in \mathcal { D }$ </td></tr><tr><td> $I _ { \mathbf { d } }$ </td><td>Crop induced by design d</td><td>Image region specified by d</td></tr><tr><td> $r _ { t }$ </td><td>Resolvability signal</td><td> $r _ { t } \in \{ 0 , 1 \}$ </td></tr><tr><td> $\mathcal { H } _ { t }$ </td><td>Multimodal interaction history</td><td> $\{ ( I _ { \mathbf { d } _ { \tau } } , \mathbf { d } _ { \tau } , \hat { \mathcal { I } } ( \mathbf { d } _ { \tau } ) , \mathbf { z } _ { \tau } ) \} _ { \tau = 1 } ^ { t }$ </td></tr><tr><td colspan="3">Physical Constraints</td></tr><tr><td> $\boldsymbol { B }$ </td><td>Perceptual bandwidth budget</td><td>Fixed visual-token / encoder capacity</td></tr><tr><td> $A ( \mathbf { d } )$ </td><td>Area of foveation crop</td><td>Normalised crop area</td></tr><tr><td> $\rho ( \mathbf { d } )$ </td><td>Effective perceptual density</td><td> $\rho ( \mathbf { d } ) = B / A ( \mathbf { d } )$ </td></tr><tr><td> $\phi ( \mathbf { d } )$ </td><td>Resolution probability</td><td> $P ( \mathrm { R e s o l v e d } \mid \mathbf { d } )$ </td></tr><tr><td colspan="3">Objectives &amp; Optimisation</td></tr><tr><td> $\mathrm { E I G } _ { t } ^ { \mathrm { f u l l } } ( \mathbf { d } )$ </td><td>Full Expected Information Gain</td><td> $\mathcal { T } _ { t } ( \mathbf { z } _ { t } ; \pmb { \theta } \mid \mathbf { d } )$ </td></tr><tr><td> $U _ { t } ( \mathbf { d } )$ </td><td>Task-relevant information gain</td><td> $\mathcal { T } _ { t } ( y ; \mathbf { z } _ { t } \mid \mathbf { d } )$ </td></tr><tr><td> $\mathcal { T } _ { t } ( \mathbf { d } )$ </td><td>Coverage-Resolution objective</td><td>Theoretical proxy (Eq. 8)</td></tr><tr><td> $\hat { \mathcal { I } } ( \mathbf { d } )$ </td><td>Empirical crop-utility estimator</td><td> $P ( r = 1 \mid I _ { \bf d } , Q ) ( \mathrm { E q . ~ } 1 0 )$ </td></tr><tr><td> $V ^ { * } ( { \mathcal { H } } )$ </td><td>Optimal sequential value function</td><td>Bellman Eq. 4</td></tr><tr><td> $\hat { V } ( \mathbf { d } , \mathcal { H } )$ </td><td>Estimated look-ahead value</td><td>Future resolvability estimate</td></tr><tr><td colspan="3">Practical Instantiation</td></tr><tr><td>FOVEA-Greedy</td><td>Greedy local perturbation variant</td><td>Samples local candidates and selects arg maxd  $\hat { \mathcal { I } } ( \mathbf { d } )$ </td></tr><tr><td>FOVEA-MCMC</td><td>MCMC-style local search variant</td><td>Uses adaptive proposals to search for high-  $\cdot \hat { \mathcal { I } }$  crops</td></tr><tr><td>FOVEA-Lookahead</td><td>One-step look-ahead variant</td><td>Selects crops by estimated future value  $\hat { V } ( \mathbf { d } , \mathcal { H } )$ </td></tr></table>

## C. Algorithms

Algorithm 2 FOVEA-Greedy: Local Predictive Sampling   
1: Input: Global view $I _ { \mathrm { g l o b a l } } .$ , User query Q, Token budget B   
2: Initialise: $\mathbf { d } _ { \mathrm { p r o p } }  \mathrm { V L M } ( I _ { \mathrm { g l o b a l } } , Q ) / /$ Initial region proposal   
3: Perturbation: Generate candidates $\mathcal { D } _ { \mathrm { c a n d } } = \{ \mathbf { d } _ { \mathrm { p r o p } } , \mathbf { d } _ { \mathrm { s m a l l } } , \mathbf { d } _ { \mathrm { l a r g e } } \}$   
where ${ \bf d } _ { \mathrm { s m a l l } } = 0 . 8 \times { \bf d } _ { \mathrm { p r o p } }$ (high resolution), ${ \bf d } _ { \mathrm { l a r g e } } = 1 . 5 \times { \bf d } _ { \mathrm { p r o p } }$ (high coverage).   
4: for each $\mathbf { d } _ { i } \in \mathcal { D } _ { \mathrm { { c a n d } } }$ do   
5: Extract crop $I _ { \mathbf { d } _ { i } }$   
6: $S _ { i } \gets P ( r = 1 \mid I _ { \mathbf { d } _ { i } } , Q ) / /$ Estimate crop utility   
7: end for   
8: Execution: $\mathbf { d } ^ { * } = \arg \operatorname* { m a x } _ { \mathbf { d } _ { i } } S _ { i }$   
9: Return: d<sup>∗</sup> to execute tool call

Algorithm 3 FOVEA-MCMC: Adaptive Crop Refinement   
1: Input: Global view $I _ { \mathrm { g l o b a l } } .$ , Query Q, Iterations N   
2: Utility Function: $\hat { \mathcal { I } } ( \mathbf { d } ) = P ( r = 1 \mid I _ { \mathbf { d } } , Q )$   
3: Initialise: $\mathbf { d } ^ { ( 0 ) } \gets$ Random or VLM Proposal   
4: for t = 1 to $N$ do   
5: $\tilde { \mathbf { d } } \sim q ( \cdot \mid \mathbf { d } ^ { ( t - 1 ) } ) \mathrm { ~ / ~ / ~ }$ Propose new region via Gaussian perturbation   
6: α = min $\begin{array} { r } { \left( 1 , \frac { { \hat { \mathcal { I } } ( \tilde { d } ) } + \epsilon } { \hat { \mathcal { I } } ( d ^ { ( t - 1 ) } ) + \epsilon } \right) } \end{array}$ . // Acceptance probability   
7: u ∼ Uniform(0, 1)   
8: if u < α then   
9: $\mathbf { d } ^ { ( t ) }  \mathbf { \tilde { d } } / /$ Accept move to higher information density   
10: else   
11: $\mathbf { d } ^ { ( t ) }  \mathbf { d } ^ { ( t - 1 ) } / /$ Reject move   
12: end if   
13: end for   
14: Burn-in & Selection: d<sup>∗</sup> = Mode or Best of $\{ \mathbf { d } ^ { ( t ) } \} _ { t = M } ^ { N }$   
15: Return: d<sup>∗</sup>

Algorithm 4 FOVEA-Lookahead: One-Step Planning   
1: Input: Global view $I _ { \mathrm { g l o b a l } } .$ , Query Q, Proposal $\mathbf { d } _ { \mathrm { { p r o p } } }$ , Scaling factors $s$   
2: Initialise: ${ \mathcal { D } } _ { \mathrm { r o o t } } \gets \mathrm { P e r t u r b } ( \mathbf { d } _ { \mathrm { p r o p } } , S )$   
3: for each $\mathbf { d } _ { i } \in \mathcal { D } _ { \mathrm { r o o t } }$ do   
4: $/ /$ Simulation (Expansion):   
5: $I _ { \mathrm { s i m } } \gets \mathbf { C r o p } ( I _ { \mathrm { g l o b a l } } , \mathbf { d } _ { i } )$   
6: $\mathbf { d } _ { \mathrm { n e x t } }  \pi _ { \mathrm { V L M } } ( { \bar { I } } _ { \mathrm { s i m } } , Q ) / /$ Predict agent’s next move given $\mathbf { d } _ { i }$   
7: $/ /$ Evaluation (Future Resolvability):   
8: if $\mathbf { d } _ { \mathrm { n e x t } }$ is valid then   
9: ${ \mathcal { D } } _ { \mathrm { l e a f } }  \mathrm { P e r t u r b } ( \mathbf { d } _ { \mathrm { n e x t } } , S )$   
10: $S _ { i } \gets \mathbf { M e a n _ { d ^ { \prime } \in \mathcal { D } _ { \mathrm { l e a f } } } } P ( r = 1 \mid I _ { \mathbf { d } ^ { \prime } } , Q ) \mathbf { \Omega } / /$ Average utility of future leaves   
11: else   
12: $S _ { i } \gets 0$   
13: end if   
14: end for   
15: Execution: $\mathbf { d } ^ { * } = \arg \operatorname* { m a x } _ { \mathbf { d } _ { i } } S _ { i }$   
16: Return: d<sup>∗</sup>

## D. Implementation Details

In this section, we detail FOVEA, the practical training-free instantiation of the S-BOED framework. We describe the agent architecture, the empirical crop-utility estimator, and the concrete search variants used in our experiments.

## D.1. Agent Architecture

Our system is built upon the ReAct agent framework. The core reasoing backbone is Qwen3-VL-30B-A3B-Instruct, a state-of-the-art multimodal model. The agent operates in a sequential loop: at each time step t, it receives the current visual observation and the interaction history. It then generates a structured “thought” trace followed by a specific tool invocation. The agent has access to four vision tools: cropping, detection, segmentation, and depth estimation. In the end, the agent aggregates both the ReAct trajectory and its direct reasoing to synthesise a final analysis and provide the answer. The system prompt is listed in Appendix J.

Tool Specifications. We employ specific state-of-the-art backbones for the vision tools to ensure robust perception. For open-vocabulary object detection, we utilise Grounding DINO (Liu et al., 2024b); it accepts an image and a text prompt as inputs to output bounding box coordinates, and we draw these bounding boxes on the input image which will be returned to the agent with the coordinates. Segmentation is handled by SAM 2 (Ravi et al., 2024), which takes an image and prompt cues to generate high-quality pixel masks. We use different colours to mask the input image and return it to the agent. Depth estimation relies on Depth Anything (Yang et al., 2024), mapping a single RGB image to a relative depth map for spatial understanding. The agent will receive a heat map representing the depth along with the original image. Finally for OCR, we leverage MinerU (Niu et al., 2025), which processes an image region to extract and return structured text content.

Tool Interception Mechanism. To implement active foraging without retraining the backbone model, we use a tool interception mechanism. The agent is provided with a standard crop tool definition, but when it invokes the tool with proposed coordinates $\mathbf { d } _ { \mathrm { p r o p } } .$ , the FOVEA module intercepts this call before execution. Instead of directly executing the raw proposal, FOVEA treats $\mathbf { d } _ { \mathrm { p r o p } }$ as a noisy spatial prior, refines it with one of the search strategies, and executes the refined crop $\mathbf { d } ^ { * }$ to return a more informative observation to the agent.

## D.2. Probabilistic Objective Realisation

To evaluate the coverage–resolution objective in a tractable manner, we employ the VLM itself as a stochastic evaluator of crop utility.

The Empirical Estimator. We approximate the resolution probability ϕ(d) and the visibility event S by constructing a “resolvability verification” task. For a candidate design d, we extract the corresponding image region $I _ { \mathbf { d } }$ and prompt the VLM with the user query Q alongside a specific system instruction (see Appendix J). The model is constrained to output a binary “Yes” or “No” indicating whether the region contains sufficient information to resolve the query.

Monte Carlo Estimation. Since the VLM output is stochastic, we perform Monte Carlo estimation to smooth the objective surface. For every candidate design, we sample the model’s response N times. The estimator $\hat { \mathcal { I } } ( \mathbf { d } )$ is calculated as the expectation of the affirmative response. This score serves as an empirical crop-utility surrogate for the coverage–resolution objective, guiding the selection of the refined crop.

## D.3. Search Strategy Implementation

We implement three FOVEA variants corresponding to different search budgets.

Greedy Perturbation Strategy. Algorithm 2 implements the computationally efficient FOVEA-Greedy variant. Instead of sampling from the entire image space, we generate a local search space $\mathcal { D } _ { \mathrm { l o c a l } }$ around the agent’s initial proposal $\mathbf { d } _ { \mathrm { p r o p } }$ by applying a set of fixed scaling factors $\mathcal { C } = \{ 1 . 5 , 1 . 0 , 0 . 8 \}$ to the proposed bounding box. The high-coverage variant (factor 1.5) expands the field of view to capture context that might have been narrowly missed by the initial proposal. The high-resolution variant (factor 0.8) zooms in further to maximise feature density, trading off spatial coverage. The candidate with the highest estimated score $\hat { \mathcal { I } } ( \mathbf { d } )$ is selected for execution.

MCMC-based Adaptive Sampling. Algorithm 3 implements the FOVEA-MCMC variant that treats the agent’s initial proposal $\mathbf { d } _ { \mathrm { p r o p } }$ as the starting state $X _ { 0 }$ of a Markov chain. We employ a Metropolis-Hastings framework with an adaptive step size to balance exploration and exploitation. For each iteration, we generate a candidate d<sup>′</sup> using a Gaussian perturbation. The step size is dynamically scaled based on the current crop’s dimensions (setting σ to 15% of the width/height), allowing the agent to perform coarse global shifts or fine local adjustments. We approximate the utility $U ( \mathbf { d } )$ using Eq. (10). A transition to d<sup>′</sup> is accepted if $\hat { \mathcal { I } } ( \mathbf { d } ^ { \prime } ) \geq \hat { \mathcal { I } } ( \mathbf { d } )$ or with a small probability (0.1) to prevent stagnation in local optima. We set a maximum of 6 iterations and utilise 3 stochastic probings per candidate. The process terminates early if a state achieves $\hat { \mathcal { I } } ( \mathbf { d } ) = 1 . 0$

Look-Ahead Planning Strategy. Algorithm 4 implements the FOVEA-Lookahead variant, a one-step planner motivated by the Bellman update. For each candidate d in the local search space, the system simulates the crop and prompts the VLM to predict the subsequent action it would take given that observation. If the predicted next action is a further refinemen (a sub-crop for example), we generate a hypothetical leaf node whose resolvability score will be evaluated. The value of the current candidate $\mathbf { d } _ { i }$ is updated to reflect the expected gain of the future state, thereby favouring actions that lead to high-information states even if they do not immediately resolve the query.

## D.4. Hyperparameters

Table 4 lists the hyperparameters used across all experiments.

Table 4. Hyperparameters for FOVEA Inference
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Backbone Model</td><td>Qwen3-VL-30B-A3B-Instruct</td></tr><tr><td>Max Interaction Turns</td><td>10</td></tr><tr><td>Monte Carlo Samples</td><td>3</td></tr><tr><td>Perturbation Scaling Factors</td><td>{1.5, 1.0, 0.8}</td></tr></table>

## D.5. Inference Cost

Table 5 presents a cost-benefit analysis of the FOVEA variants. Active foraging introduces additional inference-time cost compared with the ReAct baseline, but the results reveal a clear compute–accuracy frontier. FOVEA-Greedy and FOVEA-MCMC incur moderate increases in input and output tokens due to candidate sampling and resolvability probing. FOVEA-Lookahead requires substantially more output tokens because it generates hypothetical future trajectories, but yields the largest accuracy gain. These results show that in search-dominated gigapixel regimes, additional compute for active perceptual planning can translate into meaningful accuracy improvements.

Table 5. Inference cost of search strategies compared to ReAct.
<table><tr><td>Search Policy</td><td>Accuracy (%)</td><td>Accuracy Gain</td><td>Avg. Input Tokens / Query</td><td>Avg. Output Tokens / Query</td></tr><tr><td>ReAct (Baseline)</td><td>45.1</td><td></td><td>46.5k (1×)</td><td>0.3k (1×)</td></tr><tr><td>FOVEA-Greedy</td><td>47.6</td><td>5.54%</td><td>301.9k (6.5×)</td><td>3.1k (9.8×)</td></tr><tr><td>FOVEA-MCMC</td><td>51.4</td><td>13.97%</td><td>359.3k (7.7×)</td><td>4.0k (12.5×)</td></tr><tr><td>FOVEA-Lookahead</td><td>54.7</td><td>21.29%</td><td>441.4k (9.5×)</td><td>17.5k (55.2×)</td></tr></table>

## E. Empirical Evidence for History-Based Belief Calibration

In Section 4, we posit that the VLM can use interaction history to revise its search preference over candidate regions. This appendix provides empirical evidence for this history-based belief calibration, detailing the experimental setup, metric formulation, and quantitative results of a counterfactual intervention study.

![](images/c4b98488428cc518ada96689dc18abd182dc85a3dd1a404907a24fdacda41024.jpg)  
Figure 5. Bayesian belief update analysis across $N = 1 5 0$ samples. (a) Verification Test: Given positive evidence, the model concentrates probability mass on the target. (b) Falsification Test: Given negative evidence, the model prunes the viewed region (red bar collapses), shifting belief back to the valid search space. Error bars denote 95% CI.

## E.1. Experimental Setup

To rigorously test the model’s ability to perform belief update, specifically verification (exploiting positive evidence) and falsification (pruning the search space given negative evidence), we utilised the 150 challenging visual search instances of the Remote Sensing subset of MME-RealWorld-lite. For each instance, we manually annotated:

1. Ground Truth Grids $( G _ { G T } ) \ d t$ The set of $3 \times 3$ grid indices containing the target.

2. Oracle Crop $( C _ { p o s } ) \colon$ A high-resolution crop perfectly centring the target object.

3. Distractor Crop $( C _ { n e g } ) \colon$ A high-resolution crop of an irrelevant region.

We first queried the VLM with the global image to obtain a prior distribution over the nine grid regions. We then intervened by providing either the Oracle Crop (positive evidence) or the Distractor Crop (negative evidence) and queried the VLM for its posterior decision.

## E.2. Metric Formulation

To address quantisation artefacts where ground truth objects span grid boundaries, we aggregate probabilities rather than relying on single-label accuracy. We track two dynamic variables across the belief update process:

1. Target Probability $( P _ { t a r g e t } ) ;$ The total probability mass assigned to the true location of the object.

$$
P _ { t a r g e t } = \sum _ { i \in G _ { G T } } P ( \mathrm { G r i d } _ { i } )
$$

2. Viewed Region Probability $( P _ { v i e w e d } ) \mathrm { : }$ The total probability mass assigned to the region currently visible in the zoomed-in crop.

$$
P _ { v i e w e d } = \sum _ { j \in G _ { o v e r l a p } } P ( \mathrm { G r i d } _ { i } )
$$

where $G _ { o v e r l a p }$ denotes the grid cells covered by the current intervention crop to a certain extent (15% of the grid area in our experiments).

## E.3. Quantitative Analysis

Figure 5 illustrates the aggregated results of the belief update experiment.

Verification Test (Positive Evidence). As shown in Figure 5 (a), providing the Oracle Crop triggers an obvious reduction in entropy. The VLM successfully identifies the evidence as sufficient, causing both $P _ { t a r g e t }$ and $P _ { v i e w e d }$ to converge towards 1. This confirms the model’s capacity for exploitation, that is, when presented with the correct view, it confidently locks onto the target.

Falsification Test (Negative Evidence). Figure 5 (b) demonstrates the model’s capacity for exploration and error correction. In the Prior state, the model assigns non-zero probability to the distractor region $( P _ { v i e w e d } > 0 )$ . However, upon observing the detailed Distractor Crop, the probability assigned to this region collapses $( P _ { v i e w e d } \approx 0 )$ . Crucially, the probability mass is not lost but is redistributed to the remaining search space, increasing the confidence in the true target $P _ { t a r g e t }$

This behaviour—rejecting false evidence and reallocating probability mass to alternative regions—is consistent with the Bayesian-update view: the model can use positive and negative evidence in the interaction history to revise its search preference, rather than merely repeating its initial proposal.

## F. Probe Validity and Selector Ablation

A core component of our framework is the empirical utility estimator $\hat { \mathcal { I } } ( \mathbf { d } ) = P ( r = 1 \mid I _ { \mathbf { d } } , Q )$ defined in Eq. 10, which uses the VLM’s Yes/No response on a candidate crop as a surrogate for crop utility. A natural concern, raised during review, is whether this signal genuinely tracks crop usefulness for the downstream task or merely reflects generic answerability confidence. This appendix presents two controlled diagnostics that address this concern. Both studies use the same 50 annotated remote-sensing position-reasoing examples from MME-RealWorld-Lite (Zhang et al., 2024).

## F.1. Probe Validity: Oracle vs. Distractor vs. Random Crops

Setup. For each example we compare three crop types: an oracle crop drawn from the human annotation that contains the answer-relevant region; a distractor crop drawn from a visually plausible but task-irrelevant region; and a random crop sampled uniformly from the image. For each crop we record (i) the Yes/No probe score J<sup>ˆ</sup>(d), (ii) whether the crop centre falls inside the annotated oracle region (hit centre), and (iii) the downstream QA accuracy when the model answers using the crop together with its normalised crop coordinates.

Results. Table 6 summarises the results. Oracle crops receive substantially higher probe scores than either distractor or random crops (0.633 vs. 0.187 / 0.187), and they also yield much higher downstream QA accuracy (52.0% vs. 10.0% / 12.0%). The effect size between oracle and distractor crops is large (Cohen’s d = 1.22), indicating strong separation rather than a marginal trend. The probe score is also positively rank-correlated with both spatial grounding (Spearman $\rho = 0 . 5 3 8$ with hit centre) and downstream correctness $( \rho = 0 . 3 9 2$ with QA accuracy).

Table 6. Probe-score validity on annotated remote-sensing position-reasoing examples. Oracle crops receive substantially higher Yes/No probe scores than distractor or random crops, and they also yield much higher answer accuracy when the model answers using the crop together with its normalised crop coordinates. The large effect size (Cohen’s d = 1.22) indicates strong separation between oracle and distractor crops. The positive rank correlations $( \rho = 0 . 5 3 8$ with hit centre and $\rho = 0 . 3 9 2$ with QA accuracy) indicate tha higher proxy scores are associated with both better spatial grounding and better downstream task performance. hit centre denotes that the centre of the candidate crop falls inside the human-annotated oracle region.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Oracle proxy score</td><td> $0 . 6 3 3 \pm 0 . 4 2 7$ </td></tr><tr><td>Distractor proxy score</td><td> $0 . 1 8 7 \pm 0 . 3 1 0$ </td></tr><tr><td>Random proxy score</td><td> $0 . 1 8 7 \pm 0 . 3 5 1$ </td></tr><tr><td>Oracle QA accuracy Distractor QA accuracy</td><td>52.0% 10.0%</td></tr><tr><td>Random QA accuracy</td><td>12.0%</td></tr><tr><td>Cohen&#x27;s d (oracle vs. distractor)</td><td>1.22</td></tr><tr><td>ρ(score, hit_centre)</td><td>0.538</td></tr><tr><td>ρ(score, QA accuracy)</td><td>0.392</td></tr></table>

Interpretation. These results indicate that the Yes/No probe is not behaving as a generic “can the model produce an answer” confidence signal: if it were, oracle and distractor crops would receive similar scores, since the model can fluently produce some answer in either case. Instead, the probe assigns markedly higher scores to crops that actually contain the answer-relevant evidence, and these higher scores translate into better downstream QA accuracy. We therefore use the probe as an empirical surrogate for crop utility, while explicitly noting that it is not an exact estimator of information gain.

## F.2. Selector Ablation: Probe vs. VLM-Direct vs. Random

Setup. The previous study evaluates probe scores on individual crops in isolation. To test whether the probe is also a better selector than alternative strategies under a fixed candidate pool, we construct a controlled pool by partitioning each image into a $3 \times 3$ grid of equal cells and compare three ways of selecting one cell: (1) Probe, which scores each cell independently with J<sup>ˆ</sup>(d) and picks the highest-scoring cell; (2) VLM-direct, which presents all nine cells jointly to the VLM and asks it to pick the single best one in a single forward pass; (3) Random, which selects a cell uniformly. All three selectors operate on the same fixed pool, isolating the effect of the selection mechanism from candidate generation.

We evaluate three metrics on the selected cell: downstream QA accuracy, hit centre (whether the selected cell’s centre falls in the oracle region), and IoU between the selected cell and the oracle region.

Results. Table 7 reports the outcome. Probe-based selection outperforms both alternatives on all three metrics. In particular, Probe more than doubles the QA accuracy of VLM-direct selection (30% vs. 20%) and substantially improves localisation (hit centre 52% vs. 34%; IoU 0.260 vs. 0.188).

Table 7. Selector ablation on a fixed 3 × 3 candidate pool (50 remote-sensing position-reasoing examples). For each image, we partition the full image into 9 equal grid cells and compare three ways of selecting one candidate cell: (1) Probe, which scores each cell independently with the Yes/No crop-utility question; (2) VLM-direct, which asks the model to choose the single best region among all 9 cells jointly; and (3) Random. Probe outperforms both alternatives in downstream QA accuracy, while also improving localisation quality (hit centre) and average overlap (IoU).
<table><tr><td>Selector</td><td>QA Acc. (%)</td><td>Hit Centre (%)</td><td>IoU</td></tr><tr><td>Random</td><td>10</td><td>12</td><td>0.061</td></tr><tr><td>VLM-direct</td><td>20</td><td>34</td><td>0.188</td></tr><tr><td>Probe</td><td>30</td><td>52</td><td>0.260</td></tr></table>

Interpretation. Two observations are worth noting. First, the gap between Probe and VLM-direct shows that the gain does not come from candidate generation (the pool is identical) but from how candidates are scored: independently probing each crop at high resolution recovers more useful local evidence than asking the VLM to compare nine downsampled cells in a single forward pass. This is consistent with the perceptual-bandwidth view in Section 2.2, since the joint comparison forces the encoder to compress all nine regions simultaneously. Second, Probe improves not only QA accuracy but also hit centre and IoU, indicating that the gain is grounded in better region selection rather than only in answer formatting.

## G. The Role of Intermediate Reasoing in Multi-Step Interaction

A natural concern with multi-step interaction is that the agent may accumulate language drift: by repeatedly generating intermediate conclusions and plans for the next view, the model could gradually commit to an incorrect hypothesis and reinforce it over subsequent steps. This appendix provides a simple ablation and a qualitative case showing that, on this class of questions, intermediate reasoing instead functions predominantly as an integration mechanism. Both studies use the same 50 remote-sensing position-reasoing examples introduced in Appendix F; absolute numbers in this appendix are therefore not directly comparable to the main-text Remote-Sensing results in Section 5.2, which use a different protocol on a different subset.

## G.1. Ablation: Evidence-Only Multi-Step Interaction

We compare the standard multi-step setting against an evidence-only variant in which all intermediate textual reasoing is removed from context, while the sequence of intermediate crops is kept identical. The two variants therefore receive the same visual information; the only difference is whether the model’s intermediate reasoing trace is preserved.

As shown in Table 8, removing intermediate reasoing lowers accuracy from 66.0% to 48.0%. Since the visual evidence is unchanged, this 18-point gap is attributable to the role of intermediate reasoing in integrating that evidence across views.

![](images/b4498ad87cd32686d7b2302e4a5f8befdadb48db02eec894e66fb6055b492041.jpg)

Table 8. Effect of removing intermediate textual reasoing in multi-step interaction. Both variants share the same sequence of intermediate crops on the same 50 remote-sensing questions.
<table><tr><td>Setting</td><td>QA Acc. (%)</td><td>Description</td></tr><tr><td>Full multi-step reasoing</td><td>66.0</td><td>Intermediate reasoing and crops</td></tr><tr><td>Evidence only</td><td>48.0</td><td>Intermediate crops only</td></tr></table>

## G.2. Qualitative Case

Figure 6 shows a representative example. Without intermediate exploration reasoing, the model commits to the wrong quadrant in a single shot (answer D). With sequential scanning and local verification, the same backbone recovers the correct quadrant (answer C), illustrating that the multi-step trace can revise an early hypothesis rather than lock it in.

Question: Where is the house with a yellow roof in the picture? (A) In the bottom left of the picture (B) In the bottom right of the picture (C) In the upper right of the picture (D) In the upper left of the picture (E) This image doesn't feature the position

![](images/99e535125cdc8b9ccfc2f85958dd167e572e20b708d2cd357e1f2e4d355cfe76.jpg)

![](images/ec1a08ac3652b95d2114a905b54281b14f1fbcfad7455ed0cb88ed5192617696.jpg)  
(No intermediate exploration reasoing provided)  
I'll start by scanning the entire picture... I'll begin by checking the upper left portion of the image...

![](images/c3465eddc486f845ffee366f6083feca878448b4fb85b92f6ca8dcbe8cdba4fd.jpg)  
... The house with the yellow roof is in the upper left of the picture  
I will now examine the upper right portion of the image...

![](images/471a24c5c60366d44d1382c198bcc4e02747eddac9849e9c51782a3a844388fe.jpg)  
Based on the scan, the most prominent yellow feature in the image is a large building complex located in the upper right section,

Figure 6. Sequential reasoing revising an early hypothesis. Without intermediate reasoing (left), the model predicts the wrong quadrant (D). With sequential scanning and local verification (right), it recovers the correct answer (C).

## H. Failure Analysis

As shown in Figure 3, while the FOVEA-Lookahead outperforms greedy baselines by simulating future belief states, a performance gap relative to the Oracle persists (54.7% vs. 68.0%). Since the oracle-crop baseline uses human-annotated crops that largely remove the search bottleneck, this gap isolates failures in search dynamics rather than recognition

capability. The remaining gap to 100% within the Oracle itself is attributable to a separate bottleneck, namely the backbone’s reasoing reliability, which we discuss in Appendix H.3.

This appendix examines the search-side failures qualitatively (Section H.1) and then quantifies the dominant mode through a controlled multi-seed diagnostic (Section H.2).

## H.1. Failure Modes

Qualitative analysis of the Look-ahead error cases reveals two primary failure modes where the active search strategy diverges from the optimal path.

![](images/accaa3e9a26d2043754b752c3718846005cdc49c446825179ccd0fa225d2c562.jpg)  
(a) Prior Misalignment (Cold Start)

![](images/f6f6158ceec2c2f3d0180a24fc97f1c2f5b50c6884a399a3ee13f1457883dafd.jpg)  
(b) Semantic Distractor Traps  
Figure 7. Qualitative failure analysis. (a) In the Cold Start scenario, the target (a white circular pattern) is imperceptible at the globa scale, leading to a near-zero initial probability mass in the correct region. (b) In the Distractor scenario, the task requires finding the “largest football field”. The agent exhausts its inference budget investigating visually similar fields (white box) before it can identify the true target (red box).

Cold Start (Prior Misalignment). The efficacy of the Look-ahead search is bounded by the quality of the initial belief distribution. As shown in Figure 7a, small targets such as the white circular pattern of the target green rectangular grassland are often completely imperceptible in the downsampled global view. Consequently, the VLM assigns a negligible probability mass to the target region during the initial scan. The FOVEA-Lookahead, relying on entropy to guide exploration, interprets these false-negative regions as “resolved background” and directs its budget towards more ambiguous but incorrect regions. Unlike the Oracle, which is initialised with ground truth coordinates, the planner cannot recover from this initial misalignment.

Semantic Distractor Traps. In dense remote-sensing imagery, the planner occasionally falls victim to semantic distractors: objects that share visual features with the target but do not satisfy the specific query constraints. Figure 7b illustrates this for the query “Find the largest football field”. The environment contains multiple candidate fields; the distractor (white box) is visually salient and generates high expected information gain. The agent, limited by a finite inference budget, expends its steps zooming into and verifying these smaller fields. By the time the ambiguity is resolved (determining the field is too small), the episode terminates before the agent can explore the true target location (red box).

These cases highlight that while Look-ahead search effectively handles local spatial reasoing, it remains susceptible to global priors and budget constraints that the Oracle does not face.

## H.2. Quantifying the Proposal Bottleneck

The cold-start failure mode in Section H.1 is fundamentally a proposal bottleneck: when the global view does not surface the target, the single seed proposal $\mathbf { d } _ { \mathrm { s e e d } }$ is misaligned, and local refinement around it cannot recover a region that never enters the candidate set. To quantify how much of the Oracle gap is attributable to this bottleneck, we run a controlled diagnostic on 50 remote-sensing position-reasoing examples comparing two initialisation regimes under the same downstream scoring rule:

• Single-seed (current default): one VLM proposal, refined into 3 local candidates.

• Multi-seed: 9 seed proposals, each refined into 3 local candidates, yielding a candidate pool of 27.

We decompose each failure into one of three categories. A failure is proposal-limited if the correct region never enters the candidate set; search-limited if the correct region is present in the candidate set but is not selected; and reasoing-limited if a target-relevant crop is selected but the backbone still answers incorrectly. This decomposition isolates whether errors are caused by the candidate generator, the selector, or the backbone.

Table 9. Failure decomposition under single-seed vs. multi-seed initialisation. Multi-seed improves accuracy from 50.0% to 54.0% and, more importantly, reduces proposal-limited failures from 25 to 7, indicating that broadening proposal support substantially mitigates the cold-start bottleneck. The remaining failures shift towards reasoing-limited cases, where the correct region is recovered but the backbone still answers incorrectly.
<table><tr><td>Metric</td><td>Single-seed</td><td>Multi-seed</td></tr><tr><td>Candidate pool size</td><td>3</td><td>27</td></tr><tr><td>Accuracy (%)</td><td>50.0</td><td>54.0</td></tr><tr><td>Correct</td><td>25</td><td>27</td></tr><tr><td>Proposal-limited failures</td><td>25</td><td>7</td></tr><tr><td>Search-limited failures</td><td>0</td><td>3</td></tr><tr><td>Reasoing-limited failures</td><td>0</td><td>13</td></tr></table>

Two observations follow. First, under the single-seed default, all 25 failures on this subset are proposal-limited: the correct region is not in the candidate pool to begin with, so no amount of better selection or planning can recover it. Second, expanding to nine seeds reduces proposal-limited failures from 25 to 7, and the remaining errors shift towards reasoinglimited cases (13 of 27 failures), where the correct region is recovered but the backbone still answers incorrectly. This is consistent with the Oracle saturation phenomenon discussed in Appendix H.3: once the search bottleneck is alleviated, residual errors are bounded by the backbone’s recognition reliability rather than by the search strategy. We therefore view broader proposal support and stronger search planning as complementary rather than substitutable, and a more adaptive proposal mechanism is a natural direction for future work.

## H.3. The Limits of the Ideal Observer Assumption

It is pertinent to address why the human-annotated Oracle saturates at an accuracy of 68.0% rather than reaching 100% despite utilising “golden crops” that guarantee target visibility. This performance ceiling highlights a practical deviation from Assumption 3.3 that is central to our theoretical derivation.

We assumed that successful foveation $( S = 1 )$ causes the conditional entropy to collapse to zero: $H ( y \mid \mathbf { z } , S = 1 , \mathbf { d } ) \approx 0$ However, in information-theoretic terms, this collapse implies certainty regarding the posterior distribution, but not necessarily correctness relative to the ground truth. The residual 32% error rate indicates that even when the perceptua bandwidth bottleneck is fully resolved, the agent remains constrained by the intrinsic reasoing capabilities of the underlying backbone model.

Figure 8 illustrates a representative failure case. The query requires counting light orange rectangular structures. Although the “golden crop” renders these features with high fidelity (clearly showing four distinct structures), the model confidently reasos: By carefully counting them, we can identify exactly three such structures... and outputs an incorrect answer. In these failure modes, the model effectively hallucinates a confident but incorrect answer $( y _ { \mathrm { p r e d } } \neq y _ { \mathrm { G T } } )$ despite high-fidelity observation. This distinction clarifies that while S-BOED-guided search improves the acquisition of visual evidence, final answering accuracy remains bounded by the recognition and reasoing limits of the underlying foundation model.

![](images/36ef735ba2d70f07ce3c7f617a4fd00ca988f524b8bc2ad65c6ea8672c11ff20.jpg)  
Figure 8. Oracle failure case. Even with a perfect “golden crop” that clearly resolves the target (four orange rectangular structures), the VLM hallucinates a count of “three”, demonstrating that visual resolvability does not guarantee reasoing correctness.

## I. Foraging Strategy Examples

Figure 9 contrasts the trajectories of the Look-ahead and Greedy foraging policies on the same query. The FOVEA-Lookahead explores multiple candidate regions before committing to a final crop, whereas the FOVEA-Greedy commits early and exhausts its budget on a sub-optimal region.

![](images/748a8ce553cd0691c4a87afe7cc58ada0fe65e1c55ff61989460724a90585ce8.jpg)  
(a) Look-ahead Foraging Strategy  
(b) Greedy Foraging Strategy  
Figure 9. Foraging Strategy Examples

## J. System Prompts

We utilise distinct system prompts for the reasoing agent and the resolvability evaluator to ensure role separation.

## Resolvability Evaluator

## Task

Evaluate whether a cropped region contains sufficient information to answer a question.

## Context

You will be shown three images (or two after deduplication):

1. The original image (provided with the question)

2. The image the agent wants to crop

3. A candidate cropped region

## Instructions

1. Analyze the correlation: What’s the relationship between these images? How does the candidate cropped region relate to the original question?

2. Identify mismatch: If the candidate cropped region is on a different part of the original image from where the question asks, this region contains no information and you should answer with “No”.

3. Be confident of your choice: Trust your reasoing and perception, based on which you can always give a “Yes” or “No” to whether this cropped region helps answer the question.

## Constraints

• MUST NOT answer anything other than “Yes” and “No”; you don’t need to answer the original question.

• MUST use the thinking section to reaso about the relationships between images and the question, and give a thorough analysis.

• ALWAYS output reasoing under \*\*Thinking:\*\* and “Yes” or “No” within <answer>.

## Output Format

<sub>\*\*</sub>Thinking:<sub>\*\*</sub>   
1. Analyze the correlation:   
2. Identify mismatch: ...   
3. Thorough analysis (Reaso about if you can answer the question correctly given the cropped region based on the   
previous thinking): ...   
<sub>\*\*</sub>Answer:<sub>\*\*</sub>   
<answer>[Your Answer]</answer>

## System Prompt for CV Agent

## Persona

You are an advanced Computer Vision (CV) Agent equipped with sophisticated image analysis tools and the ability to visually verify environmental data. You possess high-level reasoing capabilities to interpret complex visual scenes and resolve discrepancies between automated tool outputs and direct visual observation.

## Task

Your objective is to provide accurate answers to user questions regarding a provided image. You must always think before taking an action, utilizing specialized tools (e.g., cropping, detection) to gather data, while maintaining a critical perspective on tool reliability by prioritizing your own visual perception.

## Instructions

1. Turn Management: At the start of every response, check the current turn against max turns. If current turn == max turns, you must skip tool usage and provide the most accurate <answer> possible based on existing information.

2. Heavy Reasoing: Think first. Analyze the current state, the user’s question, and previous observations. Plan the next logical step.

3. Good Tool Use Strategy:

• “Crop first, perception (e.g., detection) later” is always a good exploration strategy in early rounds. It’s not wise to call detection on a quite large picture without zooming in first.

• When you want to zoom in by cropping, prefer a larger area to crop, rather than cropping precisely the area of interest. You are not doing grounding; you just want to zoom in so that you can see that area more clearly.

• When you want to zoom in further, always prefer cropping those which are already zoomed-in, as this allows more precise control and produces better results for you to perceive.

## 4. Handling Tool Discrepancies:

• If a tool returns a null or empty result, do not assume the object is absent.

• Compare tool results with your own visual analysis of the image.

• If a tool fails but you can see the object, describe what you see and use that visual evidence for your answer.

• If a tool fails and you cannot see the object, change your strategy (e.g., crop a sub-region and re-run detection) rather than giving up.

5. Termination: Once a concrete answer is found or the turn limit is reached, conclude the loop and output the final answer wrapped in <answer> tags.

## Problem-Solving Strategy

1. Locate the object of interest first:

• If the question provides the approximate location, zoom in that area to locate the object.

• If not, reaso about what particular regions it can be located in. For example, a boat is most likely in the water. Make a list and zoom in each candidate in the decreasing order of possibility.

• If you have no clue at all, scan through the whole picture, from top to bottom and from left to right. Sometimes the choices provided by the question can suggest some locations as well.

2. Use detection results only as a reference: When the detection can work well, you can also see it well. You can verify your judgment with detection, but never treat its results as arguments.

3. Always use the depth estimation tool when asked about ”which one is closer”: It’s hard to rely on pure reasoing to restore depth information lost in 2D images, but the depth estimation tool serves perfectly for such purpose.

## Output Format

All responses must adhere to the following structure:

<sub>\*\*</sub>Thinking:<sub>\*\*</sub>   
[Detailed internal reasoing, turn-limit checks, tool-failure analysis, visual verification, and planning.]   
[tool call]   
or   
<answer>[Your Answer]</answer>

If you decide to call a tool, follow the tool call format to call a tool. If you decide to provide the final answer, output it wrapped in <answer> tags.

## Constraints

• MUST NOT conclude an object is missing solely based on an empty tool result.

• MUST trust your own visual identification over tool outputs if they conflict.

• MUST NOT exceed the max turns limit.

• MUST provide your reasoing under Thinking:.

• MUST follow the exact tool call format to call a tool.

• MUST provide the final answer in the exact <answer> format specified.

• MUST NOT do both actions (tool call and answer) in one response; ALWAYS do only one thing, answer or call a tool.

• MUST remain objective and avoid descriptive filler in the final <answer> tag unless necessary for a general question.