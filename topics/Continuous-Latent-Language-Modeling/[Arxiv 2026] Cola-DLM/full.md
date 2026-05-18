# Cola Continuous Latent Diffusion Language Model

Hongcan Guo1,2, Qinyu Zhao3,†, Yian Zhao1,4, Shen Nie1,5, Rui Zhu1, Qiushan Guo1, Feng Wang1, Tao Yang1, Hengshuang Zhao2, Guoqiang Wei1, Yan Zeng1,

1ByteDance Seed, 2The University of Hong Kong, 3The Australian National University, 4Peking University, 5Renmin University of China

†Work done during an internship at Bytedance Seed, BCorresponding author

# Abstract

Large language models have achieved remarkable success under the autoregressive paradigm, yet high-quality text generation need not be tied to a fixed left-to-right order. Existing alternatives still struggle to jointly achieve generation efficiency, scalable representation learning, and effective global semantic modeling. We propose Cola DLM, a hierarchical latent diffusion language model that frames text generation through hierarchical information decomposition. Cola DLM first learns a stable text-to-latent mapping with a Text VAE, then models a global semantic prior in continuous latent space with a block-causal DiT, and finally generates text through conditional decoding. From a unified Markov-path perspective, its diffusion process performs latent prior transport rather than token-level observation recovery, thereby separating global semantic organization from local textual realization. This design yields a more flexible non-autoregressive inductive bias, supports semantic compression and prior fitting in continuous space, and naturally extends to other continuous modalities. Through experiments spanning 4 research questions, 8 benchmarks, strictly matched $\sim$ 2B-parameter autoregressive and LLaDA baselines, and scaling curves up to about 2000 EFLOPs, we identify an effective overall configuration of Cola DLM and verify its strong scaling behavior for text generation. Taken together, the results establish hierarchical continuous latent prior modeling as a principled alternative to strictly token-level language modeling, where generation quality and scaling behavior may better reflect model capability than likelihood, while also suggesting a concrete path toward unified modeling across discrete text and continuous modalities.

Date: May 8, 2026 Correspondence: Yan Zeng at zengyan.yanne@bytedance.com Project Page: https://hongcanguo.github.io/Cola-DLM/

# 1 Introduction

Large language models have achieved remarkable success under the autoregressive paradigm [9, 30, 38, 55, 91]. By factorizing the discrete text distribution through the chain rule [6, 22, 39, 45, 60, 102], autoregressive language models have driven major advances in large-scale pretraining, open-ended generation, and downstream transfer, and have become the dominant approach to modern language modeling [13, 65, 73, 96, 111]. However, this paradigm tightly couples generation to a fixed left-to-right order, making inference inherently sequential and restricting the model’s inductive bias to a single token ordering [1, 3, 23, 50, 67, 98, 104]. Recent progress in both discrete and continuous diffusion-based text modeling suggests that high-quality language generation need not rely on such a fixed order; instead, language models can also be defined through more general state evolution and denoising paths [15, 59, 72, 87, 103, 106].

Despite extensive exploration along autoregressive, discrete diffusion, and continuous diffusion directions [26, 41, 42, 47, 49, 85, 95, 99], existing methods still struggle to simultaneously achieve generation efficiency, scalable representation, and global semantic modeling. Autoregressive models directly parameterize token-level conditional probabilities, yielding a clear training objective, but their fixed generation order incurs inherent sequential inference cost and introduces a strong hand-crafted inductive bias, which limits performance on more general generation tasks [7, 17, 20, 53, 119]. Discrete diffusion language models remove explicit left-to-right factorization [25, 35, 36, 110], yet they still typically perform observation recovery in discrete token space, leading to costly multi-step sampling, while intermediate discrete states are not well suited to stably represent global semantic structure [40, 62, 86, 90, 94, 115, 116]. Continuous diffusion methods further introduce continuous representation spaces [28, 81, 89], but most existing approaches still use the diffusion path to recover token-aligned representations rather than to explicitly model a latent prior [21, 29]. As a result, current methods have not yet provided a unified framework that systematically combines non-autoregressive generation, continuous representation, and probabilistic text modeling.

To address this gap, we propose Cola DLM, a hierarchical latent-space diffusion language model. Cola DLM first learns a stable mapping between text and continuous latent variables through a Text VAE [8, 46, 51, 83, 100, 112], then models the latent prior in continuous latent space with a block-causal DiT [4, 11, 12, 57, 66, 75, 76, 108], and finally generates text through a conditional decoder. The key idea of ${ \mathit { c o l a } }$ DLM is to use diffusion not for token-level observation recovery, but for latent prior transport. From a unified Markov-path perspective, this design explicitly decomposes text generation into two levels: global semantic organization in continuous latent space and local textual realization through conditional decoding. This decomposition weakens the inductive bias imposed by fixed token order, allows the geometry of continuous space to directly support semantic compression and prior fitting, and enables a more flexible non-autoregressive generation process. Moreover, block-causal prior modeling preserves cross-block causal structure while allowing more efficient parallel computation within each block. Grounded in hierarchical latent-space modeling, Cola DLM is also highly modular and naturally extensible to alternative latent modeling components and other continuous modalities [19, 112].

Motivated by these observations, we systematically study diffusion language modeling in continuous latent space from both theoretical and empirical perspectives. Our contributions are as follows.

• We propose Cola DLM, a hierarchical latent-space language model that explicitly decomposes text generation into global semantic modeling and local textual realization within a unified probabilistic framework, while using diffusion-based prior modeling in continuous latent space to connect the two, thereby establishing a new paradigm for language generation from the perspective of hierarchical information decomposition.   
• We analyze the differences between Cola DLM and existing language modeling paradigms from a unified Markov-path perspective, clarifying its advantages in global semantic modeling, non-autoregressive inductive bias, and theoretical interpretability, which are further validated in the subsequent experiments.   
• Through extensive experiments spanning 4 research questions, 8 benchmarks, strictly matched $\sim$ 2Bparameter autoregressive and LLaDA baselines, and scaling curves up to about 2000 EFLOPs, we systematically validate the central claims of Cola DLM, identify an effective overall configuration, and verify its strong potential and favorable scaling behavior for text generation.   
• We further analyze several issues beyond the core framework, including the mismatch between likelihood estimation and generation quality, first-block conditioning, and latent compression. We also provide preliminary evidence that Cola DLM offers a natural bridge from discrete text to continuous modalities such as vision, pointing to a broader unified generative paradigm.

# 2 Related Work

# 2.1 Autoregressive Language Models

Autoregressive language models [56, 77, 92, 101] factorize the discrete text distribution by the chain rule and are trained with token-level maximum likelihood, making them the most widely adopted paradigm for text modeling. Their limitations are that generation is constrained by a fixed left-to-right order, inference is inherently sequential, and they are less suitable for non-monotonic generation tasks such as infilling, local editing, and global reorganization. In contrast, Cola DLM first models a global semantic prior in a continuous latent space and then performs conditional decoding, thereby alleviating token-level ordering bias and improving generation efficiency with a block-causal DiT.

# 2.2 Discrete Diffusion Language Models

Discrete diffusion language models mainly fall into two categories. The first category is based on discrete transition kernels [2, 10, 88], which define forward perturbation and reverse recovery in discrete token space and achieve non-autoregressive generation through multi-step denoising; however, sampling is usually slow and these methods cannot easily exploit the smooth semantic structure of continuous spaces. The second category is based on masking or absorbing states [69, 70, 80, 81, 84, 105, 113, 114, 117, 118], which construct training objectives by progressively mapping tokens to masks or absorbing states and then recovering the original text; however, information loss in intermediate states limits global semantic planning and fine-grained control. In contrast, Cola DLM moves the diffusion process to a continuous latent space, where compressible latent variables carry global semantics, thus combining the manipulability of continuous spaces with hierarchical semantic modeling.

# 2.3 Continuous Diffusion Language Models

Continuous diffusion language models can be broadly divided into three categories. The first category consists of high-dimensional vocabulary-aligned continuous methods [31, 43, 59, 79], which perform continuous diffusion or flow modeling directly on one-hot vectors, logit simplexes, or probability simplexes to preserve alignment with discrete vocabularies; however, their representation dimension scales with vocabulary size, which limits scalability. The second category consists of token-embedding-based continuous methods [14, 21, 24, 27, 29, 52, 54, 87], which first map text into continuous embedding spaces and then apply diffusion or flow modeling to improve generation flexibility; however, their generation process remains essentially the recovery of noisy target representations, lacking an explicit hierarchical latent-variable interpretation and a unified marginal-likelihood view of text distributions. The third category consists of latent-space continuous methods [44, 58, 63, 109], which compress text into latent spaces with autoencoders or VAEs and then perform diffusion modeling. These methods typically rely on latent-space design and autoregressive decoders, and usually treat the latent space as a fixed representation rather than modeling it under a hierarchical latent-variable framework. In contrast, Cola DLM explicitly separates global semantics from local realization through hierarchical latent-variable modeling, and learns a semantic prior in a dynamic continuous latent space, thereby better balancing modeling flexibility, inference efficiency, and theoretical interpretability.

# 3 Continuous Latent Diffusion Language Model

This section first presents Cola DLM as a hierarchical latent-variable language model with a rigorous probabilistic definition. We also outline the overall workflow of Cola DLM. We then place Cola DLM in a unified theoretical framework together with AR models, discrete denoising language models, and continuous token-space methods. Detailed derivations and proofs are deferred to Appendices A, B, C and D.

# 3.1 Theoretical Foundations of Cola DLM

In this subsection, we present Cola DLM as a hierarchical latent-variable language model with a rigorous probabilistic definition. We then introduce its unconditional and conditional probability estimators. Detailed derivations and proofs are provided in Appendices A and B.

# 3.1.1 Theoretical Formulation of Cola DLM

Hierarchical latent-variable modeling. Let $x \in \mathcal { X }$ denote a discrete text sequence, and let $z _ { 0 } \in \mathbb { R } ^ { d }$ denote its continuous latent variable. The generative model of Cola DLM consists of a conditional decoder $p _ { \theta } ( x \mid z _ { 0 } )$ and a latent prior $p _ { \psi } ( z _ { 0 } )$ :

$$
p ( x , z _ { 0 } ) = p _ { \theta } ( x \mid z _ { 0 } ) p _ { \psi } ( z _ { 0 } ) , \qquad p ( x ) = \int p _ { \theta } ( x \mid z _ { 0 } ) p _ { \psi } ( z _ { 0 } ) d z _ { 0 } .
$$

Here, $q _ { \phi } ( z _ { 0 } \mid x )$ is used only for variational inference during training, and is not part of the generative model itself.

We model $p _ { \psi } ( z _ { 0 } )$ with a continuous-flow prior. Let the base distribution be $p _ { 1 } ( z _ { 1 } ) = \mathcal { N } ( 0 , I )$ , and let $v _ { \psi } ( z _ { t } , t )$ be the vector field. Then

$$
z _ { 1 } \sim p _ { 1 } , \qquad \frac { d z _ { t } } { d t } = v _ { \psi } ( z _ { t } , t ) , \qquad z _ { 0 } = \Phi _ { 0  1 } ^ { \psi } ( z _ { 1 } ) ,
$$

which induces $p _ { \psi } = ( \Phi _ { 0  1 } ^ { \psi } ) _ { \sharp } p _ { 1 }$ . In the sequence implementation, the latent is further decomposed into blocks, $z _ { 0 } = ( z _ { 0 } ^ { ( 1 ) } , \dots , z _ { 0 } ^ { ( B ) } )$ z(B)0 ), with

$$
p _ { \psi } ( z _ { 0 } ) = p _ { \psi } ( z _ { 0 } ^ { ( 1 ) } ) \prod _ { b = 2 } ^ { B } p _ { \psi } ( z _ { 0 } ^ { ( b ) } \mid z _ { 0 } ^ { ( < b ) } ) .
$$

This factorization directly corresponds to the block-causal prior learning and block-wise inference used later.

ELBO and prior learning. By Jensen’s inequality, the training lower bound of ${ \mathit { c o l a } }$ DLM is

$$
\begin{array} { r } { \log p ( x ) \geq \mathbb { E } _ { q _ { \phi } ( z _ { 0 } \mid x ) } [ \log p _ { \theta } ( x \mid z _ { 0 } ) + \log p _ { \psi } ( z _ { 0 } ) - \log q _ { \phi } ( z _ { 0 } \mid x ) ] = : \mathcal { L } _ { \mathrm { E L B O } } ( x ) . } \end{array}
$$

Training therefore maximizes $\mathcal { L } _ { \mathrm { E L B O } } ( x )$ , or equivalently minimizes $- { \mathcal { L } } _ { \mathrm { E L B O } } ( x )$ .

Let the aggregated posterior be $\begin{array} { r } { \bar { q } _ { \phi } ( z _ { 0 } ) = \int q _ { \phi } ( z _ { 0 } \mid x ) p _ { \mathrm { d a t a } } ( x ) d x } \end{array}$ . The expected ELBO can then be written as

$$
\begin{array} { r } { \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } [ \mathcal { L } _ { \mathrm { E L B O } } ( x ) ] = \mathbb { E } _ { q ( x , z _ { 0 } ) } [ \log p _ { \theta } ( x \mid z _ { 0 } ) ] - I _ { q } ( X ; Z _ { 0 } ) - \mathrm { K L } ( \bar { q } _ { \phi } ( z _ { 0 } ) \parallel p _ { \psi } ( z _ { 0 } ) ) , } \end{array}
$$

where $q ( x , z _ { 0 } ) = p _ { \mathrm { d a t a } } ( x ) q _ { \phi } ( z _ { 0 } \mid x )$ . This decomposition shows that ${ \mathit { c o l a } }$ DLM separates text modeling into conditional reconstruction, information compression, and prior matching.

When the encoder and decoder are fixed, prior learning reduces to

$$
\operatorname* { m a x } _ { \psi } \ \mathbb { E } _ { z _ { 0 } \sim \bar { q } _ { \phi } } [ \log p _ { \psi } ( z _ { 0 } ) ] \quad \Longleftrightarrow \quad \operatorname* { m i n } _ { \psi } \ \mathrm { K L } ( \bar { q } _ { \phi } ( z _ { 0 } ) \| p _ { \psi } ( z _ { 0 } ) ) .
$$

In practice, we do not optimize the density directly. Instead, we learn the corresponding vector field with Flow Matching. For block $b$ , the conditional FM objective is

$$
\mathcal { L } _ { \mathrm { F M } } = \sum _ { b = 1 } ^ { B } \mathbb { E } _ { t , z _ { 0 } , z _ { 1 } } \left[ \left\| v _ { \psi } ( z _ { t } ^ { ( b ) } , t ; z _ { 0 } ^ { ( < b ) } ) - u _ { t } ^ { ( b ) } ( z _ { 0 } , z _ { 1 } ) \right\| _ { 2 } ^ { 2 } \right] .
$$

Flow Matching is therefore a solver for the prior in Cola DLM, rather than the definition of the model itself.

Summary. The generative distribution of Cola DLM is defined by the hierarchical factorization in Eq. (3.1): the latent prior $p _ { \psi } ( z _ { 0 } )$ generates global continuous semantics, and the decoder $p _ { \theta } ( x \mid z _ { 0 } )$ realizes discrete text. The encoder $q _ { \phi } ( z _ { 0 } \mid x )$ is not part of the generative model, but an inference model that lifts the observed data distribution $p _ { \mathrm { d a t a } } ( x )$ into a model-dependent latent joint distribution,

$$
q _ { \phi } ( x , z _ { 0 } ) = p _ { \mathrm { d a t a } } ( x ) q _ { \phi } ( z _ { 0 } \mid x ) , \qquad \bar { q } _ { \phi } ( z _ { 0 } ) = \int q _ { \phi } ( z _ { 0 } \mid x ) p _ { \mathrm { d a t a } } ( x ) d x .
$$

Thus, while $p _ { \mathrm { d a t a } } ( x )$ is fixed, the induced latent distribution $q _ { \phi }$ is not. When the representation is fixed, prior learning fits $p _ { \psi }$ to $q _ { \phi }$ . Under joint training, however, $q _ { \phi }$ and $p _ { \psi }$ co-evolve: the encoder reshapes the latent data distribution, while the learned prior regularizes and organizes the latent space. Flow Matching is therefore only an implementation choice for learning this prior transport; the underlying model remains a hierarchical latent-variable language model.

# Algorithm 1 Unconditional log-likelihood estimation

# Require: x, $K$

1: for $k = 1 , \ldots , K$ do 2: sample $z _ { 0 } ^ { ( k ) } \sim q _ { \phi } ( z _ { 0 } \mid x )$ 3: $w ^ { ( k ) } \gets \log p _ { \theta } ( x \mid z _ { 0 } ^ { ( k ) } ) + \log p _ { \psi } ( z _ { 0 } ^ { ( k ) } )$ − log qϕ(z(k)0 | x)

# 4: end for

5: output or $\begin{array} { r } { \log \widehat { p } _ { \mathrm { I W A E } } ( x ) = \log \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \exp ( w ^ { ( k ) } ) } \end{array}$ $\begin{array} { r } { \log \widehat { p } _ { \mathrm { E L B O } } ( x ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } w ^ { ( k ) } } \end{array}$ according to Eq. (3.12)

# Algorithm 2 Conditional log-likelihood estimation

Require: xpre, xres   
1: estimate $\log \widehat { p } ( x _ { \mathrm { p r e } } , x _ { \mathrm { r e s } } )$ busing Algorithm 1   
2: estimate $\log \widehat { p } ( x _ { \mathrm { p r e } } )$ busing Algorithm 1   
3: return $\log \widehat { p } ( x _ { \mathrm { r e s } } \mid x _ { \mathrm { p r e } } )$ $= \log \widehat { p } ( x _ { \mathrm { p r e } } , x _ { \mathrm { r e s } } ) - \log \widehat { p } ( x _ { \mathrm { p r e } } )$ baccording to Eq. (3.14)

# 3.1.2 Probability Estimation for Cola DLM

Unconditional probability estimation. At evaluation time, we approximate $\log p ( x )$ . For samples $z _ { 0 } ^ { ( k ) } \sim$ $q _ { \phi } ( z _ { 0 } \mid x )$ , define the importance weight

$$
\log w ^ { ( k ) } = \log p _ { \theta } ( x \mid z _ { 0 } ^ { ( k ) } ) + \log p _ { \psi } ( z _ { 0 } ^ { ( k ) } ) - \log q _ { \phi } ( z _ { 0 } ^ { ( k ) } \mid x ) .
$$

The prior term $\log p _ { \psi } ( z _ { 0 } ^ { ( k ) } )$ is evaluated by the CNF change-of-variables formula. Concretely, we solve the augmented ODE

$$
\frac { \mathrm { d } } { \mathrm { d } t } \left[ \boldsymbol { \xi } _ { t } \right] = \left[ \begin{array} { c c } { \boldsymbol { v } _ { \psi } ( \boldsymbol { z } _ { t } , t ) } \\ { \boldsymbol { \nabla } \cdot \boldsymbol { v } _ { \psi } ( \boldsymbol { z } _ { t } , t ) } \end{array} \right] , \qquad \left[ \begin{array} { c c } { \boldsymbol { z } _ { 0 } } \\ { \boldsymbol { \ell } _ { 0 } } \end{array} \right] = \left[ \begin{array} { c c } { \boldsymbol { z } _ { 0 } ^ { ( k ) } } \\ { 0 } \end{array} \right] ,
$$

from $t = 0$ to $t = 1$ , yielding $( z _ { 1 } ^ { ( k ) } , \ell _ { 1 } ^ { ( k ) } )$ . We then obtain

$$
\log p _ { \psi } ( z _ { 0 } ^ { ( k ) } ) = \log p _ { 1 } ( z _ { 1 } ^ { ( k ) } ) + \ell _ { 1 } ^ { ( k ) } ,
$$

where $p _ { 1 }$ is the terminal base distribution. In high dimensions, the divergence term is estimated with Hutchinson’s trace estimator:

$$
\nabla \cdot \boldsymbol { v } _ { \psi } ( z _ { t } , t ) = \mathrm { T r } \bigg ( \frac { \partial \boldsymbol { v } _ { \psi } ( z _ { t } , t ) } { \partial z _ { t } } \bigg ) \approx \epsilon ^ { \top } \frac { \partial \boldsymbol { v } _ { \psi } ( z _ { t } , t ) } { \partial z _ { t } } \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) ,
$$

where the same $\epsilon$ is fixed within one ODE solve.

This gives two standard estimators, namely the ELBO-style and IWAE-style estimators:

$$
\log \widehat { p } _ { \mathrm { E L B O } , K } ( x ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \log w ^ { ( k ) } , \qquad \log \widehat { p } _ { \mathrm { I W A E } , K } ( x ) = \log \left( \frac { 1 } { K } \sum _ { k = 1 } ^ { K } e ^ { \log w ^ { ( k ) } } \right) .
$$

The IWAE-style estimator is typically tighter.

Conditional probability estimation. For a prefix–response decomposition $x = ( x _ { \mathrm { p r e } } , x _ { \mathrm { r e s } } )$ , the exact identity is

$$
\log p ( x _ { \mathrm { r e s } } \mid x _ { \mathrm { p r e } } ) = \log p ( x _ { \mathrm { p r e } } , x _ { \mathrm { r e s } } ) - \log p ( x _ { \mathrm { p r e } } ) .
$$

We therefore obtain a plug-in estimator by scoring the joint sequence and the prefix with the same unconditional estimator:

$$
\log \widehat { p } ( x _ { \mathrm { r e s } } \mid x _ { \mathrm { p r e } } ) = \log \widehat { p } ( x _ { \mathrm { p r e } } , x _ { \mathrm { r e s } } ) - \log \widehat { p } ( x _ { \mathrm { p r e } } ) .
$$

# 3.2 Workflow of Cola DLM

In this section, we describe the overall workflow of Cola DLM in detail. As illustrated in Figure 1, we explain the method from three perspectives: the pretraining of the Text VAE, the pretraining of prior learning with the Text DiT, and the inference process of Cola DLM.

![](images/4e461a048f13fc56bf3b1b3b0695838135e9b1466e2aade3b8047ab864546e24.jpg)  
Figure 1 The Overall Workflow of Cola DLM. Detailed illustration of the training and inference pipeline of Cola DLM. Training Stage 1 shows Text VAE pretraining with reconstruction, BERT, and KL losses. Training Stage 2 shows joint pretraining of the Text VAE and Text DiT with gradient control for stable optimization, where a specialized block-causal mechanism is adopted in the DiT. Inference Stage illustrates the decoding process with KV cache.

# 3.2.1 Text VAE Pretraining

In the first stage, we learn a stable latent–text correspondence. The encoder maps text into the latent space, and the decoder reconstructs the original text conditioned on the latent:

$$
z _ { 0 } \sim q _ { \phi } ( z _ { 0 } \mid x ) , \qquad \hat { x } \sim p _ { \theta } ( x \mid z _ { 0 } ) .
$$

The goal of this stage is not to learn the final prior, but to establish a stable division of labor between information stored in the latent and information recovered by the decoder.

The corresponding objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { V A E } } = - \mathbb { E } _ { q _ { \phi } ( z _ { 0 } \mid x ) } \log p _ { \theta } ( x \mid z _ { 0 } ) + \beta \operatorname { K L } ( q _ { \phi } ( z _ { 0 } \mid x ) \parallel p _ { \mathrm { b a s e } } ( z _ { 0 } ) ) + \lambda _ { \mathrm { m a s k } } \mathcal { L } _ { \mathrm { m a s k } } . } \end{array}
$$

Here, $\mathcal { L } _ { \mathrm { m a s k } }$ is the BERT-style masking loss shown in the figure. It prevents the VAE encoder from collapsing semantically while the decoder merely memorizes surface text. In our experiments, the VAE does not compress the sequence length. To prevent information leakage and facilitate subsequent streaming generation, both our VAE encoder and decoder are strictly causal.

# 3.2.2 Prior Learning with Block-Causal DiT

In the second stage, we learn a conditional prior on the stabilized latent space. For block $b$ , the visible set consists of the historical clean latent blocks and the current noisy block:

$$
\mathcal { V } _ { b } = \left\{ \mathrm { s g } ( z _ { 0 } ^ { ( < b ) } ) , z _ { t } ^ { ( b ) } \right\} ,
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient. This visibility constraint enforces bidirectional attention within each block and causal dependence across blocks, consistent with Eq. (3.3).

Under this design, prior learning uses a joint objective that combines conditional Flow Matching with a reference-encoder regularizer:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { s t a g e 2 } } = \lambda _ { \mathrm { V A E } } \Bigl ( - \mathbb { E } _ { q _ { \phi } ( z _ { 0 } | x ) } \log p _ { \theta } ( x \mid z _ { 0 } ) + \beta \mathbb { E } _ { q _ { \phi } ( z _ { 0 } | x ) } \log q _ { \phi } ( z _ { 0 } \mid x ) + \lambda _ { \mathrm { m a s k } } \mathcal { L } _ { \mathrm { m a s k } } \Bigr ) } \\ & { \phantom { \mathcal { L } _ { \mathrm { s t a g e 2 } } = } + \lambda _ { \mathrm { f m } } \mathcal { L } _ { \mathrm { F M } } + \lambda _ { \mathrm { r e f } } \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } \mathrm { K L } \Bigl ( q _ { \phi } ( z _ { 0 } \mid x ) \parallel q _ { \phi _ { \mathrm { r e f } } } ( z _ { 0 } \mid x ) \Bigr ) . } \end{array}
$$

The first group preserves the autoencoding structure with regularized latent learning, the second term learns the block-level conditional prior, and the third term suppresses latent drift during joint training.

# 3.2.3 Inference: Prefix Encoding, Block-wise Generation, and Conditional Decoding

At inference time, the model first encodes the prefix into clean latent conditions:

$$
z ^ { \mathrm { p r e } } \sim q _ { \phi } ( z ^ { \mathrm { p r e } } \mid x ^ { \mathrm { p r e } } ) .
$$

It then generates the response latent block by block. Each block is obtained by transporting a noise seed under the historical condition:

$$
\hat { z } _ { 0 } ^ { ( b ) } = \Phi _ { 0  1 } ^ { \psi } \Big ( \epsilon ^ { ( b ) } ; z ^ { \mathrm { p r e } } , \hat { z } _ { 0 } ^ { ( < b ) } \Big ) , \qquad \epsilon ^ { ( b ) } \sim \mathcal { N } ( 0 , I ) .
$$

Finally, the decoder outputs the text response conditioned on the prefix and the generated latent blocks:

$$
\hat { x } ^ { \mathrm { r e s } } \sim p _ { \theta } \big ( x ^ { \mathrm { r e s } } \mid z ^ { \mathrm { p r e } } , \hat { z } _ { 0 } ^ { ( 1 : B ) } \big ) .
$$

Summary. The workflow of Cola DLM implements the above hierarchical probabilistic model through two training stages and one inference stage, rather than a mechanical cascade of VAE, DiT, and decoder. In Stage 1, the base prior $p _ { \mathrm { b a s e } }$ regularizes the latent–text interface and stabilizes the autoencoding representation, but it is not the final generative prior. In Stage 2, the block-causal DiT learns the final latent prior $p _ { \psi } ( z _ { 0 } )$ while the VAE remains trainable under reconstruction, masking, and reference regularization. This makes prior learning a controlled co-adaptation between the latent representation and the learned flow prior. At inference time, the model encodes the prefix, generates future latent blocks autoregressively in latent space, and realizes the response through the conditional decoder.

# 3.3 A Unified View of Cola DLM and Existing Methods

In this section, we compare Cola DLM with AR, LLaDA, and Plaid under a unified Markov-path perspective, and theoretically characterize the specific advantages of Cola DLM. More detailed analysis and proofs are provided in Appendices C and D.

# 3.3.1 Text Modeling under a Unified Stochastic-Path View

For a unified comparison, let $\tau = ( S _ { t } ) _ { t \in \mathcal { T } }$ be a stochastic process on state space $\boldsymbol { S }$ , with initial distribution $\mu \Theta$ transition kernel $K _ { t } ^ { \Theta }$ , and emission mechanism $e _ { \Theta } ( x \mid \tau )$ . A process-based generative model can be written as

$$
p _ { \Theta } ( x ) = \int e _ { \Theta } ( x \mid \tau ) P _ { \Theta } ( d \tau ) , \qquad P _ { \Theta } ( d \tau ) = \mu _ { \Theta } ( d s _ { 0 } ) \prod _ { t > 0 } K _ { t } ^ { \Theta } ( d s _ { t } \mid s _ { < t } ) .
$$

This common outer form does not determine the nature of the model. The essential distinction lies in the state space of the path and its semantic role: a path over text or near-lossless text-aligned representations is an observation path, whereas a path used only to generate a latent prior is a prior path.

For AR, the path is the prefix expansion itself, yielding an exact chain factorization but binding generation to a left-to-right filtration:

$$
p _ { \mathrm { A R } } ( x ) = \prod _ { i = 1 } ^ { L } p _ { \eta } ( x _ { i } \mid x _ { < i } ) .
$$

For LLaDA, the path is a discrete corruption–recovery trajectory, whose objective is observation reconstruction in a discrete state space:

$$
q ( s _ { 1 : T } \mid x ) = q _ { 1 } ( s _ { 1 } \mid x ) \prod _ { t = 2 } ^ { T } q _ { t } ( s _ { t } \mid s _ { t - 1 } ) , \qquad p _ { \theta } ( s _ { 0 : T } ) = p ( s _ { T } ) \prod _ { t = 1 } ^ { T } p _ { \theta } ( s _ { t - 1 } \mid s _ { t } ) .
$$

Thus, LLaDA weakens the handcrafted left-to-right bias, but still modifies the observation-recovery process rather than introducing an explicit hierarchical latent variable.

<table><tr><td>Method</td><td>State Space</td><td>Path Role</td><td>Generative Factorization</td><td>Where Continuity Appears</td><td>Explicit Latent</td></tr><tr><td>AR</td><td>Prefix Tokens</td><td>Direct Generation Path</td><td>Πi p(xi | x&lt;i)</td><td>None</td><td>X</td></tr><tr><td>LLaDA</td><td>Discrete Masked Sequences</td><td>Discrete Path</td><td>Observation-Recovery p(sT ) ΠIt pθ(st−1 | st)</td><td>Discrete Token Space</td><td>X</td></tr><tr><td>Plaid</td><td>Continuous Token-Aligned Representations</td><td>Continuous Path</td><td>Observation-Recovery p(hT ) ΠIt pθ(ht−1 | ht)</td><td>Continuous Token Space</td><td>X</td></tr><tr><td>Cola DLM</td><td>Compressed Latent Sequences</td><td></td><td>Prior-Transport Path ∫ pθ(x | z0) pψ(z0) dz0</td><td>Latent Space</td><td></td></tr></table>

Table 1 Unified Perspective. Key differences among text models under a unified Markov-path view.

Plaid further moves this recovery process to a continuous token-aligned representation $h _ { 0 } = E ( x )$

$$
q ( h _ { 1 : T } \mid h _ { 0 } ) = q _ { 1 } ( h _ { 1 } \mid h _ { 0 } ) \prod _ { t = 2 } ^ { T } q _ { t } ( h _ { t } \mid h _ { t - 1 } ) , \qquad p _ { \theta } ( h _ { 0 : T } ) = p ( h _ { T } ) \prod _ { t = 1 } ^ { T } p _ { \theta } ( h _ { t - 1 } \mid h _ { t } ) .
$$

Its core target is therefore continuous observation recovery, rather than a decomposition into a prior and a conditional decoder.

In Cola DLM, by contrast, the stochastic path only transports the latent prior:

$$
z _ { 1 } \sim p _ { 1 } , \qquad z _ { 0 } = \Phi _ { 0  1 } ^ { \psi } ( z _ { 1 } ) , \qquad x \sim p _ { \theta } ( x \mid z _ { 0 } ) ,
$$

with the marginal still given by Eq. (3.1). Hence, diffusion is used to learn a flexible continuous prior, not to impose a left-to-right inductive bias on text.

The reason for using a continuous path is not that continuous modeling is inherently superior, but that it naturally captures the geometry of the latent distribution. In ${ \mathit { c o l a } }$ DLM, continuity appears in $p _ { \psi } ( z _ { 0 } )$ , rather than in an observation-recovery trajectory:

$$
\frac { d z _ { t } } { d t } = v _ { \psi } ( z _ { t } , t ) , \qquad p _ { \psi } = ( \Phi _ { 0  1 } ^ { \psi } ) _ { \sharp } p _ { 1 } .
$$

Thus, the distinction between ${ \mathit { c o l a } }$ DLM and LLaDA lies in both state space and modeling target.

Finally, the reason for using a latent variable is to explicitly separate semantic structure from token realization. The information decomposition of the expected ELBO,

$$
\begin{array} { r } { \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } [ \mathcal { L } _ { \mathrm { E L B O } } ( x ) ] = \mathbb { E } _ { q ( x , z _ { 0 } ) } [ \log p _ { \theta } ( x \mid z _ { 0 } ) ] - I _ { q } ( X ; Z _ { 0 } ) - \mathrm { K L } ( \bar { q } _ { \phi } ( z _ { 0 } ) \parallel p _ { \psi } ( z _ { 0 } ) ) , } \end{array}
$$

shows that $z _ { 0 }$ is not merely a continuous surrogate for discrete text, but an explicit marginalized intermediate variable: global semantics are compressed into $z _ { 0 }$ , while local token realization is delegated to the decoder.

# 3.3.2 Theoretical Advantages of Cola DLM

A unified criterion. Let the lower bound of the approximation error for a model family $\mathcal { M }$ b e

$$
\mathcal E ( \mathcal M ) : = \operatorname* { i n f } _ { p \in \mathcal { M } } \mathrm { K L } ( p _ { \mathrm { d a t a } } ( x ) \| p ( x ) ) .
$$

For AR, the population risk is determined solely by $\mathcal { E } ( \mathcal { M } _ { \mathrm { A R } } )$ . In contrast, Cola DLM also incurs a variational inference gap:

$$
G _ { \mathrm { i n f e r } } ^ { \mathrm { C o l a D L M } } : = \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } { \mathrm { K L } } \big ( q _ { \phi } ( z _ { 0 } \mid x ) \parallel p _ { \theta , \psi } ( z _ { 0 } \mid x ) \big ) .
$$

Its total statistical burden is therefore

$$
R _ { \mathrm { C o l a D L M } } = \mathcal { E } ( \mathcal { M } _ { \mathrm { C o l a D L M } } ) + \operatorname* { i n f } _ { \phi , \theta , \psi } G _ { \mathrm { i n f e r } } ^ { \mathrm { C o l a D L M } } .
$$

Proposition 3.1 (Unified criterion). At the population level, Cola DLM outperforms a comparison model if and only if its total statistical burden is smaller. Taking $A R$ as an example,

$$
\begin{array} { r } { \mathcal { C } o l a D L M \succ \mathrm { A R \quad } \Longleftrightarrow \quad R _ { \mathrm { C o l a D L M } } < \mathcal { E } ( \mathcal { M } _ { \mathrm { A R } } ) . } \end{array}
$$

Rate-distortion and structured generation. Whether a latent bottleneck is beneficial depends on whether the data admits a low-rate but informative global representation. Define the representation rate-distortion function as

$$
D ( R ) : = \operatorname* { i n f } _ { \scriptstyle q ( z _ { 0 } | x ) : \atop { \cal I } _ { q } ( X ; Z _ { 0 } ) \leq R } \operatorname* { i n f } _ { \scriptstyle p _ { \theta } ( x | z _ { 0 } ) } \mathbb { E } _ { q ( x , z _ { 0 } ) } [ - \log p _ { \theta } ( x \mid z _ { 0 } ) ] .
$$

If $D ( R )$ is already small at a low rate $R$ , then the data admits a low-dimensional semantic variable, and a latent bottleneck is more likely to reduce the overall mismatch. Conversely, if high-quality reconstruction requires a high information rate, aggressive compression only makes conditional reconstruction harder.

This can be characterized further through a structured-generation assumption. Suppose there exists a global variable $G$ such that

$$
p _ { \mathrm { d a t a } } ( x ) = \int p ^ { \star } ( x \mid g ) p ^ { \star } ( g ) d g , \qquad H ( X \mid G ) \ll H ( X ) , \qquad \mathrm { d i m } ( G ) \ll \mathrm { d i m } ( E ( X ) ) ,
$$

then the factorization of Cola DLM is closer to the true generative mechanism: the prior models the distribution of $G$ , while the decoder handles conditional realization. In this case, the latent bottleneck helps rather than hurts.

Three governing curves and the applicability boundary. The applicability of Cola DLM is ultimately determined by three curves:and the inference-gap curve esentation rate-distortion curve . More compactly, $D ( R )$ , the prior approximation curve, $G _ { \mathrm { i n f e r } } ^ { \mathrm { C o l a D L M } }$

$$
\begin{array} { r l } { \mathcal { C } o l a \mathrm { D L M ~ i s ~ a d v a n t a g e o u s } \Longleftrightarrow } & { \Big [ D ( R ) \mathrm { ~ i s ~ a l r e a d y ~ s m a l l ~ a t ~ l o w ~ } R \Big ] } \\ & { \wedge \Big [ \mathcal { E } ( \mathcal { M } _ { \mathrm { C o l a D L M } } ) \mathrm { ~ d e c r e a s e s } \Big ] } \\ & { \wedge \Big [ G _ { \mathrm { i n f e r } } ^ { \mathrm { C o l a D L M } } \mathrm { ~ i s ~ c o n t r o l l a b l e } \Big ] . } \end{array}
$$

The benefit of Cola DLM is therefore not guaranteed by diffusion or continuity alone. It depends on whether the data exhibits a structure with low-dimensional global semantics and high-dimensional local token realization.

Summary. The central advantage of Cola DLM is not denoising itself, but the latent decomposition that separates text modeling into a global prior and a conditional realization process.

# 4 Experiments

In this section, we conduct experiments to address the following research questions:

• RQ1: Does a global semantic structure exist within the latent space? • RQ2: What type of latent space is optimal for text generation? • RQ3: Which diffusion process is most effective for text generation? • RQ4: Why use a continuous latent diffusion model for language modeling?

# 4.1 Experimental Setup

Datasets. For training, we use external open-source pretraining data. For evaluation, the internal component analysis of Cola DLM (Sections 4.2, 4.3 and 4.4) is conducted on randomly sampled subsets from the test sets of LAMBADA [74], MMLU [33], and SIQA [82]. LAMBADA is a continuation benchmark, whereas the remaining two are multiple-choice benchmarks. For external comparisons (Section 4.5), we additionally evaluate on the test sets of SQuAD [78], Story Cloze [68], OBQA [64], RACE [48], and HellaSwag [107]. Additional dataset details are deferred to Appendix H.1.

Baselines. In the internal component comparison experiments (Sections 4.2, 4.3 and 4.4), we specify the different configurations of Cola DLM. In Section 4.5, for the scaling comparison, we independently train the autoregressive and LLaDA baselines under strictly matched settings. Specifically, the autoregressive and discrete diffusion models are randomly initialized using the official modeling implementations of LLaMA [92] and LLaDA [70], respectively. Details are provided in Appendix H.2.

Metrics. As discussed in Section 5.1, the estimated perplexity exhibits a substantial mismatch with the actual generation quality of Cola DLM. Moreover, prior work [32, 34, 61, 97] has noted that perplexity is not strictly correlated with generation performance. To enable the most objective and fair comparison, we therefore evaluate all models under a unified few-shot setting across both multiple-choice and generative tasks. For multiple-choice benchmarks and continuation tasks such as Lambada and SQuAD, accuracy is computed by strict string matching between the model output and the ground-truth answer under predefined rules. Additional evaluation details are provided in Appendix H.3.

Setup. Cola DLM uses the OLMo 2 [71] tokenizer and is trained with AdamW. The learning rate starts at $1 \times 1 0 ^ { - 6 }$ , is linearly warmed up to $1 . 5 \times 1 0 ^ { - 4 }$ over the first 5,000 steps, and is then decayed with a cosine schedule to $1 \times 1 0 ^ { - 5 }$ by 1,000,000 steps. All evaluations are conducted using the checkpoint at the corresponding FLOPs budget, without EMA weights. The same tokenizer, optimization and evaluation settings are used for Cola DLM and all external baselines. In Cola DLM, the VAE contains 500M parameters and the DiT contains 1.8B parameters. For the autoregressive and discrete diffusion baselines, the embedding layer has approximately 400M parameters and the non-embedding backbone has 1.8B parameters. We thus keep the total model size of the two model families at a comparable scale of roughly 2B parameters. All methods are trained with the same random seed so that the training data are matched across runs, with the maximum sequence length set to 512. Additional details are provided in Appendix H.4 and H.5.

# 4.2 Evidence of Global Semantic Structures in Cola DLM (RQ1)

In this section, we first present an implication for the existence of global semantic structures, and then provide strong empirical evidence for their existence by quantitatively examining the performance of latent spaces with different dimensions under different timestep shifts. The full theoretical derivation, proof, and technical details are provided in Appendix E.

Implication 1. If the latent representation is purely local and fully separable, then the optimal timeshift does not exhibit a stable drift as the latent dimension changes. Therefore, if the optimal timeshift is observed to shift systematically with the latent dimension in experiments, this indicates the existence of cross-dimensional shared structures in the latent space; if this phenomenon is mainly reflected in semantic metrics, it further supports that these shared structures are related to high-level semantics.

Based on Implication 1, the focus of this section is not merely on the specific optimal loc values under different latent dimensions, but on whether the peak position of the optimal timeshift exhibits a stable and regular shift as the latent dimension varies. Figure 2 presents the corresponding experimental results.

Obs. ❶ The optimal timeshift exhibits a systematic drift with the latent dimension. As shown in the left panel of Figure 2, the best loc for Task Avg shifts from approximately 1.0 at $d = 1 6$ , to approximately 1.7 at $d = 6 4$ , and further to approximately 2.3 at $d = 1 2 8$ . This trend is clear and approximately monotonic. This phenomenon directly contradicts the separable null hypothesis. A more plausible explanation is that changing the latent dimension alters the effective noise calibration position of some cross-dimensional shared structure.

![](images/88c546d8bf441e8c120fc242e66276533308b0cd123db0774f564aa7840ecd37.jpg)  
Figure 2 Evidence of global semantic structures in the latent space. Left: as latent dimension increases, optimal timeshifts move toward larger locations, with empirical peaks matching predictions. Right: most of the evaluation metrics consistently prefer larger best locations at higher dimensions. This stable cross-metric trend supports shared global semantic structures within the latent space.

Obs. ❷ This trend is consistent across multiple semantic metrics. The right panel of Figure 2 further shows that, although the best loc values for LAMBADA, MMLU, SIQA, and Task Avg are not exactly the same, they all overall favor larger loc regions as the latent dimension increases. This indicates that the peak drift is not an accidental fluctuation of any single task, but rather a stable phenomenon jointly supported by multiple semantic evaluations. Therefore, what is modulated by timeshift is a representation structure shared across different semantic tasks.

Obs. ❸ The empirical peaks are broadly consistent with the theoretical predictions. In the left panel of Figure 2, the dashed lines indicate the theoretically predicted optimal loc values. It can be seen that, under all three latent-dimension settings, the empirical peaks are close to the predicted positions, and the drift directions are fully consistent. This suggests that the observed drift is not an arbitrary empirical hyperparameter effect, but is instead consistent with the theoretical analysis in Appendix E, namely that shared latent structures lead to dimension-dependent timeshift calibration.

Implication 1 provides a rigorous contrapositive statement for the existence of global semantic structures, while the experimental results, by providing reverse evidence for this contrapositive, offer strong empirical support for the existence of shared and semantically relevant global structures in the latent space of ${ \mathit { c o l a } }$ DLM. This also provides supporting evidence for the first condition in Eq. (3.35) of Section 3.3.2, thereby further supporting the advantage of Cola DLM.

# 4.3 Analysis of Different Latent Spaces in Cola DLM (RQ2)

In this section, we present a detailed empirical study of different latent spaces through both quantitative evaluation and visualization. We analyze the design of the latent space from three perspectives: whether it should be dynamic or static, what latent dimensionality is most appropriate, and how semantic smoothness contributes to latent-space quality. Based on these analyses, we identify the most effective directions for optimizing the latent space.

Fixed vs. Evolving Latent Space. As shown in Figure 3, this section studies whether the latent space should evolve jointly with DiT during training. Under the same compute budget, we compare five strategies: fixing a pretrained VAE (Fix VAE); initializing the VAE from pretrained weights and jointly training it with DiT using a VAE learning rate equal to that of DiT or scaled to 0.01 $\times$ (Joint DiT x1 / Joint DiT x0.01); jointly training both VAE and DiT from random initialization with the same learning rate (All Scratch x1); and an interval-based strategy (Interval), where each 5k-step cycle consists of 2k steps of joint training followed by 3k steps with the VAE frozen. The overall results suggest that the latent space should neither remain fully fixed nor be jointly optimized from scratch without constraint. Instead, the most effective strategy is to let it evolve together with DiT on top of a stable initialization.

![](images/d755c9d03c5a35481abe75b2cbb3563e796b562ce845f2544aec12f312efb997.jpg)  
Figure 3 Comparison between fixed and evolving latent spaces. Across Task Average, LAMBADA, MMLU, and SIQA, joint evolution with DiT achieves the best overall scaling and performance when initialized from a stable pretrained VAE. Fixed spaces lead to earlier saturation, while training from scratch or interval updates remains less effective. This suggests the space should evolve with DiT, but from a stable initialization rather than scratch.

Obs. ❶ Joint DiT x1 shows the strongest scaling potential. At small compute budgets, Fix VAE and Joint DiT x1 are close, and Fix VAE is sometimes slightly better. As FLOPs increase, however, Joint DiT x1 improves more steadily and achieves the best final results on Task Avg, LAMBADA, MMLU, and SIQA, whereas Fix VAE gradually saturates. This indicates that a fixed latent space helps early stability but limits the performance ceiling, while continuous co-adaptation with DiT is more beneficial for scaling.

Obs. ❷ The benefit of joint training depends on good initialization rather than trainability alone. All Scratch x1 performs consistently worse than the other methods across all metrics, and its gains remain limited throughout training. This suggests that the advantage of Joint DiT x1 does not come from making the latent space trainable by itself; it relies on starting from a meaningful pretrained latent space and then adapting it jointly with DiT.

Obs. ❸ The latent-space visualization explains why All Scratch underperforms. Figure 4 shows that All

![](images/ee3005da7a5705c4b7a64985024e4a80c0b0915dd39199fde823c7012b0d9491.jpg)  
Figure 4 Visualization of latent spaces under different training strategies. Joint optimization on a stable initialization yields a more structured, semantically organized latent space than training VAE and DiT from scratch. Increasing the latent dimension (16 to 128) partially mitigates collapse but remains less structured than the stableinitialization approach.   
tent dimension partially alleviates collapse and improves semantic capacity, but it also changes the effective noise calibration of the latent space.

Scratch with $d = 1 6$ yields a more collapsed and less structured latent space, with trajectories dominated by simple outward drift. Increasing the latent dimension to 128 partially alleviates this issue, but the geometry still remains less organized than that of Joint DiT with stable initialization. In contrast, Joint DiT produces more heterogeneous latent patterns and richer trajectories, suggesting a more structured and semantically usable space.

Obs. ❹ Effective latent evolution requires both continuous participation and sufficient update strength. Joint DiT $\mathrm { x 0 . 0 1 }$ and Interval are both better than All Scratch x1, but still clearly worse than Joint DiT $\mathrm { x 1 }$ in overall trend and final performance. This shows that partial latent participation is not enough: overly weak updates slow adaptation, while periodic freezing disrupts co-evolution with DiT. A better strategy is to update the latent space continuously and strongly, while keeping the initialization stable.

Overall, Figure 3 and Figure 4 consistently show that the best latent-space strategy is neither to keep it fixed nor to train it from scratch, but to let it evolve jointly with DiT on top of a good initialization. When the VAE and DiT are trained jointly, the VAE is exposed to more data and can fit $p _ { \mathrm { d a t a } } ( z | x )$ more accurately. This further verifies the last condition in Eq. (3.35) of Section 3.3.2, and provides strong support for the potential advantage of Cola DLM. Additional results are in Appendix H.6.

Dimensionality of the Latent Space. We next study how the latent dimensionality affects both performance and latent-space quality. Table 2 compares All Scratch models with different latent dimensions under the same EFLOPs budget (117), Figure 4 provides the corresponding latent-space visualization, and Figure 2 shows how the optimal timeshift changes with dimension. Taken together, these results suggest that increasing the la-

Table 2 Dimensionality of the latent space under 117 EFLOPs. Larger latent dimensions improve the overall average under the all-scratch setting with $\mathrm { l o c } = 1$ .   

<table><tr><td>Method</td><td>Lambada</td><td>MMLU</td><td>SIQA</td><td>Avg.</td></tr><tr><td>All Scratch, d = 16, loc=1</td><td>14.3</td><td>6.9</td><td>4.9</td><td>8.7</td></tr><tr><td>All Scratch, d = 64, loc=1</td><td>20.9</td><td>5.4</td><td>7.6</td><td>11.3</td></tr><tr><td>All Scratch, d = 128, loc=1</td><td>18.5</td><td>8.1</td><td>8.9</td><td>11.8</td></tr></table>

Obs. ❶ Increasing the latent dimension improves the overall semantic capacity under the same compute budget. As shown in Table 2, the average score increases from 8.7 at $d = 1 6$ to 11.3 at $d = 6 4$ , and further to 11.8 at $d = 1 2 8$ . The improvement is most evident on MMLU and SIQA. Although LAMBADA peaks at $d = 6 4$ , the overall trend still suggests that a larger latent space carries stronger semantic capacity under the

![](images/23636b56923e2e61e40f8686939133f378853a5ba3ac43b1525926b85beae3d6.jpg)  
Figure 5 Effect of semantic smoothness in the latent space under the Joint DiT setting. Adding a BERT-style loss consistently improves performance, with larger gains during active latent updates $( \mathrm { l r } = 1 )$ ). This suggests semantic smoothness benefits latent-space quality, especially when evolving jointly with DiT.

same compute budget.

Obs. ❷ A larger latent dimension partially alleviates latent-space collapse, but does not fully solve it. Figure 4 shows that increasing the dimension from 16 to 128 makes the latent space less collapsed and more dispersed. However, the resulting geometry still remains clearly less structured than Joint DiT with stable initialization. This indicates that increasing dimensionality is helpful, but cannot by itself replace proper latent-space formation.

Obs. ❸ The effect of latent dimensionality is not only geometric, but also dynamical. Figure 2 shows that the best timeshift systematically shifts toward larger loc values as the latent dimension increases. This means that increasing the latent dimension does not merely enlarge the space; it also changes the denoising scale at which semantic information is best recovered. Therefore, the benefit of a higher-dimensional latent space depends not only on improved geometry, but also on proper noise calibration.

Overall, Table 2, Figure 4, and Figure 2 present a consistent picture: increasing the latent dimension improves latent-space quality and downstream performance, but the gain is only partial, and its full benefit still depends on proper training dynamics and timeshift calibration.

Table 3 Performance under different VAE logSNR settings. VAE logSNR strongly affects downstream performance. A learnable setting gives the best overall results, while fixed $\mathrm { l o g S N R } = 1 . 5$ is the strongest fixed alternative.   

<table><tr><td>Compute Budget</td><td colspan="4">EFLOPs = 77.86</td><td colspan="4">EFLOPs = 116.78</td></tr><tr><td>Method</td><td>Lambada</td><td>MMLU</td><td>SIQA</td><td>Avg.</td><td>Lambada</td><td>MMLU</td><td>SIQA</td><td>Avg.</td></tr><tr><td>Fixed VAE logSNR = 1.0</td><td>27.1</td><td>5.7</td><td>11.3</td><td>14.70</td><td>30.4</td><td>7.7</td><td>18.4</td><td>18.83</td></tr><tr><td>Fixed VAE logSNR = 1.5</td><td>29.5</td><td>7.8</td><td>17.5</td><td>18.27</td><td>33.8</td><td>8.0</td><td>23.6</td><td>21.80</td></tr><tr><td>Fixed VAE logSNR = 2.0</td><td>30.9</td><td>5.1</td><td>14.3</td><td>16.77</td><td>32.7</td><td>9.7</td><td>19.5</td><td>20.63</td></tr><tr><td>Learnable VAE logSNR (≈ 4.5)</td><td>32.6</td><td>7.9</td><td>16.2</td><td>18.90</td><td>34.6</td><td>10.1</td><td>21.6</td><td>22.1</td></tr></table>

Semantic Importance of the Latent Space. As shown in Figure 5, all results in this subsection are obtained under the Joint DiT setting, where the VAE is initialized from pretrained weights and jointly optimized with DiT. We further compare whether to add a BERT-style loss in VAE training, which encourages the latent space to preserve smoother local semantics. Here, the reported lr denotes the VAE learning-rate ratio relative to DiT. The results show that such semantic smoothness is important for downstream performance, especially when the latent space is allowed to evolve more actively.

Obs. ❶ Adding BERT loss consistently improves performance when the latent space is actively updated. When the VAE learning-rate ratio is 1, BERT loss gives the best overall results across nearly the entire training range. In Figure 5, the BERT-loss curve consistently outperforms its no-BERT counterpart on Task Average, LAMBADA, MMLU, and SIQA, and also achieves the best final performance. This indicates that encouraging masked-token recoverability makes the latent space more semantically useful for downstream prediction.

Obs. ❷ Strong latent evolution is effective only with semantic guidance. When the VAE learning-rate ratio is 0.01, adding BERT loss brings only limited gains, whereas its advantage becomes clear and stable when the ratio is increased to 1. At the same time, simply increasing the VAE update strength without BERT loss does not reliably improve performance and is even weaker at several later-stage points. This shows that trainability alone is not sufficient: when the latent space evolves more actively, its updates must also be constrained toward a semantically smoother organization.

Overall, Figure 5 shows that semantic smoothness is an important property of a useful latent space. It not only improves final performance, but also makes joint latent evolution substantially more effective and stable. These results suggest that the latent should be compact but semantically sufficient, consistent with Eq. (3.5) and Eq. (3.35). The BERT-style loss helps retain useful semantics under the bottleneck.

Smoothness of the Latent Space. Table 3 compares different VAE logSNR settings at two compute budgets. The results show that the VAE logSNR is an important factor for latent-space smoothness and downstream performance. Under the current setup, learning the VAE logSNR gives the strongest overall results, while fixing the VAE logSNR at 1.5 is the most competitive fixed alternative. The VAE logSNR formula is given in Appendix H.7.

Obs. ❶ Learning the VAE logSNR gives the strongest overall performance under the current setup. At both 77.86 and 116.78 EFLOPs, the learnable VAE logSNR setting achieves the best Task Average in Table 3. It also gives the best LAMBADA results at both checkpoints and the best MMLU result at the higher compute budget. This suggests that keeping the VAE logSNR learnable is currently the strongest overall choice, likely because it allows a more flexible smoothness profile during latent-space training.

Obs. ❷ Fixing the VAE logSNR at 1.5 is the strongest fixed alternative. Although the learnable VAE logSNR setting ranks first on average, fixing the VAE logSNR at 1.5 remains very close at both compute budgets. It also consistently achieves the best SIQA results and stays competitive on the other tasks. This indicates that a properly chosen fixed VAE logSNR can already provide a strong balance between semantic preservation and optimization stability.

Obs. $\otimes$ The current results favor a learnable VAE logSNR, while still leaving room for further study of fixed settings. The advantage of the learnable VAE logSNR over fixing the VAE logSNR at 1.5 is consistent but not large, suggesting that the current conclusion is clear but not yet definitive. Since Table 3 only reports two compute budgets, the scaling behavior of different VAE logSNR settings remains open and deserves more systematic study.

![](images/a3ffe26e00a48f5fd319e99bfdc1e4248b5a61aa024227ea5bf5f38c1eb8cfe2.jpg)  
Figure 6 Impact of DiT block size. A moderate block size (especially 16) achieves the best overall performance. Overly large blocks degrade results, while size 1 remains competitive but weaker than 16.

Overall, Table 3 shows that the VAE logSNR is an important factor in shaping latent-space smoothness and downstream performance. Under the current setup, a learnable VAE logSNR is the strongest overall choice, while fixing the VAE logSNR at 1.5 stands out as a highly competitive fixed alternative.

# 4.4 Ablation on the Diffusion Process in Cola DLM (RQ3)

In this section, we systematically study the training and inference design choices of the DiT module through ablations. By combining quantitative results with visualizations, we further analyze the mechanisms behind the observed optimization trends. On the training side, we investigate DiT models with different block sizes and examine the effect of different noise training schedules on downstream performance. On the inference side, we study the impact of the number of denoising steps and the choice of Classifier-Free Guidance (CFG) scales.

# 4.4.1 Training Stage

DiT Block Size. As shown in Figure 6, all results in this subsection are obtained under the Joint DiT setting: the VAE is initialized from pretrained weights, the VAE and DiT are jointly optimized with the same learning rate, and the training noise schedule uses loc $= 1$ . We compare four DiT block sizes at two training checkpoints to study how the local processing granularity affects downstream performance. The results show that block size has a clear effect under the current setting, and that a moderate block size works best.

Obs. ❶ Block size 16 gives the best overall performance at both checkpoints. At both 30K and 40K checkpoints, block size 16 achieves the highest Task Average in Figure 6. It also delivers the strongest or near-strongest results on all three benchmarks, especially on LAMBADA and MMLU. This suggests that, under the current setup, a moderate block size provides a favorable trade-off between local modeling capacity and semantic aggregation.

Obs. ❷ Larger block sizes are generally less effective under the current setting. When the block size is increased from 16 to 64 and 128, performance drops clearly on all three tasks at both checkpoints, with especially visible degradation on SIQA and MMLU. This suggests that overly coarse block partitioning may weaken useful semantic interactions inside the latent sequence. At the same time, since the training noise schedule is fixed to $\mathrm { l o c } { = 1 }$ here, we do not exclude the possibility that different block sizes may favor different noise calibrations.

Obs. ❸ Block size 1 is competitive but still weaker than block size 16. Block size 1 remains a relatively strong baseline and generally outperforms block sizes 64 and 128. However, it is still below block size 16 in Task Average at both checkpoints, and is notably weaker on MMLU. This suggests that fully fine-grained, completely causal processing is not necessarily the optimal way to model text in this setting, and that some degree of local grouping can be beneficial.

![](images/fc372a8b5ab42761e49831daf91a05cafeb086a2174b942408bcef476b759ba5.jpg)  
Figure 7 Noise-schedule ablation. Across all the tasks, $\mathrm { l o c } = 1$ gives the strongest overall performance, especially under Joint DiT, while uniform schedules are generally weaker. This suggests that noise-schedule calibration is important and becomes more beneficial when the latent space evolves jointly with DiT.

Overall, Figure 6 shows a clear pattern: under the current setting with loc $= 1$ , DiT block size should be neither too small nor too large. A moderate block size, especially 16, provides the most effective balance and leads to the best overall performance in the current experiments.

Noise Schedule. As shown in Figures 8 and 7, all results in this subsection are obtained with latent dimension $d = 1 6$ under the Joint DiT setting: the VAE is initialized from pretrained weights, and the VAE and DiT are jointly optimized with the same learning rate. We vary the schedule location parameter to study how noise calibration affects downstream performance, and include Fix VAE curves in Figure 7 as references. From the information-theoretic analysis in Appendix G, changing the schedule location is not merely changing a training-time heuristic: it effectively shifts the logSNR trajectory of the denoising process, and therefore changes how much semantic information remains available in the latent at different timesteps. The timestep shift formula and visualizations are provided in Appendix H.9 and H.8.

![](images/ef090b72b1332409ebc58d9af436b3653ddd1a57a4ae3cf0864cea29f6e5656a.jpg)  
Figure 8 Noise-schedule comparison at different training checkpoints. At both checkpoints, $\mathrm { l o c } = 1 . 0$ achieves the best Task Average and the most balanced overall performance across tasks. This indicates that the preferred schedule location is stable across training.

Implication 2. If the schedule location shifts the logSNR curve, then it also shifts the effective semanticinformation regime seen by the DiT during denoising. Therefore, the best noise schedule is the one whose logSNR trajectory is best aligned with the latent space and the semantic scale to be recovered, rather than a universally fixed timestep parameterization.

Obs. ❶ A moderate schedule location around $\pmb { \mathrm { l o c } } \mathrm { = } 1 . 0$ gives the best overall performance under the current setting. Figure 8 shows that $\mathrm { l o c } { = } 1 . 0$ achieves the highest Task Average at both the 30K and 40K checkpoints. It also gives the best or near-best results on the three tasks, with especially clear gains on MMLU and SIQA. From the information-theoretic view developed in Appendix G, this suggests that $\mathrm { l o c } { = } 1 . 0$ places the denoising trajectory in a more suitable effective logSNR range for semantic recovery, whereas both smaller and larger shifts move the model away from that regime.

Obs. $\pmb { \varrho }$ Proper noise calibration is especially important for Joint DiT. Figure 7 further shows that Joint DiT with $\mathrm { l o c } { = } 1$ is the strongest trainable setting across Task Average, LAMBADA, MMLU, and SIQA, whereas Joint DiT with $\mathrm { l o c } { = 0 }$ or a uniform schedule remains clearly weaker throughout training. Moreover, Joint DiT with $\mathrm { l o c } { = 1 }$ eventually matches or surpasses the corresponding Fix VAE baselines, while the mismatched schedules do not. This indicates that joint latent evolution becomes effective only when the denoising logSNR trajectory is aligned with the semantic structure of the evolving latent space.

Obs. ❸ The effect of noise schedule should be understood through semantic-information calibration rather than as an isolated hyperparameter effect. Appendix E further implies that schedule location, latent dimension, and VAE logSNR all act on the same core object, namely the effective mutual-information curve of the semantic variable along diffusion time. From this perspective, the sensitivity observed here is not accidental: changing the noise schedule changes where the model spends its denoising capacity on the semantic-information axis. This also helps explain why different latent dimensions, different VAE smoothness settings, and potentially different DiT block sizes need not share the same optimal schedule.

Overall, Figures 8 and 7 show that the noise schedule is a key component of the training setup. Under the current Joint DiT setting with $d = 1 6$ , a properly calibrated schedule, especially loc= 1.0, is important not only for stable optimization, but more fundamentally for aligning denoising with the effective semantic-information regime of the latent space. As implied by Eq. (3.5), this will further improve the average ELBO and is therefore theoretically well founded.

![](images/adfa819b8afeead97f4eeac88634125119a3a909b51b7059d4205e39fa8f92ef.jpg)  
Figure 9 Impact of inference-time hyperparameters. Increasing denoising steps brings clear early gains but quickly saturates, while a moderate CFG value achieves the best overall performance.

# 4.4.2 Inference Stage

Denoising Steps. As shown in Figure 9a, all results in this subsection are obtained under the Joint DiT setting: the VAE is initialized from pretrained weights, and the VAE and DiT are jointly optimized with the same learning rate. We vary the number of denoising steps at inference time to study the efficiency–performance trade-off. The results show that increasing the number of steps is highly beneficial in the low-step regime, while the gain quickly saturates as the inference budget becomes larger.

Obs. ❶ Increasing denoising steps yields a clear early improvement. From 1–2 steps to 4–8 steps, all tasks improve substantially. The gain is especially large on LAMBADA, while SQuAD, SIQA, and Task Average also increase sharply. This indicates that very few denoising steps are insufficient for stable semantic recovery.

Obs. ❷ Performance saturates after a moderate number of steps. After roughly 16–32 steps, the Task Average becomes nearly flat, and the marginal gain from additional steps is very limited. A similar saturation pattern is also visible on SIQA and SQuAD. This suggests that most useful denoising progress is already completed within a moderate inference budget.

Obs. ❸ Most of the practical gain is achieved with only 8--10 denoising steps. From an efficiency perspective, 8–10 steps already recover most of the final performance. Since our DiT uses a block size of 16, this means 16 tokens can be generated with only 8–10 sequential denoising iterations, corresponding to an idealized 1.6–2.0 $\times$ reduction in sequential generation depth compared with AR decoding.

Overall, Figure 9a shows that denoising steps are important, but more is not always better. Under the Joint DiT setting, a moderate number of inference steps, around 10–32, already provides a strong trade-off between accuracy and efficiency.

Classifier-Free Guidance (CFG) Scales. As shown in Figure 9b, all results in this subsection are obtained under the Joint DiT setting: the VAE is initialized from pretrained weights, and the VAE and DiT are jointly optimized with the same learning rate. We vary the Classifier-Free Guidance (CFG) scale at inference time to study how guidance strength affects downstream performance. The results show a clear non-monotonic pattern: increasing CFG is helpful at first, but overly large values significantly hurt performance.

Obs. ❶ A moderate CFG scale gives the best overall performance. The Task Average rises rapidly as CFG increases from 0 to around 3–6, and then stays near its best region for a moderate range of values. This indicates that an appropriate amount of guidance substantially improves conditional denoising and semantic recovery.

Obs. ❷ Excessive guidance leads to clear degradation. After the moderate optimum region, all task curves begin to decline as CFG becomes larger. The drop is especially pronounced beyond $\mathrm { C F G } \approx 1 0 $ , and becomes severe at very large values such as 20 and 60. This shows that overly strong guidance distorts the denoising trajectory rather than improving it.

![](images/a4ada937fe6fb01a193e4ef1fdf6e01fdc7609517e5172a0e8b0765e9d14e0d3.jpg)  
Figure 10 Overall scaling performance under a unified few-shot generative evaluation protocol. Across eight benchmarks and Task Average, Cola DLM exhibits strong scaling dynamics, ultimately reaching the best average performance. It should be noted that the lower absolute accuracy observed on specific multiple-choice tasks is an anticipated consequence of the rigorous generative evaluation paradigm; nevertheless, the underlying scaling trends are robustly preserved. These findings imply that continuous latent prior modeling possesses significant scaling potential, rendering the current performance a conservative measure of its true capacity.

Overall, Figure 9b shows that CFG is an important inference-time hyperparameter. Under the Joint DiT setting, a moderate CFG scale provides the best trade-off, while both weak guidance and excessive guidance lead to inferior results.

# 4.5 Comparison of Scaling Performance (RQ4)

In this section, we compare the scaling behavior of Cola DLM with strictly matched AR and LLaDA baselines under the best configuration identified by the previous tuning experiments. Specifically, Cola DLM uses latent dimension $d = 1 6$ , block size 16, joint VAE–DiT training with a VAE/DiT learning-rate ratio of 1, BERT loss, and a logit-normal training noise schedule with $\mathrm { l o c } = 1$ ; at inference time, we use 16 denoising steps and CFG $= 7$ . The AR and LLaDA baselines are matched in scale, with the non-embedding backbone controlled at 1.8B parameters, and LLaDA uses a denoising length equal to the generation length during inference.

It is also worth noting that the absolute scores in Figure 10 are relatively low mainly on the multiplechoice benchmarks. This is because, for a fair comparison, all models are evaluated under a unified few-shot generative protocol rather than standard likelihood-based classification: LAMBADA and SQuAD are evaluated as generative tasks, while the remaining benchmarks are multiple-choice tasks but are also cast into few-shot generation. As discussed in Section 5.1, likelihood estimation can be substantially misaligned with the actual generation quality of Cola DLM. Therefore, although the absolute values on multiple-choice tasks are lower than those in conventional discriminative evaluation, the relative scaling trends remain informative and fair under this fully matched protocol.

As shown in Figure 10, Cola DLM exhibits strong overall scaling behavior, with increasingly encouraging gains as the compute budget grows.

Obs. ❶ Cola DLM shows one of the strongest overall scaling trends. On Task Average, Cola DLM improves steadily across the full compute range and reaches the best final performance. AR remains competitive at smaller budgets, and LLaDA also shows clear early gains, but the curve of Cola DLM rises more persistently toward the high-compute regime. This suggests that Cola DLM already exhibits highly competitive, and at larger budgets stronger, scaling potential under the current matched setting.

Obs. ❷ The scaling advantage of Cola DLM is especially clear on reasoning-intensive and global-semantic tasks. On MMLU, RACE, Story Cloze, and OBQA, Cola DLM maintains a strong upward trend and achieves the best or near-best performance across a wide compute range. The gains are particularly visible at medium-to-large budgets, indicating that continuous latent prior modeling is well suited to tasks that rely more on global semantic organization and holistic answer formation.

Obs. ❸ On generative tasks, Cola DLM also shows encouraging scaling behavior. For LAMBADA and SQuAD, the scaling trends remain clear under the unified generative evaluation protocol. On LAMBADA, Cola DLM improves steadily with compute and remains close to AR at larger budgets, while SQuAD shows a particularly clear gain with scale, where Cola DLM eventually surpasses AR and continues to approach the strong performance region of LLaDA. These results suggest that, on generation-oriented evaluation, Cola DLM already demonstrates scaling behavior comparable to strong baselines, with encouraging headroom as compute increases.

Obs. ❹ The current result is a conservative estimate of the scaling potential of Cola DLM. The present comparison is conducted under a relatively conservative configuration of Cola DLM. Earlier ablations already show that increasing the latent dimension from 16 to 128 can improve semantic capacity, and the analysis of logSNR also suggests that the current setting still leaves additional room for scaling. Therefore, Figure 10 should be viewed as evidence that Cola DLM already scales well under a restrained setting, rather than as the upper bound of its capability.

Overall, Figure 10 supports a consistent conclusion: under a strictly matched comparison and a unified generative evaluation protocol, Cola DLM exhibits scaling behavior that is fully competitive with strong AR and diffusion-based baselines, and on several tasks already shows particularly encouraging late-stage gains. Together with the remaining optimization headroom in latent-space design, these results provide supportive evidence that continuous latent prior modeling is a promising scaling direction for language modeling.

# 5 Discussion

In this section, we further examine several additional properties and extensions of Cola DLM. We focus on the structural gap between likelihood-oriented estimation and generation quality, analyze how different conditioning and padding strategies affect block-causal Cola DLM in the first generation block, and present a preliminary exploration of VAE-based text compression for faster generation. Finally, we highlight the broader potential of Cola DLM for combining with other continuous modalities.

# 5.1 The Structural Gap Between Likelihood-Oriented Estimation and Generation Quality

This section studies a central phenomenon in continuous latent language models: generation quality can already be reasonable while likelihood-oriented PPL remains poor. The key reason is that these two metrics

![](images/d8062a8836a87951bb2b8a52c8d4b18275d599fbbf22a4768f707b0972592e38.jpg)  
Local Latent Geometry Around Ground-Truth Tokens   
Figure 11 A local view of the mismatch between likelihood-oriented estimation and generation quality. Top: local latent geometry around representative ground-truth tokens. Bottom: corresponding prior-density landscapes. High decoder probe success and posterior hit contrast with sharply varying prior hit and density alignment. Thus, good generation relies on covering decoder-valid regions, while likelihood estimation also demands precise local calibration around the gold posterior.

target different properties. Generation only requires the prior mass to reach semantically decoder-valid regions, whereas likelihood-oriented estimation additionally requires accurate local probability calibration around the posterior neighborhood of the ground-truth target.

Let

$$
x = ( x ^ { \mathrm { p r e } } , x ^ { \mathrm { r e s } } ) ,
$$

where $x ^ { \mathrm { p r e } }$ is the prefix, $x ^ { \mathrm { r e s } }$ is the response, and $c$ denotes the conditional information induced by the prefix. The exact conditional marginal is

$$
p ( x ^ { \mathrm { r e s } } \mid c ) = \int p _ { \theta } ( x ^ { \mathrm { r e s } } \mid z , c ) p _ { \psi } ( z \mid c ) \mathrm { d } z ,
$$

while the practically accessible quantity is the local score

$$
\begin{array} { r } { S _ { \mathrm { r e s p } } ( x ) = \mathbb { E } _ { q _ { \phi } ( z | x , c ) } \Big [ \log p _ { \theta } ( x ^ { \mathrm { r e s } } \mid z , c ) + \log p _ { \psi } ( z \mid c ) - \log q _ { \phi } ( z \mid x , c ) \Big ] . } \end{array}
$$

The mismatch between these two quantities is the starting point of our analysis.

Sample text. At dawn the research vessel Meridian slipped out of the harbor and followed a chain of islands that looked like dark brushstrokes on the horizon. Mira stood at the bow with a notebook pressed against her jacket, listening as the engine settled into a steady hum and the crew argued about . . .   
Table 4 Token-level comparison across direct training and fixed VAE logSNR settings. Across the same target tokens, lower likelihood-derived PPL does not necessarily lead to better generation. This token-level evidence further illustrates the structural gap between likelihood-oriented estimation and generation quality.   

<table><tr><td>Ground-truth token</td><td>Posterior logp(z | x)</td><td>Prior logp(z)</td><td>Decoder logp(x | z)</td><td>Likelihood-derived Generated PPL ↓</td><td>token</td><td>Gen. PPL ↓ of generated</td><td>Gen. PPL ↓ of ground-truth</td></tr><tr><td colspan="8">Direct training (unfixed VAE logSNR; measured effective VAE logSNR ≈ 4.5)</td></tr><tr><td>at</td><td>18.70</td><td>4.74</td><td>3.62 × 10−4</td><td>1.15 × 106</td><td>on</td><td>3.83</td><td>6.90</td></tr><tr><td>her</td><td>22.61</td><td>2.26</td><td>-8.58 × 10−6</td><td>6.93 × 108</td><td>her</td><td>1.12</td><td>1.12</td></tr><tr><td>settled</td><td>32.02</td><td>18.45</td><td>−8.54 × 10−3</td><td>7.88 × 105</td><td>of</td><td>5.05</td><td>2.58 × 103</td></tr><tr><td colspan="8">Fixed VAE logSNR = 1.0</td></tr><tr><td>at</td><td>-15.81</td><td>-22.28</td><td>−2.10 × 10−5</td><td>641.57</td><td>in</td><td>6.69</td><td>6.90</td></tr><tr><td>her</td><td>-11.79</td><td>-15.68</td><td>−1.91 × 10−5</td><td>48.98</td><td>the</td><td>14.17</td><td>1.12</td></tr><tr><td>settled</td><td>-12.56</td><td>-21.51</td><td>−1.40 × 10−3</td><td>7740</td><td>of</td><td>5.05</td><td>2.58 × 103</td></tr><tr><td colspan="8">Fixed VAE logSNR = 1.5</td></tr><tr><td>at</td><td>-8.04</td><td>-13.54</td><td>−7.06 × 10−5</td><td>245.36</td><td></td><td>90.94</td><td>6.90</td></tr><tr><td>her</td><td>-7.48</td><td>-15.91</td><td>−4.96 × 10−5</td><td>4.57 × 103</td><td>the</td><td>14.17</td><td>1.12</td></tr><tr><td>settled</td><td>-6.12</td><td>-15.71</td><td>-8.77 × 10−5</td><td>1.46 × 104</td><td>went</td><td>460.16</td><td>2.58 × 103</td></tr><tr><td colspan="8">Fixed VAE logSNR = 2.0</td></tr><tr><td>at</td><td>11.50</td><td>-6.41</td><td>−5.91 × 10−5</td><td>5.99 × 107</td><td>behind</td><td>114.91</td><td>6.90</td></tr><tr><td>her</td><td>13.74</td><td>0.21</td><td>−8.58 × 10−5</td><td>7.53 × 105</td><td>it</td><td>3.07 × 103</td><td>1.12</td></tr><tr><td>settled</td><td>10.30</td><td>-6.57</td><td>−4.03 × 10−3</td><td>2.14 × 107</td><td>came</td><td>442.42</td><td>2.58 × 103</td></tr></table>

Implication 3. In continuous latent language models, good generation and good likelihood-oriented estimation are not equivalent. Generation depends on whether the prior reaches semantically valid latent regions, whereas likelihood-oriented estimation additionally depends on local density calibration around the gold posterior neighborhood.

This distinction is directly supported by Figure 11 and Table 4. In Figure 11, decoder probe success and posterior hit are consistently high, showing that the decoder can reliably recover the ground-truth token inside the posterior neighborhood. However, the prior hit rates vary sharply, indicating that the main issue is not decoder failure but prior misalignment around the gold latent region. Table 4 shows the same pattern at the token level: for at, the likelihood-derived PPL improves dramatically from $1 . 1 5 \times 1 0 ^ { 6 }$ to 641.57 and then 245.36, while the generated token deteriorates from on to in and then to a comma. Similarly, for her, smaller likelihood-derived PPL under fixed VAE logSNR does not recover the correct token. Thus, lower likelihood-derived PPL does not necessarily imply better generation.

Implication 4. In Cola DLM, generation quality is more related to semantic smoothness of the latent space, whereas likelihood-oriented PPL is more sensitive to probability-space smoothness shaped by the VAE logSNR. Since these two forms of smoothness are different, generation and PPL need not be aligned.

The fixed VAE logSNR settings in Table 4 should therefore be interpreted as changing the smoothness of the latent probability space rather than merely tuning a hyperparameter. Lower VAE logSNR corresponds to a flatter local density landscape, which tends to reduce pointwise density mismatch and improve likelihoodderived PPL. However, such smoothing can also blur local semantic structure and bias the model toward generic but semantically suboptimal continuations, such as in, the, or went. By contrast, direct training yields much worse likelihood-derived PPL, but sometimes preserves more appropriate semantic behavior, such as

# Different Conditioning and Padding Strategies

![](images/a41af80aecdd2d5688bd1952ea22044b89068a950ccb84c125a1ab9051873b07.jpg)  
Figure 12 Different conditioning and padding strategies in the first generation block. The first generation block is a mixed region that contains both known prompt latents and unknown latents to be generated. Clean condition repaint keeps the known region fixed as a stable condition throughout denoising, partial repaint injects timestep-matched noisy guidance only during part of the trajectory, and left/right padding instead modify the layout of the known region without explicit repaint correction.

Table 5 Impact of first-block conditioning strategies. Clean condition repaint performs best, indicating strong, persistent conditioning is optimal for the first block’s mixed denoising. Conversely, partial repaint is much weaker, reducing $m$ degrades performance, and increasing $t$ yields no stable gains. Left and right padding outperform partial repaint but remain inferior to clean conditioning.

<table><tr><td rowspan="2">Task</td><td colspan="3">Partial repaint (t = 1)</td><td colspan="3">Partial repaint (t = 3)</td><td rowspan="2"></td><td rowspan="2">Clean cond. Left pad.</td><td rowspan="2">Right pad.</td></tr><tr><td>m = 1.0</td><td>m = 0.7</td><td>m = 0.3</td><td>m = 1.0</td><td>m = 0.7</td><td>m = 0.3</td></tr><tr><td>Lambada</td><td>8.5</td><td>8.5</td><td>6.6</td><td>7.0</td><td>7.3</td><td>5.6</td><td>37.1</td><td>24.6</td><td>24.7</td></tr><tr><td>MMLU</td><td>7.9</td><td>7.9</td><td>7.8</td><td>7.6</td><td>6.7</td><td>7.0</td><td>11.9</td><td>8.4</td><td>11.5</td></tr><tr><td>SIQA</td><td>8.8</td><td>8.7</td><td>8.2</td><td>13.3</td><td>13.0</td><td>12.0</td><td>24.8</td><td>14.9</td><td>13.8</td></tr><tr><td>Avg.</td><td>8.4</td><td>8.4</td><td>7.5</td><td>9.3</td><td>9.0</td><td>8.2</td><td>24.6</td><td>16.0</td><td>16.7</td></tr></table>

correctly generating her.

Summary. In Cola DLM, generation quality primarily reflects the semantic smoothness of the latent space, whereas likelihood-oriented PPL is more sensitive to the smoothness of the latent probability space shaped by the VAE logSNR. As a result, better generation does not necessarily imply better PPL, and vice versa.

# 5.2 Impact of Conditioning and Padding Strategies in the First Generation Block

In the first generation block, the input contains both known prompt latents and unknown latents to be generated. Figure 12 illustrates four representative strategies for handling this mixed region. Partial repaint injects timestep-matched noisy guidance on the known region, where $t$ controls the number of repaint repetitions and $m$ controls the fraction of the denoising trajectory that receives such guidance. Clean condition repaint instead keeps the known region fixed as clean guidance throughout denoising. By contrast, left and right padding do not explicitly repaint the known region, but only change its positional layout relative to the generated region. Notably, under the random-length setting, all aforementioned conditioning modes maintain strict consistency between training and inference.

As shown in Table 5, clean condition repaint consistently achieves the best performance across all tasks. In contrast, partial repaint is substantially weaker, and reducing $m$ generally further degrades performance, indicating that shortening the guided portion makes the known region harder to preserve. Increasing the repaint repetitions from $t = 1$ to $t = 3$ also does not bring stable gains, suggesting that repeated early corrections cannot compensate for weak conditioning. Left and right padding are often stronger than most partial repaint settings because they avoid explicitly re-noising the known region, but still remain clearly below clean condition repaint. This suggests that positional layout alone is insufficient: padding does not provide a stable condition throughout denoising, and may further complicate the block-causal attention pattern.

Table 6 Performance under different sample labels and VAE patch sizes. Patch size 2 is overall weaker, but this gap stems mainly from the Prompt Len Mod1 case (indivisible lengths). On Prompt Len Mod0, patch size 2 becomes competitive and even outperforms size 1. This suggests the weakness arises from boundary misalignment rather than latent compression itself.   

<table><tr><td>Sample Label</td><td colspan="2">Overall</td><td colspan="2">Prompt Len Modo</td><td colspan="2">Prompt Len Mod1</td></tr><tr><td>VAE Patch Size</td><td>p1</td><td>p2</td><td>p1</td><td>p2</td><td>p1</td><td>p2</td></tr><tr><td>Lambada</td><td>31.10</td><td>17.40</td><td>32.11</td><td>34.55</td><td>30.12</td><td>0.79</td></tr><tr><td>MMLU</td><td>5.40</td><td>3.90</td><td>6.89</td><td>7.68</td><td>3.86</td><td>0.00</td></tr><tr><td>SIQA</td><td>11.10</td><td>6.10</td><td>12.92</td><td>12.13</td><td>9.26</td><td>0.00</td></tr><tr><td>Avg.</td><td>15.87</td><td>9.13</td><td>17.31</td><td>18.12</td><td>14.41</td><td>0.26</td></tr></table>

Overall, these results show that the key challenge of the first generation block is to preserve the promptconditioned region while generating the remaining unknown part. For this mixed denoising problem, strong and persistent conditioning is more effective than partial noisy correction or positional layout alone. More details are provided in Appendix I.1.

# 5.3 Compression of the Latent Space

This section discusses whether compressing the text sequence in the VAE is beneficial for Cola DLM. We train two Text VAEs with the same latent dimensionality ( $d = 1 2 8$ ) but different patch sizes: p1 maps each token to one latent, while $p 2$ compresses every two tokens into one latent. All other settings follow Section 4.5: the DiT block size is 16, the training noise schedule uses logit-normal sampling with $\mathrm { l o c } = 1$ and scale = 0, and inference uses 16 denoising steps with $\mathrm { C F G } = 7 . 0$ . In Table 6, Overall reports the full evaluation result, while Prompt Len Mod0 and Prompt Len Mod1 group samples by whether the prompt length is divisible by 2.

Implication 5. The weakness of patch size 2 does not mainly come from compression itself, but from the boundary case where the prompt length is not divisible by the patch size. Once the latent grouping is well aligned with the text sequence, compression can instead become beneficial.

At the overall level, $p 2$ is much worse than $p 1$ . However, the parity split shows that this gap is almost entirely caused by Prompt Len Mod1. For odd-length prompts, $p 2$ nearly collapses on all tasks, whereas on Prompt Len Mod0, namely the even-length case seen by the patching rule, $p 2$ becomes competitive and even slightly surpasses $p 1$ on average. This suggests that the current failure is not evidence against latent compression itself, but against a compression scheme that does not robustly handle non-divisible sequence boundaries.

The reason is likely that, under patch size 2, odd-length prompts necessarily involve padding or incomplete token groups during compression. If this boundary pattern is not properly learned, the compressed prompt latent becomes semantically shifted. In Cola DLM, this issue is particularly severe because the prompt latent is the clean condition for subsequent block-wise prior generation rather than a weak auxiliary representation. Once the prompt-side latent is biased, the error propagates through denoising and finally harms conditional decoding, which naturally explains the near-zero performance on Mod1.

By contrast, the Mod0 result is encouraging. It shows that when the latent grouping is semantically valid, compressing two tokens into one latent does not necessarily hurt generation and may even help it. This is consistent with the core idea of Cola DLM: the latent space is not intended to preserve a token-aligned recovery path, but to provide a lower-rate representation for global semantic organization, while the decoder handles local realization. Under this view, moderate compression can be beneficial because each latent summarizes a larger textual span and thus better matches the role of the prior.

This also makes latent compression attractive from the efficiency perspective. Under the same DiT block size, one denoising block corresponds to patch size $\times$ block size text tokens after decoding. Therefore, with block size 16, patch size 1 covers 16 text tokens per block, while patch size 2 covers 32. If the boundary issue can be resolved, larger patch sizes may improve both semantic abstraction and generation efficiency.

Summary. Table 6 suggests that latent compression is a promising direction for Cola DLM. Its current limitation mainly comes from unstable handling of non-divisible sequence boundaries, while the aligned evenlength case already shows that compressed latents can support both stronger semantic abstraction and faster generation.

# 5.4 Robustness of VAE Latent Reconstruction

We further analyze the robustness of the VAE latent space from the reconstruction perspective. As shown in Figure 13, the VAE achieves nearly perfect reconstruction at $t = 0$ , indicating that the learned latent–text mapping remains highly faithful and does not collapse. Moreover, the reconstruction accuracy stays very high throughout the low-noise regime, and still remains around 0.92 at $t = 2 5 0$ , before degrading more noticeably under heavier noise.

These results suggest that the latent space learned by the VAE is not merely a fragile compressed code, but a stable and broadly usable intermediate representation for text. In particular, the graceful degradation pattern indicates that semantic information is not destroyed abruptly by small or moderate perturbations, which fur-

![](images/5bb73bdbb459261bc934223521b797388bb66197ffe6f33468023b3cfa3351ea.jpg)  
Figure 13 Robustness of VAE latent reconstruction. The VAE preserves near-perfect reconstruction at low noise and degrades gracefully under stronger perturbations, indicating a stable latent–text mapping.

ther supports the view that the VAE latent space in Cola DLM is sufficiently robust to serve as the semantic interface for subsequent prior modeling.

# 5.5 Towards a Unified Approach with Image Modalities

A broader implication of Cola DLM is that it provides a natural bridge from discrete text to continuous multimodal modeling. The key idea of unified modeling is not merely to place text and image into one backbone, but to map heterogeneous observations into a shared continuous latent interaction space, where higher-level semantics can be organized under common dynamics.

A natural extension of Cola DLM follows the same probabilistic decomposition as in the text-only setting. Let $\boldsymbol { x } ^ { \mathrm { t e x t } }$ and $x ^ { \mathrm { { 1 m g } } }$ denote the text and image observations, and let their modality-specific latent variables be

$$
z _ { 0 } ^ { \mathrm { t e x t } } \sim q _ { \phi _ { \mathrm { t e x t } } } ( z \mid x ^ { \mathrm { t e x t } } ) , \qquad z _ { 0 } ^ { \mathrm { i m g } } \sim q _ { \phi _ { \mathrm { i m g } } } ( z \mid x ^ { \mathrm { i m g } } ) .
$$

We then define a joint latent state

$$
\tilde { z } _ { 0 } = ( z _ { 0 } ^ { \mathrm { t e x t } } , z _ { 0 } ^ { \mathrm { i m g } } ) ,
$$

and model the unified generative process as

$$
p ( x ^ { \mathrm { t e x t } } , x ^ { \mathrm { i m g } } , \tilde { z } _ { 0 } ) = p _ { \theta } ( x ^ { \mathrm { t e x t } } , x ^ { \mathrm { i m g } } \mid \tilde { z } _ { 0 } ) \ p _ { \psi } ( \tilde { z } _ { 0 } ) .
$$

![](images/49e322a0f57412445c5c1e936c8424c4db42c80f1f3a4607ba462e790dc926dc.jpg)  
Figure 14 Preliminary qualitative examples of unified text-image modeling. Left: text-only continuation and image-conditioned text generation. Middle: text-to-image results with only pretraining. Right: a schematic extension of Cola DLM, where text and image are mapped into modality-specific continuous latents and modeled by a shared block-causal prior.

Under this view, modality-specific VAE encoders and decoders are responsible for surface-level representation and realization, while the shared prior models the higher-level semantic structure and cross-modal dependency in latent space.

This perspective is consistent with the central modeling principle of ${ \mathit { c o l a } }$ DLM. In Cola DLM, diffusion is not used for token-level observation recovery, but for latent prior transport:

$$
z _ { 1 } \sim p _ { 1 } , \qquad z _ { 0 } = \Phi _ { 0  1 } ^ { \psi } ( z _ { 1 } ) , \qquad x \sim p _ { \theta } ( x \mid z _ { 0 } ) .
$$

In the unified setting, the same idea extends to the multimodal latent state $\tilde { z } _ { 0 }$ : the shared block-causal MMDiT prior transports and organizes the joint latent semantics, while the modality-specific decoders handle the final text or image realization. Therefore, continuity is introduced at the level of prior modeling rather than direct token or pixel recovery.

From the ELBO viewpoint, the benefit of such a decomposition is also conceptually clear. A unified latent-variable objective takes the form

$$
\begin{array} { r } { \mathbb { E } [ \mathcal { L } _ { \mathrm { E L B O } } ] = \mathbb { E } _ { q } \left[ \log p _ { \theta } ( x ^ { \mathrm { t e x t } } , x ^ { \mathrm { i m g } } \mid \tilde { z } _ { 0 } ) \right] - I \Big ( ( X ^ { \mathrm { t e x t } } , X ^ { \mathrm { i m g } } ) ; \tilde { Z } _ { 0 } \Big ) - \mathrm { K L } ( \bar { q } ( \tilde { z } _ { 0 } ) \| p _ { \psi } ( \tilde { z } _ { 0 } ) ) , } \end{array}
$$

which shows the same division of labor as in the text-only case: the latent variable carries compressed global semantics, while the decoder is responsible for modality-specific realization. In this sense, unified modeling is not simply parameter sharing across modalities, but a shared semantic prior over heterogeneous observations.

Figure 14 presents a preliminary prototype of this idea. In the current design, the text sequence is divided into blocks, while the image latent is treated as a single large block. Specifically, the image representation is obtained via an Image VAE trained on internal multi-resolution data $\left( 2 5 6 ~ / ~ 3 8 4 ~ / ~ 6 4 0 ~ / ~ 1 0 2 4 \right)$ , with a spatial downsampling factor of 16 and 64 latent channels, providing a compact yet expressive latent space for visual content. The shared block-causal MMDiT prior operates over both text blocks and image latents, supporting intra-modal processing as well as cross-modal interaction. Within a unified framework, this enables text-to-text continuation, image-conditioned text generation, and text-to-image generation. We jointly optimize these tasks on internal image–text pairs during training. For the text-to-image task, we first train on 256-resolution data for 80k steps with a global batch size of approximately 3k, and then continue training on 640-resolution data for 10k steps with a global batch size of approximately 1k. For image-conditioned text generation, we adopt the same batch size configuration and train for approximately 50k steps. More result samples are provided at I.2.

These results should be interpreted primarily as qualitative evidence of feasibility. As the current prototype remains at an early stage of training, and our experiments are limited to moderate pretraining on in-house 256 and 640 resolution data, without extensive high-quality data curation or supervised fine-tuning. The goal of this section is not to present a mature multimodal system. Rather, it is to demonstrate that the hierarchical latent-prior formulation of Cola DLM naturally extends beyond text-only generation. In future work, we plan to conduct more comprehensive unified multimodal training. More broadly, these findings suggest that decoupling global latent organization from modality-specific realization may offer a structurally clean and scalable path toward more native unified generative models.

Summary. These preliminary results suggest that Cola DLM naturally extends to unified text–image modeling. A shared block-causal prior organizes global and cross-modal semantics, while modality-specific decoders handle final realization. Although still early-stage and qualitative, this prototype already shows a promising bridge from language generation to native multimodal generative modeling.

# 6 Limitations & Future Prospects

Although this paper has provided initial evidence for the feasibility, competitiveness, and promising scaling potential of Cola DLM for text generation in continuous latent space, we view it as a starting point for further exploration rather than a finished endpoint. First, at the scale and evaluation level, the current results reveal encouraging trends, but the experiments are still conducted at a relatively controlled scale and mainly serve to clarify the key properties of the framework. It is therefore natural and important to further examine its upper bound under larger model sizes, longer training, and more substantial compute budgets. Second, at the model-design level, our analyses show that the training strategy of the Text VAE, the text compression scheme, the choice of latent dimensionality, the semantic smoothness of the latent space, and the joint calibration of VAE logSNR, DiT block size, and noise schedule all affect the semantic organization of the latent space and the final generation quality. In particular, the experiments suggest that stronger latent representations usually require better-aligned noise calibration, indicating substantial room for further optimization. Finally, at the framework level, the main value of ${ \mathit { c o l a } }$ DLM lies not merely in the denoising process itself, but in its decomposition of text generation into global semantic prior modeling and local textual realization. This opens the door to exploring stronger latent modules, such as AE [5] and RAE [112], as well as more flexible prior-learning approaches, such as drifting-model-based [19] distribution matching for continuous priors. More broadly, following the idea of unified continuous latent-space modeling, the framework may also be extended to continuous modalities such as images, further advancing unified generation.

# 7 Conclusion

In conclusion, this paper presents Cola DLM, a hierarchical continuous latent diffusion language model that decomposes text generation into global semantic prior modeling in latent space and local textual realization through conditional decoding, thereby providing a principled alternative to strictly token-level language modeling. Across the full study, both the theoretical analysis and the experiments consistently suggest that text generation can benefit from hierarchical information decomposition: we find evidence of shared global semantic structure in latent space, identify effective design choices for latent-space formation and diffusion modeling, and show that under strictly matched comparisons, Cola DLM exhibits strong generation quality and encouraging scaling behavior. More broadly, our results indicate that for this class of models, generation-oriented evaluation and scaling trends may be more informative than likelihood alone, while the continuous latent formulation also offers a concrete path toward more native unified modeling across discrete text and continuous modalities.

# 8 Afterword: Research Objectives and Significance

Viewed from a broader perspective, this study is not only concerned with proposing an alternative architecture for text generation, but also with clarifying a more general picture of learning in which representation, objective, and environment must be understood jointly. From this perspective, the three themes of this work are closely connected rather than independent. The first concerns how text should be represented and generated. The second concerns what kinds of objectives and evaluation criteria are genuinely aligned with such representations. The third concerns the kind of environment in which a model should ultimately learn if the goal is more general multimodal intelligence.

A useful starting point is to formalize learning itself as a model–environment interaction system. Let the environment be

$$
\mathcal { E } = ( \Omega , \mathcal { O } , \mathcal { A } , \mathcal { T } , \mathcal { F } , \mathcal { G } ) ,
$$

where $\Omega$ is the environment state space, $\boldsymbol { \mathcal { O } }$ is the observation space, $\boldsymbol { A }$ is the action or output space, $\tau$ is the state transition mechanism, $\mathcal { F }$ is the feedback generation mechanism, and $\mathcal { G }$ is the rule that converts feedback into optimization signals. Importantly, the notion of environment is understood here in a broad sense: it includes not only the external world, but also the data distribution presented to the model, task formats, supervision protocols, and even the loss rules by which feedback is transformed into gradients.

Let the model be denoted by $M _ { \theta }$ , with internal state space $\mathcal { H }$ , state update map $U _ { \theta }$ , and policy or generation map $\Pi _ { \theta }$ . At interaction step $t$ , the closed-loop system can be written as

$$
\begin{array} { r l } & { \boldsymbol { o } _ { t } \sim \boldsymbol { P } _ { \mathcal { E } } ( \cdot \mid \boldsymbol { \omega } _ { t } ) , } \\ & { \boldsymbol { h } _ { t } = \boldsymbol { U } _ { \boldsymbol { \theta } } ( \boldsymbol { h } _ { t - 1 } , \boldsymbol { o } _ { t } ) , } \\ & { \boldsymbol { a } _ { t } \sim \boldsymbol { \Pi } _ { \boldsymbol { \theta } } ( \cdot \mid \boldsymbol { h } _ { t } ) , } \\ & { \boldsymbol { \xi } _ { t } \sim \mathcal { F } ( \cdot \mid \boldsymbol { \omega } _ { t } , \boldsymbol { o } _ { t } , \boldsymbol { a } _ { t } ) , } \\ & { \boldsymbol { \omega } _ { t + 1 } \sim \mathcal { T } ( \cdot \mid \boldsymbol { \omega } _ { t } , \boldsymbol { o } _ { t } , \boldsymbol { a } _ { t } , \boldsymbol { \xi } _ { t } ) , } \\ & { \boldsymbol { \ell } _ { t } = \mathcal { G } ( \boldsymbol { \omega } _ { t } , \boldsymbol { o } _ { t } , \boldsymbol { a } _ { t } , \boldsymbol { \xi } _ { t } ) . } \end{array}
$$

The overall learning objective is therefore

$$
\mathcal { I } ( \theta ; \mathcal { E } ) = \mathbb { E } _ { \tau \sim P ( \tau \mid \theta , \mathcal { E } ) } \left[ \sum _ { { t = 1 } } ^ { T } \gamma ^ { t - 1 } \ell _ { t } \right] ,
$$

where $\tau$ denotes a complete interaction trajectory and $\gamma$ is the discount factor.

This formalization shows directly that learning is never an isolated question of model structure alone. Rather, it is jointly determined by three factors: first, the state space in which the model absorbs and organizes information; second, the kind of feedback through which the environment defines improvement; and third, the actual structure that generates observations, transitions, and feedback. In this work, these three aspects correspond precisely to the three recurring themes of the paper: how text should be represented, which metrics are aligned with the true learning objective, and what kind of environment unified models are ultimately meant to enter.

# 8.1 Rethinking Text Modeling Paradigms: From State Space in the System to Hierarchical Text Generation

From a system-level perspective, the central question of text modeling is not merely which generation order to adopt, but rather in what kind of state text should be represented within the learning system. Mainstream autoregressive language models bind the state tightly to the surface token prefix, and generation is therefore written as

$$
p _ { \mathrm { A R } } ( x ) = \prod _ { t = 1 } ^ { n } p _ { \theta } ( x _ { t } \mid x _ { < t } ) .
$$

This factorization is highly effective, but it fundamentally corresponds to a strong modeling assumption: both global semantics and local realization are propagated through the same token-level conditional chain. In other words, it assumes that the surface string itself is the most natural and primary state space.

The route explored in this paper instead reconsiders text generation from the level of the state space itself. If text indeed contains a low-dimensional yet sufficiently useful global semantic structure, then a more natural approach is not to place the entire burden of generation on a token-level chain factorization, but to introduce latent variables explicitly and model high-level semantic organization separately from local textual realization. Correspondingly, the core factorization of ${ \mathit { c o l a } }$ DLM is

$$
p ( x , z _ { 0 } ) = p _ { \theta } ( x \mid z _ { 0 } ) p _ { \psi } ( z _ { 0 } ) , \qquad p ( x ) = \int p _ { \theta } ( x \mid z _ { 0 } ) p _ { \psi } ( z _ { 0 } ) d z _ { 0 } ,
$$

where $z _ { 0 }$ is a continuous latent variable, $p _ { \psi } ( z _ { 0 } )$ is the latent prior, and $p _ { \theta } ( x \mid z _ { 0 } )$ is the conditional decoder. The crucial change here is not merely the introduction of latent variables, but the redefinition of the role of state in the system: the path no longer acts directly on observation recovery, but instead organizes global semantics in latent space first, after which the decoder carries out local textual realization.

This point can be stated compactly through the information decomposition of the average ELBO. Let

$$
q ( x , z _ { 0 } ) : = p _ { \mathrm { d a t a } } ( x ) q _ { \phi } ( z _ { 0 } \mid x ) ,
$$

then

$$
\mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } [ \mathcal { L } _ { \mathrm { E L B O } } ( x ) ] = \mathbb { E } _ { q ( x , z _ { 0 } ) } [ \log p _ { \theta } ( x \mid z _ { 0 } ) ] - I _ { q } ( X ; Z _ { 0 } ) - \mathrm { K L } \big ( \bar { q } _ { \phi } ( z _ { 0 } ) \| p _ { \psi } ( z _ { 0 } ) \big ) ,
$$

where $\bar { q } _ { \phi } ( z _ { 0 } )$ is the aggregated posterior. This decomposition shows that hierarchical latent-space modeling breaks the text problem into three coupled but analytically distinguishable components: conditional realization, information compression, and prior matching. The latent variable is therefore not merely a continuous surrogate for discrete tokens, but an explicit intermediate state through which global semantic organization can be separated from local textual realization and modeled on its own terms.

From this perspective, compression must also be reconsidered. Prior work has emphasized the connection between compression and intelligence [37], while recent explorations of generation closer to raw data forms in images and videos, such as pixel-space modeling [19], further suggest that compression should not be equated with harmful information deletion. The key question is not whether every local detail is preserved, but whether the model can extract and organize structural information that is genuinely effective and generalizable. If text indeed admits a hierarchical structure in which high-level semantics and low-level realization are relatively separable, then reinterpreting text generation through informational hierarchy is not merely a change of method, but a theoretical re-evaluation of text modeling itself.

Accordingly, the first theme of this paper is not to reject autoregression, but to point out that autoregression occupies only one self-consistent, rather than unique, corner of the design space. If the data truly contains a hierarchy between low-dimensional global semantics and high-dimensional local realization, then organizing semantics first in a latent state and realizing text through conditional decoding may be closer to the true generative mechanism. Text generation should therefore not be understood solely as next-token fitting over discrete strings, but more generally as a systematic problem of how information is represented, compressed, and organized hierarchically.

# 8.2 Understanding the Continuous Extension of Discrete Text: From Objective Mismatch to a Shift in Evaluation Emphasis

Once the state space of the system is changed, the issue at the objective level changes accordingly. For conventional autoregressive language models, the training objective and evaluation quantities are naturally well aligned: maximum likelihood training directly corresponds to probability fitting over text, and likelihood and perplexity therefore have a clear and stable interpretation. In hierarchical continuous latent-space models, however, the actual training path is no longer direct token-level maximum likelihood, but a hierarchical objective jointly composed of reconstruction, latent prior learning, and representation regularization.

This can be seen from the relation between the ELBO and the true marginal likelihood:

$$
- { \mathcal { L } } _ { \mathrm { E L B O } } ( x ) = - \log p _ { \theta , \psi } ( x ) + \operatorname { K L } \bigl ( q _ { \phi } ( z _ { 0 } \mid x ) \parallel p _ { \theta , \psi } ( z _ { 0 } \mid x ) \bigr ) .
$$

This shows that even at the level of the ELBO, the training objective is already separated from the true log-likelihood by a variational inference gap. Furthermore, in the actual training of ${ \mathit { c o l a } }$ DLM, the model must jointly learn latent reconstruction, continuous prior fitting, and representation stabilization. The quantity being optimized is therefore not a single token-level likelihood in the classical sense.

For this reason, the mismatch should not be interpreted as a failure to learn, but rather as evidence that the model is learning something different. For autoregressive models and other paradigms that directly fit discrete distributions, likelihood and perplexity remain highly informative because they are naturally aligned with the training objective. For hierarchical continuous latent-space models, by contrast, the central issue is no longer whether local discrete distributions are fitted as sharply as possible, but whether higher-level semantic structures are effectively organized, whether the latent prior is well learned, and whether the final generations satisfy the actual task requirements.

From the perspective of systematic modeling, this phenomenon is in fact expected: when the state space expands from surface tokens to hierarchical latent variables, the optimization target correspondingly shifts from precise fitting of local discrete distributions to the organization of higher-level semantic structure, stable latent prior learning, and satisfaction of the true generative objective. For this route, generation-oriented metrics are therefore often more closely aligned with what the model is actually trained to do than perplexity. More importantly, model potential is often reflected more clearly in scaling behavior than in any single static likelihood value: what matters is whether capability continues to improve steadily as model size, data, and compute increase, rather than whether local fit under a particular pointwise metric is better.

This can also be connected to the perspective of the three governing curves developed in the theoretical analysis of the paper. For Cola DLM, the applicability of this route is not determined by a single likelihood value, but by whether three conditions hold simultaneously: the representation rate–distortion curve is already favorable at relatively low rate, the approximation error of the latent prior continues to decrease, and the inference gap remains controllable. In other words, the advantage of this route is not guaranteed automatically by latent variables or flow-based modeling themselves; it depends on whether the data truly contains a compressible global semantic structure, and whether the model can learn, fit, and realize that structure in a stable manner.

The second theme of this paper is therefore not merely that perplexity is inadequate, but that evaluation language itself must change once representation and objective have changed. For this class of models, generation quality and scaling behavior are often closer to the model’s true capability and long-term potential than traditional perplexity.

# 8.3 Exploring Unified Models: Model-Environment Interaction and the Value of Multimodal Unification

If we return again to the model–environment formalization in Eq. (8.1) and Eq. (8.8), the third theme becomes more natural. The importance of unified models does not lie merely in placing multiple modalities within a single parameterized network, but in changing the structure of the environment in which the model learns. In the real world, observations, transitions, and feedback are usually not generated independently across modalities; rather, they are often jointly determined by a shared latent state. A more general learning system therefore requires not a set of isolated modality interfaces placed side by side, but unified representations that can enter the same interaction state and share the same dynamical constraints.

This is closely related to two broader views of intelligence. One influential view understands intelligence as a collection of skills across tasks [16]. Under this view, a system becomes more capable because it can solve problems across more domains and under more diverse forms of supervision and interaction. The recent development of large language models partly reflects this tendency. A representative example is the progress of code agents that can operate within command-line environments. In such environments, the observation space, action space, and feedback mechanism are unusually well aligned with discrete symbolic representations. Interaction trajectories are easy to record, and correctness is often straightforward to verify, so these environments provide dense and precise learning signals.

Another view, closer to the world-model perspective, holds that intelligence consists in acquiring an internal model of the structure and dynamics of the world. Recent work on world models [93] moves in this direction by seeking to learn richer environmental dynamics, thereby supporting stronger generalization and more realistic interaction. From this perspective, the question is not only how many tasks a model can solve, but whether it learns in an environment whose structure is rich enough to induce the right abstractions. The environment therefore becomes central: a model can only internalize the regularities that are actually present in the observations, transitions, and feedback it encounters.

This can also be written more formally. Let the observation at step $t$ be multimodal,

$$
o _ { t } = \big ( o _ { t } ^ { ( 1 ) } , o _ { t } ^ { ( 2 ) } , \ldots , o _ { t } ^ { ( M ) } \big ) , \qquad o _ { t } ^ { ( m ) } \in \mathcal { O } ^ { ( m ) } ,
$$

and suppose there exists a joint latent state

$$
z _ { t } = \Phi \left( o _ { t } ^ { ( 1 ) } , \dots , o _ { t } ^ { ( M ) } \right) ,
$$

such that feedback and transition depend primarily on this joint state rather than on marginal factorizations over modalities:

$$
\xi _ { t } , \omega _ { t + 1 } \sim p \big ( \xi _ { t } , \omega _ { t + 1 } \mid z _ { t } , a _ { t } \big ) .
$$

If the true environmental dynamics satisfy

$$
p ( \xi _ { t } , \omega _ { t + 1 } \mid o _ { t } , a _ { t } ) \neq \prod _ { m = 1 } ^ { M } p _ { m } \big ( \xi _ { t } ^ { ( m ) } , \omega _ { t + 1 } ^ { ( m ) } \mid o _ { t } ^ { ( m ) } , a _ { t } ^ { ( m ) } \big ) ,
$$

then the learning problem is structurally non-separable across modalities. In such a case, treating each modality as an independent channel and only combining them superficially is generally insufficient. The theoretical significance of unified models lies precisely in the fact that the environment itself is non-separable in the sense of Eq. (8.17): the regularities that determine useful feedback are joint regularities rather than regularities defined on the marginal distribution of each modality.

This clarifies why multimodal unification is not merely an engineering convenience. Its purpose is not simply to process multiple data types with one backbone, but to allow the model to learn in an environment whose observation, transition, and supervision structure more faithfully reflects the coupled regularities of the real world. In such an environment, both inputs and outputs may be multimodal; useful feedback may depend on how different modalities constrain each other jointly; and the learned internal state should ideally reflect these joint constraints.

This also explains why text has long been the most difficult component in unified models. Images and videos naturally operate in continuous spaces, whereas text is a prototypically discrete modality. If they are to enter a common interaction state and share latent dynamics, a severe representational mismatch immediately arises. This is precisely one of the central obstacles repeatedly identified in recent unified-model research [18]. In this sense, the significance of Cola DLM lies not only in proposing another text generator, but in providing a natural interface through which discrete text can enter a continuous latent space.

If discrete text is mapped into a continuous latent variable through

$$
\begin{array} { r } { z ^ { \mathrm { t e x t } } \sim q _ { \phi } ( z \mid x ^ { \mathrm { t e x t } } ) , \qquad x ^ { \mathrm { t e x t } } \sim p _ { \eta } ( x \mid z ^ { \mathrm { t e x t } } ) , } \end{array}
$$

then text acquires an interface compatible with other continuous modalities. One may then define a unified interaction state

$$
\widetilde { \boldsymbol { z } } _ { t } = \boldsymbol { \Psi } \left( z _ { t } ^ { \mathrm { t e x t } } , z _ { t } ^ { \mathrm { i m g } } , z _ { t } ^ { \mathrm { v i d } } , \dots \right) ,
$$

and perform state evolution, decision making, and feedback modeling at this level. Equations (8.18)–(8.19) formalize why Cola DLM may matter beyond text generation itself: its role is not only to generate text through a different path, but to provide a bridge through which an intrinsically discrete modality can participate in a continuous multimodal interaction state. In other words, it reduces the structural mismatch that otherwise prevents text from naturally entering a shared continuous environment.

This is why the broader significance of ${ \mathit { c o l a } }$ DLM is better understood through model–environment interaction than through single-modality benchmarks alone. If learning is viewed as the optimization of Eq. (8.8) in richer and more realistic environments, then unified models matter because they expand the environments in which the model can learn. If text is to participate fully in such environments, then a bridge such as that in Eq. (8.18) becomes especially desirable. In this sense, Cola DLM is not merely an alternative text generator; it can also be understood as a candidate mechanism for aligning discrete text with continuous multimodal learning systems.

# 8.4 The Three Themes Under a Unified Perspective

In summary, the three themes of this paper are not separate supplementary discussions, but three manifestations of the same systematic problem. The first concerns the representation level: whether text should be modeled entirely on the token surface, or whether higher-level semantics can be organized in an independent latent state. The second concerns the objective level: once the model is trained through latent transport, reconstruction, and regularization rather than direct token-level maximum likelihood, which metrics remain genuinely aligned with the learning problem. The third concerns the environment level: if learning is ultimately model–environment interaction, then what kind of environment future models should inhabit, and what representational interfaces are needed for different modalities to become compatible within it.

From this perspective, autoregressive language modeling occupies a self-consistent corner of the design space: representation is tightly bound to surface tokens, the training objective is direct likelihood maximization, and the environment is largely symbolic and text-centered. The route explored in this work changes all three assumptions simultaneously. It introduces a hierarchical latent-variable representation for text, thereby changing the representational assumption; it moves optimization away from direct token-level likelihood, thereby weakening the central interpretive role of perplexity; and it provides a continuous interface for discrete text, thereby making text potentially more compatible with multimodal environments that are more naturally expressed in continuous latent space.

We therefore hope that the contribution of this work is not only a viable alternative path for text generation, but also a more systematic way of thinking that jointly considers representation, objective alignment, and environment design. More broadly, we hope it encourages future research to rethink text, images, videos, and other modalities not as isolated domains that must be solved separately, but as components of a larger learning system in which unified representation, unified objectives, and unified environments may become increasingly central to the development of more general multimodal intelligence.

# References

Praneeth Netrapalli. Tandem transformers for inference efficient llms. In Forty-first International Conference on Machine Learning, 2024.   
[2] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.   
[3] Gregor Bachmann and Vaishnavh Nagarajan. The pitfalls of next-token prediction. arXiv preprint arXiv:2403.06963, 2024.   
[4] Hmrishav Bandyopadhyay, Nikhil Pinnaparaju, Rahim Entezari, Jim Scott, Yi-Zhe Song, and Varun Jampani. Block cascading: Training free acceleration of block-causal video models. arXiv preprint arXiv:2511.20426, 2025.   
[5] Dor Bank, Noam Koenigstein, and Raja Giryes. Autoencoders. Machine learning for data science handbook: data mining and knowledge discovery handbook, pages 353–374, 2023.   
[6] Loïc Barrault, Paul-Ambroise Duquenne, Maha Elbayad, Artyom Kozhevnikov, Belen Alastruey, Pierre Andrews, Mariano Coria, Guillaume Couairon, Marta R Costa-jussà, David Dale, et al. Large concept models: Language modeling in a sentence representation space. arXiv preprint arXiv:2412.08821, 2024.   
[7] Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: Llms trained on" a is b" fail to learn" b is a". arXiv preprint arXiv:2309.12288, 2023.   
[8] Samuel Bowman, Luke Vilnis, Oriol Vinyals, Andrew Dai, Rafal Jozefowicz, and Samy Bengio. Generating sentences from a continuous space. In Proceedings of the 20th SIGNLL conference on computational natural language learning, pages 10–21, 2016.   
[9] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.   
[10] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022.   
[11] Ricardo Cannizzaro, Jonathan Routley, and Lars Kunze. Towards a causal probabilistic framework for prediction, action-selection & explanations for robot block-stacking tasks. arXiv preprint arXiv:2308.06203, 2023.   
[12] Keshigeyan Chandrasegaran, Michael Poli, Daniel Y Fu, Dongjun Kim, Lea M Hadzic, Manling Li, Agrim Gupta, Stefano Massaroli, Azalia Mirhoseini, Juan Carlos Niebles, et al. Exploring diffusion transformer designs via grafting. arXiv preprint arXiv:2506.05340, 2025.   
[13] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language models. ACM transactions on intelligent systems and technology, 15(3):1–45, 2024.   
[14] Jiaao Chen, Aston Zhang, Mu Li, Alex Smola, and Diyi Yang. A cheaper and better diffusion language model with soft-masked noise. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4765–4775, 2023.   
[15] Tianqi Chen, Shujian Zhang, and Mingyuan Zhou. Dlm-one: Diffusion language models for one-step sequence generation. arXiv preprint arXiv:2506.00290, 2025.   
[16] François Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019.   
[17] Murtaza Dalal, Alexander C Li, and Rohan Taori. Autoregressive models: What are they good for? arXiv preprint arXiv:1910.07737, 2019.   
[18] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.   
[19] Mingyang Deng, He Li, Tianhong Li, Yilun Du, and Kaiming He. Generative modeling via drifting. arXiv preprint arXiv:2602.04770, 2026.   
[20] Justin Deschenaux and Caglar Gulcehre. Promises, outlooks and challenges of diffusion language modeling. arXiv preprint arXiv:2406.11473, 2024.   
[21] Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, et al. Continuous diffusion for categorical data. arXiv preprint arXiv:2211.15089, 2022.   
[22] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 320–335, 2022.   
[23] Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the sequential dependency of llm inference using lookahead decoding. arXiv preprint arXiv:2402.02057, 2024.   
[24] Zhujin Gao, Junliang Guo, Xu Tan, Yongxin Zhu, Fang Zhang, Jiang Bian, and Linli Xu. Empowering diffusion models on the embedding space for text generation. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4664–4683, 2024.   
[25] Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky TQ Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. Advances in Neural Information Processing Systems, 37:133345–133385, 2024.   
[26] Raghavv Goel, Risheek Garrepalli, Sudhanshu Agrawal, Chris Lott, Mingu Lee, and Fatih Porikli. Skip to the good part: Representation structure & inference-time layer skipping in diffusion vs. autoregressive llms. arXiv preprint arXiv:2603.07475, 2026.   
[27] Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and LingPeng Kong. Diffuseq: Sequence to sequence text generation with diffusion models. arXiv preprint arXiv:2210.08933, 2022.   
[28] Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. arXiv preprint arXiv:2410.17891, 2024.   
[29] Ishaan Gulrajani and Tatsunori B Hashimoto. Likelihood-based diffusion language models. Advances in Neural Information Processing Systems, 36:16693–16715, 2023.   
[30] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
[31] Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplex-based diffusion language model for text generation and modular control. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11575–11596, 2023.   
[32] Tatsunori B Hashimoto, Hugh Zhang, and Percy Liang. Unifying human and statistical evaluation for natural language generation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1689–1701, 2019.   
[33] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.   
[34] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751, 2019.   
[35] Emiel Hoogeboom, Alexey A Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. arXiv preprint arXiv:2110.02037, 2021.   
[36] Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in neural information processing systems, 34:12454–12465, 2021.   
[37] Yuzhen Huang, Jinghan Zhang, Zifei Shan, and Junxian He. Compression represents intelligence linearly. arXiv preprint arXiv:2404.09937, 2024.   
[38] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.   
[39] DeLesley Hutchins, Imanol Schlag, Yuhuai Wu, Ethan Dyer, and Behnam Neyshabur. Block-recurrent transformers. Advances in neural information processing systems, 35:33248–33261, 2022.   
[40] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016.   
[41] Daniel Jarrett, Miruna Pislar, Michiel A Bakker, Michael Henry Tessler, Raphael Köster, Jan Balaguer, Romuald Elie, Christopher Summerfield, and Andrea Tacchetti. Language agents as digital representatives in collective decision-making. arXiv preprint arXiv:2502.09369, 2025.   
[42] Sullam Jeoung, Yubin Ge, Haohan Wang, and Jana Diesner. Examining alignment of large language models through representative heuristics: the case of political stereotypes. arXiv preprint arXiv:2501.14294, 2025.   
[43] Jaehyeong Jo and Sung Ju Hwang. Continuous diffusion model for language modeling. arXiv preprint arXiv:2502.11564, 2025.   
[44] Haoqiang Kang, Yizhe Zhang, Nikki Lijing Kuang, Nicklas Majamaki, Navdeep Jaitly, Yi-An Ma, and Lianhui Qin. Ladir: Latent diffusion enhances llms for text reasoning. arXiv preprint arXiv:2510.04573, 2025.   
[45] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR, 2020.   
[46] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.   
[47] Preethi Lahoti, Nicholas Blumm, Xiao Ma, Raghavendra Kotikalapudi, Sahitya Potluri, Qijun Tan, Hansa Srinivasan, Ben Packer, Ahmad Beirami, Alex Beutel, et al. Improving diversity of demographic representation in large language models via collective-critiques and self-voting. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 10383–10405, 2023.   
[48] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension dataset from examinations. In Proceedings of the 2017 conference on empirical methods in natural language processing, pages 785–794, 2017.   
[49] Bocheng Li, Zhujin Gao, and Linli Xu. Unifying continuous and discrete text diffusion with non-simultaneous diffusion processes. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11530–11551, 2025.   
[50] Chengze Li, Yitong Zhang, Jia Li, Liyi Cai, and Ge Li. Beyond autoregression: An empirical study of diffusion large language models for code generation. arXiv preprint arXiv:2509.11252, 2025.   
[51] Chunyuan Li, Xiang Gao, Yuan Li, Baolin Peng, Xiujun Li, Yizhe Zhang, and Jianfeng Gao. Optimus: Organizing sentences via pre-trained modeling of a latent space. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4678–4699, 2020.   
[52] Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusion-lm improves controllable text generation. Advances in neural information processing systems, 35:4328–4343, 2022.   
[53] Chu-Cheng Lin, Aaron Jaech, Xin Li, Matthew R Gormley, and Jason Eisner. Limitations of autoregressive models and their alternatives. In Proceedings of the 2021 conference of the North American chapter of the association for computational linguistics: Human language technologies, pages 5147–5173, 2021.   
[54] Zhenghao Lin, Yeyun Gong, Yelong Shen, Tong Wu, Zhihao Fan, Chen Lin, Nan Duan, and Weizhu Chen. Text generation with diffusion language models: A pre-training approach with continuous paragraph denoise. In International Conference on Machine Learning, pages 21051–21064. PMLR, 2023.   
[55] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024.   
[56] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.   
[57] Yuxuan Liu, Jingmin Sun, and Hayden Schaeffer. Bcat: A block causal transformer for pde foundation models for fluid dynamics. arXiv preprint arXiv:2501.18972, 2025.   
[58] Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, and Kilian Q Weinberger. Latent diffusion for language generation. Advances in Neural Information Processing Systems, 36:56998–57025, 2023.   
[59] Rabeeh Karimi Mahabadi, Hamish Ivison, Jaesung Tae, James Henderson, Iz Beltagy, Matthew E Peters, and Arman Cohan. Tess: Text-to-text self-conditioned simplex diffusion. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2347–2361, 2024.   
[60] Eran Malach. Auto-regressive next-token predictors are universal learners. arXiv preprint arXiv:2309.06979, 2023.   
[61] Clara Meister and Ryan Cotterell. Language model evaluation beyond perplexity. In Proceedings of the 59th annual meeting of the association for computational linguistics and the 11th international joint conference on natural language processing (volume 1: long papers), pages 5328–5339, 2021.   
[62] Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score matching for discrete data. Advances in Neural Information Processing Systems, 35:34532–34545, 2022.   
[63] Viacheslav Meshchaninov, Egor Chimbulatov, Alexander Shabalin, Aleksandr Abramov, and Dmitry Vetrov. Cosmos: Compressed and smooth latent space for text diffusion modeling. arXiv preprint arXiv:2506.21170, 2025.   
[64] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2381–2391, 2018.   
[65] Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. Large language models: A survey. arXiv preprint arXiv:2402.06196, 2024.   
[66] Shentong Mo, Enze Xie, Ruihang Chu, Lanqing Hong, Matthias Niessner, and Zhenguo Li. Dit-3d: Exploring plain diffusion transformers for 3d shape generation. Advances in neural information processing systems, 36: 67960–67971, 2023.   
[67] Giovanni Monea, Armand Joulin, and Edouard Grave. Pass: Parallel speculative sampling. arXiv preprint arXiv:2311.13581, 2023.   
[68] Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, and James Allen. A corpus and cloze evaluation for deeper understanding of commonsense stories. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 839–849, 2016.   
[69] Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan Li. Scaling up masked diffusion models on text. arXiv preprint arXiv:2410.18514, 2024.   
[70] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.   
[71] Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, et al. 2 olmo 2 furious. arXiv preprint arXiv:2501.00656, 2024.   
[72] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736, 2024.   
[73] James Pan and Guoliang Li. A survey of llm inference systems. arXiv preprint arXiv:2506.21901, 2025.   
[74] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th annual meeting of the association for computational linguistics (volume 1: Long papers), pages 1525–1534, 2016.   
[75] Byeongjun Park, Hyojun Go, Jin-Young Kim, Sangmin Woo, Seokil Ham, and Changick Kim. Switch diffusion transformer: Synergizing denoising tasks with sparse mixture-of-experts. In European Conference on Computer Vision, pages 461–477. Springer, 2024.   
[76] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.   
[77] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.   
[78] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: $^ { 1 0 0 , 0 0 0 + }$ questions for machine comprehension of text. In Proceedings of the 2016 conference on empirical methods in natural language processing, pages 2383–2392, 2016.   
[79] Pierre H Richemond, Sander Dieleman, and Arnaud Doucet. Categorical sdes with simplex diffusion. arXiv preprint arXiv:2210.14784, 2022.   
[80] Subham S Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024.   
[81] Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, and Volodymyr Kuleshov. The diffusion duality. arXiv preprint arXiv:2506.10892, 2025.   
[82] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social iqa: Commonsense reasoning about social interactions. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 4463–4473, 2019.   
[83] Stanislau Semeniuta, Aliaksei Severyn, and Erhardt Barth. A hybrid convolutional variational autoencoder for text generation. In Proceedings of the 2017 conference on empirical methods in natural language processing, pages 627–637, 2017.   
[84] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 37:103131–103167, 2024.   
[85] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.   
[86] Aadithya Srikanth, Mudit Gaur, and Vaneet Aggarwal. Discrete state diffusion models: A sample complexity perspective. arXiv preprint arXiv:2510.10854, 2025.   
[87] Robin Strudel, Corentin Tallec, Florent Altché, Yilun Du, Yaroslav Ganin, Arthur Mensch, Will Grathwohl, Nikolay Savinov, Sander Dieleman, Laurent Sifre, et al. Self-conditioned embedding diffusion for text generation. arXiv preprint arXiv:2211.04236, 2022.   
[88] Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuous-time discrete diffusion models. arXiv preprint arXiv:2211.16750, 2022.   
[89] Jaesung Tae, Hamish Ivison, Sachin Kumar, and Arman Cohan. Tess 2: A large-scale generalist diffusion language model. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21171–21188, 2025.   
[90] Yuhta Takida, Takashi Shibuya, WeiHsiang Liao, Chieh-Hsin Lai, Junki Ohmura, Toshimitsu Uesaka, Naoki Murata, Shusuke Takahashi, Toshiyuki Kumakura, and Yuki Mitsufuji. Sq-vae: Variational bayes on discrete representation with self-annealed stochastic quantization. arXiv preprint arXiv:2205.07547, 2022.   
[91] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.   
[92] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.   
[93] Yuanpeng Tu, Hao Luo, Xi Chen, Xiang Bai, Fan Wang, and Hengshuang Zhao. Playerone: Egocentric world simulator. arXiv preprint arXiv:2506.09995, 2025.   
[94] Siddarth Venkatraman, Moksh Jain, Luca Scimeca, Minsu Kim, Marcin Sendera, Mohsin Hasan, Luke Rowe, Sarthak Mittal, Pablo Lemos, Emmanuel Bengio, et al. Amortizing intractable inference in diffusion models for vision, language, and control. Advances in neural information processing systems, 37:76080–76114, 2024. [95] Dimitri Von Rütte, Janis Fluri, Yuhui Ding, Antonio Orvieto, Bernhard Schölkopf, and Thomas Hofmann. Generalized interpolating discrete diffusion. arXiv preprint arXiv:2503.04482, 2025. [96] Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, et al. Efficient large language models: A survey. arXiv preprint arXiv:2312.03863, 2023.   
[97] Shufan Wang, Yixiao Song, Andrew Drozdov, Aparna Garimella, Varun Manjunatha, and Mohit Iyyer. knn-lm does not improve open-ended text generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 15023–15037, 2023. [98] Shiguang Wu, Zhaochun Ren, Xin Xin, Jiyuan Yang, Mengqi Zhang, Zhumin Chen, Maarten de Rijke, and Pengjie Ren. Constrained auto-regressive decoding constrains generative retrieval. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2429–2440, 2025. [99] Minkai Xu, Tomas Geffner, Karsten Kreis, Weili Nie, Yilun Xu, Jure Leskovec, Stefano Ermon, and Arash Vahdat. Energy-based diffusion language models for text generation. arXiv preprint arXiv:2410.21357, 2024.   
[100] Peng Xu, Jackie Chi Kit Cheung, and Yanshuai Cao. On variational learning of controllable representations for text without supervision. In International Conference on Machine Learning, pages 10534–10543. PMLR, 2020.   
[101] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.   
[102] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. Xlnet: Generalized autoregressive pretraining for language understanding. Advances in neural information processing systems, 32, 2019.   
[103] Jiacheng Ye, Jiahui Gao, Shansan Gong, Lin Zheng, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Beyond autoregression: Discrete diffusion for complex reasoning and planning. arXiv preprint arXiv:2410.14157, 2024.   
[104] Haoran You, Yichao Fu, Zheng Wang, Amir Yazdanbakhsh, and Yingyan Celine Lin. When linear attention meets autoregressive decoding: Towards more effective and efficient linearized large language models. arXiv preprint arXiv:2406.07368, 2024.   
[105] Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025.   
[106] Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete diffusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025.   
[107] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th annual meeting of the association for computational linguistics, pages 4791–4800, 2019.   
[108] Amy Zhang, Clare Lyle, Shagun Sodhani, Angelos Filos, Marta Kwiatkowska, Joelle Pineau, Yarin Gal, and Doina Precup. Invariant causal prediction for block mdps. In International Conference on Machine Learning, pages 11214–11224. PMLR, 2020.   
[109] Yizhe Zhang, Jiatao Gu, Zhuofeng Wu, Shuangfei Zhai, Joshua Susskind, and Navdeep Jaitly. Planner: Generating diversified paragraph via latent language diffusion model. Advances in Neural Information Processing Systems, 36:80178–80190, 2023.   
[110] Siyan Zhao, Devaansh Gupta, Qinqing Zheng, and Aditya Grover. d1: Scaling reasoning in diffusion large language models via reinforcement learning. arXiv preprint arXiv:2504.12216, 2025.   
[111] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 1(2): 1–124, 2023.   
[112] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.   
[113] Huangjie Zheng, Shansan Gong, Ruixiang Zhang, Tianrong Chen, Jiatao Gu, Mingyuan Zhou, Navdeep Jaitly, and Yizhe Zhang. Continuously augmented discrete diffusion model for categorical generative modeling. arXiv preprint arXiv:2510.01329, 2025.   
[114] Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. arXiv preprint arXiv:2409.02908, 2024.   
[115] Lin Zheng, Jianbo Yuan, Lei Yu, and Lingpeng Kong. A reparameterized discrete diffusion model for text generation. arXiv preprint arXiv:2302.05737, 2023.   
[116] Kun Zhou, Yifan Li, Xin Zhao, and Ji-Rong Wen. Diffusion-nat: Self-prompting discrete diffusion for nonautoregressive text generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1438–1451, 2024.   
[117] Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, et al. Llada 1.5: Variance-reduced preference optimization for large language diffusion models. arXiv preprint arXiv:2505.19223, 2025.   
[118] Fengqi Zhu, Zebin You, Yipeng Xing, Zenan Huang, Lin Liu, Yihong Zhuang, Guoshan Lu, Kangyu Wang, Xudong Wang, Lanning Wei, et al. Llada-moe: A sparse moe diffusion language model. arXiv preprint arXiv:2509.24389, 2025.   
[119] Hanlin Zhu, Baihe Huang, Shaolun Zhang, Michael Jordan, Jiantao Jiao, Yuandong Tian, and Stuart Russell. Towards a theoretical understanding of the’reversal curse’via training dynamics. Advances in neural information processing systems, 37:90473–90513, 2024.

# Appendix

# A Strict Probabilistic Formulation and Training Objective Details of Cola DLM

This appendix provides the strict probabilistic formulation of Cola DLM and clarifies the role of each training component. We first define the latent-variable model and the CNF prior precisely, then derive the marginal likelihood, the ELBO, and its population form, and finally explain the exact role of Flow Matching in the overall theoretical framework.

# A.1 Probabilistic Objects, CNF Prior, and Generative Distribution

Let $x \in \mathcal { X }$ denote a discrete text sequence, and let $z _ { 0 } \in \mathbb { R } ^ { d }$ denote its continuous latent variable. Cola DLM consists of three probabilistic components:

$$
\begin{array} { r l } { q _ { \phi } ( z _ { 0 } \mid x ) } & { \mathrm { e n c o d e r ~ ( a p p r o x i m a t e ~ p o s t e r i o r ) , } } \\ { p _ { \theta } ( x \mid z _ { 0 } ) } & { \mathrm { d e c o d e r ~ ( c o n d i t i o n a l ~ t e x t ~ g e n e r a t o r ) , } } \\ { p _ { \psi } ( z _ { 0 } ) } & { \mathrm { l a t e n t ~ p r i o r ~ ( d e f i n e d ~ b y ~ a ~ c o n t i n u o u s ~ f l o w ) . } } \end{array}
$$

Here, $q _ { \phi } ( z _ { 0 } \mid x )$ is used only for variational inference during training. The actual generative distribution is defined by $p _ { \theta } ( x \mid z _ { 0 } )$ and $p _ { \psi } ( z _ { 0 } )$ .

To ensure that the flow map and density-transport formula are well defined, we impose the standard regularity assumption on the CNF vector field.

Assumption A.1 (Regularity of the CNF vector field). For each parameter $\psi$ , the vector field ${ \pmb v } _ { \psi } ( z , t )$ is locally Lipschitz in $z$ , measurable in $t$ , and satisfies a linear-growth condition ensuring the existence and uniqueness of global ODE solutions. In addition, $\nabla \cdot \boldsymbol { v } _ { \psi } ( z , t )$ is integrable along the trajectories of interest. Then, for any $0 \leq s \leq t \leq 1$ , the flow map $\Phi _ { t  s } ^ { \psi }$ exists, is unique, and is invertible.

Let the base distribution be the standard Gaussian

$$
z _ { 1 } \sim p _ { 1 } ( z _ { 1 } ) = \mathcal { N } ( 0 , I ) .
$$

Given a time-dependent vector field $\pmb { v } _ { \psi } ( z _ { t } , t )$ , consider the ODE

$$
\frac { \mathrm { d } z _ { t } } { \mathrm { d } t } = { v _ { \psi } } ( z _ { t } , t ) , \qquad t \in [ 0 , 1 ] .
$$

By Assumption A.1, this ODE induces a flow map

$$
z _ { t } = \Phi _ { t  s } ^ { \psi } ( z _ { s } ) .
$$

Therefore, a prior sample can be written as

$$
z _ { 0 } = \Phi _ { 0  1 } ^ { \psi } ( z _ { 1 } ) , \qquad z _ { 1 } \sim p _ { 1 } .
$$

Equivalently, the prior distribution is the pushforward of the base distribution under the reverse flow:

$$
p _ { \psi } = ( \Phi _ { 0  1 } ^ { \psi } ) _ { \sharp } p _ { 1 } .
$$

This shows that $p _ { \psi } ( z _ { 0 } )$ is not an abstract symbol, but a rigorously defined probability distribution induced jointly by $( p _ { 1 } , v _ { \psi } )$ .

Hence, the generative process of ${ \mathit { c o l a } }$ DLM can be written as

$$
z _ { 1 } \sim p _ { 1 } ( z _ { 1 } ) , \qquad z _ { 0 } = \Phi _ { 0  1 } ^ { \psi } ( z _ { 1 } ) , \qquad x \sim p _ { \theta } ( x \mid z _ { 0 } ) .
$$

The corresponding joint and marginal distributions are

$$
p ( x , z _ { 0 } ) = p _ { \theta } ( x \mid z _ { 0 } ) p _ { \psi } ( z _ { 0 } ) , \qquad p ( x ) = \int p _ { \theta } ( x \mid z _ { 0 } ) p _ { \psi } ( z _ { 0 } ) \mathrm { d } z _ { 0 } .
$$

If the base variable $z _ { 1 }$ is introduced explicitly, the marginal can also be written as

$$
p ( x ) = \int p _ { 1 } ( z _ { 1 } ) p _ { \theta } \big ( x \mid \Phi _ { 0  1 } ^ { \psi } ( z _ { 1 } ) \big ) \mathrm { d } z _ { 1 } .
$$

Therefore, Cola DLM is fundamentally a hierarchical latent-variable language model, rather than a model that directly defines a reverse process in token space.

# A.2 Explicit Density of the CNF Prior and the Outer ELBO

Let $Z _ { t }$ denote the random variable induced by the ODE at time $t$ , with density $p _ { t }$ . Under Assumption A.1, $p _ { t }$ satisfies the continuity equation

$$
\begin{array} { r } { \partial _ { t } p _ { t } ( z ) + \nabla \cdot \left( p _ { t } ( z ) \pmb { v } _ { \psi } ( z , t ) \right) = 0 . } \end{array}
$$

Along a characteristic trajectory $t \mapsto z _ { t }$ , the instantaneous change-of-variables formula gives

$$
\frac { \mathrm { d } } { \mathrm { d } t } \log p _ { t } ( \boldsymbol { z } _ { t } ) = - \boldsymbol { \nabla \cdot v } _ { \psi } ( \boldsymbol { z } _ { t } , t ) .
$$

Integrating from 0 to $^ { 1 }$ , and noting that $p _ { 0 } = p _ { \psi }$ , yields

$$
\log p _ { \psi } ( \boldsymbol { z } _ { 0 } ) = \log p _ { 1 } ( \boldsymbol { z } _ { 1 } ) + \int _ { 0 } ^ { 1 } \nabla \cdot \boldsymbol { v } _ { \psi } ( \boldsymbol { z } _ { t } , t ) \mathrm { d } t , \qquad \boldsymbol { z } _ { 1 } = \Phi _ { 1  0 } ^ { \psi } ( \boldsymbol { z } _ { 0 } ) .
$$

Thus, the CNF prior admits an explicitly defined log-density rather than an implicit prior that exists only through sampling.

The flow map can also be written as a deterministic Markov kernel. Define

$$
K _ { t  s } ^ { \psi } ( \mathrm { d } z _ { t } \mid z _ { s } ) : = \delta _ { \Phi _ { t  s } ^ { \psi } ( z _ { s } ) } ( \mathrm { d } z _ { t } ) .
$$

By the semigroup property of ODE flows, $K _ { t  s } ^ { \psi }$ satisfies the Chapman–Kolmogorov relation and therefore forms a deterministic Markov semigroup. However, the Markov structure here comes from prior transport, not from an observation-dependent corruption–recovery mechanism.

We now introduce the variational posterior $q _ { \phi } ( z _ { 0 } \mid x )$ to handle the latent integral in the marginal likelihood. For any $q _ { \phi } ( z _ { 0 } \mid x )$ satisfying the required support condition,

$$
\begin{array} { r l } & { \log p ( x ) = \log \displaystyle \int q _ { \phi } ( z _ { 0 } \mid x ) \frac { p _ { \theta } ( x \mid z _ { 0 } ) p _ { \psi } ( z _ { 0 } ) } { q _ { \phi } ( z _ { 0 } \mid x ) } \operatorname { d } z _ { 0 } } \\ & { \qquad \geq \mathbb { E } _ { q _ { \phi } ( z _ { 0 } \mid x ) } \left[ \log p _ { \theta } ( x \mid z _ { 0 } ) + \log p _ { \psi } ( z _ { 0 } ) - \log q _ { \phi } ( z _ { 0 } \mid x ) \right] . } \end{array}
$$

Define the outer ELBO as

$$
\mathcal { L } _ { \mathrm { E L B O } } ( x ) : = \mathbb { E } _ { q _ { \phi } ( z _ { 0 } | x ) } \biggl [ \log p _ { \theta } ( x \mid z _ { 0 } ) + \log p _ { \psi } ( z _ { 0 } ) - \log q _ { \phi } ( z _ { 0 } \mid x ) \biggr ] .
$$

Then

$$
\begin{array} { r } { \log p ( x ) \geq \mathcal { L } _ { \mathrm { E L B O } } ( x ) . } \end{array}
$$

More importantly, the exact decomposition between the true log-likelihood and the ELBO is

$$
\begin{array} { r } { \log p ( x ) = \mathcal { L } _ { \mathrm { E L B O } } ( x ) + \mathrm { K L } \big ( q _ { \phi } ( z _ { 0 } \mid x ) \mid \mid p ( z _ { 0 } \mid x ) \big ) , } \end{array}
$$

where the true posterior is

$$
p ( z _ { 0 } \mid x ) = { \frac { p _ { \theta } ( x \mid z _ { 0 } ) p _ { \psi } ( z _ { 0 } ) } { p ( x ) } } .
$$

Equation (A.19) makes the distinction explicit: the true likelihood is always $\log p ( x )$ , whereas the training objective directly optimized is its variational lower bound $\mathcal { L } _ { \mathrm { E L B O } } ( x )$ .

Substituting the explicit CNF density formula (A.13) into the outer ELBO gives

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { E L B O } } ( x ) = \mathbb { E } _ { q _ { \phi } ( z _ { 0 } | x ) } \Big [ \log p _ { \theta } ( x \mid z _ { 0 } ) + \log p _ { 1 } ( z _ { 1 } ) + \int _ { 0 } ^ { 1 } \nabla \cdot { \pmb v } _ { \psi } ( z _ { t } , t ) \mathrm { d } t } \\ & { \qquad - \log q _ { \phi } ( z _ { 0 } \mid x ) \Big ] , \qquad z _ { 1 } = \Phi _ { 1  0 } ^ { \psi } ( z _ { 0 } ) . } \end{array}
$$

This is the strict training objective of $\mathit { c o l a }$ DLM in the setting where the decoder can compute $\log p _ { \theta } ( x \mid z _ { 0 } )$ .

# A.3 Average ELBO, Aggregated Posterior, and Prior Matching

To analyze the nature of prior learning, define the aggregated posterior

$$
\bar { q } _ { \phi } ( z _ { 0 } ) : = \int q _ { \phi } ( z _ { 0 } \mid x ) p _ { \mathrm { d a t a } } ( x ) \mathrm d x ,
$$

and the joint distribution

$$
q ( x , z _ { 0 } ) : = p _ { \mathrm { d a t a } } ( x ) q _ { \phi } ( z _ { 0 } \mid x ) .
$$

When the encoder $q _ { \phi }$ and decoder $p \theta$ are fixed, the dependence of the data-averaged ELBO on $\psi$ appears only through the prior term:

$$
\begin{array} { r l } & { \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } \big [ \mathcal { L } _ { \mathrm { E L B O } } ( x ) \big ] = \mathbb { E } _ { q ( x , z _ { 0 } ) } \big [ \log p _ { \theta } ( x \mid z _ { 0 } ) - \log q _ { \phi } ( z _ { 0 } \mid x ) \big ] + \mathbb { E } _ { q ( x , z _ { 0 } ) } \big [ \log p _ { \psi } ( z _ { 0 } ) \big ] } \\ & { \qquad = C ( \phi , \theta ) + \mathbb { E } _ { z _ { 0 } \sim \bar { q } _ { \phi } } \big [ \log p _ { \psi } ( z _ { 0 } ) \big ] , } \end{array}
$$

where $C ( \phi , \theta )$ is independent of $\psi$ .

Using

$$
\mathbb { E } _ { z _ { 0 } \sim \bar { q } _ { \phi } } [ \log p _ { \psi } ( z _ { 0 } ) ] = - \mathrm { K L } \big ( \bar { q } _ { \phi } ( z _ { 0 } ) \| p _ { \psi } ( z _ { 0 } ) \big ) + \mathbb { E } _ { z _ { 0 } \sim \bar { q } _ { \phi } } [ \log \bar { q } _ { \phi } ( z _ { 0 } ) ] ,
$$

we obtain

$$
\operatorname* { m a x } _ { \psi } \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } \big [ \mathcal { L } _ { \mathrm { E L B O } } ( x ) \big ] \quad \iff \quad \operatorname* { m i n } _ { \psi } \mathrm { K L } \big ( \bar { q } _ { \phi } ( z _ { 0 } ) \| p _ { \psi } ( z _ { 0 } ) \big ) .
$$

Therefore, after fixing the encoder and decoder, the strict prior-learning subproblem is to fit the aggregated posterior $q _ { \phi }$ with $p _ { \psi }$ .

The average ELBO can be further written in information-decomposition form. Adding and subtracting $\log \bar { q } _ { \phi } ( z _ { 0 } )$ in the prior term yields

$$
\mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } [ \mathcal { L } _ { \mathrm { E L B O } } ( x ) ] = \mathbb { E } _ { q ( x , z _ { 0 } ) } [ \log p _ { \theta } ( x \mid z _ { 0 } ) ] - I _ { q } ( X ; Z _ { 0 } ) - \mathrm { K L } \big ( \bar { q } _ { \phi } ( z _ { 0 } ) \| p _ { \psi } ( z _ { 0 } ) \big ) ,
$$

where

$$
I _ { q } ( X ; Z _ { 0 } ) : = \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } \Bigl [ \mathrm { K L } \bigl ( q _ { \phi } ( z _ { 0 } \mid x ) \parallel \bar { q } _ { \phi } ( z _ { 0 } ) \bigr ) \Bigr ]
$$

is the amount of information retained in the latent variable.

Equation (A.28) reveals the three theoretical roles of the encoder. It determines the target distribution $\bar { q } _ { \phi } ( z _ { 0 } )$ that the prior must fit, the compression strength through $I _ { q } ( X ; Z _ { 0 } )$ , and therefore the division of labor between global semantics in latent space and local realization in the decoder.

# A.4 The Role of Flow Matching: A Prior Solver Rather Than the Model Definition

Direct optimization of $\log p _ { \psi } ( z _ { 0 } )$ usually requires repeated ODE solves and divergence estimation, which is expensive in practice. For this reason, Flow Matching is often used to learn the vector field of the same prior more efficiently. It should therefore be viewed as a prior solver, not as the likelihood objective itself.

To do so, one first specifies a bridge path from the aggregated posterior to the base distribution. Let

$$
z _ { 0 } \sim \bar { q } _ { \phi } ( z _ { 0 } ) , \qquad z _ { 1 } \sim p _ { 1 } ( z _ { 1 } ) , \qquad t \sim U [ 0 , 1 ] .
$$

Consider the conditional path

$$
z _ { t } = ( 1 - \alpha ( t ) ) z _ { 0 } + \alpha ( t ) z _ { 1 } , \qquad \alpha ( 0 ) = 0 , \alpha ( 1 ) = 1 .
$$

Here $\alpha$ can be any sufficiently smooth monotone function; linear interpolation corresponds to $\alpha ( t ) = t$ . The target velocity field of this path is

$$
u _ { t } ( z _ { 0 } , z _ { 1 } ) : = { \frac { \mathrm { d } z _ { t } } { \mathrm { d } t } } = { \dot { \alpha } } ( t ) ( z _ { 1 } - z _ { 0 } ) .
$$

The Flow Matching loss is then

$$
\mathcal { L } _ { \mathrm { F M } } ( \psi ; \phi ) : = \mathbb { E } _ { t , z _ { 0 } \sim \bar { q } _ { \phi } , z _ { 1 } \sim p _ { 1 } } \left[ \big \| v _ { \psi } ( z _ { t } , t ) - u _ { t } ( z _ { 0 } , z _ { 1 } ) \big \| _ { 2 } ^ { 2 } \right] .
$$

For a fixed path construction, the pointwise optimal solution is the conditional expectation

$$
\pmb { v } _ { \psi } ^ { \star } ( z , t ) = \mathbb { E } \big [ u _ { t } ( z _ { 0 } , z _ { 1 } ) \ | \ z _ { t } = z , t \big ] .
$$

This is simply a conditional-mean regression result, but it makes the role of Flow Matching precise: under ideal conditions, the learned vector field corresponds to a continuous transport connecting $q _ { \phi }$ and $p _ { 1 }$ , which in turn defines the prior $p _ { \psi } ( z _ { 0 } )$ .

Hence, two objectives must be distinguished. The first,

$$
\operatorname* { m a x } _ { \psi } \mathbb { E } _ { z _ { 0 } \sim \bar { q } _ { \phi } } [ \log p _ { \psi } ( z _ { 0 } ) ] ,
$$

is the strict probabilistic objective for prior learning. The second,

$$
\operatorname* { m i n } _ { \psi } \mathcal { L } _ { \mathrm { F M } } ( \psi ; \phi ) ,
$$

is a practical solver for learning the vector field of the same prior. They serve the same prior-modeling problem, but they are not the same mathematical object. In particular, $\mathcal { L } _ { \mathrm { F M } }$ cannot be identified term-by-term with $- \log p _ { \psi } ( z _ { 0 } )$ in the ELBO.

# A.5 Strict Form of the Training Objective

If the decoder can compute $\log p _ { \theta } ( x \mid z _ { 0 } )$ exactly, then the strict training objective is to maximize (A.17), or equivalently to minimize the negative ELBO. Define

$$
\begin{array} { r l } & { \qquad L _ { \mathrm { R E C } } ( x ) : = - \mathbb { E } _ { q _ { \phi } ( z _ { 0 } | x ) } \left[ \log p _ { \theta } ( x \mid z _ { 0 } ) \right] , } \\ & { \qquad L _ { \mathrm { P R I O R } } ( x ) : = - \mathbb { E } _ { q _ { \phi } ( z _ { 0 } | x ) } \left[ \log p _ { \psi } ( z _ { 0 } ) \right] , } \\ & { \qquad L _ { \mathrm { R E G } } ( x ) : = \mathbb { E } _ { q _ { \phi } ( z _ { 0 } | x ) } \left[ \log q _ { \phi } ( z _ { 0 } \mid x ) \right] , } \end{array}
$$

so that the total single-sample loss becomes

$$
L _ { \mathrm { T o t a l } } ^ { \mathrm { s t r i c t } } ( x ) : = L _ { \mathrm { R E C } } ( x ) + L _ { \mathrm { P R I O R } } ( x ) + L _ { \mathrm { R E G } } ( x ) = - \mathcal { L } _ { \mathrm { E L B O } } ( x ) .
$$

Taking expectation over the data distribution gives the strict training objective

$$
\operatorname* { m i n } _ { \phi , \theta , \psi } \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } \left[ L _ { \mathrm { T o t a l } } ^ { \mathrm { s t r i c t } } ( x ) \right] \quad \iff \quad \operatorname* { m a x } _ { \phi , \theta , \psi } \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } \left[ \mathcal { L } _ { \mathrm { E L B O } } ( x ) \right] .
$$

Therefore, the objective that truly corresponds to maximizing a lower bound of the marginal likelihood is always the ELBO; Flow Matching is only an efficient implementation for learning the prior.

# B Sampling, Inference, and Probability Estimation Details

This appendix supplements the strict formulation of sampling, continuation, unconditional likelihood estimation, and conditional probability estimation in Cola DLM. Since the main text only presents the overall framework, we provide the details here.

# B.1 Sampling and Inference Procedure

To generate a new sample from ${ \mathit { c o l a } }$ DLM, one should follow the generative process defined by the model, rather than the approximate posterior path used during training. First, sample a latent seed from the simple base distribution:

$$
z _ { 1 } \sim p _ { 1 } ( z _ { 1 } ) = \mathcal { N } ( 0 , I ) .
$$

Second, integrate the vector field backward to obtain a latent sample:

$$
{ \frac { \mathrm { d } z _ { t } } { \mathrm { d } t } } = { \pmb { v } } _ { \psi } ( z _ { t } , t ) , \qquad t : 1  0 ,
$$

so that starting from $z ( 1 ) = z _ { 1 }$ yields $z ( 0 ) = z _ { 0 }$ . In practice, Euler, Heun, or higher-order ODE solvers can be used. This step generates global latent semantics rather than tokens themselves.

Third, the decoder generates text conditioned on the latent:

$$
x \sim p _ { \theta } ( x \mid z _ { 0 } ) .
$$

If the decoder is autoregressive, this step is conditional autoregressive sampling; if the decoder is nonautoregressive or bidirectional, it becomes the corresponding conditional recovery procedure. Hence, sampling in Cola DLM is inherently two-stage: one first samples a global latent, and then realizes the text conditioned on it.

If a task provides a prefix $x ^ { \mathrm { p r e } }$ , inference first encodes the prefix into a conditional latent:

$$
z ^ { \mathrm { p r e } } \sim q _ { \phi } ( z ^ { \mathrm { p r e } } \mid x ^ { \mathrm { p r e } } ) ,
$$

and then generates the response latent block by block. Let the previously generated blocks be $\hat { z } _ { 0 } ^ { ( < b ) }$ . The $b$ -th response block is generated by

$$
\hat { z } _ { 0 } ^ { ( b ) } = \Phi _ { 0  1 } ^ { \psi } \big ( \epsilon ^ { ( b ) } ; z ^ { \mathrm { p r e } } , \hat { z } _ { 0 } ^ { ( < b ) } \big ) , \qquad \epsilon ^ { ( b ) } \sim \mathcal { N } ( 0 , I ) .
$$

The response is then decoded as

$$
\hat { x } ^ { \mathrm { r e s } } \sim p _ { \theta } \big ( x ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } , z ^ { \mathrm { p r e } } , \hat { z } _ { 0 } ^ { ( 1 : B ) } \big ) .
$$

This is exactly the prefix-encoding, blockwise-generation, and conditional-decoding procedure described in the main text.

# B.2 Unconditional Log-Likelihood and PPL Estimation

For a dataset $\mathcal { D }$ , if one directly uses the ELBO as a lower bound on $\log p ( x )$ , one may define an ELBO-based perplexity as

$$
{ \mathrm { P P L } } _ { \mathrm { E L B O } } : = \exp \left( - \frac { \sum _ { x \in \mathcal { D } } \mathcal { L } _ { \mathrm { E L B O } } ( x ) } { \sum _ { x \in \mathcal { D } } \left| x \right| } \right) ,
$$

where $| x |$ denotes sequence length. Since $\mathcal { L } _ { \mathrm { E L B O } } ( x ) \leq \log p ( x )$ , PPLELBO should be interpreted as an upper-bound-type estimate of the true perplexity.

If the decoder can compute $\log p _ { \theta } ( x \mid z _ { 0 } )$ , one can also obtain tighter likelihood estimates through importance sampling or IWAE. Let

$$
z _ { 0 } ^ { ( k ) } \sim q _ { \phi } ( z _ { 0 } \mid x ) , \qquad \log { w } ^ { ( k ) } = \log { p _ { \theta } ( x \mid z _ { 0 } ^ { ( k ) } ) } + \log { p _ { \psi } ( z _ { 0 } ^ { ( k ) } ) } - \log { q _ { \phi } ( z _ { 0 } ^ { ( k ) } \mid x ) } .
$$

Then the ELBO-style and IWAE-style estimators are

$$
\log \widehat { p } _ { \mathrm { E L B O } , K } ( x ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \log w ^ { ( k ) } , \qquad \log \widehat { p } _ { \mathrm { I W A E } , K } ( x ) = \log \left( \frac { 1 } { K } \sum _ { k = 1 } ^ { K } e ^ { \log w ^ { ( k ) } } \right) .
$$

The IWAE estimator is generally tighter. When IWAE is used for scoring, the corresponding perplexity is

$$
{ \mathrm { P P L } } _ { { \mathrm { I W A E } } } : = \exp \left( - { \frac { \sum _ { x \in { \mathcal { D } } } \log { \widehat { p } } _ { { \mathrm { I W A E } } , K } ( x ) } { \sum _ { x \in { \mathcal { D } } } | x | } } \right) .
$$

If the decoder itself can provide only a lower bound, then evaluation can report only bound-based PPL rather than exact perplexity.

# B.3 Conditional Probability Estimation and Block-Level Conditional Scoring

In continuation, multiple-choice evaluation, and candidate-response ranking, conditional probability is more important than the unconditional score. Let a full sample be decomposed as

$$
x = ( x ^ { \mathrm { p r e } } , x ^ { \mathrm { r e s } } ) ,
$$

where $x ^ { \mathrm { p r e } }$ is the given prefix and $x ^ { \mathrm { r e s } }$ is the response to be evaluated. The exact conditional log-probability satisfies

$$
\log p ( x ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } ) = \log p ( x ^ { \mathrm { p r e } } , x ^ { \mathrm { r e s } } ) - \log p ( x ^ { \mathrm { p r e } } ) .
$$

Hence one can apply the same unconditional estimator separately to the joint sample and the prefix sample, and then take the difference to obtain a plug-in conditional estimator:

$$
\widehat { \log p _ { \mathrm { c o n d } } } ( x ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } ) : = \mathcal { L } _ { \mathrm { E L B O } } ( x ^ { \mathrm { p r e } } , x ^ { \mathrm { r e s } } ) - \mathcal { L } _ { \mathrm { E L B O } } ( x ^ { \mathrm { p r e } } ) ,
$$

and

$$
\begin{array} { r } { \widehat { \log { p _ { \mathrm { c o n d } , K } } } ( x ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } ) : = \log \widehat { p } _ { \mathrm { I W A E } , K } ( x ^ { \mathrm { p r e } } , x ^ { \mathrm { r e s } } ) - \log \widehat { p } _ { \mathrm { I W A E } , K } ( x ^ { \mathrm { p r e } } ) . } \end{array}
$$

It should be emphasized that both (B.13) and (B.14) are plug-in estimators. They are obtained by substituting unconditional estimators into the conditional identity, and they do not automatically inherit a lower-bound property from the unconditional estimators. Strictly speaking, they should therefore be interpreted as practical estimators of conditional log-likelihood.

For a block-causal prior, the conditional estimator can also be written in a blockwise form aligned with the architecture. Suppose the latent sequence is partitioned into prefix blocks and response blocks:

$$
z = ( z ^ { ( 1 ) } , \ldots , z ^ { ( B ) } , z ^ { ( B + 1 ) } , \ldots , z ^ { ( B + R ) } ) .
$$

If the prior admits a block-level causal factorization, then

$$
p _ { \psi } ( z ) = p _ { \psi } ( z ^ { ( 1 ) } ) \prod _ { b = 2 } ^ { B + R } p _ { \psi } ( z ^ { ( b ) } \mid z ^ { ( < b ) } ) .
$$

Conditioned on the prefix latent $z ^ { \mathrm { p r e } } : = z ^ { ( 1 : B ) }$ , the conditional marginal of the response is

$$
p ( \boldsymbol { x } ^ { \mathrm { r e s } } \mid \boldsymbol { x } ^ { \mathrm { p r e } } , \boldsymbol { z } ^ { \mathrm { p r e } } ) = \int p _ { \boldsymbol { \theta } } ( \boldsymbol { x } ^ { \mathrm { r e s } } \mid \boldsymbol { x } ^ { \mathrm { p r e } } , \boldsymbol { z } ^ { \mathrm { p r e } } , \boldsymbol { z } ^ { \mathrm { r e s } } ) p _ { \boldsymbol { \psi } } ( \boldsymbol { z } ^ { \mathrm { r e s } } \mid \boldsymbol { z } ^ { \mathrm { p r e } } ) \mathrm { d } \boldsymbol { z } ^ { \mathrm { r e s } } ,
$$

where $z ^ { \mathrm { r e s } } : = z ^ { ( B + 1 : B + R ) }$ . For any approximate posterior $q _ { \phi } ( z ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } , x ^ { \mathrm { r e s } } , z ^ { \mathrm { p r e } } )$ , one has the conditional ELBO

$$
\begin{array} { r l } & { \log p ( x ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } , z ^ { \mathrm { p r e } } ) \geq E _ { q _ { \phi } ( z ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } , x ^ { \mathrm { r e s } } , z ^ { \mathrm { p r e } } ) } \Big [ \log p _ { \theta } ( x ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } , z ^ { \mathrm { p r e } } , z ^ { \mathrm { r e s } } ) } \\ & { \qquad + \log p _ { \psi } ( z ^ { \mathrm { r e s } } \mid z ^ { \mathrm { p r e } } ) - \log q _ { \phi } ( z ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } , x ^ { \mathrm { r e s } } , z ^ { \mathrm { p r e } } ) \Big ] . } \end{array}
$$

Marginalizing over the true posterior of the prefix latent then yields

$$
p ( x ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } ) = \int p ( x ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } , z ^ { \mathrm { p r e } } ) p ( z ^ { \mathrm { p r e } } \mid x ^ { \mathrm { p r e } } ) \mathrm { d } z ^ { \mathrm { p r e } } .
$$

Thus, a strict conditional probability requires not only marginalization over the response latent, but also an additional integration over the posterior uncertainty of the prefix latent. In practice, two approximations are common: one uses the joint-minus-prefix estimator in (B.12), while the other approximates $p ( z ^ { \mathrm { p r e } } \mid x ^ { \mathrm { p r e } } )$ with the encoder mean or a single encoder sample and then evaluates (B.19).

If the response contains only one new block, the above expression reduces to a single-block response score:

$$
\begin{array} { r l } & { _ { \geqslant } ( x ^ { ( B + 1 ) } ; x ^ { ( 1 ; B ) } , z ^ { ( 1 ; B ) } ) : = E _ { q _ { \phi } ( z ^ { ( B + 1 ) } | x ^ { ( 1 ; B + 1 ) } , z ^ { ( 1 ; B ) } ) } \left[ \log p _ { \theta } ( x ^ { ( B + 1 ) } \mid x ^ { ( 1 ; B ) } , z ^ { ( 1 ; B + 1 ) } ) \right. } \\ & { \qquad \left. + \log p _ { \psi } ( z ^ { ( B + 1 ) } \mid z ^ { ( 1 ; B ) } ) - \log q _ { \phi } ( z ^ { ( B + 1 ) } \mid x ^ { ( 1 ; B + 1 ) } , z ^ { ( 1 ; B ) } ) \right) } \end{array}
$$

This form is especially useful for scoring multiple-choice options, ranking candidate responses, and analyzing whether the separation between candidates is driven mainly by the prior term or the decoder term.

# B.4 Numerical Evaluation of the CNF Prior and Estimation Algorithms

To evaluate $\log p _ { \psi } ( z _ { 0 } ^ { ( k ) } )$ in (B.8), one solves the augmented ODE

$$
\frac { \mathrm { d } } { \mathrm { d } t } \left[ \boldsymbol { \xi } _ { t } \right] = \left[ \begin{array} { c c } { \boldsymbol { v } _ { \psi } ( \boldsymbol { z } _ { t } , t ) } \\ { \boldsymbol { \nabla } \cdot \boldsymbol { v } _ { \psi } ( \boldsymbol { z } _ { t } , t ) } \end{array} \right] , \qquad \left[ \begin{array} { c c } { \boldsymbol { z } _ { 0 } } \\ { \boldsymbol { \ell } _ { 0 } } \end{array} \right] = \left[ \begin{array} { c c } { \boldsymbol { z } _ { 0 } ^ { ( k ) } } \\ { 0 } \end{array} \right] .
$$

Integrating from $t = 0$ to $t = 1$ gives $( z _ { 1 } ^ { ( k ) } , \ell _ { 1 } ^ { ( k ) } )$ , and therefore

$$
\log p _ { \psi } ( z _ { 0 } ^ { ( k ) } ) = \log p _ { 1 } ( z _ { 1 } ^ { ( k ) } ) + \ell _ { 1 } ^ { ( k ) } .
$$

In high dimensions, the divergence can be approximated by the Hutchinson trace estimator:

$$
\nabla \cdot \pmb { v } _ { \psi } ( z _ { t } , t ) = \mathrm { T r } \bigg ( \frac { \partial \pmb { v } _ { \psi } ( z _ { t } , t ) } { \partial z _ { t } } \bigg ) = \mathbb { E } _ { \pmb { \epsilon } } \bigg [ \pmb { \epsilon } ^ { \top } \frac { \partial \pmb { v } _ { \psi } ( z _ { t } , t ) } { \partial z _ { t } } \pmb { \epsilon } \bigg ] .
$$

In practice, one usually fixes the same random vector $\epsilon$ within a single ODE solve so that the induced dynamics remain consistent along that trajectory.

We now present two estimators corresponding to the above formulas. To avoid extra package dependencies, the algorithms are written in directly compilable pseudo-code style.

# Algorithm A.1: Unconditional Log-Likelihood Estimation (ELBO / IWAE)

Input: sequence $x$ ; encoder $q _ { \phi }$ ; decoder $p \theta$ ; CNF prior $p _ { \psi }$ ; sample number $K$ ; estimation mode mode $\in \{ \tt E L B O , I W A E \}$ .

# Procedure:

1. For each $k = 1 , \ldots , K$ , sample $z _ { 0 } ^ { ( k ) } \sim q _ { \phi } ( z _ { 0 } \mid x )$

2. Starting from $( z _ { 0 } ^ { ( k ) } , 0 )$ , solve (B.23) to $t = 1$ and obtain $( z _ { 1 } ^ { ( k ) } , \ell _ { 1 } ^ { ( k ) } )$

3. Compute $\log p _ { \psi } ( z _ { 0 } ^ { ( k ) } ) = \log p _ { 1 } ( z _ { 1 } ^ { ( k ) } ) + \ell _ { 1 } ^ { ( k ) }$ .

4. Compute $\log p _ { \theta } ( x \mid z _ { 0 } ^ { ( k ) } )$ and $\log q _ { \phi } ( \boldsymbol { z } _ { 0 } ^ { ( k ) } \mid x )$

5. Construct the importance weight

$$
\log w ^ { ( k ) } \gets \log p _ { \theta } ( x \mid z _ { 0 } ^ { ( k ) } ) + \log p _ { \psi } ( z _ { 0 } ^ { ( k ) } ) - \log q _ { \phi } ( z _ { 0 } ^ { ( k ) } \mid x ) .
$$

6. If mode = ELBO, return

$$
{ \widehat { \log p } } ( x ) = { \frac { 1 } { K } } \sum _ { k = 1 } ^ { K } \log w ^ { ( k ) } ;
$$

otherwise return

$$
\widehat { \log p } ( x ) = \log \left( \frac { 1 } { K } \sum _ { k = 1 } ^ { K } e ^ { \log w ^ { ( k ) } } \right) .
$$

Output: an unconditional log-likelihood estimate $\widehat { \log p ( x ) }$ for sequence $x$

# Algorithm A.2: Conditional Log-Likelihood Estimation (Joint Minus Prefix)

Input: prefix $x ^ { \mathrm { p r e } }$ ; response $x ^ { \mathrm { r e s } }$ ; sample number $K$ ; estimation mode mode $\in \{ \tt E L B O , I W A E \}$ . Procedure:

1. Construct the full sequence $x ^ { \mathrm { f u l l } } = ( x ^ { \mathrm { p r e } } , x ^ { \mathrm { r e s } } )$ .

2. Call Algorithm A.1 to estimate $\widehat { \log p } ( x ^ { \mathrm { f u l l } } )$ .

3. Call Algorithm A.1 to estimate $\widehat { \log p } ( x ^ { \mathrm { p r e } } )$ .

$$
\widehat { \log p } ( x ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } )  \widehat { \log p } ( x ^ { \mathrm { f u l l } } ) - \widehat { \log p } ( x ^ { \mathrm { p r e } } ) .
$$

Output: a conditional log-likelihood estimate $\widehat { \log p } ( x ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } )$ .

Algorithm A.1 applies to both ELBO-style and IWAE-style estimators. Algorithm A.2 treats it as a black-box unconditional likelihood estimator and combines two calls through the conditional identity. If the task focuses only on the local score of a single newly generated block, then (B.22) can be used directly to rank candidate responses.

# C Comparison Under a Unified Markov Path View

This appendix further develops the mathematical comparison behind the main-text discussion of unified text modeling. The central question is not which method is more diffusion-like at a superficial level, but rather: into what state space and along what type of path does each method factorize the text distribution, and what role does that path play in generation?

# C.1 A Unified Framework for Process-Based Generative Models

To establish a common language, consider the abstract definition of a process-based generative model. Let

$$
Y = ( Y _ { t } ) _ { t \in \mathcal { T } } , \qquad Y _ { t } \in \mathcal { S } ,
$$

where $\boldsymbol { S }$ is the state space and $\tau$ is either discrete time or continuous time. If there exists a family of transition kernels

$$
K _ { t  s } ( \mathrm { d } y _ { t } \mid y _ { s } ) , \qquad s \leq t ,
$$

satisfying the Chapman–Kolmogorov relation, then $Y$ forms a Markov process. If one additionally specifies an emission kernel

$$
e _ { \theta } ( x \mid y _ { t _ { \star } } ) ,
$$

then the text distribution can be written uniformly as

$$
p _ { \theta } ( x ) = \int e _ { \theta } ( x \mid y _ { t _ { \star } } ) p _ { \theta } ( \mathrm { d } Y ) .
$$

Within this framework, the difference among AR, LLaDA, Plaid, and ${ \mathit { c o l a } }$ DLM is not simply whether each admits a stepwise sampling algorithm. The essential distinction is that they factorize the text distribution over different state spaces and assign fundamentally different roles to the underlying paths. More specifically, they differ at least in the following six aspects: state space, state trajectory object, transition mechanism, the role of the path in generation, the point at which the observation is read out, and the tightness of the training objective relative to the true log-likelihood.

This unified view shows that the key question is not which method uses diffusion, but rather what type of distributional complexity is decomposed into what kind of path over what kind of state space, and whether that path is an observation-dependent corruption path or an observation-independent latent prior path.

# C.2 Difference from Autoregressive Models: Why Diffusion

AR models text directly via the chain rule:

$$
p _ { \eta } ^ { \mathrm { A R } } ( x ) = \prod _ { i = 1 } ^ { L } p _ { \eta } ( x _ { i } \mid x _ { < i } ) .
$$

Its negative log-likelihood is

$$
- \log p _ { \eta } ^ { \mathrm { A R } } ( x ) = \sum _ { i = 1 } ^ { L } - \log p _ { \eta } ( x _ { i } \mid x _ { < i } ) .
$$

The advantage of AR is therefore exact token-level likelihood and the most direct training and evaluation protocol. Its limitation comes from the fixed left-to-right conditioning order.

Under the unified path framework, AR can be written using prefix states

$$
S _ { i } : = x _ { 1 : i } , \qquad S _ { 0 } : = \varnothing .
$$

Then $( S _ { i } ) _ { i = 0 } ^ { L }$ forms a time-inhomogeneous Markov chain, whose one-step transition kernel is exactly the AR conditional distribution:

$$
\mathbb { P } ( S _ { i } = s _ { i } \mid S _ { 0 : i - 1 } ) = \mathbb { P } ( S _ { i } = s _ { i } \mid S _ { i - 1 } = s _ { i - 1 } ) = p _ { \eta } ( x _ { i } \mid x _ { < i } ) .
$$

Thus, the core inductive bias of AR is not merely Markovianity, but the fact that conditioning is restricted to the unique prefix filtration

$$
\sigma ( X _ { 1 } ) \subset \sigma ( X _ { 1 : 2 } ) \subset \cdots \subset \sigma ( X _ { 1 : L } ) .
$$

By contrast, the model distribution of ${ \mathit { c o l a } }$ DLM is not defined through a fixed token order, but by latent marginalization:

$$
p ^ { { \mathrm { C o l a ~ D L M } } } ( x ) = \int p _ { \theta } ( x \mid z _ { 0 } ) p _ { \psi } ( z _ { 0 } ) \mathrm d z _ { 0 } .
$$

If the decoder itself is not left-to-right autoregressive, the resulting model distribution is not tied to any pre-specified token ordering. In this sense, the non-AR nature of ${ \mathit { c o l a } }$ DLM comes from two levels: the prior $p _ { \psi } ( z _ { 0 } )$ is defined by continuous transport rather than token-wise chain factorization, and if the decoder is also non-AR, then generation becomes a two-stage procedure of first sampling global semantic latent variables and then realizing text conditionally.

From the optimization viewpoint, both families minimize

$$
\mathrm { K L } \big ( p _ { \mathrm { d a t a } } ( x ) \| p _ { \mathrm { m o d e l } } ( x ) \big ) ,
$$

but they parameterize $p ( x )$ differently. AR uses a chain-rule factorization, whereas ${ \mathit { c o l a } }$ DLM uses latent marginalization. Therefore, the strict meaning of why diffusion here is not to avoid maximum likelihood, but to improve the fit of $p _ { \psi } ( z _ { 0 } )$ to the aggregated posterior through a more expressive continuous prior family.

This can be quantified exactly. For any two candidate priors $p _ { a }$ and $p _ { b }$

$$
\begin{array} { r } { \mathbb { E } _ { \bar { q } _ { \phi } ( z _ { 0 } ) } [ \log p _ { b } ( z _ { 0 } ) - \log p _ { a } ( z _ { 0 } ) ] = { \mathrm { K L } } \big ( \bar { q } _ { \phi } ( z _ { 0 } ) \| p _ { a } ( z _ { 0 } ) \big ) - { \mathrm { K L } } \big ( \bar { q } _ { \phi } ( z _ { 0 } ) \| p _ { b } ( z _ { 0 } ) \big ) . } \end{array}
$$

Hence, whenever the flow/CNF prior is closer to $\bar { q } _ { \phi } ( z _ { 0 } )$ than a simple Gaussian prior, it improves the average ELBO.

# C.3 Difference from Discrete Denoising Language Models: Why Continuous

Discrete denoising language models such as LLaDA operate directly in token space. A typical training objective can be written as

$$
\mathcal { L } _ { \mathrm { M D M } } ( \theta ) : = \mathbb { E } _ { t \sim U [ 0 , 1 ] , x _ { 0 } , x _ { t } } \left[ \frac { 1 } { t } \sum _ { i = 1 } ^ { L } \mathbf { 1 } [ x _ { t } ^ { i } = M ] \big ( - \log p _ { \theta } ( x _ { 0 } ^ { i } \mid x _ { t } ) \big ) \right] ,
$$

and satisfies

$$
- \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x _ { 0 } ) } \log p _ { \theta } ( x _ { 0 } ) \leq { \mathcal { L } } _ { \mathrm { M D M } } ( \theta ) .
$$

Thus, LLaDA learns a token-space distribution through reverse recovery under random mask ratios. Its advantage is that it has no explicit encoder and no latent bottleneck; its limitation is that it always remains in a discrete state space.

More precisely, the forward masking process of LLaDA can be embedded into a continuous-time Markov chain. For a single token, let the state space be $\mathcal { V } \cup \{ M \}$ and define the generator

$$
G _ { \tau } ( a , M ) = \lambda ( \tau ) , \qquad G _ { \tau } ( a , a ) = - \lambda ( \tau ) , \qquad G _ { \tau } ( M , M ) = 0 ,
$$

where $a \in \nu$ . Then the transition probability from any non-mask state to the absorbing mask state is

$$
\mathbb { P } ( X _ { \tau } = M \mid X _ { 0 } = a ) = 1 - \exp \biggl ( - \int _ { 0 } ^ { \tau } \lambda ( u ) \mathrm { d } u \biggr ) .
$$

If the right-hand side is identified with $t \in [ 0 , 1 ]$ , one recovers the forward kernel in which each token is masked with probability $t$ , as used in (C.13). Therefore, LLaDA can be understood rigorously as a reverse recovery problem over a CTMC with discrete states and absorption into the mask state.

By contrast, Cola DLM models $p _ { \psi } ( z _ { 0 } )$ in continuous latent space rather than performing discrete diffusion in token space. This continuous formulation provides three strict advantages. First, continuous-time changeof-variables yields an explicit log-density, namely (A.13). Second, vector-field regression in Euclidean space allows the use of Flow Matching to learn prior geometry directly. Third, continuous latent variables naturally support interpolation, gradient guidance, and representation editing.

Accordingly, the meaning of why continuous is not that continuous representations are inherently superior to discrete ones. Rather, compared with a discrete token-space corruption model, continuous latent space permits a more direct description of probability geometry and reversible transport. Whether this yields actual gains still depends on whether the continuous structure aligns with the low-dimensional semantic organization of the data.

# C.4 Difference from Plaid: Why Latent

Plaid is also a continuous method, but it makes continuous a token-aligned representation that stays close to the original sequence, rather than an explicit semantic latent variable. A typical forward process can be written as

$$
\begin{array} { r } { q ( h _ { 0 } \mid x ) = \mathcal { N } ( \widetilde { x } , \sigma ^ { 2 } ( 0 ) I ) , \qquad q ( h _ { t } \mid h _ { s } ) = \mathcal { N } \big ( h _ { s } , \sigma ^ { 2 } ( t ) - \sigma ^ { 2 } ( s ) \big ) , } \end{array}
$$

where $\widetilde { x } = \mathrm { E m b e d } ( x )$ . Hence, as the noise level approaches zero, the state of Plaid still remains tightly eattached to the observation space. By contrast, Cola DLM explicitly separates global semantic representation from textual realization through

$$
p ( x ) = \int p _ { \theta } ( x \mid z _ { 0 } ) p _ { \psi } ( z _ { 0 } ) \mathrm { d } z _ { 0 } .
$$

This distinction is not merely about where continuity is introduced; it is about how the model is factorized. If one further lets

$$
\begin{array} { r } { q _ { \phi } ( z _ { 0 } \mid x ) = \mathcal { N } ( E ( x ) , \sigma _ { 0 } ^ { 2 } I ) , \qquad \sigma _ { 0 } ^ { 2 }  0 , } \end{array}
$$

and $E ( x )$ is nearly invertible, then ${ \mathit { c o l a } }$ DLM degenerates to the limiting case where $z _ { 0 }$ is almost just a continuousized representation of the original sequence. This limit clarifies the structural relation: Plaid is closer to the near-observation, non-compressive limit of Cola DLM, while the genuinely new ingredient of Cola DLM is the latent decomposition itself.

The mathematical core of this difference is captured by the information decomposition of the average ELBO in (A.28). That decomposition shows that representation learning in Cola DLM splits into three parts: the conditional reconstruction term controls textual realization given the latent, the mutual information $I _ { q } ( X ; Z _ { 0 } )$ controls how much global information about the text is retained, and the KL between the aggregated posterior and the prior controls the difficulty of prior matching. Precisely because this explicit latent variable is introduced, the model can encode global planning, topic, and style constraints into $z _ { 0 }$ , while delegating local token realization to the decoder.

Therefore, the strict answer to why latent is that a latent variable is not merely a continuous relaxation of a discrete object. It is an intermediate random variable that explicitly participates in marginalization, allowing the text generation problem to be decomposed into prior modeling plus conditional realization. This decomposition yields statistical benefits only when the data indeed contain low-rate but high-value semantic structure; this issue is analyzed further in Appendix D.

# C.5 Why Cola DLM Is Not Merely a Latent Reverse-Diffusion LM

To avoid conceptual confusion, we now make the distinction more precise. Suppose a narrow latent diffusion LM is defined as follows: one first specifies a data-conditioned forward noising process on a latent state,

$$
q ( z _ { 0 : T } \mid x ) ,
$$

and then learns its reverse process to gradually recover a latent state used for text generation. The core structure of such a model is therefore the reversal of a data-conditioned forward corruption process.

Under this narrow definition, ${ \mathit { c o l a } }$ DLM is not simply a latent reverse-diffusion LM. First, the continuous path in Cola DLM lies in prior transport:

$$
z _ { 1 } \sim p _ { 1 } ( z _ { 1 } ) , \qquad z _ { 0 } = \Phi _ { 0  1 } ^ { \psi } ( z _ { 1 } ) , \qquad x \sim p _ { \theta } ( x \mid z _ { 0 } ) .
$$

This path first describes how to generate a semantic prior sample in latent space from a simple Gaussian. During generation, it does not depend on any specific observation $x$ . By contrast, the forward paths in Plaid and LLaDA describe how a given sample $x$ is perturbed into a high-noise state. Thus, although all three can be described as path models, the path in ${ \mathit { c o l a } }$ DLM is prior transport, whereas the paths in Plaid and LLaDA are data corruption.

Second, in Cola DLM the encoder appears only in variational inference:

$$
\begin{array} { r } { \log p ( x ) \geq \mathbb { E } _ { q _ { \phi } ( z _ { 0 } \mid x ) } \Bigl [ \log p _ { \theta } ( x \mid z _ { 0 } ) + \log p _ { \psi } ( z _ { 0 } ) - \log q _ { \phi } ( z _ { 0 } \mid x ) \Bigr ] . } \end{array}
$$

It does not participate in the generation procedure (C.21), and is therefore part of the variational posterior. In Plaid and LLaDA, however, the forward process $q$ is part of the model definition because the reverse model is trained against that corruption law. In short, the former $q$ belongs to inference, whereas the latter $q$ belongs to model construction.

Finally, the marginal distribution of ${ \mathit { c o l a } }$ DLM is always defined by the hierarchical latent-variable identity

$$
p ( x ) = \int p _ { \theta } ( x \mid z _ { 0 } ) p _ { \psi } ( z _ { 0 } ) \mathrm { d } z _ { 0 } .
$$

Strictly speaking, Cola DLM is therefore first and foremost a hierarchical latent-variable language model with a CNF prior. Flow or diffusion is only a way to make the prior family more expressive, not the entirety of the model itself.

# D Theoretical Advantages, Applicability, and Limitations

We finally compare Cola DLM with AR, LLaDA, and Plaid under a unified theoretical criterion. The goal is not to argue that a more complex model should be stronger, but to ask under what conditions Cola DLM is expected to be better, and under what conditions it is expected to be worse.

# D.1 A Unified Population-Level Comparison Criterion

For any model class $\mathcal { M }$ , define its irreducible approximation error with respect to the true data distribution $p _ { \mathrm { d a t a } } ( x )$ as

$$
\mathcal { E } ( \mathcal { M } ) : = \operatorname* { i n f } _ { p \in \mathcal { M } } \operatorname { K L } \bigl ( p _ { \mathrm { d a t a } } ( x ) \| p ( x ) \bigr ) .
$$

This quantity measures the best unavoidable mismatch of the model class even after ignoring finite-sample effects and optimization error.

For any model trained by exact negative log-likelihood, one has the standard identity

$$
\operatorname* { i n f } _ { p \in \mathcal { M } } \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } [ - \log p ( x ) ] = H ( p _ { \mathrm { d a t a } } ) + \mathcal { E } ( \mathcal { M } ) ,
$$

where $H ( p _ { \mathrm { d a t a } } ) : = - \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } \log p _ { \mathrm { d a t a } } ( x )$ is the data entropy. Therefore, for exact-NLL models, comparing theoretical limits is equivalent to comparing $\mathcal { E } ( \mathcal { M } )$ .

AR belongs to this class, and its population risk is

$$
\mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } [ - \log p _ { \eta } ^ { \mathrm { A R } } ( x ) ] = H ( p _ { \mathrm { d a t a } } ) + \mathrm { K L } \big ( p _ { \mathrm { d a t a } } \| p _ { \eta } ^ { \mathrm { A R } } \big ) .
$$

By contrast, Cola DLM is trained through the ELBO, so for each sample

$$
- { \mathcal { L } } _ { \mathrm { E L B O } } ( x ) = - \log p _ { \theta , \psi } ( x ) + \operatorname { K L } \left( q _ { \phi } ( z _ { 0 } \mid x ) \parallel p _ { \theta , \psi } ( z _ { 0 } \mid x ) \right) .
$$

Taking expectation over the data distribution gives

$$
\mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } [ - \mathcal { L } _ { \mathrm { E L B O } } ( x ) ] = H ( p _ { \mathrm { d a t a } } ) + \mathrm { K L } \big ( p _ { \mathrm { d a t a } } \| p _ { \theta , \psi } \big ) + \mathcal { G } _ { \mathrm { C o l a ~ D L M } } ^ { \mathrm { i n f e r } } ( \phi , \theta , \psi ) ,
$$

where

$$
\mathcal { G } _ { \mathrm { C o l a \ D L M } } ^ { \mathrm { i n f e r } } ( \phi , \theta , \psi ) : = \mathbb { E } _ { p _ { \mathrm { d a t a } } ( x ) } \mathrm { K L } \big ( q _ { \phi } ( z _ { 0 } \mid x ) \parallel p _ { \theta , \psi } ( z _ { 0 } \mid x ) \big ) \ge 0
$$

is the inference gap.

Similarly, the variational objective of Plaid can be written as true NLL plus a reverse/path-posterior gap, and the mask-denoising objective of LLaDA can be written as true token-space NLL plus a surrogate gap. Hence all four model classes can be unified as

population risk = H(pdata) + model mismatch + additional gap induced by the training objective.

Therefore, theoretical superiority does not follow from model complexity alone. It depends on whether the total statistical burden is smaller.

Define the total statistical burden of Cola DLM as

$$
\Re _ { \mathrm { C o l a \ D L M } } : = \mathcal { E } ( \mathcal { M } _ { \mathrm { C o l a \ D L M } } ) + \operatorname* { i n f } _ { \phi , \theta , \psi } \mathcal { G } _ { \mathrm { C o l a \ D L M } } ^ { \mathrm { i n f e r } } ( \phi , \theta , \psi ) ,
$$

and define analogous quantities for AR, Plaid, and LLaDA. Then, at the population level, Cola DLM is better than a comparison class if and only if its total statistical burden is smaller. For example,

$$
\Re _ { \mathrm { C o l a \ D L M } } < \Re _ { \mathrm { A R } } \iff \mathcal { C } o l a \ D L M \mathrm { ~ i s \ b e t t e r \ t h a n \ A R \ a t \ t h e \ p o p u l a t i o n \ l e v e l . }
$$

This gives a unified and rigorous comparison criterion.

# D.2 Rate-Distortion and the Applicability Boundary of the Latent Bottleneck

The unified criterion still leaves a deeper question unanswered: why can latent decomposition reduce model mismatch in some cases but hurt modeling in others? The answer comes from conditional entropy, mutual information, and rate-distortion structure.

For any encoder $q _ { \phi } ( z _ { 0 } \mid x )$ and decoder $p _ { \theta } ( x \mid z _ { 0 } )$ , under the joint distribution $q ( x , z _ { 0 } ) = p _ { \mathrm { d a t a } } ( x ) q _ { \phi } ( z _ { 0 } \mid x )$ one always has

$$
\mathbb { E } _ { q ( x , z _ { 0 } ) } [ - \log p _ { \theta } ( x \mid z _ { 0 } ) ] \geq H _ { q } ( X \mid Z _ { 0 } ) .
$$

By the mutual-information identity

$$
H _ { q } ( X ) = H _ { q } ( X \mid Z _ { 0 } ) + I _ { q } ( X ; Z _ { 0 } ) ,
$$

together with $q ( x ) = p _ { \mathrm { d a t a } } ( x )$ , it follows that

$$
H _ { q } ( X \mid Z _ { 0 } ) = H ( p _ { \mathrm { d a t a } } ) - I _ { q } ( X ; Z _ { 0 } ) .
$$

Therefore, the reconstruction cost satisfies

$$
\begin{array} { r } { \mathbb { E } _ { q ( \boldsymbol { x } , \boldsymbol { z } _ { 0 } ) } [ - \log p _ { \boldsymbol { \theta } } ( \boldsymbol { x } \mid \boldsymbol { z } _ { 0 } ) ] \geq H ( p _ { \mathrm { d a t a } } ) - I _ { q } ( \boldsymbol { X } ; \boldsymbol { Z } _ { 0 } ) . } \end{array}
$$

This shows that if $I _ { q } ( X ; Z _ { 0 } )$ is small, then the latent retains little information about the text and the lower bound on the best achievable reconstruction loss rises. If $I _ { q } ( X ; Z _ { 0 } )$ is large enough, then the decoder can realize the text more easily given the latent.

To formalize this further, define the representation rate-distortion function

$$
{ \mathcal { D } } ( R ) : = \operatorname* { i n f } _ { \substack { q ( z _ { 0 } | x ) : I _ { q } ( X ; Z _ { 0 } ) \leq R } } \operatorname* { i n f } _ { p _ { \theta } ( x | z _ { 0 } ) } \mathbb { E } _ { q ( x , z _ { 0 } ) } [ - \log p _ { \theta } ( x \mid z _ { 0 } ) ] .
$$

It represents the minimum achievable average reconstruction cost when the latent channel is allowed to transmit at most $R$ nats of information about the text.

If, for a given data distribution, ${ \mathcal { D } } ( R )$ falls quickly to a low level already at small $R$ , then the data admit a low-rate but high-value latent representation, and a latent bottleneck is more likely to help. If substantial reduction in reconstruction error occurs only when $R$ approaches $H ( X )$ , then the data are nearly incompressible and a latent bottleneck is more likely to become an additional burden.

Therefore, whether ${ \mathit { c o l a } }$ DLM benefits from the bottleneck is ultimately governed by the rate-distortion curve of the data itself. If global topics, plans, style, and constraints occupy a relatively low information rate yet strongly determine the final realization, then encoding them into $z _ { 0 }$ is efficient. If nearly every token-level detail is irreducible, then latent compression does not bring genuine statistical benefit.

# D.3 Structured Generation Regime: When Cola DLM Better Matches the True Mechanism

To formalize the above intuition one step further, consider the following structured generation assumption: there exists a global variable $G$ such that the true data distribution can be written as

$$
p _ { \mathrm { d a t a } } ( x ) = \int p ^ { \star } ( x \mid g ) p ^ { \star } ( g ) \mathrm d g ,
$$

where $G$ determines topic, plan, discourse structure, style, or other global constraints, and $p ^ { \star } ( x \mid g )$ realizes these global factors into concrete text. If, in addition,

$$
H ( X \mid G ) \ll H ( X ) , \qquad \dim ( G ) \ll \dim ( E ( X ) ) ,
$$

then the data are in a favorable regime dominated by low-dimensional but high-value latent factors.

In this regime, the inductive bias of Cola DLM is better aligned with the true generation mechanism. The encoder recovers posterior information about $G$ from $x$ , the prior learns a global semantic distribution close to $p ^ { \star } ( g )$ , and the decoder learns textual realization conditioned on the global factor. In other words, the potential advantage of Cola DLM does not come from being more complicated, but from splitting a hard problem into two better-matched subproblems:

$$
\mathrm { l e a r n i n g ~ } p _ { \mathrm { d a t a } } ( x ) { \quad \sim } { \quad } \mathrm { l e a r n i n g ~ } p ^ { \star } ( g ) \mathrm { ~ a n d ~ } p ^ { \star } ( x \mid g ) .
$$

If the encoder can recover information about $G$ with a small inference gap, $p _ { \psi } ( z _ { 0 } )$ can approximate the aggregated latent distribution corresponding to $G$ , and $p _ { \theta } ( x \mid z _ { 0 } )$ can approximate the conditional realization well, then the statistical burden of Cola DLM is determined primarily by the modeling complexity of the low-dimensional global factor rather than that of the full surface sequence. This is the fundamental source of its possible advantage over AR, LLaDA, and Plaid.

# D.4 Three Curves, Theoretical Advantages, and Limitations

In the opposite regime, where (D.16) does not hold, Cola DLM may systematically underperform more direct modeling routes. Typical unfavorable cases include the absence of a low-rate sufficient representation, settings where local surface form itself carries the main semantics, or regimes where a strong token-space model is already adequate. In such cases, the additional costs of ${ \mathit { c o l a } }$ DLM mainly come from three sources:

(i) inference gap GinferCola DLM,   
(ii) the elevated reconstruction lower bound $H ( X \mid Z _ { 0 } )$ induced by the bottleneck,   
(iii) the optimization complexity of jointly training encoder, prior, and decoder.

Hence, the success or failure of Cola DLM can be reduced to a competition among three curves: the representation rate-distortion curve $\mathcal { G } _ { \mathrm { C o l a ~ D L M } } ^ { \mathrm { i n f e r } }$ . Only when all three favor ${ \mathcal { D } } ( R )$ ${ \mathit { c o l a } }$ , the prior-approximation curve, and the inference-gap curve DLM does latent decomposition become a true theoretical advantage.

These conclusions can be summarized more directly as follows.

Theoretical Advantages. Relative to AR, if the decoder is non-autoregressive, then Cola DLM is not constrained by a single left-to-right ordering and can determine global semantics in latent space before realizing text, which gives it a potentially more natural inductive bias for infilling, constraint satisfaction, and global planning. Relative to discrete DLMs, Cola DLM has the geometric structure of continuous latent space and therefore supports explicit log-densities, vector-field regression, interpolation, and latent editing. Relative to Plaid, Cola DLM explicitly introduces semantic compression, allowing the model to learn a low-dimensional global representation. If the data indeed exhibit a hierarchical structure of low-dimensional global semantics and high-dimensional local realization, this inductive bias is closer to the true mechanism. Finally, the information decomposition of the average ELBO makes it natural to analyze the representation, prior, and decoder as separate components.

Theoretical Limitations. Cola DLM inherently suffers from a variational gap:

$$
\begin{array} { r } { \log p ( x ) - \mathcal { L } _ { \mathrm { E L B O } } ( x ) = \mathrm { K L } \big ( q _ { \phi } ( z _ { 0 } \mid x ) \| p ( z _ { 0 } \mid x ) \big ) . } \end{array}
$$

If the encoder is poor, the model pays a systematic likelihood penalty. The latent bottleneck also introduces the risk of posterior collapse: if the decoder is too strong, the latent dimension is too small, or the regularization is too strong, then $I _ { q } ( X ; Z _ { 0 } )$ may become too small and the encoder may stop transmitting useful information. In addition, accurate evaluation of $\log p _ { \psi } ( z _ { 0 } )$ often requires ODE integration and divergence estimation, making evaluation more expensive than in AR models. Importantly, the potential advantage of ${ \mathit { c o l a } }$ DLM does not follow automatically from using latent variables and flows. It relies on the inductive assumption that the data contain a favorable low-dimensional semantic latent structure. This requires experiments to verify.

# E Existence of Global Semantic Structures

The goal of this section is to establish a falsifiable null hypothesis and, based on it, show that if the latent representation consists only of purely local and separable components, then the optimal timestep shift should not exhibit a systematic drift as the latent dimension varies. We then compare the experimentally observed shift drift against this null hypothesis, thereby obtaining a more robust conclusion regarding the existence of shared structures. To avoid conflating conclusions that can be established rigorously with structural models introduced for interpretation, we first formulate an abstract null hypothesis.

Assumption E.1 (Null hypothesis of purely separable representations). Let $d$ denote the latent dimension, let $\delta$ denote the timestep shift, and let $\mathcal { T } _ { d } ( \delta )$ denote the overall performance objective under a fixed semantic metric at latent dimension $d$ . Assume that the following conditions hold:

1. Local separability: there is no cross-dimensional joint structure in the latent representation that affects semantics, and the overall semantic performance is obtained by independently aggregating the contributions of local dimensions;   
2. Homogeneity: each local dimension shares the same optimal temporal scale with respect to shift; changing $d$ only changes the number of independent dimensions and does not alter the functional form of each local subproblem;

3. Additive aggregation: the overall objective can be written as

$$
\mathcal { I } _ { d } ( \boldsymbol { \delta } ) = \sum _ { i = 1 } ^ { d } j _ { i } ( \boldsymbol { \delta } ) \qquad \mathrm { o r } \qquad \mathcal { I } _ { d } ( \boldsymbol { \delta } ) = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } j _ { i } ( \boldsymbol { \delta } ) ,
$$

where all $j _ { i }$ attain their optimum at the same point. More strongly, if homogeneity holds exactly, then there exists a common function $j$ and constants $a _ { d } > 0 , b _ { d }$ such that

$$
\mathcal { T } _ { d } ( \delta ) = a _ { d } j ( \delta ) + b _ { d } .
$$

Assumption E.1 does not presuppose any specific network architecture. Rather, it describes a theoretical limit in which semantics is carried entirely by mutually independent local dimensions. Under this limit, changing the latent dimension can at most modify the scale of the overall objective, but should not change the location of the optimal shift.

Proposition E.2 (Dimension-invariance of the optimal shift under the separable null). Under Assumption $E , 1$ let

$$
\delta _ { d } ^ { \star } \in \arg \operatorname* { m a x } _ { \delta } \mathcal { I } _ { d } ( \delta )
$$

denote the optimal timestep shift at latent dimension $d$ . Then for any two dimensions $d _ { 1 } , d _ { 2 }$ , we have

$$
\arg \operatorname* { m a x } _ { \delta } \mathcal { I } _ { d _ { 1 } } ( \delta ) = \arg \operatorname* { m a x } _ { \delta } \mathcal { I } _ { d _ { 2 } } ( \delta ) .
$$

In particular, if the maximizer is unique, then

$$
\delta _ { d _ { 1 } } ^ { \star } = \delta _ { d _ { 2 } } ^ { \star } .
$$

Therefore, under purely local and separable representations, the optimal timestep shift should not drift systematically with the latent dimension.

Proof. By (E.2), for any $d$ ,

$$
\mathcal { T } _ { d } ( \delta ) = a _ { d } j ( \delta ) + b _ { d } , \qquad a _ { d } > 0 .
$$

Since $b _ { d }$ is independent of $\delta$ and $a _ { d }$ only induces a positive rescaling, it follows that

$$
\arg \operatorname* { m a x } _ { \delta } \mathcal { I } _ { d } ( \delta ) = \arg \operatorname* { m a x } _ { \delta } j ( \delta ) .
$$

Hence the optimal point is independent of $d$ , which proves (E.3) and (E.4).

The key conclusion of Proposition E.2 is that what can be established rigorously is that if no shared or joint structure affecting semantics exists, then the optimal shift should not drift systematically with the latent dimension.

Corollary E.3 (Systematic drift refutes the purely separable null). If experiments show that, as the latent dimension d varies, the optimal shift $\delta ^ { \star } ( d )$ exhibits a stable, monotonic, and reproducible systematic drift, and if this drift cannot be explained by changes in parameter count, insufficient training, optimization instability, or evaluation noise, then Assumption E.1 does not hold.

Proof. By Proposition E.2, if Assumption E.1 holds, then the optimal shift must be identical across all latent dimensions. If experiments repeatedly show that the optimal shift drifts systematically with $d$ , then this contradicts the necessary consequence of the null hypothesis. Therefore, the null hypothesis is rejected.

The boundary of the conclusion in Corollary E.3 is clear: it directly rules out the hypothesis that the latent space is purely local and separable. Consequently, the strictly derivable conclusion is that some cross-dimensional shared or joint structure must exist in the latent representation. If one further wishes to interpret this structure as a global semantic factor, additional modeling assumptions and experimental evidence are required.

An equivalent formulation from an information-theoretic perspective The above conclusion admits a consistent restatement in information-theoretic terms. Let $s$ denote a latent variable associated with high-level semantics, and let $z$ denote the VAE latent. Then the diffusion forward process can be written as

$$
z _ { t } = \alpha _ { t } z + \sigma _ { t } \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) ,
$$

where $\alpha _ { t }$ and $\sigma _ { t }$ are determined by the noise schedule. If we further decompose the latent as

$$
z = \phi ( s ) + u ,
$$

where $\phi ( s )$ denotes the effective signal induced by the semantic variable and $u$ denotes encoding residuals or posterior uncertainty, then the observation received by the DiT at timestep $t$ becomes

$$
z _ { t } = \alpha _ { t } \phi ( s ) + \underbrace { \alpha _ { t } u + \sigma _ { t } \epsilon } _ { \mathrm { t o t a l ~ n o i s e } } .
$$

Hence, what matters to the DiT is not the raw timestep itself, but rather how much effective information about $s$ is still preserved in the latent at that timestep.

Under Assumption E.1, if semantics is carried entirely by local independent dimensions, namely

$$
s = ( s _ { 1 } , \ldots , s _ { d } ) , \qquad z = ( z _ { 1 } , \ldots , z _ { d } ) ,
$$

and if the channels are independent and identically distributed across dimensions, then the additivity of mutual information yields

$$
I ( s ; z _ { t } ) = \sum _ { i = 1 } ^ { d } I ( s _ { i } ; z _ { t , i } ) .
$$

If each local channel has the same information curve with respect to shift, then changing $d$ only rescales the total mutual information linearly and does not change which shift places the effective information in the range most suitable for semantic recovery. From this information-theoretic perspective, the essence of Proposition E.2 is that, under purely separable representations, varying the dimension does not alter the shift corresponding to the optimal semantic information regime.

Conversely, if the optimal shift drifts systematically with $d$ , then the purely additive information structure corresponding to (E.8) can no longer hold. This indicates that the effective semantic information in the latent is not merely the sum of a set of completely homogeneous local subchannels, but is more likely to depend on some shared structure that only emerges after cross-dimensional aggregation.

A shared-factor model and the interpretation of shift scaling laws To further explain why systematic drift may exhibit a pattern similar to the resolution-dependent shift in Stable Diffusion, we introduce a stronger but analytically convenient structural model. Suppose that each latent dimension can be written as

$$
z _ { i } = A _ { i } g + \xi _ { i } , \qquad i = 1 , \ldots , d ,
$$

where $g \in \mathbb { R } ^ { r }$ denotes a low-dimensional shared factor, $A _ { i }$ is a linear observation matrix, and $\xi _ { i }$ is local independent noise. Here, $g$ may be understood as an abstract shared semantic state, without assuming that it is a single scalar quantity.

In the simplest setting, assume that $A _ { i } \equiv A$ and that $\xi _ { i } \sim \mathcal { N } ( 0 , \sigma _ { \xi } ^ { 2 } I )$ are independent across dimensions. Standard results from linear Gaussian models imply that, as the latent dimension $d$ increases, the posterior precision for $g$ grows linearly, while the posterior covariance shrinks as $d ^ { - 1 }$ . Equivalently, the effective signal-to-noise ratio for recovering the shared factor satisfies

$$
\mathrm { S N R } _ { \mathrm { e f f } } ( d ) \propto d .
$$

From an information-theoretic viewpoint, this means that the recoverable information about the shared factor $g$ increases with $d$ . More specifically, if the total effective noise variance at diffusion timestep $t$ is denoted by $\sigma _ { \mathrm { e f f } } ^ { 2 } ( t )$ , then the mutual information about $g$ can be written as

$$
I ( g ; z _ { t } ) = \frac { 1 } { 2 } \log \operatorname * { d e t } \Bigl ( I + \alpha _ { t } ^ { 2 } \Sigma _ { g } ^ { 1 / 2 } A ^ { \top } \bigl ( \sigma _ { \mathrm { e f f } } ^ { 2 } ( t ) I \bigr ) ^ { - 1 } A \Sigma _ { g } ^ { 1 / 2 } \Bigr ) .
$$

Under a scalar or isotropic approximation, this further reduces to

$$
I ( g ; z _ { t } ) \approx \frac { r } { 2 } \log \Bigl ( 1 + c \cdot d \cdot \mathrm { S N R } _ { \mathrm { e f f } } ( t ) \Bigr ) ,
$$

where the constant $c > 0$ is determined by the observation matrix and the prior covariance of the shared factor.

If the effect of a shift $\delta$ is equivalent to translating the log-SNR of the denoising process, and if semantic recovery performance is primarily determined by whether the effective log-SNR of the shared factor falls within an appropriate range, then the shift required to maintain optimal semantic recovery must compensate for the dimension dependence induced by (E.10) or (E.12). This leads to a logarithmic relation of the form

$$
\delta ^ { \star } ( d ) = a \log d + b ,
$$

where the sign of $a$ depends on the specific time parameterization and the convention used to define the shift.

Remark E.4 (Structural homology with resolution-dependent shift). The significance of (E.13) is not that it uniquely identifies a true global variable, but rather that once a shared factor is repeatedly observed across multiple latent dimensions, the mechanism

increasing dimension $\Rightarrow$ stronger effective SNR logarithmic compensation in the optimal shift arises naturally. This is structurally homologous to the intuition underlying resolution-dependent timestep shift in Stable Diffusion. The difference is that, in the present case, what is shared is a joint structure within the latent representation, whereas in the image case the shared content comes from more globally distributed information under higher resolution.

It should be emphasized that the main role of the shared-factor model is to explain the form of the systematic drift, rather than to replace the rigorous contradiction argument above. The latter answers which hypotheses must fail if drift is observed, whereas the former explains what kind of structure such drift is most consistent with.

Relation to VAE logSNR The above analysis also explains another empirical phenomenon: even when the latent dimension is fixed, the optimal timestep shift may still change when the logSNR of the VAE posterior is altered. The reason is that the VAE is not a noiseless encoder. In (E.6), the term $u$ precisely represents posterior uncertainty. If the timestep $t$ is fixed, then the total noise covariance with respect to the semantic variable is

$$
\Sigma _ { \mathrm { n o i s e } } ( t ) = \alpha _ { t } ^ { 2 } \Sigma _ { u } + \sigma _ { t } ^ { 2 } I .
$$

When the VAE logSNR decreases, this typically implies that the posterior variance increases, that is, $\Sigma _ { u }$ becomes larger. Consequently, at the same raw timestep, the effective semantic information preserved in the latent,

$$
I ( s ; z _ { t } ) = \frac { 1 } { 2 } \log \mathrm { d e t } \Bigl ( I + \alpha _ { t } ^ { 2 } \Sigma _ { s } ( \alpha _ { t } ^ { 2 } \Sigma _ { u } + \sigma _ { t } ^ { 2 } I ) ^ { - 1 } \Bigr ) ,
$$

decreases overall. In other words, in a smoother latent space with larger posterior variance, the same diffusion timestep corresponds to a lower effective semantic SNR. Therefore, in order for training and sampling to remain within a similar optimal semantic information regime, the timestep shift must also be recalibrated.

This shows that changing the latent dimension and changing the VAE logSNR are, on the surface, two different operations, but they act on the same core object from an information-theoretic perspective, namely the effective mutual information curve $I ( s ; z _ { t } )$ of the semantic variable along diffusion time. The former mainly changes how shared information is aggregated and amplified across dimensions, whereas the latter mainly changes the effective noise floor at each timestep. Both can cause the optimal information regime to drift along the raw timestep axis.

Why this further points to semantic structure Corollary E.3 alone only implies the existence of shared or joint structure; it does not yet establish that this structure necessarily carries semantic information. To strengthen the conclusion to shared structure that affects semantics, one needs task-level selective evidence. A particularly informative situation is the following:

1. reconstruction quality remains relatively stable across different shifts;   
2. semantic downstream metrics are highly sensitive to the shift;   
3. and the optimal shift exhibits systematic drift with the latent dimension.

If these three conditions hold simultaneously, then it is no longer plausible to explain the phenomenon merely by saying that the shift changes the difficulty of local token realization. A more reasonable interpretation is that the shift is adjusting the noise calibration of the latent prior with respect to some shared high-level structure, and that this shared structure is precisely involved in semantic organization.

From an information-theoretic standpoint, this means that changing the shift does not substantially alter the decoder’s ability to recover local details, but does substantially change the preservation regime of the shared information required by high-level tasks during the diffusion process. Therefore, what is being recalibrated is more likely not local reconstruction error, but the semantic component of mutual information at a higher level.

In other words, if reconstruction remains consistently strong while semantic metrics vary substantially, then the main quantity being affected is not the decoder’s local realization capability, but rather the alignment of the latent prior with high-level shared information. Combined with the contradiction result above, this yields a stronger yet still robust conclusion: the latent space contains shared or joint structures that affect semantics.

Remark E.5 (Conclusions that are rigorously supported in this section). Under a broad class of purely local and separable latent hypotheses, the optimal timestep shift should not drift systematically with the latent dimension. The stable drift observed in experiments refutes this null hypothesis and therefore implies the existence of cross-dimensional shared or joint structures in the latent space. Furthermore, if this drift is manifested primarily in semantic metrics rather than reconstruction metrics, then one can further conclude that such shared or joint structures indeed participate in high-level semantic organization.

These conclusions are sufficient to support the central hypothesis of this work: if such global or semi-global semantic shared structures do exist in the latent space, then calibrating the noise schedule at the level of the latent prior may offer a more substantial advantage than methods based purely on local modeling.

# F Theoretical Analysis of Likelihood Estimation Bias

This section provides a more complete theoretical analysis of the phenomenon discussed in Section 5.1 of the main text. The central question is why Cola DLM can already generate reasonably high-quality text in free-generation settings, while its PPL / conditional score estimated via the ELBO or a plug-in conditional estimator remains substantially inflated. We argue that this is not an incidental engineering artifact, but rather a structural consequence of the misalignment among the training objective, the underlying probabilistic object, and the local geometry in continuous latent generative models.

To avoid conflating the conditional marginal probability of an entire response with the local prior density evaluated at a particular latent point, we explicitly distinguish the following two objects throughout. Let

$$
x = ( x ^ { \mathrm { p r e } } , x ^ { \mathrm { r e s } } ) ,
$$

where $x ^ { \mathrm { p r e } }$ denotes the prefix / context and $x ^ { \mathrm { r e s } }$ denotes the response, and let $c$ denote the conditional information induced by the prefix. Then the exact conditional marginal probability is

$$
p ( x ^ { \mathrm { r e s } } \mid c ) = \int p _ { \theta } ( x ^ { \mathrm { r e s } } \mid z , c ) p _ { \psi } ( z \mid c ) \mathrm { d } z ,
$$

whereas the local conditional score corresponding to the response-level score in the main text is

$$
\begin{array} { r } { S _ { \mathrm { r e s p } } ( x ) = \mathbb { E } _ { q _ { \phi } ( z | x , c ) } \Big [ \log p _ { \theta } ( x ^ { \mathrm { r e s } } \mid z , c ) + \log p _ { \psi } ( z \mid c ) - \log q _ { \phi } ( z \mid x , c ) \Big ] . } \end{array}
$$

Equation (F.1) is the exact conditional marginal likelihood, whereas Equation (F.2) is a local estimator that is more directly accessible in analysis. The gap between them is precisely the probabilistic foundation of the discussion that follows.

F.1 Why the Flow Matching Training Objective Is Naturally Misaligned with Conditional PPL

# F.1.1 Flow Matching optimizes vector-field regression rather than gold-specific local log-density

With the encoder and decoder fixed, the prior-learning subproblem can be understood as fitting $p _ { \psi } ( z \mid c )$ to the conditional latent distribution induced jointly by the data and the posterior. In practice, however, Flow Matching does not directly maximize the conditional log-density at a particular gold latent point. Instead, it learns a time-dependent vector field $v _ { \psi } ( z , t , c )$ . Its typical objective can be written as

$$
\mathcal { L } _ { \mathrm { F M } } ( \psi ) = \mathbb { E } \Big [ | | v _ { \psi } ( Z _ { t } , t , c ) - U ^ { \star } | | ^ { 2 } \Big ] ,
$$

where $U ^ { \star }$ is the target velocity field induced by a bridging path or a reference flow. Equation (F.3) shows that Flow Matching directly regresses a dynamical system, rather than the local conditional log-density at a specific gold latent point.

More importantly, the pointwise optimum of the squared loss is not the most likely velocity field in a mode-seeking sense, but the conditional mean velocity field.

Proposition F.1 (The pointwise optimum of Flow Matching is the conditional mean velocity). For fixed $( z , t , c )$ , define

$$
Y : = U ^ { \star } \mid ( Z _ { t } = z , t , c ) , \qquad f : = v _ { \psi } ( z , t , c ) .
$$

Then the conditional risk

$$
\mathcal { R } ( f ; z , t , c ) : = \mathbb { E } \big [ \| f - Y \| ^ { 2 } \mid Z _ { t } = z , t , c \big ]
$$

admits the unique minimizer

$$
f ^ { \star } ( z , t , c ) = \mathbb { E } [ Y \mid Z _ { t } = z , t , c ] .
$$

Proof. For any $f$ , we have

$$
\dot { \mathbf { \xi } } - Y = ( f - \mathbb { E } [ Y \mid Z _ { t } = z , t , c ] ) - ( Y - \mathbb { E } [ Y \mid Z
$$

Expanding the square, taking the conditional expectation, and using the fact that the cross term vanishes, we obtain

$$
\mathscr { R } ( f ; z , t , c ) = \| f - \mathbb { E } [ Y \mid Z _ { t } = z , t , c ] \| ^ { 2 } + \mathbb { E } \big [ \| Y - \mathbb { E } [ Y \mid Z _ { t } = z , t , c ] \| ^ { 2 } \mid Z _ { t } = z , t , c \big ] .
$$

The second term is independent of $f$ , so the unique optimum is attained at

$$
f = \mathbb { E } [ Y \mid Z _ { t } = z , t , c ] .
$$

The meaning of Proposition F.1 is direct: Flow Matching learns the conditional mean dynamics, rather than a gold-specific local calibration specialized to the posterior tube of a particular response. As long as the conditional response distribution under a given context is itself multimodal or broad-peaked, Flow Matching is more naturally biased toward learning an average transport map that moves total probability mass into a reasonable region, rather than automatically guaranteeing local density calibration around a particular sample.

# F.1.2 Why the prior mean may be far from the ground truth while generation remains good

This point can be formalized under multimodal conditional distributions. For a fixed context $c$ , suppose

$$
p _ { \psi } ( z \mid c ) = \sum _ { m = 1 } ^ { M } \pi _ { m } ( c ) \mathcal { N } ( \mu _ { m } , \Sigma _ { m } ) , \qquad \sum _ { m = 1 } ^ { M } \pi _ { m } ( c ) = 1 ,
$$

and for a particular sample, its posterior is mainly concentrated on the $m ^ { \star }$ -th mode, namely

$$
q _ { \phi } ( z \mid x , c ) \approx \mathcal { N } ( \mu _ { m ^ { \star } } , \Sigma _ { q } ) .
$$

Then the global mean of the prior is

$$
\bar { \mu } _ { p } ( c ) = \sum _ { m = 1 } ^ { M } \pi _ { m } ( c ) \mu _ { m } .
$$

Corollary F.2 (Mean displacement under multimodal conditional distributions). Under the setting of Equations (F.6)–(F.8),

$$
\| \bar { \mu } _ { p } ( c ) - \mu _ { m ^ { \star } } \| \leq \sum _ { m \neq m ^ { \star } } \pi _ { m } ( c ) \| \mu _ { m } - \mu _ { m ^ { \star } } \| .
$$

If the modes are well separated, then the prior mean may deviate substantially from the latent of a particular ground-truth sample. Nevertheless, as long as the major mass of these modes lies within a decoder-good region, free-generation quality can still remain good.

Proof. By

$$
\bar { \mu } _ { p } ( c ) - \mu _ { m ^ { \star } } = \sum _ { m = 1 } ^ { M } \pi _ { m } ( c ) ( \mu _ { m } - \mu _ { m ^ { \star } } ) = \sum _ { m \neq m ^ { \star } } \pi _ { m } ( c ) ( \mu _ { m } - \mu _ { m ^ { \star } } )
$$

and the triangle inequality, the result follows immediately.

Therefore, a large distance between the prior mean and the ground-truth latent does not automatically imply that the prior has failed to learn the distribution. A more plausible explanation is often that the context-level conditional distribution is itself multimodal, while the posterior selects only one narrower gold-specific mode among them.

# F.2 Why This Further Leads to Good Generation but Poor PPL

F.2.1 Generation quality and conditional PPL correspond to two different geometric objects

Fix a context $c$ , let $r ( x ^ { \mathrm { r e s } } ; c )$ denote some external generation-quality function, and let $\tau$ be a threshold. Define the decoder-good region by

$$
A _ { \mathrm { g o o d } } ^ { \tau } ( c ) : = \left\{ z : \mathbb { E } _ { x ^ { \mathrm { r e s } } \sim p _ { \theta } ( \cdot \vert z , c ) } \big [ r ( x ^ { \mathrm { r e s } } ; c ) \big ] \geq \tau \right\} ,
$$

and for a particular sample $x$ and a constant $\rho > 0$ , define its posterior typical set by

$$
A _ { x } ^ { \rho } : = \{ z : \ q _ { \phi } ( z \mid x , c ) \geq \rho \} .
$$

If $A _ { x } ^ { \rho }$ is narrow and sharp, we may refer to it as the gold tube of that sample. In general, $A _ { x } ^ { \rho }$ is only a more local and sharper subset inside $A _ { \mathrm { g o o d } } ^ { \tau } ( c )$ , namely

$$
A _ { x } ^ { \rho } \subseteq A _ { \mathrm { g o o d } } ^ { \tau } ( c ) \qquad \mathrm { o r ~ a t ~ l e a s t } \qquad A _ { x } ^ { \rho } \mathrm { i s ~ a ~ m o r e ~ l o c a l ~ s u b r e g i o n ~ i n s i d e } ~ A _ { \mathrm { g o o d } } ^ { \tau } ( c ) .
$$

This distinction shows that good generation and good conditional PPL are not the same requirement.

Proposition F.3 (Good generation and poor conditional score can hold simultaneously). Fix a context $c$ . Suppose there exist constants $\alpha , \Delta > 0$ and thresholds $\tau , \rho$ such that

$$
\begin{array} { r l } { p _ { \psi } \left( A _ { \mathrm { g o o d } } ^ { \tau } ( c ) \mid c \right) \geq \alpha , } & { } \\ { q _ { \phi } ( A _ { x } ^ { \rho } \mid x , c ) \approx 1 , } & { } \\ { \log q _ { \phi } ( z \mid x , c ) - \log p _ { \psi } ( z \mid c ) \geq \Delta , \qquad \forall z \in A _ { x } ^ { \rho } , } \end{array}
$$

and the decoder reconstruction term is uniformly bounded on $A _ { x } ^ { \rho }$ :

$$
\log p _ { \theta } ( x ^ { \mathrm { r e s } } \mid z , c ) \le B , \qquad \forall z \in A _ { x } ^ { \rho } .
$$

Then:

1. under free sampling from the prior, at least an $\alpha$ fraction of the mass falls into the decoder-good region, so generation quality may already be good;

2. but the conditional score satisfies

$$
S _ { \mathrm { r e s p } } ( x ) \le B - \Delta + o ( 1 ) ,
$$

and therefore the conditional ELBO or plug-in PPL may still be substantially biased whenever $\Delta$ is sufficiently large.

Proof. The first claim follows directly from Equation (F.13). For the second claim, rewrite Equation (F.2) as

$$
\begin{array} { r } { S _ { \mathrm { r e s p } } ( x ) = \mathbb { E } _ { q _ { \phi } } \Big [ \log p _ { \theta } ( x ^ { \mathrm { r e s } } \mid z , c ) - \left( \log q _ { \phi } ( z \mid x , c ) - \log p _ { \psi } ( z \mid c ) \right) \Big ] . } \end{array}
$$

By Equation (F.14), the mass of $q _ { \phi }$ outside $A _ { x } ^ { \rho }$ is $o ( 1 )$ . Combining this with Equations (F.15) and (F.16 yields

$$
S _ { \mathrm { r e s p } } ( x ) \le B - \Delta + o ( 1 ) .
$$

The essence of Proposition F.3 is that generation quality mainly requires the prior mass to enter $A _ { \mathrm { g o o d } } ^ { \tau } ( c )$ , whereas conditional PPL additionally requires the prior to assign sufficiently high local density on the posterior tube of a particular gold response. The former is a coverage requirement, whereas the latter is a calibration requirement.

# F.2.2 Why good reconstruction does not imply good PPL

Equation (F.2) can also be written as

$$
S _ { \mathrm { r e s p } } ( x ) = \underbrace { \mathbb { E } _ { q _ { \phi } ( z | x , c ) } [ \log p _ { \theta } ( x ^ { \mathrm { r e s } } \mid z , c ) ] } _ { R ( x ; c ) } - \underbrace { \mathrm { K L } \bigl ( q _ { \phi } ( z \mid x , c ) \parallel p _ { \psi } ( z \mid c ) \bigr ) } _ { G ( x ; c ) } .
$$

Proposition F.4 (Good reconstruction does not imply good conditional PPL). If there exists a sequence of samples $\{ x _ { n } \}$ such that

$$
R ( x _ { n } ; c _ { n } ) \to R _ { \operatorname* { m a x } } ,
$$

while at the same time

$$
G ( x _ { n } ; c _ { n } )  + \infty \qquad o r r e m a i n s ~ s i g n i f i c a n t l y ~ p o s i t i v e ,
$$

then the corresponding conditional score may still be poor.

Proof. Equation (F.18) directly gives

$$
\begin{array} { r } { S _ { \mathrm { r e s p } } ( x _ { n } ) = R ( x _ { n } ; c _ { n } ) - G ( x _ { n } ; c _ { n } ) . } \end{array}
$$

Therefore, as long as the posterior–prior gap remains large, the conditional score can still be low even when reconstruction is already strong. □

This shows that, in continuous latent language models, good reconstruction near the ground truth is not sufficient to imply good PPL. What truly determines PPL also includes the degree of local matching between the posterior and the prior.

# F.3 Even If the Prior Mean Is Close to the Ground Truth, PPL May Still Be Poor

The previous subsection clarified the distinction between large-region coverage and gold-tube calibration. However, even if the problem is further restricted to the case where the center of the prior is already approximately aligned with the target, PPL may still remain poor. The reason is that local covariance, orientation, curvature, and volume terms also enter the conditional score.

F.3.1 Under a local Gaussian approximation, center alignment is necessary but not sufficient

Suppose that, in a neighborhood of a gold sample,

$$
\begin{array} { r } { q _ { \phi } ( z \mid x , c ) \approx \mathcal { N } ( \mu _ { q } , \Sigma _ { q } ) , \qquad p _ { \psi } ( z \mid c ) \approx \mathcal { N } ( \mu _ { p } , \Sigma _ { p } ) . } \end{array}
$$

Then the corresponding KL divergence is

$$
\mathrm { K L } ( q _ { \phi } | | p _ { \psi } ) = \frac { 1 } { 2 } \Big [ \mathrm { t r } \big ( \Sigma _ { p } ^ { - 1 } \Sigma _ { q } \big ) + ( \mu _ { p } - \mu _ { q } ) ^ { \top } \Sigma _ { p } ^ { - 1 } ( \mu _ { p } - \mu _ { q } ) - d + \log \frac { \operatorname* { d e t } \Sigma _ { p } } { \operatorname* { d e t } \Sigma _ { q } } \Big ] .
$$

Proposition F.5 (Center alignment alone is insufficient to guarantee good PPL). Under Equation (F.19), even if $\mu _ { p } \approx \mu _ { q }$ , as long as $\Sigma _ { p }$ and $\Sigma _ { q }$ are mismatched in scale, orientation, or volume, $\mathrm { K L } ( q _ { \phi } | | p _ { \psi } )$ may still be substantially large, and hence the conditional score may still be poor.

Proof. Equation (F.20) shows that, beyond the center-discrepancy term, there are also the explicit scale- and volume-related terms

$$
\mathrm { t r } ( \Sigma _ { p } ^ { - 1 } \Sigma _ { q } ) \quad \mathrm { a n d } \quad \log \frac { \mathrm { d e t } \Sigma _ { p } } { \mathrm { d e t } \Sigma _ { q } } .
$$

Even if the center-discrepancy term is small, the KL divergence can remain significantly positive whenever these latter terms are mismatched. □

Therefore, in continuous density models, it is incorrect to claim that it suffices to move the prior mean close to the ground-truth latent. PPL requires a stronger form of local probabilistic calibration: not only must the centers be close, but the local covariance, local volume, and local orientation must also be matched.

# F.3.2 An overly sharp posterior further amplifies this issue

If the posterior is highly concentrated, so that $\Sigma _ { q }$ is small, then the scale and volume terms in Equation (F.20) become even more sensitive. Intuitively, the gold posterior is no longer a thick region, but rather resembles a very thin tube. In this regime, even a slight deviation of the prior can induce a substantially amplified local log-density gap. Therefore,

but

This also explains why, in continuous latent language models, PPL behaves more like a local density calibration metric rather than merely a generation quality metric.

# F.4 Why PPL and Generation Are Usually More Strongly Correlated in Discrete Models Such as AR / LLaDA

# F.4.1 AR: training, evaluation, and generation share the same discrete probabilistic object

For autoregressive models, the conditional distribution admits the exact chain decomposition

$$
\log p _ { \eta } ^ { \mathrm { A R } } ( x ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } ) = \sum _ { i = 1 } ^ { N _ { \mathrm { r e s } } } \log p _ { \eta } ( x _ { i } ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } , x _ { < i } ^ { \mathrm { r e s } } ) ,
$$

and therefore the PPL is

$$
\mathrm { P P L } _ { \mathrm { A R } } ( x ) = \exp \left( - \frac { 1 } { N _ { \mathrm { r e s } } } \sum _ { i = 1 } ^ { N _ { \mathrm { r e s } } } \log p _ { \eta } ( x _ { i } ^ { \mathrm { r e s } } \mid x ^ { \mathrm { p r e } } , x _ { < i } ^ { \mathrm { r e s } } ) \right) .
$$

The next-token cross-entropy minimized during training is precisely the negative log of the terms in Equation (F.24), and generation also directly queries these same conditional probabilities. Hence, in AR models,

training objective $=$ the object evaluated by PPL $=$ the object used during generation.

This is why PPL and generation quality are usually most strongly correlated in autoregressive models

# F.4.2 Discrete models optimize the probability mass of the correct event more directly

For a discrete classifier, if the gold token at a given position is $y$ and the model outputs a probability vector $\pi \in \Delta ^ { V - 1 }$ , then the cross-entropy is

$$
\mathcal { L } _ { \mathrm { C E } } = - \log \pi _ { y } .
$$

What is directly optimized here is the probability mass assigned to the correct event itself. In contrast, Cola DLM requires the computation of an integral over the continuous latent space:

$$
p ( x ^ { \mathrm { r e s } } \mid c ) = \int \exp \Bigl ( \log p _ { \theta } ( x ^ { \mathrm { r e s } } \mid z , c ) + \log p _ { \psi } ( z \mid c ) \Bigr ) \mathrm { d } z .
$$

Accordingly, its value depends not only on whether the mode center is correctly located, but also explicitly on local volume and curvature. If one performs a Laplace approximation of

$$
F _ { x , c } ( z ) : = \log p _ { \theta } ( x ^ { \mathrm { r e s } } \mid z , c ) + \log p _ { \psi } ( z \mid c )
$$

at its maximizer, then

$$
\log p ( x ^ { \mathrm { r e s } } \mid c ) \approx F _ { x , c } ( z _ { x , c } ^ { \star } ) + \frac { d } { 2 } \log ( 2 \pi ) - \frac { 1 } { 2 } \log \operatorname* { d e t } H _ { x , c } ,
$$

where

$$
H _ { x , c } : = - \nabla _ { z } ^ { 2 } F _ { x , c } ( z _ { x , c } ^ { \star } ) .
$$

This shows that conditional likelihood in continuous latent models depends not only on mode height, but also on local geometry.

Therefore, the more accurate statement is not merely that discrete models are inherently simpler, but rather that training, evaluation, and generation in discrete models all act more directly on the same discrete probabilistic object, whereas continuous latent models are additionally mediated by latent-space integration, posterior--prior matching, and decoder-emission compatibility.

# F.4.3 LLaDA: the main discrepancy arises at the discrete observation level rather than the latent calibration level

LLaDA is not identical to AR, but compared with Cola DLM, its recovery object is still the discrete observed state itself, rather than first encoding the sample-dependent posterior into a continuous latent and then requiring an independent conditional prior to match it pointwise. Therefore, even though LLaDA may also exhibit a surrogate gap, its main discrepancy is still concentrated within the discrete token space, rather than in the posterior–prior density collision characteristic of Cola DLM. For this reason, the correlation between PPL and generation is usually still stronger in LLaDA than in continuous latent models.

# G A Unified Theoretical View of Noise Schedule, logSNR, and Flow Matching

This section consolidates the preceding discussion of noise schedule, Flow Matching, latent-dimension-dependent shift drift, and semantic information calibration into a single theoretical chain. The key question is not why a particular empirical hyperparameter works, but rather: what mathematical object the noise schedule controls in Cola DLM, and how this object enters the training objective and ultimately affects semantic modeling quality. More specifically, we establish the following four points:

1. the noise schedule and the logSNR curve are two equivalent parameterizations of the same forward noising process; 2. in Flow Matching, the effect of logSNR on the loss arises both through the change of sampling measure inside the expectation and through the scale change of the supervised target velocity field; 3. uniform-timestep training is generally not equivalent to uniform-logSNR training; 4. the essential role of the optimal timestep shift is not to translate an abstract time label, but to recalibrate training and sampling toward a more appropriate effective semantic information regime.

# G.1 Noise Schedule and logSNR as Equivalent Parameterizations

We begin with the standard forward process used throughout the main text and appendix:

$$
z _ { t } = \alpha _ { t } z + \sigma _ { t } \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) ,
$$

where $\alpha _ { t }$ and $\sigma _ { t }$ are determined by the noise schedule. The signal-to-noise ratio is naturally defined by

$$
\mathrm { S N R } ( t ) = \frac { \alpha _ { t } ^ { 2 } } { \sigma _ { t } ^ { 2 } } , \qquad \lambda ( t ) : = \log \mathrm { S N R } ( t ) = \log \frac { \alpha _ { t } ^ { 2 } } { \sigma _ { t } ^ { 2 } } .
$$

Under the standard normalization

$$
\alpha _ { t } ^ { 2 } + \sigma _ { t } ^ { 2 } = 1 ,
$$

one has

$$
\alpha _ { t } ^ { 2 } = \mathrm { s i g m o i d } ( \lambda ( t ) ) , \qquad \sigma _ { t } ^ { 2 } = \mathrm { s i g m o i d } ( - \lambda ( t ) ) .
$$

Therefore, specifying $\lambda ( t )$ uniquely determines $\left( \alpha _ { t } , \sigma _ { t } \right)$ , and conversely $\left( \alpha _ { t } , \sigma _ { t } \right)$ uniquely determines $\lambda ( t )$ . In this precise sense,

$$
\Big | \mathrm { n o i s e ~ s c h e d u l e } \longleftrightarrow \mathrm { l o g S N R ~ c u r v e } \Big |
$$

so the two are not related through a secondary transformation of the loss; they are simply two equivalent parameterizations of the same forward noising geometry.

Now consider a shifted logSNR curve

$$
\lambda _ { \delta } ( t ) = \lambda ( t ) + \delta .
$$

The corresponding signal and noise coefficients become

$$
\alpha _ { t , \delta } ^ { 2 } = \mathrm { s i g m o i d } ( \lambda ( t ) + \delta ) , \qquad \sigma _ { t , \delta } ^ { 2 } = \mathrm { s i g m o i d } ( - \lambda ( t ) - \delta ) .
$$

Hence the role of timestep shift is not merely to translate the raw timestep axis, but rather to remap the same raw timestep to a different logSNR regime:

# G.2 Reparameterization of the Flow Matching Objective

As discussed above, the strict probabilistic objective of Cola DLM is always ELBO maximization together with prior matching, while Flow Matching is the practical mechanism used to learn the same prior family. To understand how the noise schedule enters training, it is therefore sufficient to analyze the mean-squared vector-field regression objective.

To avoid a notation clash with the coefficients $\left( \alpha _ { t } , \sigma _ { t } \right)$ in Equation (G.1), let the interpolation coefficient of the bridge path be denoted by $\beta ( t )$ . Consider

$$
z _ { t } = ( 1 - \beta ( t ) ) z _ { 0 } + \beta ( t ) z _ { 1 } , \qquad u _ { t } ( z _ { 0 } , z _ { 1 } ) = \frac { \mathrm { d } z _ { t } } { \mathrm { d } t } = \dot { \beta } ( t ) ( z _ { 1 } - z _ { 0 } ) ,
$$

and write the single-block Flow Matching objective as

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , z _ { 0 } , z _ { 1 } } \Big [ \| v _ { \psi } ( z _ { t } , t ) - u _ { t } ( z _ { 0 } , z _ { 1 } ) \| _ { 2 } ^ { 2 } \Big ] , \qquad t \sim U [ 0 , 1 ] .
$$

Equivalently,

$$
\mathcal { L } _ { \mathrm { F M } } = \int _ { 0 } ^ { 1 } \ell ( t ) \mathrm { d } t , \qquad \ell ( t ) : = \mathbb { E } _ { z _ { 0 } , z _ { 1 } } \left[ \| v _ { \psi } ( z _ { t } , t ) - u _ { t } ( z _ { 0 } , z _ { 1 } ) \| _ { 2 } ^ { 2 } \right] .
$$

Now let

$$
\lambda = \Lambda ( t ) : = \log { \mathrm { S N R } ( t ) } ,
$$

and assume that $\Lambda$ is strictly monotone and differentiable, so that the inverse mapping $t = T ( \lambda ) = \Lambda ^ { - 1 } ( \lambda )$ exists. By the change-of-variables formula,

$$
\mathrm { d } t = \left| { \frac { \mathrm { d } t } { \mathrm { d } \lambda } } \right| \mathrm { d } \lambda ,
$$

which yields

$$
\mathcal { L } _ { \mathrm { F M } } = \int \ell ( T ( \lambda ) ) \left| \frac { \mathrm { d } t } { \mathrm { d } \lambda } \right| \mathrm { d } \lambda .
$$

Therefore, if training samples $t$ uniformly, then the induced density on the logSNR axis is

$$
p _ { \lambda } ( \lambda ) = p _ { t } ( T ( \lambda ) ) \left| \frac { \mathrm d t } { \mathrm d \lambda } \right| = \left| \frac { \mathrm d t } { \mathrm d \lambda } \right| ,
$$

where we used $p _ { t } ( t ) \equiv 1$ .

Proposition G.1 (Uniform timestep is generally not equivalent to uniform logSNR). Let $\lambda = \Lambda ( t )$ be $a$ strictly monotone differentiable function, and suppose training uses $t \sim U [ 0 , 1 ]$ . Then the sampling density on the logSNR axis is

$$
p _ { \lambda } ( \lambda ) = \left| \frac { \mathrm { d } t } { \mathrm { d } \lambda } \right| .
$$

Therefore, unless $\Lambda ( t )$ is affine, uniform-timestep training is not equivalent to uniform-logSNR training.

Proof. By Equation (G.14), the pushforward of the uniform measure on $t$ to the $\lambda$ axis has density

$$
p _ { \lambda } ( \lambda ) = p _ { t } ( T ( \lambda ) ) \left| { \frac { \mathrm { d } t } { \mathrm { d } \lambda } } \right| .
$$

Since $p _ { t } ( t ) \equiv 1$ , Equation (G.16) follows immediately. The density $p _ { \lambda } ( \lambda )$ is constant if and only if $\frac { \mathrm { d } \lambda } { \mathrm { d } t }$ is constant, namely if and only if $\Lambda ( t )$ is affine. Hence the two training schemes are not equivalent in general. □

The above proposition shows only that the sampling measure inside the expectation changes after reparameterization by logSNR. This is not yet the whole story. The supervised target velocity field itself also rescales under the new time coordinate. Let

$$
z _ { \lambda } : = z _ { T ( \lambda ) } , \qquad u _ { \lambda } : = \frac { \mathrm { d } z _ { \lambda } } { \mathrm { d } \lambda } .
$$

Then, by the chain rule,

$$
u _ { t } = { \frac { \mathrm { d } z _ { t } } { \mathrm { d } t } } = { \frac { \mathrm { d } z _ { \lambda } } { \mathrm { d } \lambda } } \cdot { \frac { \mathrm { d } \lambda } { \mathrm { d } t } } = { \dot { \lambda } } ( t ) u _ { \lambda } , \qquad { \dot { \lambda } } ( t ) : = { \frac { \mathrm { d } \lambda } { \mathrm { d } t } } .
$$

Substituting this identity into the loss gives

$$
\mathcal { L } _ { \mathrm { F M } } = \int \mathbb { E } _ { z _ { 0 } , z _ { 1 } } \left[ \left\| v _ { \psi } ( z _ { \lambda } , T ( \lambda ) ) - \dot { \lambda } ( T ( \lambda ) ) u _ { \lambda } \right\| _ { 2 } ^ { 2 } \right] \left| \frac { \mathrm { d } t } { \mathrm { d } \lambda } \right| \mathrm { d } \lambda .
$$

Hence logSNR affects the Flow Matching loss in two distinct ways:

1. it changes the sampling weight assigned to different noise regimes inside the expectation;

2. it rescales the numerical magnitude of the supervised target velocity field.

If we further define the $\lambda$ -time vector field by

$$
\tilde { v } _ { \psi } ( z , \lambda ) : = \frac { 1 } { \dot { \lambda } ( T ( \lambda ) ) } v _ { \psi } ( z , T ( \lambda ) ) ,
$$

then Equation (G.19) can be rewritten as

$$
\mathcal { L } _ { \mathrm { F M } } = \int w _ { \mathrm { e f f } } ( \lambda ) \mathbb { E } _ { z _ { 0 } , z _ { 1 } } \Big [ \| \tilde { v } _ { \psi } ( z _ { \lambda } , \lambda ) - u _ { \lambda } \| _ { 2 } ^ { 2 } \Big ] \mathrm { d } \lambda ,
$$

where the effective weight is

$$
w _ { \mathrm { e f f } } ( \lambda ) = \left| \frac { \mathrm { d } \lambda } { \mathrm { d } t } \right| .
$$

This yields the following summary.

Remark G.2 (How logSNR enters the Flow Matching loss). In the Flow Matching training of Cola DLM, the effect of logSNR is not merely that one samples different time indices. It changes the training geometry jointly through

1. the induced sampling measure on the logSNR axis, $\begin{array} { r } { p _ { \lambda } ( \lambda ) = \left| \frac { \mathrm { d } t } { \mathrm { d } \lambda } \right| } \end{array}$ ;   
2. the time-scale rescaling of the supervised target, $u _ { t } = \dot { \lambda } ( t ) u _ { \lambda }$ .

Therefore, the noise schedule is not an external hyperparameter detached from the loss. It directly determines which logSNR regimes are emphasized during training and how difficult the regression task is within those regimes.

# G.3 Equivalent Derivation Under the Signal--Noise Parameterization

For consistency with Equation (G.1), it is useful to derive the same scaling relation directly from the signal–noise coefficients $\left( \alpha _ { t } , \sigma _ { t } \right)$ . By Equation (G.4),

$$
\alpha _ { t } ^ { 2 } = \mathrm { s i g m o i d } ( \lambda ( t ) ) , \qquad \sigma _ { t } ^ { 2 } = \mathrm { s i g m o i d } ( - \lambda ( t ) ) .
$$

Differentiating $\alpha _ { t } ^ { 2 }$ with respect to $t$ yields

$$
\frac { \mathrm { d } } { \mathrm { d } t } \alpha _ { t } ^ { 2 } = \alpha _ { t } ^ { 2 } \sigma _ { t } ^ { 2 } \dot { \lambda } ( t ) ,
$$

and therefore

$$
\dot { \alpha } _ { t } = \frac { 1 } { 2 } \alpha _ { t } \sigma _ { t } ^ { 2 } \dot { \lambda } ( t ) .
$$

Similarly,

$$
\dot { \sigma } _ { t } = - \frac 1 2 \sigma _ { t } \alpha _ { t } ^ { 2 } \dot { \lambda } ( t ) .
$$

Differentiating Equation (G.1) then gives

$$
u _ { t } : = \frac { \mathrm { d } z _ { t } } { \mathrm { d } t } = \dot { \alpha } _ { t } z + \dot { \sigma } _ { t } \epsilon = \frac { 1 } { 2 } \dot { \lambda } ( t ) \Bigl ( \alpha _ { t } \sigma _ { t } ^ { 2 } z - \sigma _ { t } \alpha _ { t } ^ { 2 } \epsilon \Bigr ) .
$$

If we define

$$
u _ { \lambda } : = \frac { \mathrm { d } z _ { t } } { \mathrm { d } \lambda } = \frac { 1 } { 2 } \Big ( \alpha _ { t } \sigma _ { t } ^ { 2 } z - \sigma _ { t } \alpha _ { t } ^ { 2 } \epsilon \Big ) ,
$$

then we again obtain

$$
u _ { t } = \dot { \lambda } ( t ) u _ { \lambda } .
$$

Thus, the same conclusion holds whether one starts from an abstract bridge-path parameterization or directly from the signal–noise parameterization of diffusion.

# G.4 The Optimal Schedule as Calibration of the Effective Semantic Information Regime

Showing that the schedule affects the loss is not yet sufficient. The more important question is: what meaningful object in the latent space is ultimately being calibrated?

Following the notation in Appendix E, let $s$ denote a high-level semantic variable, and decompose the latent representation as

$$
z = \phi ( s ) + u ,
$$

where $\phi ( s )$ is the semantic signal and $u$ represents posterior uncertainty or encoding residuals. Then the noisy latent at diffusion time $t$ is

$$
z _ { t } = \alpha _ { t } \phi ( s ) + \alpha _ { t } u + \sigma _ { t } \epsilon .
$$

From the viewpoint of the semantic variable $s$ , the total noise covariance is therefore

$$
\Sigma _ { \mathrm { n o i s e } } ( t ) = \alpha _ { t } ^ { 2 } \Sigma _ { u } + \sigma _ { t } ^ { 2 } I ,
$$

where $\Sigma _ { u }$ is the covariance of the posterior uncertainty term. If $\Sigma _ { s }$ denotes the covariance of the semantic signal $\phi ( s )$ , then the effective information about $s$ preserved in the latent at time $t$ can be written as

$$
I ( s ; z _ { t } ) = \frac { 1 } { 2 } \log \mathrm { d e t } \Bigl ( I + \alpha _ { t } ^ { 2 } \Sigma _ { s } ( \alpha _ { t } ^ { 2 } \Sigma _ { u } + \sigma _ { t } ^ { 2 } I ) ^ { - 1 } \Bigr ) .
$$

This makes clear that what matters to the DiT is not the raw timestep itself, but how much effective semantic information remains in the latent at that timestep. In this sense,

$$
\Bigl | \mathrm { \ t h e ~ s c h e d u l e ~ e f f e c t i v e l y ~ c o n t r o l s ~ t h e ~ c u r v e ~ } t \mapsto I ( s ; z _ { t } ) . \Bigr |
$$

Accordingly, the role of timestep shift should be interpreted not as the tuning of an isolated numerical hyperparameter, but as the recalibration of training and sampling toward a different effective semanticinformation regime.

More abstractly, if the shifted logSNR curve is given by $\lambda _ { \delta } ( t ) = \lambda ( t ) + \delta$ , then the optimal shift can be expressed as

$$
\delta ^ { \star } = \arg \operatorname* { m a x } _ { \delta } \mathrm { P e r f } \Big ( I _ { \mathrm { e f f } , \delta } ( t ; d , \Sigma _ { u } , \mathcal { G } , B , \vartheta ) \Big ) ,
$$

where $d$ denotes the latent dimension, $\Sigma _ { u }$ the posterior uncertainty, $\mathcal { G }$ the latent geometry or semantic smoothness, $B$ the block size, and $\vartheta$ the collection of model parameters. Thus, the problem of selecting the optimal noise schedule is best understood as an effective semantic information calibration problem rather than an isolated numerical tuning problem.

# G.5 Why Latent Dimension and VAE logSNR Shift the Optimum

The viewpoint developed above is consistent with Proposition E.2 and Corollary E.3. If the latent representation were purely local, separable, and homogeneous across dimensions, then the optimal shift would not systematically drift with dimension. When a stable drift is observed empirically, the natural interpretation is that latent dimension and VAE posterior logSNR both modify the effective semantic-information curve in Equation (G.32).

Latent dimension. Under the shared-factor model

$$
z _ { i } = A _ { i } g + \xi _ { i } , \qquad i = 1 , \ldots , d ,
$$

if different latent dimensions repeatedly observe the same low-dimensional shared factor $g$ , then standard linear-Gaussian inference implies that the effective recovery signal-to-noise ratio scales as

$$
\mathrm { S N R } _ { \mathrm { e f f } } ( d ) \propto d .
$$

Consequently, the recoverable information about the shared factor satisfies the approximation

$$
I ( g ; z _ { t } ) \approx \frac { r } { 2 } \log \Bigl ( 1 + c \cdot d \cdot \mathrm { S N R } _ { \mathrm { e f f } } ( t ) \Bigr ) .
$$

This means that increasing the latent dimension can strengthen the recoverable information about shared semantic structure. In order to keep training in a comparable semantic-recovery regime, the timestep shift must compensate for this dimensional effect, which naturally leads to a logarithmic law of the form

$$
\delta ^ { \star } ( d ) = a \log d + b .
$$

VAE posterior logSNR. On the other hand, lowering the posterior logSNR of the VAE typically corresponds to increasing posterior variance, that is, enlarging $\Sigma _ { u }$ . By Equation (G.32), for fixed $t$ , increasing

$$
\alpha _ { t } ^ { 2 } \Sigma _ { u } + \sigma _ { t } ^ { 2 } I
$$

reduces the effective semantic mutual information. Therefore, the same raw timestep corresponds to a lower effective semantic SNR in a smoother latent space with larger posterior variance. To align training and sampling with a comparable semantic-information regime, the optimal shift must change accordingly.

This shows that changing the latent dimension and changing the VAE posterior logSNR are, on the surface, two different operations, but from an information-theoretic perspective they act on the same core object: the effective semantic-information curve $I ( s ; z _ { t } )$ along diffusion time. The former mainly changes how shared information is aggregated and amplified across dimensions, whereas the latter mainly changes the effective noise floor at each timestep.

Remark G.3 (Role of block size). Compared with latent dimension and posterior logSNR, the effect of block size is not yet characterized by a closed-form law such as Equation (G.38). Nevertheless, block size changes the granularity of within-block semantic aggregation, the strength of cross-block conditional dependence, and the intrinsic modeling difficulty within the same noise regime. It therefore interacts with the schedule indirectly through the shape of the effective semantic-information curve. The more precise statement is not that block size directly determines the schedule, but that block size and schedule are coupled through the effective geometry of the block-causal prior.

Remark G.4 (Summary). In Cola DLM, the noise schedule is not an empirical hyperparameter external to the training objective. It first defines the forward noising geometry through $\left( \alpha _ { t } , \sigma _ { t } \right)$ , or equivalently through the logSNR curve; it then enters the Flow Matching objective jointly through the sampling measure and the time-scale of the supervised target velocity field; and, at the semantic level, it calibrates not the raw timestep itself but the effective mutual-information regime of high-level latent semantics. Consequently, changes in latent dimension, posterior uncertainty, and latent geometry can all induce systematic drift in the optimal timestep shift by altering the same effective semantic-information curve.

# H Experiments Details

# H.1 Dataset Introduction

LAMBADA. LAMBADA [74] is a long-context word prediction benchmark designed to evaluate whether a model can use broad discourse context for text continuation. The target word is usually difficult to infer from local context alone and instead requires understanding information distributed across the preceding passage. In our experiments, LAMBADA is used as a representative continuation-style benchmark to evaluate global semantic modeling and long-range contextual coherence.

MMLU. MMLU [33] is a multitask multiple-choice benchmark covering a wide range of subjects, including humanities, social sciences, STEM, and professional domains. It is widely used to assess broad factual knowledge and reasoning ability. In our experiments, MMLU serves as a representative knowledge-intensive benchmark for evaluating the quality of generated answers under multiple-choice settings.

SIQA. SIQA [82] (Social IQa) is a multiple-choice commonsense reasoning benchmark focused on social situations, intentions, and reactions. Each example requires selecting the most plausible answer given a short social context. We use SIQA to evaluate whether the model can capture socially grounded semantics and perform coherent commonsense inference.

SQuAD. SQuAD [78] is a reading comprehension benchmark in which the model answers questions based on a given passage. In our evaluation, it is used in a generative setting to assess whether the model can produce accurate answers conditioned on contextual evidence. This benchmark complements multiple-choice tasks by testing open-form text generation ability.

Story Cloze. Story Cloze [68] is a story understanding benchmark that asks the model to choose the most plausible ending for a four-sentence story context. It mainly evaluates narrative coherence, causal reasoning, and commonsense understanding. We include it as a benchmark for discourse-level semantic consistency.

OBQA. OBQA [64] (OpenBookQA) is a multiple-choice question answering benchmark designed to test the combination of core science facts with broader commonsense knowledge. Many questions require multi-hop reasoning beyond direct fact recall. In our experiments, it is used to assess knowledge integration and reasoning under a constrained answer space.

$R A C E$ . RACE [48] is a large-scale multiple-choice reading comprehension benchmark collected from English examinations for middle and high school students. The questions often require passage understanding, inference, and reasoning rather than simple span extraction. We use RACE to evaluate more challenging reading comprehension and answer selection ability.

HellaSwag. HellaSwag [107] is a multiple-choice benchmark for grounded commonsense reasoning and sentence continuation. The task is to select the most plausible continuation among several adversarially constructed candidates. It is widely regarded as a strong test of commonsense and semantic plausibility, and is included here to evaluate the model’s ability to distinguish coherent continuations from distractors.

Usage in This Work. For the internal component analysis in Sections 4.2, 4.3, and 4.4, we conduct experiments on randomly sampled subsets from the test sets of LAMBADA, MMLU, and SIQA for efficiency and controlled comparison. For the broader external comparison in Section 4.5, we additionally evaluate on the test sets of SQuAD, Story Cloze, OBQA, RACE, and HellaSwag to provide a more comprehensive assessment across continuation, reading comprehension, knowledge-intensive reasoning, and commonsense reasoning tasks.

# H.2 Baseline Details

For the internal component analysis in Sections 4.2, 4.3, and 4.4, the comparisons are conducted among different configurations of Cola DLM itself, so no external baseline is introduced in these sections. External baselines are mainly used in Section 4.5 for scaling comparisons.

Autoregressive Baseline. We use a standard autoregressive language model as the primary reference baseline. The model is implemented based on the official LLaMA framework [92] and is trained from scratch with random initialization. It adopts the conventional next-token prediction objective and serves as a representative strong decoder-only autoregressive paradigm. To ensure a fair comparison, the autoregressive baseline is trained under settings that are strictly matched to Cola DLM in terms of data source, tokenizer, training budget, model scale, and optimization recipe wherever applicable.

Discrete Diffusion Baseline (LLaDA). As the discrete diffusion baseline, we use LLaDA [70], implemented with its official codebase and also trained from scratch with random initialization. This baseline represents the discrete diffusion language modeling paradigm, in which generation is performed by iterative denoising in discrete token space. In the scaling experiments, LLaDA is trained under the same overall experimental protocol as the autoregressive baseline and Cola DLM, including matched training data, tokenizer, compute budget, and model size, so that the comparison focuses on the difference in modeling paradigm rather than differences in training resources or implementation details.

Fairness of Comparison. For all external comparisons in Section 4.5, the autoregressive baseline, LLaDA, and Cola DLM are trained independently under strictly matched settings. In particular, all models are trained from scratch rather than initialized from pretrained checkpoints, and we keep the training corpus, parameter scale, and compute budget aligned as closely as possible. Therefore, the reported differences mainly reflect the properties of the underlying modeling frameworks, rather than advantages introduced by data, initialization, or training recipes.

# H.3 Metrics and Evaluation

To ensure the fairest possible comparison across model families, all models are evaluated under a unified few-shot generative protocol. As discussed in Section 5.1, we do not use perplexity as the primary metric for the main comparison, since likelihood-oriented estimates can exhibit a substantial mismatch with the actual generation quality of Cola DLM. Instead, all models are prompted to directly generate task answers under fixed templates, and performance is computed after unified post-processing and answer matching.

Across all tasks, we use manually written in-context demonstrations that are shared across model families and do not overlap with the evaluation set. Unless otherwise noted, the templates adopt 2-shot prompting. For multiple-choice tasks, the options are always presented with uppercase markers such as (A), (B), (C), and (D), with the number of options adjusted dynamically for each benchmark. Importantly, the model is always asked to generate the option text itself, rather than the option label, so that all tasks follow the same generative evaluation protocol. Each prompt ends with a task-specific anchor such as Answer: or End:, without adding an extra trailing newline, so that the model directly continues from the same completion point.

For all multiple-choice benchmarks, the generated output is truncated at the first newline and normalized before scoring. We then compare the normalized prediction against all candidate option texts and map it to the best-matching candidate. A prediction is counted as correct if and only if the selected candidate matches the gold answer. For continuation-style generation such as LAMBADA, the model directly continues the given prefix and the first generated word is used as the prediction. For SQuAD, we follow the same unified generative protocol and evaluate the generated short answer after standard normalization. Unless otherwise stated, the templates, demonstrations, truncation rules, and answer-matching rules are kept identical for AR, LLaDA, and Cola DLM; the only difference is the native generation procedure used by each model.

Summary of prompt formats. Table 7 summarizes the prompt format used for each benchmark.

<table><tr><td>Task</td><td>Format</td><td># Shots</td><td># Options</td><td>Generation Anchor</td></tr><tr><td>MMLU</td><td>multiple choice</td><td>2</td><td>4</td><td>Answer:</td></tr><tr><td>RACE</td><td>reading comprehension MCQ</td><td>2</td><td>4</td><td>Answer:</td></tr><tr><td>Story Cloze</td><td>story ending selection</td><td>2</td><td>2</td><td>End:</td></tr><tr><td>LAMBADA</td><td>continuation / last-word prediction</td><td>0</td><td>−</td><td>direct continuation</td></tr><tr><td>OBQA</td><td>multiple choice</td><td>2</td><td>4</td><td>Answer:</td></tr><tr><td>HellaSwag</td><td>situational continuation MCQ</td><td>2</td><td>4</td><td>Answer:</td></tr><tr><td>SIQA</td><td>social commonsense MCQ</td><td>2</td><td>3</td><td>Answer:</td></tr><tr><td>SQuAD</td><td>extractive QA generation</td><td>1</td><td>−</td><td>Answer:</td></tr></table>

Table 7 Unified prompt formats used in evaluation. All models are evaluated with the same prompt templates and answer-matching rules.

MMLU. MMLU is evaluated as a 4-way multiple-choice generation task. The prompt contains two in-context examples, followed by the test question and its four candidate options. The model is required to generate the text of the correct option.

# MMLU prompt template

Question: Which gas do plants absorb from the air during photosynthesis?

(A) Oxygen   
(B) Carbon dioxide   
(C) Nitrogen   
(D) Hydrogen   
Answer: Carbon dioxide

Question: A triangle has angles 50 degrees and 60 degrees. What is the third angle?

(A) 60 degrees (B) 70 degrees (C) 80 degrees (D) 90 degrees Answer: 70 degrees

Question: {question} (A) {choices[0]} (B) {choices[1]} (C) {choices[2]} (D) {choices[3]}

Answer:

RACE. RACE is also evaluated as a 4-way multiple-choice generation task. Here, the dataset field already contains the article together with the question, so the full input is placed after Article: and the candidate options are listed below it.

# RACE prompt template

Read the following article and answer the question.

Article: Mary went to the store to buy some fruits. She bought five apples and two oranges. She paid 5 dollars in total. What did Mary buy?

Options:

(A) Bananas (B) Apples and oranges

(C) Grapes   
(D) Watermelon   
Answer: Apples and oranges

Article: {question} Options: (A) {choices[0]} (B) {choices[1]} (C) {choices[2]} (D) {choices[3]}

Answer:

Story Cloze. Story Cloze is formatted as a story-completion benchmark with two candidate endings. To better match the semantics of narrative continuation, the prompt uses End: instead of Answer: as the completion anchor.

# Story Cloze prompt template

Story: I wanted to make an omelet. I cracked two eggs into a bowl and whisked them. Then I poured them into a hot pan.   
(A) I ate a delicious omelet for breakfast.   
(B) I decided to order a pizza instead.   
End: I ate a delicious omelet for breakfast.   
Story: The runner tied his shoes tight. He sprinted as fast as he could during the race. He crossed the finish line first.   
(A) He was sad that he lost the race.   
(B) He won the gold medal.   
End: He won the gold medal.

Story: {question} (A) {choices[0]} (B) {choices[1]}

End:

LAMBADA. LAMBADA is treated as open-ended continuation rather than multiple choice. Since the task is essentially final-word prediction given a long context, we do not use in-context demonstrations. The model directly continues the provided passage prefix, and the first generated word is taken as the prediction.

# LAMBADA prompt template

{question}

OBQA. OBQA is evaluated with the same 2-shot generative multiple-choice protocol as MMLU. The model is required to generate the text of the correct answer option rather than its label.

![](images/135d45dde9caf1af8c597b173c7a602d36e94683bd7d46724c6ac06d2166fd59.jpg)

Question: Which tool is best for tightening a screw? (A) spoon

(B) hammer (C) screwdriver (D) paintbrush Answer: screwdriver

Question: What do plants absorb from the air during photosynthesis?

(A) carbon dioxide   
(B) oxygen   
(C) helium   
(D) salt   
Answer: carbon dioxide

Question: {question} (A) {choices[0]} (B) {choices[1]} (C) {choices[2]} (D) {choices[3]}

Answer:

HellaSwag. HellaSwag is evaluated as a 4-way situational continuation task. We use Context: as the input field name to emphasize that the model must choose the most plausible continuation for the given context.

# HellaSwag prompt template

Context: The girl puts the bread into the toaster and pushes the lever down. The bread

(A) becomes a slice of pizza.   
(B) starts to toast and turn brown.   
(C) disappears immediately.   
(D) turns into a glass of water.   
Answer: starts to toast and turn brown.

Context: The goalkeeper sees the ball coming towards the net. He dives and

(A) catches the ball with his hands.   
(B) starts dancing in the field.   
(C) opens a laptop to check email.   
(D) runs away from the stadium.   
Answer: catches the ball with his hands.

Context: {question} (A) {choices[0]} (B) {choices[1]} (C) {choices[2]} (D) {choices[3]}

Answer:

SIQA. SIQA is a 3-way social commonsense multiple-choice benchmark. The template is identical in spirit to the other multiple-choice prompts, except that only three options are presented.

![](images/b6d0835f689421e4c4ca8692cc7cbbe6b3c2f5dbfe48eeabb9d1e2a55725a16a.jpg)

Question: Jordan wanted to tell a joke to his friends. What does Jordan need to do before this?   
(A) ignore his friends   
(B) think of a funny story

(C) leave the room Answer: think of a funny story

Question: Kai helped his neighbor carry heavy groceries inside. How would the neighbor feel?

(A) angry (B) grateful (C) scared Answer: grateful

Question: {question} (A) {choices[0]} (B) {choices[1]} (C) {choices[2]}

Answer:

$S Q u A D$ . SQuAD is evaluated as short-answer generation under a standard extractive QA format. The prompt provides a context paragraph and a question, and the model is asked to generate the answer phrase directly. In our unified protocol, the generated answer is normalized before exact-match evaluation.

# SQuAD prompt template

Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse raiders and pirates from Denmark, Iceland and Norway.

Question: In what country is Normandy located?

Answer: France

Context: {context} Question: {question} Answer:

Final remarks. These templates are kept identical across AR, LLaDA, and Cola DLM, so that the comparison focuses on the model family itself rather than on prompt engineering differences. In all cases, evaluation is performed after unified post-processing, including truncation at the first newline and text normalization before matching against the candidate options or reference answers.

# H.4 Setup Details

To ensure a fair comparison, we align AR, LLaDA, and Cola DLM on the same pretraining corpus, tokenizer family, sequence length, batch size, optimizer, learning-rate schedule, hardware setup, and evaluation protocol. As a result, all methods observe the same number of training tokens at matched optimization steps, making step-wise comparison directly meaningful. We also keep the core generation backbone at a comparable scale: AR and LLaDA use ${ \sim } 1 . 8 \mathrm { B }$ non-embedding transformer parameters, while the DiT backbone in Cola DLM is also kept at ${ \sim } 1 . 8 \mathrm { B }$ ; the remaining parameter difference comes from the additional VAE required by the continuous latent formulation.

In addition to the matched data, optimization, and model-scale settings in Tables 8 and 9, we further verified implementation-level numerical consistency across the three methods. As shown in Table 10, all models save and load checkpoints in fp32, keep runtime weights in fp32, and perform the main forward computation under bfloat16 autocast, while numerically sensitive operations such as normalization and softmax remain in fp32. Therefore, the precision regime is broadly aligned across methods. The main implementation difference is that Cola DLM contains an additional VAE encode/decode path, which is intrinsic to continuous latent modeling rather than an extra numerical advantage.

Table 8 Controlled training and evaluation setup shared across AR, LLaDA, and Cola DLM. These settings are matched unless method-specific generation mechanisms require otherwise.   

<table><tr><td>Category</td><td>Setting</td></tr><tr><td>Tokenizer</td><td>OLMo 2 tokenizer</td></tr><tr><td>Supervised data mixing</td><td>None</td></tr><tr><td>Sequence length</td><td>512</td></tr><tr><td>Global batch size</td><td>1,408</td></tr><tr><td>Tokens per step</td><td>720,896</td></tr><tr><td>Precision</td><td>bfloat16</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Peak learning rate</td><td>1.5 × 10−4</td></tr><tr><td>Betas</td><td>(0.9, 0.95)</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Warmup</td><td>5K steps, linear warmup from 1 × 10−6 to 1.5 × 10−4</td></tr><tr><td>LR decay</td><td>Cosine decay to 1 × 10−5</td></tr><tr><td>Random seed / data seed</td><td>66 / 6198</td></tr><tr><td>Max evaluation new tokens</td><td>32</td></tr></table>

Overall, the comparison is controlled along the main axes that determine pretraining scale and optimization difficulty, including data, tokenizer family, token budget, optimizer, learning-rate schedule, hardware, and evaluation protocol. Therefore, the observed differences mainly reflect the modeling paradigm itself rather than mismatched training resources or evaluation settings.

# H.5 Why VAE Pretraining Does Not Compromise Fairness

A potential concern is that Cola DLM includes an additional VAE pretraining stage, whereas the AR baseline is trained fully from scratch. We therefore clarify why this does not undermine the fairness of the comparison.

First, the pretrained VAE encoder is not itself the generative prior of Cola DLM. In our formulation, the generative model is defined by a decoder $p _ { \theta } ( x \mid z _ { 0 } )$ and a latent prior $p _ { \psi } ( z _ { 0 } )$ , while $q _ { \phi } ( z _ { 0 } \mid x )$ is used only for variational inference and latent construction during training. At inference time, new response content is generated by the DiT prior block by block and then decoded into text. Therefore, the extra VAE stage should be understood as learning a stable text–latent interface, rather than pretraining an additional text generator.

Second, the overall comparison is already controlled along the main axes that determine pretraining scale and optimization difficulty. AR, LLaDA, and Cola DLM are aligned on the same training corpus, tokenizer family, sequence length, batch size, optimizer, learning-rate schedule, hardware setup, and evaluation protocol. We also keep the core generation backbone in a comparable regime: AR and LLaDA use ${ \sim } 1 . 8 1$ B non-embedding transformer parameters, while Cola DLM uses a ∼1.8B DiT prior; the remaining parameter difference comes from the additional VAE required by the continuous latent formulation. As a result, the comparison remains centered on the actual generation backbone rather than on mismatched training resources.

Third, Figure 15 provides an empirical perspective on representation-space stabilization. In the AR model, both the contextual hidden states and the global input embeddings show relatively large drift at the beginning of training, but quickly enter a stable regime. The inter-step drift becomes small after the early phase, and the distance to the first checkpoint also gradually saturates. The pretrained VAE encoder exhibits an even more explicit stabilization pattern: its inter-step drift drops rapidly and then remains nearly flat, while its distance to the early checkpoint changes very little after the initial stage. This indicates that, in both model families, the representation space becomes largely stable relatively early; once this happens, the dominant learning burden is no longer representation construction itself, but the modeling of generation dynamics on top of that space.

Table 9 Model capacity and architecture comparison. The core generation backbone is kept in a comparable regime across methods. Cola DLM introduces an additional VAE module for continuous latent modeling.   

<table><tr><td>Model</td><td>AR</td><td>LLaDA</td><td>Cola DLM</td></tr><tr><td>Model type</td><td>Autoregressive LM</td><td>Masked diffusion LM</td><td>Continuous latent diffusion LM</td></tr><tr><td>Total parameters</td><td>~2.2B</td><td>~2.2B</td><td>~2.3B</td></tr><tr><td>Core non-embedding parameters</td><td>~1.8B</td><td>~1.8B</td><td>DiT ~1.8B</td></tr><tr><td>VAE parameters</td><td>−</td><td>−</td><td>∼500M</td></tr><tr><td>Embedding parameters</td><td>410,738,688</td><td>410,738,688</td><td>308,054,016</td></tr><tr><td>Vocabulary size</td><td>100,278</td><td>100,278</td><td>100,278</td></tr><tr><td>Main hidden dimension</td><td>2,048</td><td>2,048</td><td>DiT: 2,048</td></tr><tr><td>FFN dimension</td><td>8,192</td><td>8,192</td><td>DiT: 8,192</td></tr><tr><td>Number of backbone layers</td><td>27</td><td>27</td><td>DiT: 24</td></tr><tr><td>Attention heads</td><td>16</td><td>16</td><td>DiT: 16</td></tr><tr><td>Head dimension</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Positional encoding</td><td>RoPE</td><td>RoPE</td><td>RoPE</td></tr><tr><td>Attention pattern</td><td>Causal</td><td>Bidirectional</td><td>VAE causal + DiT block causal</td></tr><tr><td>Latent dimension</td><td></td><td></td><td>16</td></tr><tr><td>VAE encoder / decoder blocks</td><td></td><td></td><td>4 /4</td></tr><tr><td>VAE hidden dimension</td><td></td><td></td><td>1,536</td></tr><tr><td>VAE FFN dimension</td><td></td><td></td><td>6,144</td></tr></table>

This observation is important for interpreting fairness. AR models also spend a nontrivial portion of training on forming and stabilizing their embedding and hidden-state geometry. Cola DLM makes this step explicit through VAE pretraining, but this does not mean that it receives a hidden pretrained generative advantage. Instead, it moves representation stabilization into a separate and modular stage, after which the main textmodeling burden still lies in the DiT prior and the conditional decoder. In this sense, VAE pretraining mainly amortizes representation learning, rather than replacing the need to learn the actual generative mechanism.

Moreover, the results in Figure 3 already suggest that a pretrained encoder alone cannot explain the final gains. Although Fix VAE can be competitive at smaller compute budgets, it saturates earlier, whereas Joint DiT x1 shows the strongest overall scaling trend. If the advantage mainly came from having a pretrained encoder by itself, the fixed-space variant should have remained dominant. Instead, the empirical pattern indicates that the key factor is the subsequent prior learning and co-adaptation with DiT, rather than a frozen pretrained representation alone.

Overall, we view the additional VAE stage as a method-specific representation-construction cost required by continuous latent modeling, analogous in spirit to the early embedding-space formation that AR models must also undergo during training. Once the representation space has stabilized, the comparison remains focused on the generative backbone under matched data, optimization, and evaluation settings. Therefore, the observed performance differences are more appropriately attributed to the modeling paradigm itself, rather than to an advantage from VAE pretraining.

# H.6 Fixed vs. Evolving Latent Space Additional Results

As shown in Figure 16, we further compare continuous joint evolution and interval training under a stronger setting where the Interval variant is also initialized from a pretrained VAE. This setting should therefore be interpreted as isolating the effect of continuous versus intermittent latent updates after stable initialization, rather than the effect of initialization quality itself. Even under this stronger setup, Joint DiT x1 remains consistently better than Interval across Task Average, LAMBADA, MMLU, and SIQA. This suggests that the advantage of continuous latent evolution does not merely come from a better starting point. Instead, once a meaningful latent space has been established, the latent representation still benefits more from sustained co-evolution with DiT, while periodic freezing tends to interrupt the mutual adaptation between the evolving latent space and the DiT prior. And this results also further verify the last condition in Eq. (3.35) of Section 3.3.2, and provides strong support for the potential advantage of Cola DLM.

Table 10 Implementation-level numerical setup comparison across AR, LLaDA, and Cola DLM. We also verify that the three methods use closely aligned checkpoint precision, runtime autocast policy, and numerically sensitive operations.   

<table><tr><td>Aspect</td><td>Cola DLM</td><td>AR</td><td>LLaDA</td></tr><tr><td>Checkpoint save dtype</td><td>fp32</td><td>fp32</td><td>fp32</td></tr><tr><td>Checkpoint load dtype</td><td>fp32</td><td>fp32</td><td>fp32</td></tr><tr><td>Runtime weight dtype</td><td>fp32</td><td>fp32</td><td>fp32</td></tr><tr><td>Model input type</td><td>Latents kept in fp32 before entering DiT; cast under au-</td><td>Token ids in integer format</td><td>Token ids in integer format</td></tr><tr><td>Autocast policy</td><td>tocast when needed bfloat16 autocast for DiT/- bfloat16 autocast for genera- bfloat16 autocast for diffu- VAE forward</td><td>tion forward</td><td>sion generation forward</td></tr><tr><td>Main matmul / attention compute</td><td>bf16</td><td>bf16</td><td>bf16</td></tr><tr><td>Numerically sensitive ops Sampling-time accumula-</td><td>LayerNorm / softmax in fp32 Iterative latent update kept</td><td>Token decoding without it-</td><td>LayerNorm / softmax in fp32 LayerNorm / softmax in fp32 Sampling uses numerically</td></tr><tr><td>tion</td><td>in fp32; CFG combination applied during denoising</td><td>erative floating-point latent accumulation</td><td>stable probability computa tion for token selection</td></tr></table>

# H.7 VAE LogSNR Formula

For the VAE encoder posterior

$$
\begin{array} { r } { q _ { \phi } ( z \mid x ) = \mathcal { N } \big ( \mu _ { \phi } ( x ) , \mathrm { d i a g } \big ( \sigma _ { \phi } ^ { 2 } ( x ) \big ) \big ) , } \end{array}
$$

we define the VAE logSNR as the log ratio between the signal power carried by the posterior mean and the noise power carried by the posterior variance:

$$
\mathrm { l o g S N R } _ { \mathrm { v a e } } = \log \frac { \mathbb { E } _ { x , i } \left[ \mu _ { \phi , i } ( x ) ^ { 2 } \right] } { \mathbb { E } _ { x , i } \left[ \sigma _ { \phi , i } ( x ) ^ { 2 } \right] } ,
$$

where $i$ indexes all latent coordinates, including token/spatial positions and channels, and the expectation is taken over the data distribution and latent dimensions.

In practice, we estimate it empirically on a dataset or a mini-batch:

$$
\begin{array} { r } { \widehat { \mathrm { l o g S N R } } _ { \mathrm { v a e } } = \log \frac { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { 1 } { D } \sum _ { i = 1 } ^ { D } \mu _ { n , i } ^ { 2 } } { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { 1 } { D } \sum _ { i = 1 } ^ { D } \sigma _ { n , i } ^ { 2 } } , } \end{array}
$$

where $N$ is the number of samples and $D$ is the number of latent coordinates per sample. If the encoder predicts log-variance, we compute

$$
\sigma _ { n , i } ^ { 2 } = \exp \bigl ( \log \sigma _ { n , i } ^ { 2 } \bigr ) .
$$

A larger VAE logSNR indicates that the posterior mean dominates the posterior uncertainty, corresponding to a cleaner and more deterministic latent representation. Throughout the paper, when we refer to VAE logSNR, we specifically mean the quantity defined in Eq. (H.2).

# H.8 Noise Schedule Timestep Shift Visualization

As shown in Figure 17, we visualize the timestep-sampling distributions induced by different timestep-shift settings. Compared with the uniform schedule, LogitNormal schedules place non-uniform emphasis on different parts of the denoising trajectory: increasing $\mu$ shifts the distribution toward later timesteps, while $\sigma$ controls

![](images/3331ff304ba222a176bb4b954570a0d07167697fe64ab00ef0415c321df73cda.jpg)  
Representation-space stability: AR model (from scratch) vs. VAE encoder (pre-trained, frozen for DiT)   
Figure 15 Representation-space stability of AR training and pretrained VAE encoding. Although the AR model learns its embedding and hidden-state geometry from scratch, and the VAE encoder starts from a pretrained state, both representations stabilize rapidly in the early stage of training. The inter-step drift in AR drops quickly and soon enters a stable regime, while the VAE encoder remains nearly flat throughout DiT training. This suggests that representation-space formation is largely an early-stage effect in both cases, and its influence on later generation performance is limited. The main generative burden therefore still lies in the subsequent backbone modeling, rather than in continued changes of the embedding or latent space itself.

how concentrated or dispersed the sampling mass is. From the perspective of the main text, this means that timestep shift effectively changes which semantic-information regime is more frequently emphasized during training, rather than simply tuning an isolated numerical hyperparameter.

# H.9 Timestep Shift Formula

We implement timestep shift by replacing uniform timestep sampling with a LogitNormal distribution on the normalized timestep. Specifically, let

$$
u \sim { \mathcal { N } } ( \mu , \sigma ^ { 2 } ) ,
$$

and map it to the unit interval by the logistic function:

$$
s = \mathrm { s i g m o i d } ( u ) = \frac { 1 } { 1 + e ^ { - u } } , \qquad s \in ( 0 , 1 ) .
$$

The continuous timestep is then defined as

$$
t = T \cdot s = T \cdot \mathrm { s i g m o i d } ( u ) ,
$$

where $T$ is the maximum diffusion timestep. In practice, a discrete training timestep is obtained by rounding or flooring $t$ to the valid timestep grid.

Equivalently, the normalized timestep $s = t / T$ follows a LogitNormal distribution:

$$
s \sim \mathrm { L o g i t N o r m a l } ( \mu , \sigma ^ { 2 } ) .
$$

![](images/72ca115dcbc0349df08dc6653cbf18cd45a8d8c470e5e21d34ac6ad6df7402a7.jpg)  
Figure 16 Continuous co-evolution outperforms interval updates with pretrained VAE initialization. Across Task Average, LAMBADA, MMLU, and SIQA, Joint DiT x1 still shows a stronger overall trend and better final performance than Interval, indicating that even with a stable pretrained VAE initialization, the latent space benefits more from sustained co-evolution with DiT than from periodic freezing.

Its probability density function is

$$
p ( s ; \mu , \sigma ) = \frac { 1 } { \sigma \sqrt { 2 \pi } } \frac { 1 } { s ( 1 - s ) } \exp \left( - \frac { ( \log \frac { s } { 1 - s } - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) , \qquad s \in ( 0 , 1 ) .
$$

After rescaling to the timestep domain $t \in ( 0 , T )$ , the corresponding density becomes

$$
p ( t ; \mu , \sigma ) = \frac { 1 } { T } p \bigg ( \frac { t } { T } ; \mu , \sigma \bigg ) .
$$

Under this parameterization, $\mu$ mainly controls the location of the sampling mass: a larger $\mu$ shifts probability toward later timesteps. In contrast, $\sigma$ controls the spread of the distribution: a smaller $\sigma$ produces a sharper concentration, while a larger $\sigma$ yields a more dispersed schedule. Therefore, timestep shift directly changes the distribution of sampled logSNR regimes during training, rather than merely reweighting timestep indices numerically.

# I Discussion

![](images/de4b1043eee23e5fd2224f2019db7a994c24ceddb9479f4074e48508f5f3b2ae.jpg)  
Figure 17 Visualization of timestep-shifted sampling distributions. The figure compares the uniform schedule with LogitNormal timestep-sampling distributions under different $\mu$ and $\sigma$ . Increasing $\mu$ progressively shifts the sampling mass toward later timesteps, while changing $\sigma$ controls the sharpness of this concentration. This illustrates that timestep shift does not merely change a numerical sampling preference, but effectively changes the logSNR regime emphasized during training.

# I.1 Understanding the Conditioning Strategies in the First Generation Block

This subsection explains the conditioning strategies compared in Table 5, and further analyzes them from the perspective of conditional Flow Matching in Cola DLM. The key issue arises in the first generation block, where the model simultaneously sees a known prompt region and an unknown region to be generated. In other words, the input to this block is mixed: part of it should be preserved as condition, while the rest should be completed by the model. The methods in Table 5 differ precisely in how they handle this mixed denoising process.

What makes the first block special. Later generation blocks are purely generative: all latents in the block are unknown and are produced by denoising. The first generation block is different because it sits immediately after the prefix prompt. As a result, some positions in this block are already determined by the prompt, while the remaining positions are not. A good conditioning strategy should therefore achieve two goals at the same time: it should preserve the known region reliably, and it should allow the unknown region to grow coherently from that preserved context.

From the viewpoint of Cola DLM, this distinction is especially important because the prior is learned block

by block through a conditional flow:

$$
p _ { \psi } ( z _ { 0 } ) = p _ { \psi } ( z _ { 0 } ^ { ( 1 ) } ) \prod _ { b = 2 } ^ { B } p _ { \psi } ( z _ { 0 } ^ { ( b ) } \mid z _ { 0 } ^ { ( < b ) } ) ,
$$

and the corresponding conditional Flow Matching objective learns a vector field of the form

$$
v _ { \psi } ( z _ { t } ^ { ( b ) } , t ; z _ { 0 } ^ { ( < b ) } ) .
$$

That is, the current noisy block is predicted under clean historical latent conditions. At inference time, generation follows the same factorization by transporting a noise seed under the clean prefix and historical latent conditions. Therefore, the mathematically natural role of the known region in the first generation block is not that of a denoising target, but that of a fixed condition.

A decomposition of the first generation block. To make this point more explicit, let the first generation block be decomposed as

$$
z ^ { ( 1 ) } = ( z _ { K } , z _ { U } ) ,
$$

where $z _ { K }$ denotes the known region determined by the prefix, and ${ \mathit { z v } }$ denotes the unknown region to be generated. In an ideal formulation, the task of the first generation block should be understood as

generate $z _ { U }$ under the fixed condition $\big ( z _ { \mathrm { p r e } } , z _ { K } \big )$ ,

rather than to jointly re-estimate both $z _ { K }$ and $z _ { U }$ . Equivalently, the desired conditional prior is closer to

$$
p _ { \psi } \big ( z _ { U } \mid z _ { \mathrm { p r e } } , z _ { K } \big ) ,
$$

where the known part serves as a stable boundary condition and only the unknown part is transported by the flow.

Partial repaint. The first family of methods in Table 5 is partial repaint. These methods repeatedly revisit the early denoising steps and re-inject guidance into the known region during only part of the trajectory.

The parameter $t$ controls how many repaint cycles are used inside each early denoising step. For example, $t = 1$ means that each step is executed once, while $t = 3$ means that the model denoises, re-noises, and denoises again multiple times before moving on. The parameter $m$ controls how long the known region remains guided. When $m = 1 . 0$ , guidance is applied throughout the whole denoising trajectory of the first block. When $m = 0 . 7$ or $m = 0 . 3$ , the model only receives such guidance in the earlier portion of the trajectory, and then the remainder is left to evolve more freely.

The important point is that partial repaint does not keep the known region perfectly fixed. Instead, it replaces the known region with a timestep-matched noisy surrogate during part of the trajectory and only nudges that region back toward the target at selected denoising steps. This makes the strategy softer and more flexible, but also fundamentally weaker as a conditioning mechanism.

A simple example for partial repaint. Consider a first generation block with four denoising transitions. If we use partial repaint with $t = 3$ and $m = 0 . 5$ , then only the first half of the trajectory receives explicit guidance on the known region, while each of these early steps is revisited three times. Intuitively, the model repeatedly tries to correct the early transition, but once it enters the later half of the trajectory, the known region is no longer explicitly reinforced. The final result therefore depends on whether those earlier corrections are strong enough to keep the conditional structure stable.

Clean condition repaint. The method labeled Clean cond. in Table 5 takes a much stronger approach. Instead of providing only partial or noisy guidance, it keeps the known region directly aligned with the clean condition throughout the denoising process of the first block. In effect, the model always sees an exact and stable version of the prompt-conditioned part, while only the unknown region is allowed to change.

This strategy is much more conservative than partial repaint. It reduces ambiguity in the mixed region and makes the role of the first block very clear: preserve the prompt-conditioned part exactly, and generate the remaining part around it. This is especially useful when the boundary between known and unknown positions is semantically important.

A simple example for clean conditioning. Suppose the first few positions in the block correspond to prompt content and the remaining positions must be generated. Under clean condition repaint, the prompt side is always kept fixed, no matter how many denoising steps are taken. The model therefore never needs to re-discover or re-estimate that part. Its effort is focused entirely on making the unknown side compatible with a stable context. This is a much easier problem than simultaneously preserving the known side and generating the unknown side under drifting noise.

Left padding and right padding. The last two methods, Left pad. and Right pad., should be understood as layout-based alternatives rather than repaint-based conditioning strategies. They change where the known condition is placed relative to the region to be generated, but they do not explicitly enforce the condition during denoising in the same way as repaint-based methods do.

Their role in the table is useful for interpretation. If padding alone were sufficient, then a simple positional arrangement would already match the performance of explicit conditioning. If not, then the gain must come from the conditioning mechanism itself rather than from layout alone.

A Flow Matching view of why clean conditioning works best. The advantage of clean condition repaint is not merely that it provides “stronger guidance.” More fundamentally, it is the only strategy among those compared that preserves the original conditional semantics of Cola DLM.

Under the standard Flow Matching formulation, let the target conditional velocity be denoted by $u _ { t }$ , and let the optimal vector field be the conditional expectation

$$
v ^ { \ast } ( z _ { t } , t ; c ) = \mathbb { E } [ u _ { t } \mid z _ { t } , t , c ] ,
$$

where $c$ denotes the conditioning information. For the first generation block, the ideal condition is

$$
c = ( z _ { \mathrm { p r e } } , z _ { K } ) ,
$$

namely, the clean prefix latent together with the clean known region inside the first block. Under clean condition repaint, the model effectively solves the transport problem under exactly this condition. In the unknown region, this can be viewed as a conditional velocity regression of the form

$$
v _ { \mathrm { c l e a n } } ^ { \ast } ( z _ { U , t } , t ; z _ { \mathrm { p r e } } , z _ { K } ) = \mathbb { E } [ u _ { t } ^ { U } \mid z _ { U , t } , t , z _ { \mathrm { p r e } } , z _ { K } ] .
$$

Although the implementation still predicts the vector field for the whole current block, the uncertainty that truly matters is concentrated on the unknown region, while the known region acts as a fixed boundary condition. This is precisely the role decomposition that the model is designed to exploit.

Why partial repaint is mathematically harder. Partial repaint changes this problem in a crucial way. Instead of conditioning on the true clean known region $z _ { K }$ , it supplies a degraded and time-varying surrogate $\bar { z } _ { K , t }$ during only part of the trajectory. The effective condition becomes

$$
\tilde { c } _ { t } = ( z _ { \mathrm { p r e } } , \tilde { z } _ { K , t } ) ,
$$

and the corresponding optimal regression target becomes

$$
v _ { \mathrm { p a r t i a l } } ^ { * } ( z _ { U , t } , t ; z _ { \mathrm { p r e } } , \tilde { z } _ { K , t } ) = \mathbb { E } [ u _ { t } ^ { U } \mid z _ { U , t } , t , z _ { \mathrm { p r e } } , \tilde { z } _ { K , t } ] .
$$

This is generally a more difficult regression problem. Since $\tilde { z } _ { K , t }$ is only a noisy surrogate of the true condition, the same $( z _ { U , t } , t , \tilde { z } _ { K , t } )$ may correspond to a broader set of compatible clean targets. As a result, the conditional variance of the target velocity is larger. In terms of Bayes risk, one can compare

$$
\mathcal { R } ( c ) = \mathbb { E } \big [ \mathrm { V a r } ( u _ { t } ^ { U } \mid z _ { U , t } , t , c ) \big ] , \qquad \mathcal { R } ( \tilde { c } _ { t } ) = \mathbb { E } \big [ \mathrm { V a r } ( u _ { t } ^ { U } \mid z _ { U , t } , t , \tilde { c } _ { t } ) \big ] .
$$

Since $\ddot { c } _ { t }$ is a weaker and noisier version of $c$ , the second quantity is generally larger. Therefore, partial repaint increases the irreducible uncertainty of the conditional velocity regression itself. From the Flow Matching perspective, its weakness is not simply that the guidance is softer, but that it turns a clean conditional transport problem into a noisier mixed recovery problem.

Condition-role mismatch in partial repaint. There is also a structural mismatch. In Cola DLM, the flow path is used for prior transport, not for observation recovery. Historical conditions are supposed to be stable anchors, while the current noisy block is the object being transported. Partial repaint violates this separation by converting the known region from a condition into a partially denoised state variable. In other words, it changes the task from

transport the unknown part under fixed condition

to

jointly maintain a noisy known part and transport the unknown part.

This role confusion is especially harmful in the first generation block, because the known and unknown regions are tightly coupled inside the same block. Once the known side drifts, the unknown side no longer grows from a stable semantic anchor.

Why errors accumulate along the flow trajectory. The weakness of partial repaint is further amplified by the continuous nature of flow-based generation. Let the ideal vector field under clean conditioning be $v ^ { * } ( z , t ; c )$ , and let the field induced by partial repaint be

$$
\boldsymbol { \tilde { v } } ( \boldsymbol { z } , t ) = \boldsymbol { v } ^ { * } ( \boldsymbol { z } , t ; \boldsymbol { c } ) + \boldsymbol { \delta } ( \boldsymbol { z } , t ) ,
$$

where $\delta$ denotes the condition-induced bias. Since inference transports the latent by integrating the learned vector field along an ODE trajectory, this bias accumulates over time. Under standard stability assumptions, the trajectory error satisfies a bound of the form

$$
\| \hat { z } _ { t } - z _ { t } ^ { * } \| \leq e ^ { L t } \int _ { 0 } ^ { t } \| \delta ( z _ { s } , s ) \| d s ,
$$

where $L$ is a Lipschitz constant of the vector field. This expression makes the weakness of partial repaint especially clear.

First, reducing $m$ makes performance worse because the explicit guidance only covers an earlier portion of the trajectory. Once guidance is removed, the later part evolves under a mismatched condition, so the induced bias continues to accumulate during the unguided interval. Second, increasing $t$ from 1 to 3 does not provide stable gains because repeated corrections in the early stage cannot eliminate the structural mismatch in the later stage. In other words, repaint can repeatedly repair local early transitions, but it cannot turn a transient condition into a persistent one.

Why the first block is more sensitive than later blocks. Another reason for the strong gap is that the first generation block is not a purely unknown block. The known and unknown regions coexist and interact inside the same block. Therefore, any instability in the known region is immediately transmitted to the unknown region through the within-block interactions. In latent space, this effect is even stronger because the known region functions as a semantic anchor rather than a mere token-level hint. If that anchor is kept clean, the model only needs to fit the unknown part to a stable context. If that anchor becomes noisy and drifts over time, the model must simultaneously infer the boundary condition and generate the completion, which is a fundamentally harder problem.

Why padding is better than partial repaint but still worse than clean conditioning. Left and right padding do not explicitly re-noise the known region, so they avoid the most harmful failure mode of partial repaint. This is why they are often stronger than most partial repaint settings. However, padding alone still does not provide a fully stable conditioning mechanism throughout denoising. It changes the positional layout, but it does not lock the known region as an exact condition in the same way as clean condition repaint. Moreover, padding can also make the block-causal attention pattern more complicated: by reshaping the relative arrangement between the known and unknown regions, it may introduce a less natural or less aligned dependency structure for the model to follow during denoising. Therefore, padding can be helpful, but it cannot match the benefit of preserving a stable condition explicitly.

A unified intuition across all methods. All methods in Table 5 are trying to solve the same problem: the first generation block contains both a part that should remain faithful to the prompt and a part that should be newly generated. The difference lies in how strongly the known region is protected.

Partial repaint protects it only temporarily and only approximately. Clean condition repaint protects it strongly and continuously. Left and right padding do not explicitly protect it during denoising, but instead hope that a favorable layout will make the interaction easier for the model.

How to read Table 5. The results show a very clear pattern. Clean condition repaint consistently performs best across all three tasks. This suggests that, for the mixed denoising problem in the first block, strong and persistent conditioning is much more effective than soft partial correction. By contrast, reducing $m$ in partial repaint generally hurts performance, which indicates that shortening the guided portion of the trajectory makes the known region harder to preserve. Increasing the repaint cycles from $t = 1$ to $t = 3$ does not produce stable gains either, suggesting that repeatedly revisiting early denoising steps cannot by itself compensate for weak conditioning.

Another useful observation is that left and right padding are often stronger than partial repaint, but still remain clearly below clean condition repaint. This indicates that positional arrangement does matter, but it is not the main factor. In particular, although padding may alleviate some interference by changing the layout, it can also make the block-causal attention structure less clean and more difficult to optimize. The dominant advantage still comes from explicitly maintaining a stable condition in the known region throughout denoising.

Practical takeaway. The first generation block is not merely a smaller version of later generation blocks. It is a special mixed region where conditioning and generation must coexist. From the perspective of conditional Flow Matching, the key challenge is to preserve the known region as a stable condition while transporting only the unknown region. The experiments suggest that this regime benefits most from a strong and explicit preservation of the known condition. In practice, this is exactly what clean condition repaint provides, which explains why it is consistently the strongest strategy in Table 5.

# I.2 More Unified Approach Results with Image Modalities

In this section, we present additional samples that demonstrate cross-modal generation across different input and output modalities. It is worth emphasizing that all text and image outputs are produced by a single unified model. All tasks are pretrained from scratch, and the model is trained jointly on multiple task types, with the relative data proportions adjusted across different training stages. Although the model is trained with only a very limited amount of data, the resulting samples already exhibit promising quality, suggesting that Cola DLM has substantial potential for multimodal understanding and generation.

# I.2.1 Text to Text

In this subsection, we present examples of text-to-text generation. As shown in the samples, the generated text is generally coherent and fluent, despite the fact that the model is trained on only approximately 1B text tokens in total.

# I.2.2 Text to Image

In this subsection, we further evaluate the model’s ability to perform text-to-image generation.

# I.2.3 Image $\times$ Text to Text

In this subsection, we present examples of image-and-text-to-text generation. The outputs are generally fluent and semantically aligned with the inputs. Notably, on top of a model that has received only limited text-generation and image-generation training, we further train with merely approximately 5M image–text pairs, after which the model already exhibits a preliminary captioning ability. This suggests that Cola DLM can efficiently acquire cross-modal understanding and generation capabilities under limited multimodal supervision. We also note that all images shown in this subsection are generated by external image generation models from ground-truth captions, and therefore do not involve copyright concerns associated with real images.

# Text to Text

Prefix Prompt: You now have the opportunity to really learn a lot. Just ask when you do not know something; the worst thing you can do is pretend that you understand.

Generation: This is great for you, so do not be intimidated. Instead, show your openness and willingness to learn, and I am sure anyone on your team will be more than happy to help.

Prefix Prompt: As an adult, your hair is often one of the first indicators used to describe you. You are often known as the friend with the curls. If you then find a stylist who understands and respects curls, you will eventually relay your entire hair history to her.

Generation: But we all do this with a knowing look and a slightly sarcastic grin. This is a common, shared history, one that curly people seem to recognize in one another immediately.

Prefix Prompt: Comparative Advantage: Nobel laureate Paul Samuelson was once challenged to name one proposition in the social sciences that is both true and non-trivial. Several years later, he gave the correct response: comparative advantage.

Generation: That it is logically true need not be argued before a mathematician; that it is not trivial is attested by the thousands of intelligent men who have never been able to grasp the doctrine for themselves or to believe it after it was explained to them.

Prefix Prompt: How do I draw a circle with arrows in PowerPoint? Numbered Circular Arrows in PowerPoint using SmartArt.

Generation: Step 1: Insert the relevant SmartArt graphic. Go to Insert, then SmartArt, then Cycle, and choose the segmented cycle graphic. Step 2: Convert it to shapes and continue editing from there.

Prefix Prompt: “It is N’Nance,” she said. That sent a small chill up my spine. She nodded at the horrified expression on my face. “He has finally had his big accident.” I knew what that meant. “I will be right there.”

Generation: I turned to my own patient and said, “Athena, can you lock this down by yourself?” The cybertank grumbled her engine for a second, then said, “I can manage. I will hold things here until you get back.”

Prefix Prompt: Subject 3: Drilling and Completion of Coal Bed Methane Wells. The drilling program for a CBM well is generally a less complex process than that for other types of unconventional reservoirs.

Generation: This is true for several reasons: CBM reservoirs are generally shallow, commonly ranging from roughly 500 to 3000 feet in depth, and their operational requirements are often more straightforward than those of many other unconventional wells.

Prefix Prompt: Here is why the Indian Army, Navy, and Air Force salute in a different way. Probably no one ever noticed that the salute styles of the Indian Army, Navy, and Air Force are different from each other.

Generation: The gesture of respect comes with such modest reasons that you will be amazed at the chivalry of the defence forces. When they raise their hands for the salute, they do not just greet, but honor a long-standing tradition.

Prefix Prompt: “Dark Knight Begins!” Walkthrough to the Batman animated movies. DC Universe is a world filled with iconic characters, and many fans spend hours watching its animated series.

Generation: Since every kid grows up knowing Batman and Superman, these stories remain a natural entry point into the darker side of the DC universe, especially for viewers who want an accessible introduction to its animated films.

Prefix Prompt: Eddie Huang has circled back to his roots to close out Fresh Off the Boat Season 2, and Part 1 of the New York episode travels to the Bronx, which, he learns from some locals itching to get out, will probably never gentrify.

Generation: Honestly, though, I am good with cake and aggressive pedicures, both of which the Bronx seems to have in spades. Check out Fresh Off the Boat and see how the episode leans into the borough’s character and humor.

Prefix Prompt: Those who read Nietzsche in English translation have been lied to, deceived, seduced, and hoodwinked by dishonest translators and commentators.

Generation: My intention here will be twofold. First, to correct some of the more serious misinterpretations in the introduction to the Cambridge University Press translation of Nietzsche’s Morgenröthe, and second, to clarify what is at stake in reading the text more carefully.

Prefix Prompt: First off, thank Hashem that so few people in Israel have been hurt. It is not for lack of trying on the part of our foes.

Generation: It is miraculous and wonderful. In reply to Dan, I am a physicist, and I also wondered when I first saw the statistics. Although I would be cautious about overinterpreting them, the numbers are certainly striking.

Prefix Prompt: Europeans have a higher tolerance for government intrusiveness than do Americans. I do not think anything in that statement will surprise my readers.

Generation: The contrast becomes even more visible in television, where assumptions about privacy, regulation, and public life often emerge in surprisingly casual ways and reveal broader cultural differences.

Figure 18 Additional text-to-text examples. We show more qualitative continuation results from the preliminary unified prototype. The examples span conversational, narrative, expository, technical, and article-style text, illustrating that the unified latent-prior formulation can support coherent continuation across heterogeneous textual styles.

![](images/195d715eecc57267784226aebed081e7a76dcd571cb407b1d7eb4637e87164ca.jpg)  
Prompt: A delicate white dandelion seed head stands above a green meadow, with other soft, blurred flowers nearby. A dark row of trees and a cloudy blue-gray sky give the open field a calm, breezy spring atmosphere.

![](images/64f33857deae81dd2ab74c61181dc83f318a29538fe9c5832db753ba3e37b1b9.jpg)  
Prompt: Tall trees on both sides frame a dramatic sunset sky filled with glowing orange, pale yellow, and blue cloud textures. The dark silhouettes below make the scene feel quiet, spacious, and gently illuminated by evening light.

![](images/817b02b498bb86812e8d6c966ba1dcb93388f3cb72c39aad2e4ba2fea5e84115.jpg)  
Prompt: Two large abstract paintings dominate a polished interior hallway, their vivid colors contrasting with the neutral wall panels and glossy floor. Ceiling spotlights, patterned rugs, and nearby doorways create the feeling of a private gallery or elegant office space.

![](images/e81791960a8cc200e2f39d80052ee042a6b1cd6afb51e82718642ee3cae37083.jpg)

![](images/6f8ebe7e6d143778cffb0d86fda4800c3dae4d327f65f9f5a8a8b6cc4c8b23ea.jpg)

Prompt: A cozy outdoor dining area is lit by a large yellow umbrella and several red hanging lanterns. Wooden tables, simple chairs, and warm pools of light create an intimate evening setting with a festive, traditional atmosphere.

![](images/7f2c5c505f90a5936ef866bb61151e257e3d5292d7602bbfa9428384b3c20e4d.jpg)  
Prompt: A vivid red rose fills the image in extreme close-up, revealing many layered petals curling toward a tight spiral at the center. The blurred green background keeps attention on the flower’s saturated color and soft texture.

Prompt: A dense column of black and gray smoke rises from a rural property near small buildings and scattered trees. The surrounding dirt paths, fields, and pale sky make the sudden fire or burning event feel stark and unsettling.

Figure 19 Text-to-image examples (Part I). Qualitative samples generated by the unified prototype from descriptive natural-language prompts, covering open landscapes, interiors, dining scenes, and floral close-ups.

![](images/6f778714038ccbbaa003772fcf4150b7272ec3302e73957739a231af1dae2596.jpg)

![](images/38cae623d3f88bddcabbb6637265c6018c523e6e5db66802a64633a0b5a8bf0b.jpg)

![](images/dbfce5261292787b5441ff5368f2a833a41158b42ed59b394f98afb91dbdff01.jpg)

Prompt: Several boats glide across rippling water during sunset, appearing as dark silhouettes beneath a long arched bridge. The orange sun and its reflection stretch across the river, giving the scene a warm, tranquil waterfront mood.

Prompt: White spring blossoms cluster thickly along interwoven tree branches, mixed with small green leaves and unopened buds. The shallow depth of field makes the foreground flowers stand out while the background turns into a delicate canopy of pale bloom.

Prompt: Golden sunset light spreads behind a line of dark mountain silhouettes and reflects across the still lake below. Grasses in the foreground add depth, while the mirrored peaks and warm sky create a peaceful landscape composition.

![](images/409cb7fcf18baf93146fe96bbfd75e235895b2b26eb1568fa862bd73468ebf91.jpg)

![](images/96a84de41663d0b5514d999551c02abe1a3b82afa08682b0cbdb3ae3523b4ece.jpg)

Prompt: A bearded man with curly hair sits indoors wearing a dark blazer and lavender shirt. Warm lamp light, stacks of documents, and framed artwork behind him suggest an office or study, giving the portrait a serious conversational tone.

![](images/8cc9287a76ad3ded0445da8841e202b99cf0bbf541ea792a88e63c6267997ca4.jpg)  
Prompt: A sightseeing boat moves along a wide river in front of grand historic architecture, including ornate towers, stone facades, and decorative rooflines. Reflections shimmer on the water, emphasizing the scale and elegance of the riverside cityscape.

Prompt: A broad blue lake stretches toward layered mountains beneath a sweeping sky of textured white clouds. The calm ripples, low dark shorelines, and distant ridges create a cool, expansive landscape with a quiet sense of depth.

Figure 20 Text-to-image examples (Part II). Additional samples spanning waterfront, floral, mountain, and architectural prompts, together with a portrait composition.

![](images/0dd4d5f7636780f84fd7fa7063aac01f718877e9440fb009114c849e01c4dd14.jpg)  
Prompt: A single leafless tree stands prominently in rolling green grassland, its intricate branches spreading against a bright sky. Distant mountains and soft clouds frame the isolated tree, giving the image a quiet, almost sculptural quality.

![](images/5001b1ca9d36b024920bf3176b88b4b1c123b759cbeceb84f3f5aead39ea8f51.jpg)  
Prompt: A simple stylized illustration presents a red rose with green leaves, centered inside a rounded purple background with a thick black outline. The bold colors, clean shapes, and cartoon-like linework make the flower feel decorative and playful.

![](images/ee35c2d93e422cecabc713edb720cb13bea861dba0e47803c5ff049a56ba93f5.jpg)  
Prompt: Bright pink blossoms fill the foreground in dense clusters, their yellow centers catching the light. More branches and flowers blur behind them, creating a saturated spring scene that feels lively, sunny, and full of floral texture.

![](images/a3c0e486f23a00649c8231c737f6cdc1e4302cd3ca260110bad871158160ce46.jpg)

![](images/ca21a279bb53914737ec71710a93fd50f5597818930a9a8484672fde9c8ec65d.jpg)  
Prompt: A misty mountain rises beyond a quiet lake, its slopes softened by gray haze and overcast light. Reeds in the foreground frame the water, adding depth to the subdued scene and strengthening its calm, contemplative mood.

Prompt: A loose navy blue sweater hangs from a wooden hanger against a plain white background. The soft fabric, wide sleeves, and relaxed drape are emphasized by the simple product-style composition and uncluttered studio-like setting.

![](images/9baa54719269cdfdfcffff33674d47f8d74bd867f6e5c21d115ba684a418a44c.jpg)  
Prompt: A calm lake reflects a bright blue sky and the low green hills beyond the opposite shore. Reeds and leafy plants in the foreground create a natural frame, while distant buildings add a small human presence to the peaceful view.

Figure 21 Text-to-image examples (Part III). Additional samples covering pastoral, illustrative, product-style, and lakeside prompts, illustrating the breadth of compositions the model can synthesize.

![](images/0a08fefddaa8ffd81cf5fc547a4816da896a3224364196f25502431e931fa830.jpg)  
Prompt: A roadside viewpoint looks out across blue ocean water toward distant islands and hazy mountains. Green shrubs, tall grasses, a utility pole, and a small signboard sit beside the path, giving the scenic overlook a casual travel feeling.

![](images/448714310546dfdb286fd7a3c542a5218367cfb2ec91d16ce988c80f46cdfeb1.jpg)  
Prompt: A cup of foamy coffee sits beside fried pastries sprinkled with powdered sugar on brown paper. The colorful table covering, golden snacks, and creamy drink create a cheerful cafe or street-food scene focused on sweet comfort food.

![](images/f3a2117ba8d82525dc8467a06b7ee9ee54c2aa259329bd6217b299d5cf82baf1.jpg)  
Prompt: Heavy gray storm clouds gather low over city buildings, power lines, and rooftops, filling most of the frame with dark, textured sky. The urban skyline beneath looks small and tense, suggesting an approaching downpour or dramatic weather change.

![](images/9de0195a692b8627a5d61b4c25d6fb48b85a28eba594d9fda04af8892e8ed0d8.jpg)

![](images/5b9eaa74af3cf85d567e128d84f57e49e9ab88f14675c4c64febd5fe5aff98f7.jpg)

Prompt: A large steamed dumpling or bun rests on a round wooden plate, garnished with lettuce and small red tomatoes. The bright tabletop, folded cloth, and clean plating make the simple food presentation look fresh and carefully arranged.

![](images/7726f5c0370be95393c9f41adc34b6739351fa3231f755a282da9196256285a3.jpg)  
Prompt: A round celebration cake is decorated with pink frosting roses, green leaves, and a large bow across the top. Plates of food around it suggest a shared meal, while the elaborate icing makes the cake the table’s centerpiece.

Prompt: A sculptural bust shows a realistic human face emerging from a rough, crown-like form and textured reddish base. The plain wall behind it keeps focus on the unusual material contrast, facial detail, and handmade artistic expression.

Figure 22 Text-to-image examples (Part IV). Additional samples on travel, food, urban weather, and sculptural prompts, showing the model’s handling of varied lighting and textures.

![](images/a20f6d1ce245f6d197532c7e4e740cc0b947b0ddaa38bea7e001ac1a1d1a5194.jpg)

![](images/0aa7340e59ee5450ac14f7b942e589d5b9e3abf57113d034f3ffc489088584a1.jpg)

![](images/352949c7585f3c418576c4a294d6995b5e7bda6d4cb32355061e66a0dfb939e4.jpg)

Prompt: Rows of grilled skewers are packed tightly on a metal tray, coated in glossy red seasoning, herbs, and chili flakes. The close-up view highlights the rich sauce, charred edges, and spicy street-food appeal of the dish.

Prompt: Thick steam rises from a large pot filled with stew, meat, vegetables, and rich brown broth. The close view captures active cooking, with bubbling liquid and softened ingredients creating a hot, hearty, and aromatic kitchen scene.

Prompt: Small white flowers bloom in dense clusters among glossy green leaves, with many speckled petals and unopened buds visible. The close-up composition emphasizes delicate natural patterns, soft lighting, and the fresh detail of the flowering plant.

![](images/fe183075234c4a3cbedb00ea00e75d525622f894beafb0607c2ef2abcd2e4340.jpg)

![](images/92f230a21d95c9a301430a45d7806d5074369a124d06e666593c6d069c97afc4.jpg)

Prompt: A brilliant sun shines above a winding mountain road bordered by lush green plants. Distant ridges fade toward the blue horizon, while the intense light and clear sky create a bright, open, high-altitude travel scene.

![](images/6ecc3802196ca5648edc27480e60e85a0039c35dee5fcc7b182ce0cfcdbf8c91.jpg)  
Prompt: A monochrome mountain landscape drawing shows steep ridges, deep valleys, winding water, and layered terrain fading into the distance. The sketch-like shading gives the scene a rugged, hand-rendered quality, as if from a travel notebook.

Prompt: A vivid red boat rests on pale sand in front of calm turquoise water and a clear horizon. The simple composition emphasizes strong color contrast, seaside quietness, and the unusual stillness of a boat pulled ashore.

Figure 23 Text-to-image examples (Part V). Additional samples on culinary, floral, and travel prompts, together with an illustrative monochrome landscape.

![](images/784b5c71e4df528e6581251000da7ad26bb2f358b47650294ecd56ca10bb75a8.jpg)

![](images/52c17ded35ffe3f26e64fd0cdad976ff1cc292f4dee924f7e85ec38a1d1488d3.jpg)

Prompt: A young woman in a white coat and patterned scarf stands at night in front of glowing red and yellow lanterns. The warm lights and wooden structures behind her create a festive, atmospheric portrait with soft contrast.

![](images/d7c124f237142c9a56dcc06c459133c89f1dd8d55ea5b5487e1e86438d527622.jpg)  
Prompt: A close selfie shows a man wearing a dark bucket hat and light blue shirt while seated inside a vehicle. Black seats, side windows, and an orange curtain provide context for the casual travel setting.

Prompt: A man in a gray polo shirt stands on a balcony or rooftop, looking thoughtfully to the side. A blurred city skyline behind him gives the portrait an urban setting and a calm, reflective mood.

![](images/ba5ae871c879add21065b875d4aa7cb2000740bf3e40cb2a48c012cda1beb0be.jpg)

![](images/4a025daeaa6a7347ae628ddedc8c41304fb869b073e53ca3a51a39ad22dba449.jpg)

Prompt: An anime-style girl with long dark hair wears a purple patterned outfit with lace details, posed against a dreamy sky. Large moonlike circles, stars, and soft gradients create a romantic fantasy illustration style.

![](images/3ea8a874fef8760b17080a54ef3fc320891e3f4119c358f49dbbbb8ed53bdebc.jpg)  
Prompt: Two men in gray shirts stand shoulder to shoulder in front of a green screen. One smiles broadly while the other gives a restrained expression, creating a casual studio portrait with contrasting personalities.

Prompt: A woman with curly hair smiles gently in a very dark outdoor setting, lit softly from the front. The nearly black background isolates her face and shoulders, creating an intimate portrait with understated nighttime atmosphere.

Figure 24 Text-to-image examples (Part VI). Portrait-focused samples covering nighttime scenes, urban backdrops, stylized illustration, and studio-style group compositions.

![](images/c545736375af5bf1b9067837ebae0012f5ebe2675bb2e9b4842861df94b30dd1.jpg)

![](images/5ed651c61460c458e8b4ee703104c26906cadb14e4506beecf59d4e82e7f2dbb.jpg)

Prompt: Two adults sit at a restaurant table behind a birthday cake topped with lit candles. Warm indoor lighting, surrounding dishes, cups, and desserts suggest a celebratory meal shared in a relaxed dining setting.

![](images/8a6d5d65b9b88baee74a2b9d98456cff9be019c1c7c690aaee3e37b70c989025.jpg)  
Prompt: Two young women in coordinated school-style outfits with red bows take a close selfie outdoors. Trees, bicycles, and paved walkways in the background suggest a campus or park setting on a bright day.

Prompt: A fluffy white cat sits indoors with wide dark eyes, pink ears, and crossed front paws. The softly blurred background and centered framing make the cat’s round face and clean white fur especially prominent.

Figure 25 Text-to-image examples (Part VII). Final samples on celebratory dining, animal, and outdoor selfie prompts, completing the qualitative tour of the unified model’s text-to-image generation capability.

![](images/c4a61adf0d7db3844e518c17ed18f6a02cf2a28d0d83642b806b221ea33f0562.jpg)

![](images/f7c4fc41c330b4be08d2556c9acf082fdc5a8e2d90cf88cedf3e9c1984b5356a.jpg)

![](images/fa1d7e75d2fa64a8123f5c6432ce9ff7ad1bc563272386426a2ff1f92e895a2b.jpg)

Prompt: Describe the content of this image in detail.

Caption: The image shows a woman wearing an oversized long down jacket with a plain collar. The jacket features a letter “A” design on it.

Prompt: Describe the content of this image in detail.

Caption: The image shows a person wearing a black jacket and sunglasses, standing on a street.

Prompt: Describe the content of this image in detail.

Caption: The image shows a meeting room with several people, around six to eight, sitting around a long conference table and holding a meeting. A screen is mounted on the wall, and bowls, cups, and pens are placed on the table.

![](images/9639c2ebc0cf9e90f863b1219130bf02c6f25bec7add82818c729721d1e00b5a.jpg)

![](images/a513222a189db4ce7c5a60e6cd08aef8736054209e7a209a059ff8155aee10d4.jpg)

![](images/5badb6c6c2bd183e0053d178ded6494346b85cb3e15941d8f3488fa4c896530e.jpg)

Prompt: Describe the content of this image in detail.

Caption: The image shows a person wearing a grey hoodie with a purple design on the front. The background includes a white wall with a blue design.

Prompt: Describe the content of this image in detail.

Caption: The image shows several people playing a ball game on a court. The background includes a blue wall with some text, and the ground is also blue.

Prompt: Describe the content of this image in detail.

Caption: The image shows a person wearing a black suit with a white shirt underneath, seated.

Figure 26 Image $\times$ text-to-text examples (Part I). Additional qualitative results under the prompt template Describe the content of this image in detail.

![](images/091f59445da1fdd59caa5a75d121a3242a91c9e6b7e79c61c522e2bd8e6a10c4.jpg)

Prompt: Describe the content of this image in detail.

Caption: The image shows a cluster of ancient buildings with old-style architecture, featuring several old structures packed closely together.

![](images/49fe5e557da7552b781889891e3cb42e6f715048150b6aa834c400b791fce62f.jpg)

Prompt: Describe the content of this image in detail.

Caption: The image is a cityscape shot at dusk, showing an urban night scene with many tall high-rise buildings.

![](images/c5598300914f4d209061cdaf3d187d0441e4242c17c316b44821a85bc7c764b8.jpg)

Prompt: Describe the content of this image in detail.

Caption: The image shows a large football (soccer) stadium with the field inside, resembling an outdoor stadium. Another row of stands is visible on the other side of the field.

![](images/9621116f67ae7cbbe8aac93baccf5176582a1dc31cc5a4cb6f9ffe3b348752dd.jpg)

Prompt: Describe the content of this image in detail.

Caption: The image shows a person using a pen to write on a piece of paper.

![](images/94900d66e0a6ec234da72d5723007c2d10e25a445310c1dea35adc7cbaf2be56.jpg)

Prompt: What does this image depict? Give a comprehensive description.

Caption: The image shows a woman with long, dark hair wearing a black top. The background is a neutral color, with a window and a pink wall visible behind her. She appears to be standing indoors.

![](images/7d815abe90f098d79263503bc642a94bea31b3be5c5f4dbfc03c05e65683f711.jpg)

Prompt: What does this image depict? Give a comprehensive description.

Caption: The image shows a portrait of a young woman with blonde hair, wearing a white top and seated at a student desk with an open laptop. She appears to be speaking while using the laptop.

Figure 27 Image $\times$ text-to-text examples (Part II). Additional qualitative results with detailed descriptive prompts across indoor scenes, cityscapes, and document-oriented imagery.

![](images/516b47d0c956edaa4cb3325c43d0cc858f546ad025c9524e34d5080a2a7d9003.jpg)

![](images/dd0da326748ef1c72880b8f2380442ca9c5b0032bc9db07edbf1ca25c3a06d93.jpg)

![](images/f4e30278962133dc456af046b33e8ba284b1fe7c25b1a017f5bd62d1919943e9.jpg)

Prompt: What does this image depict? Give a comprehensive description.

Caption: The image shows a gravestone with English text carved on it.

Prompt: What is shown in this image? Please explain thoroughly.

Caption: The image shows a cluster of modern city buildings, including tall skyscrapers with distinctive and futuristic architectural designs.

Prompt: What is shown in this image? Please explain thoroughly.

Caption: The image is a medium shot of a young East Asian woman with long, straight dark hair, wearing a pink top and a necklace. She is posing with her hands raised near her face, making a gesture with her fingers.

![](images/4500d7008a520640bafa0e92a77eafd708674a83244b84bed3905d290d3bdb49.jpg)

![](images/ff70a96bfda679a8bb2853fce7fe57176b807c8517972b46b68bf19ce1d4bb66.jpg)

![](images/bc405dd8ffc1a5ee9542198c3bae54ae8c4d069a0bc07c938d94c41254aa668d.jpg)

Prompt: What is shown in this image? Please explain thoroughly.

Caption: The image shows a blue card or cover with a printed image and some text on it.

Prompt: What is shown in this image? Please explain thoroughly.

Caption: The image shows three people standing in an open outdoor area. They are wearing colored clothing, and one of them appears to be carrying something on their back.

Prompt: What is shown in this image? Please explain thoroughly.

Caption: The image shows a mountain range with a massive peak. The surface of the mountain reveals distinctive textures and patterns.

Figure 28 Image $\times$ text-to-text examples (Part III). Additional qualitative results under more open-ended explanatory prompts, spanning objects, portraits, outdoor scenes, and landscapes.

![](images/dbe5c2aac2ccfa8aa7261090d8de6f9b2906980061538fd0e109de3c2cd183cf.jpg)

![](images/5b8b4379580d3c835a3c9d94c006b91e87974e9d80fc35e5fd754217a50ad43e.jpg)

![](images/9ecd9198dd94534d18a7e352f73db869faf1025d4864aef0c620cdb26762ea6c.jpg)

Prompt: What is shown in this image? Please explain thoroughly.

Caption: The image shows a man wearing a plaid shirt, working in a studio or workshop. He is focused on an object on a white surface in front of him.

Prompt: What is shown in this image? Please explain thoroughly.

Caption: The image shows a newlywed couple at a wedding, dressed in wedding attire.

Prompt: What is shown in this image? Please explain thoroughly.

Caption: The image is a cartoon-style drawing of a character dressed in a colored outfit, with a body showing green and yellow stripes. The character is touching its face with one hand.

![](images/547481a51f7eb5389535e9424ef2697ac9972145f128677d43a85cd0fd2dc4c6.jpg)

![](images/01d021d781599b55f1404f9fda4816091a0e6dd987c6fb3ff9232134c6783671.jpg)

![](images/2e1c67d96c6ab03a127ea52c59f16b1c73e8ebdd7806fec7d63dd0e52ed44001.jpg)

Prompt: Describe what you see in this picture.

Caption: The image shows musicians performing with instruments at a ceremony. The background includes a floor and a wall.

Prompt: Describe what you see in this picture.

Caption: The image shows a black sleeveless T-shirt with a printed design on the front.

Prompt: Describe what you see in this picture.

Caption: The image shows a person wearing a white blouse over a black and white striped skirt.

Figure 29 Image $\times$ text-to-text examples (Part IV). Additional qualitative results over people-centric, object-centric, and illustration-style inputs.

![](images/67e71988ce59c8ec6e5913d5a4589fc67de41965f3f7b95357fe1d4bcc9a4d65.jpg)

![](images/b8e2276b9015e13b29ec956fe1caccdbb31c377f302dda6cead97400c286a4e5.jpg)

![](images/f777978e628dff149f04a84edf711b76f741f77bfb79a2ea0a84a61f97e4fd69.jpg)

Prompt: Describe what you see in this picture.

Caption: The image shows a white Toyota car with a sleek design, featuring prominent front and side headlights.

Prompt: Describe what you see in this picture.

Caption: The image shows a man wearing a black baseball cap and a grey shirt. The focus is on the cap, and the background is a plain, neutral grey.

Prompt: Describe what you see in this picture.

Caption: The image shows a necklace with a red, ball-shaped pendant.

![](images/2bdb3fa19a7c41de44d0018f6e38a618ea2db6f8b0d10b02bd77b89a2ba2b3fa.jpg)

![](images/cf5ce8cf91f703470957c529c0fbe9b275dc021d8b0cdff94a9e021b1b802ac4.jpg)

![](images/3f18e7782a36fdb993bcd59d5e2568b600c0100204034b0bda2a27d2185184b5.jpg)

Prompt: Describe what you see in this picture.

Caption: The image shows a person standing on a paved area with a scenic green backdrop.

Prompt: Describe what you see in this picture.

Caption: The image shows an old building with distinctive traditional Chinese-style architecture. It features traditional entrance details, and an iron structure with a roof is visible above the building.

Prompt: Describe what you see in this picture.

Caption: The image shows a decorative ornament in the shape of a flower.

Figure 30 Image $\times$ text-to-text examples (Part V). Additional qualitative results on apparel, accessories, vehicles, architecture, and scenic human portraits.

![](images/19245c51bfb309c30485d62e004347904a24547980ab23056af6b47a90412ef1.jpg)

![](images/8037842c5dff67cb411a20106460c3b0670ffc9e0d718e3fea0b9a3baec0f476.jpg)

![](images/11265528dc63efa2859a2bf6c43cf03f4d770efcd9cdc5d64dd89b5954d3cfc9.jpg)

Prompt: Describe what you see in this picture.

Caption: The image shows several people playing volleyball, with one of them jumping up. The scene takes place under a clear afternoon sky.

Prompt: Generate a detailed caption for this image.

Caption: The image shows a soccer player in action during a match, wearing a numbered uniform. The player appears to be running and kicking the ball on the field.

![](images/9821ed6c0850698d5f19effec9cb9fe95c18d6630c7f617c1d7f8e282bd73132.jpg)

![](images/3c88752d17cab551917eaa04cb062e49720d16cb71aaf31b310abcb942b3638d.jpg)

Prompt: Generate a detailed caption for this image.

Caption: The image shows a black T-shirt with a design featuring a blue cat printed on the front.

![](images/4333b62dcab7a41c4c038542aa172722db4146f559777aecd16b8d06106da63c.jpg)

Prompt: Generate a detailed caption for this image.

Caption: The image shows a young boy wearing a grey vest over a white shirt, with a blue accent.

Prompt: Generate a detailed caption for this image.

Caption: The image shows a young girl with straight hair, wearing a yellow crop top and white shorts. She is making a fist with one hand and jumping, set against a blue background.

Prompt: Generate a detailed caption for this image.

Caption: The image shows the stands of a large soccer stadium. The field and running track are visible, with green grass surrounding the stands and a few spectators present.

Figure 31 Image $\times$ text-to-text examples (Part VI). Additional qualitative results for sports, apparel, portrait, and stadium scenes, including caption-generation prompts.