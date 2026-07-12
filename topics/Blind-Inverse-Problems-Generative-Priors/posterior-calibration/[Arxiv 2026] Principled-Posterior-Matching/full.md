# Unbiased Diffusion Variational Inversion via Principled Posterior Matching

Weimin Bai<sup>\*</sup>, Yuxuan Gu<sup>\*</sup>, Yifei Wang, Weijian Luo, He Sun<sup>†</sup> Peking University

Abstract—Existing score-based methods for inverse problems often resort to approximate minimization of the KL divergence between the inversion distribution and the Bayesian posterior. Such an approximation leads to severe mode collapse and unreliable uncertainty quantification. In this paper, we propose Principled Posterior Matching (PPM), a framework that returns to the fundamentals of variational inference, rather than using tricky approximations. Instead of relying on heuristic approximations, we rigorously formulate the exact optimization of the KL divergence via the integration of Fisher divergence. We derive a tractable, equivalent gradient form of this integral, enabling precise optimization without the biases introduced by prior approximations. Our analysis clearly reveals that the mode collapse in previous methods stems directly from this approximation gap. Supported by our theoretical solution, PPM unifies two complementary paradigms: (1) In variational inference, PPM adopts masscovering divergences that significantly improve the inversion diversity and uncertainty quantification; (2) In amortized inference, it enables the training of an efficient reconstruction network for rapid, single-step reconstruction. Furthermore, our formulation naturally extends to a broader family of divergence measures by generalizing the integral of the Fisher divergence. We validate PPM across challenging computational imaging tasks, including inpainting, super-resolution fluorescent microscopy, and radio interferometric black-hole imaging. In all experiments, PPM achieves superior reconstruction fidelity, faithful multimodal posterior recovery, and well-calibrated uncertainty estimates, establishing a robust framework for scientific imaging.

Index Terms—Computational Imaging, Variational Inference, Amortized Inference, Diffusion Models, Uncertainty Quantification

## I. INTRODUCTION

$\pmb { \mathcal { A } } ( \pmb { x } ) + \pmb { \eta } .$ , where A is a linear or nonlinear, ill-posed forward operator, and η denotes measurement noise. Computational imaging finds applications in diverse scientific fields such as astronomy Chael et al. [2019], optical microscopy Choi et al. [2007], medical imaging Lustig et al. [2008], and fluid dynamics Iglesias et al. [2013]. A crucial aspect of imaging in this context is the incorporation of an image prior, or so-called regularization, to guide the reconstruction towards desired image characteristics. From a Bayesian perspective, the prior shapes the posterior distribution, namely the uncertainty and diversity, of the reconstructed images. Classical methods impose handcrafted priors on x—for example, sparsity Candes and Romberg [2007], total variation (TV) Vogel and Oman [1996], or wavelet-based regularizers—to constrain the solution space. While effective in some settings, these analytical priors usually fail to capture the rich statistics of natural or scientific images.

Recent advances in generative AI have established diffusion models Ho et al. [2020], Sohl-Dickstein et al. [2015], Song et al. [2020] as powerful, data-driven image priors. A diffusion model learns to approximate the distribution of clean images by progressively adding noise (the forward process) and then training a neural network to reverse this corruption through denoising (the reverse process). When integrated into inverse problem solvers, diffusion priors not only yield high-quality point estimates but also facilitate full posterior exploration for uncertainty quantification, a capability crucial in scientific and medical imaging applications.

Existing approaches for posterior sampling from p(x|y) with a pretrained diffusion prior primarily fall into three categories: gradient-guided Monte Carlo sampling, optimizationbased variational inference, and amortized inference. The first category comprises gradient-guided Monte Carlo (MC) techniques, exemplified by Diffusion Posterior Sampling (DPS) Chung et al. [2022]. DPS iteratively interleaves denoising updates from a score-based model with gradient-driven data-consistency steps within a Markov Chain Monte Carlo (MCMC) framework Brooks et al. [2011]. While this procedure seeks to satisfy both the measurement likelihood p(y|x) and the learned image prior, it enforces data consistency by approximating the intractable time-dependent likelihood with strong assumptions. This often results in unstable sampling trajectories and biased posterior estimates. Moreover, these methods remain inherently inefficient, as their sequential nature incurs substantial computational costs during inference.

The second category encompasses optimization-based variational inference (VI), such as RED-Diff Mardani et al. [2023] and Score Distillation Sampling (SDS) Poole et al. [2022]. These methods typically formulate the inverse problem as optimizing a weighted denoising score matching objective. However, a critical theoretical limitation arises from their implicit assumption of a degenerate variational distribution (e.g., a Dirac delta). Such a point-estimate assumption effectively eliminates the entropy term from the variational objective, theoretically degrading the optimization into Maximum A Posteriori (MAP) estimation rather than probabilistic inference. Consequently, without the entropy term to encourage diversity, the learned posterior suffers from severe modeseeking behavior Luo et al. [2024a] and structural collapse, leading to an underestimation of uncertainty and a failure to capture the complex, multi-modal structure of the posterior.

![](images/80903e358f47d7bf8f673112fc75e63fd26d60c5df0dde77c9bc210464f54f8f.jpg)  
Fig. 1. Comparison of PPM and diffusion-based VI baselines (RED-Diff and RLSD) on 2D posterior estimation tasks. Priors are multimodal Gaussia mixtures, and observations are linear projections of the 2D latent state, producing inherently multimodal ground-truth posteriors. PPM accurately recovers all modes, whereas RED-Diff suffers from mode collapse, and RLSD exhibits measurement inconsistency (top) or poor prior adherence (bottom).

The third paradigm focuses on amortized inference (AI), represented by methods like DAVI Lee et al. [2024]. These approaches aim to train a feed-forward network to predict the posterior directly for rapid inference.

Though practically useful, DAVI actually deviates from the standard unsupervised inverse problem framework – solving the Bayesian posterior problem – in two critical aspects. First, DAVI typically rely on paired training data, introducing supervised objectives—including pixel-wise reconstruction loss against ground truth and adversarial (GAN) losses—to align the reconstruction network for reconstruction. This reliance on ground truth supervision restricts their applicability in scientific imaging, where obtaining paired data is often impractical. Second, regarding the optimization objective, these methods fail to minimize the exact KL divergence. Instead, they adopt the Integral KL (IKL) divergence proposed in Diff-Instruct Luo et al. [2023], which serves as a heuristic approximation rather than a rigorous variational lower bound. This deviation from exact KL minimization, as well as the dependence on supervision, shifts the training paradigm from sampling from the exact Bayesian posterior distribution to some approximated distribution with a significant bias, which fundamentally limits their ability to provide the reliable, zeroshot generalization required for solving inverse problems with generic priors.

To overcome the limitations imposed by the inexact approximations of the KL divergence in existing works Mardani et al. [2023], Zilberstein et al. [2024], we propose Reconstruction Score Matching (PPM). Instead of relying on the delta distribution assumption or IKL objectives, PPM returns to the principled minimization of the exact KL divergence. We achieve this via a classical theoretical result: the Kullback-Leibler divergence can be expanded as the integration of Fisher divergence along the diffusion process. Which leads us to optimize the integral of Fisher divergence rather than the integral of KL divergence in order to sample from the exact Bayesian posterior distribution. Such a problem formulation provides a rigorous theoretical grounding: we prove that minimizing the exact KL via Fisher integration inherently encourages mass-covering behavior, effectively mitigating mode collapse. Furthermore, we demonstrate that this Fisher-based perspective can be naturally generalized to a broader family of divergences via generalized score matching, theoretically unifying existing score-based methods as special cases within our framework. Table I provides a systematic theoretical comparison, summarizing the specific sources of bias in these baselines—ranging from likelihood approximation in DPS to the independence assumption in IKL—and contrasting them with the unbiased nature of PPM. The detailed theoretical analysis of baseline bias is provided in Section IV.

To make the Fisher divergence optimization tractable, we derive a novel equivalent gradient formula from the Fisher divergence integral. Unlike prior methods that rely on biased estimators, our derivation yields a precise gradient with respect to the variational parameters, enabling efficient optimization via stochastic gradient descent. In PPM, we model the variational posterior $p ( { \pmb x } | { \pmb y } )$ flexibly to support two paradigms: either as an amortized reconstruction network $\{ \pmb { x } = g _ { \varphi } ( \pmb { z } )$ , z ∼ $\mathcal { N } ( 0 , 1 ) \}$ for rapid, single-step inference, or as an ensemble of image particles $\{ \mu _ { k } \} _ { k = } ^ { K }$ for high-fidelity particle-based VI. To accurately compute the divergence, we introduce an auxiliary score network $s _ { \phi } ,$ adapted from the pre-trained diffusion score $s _ { p }$ via Low-Rank Adaptation (LoRA) Hu et al. [2022], Ryu [2023], which bridges the gap between the variational score and the true posterior score.

We evaluate PPM on a comprehensive suite of inverse imaging benchmarks. To ensure robust and comparable assessments, we first validate our method on standard tasks—including image inpainting, super-resolution, and deblurring—using established datasets such as FFHQ Karras et al. [2019] and ImageNet Deng et al. [2009] at $2 5 6 \times 2 5 6$ resolution. Beyond natural images, we rigorously test PPM on two challenging scientific imaging applications: superresolution fluorescence microscopy Qiao et al. [2021] and radio-interferometric imaging of black holes Chael et al. [2019], Sun and Bouman [2021], Sun et al. [2022]. Across all experiments, PPM consistently outperforms leading baselines, including DPS Chung et al. [2022], KL-based VI methods Mardani et al. [2023], Zilberstein et al. [2024], and the amortized approach DAVI Lee et al. [2024]. Crucially, PPM achieves these results without relying on paired training data, delivering superior reconstruction fidelity, markedly greater sample diversity, and more reliable uncertainty quantification.

TABLE I  
THEORETICAL COMPARISON OF OPTIMIZATION OBJECTIVES ACROSS DIFFERENT FRAMEWORKS.
<table><tr><td>Method</td><td>Objective / Mechanism</td><td>Bias Source</td><td>Posterior Type</td><td>Uncertainty</td></tr><tr><td>DPS Chung et al. [2022] RED-Diff Mardani et al. [2023]</td><td>Likelihood Approx. Weighted Score Matching</td><td>Laplacian Approximation No Entropy (Dirac)</td><td>Approx. Sampler Point Estimate (MAP)</td><td>Unreliable None</td></tr><tr><td>RLSD Zilberstein et al. [2024]</td><td>SDS + Repulsion</td><td>Heuristic Repulsion (Proxy Entropy)</td><td>Particles + Forced Diversity</td><td>Artificial</td></tr><tr><td>DAVI (IKL) Lee et al. [2024]</td><td> $\textstyle \int D _ { \mathrm { K L } } ( q _ { t } ] | p _ { t } ) d t$ </td><td>Ignores Temporal Dependency</td><td>Amortized Approx.</td><td>Biased</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PPM (Ours)</td><td> $\textstyle { \int } \mathrm { F i s h e r } ( q _ { t } \| p _ { t } ) d t$ </td><td>Unbiased (Exact KL)</td><td>True Variational</td><td>Exact &amp; Calibrated</td></tr></table>

## II. BACKGROUND

## A. Computational Imaging Inverse Problems

In general, the inverse problem aims to reconstruct underlying image signals $\pmb { x } \in \mathbb { R } ^ { d }$ from corrupted observations $\pmb { y } \in \mathbb { R } ^ { m }$ , where the image formation process is probabilistically modeled as:

$$
\begin{array} { r } { \pmb { y } \sim p ( \pmb { y } | \pmb { x } ) . } \end{array}\tag{1}
$$

Since the observation are usually under-determined $( m \leq d )$ and observation noise is inevitable, inverse problems in computational imaging are typically ill-posed, with the inverse mapping ${ \textbf { \textit { y } } } \to { \textbf { \em x } }$ being one-to-many. To address this complexity, Bayesian inference framework introduces a prior distribution of underlying images, $p ( { \pmb x } )$ , to constrain the solution space for the image posterior, $p ( { \pmb x } | { \pmb y } )$ , as illustrated by:

$$
p ( { \pmb x } | { \pmb y } ) = \frac { p ( { \pmb y } | { \pmb x } ) p ( { \pmb x } ) } { p ( { \pmb y } ) } \propto \underbrace { p ( { \pmb y } | { \pmb x } ) } _ { \mathrm { L i k e l i h o o d ~ P r i o r } } \underbrace { p ( { \pmb x } ) } _ { \mathrm { P r i o r } }\tag{2}
$$

Employing Maximum a Posteriori (MAP) estimation, one can derive a point estimate of the underlying image by maximizing $\log p ( { \pmb x } | { \pmb y } )$ . Alternatively, posterior image samples of reconstructed images can be obtained through methods like Markov Chain Monte Carlo (MCMC) Brooks et al. [2011] or Variational Inference (VI) Blei et al. [2017], Sun and Bouman [2021], Sun et al. [2022]. However, the performance of many computational imaging solvers is limited by their reliance on oversimplified, handcrafted priors such as sparsity Candes and Romberg [2007] and total variation (TV) Vogel and Oman [1996]. These priors fail to capture the true complexity of natural image distributions, hindering the solvers’ ability to achieve high-quality reconstructions.

## B. Diffusion Models

Diffusion models Ho et al. [2020], Sohl-Dickstein et al. [2015], Song et al. [2020] formulate generation as the reverse of a continuous-time diffusion process defined by a stochastic differential equation (SDE). The forward SDE gradually corrupts data by injecting noise:

$$
\mathrm { d } \pmb { x } _ { t } = f ( \pmb { x } _ { t } , t ) \mathrm { d } t + g ( t ) \mathrm { d } \pmb { w } ,\tag{3}
$$

where $t \in [ 0 , T ]$ indexes the diffusion time, $f ( \cdot , t ) : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ controls the drift coefficient, g(t) scales the Brownian motion w, and $\mathbf { \boldsymbol { x } } _ { 0 } ~ \sim ~ p _ { \mathrm { d a t a } }$ . This process gradually transforms data samples into a tractable Gaussian distribution $\pmb { x } _ { T } \sim \mathcal { N } ( \mathbf { 0 } , I )$ The generative process then follows the corresponding reversetime SDE:

$$
\mathrm { d } \pmb { x } _ { t } = \left[ f ( \pmb { x } _ { t } , t ) - g ( t ) ^ { 2 } \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } ) \right] \mathrm { d } t + g ( t ) \mathrm { d } \overline { { \pmb { w } } } ,\tag{4}
$$

where $\nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } )$ is the score function estimated by a neural network $s _ { \theta } ( \pmb { x } _ { t } , t )$ . Training such a neural network usually involves optimizing a score matching objective Song and Ermon [2019]:

$$
\begin{array} { r } { \mathcal { L } ( \theta ) = \mathbb { E } _ { t , \mathbf { x } _ { t } } \left[ \lambda ( t ) \left| \left| s _ { \theta } ( \mathbf { x } _ { t } , t ) - \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) \right| \right| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{5}
$$

where $\lambda ( t )$ reweights time steps and $p _ { t }$ is the perturbation kernel of the forward process. Once trained, we can plug $s _ { \theta } ( \pmb { x } _ { t } , t )$ into Eq. 4 and sample images from a random noise following Eq. 4 or variants Karras et al. [2022], Li et al. [2025], Lu et al. [2022], Xue et al. [2023]. Supported by solid theories, the diffusion model has successes in a wide range of applications Bai et al. [2025a,b], Chen et al. [2023], Chi et al. [2025], Deng et al. [2024], Janner et al. [2022], Saharia et al. [2022], Ye et al. [2024], Zhang et al. [2023]. In the next section, we will focus on diffusion models for inverse problems.

## C. Diffusion Model for Inverse Problems

Diffusion models Ho et al. [2020], Sohl-Dickstein et al. [2015], Song et al. [2020], owing to their strong ability to accurately approximate complex image distributions, have emerged as powerful data-driven priors for imaging inverse problems. There are two main categories of diffusion-model–based solvers for imaging inverse problems. The first builds on MCMC sampling Brooks et al. [2011], using score functions to guide gradient-based samplers that steer reconstructions toward the learned image prior. The second employs variational inference frameworks with score distillation sampling (SDS) techniques Mardani et al. [2023], Poole et al. [2022], Zilberstein et al. [2024], enforcing similarity between reconstructed images and the diffusion prior by minimizing the KL divergence between them. Below, we provide an overview of each approach.

a) Monte Carlo sampling methods: After training on large image datasets, a diffusion model provides the unconditional score $\nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } )$ . Posterior sampling replaces this with the conditional score $\nabla _ { \mathbf { x } _ { t } }$ log $p _ { t } ( \mathbf { x } _ { t } \mid \mathbf { \mu } y )$ during reverse diffusion, via Bayes’ rule:

$$
\begin{array} { r l } & { \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } \mid y ) = \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } ) + \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( y \mid \mathbf { x } _ { t } ) } \\ & { ~ \approx s _ { \theta } ( \mathbf { x } _ { t } , t ) + \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( y \mid \mathbf { x } _ { t } ) , } \end{array}\tag{6}
$$

where $s _ { \theta } ( \mathbf { x } _ { t } , t )$ is the learned score network. The principal challenge of posterior sampling lies in approximating the timedependent likelihood term $\nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( y \mid \mathbf { x } _ { t } )$

A popular solution—Diffusion Posterior Sampling (DPS) Chung et al. [2022]—approximates

$$
p _ { t } ( y \mid \mathbf { x } _ { t } ) = \int p ( y \mid \mathbf { x } _ { 0 } ) p ( \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ) d \mathbf { x } _ { 0 } \approx p \big ( y \mid \hat { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { t } ) \big ) ,\tag{7}
$$

where $\hat { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { t } ) ~ = ~ \mathbb { E } [ \mathbf { x } _ { 0 } ~ | ~ \mathbf { x } _ { t } ]$ . This point-estimate approach is computationally efficient and yields strong empirical performance. Alternative schemes approximate both $p ( \boldsymbol { y } \mid \mathbf { x } _ { 0 } )$ and $p ( \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } )$ as Gaussians to better capture uncertainty Zhu et al. [2023]. However, most of these approaches assume linear forward models and do not readily extend to nonlinear inverse problems.

Building on Plug-and-Play (PnP) optimization Graikos et al. [2022], Zhu et al. [2023], stochastic PnP Monte Carlo algorithms—such as Generative PnP (GPnP) Bouman and Buzzard [2023] and PnP Monte Carlo (PMC) Sun et al. [2023]—alternate between data-consistency and prior-refinement steps to approximate the full posterior. Structurally, these methods resemble DPS, but by avoiding point-estimate approximations, they admit theoretical convergence to the true posterior (albeit at a higher cost). Recent advances further improve sampling via Sequential Monte Carlo Cardoso et al. [2023], Dou and Song [2024], Trippe et al. [2022], Wu et al. [2023] or variablesplitting techniques Cai et al. [2025], Chen et al. [2022], Coeurdoux et al. [2024], Hu et al. [2026], Lee et al. [2021], Li et al. [2024], Song et al. [2023], Wu et al. [2024], Xu and Chi [2024], Zhang et al. [2024], and extend applicability to nonlinear inverse problems.

Despite their strengths, Monte Carlo sampling methods still face key limitations: they may require many iterations (leading to high computational cost and slow convergence in high dimensions), their approximation errors can introduce bias, performance often depends sensitively on hyperparameters, and they lack amortized inference for rapid repeated use.

b) Variational inference methods via Weighted score matching objective: Inverse imaging problems are inherently ill-posed, where a single observation y can be consistent with multiple latent ground-truth images $\mathbf { x } _ { \mathrm { 0 } }$ . By combining the measurement forward model with a learned diffusion prior via Bayes’ rule, one can define the posterior distribution $p ( \mathbf { x } _ { 0 } | \pmb { y } ) \propto p ( \pmb { y } | \mathbf { x } _ { 0 } ) p ( \mathbf { x } _ { 0 } )$ . However, directly sampling from this posterior is intractable. Variational Inference (VI) addresses this by approximating the true posterior $p ( \mathbf { x } _ { 0 } | \mathbf { y } )$ with a tractable variational distribution $q ( \mathbf { x } _ { 0 } | \mathbf { y } )$ . The objective is to minimize the Kullback-Leibler (KL) divergence between this variational approximation and the true posterior:

$$
\begin{array} { r l } & { \underset { \ b { q } } { \mathop { \operatorname* { m i n } } } D _ { \mathrm { K L } } \big ( q ( \mathbf { x } _ { 0 } | \mathbf { y } ) \ \| \ p ( \mathbf { x } _ { 0 } | \mathbf { y } ) \big ) } \\ & { = \underbrace { - \mathbb { E } _ { q ( \mathbf { x } _ { 0 } | \mathbf { y } ) } \left[ \log p ( \mathbf { y } | \mathbf { x } _ { 0 } ) \right] } _ { \mathcal { L } _ { \mathrm { d a a } } } + \underbrace { D _ { \mathrm { K L } } \big ( q ( \mathbf { x } _ { 0 } | \mathbf { y } ) \ \| \ p ( \mathbf { x } _ { 0 } ) \big ) } _ { \mathcal { L } _ { \mathrm { p r i o r } } } + \underbrace { \log p ( \mathbf { y } ) } _ { \mathrm { c o n s t } } } \end{array}\tag{8}
$$

The first term, ${ \mathcal { L } } _ { \mathrm { d a t a } }$ , enforces data fidelity consistent with the forward operator. The core challenge lies in minimizing the second term, ${ \mathcal { L } } _ { \mathrm { p r i o r } }$ , which aligns the variational distribution with the diffusion prior.

Existing methods, such as Score Distillation Sampling (SDS) Poole et al. [2022] and RED-Diff Mardani et al. [2023], simplify this optimization by implicitly assuming that the variational posterior $q ( \mathbf { x } _ { 0 } | \mathbf { y } )$ is a degenerate Dirac delta distribution $q ( \mathbf { x } _ { 0 } | \mathbf { y } ) = \delta ( \mathbf { x } _ { 0 } - \pmb { \mu } )$ (or a Gaussian with vanishing variance $\sigma \  \ 0 )$ centered at the estimated parameters $\mu .$ Under this point-estimate assumption, the entropy term of the variational distribution is effectively discarded. Consequently, the minimization of the KL divergence simplifies to a weighted score matching objective for the point estimate $\textstyle \mu ($

$$
\operatorname* { m i n } _ { \mu } \quad \underbrace { \| y - \mathcal { A } ( \mu ) \| ^ { 2 } } _ { \mathcal { L } _ { \mathrm { d a t a } } } + \lambda \underbrace { \mathbb { E } _ { t , \epsilon } \left[ \omega ( t ) \left\| \epsilon _ { \theta } ( \alpha _ { t } \mu + \sigma _ { t } \epsilon , t ) - \epsilon \right\| _ { 2 } ^ { 2 } \right] } _ { \mathcal { L } _ { \mathrm { p r i o r } } } ,\tag{9}
$$

where $\boldsymbol { \mathcal { A } } ( \cdot )$ represents the forward operator, and $\omega ( t )$ is a weighting function (often chosen heuristically based on SNR). While computationally tractable, this formulation represents a \*\*biased approximation\*\* of the true variational objective. By enforcing a degenerate distribution and neglecting the entropy term, the optimization theoretically degrades into Maximum A Posteriori (MAP) estimation. This induces significant optimization bias, manifesting as mode-seeking behavior, where the single estimate $\pmb { \mu }$ fails to capture the necessary diversity and uncertainty of the full solution space.

Several recent methods have adopted Eq. 9 for VI posterior estimation. For instance, Feng and Bouman [2023], Feng et al. [2023] integrate normalizing flows Dinh et al. [2016], Kingma and Dhariwal [2018] with diffusion models for accurate posterior modeling. However, their performance is constrained to lower-dimensional signals $( \mathbf { e . g . , \ 6 4 \times 6 4 } )$ due to inherent limitations in normalizing flow’s scalability. Recently, RED-Diff Mardani et al. [2023] proposes a variational approach that combines the prior loss with a data fidelity term to optimize an estimate of the clean image x. VSS He et al. [2024] manages to adopt the VI approach to solve zero-shot sparse-view CT reconstruction with a latent diffusion model. However, these methods have been observed to usually suffer from mode collapse issues. To address this, RLSD Zilberstein et al. [2024] adds a repulsive penalty between similar reconstructions. Although this increases sample diversity, its empirical assumptions limit gains in full-posterior recovery, and mode collapse remains an issue. Our method aims to fundamentally overcomes these challenges by embedding a score-based divergence distillation loss within a variational inference framework.

c) Amortized Inference for Inverse Problems via Integral KL Divergence: Unlike optimization-based methods that solve for a specific instance, amortized inference aims to learn a parametric reconstruction network $\pmb { x } _ { 0 } = g _ { \varphi } ( \pmb { y } )$ that maps observations directly to the posterior samples. Recent approaches like DAVI Lee et al. [2024] adopt the training objective from Diff-Instruct Luo et al. [2023], replacing the standard KL divergence with a heuristic metric known as the IKL divergence. IKL modifies the objective by manually introducing a time-weighting function $\omega _ { t }$ and integrating the marginal KL divergences over the entire diffusion process:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { I K L } } ( \varphi ) = \displaystyle \int _ { 0 } ^ { T } \omega _ { t } D _ { \mathrm { K L } } \big ( q _ { \varphi , t } ( \pmb { x } _ { t } | \boldsymbol { y } ) \big | \big | p _ { t } ( \pmb { x } _ { t } ) \big ) \mathrm { d } t } \\ & { = \displaystyle \int _ { 0 } ^ { T } \omega _ { t } \mathbb { E } _ { q _ { \varphi , t } ( \pmb { x } _ { t } | \boldsymbol { y } ) } \left[ \log \frac { q _ { \varphi , t } ( \pmb { x } _ { t } | \boldsymbol { y } ) } { p _ { t } ( \pmb { x } _ { t } ) } \right] \mathrm { d } t , } \end{array}\tag{10}
$$

where $q _ { t } ( \pmb { x } _ { t } )$ is the distribution of the generated sample diffused to time t. Assuming the reconstruction network output is deterministic given y (or the implicit distribution is approximated as Gaussian), the marginal distribution $q _ { \varphi , t } ( { \pmb x } _ { t } )$ becomes a Gaussian centered at $\alpha _ { t } g _ { \varphi } ( \pmb { y } )$ . Consequently, its score $\nabla _ { \pmb { x } _ { t } } \log q _ { \varphi , t } ( \pmb { x } _ { t } | y )$ is analytically computable. Following the derivation in Diff-Instruct Luo et al. [2023], the gradient of this objective with respect to the reconstruction network parameters $\varphi$ avoids backpropagation through the frozen score network $p _ { t }$ , and is given by:

$$
\begin{array} { l } { \displaystyle \int _ { 0 } ^ { T } \omega _ { t } \mathbb { E } _ { \pmb { x } _ { 0 } = g _ { \varphi } ( y ) , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) , \pmb { x } _ { t } = \alpha _ { t } \pmb { x } _ { 0 } + \sigma _ { t } \epsilon } } \\ { \displaystyle \Bigg [ \big ( \nabla _ { \pmb { x } _ { t } } \log q _ { \varphi , t } ( \pmb { x } _ { t } | y ) - \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } ) \big ) ^ { \top } \frac { \partial \pmb { x } _ { t } } { \partial \varphi } \Bigg ] \mathrm { d } t , } \end{array}\tag{11}
$$

where $\omega _ { t }$ is the weight of different time step t.

While ${ \mathcal { L } } _ { \mathrm { I K I } }$ provides a gradient signal for aligning the reconstruction network with the diffusion prior, it is crucial to note that Eq. 10 is not an exact estimation of the true posterior KL divergence $D _ { \mathrm { K L } } ( q _ { 0 } ( \pmb { x } _ { 0 } | \pmb { y } ) | | p ( \pmb { x } _ { 0 } ) )$ ). The transformation from the original KL to the time-integrated IKL relies on heuristic weighting $\omega _ { t }$ and ignores the temporal dependencies of the diffusion trajectory. This discrepancy implies that minimizing ${ \mathcal { L } } _ { \mathrm { I K L } }$ does not guarantee minimization of the actual variational bound, leading to biased posterior estimation and limited sample diversity compared to exact optimization. Besides the IKL divergence minimization, some other works have studied amortized inference in the context of diffusion acceleration Luo [2023, 2024], Luo et al., 2024a,b, 2025], Wang et al. [2024, 2025], Yin et al. [2024], Zhou et al. [2024a,b].

## III. METHOD

In this section, we present PPM, a principled framework for posterior recovery in computational imaging inverse problems. PPM introduces a novel score-based divergence guides optimization via an unbiased gradient estimator, which demonstrates unparalleled performance in both variational inference and amortized inference. Unlike the asymmetric mathematical form of KL divergence—whose tendency toward mode collapse limits reliable posterior estimation—our divergence provides a stable, unbiased surrogate objective that extends naturally to amortized settings, where an inference network learns to approximate posterior samples across instances. This formulation enables PPM to enhance both VI-based and AIbased methods, outperforming existing approaches including Score Distillation Sampling (SDS) Poole et al. [2022], RED-Diff Mardani et al. [2023], and Diffusion Prior-Based Amortized Variational Inference (DAVI) Lee et al. [2024].

## A. Problem Formulation

Following Bayes’ rule in Eq. 8, the optimization problem of classical VI-based inverse imaging solvers can be formulated as (neglecting the constant log p(y)):

$$
\begin{array} { l } { \displaystyle \varphi ^ { * } : = \arg \operatorname* { m i n } _ { \varphi } D _ { \mathrm { K L } } \big ( q _ { \varphi } ( { \pmb x } | { \pmb y } ) | | p ( { \pmb x } | { \pmb y } ) \big ) } \\ { \displaystyle \qquad = \arg \operatorname* { m i n } _ { \varphi } \frac { 1 } { 2 \sigma ^ { 2 } } | | { \pmb y } - \mathcal { A } ( \pmb { x } ) | | ^ { 2 } + D _ { \mathrm { K L } } \left( q _ { \varphi } ( { \pmb x } | { \pmb y } ) \| p ( { \pmb x } ) \right) , } \end{array}\tag{12}
$$

where the first term enforces data fidelity consistent with observations y via the forward model $\boldsymbol { \mathcal { A } } ( \cdot )$ . The second term encourages the variational posterior distribution $q _ { \varphi } ( { \pmb x } | { \pmb y } )$ to align with the prior distribution $p ( { \pmb x } )$ implicitly learned by a pre-trained diffusion model.

However, directly optimizing Eq. 12 with approximate objectives often induces mode collapse. To address this, we replace the standard KL term with a formulation based on the integration of Fisher divergence, which allows for exact and unbiased optimization Song et al. [2021].

## B. Exact Optimization via Reconstruction Score Matching

We reformulate the KL divergence in Eq. 12 using the integral of the Fisher divergence. This leads to the following exact optimization objective:

$$
\begin{array} { l } { \displaystyle D _ { \mathrm { K L } } ( q _ { \varphi } ( \boldsymbol x | \boldsymbol y ) | | p ( \boldsymbol x ) ) } \\ { \displaystyle = \frac { 1 } { 2 } \int _ { 0 } ^ { T } g ( t ) ^ { 2 } \mathbb { E } _ { \begin{array} { l } { \boldsymbol x _ { 0 } \sim q _ { \varphi } ( \cdot | \boldsymbol y ) } \\ { \boldsymbol x _ { t } | \boldsymbol x _ { 0 } \sim p ( \boldsymbol x _ { t } | \boldsymbol x _ { 0 } ) } \end{array} } \left[ d ( s _ { q _ { \varphi , t } } ( \boldsymbol x _ { t } | \boldsymbol y ) - s _ { p _ { t } } ( \boldsymbol x _ { t } ) ) \right] \mathrm { ~ d } \boldsymbol \mathbf { \dot { c } } } \\ { \mathrm { w i t h } _ { \quad { \boldsymbol s q } _ { \varphi , t } } ( \boldsymbol x _ { t } | \boldsymbol y ) = \nabla _ { \boldsymbol x _ { t } } \log q _ { \varphi , t } ( \boldsymbol x _ { t } | \boldsymbol y ) , } \\ { \displaystyle s _ { p _ { t } } ( \boldsymbol x _ { t } ) = \nabla _ { \boldsymbol x _ { t } } \log p _ { t } ( \boldsymbol x _ { t } ) , } \\ { \displaystyle d = | | \cdot | | _ { 2 } ^ { 2 } , } \end{array}\tag{t}
$$

(13)

where $q _ { \varphi } ( \pmb { y } )$ denotes sampling from the variational posterior $( \mathrm { e . g . } , x _ { 0 } = \mu ( \pmb { y } ) ) , s _ { p _ { t } } ( \pmb { x } _ { t } )$ and $s _ { q _ { \varphi , t } } ( \pmb { x } _ { t } )$ are the scores of the prior and posterior distribution respectively.

Optimizing this objective requires differentiating through the score of the variational distribution, which depends on $\varphi .$ To make this tractable, we derive the following gradient equivalence theorem.

Theorem 1 (Gradient Equivalence Theorem). If distribution $q _ { \varphi } ( { \pmb x } | { \pmb y } )$ satisfies mild regularity conditions, for any score function $s _ { p _ { t } } ( \cdot )$ , the following gradient equivalence holds:

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathbf { \Phi } _ { x \in \mathcal { N } _ { \varepsilon } \left( x , \lfloor y \rfloor \right) } } \frac { \partial } { \partial \varphi } \left[ \mathcal { A } \left( s _ { q _ { \sigma _ { \varepsilon } \left( x \right) } } ( \mathbf { x } _ { t } | y ) - s _ { p _ { \sigma _ { \varepsilon } } } ( \mathbf { x } _ { t } ) \right) \right] } \\ & { = \mathbb { E } \Bigg [ \mathcal { A } ^ { \prime } \left( s _ { q _ { \sigma _ { \varepsilon } \left( x \right) } } ( \mathbf { x } _ { t } | y ) - s _ { p _ { \varepsilon } } ( \mathbf { x } _ { t } ) \right) \frac { \partial } { \partial \varphi } s _ { q _ { \sigma _ { \varepsilon } } \left( x \right) } ( y ) } \\ & { \quad + d ^ { \prime } \left( s _ { q _ { \sigma _ { \varepsilon } \left( x \right) } } ( \mathbf { x } _ { t } | y ) - s _ { p _ { \varepsilon } } ( \mathbf { x } _ { t } ) \right) \frac { \partial \mathbf { x } _ { t } } { \partial \varphi } \Bigg ] } \\ & { = \frac { \partial } { \partial \varphi } \mathbb { E } \Bigg [ - \left\{ d ^ { \prime } \left( s _ { q _ { \sigma _ { \varepsilon } \left( y \right) } } , \left( x _ { t } | y \right) - s _ { p _ { \varepsilon } } ( \mathbf { x } _ { t } ) \right) \right\} ^ { T } } \\ & { \quad \cdot \left\{ s _ { q _ { \sigma _ { \varepsilon } \left( x \right) , 1 } } \left( \mathbf { x } _ { t } | y \right) - \nabla _ { x \pm } \log p _ { \varepsilon } ( \mathbf { x } _ { t } | x _ { 0 } ) \right\} } \\ & { \quad + d \left( s _ { q _ { \sigma _ { \varepsilon } \left( x \right) } } , \left( x _ { t } | y \right) - s _ { p _ { \varepsilon } } ( \mathbf { x } _ { t } ) \right) \Bigg ] , } \end{array}\tag{14}
$$

![](images/c43d9e6d4ce6783b1136597608312981a2400cc512fb2a811b47d88508e1aea2.jpg)  
Fig. 2. Overview of PPM. PPM approximates the posterior by optimizing a variational posterior distribution $q _ { \varphi } ,$ , which is parameterized either by a set of particles or a neural network. The optimization minimizes a loss function composed of a data fidelity term and a novel score-based divergence. This divergence is computed between a pre-trained prior score model $\scriptstyle { \pmb { s } } _ { \boldsymbol { \theta } }$ and an auxiliary score network ${ \pmb s } _ { \phi } ,$ which approximates the score of $q _ { \varphi }$ . Computing this divergence requires an auxiliary score network ${ \pmb s } _ { \phi } .$ The parameters of $q _ { \varphi }$ and ${ \pmb s } _ { \phi }$ are optimized alternatively.

## where sg denotes the stop gradient operator.

Proof. The proof is based on the Score-projection identity, which bridges denoising score matching and denoising autoencoders. Let $\pmb { u } ( \cdot , \varphi )$ be a vector-valued function. Using the notations of Theorem 1, under mild conditions, the following identity holds:

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathbf { \Phi } _ { \mathbf { x } _ { t } \sim q _ { \varphi } ( \cdot | \mathbf { y } ) } } \bigg [ \mathbf { u } ( \mathbf { x } _ { t } , \varphi ) ^ { T } \bigg \{ s _ { q _ { \varphi , t } } ( \mathbf { x } _ { t } | \mathbf { y } ) } \\ & { \quad \quad - \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) \bigg \} \bigg ] = 0 . } \end{array}\tag{15}
$$

We start by applying the chain rule for the total derivative with respect to $\varphi .$ The function $d ( \cdot )$ depends on $\varphi$ both directly through the score function $s _ { q _ { \varphi , } }$ and indirectly through the distribution $\mathbf { \boldsymbol { x } } _ { t } \sim q _ { \varphi , t }$ (as $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ depends on $\pmb { x } _ { 0 } \sim q _ { \varphi } ( \cdot | \pmb { y } ) )$ . This gives two terms:

$$
\begin{array} { r l } & { \mathbb { E } _ { \pmb { x } _ { t } \sim q _ { \varphi , t } } \frac { \partial } { \partial \varphi } d ( s _ { q _ { \varphi , t } } ( \pmb { x } _ { t } \vert \pmb { y } ) - s _ { p _ { t } } ( \pmb { x } _ { t } ) ) } \\ & { \quad = \mathbb { E } \bigg [ d ^ { \prime } ( s _ { q _ { \varphi , t } } ( \pmb { x } _ { t } \vert \pmb { y } ) - s _ { p _ { t } } ( \pmb { x } _ { t } ) ) ^ { T } \frac { \partial } { \partial \varphi } s _ { q _ { \varphi , t } } ( \pmb { x } _ { t } \vert \pmb { y } ) } \\ & { \quad \quad + \left. \frac { \partial } { \partial \pmb { x } _ { t } } d ( s _ { q _ { \varphi , t } } ( \pmb { x } _ { t } \vert \pmb { y } ) - s _ { p _ { t } } ( \pmb { x } _ { t } ) ) ^ { T } \frac { \partial \pmb { x } _ { t } } { \partial \varphi } \right] . } \end{array}\tag{16}
$$

To resolve the first term, we differentiate Eq. (15) with respect to $\varphi .$ . Since the expectation is zero for all $\varphi ,$ its derivative is also zero. We apply the total derivative:

$$
\begin{array} { l } { { \displaystyle 0 = \frac { \partial } { \partial \varphi } \mathbb E \left[ { \pmb u } ( { \pmb x } _ { t } , \varphi ) ^ { T } \{ { \boldsymbol s } _ { { \boldsymbol q } _ { \varphi , t } } ( { \pmb x } _ { t } | { \pmb y } ) - \nabla _ { { \pmb x } _ { t } } \log p _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) \} \right] } } \\ { ~ } \\ { { \displaystyle ~ = \mathbb E \bigg [ \frac { \partial } { \partial \varphi } \{ { \pmb u } ( { \pmb x } _ { t } , \varphi ) ^ { T } ( s _ { { \boldsymbol q } _ { s [ \varphi ] , t } } ( { \pmb x } _ { t } | { \pmb y } ) - \nabla _ { { \pmb x } _ { t } } \log p _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) ) \} } } \\ { { \displaystyle ~ + ~ { \pmb u } ( { \pmb x } _ { t } , \varphi ) ^ { T } \frac { \partial } { \partial \varphi } \{ { s } _ { { \boldsymbol q } _ { \varphi , t } } ( { \pmb x } _ { t } | { \pmb y } ) \} \bigg ] . } } \end{array}\tag{17}
$$

Rearranging the terms yields:

$$
\begin{array} { r l } {  { \mathbb { E } [ u ( \boldsymbol { x } _ { t } , \varphi ) ^ { T } \frac { \partial } { \partial \varphi } \{ \boldsymbol { s } _ { \boldsymbol { q } _ { \varphi , t } } ( \boldsymbol { x } _ { t } | \boldsymbol { y } ) \} ] } } \\ & { = - \frac { \partial } { \partial \varphi } \mathbb { E } [ u ( \boldsymbol { x } _ { t } , \varphi ) ^ { T } \{ \boldsymbol { s } _ { \boldsymbol { q } _ { \mathrm { s g } [ \varphi ] , t } } ( \boldsymbol { x } _ { t } | \boldsymbol { y } ) - \nabla _ { \boldsymbol { x } _ { t } } \log p _ { t } ( \boldsymbol { x } _ { t } | \boldsymbol { x } _ { 0 } ) \} ] . } \end{array}\tag{18}
$$

Let ${ \pmb u } ( { \pmb x } _ { t } , \varphi ) = d ^ { \prime } ( s _ { q _ { s \mathrm { g } [ \varphi ] , t } } ( { \pmb x } _ { t } | { \pmb y } ) - s _ { p _ { t } } ( { \pmb x } _ { t } ) )$ . Substituting this specific function u into Eq. (18) allows us to replace the first term in our objective expansion. Furthermore, $\varphi$ does not appear in the differentiation with respect to $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ for the second term:

$$
\begin{array} { r l } & { \mathbb { E } \left[ \displaystyle \frac { \partial } { \partial \pmb { x } _ { t } } d ( s _ { q _ { \varphi , t } } ( \pmb { x } _ { t } | \pmb { y } ) - s _ { p _ { t } } ( \pmb { x } _ { t } ) ) ^ { T } \frac { \partial \pmb { x } _ { t } } { \partial \varphi } \right] } \\ & { = \displaystyle \frac { \partial } { \partial \varphi } \mathbb { E } \left[ d ( s _ { q _ { s \xi [ \varphi ] , t } } ( \pmb { x } _ { t } | \pmb { y } ) - s _ { p _ { t } } ( \pmb { x } _ { t } ) ) \right] . } \end{array}\tag{19}
$$

Combining these results yields exactly the Gradient Equivalence Theorem. □

In Theorem 1, the variational score ${ \pmb { s } } _ { q _ { \varphi , t } } ( { \pmb { x } } _ { t } | { \pmb { y } } )$ is estimated by an auxiliary neural network $\mathbf { \boldsymbol { s } } _ { \phi } ( \mathbf { \boldsymbol { x } } _ { t } , t )$ . This network is trained on the current reconstructions x $\sim q _ { \varphi } ( \cdot | \pmb { y } )$ using a standard denoising score matching objective. We refer to ${ \pmb s } _ { \phi }$ as the auxiliary model, and its training loss is:

$$
\begin{array} { r l } & { \mathcal { L } _ { a u x } ( \phi ) } \\ & { \quad = \displaystyle \int _ { 0 } ^ { T } \lambda ( t ) \mathbb { E } \underset { \substack { \boldsymbol { x } _ { t } | \mathbf { x } _ { 0 } \sim p _ { t } ( \boldsymbol { \cdot } | \boldsymbol { y } ) } } { \boldsymbol { x } _ { t } \sim q _ { \varphi } ( \cdot | \boldsymbol { y } ) } \left\| s _ { \phi } ( \mathbf { x } _ { t } , t ) - \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) \right\| _ { 2 } ^ { 2 } \mathrm { d } t . } \end{array}\tag{20}
$$

We implement ${ \pmb s } _ { \phi }$ as a copy of the pre-trained model $s _ { p } .$ . This preserves the prior information in $s _ { p }$ while adapting to the conditional distribution $q _ { \varphi , t } ( \pmb { x } _ { t } | \pmb { y } )$ during optimization.

## C. Unified Optimization Framework

Based on the gradient equivalence in Theorem 1, we formulate the total objective function for PPM as a combination of a data fidelity term and an unbiased prior regularization term:

Algorithm 1 Principled Posterior Matching (PPM) for Inverse Problems   
Require: Pretrained diffusion network $\begin{array} { r } { { \pmb s } \theta ; } \end{array}$ , Auxiliary network ${ \pmb s } _ { \phi }$ (init via a copy of $s _ { p } )$ , Noising schedule $\left\{ \alpha _ { t } , \sigma _ { t } \right\}$ , Any   
distance function $d ( \cdot )$ , Noise scale h, Data fidelity weight $\lambda ,$ Time steps weight $w ( t )$ , Total time steps $T .$   
Require: Mode: Variational (optimize particles $\pmb { \mu }$ for fixed $\mathbf {  { y } }$ or Amortized (optimize reconstruction network $g _ { \varphi }$ for   
observations $y ) _ { \ l }$   
1: Initialize $\varphi \colon$   
2: if Mode is Variational then   
3: $\varphi \gets \{ \mu _ { k } \} _ { k = 1 } ^ { K }$ initialized with measurement $\mathbf { \nabla } _ { \mathbf { \mu } _ { y . } }$   
4: else if Mode is Amortized then   
5: $\varphi \gets$ weights of reconstruction network $g _ { \varphi }$ initialized from $\begin{array} { r } { { \pmb S } \theta \cdot \mathbf { \sigma } } \end{array}$   
6: end if   
7: while not converged do   
8: 1. Sampling:   
9: if Variational then ▷ Variational inference   
10: Use target measurement y. Sample $\mathbf { \boldsymbol { x } } _ { 0 } = \boldsymbol { \mu _ { k } }$ from current particles.   
11: else ▷ Amortized inference   
12: Sample batch $\mathbf { \boldsymbol { y } } \sim \mathcal { V } .$ . Add noise $\sigma \sim \mathcal { N } ( \mathbf { 0 } , h \mathbf { I } )$ to $\begin{array} { r } { \mathrm { y . } \ y ^ { \prime } = \pmb { y } + \pmb { \sigma } . } \end{array}$   
13: Generate $\pmb { x } _ { 0 } = g _ { \varphi } ( \pmb { y } ^ { \prime } )$   
14: end if   
15: Sample $t \sim \mathcal { U } ( 0 , T ) , \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . Compute ${ \pmb x } _ { t } = \alpha _ { t } { \pmb x } _ { 0 } + \sigma _ { t } { \pmb \epsilon }$   
16: 2. Auxiliary Score Update (Learning $\nabla \log q _ { \varphi , t } ( \pmb { x } _ { t } | \pmb { y } )$ with auxiliary network):   
17: $\mathcal { L } _ { \mathrm { a u x } }  w ( t ) \| s _ { \phi } ( { \pmb x } _ { t } , t ) - \nabla _ { { \pmb x } _ { t } } \log p ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) \| _ { 2 } ^ { 2 } .$   
18: ϕ ← OptimizerStep(ϕ, $\mathcal { L } _ { \mathrm { a u x } } \big )$   
19: 3. Reconstruction Optimization (Update φ):   
20: $\mathcal { L } _ { \mathrm { p r i o r } }  - d ^ { \prime } ( s _ { \phi } ( \mathbf { x } _ { t } , \bar { t } ) - s _ { \theta } ( \mathbf { x } _ { t } , t ) ) ^ { T } \{ s _ { \phi } ( \mathbf { \dot { x _ { t } } } , t ) - \epsilon \} + d ( s _ { \phi } ( \mathbf { x } _ { t } , t ) - s _ { \theta } ( \mathbf { x } _ { t } , t ) )$ via Theorem 1.   
21: $\mathcal { L } _ { \mathrm { d a t a } }  \| \pmb { y } - \mathcal { A } ( \pmb { x } _ { 0 } ) \| ^ { 2 } .$   
22: $\varphi \gets \mathrm { O p t i m i z e r S t e p } ( \varphi , \mathcal { L } _ { \mathrm { p r i o r } } + \lambda \mathcal { L } _ { \mathrm { d a t a } } ) .$   
23: end while   
24: return Optimized results $\varphi$ (Particles $\pmb { \mu }$ or reconstruction network $g _ { \varphi } )$

```latex
L<sub>PPM</sub>(φ) = L<sub>prior</sub>(φ) + λL<sub>data</sub>(φ),
L<sub>data</sub>(φ) = <sup>E</sup><sub>x0∼qφ(·|y)</sub> -∥y − A(x<sub>0</sub>)∥<sup>2</sup> ,
$\begin{array} { r l } { \mathcal { L } _ { \mathrm { p r i o r } } ( \varphi ) = \mathbb { E } \ } & { { } _ { t \sim \mathcal { U } ( 0 , T ) } \quad \Big [ - d ^ { \prime } \big ( s _ { \phi } ( x _ { t } , t ) - s _ { p } ( x _ { t } , t ) \big ) ^ { \top } } \\ { x _ { 0 } \sim q _ { \varphi } ( \cdot | y ) } & { { } } \end{array}$
x<sub>t</sub>|x<sub>0</sub>∼p(x<sub>t</sub>|x<sub>0</sub>)
$\begin{array} { r } { \big ( { \pmb s } _ { \phi } ( { \pmb x } _ { t } , t ) - \nabla _ { { \pmb x } _ { t } } \log p _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) \big ) + d \big ( { \pmb s } _ { \phi } ( { \pmb x } _ { t } , t ) - { \pmb s } _ { p } ( { \pmb x } _ { t } , t ) \big ) \Big ] , } \end{array}$
(21)
```

where ${ \mathcal { L } } _ { \mathrm { d a t a } }$ enforces consistency with the measurement $^ { y , }$ and ${ \mathcal { L } } _ { \mathrm { p r i o r } }$ aligns the reconstruction with the diffusion prior using the unbiased gradient estimator derived in Theorem 1. While our standard implementation utilizes the $L _ { 2 }$ norm (Fisher divergence) where $d ( \pmb { u } , \pmb { v } ) = \lVert \pmb { u } - \pmb { v } \rVert _ { 2 } ^ { 2 } ,$ , our framework is theoretically general: it naturally extends to other divergences by selecting different convex distance metrics $d ,$ offering scalability to various score matching variants.

PPM provides a unified training logic for both VI and AI inference paradigms. The unified training procedure is summarized in Algorithm 1. Despite the difference in parameterization, both paradigms operate via an identical alternating two-stage process:

• Stage 1: Auxiliary Score Learning (Update ϕ). We update the auxiliary network ${ \pmb s } _ { \phi }$ to minimize $\mathcal { L } _ { \mathrm { a u x } }$ (Eq. 20). This step effectively learns the score ∇ log $q _ { \varphi , t }$ of the current variational distribution (defined either by particles or a reconstruction network).

• Stage 2: Reconstruction Optimization (Update $\varphi ) _ { \cdot }$ . We update the variational parameters $\varphi$ to minimize L<sub>PPM</sub> (Eq. 21). This step utilizes the gradient provided by the now-fixed ${ \pmb s } _ { \phi }$ to drive the posterior estimate towards the true prior and measurement.

Variational Inference (Particle-based). In the VI setting, we optimize for a specific single observation y. The variational parameters are defined as a set of image particles $\varphi = \{ \mu _ { k } \} _ { k = 1 } ^ { K }$ , initialized as $\mathbf { \mu } _ { \mu _ { k } } = \mathbf { \mu } _ { y }$ (or a rough inverse). The distribution $q _ { \varphi } ( { \pmb x } | { \pmb y } )$ is represented empirically by these particles. The optimization refines $\pmb { \mu } _ { k }$ to capture the complex, multi-modal posterior landscape specific to $\mathbf { \pmb { y } } .$

Amortized Inference (Neural network-based). In the AI setting, we learn a global mapping for any observation $y \sim p ( y )$ . The variational parameter $\varphi$ denotes the weights of a neural network $g _ { \varphi }$ , such that $\pmb { x } = g _ { \varphi } ( \pmb { y } )$ . To accelerate convergence, we implement $g _ { \varphi }$ as a copy of the pre-trained diffusion U-Net (initialized with $\theta ) ,$ , enabling efficient, singlestep reconstruction. This amortizes the optimization cost, allowing rapid inference at test time.

Beyond the standard formulation presented above, we highlight the inherent extensibility of the PPM framework. While this work primarily employs the squared $L _ { 2 }$ norm as the metric function $d ( \cdot )$ —which corresponds to the standard Fisher divergence and minimizes the Kullback-Leibler divergence—our theoretical derivations (Theorem 1) are not restricted to this choice. The metric $d ( \cdot )$ can be substituted with a broader class of convex distance functions. This flexibility allows PPM to be naturally generalized to measure and minimize a wider spectrum of divergences, positioning it as a versatile foundation for score-based variational inference.

## IV. THEORETICAL ANALYSIS

In this section, we provide a rigorous theoretical comparison between PPM and three leading categories of baselines of diffusion-based inverse problem solvers. We demonstrate that while these methods offer practical utility, they all rely on biased approximations of the true posterior objective.

## A. Bias in Optimization-based VI (RED-Diff and RLSD)

Optimization-based methods like RED-Diff Mardani et al. [2023] and RLSD Zilberstein et al. [2024] formulate the inverse problem as a variational optimization but share a fundamental theoretical flaw in their handling of entropy.

1) Missing Exact Entropy (RED-Diff): RED-Diff implicitly models the posterior as a Dirac delta distribution. This effectively removes the entropy term $H ( q )$ from the KL divergence, collapsing the problem to Maximum A Posteriori (MAP) estimation. Consider the standard decomposition:

$$
D _ { K L } ( q ( \pmb { x } ) | | p ( \pmb { x } | \pmb { y } ) ) = - \mathbb { E } _ { q } [ \log p ( \pmb { x } | \pmb { y } ) ] - H ( q ) .\tag{22}
$$

Under the Dirac assumption $q ( { \pmb x } ) = \delta ( { \pmb x } - { \pmb \mu } )$ , the entropy vanishes $( \nabla _ { \mu } H ~ = ~ 0 )$ and the energy term collapses to $\log p ( \pmb { \mu } | \pmb { y } )$ . Consequently, the minimization problem becomes mathematically equivalent to MAP estimation:

$$
\operatorname* { m i n } _ { \mu } D _ { K L } ( q | | p ) \Longleftrightarrow \operatorname* { m a x } _ { \mu } \log p ( \pmb { \mu } | \pmb { y } ) .\tag{23}
$$

This confirms that without the entropy term, the optimization inherently seeks the single most probable mode rather than the full distribution, explaining the severe mode-seeking behavior observed in Figure 1.

2) Surrogate Entropy via Repulsion (RLSD): While RLSD mitigates mode collapse using particles, it introduces ad-hoc repulsive regularization instead of optimizing the true entropy. This repulsion acts as a heuristic proxy and does not correspond to the true score of the variational distribution, resulting in artificial uncertainty dependent on hyperparameters.

## B. Bias in MCMC Sampling (DPS)

Diffusion Posterior Sampling (DPS) Chung et al. [2022] approximates samples by modifying the reverse diffusion score with a likelihood guidance term. Since the likelihood $p ( \pmb { y } | \pmb { x } _ { t } )$ is intractable, DPS approximates it using a clean data estimate $\hat { \pmb { x } } _ { 0 } ( { \pmb { x } } _ { t } ) ~ = ~ \mathbb { E } [ { \pmb { x } } _ { 0 } | { \pmb { x } } _ { t } ]$ This violates Jensen’s inequality by treating the expectation of the likelihood as the likelihood of the expectation. This introduces systematic score estimation error, particularly in early diffusion stages, which accumulates over the trajectory.

## C. Bias in Amortized Inference (DAVI)

DAVI Lee et al. [2024] employs the Integral KL (IKL) divergence Luo et al. [2023] to train amortized generators. Here, we formally prove that replacing the standard prior KL divergence with the IKL objective fundamentally alters the optimization target.

1) Problem Formulation: Standard VI minimizes $\mathcal { I } _ { V I } =$ $\mathbb { E } _ { q _ { \varphi } } [ - \log p ( \pmb { y } | \pmb { x } ) ] \ + \ D _ { K L } ( q _ { \varphi } | | p )$ . IKL-based methods replace the prior term with an integrated objective $\mathcal { I } _ { M o d } ~ =$ $\begin{array} { r } { \mathbb { E } [ - \log p ( \pmb { y } | \pmb { x } ) ] + \int \omega ( t ) D _ { K L } ( q _ { t } | | p _ { t } ) d t } \end{array}$

2) KL Contraction and Implicit Prior: Assuming a Variance Preserving (VP) schedule and a Gaussian Mean Shift assumption $( p ( \pmb { x } ) = \mathcal { N } ( \mathbf { 0 } , \pmb { I } ) , q ( \pmb { x } | \pmb { y } ) = \mathcal { N } ( \pmb { \Delta } , \pmb { I } ) )$ , the KL divergence scales quadratically: $D _ { K L } ( q _ { t } | | p _ { t } ) ~ \approx ~ \alpha _ { t } ^ { 2 }$ $D _ { K L } ( q _ { 0 } | | p _ { 0 } )$

Substituting this scaling law into the IKL integral, we rewrite the modified objective as:

$$
\mathcal { I } _ { I K L } \approx \left( \int _ { 0 } ^ { T } \omega ( t ) \alpha _ { t } ^ { 2 } d t \right) \cdot D _ { K L } ( q _ { \varphi } ( \pmb { x } | \pmb { y } ) | | p ( \pmb { x } ) ) = \beta D _ { K L } ( q _ { \varphi } | | p ) .\tag{24}
$$

This effectively scales the prior weight by β. Expanding the terms reveals the implicit posterior target $p ^ { \prime } ( \pmb { x } | \pmb { y } )$

$$
\mathcal { I } _ { M o d } = \beta \mathbb { E } _ { q _ { \varphi } } \left[ \log q _ { \varphi } ( \pmb { x } | \pmb { y } ) - \left( \frac { 1 } { \beta } \log p ( \pmb { y } | \pmb { x } ) + \log p ( \pmb { x } ) \right) \right] .\tag{25}
$$

Exponentiating this result implies optimization against a Distorted Prior $\bar { p ^ { \prime } } ( { \pmb x } ) \ \propto \ p ( { \pmb x } ) ^ { \bar { \beta } }$ . Since typically $\beta ~ < ~ 1$ , the effective prior is a flattened, high-temperature version of the true prior, proving that IKL leads to biased posterior estimation.

## V. EXPERIMENT

In this section, we compare our proposed method, PPM, with state-of-the-art (SoTA) diffusion model-based methods for solving inverse problems, particularly those employing variational inference. Our experiments aim to demonstrate PPM’s capability to generate diverse reconstructions while maintaining fidelity, thereby recovering the full posterior and surpassing baselines that typically yield homogeneous results.

## A. Toy Examples of 2D Posterior Estimation

We first validated PPM on two simple 2D posteriorestimation tasks. In each task, the hidden variable $\bar { x } \in \mathbb { R } ^ { 2 \times 1 }$ follows a Gaussian-mixture prior, and measurements obey the linear model

$$
y = F x + n ,\tag{26}
$$

where F is a linear projection matrix and $n \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ is Gaussian noise. We compare PPM against two related baselines, RED-Diff Mardani et al. [2023] and RLSD Zilberstein et al. [2024], using the same pre-trained pixel-space diffusion prior. To ensure fair compaison, we adapt RLSD by replacing its latent diffusion backbone with our pixel-space prior and incorporating its repulsive term into a pixel-space SDS loss, so that all methods operate under identical conditions.

Figure 1 highlights the significant differences in each method’s ability to capture multimodal posteriors. RED-Diff exhibits severe mode collapse: its samples converge to accurate point estimates but lack diversity. RLSD produces more varied samples, yet its empirical repulsive term leads to either measurement inconsistency (top example) or poor prior adherence (bottom example). In contrast, PPM faithfully recovers every posterior mode, yielding both accurate and diverse samples that closely match the ground-truth bimodal distribution.

![](images/d31660fd6233dbdfd6cd0c864b31c88c3bc386146bc8c10c80098542d9ee5f40.jpg)  
Fig. 3. Comparison of PPM and diffusion-based VI baselines (DPS, RED-Diff and RLSD) on box inpainting with FFHQ. From top to bottom: the masked observation and the ground truth, followed by four random posterior samples from DPS, RED-Diff, RLSD, and PPM. Although all methods yield plausible completions, PPM produces markedly more diverse and higher-fidelity samples within the inpainted region (red box). By contrast, the baselines generate nearly identical outputs, indicating a failure to capture posterior uncertainty.

## B. Computational Photography

We then evaluated PPM on various computational photography tasks, including box inpainting, motion deblurring, and super-resolution, using natural images.

a) Datasets and Pretrained Models : We primarily evaluate on two natural-image datasets with distinct characteristics—FFHQ (256 × 256) Karras et al. [2019] and ImageNet (256 × 256) Deng et al. [2009]—using 32 randomly selected images from each validation set. For FFHQ, we use the diffusion prior released by DPS Chung et al. [2022], and for ImageNet we employ the model from Dhariwal and Nichol [2021]. Both are used off-the-shelf, without task-specific finetuning. RLSD Zilberstein et al. [2024] requires a latent diffusion backbone, so we use Stable Diffusion v2.1 Rombach et al. [2022] following its original setup.

b) Baselines: We compare PPM against state-of-the-art computational imaging methods that leverage diffusion priors, spanning both variational-inference and MCMC-sampling paradigms. Our VI baselines include RED-Diff Mardani et al. [2023] and RLSD Zilberstein et al. [2024], while our MCMC baselines are ΠGDM Song et al. [2022] and DPS Chung et al. [2022]. All methods use the same pixel-space diffusion prior—except RLSD, which retains its Stable Diffusion prior—and are evaluated with hyperparameters set to their original defaults to ensure a fair comparison.

c) Evaluation Metrics: We assess reconstruction fidelity using PSNR and SSIM, and quantify sampling diversity using the pairwise cosine similarity among N reconstruction samples from the same observation. The final diversity is the average of all O observations:

$$
\mathrm { ~ 1 ~ - ~ } \frac { 1 } { | O | } \sum _ { o } \frac { 2 } { N ( N - 1 ) } \sum _ { i < j } \frac { \langle x _ { o i } , x _ { o j } \rangle } { \| x _ { o i } \| _ { 2 } \| x _ { o j } \| _ { 2 } } .
$$

For quantitative and efficient comparison, we estimate over 64 observations in FFHQ validation dataset and reconstruct 8 samples from each observation.

d) Inpainting: Box-inpainting results are assessed on FFHQ validation images. Qualitative comparisons with 80×80 masked boxes are visualized in Fig. 3, while quantitative metrics evaluated on larger 128×128 center masks are reported in Table II. We compare PPM against two VI baselines, RED-Diff Mardani et al. [2023] and RLSD Zilberstein et al. [2024]. PPM significantly surpasses both baselines in diversity while simultaneously delivering superior reconstruction quality (PSNR and SSIM). In challenging cases where critical features like hair (Fig. 3 left) or facial contours (Fig. 3 right) are masked, RED-Diff collapses to nearly identical outputs and RLSD struggles to trade off data fidelity against prior adherence. By contrast, PPM produces varied yet plausible reconstructions (e.g., different hairstyles and lip shapes) thanks to its principled score-based divergence and tailored optimization strategy. As Table II shows, PPM attains the top scores across all metrics on the validation set—demonstrating superior quality and diversity without extensive tuning—and also outperforms MCMC-based methods such as DPS.

![](images/c56537a209392c85fff6ec1a36fdd17a4671f9ce3e16fff0d7df039c213a2091.jpg)  
Fig. 4. Comparison of PPM, DPS, and diffusion-based VI baselines (RED-Diff and RLSD) on motion deblurring, super-resolution and gaussian deblurring tasks with ImageNet. For each VI method, we show one sample reconstruction (left) alongside its standard-deviation uncertainty map (right). Compared to DPS, PPM achieves higher fidelity—accurately rendering details like the elephant’s tusks, balloon logos, and human faces. Compared to VI methods, PPM produces better calibrated uncertainty: RED-Diff’s standard-deviation maps reveal mode collapse, and RLSD’s contain pronounced artifacts.

TABLE II  
QUANTITATIVE COMPARISON OF PPM AND BASELINE METHODS ACROSS COMPUTATIONAL PHOTOGRAPHY TASKS, INCLUDING SUPER-RESOLUTION, MOTION DEBLURRING, AND BOX INPAINTING, ON FFHQ KARRAS ET AL. [2019] AND IMAGENET DENG ET AL. [2009] (256×256 RESOLUTION). ALL METHODS USE THE SAME PRETRAINED UNCONDITIONAL DIFFUSION MODEL, EXCEPT RLSD ZILBERSTEIN ET AL. [2024], WHICH EMPLOYS STABLE DIFFUSION ROMBACH ET AL. [2022]. BEST RESULTS ARE SHOWN IN BOLD, AND SECOND-BEST ARE UNDERLINED.
<table><tr><td rowspan="2">Method</td><td colspan="3">Super Resolution</td><td colspan="3">Motion Deblurring</td><td colspan="3">Box Inpainting</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>Diversity ↑</td><td>PSNR ↑</td><td>SSIM ↑</td><td>Diversity ↑</td><td>PSNR ↑</td><td>SSIM↑</td><td>Diversity ↑</td></tr><tr><td>Sampling</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DDRM Chung et al. [2022]</td><td>22.62</td><td>0.62</td><td>0.001</td><td></td><td></td><td></td><td>22.11</td><td>0.78</td><td>0.004</td></tr><tr><td>DPS Chung et al. [2022]</td><td>21.02</td><td>0.57</td><td>0.010</td><td>20.34</td><td>0.55</td><td>0.006</td><td>23.43</td><td>0.80</td><td>0.009</td></tr><tr><td>IIGDM Song et al. [2022]</td><td>23.92</td><td>0.67</td><td>0.011</td><td>25.82</td><td>0.75</td><td>0.006</td><td>23.25</td><td>0.86</td><td>0.010</td></tr><tr><td>Particle variational inference</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RED-Diff Mardani et al. [2023]</td><td>26.54</td><td>0.76</td><td>0.001</td><td>29.02</td><td>0.84</td><td>0.001</td><td>24.69</td><td>0.87</td><td>0.002</td></tr><tr><td>RLSD Zilberstein et al. [2024]</td><td>27.28</td><td>0.79</td><td>0.003</td><td>25.64</td><td>0.82</td><td>0.002</td><td>28.27</td><td>0.93</td><td>0.006</td></tr><tr><td>Ours(VI)</td><td>25.63</td><td>0.72</td><td>0.013</td><td>28.18</td><td>0.83</td><td>0.009</td><td>28.73</td><td>0.97</td><td>0.016</td></tr><tr><td>Amortized inference</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DAVI Lee et al. [2024]</td><td>24.49</td><td>0.70</td><td>0.006</td><td>27.69</td><td>0.82</td><td>0.003</td><td>24.58</td><td>0.83</td><td>0.005</td></tr><tr><td>Ours(AI)</td><td>24.85</td><td>0.73</td><td>0.008</td><td>29.17</td><td>0.85</td><td>0.004</td><td>25.26</td><td>0.88</td><td>0.008</td></tr></table>

e) Motion Deblurring, Super Resolution, and Gaussian Denoising: For motion deblurring, we follow Chung et al. [2022], Zilberstein et al. [2024] by convolving each image with a randomly sampled 61 × 61 motion kernel (variance = 0.3<sup>2</sup>). For super-resolution, we downsample images by a factor of 8. Additionally, for Gaussian denoising, we corrupt the images with additive white Gaussian noise with a standard deviation of σ = 0.2. We evaluate PPM on these tasks using FFHQ and ImageNet validation sets, comparing against DPS

Chung et al. [2022], ΠGDM Song et al. [2022], RED-Diff Mardani et al. [2023], RLSD Zilberstein et al. [2024], and the amortized baseline DAVI Lee et al. [2024].

Figure 4 shows that PPM delivers sharper, more observation-consistent reconstructions than DPS, evident in fine details such as the elephant’s tusks and the balloon’s logo—thanks to PPM’s exact likelihood term versus DPS’s approximation. Compared to VI methods (RED-Diff and RLSD), PPM also excels at uncertainty quantification: its standarddeviation maps accurately reflect positional uncertainty (e.g., the elephant’s tusks, the boy’s facial edge), whereas RED-Diff collapses modes and RLSD introduces p persistent speckle artifacts due to latent-space optimization.

Figure 5 presents a qualitative comparison of the amortized inference results on the FFHQ dataset. We contrast our PPM framework against the KL-divergence-based baseline, DAVI. The pixel-wise uncertainty maps demonstrate that our method captures meaningful posterior diversity, particularly in ambiguous regions (e.g., edges and textures), whereas the KLbased approach exhibits signs of posterior collapse with largely

Observation  
DAVI Lee et al. [2024]  
Ours  
Ground Truth  
![](images/6c89223e137a4e30a48bb5e3562bb253c2576348fd4cf670dd25e41b352c5d2e.jpg)  
Fig. 5. Qualitative comparison of amortized inference on the FFHQ validation set. We compare our PPM framework against the KL-divergence-based baseline, DAVI Lee et al. [2024]. The rows correspond to three inverse problems: Motion Deblurring (top), 8× Super-Resolution (middle), and Gaussian Denoising (bottom). For each task, we visualize the degraded observation, the reconstruction and pixel-wise uncertainty map from DAVI, the reconstruction and uncertainty map from our method, and the ground truth. Our method yields significantly sharper structural details (e.g., facial features and hair textures) and provides more informative uncertainty estimates that capture the rich diversity of local details, whereas the IKL-based approach tends to over-smooth the results.

suppressed uncertainty.

Table II confirms that PPM achieves the best overall balance of PSNR, SSIM, and diversity. While RLSD slightly outperforms in PSNR/SSIM for super-resolution—due to its higher-resolution Stable Diffusion prior (512 × 512)—its artifacts undermine visual quality. MCMC sampling methods like DPS match PPM’s diversity but fall short in fidelity. In summary, PPM consistently outperforms both variational and sampling-based baselines across computational photography tasks, delivering superior reconstruction quality and reliable uncertainty estimates.

## C. Super-resolution Fluorescent Microscopic Imaging

Beyond standard computational photography tasks, we also applied our method to real-world scientific imaging challenges in biomedicine and astronomy. We evaluated PPM on superresolution fluorescent microscopy, a critical tool for visualizing subcellular structures. Here, the observation y is a wide-field microscope image (approximately 200 nm resolution), whose measurement model is

$$
y = \mathrm { P S F } \circledast x + n ,\tag{27}
$$

where x is the underlying high-resolution fluorescence signal, PSF is the microscope point-spread function, and n is additive Gaussian noise. Accurately recovering x, along with precise uncertainty quantification, is essential for resolving the finegrained dynamics of subcellular structures, such as organelles, and their interactions. In our experiment, we primarily benchmark PPM with RED-Diff. Both methods use the same diffusion prior pretrained on the BioSR dataset Aali et al. [2023], which comprises over 10,000 256 × 256 super-resolution images of diverse subcellular structures—microtubules, endoplasmic reticulum (ER), clathrin-coated pits (CCPs), and F-actin—captured with a structured illumination microscope (approximately 100 nm resolution).

Figure 6 demonstrates that PPM delivers higher-fidelity reconstructions than RED-Diff, faithfully rendering thin filaments in microtubules and the mesh-like ER. Quantitatively, PPM also achieves superior PSNR and SSIM scores. Crucially, PPM’s uncertainty maps align with imaging physics: confidence peaks at structure centers and decreases toward blurred edges, revealing hollow structures. This behavior reflects the influence of microscope’s PSF, which preserves feature presence while smearing precise boundaries. PPM accurately captures this boundary uncertainty, whereas RED-Diff’s estimates fail to indicate edge ambiguity. These results underscore PPM’s reliability for nanometer-scale biomedical imaging, where uncertainty quantification is indispensable.

## D. Radio Interferometric Black Hole Imaging

We applied PPM to reconstruct and quantify uncertainty in black hole images from very long baseline interferometry (VLBI) measurements. Using a general-relativistic magnetohydrodynamics (GRMHD) simulated Sagittarius A<sup>⋆</sup> black hole image, we emulate a synthetic observation of the Event Horizon Telescope (EHT) array, which comprises nine telescopes worldwide to form an Earth-sized interferometer. Ignoring atmospheric turbulence, the measurement model that maps the true image x to the observed visibilities $y$ is

![](images/cf9332c7397d6383d9a84f68d1e1f7ff082b6ee1c89f344a397cfa27dcff2066.jpg)  
Fig. 6. Fluorescent super-resolution microscopic imaging results. We compare our method with RED-Diff on microscopic images of Microtubules and ER samples. For each method, the reconstruction (with PSNR/SSIM scores) and its corresponding uncertainty map are reported. Our uncertainty maps accuratel characterize the physical blur caused by the Point Spread Function in biological imaging: the uncertainty is lower at the center of the reconstructed structures and higher at the edges, effectively capturing the transition between the confirmed structures and the background.

$$
y = M { \mathcal { F } } \{ x \} + n ,\tag{28}
$$

where $\mathcal { F }$ is the Fourier transform, M selects the measured frequency components, and $n$ is additive Gaussian noise. Because the EHT sampling is extremely sparse in the Fourier domain (Fig. 7(a)), this defines a highly ill-posed inverse problem: enforcing only data consistency yields the classical dirty image, riddled with sidelobe artifacts (top right of Fig. 7).

Robust uncertainty quantification is therefore critical before making scientific inferences. Our PPM reconstruction follows the InverseBench Zheng et al. [2025] protocol, using a diffusion prior trained on approximately 50,000 synthetic black hole images. Figure 7 (b) shows the ground-truth GRMHD image, the target blurred to the EHT’s resolution, 16 independent PPM posterior samples, and the resulting mean reconstruction alongside its standard-deviation map. We further compare PPM with baselines in Fig. 7(c). While RED-Diff suffers from severe mode collapse (indicated by the suppressed uncertainty map) and DPS yields blurry reconstructions, only PPM faithfully captures the key morphology—ring diameter, azimuthal position of the bright crescent, and the black hole’s swirling signature, demonstrating PPM’s ability to deliver accurate reconstructions with reliable uncertainty estimates in challenging VLBI black hole imaging scenarios.

## VI. CONCLUSION

In this paper, we presented Principled Posterior Matching (PPM), a principled framework that addresses the fundamental limitations of existing variational diffusion-based inverse problem solvers. By identifying that the mode collapse in prior works stems from biased approximations of the KL divergence, we proposed a rigorous alternative based on the integration of Fisher divergence. This enables an unbiased gradient estimator, allowing for the exact minimization of the variational objective without structural collapse. PPM unifies variational and amortized inference, enabling both faithful posterior recovery and efficient, unsupervised generation. Validated across computational and scientific imaging tasks—including microscopy and black-hole imaging—PPM consistently outperforms baselines. Its superior fidelity and reliable uncertainty quantification establish it as a robust foundation for trustworthy imaging.

![](images/1072134e3749e2257b90012ac4a78e6cce4e6f5fa9280ab7faff7eb3905505d1.jpg)  
Fig. 7. Black hole interferometric imaging from synthetic EHT observations. This highly ill-posed inverse problem recovers an image from the sparse Fourier samples of a VLBI array (top left). (a) shows the EHT’s $( u , v )$ coverage and the “dirty” image reconstructed solely from observations, without any image priors. (b) presents the ground-truth GRMHD image, the target blurred to EHT resolution, 16 independent PPM posterior samples, and the resulting mean reconstruction with its standard-deviation map. PPM accurately captures critical features, the ring structure and bright crescent, while providing reliable uncertainty estimates. (c) Comparison with baselines. We report the mean reconstruction and pixel-wise standard deviation for baselines. Consistent with our theoretical analysis, RED-Diff exhibits severe mode collapse, characterized by a suppressed standard deviation map. While DPS captures uncertainty, its reconstruction lacks sharpness. In contrast, PPM achieves superior fidelity with a physically meaningful uncertainty distribution that accurately captures the structural variance of the black hole shadow.

## REFERENCES

Asad Aali, Marius Arvinte, Sidharth Kumar, and Jonathan I Tamir. Solving inverse problems with score-based generative priors learned from noisy data. arXiv preprint arXiv:2305.01166, 2023.

Weimin Bai, Yubo Li, Wenzheng Chen, Weijian Luo, and He Sun. Dive3d: Diverse distillation-based text-to-3d generation via score implicit matching. arXiv preprint arXiv:2506.13594, 2025a.

Weimin Bai, Yubo Li, Weijian Luo, Wenzheng Chen, and He Sun. Vision-language models as differentiable semantic and spatial rewards for text-to-3d generation. arXiv preprint arXiv:2509.15772, 2025b.

David M Blei, Alp Kucukelbir, and Jon D McAuliffe. Variational inference: A review for statisticians. Journal of the American statistical Association, 112(518):859–877, 2017.

Charles A Bouman and Gregery T Buzzard. Generative plug and play: Posterior sampling for inverse problems. arXiv preprint arXiv:2306.07233, 2023.

Steve Brooks, Andrew Gelman, Galin Jones, and Xiao-Li Meng. Handbook of markov chain monte carlo. CRC press, 2011.

Hanyu Cai, Binqi Shen, Lier Jin, Lan Hu, and Xiaojing Fan. Does tone change the answer? evaluating prompt politeness effects on modern llms: Gpt, gemini, llama. arXiv preprint arXiv:2512.12812, 2025. doi: 10.48550/arXiv.2512.12812. URL https://arxiv.org/abs/2512.12812.

Emmanuel Candes and Justin Romberg. Sparsity and incoherence in compressive sampling. Inverse problems, 23(3): 969, 2007.

Gabriel Cardoso, Yazid Janati El Idrissi, Sylvain Le Corff, and Eric Moulines. Monte carlo guided diffusion for bayesian linear inverse problems. arXiv preprint arXiv:2308.07983, 2023.

Andrew Chael, Katie Bouman, Michael Johnson, Maciek Wielgus, Lindy Blackburn, Chi-Kwan Chan, Joseph Rachid Farah, Daniel Palumbo, and Dominic Pesce. eht-imaging: v1. 1.0: Imaging interferometric data with regularized maximum likelihood. Zenodo, 2019.

Shoufa Chen, Peize Sun, Yibing Song, and Ping Luo. Diffusiondet: Diffusion model for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 19830–19843, 2023.

Yongxin Chen, Sinho Chewi, Adil Salim, and Andre Wibisono. Improved analysis for a proximal algorithm for sampling. In Conference on Learning Theory, pages 2984–3014. PMLR, 2022.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song.

Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.

Wonshik Choi, Christopher Fang-Yen, Kamran Badizadegan, Seungeun Oh, Niyom Lue, Ramachandra R Dasari, and Michael S Feld. Tomographic phase microscopy. Nature methods, 4(9):717–719, 2007.

Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022.

Florentin Coeurdoux, Nicolas Dobigeon, and Pierre Chainais. Plug-and-play split gibbs sampler: embedding deep generative priors in bayesian inference. IEEE Transactions on Image Processing, 33:3496–3507, 2024.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

Wei Deng, Weijian Luo, Yixin Tan, Marin Bilos, Yu Chen,ˇ Yuriy Nevmyvaka, and Ricky TQ Chen. Variational schr\” odinger diffusion models. arXiv preprint arXiv:2405.04795, 2024.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. arXiv preprint arXiv:1605.08803, 2016.

Zehao Dou and Yang Song. Diffusion posterior sampling for linear inverse problem solving: A filtering perspective. In The Twelfth International Conference on Learning Representations, 2024.

Berthy T Feng and Katherine L Bouman. Efficient bayesian computational imaging with a surrogate score-based prior. arXiv preprint arXiv:2309.01949, 2023.

Berthy T Feng, Jamie Smith, Michael Rubinstein, Huiwen Chang, Katherine L Bouman, and William T Freeman. Score-based diffusion models as principled priors for inverse imaging. arXiv preprint arXiv:2304.11751, 2023.

Alexandros Graikos, Nikolay Malkin, Nebojsa Jojic, and Dimitris Samaras. Diffusion models as plug-and-play priors. Advances in Neural Information Processing Systems, 35: 14715–14728, 2022.

Linchao He, Wenchao Du, Peixi Liao, Fenglei Fan, Hu Chen, Hongyu Yang, and Yi Zhang. Solving zero-shot sparseview ct reconstruction with variational score solver. IEEE Transactions on Medical Imaging, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Lan Hu, Yuting Xin, Binqi Shen, Hanyu Cai, and Lier Jin. Codes: A context-efficient framework for enhancing small language models via domain-specific adaptation

and model ensembling. Preprints, March 2026. doi: 10.20944/preprints202603.1152.v1. URL https://doi.org/10. 20944/preprints202603.1152.v1.

Marco A Iglesias, Kody JH Law, and Andrew M Stuart. Ensemble kalman methods for inverse problems. Inverse Problems, 29(4):045001, 2013.

Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. arXiv preprint arXiv:2205.09991, 2022.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022.

Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. Advances in neural information processing systems, 31, 2018.

Sojin Lee, Dogyun Park, Inho Kong, and Hyunwoo J Kim. Diffusion prior-based amortized variational inference for noisy inverse problems. In European Conference on Computer Vision, pages 288–304. Springer, 2024.

Yin Tat Lee, Ruoqi Shen, and Kevin Tian. Structured logconcave sampling with a restricted gaussian oracle. In Conference on Learning Theory, pages 2993–3050. PMLR, 2021.

Tiancheng Li, Weijian Luo, Zhiyang Chen, Liyuan Ma, and Guo-Jun Qi. Self-guidance: Boosting flow and diffusion generation on their own. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Xiang Li, Soo Min Kwon, Ismail R Alkhouri, Saiprasad Ravishankar, and Qing Qu. Decoupled data consistency with diffusion purification for image restoration. arXiv preprint arXiv:2403.06054, 2024.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. arXiv preprint arXiv:2206.00927, 2022.

Weijian Luo. A comprehensive survey on knowledge distillation of diffusion models. arXiv preprint arXiv:2304.04262, 2023.

Weijian Luo. Diff-instruct++: Training one-step text-to-image generator model to align with human preferences. arXiv preprint arXiv:2410.18881, 2024.

Weijian Luo, Debing Zhang, Zhengyang Geng, et al. David and goliath: Small one-step model beats large diffusion with score post-training. In Forty-second International Conference on Machine Learning.

Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. Advances in Neural Information Processing Systems, 36:76525–76546, 2023.

Weijian Luo, Zemin Huang, Zhengyang Geng, J Zico Kolter, and Guo-jun Qi. One-step diffusion distillation through score implicit matching. Advances in Neural Information

Processing Systems, 37:115377–115408, 2024a.

Weijian Luo, Colin Zhang, Debing Zhang, and Zhengyang Geng. Diff-instruct\*: Towards human-preferred onestep text-to-image generative models. arXiv preprint arXiv:2410.20898, 2024b.

Yihong Luo, Tianyang Hu, Weijian Luo, Kenji Kawaguchi, and Jing Tang. Reward-instruct: A reward-centric approach to fast photo-realistic image generation. arXiv preprint arXiv:2503.13070, 2025.

Michael Lustig, David L Donoho, Juan M Santos, and John M Pauly. Compressed sensing mri. IEEE signal processing magazine, 25(2):72–82, 2008.

Morteza Mardani, Jiaming Song, Jan Kautz, and Arash Vahdat. A variational perspective on solving inverse problems with diffusion models. arXiv preprint arXiv:2305.04391, 2023.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.

Chang Qiao, Di Li, Yuting Guo, Chong Liu, Tao Jiang, Qionghai Dai, and Dong Li. Evaluation and development of deep neural networks for image super-resolution in optical microscopy. Nature methods, 18(2):194–202, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn-¨ thesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Simo Ryu. Low-rank adaptation for fast text-to-image diffusion fine-tuning. Low-rank adaptation for fast text-to-image diffusion fine-tuning, 3, 2023.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.

Bowen Song, Soo Min Kwon, Zecheng Zhang, Xinyu Hu, Qing Qu, and Liyue Shen. Solving inverse problems with latent diffusion models via hard data consistency. arXiv preprint arXiv:2307.08123, 2023.

Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations, 2022.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. Advances in neural information processing systems, 34:

1415–1428, 2021.

He Sun and Katherine L Bouman. Deep probabilistic imaging: Uncertainty quantification and multi-modal solution characterization for computational imaging. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 2628–2637, 2021.

He Sun, Katherine L Bouman, Paul Tiede, Jason J Wang, Sarah Blunt, and Dimitri Mawet. α-deep probabilistic inference (α-dpi): efficient uncertainty quantification from exoplanet astrometry to black hole feature extraction. The Astrophysical Journal, 932(2):99, 2022.

Yu Sun, Zihui Wu, Yifan Chen, Berthy T Feng, and Katherine L Bouman. Provable probabilistic imaging using scorebased generative priors. arXiv preprint arXiv:2310.10835, 2023.

Brian L Trippe, Jason Yim, Doug Tischer, David Baker, Tamara Broderick, Regina Barzilay, and Tommi Jaakkola. Diffusion probabilistic modeling of protein backbones in 3d for the motif-scaffolding problem. arXiv preprint arXiv:2206.04119, 2022.

Curtis R Vogel and Mary E Oman. Iterative methods for total variation denoising. SIAM Journal on Scientific Computing, 17(1):227–238, 1996.

Yifei Wang, Weimin Bai, Weijian Luo, Wenzheng Chen, and He Sun. Integrating amortized inference with diffusion models for learning clean distribution from corrupted images. arXiv preprint arXiv:2407.11162, 2024.

Yifei Wang, Weimin Bai, Colin Zhang, Debing Zhang, Weijian Luo, and He Sun. Uni-instruct: One-step diffusion model through unified diffusion divergence instruction. arXiv preprint arXiv:2505.20755, 2025.

Luhuan Wu, Brian Trippe, Christian Naesseth, David Blei, and John P Cunningham. Practical and asymptotically exact conditional sampling in diffusion models. Advances in Neural Information Processing Systems, 36:31372–31403, 2023.

Zihui Wu, Yu Sun, Yifan Chen, Bingliang Zhang, Yisong Yue, and Katherine Bouman. Principled probabilistic imaging using diffusion models as plug-and-play priors. Advances in Neural Information Processing Systems, 37:118389– 118427, 2024.

Xingyu Xu and Yuejie Chi. Provably robust score-based diffusion posterior sampling for plug-and-play image reconstruction. arXiv preprint arXiv:2403.17042, 2024.

Shuchen Xue, Mingyang Yi, Weijian Luo, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhi-Ming Ma. Sa-solver: Stochastic adams solver for fast sampling of diffusion models. Advances in Neural Information Processing Systems, 36:77632–77674, 2023.

Zilyu Ye, Zhiyang Chen, Tiancheng Li, Zemin Huang, Weijian Luo, and Guo-Jun Qi. Schedule on the fly: Diffusion time prediction for faster and better image generation. arXiv preprint arXiv:2412.01243, 2024.

Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman,¨ Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.

Bingliang Zhang, Wenda Chu, Julius Berner, Chenlin Meng, Anima Anandkumar, and Yang Song. Improving diffusion inverse problem solving with decoupled noise annealing. arXiv preprint arXiv:2407.01521, 2024.

Boya Zhang, Weijian Luo, and Zhihua Zhang. Enhancing adversarial robustness via score-based optimization. Advances in Neural Information Processing Systems, 36: 51810–51829, 2023.

Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu, Austin Wang, Berthy T Feng, Caifeng Zou, Yu Sun, Nikola Kovachki, Zachary E Ross, et al. Inversebench: Benchmarking plug-and-play diffusion priors for inverse problems in physical sciences. arXiv preprint arXiv:2503.11043, 2025.

Mingyuan Zhou, Huangjie Zheng, Yi Gu, Zhendong Wang, and Hai Huang. Adversarial score identity distillation: Rapidly surpassing the teacher in one step. arXiv preprint arXiv:2410.14919, 2024a.

Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024b.

Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1219–1229, 2023.

Nicolas Zilberstein, Morteza Mardani, and Santiago Segarra. Repulsive latent score distillation for solving inverse problems. arXiv preprint arXiv:2406.16683, 2024.

![](images/66f91a7e20f1425efcb5bcc4cadd18e969377ac14da6937abd0de8e011a69f17.jpg)  
Weimin Bai received the B.S. degree from China Agricultural University and the M.E. degree from the School of Computer Science, Southeast University. He is currently pursuing a PhD degree with the Academy for Advanced Interdisciplinary Studies, Peking University, China. His main research interests include generative models and inverse problems.

![](images/4768f28576358e19079212a95d4cd20575cee9c54d74f736f40841af39bebe4b.jpg)  
Yuxuan Gu is currently pursuing a master’s degree at the School of Software and Microelectronics, Peking University, where his research interests primarily focus on the distillation of diffusion models, inverse problem solving, and the development of intelligent agents.

![](images/dd0026c633d716beeab9e6a9e6bd1eb41ebc381c823e6dcf7b2f451259f3be5e.jpg)

Yifei Wang received the BS degree in artificial intelligence from Peking University, Beijing, China, in 2025. He is currently working toward the doctoral degree with Rice University. His current research interests include representation learning and generative modeling.

![](images/9cda50350e178516eb1a6fbd91eb47dbf2bbd46a501462879ede0b507ea87964.jpg)

Weijian Luo is a RedStar Senior Research Scientist at the Humane Intelligence (hi) Lab, Xiaohongshu Inc. He received his B.S. degree from the University of Science and Technology of China and his M.S. and Ph.D. degrees in statistics and generative modeling from Peking University. His research focuses on building generic AGI systems. He has published more than 15 academic papers in international journals and conferences, such as IEEE TPAMI, Transactions on Machine Learning Research, NeurIPS, ICML, ICLR, CVPR, etc.

![](images/c9db2d3e3cf1e66b032a447e8b0b21e21154427a313ee778b0b4c4208c56e076.jpg)

He Sun is an Assistant Professor in the College of Future Technology and the National Biomedical Imaging Center, Peking University. He obtained his Ph.D. from Princeton University in 2019 and his bachelor’s degree from Peking University in 2014. Prior to joining the faculty of Peking University, he was a postdoctoral researcher and an Amazon AI4Science Fellow at California Institute of Technology. His research primarily focuses on computational imaging, which tightly integrates optics, control, signal processing and machine learning to

push the boundary of scientific imaging. His past work has contributed to multiple challenging science missions, including the Event Horizon Telescope for black hole interferometric imaging, as well as to a range of biomedical imaging modalities such as ultrasound and computational microscopy.