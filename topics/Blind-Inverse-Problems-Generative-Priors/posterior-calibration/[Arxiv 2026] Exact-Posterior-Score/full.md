# Exact Posterior Score Estimation for Solving Linear Inverse Problems

Abbas Mammadov<sup>∗</sup> University of Oxford

Ozgur Kara<sup>∗</sup> UIUC

Kaan Oktay fal

Adil Kaan Akan fal

Hyungjin Chung EverEx

James Matthew Rehg UIUC

Iskander Azangulov University of Oxford

Yee Whye Teh University of Oxford

## Abstract

Diffusion and flow-based models learn powerful data priors by training a denoiser to reverse Gaussian corruption. To use this prior to solve a linear inverse problem, one needs to sample from the posterior, but the score that the prior provides is the unconditional score, not the posterior score. Existing methods either steer a fixed pretrained denoiser with approximate measurement-matching corrections, or train a conditional restoration model that abandons the denoising structure of the prior. We derive the exact posterior score in closed form for linear Gaussian inverse problems under general Gaussian interpolants, and show that posterior sampling reduces to a denoising problem at an operator-dependent shifted pivot under an anisotropic noise covariance. We turn this identity into Exact Posterior Score (EPS), a denoising training objective that preserves the input/output structure of standard pretraining and can therefore be trained from scratch or fine-tuned from a pretrained denoiser. At inference, EPS uses the same sampler as the underlying backbone, with no likelihood gradients or projections. We evaluate EPS on five linear inverse problems across FFHQ and ImageNet, where it outperforms training-free and training-based baselines on fidelity, perceptual, and distributional metrics, while using roughly an order of magnitude fewer denoiser evaluations than gradient-based posterior samplers.

## 1 Introduction

Linear inverse problems, in which an unknown signal $x _ { 0 }$ must be recovered from a noisy linear measurement $y = A x _ { 0 } + \eta$ with known forward operator A and observation noise η, are pervasive across imaging and the sciences, including compressive sensing [1, 2], accelerated medical imaging [3, 4], super-resolution [5, 6], deblurring [7, 8], and inpainting in computational photography [9, 10]. The forward operator A is typically ill-conditioned or rank-deficient, so many candidate signals are consistent with the same observation, and the right object to recover is the posterior $p ( x _ { 0 } | y )$ rather than any single point estimate. The posterior captures uncertainty over reconstructions, supports downstream decisions, and exposes the trade-off between data fidelity and prior plausibility.

Diffusion and flow-based generative models offer powerful, expressive data priors for this task, learning a denoising trajectory from noise back to clean samples [11–17]. The central question is how to turn this trajectory into a sampler from $p ( x _ { 0 } | y )$ . The reverse-time sampler needs the posterior score $\nabla _ { x _ { t } } \log { p ( x _ { t } | y ) }$ , not the unconditional prior score $\nabla _ { x _ { t } } \log { p ( x _ { t } ) }$ that diffusion training provides. Replacing the former by an approximation introduces bias at every step, which compounds into oversmoothing, hallucinated structure, or poorly calibrated uncertainty. Existing methods fall into two broad camps.

![](images/baa4f58aaeb73c684a44461261846a587d516f4d1e6a30bd5c168b5095cd88a3.jpg)  
Figure 1: EPS turns posterior sampling into denoising with the right query geometry. Instead of denoising an isotropic query at $x _ { t }$ , the measurement shifts the query to the posterior pivot $\mu _ { \star }$ and reshapes the noise into an anisotropic covariance $\Sigma _ { \star }$ . Measured directions become more certain, while unobserved directions remain uncertain. EPS trains a denoiser for this anisotropic geometry and reuses the backbone’s unconditional sampler unchanged. The first step of the resulting sampler corresponds to an estimate of the posterior mean $\mathbb { E } [ x _ { 0 } | y ]$ , which typically has higher PSNR but is over-smoothed, while the sample produced at the end (in this case, 100 steps) has more details.

Training-free methods keep a pretrained denoising backbone fixed and add a measurement-matching update at each reverse step. The prototypical example is Diffusion Posterior Sampling (DPS) [18], which differentiates a measurement loss through the unconditional denoiser, with variants using projections, denoised estimates, or task-specific correction rules [18–25]. This route is attractive because it is zero-shot and inherits the strong unconditional prior of a pretrained model. However, the added update is only an approximation to the true measurement-matching score, and even momentmatching variants that track anisotropic uncertainty in $p ( x _ { 0 } | x _ { t } )$ [26–28] only refine the unconditional denoising query. Asymptotically exact alternatives based on sequential Monte Carlo [29–31] avoid this approximation, but at the cost of running many particle trajectories per observation.

Training-based methods sidestep the approximation question by training a new model specifically for the inverse problem, with the measurement y as input. This family includes conditional diffusion models that learn a measurement-conditional score [32–34], bridge-based methods that build a trajectory directly from $y$ to the data [35, 36], and methods that distill a posterior sampler from a pretrained diffusion prior [37, 38]. In all cases, the network is exposed to the raw measurement rather than to the geometry of the exact posterior denoising query, so it must learn the operator dependence end-to-end.

In contrast, we observe that for linear Gaussian inverse problems the exact posterior score has a closed form, with a simple structural meaning. As Figure 1 illustrates, posterior sampling is still a denoising problem, but with a measurement-aware input and an operator-dependent anisotropic noise covariance. We use this identity to define Exact Posterior Score (EPS), a denoising training objective whose target and loss match those of standard pretraining, with the input replaced by a measurement-dependent pivot. EPS can therefore be trained from scratch or fine-tuned efficiently from a pretrained denoiser. At inference, it runs the underlying backbone’s sampler unchanged, with no likelihood gradients, projections, or inner optimization.

Our contributions are as follows.

• We derive the exact posterior score for linear Gaussian inverse problems under general Gaussian interpolants, and show that posterior sampling reduces to denoising at an operator-dependent shifted pivot under an anisotropic covariance. We also pinpoint where existing approximate-guidance methods deviate from this exact identity.

• We turn the identity into EPS, a denoising training objective and sampling algorithm that preserves the structure of standard pretraining while incorporating the exact posterior geometry. EPS can be trained from scratch or fine-tuned from a pretrained checkpoint, and it uses the underlying backbone’s sampler at inference.

• We evaluate EPS on five linear inverse problems across FFHQ and ImageNet, reporting pointwise fidelity, perceptual quality, and distributional calibration metrics, and find consistent improvements over both training-free and training-based baselines at substantially smaller sampling budgets.

## 2 Background

## 2.1 Generative Models as Denoising Trajectories

Let $x _ { 0 } \sim p _ { \mathrm { d a t a } }$ and $\epsilon \sim \mathcal { N } ( 0 , I _ { d } )$ . A stochastic interpolant defines

$$
x _ { t } = \alpha _ { t } x _ { 0 } + \beta _ { t } \epsilon , \qquad t \in [ 0 , T ] ,\tag{1}
$$

with $( \alpha _ { 0 } , \beta _ { 0 } ) = ( 1 , 0 )$ and $( \alpha _ { T } , \beta _ { T } ) = ( 0 , 1 )$ in the data-to-noise convention. Different choices of $\left( \alpha _ { t } , \beta _ { t } \right)$ recover variance-preserving diffusion [12, 13], rectified flow [16], and EDM [17], with corresponding network parameterizations as a score, noise predictor, velocity, or EDM-preconditioned denoiser; the conversion between them is deterministic once $\left( \alpha _ { t } , \beta _ { t } \right)$ are fixed, so we write all backbones through a learned denoiser $D _ { \theta }$ . The forward kernel is

$$
p ( x _ { t } | x _ { 0 } ) = \mathcal { N } ( x _ { t } ; \alpha _ { t } x _ { 0 } , \beta _ { t } ^ { 2 } I _ { d } ) .\tag{2}
$$

The marginal score $s _ { t } ( x ) = \nabla _ { x } \log p ( x )$ is equivalent to the optimal denoiser through Tweedie’s identity,

$$
D _ { t } ( x ) : = \mathbb { E } [ x _ { 0 } | x _ { t } = x ] = \frac { 1 } { \alpha _ { t } } \big ( x + \beta _ { t } ^ { 2 } s _ { t } ( x ) \big ) ,\tag{3}
$$

and the reverse velocity follows from the same identity:

$$
v _ { t } ( x ) = \mathbb { E } [ \dot { \alpha } _ { t } x _ { 0 } + \dot { \beta } _ { t } \epsilon \mid x _ { t } = x ] = \frac { \dot { \beta } _ { t } } { \beta _ { t } } x + \left( \dot { \alpha } _ { t } - \frac { \alpha _ { t } \dot { \beta } _ { t } } { \beta _ { t } } \right) D _ { t } ( x ) .\tag{4}
$$

This viewpoint sets up EPS: if the posterior denoiser is known, the posterior score and posterior velocity follow immediately from the same identities, and the base model’s sampler can be reused.

## 2.2 Inverse Problems and Approximate Posterior Sampling

We observe

$$
y = A x _ { 0 } + \eta , \qquad \eta \sim \mathcal { N } ( 0 , \sigma _ { y } ^ { 2 } I _ { m } ) ,\tag{5}
$$

for a known linear operator $A \in \mathbb { R } ^ { m \times d }$ . This notation covers masks, downsampling, and convolutional blur operators, and includes rank-deficient settings where many signals are consistent with the same observation. The target is the posterior $p ( x _ { 0 } | y ) \propto p ( y | x _ { 0 } ) p _ { \mathrm { d a t a } } ( x _ { 0 } )$ . A reverse sampler should therefore use

$$
\begin{array} { r } { \nabla _ { x _ { t } } \log p ( x _ { t } | y ) = \nabla _ { x _ { t } } \log p ( x _ { t } ) + \nabla _ { x _ { t } } \log p ( y | x _ { t } ) , } \end{array}\tag{6}
$$

where the second term is the measurement-matching score. Zero-shot solvers approximate it with the template $\nabla _ { x _ { t } } \log { p ( y | x _ { t } ) } \approx - L _ { t } M _ { t } / G _ { t }$ [28], where $M _ { t }$ is a measurement residual, $L _ { t }$ lifts it back to sample space, and $G _ { t }$ is the guidance strength.

## 3 Exact Posterior Score (EPS)

We now derive the posterior score and convert it into a training objective. The derivation has two pieces. First, a Gaussian product identifies the correct pivot and covariance. Second, an anisotropic Tweedie identity turns the resulting smoothed density into a denoiser.

## 3.1 Anisotropic Tweedie Identity

For a positive definite covariance matrix Σ, define the Gaussian-smoothed data density

$$
p _ { \mathrm { d a t a } } ^ { \Sigma } ( \mu ) = \int { \mathcal N } ( \mu ; x _ { 0 } , \Sigma ) p _ { \mathrm { d a t a } } ( x _ { 0 } ) \mathrm d x _ { 0 } ,\tag{7}
$$

and the corresponding optimal denoiser

$$
D _ { \Sigma } ( \mu ) = \mathbb { E } [ x _ { 0 } | x _ { 0 } + \xi = \mu ] , \qquad \xi \sim \mathcal { N } ( 0 , \Sigma ) .\tag{8}
$$

Then, by the anisotropic form of Tweedie’s formula [39, 40],

$$
D _ { \Sigma } ( \mu ) = \mu + \Sigma \nabla _ { \mu } \log p _ { \mathrm { d a t a } } ^ { \Sigma } ( \mu ) ,\tag{9}
$$

which is the usual Tweedie formula when Σ is a scalar multiple of the identity. EPS uses this identity with a covariance that is not chosen by hand, but rather is derived from the inverse problem formulation.

## 3.2 Closed-Form Posterior Score

The posterior marginal can be written as

$$
p ( x _ { t } | y ) \propto \int p ( x _ { t } | x _ { 0 } ) p ( y | x _ { 0 } ) p _ { \mathrm { d a t a } } ( x _ { 0 } ) \mathrm { d } x _ { 0 } .\tag{10}
$$

Both factors inside the integral are Gaussian in $x _ { 0 }$ . Completing the square gives the following result. Theorem 1 (Exact posterior score). Under the linear Gaussian inverse problem (5) and the interpolant (1), the posterior score at time t is

$$
\nabla _ { x _ { t } } \log { p ( x _ { t } | y ) } = \frac { 1 } { \beta _ { t } ^ { 2 } } \Big ( \alpha _ { t } D _ { \Sigma _ { \star } ( t ) } \big ( \mu _ { \star } ( x _ { t } , y , t ) \big ) - x _ { t } \Big ) ,\tag{11}
$$

where

$$
\Sigma _ { \star } ( t ) = \left( { \frac { \alpha _ { t } ^ { 2 } } { \beta _ { t } ^ { 2 } } } I _ { d } + { \frac { 1 } { \sigma _ { y } ^ { 2 } } } A ^ { \top } A \right) ^ { - 1 } , \qquad \mu _ { \star } ( x _ { t } , y , t ) = \Sigma _ { \star } ( t ) \left( { \frac { \alpha _ { t } } { \beta _ { t } ^ { 2 } } } x _ { t } + { \frac { 1 } { \sigma _ { y } ^ { 2 } } } A ^ { \top } y \right) .\tag{12}
$$

Equivalently, $D _ { \Sigma _ { \star } ( t ) } ( \mu _ { \star } ( x _ { t } , y , t ) ) = \mathbb { E } [ x _ { 0 } | x _ { t } , y ] .$

The proof is given in Appendix A.2. The theorem says that posterior sampling is still a denoising problem, but not the isotropic one seen in unconditional pretraining. The denoiser must be queried at a measurement-aware input $\mu _ { \star }$ under a measurement-aware anisotropic noise covariance $\Sigma _ { \star }$ .

Proposition 2 (Posterior velocity). The posterior velocity associated with the interpolant (1) is

$$
v _ { t } ^ { y } ( x _ { t } ) = \mathbb { E } [ \dot { \alpha } _ { t } x _ { 0 } + \dot { \beta } _ { t } \epsilon | x _ { t } , y ] = \frac { \dot { \beta } _ { t } } { \beta _ { t } } x _ { t } + \left( \dot { \alpha } _ { t } - \frac { \alpha _ { t } \dot { \beta } _ { t } } { \beta _ { t } } \right) D _ { \Sigma _ { \star } ( t ) } ( \mu _ { \star } ( x _ { t } , y , t ) ) .\tag{13}
$$

Thus estimating the exact posterior denoiser is equivalent to estimating the exact posterior flow.

The proof is given in Appendix A.3.

Posterior pivot. We call $\mu _ { \star }$ the posterior pivot because the proof of Theorem 1 (Appendix A.2) shows that the joint quadratic form in $( x _ { t } , y , x _ { 0 } )$ pivots about $\mu _ { \star } ( x _ { t } , y , t )$ : Completing the square sends the entire dependence on $x _ { 0 }$ into a single Gaussian centered at $\mu _ { \star }$ with covariance $\dot { \Sigma } _ { \star } ( t )$ while $x _ { t }$ and $y$ enter only through this pivot and a multiplicative normalizer. Equivalently, $\mu _ { \star }$ is the precision-weighted Bayesian fusion of the current state and the measurement under the two Gaussian likelihoods, before the data prior $p _ { \mathrm { d a t a } }$ is folded in by the denoiser $D _ { \Sigma _ { \star } ( t ) }$ . The pivot is therefore the only summary statistic of $( x _ { t } , y )$ that the posterior denoiser needs to see.

Computing $\mu _ { \star }$ . Although (12) involves inverting a $d \times d$ matrix in general, every operator we consider admits a fast structured solve. For binary inpainting masks $A ^ { \top } \bar { A }$ is diagonal, for downsam pling it is block-diagonal, and for circular convolutions used in deblurring it is diagonalized by the FFT. The per-step cost of computing $\mu _ { \star }$ is therefore negligible relative to a denoiser forward pass. For more general operators, $\mu ,$ <sub>⋆</sub> can still be obtained efficiently via conjugate gradient applied to the symmetric positive-definite system $\left( \alpha _ { t } ^ { 2 } / \beta _ { t } ^ { 2 } I + \sigma _ { y } ^ { - 2 } A ^ { \top } A \right) \mu _ { \star } = \left( \alpha _ { t } / \tilde { \beta _ { t } ^ { 2 } } \right) \smile \kappa _ { t } + \sigma _ { y } ^ { - 2 } A ^ { \top } y$ , which only requires matrix-vector products with A and $A ^ { \top }$ . We measure these costs directly in Appendix D.12, where the structured $\mu _ { \star }$ solve adds only sub-millisecond overhead per sampling step.

## 3.3 What was missing in Training-Free Methods

Theorem 1 also pinpoints what existing training-free methods miss. Combining (11) with the unconditional Tweedie identity (3), the measurement-matching score can be written as a difference of two denoisers,

$$
\nabla _ { x _ { t } } \log p ( y | x _ { t } ) \ = \ \nabla _ { x _ { t } } \log p ( x _ { t } | y ) - \nabla _ { x _ { t } } \log p ( x _ { t } ) \ = \ \frac { \alpha _ { t } } { \beta _ { t } ^ { 2 } } \Big ( D _ { \Sigma _ { \star } ( t ) } \big ( \mu _ { \star } ( x _ { t } , y , t ) \big ) \ - \ D _ { t } ( x _ { t } ) \Big ) .\tag{14}
$$

The exact guidance is the gap between the posterior denoiser evaluated at the pivot $\mu _ { \star }$ under the anisotropic covariance $\Sigma _ { \star } ( t )$ , and the unconditional denoiser evaluated at $x _ { t }$ . Methods that follow the template from [28], including DPS, DDNM, ΠGDM, and moment-matching variants [18–20, 26, 27], all approximate the first denoiser using only the second, by differentiating a measurement loss through $D _ { t } ( x _ { t } )$ , projecting $D _ { t } ( x _ { t } )$ onto an affine subspace, or fitting a Gaussian to $p ( x _ { 0 } | x _ { t } )$ and comparing it to $y .$ . They thus evaluate the network at a different input than the exact identity, querying $D _ { \theta }$ at $x _ { t }$ rather than at the pivot $\mu _ { \star }$ , which itself depends on $y .$ Even moment-matching methods that use anisotropic information approximate $p ( x _ { 0 } | x _ { t } )$ , the denoising distribution before the measurement is incorporated, whereas the exact object is $p ( x _ { 0 } | x _ { t } , y )$ , the denoising distribution after the measurement is fused into the kernel. The two coincide only in degenerate cases such as isotropic $A ^ { \top } A$ or high-noise limits, but not generically.

## 3.4 The EPS Training Objective

Theorem 1 reduces posterior sampling to a single object, the anisotropic posterior denoiser $D _ { \Sigma _ { \star } ( t ) } ( \mu _ { \star } ( x _ { t } , y , t ) )$ . Two of the three quantities involved are analytically tractable. Given $( x _ { t } , y , t )$ and the operator parameters $( A , \sigma _ { y } )$ , the pivot $\mu _ { \star } ( x _ { t } , y , t )$ and covariance $\Sigma _ { \star } ( t )$ are deterministic, closed-form functions defined by (12), and for the structured operators we consider, both can be computed in essentially the cost of an FFT or an element-wise solve (Section 3.2). What we cannot compute analytically is the denoiser itself. The expression $D _ { \Sigma _ { \star } ( t ) } ( \mu ) = \mathbb { E } [ x _ { 0 } | x _ { 0 } + \xi = \mu ]$ with $\xi \sim \mathcal { N } ( 0 , \Sigma _ { \star } ( t ) )$ requires the data distribution $p _ { \mathrm { d a t a } }$ , which is accessible only through data samples, while the pretrained unconditional denoiser was trained with isotropic noise so is not directly applicable (but can be used to initialize the EPS training).

Following standard approaches for training diffusion models, we therefore learn it by regression. To enable efficient noising of $x _ { 0 }$ using the anistropic noise covariance, we note the following result:

Proposition 3 (Isotropic simulation of the anisotropic pivot). Let $x _ { t } = \alpha _ { t } x _ { 0 } + \beta _ { t } \epsilon$ with $\epsilon \sim \mathcal { N } ( 0 , I _ { d } )$ and $y = A x _ { 0 } + \sigma _ { y } \eta$ with $\eta \sim \mathcal { N } ( 0 , I _ { m } )$ , independently. Define $\mu _ { \star } ( x _ { t } , y , t )$ and $\Sigma _ { \star } ( t )$ as in (12). Then, conditional on $x _ { 0 }$

$$
\mu _ { \star } ( x _ { t } , y , t ) | x _ { 0 } \sim \mathcal { N } ( x _ { 0 } , \Sigma _ { \star } ( t ) ) .\tag{15}
$$

Thus the anisotropic corruption required by the exact posterior denoiser is induced by the closed-form pivot construction itself; it is not necessary to sample anisotropic noise directly.

The proof is given in Appendix A.4. This result is related to existing GP and linear regression literature (Appendix $\mathbf { A . 7 } )$ , and shows that the required anistropic noising can be computed efficiently.

The objective. EPS regresses a denoising network $D _ { \theta }$ onto clean targets given the pivot input,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { E P S } } ( \theta ) = \mathbb { E } _ { \boldsymbol { x } _ { 0 } , \boldsymbol { y } , t , \epsilon } \Big [ w ( t ) \ \| D _ { \theta } ( \mu _ { \star } ( \boldsymbol { x } _ { t } , \boldsymbol { y } , t ) , \boldsymbol { y } , t ) - \boldsymbol { x } _ { 0 } \| ^ { 2 } \Big ] , } \end{array}\tag{16}
$$

where $x _ { t } = \alpha _ { t } x _ { 0 } + \beta _ { t } \epsilon$ and $y \sim \mathcal { N } ( A x _ { 0 } , \sigma _ { y } ^ { 2 } I _ { m } )$ . Standard arguments show that the squared-loss minimizer of (16) is $\mathbb { E } [ x _ { 0 } \vert \mu _ { \star } , y , t ]$ , which by Theorem 1 equals $D _ { \Sigma _ { \star } ( t ) } ( \mu _ { \star } ) = \mathbb { E } [ x _ { 0 } | x _ { t } , y ]$ . Once trained, the posterior score and posterior velocity follow without further effort from (11) and (13). While strictly unnecessary, note that EPS also passes y to the learned denoiser so that the network is explicitly conditioned on the observed measurement while learning the posterior denoising map; for a fixed operator, the closed-form pivot is a sufficient statistic, but conditioning on $y$ makes the dependence on the particular inverse-problem instance explicit in the learned model, and slightly improves results (Appendix D.1).

The upshot. The minimizer of (16) has the same target type (a clean image $x _ { 0 } ~ \in ~ \mathbb { R } ^ { d } )$ and the same squared-loss regression structure as the standard pretrained denoiser $D _ { \theta _ { 0 } } ( x _ { t } , t )$ The structural changes are (i) the input is the pivot $\mu _ { \star }$ <sub>⋆</sub> rather than $x _ { t } ,$ and (ii) implicitly, through the pivot construction, the input is corrupted by anisotropic noise of covariance $\Sigma _ { \star } ^ { \bar { } } ( t )$ rather than isotropic noise of scale $\beta _ { t }$ . Why does this matter? At the same input $\mu _ { \star }$ , the unconditional denoiser $D _ { \theta _ { \mathrm { f } } }$ would return a biased estimate, because it implicitly assumes its input is corrupted by isotropic noise, whereas the true noise on $\mu _ { \star }$ is operator-dependent and anisotropic. The network must therefore learn how to denoise under this measurement-induced anisotropic geometry.

Empirically, when warm-started from the pretrained unconditional denoiser, EPS converges in a small fraction of the iterations needed by other training-based posterior solvers (Appendix D.3). We attribute this to its proximity to the pretraining task. Conditional methods such as Palette [32] and InvFusion [34] preserve the noise schedule and noisy intermediates of pretraining but condition the score on the raw measurement, so the network must spend capacity learning the operator dependence on top of the pretrained denoising prior. Bridge methods such as InDI [35] and I2SB [36] go further by replacing the noise-to-data forward process with a measurement-to-data one, so the network must learn a different conditional mapping from scratch. EPS preserves the forward process and the per-time denoising query, and only adapts to operator-induced anisotropic geometry $\textstyle { \mathrm { \bar { E } } } _ { \star } ( t )$

Algorithm 1 EPS training   
Require: Pretrained denoiser $D _ { \theta _ { 0 } }$ (or random init), data distribution $p _ { \mathrm { d a t a } } ,$ operator distribution   
$p ( A )$ , observation noise $\sigma _ { y } ,$ noise schedule $\left( \alpha _ { t } , \beta _ { t } \right)$   
1: Initialize $\theta  \theta _ { 0 }$ (or randomly).   
2: while not converged do   
3: Sample $x _ { 0 } \sim p _ { \mathrm { d a t a } } ,$ A ∼ p(A), t ∼ p(t), ϵ ∼ N (0, I<sub>d</sub>), and $\eta \sim \mathcal { N } ( 0 , I _ { m } )$   
4: Form $y  A x _ { 0 } + \sigma _ { y } \eta$ and $x _ { t } \gets \alpha _ { t } x _ { 0 } + \beta _ { t } \epsilon .$   
5: Compute $\Sigma _ { \star } ( t )$ and $\mu _ { \star } ( x _ { t } , y , t )$ from (12) via the structured solve for A.   
6: Evaluate the posterior denoising loss $\mathcal { L } = w ( t ) \| D _ { \theta } ( \mu _ { \star } , y , t ) - x _ { 0 } \| ^ { 2 }$   
7: Update θ by gradient descent on ${ \mathcal { L } } ,$ and update EMA weights if used by the base sampler.   
8: end while   
9: return Trained denoiser $D _ { \theta } .$

## 3.5 Sampling

At inference, EPS uses the same deterministic or stochastic sampler as the underlying diffusion backbone, replacing every denoiser call by $D _ { \theta } ( \mu _ { \star } ( x _ { t } , y , t ) , y , t )$ . No likelihood gradient, projection, or inner optimization is required during sampling. Since $\mu ,$ is obtained by a structured linear solve, the per-step overhead is negligible relative to a denoiser forward pass (see Section 3.2).

The High-Noise Limit. Theorem 1 also characterizes the sampler at the start of the reverse trajectory, where the noise scale $\sigma _ { t }$ is largest. In this regime both the pivot and its anisotropic denoiser take a particularly simple form:

Observation 4 (High-noise posterior-mean limit). With EDM parameterization $\alpha _ { t } = 1 , \beta _ { t } = \sigma _ { t }$ and let $P _ { \mathcal { N } ( A ) } = I - A ^ { \dagger } A$ be the orthogonal projector onto the nullspace of A. Then, as $\sigma _ { t }  \infty ,$

$$
\mu _ { \star } ( x _ { t } , y , t ) \longrightarrow A ^ { \dagger } y + P _ { \mathit { N } ( A ) } x _ { t } .\tag{17}
$$

Moreover, the corresponding anisotropic denoiser satisfies

$$
D _ { \Sigma _ { \star } ( t ) } ( \mu _ { \star } ( x _ { t } , y , t ) ) \longrightarrow \mathbb { E } [ x _ { 0 } | y ] .\tag{18}
$$

Thus a single high-noise EPS denoiser evaluation is a posterior-mean estimator.

The proof is given in Appendix A.5. The pivot limit (17) is the EPS-specific part of the statement: at the start of sampling, the network is queried at the pseudo-inverse reconstruction $A ^ { \dagger } y$ plus pure noise in the nullspace of A. The posterior-mean limit (18), by contrast, is generic rather than a contribution of EPS: by Theorem 1 it is equivalent to $\mathbb { E } [ x _ { 0 } | x _ { t } , \dot { y } ]  \mathbb { E } [ x _ { 0 } | y ]$ , which holds simply because $x _ { t }$ carries vanishing information about $x _ { 0 }$ as $\sigma _ { t } \to \infty$ . Under perfect learning, the same is therefore true of any training-based method whose denoiser targets $\mathbb { E } [ x _ { 0 } | x _ { t } , y ]$ along the same path, $\mathrm { e . g . }$ ., Palette [32]; Appendix D.11 confirms this empirically. We nevertheless find it a useful way to read the sampling path of all such methods: it starts at the posterior mean $\mathbb { E } [ x _ { 0 } | y ]$ and ends at a sample from $p ( x _ { 0 } | y )$

## 4 Experiments

We evaluate EPS on five linear inverse problems across two datasets. The main comparison is at 64×64, where every baseline is run under the same backbone, task, and evaluation protocol and where distributional metrics can be computed fairly. Additional 256×256 results are in Appendix D.10.

## 4.1 Experimental Setup

Datasets, backbones, and tasks. We use FFHQ and ImageNet. For ImageNet we use the publicly available class-conditional EDM [17] checkpoint released with the original codebase. For FFHQ we train an EDM checkpoint from scratch because the released FFHQ model does not reserve images for validation. All methods that require training or fine-tuning use the same backbone as EPS, and all zero-shot solvers use the same pretrained denoiser. We consider five linear inverse problems: random inpainting with 70% missing pixels, centered box inpainting, 4× super-resolution, Gaussian deblurring, and motion deblurring. In all cases we add Gaussian observation noise with standard deviation $\sigma _ { y } = 0 . 0 5$ . Operator details and randomization protocols are in Appendix B.

![](images/bcab6b4ca7189588b7f82229fcffac296cdeafb48e2c7ba0a25405492add0d76.jpg)  
Figure 2: Qualitative reconstructions across the five inverse problems. Numbers indicate PSNR values.

![](images/457d9845f0cf4bf8faf5161f022558361fb8670c2c4e07cb7fa15a1aecefb8f5.jpg)

![](images/267cb456e9c531d5d934570a27ac1b82e8f11dac58dce339e1803dca4f03f22f.jpg)

![](images/f047c08487a8983b33b29271748cf1be2b7f00006561faaed1acfcf3f0e6dfe4.jpg)

![](images/b73502cca8bd9db66719257a2e795cf912e48575c04bfeebc2d66e067b92b250.jpg)  
Figure 3: EPS converges in ∼20 NFE; baselines never catch up. Sampling-step sensitivity for FID and CRPS-Inception on random inpainting and 4× super-resolution, across both ImageNet-64 and FFHQ-64. EPS plateaus within ∼20 NFE on every panel, while DPS, DAPS, DDNM, ΠGDM, and MPGD continue to improve out to 100 NFE without reaching the EPS asymptote.

Baselines and metrics. The sampling-based family comprises DPS [18], DAPS [22], DDNM [19], ΠGDM [20], and MPGD [23]. The training-based family is represented by Palette [32], implemented under the same EDM backbone and compute budget; Palette can be viewed as the EPS pipeline with $x _ { t }$ replacing $\mu _ { \star } ,$ isolating the contribution of the shifted pivot. We report PSNR and SSIM [41], LPIPS [42], and FID [43] for pointwise and perceptual quality, and CRPS [44] and MMD [45] in pixel and Inception feature space [46, 47] for distributional calibration. We use raw Inception features rather than the L2-normalized features of Mammadov et al. [46], so absolute values differ but relative comparisons across methods are preserved. Definitions are in Appendix F.

## 4.2 Main Results

Tables 1 and 6 report all five tasks, all baselines, and all metrics in a single view per dataset. We include EPS at 100, 20, and 1 NFE: the first two correspond to the posterior-sampling regime at different budgets, and the 1-NFE row tests the posterior-mean prediction from Section 3.5.

Table 1: Quantitative comparison on ImageNet-64. Five linear inverse problems, 100 images × 10 seeds. Baselines follow the sampler and hyperparameters from their respective papers; Palette and EPS use the EDM Euler sampler at 1 NFE per step. Best in bold, second-best underlined; EPS rows highlighted in light pink. † The NFE=1 row applies a single Tweedie evaluation $D _ { \theta } ( \mu _ { \star } , \sigma _ { \operatorname* { m a x } } )$ returning the conditional posterior mean $\mathbb { E } [ x _ { 0 } \mid y ]$ in one shot rather than a posterior sample.
<table><tr><td>Task</td><td>Method</td><td>NFE</td><td>PSNR (↑)</td><td>SSIM (↑)</td><td>LPIPS (↓)</td><td>FID (↓)</td><td>MMDpix (↓)</td><td>MMDInc (↓)</td><td>CRPSpix (↓)</td><td>CRPSInc (↓)</td></tr><tr><td rowspan="8">Inpaint (random)</td><td>DPS</td><td>250</td><td>21.56</td><td>0.6573</td><td>0.2158</td><td>183.94</td><td>-4.88e-03</td><td>2.51e-02</td><td>5.88</td><td>9.89</td></tr><tr><td>DAPS</td><td>100</td><td>20.89</td><td>0.5555</td><td>0.3168</td><td>224.39</td><td>-2.22e-03</td><td>4.94e-02</td><td>6.91</td><td>12.09</td></tr><tr><td>DDNM</td><td>100</td><td>22.63</td><td>0.7127</td><td>0.1758</td><td>138.53</td><td>-5.78e-03</td><td>7.29e-03</td><td>5.23</td><td>7.99</td></tr><tr><td>IIGDM</td><td>100</td><td>23.95</td><td>0.7780</td><td>0.1198</td><td>99.60</td><td>-6.40e-03</td><td>-2.22e-03</td><td>4.54</td><td>6.32</td></tr><tr><td>MPGD</td><td>100</td><td>19.62</td><td>0.5447</td><td>0.3151</td><td>223.43</td><td>2.56e-03</td><td>4.92e-02</td><td>7.81</td><td>11.94</td></tr><tr><td>Palette</td><td>100</td><td>24.09</td><td>0.7869</td><td>0.1011</td><td>81.88</td><td>-6.50e-03</td><td>-4.41e-03</td><td>4.16</td><td>5.52</td></tr><tr><td>EPS (ours)</td><td>100</td><td>24.34</td><td>0.7948</td><td>0.0979</td><td>79.60</td><td>-6.52e-03</td><td>-4.51e-03</td><td>4.04</td><td>5.41</td></tr><tr><td>EPS (ours) EPS (ours)†</td><td>20</td><td>24.87</td><td>0.8122</td><td>0.0910</td><td>77.06</td><td>-6.52e-03</td><td>-4.51e-03</td><td>4.02</td><td>5.44</td></tr><tr><td></td><td>1</td><td>26.60</td><td>0.8580</td><td>0.0933</td><td>88.59</td><td>-6.15e-03</td><td>-4.84e-04</td><td>4.98</td><td>8.32</td></tr><tr><td rowspan="10">Inpaint (box)</td><td>DPS</td><td>250</td><td>19.44</td><td>0.6587</td><td>0.1891</td><td>142.05</td><td>-5.63e-03</td><td>1.01e-02</td><td>7.25</td><td>8.19</td></tr><tr><td>DAPS</td><td>100</td><td>20.62</td><td>0.6600</td><td>0.2100</td><td>161.62</td><td>-5.00e-03</td><td>2.38e-02</td><td>7.36</td><td>9.57</td></tr><tr><td>DDNM</td><td>100</td><td>20.38</td><td>0.7049</td><td>0.1824</td><td>125.82</td><td>-5.37e-03</td><td>4.18e-03</td><td>7.00</td><td>7.54</td></tr><tr><td>IIGDM</td><td>100</td><td>20.53</td><td>0.7292</td><td>0.1490</td><td>104.50</td><td>-5.87e-03</td><td>-2.69e-03</td><td>6.59</td><td>6.50</td></tr><tr><td>MPGD</td><td>100</td><td>19.35</td><td>0.6467</td><td>0.2463</td><td>167.19</td><td>-3.44e-03</td><td>1.93e-02</td><td>8.67</td><td>9.68</td></tr><tr><td>Palette</td><td>100</td><td>21.12</td><td>0.7541</td><td>0.1218</td><td>92.73</td><td>-6.10e-03</td><td>-4.12e-03</td><td>5.92</td><td>5.93</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EPS (ours)</td><td>100 20</td><td>21.24 21.72</td><td>0.7569 0.7667</td><td>0.1196 0.1166</td><td>91.07 90.12</td><td>-6.11e-03</td><td>-4.23e-03</td><td>5.87</td><td>5.84</td></tr><tr><td>EPS (ours) EPS (ours)†</td><td>1</td><td>23.60</td><td>0.7908</td><td>0.1514</td><td>129.11</td><td>-6.08e-03 -5.60e-03</td><td>-4.14e-03 7.34e-03</td><td>5.86 7.16</td><td>5.90</td></tr><tr><td></td><td></td><td>19.59</td><td>0.4511</td><td>0.3367</td><td>200.98</td><td></td><td>2.67e-02</td><td></td><td>10.38</td></tr><tr><td rowspan="10">Super-res. (4×)</td><td>DPS DAPS</td><td>250 100</td><td>18.46</td><td>0.3324</td><td>0.4772</td><td>258.25</td><td>-5.06e-03 6.09e-03</td><td>8.54e-02</td><td>7.77 10.01</td><td>10.36 12.52</td></tr><tr><td>DDNM</td><td>100</td><td>21.10</td><td>0.5523</td><td>0.3055</td><td>169.94</td><td>-5.10e-03</td><td>1.41e-02</td><td></td><td></td></tr><tr><td>ΠIGDM</td><td>100</td><td>20.25</td><td></td><td></td><td></td><td></td><td></td><td>7.32</td><td>9.74</td></tr><tr><td>MPGD</td><td>100</td><td>20.57</td><td>0.5318</td><td>0.2499</td><td>144.36</td><td>-5.78e-03</td><td>4.61e-03</td><td>6.76</td><td>8.05</td></tr><tr><td></td><td>100</td><td>20.24</td><td>0.5103</td><td>0.3426</td><td>221.50</td><td>-3.20e-03</td><td>6.97e-02</td><td>9.02</td><td>13.08</td></tr><tr><td>Palette</td><td></td><td></td><td>0.5364</td><td>0.2220</td><td>128.76</td><td>-5.86e-03</td><td>-2.79e-03</td><td>6.50</td><td>7.33</td></tr><tr><td>EPS (ours)</td><td>100</td><td>20.25</td><td>0.5369</td><td>0.2207</td><td>128.80</td><td>-5.86e-03</td><td>-2.84e-03</td><td>6.52</td><td>7.35</td></tr><tr><td>EPS (ours)</td><td>20</td><td>20.90</td><td>0.5692</td><td>0.2127</td><td>129.13</td><td>-5.72e-03</td><td>-1.98e-03</td><td>6.51</td><td>7.54</td></tr><tr><td>EPS (ours)†</td><td>1</td><td>22.78</td><td>0.6530</td><td>0.2455</td><td>182.92</td><td>-4.97e-03</td><td>1.98e-02</td><td>8.06</td><td>13.47</td></tr><tr><td>DPS</td><td>250</td><td>25.05</td><td>0.7804</td><td>0.1511</td><td>158.46</td><td>-5.24e-03</td><td>1.57e-02</td><td>4.48</td><td>8.96</td></tr><tr><td rowspan="8">Gaussian deblur</td><td>DAPS</td><td>100</td><td>25.97</td><td>0.7680</td><td>0.1466</td><td>152.47</td><td>-6.50e-03</td><td>2.78e-02</td><td>3.69</td><td>9.58</td></tr><tr><td>DDNM</td><td>100</td><td>26.28</td><td>0.7903</td><td>0.1227</td><td>101.61</td><td>-6.71e-03</td><td>9.01e-03</td><td>2.92</td><td>7.18</td></tr><tr><td>ΠIGDM</td><td>100</td><td>28.13</td><td>0.8785</td><td>0.0742</td><td>66.10</td><td>-6.70e-03</td><td>-2.49e-03</td><td>2.91</td><td>5.68</td></tr><tr><td>MPGD</td><td>100</td><td>20.74</td><td>0.6375</td><td>0.2396</td><td>175.42</td><td>2.24e-02</td><td>2.13e-02</td><td>8.01</td><td>10.20</td></tr><tr><td>Palette</td><td>100</td><td>29.15</td><td>0.9010</td><td>0.0491</td><td>46.62</td><td>-6.83e-03</td><td>-5.56e-03</td><td>2.26</td><td>4.11</td></tr><tr><td>EPS (ours)</td><td>100</td><td>29.18</td><td>0.9015</td><td>0.0486</td><td>46.55</td><td></td><td></td><td></td><td></td></tr><tr><td>EPS (ours)</td><td>20</td><td>29.68</td><td>0.9108</td><td>0.0449</td><td>44.25</td><td>-6.83e-03 -6.83e-03</td><td>-5.55e-03 -5.55e-03</td><td>2.25 2.24</td><td>4.11 4.13</td></tr><tr><td>EPS (ours)†</td><td>1</td><td>28.82</td><td>0.9194</td><td>0.0606</td><td>56.53</td><td>-5.14e-03</td><td>-3.46e-03</td><td>4.09</td><td>7.73</td></tr><tr><td rowspan="8">Motion deblur</td><td>DPS</td><td>250</td><td>25.63</td><td>0.7981</td><td>0.1398</td><td>162.66</td><td>-6.41e-03</td><td>2.58e-02</td><td>4.42</td><td>9.82</td></tr><tr><td>DAPS DDNM</td><td>100 100</td><td>24.07 25.73</td><td>0.6859 0.7758</td><td>0.1863 0.1299</td><td>190.39 109.80</td><td>-6.06e-03 -6.68e-03</td><td>3.99e-02 7.60e-03</td><td>4.68 3.16</td><td>11.19 7.21</td></tr></table>

Quantitative comparison and sampling efficiency. EPS attains the best or second-best score on every metric and task in Tables 1 and 6 , across pointwise fidelity (PSNR, SSIM), perceptual quality (LPIPS, FID), and distributional calibration (CRPS, MMD). The closest competitor is consistently Palette, which shares the EPS pipeline except that the network input is $x _ { t }$ rather than $\mu _ { \star }$ ; holding the backbone, compute budget, and training data fixed, EPS still outperforms Palette on every task, isolating the contribution of the shifted pivot. Sampling-based baselines that approximate the measurement-matching score (ΠGDM, DPS, DDNM, DAPS, MPGD) lag further behind, even at substantially larger NFE budgets (ΠGDM, DDNM, DAPS, MPGD at 100 NFE; DPS at 250), and the gap widens where the operator is most ill-conditioned, namely random inpainting with 70% missing pixels and 4× super-resolution. Figure 3 confirms the same trend across budgets: EPS plateaus within ∼ 20 NFE on every panel, while every baseline keeps improving out to 100 NFE without reaching the EPS asymptote. The 1-NFE row is the strongest pointwise estimator on PSNR and SSIM (e.g., 26.60 PSNR on ImageNet random inpainting against 24.34 at 100 NFE), consistent with Observation 4, and trades distributional calibration for that pointwise sharpness in line with the perception-distortion trade-off [48].

Qualitative comparison and ablations. Figure 2 shows reconstructions on the five tasks. The largest visual gaps appear under aggressive inpainting and deblurring, where DPS, DAPS, and MPGD oversmooth or introduce texture inconsistent with the measurement, and DDNM and ΠGDM match measured directions but leave the unmeasured subspace blurry. EPS preserves sharp prior structure while matching the observation, since the pivot $\mu _ { \star }$ explicitly separates measured and unmeasured directions and $\bar { \Sigma } _ { \star } ( t )$ specifies how much to denoise along each. Appendix D studies the input pivot, zero-shot behavior, warm-start convergence, sampling-step sensitivity, amortization across tasks, and 256×256 scaling, and confirms the two central mechanisms: the shifted pivot is the right input, and preserving pretrained denoising marginals explains the fast convergence.

## 5 Related Work

Posterior sampling with pretrained generative priors. A large literature uses pretrained diffusion or flow models as priors for inverse problems. Explicit guidance methods, including Score-SDE/ALD, RePaint, DDNM, DDRM, DPS, ΠGDM, DAPS, MPGD, and FlowDPS [49, 50, 19, 21, 18, 20, 22, 23, 51, 28], approximate the measurement-matching score $\nabla _ { x _ { t } } \log { p ( y | x _ { t } ) } \approx - L _ { t } M _ { t } / G _ { t } \left[ 2 8 \right]$ , where $M _ { t }$ is a measurement residual, $L _ { t }$ lifts it back to the sample space, and $G _ { t }$ controls the guidance strength. They differ in the form of $M _ { t } , L _ { t } .$ and $G _ { t }$ , instantiating the template via projections, denoised estimates, Jacobian corrections, or moment approximations of $p ( x _ { 0 } | x _ { t } )$ . Moment-matching variants [26, 27] go beyond first-order Tweedie by approximating $p ( x _ { 0 } | x _ { t } )$ with an anisotropic Gaussian. Section 3.3 makes precise the gap between all of these and the exact posterior score, namely that each method substitutes the unconditional denoising query for the posterior denoising query, querying the network at $x _ { t }$ rather than at the pivot $\mu _ { \star } . \mathrm { G L A S S } \left[ 5 2 \right]$ is the closest training-free relative, and its equivalent-time formula coincides with EPS in the special case where $A ^ { \top } \bar { A }$ is a scalar multiple of the identity, equivalently when $\Sigma _ { \star } ( t )$ is isotropic (Appendix A.6). EPS handles the general operator-dependent anisotropic case at the cost of a training step.

Conditional training and restoration bridges. Palette and conditional image-to-image diffusion models train a network directly on $( x _ { t } , y )$ pairs [32, 33, 37, 34], and bridge-based restoration methods such as InDI [35] and image-to-image Schrödinger bridges [36] construct trajectories from the measurement distribution to the data distribution. These methods are expressive and avoid handdesigned corrections, but they expose the network to a conditional path whose intermediate marginals do not match the unconditional denoising marginals of the prior, so they cannot leverage a pretrained denoiser as a warm start in a structurally aligned way. EPS instead derives the conditional path induced by the exact linear-Gaussian posterior kernel, which preserves the input/output type of standard denoising pretraining (Section 3.4). This makes both random initialization and warmstarting from a pretrained checkpoint natural training options. We view bridge-based methods as complementary, since they define useful restoration dynamics for general degradations, while EPS identifies the specific bridge that solves the linear-Gaussian posterior exactly.

## 6 Conclusion

We derived the exact posterior score for linear Gaussian inverse problems and showed that posterior sampling reduces to a denoising problem at a measurement-induced pivot $\mu _ { \star }$ under an operatordependent anisotropic covariance $\Sigma _ { \star }$ . We turned this identity into EPS, a denoising training objective whose input/output structure matches standard pretraining, and which can therefore be either trained from scratch or fine-tuned efficiently from a pretrained checkpoint. EPS samples with the underlying backbone’s sampler, requires no measurement gradients or projections, and admits a one-evaluation posterior-mean estimator in the high-noise limit. Empirically, EPS improves both reconstruction fidelity and distributional calibration over sampling-based and conditional-training baselines on five linear inverse problems on FFHQ and ImageNet, while exposing an explicit sampling-budget trade-off through the same sampler used by the backbone.

Limitations. The exact derivation assumes a linear forward operator and Gaussian observation noise. Nonlinear operators can be approached by local linearization or by training against the true likelihood, but the closed form of Theorem 1 no longer applies directly. Pixel-space inverse problems with latent diffusion backbones also require care because the decoder makes a linear pixel-space operator nonlinear in latent space.

## Acknowledgments and Disclosure of Funding

AM is supported by the Clarendon Fund Scholarship, University of Oxford. We thank fal for the compute grants that supported this research. This work was supported in part by the Engineering and Physical Sciences Research Council (EPSRC) through the AI Hub in Generative Models [grant number EP/Y028805/1]. The authors acknowledge the use of resources provided by the Isambard-AI National AI Research Resource (AIRR) [53]. Isambard-AI is operated by the University of Bristol and is funded by the UK Government’s Department for Science, Innovation and Technology (DSIT) via UK Research and Innovation; and the Science and Technology Facilities Council [ST/AIRR/I-A-I/1023].

## References

[1] David L Donoho. Compressed sensing. IEEE Transactions on information theory, 52(4): 1289–1306, 2006.

[2] Emmanuel J Candès, Justin Romberg, and Terence Tao. Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information. IEEE Transactions on information theory, 52(2):489–509, 2006.

[3] Michael Lustig, David Donoho, and John M Pauly. Sparse mri: The application of compressed sensing for rapid mr imaging. Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine, 58(6):1182–1195, 2007.

[4] Florian Knoll, Jure Zbontar, Anuroop Sriram, Matthew J Muckley, Mary Bruno, Aaron Defazio, Marc Parente, Krzysztof J Geras, Joe Katsnelson, Hersh Chandarana, et al. fastmri: A publicly available raw k-space and dicom dataset of knee images for accelerated mr image reconstruction using machine learning. Radiology: Artificial Intelligence, 2(1):e190007, 2020.

[5] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Learning a deep convolutional network for image super-resolution. In European conference on computer vision, pages 184–199. Springer, 2014.

[6] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1905–1914, 2021.

[7] Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3883–3891, 2017.

[8] Orest Kupyn, Volodymyr Budzan, Mykola Mykhailych, Dmytro Mishkin, and Jiˇrí Matas. Deblurgan: Blind motion deblurring using conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8183–8192, 2018.

[9] Marcelo Bertalmio, Guillermo Sapiro, Vincent Caselles, and Coloma Ballester. Image inpainting. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pages 417–424, 2000.

[10] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2536–2544, 2016.

[11] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. arXiv:1503.03585, 2015.

[12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in neural information processing systems, volume 33, pages 6840–6851, 2020.

[13] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv:2011.13456, 2020.

[14] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[15] Michael Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. Journal of Machine Learning Research, 26(209): 1–80, 2025.

[16] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

[17] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. arXiv:2206.00364, 2022.

[18] Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022.

[19] Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. arXiv preprint arXiv:2212.00490, 2022.

[20] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=9\_gsMA8MRKQ.

[21] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. Advances in neural information processing systems, 35:23593–23606, 2022.

[22] Bingliang Zhang, Wenda Chu, Julius Berner, Chenlin Meng, Anima Anandkumar, and Yang Song. Improving diffusion inverse problem solving with decoupled noise annealing. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 20895–20905, 2025.

[23] Yutong He, Naoki Murata, Chieh-Hsin Lai, Yuhta Takida, Toshimitsu Uesaka, Dongjun Kim, Wei-Hsiang Liao, Yuki Mitsufuji, J. Zico Kolter, Ruslan Salakhutdinov, and Stefano Ermon. Manifold preserving guided diffusion, 2023. URL https://arxiv.org/abs/2311.16424.

[24] Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. Advances in Neural Information Processing Systems, 35:25683–25696, 2022.

[25] Morteza Mardani, Jiaming Song, Jan Kautz, and Arash Vahdat. A variational perspective on solving inverse problems with diffusion models. arXiv preprint arXiv:2305.04391, 2023.

[26] François Rozet, Gérôme Andry, François Lanusse, and Gilles Louppe. Learning diffusion priors from observations by expectation maximization. Advances in Neural Information Processing Systems, 37:87647–87682, 2024.

[27] Litu Rout, Yujia Chen, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Beyond first-order tweedie: Solving inverse problems using latent diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9472–9481, 2024.

[28] Giannis Daras, Hyungjin Chung, Chieh-Hsin Lai, Yuki Mitsufuji, Jong Chul Ye, Peyman Milanfar, Alexandros G Dimakis, and Mauricio Delbracio. A survey on diffusion models for inverse problems. arXiv preprint arXiv:2410.00083, 2024.

[29] Luhuan Wu, Brian Trippe, Christian Naesseth, David Blei, and John P Cunningham. Practical and asymptotically exact conditional sampling in diffusion models. Advances in Neural Information Processing Systems, 36:31372–31403, 2023.

[30] Gabriel Cardoso, Yazid Janati El Idrissi, Sylvain Le Corff, and Eric Moulines. Monte carlo guided diffusion for bayesian linear inverse problems. arXiv preprint arXiv:2308.07983, 2023.

[31] Zehao Dou and Yang Song. Diffusion posterior sampling for linear inverse problem solving: A filtering perspective. In The Twelfth International Conference on Learning Representations, 2024.

[32] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10, 2022.

[33] Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Schönlieb, and Christian Etmann. Conditional image generation with score-based diffusion models. arXiv preprint arXiv:2111.13606, 2021.

[34] Noam Elata, Hyungjin Chung, Jong Chul Ye, Tomer Michaeli, and Michael Elad. Invfusion: Bridging supervised and zero-shot diffusion for inverse problems. arXiv preprint arXiv:2504.01689, 2025.

[35] Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration. arXiv preprint arXiv:2303.11435, 2023.

[36] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A Theodorou, Weili Nie, and Anima Anandkumar. I<sup>2</sup>SB: Image-to-image Schrödinger bridge. arXiv preprint arXiv:2302.05872, 2023.

[37] Abbas Mammadov, Hyungjin Chung, and Jong Chul Ye. Amortized posterior sampling with diffusion prior distillation. arXiv preprint arXiv:2407.17907, 2024.

[38] Berthy T Feng, Jamie Smith, Michael Rubinstein, Huiwen Chang, Katherine L Bouman, and William T Freeman. Score-based diffusion models as principled priors for inverse imaging. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10520–10531, 2023.

[39] Herbert E Robbins. An empirical bayes approach to statistics. In Breakthroughs in Statistics: Foundations and basic theory, pages 388–394. Springer, 1992.

[40] Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106(496):1602–1614, 2011.

[41] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612, 2004.

[42] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

[43] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in neural information processing systems, volume 30, 2017.

[44] Tilmann Gneiting and Matthias Katzfuss. Probabilistic forecasting. Annual Review of Statistics and Its Application, 1:125–151, 2014.

[45] Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. The journal of machine learning research, 13(1):723–773, 2012.

[46] Abbas Mammadov, So Takao, Bohan Chen, Ricardo Baptista, Morteza Mardani, Yee Whye Teh, and Julius Berner. Variational flow maps: Make some noise for one-step conditional generation. arXiv preprint arXiv:2603.07276, 2026.

[47] Valentin De Bortoli, Alexandre Galashov, J. Swaroop Guntupalli, Guangyao Zhou, Kevin Murphy, Arthur Gretton, and Arnaud Doucet. Distributional diffusion models with scoring rules, 2025. URL https://arxiv.org/abs/2502.02483.

[48] Yochai Blau and Tomer Michaeli. The perception-distortion tradeoff. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6228–6237, 2018.

[49] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.

[50] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11461–11471, 2022.

[51] Jeongsol Kim, Bryan Sangwoo Kim, and Jong Chul Ye. FlowDPS: Flow-driven posterior sampling for inverse problems. arXiv preprint arXiv:2503.08136, 2025.

[52] Peter Holderrieth, Uriel Singer, Tommi Jaakkola, Ricky T. Q. Chen, Yaron Lipman, and Brian Karrer. Glass flows: Transition sampling for alignment of flow and diffusion models, 2025. URL https://arxiv.org/abs/2509.25170.

[53] Simon McIntosh-Smith, Sadaf R Alam, and Christopher Woods. Isambard-ai: a leadership class supercomputer optimised specifically for artificial intelligence, 2024. URL https: //arxiv.org/abs/2410.11199.

[54] Christopher KI Williams and Carl Edward Rasmussen. Gaussian processes for machine learning, volume 2. MIT Press Cambridge, MA, 2006.

## Appendix Contents

A Theory and Derivations 16   
A.1 Anisotropic Tweedie Formula . 16   
A.2 Proof of Theorem 1 16   
A.3 Proof of Proposition 2 17   
A.4 Proof of Proposition 3 . 18   
A.5 Proof of Observation 4 18   
A.6 Equivalent-Time and GLASS Limit 19   
A.7 Connection to ridge regression and Gaussian processes. 19   
B Implementation Details 20   
C Broader Impact 21   
D Additional Experiments 22   
D.1 Input Configuration Ablation . 22   
D.2 Zero-Shot Pivoting 24   
D.3 Palette vs EPS 25   
D.4 Sampling Efficiency 26   
D.5 Amortized Variant Across All Five Tasks . . 29   
D.6 One-Step Posterior Mean Check 30   
D.7 Additional 64×64 Results 31   
D.8 Extreme Tasks 64×64 . 34   
D.9 OOD Mask-Density Experiments . 36   
D.10 Additional 256×256 Results 38   
D.11 Palette vs. EPS at Matched NFE 40   
D.12 Runtime Analysis . 42   
D.13 Posterior diversity. 43   
D.14 Sampling Budget 45   
E Baseline Configurations 46   
F Metric Definitions 47

## A Theory and Derivations

This appendix gives the full derivation of the EPS score identity, the posterior velocity, the equivalenttime special case, and the one-step posterior-mean limit. We keep the assumptions of the main text: $x _ { t } = \alpha _ { t } x _ { 0 } + \beta _ { t } \epsilon$ with $\epsilon \sim \mathcal { N } ( 0 , \bar { I } _ { d } )$ and $y = A x _ { 0 } + \eta$ with $\eta \sim \dot { \mathcal { N } } ( 0 , \sigma _ { y } ^ { 2 } I _ { m } )$

## A.1 Anisotropic Tweedie Formula

For any positive definite $\Sigma ,$ define

$$
p _ { \mathrm { d a t a } } ^ { \Sigma } ( \mu ) = \int { \mathcal N } ( \mu ; x _ { 0 } , \Sigma ) p _ { \mathrm { d a t a } } ( x _ { 0 } ) \mathrm { d } x _ { 0 } , \qquad D _ { \Sigma } ( \mu ) = { \mathbb E } [ x _ { 0 } | x _ { 0 } + \xi = \mu ] , \quad \xi \sim { \mathcal N } ( 0 , \Sigma ) .\tag{19}
$$

Then

$$
D _ { \Sigma } ( \mu ) = \mu + \Sigma \nabla _ { \mu } \log p _ { \mathrm { d a t a } } ^ { \Sigma } ( \mu ) .\tag{20}
$$

Indeed, differentiating under the integral gives

$$
\nabla _ { \mu } p _ { \mathrm { d a t a } } ^ { \Sigma } ( \mu ) = \int \nabla _ { \mu } \mathcal { N } ( \mu ; x _ { 0 } , \Sigma ) p _ { \mathrm { d a t a } } ( x _ { 0 } ) \mathrm { d } x _ { 0 }\tag{21}
$$

$$
= \int \mathcal { N } ( \mu ; x _ { 0 } , \Sigma ) \Sigma ^ { - 1 } ( x _ { 0 } - \mu ) p _ { \mathrm { d a t a } } ( x _ { 0 } ) \mathrm { d } x _ { 0 } .\tag{22}
$$

Dividing by $p _ { \mathrm { d a t a } } ^ { \Sigma } ( \mu )$ yields

$$
\begin{array} { r } { \nabla _ { \mu } \log p _ { \mathrm { d a t a } } ^ { \Sigma } ( \mu ) = \Sigma ^ { - 1 } \left( \mathbb { E } [ x _ { 0 } | x _ { 0 } + \xi = \mu ] - \mu \right) , } \end{array}\tag{23}
$$

which proves (20). The key point for EPS is that Σ is a full covariance matrix fixed by the inverse problem, not a scalar noise level chosen to match an unconditional diffusion time.

## A.2 Proof of Theorem 1

Theorem 1 restated. Under the linear Gaussian inverse problem (5) and the interpolant (1), the posterior score at time t is

$$
\nabla _ { x _ { t } } \log { p ( x _ { t } | y ) } = \frac { 1 } { \beta _ { t } ^ { 2 } } \Big ( \alpha _ { t } D _ { \Sigma _ { \star } ( t ) } \big ( \mu _ { \star } ( x _ { t } , y , t ) \big ) - x _ { t } \Big ) ,\tag{24}
$$

where

$$
\Sigma _ { \star } ( t ) = \left( { \frac { \alpha _ { t } ^ { 2 } } { \beta _ { t } ^ { 2 } } } I _ { d } + { \frac { 1 } { \sigma _ { y } ^ { 2 } } } A ^ { \top } A \right) ^ { - 1 } , \qquad \mu _ { \star } ( x _ { t } , y , t ) = \Sigma _ { \star } ( t ) \left( { \frac { \alpha _ { t } } { \beta _ { t } ^ { 2 } } } x _ { t } + { \frac { 1 } { \sigma _ { y } ^ { 2 } } } A ^ { \top } y \right) .\tag{25}
$$

Equivalently, $D _ { \Sigma _ { \star } ( t ) } ( \mu _ { \star } ( x _ { t } , y , t ) ) = \mathbb { E } [ x _ { 0 } | x _ { t } , y ] .$

Proof. By conditional independence of $x _ { t }$ and y given $x _ { 0 }$ ,

$$
p ( x _ { t } | y ) = \frac { 1 } { p ( y ) } \int p ( x _ { t } | x _ { 0 } ) p ( y | x _ { 0 } ) p _ { \mathrm { d a t a } } ( x _ { 0 } ) \mathrm { d } x _ { 0 } .\tag{26}
$$

The two Gaussian factors are

$$
p ( x _ { t } | \boldsymbol { x } _ { 0 } ) = ( 2 \pi \beta _ { t } ^ { 2 } ) ^ { - d / 2 } \exp \left( - \frac { \| \boldsymbol { x } _ { t } - \boldsymbol { \alpha } _ { t } \boldsymbol { x } _ { 0 } \| ^ { 2 } } { 2 \beta _ { t } ^ { 2 } } \right) ,\tag{27}
$$

$$
p ( y | x _ { 0 } ) = ( 2 \pi \sigma _ { y } ^ { 2 } ) ^ { - m / 2 } \exp \left( - \frac { \| y - A x _ { 0 } \| ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } \right) .\tag{28}
$$

Collect the exponent terms that depend on $x _ { 0 } \colon$

$$
\begin{array} { r l } & { - \frac { \| x _ { t } - \alpha _ { t } x _ { 0 } \| ^ { 2 } } { 2 \beta _ { t } ^ { 2 } } - \frac { \| y - A x _ { 0 } \| ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } } \\ & { \quad = - \frac { 1 } { 2 } x _ { 0 } ^ { \top } \underbrace { \left( \frac { \alpha _ { t } ^ { 2 } } { \beta _ { t } ^ { 2 } } I _ { d } + \frac { 1 } { \sigma _ { y } ^ { 2 } } A ^ { \top } A \right) } _ { \Lambda _ { t } } x _ { 0 } + \underbrace { \left( \frac { \alpha _ { t } } { \beta _ { t } ^ { 2 } } x _ { t } + \frac { 1 } { \sigma _ { y } ^ { 2 } } A ^ { \top } y \right) } _ { b _ { t } ^ { \top } } x _ { 0 } - \frac { \| x _ { t } \| ^ { 2 } } { 2 \beta _ { t } ^ { 2 } } - \frac { \| y \| ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } . } \end{array}\tag{29}
$$

(30)

Because $\beta _ { t } > 0 ,$ the matrix $\Lambda _ { t }$ is positive definite even when A is rank deficient. Let $\Sigma _ { \star } = \Lambda _ { t } ^ { - 1 }$ and $\mu _ { \star } = \Sigma _ { \star } b _ { t }$ . Completing the square in (30),

$$
- \frac { 1 } { 2 } x _ { 0 } ^ { \top } \Lambda _ { t } x _ { 0 } + b _ { t } ^ { \top } x _ { 0 } = - \frac { 1 } { 2 } ( x _ { 0 } - \mu _ { \star } ) ^ { \top } \Sigma _ { \star } ^ { - 1 } ( x _ { 0 } - \mu _ { \star } ) + \frac { 1 } { 2 } b _ { t } ^ { \top } \Sigma _ { \star } b _ { t } .\tag{31}
$$

Therefore the product of the two likelihoods can be factored as

$$
p ( x _ { t } | x _ { 0 } ) p ( y | x _ { 0 } ) = C _ { t } ( x _ { t } , y ) \mathcal { N } ( x _ { 0 } ; \mu _ { \star } , \Sigma _ { \star } ) ,\tag{32}
$$

where all dependence on $x _ { 0 }$ is contained in the displayed Gaussian and

$$
\log C _ { t } ( x _ { t } , y ) = \mathrm { c o n s t } ( t ) - \frac { \Vert x _ { t } \Vert ^ { 2 } } { 2 \beta _ { t } ^ { 2 } } - \frac { \Vert y \Vert ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } + \frac { 1 } { 2 } b _ { t } ^ { \top } \Sigma _ { \star } b _ { t } .\tag{33}
$$

Substituting (32) into (26) gives

$$
p ( x _ { t } | y ) = \frac { C _ { t } ( x _ { t } , y ) } { p ( y ) } p _ { \mathrm { d a t a } } ^ { \Sigma _ { \star } } ( \mu _ { \star } ) , \qquad p _ { \mathrm { d a t a } } ^ { \Sigma _ { \star } } ( \mu _ { \star } ) = \int \mathcal { N } ( x _ { 0 } ; \mu _ { \star } , \Sigma _ { \star } ) p _ { \mathrm { d a t a } } ( x _ { 0 } ) \mathrm { d } x _ { 0 } .\tag{34}
$$

Now differentiate (34) with respect to $x _ { t }$ . Since $\Sigma _ { \star }$ does not depend on $x _ { t }$ and $\partial b _ { t } / \partial x _ { t } = ( \alpha _ { t } / \beta _ { t } ^ { 2 } ) I _ { d }$

$$
\frac { \partial \mu _ { \star } } { \partial x _ { t } } = \frac { \alpha _ { t } } { \beta _ { t } ^ { 2 } } \Sigma _ { \star } .\tag{35}
$$

The normalizer derivative is

$$
\nabla _ { x _ { t } } \log C _ { t } ( x _ { t } , y ) = - \frac { x _ { t } } { \beta _ { t } ^ { 2 } } + \frac { 1 } { 2 } \nabla _ { x _ { t } } \left( b _ { t } ^ { \top } \Sigma _ { \star } b _ { t } \right)\tag{36}
$$

$$
= - \frac { x _ { t } } { \beta _ { t } ^ { 2 } } + \frac { \alpha _ { t } } { \beta _ { t } ^ { 2 } } \Sigma _ { \star } b _ { t } = - \frac { x _ { t } } { \beta _ { t } ^ { 2 } } + \frac { \alpha _ { t } } { \beta _ { t } ^ { 2 } } \mu _ { \star } .\tag{37}
$$

The smoothed-density derivative is, by the chain rule and (35),

$$
\nabla _ { x _ { t } } \log { p _ { \mathrm { d a t a } } ^ { \Sigma _ { \star } } ( \mu _ { \star } ) } = \frac { \alpha _ { t } } { \beta _ { t } ^ { 2 } } \Sigma _ { \star } \nabla _ { \mu } \log { p _ { \mathrm { d a t a } } ^ { \Sigma _ { \star } } ( \mu ) } \big | _ { \mu = \mu _ { \star } } .\tag{38}
$$

Using the anisotropic Tweedie identity (20),

$$
\begin{array} { r } { \Sigma _ { \star } \nabla _ { \mu } \log p _ { \mathrm { d a t a } } ^ { \Sigma _ { \star } } ( \mu _ { \star } ) = D _ { \Sigma _ { \star } } ( \mu _ { \star } ) - \mu _ { \star } . } \end{array}\tag{39}
$$

Combining (37), (38), and (39), the pivot terms cancel:

$$
\nabla _ { x _ { t } } \log { p ( x _ { t } | y ) } = - \frac { x _ { t } } { \beta _ { t } ^ { 2 } } + \frac { \alpha _ { t } } { \beta _ { t } ^ { 2 } } \mu _ { \star } + \frac { \alpha _ { t } } { \beta _ { t } ^ { 2 } } ( D _ { \Sigma _ { \star } } ( \mu _ { \star } ) - \mu _ { \star } )\tag{40}
$$

$$
= \frac { 1 } { \beta _ { t } ^ { 2 } } \left( \alpha _ { t } D _ { \Sigma _ { \star } } ( \mu _ { \star } ) - x _ { t } \right) .\tag{41}
$$

Finally, under the joint density proportional to $\mathcal N ( x _ { 0 } ; \mu _ { \star } , \Sigma _ { \star } ) p _ { \mathrm { d a t a } } ( x _ { 0 } )$ , the posterior mean of $x _ { 0 }$ is exactly $D _ { \Sigma _ { \star } } ( \mu _ { \star } )$ , so $D _ { \Sigma _ { \star } ( t ) } ( \mu _ { \star } ( x _ { t } , y , t ) ) = \mathbb { E } [ x _ { 0 } | x _ { t } , y ]$ □

## A.3 Proof of Proposition 2

For brevity in this proof we write $D ^ { \star } ( x _ { t } , y , t ) : = D _ { \Sigma _ { \star } ( t ) } ( \mu _ { \star } ( x _ { t } , y , t ) )$ , which by Theorem 1 equals $\mathbb { E } [ x _ { 0 } | x _ { t } , y ]$ . The interpolant satisfies $x _ { t } = \alpha _ { t } x _ { 0 } + \beta _ { t } \epsilon .$ , hence conditional on $( x _ { t } , y )$ ，

$$
\mathbb { E } [ \epsilon | x _ { t } , y ] = \frac { x _ { t } - \alpha _ { t } \mathbb { E } [ x _ { 0 } | x _ { t } , y ] } { \beta _ { t } } = \frac { x _ { t } - \alpha _ { t } D ^ { \star } ( x _ { t } , y , t ) } { \beta _ { t } } .\tag{42}
$$

The posterior velocity is the conditional expectation of the path derivative:

$$
v _ { t } ^ { y } ( x _ { t } ) = \mathbb { E } [ \dot { \alpha } _ { t } x _ { 0 } + \dot { \beta } _ { t } \epsilon | x _ { t } , y ]\tag{43}
$$

$$
= \dot { \alpha } _ { t } D ^ { \star } ( x _ { t } , y , t ) + \dot { \beta } _ { t } \frac { x _ { t } - \alpha _ { t } D ^ { \star } ( x _ { t } , y , t ) } { \beta _ { t } }\tag{44}
$$

$$
= \frac { \dot { \beta } _ { t } } { \beta _ { t } } x _ { t } + \left( \dot { \alpha } _ { t } - \frac { \alpha _ { t } \dot { \beta } _ { t } } { \beta _ { t } } \right) D ^ { \star } ( x _ { t } , y , t ) .\tag{45}
$$

Thus estimating the exact posterior denoiser is equivalent to estimating the exact posterior velocity for the interpolant.

## A.4 Proof of Proposition 3

Let

$$
\Lambda _ { t } = \frac { \alpha _ { t } ^ { 2 } } { \beta _ { t } ^ { 2 } } I _ { d } + \frac { 1 } { \sigma _ { y } ^ { 2 } } A ^ { \top } A , \quad \quad \Sigma _ { \star } ( t ) = \Lambda _ { t } ^ { - 1 } .\tag{46}
$$

Using $x _ { t } = \alpha _ { t } x _ { 0 } + \beta _ { t } \epsilon$ ϵ and $y = A x _ { 0 } + \sigma _ { y } r$ , the pivot is

$$
\boldsymbol { \mu } _ { \star } = \Sigma _ { \star } \left( \frac { \alpha _ { t } } { \beta _ { t } ^ { 2 } } x _ { t } + \frac { 1 } { \sigma _ { y } ^ { 2 } } \boldsymbol { A } ^ { \top } \boldsymbol { y } \right)\tag{47}
$$

$$
= \Sigma _ { \star } \left[ \left( \frac { \alpha _ { t } ^ { 2 } } { \beta _ { t } ^ { 2 } } I _ { d } + \frac { 1 } { \sigma _ { y } ^ { 2 } } A ^ { \top } A \right) x _ { 0 } + \frac { \alpha _ { t } } { \beta _ { t } } \epsilon + \frac { 1 } { \sigma _ { y } } A ^ { \top } \eta \right]\tag{48}
$$

$$
= x _ { 0 } + \Sigma _ { \star } \left( \frac { \alpha _ { t } } { \beta _ { t } } \epsilon + \frac { 1 } { \sigma _ { y } } A ^ { \top } \eta \right) .\tag{49}
$$

The noise term is Gaussian with mean zero and covariance

$$
\Sigma _ { \star } \left( \frac { \alpha _ { t } ^ { 2 } } { \beta _ { t } ^ { 2 } } I _ { d } + \frac { 1 } { \sigma _ { y } ^ { 2 } } A ^ { \top } A \right) \Sigma _ { \star } = \Sigma _ { \star } \Lambda _ { t } \Sigma _ { \star } = \Sigma _ { \star } .\tag{50}
$$

Therefore $\mu _ { \star } | x _ { 0 } , t \sim \mathcal { N } ( x _ { 0 } , \Sigma _ { \star } ( t ) )$ , as claimed.

## A.5 Proof of Observation 4

For EDM, $\alpha _ { t } = 1$ and $\beta _ { t } = \sigma _ { t }$ . Let

$$
A = U _ { r } S V _ { r } ^ { \top }\tag{51}
$$

be the compact SVD of A, where ${ \cal S } = \mathrm { d i a g } ( s _ { 1 } , \ldots , s _ { r } )$ contains the positive singular values. Let $V _ { 0 }$ be an orthonormal basis for ${ \mathcal { N } } ( A )$ . Then

$$
P _ { \mathcal { N } ( A ) } = V _ { 0 } V _ { 0 } ^ { \top } , \qquad A ^ { \dag } = V _ { r } S ^ { - 1 } U _ { r } ^ { \top } .\tag{52}
$$

With $\lambda = \sigma _ { y } ^ { 2 } / \sigma _ { t } ^ { 2 }$ , the pivot can be written as

$$
\mu _ { \star } = V _ { r } ( S ^ { 2 } + \lambda I ) ^ { - 1 } \left( S U _ { r } ^ { \top } y + \lambda V _ { r } ^ { \top } x _ { t } \right) + V _ { 0 } V _ { 0 } ^ { \top } x _ { t } .\tag{53}
$$

Taking $\lambda  0$ gives

$$
\mu _ { \star } \longrightarrow V _ { r } S ^ { - 1 } U _ { r } ^ { \top } y + V _ { 0 } V _ { 0 } ^ { \top } x _ { t } = A ^ { \dagger } y + P _ { \mathcal { N } ( A ) } x _ { t } .\tag{54}
$$

It remains to show that the corresponding anisotropic denoiser converges to the posterior mean $\mathbb { E } [ x _ { 0 } | y ]$ . In the same SVD coordinates, the covariance is

$$
\Sigma _ { \star } ( t ) = \left( \frac { 1 } { \sigma _ { t } ^ { 2 } } I + \frac { 1 } { \sigma _ { y } ^ { 2 } } A ^ { \top } A \right) ^ { - 1 }\tag{55}
$$

$$
= \sigma _ { y } ^ { 2 } V _ { r } ( S ^ { 2 } + \lambda I ) ^ { - 1 } V _ { r } ^ { \top } + \sigma _ { t } ^ { 2 } V _ { 0 } V _ { 0 } ^ { \top } .\tag{56}
$$

Thus, as $\sigma _ { t } \to \infty$ , the row-space covariance converges to

$$
\sigma _ { y } ^ { 2 } V _ { r } S ^ { - 2 } V _ { r } ^ { \top } = \sigma _ { y } ^ { 2 } ( A ^ { \top } A ) ^ { \dagger } ,\tag{57}
$$

while the nullspace variance diverges. Therefore the limiting denoising query contains finite information only in the row space of A and no information in the nullspace.

More explicitly, the limiting finite-noise observation is

$$
z = A ^ { \dagger } y = A ^ { \dagger } A x _ { 0 } + A ^ { \dagger } \eta = P _ { { \mathcal R } ( A ^ { \top } ) } x _ { 0 } + \zeta , \qquad \zeta \sim { \mathcal N } \big ( 0 , \sigma _ { y } ^ { 2 } ( A ^ { \top } A ) ^ { \dagger } \big ) .\tag{58}
$$

Since $A ^ { \dagger }$ is a deterministic function of $y ,$ conditioning on y implies conditioning on $z .$ Conversely, for Gaussian measurement noise with known covariance, $z = A ^ { \dagger } y$ is a sufficient statistic for $y$ with respect to $x _ { 0 } \colon$ the remaining component of $y$ orthogonal to $\mathcal { R } ( A )$ carries no information about $x _ { 0 }$ Hence

$$
\mathbb { E } [ x _ { 0 } | z ] = \mathbb { E } [ x _ { 0 } | y ] .\tag{59}
$$

The limiting anisotropic denoiser is exactly the Bayes estimator associated with this row-space observation and infinite nullspace uncertainty. Therefore

$$
\operatorname* { l i m } _ { \sigma _ { t } \to \infty } D _ { \Sigma _ { \star } ( t ) } \bigl ( \mu _ { \star } ( x _ { t } , y , t ) \bigr ) = \mathbb { E } [ x _ { 0 } | y ] .
$$

This proves Observation 4.

(60)

## A.6 Equivalent-Time and GLASS Limit

When $A ^ { \top } A = \gamma ^ { 2 } I _ { d }$ , the covariance in (12) is isotropic:

$$
\Sigma _ { \star } ( t ) = \sigma _ { \star } ^ { 2 } ( t ) I _ { d } , \qquad \sigma _ { \star } ^ { 2 } ( t ) = \frac { \beta _ { t } ^ { 2 } \sigma _ { y } ^ { 2 } } { \alpha _ { t } ^ { 2 } \sigma _ { y } ^ { 2 } + \beta _ { t } ^ { 2 } \gamma ^ { 2 } } .\tag{61}
$$

In this special case the posterior denoising query can be represented by an equivalent scalar noise level. If the base denoiser is trained for isotropic corruptions with effective noise ratio $\beta _ { s } ^ { 2 } / \alpha _ { s } ^ { 2 }$ , we choose $s = t ^ { \star }$ such that

$$
\frac { \beta _ { t ^ { \star } } ^ { 2 } } { \alpha _ { t ^ { \star } } ^ { 2 } } = \sigma _ { \star } ^ { 2 } ( t ) .\tag{62}
$$

Then $D _ { \Sigma _ { \star } ( t ) } ( \mu _ { \star } )$ can be approximated by the pretrained isotropic denoiser at time $t ^ { \star }$ . This is the setting in which an equivalent-time, training-free reduction such as GLASS [52] is available. For a general inverse problem, however, $A ^ { \top } A$ has different eigenvalues and often a nontrivial nullspace; $\Sigma _ { \star }$ is then anisotropic and no single scalar time can represent the posterior denoising kernel. EPS fine-tuning is introduced precisely to learn this missing anisotropic denoising geometry.

## A.7 Connection to ridge regression and Gaussian processes.

Equation (12) is exactly the linear-Gaussian Bayesian update familiar from ridge regression and Gaussian process regression. Here $\sigma _ { u } ^ { - 2 } A ^ { \top } A$ plays the role of the data precision, $( \breve { \alpha _ { t } ^ { 2 } } / \beta _ { t } ^ { 2 } ) I _ { d }$ plays the role of the prior precision (set by the current diffusion noise level), and $\mu ,$ <sub>⋆</sub> is their precision-weighted mean. $\operatorname { A s } \ \sigma _ { t }$ grows the prior precision vanishes and $\mu _ { \star }$ converges to the data-only ridge solution, while as $\sigma _ { t }$ shrinks $\mu _ { \star }$ collapses onto $x _ { t }$ . The covariance $\Sigma _ { \star } ( t )$ is the corresponding posterior covariance, which shrinks along measured directions (large eigenvalues of $A ^ { \top } A )$ and remains diffuse along weakly observed directions, mirroring the heteroscedastic uncertainty of GP posteriors [54].

## B Implementation Details

Architecture. We use the EDM-ADM checkpoint [17] for ImageNet-64×64 (edm-imagenet-64x64-cond-adm.pkl, ∼296M parameters, class-conditional with 1000- way one-hot embedding), and an EDM-DDPM++ checkpoint we trained from scratch for FFHQ-64×64. EPS extends the first conv from 3 to 6 input channels: the mean pivot $\mu _ { \star }$ (3 channels) concatenated with a task-specific observation tensor (3 channels). The added input channels are zero-initialised so the network reproduces the unconditional pretrained mapping at step zero of fine-tuning. The observation tensor is the masked observation $y$ for inpainting, the nearest-neighbour upsampling of $y$ for super-resolution, and the blurred observation $y$ for deblurring; the total input width is therefore 6 channels for every task.

Pivot solver. For binary masks $\Sigma _ { \star }$ is diagonal and the pivot solve is element-wise. For 4× superresolution the structured solve uses average-pool / nearest-upsample primitives in $O ( d )$ . For circular blur kernels we diagonalize $A ^ { \top } A$ by the 2D FFT; the per-step solve is then a complex element-wise divide plus an inverse FFT, O(d log d). Empirically the pivot solve contributes <1 ms per step compared to a U-Net forward of ∼19 ms at batch 1.

Optimization. We use the EDM optimizer stack unchanged: Adam $( \beta _ { 1 } { = } 0 . 9 , \beta _ { 2 } { = } 0 . 9 9 9 , \varepsilon { = } 1 0 ^ { - 8 } ) ;$ log-normal noise sampling with $\stackrel { \cdot } { P _ { \mathrm { m e a n } } } = - 1 . 2 , P _ { \mathrm { s t d } } = 1 . 2 , \stackrel { \cdot } { \sigma } _ { \mathrm { d a t a } } = 0 . 5 ;$ ; schedule extrema $\sigma _ { \mathrm { m i n } } { = } 0 . 0 0 2$ $\sigma _ { \mathrm { m a x } } { = } 8 0 , \rho { = } 7$ . Learning rate is $1 0 ^ { - 4 }$ with $\mathrm { 1 2 } { \times } \mathrm { 1 0 ^ { 6 } } { \mathrm { - i m a g e } }$ linear warm-up. EMA uses a half-life of 500 kimg with a $5 \%$ ramp-up ratio. We weight the loss by the standard EDM weighting $\lambda ( \sigma ) =$ $( \sigma ^ { 2 } + \sigma _ { \mathrm { { d a t a } } } ^ { 2 } ) \breve { / } ( \sigma \sigma _ { \mathrm { { d a t a } } } ) ^ { 2 }$ . Per-task fine-tuning runs for 10 epochs (∼25k iterations) on ImageNet-64 at batch 128 across 4 NVIDIA B200 GPUs (gradient accumulation=1); FFHQ-64 trains for the same iteration budget at batch 192. End-to-end fine-tuning takes ∼24 h on ImageNet and ∼10 h on FFHQ per task.

Operator randomization during training. We re-sample the operator at every minibatch step. Random inpainting samples a per-pixel Bernoulli mask with mask-density $\sim \mathcal { U } ( 5 0 \% , 7 0 \% )$ (default training density 70%). Box inpainting samples a uniformly random rectangle whose side lengths are drawn from $\mathcal { U } ( [ H / 4 , H / 2 ] ) ^ { \cdot } \times \mathcal { U } ( [ \bar { W } / 4 , \dot { W } / 2 ] )$ ) with margins $H / 1 6$ at every edge. Motion-deblur kernels are generated with random length $\in \{ 7 , 9$ , 11, 13, 15} and random angle $\in [ 0 ,$ , 180<sup>◦</sup>]; Gaussiandeblur kernels use a fixed bandwidth $\sigma _ { \mathrm { b l u r } } { = } 0 . 7 5$ at length 11. Super-resolution applies a fixed 4× average-pool. Observation noise is fixed at $\sigma _ { y } { = } 0 . 0 5$ throughout both training and evaluation.

Sampling at inference. We use the deterministic EDM Euler ODE sampler for the 20- and 100- NFE variants (second\_order=False). The 1-NFE variant evaluates the denoiser once at the highest noise level $\sigma _ { \mathrm { m a x } }$ via the high-noise pivot of Observation 4; the resulting single forward pass returns the conditional MMSE estimator $\mathbb { E } [ \bar { x } _ { 0 } | y ]$ directly. Class labels at inference time use the ground-truth ImageNet-1k class for ImageNet-64 and are not used for FFHQ-64.

Reproducibility. All random seeds are fixed; the same 100 evaluation images and 10 posterior seeds per image are used for every method. The structured-solve, training-loop, and sampler implementations will be released as open-source upon acceptance, along with all per-task fine-tuned checkpoints.

## C Broader Impact

EPS provides a principled, calibrated approach to posterior sampling for linear inverse problems. Because it preserves the input/output structure of standard denoising pretraining, an existing pretrained prior can be repurposed into an uncertainty-aware posterior sampler with a lightweight fine-tune rather than a from-scratch retraining. This makes the method straightforward to adapt across scientific imaging applications where reliable reconstructions and quantified posterior uncertainty are valuable, and the closed-form pivot and covariance offer a transparent handle for analyzing the sampler in any downstream pipeline. Beyond these benefits, our work does not introduce societal impacts that go meaningfully beyond those of the existing generative diffusion priors it builds on; the standard considerations around dual-use of high-fidelity image generation and the demographic or domain biases of the underlying training data continue to apply.

## D Additional Experiments

We collect ablations and extended tables that support the main claims.

## D.1 Input Configuration Ablation

The central input ablation compares raw-state conditioning to shifted-pivot conditioning while keeping the backbone, compute budget, and EDM warm start fixed. We evaluate four input streams to the denoiser: $\mathbf { \rho } ( \mathbf { a } ) \left[ x _ { t } , y , t \right]$ (Palette-style), the standard conditional baseline that feeds the noised latent alongside the observation and exposes no closed-form posterior structure; (b) $[ \mu _ { \star } , t ]$ , which replaces $x _ { t }$ with the closed-form posterior mean $\mu _ { \star } ( x _ { t } , y , \sigma _ { t } )$ obtained by Gaussian-merging $p ( x _ { t } \mid x _ { 0 } )$ and $p ( y \mid x _ { 0 } )$ inside the integral, dropping y from the input; (c) $[ \mu _ { \star } , \dot { \Sigma } _ { \star } , t ]$ , which additionally passes the per-component posterior covariance $\dot { \Sigma } _ { \star }$ as a side channel, giving the network explicit access to the local anisotropic uncertainty; and (d) $[ \mu _ { \star } , y , t ]$ (EPS, ours), the full EPS input where the posterior mean is concatenated with the raw observation, allowing the network to use $y$ both directly and through the analytical pivot.

Table 2 reports this ablation on FFHQ-64 (DDPM++/EDM backbone) and ImageNet-64 (EDM-ADM backbone) at NFE=100 with the EDM Euler sampler. Both backbones are warm-started from the same pretrained checkpoint and fine-tuned under matched protocols. The progression Palette $\to \mu _ { \star } \to \mu _ { \star } { + } \Sigma _ { \star } { \to } \mathrm { E P S }$ is monotone on every distortion and distributional metric in the average and on most tasks individually, on both datasets. Replacing $x _ { t }$ by $\mu _ { \star }$ explains most of the gain, and the auxiliary observation channel provides an additional anchor.

Table 2: The shifted pivot $\mu _ { \star }$ is the right input. Input-configuration ablation on FFHQ-64 (top) and ImageNet-64 (bottom) at ${ \mathrm { N F E } } { = } 1 0 0$ with the EDM Euler sampler. The Palette → $\mu _ { \star } $ $\mu _ { \star } { + } \Sigma _ { \star } { \ \to } \mathrm { E P S }$ progression is monotone on every metric in the average and on most tasks individually, on both datasets. Best in bold, second-best underlined; the EPS row is highlighted in light pink.
<table><tr><td>Task</td><td>Input</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID↓</td><td>MMD-pix ↓</td><td>MMD-Inc ↓</td><td>CRPS-pix ↓</td><td>CRPS-Inc ↓</td></tr><tr><td colspan="10">FFHQ 64×64</td></tr><tr><td>Average</td><td> $\left[ x _ { t } , y , t \right] \left( \mathrm { P a l e t t e } \right)$   $[ \mu _ { \star } , t ]$   $[ \mu _ { \star } , \Sigma _ { \star } , t ]$  _  $\bar { [ } \mu _ { \star } , y , t ] \mathrm { ( E P S ) }$ </td><td>26.03 26.39 26.62 26.69</td><td>0.8590 0.8583 0.8636 0.8661</td><td>0.0626 0.0632 0.0603 0.0590</td><td>31.50 31.34 30.20 29.94</td><td>-6.6e-03 -6.7e-03 -6.7e-03 -6.7e-03</td><td>-4.7e-03 -4.6e-03 -4.8e-03 -4.9e-03</td><td>3.43 3.30 3.27 3.24</td><td>3.86 3.81 3.76 3.73</td></tr><tr><td>Random inpaint</td><td> $\left[ x _ { t } , y , t \right] \left( \mathrm { P a l e t t e } \right)$   $[ \mu _ { \star } , t ]$   $[ \mu _ { \star } , \dot { \Sigma } _ { \star } , t ]$   $\left[ \mu _ { \star } , y , t \right] \left( \mathrm { E P S } \right)$ </td><td>25.76 24.87 26.00 26.16</td><td>0.8809 0.8564 0.8845 0.8879</td><td>0.0593 0.0698 0.0546 0.0533</td><td>33.30 37.92 32.10 31.87</td><td>-6.7e-03 -6.7e-03 -6.7e-03 -6.7e-03</td><td>-4.8e-03 -3.9e-03 -5.0e-03 -5.0e-03</td><td>3.31 3.39 3.22 3.16</td><td>3.87 4.08 3.76 3.75</td></tr><tr><td>Box inpaint</td><td>[xt , y, t] (Palette)  $[ \mu _ { \star } , t ]$   $\ [ \mu _ { \star } , \dot { \Sigma } _ { \star } , t ]$  _  $\left[ \mu _ { \star } , y , t \right] \left( \mathrm { E P S } \right)$ </td><td>24.18 24.14 24.17 24.23</td><td>0.8426 0.8431 0.8430 0.8448</td><td>0.0577 0.0575 0.0574 0.0567</td><td>25.09 24.94 24.88</td><td>-6.6e-03 -6.6e-03 -6.6e-03</td><td>-5.3e-03 -5.4e-03 -5.4e-03</td><td>4.02 4.02 4.03</td><td>3.47 3.45 3.45 3.45</td></tr><tr><td>Super-res (4×)</td><td>[xt , y, t] (Palette)  $[ \mu _ { \star } , t ]$   $\ [ \mu _ { \star } , \dot { \Sigma } _ { \star } , t ]$  _  $\left[ \mu _ { \star } , y , t \right] \left( \mathrm { E P S } \right)$ </td><td>21.95 21.89 21.86 21.96</td><td>0.7220 0.7179 0.7162 0.7232</td><td>0.1273 0.1300 0.1308 0.1262</td><td>49.28 49.85 50.14 49.29</td><td>-6.3e-03 -6.3e-03 -6.2e-03 -6.3e-03</td><td>-3.0e-03 -2.8e-03 -2.7e-03</td><td>5.22 5.29 5.30 5.23</td><td>5.29 5.36 5.39 5.30</td></tr><tr><td>Gaussian deblur</td><td> $\left[ x _ { t } , y , t \right] \left( \mathrm { P a l e t t e } \right)$   $[ \mu _ { \star } , t ]$   $[ \mu _ { \star } , \dot { \Sigma } _ { \star } , t ]$   $\bar { [ } \mu _ { \star } , y , t ] \mathrm { ( E P S ) }$ </td><td>30.47 30.82 30.82 30.82</td><td>0.9397 0.9408 0.9408 0.9408</td><td>0.0286 0.0273 0.0273 0.0273</td><td>21.63 20.72 20.70 20.68</td><td>-6.9e-03 -6.9e-03 -6.9e-03 -6.9e-03</td><td>-5.6e-03 -5.7e-03 -5.7e-03</td><td>1.91 1.84 1.84</td><td>3.06 2.99 2.99 2.99</td></tr><tr><td>Motion deblur</td><td> $\left[ x _ { t } , y , t \right] \left( \mathrm { P a l e t t e } \right)$   $[ \mu _ { \star } , t ]$   $[ \mu _ { \star } , \Sigma _ { \star } , t ]$  _  $\left[ \mu _ { \star } , y , t \right] \left( \mathrm { E P S } \right)$ </td><td>27.79 30.23 30.25 30.27</td><td>0.9099 0.9334 0.9337 0.9339</td><td>0.0404 0.0314 0.0312 0.0311</td><td>28.23 23.24 23.16</td><td>-6.8e-03 -6.9e-03 -6.9e-03</td><td>-5.7e-03 -5.0e-03 -5.4e-03 -5.4e-03</td><td>1.84 2.67 1.94 1.94</td><td>3.59 3.18 3.18 3.18</td></tr><tr><td colspan="10">[xt, y, t] (Palette) 24.32</td></tr><tr><td></td><td> $[ \mu _ { \star } , \dot { \Sigma } _ { \star } , t ]$   $\left[ \mu _ { \star } , y , t \right] \left( \mathrm { E P S } \right)$ </td><td>24.52 24.53 [xt, y, t] (Palette) 24.09</td><td>0.7699 0.7712 0.7869</td><td>0.1115 0.1103</td><td>82.06 81.46</td><td>-6.4e-03 -6.4e-03</td><td>-4.3e-03 -4.4e-03 -4.4e-03</td><td>4.31 4.29 4.27</td><td>5.55 5.50 5.48</td></tr><tr><td></td><td> $[ \mu _ { \star } , t ]$   $[ \mu _ { \star } , \dot { \Sigma } _ { \star } , t ]$   $\left[ \mu _ { \star } , y , t \right] \left( \mathrm { E P S } \right)$   $\left[ x _ { t } , y , t \right] \left( \mathrm { P a l e t t e } \right)$ </td><td>23.53 24.35 24.34 21.12</td><td>0.7588 0.7944 0.7948 0.7541</td><td>0.1173 0.0986 0.0979 0.1218</td><td>88.24 80.19 79.60 92.73</td><td>-6.5e-03 -6.5e-03 -6.5e-03</td><td>-4.2e-03 -4.4e-03 -4.5e-03</td><td>4.18 4.05 4.04</td><td>5.66 5.43 5.41 5.93</td></tr><tr><td>Box inpaint</td><td> $[ \mu _ { \star } , t ]$   $[ \mu _ { \star } , \dot { \Sigma } _ { \star } , t ]$   $\left[ \mu _ { \star } , y , t \right] \left( \mathrm { E P S } \right)$   $\left[ x _ { t } , y , t \right] \left( \mathrm { P a l e t t e } \right)$ </td><td>21.19 21.18 21.24 20.24</td><td>0.7552 0.7544 0.7569 0.5364</td><td>0.1209 0.1212 0.1196 0.2220</td><td>91.74 91.03 91.07</td><td>-6.1e-03 -6.1e-03 -6.1e-03</td><td>-4.2e-03 -4.2e-03 -4.2e-03</td><td>5.88 5.92 5.87</td><td>5.86 5.82 5.84 7.33</td></tr><tr><td>Super-res (4×)</td><td> $[ \mu _ { \star } , t ]$   $[ \mu _ { \star } , \Sigma _ { \star } , t ]$   $\bar { [ } \mu _ { \star } , y , t ] \mathrm { ( E P S ) }$ </td><td>20.20 20.20 20.25</td><td>0.5308 0.5319 0.5369</td><td>0.2246 0.2250 0.2207</td><td>130.53 131.17 128.80</td><td>-5.8e-03 -5.8e-03 -5.9e-03</td><td>-2.6e-03 -2.7e-03 -2.8e-03</td><td>6.55 6.54 6.52</td><td>7.39 7.40 7.35</td></tr><tr><td>Gaussian deblur</td><td> $\left[ x _ { t } , y , t \right] \left( \mathrm { P a l e t t e } \right)$   $[ \mu _ { \star } , t ]$   $[ \mu _ { \star } , \dot { \Sigma } _ { \star } , t ]$ </td><td>29.15 29.18 29.18</td><td>0.9010 0.9014 0.9014</td><td>0.0491 0.0486 0.0487</td><td>46.62 46.76 46.73</td><td>-6.8e-03 -6.8e-03 -6.8e-03</td><td>-5.6e-03 -5.5e-03 -5.5e-03</td><td>2.26 2.25 2.25</td><td>4.11 4.12 4.13</td></tr><tr><td rowspan="6">Motion deblur</td><td>_  $\left[ \mu _ { \star } , y , t \right] \left( \mathrm { E P S } \right)$ </td><td>29.18</td><td>0.9015</td><td>0.0486</td><td>46.55</td><td>-6.8e-03</td><td>-5.6e-03</td><td>2.25</td><td>4.11</td></tr><tr><td> $\left[ x _ { t } , y , t \right] \left( \mathrm { P a l e t t e } \right)$ </td><td>27.02</td><td>0.8582</td><td>0.0680</td><td>62.86</td><td>-6.7e-03</td><td>-5.0e-03</td><td>2.93</td><td>4.81</td></tr><tr><td> $[ \mu _ { \star } , t ]$ </td><td></td><td>27.68 0.8674 27.67</td><td>0.0641 0.0642</td><td>60.94</td><td>-6.8e-03 61.19</td><td>-5.1e-03 -5.1e-03</td><td>2.68</td><td>4.70 4.70</td></tr></table>

## D.2 Zero-Shot Pivoting

Before fine-tuning, we can feed $\mu _ { \star }$ directly to the pretrained denoiser (with the closest available EDM noise level, or the exact equivalent time in isotropic cases). This is not exact for general A because the pretrained model has not learned anisotropic denoising, but it tests whether the pivot already carries useful posterior information.

Table 3 compares zero-shot pivoting to fine-tuned EPS and to Palette under the same 100-step Euler sampler on FFHQ-64 and ImageNet-64. Zero-shot pivoting feeds $\mu _ { \star }$ directly to the pretrained EDM denoiser at the current $\sigma _ { t }$ (no fine-tuning) and runs the standard 100-step Euler loop; fine-tuned EPS uses the same sampler but with the denoiser adapted to take $[ \mu _ { \star } , y ]$ on each task. Zero-shot pivoting underperforms both Palette and fine-tuned EPS on every task, confirming that the pretrained denoiser does not natively handle the anisotropic geometry of $\Sigma _ { \star } ( t )$ , and that the EPS fine-tuning step is what unlocks the benefit of the pivot.

Table 3: The pivot needs the fine-tuning step. Zero-shot pivoting vs. fine-tuned EPS at NFE=100 (Euler sampler) on FFHQ-64 (top) and ImageNet-64 (bottom). Feeding $\mu _ { \star }$ to the pretrained denoiser without adaptation underperforms Palette and fine-tuned EPS on every task and every metric, on both datasets. Best in bold, second-best underlined; the EPS row is highlighted in light pink.
<table><tr><td>Task</td><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID↓</td><td>MMD-pix ↓</td><td>MMD-Inc ↓</td><td>CRPS-pix ↓</td><td>CRPS-Inc ↓</td></tr><tr><td colspan="10">FFHQ 64×64</td></tr><tr><td rowspan="3">Average</td><td>Palette</td><td>26.03</td><td>0.8590</td><td>0.0626</td><td>31.50</td><td>-6.65e-03</td><td>-4.74e-03</td><td>3.43</td><td>3.86</td></tr><tr><td>Zero-shot EPS pivot</td><td>22.97</td><td>0.7386</td><td>0.1503</td><td>70.88</td><td>-3.14e-03</td><td>3.56e-02</td><td>5.61</td><td>6.68</td></tr><tr><td>EPS fine-tuned</td><td>26.69</td><td>0.8661</td><td>0.0590</td><td>29.94</td><td>-6.69e-03</td><td>-4.89e-03</td><td>3.24</td><td>3.73</td></tr><tr><td colspan="10">Random inpaint</td></tr><tr><td rowspan="3"></td><td>Palette Zero-shot EPS pivot</td><td>25.76 16.36</td><td>0.8809</td><td>0.0593</td><td>33.30 121.98</td><td>-6.71e-03 6.58e-03</td><td>-4.76e-03 1.02e-01</td><td>3.31 9.08</td><td>3.87 8.66</td></tr><tr><td>EPS fine-tuned</td><td>26.16</td><td>0.5302</td><td>0.2985</td><td></td><td></td><td>-5.00e-03</td><td></td><td></td></tr><tr><td>Palette</td><td></td><td>0.8879</td><td>0.0533</td><td>31.87</td><td>-6.74e-03</td><td></td><td>3.16</td><td>3.75</td></tr><tr><td colspan="10">Box inpaint</td></tr><tr><td rowspan="3"></td><td>Zero-shot EPS pivot</td><td>24.18 18.75</td><td>0.8426 0.7065</td><td>0.0577 0.1699</td><td>25.09 84.24</td><td>-6.58e-03 -5.19e-03</td><td>-5.30e-03 4.32e-02</td><td>4.02 6.71</td><td>3.47 6.69</td></tr><tr><td>EPS fine-tuned</td><td>24.23</td><td>0.8448</td><td>0.0567</td><td>24.74</td><td>-6.59e-03</td><td>-5.35e-03</td><td>4.01</td><td>3.45</td></tr><tr><td>Palette</td><td>21.95</td><td>0.7220</td><td>0.1273</td><td>49.28</td><td>-6.29e-03</td><td>-2.98e-03</td><td>5.22</td><td>5.29</td></tr><tr><td colspan="10">Super-res (4×)</td></tr><tr><td rowspan="3"></td><td>Zero-shot EPS pivot</td><td>20.50</td><td>0.6234</td><td>0.1713</td><td>57.45</td><td>-3.68e-03</td><td>4.23e-03</td><td>6.62</td><td>6.20</td></tr><tr><td>EPS fine-tuned</td><td>21.96</td><td>0.7232</td><td>0.1262</td><td>49.29</td><td>-6.30e-03</td><td>-3.00e-03</td><td>5.23</td><td>5.30</td></tr><tr><td>Palette</td><td>30.47</td><td>0.9397</td><td>0.0286</td><td>21.63</td><td>-6.90e-03</td><td>-5.65e-03</td><td>1.91</td><td>3.06</td></tr><tr><td colspan="10">Gaussian deblur</td></tr><tr><td rowspan="3"></td><td>Zero-shot EPS pivot EPS fine-tuned</td><td>29.79 30.82</td><td>0.9197</td><td>0.0540</td><td>43.77</td><td>-6.73e-03</td><td>1.31e-02</td><td>2.76</td><td>5.79</td></tr><tr><td></td><td></td><td>0.9408</td><td>0.0273</td><td>20.68</td><td>-6.91e-03</td><td>-5.73e-03</td><td>1.84</td><td>2.99</td></tr><tr><td>Palette Zero-shot EPS pivot</td><td>27.79 29.42</td><td>0.9099 0.9135</td><td>0.0404 0.0579</td><td>28.23 46.95</td><td>-6.77e-03 -6.69e-03</td><td>-4.98e-03 1.54e-02</td><td>2.67 2.89</td><td>3.59 6.05</td></tr><tr><td colspan="10">Motion deblur EPS fine-tuned</td></tr><tr><td colspan="10"></td></tr><tr><td colspan="10">Palette Average</td></tr><tr><td rowspan="3"></td><td>Zero-shot EPS pivot EPS fine-tuned</td><td>24.32 22.02</td><td>0.7673 0.6398</td><td>0.1124 0.2461</td><td>82.57 146.88</td><td>-6.40e-03 2.48e-03</td><td>-4.38e-03 1.06e-02</td><td>4.35 6.96</td><td>5.54 8.89</td></tr><tr><td></td><td>24.53</td><td>0.7712</td><td>0.1103</td><td>81.46</td><td>-6.42e-03</td><td>-4.45e-03</td><td>4.27</td><td>5.48</td></tr><tr><td>Palette</td><td>24.09</td><td></td><td>0.1011</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Random inpaint</td></tr><tr><td rowspan="3"></td><td>Zero-shot EPS pivot</td><td>16.92</td><td>0.7869 0.4168</td><td></td><td>81.88 223.68</td><td>-6.50e-03 2.61e-02</td><td>-4.41e-03 4.15e-02</td><td>4.16 10.14</td><td>5.52 11.23</td></tr><tr><td>EPS fine-tuned</td><td>24.34</td><td>0.7948</td><td>0.4350</td><td></td><td>-6.52e-03</td><td>-4.51e-03</td><td></td><td></td></tr><tr><td>Palette</td><td></td><td></td><td>0.0979</td><td>79.60</td><td></td><td></td><td>4.04</td><td>5.41</td></tr><tr><td colspan="10">Box inpaint</td></tr><tr><td rowspan="3"></td><td>Zero-shot EPS pivot</td><td>21.12 18.70</td><td>0.7541</td><td>0.1218</td><td>92.73 139.38</td><td>-6.10e-03 -1.38e-03</td><td>-4.12e-03 2.45e-03</td><td>5.92 8.56</td><td>5.93 8.10</td></tr><tr><td>EPS fine-tuned</td><td>21.24</td><td>0.6442 0.7569</td><td>0.2333 0.1196</td><td>91.07</td><td>-6.11e-03</td><td>-4.23e-03</td><td>5.87</td><td>5.84</td></tr><tr><td>Palette</td><td>20.24</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Super-res (4×)</td></tr><tr><td rowspan="3"></td><td>Zero-shot EPS pivot</td><td>19.85</td><td>0.5364 0.4663</td><td>0.2220 0.3035</td><td>128.76 160.80</td><td>-5.86e-03 -9.25e-04</td><td>-2.79e-03 6.29e-03</td><td>6.50 7.91</td><td>7.33 8.72</td></tr><tr><td>EPS fine-tuned</td><td>20.25</td><td>0.5369</td><td>0.2207</td><td>128.80</td><td>-5.86e-03</td><td>-2.84e-03</td><td>6.52</td><td>7.35</td></tr><tr><td>Palette</td><td>29.15</td><td>0.9010</td><td>0.0491</td><td>46.62</td><td>-6.83e-03</td><td>-5.56e-03</td><td>2.26</td><td>4.11</td></tr><tr><td colspan="10">Gaussian deblur</td></tr><tr><td rowspan="3"></td><td>Zero-shot EPS pivot</td><td>27.83</td><td>0.8507</td><td>0.1211</td><td>97.87</td><td>-5.85e-03</td><td>7.94e-04</td><td>3.87</td><td>7.97</td></tr><tr><td>EPS fine-tuned</td><td>29.18</td><td>0.9015</td><td>0.0486</td><td>46.55</td><td>-6.83e-03</td><td>-5.55e-03</td><td>2.25</td><td>4.11</td></tr><tr><td>Palette</td><td>27.02</td><td>0.8582</td><td>0.0680</td><td>62.86</td><td>-6.73e-03</td><td>-5.02e-03</td><td>2.93</td><td>4.81</td></tr><tr><td rowspan="3">Motion deblur</td><td colspan="10">Zero-shot EPS pivot</td></tr><tr><td>EPS fine-tuned</td><td>26.82</td><td>0.8212</td><td>0.1374</td><td>112.64</td><td>-5.59e-03</td><td>2.19e-03</td><td>4.31</td><td>8.42</td></tr><tr><td></td><td>27.62</td><td>0.8661</td><td>0.0647</td><td>61.29</td><td>-6.77e-03</td><td>-5.11e-03</td><td>2.69</td><td>4.70</td></tr></table>

## D.3 Palette vs EPS

Figure 4 reports fine-tuning convergence curves for EPS and Palette warm-started from the same pretrained EDM checkpoint. Rows index five (dataset, task) pairs (ImageNet-64 random inpainting, ImageNet-64 motion deblurring, FFHQ-64 random inpainting, FFHQ-64 motion deblurring, FFHQ-64 Gaussian deblurring); columns track training loss, PSNR, SSIM, LPIPS, and FID against the number of fine-tuning iterations under the matched 100-step Euler sampler. Three patterns hold across all rows. First, EPS starts from a much better initialization on every metric: at iteration zero, the shifted pivot $\mu _ { \star }$ already encodes enough of the measurement geometry that PSNR and SSIM are within a few units of their converged values, while Palette has to climb from the unconditional prior. Second, EPS reaches its asymptotic LPIPS and FID in a small fraction of the iterations Palette needs and stays at least as low for the rest of training. Third, the gap is largest on the deblurring tasks, where the pivot provides the strongest measurement signal at initialization, and on the perceptual and distributional metrics (LPIPS, FID), where the conditioning structure matters most. This is consistent with the structural-locality argument: EPS preserves the input/output type and Gaussian-corruptedtarget geometry of the pretrained denoising task, and only adapts to the operator-induced anisotropic covariance Σ<sub>⋆</sub>(t).

![](images/32cfb72d1a5e30e76d6f18cc522bf59014aa96adf9a1f805d9bf3a68a0a9c99d.jpg)  
Figure 4: EPS converges faster than Palette from the same warm start. Fine-tuning curves for EPS and Palette, both initialized from the same pretrained EDM checkpoint, on five (dataset, task) pairs (rows) and five metrics (columns: training loss, PSNR, SSIM, LPIPS, FID) under the matched 100-step Euler sampler. EPS starts from a markedly better initialization on every metric and reaches its asymptote in a small fraction of the iterations Palette needs.

## D.4 Sampling Efficiency

We sweep NFE at inference from 5 to 100 across all five tasks, with all methods using a 1-NFE-perstep Euler sampler under matched conditions.

Figures 5 and 6 report PSNR, FID, and Inception-feature CRPS as a function of sampler iterations on FFHQ-64 and ImageNet-64 respectively. Rows index the five tasks (random inpaint, box inpaint, 4× super-resolution, Gaussian deblur, motion deblur); columns track the three reported metrics. EPS reaches its asymptotic FID and CRPS within roughly 15-20 steps on every task and stays flat thereafter, while sampling-based baselines either fail to reach the same level (DPS, MPGD) or are slower to converge (DDNM, ΠGDM). The flat right tail of the EPS curves is the practical justification for the 20-NFE setting reported in the main results (Tables 1 and 6), and the gap to the strongest sampling-based baseline (ΠGDM) widens on the deblurring tasks under the more diverse ImageNet distribution.

![](images/9f8091ba207162f4771f45058078b637f7934990050db04d3807de2b73b34259.jpg)  
Figure 5: EPS plateaus by 15-20 steps on FFHQ-64. Sampling-step sensitivity on FFHQ-64 across the five tasks (rows). Columns report PSNR (↑), FID (↓), and Inception-feature CRPS (↓) versus sampler iterations under a 1-NFE-per-step Euler sampler. EPS reaches its asymptote within roughly 15-20 steps on every task and remains best or tied-best on FID and CRPS thereafter.

![](images/10b6d753eeb933ed6e95ef7edb7a70a3af5a053626bfe58b6b8827bfdc43f701.jpg)  
Figure 6: The plateau transfers to ImageNet-64. Sampling-step sensitivity on ImageNet-64; same layout as Fig. 5. EPS plateaus at the same step count on a more diverse class-conditional distribution, and the gap between EPS and the strongest sampling-based baseline (ΠGDM) widens on the deblurring tasks.

## D.5 Amortized Variant Across All Five Tasks

As a deployment-friendly alternative to per-task fine-tuning, we train a single EPS checkpoint across all five tasks using uniformly sampled operators per training step and no task indicator at the input.

Table 4 compares per-task EPS, amortized EPS, and Palette under matched compute on FFHQ-64 (top) and ImageNet-64 (bottom). On FFHQ-64 we report two amortized snapshots, at 55k and 160k training steps; the 160k checkpoint surpasses per-task EPS on every task and every metric, indicating that a single network can absorb all five operators without loss. On ImageNet-64, the amortized model is within 0.1-0.2 dB PSNR of per-task EPS and matches Palette or better on the distributional metrics, while requiring 5× less storage and a single set of weights at deployment.

Table 4: Amortization works. Amortized EPS vs. per-task EPS on FFHQ-64 (top) and ImageNet-64 (bottom) at NFE=100 with the EDM Euler sampler. The FFHQ amortized model is reported at two training-step snapshots (55k, 160k); the 160k snapshot surpasses per-task EPS across every task and metric. The ImageNet amortized model (single 296M-param ADM checkpoint, ∼60 epochs) matches per-task EPS within 0.1–0.2 dB PSNR and matches Palette or better on the distributional metrics. Best in bold, second-best underlined; amortized rows are highlighted in light pink.
<table><tr><td>Task</td><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID↓</td><td>MMD-pix ↓</td><td>MMD-Inc ↓</td><td>CRPS-pix ↓</td><td>CRPS-Inc ↓</td></tr><tr><td colspan="10">FFHQ 64×64</td></tr><tr><td rowspan="4">Average</td><td>Palette Per-task EPS</td><td>26.03 26.69</td><td>0.8590 0.8661</td><td>0.0626 0.0590</td><td>31.50 29.94</td><td>-6.65e-03 -6.69e-03</td><td>-4.74e-03 -4.89e-03</td><td>3.43 3.24</td><td>3.86 3.73</td></tr><tr><td>Amortized EPS (55k steps)</td><td>26.95</td><td>0.8745</td><td>0.0552</td><td>27.86</td><td>-6.71e-03</td><td>-5.20e-03</td><td>3.10</td><td>3.51</td></tr><tr><td>Amortized EPS (160k steps)</td><td>27.31</td><td>0.8830</td><td>0.0513</td><td>26.16</td><td>-6.74e-03</td><td>-5.25e-03</td><td>2.98</td><td>3.38</td></tr><tr><td>Palette</td><td>25.76</td><td>0.8809</td><td>0.0593</td><td>33.30</td><td>-6.71e-03</td><td>-4.76e-03</td><td>3.31</td><td>3.87</td></tr><tr><td rowspan="4">Random inpaint</td><td>Per-task EPS</td><td>26.16</td><td>0.8879</td><td>0.0533</td><td>31.87</td><td>-6.74e-03</td><td>-5.00e-03</td><td>3.16</td><td>3.75</td></tr><tr><td>Amortized EPS (55k steps)</td><td>26.27</td><td>0.8920</td><td>0.0519</td><td>29.76</td><td>-6.74e-03</td><td>-5.20e-03</td><td>3.10</td><td>3.56</td></tr><tr><td>Amortized EPS (160k steps)</td><td>26.73</td><td>0.9001</td><td>0.0478</td><td>27.57</td><td>-6.77e-03</td><td>-5.34e-03</td><td>2.96</td><td>3.40</td></tr><tr><td>Palette</td><td>24.18</td><td>0.8426</td><td>0.0577</td><td>25.09</td><td>-6.58e-03</td><td>-5.30e-03</td><td>4.02</td><td>3.47</td></tr><tr><td rowspan="4">Box inpaint</td><td>Per-task EPS</td><td>24.23</td><td>0.8448</td><td>0.0567</td><td>24.74</td><td>-6.59e-03</td><td>-5.35e-03</td><td>4.01</td><td>3.45</td></tr><tr><td>Amortized EPS (55k steps)</td><td>24.75</td><td>0.8580</td><td>0.0517</td><td>22.25</td><td>-6.64e-03</td><td>-5.60e-03</td><td>3.74</td><td>3.18</td></tr><tr><td>Amortized EPS (160k steps)</td><td>25.14</td><td>0.8663</td><td>0.0486</td><td>21.07</td><td>-6.67e-03</td><td>-5.68e-03</td><td>3.61</td><td>3.07</td></tr><tr><td>Palette</td><td>21.95</td><td>0.7220</td><td>0.1273</td><td>49.28</td><td>-6.29e-03</td><td>-2.98e-03</td><td>5.22</td><td>5.29</td></tr><tr><td rowspan="4">Super-res (4×)</td><td>Per-task EPS</td><td>21.96</td><td>0.7232</td><td>0.1262</td><td>49.29</td><td>-6.30e-03</td><td>-3.00e-03</td><td>5.23</td><td>5.30</td></tr><tr><td>Amortized EPS (55k steps)</td><td>22.33</td><td>0.7433</td><td>0.1158</td><td>45.53</td><td>-6.36e-03</td><td>-3.80e-03</td><td>4.96</td><td>4.89</td></tr><tr><td>Amortized EPS (160k steps)</td><td>22.74</td><td>0.7633</td><td>0.1072</td><td>43.42</td><td>-6.42e-03</td><td>-3.66e-03</td><td>4.74</td><td>4.74</td></tr><tr><td>Palette</td><td>30.47</td><td>0.9397</td><td>0.0286</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Gaussian deblur</td><td></td><td></td><td></td><td></td><td>21.63</td><td>-6.90e-03</td><td>-5.65e-03</td><td>1.91</td><td>3.06</td></tr><tr><td>Per-task EPS</td><td>30.82 31.01</td><td>0.9408</td><td>0.0273</td><td>20.68</td><td>-6.91e-03</td><td>-5.73e-03</td><td>1.84</td><td>2.99</td></tr><tr><td>Amortized EPS (55k steps)</td><td>31.19</td><td>0.9433</td><td>0.0260</td><td>19.54</td><td>-6.92e-03</td><td>-5.82e-03</td><td>1.79</td><td>2.87</td></tr><tr><td>Amortized EPS (160k steps)</td><td></td><td>0.9451</td><td>0.0250</td><td>18.57</td><td>-6.92e-03</td><td>-5.87e-03</td><td>1.76</td><td>2.80</td></tr><tr><td rowspan="4">Motion deblur</td><td>Palette</td><td>27.79</td><td>0.9099</td><td>0.0404</td><td>28.23</td><td>-6.77e-03</td><td>-4.98e-03</td><td>2.67</td><td>3.59</td></tr><tr><td>Per-task EPS</td><td>30.27</td><td>0.9339</td><td>0.0311</td><td>23.10</td><td>-6.90e-03</td><td>-5.39e-03</td><td>1.94</td><td>3.18</td></tr><tr><td>Amortized EPS (55k steps)</td><td>30.37</td><td>0.9358</td><td>0.0307</td><td>22.21</td><td>-6.91e-03</td><td>-5.59e-03</td><td>1.90</td><td>3.03</td></tr><tr><td>Amortized EPS (160k steps)</td><td>30.74</td><td>0.9402</td><td>0.0281</td><td>20.17</td><td>-6.91e-03</td><td>-5.71e-03</td><td>1.83</td><td>2.88</td></tr><tr><td colspan="10">ImageNet 64×64</td></tr><tr><td rowspan="3">Average</td><td>Palette</td><td>24.32</td><td>0.7673</td><td>0.1124</td><td>82.57</td><td>-6.40e-03</td><td>-4.38e-03</td><td>4.35</td><td>5.54</td></tr><tr><td>Per-task EPS</td><td>24.53</td><td>0.7712</td><td>0.1103</td><td>81.46</td><td>-6.42e-03</td><td>-4.45e-03</td><td>4.27</td><td>5.48</td></tr><tr><td>Amortized EPS</td><td>24.41</td><td>0.7677</td><td>0.1116</td><td>81.82</td><td>-6.41e-03</td><td>-4.39e-03</td><td>4.29</td><td>5.50</td></tr><tr><td rowspan="3">Random inpaint</td><td>Palette</td><td>24.09</td><td>0.7869</td><td>0.1011</td><td>81.88</td><td>-6.50e-03</td><td>-4.41e-03</td><td>4.16</td><td>5.52</td></tr><tr><td>Per-task EPS</td><td>24.34</td><td>0.7948</td><td>0.0979</td><td>79.60</td><td>-6.52e-03</td><td>-4.51e-03</td><td>4.04</td><td>5.41</td></tr><tr><td>Amortized EPS</td><td>24.14</td><td>0.7890</td><td>0.1007</td><td>81.27</td><td>-6.52e-03</td><td>-4.42e-03</td><td>4.08</td><td>5.48</td></tr><tr><td rowspan="3">Box inpaint</td><td>Palette</td><td>21.12</td><td>0.7541</td><td>0.1218</td><td>92.73</td><td>-6.10e-03</td><td>-4.12e-03</td><td>5.92</td><td>5.93</td></tr><tr><td>Per-task EPS</td><td>21.24</td><td>0.7569</td><td>0.1196</td><td>91.07</td><td>-6.11e-03</td><td>-4.23e-03</td><td>5.87</td><td>5.84</td></tr><tr><td>Amortized EPS</td><td>21.12</td><td>0.7549</td><td>0.1199</td><td>89.59</td><td>-6.10e-03</td><td>-4.34e-03</td><td>5.88</td><td>5.81</td></tr><tr><td rowspan="3">Super-res (4×)</td><td>Palette</td><td>20.24</td><td>0.5364</td><td>0.2220</td><td>128.76</td><td>-5.86e-03</td><td>-2.79e-03</td><td>6.50</td><td>7.33</td></tr><tr><td>Per-task EPS</td><td>20.25</td><td>0.5369</td><td>0.2207</td><td>128.80</td><td>-5.86e-03</td><td>-2.84e-03</td><td>6.52</td><td>7.35</td></tr><tr><td>Amortized EPS</td><td>20.15</td><td>0.5308</td><td>0.2238</td><td>130.55</td><td>-5.85e-03</td><td>-2.54e-03</td><td>6.56</td><td>7.38</td></tr><tr><td rowspan="3">Gaussian deblur</td><td>Palette</td><td>29.15</td><td>0.9010</td><td>0.0491</td><td>46.62</td><td>-6.83e-03</td><td>-5.56e-03</td><td>2.26</td><td>4.11</td></tr><tr><td>Per-task EPS</td><td>29.18</td><td>0.9015</td><td>0.0486</td><td>46.55</td><td>-6.83e-03</td><td>-5.55e-03</td><td>2.25</td><td>4.11</td></tr><tr><td>Amortized EPS Palette</td><td>29.19 27.02</td><td>0.9017 0.8582</td><td>0.0478</td><td>45.33</td><td>-6.83e-03</td><td>-5.61e-03</td><td>2.24</td><td>4.06</td></tr><tr></table>

## D.6 One-Step Posterior Mean Check

Section 3.5 predicts that a single high-noise evaluation returns a posterior-mean estimator. We compare EPS at 1 NFE (a single direct Tweedie call $D _ { \theta } ( \mu _ { \star } , \sigma _ { \operatorname* { m a x } } )$ at $\sigma _ { \mathrm { m a x } } { = } 8 0$ , no sampler loop) to the empirical mean of J=10 multi-step posterior samples drawn with the standard 100-step Euler sampler (1000 NFE per image), and to a single posterior sample from the same multi-step sampler (the EPS NFE=100 row used in our main results).

Table 5 reports this comparison on FFHQ-64 (top) and ImageNet-64 (bottom) across all five tasks. The 1-NFE Tweedie call recovers most of the PSNR/SSIM gain that the multi-step empirical mean attains while using 1000× fewer denoiser evaluations, confirming the high-noise posterior-mean prediction. As expected, both rows that target the conditional mean (the 1-NFE row and the empirical-mean row) score better on distortion (PSNR, SSIM) than the single-sample row, while the single-sample row scores better on perceptual and distributional metrics (LPIPS, FID, CRPS, MMD). Note that MMD/CRPS for the empirical-mean row degenerate to deterministic distances since the per-image ensemble has size J=1 after averaging.

Table 5: One Tweedie call recovers most of the multi-step gain. One-step EPS vs. empirical posterior mean from multi-step EPS samples on FFHQ-64 (top) and ImageNet-64 (bottom). The 1-NFE row is a single direct Tweedie call at $\sigma _ { \mathrm { m a x } } { = } 8 0$ (no sampler loop). The posterior-mean row averages 10 independent 100-step Euler samples (1000 NFE per image). The single-sample row reports per-seed metrics from the same 100-step sampler (matching the main-text NFE=100 row). The 1-NFE row matches or trails the empirical-mean row by a small margin on distortion metrics while using 1000× fewer denoiser calls. Best in bold, second-best underlined.
<table><tr><td>Task</td><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID↓</td><td>MMD-pix ↓</td><td>MMD-Inc ↓</td><td>CRPS-pix ↓</td><td>CRPS-Inc ↓</td></tr><tr><td colspan="10">FFHQ 64×64</td></tr><tr><td rowspan="3">Average</td><td>EPS, 1 NFE</td><td>27.34</td><td>0.8887</td><td>0.0699</td><td>46.30</td><td>-6.31e-03</td><td>1.13e-02</td><td>4.67</td><td>6.91</td></tr><tr><td>EPS, posterior mean from samples</td><td>29.10</td><td>0.9095</td><td>0.0480</td><td>34.17</td><td>-1.22e-02</td><td>-2.77e-03</td><td>4.35</td><td>6.46</td></tr><tr><td>EPS, single posterior sample</td><td>26.69</td><td>0.8661</td><td>0.0590</td><td>29.94</td><td>-6.69e-03</td><td>-4.89e-03</td><td>3.24</td><td>3.73</td></tr><tr><td colspan="10">Random inpaint</td></tr><tr><td rowspan="3"></td><td>EPS, 1 NFE EPS, posterior mean from samples</td><td>27.90 28.66</td><td>0.9148 0.9293</td><td>0.0549 0.0399</td><td>39.71 34.17</td><td>-6.63e-03 -1.23e-02</td><td>2.18e-03 -5.15e-03</td><td>3.94 4.26</td><td>5.75 6.48</td></tr><tr><td>EPS, single posterior sample</td><td>26.16</td><td>0.8879</td><td>0.0533</td><td>31.87</td><td>-6.74e-03</td><td>-5.00e-03</td><td>3.16</td><td>3.75</td></tr><tr><td>EPS, 1 NFE</td><td>26.01</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Box inpaint</td></tr><tr><td rowspan="3"></td><td>EPS, posterior mean from samples</td><td>26.72</td><td>0.8706 0.8906</td><td>0.0644 0.0464</td><td>32.90 26.75</td><td>-6.36e-03 -1.20e-02</td><td>-8.49e-04 -8.08e-03</td><td>5.05 5.38</td><td>5.62 5.93</td></tr><tr><td>EPS, single posterior sample</td><td>24.23</td><td>0.8448</td><td>0.0567</td><td>24.74</td><td>-6.59e-03</td><td>-5.35e-03</td><td>4.01</td><td>3.45</td></tr><tr><td>EPS, 1 NFE</td><td>24.21</td><td>0.7999</td><td>0.1265</td><td>77.97</td><td>-5.63e-03</td><td>4.09e-02</td><td>6.57</td><td>9.27</td></tr><tr><td colspan="10">Super-res (4×)</td></tr><tr><td rowspan="3"></td><td>EPS, posterior mean from samples</td><td>24.21</td><td>0.8042</td><td>0.1087</td><td>64.71</td><td>-1.14e-02</td><td>1.35e-02</td><td>7.03</td><td>9.36</td></tr><tr><td>EPS, single posterior sample</td><td>21.96</td><td>0.7232</td><td>0.1262</td><td>49.29</td><td>-6.30e-03</td><td>-3.00e-03</td><td>5.23</td><td>5.30</td></tr><tr><td>EPS, 1 NFE</td><td>30.86</td><td>0.9503</td><td>0.0369</td><td>31.39</td><td>-6.62e-03</td><td>2.35e-03</td><td>3.21</td><td>6.15</td></tr><tr><td colspan="10">Gaussian deblur</td></tr><tr><td rowspan="3"></td><td>EPS, posterior mean from samples EPS, single posterior sample</td><td>33.18 30.82</td><td>0.9637</td><td>0.0211</td><td>21.15</td><td>-1.26e-02</td><td>-7.75e-03 -5.73e-03</td><td>2.48</td><td>5.05 2.99</td></tr><tr><td></td><td></td><td>0.9408</td><td>0.0273</td><td>20.68</td><td>-6.91e-03</td><td></td><td>1.84</td><td></td></tr><tr><td>EPS, 1 NFE EPS, posterior mean from samples</td><td>27.73 32.71</td><td>0.9078 0.9597</td><td>0.0669 0.0239</td><td>49.55 24.06</td><td>-6.31e-03 -1.25e-02</td><td>1.21e-02 -6.39e-03</td><td>4.60 2.61</td><td>7.76 5.46</td></tr><tr><td colspan="10">Motion deblur EPS, single posterior sample</td></tr><tr><td colspan="10"></td></tr><tr><td colspan="10"></td></tr><tr><td rowspan="3">Average</td><td>EPS, 1 NFE EPS, posterior mean from samples</td><td>25.67</td><td>0.8165</td><td>0.1318</td><td>110.95 90.93</td><td>-5.58e-03 -1.16e-02</td><td>4.49e-03 -5.77e-03</td><td>5.91</td><td>9.96</td></tr><tr><td>EPS, single posterior sample</td><td>26.97 24.53</td><td>0.8330</td><td>0.0999 0.1103</td><td>81.46</td><td>-6.42e-03</td><td>-4.45e-03</td><td>5.74 4.27</td><td>9.76 5.48</td></tr><tr><td></td><td></td><td>0.7712</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Random inpaint</td></tr><tr><td rowspan="3"></td><td>EPS, 1 NFE</td><td>26.60</td><td>0.8580</td><td>0.0933</td><td>88.59</td><td>-6.15e-03</td><td>-4.84e-04</td><td>4.98</td><td>8.32</td></tr><tr><td>EPS, posterior mean from samples</td><td>26.81</td><td>0.8651</td><td>0.0790</td><td>81.97</td><td>-1.19e-02</td><td>-7.41e-03</td><td>5.44</td><td>9.35</td></tr><tr><td>EPS, single posterior sample</td><td>24.34</td><td>0.7948</td><td>0.0979</td><td>79.60</td><td>-6.52e-03</td><td>-4.51e-03</td><td>4.04</td><td>5.41</td></tr><tr><td colspan="10"></td></tr><tr><td rowspan="3">Box inpaint</td><td>EPS, 1 NFE</td><td>23.60 23.65</td><td>0.7908</td><td>0.1514</td><td>129.11</td><td>-5.60e-03</td><td>7.34e-03</td><td>7.16</td><td>10.38</td></tr><tr><td>EPS, posterior mean from samples EPS, single posterior sample</td><td>21.24</td><td>0.7985</td><td>0.1149</td><td>111.46 91.07</td><td>-1.10e-02 -6.11e-03</td><td>-3.94e-03 -4.23e-03</td><td>7.86 5.87</td><td>11.12 5.84</td></tr><tr><td>EPS, 1 NFE</td><td>22.78</td><td>0.7569</td><td>0.1196</td><td></td><td>-4.97e-03</td><td></td><td></td><td></td></tr><tr><td colspan="10">Super-res (4×)</td></tr><tr><td rowspan="3"></td><td>EPS, posterior mean from samples</td><td>22.57</td><td>0.6530 0.6447</td><td>0.2455 0.2078</td><td>182.92 158.62</td><td>-1.03e-02</td><td>1.98e-02 1.40e-03</td><td>8.06 8.75</td><td>13.47 13.68</td></tr><tr><td>EPS, single posterior sample</td><td>20.25</td><td>0.5369</td><td>0.2207</td><td>128.80</td><td>-5.86e-03</td><td>-2.84e-03</td><td>6.52</td><td>7.35</td></tr><tr><td>EPS, 1 NFE</td><td>28.82</td><td>0.9194</td><td>0.0606</td><td>56.53</td><td>-5.14e-03</td><td>-3.46e-03</td><td>4.09</td><td>7.73</td></tr><tr><td colspan="10">Gaussian deblur</td></tr><tr><td rowspan="3"></td><td>EPS, posterior mean from samples</td><td>31.66</td><td>0.9399</td><td>0.0404</td><td>41.79</td><td>-1.24e-02</td><td>-1.02e-02</td><td>3.03</td><td>6.63</td></tr><tr><td>EPS, single posterior sample</td><td>29.18</td><td>0.9015</td><td>0.0486</td><td>46.55</td><td>-6.83e-03</td><td>-5.55e-03</td><td>2.25</td><td>4.11</td></tr><tr><td>EPS, 1 NFE</td><td>26.56</td><td>0.8613</td><td>0.1079</td><td>97.59</td><td>-6.03e-03</td><td>-7.30e-04</td><td>5.25</td><td>9.91</td></tr><tr><td rowspan="3">Motion deblur</td><td>EPS, posterior mean from samples</td><td>30.17</td><td>0.9167</td><td>0.0573</td><td>60.82</td><td>-1.23e-02</td><td>-8.67e-03</td><td>3.63</td><td>8.03</td></tr><tr><td>EPS, single posterior sample</td><td>27.62</td><td>0.8661</td><td>0.0647</td><td>61.29</td><td>-6.77e-03</td><td>-5.11e-03</td><td>2.69</td><td>4.70</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## D.7 Additional 64×64 Results

For completeness, Table 6 reproduces the FFHQ-64 main-table comparison from the body of the paper, broken out by task with all metrics. EPS at NFE=20 is the strongest configuration on perceptual and distributional metrics across all five tasks, while the NFE=1 Tweedie variant trades distributional fidelity for distortion (PSNR, SSIM), as predicted by the high-noise posterior-mean limit of Section 3.5.

Table 6: Detailed FFHQ-64 results. Quantitative comparison across the five inverse problems on FFHQ 64×64. All methods use a 1-NFE-per-step Euler sampler; reported NFE equals the number of sampler iterations. Best in bold, second-best underlined; EPS rows highlighted in light pink. † The NFE=1 row evaluates the deterministic high-noise posterior-mean limit (one direct Tweedie call $D _ { \theta } ( \mu _ { \star } , \sigma _ { \operatorname* { m a x } } ) )$ ; MMSE-optimal in pixel space but does not produce posterior samples, hence its strong PSNR/SSIM but weaker distributional metrics.
<table><tr><td>Task</td><td>Method</td><td>NFE</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID↓</td><td>MMD-pix ↓</td><td>MMD-Inc ↓</td><td>CRPS-pix ↓</td><td>CRPS-Inc ↓</td></tr><tr><td rowspan="9">Random inpaint</td><td>DPS</td><td>250</td><td>23.09</td><td>0.8003</td><td>0.1080</td><td>70.27</td><td>-6.04e-03</td><td>3.00e-02</td><td>4.56</td><td>6.33</td></tr><tr><td>DAPS</td><td>100</td><td>22.45</td><td>0.7576</td><td>0.1519</td><td>83.31</td><td>-4.74e-03</td><td>4.95e-02</td><td>5.47</td><td>7.52</td></tr><tr><td>DDNM</td><td>100</td><td>24.50</td><td>0.8566</td><td>0.0766</td><td>45.40</td><td>-6.44e-03</td><td>4.88e-04</td><td>4.03</td><td>4.87</td></tr><tr><td>IIGDM</td><td>100</td><td>26.00</td><td>0.8898</td><td>0.0556</td><td>35.17</td><td>-6.69e-03</td><td>-3.83e-03</td><td>3.44</td><td>4.12</td></tr><tr><td>MPGD</td><td>100</td><td>20.60</td><td>0.7097</td><td>0.1755</td><td>102.37</td><td>-1.67e-03</td><td>7.62e-02</td><td>6.29</td><td>8.05</td></tr><tr><td>Palette</td><td>100</td><td>25.76</td><td>0.8809</td><td>0.0593</td><td>33.30</td><td>-6.71e-03</td><td>-4.76e-03</td><td>3.31</td><td>3.87</td></tr><tr><td>EPS (ours)</td><td>100</td><td>26.16</td><td>0.8879</td><td>0.0533</td><td>31.87</td><td>-6.74e-03</td><td>-5.00e-03</td><td>3.16</td><td>3.75</td></tr><tr><td>EPS (ours)</td><td>20</td><td>26.75</td><td>0.9006</td><td>0.0489</td><td>31.56</td><td>-6.74e-03</td><td>-3.94e-03</td><td>3.15</td><td>3.89</td></tr><tr><td>EPS (ours)†</td><td>1</td><td>27.90</td><td>0.9148</td><td>0.0549</td><td>39.71</td><td>-6.63e-03</td><td>2.18e-03</td><td>3.94</td><td>5.75</td></tr><tr><td rowspan="9">Box inpaint</td><td>DPS</td><td>250</td><td>22.33</td><td>0.7692</td><td>0.0946</td><td>41.58</td><td>-6.30e-03</td><td>5.29e-03</td><td>5.09</td><td>5.08</td></tr><tr><td>DAPS</td><td>100</td><td>23.06</td><td>0.7968</td><td>0.0900</td><td>39.35</td><td>-6.01e-03</td><td>6.47e-03</td><td>5.09</td><td>5.00</td></tr><tr><td>DDNM</td><td>100</td><td>23.15</td><td>0.8231</td><td>0.0737</td><td>28.75</td><td>-6.20e-03</td><td>-4.28e-03</td><td>4.82</td><td>4.03</td></tr><tr><td>IIGDM</td><td>100</td><td>23.80</td><td>0.8359</td><td>0.0623</td><td>26.81</td><td>-6.51e-03</td><td>-4.33e-03</td><td>4.36</td><td>3.78</td></tr><tr><td>MPGD</td><td>100</td><td>20.39</td><td>0.7211</td><td>0.1229</td><td>49.51</td><td>-4.55e-03</td><td>9.16e-03</td><td>6.98</td><td>6.00</td></tr><tr><td>Palette</td><td>100</td><td>24.18</td><td>0.8426</td><td>0.0577</td><td>25.09</td><td>-6.58e-03</td><td>-5.30e-03</td><td>4.02</td><td>3.47</td></tr><tr><td>EPS (ours)</td><td>100</td><td>24.23</td><td>0.8448</td><td>0.0567</td><td>24.74</td><td>-6.59e-03</td><td>-5.35e-03</td><td>4.01</td><td>3.45</td></tr><tr><td>EPS (ours)</td><td>20</td><td>24.70</td><td>0.8560</td><td>0.0536</td><td>24.45</td><td>-6.59e-03</td><td>-4.77e-03</td><td>3.99</td><td>3.55</td></tr><tr><td>EPS (ours)†</td><td>1</td><td>26.01</td><td>0.8706</td><td>0.0644</td><td>32.90</td><td>-6.36e-03</td><td>-8.49e-04</td><td>5.05</td><td>5.62</td></tr><tr><td rowspan="9">Super-res (4×)</td><td>DPS</td><td>250</td><td>20.43</td><td>0.5968</td><td>0.2162</td><td>90.79</td><td>-5.71e-03</td><td>4.85e-02</td><td>6.62</td><td>7.69</td></tr><tr><td>DAPS</td><td>100</td><td>18.87</td><td>0.5112</td><td>0.2658</td><td>110.31</td><td>-1.81e-03</td><td>1.15e-01</td><td>8.28</td><td>9.35</td></tr><tr><td>DDNM</td><td>100</td><td>22.35</td><td>0.7201</td><td>0.1592</td><td>62.67</td><td>-6.06e-03</td><td>1.90e-02</td><td>5.79</td><td>6.63</td></tr><tr><td>IIGDM</td><td>100</td><td>22.02</td><td>0.7242</td><td>0.1281</td><td>50.92</td><td>-6.28e-03</td><td>-1.23e-03</td><td>5.39</td><td>5.48</td></tr><tr><td>MPGD</td><td>100</td><td>20.49</td><td>0.5809</td><td>0.2496</td><td>120.27</td><td>-4.69e-03</td><td>1.05e-01</td><td>7.59</td><td>9.29</td></tr><tr><td>Palette</td><td>100</td><td>21.95</td><td>0.7220</td><td>0.1273</td><td>49.28</td><td>-6.29e-03</td><td>-2.98e-03</td><td>5.22</td><td>5.29</td></tr><tr><td>EPS (ours)</td><td>100</td><td>21.96</td><td>0.7232 0.7480</td><td>0.1262</td><td>49.29</td><td>-6.30e-03</td><td>-3.00e-03</td><td>5.23</td><td>5.30</td></tr><tr><td>EPS (ours)</td><td>20</td><td>22.56</td><td>0.7999</td><td>0.1188</td><td>50.11</td><td>-6.24e-03</td><td>1.01e-03</td><td>5.24</td><td>5.50</td></tr><tr><td>EPS (ours)†</td><td>1</td><td>24.21</td><td></td><td>0.1265</td><td>77.97</td><td>-5.63e-03</td><td>4.09e-02</td><td>6.57</td><td>9.27</td></tr><tr><td rowspan="9">Gaussian deblur</td><td>DPS DAPS</td><td>250</td><td>25.89 27.32</td><td>0.8530 0.8494</td><td>0.0901</td><td>68.78 68.96</td><td>-5.78e-03 -6.80e-03</td><td>2.85e-02 5.00e-02</td><td>3.75</td><td>6.40</td></tr><tr><td>DDNM</td><td>100</td><td>31.27</td><td></td><td>0.0771</td><td>20.96</td><td></td><td>-5.45e-03</td><td>2.75</td><td>6.89</td></tr><tr><td>IIGDM</td><td>100</td><td></td><td>0.9454</td><td>0.0273</td><td></td><td>-6.91e-03</td><td></td><td>1.87</td><td>3.15</td></tr><tr><td>MPGD</td><td>100 100</td><td>30.18 19.88</td><td>0.9332 0.6716</td><td>0.0339</td><td>27.30 83.31</td><td>-6.85e-03</td><td>-8.48e-04</td><td>2.26</td><td>3.86</td></tr><tr><td></td><td></td><td></td><td></td><td>0.1770</td><td></td><td>2.72e-02</td><td>5.93e-02</td><td>7.59</td><td>7.84</td></tr><tr><td>Palette</td><td>100</td><td>30.47</td><td>0.9397</td><td>0.0286</td><td>21.63</td><td>-6.90e-03</td><td>-5.65e-03</td><td>1.91</td><td>3.06</td></tr><tr><td>EPS (ours) EPS (ours)</td><td>100</td><td>30.82 31.47</td><td>0.9408</td><td>0.0273</td><td>20.68</td><td>-6.91e-03</td><td>-5.73e-03</td><td>1.84</td><td>2.99</td></tr><tr><td>EPS (ours)†</td><td>20 1</td><td>30.86</td><td>0.9488 0.9503</td><td>0.0249 0.0369</td><td>20.36 31.39</td><td>-6.91e-03 -6.62e-03</td><td>-4.97e-03 2.35e-03</td><td>1.84</td><td>3.13</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>3.21</td><td>6.15</td></tr><tr><td rowspan="9">Motion deblur</td><td>DPS DAPS</td><td>250</td><td>27.85 27.18</td><td>0.8694</td><td>0.0752</td><td>70.24 53.53</td><td>-6.76e-03 -6.70e-03</td><td>4.68e-02 1.85e-02</td><td>3.29</td><td>7.20</td></tr><tr><td>DDNM</td><td>100</td><td></td><td>0.8703</td><td>0.0732</td><td>22.06</td><td></td><td></td><td>3.12 1.93</td><td>5.92 3.23</td></tr><tr><td>IIGDM</td><td>100 100</td><td>30.96 25.04</td><td>0.9411 0.8529</td><td>0.0297 0.0940</td><td>54.68</td><td>-6.91e-03 -6.23e-03</td><td>-5.32e-03 1.24e-02</td></table>

Figures 7 and 8 show qualitative reconstructions on FFHQ-64 and ImageNet-64 across the five inverse problems, comparing EPS against DPS, DAPS, DDNM, ΠGDM, MPGD, and Palette. Two example observations per task; numbers in the bottom-right corner of each panel are per-image PSNR. On FFHQ-64, EPS recovers facial structure (eyes, mouth, hairline) under aggressive random inpainting and box inpainting, and produces sharper texture and edge geometry on super-resolution and deblurring than the sampling-based and Palette baselines. On ImageNet-64 the same pattern holds on a more diverse class-conditional distribution, with EPS preserving operator-consistent structure where DPS, DAPS, and MPGD oversmooth or hallucinate texture inconsistent with the measurement.

GT  
Observation  
DPS  
DAPS  
DDNM  
PiGDM  
MPGD  
Palette  
(OPS)  
![](images/dfffb456b0d8ad440bb1e3559e4c5a8ad59d63006f3e3a549e4e748b759c9467.jpg)  
Figure 7: Qualitative reconstructions on FFHQ-64. Two example observations per task across the five inverse problems; numbers in the bottom-right corner of each panel are per-image PSNR. EPS recovers facial structure under aggressive random inpainting and box inpainting, and produces sharper texture and edge geometry on super-resolution and deblurring than the sampling-based and Palette baselines.

![](images/b5863df84ad7524aaf6718589e72d23dd7e95d4950b803a081a39368512ac717.jpg)  
Figure 8: Qualitative reconstructions on ImageNet-64. Same layout as Fig. 7. EPS preserves operator-consistent structure where DPS, DAPS, and MPGD oversmooth or hallucinate texture inconsistent with the measurement.

## D.8 Extreme Tasks 64×64

We test EPS on two extreme regimes that fall outside the main-text protocol: random inpainting with 95% of pixels missing (only 5% observed) and 16× super-resolution (a 4×4 low-resolution observation upsampled to 64×64). Both push the operator nullspace to occupy almost the entire signal space, so the prior must do most of the reconstruction work and the measurement-matching score is correspondingly noisier.

Table 7 reports this comparison on ImageNet-64 against DPS, DAPS, DDNM, ΠGDM, and MPGD; Palette is omitted because no Palette checkpoint was trained for these regimes. EPS at NFE=20 is the strongest method on perceptual and distributional metrics across both tasks (LPIPS, FID, MMD, CRPS), and is competitive on PSNR/SSIM with the strongest sampling-based baselines despite their having access to a full sampler trajectory. On 95% inpainting, EPS-20 reduces FID by roughly 25% over the best sampling-based baseline (ΠGDM at 195) and roughly 30% over DPS. On 16× super-resolution, the perceptual gap is smaller because the operator preserves only a single anchor pixel per 4×4 block, leaving little measurement signal in the pivot to exploit.

Table 7: EPS holds up under extreme operator nullspace. Quantitative comparison on ImageNet-64 in two extreme inverse-problem regimes: 95% random inpainting (only 5% of pixels observed) and 16× super-resolution (a 4×4 low-resolution observation upsampled to 64×64). All baselines use a 1-NFE-per-step Euler/DDIM sampler; EPS uses the EDM Euler sampler. Reported NFE equals the number of sampler iterations. Best in bold, second-best underlined; EPS rows highlighted in light pink. Palette is omitted because no Palette checkpoint was trained for these regimes.
<table><tr><td>Task</td><td>Method</td><td>NFE</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID↓</td><td>MMD-pix ↓</td><td>MMD-Inc ↓</td><td>CRPS-pix ↓</td><td>CRPS-Inc ↓</td></tr><tr><td rowspan="7">Inpaint (95% masked)</td><td>DPS</td><td>250</td><td>14.14</td><td>0.2101</td><td>0.5066</td><td>210.05</td><td>1.63e-02</td><td>2.48e-02</td><td>12.78</td><td>10.68</td></tr><tr><td>DAPS</td><td>100</td><td>14.20</td><td>0.1753</td><td>0.5962</td><td>329.19</td><td>1.08e-01</td><td>2.14e-01</td><td>15.96</td><td>14.77</td></tr><tr><td>DDNM</td><td>100</td><td>14.47</td><td>0.2316</td><td>0.5480</td><td>251.66</td><td>6.91e-02</td><td>7.34e-02</td><td>14.37</td><td>11.91</td></tr><tr><td>IIGDM</td><td>100</td><td>16.09</td><td>0.3015</td><td>0.4481</td><td>195.18</td><td>9.08e-03</td><td>1.50e-02</td><td>10.96</td><td>10.10</td></tr><tr><td>MPGD</td><td>100</td><td>13.87</td><td>0.2073</td><td>0.5632</td><td>267.41</td><td>7.01e-02</td><td>9.36e-02</td><td>15.15</td><td>12.34</td></tr><tr><td>EPS (ours)</td><td>100</td><td>17.85</td><td>0.4468</td><td>0.2813</td><td>145.17</td><td>-5.27e-03</td><td>-2.11e-03</td><td>8.38</td><td>7.93</td></tr><tr><td>EPS (ours)</td><td>20</td><td>18.41</td><td>0.4755</td><td>0.2712</td><td>144.18</td><td>-5.08e-03</td><td>-1.68e-03</td><td>8.34</td><td>8.00</td></tr><tr><td rowspan="7">Super-res (16×)</td><td>DPS</td><td>250</td><td>13.30</td><td>0.1171</td><td>0.5550</td><td>191.98</td><td>2.63e-03</td><td>7.74e-03</td><td>13.88</td><td>10.23</td></tr><tr><td>DAPS</td><td>100</td><td>12.70</td><td>0.1437</td><td>0.6439</td><td>267.84</td><td>7.64e-02</td><td>9.77e-02</td><td>17.51</td><td>12.47</td></tr><tr><td>DDNM</td><td>100</td><td>15.84</td><td>0.2295</td><td>0.5787</td><td>224.62</td><td>7.58e-03</td><td>5.73e-02</td><td>13.30</td><td>11.62</td></tr><tr><td>IIGDM</td><td>100</td><td>14.43</td><td>0.1703</td><td>0.5073</td><td>181.06</td><td>-2.85e-03</td><td>8.31e-03</td><td>12.71</td><td>9.72</td></tr><tr><td>MPGD</td><td>100</td><td>13.84</td><td>0.1875</td><td>0.6176</td><td>233.46</td><td>8.89e-02</td><td>5.67e-02</td><td>15.85</td><td>11.61</td></tr><tr><td>EPS (ours)</td><td>100</td><td>14.17</td><td>0.1585</td><td>0.5007</td><td>175.76</td><td>-3.10e-03</td><td>3.34e-04</td><td>12.71</td><td>9.60</td></tr><tr><td>EPS (ours)</td><td>20</td><td>14.86</td><td>0.1822</td><td>0.4948</td><td>180.14</td><td>-6.49e-04</td><td>3.63e-03</td><td>12.67</td><td>9.81</td></tr></table>

Figure 9 shows EPS reconstructions across six independent latent seeds on three 95%-inpainting and three 16× super-resolution observations. Samples agree on the broad spatial layout dictated by the few observed pixels but diverge sharply in fine structure and unobserved content (foreground identity, background texture, occluded geometry), which is the qualitative signature of a calibrated posterior in a regime where the operator nullspace dominates.

Observation  
GT  
EPS(seed 0)  
EPS(seed 1)  
EPS(seed 2)  
EPS(seed 3)  
EPS(seed 4)  
EPS(seed 5)  
![](images/5679d5810c8a4af5e603265607393192f9c3a0b70001f42b4c00d6ed915c910e.jpg)  
Figure 9: Posterior diversity under extreme operator nullspace. EPS reconstructions on ImageNet-64 across six independent latent seeds. Top three rows: 95% random inpainting (only 5% of pixels observed). Bottom three rows: 16× super-resolution (a 4×4 low-resolution observation upsampled to 64×64). Samples agree on the broad spatial layout consistent with the measurement but vary substantially in unobserved directions, illustrating that EPS produces genuinely distinct posterior samples rather than near-duplicates of a single conditional mean.

## D.9 OOD Mask-Density Experiments

We test how EPS and Palette generalize when the test-time mask density differs from training. Both checkpoints are trained on random inpainting at 70% masking and frozen; at evaluation we re-sample masks at five densities, ranging from 50% (easier than training) through 70% (in-distribution) to 90% (much harder than training). All results use NFE=100 with the EDM Euler sampler over 100 images × 10 seeds, with $\sigma _ { y } { = } 0 . 0 5$

Table 8 reports this comparison on ImageNet-64 (top) and FFHQ-64 (bottom). On ImageNet, EPS wins on every metric at every density except 80% (near-tie). On FFHQ, EPS dominates in- and neardistribution (50%-70%) but degrades faster than Palette on PSNR/SSIM at heavily-OOD densities (80%-90%); EPS still wins on the distributional metrics (MMD-pix, CRPS) at every density. The PSNR/SSIM crossover at high mask fraction is consistent with the closed-form pivot µ<sub>⋆</sub> extrapolating poorly when the operator shifts far from training at test time, since the precision weighting of the pivot is calibrated to a 70% mask under $\sigma _ { y } { = } 0 . 0 5$ and is mismatched when the actual mask density changes.

Table 8: OOD generalization across mask density. Both checkpoints are trained on random inpainting at 70% masking and frozen; evaluation re-samples masks at five densities (50%, 60%, 70% in-distribution, 80%, 90%) on the same eval $x _ { 0 }$ at $\sigma _ { y } { = } 0 . 0 5$ . ImageNet-64 (top): EPS wins on every metric at every density except 80% (near-tie). FFHQ-64 (bottom): EPS dominates in- and near-distribution but degrades faster than Palette on PSNR/SSIM at heavily-OOD densities, while still winning on MMD and CRPS at every density. All numbers at NFE=100 (EDM Euler). Best per row in bold; EPS rows highlighted in light pink.
<table><tr><td>Mask %</td><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID↓</td><td>MMD-pix ↓</td><td>MMD-Inc ↓</td><td>CRPS-pix ↓</td><td>CRPS-Inc ↓</td></tr><tr><td colspan="10">ImageNet 64×64</td></tr><tr><td rowspan="2">50%</td><td>Palette</td><td>27.10</td><td>0.8820</td><td>0.0608</td><td>55.12</td><td>-6.71e-03</td><td>-5.26e-03</td><td>2.87</td><td>4.41</td></tr><tr><td>EPS (ours)</td><td>27.88</td><td>0.8969</td><td>0.0545</td><td>51.79</td><td>-6.76e-03</td><td>-5.32e-03</td><td>2.69</td><td>4.31</td></tr><tr><td rowspan="2">60%</td><td>Palette</td><td>25.62</td><td>0.8415</td><td>0.0778</td><td>65.85</td><td>-6.63e-03</td><td>-5.01e-03</td><td>3.44</td><td>4.83</td></tr><tr><td>EPS (ours)</td><td>26.07</td><td>0.8526</td><td>0.0729</td><td>62.98</td><td>-6.66e-03</td><td>-5.10e-03</td><td>3.31</td><td>4.73</td></tr><tr><td rowspan="2">70% (in-dist.)</td><td>Palette</td><td>24.09</td><td>0.7869</td><td>0.1011</td><td>81.88</td><td>-6.50e-03</td><td>-4.41e-03</td><td>4.16</td><td>5.52</td></tr><tr><td>EPS (ours)</td><td>24.34</td><td>0.7948</td><td>0.0979</td><td>79.60</td><td>-6.52e-03</td><td>-4.51e-03</td><td>4.04</td><td>5.41</td></tr><tr><td rowspan="2">80%</td><td>Palette</td><td>22.29</td><td>0.7084</td><td>0.1399</td><td>98.49</td><td>-6.27e-03</td><td>-3.69e-03</td><td>5.14</td><td>6.06</td></tr><tr><td>EPS (ours)</td><td>22.30</td><td>0.7080</td><td>0.1403</td><td>98.91</td><td>-6.28e-03</td><td>-3.61e-03</td><td>5.08</td><td>6.04</td></tr><tr><td rowspan="2">90%</td><td>Palette</td><td>18.60</td><td>0.5135</td><td>0.2635</td><td>152.54</td><td>-3.54e-03</td><td>3.38e-03</td><td>8.04</td><td>8.33</td></tr><tr><td>EPS (ours)</td><td>19.06</td><td>0.5258</td><td>0.2456</td><td>144.18</td><td>-5.01e-03</td><td>6.27e-04</td><td>7.19</td><td>7.78</td></tr><tr><td colspan="2">FFHQ 64×64</td><td colspan="10"></td></tr><tr><td rowspan="2">50%</td><td>Palette EPS (ours)</td><td>27.95 29.64</td><td>0.9171 0.9409</td><td>0.0414</td><td>25.07</td><td>-6.47e-03 -6.86e-03</td><td>-5.32e-03</td><td>2.50</td><td>3.32</td></tr><tr><td></td><td></td><td></td><td>0.0307</td><td>22.92</td><td></td><td>-3.85e-03</td><td>2.19</td><td>3.34</td></tr><tr><td rowspan="2">60%</td><td>Palette EPS (ours)</td><td>27.11 28.05</td><td>0.9061 0.9206</td><td>0.0470</td><td>28.16</td><td>-6.70e-03</td><td>-5.11e-03</td><td>2.77</td><td>3.55</td></tr><tr><td></td><td></td><td></td><td>0.0393</td><td>26.19</td><td>-6.82e-03</td><td>-4.48e-03</td><td>2.58</td><td>3.49</td></tr><tr><td rowspan="2">70% (in-dist.)</td><td>Palette EPS (ours)</td><td>25.76</td><td>0.8809</td><td>0.0593</td><td>33.30</td><td>-6.71e-03</td><td>-4.76e-03</td><td>3.31</td><td>3.87</td></tr><tr><td></td><td>26.16</td><td>0.8879</td><td>0.0533</td><td>31.87</td><td>-6.74e-03</td><td>-5.00e-03</td><td>3.16</td><td>3.75</td></tr><tr><td rowspan="2">80%</td><td>Palette</td><td>23.78</td><td>0.8321</td><td>0.0826</td><td>40.23</td><td>-6.42e-03</td><td>-3.44e-03</td><td>4.28</td><td>4.51</td></tr><tr><td>EPS (ours)</td><td>23.57</td><td>0.8250</td><td>0.0838</td><td>40.83</td><td>-6.53e-03</td><td>-3.58e-03</td><td>4.19</td><td>4.45</td></tr><tr><td rowspan="2">90%</td><td>Palette EPS (ours)</td><td>18.34</td><td>0.6023 0.5683</td><td>0.2286</td><td>83.35 90.78</td><td>4.20e-03 -1.54e-03</td><td>2.17e-02 3.45e-02</td><td>8.68</td><td>7.09</td></tr><tr><td></td><td>17.82</td><td></td><td>0.2284</td><td></td><td></td><td></td><td>7.71</td><td>7.08</td></tr></table>

Figure 10 shows qualitative reconstructions from EPS and Palette across the five mask densities on both datasets. The visual gap between the two methods is largest in the in-distribution regime and narrows at the extremes: at 50% masking both methods recover most of the image structure, while at 90% both methods struggle and the reconstructions diverge sharply from the ground truth.

![](images/3baa7db024c9c81625b66e199f73eef974ba962968e3ce5b8697aba238886d18.jpg)  
Figure 10: Qualitative OOD generalization across mask density. EPS and Palette reconstructions on ImageNet-64 and FFHQ-64 across five mask densities (50%, 60%, 70% in-distribution, 80%, 90%). Both checkpoints were trained on 70% masking and frozen at evaluation. EPS preserves operator-consistent structure best in- and near-distribution; degradation at heavily-OOD densities (80%, 90%) is consistent with the pivot $\mu _ { \star }$ being calibrated to a 70% mask.

## D.10 Additional 256×256 Results

Table 9 reports EPS at NFE∈ {1, 20, 100} against sampling-based and training-based baselines on ImageNet 256×256 across five inverse problems. EDM-DDPM++ architecture is used, and trained from scratch on each task (no pretrained backbone). Baseline numbers are taken directly from the DAPS paper [22]; we did not re-run them under our protocol, and distributional metrics (CRPS, MMD) are not reported because they are not available in the source paper. EPS leads on both inpainting tasks and on Gaussian deblurring, particularly on perceptual metrics (LPIPS, FID), and is competitive with DAPS on motion deblurring (taking second place on SSIM, LPIPS, and FID at NFE=20). DAPS retains an edge on 4× super-resolution. As at 64×64, the 1-NFE EPS row is consistently the strongest EPS variant on PSNR/SSIM, mirroring the high-noise posterior-mean check of Section 3.5.

Table 9: ImageNet 256×256 results. EPS vs. sampling-based and training-based baselines on five linear inverse problems at 256×256 resolution. Baseline numbers are taken from the DAPS paper [22]. Best in bold, second-best underlined; EPS rows highlighted in light pink. † The NFE=1 row applies a single direct Tweedie evaluation $D _ { \theta } ( \mu _ { \star } , \sigma _ { \operatorname* { m a x } } )$ , returning the conditional posterior mean rather than a posterior sample.
<table><tr><td>Task</td><td>Method</td><td>PSNR↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID↓</td></tr><tr><td rowspan="10">Super-res (4×)</td><td>DAPS</td><td>25.89</td><td>0.694</td><td>0.276</td><td>83.57</td></tr><tr><td>DPS</td><td>21.13</td><td>0.489</td><td>0.361</td><td>106.32</td></tr><tr><td>DDRM</td><td>22.62</td><td>0.521</td><td>0.324</td><td>103.85</td></tr><tr><td>DDNM</td><td>23.96</td><td>0.604</td><td>0.475</td><td>98.62</td></tr><tr><td>DCDP</td><td></td><td></td><td></td><td></td></tr><tr><td>FPS-SMC</td><td>24.82</td><td>0.703</td><td>0.313</td><td>97.51</td></tr><tr><td>DiffPIR</td><td>23.18</td><td></td><td>0.371</td><td>106.32</td></tr><tr><td>EPS, NFE=100</td><td>19.60</td><td>0.520</td><td>0.294</td><td>138.55</td></tr><tr><td>EPS, NFE=20</td><td>20.32</td><td>0.561</td><td>0.277</td><td>130.66</td></tr><tr><td>EPS, NFE=1†</td><td>22.65</td><td>0.671</td><td>0.254</td><td>116.82</td></tr><tr><td rowspan="10">Gaussian deblur</td><td>DAPS</td><td>26.15</td><td>0.684</td><td>0.253</td><td>75.68</td></tr><tr><td>DPS</td><td>20.31</td><td>0.598</td><td>0.397</td><td>116.42</td></tr><tr><td>DDRM</td><td>21.26</td><td>0.564</td><td>0.443</td><td>146.89</td></tr><tr><td>DDNM</td><td>28.06</td><td>0.703</td><td>0.278</td><td>81.43</td></tr><tr><td>DCDP</td><td></td><td></td><td></td><td></td></tr><tr><td>FPS-SMC</td><td>23.91</td><td>0.601</td><td>0.387</td><td>91.72</td></tr><tr><td>DiffPIR</td><td>22.80</td><td></td><td>0.355</td><td>93.36</td></tr><tr><td>EPS, NFE=100</td><td>24.34</td><td>0.640</td><td>0.224</td><td>72.38</td></tr><tr><td>EPS, NFE=20</td><td>24.92</td><td>0.666</td><td>0.222</td><td>71.31</td></tr><tr><td>EPS, NFE=1†</td><td>25.90</td><td>0.704</td><td>0.249</td><td>100.10</td></tr><tr><td rowspan="8">Motion deblur</td><td>DAPS</td><td>27.86</td><td></td><td>0.196</td><td>61.83</td></tr><tr><td>DPS</td><td>18.96</td><td>0.766 0.629</td><td>0.423</td><td>137.81</td></tr><tr><td>DCDP</td><td></td><td></td><td></td><td></td></tr><tr><td>FPS-SMC</td><td>24.52</td><td>0.647</td><td>0.326</td><td>87.43</td></tr><tr><td>DiffPIR</td><td>24.01</td><td></td><td>0.366</td><td>94.63</td></tr><tr><td>EPS, NFE=100</td><td>23.65</td><td>0.616</td><td>0.254</td><td>79.79</td></tr><tr><td>EPS, NFE=20</td><td>24.39</td><td>0.655</td><td>0.240</td><td>75.52</td></tr><tr><td>EPS, NFE=1†</td><td>24.12</td><td>0.647</td><td>0.318</td><td>137.59</td></tr></table>

<table><tr><td>Task</td><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID↓</td></tr><tr><td rowspan="10">Inpaint (box)</td><td>DAPS</td><td>21.43</td><td>0.725</td><td>0.214</td><td>109.85</td></tr><tr><td>DPS</td><td>18.94</td><td>0.722</td><td>0.257</td><td>126.52</td></tr><tr><td>DDRM</td><td>18.63</td><td>0.733</td><td>0.254</td><td>116.37</td></tr><tr><td>DDNM</td><td>21.64</td><td>0.748</td><td>0.319</td><td>103.97</td></tr><tr><td>DCDP</td><td></td><td></td><td></td><td></td></tr><tr><td>FPS-SMC</td><td>22.16</td><td>0.726</td><td>0.208</td><td>111.58</td></tr><tr><td>EPS, NFE=100</td><td>19.57</td><td>0.785</td><td>0.154</td><td>100.30</td></tr><tr><td>EPS, NFE=20</td><td>20.12</td><td>0.795</td><td>0.149</td><td>94.22</td></tr><tr><td>EPS, NFE=1†</td><td>22.13</td><td>0.809</td><td>0.177</td><td>113.50</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="7">Inpaint (random)</td><td>DAPS</td><td>28.44</td><td>0.775</td><td>0.135</td><td>54.25</td></tr><tr><td>DPS</td><td>23.52</td><td>0.745</td><td>0.297</td><td>87.53</td></tr><tr><td>DDNM</td><td>31.16</td><td>0.841</td><td>0.191</td><td>63.84</td></tr><tr><td>DCDP</td><td></td><td></td><td></td><td></td></tr><tr><td>FPS-SMC</td><td>24.52</td><td>0.701</td><td>0.316</td><td>79.12</td></tr><tr><td>EPS, NFE=100</td><td>27.26</td><td>0.817</td><td>0.113</td><td>21.39</td></tr><tr><td>EPS, NFE=20 EPS, NFE=1†</td><td>27.90 29.73</td><td>0.836 0.876</td><td>0.104 0.103</td><td>20.17 24.03</td></tr></table>

Figure 11 shows qualitative reconstructions of EPS on ImageNet 256×256 across the five inverse problems, with one example observation per task. EPS recovers sharp object structure under aggressive random inpainting and box inpainting, and produces texture and edge geometry consistent with the measurement on super-resolution and deblurring at this higher resolution.

![](images/6fc72d95671776786c08c7d70657f5b5a62df876e5dfaf4d5b90707a0daccf5f.jpg)  
Figure 11: Qualitative reconstructions on ImageNet-256. EPS reconstructions across the five inverse problems on ImageNet 256×256. Numbers in the bottom-right corner of each panel are perimage PSNR. EPS preserves operator-consistent structure under aggressive inpainting and produces sharp texture and edges on super-resolution and deblurring at this higher resolution.

## D.11 Palette vs. EPS at Matched NFE

The one-step posterior-mean check in Section D.6 showed that a single Tweedie call recovers most of the multi-step PSNR/SSIM gain. Here we ask the matched question for Palette: does the same one-step shortcut close the gap to EPS? Concretely, we evaluate Palette $[ x _ { t } , y , t ]$ and EPS $[ \mu _ { \star } , y , t ]$ at $\mathrm { N F E } { = } 1$ (a single direct denoiser call at $\sigma _ { \mathrm { m a x } } { = } 8 0$ , no sampler loop) and at NFE=100 (EDM Euler sampler), on both datasets and all five tasks.

Table 10 reports this comparison on FFHQ-64 (top) and ImageNet-64 (bottom). Two patterns dominate. First, at NFE=1 the two methods coincide, as predicted by Observation 4: at $\sigma _ { \mathrm { m a x } }$ the noisy state carries vanishing information about $x _ { 0 } .$ , so both the Palette input $( x _ { t } , y )$ and the EPS input $( \mu _ { \star } , y )$ are informationally equivalent to y alone, and both networks estimate $\mathbb { E } [ x _ { 0 } | y ] ;$ the small differences here are training noise rather than methodological difference. Second, at NFE=100 the EPS pivot becomes informative and EPS dominates Palette across distortion and distributional metrics on every task. Both 1-NFE rows trade perceptual quality for distortion: PSNR/SSIM jump sharply because the conditional mean is the MMSE-optimal estimator under squared error, but FID, LPIPS, CRPS, and MMD all degrade because no actual sample is produced.

Table 10: Palette vs. EPS at NFE=1 and $\mathbf { N F E { = } 1 0 0 } .$ . One-step (single Tweedie call at $\sigma _ { \mathrm { m a x } } { = } 8 0 )$ vs. 100-step (EDM Euler) generation for Palette $[ x _ { t } , y , t ]$ and EPS $[ \mu _ { \star } , y , t ]$ on FFHQ-64 (top) and ImageNet-64 (bottom). $\mathbf { A t } \mathbf { N F E { = } } 1$ , the two methods receive nearly identical inputs since µ<sub>⋆</sub> ≈ x<sub>t</sub> at $\sigma _ { \mathrm { m a x } } ;$ at ${ \mathrm { N F E } } { = } 1 0 0$ the pivot becomes informative and EPS dominates across distortion and distributional metrics. Best in bold, second-best underlined; EPS rows highlighted in light pink.
<table><tr><td>Task</td><td>Method</td><td>PSNR↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID↓</td><td>MMD-pix ↓</td><td>MMD-Inc ↓</td><td>CRPS-pix ↓</td><td>CRPS-Inc ↓</td></tr><tr><td colspan="10">FFHQ 64×64 Palette (NFE=100) 0.8590</td></tr><tr><td>Average</td><td>Palette (NFE=1) EPS (NFE=1) EPS (NFE=100)</td><td>26.03 27.77 27.34 26.69</td><td>0.8920 0.8887 0.8661</td><td>0.0626 0.0638 0.0699 0.0590</td><td>31.50 41.93 46.30 29.94</td><td>-6.6e-03 -6.4e-03 -6.3e-03 -6.7e-03</td><td>-4.7e-03 8.0e-03 1.1e-02 -4.9e-03</td><td>3.43 4.36 4.67 3.24</td><td>3.86 6.21 6.91 3.73</td></tr><tr><td>Random inpaint</td><td>Palette (NFE=100) Palette (NFE=1) EPS (NFE=1) EPS (NFE=100)</td><td>25.76 27.58 27.90 26.16</td><td>0.8809 0.9095 0.9148 0.8879</td><td>0.0593 0.0551 0.0549 0.0533</td><td>33.30 39.20 39.71 31.87</td><td>-6.7e-03 -6.5e-03 -6.6e-03 -6.7e-03</td><td>-4.8e-03 1.1e-03 2.2e-03 -5.0e-03</td><td>3.31 4.24 3.94 3.16</td><td>3.87 5.96 5.75 3.75</td></tr><tr><td>Box inpaint</td><td>Palette (NFE=100) Palette (NFE=1) EPS (NFE=1) EPS (NFE=100)</td><td>24.18 26.14 26.01 24.23</td><td>0.8426 0.8727 0.8706 0.8448</td><td>0.0577 0.0639 0.0644 0.0567</td><td>25.09 32.60 32.90 24.74</td><td>-6.6e-03 -6.3e-03 -6.4e-03 -6.6e-03</td><td>-5.3e-03 -1.2e-03 -8.5e-04 -5.4e-03</td><td>4.02 5.01 5.05 4.01</td><td>3.47 5.51 5.62 3.45</td></tr><tr><td>Super-res (4×)</td><td>Palette (NFE=100) Palette (NFE=1) EPS (NFE=1) EPS (NFE=100)</td><td>21.95 24.23 24.21 21.96</td><td>0.7220 0.8009 0.7999 0.7232</td><td>0.1273 0.1255 0.1265 0.1262</td><td>49.28 77.61 77.97 49.29</td><td>-6.3e-03 -5.6e-03 -5.6e-03 -6.3e-03</td><td>-3.0e-03 4.1e-02 4.1e-02 -3.0e-03</td><td>5.22 6.56 6.57 5.23</td><td>5.29 9.31 9.27 5.30</td></tr><tr><td>Gaussian deblur</td><td>Palette (NFE=100) Palette (NFE=1) EPS (NFE=1) EPS (NFE=100)</td><td>30.47 32.04 30.86 30.82</td><td>0.9397 0.9529 0.9503 0.9408</td><td>0.0286 0.0285 0.0369</td><td>21.63 24.49 31.39 20.68</td><td>-6.9e-03 -6.8e-03 -6.6e-03</td><td>-5.6e-03 -1.9e-03 2.4e-03</td><td>1.91 2.40 3.21 1.84</td><td>3.06 4.61 6.15 2.99</td></tr><tr><td>Motion deblur</td><td>Palette (NFE=100) Palette (NFE=1) EPS (NFE=1) EPS (NFE=100)</td><td>27.79 28.87 27.73 30.27</td><td>0.9099 0.9240 0.9078</td><td>0.0273 0.0404 0.0461 0.0669</td><td>28.23 35.76 49.55</td><td>-6.9e-03 -6.8e-03 -6.6e-03 -6.3e-03</td><td>-5.7e-03 -5.0e-03 1.4e-03 1.2e-02</td><td>2.67 3.59 4.60</td><td>3.59 5.68 7.76</td></tr><tr><td></td><td></td><td></td><td>0.9339</td><td>0.0311</td><td>23.10 ImageNet 64×64</td><td>-6.9e-03</td><td>-5.4e-03</td><td>1.94</td><td>3.18</td></tr><tr><td>Average</td><td>Palette (NFE=100) Palette (NFE=1) EPS (NFE=1) EPS (NFE=100)</td><td>24.32 26.73 25.67 24.53</td><td>0.7673 0.8289 0.8165</td><td>0.1124 0.1215 0.1318</td><td>82.57 104.38 110.95 81.46</td><td>-6.4e-03 -6.1e-03 -5.6e-03 -6.4e-03</td><td>-4.4e-03 4.3e-03 4.5e-03 -4.4e-03</td><td>4.35 5.35 5.91 4.27</td><td>5.54 9.37 9.96 5.48</td></tr><tr><td>Random inpaint</td><td>Palette (NFE=100) Palette (NFE=1) EPS (NFE=1) EPS (NFE=100)</td><td>24.09 26.50 26.60 24.34</td><td>0.7869 0.8550 0.8580 0.7948</td><td>0.1011 0.0926 0.0933</td><td>81.88 91.79 88.59 79.60</td><td>-6.5e-03 -6.4e-03 -6.1e-03 -6.5e-03</td><td>-4.4e-03 4.3e-04 -4.8e-04 -4.5e-03</td><td>4.16 5.16 4.98 4.04</td><td>5.52 8.90 8.32 5.41</td></tr><tr><td>Box inpaint</td><td>Palette (NFE=100) Palette (NFE=1) EPS (NFE=1) EPS (NFE=100)</td><td>21.12 23.60 23.60 21.24</td><td>0.7541 0.7926 0.7908</td><td>0.1218 0.1539 0.1514</td><td>92.73 131.54 129.11</td><td>-6.1e-03 -5.5e-03 -5.6e-03</td><td>-4.1e-03 8.7e-03 7.3e-03</td><td>5.92 7.19 7.16</td><td>5.93 10.67 10.38 5.84</td></tr><tr><td>Super-res (4×)</td><td>Palette (NFE=100) Palette (NFE=1) EPS (NFE=1)</td><td>20.24 22.79 22.78</td><td>0.7569 0.5364 0.6538 0.6530</td><td>0.1196 0.2220 0.2452 0.2455</td><td>91.07 128.76 185.15 182.92</td><td>-6.1e-03 -5.9e-03 -4.9e-03 -5.0e-03</td><td>-4.2e-03 -2.8e-03 1.9e-02 2.0e-02</td><td>5.87 6.50 8.05 8.06</td><td>7.33 13.63 13.47</td></tr><tr><td>Gaussian deblur</td><td>EPS (NFE=100) Palette (NFE=100) Palette (NFE=1) EPS (NFE=1)</td><td>20.25 29.15 31.73 28.82</td><td>0.5369 0.9010 0.9407 0.9194</td><td>0.2207 0.0491 0.0441 0.0606</td><td>128.80 46.62 43.40 56.53</td><td>-5.9e-03 -6.8e-03 -6.8e-03 -5.1e-03</td><td>-2.8e-03 -5.6e-03 -4.2e-03</td><td>6.52 2.26 2.74</td><td>7.35 4.11 6.15</td></tr><tr><td></td><td>EPS (NFE=100) Palette (NFE=100)</td><td>29.18 27.02</td><td>0.9015 0.8582</td><td>0.0486 0.0680</td><td>46.55 62.86</td><td>-6.8e-03 -6.7e-03</td><td>-3.5e-03 -5.6e-03 -5.0e-03</td><td>4.09 2.25 2.93</td><td>7.73 4.11 4.81</td></tr><tr><td>Motion deblur</td><td>Palette (NFE=1) EPS (NFE=1) EPS (NFE=100)</td><td>29.04 26.56 27.62</td><td>0.9023 0.8613 0.8661</td><td>0.0715 0.1079 0.0647</td><td>70.01 97.59 61.29</td><td>-6.7e-03 -6.0e-03 -6.8e-03</td><td>-2.3e-03 -7.3e-04 -5.1e-03</td><td>3.63 5.25 2.69</td><td>7.51 9.91 4.70</td></tr></table>

## D.12 Runtime Analysis

We measure single-image wall-clock sampling latency on ImageNet 64×64 at NFE=100 with batch size 1. All methods use the same EDM-ADM [17] denoiser checkpoint (edm-imagenet-64x64-cond-adm.pkl, ∼296M parameters) on a single NVIDIA B200 GPU, with class labels set to the evaluation-set ground-truth ImageNet-1k classes. Each cell in Table 11 is the mean of five independent sampling runs after two warm-up runs that amortise CUDA kernel JIT and cuDNN auto-tuning. All methods use the Euler ODE schedule (second\_order=False) so NFE equals the number of sampler steps; for DAPS we set annealing\_steps=20 and ode\_steps=5 so total NFE matches.

The dominant cost in all methods is the U-Net forward, and for DPS and ΠGDM the U-Net backward as well. EDM (uncond.) runs the bare pretrained denoiser with no measurement-aware updates and is task-independent. DPS and ΠGDM require a backward pass per step (likelihood gradient / Jacobian-vector product), $\sim 2 . 2 \times$ the EDM unconditional cost. DAPS runs a nested ODE rollout plus Langevin correction at each annealing step, ∼ 2.0× unconditional cost. DDNM and MPGD (∼ 1.1×) add only a closed-form nullspace or manifold projection on top of a single denoiser forward. EPS is the fastest sampler in the comparison: at batch 1 it runs $\sim 0 . 8 \times$ the wall-clock of EDM unconditional. EPS’s freshly constructed preconditioning wrapper avoids deserialisation overhead present in the pretrained EDMPrecond pickle, while the structured solve for $\mu _ { \star }$ (FFT for deblurring; element-wise for inpainting and super-resolution) is sub-millisecond. EPS runtime is essentially task-independent: all five tasks land within ±0.04 s of the average. Palette shares EPS’s architecture and per-step forward cost, so its runtime is well approximated by the EPS row. At larger batch sizes the per-image gap closes (at batch 8, EPS is only ∼ 3.5% slower than EDM unconditional), so EPS is the right pick for low-latency single-image inference and is essentially free vs. EDM unconditional in throughput terms.

Table 11: EPS matches the bare denoiser in wall-clock cost. Per-image sampling latency (s, lower is better) on ImageNet-64 at NFE=100, batch size 1, on a single B200 GPU. Among methods that solve the inverse problem, EPS is fastest on every task and on average, at essentially the same cost as the bare EDM unconditional sampler (1.006×). Sampling-based baselines that require a backward pass (DPS, ΠGDM) are $\sim 2 . 3 \times$ slower; nested-rollout methods (DAPS) are $\sim 2 . 2 \times$ slower. † EDM (uncond.) is shown as a reference: it runs the bare pretrained denoiser without any measurement-aware update, so it does not actually solve the inverse problem.
<table><tr><td>Method</td><td>Inpaint (Random)</td><td>Inpaint (Box)</td><td>Super-res (4×)</td><td>Gaussian deblur</td><td>Motion deblur</td><td>Avg</td><td>× EDM</td></tr><tr><td>EDM (uncond.)†</td><td>1.87</td><td>1.87</td><td>1.87</td><td>1.87</td><td>1.87</td><td>1.87</td><td>1.00×</td></tr><tr><td>DPS</td><td>4.61</td><td>4.17</td><td>4.05</td><td>4.14</td><td>4.42</td><td>4.28</td><td>2.29×</td></tr><tr><td>DAPS</td><td>3.98</td><td>4.02</td><td>3.99</td><td>4.19</td><td>4.20</td><td>4.07</td><td>2.18×</td></tr><tr><td>DDNM</td><td>2.52</td><td>2.50</td><td>2.49</td><td>2.51</td><td>2.51</td><td>2.51</td><td>1.34×</td></tr><tr><td>IIGDM</td><td>4.55</td><td>4.55</td><td>4.53</td><td>4.54</td><td>4.55</td><td>4.54</td><td>2.43×</td></tr><tr><td>MPGD</td><td>2.51</td><td>2.51</td><td>2.52</td><td>2.53</td><td>2.54</td><td>2.52</td><td>1.35×</td></tr><tr><td>EPS (ours)</td><td>1.87</td><td>1.87</td><td>1.88</td><td>1.89</td><td>1.89</td><td>1.88</td><td>1.006×</td></tr></table>

(seed 2)

(seed 3)

EPS(seed 0)

## D.13 Posterior diversity.

Figures 12 and 13 show four EPS reconstructions of the same observation drawn with independent latent seeds, alongside the ground truth, on box inpainting and 4× super-resolution. Samples agree on observed structure while differing in the unobserved directions (skin texture, hair detail, background, occluded foreground content) - the qualitative signature of a calibrated posterior under a non-trivial operator nullspace.

EPS  
EPS(seed 1)  
EPS  
GT  
![](images/be549b18e2b6131994894d262762e6210a64458eb0bf8bdc96ec29c48aabd344.jpg)  
Figure 12: Posterior diversity from EPS on FFHQ-64. Four reconstructions per observation drawn with independent latent seeds, alongside the ground truth, on box inpainting and 4× super-resolution. Samples agree on observed structure while differing in unobserved directions (skin texture, hair detail, background) — the qualitative signature of a calibrated posterior under a non-trivial operator nullspace.

EPS(seed 0)  
EPS(seed 1)  
EPS(seed 2)  
EPS(seed 3)  
GT  
![](images/1ba7cc4d59e9c81a5515ef1ea61aa435d3b63d3cab01be6bff676857b52a95d8.jpg)  
Figure 13: Posterior diversity from EPS on ImageNet-64. Same layout as Fig. 12. Diversity is concentrated in the operator nullspace: occluded foreground content varies under box inpainting, while sharp high-frequency detail varies under 4× super-resolution.

## D.14 Sampling Budget

Figure 14 compares EPS reconstructions at NFE=1, 20, and 100 on the same observation across all five tasks, on both ImageNet-64 (left) and FFHQ-64 (right). The NFE=1 column is the deterministic high-noise posterior-mean limit (Section 3.5), which is MMSE-optimal in pixel space and yields the highest per-image PSNR. The NFE=20 and ${ \mathrm { N F E } } { = } 1 0 0$ columns target posterior samples and trade pointwise fidelity for distributional sharpness, in line with the perception-distortion pattern visible in Table 5.

![](images/d637eead15e402ed935e1c61de3d336f6b966a9abb958d6f382b835e678ca203.jpg)  
Figure 14: EPS reconstructions at varying sampling budgets. ImageNet-64 (left) and FFHQ-64 (right) across the five inverse problems. For each observation, columns show NFE=1, 20, and 100. The NFE=1 column is the deterministic high-noise posterior-mean limit (Section 3.5), MMSEoptimal in pixel space; NFE=20 and NFE=100 target posterior samples and trade pointwise fidelity for distributional sharpness.

## E Baseline Configurations

Table 12 reports per-task hyperparameters for the sampling-based and training-based baselines (DPS, DAPS, DDNM, ΠGDM, MPGD, Palette), including NFE, step size or guidance scale, noise schedule, projection or correction rule, and additional notes. Each baseline is tuned on a disjoint validation split, and the values reported are those used in the main and appendix tables.

Table 12: Baseline hyperparameter configurations. Per-task hyperparameters for the samplingbased (DPS, DAPS, DDNM, ΠGDM, MPGD) and training-based (Palette) baselines. NFE is the number of denoiser forward passes per image (DAPS NFE=annealing×ode Euler steps). All samplers use the EDM VE noise schedule with $\sigma _ { \mathrm { m a x } } { = } 8 0 , \sigma _ { \mathrm { m i n } } { = } 0 . 0 0 2$ $\rho { = } 7$ unless noted. Observation noise is $\sigma _ { y } { = } 0 . 0 5$ for every cell. Identical settings are used on ImageNet-64 and FFHQ-64; only the underlying denoiser changes.
<table><tr><td>Task</td><td>Method</td><td>NFE</td><td>Step size / guidance</td><td>Noise schedule</td><td>Projection / correction</td><td>Notes</td></tr><tr><td>Random inpaint</td><td>DPS</td><td>250</td><td> $\zeta ^ { \prime } { = } 5 . 0$ </td><td>EDM VE</td><td>likelihood grad, ζ′/∥r||</td><td>Euler,  $S _ { \mathrm { c h u m } } { = } 0$ </td></tr><tr><td>Random inpaint</td><td>DAPS</td><td>500</td><td> $\mathrm { m c m c 5 } \times 1 0 ^ { - 4 }$ </td><td>EDM VE</td><td>nested ODE + Langevin</td><td>annealing=100, ode=5, mcmc=100, Euler</td></tr><tr><td>Random inpaint</td><td>DDNM</td><td>100</td><td></td><td>EDM VE</td><td>null-space t=σ2/(σt+σ2)</td><td>DDIM-VE, η=0.85</td></tr><tr><td>Random inpaint</td><td>IIGDM</td><td>100</td><td> $\zeta = 1 . 0$ </td><td>EDM VE</td><td>VJP guidance</td><td> $\mathrm { D D I M - V E } , \eta { = } 1 . 0$ </td></tr><tr><td>Random inpaint</td><td>MPGD</td><td>100</td><td> $\zeta = 2 0 . 0$ </td><td>EDM VE</td><td>manifold projection</td><td>DDIM-VE, η=0.85</td></tr><tr><td>Random inpaint</td><td>Palette</td><td>100</td><td></td><td>EDM VE</td><td></td><td>training-based; per-task EPS backbone</td></tr><tr><td>Box inpaint</td><td>DPS</td><td>250</td><td> $\zeta ^ { \prime } { = } 1 0 . 0$ </td><td>EDM VE</td><td>likelihood grad, ζ′/∥|r∥|</td><td>Euler,  $S _ { \mathrm { c h u m } } { = } 0$ </td></tr><tr><td>Box inpaint</td><td>DAPS</td><td>500</td><td> $\mathrm { m c m c } 5 \times 1 0 ^ { - 4 }$ </td><td>EDM VE</td><td>nested ODE + Langevin</td><td>annealing=100, ode=5, mcmc=100, Euler</td></tr><tr><td>Box inpaint</td><td>DDNM</td><td>100</td><td></td><td>EDM VE</td><td>null-space λt=σ2/(σ2 +σ2)</td><td>DDIM-VE, η=0.85</td></tr><tr><td>Box inpaint</td><td>ΠIGDM</td><td>100</td><td> $\zeta = 1 . 0$ </td><td>EDM VE</td><td>VJP guidance</td><td> $\mathrm { D D I M - V E } , \eta { = } 1 . 0$ </td></tr><tr><td>Box inpaint Box inpaint</td><td>MPGD Palette</td><td>100</td><td> $\zeta { = } 1 5 . 0$ </td><td>EDM VE</td><td>manifold projection</td><td> $\mathrm { D D I M - V E } , \boldsymbol { \eta } { = } 0 . 8 5$ </td></tr><tr><td></td><td></td><td>100</td><td></td><td>EDM VE</td><td></td><td>training-based; per-task EPS backbone</td></tr><tr><td>Super-res (4×)</td><td>DPS</td><td>250</td><td> $\zeta ^ { \prime } { = } 1 0 . 0$ </td><td>EDM VE</td><td>likelihood grad, ζ′/∥r||</td><td> $\mathrm { E u l e r } , S _ { \mathrm { c h u m } } { = } 0$ </td></tr><tr><td>Super-res (4×)</td><td>DAPS</td><td>500</td><td> $\mathrm { m c m c 9 \times 1 0 ^ { - 4 } }$ </td><td> $\mathrm { E D M V E } , \sigma _ { \mathrm { m a x } } = 3 0 , \sigma _ { \mathrm { m i n } } = 0 . 1$ </td><td>nested ODE + Langevin</td><td>annealing=100, ode=5, mcmc=100, Euler</td></tr><tr><td>Super-res (4×)</td><td>DDNM</td><td>100</td><td></td><td>EDM VE</td><td> $\mathrm { n u l l - s p a c e } ~ \lambda _ { t } { = } \sigma _ { t } ^ { 2 } / ( \sigma _ { t } ^ { 2 } { + } \sigma _ { y } ^ { 2 } )$ </td><td>DDIM-VE, η=0.85, A†=nearest upsample</td></tr><tr><td>Super-res (4×)</td><td>IIGDM</td><td>100</td><td> $\zeta = 1 . 0$ </td><td>EDM VE</td><td>VJP guidance</td><td> $\mathrm { D D I M - V E } , \eta { = } 1 . 0$ </td></tr><tr><td>Super-res (4×) Super-res (4×)</td><td>MPGD Palette</td><td>100 100</td><td> $\zeta { = } 3 0 . 0$ </td><td>EDM VE EDM VE</td><td>manifold projection</td><td> $\mathrm { D D I M - V E } , \boldsymbol { \eta } { = } 0 . 8 5$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>training-based; per-task EPS backbone</td></tr><tr><td>Gaussian deblur</td><td>DPS</td><td>250</td><td> $\zeta ^ { \prime } { = } 1 . 5$ </td><td>EDM VE</td><td>likelihood grad, ζ′/||</td><td>Euler  $\therefore S _ { \mathrm { c h u m } } { = } 0$ </td></tr><tr><td>Gaussian deblur</td><td>DAPS</td><td>500</td><td> $\mathrm { m c m c 9 \times 1 0 ^ { - 4 } }$ </td><td> $\mathrm { E D M V E } , \sigma _ { \mathrm { m a x } } = 3 0 , \sigma _ { \mathrm { m i n } } = 0 . 1$ </td><td>nested ODE + Langevin</td><td>annealing=100, ode=5, mcmc=100, Euler</td></tr><tr><td>Gaussian deblur</td><td>DDNM ΠIGDM</td><td>100</td><td></td><td>EDM VE</td><td>per-freq Wiener correction</td><td>DDIM-VE, η=0.85, Wiener e=10−3</td></tr><tr><td>Gaussian deblur Gaussian deblur</td><td>MPGD</td><td>100 100</td><td> $\zeta = 1 . 0$   $\zeta = 5 . 0$ </td><td>EDM VE</td><td>VJP guidance</td><td> $\mathrm { D D I M - V E } , \eta { = } 1 . 0$ </td></tr><tr><td>Gaussian deblur</td><td>Palette</td><td>100</td><td></td><td>EDM VE EDM VE</td><td>manifold projection</td><td>DDIM-VE, η=0.85 training-based; per-task EPS backbone</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Motion deblur Motion deblur</td><td>DPS DAPS</td><td>250</td><td> $\zeta ^ { \prime } { = } 2 . 5$ </td><td>EDM VE</td><td>likelihood grad, ζ′/∥r||</td><td> $\mathrm { E u l e r } , S _ { \mathrm { c h u m } } { = } 0$ </td></tr><tr><td></td><td></td><td>500</td><td>mcmc 5×10−4</td><td>EDM VE</td><td>nested ODE + Langevin</td><td>annealing=100, ode=5, mcmc=100, Euler</td></tr><tr><td>Motion deblur</td><td>DDNM IIGDM</td><td>100</td><td></td><td>EDM VE</td><td>per-freq Wiener correction</td><td>DDIM-VE, η=0.85, Wiener ε=10−3</td></tr><tr><td>Motion deblur Motion deblur</td><td>MPGD</td><td>100 100</td><td>ζ=1.0</td><td>EDM VE</td><td>VJP guidance</td><td> $\mathrm { D D I M - V E } , \eta { = } 1 . 0$ </td></tr><tr><td>Motion deblur</td><td>Palette</td><td>100</td><td>ζ=9.0</td><td>EDM VE EDM VE</td><td>manifold projection</td><td> $\mathrm { D D I M - V E } , \boldsymbol { \eta } { = } 0 . 8 5$  training-based; per-task EPS backbone</td></tr></table>

## F Metric Definitions

CRPS. For a scalar target z and a predictive distribution with CDF F , $\begin{array} { r } { \mathrm { C R P S } ( F , z ) = \int _ { \mathbb { R } } ( F ( u ) - } \end{array}$ ${ \bf 1 } \{ u \ge z \} ) ^ { 2 } \mathrm { d } u$ . For our multivariate settings we report the average per-coordinate CRPS, in pixel space (CRPS-pixel) and in the Inception feature space used for FID (CRPS-inception), each averaged over the evaluation images and over the posterior samples drawn for each observation [44].

MMD. With a Gaussian kernel $k ( u , v ) = \exp ( - \| u - v \| ^ { 2 } / ( 2 \ell ^ { 2 } ) )$ , we estimate $\mathrm { M M D } ^ { 2 } ( P , Q )$ between the EPS sample distribution P and the empirical distribution Q of the evaluation ground truths using the unbiased U-statistic of Gretton et al. [45]. We report MMD-pixel (kernel in pixel space) and MMD-inception (kernel in Inception feature space). The bandwidth ℓ is chosen via the median heuristic on the combined sample.