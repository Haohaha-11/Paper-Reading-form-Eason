# DIFFUSION MODELS FOR INVERSE PROBLEMS

Hyungjin Chung EverEx hj.chung@everex.co.kr

Jeongsol Kim KAIST jeongsol@kaist.ac.kr

Jong Chul Ye KAIST jong.ye@kaist.ac.kr

## ABSTRACT

Using diffusion priors to solve inverse problems in imaging have significantly matured over the years. In this chapter, we review the various different approaches that were proposed over the years. We categorize the approaches into the more classic explicit approximation approaches and others, which include variational inference, sequential monte carlo, and decoupled data consistency. We cover the extension to more challenging situations, including blind cases, high-dimensional data, and problems under data scarcity and distribution mismatch. More recent approaches that aim to leverage multimodal information through texts are covered. Through this chapter, we aim to (i) distill the common mathematical threads that connect these algorithms, (ii) systematically contrast their assumptions and performance trade-offs across representative inverse problems, and (iii) spotlight the open theoretical and practical challenges by clarifying the landscape of diffusion model based inverse problem solvers.

## 1 Introduction

We consider inverse problems in the following form

$$
{ \pmb y } = \mathcal { A } ( { \pmb x } ) + { \pmb n }\tag{1}
$$

where $\mathcal { A } : \mathbb { R } ^ { n } \mapsto \mathbb { R } ^ { m } , m < n$ is the forward operator that maps the signal that we wish to recover, $\ b { x } \in \mathbb { R } ^ { n }$ to the measurement $\ b { y } \in \mathbb { R } ^ { m }$ , and the process is corrupted by noise $\pmb { n } \in \mathbb { R } ^ { m 1 }$ . Due to the ill-posedness of the problem, infinitely many feasible solutions exist, and perfect recovery is impossible (Tarantola 2005). Among the feasible solutions, we aim to find a good set of solutions that also match the characteristics of the real-world data. Mathematically, this can be handily written down with Bayes rule

$$
p ( { \pmb x } | { \pmb y } ) = p ( { \pmb x } ) p ( { \pmb y } | { \pmb x } ) / p ( { \pmb y } ) , \quad p ( { \pmb y } ) = \int _ { { \pmb x } } p ( { \pmb y } | { \pmb x } ) .\tag{2}
$$

One of the most widely studied and used cases is when the likelihood function is a Gaussian model, i.e. $p ( \pmb { y } | \pmb { x } ) =$ $\mathcal { N } ( \pmb { y } ; \mathcal { A } ( \pmb { x } ) , \sigma _ { \pmb { y } } ^ { 2 } I )$ . It is easy to see that this corresponds to the case where $\pmb { n } = \sigma _ { y } \epsilon , \epsilon \sim \mathcal { N } ( 0 , I )$

Due to the nature of the problem, it is up to the user to define the type of recovery one wants. The following three are among the most widely opted goals:

1. Sampling from the posterior (i.e. posterior sampling): $\pmb { x } \sim p ( \pmb { x } | \pmb { y } )$

2. Finding a minimum mean-squared error (MMSE) estimate: ${ \pmb x } = \mathbb { E } [ { \pmb x } | { \pmb y } ]$

3. Finding a maximum a posteriori (MAP) estimate: $\pmb { x } = \arg \operatorname* { m a x } _ { \pmb { x } } p ( \pmb { x } | \pmb { y } )$

Blau & Michaeli (2018) shows that there is a trade-off between perception and distortion, and one cannot maximize perception and minimize distortion at the same time<sup>2</sup>. Note that any of the above goals can be solved by specifying the posterior, which, in turn, can be naturally achieved by specifying the prior. All inverse problem solvers, either explicitly or implicitly, uses this prior function. In this work, we focus mostly on posterior sampling methods that leverage the generative prior (Bora et al. 2017), in the sense that the prior function is defined through a deep generative model that is trained from data sources.

In the modern generative AI era, modeling the prior data distribution through a generative model is becoming ever more powerful and prominent. Among them, diffusion models (Ho et al. 2020, Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole 2021) have become the predominant paradigm in modeling the distribution of images and videos. While there are more recent variants of diffusion models such as flow matching (Lipman et al. 2023), rectified flow (Liu, Gong & qiang liu 2023), etc., we simply refer to them as diffusion models hereafter as the principles remain the same<sup>3</sup>.

As directly modeling the distribution is hard due to the existence of the normalization constant, a clever bypass is to learn the gradient of the log density $\nabla _ { \boldsymbol { x } } \log p ( \boldsymbol { x } )$ , often called the score function (Hyvärinen & Dayan 2005). Diffusion models learn a family of blurred score functions $\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } )$ in various noise levels $t \in [ 0 , T ]$ , with $t = 0$ corresponding to the original data distribution, and $t = T$ resulting in the reference Gaussian distribution. Once the diffusion model is trained along this forward diffusion trajectory, one can sample from the learned distribution by running a reverse diffusion trajectory, which can be characterized by a stochastic differential equation (SDE), or equivalently, an ordinary differential equation (ODE), in the continuous time limit (Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole 2021).

As the reverse diffusion process involves the score function of the prior, we are able to sample from the posterior if we use the score function of the posterior

$$
\begin{array} { r } { \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } | \pmb { y } ) = \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } ) + \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { x } _ { t } ) . } \end{array}\tag{3}
$$

While this may sound straightforward, $p ( \pmb { y } | \pmb { x } _ { t } )$ is in fact, intractable, and hence requires some form of approximation, or other ways to bypass the computation. In this chapter, we review some of the most widely used Diffusion model based Inverse problem Solvers (DIS) by comparing the categorizing the methods into the ones that make explicit approximations to this term, and other approaches. We note that Daras, Chung, Lai, Mitsufuji, Ye, Milanfar, Dimakis & Delbracio (2024) provides a comprehensive review and taxonomy of existing DIS, and we reuse parts of their layout for ease of comparison. However, our chapter diverges by identifying new classes, pushing the timeline to mid-2025, and covering other extensions (e.g. high-dimensional data).

This chapter is structured as follows: In Sec. 2, we review the fundamentals of diffusion models in both the scoreperspective and the variational perspective. In Sec. 3, we study the explicit approximation methods, with a focus on diffusion posterior sampling (Chung, Kim, Mccann, Klasky & Ye 2023). In Sec. 4, we review a taxonomy of DIS that does not belong to the explicit category, but offers other principled approaches. In Sec. 5, we extend the solvers to more challenging situations, e.g. blind inverse problems. In Sec. 6, we review approaches that leverage texts as additional source of control knob to deduce solutions. Finally, in Sec. 7, we conclude by discussing the current status and future perspectives of DIS.

## 2 Background: Diffusion Models

## 2.1 Score perspective

Consider the continuous diffusion process ${ \pmb x } _ { t } , t \in [ 0 , T ]$ with $\pmb { x } _ { t } \in \mathbb { R } ^ { d }$ (Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole 2021). We initialize the process with $\pmb { x } _ { 0 } \sim p _ { 0 } ( \pmb { x } )$ , where $p _ { 0 } = p _ { \mathrm { d a t a } }$ represents our initial data distribution, and let $\mathbf { \boldsymbol { x } } _ { T } \sim p _ { T }$ , with $p _ { T }$ being a reference distribution from which we can draw samples. The forward noising process spanning from $t = 0  T$ is characterized by the following Itˆo stochastic differential equation:

$$
\begin{array} { r } { d { \pmb x } _ { t } = { \pmb f } ( { \pmb x } _ { t } , t ) d t + g ( t ) d { \pmb w } , \quad { \pmb f } : \mathbb { R } ^ { d } \times \mathbb { R } \mapsto \mathbb { R } ^ { d } , g : \mathbb { R } \mapsto \mathbb { R } , } \end{array}\tag{4}
$$

where f denotes the drift function associated with $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ , and $g$ signifies the diffusion coefficient linked with the standard d-dimensional Brownian motion $\pmb { w } \in \mathbb { R } ^ { d }$ . Through the judicious selection of $f$ and $^ { g , }$ one can asymptotically converge towards the Gaussian distribution as $t \to T$ . When the drift function $f$ is defined as an affine function of $^ { \mathbf { \delta x } , }$ specifically $\pmb { f } ( \pmb { x } , t ) = f ( t ) \pmb { x }$ , it follows that the perturbation kernel $p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ consistently exhibits Gaussian characteristics, with its parameters being derivable in closed-form. Consequently, the process of perturbing the data utilizing the perturbation kernel $p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ can be accomplished without the necessity of executing the forward SDE.

For the specified forward SDE in (4), it can be demonstrated that a corresponding reverse-time SDE exists, which operates in a backward manner (Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole 2021, Huang et al. 2021, Anderson 1982):

$$
d { \pmb x } _ { t } = [ { \pmb f } ( { \pmb x } _ { t } , t ) - g ( t ) ^ { 2 } \nabla _ { { \pmb x } _ { t } } \log p _ { t } ( { \pmb x } _ { t } ) ] d t + g ( t ) d \bar { \pmb w }\tag{5}
$$

where dt represents the infinitesimal negative time increment, and w¯ is the standard Brownian motion progressing in reverse. Executing the reverse diffusion as delineated in (5) by initializing with a random Gaussian noise would facilitate sampling from $p _ { 0 } ( { \pmb x } )$ . It is evident that access to the time-conditional score function $\nabla _ { \pmb { x } _ { t } }$ log $p _ { t } ( \pmb { x } _ { t } )$ is requisite, which corresponds to the score function of the smoothed data distribution that has been convolved with a Gaussian kernel.

An intriguing observation is that there exists a corresponding deterministic ordinary differential equation (ODE) associated with (5), which is expressed as

$$
d \pmb { x } _ { t } = \underbrace { [ \pmb { f } ( \pmb { x } _ { t } , t ) - \frac { 1 } { 2 } g ( t ) ^ { 2 } \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } ) ] } _ { = : \tilde { \pmb { f } } _ { \theta } ( \pmb { x } _ { t } , t ) } d t .\tag{6}
$$

The ODE represented in (6) is referred to as the probability-flow ODE (PF-ODE). While both (5) and (6) yield the same law $p _ { t } ( \pmb { x } _ { t } )$ , PF-ODE possesses several notable properties. Firstly, diffusion models may be reconceptualized as a variant of continuous normalizing flows (CNF) (Chen et al. 2018) by interpreting the network as $\tilde { \pmb { f } } _ { \theta }$ , thereby facilitating tractable likelihood computations. Secondly, ODE solvers generally exhibit superior behavior in comparison to SDE solvers. Utilizing the PF-ODE instead of the reverse SDE results in expedited sampling.

It is feasible to train a neural network to approximate the true score function through score matching (Hyvärinen & Dayan 2005), thereby estimating $\mathbf { \boldsymbol { s } } _ { \theta } ( \mathbf { \boldsymbol { x } } _ { t } , t ) \approx \nabla _ { \mathbf { \boldsymbol { x } } _ { t } } \log { p _ { t } ( \mathbf { \boldsymbol { x } } _ { t } ) }$ , which can subsequently be incorporated into (5). Nonetheless, it is acknowledged that the application of either explicit or implicit score matching poses significant challenges in terms of scalability, primarily due to inherent instabilities and substantial computational demands. To address these technical obstacles, denoising score matching (DSM) is employed:

$$
\theta ^ { * } = \underset { \theta } { \operatorname { a r g m i n } } \mathbb { E } _ { t \sim \operatorname { U n i f } ( 0 , T ) , \boldsymbol { x } _ { t } \sim p ( \boldsymbol { x } _ { t } | \boldsymbol { x } _ { 0 } ) , \boldsymbol { x } _ { 0 } \sim p ( \boldsymbol { x } _ { 0 } ) } \left[ \| \boldsymbol { s } _ { \theta } ( \boldsymbol { x } _ { t } , t ) - \nabla _ { \boldsymbol { x } _ { t } } \log p ( \boldsymbol { x } _ { t } | \boldsymbol { x } _ { 0 } ) \| _ { 2 } ^ { 2 } \right] .\tag{7}
$$

It is pertinent to acknowledge that DSM is fundamentally equivalent to the training of a denoising autoencoder (DAE) across various noise levels (Vincent 2011), which are dictated by an auxiliary input t. Specifically, let us examine the most basic forward perturbation kernel defined as $p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) = \dot { \mathcal { N } } ( \pmb { x } _ { t } ; \pmb { x } _ { 0 } , t ^ { 2 } \pmb { I } )$ . By establishing a denoiser parametrization $D _ { \theta } ( { \pmb x } _ { t } , t ) \triangleq - { \pmb s } _ { \theta } ( { \pmb x } _ { t } , t ) / t ^ { 2 }$ , it becomes evident that (7) can be reformulated as:

$$
\theta ^ { * } = \underset { \theta } { \arg \operatorname* { m i n } } \mathbb { E } _ { t \sim \mathrm { U n i f } ( 0 , T ) , \mathbf { x } _ { t } \sim p ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) , \mathbf { x } _ { 0 } \sim p ( \mathbf { x } _ { 0 } ) } \left[ t \| D _ { \theta } ( \mathbf { x } _ { t } , t ) - \mathbf { x } _ { 0 } \| _ { 2 } ^ { 2 } \right] .\tag{8}
$$

The correspondence between (7) and (8) is also fundamentally linked to Tweedie’s theorem (Efron 2011).

Theorem 1 (Tweedie’s theorem). In the context of a Gaussian perturbation kernel represented as $p ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) =$ $\mathcal { N } ( \pmb { x } _ { t } ; s _ { t } \pmb { x } _ { 0 } , \sigma _ { t } ^ { 2 } \pmb { I } )$ , the posterior mean is articulated mathematically as:

$$
\mathbb { E } [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ] = \frac { 1 } { s _ { t } } ( \pmb { x } _ { t } + \sigma _ { t } ^ { 2 } \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } ) )\tag{9}
$$

In essence, the parametrization delineated in (8) serves as a direct means of estimating the posterior mean $\mathbb { E } [ { \pmb x } _ { 0 } | { \pmb x } _ { t } ]$ Irrespective of the chosen parametrization, and due to the implications of Theorem 1, diffusion models can be conceptualized as possessing two complementary representations: the noisy variable $\mathbf { \Delta } \mathbf { x } _ { t } .$ , which evolves according to the reverse SDE outlined in (5), and the posterior mean $\mathbb { E } [ { \pmb x } _ { 0 } | { \pmb x } _ { t } ]$ , which is implicitly characterized by Tweedie’s theorem and may be interpreted as the terminal point of the trajectory when adopting a tangent direction relative to the current step.

By choosing $s _ { t } = 1 , \sigma _ { t } = t .$ , the PF-ODE reads

$$
d \pmb { x } _ { t } = - t \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } ) = \frac { \pmb { x } _ { t } - \mathbb { E } [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ] } { t } d t\tag{10}
$$

## 2.2 Variational perspective

Parallel to the evolution of the score-based framework concerning diffusion models, a variational framework was concurrently established (Sohl-Dickstein et al. 2015, Ho et al. 2020), which now forges a connection between diffusion models and Variational Autoencoders (VAEs) (Kingma & Welling 2013). More specifically, within this framework, diffusion models are conceptualized as a hierarchical latent variable model referred to as denoising diffusion probabilistic models (DDPM)

$$
p _ { \theta } ( \pmb { x } _ { 0 } ) = \int p _ { \theta } ( \pmb { x } _ { T } ) \prod _ { t = 1 } ^ { T } p _ { \theta } ^ { ( t ) } ( \pmb { x } _ { t - 1 } \vert \pmb { x } _ { t } ) d \pmb { x } _ { 1 : T } ,\tag{11}
$$

where $\pmb { x } _ { \{ 1 , . . . , T \} } \in \mathbb { R } ^ { d }$ . The neural network that characterizes $p _ { \theta }$ is subsequently optimized by minimizing the evidence lower bound (ELBO)

$$
\mathbb { E } [ - \log p _ { \theta } ( \pmb { x } _ { 0 } ) ] \le \mathbb { E } _ { q } \left[ - \log \frac { p _ { \theta } \left( \pmb { x } _ { 0 : T } \right) } { q \left( \pmb { x } _ { 1 : T } | \pmb { x } _ { 0 } \right) } \right] = \mathbb { E } _ { q } \left[ - \log p ( \pmb { x } _ { T } ) - \sum _ { t \ge 1 } \log \frac { p _ { \theta } \left( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } \right) } { q \left( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } \right) } \right]\tag{12}
$$

where the inference distribution $q$ is delineated by the Markovian forward conditional densities

$$
q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } ) = \mathcal { N } ( \pmb { x } _ { t } | \sqrt { \beta _ { t } } \pmb { x } _ { t - 1 } , ( 1 - \beta _ { t } ) I ) ,\tag{13}
$$

$$
q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) = \mathcal { N } ( \pmb { x } _ { t } | \sqrt { \bar { \alpha } _ { t } } \pmb { x } _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) I ) .\tag{14}
$$

In this context, the noise schedule $\beta _ { t }$ is characterized as an increasing sequence indexed by t, with $\begin{array} { r } { \bar { \alpha } _ { t } : = \prod _ { i = 1 } ^ { t } \alpha _ { t } , \alpha _ { t } : = } \end{array}$ $1 - \beta _ { t }$ . The selection of the noise schedule is made such that the signal coefficient $\sqrt { \bar { \alpha } _ { t } }$ approaches 0 as $t \to T$ , thereby ensuring that the noise coefficient $1 - \bar { \alpha } _ { t }$ approaches 1, thereby converging towards the standard normal distribution. In contrast to the VE diffusion choice elaborated in Sec. 2.1, the selection employed here is denoted as variance preserving (VP). Notably, the discrete VP configuration in (13), when transitioned to its continuous analogue by increasing the number of discretization steps to $N  \infty ,$ , engenders the following Stochastic Differential Equation (SDE)

$$
d \pmb { x } = - \frac { 1 } { 2 } \beta _ { t } \pmb { x } d t + \sqrt { \beta _ { t } } d \pmb { w } .\tag{15}
$$

The minimization of the ELBO objective in (12) fundamentally gives rise to the following optimization challenge

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { q } \left[ \sum _ { t > 1 } D _ { \mathrm { K L } } ( q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) | | p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) ) \right] .\tag{16}
$$

The KL minimization task delineated in (16) is computationally feasible as both distributions are Gaussian. For the initial term, this derives from the application of Bayes’ rule alongside the Markov property

$$
q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) = q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } , \pmb { x } _ { 0 } ) \frac { q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { 0 } ) } { q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) } = \mathcal { N } ( \pmb { x } _ { t - 1 } ; \tilde { \pmb { \mu } } _ { t } ( \pmb { x } _ { t } , \pmb { x } _ { 0 } ) , \tilde { \beta } _ { t } I ) ,\tag{17}
$$

$$
\mathrm { w h e r e } \quad \tilde { \mu } _ { t } ( x _ { t } , x _ { 0 } ) : = \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } { \displaystyle x _ { 0 } + \frac { \sqrt { \alpha _ { t } } ( 1 - \bar { \alpha } _ { t - 1 } ) } { 1 - \bar { \alpha } _ { t } } } { \displaystyle x _ { t } } , \tilde { \beta } _ { t } : = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \beta _ { t } .\tag{18}
$$

For the subsequent term, the reverse distribution is Gaussian as we account for minimal perturbations pertinent to a singular step of forward diffusion (Ho et al. 2020). A common parametrization is established as follows

$$
p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) = \mathcal { N } ( \pmb { x } _ { t - 1 } ; \pmb { \mu } _ { \theta } ( \pmb { x } _ { t } , t ) , \tilde { \beta } I ) ,\tag{19}
$$

$$
\mathrm { w h e r e } \quad \mu _ { \theta } ( { \pmb x } _ { t } , t ) = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( { \pmb x } _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } { \pmb \epsilon } _ { \theta } ( { \pmb x } _ { t } , t ) \right) .\tag{20}
$$

Under this formulation, the ELBO objective in (12) can be streamlined to the epsilon-matching objective by disregarding the time-dependent weighting factors

$$
\theta ^ { * } = \underset { \theta } { \arg \operatorname* { m i n } } \mathbb { E } _ { \pmb { x } _ { t } \sim q ( \pmb { x } _ { t } | \pmb { x _ { 0 } } ) , \pmb { x _ { 0 } } \sim p _ { \mathrm { d a t a } } ( \pmb { x _ { 0 } } ) , \epsilon \sim \mathcal { N } ( 0 , I ) } \left[ \| \epsilon _ { \theta } ( \pmb { x _ { t } } , t ) - \epsilon \| _ { 2 } ^ { 2 } \right] .\tag{21}
$$

Epsilon matching is fundamentally analogous to the DSM/DAE objective in (7), (8), differing solely by a constant with an alternative parametrization. Given the correspondence between the forward noising distribution in (15) and the learning objective in (7),(21), it becomes evident that the two frameworks essentially converge upon the same model.

Inference can be executed by incorporating the trained $\epsilon _ { \theta }$ to approximate the expectation of $p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ , culminating in the subsequent iterative expression

$$
{ \pmb x } _ { t - 1 } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( { \pmb x } _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } { \pmb \epsilon } _ { \theta } ( { \pmb x } _ { t } , t ) \right) + \tilde { \beta } _ { t } { \pmb \epsilon } , \quad { \epsilon } \sim \mathcal { N } ( \mathbf { 0 } , I ) .\tag{22}
$$

It is noteworthy that analogous to the reverse stochastic differential equation (SDE) delineated in (5), stochastic perturbations are incorporated in each iteration throughout the DDPM sampling process, resulting in a protracted inference duration. A conventional methodology to mitigate this phenomenon, akin to the transition towards the PF-ODE, is facilitated by denoising diffusion implicit models (DDIM) (Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole 2021), wherein an alternative inference distribution is proposed

$$
q _ { \eta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) = \mathcal { N } ( \pmb { x } _ { t - 1 } ; \sqrt { \bar { \alpha } _ { t - 1 } } \pmb { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t - 1 } - \eta \tilde { \beta } _ { t } ^ { 2 } } \frac { \pmb { x } _ { t } - \sqrt { \bar { \alpha } _ { t } } \pmb { x } _ { 0 } } { \sqrt { 1 - \bar { \alpha } _ { t } } } , \eta \tilde { \beta } _ { t } ^ { 2 } I ) ,\tag{23}
$$

where $\eta \in [ 0 , 1 ]$ . By establishing $\eta = 1 . 0 ,$ the original DDPM sampling procedure is reinstated with maximal stochasticity. Conversely, by designating $\eta = 0 . 0$ , a deterministic sampling mechanism is achieved, which can be demonstrated to be equivalent to the variance preserving PF-ODE (Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole 2021). Employing diminished values of η tends to yield superior outcomes when the objective is to minimize the number of function evaluations (NFE).

## 3 Explicit approximation methods

Many of the earlier works that aimed to solve inverse problems with diffusion models, whether explicitly mentioned in the original work or not, can be perceived as explicit approximation methods for the time-dependent log-likelihood $p ( \pmb { y } | \pmb { x } _ { t } )$ in (3). In this section, we review some of the canonical works that belong to this category, with a specific focus on the DPS (Chung, Kim, Mccann, Klasky & Ye 2023) family.

The first works that used diffusion model-like annealing-denoising steps with projection-like data consistency steps were Song & Ermon (2019), Kadkhodaie & Simoncelli (2021). While the details differ, one can understand the algorithms as alternating the denoising step and the data consistency projection step, gradually decreasing the noise level, starting from pure Gaussian noise. Note that the earlier works mostly focused on linear inverse problems, where $A = A$

Score-ALD (Jalal et al. 2021) In this work, the authors focused on the task of compressed-sensing MRI, where the following approximation was used

$$
\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { x } _ { t } ) \approx - \frac { \pmb { A } ^ { \top } ( \pmb { y } - \pmb { A } \pmb { x } _ { t } ) } { \sigma _ { y } ^ { 2 } + \gamma _ { t } ^ { 2 } } ,\tag{24}
$$

where $\gamma _ { t }$ was set to be a hyperparameter that decays as t approaches 0.

Score-SDE (Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole 2021) Score-SDE focused on linear inverse problems with an orthogonal matrix A

$$
\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { x } _ { t } ) \approx - \pmb { A } ^ { \top } ( \pmb { y } + \sigma _ { t } \pmb { \epsilon } - \pmb { A } \pmb { x } _ { t } ) ,\tag{25}
$$

which corresponds to a noisy projection onto $\pmb { y } + \sigma _ { t } \pmb { \epsilon } = A \pmb { x } _ { t }$

## 3.1 DDRM family

The methods that belong to this category explicitly uses singular value decomposition (SVD) $A = U \Sigma V ^ { \top } , U \in$ R<sup>m×m</sup>, $V \in \mathbb { R } ^ { n \times n } , \Sigma \in \mathbb { R } ^ { m \times n }$ , with Σ being a rectangular diagonal matrix with singular values $\{ s _ { j } \} _ { j = 1 } ^ { m }$ as the diagonal elements. Notice that one can then rewrite the linear inverse problem as

$$
\bar { y } = \Sigma \bar { { \pmb x } } + \sigma _ { y } \bar { { \boldsymbol \epsilon } } , \quad \mathrm { w h e r e } \quad \bar { y } : = U ^ { \top } { \pmb y } , \bar { { \pmb x } } : = V ^ { \top } { \pmb x } , \bar { { \boldsymbol \epsilon } } : = U ^ { \top } { \boldsymbol \epsilon } .\tag{26}
$$

Once x¯ is recovered from (26), ${ \hat { \mathbf { x } } } = V { \bar { \mathbf { x } } }$

SNIPS (Kawar et al. 2021) The approximation reads

$$
\nabla _ { \bar { x } _ { t } } \log p ( \bar { y } | \bar { x } _ { t } ) \approx - \Sigma ^ { \top } \left| \sigma _ { y } ^ { 2 } I - \sigma _ { t } ^ { 2 } \Sigma \Sigma ^ { \top } \right| ^ { \dag } \left( \bar { y } - \Sigma \bar { x } _ { t } \right) ,\tag{27}
$$

where the gradient points to a direction weighted by the magnitude of the difference between the diffusion noise level $\sigma _ { t } ^ { 2 }$ and the measurement noise $\sigma _ { y } ^ { 2 } .$ , additionally weighted by the singular values $s _ { i } ^ { 2 }$

DDRM (Kawar et al. 2022) DDRM is an extension of SNIPS which incorporates DDIM sampling, an additional mixing hyperparameter η, and using the posterior mean $\bar { \pmb x } _ { 0 | t } : = V \mathbb { E } [ \pmb x _ { 0 } | \pmb x _ { t } ]$

$$
\nabla _ { \bar { x } _ { t } } \log p ( \bar { y } | \bar { x } _ { t } ) \approx - \Sigma ^ { \top } \left| \sigma _ { y } ^ { 2 } I - \sigma _ { t } ^ { 2 } \Sigma \Sigma ^ { \top } \right| ^ { \dag } \left( \bar { y } - \Sigma \bar { x } _ { 0 | t } \right) .\tag{28}
$$

Notice that an element-wise expression of (30) can be written as

$$
\begin{array} { r } { p ( \bar { x } _ { t } ^ { ( i ) } | x _ { t + 1 } , y ) = \left\{ \begin{array} { l l } { \mathcal { N } ( \bar { x } _ { t } ^ { ( i ) } ; \bar { x } _ { 0 | t + 1 } ^ { ( i ) } , \sigma _ { t } ^ { 2 } ) } & { \mathrm { i f ~ } s _ { i } = 0 } \\ { \mathcal { N } ( \bar { x } _ { t } ^ { ( i ) } ; \bar { x } _ { 0 | t + 1 } ^ { ( i ) } , \sigma _ { t } ^ { 2 } ) } & { \mathrm { i f ~ } \sigma _ { t } < \frac { \sigma _ { y } } { s _ { i } } } \\ { \mathcal { N } ( \bar { x } _ { t } ^ { ( i ) } ; \bar { y } ^ { ( i ) } , \sigma _ { t } ^ { 2 } - \frac { \sigma _ { y } ^ { 2 } } { s _ { i } ^ { 2 } } ) } & { \mathrm { i f ~ } \sigma _ { t } \geq \frac { \sigma _ { y } } { s _ { i } } } \end{array} \right. . } \end{array}\tag{29}
$$

Analagous to the role of mixing coefficient η in DDIM sampling, DDRM introduces a hyper-parameter $\eta \in ( 0 , 1 ]$ to get

$$
\begin{array} { r } { p ( \bar { x } _ { t } ^ { ( i ) } | x _ { t + 1 } , y ) = \left\{ \begin{array} { l l } { \mathcal { N } ( \bar { x } _ { t } ^ { ( i ) } ; \bar { x } _ { 0 | t + 1 } ^ { ( i ) } + \sqrt { 1 - \eta ^ { 2 } } \sigma _ { t } \frac { \bar { x } _ { t + 1 } ^ { ( i ) } - \bar { x } _ { 0 | t + 1 } ^ { ( i ) } } { \sigma _ { t + 1 } } , \eta ^ { 2 } \sigma _ { t } ^ { 2 } ) } & { \mathrm { i f ~ } s _ { i } = 0 } \\ { \mathcal { N } ( \bar { x } _ { t } ^ { ( i ) } ; \bar { x } _ { 0 | t + 1 } ^ { ( i ) } + \sqrt { 1 - \eta ^ { 2 } } \sigma _ { t } \frac { { \bar { y } } ^ { ( i ) } - { \bar { x } } _ { 0 | t + 1 } ^ { ( i ) } } { \sigma _ { y } / s _ { i } } , \eta ^ { 2 } \sigma _ { t } ^ { 2 } ) } & { \mathrm { i f ~ } \sigma _ { t } < \frac { \sigma _ { y } } { s _ { i } } \ . } \\ { \mathcal { N } ( \bar { x } _ { t } ^ { ( i ) } ; { \bar { y } } ^ { ( i ) } , \sigma _ { t } ^ { 2 } - \frac { \sigma _ { y } ^ { 2 } } { s _ { i } ^ { 2 } } ) } & { \mathrm { i f ~ } \sigma _ { t } \ge \frac { \sigma _ { y } } { s _ { i } } } \end{array} \right. } \end{array}\tag{30}
$$

## 3.2 DPS family

DPS (Chung, Kim, Mccann, Klasky & Ye 2023) Notice that

$$
p ( \pmb { y } | \pmb { x } _ { t } ) = \int p ( \pmb { y } | \pmb { x } _ { 0 } ) p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } ) d \pmb { x } _ { 0 } = \mathbb { E } _ { \pmb { x } _ { 0 } \sim p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } ) } [ p ( \pmb { y } | \pmb { x } _ { 0 } ) ] .\tag{31}
$$

The computation of $p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } )$ is challenging, as we would have to marginalize over all the latent steps t through 0, not to mention the integration over the trajectories. It would be computationally intractable to compute this value every time we need access to the time-conditional likelihood. The idea of DPS is to push the expectation inside

$$
p ( \pmb { y } | \pmb { x } _ { t } ) \approx p ( \pmb { y } | \hat { \pmb { x } } _ { 0 | t } ) , \quad \mathrm { w h e r e } \quad \hat { \pmb { x } } _ { 0 | t } = \mathbb { E } [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ] .\tag{32}
$$

This approximation is often referred to as Jensen’s approximation, whose approximation bound has been shown to be controllable in the context of Gaussian measurement scenarios (Chung, Kim, Mccann, Klasky & Ye 2023). Recall from Theorem 1 that one can easily compute the MMSE estimate $\hat { \mathbf { x } } _ { 0 \mid t }$ through a single forward pass through the score function. From the definition of the forward model, it is then easy to see that

$$
\nabla _ { \pmb { x } _ { t } } \log { p ( \pmb { y } | \hat { \pmb { x } } _ { 0 | t } ) } = - \frac { 1 } { 2 \sigma _ { y } ^ { 2 } } \nabla _ { \pmb { x } _ { t } } \| \pmb { y } - \pmb { A } ( \hat { \pmb { x } } _ { 0 | t } ) \| _ { 2 } ^ { 2 } ,\tag{33}
$$

where in practice, an empirical static step $\rho$ size is often employed in the place of $1 / 2 \sigma _ { y } ^ { 2 } .$ The computation of the gradient can be done through backpropagation, as it involves a backward pass through the score function. It is important to note that DPS is fully general in that it is capable of solving non-linear inverse problems with arbitrary forward models if it can be defined.

ΠGDM (Song, Vahdat, Mardani & Kautz 2023) From (32), DPS can be interpreted as using the following approximation

$$
p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } ) \approx \delta ( \pmb { x } _ { 0 } - \hat { \pmb { x } } _ { 0 | t } ) .\tag{34}
$$

ΠGDM instead places an isotropic Gaussian distribution for approximation

$$
p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } ) \approx \mathcal { N } ( \hat { \pmb { x } } _ { 0 | t } , r _ { t } ^ { 2 } I ) ,\tag{35}
$$

where $r _ { t }$ is a hyperparameter. For the case of linear inverse problems, this leads to

$$
p ( \pmb { y } | \pmb { x } _ { t } ) \approx \mathcal { N } ( A \hat { \pmb { x } } _ { 0 | t } , r _ { t } ^ { 2 } A A ^ { \top } + \sigma _ { y } ^ { 2 } I ) ,\tag{36}
$$

and subsequently

$$
\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { x } _ { t } ) \approx - \nabla _ { \pmb { x } _ { t } } \hat { \pmb { x } } _ { 0 \vert t } ( r _ { t } ^ { 2 } A A ^ { \top } + \sigma _ { y } ^ { 2 } I ) ^ { - 1 } A ^ { \top } ( \pmb { y } - A \hat { \pmb { x } } _ { 0 \vert t } )\tag{37}
$$

Moment Matching (Rozet et al. 2024) In moment matching, the authors explicitly calculate the variance matrix for $p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } )$ , leading to a better approximation

$$
p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } ) \approx \mathcal { N } ( \hat { \pmb { x } } _ { 0 | t } , \mathrm { V a r } [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ] ) , \quad \mathrm { w h e r e } \quad \mathrm { V a r } [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ] = \sigma _ { t } ^ { 2 } \nabla _ { \pmb { x } _ { t } } \hat { \pmb { x } } _ { 0 | t } .\tag{38}
$$

In turn, this leads to

$$
\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { x } _ { t } ) \approx - \nabla _ { \pmb { x } _ { t } } ( A \hat { \pmb { x } } _ { 0 | t } ) ^ { \top } ( \sigma _ { y } ^ { 2 } I + A \sigma _ { t } ^ { 2 } \nabla _ { \pmb { x } _ { t } } \hat { \pmb { x } } _ { 0 | t } A ^ { \top } ) ^ { - 1 } ( \pmb { y } - A \hat { \pmb { x } } _ { 0 | t } )\tag{39}
$$

Note that in high-dimensions, explicit computation of $\nabla _ { \pmb { x } _ { t } } \hat { \pmb { x } } _ { 0 | t }$ is expensive. Nevertheless, Jacobian-vector products (JVP) can be used for efficient computation for both ΠGDM and moment matching.

Peng et al. (2024) In a related work of Peng et al. (2024), the authors show that there exists an optimal posterior diagonal posterior covariance in by analyzing the diffusion model under the DDPM framework. The covariance matrix can be determined through maximum likelihood estimation, without relying on the computation of $\nabla _ { \pmb { x } _ { t } } \hat { \pmb { x } } _ { 0 \mid t }$ , and it was further shown that using this optimal covariance enhances the performance on robustness in all cases.

DDS (Chung, Lee & Ye 2024) One of the critical downsides of the other methods within the DPS family is that they are slow to compute, and requires excessive memory, since the computation of $\nabla _ { \pmb { x } _ { t } } \hat { \pmb { x } } _ { 0 \mid t }$ is involved. This may not be suitable for large-scale inverse problems, which the authors of Chung, Lee & Ye (2024) investigate. The key finding of DDS is that, under certain conditions on the data manifold, one can circumvent the heavy computation.

Proposition 1 (Manifold Constrained Gradient (Chung, Sim, Ryu & Ye 2022)). Suppose the clean data manifold M, where $\scriptstyle { \pmb x } _ { 0 }$ resides, is represented as an affine subspace and assumes the uniform distribution on M. Then,

$$
\frac { \partial \hat { \mathbf { x } } _ { 0 \mid t } } { \partial \mathbf { x } _ { t } } = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \mathcal { P } _ { \mathcal { M } }\tag{40}
$$

$$
\begin{array} { r } { \hat { { \pmb x } } _ { 0 \vert t } - \gamma _ { t } \nabla _ { { \pmb x } _ { t } } \ell ( \hat { { \pmb x } } _ { 0 \vert t } ) = \mathcal { P } _ { \mathcal { M } } ( \hat { { \pmb x } } _ { 0 \vert t } - \xi _ { t } \nabla _ { \hat { { \pmb x } } _ { 0 \vert t } } \ell ( \hat { { \pmb x } } _ { 0 \vert t } ) ) , } \end{array}\tag{41}
$$

for some $\xi _ { t } > 0 ;$ , where $\mathcal { P } _ { \mathcal { M } }$ denotes the orthogonal projection to $\mathcal { M }$

This implies that the manifold constrained gradient (MCG) can be regarded as the projected gradient method on the clean data manifold. To accelerate the convergence of the algorithms, the authors in Chung, Lee & Ye (2024) proposed performing multiple manifold-constrained update steps following a single neural network function evaluation (NFE) for manifold projection. This approach can be efficiently implemented using the conjugate gradient (CG) method or other Krylov subspace methods, under the assumption that the clean data manifold lies within a Krylov subspace.

Other approaches that improve DPS MPGD (He et al. 2024) proposes to project the DPS gradient to the manifold by leveraging an autoencoder. DSG (Yang et al. 2024) imposes a spherical constraint to control the steps reside on the noisy manifold, as discussed in MCG (Chung, Sim, Ryu & Ye 2022). DMAP (Xu et al. 2025) argues that DPS behaves closer to an MAP estimate rather than a posterior sampler, and thus proposes to make the algorithm behave closer to an MAP approximation method by imposing multi-step gradients, thereby improving performance. DPPS (Wu et al. 2024) reduces variance by proposing multiple candidates at each step of denoising, and only selecting the ones that maximize the data consistency.

Extension to flow models Flow-based model (Lipman et al. 2023) provide a general framework that includes diffusion models as a special case. FlowChef (Patel et al. 2024) introduces a general guidance term, $\nabla _ { \hat { \pmb { x } } _ { 0 \mid t } } \ell \big ( \hat { \pmb { x } } _ { 0 \mid t } \big )$ , into the reverse ODE, accompanied by an analysis of error dynamics, and demonstrates it effectiveness across various conditioned image generation tasks including linear inverse problems. FlowDPS (Kim, Kim & Ye 2025) extends posterior sampling theory from diffusion models to general affine flows by decomposing a single Euler step into a linear combination of clean and noise estimates, leveraging a generalized Tweedie’s formula.

## 4 Other methods

While we focused on explicit approximation methods in Sec. 3, there exists multiple different categories. In this section, we provide an introduction to some of the widely-acknowledged among them.

## 4.1 Variational inference

Another line of work on solving inverse problems with diffusion models stems from variational inference for the posterior distribution $p ( { \pmb x } | { \pmb y } )$ . The main advantage of this approach lies in distributional matching, which offers better diversity compared to the DPS family that approximates the log-likelihood at a single sample point $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ with its MMSE estimate $\hat { \mathbf { \mathscr { x } } } _ { 0 \mid t }$ . Let $q _ { \phi } ( { \pmb x } _ { 0 } | { \pmb y } )$ be a variational distribution with parameters ϕ. The goal of variational inference is to fit q to the target posterior distribution $p$ by minimizing

$$
\displaystyle \operatorname* { m i n } _ { \phi } D _ { K L } \left( q _ { \phi } (  { \mathbf { x } } _ { 0 } |  { \boldsymbol { y } } ) \| p (  { \mathbf { x } } _ { 0 } |  { \boldsymbol { y } } ) \right) = \operatorname* { m i n } _ { \phi } \mathbb { E } _ {  { \mathbf { x } } _ { 0 } \sim q _ { \phi } } \left[ \log q _ { \phi } (  { \boldsymbol { x } } _ { 0 } |  { \boldsymbol { y } } ) - \log p (  { \boldsymbol { x } } _ { 0 } |  { \boldsymbol { y } } ) \right] .\tag{42}
$$

From the definition of KL divergence, and applying Bayes’ rule to $p ( \pmb { x } _ { 0 } | \pmb { y } )$ , the objective function is reformulated as

$$
\displaystyle { \mathop { \operatorname* { m i n } } _ { \phi } } \underbrace { - \mathbb { E } _ { { \pmb x } \sim q _ { \phi } } \left[ \log p ( { \pmb y } | { \pmb x } _ { 0 } ) \right] } _ { \mathrm { d a t a ~ c o n s i s t e n c y } } + \underbrace { D _ { K L } \big ( q _ { \phi } ( { \pmb x } _ { 0 } | { \pmb y } ) \| p ( { \pmb x } _ { 0 } ) \big ) } _ { \mathrm { r e g u l a r i z e r } } + \underbrace { \log p ( { \pmb y } ) } _ { \mathrm { c o n s t a n t } } .\tag{43}
$$

RED-Diff (Mardani et al. 2023) Suppose that $q _ { \phi } ( { \pmb x } _ { 0 } | { \pmb y } )$ is the isotropic Gaussian distribution ${ \mathcal { N } } ( { \boldsymbol { \mu } } , \sigma ^ { 2 } \mathbf { I } )$ where $\phi = \{ \pmb { \mu } , \sigma \}$ . The objective function of (43) is equivalent to

$$
\mathbb { E } _ { \mathbf { x } _ { 0 } \sim q _ { \phi } ( \alpha _ { 0 } | y ) } \frac { \| y - \mathcal { A } ( \pmb { x } _ { 0 } ) \| ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } + \int _ { 0 } ^ { T } \frac { \beta _ { t } } { 2 } \mathbb { E } _ { \mathbf { x } _ { t } \sim q _ { \phi } ( \pmb { x } _ { t } | y ) } \left[ \left\| \nabla _ { \mathbf { x } _ { t } } \log q ( \pmb { x } _ { t } | y ) - \nabla _ { \mathbf { x } _ { t } } \log p ( \pmb { x } _ { t } ) \right\| ^ { 2 } \right] d t\tag{44}
$$

where ${ \pmb x } _ { t } \sim q _ { \phi } ( { \pmb x } _ { t } | { \pmb y } )$ denotes the diffusion trajectory computed by forward diffusion process in (14). The first term corresponds to data consistency derived from the definition of the forward model, and the second term denotes cumulative different of score functions along $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ which is derived from the relationship between KL divergence and score matching provided in Theorem 1 of (Song, Durkan, Murray $\&$ Ermon 2021). As $q _ { \phi } ( { \pmb x } _ { t } | { \pmb y } )$ is also Gaussian distribution, the optimization problem turns into a stochastic optimization:

$$
\operatorname* { m i n } _ { \mu } \frac { \| y - \mathcal { A } ( \mu ) \| ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } + \mathbb { E } _ { t , \epsilon } \frac { \beta _ { t } ( 1 - \bar { \alpha } _ { t } ) } { \bar { \alpha } } \| \epsilon _ { \theta } ( x _ { t } , t ) - \epsilon \| ^ { 2 }\tag{45}
$$

where the variance of $q ( { \pmb x } _ { 0 } | { \pmb y } )$ is assumed to be a constant near zero so the optimization variable becomes $\pmb { \mu } .$ . To improve efficiency and stability, back-propagation through the score network θ is omitted. Also, time t is sampled from T to 0 so the solution is reconstructed from coarse semantics to fine details which enhances perceptual quality. Notably, the second term reduces to the score-distillation loss. From a MAP perspective, the method implements regularization by denoising (Romano et al. 2017), where a pre-trained denoiser acts as the prior. Recently, FLAIR (Erbach et al. 2025) extended this framework to flow-based models by replacing the second term in (44) with a velocity difference and introducing a trajectory adjustment mechanism to ensure that the intermediate state $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ , where the score function is evaluated, lies in a high-likelihood region of the marginal distribution $p ( { \pmb x } _ { t } )$ . Specifically, they obtain $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ from $\mu$ using forward diffusion sampling that incorporates both deterministic noise, predicted via Tweeedie’s formula (Efron 2011, Kim & Ye 2021, Kim, Kim & Ye 2025), and stochastic noise.

Feng et al. (2023) A uni-modal Gaussian variational family cannot capture a complex, multi-modal posterior distribution. Feng et al. (2023) employ a normalizing flow - RealNVP (Dinh et al. 2017) to represent the variational distribution $q _ { \phi }$ . The corresponding objective starts from the same problem,

$$
\displaystyle \operatorname* { m i n } _ { \phi } D _ { K L } \big ( q _ { \phi } ( \pmb { x } ) \| p ( \pmb { x } | \pmb { y } ) \big ) = \operatorname* { m i n } _ { \phi } \mathbb { E } _ { \pmb { x } \sim q _ { \phi } } [ \log q _ { \phi } ( \pmb { x } ) - \log p ( \pmb { x } | \pmb { y } ) ]\tag{46}
$$

$$
= \underset { \phi } { \mathrm { m i n } } \mathbb { E } _ { \pmb { x } \sim q _ { \phi } } [ \log q _ { \phi } ( \pmb { x } ) - \log p ( \pmb { y } | \pmb { x } ) - \log p ( \pmb { x } ) ]\tag{47}
$$

where $- \log p ( \pmb { y } | \pmb { x } )$ is computed analytically from the forward model, log p(x) is approximated with a pre-trained diffusion model $\theta ,$ and log $q _ { \phi } ( { \pmb x } )$ is computationally tractable under RealNVP. Unlike methods that merely adjust individual samples toward higher posterior likelihood, this normalizing flow-based formulation allows direct sampling from the learned posterior. Consequently, it avoids hyper-parameter tuning (for example, step sizes for likelihood gradients) and produce diverse, robust samples. The trade-off is higher computational overhead for training and a dependence on the expressive power of the chosen normalizing-flow architecture.

This was later extended to Feng $\&$ Bouman (2024), where the computation of log $p ( { \pmb x } )$ by iterative sampling is replaced with a lower bound that involves the DSM loss, as proposed in Song, Durkan, Murray & Ermon (2021).

APS (Mammadov et al. 2024) Notice that (46) requires optimizing $\phi$ for every different observations y, which is costly. Amortized Posterior Sampling (APS) proposes the following amortization

$$
\operatorname* { m i n } _ { \phi } D _ { K L } \big ( q _ { \phi } ( \pmb { x } _ { 0 } | \pmb { y } ) \big | \big | p _ { \theta } \big ( \pmb { x } _ { 0 } | \pmb { y } \big ) \big ) ,\tag{48}
$$

which can be implemented as a conditional NF. Specifically, the authors proposed to extend RealNVP to a conditional setting, thereby enabling the use of a single network for all $\mathbf { \pmb { y } } .$

RSLD (Zilberstein et al. 2025) As another approach to estimate multi-modal posterior distribution, RSLD defines the particle-based variational inference and introduces a repulsive regularization to the score-matching term of (44). Specifically, it approximate the gradient for minimization problem (45) with ensemble of gradients:

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \nabla _ { \mu ^ { ( i ) } } \frac { \| y - \mathcal { A } ( \mu ^ { ( i ) } ) \| ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } + \mathbb { E } _ { t , \epsilon } \left[ \epsilon _ { \theta } ( x _ { t } ^ { ( i ) } , t ) - \epsilon - \nabla _ { x _ { t } ^ { ( i ) } } R ( x _ { t } ^ { ( 1 ) } . . . x _ { t } ^ { ( n ) } ) \right]\tag{49}
$$

where $\mathbf { \boldsymbol { x } } _ { t } ^ { ( i ) }$ is diffusion trajectory of i-th particle that is computed by forward diffusion process with $\mu ^ { ( i ) }$ and $\epsilon ,$ n denotes the number of particles, and $R ( \mathbf { \pmb { x } } _ { t } ^ { ( 1 ) } , . . . , \mathbf { \pmb { x } } _ { t } ^ { ( n ) } )$ ) denotes the repulsive regularization defined as

$$
\sum _ { j = 1 } ^ { n } \log \Big [ k ( \pmb { x } _ { t } ^ { ( i ) } , \pmb { x } _ { t } ^ { ( j ) } ) \Big ] ^ { r }\tag{50}
$$

This gradient is derived by incorporating ODE of each particles - transformed from variational distribution q to the posterior distribution $p$ via Wasserstein Gradient Flow - into the second term of (44). As a result, RSLD jointly update n particles using gradient in (49), yielding diverse samples that follows the posterior distribution.

DAVI (Lee et al. 2024) While normalizing flow modesl can represent more complex variational distributions, they typically require multiple iterations to obtain a solution. DAVI addresses this limitation by training a neural network to estimate $q _ { \phi } ( { \pmb x } _ { 0 } | { \pmb y } )$ in (43), enabling one-step sampling. However, the authors also highlight a challenge, the lack of overlap between the supports of $q _ { \phi } ( { \pmb x } _ { 0 } | { \pmb y } )$ and $p ( \pmb { x } _ { 0 } )$ , which leads to unstable training and limited performance. As a result, DAVI reformulate the problem using integral form of the KL divergence in (44). Unlike RedDiff that assumes $q ( { \pmb x } _ { t } | { \pmb y } )$ as Gaussian distribution, DAVI compute $\nabla _ { \pmb { x } _ { t } }$ log $q _ { \phi } ( { \pmb x } _ { t } | { \pmb y } )$ by an implicit score function $s _ { \psi }$ . Thus, during trainig, DAVI alternates between updaing $q _ { \phi } ( \pmb { x } _ { 0 } | \pmb { y } )$ and $\scriptstyle { \pmb x } _ { \phi }$ . Specifically, $q _ { \phi } ( \pmb { x } _ { 0 } | \pmb { y } )$ is updated by minimizing (44), using the approximation $\nabla _ { \pmb { x } _ { t } }$ log $q _ { \phi } ( \pmb { x } _ { t } | \pmb { y } ) \approx \pmb { s } _ { \psi }$ . In turn, $\scriptstyle { \pmb x } _ { \psi }$ is trained via denoising score matching using samples from the marginal $q _ { t } ( \pmb { x } _ { t } | \pmb { y } )$ , obtained by first drawing ${ \pmb x } _ { 0 } \sim q _ { \phi } ( { \pmb x } _ { 0 } | { \pmb y } )$ and then applying the forward diffusion process ${ \pmb x } _ { t } \sim q ( { \pmb x } _ { t } | { \pmb x } _ { 0 } )$

## 4.2 Decoupled data consistency

DAPS (Zhang et al. 2025) In explicit approximation methods, the solvers typically alternate between a small step of denoising, and a likelihood gradient step. Often, this results in the resulting samples diverging, especially in challenging cases (e.g. Fourier phase retrieval). One way to mitigate this with more compute, is to leverage more compute. Specifically, rather than relying on the Tweedie estimate as in DPS, one can first run the PF-ODE to sample from $\tilde { \pmb { x } } _ { 0 | t } ^ { ( j ) } \sim p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } )$ . Then, to impose data consistency, DAPS runs N-step Langevin dynamics

$$
\begin{array} { r } { \tilde { \mathbf { \boldsymbol { x } } } _ { 0 \mid t } ^ { ( j + 1 ) } = \tilde { \mathbf { \boldsymbol { x } } } _ { 0 \mid t } ^ { ( j ) } + \eta _ { t } \left( \nabla _ { \tilde { \mathbf { \boldsymbol { x } } } _ { 0 \mid t } } \log p ( \tilde { \mathbf { \boldsymbol { x } } } _ { 0 \mid t } ^ { ( j ) } | \mathbf { x } _ { t } ) + \nabla _ { \tilde { \mathbf { \boldsymbol { x } } } _ { 0 \mid t } } \log p ( \boldsymbol { y } | \tilde { \mathbf { \boldsymbol { x } } } _ { 0 \mid t } ^ { ( j ) } ) + \sqrt { 2 \eta _ { t } } \epsilon _ { j } \right) , } \end{array}\tag{51}
$$

where $\eta _ { t } > 0$ is a hyperparameter. This process is applied for all t, where the next iteration starts with $x _ { t - 1 } \sim$ $p ( \pmb { x } _ { t - 1 } | \pmb { x } _ { 0 } )$ . Such approach decouples the data consistency with the unconditional sampling steps, i.e. $p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } , \pmb { y } ) \propto$ $p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } ) p ( \pmb { y } | \pmb { x } _ { 0 } )$ , thereby yielding improved performance in certain challenging cases.

DCDP (Li et al. 2024) DCDP follows a similar decoupled approach, but differs in how the data consistency steps are performed. Specifically, Li et al. (2024) proposes to use proximal optimization steps

$$
\pmb { x } _ { k } = \arg \operatorname* { m i n } _ { \pmb { x } } \frac { 1 } { 2 } \| \pmb { \mathcal { A } } ( \pmb { x } ) - \pmb { y } \| _ { 2 } ^ { 2 } + \mu \| \pmb { x } - \pmb { b } _ { k - 1 } \| _ { 2 } ^ { 2 } ,\tag{52}
$$

with x initialized to $ { b _ { k - 1 } }$ at the start of optimization.

SITCOM (Alkhouri et al. 2024) SITCOM defines three different criteria in which DIS should satisfy: 1) forward consistency, 2) backward consistency, and 3) measurement consistency. To enable this, akin to CSGM (Bora et al. 2017), optimizes the input to the diffusion model with

$$
\pmb { x } _ { t } ^ { \prime } = \underset { \pmb { x } _ { t } } { \arg \operatorname* { m i n } } \left\| \pmb { \mathcal { A } } ( D _ { \theta } ( \pmb { x } _ { t } ) ) - \pmb { y } \right\| _ { 2 } ^ { 2 } ,\tag{53}
$$

and additionally imposing proximal constraints as in (52). Once the optimization is performed, the sampling steps follow the usual DDIM sampling steps, running (53) for every t reverse sampling steps.

## 4.3 Sequential Monte Carlo

Sequential Monte Carlo (SMC) methods, also known as particle filters, have emerged as a principled framework for solving inverse problems with diffusion priors. SMC methods enjoys the property that with increased compute (i.e. number of particles → ∞), the sampler approaches sampling from the true posterior. The particles, each representing a hypothesis about a solution, are propagated through the reverse diffusion sampling steps, re-weighted according to their consistency with respect to the observation. The algorithms mostly differ on how one constructs the proposal kernel and the reweighting values.

SMCDiff (Trippe et al. 2023) SMCDiff aims to construct a scaffold structure given a desired motif, which can be cast as a special case of the noiseless inpainting problem. Specifically, let y be the motif $( \mathrm { i . e . }$ measurement), x be the scaffold, and $\pmb { x } = [ \pmb { y } , z ]$ , i.e. $\pmb { y } \in \mathbb { R } ^ { m } , \bar { \pmb { z } } \in \mathbb { R } ^ { \bar { n } - m }$ are sub-vectors of ${ \pmb x } .$ Akin to Score-SDE (Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole 2021), one first constructs a forward-diffused motif

$$
{ \pmb y } _ { 1 : T } \sim q ( { \pmb y } _ { 1 : T } | { \pmb y } _ { 0 } ) ,\tag{54}
$$

which are prepared before the reverse diffusion sampling steps, then cached for later use. Then, for all the particles that are propagated, the sub-vector that corresponds to the motif are replaced

$$
\forall j , \pmb { x } _ { t } ^ { ( j ) } \gets [ \pmb { y } _ { t } , \pmb { z } _ { t } ^ { ( j ) } ] .\tag{55}
$$

Diffusion models for inverse problems

The un-normalized reweighting kernel is then constructed as

$$
\forall j , w _ { t } ^ { ( j ) } \gets p _ { \theta } ( \pmb { y } _ { t - 1 } | \pmb { x } _ { t } ) .\tag{56}
$$

For all reverse diffuison steps and particles, (54)-(56) along with the resampling steps are applied.

MCGDiff (Cardoso et al. 2024) MCGDiff first defines $q ( \pmb { x } _ { t } | \pmb { y } ) = \mathcal { N } ( \pmb { x } _ { t } ; \sqrt { \bar { \alpha } _ { t } } \pmb { y } , ( 1 - \bar { \alpha } _ { t } ) )$ . The proposal kernel for reverse distribution is defined as

$$
p _ { \theta } ^ { y } ( \pmb { x } _ { t } | \pmb { x } _ { t + 1 } ) \propto p _ { \theta } ( \pmb { x } _ { t } | \pmb { x } _ { t + 1 } ) q ( \pmb { x } _ { t } | \pmb { y } ) .\tag{57}
$$

For every propagated particle, the reweighting kernel is defined as

$$
w _ { t } ( \pmb { x } _ { t + 1 } ^ { ( j ) } ) = \frac { \int p _ { \theta } ( \pmb { x } _ { t } | \pmb { x } _ { t + 1 } ^ { ( j ) } ) q ( \pmb { x } _ { t } | \pmb { y } ) d \pmb { x } _ { t } } { q ( \pmb { x } _ { t + 1 } ^ { ( j ) } | \pmb { y } ) } .\tag{58}
$$

The sampling process follows the usual SMC procedure, with proposal, weighting, and resampling.

FPS (Dou & Song 2024) The core technical innovation of FPS is the construction of coupled diffusion process. In addition to the standard sequence of noisy data latents $\mathbf { \Delta } \mathbf { x } _ { t } .$ , the algorithm generates a corresponding sequence of noisy measurements $\pmb { y } _ { t } .$ , where the noise is correlated to $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ . Specifically, given the forward process

$$
\begin{array} { r } { \pmb { x } _ { t } = a _ { t } \pmb { x } _ { t - 1 } + b _ { t } \pmb { \epsilon } _ { t } , \quad \pmb { \epsilon } _ { t } \sim \mathcal { N } ( 0 , I ) , } \end{array}\tag{59}
$$

one can similarly define

$$
{ \pmb y } _ { 0 } = { \pmb y } , \quad { \pmb y } _ { t } = a _ { t } { \pmb y } _ { t - 1 } + b _ { t } A { \pmb \epsilon } _ { t } ,\tag{60}
$$

so that $\pmb { y } _ { t } \sim \mathcal { N } ( A \pmb { x } _ { t } , c _ { t } ^ { 2 } \sigma _ { y } ^ { 2 } I )$ , with $c _ { t } = a _ { 1 } a _ { 2 } \ldots a _ { t }$ . This construction leads to the following closed-form expression

$$
p _ { \theta } ( \pmb { y } _ { t - 1 } | \pmb { x } _ { t - 1 } ) = \mathcal { N } ( A \pmb { x } _ { t - 1 } , c _ { t - 1 } ^ { 2 } \sigma _ { y } ^ { 2 } I )\tag{61}
$$

and

$$
p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { y } _ { t - 1 } ) \propto p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) p _ { \theta } ( \pmb { y } _ { t - 1 } | \pmb { x } _ { t - 1 } ) .\tag{62}
$$

FPS uses (62) as the proposal kernel of the SMC procedure, and uses the following resampling weights

$$
w _ { t } ^ { ( j ) } = \frac { p _ { \theta } ( \pmb { y } _ { t } | \pmb { x } _ { t } ^ { ( j ) } ) p _ { \theta } ( \pmb { x } _ { t } ^ { ( j ) } | \pmb { x } _ { t - 1 } ^ { ( j ) } ) / p _ { \theta } ( \pmb { x } _ { t } ^ { ( j ) } | \pmb { x } _ { t + 1 } ^ { ( j ) } , \pmb { y } _ { t } ) } { \sum _ { j = 1 } ^ { M } p _ { \theta } ( \pmb { y } _ { t } | \pmb { x } _ { t } ^ { ( j ) } ) p _ { \theta } ( \pmb { x } _ { t } ^ { ( j ) } | \pmb { x } _ { t + 1 } ^ { ( j ) } ) / p _ { \theta } ( \pmb { x } _ { t } ^ { ( j ) } | \pmb { x } _ { t + 1 } ^ { ( j ) } , \pmb { y } _ { t } ) }\tag{63}
$$

Connections to inference-time scaling Singhal et al. (2025) recently drew connections to inference-time scaling of diffusion models, as SMC provides another axis (i.e. number of particles) to scale performance with compute, with guaranteed gains. Recently, FK-steering in Singhal et al. (2025) was extended to video diffusion models with a reward function that governs the 3D/4D physical consistency (Park et al. 2025).

## 5 Extension to complex tasks

## 5.1 Blind inverse problems

Often, the forward operator A is parameterized with $\varphi ,$ i.e. $\mathcal { A } _ { \varphi }$ , unlike the problems that we have considered so far, which assumed full knowledge of the forward operator. A prominent example is blind deconvolution, where the forward model is given as

$$
{ \pmb y } = { \pmb k } * { \pmb x } + { \pmb n } ,\tag{64}
$$

where k is the convolution kernel. In such case, one has to specify the posterior of both x and k

$$
\begin{array} { r } { p ( { \pmb x } , { \pmb k } | { \pmb y } ) \propto p ( { \pmb x } ) p ( { \pmb k } ) p ( { \pmb y } | { \pmb x } , { \pmb k } ) , } \end{array}\tag{65}
$$

where the factorization arises from the independence between x and $k ,$ and from $( 6 4 ) , p ( \pmb { y } | \pmb { x } , \pmb { k } ) = \mathcal { N } ( \pmb { y } , \pmb { k } * \pmb { x } , \sigma _ { y } ^ { 2 } I )$ In such case, $k = \varphi$

BlindDPS (Chung, Kim, Kim & Ye 2023) BlindDPS extends DPS by constructing another prior $p ( k )$ for the kernel by training a separate diffusion model. Following the choice of (10), one can construct two parallel PF-ODEs

$$
d \pmb { x } _ { t } = - t \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } ) d t\tag{66}
$$

$$
d \varphi _ { t } = - t \nabla _ { \varphi _ { t } } \log p ( \varphi _ { t } ) d t\tag{67}
$$

To be able to sample from the posterior given the measurement y, we can create a coupling

$$
\begin{array} { r } { d \pmb { x } _ { t } = - t [ \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } ) + \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { x } _ { t } , \pmb { \varphi } _ { t } ) ] d t } \end{array}\tag{68}
$$

$$
d \varphi _ { t } = - t [ \nabla _ { \varphi _ { t } } \log p ( \varphi _ { t } ) + \nabla _ { \varphi _ { t } } \log p ( \pmb { y } | \pmb { x } _ { t } , \varphi _ { t } ) ] d t .\tag{69}
$$

Similar to the case of non-blind inverse problems, $p ( \pmb { y } | \pmb { x } _ { t } , \varphi _ { t } )$ is intractable. BlindDPS uses the approximation proposed in DPS, but to both of the random variables $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and $\varphi _ { t } .$ , i.e. $p ( \pmb { y } | \pmb { x } _ { t } , \pmb { \varphi } _ { t } ) \approx p ( \pmb { y } | \hat { \pmb { x } } _ { 0 | t } , \pmb { \varphi } _ { 0 | t } )$ , leading to

$$
d \pmb { x } _ { t } = - t \big [ \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } ) + \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \hat { \pmb { x } } _ { 0 | t } , \hat { \varphi } _ { 0 | t } ) \big ] d t\tag{70}
$$

$$
d \pmb { \varphi } _ { t } = - t [ \nabla _ { \pmb { \varphi } _ { t } } \log p ( \pmb { \varphi } _ { t } ) + \nabla _ { \pmb { \varphi } _ { t } } \log p ( \pmb { y } | \hat { \pmb { x } } _ { 0 | t } , \hat { \pmb { \varphi } } _ { 0 | t } ) ] d t .\tag{71}
$$

In practice, the BlindDPS requires sampling Gaussian noise independently for $\mathbf { \nabla } _ { \mathbf { \mathcal { X } } \mathcal { T } }$ and $\varphi _ { T }$ , then running (70) and (71) in parallel. The likelihood is approximated with the posterior mean of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and $\varphi _ { t }$ at each step. Then, a gradient that maximizes this likelihood is applied separately to each stream.

GibbsDDRM (Murata et al. 2023) One downside of BlindDPS is that it requires training a score function for $\varphi ,$ and induces additional computational cost for calling $\nabla _ { \varphi _ { t } } \log { p ( \varphi _ { t } ) }$ with a neural network. GibbsDDRM tackles the blind deblurring problem within the DDRM family. Formally, consider the SVD of A that is dependent on the parameter of the forward operator $\varphi ,$ i.e. $A _ { \varphi } = U _ { \varphi } \Sigma _ { \varphi } \bar { V _ { \varphi } }$ , with singular values $\{ s _ { j , \varphi } \} _ { j = 1 } ^ { m } \}$ . Similar to DDRM, let $\bar { \pmb { y } } _ { \varphi } : = U _ { \varphi } ^ { \top } \pmb { y } _ { \varphi } , \bar { \pmb { x } } _ { \varphi } : = V _ { \varphi } ^ { \top } \pmb { x } _ { \varphi } , \bar { \pmb { \epsilon } } _ { \varphi } : = U _ { \varphi } ^ { \top } \pmb { \epsilon } _ { \varphi }$ , and further define $\bar { \pmb { x } } _ { 0 | t , \varphi } : = V _ { \varphi } \mathbb { E } [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ]$ . Then, the reverse distribution can be characterized as

$$
\begin{array} { r } { p ( \bar { x } _ { t , \varphi } ^ { ( i ) } | x _ { t + 1 } , y , \varphi ) = \left\{ \begin{array} { l l } { \mathcal { N } ( \bar { x } _ { t , \varphi } ^ { ( i ) } ; \bar { x } _ { 0 | t + 1 , \varphi } ^ { ( i ) } + \sqrt { 1 - \eta ^ { 2 } } \sigma _ { t } \frac { \bar { x } _ { t + 1 , \varphi } ^ { ( i ) } - \bar { x } _ { 0 | t + 1 , \varphi } ^ { ( i ) } } { \sigma _ { t + 1 } } , \eta ^ { 2 } \sigma _ { t } ^ { 2 } ) } & { \mathrm { ~ i f ~ } s _ { i , \varphi } = 0 } \\ { \mathcal { N } ( \bar { x } _ { t , \varphi } ^ { ( i ) } ; \bar { x } _ { 0 | t + 1 , \varphi } ^ { ( i ) } + \sqrt { 1 - \eta ^ { 2 } } \sigma _ { t } \frac { \bar { y } _ { \varphi } ^ { ( i ) } - \bar { x } _ { 0 | t + 1 , \varphi } ^ { ( i ) } } { \sigma _ { y } / s _ { i , \varphi } } , \eta ^ { 2 } \sigma _ { t } ^ { 2 } ) } & { \mathrm { ~ i f ~ } \sigma _ { t } < \frac { \sigma _ { y } } { s _ { i , \varphi } } \ . } \\ { \mathcal { N } ( \bar { x } _ { t , \varphi } ^ { ( i ) } ; \bar { y } _ { \varphi } ^ { ( i ) } , \sigma _ { t } ^ { 2 } - \frac { \sigma _ { y } ^ { 2 } } { s _ { i , \varphi } ^ { 2 } } ) } & { \mathrm { ~ i f ~ } \sigma _ { t } \ge \frac { \sigma _ { y } } { s _ { i , \varphi } } } \end{array} \right. } \end{array}\tag{72}
$$

Notice that $( 7 2 )$ assumes knowledge of $\varphi .$ . Akin to Gibbs sampling, the authors propose to update the random variable $\varphi$ with the following Langevin dynamics

$$
\varphi  \varphi + \frac { \xi } { 2 } \nabla _ { \varphi } \log p ( \varphi | x _ { t : T } , \pmb { y } ) + \sqrt { \xi } \pmb { \epsilon } ,\tag{73}
$$

with some step size $\xi .$ Following DPS, and placing the Laplacian prior on $\varphi ,$ the authors propose the following approximation

$$
\begin{array} { r l } & { \nabla _ { \varphi } \log p ( \varphi | x _ { t } , y ) = \nabla _ { \varphi } \log p ( y | x _ { t : T } , \varphi ) + \nabla _ { \varphi } \log p ( \varphi ) } \\ & { \qquad = \nabla _ { \varphi } \log p ( y | \hat { x } _ { 0 | t } , \varphi ) - \lambda \Vert \varphi \Vert _ { 1 } , } \end{array}\tag{74}
$$

(75)

with some constant λ.

Fast Diffusion EM (Laroche et al. 2024) Fast Diffusion EM takes an alternating expectation maximization (EM) approach. In the E-step, the approximated kernel φ is used for the usual DPS/ΠGDM sampling steps. In the M-step, an MAP optimization to maximize the posterior of the kernel is used, where the optimization problem is solved through a plug-and-play (PnP) (Venkatakrishnan et al. 2013) method with a DnCNN (Zhang et al. 2017) denoiser.

While the aforementioned approaches apply to a more general set of inverse problems, they are hard to apply to real-world image restoration tasks, as the forward model is either much more complicated, or hard to specify. For instance, the forward model of blind face restoration involves a convolution with a blur kernel, a down sampling operator, a noise component, and a JPEG degradation factor. Man et al. (2025) proposes to train a regressor to estimate these parameters, and show that using these estimated parameters together with an off-the-shelf inverse problem solver (e.g. DPS), is effective for solving inverse problems with complex forward operators.

## 5.2 3D inverse problems

The inverse problems considered so far assume the latent signal x that we wish to retrieve is a 2D image. Due to architectural advances and the ease of data collection, it is fairly easy to collect a dataset of high-quality 2D images, and to train a diffusion model on it. Nevertheless, there are many cases in computational imaging, especially in biomedical imaging, where the reconstruction of 3D volume is necessary. In such cases, however, it is both hard to collect gold-standard 3D data, and to train a diffusion model on such collected 3D dataset. One popular way to tackle this is to decompose the prior

$$
p ( { \pmb x } ) = \frac { 1 } { Z } \prod _ { i = 1 } ^ { K } p _ { i } ( f _ { i } ( { \pmb x } ) ) ,\tag{76}
$$

where $Z$ is a normalization constant, and $f _ { i }$ is an operator that captures complementary, lower-dimensional aspects of x. $\mathbf { A }$ concrete example for the case of 3D would be to choose slicing operators for $f ,$ , resulting in a factored prior over different planes.

DiffusionMBIR (Chung, Ryu, Mccann, Klasky & Ye 2023) The core idea of DiffusionMBIR is that the 3D prior over x is already captured well in the 2D prior over the xy slices. Thus, it may be sufficient to enforce smoothness across the other dimension, for instance, by using a total variation (TV) prior over the z direction. This can be achieved by iteratively applying denoising steps and measurement consistency steps, where in the measurement consistency step, the following sub-problem is solved

$$
\pmb { x } ^ { \prime } = \underset { \pmb { x } } { \arg \operatorname* { m i n } } \frac { 1 } { 2 } \| \pmb { y } - \pmb { A } \pmb { x } \| _ { 2 } ^ { 2 } + \lambda \| D _ { z } \pmb { x } \| _ { 1 } ,\tag{77}
$$

where $D _ { z }$ is the finite difference operator across $z .$ To solve (77), ADMM (Boyd et al. 2011) is used, with CG steps operating to solve the inner problem. However, notice that this would require immense computation cost, as the iterative ADMM would have to be solved for every t. To mitigate this cost, a variable sharing technique was proposed so that the primal and dual variables are warm-started from the previous iteration $t + 1$ , and only a single iteration of ADMM is applied to each optimization step. Later, in Chung, Lee & Ye (2024), it was shown that one can improve the performance of DiffusionMBIR by using the Tweedie estimates $\hat { \mathbf { \mathscr { x } } } _ { 0 \mid t }$ for optimization in (77), instead of the noisy variables $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$

TPDM (Lee et al. 2023) Another way to construct a factored prior is to use two diffusion diffusion priors for different slice directions. Compared to DiffusionMBIR, this further alleviates hand-crafted inductive bias and replaces it with a data-driven generative prior, and was shown to outperform DiffusionMBIR across several tasks, especially on tasks such as super-resolution. One way to implement the product distribution is by using the sum of the scores

$$
\nabla _ { \pmb { x } _ { t } } \log { p ( \pmb { x } _ { t } ) } = \alpha \nabla _ { \pmb { x } _ { t } } \log { q ^ { ( p ) } ( \pmb { x } _ { t } ) } + \beta \nabla _ { \pmb { x } _ { t } } \log { q ^ { ( a ) } ( \pmb { x } _ { t } ) } ,\tag{78}
$$

where $q ^ { ( p ) }$ is the distribution of the slices in the primary plane, $q ^ { ( a ) }$ is the distribution of the slices in the auxiliary plane (i.e. orthogonal to the primary plane), and $\alpha , \beta$ are mixing constants. In practice, directly using (78) would incur double the computation cost during inference. To mitigate this, Lee et al. (2023) proposes to use an alternating approach, using only the score from the primary plane for $\frac { \alpha } { \alpha + \beta }$ fraction of the time during reverse sampling, and using only the score from the auxiliary plane for $\frac { \beta } { \alpha + \beta }$ for the rest. In order to impose measurement consistency, DPS steps are employed.

## 5.3 Inverse problems under data scarcity

All diffusion model-based inverse problem solvers rely on the assumption that one has access to a diffusion model trained on high-quality in-distribution datasets. This condition is not satisfied. For instance, in black-hole imaging (Akiyama et al. 2019) and cryo-EM imaging (Gupta et al. 2021), one only has access to the partial measurements, with no access whatsoever on how the true image would look like. In this section, we review some of the approaches that operate under such constraints.

## 5.3.1 Test-time adaptation

One way to solve this problem is to use a diffusion model trained on a separate dataset, and try to adapt the diffusion model on out-of-distribution (OOD) measurements online Barbano et al. (2025), Chung & Ye (2024). The approaches build on top of deep image prior (DIP) (Ulyanov et al. 2018), which overfits a network on a single measurement, relying on the inductive prior of the neural network

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \| y - A G _ { \theta } ( z ) \| _ { 2 } ^ { 2 } ,\tag{79}
$$

where $G _ { \theta }$ is the network for reconstruction, which takes in a random input $z \sim \mathcal { N } ( 0 , I )$

Deep Diffusion Image Prior (DDIP) (Chung & Ye 2024) generalizes and extends DIP to work within the diffusion framework by alternating the following steps

$$
\mathrm { f o r } t = T , \dots , 1 : \theta _ { t - 1 }  \underset { \theta _ { t } } { \arg \operatorname* { m i n } } \ | | \pmb { y } - A D _ { \theta _ { t } } ( \pmb { x } _ { t } | \pmb { y } ) | | _ { 2 } ^ { 2 } ,\tag{80}
$$

$$
\begin{array} { r } { \pmb { x } _ { t - 1 }  \mathrm { D D I M } _ { \theta _ { t - 1 } } ( D _ { \theta _ { t - 1 } } ( \pmb { x } _ { t } | \pmb { y } ) , \eta ) , } \end{array}\tag{81}
$$

where $\mathrm { D D I M } _ { \theta } ( \pmb { x } _ { t } , \eta ) : = \sqrt { \bar { \alpha } _ { t - 1 } } D _ { \theta } ( \pmb { x } _ { t } | \pmb { y } ) + \sqrt { 1 - \bar { \alpha } _ { t - 1 } } \left( \eta \epsilon + ( 1 - \eta ) \epsilon ^ { \theta } \right)$ . Notice that DDIP differs from DIP in two aspects. First, the reconstructor $G _ { \theta }$ is replaced with an MMSE denoiser $D _ { \theta } .$ , which stems from a pre-trained diffusion model, and hence the generation trajectory is pivoted in the original generative process. Second, the DIP adaptation in (80) is held across multiple scales (i.e. noise levels $t ) ,$ different from a single-scale optimization of DIP. In practice, the original parameters θ are hold constant, and only the low-rank adaptation (LoRA) is applied to make partial updates to the network.

Patch-based priors Factored priors that are widely employed within the 3D medical imaging setting, but were also shown to be useful for 2D inverse problems, for instance, by using patch-based priors (Hu et al. 2024). By employing positional encodings, PaDIS (Hu et al. 2024) constructs a position-aware patch-based diffusion model, showing that such approach is better than image diffusion model counterparts, especially in the data-scare regime. Later, the patch-based diffusion approach was combined with test-time adaptation in Hu et al. (2025).

## 5.4 Training a diffusion model with noisy data

GSURE-based diffusion model (Kawar et al. 2024) Stein’s Unbiased Risk Estimator (SURE) (Stein 1981) is a widely used method to train a denoiser given only the Gaussian-noisy measurements. Later, this was extended to a general set of linear inverse problems of the form (1) in Generalized SURE (GSURE) (Eldar 2008), which states the following

$$
\begin{array} { r } { \mathbb { E } \left[ \| P ( D ( \pmb { y } ) - \pmb { x } ) \| _ { 2 } ^ { 2 } \right] = \mathbb { E } \left[ \| P ( D ( \pmb { y } ) - \pmb { x } _ { M L } \| _ { 2 } ^ { 2 } \right] + 2 \mathbb { E } \left[ \nabla _ { A ^ { \top } \pmb { y } } \cdot P D ( \pmb { y } ) \right] + c , } \end{array}\tag{82}
$$

where $P = A ^ { \top } A$ and $\begin{array} { r } { \pmb { x } _ { M L } = ( A ^ { \top } A ) ^ { \dagger } A ^ { \top } \pmb { y } . } \end{array}$ While (82) guarantees a good denoiser in the sense of projected MSE, this ceases to be a good surrogate when the operator A removes sufficient information from x (i.e. when the mask is large). In such case, one can use ENsmeble SURE (ENSURE) (Aggarwal et al. 2022) by also marginalizing over the operator A, given the assumption that we have access to A and the noise level, and the different realizations of A covers the signal space $\mathbb { R } ^ { n }$ . Note that this assumption is satisfied, for instance, in MRI acquisitions. GSURE-based diffusion follows this assumption and leverages ENSURE to train a diffusion model from the measurements only.

Following the similar procedure from the DDRM family introduced in Sec. 3.1, we transform the inverse problem into

$$
\bar { \pmb { y } } = P \bar { \pmb { x } } + \bar { \pmb { z } } , \quad \bar { \pmb { z } } \sim \mathcal { N } ( 0 , \sigma _ { y } ^ { 2 } \Sigma ^ { \dag } \Sigma ^ { \dag \top } ) ,\tag{83}
$$

where $A = U \Sigma V ^ { \top } , P = \Sigma ^ { \dagger } \Sigma , \bar { \pmb { x } } = V ^ { \top } \pmb { x } , \bar { \pmb { y } } = \Sigma ^ { \dagger } U ^ { \top } \pmb { y } , \bar { z } = \Sigma ^ { \dagger } U ^ { \top } \pmb { z }$ . GSURE-diffusion then constructs the following forward perturbation

$$
\bar { \mathbf { x } } _ { t } = \sqrt { \bar { \alpha } _ { t } } \pmb { y } + \left( \left( 1 - \bar { \alpha } _ { t } \right) \bar { I } - \bar { \alpha } _ { t } \sigma _ { y } ^ { 2 } \Sigma ^ { \dagger } \Sigma ^ { \dagger \top } \right) ^ { 1 / 2 } \epsilon ,\tag{84}
$$

and by this design choice, the marginal distribution of $\bar { \mathbf { x } } _ { t }$ reads $q ( \bar { \pmb { x } } _ { t } | \bar { \pmb { x } } , P ) = \mathcal { N } ( \sqrt { \bar { \alpha } _ { t } } P \bar { \pmb { x } } , ( 1 - \bar { \alpha } _ { t } ) I )$ . The objective function then reads

$$
\sum _ { t = 1 } ^ { T } \gamma _ { t } \mathbb { E } \left[ \left. W P \left( D _ { \theta } ( \bar { \mathbf { x } } _ { t } ) - \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \bar { \mathbf { x } } _ { t } \right) \right. _ { 2 } ^ { 2 } + 2 ( 1 - \bar { \alpha } _ { t } ) \left( \nabla _ { \bar { \mathbf { x } } _ { t } } \cdot P W ^ { 2 } D _ { \theta } ( \bar { \mathbf { x } } _ { t } ) \right) + c \right] ,\tag{85}
$$

where $W = \mathbb { E } [ P ] ^ { - \frac { 1 } { 2 } } \succ 0$ . It was shown in Kawar et al. (2024) that by training a diffusion model solely on measuremenets with (85) yields similar to performance to the diffusion models trained on clean samples x.

## 5.4.1 Ambient Diffusion Family

Ambient Diffusion (Daras, Shah, Dagan, Gollakota, Dimakis & Klivans 2023) Ambient Diffusion considers a special case of learning a diffusion model from noiseless-masked measurements $\pmb { y } _ { 0 } = \pmb { A } \pmb { x } _ { 0 }$ with the same assumptions as in GSURE-diffusion. Consider the following naive loss

$$
J ^ { \mathrm { n a i v e } } ( \theta ) = \mathbb { E } \left[ \| A ( D _ { \theta } ( A , A { \boldsymbol { x } } _ { t } , t ) - { \boldsymbol { x } } _ { 0 } ) \| _ { 2 } ^ { 2 } \right] ,\tag{86}
$$

where the loss simply ignores the missing pixels, and computes the loss only on known ones. Training a diffusion model with (86), however, would not lead the network to learn any information about the unknown pixel values. To mitigate this, the authors propose to sample a second mask $B ,$ and set $\tilde { A } = B A$ . Then, the loss of Ambient Diffusion reads

$$
J ^ { \mathrm { c o r r } } ( \theta ) = \mathbb { E } \left[ \left| \left| A ( D _ { \theta } ( \tilde { A } , \tilde { A } \mathbf { x } _ { t } , t ) - \mathbf { x } _ { 0 } ) \right| \right| _ { 2 } ^ { 2 } \right]\tag{87}
$$

Since the network $D _ { \theta }$ cannot distinguish between the old and new masked pixels, the safest way would be to reconstruct every pixel. Under mild assumptions on A, B, one can also show that $D _ { \theta ^ { * } } ( \tilde { A } , \pmb { x } _ { t } ) = \mathbb { E } [ \pmb { x } _ { 0 } | \tilde { A } \pmb { x } _ { t } , \tilde { A } ]$

Consistent Diffusion meets Tweedie (Daras, Dimakis & Daskalakis 2024) Daras, Dimakis & Daskalakis (2024) considers training a diffusion model with Gaussian noise-corrupted samples, where $A = I .$ . Let the noise level of the samples be $t _ { n } .$ . Notice that for $t > t _ { n } ,$ , we can express the random variable $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ in two distinct ways: ${ \pmb x } _ { t } = { \pmb x } _ { 0 } + \sigma _ { t } { \pmb \epsilon }$ and ${ \pmb x } _ { t } = { \pmb x } _ { t _ { n } } + \sqrt { \sigma _ { t } ^ { 2 } - \sigma _ { t _ { n } } ^ { 2 } } { \pmb \epsilon } ,$ . By applying Tweedie’s formula twice, one can conclude

$$
\mathbb { E } [ { \pmb x } _ { t _ { n } } | { \pmb x } _ { t } ] = \frac { \sigma _ { t } ^ { 2 } - \sigma _ { t _ { n } } ^ { 2 } } { \sigma _ { t } ^ { 2 } } \left( \mathbb { E } [ { \pmb x } _ { 0 } | { \pmb x } _ { t } ] - { \pmb x } _ { t } \right) + { \pmb x } _ { t } .\tag{88}
$$

The implication of (88) is that one can train an optimal denoiser for noise levels $t > t _ { n }$ by training the model to remove only the additional noise from $t _ { n }$ to $t ,$ i.e. train the model with

$$
\mathbb { E } _ { t \sim \mathcal { U } ( t _ { n } , T ] } \left[ \left. \frac { \sigma _ { t } ^ { 2 } - \sigma _ { t _ { n } } ^ { 2 } } { \sigma _ { t } ^ { 2 } } D _ { \theta } ( \pmb { x } _ { t } , t ) + \frac { \sigma _ { t _ { n } } ^ { 2 } } { \sigma _ { t } ^ { 2 } } \pmb { x } _ { t } - \pmb { x } _ { t _ { n } } \right. _ { 2 } ^ { 2 } \right] .\tag{89}
$$

For noise levels $t < t _ { n }$ , one can leverage the idea from Consistent diffusion (Daras, Dagan, Dimakis & Daskalakis 2023), where the objective reads

$$
\begin{array} { r } { \mathbb { E } _ { t \sim \mathcal { U } ( t _ { n } , T ] , t ^ { \prime } \sim \mathcal { U } ( \varepsilon , T ) , t ^ { \prime \prime } \sim \mathcal { U } ( t ^ { \prime } - \varepsilon , t ^ { \prime } ) } \left[ \left\| D _ { \theta } ( \boldsymbol { x } _ { t ^ { \prime } } , t ^ { \prime } ) - \mathbb { E } _ { \boldsymbol { x } _ { t ^ { \prime \prime } } \sim p _ { \theta } ( \boldsymbol { x } _ { t ^ { \prime \prime } } | \boldsymbol { x } _ { t ^ { \prime } } ) } [ D _ { \theta } ( \boldsymbol { x } _ { t ^ { \prime \prime } } , t ^ { \prime \prime } ) ] \right\| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{90}
$$

Notice that in $( 9 0 ) , t ^ { \prime } > t ^ { \prime \prime }$ , and the sampling of $p _ { \theta } ( \pmb { x } _ { t ^ { \prime \prime } } | \pmb { x } _ { t ^ { \prime } } )$ is achieved through taking a step ε through a diffusion model, a similar procedure to consistency models (Song, Dhariwal, Chen & Sutskever 2023). By training a diffusion model to be consistent with its counterpart taken two steps, one can achieve an optimal denoiser even for $t < t _ { n }$ . Thus, the final objective of Daras, Dimakis & Daskalakis (2024) takes a weighted sum of the two objectives (89), (90).

Ambient Diffusion Omni (Daras et al. 2025) Previous works relied on the assumption that one knows the noise level of the corrupted measurement. Ambient Diffusion Omni relaxes this assumption and considers the case where the diffusion model is trained on a mixture distribution of $p _ { 0 }$ and $q _ { 0 }$ , where $p _ { 0 }$ is the clean data distribution, and $q _ { 0 }$ is the corrupted distribution containing arbitrary mix of bad-quality data (e.g. blur, noise, JPEG artifacts, etc.). In the practical scenario when training a diffusion model for deployment, one would filter out the samples in $q _ { 0 }$ and use only the ones in $p _ { 0 }$ . However, in Ambient Diffusion Omni, the authors propose a way to utilize both the data from $p _ { 0 }$ and $q _ { 0 }$ showing that one can achieve better quality by using data from both sources.

Due to the contracting property of diffusion models (Chung, Sim & Ye 2022), when noise is added, $p _ { t }$ and $q _ { t }$ become closer to each other. The key idea of Ambient Diffusion Omni is to train a classifier that distinguishes high and low quality samples at a certain timestep. The minimum timestep $t _ { n } ^ { \mathrm { m i n } }$ is distinguished for each sample. Then, when training the diffusion model, one only uses the timesteps $t \geq t _ { n } ^ { \operatorname* { m i n } }$ for some sample $n .$

## 5.4.2 Expectation-Maximization (EM)

EM tries to find the best parameter θ of the model that best explains the observation $\mathbf { \pmb { y } } .$ . The challenge is that we do not know the underlying clean data x. To circumvent this issue, EM takes a two-stage approach.

1. (E-step): Use the current model $\theta _ { k }$ to specify the posterior $p _ { \theta _ { k } } ( \pmb { x } | \pmb { y } )$ and specify in expectation what the complete data looks like

2. (M-step): Given the probabilistic guess about the hidden data x, maximize the log-likelihood of the model to get $\theta _ { k + 1 }$

The key idea behind the EM algorithm is that for any $\theta _ { a }$ and $\theta _ { b }$ , we have

$$
\log \frac { p _ { \theta _ { a } } ( \pmb { x } ) } { p _ { \theta _ { b } } ( \pmb { x } ) } \geq \mathbb { E } _ { p ( \pmb { y } ) } \mathbb { E } _ { q _ { \theta _ { b } } ( \pmb { x } | \pmb { y } ) } \left[ \frac { p _ { \theta _ { a } } ( \pmb { x } ) } { p _ { \theta _ { b } } ( \pmb { x } ) } \right]\tag{91}
$$

Hence, the iteration of EM leads to a sequence of parameters $\theta _ { k }$ where the expected log evidence $\mathbb { E } _ { p ( \pmb { y } ) } [ \log p _ { \theta _ { k } } ( \pmb { y } ) ]$ monotonically increases and converges to a local optimum.

Rozet et al. (2024) Notice that

$$
\theta _ { k + 1 } = \arg \operatorname* { m a x } _ { \theta } \mathbb { E } _ { p ( y ) } \mathbb { E } _ { p _ { \theta _ { k } } ( x | y ) } [ \log p _ { \theta } ( \pmb { x } ) + \log p ( \pmb { y } | \pmb { x } ) ]\tag{92}
$$

$$
= \underset { \theta } { \arg \operatorname* { m a x } } \mathbb { E } _ { p ( \pmb { y } ) } \mathbb { E } _ { p _ { \theta _ { k } } ( \pmb { x } | \pmb { y } ) } [ \log p _ { \theta } ( \pmb { x } ) ]\tag{93}
$$

$$
= \underset { \theta } { \arg \operatorname* { m i n } } \mathrm { K L } ( \pi _ { k } ( \pmb { x } ) \| p _ { \theta } ( \pmb { x } ) ) ,\tag{94}
$$

where $\begin{array} { r } { \pi _ { k } ( { \pmb x } ) = \int p _ { \theta _ { k } } ( { \pmb x } | { \pmb y } ) p ( { \pmb y } ) d { \pmb y } } \end{array}$ . In practice, given a sample y, the authors propose to use a posterior sampler, namely the moment matching method discussed in Sec. 3 to draw from $\pi _ { k } ( { \pmb x } )$ . Then, with the collected samples, the M-step is performed by standard DSM. A concurrent work of Bai et al. (2024) uses the same EM framework, but uses DPS to draw posterior samples in the E-step.

## 6 Text-driven solutions

Since inverse problems are ill-posed, measurement does not provide sufficient information for perfect recovery. It is natural that if one could use additional auxiliary information for recovery, it would be beneficial to do so. As one such side information, text has recently gained attention, as they enable compact, informative, and highly versatile conditioning.

Often, Latent Diffusion Models (LDMs) (Vahdat et al. 2021, Rombach et al. 2022) enable effective entanglement of multi-modal representations, and are considered the de facto standard for modern text-to-image diffusion models. In this section, let $z _ { 0 } = \mathcal { E } ( \pmb { x } _ { 0 } )$ ) be a latent code of clean image encoded by VAE encoder E. Original image can be reconstructed $\pmb { x } _ { 0 } = \mathcal { D } ( \pmb { z } _ { 0 } ) = \mathcal { D } ( \pmb { \mathcal { E } } ( \pmb { x } _ { 0 } ) )$ ) by VAE decoder D. The diffusion model is now defined on the latent space.

P2L (Chung, Ye, Milanfar & Delbracio (2024)) P2L demonstrates the effectiveness of text embedding space for improving quality of solution. The authors propose an extension of DPS for LDMs (Rout, Raoof, Daras, Caramanis, Dimakis & Shakkottai 2024), by using

$$
\nabla _ { z _ { t } } \log p ( \pmb { y } | z _ { t } ) \approx \nabla _ { z _ { t } } \log p ( \pmb { y } | \mathcal { D } ( \mathbb { E } [ z _ { 0 } | z _ { t } ] ) ) .\tag{95}
$$

Now, to consider the text embedding as an optimization variable, they formulate an optimization problem

$$
\begin{array} { r l } { \underset { z \sim p ( z | y ) } { \operatorname* { m i n } } ~ \underset { c } { \operatorname* { m i n } } } & { { } \| y - A \mathcal { D } ( z ^ { ( c ) } ) \| ^ { 2 } } \\ { \mathrm { s u b j e c t ~ t o } } & { { } z \in F _ { X } } \end{array}\tag{96}
$$

where $F _ { X } = \{ z | z = \mathcal { E } ( z )$ for somex} denotes the set of latent that can be represented by some image x. Optimization involves two alternative updates,

$$
\pmb { c } ^ { * } = \arg \operatorname* { m i n } _ { \pmb { c } } \| \pmb { y } - \boldsymbol { A } \mathcal { D } ( \hat { z } _ { 0 | t } ^ { ( \pmb { c } ) } ) \| ^ { 2 }\tag{97}
$$

for the prompt embedding and

$$
\begin{array} { r l } & { z ^ { * } = \mathcal { E } ( x ^ { * } ) \quad \mathrm { w h e r e } } \\ & { { \pmb x } ^ { * } = \underset { { \pmb x } } { \arg \operatorname* { m i n } } \| { \pmb y } - A { \pmb x } \| ^ { 2 } + \lambda \| { \pmb x } - \mathcal { D } ( \hat { z } _ { 0 | t } ^ { ( c ^ { * } ) } ) \| ^ { 2 } } \end{array}\tag{98}
$$

for the MMSE estimate during reverse sampling, which considers not only data fidelity but also latent fidelity, encouraging the optimized latent to lie within the range of VAE encoder. This ensures that the decoded output remains on the image manifold. The joint update of prompt embedding during diffusion reverse sampling leads solution to be more aligned to the pre-trained diffusion prior, compared to using null-text embedding.

TReg (Kim, Park, Chung & Ye (2025)) While P2L reduces the gap between the latent diffusion prior and the solution obtained via null-text embedding, it does not leverage the text prompt as an additional prior to guide the solution. To address this limitation, TReg introduces the concept of Regularization by text, which further constrains the solution space toward a conditional prior distribution, implemented via a latent-space optimization problem. By applying Bayes’ rule to posterior distribution involving latent variable z, we obtain:

$$
p ( { \pmb x } | { \pmb y } , z ) \propto p ( { \pmb y } | { \pmb x } , z ) p ( { \pmb x } | z ) \propto p ( z | { \pmb x } , { \pmb y } ) p ( { \pmb y } | { \pmb x } ) p ( { \pmb x } | z ) .\tag{99}
$$

TReg formulates a Maximum A Posterior (MAP) optimization problem with text regularization term applied on MMSE estimate space during reverse sampling as

$$
\operatorname* { m i n } _ { z , x } \underbrace { \| z - \mathcal { E } ( \mathcal { D } ( z ) ) \| ^ { 2 } + \| y - A \mathcal { D } ( z ) \| ^ { 2 } } _ { \ell _ { \mathrm { M A P } } } + \lambda \underbrace { \| z - \hat { z } _ { 0 \mid t } \| ^ { 2 } } _ { \ell _ { \mathrm { T r e g } } } \quad \mathrm { s . t . } \quad x = \mathcal { D } ( z )\tag{100}
$$

where $p ( \pmb { x } | \pmb { z } ) : = \delta ( \pmb { x } - \pmb { \mathcal { D } } ( \pmb { z } ) ) , \hat { z } _ { 0 | t }$ denotes text conditioned denoised estimate, and z is initialized with $\hat { z } _ { 0 \mid t }$ . The regularization term steers the sampling trajectory toward a clean manifold aligned with the text condition c. Combined with the MAP objective enforcing data fidelity, this approach yields solutions that satisfies both the text condition and data consistency with given measurement, thereby improving reconstruction quality, especially under severe degraded conditions. TReg also introduces adaptive negation, which optimizes the null-text embedding to suppress concepts unrelated to the text condition c by minimizing CLIP similarity with the denoised estimate.

ContextMRI (Chung et al. 2025) Text-driven solutions were also adopted in the medical imaging domain, by using metadata as a conditioning signal. The leveraged metadata include patient demographics, the location including slice number and anatomy, MRI imaging parameters including TR, TE, TI, and even (optionally) pathology. The authors trained a diffusion model in the pixel space with a CLIP encoder that takes in as input the metadata represented as text, and use this model as the prior for MRI reconstruction. It was shown that in all cases, the conditional diffusion model performs better than the unconditional counterpart.

## 7 Discussion and Conclusion

In this chapter, we gave a comprehensive overview of using diffusion models for inverse problems, primarily focused on the general zero-shot solvers that does not involve task-specific training, and thus can be adapted to various applications. Our survey deliberately omitted two related areas to maintain this focus: solvers designed explicitly for Latent Diffusion Models (LDMs) (Rout et al. 2023, Rout, Chen, Kumar, Caramanis, Shakkottai & Chu 2024, Raphaeli et al. 2025)—whose underlying principles largely align with the methods discussed—and approaches based on diffusion bridges (Delbracio & Milanfar 2023, Luo et al. 2023, Liu, Vahdat, Huang, Theodorou, Nie & Anandkumar 2023, Chung, Kim & Ye 2023), which necessitate supervised training.

A key takeaway is the inherent trade-off among the surveyed methods. Solvers present a spectrum of design choices, balancing computational speed against reconstruction fidelity and exactness. The selection of an appropriate method is therefore contingent upon the specific constraints and goals of the target application.

The diversity of these powerful techniques signifies a rapidly maturing field. As these tools become more robust, they offer practitioners a versatile and adaptable toolkit for a wide range of scientific and creative applications. Future work will likely focus on reconciling the trade-offs between speed and accuracy, pushing the boundaries of what is achievable in unsupervised inverse problem-solving.

## References

Aggarwal, H. K., Pramanik, A., John, M. & Jacob, M. (2022), ‘Ensure: A general approach for unsupervised training of deep image reconstruction algorithms’, IEEE transactions on medical imaging 42(4), 1133–1144.

Akiyama, K., Alberdi, A., Alef, W., Asada, K., Azulay, R., Baczko, A.-K., Ball, D., Balokovic, M., Barrett, J., Bintley,´ D. et al. (2019), ‘First m87 event horizon telescope results. iv. imaging the central supermassive black hole’, The Astrophysical Journal Letters 875(1), L4.

Alkhouri, I., Liang, S., Huang, C.-H., Dai, J., Qu, Q., Ravishankar, S. & Wang, R. (2024), ‘Sitcom: Step-wise triple-consistent diffusion sampling for inverse problems’, arXiv preprint arXiv:2410.04479 .

Anderson, B. D. (1982), ‘Reverse-time diffusion equation models’, Stochastic Processes and their Applications 12(3), 313–326.

Bai, W., Wang, Y., Chen, W. & Sun, H. (2024), ‘An expectation-maximization algorithm for training clean diffusion models from corrupted observations’, Advances in Neural Information Processing Systems 37, 19447–19471.

Barbano, R., Denker, A., Chung, H., Roh, T. H., Arridge, S., Maass, P., Jin, B. & Ye, J. C. (2025), ‘Steerable conditional diffusion for out-of-distribution adaptation in medical image reconstruction’, IEEE Transactions on Medical Imaging

Blau, Y. & Michaeli, T. (2018), The perception-distortion tradeoff, in ‘Proceedings of the IEEE conference on computer vision and pattern recognition’, pp. 6228–6237.

Bora, A., Jalal, A., Price, E. & Dimakis, A. G. (2017), Compressed sensing using generative models, in ‘International conference on machine learning’, PMLR, pp. 537–546.

Boyd, S., Parikh, N. & Chu, E. (2011), Distributed optimization and statistical learning via the alternating direction method of multipliers, Now Publishers Inc.

Cardoso, G., el idrissi, Y. J., Corff, S. L. & Moulines, E. (2024), Monte carlo guided denoising diffusion models for bayesian linear inverse problems., in ‘The Twelfth International Conference on Learning Representations’. URL: https://openreview.net/forum?id=nHESwXvxWK

Chen, R. T. Q., Rubanova, Y., Bettencourt, J. & Duvenaud, D. K. (2018), Neural ordinary differential equations, in ‘Advances in Neural Information Processing Systems’, Vol. 31.

Chung, H., Kim, J., Kim, S. & Ye, J. C. (2023), ‘Parallel diffusion models of operator and image for blind inverse problems’, IEEE/CVF Conference on Computer Vision and Pattern Recognition .

Chung, H., Kim, J., Mccann, M. T., Klasky, M. L. & Ye, J. C. (2023), Diffusion posterior sampling for general noisy inverse problems, in ‘International Conference on Learning Representations’. URL: https://openreview.net/forum?id=OnD9zGAGT0k

Chung, H., Kim, J. & Ye, J. C. (2023), ‘Direct diffusion bridge using data consistency for inverse problems’, Advances in Neural Information Processing Systems 36, 7158–7169.

Chung, H., Lee, D., Wu, Z., Kim, B.-H., Bouman, K. L. & Ye, J. C. (2025), ‘Contextmri: Enhancing compressed sensing mri through metadata conditioning’, arXiv preprint arXiv:2501.04284 .

Chung, H., Lee, S. & Ye, J. C. (2024), Decomposed diffusion sampler for accelerating large-scale inverse problems, in ‘The Twelfth International Conference on Learning Representations’. URL: https://openreview.net/forum?id=DsEhqQtfAG

Chung, H., Ryu, D., Mccann, M. T., Klasky, M. L. & Ye, J. C. (2023), ‘Solving 3d inverse problems using pre-trained 2d diffusion models’, IEEE/CVF Conference on Computer Vision and Pattern Recognition .

Chung, H., Sim, B., Ryu, D. & Ye, J. C. (2022), Improving diffusion models for inverse problems using manifold constraints, in A. H. Oh, A. Agarwal, D. Belgrave & K. Cho, eds, ‘Advances in Neural Information Processing Systems’. URL: https://openreview.net/forum?id=nJJjv0JDJju

Chung, H., Sim, B. & Ye, J. C. (2022), Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction, in ‘Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition’.

Chung, H. & Ye, J. C. (2024), Deep diffusion image prior for efficient ood adaptation in 3d inverse problems, in ‘European Conference on Computer Vision’, Springer, pp. 432–455.

Chung, H., Ye, J. C., Milanfar, P. & Delbracio, M. (2024), Prompt-tuning latent diffusion models for inverse problems, in ‘Forty-first International Conference on Machine Learning’. URL: https://openreview.net/forum?id=hrwIndai8e

Daras, G., Chung, H., Lai, C.-H., Mitsufuji, Y., Ye, J. C., Milanfar, P., Dimakis, A. G. & Delbracio, M. (2024), ‘A survey on diffusion models for inverse problems’, arXiv preprint arXiv:2410.00083 .

Daras, G., Dagan, Y., Dimakis, A. & Daskalakis, C. (2023), ‘Consistent diffusion models: Mitigating sampling drift by learning to be consistent’, Advances in Neural Information Processing Systems 36, 42038–42063.

Daras, G., Dimakis, A. & Daskalakis, C. C. (2024), Consistent diffusion meets tweedie: Training exact ambient diffusion models with noisy data, in ‘Forty-first International Conference on Machine Learning’. URL: https://openreview.net/forum?id=PlVjIGaFdH

Daras, G., Rodriguez-Munoz, A., Klivans, A., Torralba, A. & Daskalakis, C. (2025), ‘Ambient diffusion omni: Training good models with bad data’, arXiv preprint arXiv:2506.10038 .

Daras, G., Shah, K., Dagan, Y., Gollakota, A., Dimakis, A. & Klivans, A. (2023), ‘Ambient diffusion: Learning clean distributions from corrupted data’, Advances in Neural Information Processing Systems 36, 288–313.

Delbracio, M. & Milanfar, P. (2023), ‘Inversion by direct iteration: An alternative to denoising diffusion for image restoration’, arXiv preprint arXiv:2303.11435 .

Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2017), Density estimation using real NVP, in ‘International Conference on Learning Representations’. URL: https://openreview.net/forum?id=HkpbnH9lx

Dou, Z. & Song, Y. (2024), Diffusion posterior sampling for linear inverse problem solving: A filtering perspective, in ‘The Twelfth International Conference on Learning Representations’. URL: https://openreview.net/forum?id=tplXNcHZs1

Efron, B. (2011), ‘Tweedie’s formula and selection bias’, Journal of the American Statistical Association 106(496), 1602– 1614.

Eldar, Y. C. (2008), ‘Generalized sure for exponential families: Applications to regularization’, IEEE Transactions on Signal Processing 57(2), 471–481.

Erbach, J., Narnhofer, D., Dombos, A., Schiele, B., Lenssen, J. E. & Schindler, K. (2025), ‘Solving inverse problems with flair’, arXiv preprint arXiv:2506.02680 .

Feng, B. & Bouman, K. (2024), ‘Variational bayesian imaging with an efficient surrogate score-based prior’, Transactions on Machine Learning Research . URL: https://openreview.net/forum?id=db2pFKVcm1

Feng, B. T., Smith, J., Rubinstein, M., Chang, H., Bouman, K. L. & Freeman, W. T. (2023), Score-based diffusion models as principled priors for inverse imaging, in ‘Proceedings of the IEEE/CVF International Conference on Computer Vision’, pp. 10520–10531.

Gao, R., Hoogeboom, E., Heek, J., Bortoli, V. D., Murphy, K. P. & Salimans, T. (2024), Diffusion meets flow matching: Two sides of the same coin. URL: https://diffusionflow.github.io/

Gupta, H., McCann, M. T., Donati, L. & Unser, M. (2021), ‘Cryogan: A new reconstruction paradigm for single-particle cryo-em via deep adversarial learning’, IEEE Transactions on Computational Imaging 7, 759–774.

He, Y., Murata, N., Lai, C.-H., Takida, Y., Uesaka, T., Kim, D., Liao, W.-H., Mitsufuji, Y., Kolter, J. Z., Salakhutdinov, R. & Ermon, S. (2024), Manifold preserving guided diffusion, in ‘The Twelfth International Conference on Learning Representations’. URL: https://openreview.net/forum?id=o3BxOLoxm1

Ho, J., Jain, A. & Abbeel, P. (2020), ‘Denoising diffusion probabilistic models’, Advances in Neural Information Processing Systems 33, 6840–6851.

Hu, J., Song, B., Fessler, J. A. & Shen, L. (2025), ‘Test-time adaptation improves inverse problem solving with patch-based diffusion models’, IEEE Transactions on Computational Imaging .

Hu, J., Song, B., Xu, X., Shen, L. & Fessler, J. A. (2024), ‘Learning image priors through patch-based diffusion models for solving inverse problems’, Advances in Neural Information Processing Systems 37, 1625–1660.

Huang, C.-W., Lim, J. H. & Courville, A. (2021), ‘A variational perspective on diffusion-based generative models and score matching’, arXiv preprint arXiv:2106.02808 .

Hyvärinen, A. & Dayan, P. (2005), ‘Estimation of non-normalized statistical models by score matching.’, Journal of Machine Learning Research 6(4).

Jalal, A., Arvinte, M., Daras, G., Price, E., Dimakis, A. G. & Tamir, J. (2021), ‘Robust compressed sensing mri with deep generative priors’, Advances in Neural Information Processing Systems 34.

Kadkhodaie, Z. & Simoncelli, E. P. (2021), Stochastic solutions for linear inverse problems using the prior implicit in a denoiser, in A. Beygelzimer, Y. Dauphin, P. Liang & J. W. Vaughan, eds, ‘Advances in Neural Information Processing Systems’. URL: https://openreview.net/forum?id=x5hh6N9bUUb

Kawar, B., Elad, M., Ermon, S. & Song, J. (2022), Denoising diffusion restoration models, in A. H. Oh, A. Agarwal, D. Belgrave & K. Cho, eds, ‘Advances in Neural Information Processing Systems’. URL: https://openreview.net/forum?id=kxXvopt9pWK

Kawar, B., Elata, N., Michaeli, T. & Elad, M. (2024), ‘GSURE-based diffusion model training with corrupted data’, Transactions on Machine Learning Research . URL: https://openreview.net/forum?id=BRl7fqMwaJ

Kawar, B., Vaksman, G. & Elad, M. (2021), ‘Snips: Solving noisy inverse problems stochastically’, Advances in Neural Information Processing Systems 34, 21757–21769.

Kim, J., Kim, B. S. & Ye, J. C. (2025), ‘Flowdps: Flow-driven posterior sampling for inverse problems’, arXiv preprint arXiv:2503.08136 .

Kim, J., Park, G. Y., Chung, H. & Ye, J. C. (2025), Regularization by texts for latent diffusion inverse solvers, in ‘The Thirteenth International Conference on Learning Representations’. URL: https://openreview.net/forum?id=TtUh0TOlGX

Kim, K. & Ye, J. C. (2021), ‘Noise2Score: Tweedie’s Approach to Self-Supervised Image Denoising without Clean Images’, Advances in Neural Information Processing Systems 34.

Kingma, D. P. & Welling, M. (2013), ‘Auto-encoding variational bayes’, arXiv preprint arXiv:1312.6114 .

Laroche, C., Almansa, A. & Coupete, E. (2024), Fast diffusion em: a diffusion model for blind inverse problems with application to deconvolution, in ‘Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision’, pp. 5271–5281.

Lee, S., Chung, H., Park, M., Park, J., Ryu, W.-S. & Ye, J. C. (2023), Improving 3d imaging with pre-trained perpendicular 2d diffusion models, in ‘Proceedings of the IEEE/CVF International Conference on Computer Vision’, pp. 10710–10720.

Lee, S., Park, D., Kong, I. & Kim, H. J. (2024), Diffusion prior-based amortized variational inference for noisy inverse problems, in ‘European Conference on Computer Vision’, Springer, pp. 288–304.

Li, X., Kwon, S. M., Liang, S., Alkhouri, I. R., Ravishankar, S. & Qu, Q. (2024), ‘Decoupled data consistency with diffusion purification for image restoration’, arXiv preprint arXiv:2403.06054 .

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. & Le, M. (2023), Flow matching for generative modeling, in ‘The Eleventh International Conference on Learning Representations’. URL: https://openreview.net/forum?id=PqvMRDCJT9t

Liu, G.-H., Vahdat, A., Huang, D.-A., Theodorou, E. A., Nie, W. & Anandkumar, A. (2023), ‘I<sup>2</sup>sb: Image-to-image schrödinger bridge’, arXiv preprint arXiv:2302.05872 .

Liu, X., Gong, C. & qiang liu (2023), Flow straight and fast: Learning to generate and transfer data with rectified flow, in ‘The Eleventh International Conference on Learning Representations’. URL: https://openreview.net/forum?id=XVjTT1nw5z

Luo, Z., Gustafsson, F. K., Zhao, Z., Sjölund, J. & Schön, T. B. (2023), ‘Image restoration with mean-reverting stochastic differential equations’, arXiv preprint arXiv:2301.11699 .

Mammadov, A., Chung, H. & Ye, J. C. (2024), ‘Amortized posterior sampling with diffusion prior distillation’, arXiv preprint arXiv:2407.17907 .

Man, S., Ohayon, G., Raphaeli, R. & Elad, M. (2025), ‘Proxies for distortion and consistency with applications for real-world image restoration’, arXiv preprint arXiv:2501.12102 .

Mardani, M., Song, J., Kautz, J. & Vahdat, A. (2023), ‘A variational perspective on solving inverse problems with diffusion models’, arXiv preprint arXiv:2305.04391 .

Murata, N., Saito, K., Lai, C.-H., Takida, Y., Uesaka, T., Mitsufuji, Y. & Ermon, S. (2023), Gibbsddrm: A partially collapsed gibbs sampler for solving blind inverse problems with denoising diffusion restoration, in ‘International conference on machine learning’, PMLR, pp. 25501–25522.

Park, B., Go, H., Nam, H., Kim, B.-H., Chung, H. & Kim, C. (2025), ‘Steerx: Creating any camera-free 3d and 4d scenes with geometric steering’, arXiv preprint arXiv:2503.12024 .

Patel, M., Wen, S., Metaxas, D. N. & Yang, Y. (2024), ‘Steering rectified flow models in the vector field for controlled image generation’, arXiv preprint arXiv:2412.00100 .

Peng, X., Zheng, Z., Dai, W., Xiao, N., Li, C., Zou, J. & Xiong, H. (2024), ‘Improving diffusion models for inverse problems using optimal posterior covariance’, arXiv preprint arXiv:2402.02149 .

Raphaeli, R., Man, S. & Elad, M. (2025), ‘Silo: Solving inverse problems with latent operators’, arXiv preprint arXiv:2501.11746 .

Romano, Y., Elad, M. & Milanfar, P. (2017), ‘The little engine that could: Regularization by denoising (red)’, SIAM journal on imaging sciences 10(4), 1804–1844.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P. & Ommer, B. (2022), High-resolution image synthesis with latent diffusion models, in ‘Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition’, pp. 10684–10695.

Rout, L., Chen, Y., Kumar, A., Caramanis, C., Shakkottai, S. & Chu, W.-S. (2024), Beyond first-order tweedie: Solving inverse problems using latent diffusion, in ‘Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition’, pp. 9472–9481.

Rout, L., Raoof, N., Daras, G., Caramanis, C., Dimakis, A. & Shakkottai, S. (2023), ‘Solving linear inverse problems provably via posterior sampling with latent diffusion models’, Advances in Neural Information Processing Systems 36, 49960–49990.

Rout, L., Raoof, N., Daras, G., Caramanis, C., Dimakis, A. & Shakkottai, S. (2024), ‘Solving linear inverse problems provably via posterior sampling with latent diffusion models’, Advances in Neural Information Processing Systems 36.

Rozet, F., Andry, G., Lanusse, F. & Louppe, G. (2024), ‘Learning diffusion priors from observations by expectation maximization’, Advances in Neural Information Processing Systems 37, 87647–87682.

Singhal, R., Horvitz, Z., Teehan, R., Ren, M., Yu, Z., McKeown, K. & Ranganath, R. (2025), ‘A general framework for inference-time scaling and steering of diffusion models’, arXiv preprint arXiv:2501.06848 .

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. & Ganguli, S. (2015), Deep unsupervised learning using nonequilibrium thermodynamics, in ‘International Conference on Machine Learning’, PMLR, pp. 2256–2265.

Song, J., Vahdat, A., Mardani, M. & Kautz, J. (2023), Pseudoinverse-guided diffusion models for inverse problems, in ‘International Conference on Learning Representations’. URL: https://openreview.net/forum?id=9\_gsMA8MRKQ

Song, Y., Dhariwal, P., Chen, M. & Sutskever, I. (2023), Consistency models, in ‘Proceedings of the 40th International Conference on Machine Learning’, pp. 32211–32252.

Song, Y., Durkan, C., Murray, I. & Ermon, S. (2021), ‘Maximum likelihood training of score-based diffusion models’, Advances in Neural Information Processing Systems 34.

Song, Y. & Ermon, S. (2019), Generative modeling by estimating gradients of the data distribution, in ‘Advances in Neural Information Processing Systems’, Vol. 32.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S. & Poole, B. (2021), Score-based generative modeling through stochastic differential equations, in ‘9th International Conference on Learning Representations, ICLR’.

Stein, C. M. (1981), ‘Estimation of the mean of a multivariate normal distribution’, The annals of Statistics pp. 1135– 1151.

Tarantola, A. (2005), Inverse problem theory and methods for model parameter estimation, SIAM.

Trippe, B. L., Yim, J., Tischer, D., Baker, D., Broderick, T., Barzilay, R. & Jaakkola, T. S. (2023), Diffusion probabilistic modeling of protein backbones in 3d for the motif-scaffolding problem, in ‘The Eleventh International Conference on Learning Representations’. URL: https://openreview.net/forum?id=6TxBxqNME1Y

Ulyanov, D., Vedaldi, A. & Lempitsky, V. (2018), Deep image prior, in ‘Proceedings of the IEEE conference on computer vision and pattern recognition’, pp. 9446–9454.

Vahdat, A., Kreis, K. & Kautz, J. (2021), ‘Score-based generative modeling in latent space’, Advances in neural information processing systems 34, 11287–11302.

Venkatakrishnan, S. V., Bouman, C. A. & Willett, R. M. (2013), Plug-and-play priors for model based reconstruction, in ‘2013 IEEE Global Conference on Signal and Information Processing (GlobalSIP)’, IEEE, pp. 945–948.

Vincent, P. (2011), ‘A connection between score matching and denoising autoencoders’, Neural computation 23(7), 1661– 1674.

Wu, H., He, L., Zhang, M., Chen, D., Luo, K., Luo, M., Zhou, J.-Z., Chen, H. & Lv, J. (2024), Diffusion posterior proximal sampling for image restoration, in ‘Proceedings of the 32nd ACM International Conference on Multimedia’, pp. 214–223.

Xu, T., Cai, X., Zhang, X., Ge, X., He, D., Sun, M., Liu, J., Zhang, Y.-Q., Li, J. & Wang, Y. (2025), ‘Rethinking diffusion posterior sampling: From conditional score estimator to maximizing a posterior’, arXiv preprint arXiv:2501.18913 .

Yang, L., Ding, S., Cai, Y., Yu, J., Wang, J. & Shi, Y. (2024), ‘Guidance with spherical gaussian constraint for conditional diffusion’, arXiv preprint arXiv:2402.03201 .

Zhang, B., Chu, W., Berner, J., Meng, C., Anandkumar, A. & Song, Y. (2025), Improving diffusion inverse problem solving with decoupled noise annealing, in ‘Proceedings of the Computer Vision and Pattern Recognition Conference’, pp. 20895–20905.

Zhang, K., Zuo, W., Chen, Y., Meng, D. & Zhang, L. (2017), ‘Beyond a gaussian denoiser: Residual learning of deep CNN for image denoising’, IEEE transactions on image processing 26(7), 3142–3155.

Zilberstein, N., Mardani, M. & Segarra, S. (2025), Repulsive latent score distillation for solving inverse problems, in ‘The Thirteenth International Conference on Learning Representations’. URL: https://openreview.net/forum?id=bwJxUB0y46