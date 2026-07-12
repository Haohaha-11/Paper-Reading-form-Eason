# Fast Diffusion EM: a diffusion model for blind inverse problems with application to deconvolution

Charles Laroche GoPro & MAP5 charles.laroche@u-paris.fr

Andres Almansa´ CNRS & Universite Paris Cit´ e´ andres.almansa@parisdescartes.fr

Eva Coupete   
GoPro   
ecoupete@gopro.com

## Abstract

Using diffusion models to solve inverse problems is a growing field of research. Current methods assume the degradation to be known and provide impressive results in terms of restoration quality and diversity. In this work, we leverage the efficiency of those models to jointly estimate the restored image and unknown parameters of the degradation model such as blur kernel. In particular, we designed an algorithm based on the well-known Expectation-Minimization (EM) estimation method and diffusion models. Our method alternates between approximating the expected log-likelihood of the inverse problem using samples drawn from a diffusion model and a maximization step to estimate unknown model parameters. For the maximization step, we also introduce a novel blur kernel regularization based on a Plug & Play denoiser. Diffusion models are long to run, thus we provide a fast version of our algorithm. Extensive experiments on blind image deblurring demonstrate the effectiveness of our method when compared to other state-of-the-art approaches. Our code is available at https://github.com/claroche-r/FastDiffusionEM.

## 1. Introduction

Image restoration aims to recover information that has been obscured by various degradations such as blur, noise, or compression artifacts. Deep-learning-based methods have revolutionized the field of image restoration by achieving impressive results in various tasks. They leverage the power of deep neural network architectures to learn a mapping between training data [11, 55, 57]. This data-driven approach allows deep-learning models to capture intricate patterns and relationships within the image data, enabling them to restore images with superior quality and perceptual fidelity [28, 51]. On the other hand, model-based approaches express the image restoration problem as an inverse problem and exploit the degradation process structure to design regularizations and optimization algorithms to find the optimal reconstruction [37]. They usually offer more control, flexibility, and interpretability.

However, model-based approaches highly rely on the knowledge of the degradation forward process limiting their usefulness in practical applications. Some strategies try to bring the best of both worlds such as Plug-and-Play methods or deep unfolding networks [22, 23, 26, 41, 53]. One of the challenges behind inverse problems comes from their ill-posedness. In fact, for a single degraded image, there generally exist multiple plausible solutions. A common approach is to generate a single restored image that minimizes the mean squared error, but it does not allow the models to generate or hallucinate high-quality details [42, 50]. There is a growing interest in the field of image restoration to design models that can generate all the space of plausible solutions. Those models include Generative Adversarial Networks [16, 33] , conditional or PnP Diffusion Models [25, 42, 43] or Langevin dynamics [27]. This growing interest in diverse restoration is motivated by the impressive perceptual quality obtained by such methods. In particular, diffusion models that were first introduced for image synthesis tasks [20, 21, 44] are now used for a large diversity of tasks such as inverse problem solving [7, 25, 45]. In the field of blind deconvolution, it is common to use Bayesian methods to jointly estimate the blur kernel and the restored image [3, 31, 37, 38]. The kernel estimation highly relies on the restoration method that is used and it generally requires the restoration method to produce a sharp image. To do so, image regularizations such as TV, $\ell _ { 0 }$ on the gradient can be used but they tend to over-sharpen the restored image leading to unpleasant results. Even with the sharp and blurry pairs, it is not easy to estimate any type of blur kernels without efficient regularization. Common regularization on the kernels are the $\ell _ { 1 }$ norm [6], positivity, the sum to one constraint, and in some cases Gaussian constraints [4]. Some recent works also use deep neural networks such as normalizing flows to parameterize the kernels [29]. Motivated by the impressive quality of diffusion models for both estimated conditional distribution and returning high-quality images, it is natural to believe that they could be used in the context of kernel estimation. Also, a pioneer work [6] that combines parallel diffusion models for the kernel and image exhibits impressive results. Estimating the kernel and image is jointly done in the diffusion process using gradient descent on the forward model. Similarly, methods based on Monte Carlo sampling proposed parameters estimation derived from the Expectation-Maximization (EM) algorithm [14, 18], or the SAPG algorithm [13, 47]. Those methods are very efficient but Monte Carlo sampling is time-consuming. Also, the problem of kernel estimation is a complex problem so those methods highly depend on the regularization imposed in the M-step of the EM algorithm.

![](images/9ad59baa133274488286f795959a150e4d9ee40309730d4b52a5527b6169a2a2.jpg)  
Figure 1. Performance comparison of the different models using the PSNR metric depending on the runtime, “Ours” corresponds to Fast EM ΠGDM method.

![](images/9b899e8767fa8987d1e8374e5140e2de0519d158aca34a4ddfc531577aeb4008.jpg)  
Figure 2. Overview of the method and evolution of the current estimates. We start with random noise and apply the diffusion process. The blurry image intervenes both for the guidance and for the M-step which estimates the blur kernel.

Motivated by the efficiency of diffusion models, we propose a diffusion model that solves the maximum a-posteriori estimator for blind deconvolution. Derived from the classical Expectation-Maximization algorithm, our model alternately estimates the expected value of the log-likelihood using samples drawn from a diffusion model and maximizes this quantity using half-quadratic splitting. In addition, we also propose a novel kernel regularization in a Plug & Play fashion. Finally, we proposed a fast version of our algorithm to facilitate the use of our method in real-world scenarios. Our experiments show that our proposed solution improves both in terms of fidelity and computational efficiency pushing the Pareto optimal curve further to the origin (Figure 1).

## 2. Background

Let us suppose that our deblurring problem fits the classical inverse problem formulation:

$$
y = H x + n \quad { \mathrm { w i t h } } \quad n \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } )\tag{1}
$$

where x is the clean image we want to estimate, y is the blurry and noisy image and H is the degradation operator, a convolution operator in the case of deconvolution. We suppose that we are in the real-world case where we only have access to the blurry image y and the noise level σ to reconstruct both the clean image and the blur kernel H. In such a setting, a common approach to estimate the blur kernel is to compute the marginalized maximum a-posteriori (MAP) estimator of the inverse problem described in Equation (1):

$$
\begin{array} { l } { { \displaystyle H _ { M A P } = \arg \operatorname* { m a x } _ { H } p ( H | y ) = \arg \operatorname* { m a x } _ { H } p ( y | H ) p ( H ) } } \\ { { \displaystyle ~ = \arg \operatorname* { m a x } _ { H } \left[ \log \left( \int p ( y | H , x ) p ( x ) d x \right) + \log ( p ( H ) ) \right] , } } \end{array}
$$

with $p ( x )$ a natural image prior, $p ( \boldsymbol { y } | \boldsymbol { H } , \boldsymbol { x } )$ the likelihood of the blurry image and $p ( H )$ the kernel’s prior distribution. This MAP estimator cannot be solved easily since the marginalization in the clean image x is not tractable. Expectation-Maximization (EM) [10, 32] is an iterative algorithm that computes the MAP estimator for the parameters of a statistical model (H in our case). It is very convenient when the model contains unobserved or missing data. The EM algorithm consists of two main steps. An E-step that computes the expected log-likelihood given the current model parameter estimates and an M-step, that maximizes this expected log-likelihood to update the estimated parameters. The whole algorithm alternates between the E-step and M-step until convergence. In the case of deblurring, the parameter we want to estimate is the blur kernel H and our unobserved data are the clean images associated with the blurry image y and the estimated blur kernel H. The EM algorithm can be summarized as follows in such setting: E-Step:

$$
Q ( H , H _ { l } ) = E _ { x \sim p ( x | y , H _ { l } ) } [ \log ( p ( y | x , H ) ) + \log ( p ( x ) ) ]\tag{3}
$$

M-Step:

$$
H _ { l + 1 } = \arg \operatorname* { m a x } _ { H } \left[ Q ( H , H _ { l } ) + \log ( p ( H ) ) \right]\tag{4}
$$

This formulation is very convenient but in many applications (including blind deblurring), the expected log-likelihood in Equation (3) cannot be computed explicitly, and even taking posterior samples $x \ \sim \ p ( x | y , H _ { l } )$ is challenging. Our method proposes to approximate the expectation in the E-step by an empirical mean in Monte-Carlo EM fashion [49] and to use a diffusion model to obtain posterior samples.

Diffusion models for posterior sampling: To learn $p ( x _ { 0 } )$ the distribution of the data, diffusion models define a family of distributions $p ( x _ { t } )$ by gradually adding Gaussian noise of variance $\beta ( t )$ to samples of $p ( x _ { 0 } )$ until the distribution $p ( x _ { T } )$ reduces to a standard Gaussian with zero mean. For discrete timesteps $t \in \mathbb { [ 0 , } T ]$ , we can define a Markov transition kernel $p ( x _ { t } | x _ { t - 1 } ) = \mathcal { N } ( x _ { t } ; \sqrt { 1 - \beta ( t ) } x _ { t - 1 } , \beta ( t ) I )$ between two consecutive discrete timestamps. In the general continuous case, [46] described the forward noising process with the following stochastic differentiable equation (SDE) :

$$
d x _ { t } = - \frac { \beta ( t ) } { 2 } x _ { t } d t + \sqrt { \beta ( t ) } d w\tag{5}
$$

where $w ( t )$ is the d-dimensional Wiener process. The reverse SDE of this process [2] can be written as:

$$
d x _ { t } = [ - \frac { \beta ( t ) } { 2 } x _ { t } - \beta ( t ) \nabla _ { x _ { t } } \log \pi ( x _ { t } ) ] d t + \sqrt { \beta ( t ) } d \bar { w }\tag{6}
$$

with dt corresponding to time running backwards and dw¯ to the standard Wiener process running backwards. In the case of inverse problems, we want to use diffusion models to generate the posterior distribution $\pi ( x _ { t } ) = p ( x _ { t } | y , H )$ Using Bayes’ rule Equation (6) becomes:

$$
\begin{array} { r l } & { d x _ { t } = \left[ - \displaystyle \frac { \beta ( t ) } { 2 } x _ { t } - \beta ( t ) \left( \nabla _ { x _ { t } } \log p ( x _ { t } ) \right. \right. } \\ & { ~ \quad \left. \left. + \nabla _ { x _ { t } } \log p ( y | x _ { t } , H ) \right) \right] d t + \sqrt { \beta ( t ) } d \bar { w } } \end{array}\tag{7}
$$

The main problem behind this equation is that in inverse problems, we have a relation between $y$ and $x _ { 0 }$ but not between $x _ { t }$ and $y .$ . Marginalizing in $x _ { 0 } .$ , we obtain:

$$
p ( y | x _ { t } ) = \int p ( y | x _ { 0 } ) p ( x _ { 0 } | x _ { t } ) d x _ { 0 }\tag{8}
$$

that is intractable. The main challenge of non-blind diffusion for posterior sampling is to compute or approximate this integral. In our work, we conduct experiments with DPS [7] and ΠGDM [45] that use different approximations for this integral. Both approximations are based on the mean of $p ( x _ { 0 } | x _ { t } )$ , namely:

$$
\widehat { x } _ { 0 } ( t ) : = E [ x _ { 0 } | x _ { t } ] .
$$

DPS approximates $p ( x _ { 0 } | x _ { t } )$ by a delta function

$$
p ( x _ { 0 } | x _ { t } ) \approx \delta _ { \widehat { x } _ { 0 } ( t ) } ( x _ { 0 } )\tag{9}
$$

whereas ΠGDM approximates $p ( x _ { 0 } | x _ { t } )$ by a Gaussian distribution

$$
p ( x _ { 0 } | x _ { t } ) \approx \mathcal { N } ( x _ { 0 } | \widehat { x } _ { 0 } ( t ) , r _ { t } ^ { 2 } )\tag{10}
$$

with $r _ { t }$ a hyper-parameter. Both approximations allow us to solve the marginal in Equation (8) analytically and obtain explicit expressions for $\nabla _ { x _ { t } } \log { p ( y | x _ { t } ) }$ as detailed below.

As a recall, one property of diffusion models is that we can express the noisy measurement $x _ { t }$ in the forward model using the original sample x<sub>0</sub>:

$$
x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon\tag{11}
$$

with $\alpha _ { t } = 1 - \beta _ { t }$ and ${ \bar { \alpha } } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i } .$

Using a noise predictor $\boldsymbol { \epsilon } ( x _ { t } , t )$ , we can thus estimate $\widehat { x } _ { 0 } ( t ) = E [ x _ { 0 } | x _ { t } ]$ at each step t using:

$$
\widehat { x } _ { 0 } ( t ) = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } ( x _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon ( x _ { t } , t ) ) .\tag{12}
$$

Equivalently, we can use a score network $s ( x _ { t } , t )$ using Tweedie’s identity:

$$
s ( x _ { t } , t ) = \nabla _ { x _ { t } } \log p ( x _ { t } ) = - \frac { 1 } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon ( x _ { t } , t ) .\tag{13}
$$

Using DDPM [20] to discretize the unconditional reverse diffusion process (6) we obtain the update rule

$$
x _ { t - 1 } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( x _ { t } + \beta _ { t } s ( x _ { t } , t ) \right) + \tilde { \sigma } _ { t } \mathcal { N } ( 0 , I )\tag{14}
$$

where $\begin{array} { r } { \tilde { \sigma } _ { t } = \sqrt { \beta _ { t } } \mathrm { o r } \sqrt { \frac { ( 1 - \bar { \alpha } _ { t - 1 } ) } { 1 - \bar { \alpha } _ { t } } \beta _ { t } } } \end{array}$ . To simulate the conditional reverse diffusion process (7), we just have to add the likelihood term to the score

$$
\begin{array} { c } { \displaystyle \boldsymbol { x } _ { t - 1 } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( \boldsymbol { x } _ { t } + \beta _ { t } \left[ s ( \boldsymbol { x } _ { t } , t ) + \nabla _ { \boldsymbol { x } _ { t } } \log p ( \boldsymbol { y } | \boldsymbol { x } _ { t } ) \right] \right) } \\ { + \tilde { \sigma } _ { t } \mathcal { N } ( 0 , I ) } \end{array}\tag{15}
$$

Using Equation (12), the DPS [7] approximation for $p ( x _ { 0 } | x _ { t } )$ leads to the following formula for the gradient of the log-likelihood:

$$
\nabla _ { x _ { t } } \log { p ( y | x _ { t } ) } = - \frac { 1 } { \sigma ^ { 2 } } \nabla _ { x _ { t } } \| y - H \widehat { x } _ { 0 } ( t ) \| _ { 2 } ^ { 2 }\tag{16}
$$

Similarly, the ΠGDM [45] approximation leads to the following gradient for the log-likelihood:

$$
\begin{array} { l } { \displaystyle \nabla _ { x _ { t } } \log p ( \boldsymbol { y } | \boldsymbol { x _ { t } } ) = \qquad ( 1 7 } \\ { \displaystyle \left( ( \boldsymbol { y } - H \widehat { \boldsymbol { x _ { 0 } } } ( t ) ) ^ { T } ( { r _ { t } ^ { 2 } } H H ^ { T } + \sigma ^ { 2 } I ) ^ { - 1 } H \left( \frac { \partial \widehat { \boldsymbol { x _ { 0 } } } ( t ) } { \partial \boldsymbol { x _ { t } } } \right) \right) ^ { T } } \end{array}
$$

DPS and ΠGDM derive different guidance terms for the inverse problem. While the DPS approximation leads to a gradient that is easily implemented for any degradation operator H using automatic differentiation, the ΠGDM approximated gradient of Equation (17) is much more complex to estimate for a general operator H because it requires the computation of its pseudo-inverse. On the other hand, the ΠGDM approximation is more precise and thus leads to stronger guidance which is very important for kernel estimation. We summarize in Algorithm 1 the diffusion process for inverse problems when the degradation operator H is known. This case covers both DPS and ΠGDM. The pseudo-code is written using DDPM but is not limited to this particular diffusion scheme. To compensate for the fact that the first estimations of $x _ { t }$ are uncertain, it is common to set $\zeta _ { t } = \sqrt { \bar { \alpha } _ { t } }$ , instead of the theoretical $\zeta _ { t } = 1$

## 3. Method

Our method proposes to solve the MAP of the blur kernel from a blurry and potentially noisy image. We estimate the MAP estimator in an EM fashion. Iteratively, we first draw samples from the posterior distribution knowing the current kernel estimate using a diffusion model. It corresponds to the E-step of the EM algorithm. Then, we update our estimated kernel with the M-step by maximizing the expected log-likelihood on the previously computed samples. To efficiently model the kernels’ distribution, we use a Plug & Play kernel denoiser to regularize our MAP estimator.

## 3.1. E-step: Non-blind diffusion

The E-step of the EM algorithm consists in evaluating the expectation from Equation (3). Instead of computing its exact value, we propose to approximate it using random samples in a Monte-Carlo EM fashion. To draw the random samples, we use a non-blind diffusion model. Since the diffusion model targets $p ( x | y , H _ { l } )$ , sampling several images leads to a good approximation of the expectation. The number n of samples used to approximate the expectation is a hyperparameter of the method. Having many samples leads to a slow but accurate estimation while having only one sample is equivalent to the Stochastic EM algorithm [36]. In practice, the E-step reduces to:

Drawing samples

$$
\pmb { x } = ( x ^ { 1 } , . . . , x ^ { n } ) \sim p ( x _ { 0 } | y , H _ { l } )\tag{18}
$$

and updating

$$
{ \widehat { Q } } ( H , H _ { l } ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \log ( p ( y | x ^ { i } , H ) ) .\tag{19}
$$

The samples can be drawn by n parallel runs of Algorithm 1, and the empirical mean $\widehat { Q } ( H , H _ { l } ) \approx Q ( H , H _ { l } )$ approaches the expected value in Equation (3) as n → ∞. Unlike in Equation (3), we remove the term in $p ( x )$ from $\widehat { Q } ( H , H _ { l } )$ here since it does not affect the maximization in the blur kernel H.

## 3.2. M-step: Kernel estimation

The M-step computes the MAP estimator of the blur kernel using the estimated samples from the E-step as measurements. From equations (1), (4) and (19) this step can be summarized as:

$$
H _ { l + 1 } = \arg \operatorname* { m a x } _ { H } \hat { Q } ( H , H _ { l } ) + \log ( p ( H ) )\tag{20}
$$

$$
H _ { l + 1 } = \arg \operatorname* { m i n } _ { H } \frac { 1 } { 2 n \sigma ^ { 2 } } \sum _ { i = 1 } ^ { n } \| y - H x ^ { i } \| _ { 2 } ^ { 2 } + \lambda \Phi ( H )\tag{21}
$$

where (21) is obtained using Equation (1) and (19). Common choices for $\Phi ( . )$ are $\ell _ { 2 } \mathrm { o r } \ell _ { 1 }$ regularizations on top of the simplex constraints on the blur kernel (non-negative values that add up to one). Despite being quite efficient when the blurry image does not have noise, they generally fail to provide good quality results when the noise increases. On the other side, Plug & Play regularizations have become more and more popular for many image restoration tasks. By training a deep denoiser on Gaussian denoising, one can obtain a powerful regularization in the domain on which the denoiser was trained. Generally, we train the denoiser on a dataset of natural images leading to a regularization on natural images. Here, we propose to train a denoiser on a dataset of blur kernels to build a Plug & Play regularization for the blur kernels. We observed that this approach leads to a kernel estimation algorithm that is more efficient and robust to noise, see Figure 4. To solve Equation (21), we use the Half-Quadratic Splitting (HQS) optimization scheme:

$$
\begin{array} { r } { Z _ { j + 1 } = \arg \underset { Z } { \operatorname* { m i n } } \frac { 1 } { 2 \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } \| Z x ^ { i } - y \| _ { 2 } ^ { 2 } } \\ { + \frac { \beta } { 2 } \| Z - K _ { j } \| _ { 2 } ^ { 2 } } \end{array}\tag{22}
$$

$$
K _ { j + 1 } = \arg \operatorname* { m i n } _ { K } \lambda \Phi ( K ) + \frac { \beta } { 2 } \| K - Z _ { j + 1 } \| _ { 2 } ^ { 2 }\tag{23}
$$

```latex
Algorithm 1 Diffusion model for deblurring
Require: $y , \sigma , H , T , \left( \zeta _ { t } \right) _ { t }$
Ensure: A posterior sample x<sub>0</sub> $\sim p ( x _ { 0 } | y , H )$
x<sub>T</sub> $ \mathcal { N } ( 0 , I )$
for $t = T$ to 1 do
$\widehat { \epsilon } \gets \epsilon ( x _ { t } , t )$
$\begin{array} { r } { \widehat { x } _ { 0 } = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } ( x _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \widehat { \epsilon } ) } \end{array}$
// DPS or ΠGDM approx. using $\hat { x } _ { 0 }$
$g \gets \nabla _ { x _ { t } } \log p ( y | x _ { t } , H )$ ▷ Equation (16) or (17)
// Compute conditional score $s = \nabla _ { x _ { t } } \log p ( x _ { t } | y , H )$
$\begin{array} { r } { s \gets \zeta _ { t } g - \frac { 1 } { \sqrt { 1 - \bar { \alpha } _ { t } } } \hat { \epsilon } } \end{array}$ ▷ Bayes rule and Tweedie
// DDPM update rule
$z \gets \mathcal { N } ( 0 , I )$
$\begin{array} { r } { x _ { t - 1 } \gets \frac { 1 } { \sqrt { \alpha _ { t } } } \left( x _ { t } + \beta _ { t } s \right) + \tilde { \sigma } _ { t } z } \end{array}$
end for
return $x _ { 0 }$
```

For the deconvolution problem, Equation (22) can easily be solved in the Fourier domain (more details on the computations can be found in Appendix B). Equation (23) corresponds to the regularization step. It corresponds to the MAP estimator of a Gaussian denoising problem on the variable $Z _ { j + 1 }$ . The main idea behind Plug & Play regularization is to replace this regularization step with a pre-trained denoiser D Mean Squared Error (MSE) loss. This substitution can be done thanks to the close relationship that exists between the MAP and the MMSE estimator of a Gaussian denoising problem [17]. Eventually, the M-step consists of the following iterations:

$$
Z _ { j + 1 } = \mathcal { F } ^ { - 1 } \left( \frac { \mathcal { F } ( y ) \sum _ { i = 1 } ^ { n } \overline { { \mathcal { F } ( x ^ { i } ) } } + n \beta \sigma ^ { 2 } \mathcal { F } ( K _ { j } ) } { \sum _ { i = 1 } ^ { n } \mathcal { F } ( x ^ { i } ) \overline { { \mathcal { F } ( x ^ { i } ) } } + n \beta \sigma ^ { 2 } } \right)\tag{24}
$$

$$
K _ { j + 1 } = \mathcal { D } _ { \sqrt { \lambda / \beta } } ( Z _ { j + 1 } ) .\tag{25}
$$

While complex decreasing schemes for $\beta$ are often used to help HQS converge [54], we observed that using a constant $\beta$ was sufficient in our case. For the denoiser architecture, we use a simple DnCNN [55] with 5 blocks and 32 channels. In addition to the noisy kernel, we also give the noise level as an extra channel to the network to control the denoising intensity. Eventually, the complete Diffusion EM algorithm alternates between sampling from the non-blind diffusion model and the HQS algorithm for the kernel estimation. In all our experiments, we use $L = 1 0 \mathrm { E M }$ iterations. See Algorithm A.1 in the supplementary.

## 3.3. Fast EM diffusion

The diffusion EM algorithm requires running a diffusion model at each step of the EM algorithm to produce a set of n particles. Executing diffusion models is time-consuming, particularly in cases where inverse problems are addressed using score guidance, as the guidance must be applied to the full-size image, precluding the utilization of acceleration techniques like latent diffusion [39]. Consequently, the diffusion EM algorithm’s execution time becomes excessively long, significantly restricting its practical applicability.

Algorithm 2 Fast EM DPS / ΠGDM   
Require: $y , \sigma , H _ { T } , T$   
Ensure: H ≈ arg min $_ { \textit { t } p } ( y | H )$ and $x _ { 0 } ^ { i } \sim p ( x _ { 0 } | y , H )$   
x<sub>T</sub> $ ( \mathcal { N } ( 0 , I ) , . . . , \mathcal { N } ( 0 , I ) ) \in ( \mathbb { R } ^ { h * w * 3 } ) ^ { n }$   
for $t = T \mathop { \bf t o } \mathrm { ~ 1 ~ } \mathrm { d } { \bf 0 }$   
${ \widehat { \pmb { \epsilon } } } \gets \epsilon ( { \pmb { x } } _ { t } , t )$   
$\begin{array} { r } { \widehat { \pmb x } _ { 0 } = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } ( \pmb x _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \widehat \epsilon ) } \end{array}$   
$H _ { t - 1 } \stackrel { \cdot } { = } M \substack { - s t e p } ( y , \widehat { \pmb { x } } _ { \mathbf { 0 } } , \sigma )$ ▷ Iterate (24) and (25)   
// DPS or $\Pi \mathrm { G D M }$ approx. using $\scriptstyle { \hat { x } } _ { 0 }$   
$\pmb { g } \gets \nabla _ { \pmb { x } _ { t } } \log p ( y | \pmb { x } _ { t } , H _ { t - 1 } )$ ▷ Equation (16) or (17)   
// Compute conditional score $s = \nabla _ { x _ { t } } \log p ( x _ { t } | y , H )$   
$\begin{array} { r } { \pmb { s }  \zeta _ { t } \pmb { g } - \frac { 1 } { \sqrt { 1 - \bar { \alpha } _ { t } } } \hat { \pmb { \epsilon } } } \end{array}$ ▷ Bayes rule and Tweedie   
// DDPM update rule   
$z \gets ( \mathcal { N } ( \hat { \mathrm { 0 } } , I ) , . . . , \mathcal { N } ( 0 , I ) ) \in ( \mathbb { R } ^ { h * w * 3 } ) ^ { n }$   
$\begin{array} { r } { \pmb { x } _ { t - 1 }  \frac { 1 } { \sqrt { \alpha _ { t } } } ( \pmb { x } _ { t } + \beta _ { t } \pmb { s } ) + \tilde { \sigma } _ { t } \pmb { z } } \end{array}$   
end for   
return $x _ { 0 } , H _ { 0 }$

To bypass this problem, we propose a fast version of diffusion EM that incorporates the M-step directly into the diffusion process, thereby reducing the number of required diffusion model runs to just one. To do so, we use the n current samples $x _ { t } ^ { i } \sim p ( x _ { t } | y , H )$ to build an approximation of $Q ( H , H _ { t } )$ at each timestep t, as follows. First, we use the current distribution estimates $p ( x _ { 0 } | x _ { t } )$ (Equations (9) and (10) for DPS, resp. ΠGDM approximations) for each timestep t to approximate the posterior $p ( x _ { 0 } | y , H )$ by (discretized) marginalization on x<sub>t</sub>:

$$
p ( x _ { 0 } | H , y ) = \int p ( x _ { 0 } | x _ { t } ) p ( x _ { t } | y , H ) d x _ { t }\tag{26}
$$

$$
\approx \sum _ { i = 1 } ^ { n } p ( x _ { 0 } | x _ { t } ^ { i } ) p ( x _ { t } ^ { i } | y , H )\tag{27}
$$

$$
= \frac { 1 } { n } \sum _ { i = 1 } ^ { n } p ( x _ { 0 } | x _ { t } ^ { i } ) = : q _ { t } ( x _ { 0 } | y , H ) .\tag{28}
$$

Then, using this approximation, the E-step at timestep t of the diffusion process is reformulated as follows:

$$
Q ( H , H _ { t } ) = E _ { x \sim p ( x | y , H _ { t } ) } [ \log ( p ( y | x , H ) ]\tag{29}
$$

$$
\approx E _ { x \sim q _ { t } ( x _ { 0 } | y , H _ { t } ) } [ \log ( p ( y | x , H ) ]\tag{30}
$$

Since the distribution $q _ { t } ( x _ { 0 } | y , H )$ progressively converges to the distribution $p ( x _ { 0 } | y , H )$ as $t  0 ,$ , we have a finer and finer estimation of the expected log-likelihood and thus, the blur kernel, through the iterations.

Finally, the E-step reduces in the case of the DPS approxi-

<table><tr><td>Metric type</td><td></td><td colspan="4">Reference metrics</td><td colspan="2">No-reference metrics</td><td colspan="2">Kernel error</td></tr><tr><td>↓ Method \ Metric →</td><td>Time (sec/img)</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td> $\overline { { \mathrm { F I D } \downarrow } }$ </td><td>NIQE↓</td><td>BRISQUE↓</td><td>MSE kernel ↓</td><td> $\mathcal { L } _ { r e b l u r } \downarrow$ </td></tr><tr><td>DPS*</td><td>58sec</td><td>25.81</td><td>0.76</td><td>0.34</td><td>3.46</td><td>6.28</td><td>23.52</td><td>x</td><td>x</td></tr><tr><td>IIGDM*</td><td>5sec</td><td>27.65</td><td>0.81</td><td>0.34</td><td>4.50</td><td>7.49</td><td>30.32</td><td>x</td><td>x</td></tr><tr><td>Anger l0</td><td>0.73sec</td><td>12.46</td><td>0.13</td><td>0.8</td><td>233.08</td><td>12.55</td><td>50.51</td><td>5.1e-5</td><td>1.1e-2</td></tr><tr><td>Self-Deblur</td><td>1min53sec</td><td>14.53</td><td>0.15</td><td>0.69</td><td>44.83</td><td>14.16</td><td>49.28</td><td>3.6e-4</td><td>3.5e-2</td></tr><tr><td>MPRNet</td><td>3.7sec</td><td>19.52</td><td>0.42</td><td>0.54</td><td>21.26</td><td>7.9</td><td>25.44</td><td>x</td><td>x</td></tr><tr><td>Blind DPS</td><td>1min23</td><td>24.05</td><td>0.73</td><td>0.34</td><td>2.66</td><td>6.17</td><td>20.72</td><td>3.9e-5</td><td>5.6e-3</td></tr><tr><td>EM IIGDM (n=1)</td><td>1min30sec</td><td>23.4</td><td>0.71</td><td>0.43</td><td>6.05</td><td>8.81</td><td>41.19</td><td>6.1e-5</td><td>5.3e-3</td></tr><tr><td>EM IIGDM (n=4)</td><td>2min30sec</td><td>23.21</td><td>0.71</td><td>0.4</td><td>5.43</td><td>8.23</td><td>38.02</td><td>5e-5</td><td>5.3e-3</td></tr><tr><td>EM IIGDM (n=16)</td><td>9min10sec</td><td>23.09</td><td>0.71</td><td>0.39</td><td>5.11</td><td>7.91</td><td>35.42</td><td>4.1e-5</td><td>5.3e-3</td></tr><tr><td>Fast EM DPS (n=1)</td><td>1min41</td><td>24.68</td><td>0.75</td><td>0.34</td><td>3.23</td><td>6.34</td><td>23.03</td><td>9e-6</td><td>5.1e-3</td></tr><tr><td>Fast EM IIGDM (n=1)</td><td>9sec</td><td>25.66</td><td>0.79</td><td>0.34</td><td>4.26</td><td>7.48</td><td>30.33</td><td>1.1e-5</td><td>5.1e-3</td></tr><tr><td>Fast EM IIGDM (n=4)</td><td>15sec</td><td>25.74</td><td>0.8</td><td>0.34</td><td>4.31</td><td>7.42</td><td>30.15</td><td>6e-6</td><td>5e-3</td></tr><tr><td>Fast EM IIGDM (n=16)</td><td>55sec</td><td>25.75</td><td>0.8</td><td>0.34</td><td>4.28</td><td>7.46</td><td>29.61</td><td>1.1e-5</td><td>5e-3</td></tr></table>

Table 1. Model comparison on FFHQ synthetic dataset. Models with $\textbf { a } ^ { 6 * }$ correspond to non-blind models used as baselines. Best blind models are in bold while second best are underlined. Note that baselines do not count for best model rankings.

mation (9) to:

$$
\widehat { Q } ( H , H _ { t } ) = E _ { x \sim q _ { t } ( x _ { 0 } | y , H _ { t } ) } [ \log ( p ( y | x , H ) ]\tag{31}
$$

$$
= \frac { - 1 } { 2 \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } \| H \widehat { x } _ { 0 } ^ { i } ( t ) - y \| _ { 2 } ^ { 2 } .\tag{32}
$$

In this case, the M-step is equivalent to the classical diffusion EM M-step of Equation (21) but applied in the current estimate $\widehat { x } _ { 0 } ^ { i } ( t )$ instead of the real sample $x ^ { i }$ . In the case of the ΠGDM approximation (10), we have:

$$
\widehat { Q } ( H , H _ { t } ) = \frac { - 1 } { 2 \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } E _ { x \sim \mathcal { N } ( \widehat { x } _ { 0 } ^ { i } ( t ) , r _ { t } ^ { 2 } ) } [ \| H x - y \| _ { 2 } ^ { 2 } ] .\tag{33}
$$

The computations for the M-step in that case are left in $\mathsf { A p - }$ pendix D. Eventually, the only difference between the fast EM diffusion algorithm and a classical non-blind diffusion model is that we first estimate the blur kernel before applying the guidance. Our algorithm demonstrates comparable computational efficiency to non-blind diffusion algorithms, as the computation of the M-step negligibly impacts the overall diffusion process. The algorithm’s pseudo-code can be found in Algorithm 2. Note that in the pseudo-code, the n particles are treated as a batch directly in the $x _ { t }$ . To point out this difference, all the variables that are seen as a batch are written in bold.

## 4. Experiments

## 4.1. Experimental settings

We test our algorithm on the first 1000 validation images of the widely used FFHQ [24] 256x256 dataset that we degrade with random motion blur kernels computed using [15] and random Gaussian noise with noise level $\sigma \in$ {5, 10, 20}. We also provide some results on DIV2K [1] dataset. To achieve a fair comparison, we use the code and pre-trained weights provided by the authors of Blind DPS. For ΠGDM, there is no public code so we re-implemented the model using the Blind DPS code backbone. In our experiments, we observed that DPS needs more iterations to properly converge in comparison to ΠGDM. Indeed, the DPS run needs 1000 iterations while we only use 100 iterations for ΠGDM. For the kernel estimation, we use a bias-free FFDNet [56] denoiser trained on a dataset of motion blur kernels for the Plug & Play regularization. At test time, the M-step consists of 10 HQS iterations with hyperparameters $\lambda = 1$ and $\beta = 1 e 5$ . We provide experiments with different numbers of particles for both the Diffusion EM algorithm and the Fast diffusion EM algorithm. We use $n \in \{ 1 , 4 , 1 6 \}$ . All the models are evaluated on a single A100 GPU.

## 4.2. Compared methods

To test the efficiency of our method, we compare it to state-of-the-art models for deconvolution. We chose to compare against both optimization-based methods, deep learning approaches, and diffusion models to cover all the existing approaches. More specifically, we compare our method to [3] which is a MAP-based method for kernel estimation that uses $\ell _ { 0 }$ norm on the gradient of the image as an image prior and $\ell _ { 2 }$ norm to regularize the kernel. We also compare to self-deblur [38] which is a blind deconvolution method that provides both image reconstruction and kernel estimation based on Deep Image Prior. We provide comparisons with MPRNet [52] which is a multi-scale deep learning architecture design for image restoration problems that has proven its efficiency in deblurring. Finally, we compare our kernel estimation methods to Blind DPS [6] which consists of two parallel diffusion models that jointly model the restored image and its corresponding blur kernel. We also computed the results of the non-blind model DPS and ΠGDM to highlight the loss of quality between the blind and non-blind models. For all the methods, we used the source code and pre-trained weights provided by the author.

![](images/442707f07300e9b5e1501e3cf4f5f0bb76ba64d91c5493b200343665efc01410.jpg)  
Figure 3. Visual comparison of the different models on a degraded version of the FFHQ 256x256 dataset. Ours correspond to Fast EM.

## 4.3. Quantitative results

Table 1 shows the results of the different models on FFHQ synthetic dataset. We compute both classical metrics with full or reduced reference such as PSNR, SSIM [48], LPIPS [58] and FID [19], no-reference metrics to measure perceptual quality such as NIQE [35] and BRISQUE [34] and kernel metrics such as the Mean-Squared Error (MSE) on the reconstructed kernel. We also measure the consistency of the estimated image x and kernel $\widehat { H }$ with the forward model by means of:

$$
\mathcal { L } _ { r e b l u r } ( y , \widehat { x } , \widehat { H } ) = \| \widehat { H } \widehat { x } - y \| _ { 2 } ^ { 2 } - \sigma ^ { 2 } M\tag{34}
$$

where M = 3hw is the number of elements in vector x. We observe that classical optimization-based approaches such as Anger $\ell _ { 0 }$ [3] and Self-Deblur [38] fail to estimate the blur and reconstruct the image efficiently. The main problem with those approaches is that they fail to produce pleasant results in the presence of noise. While Anger $\ell _ { 0 }$ [3] produces results with over-sharpened noise, Self-Deblur [38] completely fails to both estimate the kernel and deblur the image. MPRNet produces better results but with artifacts due to the noise, it also fails to recover high-frequency details which is a common problem when using deep-learning models trained on mean-squared error. Diffusion-based models seem to be the most efficient. Blind DPS ranks best among the no-reference perceptual metrics and FID while ranking below our model both for reference metrics and kernel estimation. Figure 3, shows some example images where we can notice the sharpness and high quality of Blind DPS results. In our experiments, we observed that Blind DPS sometimes fails to efficiently estimate the blur kernel, especially in the presence of noise. We also noticed that on some images Blind DPS was producing sharper results than our model, even with a worst kernel prediction which is surprising since we use the same diffusion model. Yet, the fact that our model has better full-reference metrics and better measurement consistency points out the fact that Blind DPS hallucinates more details. We also conducted experiments on deblurring images from DIV2K dataset while keeping the same FFHQ-trained score model for testing. In that particular case, the prior of the score model does not match the distribution of the test images so the model won’t be able to hallucinate accurate details. Some visual results of those experiments can be found in Figure 5. Those experiments showed that our model and especially the one based on ΠGDM diffusion produces sharper results. It highlights the fact that Blind DPS and DPS, in general, have weaker guidance than ΠGDM, so it requires a more accurate score model which can be a limitation in practice since training a score model on the space of natural images is not an easy task. During our experiments, we realized that Fast Diffusion EM was both faster and better in terms of quality than

![](images/f5d4fb7e66e455193ac7a85206b57396f76ef07d8272cabe439caabd75ff36b0.jpg)  
Figure 4. Comparison of the efficiency of the different kernel regularizations depending on the noise level $\sigma \in [ 0 , 2 0 ]$ . The vertical axis shows the mean MSE over the whole FFHQ dataset for kernel estimation from a noisy and blurred observation of a known image.

![](images/1cf5fe523d94ff68f53e188a4fe93e308e38c8bdd2398fb4de0da7a7e6c3c52d.jpg)  
(a) LR

![](images/5eb1bd53df02d03bc2d748d50d06e07a6fadbb8ca9b5bf99cd75781c3ccddbe9.jpg)  
(b) HR

![](images/d9ef64b0e82df93190cfd1e3d810e9b119d406944e3a53a1a858263de39c5158.jpg)  
(c) Blind DPS

![](images/e01534edd9fc2a8a2ff01976430e4f33711f3280179c8e4c3d937c78210e0a13.jpg)  
(d) Fast EM DPS

![](images/3049adfa3523f1ee4b8756a3aafc218b40d2510abf094463efa24261a789e9c2.jpg)  
(e) Fast EM ΠGDM

Figure 5. Visual comparison on out-of-distribution images. The score network is trained on FFHQ dataset while we test on DIV2K.
<table><tr><td></td><td>n-samples</td><td>Runtime</td><td>PSNR</td><td>PSNR SA</td></tr><tr><td rowspan="3">Diffusion EM</td><td>n=1</td><td>1min30sec</td><td>23.4</td><td>23.4</td></tr><tr><td>n=4</td><td>2min30sec</td><td>23.21</td><td>23.43</td></tr><tr><td>n=16</td><td>9min10sec</td><td>23.09</td><td>23.37</td></tr><tr><td rowspan="3">Fast Diffusion EM</td><td>n=1</td><td>9sec</td><td>25.66</td><td>25.66</td></tr><tr><td>n=4</td><td>15sec</td><td>25.74</td><td>26.14</td></tr><tr><td>n=16</td><td>55sec</td><td>25.75</td><td>26.16</td></tr></table>

Table 2. Influence of the number of samples used to estimate the E-step in Fast EM ΠGDM. The image PSNR is computed on the first image of the batch.

Diffusion EM. Indeed, Diffusion EM is sometimes stuck in the no blur solution while we never observed this problem for Fast Diffusion EM. In terms of metrics, both Fast EM DPS and Fast EM ΠGDM have better reference metrics than all the other methods, and for any number of particles. We observed better performance and faster runtime with the ΠGDM model, probably because it has stronger guidance, thus, it is easier for the M-step to estimate the blur kernel. Fast EM ΠGDM performance in no-reference metrics NIQE and BRISQUE is worse than the other diffusionbased methods: BlindDPS and Fast EM DPS have indeed slightly sharper results, but they are less accurate and less consistent (see the hallucinations of BlindDPS in the second line in Figure 3). In terms of runtime, our ΠGDMbased model ranks best among diffusion models but it is significantly slower than MPRNet and Anger $\ell _ { 0 }$

## 4.4. Ablation studies

In this section, we discuss the efficiency of the different blocks of our algorithm. We first provide some additional results that show the efficiency of the proposed Plug & Playbased kernel regularization. Next, we study the influence of the number of samples used to estimate the E-step on the quality of the final results. To compare the efficiency of our regularization, we compared it against the $\ell _ { 1 }$ and $\ell _ { 2 }$ regularizations. To do so, we use our FFHQ synthetic dataset and estimate the blur kernel in the non-blind setting where the sharp and blurry images are both known. We compute the MSE of the reconstructed kernel for several noise levels. For all the regularizations, we used the same optimization scheme, HQS, and fine-tuned the hyper-parameters of the regularizations separately. Figure 4 shows the obtained results. We observed that our regularization is significantly better in the presence of noise and the loss of quality between $\sigma = 5$ and $\sigma = 2 0$ is very small. Finally, we also investigated the influence of the number of samples in our algorithms. We observed in Table 1 and Table 2 that increasing the number of samples increases the image reconstruction and kernel estimation accuracy. Using all the samples, we can also compute the PSNR on the average of the samples produced by the model. We refer to this metric as the “PSNR SA” in Table 2. Usually, the PSNR SA gives a higher PSNR than the PSNR on a single image, even if the average image is less sharp. We also observed that in the case of Diffusion EM, increasing the number of samples lowers the PSNR but improves all the other metrics. Averaging several samples is also possible with methods such as Blind-DPS, the main difference is that in our approach, all the samples have the same guidance at each diffusion step since we estimate a single kernel for all the samples. In Blind-DPS, all the samples have their respective kernels.

## 5. Conclusion

In this article, we present a novel approach for blind deconvolution based on diffusion models. In particular, we designed Diffusion EM, an algorithm based on the Expectation-Maximization algorithm. This algorithm consists of an E-step, which approximates the expected value of the log-likelihood using a diffusion model, and an M-step, which maximizes this expected log-likelihood with respect to the unknown parameters (the blur kernel). For the Mstep, we introduced a novel kernel regularization based on a Plug & Play denoiser. The diffusion EM algorithm is slow since it requires running a diffusion model several times. We propose an acceleration of the algorithm that directly injects the EM iterations into the diffusion process (leveraging the intermediate diffusion steps as approximate posterior samples). We observed that this Fast EM diffusion model reaches better performance than the original diffusion EM algorithm while being significantly faster. Finally, we demonstrate the efficiency of our approach both quantitatively and visually. We compare our approach to state-ofthe-art methods for blind deconvolution and provide several ablation studies that highlight the performance of our regularization and model and give insights into the behavior of the model. In its current form, our algorithm is limited to deconvolution. Future research will address more general blind deblurring problems [5, 9]. Faster diffusion models such as latent diffusion [8, 40] or diffusion bridges [30] could also benefit our method.

## References

[1] Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, July 2017. 6

[2] Brian D.O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12, 1982. 3

[3] Jer´ emy Anger, Gabriele Facciolo, and Mauricio Delbracio.´ Blind Image Deblurring using the l0 Gradient Prior. Image Processing On Line (IPOL), 2019. 2, 6, 7

[4] Sefi Bell-Kligler, Assaf Shocher, and Michal Irani. Blind Super-Resolution Kernel Estimation using an Internal-GAN. In Advances in Neural Information Processing Systems (NIPS), 2019. 2

[5] Guillermo Carbajal, Patricia Vitoria, Jose Lezama, and Pablo´ Muse. Blind Motion Deblurring With Pixel-Wise Kernel Es-´ timation via Kernel Prediction Networks. IEEE Transactions on Computational Imaging, 9:928–943, aug 2023. 8

[6] Hyungjin Chung, Jeongsol Kim, Sehui Kim, and Jong Chul Ye. Parallel Diffusion Models of Operator and Image for Blind Inverse Problems. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2, 6

[7] Hyungjin Chung, Jeongsol Kim, Michael T. Mccann, Marc L. Klasky, and Jong Chul Ye. Diffusion Posterior Sampling for General Noisy Inverse Problems. In International Conference on Learning Representations (ICLR), 2023. arXiv:2209.14687 [cs, stat]. 1, 3, 4

[8] Hyungjin Chung, Jong Chul Ye, Peyman Milanfar, and Mauricio Delbracio. Prompt-tuning latent diffusion models for inverse problems. arXiv:2310.01110, 2023. 8

[9] Valentin Debarnot and Pierre Weiss. Deep-Blur : Blind Identification and Deblurring with Convolutional Neural Networks. Preprint hal-03687822, 2022. 8

[10] A. P. Dempster, N. M. Laird, and D. B. Rubin. Maximum likelihood from incomplete data via the em algorithm. Journal of the Royal Statistical Society. Series B (Methodological), 39(1):1–38, 1977. 2

[11] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution. In David Fleet, Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, European Conference on Computer Vision (ECCV), 2014. 1

[12] Randal Douc, Eric Moulines, and David Stoffer. Nonlinear Time Series. Chapman and Hall/CRC, jan 2014. 13

[13] Gersende Fort, Edouard Ollier, and Adeline Samson. Stochastic proximal-gradient algorithms for penalized mixed models. Statistics and Computing, 29(2):231–253, 2019. 2

[14] Angela F Gao, Jorge C Castellanos, Yisong Yue, Zachary E Ross, and Katherine L Bouman. DeepGEM: Generalized Expectation-Maximization for Blind Inversion. In Advances in Neural Information Processing Systems (NIPS), 2021. 2

[15] Fabien Gavant, Laurent Alacoque, Antoine Dupret, and Dominique David. A physiological camera shake model for image stabilization systems. In SENSORS, IEEE, pages 1461– 1464, 2011. 6

[16] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative Adversarial Nets. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems (NIPS), 2014. 1

[17] Remi Gribonval. Should penalized least squares regression´ be interpreted as maximum a posteriori estimation? IEEE Transactions on Signal Processing, 2011. 5

[18] Bichuan Guo, Yuxing Han, and Jiangtao Wen. AGEM: Solving Linear Inverse Problems via Deep Priors and Sampling. In Advances in Neural Information Processing Systems, 2019. 2

[19] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. 7

[20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. In Advances in Neural Information Processing Systems (NIPS), 2020. 1, 3

[21] Jonathan Ho and Tim Salimans. Classifier-Free Diffusion Guidance. In NeurIPS Workshop on Deep Generative Models and Downstream Applications, 2022. arXiv:2207.12598 [cs]. 1

[22] Samuel Hurault, Arthur Leclaire, and Nicolas Papadakis. Gradient Step Denoiser for convergent Plug-and-Play. In International Conference on Learning Representations (ICLR’22), International Conference on Learning Representations, Online, United States, Apr. 2022. 1

[23] Ulugbek S. Kamilov, Charles A. Bouman, Gregery T. Buzzard, and Brendt Wohlberg. Plug-and-play methods for integrating physical and learned models in computational imaging: Theory, algorithms, and applications. IEEE Signal Processing Magazine, 40(1):85–97, 2023. 1

[24] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2019. 6

[25] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising Diffusion Restoration Models. In Advances in Neural Information Processing Systems (NIPS), 2022. 1

[26] Charles Laroche, Andres Almansa, and Matias Tassano.´ Deep Model-Based Super-Resolution With Non-Uniform Blur. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2023. 1

[27] Remi Laumont, Valentin De Bortoli, Andr´ es Almansa, Julie´ Delon, Alain Durmus, and Marcelo Pereyra. Bayesian imaging using plug & play priors: When langevin meets tweedie. SIAM Journal on Imaging Sciences, 2022. 1

[28] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. SwinIR: Image Restoration Using Swin Transformer. In IEEE/CVF International Conference on Computer Vision (CVPR), 2021. 1

[29] Jingyun Liang, Kai Zhang, Shuhang Gu, Luc Van Gool, and Radu Timofte. Flow-based kernel prior with application to

blind super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 2

[30] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A. Theodorou, Weili Nie, and Anima Anandkumar. I<sup>2</sup>sb: Image-to-image schrodinger bridge, 2023.¨ 8

[31] Ziwei Luo, Haibin Huang, Lei Yu, Youwei Li, Haoqiang Fan, and Shuaicheng Liu. Deep Constrained Least Squares for Blind Image Super-Resolution. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2

[32] Geoffrey J. McLachlan and Thriyambakam Krishnan. The EM algorithm and extensions. Wiley series in probability and statistics. Wiley, Hoboken, NJ, 2. ed edition, 2008. 2

[33] Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. Arxiv, 2014. 1

[34] Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. No-reference image quality assessment in the spatial domain. IEEE Transactions on Image Processing, 21(12):4695–4708, 2012. 7

[35] Anish Mittal, Rajiv Soundararajan, and Alan C. Bovik. Making a “completely blind” image quality analyzer. IEEE Signal Processing Letters, 20(3):209–212, 2013. 7

[36] Søren Feodor Nielsen. The stochastic em algorithm: Estimation and asymptotic results. Bernoulli, 2000. 4

[37] Daniele Perrone and Paolo Favaro. Total Variation Blind Deconvolution: The Devil Is in the Details. In IEEE/CVF International Conference on Computer Vision (CVPR), 2014. 1, 2

[38] Dongwei Ren, Kai Zhang, Qilong Wang, Qinghua Hu, and Wangmeng Zuo. Neural Blind Deconvolution Using Deep Priors. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2, 6, 7

[39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 5

[40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022. 8

[41] Ernest Ryu, Jialin Liu, Sicheng Wang, Xiaohan Chen, Zhangyang Wang, and Wotao Yin. Plug-and-Play Methods Provably Converge with Properly Trained Denoisers. In International Conference on Machine Learning (ICML), 2019. 1

[42] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Image Super-Resolution via Iterative Refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023. 1

[43] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning, 2015. 1

[44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. In International Conference on Learning Representations (ICLR), 2021. 1

[45] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-Guided Diffusion Models for Inverse Problems. In International Conference on Learning Representations (ICLR), 2023. 1, 3, 4

[46] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations (ICLR), 2021. 3

[47] Ana Fernandez Vidal, Valentin De Bortoli, Marcelo Pereyra, and Alain Durmus. Maximum likelihood estimation of regularisation parameters in high-dimensional inverse problems: an empirical Bayesian approach. Part I: Methodology and Experiments. SIAM Journal on Imaging Sciences, 13(4):1945–1989, nov 2019. 2

[48] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 2004. 7

[49] Greg C. G. Wei and Martin A. Tanner. A monte carlo implementation of the em algorithm and the poor man’s data augmentation algorithms. Journal of the American Statistical Association, 85:699–704, 1990. 3

[50] Jay Whang, Mauricio Delbracio, Hossein Talebi, Chitwan Saharia, Alexandros G. Dimakis, and Peyman Milanfar. Deblurring via Stochastic Refinement. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1

[51] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao. Multi-Stage Progressive Image Restoration. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 1

[52] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao. Multi-stage progressive image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14821–14831, June 2021. 6

[53] Kai Zhang, Luc Van Gool, and Radu Timofte. Deep Unfolding Network for Image Super-Resolution. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 1

[54] Kai Zhang, Yawei Li, Wangmeng Zuo, Lei Zhang, Luc Van Gool, and Radu Timofte. Plug-and-play image restoration with deep denoiser prior. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021. 5

[55] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. IEEE Transactions on Image Processing, 2017. Conference Name: IEEE Transactions on Image Processing. 1, 5

[56] Kai Zhang, Wangmeng Zuo, and Lei Zhang. FFDNet: Toward a fast and flexible solution for cnn-based image denoising. In IEEE Transaction on Image Processing, 2018. 6

[57] Kai Zhang, Wangmeng Zuo, and Lei Zhang. Learning a Single Convolutional Super-Resolution Network for Multiple Degradations. In EEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 1

[58] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 7

## A. Iterative Diffusion EM algorithm

Algorithm A.1 summarizes the Diffusion EM algorithm described in sections 3.1 and 3.2.

Algorithm A.1 Diffusion EM algorithm   
Require: $y , \sigma , H _ { 0 } , L ,$   
Ensure: H ≈ arg min<sub>H</sub> $p ( \boldsymbol { y } | H )$ and $x _ { 0 } ^ { i } \sim p ( x _ { 0 } | y , H )$   
for l = 1 to L do   
$\pmb { x } = E \ – s t e p ( y , H _ { l - 1 } , \sigma )$ ▷ n samples from Alg. 1   
$H _ { l } = M \mathbf { - } s t e p ( y , x , \sigma )$ ▷ Iterate (24) and (25)   
end for   
return $\mathbf { \boldsymbol { x } } , H _ { L }$

## B. M-step computations

In this section, we derive the computation of the M-step. In particular, we solve Equation (22) from the main paper:

$$
Z ^ { * } = \arg \operatorname* { m i n } _ { Z \in { \mathcal { C } } } \frac { 1 } { 2 \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } \| Z x ^ { i } - y \| _ { 2 } ^ { 2 } + \frac { \beta } { 2 } \| Z - H \| _ { 2 } ^ { 2 } .\tag{B.1}
$$

with $\mathcal { C }$ the space of convolution operators.

In order to account for the fact that $H \in { \mathcal { C } }$ and $Z _ { t } \in \mathcal { C }$ are convolution operators, we rewrite the same equation in the Fourier domain, where the operators H and Z become diagonal:

$$
{ \mathcal { F } } ( H ) = \mathrm { d i a g } ( h ( 1 ) , \ldots , h ( d ) ) ,\tag{B.2}
$$

$$
{ \mathcal { F } } ( Z ) = \mathrm { d i a g } ( z ( 1 ) , \ldots , z ( d ) ) .\tag{B.3}
$$

Re-writing the minimization in the Fourier domain leads to:

$$
\begin{array} { l } { \displaystyle \mathcal { F } ( Z ^ { * } ) = \arg \underset { Z \in \mathcal { C } } { \operatorname* { m i n } } \frac { 1 } { 2 \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } \| \mathcal { F } ( Z ) \mathcal { F } ( x ^ { i } ) - \mathcal { F } ( y ) \| _ { 2 } ^ { 2 } + \frac { \beta } { 2 } \| \mathcal { F } ( Z ) - \mathcal { F } ( H ) \| _ { 2 } ^ { 2 } } \\ { \displaystyle \quad \quad = \arg \operatorname* { m i n } _ { z } \frac { 1 } { 2 \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { d } | z ( j ) \mathcal { F } ( x ^ { i } ) ( j ) - \mathcal { F } ( y ) ( j ) | ^ { 2 } + \frac { \beta } { 2 } \sum _ { j = 1 } ^ { d } | z ( j ) - h ( j ) | ^ { 2 } . } \end{array}\tag{B.4}
$$

(B.5)

It is straightforward that the solution to the problem is also diagonal, thus we have:

$$
{ \mathcal { F } } ( Z ^ { * } ) = \mathrm { d i a g } ( z ^ { * } ( 1 ) , \ldots , z ^ { * } ( d ) ) .\tag{B.6}
$$

Using the first-order condition and the diagonal structure of the problem, we get the following:

$$
\frac { 1 } { \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } \left[ z ^ { * } ( j ) \mathcal { F } ( x ^ { i } ) ( j ) - \mathcal { F } ( y ) ( j ) \right] \overline { { \mathcal { F } ( x ^ { i } ) ( j ) } } + \beta ( z ^ { * } ( j ) - k ( j ) ) = 0\tag{B.7}
$$

$$
\Leftrightarrow z ^ { * } ( j ) \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | \mathcal { F } ( x ^ { i } ) ( j ) | ^ { 2 } + \sigma ^ { 2 } \beta \right) = \mathcal { F } ( y ) ( j ) \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \overline { { \mathcal { F } ( x ^ { i } ) ( j ) } } + \sigma ^ { 2 } \beta k ( j )\tag{B.8}
$$

$$
\Leftrightarrow z ^ { * } ( j ) = \frac { \mathcal { F } ( y ) ( j ) \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \overline { { \mathcal { F } ( x ^ { i } ) ( j ) } } + \sigma ^ { 2 } \beta k ( j ) } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | \mathcal { F } ( x ^ { i } ) ( j ) | ^ { 2 } + \sigma ^ { 2 } \beta } .\tag{B.9}
$$

## C. M-step computations with DPS approximation

In this section, we develop the computation of the M-step in Fast EM for DPS. We start from Equation (32) of the main paper:

$$
\widehat { Q } ( Z , Z _ { t } ) = \frac { - 1 } { 2 \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } \| Z \widehat { x } _ { 0 } ^ { i } ( t ) - y \| _ { 2 } ^ { 2 } ] .\tag{C.1}
$$

Our goal is to compute:

$$
Z ^ { * } = a r g \operatorname* { m i n } _ { Z \in \mathcal { C } } - \widehat { Q } ( Z , Z _ { t } ) + ( \beta / 2 ) \| Z - H \| _ { 2 } ^ { 2 } .\tag{C.2}
$$

We can notice that it is similar to Equation (B.4) with $\widehat { x } _ { 0 } ^ { i } ( t )$ instead of $x ^ { i }$ . Thus we have that:

$$
z ^ { * } ( j ) = \frac { \mathcal { F } ( y ) ( j ) \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \overline { { \mathcal { F } ( \widehat { x } _ { 0 } ^ { i } ( t ) ) ( j ) } } + \sigma ^ { 2 } \beta h ( j ) } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | \mathcal { F } ( \widehat { x } _ { 0 } ^ { i } ( t ) ) ( j ) | ^ { 2 } + \sigma ^ { 2 } \beta } .\tag{C.3}
$$

## D. M-step computations with ΠGDM approximations

In this section, we develop the computation of the M-step in Fast EM for ΠGDM. We start from Equation (33) of the main paper:

$$
\widehat { Q } ( H , H _ { t } ) = \frac { - 1 } { 2 \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } E _ { x \sim \mathcal { N } ( \widehat { x } _ { 0 } ^ { i } ( t ) , r _ { t } ^ { 2 } ) } [ \| H x - y \| _ { 2 } ^ { 2 } ] .\tag{D.1}
$$

Our goal is to compute:

$$
Z ^ { * } = \arg \operatorname* { m i n } _ { Z \in { \mathcal { C } } } - \widehat { Q } ( Z , Z _ { t } ) + ( \beta / 2 ) \| Z - H \| _ { 2 } ^ { 2 } .\tag{D.2}
$$

Similarly to Section B, we work with diagonal operators so we have:

$$
\mathcal { F } ( H ) = \mathrm { d i a g } ( h ( 1 ) , \ldots , h ( d ) )\tag{D.3}
$$

$$
{ \mathcal { F } } ( Z ) = \mathrm { d i a g } ( z ( 1 ) , \ldots , z ( d ) ) .\tag{D.4}
$$

and thus:

$$
{ \mathcal { F } } ( Z ^ { * } ) = \mathrm { d i a g } ( z ^ { * } ( 1 ) , \ldots , z ^ { * } ( d ) ) .\tag{D.5}
$$

We start by rewriting Equation D.1 in the Fourier domain using the fact that the Fourier transform preserves norms:

$$
\mathcal { F } ( Z ^ { * } ) = \arg \operatorname* { m i n } _ { z } \frac { 1 } { 2 \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { d } E _ { x \sim \mathcal { N } ( \widehat { x } _ { 0 } ^ { i } ( t ) , r _ { t } ^ { 2 } ) } [ | z ( j ) \mathcal { F } ( x ) ( j ) - \mathcal { F } ( y ) ( j ) | ^ { 2 } ] + ( \beta / 2 ) \sum _ { j = 1 } ^ { d } | z ( j ) - h ( j ) | ^ { 2 } .\tag{D.6}
$$

We solve this problem using the first-order condition element by element since the problem is diagonal, the derivation inside the expectancy can be done using Fisher identity [12, Proposition D.4]:

$$
\frac { 1 } { \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } E _ { x \sim \mathcal { N } ( \hat { x } _ { 0 } ^ { i } ( t ) , r _ { t } ^ { 2 } ) } [ | z ( j ) \mathcal { F } ( x ) ( j ) - \mathcal { F } ( y ) ( j ) | \overline { { \mathcal { F } ( x ) ( j ) } } ] + \beta ( z ( j ) - h ( j ) ) = 0\tag{D.7}
$$

$$
\Leftrightarrow z ( j ) \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } E _ { x \sim \mathcal { N } ( \hat { x _ { 0 } ^ { i } } ( t ) , r _ { t } ^ { 2 } ) } [ | \mathcal { F } ( x ) ( j ) | ^ { 2 } ] + \sigma ^ { 2 } \beta \right] = \mathcal { F } ( y ) ( j ) \frac { 1 } { n } \sum _ { i = 1 } ^ { n } E _ { x \sim \mathcal { N } ( \hat { x _ { 0 } ^ { i } } ( t ) , r _ { t } ^ { 2 } ) } [ \overline { { \mathcal { F } ( x ) ( j ) } } ] + \sigma ^ { 2 } \beta h ( j )\tag{D.8}
$$

Using the fact that the Fourier transform of a white Gaussian noise of variance $\sigma ^ { 2 }$ is a white Gaussian noise of variance $\sigma ^ { 2 }$ the expected values yield:

$$
\begin{array} { r l } & { E _ { x \sim \mathcal { N } ( \widehat { x } _ { 0 } , r _ { t } ^ { 2 } ) } [ | \mathcal { F } ( x ) ( j ) | ^ { 2 } ] = r _ { t } ^ { 2 } + | \mathcal { F } ( \widehat { x } _ { 0 } ) ( j ) | ^ { 2 } } \\ & { \qquad E _ { x \sim \mathcal { N } ( \widehat { x } _ { 0 } , r _ { t } ^ { 2 } ) } [ \overline { { \mathcal { F } ( x ) ( j ) } } ] = \overline { { \mathcal { F } ( \widehat { x } _ { 0 } ) ( j ) } } } \end{array}
$$

So we can conclude that:

$$
z ^ { * } ( j ) = \frac { \mathcal { F } ( y ) ( j ) \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \overline { { \mathcal { F } ( \widehat { x } _ { 0 } ^ { i } ( t ) ) ( j ) } } + \sigma ^ { 2 } \beta h ( j ) } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | \mathcal { F } ( \widehat { x } _ { 0 } ^ { i } ( t ) ) ( j ) | ^ { 2 } + r _ { t } ^ { 2 } + \sigma ^ { 2 } \beta }\tag{D.9}
$$

The main difference with DPS approximation is that we have an extra term in the denominator $r _ { t } ^ { 2 }$

## E. Additional results

See Figure E.1.

![](images/52d309d11262a9b8fd64abba0ba09cda8124a7a3006a79b4e617df8cd911b544.jpg)  
Figure E.1. Visual comparison of the different models on a degraded version of FFHQ 256x256 dataset. Ours correspond to Fast EM.