# A Survey on Diffusion Models for Inverse Problems

Giannis Daras<sup>1</sup>, Hyungjin Chung<sup>2</sup>, Chieh-Hsin Lai<sup>3</sup>, Yuki Mitsufuji<sup>3</sup>, Jong Chul Ye<sup>2</sup>, Peyman Milanfar<sup>4</sup>, Alexandros G. Dimakis<sup>1</sup>, Mauricio Delbracio<sup>4</sup>

<sup>1</sup>UT Austin <sup>2</sup>KAIST <sup>3</sup>Sony AI <sup>4</sup>Google

## Abstract

Diffusion models have become increasingly popular for generative modeling due to their ability to generate high-quality samples. This has unlocked exciting new possibilities for solving inverse problems, especially in image restoration and reconstruction, by treating diffusion models as unsupervised priors. This survey provides a comprehensive overview of methods that utilize pre-trained diffusion models to solve inverse problems without requiring further training. We introduce taxonomies to categorize these methods based on both the problems they address and the techniques they employ. We analyze the connections between different approaches, offering insights into their practical implementation and highlighting important considerations. We further discuss specific challenges and potential solutions associated with using latent diffusion models for inverse problems. This work aims to be a valuable resource for those interested in learning about the intersection of diffusion models and inverse problems.

## 1 Introduction

## 1.1 Problem Setting

Inverse problems are ubiquitous and the associated reconstruction problems have tremendous applications across different domains such as seismic imaging [37, 38], weather prediction [39], oceanography [40], audio signal processing [41, 42, 43, 44, 45, 46], medical imaging [47, 48, 49, 50], etc. Despite their generality, inverse problems across different domains follow a fairly unified mathematical setting. Specifically, in inverse problems, the goal is to recover an unknown sample $\pmb { x } \in \mathbb { R } ^ { n }$ from a distribution $p _ { x }$ , assuming access to measurements $\pmb { y } \in \mathbb { R } ^ { m }$ and a corruption model

$$
\mathbf { \nabla } Y = \mathcal { A } ( \mathbf { \nabla } X ) + \sigma _ { y } Z , Z \sim \mathcal { N } ( \mathbf { 0 } , I _ { m } ) .\tag{1.1}
$$

In what follows, we present some well-known examples of measurement models that fit under this general formulation.

Example 1.1 (Denoising). The simplest interesting example is the denoising inverse problem, i.e. when A is the identity matrix and $\sigma _ { y } > 0$ . In fact, the noise model does not have to be Gaussian and it can be generalized to other distributions, including the Laplacian Distribution or the Poisson Distribution [51]. For the purposes of this survey, we focus on additive Gaussian noise.

A lot of practical applications arise from the non-invertible linear setting, i.e. for $\boldsymbol { \mathcal { A } } ( \boldsymbol { X } ) = \boldsymbol { \mathcal { A } } \boldsymbol { X }$ and A being an m n matrix with $m < n$

Example 1.2 (Inpainting). A is a masking matrix, i.e. $A _ { i j } = 0$ for $i \neq j$ and $A _ { i i }$ is either 0 or 1, based on whether the value at this location is observed.

Example 1.3 (Compressed Sensing). A is a matrix with entries sampled from a Gaussian random variable.

Example 1.4 (Convolutions). Here (X) represents the convolution of X with a (Gaussian or other) kernel, which is again a linear operation.

<table><tr><td>Category</td><td>Method</td><td>Non-linear</td><td>Blind</td><td>Handle noise</td><td>Pixel/Latent</td><td>Text-conditioned</td><td>Optimization Technique</td><td>Code1</td></tr><tr><td rowspan="9"></td><td>Score-ALD Jalal et al. [1]</td><td>X</td><td>x</td><td>√</td><td>Pixel</td><td>x</td><td>Grad</td><td>code</td></tr><tr><td>Score-SDE Song et al. [2]</td><td>x</td><td>x</td><td>x</td><td>Pixel</td><td>x</td><td>Proj</td><td>code</td></tr><tr><td>ILVR Choi et al. [3]</td><td>x</td><td>x</td><td>x</td><td>Pixel</td><td>x</td><td>Proj</td><td>code</td></tr><tr><td>DPS Chung et al. [4]</td><td>√</td><td>x</td><td>√</td><td>Pixel</td><td>x</td><td>Grad</td><td>code</td></tr><tr><td>IIGDM Song et al. [5]</td><td>√</td><td>x</td><td>V</td><td>Pixel</td><td>x</td><td>Grad</td><td>code</td></tr><tr><td>Moment Matching Rozet et al. [6]</td><td></td><td>x</td><td>√</td><td>Pixel</td><td>x</td><td>Grad</td><td>code</td></tr><tr><td>BlindDPS Chung et al. [7]</td><td>い√</td><td>√</td><td>V</td><td>Pixel</td><td>x</td><td>Grad</td><td>code</td></tr><tr><td>SNIPS Kawar et al. [8]</td><td>××</td><td>x</td><td>√</td><td>Pixel</td><td>x</td><td>Grad</td><td>code</td></tr><tr><td>DDRM Kawar et al. [9]</td><td></td><td>x</td><td>√</td><td>Pixel</td><td>×</td><td>Grad</td><td>code</td></tr><tr><td rowspan="5">Ex  a or meatt mhing</td><td>GibbsDDRM Murata et al. [10]</td><td></td><td>√</td><td>√</td><td>Grad</td><td>x</td><td>Samp</td><td>code</td></tr><tr><td>DDNM Wang et al. [11]</td><td></td><td>x</td><td></td><td>Pixel</td><td>x</td><td>Proj</td><td>code</td></tr><tr><td>DDS Chung et al. [12]</td><td>×××</td><td>x</td><td>√</td><td>Pixel</td><td>x</td><td>Opt</td><td>code</td></tr><tr><td>DiffPIR Zhu et al. [13]</td><td>×</td><td>x</td><td>V</td><td>Pixel</td><td>x</td><td>Opt</td><td>code</td></tr><tr><td>PSLD Rout et al. [14]</td><td>√</td><td>x</td><td>√</td><td>Latent</td><td>x</td><td>Grad</td><td>code</td></tr><tr><td></td><td>STSL Rout et al. [15]</td><td>√</td><td>x</td><td>√</td><td>Latent</td><td>x</td><td>Grad</td><td>x</td></tr><tr><td rowspan="3">Vaiitonal iunfnce</td><td>RED-Diff Mardani et al. [16]</td><td>√</td><td>x</td><td>√</td><td>Pixel</td><td>x</td><td>Opt</td><td>code</td></tr><tr><td>Blind RED-Diff Alkan et al. [17]</td><td>√</td><td>V</td><td>√</td><td>Pixel</td><td>x</td><td>Opt</td><td>x</td></tr><tr><td>Score Prior Feng et al. [18]</td><td>√</td><td>x</td><td>√</td><td>Pixel</td><td>x</td><td>Opt</td><td>code</td></tr><tr><td rowspan="3"></td><td>Efficient Score Prior Feng and Bouman [19]</td><td>√</td><td>x</td><td>√</td><td>Pixel</td><td>x</td><td>Opt</td><td>code</td></tr><tr><td></td><td></td><td>x</td><td>√</td><td>Pixel</td><td>x</td><td></td><td></td></tr><tr><td>DMPlug Wang et al. [20] SHRED Chihaoui et al. [21]</td><td>√</td><td>x</td><td>√</td><td></td><td></td><td>Opt</td><td>code</td></tr><tr><td rowspan="3">CSGM mettodds</td><td></td><td>√</td><td>x</td><td></td><td>Pixel</td><td>x x</td><td>Opt</td><td>x x</td></tr><tr><td>Consistent-CSGM [22] Score-ILO Daras et al. [23]</td><td>√ √</td><td>x</td><td></td><td>Pixel Pixel</td><td>x</td><td>Opt Opt</td><td>code</td></tr><tr><td>PnP-DM Wu et al. [24]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">Asyaly Exatct Mods</td><td>FPS Dou and Song [25]</td><td>√ √</td><td>x</td><td>√</td><td>Pixel</td><td>x</td><td>Opt</td><td>x</td></tr><tr><td>PMC Sun et al. [26]</td><td></td><td>x</td><td>√</td><td>Pixel</td><td>x x</td><td>Samp</td><td>code</td></tr><tr><td>SMCDiff Trippe et al. [27]</td><td>V</td><td>x</td><td>V</td><td>Pixel</td><td>x</td><td>Samp Samp</td><td>code</td></tr><tr><td>MCGDiff Cardoso et al. [28]</td><td>x</td><td>x x</td><td>√</td><td>Pixel Pixel</td><td>x</td><td></td><td>code</td></tr><tr><td>TDS Wu et al. [29]</td><td>x x</td><td>x</td><td>√</td><td>Pixel</td><td>x</td><td>Samp Samp</td><td>code</td></tr><tr><td rowspan="7">metodds Oter</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>code</td></tr><tr><td>Implicit denoiser prior Kadkhodaie and Simoncelli [30]</td><td>x</td><td>x</td><td>X x</td><td>Pixel</td><td>x</td><td>Proj</td><td>code</td></tr><tr><td>MCG Chung et al. [31]</td><td>x</td><td>x x</td><td>V</td><td>Pixel</td><td>x x</td><td>Grad/Proj</td><td>code</td></tr><tr><td>Resample Song et al. [32]</td><td></td><td>x</td><td></td><td>Latent Pixel/Latent</td><td>√</td><td>Grad/Opt Grad/Opt</td><td>code</td></tr><tr><td>MPGD He et al. [33]</td><td></td><td>x</td><td></td><td>Latent</td><td>√</td><td>Grad/Opt</td><td>code</td></tr><tr><td>P2L Chung et al. [34] TReg Kim et al. [35]</td><td></td><td>x</td><td></td><td>Latent</td><td>√</td><td>Grad/Opt</td><td>x</td></tr><tr><td>DreamSampler Kim et al. [36]</td><td></td><td>x</td><td></td><td>Latent</td><td>√</td><td>Grad/Opt</td><td>x code</td></tr></table>

Table 1: Categorization of Diffusion-Based Inverse Problem Solvers. This table categorizes methods by their approach to solving inverse problems with diffusion models. We identified four families of methods. Explicit Approximations $f o r$ Measurement Matching: These methods approximate the measurement matching score, $\nabla \log p _ { t } ( \pmb { y } | \pmb { x } _ { t } )$ , with a closed-form expression. Variational Inference: These methods approximate the true posterior distribution, $p ( { \pmb x } | { \pmb y } )$ , with a simpler, tractable distribution. Variational formulations are then used to optimize the parameters of this simpler distribution. CSGM-type methods: The works in this category use backpropagation to change the initial noise of the deterministic diffusion sampler, essentially optimizing over a latent space for the diffusion model. Asymptotically Exact Methods: These methods aim to sample from the true posterior distribution. This is typically achieved by constructing Markov chains (MCMC) or by propagating particles through a sequence of distributions (SMC) to obtain samples that approximate the posterior. Further categorization is based on being able to address non-linear problems, blind formulations (un known forward model), noise handling, pixel/latent space operation, text-conditioning, and the type of optimization technique used (gradient-based, projection, etc.). Code availability is also indicated.

The same inverse problem can appear across vastly different scientific fields. To illustrate this point, we can take the inpainting case as an example. In Computer Vision, inpainting can be useful for applications such as object removal or object replacement [52, 14, 53]. In the proteins domain, inpainting can be useful for protein engineering, e.g. by mutating certain aminoacids of the protein sequence to achieve better thermodynamical properties [54, 55, 56, 57]. MRI acceleration is also an inpainting problem but in the Fourier domain [58, 59, 60, 61, 62]. Particularly, for each coil measurement $y _ { i }$ within the multi-coil setting, we have $A _ { i } = P F S _ { i } ,$ where $P$ is the masking operator, $F$ is the 2D discrete Fourier transform, and $S _ { i }$ denotes the element-wise sensitivity value. For single-coil, $S _ { i }$ is the identity matrix Lustig et al. [63]. Similarly, CT can be considered an inpainting problem in the Radon-transformed domain $A = P R$ , where R is the Radon transform [64, 65, 66]. Depending on the circumstances such as sparse-view or limited-angle, the pattern of the masking operator $\breve { P }$ differs Kak and Slaney [67]. Finally, in the audio domain, the bandwidth extension problem, i.e. the task of recovering high-frequency content from an observed signal, is another example of inpainting in the spectrogram domain) Dietz et al. [68].

Inpainting is just one of many useful linear inverse problems in scientific applications and there are plenty of other important examples to consider. Cryo-EM Dubochet et al. [69] is a blind inverse problem that is defined by $A = C S R ,$ where $C$ is a blur kernel and $S$ is a shifting matrix, i.e. additional (unknown) shift and blur is applied to the projections. Deconvolution appears in several applications such as super-resolution [70, 71] of images and removing reverberant corruption [72] in audio signals.

![](images/04e39525dc78135691eea3852ef9aada6e1651f595c11948f375fecaa2b0483a.jpg)  
Figure 1: Approximations for the measurements score proposed by different methods.

There are many interesting non-linear inverse problems too, i.e. where  is a nonlinear operator. Example 1.5 (Phase Retrieval Fienup [73]). Phase retrieval considers the nonlinear operator $\mathcal { A } ( X ) : = | F X |$ , where the measurement contains only the magnitude of the Fourier signal. Example 1.6 (Compression Removal). Here ${ \mathcal { A } } ( X ; \alpha )$ represents a (non-linear) compression operator (e.g., JPEG) whose strength is controlled by the parameter α.

A famous non-linear inverse problem is the problem of imaging a black hole, where the relationship between the image to be reconstructed and the interferometric measurement can be considered as a sparse and noisy Fourier phase retrieval problem [74].

## 1.2 Recovery types

One common characteristic of these problems is that information is lost and perfect recovery is impossible [75], i.e. they are ill-posed. Hence, the type of “recovery” we are looking for should be carefully defined [76]. For instance, one might be looking for the point that maximizes the posterior distribution $p ( { \pmb x } | { \pmb y } )$ [77, 78]. Often, the Maximum a posteriori (MAP) estimation coincides with the Minimum Mean Squared Error Estimator, i.e. the conditional expectation <sup>E</sup>[x y] [79, 80]. MMSE estimation attempts to minimize distortion of the unknown signal $^ { \mathbf { \delta x } , }$ but often lead to unrealistic recoveries. A different approach is to sample from the full posterior distribution, $p ( { \pmb x } | { \pmb y } )$ . Posterior sampling accounts for the uncertainty of the estimation, and typically produces samples that have higher perception quality. Blau and Michaeli [81] show that, in general, it is impossible to find a sample that maximizes perception and minimizes distortion at the same time. Yet, posterior sampling is nearly optimal [1] in terms of distortion error.

## 1.3 Approaches for Solving Inverse Problems

Inverse problems have a rich history, with approaches evolving significantly over the decades Ribes and Schmitt [82], Barrett and Myers [83]. While a comprehensive review is beyond the scope of this survey, we highlight key trends to provide context. Early approaches, prevalent in the 2000s, often framed inverse problems as optimization tasks Daubechies et al. [84], Candès et al. [85], Donoho [86], Figueiredo and Nowak [87], Daubechies et al. [84], Hale et al. [88], Shlezinger et al. [89]. These methods sought to balance data fidelity with regularization terms that encouraged desired solution properties like smoothness Rudin et al. [90], Beck and Teboulle [91] or sparsity in specific representations (e.g., wavelets, dictionaries) Figueiredo and Nowak [87], Daubechies et al. [84], Candès et al. [85], Donoho [86], Hale et al. [88].

The advent of deep learning brought a paradigm shift Ongie et al. [92]. Researchers began leveraging large paired datasets to directly learn mappings from measurements to clean signals using neural networks Dong et al. [93], Lim et al. [94], Tao et al. [95], Chen et al. [96], Zamir et al. [97], Chen et al. [98], Tu et al. [99], Zamir et al. [100]. These approaches focus on minimizing some reconstruction loss during training, with various techniques employed to penalize distortions, and optimize for specific application goals (e.g., perceptual quality Isola et al. [101], Kupyn et al. [102]). Traditional point estimates aim to recover a single reconstruction by for example minimizing the average reconstruction error (i.e., MMSE) or by finding the most probable reconstruction through Maximum a Posteriori estimate (MAP), i.e., finding the x that maximizes $p ( { \pmb x } | { \pmb y } )$ ). While powerful, this approach can suffer from “regression to the mean”, where the network predicts an average solution that may lack important details or even be outside the desired solution space Blau and Michaeli [81], Delbracio and Milanfar [103]. In fact, learning a mapping to minimize a certain distortion metric will lead, in the best case, to an average of all the plausible reconstructions (e.g., when using a L2 reconstruction loss, the best-case solution will be the posterior mean). This reconstruction might not be in the target space (e.g., a blurry image being the average of all plausible reconstructions) Blau and Michaeli [81].

Recent research has revealed a striking connection between denoising algorithms and inverse problems. Powerful denoisers, often based on deep learning, implicitly encode valuable information about natural signals. By integrating these denoisers into optimization frameworks, we can harness their learned priors to achieve exceptional results in a variety of inverse problems Venkatakrishnan et al. [104], Sreehari et al. [105], Chan et al. [106], Romano et al. [107], Cohen et al. [108], Kad khodaie and Simoncelli [109], Kamilov et al. [110], Milanfar and Delbracio [111]. This approach bridges the gap between traditional regularization methods and modern denoising techniques, offering a promising new paradigm for solving these challenging tasks.

An alternative perspective views inverse problems through the lens of Bayesian inference. Given measurements y, the goal becomes generating plausible reconstructions by sampling from the posterior distribution $p ( \boldsymbol { X } | \boldsymbol { Y } = \boldsymbol { y } )$ – the distribution of possible signals x given the observed measurements y.

In this survey we explore a specific class of methods that utilize diffusion models as priors for p<sub>X</sub> , and then try to generate plausible reconstructions (e.g., by sampling from the posterior). While other approaches exist, such as directly learning conditional diffusion models or flows for specific inverse problems Li et al. [112], Saharia et al. [71, 113], Whang et al. [114], Luo et al. [115, 116], Albergo and Vanden-Eijnden [117], Albergo et al. [118], Lipman et al. [119], Liu et al. [120, 121], Shi et al. [122], these often require retraining for each new application. In contrast, the methods covered in this survey offer a more general framework applicable to arbitrary inverse problems without retraining or fine-tuning.

Unsupervised methods. We refer as unsupervised methods to those that focus on characterizing the distribution of target signals, $p _ { \mathbf { { X } } } .$ , and applying this knowledge during the inversion process. Since they don’t rely on paired data, they can be flexibly applied to different inverse problems using the same prior knowledge.

Unsupervised methods can be used to maximize the likelihood of $p ( { \pmb x } | { \pmb y } )$ or to sample from this distribution. Algorithmically, to solve the former problem we typically use (some variation of) Gradient Descent and to solve the latter (some variation of) Monte Carlo Simulation $( \mathrm { e . g . }$ , Langevin Dynamics). Either way, one typically requires to compute the gradient of the conditional log-likelihood, $\mathrm { i . e . , } \nabla _ { x } \log p ( \pmb { x } | \pmb { y } )$

A simple application of Bayes Rule reveals that:

$$
\underbrace { \nabla _ { \pmb { x } } \log p ( \pmb { x } | \pmb { y } ) } _ { \mathrm { c o n d i t i o n a l ~ s c o r e } } = \underbrace { \nabla _ { \pmb { x } } \log p ( \pmb { x } ) } _ { \mathrm { u n c o n d i t i o n a l ~ s c o r e } } + \underbrace { \nabla _ { \pmb { x } } \log p ( \pmb { y } | \pmb { x } ) } _ { \mathrm { m e a s u r e m e n t s ~ m a t c h i n g ~ t e r m } } .\tag{1.2}
$$

The last term typically has a closed-form expression, e.g. for the linear case, we have that: $\begin{array} { r } { \nabla _ { \pmb { x } } \log p ( \pmb { y } | \pmb { x } ) = \frac { \pmb { y } - \dot { A } \pmb { x } } { \sigma _ { \pmb { y } } ^ { 2 } } } \end{array}$ . However, the first term, known as the score function, might be hard to estimate when the data lie on low-dimensional manifolds. The problem arises from the fact that we do not get observations outside of the manifold and hence the vector-field estimation is inaccurate in these regions.

One way to sidestep this issue is by using a “smoothed” version of the score function, representing the score function of noisy data that will be supported everywhere. The central idea behind diffusion generative models is to learn score functions that correspond to different levels of smoothing. Specifically, in diffusion modeling, we attempt to learn the smoothed score functions, $\nabla _ { \pmb { x } _ { t } } \log \bar { p _ { t } } ( \pmb { x } _ { t } )$ where $\begin{array} { r } { \boldsymbol X _ { t } = \boldsymbol X _ { 0 } + \sigma _ { t } \boldsymbol Z , \quad \boldsymbol { \breve { Z } } \sim \mathcal N ( \mathbf { 0 } , \boldsymbol { \tilde { I } } ) } \end{array}$ , for different noise levels t. During sampling, we progressively move from more smoothed vector fields to the true score function. At the very end, the score function corresponding to the data distribution is only queried at points for which the estimation is accurate because of the warm-start effect of the sampling method.

Even though estimating the unconditional score becomes easier (because of the smoothing), the measurement matching term becomes time dependent and loses its closed form expression. Indeed, the likelihood of the measurements is given by the intractable integral:

$$
p _ { t } ( \pmb { y } | \pmb { x } _ { t } ) = \int p ( \pmb { y } | \pmb { x } _ { 0 } ) p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } ) \mathrm { d } \pmb { x } _ { 0 } .\tag{1.3}
$$

The computational challenge that emerges from the intractability of the conditional likelihood has led to the proposal of numerous approaches to use diffusion models to solve inverse problems [1, 4, 5, 2, 3, 9, 8, 11, 12, 13, 6, 123, 124, 16, 18, 19, 24, 25, 28, 29, 30, 20, 21, 125, 126]. The sheer number of the proposed methods, but also the different perspectives under which these methods have been developed, make it hard for both newcomers and experts in the field to understand the connections between them and the unifying underlying principles. This work attempts to explain, taxonomize and relate prominent methods in the field of using diffusion models for inverse problems. Our list of methods is by no means exhaustive. The goal of this manuscript is not to list all the methods that have been proposed but to review some representative methods of different approaches and present them under a unifying framework. We believe this survey will be useful as a reference point for people interested in this field.

## 2 Background

## 2.1 Diffusion Processes

Forward and Reverse Processes. The idea of a diffusion model is to transform a a simple distribution (e.g., normal distribution) into the unknown data distribution $p _ { 0 } ( { \pmb x } )$ , that we don’t know explicitly but we have access to some of its samples. The first step is to define a corruption process. The popular Denoising Diffusion Probabilistic Models (DDPM) Ho et al. [127], Song and Ermon [128], adopt a discrete time Markovian process to transform the input Normal distribution into the target one by incrementally adding Gaussian noise. More generally, the corruption processes of interest can be generalized to continuous time by a stochastic differential eqaution (SDE) [2]:

$$
\mathrm d \pmb { x } _ { t } = \underbrace { \pmb { f } ( \pmb { x } _ { t } , t ) } _ { \mathrm { d r i f t ~ c o e f f . } } \mathrm d t + \underbrace { \pmb { g } ( t ) } _ { \mathrm { d i f f u s i o n ~ c o e f f . } } \mathrm d \pmb { W } _ { t } ,\tag{2.1}
$$

with $\pmb { x } _ { 0 } \sim p _ { 0 } , \pmb { x } _ { 0 } \in \mathbb { R } ^ { n }$ , and $\mathbf { } W _ { t }$ denotes a Wiener process $( \mathrm { i . e . }$ , Brownian motion). This SDE gradually transforms the data distribution into Gaussian noise. We denote with $p _ { t }$ the distribution that arises by running this dynamical system up to time t.

A remarkable result by Anderson [129] shows that we can sample from $p _ { 0 }$ by running backwards in time the reverse SDE:

$$
\mathrm { d } \pmb { x } _ { t } = \left( f ( \pmb { x } _ { t } , t ) - g ^ { 2 } ( t ) \underbrace { \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } ) } _ { \mathrm { s c o r e } } \right) \mathrm { d } t + g ( t ) \mathrm { d } \pmb { W } _ { t } ,\tag{2.2}
$$

initialized at $\mathbf { \boldsymbol { x } } _ { T } \sim p _ { T }$ . For sufficiently large T and for linear drift functions $f ( \cdot , \cdot )$ , the latter distribution approaches a Gaussian distribution with known parameters that can be used for initializing the process. Hence, the remaining goal becomes to estimate the score function $\nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } )$

Probability Flow ODE. Song et al. [2], Maoutsa et al. [130] observe that the (deterministic) differential equation:

$$
\frac { \mathrm { d } \pmb { x } _ { t } } { \mathrm { d } t } = \left( \pmb { f } ( \pmb { x } _ { t } , t ) - \frac { g ^ { 2 } ( t ) } { 2 } \underbrace { \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } ) } _ { \mathrm { s c o r e } } \right)\tag{2.3}
$$

corresponds to the same Fokker-Planck equations as the SDE of Equation 2.2. An implication of this is we can use the deterministic sampling scheme of Equation 2.3. Any well-built numerical ODE solver can be used to solve Equation 2.3, such as the Euler solver:

$$
\pmb { x } _ { t - \Delta t } = \pmb { x } _ { t } + \Delta t \left( \pmb { f } ( \pmb { x } _ { t } , t ) - \frac { g ^ { 2 } ( t ) } { 2 } \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } ) \right) .\tag{2.4}
$$

SDE variants: Variance Exploding and Variance Preserving Processes. The drift coefficients, $f ( \pmb { x } _ { t } , t )$ , and the diffusion coefficients $g ( t )$ are design choices. One popular choice, known as the Variance Exploding SDE, is setting $\pmb { f } ( \pmb { x } _ { t } , t ) = \mathbf { 0 }$ and $\begin{array} { r } { g ( t ) = \sqrt { \frac { \mathrm { d } \sigma _ { t } ^ { 2 } } { \mathrm { d } t } } } \end{array}$ for some variance scheduling $\{ \sigma _ { t } \} _ { t = 0 } ^ { T }$ . Under these choices, the marginal distribution at time t of the forward process of Equation 2.1 can be alternatively described as:

$$
\mathbf { } X _ { t } = X _ { 0 } + \sigma _ { t } \mathbf { Z } , \quad \mathbf { } X _ { 0 } \sim p ( \mathbf { X } _ { 0 } ) , \quad \mathbf { } Z \sim \mathcal { N } ( \mathbf { 0 } , I _ { n } ) .\tag{2.5}
$$

The typical noise scheduling for this SDE is $\sigma _ { t } = \sqrt { t }$ (that corresponds to $g ( t ) = 1 )$ .

Another popular choice is to set the drift function to be $\begin{array} { r } { \pmb { f } ( \pmb { x } _ { t } , t ) = - \pmb { x } _ { t } , } \end{array}$ , which is known as the Variance Preserving (VP) SDE. A famous process in the VP SDE family is the Ornstein–Uhlenbeck (OU) process:

$$
\mathrm { d } \pmb { x } _ { t } = - \pmb { x } _ { t } \mathrm { d } t + \sqrt { 2 } \mathrm { d } \pmb { W } _ { t } ,\tag{2.6}
$$

which gives:

$$
\begin{array} { r } { X _ { t } = \exp ( - t ) X _ { 0 } + \sqrt { 1 - \exp ( - 2 t ) } Z , \quad Z \sim \mathcal { N } ( \mathbf { 0 } , I _ { n } ) . } \end{array}\tag{2.7}
$$

The VP SDE [127] takes a more general form:

$$
\begin{array} { r } { \pmb { X } _ { t } = \sqrt { \alpha _ { t } } \pmb { X } _ { 0 } + ( 1 - \alpha _ { t } ) \pmb { Z } , \quad \pmb { X } _ { 0 } \sim p ( \pmb { X } _ { 0 } ) , \quad \pmb { Z } \sim \mathcal { N } ( \mathbf { 0 } , I _ { n } ) . } \end{array}\tag{2.8}
$$

With reparametrization and the Euler solver, this leads to an efficient solution to Equation 2.3, known as DDIM [131]:

$$
x _ { t - 1 } = \sqrt { \alpha _ { t - 1 } } \underbrace { \left( \frac { x _ { t } + ( 1 - \alpha _ { t } ) \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } ) } { \sqrt { \alpha _ { t } } } \right) } _ { = : \widehat { x } _ { 0 } = \mathsf { p r d i c t e d } \ x _ { 0 } } + \sqrt { 1 - \alpha _ { t - 1 } - \sigma _ { t } ^ { 2 } } \underbrace { \left( - \sqrt { 1 - \alpha _ { t } } \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } ) \right) } _ { \mathrm { d i r e c t i o n ~ t o w a r d } \ x _ { t } } .\tag{2.9}
$$

For convenience, in the rest of the paper, this update will be written as: $\pmb { x } _ { t - 1 } \gets$ Unconditiona $\mathrm { \vert D D I M } ( \widehat { \pmb x } _ { 0 } , \pmb x _ { t } )$ .

## 2.2 Tweedie’s Formula and Denoising Score Matching

In what follows, we will discuss how one can learn the score function $\nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } )$ that appears in Equation 2.17. We will focus on the VE SDE, since the mathematical calculations are simpler.

Tweedie’s formula [132] is a famous result in statistics that shows that for an additive Gaussian corruption, $\mathbf { } X _ { t } = \bar { X _ { 0 } } + \sigma _ { t } Z , Z \sim \mathcal { N } ( \mathbf { 0 } , I _ { n } )$ , it holds that:

$$
\nabla _ { \pmb { x } _ { t } } \log { p _ { t } ( \pmb { x } _ { t } ) } = \frac { \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } ] - \pmb { x } _ { t } } { \sigma _ { t } ^ { 2 } } .\tag{2.10}
$$

The formal statement and a self-contained proof can be found in the Appendix, Lemma A.2.

Tweedie’s formula gives us a way to derive the unconditional score function needed in Equation 2.17, by optimizing for the conditional expectation, $\mathbb { E } [ X _ { 0 } | X _ { t } ~ = ~ { \pmb x } _ { t } ]$ . The conditional expectation $\mathbb { E } [ X _ { 0 } | X _ { t } = \mathbf { \bar { x } } _ { t } ]$ , is nothing more than the minimum mean square error estimator (MMSE) of the clean image given the noisy observation $\mathbf { \Delta } \mathbf { x } _ { t } .$ , that is a denoiser.

In practice, we don’t know analytically this denoiser but we can parametrize it using a neural network $h _ { \theta } ( { \pmb x } _ { t } )$ and learn it in a supervised way by minimizing the following objective:

$$
J _ { \mathrm { D S M } } ( \pmb \theta ) = \mathbb { E } _ { \pmb { x } _ { 0 } , \pmb { x } _ { t } } \left[ | | \pmb { h } _ { \theta } ( \pmb { x } _ { t } ) - \pmb { x } _ { 0 } | | ^ { 2 } \right] .\tag{2.11}
$$

Assuming a rich enough family $\mathbf { \Theta } _ { \Theta }$ , the minimizer of Equation 2.11 is ${ \pmb h } _ { \theta } ( { \pmb x } _ { t } ) = \mathbb { E } [ { \pmb x } _ { 0 } | { \pmb X } _ { t } = { \pmb x } _ { t } ]$ (see Lemma A.1) and the score in Equation 2.10 is approximated as $\left( h _ { \theta } ( { \pmb x } _ { t } ) - { \pmb x } _ { t } \right) / \sigma _ { t } ^ { 2 }$ . Note that for each $\sigma _ { t }$ we would need to learn a different denoiser (since the noise strength is different), or alternative the neural network $h _ { \theta }$ should also take as input the value of t or $\sigma _ { t } .$ . Diffusion models are trained following the later paradigm, i.e. the same neural network approximates the optimal denoisers at all noise levels by conditioning it on the noise level through t.

Interestingly, Vincent [133] independently discovered that the score function can be learned by minimizing an $l _ { 2 }$ objective, similar to Equation 2.11. The formal statement and a self-contained proof of this alternative derivation is included in the Appendix, Theorem A.3.

## 2.3 Latent Diffusion Processes

For high-dimensional distributions, diffusion models training (see Equation 2.11) and sampling (see Equation 2.3) require massive computational resources. To make the training and sampling more efficient, the authors of Stable Diffusion Rombach et al. [134] propose performing the diffusion in the latent space of a pre-trained powerful autoencoder. Specifically, given an encoder Enc : $\mathbb { R } ^ { n } \to$ $\mathbb { R } ^ { k }$ and a decoder Dec : $\mathbb { R } ^ { k } \to \bar { \mathbb { R } } ^ { n }$ , one can create noisy samples:

$$
\mathbf { } X _ { t } ^ { \mathrm { E } } = \underbrace { \mathbf { } X _ { 0 } ^ { \mathrm { E } } } _ { \mathrm { E n c } ( \mathbf { } X _ { 0 } ) } + \sigma _ { t } \boldsymbol { Z } , \quad \boldsymbol { Z } \sim \mathcal { N } ( \mathbf { 0 } , \boldsymbol { I } _ { k } ) ,\tag{2.12}
$$

and train a denoiser network in the latent space. At inference time, one starts with pure noise, samples a clean latent $\tilde { \mathbf { x } } _ { 0 } ^ { \mathrm { E } }$ by running the reverse process, and outputs $\pmb { x } _ { 0 } = \mathrm { D e c } ( \tilde { \pmb { x } } _ { 0 } ^ { \mathrm { E } } ) ^ { \hat { 1 } }$ . Solving

inverse problems with Latent Diffusion models requires special treatment. We discuss the reasons and approaches in this space in Section 3.5.

## 2.4 Conditional Sampling

## 2.4.1 Stochastic Samplers for Inverse Problems

The goal in inverse problems is to sample from $p _ { 0 } ( \cdot | \pmb { y } )$ assuming a corruption model ${ \bf Y _ { \alpha } } =$ $\bar { \mathcal { A } } ( \bar { X _ { 0 } } ) + \sigma _ { y } Z , Z \sim \mathcal { N } ( \mathbf { 0 } , I _ { m } )$ . We can easily adapt the original unconditional formulation given by Equation 2.2 into a conditional one to generate samples from $p _ { 0 } ( \cdot | \pmb { y } )$ . Specifically, the associated reverse process is given by the stochastic dynamical system [135]:

$$
\mathrm { d } \mathbf { { \boldsymbol { x } } } _ { t } = \left( f ( \mathbf { { \boldsymbol { x } } } _ { t } , t ) - g ^ { 2 } ( t ) \underbrace { \nabla _ { \mathbf { { \boldsymbol { x } } } _ { t } } \log p _ { t } ( \mathbf { { \boldsymbol { x } } } _ { t } | \mathbf { \nabla } y ) } _ { \mathrm { c o n d i t i o n a l s c o r e } } \right) \mathrm { d } t + g ( t ) \mathrm { d } { \mathbf { W } } _ { t } ,\tag{2.13}
$$

initialized at $\pmb { x } _ { T } \sim p _ { T } ( \cdot | \pmb { y } )$ . For sufficiently large T and for linear drift functions $f ( \cdot , \cdot )$ , the distribution $p _ { T } ( \cdot | \pmb { y } )$ is a Gaussian distribution with parameters independent of y. In the conditional case, the goal becomes to estimate the score function $\nabla _ { \pmb { x } _ { t } }$ log $p _ { t } ( \pmb { x } _ { t } | \pmb { y } )$

## 2.4.2 Deterministic Samplers for Inverse Problems

It is worth noting that (as in the unconditional setting) it is possible to derive deterministic sampling algorithms as well. Particularly, one can use the following dynamical system [2, 135]:

$$
\frac { \mathrm { d } \pmb { x } _ { t } } { \mathrm { d } t } = - \frac { g ^ { 2 } ( t ) } { 2 } \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } | \pmb { y } ) .\tag{2.14}
$$

initialized at $p _ { T } ( \cdot | \boldsymbol { y } )$ to get sample from the conditional distribution $p _ { 0 } ( \cdot | \pmb { y } )$ . Once again, to run this discrete dynamical system, one needs to know the conditional score, $\nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } | \pmb { y } )$

## 2.4.3 Conditional Diffusion Models

Similarly to the unconditional setting, one can directly train a network to approximate the conditional score, $\nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } | \pmb { y } )$ . A generalization of Tweedie’s formula gives that:

$$
\nabla \log p _ { t } ( \pmb { x } _ { t } | \pmb { y } ) = \frac { \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } , \pmb { Y } = \pmb { y } ] - \pmb { x } _ { t } } { \sigma _ { t } ^ { 2 } } .\tag{2.15}
$$

Hence, one can train a network using a generalized version of the Denoising Score Matching,

$$
J _ { \mathrm { c o n d , D S M } } ( \pmb { \theta } ) = \mathbb { E } _ { \pmb { x } _ { 0 } , \pmb { x } _ { t } , \pmb { y } } \left[ \left| \left| \pmb { h } _ { \theta } ( \pmb { x } _ { t } , \pmb { y } ) - \pmb { x } _ { 0 } \right| \right| ^ { 2 } \right] ,\tag{2.16}
$$

and then use it in Equation 2.15 in place of the conditional expectation. The main issue with this approach is that the forward model (degradation operator) needs to be known at training time. If the corruption model ${ \boldsymbol { \mathcal { A } } } ( X )$ changes, then the model needs to be retrained. Further, with this approach we need to train new models and we cannot directly leverage powerful unconditional models that are already available. The focus of this work is on methods that use pre-trained unconditional diffusion models to solve inverse problems, without further training.

## 2.4.4 Using pre-trained diffusion models to solve inverse problems

As we showed earlier, the conditional score can be decomposed using Bayes Rule into:

$$
\nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } | \pmb { y } ) = \underbrace { \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } ) } _ { \mathrm { s c o r e } } + \underbrace { \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { x } _ { t } ) } _ { \mathrm { m e a s u r e m e n t s ~ m a t c h i n g ~ t e r m } } .\tag{2.17}
$$

that is, the (smoothed) score function, and the measurements matching term that is given by the inverse problem we are interested in solving. Applying this to equation 2.13, we get that:

$$
\mathrm { d } x _ { t } = \left( f ( x _ { t } , t ) - g ^ { 2 } ( t ) \left( \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } ) + \nabla \log p ( y | x _ { t } ) \right) \right) \mathrm { d } t + g ( t ) \mathrm { d } W _ { t } .\tag{2.18}
$$

Similarly, one can use the deterministic process:

$$
\mathrm { d } \mathbf { \boldsymbol { x } } _ { t } = \left( f ( \mathbf { \boldsymbol { x } } _ { t } , t ) - \frac { 1 } { 2 } g ^ { 2 } ( t ) \left( \nabla _ { \mathbf { \boldsymbol { x } } _ { t } } \log p _ { t } ( \mathbf { \boldsymbol { x } } _ { t } ) + \nabla \log p ( \mathbf { \boldsymbol { y } } | \mathbf { \boldsymbol { x } } _ { t } ) \right) \right) \mathrm { d } t .\tag{2.19}
$$

We have already discussed how to train a neural network to approximate $\nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } )$ (using Tweedie’s Formula / Denoising Score Matching). However, here we further need access to the term $\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { x } _ { t } )$ . The likelihood of the measurements is given by the intractable integral:

$$
p _ { t } ( \pmb { y } | \pmb { x } _ { t } ) = \int p ( \pmb { y } | \pmb { x } _ { 0 } ) p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } ) \mathrm { d } \pmb { x } _ { 0 } .\tag{2.20}
$$

Gupta et. al [136] prove that there are instances of the posterior sampling problem for which every algorithm takes superpolynomial time, even though unconditional sampling is provably fast. Hence, diffusion models excel at performing unconditional sampling but are hard to use as priors for solving inverse problems because of the time dependence in the measurements matching term. Since the very introduction of diffusion models, there has been a plethora of methods proposed to use them to solve inverse problems without retraining. This survey serves as a reference point for different techniques that have been developed in this space.

## 2.4.5 Ambient Diffusion: Learning to solve inverse problems using only measurements

The goal of the unsupervised learning approach for solving inverse problems (Section 2.4.4) is to use a prior $p ( { \pmb x } )$ to approximate the measurements matching term, log $p _ { t } ( \pmb { y } | \pmb { x } _ { t } )$ . However, in certain applications, it is expensive or even impossible to get data from (and hence learn) $p ( { \pmb x } )$ in the first place. For instance, in MRI the quality of the data is proportionate to the time spent under the scanner [59] and it is infeasible to acquire full measurements from black holes [74]. This creates a chicken-egg problem: we need access to $p ( { \pmb x } )$ to solve inverse problems and we do not have access to samples from $p ( { \pmb x } )$ unless we can solve inverse problems. In certain scenarios, it is possible to break this seemingly impossible cycle.

Ambient Diffusion Daras et al. [137] was one of the first frameworks to train diffusion models with linearly corrupted data. The key concept behind the Ambient Diffusion framework is the idea of further corruption. Specifically, the given measurements get further corrupted and the model is trained to predict a clean image by using the measurements before further corruption for validation. Ambient DPS [49] shows that priors learned from corrupted data can even outperform (in terms of usefulness for inverse problems), at the high-corruption regime, priors learned from clean data. Ambient Diffusion was extended to handle additive Gaussian Noise in the measurements. The paper Consistent Diffusion Meets Tweedie Daras et al. [138] was the first diffusion-based framework to provide guarantees for sampling from the distribution of interest, given only access to noisy data. This paper extends the idea of further corruption to the noisy case and proposes a novel consistency loss Daras et al. [139] to learn the score function for diffusion times that correspond to noise levels below the level of the noise in the dataset.

Both Ambient Diffusion and Consistent Diffusion Meets Tweedie have connections to deep ideas from the literature in learning restoration models from corrupted data, such as Stein’s Unbiased Risk Estimate (SURE) Eldar [140], Stein [141] and Noise2X Lehtinen et al. [142], Krull et al. [143], Batson and Royer [144]. These connections are also leveraged by alternative frameworks to Ambient Diffusion, as in [8, 58]. A different approach for learning diffusion models from measurements is based on the Expectation-Maximization (EM) algorithm [145, 6, 146]. The convergence of these methods to the true distribution depends on the convergence of the EM algorithm, which might get stuck in a local minimum.

In this survey, we focus on the setting where a pre-trained prior $p ( { \pmb x } )$ is available, regardless of whether it was learned from clean or corrupted data.

## 3 Reconstruction Algorithms

We summarize all the methods analyzed in this work in Table 1. The methods have been taxonomized based on the approach they use to solve the inverse problem (explicit score approximations, variational methods, CSGM-type methods and asymptotically exact methods), the type of inverse problems they can solve and the optimization techniques used to solve the problem at hand (gradient descent, sampling, projections, parameter optimization). Additionally, we provide links to the official code repositories associated with the papers included in this survey. Please note that we have not conducted a review or evaluation of these codebases to verify their consistency with the corresponding papers. These links are included for informational purposes only.

Taxonomy based on the type of the reconstruction algorithm. We identified four families of methods. Explicit Approximations for Measurement Matching: These methods approximate the measurement matching score, $\nabla \log p _ { t } ( \pmb { y } | \pmb { x } _ { t } )$ , with a closed-form expression. Variational Inference: These methods approximate the true posterior distribution, $p ( { \pmb x } | { \pmb y } )$ , with a simpler, tractable distribution. Variational formulations are then used to optimize the parameters of this simpler distribution. CSGM-type methods: The works in this category use backpropagation to change the initial noise of the deterministic diffusion sampler, essentially optimizing over a latent space for the diffusion model. Asymptotically Exact Methods: These methods aim to sample from the true posterior distribution. This is typically achieved by constructing Markov chains (MCMC) or by propagating particles through a sequence of distributions (SMC) to obtain samples that approximate the posterior. Methods that do not fall into any of these categories are classified as Others.

Taxonomy based on the type of optimization techniques used. The objective of all methods is to explain the measurements. The measurement consistency can be enforced with different opti mization techniques, e.g. through gradients (Grad), projections (Proj), sampling (Samp), or other optimization techniques (Opt). Methods that belong to the Grad-type take a single gradient step (ei ther it be deterministic or stochastic) to $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ to enforce measurement consistency. Proj-type projects x<sub>t</sub> or $\mathbb { E } [ X _ { 0 } | X _ { t } ~ = ~ { \pmb x } _ { t } ]$ to the measurement subspace. Samp-type samples the next particles by defining a proposal distribution, and propagates multiple chains of particles to solve the problem. Opt-type either defines and solves an optimization problem for every timestep, or defines a global optimization problem that encompasses all timesteps. When the method belongs to more than one type, we seperate them with /. Note that the categorization of different “types” is subjective, and more often than not, the category that the method belongs to may be interpreted in multiple ways. For instance, a projection step is also a gradient descent step with a specific step size.

Taxonomy based on the type of the inverse problem. Based on the linearity of the corruption operator , the inverse problems can be classified as linear or nonlinear. The inverse problems can be further categorized based on whether there is noise in the measurements. Additionally, they are classified as non-blind or blind depending on whether full information about is available. In blind problems, the degradation operator (e.g., convolution kernel, inpainting kernel) is known, while its coefficients are unknown but parametrized. For example, we might know that we have measurements with additive Gaussian noise, but the variance of the noise might be unknown. Finally, in certain inverse problems, there is additional text-conditioning. Such inverse problems are typically solved with text-to-image latent diffusion models [134].

## 3.1 Explicit Approximations for the Measurements Matching Term

The first family of reconstruction algorithms we identify is the one were explicit approximations for the measurements matching term, $\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { X } _ { t } = \pmb { x } _ { t } )$ , are made. It is important to underline that these approximations are not always clearly stated in the works that propose them, which makes it hard to understand the differences and commonalities between different methods. In what follows, we attempt to elucidate the different approximations that are being made and present different works under a common framework. To provide some insights, we often provide the explicit approximation formulas for the measurements matching term in the setting of linear inverse problems. In general, it follows the template form:

$$
\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { X } _ { t } = \pmb { x } _ { t } ) \approx - \frac { \pmb { \mathcal { L } } _ { t } \pmb { \mathcal { M } } _ { t } } { \mathcal { G } _ { t } } .\tag{3.1}
$$

Here,

$\mathcal { M } _ { t }$ represents the error vector measuring the discrepancy between the observation y and the estimated restored vector; for example, in Score $\mathrm { A L D } \left[ 1 \right] , \mathcal { M } _ { t } = \pmb { y } - A \pmb { x } _ { t }$

$\mathcal { L } _ { t }$ denotes a matrix that projects the error vector $\mathcal { M } _ { t }$ from $\mathbb { R } ^ { m }$ back into an appropriate space in $\mathbb { R } ^ { n }$ ; for instance, in Score $\mathrm { A L D } , \mathcal { L } _ { t } = A ^ { \top }$

• <sub>t</sub> is the re-scaling scalar for the guidance vector $\mathcal { L } _ { t } \mathcal { M } _ { t }$ ; for example, in Score ALD, $\begin{array} { r } { \mathcal { G } _ { t } = \sigma _ { y } ^ { 2 } + \gamma _ { t } ^ { 2 } } \end{array}$ with a hyperparameter $\gamma _ { t }$

In Figure 1, we summarize the approximation-based methods in this section using the template above. We use to omit the guidance strength terms $\mathcal { G } _ { t }$

## 3.1.0 Sampling from a Denoiser Kadkhodaie and Simoncelli [30]

Kadkhodaie and Simoncelli [30] introduce a method for solving linear inverse problems by using the implicit prior knowledge captured by a pre-trained denoiser on multiple noise levels. The method is anchored on Tweedie’s formula that connects the least-squares solution for Gaussian denoising to the gradient of the log-density of noisy images given in Equation 2.10

$$
\pmb { \hat { x } } ( \pmb { y } ) = \pmb { y } + \sigma ^ { 2 } \nabla _ { \pmb { y } } \log p ( \pmb { y } ) ,\tag{3.2}
$$

where ${ \pmb y } = { \pmb x } + { \pmb n } , { \pmb n } \sim \mathcal { N } ( { \pmb 0 } , \sigma ^ { 2 } I _ { n } )$

By interpreting the denoiser’s output as an approximation of this gradient, the authors develop a stochastic gradient ascent algorithm to generate high-probability samples from the implicit prior

$$
{ \pmb y } _ { t } = { \pmb y } _ { t - 1 } + h _ { t } { \pmb r } ( { \pmb y } _ { t - 1 } ) + \epsilon _ { t } { \pmb z } _ { t } ,\tag{3.3}
$$

where $\pmb { r } ( \pmb { y } ) = \hat { \pmb { x } } ( \pmb { y } ) - \pmb { y }$ is the denoiser residual, $h _ { t }$ is a step size (parameter), and $\epsilon _ { t }$ controls the amount of newly introduced Gaussian noise ${ \boldsymbol { z } } _ { t }$

To solve linear inverse problems such as deblurring, super-resolution, and compressive sensing, the generative method is extended to handle constrained sampling. Given a set of linear measurements $\bar { \mathbf { x } } _ { c } = M ^ { \top }$ x of an image x, where M is a low-rank measurement matrix, the goal is to reconstruct the original image by utilizing the following gradient:

$$
\nabla _ { \boldsymbol { y } } \log { p ( \boldsymbol { y } | \boldsymbol { x } _ { c } ) } = ( I - M M ^ { \top } ) \boldsymbol { r } ( \boldsymbol { y } ) + M ( \mathbf { x } _ { c } - M ^ { \top } \boldsymbol { y } ) ,\tag{3.4}
$$

This approach is particularly interesting because its mathematical foundation relies solely on Tweedie’s formula, providing a simple yet powerful framework for tackling inverse problems using denoisers.

## 3.1.1 Score ALD [1]

One of the first proposed methods for solving linear inverse problems with diffusion models is the Score-Based Annealed Langevin Dynamics (Score ALD) [1] method. The approximation of this work is that:

$$
\begin{array} { r } { \frac { \mathbf { \Phi } ^ { \mathrm { { q } } } \parallel \mathrm { { f i l f i n g } ^ { \mathrm { { p } } } \boldsymbol { m } \mathbf { a t r i x } } } { \lambda } \qquad \begin{array} { l } { \mathrm { m e a s u r e m e n t s ~ e r r o r } } \\ { \sqrt { \mathbf { \Phi } } } \\ { \nabla _ { { \pmb { x } } _ { t } } \log p ( { \pmb y } | \mathbf { { \cal X } } _ { t } = { \pmb x } _ { t } ) \approx - \frac { { \mathbf { \Phi } } \mathbf { { \cal A } } ^ { \top } \quad ( \pmb { y } - { \pmb A x } _ { t } ) } { \sigma _ { { \pmb { y } } } ^ { 2 } + \gamma _ { t } ^ { 2 } } , } \\ { \mathrm { \qquad ~ \quad ~ \gamma ~ } \mathrm { { \mathrm { g u i d a n c e s t r e n g t h } } } } \end{array} } \end{array}\tag{3.5}
$$

where $\gamma _ { t }$ is a parameter t

It is pretty straightforward to understand what this term is doing. The diffusion process is guided towards the opposite direction of the “lifting” (application of the ${ \bf \bar { A } } ^ { \top }$ operator) of the measurements error, i.e. $\left( { \pmb y } - { \pmb A } { \pmb x } _ { t } \right)$ ), where the denominator controls the guidance strength.

## 3.1.2 Score-SDE [2]

Score-SDE [2] is another one of the first works that discussed solving inverse problems with pretrained diffusion models. For linear inverse problems, the difference between Score-ALD and Score-SDE is that the latter noises the measurements before computing the measurements error. Specifically, for $t : \sigma _ { t } > \sigma _ { y }$ , the approximation becomes:

$$
\begin{array} { r l r } & { } & { \frac { \mathbf { \sigma } _ { \psi _ { t } } ^ { * } | | \mathrm { f i l } \nabla \mathbf { g } ^ { \flat } \mathbin { \mathrm { m a t r i x } } \boldsymbol { \mathrm { \Omega } } \boldsymbol { \mathrm { \Omega } } \boldsymbol { \mathrm { \Omega } } \boldsymbol { \mathrm { \Omega } } \boldsymbol { \mathrm { \Omega } } \boldsymbol { \mathrm { \Omega } } \boldsymbol { \mathrm { \Omega } } \boldsymbol { \mathrm { \Omega } } \boldsymbol { \mathrm { \Omega } } \qquad \mathbf { \sigma } _ { \psi } ^ { y _ { t } } \mathrm { ( \cap 0 i s e d ~ \mathsf { m e a s u r e m e n t s } ) } } { \mathbf { \sigma } _ { \psi _ { t } } ^ { * } } } \\ & { } & { \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \mathbf { X } _ { t } = \pmb { x } _ { t } ) \approx - \mathbf { \sigma } _ { A } ^ { \mathsf { T } } \mathrm { \Omega } \left( \mathbf { \sigma } _ { \psi } + \sigma _ { t } \epsilon \mathbf { \sigma } - A \pmb { x } _ { t } \right) } \end{array}\tag{3.6}
$$

where ǫ is sampled from $\mathcal { N } ( \mathbf { 0 } , I _ { m } )$ . Here, A is an orthogonal matrix, and taking a gradient step with Equation $3 . 6$ yields a noisy projection to $\mathbf { \mathbf { } } y _ { t } ~ = ~ A \mathbf { \mathbf { } } x _ { t }$ where ${ \pmb y } _ { t } = { \pmb y } + \sigma _ { t } { \pmb \epsilon }$ . Hence, we categorize Score-SDE as “projection”.

Disregarding the guidance strength of Equation 3.5, Equation 3.5 and Equation 3.6 look very similar. Indeed, the only difference is that the latter has stochasticity that arises from the noising of the measurements.

Special case: Inpainting (Repaint [123]) Observe that for the simplest case of inpainting, Equation 3.6 would be replacing the pixel values in the current estimate $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ with the known pixel values from the noised ${ \mathbf { } } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } \mathbf { } } \mathbf { \Delta } _ { \mathbf { } \mathbf { } } \mathbf { \Delta } _ { \mathbf { } \mathcal { } } \mathbf { \Delta } _ { \mathbf { } \mathcal { } } \mathbf { \Delta } _ { \mathbf { } \mathcal { } } \mathbf { \Delta } _ { \mathbf { } \mathcal { } } \mathbf { \Delta } _ { \mathbf { } \mathcal { } \mathcal { } } \mathbf { \Delta } _ { \mathcal { } \mathcal { } } \mathbf { \Delta } _ { \mathcal { } \mathcal { } \mathcal { } } \mathbf { \Delta } _ { \mathcal { } \mathcal { } \mathcal { } } \mathbf { \Delta } _ { \mathcal { } \mathcal { } \mathcal { } \Delta } \mathbf { \Delta } _ { \mathcal \mathcal { } }$ . Coincidentally, this is exactly the Repaint Lugmayr et al. [123] algorithm that was proposed for solving the inpainting inverse problem with pre-trained diffusion models. Re-Paint++ Rout et al. [124] improves upon this approximation to run the forward-reverse diffusion processes multiple times, so that the errors arising (e.g. boundaries) can be mitigated. This can be thought of as analogous to running MCMC corrector steps as in predictor-corrector sampling [2].

## 3.1.3 ILVR [3]

ILVR is a similar approach that was initially proposed for the task of super-resolution. The approximation made here is the following:

$$
\begin{array} { r l r } & { } & { \frac { \mathbf { \sigma } ^ { 4 } | | \mathbf { \dot { \tau } } | | \mathbf { \dot { \eta } } \boldsymbol { 0 } | ^ { \prime } \boldsymbol { \mathrm { m a t r i x } } } { \mathbf { \sigma } ^ { \dagger } } \qquad \underset { \mathbf { \dot { \eta } } ^ { 5 } } { \mathrm { m e a s u l } } } \\ & { } & { \nabla _ { \mathbf { x } _ { t } } \log { p ( \pmb { y } | \mathbf { \cal X } _ { t } = \mathbf { x } _ { t } ) } \approx - A ^ { \dagger } ( \pmb { y } _ { t } - A \pmb { x } _ { t } ) = - \left( \mathbf { \boldsymbol { A } } ^ { \top } A \right) ^ { - 1 } A ^ { \top } \left( \mathbf { \sigma } \pmb { y } _ { t } - A \pmb { x } _ { t } \right) , } \end{array}\tag{3.7}
$$

where $A ^ { \dagger }$ is the Moore-Penrose pseudo-inverse of A, and similar to Score-SDE, $\mathbf { \boldsymbol { y } } _ { t } = \mathbf { \boldsymbol { y } } + \sigma _ { t } \mathbf { \boldsymbol { \epsilon } }$

ILVR can be regarded as a pre-conditioned version of score-SDE. In ILVR, the projection to the space of images happens using the Moore-Penrose pseudo-inverse of $\mathbf { A }$ , instead of the simple $A ^ { \top }$

## 3.1.4 DPS

All of the previous algorithms were proposed for linear inverse problems. Diffusion Posterior Sampling (DPS) is one of the most well known reconstruction algorithms for solving non-linear inverse problems. The underlying approximation behind DPS is that:

$$
\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { X } _ { t } = \pmb { x } _ { t } ) \approx \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { X } _ { 0 } = \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } ] ) .\tag{3.8}
$$

It is easy to see that:

$$
p ( \pmb { y } | \pmb { X } _ { 0 } = \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } ] ) = \mathcal { N } \left( \pmb { y } ; \pmb { \mu } = \pmb { \mathcal { A } } ( \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } ] ) , \pmb { \Sigma } = \sigma _ { \pmb { y } } ^ { 2 } I \right) .\tag{3.9}
$$

Hence, the DPS approximation can be stated as:

$$
\nabla _ { x _ { t } } \log p ( \pmb { y } | X _ { t } = \pmb { x } _ { t } ) \approx \nabla _ { \pmb { x } _ { t } } \log \mathcal { N } \left( \pmb { y } ; \pmb { \mu } = \pmb { \mathcal { A } } ( \mathbb { E } [ X _ { 0 } | X _ { t } = \pmb { x } _ { t } ] ) , \pmb { \Sigma } = \sigma _ { y } ^ { 2 } I \right)\tag{3.10}
$$

$$
= \nabla _ { \pmb { x } _ { t } } \left( \frac { 1 } { 2 \sigma _ { \pmb { y } } ^ { 2 } } | | \pmb { y } - \pmb { \mathcal { A } } ( \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } ] ) | | ^ { 2 } \right)\tag{3.11}
$$

$$
= \frac { 1 } { 2 \sigma _ { y } ^ { 2 } } \nabla _ { x _ { t } } ^ { \top } \mathcal { A } \big ( \mathbb { E } \big [ X _ { 0 } | X _ { t } = x _ { t } \big ] \big ) \left( \mathcal { A } \big ( \mathbb { E } \big [ X _ { 0 } | X _ { t } = x _ { t } \big ] \big ) - y \right) .\tag{3.12}
$$

For linear inverse problems, this simplifies to:

$$
\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { X } _ { t } = \pmb { x } _ { t } ) \approx - \frac { \frac { \mathrm { s } \mathrm { i } | \mathrm { f i f i n g } ^ { \circ } \operatorname { m a t r i x } _ { \pmb { \xi } } } { \sqrt { \pmb { \xi } } } } { 2 \sigma _ { \pmb { y } } ^ { 2 } } \nabla _ { \pmb { x } _ { t } } ^ { \top } \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } ] A ^ { \top } \left( \pmb { y } - A \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } ] \right) .\tag{s error}
$$

(3.13)

We can further use Tweedie’s formula to further write it as:

$$
\nabla _ { x _ { t } } \log p ( y | X _ { t } = x _ { t } ) \approx - \frac { 1 } { 2 \sigma _ { y } ^ { 2 } } \left( I + \nabla _ { x _ { t } } ^ { 2 } \log p _ { t } ( x _ { t } ) \right) ^ { \top } A ^ { \top } \left( y - A \mathbb { E } [ X _ { 0 } | X _ { t } = x _ { t } ] \right) .\tag{3.14}
$$

In practice, DPS does not use the theoretical guidance strength but instead proposes to use a reweighting with a step size inversely proportional to the norm of the measurement error.

MCG Chung et al. [31] provides a geometric interpretation of DPS by showing that the approximation used in DPS can guarantee the noisy samples stay on the manifold. DSG Yang et al. [147] showed that one can choose a theoretically “correct” step size under the geometric view of MCG, and combined with projected gradient descent, one can achieve superior sample quality. MPGD He et al. [33] showed that by constraining the gradient update step to stay on the low dimensional subspace by autoencoding, one can acquire better results.

## 3.1.5 ΠGDM Song et al. [5]

Recall the intractable integral in Equation 1.3. According to this relation, the DPS approximation is achieved by setting

$$
p ( \mathbf { \boldsymbol { X } } _ { 0 } | \mathbf { \boldsymbol { X } } _ { t } ) \approx \delta ( \mathbf { \boldsymbol { X } } _ { 0 } - \mathbb { E } [ \mathbf { \boldsymbol { X } } _ { 0 } | \mathbf { \boldsymbol { X } } _ { t } = \mathbf { \boldsymbol { x } } _ { t } ] ) .\tag{3.15}
$$

In ΠGDM, the authors propose to use a Gaussian distribution for approximation

$$
p ( \mathbf { \boldsymbol { X } } _ { 0 } | \mathbf { \boldsymbol { X } } _ { t } ) \approx \mathcal { N } ( \mathbb { E } [ \mathbf { \boldsymbol { X } } _ { 0 } | \mathbf { \boldsymbol { X } } _ { t } = \mathbf { \boldsymbol { x } } _ { t } ] , r _ { t } ^ { 2 } I _ { n } ) ,\tag{3.16}
$$

where $r _ { t }$ is a hyperparameter. For linear inverse problems, this leads to

$$
p ( \pmb { y } | \pmb { X } _ { t } ) \approx \mathcal { N } ( A \hat { \mathbb { E } } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } ] , r _ { t } ^ { 2 } A A ^ { \top } + \sigma _ { y } ^ { 2 } I _ { n } ) .\tag{3.17}
$$

Subsequently, we have

$$
\begin{array} { r l r } & { } & { \nabla _ { { \pmb x } _ { t } } \log p \big ( { \pmb y } | { \pmb X } _ { t } = { \pmb x } _ { t } \big ) } \\ & { } & { \approx - \frac { \partial { \mathbb E } [ { \pmb X } _ { 0 } | { \pmb X } _ { t } = { \pmb x } _ { t } ] } { \partial { \pmb x } _ { t } } ( r _ { t } ^ { 2 } A A ^ { \top } + \sigma _ { y } ^ { 2 } I ) ^ { - 1 } A ^ { \top } \mathrm { ~ ( ~ } { \pmb y } - A { \mathbb E } [ { \pmb X } _ { 0 } | { \pmb X } _ { t } = { \pmb x } _ { t } ] \mathrm { ~ ) } . \qquad ( ) } \\ & { } & { \frac { \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ i | } \mathrm { f i l i n g " ~ m a t r i x } \mathrm { ~ } \biggr ] } { \mathrm { ~ } } } \end{array}\tag{3.18}
$$

## 3.1.6 Moment Matching [6]

In ΠGDM, the distribution $p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } )$ was assumed to be isotropic Gaussian. However, one can calculate explicitly the variance matrix, $V [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ]$ . As shown in Lemma A.4, it holds that:

$$
V [ { \pmb x } _ { 0 } | { \pmb x } _ { t } ] = \sigma _ { t } ^ { 4 } H ( \log p _ { t } ( { \pmb x } _ { t } ) ) + \sigma _ { t } ^ { 2 } I _ { n }\tag{3.19}
$$

$$
\begin{array} { r } { \mathbf { \Psi } = \sigma _ { t } ^ { 2 } \nabla _ { \pmb { x } _ { t } } \mathbb { E } [ \pmb { x } _ { 0 } \vert \pmb { x } _ { t } ] . } \end{array}\tag{3.20}
$$

The Moment Matching [6] method approximates the distribution $p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } )$ with an anisotropic Gaussian:

$$
\begin{array} { r } { p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } ) \approx \mathcal { N } ( \mathbb { E } [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ] , V [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ] ) . } \end{array}\tag{3.21}
$$

For linear inverse problems, this leads to the following approximation for the measurements’ score:

$$
\begin{array} { c } { { \nabla \log p ( { \pmb y } | { \pmb X } _ { t } = { \pmb x } _ { t } ) } } \\ { { \approx - \nabla _ { { \pmb x } _ { t } } \mathbb { E } [ { \pmb x } _ { 0 } | { \pmb x } _ { t } ] ^ { \top } { \pmb A } ^ { \top } ( \sigma _ { y } ^ { 2 } I + A \sigma _ { t } ^ { 2 } \nabla _ { { \pmb x } _ { t } } \mathbb { E } [ { \pmb x } _ { 0 } | { \pmb x } _ { t } ] { \pmb A } ^ { \top } ) ^ { - 1 } \left( \begin{array} { l } { { \pmb y } - A \mathbb { E } [ { \pmb x } _ { 0 } | { \pmb x } _ { t } ] } \\ { \qquad { \vdots } } \\ { \end{array} \biggr \array} \right) . } } \\ { { { \ c } _ { | \mathrm { l i f f i n g } ^ { \top } \mathrm { ~ m a t r i x } } \biggr \} } } \end{array} \begin{array} { l } { { ( 0 . 0 ) } }  \end{array}\tag{3.22}
$$

In high-dimensions, even materializing the matrix $\nabla _ { \pmb { x } _ { t } } \mathbb { E } [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ]$ is computationally intensive. Instead, the authors of [6] use automatic differentiation to compute the Jacobian-vector products.

## 3.1.7 BlindDPS Chung et al. [7]

Methods that were considered so far were designed for non-blind inverse problems, where $A$ is fully known. BlindDPS targets the case where we have a parametrized unknown forward model $A _ { \phi } \left( \mathrm { e . g } \right.$ blurring with an unknown kernel $\phi )$ . In BlindDPS, on top of the posterior mean approximation of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ one approximates the parameter of the forward model, again, with the posterior mean. Specifically, we design two parallel generative SDEs

$$
\mathrm { d } \mathbf { \boldsymbol { x } } _ { t } = \left( \pmb { f } ( \mathbf { \boldsymbol { x } } _ { t } , t ) - g ^ { 2 } ( t ) \nabla _ { \mathbf { \boldsymbol { x } } _ { t } } \log p _ { t } ( \mathbf { \boldsymbol { x } } _ { t } , \phi _ { t } | \mathbf { \boldsymbol { y } } ) \right) \mathrm { d } t + g ( t ) \mathrm { d } W _ { t }\tag{3.23}
$$

$$
\mathrm { d } \phi _ { t } = \left( f ( \phi _ { t } , t ) - g ^ { 2 } ( t ) \nabla _ { \phi _ { t } } \log p _ { t } ( x _ { t } , \phi _ { t } | y ) \right) \mathrm { d } t + g ( t ) \mathrm { d } W _ { t } ,\tag{3.24}
$$

where the two SDEs are coupled through log $p _ { t } ( \pmb { x } _ { t } , \phi _ { t } | \pmb { y } )$ , where under the independence between $X _ { t }$ and $\Phi _ { t }$ , the Bayes rule reads

$$
\begin{array} { r } { \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } , \phi _ { t } | y ) = \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } ) + \nabla _ { x _ { t } } \log p _ { t } ( y | X _ { t } = x _ { t } , \Phi _ { t } = \phi _ { t } ) } \end{array}\tag{3.25}
$$

$$
\begin{array} { r } { \nabla _ { \phi _ { t } } \log p _ { t } ( \boldsymbol { x } _ { t } , \phi _ { t } | \boldsymbol { y } ) = \nabla _ { \phi _ { t } } \log p _ { t } ( \phi _ { t } ) + \nabla _ { \boldsymbol { x } _ { t } } \log p _ { t } ( \boldsymbol { y } | \boldsymbol { X } _ { t } = \boldsymbol { x } _ { t } , \Phi _ { t } = \phi _ { t } ) , } \end{array}\tag{3.26}
$$

where we see that $X _ { t }$ and $\Phi _ { t }$ are coupled through the likelihood $p ( \pmb { y } | \pmb { X } _ { t } , \pmb { \Phi } _ { t } )$ . In BlindDPS, the approximation used in DPS is applied to both the image and the operator, leading to

$$
p ( \boldsymbol { y } | \boldsymbol { X } _ { t } = \boldsymbol { x } _ { t } , \Phi _ { t } = \phi _ { t } ) \approx p ( \boldsymbol { y } | \boldsymbol { X } _ { 0 } = \mathbb { E } [ \boldsymbol { X } _ { 0 } | \boldsymbol { X } _ { t } = \boldsymbol { x } _ { t } ] , \Phi _ { 0 } = \mathbb { E } [ \Phi _ { 0 } | \Phi _ { t } = \phi _ { t } ] \big ) .\tag{3.27}
$$

The gradient of the coupled likelihood with respect to $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ leads to

guidance str

$$
\begin{array} { r l } {  { \frac { \mathrm {  ~ \ p n g t h ~ } } { \mathrm {  ~ \Lambda ~ } } } } & { \frac { \mathrm {  ~ \psi ~ } _ { \mathrm { i j f f i t i n g } } ^ { \mathrm { a p } } \mathrm {  ~ \psi ~ } \mathrm {  ~ \psi ~ } _ { \mathrm { r } } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \nabla _ { \mathrm { r } _ { t } } \log p ( \boldsymbol { y } | \boldsymbol { X } _ { t } = \boldsymbol { x } _ { t } , \Phi _ { t } = \phi _ { t } ) } { \mathrm {  ~ \psi ~ } } \underbrace { \mathrm {  ~ \psi ~ } _ { \mathrm { r } } \mathrm {  ~ \psi ~ } _ { \mathrm { o s u r e m e n t s ~ e r r o r } } } _ { \mathrm {  { \psi } } } } \\ & { \approx - \frac { 1 } { 2 \sigma _ { y } ^ { 2 } } \nabla _ { \boldsymbol { x } _ { t } } ^ { \top } \mathbb { E } [ \boldsymbol { X } _ { 0 } | \boldsymbol { X } _ { t } = \boldsymbol { x } _ { t } ] A _ { \mathbb { E } [ \Phi _ { 0 } | \Phi _ { t } = \phi _ { t } ] } ^ { \top } ( \boldsymbol { y } - A _ { \mathbb { E } [ \Phi _ { 0 } | \Phi _ { t } = \phi _ { t } ] } \mathbb { E } [ \boldsymbol { X } _ { 0 } | \boldsymbol { X } _ { t } = \boldsymbol { x } _ { t } ] ) . } \end{array}\tag{3.28}
$$

Similarly, for $\phi _ { t }$ , we have

guidance str

$$
\begin{array} { r l r } {  { \frac { \Theta \mathrm { n g t h } } { \Theta } } } & { \frac { \mathrm { s i l f i t i n g } ^ { \ast } \operatorname* { m a t r i x } \quad \nabla _ { \phi _ { t } } \log p ( \boldsymbol { y } | \boldsymbol { X } _ { t } = \boldsymbol { x } _ { t } , \Phi _ { t } = \phi _ { t } ) } { \mathrm { s } } } & { \underset { \boldsymbol { \varsigma } } { \mathrm { m e a s u r e m e n t s ~ e r r o r } } } \\ & { } & { \approx - \frac { 1 } { 2 \sigma _ { y } ^ { 2 } } \ \nabla _ { \phi _ { t } } ^ { \top } \mathbb { E } [ \boldsymbol { X } _ { 0 } | \boldsymbol { X } _ { t } = \boldsymbol { x } _ { t } ] A _ { \mathrm { E } [ \Phi _ { 0 } | \Phi _ { t } = \phi _ { t } ] } ^ { \top } \ \Big ( \boldsymbol { y } - A _ { \mathrm { E } [ \Phi _ { 0 } | \Phi _ { t } = \phi _ { t } ] } \mathbb { E } [ \boldsymbol { X } _ { 0 } | \boldsymbol { X } _ { t } = \boldsymbol { x } _ { t } ] \Big ) . } \end{array}\tag{3.29}
$$

## 3.1.8 DDRM Family

The methods under the DDRM family poses all linear inverse problems to a noisy inpainting problem, by decomposing the measurement matrix with singular value decomposition (SVD), i.e. $\dot { \boldsymbol { A } } = \boldsymbol { U } \dot { \boldsymbol { \Sigma } } \dot { \boldsymbol { V } } ^ { \intercal }$ , where $U ^ { ^ { - } } \in \mathbb { R } ^ { m \times m } , V \in \mathbb { R } ^ { n \times n }$ are orthogonal matrices, and $\Sigma \in \mathbb { R } ^ { m \times n }$ is a rectangular diagonal matrix with singular values $\{ s _ { j } \} _ { j = 1 } ^ { m }$ as the elements. One can then rewrite $\pmb { y } = A \pmb { x } + \sigma _ { \pmb { y } } z , z \sim \mathcal { N } ( \mathbf { 0 } , I _ { m } )$ as

$$
\bar { \pmb { y } } = \Sigma \bar { \pmb { x } } + \sigma _ { \pmb { y } } \bar { \pmb { z } } , \quad \mathrm { w h e r e } \quad \bar { \pmb { y } } : = \pmb { U } ^ { \top } \pmb { y } , \bar { \pmb { x } } : = \pmb { V } ^ { \top } \pmb { x } , \bar { \pmb { z } } : = \pmb { U } ^ { \top } \pmb { z } .\tag{3.30}
$$

Subsequently, Equation 3.30 becomes an inpainting problem in the spectral space.

SNIPS [8]. SNIPS proceeds by first solving the inverse problem posed as Equation 3.30 in the spectral space to achieve a sample $\bar { \pmb x } \sim p ( \bar { \pmb x } | \bar { \pmb y } )$ , then retrieving the posterior sample with ${ \hat { \mathbf { x } } } = V { \bar { \mathbf { x } } }$ The key approximation can be concisely represented as

$$
\nabla _ { \bar { \pmb { x } } _ { t } } \log p ( \bar { \pmb { y } } | \bar { \pmb { X } } _ { t } = \pmb { x } _ { t } ) \approx - \Sigma ^ { \top } \left| \sigma _ { \pmb { y } } ^ { 2 } I _ { m } - \sigma _ { t } ^ { 2 } \Sigma \Sigma ^ { \top } \right| ^ { \dag } ( \bar { \pmb { y } } - \Sigma \bar { \pmb { x } } _ { t } ) ,\tag{3.31}
$$

rements error

For the simplest case of denoising where $m = n$ and $\Sigma = A = I$ , the method becomes [148]

$$
\nabla _ { \pmb { X } _ { t } } \log p ( \pmb { y } | \pmb { X } _ { t } = \pmb { x } _ { t } ) \approx \frac { \pmb { y } - \pmb { x } _ { t } } { \vert \sigma _ { \pmb { y } } ^ { 2 } - \sigma _ { t } ^ { 2 } \vert } .\tag{3.32}
$$

which produces a vector direction that is weighted by the absolute difference between the diffusion noise level $\sigma _ { t } ^ { 2 }$ , and the measurement noise level $\sigma _ { y } ^ { 2 }$ . For the fully general case in Equation 3.31, elements in different indices are weighted according to the singular values contained in $\Sigma .$ In practice, SNIPS uses pre-conditioning with the approximate negative inverse Hessian of log $p ( \bar { \pmb x } _ { t } | \bar { \pmb y } )$ when running annealed Langevin dynamics.

DDRM [9]. DDRM extends SNIPS by leveraging the posterior mean $\bar { \pmb { x } } _ { 0 \mid t } : = V \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } ]$ in the place of $\bar { \mathbf { x } } _ { t }$ used in SNIPS. i.e.,

$$
\begin{array} { r l r } & { } & { \nabla _ { \bar { \boldsymbol { x } } _ { t } } \log p ( \bar { \boldsymbol { y } } | \bar { \boldsymbol { X } } _ { t } = \boldsymbol { x } _ { t } ) \approx - \sum ^ { \top } \left| \sigma _ { y } ^ { 2 } I _ { m } - \sigma _ { t } ^ { 2 } \Sigma \Sigma ^ { \top } \right| ^ { \dag } ( \bar { \boldsymbol { y } } - \Sigma \bar { \boldsymbol { x } } _ { 0 \mid t } ) . } \\ & { } & { \frac { \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } } { \mathrm { ~ } \mathrm { ~ } \mathrm { ~ I ~ i ~ f ~ t ~ i ~ n ~ g ~ } ^ { \mathrm { ~ p ~ } } \mathrm { m a t r i x } \dag } \mathrm { ~ } \hat { \left[ \begin{array} { l } { \mathrm { ~ } \mathrm { m e a s u m } } \end{array} \right] } , } \end{array}\tag{3.33}
$$

ements error

Expressing Equation 3.33 element-wise, we get

$$
p ( \bar { x } _ { t } ^ { ( i ) } | X _ { t + 1 } = x _ { t + 1 } , y ) = \left\{ \begin{array} { l l } { \mathcal { N } ( \bar { x } _ { t } ^ { ( i ) } ; \bar { x } _ { 0 | t + 1 } ^ { ( i ) } , \sigma _ { t } ^ { 2 } ) } & { \mathrm { i f ~ } s _ { i } = 0 } \\ { \mathcal { N } ( \bar { x } _ { t } ^ { ( i ) } ; \bar { x } _ { 0 | t + 1 } ^ { ( i ) } , \sigma _ { t } ^ { 2 } ) } & { \mathrm { i f ~ } \sigma _ { t } < \frac { \sigma _ { y } } { s _ { i } } \ , } \\ { \mathcal { N } ( \bar { x } _ { t } ^ { ( i ) } ; \bar { y } ^ { ( i ) } , \sigma _ { t } ^ { 2 } - \frac { \sigma _ { y } ^ { 2 } } { s _ { i } ^ { 2 } } ) } & { \mathrm { i f ~ } \sigma _ { t } \geq \frac { \sigma _ { y } } { s _ { i } } } \end{array} \right.\tag{3.34}
$$

where $\mathbf { \boldsymbol { x } } ^ { ( i ) }$ denotes the i-th element of the vector, and $s _ { i }$ its corresponding singular value. Here, DDRM introduces another hyper-parameter η to control the stochasticity of the sampling process

$$
\begin{array} { r } { p ( \bar { x } _ { t } ^ { ( i ) } | X _ { t + 1 } = x _ { t + 1 } , y ) = \left\{ \begin{array} { l l } { \mathcal { N } ( \bar { x } _ { t } ^ { ( i ) } ; \bar { x } _ { 0 | t + 1 } ^ { ( i ) } + \sqrt { 1 - \eta ^ { 2 } } \sigma _ { t } \frac { \bar { x } _ { t + 1 } ^ { ( i ) } - \bar { x } _ { 0 | t + 1 } ^ { ( i ) } } { \sigma _ { t + 1 } } , \eta ^ { 2 } \sigma _ { t } ^ { 2 } ) } & { \mathrm { i f ~ } s _ { i } = 0 } \\ { \mathcal { N } ( \bar { x } _ { t } ^ { ( i ) } ; \bar { x } _ { 0 | t + 1 } ^ { ( i ) } + \sqrt { 1 - \eta ^ { 2 } } \sigma _ { t } \frac { \bar { y } ^ { ( i ) } - \bar { x } _ { 0 | t + 1 } ^ { ( i ) } } { \sigma _ { y } / s _ { i } } , \eta ^ { 2 } \sigma _ { t } ^ { 2 } ) } & { \mathrm { i f ~ } \sigma _ { t } < \frac { \sigma _ { y } } { s _ { i } } \ , } \\ { \mathcal { N } ( \bar { x } _ { t } ^ { ( i ) } ; \bar { y } ^ { ( i ) } , \sigma _ { t } ^ { 2 } - \frac { \sigma _ { y } ^ { 2 } } { s _ { i } ^ { 2 } } ) } & { \mathrm { i f ~ } \sigma _ { t } \ge \frac { \sigma _ { y } } { s _ { i } } } \end{array} \right. } \end{array}\tag{3.35}
$$

with $\eta \in ( 0 , 1 ]$ such that $\eta = 1 . 0$ recovers Equation 3.34.

GibbsDDRM. GibbsDDRM Murata et al. [10] extends DDRM to the following blind linear problem $\begin{array} { r } { \pmb { y } = A _ { \varphi } \pmb { x } + \sigma _ { \pmb { y } } z } \end{array}$ , where $A _ { \varphi }$ is a linear operator parameterized by $\varphi .$ . Here, $A _ { \varphi } = U _ { \varphi } \Sigma _ { \varphi } ^ { \circ } V _ { \varphi } ^ { \intercal }$ has a ϕ dependence SVD decomposition with singular values $\{ s _ { j , \varphi } \} _ { j = 1 } ^ { m }$ as the elements of the diagonal matrix $\Sigma _ { \varphi }$ . In the spectral space, $\bar { \pmb { y } } _ { \pmb { \varphi } } : = U _ { \pmb { \varphi } } ^ { \top } \pmb { y } _ { \pmb { \varphi } } , \bar { \pmb { x } } _ { \pmb { \varphi } } : = V _ { \pmb { \varphi } } ^ { \top } \pmb { x } _ { \pmb { \varphi } } , \bar { z } _ { \pmb { \varphi } } : = U _ { \pmb { \varphi } } ^ { \top } z _ { \pmb { \varphi } }$ . Subsequently, the posterior mean in DDRM is replaced with $\bar { \pmb { x } } _ { 0 | t , \varphi } : = V _ { \pmb { \varphi } } \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } ]$ , also depending on $\varphi .$ Thus, it leads to the sampling process

$$
p ( \bar { x } _ { t , \varphi } ^ { ( i ) } | X _ { t + 1 } = x _ { t + 1 } , y , \varphi ) = \left\{ \begin{array} { l l } { \mathcal { N } ( \bar { x } _ { t , \varphi } ^ { ( i ) } ; \bar { x } _ { 0 | t + 1 , \varphi } ^ { ( i ) } + \sqrt { 1 - \eta ^ { 2 } } \sigma _ { t } \frac { \bar { x } _ { t + 1 , \varphi } ^ { ( i ) } - \bar { x } _ { 0 | t + 1 , \varphi } ^ { ( i ) } } { \sigma _ { t + 1 } } , \eta ^ { 2 } \sigma _ { t } ^ { 2 } ) } & { \mathrm { i f ~ } \sigma _ { i , \varphi } = 0 } \\ { \mathcal { N } ( \bar { x } _ { t , \varphi } ^ { ( i ) } ; \bar { x } _ { 0 | t + 1 , \varphi } ^ { ( i ) } + \sqrt { 1 - \eta ^ { 2 } } \sigma _ { t } \frac { \bar { y } _ { \varphi } ^ { ( i ) } - \bar { x } _ { 0 | t + 1 , \varphi } ^ { ( i ) } } { \sigma _ { y } / s _ { i , \varphi } } , \eta ^ { 2 } \sigma _ { t } ^ { 2 } ) } & { \mathrm { i f ~ } \sigma _ { t } < \frac { \sigma _ { y } } { s _ { i , \varphi } } . } \\ { \mathcal { N } ( \bar { x } _ { t , \varphi } ^ { ( i ) } ; \bar { y } _ { \varphi } ^ { ( i ) } , \sigma _ { t } ^ { 2 } - \frac { \sigma _ { y } ^ { 2 } } { s _ { i , \varphi } ^ { 2 } } ) } & { \mathrm { i f ~ } \sigma _ { t } \ge \frac { \sigma _ { y } } { s _ { i , \varphi } } . } \end{array} \right.\tag{3.36}
$$

At time step $t , \varphi$ is sampled by using the conditional distribution $p ( \varphi | \mathbf { x } _ { t : T } , \mathbf { y } )$ and updated for several iterations in a Langevin manner:

$$
\varphi  \varphi + \frac { \xi } { 2 } \nabla _ { \varphi } \log p ( \varphi | x _ { t : T } , \pmb { y } ) + \sqrt { \xi } \pmb { \epsilon } ,
$$

where $\xi$ is a stepsize and $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , I _ { n } )$ . Here, $\nabla _ { \varphi } \log { p ( \varphi | x _ { t : T } , y ) } \approx \nabla _ { \varphi } \log { p ( \varphi | \bar { x } _ { 0 | t , \varphi } , y ) }$ , and the gradient can be computed as:

$$
\nabla _ { \varphi } \log p ( \varphi | \bar { \boldsymbol { x } } _ { 0 \mid t , \varphi } , \boldsymbol { y } ) = - \frac { 1 } { 2 \sigma _ { y } ^ { 2 } } \nabla _ { \varphi } \left| \left| \boldsymbol { y } - \mathbf { A } _ { \varphi } \bar { \boldsymbol { x } } _ { 0 \mid t , \varphi } \right| \right| ^ { 2 } .\tag{3.37}
$$

## 3.1.9 DDNM Wang et al. [11] family

A different way to find meaningful approximations for the conditional score is to look at the conditional version of Tweedie’s formula, see Equation 2.15. Using Bayes rule and rearranging Ravula et al. [149], we have

$$
\begin{array} { r } { \mathbb { E } [ X _ { 0 } | X _ { t } = x _ { t } , y ] = x _ { t } + \sigma _ { t } ^ { 2 } \nabla _ { x _ { t } } \log p _ { t } ( X _ { t } | y ) } \end{array}\tag{3.38}
$$

$$
= { \pmb x } _ { t } + \sigma _ { t } ^ { 2 } \nabla _ { { \pmb x } _ { t } } \log p _ { t } ( { \pmb x } _ { t } ) + \sigma _ { t } ^ { 2 } \nabla _ { { \pmb x } _ { t } } \log p _ { t } ( { \pmb y } | { \pmb X } _ { t } = { \pmb x } _ { t } )\tag{3.39}
$$

$$
= \mathbb E [ X _ { 0 } | X _ { t } = x _ { t } ] + \sigma _ { t } ^ { 2 } \nabla _ { x _ { t } } \log p _ { t } ( y | X _ { t } = x _ { t } ) .\tag{3.40}
$$

The methods that belong to the DDNM family make approximations to $\mathbb { E } [ X _ { 0 } | X _ { t } ~ = ~ { \pmb { x } } _ { t } , { \pmb { y } } ]$ by making certain data consistency updates to $\mathbb { E } [ X _ { 0 } | X _ { t } = \bar { \mathbf { x } _ { t } } ]$

DDNM Wang et al. [11]. The simplest form of update when considering no noise can be obtained through range-null space decomposition, assuming that one can compute the pseudo-inverse. In DDNM, this condition is trivially met by considering operations that are SVD-decomposable. DDNM proposes to use the following projection step to the posterior mean to obtain an approximation of the conditional posterior mean

$$
\begin{array} { r } { \mathbb { E } [ { \pmb X } _ { 0 } | { \pmb X } _ { t } = { \pmb x } _ { t } , { \pmb y } ] \approx ( I - A ^ { \dagger } A ) \mathbb { E } [ { \pmb X } _ { 0 } | { \pmb X } _ { t } = { \pmb x } _ { t } ] + A ^ { \dagger } { \pmb y } , } \end{array}\tag{3.41}
$$

where $A ^ { \dagger }$ is the Moore-Penrose pseudo-inverse of A. One can also express Equation 3.41 as an approximation of the likelihood, consistent to other methods in the chapter. Specifically, notice that by using the relation in Equation 3.40,

$$
\nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { y } | \pmb { X } _ { t } = \pmb { x } _ { t } ) = \frac { 1 } { \sigma _ { t } ^ { 2 } } \big ( \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } , \pmb { y } ] - \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } ] \big ) .\tag{3.42}
$$

Plugging in Equation 3.41 to Equation 3.42,

$$
\begin{array} { r l r } {  { \frac { \boldsymbol { \mathfrak { u } } \mathrm { i } | \mathrm { i f f i l } \boldsymbol { \mathfrak { u } } \boldsymbol { \mathfrak { g } } ^ { \flat } \operatorname* { m a t r i x } } { \boldsymbol { \Psi } } } } \\ & { } & { \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { y } | \boldsymbol { X } _ { t } = \pmb { x } _ { t } ) \approx \ - \frac { 1 } { \sigma _ { t } ^ { 2 } } \overset { \sharp } { A ^ { \dagger } } \ ( \pmb { y } - A \mathbb { E } [ \boldsymbol { X } _ { 0 } | \boldsymbol { X } _ { t } = \pmb { x } _ { t } , \pmb { y } ] ) } \\ & { } & { \operatorname * { g u i d a n c e s t r e n g t h } \quad \uparrow \ } \end{array}\tag{3.43}
$$

When there is noise in the measurement, one can make soft updates

$$
\begin{array} { r } { \mathbb { E } [ X _ { 0 } | X _ { t } = x _ { t } , y ] \approx ( I - \Sigma _ { t } A ^ { \dagger } A ) \mathbb { E } [ X _ { 0 } | X _ { t } = x _ { t } ] + \Sigma _ { t } A ^ { \dagger } y , \quad \Sigma \in \mathbb { R } ^ { n \times n } . } \end{array}\tag{3.44}
$$

Also, similar to Equation 3.43,

$$
\begin{array} { r l r } {  { \frac { \omega \vert \mathrm { i f f i l i n g } ^ { \mathrm { * } } \operatorname* { m a t r i x } \forall } { \sigma } \qquad } } & { \underset { \mathcal { Y } } { \mathrm { m e a s u r e m e n t s ~ e r r o r } } } \\ & { } & { \nabla _ { x _ { t } } \log p _ { t } ( y \vert X _ { t } = x _ { t } ) \approx - \frac { 1 } { \sigma _ { t } ^ { 2 } } \begin{array} { r } { \sum _ { t } A ^ { \dagger } } \\ { \int ( y - A \mathbb { E } [ X _ { 0 } \vert X _ { t } = x _ { t } , y ] ) } \end{array} } \end{array}\tag{3.45}
$$

Here, one can choose a simple $\Sigma _ { t } = \lambda _ { t } I$ with $\lambda _ { t }$ set as a hyper-parameter, or use different scaling for each spectral component. Observe that due to the relationship between the (conditional) score function and the posterior mean established in Equation 3.40, we can also easily rewrite the approximation in terms of the score of the posterior.

DDS Chung et al. [12], DiffPIR Zhu et al. [13]. Both DDS and DiffPIR propose a proximal update to approximate the conditional posterior mean, albeit from different motivations. The resulting approximation reads

$$
\mathbb { E } [ X _ { 0 } | X _ { t } = x _ { t } , y ] \approx \operatorname * { a r g m i n } _ { x } { \frac { 1 } { 2 } } \| y - A x \| ^ { 2 } + { \frac { \lambda _ { t } } { 2 } } \| x - \mathbb { E } [ X _ { 0 } | X _ { t } = x _ { t } ] \| ^ { 2 } .\tag{3.46}
$$

The difference between the two algorithms comes from how one solves the optimization problem in Equation 3.46, and how one chooses the hyperparameter $\lambda _ { t }$ . In DDS, the optimization is solved with a few-step conjugate gradient (CG) update steps, by showing that DPS gradient update steps can be effectively replaced with the CG steps under assumptions on the data manifold Chung et al. [12]. λ<sub>t</sub> is taken to be a constant value across all t. DiffPIR uses a closed-form solution for Equation 3.46, and proposes a schedule for $\lambda _ { t }$ that is proportional to the signal-to-noise (SNR) ratio of the diffusion at time t. Specifically, one chooses $\lambda _ { t } = \sigma _ { t } \zeta$ , where ζ is a constant.

## 3.2 Variational Inference

These methods approximate the true posterior distribution, $p ( { \pmb x } | { \pmb y } )$ , with a simpler, tractable distribution. Variational formulations are then used to optimize the parameters of this simpler distribution.

## 3.2.1 RED-Diff Mardani et al. [16]

Mardani et al. [16] introduce RED-diff, a new approach for solving inverse problems by leveraging stochastic optimization and diffusion models. The core idea is to use variational method by introducing a simpler distribution, $q : = \mathcal { N } ( \pmb { \mu } , \sigma ^ { 2 } I _ { n } )$ , to approximate the true posterior $p ( \pmb { x } _ { 0 } | \pmb { y } )$ by minimizing the KL divergence $\mathcal { D } _ { \mathrm { K I } }$ between them:

$$
\operatorname* { m i n } _ { \boldsymbol { q } } \mathcal { D } _ { \mathrm { K L } } \big ( q ( \boldsymbol { x } _ { 0 } | \boldsymbol { y } ) \big | \big | p \big ( \boldsymbol { x } _ { 0 } | \boldsymbol { y } \big ) \big ) .\tag{3.47}
$$

Here, ${ \mathcal { D } } _ { \mathrm { K L } } ( q ( { \pmb x } _ { 0 } | { \pmb y } ) | | p ( { \pmb x } _ { 0 } | { \pmb y } ) )$ can be written as follows:

$$
\mathcal { D } _ { \mathrm { K L } } ( q ( x _ { 0 } | y ) \| p ( x _ { 0 } | y ) ) = \underbrace { - \mathbb { E } _ { q ( x _ { 0 } | y ) } [ \log p ( y | x _ { 0 } ) ] + \mathcal { D } _ { \mathrm { K L } } \big ( q ( x _ { 0 } | y ) \| p ( x _ { 0 } ) \big ) } _ { \mathrm { V a r i a t i o n a l ~ B o u n d ~ ( V B ) } } + \mathrm { c o n s t a n t } .\tag{3.48}
$$

via classic variational inference argument. The first term in VB can be simplified into reconstruction loss, and the second term can be decomposed as score-matching objective which involves matching the score function of the variational distribution with the score function of the true posterior denoisers at different timesteps:

$$
\operatorname* { m i n } _ { \mu } \frac { | | y - \mathcal { A } ( \mu ) | | ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } + \mathbb { E } _ { t , \epsilon } [ \lambda _ { t } | | \epsilon _ { \theta } ( x _ { t } ; t ) - \epsilon | | ^ { 2 } ]\tag{3.49}
$$

where $\pmb { \mu }$ is the mean of the variational distribution, and $\sigma _ { v } ^ { 2 }$ is the noise variance in the observation, $\epsilon _ { \theta } ( x _ { t } ; t )$ is the score function of the diffusion model at timestep (t) and $\lambda _ { t }$ is a time-weighting factor.

Sampling as optimization. The goal is then to find an image $\pmb { \mu }$ that reconstructs the observation y given by $f ,$ while having a high likelihood under the denoising diffusion prior (regularizer). This score-matching objective is optimized using stochastic gradient descent, effectively turning the sampling problem into an optimization problem. The weighting factor $( \lambda _ { t } )$ is chosen based on the signal-to-noise ratio (SNR) at each timestep to balance the contribution of different denoisers in the diffusion process.

## 3.2.2 Blind RED-Diff Alkan et al. [17]

In Alkan et al. [17] authors introduce blind RED-diff, an extension of the RED-diff framework Mardani et al. [16] to solve blind inverse problems. The main idea is to use variational inference to jointly estimate the latent image and the unknown forward model parameters.

Similar to RED-Diff, the key mathematical formulation is the minimization of the KL-divergence between the true posterior distribution $p ( \pmb { x } _ { 0 } , \gamma | \pmb { y } )$ and a variational approximation $q ( { \pmb x } _ { 0 } , \gamma | { \pmb y } ) \colon$

$$
\operatorname* { m i n } _ { \boldsymbol q } \mathcal D _ { \mathrm { K L } } ( q ( \boldsymbol x _ { 0 } , \gamma | \boldsymbol y ) | | p ( \boldsymbol x _ { 0 } , \gamma | \boldsymbol y ) ) .
$$

If we assume the latent image and the forward model parameters are independent, the KL-divergence can be decomposed as:

$$
\mathcal D _ { \mathrm { K L } } ( q ( x _ { 0 } | y ) | | p ( x _ { 0 } ) ) + \mathcal D _ { \mathrm { K L } } ( q ( \gamma | y ) | | p ( \gamma ) ) - \mathbb { E } _ { q ( x _ { 0 } , \gamma | y ) } [ \log p ( y | x _ { 0 } , \gamma ) ] + \log p ( y ) .
$$

The minimization with respect to $q$ involves three terms:

i. ${ \mathcal { D } } _ { \mathrm { K L } } ( q ( { \pmb x } _ { 0 } | { \pmb y } ) | | p ( { \pmb x } _ { 0 } ) )$ ) represents the KL divergence between the variational distribution of the image $\mathbf { \Gamma } ( \pmb { x } _ { 0 } )$ and its prior distribution. This term is approximated using a score-matching loss, which leverages denoising score matching with a diffusion model (as in RED-Diff).

ii. $\mathcal { D } _ { \mathrm { K L } } ( q ( \gamma | \pmb { y } ) | | p ( \gamma ) )$ is the KL divergence between the variational distribution of the forward model parameters $( \gamma )$ and their prior distribution. This term acts as a regularizer on γ.

iii. $- \mathbb { E } _ { q ( \pmb { x } _ { 0 } , \gamma | \pmb { y } ) } [ \log p ( \pmb { y } | \pmb { x } _ { 0 } , \gamma ) ]$ is the expectation of the negative log-likelihood of the observed data y given the image $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ and the forward model parameters $\gamma$ . This term ensures data consistency.

The resulting optimization can be achieved using alternating stochastic optimization, where the image x<sub>0</sub> and the forward model parameters γ are updated iteratively.

The formulation assumes conditional independence between $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ and $\gamma$ given the measurement $^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } \mathbf { \Lambda } ^ { \mathrm { \Lambda } } \mathbf { \Lambda } \mathbf { \Lambda } ^ { \mathrm { \Lambda } } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathrm { \Lambda } ^ { \mathrm { \Lambda } }$ and it also requires a specific form for the prior distribution $p ( \gamma )$

## 3.2.3 Score Prior Feng et al. [18]

We again start by introducing a variational distribution $q _ { \phi } ( { \pmb x } _ { 0 } )$ that aims to approximate the posterior distribution determined by the diffusion prior. The optimization problem becomes

$$
\operatorname* { m i n } _ { \phi } \mathcal { D } _ { \mathrm { K L } } \big ( q _ { \phi } ( \pmb { x } _ { 0 } ) \big | \big | p _ { \theta } \big ( \pmb { x } _ { 0 } \big | \pmb { y } \big ) \big )\tag{3.50}
$$

$$
\operatorname* { m i n } _ { \phi } \int q _ { \phi } ( x _ { 0 } | y ) [ - \log p ( y | x _ { 0 } ) - \log p _ { \theta } ( x _ { 0 } ) + \log q _ { \phi } ( x _ { 0 } ) ] .\tag{3.51}
$$

One of the most expressive yet tractable proposal distributions is normalizing flows (NF) Rezende and Mohamed [150], Dinh et al. [151]. Choosing $q _ { \phi }$ to be an NF, we can transform the optimization problem to

$$
\operatorname* { m i n } _ { \phi } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , I _ { n } ) } \left[ \underbrace { - \log p ( y | G _ { \phi } ( z ) ) } _ { \mathrm { L i k e l i h o o d } } - \underbrace { \log p _ { \theta } ( G _ { \phi } ( z ) ) } _ { \mathrm { P r i o r } } + \underbrace { \log \pi ( z ) - \log \left| \operatorname* { d e t } \frac { d G _ { \phi } ( z ) } { d z } \right| } _ { \mathrm { E n t r o p y } } \right]\tag{3.52}
$$

where the expectation is over the input latent variable $z ,$ and $\pi$ is the reference Gaussian distribution. Observe that the likelihood term and the entropy can be efficiently computed with a single forward/backward pass through the NF due to the parametrization of $q _ { \phi }$ with an NF. All that is left for us is to compute the prior term log $p _ { \pmb { \theta } } ( G _ { \phi } ( \pmb { z } ) )$ . In score prior Feng et al. [18], this is solved by leveraging the instantaneous change-of-variables formula with the diffusion PF-ODE, as originally proposed in Song et al. [2]

$$
\log p _ { \pmb { \theta } } ( \pmb { x } _ { 0 } ) = \log p _ { T } ( \pmb { x } _ { T } ) + \int _ { 0 } ^ { T } \nabla \cdot \tilde { \pmb { f } } _ { \pmb { \theta } } ( \pmb { x } _ { t } , t ) \mathrm { d } t ,\tag{3.53}
$$

where $f _ { \theta } ( x _ { t } , t )$ is the drift term of the reverse SDE in Equation 2.2 with the score replaced by the network approximation. Notice that by plugging in Equation 3.53 to Equation 3.52, we can optimize the NF model in an unsupervised fashion. Notice that while this formulation does not incur approximation errors, it is very costly as every optimization steps involve computing Equation 3.53. Moreover, observe that the training of NF is done for a specific measurement $\mathbf { \pmb { y } } .$ . One has to run Equation 3.52 for every different measurement that one wishes to recover.

## 3.2.4 Efficient Score Prior Feng and Bouman [19]

As computing Equation 3.53 is costly, Feng et al. proposed to optimize $q _ { \phi }$ with the evidence lower bound (ELBO), originally presented in the work of Score-flow Song et al. [152] $b _ { \pmb \theta } ( \pmb x _ { 0 } ) \leq$ log $p _ { \pmb { \theta } } ( \pmb { x } _ { 0 } )$

$$
b _ { \pmb \theta } ( { \pmb x } _ { 0 } ) = \mathbb { E } _ { p ( { \pmb x } _ { T } \mid { \pmb x } _ { 0 } ) } \left[ \log \pi ( { \pmb x } _ { T } ) \right] - \frac { 1 } { 2 } \int _ { 0 } ^ { T } g ( t ) ^ { 2 } h ( t ) \mathrm { d } t ,\tag{3.54}
$$

where

$$
h ( t ) : = \mathbb { E } _ { p ( x _ { t } | X _ { 0 } ) } \Bigg [ \underbrace { \frac { 1 } { \sigma _ { t } ^ { 4 } } \| h _ { \theta } ( x _ { t } ) - x _ { 0 } \| _ { 2 } ^ { 2 } } _ { \mathrm { D e n o i s i n g ~ l o s s } } - \| \nabla _ { x _ { t } } \log p ( x _ { t } | x _ { 0 } ) \| _ { 2 } ^ { 2 } - \frac { 2 } { g ( t ) ^ { 2 } } \nabla _ { x _ { t } } \cdot f ( x _ { t } , t ) \Bigg ] .\tag{3.55}
$$

Intuitively, the value of $b _ { \theta }$ is small when we have a small denoising loss, and large when our diffusion denoiser $h _ { \theta }$ cannot properly denoise the given image. Replacing the exact likelihood Equation 3.53 that requires hundreds to thousands of NFEs to the surrogate denoising likelihood Equation 3.54 that requires only a single NFE makes the method much more efficient and scalable to higher dimensions.

## 3.3 Asymptotically Exact Methods

These methods aim to sample from the true posterior distribution. Of course, the intractability of the posterior distribution cannot be circumvented but what these methods trade compute for ap proximation error: as the number of network evaluations increases to infinity, these methods will asymptotically converge to the true posterior (assuming no other approximation errors).

## 3.3.1 Plug and Play Diffusion Models (PnP-DM) [24]

As explained in the introduction, the end goal is to sample from the distribution $p ( \pmb { x } _ { 0 } | \pmb { y } )$ ∝ $p ( { \pmb x } _ { 0 } ) p ( { \pmb y } | { \pmb x } )$ . The authors of [24] introduce an auxiliary variable z and an auxiliary distribution:

$$
\pi ( \pmb { x } _ { 0 } , z | \pmb { y } ) \propto p ( \pmb { x } _ { 0 } ) \cdot p ( \pmb { y } | z ) \cdot \exp \left( - \frac { 1 } { 2 \rho ^ { 2 } } | | \pmb { x } _ { 0 } - z | | ^ { 2 } \right) .\tag{3.56}
$$

It is easy to see that as $\rho  0 .$ , the auxiliary distribution converges to the target distribution $p ( \pmb { x } _ { 0 } | \pmb { y } )$ To sample from the joint distribution $\pi ( \boldsymbol { x } _ { 0 } , z | \boldsymbol { y } )$ , the authors use Gibbs Sampling, i.e. the alternate between sampling from the posteriors. Specifically, the sampling algorithm alternates between two steps:

• Likelihood term:

$$
z ^ { ( k ) } \sim \pi ( z | \pmb { y } , \pmb { x } _ { 0 } ^ { ( k ) } ) \propto p ( \pmb { y } | z ) \cdot \exp \left( - \frac { 1 } { 2 \rho ^ { 2 } } | | \pmb { x } _ { 0 } ^ { ( k ) } - z | | ^ { 2 } \right) .\tag{3.57}
$$

• Prior term:

$$
\mathbf { } x _ { 0 } ^ { ( k + 1 ) } \sim \pi ( \mathbf { } x _ { 0 } | \pmb { y } , z ^ { ( k ) } ) \propto p ( \pmb { x } _ { 0 } ) \cdot \mathrm { e x p } \left( - \frac { 1 } { 2 \rho ^ { 2 } } | | \pmb { x } _ { 0 } ^ { ( k ) } - z ^ { ( k ) } | | ^ { 2 } \right) .\tag{3.58}
$$

The likelihood term samples a vector that satisfies the measurements and is close to $\pmb { x } _ { 0 } ^ { ( k ) }$ . The prior term samples a vector that is likely under $p ( \pmb { x } _ { 0 } )$ and is close to $z ^ { ( k ) }$ . For most problems of interest, sampling from Equation 3.57 is easy because the distribution is log-concave, e.g. that’s the case for linear inverse problems. The interesting observation is that sampling from Equation 3.58 corresponds to a denoising problem, for which diffusion models excel. Indeed, for any $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ at noise level $\sigma _ { t } ,$ we have that:

$$
p ( { \pmb x } _ { 0 } | { \pmb x } _ { t } ) \propto p ( { \pmb x } _ { 0 } ) p ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) = p ( { \pmb x } _ { 0 } ) \mathrm { e x p } \left( - \frac { 1 } { 2 \sigma _ { t } ^ { 2 } } | | { \pmb x } _ { 0 } - { \pmb x } _ { t } | | ^ { 2 } \right) .\tag{3.59}
$$

Hence, to sample from Equation 3.58, one initializes the reverse process at $z ^ { ( k ) }$ and time t such that: $\sigma _ { t } = \rho .$

## 3.3.2 FPS Dou and Song [25]

FPS connects posterior sampling to Bayesian filtering and uses sequential Monte Carlo methods to solve the filtering problem, avoiding the need to handcraft approximations to the posterior $p ( \pmb { y } | \pmb { x } _ { t } )$ Given an observation y, FPS proposes to first construct a sequence $\{ y _ { t } \} _ { t = 0 } ^ { N }$ from y, and then determine a tractable distribution $p ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { y } _ { t - 1 } )$ . Starting from $\mathbf { x } _ { N } \sim \mathcal { N } ( \mathbf { 0 } , I _ { n } )$ , FPS can then recursively sample $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ for $t = N - 1 , \ldots , 1$ , and finally obtain $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ . Specifically, FPS consists of two steps:

Step 1. Generating a sequence of $\{ \boldsymbol { y } _ { t } \} _ { t = 0 } ^ { N }$ with an observation $\mathbf { \nabla } _ { \mathbf { \nabla } _ { y } } .$ This can be done either using the forward process or unconditional DDIM backward sampling.

For the construction via the forward process, we recursively construct ${ \mathbf { } } _ { \mathbf { } } \mathbf { \psi } _ { \mathbf { } } \mathbf { _ { } } \mathbf { \psi } _ { \mathbf { } } \mathbf { _ { } } \mathbf { \psi } _ { \mathbf { } } \mathbf { _ { } } \mathbf { \psi } _ { \mathbf { } } \mathbf { _ { } } \mathbf { \psi } _ { \mathbf { } } \mathbf { _ { } } \mathbf { \psi } _ { \mathbf { } } \mathbf { _ { } } \mathbf { \psi } _ { \mathbf { } \psi } \mathbf { _ { } } \textbf { } \psi _ { } \psi _ { } \left. \textbf { } \psi _ { } \mathbf { } \psi _ { } \textbf { } \right.$ as follows:

$$
y _ { t } = y _ { t - 1 } + \sigma _ { t } A z _ { t } , \quad { \mathrm { i n i t i a l i z e d ~ w i t h ~ } } y _ { 0 } : = y .\tag{3.60}
$$

This arises from ${ \pmb x } _ { t } = { \pmb x } _ { t - 1 } + \sigma _ { t } { \pmb z } _ { t }$ and applying the linear operator A to it.

For the construction via backward sampling, FPS uses methods such as unconditional DDIM as in Equation 2.9,

$$
y _ { t - 1 } = \underbrace { u _ { t } y _ { 0 } } _ { \mathrm { c l e a n } } + \underbrace { v _ { t } y _ { t } } _ { \mathrm { d i r e c t i o n ~ t o ~ t i m e } ~ t \mathrm { s a m p l e } } + \underbrace { w _ { t } A z _ { t } } _ { \mathrm { n o i s e } } , \quad \mathrm { i n i t i a l i z e d ~ w i t h } ~ y _ { N } \sim \mathcal { N } ( \mathbf { 0 } , A A ^ { \top } ) .\tag{3.61}
$$

Here, $u _ { t } , \ v _ { t } .$ , and $w _ { t }$ are DDIM coefficients that can be explicitly computed. Note that $\mathbf { \nabla } _ { \mathbf { \boldsymbol { y } } \mathrm { { } } N }$ is sampled from $\mathcal { N } ( \mathbf { 0 } , A A ^ { \top } )$ because the prior distribution of the diffusion model is a standard Gaussian $\pmb { x } _ { N } \sim \mathcal { N } ( \mathbf { 0 } , I )$ , and due to the linearity of the inverse problem, ${ \bf { } } _ { { \bf { } } ^ { g } N } =$ $A { \pmb x } _ { N }$

Step 2. Generating a backward sequence of $\{ \pmb { x } _ { t } \} _ { t = 0 } ^ { N }$ from Step 1’s $\{ y _ { t } \} _ { t = 0 } ^ { N } .$ First, note that $p ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { y } _ { t - 1 } )$ is a tractable normal distribution. This results from applying Bayes’ rule and the conditional independence of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and the random vector $Y _ { t - 1 } \mathrm { g i v e n } x _ { t - 1 } .$

$$
p ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { Y } _ { t - 1 } ) \propto p ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) p ( \pmb { Y } _ { t - 1 } | \pmb { x } _ { t - 1 } ) .\tag{3.62}
$$

Here, $p ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ is approximated via backward diffusion sampling with learned scores, and $p ( \dot { \mathbfcal { Y } } _ { t - 1 } | \dot { \mathbf { x } } _ { t - 1 } ) = \bar { \mathcal { N } } ( A \mathbfit { x } _ { t - 1 } , c _ { t - 1 } ^ { 2 } I )$ , where $c _ { t - 1 }$ , dependent on $\sigma _ { y } > 0$ , can be computed explicitly [47]. Thus, with $\{ y _ { t } \} _ { t = 0 } ^ { N }$ and initial condition x $\mathbf { \Omega } _ { N } \ \sim \ { \mathcal { N } } ( \mathbf { 0 } , I _ { n } )$ , FPS recursively samples $\pmb { x } _ { N - 1 } , \cdots \pmb { x } _ { 1 }$ using $p ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { Y } _ { t - 1 } = \pmb { y } _ { t - 1 } )$ , ultimately yielding $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$

FPS algorithm is theoretically supported to recover the oracle $p ( { \pmb x } | { \pmb y } )$ once the step size is sufficiently small.

## 3.3.3 PMC Sun et al. [26]

Plug-and-Play (PnP) Kamilov et al. [110] and RED Romano et al. [107] are two representative methods of using denoisers as priors for solving inverse problems. Let $\begin{array} { r } { g _ { \pmb { y } } ( \pmb { \bar { x } } ) = \frac { 1 } { 2 \sigma _ { \ b { u } } ^ { 2 } } \lVert \pmb { \bar { y } } - A \pmb { x } \rVert _ { 2 } ^ { 2 } } \end{array}$ be the log-likelihood function, $h _ { \theta } ^ { \sigma } ( \cdot )$ an MMSE denoiser from Equation 2.11 conditioned on the noise level $\sigma ,$ , and $R _ { \pmb \theta } ^ { \sigma } ( \cdot ) : = \mathrm { I d } - h _ { \pmb \theta } ^ { \sigma } ( \cdot )$ the residual projector. Note that conditioning on the noise level $\sigma$ is equivalent to the network being conditioned no t, since the mapping is one-to-one. A single iteration of these methods read

• PnP proximal gradient method Kamilov et al. [153]:

$$
\pmb { x } _ { k + 1 } = h _ { \theta } ^ { \sigma } ( \pmb { x } _ { k } - \gamma \nabla _ { \pmb { x } _ { k } } g _ { \pmb { y } } ( \pmb { x } _ { k } ) )\tag{3.63}
$$

$$
= x _ { k } - \gamma \left( \nabla _ { x _ { k } } g _ { y } ( x _ { k } ) + \frac { 1 } { \gamma } R _ { \theta } ^ { \sigma } \left( x _ { k } - \gamma \nabla _ { x _ { k } } g _ { y } ( x _ { k } ) \right) \right) .\tag{3.64}
$$

• RED gradient descent Romano et al. [107]:

$$
{ \pmb x } _ { k + 1 } = { \pmb x } _ { k } - \gamma \left( \nabla _ { { \pmb x } _ { k } } g _ { y } ( { \pmb x } _ { k } ) + \tau \left( { \pmb x } _ { k } - { \pmb h } _ { \theta } ^ { \sigma } ( { \pmb x } _ { k } ) \right) \right)\tag{3.65}
$$

$$
= \pmb { x } _ { k } - \gamma \left( \nabla _ { \pmb { x } _ { k } } g _ { \pmb { y } } ( \pmb { x } _ { k } ) + \tau R _ { \pmb { \theta } } ^ { \sigma } ( \pmb { x } _ { k } ) \right) .\tag{3.66}
$$

Notice that by using Tweedie’s formula, we see that $R _ { \theta } ^ { \sigma } ( { \pmb x } ) = - \sigma ^ { 2 } \nabla _ { \pmb x } \log p _ { \sigma } ( { \pmb x } )$ . Rearranging Equation 3.64 and Equation 3.66,

$$
\begin{array} { r l r } { \bullet \mathrm { ~ P n P : ~ } } & { } & \\ { \quad \displaystyle \frac { x _ { k + 1 } - x _ { k } } { \gamma } = - P ( { \boldsymbol x } _ { k } ) , \quad P ( { \boldsymbol x } ) : = \nabla _ { \boldsymbol x } g _ { \boldsymbol y } ( { \boldsymbol x } ) - \frac { \sigma ^ { 2 } } { \gamma } \nabla _ { \boldsymbol x } \log p _ { \sigma } ( { \boldsymbol x } - \gamma \nabla _ { \boldsymbol x } g _ { \boldsymbol y } ( { \boldsymbol x } ) ) , } \end{array}\tag{3.67}
$$

• RED:

$$
\frac { \pmb x _ { k + 1 } - \pmb x _ { k } } { \gamma } = - G ( \pmb x _ { k } ) , \quad G ( \pmb x ) : = \nabla _ { \pmb x } g _ { \pmb y } ( \pmb x ) - \tau \sigma ^ { 2 } \nabla _ { \pmb x } \log p _ { \sigma } ( \pmb x ) .\tag{3.68}
$$

Moreover, by setting $\gamma = \sigma ^ { 2 }$ and $\tau = 1 / \sigma ^ { 2 }$ , one can show that

$$
\begin{array} { l } { \displaystyle \operatorname* { l i m } _ { \sigma \to 0 } P ( \pmb { x } ) = \nabla _ { \pmb { x } } g _ { y } ( \pmb { x } ) - \operatorname* { l i m } _ { \sigma \to 0 } \{ \nabla _ { \pmb { x } } \log p ( \pmb { x } - \sigma ^ { 2 } \nabla _ { \pmb { x } _ { k } } g _ { y } ( \pmb { x } _ { k } ) ) \} } \\ { = \nabla _ { \pmb { x } } g _ { y } ( \pmb { x } ) - \operatorname* { l i m } _ { \sigma \to 0 } \{ \operatorname* { l i m } _ { \sigma \to 0 } \log p _ { \sigma } ( \pmb { x } ) \} = \displaystyle \operatorname* { l i m } _ { \sigma \to 0 } G ( \pmb { x } ) } \\ { = - \nabla _ { \pmb { x } } \log p ( \pmb { y } | \pmb { x } ) - \nabla _ { \pmb { x } } \log p ( \pmb { x } ) = - \nabla _ { \pmb { x } } \log p ( \pmb { x } | \pmb { y } ) . } \end{array}\tag{3.69}
$$

In other words, we see that the iteration of PnP/RED in Equation 3.64 and Equation 3.66 will converge to sampling from the posterior as $\sigma ^ { 2 } = \gamma  0$

$$
\mathrm { d } \pmb { x } _ { t } = \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } | \pmb { y } ) \mathrm { d } t ,\tag{3.70}
$$

where t indexes the continuous time flow of $^ { \mathbf { \delta x } , }$ as opposed to the discrete formulations in Equation 3.64 and Equation 3.66. Note that this notion of t does not match the diffusion time t, where the time index matches a specific noise level. In PMC, the authors propose to incorporate noise level annealing as done in the usual reverse diffusion process by starting from a large noise level $\sigma$ and gradually reducing the noise level. Solving Equation 3.70 with PMC then boils down to iterative application of Equation 3.64 and Equation 3.66 with the annealing strategy. Moreover, introducing Langevin diffusion yields a stochastic version

$$
\mathrm { d } \pmb { x } _ { t } = \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } | \pmb { y } ) \mathrm { d } t + \sqrt { 2 } \mathrm { d } \pmb { W } _ { t } ,\tag{3.71}
$$

which can be solved in the same way, but with additional stochasticity.

## 3.3.4 Sequential Monte Carlo-based methods

SMCDiff Trippe et al. [27], MCGDiff Cardoso et al. [28], and TDS Wu et al. [29] belong to the category of sequential Monte Carlo (SMC)-based methods Doucet et al. [154]. SMC aims to sample from the posterior by constructing a sequence of distributions $X _ { 1 : T }$ , which terminates at the target distribution. The evolution of the distribution is approximated by K particles. In a high level, SMC can be described with three steps: 1) Transition with a proposal kernel $\{ \pmb { x } _ { t } ^ { 1 : K } \} \sim p ( \mathbf { \breve { X } } _ { t } | \mathbf { X } _ { t - 1 } ) , 2 )$ computing the weights to re-weight the importance, and 3) resampling from a reweighted multinomial distribution. Methods that belong to this category propose different ways of constructing the proposal distribution and the weighting function.

## 3.4 CSGM-Type methods

## 3.4.1 DMPlug [20], SHRED [21]

Compressed sensing generative model (CSGM) [155, 156] is a general method for solving inverse problems with deep generative models by aiming to find the input latent vector z through

$$
z ^ { * } = \arg \operatorname* { m i n } _ { z } \| y - A G _ { \theta } ( z ) \| ^ { 2 } ,\tag{3.72}
$$

where $G _ { \theta }$ is an arbitrary generative model. DMPlug and SHRED can be seen as extensions of CSGM to the case where one uses a diffusion model. Unlike GANs or Flows where the mapping from the latent space to the image space is done through a single NFE, diffusion models require multiple NFE to solve the generative SDE/ODE. One can rewrite Equation 3.72 as

$$
z ^ { * } = \underset { z } { \arg \operatorname* { m i n } } \left\| y - A \hat { { \pmb x } } ( z ) \right\| ^ { 2 } ,\tag{3.73}
$$

where ${ \hat { \mathbf { x } } } = { \hat { \mathbf { x } } } ( z )$ is the solution of the deterministic sampler initialized at z. Essentially, the models in this category optimize over the “latent” space of noises that are fed to the deterministic ODE sampler. One caveat of Equation 3.73 is the exploding memory required for backpropagation through time. To mitigate this, when sampling from $p _ { \pmb { \theta } } ( \pmb { x } _ { 0 } | \pmb { x } _ { T } )$ , a few-step sampling (e.g. 3 for DMPlug and 10 for SHRED) is used to approximate the true sampling process.

## 3.4.2 CSGM with consistent diffusion models [22]

Diffusion models can be distilled into one-step models, known as Consistency Models [157], that solve in one step the Probability Flow ODE. These models can be used in Equation 3.73, replacing the ODE sampling, to reduce the computational requirements [22].

## 3.4.3 Intermediate Layer Optimization [156, 23]

CSGM has been extended to perform the optimization in some intermediate latent space [156]. The problem is that the intermediate latents need to be regularized to avoid exiting the manifold of realistic images. Score-Guided Intermediate Layer Optimization (Score-ILO) [23] uses diffusion models to regularize the intermediate solutions.

## 3.5 Latent Diffusion Models

## 3.5.1 Motivation

In this subsection, we focus on algorithms that have been developed for solving inverse problems with latent diffusion models (see Section 2.3). There are a few additional challenges when dealing with latent diffusion models that have led to a growing literature of papers that are trying to address them.

Loss of linearity. The first challenge in solving inverse problems with latent diffusion models is that linear inverse problems become essentially non-linear. The problem stems from the fact that diffusion happens in the latent space but measurements are in the pixel-space. In order to guide the diffusion there are two potential solutions: i) either project the measurements to the latent space through the encoder, or, ii) project the latents to the pixel space as we diffuse through the decoder. Both approaches depend on non-linear functions (Enc, Dec respectively) and hence even linear inverse problems need a more general treatment.

Decoding is expensive. The other issue that arises is computational. Most of the time, we need to decode the latent to pixel-space to compare with the measurements. The motivation behind latent diffusion models is to accelerate training and sampling. Hence, we want to avoid repeated calls to the decoder as we solve inverse problems.

Decoding-encoding map is not one-to-one. Even if we ignore the computational challenges, it is not straightforward to decode the latent to the pixel-space, compare with the measurements and get meaningful guidance in the latent space since the decoding-encoding map is not an one-to-one function.

Text-conditioning. Finally, latent diffusion models typically get a textual prompt as an additional input. A lot of algorithms that have been developed in the space of using latent diffusion models to solve inverse problems innovate on how they use text conditioning.

## 3.5.2 Latent DPS

The first algorithm we review in the space of solving inverse problems with latent diffusion models is Latent DPS, i.e. the straightforward extension of DPS for latent diffusion models. The approximation made in this algorithm is:

$$
\nabla _ { { \boldsymbol x } _ { t } ^ { \mathrm { E } } } \log p ( { \boldsymbol y } | \boldsymbol X _ { t } ^ { \mathrm { E } } = { \boldsymbol x } _ { t } ^ { \mathrm { E } } ) \approx \nabla _ { { \boldsymbol x } _ { t } ^ { \mathrm { E } } } \log p ( { \boldsymbol y } | \boldsymbol X _ { 0 } = \mathrm { D e c } ( \mathbb E [ { \boldsymbol X } _ { 0 } ^ { \mathrm { E } } | \boldsymbol X _ { t } ^ { \mathrm { E } } = { \boldsymbol x } _ { t } ^ { \mathrm { E } } ] ) ) .\tag{3.74}
$$

The algorithm works by performing one-step denoising in the latent space and measuring how much the decoding of the denoised latent matches the measurements y.

## 3.5.3 PSLD Rout et al. [14]

The performance of Latent DPS is hindered by the fact that the decoding-encoding map is not an one-to-one function, as discussed earlier. The approximation made above could pull $\mathbf { \bar { x } } _ { t } ^ { \mathrm { E } }$ towards any latent $\boldsymbol { x } _ { 0 } ^ { \mathrm { E } }$ that has a decoding that matches the measurements while the score function is pulling $\pmb { x } _ { t } ^ { \check { \mathrm { E } } }$ towards a specific $\boldsymbol { x } _ { 0 } ^ { \mathrm { E } }$ , i.e. towards $\mathbb { E } [ \pmb { x } _ { 0 } ^ { \mathrm { E } } | \pmb { x } _ { t } ^ { \mathrm { E } } ]$

PSLD mitigates this problem by adding an additional term that pulls towards latents that are fixed points of the decoder-encoder map. Concretely, the approximation made in PSLD is:

$$
\begin{array} { r l r } & { } & { \nabla _ { \pmb { x } _ { t } ^ { \mathrm { E } } } \log p ( \pmb { y } | \pmb { X } _ { t } ^ { \mathrm { E } } = \pmb { x } _ { t } ^ { \mathrm { E } } ) \approx } \\ & { } & { \nabla _ { \pmb { x } _ { t } ^ { \mathrm { E } } } \log p ( \pmb { y } | ( \pmb { X } _ { 0 } = \mathrm { D e c } ( \mathbb { E } [ \pmb { X } _ { 0 } ^ { \mathrm { E } } | \pmb { X } _ { t } ^ { \mathrm { E } } = \pmb { x } _ { t } ^ { \mathrm { E } } ] ) ) } \\ & { } & { + \gamma _ { t } \nabla _ { \pmb { x } _ { t } ^ { \mathrm { E } } } \left| \left| \mathbb { E } [ \pmb { x } _ { 0 } ^ { \mathrm { E } } | \pmb { x } _ { t } ^ { \mathrm { E } } ] - \mathrm { E n c } ( \mathrm { D e c } ( \mathbb { E } [ \pmb { x } _ { 0 } ^ { \mathrm { E } } | \pmb { x } _ { t } ^ { \mathrm { E } } ] ) ) \right| \right| ^ { 2 } , } \end{array}\tag{3.75}
$$

where $\gamma _ { t }$ is a tunable parameter.

## 3.5.4 Resample Song et al. [32]

Resample, a concurrent work with PSLD, proposes an alternative way to improve the performance of Latent DPS. After each clean prediction $\widehat { \pmb { x } } _ { 0 } ( \pmb { x } _ { t + 1 } ^ { \mathrm { E } } )$ is obtained from the previous sample $\mathbf { \boldsymbol { x } } _ { t + } ^ { \mathrm { E } }$ 1 via Tweedie’s formula in Equation 2.10, and the unconditional reverse denoising process is updated using, say, DDIM:

$$
\pmb { x } _ { t } ^ { \prime } : = \mathrm { U n c o n d i t i o n a l D D I M } \big ( \widehat { \pmb { x } } _ { 0 } ( \pmb { x } _ { t + 1 } ^ { \mathrm { E } } ) , \pmb { x } _ { t + 1 } ^ { \mathrm { E } } \big )\tag{3.76}
$$

the authors project the latent back to a point $\widehat { \pmb { x } } _ { t }$ that satisfies measurements using:

$$
\mathcal { N } \bigg ( \widehat { x } _ { t } ; \frac { \sigma _ { t } ^ { 2 } \sqrt { \bar { \alpha } _ { t } } \widehat { x } _ { 0 } ( y ) + ( 1 - \bar { \alpha } _ { t } ) { \pmb x } _ { t } ^ { \prime } } { \sigma _ { t } ^ { 2 } + \big ( 1 - \bar { \alpha } _ { t } \big ) } , \frac { \sigma _ { t } ^ { 2 } \big ( 1 - \bar { \alpha } _ { t } \big ) } { \sigma _ { t } ^ { 2 } + \big ( 1 - \bar { \alpha } _ { t } \big ) } I _ { k } \bigg ) .\tag{3.77}
$$

Here, $\sigma _ { t } ^ { 2 }$ is a hyperparameter used to tune the alignment with measurements, $\bar { \alpha } _ { t }$ is predefined in forward process, and $\widehat { \pmb { x } } _ { 0 } ( \pmb { y } )$ is found by solving:

$$
\widehat { \pmb x } _ { 0 } ( \pmb y ) \in \arg \operatorname* { m i n } _ { \pmb x } \frac { 1 } { 2 } \left| | \pmb y - \pmb A ( \mathrm { D e c } ( \pmb x ) ) | \right| _ { 2 } ^ { 2 } \quad \mathrm { i n i t i a l i z e d \ a t \ } \widehat x _ { 0 } ( \pmb x _ { t + 1 } ^ { \mathrm { E } } ) .\tag{3.78}
$$

## 3.6 MPGD He et al. [158]

The MPGD authors note that some methods require expensive computations for measurement alignment during gradient updates, as they involve passing through the gradient (chain rule) of the pretrained diffusion model $\epsilon _ { \theta } ( x _ { t } ^ { \mathrm { E } } , t )$

$$
\begin{array} { r } { \pmb { x } _ { t } ^ { \mathrm { E } }  \pmb { x } _ { t } ^ { \mathrm { E } } - \eta _ { t } \nabla _ { \pmb { x } _ { t } ^ { \mathrm { E } } } | | \pmb { y } - \pmb { \mathcal { A } } \big ( \mathrm { D e c } ( \pmb { x } _ { 0 | t } ) \big ) | | _ { 2 } ^ { 2 } , } \end{array}\tag{3.79}
$$

where $\begin{array} { r } { \mathbf { \Delta x } _ { 0 \mid t } : = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \big ( \mathbf { \Delta x } _ { t } ^ { \mathrm { E } } - \sqrt { 1 - \bar { \alpha } _ { t } } \mathbf { \epsilon } _ { \theta } ( \mathbf { \Delta x } _ { t } ^ { \mathrm { E } } , t ) \big ) } \end{array}$ is a clean estimation via Tweedie’s formula in Equation 2.10. This gradient bottleneck slows down the overall inverse problem solving. MPGD proposes bypassing the direct gradient $\nabla _ { \pmb { x } _ { t } ^ { \mathrm { E } } }$ with theoretical guarantees by updating with $\nabla _ { \pmb { x } _ { 0 | t } }$

$$
\begin{array} { r } { \pmb { x } _ { 0 \vert t } ^ { \prime }  \pmb { x } _ { 0 \vert t } - \eta _ { t } \nabla _ { \pmb { x } _ { 0 \vert t } } \big \vert \big \vert \pmb { y } - \pmb { \mathcal { A } } \big ( \mathrm { D e c } ( \pmb { x } _ { 0 \vert t } ) \big ) \big \vert \big \vert _ { 2 } ^ { 2 } } \end{array}\tag{3.80}
$$

with

$$
\pmb { x } _ { 0 \vert t } : = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \big ( \pmb { x } _ { t } ^ { \mathrm { E } } - \sqrt { 1 - \bar { \alpha } _ { t } } \pmb { \epsilon } _ { \theta } ( \pmb { x } _ { t } ^ { \mathrm { E } } , t ) \big ) ,\tag{3.81}
$$

and use the obtained $\pmb { x } _ { 0 | t } ^ { \prime }$ for unconditional reverse denoising process

$$
\pmb { x } _ { t - 1 } ^ { \prime } : = \mathrm { U n c o n d i t i o n a l D D I M } \big ( \pmb { x } _ { 0 | t } ^ { \prime } , \pmb { x } _ { t } ^ { \mathrm { E } } \big ) .\tag{3.82}
$$

## 3.6.1 P2L [34]

While text conditioning is a viable option for modern latent diffusion models such as Stable diffusion, the actual use was underexplored due to ambiguities on which text to use. P2L addresses this question by proposing an algorithm that optimizes for the text embedding on the fly while solving an inverse problem.

$$
\displaystyle c _ { t } ^ { * } = \operatorname * { a r g m i n } _ { c } \| y - A \mathrm { D e c } ( \mathbb { E } [ \mathbf { x } _ { 0 } ^ { \mathrm { E } } | \mathbf { x } _ { t } ^ { \mathrm { E } } , c ] ) \| ^ { 2 } ,\tag{3.83}
$$

where c is the text embedding, and one can approximate $\mathbb { E } [ \pmb { x } _ { 0 } ^ { \mathrm { E } } | \pmb { x } _ { t } ^ { \mathrm { E } } , \pmb { c } ]$ by using the Tweedie’s formula with the denoiser conditioned on c. Using the optimized embedding at each timestep $\boldsymbol { c } _ { t } ^ { * }$ , sampling follows the procedure of Latent DPS

$$
\nabla _ { \mathbf { x } _ { t } ^ { \mathrm { E } } } \log p ( \pmb { y } | \pmb { x } _ { t } ^ { \mathrm { E } } = \pmb { x } _ { t } ^ { \mathrm { E } } , \pmb { c } ) \approx \nabla _ { \mathbf { x } _ { t } ^ { \mathrm { E } } } \log p ( \pmb { y } | \pmb { X } _ { 0 } = \mathrm { D e c } ( \mathbb { E } [ \pmb { x } _ { 0 } ^ { \mathrm { E } } | \pmb { x } _ { t } ^ { \mathrm { E } } = \pmb { x } _ { t } ^ { \mathrm { E } } , \pmb { c } _ { t } ^ { * } ] ) )\tag{3.84}
$$

In addition to the optimization of the text embedding, P2L further tries to leverage the VAE prior by decoding - running optimization in the pixel space - re-encoding

$$
\pmb { x } ^ { * } = \underset { \pmb { x } } { \arg \operatorname* { m i n } } \| \pmb { y } - \pmb { A } \pmb { x } \| _ { 2 } ^ { 2 } + \lambda \| \pmb { x } - \mathrm { D e c } ( \mathbb { E } [ \pmb { x } _ { 0 } ^ { \mathrm { E } } | \pmb { x } _ { t } ^ { \mathrm { E } } = \pmb { x } _ { t } ^ { \mathrm { E } } ] ) \| _ { 2 } ^ { 2 }\tag{3.85}
$$

$$
{ \pmb x } ^ { \mathrm { E } } = \mathrm { E n c } ( { \pmb x } ^ { * } )\tag{3.86}
$$

## 3.6.2 TReg [35], DreamSampler [36]

Instead of automatically finding a suitable text embedding to achieve maximal reconstructive performance, another advantage of text conditioning is that it can be used as an additional guiding signal to lead to a specific mode. This may seem trivial, as one has access to a conditional diffusion model. However, in practice, simply using a conditional diffusion model does not induce enough guidance as reported in [159, 160], and naively using classifier free guidance [160] (CFG) does not lead to satisfactory results. In addition to using data consistency imposing steps as in P2L, TReg proposes adaptive negation to update the null text embeddings used for CFG guidance.

$$
\pmb { c } _ { \mathcal { O } } ^ { * } = \arg \operatorname* { m i n } _ { \pmb { c } } \sin ( \mathcal { T } ( \pmb { x } ^ { * } ) , \pmb { c } ) ,\tag{3.87}
$$

where $\mathbf { \boldsymbol { x } } ^ { * }$ comes from Equation 3.85, sim denotes the CLIP similarity [161] score, and  is the CLIP image encoder. In essence, Equation 3.87 minimizes the similarity between the current estimate of the image and the null text embedding. Hence, when the optimized $c _ { \mathcal { O } }$ is used for CFG with

$$
\epsilon _ { \theta } ( \boldsymbol { x } _ { t } ^ { \mathrm { E } } , \boldsymbol { c } _ { \mathcal { O } } ^ { * } ) + \omega \left( \epsilon _ { \theta } ( \boldsymbol { x } _ { t } ^ { \mathrm { E } } , \boldsymbol { c } ) - \epsilon _ { \theta } ( \boldsymbol { x } _ { t } ^ { \mathrm { E } } , \boldsymbol { c } _ { \mathcal { O } } ^ { * } ) , \right)\tag{3.88}
$$

the conditioning vector direction ${ \epsilon _ { \theta } } ( x _ { t } ^ { \mathrm { E } } , c ) - \epsilon _ { \theta } ( x _ { t } ^ { \mathrm { E } } , c _ { \emptyset } ^ { * } )$ is amplified. Later, TReg was further advanced by devising a way to better make use of CFG by combining score distillation sampling Poole et al. [162] into the sampling framework.

## 3.6.3 STSL Rout et al. [15]

Most methods leverage the mean of the reverse diffusion distribution $p ( { X } _ { 0 } | { X } _ { t } )$ , and take a single gradient step with Equation 3.74. To further leverage the covariance of $p ( { X } _ { 0 } | { X } _ { t } )$ , Rout et al. Rout et al. [15] propose to use the following fidelity loss

$$
\begin{array} { r } { \mathcal { L } ( { \boldsymbol x } _ { t } ^ { \mathrm { E } } , { \boldsymbol y } ) = \nabla _ { { \boldsymbol x } _ { t } ^ { \mathrm { E } } } \log p ( { \boldsymbol y } | \mathbf { { X } } _ { 0 } = \mathrm { D e c } ( \mathbb { E } [ { \boldsymbol x } _ { 0 } ^ { \mathrm { E } } | { \boldsymbol x } _ { t } ^ { \mathrm { E } } = { \boldsymbol x } _ { t } ^ { \mathrm { E } } ] ) ) } \\ { + \gamma \nabla _ { { \boldsymbol x } _ { t } ^ { \mathrm { E } } } \left( \mathrm { T r a c e } \left( \nabla _ { { \boldsymbol x } _ { t } ^ { \mathrm { E } } } ^ { 2 } \log p ( { \boldsymbol x } _ { t } ^ { \mathrm { E } } ) \right) \right) , } \end{array}\tag{3.89}
$$

where $\gamma$ is a constant. To effectively compute the trace, one can further use the following approximation

$$
\begin{array} { r } { \mathrm { T r a c e } \left( \nabla _ { x _ { t } ^ { \mathrm { E } } } ^ { 2 } \log p ( x _ { t } ^ { \mathrm { E } } ) \right) \approx \mathbb { E } _ { \epsilon \sim \pi } \left[ \epsilon ^ { \top } \left( \nabla _ { x _ { t } ^ { \mathrm { E } } } \log p ( x _ { t } ^ { \mathrm { E } } + \epsilon ) - \nabla _ { x _ { t } ^ { \mathrm { E } } } \log p ( x _ { t } ^ { \mathrm { E } } ) \right) \right] , } \end{array}\tag{3.90}
$$

where $\pi$ can be a Gaussian or a Rademacher distribution. Using the loss in Equation 3.89 with Equation 3.90, STSL uses multiple steps of stochastic gradient updates per timestep.

## 4 Thoughts from the authors

In the previous section, we presented several works in the space of using diffusion models to solve inverse problems. A natural question that both experts and newcomers to the field might have is, eventually,: “which approach works the best?”. Unfortunately, we cannot provide a conclusive answer to this question within the scope of this survey, but we can share a few thoughts.

Thoughts about Explicit Approximations. In this survey we tried to express seemingly very different works, such as DPS and DDRM, under a common mathematical language that contains the explicit approximations made for the measurements score. We observed that all the methods compute an error metric that matches consistency with the measurement and then lift the error back to the image space dimensions to perform the gradient update. Some of the methods used noised versions of the measurements to compute the error while others use the clean measurements. To the best of our knowledge, it is not clear which one works the best and one can derive new approximation algorithms by simply making the dual change to any of the methods that already exist, e.g. one can propose Score-ALD++ by using the noisy measurements to compute the error. By looking at Figure 1, it is also evident that methods propose increasingly more complex “lifting” matrices. Some of these approximations require increased computation, e.g. the Moments Matching method. We strongly believe that the field would benefit from a standardized benchmark for diffusion models and inverse problems to understand better the computational performance trade-offs of different methods. We also believe that under certain distributional assumptions, it should be possible to characterize analytically the propagation of the approximation errors induced by the different methods.

Thoughts about Variational Methods. Variational Methods try to estimate the parameters of a simpler distribution. The benefit here is that one can employ well-known optimization techniques to better solve the optimization problem at hand. A potential drawback of this approach is that the proposed distribution might not be able to capture the complexity of the real posterior distribution.

Thoughts about CSGM-type Methods. CSGM-type frameworks can benefit from the plethora of techniques that have been previously developed to solve inverse problems with GANs and other deep generative modeling frameworks. The main issue here is computational since the generative model to be inverted here is the Probability Flow ODE mapping that requires several calls to the diffusion model. Consistency Models [157, 163] and other approaches such as Intermediate Layer Optimization could mitigate this issue.

Thoughts about Asymptotically Exact Methods. Asymptotically Exact Methods, usually based on Monte Carlo, could be useful when sampling from the true posterior is really important. However, the theoretical guarantees of these methods only hold under the setting of infinite computation and it remains to be seen if they can scale to more practical settings.

## 5 Conclusion

In this survey, we discussed different types of inverse problems and different approaches that have been developed to solve them using diffusion priors. We identified four distinct families: methods that propose explicit approximations for the measurement score, variational inference methods, CSGM-type frameworks and finally approaches that asymptotically guarantee exact sampling (at the cost of increased computation). The different frameworks and the works therein are all trying to address the fundamental problem of the intractability of the posterior distribution. In this survey, we tried to unify seemingly different approaches and explain the trade-offs of different methods. We hope that this survey will serve as a reference point for the vibrant field of diffusion models for inverse problems.

## Acknowledgments

This research has been supported by NSF Grants AF 1901292, CNS 2148141, Tripods CCF 1934932, IFML CCF 2019844 and research gifts by Western Digital, Amazon, WNCG IAP, UT Austin Machine Learning Lab (MLL), Cisco and the Stanly P. Finch Centennial Professorship in Engineering. Giannis Daras has been supported by the Onassis Fellowship (Scholarship ID: F ZS 012-1/2022- 2023), the Bodossaki Fellowship and the Leventis Fellowship. The authors would like to thank our colleagues Viraj Shah, Miki Rubinstein, Murata Naoki, Yutong He, and Stefano Ermon for helpful discussions.

## A Proofs

Lemma A.1 (Conditional Expectation and MMSE). Let $X _ { 0 }$ and $X _ { t }$ be two random variables, and $h _ { \theta } ( x _ { t } , t )$ be a function parameterized by θ. Then:

$$
a r g m i n _ { \theta } \mathbb { E } \left[ \left| \left| h _ { \theta } ( \mathbf { x } _ { t } , t ) - \mathbf { x } _ { 0 } \right| \right| ^ { 2 } \right] = a r g m i n _ { \theta } \mathbb { E } \left[ \left| \left| h _ { \theta } ( \mathbf { x } _ { t } , t ) - \mathbb { E } [ \mathbf { x } _ { 0 } | \mathbf { x } _ { t } ] \right| \right| ^ { 2 } \right]\tag{A.1}
$$

That is, the function $h _ { \theta } ( x _ { t } , t )$ that minimizes the mean squared error with respect to $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ is the one that best approximates the conditional expectation $\mathbb { E } [ { \pmb x } _ { 0 } | { \pmb x } _ { t } ]$ ].

Proof.

$$
\mathrm { a r g m i n } _ { \theta } \mathbb { E } \left[ | | h _ { \theta } ( x _ { t } , t ) - x _ { 0 } | | ^ { 2 } \right]\tag{A.2}
$$

$$
= \operatorname { a r g m i n } _ { \theta } \mathbb { E } \left[ | | h _ { \theta } ( x _ { t } , t ) - \mathbb { E } [ x _ { 0 } | x _ { t } ] + \mathbb { E } [ x _ { 0 } | x _ { t } ] - x _ { 0 } | | ^ { 2 } \right]\tag{A.3}
$$

$$
= \mathrm { a r g m i n } _ { \theta } \mathbb { E } \bigg [ | | h _ { \theta } ( x _ { t } , t ) - \mathbb { E } [ x _ { 0 } | x _ { t } ] | | ^ { 2 } - 2 \big ( h _ { \theta } ( x _ { t } , t ) - \mathbb { E } [ x _ { 0 } | x _ { t } ] \big ) ^ { \top } \big ( x _ { 0 } - \mathbb { E } [ x _ { 0 } | x _ { t } ] \big )
$$

$$
+ | | \pmb { x } _ { 0 } - \mathbb { E } [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ] | | ^ { 2 } ]\tag{A.4}
$$

$$
= \mathrm { a r g m i n } _ { \theta } \mathbb { E } \left[ \left| \left| h _ { \theta } ( x _ { t } , t ) - \mathbb { E } [ { \pmb x } _ { 0 } | { \pmb x } _ { t } ] \right| \right| ^ { 2 } - 2 h _ { \theta } ( { \pmb x } _ { t } , t ) ^ { \top } ( { \pmb x } _ { 0 } - \mathbb { E } [ { \pmb x } _ { 0 } | { \pmb x } _ { t } ] ) \right] .\tag{A.5}
$$

Now, for the second term, we have:

$$
\mathbb { E } _ { \boldsymbol { x } _ { 0 } , \boldsymbol { x } _ { t } } \left[ h _ { \theta } ( \boldsymbol { x } _ { t } , t ) ^ { \top } ( \boldsymbol { x } _ { 0 } - \mathbb { E } [ \boldsymbol { x } _ { 0 } | \boldsymbol { x } _ { t } ] ) \right] = \mathbb { E } _ { \boldsymbol { x } _ { t } } \mathbb { E } _ { \boldsymbol { x } _ { 0 } | \boldsymbol { x } _ { t } } \left[ h _ { \theta } ( \boldsymbol { x } _ { t } , t ) ^ { \top } ( \boldsymbol { x } _ { 0 } - \mathbb { E } [ \boldsymbol { x } _ { 0 } | \boldsymbol { x } _ { t } ] ) \right]\tag{A.6}
$$

$$
= \mathbb { E } _ { { \pmb { x } } _ { t } } \left[ h _ { \pmb { \theta } } ( { \pmb { x } } _ { t } , t ) ^ { \top } \left( \mathbb { E } _ { { \pmb { x } } _ { 0 } | { \pmb x } _ { t } } \left[ \left( { \pmb { x } } _ { 0 } - \mathbb { E } [ { \pmb x } _ { 0 } | { \pmb x } _ { t } ] \right) \right] \right) \right] = 0 ,\tag{A.7}
$$

which concludes the proof.

## A.1 Tweedie’s Formula

Lemma A.2 (Tweedie’s Formula). Let:

$$
X _ { t } = X _ { 0 } + \sigma _ { t } Z ,\tag{A.8}
$$

for $X _ { 0 } \sim p _ { X _ { 0 } }$ and $ { \boldsymbol { Z } } \sim \mathcal { N } ( 0 , I )$ . Then,

$$
\nabla _ { \pmb { x } _ { t } } \log { p _ { t } ( \pmb { x } _ { t } ) } = \frac { \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { x } _ { t } ] - \pmb { x } _ { t } } { \sigma _ { t } ^ { 2 } } .\tag{A.9}
$$

Proof.

$$
\nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } ) = \frac { 1 } { p _ { t } ( \pmb { x } _ { t } ) } \nabla _ { \pmb { x } _ { t } } p _ { t } ( \pmb { x } _ { t } ) = \frac { 1 } { p _ { t } ( \pmb { x } _ { t } ) } \nabla _ { \pmb { x } _ { t } } \int p _ { t } ( \pmb { x } _ { t } , \pmb { x } _ { 0 } ) \mathrm { d } \pmb { x } _ { 0 }\tag{A.10}
$$

$$
= \frac { 1 } { p _ { t } ( \pmb { x } _ { t } ) } \nabla _ { \pmb { x } _ { t } } \int p _ { t } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) p _ { 0 } ( \pmb { x } _ { 0 } ) \mathrm { d } \pmb { x } _ { 0 }\tag{A.11}
$$

$$
= \frac { 1 } { p _ { t } ( { \pmb x } _ { t } ) } \int \nabla _ { { \pmb x } _ { t } } p _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) p _ { 0 } ( { \pmb x } _ { 0 } ) \mathrm { d } { \pmb x } _ { 0 }\tag{A.12}
$$

$$
= \frac { 1 } { p _ { t } ( \pmb { x } _ { t } ) } \int p _ { t } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) p _ { 0 } ( \pmb { x } _ { 0 } ) \mathrm { d } \pmb { x } _ { 0 }\tag{A.13}
$$

$$
= \int p _ { 0 } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } ) \frac { \pmb { x } _ { 0 } - \pmb { x } _ { t } } { \sigma _ { t } ^ { 2 } } \mathrm { d } \pmb { x } _ { 0 }\tag{A.14}
$$

$$
= \frac { \mathbb { E } [ X _ { 0 } | { \pmb x } _ { t } ] - { \pmb x } _ { t } } { \sigma _ { t } ^ { 2 } } .\tag{A.15}
$$

## A.2 Denoising Score Matching

By leveraging the MMSE interpretation of the conditional expectation and Tweedie’s formula, one can approximate the score function by training a model to predict the clean image from a corrupted observation (via supervised learning). At inference time, the trained network can be converted to a model that approximates the score through Tweedie’s formula. This training procedure is typically known as x -prediction loss. An alternative, but equivalent, way is to train for the score directly. Vincent [133] independently discovered Denoising Score Matching, which has as a unique minimizer the score function. DSM and the $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ -prediction objective are the same up to a simple network reparametrization.

Theorem A.3 (Denoising Score Matching [133]). Let $p _ { 0 } , p _ { t }$ be two distributions in $\mathbb { R } ^ { n }$ . Assume that all the conditional distributions, $p _ { t } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ , are supported and differentiable in $\mathbb { R } ^ { n }$ . Let:

$$
J _ { 1 } ( \theta ) = \frac { 1 } { 2 } \mathbb { E } _ { \pmb { x } _ { t } \sim p _ { t } } \left[ | | \pmb { s } _ { \theta } ( \pmb { x } _ { t } ) - \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } ) | | ^ { 2 } \right] ,\tag{A.16}
$$

$$
J _ { 2 } ( \theta ) = \frac { 1 } { 2 } \mathbb { E } _ { ( \pmb { x } _ { 0 } , \pmb { x } _ { t } ) \sim p _ { 0 } ( \pmb { x } _ { 0 } ) p _ { t } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) } \left[ | | \pmb { s } _ { \theta } ( \pmb { x } _ { t } ) - \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) | | ^ { 2 } \right] .\tag{A.17}
$$

Then, $J _ { 1 }$ and $J _ { 2 }$ have the same minimizer.

We include the proof listed in [164] for completeness.

Proof.

$$
J _ { 1 } ( \theta ) = \frac { 1 } { 2 } \mathbb { E } _ { \mathbf { x } _ { t } \sim p _ { t } } \left[ | | s _ { \theta } ( \mathbf { x } _ { t } ) | | ^ { 2 } - 2 s _ { \theta } ( \mathbf { x } _ { t } ) ^ { \top } \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } ) + | | \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } ) | | ^ { 2 } \right]\tag{A.18}
$$

$$
= \frac { 1 } { 2 } \mathbb { E } _ { \mathbf { x } _ { t } \sim p _ { t } } \left[ | | s _ { \theta } ( \mathbf { x } _ { t } ) | | ^ { 2 } \right] - \mathbb { E } _ { \mathbf { x } _ { t } \sim p _ { t } } \left[ s _ { \theta } ( \mathbf { x } _ { t } ) ^ { \top } \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } ) \right] + C _ { 1 } .\tag{A.19}
$$

Similarly,

$$
J _ { 2 } ( \theta ) = \frac { 1 } { 2 } \mathbb { E } _ { \boldsymbol { x } _ { t } \sim p _ { t } } \left[ | | s _ { \theta } ( \boldsymbol { x } _ { t } ) | | ^ { 2 } \right] - \mathbb { E } _ { ( \boldsymbol { x } _ { 0 } , \boldsymbol { x } _ { t } ) } \left[ s _ { \theta } ( \boldsymbol { x } _ { t } ) ^ { \top } \nabla _ { \boldsymbol { x } _ { t } } \log p _ { t } ( \boldsymbol { x } _ { t } | \boldsymbol { x } _ { 0 } ) \right] + C _ { 2 } .\tag{A.20}
$$

It suffices to show that:

$$
\begin{array} { r l } & { \mathbb { E } _ { { \pmb { x } } _ { t } \sim p _ { t } } \left[ s _ { \theta } ( { \pmb x } _ { t } ) ^ { \top } \nabla _ { { \pmb x } _ { t } } \log p _ { t } ( { \pmb x } _ { t } ) \right] } \\ & { = \mathbb { E } _ { ( { \pmb x } _ { 0 } , { \pmb x } _ { t } ) \sim p _ { 0 } ( { \pmb x } _ { 0 } ) p _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) } \left[ s _ { \theta } ( { \pmb x } _ { t } ) ^ { \top } \nabla _ { { \pmb x } _ { t } } \log p _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) \right] . } \end{array}\tag{A.21}
$$

We start with the second term.

$$
\begin{array} { r l r } {  { \mathbb { E } _ { ( { \pmb x } _ { 0 } , { \pmb x } _ { t } ) \sim p _ { 0 } ( { \pmb x } _ { 0 } ) p _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) } [ { \pmb s } _ { \theta } ( { \pmb x } _ { t } ) ^ { \top } \nabla _ { { \pmb x } _ { t } } \log p _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) ] } } \\ & { = \int _ { { \pmb x } _ { 0 } } \int _ { { \pmb x } _ { t } } p _ { 0 } ( { \pmb x } _ { 0 } ) p _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) { \pmb s } _ { \theta } ( { \pmb x } _ { t } ) ^ { \top } \nabla _ { { \pmb x } _ { t } } \log p _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) \mathrm { d } { \pmb x } _ { t } \mathrm { d } { \pmb x } _ { 0 } } \\ & { = \int _ { { \pmb x } _ { 0 } } \int _ { { { \pmb x } _ { t } } } { \pmb s } _ { \theta } ^ { \top } ( { \pmb x } _ { t } ) ( p _ { 0 } ( { \pmb x } _ { 0 } ) p _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) \nabla _ { { \pmb x } _ { t } } \log p _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) ) \mathrm { d } { \pmb x } _ { t } \mathrm { d } { \pmb x } _ { 0 } } \end{array}\tag{A.22}
$$

(A.23)

$$
= \int _ { x _ { 0 } } \int _ { x _ { t } } s _ { \theta } ^ { \top } ( { x } _ { t } ) \left( p _ { 0 } ( { x } _ { 0 } ) p _ { t } ( { x } _ { t } | { x } _ { 0 } ) \frac { 1 } { p _ { t } ( { x } _ { t } | { x } _ { 0 } ) } \nabla _ { { x } _ { t } } p _ { t } ( { x } _ { t } | { x } _ { 0 } ) \right) \mathrm { d } { x } _ { t } \mathrm { d } { x } _ { 0 }\tag{A.24}
$$

$$
= \int _ { \pmb { x } _ { 0 } } \int _ { \pmb { x } _ { t } } \pmb { s } _ { \theta } ^ { \top } ( \pmb { x } _ { t } ) \left( p _ { 0 } ( \pmb { x } _ { 0 } ) \nabla _ { \pmb { x } _ { t } } p _ { t } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) \right) \mathrm { d } \pmb { x } _ { t } \mathrm { d } \pmb { x } _ { 0 }\tag{A.25}
$$

$$
= \int _ { \pmb { x } _ { t } } \int _ { \pmb { x } _ { 0 } } \pmb { s } _ { \theta } ^ { \top } ( \pmb { x } _ { t } ) \left( p _ { 0 } ( \pmb { x } _ { 0 } ) \nabla _ { \pmb { x } _ { t } } p _ { t } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) \right) \mathrm { d } \pmb { x } _ { 0 } \mathrm { d } \pmb { x } _ { t }\tag{A.26}
$$

$$
= \int _ { \pmb { x } _ { t } } \pmb { s } _ { \theta } ^ { \top } ( \pmb { x } _ { t } ) \left( \int _ { \pmb { x } _ { 0 } } p _ { 0 } ( \pmb { x } _ { 0 } ) \nabla _ { \pmb { x } _ { t } } p _ { t } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) \mathrm { d } \pmb { x } _ { 0 } \right) \mathrm { d } \pmb { x } _ { t }\tag{A.27}
$$

$$
= \int _ { \pmb { x } _ { t } } \pmb { s } _ { \theta } ^ { \top } ( \pmb { x } _ { t } ) \left( \int _ { \pmb { x } _ { 0 } } \nabla _ { \pmb { x } _ { t } } \left( p _ { 0 } ( \pmb { x } _ { 0 } ) p _ { t } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) \right) \mathrm { d } \pmb { x } _ { 0 } \right) \mathrm { d } \pmb { x } _ { t }\tag{A.28}
$$

$$
= \int _ { \pmb { x } _ { t } } \pmb { s } _ { \theta } ^ { \top } ( { \boldsymbol { x } } _ { t } ) \left( \nabla _ { \pmb { x } _ { t } } \left( \int _ { \pmb { x } _ { 0 } } p _ { 0 } ( \pmb { x } _ { 0 } ) p _ { t } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) \mathrm { d } \pmb { x } _ { 0 } \right) \right) \mathrm { d } \pmb { x } _ { t }\tag{A.29}
$$

$$
= \int _ { \pmb { x } _ { t } } \pmb { s } _ { \theta } ^ { \top } ( \pmb { x } _ { t } ) \nabla _ { \pmb { x } _ { t } } p _ { t } ( \pmb { x } _ { t } ) \mathrm { d } \pmb { x } _ { t }\tag{A.30}
$$

$$
= \int _ { \mathbf { x } _ { t } } p _ { t } ( \pmb { x } _ { t } ) \pmb { s } _ { \theta } ^ { \top } ( \pmb { x } _ { t } ) \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } ) \mathrm { d } \pmb { x } _ { t }\tag{A.31}
$$

$$
= \mathbb { E } _ { \pmb { x } _ { t } \sim p _ { t } ( \pmb { x } _ { t } ) } \left[ \pmb { s } _ { \theta } ^ { \top } ( \pmb { x } _ { t } ) \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } ) \right] .\tag{A.32}
$$

## A.3 Jacobian of the score

Lemma A.4 (Jacobian of score-function). Let:

$$
X _ { t } = X _ { 0 } + \sigma _ { t } Z ,\tag{A.33}
$$

for $X _ { 0 } \sim p _ { X _ { 0 } }$ and $ { \boldsymbol { Z } } \sim \mathcal { N } ( 0 , I )$ . Then,

$$
\mathrm { H } ( \log p _ { X _ { t } } ) ( { \pmb x } _ { t } ) = \frac { \mathbb { E } [ { \pmb X } _ { 0 } { \pmb X } _ { 0 } ^ { \top } | { \pmb x } _ { t } ] - \mathbb { E } [ { \pmb X } _ { 0 } | { \pmb x } _ { t } ] \mathbb { E } ^ { \top } [ { \pmb X } _ { 0 } | { \pmb x } _ { t } ] } { \sigma _ { t } ^ { 4 } } - \frac { I } { \sigma _ { t } ^ { 2 } } .\tag{A.34}
$$

Proof.

$$
\begin{array} { r l } & { \nabla _ { \pmb { x } _ { t } } \log p _ { \pmb { X } _ { t } } ( \pmb { x } _ { t } ) = \frac { \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { x } _ { t } ] - \pmb { x } _ { t } } { \sigma _ { t } ^ { 2 } } } \\ & { \Rightarrow \sigma _ { t } ^ { 2 } \mathrm { H } ( \log p _ { \pmb { X } _ { t } } ) ( \pmb { x } _ { t } ) = \mathrm { J a c o b } \left( \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { x } _ { t } ] \right) - I . } \end{array}\tag{A.35}
$$

(A.36)

We will now analyze the Jacobian.

$$
\operatorname { J a c o b } \left( \mathbb { E } [ X _ { 0 } | \pmb { x } _ { t } ] \right) = \int \nabla _ { \pmb { x } _ { t } } \left( p _ { \pmb { X } _ { 0 } } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } ) \pmb { x } _ { 0 } \right) \mathrm { d } \pmb { x } _ { 0 }\tag{A.37}
$$

$$
\mathbf { \Sigma } = \int \pmb { x } _ { 0 } \nabla _ { \pmb { x } _ { t } } ^ { \top } p _ { \mathbf { X } _ { 0 } } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } ) \mathrm { d } \pmb { x } _ { 0 }\tag{A.38}
$$

$$
= \int \displaylimits _  \mathbf { \boldsymbol { x } } _ { 0 } p _ { X _ { 0 } } ( \mathbf { \boldsymbol { x } } _ { 0 } | \mathbf { \boldsymbol { x } } _ { t } ) \nabla _ { \mathbf { \boldsymbol { x } } _ { t } } ^ { \top } \log \left( \frac { p _ { X _ { t } } ( \mathbf { \boldsymbol { x } } _ { t } | \mathbf { \boldsymbol { x } } _ { 0 } ) p _ { X _ { 0 } } ( \mathbf { \boldsymbol { x } } _ { 0 } ) } { p _ { X _ { t } } ( \mathbf { \boldsymbol { x } } _ { t } ) } \right) \mathrm { d } \mathbf { \boldsymbol { x } } _ { 0 }\tag{A.39}
$$

$$
= \int \mathbf { x } _ { 0 } p \mathbf { { } } _ { \mathbf { { X } } _ { 0 } } ( \mathbf { { } } x _ { 0 } | \mathbf { { } } \mathbf { { x } } _ { t } ) \nabla _ { \mathbf { x } _ { t } } ^ { \top } \log \left( \frac { p _ { { } \mathbf { { X } } _ { t } } ( \mathbf { { } } \mathbf { { \mathbf { x } } } _ { t } | \mathbf { { } } \mathbf { { x } } _ { 0 } ) } { p _ { { } \mathbf { { X } } _ { t } } ( \mathbf { { } } \mathbf { { x } } _ { t } ) } \right) \mathrm { d } \mathbf { { x } } _ { 0 }\tag{A.40}
$$

$$
= \int x _ { 0 } p x _ { 0 } ( x _ { 0 } | x _ { t } ) \nabla _ { x _ { t } } ^ { \top } \log p x _ { t } ( x _ { t } | x _ { 0 } ) \mathrm { d } x _ { 0 } - \int x _ { 0 } p x _ { 0 } ( x _ { 0 } | x _ { t } ) \nabla _ { x _ { t } } ^ { \top } \log p x _ { t } ( x _ { t } ) \mathrm { d } x _ { 0 }\tag{A.41}
$$

$$
= \int \displaylimits _  { \boldsymbol { x } } _ { 0 } p _ { X _ { 0 } } ( { \boldsymbol { x } } _ { 0 } | { \boldsymbol { x } } _ { t } ) \frac { \boldsymbol { x } _ { 0 } ^ { \top } - \boldsymbol { x } _ { t } ^ { \top } } { \sigma _ { t } ^ { 2 } } \mathrm { d } { \boldsymbol { x } } _ { 0 } - \int \displaylimits _  { \boldsymbol { x } } _ { 0 } p _ { X _ { 0 } } ( { \boldsymbol { x } } _ { 0 } | { \boldsymbol { x } } _ { t } ) \nabla _ { { \boldsymbol { x } } _ { t } } ^ { \top } \log p _ { X _ { t } } ( { \boldsymbol { x } } _ { t } ) \mathrm { d } { \boldsymbol { x } } _ { 0 }\tag{A.42}
$$

$$
= \int x _ { 0 } p _ { X _ { 0 } } ( x _ { 0 } | x _ { t } ) \frac { x _ { 0 } ^ { \top } - x _ { t } ^ { \top } } { \sigma _ { t } ^ { 2 } } \mathrm { d } x _ { 0 } - \int x _ { 0 } p _ { X _ { 0 } } ( x _ { 0 } | x _ { t } ) \frac { \mathbb { E } ^ { \top } [ x _ { 0 } | x _ { t } ] - x _ { t } ^ { \top } } { \sigma _ { t } ^ { 2 } } \mathrm { d } x _ { 0 }\tag{A.43}
$$

$$
= \frac { 1 } { \sigma _ { t } ^ { 2 } } \left( \mathbb { E } [ \pmb { x } _ { 0 } \pmb { x } _ { 0 } ^ { \top } | \pmb { x } _ { t } ] - \mathbb { E } [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ] \mathbb { E } [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ] ^ { \top } \right) .\tag{A.44}
$$

Corollary A.5. Let:

$$
X _ { t } = X _ { 0 } + \sigma _ { t } Z ,\tag{A.45}
$$

for $X _ { 0 } \sim p _ { X _ { 0 } } , X _ { 0 } \in \mathbb { R } ^ { n }$ and $ { \boldsymbol { Z } } \sim \mathcal { N } ( 0 , I )$ . Then,

$$
\nabla _ { x _ { t } } ^ { 2 } \log { p _ { t } ( x _ { t } ) } = \frac { \mathbb { E } [ | | X _ { 0 } | | ^ { 2 } \ | \ x _ { t } ] - | | \mathbb { E } [ X _ { 0 } | x _ { t } ] | | ^ { 2 } } { \sigma _ { t } ^ { 4 } } - \frac { n } { \sigma _ { t } ^ { 2 } } .\tag{A.46}
$$

## References

[1] A. Jalal, M. Arvinte, G. Daras, E. Price, A. G. Dimakis, and J. Tamir, “Robust compressed sensing mri with deep generative priors,” Advances in Neural Information Processing Systems, vol. 34, pp. 14 938–14 954, 2021.

[2] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Scorebased generative modeling through stochastic differential equations,” arXiv preprint arXiv:2011.13456, 2020.

[3] J. Choi, S. Kim, Y. Jeong, Y. Gwon, and S. Yoon, “Ilvr: Conditioning method for denoising diffusion probabilistic models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 14 367–14 376.

[4] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, and J. C. Ye, “Diffusion posterior sampling for general noisy inverse problems,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=OnD9zGAGT0k

[5] J. Song, A. Vahdat, M. Mardani, and J. Kautz, “Pseudoinverse-guided diffusion models for inverse problems,” in International Conference on Learning Representations, 2022.

[6] F. Rozet, G. Andry, F. Lanusse, and G. Louppe, “Learning diffusion priors from observations by expectation maximization,” arXiv preprint arXiv:2405.13712, 2024.

[7] H. Chung, J. Kim, S. Kim, and J. C. Ye, “Parallel diffusion models of operator and image for blind inverse problems,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 6059–6069.

[8] B. Kawar, G. Vaksman, and M. Elad, “Snips: Solving noisy inverse problems stochastically,” Advances in Neural Information Processing Systems, vol. 34, pp. 21 757–21 769, 2021.

[9] B. Kawar, M. Elad, S. Ermon, and J. Song, “Denoising diffusion restoration models,” in Advances in Neural Information Processing Systems, 2022.

[10] N. Murata, K. Saito, C.-H. Lai, Y. Takida, T. Uesaka, Y. Mitsufuji, and S. Ermon, “Gibbsddrm: A partially collapsed gibbs sampler for solving blind inverse problems with denoising diffusion restoration,” in International Conference on Machine Learning. PMLR, 2023, pp. 25 501–25 522.

[11] Y. Wang, J. Yu, and J. Zhang, “Zero-shot image restoration using denoising diffusion nullspace model,” arXiv preprint arXiv:2212.00490, 2022.

[12] H. Chung, S. Lee, and J. C. Ye, “Decomposed diffusion sampler for accelerating large-scale inverse problems,” arXiv preprint arXiv:2303.05754, 2023.

[13] Y. Zhu, K. Zhang, J. Liang, J. Cao, B. Wen, R. Timofte, and L. Van Gool, “Denoising diffusion models for plug-and-play image restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1219–1229.

[14] L. Rout, N. Raoof, G. Daras, C. Caramanis, A. Dimakis, and S. Shakkottai, “Solving linear inverse problems provably via posterior sampling with latent diffusion models,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[15] L. Rout, Y. Chen, A. Kumar, C. Caramanis, S. Shakkottai, and W.-S. Chu, “Beyond first-order tweedie: Solving inverse problems using latent diffusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9472–9481.

[16] M. Mardani, J. Song, J. Kautz, and A. Vahdat, “A variational perspective on solving inverse problems with diffusion models,” in The Twelfth International Conference on Learning Representations, 2024.

[17] C. Alkan, J. Oscanoa, D. Abraham, M. Gao, A. Nurdinova, K. Setsompop, J. M. Pauly, M. Mardani, and S. Vasanawala, “Variational diffusion models for blind mri inverse problems,” in NeurIPS 2023 Workshop on Deep Learning and Inverse Problems, 2023.

[18] B. T. Feng, J. Smith, M. Rubinstein, H. Chang, K. L. Bouman, and W. T. Freeman, “Score-based diffusion models as principled priors for inverse imaging,” arXiv preprint arXiv:2304.11751, 2023.

[19] B. T. Feng and K. L. Bouman, “Efficient bayesian computational imaging with a surrogate score-based prior,” arXiv preprint arXiv:2309.01949, 2023.

[20] H. Wang, X. Zhang, T. Li, Y. Wan, T. Chen, and J. Sun, “Dmplug: A plug-in method for solving inverse problems with diffusion models,” arXiv preprint arXiv:2405.16749, 2024.

[21] H. Chihaoui, A. Lemkhenter, and P. Favaro, “Zero-shot image restoration via diffusion inversion,” 2024. [Online]. Available: https://openreview.net/forum?id=ZnmofqLWMQ

[22] T. Xu, Z. Zhu, J. Li, D. He, Y. Wang, M. Sun, L. Li, H. Qin, Y. Wang, J. Liu, and Y.- Q. Zhang, “Consistency model is an effective posterior sample approximation for diffusion inverse solvers,” 2024.

[23] G. Daras, Y. Dagan, A. Dimakis, and C. Daskalakis, “Score-guided intermediate level optimization: Fast Langevin mixing for inverse problems,” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, Eds., vol. 162. PMLR, 17–23 Jul 2022, pp. 4722–4753. [Online]. Available: https://proceedings.mlr.press/v162/daras22a.html

[24] Z. Wu, Y. Sun, Y. Chen, B. Zhang, Y. Yue, and K. L. Bouman, “Principled probabilistic imaging using diffusion models as plug-and-play priors,” 2024.

[25] Z. Dou and Y. Song, “Diffusion posterior sampling for linear inverse problem solving: A filtering perspective,” in The Twelfth International Conference on Learning Representations, 2023.

[26] Y. Sun, Z. Wu, Y. Chen, B. T. Feng, and K. L. Bouman, “Provable probabilistic imaging using score-based generative priors,” IEEE Transactions on Computational Imaging, 2024.

[27] B. L. Trippe, J. Yim, D. Tischer, D. Baker, T. Broderick, R. Barzilay, and T. S. Jaakkola, “Diffusion probabilistic modeling of protein backbones in 3d for the motif-scaffolding problem,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=6TxBxqNME1Y

[28] G. Cardoso, S. Le Corff, E. Moulines et al., “Monte carlo guided denoising diffusion models for bayesian linear inverse problems.” in The Twelfth International Conference on Learning Representations, 2023.

[29] L. Wu, B. L. Trippe, C. A. Naesseth, J. P. Cunningham, and D. Blei, “Practical and asymptotically exact conditional sampling in diffusion models,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id=eWKqr1zcRv

[30] Z. Kadkhodaie and E. P. Simoncelli, “Solving linear inverse problems using the prior implicit in a denoiser,” arXiv preprint arXiv:2007.13640, 2020

[31] H. Chung, B. Sim, D. Ryu, and J. C. Ye, “Improving diffusion models for inverse problems using manifold constraints,” Advances in Neural Information Processing Systems, vol. 35, pp. 25 683–25 696, 2022.

[32] B. Song, S. M. Kwon, Z. Zhang, X. Hu, Q. Qu, and L. Shen, “Solving inverse problems with latent diffusion models via hard data consistency,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=j8hdRqOUhN

[33] Y. He, N. Murata, C.-H. Lai, Y. Takida, T. Uesaka, D. Kim, W.-H. Liao, Y. Mitsufuji, J. Z. Kolter, R. Salakhutdinov et al., “Manifold preserving guided diffusion,” arXiv preprint arXiv:2311.16424, 2023.

[34] H. Chung, J. C. Ye, P. Milanfar, and M. Delbracio, “Prompt-tuning latent diffusion models for inverse problems,” in International Conference on Machine Learning. PMLR, 2014.

[35] J. Kim, G. Y. Park, H. Chung, and J. C. Ye, “Regularization by texts for latent diffusion inverse solvers,” arXiv preprint arXiv:2311.15658, 2023.

[36] J. Kim, G. Y. Park, and J. C. Ye, “Dreamsampler: Unifying diffusion sampling and score distillation for image manipulation,” arXiv preprint arXiv:2403.11415, 2024.

[37] P. Lailly and J. Bednar, “The seismic inverse problem as a sequence of before stack migrations,” in Conference on inverse scattering: theory and application, vol. 1983. Philadelphia, Pa, 1983, pp. 206–220.

[38] J. Virieux and S. Operto, “An overview of full-waveform inversion in exploration geophysics,” Geophysics, vol. 74, no. 6, pp. WCC1–WCC26, 2009.

[39] S. Huang, J. Xiang, H. Du, and X. Cao, “Inverse problems in atmospheric science and their application,” in Journal of Physics: Conference Series, vol. 12, no. 1. IOP Publishing, 2005, p. 45.

[40] C. Wunsch, The ocean circulation inverse problem. Cambridge University Press, 1996.

[41] J.-M. Lemercier, J. Richter, S. Welker, E. Moliner, V. Välimäki, and T. Gerkmann, “Diffusion models for audio restoration,” arXiv preprint arXiv:2402.09821, 2024.

[42] K. Saito, N. Murata, T. Uesaka, C.-H. Lai, Y. Takida, T. Fukui, and Y. Mitsufuji, “Unsupervised vocal dereverberation with diffusion-based generative models,” in ICASSP 2023- 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

[43] E. Moliner, F. Elvander, and V. Välimäki, “Blind audio bandwidth extension: A diffusionbased zero-shot approach,” arXiv preprint arXiv:2306.01433, 2023.

[44] E. Moliner, J. Lehtinen, and V. Välimäki, “Solving audio inverse problems with a diffusion model,” in ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

[45] E. Moliner and V. Välimäki, “Diffusion-based audio inpainting,” arXiv preprint arXiv:2305.15266, 2023.

[46] C. Hernandez-Olivan, K. Saito, N. Murata, C.-H. Lai, M. A. Martínez-Ramirez, W.-H. Liao, and Y. Mitsufuji, “Vrdmg: Vocal restoration via diffusion posterior sampling with multiple guidance,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 596–600.

[47] Y. Song, L. Shen, L. Xing, and S. Ermon, “Solving inverse problems in medical imaging with score-based generative models,” arXiv preprint arXiv:2111.08005, 2021.

[48] H. Chung, D. Ryu, M. T. McCann, M. L. Klasky, and J. C. Ye, “Solving 3d inverse problems using pre-trained 2d diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22 542–22 551.

[49] A. Aali, G. Daras, B. Levac, S. Kumar, A. G. Dimakis, and J. I. Tamir, “Ambient diffusion posterior sampling: Solving inverse problems with diffusion models trained on corrupted data,” arXiv preprint arXiv:2403.08728, 2024.

[50] H. Chung and J. C. Ye, “Score-based diffusion models for accelerated mri,” Medical image analysis, vol. 80, p. 102479, 2022.

[51] L. Fan, F. Zhang, H. Fan, and C. Zhang, “Brief review of image denoising techniques,” Visual Computing for Industry, Biomedicine, and Art, vol. 2, no. 1, p. 7, 2019.

[52] W. Quan, J. Chen, Y. Liu, D.-M. Yan, and P. Wonka, “Deep learning-based image and video inpainting: A survey,” International Journal of Computer Vision, vol. 132, no. 7, pp. 2367– 2400, 2024.

[53] T. Yu, R. Feng, R. Feng, J. Liu, X. Jin, W. Zeng, and Z. Chen, “Inpaint anything: Segment anything meets image inpainting,” arXiv preprint arXiv:2304.06790, 2023.

[54] J. Ouyang-Zhang, D. J. Diaz, A. Klivans, and P. Krähenbühl, “Predicting a protein’s stability under a million mutations,” NeurIPS, 2023.

[55] D. J. Diaz, C. Gong, J. Ouyang-Zhang, J. M. Loy, J. Wells, D. Yang, A. D. Ellington, A. G. Dimakis, and A. R. Klivans, “Stability oracle: a structure-based graph-transformer framework for identifying stabilizing mutations,” Nature Communications, vol. 15, no. 1, p. 6170, 2024.

[56] K. K. Yang, Z. Wu, and F. H. Arnold, “Machine-learning-guided directed evolution for protein engineering,” Nature methods, vol. 16, no. 8, pp. 687–694, 2019.

[57] Y. Xu, D. Verma, R. P. Sheridan, A. Liaw, J. Ma, N. M. Marshall, J. McIntosh, E. C. Sherer, V. Svetnik, and J. M. Johnston, “Deep dive into machine learning models for protein engineering,” Journal of chemical information and modeling, vol. 60, no. 6, pp. 2773–2790, 2020.

[58] A. Aali, M. Arvinte, S. Kumar, and J. I. Tamir, “Solving inverse problems with score-based generative priors learned from noisy data,” arXiv preprint arXiv:2305.01166, 2023.

[59] J. Zbontar, F. Knoll, A. Sriram, T. Murrell, Z. Huang, M. J. Muckley, A. Defazio, R. Stern, P. Johnson, M. Bruno et al., “fastmri: An open dataset and benchmarks for accelerated mri,” arXiv preprint arXiv:1811.08839, 2018.

[60] A. D. Desai, A. M. Schmidt, E. B. Rubin, C. M. Sandino, M. S. Black, V. Mazzoli, K. J. Stevens, R. Boutin, C. Ré, G. E. Gold, B. A. Hargreaves, and A. S. Chaudhari, “Skm-tea: A dataset for accelerated mri reconstruction with dense image labels for quantitative clinical evaluation,” 2022.

[61] T. Zhang, J. Pauly, S. Vasanawala, and M. Lustig, “MRI Data: Undersampled Abdomens,” Undersampled Abdomens | MRI Data. [Online]. Available: http://old.mridata.org/undersampled/abdomens

[62] U. Tariq, P. Lai, M. Lustig, M. Alley, M. Zhang, G. Gold, and V. S. S, “MRI Data: Undersampled Knees,” Undersampled Knees | MRI Data. [Online]. Available: http://old.mridata.org/undersampled/knees

[63] M. Lustig, D. L. Donoho, J. M. Santos, and J. M. Pauly, “Compressed sensing mri,” IEEE signal processing magazine, vol. 25, no. 2, pp. 72–82, 2008.

[64] X. Pan, E. Y. Sidky, and M. Vannier, “Why do commercial ct scanners still employ traditional, filtered back-projection for image reconstruction?” Inverse problems, vol. 25, no. 12, p. 123009, 2009.

[65] M. Genzel, I. Gühring, J. Macdonald, and M. März, “Near-exact recovery for tomographic inverse problems via deep learning,” in International Conference on Machine Learning. PMLR, 2022, pp. 7368–7381.

[66] G. Beylkin, “The inversion problem and applications of the generalized radon transform,” Communications on pure and applied mathematics, vol. 37, no. 5, pp. 579–599, 1984.

[67] A. C. Kak and M. Slaney, Principles of computerized tomographic imaging. SIAM, 2001.

[68] M. Dietz, L. Liljeryd, K. Kjorling, and O. Kunz, “Spectral band replication, a novel approach in audio coding,” in Audio Engineering Society Convention 112. Audio Engineering Society, 2002.

[69] J. Dubochet, M. Adrian, J.-J. Chang, J.-C. Homo, J. Lepault, A. W. McDowall, and P. Schultz, “Cryo-electron microscopy of vitrified specimens,” Quarterly reviews of biophysics, vol. 21, no. 2, pp. 129–228, 1988.

[70] S. C. Park, M. K. Park, and M. G. Kang, “Super-resolution image reconstruction: a technical overview,” IEEE signal processing magazine, vol. 20, no. 3, pp. 21–36, 2003.

[71] C. Saharia, J. Ho, W. Chan, T. Salimans, D. J. Fleet, and M. Norouzi, “Image super-resolution via iterative refinement,” arXiv preprint arXiv:2104.07636, 2021.

[72] T. Nakatani, T. Yoshioka, K. Kinoshita, M. Miyoshi, and B.-H. Juang, “Speech dereverberation based on variance-normalized delayed linear prediction,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 18, no. 7, pp. 1717–1731, 2010.

[73] J. R. Fienup, “Phase retrieval algorithms: a comparison,” Applied optics, vol. 21, no. 15, pp. 2758–2769, 1982.

[74] K. Akiyama, A. Alberdi, W. Alef, K. Asada, R. Azulay, A.-K. Baczko, D. Ball, M. Balokovic,´ J. Barrett, D. Bintley et al., “First m87 event horizon telescope results. iv. imaging the central supermassive black hole,” The Astrophysical Journal Letters, vol. 875, no. 1, p. L4, 2019.

[75] A. Tarantola, Inverse problem theory and methods for model parameter estimation. SIAM, 2005.

[76] J. Scarlett, R. Heckel, M. R. D. Rodrigues, P. Hand, and Y. C. Eldar, “Theoretical perspectives on deep learning methods in inverse problems,” IEEE Journal on Selected Areas in Information Theory, vol. 3, no. 3, p. 433–453, Sep. 2022. [Online]. Available: http://dx.doi.org/10.1109/JSAIT.2023.3241123

[77] R. Bassett and J. Deride, “Maximum a posteriori estimators as a limit of bayes estimators,” Mathematical Programming, vol. 174, pp. 129–144, 2019.

[78] M. Pereyra, “Revisiting maximum-a-posteriori estimation in log-concave models,” SIAM Journal on Imaging Sciences, vol. 12, no. 1, pp. 650–670, 2019.

[79] G. A. Young, R. L. Smith, and R. L. Smith, Essentials of statistical inference. Cambridge University Press, 2005, vol. 16.

[80] K. P. Murphy, Machine learning: a probabilistic perspective. MIT press, 2012.

[81] Y. Blau and T. Michaeli, “The perception-distortion tradeoff,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2018.

[82] A. Ribes and F. Schmitt, “Linear inverse problems in imaging,” IEEE Signal Processing Magazine, vol. 25, no. 4, pp. 84–99, 2008.

[83] H. H. Barrett and K. J. Myers, Foundations of image science. John Wiley & Sons, 2013.

[84] I. Daubechies, M. Defrise, and C. De Mol, “An iterative thresholding algorithm for linear inverse problems with a sparsity constraint,” Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute of Mathematical Sciences, vol. 57, no. 11, pp. 1413–1457, 2004.

[85] E. J. Candès, J. Romberg, and T. Tao, “Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information,” IEEE Transactions on information theory, vol. 52, no. 2, pp. 489–509, 2006.

[86] D. L. Donoho, “Compressed sensing,” IEEE Transactions on information theory, vol. 52, no. 4, pp. 1289–1306, 2006.

[87] M. A. Figueiredo and R. D. Nowak, “An em algorithm for wavelet-based image restoration,” IEEE Transactions on Image Processing, vol. 12, no. 8, pp. 906–916, 2003.

[88] E. T. Hale, W. Yin, and Y. Zhang, “A fixed-point continuation method for l1-regularized minimization with applications to compressed sensing,” CAAM TR07-07, Rice University, vol. 43, no. 44, p. 2, 2007.

[89] N. Shlezinger, J. Whang, Y. C. Eldar, and A. G. Dimakis, “Model-based deep learning,” Proceedings of the IEEE, vol. 111, no. 5, pp. 465–499, 2023.

[90] L. I. Rudin, S. Osher, and E. Fatemi, “Nonlinear total variation based noise removal algorithms,” Physica D: Nonlinear Phenomena, vol. 60, no. 1-4, pp. 259–268, 1992.

[91] A. Beck and M. Teboulle, “Fast gradient-based algorithms for constrained total variation image denoising and deblurring problems,” IEEE transactions on image processing, vol. 18, no. 11, pp. 2419–2434, 2009.

[92] G. Ongie, A. Jalal, C. A. Metzler, R. G. Baraniuk, A. G. Dimakis, and R. Willett, “Deep learning techniques for inverse problems in imaging,” IEEE Journal on Selected Areas in Information Theory, vol. 1, no. 1, pp. 39–56, 2020.

[93] C. Dong, C. C. Loy, K. He, and X. Tang, “Image super-resolution using deep convolutional networks,” IEEE transactions on pattern analysis and machine intelligence, vol. 38, no. 2, pp. 295–307, 2015.

[94] B. Lim, S. Son, H. Kim, S. Nah, and K. Mu Lee, “Enhanced deep residual networks for single image super-resolution,” in Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2017, pp. 136–144.

[95] X. Tao, H. Gao, X. Shen, J. Wang, and J. Jia, “Scale-recurrent network for deep image deblurring,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[96] C. Chen, Q. Chen, J. Xu, and V. Koltun, “Learning to see in the dark,” in IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 3291–3300.

[97] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, “Restormer: Efficient transformer for high-resolution image restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5728–5739.

[98] L. Chen, X. Chu, X. Zhang, and J. Sun, “Simple baselines for image restoration,” in Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part VII. Springer, 2022, pp. 17–33.

[99] Z. Tu, H. Talebi, H. Zhang, F. Yang, P. Milanfar, A. Bovik, and Y. Li, “Maxim: Multi-axis mlp for image processing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5769–5780.

[100] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, “Multi-stage progressive image restoration,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 14 821–14 831.

[101] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in IEEE conference on computer vision and pattern recognition, 2017, pp. 1125–1134.

[102] O. Kupyn, V. Budzan, M. Mykhailych, D. Mishkin, and J. Matas, “Deblurgan: Blind motion deblurring using conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 8183–8192.

[103] M. Delbracio and P. Milanfar, “Inversion by direct iteration: An alternative to denoising diffusion for image restoration,” Transactions on Machine Learning Research, 2023, featured Certification. [Online]. Available: https://openreview.net/forum?id=VmyFF5lL3F

[104] S. V. Venkatakrishnan, C. A. Bouman, and B. Wohlberg, “Plug-and-play priors for model based reconstruction,” in 2013 IEEE Global Conference on Signal and Information Processing. IEEE, 2013, pp. 945–948.

[105] S. Sreehari, S. V. Venkatakrishnan, B. Wohlberg, G. T. Buzzard, L. F. Drummy, J. P. Simmons, and C. A. Bouman, “Plug-and-play priors for bright field electron tomography and sparse interpolation,” IEEE Transactions on Computational Imaging, vol. 2, no. 4, pp. 408–423, 2016.

[106] S. H. Chan, X. Wang, and O. A. Elgendy, “Plug-and-play admm for image restoration: Fixedpoint convergence and applications,” IEEE Transactions on Computational Imaging, vol. 3, no. 1, pp. 84–98, 2016.

[107] Y. Romano, M. Elad, and P. Milanfar, “The little engine that could: Regularization by denoising (red),” SIAM Journal on Imaging Sciences, vol. 10, no. 4, pp. 1804–1844, 2017.

[108] R. Cohen, M. Elad, and P. Milanfar, “Regularization by denoising via fixed-point projection (red-pro),” SIAM Journal on Imaging Sciences, vol. 14, no. 3, pp. 1374–1406, 2021.

[109] Z. Kadkhodaie and E. P. Simoncelli, “Stochastic solutions for linear inverse problems using the prior implicit in a denoiser,” in Thirty-Fifth Conference on Neural Information Processing Systems, 2021.

[110] U. S. Kamilov, C. A. Bouman, G. T. Buzzard, and B. Wohlberg, “Plug-and-play methods for integrating physical and learned models in computational imaging: Theory, algorithms, and applications,” IEEE Signal Processing Magazine, vol. 40, no. 1, pp. 85–97, 2023.

[111] P. Milanfar and M. Delbracio, “Denoising: A powerful building-block for imaging, inverse problems, and machine learning,” arXiv preprint arXiv:2409.06219, 2024.

[112] H. Li, Y. Yang, M. Chang, H. Feng, Z. Xu, Q. Li, and Y. Chen, “Srdiff: Single image superresolution with diffusion probabilistic models,” arXiv preprint arXiv:2104.14951, 2021.

[113] C. Saharia, W. Chan, H. Chang, C. Lee, J. Ho, T. Salimans, D. Fleet, and M. Norouzi, “Palette: Image-to-image diffusion models,” in ACM SIGGRAPH 2022 Conference Proceedings, 2022, pp. 1–10.

[114] J. Whang, M. Delbracio, H. Talebi, C. Saharia, A. G. Dimakis, and P. Milanfar, “Deblurring via stochastic refinement,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 293–16 303.

[115] Z. Luo, F. K. Gustafsson, Z. Zhao, J. Sjölund, and T. B. Schön, “Image restoration with mean-reverting stochastic differential equations,” arXiv preprint arXiv:2301.11699, 2023.

[116] ——, “Refusion: Enabling large-size realistic image restoration with latent-space diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1680–1691.

[117] M. S. Albergo and E. Vanden-Eijnden, “Building normalizing flows with stochastic interpolants,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=li7qeBbCR1t

[118] M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden, “Stochastic interpolants: A unifying framework for flows and diffusions,” arXiv preprint arXiv:2303.08797, 2023.

[119] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=PqvMRDCJT9t

[120] G.-H. Liu, A. Vahdat, D.-A. Huang, E. A. Theodorou, W. Nie, and A. Anandkumar, “I<sup>2</sup>sb: Image-to-image schrödinger bridge,” arXiv preprint arXiv:2302.05872, 2023.

[121] X. Liu, C. Gong, and qiang liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=XVjTT1nw5z

[122] Y. Shi, V. De Bortoli, A. Campbell, and A. Doucet, “Diffusion schr " odinger bridge matching,” arXiv preprint arXiv:2303.16852, 2023.

[123] A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. Van Gool, “Repaint: Inpainting using denoising diffusion probabilistic models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 11 461–11 471.

[124] L. Rout, A. Parulekar, C. Caramanis, and S. Shakkottai, “A theoretical justification for image inpainting using denoising diffusion probabilistic models,” arXiv preprint arXiv:2302.01217, 2023.

[125] H. Chung and J. C. Ye, “Deep diffusion image prior for efficient ood adaptation in 3d inverse problems,” in Proceedings of the European Conference on Computer Vision (ECCV), 2024.

[126] Y. Shen, X. Jiang, Y. Wang, Y. Yang, D. Han, and D. Li, “Understanding training-free diffusion guidance: Mechanisms and limitations,” arXiv preprint arXiv:2403.12404, 2024.

[127] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in Neural Information Processing Systems, vol. 33, pp. 6840–6851, 2020.

[128] Y. Song and S. Ermon, “Generative modeling by estimating gradients of the data distribution,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[129] B. D. Anderson, “Reverse-time diffusion equation models,” Stochastic Processes and their Applications, vol. 12, no. 3, pp. 313–326, 1982.

[130] D. Maoutsa, S. Reich, and M. Opper, “Interacting particle solutions of fokker–planck equations through gradient–log–density estimation,” Entropy, vol. 22, no. 8, p. 802, 2020.

[131] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[132] B. Efron, “Tweedie’s formula and selection bias,” Journal of the American Statistical Association, vol. 106, no. 496, pp. 1602–1614, 2011.

[133] P. Vincent, “A connection between score matching and denoising autoencoders,” Neural computation, vol. 23, no. 7, pp. 1661–1674, 2011.

[134] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10 684–10 695.

[135] B. Øksendal, Stochastic Differential Equations: An Introduction with Applications, 6th ed. Berlin: Springer Science & Business Media, 2010.

[136] S. Gupta, A. Jalal, A. Parulekar, E. Price, and Z. Xun, “Diffusion posterior sampling is computationally intractable,” arXiv preprint arXiv:2402.12727, 2024.

[137] G. Daras, K. Shah, Y. Dagan, A. Gollakota, A. Dimakis, and A. Klivans, “Ambient diffusion: Learning clean distributions from corrupted data,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 288–313. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2023/file/012af729c5d14d279581fc8a5db975a1-Paper-Conference.pdf

[138] G. Daras, A. Dimakis, and C. C. Daskalakis, “Consistent diffusion meets tweedie: Training exact ambient diffusion models with noisy data,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, Eds., vol. 235. PMLR, 2024, pp. 10 091–10 108. [Online]. Available: https://proceedings.mlr.press/v235/daras24a.htm

[139] G. Daras, Y. Dagan, A. G. Dimakis, and C. Daskalakis, “Consistent diffusion models: Mitigating sampling drift by learning to be consistent,” arXiv preprint arXiv:2302.09057, 2023.

[140] Y. C. Eldar, “Generalized sure for exponential families: Applications to regularization,” IEEE Transactions on Signal Processing, vol. 57, no. 2, pp. 471–481, 2009.

[141] C. M. Stein, “Estimation of the mean of a multivariate normal distribution,” The annals of Statistics, pp. 1135–1151, 1981.

[142] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, and T. Aila, “Noise2noise: Learning image restoration without clean data,” arXiv preprint arXiv:1803.04189, 2018.

[143] A. Krull, T.-O. Buchholz, and F. Jug, “Noise2void-learning denoising from single noisy images,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 2129–2137.

[144] J. Batson and L. Royer, “Noise2self: Blind denoising by self-supervision,” in International Conference on Machine Learning. PMLR, 2019, pp. 524–533.

[145] W. Bai, Y. Wang, W. Chen, and H. Sun, “An expectation-maximization algorithm for training clean diffusion models from corrupted observations,” arXiv preprint arXiv:2407.01014, 2024.

[146] Y. Wang, W. Bai, W. Luo, W. Chen, and H. Sun, “Integrating amortized inference with diffusion models for learning clean distribution from corrupted images,” arXiv preprint arXiv:2407.11162, 2024.

[147] L. Yang, S. Ding, Y. Cai, J. Yu, J. Wang, and Y. Shi, “Guidance with spherical gaussian constraint for conditional diffusion,” arXiv preprint arXiv:2402.03201, 2024.

[148] B. Kawar, G. Vaksman, and M. Elad, “Stochastic image denoising by sampling from the posterior distribution,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 1866–1875.

[149] S. Ravula, B. Levac, A. Jalal, J. I. Tamir, and A. G. Dimakis, “Optimizing sampling patterns for compressed sensing mri with diffusion generative models,” arXiv preprint arXiv:2306.03284, 2023.

[150] D. Rezende and S. Mohamed, “Variational inference with normalizing flows,” in International conference on machine learning. PMLR, 2015, pp. 1530–1538.

[151] L. Dinh, J. Sohl-Dickstein, and S. Bengio, “Density estimation using real nvp,” arXiv preprint arXiv:1605.08803, 2016.

[152] Y. Song, C. Durkan, I. Murray, and S. Ermon, “Maximum likelihood training of score-based diffusion models,” Advances in neural information processing systems, vol. 34, pp. 1415– 1428, 2021.

[153] U. S. Kamilov, H. Mansour, and B. Wohlberg, “A plug-and-play priors approach for solving nonlinear imaging inverse problems,” IEEE Signal Processing Letters, vol. 24, no. 12, pp. 1872–1876, 2017.

[154] A. Doucet, N. De Freitas, and N. Gordon, “An introduction to sequential monte carlo meth ods,” Sequential Monte Carlo methods in practice, pp. 3–14, 2001.

[155] A. Bora, A. Jalal, E. Price, and A. G. Dimakis, “Compressed sensing using generative models,” in International conference on machine learning. PMLR, 2017, pp. 537–546.

[156] G. Daras, J. Dean, A. Jalal, and A. Dimakis, “Intermediate layer optimization for inverse problems using deep generative models,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp. 2421–2432. [Online]. Available: https://proceedings.mlr.press/v139/daras21a.html

[157] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,” in International Conference on Machine Learning. PMLR, 2023, pp. 32 211–32 252.

[158] Y. He, N. Murata, C.-H. Lai, Y. Takida, T. Uesaka, D. Kim, W.-H. Liao, Y. Mitsufuji, J. Z. Kolter, R. Salakhutdinov et al., “Manifold preserving guided diffusion,” in The Twelfth International Conference on Learning Representations, 2023.

[159] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances in neural information processing systems, vol. 34, pp. 8780–8794, 2021.

[160] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.

[161] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PMLR, 2021, pp. 8748–8763.

[162] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “Dreamfusion: Text-to-3d using 2d diffusion,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=FjNys5c7VyY

[163] D. Kim, C.-H. Lai, W.-H. Liao, N. Murata, Y. Takida, T. Uesaka, Y. He, Y. Mitsufuji, and S. Ermon, “Consistency trajectory models: Learning probability flow ode trajectory of diffusion,” in The Twelfth International Conference on Learning Representations, 2023.

[164] G. Daras, M. Delbracio, H. Talebi, A. Dimakis, and P. Milanfar, “Soft diffusion: Score matching with general corruptions,” Transactions on Machine Learning Research, 2023. [Online]. Available: https://openreview.net/forum?id=W98rebBxlQ