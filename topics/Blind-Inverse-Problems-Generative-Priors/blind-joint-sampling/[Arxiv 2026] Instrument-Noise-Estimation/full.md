# Estimation of instrument and noise parameters for inverse problem based on prior diffusion model

Jean-Franc¸ois Giovannelli

Groupe Signal-Image, IMS (Univ. Bordeaux, CNRS, BINP), Talence, France

Abstract — This article addresses the issue of estimating observation parameters (response and error parameters) in inverse problems. The focus is on cases where regularization is introduced in a Bayesian framework and the prior is modeled by a diffusion process. In this context, the issue of posterior sampling is known to be thorny, and a recent paper [1] proposes a notably simple and effective solution. Additionally, it opens an remarkable flexibility when it comes to estimating observation parameters. The proposed strategy enables to define an optimal estimator for both observation parameters and image of interest. Furthermore, the strategy provides a means for uncertainty quantification. In addition, MCMC algorithms allow for the computation of estimates and properties of posteriors, while offering some guarantees. The paper presents several numerical experiments that clearly confirm the computational efficiency and the quality of both estimates and uncertainty quantification.

Index Terms — Inverse problem, deconvolution, Bayesian, Hyperparameter estimation, Diffusion prior, Gibbs sampler.

## I. INTRODUCTION AND PROBLEM FORMULATION

The present paper deals with the resolution of inverse problems [2]–[5] when the observation system is modeled by a linear operator and an additive Gaussian error:

$$
{ \pmb y } = { \pmb H } _ { \pmb { \imath } } { \pmb x } _ { 0 } + { \pmb e } ,\tag{1}
$$

where $\pmb { x } _ { 0 } \in \mathbb { R } ^ { P }$ collects the unknowns, y, $\boldsymbol { e } \in \mathbb { R } ^ { M }$ collect the measurements and errors, and $\pmb { H _ { \iota } } \in \mathbb { R } ^ { P } \times \mathbb { R } ^ { M }$ characterizes the operator, $e . g .$ , a convolution. The vector ι parametrizes the instrument response, typically the width of a Lorentz Point Spread Function (PSF). This is one of the key parameters to be estimated. The second one, denoted $\mathbf { \eta } _ { \eta } ,$ controls the error pdf, e.g., mean $m _ { e }$ and variance $v _ { e }$ of an homogeneous white noise. All these parameters are gathered in the vector $\theta = [ \iota , \eta ]$ ]. These parameters are included among the unknowns and this is a crucial feature of the proposed method to estimate them in addition to the image of interest.

The ability to estimate observation parameters, in addition to the image of interest, is crucial in practice. It is common to have information on instrument parameters or noise levels, $e . g .$ , a nominal value with an associated uncertainty, but it is rare to know them exactly. Moreover, failing to account for uncertainties in these parameters leads to erroneous uncertaintiy quantification about the image of interest.

This issue has been frequently addressed and several solutions have been proposed [6]–[13] referred to as auto-adjusted, adaptive, self-tuned, myopic/blind or self-calibrated. . . That said, in the case of priors constructed from the recent diffusion models [14]–[16], this issue remains difficult and has been very little addressed (however, see [17] and Remark 1). The difficulty may be due to the fact that the dominant approaches are inherited from ancestral sampling (designed for the prior): they attempt to correct the latter to produce posterior samples. But whilst they are exact for the prior they are approximated for the posterior. For example, to sample the posterior for the images, [18] and [19] rely on approximations that involve $H _ { \iota }$ itself and this complicates or even makes it impossible to manage the parameter ι. In contrast, G-DSP (Gibbs Diffusion Posterior Sampling) recently proposed [1] clearly takes advantage of the Markov structure and conditional independences (see also Fig. 1), which opens up noticeable possibilities for the estimation of observation parameters and gives raise to the present contribution, referred to as Hyper-G-DPS (Hyperparameter-G-DPS).

The paper is organized as follows. Section II introduces the various distributions to model measurements and unknown quantities. Section III describes the posterior sampler. The numerical assessement using the MNIST example set is given in Section IV. Finally, Section V proposes a synthesis and includes a few perspectives. Part of the calculations are reported in the Appendix.

## II. LIKELIHOOD, PRIOR, POSTERIOR

## A. Noise and measurement

The measurements are included via the likelihood deduced from the observation model (1) and a model for the error. Here, the latter is described as Gauss with mean $m _ { e }$ and covariance $C _ { e }$ that is to say $\mathcal { N } ( e ; m _ { e } , C _ { e } )$ . So, the likelihood of the unknowns $( \pmb { x } _ { 0 } , \pmb { \theta } )$ attached to the measurement y reads

$$
\begin{array} { r } { f ( \pmb { y } | \pmb { x } _ { 0 } , \pmb { \theta } ) = \mathcal { N } ( \pmb { y } ; \pmb { m } _ { e } + \pmb { H } _ { \iota } \pmb { x } _ { 0 } , \pmb { C } _ { e } ) . } \end{array}\tag{2}
$$

Regarding the error, in subsequent developments, we focus on the stationary and white case: the mean and variance are homogeneous and denoted by $m _ { e }$ and $v _ { e }$ respectively and collected in the vector $\pmb { \eta } = [ m _ { e } , v _ { e } ]$ . However, the proposed methodology can easily be generalised to cover more complex situations, and could incorporate correlation parameters, or non-Gaussian noise based on location mixture of Gaussians.

Regarding the vector $\bullet ,$ it collects the parameters of the instrument response. It may include the amplitude and width of the PSF, e.g., a Lorentzian as considered in the numerical study (Sect. IV). However, the proposed methodology can easily be generalised to more complex PSFs (see, e.g., [20], [21]).

The vector $\theta ~ = ~ [ \iota , \eta ]$ collects the unknown observation parameters (instrument and error), the other unknown being the image of interest $\scriptstyle { \pmb x } _ { 0 }$ . The aim of the rest of this section is to incorporate the available knowledge about these unknowns through probability distributions.

• With regard to images, traditional approaches rely, for example, on pixel positivity, pixel correlation, contours or pulses. . . Here, we rely on the fact that the image shares a certain resemblance with available examples.

• Regarding the observation parameters, the available information may be an order of magnitude, a nominal value with uncertainty, a minimum / maximum values,. . .

When the available information is more uncertain, it is referred to as a poorly informative prior.

Among the distributions that allow this information to be taken into account, one seeks to assign a prior so that the posterior (and especially its conditionals) is easy to manipulate and sample. With this in mind, whenever possible, one relies on simple models, such as Gaussian models, and/or on the notion of conjugacy [22].

## B. Prior for unknowns

Observation parameter ι — For the instrument parameter ι, for each component, we define a uniform prior between a minimum and a maximum values in line with the knowledge of the physical principles of the instrument. We simply write

$$
f _ { I } ( \pmb { \iota } ) = \mathcal { U } ( \pmb { \iota } ) .\tag{3}
$$

In the numerical study of Sect. IV we will consider a Lorentz PSF and ι encode the width.

Noise parameter: offset $m _ { e }$ — Regarding the level of offset in measurements, we consider a situation where a nominal value $m _ { 0 }$ and a precision $p _ { 0 }$ are available and we define

$$
f _ { M } ( m _ { e } ) = \mathcal { N } ( m _ { e } ; m _ { 0 } , p _ { 0 } ^ { - 1 } ) .\tag{4}
$$

In the numerical study of Sect. IV, we will consider the poorly informative case: $p _ { 0 }$ is small (and $m _ { 0 } = 0 )$ .

Noise parameter: scale $\gamma _ { e } \textrm { -- }$ Regarding $\gamma _ { e }$ (for notational convenience $\gamma _ { e } = 1 / v _ { e } )$ , a classical choice is a Gamma pdf:

$$
f _ { \Gamma } ( \gamma ) = \mathcal { G } ( \gamma ; a _ { 0 } , b _ { 0 } )\tag{5}
$$

This choice makes it easy to consider a nominal value with uncertainty based on the mean $a _ { 0 } / b _ { 0 }$ and the variance $a _ { 0 } / b _ { 0 } ^ { 2 }$

Diffusion prior for the images — This prior is described using a diffusion model [14]–[16]: essentially, available examples are transformed into noise, and conversely, new examples are generated by transforming noise realisations. To achieve this, the methodology consists in introducing (i) T latent variables $\pmb { x } _ { 1 : T }$ (in addition to $\scriptstyle { \pmb x } _ { 0 } )$ and an extended prior $\pi _ { 0 : T } ( \pmb { x } _ { 0 : T } )$ and (ii) two joint pdfs for $\pmb { x } _ { 0 : T } \pmb { : \mathrm { . } }$ a forward denoted $p _ { 0 : T } ^ { + }$ and a backward denoted $p _ { 0 : T } ^ { - }$ . For practical efficiency, both are chosen in Markovian form:

$$
\begin{array} { r c l } { p _ { 0 : T } ^ { + } ( { \pmb x } _ { 0 : T } ) } & { = } & { p _ { 0 } ^ { + } ( { \pmb x } _ { 0 } ) \displaystyle \prod _ { t = 1 } ^ { T } p _ { t \mid t - 1 } ^ { + } ( { \pmb x } _ { t } \mid { \pmb x } _ { t - 1 } ) } \end{array}\tag{6}
$$

$$
\begin{array} { r l r } { p _ { 0 : T } ^ { - } ( { \pmb x } _ { 0 : T } ) } & { = } & { p _ { T } ^ { - } ( { \pmb x } _ { T } ) \prod _ { t = 1 } ^ { T } p _ { t - 1 \mid t } ^ { - } ( { \pmb x } _ { t - 1 } \mid { \pmb x } _ { t } ) } \end{array}\tag{7}
$$

which involves two terminal marginal pdfs $p _ { \mathrm { 0 } } ^ { + }$ and $p _ { T } ^ { - }$ and two sets of transition pdfs $p _ { t \mid t - 1 } ^ { + }$ and $p _ { t - 1 | t } ^ { - } .$ . Regarding the terminals

$$
p _ { 0 } ^ { + } ( { \pmb x } _ { 0 } ) = \pi _ { 0 } ( { \pmb x } _ { 0 } ) \quad \mathrm { a n d } \quad p _ { T } ^ { - } ( { \pmb x } _ { T } ) = \mathcal { N } ( { \pmb x } _ { T } ; { \bf 0 } , { \pmb I } )
$$

the first is the pdf $\pi _ { 0 }$ of the example set and the second is the pdf of noise (Gaussian, white and reduced). With regard to transitions, again for practical efficiency, Gaussians are chosen with the following parameters.

$$
\begin{array} { r c l } { p _ { t \mid t - 1 } ^ { + } ( { \pmb x } _ { t } \mid { \pmb x } _ { t - 1 } ) } & { = } & { \mathcal { N } ( { \pmb x } _ { t } ; k _ { t } { \pmb x } _ { t - 1 } , v _ { t } ^ { + } I ) } \end{array}\tag{8}
$$

$$
\begin{array} { r c l } { p _ { t - 1 | t } ^ { - } ( { \pmb x } _ { t - 1 } \mid { \pmb x } _ { t } ) } & { = } & { \mathcal { N } ( { \pmb x } _ { t - 1 } ; { \pmb \mu } _ { t } ( { \pmb x } _ { t } ) , { \pmb v } _ { t } ^ { - } { \pmb I } ) } \end{array}\tag{9}
$$

The function ${ \pmb { \mu } } _ { t } ( { \pmb x } )$ is described by a neural network $\pmb { \mu } _ { t } ^ { p } ( \pmb { x } )$ with parameter p and has two inputs: the image x and the time t. Replacing $\pmb { \mu } _ { t }$ by $\pmb { \mu } _ { t } ^ { p }$ in (9), and substituting in $( 7 )$ yields $p _ { 0 : T } ^ { - , p }$ . The learning stage adjusts p to minimise the Kullback distance between the forward $p _ { 0 : T } ^ { + }$ and the parametrized backward $p _ { 0 : T } ^ { - , p }$ pdfs while ensuring that the marginal pdfs for $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$ and ${ \bf { x } } _ { T }$ are

$$
\pi _ { 0 } = p _ { 0 } ^ { + } \simeq p _ { 0 } ^ { - } \mathrm { a n d } \mathcal { N } = p _ { T } ^ { - } \simeq p _ { T } ^ { + }
$$

$i . e . ,$ , that of the example set and the noise. It suffices then to report the adjusted value of $\pmb { p }$ in $p _ { 0 : T } ^ { - , p }$ to obtain an adjusted joint backward pdf (7). Therefore, based on the latter, it is easy to sample the prior for ${ \pmb x } _ { 0 : T } .$ , starting from $t = T$ downto to $t = 0$ and it is referred to as ancestral sampling.

## C. Full posterior

We can then construct the joint pdf and the posterior. The latter is based on the likelihood (2) and the priors for the parameters (3), (4), (5), and the joint prior for the images $( 6 ) - ( 7 )$ . Its construction relies on conditional independences encoded in the hierarchical model given in Fig. 1.

$$
\begin{array} { l c l } { { \pi _ { 0 : T } ( x _ { 0 : T } , \pmb \theta | y ) } } & { { \propto } } & { { \gamma _ { e } ^ { P / 2 } \exp - \gamma _ { e } \left\| ( { \pmb y } - m _ { e } ) - { \pmb H } _ { \iota } { \pmb x } _ { 0 } \right\| ^ { 2 } / 2 } } \\ { { } } & { { } } & { { \gamma _ { e } ^ { a _ { 0 } - 1 } \exp \left[ - b _ { 0 } \gamma _ { e } \right] \mathbb { 1 } _ { + } ( \gamma _ { e } ) } } \\ { { } } & { { } } & { { \exp \left[ - p _ { 0 } ( m _ { e } - m _ { 0 } ) ^ { 2 } / 2 \right] } } \\ { { } } & { { } } & { { \mathcal { U } ( \pmb \iota ) } } \\ { { } } & { { } } & { { \pi _ { 0 : T } ( { \pmb x } _ { 0 : T } ) } } \end{array}
$$

Due to the intricate nature of this pdf, it is not possible to compute the estimations and uncertainties directly. To this end, an MCMC sampler is used, as shown below.

## III. PROPOSED SAMPLER: A GIBBS SCHEME

To explore the posterior, we resort to a Gibbs loop that splits the global sampling problem in easier sub-problems. More precisely, the conditional posterior of each unknown is sequentially sampled under its conditional density, in an iterative way. The samples form a Markov chain whose distribution converges to the posterior [22], [23].

Remark 1 — The paper [17] is related to the work proposed here but there are two key differences. First, the estimation of an instrument parameter is considered in [17] but not the estimation of noise parameters (neither the offset nor the power), and second the Gibbs algorithm in [17] structures the alternation between the images and the instrument parameters but not between the latent variables themselves.

![](images/5bf4282b306bbd0c352a774bb5e4227d0555c42caa6f7529dba3e247d7a1cef1.jpg)  
Fig. 1. Hierarchy $- \mathbf { \nabla } \mathbf { x } _ { 0 }$ is the image of interest, ${ \pmb x } _ { 1 : T }$ are the latent images and y is the measured image (blurred and noisy version of the true x ). θ contains the parameters of the observation (response and error), and its estimation is the core of the article. This graph already shows that if we know how to sample the x properly including the conditional independences encoded by this hierarchy, the difficulty of sampling θ is greatly alleviated.

For each unknown, the conditional pdf given the other unknowns is needed. Each one is proportional to the posterior (10) and hence only involves the factors including the considered unknown. Given the hierarchy in Fig. 1, several simplifications arise, which both facilitate the theoretical calculations and reduces the computational load. The conditional posteriors are now given. For notational simplicity $\bar { \pmb { y } } = \pmb { y } - m _ { e }$

## A. Image

This section describes the sampling of the extended image $\pmb { x } _ { 0 : T }$ . Up to a factor, the pdf writes

$$
\exp \left[ - \frac { 1 } { 2 } \gamma _ { e } \left\| \bar { \pmb y } - \pmb { H } _ { \iota } \pmb { x } _ { 0 } \right\| ^ { 2 } \right] \pi ( \pmb { x } _ { 0 : T } ) .
$$

We resort to G-DPS presented in [1]. It is itself a block-Gibbs sampler: it samples each x<sub>t</sub> in turn under its conditional pdf $\pi _ { t \mid \star } ( \mathbf x _ { t } | \mathbf y , \pmb \theta , \pmb x _ { \star \setminus t } )$ where (t | ⋆) is the time t given all the other times (from 0 to T ) except t and $( { \star } \backslash t )$ denotes the set of all times (from 0 to T ) except t. The original idea of [1] is to play with both forward and backward pdfs. More specifically, the sampling is based on the posterior attached to the

• forward $\pi _ { 0 : T } ^ { + } ( \pmb { x } _ { 0 : T } | \pmb { y } , \pmb { \theta } )$ for the latent variables ${ \mathbf { x } } _ { 1 : T } ,$ and

• backward $\pi _ { 0 : T } ^ { - } ( \pmb { x } _ { 0 : T } | \pmb { y } , \pmb { \theta } )$ for the image of interest $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$

This idea is justified by the fact that the two joint priors $\pi _ { 0 : T } ^ { + } ( \pmb { x } _ { 0 : T } )$ and $\pi _ { 0 : T } ^ { - } ( \pmb { x } _ { 0 : T } )$ are similar thanks to the learning stage. So, we consider here that they are identical, then the convergence is considered as guaranteed. Overall, the entire algorithm is both simple and efficient for three reasons.

1) It requires the sampling of Gaussians only (see also [24])

2) All the covariances are diagonal be it in the Fourier domain (t = 0) or in the spatial one $( t \neq 0 )$

3) In addition, means and variances are easy to compute, by FFT (t = 0) or linear combination $( t \neq 0 )$

The main technical details are reported in Appendix and the full details are [1].

## B. Noise parameter scale $\gamma _ { e }$

Up to a factor, the conditional posterior for $\gamma _ { e }$ reads

$$
\begin{array} { l } { { \ \gamma _ { e } ^ { P / 2 } \exp \left[ - \frac { \gamma _ { e } } { 2 } \left. \bar { y } - H _ { \iota } x \right. ^ { 2 } \right] \gamma _ { e } ^ { a _ { 0 } - 1 } \exp \left[ - b _ { 0 } \gamma _ { e } \right] \mathbb { 1 } _ { + } ( \gamma _ { e } ) } } \\ { { = \gamma _ { e } ^ { a _ { 0 } + P / 2 - 1 } \exp \left[ - \gamma _ { e } \left( b _ { 0 } + \frac { 1 } { 2 } \left. \bar { y } - H _ { \iota } x \right. ^ { 2 } \right) \right] \mathbb { 1 } _ { + } ( \gamma _ { e } ) } } \end{array}
$$

and the advantage of a conjugacy becomes apparent at this point: the conditional posterior for $\gamma _ { e }$ is in the same family as the prior, namely a Gamma pdf. The parameters are:

$$
\left\{ \begin{array} { l l l } { { a } } & { { = } } & { { a _ { 0 } + P / 2 } } \\ { { b } } & { { = } } & { { b _ { 0 } + \left\| \left( { \pmb y } - m _ { e } \right) - { \pmb H } _ { \iota } { \pmb x } \right\| ^ { 2 } / 2 } } \end{array} \right.
$$

the sampling is then direct and efficient.

## C. Noise parameter offset $m _ { e }$

The conditional posterior for $m _ { e }$ clearly appears as:

$$
{ \begin{array} { r l } & { \exp \left[ - { \frac { \gamma _ { e } } { 2 } } \left\| ( y - m _ { e } ) - H _ { \iota } x \right\| ^ { 2 } \right] \exp \left[ - p _ { 0 } ( m _ { e } - m _ { 0 } ) ^ { 2 } / 2 \right] } \\ { = } & { \exp \left[ - { \frac { p } { 2 } } ( m _ { e } - m ) ^ { 2 } \right] } \end{array} }
$$

up to a factor, that is a Gauss pdf with precision and mean

$$
\left\{ \begin{array} { r c l } { { p } } & { { = } } & { { p _ { 0 } + P \gamma _ { e } } } \\ { { m } } & { { = } } & { { p ^ { - 1 } \left( p _ { 0 } m _ { 0 } + \gamma _ { e } { \bf 1 } ^ { \mathrm { t } } ( y - H _ { \it x } x ) \right) } } \end{array} \right.
$$

and the sampling is also direct and efficient. At this point also, the advantage of a conjugacy is apparent (the prior and the conditional posterior are in the same family).

## D. Instrument parameter

The conditional posterior for the instrument parameter ι is also proportional to the joint posterior (10):

$$
\exp \left[ - \frac { \gamma _ { e } } { 2 } \left. \bar { \pmb { y } } - \pmb { H _ { \iota } } \pmb { x } \right. ^ { 2 } \right] \mathcal { U } ( \pmb { \iota } )
$$

This pdf is not an usual one and cannot be directly sampled. Among existing sampling algorithms [22], [23], [25], we resort to a Metropolis-Hasting step. Within this family of algorithms, several options are available (independent, random-walk,. . . ). Here it is efficient to make use of random-walk Metropolis-Hasting with a Gauss excursion.

## IV. NUMERICAL ASSESSMENT

In order to demonstrate the feasibility and interest of the proposed Hyper-G-DPS, this Section proposes an experimental study. It relies on a toy problem based on the MNIST example set. The method has been implemented<sup>1</sup> and the information regarding the architecture and learning stage are given in [26]. The ground-truth ${ \pmb x } ^ { \star }$ is a sample of the learned prior (size 32× 32, gray level roughly in [0, 1]). The PSF is a Lorentz shape with width parameter $\iota ^ { \star } = 0 . 9$ and regarding the noise $\sigma _ { e } ^ { \star } = $ 0.05 and $m _ { e } = 0 . 1$ . The ground-truth $\scriptstyle { \pmb x } ^ { \star }$ and the measurement (blurred and noisy image) $^ { \mathbf { \psi } _ { \mathbf { \psi } } }$ are shown in Fig. 4 (left and middle). Here are some implementation details.

![](images/c0611acb95bd56aa1878aedd2afd54f0185b0f5b883c4ee132d49d7205061690.jpg)

• Regarding the scan order, the algorithm repeats this pattern: update observation parameters $v _ { e }$ and $m _ { e }$ , then $\iota ,$ followed by the images for $t = 0$ up to $t = T$

• The image $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ is initialized to $\mathbf { \nabla } _ { \mathbf { \mu } _ { y . } }$ The $\pmb { x } _ { 1 : T }$ are set to successive noisy versions through the forward model. The width ι and the error mean $m _ { e }$ are initialized at random under their prior. Given the scan order and the Gibbs structure, there is no need to initialize $v _ { e }$

• As the iterations proceed, the empirical average of the images is updated. The algorithm stops when the difference between successive updates is smaller than a threshold.

Remark — The algorithm has been run numerous times under identical and different scenarios, including variations in ground truth, noise level, PSF and initialisations. It has consistently exhibited both qualitative and quantitative behaviour.

## A. Results

Fig. 2 shows a typical result regarding the three unknown parameters. The chains exhibit standard behaviour: the distributions quickly stabilise and appear stationary after about only 300 iterations (burn-in period). From a qualitative view, Fig. 2 shows that the estimated values are nearby the true values. The quantitative results are reported in Tab. I.

![](images/0d33560878bc7942886745d0279b7f7ff9e9d81088bc8843af868c38a25521e3.jpg)

![](images/2a438530664a8559d97d593a8cd7fb74a650ab9e466c63a5a4fb0ea759ef9dd9.jpg)

![](images/df0aa03086a4c691b306508b77e9a9eb2fe77214a8e22727186e5ff31fc1f47f.jpg)

![](images/942229cd5aaf711f115f304def6146357c14b6c6703611762160632f6d5b053a.jpg)

![](images/e7c6819e22ac70926a891b8b58c1a668c8085d7d3d6598675cd053216578e0c9.jpg)  
Fig. 2. Samples provided by the Gibbs algorithm as a function of iteration index (left) and as histograms (right) for the three unknown parameters from top to bottom: $\iota , m _ { e }$ and $v _ { e } .$ They are samples of one dimensional marginal pdfs. The green lines / dots give the true value. See Fig. 3 for two dimensional joint-marginal posterior and Tab. I for quantitative results.

The proposed strategy provides optimal estimations $( e . g .$ Posterior Mean as the MMSE) and additionally coherent tools for uncertainty quantification based on posterior standard deviations. For each parameter, it is clear from Tab. I that the true value lies within the interval centered on the estimate and of width two standard deviations.

Fig. 3. Point clouds for two dimensional marginals pdfs for the three unknown parameters: $\iota , m _ { e }$ and $v _ { e } .$ From left to right: $( m _ { e } , v _ { e } ) , ( \iota , v _ { e } )$ , and $( \iota , m _ { e } )$ The samples are given in blue and the true values is given in green. See also $\mathrm { F i g } . 2$ for one dimensional plots and Tab. I for quantitative assessment.
<table><tr><td></td><td>l</td><td> $m _ { e }$ </td><td> $v _ { e } ~ ( \times 1 0 ^ { 3 } )$ </td></tr><tr><td>True</td><td>0.80</td><td>-0.050</td><td>2.50</td></tr><tr><td>Estimate</td><td>0.77</td><td>-0.051</td><td>2.53</td></tr><tr><td>Error</td><td>0.030</td><td>0.0010</td><td>0.026</td></tr><tr><td>Error</td><td>3.8%</td><td>2.1%</td><td>1.1%</td></tr><tr><td>PSD</td><td>0.053</td><td>0.0049</td><td>0.122</td></tr><tr><td>± 2PSD</td><td>√</td><td>√</td><td>√</td></tr></table>

TABLE I  
Results for the three unknown parameters $\iota , m _ { e }$ and $v _ { e } \colon$ true and estimated values (first and second row) then the error (third row). The Posterior Standard Deviation (PSD) is then given and the $\checkmark$ indicates that the true value does lie within the interval centered on the estimate and of width two PSD. See also Figs. 2 and 3.

Finally, Fig. 4-right yield the estimated image. The blur and the noise are significantly reduced in the resulting image (Fig. 4-right) with respect to the measurement (Fig. 4-middle) and it closely matches the original image (Fig. 4-left). This result is confirmed by the cross-sections also shown in Fig. 4. From Fig. 5 it is clear that, for each pixel, the true value also lies within the interval centered on the estimate and of width two standard deviations.

![](images/427e2378a6e7827885498650a0b2828431683af145e4d6f530cf25ae41ec1925.jpg)

![](images/021721f2f9c55f142ee422115fa2fa0ac191f3c4e7070e2fc55c578e828a4e78.jpg)

![](images/d22e44c39a1b851904ca07ef1d59412fec403fca8c970ef3dd2a201ae02df05e.jpg)

![](images/02128cb160060c636df7b8912bb3cb1ce2e6d8bf87933b3ccf315a08425e7f74.jpg)

![](images/2ea852072be600a0e69cc033c5ddb4fdabe92ef09b83cd6d90a2483a6d5f3177.jpg)

![](images/7481f6d21fc35405cbff0b607de00479887a134f54d994757909f77854e563b1.jpg)  
Fig. 4. Left to right: true image $\mathbf { \delta } _ { \pmb { x } ^ { \star } } .$ , measurements y and estimated image x. The figure shows the images themselves (top) and cross-sections (bottom).

![](images/59c171490560cc997f5458c191cd85e4f4ce989af9cdd32f439735ae5daca399.jpg)  
Fig. 5. Cross-sections of ${ \pmb x } ^ { \star }$ (plain green) and the “uncertainty” intervals (dashed blue) centered on the estimate and of width three standard deviations.

## B. Efficiency, computation time and some comments

The algorithm produces N samples of the images and the parameters, $\pmb { x } _ { 0 : T } ^ { ( n ) }$ and ${ \pmb \theta } ^ { ( n ) }$ for $n = 1 , \dots N$ under the joint posterior pdf for $\pmb { x } _ { 0 : T }$ and θ. Note that each iteration n involves updating the three parameters $\pmb { \theta } ~ = ~ \left[ \iota , m _ { e } , v _ { e } \right]$ and all the images x<sub>t</sub> (for $t = 0 , \ldots , T )$

As mentioned earlier, as the iterations progress, the empirical average of the images $\pmb { x } _ { 0 } ^ { ( n ) }$ (which approximates the posterior mean) is updated. The iterations stop when the difference between successive updates becomes smaller than a threshold, here set to $1 0 ^ { - 2 }$ . The algorithm thus iterated $N = 9 5 2$ times taking 62 seconds. Most of the computations time (about 80%) is due to the passage through the network.

A particular feature of the Hyper-G-DPS sampling scheme, inherited from G-DPS, is that, ultimately, each iteration n (updating all the $\mathbf { \nabla } _ { \mathbf { x } _ { t } } .$ , for $t = 0 , 1 , \dots T )$ requires only a single pass through the neural network (to update $\pmb { x } _ { 0 } ^ { ( n ) } )$ ). Therefore, scaling up to larger images does not appear to be an obstacle.

Another major practical advantage of Hyper-G-DSP, also inherited from G-DPS, is that it does not require the adjustment or tuning of any algorithm parameters (apart from the threshold that puts an end to the iterations), unlike many other algorithms, e.g., [18], [19].

## V. CONCLUSION

This paper deals with numerical methods for solving inverse problems when the observation model is linear with additive Gauss noise. The focus is on the delicate issue of estimating multiple parameters of the observation system: width of the point spread function and also mean and variance of noise / error. This issue has already been addressed in the literature and several solutions have been proposed, but not really (see Remark 1) in cases where the prior for the image is defined on the basis of a diffusion model. It is the specificity of the proposed method to estimate them in addition to the image of interest in that case. To this end, our recent contribution [1] allows for proper handling of conditional distributions for images, thereby enabling the inclusion of the conditional posterior for parameters and posterior given these parameters. More precisely, a Gibbs loop splits the overall problem in far simpler sub-problems: iteratively sample each parameter and each image under its conditional posterior.

The simulation study focuses on parameter estimation issue and it is based on the MNIST example set. The proposed method provides accurate and coherent elements for uncertainty quantification, as well as accurate parameters estimation and image restoration. The numerical study also confirms the remarkable computational efficiency.

Conclusively, the paper addresses the crucial question of estimating instrument and noise parameters, in addition to the unknown image, in inverse problem based on a diffusion prior. It provides a novel solution referred to as Hyper-G-DPS, that is shown to be accurate and efficient.

To go further, it would be interesting to address model selection [27], [28], especially selection of a model for instrument an / or noise from a given list [29].

## APPENDIX

This appendix provides computational details regarding the Gibbs algorithm used to sample the extended image $\pmb { x } _ { 0 : T }$ . Our previous paper [1] introduced this algorithm and gives more details. It samples each $\mathbf { x } _ { t }$ in turn under its conditional pdf $\pi _ { t \mid \star } ( \mathbf x _ { t } | \mathbf y , \pmb \theta , \pmb x _ { \star \setminus t } )$ where $( t | \star )$ is the time t given all the other times (from 0 to T ) except t and $( { \star } \backslash t )$ denotes the set of all times (from 0 to T ) except t. The structure of these conditional pdfs relies on the hierarchy shown in Fig. 1.

1 - Image of interest — Regarding ${ \pmb x } _ { 0 } ,$ it is sampled under

$$
\begin{array} { r l } & { \pi _ { 0 | \star } ( \pmb { x } _ { 0 } | \pmb { y } , \pmb { \theta } , \pmb { x } _ { \star \setminus 0 } ) } \\ & { \qquad \propto \ \pi _ { 0 | 1 } ^ { - } ( \pmb { x } _ { 0 } | \pmb { x } _ { 1 } ) \ f ( \pmb { y } | \pmb { x } _ { 0 } , \pmb { \theta } ) } \\ & { \qquad = \ \mathcal { N } ( \pmb { x } _ { 0 } ; \pmb { \mu } _ { 1 } ( \pmb { x } _ { 1 } ) , \pmb { v } _ { 1 } ^ { - } I ) \ \mathcal { N } ( \pmb { y } ; m _ { e } + H \pmb { x } _ { 0 } , \pmb { v } _ { e } I ) } \end{array}
$$

which reveals a linear-Gauss problem and the Wiener / Tikhonov solution. ${ \mathrm { S o } } ,$ the conditional posterior is Gauss with precision and expectation.

$$
\left\{ \begin{array} { l l l } { \Gamma _ { \mathrm { 0 } } } & { = } & { H _ { \iota } ^ { \mathrm { t } } H _ { \iota } / v _ { e } + I / v _ { 1 } ^ { - } } \\ { \varepsilon _ { \mathrm { 0 } } } & { = } & { \Gamma _ { \mathrm { 0 } } ^ { - 1 } \left[ H _ { \iota } ^ { \mathrm { t } } ( y - m _ { e } ) / v _ { e } + \pmb { \mu } _ { 1 } ( \pmb { x } _ { 1 } ) / v _ { 1 } ^ { - } \right] } \end{array} \right.
$$

Sampling is particularly effective in the Fourier plane: the components are independent and Gaussian, and their mean and variance are easily obtained by simple FFT [9].

2.1 - Latent images $( t \neq T ) -$ The $\mathbf { x } _ { t }$ are sampled under

$$
\begin{array} { r l } & { \pi _ { t | \star } ( \pmb { x } _ { t } | \pmb { y } , \pmb { x } _ { \star \setminus t } ) } \\ & { \qquad \propto \ \pi _ { t | t - 1 } ^ { + } ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } ) \ \pi _ { t + 1 | t } ^ { + } ( \pmb { x } _ { t + 1 } | \pmb { x } _ { t } ) } \\ & { \qquad = \ \mathcal { N } ( \pmb { x } _ { t } ; k _ { t } \pmb { x } _ { t - 1 } , \pmb { v } _ { t } ^ { + } I ) \ \mathcal { N } ( \pmb { x } _ { t + 1 } ; k _ { t + 1 } \pmb { x } _ { t } , \pmb { v } _ { t + 1 } ^ { + } I ) } \end{array}
$$

also yields a Gauss pdf with precision $\gamma _ { t } I$ and expectation $\varepsilon _ { t }$

$$
\left\{ \begin{array} { l c l } { { \gamma _ { t } } } & { { = } } & { { 1 / v _ { t } ^ { + } + k _ { t + 1 } ^ { 2 } / v _ { t + 1 } ^ { + } } } \\ { { \varepsilon _ { t } } } & { { = } } & { { \gamma _ { t } ^ { - 1 } \left( k _ { t } \pmb x _ { t - 1 } / v _ { t } ^ { + } + k _ { t + 1 } \pmb x _ { t + 1 } / v _ { t + 1 } ^ { + } \right) } } \end{array} \right.
$$

2.2 - Latent image $( t = T ) -$ For the case of ${ \pmb x } _ { T }$

$$
\begin{array} { r } { \pi _ { T | \star } ( \pmb { x } _ { T } | \pmb { y } , \pmb { x } _ { \star \setminus T } ) = \pi _ { T | T - 1 } ^ { + } ( \pmb { x } _ { T } | \pmb { x } _ { T - 1 } ) \qquad } \\ { = \ N ( \pmb { x } _ { T } ; k _ { T } \pmb { x } _ { T - 1 } , \pmb { v } _ { T } ^ { + } \pmb { I } ) } \end{array}
$$

$i . e .$ , simply the last step in the forward process: a Gaussian with precision $\gamma _ { T } I$ and expectation $\varepsilon _ { T }$

$$
\left\{ \begin{array} { l c l } { { \gamma _ { T } } } & { { = } } & { { 1 / v _ { T } ^ { + } } } \\ { { \varepsilon _ { T } } } & { { = } } & { { k _ { T } \ : { \pmb x } _ { T - 1 } } } \end{array} \right.
$$

## ACKNOWLEDGMENT

The author warmly thanks Liam Moroy, Guillaume Bourmaud, Fred´ eric Champagnat, Marcelo Pereyra and Charlesquin´ Kemajou for their help.

This work is conducted within project PEPR Origins, reference ANR-22-EXOR-0016, supported by the France 2030 plan managed by Agence Nationale de la Recherche. It also received financial support from the French government in the framework of the University of Bordeaux’s France 2030 program RRI Origins.

[1] J.-F. Giovannelli, “A Gibbs posterior sampler for inverse problem based on prior diffusion model,” EUSIPCO, European Signal Processing Conference, Aug. 2026.

[2] P. C. Hansen, Discrete Inverse Problems: Insight and Algorithms. Philadelphia, USA: SIAM, 2010.

[3] J. Kaipio and E. Somersalo, Statistical and computational inverse problems. Berlin, Germany: Springer, 2005.

[4] J. C. Santamarina and D. Fratta, Discrete Signals and Inverse Problems: An Introduction for Engineers and Scientists. Chichester, England: WileyBlackwell, 2005.

[5] C. R. Vogel, Computational Methods for Inverse Problems, ser. Frontiers in Applied Mathematics. SIAM, 2002, vol. 23.

[6] A. Yan, R. Fetick, L. Mugnier, J.-F. Giovannelli, and C. Petit, “Ro-´ bust blind deconvolution of adaptive-optics corrected images: Marginal approach with a support constraint,” to appear in Astron. Astrophys., 2026.

[7] M. Pereyra, N. Dobigeon, H. Batatia, and J.-Y. Tourneret, “Estimating the granularity coefficient of a Potts-Markov random field within a Markov Chain Monte Carlo algorithm,” IEEE Trans. Image Processing, vol. 22, no. 6, pp. 2385–2397, 2013.

[8] F. Orieux, J.-F. Giovannelli, T. Rodet, H. Ayasso, M. Husson, and A. Abergel, “Super-resolution in map-making based on a physical instrument model and regularized inversion. Application to SPIRE/Herschel.” Astron. Astrophys., vol. 539, Mar. 2012.

[9] F. Orieux, J.-F. Giovannelli, and T. Rodet, “Bayesian estimation of regularization and point spread function parameters for Wiener–Hunt deconvolution,” J. Opt. Soc. Amer., vol. 27, no. 7, July 2010.

[10] N. Dobigeon, A. Hero, and J.-Y. Tourneret, “Hierarchical Bayesian sparse image reconstruction with application to MRFM,” IEEE Trans. Image Processing, vol. 18, no. 9, Sep. 2009.

[11] T. Bishop, R. Molina, and J. Hopgood, “Blind restoration of blurred photographs via AR modelling and MCMC,” in Proc. IEEE ICIP, Oct. 2008.

[12] P. Campisi and K. Egiazarian, Eds., Blind Image Deconvolution. CRC Press, 2007.

[13] L. Mugnier, T. Fusco, and J.-M. Conan, “MISTRAL: a myopic edgepreserving image restoration method, with application to astronomical adaptive-optics-corrected long-exposure images,” J. Opt. Soc. Amer., vol. 21, no. 10, pp. 1841–1854, Oct. 2004.

[14] S. H. Chan, Tutorial on Diffusion Models for Imaging and Vision, ser. Foundations and Trends in Machine Learning. Hanover, MA, USA: Now Publishers Inc, Jan. 2024.

[15] F. D. S. Ribeiro and B. Glocker, Demystifying Variational Diffusion Models, ser. Foundations and Trends in Machine Learning. Hanover, MA, USA: Now Publishers Inc, Jan. 2025.

[16] P. Nakkiran, A. Bradley, and M. Zhou, Hattieand Advani, Step-by-Step Diffusion: An Elementary Tutorial, ser. Foundations and Trends in Machine Learning. Hanover, MA, USA: Now Publishers Inc, 2025.

[17] N. Murata, K. Saito, C.-H. Lai, Y. Takida, T. Uesaka, Y. Mitsufuji, and S. Ermon, “GibbsDDRM: A partially collapsed Gibbs sampler for solving blind inverse problems with denoising diffusion restoration,” in International Conference on Machine Learning, Honolulu, Hawai, USA, 2023.

[18] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, and J. C. Ye, “Diffusion posterior sampling for general noisy inverse problems,” International Conference on Learning Representations, 2024.

[19] J. Song, A. Vahdat, M. Mardani, and J. Kautz, “Pseudoinverse-guided diffusion models for inverse problems,” in International Conference on Learning Representations, 2023.

[20] A. Yan, L. Mugnier, J.-F. Giovannelli, R. Fetick, and C. Petit, “Marginal-´ ized myopic deconvolution of adaptive optics corrected images using MCMC methods,” Journal of Astronomical Telescopes Instruments and Systems, vol. 9, no. 4, Nov. 2023.

[21] R. J.-L. Fetick, L. M. Mugnier, T. Fusco, and B. Neichel, “Blind de-´ convolution in astronomy with adaptive optics: the parametric marginal approach,” Monthly Notices of the Royal Astronomical Society, vol. 496, no. 4, pp. 4209–4220, Aug. 2020.

[22] C. P. Robert, The Bayesian Choice. From decision-theoretic foundations to computational implementation, ser. Springer Texts in Statistics. New York, USA: Springer Verlag, 2007.

[23] S. Brooks, A. Gelman, G. L. Jones, and X.-L. Meng, Handbook of Markov Chain Monte Carlo. Boca Raton, USA: Chapman & Hall / CRC, 2011.

[24] N. Yismaw, U. S. Kamilov, and M. S. Asif, “Gaussian is all you need: A unified framework for solving inverse problems via diffusion posterior sampling,” IEEE Trans. Computational Imaging, vol. 11, 2025.

[25] M. Girolami and B. Calderhead, “Riemannian manifold Hamiltonian Monte Carlo (with discussion),” J. R. Statist. Soc. B, vol. 73, pp. 123– 214, 2011.

[26] MatlabDoc (fr.mathworks.com/help/), “Generate images using diffusion,” deeplearning/ug/generate-images-using-diffusion.html, 2023.

[27] T. Ando, Bayesian model selection and statistical modeling. Boca Raton, USA: Chapman & Hall/CRC, 2010.

[28] J. Ding, V. Tarokh, and Y. Yang, “Model selection techniques: An overview,” IEEE Signal Proc. Mag., vol. 35, no. 6, pp. 16–34, Nov. 2018.

[29] B. Harroue, J.-F. Giovannelli, and M. Pereyra, “An optimal Bayesian´ strategy for comparing Wiener-Hunt deconvolution models in the absence of ground truth,” Inverse Problems (Special Issue on Big Data Inverse Problems), vol. 40, no. 10, 2024.