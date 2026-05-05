# ONE-STEP FLOW FOR IMAGE SUPER-RESOLUTION WITH TUNABLE FIDELITY-REALISM TRADE-OFFS

Yuanzhi ${ \bf Z } { \bf h } { \bf u } ^ { \ast \dagger { 1 } , 2 }$ Ruiqing Wang\*1 Shilin ${ \bf L } { \bf u } ^ { 3 }$ Junnan Li2 Hanshu Yan2 Kai Zhang‡1 1Nanjing University 2Rhymes.AI 3Nanyang Technological University

# ABSTRACT

Recent advances in diffusion and flow-based generative models have demonstrated remarkable success in image restoration tasks, achieving superior perceptual quality compared to traditional deep learning approaches. However, these methods either require numerous sampling steps to generate high-quality images, resulting in significant computational overhead, or rely on common model distillation, which usually imposes a fixed fidelity-realism trade-off and thus lacks flexibility. In this paper, we introduce OFTSR, a novel flow-based framework for one-step image super-resolution that can produce outputs with tunable levels of fidelity and realism. Our approach first trains a conditional flow-based superresolution model to serve as a teacher model. We then distill this teacher model by applying a specialized constraint. Specifically, we force the predictions from our one-step student model for same input to lie on the same sampling ODE trajectory of the teacher model. This alignment ensures that the student model’s singlestep predictions from initial states match the teacher’s predictions from a closer intermediate state. Through extensive experiments on datasets including FFHQ $( 2 5 6 \times 2 5 6 )$ , DIV2K, ImageNet $( 2 5 6 \times 2 5 6 )$ ) and real world SR datasets, we demonstrate that OFTSR achieves state-of-the-art performance for one-step image superresolution, while having the ability to flexibly tune the fidelity-realism tradeoff. Code and pre-trained models are available at https://github.com/yuanzhizhu/OFTSR and https://huggingface.co/Yuanzhi/OFTSR, respectively.

# 1 INTRODUCTION

Recently, diffusion and flow-based generative models have demonstrated the ability to generate images with higher quality (Ramesh et al., 2022; Nichol & Dhariwal, 2021; Dhariwal & Nichol, 2021) than earlier generative models such as Generative Adversarial Networks (GANs) (Goodfellow et al., 2020; Karras et al., 2019), Normalizing Flows (NFs) (Dinh et al., 2016) and Variational Autoencoders (VAEs) (Kingma & Welling, 2013; Razavi et al., 2019). Beyond visual generation, diffusion models have shown remarkable success across a variety of tasks, including image editing (Hertz et al., 2022; Brooks et al., 2023; Kawar et al., 2023), 3D content generation (Poole et al., 2022; Wang et al., 2023a; Liu et al., 2023b; Wu et al., 2022; Wang et al., 2024a), and image restoration (Kawar et al., 2022; Chung et al., 2022; Wang et al., 2022b; Zhu et al., 2023; Delbracio & Milanfar, 2023; Lin et al., 2023), with particularly notable advancements in image super-resolution (SR) (Saharia et al., 2021; Chen et al., 2023; Yue et al., 2024b; Wang et al., 2024b).

Existing diffusion and flow-based SR methods can be broadly divided into two approaches: trainingfree methods (Zhu et al., 2023; Kawar et al., 2022; Wang et al., 2022b; Chung et al., 2022; Alkhouri et al., 2024; Mardani et al., 2023; Song et al., 2023a), and training-based methods (Saharia et al., 2021; Luo et al., 2023b; Liu et al., 2023a; Yue et al., 2023; Wang et al., $2 0 2 4 \mathrm { c }$ ; Yue et al., 2024b; Liu et al., 2024; Delbracio & Milanfar, 2023). Training-free methods decompose the conditional probability into a prior term and a likelihood term, with each term associating directly to a specific subproblem (Zhu et al., 2023). During iterative sampling, the prior subproblem is naturally handled by pre-trained unconditional diffusion models, which serve as powerful regularizers to guide the solution toward realistic High Resolution (HR) images. Meanwhile, the likelihood subproblem is addressed through specialized optimization techniques or analytical approximations to ensure fidelity to the observed Low Resolution (LR) image. On the other hand, training-based methods directly model the conditional probability using paired data, either by training from scratch (Saharia et al., 2021; Delbracio & Milanfar, 2023) or by incorporating additional control modules into existing generative priors (Wang et al., 2024b; Yu et al., 2024; Lin et al., 2023; Rombach et al., 2022). Several other bridge-based methods (Luo et al., 2023b; Liu et al., 2023a; Yue et al., 2023; Chung et al., 2024) have also been proposed for general image-to-image translation tasks, sharing similarities with direct learning approaches.

![](images/9fc51ba765fa36d9fc9e5c103d757bcff5c74a738b1cabd535b1154ba2b9b062.jpg)  
Figure 1: (a) Our final model takes the concatenation of a low-resolution image with its noise-augmented version as input, and is able to generate high-resolution outputs with either high realism or high fidelity by adjusting the interpolation parameter $t$ . We indicate the PSNR and LPIPS value on the output images. (b) Comparison of different diffusion and flow based image super-resolution methods on the ImageNet $2 5 6 \times 2 5 6$ dataset. Bubble radius indicates the NFEs used by the methods.

Despite the promising results of the above methods, they require many iterative sampling steps to achieve high perceptual quality, and reducing the number of iterations often results in higher fidelity but lower perceptual quality. In this sense, their fidelity-realism trade-offs are achieved at the cost of more sampling steps. In order to achieve high perceptual quality with fewer sampling steps, some attempts (Wang et al., 2024c; Lee et al., 2024; Wu et al., 2024; Xie et al., 2024; Li et al., 2024) have been made to distill the diffusion sampling process into a single step with diffusion distillation approaches (Luhman & Luhman, 2021; Salimans & Ho, 2022; Liu et al., 2022; Song et al., 2023b; Yan et al., 2024; Yin et al., 2024b;a; Sauer et al., 2025). However, while these methods improve efficiency, they sacrifice flexibility by limiting control over the fidelity-realism trade-off, reducing their applicability in domains where different tasks require varying levels of fidelity and realism, such as medical imaging, remote sensing and film upscaling (Greenspan, 2009; Li et al., 2023a; Wang et al., 2022a; Mentzer et al., 2020; Joshi et al., 2025).

In this paper, we propose OFTSR that achieves one-step image SR and preserves the capability to produce outputs with tunable fidelity-realism trade-offs. Specifically, OFTSR uses a two-stage pipeline. In stage one we train a noise-augmented conditional rectified flow to expand the support of the initial distribution: noise-perturbed LR images form the initial distribution while the LR images are used as conditions, enabling diverse HR reconstructions from a single LR. In the second stage, a distillation strategy is proposed to restrict the student model’s predictions to match the same Ordinary Differential Equation (ODE) induced by the teacher model from the first stage.

Our main contributions can be summarized as follows:

• Noise-augmented Conditional Rectified Flow for Image Restoration: We introduce an enhanced conditional rectified flow model for image restoration. By leveraging a noiseaugmented LR conditioning strategy, our approach enables more effective LR-conditioned diffusion restoration, serving as both a general restoration framework and the foundational stage for our proposed distillation algorithm.

• One-Step Diffusion Distillation with Flexible Fidelity-Realism Trade-off: We introduce a distillation strategy applicable to empirical probability flow ODEs of any pre-trained conditional diffusion or flow model. Unlike prior methods that limit flexibility, ours enables one-step sampling while preserving control over fidelity and perceptual realism for SR.

• State-of-the-Art (SOTA) Performance on Benchmark Datasets: Extensive experiments on DIV2K (Agustsson & Timofte, 2017), FFHQ (Karras et al., 2019), ImageNet (Deng et al., 2009) and several real world SR dataset including RealSR (Cai et al., 2019), RealSet80 (Yue et al., 2024b) and RealLQ250 (Ai et al., 2025) show that OFTSR achieves competitive one-step reconstruction, surpassing recent SOTA methods in both perceptual quality and fidelity.

# 2 BACKGROUND

# 2.1 DIFFUSION AND FLOW-BASED GENERATIVE MODELS

Drawing inspiration from non-equilibrium thermodynamics, diffusion models operate through two core processes: a forward diffusion process that gradually adds Gaussian noise to data until it becomes pure noise, and a reverse denoising process that systematically reconstructs the original data by removing noise (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020b). Let $\mathbf { x } _ { t }$ represent the data $\mathbf { x }$ at timestep $t$ . The forward process can be formally described by the Ito Stochastic ˆ Differential Equation (SDE) (Song et al., 2020b):

$$
\mathrm { d } \mathbf { x } _ { t } = f _ { t } \mathbf { x } _ { t } \mathrm { d } t + g _ { t } \mathrm { d } \mathbf { w } ,
$$

where w is the standard Wiener process, $f _ { t } : \mathbb { R } \to \mathbb { R }$ is the drift coefficient, and $g _ { t } : \mathbb { R }  \mathbb { R }$ is a scalar function called the diffusion coefficient.

For every diffusion process described by Eq. (1), there exists a corresponding deterministic Probability Flow Ordinary Differential Equation (PF-ODE) that maintains the same marginal probability density:

$$
\frac { \mathrm { d } \mathbf { x } _ { t } } { \mathrm { d } t } = f _ { t } \mathbf { x } _ { t } - \frac { 1 } { 2 } g _ { t } ^ { 2 } \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } ) ,
$$

where $p _ { t } ( \cdot )$ represents the marginal probability density at time $t$ . The term $\nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } )$ is known as the score function, which can be approximated by a neural network $\mathbf { s } _ { \theta } ( \mathbf { x } , t )$ with parameters $\theta$ . This network is typically trained using score matching techniques (Hyvarinen¨ $\&$ Dayan, 2005; Song & Ermon, 2019; Song et al., 2020a).

To generate data samples, the process begins with Gaussian noise drawn from an initial Gaussian distribution $p _ { 0 }$ and solves Eq. (2) numerically from $t = 0$ to $t = 1$ . By utilizing the learned score function $\mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t )$ , the empirical PF-ODE can be obtained as: $\begin{array} { r } { \frac { \mathrm { d } \mathbf { x } _ { t } } { \mathrm { d } t } = f _ { t } \mathbf { x } _ { t } - \frac { 1 } { 2 } g _ { t } ^ { 2 } \mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t ) } \end{array}$ .

Rectified flow (Liu et al., 2022; Liu, 2022; Lipman et al., 2022; Esser et al., 2024) is a generative modeling framework based on ODEs. Given an initial distribution $p _ { 0 }$ and a target data distribution $p _ { 1 }$ , rectified flow trains a neural network to parameterize a velocity field using the following loss function:

$$
\mathcal { L } _ { \mathrm { r f } } ( \theta ) : = \mathbb { E } _ { \mathbf { x } _ { 1 } \sim p _ { 1 } , \mathbf { x } _ { 0 } \sim p _ { 0 } } \left[ \int _ { 0 } ^ { 1 } \bigg | \Big | \mathbf { v } _ { \theta } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \bigg | \bigg | _ { 2 } ^ { 2 } \mathrm { d } t \right] , \mathrm { ~ w h e r e ~ } \mathbf { x } _ { t } = ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 } .
$$

raito mple generation is achieved by solving the empirical ODE . In practical implementations, this empirical ODE is sol $\begin{array} { r } { \frac { \mathrm { d } \mathbf { x } _ { t } } { \mathrm { d } t } = \mathbf { v } _ { \theta } ( \mathbf { x } _ { t } , t ) } \end{array}$ from using $t ~ = ~ 0$ $t \ = \ 1$ standard ODE solvers, ranging from the simple forward Euler method to higher-order methods such as RK2 and RK45.

# 2.2 PERCEPTION-DISTORTION TRADE-OFF

The perception-distortion (realism-fidelity) trade-off (Blau & Michaeli, 2018) is a fundamental concept in image restoration. It describes the inherent trade-off between perceptual realism and fidelity to the ground truth, and mathematically proves that it is generally not possible to achieve both good perceptual realism and high fidelity simultaneously.

To address this challenge, researchers have explored various approaches to enable tunable trade-offs between these two desirable qualities. One common technique involves interpolating between the weights of two models with the same architecture, trained with GAN loss and mean squared error loss (Wang et al., 2018). Recently, diffusion models have emerged as a promising approach for this task. The iterative sampling nature of diffusion models provides a flexible means of controlling the desired trade-offs. By adjusting the Number of Function Evaluations (NFEs), users can generate reconstructions that better match their specific requirements (Chung et al., 2024). Specifically, lower NFEs tend to result in reconstructions with reduced distortion, as the output regresses towards the mean (Delbracio & Milanfar, 2023). Conversely, higher NFEs prioritize perceptual quality, even if it comes at the expense of some distortion from the ground truth (similar to Fig. 6).

# 3 METHOD

In this section, we introduce the OFTSR framework for one-step SR that can restore HR images with either high realism or high fidelity. We achieve this goal through a two-stage process: first, we train a direct flow-based model for SR, and then we distill this model into a simplified one-step variant. In Sec. 3.1, we present a simple noise-augmented conditional flow that expands the support of the initial distribution, enabling diverse reconstruction. In Sec. 3.2, we propose to distill the student model by restricting its predictions on the same ODE using teacher model from Sec. 3.1.

![](images/493ab02c54e87842102afc0d93ce9671715f26b9c2c435afa657093170189e40.jpg)  
Figure 2: Illustration of the proposed distillation loss. Rather than directly distilling from the teacher, we leverage the teacher to align the one-step intermediate outputs, $\mathbf { x } _ { t }$ and $\mathbf { x } _ { s }$ , along teacher’s PF-ODE trajectory. For simplicity, LR conditioning is omitted in this figure.

# 3.1 NOISE AUGMENTED CONDITIONAL FLOW

Unlike diffusion models, flow-based models have the advantage that their initial distribution is not limited to Gaussian distributions. This flexibility suggests a natural approach for image restoration - directly learning a flow that maps the distribution of LR images $( p _ { \mathrm { L R } } )$ to that of HR images $( p _ { \mathrm { H R } } )$ . However, our initial experiments (see Tab. 7) showed poor performance with this direct approach, aligning with findings from several recent works (Delbracio & Milanfar, 2023; Kim et al., 2024; Lee et al., 2024). This training procedure tends to collapse the LR $\mathrm {  H R }$ mapping: during inference each LR is driven toward a single HR.

To overcome this limitation, we propose a noise-augmented approach to process LR images. For any input image $\mathbf { x } _ { \mathrm { L R } }$ , we construct our initial distribution $p _ { 0 } ( \mathbf { \dot { x } } ) = p _ { \mathrm { L R } } ^ { \sigma _ { p } }$ by adding Gaussian noise with standard deviation $\sigma _ { p }$ . Specifically, we adopt a Variance-Preserving (VP) noising operation (Ho et al., 2020; Song et al., 2020b):

$$
\begin{array} { r } { \mathbf { x } _ { 0 } = \sqrt { 1 - \sigma _ { p } ^ { 2 } } \mathbf { x } _ { \mathrm { L R } } + \sigma _ { p } \epsilon , } \end{array}
$$

where $\epsilon$ is a standard Gaussian noise. While this noise perturbation facilitates better generalization, it inevitably causes information loss in the LR image. To address this, we incorporate $\mathbf { x } _ { \mathrm { L R } }$ as a conditional input to our model as in Fig. 1. This VP formulation, together with the condition $\mathbf { x } _ { \mathrm { L R } }$ , makes our method particularly versatile, encompassing previous approaches as special cases. When $\sigma _ { p } = 0$ , our method reduces to the minimal augmentation case in InDI (Delbracio & Milanfar, 2023), and when $\sigma _ { p } = 1$ , it matches the training strategy of SR3 (Saharia et al., 2022).

Given this noise-augmented formulation, we can now define our training objective as:

$$
\mathcal { L } _ { \mathrm { f l o w } } ( \theta ) = \mathbb { E } _ { \mathbf { x } _ { 1 } \sim p _ { 1 } } \left[ \int _ { 0 } ^ { 1 } \mathbb { D } \bigg ( \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t ) , ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \bigg ) \mathrm { d } t \right] ,
$$

where $\mathbb { D }$ is a discrepancy loss that measures the difference between two images (e.g., $\ell _ { 2 }$ loss or the $\ell _ { 1 }$ loss), $\mathbf { v } _ { \theta }$ is our velocity model, $\mathbf { x } _ { t , \mathrm { L R } } = \mathrm { c o n c a t } ( \mathbf { x } _ { t } , \mathbf { x } _ { \mathrm { L R } } )$ is the concatenation of $\mathbf { x } _ { t }$ and $\mathbf { x } _ { \mathrm { L R } }$ in channel dimension (see Fig. 1), The LR input of the algorithm is given by $\mathbf { x } _ { \mathrm { L R } } = \mathcal { H } ^ { T } ( \mathcal { H } ( \mathbf { x } _ { 1 } ) + \mathbf { n } )$ , where $\mathcal { H }$ is the downsampling operator, $\bar { \mathcal { H } } ^ { T }$ is its transpose and $\mathbf { n }$ is i.i.d. Gaussian noise with variance $\sigma _ { n } ^ { 2 }$ . The perturbed version of $\mathbf { x } _ { \mathrm { L R } }$ , denoted as $\mathbf { x } _ { \mathrm { 0 } }$ , is obtained using the noise augmentation strategy described in Eq. (4). Additionally, ${ \bf x } _ { t } = ( 1 - t ) { \bf x } _ { 0 } + t { \bf x } _ { 1 }$ denotes the intermediate state as in rectified flow (Liu et al., 2022; Liu, 2022).

# 3.2 DISTILLATION LOSS

We introduce a distillation loss to train a one-step student that preserves the pre-trained SR flow’s fidelity–realism trade-off, allowing control at inference via a single hyperparameter $t$ . As shown in Fig. 6 and observed in prior work (Delbracio & Milanfar, 2023; Liu et al., 2023a), single-step estimates of the final state $\mathbf { x } _ { 1 } ^ { t }$ obtained from an intermediate state $\mathbf { x } _ { t }$ lie on a fidelity–realism curve: along the ODE sampling trajectory, estimates for larger $t$ (closer to 1) exhibit richer detail and lower LPIPS (better realism), whereas estimates for smaller $t$ (closer to 0) are blurrier but achieve lower MMSE and higher PSNR (better fidelity).

To preserve the fidelity-realism trade-off, given the same input ${ \bf x } _ { \mathrm { 0 , L R } }$ , for two different timesteps $t$ and $s$ where $s > t$ , we require the student model $\mathbf { v } _ { \phi }$ to produce two corresponding intermediate states $\mathbf { x } _ { t }$ and $\mathbf { x } _ { s }$ that lie on the same ODE trajectory defined by the teacher (see Fig. 2):

$$
\mathbf { x } _ { s } = \mathbf { x } _ { t } + ( s - t ) \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t ) ,
$$

where ${ \bf x } _ { 0 , \mathrm { L R } } = \mathrm { c o n c a t } ( { \bf x } _ { 0 } , { \bf x } _ { \mathrm { L R } } )$ is the concatenation of the input image $\mathbf { x } _ { \mathrm { 0 } }$ and the LR condition $\mathbf { x } _ { \mathrm { L R } }$ along the channel dimension. The intermediate states $\mathbf { x } _ { t }$ and $\mathbf { x } _ { s }$ can be computed using our one-step student model $\mathbf { v } _ { \phi }$ :

$$
\mathbf { x } _ { t } = \mathbf { x } _ { 0 } + t \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t ) .
$$

Substituting the expression for the intermediate image $\mathbf { x } _ { t }$ and $\mathbf { x } _ { s }$ from Eq. (7) into Eq. (6), we have the following constraint on the student model:

$$
s ( { \mathbf v } _ { \phi } ( { \mathbf x } _ { 0 , \mathrm { L R } } , s ) - { \mathbf v } _ { \phi } ( { \mathbf x } _ { 0 , \mathrm { L R } } , t ) ) = ( s - t ) ( { \mathbf v } _ { \theta } ( { \mathbf x } _ { t , \mathrm { L R } } , t ) - { \mathbf v } _ { \phi } ( { \mathbf x } _ { 0 , \mathrm { L R } } , t ) ) .
$$

Similar to BOOT, we can set $\mathrm { d } t = s - t$ and derive the final distillation loss:

$$
\mathcal { L } _ { \mathrm { d i s t i l } } ( \phi ) = \mathbb { E } _ { \mathbf { x } _ { 1 } \sim p _ { 1 } , t \sim \mathcal { U } [ 0 , 1 ] } \left[ \left\| \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , s ) - S \mathbf { G } \left[ \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t ) + \frac { \mathrm { d } t } { s } \left( \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t ) - \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t ) \right) \right] \right\| _ { 2 } ^ { 2 } \right] ,
$$

where $\operatorname { S G } [ \cdot ]$ is the stop-gradient operator for training stability (Gu et al., 2023; Tee et al., 2024). Since $s - t = \mathrm { d } t$ and $t > 0$ , we do not have the ‘dividing by $0 ^ { \circ }$ issue in (Tee et al., 2024). Similarly to (Song et al., 2023b; Gu et al., 2023), we can use the Euler or general RK2 solver to calculate $\mathbf { v } _ { \theta }$ in Eq. (9). In our main experiments, we employ the midpoint method, while also evaluating two other RK2 solver variants, i.e., Heun’s method and Ralston’s method, for comparison in our ablations (see Tab. 8). In Sec. B.2, we show that our distillation loss is the discrete-time counterpart of the forward distillation loss (Boffi et al., 2025; Liu, 2025) by fixing the start timestep at 0, which is highly related to recent work MeanFlow (Geng et al., 2025) and AlignYourFlow (Sabour et al., 2025).

# 3.3 ALIGNMENT AND BOUNDARY LOSS

In BOOT (Gu et al., 2023), a boundary condition is applied to enforce that the one-step student model and teacher model perform the same at the boundary $t = 0$ . We aim to align the teacher and student outputs in our model. The student produces $\mathbf { x } _ { 0 } + \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t )$ , while the teacher generates $\mathbf { x } _ { t } + ( 1 - t ) \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t )$ based on the student’s output $\mathbf { x } _ { t }$ using Eq. (7). By minimizing the difference between these outputs, we get the following alignment loss to align the teacher and student:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a l i g n } } ( \phi ) = \mathbb { E } _ { \mathbf { x } _ { 1 } \sim p _ { 1 } , t \sim \mathcal { U } [ 0 , 1 ] } \bigg [ \bigg \| ( 1 - t ) \bigg ( \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t ) - \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t ) \bigg ) \bigg \| _ { 2 } ^ { 2 } \bigg ] . } \end{array}
$$

If we consider this alignment loss only at $t = 0$ , it becomes equivalent to the boundary loss used in BOOT:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { B C } } ( \phi ) = \mathbb { E } _ { \mathbf { x } _ { 1 } \sim p _ { 1 } } \left[ \left\| \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , 0 ) - \mathbf { v } _ { \theta } ( \mathbf { x } _ { 0 , \mathrm { L R } } , 0 ) \right\| _ { 2 } ^ { 2 } \right] . } \end{array}
$$

Since it is difficult to sample $t = 0$ for most training iterations, we add in addition the boundary loss Eq. (11) in our final training objective.

The overall training objective. The student network $\mathbf { v } _ { \phi }$ is trained to minimize the combination of the aforementioned three losses terms:

$$
\mathcal { L } ( \phi ) = \mathcal { L } _ { \mathrm { d i s t i l l } } ( \phi ) + \lambda _ { \mathrm { a l i g n } } \mathcal { L } _ { \mathrm { a l i g n } } ( \phi ) + \lambda _ { \mathrm { B C } } \mathcal { L } _ { \mathrm { B C } } ( \phi ) ,
$$

where $\lambda _ { \mathrm { a l i g n } }$ and $\lambda _ { \mathrm { B C } }$ are the weights for alignment loss and boundary condition loss, respectively.   
The distillation stage of the proposed method is summarized in Algorithm 1.

![](images/285f0f974f2f1be9725113f863df28aa28e65f7c1adde48123606b90bb704cf2.jpg)

![](images/3c4fdaa803f4f85bca8804b9cea92175c7cd5129e56fba37451107f07a07da22.jpg)  
GT LR DPS DDRM DDNM DiffPir SITCOM OursFigure 4: Qualitative comparison with training-free methods. The first row shows noiseless SR on the FFHQ dataset, the second row presents noisy SR $\sigma _ { n } = 0 . 0 5 )$ on FFHQ, and the bottom row demonstrates noiseless SR on the ImageNet dataset. Numbers next to the method names represent the required NFEs.

Inference. After training, the one-step student $\mathbf { v } _ { \phi }$ produces the final high-resolution output $\mathbf { x } _ { 1 } ^ { t }$ in a single forward pass, conditioned on the initial state $\mathbf { x } _ { \mathrm { 0 } }$ , the low-resolution input $\mathbf { x } _ { \mathrm { L R } }$ , and the trade-off parameter $t$ . Concretely,

$$
{ \bf x } _ { 1 } ^ { t } = { \bf x } _ { 0 } + { \bf v } _ { \phi } ( { \bf x } _ { 0 } , { \bf x } _ { \mathrm { L R } } , t ) ,
$$

where $\mathbf { v } _ { \phi } ( \cdot )$ predicts a residual that refines $\mathbf { x } _ { \mathrm { 0 } }$ toward the desired point on the fidelity–realism curve specified by $t$ .

# 3.4 COMPARISON TO RELATED WORKS

In this section, we distinguish the proposed OFTSR from several closely related methods.

BOOT (Gu et al., 2023). Gu et al. proposed to make the prediction of the student model fulfill the Signal-ODE. In contrast, OFTSR directly constrains the student’s implicit prediction $\mathbf { x } _ { t }$ using the PF-ODE of the teacher model, leading to more concise and intuitive derivation and distillation objective. Moreover, while BOOT was originally designed for text-to-image generation using diffusion models, our method is built on rectified flow and demonstrates a smaller distillation gap compared to BOOT loss for SR task, and empirically achieves markedly better fidelity–realism trade-offs.

DAVI (Lee et al., 2024). Lee et al. introduced DAVI, which combines Variational Score Distillation (VSD) loss (Wang et al., 2024d; Luo et al., $2 0 2 3 \mathrm { a }$ ; Yin et al., 2024b) with data consistency loss to train a one-step SR model and utilizes the perturbation trick to present robust restoration ability. However, DAVI needs to train a fake score to track the denoising score of the one-step generator, resulting in reduced training efficiency.

SinSR (Wang et al., 2024c). Wang et al. proposed SinSR, which achieves near-teacher performance by distilling ResShift (Yue et al., 2024b) without adversarial training. However, SinSR requires simulation of the teacher model’s ODE trajectory, leading to computational overhead during training.

Our OFTSR stands out from other diffusion and flow-based SR methods due to its unique ability to restore images with either high perceptual quality or low distortion. This capability is novel among diffusion and flow-based approaches.

# 4 EXPERIMENTS

In this section, we provide experimental details and empirical evaluation of OFTSR and compare it with prior works.

<table><tr><td>DIV2K</td><td>Method</td><td>NFEs (↓)</td><td>PSNR (↑)</td><td>LPIPS (↓)</td><td>FID (↓)</td></tr><tr><td rowspan="4">Training- free</td><td>DPS (Chung et al., 2022)</td><td>1000</td><td>23.05</td><td>0.447</td><td>109.35</td></tr><tr><td>DDRM (Kawar et al., 2022)</td><td>20</td><td>27.87</td><td>0.285</td><td>23.38</td></tr><tr><td>DDNM (Wang et al., 2022b)</td><td>100</td><td>28.09</td><td>0.279</td><td>20.33</td></tr><tr><td>DiffPIR (Zhu et al., 2023)</td><td>100</td><td>27.94</td><td>0.248</td><td>19.56</td></tr><tr><td rowspan="8">Training- based</td><td>IRSDE (Luo et al., 2023b)</td><td>100</td><td>26.83</td><td>0.144</td><td>14.69</td></tr><tr><td>GOUB (Yue et al., 2023)</td><td>100</td><td>26.92</td><td>0.218</td><td>21.56</td></tr><tr><td>ECDB (Yue et al., 2024a)</td><td>10</td><td>27.39</td><td>0.212</td><td>18.88</td></tr><tr><td>InDI (Delbracio &amp; Milanfar, 2023)</td><td>100</td><td>26.45</td><td>0.136</td><td>15.39</td></tr><tr><td>Ours</td><td>31</td><td>26.76</td><td>0.128</td><td>14.10</td></tr><tr><td>Ours distilled (t = 1)</td><td>1</td><td>26.87</td><td>0.127</td><td>14.58</td></tr><tr><td>Ours distilled (t = 0.5)</td><td>1</td><td>28.02</td><td>0.208</td><td>16.89</td></tr><tr><td>Ours distilled (t = 0)</td><td>1</td><td>28.99</td><td>0.271</td><td>18.07</td></tr></table>

Table 1: Noiseless quantitative results on DIV2K. We compute the average PSNR (dB), LPIPS and FID of different methods on $4 \times$ SR. The best and second best results are highlighted in bold and underline. The distilled model produces superior performance in terms of trade-off metrics through adjustment of the hyperparameter $t$ .   
Table 2: Noiseless (top) and noisy (bottom) quantitative results on FFHQ $\pmb { 2 5 6 } \times 2 5 6$ . We compute the average PSNR (dB), LPIPS and FID of different methods on $4 \times$ SR. The best and second best results are highlighted in bold and underline.   

<table><tr><td colspan="2">FFHQ</td><td>Method</td><td>NFEs (↓)</td><td>PSNR (↑)</td><td>LPIPS (↓)</td><td>FID (↓)</td></tr><tr><td rowspan="9">σn = 0</td><td rowspan="5">Training- free</td><td>DPS (Chung et al., 2022)</td><td>1000</td><td>24.08</td><td>0.180</td><td>79.71</td></tr><tr><td>DDRM (Kawar et al., 2022)</td><td>20</td><td>28.81</td><td>0.118</td><td>89.12</td></tr><tr><td>DDNM (Wang et al., 2022b)</td><td>100</td><td>29.45</td><td>0.091</td><td>60.99</td></tr><tr><td>DiffPIR (Zhu et al., 2023)</td><td>100</td><td>29.13</td><td>0.073</td><td>44.49</td></tr><tr><td>SITCOM (Alkhouri et al., 2024)</td><td>20</td><td>29.29</td><td>0.089</td><td>43.00</td></tr><tr><td rowspan="4">Training- based</td><td>Ours</td><td>20</td><td>28.83</td><td>0.053</td><td>30.54</td></tr><tr><td>Ours distilled (t = 1)</td><td>1</td><td>28.98</td><td>0.055</td><td>36.02</td></tr><tr><td>Ours distilled (t = 0.5)</td><td>1</td><td>29.955</td><td>00.093</td><td>449.08</td></tr><tr><td>Ours distilled (t = 0)</td><td>1</td><td>31.25</td><td>0.150</td><td>66.76</td></tr><tr><td rowspan="9">σn = 0.05</td><td rowspan="4">Training- free</td><td>DPS (Chung et al., 2022)</td><td>1000</td><td>23.61</td><td>0.186</td><td>81.25</td></tr><tr><td>DDRM (Kawar et al., 2022)</td><td>20</td><td>26.71</td><td>0.191</td><td>113.25</td></tr><tr><td>DDNM (Wang et al., 2022b)</td><td>100</td><td>27.66</td><td>0.174</td><td>113.26</td></tr><tr><td>DiffPIR (Zhu et al., 2023)</td><td>100</td><td>26.99</td><td>0.123</td><td>61.66</td></tr><tr><td rowspan="4">Training-</td><td>SITCOM (Alkhouri et al., 2024)</td><td>20</td><td>27.80</td><td>0.158</td><td>83.04</td></tr><tr><td>DAVI (Lee et al., 2024)</td><td>1</td><td>27.50</td><td>0.084</td><td>50.19</td></tr><tr><td>Ours</td><td>20 1</td><td>27.28 227.71</td><td>0.080 0.081</td><td>46.04</td></tr><tr><td>Ours distilled (t = 1)</td><td></td><td></td><td></td><td>49.81</td></tr><tr><td></td><td>Ours distilled (t = 0.5) Ours distilled (t = 0)</td><td>1 1</td><td>$29.47 </td><td>0.157 0.172</td><td>82.93 85.89</td></tr></table>

# 4.1 EXPERIMENTAL SETUP

Datasets. We perform extensive SR experiments on the FFHQ $2 5 6 \times 2 5 6$ (Karras et al., 2019), DIV2K (Agustsson & Timofte, 2017) and ImageNet $2 5 6 \times 2 5 6$ (Russakovsky et al., 2015) datasets to assess the bicubic SR performance of OFTSR on faces and natural images. For each dataset, we evaluate on 100 hold-out validation images without cherry-picking. For evaluating real SR, we use both synthetic set and real world set. Synthetic set includes 100 images from Imagenet for $6 4  2 5 6$ SR and 100 images from DIV2K for $1 2 8 \to 5 1 2$ SR, both degraded using RealESRGAN pipeline. Real world test set includes RealSR (Cai et al., 2019), RealSet80 (Yue et al., 2024b) and RealLQ250 (Ai et al., 2025). For distilling DiT4SR, we construct the training set using a combination of images from DIV2K (Agustsson & Timofte, 2017), DIV8K (Gu et al., 2019), Flickr2K (Timofte et al., 2017), LSDIR (Li et al., 2023b) and the first 10K images from FFHQ (Karras et al., 2019).

Teacher Models. We employ three types of teacher models in our experiments: (1) Self-trained teachers (2 types of backbones: Guided Diffusion (Dhariwal & Nichol, 2021) for bicubic SR (Tabs. 1 to 3) and ResShift (Yue et al., 2024b) for real SR (Tabs. 4 and 5)) using the noise-augmented conditional flow strategy in Sec. 3.1, which showcases the effectiveness of our training scheme; (2) An off-the-shelf DiT4SR teacher (built on Stable Diffusion (SD) 3.5). Since SD-based models possess significantly stronger generative priors and are prohibitively expensive for us to pre-train, we distill DiT4SR with our method to enable fair comparison with the latest SOTA approaches for realSR (Tab. 6); (3) An off-the-shelf ResShift teacher, allowing direct comparison with SinSR (which is distilled from ResShift) and fair computational cost comparison (Tabs. 9 and 10).

Evaluation Metrics. The metrics we use for comparison are Peak Signal-to-Noise Ratio (PSNR), Frechet Inception Distance (FID), and Learned Perceptual Image Patch Similarity (LPIPS) (Zhang ´ et al., 2018) distance. The FID evaluates the visual quality by calculating the feature distance between two image distributions. In our experiments, we calculate the FID using the HR images and the restored images from the 100 hold-out validation set with Clean-FID (Parmar et al., 2022). LPIPS measures the average perceptual similarity between the restored images and their corresponding HR images. PSNR measures the restoration faithfulness between two images. And LPIPS and PSNR are the two main metrics we use to measure the perceptual-fidelity trade-offs. For real SR task, we also use no-reference Image Quality Assessment (IQA) including NIQE Zhang et al. (2015), CLIPIQA Wang et al. (2023b), MUSIQ Ke et al. (2021), and MANIQA Yang et al. (2022).

Compared Methods. We conduct comprehensive comparisons against state-of-the-art diffusionbased image super-resolution methods, which can be categorized into two groups: (1) Training-free methods, including DPS (Chung et al., 2022), DDRM (Kawar et al., 2022), DDNM (Wang et al., 2022b), DiffPIR (Zhu et al., 2023), CDDB (Chung et al., 2024), and SITCOM (Alkhouri et al., 2024); (2) Training-based methods: GOUB (Yue et al., 2023), ECDB (Yue et al., 2024a), InDI (Delbracio & Milanfar, 2023), DAVI (Lee et al., 2024), I2SB (Liu et al., 2023a), DDC (Chen et al., 2024), ResShift (Yue et al., 2024b), SinSR (Wang et al., 2024c) and CTMSR (You et al., 2025). And large scale Stable Diffusion based methods such as OSEDiff (Wu et al., 2024), AddSR (Xie et al., 2024) and TSDSR (Dong et al., 2025). It is noteworthy that SITCOM requires K inner-iterations to evaluate and differentiate the score function at each sampling step. To further validate the effectiveness of our method, we conduct experiment on real-world image super-resolution. Following (Yue et al., 2024b; Wang et al., 2024c), we use Imagenet $2 5 6 \times 2 5 6$ as HR training data and synthesize LR images using degradation pipeline of RealESRGAN (Wang et al., 2021).

Table 4: Quantitative results of real-world image superresolution on ImageNet $2 5 6 \times 2 5 6$ and RealSR (Cai et al., 2019). The best and second best results are highlighted in bold and underline. The number of inference steps is indicated by ‘s’, which is the same as NFE when not use CFG.

Table 3: Noiseless quantitative results on ImageNet $2 5 6 \times 2 5 6$ . We compute the average PSNR (dB), LPIPS and FID of different methods on $4 \times \mathrm { S R }$ . The best and second best results are highlighted in bold and underline.   

<table><tr><td>ImageNet</td><td>Method</td><td>NFEs (↓)</td><td>PSNR (↑)</td><td>LPIPS (↓)</td><td>FID (↓)</td></tr><tr><td rowspan="6">Training- free</td><td>DPS (Chung et al., 2022)</td><td>1000</td><td>20.36</td><td>0.438</td><td>164.99</td></tr><tr><td>DDRM (Kawar et al., 2022)</td><td>20</td><td>24.55</td><td>0.292</td><td>79.99</td></tr><tr><td>DDNM (Wang et al., 2022b)</td><td>100</td><td>25.19</td><td>0.327</td><td>84.98</td></tr><tr><td>DiffPIR (Zhu et al., 2023)</td><td>100</td><td>24.88</td><td>0.306</td><td>79.42</td></tr><tr><td>SITCOM (Alkhouri et al., 2024)</td><td>20</td><td>24.79</td><td>0.277</td><td>61.88</td></tr><tr><td>CDDB (Chung et al., 2024)</td><td>100</td><td>23.64</td><td>0.191</td><td>58.25</td></tr><tr><td rowspan="9">Training- based</td><td>I2SB (Liu et al., 2023a)</td><td>1000</td><td>23.36</td><td>0.178</td><td>60.99</td></tr><tr><td>I3SB (Wang et al., 2025)</td><td>25</td><td>23.79</td><td>0.169</td><td>59.38</td></tr><tr><td>I3SB (Wang et al., 2025)</td><td>1</td><td>25.24</td><td>0.157</td><td>124.47</td></tr><tr><td>DDC (Chen et al., 2024)</td><td>5</td><td>24.67</td><td>0.156</td><td>62.06</td></tr><tr><td>ResShift (Yue et al., 2024b)</td><td>4</td><td>23.68</td><td>0.207</td><td>60.75</td></tr><tr><td>SinSR (Wang et al., 2024c)</td><td>1</td><td>22.25</td><td>0.207</td><td>94.90</td></tr><tr><td>Ours</td><td>26</td><td>23.35</td><td>0.132</td><td>46.88</td></tr><tr><td>Ours distilled (t = 1)</td><td>1</td><td>24.20</td><td>0.135</td><td>52.69</td></tr><tr><td>Ours distilled (t = 0.5)</td><td>1</td><td>24.85</td><td>0.176</td><td>60.69</td></tr><tr><td></td><td>Ours distilled (t = 0)</td><td>1</td><td>26.18</td><td>0.284</td><td>92.04</td></tr></table>

<table><tr><td>ImageNet</td><td>PSNR (↑)</td><td>LPIPS (↓)</td><td>FID (↓)</td><td>NIQE (↓)</td><td>MUSIQ (↑)</td><td>MANIQA (↑)</td><td>CLIPIQA (↑)</td></tr><tr><td>SwinIR (Liang et al., 2021)</td><td>22.24</td><td>0.320</td><td>207.75</td><td>6.0084</td><td>47.46</td><td>0.5527</td><td>0.5544</td></tr><tr><td>NAFNet (Chen et al., 2022)</td><td>23.23</td><td>0.672</td><td>223.90</td><td>10.390</td><td>17.40</td><td>0.3025</td><td>037088</td></tr><tr><td>ResShif-15s (Yue et al., 2024b)</td><td>23.55</td><td>0.308</td><td>168.82</td><td>6.8026</td><td>49.95</td><td>0.5921</td><td>0.5906</td></tr><tr><td>SinSR-Is (Wang et al., 2024c)</td><td>23.19</td><td>0.302</td><td>157.10</td><td>6.1700</td><td>50.30</td><td>05789</td><td>0.5995</td></tr><tr><td>Ours-15s</td><td>22.51</td><td>0.308</td><td>153.34</td><td>5.9657</td><td>54.90</td><td>0.6151</td><td>0.6014</td></tr><tr><td>Ours distilled-1s (t = 1)</td><td>22.24</td><td>0.292</td><td>151.39</td><td>5.4043</td><td>54.16</td><td>0.5892</td><td>06066</td></tr><tr><td>Ours distilled-1s (t = 0.5)</td><td>23.85</td><td>0.396</td><td>201.77</td><td>8.9631</td><td>40.46</td><td>0.4905</td><td>0.4176</td></tr><tr><td>Ours distilled-1s (t = 0)</td><td>23.95</td><td>0.486</td><td>235.44</td><td>10.3695</td><td>35.04</td><td>0.3610</td><td>0.2886</td></tr><tr><td>RealSR</td><td>PSNR ()</td><td>LPIPS (↓)</td><td>FID (↓)</td><td>NIQE (↓)</td><td>MUSIQ (↑)</td><td>MANIQA (↑) CLIPIQA (↑)</td><td></td></tr><tr><td>ResShift-15s (Yue et al., 2024b)</td><td>26.26</td><td>0.347</td><td>142.57</td><td>7.1780</td><td>58.47</td><td>0.5343</td><td>0.5481</td></tr><tr><td>SinSR-Is (Wang et al., 2024c)</td><td>26.27</td><td>0.321</td><td>137.59</td><td>6.2773</td><td>60.84</td><td>0.5418</td><td>0.6224</td></tr><tr><td>Ours-15s</td><td>25.41</td><td>0.297</td><td>145.34</td><td>4.9089</td><td>65.48</td><td>0.5705</td><td>0.5826</td></tr><tr><td>Ours distilled-1s (t = 1)</td><td>25.27</td><td>0.288</td><td>142.38</td><td>4.6337</td><td>65.30</td><td>0.5604</td><td>0.5891</td></tr><tr><td>Ours distilled-1s (t = 0.5)</td><td>26.76</td><td>0.311</td><td>1175.11</td><td>6.9517</td><td>57.</td><td>0.4879</td><td>0.4251</td></tr><tr><td>Ours distilled-1s (t = 0)</td><td>27.01</td><td>0.331</td><td>190.63</td><td>8.1201</td><td>53.09</td><td>0.4205</td><td>0.3129</td></tr></table>

![](images/68ab2b8212b6cddb3d7ff0663cd0b487376ad9a355c276a14e52ee388633d0d6.jpg)  
Figure 5: Qualitative comparison with training-based methods. The first two columns demonstrate $4 \times \mathrm { S R }$ results on the FFHQ dataset with noise level $\sigma _ { n } = 0 . 0 5$ . The remaining columns show noiseless $4 \times \mathrm { S R }$ results on the ImageNet dataset. Numbers next to the method names represent the required NFEs.

Training Details. We do experiments for both noisy and noiseless SR. For noiseless SR, bicubic downsampling is performed on all three datasets. For noisy SR, we conduct experiment only on FFHQ $2 5 6 \times 2 5 6$ dataset with average-pooling downsampling and Gaussian noise with a standard deviation $\sigma _ { y } = 0 . 0 5$ . All images are normalized to the range of $[ - 1 , 1 ]$ . For experiments on FFHQ $2 5 6 \times 2 5 6$ and DIV2K, we adopt the same model architecture used for FFHQ in (Chung et al., 2022); and for experiment on ImageNet $2 5 6 \times 2 5 6$ , we use the same model architecture as the pretrained unconditional model used in (Dhariwal & Nichol, 2021). We modify the input convolution layer to accept concatenated image input. The first stage models are trained from scratch and are sampled with RK45 sampler by default. The one-step model is initialized from the teacher model for distillation. We use the Adam optimizer with a linear warmup schedule over 1k training steps, followed by a learning rate of $1 e - 4$ for both stages.

# 4.2 RESULTS

Quantitative Results. We present comprehensive quantitative evaluations on several benchmark datasets: DIV2K, FFHQ, ImageNet and real world test set and different tasks (including noiseless SR, noisy SR and real world SR) (Tabs. 1 to 6). Our analysis reveals several findings: (i) The first-stage OFTSR achieves superior performance in perceptual metrics (FID and LPIPS) while requiring fewer than 32 NFEs. (ii) Our distillation

Table 5: Quantitative comparison on real world sets. The best and second best results are in bold and underline.   

<table><tr><td>Datasets</td><td>Method</td><td>NIQE ↓</td><td>MUSIQ ↑</td><td>MANIQA ↑</td><td>CLIPIQA ↑</td><td>LIQE ↑</td></tr><tr><td rowspan="6">RealSet80</td><td>SwinIR</td><td>4.1601</td><td>63.72</td><td>0.5444</td><td>0.5919</td><td>3.6479</td></tr><tr><td>NAFNet</td><td>8.8794</td><td>35.16</td><td>0.3975</td><td>0.5289</td><td>1.0969</td></tr><tr><td>ResShift-15s</td><td>6.1955</td><td>61.35</td><td>0.5318</td><td>0.6702</td><td>3.4473</td></tr><tr><td>SinSR-1s</td><td>5.6182</td><td>63.96</td><td>0.5376</td><td>0.7242</td><td>3.6072</td></tr><tr><td>Ours-15s</td><td>4.3713</td><td>66.90</td><td>0.5617</td><td>0.6797</td><td>3.9982</td></tr><tr><td>Ours distilled-1s (t = 1)</td><td>4.1826</td><td>67.46</td><td>0.5570</td><td>0.6904</td><td>4.0168</td></tr><tr><td rowspan="6">RealLQ250</td><td>SwinIR</td><td>4.1628</td><td>60.48</td><td>0.5104</td><td>0.5352</td><td>3.0883</td></tr><tr><td>NAFNet</td><td>9.5524</td><td>25.97</td><td>0.3360</td><td>0.4095</td><td>1.0512</td></tr><tr><td>ResShift-15s</td><td>6.5731</td><td>59.98</td><td>0.5003</td><td>0.6239</td><td>2.9340</td></tr><tr><td>SinSR-1s</td><td>5.8200</td><td>63.73</td><td>0.5161</td><td>0.6990</td><td>3.2578</td></tr><tr><td>Ours-15s</td><td>4.2848</td><td>67.15</td><td>0.5481</td><td>0.6520</td><td>3.8367</td></tr><tr><td>Ours distilled-1s (t = 1)</td><td>4.0731</td><td>67.32</td><td>0.5287</td><td>0.6532</td><td>3.7211</td></tr></table>

algorithm is versatile, when applied to ResShift (Yue et al., 2024b) teacher, our distilled model achieved better one-step performance than SinSR (Wang et al., 2024c) (see Tab. 9). (iii) Our distilled version of OFTSR demonstrates remarkable versatility, achieving either the highest PSNR

Table 6: Quantitative comparison of state-of-the-art one-step SR methods on synthetic (DIV2K-Val) and real-world (RealLQ250) benchmarks. Best results are in bold, second best are underlined. Our method is tested under $t = 1$ . ResShift∗ means we train our noise-augmented conditional flow in Sec. 3.1 using the ResShift model architecture then distillation; and DiT4SR means we use the pretrained DiT4SR model as the teacher model for distillation.

<table><tr><td>Dataset</td><td>Metric</td><td>SinSR-1s</td><td>CTMSR-1s</td><td>AddSR-1s</td><td>OSEDiff-1s</td><td>TSDSR-1s</td><td>Ours (ResShift*)-1s</td><td>Ours (DiT4SR)-1s</td></tr><tr><td rowspan="9">DIV2K-Val</td><td>PSNR ↑</td><td>24.50</td><td>24.87</td><td>22.39</td><td>23.86</td><td>22.17</td><td>23.91</td><td>22.80</td></tr><tr><td>SSIM ↑</td><td>0.6136</td><td>0.6349</td><td>0.5652</td><td>0.6233</td><td>0.5680</td><td>0.6073</td><td>0.5774</td></tr><tr><td>LPIPS ↓</td><td>0.3164</td><td>0.3011</td><td>0.3728</td><td>0.2896</td><td>0.2679</td><td>0.3226</td><td>0.2716</td></tr><tr><td>DDISTS </td><td>0.2110</td><td>0.2102</td><td>0.2387</td><td>0.1999</td><td>0.1901</td><td>0.2081</td><td>0.1889</td></tr><tr><td>FID ↓</td><td>131.96</td><td>126.49</td><td>133.78</td><td>100.53</td><td>103.49</td><td>133.30</td><td>98.27</td></tr><tr><td>NIQE ↓</td><td>6.1721</td><td>5.3036</td><td>5.9929</td><td>4.9741</td><td>4.6621</td><td>4.9061</td><td>4.8399</td></tr><tr><td>MUSIQ ↑</td><td>64.26</td><td>66.59</td><td>63.39</td><td>68.53</td><td>71.19</td><td>68.71</td><td>70.25</td></tr><tr><td>MANIQA ↑</td><td>0.5442</td><td>0.5146</td><td>0.5657</td><td>0.6111</td><td>0.6010</td><td>0.5464</td><td>0.6145</td></tr><tr><td>CLIPIQA ↑</td><td>0.6687</td><td>0.6602</td><td>0.5734</td><td>0.6692</td><td>0.7221</td><td>0.6545</td><td>0.7233</td></tr><tr><td rowspan="5">RealLQ250</td><td>NIQE ↓</td><td>5.8200</td><td>4.5835</td><td>4.9235</td><td>3.9656</td><td>3.4868</td><td>4.0731</td><td>3.7802</td></tr><tr><td>MUSIQ ↑</td><td>63.73</td><td>68.00</td><td>66.82</td><td>69.55</td><td>72.09</td><td>67.32</td><td>72.60</td></tr><tr><td>MANIQA ↑</td><td>0.5161</td><td>0.5078</td><td>0.5304</td><td>0.5782</td><td>0.5829</td><td>0.5287</td><td>0.5904</td></tr><tr><td>CLIPIQA ↑</td><td>0.6990</td><td>0.6706</td><td>0.6437</td><td>0.6725</td><td>0.7221</td><td>0.6532</td><td>0.7252</td></tr><tr><td>LIQE ↑</td><td>3.2578</td><td>3.3373</td><td>3.4929</td><td>3.9039</td><td>4.0834</td><td>3.7211</td><td>4.1122</td></tr></table>

scores or ranking among the top two methods for FID and LPIPS metrics in one step. This indicates minimal performance degradation between the teacher and student models. (iv) Our experiments suggest that FID serves as a more reliable indicator of perceptual quality and better captures the performance gap between teacher and student models during distillation. (v) When applied to a powerful SD-based SR model (DiT4SR), our distillation algorithm produces a one-step generator whose performance is competitive with other leading SOTA distillation methods. This also validates the versatility of our distillation algorithm.

Visual Results. Our experimental results demonstrate that OFTSR achieves high-quality image reconstructions. We evaluate OFTSR against leading training-free methods for $4 \times \mathrm { S R }$ , as shown in Fig. 4. While DPS can produce sharp reconstructions, it requires 1000 NFEs and often introduces significant distortions. In contrast, OFTSR successfully preserves structural information from lowresolution inputs while reconstructing fine details. Notably, our distilled version of OFTSR requires only one NFE, as other training-free methods suffer from severe error accumulation when using less than 10 NFEs. As illustrated in Fig. 5, we also compare OFTSR against state-of-the-art SR methods that require training. The results show that our approach generates patterns with rich, natural details. Furthermore, our distilled model enables flexible control over the fidelity-realism trade-offs in the generated high-resolution images. Fig. 3 demonstrates this capability through examples of noisy $4 \times$ SR with varying degrees of realism and fidelity. More qualitative comparison and visual examples can be found in Sec. K

# 4.3 ABLATIONS

Perturbation Strength $\sigma _ { p }$ . In Tab. 7, we evaluate the design choices in the simple conditional flow training stage. All experiments in this ablation study are conducted under identical training conditions, with performance metrics measured using the RK45 solver. The most critical hyperparameter in this ablation is the strength of the perturbation $\sigma _ { p }$ . Consistent with previous works, we confirm that perturbation is essential for generating perceptually compelling images from LR inputs. Notably, we discover that increasing perturbation strength does not necessarily improve perceptual quality but instead leads to more curved PF-ODE, requiring additional NFEs to solve (see Tab. 7). Furthermore, our experiments demonstrate that conditioning on $\mathbf { x } _ { \mathrm { L R } }$ is crucial to compensate for information loss during perturbation. We also find that $\ell _ { 1 }$ loss outperforms $\ell _ { 2 }$ loss for our specific task. While (Kim et al., 2024) previously highlighted the significance of Gaussian perturbation, our work is the first to systematically analyze the relationship between noise perturbation and the trade-off between generation quality and efficiency in flow-based models.

Distillation Design Space. In Tab. 8, we evaluate several crucial design choices for the distillation stage, including the distillation loss type, solver type, $\mathrm { d } t$ value, and the weighting of alignment and boundary losses. Since learning ${ \mathbf { v } } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , 0 )$ is considerably easier than learning $\mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , 1 )$ , we utilize metrics from the latter to decide our distillation hyperparameters. Our analysis of the step size

Table 7: Ablation on noiseless FFHQ $2 5 6 \times 2 5 6$ first stage. The default training setting is ${ \mathfrak { b s } } = 3 2$ $\mathrm { ~ k ~ } = \ 0 . 0 0 0 1$ ; loss type $\mathit { \Theta } = \mathit { \Theta } \ell _ { 1 }$ ; with condition; all experiments are trained for $1 0 0 \mathrm { k }$ steps. The final choice is highlighted to balance the performance and efficiency.   

<table><tr><td>Strength of Perturbation </td><td>NFEs (↓)</td><td>PSNR (↑)</td><td>LPIPS (↓)</td><td>FID (↓)</td></tr><tr><td>0.</td><td>20</td><td>29.04</td><td>0.244</td><td>110.29</td></tr><tr><td>0.001</td><td>20</td><td>29.56</td><td>0.115</td><td>48.39</td></tr><tr><td>0.01</td><td>20</td><td>29.56</td><td>0066</td><td>34.70</td></tr><tr><td>0.1</td><td>20</td><td>28.83</td><td>0.053</td><td>30.54</td></tr><tr><td>0.2</td><td>27</td><td>28.84</td><td>0.053</td><td>30.77</td></tr><tr><td>0.3</td><td>32</td><td>28.88</td><td>0.053</td><td>31.02</td></tr><tr><td>0.5</td><td>32</td><td>28.86</td><td>0.053</td><td>30.22</td></tr><tr><td>0.8</td><td>44</td><td>28.84</td><td>0.054</td><td>31.02</td></tr><tr><td>1.</td><td>44</td><td>28.82</td><td>0.053</td><td>30.75</td></tr><tr><td>0.1 (no cond)</td><td>20</td><td>28.09</td><td>0.073</td><td>42.47</td></tr><tr><td>0.1 (2)</td><td>20</td><td>28.60</td><td>0.055</td><td>31.86</td></tr></table>

Table 8: Ablation on noiseless FFHQ $2 5 6 \times 2 5 6$ distillation stage. The default training setting is $\ b s \ = \ 8$ ; $\sigma _ { p } = 0 . 1$ , $\mathrm { l r } = 0 . 0 0 0 1$ ; loss type $= \ell _ { 2 }$ ; with LR condition; all experiments are trained for 20k steps; And the one-step metrics are calculated with $t = 1$ . Ablations in subgroups can be ordered as $\mathrm { d } t \  \ \lambda _ { \mathrm { B C } } \  \ \lambda _ { \mathrm { a l i g n } } \ $ dt reveals that smaller values do not necessarily yield better results, leading us to select $\mathrm { d } t = 0 . 0 5$ for subsequent experiments. Our proposed loss function demonstrates substantial improvement over both the original BOOT (Gu et al., 2023) loss and PINN (Tee et al., 2024) distillation loss, achieving a significant LPIPS score improvement of more than 0.1. Further experimentation shows that both the alignment loss (Eq. (10)) and boundary loss (Eq. (11)) contribute to enhanced performance. By combining these losses with a Midpoint 2-order solver, we achieve additional improvements in our one-step model’s performance at $t = 1$ .

Solver , and $\mathrm { d } t \ $ Distillation Loss .   

<table><tr><td>Distillation Loss</td><td>Solver</td><td>dt</td><td>λalign</td><td>λBC</td><td>PSNR (↑)</td><td>LPIPS (↓)</td></tr><tr><td>Ours</td><td>Euler</td><td>0.001</td><td>0</td><td>0</td><td>28.77</td><td>0.160</td></tr><tr><td>Ours</td><td>Euler</td><td>0.01</td><td>0</td><td>0</td><td>29.35</td><td>0.076</td></tr><tr><td>Ours</td><td>Euler</td><td>0.02</td><td>0</td><td>0</td><td>29.48</td><td>0.068</td></tr><tr><td>Ours</td><td>Euler</td><td>00.05</td><td>0</td><td>0</td><td>29.73</td><td>0.065</td></tr><tr><td>Ours</td><td>Euler</td><td>0.1</td><td>0</td><td>0</td><td>30.05</td><td>0.073</td></tr><tr><td>BOOT (Gu et al., 2023)</td><td>Euler</td><td>0.05</td><td>0</td><td>0</td><td>23.81</td><td>0.483</td></tr><tr><td>PINN (Tee et al., 2024)</td><td>Euler</td><td>0.05</td><td>0</td><td>0</td><td>27.92</td><td>0.250</td></tr><tr><td>Ours</td><td>Euler</td><td>0.05</td><td>0</td><td>0.1</td><td>29.73</td><td>0.064</td></tr><tr><td>Ours</td><td>Euler</td><td>0.05</td><td>0.01</td><td>0.1</td><td>29.69</td><td>0.063</td></tr><tr><td>Ours</td><td>Heun</td><td>0.05</td><td>0.01</td><td>0.1</td><td>29.21</td><td>0.057</td></tr><tr><td>Ours</td><td>Ralston</td><td>0.05</td><td>0.01</td><td>0.1</td><td>29.15</td><td>0.056</td></tr><tr><td>Ours</td><td>Midpoint</td><td>0.05</td><td>0.01</td><td>0.1</td><td>29.14</td><td>0.056</td></tr><tr><td>Ours (bs=32)</td><td>Midpoint</td><td>0.05</td><td>0.01</td><td>0.1</td><td>29.07</td><td>0.055</td></tr></table>

# 4.4 COMPUTATIONAL OVERHEAD

Training Cost Comparison. Our distillation algorithm is highly flexible and can be easily applied to any pre-trained diffusion/flow-based conditional model. As shown in Tab. 9, we applied our distillation algorithm to the ResShift(Yue et al., 2024b) pre-trained model and achieved teacherlevel performance in one step, surpassing SinSR(Wang et al., 2024c) in FID with much less training compute. Even taking the training stage into account with a larger model, our method remains more efficient than ResShift. We use $t = 1$ for OFTSR.

Table 9: Comparison of training cost on single NVIDIA A100.   

<table><tr><td>Method</td><td>[NFE(↓)]</td><td># Iterations</td><td>s/Iter</td><td>Training Time</td><td>PSNR (↑)</td><td>LPIPS (↓)</td><td>FID (↓)</td></tr><tr><td>DDC (Chen et al., 2024) (base model frozen) [5]</td><td></td><td>160k</td><td>0.89</td><td>~1.65 days</td><td>24.67</td><td>0.156</td><td>62.06</td></tr><tr><td>ResShift (Yue et al., 2024b) (teacher)</td><td>[4]</td><td>500k</td><td>1.32</td><td>~7.64 days</td><td>23.68</td><td>0.207</td><td>60.75</td></tr><tr><td>SinSR (Wang et al., 2024c) (ResShift teacher) [1]</td><td></td><td>30k</td><td>7.41</td><td>~2.57 days</td><td>22.25</td><td>0.207</td><td>94.90</td></tr><tr><td>OFTSR (ResShift teacher)</td><td>[1]</td><td>5k</td><td>6.72</td><td>~0.39 days</td><td>24.01</td><td>0.218</td><td>60.64</td></tr><tr><td>OFTSR (pre-train+distill)</td><td>[1]</td><td>100k + 50k</td><td>3.9/4.4</td><td>~4.51+2.54 days</td><td>24.20</td><td>0.135</td><td>52.69</td></tr></table>

Inference Cost Comparison. We have included a detailed comparison of the inference cost in Tab. 10, using FLOPS and MAC to measure model complexity. We use $t = 1$ for OFTSR.

Table 10: Comparison of inference cost on single NVIDIA A100.   

<table><tr><td>Method</td><td>[NFE(↓)]</td><td># Params (+VAE)</td><td>FID (↓)</td><td>MACs (+VAE) (↓)</td><td>FLOPs (+VAE) (↓)</td><td>Runtime (↓)</td></tr><tr><td>DDNM (Wang et al., 2022b) [100]</td><td></td><td>552.8M</td><td>84.98</td><td>1.11T</td><td>2.24T</td><td>7.00s</td></tr><tr><td>DDC (Chen et al., 2024)</td><td>[5]</td><td>552.8+113.7M</td><td>62.06</td><td>1.11+0.24T</td><td>2.24+0.49T</td><td>0.74s</td></tr><tr><td>ResShift (Yue et al., 2024b)</td><td> [4]</td><td>118.6+55.3M</td><td>60.75</td><td>50.1+473.5G</td><td>100.4+948.5G</td><td>0.27s</td></tr><tr><td>SinSR (Wang et al., 2024c)</td><td>[1]</td><td>118.6+55.3M</td><td>94.90</td><td>50.1+473.5G</td><td>100.4+948.5G</td><td>0.09s</td></tr><tr><td>OFTSR (ResShift teacher)</td><td>[1]</td><td>118.6+55.3M</td><td>60.64</td><td>50.1+473.5G</td><td>100.4+948.5G</td><td>0.09s</td></tr><tr><td>OFTSR (pre-train+distill)</td><td>[1]</td><td>552.8M</td><td>52.69</td><td>1.11T</td><td>2.24T</td><td>0.21s</td></tr></table>

# 5 CONCLUSION

In this paper, we introduced OFTSR, a novel approach to developing efficient one-step image superresolution models. Our extensive experiments on FFHQ, DIV2K, ImageNet and real world SR datasets demonstrate that our method significantly improves computational efficiency while maintaining high-quality image restoration capabilities. The proposed framework represents a promising direction in efficient image SR, effectively addressing the perception-distortion trade-off.

# ACKNOWLEDGEMENTS

Acknowledgments: This work was supported by the National Natural Science Foundation of China (Grant No. 62572234), the Gusu Innovation and Entrepreneurship Leading Talent Program (Grant No. ZXL20254324), and the Suzhou Key Technologies Project (Grant No. SGY2023136).

# REFERENCES

Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pp. 126–135, 2017.

Yuang Ai, Xiaoqiang Zhou, Huaibo Huang, Xiaotian Han, Zhengyu Chen, Quanzeng You, and Hongxia Yang. Dreamclear: High-capacity real-world image restoration with privacy-safe dataset curation. In NeurIPS, 2025.

Ismail Alkhouri, Shijun Liang, Cheng-Han Huang, Jimmy Dai, Qing Qu, Saiprasad Ravishankar, and Rongrong Wang. Sitcom: Step-wise triple-consistent diffusion sampling for inverse problems. arXiv preprint arXiv:2410.04479, 2024.

Yochai Blau and Tomer Michaeli. The perception-distortion tradeoff. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6228–6237, 2018.

Nicholas M Boffi, Michael S Albergo, and Eric Vanden-Eijnden. How to build a consistency model: Learning flow maps via self-distillation. arXiv preprint arXiv:2505.18825, 2025.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18392–18402, 2023.

Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, and Lei Zhang. Toward real-world single image superresolution: A new benchmark and a new model. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3086–3095, 2019.

Hanyu Chen, Zhixiu Hao, and Liying Xiao. Deep data consistency: a fast and robust diffusion model-based solver for inverse problems. arXiv preprint arXiv:2405.10748, 2024.

Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, and Jian Sun. Simple baselines for image restoration. In European conference on computer vision, pp. 17–33. Springer, 2022.

Zheng Chen, Yulun Zhang, Jinjin Gu, Xin Yuan, Linghe Kong, Guihai Chen, and Xiaokang Yang. Image super-resolution with text prompt diffusion. arXiv preprint arXiv:2311.14282, 2023.

Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, and Sungroh Yoon. Perception prioritized training of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11472–11481, 2022.

Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022.

Hyungjin Chung, Jeongsol Kim, and Jong Chul Ye. Direct diffusion bridge using data consistency for inverse problems. Advances in Neural Information Processing Systems, 36, 2024.

Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration. arXiv preprint arXiv:2303.11435, 2023.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021.

Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. arXiv preprint arXiv:1605.08803, 2016.

Linwei Dong, Qingnan Fan, Yihong Guo, Zhonghao Wang, Qi Zhang, Jinwei Chen, Yawei Luo, and Changqing Zou. Tsd-sr: One-step diffusion with target score distillation for real-world image super-resolution. In CVPR, 2025.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Do- ¨ minik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11): 139–144, 2020.

Hayit Greenspan. Super-resolution in medical imaging. The computer journal, 52(1):43–63, 2009.

Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Lingjie Liu, and Joshua M Susskind. Boot: Data-free distillation of denoising diffusion models with bootstrapping. In ICML 2023 Workshop on Structured Probabilistic Inference $\{ \backslash \& \}$ Generative Modeling, 2023.

Shuhang Gu, Andreas Lugmayr, Martin Danelljan, Manuel Fritsche, Julien Lamour, and Radu Timofte. Div8k: Diverse 8k resolution image dataset. 2019.

Nikita Gushchin, David Li, Daniil Selikhanovych, Evgeny Burnaev, Dmitry Baranchuk, and Alexander Korotin. Inverse bridge matching distillation. arXiv preprint arXiv:2502.01362, 2025.

Guande He, Kaiwen Zheng, Jianfei Chen, Fan Bao, and Jun Zhu. Consistency diffusion bridge models. Advances in Neural Information Processing Systems, 37:23516–23548, 2024.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

Aapo Hyvarinen and Peter Dayan. Estimation of non-normalized statistical models by score matching. ¨ Journal of Machine Learning Research, 6(4), 2005.

Dipali Joshi, Amit Jana, Harsh Lone, Vijay Taru, and Siddharth Thorat. Image and video upscaling using realesrgan. Journal Publication of International Research for Engineering and Management (JOIREM), 5(04), 2025.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4401–4410, 2019.

Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. arXiv preprint arXiv:2201.11793, 2022.

Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6007–6017, 2023.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5148–5157, 2021.

Beomsu Kim, Jaemin Kim, Jeongsol Kim, and Jong Chul Ye. Generalized consistency trajectory models for image manipulation. arXiv preprint arXiv:2403.12510, 2024.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Sojin Lee, Dogyun Park, Inho Kong, and Hyunwoo J Kim. Diffusion prior-based amortized variational inference for noisy inverse problems. arXiv preprint arXiv:2407.16125, 2024.

Jianze Li, Jiezhang Cao, Zichen Zou, Xiongfei Su, Xin Yuan, Yulun Zhang, Yong Guo, and Xiaokang Yang. Distillation-free one-step diffusion for real-world image super-resolution. arXiv preprint arXiv:2410.04224, 2024.

Xin Li, Yulin Ren, Xin Jin, Cuiling Lan, Xingrui Wang, Wenjun Zeng, Xinchao Wang, and Zhibo Chen. Diffusion models for image restoration and enhancement–a comprehensive survey. arXiv preprint arXiv:2308.09388, 2023a.   
Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, et al. Lsdir: A large scale dataset for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1775–1787, 2023b.   
Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1833–1844, 2021.   
Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Wanli Ouyang, Yu Qiao, and Chao Dong. Diffbir: Towards blind image restoration with generative diffusion prior. arXiv preprint arXiv:2308.15070, 2023.   
Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.   
Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A Theodorou, Weili Nie, and Anima Anandkumar. I 2 sb: Image-to-image schr\” odinger bridge. arXiv preprint arXiv:2302.05872, 2023a.   
Jiawei Liu, Qiang Wang, Huijie Fan, Yinong Wang, Yandong Tang, and Liangqiong Qu. Residual denoising diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2773–2783, 2024.   
Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022.   
Qiang Liu. Icml tutorial on the blessing of flow. International conference on machine learning, 2025.   
Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9298–9309, 2023b.   
Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.   
Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021.   
Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. arXiv preprint arXiv:2305.18455, 2023a.   
Ziwei Luo, Fredrik K Gustafsson, Zheng Zhao, Jens Sjolund, and Thomas B Sch ¨ on. Image restoration with ¨ mean-reverting stochastic differential equations. arXiv preprint arXiv:2301.11699, 2023b.   
Morteza Mardani, Jiaming Song, Jan Kautz, and Arash Vahdat. A variational perspective on solving inverse problems with diffusion models. arXiv preprint arXiv:2305.04391, 2023.   
Fabian Mentzer, George D Toderici, Michael Tschannen, and Eirikur Agustsson. High-fidelity generative image compression. Advances in Neural Information Processing Systems, 33:11913–11924, 2020.   
Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pp. 8162–8171. PMLR, 2021.   
Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11410–11420, 2022.   
Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.   
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.   
Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019.   
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution ¨ image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.   
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211–252, 2015.   
Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align your flow: Scaling continuous-time flow map distillation. arXiv preprint arXiv:2506.14603, 2025.   
Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. arXiv preprint arXiv:2104.07636, 2021.   
Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE transactions on pattern analysis and machine intelligence, 45(4):4713–4726, 2022.   
Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.   
Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pp. 87–103. Springer, 2025.   
Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pp. 2256–2265. PMLR, 2015.   
Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations, 2023a.   
Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 32, 2019.   
Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. Sliced score matching: A scalable approach to density and score estimation. In Uncertainty in Artificial Intelligence, pp. 574–584. PMLR, 2020a.   
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020b.   
Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023b.   
Joshua Tian Jin Tee, Kang Zhang, Hee Suk Yoon, Dhananjaya Nagaraja Gowda, Chanwoo Kim, and Chang D Yoo. Physics informed distillation for diffusion models. Transactions on Machine Learning Research, 2024.   
Radu Timofte, Eirikur Agustsson, Luc Van Gool, Ming-Hsuan Yang, and Lei Zhang. Ntire 2017 challenge on single image super-resolution: Methods and results. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pp. 114–125, 2017.   
Chen Wang, Jiatao Gu, Xiaoxiao Long, Yuan Liu, and Lingjie Liu. Geco: Generative image-to-3d within a second. arXiv preprint arXiv:2405.20327, 2024a.   
Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12619–12629, 2023a.   
Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 2555–2563, 2023b.   
Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. International Journal of Computer Vision, pp. 1–21, 2024b.   
Peijuan Wang, Bulent Bayram, and Elif Sertel. A comprehensive review on deep learning based remote sensing image super-resolution methods. Earth-Science Reviews, 232:104110, 2022a.   
Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao, and Chen Change Loy. Esrgan: Enhanced super-resolution generative adversarial networks. In Proceedings of the European conference on computer vision (ECCV) workshops, pp. 0–0, 2018.   
Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind superresolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1905–1914, 2021.   
Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. arXiv preprint arXiv:2212.00490, 2022b.   
Yuang Wang, Siyeop Yoon, Pengfei Jin, Matthew Tivnan, Sifan Song, Zhennong Chen, Rui Hu, Li Zhang, Quanzheng Li, Zhiqiang Chen, et al. Implicit image-to-image schrodinger bridge for image restoration. ¨ Pattern Recognition, 165:111627, 2025.   
Yufei Wang, Wenhan Yang, Xinyuan Chen, Yaohui Wang, Lanqing Guo, Lap-Pui Chau, Ziwei Liu, Yu Qiao, Alex C Kot, and Bihan Wen. Sinsr: diffusion-based image super-resolution in a single step. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 25796–25805, 2024c.   
Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: Highfidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36, 2024d.   
Lemeng Wu, Chengyue Gong, Xingchao Liu, Mao Ye, and Qiang Liu. Diffusion-based molecule generation with informative prior bridges. Advances in Neural Information Processing Systems, 35:36533–36545, 2022.   
Rongyuan Wu, Lingchen Sun, Zhiyuan Ma, and Lei Zhang. One-step effective diffusion network for real-world image super-resolution. arXiv preprint arXiv:2406.08177, 2024.   
Rui Xie, Ying Tai, Chen Zhao, Kai Zhang, Zhenyu Zhang, Jun Zhou, Xiaoqian Ye, Qian Wang, and Jian Yang. Addsr: Accelerating diffusion-based blind super-resolution with adversarial diffusion distillation. arXiv preprint arXiv:2404.01717, 2024.   
Hanshu Yan, Xingchao Liu, Jiachun Pan, Jun Hao Liew, Qiang Liu, and Jiashi Feng. Perflow: Piecewise rectified flow as universal plug-and-play accelerator. arXiv preprint arXiv:2405.07510, 2024.   
Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1191–1200, 2022.   
Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T ¨ Freeman. Improved distribution matching distillation for fast image synthesis. arXiv preprint arXiv:2405.14867, 2024a.   
Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung ¨ Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6613–6623, 2024b.   
Weiyi You, Mingyang Zhang, Leheng Zhang, Xingyu Zhou, Kexuan Shi, and Shuhang Gu. Consistency trajectory matching for one-step generative super-resolution. arXiv preprint arXiv:2503.20349, 2025.   
Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photo-realistic image restoration in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 25669–25680, 2024.   
Conghan Yue, Zhengwei Peng, Junlong Ma, Shiyan Du, Pengxu Wei, and Dongyu Zhang. Image restoration through generalized ornstein-uhlenbeck bridge. arXiv preprint arXiv:2312.10299, 2023.   
Conghan Yue, Zhengwei Peng, Junlong Ma, and Dongyu Zhang. Enhanced control for diffusion bridge in image restoration. arXiv preprint arXiv:2408.16303, 2024a.   
Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Resshift: Efficient diffusion model for image superresolution by residual shifting. Advances in Neural Information Processing Systems, 36, 2024b.   
Lin Zhang, Lei Zhang, and Alan C Bovik. A feature-enriched completely blind image quality evaluator. IEEE Transactions on Image Processing, 24(8):2579–2591, 2015.   
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018.   
Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1219–1229, 2023.

# Appendix for OFTSR

# A USE OF LARGE LANGUAGE MODELS

We used large language models solely for text polishing and grammar correction during manuscript preparation. No LLMs were involved in the conception or design of the method, experiments, or analysis. All technical content, results, and conclusions have been independently verified and validated by the authors.

# B RELEVANT DERIVATIONS TO OUR DISTILLATION LOSS

# B.1 CONTINUOUS VERSION OF THE FINAL LOSS

We provide detailed derivation to our distillation loss used in the paper. By substitute intermediate results $\mathbf { x } _ { s }$ and $\mathbf { x } _ { t }$ from student model Eq. (7) into the ODE induced by teacher model Eq. (8), we have:

$$
\begin{array} { r l } { \pmb { \mathrm { x } } _ { \emptyset } + s \pmb { \mathrm { v } } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , s ) } & { = \pmb { \mathrm { x } } _ { \emptyset } + t \pmb { \mathrm { v } } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t ) + ( s - t ) \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t ) } \\ { \Longrightarrow s ( \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , s ) - \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t ) ) } & { = ( t - s ) \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t ) + ( s - t ) \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t ) } \\ & { = \mathrm { d } t ( \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t ) - \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t ) ) . } \end{array}
$$

Start from this constraint that applies to the student model, we can construct distillation loss in different forms. (i) In the same spirit as BOOT (Gu et al., 2023), we make only ${ \bf v } _ { \phi } ( { \bf x } _ { 0 , \mathrm { L R } } , s )$ and this will lead to loss Eq. (12). (ii) If we only detach the teacher output, we will end up with loss similar to PINN based distillation PID proposed in (Tee et al., 2024):

$$
\begin{array} { r } { \mathrm {  ~ \cdot \mathtt { \mathtt { e } } } _ { \mathtt { P I N N } } ( \phi ) : = \mathbb { E } _ { \mathbf { x } _ { 1 } \sim p _ { 1 } , t \sim \mathcal { U } [ 0 , 1 ] } \left[ \left\| \left[ \frac { s } { \mathrm { d } t } \left( \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , s ) - \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t ) \right) + \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t ) \right] - \mathrm { S G } \left[ \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t ) \right] \right\| _ { 2 } ^ { 2 } \right] . } \end{array}
$$

Both Eqs. (12) and (15) are loss variants from Eq. (8), and we did not try other variant given the already-good performance of Eq. (12).

In addition, by considering the intermediate interpolation ${ \bf x } _ { t } = ( 1 - t ) { \bf x } _ { 0 } + t { \bf x } _ { 1 }$ as a special case of ${ \bf x } _ { t } = \sigma _ { t } { \bf x } _ { 0 } + \alpha _ { t } { \bf x } _ { 1 }$ in BOOT (Gu et al., 2023), we can derive the following distillation loss:

$$
\begin{array} { r } { \mathrm { \tilde { \ r } _ {mathrm { h o o r } } } ( \phi ) : = \mathbb { E } _ { \mathbf { x } _ { 1 } \sim p _ { 1 } , t \sim \mathcal { U } [ 0 , 1 ] } \left[ \frac { 1 } { \lambda ^ { 2 } } \bigg \| \mathbf { x } _ { \phi } \big ( \mathbf { x } _ { 0 , \mathrm { I R } } , s \big ) - S G \left[ \mathbf { x } _ { \phi } \big ( \mathbf { x } _ { 0 , \mathrm { I R } } , t \big ) + \lambda \big ( \mathbf { x } _ { \theta } ( \mathbf { x } _ { t , \mathrm { I R } } , t ) - \mathbf { x } _ { \phi } \big ( \mathbf { x } _ { 0 , \mathrm { I R } } , t \big ) \big ) \right] \bigg \| _ { 2 } ^ { 2 } \right] , } \end{array}
$$

where $\begin{array} { r } { \lambda = 1 - \frac { t ( 1 - s ) } { s ( 1 - t ) } } \end{array}$ , ${ \bf x } _ { \phi } ( { \bf x } _ { 0 , \mathrm { L R } } , t ) = { \bf x } _ { 0 } + { \bf v } _ { \phi } ( { \bf x } _ { 0 , \mathrm { L R } } , t )$ , ${ \bf x } _ { \phi } ( { \bf x } _ { 0 , \mathrm { L R } } , s ) = { \bf x } _ { 0 } + { \bf v } _ { \phi } ( { \bf x } _ { 0 , \mathrm { L R } } , s )$ , and $\mathbf { x } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t ) = \mathbf { x } _ { t } + ( 1 - t ) \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t )$ with $\mathbf { x } _ { t } = \mathbf { x } _ { 0 } + t \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t )$ . We compared our proposed loss Eq. (12) with its variant Eq. (15) and Eq. (16) in Tab. 8 and our ablation shows that Eq. (12) works best for SR task.

# B.2 OFTSR AS FORWARD DISTILLATION

The general form of our OFTSR loss or BOOT (Gu et al., 2023) loss can be seen as a special case of forward distillation (Boffi et al., 2025; Sabour et al., 2025; Liu, 2025). Start from general relation:

$$
\mathbf { x } _ { t } + ( s - t ) \mathbf { v } _ { \phi } ( \mathbf { x } _ { t } , t , s ) = \mathbf { x } _ { s } .
$$

where ${ \bf v } _ { \phi } ( { \bf x } _ { t } , t , s )$ is the mean velocity defined on the time interval $[ t , s ]$ as defined in MeanFlow (Geng et al., 2025).

The MeanFlow loss can be derived directly by taking derivative w.r.t. $t$ of Eq. (17), which is also named as backward distillation loss. Similarly, when taking derivative w.r.t. $s$ , the end timestep of the interval, of Eq. (17), we get the forward distillation loss.

For our OFTSR loss, if we consider mapping from arbitrary start timestep $t$ to two close end timestep $s _ { 1 }$ and $s _ { 2 }$ , and connecting the corresponding two state $\mathbf { x } _ { s _ { 1 } }$ and $\mathbf { x } _ { s _ { 2 } }$ , we have:

$$
\begin{array} { r l } { \mathcal { K } \mathcal { H } + ( s _ { 2 } - t ) \mathbf { v } _ { \phi } ( \mathbf { x } _ { t } , t , s _ { 2 } ) } & { = \mathbf { \mathcal { K } } \mathcal { H } + ( s _ { 1 } - t ) \mathbf { v } _ { \phi } ( \mathbf { x } _ { t } , t , s _ { 1 } ) + ( s _ { 2 } - s _ { 1 } ) \mathbf { v } _ { \theta } ( \mathbf { x } _ { s _ { 1 } } , s _ { 1 } ) } \\ { \Longrightarrow ( s _ { 2 } - t ) \big ( \mathbf { v } _ { \phi } ( \mathbf { x } _ { t } , t , s _ { 2 } ) - \mathbf { v } _ { \phi } ( \mathbf { x } _ { t } , t , s _ { 1 } ) \big ) } & { = \big ( s _ { 2 } - s _ { 1 } \big ) \big ( \mathbf { v } _ { \theta } ( \mathbf { x } _ { s _ { 1 } } , s _ { 1 } ) - \mathbf { v } _ { \phi } ( \mathbf { x } _ { t } , t , s _ { 1 } ) \big ) . } \end{array}
$$

For $s _ { 2 } - s _ { 1 } = \mathrm { d } s$ and $\operatorname* { l i m } _ { \mathrm { d } s \to 0 }$ , we have $s _ { 1 } = s _ { 2 } = s$ and:

$$
( s - t ) \frac { \mathrm { d } } { \mathrm { d } s } { \mathbf { v } } _ { \phi } ( { \mathbf { x } } _ { t } , t , s ) = { \mathbf { v } } _ { \theta } ( { \mathbf { x } } _ { s } , s ) - { \mathbf { v } } _ { \phi } ( { \mathbf { x } } _ { t } , t , s ) ,
$$

which recovers the forward distillation loss as the time derivative w.r.t. $s$ of Eq. (17). Thus we can view the OFTSR loss and BOOT (Gu et al., 2023) loss as a discretization of the forward distillation loss. And it is easy to verify that the signal ODE in BOOT is equivalent to our distillation loss in the flow schedule.

# C LIMITATIONS

While our method advances one-step image super-resolution, limitations include performance constrained by teacher model capabilities. Future work will incorporate ground-truth supervision through regression loss or adversarial training.

# D DIFFUSION AND PERCEPTION-DISTORTION TRADE-OFF

In practice, we found that our distilled model is slightly off the perception-distortion frontier of the teacher model, as displayed in Fig. 7. To be specific, the corresponding timestep $t$ shifts a bit but for the same MMSE value the first-stage model and distilled model have very close LPIPS value. This might be caused by the error from large step size dt used in practice and we leave this for future investigation.

![](images/0caf81efad2b70d9abaea20e2e23b68cc87f19cdd6f9f2e0462edf3fc2e092cb.jpg)  
Figure 6: Metrics evaluation of estimated $\mathbf { x } _ { 1 } ^ { t }$ across different timesteps $t$ . During sampling, at each timestep $t$ , we estimate the final image $\mathbf { x } _ { 1 } ^ { t }$ using the current model prediction $\mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t )$ and state $\mathbf { x } _ { t }$ via $\mathbf { x } _ { 1 } ^ { t } = \mathbf { x } _ { t } + ( \bar { 1 } - t ) \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t )$ . Both MMSE and LPIPS metrics are averaged over 100 sampling processes. We present MMSE instead of PSNR for better visual effect.

![](images/2da8f2a0248dfd7a10b7bff565b1b0bd95edb8493a9b880221f5bf911ffe1c29.jpg)  
Figure 7: Metrics evaluation of estimated $\mathbf { x } _ { 1 } ^ { t }$ across different timesteps $t$ for both teacher model and distilled one-step model. The teacher model is the same as the one in Fig. 6. We present MMSE instead of PSNR for better visual effect.

# E MORE EXPERIMENTAL DETAILS

The training of all networks across both stages is smoothed using Exponential Moving Average (EMA) with a ratio of 0.9999. For FFHQ and ImageNet datasets, images are resized to 256 pixels with center cropping, while DIV2K training employs random crops of $2 5 6 \times 2 5 6$ patches. Data augmentation consists of horizontal flips with $50 \%$ probability and vertical flips with $6 \%$ probability throughout all experiments. For FFHQ noiseless experiment, we use default perturbation std $\sigma _ { p } =$ 0.1; for FFHQ noisy experiment, we use a higher perturbation std $\sigma _ { p } = 0 . 5$ to cover the resized noise from LR images, as suggested in Tab. 11; for both DIV2K and ImageNet we use $\sigma _ { p } = 0 . 2$ . For training, we employed three widely-used datasets: the standard ImageNet training set (1.28M images), the DIV2K training set (800 2K resolution images), and a subset of FFHQ consisting of the first 60,000 images from the dataset. All models are trained until convergence or up to $3 0 0 \mathrm { k }$ training iterations and we select the model based on best metrics. We train the model with uniform loss weight on $t$ . In the distillation stage, we sample the timestep $t$ using $t \sim \mathcal { U } \left[ t _ { \mathrm { m i n } } , t _ { \mathrm { m a x } } \right]$ with $t _ { \mathrm { m i n } } = 0 . 0 1$ and $t _ { \mathrm { m a x } } = 0 . 9 9$ in practice.

For DIV2K evaluation, we first segment the large 2K resolution images into $2 5 6 \times 2 5 6$ patches for model inference, then reconstruct the final image by combining the restored patches. To ensure fair

# Algorithm 1 OFTSR Distillation

Require: teacher flow $\mathbf { v } _ { \theta }$ , dataset $\mathcal { D } _ { \mathrm { H R } }$ , $\sigma _ { n }$ , $\sigma _ { p }$ , dt, $w ( t )$   
1: Initialize the one-step student $\mathbf { v } _ { \phi }$ with the weights of $\mathbf { v } _ { \theta }$   
2: repeat   
3: Randomly sample $\mathbf { x } _ { 1 } \sim \mathcal { D } _ { \mathrm { H R } }$   
4: Randomly sample $\mathbf { n } \sim { \mathcal { N } } ( \mathbf { 0 } , \sigma _ { n } \mathbf { I } )$ ; $\mathbf { n } _ { p } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { p } \mathbf { I } )$   
5: Compute $\mathbf { x } _ { \mathrm { L R } } = \mathcal { H } ^ { T } ( \mathcal { H } ( \mathbf { x } _ { 1 } ) + \mathbf { n } )$ // LR condition   
6: Compute ${ \bf x } _ { 0 } = \sqrt { 1 - \sigma _ { p } ^ { 2 } } { \bf x } _ { \mathrm { L R } } + \sigma _ { p } { \bf n } _ { p }$   
7: Sample $t \in \mathcal { U } [ 0 , 1 ]$ and $s = t + \mathrm { d } t$   
8: Generate velocities ${ \bf v } _ { \phi } ( { \bf x } _ { 0 , \mathrm { L R } } , t )$ and ${ \bf v } _ { \phi } \big ( { \bf x } _ { 0 , \mathrm { L R } } , s \big )$   
9: Calculate $\mathbf { x } _ { t , \mathrm { L R } } = \mathbf { x } _ { 0 } + t \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t )$ and generate velocity $\mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t )$ by teacher model   
10: Compute ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ with Eq. (9) and $\mathcal { L } _ { \mathrm { a l i g n } }$ with Eq. (10)   
11: Generate velocities ${ \mathbf { v } } _ { \phi } \big ( \mathbf { x } _ { 0 , \mathrm { L R } } , 0 \big )$ and ${ \bf v } _ { \theta } ( { \bf x } _ { 0 , \mathrm { L R } } , 0 )$ and compute $\mathcal { L } _ { \mathrm { B C } }$ with Eq. (11)   
12: Compute $\mathcal { L } ( \phi ) = \mathcal { L } _ { \mathrm { d i s t i l l } } ( \phi ) + \lambda _ { \mathrm { a l i g n } } \mathcal { L } _ { \mathrm { a l i g n } } ( \phi ) + \lambda _ { \mathrm { B C } } \mathcal { L } _ { \mathrm { B C } } ( \phi )$   
13: Optimize $\phi$ with a gradient-based optimizer using $\nabla _ { \phi } \mathcal { L }$

Table 11: Ablation on FFHQ $2 5 6 \times 2 5 6$ first stage with noisy SR; default: bs $\mathit { \Theta } = \ 3 2 $ ; $l r = 0 . 0 0 0 1$ ; $\bar { \ell } _ { 1 }$ loss; with LR condition.   

<table><tr><td>σp</td><td>NFEs (↓)</td><td>PSNR (↑)</td><td>LPIPS (↓)</td></tr><tr><td>0.</td><td>20</td><td>25.23</td><td>0.319</td></tr><tr><td>0.1</td><td>20</td><td>24.09</td><td>0.158</td></tr><tr><td>0.3</td><td>32</td><td>24.14</td><td>0.154</td></tr><tr><td>0.5</td><td>32</td><td>24.10</td><td>0.154</td></tr><tr><td>1.</td><td>44</td><td>24.22</td><td>0.153</td></tr></table>

14: until ${ \mathcal { L } } ( \phi )$ converges 15: Return one-step flow $\mathbf { v } _ { \phi }$

![](images/0e9e755351c5f9340aec224bf1f09995f8e0dce697440000026f9c675189af32.jpg)  
Figure 8: Straightness of conditional flows with different perturbation strength $\sigma _ { p }$ .

comparison, all generated SR images are stored in a dedicated separated folder with consistent file names across all evaluated methods, followed by metric calculation against the HR folder using our evaluation script. LPIPS scores are computed using the ‘alex’ model architecture. All experiments are conducted using 4 NVIDIA H800 GPUs.

The straightness of the learned flow v can be calculated with:

$$
S ( \mathbf { v } ) = \int _ { 0 } ^ { 1 } \mathbb { E } \left[ \| \mathbf { v } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| ^ { 2 } \right] \mathrm { d } t ,
$$

We also measured the FID among 50k imagenet validation set and the result FID is 2.458 comparing to 2.8 from I2SB.

# F ADDITIONAL EXPERIMENTS

We evaluated our first-stage training on the FFHQ $2 5 6 \times 2 5 6$ dataset using $\sigma _ { p } = 1$ without conditioning, effectively training an unconditional generative model for human faces. For this experiment, we do not use any data augmentation. Our evaluation consists of generating 1k images from random noise using the RK45 sampler (with a ODE tolerance of 1e-3) and comparing them against the full training dataset of 70k images (we train our unconditional generative flow with the whole dataset). Initial experiments with $\ell _ { 1 }$ loss yielded a FID score of 41.042 with an average of 56 NFEs, which falls short of the previous state-of-the-art P2 model’s score of 28.139 (Choi et al., 2022). However, switching to $\ell _ { 2 }$ loss for standard rectified flow training significantly improved performance, achieving a FID of 24.577 with only 44 NFEs on average. The model architecture used in our experiment is the same as the one used in P2. We leave further investigation to this discrepancy between $\ell _ { 1 }$ and $\ell _ { 2 }$ for image generation and restoration as future works. To facilitate a direct comparison with P2’s best reported results (FID scores of 6.92 and 6.97 with 1,000 and 500 NFEs respectively (Choi et al., 2022)), we generated $5 0 \mathrm { k }$ samples using our $\ell _ { 2 }$ loss-trained model. Our approach achieved a superior FID score of 5.871 with substantially fewer NFEs (44), demonstrating the effectiveness of rectified flow. Representative non-cherry-picking samples from our model are presented in Fig. 12.

![](images/bad73ae6447652dc44496f8f128a8e73f819709b43b5144de92c28342c24527e.jpg)  
Figure 9: Visual results for $8 \times$ (left) and $4 \times$ (right) SR from resolution 64 to 512 and 128 to 512 respectively.

As our distillation technique is designed for image restoration tasks, we skip the distillation of this LR teacher unconditional generation flow.

# G ADDITIONAL RESULTS

# G.1 STRAIGHTNESS VS PERTURBATION STRENGTH

In Fig. 8, we validate the observation in Sec. 4.3 by also measuring the straightness of conditional flows. We observe that for SR task, the straightness is related to the noise perturbation added to the initial distribution, and a straighter flow does not lead to better performance.

# G.2 TRAINING DATASETS

In both stages of our approach, we utilize the same dataset. The following table shows comparable performance across different datasets for distillation with FFHQ teacher.

Table 12: Comparison of distilling FFHQ OFTSR teacher on FFHQ and Celeba-HQ dataset.   

<table><tr><td></td><td>Distillation Dataset</td><td>Hyper-parameter t</td><td>PSNR (↑)</td><td>LPIPS (↓)</td><td>FID (↓)</td></tr><tr><td rowspan="2">FFHQ OFTSR Teacher</td><td>FFHQ</td><td>1</td><td>28.98</td><td>0.055</td><td>36.02</td></tr><tr><td>Celeba-HQ</td><td>1</td><td>29.75</td><td>0.56</td><td>41.25</td></tr></table>

# G.3 DIFFERENT RESOLUTION AND SCALE FACTOR (SF)

In this work, by default we follow previous works to use the setup of $4 \times \mathrm { S R }$ at $2 5 6 \times 2 5 6$ . We also test $\mathrm { S F } = 8$ on $2 5 6 \times 2 5 6$ and $\mathrm { S F } = 4 \& 8$ on 512-resolution FFHQ, the results are shown in Tab. 13. All models are trained for $3 0 \mathrm { k }$ iterations $\mathrm { { b s } = 3 2 }$ ) and distilled for $1 0 \mathrm { k }$ iterations $\mathrm { { b s } = 1 6 }$ ). We visualize $8 \times$ and $4 \times$ reconstruction of teacher and student in Fig. 9.

Table 13: A comparison of the models trained across different resolution and scale factor.   

<table><tr><td>Method</td><td>Target Resolution</td><td>Scale Factor</td><td>NFE (↓)</td><td>PSNR(↑)</td><td>LPIPS (↓)</td><td>FID (↓)</td></tr><tr><td>DDNM (Wang et al., 2022b)</td><td>256</td><td>8</td><td>100</td><td>25.65</td><td>0.178</td><td>104.47</td></tr><tr><td>OFTSR (distilled)</td><td>256</td><td>8</td><td>44 (1)</td><td>25.74 (25.89)</td><td>0.121(0.126)</td><td>72.83 (93.74)</td></tr><tr><td>Unoffcial SR3 (Saharia et al., 2022)</td><td>512</td><td>8</td><td>2000</td><td>21.93</td><td>0.386</td><td>67.31</td></tr><tr><td>OFTSR (distilled)</td><td>512</td><td>8</td><td>32 (1)</td><td>27.31 (28.12)</td><td>0.151 (0.153)</td><td>42.20 (42.33)</td></tr><tr><td>OFTSR (distilled)</td><td>512</td><td>4</td><td>32 (1)</td><td>30.80 (31.30)</td><td>0.073 (0.072)</td><td>13.21 (13.95)</td></tr></table>

# H FAILURE CASE

We show visualization of extreme t (boundary t and out of distribution t) in Fig. 10. Results of t ranges from [0,1] do not show failure case, while (ill-defined) OOD t, especially $t < 0$ fails.

![](images/6709a5fdafe0effc760b394191ccb1d6d0eb52f319bca2636a7fb79012bb5a73.jpg)  
Figure 10: Visual results of boundary $t$ (0 and 1) and out of distribution $t$ (-0.5 and 2

# I RECONSTRUCTION DIVERSITY

The noise-augmented initialization (Sec. 3.1) introduces stochasticity that enables multiple diverse HR reconstructions for the same LR input. Both the teacher and student model can give different restorations of a LR image under different random seeds, and the visualization is shown in Fig. 11.

# J THE CHOICE OF $t$

The parameter $t$ controls the fidelity–realism trade-off and is inherently guided by user preference: $t \approx 0$ favors maximal fidelity, while $t \approx 1$ emphasizes realism. In our experiments, the fidelity–realism parameter $t$ is not highly sensitive to the dataset or degradation type: its effective range stays consistent. When a target domain or evaluation metric is specified, $t$ can also be chosen automatically by optimizing it on a small validation set to best satisfy the desired objective or target. This enables both user-driven and metric-driven control of the fidelity–realism balance.

# K ADDITIONAL VISUAL SAMPLES AND COMPARISONS

In this section, we present additional visual results that demonstrate our method’s capabilities. Fig. 13 showcases multiple examples illustrating the tunable fidelity-realism trade-offs achieved on the FFHQ dataset. Figs. 14 and 15 provide comprehensive comparisons between our method and existing approaches on FFHQ and ImageNet images, respectively. In Fig. 16, we compare real-world (without synthetic degradation) SR results, under the $1 2 8  5 1 2$ SR setting. In Fig. 17, we shows OFTSR can perform arbitrary scale SR. Here, the model is trained solely on ImageNet for $6 4  2 5 6$ SR, demonstrating strong resolution and scale generalization without any retraining. Additionally, in Fig. 18, we demonstrate our method’s performance on both real-world SR tasks and AI-generated content enhancement. In Figs. 19 and 20, we compare visually our OFTSR (DiT4SR) with other SOTA method for one-step large resolution SR. Results from Figs. 14, 15 and 18 are generated with our distilled one-step model unless otherwise specified.

# L DISCUSSION OF ACCELERATED I2SB METHODS

We provide here a detailed discussion of recent accelerated variants of I2SB and clarify their relationship to our approach.

I3SB (Wang et al., 2025). I3SB introduces an improved sampling algorithm for pretrained I2SB models, analogous to DDIM for DDPM. While it yields faster sampling and moderately better results, it retains the fundamental behavior of the original I2SB: multi-step sampling is required to achieve high perceptual realism, whereas a single step primarily preserves fidelity. In contrast, our distillation framework produces a one-step model that attains much stronger realism while also enabling a controllable fidelity–realism trade-off.

CDBM (He et al., 2024). CDBM proposes consistency bridge training and consistency bridge distillation for diffusion bridge models, mirroring the consistency-training paradigm used in consistency models. However, its experimental scope is limited to relatively small image-to-image translation tasks (e.g., Edges Handbags, DIODE-Outdoor) and ImageNet inpainting. Since no SR evaluation or open-source implementation is provided, direct comparison in our setting is not feasible.

IBMD (Gushchin et al., 2025). IBMD introduces a distributional matching algorithm for conditional bridge models, conceptually related to DMD (Yin et al., 2024b) for continuous diffusion models. The method requires learning an additional auxiliary network, which increases computational overhead and can introduce training instability. Moreover, reported results show that the one-step performance of IBMD is comparable to I2SB with 1000 NFEs, suggesting limited advantages for efficient one-step SR. Therefore, our distilled model remains competitive or superior in the one-step regime.

Overall, while these works explore acceleration or distillation within the I2SB family, they differ substantially in objectives, model scope, and applicability to super-resolution. Our approach provides a reproducible and effective one-step SR framework with controllable fidelity–realism behavior not addressed in prior I2SB variants.

![](images/4458f4c8984650ba6ccfb8ab1bb1d1a420c36c2bd9284b57388ee02599bb96bc.jpg)  
Figure 11: Our method can maintain diversity in outputs. The resolution of ground truth image is $5 1 2 \times 5 1 2$ and the LR is $6 4 \times 6 4$ . The first group is generated by the teacher model, while the remaining two groups are produced by the student model.

![](images/4988eb364c3adf9b74b69f9488ff92c74d61ee6d7b825923921f324e66d9391d.jpg)  
Figure 12: Random generated samples from unconditional model trained on FFHQ dataset.

![](images/2196986ba23b5b279b819ad330eb7e691b7f5b7375232c3ff759c6d1b3a73fce.jpg)  
Figure 13: Qualitative results of one-step model with different tunable t.

![](images/e214942810b4609a4a154a44ad066f527ef677a0a48fc4b9ec449948ad6ee719.jpg)  
Figure 14: Qualitative comparisons on FFHQ dataset for $4 \times \mathrm { S R }$ with $\sigma _ { n } = 0$ (first four rows) and $\sigma _ { n } = 0 . 0 5$ (last four rows).

![](images/d5bf57f01176add98b1dac530ca0426293c9e943e0584e6e7bb44d0e41023e59.jpg)  
Figure 15: Qualitative comparisons on ImageNet dataset for noiseless $4 \times$ SR.

![](images/3ba6ea9bc01a4cc7c529fcc3b5e7f75c1bf91d0988faa7c527b286f463eab8eb.jpg)  
Figure 16: Qualitative comparisons for real-world $4 \times$ SR

![](images/86dd56ad285ef491676056cc6606e0dffa84d54955bccfeaeef39b1651c3e16e.jpg)  
Figure 17: Visual result of restoring arbitrary scale LR image.

![](images/106ee6ef1f25bcea4f029920b52a754bc4bba67a14f1013cc3ca827a31073edd.jpg)  
Figure 18: Qualitative results on real data and AI generated content using our $4 \times$ SR model trained on DIV2K.

![](images/2dc1063af13ee87f6e53fddd3918b1d8772284b97e7a4a34c82693513198e7f6.jpg)  
Figure 19: Qualitative comparisons for real-world $4 \times$ SR. OFTSR is distilled from DiT4SR. All methods perform 1 step inference.

![](images/6c19e11e8434ea6786471a9e54c5ea3bf48693120706d278264b3f557c7e7b72.jpg)  
Figure 20: Qualitative comparisons for real-world $4 \times$ SR. OFTSR is distilled from DiT4SR. All methods perform 1 step inference.