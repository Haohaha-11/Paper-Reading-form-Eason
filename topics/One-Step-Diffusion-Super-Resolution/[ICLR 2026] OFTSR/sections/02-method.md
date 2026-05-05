[← 返回 README](../README.md)

# Method

## 📌 预览
本文件合并 Method/Background，重点看 LQ latent、teacher/student、flow/ODE、控制变量和训练损失。

---

# 2 BACKGROUND

# 2.1 DIFFUSION AND FLOW-BASED GENERATIVE MODELS

Drawing inspiration from non-equilibrium thermodynamics, diffusion models operate through two core processes: a forward diffusion process that gradually adds Gaussian noise to data until it becomes pure noise, and a reverse denoising process that systematically reconstructs the original data by removing noise (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020b). Let $\mathbf { x } _ { t }$ represent the data $\mathbf { x }$ at timestep $t$ . The forward process can be formally described by the Ito Stochastic ˆ Differential Equation (SDE) (Song et al., 2020b):

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

![Equation 1](../images/8cc615f66904ec92786bbf632c952d5b4959baaf50c674ef2068dad1a1c33109.jpg)
*Equation 1: Equation extracted by MinerU.*

> 💡 **Equation 1 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where w is the standard Wiener process, $f _ { t } : \mathbb { R } \to \mathbb { R }$ is the drift coefficient, and $g _ { t } : \mathbb { R }  \mathbb { R }$ is a scalar function called the diffusion coefficient.

For every diffusion process described by Eq. (1), there exists a corresponding deterministic Probability Flow Ordinary Differential Equation (PF-ODE) that maintains the same marginal probability density:

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐。

![Equation 2](../images/60fa481b88d4805d98ca4d2b6a4ba914fa45d9293861e10d2b12ce39a741b098.jpg)
*Equation 2: Equation extracted by MinerU.*

> 💡 **Equation 2 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where $p _ { t } ( \cdot )$ represents the marginal probability density at time $t$ . The term $\nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } )$ is known as the score function, which can be approximated by a neural network $\mathbf { s } _ { \theta } ( \mathbf { x } , t )$ with parameters $\theta$ . This network is typically trained using score matching techniques (Hyvarinen¨ $\&$ Dayan, 2005; Song & Ermon, 2019; Song et al., 2020a).

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

To generate data samples, the process begins with Gaussian noise drawn from an initial Gaussian distribution $p _ { 0 }$ and solves Eq. (2) numerically from $t = 0$ to $t = 1$ . By utilizing the learned score function $\mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t )$ , the empirical PF-ODE can be obtained as: $\begin{array} { r } { \frac { \mathrm { d } \mathbf { x } _ { t } } { \mathrm { d } t } = f _ { t } \mathbf { x } _ { t } - \frac { 1 } { 2 } g _ { t } ^ { 2 } \mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t ) } \end{array}$ .

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐。

Rectified flow (Liu et al., 2022; Liu, 2022; Lipman et al., 2022; Esser et al., 2024) is a generative modeling framework based on ODEs. Given an initial distribution $p _ { 0 }$ and a target data distribution $p _ { 1 }$ , rectified flow trains a neural network to parameterize a velocity field using the following loss function:

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

![Equation 3](../images/c383920facd54057f8fe1a96915ed35c6dbbd24dd83d49afd845121872b0a12e.jpg)
*Equation 3: Equation extracted by MinerU.*

> 💡 **Equation 3 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

raito mple generation is achieved by solving the empirical ODE . In practical implementations, this empirical ODE is sol $\begin{array} { r } { \frac { \mathrm { d } \mathbf { x } _ { t } } { \mathrm { d } t } = \mathbf { v } _ { \theta } ( \mathbf { x } _ { t } , t ) } \end{array}$ from using $t ~ = ~ 0$ $t \ = \ 1$ standard ODE solvers, ranging from the simple forward Euler method to higher-order methods such as RK2 and RK45.

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐。

# 2.2 PERCEPTION-DISTORTION TRADE-OFF

The perception-distortion (realism-fidelity) trade-off (Blau & Michaeli, 2018) is a fundamental concept in image restoration. It describes the inherent trade-off between perceptual realism and fidelity to the ground truth, and mathematically proves that it is generally not possible to achieve both good perceptual realism and high fidelity simultaneously.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近结构，又要生成自然高频细节。

To address this challenge, researchers have explored various approaches to enable tunable trade-offs between these two desirable qualities. One common technique involves interpolating between the weights of two models with the same architecture, trained with GAN loss and mean squared error loss (Wang et al., 2018). Recently, diffusion models have emerged as a promising approach for this task. The iterative sampling nature of diffusion models provides a flexible means of controlling the desired trade-offs. By adjusting the Number of Function Evaluations (NFEs), users can generate reconstructions that better match their specific requirements (Chung et al., 2024). Specifically, lower NFEs tend to result in reconstructions with reduced distortion, as the output regresses towards the mean (Delbracio & Milanfar, 2023). Conversely, higher NFEs prioritize perceptual quality, even if it comes at the expense of some distortion from the ground truth (similar to Fig. 6).

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近结构，又要生成自然高频细节。

# 3 METHOD

In this section, we introduce the OFTSR framework for one-step SR that can restore HR images with either high realism or high fidelity. We achieve this goal through a two-stage process: first, we train a direct flow-based model for SR, and then we distill this model into a simplified one-step variant. In Sec. 3.1, we present a simple noise-augmented conditional flow that expands the support of the initial distribution, enabling diverse reconstruction. In Sec. 3.2, we propose to distill the student model by restricting its predictions on the same ODE using teacher model from Sec. 3.1.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Figure 2](../images/493ab02c54e87842102afc0d93ce9671715f26b9c2c435afa657093170189e40.jpg)
*Figure 2: Figure 2: Illustration of the proposed distillation loss. Rather than directly distilling from the teacher, we leverage the teacher to align the one-step intermediate outputs, $\mathbf { x } _ { t }$ and $\mathbf { x } _ { s }$ , along teacher’s PF-ODE trajectory. For simplicity, LR conditioning is omitted in this figure.*

> 💡 **Figure 2 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

# 3.1 NOISE AUGMENTED CONDITIONAL FLOW

Unlike diffusion models, flow-based models have the advantage that their initial distribution is not limited to Gaussian distributions. This flexibility suggests a natural approach for image restoration - directly learning a flow that maps the distribution of LR images $( p _ { \mathrm { L R } } )$ to that of HR images $( p _ { \mathrm { H R } } )$ . However, our initial experiments (see Tab. 7) showed poor performance with this direct approach, aligning with findings from several recent works (Delbracio & Milanfar, 2023; Kim et al., 2024; Lee et al., 2024). This training procedure tends to collapse the LR $\mathrm {  H R }$ mapping: during inference each LR is driven toward a single HR.

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐。

To overcome this limitation, we propose a noise-augmented approach to process LR images. For any input image $\mathbf { x } _ { \mathrm { L R } }$ , we construct our initial distribution $p _ { 0 } ( \mathbf { \dot { x } } ) = p _ { \mathrm { L R } } ^ { \sigma _ { p } }$ by adding Gaussian noise with standard deviation $\sigma _ { p }$ . Specifically, we adopt a Variance-Preserving (VP) noising operation (Ho et al., 2020; Song et al., 2020b):

![Equation 4](../images/9a631ee3ab8706dc79d0249c7ac0a0fd86d1e21b99f5c0553445df9d0eda0adc.jpg)
*Equation 4: Equation extracted by MinerU.*

> 💡 **Equation 4 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where $\epsilon$ is a standard Gaussian noise. While this noise perturbation facilitates better generalization, it inevitably causes information loss in the LR image. To address this, we incorporate $\mathbf { x } _ { \mathrm { L R } }$ as a conditional input to our model as in Fig. 1. This VP formulation, together with the condition $\mathbf { x } _ { \mathrm { L R } }$ , makes our method particularly versatile, encompassing previous approaches as special cases. When $\sigma _ { p } = 0$ , our method reduces to the minimal augmentation case in InDI (Delbracio & Milanfar, 2023), and when $\sigma _ { p } = 1$ , it matches the training strategy of SR3 (Saharia et al., 2022).

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

Given this noise-augmented formulation, we can now define our training objective as:

![Equation 5](../images/1d80d040be3fee51b901ddf473e42c985e7be05d634f894c585274e2ea4308fd.jpg)
*Equation 5: Equation extracted by MinerU.*

> 💡 **Equation 5 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where $\mathbb { D }$ is a discrepancy loss that measures the difference between two images (e.g., $\ell _ { 2 }$ loss or the $\ell _ { 1 }$ loss), $\mathbf { v } _ { \theta }$ is our velocity model, $\mathbf { x } _ { t , \mathrm { L R } } = \mathrm { c o n c a t } ( \mathbf { x } _ { t } , \mathbf { x } _ { \mathrm { L R } } )$ is the concatenation of $\mathbf { x } _ { t }$ and $\mathbf { x } _ { \mathrm { L R } }$ in channel dimension (see Fig. 1), The LR input of the algorithm is given by $\mathbf { x } _ { \mathrm { L R } } = \mathcal { H } ^ { T } ( \mathcal { H } ( \mathbf { x } _ { 1 } ) + \mathbf { n } )$ , where $\mathcal { H }$ is the downsampling operator, $\bar { \mathcal { H } } ^ { T }$ is its transpose and $\mathbf { n }$ is i.i.d. Gaussian noise with variance $\sigma _ { n } ^ { 2 }$ . The perturbed version of $\mathbf { x } _ { \mathrm { L R } }$ , denoted as $\mathbf { x } _ { \mathrm { 0 } }$ , is obtained using the noise augmentation strategy described in Eq. (4). Additionally, ${ \bf x } _ { t } = ( 1 - t ) { \bf x } _ { 0 } + t { \bf x } _ { 1 }$ denotes the intermediate state as in rectified flow (Liu et al., 2022; Liu, 2022).

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

# 3.2 DISTILLATION LOSS

We introduce a distillation loss to train a one-step student that preserves the pre-trained SR flow’s fidelity–realism trade-off, allowing control at inference via a single hyperparameter $t$ . As shown in Fig. 6 and observed in prior work (Delbracio & Milanfar, 2023; Liu et al., 2023a), single-step estimates of the final state $\mathbf { x } _ { 1 } ^ { t }$ obtained from an intermediate state $\mathbf { x } _ { t }$ lie on a fidelity–realism curve: along the ODE sampling trajectory, estimates for larger $t$ (closer to 1) exhibit richer detail and lower LPIPS (better realism), whereas estimates for smaller $t$ (closer to 0) are blurrier but achieve lower MMSE and higher PSNR (better fidelity).

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

To preserve the fidelity-realism trade-off, given the same input ${ \bf x } _ { \mathrm { 0 , L R } }$ , for two different timesteps $t$ and $s$ where $s > t$ , we require the student model $\mathbf { v } _ { \phi }$ to produce two corresponding intermediate states $\mathbf { x } _ { t }$ and $\mathbf { x } _ { s }$ that lie on the same ODE trajectory defined by the teacher (see Fig. 2):

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近结构，又要生成自然高频细节。

![Equation 6](../images/cfa3eeec90933682981a3ad23135ecc0fa28839a2585381534f002ae25676742.jpg)
*Equation 6: Equation extracted by MinerU.*

> 💡 **Equation 6 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where ${ \bf x } _ { 0 , \mathrm { L R } } = \mathrm { c o n c a t } ( { \bf x } _ { 0 } , { \bf x } _ { \mathrm { L R } } )$ is the concatenation of the input image $\mathbf { x } _ { \mathrm { 0 } }$ and the LR condition $\mathbf { x } _ { \mathrm { L R } }$ along the channel dimension. The intermediate states $\mathbf { x } _ { t }$ and $\mathbf { x } _ { s }$ can be computed using our one-step student model $\mathbf { v } _ { \phi }$ :

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Equation 7](../images/e696ac236efd8469d1ba12bba72e45c1aa1f83dcdd9482745836df5f2cd2958f.jpg)
*Equation 7: Equation extracted by MinerU.*

> 💡 **Equation 7 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

Substituting the expression for the intermediate image $\mathbf { x } _ { t }$ and $\mathbf { x } _ { s }$ from Eq. (7) into Eq. (6), we have the following constraint on the student model:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 8](../images/2ac502738e7ffda622ea11a93489897344e46ad30dd69b94631353118e369047.jpg)
*Equation 8: Equation extracted by MinerU.*

> 💡 **Equation 8 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

Similar to BOOT, we can set $\mathrm { d } t = s - t$ and derive the final distillation loss:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 9](../images/69827443ffa3cdfe98d589a3ebc2d885978971a00f286cf6887f4af354951184.jpg)
*Equation 9: Equation extracted by MinerU.*

> 💡 **Equation 9 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where $\operatorname { S G } [ \cdot ]$ is the stop-gradient operator for training stability (Gu et al., 2023; Tee et al., 2024). Since $s - t = \mathrm { d } t$ and $t > 0$ , we do not have the ‘dividing by $0 ^ { \circ }$ issue in (Tee et al., 2024). Similarly to (Song et al., 2023b; Gu et al., 2023), we can use the Euler or general RK2 solver to calculate $\mathbf { v } _ { \theta }$ in Eq. (9). In our main experiments, we employ the midpoint method, while also evaluating two other RK2 solver variants, i.e., Heun’s method and Ralston’s method, for comparison in our ablations (see Tab. 8). In Sec. B.2, we show that our distillation loss is the discrete-time counterpart of the forward distillation loss (Boffi et al., 2025; Liu, 2025) by fixing the start timestep at 0, which is highly related to recent work MeanFlow (Geng et al., 2025) and AlignYourFlow (Sabour et al., 2025).

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

# 3.3 ALIGNMENT AND BOUNDARY LOSS

In BOOT (Gu et al., 2023), a boundary condition is applied to enforce that the one-step student model and teacher model perform the same at the boundary $t = 0$ . We aim to align the teacher and student outputs in our model. The student produces $\mathbf { x } _ { 0 } + \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t )$ , while the teacher generates $\mathbf { x } _ { t } + ( 1 - t ) \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t )$ based on the student’s output $\mathbf { x } _ { t }$ using Eq. (7). By minimizing the difference between these outputs, we get the following alignment loss to align the teacher and student:

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Equation 10](../images/67254537d6d930e07503f563d5c5f58683ce09a7ea43d9ecf6744f4ba64a31f7.jpg)
*Equation 10: Equation extracted by MinerU.*

> 💡 **Equation 10 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

If we consider this alignment loss only at $t = 0$ , it becomes equivalent to the boundary loss used in BOOT:

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐。

![Equation 11](../images/8de7542077b80a2ff76ad7d4030ddc331e1211afb8d8bcb36c6f23d097e64c5e.jpg)
*Equation 11: Equation extracted by MinerU.*

> 💡 **Equation 11 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

Since it is difficult to sample $t = 0$ for most training iterations, we add in addition the boundary loss Eq. (11) in our final training objective.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

The overall training objective. The student network $\mathbf { v } _ { \phi }$ is trained to minimize the combination of the aforementioned three losses terms:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 12](../images/b9d1afd400a798a2fb71e05dde2d2a886ee098aa5e747321b113d62809ada6c8.jpg)
*Equation 12: Equation extracted by MinerU.*

> 💡 **Equation 12 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where $\lambda _ { \mathrm { a l i g n } }$ and $\lambda _ { \mathrm { B C } }$ are the weights for alignment loss and boundary condition loss, respectively.   
The distillation stage of the proposed method is summarized in Algorithm 1.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Figure 0](../images/285f0f974f2f1be9725113f863df28aa28e65f7c1adde48123606b90bb704cf2.jpg)
*Figure 0: MinerU 原始图片*

> 💡 **Figure 0 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

![Figure 4](../images/3c4fdaa803f4f85bca8804b9cea92175c7cd5129e56fba37451107f07a07da22.jpg)
*Figure 4: GT LR DPS DDRM DDNM DiffPir SITCOM OursFigure 4: Qualitative comparison with training-free methods. The first row shows noiseless SR on the FFHQ dataset, the second row presents noisy SR $\sigma _ { n } = 0 . 0 5 )$ on FFHQ, and the bottom row demonstrates noiseless SR on the ImageNet dataset. Numbers next to the method names represent the required NFEs.*

> 💡 **Figure 4 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

GT LR DPS DDRM DDNM DiffPir SITCOM OursFigure 4: Qualitative comparison with training-free methods. The first row shows noiseless SR on the FFHQ dataset, the second row presents noisy SR $\sigma _ { n } = 0 . 0 5 )$ on FFHQ, and the bottom row demonstrates noiseless SR on the ImageNet dataset. Numbers next to the method names represent the required NFEs.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标。

Inference. After training, the one-step student $\mathbf { v } _ { \phi }$ produces the final high-resolution output $\mathbf { x } _ { 1 } ^ { t }$ in a single forward pass, conditioned on the initial state $\mathbf { x } _ { \mathrm { 0 } }$ , the low-resolution input $\mathbf { x } _ { \mathrm { L R } }$ , and the trade-off parameter $t$ . Concretely,

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Equation 13](../images/5c7d882c6fa52d69a0b4c1fbd4ed887f3226b36c24fd7b2165f99e10bb12ba18.jpg)
*Equation 13: Equation extracted by MinerU.*

> 💡 **Equation 13 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where $\mathbf { v } _ { \phi } ( \cdot )$ predicts a residual that refines $\mathbf { x } _ { \mathrm { 0 } }$ toward the desired point on the fidelity–realism curve specified by $t$ .

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近结构，又要生成自然高频细节。

# 4.4 COMPUTATIONAL OVERHEAD

Training Cost Comparison. Our distillation algorithm is highly flexible and can be easily applied to any pre-trained diffusion/flow-based conditional model. As shown in Tab. 9, we applied our distillation algorithm to the ResShift(Yue et al., 2024b) pre-trained model and achieved teacherlevel performance in one step, surpassing SinSR(Wang et al., 2024c) in FID with much less training compute. Even taking the training stage into account with a larger model, our method remains more efficient than ResShift. We use $t = 1$ for OFTSR.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

Table 9: Comparison of training cost on single NVIDIA A100.

![Table extracted](../images/cbdfc8779ab1b331d89ffbfb34cc5277ce189448573d0f9ed39bbb881f865656.jpg)
*Table extracted: Table extracted by MinerU. DIV2K Method NFEs (↓) PSNR (↑) LPIPS (↓) FID (↓) Training- free DPS (Chung et al., 2022) 1000 23.05 0.447 109.35 DDRM (Kawar et al., 2022) 20 27.87 0.285 23.38 DDNM (Wang et al., 2*

> 💡 **Table extracted 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

Inference Cost Comparison. We have included a detailed comparison of the inference cost in Tab. 10, using FLOPS and MAC to measure model complexity. We use $t = 1$ for OFTSR.

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

Table 10: Comparison of inference cost on single NVIDIA A100.

![Table 1](../images/a12df72f12d83fb80b6dec976e4c6e326796677e345a8f2d2c11c00c62af6cd6.jpg)
*Table 1: Table 1: Noiseless quantitative results on DIV2K. We compute the average PSNR (dB), LPIPS and FID of different methods on $4 \times$ SR. The best and second best results are highlighted in bold and underline. The distilled model produces superior performance in terms of trade-off metrics through adjustment of the hyperparameter $t$ . Table 2: Noiseless (top) and noisy (bottom) quantitative results on FFHQ $\pmb { 2 5 6 } \times 2 5 6$ . We compute the average PSNR (dB), LPIPS and FID of different methods on $4 \times$ SR. The best and second best results are highlighted in bold and underline.*

> 💡 **Table 1 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

# A USE OF LARGE LANGUAGE MODELS

We used large language models solely for text polishing and grammar correction during manuscript preparation. No LLMs were involved in the conception or design of the method, experiments, or analysis. All technical content, results, and conclusions have been independently verified and validated by the authors.

> 💡 **批注**: 这里涉及条件信号：prompt 是否准确、是否退化感知，会影响生成细节与语义一致性。

# B RELEVANT DERIVATIONS TO OUR DISTILLATION LOSS

# B.1 CONTINUOUS VERSION OF THE FINAL LOSS

We provide detailed derivation to our distillation loss used in the paper. By substitute intermediate results $\mathbf { x } _ { s }$ and $\mathbf { x } _ { t }$ from student model Eq. (7) into the ODE induced by teacher model Eq. (8), we have:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 14](../images/590b25963f9fa12d970df65753e4bd920aec64b8b4d489ae14eebe6ee6675380.jpg)
*Equation 14: Equation extracted by MinerU.*

> 💡 **Equation 14 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

Start from this constraint that applies to the student model, we can construct distillation loss in different forms. (i) In the same spirit as BOOT (Gu et al., 2023), we make only ${ \bf v } _ { \phi } ( { \bf x } _ { 0 , \mathrm { L R } } , s )$ and this will lead to loss Eq. (12). (ii) If we only detach the teacher output, we will end up with loss similar to PINN based distillation PID proposed in (Tee et al., 2024):

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 15](../images/3109856f2ad3613248f5fcb3d4aaa7ee1a18057ee52e200f38075631c6996689.jpg)
*Equation 15: Equation extracted by MinerU.*

> 💡 **Equation 15 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

Both Eqs. (12) and (15) are loss variants from Eq. (8), and we did not try other variant given the already-good performance of Eq. (12).

In addition, by considering the intermediate interpolation ${ \bf x } _ { t } = ( 1 - t ) { \bf x } _ { 0 } + t { \bf x } _ { 1 }$ as a special case of ${ \bf x } _ { t } = \sigma _ { t } { \bf x } _ { 0 } + \alpha _ { t } { \bf x } _ { 1 }$ in BOOT (Gu et al., 2023), we can derive the following distillation loss:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 16](../images/74f422e7142accde1496f4f1da145b919a3f08065eb418cc9f525dfd6de367b8.jpg)
*Equation 16: Equation extracted by MinerU.*

> 💡 **Equation 16 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where $\begin{array} { r } { \lambda = 1 - \frac { t ( 1 - s ) } { s ( 1 - t ) } } \end{array}$ , ${ \bf x } _ { \phi } ( { \bf x } _ { 0 , \mathrm { L R } } , t ) = { \bf x } _ { 0 } + { \bf v } _ { \phi } ( { \bf x } _ { 0 , \mathrm { L R } } , t )$ , ${ \bf x } _ { \phi } ( { \bf x } _ { 0 , \mathrm { L R } } , s ) = { \bf x } _ { 0 } + { \bf v } _ { \phi } ( { \bf x } _ { 0 , \mathrm { L R } } , s )$ , and $\mathbf { x } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t ) = \mathbf { x } _ { t } + ( 1 - t ) \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t )$ with $\mathbf { x } _ { t } = \mathbf { x } _ { 0 } + t \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t )$ . We compared our proposed loss Eq. (12) with its variant Eq. (15) and Eq. (16) in Tab. 8 and our ablation shows that Eq. (12) works best for SR task.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标。

# B.2 OFTSR AS FORWARD DISTILLATION

The general form of our OFTSR loss or BOOT (Gu et al., 2023) loss can be seen as a special case of forward distillation (Boffi et al., 2025; Sabour et al., 2025; Liu, 2025). Start from general relation:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 17](../images/53cd7381cc4b3c30143cb286f47d3fd609822cb58cbe7b5d645dc30eb7fa9787.jpg)
*Equation 17: Equation extracted by MinerU.*

> 💡 **Equation 17 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where ${ \bf v } _ { \phi } ( { \bf x } _ { t } , t , s )$ is the mean velocity defined on the time interval $[ t , s ]$ as defined in MeanFlow (Geng et al., 2025).

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐。

The MeanFlow loss can be derived directly by taking derivative w.r.t. $t$ of Eq. (17), which is also named as backward distillation loss. Similarly, when taking derivative w.r.t. $s$ , the end timestep of the interval, of Eq. (17), we get the forward distillation loss.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

For our OFTSR loss, if we consider mapping from arbitrary start timestep $t$ to two close end timestep $s _ { 1 }$ and $s _ { 2 }$ , and connecting the corresponding two state $\mathbf { x } _ { s _ { 1 } }$ and $\mathbf { x } _ { s _ { 2 } }$ , we have:

> 💡 **批注**: 这是控制机制：作者试图把退化强度、区域差异或 timestep 映射成可操作的生成强度。

![Equation 18](../images/47040a540d74b189023a10c98bb077c874ba295ade0b134bd038c0e2ecf7b9e5.jpg)
*Equation 18: Equation extracted by MinerU.*

> 💡 **Equation 18 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

For $s _ { 2 } - s _ { 1 } = \mathrm { d } s$ and $\operatorname* { l i m } _ { \mathrm { d } s \to 0 }$ , we have $s _ { 1 } = s _ { 2 } = s$ and:

![Equation 19](../images/7308375dd1020a73e58d785b75e513d0d6126b3ee1056cfc22e92571df3c9f9d.jpg)
*Equation 19: Equation extracted by MinerU.*

> 💡 **Equation 19 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

which recovers the forward distillation loss as the time derivative w.r.t. $s$ of Eq. (17). Thus we can view the OFTSR loss and BOOT (Gu et al., 2023) loss as a discretization of the forward distillation loss. And it is easy to verify that the signal ODE in BOOT is equivalent to our distillation loss in the flow schedule.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

# D DIFFUSION AND PERCEPTION-DISTORTION TRADE-OFF

In practice, we found that our distilled model is slightly off the perception-distortion frontier of the teacher model, as displayed in Fig. 7. To be specific, the corresponding timestep $t$ shifts a bit but for the same MMSE value the first-stage model and distilled model have very close LPIPS value. This might be caused by the error from large step size dt used in practice and we leave this for future investigation.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近结构，又要生成自然高频细节。

![Figure 6](../images/0caf81efad2b70d9abaea20e2e23b68cc87f19cdd6f9f2e0462edf3fc2e092cb.jpg)
*Figure 6: Figure 6: Metrics evaluation of estimated $\mathbf { x } _ { 1 } ^ { t }$ across different timesteps $t$ . During sampling, at each timestep $t$ , we estimate the final image $\mathbf { x } _ { 1 } ^ { t }$ using the current model prediction $\mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t )$ and state $\mathbf { x } _ { t }$ via $\mathbf { x } _ { 1 } ^ { t } = \mathbf { x } _ { t } + ( \bar { 1 } - t ) \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t )$ . Both MMSE and LPIPS metrics are averaged over 100 sampling processes. We present MMSE instead of PSNR for better visual effect.*

> 💡 **Figure 6 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

![Figure 7](../images/2da8f2a0248dfd7a10b7bff565b1b0bd95edb8493a9b880221f5bf911ffe1c29.jpg)
*Figure 7: Figure 7: Metrics evaluation of estimated $\mathbf { x } _ { 1 } ^ { t }$ across different timesteps $t$ for both teacher model and distilled one-step model. The teacher model is the same as the one in Fig. 6. We present MMSE instead of PSNR for better visual effect.*

> 💡 **Figure 7 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

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

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

Table 11: Ablation on FFHQ $2 5 6 \times 2 5 6$ first stage with noisy SR; default: bs $\mathit { \Theta } = \ 3 2 $ ; $l r = 0 . 0 0 0 1$ ; $\bar { \ell } _ { 1 }$ loss; with LR condition.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

![Table 3](../images/5007da874231420f4acdcf905607c2eebd9245d5e9970520314f886806244e08.jpg)
*Table 3: Table 3: Noiseless quantitative results on ImageNet $2 5 6 \times 2 5 6$ . We compute the average PSNR (dB), LPIPS and FID of different methods on $4 \times \mathrm { S R }$ . The best and second best results are highlighted in bold and underline.*

> 💡 **Table 3 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

14: until ${ \mathcal { L } } ( \phi )$ converges 15: Return one-step flow $\mathbf { v } _ { \phi }$

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Figure 8](../images/0e9e755351c5f9340aec224bf1f09995f8e0dce697440000026f9c675189af32.jpg)
*Figure 8: Figure 8: Straightness of conditional flows with different perturbation strength $\sigma _ { p }$ .*

> 💡 **Figure 8 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

comparison, all generated SR images are stored in a dedicated separated folder with consistent file names across all evaluated methods, followed by metric calculation against the HR folder using our evaluation script. LPIPS scores are computed using the ‘alex’ model architecture. All experiments are conducted using 4 NVIDIA H800 GPUs.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标。

The straightness of the learned flow v can be calculated with:

![Equation 20](../images/9cb9488178d9c93b8236e5da811bb01344a50fb9c2589b483ea2821681a524cd.jpg)
*Equation 20: Equation extracted by MinerU.*

> 💡 **Equation 20 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

We also measured the FID among 50k imagenet validation set and the result FID is 2.458 comparing to 2.8 from I2SB.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标。

# G.1 STRAIGHTNESS VS PERTURBATION STRENGTH

In Fig. 8, we validate the observation in Sec. 4.3 by also measuring the straightness of conditional flows. We observe that for SR task, the straightness is related to the noise perturbation added to the initial distribution, and a straighter flow does not lead to better performance.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

# G.2 TRAINING DATASETS

In both stages of our approach, we utilize the same dataset. The following table shows comparable performance across different datasets for distillation with FFHQ teacher.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

Table 12: Comparison of distilling FFHQ OFTSR teacher on FFHQ and Celeba-HQ dataset.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Table extracted](../images/eedfca804729351cbb5b3cd83aa19dfde443fb5a456bb2d8eb26d9a36bc19a2d.jpg)
*Table extracted: Table extracted by MinerU. ImageNet PSNR (↑) LPIPS (↓) FID (↓) NIQE (↓) MUSIQ (↑) MANIQA (↑) CLIPIQA (↑) SwinIR (Liang et al., 2021) 22.24 0.320 207.75 6.0084 47.46 0.5527 0.5544 NAFNet (Chen et al., 2022) 2*

> 💡 **Table extracted 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

# G.3 DIFFERENT RESOLUTION AND SCALE FACTOR (SF)

In this work, by default we follow previous works to use the setup of $4 \times \mathrm { S R }$ at $2 5 6 \times 2 5 6$ . We also test $\mathrm { S F } = 8$ on $2 5 6 \times 2 5 6$ and $\mathrm { S F } = 4 \& 8$ on 512-resolution FFHQ, the results are shown in Tab. 13. All models are trained for $3 0 \mathrm { k }$ iterations $\mathrm { { b s } = 3 2 }$ ) and distilled for $1 0 \mathrm { k }$ iterations $\mathrm { { b s } = 1 6 }$ ). We visualize $8 \times$ and $4 \times$ reconstruction of teacher and student in Fig. 9.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

Table 13: A comparison of the models trained across different resolution and scale factor.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标。

![Table 5](../images/d3dcf78bde94293d8677498a06f09b4137e7999f0a5c85f33fc161bf8e4cc215.jpg)
*Table 5: Table 5: Quantitative comparison on real world sets. The best and second best results are in bold and underline.*

> 💡 **Table 5 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

# I RECONSTRUCTION DIVERSITY

The noise-augmented initialization (Sec. 3.1) introduces stochasticity that enables multiple diverse HR reconstructions for the same LR input. Both the teacher and student model can give different restorations of a LR image under different random seeds, and the visualization is shown in Fig. 11.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

# J THE CHOICE OF $t$

The parameter $t$ controls the fidelity–realism trade-off and is inherently guided by user preference: $t \approx 0$ favors maximal fidelity, while $t \approx 1$ emphasizes realism. In our experiments, the fidelity–realism parameter $t$ is not highly sensitive to the dataset or degradation type: its effective range stays consistent. When a target domain or evaluation metric is specified, $t$ can also be chosen automatically by optimizing it on a small validation set to best satisfy the desired objective or target. This enables both user-driven and metric-driven control of the fidelity–realism balance.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近结构，又要生成自然高频细节。

---

## 🔖 Section 总结

### 核心洞察
1. 明确单步路径、teacher/student 或控制变量。
2. 区分训练时机制和推理时机制。
3. 关注 loss 与模块分别解决哪个瓶颈。
