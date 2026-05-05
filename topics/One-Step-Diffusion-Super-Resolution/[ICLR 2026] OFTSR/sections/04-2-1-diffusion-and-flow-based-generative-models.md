[← 返回 README](../README.md)

# 2.1 DIFFUSION AND FLOW-BASED GENERATIVE MODELS

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 OFTSR 主线的关系**: OFTSR 用 conditional flow teacher 和 ODE-trajectory alignment distillation 构建 one-step SR，并保留可调 fidelity-realism trade-off。

---

Drawing inspiration from non-equilibrium thermodynamics, diffusion models operate through two core processes: a forward diffusion process that gradually adds Gaussian noise to data until it becomes pure noise, and a reverse denoising process that systematically reconstructs the original data by removing noise (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020b). Let $\mathbf { x } _ { t }$ represent the data $\mathbf { x }$ at timestep $t$ . The forward process can be formally described by the Ito Stochastic ˆ Differential Equation (SDE) (Song et al., 2020b):

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

![Equation 1](../images/8cc615f66904ec92786bbf632c952d5b4959baaf50c674ef2068dad1a1c33109.jpg)
*Equation 1: Equation extracted by MinerU.*

> 💡 **Equation 1 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where w is the standard Wiener process, $f _ { t } : \mathbb { R } \to \mathbb { R }$ is the drift coefficient, and $g _ { t } : \mathbb { R }  \mathbb { R }$ is a scalar function called the diffusion coefficient.

For every diffusion process described by Eq. (1), there exists a corresponding deterministic Probability Flow Ordinary Differential Equation (PF-ODE) that maintains the same marginal probability density:

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐，而不只是匹配终点。

![Equation 2](../images/60fa481b88d4805d98ca4d2b6a4ba914fa45d9293861e10d2b12ce39a741b098.jpg)
*Equation 2: Equation extracted by MinerU.*

> 💡 **Equation 2 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where $p _ { t } ( \cdot )$ represents the marginal probability density at time $t$ . The term $\nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } )$ is known as the score function, which can be approximated by a neural network $\mathbf { s } _ { \theta } ( \mathbf { x } , t )$ with parameters $\theta$ . This network is typically trained using score matching techniques (Hyvarinen¨ $\&$ Dayan, 2005; Song & Ermon, 2019; Song et al., 2020a).

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

To generate data samples, the process begins with Gaussian noise drawn from an initial Gaussian distribution $p _ { 0 }$ and solves Eq. (2) numerically from $t = 0$ to $t = 1$ . By utilizing the learned score function $\mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t )$ , the empirical PF-ODE can be obtained as: $\begin{array} { r } { \frac { \mathrm { d } \mathbf { x } _ { t } } { \mathrm { d } t } = f _ { t } \mathbf { x } _ { t } - \frac { 1 } { 2 } g _ { t } ^ { 2 } \mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t ) } \end{array}$ .

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐，而不只是匹配终点。

Rectified flow (Liu et al., 2022; Liu, 2022; Lipman et al., 2022; Esser et al., 2024) is a generative modeling framework based on ODEs. Given an initial distribution $p _ { 0 }$ and a target data distribution $p _ { 1 }$ , rectified flow trains a neural network to parameterize a velocity field using the following loss function:

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

![Equation 3](../images/c383920facd54057f8fe1a96915ed35c6dbbd24dd83d49afd845121872b0a12e.jpg)
*Equation 3: Equation extracted by MinerU.*

> 💡 **Equation 3 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

raito mple generation is achieved by solving the empirical ODE . In practical implementations, this empirical ODE is sol $\begin{array} { r } { \frac { \mathrm { d } \mathbf { x } _ { t } } { \mathrm { d } t } = \mathbf { v } _ { \theta } ( \mathbf { x } _ { t } , t ) } \end{array}$ from using $t ~ = ~ 0$ $t \ = \ 1$ standard ODE solvers, ranging from the simple forward Euler method to higher-order methods such as RK2 and RK45.

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐，而不只是匹配终点。

---

## 🔖 Section 总结

### 核心洞察

1. 明确输入、输出、teacher/student 或控制变量。
2. 把每个 loss/模块对应到 fidelity、realism、speed 或 controllability。
3. 关注哪些组件是训练时使用，哪些是推理时仍有成本。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Teacher type | conditional flow-based SR model |
| Distillation target | same sampling ODE trajectory alignment |
| Datasets | FFHQ 256×256, DIV2K, ImageNet 256×256 |
