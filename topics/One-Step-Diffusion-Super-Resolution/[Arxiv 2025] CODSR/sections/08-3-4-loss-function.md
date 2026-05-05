[← 返回 README](../README.md)

# 3.4. Loss function

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 CODSR 主线的关系**: CODSR 从 LQ 信息损失、区域差异化先验激活和文本-区域错配三方面补强 controllable one-step diffusion SR，在单步推理下兼顾感知质量与局部保真。

---

We train the proposed network using a two-stage training strategy. To better constrain the network training in the first stage, we use the pixel-wise content loss function and the learned perceptual image patch similarity (LPIPS) [64] loss function. In addition, the GAN-based loss function [59] is used to enhance the generation quality, particularly by improving realistic details and textures.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

In the second stage, we adopt a dual-LoRA training strategy inspired by PiSA-SR [40]. To further enhance the semantic generation capability, semantic knowledge is distilled from a pretrained model using VSD loss [50]. The LoRA adapters trained in the first stage are frozen, while additional LoRA adapters are integrated exclusively into the cross-attention layers of the U-Net to enhance text alignment. This stage focuses on optimizing the following objective to address potential text misalignment issues:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 5](../images/582053e77467779792fc1e1e20dae7df8ceb4d56e47875e6cb04a1992da91952.jpg)
*Equation 5: Equation extracted by MinerU.*

> 💡 **Equation 5 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where $\mathcal { L } _ { \mathrm { { O S E D i f f } } }$ denotes the loss function used in OSEDiff [50], $\eta _ { \mathrm { p o s } }$ is a weight parameter that is empirically set to 1 in the proposed method and ${ \mathcal { L } } _ { \mathrm { p o s } }$ is the positive area loss used in CoMat [19] to supervise text-matching interaction.

> 💡 **批注**: 这里涉及条件信号：prompt 是否准确、是否退化感知，会直接影响生成细节与语义一致性。

Table 1. Quantitative comparison with state-of-the-art diffusion-based SR methods on four real-world test datasets. $\uparrow$ indicates higher is better, $\downarrow$ indicates lower is better. The best and the second-best results are highlighted in red and blue, respectively.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标，避免被单一数字误导。

![Table 1.](../images/13f35ce3bc9936c12d26fc73ee03afdc0d54458879e3e4d0bf1bc9f9db9d45bb.jpg)
*Table 1.: Table 1. Quantitative comparison with state-of-the-art diffusion-based SR methods on four real-world test datasets. $\uparrow$ indicates higher is better, $\downarrow$ indicates lower is better. The best and the second-best results are highlighted in red and blue, respectively.*

> 💡 **Table 1. 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

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
| Core controls | region-adaptive prior activation + text-matching guidance |
| Project | https://github.com/Chanson94/CODSR |
| Benchmarks | RealSR/DrealSR and other real-world SR test sets in paper tables |
