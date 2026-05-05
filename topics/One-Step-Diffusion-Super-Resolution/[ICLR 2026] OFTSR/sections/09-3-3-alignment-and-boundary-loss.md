[← 返回 README](../README.md)

# 3.3 ALIGNMENT AND BOUNDARY LOSS

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 OFTSR 主线的关系**: OFTSR 用 conditional flow teacher 和 ODE-trajectory alignment distillation 构建 one-step SR，并保留可调 fidelity-realism trade-off。

---

In BOOT (Gu et al., 2023), a boundary condition is applied to enforce that the one-step student model and teacher model perform the same at the boundary $t = 0$ . We aim to align the teacher and student outputs in our model. The student produces $\mathbf { x } _ { 0 } + \mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , t )$ , while the teacher generates $\mathbf { x } _ { t } + ( 1 - t ) \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t )$ based on the student’s output $\mathbf { x } _ { t }$ using Eq. (7). By minimizing the difference between these outputs, we get the following alignment loss to align the teacher and student:

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Equation 10](../images/67254537d6d930e07503f563d5c5f58683ce09a7ea43d9ecf6744f4ba64a31f7.jpg)
*Equation 10: Equation extracted by MinerU.*

> 💡 **Equation 10 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

If we consider this alignment loss only at $t = 0$ , it becomes equivalent to the boundary loss used in BOOT:

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐，而不只是匹配终点。

![Equation 11](../images/8de7542077b80a2ff76ad7d4030ddc331e1211afb8d8bcb36c6f23d097e64c5e.jpg)
*Equation 11: Equation extracted by MinerU.*

> 💡 **Equation 11 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

Since it is difficult to sample $t = 0$ for most training iterations, we add in addition the boundary loss Eq. (11) in our final training objective.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

The overall training objective. The student network $\mathbf { v } _ { \phi }$ is trained to minimize the combination of the aforementioned three losses terms:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 12](../images/b9d1afd400a798a2fb71e05dde2d2a886ee098aa5e747321b113d62809ada6c8.jpg)
*Equation 12: Equation extracted by MinerU.*

> 💡 **Equation 12 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where $\lambda _ { \mathrm { a l i g n } }$ and $\lambda _ { \mathrm { B C } }$ are the weights for alignment loss and boundary condition loss, respectively.   
The distillation stage of the proposed method is summarized in Algorithm 1.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Figure 3](../images/285f0f974f2f1be9725113f863df28aa28e65f7c1adde48123606b90bb704cf2.jpg)
*Figure 3: 原始图片*

> 💡 **Figure 3 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

![Figure 4](../images/3c4fdaa803f4f85bca8804b9cea92175c7cd5129e56fba37451107f07a07da22.jpg)
*Figure 4: GT LR DPS DDRM DDNM DiffPir SITCOM OursFigure 4: Qualitative comparison with training-free methods. The first row shows noiseless SR on the FFHQ dataset, the second row presents noisy SR $\sigma _ { n } = 0 . 0 5 )$ on FFHQ, and the bottom row demonstrates noiseless SR on the ImageNet dataset. Numbers next to the method names represent the required NFEs.*

> 💡 **Figure 4 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

GT LR DPS DDRM DDNM DiffPir SITCOM OursFigure 4: Qualitative comparison with training-free methods. The first row shows noiseless SR on the FFHQ dataset, the second row presents noisy SR $\sigma _ { n } = 0 . 0 5 )$ on FFHQ, and the bottom row demonstrates noiseless SR on the ImageNet dataset. Numbers next to the method names represent the required NFEs.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标，避免被单一数字误导。

Inference. After training, the one-step student $\mathbf { v } _ { \phi }$ produces the final high-resolution output $\mathbf { x } _ { 1 } ^ { t }$ in a single forward pass, conditioned on the initial state $\mathbf { x } _ { \mathrm { 0 } }$ , the low-resolution input $\mathbf { x } _ { \mathrm { L R } }$ , and the trade-off parameter $t$ . Concretely,

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Equation 13](../images/5c7d882c6fa52d69a0b4c1fbd4ed887f3226b36c24fd7b2165f99e10bb12ba18.jpg)
*Equation 13: Equation extracted by MinerU.*

> 💡 **Equation 13 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where $\mathbf { v } _ { \phi } ( \cdot )$ predicts a residual that refines $\mathbf { x } _ { \mathrm { 0 } }$ toward the desired point on the fidelity–realism curve specified by $t$ .

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

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
