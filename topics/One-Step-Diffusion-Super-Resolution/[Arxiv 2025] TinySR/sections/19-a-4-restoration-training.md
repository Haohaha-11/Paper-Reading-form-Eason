[← 返回 README](../README.md)

# A.4. Restoration Training.

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 TinySR 主线的关系**: TinySR 面向 Real-ISR 的实时落地，把已有 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存策略压成轻量单步模型。

---

We perform depth pruning on the TSD-SR according to the pruning mask and discard the condition-related components to initialize our student network. To achieve rapid convergence, we divided the model’s training into two stages. In the first stage, training is exclusively conducted within the latent space. We employ $\mathrm { L _ { 1 } }$ loss for teacher-student knowledge distillation to align features. The formulation of this distillation loss is consistent with that described in Equation (A.3):

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 17](../images/5759882294fe5b24171b7318abd24a98b5b0d206898d73bcde562b1b80f1b763.jpg)
*Equation 17: Equation extracted by MinerU.*

> 💡 **Equation 17 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

The meaning of $z _ { s t u }$ and $z _ { t e a }$ is the same as mentioned above. Training in the latent space enables us to use a larger global batch size (128) on 8 V100 GPUs. We set the learning rate to 1e-4 and the LoRA rank to 64. We iterate training $1 5 0 \mathrm { k }$ steps until convergence.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

In stage 2, we further enhance the perceptual quality of the results in image space by fine-tuning the model directly within the image domain. We additionally incorporate LPIPS loss and GAN loss to enhance image restoration. The total loss is expressed as follows:

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

![Equation 18](../images/d5d3be3456ff3342ac4e03b666da8698940d3a5f2f90e2fda2fc01c97ecad511.jpg)
*Equation 18: Equation extracted by MinerU.*

> 💡 **Equation 18 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

The meaning of $z _ { s t u }$ , $z _ { t e a }$ and $\mathcal { D } _ { t i n y }$ is the same as mentioned above. $\lambda _ { \mathrm { 5 } } , \lambda _ { 6 }$ , and $\lambda _ { 7 }$ are set to 5, 1, and 0.3, respectively. We fine-tune our model for 50k steps on 8 V100 GPUs, with a global batch size of 96, a learning rate of 1e-6 for student (5e-6 for discriminator), and a LoRA rank of 64. The random seed for the entire training process is set to 80. And all training is done on fp16 precision.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

---

## 🔖 Section 总结

### 核心洞察

1. 明确输入、输出、teacher/student 或控制变量。
2. 把每个 loss/模块对应到 fidelity、realism、speed 或 controllability。
3. 关注哪些组件是训练时使用，哪些是推理时仍有成本。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Speedup | up to 5.68× over TSD-SR teacher |
| Parameter reduction | 83% reduction claimed in abstract |
| Inference regime | one-step Real-ISR |
| Compression targets | Diffusion backbone + VAE + prompt/time modules |
