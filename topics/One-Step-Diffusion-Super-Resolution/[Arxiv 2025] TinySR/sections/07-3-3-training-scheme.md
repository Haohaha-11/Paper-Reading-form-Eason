[← 返回 README](../README.md)

# 3.3. Training Scheme

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 TinySR 主线的关系**: TinySR 面向 Real-ISR 的实时落地，把已有 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存策略压成轻量单步模型。

---

VAE Training. We train the VAE encoder $\mathcal { E } _ { t i n y }$ by aligning latent space features using MSE loss:

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

![Equation 9](../images/25b4b60bf353bf2931cd91ffbe1772c57fc24a92f6c2f84475d264cd27abbf49.jpg)
*Equation 9: Equation extracted by MinerU.*

> 💡 **Equation 9 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

here, $x _ { \mathrm { L R } }$ represents low-quality data, $\mathcal { E } _ { \mathrm { p r e } }$ is the pretrained encoder. Training is conducted for $1 0 0 \mathrm { k }$ steps using a batch size of 64 and a learning rate (AdamW optimizer [32]) of 3e-4 for this phase. We use LPIPS loss and GAN loss to train the VAE decoder $\mathcal { D } _ { t i n y }$ :

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

![Equation 10](../images/4eba764a85a89ca107f9228195654b35c9f84be731e0ed84becb6abfd6a1124f.jpg)
*Equation 10: Equation extracted by MinerU.*

> 💡 **Equation 10 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

here, $x _ { \mathrm { H R } }$ represents high-quality data. We set $\lambda _ { 1 }$ to 3 and $\lambda _ { 2 }$ to 1. These two loss functions are jointly optimized to ensure both high fidelity and superior perceptual quality in the reconstructed images.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

Pruning Decision Training. To ensure the recoverability of pruning, we design the optimization of mask learning using distillation and task losses. Specifically, the task loss is defined as LPIPS loss and $\mathrm { L _ { 1 } }$ loss is utilized for the distillation loss. The total loss is expressed as follows:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 11](../images/d0ee0cc9804c155fb63689b4f52e307b6ffa5cff02fbc38931a80aefbc7dc9d1.jpg)
*Equation 11: Equation extracted by MinerU.*

> 💡 **Equation 11 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

$\epsilon _ { s t u }$ denotes the student denoising network, while $\epsilon _ { t e a }$ represents the teacher. $t$ denotes timesteps, and $\mathcal { E } _ { t e a }$ denotes the teacher encoder. Training is conducted for $1 0 0 \mathrm { k }$ iterations across 8 NVIDIA V100 GPUs, employing a learning rate of 5e-5 and a global batch size of 8. We use LoRA [21] for training. $\lambda _ { 3 }$ and $\lambda _ { 4 }$ are both set to 1.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

Restoration Training. A two-stage training approach is utilized to expedite model convergence. The first stage involves latent space training with the distillation loss for efficient feature alignment:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 12](../images/64cf252cb55e7c7ef7880f5830655ba9b9c4e7303725ac91b6430130320c97c3.jpg)
*Equation 12: Equation extracted by MinerU.*

> 💡 **Equation 12 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

The meaning of $z _ { s t u }$ and $z _ { t e a }$ is the same as mentioned above. Training in the latent space enables us to use a larger global batch size (128) on 8 V100 GPUs. We set the learning rate to 1e-4 and the LoRA rank to 64. The second stage then adds LPIPS and GAN losses in the pixel space to improve image-level fidelity:

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

![Figure 5.](../images/452410207386848bdce6afe90cd93a7d1cd372a351829a22216fedbe8c90fada.jpg)
*Figure 5.: Figure 5. Comparisons of performance and efficiency for various design of efficient TinySR. Quality is measured by the MANIQA score, calculated on RealSR dataset. Efficiency is assessed via latency, MACs, and parameters. Latency and MACs are benchmarked for superresolution a $1 2 8 \times 1 2 8$ low-quality image on a n NVIDIA V100 GPU. Our model achieves a $5 . 6 8 \times$ acceleration with $83 \%$ fewer parameters and $84 \%$ lower MACs, while maintaining comparable quality to the baseline.*

> 💡 **Figure 5. 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

![Figure 6.](../images/6b0d0436f58e7f665bec82d9dcc9e9baf6ac034ef88e37fee5a7574aed527825.jpg)
*Figure 6.: Figure 6. Left: Channel pruning effectively reduces MACs across all modules, particularly in the computationally intensive down/up and middle blocks. Right: Attention mechanism dominates the computational cost in a 64-channel VAE.*

> 💡 **Figure 6. 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

![Figure 7.](../images/a69465b7256df28b4bbc35a2f5945525ab197de079f2f2b20a4d513297ef2767.jpg)
*Figure 7.: Figure 7. Applying $90 \%$ token pruning (TP) yields visually comparable results to the baseline with a slight quality drop, indicating the limited contribution of the default prompt.*

> 💡 **Figure 7. 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

![Equation 13](../images/c3f74d51dac2210ca5539119576e7507536b47a68f38fe75feafae80e719a2a4.jpg)
*Equation 13: Equation extracted by MinerU.*

> 💡 **Equation 13 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where $\lambda _ { 5 } , \lambda _ { 6 }$ , and $\lambda _ { 7 }$ are set to 5, 1, and 0.3, respectively. We fine-tune our model for $5 0 \mathrm { k }$ steps on 8 V100 GPUs, with a global batch size of 96, a learning rate of 1e-6 for student (5e-6 for discriminator), and a LoRA rank of 64.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

Comprehensive experimental details for the training procedure are available in the supplementary material.

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
