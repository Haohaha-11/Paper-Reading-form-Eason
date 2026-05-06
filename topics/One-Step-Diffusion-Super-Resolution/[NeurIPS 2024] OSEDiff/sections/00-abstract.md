[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要快速定位 OSEDiff 的问题、方法与结果。先记住主线：把 LQ latent 直接作为扩散起点，用 latent-space VSD 将 Stable Diffusion 先验压缩到单步 Real-ISR。
---

> 💡 **Q&A 批注记录**:
> - Q: 为什么从 LQ latent 而不是纯噪声开始？
> - A: LQ latent 已经包含布局和低频结构，单步模型只需补细节；纯噪声则需要在一步内同时生成结构和纹理，难度更高且不确定性更强。


# One-Step Effective Diffusion Network for Real-World Image Super-Resolution

Rongyuan $\mathbf { W _ { u } } ^ { 1 , 2 , \star }$ , Lingchen $\mathbf { S u n ^ { 1 , 2 , \star } }$ , Zhiyuan $\mathbf { M } \mathbf { a } ^ { 1 , \star }$ , Lei Zhang1,2,†
> 💡 **作者信息**: 港理工 + OPPO Research Institute 的组合也解释了论文对效率和部署实用性的强调，不只是追求视觉上限。


1The Hong Kong Polytechnic University 2OPPO Research Institute {rong-yuan.wu, ling-chen.sun, zm2354.ma}@connect.polyu.hk, cslzhang $@$ comp.polyu.edu.hk ⋆Equal contribution †Corresponding author
> 💡 **读法提示**: 摘要前的元信息不用细读，真正要抓的是后面四个关键词：random noise、one diffusion step、LoRA finetune、latent-space VSD。


# Abstract

The pre-trained text-to-image diffusion models have been increasingly employed to tackle the real-world image super-resolution (Real-ISR) problem due to their powerful generative image priors. Most of the existing methods start from random noise to reconstruct the high-quality (HQ) image under the guidance of the given low-quality (LQ) image. While promising results have been achieved, such Real-ISR methods require multiple diffusion steps to reproduce the HQ image, increasing the computational cost. Meanwhile, the random noise introduces uncertainty in the output, which is unfriendly to image restoration tasks. To address these issues, we propose a one-step effective diffusion network, namely OSEDiff, for the Real-ISR problem. We argue that the LQ image contains rich information to restore its HQ counterpart, and hence the given LQ image can be directly taken as the starting point for diffusion, eliminating the uncertainty introduced by random noise sampling. We finetune the pre-trained diffusion network with trainable layers to adapt it to complex image degradations. To ensure that the one-step diffusion model could yield HQ Real-ISR output, we apply variational score distillation in the latent space to conduct KL-divergence regularization. As a result, our OSEDiff model can efficiently and effectively generate HQ images in just one diffusion step. Our experiments demonstrate that OSEDiff achieves comparable or even better Real-ISR results, in terms of both objective metrics and subjective evaluations, than previous diffusion model-based Real-ISR methods that require dozens or hundreds of steps. The source codes are released at https://github.com/cswry/OSEDiff.
> 💡 **摘要主线**: 摘要把问题压得很准：多步 SD-ISR 质量好但慢，随机噪声对 restoration 不友好；OSEDiff 用 LQ latent 起点去随机性，用 LoRA 适配退化，用 latent-space VSD 给一步输出补 HQ 分布先验。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 一句话 | 把 LQ latent 直接作为扩散起点，用 latent-space VSD 将 Stable Diffusion 先验压缩到单步 Real-ISR。 |
| 输入 | LQ image + pretrained Stable Diffusion prior |
| 输出 | 单步 Real-ISR image |

### 核心洞察
1. 摘要先建立全文地图：Real-ISR、随机噪声/多步成本、LQ 起点、LoRA、latent VSD、一步输出。
2. 后文要验证三个 claim：一步是否真的快，质量是否接近多步 SD，VSD/LoRA/LQ 起点是否各有证据。
3. 最重要的变量是 one-step 约束下如何同时保留输入结构和 SD 生成先验。

### 可追问点
- 为什么从 LQ latent 而不是纯噪声开始？
- VSD 在这里起什么作用？
