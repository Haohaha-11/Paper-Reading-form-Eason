[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要快速给出本文问题、方法和结果。读完先记住一句话：OSEDiff 将低质量图像直接作为扩散起点，用 latent-space VSD 把 Stable Diffusion 先验压缩到单步 Real-ISR 推理。

---

# One-Step Effective Diffusion Network for Real-World Image Super-Resolution

Rongyuan $\mathbf { W _ { u } } ^ { 1 , 2 , \star }$ , Lingchen $\mathbf { S u n ^ { 1 , 2 , \star } }$ , Zhiyuan $\mathbf { M } \mathbf { a } ^ { 1 , \star }$ , Lei Zhang1,2,†

1The Hong Kong Polytechnic University 2OPPO Research Institute {rong-yuan.wu, ling-chen.sun, zm2354.ma}@connect.polyu.hk, cslzhang $@$ comp.polyu.edu.hk ⋆Equal contribution †Corresponding author

# Abstract

The pre-trained text-to-image diffusion models have been increasingly employed to tackle the real-world image super-resolution (Real-ISR) problem due to their powerful generative image priors. Most of the existing methods start from random noise to reconstruct the high-quality (HQ) image under the guidance of the given low-quality (LQ) image. While promising results have been achieved, such Real-ISR methods require multiple diffusion steps to reproduce the HQ image, increasing the computational cost. Meanwhile, the random noise introduces uncertainty in the output, which is unfriendly to image restoration tasks. To address these issues, we propose a one-step effective diffusion network, namely OSEDiff, for the Real-ISR problem. We argue that the LQ image contains rich information to restore its HQ counterpart, and hence the given LQ image can be directly taken as the starting point for diffusion, eliminating the uncertainty introduced by random noise sampling. We finetune the pre-trained diffusion network with trainable layers to adapt it to complex image degradations. To ensure that the one-step diffusion model could yield HQ Real-ISR output, we apply variational score distillation in the latent space to conduct KL-divergence regularization. As a result, our OSEDiff model can efficiently and effectively generate HQ images in just one diffusion step. Our experiments demonstrate that OSEDiff achieves comparable or even better Real-ISR results, in terms of both objective metrics and subjective evaluations, than previous diffusion model-based Real-ISR methods that require dozens or hundreds of steps. The source codes are released at https://github.com/cswry/OSEDiff.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
