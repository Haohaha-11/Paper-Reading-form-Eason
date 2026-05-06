[← 返回 README](../README.md)

# Related Work

## 📌 预览
本节定位 RCOD-SR 的方法谱系：GAN/传统 Real-ISR 解决退化合成但容易出伪影，多步 diffusion SR 质量强但慢，one-step diffusion SR 快但缺少推理时的 fidelity-realism 控制。
---

> 💡 **Q&A 批注记录**:
> - Q: 为什么单步 SR 难以控制 realism？
> - A: 因为多数 one-step 模型在固定 timestep/固定蒸馏强度下学习一个确定映射，推理时没有多步采样那样的 noise schedule 可调自由度。
> - Q: RCOD 和 PiSA-SR/OFTSR 的区别应怎么抓？
> - A: RCOD 强调 degradation-aware 的 timestep/group 控制，并保持单步；PiSA-SR 用双 LoRA/两步带来额外成本，OFTSR 有可调 trade-off 但缺少面向真实退化的 LDG/DAS 机制。


# Real-World Image Super-Resolution

To address the challenge of recovering real-world lowresolution (LR) observations with unknown degradations, the Real-ISR task has been introduced. Due to the complex degradations involved, Real-ISR has been a challenging problem for some time (Ignatov et al. 2017; Liu et al. 2022; Ji et al. 2020). Initial methods trained their models with simple downsampling techniques (Kim, Kwon Lee, and Mu Lee 2016; Zhang et al. 2018b), which led to poor performance on real-world datasets. BSRGAN (Zhang et al. 2021) and Real-ESRGAN (Wang et al. 2021) were among the first to introduce a more realistic synthesis pipeline for LR images, enabling deep learning methods to be applied to real-world scenarios. However, these GAN-based methods often suffer from artifacts and training instability. With the emergence of the powerful pre-trained text-to-image generation model Stable Diffusion (SD) (Rombach et al. 2022), many efforts have been made to leverage its strong generative capabilities for solving the Real-ISR problem, such as DiffBIR (Lin et al. 2023), StableSR (Wang et al. 2024a) and SeeSR (Wu et al. 2024b), and FaithDiff (Chen, Pan, and Dong 2025). These SD-based methods improve fidelity and perceptual quality, but their application is limited due to the significant computational resources and time required. This is primarily due to the dozens to hundreds of timesteps involved in the diffusion denoising process.
> 💡 **谱系批读**: 这段把 RCOD 的两端参照物摆出来：GAN Real-ISR 快但可能伪影/不稳定，多步 SD-SR 借强生成先验提升观感但采样慢。RCOD 的位置就是想保留 SD prior，同时避免多步延迟。


# One-step Diffusion Models for Real-ISR
> 💡 **小节预览**: 这里要看 one-step 路线怎样获得速度，以及为什么速度提升后又暴露出可控性问题。


To reduce the computational cost during inference, onestep diffusion methods have been proposed. These methods utilize model distillation techniques specifically designed for diffusion models, including progressive distillation (Meng et al. 2023; Salimans and Ho 2022), consistency models (Song et al. 2023), distribution matching distillation (Yin et al. 2024b,a), and variational score distillation (VSD) (Wang et al. 2024c; Nguyen and Tran 2024). In the context of Real-ISR, OSEDiff (Wu et al. 2024a) employs the VSD strategy to achieve one-step diffusion based on a pre-trained SD model. Similarly, S3Diff (Zhang et al. 2024) directly employs a distilled Stable Diffusion Turbo (SD-T) model for Real-ISR. Recent works including ADCSR (Chen et al. 2025) and TSD-SR (Dong et al. 2025) also improve OSD performance and efficiency. InvSR(Yue, Liao, and Loy 2025) offers a flexible sampling mechanism with arbitrarysteps (1-5) diffusion.
> 💡 **蒸馏批读**: OSEDiff/S3Diff 是 RCOD 的直接底座。读到这里要记住：one-step 的能力来自把多步 prior 压进 student 或使用 distilled SD-Turbo，但压缩后也丢掉了多步过程中可调 timestep 的自由度。


However, while time efficiency is improved, another dilemma arises: one-step diffusion methods struggle to generate diverse outputs balancing fidelity and realism—a critical requirement for real-world applications—unlike multistep approaches that achieve this through progressive noise scheduling. In other words, current one-step diffusion methods for Real-ISR cannot control fidelity and realism as effectively as their multi-step counterparts. Some techniques, such as ControlNet models (Zhang, Rao, and Agrawala 2023), can enhance HR image generation by controlling semantic content but cannot achieve pixel-level control and require additional conditions, such as extra images or depth maps. Therefore, these techniques cannot be directly employed in one-step diffusion methods for Real-ISR. OFTSR (Zhu et al. 2024) achieves fidelity trade-offs through trajectory alignment distillation, but lacks degradationaware mechanisms for real-world scenarios. PiSA-SR (Sun et al. 2025) controls fidelity and realism by training and inference with two different LoRA (Hu et al. 2022) modules and two-step diffusion, which obviously increases the computational cost compared to using a single step. Thus, a simple, efficient, and generalizable method for fidelity-realism control in OSD for Real-ISR remains essential.
> 💡 **差异化批读**: 这段是 related work 的落点。ControlNet 控语义但条件重且不适合像素级 SR；OFTSR 有 trade-off 但没有退化感知机制；PiSA-SR 可调但需要双 LoRA/两步。RCOD 要占的位置是：单步、可插拔、degradation-aware、推理时用 timestep 控制 realism。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 谱系 | GAN/Restoration → multi-step diffusion → one-step diffusion/flow |
| 本文位置 | 把退化严重度映射到 latent domain group / timestep，使单步 Real-ISR 也能在推理时调 fidelity-realism trade-off。 |
| 比较维度 | 速度、保真、真实感、可控性、部署 |

### 核心洞察
1. 本节不是在证明 diffusion SR 比 GAN 好，而是在说明速度、质量、可控性三者目前没有被同时解决。
2. RCOD 的关键差异是把真实退化程度纳入 timestep/蒸馏采样，而不是只给一个语义控制或多模型切换。
3. 后文实验应重点比较 OSEDiff、S3Diff、OFTSR、PiSA-SR 这几类最接近的路线。

### 可追问点
- 为什么单步 SR 难以控制 realism？
- LDG 真正控制的是什么？
- RCOD 的单步控制是否真的比双 LoRA/两步方案更省？
