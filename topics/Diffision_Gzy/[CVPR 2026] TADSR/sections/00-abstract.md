[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要提出 TADSR 的核心问题：现有 one-step VSD 方法固定 student timestep，无法充分利用 SD 在不同 timestep 的不同 generative prior。


Diffusion-based real-world image super-resolution (Real-ISR) methods have demonstrated impressive performance. To achieve efficient Real-ISR, many works employ Variational Score Distillation (VSD) to distill pre-trained stablediffusion (SD) model for one-step SR with a fixed timestep. However, since SD will perform different generative priors at different timesteps, a fixed timestep is difficult for these methods to fully leverage the generative priors in SD, leading to suboptimal performance. To address this, we propose a Time-Aware one-step Diffusion Network for Real-ISR (TADSR). We first introduce a Time-Aware VAE Encoder, which projects the same image into different latent features based on timesteps. Through joint dynamic variation of timesteps and latent features, the student model can better align with the input pattern distribution of the pretrained SD, thereby enabling more effective utilization of SD’s generative capabilities. To better activate the generative prior of SD at different timesteps, we propose a Time-Aware VSD loss that bridges the timesteps of the student model and those of the teacher model, thereby producing more consistent generative prior guidance conditioned on timesteps. Additionally, though utilizing the generative prior in SD at different timesteps, our method can naturally achieve controllable trade-offs between fidelity and realism by changing the timestep. Experimental results demonstrate that our method achieves both state-ofthe-art performance and controllable SR results with only a single step. The source codes are released at https: //github.com/zty557/TADSR

> 💡 **问题动机**: 现有 one-step VSD 把 SD prior 压成一步，但固定 timestep 让 student 只能学一个固定噪声语义，难以覆盖 SD 在不同 timestep 的生成能力。


---

## 🔖 Section 总结
- TADSR 的一句话目标是让 one-step SR 真正使用可变 timestep。
- 核心模块是 TAE 和 TAVSD，分别改变 latent 分布和 score guidance。
- 可追问：timestep 控制是否可校准成用户可理解的 realism slider？
