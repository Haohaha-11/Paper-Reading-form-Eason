[← 返回 README](../README.md)

# One-step Diffusion Models for Real-ISR

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 RCOD-SR 主线的关系**: RCOD-SR 在 one-step diffusion Real-ISR 中引入 latent domain grouping、退化感知采样蒸馏和视觉 prompt 注入，让单步模型在推理时可控地调节 realism。

---

To reduce the computational cost during inference, onestep diffusion methods have been proposed. These methods utilize model distillation techniques specifically designed for diffusion models, including progressive distillation (Meng et al. 2023; Salimans and Ho 2022), consistency models (Song et al. 2023), distribution matching distillation (Yin et al. 2024b,a), and variational score distillation (VSD) (Wang et al. 2024c; Nguyen and Tran 2024). In the context of Real-ISR, OSEDiff (Wu et al. 2024a) employs the VSD strategy to achieve one-step diffusion based on a pre-trained SD model. Similarly, S3Diff (Zhang et al. 2024) directly employs a distilled Stable Diffusion Turbo (SD-T) model for Real-ISR. Recent works including ADCSR (Chen et al. 2025) and TSD-SR (Dong et al. 2025) also improve OSD performance and efficiency. InvSR(Yue, Liao, and Loy 2025) offers a flexible sampling mechanism with arbitrarysteps (1-5) diffusion.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

However, while time efficiency is improved, another dilemma arises: one-step diffusion methods struggle to generate diverse outputs balancing fidelity and realism—a critical requirement for real-world applications—unlike multistep approaches that achieve this through progressive noise scheduling. In other words, current one-step diffusion methods for Real-ISR cannot control fidelity and realism as effectively as their multi-step counterparts. Some techniques, such as ControlNet models (Zhang, Rao, and Agrawala 2023), can enhance HR image generation by controlling semantic content but cannot achieve pixel-level control and require additional conditions, such as extra images or depth maps. Therefore, these techniques cannot be directly employed in one-step diffusion methods for Real-ISR. OFTSR (Zhu et al. 2024) achieves fidelity trade-offs through trajectory alignment distillation, but lacks degradationaware mechanisms for real-world scenarios. PiSA-SR (Sun et al. 2025) controls fidelity and realism by training and inference with two different LoRA (Hu et al. 2022) modules and two-step diffusion, which obviously increases the computational cost compared to using a single step. Thus, a simple, efficient, and generalizable method for fidelity-realism control in OSD for Real-ISR remains essential.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

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
| Control variable | latent-domain group / denoising degree |
| Accepted venue | AAAI 2026 Oral according to arXiv comment |
| Training data change | minimal paradigm modification and original training data claimed |
