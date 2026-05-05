[← 返回 README](../README.md)

# Abstract

## 📌 预览
本文件保留论文题名、作者、摘要等开场信息，先抓住方法解决的 one-step SR 核心瓶颈。

---

# Realism Control One-step Diffusion for Real-World Image Super-Resolution

Zongliang Wu1, 2, 3, \*, Siming Zheng 3, \*, Peng-Tao Jiang3, #, Xin Yuan2, #

1 Zhejiang University, Hangzhou, China 2 School of Engineering, Westlake University, Hangzhou, China 3 vivo Mobile Communication Co., Ltd \* Equal Contribution # Corresponding Author {wuzongliang, xyuan}@westlake.edu.cn, {zhengsiming, pt.jiang}@vivo.com

# Abstract

Pre-trained diffusion models have shown great potential in real-world image super-resolution (Real-ISR) tasks by enabling high-resolution reconstructions. While one-step diffusion (OSD) methods significantly improve efficiency compared to traditional multi-step approaches, they still have limitations in balancing fidelity and realism across diverse scenarios. Since the OSDs for SR are usually trained or distilled by a single timestep, they lack flexible control mechanisms to adaptively prioritize these competing objectives, which are inherently manageable in multi-step methods through adjusting sampling steps. To address this challenge, we propose a Realism Controlled One-step Diffusion (RCOD) framework for Real-ISR. RCOD provides a latent domain grouping strategy that enables explicit control over fidelity-realism trade-offs during the noise prediction phase with minimal training paradigm modifications and original training data. A degradation-aware sampling strategy is also introduced to align distillation regularization with the grouping strategy and enhance the controlling of trade-offs. Moreover, a visual prompt injection module is used to replace conventional text prompts with degradation-aware visual tokens, enhancing both restoration accuracy and semantic consistency. Our method achieves superior fidelity and perceptual quality while maintaining computational efficiency. Extensive experiments demonstrate that RCOD outperforms state-of-the-art OSD methods quantitatively and visually, with flexible realism control capabilities in the inference stage. Code is available at https://zongliang-wu.github.io/RCOD-SR.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

# In the supplementary material, we provide the following contents:

• Algorithm Details: Pseudo-code of our RCODO training strategy (referring to Section Methodology in the main paper).   
• Implement Details: Hyper-parameters setting and experimental environments (As a complement to Section Implement Details in the main paper).   
• Experiments: Full table of quantitative comparisons and more visual Comparisons (as a complement to Section Experiments in the main paper).   
• Ablation Study (referring to Section Ablation Study in the main paper).

> 💡 **列表批读**: 这组条目通常是在列贡献、设置或发现；建议逐条对应到论文声称解决的 gap。

---

## 🔖 Section 总结

### 核心洞察
1. 先记住本文的问题定义、方法名和主要 claim。
2. 后续阅读时回看摘要里的 claim 是否被实验支持。
