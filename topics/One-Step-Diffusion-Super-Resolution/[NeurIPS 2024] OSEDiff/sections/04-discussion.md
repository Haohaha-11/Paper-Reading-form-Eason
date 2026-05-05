[← 返回 README](../README.md)

# Discussion

## 📌 预览
本文件合并 Conclusion/Appendix/References，重点看实现细节、局限、额外结果和可追踪文献。

---

# 5 Conclusion and Limitation

We proposed OSEDiff, a one-step effective diffusion network for Real-ISR, by utilizing the pretrained text-to-image model as both the generator and the regularizer in training. Unlike traditional multi-step diffusion models, OSEDiff directly took the given LQ image as the starting point for diffusion, eliminating the uncertainty associated with random noise. By fine-tuning the pre-trained diffusion network with trainable LoRA layers, OSEDiff can well adapt to the complex real-world image degradations. Meanwhile, we performed the variational score distillation in the latent space to ensure that the model’s predicted scores align with those of multi-step pre-trained models, enabling OSEDiff to efficiently produce HQ images in one diffusion step. Our experiments showed that OSEDiff achieved comparable or superior Real-ISR outcomes to previous multi-step diffusion-based methods in both objective metrics and subjective assessments. We believe our exploration can facilitate the practical application of pre-trained T2I models to Real-ISR tasks.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

There are some limitations of OSEDiff. First, the details generation capability of OSEDiff can be further improved. Second, like other SD-based methods, OSEDiff is limited in reconstructing fine-scale structures such as small scene texts. We will investigate these problems in further work.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

# A Appendix

In the appendix, we provide the following materials:

• Comparison with GAN-based methods (referring to Section 4.1 in the main paper).   
• Results of user study (referring to Section 4.1 in the main paper).   
• More real-world visual comparisons under scaling factor $4 \times$ (referring to Section 4.2 in the main paper).   
• Training algorithm of OSEDiff (referring to Section 3.2 in the main paper).

> 💡 **列表批读**: 这组条目通常是在列贡献、设置或发现；建议逐条对应到论文声称解决的 gap。

---

## 🔖 Section 总结

### 核心洞察
1. 总结可复用设计和局限。
2. References 可用于扩展本课题文献图谱。
