[← 返回 README](../README.md)

# 5 Conclusion and Limitation

## 📌 预览
本节总结贡献和局限，适合回看方法真正解决了什么问题。

---

We proposed OSEDiff, a one-step effective diffusion network for Real-ISR, by utilizing the pretrained text-to-image model as both the generator and the regularizer in training. Unlike traditional multi-step diffusion models, OSEDiff directly took the given LQ image as the starting point for diffusion, eliminating the uncertainty associated with random noise. By fine-tuning the pre-trained diffusion network with trainable LoRA layers, OSEDiff can well adapt to the complex real-world image degradations. Meanwhile, we performed the variational score distillation in the latent space to ensure that the model’s predicted scores align with those of multi-step pre-trained models, enabling OSEDiff to efficiently produce HQ images in one diffusion step. Our experiments showed that OSEDiff achieved comparable or superior Real-ISR outcomes to previous multi-step diffusion-based methods in both objective metrics and subjective assessments. We believe our exploration can facilitate the practical application of pre-trained T2I models to Real-ISR tasks.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

There are some limitations of OSEDiff. First, the details generation capability of OSEDiff can be further improved. Second, like other SD-based methods, OSEDiff is limited in reconstructing fine-scale structures such as small scene texts. We will investigate these problems in further work.

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
