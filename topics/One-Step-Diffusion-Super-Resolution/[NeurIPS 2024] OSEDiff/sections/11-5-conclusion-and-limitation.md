[← 返回 README](../README.md)

# 5 Conclusion and Limitation

## 📌 预览
Conclusion 总结贡献和限制；适合回看本文真正解决了 one-step SR 的哪一个子问题。

---

We proposed OSEDiff, a one-step effective diffusion network for Real-ISR, by utilizing the pretrained text-to-image model as both the generator and the regularizer in training. Unlike traditional multi-step diffusion models, OSEDiff directly took the given LQ image as the starting point for diffusion, eliminating the uncertainty associated with random noise. By fine-tuning the pre-trained diffusion network with trainable LoRA layers, OSEDiff can well adapt to the complex real-world image degradations. Meanwhile, we performed the variational score distillation in the latent space to ensure that the model’s predicted scores align with those of multi-step pre-trained models, enabling OSEDiff to efficiently produce HQ images in one diffusion step. Our experiments showed that OSEDiff achieved comparable or superior Real-ISR outcomes to previous multi-step diffusion-based methods in both objective metrics and subjective assessments. We believe our exploration can facilitate the practical application of pre-trained T2I models to Real-ISR tasks.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

There are some limitations of OSEDiff. First, the details generation capability of OSEDiff can be further improved. Second, like other SD-based methods, OSEDiff is limited in reconstructing fine-scale structures such as small scene texts. We will investigate these problems in further work.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

---

## 🔖 Section 总结

### 核心洞察

1. 本节帮助定位论文贡献边界。
2. 读完后回到 README 的方法对比表中归纳。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Inference time | 0.11s on A100 for 512×512 input |
| Trainable parameters | 8.5M |
| Speedup claim | over 100× faster than StableSR in paper comparison |
