[← 返回 README](../README.md)

# C.4. Ablation Study of Losses in Stage 2

## 📌 预览
Analysis/Ablation 用来判断每个模块是否必要，特别要看控制变量、loss 和轻量化组件是否各自贡献明确。

---

We conduct an ablation study on the Stage 2 training losses, as shown in Table C.4. The results indicate that Stage 2 training significantly improved image quality, particularly

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标，避免被单一数字误导。

Table C.2. Ablation study of pruning ratio on DIV2K-Val dataset. The best is highlighted in bold.

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

![Table C.2.](../images/b2d894764d0351a0f7ac9bfc21954f5416b9fa833e6fe8ebb2518aba30d7878e.jpg)
*Table C.2.: Table C.2. Ablation study of pruning ratio on DIV2K-Val dataset. The best is highlighted in bold.*

> 💡 **Table C.2. 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

in terms of image fidelity. Specifically, we find that the inclusion of LPIPS loss is highly beneficial for improving

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

![Figure B.1.](../images/67371e5642e7ef5f123719257a6647f8d8cacd1639515671b52a836d3c07a853.jpg)
*Figure B.1.: Figure B.1. Qualitative comparisons of GAN-based and diffusion-based Real-ISR methods. Please zoom in for a better view.*

> 💡 **Figure B.1. 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

reference metrics such as DISTS and FID. The addition

of GAN loss, in turn, is helpful for enhancing several no-

Table C.3. Ablation studies of Stage 1 distillation loss on DrealSR dataset. The best (other than Teacher) is highlighted in bold.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Table C.3.](../images/28e9bf6e6df7006daca9e7b0de4db5d40f9c3ce4281459243c0012d74a7565b0.jpg)
*Table C.3.: Table C.3. Ablation studies of Stage 1 distillation loss on DrealSR dataset. The best (other than Teacher) is highlighted in bold.*

> 💡 **Table C.3. 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

![Figure C.1.](../images/5722b64cae7fcc591cdd90eb19d459dc7d304e04b3041e817f4e76b37e77c4a1.jpg)
*Figure C.1.: Figure C.1. Applying $90 \%$ token pruning (TP) yields visually comparable results to the baseline with a slight quality drop, indicating the limited contribution of the default prompt.*

> 💡 **Figure C.1. 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

![Figure C.2.](../images/91db0f414b77629e89e6ede514b1a19bf758394471b363d38d7a0d0027145cc5.jpg)
*Figure C.2.: Figure C.2. Visual comparison of knowledge distillation: highresolution ground truth versus teacher.*

> 💡 **Figure C.2. 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

reference metrics, including NIQE and MANIQA. We ultimately weighted the two new losses to balance the trade-off between fidelity and the generative ability.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

---

## 🔖 Section 总结

### 核心洞察

1. 本节帮助定位论文贡献边界。
2. 读完后回到 README 的方法对比表中归纳。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Speedup | up to 5.68× over TSD-SR teacher |
| Parameter reduction | 83% reduction claimed in abstract |
| Inference regime | one-step Real-ISR |
| Compression targets | Diffusion backbone + VAE + prompt/time modules |
