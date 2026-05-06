[← 返回 README](../README.md)

# 5 Conclusion and Limitation

## 📌 预览
本节收束贡献与局限，重点转化为“还能做什么”的研究问题。
---

> 💡 **Q&A 批注记录**:
> - Q: 为什么从 LQ latent 而不是纯噪声开始？
> - A: LQ latent 已经包含布局和低频结构，单步模型只需补细节；纯噪声则需要在一步内同时生成结构和纹理，难度更高且不确定性更强。


We proposed OSEDiff, a one-step effective diffusion network for Real-ISR, by utilizing the pretrained text-to-image model as both the generator and the regularizer in training. Unlike traditional multi-step diffusion models, OSEDiff directly took the given LQ image as the starting point for diffusion, eliminating the uncertainty associated with random noise. By fine-tuning the pre-trained diffusion network with trainable LoRA layers, OSEDiff can well adapt to the complex real-world image degradations. Meanwhile, we performed the variational score distillation in the latent space to ensure that the model’s predicted scores align with those of multi-step pre-trained models, enabling OSEDiff to efficiently produce HQ images in one diffusion step. Our experiments showed that OSEDiff achieved comparable or superior Real-ISR outcomes to previous multi-step diffusion-based methods in both objective metrics and subjective assessments. We believe our exploration can facilitate the practical application of pre-trained T2I models to Real-ISR tasks.
> 💡 **结论批读**: 结论把四个模块重新串了一遍：LQ 起点去随机性，LoRA 适配真实退化，latent VSD 提供 HQ 分布正则，一步推理负责效率。推理阶段不需要 teacher/regularizer，这是效率 claim 成立的关键。


There are some limitations of OSEDiff. First, the details generation capability of OSEDiff can be further improved. Second, like other SD-based methods, OSEDiff is limited in reconstructing fine-scale structures such as small scene texts. We will investigate these problems in further work.
> 💡 **局限批读**: 作者承认的短板集中在细节生成和小场景文字。这个局限和 VSD/SD prior 的风险一致：生成先验能补纹理，但对严格结构、文字笔画和可验证语义未必保真。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 本节作用 | 收束贡献并暴露失败边界 |
| 主线 | 把 LQ latent 直接作为扩散起点，用 latent-space VSD 将 Stable Diffusion 先验压缩到单步 Real-ISR。 |
| 后续关注 | 小文字、细结构、可控 realism/fidelity、端侧部署 |

### 核心洞察
1. 结论确认 OSEDiff 的实用价值主要来自 one-step + 少量 LoRA 参数，而不是比多步 SD 更强的上限生成能力。
2. 论文的局限和实验 trade-off 对上了：感知指标强，不代表小结构/文字绝对可信。
3. 后续工作自然指向结构保护、可控先验强度、模型压缩和更细粒度的失败案例评估。

### 可追问点
- 为什么从 LQ latent 而不是纯噪声开始？
- VSD 在这里起什么作用？
