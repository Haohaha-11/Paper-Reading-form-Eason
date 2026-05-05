[← 返回 README](../README.md)

# 4.4 COMPUTATIONAL OVERHEAD

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 OFTSR 主线的关系**: OFTSR 用 conditional flow teacher 和 ODE-trajectory alignment distillation 构建 one-step SR，并保留可调 fidelity-realism trade-off。

---

Training Cost Comparison. Our distillation algorithm is highly flexible and can be easily applied to any pre-trained diffusion/flow-based conditional model. As shown in Tab. 9, we applied our distillation algorithm to the ResShift(Yue et al., 2024b) pre-trained model and achieved teacherlevel performance in one step, surpassing SinSR(Wang et al., 2024c) in FID with much less training compute. Even taking the training stage into account with a larger model, our method remains more efficient than ResShift. We use $t = 1$ for OFTSR.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

Table 9: Comparison of training cost on single NVIDIA A100.

![Table 9](../images/dc921e452b879de5adc6c67378902cd0c50049348d96a6d3f9ffa98638c97738.jpg)
*Table 9: Table 9: Comparison of training cost on single NVIDIA A100.*

> 💡 **Table 9 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

Inference Cost Comparison. We have included a detailed comparison of the inference cost in Tab. 10, using FLOPS and MAC to measure model complexity. We use $t = 1$ for OFTSR.

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

Table 10: Comparison of inference cost on single NVIDIA A100.

![Table 10](../images/acb01dc280931857b3c1d9debfad33ed63a5cbfa0ba3f543cc3265c0a50bd670.jpg)
*Table 10: Table 10: Comparison of inference cost on single NVIDIA A100.*

> 💡 **Table 10 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

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
| Teacher type | conditional flow-based SR model |
| Distillation target | same sampling ODE trajectory alignment |
| Datasets | FFHQ 256×256, DIV2K, ImageNet 256×256 |
