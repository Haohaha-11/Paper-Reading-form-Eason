[← 返回 README](../README.md)

# 4.3 ABLATIONS

## 📌 预览
Analysis/Ablation 用来判断每个模块是否必要，特别要看控制变量、loss 和轻量化组件是否各自贡献明确。

---

Perturbation Strength $\sigma _ { p }$ . In Tab. 7, we evaluate the design choices in the simple conditional flow training stage. All experiments in this ablation study are conducted under identical training conditions, with performance metrics measured using the RK45 solver. The most critical hyperparameter in this ablation is the strength of the perturbation $\sigma _ { p }$ . Consistent with previous works, we confirm that perturbation is essential for generating perceptually compelling images from LR inputs. Notably, we discover that increasing perturbation strength does not necessarily improve perceptual quality but instead leads to more curved PF-ODE, requiring additional NFEs to solve (see Tab. 7). Furthermore, our experiments demonstrate that conditioning on $\mathbf { x } _ { \mathrm { L R } }$ is crucial to compensate for information loss during perturbation. We also find that $\ell _ { 1 }$ loss outperforms $\ell _ { 2 }$ loss for our specific task. While (Kim et al., 2024) previously highlighted the significance of Gaussian perturbation, our work is the first to systematically analyze the relationship between noise perturbation and the trade-off between generation quality and efficiency in flow-based models.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

Distillation Design Space. In Tab. 8, we evaluate several crucial design choices for the distillation stage, including the distillation loss type, solver type, $\mathrm { d } t$ value, and the weighting of alignment and boundary losses. Since learning ${ \mathbf { v } } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , 0 )$ is considerably easier than learning $\mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , 1 )$ , we utilize metrics from the latter to decide our distillation hyperparameters. Our analysis of the step size

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

Table 7: Ablation on noiseless FFHQ $2 5 6 \times 2 5 6$ first stage. The default training setting is ${ \mathfrak { b s } } = 3 2$ $\mathrm { ~ k ~ } = \ 0 . 0 0 0 1$ ; loss type $\mathit { \Theta } = \mathit { \Theta } \ell _ { 1 }$ ; with condition; all experiments are trained for $1 0 0 \mathrm { k }$ steps. The final choice is highlighted to balance the performance and efficiency.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

![Table 7](../images/6797bd1a22bec3d2e94e483c9c065869a955f2281342589e7fe61a13747d6bad.jpg)
*Table 7: Table 7: Ablation on noiseless FFHQ $2 5 6 \times 2 5 6$ first stage. The default training setting is ${ \mathfrak { b s } } = 3 2$ $\mathrm { ~ k ~ } = \ 0 . 0 0 0 1$ ; loss type $\mathit { \Theta } = \mathit { \Theta } \ell _ { 1 }$ ; with condition; all experiments are trained for $1 0 0 \mathrm { k }$ steps. The final choice is highlighted to balance the performance and efficiency.*

> 💡 **Table 7 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

Table 8: Ablation on noiseless FFHQ $2 5 6 \times 2 5 6$ distillation stage. The default training setting is $\ b s \ = \ 8$ ; $\sigma _ { p } = 0 . 1$ , $\mathrm { l r } = 0 . 0 0 0 1$ ; loss type $= \ell _ { 2 }$ ; with LR condition; all experiments are trained for 20k steps; And the one-step metrics are calculated with $t = 1$ . Ablations in subgroups can be ordered as $\mathrm { d } t \  \ \lambda _ { \mathrm { B C } } \  \ \lambda _ { \mathrm { a l i g n } } \ $ dt reveals that smaller values do not necessarily yield better results, leading us to select $\mathrm { d } t = 0 . 0 5$ for subsequent experiments. Our proposed loss function demonstrates substantial improvement over both the original BOOT (Gu et al., 2023) loss and PINN (Tee et al., 2024) distillation loss, achieving a significant LPIPS score improvement of more than 0.1. Further experimentation shows that both the alignment loss (Eq. (10)) and boundary loss (Eq. (11)) contribute to enhanced performance. By combining these losses with a Midpoint 2-order solver, we achieve additional improvements in our one-step model’s performance at $t = 1$ .

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

Solver , and $\mathrm { d } t \ $ Distillation Loss .

![Table 8](../images/f7518f32363a4dc32c1c99ced8af2132a3656553e67de2414f482150567d4da9.jpg)
*Table 8: Solver , and $\mathrm { d } t \ $ Distillation Loss .*

> 💡 **Table 8 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

---

## 🔖 Section 总结

### 核心洞察

1. 本节帮助定位论文贡献边界。
2. 读完后回到 README 的方法对比表中归纳。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Teacher type | conditional flow-based SR model |
| Distillation target | same sampling ODE trajectory alignment |
| Datasets | FFHQ 256×256, DIV2K, ImageNet 256×256 |
