[← 返回 README](../README.md)

# B.2 OFTSR AS FORWARD DISTILLATION

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 OFTSR 主线的关系**: OFTSR 用 conditional flow teacher 和 ODE-trajectory alignment distillation 构建 one-step SR，并保留可调 fidelity-realism trade-off。

---

The general form of our OFTSR loss or BOOT (Gu et al., 2023) loss can be seen as a special case of forward distillation (Boffi et al., 2025; Sabour et al., 2025; Liu, 2025). Start from general relation:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 17](../images/53cd7381cc4b3c30143cb286f47d3fd609822cb58cbe7b5d645dc30eb7fa9787.jpg)
*Equation 17: Equation extracted by MinerU.*

> 💡 **Equation 17 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where ${ \bf v } _ { \phi } ( { \bf x } _ { t } , t , s )$ is the mean velocity defined on the time interval $[ t , s ]$ as defined in MeanFlow (Geng et al., 2025).

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐，而不只是匹配终点。

The MeanFlow loss can be derived directly by taking derivative w.r.t. $t$ of Eq. (17), which is also named as backward distillation loss. Similarly, when taking derivative w.r.t. $s$ , the end timestep of the interval, of Eq. (17), we get the forward distillation loss.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

For our OFTSR loss, if we consider mapping from arbitrary start timestep $t$ to two close end timestep $s _ { 1 }$ and $s _ { 2 }$ , and connecting the corresponding two state $\mathbf { x } _ { s _ { 1 } }$ and $\mathbf { x } _ { s _ { 2 } }$ , we have:

> 💡 **批注**: 这是控制机制：作者试图把退化强度、区域差异或 timestep 映射成可操作的生成强度。

![Equation 18](../images/47040a540d74b189023a10c98bb077c874ba295ade0b134bd038c0e2ecf7b9e5.jpg)
*Equation 18: Equation extracted by MinerU.*

> 💡 **Equation 18 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

For $s _ { 2 } - s _ { 1 } = \mathrm { d } s$ and $\operatorname* { l i m } _ { \mathrm { d } s \to 0 }$ , we have $s _ { 1 } = s _ { 2 } = s$ and:

![Equation 19](../images/7308375dd1020a73e58d785b75e513d0d6126b3ee1056cfc22e92571df3c9f9d.jpg)
*Equation 19: Equation extracted by MinerU.*

> 💡 **Equation 19 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

which recovers the forward distillation loss as the time derivative w.r.t. $s$ of Eq. (17). Thus we can view the OFTSR loss and BOOT (Gu et al., 2023) loss as a discretization of the forward distillation loss. And it is easy to verify that the signal ODE in BOOT is equivalent to our distillation loss in the flow schedule.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

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
