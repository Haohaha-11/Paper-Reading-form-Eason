[← 返回 README](../README.md)

# G.3 DIFFERENT RESOLUTION AND SCALE FACTOR (SF)

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 OFTSR 主线的关系**: OFTSR 用 conditional flow teacher 和 ODE-trajectory alignment distillation 构建 one-step SR，并保留可调 fidelity-realism trade-off。

---

In this work, by default we follow previous works to use the setup of $4 \times \mathrm { S R }$ at $2 5 6 \times 2 5 6$ . We also test $\mathrm { S F } = 8$ on $2 5 6 \times 2 5 6$ and $\mathrm { S F } = 4 \& 8$ on 512-resolution FFHQ, the results are shown in Tab. 13. All models are trained for $3 0 \mathrm { k }$ iterations $\mathrm { { b s } = 3 2 }$ ) and distilled for $1 0 \mathrm { k }$ iterations $\mathrm { { b s } = 1 6 }$ ). We visualize $8 \times$ and $4 \times$ reconstruction of teacher and student in Fig. 9.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

Table 13: A comparison of the models trained across different resolution and scale factor.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标，避免被单一数字误导。

![Table 13](../images/0a5a50cf1c851d9b7e6ac1e95ddf94443a5513aec91c717b6e057714af135832.jpg)
*Table 13: Table 13: A comparison of the models trained across different resolution and scale factor.*

> 💡 **Table 13 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

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
