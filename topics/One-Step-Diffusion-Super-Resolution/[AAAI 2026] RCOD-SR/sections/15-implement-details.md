[← 返回 README](../README.md)

# Implement Details

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 RCOD-SR 主线的关系**: RCOD-SR 在 one-step diffusion Real-ISR 中引入 latent domain grouping、退化感知采样蒸馏和视觉 prompt 注入，让单步模型在推理时可控地调节 realism。

---

Model Setting: To verify our framework, we select two types of recent SOTA OSD Real-ISR methods: OSEDiff (Wu et al. 2024a), which is distilled from a pre-trained multi-step Stable Diffusion (SD) model, and S3Diff (Zhang et al. 2024), which directly uses SD-T (a distilled version of SD), as our base models. When RCOD is applied to them, they are named $\mathrm { R C O D _ { O } }$ and $\mathrm { R C O D } _ { \mathrm { S } }$ , respectively. The choice of $n = 3$ corresponds to three distinct generation levels in the inference stage: Fidelity, Neutral, and Realism. i.e, $t \ : = \ : 2 5 0$ , 500, and 750 during inference. Since the S3Diff is not directly distilled from a multi-step diffusion, we do not apply DAS on it. ${ \mathrm { R C O D } } _ { \mathrm { S } }$ -Adap. employs MEM during inference. The input to the MEM in this case is $z _ { L }$ and the features in the last layer degradation estimation model used in (Zhang et al. 2024). The MEM is trained after $\mathrm { R C O D } _ { \mathrm { S } }$ training, utilizing the same training data. More details please refer to SM.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

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
| Control variable | latent-domain group / denoising degree |
| Accepted venue | AAAI 2026 Oral according to arXiv comment |
| Training data change | minimal paradigm modification and original training data claimed |
