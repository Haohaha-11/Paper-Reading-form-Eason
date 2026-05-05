[← 返回 README](../README.md)

# Realism Control One-step Diffusion for Real-World Image Super-Resolution

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 RCOD-SR 主线的关系**: RCOD-SR 在 one-step diffusion Real-ISR 中引入 latent domain grouping、退化感知采样蒸馏和视觉 prompt 注入，让单步模型在推理时可控地调节 realism。

---

Zongliang Wu1, 2, 3, \*, Siming Zheng 3, \*, Peng-Tao Jiang3, #, Xin Yuan2, #

1 Zhejiang University, Hangzhou, China 2 School of Engineering, Westlake University, Hangzhou, China 3 vivo Mobile Communication Co., Ltd \* Equal Contribution # Corresponding Author {wuzongliang, xyuan}@westlake.edu.cn, {zhengsiming, pt.jiang}@vivo.com

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
