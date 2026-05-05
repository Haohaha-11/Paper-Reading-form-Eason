[← 返回 README](../README.md)

# Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image Super-Resolution

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 CODSR 主线的关系**: CODSR 从 LQ 信息损失、区域差异化先验激活和文本-区域错配三方面补强 controllable one-step diffusion SR，在单步推理下兼顾感知质量与局部保真。

---

Hao Chen Junyang Chen Jinshan Pan Jiangxin Dong† School of Computer Science and Engineering, Nanjing University of Science and Technology https://github.com/Chanson94/CODSR

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
| Core controls | region-adaptive prior activation + text-matching guidance |
| Project | https://github.com/Chanson94/CODSR |
| Benchmarks | RealSR/DrealSR and other real-world SR test sets in paper tables |
