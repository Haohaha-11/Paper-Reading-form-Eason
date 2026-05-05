[← 返回 README](../README.md)

# H FAILURE CASE

## 📌 预览
Appendix/Supplementary 通常补充实现细节、更多图表和推导，是复现与细节核对的重点。

---

We show visualization of extreme t (boundary t and out of distribution t) in Fig. 10. Results of t ranges from [0,1] do not show failure case, while (ill-defined) OOD t, especially $t < 0$ fails.

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐，而不只是匹配终点。

![Figure 10](../images/6709a5fdafe0effc760b394191ccb1d6d0eb52f319bca2636a7fb79012bb5a73.jpg)
*Figure 10: Figure 10: Visual results of boundary $t$ (0 and 1) and out of distribution $t$ (-0.5 and 2*

> 💡 **Figure 10 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

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
