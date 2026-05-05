[← 返回 README](../README.md)

# A Appendix

## 📌 预览
Appendix/Supplementary 通常补充实现细节、更多图表和推导，是复现与细节核对的重点。

---

In the appendix, we provide the following materials:

• Comparison with GAN-based methods (referring to Section 4.1 in the main paper).   
• Results of user study (referring to Section 4.1 in the main paper).   
• More real-world visual comparisons under scaling factor $4 \times$ (referring to Section 4.2 in the main paper).   
• Training algorithm of OSEDiff (referring to Section 3.2 in the main paper).

> 💡 **列表批读**: 这组条目通常是在列贡献、设置或发现；建议逐条对应到论文声称解决的 gap。

---

## 🔖 Section 总结

### 核心洞察

1. 本节帮助定位论文贡献边界。
2. 读完后回到 README 的方法对比表中归纳。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Inference time | 0.11s on A100 for 512×512 input |
| Trainable parameters | 8.5M |
| Speedup claim | over 100× faster than StableSR in paper comparison |
