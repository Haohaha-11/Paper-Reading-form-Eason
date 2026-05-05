[← 返回 README](../README.md)

# 4.4. Ablation Study

## 📌 预览
Analysis/Ablation 用来判断每个模块是否必要，特别要看控制变量、loss 和轻量化组件是否各自贡献明确。

---

Effect of VAE Compression. Tab. 3 presents the ablation studies of VAE compression process. Channel pruning offers a significant reduction in computational overhead, with only a minor compromise to perceptual reconstruction fidelity. While removing the attention module effectively doubles inference speed, it can somewhat impact reconstruction quality. For lightweight convolution, an alternative approach, employing SnapGen’s [8] strategy of expanding channels in SepConv intermediate layers, yields performance comparable to our proposed solution. However, our method notably exhibit reduced computational overhead and superior inference efficiency. Our efforts culminated in a lightweight VAE that delivers reconstruction quality on par with the teacher, concurrently achieving a $1 0 \times$ increase in inference speed and a $2 2 \times$ reduction in MACs.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

Table 4. Ablation study of removing the text embeddings, time embeddings, and related modules on RealSR.

> 💡 **批注**: 这里涉及条件信号：prompt 是否准确、是否退化感知，会直接影响生成细节与语义一致性。

![Table 4.](../images/fae10dce475dfde160c82a5587e693d9e18d422bfbf8057d503a72b74f797fa3.jpg)
*Table 4.: Table 4. Ablation study of removing the text embeddings, time embeddings, and related modules on RealSR.*

> 💡 **Table 4. 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

Effect of Removing the Text and Time Modules. Tab. 4 presents the ablation study on the elimination of text and time conditions, which demonstrates that our approach achieves a highly favorable trade-off between efficiency and quality. Excising the text embeddings and related context modules yields a 486M parameter, 108G MACs, and 8ms time reduction, while only marginally decreasing the CLIP-IQA score by 0.0046 and the MANIQA score by 0.0048. Subsequent removal of the time modules further reduces parameters by 8M with a negligible impact on final quality.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

---

## 🔖 Section 总结

### 核心洞察

1. 本节帮助定位论文贡献边界。
2. 读完后回到 README 的方法对比表中归纳。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Speedup | up to 5.68× over TSD-SR teacher |
| Parameter reduction | 83% reduction claimed in abstract |
| Inference regime | one-step Real-ISR |
| Compression targets | Diffusion backbone + VAE + prompt/time modules |
