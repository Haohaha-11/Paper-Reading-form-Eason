[← 返回 README](../README.md)

# D DIFFUSION AND PERCEPTION-DISTORTION TRADE-OFF

## 📌 预览
Appendix/Supplementary 通常补充实现细节、更多图表和推导，是复现与细节核对的重点。

---

In practice, we found that our distilled model is slightly off the perception-distortion frontier of the teacher model, as displayed in Fig. 7. To be specific, the corresponding timestep $t$ shifts a bit but for the same MMSE value the first-stage model and distilled model have very close LPIPS value. This might be caused by the error from large step size dt used in practice and we leave this for future investigation.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

![Figure 6](../images/0caf81efad2b70d9abaea20e2e23b68cc87f19cdd6f9f2e0462edf3fc2e092cb.jpg)
*Figure 6: Figure 6: Metrics evaluation of estimated $\mathbf { x } _ { 1 } ^ { t }$ across different timesteps $t$ . During sampling, at each timestep $t$ , we estimate the final image $\mathbf { x } _ { 1 } ^ { t }$ using the current model prediction $\mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t )$ and state $\mathbf { x } _ { t }$ via $\mathbf { x } _ { 1 } ^ { t } = \mathbf { x } _ { t } + ( \bar { 1 } - t ) \mathbf { v } _ { \theta } ( \mathbf { x } _ { t , \mathrm { L R } } , t )$ . Both MMSE and LPIPS metrics are averaged over 100 sampling processes. We present MMSE instead of PSNR for better visual effect.*

> 💡 **Figure 6 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

![Figure 7](../images/2da8f2a0248dfd7a10b7bff565b1b0bd95edb8493a9b880221f5bf911ffe1c29.jpg)
*Figure 7: Figure 7: Metrics evaluation of estimated $\mathbf { x } _ { 1 } ^ { t }$ across different timesteps $t$ for both teacher model and distilled one-step model. The teacher model is the same as the one in Fig. 6. We present MMSE instead of PSNR for better visual effect.*

> 💡 **Figure 7 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

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
