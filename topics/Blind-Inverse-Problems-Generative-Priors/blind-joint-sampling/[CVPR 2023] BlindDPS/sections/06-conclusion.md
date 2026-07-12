[← 返回 README](../README.md)

# 6. Conclusion

## 📌 预览

一段话收尾：BlindDPS 通过为前向算子每个分量各建一个 score function，构造一组反向 SDE 来近似盲逆问题的后验采样，在盲去模糊和湍流成像上取得 SOTA（即使退化和噪声很重）。

---

In this work, we proposed BlindDPS, a framework for solving blind inverse problems by jointly estimating the parameters of the forward measurement operator and the image to be reconstructed. We theoretically show how we can construct a system of reverse SDEs to approximate posterior sampling for blind inverse problems, by using multiple score functions designed to estimate each part of the component. With extensive experiments, we show that Blind-DPS establishes state-of-the-art performance on both blind deblurring and imaging through turbulence, even when the degradation and the measurement noise are heavy.

> 💡 **结论批读（Hao 批注）**: 全文可压成一句话——**"DPS + 每个算子分量各配一条扩散链 + 只用似然残差梯度耦合 = 盲逆问题的联合点估计器"**。
> - **它真正证明的**：这套并行采样能给出**高质量的联合点估计**（图像 + 核/tilt），感知指标 SOTA。
> - **它没证明的（我们的接续点）**：它反复用"posterior sampling"措辞，但从未验证得到的样本是否构成**校准的联合后验**——没有 coverage、没有 SBC、没有 CRPS，没有展示后验宽度。加上 Eq.(16) 的独立先验假设、Theorem 1 的 Jensen 点估计近似、手调 $\alpha/\lambda$ 与硬投影 gauge 固定，其"联合样本"很可能是有偏且过自信的。
> - **对本课题的最终定位**：BlindDPS 是"生成先验 + 参数化盲 + 联合"这条线的**奠基基线**。我们的工作 = 保留"联合采样"骨架，但把它升级成 **gauge-aware、可校准的联合后验**（低维 $\varphi$、显式规范处理、用 SBC/coverage/CRPS 检验），正面回答 BlindDPS 回避的"这些样本可信吗"这一问题。
