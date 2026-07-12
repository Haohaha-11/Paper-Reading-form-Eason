[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览

结论复述三件事：Diffusion EM（E 步扩散采样近似期望对数似然，M 步 MAP 估核 + PnP 核正则）；Fast EM（把 EM 迭代注入扩散过程，只跑一次扩散，反而比经典版更好更快）；以及局限——目前只做**卷积型**反卷积，未来要推广到更一般的盲去模糊，并可借 latent diffusion / diffusion bridge 进一步提速。

---

In this article, we present a novel approach for blind deconvolution based on diffusion models. In particular, we designed Diffusion EM, an algorithm based on the Expectation-Maximization algorithm. This algorithm consists of an E-step, which approximates the expected value of the log-likelihood using a diffusion model, and an M-step, which maximizes this expected log-likelihood with respect to the unknown parameters (the blur kernel). For the Mstep, we introduced a novel kernel regularization based on a Plug & Play denoiser. The diffusion EM algorithm is slow since it requires running a diffusion model several times. We propose an acceleration of the algorithm that directly injects the EM iterations into the diffusion process (leveraging the intermediate diffusion steps as approximate posterior samples). We observed that this Fast EM diffusion model reaches better performance than the original diffusion EM algorithm while being significantly faster. Finally, we demonstrate the efficiency of our approach both quantitatively and visually. We compare our approach to state-ofthe-art methods for blind deconvolution and provide several ablation studies that highlight the performance of our regularization and model and give insights into the behavior of the model. In its current form, our algorithm is limited to deconvolution. Future research will address more general blind deblurring problems [5, 9]. Faster diffusion models such as latent diffusion [8, 40] or diffusion bridges [30] could also benefit our method.

> 💡 **机制复述 + 反直觉点（Hao 批注）**: 结论明确 M 步"maximizes ... with respect to the unknown parameters (the blur kernel)"——再次坐实**参数是点估计**。一个反直觉的核心发现值得记住：**Fast EM 不仅更快，还比原始 Diffusion EM 更好**。通常"加速版"意味着精度妥协，但这里因为把核估计嵌入扩散时间轴（用中间态 $\hat{x}_0(t)$ 当渐进后验样本），核随扩散持续修正，反而避开了经典版"卡在 no-blur 解"的失败模式（见 04 节）。"leveraging intermediate diffusion steps as approximate posterior samples" 是全文最凝练的一句方法论。

> 💡 **局限与本课题接口（Hao 批注）**: 作者自陈局限是"仅限卷积（deconvolution）"，即算子 $H$ 必须是卷积核。未来方向：(1) 更一般盲去模糊（空间变化模糊 [5,9]）；(2) latent diffusion / diffusion bridge 提速。
>
> **对本课题的直接对照**：本文全篇没有触及"退化参数的不确定性量化"。它给的是高质量单核 $\hat{H}$（point estimate），从不给 $p(H|y)$。这恰好定义了它作为对照组的价值——我们的 gauge-aware 联合后验采样要在**同样的 $y=Hx+n$ 设定下**输出 $x,\varphi,\sigma$ 的联合后验，并用 SBC/coverage/CRPS 检验校准。可复用的两点：(a) Eq (34) 的 $\mathcal{L}_{reblur}$ 一致性度量可作为我们 posterior predictive check 的一个组件；(b) PnP 核去噪先验（Figure 4 证明抗噪强）可作为我们低维算子参数先验的候选，但要改造成可采样的（如 score-based kernel prior）而非仅去噪算子。

---

## 🔖 Section 总结

### 核心洞察
1. **方法一句话**：EM + 扩散，E 步采样、M 步点估计核（PnP 核正则）；Fast 版把 EM 折进单次扩散。
2. **反直觉发现**：加速版 Fast EM 反而更好——因中间态当渐进后验样本，核持续修正，避开 no-blur 陷阱。
3. **局限**：仅卷积；无参数不确定性量化。

### 可追问点（本课题）
- 如何把本文的"单核点估计"升级为"核的可校准后验"？→ 需把 M 步 $\arg\max$ 换成对 $H$ 的采样（Langevin/score-based kernel prior），并引入 SBC/coverage/CRPS。
- $\mathcal{L}_{reblur}$ 与 gauge 对称性如何交互？→ 卷积在平移/缩放下有 gauge 自由度，值得在联合后验里显式处理。
