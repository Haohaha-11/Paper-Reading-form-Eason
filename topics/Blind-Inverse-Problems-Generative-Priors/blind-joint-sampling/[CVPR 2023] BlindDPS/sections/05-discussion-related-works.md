[← 返回 README](../README.md)

# 5. Discussion and Related Works

## 📌 预览

本节把 BlindDPS 放进"扩散解逆问题"的谱系里定位（POCS → ALD/SVD → Tweedie 中间估计系，DPS 是最近亲），强调"首次把扩散后验采样推广到盲设定"。最后诚实列出局限：联合优化不如非盲稳、tilt 常估错、多分量导致推理变慢且 score 数线性增长、以及**尚未解决真正的 fully-blind（函数形式未知）**。

---

This work follows the line of endeavors to develop methods that can solve inverse problems through diffusion models. Methods that are based on iterative projection onto convex sets (POCS) were the first to be developed, iterating between the denoising step, and the projection step [10,14,15,52,53]. Methods that attempt to approximate posterior sampling via annealed Langevin dynamics (ALD) [25], and singular value decomposition (SVD) [28] were proposed, with the latter showing particular robustness to noisy measurements.

The trend recently shifted towards leveraging the denoised estimate via Tweedie's formula at the intermediate steps under various names — manifold constrained gradient (MCG) [13], gradient guidance [23], and reconstruction-based method [29]. Diffusion posterior sampling (DPS) [12] is the method that is the most similar to ours, showing that such method is an approximation of the posterior sampling process. However, none of the methods so far considered blind inverse problem, and to the best of our knowledge, we are the first to show that posterior sampling with diffusion scales to blind settings.

> 💡 **谱系定位（Hao 批注）**: 作者把扩散逆问题求解分三代：(1) **POCS** 投影法（ILVR/CCDF/Song 等，去噪+投影交替）；(2) **ALD / SVD** 后验采样近似（SVD 对噪声鲁棒）；(3) **Tweedie 中间估计系**（MCG/gradient guidance/DPS）。BlindDPS 属第 (3) 代、直接建在 DPS 上。**本文的新颖性声明**——首次把扩散后验采样扩到**盲**设定。这个 claim 对我们课题成立且重要：BlindDPS 是"盲 + 扩散 + 联合"的起点基线，我们的 gauge-aware 校准是对它"只给点估计、不管校准"的正面升级。

## Limitations and future directions

As BlindDPS performs joint minimization on multiple factors (e.g. kernel, tilt-map, image), it is typically less robust than the nonblind reconstruction scheme. At times, the solution diverges when the parameters are incorrectly tuned. For imaging through turblence, it is often the case where the tilt map is incorrectly estimated whereas the kernel and the ground truth image are accurately estimated. Furthermore, as we train and use specified diffusion score functions for each of the component, inference speed is delayed, due to the additional forward/backward passes through the newly involved score functions. When the forward functional involves estimating additional parameters, the number of score functions required will scale linearly, not being efficient with complex functional forms. Finally, we note that our method is yet to solve the truly blind case, where we do not know the functional form of the forward mapping. Solving the truly blind case would be an interesting direction of future studies.

> 💡 **Limitation 批读：对我们最有价值的一段（Hao 批注）**: 作者自曝四条短板，条条都是我们课题的机会：
> 1. **联合优化不如非盲稳，参数调不好会发散**——印证联合后验的病态。我们用校准过的联合采样正面回应"稳定性 + 不确定性量化"。
> 2. **tilt 常估错、核和图像却估对**——揭示"高维算子参数最难估准"。这反证了**低维参数化**（我们的路线）的价值：$\varphi$ 维度低，后验更可控、可校准。
> 3. **每加一个分量就多一个 score，推理线性变慢**（去模糊 2 网 180s、湍流 3 网 220s，见 F.2）——高维每分量建扩散先验的架构不 scalable。低维参数用轻量先验可避开。
> 4. **未解决 fully-blind（函数形式未知）**——本文只是"参数化盲"。与我们设定一致（我们也假设参数化前向），所以本文是恰当的直接基线，而非跨设定对比。
> - **一句话**：这四条 Limitation 基本就是我们方法的"卖点清单"的镜像。

> 💡 **Section 小结（Hao 批注）**:
> - **定位**：DPS 的直系盲扩展，属 Tweedie 中间估计一代；首个盲扩散后验采样。
> - **局限**：不稳（发散）、tilt 高维难估、多 score 线性变慢、非 fully-blind。
> - **可追问点**：作者用"less robust"含糊带过联合估计的病态，从未量化后验校准——这是我们 SBC/coverage/CRPS 要填的坑。
