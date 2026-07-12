[← 返回 README](../README.md)

# 4. Thoughts from the authors

## 📌 预览

作者不给"哪个方法最好"的结论，而是分家族点评：Explicit 家族缺标准 benchmark、缺误差传播理论；Variational 受限于近似分布表达力；CSGM 受限于 ODE 反传算力；Asymptotically Exact 只在无限算力下有保证。对本课题最有用的是第一条——作者明说"可以对任何方法做 dual change（clean↔noisy measurement）造新算法"，说明这类近似有大量未定的设计自由度。

---

## 4 Thoughts from the authors

In the previous section, we presented several works in the space of using diffusion models to solve inverse problems. A natural question that both experts and newcomers to the field might have is, eventually,: “which approach works the best?”. Unfortunately, we cannot provide a conclusive answer to this question within the scope of this survey, but we can share a few thoughts.

> 💡 **证据链批注 (Hao 批注)**: 注意——**这篇综述没有任何自己的实验表格**。它是纯理论/分类综述，"which works best" 明确不答。所以本文的"证据"全在数学统一性（把 DPS、DDRM 等写成同一模板 Eq. 3.1）和分类完备性上，而非 PSNR/LPIPS 数字。这也是作者呼吁"标准 benchmark"的原因。

Thoughts about Explicit Approximations. In this survey we tried to express seemingly very different works, such as DPS and DDRM, under a common mathematical language that contains the explicit approximations made for the measurements score. We observed that all the methods compute an error metric that matches consistency with the measurement and then lift the error back to the image space dimensions to perform the gradient update. Some of the methods used noised versions of the measurements to compute the error while others use the clean measurements. To the best of our knowledge, it is not clear which one works the best and one can derive new approximation algorithms by simply making the dual change to any of the methods that already exist, e.g. one can propose Score-ALD++ by using the noisy measurements to compute the error. By looking at Figure 1, it is also evident that methods propose increasingly more complex “lifting” matrices. Some of these approximations require increased computation, e.g. the Moments Matching method. We strongly believe that the field would benefit from a standardized benchmark for diffusion models and inverse problems to understand better the computational performance trade-offs of different methods. We also believe that under certain distributional assumptions, it should be possible to characterize analytically the propagation of the approximation errors induced by the different methods.

> 💡 **机制拆解 (Hao 批注)**: 这段是全文最凝练的洞察——Explicit 家族的通用骨架是"误差度量 $\mathcal{M}_t$ → 用 lifting 矩阵 $\mathcal{L}_t$ 抬回图像空间 → 除以 guidance 强度 $\mathcal{G}_t$"（即 Eq. 3.1）。两个正交的设计轴：(1) 误差用 clean measurement 还是 noised measurement；(2) lifting 矩阵从 $A^\top$（Score-ALD）到伪逆 $A^\dagger$（ILVR/DDNM）到带协方差的 $(r_t^2 AA^\top+\sigma_y^2 I)^{-1}$（ΠGDM/Moment Matching）越来越复杂。**"越复杂的 lifting = 越贴近真 posterior score 的二阶信息"**，这正是"prior score 与 posterior score 差距"的核心：所有方法差在如何近似那个 intractable 的 $\nabla\log p_t(y|x_t)$。

Thoughts about Variational Methods. Variational Methods try to estimate the parameters of a simpler distribution. The benefit here is that one can employ well-known optimization techniques to better solve the optimization problem at hand. A potential drawback of this approach is that the proposed distribution might not be able to capture the complexity of the real posterior distribution.

> 💡 **与本课题的关系 (Hao 批注)**: 变分家族（RED-Diff、Blind RED-Diff、Score Prior）用一个简单 $q$ 逼近后验。这直接触及本课题的校准关切：**如果 $q$ 表达力不够（如各向同性高斯），后验必然 mis-calibrated**——SBC/coverage 会暴露这一点。综述只说"可能捕捉不到真后验复杂度"，但没给校准诊断工具，这正是本课题要补的空白。

Thoughts about CSGM-type Methods. CSGM-type frameworks can benefit from the plethora of techniques that have been previously developed to solve inverse problems with GANs and other deep generative modeling frameworks. The main issue here is computational since the generative model to be inverted here is the Probability Flow ODE mapping that requires several calls to the diffusion model. Consistency Models [157, 163] and other approaches such as Intermediate Layer Optimization could mitigate this issue.

Thoughts about Asymptotically Exact Methods. Asymptotically Exact Methods, usually based on Monte Carlo, could be useful when sampling from the true posterior is really important. However, the theoretical guarantees of these methods only hold under the setting of infinite computation and it remains to be seen if they can scale to more practical settings.

> 💡 **消融解读 (Hao 批注)**: 四家族的"精度 vs 算力"取舍一目了然：Explicit 最快但引入不可控近似误差；Asymptotically Exact (SMC/MCMC) 理论上采到真后验但要无限 NFE。**本课题若要"可校准的盲后验"，最诚实的落点是 Asymptotically Exact 家族（因为它有 exactness 保证，SBC 才有意义），但必须解决其算力和盲扩展（$\phi$ 也要进 MCMC）问题。** Explicit 家族快但校准无保证，只能作为 proposal/warm-start。

---

## 🔖 Section 总结

### 核心洞察
1. 全文无实验，价值在"统一数学语言 + 分类"，作者主动呼吁社区建 benchmark。
2. Explicit 家族的自由度：clean vs noised measurement × lifting 矩阵复杂度，二者可任意组合造新方法。
3. 四家族本质都在近似同一个 intractable 项 $\nabla\log p_t(y|x_t)$，差别只在近似的保真度与算力。

### 可追问点
- 哪种 lifting 矩阵在盲设定下最稳？综述没答，本课题可实证。
- 变分/Explicit 家族的后验校准如何量化？综述完全没涉及。
