[← 返回 README](../README.md)

# Abstract

## 📌 预览

这篇论文要回答一个在我们课题里极其关键的理论问题：**用扩散先验做后验采样（diffusion-based posterior sampling），得到的分布到底是不是真正的贝叶斯后验？** 答案是否定的——即使 prior score 完全精确、算力无限，DPS 这类算法的输出仍然是**有偏的**。作者用经典的 Feynman–Kac 公式把"偏差"变成一个可以显式写出来的路径期望，从而回答两个此前悬而未决的问题：(1) 偏差究竟把哪些样本采多了、哪些采少了；(2) 低温（强约束）下的数值不稳定从哪来、early-stopping 这类补救措施到底改变了什么分布。

---

Difusion-based posterior samplers use pretrained difusion priors to sample from measurement- or reward-conditioned posteriors, and are widely used for inverse problems. Yet their theoretical behavior remains poorly understood: even with exact prior scores, their outputs are biased, and in low-temperature regimes their discretizations can become unstable. We characterize this bias by introducing a tractable surrogate path connecting the true posterior to a standard Gaussian and comparing it to the sampler’s path. Their density ratio satisfies a parabolic PDE whose reaction term measures the accumulated bias. A Feynman-Kac representation then expresses the Radon-Nikodym correction as an explicit path expectation, identifying which posterior regions are over- or under-sampled.

> 💡 **问题动机（Hao 批注）**: 这段把全文的核心 claim 压缩成一句话——**exact prior scores ≠ unbiased posterior**。注意作者把"偏差"的来源和"数值不稳定"的来源明确分成两件事：
> - **偏差（bias）**：是算法结构性的、连续时间层面就存在的问题，即使离散化误差为零也甩不掉。来源是"用了一条可计算但不精确的插值路径"。
> - **不稳定（instability）**：是离散化（forward-Euler）层面的问题，出现在低温/强约束区。
> 
> 技术主线三步走：① 构造一条 tractable **surrogate path**（代理路径），把真后验 $\mu_y$ 和标准高斯 $\gamma$ 连起来；② 代理路径的边际与算法实际走的路径，其**密度比 $g_t$ 满足一个抛物型 PDE**，PDE 里的 **reaction term（反应项）$c$** 正是偏差的累积速率；③ 对这个 PDE 用 Feynman–Kac 表示，把 Radon–Nikodym 修正权重写成沿路径的期望 $\mathbb{E}[\exp(-\int c\,dt)]$。这就是"偏差从哪一步进入"的答案：**算法为了可计算，直接丢掉了 reaction term $c$**，丢掉的这一项就是全部偏差。

We apply this framework to DPS and STSL, a related sampler. For DPS, the correction is an Ornstein-Uhlenbeck path expectation coupling the data conditional covariance with the reward curvature, revealing where DPS overor under-samples. Next, we reinterpret STSL as an auxiliary drift that steers trajectories toward low-uncertainty regions, flattening the spatially varying part of the DPS reaction term. Finally, we characterize early guidancestopping, a common mitigation for low-temperature instabilities caused by forward-Euler integration of the vector field. Together, these results clarify sampler bias, explain existing correctives, and guide stable variant designs.

> 💡 **机制拆解（Hao 批注）**: 三个应用对应三个结论，务必记牢它们的物理含义：
> - **DPS 的偏差 = 数据条件协方差 $\Sigma_t$ 与 reward 曲率 $D^2 R_y$ 的耦合**。直觉：在数据流形"宽"（后验对 $X_0$ 不确定、$\Sigma_t$ 大）且 reward 在同方向"活跃"（曲率/梯度大）的地方，偏差被放大。这正是**漏模态**的机制来源——多模态先验里，某些模态所在方向恰好是高不确定+高 reward 敏感，就会被系统性采不到（见 Section 3 的 Fig 2：$x_1$ 方向的极端模态几乎消失）。
> - **STSL = 一个把轨迹推向低不确定区的辅助 drift $\nabla U$**，取 $U=\mathrm{tr}(\Sigma_t)$，作用是"抹平" reaction term 的空间变化部分，从而减小偏差。作者证明这个经验成功的 trick 其实等价于在 surrogate path 上加一个修正反应项。
> - **early guidance-stopping = 对 prior 的一个加权 tilt**。这是本文首次给这个工程 trick 一个精确的分布刻画（Theorem 2）。
> 
> **对本课题（生成先验下的盲逆问题）的延伸**：本文只分析"图像条件步"（对 $x$ 的 reward guidance）的偏差。我们的联合后验要同时估计 $x$、算子参数 $\varphi$、噪声 $\sigma$。本文的框架提示：**联合采样的总偏差 = 图像条件步近似引入的 $c_{\text{DPS}}$ + 算子条件步近似引入的额外反应项**，两者会在 Feynman–Kac 路径期望里相乘/叠加。这也解释了为什么必须用 SBC/coverage/CRPS 去外部校准——因为采样器内部没有任何机制保证输出是真后验，理论上就有一个非零的 $\omega(x)$ 权重在扭曲联合分布。（注意：这里只是把本文理论往我们问题上外推，本文本身不涉及算子参数估计。）

---

## 🔖 Section 总结

### 核心洞察
1. **本文最重要的一句话**：diffusion posterior sampling 即便在理想（exact score + 无限算力）条件下也是有偏的，偏差不是数值误差而是算法结构性的。这直接支撑我们项目"用扩散模型 ≠ 得到贝叶斯后验"的论断。
2. **偏差的数学身份**：真后验与算法输出的密度比 $\omega(x)$ 是一个 Feynman–Kac 路径期望 $\mathbb{E}[\exp(-\int_0^T c_{\text{DPS}}\,dt)]$；$\omega\gt 1$ 处欠采样，$\omega\lt 1$ 处过采样。
3. **偏差的物理来源**：算法为可计算性丢弃了 surrogate-path PDE 里的 reaction term，这一项耦合了 $\Sigma_t$（数据流形不确定性）与 reward 曲率。
4. **两类问题分离**：bias（连续层面、结构性）vs instability（离散层面、forward-Euler 在无约束流形上的极限环）。

### 可追问点
- reaction term $c_{\text{DPS}}$ 的正负号如何决定 over/under-sampling？（见 Section 3 Theorem 1 与 Fig 2）
- 若把 image-conditioning 换成 joint (x, φ, σ) conditioning，反应项会多出哪些交叉项？（本文未做，是我们的延伸空间）
