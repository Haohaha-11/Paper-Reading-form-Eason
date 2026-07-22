[← 返回 README](../README.md)

# I. INTRODUCTION

## 📌 预览

引言把逻辑链搭得很清楚：逆问题常常 ill-posed → 贝叶斯框架靠先验补信息、同时给 UQ → 但好先验难手工设计 → 扩散模型是极强的数据驱动先验 → 但「扩散先验 + 似然」的最优融合方式没有共识，且现有方法在「先验未知」的图像任务上评测，无法验证 global 后验结构。于是本文给出 BIPSDA 统一框架（含新的 RTO 技术）+ Gaussian mixture benchmark（可解耦「算法误差」与「先验建模误差」）+ 四指标评测。

---

Inverse problems, which aim to estimate an unknown parameter of interest from observed data (measurements), provide a principled way to integrate data with existing scientific knowledge [1], [2]. However, many important inverse problems throughout the sciences are ill-posed [3], [4]—the measurements alone do not contain enough information to uniquely and stably estimate the parameter of interest [5]. In the Bayesian inverse problem framework, this dificulty is addressed through the integration of prior knowledge regarding the parameter of interest, which enables both point estimation of the unknown parameter and rigorous uncertainty quantification [6], [7]. This can facilitate risk-informed analysis and decision making, which is particularly important in the context of safety-critical systems (e.g., medical imaging [8] or earthquake detection [9] systems), where accurate risk assessments can save lives.

> 💡 **问题动机 (Hao 批注)**: 这段确立了「为什么要做严格 UQ」的价值锚点：safety-critical 系统（医学成像、地震检测）里，错误的风险估计会致命。这不是玩具动机——它直接决定了后面为什么不满足于「点估计好看」，而要死磕后验方差、多峰权重是否准。贝叶斯框架的卖点正是「先验补信息 + 顺带给 UQ」，本文要检验的就是扩散先验版本能否兑现这第二个承诺。

The Bayesian framework requires knowledge of the prior probability distribution of the unknown parameter.

However, for many problems of interest (e.g., image reconstruction) the distribution of the unknown parameters has a complex structure that is not easily captured by hand-crafted prior distributions (e.g., Gaussian, total variation, or sparsity-inducing priors). Using an inaccurate prior can lead to biased point estimates of the unknown parameter and incorrect predictions of the parameter uncertainty.

> 💡 **机制拆解 (Hao 批注)**: 这里点出经典先验（高斯、TV、稀疏）的根本局限：**先验不准会同时污染点估计和 UQ**。这就是扩散模型进场的理由——用数据学出复杂高维分布的结构。但注意作者的措辞很克制：换成扩散先验只是解决了「先验表达力」问题，并没有自动解决「先验 + 似然如何融合」问题，后者才是本文的战场。

The success of generative modeling [10], [11] in capturing the structure of complex high-dimensional probability distributions in recent years has spurred considerable interest in the use of generative modeling to overcome this limitation [12]. In particular, over the last four years the rise to prominence of difusion modeling [13]–[15] as a state-of-the-art generative modeling approach has been coupled with significant interest in their use as prior distributions in Bayesian inference. Here a common strategy is to obtain posterior samples by pretraining a difusion model on samples from the prior distribution and integrating the pretrained model with the likelihood function (which is assumed to be known analytically) at inference time [16]–[21]. This general strategy has the advantage of enabling the same difusion model to be used in conjunction with many diferent data acquisition designs without retraining, and has shown considerable promise. However, there is no consensus on how to optimally integrate the difusion model and the likelihood function. While several algorithms have been proposed for this purpose, test-bed problems with no analyticallyknown ground-truth prior are often used to illustrate the performance of such methods. In this setting, the performance of the algorithms is often assessed using sample quality metrics like peak signal-to-noise ratio, as well as heuristic arguments regarding the sample diversity. This makes it dificult to assess the accuracy of these methods in capturing the global structure of the posterior.

> 💡 **机制拆解 (Hao 批注)**: 这段是全文的「痛点定型」。核心卖点是**先验与似然解耦**：一个扩散模型训练一次，就能配合任意前向/采集设计（不同 mask、不同噪声）而无需重训——这正是「盲/参数化逆问题」里最诱人的性质。但代价是「怎么把预训练分数和似然最优融合」没有定论，而评测又都在 PSNR + 多样性上打转。作者要做的就是把评测从「感知质量」拉回「后验保真度（global structure）」。

In this paper, we provide a general framework, Bayesian Inverse Problem Solvers through Difusion Annealing (BIPSDA), which—by abstracting the algorith mic components of the Difusion Annealing Posterior Sampling (DAPS) method described in [22]—provides an unified formulation for developing and analyzing annealing-based solvers for Bayesian inverse problems with priors implicitly defined by a difusion model. The framework generalizes and extends two recently proposed algorithms, DAPS [22] and DifPIR [23], for solving Bayesian inverse problems with difusion models. These two algorithms have achieved strong performance on a number of canonical imaging reconstruction problems, including non-linear problems such as phase retrieval that were previously considered too dificult for difusionmodel-based solvers [22]. In the unified framework, the DAPS and DifPIR algorithms can be recovered through specific design choices, while new algorithms can be unveiled through exploration of the rich algorithmic design space. An original contribution of our work is the use of randomize-then-optimize (RTO) techniques, originally proposed in [24]–[26] for approximate sampling from posterior distributions, to solve a sampling subproblem that arises in the BIPSDA framework. Here the RTO technique transforms this subproblem into an optimization problem (as in DifPIR) that can be solved with “of-the-shelf” deterministic numerical optimization methods, while still accurately accounting for measurement noise statistics (as in the DAPS method).

> 💡 **机制拆解 (Hao 批注)**: 框架贡献的核心叙事——BIPSDA 是对 DAPS「解耦噪声退火」骨架的抽象化。关键设计哲学：把一个采样迭代拆成两个正交选择（去噪分布近似 + 预测分布采样），DAPS 与 DiffPIR 各占一格。而 **RTO 是作者的原创技术卖点**：它兼取两家之长——像 DiffPIR 一样把采样变成可用现成优化器求解的优化问题（快），又像 DAPS 一样精确考虑测量噪声统计（准）。一句话记忆：RTO = 先对测量和均值加噪、再求 MAP，这样解优化问题就等价于从预测分布采样（线性高斯下精确）。

We systematically evaluated the performance of algorithms in the proposed framework, including the DAPS and DifPIR algorithms and the novel RTO-based algo rithms, on a set of model problems. Each model problem uses a Gaussian mixture prior. This choice of prior has two key advantages. First, under this choice the posterior density is known and approximate ground-truth posterior samples can be obtained, enabling rigorous analysis of the performance of the BIPSDA algorithms. Second, under this choice the noisy prior scores, a key component of difusion models that is learned from data, can be formed analytically. This allows us to decompose the error in the posterior sampler into two components: error inherent to the algorithm, and error due to incorrect modeling of the prior distribution. For the likelihood functions, we considered benchmark problems inspired by classic image restoration/reconstruction problems: simple linear inpainting problems (in both low and high noise regimes), as well as nonlinear x-ray tomography and phase retrieval based problems.

> 💡 **公式批读 (Hao 批注)**: 这是全文最聪明的实验设计。选 Gaussian mixture 先验有两个「作弊」优势：(1) 后验密度已知 → 能拿到 ground-truth 后验样本做对照；(2) **加噪后的先验分数 $\nabla \log \pi_t$ 有解析式**。第二点极其关键：它让作者能把误差**精确解耦**为「算法固有误差（analytic score）」和「先验建模误差（learned score）」两部分。所以后面每张表都并列 Analytic Score / Learned Score 两栏——这正是能下「问题出在算法而非先验」这种强结论的方法论基础。

> 💡 **Q&A 批注记录 (Hao 批注)**:
> - Q: 为什么非要能解析算出「加噪先验分数」？普通 benchmark 不也能给 ground-truth 后验样本吗？
> - A: 因为普通 benchmark 只能对照「最终采样质量」，无法回答「误差到底是先验没学好、还是采样算法本身有病」。Gaussian mixture 的 $\pi_t$ 仍是 Gaussian mixture，其分数闭式可得，于是可以把学到的 score 换成真值，观察算法在「零先验误差」下的表现——这是本文能验证「扩散不给严格 UQ 是结构性问题」的唯一途径。

In each experiment, the performance of the BIPSDA algorithms was evaluated by comparing the samples produced by each algorithm with the approximate ground truth samples using four diferent metrics: the central moment discrepancy (CMD) [27], maximum mean discrepancy (MMD) [28], and the errors in both the predicted posterior mean and pointwise variance. This enables principled evaluation of the ability of various BIPSDA algorithms to capture both the local and global posterior structure.

> 💡 **机制拆解 (Hao 批注)**: 四指标分工要记牢：**均值误差 + 逐点方差误差**抓 local（一阶、二阶矩是否对）；**CMD + MMD** 抓 global（整个分布形状、多峰结构是否对）。一个方法可能均值很准但方差系统性偏低（后面 MAP 变体正是如此），只看 PSNR 完全暴露不出来——这正是本文相对旧评测的方法论升级。

The remainder of this paper is organized as follows. In Section II, we provide relevant background on Bayesian inverse problems and difusion models. In Section III we introduce the proposed BIPSDA framework, discuss its relationship to previously proposed algorithms, and provide examples of algorithms that can be realized within the framework. Section IV provides details regarding the numerical studies, while results are shown in Section V. A discussion of the results and the conclusion is given in Section VI.

---

## 🔖 Section 总结

### 核心洞察
1. **动机三段论**：ill-posed → 贝叶斯先验补信息且给 UQ → 扩散先验表达力强但「先验+似然融合」无共识、评测又只看感知质量。
2. **框架卖点**：BIPSDA 把 DAPS/DiffPIR 抽象为两个正交设计维度；RTO 是原创，兼取「快（优化求解）」与「准（精确考虑噪声）」。
3. **实验设计卖点**：Gaussian mixture 先验让「后验可知」且「加噪分数解析可得」，从而能解耦算法误差 vs 先验建模误差。

### 可追问点
- RTO 相比 MAP 到底在什么条件下等价于精确采样？→ 03 节：线性高斯似然下可证明为精确采样。
- 四指标各自敏感于什么？→ 均值/方差=local，CMD/MMD=global。
