[← 返回 README](../README.md)

# VI. DISCUSSION AND CONCLUSION + Appendix A

## 📌 预览

讨论节把结论收束成三层：(1) 框架与 benchmark 的贡献重述（RTO、TC 是原创；benchmark 特色是「后验 + 加噪先验分数双解析」）；(2) 设计选择的结论（TC 去噪近似最优但受限于二阶分数；RTO 采样修正 MAP 方差低估；Lang 多峰崩）；(3) 总体判断——单峰 OK、线性多峰看设计、非线性多峰全崩，加三条局限与未来方向。Appendix A 证明 $\sigma(t)=t$、单步反向 Euler 时 ODE 均值 = Tweedie 均值。

---

Difusion annealing based inverse problem solvers are an emerging class of algorithms for solving Bayesian inverse problems using a difusion model trained on the prior distribution. In this work, we introduced a general framework, Bayesian Inverse Problem Solvers through Difusion Annealing (BIPSDA), that provides an unified formulation for this class of algorithms. This framework has two key design choices—the denoising distribution approximation and the sampler for the prediction distribution—that can be “mixed and matched” in a flexible manner. The large algorithmic design space of BIPSDA includes the previously introduced DAPS [22] and DifPIR [23] algorithms, which have both achieved state-of-the-art performance on diferent image reconstruction problems. Novel approaches can be unveiled through combinations of ideas from DAPS and DifPIR, as well as the novel techniques proposed here to approximate the denoising distribution and to sample from the prediction distribution. Specifically, the Tweedie Correlated (‘TC’) technique, previously introduced in the context of the hijacking class of difusion-based Bayesian inverse problem solvers [33], is a novel contribution in the difusion annealing context. This approach provides a theoretically-sound, hyperparameter free Gaussian approximation to the denoising distribution by leveraging the second moments of the data distribution and the generalized Tweedie formula [37]. The randomize-thenoptimize (‘RTO’) technique, which was originally developed as a proposal distribution for Markov chain Monte Carlo methods [24]–[26], is also introduced here for the first time to generate fast, approximate samples from the prediction distribution by use of an “of-the-shelf” numerical optimization method.

> 💡 **机制拆解 (Hao 批注)**: 贡献重述，两个原创要点钉死：**TC**（把 hijacking 里的广义 Tweedie 协方差移植到退火语境，零超参、理论自洽，靠二阶矩）和 **RTO**（把 MCMC 的 randomize-then-optimize 借来做预测分布快速采样）。注意措辞——TC 和 RTO 都是「移植 + 重新语境化」而非全新发明，这是诚实的定位。框架的元价值在于「mix and match」暴露了 DAPS/DiffPIR 之外的 7 个新组合。

We also proposed a suite of four benchmark prob lems (inspired by image inpainting, x-ray tomography, and phase retrieval) for the rigorous assessment of the proposed BIPSDA framework, as well as other difusionbased posterior samplers. A key feature of these benchmark problems is that they not only have an analyticallyknown posterior distribution, but also feature an analytical close-form expression for the noisy prior score. This enables analysis of the robustness of algorithms to errors in the score model by examining the idealized scenario of a known-exactly prior score.

> 💡 **机制拆解 (Hao 批注)**: benchmark 的双解析特色再次强调——不只后验可知，**加噪先验分数也闭式**。这是它区别于其他 benchmark 的核心竞争力：能构造「score 零误差」的理想场景，从而把算法固有缺陷单独拎出来。这正是本文能理直气壮说「phase retrieval 全崩不是 score 没学好、是算法结构问题」的底气。

## A. RESULTS ANALYSIS

The results provide insights into both the impact of specific design choices within our framework and the performance of difusion-model-based inverse problem solvers generally. Regarding the choice of the denoising distribution approximation, we note that approaches that use the ‘TC’ variant provide the best performance on the two non-linear inverse problems we tested when the prior score is assumed to be known analytically. However, in our implementation this technique is currently only applicable when the score is known exactly. The training of an auxiliary neural network model to approximate higher-order moments of the data distribution is required in the learned score setting [37]. It is also worth noting that in general, the ‘TU’ variant performs better than the ‘ODE’ variant.

> 💡 **消融解读 (Hao 批注)**: 去噪近似维度的结论：**analytic score 下 TC 在两个非线性问题上最优**（用了二阶矩信息），但 learned score 下用不了（需辅助网络学高阶矩，[37]）。次优结论：**TU 普遍优于 ODE**——因为 Tweedie 闭式估均值比 ODE 反解干净（无离散化误差），且更快（Table 6）。实践建议清晰：learned score 部署选 TU，研究/理想场景可试 TC。

Regarding the sampler for the prediction distribution, the results demonstrate that the MAP estimation based approaches, despite lacking firm theoretical foundations, perform well in recovering the global structure of the posterior distribution. However, they systematically underestimate the variance of each of the posterior modes, which is unsurprising given that the ‘MAP’ variants do not actually sample from the prediction distribution. The ‘RTO’ technique, which is equivalent to exact sampling from the prediction distribution when the likelihood function is linear-Gaussian [43], partially resolves this issue and provides better performance than the ‘MAP’ variants on the stylized x-ray tomography problem and inpainting problems. Finally, the ‘Lang’ variants work well when the posterior is unimodal (low noise regime of the stylized inpainting problem and stylized x-ray tomography problem). However, in the case of multimodal posteriors (as in the high noise regime of the stylized inpainting problem), the ‘Lang’ variants struggle to properly incorporate the measurement information. Further, on the phase retrieval problem, Langevin dynamics without Metropolis correction completely fails, and the incorporation of a Metropolis adjustment and preconditioning was required to obtain competitive performance.

> 💡 **消融解读 (Hao 批注)**: 采样器维度的完整结论表——**MAP**：抓 global 结构（多峰权重）好，但系统性低估峰内方差（因为它只求众数、不采样）；**RTO**：部分修正 MAP 的方差低估，线性高斯下等价精确采样，inpainting/CT 上优于 MAP；**Lang**：单峰下好，多峰下「无法正确吸收测量信息」（撒到低密度区、高估方差），无校正 Langevin 在 phase retrieval 上直接失败。这段就是给实践者的选型指南：**默认用 RTO，慎用 Lang（尤其多峰），MAP 适合只关心均值/多峰权重的场景**。

Overall, the results demonstrate that the BIPSDA framework can provide strong performance on problems with unimodal posterior distributions. Strong performance is also attainable on problems that have multimodal posteriors and linear forward models, although in this setting the performance of the framework is sensitive to the algorithmic design choices made. On the stylized phase retrieval problem, however, all of the BIPSDA algorithms we tested produced inaccurate uncertainty estimates. This is reflective of the extremely challenging nature of the problem, for which even conventional MCMC algorithms that require knowledge of the posterior density struggle to perform well.

> 💡 **机制拆解 (Hao 批注)**: 这是全文的「一句话总判决」，也是本 topic 最该引用的句子——**扩散退火 UQ 能力的三级台阶**：(1) 单峰后验→强；(2) 多峰+线性→可强但敏感于设计选择；(3) 非线性多峰（phase retrieval）→全员给出不准确的 UQ。作者小心地为扩散方法留了台阶：连需要已知后验密度的传统 MCMC 都难搞定 phase retrieval，所以扩散方法失败「情有可原」。但对读者的核心 claim 而言：**这证明扩散不提供无条件的严格 UQ**——它是「条件性可信」，在非线性多峰上失效。这正好支撑「诊断+修复是增量的」：修好了单峰、修好了线性多峰（靠 RTO），但非线性多峰还没修——是一个持续增量的过程，不是一步到位的严格保证。

## B. LIMITATIONS AND FUTURE DIRECTIONS

The proposed benchmark problems provide useful insight on the ability of difusion annealing based inverse problem solvers to provide rigorous estimates of the uncertainty of the posterior distribution. While results are promising, there are a few limitations of the present work that require further investigation.

First, the performance of BIPSDA algorithms, and difusion-annealing approaches in general, strongly depends on both the choice of algorithm-specific hyperparameters and the accuracy of the underlying pretrained difusion model. In our work, we carefully controlled for these efects by fixing some of the hyperparameters (e.g. the same sequence of annealing time steps $\left[ t _ { N _ { A } } , t _ { N _ { A } - 1 } , \cdot \cdot \cdot \ , t _ { 0 } \right]$ for the outer loop of Algorithm 1) and by comparing performances using both learned and analytic noisy prior scores. Furthermore, some of the novel algorithms that we propose within the BIPSDA framework achieve similar or superior performance to state-of-the-art methods, such as DAPS, while drastically reducing the number of hyperparameters that the user must provide. Specifically, the ‘TC’ variant, which our work is the first to explore in the contest of difusionannealing approaches, provides a hyperparameter-free approximation of the denoising distribution, while on the contrary the ‘ODE’ variant used by DAPS is highly sensitive to the choice of the discretization parameters of the probability flow ODE. Additionally, the ‘RTO’ variant that we proposed here for the first time computes approximate samples from the prediction distribution using “of-the-shelf” deterministic optimization algorithms. Conversely, the ‘Lang’ variant used by DAPS requires careful tuning of the time step size and number of time steps in the Langevin dynamics. Nevertheless, while in this work we followed recommendations in the literature when choosing hyperparameters [22], [54], more work is required to understand the full impact of user provided hyperparameter choices (such as the noise annealing schedule) and how to optimize the accuracycomputational cost trade-of.

> 💡 **消融解读 (Hao 批注)**: 局限一 = 超参敏感性。作者反过来把它变成卖点：**新变体（TC 零超参、RTO 用现成优化器）比 DAPS（ODE 对离散化敏感、Lang 需调 step/步数）大幅减少超参**。潜台词：BIPSDA 不只是更全，还更省心。但诚实承认退火调度等超参的完整影响还没摸透。这对复现很重要——DAPS 类方法的「SOTA」可能部分来自精调超参，本文的固定超参对比更公平。

Second, the algorithms in our framework were systematically tested on stylized model problems that enable principled performance analysis but have low dimensionality. In the Supplementary Material we also apply algorithms from the BIPSDA family to imaging prob lems of practical relevance. Proof-of-principle numerical results related to image inpainting (c.f. Fig. S.5 and Table S.1 in the Supplementary Material) demonstrate the computational feasibility of applying algorithms from the BIPSDA family at scale and their ability to produce plausible, diverse samples. However, to systematically evaluate the performance BIPSDA framework in these high-dimensional settings, there remains the need for image reconstruction-relevant, large-scale benchmarks with well-characterized image priors.

> 💡 **机制拆解 (Hao 批注)**: 局限二 = 维度 gap。这是本文最大的软肋——所有严格结论都在 10 维 GM 上得出，真实图像是高维、先验隐式。作者在 Supplementary 做了 image inpainting 的 proof-of-principle（Fig S.5、Table S.1）证明「能 scale、能产出合理多样的样本」，但坦承缺「先验良好刻画的大规模图像 benchmark」来严格评测。对本 topic：这提醒读者，「扩散不给严格 UQ」的结论是在可控低维验证的，外推到高维图像时是「有理由担心」而非「已证明」——这是一个开放的验证空白。

Third, it is also of interest to improve the performance of BIPSDA algorithms on challenging non-linear problems like phase retrieval. Here we first note that in this study there were cases where posterior samples numerically diverged due to large error in the learned score in low probability regions of the prior. This could potentially be addressed through the incorporation of second-order score model derivative information into the score training, which has been shown to improve model accuracy in low-density regions in other problem contexts [55]. Further, even with the score known analytically, the BIPSDA algorithms performed poorly on some of the trials, and additional algorithmic innovations are required to address this. In particular, it is of interest to explore modifications to the proposed ‘RTO’ technique, such as apodizing the noise perturbation in early BIPSDA iterations or metropolizing the RTO sampling, to improve performance on challenging problems like this one.

> 💡 **机制拆解 (Hao 批注)**: 局限三 = phase retrieval 的两条改进路线，直接呼应 Table 5 的发散现象。(1) **learned score 发散** → 用二阶分数导数信息改进 score 训练（[55]，在低密度区更准）；(2) **即便解析 score 也差** → 改进 RTO 本身：早期迭代 apodize 噪声扰动、或给 RTO 加 Metropolis 校正。这两条正是「诊断+修复增量」的教科书示范——先诊断出「发散来自低密度区 score 误差」和「算法结构缺陷」两个不同病因，再分别开药方。读者可以引用这一段说明：修复是一个个具体病因逐步啃下来的增量过程，不存在一劳永逸的严格 UQ 保证。

Finally, while the focus of this work was to provide numerical insight on the ability of decoupled noise annealing type approaches to accurately sample from the posterior distribution, the three benchmark problems that we designed are expected to become a useful tool to the community to develop, refine, and rigorously assess existing and novel approaches (including difusiontype posterior samplers that lie outside of the BIPSDA framework), for solving Bayesian inverse problems with data-driven priors.

> 💡 **机制拆解 (Hao 批注)**: 收尾把 benchmark 定位为「社区共用的严格检验工具」，明确欢迎 BIPSDA 框架之外的扩散采样器来测。这是本文长期价值所在——不是又提一个算法，而是提供一把可复现的「UQ 尺子」。对本 topic：这是「posterior-calibration」支线的地基性工作，任何声称「我的盲逆问题采样器给了可信 UQ」的方法，都应该先过这三个 benchmark。

## Appendix A — RELATIONSHIP BETWEEN TWEEDIE FORMULA AND THE PROBABILITY FLOW ODE

In this appendix we analyze the relationship between the Tweedie’s formula and probability flow ODE based approaches for estimation the mean of the denoising distribution $\mathbb { E } _ { 0 \mid t } [ \mathbf { m } ( 0 ) \mid \mathbf { m } ( t ) ]$

In the Tweedie’s formula based approach, the estimate of the predicted mean, denoted $\mathbf { m } _ { \mathrm { t w e e d i e } } .$ , is computed using the pretrained score model as

$$
\mathbf { m } _ { \mathrm { t w e e d i e } } = \mathbf { m } ( t ) + \sigma ^ { 2 } ( t ) s _ { \theta ^ { * } } ( \mathbf { m } ( t ) , t )
$$

where we have again assumed $\sigma ( 0 ) ~ = ~ 0$ . Since the conditional distribution of $\mathbf { m } ( t )$ given m(0) is Gaussian, Tweedie’s formula is exact and

$$
\mathbb { E } _ { 0 | t } [ \mathbf { m } ( 0 ) \mid \mathbf { m } ( t ) ] = \mathbf { m } ( t ) + \sigma ^ { 2 } ( t ) \nabla _ { \mathbf { m } ( t ) } \log \pi _ { t } ( \mathbf { m } ( t ) )
$$

The only source of error in (14) is therefore score modeling error.

> 💡 **公式批读 (Hao 批注)**: Tweedie 公式在这里是**精确的**（因为 $\pi_{t|0}$ 是高斯），所以 TU 变体估去噪均值的唯一误差来源就是 score 模型误差。这解释了为什么 analytic-score 下 TU 的均值估计几乎无误差——把 $s_{\theta^*}$ 换成真值 $\nabla\log\pi_t$ 就精确了。

In the probability flow ODE based approach, the estimated mean is obtained by solving the probability flow ODE $\left( \mathrm { E q . \ ( 5 ) } \right)$ backwards in time. For concreteness, here we consider the performance of the approach when using the backward Euler method to solve the ODE, which is commonly used in the literature [15], [22], [56]. In particular, we first consider the case with only a single reverse Euler step. In this setting, the estimate of the predicted mean, denoted $\mathbf { m } _ { \mathrm { o d e } } .$ , is given by

$$
\mathbf { m } _ { \mathrm { o d e } } = \mathbf { m } ( t ) + t \sigma ( t ) { \dot { \sigma } } ( t ) s _ { \pmb { \theta } ^ { * } } ( \mathbf { m } ( t ) , t )
$$

As can be seen, the ODE and Tweedie’s formula based approaches both update m(t) in the direction of $s _ { \pmb { \theta } ^ { * } } ( \mathbf { m } ( t ) , t )$ . However, the magnitude of the update difers by a factor of $~ d ~ = ~ t \sigma ( t ) \dot { \sigma } ( t ) / \sigma ^ { 2 } ( t )$ , which in general will not be equal to one, and so therefore $\mathbf { m } _ { \mathrm { { o d e } } } \neq \mathbf { m } _ { \mathrm { { t w e e d i e } } }$ . However, under the particular choice of parameterization $\sigma ( t ) = t , d = 1$ and the updates coincide.

The above analysis shows that the estimate of the mean of the denoising distribution given by the Tweedie’s formula based approach is exact up to error in the score model. Under a particular choice of parameterization and discretization, the probability flow ODE based approach can be made to coincide with the Tweedie’s formula based approach. However, in general the probability flow ODE based approach will introduce additional approximations in the mean estimate.

> 💡 **公式批读 (Hao 批注)**: 这个附录解释了 03 节的伏笔——ODE 更新和 Tweedie 更新方向相同（都沿 $s_{\theta^*}$），但幅度差一个因子 $d=t\sigma(t)\dot\sigma(t)/\sigma^2(t)$。取 $\sigma(t)=t$ 时 $d=1$，单步反向 Euler 的 ODE 均值恰好等于 Tweedie 均值。本文正是用 $\sigma(t)=t$。意义：这说明 ODE 变体（DAPS）的额外误差来自「多步 ODE 离散化」而非「方向错误」——TU 用闭式 Tweedie 一步到位，所以更干净更快。这就从理论上解释了「TU 普遍优于 ODE」这个实验结论。

---

## 🔖 Section 总结

### 核心洞察
1. **UQ 三级台阶**：单峰强 / 线性多峰可强但敏感设计 / 非线性多峰全崩——扩散不给无条件严格 UQ。
2. **设计选型指南**：去噪近似用 TU（learned）或 TC（理想）；采样用 RTO（默认最优）> MAP（低估方差）> Lang（多峰崩）。
3. **三条局限**：超参敏感、低维→高维 gap、phase retrieval 待攻克（二阶分数训练 / 改进 RTO）。
4. **Appendix A**：$\sigma(t)=t$ 时 ODE 单步 = Tweedie，解释「TU>ODE」的理论根源。

### 对本 topic 的可复用洞察
- 「诊断准 ≠ 修复完」的硬证据：Lang 用解析 score 仍在多峰崩，phase retrieval 用解析 score 仍全崩——先验/score 诊断准了，采样器结构问题仍需逐个修，是增量过程。
- benchmark 是「posterior-calibration」支线的地基：任何声称给可信 UQ 的盲逆问题采样器都应先过这三关。

### 可追问点
- Supplementary 的高维 image inpainting 结果如何？→ 仅 proof-of-principle（Fig S.5、Table S.1），证明可 scale + 产出合理多样样本，但无严格 UQ 评测。
