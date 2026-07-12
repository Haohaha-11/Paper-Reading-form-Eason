[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览

引言把"评价缺口"讲透：DIS 算法进展飞快，但基准评价停留在自然图像复原 + 像素精度（PSNR）。而逆问题本质是病态 + 带噪声，一个观测对应**多个物理上都合理的解**，需要的是不确定性量化（UQ）。作者由此命名 **Accuracy Trap**，并抛出中心问题：随机 DIS 能否还原后验 $p(x\mid y)$，以及在没有真后验采样器/密度的真实场景里如何评价这种后验保真度。最后给出三点贡献：受控仿真系统研究、提出 score-KSD、实验证明"高精度 ≠ 高后验一致"。

---

Inverse problems are ubiquitous and fundamental across diverse scientific and engineering applications, including astronomy [8], oceanography [44], medical imaging [37, 6], geophysics [40], and audio signal processing [27, 32], among others. Recently, Diffusion Inverse Solvers (DIS) have emerged as a promising paradigm for solving these inverse problems, leveraging the generative power of pretrained diffusion models to regularize solutions effectively [5, 4, 2, 34]. Despite rapid algorithmic advancements, evaluation and benchmarking efforts lag behind, typically focusing on a set of natural image restoration tasks such as image denoising, deblurring and super-resolution [20, 36, 31]. Furthermore, to evaluate real-world scientific applications with greater structural challenges in forward modeling, where priors and observations are governed by underlying physics, Zheng et al. introduced InverseBench [49], a comprehensive evaluation of existing diffusion inverse solver methods focused on scientific tasks.

> 💡 **问题动机 (Hao 批注)**: 这段先立"算法快、评价慢"的对比。DIS 的核心配方是"预训练扩散模型当先验 + 已知前向模型 = 对后验 $p_\theta(x\mid y)\propto p(y\mid x)p_\theta(x)$ 做推断"。但基准仍是自然图像复原三件套（去噪/去模糊/超分），指标是 PSNR。InverseBench [49] 是本文最重要的参照物——它把评价推向科学任务（物理约束的前向模型），但仍只报精度。本文正是要在 InverseBench 的任务/setup 上补一个"后验保真度"的维度。

However, another gap remains in evaluation objective. Natural image restoration tasks often reward pixel-wise accuracy (e.g., Peak Signal-to-Noise Ratio (PSNR)) from a random reconstruction [20]. In contrast, inverse problems are inherently ill-posed with measurement noise which can leads to multiple physically plausible solutions (Fig. 1), naturally leading to statistical uncertainty quantification [21, 39].

> 💡 **机制拆解 (Hao 批注)**: 这里点破"评价目标"的错位。病态 (ill-posed) + 测量噪声 = 一个 $y$ 对应一整族合理的 $x$。PSNR 奖励的是"某一个随机重建碰巧接近 $x^*$"，而真正该问的是"这一族解被采样得全不全、比例对不对"。这是 UQ 的入口。

Moreover, such uncertainty analysis is especially important and required in engineering and scientific applications, i.e., calibrated uncertainty that preserves all physically valid solutions and enables principled risk quantification [11, 26]. This creates a crucial gap across evaluation of existing DIS works: not only are we ignoring the inherent stochastic nature of DIS, but we are also overlooking the critical role of uncertainty behavior of the sampled distribution requested in scientific applications. This mismatch is evident as shown in Fig. 1, where several DIS methods produce similar accuracy performance in reconstructions for the same task, yet induce markedly different distributional behaviors, reflecting distinct posterior fidelity.

> 💡 **机制拆解 (Hao 批注)**: "calibrated uncertainty that preserves all physically valid solutions"——这句话几乎就是本课题校准目标的英文版。科学应用要求：不确定性要覆盖所有物理有效解，才能做可靠的风险量化。这正是我们用 coverage/SBC 检验的东西。本文的独特论据在最后一句：**同一任务下精度相近的方法，分布行为却截然不同**——这就是"精度不可观测后验行为"的直接观察。

We call this phenomenon the Accuracy Trap. As illustrated in Fig. 1, relying solely on point accuracy metrics (e.g., PSNR) can fundamentally mischaracterize posterior samplers. For instance, an offposterior reconstruction $\hat{x}_2$ may achieve a higher PSNR than a posterior-plausible reconstruction $\hat{x}_3$ simply because $\hat{x}_2$ happens to be closer to the ground-truth $x^*$. Moreover, different DISs can exhibit qualitatively different uncertainty behaviors. Some solvers may produce well-dispersed samples that largely reflect the posterior uncertainty, some may generate a mixture of posterior-plausible and off-posterior samples, and others may collapse to nearly deterministic outputs. Consequently, robust uncertainty quantification (UQ) is not an optional add-on, but a prerequisite for deploying DIS methods in risk-sensitive scientific applications.

> 💡 **机制拆解 (Hao 批注)**: 这里把 DIS 的"失败模式"分成三类，很值得记住，因为后面 Figure 2 的散点图正是按这三类来看的：
> - **过度分散/合理分散**：样本铺开且贴合真后验（理想）。
> - **混合型**：一部分落在后验合理区、一部分是 off-posterior（假样本混入）。
> - **模态坍塌 (mode collapse)**：几乎退化成确定性输出，丢掉不确定性。
> PSNR 对这三类几乎无区分力——因为它只看均值/单点离 $x^*$ 的距离。给本课题的启示：坍塌型采样器在 SBC 里会表现为严重 under-dispersion（coverage 远低于名义值），这正是我们要抓的病。

Since stochastic inverse solvers represent uncertainty through posterior samples, evaluating how well their generated samples capture the target posterior distribution becomes an important aspects of uncertainty quantification. Aligned with this goal, some recent DIS methods make efforts to introduce provable samplers [43, 2, 10, 7], and validate the posterior estimation on controlled simulations where the analytical posterior is known using metrics such as sliced Wasserstein distance [2]. However, evaluating posterior fidelity in realistic inverse problems remains largely unsolved. Existing distributional metrics, such as FID and LPIPS, require samples from both compared distributions and therefore inapplicable to real-world inverse problem without ground truth posterior samples.

> 💡 **机制拆解 (Hao 批注)**: 这段划出方法学的真空地带。已有"可证明采样器"（MCG-Diff [2]、PnPDM [43]、FPS [10] 等）会在仿真里用 sliced-Wasserstein 自证后验估计准确——但这只在解析后验可得时成立。到了真实 MRI/CT，FID/LPIPS 需要"两个分布都能采样"，而真后验采样器根本不存在。**这就是 score-KSD 要填的洞：把"两边采样"的双边指标换成"只需一边样本 + 一个可算 score"的单边指标。**

Encouragingly, UQ has received growing attention in machine learning, through aleatoric uncertainty (AU) and epistemic uncertainty (EU) decomposition [18], single-model uncertainty estimation [17, 3, 30], uncertainty-based distribution shift detection for DIS [25], and controlled statistical benchmarking studies [45]. Yet, to our knowledge, no existing work address the following central question: Can stochastic DIS recover the posterior p(x | y), and how should we evaluate such posterior fidelity without true posterior sampler and density, as in real-world inverse problems?

> 💡 **Q&A 批注记录 (Hao 批注)**:
> - Q: 本文的"中心问题"到底新在哪里？和 [45] 的统计基准有何区别？
> - A: [45]（Zach et al. 的统计基准）仍属"受控设置下比后验采样算法"，需要可得的参考后验；本文的增量是把问题推到**真实逆问题、无真后验采样器/密度**的情形，并给出一个可操作的诊断（score-KSD）。这句斜体的中心问题分两半——前半"DIS 能否还原 $p(x\mid y)$"是描述性研究（Section 4 仿真回答），后半"无真后验时如何评价"是方法贡献（Section 3 + Section 5 回答）。

Contributions. To address this challenge, we provide a systematic study and propose a new metric to evaluate the posterior fidelity for DIS methods:

• We conduct a systematic study of posterior fidelity for a broad range of DIS in controlled simulation settings with known analytical true posterior. Beyond reconstruction accuracy, we analyze how well generated samples capture the target posterior distribution and characterize the distributional behavior of different DIS methods.

• We propose score-based Kernel Stein Discrepancy (score-KSD), a theoretically grounded and ground-truth-free metric for evaluating posterior consistency of DIS methods in inverse problem solving. The proposed metric measures agreement between generated samples and posterior score field induced by the forward model and learned diffusion prior, enabling posterior-aware evaluation even when exact posterior samplers or densities are unavailable.

• Through experiments on both toy models and real-world inverse problems, we demonstrate that score-KSD provides meaningful diagnostics of posterior fidelity beyond reconstruction accuracy, revealing that strong reconstruction performance does not necessarily imply better posterior consistency (Fig. 1), highlighting the importance of distribution-aware evaluation for stochastic inverse solvers.

> 💡 **贡献批读 (Hao 批注)**: 三点贡献恰好对应三种验证证据，读实验节时要按这个链条核对：
> 1. **描述性研究**（贡献 1 → Section 4.1 + Figure 2）：在解析后验可得的 toy 模型里，画散点看谁坍塌、谁漏模态、谁分散得当。
> 2. **方法**（贡献 2 → Section 3）：score-KSD 的构造 = 似然 score（前向模型给）+ 先验 score（扩散模型给）→ 拼出近似后验 score field → 用 KSD 打分。**"ground-truth-free"是全文的技术卖点**。
> 3. **验证**（贡献 3 → Section 4.3 + Section 5）：先在 toy 上证明 score-KSD 的排序和散点可视化一致（内部效度），再在真实 MRI/CT 上证明"精度和 score-KSD 无单调关系"（外部效度）。给本课题的直接可复用点：贡献 2 的构造思路可以原样搬到我们对 $x$-后验的校准诊断上。
