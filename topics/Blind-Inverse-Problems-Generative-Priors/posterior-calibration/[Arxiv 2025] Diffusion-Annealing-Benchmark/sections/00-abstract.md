[← 返回 README](../README.md)

# ABSTRACT

## 📌 预览

这篇论文要解决的核心痛点是：现在大量把「预训练扩散模型当先验」用于贝叶斯逆问题的采样算法，几乎都在「先验解析未知」的图像任务上评测，只能靠 PSNR、样本多样性这类启发式指标判断好坏——**这根本无法验证它们是否给出了严格、可信的不确定性量化（UQ）**。本文的做法是：设计三个「后验解析可知」的 benchmark（inpainting / X-ray CT / phase retrieval 风格），并提出一个统一框架 BIPSDA 把 DAPS、DiffPIR 等方法收编进去，然后用可对照的 ground-truth 后验样本做严格评估。

---

In recent years, the ascendance of difusion modeling as a state-of-the-art generative modeling approach has spurred significant interest in their use as priors in Bayesian inverse problems. However, it is unclear how to optimally integrate a difusion model trained on the prior distribution with a given likelihood function to obtain posterior samples. While algorithms developed for this purpose can produce high-quality, diverse point estimates of the unknown parameters of interest, they are often tested on problems where the prior distribution is analytically unknown, making it dificult to assess their performance in providing rigorous uncertainty quantification. Motivated by this challenge, this work introduces three benchmark problems for evaluating the performance of difusion model based samplers. The benchmark problems, which are inspired by problems in image inpainting, x-ray tomography, and phase retrieval, have a posterior density that is analytically known. In this setting, approximate ground-truth posterior samples can be obtained, enabling principled evaluation of the performance of posterior sampling algorithms. This work also introduces a general framework for difusion model based posterior sampling, Bayesian Inverse Problem Solvers through Difusion Annealing (BIPSDA). This framework unifies several recently proposed difusionmodel-based posterior sampling algorithms and contains novel algorithms that can be realized through flexible combinations of design choices. We tested the performance of a set of BIPSDA algorithms, including previously proposed state-of-the-art approaches, on the proposed benchmark problems. The results provide insight into the strengths and limitations of existing difusion model based posterior samplers, while the benchmark problems provide a testing ground for future algorithmic developments.

> 💡 **问题动机 (Hao 批注)**: 摘要点破了整个「生成先验 + 逆问题」子领域的**评测盲区**。DPS/DAPS/DiffPIR 之类方法都能产出「看起来对、也多样」的点估计，但从来没人能验证它们输出的后验方差、后验多峰权重是否等于真实后验——因为真实后验在自然图像上根本算不出来。作者的破局点是**故意退回到低维 Gaussian mixture 先验**：牺牲「真实感」换取「后验解析可算」，从而第一次能量化地问「扩散采样到底给没给严格 UQ」。这正是本 topic 里「posterior-calibration」这条支线最需要的参照物。

> 💡 **机制拆解 (Hao 批注)**: 摘要藏着两条主线要分清——(1) **Benchmark 贡献**：三个后验可知的模型问题（inpainting 线性、X-ray 泊松非线性、phase retrieval 非线性多峰）；(2) **框架贡献**：BIPSDA 把 DAPS 和 DiffPIR 抽象成「去噪分布近似 + 预测分布采样」两个正交设计维度，DAPS = Lang-ODE、DiffPIR = MAP-TU，二者只是 3×3 网格里的两格，其余 7 格是新算法（含作者主推的 RTO 采样）。

> 💡 **与本 topic 的关系 (Hao 批注)**: 对读者自己的「诊断 + 修复是增量的」主张，这篇是关键背书。它证明了：即便先验分数**解析已知（零建模误差）**，扩散退火采样在 phase retrieval 这种多峰非线性问题上仍然给出错误的 UQ——也就是说误差不只来自「先验学得不准」，而是**算法本身的结构性缺陷**。这支持「盲逆问题里光把先验/前向诊断准还不够，采样器本身要修」的增量式论证。

INDEX TERMS Bayesian inference, difusion models, generative AI, optimization, machine learning, posterior probability, uncertainty quantification

---

## 🔖 Section 总结

### 核心洞察
1. **评测盲区是母题**：现有扩散逆问题采样器缺乏「后验解析可知」的严格 UQ 检验，本文用低维 Gaussian mixture 补上。
2. **两条贡献线**：三个 benchmark（inpainting / X-ray / phase retrieval）+ 统一框架 BIPSDA（含新算法 RTO）。
3. **关键判据**：能否同时抓住后验的 local 结构（均值/逐点方差）与 global 结构（CMD/MMD、多峰权重）。

### 可追问点
- Gaussian mixture 的「后验可知」是否会让结论无法外推到真实高维图像先验？（作者在 Supplementary 用 image inpainting 做了 proof-of-principle，见 06 节讨论）
- 「严格 UQ」在本文具体用哪些指标衡量？→ 见 04 节 Evaluation：均值误差、逐点方差误差、CMD、MMD 四件套。
