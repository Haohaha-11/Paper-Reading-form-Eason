[← 返回 README](../README.md)

# Abstract

## 📌 预览

摘要一句话讲清全篇：把"扩散模型解逆问题"从**已知退化**推广到**盲退化（模糊核未知）**，用 **EM 算法**联合估计干净图 $x$ 和模糊核 $H$——E 步用扩散模型采样近似期望对数似然，M 步用带 Plug & Play 核去噪先验的半二次分裂（HQS）更新核。核心工程贡献是把 EM 迭代**直接塞进扩散采样过程**，从而只跑一次扩散就完成盲反卷积（Fast Diffusion EM）。

> 💡 **本课题定位（Hao 批注）**: 本文是我们"生成先验下参数化盲逆问题"课题的**点估计对照组**。它同样是"联合估计图像 $x$ + 低维退化参数（模糊核 $H$）"，但退化参数 $H$ 始终以 **arg max（MAP 点估计）** 出现——M 步求的是 $H_{l+1} = \arg\max_H [\dots]$，输出的是单个最优核，而不是核的完整后验 $p(H|y)$。这正是我们要对照的"高质量点估计但无完整后验、无法做 SBC/coverage/CRPS 校准"的典型代表。读的时候重点盯：E 步到底采什么、M 步到底更新哪些量、为何 $H$ 是点估计。

---

Using diffusion models to solve inverse problems is a growing field of research. Current methods assume the degradation to be known and provide impressive results in terms of restoration quality and diversity. In this work, we leverage the efficiency of those models to jointly estimate the restored image and unknown parameters of the degradation model such as blur kernel. In particular, we designed an algorithm based on the well-known Expectation-Minimization (EM) estimation method and diffusion models. Our method alternates between approximating the expected log-likelihood of the inverse problem using samples drawn from a diffusion model and a maximization step to estimate unknown model parameters. For the maximization step, we also introduce a novel blur kernel regularization based on a Plug & Play denoiser. Diffusion models are long to run, thus we provide a fast version of our algorithm. Extensive experiments on blind image deblurring demonstrate the effectiveness of our method when compared to other state-of-the-art approaches. Our code is available at https://github.com/claroche-r/FastDiffusionEM.

> 💡 **机制拆解（Hao 批注）**: 摘要按"问题—方法—加速—验证"四段展开，对应下文的数据流：
> - **问题**：现有扩散逆问题方法假设退化 $H$ 已知（non-blind）。本文放宽到 $H$ 未知（blind），目标是**联合**估计 $x$ 和 $H$。
> - **方法**：EM 框架。E 步"用扩散模型采样近似期望对数似然"——即用非盲扩散在**当前核估计** $H_l$ 下采样后验图像 $x \sim p(x|y,H_l)$；M 步"最大化以估计未知参数"——用这些样本反解模糊核 $H$。
> - **新正则**：M 步引入基于 Plug & Play 去噪器的**核正则**（对模糊核而非图像训练去噪器），后文 Figure 4 证明它在噪声下显著优于 $\ell_1/\ell_2$。
> - **加速**：扩散慢，故把 EM 塞进单次扩散过程（Fast EM），只跑一次扩散。
> - **注意笔误**：摘要写 "Expectation-Minimization"，实际应为 Expectation-**Maximization**（期望最大化），正文均用 Maximization。

> 💡 **和 Blind DPS 的关键区别（Hao 批注）**: 摘要虽未点名，但本文最强对手是 Blind DPS [6]（两个并行扩散分别建模图像和核）。本文用 **EM + 单个确定性核**替代"核也走扩散"。后果是：本文对 $H$ 给的是**点估计**（更一致、更少幻觉、核 MSE 更低），Blind DPS 对 $H$ 走扩散（更锐但更易幻觉、核估计更不稳）。这条差异贯穿全文实验。

> 💡 **Q&A 批注记录**:
> - Q: 为什么是"点估计"而不是完整后验？摘要哪里能看出来？
> - A: "a maximization step to estimate unknown model parameters"——estimate + maximization 就是 MAP 点估计。对比我们课题要的是 $p(H|y)$ 的完整分布（能采样、能算覆盖率）。本文只给 $\hat{H} = \arg\max$，天然无法做 SBC/coverage/CRPS 校准。

---

## 🔖 Section 总结

### 核心洞察
1. **一句话**：扩散模型（E 步采样）+ 快速 EM（M 步点估计核）解盲反卷积；核估计用 Plug & Play 核去噪先验正则。
2. **数据流骨架**：噪声 → 非盲扩散采样 $x \sim p(x|y,H_l)$（E 步）→ HQS + 核去噪反解 $\hat{H}$（M 步）→ 交替直至收敛；Fast 版把交替折叠进单次扩散。
3. **对照价值**：$H$ 是 MAP 点估计（$\arg\max_H$），不是后验分布——这是本课题的点估计对照锚点。

### 可追问点
- E 步用几个样本 $n$？（后文 $n \in \{1,4,16\}$，Stochastic EM 对应 $n=1$）
- M 步的核去噪器如何训练？（DnCNN/FFDNet，在模糊核数据集上训练，噪声等级作额外通道）
- Fast 版为什么能只跑一次扩散？（用中间步 $\hat{x}_0(t)$ 作近似后验样本，见 03 节 Eq 26-33）
