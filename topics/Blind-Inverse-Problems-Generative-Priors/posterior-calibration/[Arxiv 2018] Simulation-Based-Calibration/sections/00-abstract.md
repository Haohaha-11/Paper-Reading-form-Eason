[← 返回 README](../README.md)

# Abstract

## 📌 预览

摘要抛出一个被大多数贝叶斯实践者忽略的问题：**你怎么知道你的后验采样代码是对的？** 模型越复杂、算法越花哨，出错的机会越多，而你从算法里"总能拿到一个结果"，却无从判断结果的好坏。本文提出 **Simulation-Based Calibration (SBC)**：一个只要模型能生成数据、算法能产出后验样本就能套用的通用校验流程，它不但能发现"算得不对/模型写错了"，还能用图形化的 rank histogram 告诉你**错在哪个方向**（偏大、偏小、过宽、过窄）。

---

Abstract. Verifying the correctness of Bayesian computation is challenging. This is especially true for complex models that are common in practice, as these require sophisticated model implementations and algorithms. In this paper we introduce simulation-based calibration (SBC), a general procedure for validating inferences from Bayesian algorithms capable of generating posterior samples. This procedure not only identifies inaccurate computation and inconsistencies in model implementations but also provides graphical summaries that can indicate the nature of the problems that arise. We argue that SBC is a critical part of a robust Bayesian workflow, as well as being a useful tool for those developing computational algorithms and statistical software.

> 💡 **问题动机 (Hao 批注)**: 这段摘要把 SBC 的适用边界一句话钉死了——它验证的是 "Bayesian algorithms **capable of generating posterior samples**"。也就是说 SBC 不关心你的采样器是 MCMC、VI 还是 INLA，只要你能吐出一批后验样本就能查。它检查两件事：(1) **inaccurate computation**（采样器本身有偏，比如 MCMC 没收敛）；(2) **inconsistencies in model implementations**（你写的推断模型和你以为的生成模型不一致，比如先验写错）。注意它**不**检查"你的模型是否符合真实世界数据"——这是后面反复强调的 model-internal 校准边界。

> 💡 **机制拆解 (Hao 批注)**: "graphical summaries that can indicate the nature of the problems" 是 SBC 相对前人方法的最大卖点。前人（Geweke 2004、Cook-Gelman-Rubin 2006）只给一个"通过/不通过"的检验统计量，而 SBC 给出一张 rank histogram，其**形状**本身编码了错误类型：∩ 形=后验过宽（overdispersed），∪ 形=后验过窄（underdispersed），倾斜=后验有偏（biased），两端尖峰=后验样本有自相关。这套"形状→病因"的诊断字典是第 4.2 节的核心。

> 💡 **本课题定位 (Hao 批注)**: 对我们"生成先验下参数化盲逆问题"的主线，SBC 是**校准方法学的地基**。我们要联合估计图像 $x$、低维算子参数 $\varphi$、噪声 $\sigma$，然后用 SBC/coverage/CRPS 检验联合后验是否校准。SBC 在这里扮演的角色是：给定我们**假设的**前向模型 + 先验，先确认联合采样器"在假设内部"是自洽的（rank 均匀）；只有排除了算法错误，rank histogram 上残留的偏离才可能指向 model misspecification。这正是我们区分"算法错误 vs 模型错误"两层结论时必须先跨过的第一道闸门。
