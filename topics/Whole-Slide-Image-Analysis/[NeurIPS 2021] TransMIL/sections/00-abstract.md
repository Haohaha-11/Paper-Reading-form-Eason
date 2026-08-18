[← 返回 README](../README.md)

# Abstract 摘要

## 📌 预览

TransMIL 的核心命题：现有 MIL 都假设 bag 内 instance **i.i.d.（独立同分布）**，忽略了 instance 间的相关性。作者提出 **correlated MIL 框架**（含收敛证明 + 通用三步算法），并用 **Transformer 的 self-attention** 建模 instance 间的两两相关（morphological + spatial），配 Nyström 近似把 $O(n^2)$ 降到 $O(n)$、PPEG 做条件位置编码。在 baseline set 里对应"关键只是 instance contextual interaction"这一竞争解释。

---

## Abstract

Multiple instance learning (MIL) is a powerful tool to solve the weakly supervised classification in whole slide image (WSI) based pathology diagnosis. However, the current MIL methods are usually based on independent and identical distribution hypothesis, thus neglect the correlation among different instances. To address this problem, we proposed a new framework, called correlated MIL, and provided a proof for convergence. Based on this framework, we devised a Transformer based MIL (TransMIL), which explored both morphological and spatial information. The proposed TransMIL can effectively deal with unbalanced/balanced and binary/multiple classification with great visualization and interpretability. We conducted various experiments for three different computational pathology problems and achieved better performance and faster convergence compared with state-of-the-art methods. The test AUC for the binary tumor classification can be up to 93.09% over CAMELYON16 dataset. And the AUC over the cancer subtypes classification can be up to 96.03% and 98.82% over TCGA-NSCLC dataset and TCGA-RCC dataset, respectively. Implementation is available at: https://github.com/szc19990412/TransMIL.

> 💡 **问题动机（i.i.d. 假设的破除）**（Hao 批注）：TransMIL 相对 [DeepSets](../../%5BNeurIPS%202017%5D%20DeepSets/)/ABMIL 的关键一步——**打破 i.i.d. 假设**。DeepSets 的 $\rho(\sum\phi)$ 和 ABMIL 的 $\sum a_n h_n$ 都把 instance 当独立处理（聚合时不看 instance 之间的关系）。但病理学家诊断时会同时看"单个区域的上下文"和"不同区域间的关联"。TransMIL 用 self-attention 建模 instance 两两相关（correlated MIL），这是它相对 baseline set 里前几个方法（MeanPool/ABMIL）的本质增量。

> 💡 **机制拆解（correlated MIL 的三个技术支点）**（Hao 批注）：TransMIL 不只是"把 Transformer 套进 MIL"，有三个支点：
> 1. **理论**：correlated MIL 框架 + 收敛证明（Theorem 1 近似能力、Theorem 2 相关性降低信息熵）——论证"考虑相关性"有理论收益。
> 2. **效率**：标准 self-attention $O(n^2)$ 对 WSI（上万 patch）不可行 → 用 **Nyström 近似**降到 $O(n)$（Eq.9）。这是让 Transformer 能上 WSI 的关键工程。
> 3. **空间**：**PPEG（Pyramid Position Encoding Generator）** 用不同大小卷积核编码多粒度空间位置——因为 WSI patch 数变长，不能用固定长度的绝对位置编码。
>
> **对 baseline set 的定位**：TransMIL = "关键只是 instance contextual interaction / self-attention" 的竞争解释。若新方法超不过 TransMIL，说明增益不只来自"建模实例相关性"。
