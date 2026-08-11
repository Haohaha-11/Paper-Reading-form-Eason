[← 返回 README](../README.md)

# 3. Method 方法

## 📌 预览

方法：**(3.1) 预备**——MIL 形式化（Eq.1）+ SSM/Mamba（Eq.2-4）；**(3.2) 总览**——location-based K-Means 分组 → IMM 组内去冗余 → CSS 组间聚合 → class token + MHA → MLP 预测；**(3.3) IMM**——组内双向 Mamba + attention mask 丢弃低注意力 $M\times M_r$ 个 instance（Eq.5-7）；**(3.4) CSS**——Max-Pooling 初始 super-feature → cross-attention → MHA → 关联矩阵 Q 桥接局部全局（Eq.8-12）。

---

## 3.1 Preliminaries

MIL (Eq.1): divide WSI into B instances, embed to $f_{ins}^b$, aggregate via $\phi_{aggr}$, MLP predicts $\hat{Y}$. SSM (Eq.2-4): $h(t)=\bar{A}h(t-1)+\bar{B}x(t)$, $y(t)=Ch(t)+Dx(t)$ (D=skip connection). Mamba makes B, C, Δ input-dependent.

## 3.2 Overview

![Fig 3](../images/56c86a112fc0ddf95fca474d9d913f73fb334a2f7d2c890caf4aa569d9694632.jpg)

*Figure 3: GMMamba 架构。location-based clustering 分 G 组 → IMM 预测 attention mask 供 BiMamba 去冗余得紧凑组表示 → CSS 采样判别特征得 super-feature 组表示 → class token 与 super-feature 经 MHA 聚合预测。*

Location-based clustering partitions bag $\{f_{ins}^b\}_{b=1}^B$ into $G$ groups $\{f_{ins}^g = \{f_m^g\}_{m=1}^M\}$ where $G=B/M$. IMM (BiMamba + attention block) captures within-group features and selects informative instances → refined group representation $f_{gr}^g$. CSS extracts/aggregates discriminative tumor features across groups → super-feature group representations $f_{sgr}$. Multi-head attention with class token → $f_{cls}$ → MLP → $\hat{Y}$.

## 3.3 Intra-group Masking Mamba (IMM)

Location-based K-Means clustering (uses instance coordinates) generates groups. Within each group, BiMamba explores relationships; attention block masks $M\times M_r$ low-attention instances ($M_r$ = mask ratio); informative instances processed through BiMamba + attention → group representation:

![Eq 5-7](../images/56c86a112fc0ddf95fca474d9d913f73fb334a2f7d2c890caf4aa569d9694632.jpg)

*Eq. (5-7): BiMamba 建模 → AttentionMask 丢弃低注意力 $M_r$ 比例 instance → BiMamba+Attention 得组表示 $f_{gr}^g$。*

> 💡 **机制拆解（IMM 的自适应 mask = evidence selection）**（Hao 批注）：IMM 是 GMMamba 相对 MambaMIL 的核心增量。流程：组内 BiMamba 建模 → **attention block 预测每个 instance 的注意力分数 → 丢弃分数最低的 $M_r$ 比例 instance（sparse mask）** → 剩余 instance 再过 BiMamba。含义：
> - **在 Mamba 建模中插入 evidence selection**——不是均匀处理所有 instance（MambaMIL 的做法），而是**动态剪枝低价值 instance**，让 Mamba 只扫关键 instance。
> - **好处**：既减冗余计算（少扫 instance）又减干扰（无信息特征不参与）。
> - **vs [ACMIL](../%5BECCV%202024%5D%20ACMIL/) 的 STKIM**：STKIM 遮 Top-K 高注意力（逼模型看更多）；IMM 反过来**丢低注意力**（去冗余保关键）。两者哲学相反——ACMIL 怕过度集中、GMMamba 怕冗余稀释。这个对比很有意思：取决于问题是"注意力过度集中"还是"冗余过多"。
> - **对 CKMIL/ReadySlide**：IMM 是"聚合器内做 evidence selection"的又一实现（类似 [PAMoE](../%5BCVPR%202025%5D%20PAMoE/) 的 expert-choice 丢 patch），且是可学习的自适应 mask。

## 3.4 Cross-Group Super-Feature Sampling (CSS)

![Fig 4](../images/1bc935dab825445dd98cb9c786963f10da906c0ba1ef98e3b89fc2c60eceb521.jpg)

*Figure 4: CSS 模块。Max-Pooling 提初始 super-feature → cross-attention 跨组聚合 → MHA 精炼 → 关联矩阵 Q 桥接局部与全局。*

Max-Pooling extracts salient features → initial super-feature $f_s^{max}$ (Eq.8). Cross-attention with $f_s^{max}$ as query aggregates across groups → $f_s^{ca}$ (Eq.9). MHA refines → $f_s^{mha}$ (Eq.10). Association matrix $Q=\text{Softmax}(f_s^{max}\times \{f_{gr}^g\})$ (Eq.11) bridges local & global: $f_{sgr}^g = f_s^{ca} + Q^g \times f_s^{mha}$ (Eq.12).

> 💡 **机制拆解（CSS 为何要"super-feature"）**（Hao 批注）：CSS 解决"肿瘤散布在不同组、组间相关建模不足"的问题：
> - **Max-Pooling 起点**：先从各组抽最显著特征作初始 super-feature（$f_s^{max}$）——抓每组的"代表"。
> - **cross-attention + MHA**：用 super-feature 作 query 跨组聚合判别信息——建模散布肿瘤区的长程依赖（一个组的肿瘤与另一个组的肿瘤关联）。
> - **关联矩阵 Q 桥接局部全局**（Eq.11-12）：$f_s^{mha}$ 抓全局但可能丢组内局部细节，用 Q（super-feature 与各组表示的关联）把全局信息按关联度加权回各组 → 既全局又保局部。
> - **对比 MambaMIL**：MambaMIL 靠 SR-Mamba 的重排序建模长程；GMMamba 用 CSS 的显式 super-feature 采样 + cross-attention。CSS 更像"先选组代表、再组间 attention"，是层次化的（组内 Mamba + 组间 attention）——与 [RetMIL](../%5BMICCAI%202024%5D%20RetMIL/) 的局部/全局两级思路相通。
