[← 返回 README](../README.md)

# Abstract

## 📌 预览

这是一篇 book chapter 式的综述，作者是 DPS / BlindDPS / DDS 系列的原班人马（Hyungjin Chung、Jeongsol Kim、Jong Chul Ye）。它把"用扩散先验解逆问题"（Diffusion-based Inverse problem Solvers, DIS）的所有主流路线放在同一个贝叶斯后验采样框架下对照，核心矛盾始终是同一句话：**reverse SDE 里出现的是 prior score $\nabla_{x_t}\log p(x_t)$，但我们真正想要的是 posterior score，二者差一个 $\nabla_{x_t}\log p(y|x_t)$，而这一项是 intractable 的**。全篇就是在讲"如何近似 / 绕过这一项"。

---

Using diffusion priors to solve inverse problems in imaging have significantly matured over the years. In this chapter, we review the various different approaches that were proposed over the years. We categorize the approaches into the more classic explicit approximation approaches and others, which include variational inference, sequential monte carlo, and decoupled data consistency. We cover the extension to more challenging situations, including blind cases, high-dimensional data, and problems under data scarcity and distribution mismatch. More recent approaches that aim to leverage multimodal information through texts are covered. Through this chapter, we aim to (i) distill the common mathematical threads that connect these algorithms, (ii) systematically contrast their assumptions and performance trade-offs across representative inverse problems, and (iii) spotlight the open theoretical and practical challenges by clarifying the landscape of diffusion model based inverse problem solvers.

> 💡 **Abstract 主线拆解** (Hao 批注):
> - **一句话定位**: 这是一篇"地图型"综述，不提新方法，而是把 2021–2025 中所有 DIS 算法按"如何处理 likelihood 项 $p(y|x_t)$"分类。它是本课题的第二张全局地图。
> - **分类骨架**: (1) explicit approximation（显式近似 likelihood，代表 DPS）；(2) other（variational inference / SMC / decoupled data consistency）；(3) 复杂扩展（blind、3D 高维、数据稀缺、noisy-data 训练）；(4) text-driven。
> - **和本课题的关系**: 本课题做的是"生成先验下的参数化盲逆问题"，联合估计图像 $x$、算子参数 $\varphi$、噪声 $\sigma$。本文 Sec. 5.1（Blind inverse problems）正是我们的直接上游——BlindDPS / GibbsDDRM / Fast Diffusion EM 都在联合后验 $p(x,\varphi|y)$ 上做文章。**但要警惕**：本文里这些盲方法全部继承了 DPS 的 Jensen 近似，即用 posterior mean $\hat x_{0|t}$ 代替对整条后验的积分——这正是"数据一致性修正 ≠ 严格后验 score"这条主线在盲设置下的放大版，也是本课题 gauge-aware 校准想要修正的漏洞。
> - **要盯的三个 claim**: (i) 能否把所有算法归约成同一条数学线（posterior score 分解）；(ii) fidelity vs perception vs 计算量的三角权衡；(iii) 开放问题（尤其是盲、高维、分布不匹配下的严格性缺失）。
