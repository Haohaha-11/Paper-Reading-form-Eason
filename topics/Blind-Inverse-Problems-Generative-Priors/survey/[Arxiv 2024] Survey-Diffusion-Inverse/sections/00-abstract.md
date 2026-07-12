[← 返回 README](../README.md)

# Abstract

## 📌 预览

这是一篇"用预训练扩散模型解逆问题"的综述。核心贡献是提出两套 taxonomy：一套按**要解的问题**（线性/非线性、盲/非盲、有无噪声、像素/latent、是否文本条件），一套按**求解技术**（四大方法家族）。对本课题（生成先验下的盲逆问题）而言，这篇是"全局地图"，帮我们把所有条件采样器归类，并定位哪些方法处理盲/联合后验。

---

## Abstract

Diffusion models have become increasingly popular for generative modeling due to their ability to generate high-quality samples. This has unlocked exciting new possibilities for solving inverse problems, especially in image restoration and reconstruction, by treating diffusion models as unsupervised priors. This survey provides a comprehensive overview of methods that utilize pre-trained diffusion models to solve inverse problems without requiring further training. We introduce taxonomies to categorize these methods based on both the problems they address and the techniques they employ. We analyze the connections between different approaches, offering insights into their practical implementation and highlighting important considerations. We further discuss specific challenges and potential solutions associated with using latent diffusion models for inverse problems. This work aims to be a valuable resource for those interested in learning about the intersection of diffusion models and inverse problems.

> 💡 **问题动机 (Hao 批注)**: 摘要点出全文的立足点——**用预训练无条件扩散模型当先验，不再针对每个逆问题重新训练**。这跟"直接训练条件扩散/流"（如 SR3、Palette）是对立路线。综述只覆盖前者，因为它对任意 forward 算子 $\mathcal{A}$ 都通用。对本课题而言，这正是我们要的设定：先验 $p(x)$ 固定，$\phi$（算子参数）和 $\sigma$（噪声）在推理时联合估计。
>
> 💡 **与本课题的关系 (Hao 批注)**: 摘要里两次强调 "without requiring further training" 和 "unsupervised priors"。本课题的 gauge-aware 联合后验采样器，本质上就是这类"训练无关"求解器的一个盲扩展——把摘要里的 taxonomy 当作出发点，我们要补的是"联合估计 $(x,\phi,\sigma)$ 且带校准检验 (SBC/coverage/CRPS)"这一维度，而综述本身几乎不谈校准，只谈点估计 vs 后验采样的对立（见 01 节 Recovery types）。
