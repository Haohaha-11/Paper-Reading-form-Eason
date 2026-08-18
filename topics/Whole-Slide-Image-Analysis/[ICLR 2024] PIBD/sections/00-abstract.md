[← 返回 README](../README.md)

# Abstract 摘要

## 📌 预览

PIBD 用**信息论**治多模态生存预测的两种冗余：**(1) 模态内冗余（intra-modal）**——WSI 上万 patch、基因上千 pathway 里大量任务无关信息稀释判别性 → **PIB（原型信息瓶颈）** 用不同风险等级的原型近似一堆 instance、筛判别 instance；**(2) 模态间冗余（inter-modal）**——两模态重复信息主导、压制模态特有信息 → **PID（原型信息解耦）** 在联合原型分布引导下把特征解耦成"模态共有 + 模态特有"。

---

## ABSTRACT

Multimodal learning significantly benefits cancer survival prediction, especially the integration of pathological images and genomic data. Despite advantages of multimodal learning for cancer survival prediction, massive redundancy in multimodal data prevents it from extracting discriminative and compact information: (1) An extensive amount of intra-modal task-unrelated information blurs discriminability, especially for gigapixel whole slide images (WSIs) with many patches in pathology and thousands of pathways in genomic data, leading to an "intramodal redundancy" issue. (2) Duplicated information among modalities dominates the representation of multimodal data, which makes modality-specific information prone to being ignored, resulting in an "inter-modal redundancy" issue. To address these, we propose a new framework, Prototypical Information Bottlenecking and Disentangling (PIBD), consisting of Prototypical Information Bottleneck (PIB) module for intra-modal redundancy and Prototypical Information Disentanglement (PID) module for inter-modal redundancy. Specifically, a variant of information bottleneck, PIB, is proposed to model prototypes approximating a bunch of instances for different risk levels, which can be used for selection of discriminative instances within modality. PID module decouples entangled multimodal data into compact distinct components: modality-common and modality-specific knowledge, under the guidance of the joint prototypical distribution. Extensive experiments on five cancer benchmark datasets demonstrated our superiority over other methods.

> 💡 **问题动机**（用信息论重新定义问题）（Hao 批注）：PIBD 与同目录 [MOTCat](../../%5BICCV%202023%5D%20MOTCat/) 是姊妹（同一课题组，都做病理+基因生存预测），但视角不同：MOTCat 用 OT 全局匹配选 patch，PIBD 用**信息瓶颈（IB）+ 解耦**从信息论角度压冗余。这是首次把"信息解耦"引入多模态癌症生存预测。两种冗余的划分很清晰——**intra-modal**（单模态内任务无关信息）、**inter-modal**（跨模态重复信息压制模态特有信息）。

> 💡 **机制拆解**（PIB/PID 各治一种冗余）（Hao 批注）：
> - **PIB（治 intra-modal）**：标准 IB 要建 $p(z|\mathbf{x})$，但 bag 有上万 instance，高维不可算。PIB 的巧解——**不为每个 instance 建分布，而是为每个风险等级建一个原型（高斯分布）$p(\hat z|y)$**，让 instance 靠近同标签原型 → 用与原型的相似度筛判别 instance（丢冗余）。
> - **PID（治 inter-modal）**：把纠缠的多模态特征解耦成"模态共有 C + 模态特有 $S_h, S_g$"，用 PIB 建好的联合原型分布引导 C 的提取，再最小化 C 与 S 的互信息保住模态特有信息（防被共有信息淹没）。
>
> **对压缩研究的直接启示**：PIB 的"信息保留率 Irr"就是一个**显式压缩旋钮**——实验显示病理只留 25-40% instance、基因留 55-70% 就能保持性能（减 60-75% 数据）。这与 ReadySlide 的"retention 率"高度同构，是"信息论指导的 patch 保留"的一个范例。
