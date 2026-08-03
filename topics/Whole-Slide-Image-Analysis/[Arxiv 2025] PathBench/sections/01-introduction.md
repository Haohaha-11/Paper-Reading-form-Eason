[← 返回 README](../README.md)

# Introduction 引言

## 📌 预览

引言论证 PFM 临床转化的三大障碍——最优架构随癌种/任务变、评测有泄漏/选择偏差、缺标准基准。现有 benchmark 的局限：多用公开数据（有隐蔽重叠）、聚焦单一癌种（前列腺/卵巢）、任务覆盖不全（忽略预后/多模态）。PathBench 用全私有多中心数据 + 全临床谱系任务 + 自动 live leaderboard 补齐。

---

## Introduction

Histopathology serves as the cornerstone of modern oncology. While CNNs and ViTs have demonstrated remarkable success in computational pathology through supervised learning, the field now stands at an inflection point with the rise of pathology foundation models (PFMs). These pre-trained models leverage self-supervised training on massive amounts of pathological images to learn powerful visual representations, or employ contrastive learning to align images, text, and even genetic information. By pretraining on large-scale diverse data, PFMs are revolutionizing WSI analysis through three key advantages: superior generalization across institutions and staining protocols, reduced reliance on expensive manual annotations, and emergent capabilities for multimodal reasoning.

Despite these advances, three critical challenges hinder clinical translation of PFMs. First, optimal architecture and pretraining strategies show significant variability across cancer subtypes and clinical applications. Second, evaluation methodologies may suffer from data leakage or selection bias, particularly when test datasets overlap with pretraining data or share similar demographic characteristics. Third, the absence of standardized benchmarks makes it difficult to validate performance claims across real-world clinical settings.

> 💡 **机制拆解**（三大障碍 → PathBench 三对策）（Hao 批注）：
> 1. **最优模型随癌种/任务变** → PathBench 用 64 任务 × 5 癌种全覆盖，给出"哪个 FM 在哪类任务/器官最优"的细粒度地图，而非单一排名。
> 2. **数据泄漏/选择偏差** → **全私有数据 + 严格排除预训练用过的**（这是最硬核的改进——公开 benchmark 常有隐蔽重叠致虚高）。
> 3. **缺标准基准** → 自动化 leaderboard + 标准评测协议 + PR 提交新模型。
> 三对策直击三障碍。**数据泄漏防控是本文最大贡献**——它让 FM 排名可信。

While existing benchmarking efforts have made valuable contributions, they face notable limitations. Many rely exclusively on public datasets that may not reflect clinical diversity and often contain hidden overlaps with model pretraining data. Others focus narrowly on specific cancer types like prostate or ovarian cancer, limiting their generalizability. Even the most comprehensive studies typically evaluate only a subset of clinically relevant tasks, neglecting critical aspects such as prognosis prediction and other multimodal tasks.

To address these gaps, we present PathBench, the first comprehensive benchmark for PFMs in clinical data across common cancers. PathBench is designed to evaluate PFM performance on a wide range of tasks—from diagnosis to prognosis—using large-scale, multi-center datasets. The benchmark data are obtained solely from private medical institutions, and rigorous protocols are employed to guarantee that none of the data had been exposed to evaluated PFMs during pretraining. We also establish a live leaderboard, hosted on our GitHub repository, to streamline the evaluation of new models and datasets.

> 💡 **定位**（PathBench 在生态里的角色）（Hao 批注）：PathBench 是病理 FM 的"公正裁判 + 活体排行榜"。它与本目录其他论文的关系网：**[LitePath](../%5BArxiv%202026%5D%20Deployment-Friendly-CPath/) 基于 PathBench 选型**（"小模型可行"）；**[EAGLE](../%5BNat%20Commun%202026%5D%20DL-Efficient-Pathology/) 用类似 benchmark 框架**；**[Confounders](../%5BNat%20Biomed%20Eng%202026%5D%20Confounders-Biomarker-Prediction/) 关切的泄漏/混杂正是 PathBench 防控的**。对 ReadySlide：任何"压缩后仍保诊断信息"的声明，都应在 PathBench 式的防泄漏、多任务、多癌种基准上验证，而非单数据集单任务。
