[← 返回 README](../README.md)

# Abstract 摘要

## 📌 预览

**PathBench** 是首个针对病理基础模型（PFM）的**全面、防泄漏、可持续**评测基准：全部用私有多中心数据（严格排除任何 PFM 预训练用过的数据，杜绝泄漏），覆盖从诊断到预后的全临床谱系，配自动 leaderboard。收集 15,888 WSI / 8,549 患者 / 10 医院、64+ 任务。评测 19 个 PFM 发现 **Virchow2 和 H-Optimus-1 综合最优**，且**视觉 FM 仍优于视觉-语言 FM**（临床任务上）。

---

The emergence of pathology foundation models has revolutionized computational histopathology, enabling highly accurate, generalized whole-slide image analysis for improved cancer diagnosis, treatment planning, and prognosis assessment. While these models show remarkable potential across cancer diagnostics and prognostics, their clinical translation faces critical challenges including variability in optimal model across cancer types, potential data leakage in evaluation, and lack of standardized benchmarks. Without rigorous, unbiased evaluation, even the most advanced PFMs risk remaining confined to research settings, delaying their life-saving applications. Existing benchmarking efforts remain limited by narrow cancer-type focus, potential pretraining data overlaps, or incomplete task coverage. We present PathBench, the first comprehensive benchmark addressing these gaps through: multi-center in-house datasets spanning common cancers with rigorous leakage prevention, evaluation across the full clinical spectrum from diagnosis to prognosis, and an automated leaderboard system for continuous model assessment. Our framework incorporates large-scale, clinically diverse data with standardized evaluation protocols, enabling objective comparison of PFMs while reflecting real-world clinical complexity. All evaluation data comes from private medical providers, with strict exclusion of any pretraining usage to avoid data leakage risks. We have collected 15,888 whole-slide images (WSIs) from 8,549 patients across 10 hospitals, encompassing over 64 diagnosis and prognosis tasks. Currently, our evaluation of 19 PFMs shows that Virchow2 and H-Optimus-1 are the most effective models overall. PathBench's dynamic benchmark supports ongoing community contributions through an automated evaluation pipeline.

> 💡 **问题动机**（为什么需要 PathBench）（Hao 批注）：PFM 评测有三大顽疾——**(1) 最优模型随癌种/任务变**（没有普适赢家）；**(2) 数据泄漏**（测试集与预训练数据重叠 → 虚高）；**(3) 无标准基准**（性能声明难验证）。PathBench 的核心价值不是提新模型，而是提供一个**防泄漏的公正裁判**：全私有数据 + 严格排除预训练用过的 → 这是它相对现有 benchmark（多用公开数据、有隐蔽重叠）的关键改进。这与本目录 [Confounders](../../%5BNat%20Biomed%20Eng%202026%5D%20Confounders-Biomarker-Prediction/) 的数据泄漏/混杂关切一脉相承。

> 💡 **对 ReadySlide/本主题的相关性**（Hao 批注）：PathBench 是 [EAGLE](../../%5BNat%20Commun%202026%5D%20DL-Efficient-Pathology/) 和 [LitePath](../../%5BArxiv%202026%5D%20Deployment-Friendly-CPath/) 共同引用的基准（同 HKUST Hao Chen 组）——LitePath 正是基于 PathBench "小模型未必输大模型"的发现选型的。**对 ReadySlide 的 cross-FM transfer 实验**：PathBench 提供了(1) 权威的 FM 排名（Virchow2/H-Optimus-1 是该对标的强 FM）；(2) 防泄漏的评测协议（cross-FM 迁移必须防泄漏）；(3) 64 任务的多样性（压缩方法需在多任务上验证，不能只看一个）。**关键结论"视觉 FM > 视觉-语言 FM"** 也指导 substrate 选择。
