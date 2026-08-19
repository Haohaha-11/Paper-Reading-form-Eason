[← 返回 README](../README.md)

## 📌 批读预览

本节定义 attention ranking 与 output-consistent rationale 的差别，并把 FOCI 定位成冻结模型上的审计读出头。

## 1 Introduction

Whole-slide image (WSI) classification plays a central role in computational pathology, supporting cancer subtyping, grading, and prognosis. The dominant approach extracts patch features with a frozen foundation encoder, aggregates them through a MIL backbone, and trains on slide-level labels alone [1, 2, 3, 4]. This pipeline reaches competitive diagnostic accuracy across several benchmarks [5, 6], but the full-bag prediction remains opaque: it gives a single slide label without surfacing the tiles that support it.

Attention scores are commonly repurposed as post-hoc explanations [7, 8, 9], but high attention does not by itself answer whether a compact subset can recover the model output; in some settings, attention can reflect aggregation or training dynamics rather than an output-consistent rationale [10, 11]. Such rationale highlighting may support downstream review by surfacing candidate regions for inspection, but we do not evaluate reader performance or claim clinical sufficiency.

> 💡 **claude 批注｜问题动机**: attention 高只说明聚合器如何加权，不等价于把其余 tile 删除后还能支持真标签。本文主协议把问题改写成真标签导向的 keep/drop 干预：keep 集要使冻结 consumer 预测 $y$ 且真类概率过阈值，drop 集要失去真类证据；full-bag 输出不为 selector 提供 prediction-fidelity target。这个协议比只和热图或 attention 做相关性比较更严格，但它允许误分类 full bag 的某个子集转而支持真类。

![](../images/da9488cf363e4bb4c7ce30223f81b02124a85379d7bfd9daa684adb7a5bafdaf.jpg)

*Figure 1: Selection headroom for post-hoc rationale highlighting in frozen WSI-MIL: a frozen MIL classifier produces an opaque slide-level prediction, FOCI selects a compact output-consistent tile subset that recovers it, and selection headroom across backbones determines when such compact rationales exist. On TransMIL, relative to its documented CLS-proxy ranking, FOCI reduces MSK by 32–56% across the three benchmarks while leaving the full-bag classifier unchanged.*

> 💡 **claude 批注｜Figure 1 批读**: 图中 full bag 与 selected bag 使用同一个冻结 consumer，但主损失/主 SRP 的 target 是真标签 $y$，不是要求 selected bag 匹配 full-bag predicted class。32–56% 的 MSK 降幅只给 native/proxy→learned gap。若再测 learned→consumer-optimal gap，必须固定 consumer、候选池、目标类、阈值和可行子集空间；tumor/region annotation 只能另报 clinical alignment，不能充当该性能上界。

To address this gap, we study post-hoc rationale highlighting for frozen WSI-MIL classifiers: given a trained classifier, can its slide-level prediction be recovered from a compact, output-consistent tile subset without retraining the backbone? Figure 1 summarizes this audit setting. We instantiate this question with Finding Optimal Contextual Instances (FOCI), a lightweight rationale-readout layer attached to any backbone exposing per-tile features without modifying the existing inference pipeline.

We adapt perturbation-curve evaluation to WSI-MIL through an insertion-style Sequential Reveal Protocol (SRP): tiles are revealed in rank order and the frozen classifier’s confidence is tracked as a function of K. We summarize this curve with AUKC, Minimum Sufficient K (MSK; the smallest K that reaches κ), and Reach (fraction of slides reaching κ). SRP applies to any per-tile ranking, making it a backbone-agnostic operating-point analysis. We further introduce the Selection Headroom Index (SHI) to quantify per-backbone compression of FOCI relative to the frozen backbone’s own ranking, and we triangulate compactness with deletion-based perturbation and selected-only downstream evaluation (§4).

> 💡 **claude 批注｜评估协议拆解**: SRP 给出的是一条置信度—tile 数曲线；MSK 是阈值化的最小前缀，Reach 防止只在容易样本上报很小 K，AUKC 则看完整曲线。三者应共同报告。对 ReadySlide 而言，这比单点 top-K AUC 更可复用：预算既可定义为绝对 K，也可定义为候选池占比，并可显式观察低预算区间。

Across three datasets—TCGA-NSCLC, TCGA-BRCA [12], and PANDA [13]—and seven MIL backbones, FOCI reveals that compact post-hoc rationales are selection-headroom dependent rather than universally available. Soft-aggregation backbones with rationale-compression headroom can be highlighted with a small tile subset, near-minimal attention-pooling baselines enter a selection-saturation regime, and hard-selection backbones can conflict with an external readout. This architecture-dependent pattern is not captured by slide-level AUC alone.

In summary, we present three contributions:

• We formulate post-hoc rationale highlighting as a model-level audit layer for frozen WSI-MIL classifiers: the full-bag classifier remains unchanged, and sufficiency is used strictly in the model-output sense rather than as clinical or pathologist-level diagnostic sufficiency.

• We introduce FOCI, a lightweight rationale-readout module trained with keep/drop model-output sufficiency and exclusion objectives, and evaluate ranked subsets with SRP, MSK, AUKC, Reach, and SHI.

• We show that compact rationale highlighting is architecture-dependent: soft-aggregation backbones can admit compact rationales, selection-saturation regimes leave little room to improve, and hard-selection backbones can conflict with an external selector. Deletion-based perturbation and selected-only downstream evaluation provide complementary checks, with per-backbone SHI values reported in §4.3.

> 💡 **claude 批注｜首创性约束**: 本文已经系统展示：冻结 backbone 后训练轻量 tile selector、通过同一 consumer 计算真标签导向的 keep/drop 损失，并衡量 native/proxy→learned 的压缩空间。因此 ReadySlide 若采用相同范式，贡献应推进到 learned→consumer-optimal combinatorial gap、跨 selector/consumer 矩阵、统一计算/存储预算以及独立的 clinical alignment，而不是重复“外接 selection head 不改诊断器”。

Although we do not evaluate reader-level performance, the resulting candidate rationales provide a compact, reviewable view of when frozen MIL predictions can be localized to small output-consistent subsets.

## 🔖 本节总结

- selection headroom 是“原生排序还有多少可压缩空间”，不是分类 AUC 的同义词。
- SRP 同时需要 MSK、Reach、AUKC；单报 MSK 会受可达样本集合影响。
- ReadySlide 可复用冻结 consumer + true-label reveal，并分别补 learned→consumer-optimal gap、跨 consumer 测试与 clinical/annotation alignment。
