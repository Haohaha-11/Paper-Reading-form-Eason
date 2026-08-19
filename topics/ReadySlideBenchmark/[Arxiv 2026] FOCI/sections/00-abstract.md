[← 返回 README](../README.md)

# Are Compact Rationales Free? Measuring Tile Selection Headroom in Frozen WSI-MIL

Hyun Do Jung<sup>1</sup>, Jungwon Choi<sup>2</sup>, Soojung Choi<sup>3</sup>, Yujin Oh<sup>†,4</sup>, and Hwiyoung Kim<sup>†,5</sup>

<sup>1</sup>Department of Artificial Intelligence, Yonsei University, Seoul, South Korea <sup>2</sup>Kim Jaechul Graduate School of AI, KAIST, Daejeon, South Korea

<sup>3</sup>Department of Integrative Medicine, College of Medicine, Yonsei University, Seoul, South Korea <sup>4</sup>Department of Biomedical Systems Informatics, College of Medicine, Yonsei University, Seoul, South Korea <sup>5</sup>H-Data Strategy Center, Hallym University Chuncheon Sacred Heart Hospital, Chuncheon, South Korea <sup>†</sup>Co-corresponding authors: yujinoh@yuhs.ac, hykim@hallym.or.kr

## Abstract

## 📌 批读预览

本节把论文的问题、读出头、评估协议和主要结论压缩在一起。阅读时要把“模型输出充分”与“临床诊断充分”严格分开。

Whole-slide image (WSI) multiple instance learning (MIL) classifiers can achieve strong slide-level AUC while leaving the full-bag prediction opaque. Attention scores are widely reused as post-hoc explanations, but high attention can reflect aggregation preference rather than a compact, model-sufficient rationale. We study post-hoc rationale highlighting for frozen WSI-MIL: given a trained classifier, can its slide-level prediction be recovered from a compact, output-consistent tile subset without retraining the backbone? We instantiate this question with Finding Optimal Contextual Instances (FOCI), a lightweight rationale-readout layer over a frozen MIL backbone. FOCI is trained with model-output sufficiency and exclusion objectives over keep/drop tile subsets, evaluated with an insertion-style Sequential Reveal Protocol (SRP) adapted to WSI-MIL, and summarized by the Selection Headroom Index (SHI). Across three WSI benchmarks and seven MIL backbones, FOCI reveals that compact rationales are selection-headroom dependent rather than universally available: transformer and multi-branch attention aggregators can admit compact rationales, near-minimal attention-pooling baselines enter a selection-saturation regime, and hard-selection backbones can conflict with an external readout. For TransMIL, relative to its documented CLS-proxy ranking, FOCI reduces the Minimum Sufficient K (MSK) tile count by 32–56% across the three benchmarks, while ACMIL+FOCI attains the highest mean SHI (+0.465). Deletion-based perturbation and selected-only downstream evaluation provide complementary checks. These results position FOCI as a model-level interpretability and audit layer: selected tiles are not claims of clinical or pathologist-level diagnostic sufficiency, but candidate rationales that offer a compact, reviewable view of when a frozen MIL prediction can be localized to a small output-consistent subset.

> 💡 **claude 批注｜摘要主张的逻辑边界**: FOCI 不改变主分类器，但主 selector 训练并不是蒸馏 frozen full-bag prediction：keep CE/hinge 与 drop exclusion 以真标签 $y$ 为目标，主 SRP/MSK 也要求子集预测为 $y$ 且达到阈值；full-bag 输出只是无 selector 梯度的保持监控。因此误分类 slide 仍可能找到支持真类的子集。Appendix N 才在评估时跟踪 predicted class $\hat y$，训练仍使用 $y$。这些 tile 也只说明 consumer-relative、true-label-directed sufficiency，不等于病理诊断充分。

## 🔖 本节总结

- 论文对象：冻结的 WSI-MIL，而非端到端重训模型。
- 核心变量：MSK、Reach、AUKC 与相对基线排序的 SHI。
- 关键结论：compact rationale 是否存在取决于 backbone 的 selection headroom；正 SHI 不是普遍现象。
