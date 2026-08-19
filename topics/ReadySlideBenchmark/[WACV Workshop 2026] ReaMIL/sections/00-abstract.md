[← 返回 README](../README.md)

# ReaMIL: Reasoning- and Evidence-Aware Multiple Instance Learning for Whole-Slide Histopathology

Hyun Do Jung<sup>1</sup> Jungwon Choi<sup>2</sup> Hwiyoung Kim<sup>1\*</sup> <sup>1</sup>Yonsei University <sup>2</sup>KAIST

## Abstract

> 💡 **摘要预览（claude 批注）**: ReaMIL 不替换病理 FM，而是在 UNI2-h 冻结特征与 TransMIL 之上增加一个轻量 selector。核心证据链是：选中子集要足以维持真类置信度，补集要失去真类证据，同时选择应稀疏且空间连续；模型是否“会选”再用 MSK 与 AUKC 定量评估。

We introduce ReaMIL (Reasoning- and Evidence-Aware MIL), a multiple instance learning approachfor whole-slide histopathology that adds a light selection head to a strong MIL backbone. The head produces soft per-tile gates and is trained with a budgeted-sufficiency objective: a hinge loss that enforces the true-class probability to be ≥ τ using only the kept evidence, under a sparsity budget on the number ofselected tiles. The budgeted-sufficiency objective yields small, spatially compact evidence sets without sacrificing baseline performance. Across TCGA-NSCLC (LUAD vs. LUSC), TCGA-BRCA (IDC vs. Others), and PANDA, ReaMIL matches or slightly improves baseline AUC and provides quantitative evidence-efficiency diagnostics. On NSCLC, it attains AUC 0.983 with a mean minimal sufficient K (MSK) ≈ 8.2 tiles at τ = 0.90 and AUKC ≈ 0.864, showing that class confidence rises sharply and stabilizes once a small set of tiles is kept. The method requires no extra supervision, integrates seamlessly with standard MIL training, and naturally yields slide-level overlays. We report accuracy alongside MSK, AUKC, and contiguity for rigorous evaluation ofmodel behavior on WSIs.

> 💡 **结果边界（claude 批注）**: “无需额外监督”指无需 tile 标注，不等于 selector 无标签：sufficiency 与 exclusion 都通过 slide 真类概率训练。对 ReadySlide 更重要的是，本文只冻结 UNI2-h 特征，TransMIL 从基线 checkpoint 继续参与联合优化；因此“完全冻结既有 consumer，只训练 selector”仍需单独验证。

## 🔖 Section 总结

- ReaMIL 把 attention heatmap 升级为受干预约束的 evidence selector。
- NSCLC 的代表数字是 AUC 0.983、MSK@0.90 为 8.2、AUKC 0.864。
- 论文占据了“轻量 selector + 最小充分 tile 指标”的直接 novelty，但尚未形成跨 FM selector–consumer–budget 矩阵。
