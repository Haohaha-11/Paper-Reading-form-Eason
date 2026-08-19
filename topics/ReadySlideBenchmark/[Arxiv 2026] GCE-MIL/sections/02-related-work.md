[← 返回 README](../README.md)

> 💡 **claude 批注｜本节预览**: 本节定位 GCE-MIL 与预测型 MIL、稀疏选择、事后归因和概念解释的差异：它不是新聚合器，而是训练期证据接口。

# 2 Related Work

MIL research for WSI can be read as a sequence of improvements to bag prediction. Pooling design is addressed by ABMIL, CLAM, DSMIL, and TransMIL [Ilse et al., 2018, Lu et al., 2021, Li et al., 2021, Shao et al., 2021]; attention concentration is mitigated by ACMIL, AEM, and ASMIL [Zhang et al., 2024, 2025, Ye et al., 2026]; overfitting is reduced by DTFD-MIL and MHIM-MIL [Zhang et al., 2022, Tang et al., 2023]; spatial context and efficiency are modeled by CAMIL and HDMIL [Fourkioti et al., 2024, Dong et al., 2025]. These works are useful host architectures for GCE-MIL. Their optimization target remains slide-level prediction, however, while evidence sufficiency, necessity, and recoverability are usually evaluated only after training. GCE-MIL is therefore orthogonal: it plugs into these backbones and adds evidence objectives rather than competing as another pooling module.

> 💡 **claude 批注｜与预测型 MIL 的边界**: GCE 把九类 MIL 视为 consumer/host，不替换其聚合结构；本文评估的是语义锚点、noisy-OR、连续门控和 repair 这一 GCE 特定 wrapper 分别与各 host 联合训练。通用“可插拔层”或跨 backbone 兼容性本身不是安全 novelty，更不能据此声称固定 selector transfer。

Sparse selection and post-hoc attribution address explanation more directly but still leave S/N/R under-specified. $L _ { 0 } .$ , Concrete, and Gumbel relaxations provide differentiable gates [Louizos et al., 2018, Maddison et al., 2017, Jang et al., 2017], yet they do not ground selected patches in pathology concepts or model multi-source diagnostic coverage. Gradient saliency, integrated gradients, and occlusion provide post-hoc scores [Simonyan et al., 2013, Sundararajan et al., 2017, Zeiler and Fergus, 2014], but the predictor is already fixed and the thresholded subset is not optimized to be sufficient or necessary. GCE-MIL differs by training selection and prediction jointly, grounding gates with TITAN text anchors, and evaluating the recovered subset through explicit S/N/R interventions.

Subset and concept explanation methods provide useful context for this formulation. L2X and INVASE learn instance- or feature-level rationales [Chen et al., 2018, Yoon et al., 2018], perturbation methods such as Meaningful Perturbations and RISE score regions through input interventions [Fong and Vedaldi, 2017, Petsiuk et al., 2018], and ERASER popularizes sufficiency/necessity-style rationale evaluation in NLP [DeYoung et al., 2020]. Concept methods such as SENN, ProtoPNet, concept bottleneck models, and TCAV connect predictions to human-readable concepts [Alvarez Melis and Jaakkola, 2018, Chen et al., 2019, Koh et al., 2020, Kim et al., 2018]. GCE-MIL draws on these ideas but targets the WSI MIL setting: the object being recovered is a slide-level, sparse patch subset whose continuous selector and discrete evidence are evaluated under the same bag predictor.

> 💡 **claude 批注｜创新边界**: attention 非解释、Sufficiency/Necessity 干预、稀疏连续门控、概念方法和同 fraction post-hoc 对照都有先例。较安全的论文贡献是 WSI MIL 中的 GCE 特定组合：TITAN grounding、noisy-OR 多源覆盖、连续 selector、threshold-plus-repair 与三种 host 注入模式。Table 11 的 joint-training GCE 与 fixed-predictor post-hoc 制度不同，严格 frozen-consumer 的公平 benchmark 仍是空白。

> 💡 **claude 批注｜本节小结**: ReadySlide 不应把“attention 不等于解释”、S/N 干预或 post-hoc 同 fraction 比较写成本文首创；应把 GCE 作为 WSI 特定组合基线。可扩展方向包括 frozen-consumer 公平比较、跨模型证据一致性、多个等价证据集合和临床预算自适应。
