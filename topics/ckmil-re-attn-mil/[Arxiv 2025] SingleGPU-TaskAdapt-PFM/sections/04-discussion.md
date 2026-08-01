[← 返回 README](../README.md)

# 04 — Discussion & Conclusion

## 5 Clinical Impact

> **原文**:

Because LUAD harbors a large number of drug-targetable mutations, molecular diagnostics are especially valuable for guiding its targeted therapies [34, 35]. LUAD was the first tumor type to develop official guidelines recommending universal EGFR and ALK testing [36], which were subsequently expanded in 2018 to include additional actionable genes when sufficient tissue is available [37]. One of the principal challenges remains obtaining adequate DNA from small core biopsies to perform this extensive molecular workup [38]. Although this paradigm has been embraced in LUAD, other malignancies such as BLCA have been sequenced far less frequently outside major centers despite the availability of FGFR3-directed therapies [39]. The ability to predict actionable mutations directly from H&E-stained WSIs promises to streamline and economize current diagnostic workflows. By triaging or potentially obviating the need for costly and tissue-requiring molecular assays, TAPFM can help overcome the infrastructure and financial barriers that preclude precision oncology in resource-limited settings. Even when resources are available, TAPFM offers inference that can be performed as soon as the digital image is scanned. Therefore, clinicians can receive molecularly informed results within hours instead of days/weeks. This study demonstrates that a single model (TAPFM) can simultaneously detect multiple targetable mutations in LUAD, and that its performance generalizes from the originating institution to an independent TCGA cohort.

> 💡 **临床叙事**: Hao 批注 — 临床影响的论证线：分子检测耗时（days/weeks）且需要足够组织（small core biopsies 常不足）→ H&E WSI 是标准流程中必然产生的数字资产→ TAPFM 可在 hours 内从 H&E 预测多种可操作突变→ 节省时间 + 节省组织 + 扩展精准肿瘤学到资源有限地区。这个叙事合理但有前提：需要验证 TAPFM 的预测性能足以替代（而非辅助）分子检测——目前 AUC ~0.85-0.90 的级别更适合 triaging（筛查高风险患者去做确定性分子检测），而非 obviating（完全替代）。

> 💡 **BLCA 的特殊意义**: Hao 批注 — 作者指出 BLCA 在主要中心以外很少进行测序，尽管有 FGFR3 靶向疗法可用。这凸显了 H&E-based 突变预测在降低精准肿瘤学门槛方面的价值——不仅仅是"更快"，而是"让本不会接受分子检测的患者获得靶向治疗机会"。

## 6 Conclusion

> **原文**:

This work introduces TAPFM, a novel approach for adapting PFMs to specific clinical tasks by leveraging ViT's attention mechanism for MIL aggregation and a detached dual-gradient approach for updating PFM parameters on a single GPU. TAPFM bridges the gap between self-supervised pretraining and supervised downstream adaptation in computational pathology, enabling more effective use of PFMs for clinical applications. The experimental results establish its effectiveness for clinically relevant mutation prediction tasks for BLCA (FGFR3) and LUAD (EGFR, KRAS, MET, ALK) patients while maintaining computational efficiency. Notably, TAPFM successfully tackles the challenging task of simultaneous prediction of four actionable mutations in LUAD patients, maintaining reasonable performance even for rare mutations like MET and ALK. Despite promising results, certain limitations of this work can be addressed in future studies. The approach shows potential for extension to additional clinical endpoints including survival analysis, recurrence prediction, and treatment response estimation. Investigations into which specific transformer layers benefit most from task adaptation could optimize the approach by selectively updating only those parameters. Additional external validation across multi-institutional cohorts with diverse scanning protocols and expanded biomarker panels would strengthen the clinical utility of the proposed approach. Implementations that scale to distributed training across multiple GPUs to increase the number of tiles processed per WSI during training may enhance TAPFM's generalization performance.

> 💡 **作者自述局限性**: Hao 批注 — 作者明确指出的局限：(1) 尚未扩展到生存分析等其他临床终点；(2) 尚未探索选择性更新哪些 transformer 层（而非全量微调）；(3) 需要多机构外部验证；(4) 多 GPU 扩展可增加 tile 处理量。这些都是合理的后续方向，但我觉得缺少一个关键局限的认识：**TAPFM 的 detach 双图机制本质上是交替优化而非真正的联合优化**——有可能收敛到比联合优化更差的局部最优，因为 PFM 和 aggregator 的参数缺乏直接的梯度通信。

> 💡 **"哪层受益最大"问题的深层含义**: Hao 批注 — PFM 的不同 transformer 层捕获不同抽象层次的特征（低层：纹理/边缘，中层：组织结构，高层：语义概念）。某些 clinical tasks 可能只需要调整特定层——例如突变预测可能更多依赖高层语义（细胞异型性），而分级可能更多依赖中层结构（腺体形态）。识别这一点的实际价值是：选择性微调可以进一步减少计算和显存开销，让更多 tiles 被处理。

> 💡 **对我们的启示**: Hao 批注 — TAPFM 验证了一个重要原则：在 PFM 时代，feature quality > aggregation complexity。TAPFM 用一个简单的线性头 + 无参数注意力聚合就超越了复杂的 MIL 方法——前提是 PFM 特征被适当地微调。这对我们 ReadySlide 的启示是：compression 方法的设计也应该遵循这个原则——与其设计复杂的 adaptive compression policy，不如确保保留的 patches 覆盖高质量的 PFM 特征。我们当前的 importance-retention 框架符合这一思路：用简单的 CHIEF importance 排序 + retention ratio，已经展示了接近 oracle 的性能——remaining gap 可能在"supervised action-value predictor"而非更复杂的 compression 机制。

> 💡 **多 GPU 扩展 vs 单 GPU 效率**: Hao 批注 — 作者提到多 GPU 扩展的未来方向，但 TAPFM 的核心卖点就是"单 GPU 可行"。多 GPU 扩展会使其失去区分度（因为已经有其他方法做多 GPU 端到端微调）。更值得做的方向可能是：在单 GPU 约束下，通过更好的 tile 选择策略（而非随机采样）来提升训练效果——例如基于预训练的 importance score 做重要性采样，或 curriculum learning 逐步增加 tile 数量。这与我们的 allocator learning 有相似之处。

## References (部分关键引用)

- [2] Ilse et al. (2018) — ABMIL
- [5] Chen et al. (2024) — UNI
- [8] Xu et al. (2024) — Prov-GigaPath
- [9] Bioptimus (2024) — H-Optimus-0
- [10] Lu et al. (2021) — CLAM
- [13] Campanella et al. (2025) — PFM clinical benchmark
- [24] Li et al. (2021) — DSMIL
- [25] Schirris et al. (2022) — VarMIL
- [28] Chen et al. (2024) — MIL benchmark (MICCAI COMPAYL)
- [29] Li et al. (2023) — IB finetuning
- [30] Campanella et al. (2024) — Full resolution end-to-end
- [31] Kumar et al. (2024) — DEMO
