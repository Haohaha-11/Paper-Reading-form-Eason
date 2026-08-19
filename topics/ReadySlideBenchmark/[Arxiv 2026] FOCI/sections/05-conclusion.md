[← 返回 README](../README.md)

## 📌 批读预览

结论把 FOCI 限定为模型审计层，并再次收紧临床外推范围。

## 5 Conclusion

FOCI is a post-hoc rationale-highlighting layer for frozen WSI-MIL classifiers: the full-bag prediction is preserved while FOCI selects a compact, output-consistent tile subset that recovers it, evaluated through SRP and the Selection Headroom Index. Across three benchmarks and seven backbones, compact rationales are selection-headroom dependent — TransMIL and ACMIL admit consistent compression, attention-pooling backbones saturate on near-minimal baselines, and hard-selection backbones conflict with an external readout. High slide-level AUC does not by itself imply that a frozen MIL classifier admits a compact rationale; SHI motivates a selection-headroom audit before treating selected tiles as faithful explanations.

> 💡 **claude 批注｜对 ReadySlide 的直接结论**: FOCI 只给主 true-label 协议下的 native/proxy→learned gap。ReadySlide 应另定义 consumer-optimal combinatorial Oracle：固定 consumer、候选池、$y$、$\kappa$ 与可行子集空间，最小化 K；它给 selection performance 上界（MSK 下界），从而得到 learned→consumer-optimal gap。tumor mask、region annotation 或 reader 标注只用于 clinical alignment，不必保留 consumer 证据，不能与该 Oracle 混称。

Limitations. We measure model-output sufficiency only: selected tiles are candidate rationales for the frozen classifier, not annotation-validated clinical evidence. Our experiments use UNI2-h features and binary WSI tasks, so broader encoder evaluation, multiclass settings, external clinical cohorts, and multi-reader validation remain future work; details are in Appendix M.

> 💡 **claude 批注｜外推边界**: 缺失的不是更多 AUC 表，而是 ground-truth tumor annotation、跨 encoder、跨院外部验证与 reader study。ReadySlide 若面向临床可用性，还要把组织学覆盖、罕见亚型漏检和人工复核成本加入选择质量，而不能只保证 consumer 输出不变。

## 🔖 本节总结

- 贡献成立在 consumer-relative、true-label-directed sufficiency 审计层，不是临床证据发现或 full-bag prediction fidelity。
- 新 selector 是否值得训练，应分别看 native→learned、learned→consumer-optimal 与 clinical alignment，三项不得合并成单一指标。
- 未来 benchmark 需补 multiclass、跨 encoder、外部队列和 reader-level 结果。
