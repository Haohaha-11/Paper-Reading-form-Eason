[← 返回 README](../README.md)

## 5. Conclusion

> 💡 **结论预览（claude 批注）**: 论文的可靠结论是“显式 evidence objective 能在三个研究数据集上把高置信度集中到很少 tile，同时大体保持 AUC”，而不是“这些 tile 已被临床证明为因果证据”。

We presented ReaMIL, a method that transforms wholeslide classification into an evidence-seeking problem by adding a budgeted selection head to standard MIL backbones. Training the selector so that a small, spatially compact subset suffices for prediction while forcing complementary tiles to be non-predictive for the true class preserves baseline AUC while producing compact evidence— on TCGA-NSCLC, AUC 0.983 with MSK ≈ 8.2 at τ = 0.90 and AUKC ≈ 0.864. The framework requires only slide-level supervision, fits existing pipelines, and shows that accurate yet interpretable MIL is achievable without extra annotation—critical as computational pathology moves toward clinical deployment.

Limitations. Our approach relies on pre-extracted features from a single foundation model (UNI2-h) and has been evaluated on relatively balanced research datasets. Validation on more diverse clinical cohorts with class imbalance and domain shift, as well as user studies with pathologists to assess clinical utility, remain important directions for future work.

> 💡 **局限映射（claude 批注）**: 单 FM、单 consumer、平衡研究队列正是 ReadySlideBenchmark 要补的三层外推风险。还应增加：selector 是否跨 consumer 迁移、预算变化是否改变 FM 排名、严格冻结 consumer 时 head 能否修复，以及节省的是昂贵 FM 编码、MIL 聚合还是仅展示成本。

## 🔖 Section 总结

- ReaMIL 是同模型 selector adaptation 的直接实现参考。
- 临床因果性、跨模型迁移、域偏移和真实算力节省仍属开放问题。
