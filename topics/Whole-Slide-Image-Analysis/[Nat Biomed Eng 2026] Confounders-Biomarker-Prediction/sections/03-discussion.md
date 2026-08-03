[← 返回 README](../README.md)

# Discussion 讨论

## 📌 预览

讨论把 BRAF-MSI 的临床后果讲透（两者共现但治疗完全不同，混淆会误导用药），重申 grade/TMB 代理问题，警示"外部验证高 AUROC 可能是关联漂移造成的假象"，给出临床定位（分诊/辅助而非替代分子检测）与四条试验设计建议，并指出未来方向 = 因果结构化多标签学习。

---

Deep learning models trained on routine WSIs are increasingly discussed as rapid and cost-effective tools to infer molecular biomarker status. In this study, we identified key limitations... For example, the PR predictor showed a marked drop in performance in CDH1-mutant cases, with AUROC decreasing from 0.79 to 0.50. This decline suggests that current ML models cannot fully disentangle biomarker-specific signals from the multifaceted influence of molecular characteristics on tissue phenotypes.

The inability of WSI-based models to discern biomarker-specific signals has direct clinical implications when codependent biomarkers have divergent therapeutic roles. An example is the BRAF-MSI association in CRC... MSI-H is a strong predictor of response to immune checkpoint inhibitors (pembrolizumab/nivolumab), whereas BRAF V600E mutations are targeted using BRAF and MEK inhibitors with EGFR blockade. A model that cannot disentangle MSI-H from BRAF status may achieve high aggregate AUROC but lacks clinical utility, as confusing the two would misguide treatment selection.

> 💡 **机制拆解**（为什么混杂在临床上是致命的）（Hao 批注）：这是全文最有临床分量的论证。BRAF 和 MSI 在 CRC 里生物学共现，模型可以靠这个共现刷高 AUROC；**但两者的治疗方案完全相反**（MSI-H→免疫检查点抑制剂，BRAF V600E→BRAF/MEK 抑制剂）。一个分不清两者的模型，聚合 AUROC 再高也可能给出错误用药建议。**这把"统计混杂"翻译成"病人吃错药"**——远比"AUROC 掉点"严重。方法论启示：评估必须包含"能否区分治疗方案相反的相关 biomarker"。

Beyond biomarker interdependencies, we showed that these models exploit prominent grade- or TMB-associated features as proxies... These patterns reflect a broader challenge in computational pathology: models tend to exploit confounding variables (grade, TMB) and conflate them with biomarkers of interest, thereby obscuring true genotype–phenotype relationships, limiting generalizability and introducing bias.

These findings underscore the need to interpret external validation results with caution. In our analysis, the ER predictor achieved a high AUROC of 0.87 in cross-validation on TCGA-BRCA and 0.90 in a larger independent cohort (ABCTB), which could be interpreted as excellent generalizability. However, the apparent improvement was largely driven by a stronger grade-ER association in ABCTB than in the training cohort.

> 💡 **方法论批读**（"外部验证高分"陷阱）（Hao 批注）：这段是给所有做 WSI 迁移/泛化研究的警钟——**在更大的外部队列上 AUROC 更高（0.87→0.90），看起来泛化极好，实则是因为外部队列里 grade-ER 关联更强，模型的 proxy 更好使**。这意味着"外部验证通过"根本不等于"学到了因果信号"。对 ReadySlide 的跨中心/跨 FM 迁移实验：**必须做分层验证**，否则"跨中心 AUROC 保持"可能只是"混杂结构恰好相似"。

The confounding influence... suggests that current models are not yet ready to replace genomic testing in routine care. Instead, they are better positioned for triaging, screening or supplementary decision support. To ensure true clinical utility, we suggest bias-aware evaluation, including reporting grade- and TMB-stratified metrics and subgroup calibration rather than relying solely on aggregate AUROC. We recommend: (1) preserving variation in the target biomarker relative to correlated variables during enrolment; (2) prespecifying stratification factors and conducting prospective subgroup analyses; (3) including a dependency-aware analysis plan; and (4) conducting per stratum power calculations.

Although predicting biomarker status from routine H&E WSIs may appear to be a simple image-to-label mapping, it is considerably more complex because phenotypes in WSIs are rarely driven by a single factor. Our analyses show that current approaches fail to reliably learn biomarker-specific genotype–phenotype mapping; instead, they exploit aggregated phenotypes or cohort-specific associations as proxies. These findings motivate the need for methods that formalize the problem as causal, structured multilabel learning: explicitly encode dependencies among biomarkers in the label space, learn disentangled image representations guided by conditional-independence objectives, mitigate confounding via causal adjustment and counterfactual data augmentation and optimize for invariance and distributional robustness.

> 💡 **未来方向解读 + 对本主题的启示**（Hao 批注）：作者开出的药方是一整套因果 ML：**标签空间显式编码 biomarker 依赖 + 条件独立目标学 disentangled 表征 + 因果调整/反事实增强 + 分布鲁棒 + 分层校准评估**。
> - **对 WSI Analysis 主题**：这是对"刷 AUROC 范式"的根本性纠偏。与 [PathBench](../%5BArxiv%202025%5D%20PathBench/)（防数据泄漏的标准 benchmark）互补——一个管"评测标准"，一个管"评测维度（分层去混杂）"。
> - **对 ReadySlide/压缩研究**：最直接的可复用方法论——**分层+permutation 协议**可用来验证"压缩/保留是否保住了因果诊断信号 vs 只保住了 shortcut（grade/染色/TMB 形态）"。若压缩后整体 AUROC 不掉但分层后崩，说明压缩保的是 shortcut。这把 memory 里"必须对抗 shortcut-learning"落到了可执行的检验上。

## 局限（作者自述）

作者诚实列出：(1) 只分析 H&E WSI + WSI 级粗标签，未评估 IHC 或空间组学监督；(2) 虽用了 8221 例多中心，仍需前瞻研究定临床指南；(3) 学 disentangled genotype-phenotype 映射需要组合更丰富（穷尽共突变组合）的数据集，curation 成本极高；(4) 提出的方法方向有效性待验证，现在下临床指南为时过早。

> 💡 **局限解读**（Hao 批注）：第 (3) 条尤其重要——**要真正去混杂，需要"共突变组合被充分覆盖"的数据集**（这样才能在固定一个 biomarker 时仍有足够样本估另一个）。现有 cohort 做不到，这是因果 WSI 学习的数据瓶颈。作者没有给出可用的解法，只指方向——所以这篇是"诊断"论文而非"解药"论文，价值在于确立问题和检验协议。
