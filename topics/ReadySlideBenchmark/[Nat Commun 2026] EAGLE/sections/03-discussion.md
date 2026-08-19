[← 返回 README](../README.md)

# Discussion & Methods 讨论与方法

## 📌 预览

Discussion 把"25 tile"定位为弱监督下的偏差-方差权衡（信号空间稀疏时限制到高显著子集改善统计条件），坦承稀疏采样对形态学/长程上下文任务有短板、CHIEF 偏癌症预训练限制非癌任务。Methods 给出 EAGLE 两阶段细节（CHIEF 在 6 万+ slide 预训练、Virchow2 在 3M+ WSI）、25-tile 等权平均、负对照与注意力集中度量。

---

## Discussion

EAGLE's design centers on processing only 25 tiles per WSI, a strategy that substantially reduces computational requirements while preserving predictive power. **This design is not a heuristic truncation but reflects a bias–variance trade-off under weak supervision: when the predictive signal is spatially sparse relative to total tissue area, restricting inference to a reproducible, high-saliency subset can improve statistical conditioning.** This approach aligns with clinical practice, in which pathologists focus on diagnostically relevant regions rather than exhaustively scanning each tile.

Our benchmarking shows that combining CHIEF and Virchow2 in EAGLE outperforms both individual models in the majority of cases and is never worse than both on any task. Targeted selection of informative tiles proved essential; selecting more than 25 tiles tended to introduce noise and reduce performance. **The strong performance of a fixed 25-tile budget should be interpreted as a robust empirical operating point rather than as a biologically universal optimum.** Histopathology contains substantial spatial redundancy, and CHIEF ranking appears to recover much of the predictive signal early. However, this advantage is task-dependent: for morphology-heavy endpoints that require broader architectural context or longer-range spatial dependencies, dense slide encoders can remain competitive or superior (e.g., TITAN in NSCLC subtyping). Any sparse-sampling strategy carries a residual risk of missing rare, spatially dispersed, or globally contextual cues.

> 💡 **机制拆解**（"25 tile" 的原理性辩护 + 边界）（Hao 批注）：这是全文最有理论分量的一段，也是对 ReadySlide 最有价值的洞察：
> - **为什么少即是多**：不是截断，而是**偏差-方差权衡**。信号空间稀疏时，处理全部 tile 引入大量无信号 tile → 高方差；限制到高显著子集 → 改善统计条件。这给"激进保留"提供了统计学理由（而非仅仅"省算力"）。
> - **边界（诚实）**：(1) 25 是**经验操作点**，非生物学普适最优；(2) **形态学/长程上下文任务**（如肺癌亚型、血管侵犯）稀疏采样会漏全局线索，dense encoder（TITAN）更好；(3) 稀疏采样有"漏掉罕见/分散线索"的残余风险。
> - **对 ReadySlide**：保留率应**任务自适应**（生物标志物任务可激进保留，形态学任务需保更多/保空间上下文）——这与 memory 里"importance→retention 对 BRACS/diffuse 成立、对 PANDA/ordinal 不成立"的任务分层高度一致。

Importantly, performance gains cannot be attributed to arbitrary subsampling: saliency-guided selection consistently outperformed repeated uniform random selection under identical splits. Extensive hyperparameter tuning provided only negligible benefit on external cohorts, suggesting performance is primarily determined by representation quality rather than classifier optimization. EAGLE provides explicit spatial localization: the exact regions used for each prediction can be enumerated, reproduced, and rapidly reviewed. While EAGLE achieved AUROCs >0.900 for some tasks, the average AUROC of 0.742 indicates that EAGLE's performance might not yet be adequate to replace standard clinical procedures.

> 💡 **局限与定位解读**（Hao 批注）：作者难得诚实——**平均 AUROC 0.742 还不足以替代临床流程**，定位为辅助/分诊。且明确"本研究优先 benchmark 预测性能而非混杂分析"，未做 [Confounders](../../%5BNat%20Biomed%20Eng%202026%5D%20Confounders-Biomarker-Prediction/) 式的分层去混杂——**这是 EAGLE 的一个 open question：它选的 25 个"信息 tile"是承载因果诊断信号，还是承载 grade/染色 shortcut？** 结合本目录 Confounders 的方法论，这是评估 EAGLE（及任何 retention 方法）该补的一步。CHIEF 偏癌症 slide 预训练 → 非癌任务上选择可能失效，也是边界。

## Methods（核心）

**EAGLE two-stage**: (1) CHIEF (frozen, task-agnostic ABMIL pretrained on 60,000+ WSIs across 19 anatomical sites via tile SSL + weakly-supervised slide contrastive + anatomical site encoding) runs on CTransPath embeddings (2 MPP) → produces slide representation + attention vector → **repurposed to select top 25 most informative tiles**. (2) Those 25 tiles re-extracted with Virchow2 (ViT trained on 3M+ WSIs) → 25 embeddings **averaged equally** (avoid single-tile dominance) → compact unsupervised slide embedding → small MLP for downstream tasks.

**Benchmark**: 31 tasks (morphology/biomarker/prognosis) × 4 cancers, 5-fold CV on TCGA, external test on CPTAC/DACHS/Kiel/Bern/IEO. Tiles 224×224, Canny edge background rejection (<2% edges). MLP: input 768, hidden 256, SiLU, 32 epochs, AdamW (lr 1e-4, wd 1e-2), class-weighted cross-entropy.

**Negative control**: uniform random tile selection, R=100 replicates per budget N∈[5,10,25,50,100], Monte Carlo p=(r+1)/(R+1).

**Attention concentration**: Lorenz curves of cumulative attention mass, Gini coefficient, fraction of tiles for 50%/80% mass, top-k mass — computed per patient, aggregated by averaging rank profiles (not tile identities).

> 💡 **方法论批读**（EAGLE 可复用的三个方法要素）（Hao 批注）：
> 1. **"repurpose CHIEF 的注意力向量选 tile"**：CHIEF 的注意力本是为聚合成 slide embedding 设计的，EAGLE 拿它当**tile 选择器**。任何提供 tile 级相关性分数的 slide encoder 都可替换 CHIEF——这是一个通用的"用预训练 FM 做保留"的接口。
> 2. **等权平均而非注意力加权**：作者选 25 tile 等权平均（不用 CHIEF 注意力加权），因为二者在 25 tile 时表现相近，等权更透明可审计。对 ReadySlide：保留后**简单平均可能就够**（呼应 [SiMLP](../../../ckmil-re-attn-mil/) 的 mean pooling），不必再上复杂聚合器。
> 3. **负对照 + 注意力集中度量**：这两套协议是验证"选择器是否真有效"的黄金标准——负对照证明超随机、Lorenz/Gini 量化集中度。**ReadySlide 的 allocator 评估应标配这两项**：(a) 显著超随机保留；(b) 报保留分布的集中度。

## Conclusion

EAGLE demonstrates that combining complementary pathology foundation models (task-agnostic slide-level selector + strong tile encoder) yields representations that are both efficient and broadly generalizable, establishing an effective paradigm for building high-performing slide-level models without additional large-scale data collection.

> 💡 **总结对 ReadySlide 的映射**（Hao 批注）：EAGLE = **"compress-by-selection once (CHIEF 选 25 tile), analyze with strong FM (Virchow2)"** 的成功范例，且 task-agnostic（选一次、多任务复用）。这几乎就是 ReadySlide "analysis-ready transfer：压一次、任意 FM/任务分析"的病理版实证。差异与可追问：(1) EAGLE 是"选 tile"（离散保留），ReadySlide 探索的是"分辨率阶梯/码率分配"（连续压缩）——选择 vs 压缩哪个 Pareto 更优？(2) EAGLE 未验去混杂——选的 tile 是否 shortcut？(3) EAGLE 的 CHIEF 选择器是否可迁移到 ReadySlide 的 allocator，替代 importance_chief？
