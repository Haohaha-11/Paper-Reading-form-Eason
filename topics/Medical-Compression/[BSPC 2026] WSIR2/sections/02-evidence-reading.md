[← 返回 README](../README.md)

# Evidence Reading

## 📌 预览

当前可访问证据集中在摘要页和 snippets：数据集覆盖 Camelyon16、TCGA-NSCLC、UniToPatho；效率 claim 包括最高 24x GFLOPs 降低、1.02 或 0.29 GFLOPs、225K 参数；讨论 snippet 承认固定 top-k ratio 可能是限制。

> 💡 **证据链批注**: 对 Medical-Compression 主题来说，WSIR2 的关键不是单个 accuracy 数字，而是 accuracy、FLOPs、吞吐、参数量、可插拔性是否一起成立。当前公开页只支持“有这样的 claim”，不能支持“所有实验均已核验”。

## Datasets

公开 snippets 显示 WSIR2 使用 Camelyon16、TCGA-NSCLC 和 UniToPatho。Camelyon16 片段给出 270 training samples、129 testing samples，以及约 57.18M patches at 20x magnification。

> 💡 **数据集批读**: 这三个数据集覆盖 breast metastasis、lung cancer subtype 和 colon pathology 场景，能初步检验跨器官/任务泛化。但没有全文表格前，仍无法判断各数据集的 split、preprocessing 和 encoder 是否与 baselines 完全一致。

## Efficiency claims

公开摘要页给出的效率信息包括：最高 24x lower GFLOPs，模型规模 225K parameters，在相关数据集上 1.02 或 0.29 GFLOPs。

> 💡 **效率批读**: 如果这些数字来自同一 pipeline 的端到端统计，说明 WSIR2 的压缩确实传导到了分类器成本；如果只统计 classifier FLOPs 而不统计 patch encoder，则临床端到端收益会被高估。这个区别需要全文实验设置确认。

## Plug-and-play compatibility

摘要页 claim 表示 token compression module 可兼容其他 mainstream MIL methods。

> 💡 **可复用性批注**: plug-and-play 是该论文最有价值的外推点，因为它把 WSIR2 从一个模型变成一类 WSI MIL 加速策略。后续应重点检查：插入其他 MIL 后 accuracy 是否稳定、压缩率是否需要每个方法重调、以及低资源数据集上是否更容易误删关键 token。

## Limitations

Discussion snippet 提到固定 top-k ratio 可能带来限制。

> 💡 **风险批注**: 固定 top-k ratio 在病理里尤其敏感：小病灶、低肿瘤比例或大面积正常组织的 slide，对“保留多少 token”需求不同。更合理的下一步是让 compression budget 跟 slide-level uncertainty、token entropy 或 tissue area 自适应变化。

## Q&A 批注记录

- Q: 当前最该补的实验是什么?
  A: 补全文后先找 ablation：只 pruning、只 merging、两者结合、不同 top-k ratio、不同 MIL backbone 的准确率和 FLOPs。
- Q: 这篇能否直接指导系统部署?
  A: 还不够。需要知道 patch encoder 成本是否计入、吞吐在真实 WSI batch/IO 中是否成立，以及压缩后是否保留稀疏异常区域。
