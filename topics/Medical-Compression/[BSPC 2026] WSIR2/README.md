# Whole slide image redundancy reduction for efficient pathological diagnosis

**简称**: WSIR2
**作者**: Shijie Li; Zhineng Chen; Feng-Jung Chen; Xieping Gao
**期刊/年份**: Biomedical Signal Processing and Control 2026, Vol. 114, Article 109289
**链接**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1746809425018002) | [DOI](https://doi.org/10.1016/j.bspc.2025.109289)
**本地材料**: [paper.pdf](paper.pdf), [full.md](full.md), [content_list.json](content_list.json), [paper-meta.json](paper-meta.json)

## 一句话总结

WSIR2 把 WSI 弱监督诊断里的计算瓶颈定义为 patch/token 冗余问题，用 cross-attention 估计 token 重要性、保留 top-k patches、合并非关键 token，并用轻量 cross-attention classifier 在接近 SOTA MIL 性能的同时显著降低 FLOPs、参数量和推理成本。

## 核心贡献

1. 将病理 WSI 的重复组织形态显式建模为可压缩冗余，而不是默认所有 patch token 都需要完整送入 MIL 聚合器。
2. 提出 token compression module：先把 patch features 投影到较小维度，再用 learnable query 的 cross-attention scores 选择 top-k informative tokens。
3. 对 non-top-k tokens 不直接丢弃，而是用复用的 attention weights 聚合成一个 redundant token，保留背景/上下文信息。
4. 用 O(n) cross-attention blocks 构建 lightweight classifier，使压缩收益传导到 slide-level aggregation，而不只是减少中间 token 数。
5. 在 Camelyon16、TCGA-NSCLC、UniToPatho 上给出 accuracy/AUC、GFLOPs、throughput、train time、memory 和 plug-and-play 对照，证明 WSIR2 是 WSI MIL 内部的诊断加速模块。

## 📖 批读导航

| Section | 内容 |
|---|---|
| [00 - Abstract](sections/00-abstract.md) | 摘要、问题设定、核心 claim 与快速读法 |
| [01 - Introduction](sections/01-introduction.md) | WSI 冗余动机、Figure 1、两模块贡献和效率目标 |
| [02 - Related Work](sections/02-related-work.md) | MIL for WSI、vision redundancy reduction 与本文差异 |
| [03 - Methodology](sections/03-methodology.md) | MIL 公式、top-k selection、non-top-k merging、lightweight classifier |
| [04 - Experiments](sections/04-experiments.md) | 数据集、实现细节、Table 1-4、Figure 3-4、主结果与消融 |
| [05 - Discussion](sections/05-discussion.md) | fixed top-k ratio、预提取特征、空间上下文等风险 |
| [06 - Conclusion](sections/06-conclusion.md) | 结论、数据可用性、引用脉络与 BibTeX |

## 关键数字

| 指标 | 数值 / 读法 |
|---|---|
| DOI | 10.1016/j.bspc.2025.109289 |
| 期刊 | Biomedical Signal Processing and Control, Volume 114, Article 109289 |
| PDF 页数 | 7 页 |
| 数据集 | Camelyon16, TCGA-NSCLC, UniToPatho |
| Camelyon16 | 270 train, 129 test, about 57.18M patches at 20x |
| TCGA-NSCLC | 1042 slides, about 145.63M patches at 20x, 8:2 split |
| UniToPatho | 243 WSIs used, about 9.87M patches at 20x, official train/test split |
| Patch / encoder | 224x224 non-overlapping patches at 20x; DINO-pretrained ViT-small/16, hidden dim 384 |
| WSIR2 default compressed dim | d' = 96 in main results |
| top-k ratios | r = 1.00, 0.50, 0.10, 0.01 |
| 模型参数 | 225K |
| GFLOPs at r=0.50 | 1.02G for 14,000 patches; 0.29G for 4,000 patches |
| GFLOPs at r=0.01 | 0.69G for 14,000 patches; 0.20G for 4,000 patches |
| Throughput on TCGA-NSCLC | 985 WSIs/s at r=1.00; 1096 WSIs/s at r=0.01 |
| Memory usage | 0.43 GB for WSIR2 variants |
| Plug-in ABMIL at r=0.50 | GFLOPs 4.24G -> 0.75G; throughput 965 -> 1257 WSIs/s; Acc 0.9450 -> 0.9402 |
| Plug-in TransMIL at r=0.50 | GFLOPs 34.61G -> 17.14G; throughput 62 -> 136 WSIs/s |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|---|---|---|---|
| 1. WSI tiling | gigapixel WSI | tissue/background filtering 后切成 224x224 patches at 20x | patch set |
| 2. Patch encoding | patches | DINO self-supervised ViT-small/16 提取 patch features | slide-level feature matrix X in R^{n x d} |
| 3. Feature compression | X | learnable W_k 投影到 K in R^{n x d'} | 低维 patch token features |
| 4. Importance scoring | K + learnable query q* | cross-attention softmax(q*K^T / sqrt(d')) | token attention scores A |
| 5. Token selection | A + K | k=max(1, ceil(n*r))，选 TopK scores | informative tokens K_t |
| 6. Token merging | non-top-k tokens K_r | 复用 A_r，加 W_v 投影，聚合成 x_r | one redundant token |
| 7. Lightweight aggregation | concat(K_t, x_r) | class token Q cross-attends to compressed tokens | global representation X_g |
| 8. Diagnosis | X_g | scoring head S(.) | slide-level label prediction |

## 优缺点与还能做什么

### 优点

- 思路清楚：病理 WSI 中大量相似 tissue/cell morphology 是可压缩对象，和 Medical-Compression 主题直接对齐。
- top-k selection 与 non-top-k merging 同时使用，比“只剪掉低分 token”更稳，因为背景和上下文仍被压成一个 redundant token。
- cross-attention 查询 token bag 的复杂度是 O(n)，比 self-attention 更适合上万 patch 的 WSI 聚合。
- 实验不仅看 accuracy/AUC，也看 GFLOPs、inference speed、train time、memory usage 和 plug-and-play 加速，证据链围绕效率展开。
- 轻量 classifier 只有 225K 参数，说明收益来自结构压缩，而不是更大聚合器。

### 局限 / 风险

- top-k ratio r 是固定超参数；不同 WSI 的组织面积、肿瘤比例和病灶稀疏性不同，固定预算可能导致小病灶被过度压缩。
- attention value 被当作 token significance，但 MIL attention 的可解释性并不自动等价于临床重要性，需要更细的 lesion-level recall 或 pathologist review 支撑。
- 方法依赖预提取 patch features，没有把 patch encoder 成本、I/O、slide tiling 和缓存一起纳入端到端部署成本。
- token merging 将所有 non-top-k token 压成单一 redundant token，可能对需要多种背景 subtype 或 spatial context 的任务不够表达。
- 实验任务都是分类；对 survival prediction、retrieval、report generation、VQA、region-level localization 的可迁移性还需要验证。

### 还能做什么

- 将固定 top-k ratio 改成 per-slide adaptive budget，依据 tissue area、attention entropy、uncertainty 或 tumor purity 自动调节。
- 给 non-top-k tokens 做多个 prototype / cluster merging，而不是只压成一个 redundant token。
- 加入空间邻接约束或 multi-scale features，避免跨组织区域的 token 被过度合并。
- 把 compression module 和 DTC-WSI 的 multi-stage dynamic pruning 对照，分析“一次压缩 + 轻量分类器”和“多阶段压缩”的边界。
- 在端到端系统中统计 patch extraction、feature encoding、MIL aggregation、I/O 的真实耗时，避免只优化 classifier FLOPs。

## 阅读 Q&A 记录

- **Q: WSIR2 和 DTC-WSI 都做 WSI token compression，差别在哪里？**
  A: WSIR2 是单个 cross-attention compression block：top-k 保留 + non-top-k 合并 + 轻量 classifier；DTC-WSI 是 multi-stage dynamic merge/prune，并用 importance network 与 bipartite soft matching。WSIR2 更像“把 MIL 聚合器做薄”，DTC-WSI 更像“逐阶段管理 token budget”。

- **Q: 为什么 non-top-k token 不直接删掉？**
  A: 病理诊断虽然常由少数 tumor/fat/atypia 区域决定，但背景组织和 normal morphology 也提供上下文。作者把 non-top-k tokens 聚合为 redundant token，是在效率和信息保留之间折中。

- **Q: r=0.01 是否总是最好？**
  A: 不一定。r=0.01 FLOPs 最低、throughput 最高，但 Camelyon16 accuracy/AUC 明显低于 r=1.00/0.50；这说明极端压缩会丢诊断信息。实践上更应按数据集和临床风险选择 r。

- **Q: 最能支持方法有效性的表是哪张？**
  A: Table 2 和 Table 3。Table 2 说明 WSIR2 本身在参数/FLOPs/throughput 上有系统收益；Table 3 说明 compression module 插到 ABMIL/DSMIL/TransMIL 后仍能加速，支持 plug-and-play claim。

- **Q: Figure 4 证明了什么？**
  A: 它显示 top-k patches 更集中在 tumor/fat 等诊断区域，而 merged patches 覆盖大量相似背景。它支持“attention score 能抓到诊断相关 token”的直观假设，但还不能替代定量 region-level 评估。

## 📊 Citation Landscape

> 数据来源: 本地 PDF references、`crossref.json`、`semantic-scholar/detail.json`。Semantic Scholar 在此前调用中返回 HTTP 429，因此不填造 citation count、influential citation count 或自动 TLDR。

| 项目 | 当前记录 |
|---|---|
| Crossref reference count | 43 |
| Crossref cited-by count | 0 |
| Semantic Scholar TLDR | unavailable, API 429 |
| Semantic Scholar citation count | unavailable, API 429 |
| Connected Papers | 暂缺，ScienceDirect/DOI 条目未在本地解析出 Semantic Scholar paperId |

### 参考文献脉络分组

| 主题 | 代表引用 | 与 WSIR2 的关系 |
|---|---|---|
| Computational pathology 背景 | Cui & Zhang 2021; Bera et al. 2019; Srinidhi et al. 2021 | 说明 WSI 数字化和病理 AI 的场景需求 |
| WSI MIL baselines | ABMIL; CLAM; TransMIL; HIPT; DTFD-MIL; MHIM; IBMIL | 提供 WSIR2 的分类性能和效率对照 |
| Efficient WSI / token-efficient MIL | Remix; Long-MIL; DTFD-MIL | 对比已有尝试，突出本文聚焦 WSI 内部冗余 reduction |
| Vision token pruning / merging | DynamicViT; A-ViT; token reorganization; ToMe | 提供 top-k / token merging 灵感，但这些方法多面向固定尺寸自然图像 |
| Redundancy reduction 理论 | Barlow; Barlow Twins; image compression; prompt compression | 提供“减少冗余表示”的广义理论背景 |
| 数据集 | Camelyon16; UniToPatho | 支撑 breast metastasis 和 colorectal pathology 场景实验 |

### 推荐追读

- DTC-WSI: 对比 multi-stage token compression 与 WSIR2 单块压缩。
- FOCUS: 对比 foundation-model/language-guided token filtering 与 attention-score filtering。
- TransMIL / CLAM / ABMIL: 作为 WSIR2 plug-in 和性能对照的基本 MIL 聚合器。
- DynamicViT / ToMe: 理解 vision token pruning/merging 如何迁移到 WSI。

## BibTeX

```bibtex
@article{li2026whole,
  title = {Whole slide image redundancy reduction for efficient pathological diagnosis},
  author = {Li, Shijie and Chen, Zhineng and Chen, Feng-Jung and Gao, Xieping},
  journal = {Biomedical Signal Processing and Control},
  volume = {114},
  pages = {109289},
  year = {2026},
  doi = {10.1016/j.bspc.2025.109289}
}
```
