# Whole Slide Image Redundancy Reduction for Efficient Pathological Diagnosis

**作者**: Shijie Li; Zhineng Chen; Feng-Jung Chen; Xieping Gao
**期刊/年份**: Biomedical Signal Processing and Control 2026, Vol. 114, Article 109289
**阅读状态**: 受限来源笔记，稳定页只开放摘要、Introduction 片段和 section snippets；原 PDF 为短时 ScienceDirect signed URL，当前不可复用
**链接**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1746809425018002) | [DOI](https://doi.org/10.1016/j.bspc.2025.109289)

## 一句话总结

WSIR2 把 WSI 弱监督诊断中的计算瓶颈解释为 patch/token 冗余问题，用 cross-attention 估计 token 重要性，同时做 top-k 保留、非关键 token 融合和轻量分类，从而在不明显牺牲诊断性能的前提下降低 MIL 推理成本。

## 核心贡献

1. 将 WSI 诊断效率问题从“更强 MIL 聚合器”转向“先减少冗余 token”，强调病理图像内部存在大量重复组织/细胞模式。
2. 提出 WSIR2，由 token compression module 和 lightweight classifier 组成，分别降低 token 数量、特征维度和分类器聚合开销。
3. 用 cross-attention 的 attention value 作为 token significance 信号，选择 top-k 关键 patches，并融合非 top-k patches，避免简单丢弃所有低分 token。
4. 在 Camelyon16、TCGA-NSCLC、UniToPatho 上报告了效率/性能平衡，摘要页给出的关键 claim 是最高 24x GFLOPs 降低、1.02 或 0.29 GFLOPs、225K 参数。
5. 论文声称 token compression 可插入其他主流 MIL 方法，说明它更像通用 WSI 加速模块，而不是单一分类器结构。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Source Scope](sections/00-source-scope.md) | 当前可访问来源、不可完整 MinerU 的原因、可信/不可信结论边界 |
| [01 - Method Reading](sections/01-method-reading.md) | WSIR2 的 token pruning、token merging、cross-attention classifier 数据流 |
| [02 - Evidence Reading](sections/02-evidence-reading.md) | 数据集、效率数字、plug-and-play claim、固定 top-k ratio 风险 |

## 关键数字

| 指标 | 数值 |
|------|------|
| DOI | 10.1016/j.bspc.2025.109289 |
| 期刊卷号 | Biomedical Signal Processing and Control, Volume 114 |
| Article number | 109289 |
| Crossref reference count | 43 |
| Crossref cited-by count | 0 |
| 报告数据集 | Camelyon16, TCGA-NSCLC, UniToPatho |
| Camelyon16 来源页数字 | 270 training samples, 129 testing samples, about 57.18M patches at 20x |
| 摘要页效率 claim | up to 24x lower GFLOPs |
| 摘要页模型规模 claim | 225K parameters |
| 摘要页计算 claim | 1.02 or 0.29 GFLOPs on evaluated datasets |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. WSI patching | gigapixel WSI | 切成 patch，并经 encoder 得到 patch features | token bag |
| 2. Token scoring | token bag | cross-attention / attention value 估计 token significance | 每个 token 的重要性分数 |
| 3. Token compression | token features + scores | 保留 top-k 关键 token，合并非关键 token，同时压缩 token 维与 feature 维 | 更小的 WSI token set |
| 4. Lightweight classifier | compressed tokens | 多个 cross-attention blocks 聚合 slide-level 表征 | WSI representation |
| 5. Diagnosis | slide representation | 分类头预测 slide label | 病理诊断结果 |

## 优缺点与还能做什么

### 优点

- 直接攻击 WSI MIL 的高通量瓶颈：减少 token 数和特征维，而不是只换更复杂的聚合器。
- token pruning 与 merging 同时出现，说明作者意识到低分 token 可能仍携带背景/上下文信息，不能完全丢弃。
- 轻量 classifier 与压缩模块同向优化，把压缩收益传到端到端 FLOPs，而不只是中间 token 数。
- plug-and-play 设定对 Medical-Compression topic 很重要：它可作为 WSI MIL 加速器，和 FOCUS/AdaSlide/WISE 的目标不同。

### 局限 / 风险

- 当前目录没有全文 PDF/MinerU 解析，无法核对完整实验表、消融细节、公式定义和可视化图。
- 来源页 snippets 提到固定 top-k ratio 依赖，这是临床场景的实际风险：不同组织面积、肿瘤负荷和染色质量下，固定压缩率可能不稳定。
- attention value 是否等价于诊断重要性仍需谨慎；MIL attention 常可解释性不足，需要病理可视化和跨数据集检验支撑。
- 只从摘要页无法确认与具体 baselines 的公平性，例如 encoder、patch size、magnification、batching 和硬件设置。

### 还能做什么

- 重新获取 PDF 后补全 `full.md`、`content_list.json`、figures/tables 和逐 section 批读。
- 将固定 top-k 改为按 slide complexity 或 uncertainty 自适应的动态 token budget。
- 和 FOCUS 的 foundation-model/language relevance 对齐，比较“attention 重要性”和“诊断语义重要性”是否一致。
- 在下游诊断之外增加 pathologist review 或 region-level recall，检查压缩是否漏掉稀疏病灶。

## 阅读 Q&A 记录

- **Q: WSIR2 和 DTC-WSI 都压缩 WSI token，关键差别是什么？**
  A: 从可访问来源看，WSIR2 更强调 cross-attention 下的 top-k selection + non-top-k merging + lightweight classifier；DTC-WSI 则以动态多阶段 compression、importance network、bipartite soft matching 和 pruning 组成更细的逐阶段压缩流水线。
- **Q: 为什么这篇放在 Medical-Compression，而不是一般 MIL？**
  A: 论文的主 claim 是冗余 reduction 和诊断加速，核心对象是 WSI token/features 的压缩，不只是提升分类准确率。
- **Q: 当前笔记能否替代全文批读？**
  A: 不能。当前只适合做 topic 索引和初读判断；需要新的可下载 PDF 链接后才能做完整 MinerU 与逐段批注。

## 📊 Citation Landscape

> 数据来源: `crossref.json`、`semantic-scholar/detail.json`。Semantic Scholar 在本次调用中返回 HTTP 429，因此 citation landscape 只使用 Crossref 可用字段。

### TLDR

暂缺：Semantic Scholar detail API 返回 429。

### 引用统计

| 指标 | 数值 |
|------|------|
| Crossref reference count | 43 |
| Crossref cited-by count | 0 |
| Semantic Scholar detail | HTTP 429 |
| Semantic Scholar recommendations | 暂缺 |

### 参考文献分组

暂缺：Crossref 元数据未提供可直接分组的完整参考文献标题，需全文 PDF 或 Semantic Scholar references 后补。

### 推荐论文

暂缺：Semantic Scholar recommendations 因 detail API 限流未调用成功。

## BibTeX

```bibtex
@article{li2026whole,
  title={Whole slide image redundancy reduction for efficient pathological diagnosis},
  author={Li, Shijie and Chen, Zhineng and Chen, Feng-Jung and Gao, Xieping},
  journal={Biomedical Signal Processing and Control},
  volume={114},
  pages={109289},
  year={2026},
  doi={10.1016/j.bspc.2025.109289}
}
```
