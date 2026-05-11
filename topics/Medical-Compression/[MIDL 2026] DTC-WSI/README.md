# DTC-WSI: Dynamic Token Compression for Whole-Slide Images

**简称**: DTC-WSI
**作者**: Tawsifur Rahman; Aliasghar Tarkhan; Rama Chellappa; Alexander S. Baras
**会议/期刊**: Medical Imaging with Deep Learning (MIDL) / PMLR | **年份**: 2026
**页码**: 3846-3865 | **发布日期**: 2026-05-05
**链接**: [PMLR](https://proceedings.mlr.press/v315/rahman26b.html) | [PDF](https://raw.githubusercontent.com/mlresearch/v315/main/assets/rahman26b/rahman26b.pdf)
**本地材料**: [full.md](full.md), [paper.pdf](paper.pdf), [content_list.json](content_list.json), [paper-meta.json](paper-meta.json)

## 一句话总结

DTC-WSI 把 WSI 的 patch token 先按相似性做 bipartite soft matching 合并，再用 importance network 做多阶段 saliency-guided pruning，在弱监督 MIL 中用更少 token 同时提升分类性能和推理效率。

## 核心贡献

1. 提出面向 WSI 的 dynamic multi-stage token compression：不是一次性删 token，而是在多个阶段渐进压缩，降低弱监督早期误删诊断区域的风险。
2. 将 similarity-guided token merging 和 importance-guided pruning 放在同一 MIL pipeline 中：合并负责吸收冗余形态，剪枝负责去掉低 saliency 区域。
3. 用轻量 importance network 在 slide-level label 下学习 patch saliency，训练时用 soft gates 保持梯度流，推理时用 deterministic Top-K 获得实际加速。
4. 采用 bipartite soft matching 避免全量 $O(N^2)$ 相似度搜索，并在 merge utility 中加入 importance consistency，防止把 saliency 差异大的 token 合并。
5. 在 TCGA-NSCLC、TCGA-BRCA、TCGA-RCC、PANDA 上报告 r=0.4 时准确率/AUC 全面优于 MIL 与 token-efficient baselines，并给出效率、消融、可视化证据链。

## 📖 批读导航

| Section | 内容 |
|---|---|
| [00 - Abstract](sections/00-abstract.md) | 摘要 claim、token reduction / speedup / memory / accuracy 数字入口 |
| [01 - Introduction](sections/01-introduction.md) | WSI token 冗余、pruning 风险、为什么要 merging + pruning |
| [02 - Methods](sections/02-methods.md) | patch encoding、importance network、multi-stage compression、BiSoft matching、pruning、MIL loss |
| [03 - Results](sections/03-results.md) | 四个 WSI 数据集、主结果、效率对比、消融、backbone/encoder 泛化 |
| [04 - Visualization](sections/04-visualization.md) | Figure 2/3 与 sparsity regularization，压缩行为可解释性 |
| [05 - Conclusion](sections/05-conclusion.md) | 结论 claim 与读后风险提示 |
| [06 - Appendix](sections/06-appendix.md) | WSI 预处理、retention ratio 扩展消融、Figure 4、Algorithm 1 |

## 关键数字

| 指标 | 数值 / 读法 |
|---|---|
| 主实验 retention | $r=0.4$，保留 40% token；多阶段默认 $r_1=r_2=r_3=0.74$，最终约 40% |
| 摘要压缩 claim | 5-10x token reduction；主表 r=0.4 更直接对应 2.5x token 数减少，需要区分表述口径 |
| 推理加速 | full pipeline 2150 ms/WSI -> 410 ms/WSI，5.3x speedup |
| FLOPs | 118.4G -> 24.3G at r=0.4 |
| GPU memory | 14.2 GB -> 6.4 GB at r=0.4 |
| 主性能 | NSCLC 98.3/98.9 Acc/AUC；BRCA 97.4/97.9；RCC 96.8/97.5；PANDA 94.8/95.6 |
| Random Sampling 对照 | 同样 r=0.4，但 NSCLC Acc 78.5，说明不是“少 token”本身带来提升 |
| 合并/剪枝消融 | Only Merge NSCLC Acc 96.4；Only Prune 95.7；Dynamic Merge+Prune 98.3 |
| retention ratio 扩展 | r=0.4 最优；r=0.3 NSCLC Acc 从 98.3 降到 90.4，说明过压缩会丢诊断 token |
| sparsity loss | 去掉 $\mathcal{L}_{sparse}$ 后 retained token 0.52、610 ms、Acc 96.9；完整模型 0.40、410 ms、Acc 98.3 |
| 数据规模 | NSCLC 993 WSIs；BRCA 938 WSIs；RCC 884 WSIs；PANDA 12,625 WSIs，任务聚焦 benign vs cancerous |
| patch 设置 | 256x256 patches；RCC 平均约 13,900 patches/slide at 20x |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|---|---|---|---|
| 1. WSI preprocessing | gigapixel WSI | tissue segmentation、background removal、256x256 patch extraction | patch set $\{x_i\}_{i=1}^N$ |
| 2. Feature encoding | patches | CONCH / Virchow2 encoder，optional positional embedding | token matrix $H^{(0)}$ |
| 3. Importance estimation | stage token embeddings | two-layer MLP + softmax | per-token saliency weights $\alpha_i^{(t)}$ |
| 4. Bipartite soft matching | tokens split into odd/even subsets A/B | cosine similarity + importance consistency utility | redundant token pairs selected for fusion |
| 5. Importance-weighted merging | selected token pairs | weighted average by $\alpha_i,\alpha_j$ | merged token representations |
| 6. Importance-guided pruning | merged + carry-over tokens | training: sigmoid soft gate; inference: hard Top-K | compact token set at stage t |
| 7. MIL aggregation | compressed tokens | attention-based MIL weighted pooling | slide embedding $z$ |
| 8. Slide prediction | $z$ | linear classifier + softmax，loss = CE + sparse regularizer | slide-level class prediction |

## 优缺点与还能做什么

### 优点

- 机制上比纯 pruning 稳：先 merge 冗余形态，再 prune 低重要度区域，减少弱监督下“一刀切丢证据”的风险。
- 消融证据比较完整：random sampling、random merging/pruning、only merge、only prune、retention schedules、encoder/backbone、sparsity loss 都覆盖了。
- r=0.4 同时提升 accuracy 和效率，支持“压缩也能去噪”的观点，而不是只做推理加速。
- 与 ABMIL/TransMIL 都能结合，说明它更像 token pre-aggregation module，而不是单一 MIL classifier。
- 可视化直接展示合并区域、剪枝区域、保留 tumor-rich 区域，和方法假设基本对齐。

### 局限 / 风险

- r=0.4 保留 40% token 与摘要 5-10x token reduction 的口径不完全一致，读数时应优先看表格中的 retention/FLOPs/time。
- importance network 仍靠 slide-level supervision 学 saliency，若病灶极小或 label noise 高，早期 saliency 估计可能误导后续剪枝。
- bipartite matching 只在交错子集间配对，复杂度友好，但可能错过空间/语义上更优的非相邻匹配；论文未充分展开匹配质量与 runtime 的权衡。
- 数据集都是分类任务，缺少 segmentation、retrieval、VQA、survival 等更细粒度任务来验证被压缩掉的 token 是否仍可支持其他用途。
- 表格报告多为同一预训练 encoder 与同一硬件设置下的结果，真实 WSI 系统还要考虑 patch extraction、I/O、缓存、multi-resolution pyramid 等端到端瓶颈。

### 还能做什么

- 做 lesion-size / tumor purity 分层分析，观察小病灶、低肿瘤含量和高度异质 slide 是否更容易被过压缩。
- 将 merge utility 加入空间邻近或 tissue-type constraint，避免高相似但空间语义不同的 patch 被合并。
- 把 retention ratio 变成 per-slide adaptive budget，而不是全局固定 r=0.4。
- 在 MIL 之外测试 WSI retrieval、report generation、pathology VQA、survival prediction，检查压缩 token 的通用性。
- 与 streaming / chunked attention / state-space MIL 联合，评估 token compression 和长序列建模是否互补。

## 阅读 Q&A 记录

- **Q: DTC-WSI 解决的核心瓶颈是什么？**
  A: WSI 会产生上万到数十万 patch token，MIL/Transformer 聚合有显著计算和显存成本；同时病理形态里 stroma、adipose、background、重复 tumor texture 很冗余。

- **Q: 为什么不直接用 token pruning？**
  A: 弱监督只有 slide label，早期 pruning 很容易删掉少量但关键的诊断区域。DTC-WSI 先把相似冗余 token 合并，再按 importance pruning，目标是少丢信息。

- **Q: importance network 如何进入合并？**
  A: merge utility 是 $\lambda \cos(i,j) - (1-\lambda)|\alpha_i-\alpha_j|$，即相似度高且 importance 接近的 token 更容易被合并，避免把高低 saliency token 混在一起。

- **Q: soft pruning 的意义是什么？**
  A: 训练时用 sigmoid gate 抑制 token 而不是硬删除，让梯度仍能回传到 importance network；推理时再 hard Top-K，得到确定的 token budget 和速度收益。

- **Q: 最能支持方法有效性的表是哪一张？**
  A: Table 4。Only Merge 和 Only Prune 都有提升，但都不如 Dynamic Merge + Prune；这证明性能不是来自任意 token reduction，而是来自两种压缩机制的互补。

- **Q: 最需要谨慎的结论是什么？**
  A: “压缩提升 accuracy”在这四个分类数据集上成立，但不能自动推广到所有病理任务；更细粒度下游任务可能更依赖被剪掉的局部结构。

## 📊 Citation Landscape

本目录的 Semantic Scholar 本地结果不可用：`semantic-scholar/detail.json` 为 `not_indexed`，`semantic-scholar/search.json` 返回 HTTP 429 rate limit。因此不填造 citation count、influential citation count 或 TLDR；以下仅基于本地 `full.md`、`paper-meta.json`、`source-url.json` 和正文 references 归纳引用脉络。

| 项目 | 当前记录 |
|---|---|
| Semantic Scholar TLDR | unavailable |
| Citation count | unavailable |
| Influential citation count | unavailable |
| Semantic Scholar status | detail: not_indexed; search: 429 rate limited |
| 官方来源 | PMLR v315, rahman26b |

### 参考文献脉络分组

| 主题 | 代表引用 | 与 DTC-WSI 的关系 |
|---|---|---|
| WSI / computational pathology 背景 | Gurcan 2009; Srinidhi 2021; Cui & Zhang 2021; El Nahhas 2025 | 说明 WSI gigapixel 与弱监督分析背景 |
| MIL baselines | ABMIL; CLAM; DSMIL; TransMIL; HIPT | DTC-WSI 的下游聚合与性能对照 |
| WSI efficiency / token-efficient MIL | PatchGD; MHIM-MIL; Longformer; 2DMambaMIL | 对比“处理长 token 序列”或“减少 token”的效率路线 |
| Token merging / pruning | ToMe; DynamicViT; importance-based token merging | 方法灵感来源，但 DTC-WSI 强调 WSI saliency 与弱监督稳定性 |
| Pathology foundation encoder | CONCH; Virchow2 | 提供 patch feature，DTC-WSI 主要改 token compression 而不是 encoder |
| 可解释与 morphology prototype | PANTHER; SI-MIL | 对照 slide representation 与可解释区域选择 |

## BibTeX

```bibtex
@inproceedings{rahman2026dtcwsi,
  title = {DTC-WSI: Dynamic Token Compression for Whole-Slide Images},
  author = {Rahman, Tawsifur and Tarkhan, Aliasghar and Chellappa, Rama and Baras, Alexander S.},
  booktitle = {Proceedings of Medical Imaging with Deep Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {315},
  pages = {3846--3865},
  year = {2026},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v315/rahman26b.html}
}
```
