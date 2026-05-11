# FOCUS: Knowledge-enhanced Adaptive Visual Compression for Few-shot Whole Slide Image Classification

**作者**: Zhengrui Guo; Conghao Xiong; Jiabo Ma; Qichen Sun; Lishuang Feng; Jinzhuo Wang; Hao Chen  
**机构**: HKUST; BICI; CUHK; Peking University  
**会议/期刊**: CVPR | **年份**: 2025  
**Semantic Scholar 年份字段**: 2024（本地 detail.json 如此记录；论文目录与 venue 按 CVPR 2025 处理）  
**链接**: [arXiv Abs](https://arxiv.org/abs/2411.14743) | [PDF](https://arxiv.org/pdf/2411.14743) | [Code](https://github.com/dddavid4real/FOCUS) | [Semantic Scholar](https://www.semanticscholar.org/paper/2dc14d75b94d8a3b5d81b794404aa7503cf87b06)

## 一句话总结

FOCUS 用 pathology foundation model 与 LLM 生成的病理语言先验做三阶段 adaptive visual token compression，先去除 WSI 全局冗余 patch，再按文本语义筛选诊断相关 token，最后用 neighbor-aware filtering 保留空间连续的差异性区域，从而提升 few-shot WSI 分类。

## 核心贡献

1. 将 pathology FM 从单纯 feature extractor 提升为 compression controller：用 CONCH patch embedding 的 sliding-window similarity 做全局冗余去除。
2. 将类别级病理描述 prompt 引入 token selection：LLM 生成核形态、细胞形态、组织结构等描述，文本-视觉 attention score 决定哪些 patch 更值得保留。
3. 提出 sequential visual token compression：按相邻 token 的 cosine similarity 逐级过滤局部冗余，同时尽量保留 WSI patch 序列中的空间连续性。
4. 用 CrossAgg 在压缩后的视觉 token 和 pathology prompt 之间做 multi-head attention 聚合，形成 slide-level prediction。
5. 在 TCGA-NSCLC、CAMELYON、UBC-OCEAN 的 4/8/16-shot 设置下对比 MIL 与 few-shot WSI baseline，并用模块、FM、LLM prompt 消融支撑设计选择。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：few-shot WSI 的 patch 冗余问题与 FOCUS 三阶段压缩主线 |
| [01 - Introduction](sections/01-introduction.md) | 背景、两个关键缺口、FOCUS 的 progressive adaptive compression 贡献 |
| [02 - Related Works](sections/02-related-works.md) | MIL/FSWL、pathology FM、vision-language CPath 的关系 |
| [03 - Methodology](sections/03-methodology.md) | WSI bag、prompt generation、KAVTC、SVTC、CrossAgg、training strategy |
| [04 - Experiments and Results](sections/04-experiments-and-results.md) | 数据集、实现、主结果、模块消融、FM/LLM 消融 |
| [05 - Conclusion](sections/05-conclusion.md) | 结论与未展开的运行效率/可解释性问题 |
| [06 - Acknowledgment / References](sections/06-acknowledgment-references.md) | 致谢与参考文献 |
| [07 - Supplementary Material](sections/07-supplementary-material.md) | 各数据集 prompt、不同 LLM prompt 分析、算法流程 |

## 关键数字

| 指标 | 数值 |
|------|------|
| WSI 分辨率背景 | 常超过 100,000 x 100,000 pixels |
| Few-shot 设置 | K=4, 8, 16（论文实现）；Introduction 也列出常见 K=1,2,4,8,16,32 |
| Patch size | 512 x 512 pixels |
| WSI scale | 20x 或 40x high-resolution WSI |
| Visual/text encoder | CONCH，feature dimension d=512 |
| LLM prompt 生成 | Claude-3.5-Sonnet 为主；UBC-OCEAN 消融含 ChatGPT-3.5Turbo、ChatGPT-4o、OpenAI-o1-mini、Llama3.1-405B |
| KAVTC window size | w=32 |
| Language-guided compression ratio | gamma=0.8 |
| SVTC base threshold | theta_base=0.7 |
| 训练 | AdamW，learning rate 1e-4，最多 80 epochs，early stopping |
| 评估 | balanced accuracy、F1、AUC；10-fold cross-validation |
| TCGA-NSCLC | LUAD 541 slides，LUSC 512 slides |
| CAMELYON | normal 577 slides，metastasis 341 slides |
| UBC-OCEAN | CC 98，EC 122，HGSC 221，LGSC 43，MC 43 slides |
| 4-shot TCGA-NSCLC | Balanced ACC 81.9%，AUC 91.5%，比 ViLa-MIL AUC 高 3.0 points |
| 4-shot CAMELYON | ACC/Balanced ACC 70.1%，比 ViLa-MIL 65.8% 高 4.3 points |
| 4-shot UBC-OCEAN | ACC/Balanced ACC 70.4%，AUC 91.1% |
| 16-shot AUC | TCGA-NSCLC 97.2%，CAMELYON 94.3%，UBC-OCEAN 96.7% |
| FM 消融最佳 | CONCH: 70.4%、77.3%、86.4% Balanced ACC on UBC-OCEAN 4/8/16-shot |
| LLM 消融最佳 | Claude-3.5-Sonnet: 86.4% Balanced ACC on UBC-OCEAN 16-shot |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. WSI bag construction | 一张 WSI X | foreground 切成 N 个 512x512 non-overlapping patches；CONCH/病理 FM 编码 | patch feature bag B=\{b_i\}, b_i in R^d |
| 2. Pathology prompt generation | class name / cancer subtype | LLM 回答“high-resolution WSI 中该类别的 essential visual characteristics”；文本编码器映射；拼接 learnable prompts | pathology prompt embedding T=[T^L; T^P] |
| 3. Global redundancy removal | FM patch features B | sliding window 内归一化特征并算 pairwise cosine similarity；动态阈值 tau_g=mean+std | 粗筛后的 visual token sequence |
| 4. Language-guided token prioritization | 粗筛 token + prompt embedding | text-to-visual attention 得到 token relevance r_i；按 min(M_max, gamma N') 选 top-k | 语义相关 token B_s |
| 5. Sequential visual token compression | B_s | 相邻 token cosine similarity；多阶段逐步提高阈值；保留与邻居不相似的 token | compressed token sequence B_c |
| 6. Cross-modal aggregation | B_c + T | multi-head attention 聚合文本和压缩视觉 token；LayerNorm + classifier | slide-level class probability P(Y|B_c,T) |
| 7. Training/evaluation | K-shot labeled WSIs | slide-level cross entropy；10-fold evaluation | few-shot WSI classification metrics |

## 优缺点与还能做什么

### 优点

- 抓住 few-shot WSI 的主要矛盾：标注 slide 少，但每张 slide patch/token 极多且诊断相关区域稀疏。
- 压缩链路有明确语义分工：FM 去视觉冗余，language prompt 找诊断相关性，neighbor-aware filtering 控局部连续性。
- 对比设置较公平：多数 baseline 同样使用 CONCH features，主差异集中在聚合/压缩机制。
- 消融覆盖了关键依赖：模块增量、不同 pathology FM、不同 LLM prompt。

### 局限 / 风险

- 论文强调分类性能，但没有系统报告压缩后 token 数、速度、显存、吞吐等 compression 直接收益。
- Token filtering 依赖 patch 序列顺序；如果 patch ordering 与空间邻接关系不一致，SVTC 的 neighbor-aware 假设会变弱。
- Prompt 来自 LLM，虽然有 LLM 消融，但没有充分讨论病理专家审核、prompt 偏差或跨中心术语差异。
- 主要实验是公开数据集与固定 6:2:2 split，真实低资源临床部署还需要验证 scanner/staining/domain shift。

### 还能做什么

- 报告每个阶段的 token retention ratio、运行时间、显存变化，并与分类收益关联。
- 做 token 可视化：展示被 KAVTC/SVTC 保留或丢弃的 patch 是否对应病理诊断区域。
- 测试 patch ordering、window size、gamma、theta_base 对不同癌种和 WSI 尺度的敏感性。
- 将 LLM prompt 改成病理专家可编辑模板或 retrieval-augmented pathology knowledge，降低 hallucination 风险。
- 在 external validation cohort 上测试 FOCUS 是否仍能保持 4-shot/8-shot 优势。

## 阅读 Q&A 记录

- **Q: FOCUS 的 compression 是不是图像压缩？**  
  A: 不是为存储或重建 WSI 做像素压缩，而是为 MIL/FSWL 分类做 visual token compression。被压缩的是 patch feature sequence，目标是减少冗余和突出诊断 token。

- **Q: 三阶段 compression 分别压掉什么？**  
  A: Global redundancy removal 压掉 FM embedding 上高度相似的全局/窗口内冗余；language-guided prioritization 压掉与病理描述弱相关的 token；SVTC 压掉与邻近 token 过于相似的局部冗余。

- **Q: 为什么 foundation model 不只是 feature extractor？**  
  A: FOCUS 用 CONCH feature 的 inter-patch similarity 来控制 token selection，并用 CONCH text/image embedding 支持 prompt 与 patch 的相关性计算。

- **Q: Language prompt 起什么作用？**  
  A: Prompt 把类别的核形态、细胞质、结构模式、间质特征等病理知识变成 text embedding，参与视觉 token relevance ranking，而不是只作为最终分类标签文本。

- **Q: Neighbor-aware filtering 为什么重要？**  
  A: WSI patch 有空间局部结构，直接全局 top-k 容易破坏连续诊断区域；SVTC 比较相邻 token 并保留差异性 token，试图在减少冗余和保留组织连续性之间折中。

- **Q: 结果最能支持论文主张的是哪部分？**  
  A: 主表说明 FOCUS 在 4-shot 最困难场景有稳定提升；Table 2 模块消融说明 Prompt、KAVTC、SVTC、CrossAgg 逐步增加 Balanced ACC；FM/LLM 消融说明 CONCH 和 Claude prompt 对图文对齐链路更有利。

## 📊 Citation Landscape

**Semantic Scholar 状态**: 本目录已有 `semantic-scholar/detail.json`、`references.json`、`recommendations.json`。以下数字全部来自本地 JSON；未使用联网补全。注意 detail.json 的 `year` 为 2024，但目录元数据和论文发表信息按 CVPR 2025 记录。

| 项目 | 数值 / 状态 |
|------|-------------|
| Semantic Scholar paperId | `2dc14d75b94d8a3b5d81b794404aa7503cf87b06` |
| DOI | 10.1109/CVPR52734.2025.01453 |
| ArXiv | 2411.14743 |
| TLDR | The knowledge-enhanced adaptive visual compression framework, dubbed FOCUS, is introduced, which uniquely combines pathology FMs with language prior knowledge to enable a focused analysis of diagnostically relevant regions by prioritizing discriminative WSI patches. |
| reference count | detail.json: 61；references.json 本地条目: 61 |
| citation count | 17 |
| influential citation count | 3 |
| recommendations | recommendations.json 本地 `recommendedPapers`: 100 条 |
| fallback | 无需 fallback；三类文件均可读取。年份字段与 CVPR 2025 目录不一致处已显式标注。 |

### 参考文献分组（按本地 Semantic Scholar references）

| 主题 | 高相关参考文献 |
|------|----------------|
| Few-shot WSI / prompt MIL | FAST; MSCPT; TOP-MIL; ViLa-MIL; Pathology-knowledge Enhanced Multi-instance Prompt Learning |
| Pathology foundation models | CONCH; UNI; Prov-GigaPath; Virchow; CHIEF; GPFM; PLIP |
| MIL / long sequence WSI | CLAM; TransMIL; DS-MIL; DTFD-MIL; WiKG-MIL; MambaMIL |
| Vision-language / multimodal CPath | HistGen; mSTAR; multimodal knowledge-enhanced WSI foundation model; pathology report / image-caption pretraining works |
| Datasets / clinical context | CAMELYON17; TCGA-related WSI analysis; UBC-OCEAN ovarian cancer subtyping context |

### 推荐继续读（按本地 recommendations 与引用关系）

1. **Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification** - recommendation 中与 few-shot WSI + textual guidance 最贴近的后续方向。
2. **Multimodal Model for Computational Pathology: Representation Learning and Image Compression** - recommendation 中直接连接 representation learning 与 compression 的 CPath 工作。
3. **MLLM-HWSI: A Multimodal Large Language Model for Hierarchical Whole Slide Image Understanding** - 可对照 FOCUS 的 prompt 级语言先验和更强 MLLM 式 WSI 理解。
4. **FAST: A Dual-tier Few-Shot Learning Paradigm for Whole Slide Image Classification** - FOCUS 引用的 few-shot WSI 基线之一。
5. **CONCH: A vision-language foundation model for computational pathology** - 理解 FOCUS 为什么用 CONCH 同时编码视觉 patch 和文本 prompt。

## BibTeX

```bibtex
@inproceedings{guo2025focus,
  title={FOCUS: Knowledge-enhanced Adaptive Visual Compression for Few-shot Whole Slide Image Classification},
  author={Guo, Zhengrui and Xiong, Conghao and Ma, Jiabo and Sun, Qichen and Feng, Lishuang and Wang, Jinzhuo and Chen, Hao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
