# Whole Slide Image Analysis

这个 topic 关注全切片病理图像（WSI）分析：从"像病理学家一样在 gigapixel 组织图像中做空间感知推理与诊断"，到 MIL 聚合器设计、多模态生存预测、病理基础模型（PFM）的高效部署与公正评测。核心追问一以贯之：**在弱监督、超高分辨率、大量冗余的 WSI 上，模型到底学到了诊断相关的信号，还是 shortcut / 冗余噪声？**

## 论文列表（按子主题）

### 一、空间推理 · reasoning
| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [TissueWSI](./%5BArxiv%202026%5D%20TissueWSI/) | Arxiv 2026 | Tissue-Aware WSI Reasoning：像病理学家一样在组织空间上下文中推理，端到端从区域定位到诊断。 |

### 二、MIL 注意力机制 · mil-attention（挑战/纠正注意力）
| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [ACMIL](./%5BECCV%202024%5D%20ACMIL/) | ECCV 2024 | 把 WSI-MIL 过拟合归因到"注意力过度集中"，用 MBA（多分支注意力）+ STKIM（随机遮 Top-K）分散注意力。Top-10 占 85% 注意力——警示按注意力 Top-k 保留会漏大量判别 patch。 |
| [MHIM-MIL](./%5BICCV%202023%5D%20MHIM-MIL/) | ICCV 2023 | 反直觉遮掉高注意力"易 instance"、逼模型学"难 instance"；动量 Teacher-Student + 一致性损失，可插任意注意力 MIL。 |
| [ILRA-MIL](./%5BICLR%202023%5D%20ILRA-MIL/) | ICLR 2023 | ⚠️ 速览卡（OpenReview PDF 无法自动抓取，待补 PDF）。低秩性质：LRC 对比预训练 + 迭代低秩注意力 MIL（GAB+NLP），避免 Transformer O(n²)。 |

### 三、多模态生存预测 · multimodal-survival（对抗冗余）
| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [MOTCat](./%5BICCV%202023%5D%20MOTCat/) | ICCV 2023 | 最优传输（OT）驱动的 co-attention 融合病理+基因组，从全局结构一致性选信息 patch；UMBOT 降 OT 复杂度使其可跑。 |
| [PIBD](./%5BICLR%202024%5D%20PIBD/) | ICLR 2024 | 信息瓶颈治两种冗余：PIB（原型信息瓶颈，按风险等级原型筛 patch）+ PID（原型信息解耦，保护模态特有信息）。病理只留 25-40% patch 即保持性能。 |

### 四、高效 / 部署友好 CPath · efficient-cpath
| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [EAGLE (DL-Efficient-Pathology)](./%5BNat%20Commun%202026%5D%20DL-Efficient-Pathology/) | Nat Commun 2026 | CHIEF 选 25 tile + Virchow2 精提 → 一个 slide embedding。43 任务超 patch 聚合最多 23%，2.27s/slide（省 >99%）。top 5 tile 就超全部 tile mean pooling——"极端保留反而更好"的最强证据。 |
| [LitePath (Deployment-Friendly-CPath)](./%5BArxiv%202026%5D%20Deployment-Friendly-CPath/) | Arxiv 2026 | LiteFM（三 PFM 蒸馏，22.5M 参数）+ APS（自适应 patch 选择，均匀+注意力混合）；双轴削减 403.5× FLOPs，可跑 $249 Jetson 边缘设备。提出 D-Score 精度-效率标尺。 |

### 五、评测 / 可信度 · benchmark & reliability（现实检验）
| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [PathBench](./%5BArxiv%202025%5D%20PathBench/) | Arxiv 2025 | 首个防泄漏、全私有、全临床谱系的 PFM 基准（19 PFM × 64 任务 × 5 癌种）。Virchow2/H-Optimus-1 综合最优、视觉 FM>视觉-语言 FM、无普适赢家。 |
| [Confounders-Biomarker-Prediction](./%5BNat%20Biomed%20Eng%202026%5D%20Confounders-Biomarker-Prediction/) | Nat Biomed Eng 2026 | 批判性研究：H&E→分子 biomarker 预测被 grade/TMB/共依赖 biomarker 混杂，分层后 AUROC 大跌（ER 0.89→0.57）。提供分层+permutation 去混杂检验协议。 |

### 六、WSI MIL Baseline Set · FM-era 主对比方法集
> 为 foundation-model-era WSI MIL 实验确定的一组规模适中、覆盖面完整的主对比方法——每个 baseline 对应一种不同的**竞争解释**。完整设计文档见 CKMIL 写作仓 `related-work/wsi-mil-baseline-set.md`。

| 论文 | 会议 | 竞争解释 / 方法特点 |
|------|------|----------|
| [DeepSets](./%5BNeurIPS%202017%5D%20DeepSets/) | NeurIPS 2017 | **MeanPool 的理论根据**：排列不变集合函数 = $\rho(\sum\phi)$。"强 FM + 均值池化就够了"的 sanity control 理论出处。 |
| [TransMIL](./%5BNeurIPS%202021%5D%20TransMIL/) | NeurIPS 2021 | **contextual aggregation**：correlated MIL + self-attention 建 instance 相关（Nyström 近似 + PPEG）。"关键只是 instance 上下文交互"。 |
| [MambaMIL](./%5BMICCAI%202024%5D%20MambaMIL/) | MICCAI 2024 | **SSM 长序列**：Mamba 线性复杂度 + Sequence Reordering。"关键只是高效长序列建模"；GMMamba 的 base control。 |
| [RetMIL](./%5BMICCAI%202024%5D%20RetMIL/) | MICCAI 2024 | **retention 长上下文**：层次 retention（局部并行+全局串行），内存近常数。"更合适的 retention 聚合就够"。 |
| [PAMoE](./%5BCVPR%202025%5D%20PAMoE/) | CVPR 2025 | **pathology-aware MoE routing**：expert-choice 路由 + 组织原型监督，丢无关 patch。"关键只是组织异质性 + MoE 路由"。 |
| [GMMamba](./%5BICCV%202025%5D%20GMMamba/) | ICCV 2025 | **evidence selection + Mamba**：组内掩码 Mamba（IMM）+ 跨组超特征采样（CSS）。"关键只是去冗余/证据选择"。 |
| [MAMMOTH](./%5BICLR%202026%5D%20MAMMOTH/) | ICLR 2026 | **⭐ task-specific feature transformation**：多头 soft MoE 替换被忽略的线性层。装上后 mean/max pooling 超复杂 MIL——"真正瓶颈是特征变换而非聚合器"。CKMIL 最强竞争解释。 |
| [Shazam](./%5BArxiv%202025%5D%20Shazam/) | Arxiv 2025 | **⭐ 多层/多 FM 表示融合**：在线融合 5 FM × 3 层 + MoE 加权 + 在线蒸馏。**CKMIL "多层 FM 表示" 主线的最近竞争工作**（已占据"多层互补"novelty）。 |

**baseline set 合读洞察**（对 CKMIL/ReadySlide）：
- **MeanPool 是有理论根据的 sanity control**（DeepSets）——新方法超不过 FM+MeanPool 则增益来自特征而非聚合。
- **MAMMOTH 是最强竞争解释**：任务特定特征变换 > 聚合器选择，装上后 mean pooling 超复杂 MIL。CKMIL 若动特征变换必须正面对比。
- **Shazam 已占据"多层病理表示互补"novelty**：CKMIL 需把差异化落在"单 FM + 条件式 depth SELECTION"（选层 vs 全融），而非"多层有用"本身。
- **去冗余/证据选择反复出现**（PAMoE expert-choice / GMMamba IMM masking / PIBD Irr / EAGLE 25-tile），且多呈"适度保留"倒 U 形——retention 是贯穿主线的杠杆。

> 📌 **相关交叉引用**：[Revisiting-E2E-Slide-Supervision](../ckmil-re-attn-mil/%5BArxiv%202025%5D%20Revisiting-E2E-Slide-Supervision/)（Arxiv 2025，MRIS 稀疏梯度端到端 WSI MIL）由 `ckmil-re-attn-mil` topic 收录批读，与本 topic 的端到端/效率线密切相关。ABMIL（baseline set 的注意力经典基线）也在 `ckmil-re-attn-mil` 有交叉。

## 推荐阅读顺序

1. **建立 WSI 推理认知** → TissueWSI（病理学家式空间推理）。
2. **理解 MIL 注意力的不可靠** → ACMIL（注意力过度集中↔过拟合）+ MHIM-MIL（难 instance 更有用）+ ILRA-MIL（低秩高效注意力）。
3. **多模态融合与冗余对抗** → MOTCat（OT 全局匹配）+ PIBD（信息瓶颈 + 解耦）。
4. **高效/部署** → EAGLE（选 25 tile）+ LitePath（蒸馏 + APS，边缘部署）。
5. **现实检验** → PathBench（防泄漏 FM 评测）+ Confounders（去混杂，AUC 高≠学到生物学）。

## 横向数据流与研究机会

| 层级 | 输入 | 中间变化 | 输出 | 可追问的问题 |
|------|------|----------|------|--------------|
| patch 选择/保留 | gigapixel WSI 全部 patch | 注意力/OT/IB 原型/CHIEF 显著性 选子集 | 少量信息 patch | 保留哪些 patch？纯 importance 会漏覆盖/难样本吗？（ACMIL/EAGLE/LitePath/PIBD） |
| 聚合 | patch 特征 | 注意力/mean/OT/解耦 transformer | slide/patient embedding | 复杂聚合器真的比 mean pooling 好吗？注意力可靠吗？ |
| 多模态融合 | 病理 + 基因组 | 共有/特有解耦、OT 对齐 | 融合特征 | 加模态一定更好吗？（MOTCat/PIBD 显示 UCEC 上不一定） |
| 评测 | 模型预测 + 标签 + 混杂变量 | 分层 + permutation + 防泄漏 | 去混杂/去泄漏的排名 | AUC 高是学到生物学还是 shortcut？（Confounders/PathBench） |

## 统一阅读问题

- **Q1: WSI 分析里"保留少数 patch 反而更好"是普遍现象吗？**
  A: 是（ACMIL Top-10 占 85% 注意力、PIBD 留 25-40%、EAGLE top-5>全部）——因为诊断信号空间稀疏、大量 patch 冗余，保留高价值子集改善统计条件（偏差-方差权衡）。但**任务依赖**：形态学/长程上下文任务需更多/更全局。

- **Q2: MIL 的注意力权重可靠吗？能当 patch 重要性用吗？**
  A: 不完全可靠。ACMIL 证明注意力过度集中→过拟合；MHIM 证明高注意力=易 instance、难 instance 更有训练价值；EAGLE 证明端到端学的 ABMIL 注意力在细微任务上退化成近均匀。用预训练 task-agnostic 显著性（CHIEF）比逐任务学的注意力更稳。

- **Q3: 高 AUROC 说明模型学到诊断信号了吗？**
  A: 未必。Confounders 证明 H&E→biomarker 预测被 grade/TMB/共依赖 biomarker 混杂，分层后崩。PathBench 强调防数据泄漏。任何"压缩/模型保留了诊断信息"的声明都应做防泄漏 + 分层去混杂验证，而非只看聚合 AUC。

- **Q4: 病理基础模型该怎么选、怎么高效部署？**
  A: PathBench——Virchow2/H-Optimus-1 综合最优、视觉 FM>视觉-语言 FM、无普适赢家（看任务）。EAGLE/LitePath——用"粗筛+精提"或"蒸馏+patch选择"把算力降两个数量级，可跑边缘设备，几乎不掉精度。
