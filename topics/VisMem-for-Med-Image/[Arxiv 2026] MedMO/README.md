# MedMO: Grounding and Understanding Multimodal Large Language Model for Medical Images

**Authors**: Ankan Deria\*, Komal Kumar\*, Adinath Madhavrao Dukre, Eran Segal, Salman Khan, Imran Razzak

**Affiliation**: Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)

**Venue**: arXiv 2026 (arXiv:2602.06965)

**Resources**: [Models](https://huggingface.co/collections/MBZUAI/medmo) | [GitHub](https://github.com/genmilab/MedMO) | [Project Page](https://genmilab.github.io/MedMO-Page)

---

## 一句话总结

MedMO 是一个基于 Qwen3-VL 构建的完全开源的多模态医学基础模型，通过四阶段渐进式后训练流程（大规模对齐、高分辨率微调、指令微调，以及基于 GRPO 的 RL 并采用边界框可验证奖励），在来自 **45 个多模态医学数据集的 2600 万+样本**上进行训练，在医学 VQA、文本 QA、报告生成和空间定位（grounding）任务上均达到 SOTA 表现，同时支持带边界框的空间定位。

---

## 核心贡献

1. **开源医学基础模型**：开发了 MedMO（4B 和 8B 两个变体），这是一个经过后训练的多模态 VLM，统一了放射学、病理学、眼科、皮肤科、CT、MRI、超声和手术视频等领域的视觉定位、临床推理和语言理解能力。

2. **大规模数据构建**：从 **45 个多样化的开源数据集**中收集了超过 **2600 万条多模态医学和生物医学样本**，涵盖多种成像模态和生物系统，并额外构建了一个基于开源显微镜图像的细胞检测评测基准。

3. **多阶段后训练流程**：设计了四阶段渐进式训练方案：(i) 在 18.5M 图文对上以 768x768 分辨率进行通用 SFT，(ii) 在 3M 样本上以 1280x1280 分辨率进行高分辨率 SFT 以引入 grounding 能力，(iii) 在 4.3M 多模态 QA/推理对上进行指令微调，(iv) 基于 GRPO 的 RL，并引入了新颖的**边界框可验证奖励**（结合 Hungarian 匹配的 GIoU + 归一化 L1，以及 FP/FN 惩罚项）。

4. **SOTA 结果**：MedMO-8B-Next 在 VQA 平均指标上超越 Fleming-VL-8B **+6.6%**，文本 QA 平均指标上超越 **+14.4%**，MIMIC-CXR CIDEr 上超越 **+6.7%**，Bacteria grounding IoU 上超越 **+47.0**。MedMO-4B-Next 与 8B 规模的 baseline 模型相比仍具有竞争力。

5. **全面的消融框架**：提供了开放、可复现的分阶段消融研究和边界框奖励分析，为未来的医学 MLLM 研究建立了基准和训练方案参考。

---

## 📖 批读导航

| Section | File | Description |
|---------|------|-------------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | 论文摘要及批读标注 |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | 研究动机、差距分析与核心贡献 |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) | 医学 MLLM 与多模态模型 grounding |
| 3. Methodology | [03-methodology.md](sections/03-methodology.md) | 四阶段训练流程、SFT、RL、边界框奖励 |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) | 实验设置、数据集、SOTA 结果、消融研究 |
| 5. Conclusion | [05-conclusion.md](sections/05-conclusion.md) | 总结、局限性与未来工作 |

---

## 关键数字

| 指标 | 数值 |
|--------|-------|
| 基础架构 | Qwen3-VL-8B-Instruct |
| 模型变体 | MedMO-4B, MedMO-4B-Next, MedMO-8B, MedMO-8B-Next |
| 训练数据总量 | **2600 万+样本**，来自 **45 个数据集** |
| 训练算力 | 64x AMD Instinct MI210 (64 GB)，**25 天** |
| Stage 1（通用 SFT） | 18.5M 图文对，768x768，BS=10，LR=1e-5，**225 小时** |
| Stage 2（高分辨率 SFT） | 3M 样本，1280x1280，BS=2，LR=8e-6，**155 小时** |
| Stage 3（指令微调） | 4.3M 指令对，BS=10，LR=5e-6，**110 小时** |
| Stage 4（RL/DAPO） | 300K 样本，每 prompt 生成 8 个响应，**98 小时** |
| VQA 平均分（MedMO-8B-Next） | **72.7%**（比 Fleming-VL-8B +6.6%） |
| QA 平均分（MedMO-8B-Next） | **60.1%**（比 Fleming-VL-8B +14.4%） |
| MIMIC-CXR CIDEr（MedMO-8B-Next） | **143.4** |
| Grounding 平均 IoU（MedMO-8B-Next） | **56.8%** |
| Bacteria IoU（MedMO-8B-Next） | **56.1**（比 Fleming-VL-8B +47.0） |

---

## 数据流：输入 → 中间表示 → 输出

```
[输入] 多模态医学图像（X 光、CT、MRI、超声、病理、OCT、眼底、手术视频等）
  + 文本查询（VQA 问题、临床提示、报告请求）
    │
    ▼
[Stage 1：通用医学 SFT] 18.5M 图文对 @ 768×768
  • 视觉编码器（ViT）→ 视觉-语言适配器（DeepStack）→ LLM 解码器
  • 任务：图像描述、VQA、通用多模态对齐
  • 输出：具备基础医学知识的 base 模型
    │
    ▼
[Stage 2：高分辨率医学图像 + Grounding SFT] 3M 样本 @ 1280×1280
  • 高分辨率专家标注的图文对
  • 引入边界框预测，实现空间定位能力
  • 任务：图像描述 + VQA + 监督式 grounding 信号
  • 输出：具备空间感知和定位能力的模型
    │
    ▼
[Stage 3：指令微调] 4.3M 指令-响应对
  • 医学 QA、推理、报告总结、检索
  • 将模型响应与人类风格的医学推理对齐
  • 输出：具备临床对齐指令遵循能力的模型
    │
    ▼
[Stage 4：带可验证奖励的 RL] GRPO/DAPO，300K 样本
  • 4 种奖励信号：标签准确率、边界框 GIoU、标签计数、soft-overlap 惩罚
  • Hungarian 匹配用于边界框分配
  • 边界框奖励 = clip(基础分 - 惩罚项)
  • 输出：MedMO（base）/ MedMO-Next（含 RL）
    │
    ▼
[输出] 文本响应 + 边界框坐标
  • 医学 VQA 答案、诊断报告、空间定位
  • 评测基准：VQA、文本 QA、报告生成（ROUGE-L、CIDEr、RaTE、Semb）、Grounding（IoU）
```

---

## 优缺点与还能做什么

### 优点

- **完全开源**：所有模型、数据集和训练方案均已公开发布。
- **多模态覆盖全面**：涵盖放射学、病理学、眼科、皮肤科、CT、MRI、超声和手术视频，覆盖范围远超大多数医学 MLLM。
- **原生视觉定位能力**：通过新颖的 GRPO 边界框奖励机制，具备原生边界框定位能力，区别于仅支持 VQA/图像描述的大多数医学 MLLM。
- **高效缩放**：MedMO-4B-Next 在多个基准上超越 Fleming-VL-8B，证明了即使在较小规模下也能实现强劲性能。
- **透明的消融分析**：完整的分阶段消融研究展示了每个训练阶段的贡献。
- **渐进式课程设计**：四阶段流程的设计提供了一条从通用对齐到细粒度空间推理的可扩展路线图。

### 局限 / 风险

- **灾难性遗忘**：分阶段训练会引入轻微的任务级性能波动（例如，Stage 1 在 MedTrinity 上有所提升，但在其他数据集上略有下降）。
- **IU-Xray 未达最优**：在 IU-Xray 报告生成上，Fleming-VL-8B 仍保持领先（CIDEr 198.6 vs MedMO-8B-Next 171.9）。
- **RL 训练开销**：Next 变体额外增加了 98 小时的 RL 训练，且在某些基准上 RL 带来的增益有限（例如，Qwen3VL-8B 即使不经过 RL 也在 QA 上取得强劲分数）。
- **RL 奖励组件的消融不充分**：仅展示了边界框奖励的消融；标签计数和 soft-overlap 惩罚的相对贡献未被单独分析。
- **仅限英语**：未对非英语临床场景进行显式的多语言评估。
- **25 天训练周期**：训练需要大量 GPU 资源（64x MI210），在一定程度上限制了小型实验室的可及性。

---

## 阅读 Q&A 记录

> **Q1**：为什么 MedMO 选择 Qwen3-VL 作为基础模型，而不是其他 VLM？
>
> MedMO 基于 Qwen3-VL-8B-Instruct 构建。作者未对基础模型的选择进行显式消融分析，但 Qwen3-VL 提供了原生的动态分辨率处理和 DeepStack 视觉-语言融合机制，这可能有助于医学 grounding 任务所需的多尺度特征对齐。Qwen3VL-8B 在文本 QA 上的强劲基线表现（平均 53.6%，接近 MedMO-8B 的 61.3%）也表明它是一个有能力的起点。
>
> > **Q1 追问**：更强的 VLM（如 InternVL3）会不会是更好的起点？
> >
> > InternVL3-8B 在 VQA 上表现不错（平均 57.4%），但在 DeepLesion 上 IoU 为 0.00，在 Bacteria grounding 上也接近零（0.7）（见表 3），说明其空间定位能力在医学检测任务上根本不可用。而 Qwen3VL-8B 在 NIH 上达到 16.4 IoU，在 Bacteria 上达到 9.16，表明其在微调之前就已经具备 emergent grounding 能力，是空间定位任务更合适的起点。

> **Q2**："MedMO" 和 "MedMO-Next" 之间有什么区别？
>
> "Next" 后缀表示模型经过了 Stage 4（使用 GRPO/DAPO 和边界框可验证奖励进行强化学习）。普通 MedMO-4B/8B 是 Stage 3 的检查点。有趣的是，在某些指标上（例如 MedQA，MedMO-8B 得分 84.3% vs Next 的 83.8%；MedMCQA，base 得分 65.0% vs Next 的 62.0%），base 变体反而表现更好，说明 RL 微调在提升 grounding 的同时可能会削弱某些文本 QA 能力。

> **Q3**：为什么边界框奖励使用 Hungarian 匹配而非更简单的 greedy 匹配？
>
> Hungarian 算法能在预测框与真值框之间找到全局最优的一对一匹配，这在图像中存在多个目标（如多个病灶、多个细菌细胞）时至关重要。Greedy 匹配可能产生次优的配对，导致 RL 训练中的信用分配错误。代价矩阵结合了加权 L1 距离（w=5）和 GIoU（1 - GIoU，w=2），优先考虑空间邻近度而非重叠度，这与检测评估指标判断定位质量的方式一致。

> **Q4**：MedMO 是否适合在真实临床环境中部署？
>
> 论文未报告临床验证或 FDA/CE 认证。虽然 MedMO 在基准测试中取得了强劲表现，但作者承认灾难性遗忘是一个局限性，并指出存在"轻微的任务级性能波动"。论文未讨论偏差/公平性评估、推理延迟或临床工作流集成。MedMO 主要定位为面向开放科学研究的 foundation model 研究项目，而非经过临床认证的诊断工具。

> **Q5**：Cell Benchmark Dataset 与现有的显微镜基准有何不同？
>
> MedMO 引入了一个基于开源显微镜图像（DeepCell、Bacteria）构建的 Cell 数据集，涵盖不同大小、形状和密度的细胞。关键创新在于，该数据集专门设计用于评估 VLM 在检测任务上的表现（基于 IoU），填补了当前医学 VLM 基准几乎只聚焦于 VQA 和文本 QA 而非空间检测精度的空白。这使得评估 VLM 是否能够正确"定位"目标而不仅仅是"回答"关于图像的问题成为可能。

---

## Citation Landscape

[Connected Papers: MedMO](https://www.connectedpapers.com/main/2602.06965)

MedMO sits at the intersection of several active research directions:

1. **Medical MLLMs**: Builds on and substantially improves over LLaVA-Med, HuatuoGPT-Vision, Med-Flamingo, BioMedGPT, GMAI-VL, Lingshu, and Fleming-VL. The key differentiator is the combined VQA + QA + grounding + report generation capability.

2. **Reinforcement Learning for Reasoning**: Adopts GRPO (from DeepSeekMath/DeepSeek-R1) and DAPO for preference optimization with verifiable rewards, extending the RLVR paradigm from math/CS (e.g., RLVR-tuned code models) into medical vision-language tasks.

3. **Visual Grounding**: Extends detection-oriented grounding (Grounding-DINO) and general-domain VLM grounding (Qwen2.5-VL) into the medical domain, validated on MedSG-Bench for sequential, multi-view, and referring expression grounding.

4. **Multi-modal Medical Datasets**: Leverages MedTrinity-25M as the cornerstone (18.5M pairs) and supplements with 26 additional datasets spanning report generation, VQA, and text QA.

[← 返回 README](README.md)
