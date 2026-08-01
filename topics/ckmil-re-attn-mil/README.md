# CKMIL & Re-Attention MIL 🔬

> 从 CKMIL（Cascaded Key Instance MIL）出发的注意力机制与 MIL 聚合器研究线。核心追问：**MIL 的"注意力权重"真的承载了空间/语义推理吗，还是只是另一个线性分类器？** 本主题覆盖四层：注意力结构设计（Swin/Attention Residuals/多层融合）→ MIL 聚合器有效性审查（Spatial Blindness/Instance-MIL Benchmark）→ 端到端训练范式（Revisiting-E2E/EXAONE P2）→ PFM 轻量适配（TAPFM/SiMLP/GigaPath-Flash）。

---

## 📚 阅读顺序

1. **打好注意力地基** → `Swin-Transformer`（层次化窗口注意力的鼻祖）、`Attention-Residuals`（残差连接的注意力化改造）、`Attentive-Multilayer-Fusion-ViT`（跨层注意力融合）
2. **审视 MIL 聚合器的有效性** → `Spatial-Blindness-WSI-MIL`（空间结构是否被优化利用？否定） + `SSL-Enhances-Instance-MIL`（SSL 特征下简单 MIL 是否够用？肯定）
3. **看懂端到端范式的边界** → `Revisiting-E2E-Slide-Supervision`（MRIS 稀疏梯度） ↔ `EXAONE-Path2-E2E`（HIPT 端到端）
4. **评估 PFM 适配的简单 vs 复杂** → `Simplify-Slide-Finetune-PFM`（均值池化 + 2 层 MLP 够了吗？） ↔ `SingleGPU-TaskAdapt-PFM`（ViT 内部注意力做聚合器）
5. **落地高效系统** → `GigaPath-Flash`（蒸馏 + LongNet 的高效病理 FM）

---

## 一、注意力机制基础 · attention-foundations

| 论文 | 会议 | 与本主题的关系 |
|------|------|----------------|
| [Swin-Transformer](./%5BICCV%202021%5D%20Swin-Transformer/) | ICCV 2021 | 层次化 Vision Transformer + 移位窗口注意力。CKMIL/MIL 方法的**底层 backbone**：窗口注意力的空间归纳偏置、相对位置偏置的组织空间编码、层次化特征的多尺度聚合——都是后续 MIL 注意力设计的结构前提。 |
| [Attention-Residuals](./%5BArxiv%202026%5D%20Attention-Residuals/) | Arxiv 2026 (Kimi Team) | 把残差连接重构为**可学习的注意力残差**（Full/Block Attention Residuals），本质是把"各层输出的等权累加"升级成"跨层动态路由"。对 MIL 的启发：bag 特征聚合的"等权求和"（mean pooling）能否用跨实例的 re-attention 替代？ |
| [Attentive-Multilayer-Fusion-ViT](./%5BArxiv%202026%5D%20Attentive-Multilayer-Fusion-ViT/) | Arxiv 2026 | 用 **multi-head cross-attention** 融合 ViT 多层 CLS token，在线性探测的约束下自适应利用不同层的语义信息。**MIL 的直接类比**：不同 patch 的"不同层特征"需要差异化融合——这正是 re-attention MIL 的核心操作。 |

## 二、MIL 聚合器有效性审查 · mil-effectiveness

| 论文 | 会议 | 核心发现 |
|------|------|----------|
| [Spatial-Blindness-WSI-MIL](./%5BArxiv%202026%5D%20Spatial-Blindness-WSI-MIL/) | Arxiv 2026 | **MIL 的空间盲**：即使把 WSI 的 patch 位置随机打乱（破坏所有空间拓扑），Transformer/Graph/SSM MIL 的性能几乎不变。论证：梯度竞争导致模型"选择不依赖空间位置"——复杂空间模块未被实际利用，等同于注意力池化。结论：目前的空间 MIL 方法是"名义上的空间感知"，优化过程本身抑制了空间结构的利用。 |
| [SSL-Enhances-Instance-MIL](./%5BArxiv%202025%5D%20SSL-Enhances-Instance-MIL/) | Arxiv 2025 | **Instance-based MIL + 自监督特征 = embedding-based MIL**：在 15 个数据集上的系统 benchmark 发现，用 SSL 预训练（DINOv2/iBOT）增强的 instance-level MIL（如 ABMIL 的注意力权重应用于 SSL 特征）能够匹敌或超越复杂的 embedding-based MIL。结论：**特征质量 > 聚合器复杂度**。 |

**合读洞察**：这两篇从两个独立方向指出同一件事——Spatial Blindness 说"用了的空间模块没被优化利用"（梯度竞争抑制空间依赖），SSL-Instance-MIL 说"简单 instance-MIL + 好 SSL 特征已经够用"（聚合器复杂度没有额外收益）。合起来暗示：**假设空间/重注意力 MIL 要真正生效，必须解决"优化过程会主动避开空间结构"的问题**（如梯度解耦、显式空间正则化），而不是只堆更复杂的注意力架构。

## 三、端到端训练范式 · end-to-end-wsi

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [Revisiting-E2E-Slide-Supervision](./%5BArxiv%202025%5D%20Revisiting-E2E-Slide-Supervision/) | Arxiv 2025 | 提出 **MRIS**（Multi-Resolution Instance Sampling）+ MHLA（Multi-Head Linear Attention），用稀疏梯度反向传播实现端到端训练（~10% token），配合 Adaptive Positive Suppression 防过拟合。与 EXAONE P2 构成"sparse gradient vs full end-to-end"的对立。 |
| [EXAONE-Path2-E2E](./%5BArxiv%202025%5D%20EXAONE-Path2-E2E/) | Arxiv 2025 | **HIPT 架构的端到端病理 FM**：用 hierarchical ViT + curriculum learning + memory-efficient training 在 10 个 benchmark 上验证。关键局限：消融缺失（curriculum/early exit/MTL 各贡献多少未知）、比 MRIS 更内存密集。 |

## 四、PFM 轻量适配 · pfm-adaptation

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [Simplify-Slide-Finetune-PFM](./%5BArxiv%202025%5D%20Simplify-Slide-Finetune-PFM/) | Arxiv 2025 | **SiMLP = 均值池化 + 2 层 MLP**：3 PFM × 6 task 上系统优于 ABMIL/DTFD-MIL/ACMIL/RRTMIL/DiffMIL，TCGA OncoTree +3.52%。核心论证：PFM 时代复杂 MIL 聚合器不仅不必要、还可能伤害泛化。**Mean >> Max 10+pp**，挑战 MIL"聚焦显著 patch"的根基假设。 |
| [SingleGPU-TaskAdapt-PFM](./%5BArxiv%202025%5D%20SingleGPU-TaskAdapt-PFM/) | Arxiv 2025 | **TAPFM**：用 ViT 内部 CLS 注意力作为无需学习的 MIL 聚合器，配合**分离计算图**的两阶段优化。单 H100 训练。局限：仅最后层注意力、无 detach 消融、AUC 提升 2-7pp（增量温和）。 |
| [GigaPath-Flash](./%5BArxiv%202026%5D%20GigaPath-Flash/) | Arxiv 2026 | **GigaPath-Flash + GigaTIME-Flash**：用 DINOv2 蒸馏 + LongNet 长序列预训练 + LoRA 微调，自称"单 GPU 训练的高效病理 FM"。亮点：诚实自评局限（LongNet 序列建模不准/FLOPs 可能更高/少数类别差）。 |

**合读洞察**：TAPFM（用 ViT 内部注意力做聚合）与 SiMLP（均值池化 + MLP）构成鲜明对立——一个说"更好的聚合靠 Transformer 内部机制"，一个说"更简单的聚合效果更好"。GigaPath-Flash 则从系统端补充了"如何廉价获得好特征"。三者共同指向：**PFM 下的 MIL 聚合器设计正从"越复杂越好"转向"特征质量为主 + 聚合器做稳健正则化"**。

---

## 🧭 对本课题（ReadySlide）的可复用结论

1. **SiMLP 的 mean pooling 必须成为 allocator 的 baseline**：如果内容感知的 patch 保留不能显著超过"全保留后均值池化"，则压缩合理性的论证会很弱。
2. **Spatial Blindness 的梯度竞争是 re-attention/重排序方法的通用障碍**：本文已证明"加空间模块 ≠ 模型会去用空间信息"——设计注意力-based MIL 时，必须显式做梯度解耦或空间正则化，否则注意力仍退化为普通池化。
3. **HER2 等局部信息任务是 intelligent retention 的最佳场景**（SiMLP 发现非线性 MLP 在小数据 HER2 上伤害性能）：均值池化不足的任务正是内容自适应保留能提供最大增益的地方。
4. **特征质量 > 聚合器复杂度**（SSL-Instance-MIL + SiMLP 的双重证据）：对压缩方法而言，这意味着"保留哪些 patch"比"怎样聚合保留的 patch"重要得多——allocator 的排序质量才是关键杠杆。

---

*每篇论文目录内含「批读格式」笔记：原文完整保留 + 内嵌 `Hao 批注`，README 含数据流（Mermaid）、关键数字、优缺点与阅读 Q&A。*
