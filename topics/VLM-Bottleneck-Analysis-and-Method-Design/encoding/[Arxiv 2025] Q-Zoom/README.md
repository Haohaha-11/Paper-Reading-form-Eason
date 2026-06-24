# Q-Zoom: Query-Aware Adaptive Perception for Efficient Multimodal LLMs

## Paper Metadata

| 项目 | 内容 |
|------|------|
| **Title** | Q-Zoom: Query-Aware Adaptive Perception for Efficient Multimodal Large Language Models |
| **Authors** | Yuheng Shi, Xiaohuan Pei, Linfeng Wen, Minjing Dong, Chang Xu |
| **Venue** | arXiv 2025 |
| **Project Page** | https://yuhengsss.github.io/Q-Zoom/ |
| **References** | 79 |
| **Preliminary Work** | SD-RPN (ICLR 2026) [37] |

## One-Sentence Summary

Q-Zoom 提出一种查询感知的自适应高分辨率感知框架，通过轻量级的动态门控网络（Dynamic Gating Network）判断粗粒度特征是否足够，若不足则启动自蒸馏区域提议网络（SD-RPN）在单次 prefill 中精确定位任务相关 RoI 并重编码，配合连续时空位置编码和定向 Post-SFT 消除空间失准，在 Document & OCR 和 High-Resolution 场景下分别实现 2.52x 和 4.39x 推理加速，同时匹配甚至超越暴力高分辨率基线的峰值精度。

## Core Contributions

1. **Q-Zoom 整体框架** (Section III): 提出 query-aware 自适应两阶段框架，将感知保真度与二次计算代价解耦。通过 Dynamic Gating Network + SD-RPN 在单次 prefill pass 中同时消除 query-level 和 spatial 两种冗余。

2. **数据高效的优化策略** (Section III-B, III-C): 
   - Consistency-Aware Sample Generation: 沿分辨率单调递增轨迹评估模型响应，提取干净的二元路由标签训练门控网络
   - Tri-State Self-Distillation: 从 MLLM 内部交叉注意力中挖掘、去噪（消除 sink token + 三态标签分配）生成伪标签，完全免除人工标注和 RL

3. **时空对齐与定向微调** (Section III-D): 提出连续时空位置编码方案（Temporal Shift + Spatial Interpolation），配合 LLM-as-a-Judge 挖掘的硬样本进行定向 Post-SFT，消除裁剪 RoI 与全局上下文的感知断层

4. **全面实验验证** (Section IV): 在 LLaVA-1.5 (7B/13B)、Qwen2.5-VL (3B/7B)、Qwen3-VL-4B 以及 RL-trained ZwZ 模型上验证，在 Document & OCR 和 High-Resolution 基准上建立了新的 Pareto 前沿，且可作为即插即用模块叠加在 RL 推理模型之上

## Section Navigation

| 章节 | 文件 | 核心内容 |
|------|------|---------|
| Abstract & Overview | [00-abstract.md](sections/00-abstract.md) | 论文概述、Figure 1 感知范式对比、贡献总结 |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | 高分辨率感知瓶颈、现有方案局限、Q-Zoom 设计动机 |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) | 通用感知演进、Query-aware 感知三范式（training-free / SFT / RL） |
| 3. Methodology (I) | [03-methodology-1.md](sections/03-methodology-1.md) | Preliminaries、Dynamic Gating Mechanism |
| 4. Methodology (II) | [04-methodology-2.md](sections/04-methodology-2.md) | SD-RPN 的设计与 Self-Distillation、Spatio-Temporal Alignment + Post-SFT |
| 5. Experiments & Conclusion | [05-experiments.md](sections/05-experiments.md) | 主实验、消融研究、Pareto 前沿分析、结论与展望 |

## Key Numbers

| 指标 | 数值 |
|------|------|
| 评估 Benchmark 类别 | 3 (Document & OCR, High-Resolution & Vision-Centric, General QA) |
| 评估数据集数量 | 11 (DocVQA, InfoVQA, ChartQA, OCRBench, TextVQA, V*, MME-RealWorld, HR-Bench, MME, MMStar) |
| Backbone 模型数 | 5+ (LLaVA-1.5-7B/13B, Qwen2.5-VL-3B/7B, Qwen3-VL-4B, ZwZ variants) |
| 对比方法类别 | 4 (直接分辨率缩放 S^2, training-free ViCrop, RL-based AdaptVision/Thyme/DeepEyes, 前序工作 SD-RPN) |
| Document & OCR 加速比 (vs 4096-token baseline) | 2.52x (visual token 减少 53.0%) |
| High-Resolution 加速比 (vs 4096-token baseline) | 4.39x (visual token 减少 73.2%) |
| Qwen2.5-VL-7B Doc/OCR 平均提升 | +3.8% (with 0.81x throughput) |
| Qwen2.5-VL-7B High-Res 平均提升 | +8.1% |
| 最大 visual token 约束 | 576 (标准), 4096 (高分辨率场景) |
| SD-RPN 分支深度 R | 3 (all models) |
| Dynamic Gate 训练样本量 | 40K-60K (依模型而定) |
| Post-SFT 训练样本量 | ~7K hard samples |
| SD-RPN 伪标签数据量峰值 | 185K (Qwen), 152K (LLaVA) |
| GPU | Single NVIDIA RTX A6000 |

## Data Flow: Coarse → Gate → (RoI) → Fusion → Answer

```
┌─────────────────────────────────────────────────────────────────┐
│                       Q-Zoom Data Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Input]                                                          │
│    ├── Raw Image x_v                                              │
│    └── Text Query x_t (system prompt + user question)            │
│                                                                   │
│  [Stage 0: Coarse Encoding - Single Prefill]                      │
│    │                                                              │
│    ├── Encode x_v at coarse resolution → H_v^0                   │
│    ├── Forward through frozen backbone layers 1→B:                │
│    │     H_context^B = L_1:B([H_sys^0, H_v^0, H_user^0])        │
│    │                                                              │
│  [Stage 1: Dynamic Gating - Query-Level Routing]                  │
│    │                                                              │
│    ├── Gate G (layers B+1→B+R) processes H_context^B             │
│    ├── Extract last query token → Y_pred via LP_gate + sigmoid   │
│    ├── IF Y_pred < τ_gate:                                        │
│    │     → Route to Coarse Path: directly decode with coarse ctx │
│    │     → SKIP all below, generate answer                       │
│    └── IF Y_pred ≥ τ_gate:                                        │
│          → Trigger RoI Refinement Path (continue below)          │
│                                                                   │
│  [Stage 2: SD-RPN - Spatial RoI Localization]                     │
│    │                                                              │
│    ├── RPN (layers B+1→B+R-1) processes H_context^B              │
│    ├── Last query token + visual tokens → Q_RoI · K_v^T          │
│    ├── Dense heatmap → sigmoid → gaussian smooth → binarize      │
│    ├── Compute minimal bounding box → crop x_v_roi               │
│    └── Re-encode x_v_roi at high resolution → H_v_roi^0          │
│                                                                   │
│  [Stage 3: KV-Cache Reuse + Spatio-Temporal Alignment]            │
│    │                                                              │
│    ├── Reuse cached H_sys^B and H_v^B from Stage 0               │
│    ├── Forward only [H_v_roi^0, H_user^0] through layers 1→B    │
│    ├── Apply continuous spatio-temporal MRoPE:                    │
│    │     p_roi = Embed(t_src+δ, interpolated spatial coords)     │
│    └── Concatenate: [H_sys^B, H_v^B, H_v_roi^B, H_user^B]       │
│                                                                   │
│  [Output]                                                         │
│    └── Layers B+1→L → Final hidden states → Autoregressive decode │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Pros/Cons & Future Work

### Strengths

1. **单次 Prefill 完成全部感知**: 门控判断和 RoI 定位均在初始 prefill 阶段完成，避免了 training-free 方法的多轮 prefill 和 RL 方法的自回归 CoT 解码瓶颈
2. **即插即用 + 跨架构泛化**: 在 LLaVA、Qwen2.5-VL、Qwen3-VL 和 RL-trained ZwZ 等多种架构上均有显著提升，展现出极强的通用性
3. **数据效率极高**: SD-RPN 采用完全自监督的伪标签生成（仅 10K 伪标签即可获得 77.7% 均值），门控网络通过 consistency-aware 生成策略无需人工标注
4. **建立 Pareto 前沿**: 在显著降低 visual token 数量（53%-73%）的同时，匹配甚至超越暴力高分辨率基线的峰值精度
5. **可叠加于 RL 推理模型**: Q-Zoom 与 ZwZ 等 RL-trained thinking 模型正交互补，叠加后进一步提升绝对性能

### Weaknesses / Limitations

1. **依赖 MRoPE 进行空间对齐**: 连续时空位置编码方案依赖 MRoPE，LLaVA 系列不原生支持，因此 Post-SFT 阶段仅适用于 Qwen 系列
2. **LLaVA 低基础分辨率限制**: LLaVA-1.5 的基础分辨率仅 336x336，迫使门控网络几乎总是触发 RoI 分支，限制了吞吐量提升
3. **门控阈值的手动调节**: inference 时的 τ_gate 阈值需要人为设定，缺少自适应机制
4. **仅处理单 RoI**: SD-RPN 目前仅预测单块 RoI，对于需要多区域联合推理的任务可能受限
5. **训练流程较复杂**: 涉及三个独立训练阶段（SD-RPN、Post-SFT、Dynamic Gate），虽然每个阶段数据量不大，但整体 pipeline 仍有一定复杂度

### Future Work

1. 扩展 SD-RPN 支持多 RoI 同时定位，处理需要多区域协同推理的任务
2. 探索自适应门控阈值 τ_gate 的自动学习机制
3. 将连续时空对齐方案扩展到非 MRoPE 架构（如 LLaVA 系列）
4. 探索将 Q-Zoom 的思想应用于视频理解（时间维度的动态感知）
5. 进一步降低训练流程复杂度，考虑端到端联合优化

## Reading Q&A Record

| # | 问题 | 答案位置 | 解答 |
|---|------|---------|------|
| 1 | 为什么门控网络用 consistency-aware generation 而非简单的 low-res correctness 作为标签？ | Section III-B1 | 单次低分辨率结果的正确性受幻觉和歧义问题影响，标签噪声大。沿单调递增分辨率轨迹评估多个分辨率下的答案，保留"低分辨率错→高分辨率对"的有效转移样本，丢弃不稳定样本，确保 visual resolution 是决定正确性的唯一因素。 |
| 2 | Tri-state label assignment 中的三态分别是什么？为什么需要 ignore 标签？ | Section III-C2, Eq.12 | 三态为：前景 (1)、背景 (0)、忽略 (-1)。Fig.5 显示 attention score 中等范围的 token 处于高度模糊区，强制二元分类会导致噪声梯度。将高置信度前景 token 的 bounding box 内但不属于前景的 token 设为 ignore，避免不完整的对象激活错误惩罚网络。 |
| 3 | 为什么 SD-RPN 需要 sink token 过滤？ | Section III-C2, Eq.11 | 某些 visual token 尽管语义无关却积累了不成比例的 attention mass（sink token 现象），这些 token 在特征空间中 L2-norm 异常大。过滤后可获得更干净的空间注意力图用于伪标签生成。 |
| 4 | Temporal Shift (δ = min(H,W)) 的设计原理是什么？ | Section III-D, Eq.15 | MRoPE 中每个 token 有 (t, h, w) 三维位置编码。将 RoI token 的 t 坐标设为 t_src + δ，等于将高分辨率 RoI 投影到源图像上方的"辅助时间层"，避免与共享相同空间坐标的粗粒度 token 发生位置冲突，同时通过空间坐标插值保持语义定位。 |
| 5 | Q-Zoom 与 SD-RPN 的核心区别是什么？ | Section I (contributions), Section IV-B | SD-RPN 仅解决空间冗余（定位 RoI），缺乏 query-aware routing 和空间对齐。Q-Zoom 新增：1) Dynamic Gating Network 消除 query-level 冗余；2) 连续时空位置编码 + Post-SFT 恢复全局空间推理。在 Qwen 上效率提升 >30%，精度也有显著提升。 |
| 6 | 为什么在 LLaVA 上门控网络的吞吐提升有限？ | Section IV-B | LLaVA-1.5 基础分辨率仅 336x336，测试图片多为高分辨率，门控网络几乎普遍触发 RoI 分支，无法通过 bypass 获得加速。而 Qwen 原生支持动态分辨率，在简单 query 上可安全 bypass。 |
| 7 | 为什么不直接微调整个 LLM backbone 来做 RoI 预测？ | Section II-B | 全量微调 LLM backbone 计算代价高且存在灾难性遗忘风险，会损害基础通用能力。SD-RPN 仅 tune 复用的 R 层分支 + 投影头，冻结 backbone，避免了这一问题。 |

## Citation Landscape

### Reference Grouping by Topic

**MLLM Backbones & Architectures**:
- LLaVA 系列 [1, 14, 15, 49, 50], Qwen2-VL/Qwen2.5-VL/Qwen3-VL [2, 3, 20], InternVL [21]
- BLIP-2/InstructBLIP [13, 38], DeepSeek-VL [41, 51], Ovis [52]

**Vision Encoders**:
- ViT [16], CLIP [18], SigLIP [17, 39], Perception Encoder [40], NaViT [22]

**High-Resolution Perception**:
- S^2 [67], AnyRes [21, 49], Mini-Monkey [8], LLaVA-UHD [19], Honeybee [48]
- FastVLM [42], MG-LLaVA [44], Mini-Gemini [45], HyperVL [58], HIDE [53]

**Query-aware & Adaptive Perception**:
- Training-free: ViCrop [23], FoCus [24], ZoomEye [25]
- SFT-based: Token-Efficient VLM [56]
- RL-based: DeepEyes [4, 70], Thyme [5], Mini-o3 [6], AdaptVision [69], VisionThink [68]

**Thinking-with-Image & Latent Reasoing**:
- OpenAI Thinking with Images [26], Pixel Reasoer [8], ZwZ [36]
- MoNet [61], Latent Visual Reasoing [62]

**Visual Grounding & Attention Analysis**:
- Vision Function Layer [27], Attention Heads for Grounding [28]
- Sink Token: Vision Transformers Need Registers [65], Visual Attention Sink [66]
- Cross-attention analysis: Seeing but Not Believing [55], FoCus [24]

**Benchmarks**:
- DocVQA [29], InfoVQA [30], ChartQA [31], OCRBench [71], TextVQA [72]
- V* [33], MME-RealWorld [35], HR-Bench [34], MME [73], MMStar [74]

**SD-RPN Preliminary**:
- SD-RPN (ICLR 2026) [37]

---

*Batch reading created on 2026-06-24*
