[← 返回 README](../README.md)

# 3. Data Synthesis Pipeline

## 一、Preview

这是本文最核心的工程贡献。一个 6 阶段的自动化 pipeline，从开放数据集的图像-问题对出发，通过 VLM 蒸馏推理链 → LLM 提取可 grounding 对象 → SAM3 agentic grounding → box/point 对齐标注，最终产出同时适合 SFT 和 RL 的训练数据。整个过程无需人工标注。

---

## 二、原始文本

### Overview

We synthesize visually grounded thinking data from open-source datasets for counting and spatial reasoning: TallyQA (Acharya et al., 2018), Pixmo-Count (Deitke et al., 2024), VSR (Liu et al., 2023), MultihopSpatial (Lee et al., 2026), and SpatialMQA (Liu et al., 2025), with all test sets held out. Our goal is to identify the visual objects needed for correct thinking, obtain their image coordinates, and synthesize reasoning traces with explicit grounding annotations.

> 💡 **数据源概览**:
>
> | 数据集 | 任务类型 | Train Samples |
> |--------|---------|--------------|
> | TallyQA (AMT complex) | Counting | 7,197 |
> | PixMo-Count | Counting | 2,852 |
> | VSR | Spatial (yes/no) | 3,489 |
> | MultihopSpatial | Multi-hop spatial | 6,791 |
> | SpatialMQA | Spatial reasoning | 4,316 |
> | **Total (filtered)** | -- | **24,645** |
>
> 所有 test sets 被严格 hold out，确保 evaluation 无泄漏。

### Stage 1: Distilling visual thinking from VLMs

For each image-question pair, we prompt Qwen3-VL-Plus (Bai et al., 2025) to generate a thinking-mode response. We parse the final answer and keep examples whose predictions match the ground-truth answers. For examples not answered correctly in the first pass, we run a second pass with Qwen3.5-Plus (QwenTeam, 2026a) and keep examples that are answered correctly in either pass.

> 💡 **双模型两轮蒸馏策略**:
> ```
> Pass 1: Qwen3-VL-Plus → correct? → keep
>                      → wrong?   → Pass 2: Qwen3.5-Plus → correct? → keep
>                                                         → wrong?   → discard
> ```
> 设计动机：不同 VLM 在不同类型的 visual question 上有互补能力，两轮蒸馏最大化正确推理链的产出率。

### Stage 2: Extracting groundable objects

Given a correct reasoning trace, we use an LLM to identify the visual objects needed for the thinking process. These objects include answer objects, visible multiple-choice alternatives, spatial anchors, counted instances, and endpoints of spatial relations. Each object is represented by a name (e.g., "red car") and a disambiguating context (e.g., "in the back row"). The context separates visually or semantically similar instances, so two occurrences of "red car" can be distinguished by scene cues such as "near the entrance" or "in the back row".

> 💡 **为什么要 (name, context) 二元组？**
>
> 同一个图像中可能有多个 "red car"，如果只用 name，ground truth 中将出现不可区分的重复 object。disambiguating context 提供场景线索来区分它们：
> - "red car | near the entrance" vs "red car | in the back row"
> - Context 在后续 object router 中用于匹配 rollout 中的 grounding objects
>
> **Object 类型分类**:
> - Answer objects: 最终的答案实体
> - Visible alternatives: 多选题中的错误选项对应的物体
> - Spatial anchors: 空间关系中的参照物
> - Counted instances: 计数任务的目标物体
> - Endpoints of spatial relations: 空间关系的起点和终点

### Stage 3: Agentic visual grounding (核心工程)

The main challenge in data synthesis is to obtain accurate grounding for each extracted visual object. Direct prompting of VLMs does not produce RLE masks, and their predicted boxes are often noisy. SAM3 (Carion et al., 2026) can produce high-quality instance masks from simple noun prompts, but it is not well suited to complex context-dependent queries. We therefore use a SAM3-centered grounding agent powered by a VLM, adapted from the SAM 3 Agent in Carion et al. (2026).

> 💡 **核心矛盾的巧妙解决**:
>
> | 方案 | 优点 | 缺点 |
> |------|------|------|
> | VLM 直接输出坐标 | 理解复杂语义 | 坐标噪声大，无法输出 mask |
> | SAM3 直接接收文本 | 高质量 mask | 只能理解简单 noun phrase |
> | **Agent (VLM + SAM3)** | 语义理解 + 精确 mask | 需要迭代交互 |
>
> Agent 的精妙之处：**VLM 做"理解"（将复杂描述转成 SAM3 能懂的 noun phrase），SAM3 做"定位"（输出精确的 instance mask）。两者通过迭代闭环协作，互相修正。**

The agent uses four tool actions:
1. Call SAM3 with a short noun phrase and receive candidate instance masks with confidence scores
2. Verify rendered masks (using raw image, full-image mask overlay, and zoomed-in crop) to accept/reject candidates
3. Select final mask IDs from the current candidate set
4. Report no valid detection if needed

Importantly, the agent **cannot directly write coordinates**; all geometric supervision must be derived from selected SAM3 masks.

> 💡 **Agent 的四类工具操作**:
>
> ```
> ┌──────────────────────────────────────────────────┐
> │        SAM3 Agent Tool Actions                     │
> ├──────────────────────────────────────────────────┤
> │ 1. query(noun_phrase) → [(mask, confidence), ...] │
> │ 2. verify(mask_id) → accept / reject               │
> │ 3. select(mask_ids) → final masks                  │
> │ 4. report_no_detection() → fail                    │
> └──────────────────────────────────────────────────┘
> ```
>
> 一个关键的**硬约束**: agent 不能直接写坐标！这防止了 VLM 的坐标幻觉污染数据。所有 geometric supervision 都从 SAM3 mask 中推导出来，保证了精度。

For each object, the agent uses these tools in an iterative grounding loop:
1. Receives the raw image and the object (name + context)
2. Converts the name-context description into a SAM3-compatible noun phrase
3. If the initial prompt misses the target or returns confusing candidates, revises the noun phrase and tries again
4. When candidates are small/overlapping/ambiguous, invokes the verifier and re-renders accepted masks
5. Once sufficient evidence, selects final mask IDs; if no valid target, reports no detection

> 💡 **Agent 的迭代闭环**:
>
> ```
> (name, context) ──→ VLM rewrites to noun phrase
>                       │
>                       ▼
>                 SAM3 query
>                       │
>                       ▼
>                candidates received
>                       │
>              ┌────────┴────────┐
>              │                 │
>          good match       ambiguous/small/overlap
>              │                 │
>              ▼                 ▼
>           select()        verify()
>                              │
>                              ▼
>                         rerender masks
>                              │
>                              ▼
>                          select()
> ```
>
> **Fallback 策略**: 失败的 grounding 按顺序重试：Qwen3.6-Plus → Gemini-3-Flash（更强的 VLM）。仍然无法解决的 object 从 grounded object list 中移除，确保后续阶段不使用不可靠的 grounding。

### Stage 4: Mask to Box and Point Conversion

The selected masks are stored as RLE masks and used as the shared supervision signal for both grounding modes:
- **Box mode**: RLE mask → normalized bounding box `[x1, y1, x2, y2]` in [0, 1000] coordinate system
- **Point mode**: RLE mask → interior point farthest from mask boundary (distance transform)，确保即使是 nonconvex mask，点也在物体内部

> 💡 **Point 选择策略的精妙之处**:
>
> "choose the interior point farthest from the mask boundary" -- 这是 distance transform 的一个经典应用。对于任意形状的 mask，找到最大内切圆的圆心作为 point。这确保了：
> 1. Point 始终在 object 内部（而不是边界上）
> 2. 对 nonconvex mask 也有效（如环形、凹形物体）
> 3. Point 具有"代表性"（是 mask 的几何中心）

### Stage 5: Writing Box and Point Supervision

In the final annotation stage, we insert placeholder object tags into the validated reasoning text using **only the extracted object phrases and their contexts, without exposing coordinates to the annotation model**. We then fill in the coordinates from the SAM3 outputs. This design prevents the annotation model from hallucinating spatial values.

A single placeholder pass therefore produces two aligned SFT variants:
- `<obj> name phrase | [x1,y1,x2,y2] </obj>` for box supervision
- `<obj> name phrase | [x,y] </obj>` for point supervision

> 💡 **关键设计决策: 坐标注入而非坐标生成**
>
> ```
> Wrong approach:  LLM sees image + reasoning + object → generates coordinates
>                  ^-- LLM may hallucinate coordinates
>
> Right approach:  LLM sees image + reasoning + object → generates <obj> tags (w/o coords)
>                  SAM3 masks are mapped to tags → coordinates filled in post-hoc
>                  ^-- Coordinates are geometrically accurate, from SAM3
> ```
>
> 这保证了 box mode 和 point mode 的数据来自**同一套 SAM3 masks**，实验对比完全公平。

**Filtering**: We filter out rows whose tag-stripped annotated thinking differs substantially from the original thinking, as well as rows with malformed tags or highly repetitive reasoning.

### Dataset Statistics

| 指标 | 数值 |
|------|------|
| SFT reasoning traces | 19,909 |
| Grounding annotations (<obj> tags) | 107,613 |
| Unique grounded objects | 72,381 |
| Avg grounded objects per row | 3.64 |
| Avg annotations per row | 5.41 |

> 💡 **数据密度分析**: SFT annotation 密度 (5.41/row) 高于 grounded objects (3.64/row)，因为同一个 grounded object 在推理链中可能被多次引用（比如推理开始时提到一次，验证时再提一次）。

---

**Figure 3** (images/247da2ef197bed04ac71aae66a96a7b8b5a5495825726d24d4104b9f721ba273.jpg): 完整的 pipeline 流程图。

**Figure 4** (images/5a3f8ef92be4c6a372537b9a21912cadaf950e5307fe2cb634c17a980ea97939.jpg): Box mode 和 point mode 的合成数据示例对比。

**Figure 5** (images/37f4ac3ac46ef50089d7b6ac839b880c3e42a42e84d2c7c14a7f6bcdc7babf70.jpg): Grounding object router 的匹配流程图。

---

## 三、Summary

- **Pipeline 六阶段**: Reasoning distillation → Object extraction → Agentic grounding → Mask conversion → Tag annotation → Filtering
- **SAM3 Agent 是核心**: VLM 做语义理解 + SAM3 做精确定位，通过迭代闭环协作
- **对齐设计**: box 和 point 模式共享同一套 SAM3 masks，保证对比公平
- **坐标注入而非生成**: annotation model 只产生 placeholder tag，坐标从 SAM3 填入，防止 VLM 坐标幻觉
- **鲁棒性**: failed grounding 多轮重试 + 最终移除，确保监督信号质量
