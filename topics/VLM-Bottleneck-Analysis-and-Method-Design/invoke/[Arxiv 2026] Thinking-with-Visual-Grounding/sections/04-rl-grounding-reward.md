[← 返回 README](../README.md)

# 4. Reinforcement Learning with Grounding Reward

## 一、Preview

本节是本文的方法学核心。从 grounding tag 的解析格式出发，到 object router 的匹配策略，再到 box IoU 和 point F1 两种 grounding quality 的计算，最后到与 answer reward 的联合归一化。整个 reward 设计的目标是：在 RL 中既优化答案正确性，又直接监督 grounding 质量。

---

## 二、原始文本

### Grounding Tag Parsing

In visually grounded thinking, a valid tag must have the form `<obj> name phrase | coordinates </obj>`.

The coordinate format is mode-specific:
- **Box mode**: expects `[x1, y1, x2, y2]` with $x_1 < x_2$, $y_1 < y_2$
- **Point mode**: expects `[x, y]`

Coordinates must fall within the [0, 1000] image coordinate system. A single tag may contain multiple coordinates separated by semicolons, as one object can refer to multiple instances (e.g. "birds in the sky" corresponds to several birds).

> 💡 **Tag 格式规范**:
>
> | 属性 | 规范 |
> |------|------|
> | Tag 边界 | `<obj> ... </obj>` |
> | 分隔符 | `name phrase` 和 `coordinates` 之间用 `\|` |
> | 坐标范围 | [0, 1000] 归一化坐标 |
> | Box 约束 | `x1 < x2` 且 `y1 < y2` |
> | 多实例 | 坐标间用 `;` 分隔 |
> | Point 约束 | 单点 `[x, y]`，应在物体内部 |

### Grounding Objects Routing

The grounding reward is computed between model-generated grounding objects and the ground-truth grounding objects saved in the data. Each grounding object in the data stores:
- A name phrase
- A disambiguating context
- Geometric supervision (RLE masks)

The model, however, may name the same object with different wording, and the same name phrase can refer to multiple distinct objects in the image. We therefore use a **VLM grounding object router** before scoring grounding quality.

> 💡 **Object Router 的三个核心问题**:
>
> 1. **Name mismatch**: 模型可能用不同措辞描述同一个 object（"the red car" → "the automobile on the left"）
> 2. **Name ambiguity**: 同一名称可能指图像中的不同实例（两个 "red car"）
> 3. **Context matching**: 需要通过 disambiguating context 区分同名的不同实例
>
> **Router 的设计选择**:
> - 轻量 VLM: Qwen3.5-4B（保证 RL 训练效率）
> - 输入: image + (gt_name, gt_context) + [list of model-generated objects]
> - 输出: 与 gt_object 匹配的 model-generated objects 子集
> - 多个 model objects 匹配同一 gt object 时，只保留最早出现的那个

### Box Grounding Quality (IoU-based)

Each saved object $i$ is associated with a set of ground-truth boxes $G_i$. After grounding object routing, let $P_i$ denote the set of boxes generated for the matched generated grounding object.

We treat each set of boxes as a union of regions and compute their intersection-over-union (IoU):

$$

\mathrm{IoU}_i = \frac{I_i}{U_i}

$$

where $I_i$ is the area covered by both $P_i$ and $G_i$, and $U_i$ is the area covered by either $P_i$ or $G_i$.

If no model-generated grounding object is matched to ground-truth object $i$, we set $\mathrm{IoU}_i = 0$.

**Final box grounding quality** = mean score over all $T$ ground-truth objects.

> 💡 **Box Reward 的设计细节**:
>
> - **Multi-box handling**: 多个 box 的 union 作为整体算 IoU。只有当生成 boxes 的 union 正好等于 GT boxes 的 union 时，才得满分
> - **Equal weight per object**: 每个 ground-truth object 权重相同，不论它包含几个 box
> - **Missing penalty**: 如果 rollout 中没有匹配到某个 gt object，该 object 的 IoU 直接为 0
> - **连续信号**: IoU 随重叠程度平滑变化，提供密集的优化梯度

### Point Grounding Quality (F1-based)

Let $M_i$ be the set of ground-truth masks for object $i$, and $P_i$ be the set of points from the matched rollout grounding object.

We form a **one-to-one assignment** between generated points and ground-truth masks:
- A point can be assigned to a mask only if it lies inside that mask
- This constraint prevents duplicate points from receiving repeated credit for the same object instance

For each object:

$$

\mathrm{TP}_i = \text{number of masks matched by the assignment}

$$

$$

\mathrm{FP}_i = |P_i| - \mathrm{TP}_i, \quad \mathrm{FN}_i = |M_i| - \mathrm{TP}_i

$$

$$

F1_i = \frac{2\mathrm{TP}_i}{2\mathrm{TP}_i + \mathrm{FP}_i + \mathrm{FN}_i}

$$

If no rollout grounding object is matched to ground-truth object $i$, set $F1_i = 0$.

**Final point grounding quality** = mean over all supervised targets.

> 💡 **Point Reward 的几点关键设计**:
>
> - **One-to-one assignment**: 每个 generated point 最多对应一个 mask，防止同一点被重复计分
> - **Inside-mask constraint**: 点必须落在 mask 内部才计入 TP，否则是 FP
> - **F1 而非 accuracy**: 平衡 precision（不要多生成）和 recall（不要漏掉 instance）
> - **离散信号**: F1 只有点进出 mask 时才会变化，在 mask 内部移动不影响分数

### Remarks: Box vs Point Reward 的本质对比

The point grounding quality can be viewed as a **discrete analogue** of the box grounding quality.

| 维度 | Box Reward (IoU) | Point Reward (F1) |
|------|-----------------|-------------------|
| 度量对象 | 区域重叠 (spatial overlap) | 实例匹配 (instance matching) |
| 信号类型 | 连续 (continuous) | 分段常数 (piecewise constant) |
| 密度 | 密集 (dense feedback) | 粗粒度 (coarse feedback) |
| 优化难度 | 较易 | 较难 |
| 所鼓励的 evidence | 相同（都鼓励 grounding 正确的视觉对象） | 相同 |

Point mode 的这种离散性使 reward 更难优化：在 mask 内部任意移动 point 不改变分数，crossing a mask boundary 才突然改变。这解释了为什么 point grounding reward 在实验中带来的提升不如 box grounding reward 稳定。

> 💡 **核心洞察**: Box IoU 和 Point F1 虽然在鼓励的视觉 evidence 方向一致，但 feedback 的密度和连续性不同。这在实验中导致 box reward 更有效，尤其在空间推理（需要精确的 box extent）上。

### Unmatched Grounding Objects: 为什么不做惩罚？

We **intentionally do not penalize unmatched grounding objects** in the rollout. The grounding objects extracted by the data synthesis pipeline are not a complete enumeration of all visual cues. During thinking, the model may identify additional visual evidence that is useful and reasonable to ground. Therefore, unmatched rollout grounding objects neither increase nor decrease the grounding quality.

We only apply a **hard-coded cap** on the number of grounding tags to prevent the model from over-emitting them.

> 💡 **设计哲学**: 开放式 grounding vs 封闭式 grounding
>
> - **封闭式**: 只允许 grounding pipeline 提取的 objects，额外的视为错误 → 但这过于严格，可能惩罚合理的探索
> - **开放式 (本文选择)**: 不惩罚额外的 grounding，只对匹配上的做质量评估 → 鼓励模型发现 pipeline 未提取的视觉线索，但通过 cap 防止滥用
>
> 这个选择体现了对 RL 探索行为的宽容和对数据不完整性的认识。

### Final Reward

For each rollout $i$, the total reward includes:
- **Dense grounding reward**: $r_i^{\text{ground}}$ (box IoU or point F1)
- **Sparse response-level rewards**:
  - $r_i^{\text{ans}}$: answer correctness
  - $r_i^{\text{think}}$: thinking-format reward (checks `<think>...</think>` and `\boxed{}`)
  - $r_i^{\text{gfmt}}$: grounding-format reward (checks valid `<obj>...|...</obj>` tags)
  - $r_i^{\text{trunc}}$: truncation penalty (-1 if truncated, 0 otherwise)

Because dense grounding reward and sparse rewards have different scales, we **normalize them separately**:

$$

$R_{i}^{{\text{base}}$} = w_{\text{ans}} r_i^{\text{ans}} + w_{\text{think}} r_i^{\text{think}} + w_{\text{gfmt}} r_i^{\text{gfmt}} + r_i^{\text{trunc}}

$$

$$

R_i = \mathcal{N}_{\mathcal{B}}(R^{\text{base}})_i + w_{\text{ground}} \mathcal{N}_{\mathcal{B}}(r^{\text{ground}})_i

$$

where $\mathcal{N}_{\mathcal{B}}(\cdot)$ is batch-wise normalization over batch $\mathcal{B}$.

**Hyperparameters**:
- $w_{\text{ans}} = 1.0$
- $w_{\text{ground}} = 0.5$
- $w_{\text{think}} = w_{\text{gfmt}} = 0.1$

> 💡 **Reward 设计的三个关键选择**:
>
> 1. **分别归一化**: $R^{\text{base}}$ 和 $r^{\text{ground}}$ 先各自 batch-normalize，再线性组合。这防止了量级差异导致的某一项主导整个 reward。
>
> 2. **Grounding weight = 0.5**: grounding 对 total reward 的贡献约为 answer 的一半。这是一个工程权衡：既要充分激励 grounding，又不能让其压倒 answer correctness。
>
> 3. **Format rewards 权重很小 (0.1)**: 格式正确性只是"门槛"，不应该主导优化方向。主要优化信号来自 answer 和 grounding。

---

## 三、Summary

- **Reward 双支柱**: Answer correctness (稀疏, scale 大) + Grounding quality (密集, scale 小) → 分别归一化
- **Box Reward**: IoU-based, 连续信号, 平滑优化
- **Point Reward**: F1-based, 离散信号, 更难优化
- **Object Router**: VLM-based 匹配，解决 name mismatch 和 name ambiguity
- **设计哲学**: 不惩罚额外 grounding，只对匹配上的评估质量
- **Weight**: answer = 1.0, grounding = 0.5, format = 0.1
