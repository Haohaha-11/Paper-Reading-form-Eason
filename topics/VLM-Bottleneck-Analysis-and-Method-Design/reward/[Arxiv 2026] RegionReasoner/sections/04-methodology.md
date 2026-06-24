[← 返回 README](../README.md)

# 4. RegionReasoner

## 一、Preview

方法论部分分为四个子节：
1. **4.1 Pipeline Formulation**: 端到端流程形式化（输入/状态/策略/动作空间/记忆更新）
2. **4.2 RegionReasoner Model**: 模型架构、4-tag 结构化 I/O、reference-grounded thinking、global--local consistency
3. **4.3 Reward Functions**: 三条奖励线的形式化定义（base + reference citation + global--local consistency）
4. **4.4 Training**: GRPO 训练流程、advantage estimation、总损失

---

## 二、原始文本

### 4.1 Pipeline Formulation

**Inputs and state.** At turn t, the agent observes the image I, the current textual query q_t, an optional set of reference boxes B_t^ref = {[x1^(k), y1^(k), x2^(k), y2^(k)]} (propagated or newly provided), and a memory M_{t-1} that stores structured outputs from previous turns. We serialize B_t^ref and M_{t-1} into the prompt to make them available to the model.

**Policy and action space.** RegionReasoner is an auto-regressive VLM policy π_θ that generates a structured text action composed of four tagged blocks y_t = (s_t, f_t, h_t, a_t) with tags <scene>, <focus>, <think>, <answer>. Let y_t = (w_{t,1}, ..., w_{t,N_t}) denote the token sequence for the whole action; then:

```
π_θ(y_t | I, q_t, B_t^ref, M_{t-1}) = ∏_{n=1}^{N_t} π_θ(w_{t,n} | I, q_t, B_t^ref, M_{t-1}, w_{t,<n})
```

Constrained decoding enforces the tag schema and JSON validity for <answer>, while allowing free-form natural language in <scene>, <focus>, and <think>.

> 💡 **Pipeline 形式化解读**:
> - **状态定义**: s_t = (I, q_t, B_t^ref, M_{t-1})，其中 B_t^ref 和 M_{t-1} 被序列化到 prompt 中
> - **动作空间**: y_t = (s_t, f_t, h_t, a_t) 四个 tagged block，自回归生成
> - **约束解码**: 强制 tag schema 和 JSON 有效性，<answer> 必须合法 JSON，<scene>/<focus>/<think> 允许自由自然语言

**Turn update and termination.** After decoding finishes, we parse a_t to obtain task outputs (e.g., 2D boxes or points) and update the memory:

```
M_t = M_{t-1} ∪ {(s_t, f_t, h_t, a_t)}
```

A multi-round episode consists of T turns. The per-turn reward R(t) is computed from (s_t, f_t, h_t, a_t) and aggregated across turns.

> 💡 **关键设计 — 记忆更新**: 每轮保存的不只是最终输出 a_t，而是完整的结构化轨迹 (s_t, f_t, h_t, a_t)。这意味着后续轮次可以访问前轮的全局场景描述、局部区域描述和推理过程，支持更丰富的跨轮语义交互。

**Compact notation for the loop:**
```
(s_t, f_t, h_t, a_t) ~ π_θ(· | I, q_t, B_t^ref, M_{t-1})
M_t ← M_{t-1} ∪ {(s_t, f_t, h_t, a_t)}
```

---

### 4.2 RegionReasoner Model

**Unified perception--reasoning backbone.** RegionReasoner extends the unified perception--reasoning framework of VisionReasoner to a multi-round setting, where each turn emits a structured and verifiable trajectory. The model is initialized from a large VLM backbone and performs chain-of-thought reasoning purely in text, while remaining explicitly grounded to image regions through serialized bounding-box references. Each turn-t output is organized into four tagged blocks: a global scene caption s_t (<scene>), a localized caption f_t tied to a provided reference box (<focus>, optional), a reasoning trace h_t (<think>), and a JSON answer a_t (<answer>). Constrained decoding with schema and tag guards ensures format validity, supports automatic post-hoc parsing, and prevents untagged content from leaking into <answer>.

> 💡 **4-tag 架构解读**:
> | Tag | 内容 | 是否必需 | 约束 |
> |-----|------|---------|------|
> | `<scene>` s_t | 全局场景描述 | 是 | 自由自然语言 |
> | `<focus>` f_t | 参考框内局部描述 | 否 (有 ref 时) | 自由自然语言 + 序列化坐标 |
> | `<think>` h_t | 推理过程 | 是 | 自由自然语言 + 必须显式引用空间关系和参考坐标 |
> | `<answer>` a_t | JSON 定位输出 | 是 | 约束解码确保 JSON 合法性 |

**Reference-grounded thinking.** To improve verifiability and reduce free-form hallucination, RegionReasoner requires that reasoning must cite evidence. When a query specifies references, the prompt encodes the set B_t^ref in a canonical textual form and instructs the model to reason with verbatim coordinate mentions inside <think>. The same coordinates are injected in q_t so attention aligns with the intended regions across turns. During decoding, h_t must explicitly reference the used boxes and, when relevant, name spatial relations (e.g., "to the right of bbox [x1,y1,x2,y2]"). This design yields a causal chain from evidence to conclusion that is parsable into cited coordinates S(h_t) and directly comparable to B_t^ref, enabling automatic grounding checks and precise credit assignment in RL. In multi-round interaction, previously cited boxes can be re-used or refined; the explicit citation acts as a stable interface across turns, which improves temporal coherence of the reasoning trajectory and curbs region drift.

> 💡 **机制拆解 — Reference-Grounded Thinking 的完整逻辑**:
>
> **输入侧**:
> 1. B_t^ref 中的坐标被同时注入到 prompt 和 q_t 中
> 2. 模型被告知需要显式引用这些坐标
>
> **生成侧**:
> 1. h_t 必须包含具体的坐标引用 (e.g., "target is above bbox [100,200,300,400]")
> 2. 坐标可被解析为 S(h_t)
>
> **验证侧**:
> 1. S(h_t) 与 B_t^ref 直接可比 —— 检查是否引用了正确的参考框
> 2. 是否存在不在 B_t^ref 中的坐标 —— 幻觉检测
>
> **效果**:
> - 证据→结论的因果链可追踪
> - 精确的 credit assignment（知道模型用了哪个参考框）
> - 跨轮稳定接口（显式引用作为前轮信息的"锚点"）
> - 抑制 region drift（不会在后续轮次中悄悄"漂移"到错误的区域）

**Global--local semantic consistency.** Iterative reasoning often breaks down when global descriptions and local evidence diverge; to prevent this, RegionReasoner jointly produces s_t (global) and f_t (localized to the reference) before generating h_t, and then enforces that the semantics of s_t and f_t are reflected within h_t. Concretely, a lightweight deterministic pipeline extracts keyword sets κ(s_t), κ(f_t), and κ(h_t) (lowercasing, stop-word removal, lemmatization, and a noun/object filter). We later compute asymmetric overlaps Ov(s_t, h_t) and Ov(f_t, h_t) as part of the reward (Sec. 4.3), pushing the model to propagate entities and relations from the global and local captions into the reasoning itself. Making <think> the alignment nexus -- rather than correcting only at the final answer -- yields finer-grained RL signals, better consistency across turns, and improved spatial reasoning, especially when h_t is encouraged to include localization lexicon (e.g., left/right/inside/overlap/next to) together with explicit box mentions.

> 💡 **机制拆解 — Global--Local Consistency 为什么有效**:
>
> **关键词提取流水线**:
> ```
> 原始文本 → lowercasing → stop-word removal → lemmatization → noun/object filter → κ(·)
> ```
>
> **对齐信号流向**:
> ```
> κ(s_t) ──→ Ov(s_t, h_t) ──→ R_cons (第一部分)
> κ(f_t) ──→ Ov(f_t, h_t) ──→ R_cons (第二部分)
> ℓ(h_t)  ──→              ──→ R_cons (第三部分: 空间/比较/定位词)
> ```
>
> **为什么在 <think> 层面做对齐而非在 <answer>?**
> - 更细粒度的 RL 信号（不只是最终输出是否正确，而是推理过程是否语义一致）
> - 跨轮一致性更好（<think> 中的实体锚定到后续轮次）
> - 在空间线索弱时尤其有帮助（通过关键词重叠保持话题聚焦）

**Task output without extra heads.** Detection and segmentation are expressed directly through the JSON <answer> without introducing task-specific heads. For segmentation, we use sparse point_2d outputs to probe masks following our benchmark protocol. This head-free design keeps the learning signal unified: structural validity and geometric precision are attributed to <answer>, while grounding fidelity and global--local agreement are attributed to <think> in conjunction with <scene> and <focus>. The result is a closed loop where interpretable trajectories, verifiable references, and final predictions are optimized jointly under multi-round supervision.

> 💡 **Head-free 设计的巧妙之处**:
> - 检测和分割都通过 JSON 表达，不需要额外的检测头或分割头
> - 奖励信号分工明确: <answer> 负责结构有效性和几何精度 (base rewards)；<think> + <scene> + <focus> 负责 grounding 保真度和语义一致性 (new rewards)
> - 形成闭环: 可解释轨迹 + 可验证引用 + 最终预测 → 多轮监督下联合优化

---

### 4.3 Reward Functions

We optimize RegionReasoner with reinforcement learning, shaping both intermediate reasoning and final predictions. Besides the base rewards inherited from prior work (VisionReasoner): Thinking Format, Answer Format, Non-Repeat, Bboxes IoU, Bboxes L1, and Points L1, we introduce two multi-round objectives that explicitly encode (i) citation of required references inside the reasoning trace and (ii) semantic alignment between global and local evidence.

> 💡 **Reward 全景**:
> | 类别 | Reward | 目标 |
> |------|--------|------|
> | **Base** (继承自 VisionReasoner) | Thinking Format | <think> tag 格式正确 |
> | | Answer Format | <answer> JSON 格式正确 |
> | | Non-Repeat | 防止重复输出 |
> | | Bboxes IoU | 检测框 IoU |
> | | Bboxes L1 | 检测框 L1 距离 |
> | | Points L1 | 分割点 L1 距离 |
> | **New: Reference Citation** | R_ref | 显式引用 + 幻觉惩罚 |
> | **New: Consistency** | R_cons | 全局-局部语义对齐 + 空间词先验 |

**Notation.** At turn t, the model outputs s_t (<scene>), f_t (<focus> if any), h_t (<think>), and a_t (<answer>). Required references are B_t^ref = {b_k^ref} (possibly empty). A lightweight extractor κ(·) returns keyword sets. We parse bbox mentions from h_t as S(h_t) and use kν(h_t) ∈ {0,1} to flag bbox-related tokens.

**Reference citation reward.** To make the reasoning verifiable and grounded, the trace must explicitly cite the referenced boxes when they are required. We reward correct citation and penalize hallucinated coordinates:

```
R_ref(t) = {
  1,                                            if B_t^ref = ∅
  λ·kν(h_t) + μ·|S(h_t)∩B_t^ref| / max(|S(h_t)|,1),   otherwise
}

R_ref(t) ← {
  η·R_ref(t),  if S(h_t)\B_t^ref ≠ ∅     (hallucination penalty)
  R_ref(t),    otherwise
}
```

with λ=μ=1.0, η=0.5, and clipping R_ref(t) ∈ [0, 2].

> 💡 **公式批读 — Reference Citation Reward (Eq.4)**:
>
> **无参考时** (B_t^ref = ∅): R_ref = 1（中性，不奖励也不惩罚）
>
> **有参考时**:
> - λ·kν(h_t): 奖励在 <think> 中提到了 bbox 相关 token（存在性检查）
> - μ·|S(h_t)∩B_t^ref|/max(|S(h_t)|,1): 奖励正确引用的比例（精确性检查）
> - max 操作避免了稀疏引用时的除零问题
>
> **幻觉惩罚**:
> - 如果 S(h_t) 中存在不在 B_t^ref 中的坐标
> - Reward 乘以 η=0.5 衰减因子
> - 效果: 模型学会"不要编造不存在的参考框坐标"
>
> **设计巧妙之处**:
> - 存在性 + 精确性 双重检查
> - 幻觉惩罚与奖励分离（先算基础分，再乘衰减因子）—— 比直接减分更平滑
> - 值域 [0, 2]，与 base rewards 保持一致的 scale

**Global--local consistency reward.** To keep the reasoning coherent with both global scene context and localized evidence, we align h_t with s_t and (when present) f_t. Let the asymmetric keyword overlap be:

```
Ov(X, Y) = |κ(X) ∩ κ(Y)| / max(|κ(X)|, 1)
```

We also include a light logic prior ℓ(h_t) ∈ [0, 1] counting spatial/comparison/localization terms (capped at 1). The consistency reward is:

```
R_cons(t) = w_s·Ov(s_t, h_t) + w_f·[B_t^ref ≠ ∅]·Ov(f_t, h_t) + w_ℓ·ℓ(h_t)
```

with w_s=1.0, w_f=0.6, w_ℓ=0.4, clipped to [0, 2].

> 💡 **公式批读 — Global--Local Consistency Reward (Eq.5-6)**:
>
> **Ov(X, Y) = |κ(X)∩κ(Y)| / max(|κ(X)|, 1)**:
> - **非对称重叠**: 分母是 |κ(X)|（source 的关键词数），不是 |κ(X)∪κ(Y)|
> - 含义: <think> 覆盖了 <scene>/<focus> 中多少比例的实体
> - 用 max(·, 1) 处理空集边缘情况
>
> **R_cons 三部分**:
> 1. w_s·Ov(s_t, h_t): 全局描述中的实体在推理中被提及的比例 (权重 1.0)
> 2. w_f·[B_t^ref≠∅]·Ov(f_t, h_t): 局部描述中的实体在推理中被提及的比例 (权重 0.6)
>    - [B_t^ref≠∅] 是指示函数：无参考时该项为 0
> 3. w_ℓ·ℓ(h_t): 空间/比较/定位词表先验 (权重 0.4)
>
> **权重设计含义**:
> - w_s > w_f: 全局上下文比局部区域描述更重要（可能是为了防止过度关注局部而丢失全局视角）
> - w_ℓ 最小: 空间词只是辅助，不应主导 reward
> - 为什么 w_f=0.6 而非更大: 局部描述可能在边界模糊时不准确

**Total per-turn objective and episode return.** Let R_base(t) denote the base rewards. The per-turn reward aggregates as:

```
R(t) = R_base(t) + α·R_ref(t) + β·R_cons(t)
```

where α=β=1 by default. Each component is normalized to [0, 2] prior to aggregation to balance scales, and the episode return is Σ_t R(t) over turns. Compared to baselines, these rewards are used only as internal training signals; all evaluation metrics remain purely geometry-based (AP and gIoU) and are computed identically for all models.

> 💡 **Reward 聚合策略**:
> - 每个分量先独立归一化到 [0, 2]，再加权求和
> - α=β=1.0 为默认平衡设置
> - **关键**: 这些 reward 是训练时的内部信号，评估时所有模型使用相同的纯几何指标 (AP50, gIoU)
> - 附录 H 的灵敏度分析表明 α/β 在 [0.5, 1.5] 范围内性能稳定

---

### 4.4 Training

We optimize the policy π_θ with GRPO over multi-turn rollouts. For each batch, the model generates structured actions y_t = (s_t, f_t, h_t, a_t) at turns t=1...T conditioned on (I, q_t, B_t^ref, M_{t-1}). Per-turn rewards follow the decomposition in Sec. 4.3 with componentwise normalization to [0, 2]; the episode return is Σ_{t=1}^T R(t).

> 💡 **训练流程核心要素**:
> - **算法**: GRPO (Group Relative Policy Optimization)
> - **采样**: K=8 rollout per prompt (per step)
> - **数据流**: Multi-turn rollout → 逐轮 compute reward → episode return 聚合 → GRPO update

**Objective.** We optimize the clipped policy objective GRPO on the autoregressive likelihood of the structured action:

```
L_clip(θ) = E[ min(ρ_t(θ)·Â_t, clip(ρ_t(θ), 1-ε, 1+ε)·Â_t) ]
ρ_t(θ) = π_θ(y_t|I, q_t, B_t^ref, M_{t-1}) / π_θold(y_t|I, q_t, B_t^ref, M_{t-1})
```

**Advantage estimation and value targets.** Let s_t = (I, q_t, B_t^ref, M_{t-1}) denote the turn-t state and r_t the per-turn reward. We use a learned value head V_φ(s) and compute advantages with GAE:

```
δ_t = r_t + γ·V_φ(s_{t+1}) - V_φ(s_t)
Â_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
```

Each dialogue is a finite episode; the last turn T is terminal, so V_φ(s_{T+1}) = 0. The value target is R̂_t = Â_t + V_φ(s_t) and the critic is trained with L_value = 1/2 (V_φ(s_t) - R̂_t)^2. We add a small entropy bonus to encourage exploration and, optionally, a KL penalty to a frozen reference policy for stability:

```
L_total = L_clip + c_v·L_value - c_e·H[π_θ(·|s_t)] + β·KL(π_θ(·|s_t) || π_ref(·|s_t))
```

> 💡 **训练细节解读**:
>
> **GRPO + GAE**:
> - GRPO 的 clipped objective 防止策略更新过大
> - GAE 提供低方差的 advantage 估计
> - V_φ(s_{T+1}) = 0: 对话是有限 episode，最后一轮后无后续状态
>
> **额外正则项**:
> - Entropy bonus (-c_e·H): 鼓励探索
> - KL penalty to frozen ref policy: 防止策略偏离太远
>
> **训练技巧**:
> - Sliding memory M_{t-1}: 在上下文预算内保留前轮轨迹
> - Turn-depth curriculum: 训练早期逐步增加最大轮数 T
> - 约束解码: 确保 tag/schema/JSON 有效性，使 reward 对所有输出都 well-defined

---

## 三、Summary

- **Pipeline**: (I, q_t, B_t^ref, M_{t-1}) → π_θ → (s_t, f_t, h_t, a_t) → 解析 a_t → 更新 M_t
- **4-tag 架构**: <scene> (全局) → <focus> (局部) → <think> (推理+显式引用) → <answer> (JSON 定位)
- **两条新 Reward**:
  - R_ref: citation + hallucination penalty, λ=μ=1.0, η=0.5
  - R_cons: w_s=1.0·Ov(s,h) + w_f=0.6·Ov(f,h) + w_ℓ=0.4·ℓ(h)
- **训练**: GRPO + GAE + entropy bonus + KL penalty, 4×H100 ~10h
