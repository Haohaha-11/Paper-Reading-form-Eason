[← 返回 README](../README.md)

# 3. Methodology

## 一、Preview

三部分：(1) Perceval 设计——Error-finding Schema（think-then-answer 结构化输出）+ 四阶段 SFT 训练流水线；(2) Token-level Advantage 重分配——将 Perceval 的检测结果（错误子串列表）转化为 binary mask，用 Eq.3 重分配 token-level advantage；(3) Test-time Scaling——Truncate-then-Regenerate 和 Truncate-Thinking-then-Regenerate 两种迭代策略。

---

## 二、原始文本

In this section, we devise our perception-centric process reward model for providing fine-grained, process-level supervision to guide VLMs. We first introduce the design and how to train the PRM, and then present how to integrate it with RLVR during training and how to perform test-time scaling with PRM guidance.

### 3.1. Perception-Centric Process Reward Model

To overcome the sparse supervision issue, we propose PERCEVAL (Perception-centric process reward evaluation model), which serves as an external, fine-grained, and interpretable critic for guiding VLM policy.

**Error-finding Schema Design.** Given a tuple of image, text query, and model's response ⟨v, q, o⟩, PERCEVAL generates a structured verification V to assess the factual consistency with respect to v (conditioned on q). To improve reliability, PERCEVAL follows the well-known think-then-answer paradigm [13]: it first analyzes each claim and outputs the thought process within <think>...</think>, where each statement in o is evaluated for consistency with the visual evidence in v. Based on these analyses, PERCEVAL provides the final decision wrapped in <answer>...</answer>. If no perceptual errors are found, the final answer is simply "The response is correct."; otherwise, the answer is formatted as a Python list containing the exact strings from o that are identified as errors.

> 💡 **机制拆解 — Perceval 的 Error-finding Schema**:
>
> ```
> Input: ⟨image v, query q, response o⟩
>
> ├── <think>
> │   Claim 1: "The blue mug is on the wooden table"
> │     → 图像验证: 桌子上确实有一个蓝色杯子 → CORRECT
> │   Claim 2: "There is a red brick next to the mug"
> │     → 图像验证: 桌子上没有红色砖块 → MISALIGNMENT
> │   Claim 3: "The mug has a white handle"
> │     → 图像验证: 杯子的把手是黑色的 → MISALIGNMENT
> │   ...
> │ </think>
> │
> └── <answer>
>     ["There is a red brick next to the mug",
>      "The mug has a white handle"]
>     </answer>
> ```
>
> **关键设计**:
> 1. **Think-then-Answer 范式**: 分析过程与结论分离——think 部分做 claim-by-claim 校验，answer 部分只输出错误子串列表。这种设计让输出既可解释又便于下游（如 mask 构建）直接使用。
> 2. **Exact Substring 输出**: answer 返回的是原文中的精确子字符串，不是摘要或泛化描述。这个设计确保后续能通过字符串精确匹配定位到具体的 token span。
> 3. **Claim 粒度**: Perceval 将回复拆解为独立的 claims 逐一验证，而不是对整个回复做一次性判断。这种分解降低了单次判断的难度，提高了准确性。

**Process Reward Model Training.** We train PERCEVAL using a dataset constructed via a four-stage pipeline:

- **Query selection**: to emphasize perceptual grounding, we primarily source the images and queries from visual search datasets [42, 56] that require locating specific objects in large images, and we include a small proportion from other domains (e.g., mathematical reasoning and general understanding [10]) to preserve breadth;

- **Rollout generation**: based on the images and queries, we use an open-source VLM (e.g., Qwen2.5-VL-7B) to produce responses, whose imperfect perceptual alignment yields realistic hallucinations as negative examples;

- **Automated annotation and verification**: for each response, we adopt a strong models (e.g., Gemini-2.5-Pro) to perform hallucination-focused, step-by-step checks. The generated annotations follow our designed format.

- **Supervised fine-tuning**: we fine-tune the PERCEVAL backbone with a standard SFT objective on the aggregated data to emulate detailed, perception-centric verification and produce the prescribed structured output.

> 💡 **机制拆解 — 四阶段训练流水线的设计考量**:
>
> | 阶段 | 输入 | 输出 | 设计为什么这么做 |
> |------|------|------|-----------------|
> | Query Selection | 开源数据集 | 感知密集 queries + 少量通用 queries | 以 visual search 为主保证标注密度（这些任务天然产生大量可比的 perceptual claims），掺入少量通用数据防止过拟合到单一任务类型 |
> | Rollout | 选定 queries | 含真实幻觉的 VLM 回复 | 使用开源 VLM 而非合成数据——保证幻觉分布与真实推理场景一致 |
> | Auto Annotation | Rollout 回复 | 结构化标注 | Gemini-2.5-Pro 做 hallucination-focused 标注——利用强模型的视觉能力生成高质量标注 |
> | SFT | aggregated data | Perceval 模型 | 标准 SFT，目标让 Perceval 学会模仿强模型的标注行为 |
>
> **隐含的设计权衡**:
> - 标注质量 vs 成本：用 Gemini-2.5-Pro 保证质量，但也意味着训练数据质量受限于标注模型的视觉能力上限
> - 正负样本比例：rollout 来自开源 VLM 的自然错误 → 自然蕴含合理比例的正负例

### 3.2. RLVR with Process-level Supervision

Building on PERCEVAL, we revise the GRPO objective to support process-level supervision by replacing the coarse sequence-level advantage Â_i (Eq. 1) with a token-level advantage Â'_{i,t}. The key change is to let advantage computation accept per-token signals so that perceptual errors within a response are directly penalized during learning. To achieve it, for each response, we use PERCEVAL to identify the token spans that realize perception-induced hallucinations, and then re-assign advantages for those tokens to provide a reduced (or more negative) learning signal.

Given a response $o_{i}$ of length $L_{i}$ and the PERCEVAL verification, we parse the <answer> content and select the identified problematic substrings. We locate each substring in $o_{i}$ via exact string match to obtain its token span [$j_{k}$, $l_{k}$] and define $U_{i}$ = ⋃_{k=1}^K [$j_{k}$, $l_{k}$]. From $U_{i}$ we construct a binary mask $M_{i}$ = [$m_{{i,1}}$, ..., $m_{i,$L_{i}$}$] with $m_{{i,t}}$ = 1 if t ∈ $U_{i}$ and 0 otherwise. Then, we modulate the sequence-level signal with this mask to form the token-level advantage:

$\hat{A}'_{i,t} := \hat{A}_i - \alpha \cdot m_{i,t} \cdot |\hat{A}_i|$ (3)

where α ∈ [0, 1] controls penalty strength. Thus, correct tokens ($m_{{i,t}}$ = 0) keep Â'_{i,t} = Â_i, while hallucination tokens ($m_{{i,t}}$ = 1) are downweighted: when Â_i > 0, Â'_{i,t} = Â_i(1 - α); when Â_i < 0, Â'_{i,t} = Â_i(1 + α), making the penalty stronger. Finally, we substitute Â'_{i,t} into the GRPO objective in Eq. 2 to add the process supervision. Such a way injects direct, token-level corrective pressure into GRPO, which preserve sequence-level preferences while explicitly suppressing ungrounded content.

> 💡 **公式批读 — Token-level Advantage 重分配 (Eq.3)**:
>
> 这是全文最核心的公式，拆解如下：
>
> **两个输入信号**:
> - Â_i: 原始 GRPO 的序列级 advantage（组内标准化后的标量）
> - $m_{i,t}$ ∈ {0, 1}: Perceval 提供的 token-level binary mask（1 = 幻觉 token）
>
> **四种情况**:
>
> | $m_{i,t}$ | Â_i 符号 | Â'_{i,t} 结果 | 直观解释 |
> |---------|---------|--------------|---------|
> | 0 (正确) | 任意 | Â_i | 不做任何修改，保持原始 advantage |
> | 1 (幻觉) | > 0 (好回复) | Â_i × (1-α) | 正 advantage 被衰减——"整体不错，但这句话说错了" |
> | 1 (幻觉) | < 0 (差回复) | Â_i × (1+α) | 负 advantage 被放大——"已经够差了，这句话错得更离谱" |
> | 1 (幻觉) | ≈ 0 | 接近 0 | 中性的回复中的幻觉，惩罚也轻 |
>
> **参数 α 的作用**:
> - α = 0: 退化为标准 GRPO（无 token-level 监督）
> - α = 1: 幻觉 token 的 advantage 归零（Â_i > 0 时）或翻倍惩罚（Â_i < 0 时）
> - α ∈ (0, 1): 在"不惩罚"和"最强惩罚"之间插值
> - 实验发现 α = 0.1 最优（Section 4.3, Table 3）
>
> **设计巧妙之处**:
> 1. 使用 |Â_i| 而非固定值——惩罚强度与序列级信号自适性耦合。如果 Â_i 很大正，意味着这整体是一个不错的回复，其中的幻觉惩罚也相对温和（1-α 衰减）；如果 Â_i 很大负，说明整个回复可能就很差，幻觉部分火上浇油。
> 2. Eq.3 直接将新 advantage 代入 Eq.2——不需要修改 GRPO 的目标函数本身，只需要把 Â_i 替换成 Â'_{i,t}。这保持了与 GRPO 框架的完全兼容，实现简单。

> 💡 **机制拆解 — 从 Perceval 输出到 Mask 构建的完整流程**:
>
> ```
> Step 1: Perceval 检测
>   $o_i$ = "The blue mug is on the table. There is a red brick next to it."
>   Perceval → <answer>["There is a red brick next to it"]</answer>
>
> Step 2: 精确字符串匹配
>   在 $o_i$ 中找到 "There is a red brick next to it" 的起始位置
>   → 假设对应 token 位置 15-28
>
> Step 3: Mask 构建
>   $M_i$ = [0,0,0,...,0,1,1,...,1,0,...,0]
>                    └─ positions 15-28 ─┘
>
> Step 4: Advantage 重分配
>   Â_i = +0.5 (整体是好的回复)
>   hallucination token 的 Â'_{i,t} = 0.5 × (1-0.1) = 0.45  (被轻微惩罚)
>   正确 token 的 Â'_{i,t} = 0.5  (不变)
> ```

### 3.3. Test-Time Scaling with PRM Guidance

Beyond training-time use, PERCEVAL (our perception-centric PRM) enables test-time scaling by supplying targeted error-correction during inference. We introduce two pragmatic refinement loops:

**Truncate-then-Regenerate.** When PERCEVAL detects an erroneous claim, it returns the offending span in the model's rationale. We truncate the hypothesis before the first token of that span, preserving only the verified prefix as context. The policy model then continues to regenerate the answer following this cleaned prefix. As the original image and question are given, the VLM just needs to resample the detected hallucinated part, without rewriting verified content. This truncate-continue cycle repeats until no new errors are flagged or a maximum of k iterations is reached. The iteration cap k bounds latency while typically yielding large accuracy gains with only a few refinement steps.

**Truncate-Thinking-then-Regenerate.** To further encourage self-correction, we augment the above method with a lightweight guidance for thinking. After truncating at the error, we append a brief thinking prompt in PERCEVAL's output, e.g., "Wait, I need to reconsider this reasoning more carefully: the mug is not on the brick in the image.", which guides the model to think and then regenerate from the augmented context. The added thinking process enables self-reflection on the failure mode (object/attribute/spatial mismatch), improving the likelihood that the continuation repairs the specific misalignment. As with Truncate-then-Regenerate, we iterate up to k times or stop early when no further errors are found, trading modest extra compute for stronger factual grounding.

> 💡 **机制拆解 — 两种 Test-time Scaling 策略对比**:
>
> ```
> Truncate-then-Regenerate:
>   [生成首段]  →  Perceval 检测到错误 "red brick"  →
>   [截断: 保留 "The blue mug is on the table."]  →
>   [policy 基于截断前缀继续生成余下部分]  →
>   → 重复直到无新错误 或 k 次
>
> Truncate-Thinking-then-Regenerate:
>   [生成首段]  →  Perceval 检测到错误 "red brick"  →
>   [截断 + 插入引导反思: "Wait, I need to reconsider..."]  →
>   [policy 基于截断前缀 + 反思提示重新生成]  →
>   → 重复直到无新错误 或 k 次
> ```
>
> **关键对比**:
> | 维度 | Truncate | Truncate-Thinking |
> |------|----------|-------------------|
> | 上下文 | 仅模型自己的正确前缀 | 前缀 + Perceval 反思提示 |
> | 与训练分布对齐 | 高（模型自回归） | 中（插入了外部反思文本） |
> | 稳定性 | 更稳定 | 可能因反思提示格式不匹配而不稳定 |
> | 需要反思能力 | 不需要 | 需要模型能理解并遵循反思提示 |
>
> 实验（Table 2）验证了这一分析：Truncate 整体比 Truncate-Thinking 更稳定且提升更大，尤其是在 k 较大时。作者将 Truncate-Thinking 的低效归因于"模型的训练数据中缺乏足够的反思数据，反思提示与模型原有分布不完全对齐"。

---

## 三、Summary

- **Perceval 设计**: think-then-answer schema（<think>claim-by-claim 分析</think> + <answer>错误子串 Python list</answer>）
- **PRM 训练**: 四阶段流水线——Query → Rollout → Auto Annotation → SFT
- **Token-level Advantage**: Eq.3 = Â'_i,t = Â_i - α · $m_{i}$,t · |Â_i|，核心洞察：幻觉 token 的正确 advantage 被打折，错误的 advantage 被打折+"加热"
- **Test-time Scaling**: Truncate（基于前缀重新生成）优于 Truncate-Thinking（反思提示），因为前者更接近模型训练分布
- **整体数据流**: Policy 生成 → Perceval 检测 → Mask 构建 → Advantage 重分配 → GRPO 更新
