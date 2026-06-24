[← 返回 README](../README.md)

# 4. Experiments

## 一、Preview

实验部分涵盖五个维度：(1) 实验设置（8 benchmarks + 12 baselines + 3B/7B scales）；(2) RL Training 主结果（关键是仅感知任务 PRM 监督，复杂推理也能受益的能力迁移现象）；(3) Test-time Scaling（PRM 驱动的截断-重生成 vs major voting）；(4) 进一步分析（Reward Hacking 测试 + α 超参数调优 + Case Study）；(5) 定性可视化。

---

## 二、原始文本

### 4.1. Experimental Setup

**Benchmarks.** We select multiple visual reasoing benchmarks, covering visual search, perception-intensive reasoing, mathematical and chart-based reasoing.

1. V\* (V-Star) [42]: introduces an LLM-guided visual search mechanism and a dedicated benchmark, to assess models' ability to localize and reaso about small, target objects within information-dense images. It contains 191 high-resolution images with two subtasks, i.e. attribute recognition and spatial-relation reasoing that require precise grounding before reasoing.

2. MME-RealWorld [50]: targets practical applications across five domains (OCR-in-the-wild, remote sensing, diagrams/tables, monitoring, autonomous driving). We use its subset MME-RealWorld-Lite for testing.

3. BLINK [11]: reframes 14 classic computer-vision tasks (e.g. relative depth, visual correspondence, image forensics, multi-view reasoing) into 3,807 multiple-choice items to probe foundational perceptual skills that resist purely linguistic mediation.

4. MMStar [6]: compiles 1,500 carefully selected, human-curated samples to probe six core capability areas along 18 fine-grained axes, focusing on cases where vision is indispensable (rather than solvable by text priors).

5. RealWorldQA [43]: contains 700 images captured from vehicles and other real-world settings, each paired with a question and an easily verifiable answer.

6. MathVista [26]: aggregates 6,141 examples from 28 existing multimodal sources and three new sets (IQTest, FunctionQA, PaperQA) to test numeracy, geometry/diagram understanding, tables/plots, and compositional visual-math reasoing.

7. MATH-Vision [38]: offers 3,040 problems sourced from real competitions, spanning 16 mathematical disciplines and five difficulty levels, each embedded in a visual context (figures, diagrams, plots).

8. ChartQA [27]: contains 9.6K human-written and 23.1K generated questions over diverse chart types, requiring both visual parsing and table/logic operations.

**Baselines.** We compare our methods with multiple reasoing-oriented VLMs: VLM-R1 [34], LMM-R1 [30], R1-VL [47], Perception-R1 [45], Jigsaw-R1 [41], DeepEyes [56], PixelReasoer [37], Vision-R1 [14], VL-Rethinker [36], VLAA-Thinker [5], OpenVLThinker [8], MM-Eureka [28].

**Implementation Details.** We select Qwen2.5-VL as the backbone for both reward and policy models. We first train two versions of PERCEVAL of 3B and 7B sizes, following the procedures outlined in section 3.1, and then correspondingly train two policy models of the same sizes using the proposed method. As for the training data, the supervised fine-tuning data are collected from DeepEyes [56] and SophiaVL-R1 [10], each of which is rolled out 3 times using the backbone models. The RL training data is also derived from [56], with the primary objective of enhancing the model's perception capabilities, while also containing a subset of general-purpose reasoing data. Consequently, during the RL training phase, we implement a conditional strategy: PERCEVAL is used only on perception-related data to perform fine-grained advantage rescaling. For all other training data (e.g., mathematical reasoing), no additional intervention is applied, and we revert to using direct GRPO. This experimental design allows us to investigate whether fine-grained supervision focused on perception tasks can generalize and yield performance gains in other domains.

> 💡 **实验设计解读 — 条件性 PRM 干预策略**:
>
> 这是一个精心设计的实验方案：
> - **感知数据**: Perceval 介入做 token-level advantage rescaling
> - **数学/推理数据**: 退回到直接的 GRPO（不做 PRM 干预）
>
> 这样设计的意图是测试一个关键假设：**感知是上层推理的公共基础组件，把感知做扎实了，复杂推理自然受益**。
>
> 正反两方面解读：
> - 如果只有感知任务提升：Perceval 只帮助了它直接监督的任务
> - 如果复杂推理也提升：感知能力的增强存在泛化效应 → 印证"感知是 VLM 推理的公共瓶颈"
>
> 实验结果（Table 1）支持了后者。

**Evaluation Setup.** To ensure fair and reproducible evaluation, we establish a unified evaluation pipeline. We employ greedy decoding for all models and utilize the same prompt template to collect responses. We then extract the final answer following the official procedures of each benchmark. Finally, the accuracy is determined through a two-stage judging process: we first apply an exact match (EM) judge for each extracted answer against the ground truth. For any answer that does not match, a robust judge mode (i.e. GPT-4o-mini) is utilized for a final verification to account for minor formatting variations. Additionally, we report the relaxed accuracy for ChartQA [27], aligned with the official evaluation of the benchmark, which uses the methodology of PlotQA [29].

### 4.2. Main Results

**RL Training with PRM.** As shown in Table 1, our method significantly and consistently outperforms the GRPO baseline across both 3B and 7B model scales. Specifically, for the 3B model, our approach achieves average improvements of approximately 4% in the Visual Search category, 3% in Math and Chart reasoing, and 1% in Perception-intensive Reasoing relative to the GRPO baseline. This result strongly demonstrates that our method provides richer and more fine-grained supervision. A deeper analysis of the Visual Search sub-tasks reveals that (Positional Perception), particularly at the 3B scale (e.g., improving from 86.95 to 90.43). This strongly suggests that our fine-grained process supervision has successfully guided the model to enhance its precise spatial localization capabilities. Concurrently, the improvements on benchmarks like BLINK and MMStar also indicate that this enhanced perception leads to higher fidelity and fewer hallucinations. A crucial finding is the model's strong generalization ability. As discussed in Section 4.1, although our PRM training and RL intervention were predominantly focused on Visual Search tasks, the model still exhibits consistent performance gains across all other domains, including general perception and math reasoing. We attribute this "capability transfer" to the fact that tasks in Math & Chart (such as MathVision and ChartQA) are fundamentally reliant on precise, fine-grained perceptual abilities (e.g., localizing data points on a chart, reading text). By strengthening the model's foundational perceptual accuracy, our method successfully generalizes this improvement to broader and more complex reasoing tasks. Furthermore, our 7B model trained with our method also surpasses Pixel-Reasoer and achieves performance competitive with DeepEyes on Visual Search tasks. It is noteworthy that the latter two models both rely on external tool manipulation to assist in object grounding. This result indicates that enhancing the intrinsic perceptual abilities of multimodal base models is a highly promising research direction, capable of rivaling the performance of tool-augmented SOTA methods.

> 💡 **Table 1 批读 — 主结果分析**:
>
> **3B 模型 — 三类任务提升**:
> | 任务类别 | GRPO → Ours 提升 | 关键 benchmark 变化 |
> |---------|-----------------|-------------------|
> | Visual Search | ~+4% | V*_attr: 86.95→90.43, V*_pos: 69.73→72.37 |
> | Math & Chart | ~+3% | MathVision: 23.36→26.32, ChartQA: 83.32→86.48 |
> | Perception-intensive | ~+1% | BLINK: 49.13→48.75 (微降), MMStar: 55.3→55.8 |
>
> **关键发现**:
> 1. **位置感知 (V*_pos) 提升最显著**: 3B 上从 69.73 到 72.37 (+2.64%)，7B 上从 82.89 到 86.84 (+3.95%)。因为 PRM 直接针对空间关系类的幻觉施加惩罚，模型被迫学会"看清楚再说"。
> 2. **能力迁移清晰**: MathVision (3B: 23.36→26.32, +2.96%) 和 ChartQA (3B: 83.32→86.48, +3.16%) 的提升说明——虽然训练时未对数学数据做 PRM 干预，但底层感知精度的增强自动溢出到了需要精确读取图表数据的复杂推理任务。
> 3. **7B 竞争力**: 不依赖外部工具的 Perceval 在 Visual Search 上超越了依赖 zoom/crop 的 PixelReasoer，接近同样依赖工具的 DeepEyes。
>
> **vs. GRPO baseline 的解读**:
> GRPO 已经是一个很强的 baseline（在大多数任务上已经是 backbone 的显著提升）。Perceval 在此基础上进一步提升了约 1-4%，说明 token-level 的监督确实补足了 sequence-level 无法覆盖的"冷启动"部分——即那些"最终答案对但中间感知错了"或者"中间感知对了但最终答案错了"的情况。

**Test-time Scaling with PRM.** As mentioned earlier, PERCEVAL has the potential to assist in the test-time scaling of policy models with the Truncate or Feedback strategies. To validate their effectiveness, we compare them with the major voting strategy, a classic test-time scaling method, where the policy model generate responses for multiple times and selects the most common answer as the final response. We conducted the experiment on the 3B policy model and present the results in Table 2. With different sampling times k, the PRM-based strategies consistently outperform major voting on V\* and BLINK. The Truncate strategy, in particular, shows a more significant improvement compared to the Feedback strategy. We hypothesize that the model's training data does not contain sufficient reflective data, which results in poorer instruction-following quality when the reflective prompts are inserted in the Feedback strategy. In contrast, the Truncate strategy allows the model to regenerate the response based on its own generated context, aligning more closely with the model's original distribution, thus producing more stable and reliable outputs. Another observation is that the major voting strategy quickly converges on difficult tasks (e.g., the Pos subset of V\*) and fails to show further improvement. This suggests that without external intervention, the model's inherent capabilities are insufficient to rectify its errors.

> 💡 **Table 2 批读 — Test-time Scaling 对比**:
>
> | k | 方法 | V\* All | 关键观察 |
> |---|------|---------|---------|
> | 4 | Major Voting | 85.34 | baseline |
> | 4 | Truncate | 87.96 | +2.62% |
> | 4 | Truncate-Thinking | 86.91 | +1.57% |
> | 8 | Major Voting | 85.86 | 几乎无提升（平台效应） |
> | 8 | Truncate | 87.96 | 稳定 |
> | 8 | Truncate-Thinking | 87.96 | 追上 Truncate |
> | 16 | Major Voting | 85.86 | 完全停滞 |
> | 16 | Truncate | 89.53 | +3.67% (最佳) |
> | 16 | Truncate-Thinking | 88.48 | +2.62% |
>
> **关键发现**:
> 1. **Major Voting 的天花板效应**: 在困难的 V\* Pos 子任务上，major voting 在 k=4 后几乎没有提升——"重复问同样的模型同样的问题"，如果模型的本征能力不够，投再多票也无法修正错误。
> 2. **Truncate > Truncate-Thinking**: Truncate 在所有 k 下都领先或持平 Truncate-Thinking，尤其是在 k=16 时（89.53 vs 88.48）。反思提示虽然提供了更丰富的上下文，但格式与模型训练分布不对齐，可能降低了指令跟随质量。
> 3. **PRM 驱动的 test-time scaling 有效**: 与 major voting 的"横向扩增"（多采样+投票）不同，PRM 的策略是"纵向精炼"（检测-截断-重生成），在困难任务上持续获得提升。

### 4.3. Further Analysis

**Reward Hacking Test.** A critical challenge in reinforcement learning with reward models (RMs) is reward hacking, where the policy overfits the RM's scoring function. This issue is particularly pronounced with traditional RMs that output a single scalar reward for an entire response. Such a direct and holistic score, which is often influenced by the RM's own intrinsic biases, provides a simple signal for the policy to exploit, leading to score inflation without genuine quality improvement. Our proposed PERCEVAL is designed to mitigate this specific vulnerability. Instead of providing a direct scalar reward, PERCEVAL intervenes during the advantage calculation stage. Specifically, it reduces the advantage values of only those tokens within a response that are identified as contributing to a hallucination. This fine-grained, indirect guidance mechanism is inherently more difficult for the policy to overfit and simultaneously enhances the contrast between correct and incorrect tokens within the same sequence. The effectiveness of this approach is demonstrated in Figure 2, which plots the proportion of responses identified by PERCEVAL as containing hallucinations during training. The curve initially shows a decline, indicating that the policy is successfully learning to reduce hallucinations. Crucially, the rate then stabilizes rather than continuing to drop. A continuously decreasing curve would suggest that the policy is learning to deceive the PRM — a clear sign of reward hacking. The observed stability therefore confirms that our proposed PERCEVAL effectively guides the policy toward genuine improvement while avoiding significant reward hacking.

> 💡 **Figure 2 批读 — Reward Hacking 分析**:
>
> 曲线解读：
> - **初始下降阶段**: Policy 正在学会减少幻觉——Perceval 检测到的含幻觉回复比例在降低。这是正常的正向学习。
> - **稳定阶段**: 比例不再持续下降，进入平台期。这是正确答案！说明 policy 不是在学习"如何骗过 Perceval"，而是在学习"真正减少幻觉"。如果出现 reward hacking，曲线应该持续下降直到接近 0（但真实答案质量不会对应提升）。
>
> **抗 Reward Hacking 机制的设计哲学**:
> - 传统 RM: policy → RM scoring → scalar reward → policy learns to game the scoring
> - Perceval: policy → Perceval detects errors → mask → advantage rescaling → policy learns to avoid detected errors
> - 关键区别：Perceval 不提供一个直接可优化的标量目标，而是在 advantage 层面做"局部削弱"。Policy 没有单一的数字可以"刷高"，只能在 token 层面被微妙地引导。

**Hyperparameter Tuning.** Our proposed RL training with PRM framework introduces the hyperparameter α (Equation 3), which governs the penalty strength applied to tokens identified as hallucinatory. The selection of an optimal α is critical, as it requires balancing the suppression of hallucinations against the preservation of overall response quality. To quantitatively determine this optimal value, we conduct a series of experiments, varying α across {0.03, 0.1, 0.3} and benchmarking against a standard GRPO baseline (α = 0). The results, summarized in Table 3, reveal a distinct non-monotonic trend. A minimal value of α = 0.03 provides an insufficient corrective gradient. While offering a marginal improvement over the baseline, the penalty is too subtle to effectively steer the model away from ingrained hallucinatory patterns. Conversely, an excessively large α of 0.3 proves counterproductive. We attribute this to collateral "penalization": since the PERCEVAL flags entire substrings, a high penalty indiscriminately punishes all tokens within that span, including syntactically necessary but factually benign words (e.g., articles, prepositions). This introduces significant training noise and degrades overall performance. The analysis reveals that α = 0.1 strikes the optimal balance. It is potent enough to achieve a substantial reduction in hallucinations while avoiding the destabilizing effects of over-penalization. Therefore, we adopt α = 0.1 as the canonical value for all other experiments.

> 💡 **Table 3 批读 — α 消融实验**:
>
> | α | V\* | RealWorldQA | MathVision | ChartQA | 解读 |
> |----|-----|-------------|------------|---------|------|
> | 0.0 (GRPO) | 80.10 | 62.17 | 23.36 | 83.32 | baseline |
> | 0.03 | 81.68 | 63.09 | 22.70 | 84.44 | 惩罚太弱，矫正力道不足 |
> | 0.1 | **83.25** | **64.92** | **26.32** | **85.04** | 最优平衡点 |
> | 0.3 | 78.53 | 61.78 | 22.04 | 84.56 | 过惩罚，低于 baseline |
>
> **非单调特性的解读**:
> - α = 0.03: "正在摸"幻觉，但没有足够的力量去改变已固化的幻觉模式
> - α = 0.1: 恰到好处——足以压制幻觉，但不会误伤正常 token
> - α = 0.3: "一棍子打死"——整个标记的 span 被整体打压，包括必要的功能词。作者用 **collateral penalization** 来形容这种副作用：语法结构正常的 token（冠词、介词）被无辜牵连，引入了训练噪声
>
> **设计中的隐含假设**: Perceval 的 span 是粗粒度的 substring（如整句话），而 α 以 token 为粒度操作。当 α 过大时，整句话的所有 token 都受到同等力度的惩罚，而实际上可能只有部分 token（如修饰词）是错的。这就是为什么 α 需要足够小——为幻觉 token 提供方向性的负反馈，但又不至于过度惩罚。

**Qualitative Analysis.** To clearly demonstrate the efficacy of our method, we present a qualitative analysis of model outputs in Figure 3. This case study compares the outputs from a model trained with direct GRPO against one trained with our method on an identical query. The task necessitates locating two minuscule objects (i.e., a blue vehicle and a vehicle car) to determine their spatial relationship. The baseline model, trained with direct GRPO, bypasses the perceptual task and directly outputs a relative position ("left"). This is a classic example of hallucination, as the model provides an answer without seemingly grounding its response in the visual evidence. In sharp contrast, our model exhibits a deliberate, step-by-step process. It first attempts to locate the white car, subsequently searches for the blue car, and then correctly deduces their relative positions. This case study demonstrates that our RL training process significantly enhances the model's perceptual capabilities, compelling its responses to be faithfully grounded in the visual content.

> 💡 **Figure 3 批读 — Case Study 对比**:
>
> **问题**: "Is the blue truck on the left or right side of the white vehicle?"
>
> **GRPO 模型输出**:
> ```
> <think>The blue truck is visible in the image, positioned to the
> left of the white vehicle. This can be determined by observing the
> relative positions of the vehicles in the image.</think>
> <answer>B. The blue truck is on the left side.</answer>
> ```
> → 问题：think 部分没有显示任何实际的定位过程，直接跳到结论。虽然声称 "observing the relative positions"，但没有任何视觉 grounding 的痕迹——这是典型的"不看不查就张口说"式幻觉。
>
> **Ours 模型输出**:
> ```
> <think>To determine the correct position of the blue truck, I need
> to identify the white vehicle first. The white vehicle appears to
> be a car parked near the building with the red roof in the center
> of the image. The blue truck is positioned to the right of this
> white vehicle.</think>
> <answer>A. The blue truck is on the right side.</answer>
> ```
> → 关键差异：Ours 模型显示了明确的定位步骤——先找白色车、再找蓝色车、然后判断空间关系。这种显式的 grounded reasoing 正是 token-level advantage 惩罚幻觉 token 后模型学会的模式：先感知，再推理。

---

## 三、Summary

- **Main Results**: 3B Visual Search +4%, Math&Chart +3%, Perception-intensive +1%；7B 超越 PixelReasoer 并接近 DeepEyes
- **能力迁移**: 仅在感知数据做 PRM 干预 → 复杂推理也受益 → "感知是公共瓶颈组件"
- **Test-time Scaling**: Truncate > Truncate-Thinking > Major Voting；PRM 驱动的"纵向精炼"在困难任务上优于多采样的"横向扩增"
- **Reward Hacking**: Perceval 的间接干预机制天然抗过拟合——Figure 2 验证了稳定性
- **α = 0.1**: 最佳惩罚强度——太小矫正不足，太大 collateral penalization
- **Case Study**: Token-level 监督迫使模型学会"先看再说"，而非跳过感知直接输出推断
