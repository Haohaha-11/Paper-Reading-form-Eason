[← 返回 README](../README.md)

# 5. Related Work & 6. Conclusion

## 一、Preview

相关工作总结了三条发展脉络：(1) VLM 从对齐到推理；(2) RL for VLM 从适配到激励机制创新；(3) Multimodal RM 从标量打分到过程监督。Conclusion 总结了 Perceval 的核心价值和未来方向。

---

## 二、原始文本

### 5. Related Work

**Vision-language Models.** The field of vision-language models (VLMs) has evolved from foundational representation alignment to complex multimodal reasoning. Early breakthroughs such as CLIP [32] and ALIGN [16] demonstrate that contrastive pre-training on web-scale image-text pairs yields powerful, transferable representations, setting the stage for Large Vision Language Models (LVLMs) that bridge pre-trained visual encoders with LLMs [2, 18, 23]. "Visual Instruction Tuning" [23] emerges as a critical paradigm for unlocking multimodal instruction-following, rapidly scaled in open-source models like Qwen-VL [3] and InternVL [57]. By incorporating large-scale SFT and RL, advanced VLMs [12, 44] achieve strong performance on complex reasoning tasks. However, perceptual capabilities remain a critical bottleneck: models frequently exhibit hallucinations [19, 20] or are unduly dominated by textual priors [1, 22, 53], highlighting a persistent gap in reliable, fine-grained visual perception.

> 💡 **脉络一 — VLM 从对齐到推理**:
>
> ```
> CLIP/ALIGN (对比对齐)
>     → BLIP-2/Flamingo (桥接视觉编码器与 LLM)
>         → Visual Instruction Tuning (指令跟随)
>             → Qwen-VL/InternVL (大规模 SFT + RL)
>                 → 瓶颈: 感知幻觉 + 文本先验主导
> ```
>
> 本文定位：不是要做一个更强大的 VLM，而是要在现有 VLM 的（RLVR）框架中注入更精细的感知监督信号。问题的根因一直被反复提及（幻觉、文本先验）但缺乏系统性的精细干预方案。

**Reinforcement Learning for VLMs.** The application of RL to VLMs has rapidly evolved toward capability incentivization for complex multimodal reasoning. This shift was catalyzed by breakthroughs in LLMs demonstrating that large-scale RL can elicit emergent "slow-thinking" behaviors [13, 15, 35], inspiring a new wave of VLM research that optimizes the synergy between visual perception and logical deliberation [14, 30, 34]. Beyond adapting LLM strategies, researchers explore reflection techniques tailored to the visual domain and "thinking with images" paradigms that leverage image manipulation tools to support reasoning. However, a critical limitation persists: methods based on RLVR predominantly rely on GRPO, which provides only coarse, outcome-level supervision and lacks the fine-grained signals necessary for improving complex, step-by-step reasoning.

> 💡 **脉络二 — RL for VLM 的分类**:
>
> | 类别 | 代表方法 | 特点 | 共享瓶颈 |
> |------|---------|------|---------|
> | LLM RL 适配 | VLM-R1, LMM-R1, R1-VL | 将 DeepSeek-R1 的 RLVR 范式迁移到 VLM | 全部使用 GRPO → 序列级稀疏奖励 |
> | 反思机制 | VL-Rethinker, Vision-R1 | 在 RL 中引入反思或 rethinking 触发 | GRPO + forced triggers |
> | 看图思考 | DeepEyes, PixelReasoner | 外部工具操作图像 (zoom/crop) | 工具调用不稳定 + 开销 |
> | 感知专项 | Perception-R1 | 面向感知任务的 GRPO | 仍需外部奖励函数 |
>
> 本文的差异化定位：不改变 GRPO 算法本身，而是在 advantage 计算阶段插入 token-level 的感知监督信号。这种"不动基础架构，只在中间插入"的策略保证了与所有 GRPO-based 方法的兼容性。

**Multimodal Reward Models.** Multimodal reward models [40, 46, 52] play a pivotal role in Reinforcement Learning from Human Feedback (RLHF) by aligning model outputs with human preferences. With the recent proliferation of reinforcement learning in complex reasoning tasks, RMs are also increasingly employed to supplement methods like Reinforcement Learning with Verifiable Rewards (RLVR). This becomes particularly crucial in domains where verifiable ground truth is inaccessible, such as open-ended creative tasks [25], which are environments where methods reliant on verifiable rewards consequently struggle. The predominant approach for these RMs involves training them to directly output a single scalar score, which represents the overall quality of a given trajectory [39, 46]. Recognizing the limitations of this direct scoring, more recent research efforts have sought to integrate "slow thinking" or deliberate reasoning paradigms into reward modeling [49, 51]. These approaches enable the RM to generate a rationale or critique before assigning the final score, aiming for more meticulous and robust evaluations [9]. However, a fundamental limitation persists: whether generated directly or after deliberation, the feedback from existing RMs ultimately collapses into a single scalar reward. This offers only sparse, outcome-level supervision for algorithms like GRPO. We propose a perception-centric reward model that provides a more fine-grained signal, which enabling token-level adjustments of advantages, thereby offering a more precise supervision.

> 💡 **脉络三 — Multimodal RM 的演进与本文的突破**:
>
> ```
> 传统 RM: 输入 → RM → 标量分数 (single scalar)
>     ↓ [问题: 粗粒度]
> Slow-Thinking RM: 输入 → RM → 思考过程 → 标量分数
>     ↓ [进步: 可解释但输出仍是标量]
> Perceval (Ours): 输入 → PRM → 错误子串列表 → token-level mask
>                     [不输出标量，输出结构化的错误定位]
> ```
>
> 根本差异：前两种方法最终都"坍缩"为一个标量分数——无论中间多想，输出只有一个数字。Perceval 拒绝这种坍缩，输出的是精确的错误位置，这使得它能被用于 token-level 的 advantage 重分配，而不仅仅是给 GRPO 提供另一个标量奖励。

### 6. Conclusion

In this work, we introduced PERCEVAL, a perception-centric process reward model (PRM) that addresses the sparse reward issue in RLVR by enabling token-level error grounding. Unlike traditional outcome-level supervision, PERCEVAL detects image-text misalignments within the model's reasoning process and provides grounded, step-aware feedback. We trained PERCEVAL with perception-intensive data and integrate it into both the training and inference stages of VLMs. At the training stage, we leverage PERCEVAL to apply token-level penalties to hallucinated spans, improving fine-grained credit assignment and surpassing the capabilities of sequence-level methods like GRPO. During inference, PERCEVAL enables a Truncation-Regeneration loop that prunes erroneous responses and induces model reflection. Our experiments demonstrate that PERCEVAL substantially improves visual grounding on perception-heavy benchmarks and facilitates better transfer to multi-step reasoning tasks. This method represents a significant advancement in fine-tuning the reasoning capabilities of VLMs, with the potential to generalize across domains and tasks.

> 💡 **全文核心逻辑总结**:
>
> 1. **Problem**: RLVR 的序列级稀疏奖励 → credit-assignment 困难
> 2. **Opportunity**: 视觉推理中间步骤是 perceptual claims → 可直接与图像比对 → 自动化 hallucination 检测可行
> 3. **Solution**: Perceval (claim-by-claim 错误检测) → Token-level mask → Advantage 重分配 → 精细化 GRPO; 同时支持推理时截断-重生成
> 4. **Evidence**: 3B/7B 多 benchmark 一致提升 + 能力迁移 + Test-time scaling 优于 major voting + 抗 reward hacking

---

## 三、Summary

- **VLM 脉络**: 从对齐到推理，感知瓶颈（幻觉 + 文本先验）一直存在但缺乏精细的干预手段
- **RL for VLM 脉络**: GRPO 是主导框架但序列级信号是共享瓶颈，Perceval 以"中间插入"的方式增强而不修改 GRPO 本身
- **Multimodal RM 脉络**: 从标量评分到"慢思考"再到结构化错误定位——Perceval 是第 3 代
- **本文贡献**: Token-level error grounding 使 RLVR 从"一人犯错全队受罚"变为"谁犯错谁受罚"
