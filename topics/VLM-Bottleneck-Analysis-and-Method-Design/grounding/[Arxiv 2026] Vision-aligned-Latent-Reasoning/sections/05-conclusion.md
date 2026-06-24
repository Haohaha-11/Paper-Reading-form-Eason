[← 返回 README](../README.md)

# 5. Conclusion & Impact Statement

## 一、Preview

本章包括正式的 Conclusion（简要总结贡献和展望）和 Impact Statement（讨论社会影响和应用前景）。结论简短但信息密度高，Impact Statement 则重点阐述了 VaLR 在 VLA 和 CUA 等具身智能应用中的潜在影响。

---

## 二、原始文本

### 5. Conclusion

In this paper, we have presented VaLR, a multi-modal reasoning framework that generates vision-aligned latent tokens during the reasoning process. Our experiments showed that VaLR performs test-time scaling behavior and consistently improves performance on various benchmarks that require a long or short context. We hope our work will facilitate future research on reasoning in multi-modal large language models.

> 💡 **结论解读**: 作者选择了极为精简的结论风格——仅 3 句话。这反映出作者对论文贡献的信心：核心贡献（vision-aligned latent tokens）、核心证据（test-time scaling）、核心展望（facilitate future research）。没有多余的修辞，每个词都承载信息。

### Impact Statement

Recent advancements of Multi-modal Large Language Models (MLLMs) have enabled remarkable performance in vision question answering. However, these models suffer from dilution of visual information during autoregressive text generation. This phenomenon is emphasized during long-context reasoning. Consequently, it prevents the use of MLLM in domains that require long-context reasoning, such as the Vision Language Action (VLA) model and the Computer Use Agent (CUA).

Our work addresses these challenges by proposing an effective way to inject a visual checkpoint with a latent token. Our approach improves long-context reasoning with test-time scalability and general VQA performance. We believe VaLR suggests future directions for mitigating the dilution of vision information.

In context of applications, alleviating the long-context reasoning reveals the use of MLLM in much more complex tasks, especially when visual information is involved, such as a robot with VLA or proactive CUA. This automatic agentic system can facilitate the innovation of human society.

> 💡 **Impact Statement 批读**: 这段话实际上是 Introduction 中问题的回响，但视角转向了社会影响。关键论点是：VaLR 解决"视觉信息稀释"问题后，MLLM 可以部署到 VLA（机器人操作）和 CUA（计算机使用代理）等更长时域、更复杂的应用中。这不仅是性能提升，而是**打开了 MLLM 应用的新空间**。

> 💡 **VaLR 的终极愿景**: 将论文放在更大的研究图景中看，VaLR 实际上在尝试回答一个根本问题：**如何让 AI 系统在长时间交互中持续保持对视觉世界的感知？** 当前的 MLLM 更像是"看一眼图然后闭眼推理"，而 VaLR 试图让模型在推理过程中"保持睁眼"。这种能力对于任何需要感知-推理闭环的 AI 应用（机器人、自动驾驶、AR 助手等）都是基础设施级别的要求。

---

## 三、关键限制与未来方向（基于论文内容的综合分析）

### 当前限制

1. **单模型验证**: 所有实验基于 Qwen2.5-VL-7B，未在 InternVL、LLaVA、PaliGemma 等架构上验证，泛化性尚未证明。
2. **CoT 数据依赖**: latent token 的插入位置由人工标注的推理步骤决定，这限制了可扩展性——如果推理步骤划分不合理，latent token 可能插入在不恰当的位置。
3. **K 和 λ 的敏感性**: K=16 和 λ=0.5 是实验最优值，但论文未深入讨论这两个超参数的选择原理，也未提供自适应选择策略。
4. **时钟时间开销**: 虽然推理时不需要外部编码器，但 latent mode 的 K=16 步和两阶段交替确实增加了推理时间（Table 10: 1.55s vs 1.21s for 32-view）。
5. **仅 7B 规模验证**: 未在更大规模模型（13B, 72B）上测试，不清楚 VaLR 的增益是否随模型规模变化。

### 未来方向

1. **扩展到视频理解**: 视频天然具有时序视觉信息，VaLR 的"每步视觉检查点"机制可能特别适合长视频理解。
2. **VLA 和 CUA 的实际部署**: 在机器人操作和桌面代理任务上验证 VaLR 的 real-world 效果。
3. **自动推理步发现**: 不依赖人工标注的 CoT 推理步骤，让模型自动学习何时触发 latent mode。
4. **与 RL 结合**: 将 REPA 作为辅助 loss 结合 RL 训练，可能产生更强的推理模型。
5. **更细粒度的视觉对齐**: 当前是 patch-wise，未来可以探索 object-level 或 semantic-region-level 的对齐策略。
6. **多模态编码器融合**: 探索更智能的多编码器融合策略（当前是简单的平均），如 attention-based weighting 或 task-adaptive selection。

---

## 四、Summary

- **结论**: VaLR 通过视觉对齐的潜推理实现了 MLLM 的 test-time scaling，在长短上下文 benchmark 上一致提升。
- **影响**: 为 VLA、CUA 等需要长时域感知-推理闭环的应用打开了可能性。
- **核心启示**: 潜推理 + 动态视觉对齐是解决 MLLM 视觉衰减问题的有效方案。训练时的视觉对齐可以蒸馏到推理时的潜空间表征中。
