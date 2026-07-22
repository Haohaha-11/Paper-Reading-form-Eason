# TTT in LLM

Test-Time Training / Test-Time Learning / long-context continual adaptation：围绕 LLM 在推理时用上下文或无标签测试数据更新 fast weights、LoRA 或部分权重，让模型把新信息压进参数并适应分布变化。

## 论文列表

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [Absorber-LLM](./%5BArxiv%202026%5D%20Absorber-LLM/) | arXiv 2026 | Absorber LLM 用 self-supervised causal synchronization 把历史上下文吸收到参数中，让无上下文更新模型在未来生成上匹配有完整上下文的原模型。 |
| [Layer-Wise-Dynamic-TTA](./%5BArxiv%202026%5D%20Layer-Wise-Dynamic-TTA/) | arXiv 2026 | Layer-Wise Dynamic TTA 在无监督 sample-specific TTA 中用 layer-wise hypernetwork 动态调节各层 LoRA 更新强度，缓解固定学习率带来的过拟合与分布漂移。 |
| [In-Place-TTT](./%5BICLR%202026%5D%20In-Place-TTT/) | ICLR 2026 Oral | In-Place TTT 把 LLM MLP block 的 final projection matrix 当作 fast weights，用 LM-aligned objective 和 chunk-wise update 做可插拔 test-time training。 |
| [TTT-E2E](./%5BArxiv%202025%5D%20TTT-E2E/) | arXiv 2025 | TTT-E2E 将 long-context LM 视作 continual learning，用测试时 next-token prediction 更新权重，并用训练时 meta-learning 学适合测试时学习的初始化。 |
| [TLM](./%5BICML%202025%5D%20TLM/) | ICML 2025 | TLM 把 LLM test-time learning 表述为无标签测试数据的 input perplexity minimization，用高困惑度样本选择和 LoRA 更新做域适应。 |
| [TTT4LC](./%5BArxiv%202025%5D%20TTT4LC/) | arXiv 2025 | TTT4LC 揭示长上下文 attention 的 score dilution 导致 thinking tokens 失效，提出 Query-Only TTT：对 query 做少量梯度更新即可恢复长上下文信号。 |

## 子专题

- 📝 [paper-writing](./paper-writing/) — 为撰写 TTT 论文精读的核心参考文献集：TTT-NTP、Self-Guided-TTT、EASE-TTT，以及 TTT4LC、In-Place-TTT、TTT-E2E。

## 课题主线

1. **更新对象不同**：In-Place TTT 更新 MLP final projection fast weights；TTT-E2E 更新 sliding-window Transformer 的部分权重并把 context 压入参数；Absorber LLM 把历史上下文吸收到参数/hidden behavior 中；TLM 和 Layer-Wise Dynamic TTA 则用 LoRA 或受控更新做测试时适应。
2. **训练/测试耦合方式不同**：In-Place TTT 强调 drop-in 与从头预训练两种使用方式；TTT-E2E 在训练时 meta-learn 初始化、测试时 next-token 更新；Absorber LLM 用有上下文 teacher 与无上下文 updated model 做 causal synchronization；TLM 和 Layer-Wise Dynamic TTA 更像无标签目标域/样本级适应。
3. **目标函数不同**：In-Place TTT 用 next-token-prediction aligned value target 替代 generic reconstruction；TTT-E2E 直接以内层 next-token loss 更新；Absorber LLM 同步未来生成所需的内部行为；TLM 最小化 input perplexity；Layer-Wise Dynamic TTA 关注 prompt-only TTA 的动态更新强度。
4. **稳定性机制不同**：Layer-Wise Dynamic TTA 用 hypernetwork 控制分层更新强度，避免固定学习率在单 prompt 上过拟合；Absorber LLM 用 teacher-student causal effect preservation 约束“写入上下文”不要退化成 token projection memorization。
5. **应用压力不同**：In-Place TTT、TTT-E2E 和 Absorber LLM 主要服务长上下文利用与低/常数记忆推理；TLM 和 Layer-Wise Dynamic TTA 主要服务 domain shift、语言变体、任务分布变化和 sample-specific prompt adaptation；TTT4LC 直接面向 long-context QA/检索，用 TTT 替代 thinking tokens 作为推理时 compute 的最佳使用方式。
6. **TTT vs Thinking Tokens 之争**：TTT4LC 从理论和实验两条线证明，在长上下文场景下 static attention 的 score dilution 使 thinking tokens 失效，而 query-only TTT 通过梯度更新打破 attention 的静态限制，以相同或更少 FLOPs 获得大幅提升。

## 关键问题

- 推理时更新参数到底是在“记忆上下文”，还是在“适配分布”？
- fast weights / LoRA / ordinary weights 作为测试时状态，各自的容量、稳定性和计算开销边界在哪里？
- next-token prediction、input perplexity 和 reconstruction-style objective 哪个更适合 LLM 的 test-time adaptation？
- 如何防止 test-time learning 在连续输入流中累积偏差、遗忘原知识或被恶意上下文污染？
