# TTT in LLM

Test-Time Training / Test-Time Learning / long-context continual adaptation：围绕 LLM 在推理时用上下文或无标签测试数据更新 fast weights、LoRA 或部分权重，让模型把新信息压进参数并适应分布变化。

## 论文列表

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [In-Place-TTT](./%5BICLR%202026%5D%20In-Place-TTT/) | ICLR 2026 Oral | In-Place TTT 把 LLM MLP block 的 final projection matrix 当作 fast weights，用 LM-aligned objective 和 chunk-wise update 做可插拔 test-time training。 |
| [TTT-E2E](./%5BArxiv%202025%5D%20TTT-E2E/) | arXiv 2025 | TTT-E2E 将 long-context LM 视作 continual learning，用测试时 next-token prediction 更新权重，并用训练时 meta-learning 学适合测试时学习的初始化。 |
| [TLM](./%5BICML%202025%5D%20TLM/) | ICML 2025 | TLM 把 LLM test-time learning 表述为无标签测试数据的 input perplexity minimization，用高困惑度样本选择和 LoRA 更新做域适应。 |

## 课题主线

1. **更新对象不同**：In-Place TTT 更新 MLP final projection fast weights；TTT-E2E 更新 sliding-window Transformer 的部分权重并把 context 压入参数；TLM 用 LoRA 在测试域上做轻量更新。
2. **训练/测试耦合方式不同**：In-Place TTT 强调 drop-in 与从头预训练两种使用方式；TTT-E2E 在训练时 meta-learn 初始化、测试时 next-token 更新；TLM 更像无标签目标域适应，直接以测试输入 perplexity 为自监督目标。
3. **目标函数不同**：In-Place TTT 用 next-token-prediction aligned value target 替代 generic reconstruction；TTT-E2E 直接以内层 next-token loss 更新；TLM 最小化 input perplexity 并偏重高困惑度样本。
4. **应用压力不同**：In-Place TTT 和 TTT-E2E 主要服务长上下文利用与常数/低延迟推理；TLM 主要服务 domain shift、语言变体和专门领域适应。

## 关键问题

- 推理时更新参数到底是在“记忆上下文”，还是在“适配分布”？
- fast weights / LoRA / ordinary weights 作为测试时状态，各自的容量、稳定性和计算开销边界在哪里？
- next-token prediction、input perplexity 和 reconstruction-style objective 哪个更适合 LLM 的 test-time adaptation？
- 如何防止 test-time learning 在连续输入流中累积偏差、遗忘原知识或被恶意上下文污染？
