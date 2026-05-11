# Tech-repoorts

工业界技术报告 / frontier model system reports：重点读模型架构、训练基础设施、推理系统、评测协议和工程 trade-off，而不只看算法贡献。

## 报告列表

| 报告 | 来源 | 方法特点 |
|------|------|----------|
| [Claude Opus 4.7 System Card](./%5BSystem%20Card%202026%5D%20Claude-Opus-4.7/) | Anthropic System Card 2026 | Anthropic 2026 年 4 月的 Opus 系统卡，重点记录 frontier Claude 的 coding/agentic/search/long-context 能力、安全评估、RSP/ASL 部署边界和 agentic misuse / prompt injection 风险。 |
| [Mythos Preview System Card](./%5BSystem%20Card%202026%5D%20Mythos-Preview/) | Anthropic System Card 2026 | Anthropic 2026 年 4 月的 Mythos Preview 系统卡，适合作为 Claude 产品线非标准命名模型的能力定位、安全边界和部署策略案例。 |
| [Claude Sonnet 4.6 System Card](./%5BSystem%20Card%202026%5D%20Claude-Sonnet-4.6/) | Anthropic System Card 2026 | Anthropic 2026 年 2 月 Sonnet 线系统卡，覆盖 coding、agentic/search、long context、multimodal、harmlessness、alignment、agentic safety 与 ASL-3 部署判断。 |
| [DeepSeek-V4-Pro](./%5BTech%20Report%202026%5D%20DeepSeek-V4-Pro/) | Hugging Face Technical Report 2026 | DeepSeek-V4 系列报告覆盖 1M context MoE 模型的 CSA/HCA hybrid attention、mHC、Muon、>32T token 训练、post-training 与工业推理基础设施。 |
| [Attention-Residuals](./%5BTech%20Report%202026%5D%20Attention-Residuals/) | arXiv Technical Report 2026 | Kimi Team 将残差连接视为深度维度的信息聚合问题，用 Full/Block Attention Residuals 替换固定残差累加，并讨论可规模化训练/推理实现。 |

## 课题主线

1. **模型能力来自系统组合**：DeepSeek-V4-Pro 把 MoE、长上下文 attention、optimizer、训练数据、post-training 和推理框架作为整体优化；Claude system cards 则把能力、产品形态、安全策略和部署门槛一起公开。
2. **长上下文是架构、评测和安全共同问题**：1M context 不只靠 attention trick，也依赖 KV cache 布局、contextual parallelism、on-disk cache、rollout service、needle/RAG/search 评测和 prompt injection 防护。
3. **agentic coding 已经进入系统卡主线**：Claude Opus/Sonnet/Mythos 把 coding、sub-agent、browser/search、computer use 和工具调用作为核心能力评估，同时也把 agentic misuse、欺骗性行为和权限边界作为风险对象。
4. **残差路径不再是“默认不动”的管线**：mHC 与 AttnRes 都把 residual/state aggregation 作为可学习、可约束或可选择的模块，说明 frontier 模型开始重新设计深度维的信息流。
5. **技术报告读法要关注可落地性**：除了 benchmark，还要看计算开销、通信开销、推理延迟、cache 规模、安全缓解、训练稳定性、infra 复杂度和公开可复现程度。

## 关键问题

- 这些报告里的性能提升来自模型结构、数据/post-training、还是推理/评测设定？
- 长上下文压缩是在牺牲细粒度证据，还是通过局部窗口与稀疏选择保持可用信息？
- system card 里的 benchmark 提升是否能转化为实际 agent 工作流收益，尤其是 coding、search、computer use 和长上下文任务？
- ASL/RSP、安全评估和产品部署之间如何闭环：哪些风险被 mitigation 控制，哪些仍依赖使用场景和权限约束？
- residual/attention/cache 这些“基础管线”被重新设计后，训练稳定性和推理成本的真实边界在哪里？
- 工业报告公开的信息能否支撑复现，哪些关键细节仍然依赖闭源基础设施？
