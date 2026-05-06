# Latent-Space-Processing

这个 topic 关注 LLM/VLM 的 latent-space processing：把额外推理、记忆整理、CoT 压缩或上下文再加工放到连续表示、hidden state 或 kv-cache 中完成，而不是完全依赖显式自然语言思维链。

## 论文列表

| 顺序 | 论文 | 会议 | 方法特点 |
|------|------|------|----------|
| 1 | [Differentiable-Cache-Augmentation](./%5BArxiv%202024%5D%20Differentiable-Cache-Augmentation/) | Arxiv 2024 | DCA 用离线 coprocessor 读取 frozen LLM 的 kv-cache，生成动态 latent embeddings 追加回 cache，实现 cache-level latent deliberation。 |
| 2 | [Compressed-Chain-of-Thought](./%5BArxiv%202024%5D%20Compressed-Chain-of-Thought/) | Arxiv 2024 | CCoT 把显式 CoT hidden states 压缩成 contentful continuous contemplation tokens，用 compression ratio 控制准确率/延迟折中。 |

## 推荐阅读顺序

1. **Differentiable-Cache-Augmentation**：先看“latent processing 可以直接发生在 kv-cache 上”，理解 frozen decoder + coprocessor 的系统边界。
2. **Compressed-Chain-of-Thought**：再看“显式 CoT 如何压缩成 dense latent tokens”，理解 contentful contemplation token 与 token-level CoT 的关系。

## 方法对比

| 维度 | DCA | CCoT |
|------|-----|------|
| 处理对象 | frozen LLM 的 kv-cache | explicit CoT 的 hidden states |
| latent 来源 | coprocessor 读 cache 动态生成 | full reasoning chain hidden-state subset 的压缩近似 |
| base model 是否冻结 | decoder 冻结，训练 coprocessor | 使用 LoRA 训练 CCOT/DECODE 模块 |
| 推理形态 | 可选/异步 cache augmentation，一次生成多个 latents | autoregressive 生成 variable-length contemplation tokens |
| 主要收益 | 多 benchmark perplexity/accuracy 提升 | GSM8K 上比无思考/PAUSE 更好的 accuracy-latency 折中 |
| 最大风险 | latent 不可解释，coprocessor 成本和触发策略未解决 | 需要 CoT teacher，压缩后仍弱于完整 CoT |

## 横向数据流与研究机会

| 层级 | 输入 | 中间变化 | 输出 | 可追问的问题 |
|------|------|----------|------|--------------|
| Cache processing | prompt cache | coprocessor 对 kv-cache 生成 latent embeddings | augmented cache | 何时调用 coprocessor、多少 latents 才划算？ |
| CoT compression | query + explicit reasoning chain | hidden-state subset selection + dense token generation | compressed contemplation tokens | 压缩 latent 能否还原 faithful reasoning chain？ |
| Compute allocation | 简单/困难问题 | 调节 latent 数或 compression ratio | accuracy-latency trade-off | 能否训练 difficulty-aware latent budget controller？ |
| Interpretability | hidden states / kv-cache / latents | probing、decoding、causal intervention | 可审计 latent reasoning evidence | latent 出错时如何定位是哪一步错？ |

## 统一阅读问题

- **Q1: latent-space processing 和显式 CoT 是替代关系吗？**
  A: 不是完全替代。DCA 更像 cache-level 增强，CCoT 是 CoT 压缩；二者都说明显式 token 不是唯一推理载体，但可解释性仍不如自然语言链。
- **Q2: 为什么先读 DCA 再读 CCoT？**
  A: DCA 先建立“latent 可以直接写入模型内部状态”的观念；CCoT 再说明“显式 CoT 的 hidden trajectory 可以被压缩成 latent tokens”。
- **Q3: 这个 topic 最值得做的下一步是什么？**
  A: 把 latent budget control、faithfulness probing、latent-to-text inspection 和任务难度估计连起来，避免只报告 accuracy/latency。
