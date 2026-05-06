# Compressed Chain of Thought: Efficient Reasoning through Dense Representations

**作者**: Jeffrey Cheng; Benjamin Van Durme
**会议**: Arxiv 2024
**链接**: [arXiv 2412.13171](https://arxiv.org/abs/2412.13171) | [HTML](https://arxiv.org/html/2412.13171) | [PDF](https://arxiv.org/pdf/2412.13171)
**阅读顺序**: Latent-Space-Processing 课题第 2 篇
**MinerU task**: `25d7520c-06bd-4c71-a62f-10d149ea2a18`

## 一句话总结

CCoT 把显式 CoT hidden states 压缩成可变长度的 contentful continuous contemplation tokens，用较少的 latent token 换取比无思考/PAUSE 更好的 GSM8K 推理准确率和远低于完整 CoT 的延迟。

## 核心贡献

1. **Contemplation token taxonomy**: 用 contentful / continuous / inference 方式统一 CoT、Pause、Filler、COCONUT 和 CCoT。
2. **Compressed CoT hidden-state supervision**: 从完整 reasoning chain 的 hidden states 中选子集作为 gold dense targets。
3. **Two-stage adaptation**: 先训练 $CCOT_\varphi$ 生成 contemplation tokens，再训练 $DECODE_\psi$ 从 query + latent 解码答案。
4. **Adjustable compression ratio**: 通过 $r$ 控制 $k=\lceil rm\rceil$，实现 accuracy-latency trade-off。
5. **Autoregressive continuous reasoning**: 用中间层 hidden state 作为下一 latent 输入，提供比并行 pause token 更强的 sequential depth。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 核心框架 |
| [01 - Introduction](sections/01-introduction.md) | CoT 延迟动机与贡献 |
| [02 - Related Work](sections/02-related-work.md) | contemplation token 分类与相关工作 |
| [03 - Contemplation Tokens](sections/03-contemplation-tokens.md) | 符号、定义和压缩观察 |
| [04 - Approach](sections/04-approach.md) | CCOT/DECODE 两阶段训练 |
| [05 - Experiments](sections/05-experiments.md) | GSM8K accuracy/latency |
| [06 - Further Discussion](sections/06-further-discussion.md) | 超参和理论直觉 |
| [07 - Conclusion](sections/07-conclusion.md) | 总结 |
| [08 - Appendix](sections/08-appendix.md) | References、层选择和理论补充 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Base model | Llama2-7B-chat |
| Dataset | GSM8K |
| LoRA ranks | $\varphi$: 128, $\psi$: 64 |
| Compression ratios | $r=0.05$ (20x), $r=0.10$ (10x), $r=1$ full CoT |
| CCoT 20x | EM 0.151, decode 0.49s |
| CCoT 10x | EM 0.179, decode 0.78s |
| Full CoT | EM 0.315, decode 8.10s |
| PAUSE 10x | EM 0.099, decode 0.37s |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. Teacher trace | query $w$, full CoT $t$, answer $a$ | base LM 计算 full sequence hidden states | reasoning-chain hidden states $\hat t$ |
| 2. Subset selection | $\hat t_{1:m}$ + compression ratio $r$ | scorer 选出 $k=\lceil rm\rceil$ 个 gold states | gold compressed states $z_{1:k}$ |
| 3. CCOT training | query hidden state + previous latent state | $CCOT_\varphi$ layerwise MSE 拟合 gold states | generated contemplation tokens $\hat z$ |
| 4. DECODE training | query + generated contemplation tokens + answer prefix | $DECODE_\psi$ cross-entropy 预测答案 | latent-conditioned answer decoder |
| 5. Inference | query only | autoregressively roll out latent tokens until END，再解码答案 | compressed reasoning answer |

**整体输入**: 问题 query。
**整体输出**: 不生成完整自然语言 CoT 的 latent-conditioned answer。

## 优缺点与还能做什么

### 优点
- 明确把 CoT 延迟问题转成 compression ratio 的可调问题。
- latent tokens 蒸馏自 reasoning chain hidden states，比 noncontentful pause tokens 更有内容来源。
- 中间层 autoregressive rollout 提供 sequential compute depth，适合多步推理任务。

### 局限 / 风险
- GSM8K 单任务实验较窄，且完整 CoT 准确率仍显著更高。
- compressed latent 本身不可读，虽然作者提出未来可解码检查，但本文未真正实现审计。
- 训练需要可用的 full reasoning chains，不适用于没有 CoT teacher 的任务。

### 还能做什么
- 训练一个 latent-to-text decoder，检查 compressed tokens 是否真的保留 faithful reasoning。
- 做 adaptive $r$：按题目难度动态决定生成多少 contemplation tokens。
- 扩展到 MATH、BBH、代码、多跳 QA，检验 sequential latent depth 是否泛化。

## 阅读 Q&A 记录

- **Q: CCoT 和 DCA 有什么不同？**
  A: DCA 读 kv-cache 并追加 latent；CCoT 蒸馏显式 CoT hidden states 并生成 compressed latent。前者是 cache-level augmentation，后者是 reasoning-chain compression。
- **Q: 为什么 CCoT 比 PAUSE 准？**
  A: PAUSE token 主要提供额外计算宽度但没有内容；CCoT token 来源于 CoT hidden states，带有 teacher reasoning trace 的语义信息，并且 autoregressive 生成提供计算深度。
- **Q: 为什么完整 CoT 仍然更强？**
  A: 压缩带来信息损失，且 latent approximation 不一定完全拟合 gold hidden states。Table 2 显示 CCoT 是速度/准确率折中，而不是无损替代。

## 📊 Citation Landscape

> 数据来源: Semantic Scholar API 本次返回 rate limit / timeout；保留 arXiv 与 Connected Papers 入口，后续可补全。
> [Connected Papers](https://www.connectedpapers.com/main/2412.13171)

### TLDR (Semantic Scholar 自动生成)

> 暂缺。

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献数 | 暂缺 |
| 被引次数 | 暂缺 |
| Influential Citations | 暂缺 |

### 参考文献分组 (Top 5 per category, by citations)

- Semantic Scholar API 暂缺或限流；本次先保留论文 References 于 [08 - Appendix](sections/08-appendix.md)。

### 🔗 推荐论文 (Semantic Scholar Recommendations)

- 暂缺，待 API 恢复后补充。

## BibTeX

```bibtex
@article{cheng2024compressedcot,
  title={Compressed Chain of Thought: Efficient Reasoning through Dense Representations},
  author={Jeffrey Cheng and Benjamin Van Durme},
  journal={arXiv preprint arXiv:2412.13171},
  year={2024}
}
```
