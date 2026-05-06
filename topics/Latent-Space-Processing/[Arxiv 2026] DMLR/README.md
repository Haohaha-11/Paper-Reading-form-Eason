# Reasoning Within the Mind: Dynamic Multimodal Interleaving in Latent Space

**作者**: Chengzhi Liu; Yuzhe Yang; Yue Fan; Qingyue Wei; Sheng Liu; Xin Eric Wang
**会议**: Arxiv 2026
**版本**: v3, 2026-04-09
**链接**: [arXiv 2512.12623](https://arxiv.org/abs/2512.12623) | [HTML](https://arxiv.org/html/2512.12623) | [PDF](https://arxiv.org/pdf/2512.12623) | [Project](https://mllm-dmlr.github.io/)
**MinerU task**: `df2907eb-6e29-48d4-821e-dbec489ddf72`

## 一句话总结

DMLR 在测试时把少量 latent think tokens 当作“脑内草稿”，用 confidence-guided policy gradient 优化它们，并动态注入最相关视觉 patch，从而在不训练参数、不生成长 CoT 的情况下提升多模态推理。

## 核心贡献

1. **Dynamic Multimodal Latent Reasoning**: 提出 test-time、training-free 的多模态 latent reasoning 框架。
2. **Motivation measurement**: 发现视觉依赖集中在少数 reasoning tokens，且 confidence 与正确性、faithfulness、visual grounding 相关。
3. **Confidence-guided latent optimization**: 用 top-k truncated entropy 构造 reward，通过 policy gradient 更新 latent think tokens。
4. **Dynamic Visual Injection**: 每轮选择 candidate visual patches，并用 reward-gated best patch 机制过滤冗余视觉信息。
5. **跨模型跨任务验证**: 在 Qwen2.5/Qwen3/R1-OneVision/VLAA-Thinking 与 7 个 benchmark 上验证 reasoning、perception、efficiency。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要、项目入口、Figure 1 范式对比 |
| [01 - Introduction](sections/01-introduction.md) | 三类现有范式、动态视觉使用动机、贡献 |
| [02 - Related Work](sections/02-related-work.md) | explicit reasoning 与 latent reasoning 脉络 |
| [03 - Preliminary and Motivation](sections/03-preliminary-and-motivation.md) | Visual Dependency Score、Confidence Gain、三条 observation |
| [04 - Methodology](sections/04-methodology.md) | latent token 优化、confidence reward、DVI、理论分析 |
| [05 - Experiments](sections/05-experiments.md) | 主结果、DVI 消融、超参消融、效率和可视化 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结与部署边界 |
| [07 - Appendix](sections/07-appendix.md) | 数据集、prompt、参数、case study、references |

## 关键数字

| 指标 | 数值 |
|------|------|
| Benchmark | MathVista mini, MathVision mini, MM-Math, HallusionBench, MMVP, MMStar, ScienceQA |
| Backbone | Qwen2.5-VL-3B/7B, Qwen3-VL-4B/8B, R1-OneVision, VLAA-Thinking |
| latent think tokens | 4 |
| candidate visual patches | $m=2$ |
| max patches per iteration | 16 |
| optimization steps | 15 |
| learning rate | $1 \times 10^{-3}$ |
| perturbation | $\sigma=0.1$ / 10%, decay 0.95 |
| Qwen2.5-VL-7B gain | math +1.5%, visual reasoning +0.9% |
| R1-OneVision gain | math +4.5%, visual reasoning +3.45% |
| VLAA-Thinking gain | average +2.43% |
| DVI ablation | MathVision 0.340 vs all injection 0.327 vs w/o injection 0.321 |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. Base encoding | text query $Q$ + image embeddings $Z$ | MLLM 编码图文上下文 | 原始 multimodal hidden states |
| 2. Latent insertion | $Q, Z$ | 插入 $L=4$ latent think tokens $T$ | 可优化 latent reasoning state |
| 3. Exploration | 当前 $T^{(t)}$ | 加 Gaussian perturbation $\xi \sim N(0,\sigma^2I)$ | perturbed latent tokens $T'^{(t)}$ |
| 4. Confidence reward | latent token distribution | 计算 top-k truncated entropy，取 complement | reward $R(T^{(t)})$ |
| 5. Policy update | reward + perturbation | REINFORCE-style latent policy gradient | 更新后的 latent tokens |
| 6. Dynamic Visual Injection | attention-selected candidate patches | reward-gated 更新 $V_{best}$ | 少量相关 visual patches |
| 7. Decode | $Q, Z, T^*, V_{best}$ | 原 MLLM 解码 | final answer |

**整体输入**: 图像 + 问题。
**整体输出**: 不依赖长文本 CoT 的 multimodal answer。
**核心中间表示**: optimized latent think tokens + reward-filtered visual patches。

## 优缺点与还能做什么

### 优点
- 不训练模型参数，适合接到现有 MLLM 上做 test-time scaling。
- 同时处理 reasoning 和 perception，而不是只增加语言 CoT 长度。
- DVI 证明“相关视觉 patch”比全量视觉注入更稳。
- 对 reasoning model 和 non-reasoning model 都有收益。

### 局限 / 风险
- confidence 是 proxy，不是 correctness oracle；高自信仍可能错。
- 需要 eager attention 和内部 attention map，工程上比普通推理更侵入。
- 15 步 latent optimization 仍有额外推理成本，吞吐部署要单独评估。
- 主要实验是 benchmark accuracy，OOD、医学、遥感等高风险域未验证。
- case study 中正确答案的解释仍可能不完全严谨，faithfulness 不能只靠答案判断。

### 还能做什么
- 做 confidence calibration：判断哪些任务上 entropy reward 会鼓励错误自信。
- 做 adaptive compute：按 reward 收敛情况动态停止，而不是固定 15 steps。
- 在更强闭源/大模型、长视频、多图、多页面文档上验证 DVI 是否稳定。
- 增加 failure case 分类：视觉 patch 选错、latent 优化发散、confidence-reasoning 不对齐。
- 把 DVI 与显式可解释 grounding 输出结合，提升可审计性。

## 阅读 Q&A 记录

- **Q: DMLR 和显式 CoT 最大区别是什么？**
  A: 显式 CoT 通过生成自然语言中间步骤增加推理；DMLR 在 latent space 优化 think tokens，避免长链 token 生成，并用动态视觉 patch 校正 grounding。

- **Q: 推理时没有 ground truth，为什么能优化 confidence？**
  A: 第 3 节用 ground-truth Confidence Gain 做分析；第 4 节实际 reward 使用 top-k truncated entropy，不需要标签。

- **Q: 为什么不把所有 visual patches 都注入？**
  A: Table 2 显示全量注入会引入冗余视觉信息，MathVista 和 ScienceQA 甚至低于 w/o Injection；DVI 只保留 reward 更高的 patches。

- **Q: DMLR 是否真的没有额外推理成本？**
  A: 没有额外长文本生成成本，但有 test-time latent optimization 成本。Figure 11 说明综合 accuracy/efficiency trade-off 更好，不代表零开销。

- **Q: 默认超参怎么设？**
  A: 4 latent tokens、$m=2$ candidate patches、15 optimization steps、lr=1e-3、$\sigma=0.1$、decay 0.95、seed 42。

## 📊 Citation Landscape

> Semantic Scholar `detail` endpoint 本次返回 `429 Too Many Requests`，因此 TLDR、被引次数和 influential citation count 暂缺；`references` 与 `recommendations` 已成功缓存到 `semantic-scholar/`。
> [Semantic Scholar 查询](https://www.semanticscholar.org/search?q=Reasoning%20Within%20the%20Mind%20Dynamic%20Multimodal%20Interleaving%20in%20Latent%20Space) | [Connected Papers](https://www.connectedpapers.com/main/2512.12623)

### TLDR (Semantic Scholar 自动生成)

> 暂缺，Semantic Scholar detail endpoint 当前限流。

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献数 | detail 暂缺；references 缓存 53 条 |
| 被引次数 | 暂缺 |
| Influential Citations | 暂缺 |

### 参考文献分组 (Top references by citation count)

- **优化与可视化基础**: Visualizing Data using t-SNE (2008, 48754 cites), REINFORCE / Simple Statistical Gradient-Following Algorithms (2004, 10088)。
- **Backbone / MLLM**: Qwen2.5-VL Technical Report (2025, 4411), InternVL3.5 (2025, 701)。
- **Multimodal CoT**: Learn to Explain / ScienceQA Thought Chains (2022, 2225), Multimodal Chain-of-Thought Reasoning (2023, 831)。
- **视觉推理评测**: HallusionBench (2023, 499), Eyes Wide Shut / MMVP (2024, 699), Are We on the Right Way for Evaluating LVLMs? (2024, 747)。
- **RLVR / reasoning model**: Vision-R1 (2025, 501)。

### 🔗 推荐论文 (Semantic Scholar Recommendations)

1. Language-Guided Token Compression with Reinforcement Learning in Large Vision-Language Models (2026, 8 cites, arXiv:2603.13394)
2. Thinking in Uncertainty: Mitigating Hallucinations in MLRMs with Latent Entropy-Aware Decoding (2026, 2 cites, arXiv:2603.13366)
3. ΔVLA: Prior-Guided Vision-Language-Action Models via World Knowledge Variation (2026, 2 cites, arXiv:2603.08361)
4. EndoCoT: Scaling Endogenous Chain-of-Thought Reasoning in Diffusion Models (2026, 1 cite, arXiv:2603.12252)
5. Bridging Perception and Reasoning: Token Reweighting for RLVR in Multimodal LLMs (2026, 1 cite, arXiv:2603.25077)
6. VLA-Thinker: Boosting Vision-Language-Action Models through Thinking-with-Image Reasoning (2026, 1 cite, arXiv:2603.14523)
7. Improving Visual Reasoning with Iterative Evidence Refinement (2026, 1 cite, arXiv:2603.14117)
8. Insight-V++: Towards Advanced Long-Chain Visual Reasoning with Multimodal Large Language Models (2026, 1 cite, arXiv:2603.18118)
9. Thinking in Streaming Video (2026, 1 cite, arXiv:2603.12938)
10. SVAgent: Storyline-Guided Long Video Understanding via Cross-Modal Multi-Agent Collaboration (2026, 1 cite, arXiv:2604.05079)

## BibTeX

```bibtex
@article{liu2026reasoningminddmlr,
  title={Reasoning Within the Mind: Dynamic Multimodal Interleaving in Latent Space},
  author={Liu, Chengzhi and Yang, Yuzhe and Fan, Yue and Wei, Qingyue and Liu, Sheng and Wang, Xin Eric},
  journal={arXiv preprint arXiv:2512.12623},
  year={2026}
}
```
