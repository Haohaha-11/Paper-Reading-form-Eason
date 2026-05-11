# Unsupervised Layer-Wise Dynamic Test Time Adaptation for LLMs

## 论文元信息

**作者**: Longhuan Xu; Cunjian Chen; Feng Yin
**会议/年份**: arXiv 2026
**arXiv**: [2602.09719](https://arxiv.org/abs/2602.09719) | [PDF](https://arxiv.org/pdf/2602.09719)
**主题**: TTT in LLM
**阅读顺序**: TTT in LLM topic paper 4
**MinerU task**: `01cdfdb9-2c7b-470f-8259-49bd6ec24b19`

## 一句话总结

这篇把无监督、单样本、adapt-and-reset 的 LLM TTA 做成一个可学习的“更新强度控制”问题：用 prompt NLL 更新 LoRA，但让 SCALENET 根据 prompt、层、Q/V 投影和 TTA step 动态缩放学习率，从而缓解固定学习率在少步更新中的失稳。

## 核心贡献

1. **无监督 sample-specific TTA 定位清楚**: 每个 prompt 独立适应，只用 prompt token 的 next-token NLL，不看 gold answer，生成后重置 LoRA 状态。
2. **把 input perplexity / prompt NLL 的风险显式化**: prompt-only loss 可能提升 prompt 边际拟合，却不一定提升 answer 条件分布；固定 LR 容易把单样本高方差梯度放大成 destructive drift。
3. **LoRA-only test-time update**: 冻结 backbone，只在 attention query/value projections 上更新 LoRA，降低单 prompt 反向传播的成本和破坏面。
4. **Layer-wise hypernetwork control**: SCALENET 输出非负学习率 multiplier，对每层、每个 TTA step、Q/V 投影分别控制更新强度，比全局 LR 或 step-wise schedule 更细。
5. **Dynamic control + first-order training**: 训练时 unroll 与推理相同的 K-step TTA，再用 answer loss 学超网络；为避开二阶 Hessian/FlashAttention 支持问题，丢掉二阶路径做 first-order approximation。
6. **实验直接击中固定 LR instability**: XSum/SQuAD/NQ/AdaptEval、Llama/Qwen 多模型结果显示，固定 LR 初期可能有小收益，但多步后 NLL/ROUGE 常崩；layer-wise dynamic control 更稳，且主要收益集中在第一步。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：sample-specific TTA、固定 LR 失稳、LoRA + hypernetwork |
| [01 - Introduction](sections/01-introduction.md) | 动机：instance objective mismatch、adapt-and-reset、SCALENET |
| [02 - Related Work](sections/02-related-work.md) | LLM、TTA、LoRA 背景；与 TLM/SLOT/TENT 的位置关系 |
| [03 - Method](sections/03-method.md) | prompt NLL 推导、LoRA 更新、layer-wise scaler、first-order hypernetwork training |
| [04 - Experiments](sections/04-experiments.md) | 实现细节、数据集、NLL/ROUGE 主结果、Figure 3-5、Limitations |
| [05 - Conclusion](sections/05-conclusion.md) | 结论：learned fine-grained control 比 handcrafted schedule 更必要 |
| [06 - References](sections/06-references.md) | 参考文献原文 |
| [07 - Appendix Prompts](sections/07-appendix-prompts.md) | XSum/SQuAD/NQ/AdaptEval prompt format，解释 prompt-only TTA 为什么依赖完整任务上下文 |

## 关键数字

| 指标 | 数值 |
|------|------|
| SCALENET | 2-layer MLP，hidden size 128 |
| SCALENET 训练优化器 | AdamW，learning rate $10^{-4}$ |
| 动态 TTA base LR | $10^{-2}$，再乘 hypernetwork scaler |
| 固定 LR baseline | $5 \times 10^{-2}$，所有层和 step 共用 |
| LoRA 配置 | rank $r=4$，LoRA $\alpha=16$ |
| LoRA 初始化 | 每个 prompt 重置；$B=0$，$A \sim 10^{-2}\mathcal{N}(0,I)$ |
| 最大 TTA step | $K_{\max}=5$，训练时随机采样 $K \in \{0,\ldots,K_{\max}\}$ |
| 训练/测试规模 | 每个 dataset-model pair 约 30k train samples，300 test samples |
| 更新位置 | attention query/value projections 的 LoRA 参数 |
| backbone dtype | bfloat16 |
| 中等模型 | Llama-3.2-3B / 3B-Instruct，Qwen3-4B / 4B-Instruct |
| 大模型 AdaptEval | Llama3.3-70B-Instruct，Qwen3-32B |
| 数据集 | XSum、SQuAD、NQ-Open、AdaptEval |
| Llama70B AdaptEval NLL | No TTA 2.2114；Layer-wise 1 step 1.6598；5 steps 约 1.7048 |
| Llama70B fixed LR instability | 1 step 2.1907 后 2/3/4/5 steps 约 5.8488 / 9.6803 / 11.5891 / 11.4970 |
| Qwen32B AdaptEval NLL | No TTA 2.1805；Layer-wise 4 steps 1.8739；5 steps 1.8889 |
| XSum ROUGE-Lsum | Llama3B-Instruct 0.1821 → 0.2090；Qwen4B-Instruct 0.1700 → 0.2247，均为 layer-wise 5 steps |
| SQuAD ROUGE-Lsum | Llama3B-Instruct 0.7080 → 0.7353；Qwen4B-Instruct 0.7722 → 0.8306，均为 layer-wise 5 steps |
| NQ-Open ROUGE-Lsum | Qwen4B-Instruct 0.1320 → 0.1706，但 Llama3B-Instruct 0.2766 → 0.2507，收益不稳定 |
| AdaptEval ROUGE-Lsum | Llama70B 0.2327 → 0.2733；Qwen32B 0.0987 → 0.1043，均为 layer-wise 5 steps |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. Prompt episode | 单个测试 prompt $x$ | 为该 prompt 重置 LoRA，backbone 冻结 | fresh LoRA state $\Phi^{(0)}$ |
| 2. LLM forward | prompt tokens | teacher-forced forward，取 first-layer 与 last-layer hidden states 的 mean pooling | prompt representation $h(x) \in \mathbb{R}^{2d}$ |
| 3. Hypernetwork control | $h(x)$、当前 step $k$、总 step $K$ | SCALENET 输出非负 scaler，区分 layer 与 Q/V projection | $s_i^{(k)}$ |
| 4. Prompt-only loss | prompt token sequence | 计算 input perplexity / next-token NLL $-\log P(x;\Phi^{(k)})$ | 自监督 TTA loss |
| 5. LoRA update | prompt loss gradient + scaler | 用 $\eta s_i^{(k)}$ 更新 query/value LoRA，backbone 不动 | $\Phi^{(k+1)}$ |
| 6. K-step dynamic TTA | 重复 step 2-5 | 第一更新通常更激进，后续快速 damping | prompt-specific adapted LoRA |
| 7. Answer generation/eval | adapted model + prompt | 生成答案或 teacher-forced 计算 answer NLL | response $y$ / NLL / ROUGE-Lsum |
| 8. Reset | 该 prompt 结束 | 丢弃 LoRA 状态 | 下一 prompt 重新开始 |

**整体输入**: 无标签测试 prompt。
**整体输出**: 经过单样本 LoRA TTA 后的回答；控制信号是 prompt-conditioned、layer-wise、step-wise 的学习率 multiplier。

## 优缺点与还能做什么

### 优点

- 问题切得准：它不是泛泛做“测试时再训练”，而是专门处理最常见也最脆的 unsupervised sample-specific adapt-and-reset 场景。
- 机制抓住关键旋钮：固定 LR 的失败在这类单样本少步 TTA 中很自然，学习 per-layer/per-step scaler 比手调 schedule 更合理。
- LoRA 更新范围克制：只动 Q/V LoRA，避免全参 update 对推理稳定性和显存造成更大压力。
- 实验证据能解释机制：Figure 3 的 fixed LR 崩溃、Figure 4 的 Q/V 层间差异、Figure 5 的首步峰值和后续衰减互相支撑。
- ROUGE 结果没有只停留在 NLL：XSum/SQuAD 上 layer-wise 5 steps 能转化为明显生成质量提升。

### 局限 / 风险

- SCALENET 训练仍需要 gold answer loss；推理时无监督，但控制器本身不是完全无监督学出来的。
- 每个 dataset-model pair 训练约 30k 样本，说明当前证据更像任务/模型定制控制器，而不是一次训练跨域泛化的通用 TTA policy。
- prompt-only NLL 是代理目标；NQ-Open 和 AdaptEval 这类缺少完整答案信息或依赖推理的任务收益更小、更不稳定。
- per-prompt 反向传播和多步 LoRA 更新会增加延迟、显存和服务复杂度，论文没有给出系统级吞吐/成本评估。
- LoRA 每个 prompt 随机初始化会引入额外方差；论文提到合适 scale 甚至依赖 LoRA initialization，这会影响可复现和部署稳定性。
- 固定 LR baseline 很容易被攻击为“手调不充分”；虽然 step-wise ablation 缓解了这个问题，但还缺少更强 optimizer / trust-region / clipping baseline。

### 还能做什么

- 训练跨任务/跨模型的通用 SCALENET，检验它是否学到真正的层敏感性，而不是 dataset-model pair 的经验表。
- 引入 reset/rollback/guardrail：当 prompt NLL 降低但 answer proxy 或 generation quality 变差时回滚 LoRA。
- 比较 Adam/SGD、layer-wise clipping、trust region、梯度归一化等更强 TTA optimizer baseline。
- 把 scaler 输入从 mean-pooled hidden states 扩展为 task type、prompt uncertainty、gradient statistics 或 LoRA state statistics。
- 做在线服务测量：单 prompt latency、显存、batching 难度、KV cache 兼容性，以及多用户隔离成本。
- 在 adversarial / noisy / prompt injection 场景下测动态 TTA 是否会把恶意 prompt 写进临时 LoRA。

## 阅读 Q&A 记录

- **Q: 这篇和 TLM 的关系是什么？**
  A: TLM 提出 LLM test-time learning 可用 input perplexity / prompt NLL 和 LoRA 更新；这篇接受这个目标，但指出固定 LR 在单 prompt 少步更新中不稳，于是学习 layer-wise dynamic LR 控制。

- **Q: 为什么必须 sample-specific 并 reset？**
  A: prompt 流异质且非平稳，把一个 prompt 的 LoRA 状态带到下一个 prompt 容易污染；如果有稳定目标域数据，offline 或 continual fine-tuning 更合适。

- **Q: prompt NLL 为什么可能提升 answer？**
  A: 论文用一阶 Taylor 展开解释：如果 prompt gradient $g_x$ 与 answer gradient $g_y$ 的内积为正，prompt-only 更新会增加 $\log P(y|x)$。问题是这个条件无法手写保证，所以用 hypernetwork 学控制约束。

- **Q: SCALENET 具体控制什么？**
  A: 它不直接生成 LoRA 权重，而是生成非负学习率 multiplier，乘到 base LR 上，作用于每个 step、每层和 query/value LoRA projection。

- **Q: 为什么 layer-wise 比 step-wise 重要？**
  A: step-wise 只能学“第几步该多大”，相当于可学习 LR schedule；Figure 4 显示相邻层、同层 Q/V 的合适 scale 可差几个数量级，统一 scale 会过度更新敏感层。

- **Q: 固定 LR instability 的证据是什么？**
  A: Llama70B AdaptEval 中 fixed LR NLL 从 No TTA 2.2114 到 1 step 2.1907 只有小幅下降，但 2/3/4/5 steps 约升到 5.8488 / 9.6803 / 11.5891 / 11.4970；ROUGE 上 Llama70B AdaptEval fixed 5 steps 也掉到 0.0029。

- **Q: 生成指标是否真的变好？**
  A: XSum 和 SQuAD 很清楚，layer-wise 5 steps 分别把 Llama3B-Instruct ROUGE-Lsum 提到 0.2090 和 0.7353，把 Qwen4B-Instruct 提到 0.2247 和 0.8306；NQ-Open/AdaptEval 更不稳定。

- **Q: 这是不是“纯无监督”方法？**
  A: 推理时是无监督的，只用 prompt；但 SCALENET 训练用 answer loss 学会怎样调学习率，所以更准确说是 supervised-learned controller for unsupervised test-time adaptation。

## 📊 Citation Landscape

> 数据来源: 本目录 `semantic-scholar/` JSON。`detail.json`、`references.json`、`recommendations.json` 均可用；统计为 Semantic Scholar 当前缓存值，可能随时间变化。
> [Semantic Scholar](https://www.semanticscholar.org/paper/1ec7975d06044ed111b4a97a708e8a1e353dd1dc) | [Connected Papers](https://www.connectedpapers.com/main/2602.09719)

### TLDR (Semantic Scholar 自动生成)

> 暂缺：`detail.json` 中 `tldr.text` 为 `null`。

### 引用统计

| 指标 | 数值 |
|------|------|
| Semantic Scholar Paper ID | `1ec7975d06044ed111b4a97a708e8a1e353dd1dc` |
| Corpus ID | `285463170` |
| DOI | `10.48550/arXiv.2602.09719` |
| 参考文献数 | 23 |
| 被引次数 | 0 |
| Influential Citations | 0 |
| 推荐论文条目数 | 100 |

### 参考文献 Top by citations

| 论文 | 年份 | Venue | Citations |
|------|------|-------|-----------|
| [Training language models to follow instructions with human feedback](https://www.semanticscholar.org/paper/d766bffc357127e0dc86dd69561d5aeb520d6f4c) | 2022 | NeurIPS | 20414 |
| [ROUGE: A Package for Automatic Evaluation of Summaries](https://www.semanticscholar.org/paper/60b05f32c32519a809f21642ef1eb3eaf3848008) | 2004 | ACL | 20238 |
| [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://www.semanticscholar.org/paper/1b6e810ce0afd0dd093f789d2b2742d047e316d5) | 2022 | NeurIPS | 17863 |
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://www.semanticscholar.org/paper/659bf9ce7175e1ec266ff54359e2bd76e0b7ff31) | 2020 | NeurIPS | 13579 |
| [SGDR: Stochastic Gradient Descent with Warm Restarts](https://www.semanticscholar.org/paper/b022f2a277a4bf5f42382e86e4380b96340b9e86) | 2016 | ICLR | 10356 |
| [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://www.semanticscholar.org/paper/05dd7254b632376973f3a1b4d39485da17814df5) | 2016 | EMNLP | 9305 |
| [Transformer-XL: Attentive Language Models beyond a Fixed-Length Context](https://www.semanticscholar.org/paper/c4744a7c2bb298e4a52289a1e085c71cc3d37bc6) | 2019 | ACL | 4297 |
| [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.semanticscholar.org/paper/fdacf2a732f55befdc410ea927091cad3b791f13) | 2021 | JMLR | 3800 |
| [Don't Give Me the Details, Just the Summary!](https://www.semanticscholar.org/paper/305b2cf37e5dece81e95c92883d5a6e28ac93b22) | 2018 | EMNLP | 2019 |
| [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://www.semanticscholar.org/paper/180c78b132f6369a384d22a9529551d86c8788d3) | 2021 | ICLR | 1732 |

### 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | Citations | 链接 |
|------|------|-----------|------|
| Test-Time Instance-Specific Parameter Composition | 2026 | 0 | [Semantic Scholar](https://www.semanticscholar.org/paper/80c66d2b868beede25df341ccbeb5a9e83ecd277) |
| ShadowPEFT: Shadow Network for Parameter-Efficient Fine-Tuning | 2026 | 0 | [Semantic Scholar](https://www.semanticscholar.org/paper/9b162eb6bc718edc2e808d2fc2bf7b1f10e71de3) |
| Training-Free Test-Time Contrastive Learning for Large Language Models | 2026 | 0 | [Semantic Scholar](https://www.semanticscholar.org/paper/1427454a24da2dcd1ced9dd59b3aed86b3a7e4bc) |
| TLoRA+: A Low-Rank Parameter-Efficient Fine-Tuning Method for Large Language Models | 2026 | 0 | [Semantic Scholar](https://www.semanticscholar.org/paper/929bc0a5291c5f0b2bad0b2b0b7c0ca048715083) |
| Rethinking Adapter Placement: A Dominant Adaptation Module Perspective | 2026 | 0 | [Semantic Scholar](https://www.semanticscholar.org/paper/c2e4a2a187d31c452ab18e8583f4f637936465fb) |
| TARo: Token-level Adaptive Routing for LLM Test-time Alignment | 2026 | 0 | [Semantic Scholar](https://www.semanticscholar.org/paper/1b14f0d4b04a7993b7ea62a4a79cd5a3e985ff33) |
| Not All Directions Matter: Towards Structured and Task-Aware Low-Rank Model Adaptation | 2026 | 5 | [Semantic Scholar](https://www.semanticscholar.org/paper/6a34fbd59a656b2b3bb2a35cfa9b4bb235046ce1) |
| Aletheia: Gradient-Guided Layer Selection for Efficient LoRA Fine-Tuning Across Architectures | 2026 | 0 | [Semantic Scholar](https://www.semanticscholar.org/paper/1893aed0ad01bcdce84888d644c91ab39e5a3f0c) |
| In-Place Test-Time Training | 2026 | 1 | [Semantic Scholar](https://www.semanticscholar.org/paper/3008be7598614b60f81742bc80a47d63d1c604b7) |
| AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation | 2026 | 0 | [Semantic Scholar](https://www.semanticscholar.org/paper/5b91002c738e6a97a37560bae1dff774a0ce31f5) |

## BibTeX

```bibtex
@article{xu2026unsupervisedlayerwisedynamictta,
  title={Unsupervised Layer-Wise Dynamic Test Time Adaptation for LLMs},
  author={Longhuan Xu and Cunjian Chen and Feng Yin},
  journal={arXiv preprint arXiv:2602.09719},
  year={2026}
}
```
