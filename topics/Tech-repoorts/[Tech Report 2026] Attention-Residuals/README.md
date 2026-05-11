# Attention Residuals

> Kimi Team; Guangyu Chen; Yu Zhang; Jianlin Su. Technical Report / arXiv, 2026.  
> arXiv: [2603.15031](https://arxiv.org/abs/2603.15031) · PDF: [paper.pdf](paper.pdf) · Source text: [full.md](full.md)

## 一句话总结

Attention Residuals 把 PreNorm Transformer 中“所有层输出按固定 1 权重累加”的残差流，改成沿深度维度做 softmax attention：每层用一个学习到的伪 query，在前面层或前面 block 的表示中按输入内容选择信息；Block AttnRes 再把跨层访问从 $O(Ld)$ 压到 $O(Nd)$，使其能接入 48B total / 3B activated 的 Kimi Linear 并在 1.4T tokens 训练中保持低系统开销。

## 核心贡献

1. **机制贡献：从固定累加到深度注意力。** 标准残差可展开为 $\boldsymbol h_l=\boldsymbol h_1+\sum_i f_i(\boldsymbol h_i)$，所有历史输出权重恒为 1；AttnRes 改为 $\boldsymbol h_l=\sum_i \alpha_{i\to l}\boldsymbol v_i$，其中 $\alpha$ 来自 softmax attention，使深度方向的信息聚合变成可学习、输入相关的选择。
2. **可扩展变体：Block AttnRes。** Full AttnRes 访问所有先前层输出；Block AttnRes 把层分成 $N$ 个 block，在 block 内保留普通累加，在 block 间注意力选择 block summary，将存储与跨流水线通信从 $O(Ld)$ 降到 $O(Nd)$。
3. **系统工程：让深度注意力可训练、可推理。** 报告给出 cache-based pipeline communication，避免虚拟流水线阶段重复传输历史 block；推理端用 two-phase inference，把跨 block query 批处理，再用 online softmax 合并 block 内顺序依赖，典型推理延迟开销小于 2%。
4. **实证结论：收益跨尺度且能落到大模型。** Scaling law 中 Block AttnRes 在 5.6 PFLOP/s-days 达到 baseline 约 1.25 倍计算量才达到的 loss；48B total / 3B activated Kimi Linear 训练 1.4T tokens 后，在全部评测任务上持平或优于 baseline。

## 📖 批读导航

| 顺序 | 文件 | 读法 |
| --- | --- | --- |
| 00 | [Abstract](sections/00-abstract.md) | 先抓住 PreNorm dilution、Full/Block AttnRes、系统开销与大模型结果。 |
| 01 | [Introduction](sections/01-introduction.md) | 看论文如何把残差从“梯度高速路”重述为“深度聚合算子”。 |
| 02 | [Motivation](sections/02-motivation.md) | 理解为什么固定单位权重会导致不可选择访问、信息不可恢复和输出幅度增长。 |
| 03 | [Attention Residuals](sections/03-attention-residuals.md) | 重点读 Full AttnRes、Block AttnRes、伪 query、RMSNorm key 和 $O(Ld)\to O(Nd)$。 |
| 04 | [Infrastructure Design](sections/04-infrastructure-design.md) | 看 cache-based pipeline communication、two-phase inference、prefill sequence sharding。 |
| 05 | [Experiments](sections/05-experiments.md) | 关注 scaling law、48B/3B Kimi Linear 集成、1.4T tokens、ablation 与 learned pattern。 |
| 06 | [Discussions / Related Work / Conclusion](sections/06-discussions-related-work-conclusion.md) | 把 AttnRes 放入 time-depth duality、structured matrix、残差连接谱系中理解。 |
| 07 | [Appendix](sections/07-appendix.md) | 查作者贡献、Full AttnRes two-phase I/O 推导。 |

## 关键数字

| 数字 | 含义 | 读法 |
| --- | --- | --- |
| $O(Ld)\to O(Nd)$ | Block AttnRes 将需要保存/通信的历史表示从层级降到 block 级 | 这是“能进大模型训练”的关键，不只是算法复杂度优化。 |
| $N\approx 8$ | 报告称约 8 个 block 能恢复 Full AttnRes 大部分收益 | 固定 block 数让系统开销稳定，但太粗会向 baseline 退化。 |
| $S=L/N$ | 每个 block 的层数 | 大模型实验中 54 layers、6 layers/block，得到 9 blocks，加 embedding 共 10 个 depth-wise sources。 |
| 48B total / 3B activated | Kimi Linear MoE 大模型配置 | AttnRes 不是只在小模型上验证，而是接入稀疏激活大模型。 |
| 1.4T tokens | 训练规模 | 包括 1T WSD pre-training 与约 400B high-quality mid-training。 |
| 8192 / 4096 / 32K | Scaling law context / 主训练 context / 延长后 context | 主训练用 4096，mid-training 后继续扩到 32K。 |
| 1.25x compute advantage | Block AttnRes 在 scaling law 拟合上达到 baseline 需约 1.25 倍计算量的 loss | 这是报告最直接的训练效率表述。 |
| <4% / <2% | pipeline training overhead / inference latency overhead | 训练有流水线并行时端到端开销小于 4%，典型推理延迟开销小于 2%。 |
| 15GB → 1.9GB → <0.3GB | 128K prefill block representation memory | sequence sharding 到 $P$ 个 TP device，再叠加 16K chunked prefill。 |

## 数据流：输入 → 中间表示 → 输出

**Standard PreNorm residuals**

输入 token embedding $\boldsymbol h_1$ → 每层只接收上一层压缩状态 $\boldsymbol h_{l-1}$ → 更新 $\boldsymbol h_l=\boldsymbol h_{l-1}+f_{l-1}(\boldsymbol h_{l-1})$ → 展开后所有前层输出以固定单位权重累加 → 随深度输出幅度增长，单层相对贡献被稀释。

**Full AttnRes**

输入 embedding 与每个前层输出 $\boldsymbol v_i$ → 每层持有一个学习到的伪 query $\boldsymbol w_l$ → keys/values 取历史 layer outputs，key 先 RMSNorm → 对所有前序源做 softmax attention → 输出 $\boldsymbol h_l=\sum_i\alpha_{i\to l}\boldsymbol v_i$。它保留最完整的跨层选择能力，但在激活重算和 pipeline parallelism 下需要跨 stage 保留/传输 $O(Ld)$ 历史表示。

**Block AttnRes**

输入 embedding 作为 $b_0$ → 层被划分为 $N$ 个 blocks → block 内先用普通残差累加成 partial sum $b_n^i$ 与 block summary $b_n$ → 每层跨 block 注意力只访问 $[b_0,\dots,b_{n-1}]$，block 内后续层额外访问 $b_n^{i-1}$ → 输出层聚合所有 block summaries。这样保留“跨深度选择”，但把工程对象从每层输出变成 block summary。

**训练与推理系统路径**

训练时，pipeline stage 缓存已经收到的 block summaries，后续虚拟 stage 只传新增 block；推理时，Phase 1 批量算一个 block 内所有层对历史 block 的 attention，Phase 2 顺序处理 block 内 partial sum 并用 online softmax 合并两部分统计量。

## 优点与局限

**优点**

- 把 PreNorm 的固定单位累加问题精确定位为 depth-wise aggregation 问题，而不是只讨论 normalization placement。
- Full AttnRes 机制简单：每层只引入一个 $d$ 维伪 query 和一个 RMSNorm，能直接解释为深度维 softmax attention。
- Block AttnRes 给出实际可部署路径：$O(Ld)\to O(Nd)$、cache-based pipeline communication、two-phase inference、sequence-sharded prefill 都围绕同一个瓶颈展开。
- 实验覆盖 scaling law、ablation、48B/3B 大模型训练和下游评测，且分析了 hidden-state magnitude 与 gradient magnitude 的变化。

**局限**

- Full AttnRes 的最佳形式仍受当前跨 stage 通信与显存限制，报告实际主推的是近似的 Block AttnRes。
- Block size / block count 是关键超参；$S$ 太大时效果向标准残差退化，$S$ 太小时系统成本上升。
- learned pseudo-query 比 input-dependent query 更便宜，但 ablation 显示后者 loss 更低；报告选择的是工程折中。
- 大模型结果主要在 Kimi Linear / MoE / hybrid KDA-MLA 架构中验证，迁移到 dense Transformer、不同并行策略或不同训练配方仍需复验。

## Q&A

**Q1：AttnRes 和普通 attention 的区别是什么？**  
A：普通 attention 沿序列维选择 token；AttnRes 沿深度维选择 layer/block outputs。query 不是当前 hidden state 投影，而是每层一个学习到的 pseudo-query，因此能提前批处理部分计算。

**Q2：为什么说标准残差是问题源之一？**  
A：标准残差让每层输出都以权重 1 进入后续状态。深层只能看到混合后的 $\boldsymbol h_{l-1}$，不能重新选择早期层；PreNorm 又让 hidden-state magnitude 随深度增长，导致单层贡献被稀释。

**Q3：Block AttnRes 是否只是少存一些层输出？**  
A：不只是压缩存储。它定义了新的信息路径：block 内局部累加，block 间全局选择。系统上，block boundary 还成为 pipeline communication 和 two-phase inference 的调度单位。

**Q4：为什么 pseudo-query 要初始化为 0？**  
A：0 初始化让初始 attention weights 在 source layers 上近似均匀，等价于训练初期的 equal-weight average，避免一开始就对某些层过度偏置而造成训练波动。

**Q5：为什么 key 上要 RMSNorm？**  
A：如果 key/value 直接用层输出或 block summary，幅度大的层会主导 softmax。RMSNorm 把“相关性选择”和“表示幅度增长”分开，尤其对 block summary 更重要。

**Q6：论文最值得复现的最小实验是什么？**  
A：16-layer 或 Table 2 中中小规模 MoE 上比较 PreNorm baseline、Full AttnRes、Block AttnRes，并复现 Table 4 的 input-dependent query、input-independent mixing、sigmoid、w/o RMSNorm ablation。

## 📊 Citation Landscape

> Semantic Scholar detail 接口本次返回 429 限流；`references.json` 已获取 70 条参考文献，`recommendations.json` 已获取 100 条推荐论文。因此这里不编造 citation count / TLDR，只使用可验证的 references 和 recommendations。

这篇报告的引用谱系可以按“残差稳定性 → 跨层访问 → 深度/序列对偶 → 系统化落地”理解：

- **残差与 normalization 基线：** ResNet、PreNorm/PostNorm、DeepNorm、LayerScale、ReZero、Highway、SiameseNorm 等工作定义了“如何稳定训练深网络”的问题背景。
- **多状态或跨层连接：** DenseNet/DenseFormer、Hyper-Connections/mHC、DDL、MRLA、MUDDFormer、LAuReL、Value Residual Learning 等试图缓解单一 residual stream 的压缩瓶颈；AttnRes 的差异点是 softmax-normalized、input-dependent、跨层或跨 block 选择。
- **序列模型类比：** Transformer self-attention、linear attention、GLA、DeltaNet、TTT/Fast Weight Programmers 提供“从 recurrence 到 attention”的时间维类比；报告把这个迁移到深度维。
- **系统依赖：** GPipe、Megatron interleaved pipeline、online softmax、tensor parallel prefill 等工程技术是 Block AttnRes 能被放进大模型训练/推理路径的前提。
- **实证锚点：** 最强实验证据来自 Kimi Linear 48B total / 3B activated、1.4T tokens 训练；因此它更像一篇“机制 + 大模型系统验证”的 technical report，而不是单纯结构小改的 ablation paper。

### 高被引参考锚点

| 参考工作 | 作用 |
|----------|------|
| Deep Residual Learning for Image Recognition | 残差连接的原始训练稳定性动机。 |
| Attention is All You Need | 把 recurrence 替换成 attention 的序列维类比来源。 |
| Densely Connected Convolutional Networks | 跨层表示复用的早期强基线。 |
| Neural Machine Translation by Jointly Learning to Align and Translate | attention 机制在序列建模中的早期来源。 |
| LLaMA / MMLU / Scaling Laws / MATH | 大模型训练、评测和 scaling law 的基础参照。 |

### Semantic Scholar 推荐方向

- **Mixture-of-Depths Attention / Recurrent Transformer**: 与“有效深度”和动态层访问直接相关。
- **Transformers with Selective Access to Early Representations**: 和 AttnRes 的 early-layer selective retrieval 问题最接近。
- **Learning When to Attend / Conditional Memory Access**: 连接长上下文 memory access 与按需访问。
- **Residual Stream Duality in Modern Transformer Architectures**: 与本文 time-depth duality 的理论方向相邻。
- **UniPrefill / Lighthouse Attention / Mixture of Chapters**: 更偏系统与长上下文 prefill/learned memory，可作为 Block AttnRes 工程路线的横向比较。
