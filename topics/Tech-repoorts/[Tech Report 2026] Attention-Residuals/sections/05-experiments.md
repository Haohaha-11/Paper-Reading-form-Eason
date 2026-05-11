[← 返回 README](../README.md)

# 05. Experiments

## 原文保留

Architecture Details. Our architecture is identical to Kimi Linear [69], a Mixture-of-Experts (MoE) Transformer following the Moonlight [28] / DeepSeek-V3 [9] design, which interleaves Kimi Delta Attention (KDA) and Multi-Head Latent Attention (MLA) layers in a 3:1 ratio, each followed by an MoE feed-forward layer. The only modification is the addition of AttnRes to the residual connections; all other components (model depth, hidden dimensions, expert routing, and MLP structure) remain unchanged. AttnRes introduces only one RMSNorm and one pseudo-query vector $\boldsymbol w_l\in\mathbb R^d$ per layer, amounting to a negligible fraction of the total parameter count. Crucially, all pseudo-query vectors must be initialized to zero. This ensures that the initial attention weights $\alpha_{i\to l}$ are uniform across source layers, which reduces AttnRes to an equal-weight average at the start of training and prevents training volatility, as we validated empirically.

> 💡 **实验控制**: 除 residual connection 外，其它 Kimi Linear 组件不变。这让收益更容易归因到 AttnRes，而不是 MoE 路由、注意力层比例或 MLP 配置变化。

> 💡 **初始化细节**: pseudo-query 全 0 初始化是训练稳定性的关键。softmax 初始接近均匀，相当于先从平均 residual aggregation 起步，再逐渐学会选择。

## 5.1 Scaling Laws

We sweep five model sizes (Table 2) and train three variants per size: a PreNorm baseline, Full AttnRes, and Block AttnRes with $\approx8$ blocks. They are trained with an 8192-token context window and a cosine learning rate schedule. Within each scaling law size group, all variants share identical hyperparameters selected under the baseline to ensure fair comparison; this setup intentionally favors the baseline and thus makes the comparison conservative. Following standard practice, we fit power-law curves of the form $\mathcal L=A\times C^{-\alpha}$ [22, 15], where $\mathcal L$ is validation loss and $C$ is compute measured in PFLOP/s-days.

Scaling Behavior. Fig. 4 presents the fitted scaling curves. The Baseline follows $\mathcal L=1.891\times C^{-0.057}$, while Block AttnRes fits $\mathcal L=1.870\times C^{-0.058}$, and Full AttnRes fits $\mathcal L=1.865\times C^{-0.057}$. All three variants exhibit a similar slope, but AttnRes consistently achieves lower loss across the entire compute range.

![Scaling law curves for Attention Residuals](../images/31dc3e3bdeadab2b184fd452acb6854d4e99f92828016b319b260a66245424ea.jpg)

Figure 4: Scaling law curves for Attention Residuals. Both Full and Block AttnRes consistently outperform the baseline across all scales. Block AttnRes closely tracks Full AttnRes, recovering most of the gain at the largest scale.

Based on the fitted curves, at 5.6 PFLOP/s-days, Block AttnRes reaches 1.692 versus the Baseline’s 1.714, equivalent to a $1.25\times$ compute advantage. The gap between Full and Block AttnRes narrows with scale, shrinking to just 0.001 at the largest size. Full AttnRes outperforms mHC, while Block AttnRes matches it at lower memory I/O per layer: $5.5d$ versus $34d$ for mHC with $m=4$ streams.

### Table 2 速读

| Activated params | Tokens | Baseline | Block AttnRes | Full AttnRes | mHC-lite |
| --- | --- | --- | --- | --- | --- |
| 194M | 38.7B | 1.931 | 1.909 | 1.899 | 1.906 |
| 241M | 45.4B | 1.895 | 1.875 | 1.874 | 1.869 |
| 296M | 62.1B | 1.829 | 1.809 | 1.804 | 1.807 |
| 436M | 87.9B | 1.766 | 1.746 | 1.737 | 1.747 |
| 528M | 119.0B | 1.719 | 1.693 | 1.692 | 1.694 |

> 💡 **scaling law 解读**: 三条曲线 slope 接近，说明 AttnRes 主要带来近似平移的 loss 改善，而不是改变 scaling exponent。对工程决策而言，这等价于在同等 compute 下稳定拿到更低 loss。

## 5.2 Main Results

Training recipe. The largest models we study are based on the full Kimi Linear 48B configuration: 27 Transformer blocks (54 layers) with 8 out of 256 routed experts plus 1 shared expert, yielding 48B total and 3B activated parameters. This model applies Block AttnRes with 6 layers per block, producing 9 blocks plus the token embedding for a total of 10 depth-wise sources.

We follow the same data and training recipe as the Kimi Linear 1.4T-token runs [69]: all models are pre-trained with a 4096-token context window, the Muon optimizer [28], and a WSD (Warmup-Stable-Decay) learning rate schedule [16], with a global batch size of 8M tokens. Training of the final model proceeds in two stages: (i) a WSD pre-training phase on 1T tokens, followed by (ii) a mid-training phase on approximately 400B high-quality tokens, following the annealing recipe of Moonlight [28].

After mid-training, we continue training with progressively longer sequence length of 32K tokens. Since our architecture uses hybrid KDA/MLA attention [69], where MLA operates without positional encodings (NoPE) [61], context extension requires no modifications such as YaRN [37] or attention temperature rescaling.

![Training dynamics of Baseline and Block AttnRes](../images/0eb19ef2bed04b434d606496dfd725082d2099b8a13c2b59da42bfb76a2eea39.jpg)

Figure 5: Training dynamics of Baseline and Block AttnRes. (a) Validation loss during training. (b) Each transformer block’s output magnitude at the end of training. (c) Each transformer block’s gradient magnitude.

Training dynamics. We compare the training dynamics of our final Baseline and Block AttnRes models over 1T tokens in Fig. 5.

• Validation loss: AttnRes achieves consistently lower validation loss throughout training, with the gap widening during the decay phase and resulting in a notably lower final loss.

• Output magnitude: The Baseline suffers from the PreNorm dilution problem [60, 27]: as hidden-state magnitudes grow monotonically with depth, deeper layers are compelled to learn increasingly large outputs from fixed-scale normalized inputs to remain influential. Block AttnRes confines this growth within each block, as selective aggregation at block boundaries resets the accumulation, yielding a bounded periodic pattern.

• Gradient magnitude: With all residual weights fixed to 1, the Baseline provides no means of regulating gradient flow across depth, leading to disproportionately large gradients in the earliest layers. The learnable softmax weights in Block AttnRes introduce competition among sources for probability mass, resulting in a substantially more uniform gradient distribution.

### Table 3 速读

| Area | Benchmark | Baseline | AttnRes | Delta |
| --- | --- | --- | --- | --- |
| General | MMLU | 73.5 | 74.6 | +1.1 |
| General | MMLU-Pro | 52.2 | 52.2 | 0.0 |
| General | GPQA-Diamond | 36.9 | 44.4 | +7.5 |
| General | BBH | 76.3 | 78.0 | +1.7 |
| General | ARC-Challenge | 64.6 | 65.7 | +1.1 |
| General | HellaSwag | 83.2 | 83.4 | +0.2 |
| General | TriviaQA | 69.9 | 71.8 | +1.9 |
| Math & Code | GSM8K | 81.7 | 82.4 | +0.7 |
| Math & Code | MGSM | 64.9 | 66.1 | +1.2 |
| Math & Code | Math | 53.5 | 57.1 | +3.6 |
| Math & Code | CMath | 84.7 | 85.1 | +0.4 |
| Math & Code | HumanEval | 59.1 | 62.2 | +3.1 |
| Math & Code | MBPP | 72.0 | 73.9 | +1.9 |
| Chinese | CMMLU | 82.0 | 82.9 | +0.9 |
| Chinese | C-Eval | 79.6 | 82.5 | +2.9 |

Downstream performance. Following the evaluation protocol of Kimi Linear [69], we assess both models across language understanding and reasoning, code and math, and Chinese language understanding. Block AttnRes matches or outperforms the baseline on all benchmarks. The improvements are particularly pronounced on multi-step reasoning tasks such as GPQA-Diamond $(+7.5)$ and Math $(+3.6)$, as well as code generation such as HumanEval $(+3.1)$, while knowledge-oriented benchmarks such as MMLU $(+1.1)$ and TriviaQA $(+1.9)$ also show solid gains. This pattern is consistent with the hypothesis that improved depth-wise information flow benefits compositional tasks, where later layers can selectively retrieve and build upon earlier representations.

> 💡 **大模型结果读法**: 最值得看的是 GPQA、Math、HumanEval 这些多步组合任务的提升。论文的解释是：后层需要重新组合早期/中期表征时，depth-wise retrieval 更有价值。

## 5.3 Ablation Study

We conduct ablation studies on the 16-head model from Table 2 to validate key design choices in AttnRes. All models share identical hyperparameters and compute budget.

Comparison with prior methods. We compare AttnRes against the PreNorm baseline (loss 1.766) and two representative methods that generalize residual connections. DenseFormer [36] grants each layer access to all previous outputs but combines them with fixed, input-independent scalar coefficients; it shows no gain over the baseline (1.767), highlighting the importance of input-dependent weighting. mHC [59] introduces input dependence through $m$ parallel streams with learned mixing matrices, improving to 1.747. AttnRes takes this further with explicit content-dependent selection via softmax attention: Full AttnRes achieves 1.737 and Block AttnRes 1.746, outperforming both methods with only a single query vector per layer.

Cross-layer access. We compare three granularities of cross-layer access. Full AttnRes follows directly from the time-depth duality (§3), applying attention over all previous layers, and achieves the lowest loss (1.737). A simple way to reduce its memory cost is sliding-window aggregation (SWA), which retains only the most recent $W=8$ layer outputs plus the token embedding; it improves over baseline (1.764) but falls well short of both Full and Block AttnRes, suggesting that selectively accessing distant layers matters more than attending to many nearby ones.

Block AttnRes offers a better trade-off: with block size $S=4$ it reaches 1.746 while keeping memory overhead constant per layer. Fig. 6 sweeps $S$ across the full spectrum from $S=1$ (i.e. Full AttnRes) to increasingly coarse groupings. Loss degrades gracefully as $S$ grows, with $S=2,4,8$ all landing near 1.746 while larger blocks move toward baseline. In practice, we fix the number of blocks to $\approx8$ for infrastructure efficiency (§4).

![Effect of block size on validation loss](../images/dcd6e8772b010f09566be8b24dd1eea253c31c487879856f7afe3e1dddd5de17.jpg)

Figure 6: Effect of block size on validation loss (16-layer model).

![Architecture sweep under fixed compute](../images/590c9db6b08b2f35c3aba7c8618fd2f190c6f55e4b9058f5a0a4ae059fe64fb2.jpg)

Figure 7: Architecture sweep under fixed compute. Each cell reports validation loss for a $(d_{\mathrm{model}}/L_b, H/L_b)$ configuration; the star marks the optimum.

Component design. We further ablate individual components of the attention mechanism:

• Input-dependent query. A natural extension is to make the query input-dependent by projecting it from the current hidden state. This further lowers loss to 1.731, but introduces a $d\times d$ projection per layer and requires sequential memory access during decoding, so we default to the learned query.

• Input-independent mixing. We removed the query and key and replaced them with learnable, input-independent scalars to weigh previous layers, which hurts performance (1.749 vs. 1.737).

• softmax vs. sigmoid. Replacing softmax with sigmoid degrades performance (1.741). We attribute this to softmax’s competitive normalization, which forces sharper selection among sources.

• Multihead attention. We test per-head depth aggregation $H=16$ on Block AttnRes, allowing different channel groups to attend to different source layers. This hurts performance (1.752 vs. 1.746), indicating that the optimal depth-wise mixture is largely uniform across channels: when a layer’s output is relevant, it is relevant as a whole.

• RMSNorm on keys. Removing RMSNorm degrades both Full AttnRes (1.743) and Block AttnRes (1.750). For Full AttnRes, it prevents individual layers with naturally larger outputs from dominating the softmax. This becomes even more critical for Block AttnRes, as block-level representations accumulate over more layers and can develop large magnitude differences; RMSNorm prevents these from biasing the attention weights.

> 💡 **ablation 主线**: DenseFormer 无收益说明“能看见所有层”不够；input-independent mixing 变差说明静态权重不够；sigmoid 变差说明 softmax 的概率竞争有用；w/o RMSNorm 变差说明需要控制 source magnitude。

## 5.4 Analysis

## 5.4.1 Optimal Architecture

To understand how AttnRes reshapes optimal architectural scaling, we perform a controlled capacity reallocation study under a fixed compute and parameter budget. Our central question is whether AttnRes alters the preferred depth-width-attention trade-off, and in particular, given its potential strength on the depth dimension, whether it favors deeper models compared to conventional Transformer design heuristics. We fix total training compute (approximately $6.5\times10^{19}$ FLOPs) and active parameters (approximately $2.3\times10^8$), ensuring that any performance variation arises purely from architectural reallocation rather than overall capacity differences. Under this constrained budget, we enumerate 25 configurations on a $5\times5$ grid over $d_{\mathrm{model}}/L_b$ and $H/L_b$, where $L_b=L/2$ is the number of Transformer blocks and $H$ the number of attention heads.

Both heatmaps exhibit a shared pattern: loss decreases with growing $d_{\mathrm{model}}/L_b$ and shrinking $H/L_b$, and both methods reach their optima at $H/L_b\approx0.3$. Despite this shared trend, AttnRes achieves a lower loss than the baseline in each of the 25 configurations, by 0.019-0.063. The most apparent difference lies in the location of the optimum: the baseline achieves its lowest loss at $d_{\mathrm{model}}/L_b\approx60$ (1.847), whereas AttnRes shifts it to $d_{\mathrm{model}}/L_b\approx45$ (1.802). Under a fixed parameter budget, a lower $d_{\mathrm{model}}/L_b$ corresponds to a deeper, narrower network, suggesting that AttnRes can exploit additional depth more effectively. This preference for depth does not directly translate to a deployment recommendation, as deeper models generally incur higher inference latency due to their sequential computation [39].

> 💡 **架构含义**: AttnRes 让更深、更窄的配置相对更优，符合“它改善深度信息流”的机制假设。但更深模型会增加串行推理延迟，所以这是诊断结论，不是无条件部署建议。

## 5.4.2 Analyzing Learned AttnRes Patterns

![Depth-wise attention weight distributions](../images/1a1a6db1c8b7fbd25e80c21efe466864cb4d505cb97c15841f7d9c3cf838830f.jpg)

Figure 8: Depth-wise attention weight distributions for a 16-head model with full (top) and block (bottom) Attention Residuals, averaged over tokens. Diagonal dominance indicates locality remains the primary information pathway, while persistent weights on source 0 (embedding) and occasional off-diagonal concentrations reveal learned skip connections.

We visualize the learned weights $\alpha_{i\to l}$ in Fig. 8 for the 16-head model with both full and block $(N=8)$ AttnRes. Each heatmap shows how the $l$-th attention or MLP layer allocates its attention over previous sources, with pre-attention and pre-MLP layers shown separately. We highlight three key observations:

• Preserved locality. Each layer attends most strongly to its immediate predecessor, yet selective off-diagonal concentrations emerge, indicating learned skip connections beyond the standard residual path.

• Layer specialization. The embedding $h_1$ retains non-trivial weight throughout, especially in pre-attention layers. Pre-MLP inputs show sharper diagonal reliance on recent representations, while pre-attention inputs maintain broader receptive fields, consistent with attention routing information across layers and MLPs operating locally.

• Block AttnRes preserves structure. Diagonal dominance, embedding persistence, and layer specialization all transfer from the full to the block variant, suggesting that block-wise compression acts as implicit regularization while preserving the essential information pathways.

> 💡 **权重图怎么读**: 对角线强说明模型没有抛弃局部深度路径；embedding 长期有权重说明早期 token 表示会作为深层 source；离对角热点说明 AttnRes 学到了跨层 skip retrieval。
