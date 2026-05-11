[← 返回 README](../README.md)

## 📌 预览
Drop-in enhancement、from-scratch 对比、消融和效率证据。

---

# 4 Experiments

In this section, we conduct a series of experiments to empirically validate the effectiveness of our In-Place TTT framework. Specifically, we aim to answer the following research questions:

• Q1: How effectively can In-Place TTT enhance pre-trained LLMs in a “drop-in" manner? • Q2: When trained from scratch, how does In-Place TTT compare against prior TTT approaches? • Q3: What are the effects of key design choices in our In-Place TTT framework?

> 💡 **实验问题映射**: Q1 验证 drop-in enhancement，Q2 验证从头预训练时的架构价值，Q3 验证 state/chunk/objective 三个核心设计的必要性。

Using language modeling tasks of various scales as a practical proxy, we answer each question with carefully designed experiments in the following sub-sections. Due to space limits, we present detailed descriptions of experimental settings in Appendix C.

Table 1 Evaluation results on RULER [28]. Average accuracy ( $\%$ ) is reported, with the best results in bold.   

> 💡 **128k/256k 数字**: Qwen3-4B continual training 后，In-Place TTT 在 128k RULER 从 74.8 提到 77.0，在 256k extrapolation 从 41.7 到 43.9；提升不大但方向稳定，关键是没有改 backbone 架构。

<table><tr><td></td><td colspan="6">In-Domain Evaluation</td><td>Extrapolation</td></tr><tr><td>Model</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td><td>64k</td><td>128k</td><td>256k</td></tr><tr><td>Mistral-7B [31]</td><td>93.6</td><td>91.2</td><td>87.2</td><td>75.4</td><td>49.0</td><td>13.8</td><td></td></tr><tr><td>GLM3-6B [23]</td><td>87.8</td><td>83.4</td><td>78.6</td><td>69.9</td><td>56.0</td><td>42.0</td><td></td></tr><tr><td>Phi3-medium-14B [1]</td><td>93.3</td><td>93.2</td><td>91.1</td><td>86.8</td><td>78.6</td><td>46.1</td><td></td></tr><tr><td>Llama3-8B [41]</td><td>92.8</td><td>90.3</td><td>85.7</td><td>79.9</td><td>76.3</td><td>69.5</td><td></td></tr><tr><td>Qwen3-4B (Instruct) [57]</td><td>95.1</td><td>93.6</td><td>91.0</td><td>87.8</td><td>77.8</td><td>66.0</td><td></td></tr><tr><td>Baseline</td><td>96.6</td><td>94.1</td><td>92.1</td><td>88.7</td><td>74.3</td><td>74.8</td><td>41.7</td></tr><tr><td>In-Place TTT</td><td>96.1</td><td>95.6</td><td>92.7</td><td>89.3</td><td>78.7</td><td>77.0</td><td>43.9</td></tr></table>

# 4.1 In-Place TTT as a Drop-in Enhancement for Pre-trained LLMs

To validate In-Place TTT as a “drop-in” enhancement for existing, pre-trained LLMs, we start with the competitive open-sourced Qwen3-4B-Base model. Its original context window is $3 2 \mathrm { k }$ , thereby we can simulate the long-horizon, evolving tasks requiring Test-Time Training capabilities by language modeling tasks of varying context lengths. In particular, we compare the performance of (1) Qwen3-4B-Base (Baseline); (2) Qwen3-4B-Base $^ { + }$ In-Place TTT (In-Place TTT). Both models undergo the exact same continual training curriculum, ensuring a fair comparison where our In-Place TTT is the only variable.

Table 2 Extension of In-Place TTT to LLaMA-3.1-8B and Qwen3-14B-Base on the RULER benchmark. We report the average accuracy ( $\%$ ) with the best results in bold.   

<table><tr><td>Base Model</td><td>Method</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td><td>64k</td><td>64k+YaRN</td></tr><tr><td rowspan="2">LLaMA-3.1-8B</td><td>Baseline</td><td>93.9</td><td>92.1</td><td>92.5</td><td>91.1</td><td>81.6</td><td></td></tr><tr><td>In-Place TTT</td><td>94.4</td><td>93.0</td><td>93.3</td><td>91.7</td><td>83.7</td><td>−</td></tr><tr><td rowspan="2">Qwen3-14B</td><td>Baseline</td><td>96.8</td><td>95.0</td><td>94.6</td><td>90.7</td><td>67.9</td><td>81.3</td></tr><tr><td>In-Place TTT</td><td>97.2</td><td>95.7</td><td>95.2</td><td>91.2</td><td>70.6</td><td>82.5</td></tr></table>

Training and Evaluation. The continual training curriculum is divided into two stages: an initial phase of ${ \sim } 2 0 \mathrm { B }$ tokens with $3 2 \mathrm { k }$ context length, followed by a second phase of ${ \sim } 1 5 \mathrm { B }$ tokens with 128k context length. The detailed descriptions of training dataset can be found in Appendix C.1. To effectively manage these long sequences, we adapt the model’s Rotary Position Embeddings using YaRN [42]. We evaluate the long-context performance of both models on the RULER benchmark [28] using the popular OpenCompass framework [13], with context lengths ranging from 4k to 256k. The 256k setting specifically measures the models’ ability to extrapolate beyond the 128k context length limit. Detailed descriptions of training details can be found in Appendix C.2.

> 💡 **训练成本定位**: drop-in 实验不是零训练，而是两阶段 continual pretraining：约 20B tokens @32k + 15B tokens @128k。论文的 claim 应理解为“无需从零预训练”，不是“无需训练”。

Results and Discussion. The results, summarized in table 1, demonstrate that In-Place TTT significantly boosts the long-context proficiency of the pre-trained model. In particular, a clear trend can be easily seen from the results: while both models are competitive at short contexts, Qwen3-4B-Base enhanced by our In-Place TTT establishes a consistent and widening advantage as the sequence length increases. It achieves substantial gains at the 64k and 128k context lengths. Crucially, this advantage is maintained when extrapolating to a 256k context, demonstrating superior generalization. These findings confirm that In-Place TTT can be seamlessly integrated into a pre-trained LLM to boost its long-context proficiency. The model’s strong performance at and beyond the context length validates our method as a practical and powerful tool for extending the capabilities of existing LLMs.

Extension to More Models. To verify the generality of our approach, we further apply In-Place TTT as a drop-in enhancement to two additional models: LLaMA-3.1-8B [24] and Qwen3-14B-Base [57], following the same continual training protocol. As shown in Table 2, In-Place TTT consistently improves the RULER scores across all context lengths on both models. The gains are particularly pronounced at longer contexts, e.g., $+ 2 . 1$ at 64k for LLaMA-3.1-8B and $+ 2 . 7$ at 64k for Qwen3-14B-Base. Moreover, the improvement is maintained when combined with YaRN [42] for position extrapolation, confirming that our method is orthogonal to RoPE extension techniques. These results, spanning different model families and scales (4B–14B), reinforce that In-Place TTT is a broadly applicable drop-in enhancement for pre-trained LLMs.

> 💡 **泛化性证据**: LLaMA-3.1-8B 和 Qwen3-14B 也有增益，说明机制不只绑定 Qwen3-4B；与 YaRN 叠加仍有效，说明它和 RoPE extension 是正交维度。

# 4.2 Pre-training from Scratch: A Comparative Analysis

Having demonstrated In-Place TTT’s effectiveness as a “drop-in” module, we further evaluate its performance and scalability when integrated into models pre-trained from scratch. Our analysis proceeds in two stages: we first establish its language modeling capabilities at the 500M and 1.5B scales, and then assess its scalability and impact on a larger 4B model.

Experimental Setup. Firstly, we benchmark our In-Place TTT against prior TTT-related approaches and efficient attention methods based on TogetherAI [50] at 500M and 1.5B parameter scales. Various competitive baselines are compared: (1) standard Transformer with sliding window attention (SWA) [6, 10] (2) Gated Linear Attention (GLA) [60]; (3) DeltaNet [43, 59, 62] (4) Large Chunk Test-Time Training (LaCT) [67]. For a fair comparison, both In-Place TTT and LaCT are built upon an SWA backbone. All models are trained on sequences with a $3 2 \mathrm { k }$ context length.

> 💡 **from-scratch 对比**: 这里把 In-Place TTT 放到 SWA backbone 上，与 GLA、DeltaNet、LaCT 等比较，用来证明它不是只靠继承强预训练模型。

![](../images/4e593de9a10f5190f12121957d8d880b7300180ae90d67eb906787df7fa4b97d.jpg)  
Figure 2 Sliding Window Perplexity at varying context lengths on the Pile dataset for 500M (left) and 1.5B (right) parameter models. Our In-Place TTT consistently achieves lower perplexity than all competitive baselines.

Table 3 Evaluation results of 4B models on common sense reasoning and long-context evaluation benchmarks. Best performance is in bold. “SWA” is Sliding-Window Attention, “Full Attn.” is Full Attention, and “I.P. TTT” is our In-Place TTT.   

<table><tr><td></td><td colspan="5">Common Sense Reasoning</td><td colspan="3">Long-Context Evaluation</td></tr><tr><td>Model Architecture HellaSwag ARC-E ARC-C MMLU PIQA RULER-4k RULER-8k RULER-16k</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Full Attn. Baselines SWA</td><td>55.67</td><td>64.52</td><td>33.19</td><td>36.43</td><td>72.63</td><td>45.77</td><td>38.09</td><td>6.58</td></tr><tr><td></td><td>54.92</td><td>64.18</td><td>32.85</td><td>36.06</td><td>72.58</td><td>14.77</td><td>9.91</td></tr><tr><td rowspan="2">Full Attn. I.P. TTT SWA</td><td>55.85</td><td>64.98</td><td>32.34</td><td>37.42</td><td>73.29</td><td>49.98</td><td>43.82</td><td>19.99</td></tr><tr><td>55.24</td><td></td><td>64.60 33.70</td><td>36.48</td><td>72.03</td><td>28.33</td><td>26.80</td><td>7.57</td></tr></table>

Building on these results, we further scale up to 4B-parameter models to evaluate scalability of our In-Place TTT approach. In particular, we compare Transformers with Full Attention and Transformers with SWA against their counterparts enhanced by our In-Place TTT. These models are trained for 120B tokens with an 8k context length. Detailed descriptions of datasets, model configurations, and training procedures are available in sections C.1 to C.3.

Evaluation. For the 500M and 1.5B models, we evaluate their long-context utilization using Sliding Window Perplexity on a validation set comprised of Pile [20] and Proof-Pile-2 [40]. This metric measures perplexity on a fixed final block of tokens when extending the preceding context, where a decreasing perplexity trend indicates effective context usage. For the 4B models, we conduct a broader evaluation on a suite of downstream tasks, including common sense reasoning benchmarks (HellaSwag [66], ARC [12], MMLU [26, 27], PIQA [7]) and the long-context RULER benchmark [28].

Results and Discussion. In Figure 2, we plot the sliding window perplexity against context length for 500M/1.5B model. It can be easily seen that our In-Place TTT consistently achieves lower validation perplexity than all competitive baselines, with its performance steadily improving up to the full 32k context. This sustained improvement suggests its core mechanism successfully compresses and utilizes information from incoming context.

> 💡 **上下文利用证据**: Sliding Window Perplexity 随上下文长度增加持续下降，说明 fast weights 确实在压缩并利用更长上下文，而不是只在固定窗口内做局部修补。

Moreover, the results in Table 3 further show that 4B-parameter Transformers with both Full Attention and SWA are consistently improved across most common sense reasoning tasks. Furthermore, models with our In-Place TTT yield superior performance on the long-context evaluation, e.g., RULER-16k score is improved from 6.58 to 19.99 for the Transformer with Full Attention and RULER-8k score is boosted from 9.91 to 26.80 for the SWA model. These substantial gains, particularly across models of various scales, establish our In-Place TTT as a highly effective and scalable approach.

> 💡 **4B 规模证据**: Full Attention + In-Place TTT 的 RULER-16k 从 6.58 到 19.99，SWA + In-Place TTT 的 RULER-8k 从 9.91 到 26.80；这是从头训练场景下更强的架构级证据。

![](../images/a52b02d544159209a5665ed805a9eec1bb7580ed5d88657212c281f43bd4f9b6.jpg)  
Figure 3 Ablation studies on the key design choices of the In-Place TTT framework, evaluated on the RULER benchmark with a 1.7B parameter model. The plots illustrate the impact of: (a) State size, showing that performance improves as the state size scales; (b) Chunk size, demonstrating a performance trade-off where intermediate sizes (e.g., 512, 1024) are optimal; and (c) The LM-Aligned Value objective, confirming that both the convolution (w Conv) and the projection (w Proj) are crucial.

![](../images/24cd2789966d4823f2dd3cf8fe0bf04c117e32fbb3c5c36ddf4b5224182cdd30.jpg)  
Figure 4 Efficiency analysis of In-Place TTT. Both prefill throughput (a, b) and peak memory (c, d) metrics are presented for 4B models with Sliding-Window Attention (SWA) and Full Attention at various context lengths. Our In-Place TTT introduces negligible overhead in practical scenarios.

# 4.3 Ablation Studies: On the Impact of Key Design Choices

Lastly, we conduct a series of ablation studies on RULER with a 1.7B-parameter model, providing deeper insights into our design choices. Detailed settings are presented in Appendix C.3 and C.2.

Impact of State Size. We first investigate how performance scales with the fast weights size, which can be controlled by varying the number of TTT-enabled layers. Figure 3 (a) shows a clear trend that the performance of our In-Place TTT consistently improves along with the state size scaling. This confirms that larger fast weights allow the model to more effectively adapt to contextual information, which further supports our repurposing approach leveraging the large amount of MLP states.

> 💡 **state size 消融**: TTT-enabled layers 越多，fast-weight state 越大，RULER 表现越好。这支持“复用大 MLP 参数作为高容量动态记忆”的核心假设。

Impact of Chunk Size. The chunk size $C$ in section 3.1 controls both the granularity of fast weights updating and parallelism, exposing a tradeoff between efficiency and performance. By varying the chunk size, Figure 3 (b) shows that both $C = 5 1 2$ and $C = 1 0 2 4$ competitively achieve better performance compared to other choices, while $C = 1 0 2 4$ has better efficiency.

> 💡 **chunk size 消融**: 最优区间在 512/1024，说明 per-token 或过小 chunk 不是必要条件；这和论文的大 chunk + context parallelism 效率主张闭环。

Impact of LM-Aligned Objective. Next, we delve deep into our tailored LM-Aligned objective in section 3.2, where the target is defined as $\hat { \mathbf { V } } = \operatorname { C o n v } 1 \mathrm { D } ( \mathbf { X } _ { 0 } ) \mathbf { W } _ { \mathrm { t a r g e t } }$ . In particular, the Conv1D operator is used to yield targets containing future token information, and the $\mathbf { W } _ { \mathrm { t a r g e t } }$ is a projection transformation. In Figure 3 (c), we comprehensively ablate combinations of these components. The result shows that both of them are necessary for performance guarantee, while Conv1D plays an essential role on long context and $\mathbf { W } _ { \mathrm { t a r g e t } }$ is crucial on short context. These results align with our theoretical analysis in section 3.3, strongly supporting our motivation to derive a tailored objective for language modeling.

> 💡 **objective 消融**: Conv1D 对长上下文更关键，$W_{target}$ 对短上下文更关键；两者都必要，和理论部分“未来 token 信息 + 可学习投影”对应。

Efficiency Impact of In-Place TTT. We further study the computational overhead introduced by our In-Place TTT. In figure 4, we compare both prefill throughput and memory consumptions of with and without our In-Place TTT. The results indeed verify the efficiency of our practical implementations.

> 💡 **效率证据**: Figure 4 把 prefill throughput 和 peak memory 放在一起看，支持“实际开销可忽略”的工程 claim；但细节依赖 H800、batch size 1 和实现质量。

---

## 🔖 Section 总结

> 💡 **Section 小结**: 实验链路覆盖 drop-in、from-scratch、ablation、efficiency；关键证据集中在 128k 长上下文增益、4B from-scratch RULER 提升，以及 512/1024 chunk 与 objective 组件消融。
