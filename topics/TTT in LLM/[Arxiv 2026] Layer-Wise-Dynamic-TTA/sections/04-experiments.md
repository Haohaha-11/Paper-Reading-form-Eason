[← 返回 README](../README.md)

# 4. Experiments

# 4.1. Implementation Details

We employ dynamic TTA and train SCALENET, a twolayer MLP with hidden size 128, using AdamW at learning rate $1 0 ^ { - 4 }$ . The base TTA learning rate (to be scaled) is set to $1 0 ^ { - 2 }$ . For each dataset–model pair, we train on roughly $3 0 \mathrm { k }$ samples (mostly without repeats) and evaluate on 300 test samples. We set the LoRA rank to $r = 4$ and LoRA $\alpha$ to 16. The maximum total number of TTA steps is $K _ { \operatorname* { m a x } } = 5$ . LoRA parameters are initialized with $B = 0$ and $A \sim 1 0 ^ { - 2 } \mathcal { N } ( 0 , I )$ , and are re-initialized randomly for each prompt. All LLM backbones use bfloat16.

> 💡 **实现批注**: 这里的 30k train samples per dataset-model pair 很关键：推理 episode 是无监督，但 SCALENET 的控制策略是按任务/模型训练出来的。它目前更像“给某个模型和任务族训练一个 TTA 控制器”，不是完全开箱即用的通用策略。

across several datasets. XSum (Narayan et al., 2018) is a summarization dataset of BBC news articles paired with single-sentence summaries. SQuAD (Rajpurkar et al., 2016) is a reading-comprehension benchmark where answers are spans from a provided Wikipedia passage. NQ-Open (Lee et al., 2019) is an open-domain QA setting that evaluates predicting short answers to real user queries. AdaptEval (Hu et al., 2025a) is a comprehensive benchmark for testtime learning that groups datasets into three categories: DomainBench (Geography, Agriculture, Medicine, Finance), InstructionBench (Alpaca-GPT4, Dolly, InstructionWild), and ReasoningBench (GSM8K, MetaMath, LogiQA), covering diverse tasks and domains.

> 💡 **数据集批注**: XSum/SQuAD 的答案信息大多在 prompt 内，prompt NLL 的自监督信号更可能对 answer 有帮助；NQ-Open/AdaptEval 更开放，常需外部知识或推理，所以 prompt-only adaptation 的天花板更低。

# 4.2. LLMs and Datasets

We experiment with two families of LLMs: Llama-3.2- 3B, Llama-3.2-3B-Instruct and Llama3.3-70B-Instruct; Qwen3-4B, Qwen3-4B-Instruct and Qwen3-32B. For each family, the “Instruct” variant is post-trained for instructionfollowing and multi-turn dialogue. Evaluation is conducted

> 💡 **模型批注**: 中等模型用于多数据集曲线，大模型只在 AdaptEval 上补充。这能说明方法可能扩到 32B/70B，但还不能证明大模型上所有任务都稳。

# 4.3. Main Results

Evaluation metric. Since we train with teacher-forced next-token negative log-likelihood (NLL), NLL is the most directly aligned evaluation metric. In practice, however, users often care more about generation-oriented metrics such as ROUGE-Lsum (Lin, 2004). Without dataset-level fine-tuning—which effectively teaches the LLM the desired answer distribution—reductions in NLL from unsupervised sample-specific TTA are not guaranteed to translate into consistent improvements in ROUGE-Lsum. This is justifiable because ROUGE-Lsum is based on lexical overlap, while natural language admits many valid paraphrases; the overlap score is therefore highly sensitive to surface form and stylistic choices, which are primarily learned through dataset-level supervision. Nevertheless, we report both NLL and ROUGE-Lsum.

> 💡 **指标批注**: NLL 与训练目标对齐，但用户关心生成质量。ROUGE-Lsum 的加入很必要，因为 prompt NLL 可能只是“loss hacking”；后面 XSum/SQuAD 的 ROUGE 提升支持有效性，而 NQ/AdaptEval 的不稳定则暴露边界。

Table 1. NLL results on AdaptEval for Llama3.3-70B-Instruct and Qwen3-32B under different TTA steps (lower is better).

<table><tr><td>LLM</td><td>Steps</td><td>Fixed</td><td>Step-wise</td><td>Layer-wise</td></tr><tr><td rowspan="5">Llama3.3 -70B -Instruct</td><td>No TTA</td><td>2.2114</td><td>2.2114</td><td>2.2114</td></tr><tr><td>1 step</td><td>2.1907</td><td>2.1665</td><td>1.6598</td></tr><tr><td>2 steps</td><td>5.8488</td><td>2.1399</td><td>1.6692</td></tr><tr><td>3 steps</td><td>9.6803</td><td>2.1850</td><td>1.6790</td></tr><tr><td>4 steps</td><td>11.5891 11.4970</td><td>2.1997 2.2657</td><td>1.6741 1.7048</td></tr><tr><td rowspan="6">Qwen3 -32B</td><td>5 steps No TTA</td><td>2.1805</td><td></td><td>2.1805</td></tr><tr><td>1 step</td><td>2.1809</td><td>2.1805 2.1303</td><td>2.1107</td></tr><tr><td>2 steps</td><td>2.1784</td><td>2.0892</td><td>1.9168</td></tr><tr><td>3 steps</td><td>2.1757</td><td>2.0842</td><td>1.8777</td></tr><tr><td>4 steps</td><td>2.1725</td><td>2.0829</td><td>1.8739</td></tr><tr><td>5 steps</td><td>2.1682</td><td>2.0925</td><td>1.8889</td></tr></table>

> 💡 **Table 1 批注**: MinerU 抽表有错位，但趋势足够清楚。Llama70B 上 fixed LR 1 step 只有小幅改善，2 step 后 NLL 直接崩到 5.8/9.7/11+；layer-wise 则稳定在约 1.66-1.70。Qwen32B 上 fixed 没崩但收益很小，layer-wise 到 4 steps 约 1.8739，说明动态控制不是只在小模型上有效。

Analysis. We first report results in terms of NLL per answer token on a set of moderate-size LLMs: Llama-3.2-3B, Llama-3.2-3B-Instruct, Qwen3-4B, and Qwen3-4B-Instruct. As an empirical baseline, we run na¨ıve TTA with a fixed learning rate of $5 \times 1 0 ^ { - 2 }$ applied uniformly across all layers and adaptation steps. For our dynamically controlled TTA, we use a base learning rate of $1 0 ^ { - 2 }$ and rescale it using the predicted multipliers. Because our protocol is samplespecific (memoryless), each prompt is permitted to contain the full task context; otherwise, the prompt-only objective can be ill-posed (see Appendix for detailed format).

> 💡 **baseline 批注**: 固定 LR baseline 是 $5 \times 10^{-2}$，dynamic base 是 $10^{-2}$ 加 scaler。这里要注意公平性质疑：如果 fixed LR 调得更保守，可能不崩但少步收益更小；step-wise ablation 的存在就是为了证明 learned schedule 仍不如 layer-wise。

To validate the effect of layer-wise control, we include a layer-agnostic/step-wise ablation that predicts a single multiplier per adaptation step and applies it uniformly across layers. This step-wise variant is indispensable: it can be viewed as a learned upper bound over common handcrafted learning-rate schedules (e.g., linear decay, cosine annealing (Loshchilov & Hutter, 2017), exponential decay (Li & Arora, 2019)), since any such schedule can be expressed as a particular choice of per-step scalar multipliers. Remember that our layer-wise hypernetwork predicts perstep, per-layer multipliers (separately for Q/V projections), strictly generalizing step-wise control by enabling layerspecific modulation. Finally, we consider an additional ablation that averages the layer-wise scaling across samples to remove prompt dependence.

> 💡 **消融批注**: 这一段把三种控制粒度分清了：fixed 是全局常数，step-wise 是可学习时间 schedule，layer-wise 是时间 + 层 + Q/V + prompt。sample-averaged layer-wise 又去掉 prompt dependence，用来检验动态控制是否真的随输入变化。

Table 2. ROUGE-Lsum results (higher is better). Due to computation budget, we cut the maximum generated new token number at 64 for XSum, 32 for SQuAD and NQ, and 256 for AdaptEval.

<table><tr><td>Pair</td><td>Steps</td><td>Fixed</td><td>Step-wise</td><td>Layer-wise</td></tr><tr><td>Llama3B</td><td>No TTA</td><td>0.1821</td><td>0.1821</td><td>0.1821</td></tr><tr><td>-Instruct</td><td>1 step</td><td>0.1800</td><td>0.1838</td><td>0.1964</td></tr><tr><td>on XSum</td><td>5 steps</td><td>0.1780</td><td>0.1904</td><td>0.2090</td></tr><tr><td>Qwen4B</td><td>No TTA</td><td>0.1700</td><td>0.1700</td><td>0.1700</td></tr><tr><td>-Instruct</td><td>1 step</td><td>0.1700</td><td>0.1872</td><td>0.2157</td></tr><tr><td>on XSum</td><td>5 steps</td><td>0.1773</td><td>0.1864</td><td>0.2247</td></tr><tr><td>Llama3B</td><td>No TTA</td><td>0.7080</td><td>0.7080</td><td>0.7080</td></tr><tr><td>-Instruct</td><td>1 step</td><td>0.7058</td><td>0.7133</td><td>0.7238</td></tr><tr><td>on SQuAD</td><td>5 steps</td><td>0.6532</td><td>0.7114</td><td>0.7353</td></tr><tr><td>Qwen4B</td><td>No TTA</td><td>0.7722</td><td>0.7722</td><td>0.7722</td></tr><tr><td>-Instruct</td><td>1 step</td><td>0.7657</td><td>0.5361</td><td>0.7958</td></tr><tr><td>on SQuAD</td><td>5 steps</td><td>0.7634</td><td>0.5402</td><td>0.8306</td></tr><tr><td>Llama3B</td><td>No TTA</td><td>0.2766</td><td>0.2766</td><td>0.2766</td></tr><tr><td>-Instruct</td><td>1 step</td><td>0.2722</td><td>0.2662</td><td>0.2398</td></tr><tr><td>on NQ-Open</td><td>5 steps</td><td>0.0288</td><td>0.2674</td><td>0.2507</td></tr><tr><td>Qwen4B</td><td>No TTA</td><td>0.1320</td><td>0.1320</td><td>0.1320</td></tr><tr><td>-Instruct</td><td>1 step</td><td>0.1424</td><td>0.0866</td><td>0.1513</td></tr><tr><td>on NQ-Open</td><td>5 steps</td><td>0.1495</td><td>0.1041</td><td>0.1706</td></tr><tr><td>Llama70B</td><td>No TTA</td><td>0.2327</td><td>0.2327</td><td>0.2327</td></tr><tr><td>-Instruct</td><td>1 step</td><td>0.2325</td><td>0.2068</td><td>0.2237</td></tr><tr><td>on AdaptEval</td><td>5 steps</td><td>0.0029</td><td>0.2223</td><td>0.2733</td></tr><tr><td>Qwen32B</td><td>No TTA</td><td>0.0987</td><td>0.0987</td><td>0.0987</td></tr><tr><td>on AdaptEval</td><td>1 step</td><td>0.0992</td><td>0.0982</td><td>0.0983</td></tr><tr><td></td><td>5 steps</td><td>0.1004</td><td>0.0990</td><td>0.1043</td></tr></table>

> 💡 **Table 2 批注**: ROUGE 上最强证据来自 XSum/SQuAD：Llama3B-Instruct XSum 0.1821 → 0.2090，Qwen4B-Instruct XSum 0.1700 → 0.2247；SQuAD 也分别到 0.7353 和 0.8306。NQ-Open 则分裂：Qwen4B 变好，Llama3B layer-wise 反而低于 No TTA，符合“open-domain prompt 不含足够答案信息”的解释。

As shown in Figure 3, the na¨ıve baseline can rarely achieve optimal performance: it typically yields small initial gains, but then quickly degrades as NLL rises sharply with further steps, indicating destructive drift. In contrast, our proposed method substantially improves stability and effectiveness. Both the original layer-wise scaling and layer-agnostic/step-wise variants consistently outperform the fixed-rate baseline. Introducing layer-wise control further reduces NLL, suggesting that different transformer blocks prefer different TTA update magnitudes. Across datasets, most improvements occur in the first adaptation step, with subsequent steps providing diminishing returns. In addition, crudely averaging scales across samples can reduce the performance gains, indicating that the dynamic scaling varies with prompt content.

> 💡 **Figure 3 批注**: 这段给了本文最重要的机制结论：第一步要大，后面要衰减；不同层要不同；不同 prompt 也要不同。fixed LR 的失败正好是把这些差异全抹掉。

Overall, substantial variation is observed across dataset– model pairs. In particular, Llama-3.2-3B on AdaptEval shows very limited TTA gains. Since AdaptEval is more diverse and challenging—often requiring deeper logical reasoning—and to test whether our proposed unsupervised dynamic TTA framework scales to larger LLMs, we further evaluate Llama3.3-70B-Instruct and Qwen3-32B on AdaptEval. As listed in Table 1, our dynamic TTA consistently outperforms the na¨ıve baseline and the layeragnostic/step-wise ablation on the 32B and 70B LLMs, with gains for Llama3.3-70B-Instruct even larger than for Llama-3.2-3B-Instruct on AdaptEval. This implies our method has the potential to scale well into large-size LLMs.

> 💡 **AdaptEval 批注**: AdaptEval 是 harder setting，因为包含 domain/instruction/reasoning 混合任务。大模型上 NLL 仍明显改善，说明控制器不是小模型过拟合现象；但 ROUGE 改善在 Qwen32B 上很小，说明 NLL 改善和最终文本质量仍可能脱钩。

![](../images/c47057a43414fb6c17a03a36bded7962288921ff51a4efe4f53589c696a87f4c.jpg)
Figure 4. SCALENET output heatmap averaged over 4 datasets and 4 moderate-size LLMs. Along horizontal axis, $q _ { k }$ and $v _ { k }$ denote query and value projection at step $k$ . From top to bottom, transformer layers are ordered from shallow (first) to deep (last).

> 💡 **Figure 4 批注**: heatmap 是 layer-wise control 的直观证据：Q/V 投影、相邻层、不同 step 的 multiplier 可差几个数量级。若某些层天然敏感，固定 LR 或 step-wise schedule 都会过度更新这些层。

We next report ROUGE-Lsum results. For brevity, we report ROUGE-Lsum only at total step $K \in \{ 0 , 1 , 5 \}$ for Llama-3.2-3B-Instruct and Qwen3-4B-Instruct on XSum, SQuAD, and NQ-Open, and for Llama-3.3-70B-Instruct and Qwen3- 32B on AdaptEval. Table 2 shows that our method can also improve ROUGE-Lsum, indicating that it is not simply a next-token NLL “hacking” trick, but can meaningfully improve generated text without requiring additional supervision or external material. The ROUGE-Lsum gains on XSum and SQuAD are clear and stable, while on NQ-Open and AdaptEval the changes are smaller and less stable. As a zero-shot setting, the former two datasets contain most of the answer information in the prompt itself, whereas the latter two are more open and often require harder reasoning, and are therefore more difficult to optimize using promptonly adaptation (Ma et al., 2022).

> 💡 **生成质量批注**: 这段是最诚实的边界说明：prompt-only TTA 擅长把 prompt 内已有信息更好地转成 answer，不擅长凭空补外部知识或复杂推理。XSum/SQuAD 清楚受益，NQ/AdaptEval 更像压力测试。

Visualization. Here we visualize the learned learning rate multipliers. Figure 4 shows that effective fine-grained dynamic learning-rate modulation has a rich structure, and that separating query/value projections matters. We do not observe a consistent monotonic trend along either the step dimension or the depth dimension (e.g., “query projection always larger than value projection,” or “early layers always smaller than late layers”). Instead, the preferred scales vary sharply across neighboring blocks: adjacent layers (or Q vs. V within the same layer) can differ by orders of magnitude even at the same TTA step. Notably, this heatmap is averaged over four datasets; the fact that high-contrast patterns remain after averaging suggests shared regularities rather than noise. For example, in Llama3.2-3B, several layers consistently favor much smaller update magnitudes, indicating that uniform learning-rate choices would overupdate sensitive blocks. Together with Figure 5, we observe that the average magnitude of SCALENET outputs peaks at the first adaptation step $k { = } 1 )$ ) and then decays rapidly in subsequent steps, offering a natural explanation for why the majority of performance gains arise from the initial update.

> 💡 **层级结构批注**: 没有简单单调规律很重要：如果只是“浅层小、深层大”或“Q 一直大于 V”，手写 schedule 也可能够用；但图中是非平滑的局部模式，支持用超网络而不是规则表。

![](../images/4f37c2d402b31b66545ff3bddc9dfbd1917c91868aedf14dd0c8bd800822767c.jpg)
Figure 5. SCALENET output percentage difference with $9 5 \%$ CI across schedules (baseline $\mathrm { K } { = } 5$ ) and all-schedules scaling magnitude mean. Both averaged over 4 dataset test samples and 4 moderate-size LLMs.

> 💡 **Figure 5 批注**: 平均 scaler 在第 1 步最大、后续快速衰减，解释了“多数收益来自第一步”。这也给部署启发：如果延迟敏感，可以重点优化 1-step dynamic TTA，未必需要完整 5 steps。

Besides, for a stable TTA process, the update at a given step index $k$ should be comparable across schedules with different total steps $K$ , since the per-step update rule is identical and only the horizon differs. As shown in Figure 5, taking $K { = } 5$ as the baseline, the average percentage differences (averaged over test samples) between scales from shorter schedules and the baseline at the same step index are relatively small. Given that the scales span several orders of magnitude, this indicates strong schedule consistency.

> 💡 **schedule consistency 批注**: 随机训练 $K$ 的目标是让同一 step index 的控制在不同总步数下可比。若这个性质不成立，部署时改 TTA step 数就会破坏控制器行为。

# 4.4. Limitations

Our current work prioritizes dataset-tailored, fine-grained, prompt-conditioned adaptation, which may limit transferability across substantially different task distributions. We believe this can be mitigated by training on larger corpus and using higher-capacity models beyond shallow MLP.

> 💡 **局限批注**: 作者承认 dataset-tailored，这正是当前最大风险。要把方法变成通用 LLM TTA 基础设施，需要证明 SCALENET 能跨任务、跨模型、跨 prompt format 泛化，而不是每个 benchmark 训练一个控制器。
