[← 返回 README](../README.md)

# 4. Experiments

## 📌 预览
本节验证方法是否成立，重点看主结果、消融、效率和视觉/病例案例。

---

We first conducted extensive evaluations on twelve diverse benchmarks against state-of-the-art competitors to validate the superiority of our method. Then, we provide comprehensive analysis through multi-faceted ablation studies and in-depth investigations to elucidate the underlying mechanisms of our approach.

> 💡 **实验路线**: 实验分三层验证：主 benchmark 上是否比显式 CoT、工具/RL 和 latent reasoning baseline 强；是否保留低 AR steps/低延迟；消融是否说明 SCF-VR、RDS、curriculum 和超参选择确实有贡献。

# 4.1. Experimental Setup

Evaluation Benchmark. We evaluate our method on three tasks across twelve benchmarks: (1) Mathematics Reasoning (MathVista [37], MathVision [56], MM-Math [50]); (2) Vision-centric Reasoning (Hallusion-Bench [14], MMVP [53], Seed-Bench-2-Plus [28], HR-Bench [59], GQA [1]); (3) Multimodal Composition Reasoning (MMStar [4], BLINK [11], ScienceQA [38], $\mathbf { M } ^ { 3 } \mathbf { C o T }$ [5]). Additional descriptive details can be found in the Appendix.

> 💡 **任务覆盖**: 12 个 benchmark 覆盖数学、视觉中心推理和组合推理。对本文最关键的是 vision-centric 组，因为 SCF-VR 直接声称改善细粒度视觉 grounding；数学/text-heavy 任务更多检验 latent reasoning 是否损害语言推理。

Baselines. We evaluate our method against a comprehensive set of state-of-the-art baselines, which can be categorized into four paradigms: (1) Zero-shot VLMs (GPT-4o [42], LLaVA-OneVision [29], InternVL3.5-8B [60], Qwen2.5-VL-7B [2]), (2) Explicit CoT-based methods (SCAFFOLD [27], ICoT [13], Multimodal-CoT [77], CCoT [40], Chain-of-Focus [75]), (3) Visual Enhanced methods, including tool-augmented reasoning (Deep-Eyes [80]) and RL-enhanced VLM reasoning (Vision-R1 [25], PAPO [63], VL-Rethinker [55]), and (4) Multimodal Latent Reasoning approaches (Laser [62], LVR [31], Monet [58], DMRL [34]).

> 💡 **baseline 设计**: Baseline 覆盖四类：zero-shot VLM、显式 CoT、tool-use/RL、latent reasoning。最公平的比较不是只看 GPT-4o 等闭源强模型，而是看同等训练配置下的 IVT-LR、Laser、DMLR 等 latent reasoning 近邻。

Table 1. Comparison of various multimodal reasoning baselines across three benchmarks. We selected three datasets featuring detailed reasoning chains for training and evaluate on their test split, respectively. Three metrics are reported: Accuracy $( \% )$ , Average number of Autoregressive Steps (# AR steps), and Average Generation Time (AVG. Time). Evaluations were conducted on the $\mathbf { M } ^ { 3 } \mathbf { C o T } .$ , ScienceQA, and GQA benchmarks using Qwen2-VL-7B and Chameleon-7B. † denotes the reimplementation for the methods with the same configuration with ours. No-CoT notes that directly predicts answers without generating intermediate steps.

> 💡 **Table 1 入口**: Table 1 同时报 Accuracy、平均 AR steps 和平均时间，正好检验“快且准”。它只用带细粒度 CoT 标注的 M3CoT、ScienceQA、GQA，并在 Qwen2-VL-7B 与 Chameleon-7B 两个 backbone 上测。

![Table 1.](../images/c2425b83fd4b69ff979a033e07013cacc98bb7c123ec9373070f20e848edb57f.jpg)
*Table 1.: Table 1. Comparison of various multimodal reasoning baselines across three benchmarks. We selected three datasets featuring detailed reasoning chains for training and evaluate on their test split, respectively. Three metrics are reported: Accuracy $( \% )$ , Average number of Autoregressive Steps (# AR steps), and Average Generation Time (AVG. Time). Evaluations were conducted on the $\mathbf { M } ^ { 3 } \mathbf { C o T } .$ , ScienceQA, and GQA benchmarks using Qwen2-VL-7B and Chameleon-7B. † denotes the reimplementation for the methods with the same configuration with ours. No-CoT notes that directly predicts answers without generating intermediate steps.*

> 💡 **Table 1 批读**: Qwen2-VL-7B 上 Ours 达到 M3CoT 73.0、ScienceQA 95.9、GQA 67.4，超过 IVT-LR 的 69.8/92.8/65.8；AR steps 约 7-9，比显式 CoT 的几十到上百步少一个数量级。Chameleon 上也从 IVT-LR 的 40.8/63.2/38.1 提到 43.4/65.7/39.4，说明收益不是只绑定 Qwen。

Table 2. Performance comparison of our method against baselines across four paradigms: Zero-Shot VLMs, Explicit CoT, Tool-Use & RL, and Latent Reasoning. Benchmarks are categorized into three domains: Visual Perception (MMVP, Seed-Bench-2-Plus, HallusionBench, HR-Bench), Compositional Reasoning (BLINK, MMStar), and Mathematical Reasoning (MathVista, MathVision, MM-Math). Among latent reasoning approaches, the best results are highlighted in bold, and the second-best are underlined. Our method is built upon the Qwen2.5-VL-7B backbone.

> 💡 **Table 2 入口**: Table 2 是更宽的泛化表，按视觉感知、组合推理、数学推理分组。作者特别标注 latent reasoning 内部最佳/次佳，读的时候要区分“全体模型排名”和“同类 latent 方法排名”。

![Table 2.](../images/e3011ec643adc429c1eeeedd56278424912c34a1bd0eb442b08de670e2b36ff9.jpg)
*Table 2.: Table 2. Performance comparison of our method against baselines across four paradigms: Zero-Shot VLMs, Explicit CoT, Tool-Use & RL, and Latent Reasoning. Benchmarks are categorized into three domains: Visual Perception (MMVP, Seed-Bench-2-Plus, HallusionBench, HR-Bench), Compositional Reasoning (BLINK, MMStar), and Mathematical Reasoning (MathVista, MathVision, MM-Math). Among latent reasoning approaches, the best results are highlighted in bold, and the second-best are underlined. Our method is built upon the Qwen2.5-VL-7B backbone.*

> 💡 **Table 2 批读**: Ours 用 30K 数据，在 MMVP 上 76.67，明显高于 Laser 72.00 和多数 tool/RL baseline；HallusionBench 68.63、HRBench 74.21 也支撑视觉 grounding claim。但 SeedBench-2-Plus 66.86 不如 VL-Rethinker/DeepEyes 等，说明 text-rich 或特定域数据不足时 SCF-VR 不是万能增益。

Training Dataset. To facilitate reproducibility, we first detail the training data configuration. During the supervised fine-tuning (SFT) stage, we curate a subset of approximately 30K samples from OneThinker [10], selected for its diverse distribution of reasoning chain lengths. The statistical distributions of CoT length, reasoning steps, and topic categories within this subset are illustrated in Fig. 3.

Table 3. We perform comprehensive ablation studies to analyze the contribution of each design choice, including hyper-parameter configurations and architectural components. Experiments are conducted on three CoT benchmarks with multiple backbone models. To save the training overhead, all models are trained for 16 epochs with a balanced sampled subset.

> 💡 **训练数据批读**: 训练只采 OneThinker 约 30K，而不是几百 K 级别数据。Figure 3 的 CoT 长度和步骤分布说明课程学习的数据基础：需要足够多可分段 reasoning chains，才能逐步把 step 压成 latent。

![Table 3.](../images/89c7a2b8c38840034ea7f26304a42a0717095ec93d79e1bd87b9581f2ae85613.jpg)
*Table 3.: Table 3. We perform comprehensive ablation studies to analyze the contribution of each design choice, including hyper-parameter configurations and architectural components. Experiments are conducted on three CoT benchmarks with multiple backbone models. To save the training overhead, all models are trained for 16 epochs with a balanced sampled subset.*

> 💡 **Table 3 批读**: 这是模块消融主表。四个 backbone 上 +RDS 都涨，+RDS&SCF-VR 继续涨；例如 Qwen2-VL-2B 的 M3CoT 从 44.2 到 47.6 再到 51.2，说明视觉重放和深度调度是叠加收益。GQA 增益较小，符合其相对简单、视觉组合压力较低的解释。

Backbone Models. To comprehensively evaluate the effectiveness and scalability of our approach, we instantiate our method across a diverse set of backbone architectures, including Qwen2-VL-2B/7B [57], Qwen2.5-VL-3B/7B [2], and Chameleon-7B [52].

> 💡 **Backbone 覆盖**: 选 Qwen2、Qwen2.5、Chameleon 是为了说明机制可插到不同 VLM 架构。这里仍需注意：都属于 2B-7B 级别开源 backbone，尚未说明 30B/70B 或医学专用 VLM 上是否同样有效。

Implementation Details. All frameworks employ eager attention mode to enable explicit access to internal attention maps. We set the number of latent reasoning tokens to $T _ { r } = 4$ , with $\alpha = 3 2$ salient regions injected per refinement iteration. To balance computational efficiency and reasoning accuracy, we cap the maximum refinement depth at $D = 1$ For the cosine annealing schedule, the region injection count decays from $\alpha _ { s } = 6 4$ to $\alpha _ { e } = 1 6$ . Models are trained for 16 epochs by default.

> 💡 **实现细节**: eager attention mode 是为了拿到内部 attention map，SCF-VR 对框架实现有侵入性。默认 $T_r=4$、$\alpha=32$、$D=1$，说明作者把 latent 步数和额外深度都控制得很小，优先守住延迟。

We employ DeepSpeed ZeRO-2 optimization without CPU offloading, with a per-GPU batch size of 8. Training utilizes the Adam optimizer with a learning rate of $4 \times 1 0 ^ { - 5 }$ and $\beta _ { 1 } = 0 . 9$ . For the proposed self-distillation loss, we set the weighting coefficient to $\lambda = 1 . 0$ for 2B/3B-scale models and $\lambda = 0 . 2$ for 7B-scale models, respectively. Input images are resized to different resolutions according to training dataset, please refer to Appendix for detailed preprocessing protocols. All experiments are conducted on 16 NVIDIA H20 GPUs (96GB VRAM each).

> 💡 **训练成本**: 16 张 H20、ZeRO-2、batch 8、16 epochs，训练规模不算轻。$\lambda$ 对 2B/3B 与 7B 设置不同，提示视觉自蒸馏强度需要随模型容量调参，不能无脑迁移。

# 4.2. Overall Quantitative Results

Reasoning Accuracy. As summarized in Table 1, we train and evaluate our method on three datasets that provide finegrained CoT annotations. With the same training protocols, our approach achieves the best performance across both model backbones and all three datasets. Compared to traditional explicit CoT methods, our method yields clear-cut improvements, with an average accuracy gain of nearly $30 \%$ built upon the Qwen-2-VL architecture. Our approach further surpasses the second-best latent-based reasoning baseline (IVT-LR) by an average of $+ 2 . 6 3 \%$ across all benchmarks.

> 💡 **主结果解读**: 这里作者强调在相同训练协议下优于显式 CoT 和 latent baseline。关键数字是对 IVT-LR 平均 +2.63%，同时自回归步数维持在约 7-9，这比单纯提高准确率更能说明 latent 压缩路线有效。

As shown in Table 2, we further evaluate our method against widely-used mainstream benchmarks that diagnose multi-faceted reasoning capabilities. On most benchmarks, our approach achieves consistent improvements. Particularly on vision-centric benchmarks, our method attains remarkable gains in overall scores compared to previous state-of-the-art methods. Notably, we observe the most substantial improvement on MMVP $( + 4 . 6 7 \% )$ , which employs CLIP-blind patterns and emphasizes the requirement for finegrained visual discrimination. We attribute these gains to our designed RCF-VR. By maintaining a set of visual latents with fine-grained spatially-coherent constraints, our method effectively mitigates hallucination while capturing fine-grained visual details. Conversely, our approach exhibits modest performance on text-intensive benchmarks, notably SeedBench-2-Plus. This gap likely arises from the scarcity of domain-specific training data, underscoring the potential benefits of targeted fine-tuning for such tasks. Remarkably, our method achieves competitive performance against computationally intensive alternatives, including explicit Chainof-Thought (CoT) and tool-augmented frameworks. Despite operating purely within the latent space—without relying on external knowledge retrieval or reinforcement learningbased optimization—our framework effectively captures rich semantic representations through token-wise depth scaling.

> 💡 **跨基准解读**: 视觉中心 benchmark 提升更明显，作者归因于 RCF/SCF-VR 对细粒度视觉 latent 的保持；SeedBench-2-Plus 较弱则暴露 text-intensive 数据覆盖不足。这个 pattern 支撑“视觉增强”而不是纯语言能力提升。

Reasoning Efficiency. Beyond predictive accuracy, a critical advantage of our approach lies in its substantially improved inference efficiency, which we quantify through two similar metrics: autoregressive generation steps and wallclock inference latency. (1) Autoregressive Steps. Across all evaluated backbones, our method achieves at least a $1 0 \times$ reduction in autoregressive generation steps relative to conventional baselines. This efficiency gain arises from performing compact reasoning directly in the latent space, thereby obviating the need for verbose, explicitly generated textual rationales characteristic of standard Chain-of-Thought (CoT) approaches. (2) Inference Latency. Built upon the Qwen backbone, our method attains an average wall-clock inference time of approximately 0.9s, comparable to IVT-IR while delivering $3 { - } 6 \times$ faster rationale generation than explicit CoT-based competitors. Similar acceleration patterns are consistently observed on the Chameleon architecture. Although the No-CoT baseline achieves the lowest latency (0.35s), this efficiency comes at the cost of sophisticated multi-step reasoning capabilities. In contrast, our approach operates near this efficiency frontier: it delivers state-of-theart accuracy while incurring only marginal latency overhead relative to the fastest yet reasoning-limited No-CoT baseline. This favorable trade-off underscores a key advantage of our framework—enabling rapid inference without compromising the nuanced, multi-step understanding essential for complex multimodal reasoning tasks.

> 💡 **效率解读**: No-CoT 最快但缺少多步推理；显式 CoT 准确率有时高但步骤多、耗时长。本文落在中间偏优位置：用少量 latent steps 获得接近甚至超过显式 CoT 的准确率，延迟接近 latent baseline。

![Figure 3.](../images/0a72f7e5aa9663a9860ef13162bdd7a7b046d1612e4ba9f1dbf13ff06147bed6.jpg)
*Figure 3.: Figure 3. Details of sampled training distribution. Panel (a) depicts the sample distribution across different CoT lengths. The distribution exhibits a boundary at 125 tokens, with sample counts gradually decreasing toward both shorter and longer extremes. Panel (b) illustrates the distribution of reasoning steps in the original data. Specifically, we segment reasoning steps using $\backslash \mathrm { n } \backslash \mathrm { n }$ as delimiters; during training, we retain up to 4 delimiters to evenly partition sequence length, while samples containing fewer than 4 delimiters preserve their original segmentation. Panel (c) summarizes the topic-wise distribution of samples in the dataset.*

> 💡 **Figure 3 批读**: 图中 CoT 长度在 125 tokens 附近有边界，reasoning steps 被按 `\n\n` 分段并最多保留 4 段。这解释了为什么默认 $T_r=4$：latent token 数量与训练数据中的步骤划分对应。

# 4.3. Ablation Analysis

Effect of Designed SCF-VR and RDS. We introduce several model variants to validate the effectiveness of our proposed modules across four representative MLLM backbones. As summarized in Table 3, consistent performance improvements are observed across all architectures. Notably, when instantiated on Qwen2-VL-2B, our method achieves a substantial improvement of $+ 7 . 0 \%$ on $\mathbf { M } ^ { 3 } \mathbf { C o T } .$ . Furthermore, we observe that performance gains are less pronounced on the GQA benchmark. We hypothesize that this trend arises from the relatively straightforward nature of GQA, where both visual inputs and queries demand less complex reasoning. Consequently, baseline models can adequately address the task requirements, rendering the contributions of our designed modules less impactful in such scenarios.

> 💡 **模块消融解读**: SCF-VR 和 RDS 在四个 backbone 上都稳定提升，尤其小模型提升更大。这可能说明小模型更缺视觉 grounding 和深度预算，大模型已有一定能力但仍可从 replay/refinement 中获益。

Effect of Curriculum Training. Curriculum training is critical to our method design. As evidenced in Table 5(a), incorporating the progressive training strategy yields a $0 . 9 \%$ performance improvement over the baseline. This paradigm facilitates the gradual integration of latent representations into the optimization process, enabling each latent to progressively ground relevant contextual cues.

> 💡 **课程学习消融**: Base + Curriculum 在 M3CoT/ScienceQA/GQA 上分别 +0.9/+1.1/+0.6。提升不算巨大，但说明逐步把 CoT 封装进 latent 比直接训练 latent 更稳。

Table 4. Performance comparison with different mapping networks of $\mathcal { F } _ { \mathrm { S R } }$ on four kinds of backbones.

> 💡 **Table 4 入口**: Table 4 比较 TokenSR 的 mapping network。它回答的是：从 coarse visual latent 到 high-fidelity reference 的映射，用线性投影、普通卷积还是 point-wise conv 更好。

![Table 4.](../images/715b8d8d825c51e023d372949319573abb4eacc3b3c3777a5262b5b7dfed30a0.jpg)
*Table 4.: Table 4. Performance comparison with different mapping networks of $\mathcal { F } _ { \mathrm { S R } }$ on four kinds of backbones.*

> 💡 **Table 4 批读**: Linear projection 在四个 backbone 上都最好，例如 Qwen2.5-VL-7B 为 73.9，高于 vanilla conv 69.2 和 point-wise conv 69.8。这说明 feature-level TokenSR 不需要复杂卷积结构，过强局部混合反而可能破坏 token 语义空间。

![Table extracted](../images/6e4565056766a2dedb18fb6d0f11df91db18c96e94a3a00619f8c7794b0a4af7.jpg)
*Table extracted: Table extracted by MinerU. Dataset M3CoT ScienceQA GQA (a) Curriculum Training Base 62.0 75.4 51.3 Base + Curr. 62.9 (+0.9) 76.5 (+1.1) 51.9 (+0.6) (b) Token Selection in Router Tα(s) α = 32 62.9 76.5 51.9 α*

> 💡 **Table 5 快读**: 固定 $\alpha=32$ 是最佳或并列最佳；32-patch 直接保留优于 16-patch 和 Soft-Mix；Keep Old Position 优于 Rearrange Position。总体结论是：保留足够多的原始空间 token 和原位置编码，比过度压缩/重排更稳。

Table 5. Ablation Experiments. We provide ablation analysis of key parameters and experimental settings on three benchmarks: $\mathbf { M } ^ { 3 } \mathbf { C o T }$ , ScienceQA, and GQA. All the variants adopt Qwen2.5- VL-3B as the base model. Base refers the model that adopts RDS module and SCF-VR module.

> 💡 **Table 5 说明**: Base 在这里指已经采用 RDS 和 SCF-VR 的配置，因此 Table 5 是超参与训练策略消融，不是模块有无消融。不要和 Table 3 的 Base 混读。

Effect of Retention Parameter $\alpha$ . For the parameter $\alpha$ in $T _ { \alpha } ( \cdot )$ , we compare different retention strategies. As illustrated in Table 5, we evaluate fixed top- $\alpha$ settings with $\alpha \in \{ 1 6 , 3 2 , 6 4 \}$ and a cosine-annealed token retention schedule (CTR), which gradually reduces the token retention ratio layer by layer following a cosine schedule. Formally, for a model with $L$ layers, the retention ratio for the $l$ -th layer is,

> 💡 **Retention 超参**: $\alpha$ 控制 RDS 每层保留多少 token 深推理。结果显示 $\alpha=16$ 太少，$\alpha=64$ 没有明显更好；cosine-annealed retention 因后层 token 变少而限制交互，表现略差。

![Equation 15](../images/6fa9284c7d65f4a346b211a407c0f4d915f88dfe474d5fbc4f1c88fc2dcda458.jpg)
*Equation 15: Equation extracted by MinerU.*

> 💡 **Equation 15 批读**: Eq. 15 给出随层数变化的 cosine retention schedule，从 $\alpha_s$ 逐步到 $\alpha_e$。实验反而显示递减保留不如固定 32，说明随着 visual latents 和 reasoning latents 追加，后层仍需要足够上下文范围。

where $\alpha _ { s }$ and $\alpha _ { e }$ are two endpoints that enables flexible control over computational cost. As can be seen panel (b) in Table 5, results show that fixed top- $\alpha$ with $\alpha = 3 2$ achieves optimal performance, while cosine-annealed schedules yield consistently inferior results. We attribute this result to that: as visual tokens and latent hidden states are gradually appended to the token sequence, larger contextual scopes becomes essential for deeper token interaction. Consequently, strategies that progressively decrease the token retention ratio during depth scaling unavoidably limit these critical interactions, thereby hindering the model’s capacity to develop comprehensive contextual understanding.

> 💡 **Retention 结论**: 对这类 latent reasoning，省计算不能只看 token 数；如果后层上下文被过度削减，复杂关系无法建立，RDS 的“深想”会变成窄视野重复计算。

Visual Latents Formation. The Soft-Mix strategy aggregates 32 selected patches into a single visual latent token via weighted summation, guided by reweighted attention scores. In contrast, the 32-patch and 16-patch strategies directly utilize the top 32 and 16 patches with the highest attention scores as visual latents at each reasoning step, respectively. As summarized in Table 5, the 32-patch strategy yields superior performance. Notably, Soft-Mix achieves performance comparable to 16-patch while utilizing only a single compact representation. We attribute this performance to the expanded search scope provided by the 32-patch strategy, which offers the router a richer pool of visual tokens, thereby effectively raising the upper bound of depth scaling capabilities.

> 💡 **Visual latents 形成**: 32-patch 最好说明保留多个独立视觉 token 比把它们 Soft-Mix 成单 token 更有利于 router 选择和后续上下文交互。对医学小病灶场景，这意味着过早池化可能损失关键局部差异。

# 4.4. In-depth Analysis

Impact of $\lambda$ . As shown in Figure 4, model performance exhibits high sensitivity to variations in the hyperparameter $\lambda$ Notably, larger models (e.g., 7B) achieve peak performance at $\lambda = 1 . 0$ , whereas the 2B model attains optimal results at a substantially lower value of $\lambda = 0 . 2$ . We attribute this divergence to distinct capacity across model scales: smaller models are primarily bottlenecked by visual perception capabilities, thus requiring larger degree of visual play to establish effective visual grounding for downstream reasoning. In contrast, larger models possess sufficient grounding capability, and they benefit from a more balanced allocation between visual and linguistic signals, avoiding over-reliance on visual features that could otherwise suppress the development of complex reasoning chains.

> 💡 **$\lambda$ 敏感性**: $\lambda$ 控制视觉自蒸馏强度，模型容量不同最优值不同。正文这段与 README 里我标注的风险有关：OCR/描述中 2B/7B 最优表述可能不一致，复现时应核对原 PDF 图和代码默认配置。

![Figure 4.](../images/8bc468c6b8db202aee40c04dbed7c2a072ce6cbc6bb55b763fa24499a79e3162.jpg)
*Figure 4.: Figure 4. Sensitivity analysis of $\lambda$ .*

> 💡 **Figure 4 批读**: 这张图看 $\lambda$ 调参曲线，而不是最终 SOTA。它提醒 SCF-VR 是辅助监督，强度过高/过低都会影响视觉-语言平衡。

![Figure 5.](../images/ab83fbaa0a06aa185bba87008a5f3f6f3031d51cba3eae55fbed871f23fbae9e.jpg)
*Figure 5.: Figure 5. (a) and (b) illustrate the performance with different number of event prototypes and different ratio of filtered event prototypes, respectively.*

> 💡 **Figure 5 批读**: 图注 OCR 似乎串入了“event prototypes”，但正文说明它用于 latent behavior analysis：看 baseline 与本文 latent token 在 embedding space 的分布差异。应以原图内容为准，避免过度解读错误图注。

Impact of different window size $W$ . As shown in Table 6, $W = 3$ achieves optimal performance. We reckon that smaller window sizes may fail to comprehensively cover salient visual regions, whereas larger window sizes, while providing broader contextual information, tend to introduce excessive visual noise.

> 💡 **窗口大小消融**: $W=3$ 最好，说明空间连续 crop 需要覆盖足够上下文，但窗口太大又会带入噪声。这个结果支持 SCF-VR 不是简单“裁越大越好”。

Latent Behavior Analysis. As depicted in Figure 5, we visualize the embedding space distribution of multimodal features for both the baseline and our method. The latent tokens learned by our approach form several distinct clusters, situated centrally within the text embedding manifold. Notably, compared to the baseline, our latents exhibit closer proximity to visual embeddings. This observation suggests that our latent tokens encapsulate richer reasoning semantics while facilitating deeper integration of visual information.

> 💡 **latent 行为分析**: 作者观察到本文 latents 更靠近 visual embeddings，同时仍位于 text manifold 中央，说明它们既吸收视觉信息，又能被语言 decoder 使用。这是 latent space 级别的定性证据。

Visualization of Crop Region. As illustrated in Figure 6, we visualize the attention-guided crops across successive reasoning steps. Taking the third case as a representative example, which queries the purpose of the train cart, the model initially localizes the region corresponding to the description “shape like a house” during the first reasoning step. In subsequent steps, the attention focus progressively shifts toward the “entrance door”. Remarkably, without any explicit fine-grained supervision, our curriculum training paradigm enables latent tokens to progressively capture logical dependencies and establish robust visual-semantic alignments.

> 💡 **crop 可视化**: Figure 6 展示不同 latent step 的 crop 位置随推理目标变化，例如从“像房子的车厢”转到“入口门”。这类图是 SCF-VR 最直观的证据：latent steps 不只是抽象向量，也对应逐步更新的视觉关注。

![Figure 6](../images/cb499b9d4de8c85ca2d7c109c1fa8e84c8dc0f9f42aaf7a07fe939df53450a39.jpg)
*Figure 6: Q: What can be inferred about the train cars in the picture?*

> 💡 **Figure 6 批读**: 第一组案例把轨道、偏离轨道的车厢、锈迹分别作为推理证据。可视化的价值在于检查 attention-guided crop 是否与文字 rationale 对齐；局限是它仍不是因果证明。

Q: What can be inferred about the train cars in the picture?

Step 1: First, we see the train tracks in the picture that lead to nowhere, implying that there is an obstruction or a diversion in the tracks. Step 2: Secondly, we see the train cars diverted off of the tracks, which suggests that they are no longer able to continue along their usual track route. Step 3: Yellow rust on the train suggests it hasn't been cleaned.

![Figure 7](../images/408b4638ef317fe5cf5004a7f3b2f8aab1061ea28c974c3f0a0ebf9f14828ea7.jpg)
*Figure 7: Step 1: The red train cart shaped like a house suggests that it is providing living quarters. Step 2: The red door on the train suggests that it is an entrance door, further indicating that it is used for living quarters.*

> 💡 **Figure 7 批读**: 这一组强调“像房子的车厢”和“红门”两个局部线索如何支持 living quarters 结论。它说明 replay 区域可以随 reasoning step 从整体形状转向功能部件。

Step 1: The red train cart shaped like a house suggests that it is providing living quarters. Step 2: The red door on the train suggests that it is an entrance door, further indicating that it is used for living quarters.

![Figure 6.](../images/4038b7befdb9fef2b2dbd32d335f42048f280d33ddd6b80d58155d5d5c0b70f8.jpg)
*Figure 6.: Step 1: The red and white sign that reads "do not enter" suggests that the street behind the sign is one way only. Step 2: The street warning sign indicates that cars are not allowed in but only exiting. Therefore, the most likely reason for the presence of the warning sign is that it is a oneway street. Figure 6. Visualization of cropped region in each latent reasoning step. Dotted line denotes the correspondence between reasoning rationales and cropped region.*

> 💡 **Figure 6 批读**: 路牌案例展示文字/符号类小区域定位，这正是视觉中心 benchmark 中容易被语言先验误判的类型。医学影像中类似的是小病灶、测量标记或局部异常纹理。

Step 1: The red and white sign that reads "do not enter" suggests that the street behind the sign is one way only. Step 2: The street warning sign indicates that cars are not allowed in but only exiting. Therefore, the most likely reason for the presence of the warning sign is that it is a oneway street.   
Figure 6. Visualization of cropped region in each latent reasoning step. Dotted line denotes the correspondence between reasoning rationales and cropped region.

> 💡 **可视化结论**: 作者用 dotted line 连接 rationale 与 crop 区域，说明 latent step 能形成多步 visual grounding。但还需要遮挡实验或反事实区域替换来证明这些 crop 对答案是必要的。

This empirically demonstrates the model’s capacity to jointly perform multi-step visual grounding and generate coherent reasoning chains.

> 💡 **整体解读**: 这句话是定性 claim：模型能同时做多步视觉 grounding 和连贯推理。批读时要把它和 Table 2 的 MMVP/HallusionBench 提升一起看，定量+定性才比较完整。

Impact of Position Encoding. Regarding positional encoding during depth scaling, we evaluate two strategies. The Keep Old Position variant preserves the original relative positional embeddings of the selected tokens. In contrast, the Rearrange Position variant sequentially re-indexes the selected tokens from 1 to $K$ according to the reduced sequence length. As demonstrated in Table 3, preserving the original positions yields superior performance. Although re-indexing may enhance local contextual modeling among salient tokens, it risks introducing positional inconsistencies with embeddings from preceding reasoning steps and discards inherent structural priors, ultimately leading to suboptimal results.

> 💡 **位置编码消融**: Keep Old Position 更好，说明 selected tokens 即便进入压缩/重算路径，也应该保留原图文序列中的结构先验。重排位置会让当前 step 与之前 latent/replay 的位置关系断裂。

Table 6. Performance comparison of different sub-grid window sizes on the $\mathbf { M } ^ { 3 } \mathbf { C o T }$ benchmark. We set $\lambda = 1 . 0$ for the 2B/3B models and $\lambda = 0 . 2$ for the 7B model. Here, $W$ denotes the subgrid window size, with $W = 1 0$ is the window size of extracted visual feature.

> 💡 **Table 6 入口**: Table 6 只比较 sub-grid window size $W$，grid size 固定为 $10\times10$。它对应 SCF-VR 的空间一致 crop，不是 RDS 的 token budget。

![Table 6.](../images/71b1eacfe52a1cc7a0a8d6e9b4c7c5159344b1c0778558223e838c5237da9d06.jpg)
*Table 6.: Table 6. Performance comparison of different sub-grid window sizes on the $\mathbf { M } ^ { 3 } \mathbf { C o T }$ benchmark. We set $\lambda = 1 . 0$ for the 2B/3B models and $\lambda = 0 . 2$ for the 7B model. Here, $W$ denotes the subgrid window size, with $W = 1 0$ is the window size of extracted visual feature.*

> 💡 **Table 6 批读**: 四个 backbone 都是 $W=3$ 最好或最高：Qwen2-VL-2B 51.2，Qwen2.5-VL-7B 73.9。$W=2$ 覆盖不足，$W=5$ 噪声变多，说明空间一致约束需要适中窗口。

---

## 🔖 Section 总结

### 核心洞察
1. 主结果同时看准确率、AR steps 和延迟；本文在少量 latent steps 下超过 IVT-LR 等 latent baseline。
2. Table 2 显示视觉中心 benchmark 提升更明显，支持 SCF-VR 的视觉 grounding 目标，但 text-rich benchmark 仍有短板。
3. 消融表明 RDS、SCF-VR、curriculum、$\alpha=32$、32-patch、保留原位置、$W=3$ 都是较关键的设计选择。
4. 可视化能说明 replay 区域与推理步骤有对应关系，但还不足以证明因果性。

### 可追问点
- Table 2 中 SeedBench-2-Plus 不占优，是否说明 text-rich 场景需要额外数据或不同 replay 策略？
- SCF-VR 的 attention crop 能否通过遮挡验证因果贡献？
- 在医学图像上，benchmark accuracy 是否需要替换为病灶定位、报告一致性和专家评估？
