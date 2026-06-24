[← 返回 README](../README.md)

# 4. Experiments

## 一、Preview

Experiments 分为四个部分：(4.1) 三阶段训练配置（Vision-Language Alignment → Multimodal Pretraining → Instruction Tuning）；(4.2) 评估 Setup（tokenization, hyperparameters, benchmarks）；(4.3) 核心 Benchmark 结果（主表 vs 开源模型 + 三变体消融 + 层级增益精细分析）；(4.4) 自适应计算分析（norm_diff 早停 + KV-cache 优化 + 层级注入加速收敛）。

核心发现：(1) Recurrence 是中等到大幅性能提升的主要来源（r=1→r=32，avg +28.09）；(2) 层级视觉注入提供额外 modest gain（+1.10 avg），且在特定维度（Logic Reasoing +2.54%, Coarse Perception +3.04%）更显著；(3) 层级注入使收敛加速 25-40%，体现出明确的推理效率收益。

---

## 二、原始文本

### 4.1. Training Configuration

Following the established methodology of MLLMs like LLaVA-NeXT (Liu et al., 2024b), we decouple the training of HIVE into a three-stage pipeline.

> 💡 **Table 2** — Backbone LLM 对比关键数据: Huginn 3.5B 在 GSM8K-CoT (8-shot) 上仅 34.57，而 Phi-3-mini 3.8B 达 82.5，LLaMA2 7B 无 8-shot CoT 数据但 base 能力显然更强。Huginn 0.8T 预训练 token 远少于竞品 (Gemma3 4T, Phi-3 3.3T, LLaMA2 2T)。这个骨干语言能力的 gap 是 HIVE 实验性能解读的关键 context——HIVE 在 "弱骨干" 上依然取得了有竞争力的多模态结果。

> 💡 **Table 3** — 三阶段训练配置:
>
> | 参数 | Stage 1 | Stage 2 | Stage 3 |
> |------|---------|---------|---------|
> | **目的** | V-L 对齐 (仅训练 Projector + Patch Merger) | 图文对齐 + OCR + 语言能力维持 | 指令微调 |
> | **Learning Rate** | 1e-3 | 1e-5 | 1e-6 |
> | **Max Dynamic Patches** | 4 | 2 | 2 |
> | **Max Tokens** | 1536 | 2048 | 2048 |
>
> LR 递减反映了从"快速适配新模态"到"精细微调"的切换。Dynamic Patches 在 Stage 1 用 4（便于快速学习多尺度感知），后续减至 2（控制计算开销）。

To capture fine-grained visual details across varying aspect ratios, we employ InternViT-300M-448px-V2.5 as our visual backbone. Unlike static encoders that resize images to a fixed square, our model leverages a dynamic high-resolution strategy. In our implementation, we set the maximum number of image tiles to a relatively conservative value to align with our available computational budget. Details are shown in Table 3.

**Stage 1.** To pre-align the vision-language modality, we only train the projector as well as the patch mergers in this stage, using the LCS-558K dataset (Liu et al., 2023b).

**Stage 2.** To enhance image-text alignment, we use the subset of the EMOVA alignment dataset (Chen et al., 2025a). General visual-language pre-training is sourced from ShareGPT4V (Chen et al., 2024b), ALLaVA (Chen et al., 2024a) (English and translated Chinese), and ShareGPT-4o (Cui et al., 2024), while OCR-related capabilities are supported by SynthDog (Kim et al., 2022), MMC-Alignment (Liu et al., 2024a), K12 Printing, and the UReader Text Reading subset (Ye et al., 2023). We also integrate the text-only corpus from Magpie Pro (Xu et al., 2025d) into our multi-modal training pipeline to maintain strong linguistic proficiency.

**Stage 3.** Our Stage 3 training data comprises 3.4M samples from the EMOVA-SFT subset, together with a collection of high-quality open-source visual instruction datasets, including ShareGPT4V (Chen et al., 2024b), InternVL (Chen et al., 2024d), Meteor (Lee et al., 2024), Idefics-2 (Laurençon et al., 2024), Cambrian (Shengbang et al., 2024), and LLaVA-OneVision (Li et al., 2025a).

> 💡 **训练数据管线分析**:
> - **Stage 1 (V-L Alignment)**: 仅 LCS-558K，训练参数最少（Projector + Patch Merger）——这是标准的 warm start，让 vision features 学会"说 LLM 的语言"
> - **Stage 2 (多模态预训练)**: 数据多样性高——Caption (ShareGPT4V, ALLaVA, ShareGPT-4o) + OCR (SynthDog, MMC-Alignment, UReader) + 纯文本 (Magpie Pro)。纯文本的加入值得注意——防止多模态训练 damage 语言能力
> - **Stage 3 (指令微调)**: 3.4M SFT 样本，主要来自 EMOVA + 多个开源 VQA/Instruction 数据集，覆盖面广
> - **总计 6.5M 样本**——相对于同行是中等规模。考虑到 Huginn 骨干 0.8T 的预训练，多模态训练数据的质量和多样性尤为关键

### 4.2. Setup

**Tokenization.** To bridge the modalities, we introduce three dedicated tokens: `<|image start|>`, `<|image|>`, and `<|image end|>`. Specifically, the `<|image|>` token serves as a placeholder and is substituted by the visual features projected from the vision encoder into the language embedding space.

**Hyperparameters.** We train the model with a weight decay of 1e-3. We adhere to the optimization configuration of the original Huginn, employing the AdamW optimizer with β1 = 0.9 and β2 = 0.95. The learning rate is set with a cosine decay scheduler.

**Evaluation.** We evaluated the effectiveness of our approach across several challenging benchmarks via LMMs-Eval (Zhang et al., 2024), including MMStar (Chen et al., 2024c), MMBench (Liu et al., 2024c), ScienceQA (Lu et al., 2022), SEED-Bench (Li et al., 2023a), and RealWorldQA for general visual question answering capability. For OCR & Chart, we utilize ChartQA (Masry et al., 2022), TextVQA (Singh et al., 2019), and DocVQA (Mathew et al., 2021). In addition, we use MathVista (Lu et al., 2024) for math and reasoing evaluation. POPE (Li et al., 2023c) and GQA (Hudson & Manning, 2019) are adopted to assess model capabilities in hallucination-prone scenarios and complex visual reasoing challenges.

> 💡 **评估矩阵 — 多维覆盖**:
>
> | 能力维度 | Benchmarks |
> |---------|-----------|
> | General VQA | MMBench, MMStar, ScienceQA-Img, SEED-Bench-Img, RealWorldQA |
> | OCR & Chart | TextVQA, ChartQA, DocVQA |
> | Math & Reasoing | MathVista |
> | Hallucination | POPE |
> | Complex Visual Reasoing | GQA |
>
> 覆盖面广但缺少一些 high-profile benchmarks（如 MMMU, MME, MM-Vet），可能与骨干能力限制有关——在"更难的推理"上基线可能过低。

### 4.3. Benchmark Results

We developed three models for a comparative study: a baseline model trained without recurrence, a model trained with a mean recurrence of 32, and a third model that incorporates hierarchical visual information with the same recurrence level. To assess performance, we benchmarked these models against several open-source models, including Gemma-3-4B-PT (Team, 2025), MobileVLM V2 7B (Chu et al., 2024), Bunny-v1.1-4B (He et al., 2024), Imp-4B (Shao et al., 2025), and Emu3 (Wang et al., 2024c). The main results are shown in Table 4. HIVE is evaluated using r = 32.

> 💡 **Table 4（主表）精读 — HIVE vs 开源模型**:
>
> | 维度 | HIVE 表现 | 分析 |
> |------|---------|------|
> | General VQA | 竞争性但非最优 | MMBench 69.6（最高 Bunny 74.2），SEED 70.5（最高 Bunny 72.5）——但考虑到仅有 4B 参数，已相当有竞争力 |
> | ScienceQA-Img | **91.6（最佳）** | 超越 8B Emu3（89.2）和 7B MobileVLM（74.8）——HIVE 的招牌成绩，体现复杂知识型视觉任务的 strength |
> | RealWorldQA | 57.9 | 接近 Emu3（57.4），大幅领先 Gemma3-4B（45.5） |
> | OCR | 中等偏下 | TextVQA 57.5（低于 Emu3 64.7），ChartQA 67.0（低于 Emu3 68.6）——OCR/text-heavy 场景不是 HIVE 的强项 |
> | DocVQA | 73.2 | 有竞争力但不是 best（Emu3 76.3） |
> | POPE | **87.6（最佳）** | 抗幻觉能力 best——循环 refinement 有助于减少幻觉，符合预期 |

> 💡 **关键洞察 — HIVE 的 "强项" 与 "弱项" 模式**:
> - **强项**: 需要深层次视觉理解的任务（ScienceQA：科学推理 + 图像；POPE：精准视觉 grounding）
> - **弱项**: 需要细粒度文本识别的任务（TextVQA, ChartQA，本质上是 OCR 导向的）
> - **原因推测**: 层级视觉注入偏重语义理解（ViT 的层级特征天然适合语义聚合），但文本/OCR 信号在 ViT 预训练中不够充分

**Table 5（三变体消融）:**

| Benchmark | Baseline (r=1) | w/o Hier (r=32) | w/ Hier (r=32) | Gain (Recurrence) | Gain (Hier) |
|-----------|---------------|-----------------|-----------------|-------------------|-------------|
| MMStar | 33.28 | 48.44 | 49.79 | +15.16 | +1.35 |
| SEED_img | 42.37 | 70.46 | 70.48 | +28.09 | +0.02 |
| MMB_dev | 21.74 | 68.04 | 69.59 | +46.30 | +1.55 |
| RWQA | 41.44 | 57.52 | 57.91 | +16.08 | +0.39 |
| SQA_img | 60.09 | 89.39 | 91.57 | +29.30 | +2.18 |
| **Avg. (General)** | **39.78** | **66.77** | **67.87** | **+26.99** | **+1.10** |
| DVQA_val | 24.04 | 73.72 | 73.20 | +49.68 | -0.52 |
| TVQA_val | 30.56 | 57.66 | 57.54 | +27.10 | -0.12 |
| **Avg. (OCR)** | **27.30** | **65.69** | **65.37** | **+38.39** | **-0.32** |
| MathV_mini | 24.50 | 35.00 | 34.70 | +10.50 | -0.30 |
| POPE | 74.84 | 87.02 | 87.61 | +12.18 | +0.59 |
| GQA | 44.80 | 57.71 | 57.89 | +12.91 | +0.18 |
| **Avg. (Others)** | **48.05** | **59.91** | **60.07** | **+11.86** | **+0.16** |

> 💡 **消融分析 — 三条核心发现**:
>
> **1. Recurrence 是性能提升的主驱动力**: 从 r=1 到 r=32，各维度均有大幅提升。尤其是 SEED-Bench (+28.09), MMBench (+46.30), Document VQA (+49.68)——这些是需要多步推理或复杂视觉理解的任务，recurrence 的迭代 refinement 效果显著。
>
> **2. 层级视觉注入收益 modest 且存在 trade-off**:
>    - 在 General VQA 上平均 +1.10，ScienceQA +2.18——收益存在但 modest
>    - 在 OCR 上平均 -0.32——层级视觉特征对 OCR 场景的帮助有限
>    - 在 MathVista 上 -0.30——数学推理更多依赖语言推理，额外的视觉信息可能引入噪声
>
> **3. Baseline (r=1) 的低起点值得注意**: r=1 相当于只有一次前向的标准 MLLM。39.78 的 General VQA avg 非常低，说明单纯的 Huginn 骨干（不加 recurrence）在视觉任务上的表现很弱。这佐证了 Table 2 中 Huginn 语言能力偏弱的事实。

Based on the results, HIVE demonstrates a competitive edge in parameter efficiency and specialized visual reasoing. Despite its compact 4B architecture, the model achieves 91.6 on ScienceQA-Img, notably outperforming the larger 8B Emu3 and the 7B MobileVLM V2. This indicates that HIVE is particularly effective at handling complex, knowledge-based visual tasks. Furthermore, it achieves the highest reliability in the POPE benchmark (87.6), suggesting a robust capability to mitigate object hallucination compared to its peers.

The model also exhibits impressive data efficiency. While trained on 6.5M samples, HIVE consistently outperforms or matches models like Gemma3-4B-PT, which benefit from a much larger 4T token pre-training scale, across benchmarks such as RealWorldQA and DocVQA. Overall, HIVE strikes a balance between model size and performance. This is particularly notable because it manages to overcome the inherent limitations of its relatively lightweight Huginn backbone to achieve results that rival established baselines.

There remains substantial space for performance optimization. We recognize that the current model can be further elevated through finer hyperparameter tuning and more sophisticated dynamic resolution configurations, which could better capture the intricate spatial details required for advanced OCR and document understanding tasks.

> 💡 **性能天花板分析**: 论文明确承认"有大量优化空间"——更大的动态分辨率、更精细的超参调优。结合 OCR 的弱表现，动态分辨率可能是更关键的提升方向（而非继续在层级注入上做文章）。

**Recurrence improves the performance.** Figure 4 illustrates the performance scaling of three model variants across varying recurrence steps r: (1) a non-recurrent baseline (trained with r = 1), (2) a recurrent variant without hierarchical cues (r_bar = 32, w/o Hier.), and (3) our full recurrent model with hierarchical visual cues (r_bar = 32, w/ Hier.). The empirical results yield several key insights:

- **Iterative Refinement Gains**: While the non-recurrent baseline remains stagnant at a low performance level (averaging 59.0% across all steps), both recurrent variants exhibit a dramatic upward trajectory as r increases. For instance, the hierarchical model climbs from 32.82% at r = 1 to a peak of 91.57% at r = 32 validating that iterative recurrence allows the model to progressively refine its internal representations.

- **Impact of Hierarchical Cues**: The incorporation of hierarchical cues is associated with modest performance gains in this setting. At the recurrence depth of r = 32 the full model reaches 91.57%, compared with 89.39% for the "w/o Hier." variant.

- **Performance Saturation**: We observe a clear "diminishing returns" effect beyond r = 32. The performance gains for the hierarchical model plateau, moving from 91.57% (r = 32) to a slight fluctuation at 91.27% (r = 64). This convergence indicates that the model's representational capacity saturates at this depth, where additional computational steps no longer yield meaningful accuracy improvements.

> 💡 **Figure 3/4 批读 — Recurrence Scaling 曲线**:
> - **r=1 基准线 (59.0%)**: 三种模型在 r=1 时无差异（都只走一次），基本持平
> - **r=1→r=32**: 两种 recurrency 变体急速爬升，验证了"迭代 refinement 有效"的核心 claim
> - **r=32→r=64**: 收益递减，plateau——representational capacity 饱和。这在 loop transformer 的预期之中：同一组参数做太多次迭代，边际收益递减
> - **层级注入的增益**: 整条曲线上 w/ Hier 始终略高于 w/o Hier，但差距不大且随 r 增大而收窄

**Hierarchical cues help understanding.** To further examine the role of hierarchical visual cues, we report fine-grained results across six core dimensions in Table 4. Compared with the recurrent baseline (r_bar = 32, w/o Hier.), the hierarchical recurrent variant (r_bar = 32, w/ Hier.) shows generally positive trends in several categories. In particular, HIVE yields moderate gains in Logic Reasoing (LR, +2.54%), Attribute Reasoing (AR, +1.99%), Relation Reasoing (RR, +1.74%), and Coarse Perception (CP, +3.04%), suggesting that hierarchical visual priors can be incorporated effectively into the recurrent framework. Although the differences are limited in instance-level perception (FI), the overall results indicate that hierarchical cues are compatible with loop-based latent reasoing and can provide additional support in complex visual understanding.

> 💡 **Figure 4 (MMBench 六维度精细分析)**:
>
> | 维度 | w/o Hier | w/ Hier | Gain | 分析 |
> |------|---------|---------|------|------|
> | CP (Coarse Perception) | baseline | +3.04% | **最大增益** | 全局/粗粒度感知受益最明显——层级注入的"从浅到深"课程天然适合 coarse-to-fine 理解 |
> | LR (Logic Reasoing) | baseline | +2.54% | 第二大增益 | 逻辑推理受益——可能因为多尺度视觉信息提供了更丰富的推理证据 |
> | AR (Attribute Reasoing) | baseline | +1.99% | | 属性推理受益——中层视觉特征有助于 objects/attributes 识别 |
> | RR (Relation Reasoing) | baseline | +1.74% | | 关系推理受益——高层语义特征有助于理解 objects 间的 spatial/semantic 关系 |
> | FC (Fine-grained Cross-instance) | baseline | minimal | | 跨实例细粒度感知增益有限 |
> | FI (Fine-grained Instance-level) | baseline | minimal | **几乎无增益** | 实例级细粒度感知没受益——这个结果合理：层级抽取的 global-to-regional 特征可能丢失了 instance-level 的精确 spatial 信息 |

### 4.4. Adaptive Compute

To optimize the efficiency-performance trade-off, Huginn has implemented an adaptive computation mechanism that dynamically adjusts the number of recurrence iterations during inference. This optional mechanism lets the model determine the termination of recurrence based on the convergence of hidden states. A relative change metric is defined as follows:

$$
\mathrm { n o r m \_ d i f f } = \frac { \| \mathbf h _ { t } - \mathbf h _ { t - 1 } \| _ { 2 } } { \| \mathbf h _ { t } \| _ { 2 } } .
$$

> 💡 **公式批读 — Norm Diff (收敛度量)**: 连续两次迭代的 hidden state 相对变化量。当变化足够小时，说明 hidden state 已经收敛，继续迭代的边际收益低。这是自适应早停的信号基础。

To further enhance inference efficiency, Huginn adopts a specialized KV-cache management scheme with a periodic retrieval strategy. For the i-th token during the r-th recurrence step, the latest-m4 mechanism retrieves the KV-cache from the most recent valid step j that aligns with the current block's functional cycle. Specifically:

$$
j ^ { * } = \left\{ \begin{array} { l l } { \operatorname* { m a x } \{ j \mid j \leq r , j \equiv _ { 4 } r , \mathcal { T } _ { j , i } = 1 \} } & { r \geq 2 , } \\ { r } & { r < 2 , } \end{array} \right.
$$

where T_{j,i} ∈ {0, 1} denotes the validity of the cache at step j for token i, and ≡_4 denotes congruence modulo 4. This periodic reuse of cache states maintains temporal consistency while significantly reducing redundant computations.

> 💡 **机制拆解 — Latest-m4 KV-Cache**: 每 4 步重用一次 KV-Cache（通过模 4 同余约束），在保持时序一致性的同时减少冗余计算。r < 2 时不重用（因为还没有足够的历史缓存）。这是一个工程优化，与核心方法无关但对推理效率重要。

To quantify the computational effect, we analyze the average recurrence steps required for the first token under the adaptive early-exit setting (max r = 32). As shown in Figure 5, incorporating hierarchical cues is associated with faster convergence of hidden states across MMBench, MMStar, RealWorldQA, and ScienceQA_img. Concretely, the mean reasoing steps decrease from 25.4 to 18.1 on MMBench, 24.9 to 17.7 on MM-Star, and 24.8 to 17.0 on ScienceQA_img. On RealWorldQA, the average computation depth decreases from 25.5 to 14.5. This leftward shift in the step distribution suggests that hierarchical visual cues can provide useful multi-scale information that helps the model meet the exit criterion earlier in some cases. Overall, these results indicate that hierarchical cue injection is compatible with reducing the number of recurrence steps required under adaptive computation.

> 💡 **Figure 5 批读 — 自适应早停步数分布**:
>
> | Benchmark | w/o Hier (mean steps) | w/ Hier (mean steps) | 节省步数 | 加速比 |
> |-----------|----------------------|---------------------|---------|--------|
> | MMBench | 25.4 | 18.1 | 7.3 | 28.7% |
> | MMStar | 24.9 | 17.7 | 7.2 | 28.9% |
> | ScienceQA | 24.8 | 17.0 | 7.8 | 31.5% |
> | RealWorldQA | 25.5 | 14.5 | 11.0 | **43.1%** |
>
> **关键发现**: 层级视觉注入在所有 benchmark 上均使收敛步数显著减少（25-43% 的加速比）。这可能是 HIVE 最有 practical value 的结果——层级注入不仅是 accuracy gain，更重要的是 **efficiency gain**。多尺度视觉信息为 hidden state 提供了更丰富的初始化信号，使其更快达到稳态。
>
> **为什么 RealWorldQA 加速最大？** RealWorldQA 是真实场景理解，global-to-local 的多尺度信息天然有助于场景理解，所以层级注入提供的信号最有用。

---

## 三、Summary

- **三阶段训练**: Stage1 (V-L Alignment, LR=1e-3) → Stage2 (多模态预训练+OCR+纯文本, LR=1e-5) → Stage3 (SFT 3.4M, LR=1e-6)
- **核心 Benchmark 结果**:
  - ScienceQA-Img: 91.6 (best), POPE: 87.6 (best)
  - OCR 弱（TextVQA 57.5, ChartQA 67.0）——非 HIVE 强项
- **消融三大发现**:
  1. Recurrence 贡献巨大（+27 avg vs r=1）
  2. 层级注入额外 modest gain（+1.10 avg, 部分子任务负收益）
  3. 层级注入在 Coarse Perception (+3.04%) 和 Logic Reasoing (+2.54%) 上增益最显著
- **Recurrence Scaling**: r=1→r=32 性能线性增长，r=32→r=64 饱和（diminishing returns）
- **自适应早停**: 层级注入使收敛加速 25-43%——这是 practical deployment 的关键收益
