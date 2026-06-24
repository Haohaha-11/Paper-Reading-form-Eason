[← 返回 README](../README.md)

# 4. Experiments

## 一、Preview

实验部分覆盖五个子维度：(4.1) 实验设置（LLaVA-1.5 框架、CLIP-L-336、三 backbone 对比、1.1x 额外 GPU 小时）；(4.2) MMStar 主结果（跨 backbone 一致提升 +4.4/+3.6/+4.5，且保持吞吐量）；(4.3) MM4 结果（n-out-of-4 一致性分析）；(4.4) 其他 benchmark 泛化性验证；(4.5) 消融实验（AdaLN/Zero-FFN/Static Branch 分别移除的影响、替代调制设计、多 backbone 扩展）。

---

## 二、原始文本

### 4.1. Experimental Settings

**Training Setup.** All experiments are conducted within the LLaVA-1.5 training framework to ensure a controlled and fair comparison. We use the same open-source training data as LLaVA-1.5, including 558K image–text pairs for alignment pretraining and 665K samples for instruction tuning, without introducing any additional data. All models share the same visual backbone, CLIP-Large-336 (Radford et al., 2021), and differ only in how visual features are conditioned on textual instructions. We evaluate iGVLM across multiple language backbones, including Vicuna-7B, Vicuna-13B (Chiang et al., 2023), and Qwen2.5-3B (Yang et al., 2024), enabling analysis of both intra-family scaling and cross-architecture generalization. All experiments are run on 8 NVIDIA A100 GPUs under identical hardware and software configurations. Compared to LLaVA-1.5, training a 7B version of iGVLM incurs a moderate computational overhead of approximately 1.1× GPU hours, reflecting the lightweight nature of the proposed instruction-guided visual modulation.

> 💡 **实验控制的严谨性**: 所有对比方法使用相同的训练数据（558K + 665K）、相同的视觉 backbone（CLIP-L-336）、相同的训练框架。唯一的变量是视觉特征如何被指令条件化。这确保了性能差异可归因于方法本身而非数据或基础设施。

> 💡 **三 backbone 的评测意图**: Vicuna-7B 和 Vicuna-13B 测**同系列 scaling**；Qwen2.5-3B 测**跨架构泛化**。特别值得注意的是 Qwen2.5-3B 远小于 Vicuna-7B（3B vs 7B），但在 MM4 上 iGVLM-3B 表现最好——这暗示 instruction-conditioned visual modulation 的效果与语言模型的**质量**（而非绝对规模）更相关。

**Evaluation Benchmarks.** Our primary evaluation is conducted on MMStar (Chen et al., 2024b), a vision-dependent multimodal benchmark designed to assess fine-grained reasoning while minimizing data leakage, and we report MM-Star results as the main indicator of general-purpose multimodal performance. To further examine instruction sensitivity and generalization, we additionally evaluate on a range of established benchmarks, including VQAv2 (Goyal et al., 2017), GQA (Hudson & Manning, 2019), POPE (Li et al., 2023), VizWiz (Gurari et al., 2018), and ScienceQA (Lu et al., 2022), which collectively assess open-ended visual understanding, robustness to hallucination, zero-shot generalization, and scientific reasoning.

**Baselines.** We adopt LLaVA-1.5 (Liu et al., 2024) as the primary baseline, as it provides fully open-source data, model weights, and training code. All baseline models are trained using the same data, optimization settings, and vision backbone as iGVLM ensuring that performance differences arise solely from differences in visual encoding strategies. For Vicuna-based backbones, we additionally compare with representative instruction-aware modulation methods, including QA-ViT (Ganz et al., 2024) and DyFo (Li et al., 2025), which introduce query-aware modulation and expertguided search, respectively. For Qwen2.5-3B, we compare against LLaVA-1.5 with the same backbone due to architectural compatibility, ensuring consistent evaluation across different model families and instruction-conditioning strategies.

---

### 4.2. Results on MMStar

We evaluate iGVLM on MMStar under three representative settings. For Vicuna-based models (7B and 13B), we compare iGVLM with both the static baseline LLaVA-1.5 and two instruction-aware modulation methods, QA-ViT and DyFo. For the Qwen2.5-3B backbone, we compare against the corresponding LLaVA-1.5 model due to architectural compatibility. All methods are evaluated using identical training data, vision backbones, and inference configurations to ensure a controlled comparison.

As shown in Table 1, iGVLM achieves the best overall performance across all backbones. On Vicuna-7B, iGVLM improves the average MMStar score by +4.4 points over LLaVA-1.5, outperforming both QA-ViT (+1.2) and DyFo (+2.7). Notably, the gains are most pronounced on instruction-sensitive and fine-grained reasoning dimensions, including Instance Reasoning (IR), Logical Reasoning (LR), and Science & Technology (ST), where iGVLM consistently surpasses both baselines. A similar trend is observed on Vicuna-13B: iGVLM achieves an average improvement of +3.6 points, exceeding QA-ViT (+3.1) and DyFo (+1.8), with particularly strong gains on Fine-grained Perception (FP) and ST. These results indicate that explicitly conditioning the utilization of visual features enables more effective instruction-aware reasoning than single-path modulation (QA-ViT) or expert-guided search (DyFo).

> 💡 **Table 1 深度解读 — MMStar 各维度表现**:
>
> **Vicuna-7B 下 iGVLM 的关键增益来源**:
> - LR (Logical Reasoning): 24.0 → **32.8** (+8.8) — 最大单项增益，说明指令引导的视觉调制对逻辑推理至关重要
> - ST (Science & Technology): 13.6 → 19.6 (+6.0) — 科学图表理解受益显著
> - IR (Instance Reasoning): 38.8 → 41.2 (+2.4)
>
> **为什么 QA-ViT 的增益有限？** QA-ViT 只在 ViT 高层注入文本信息，条件化信号太弱，底层视觉特征仍是任务无关的。iGVLM 的 AdaLN 在**所有层**都做调制，从浅到深形成层次化指令感知。

> 💡 **吞吐量 vs 精度 — iGVLM 的 Pareto 最优性**:
>
> | 方法 | Vicuna-7B Avg. | Throughput (it/s) | 效率评分 |
> |------|---------------|-------------------|---------|
> | LLaVA-1.5 | 30.3 | 13.5 | baseline |
> | QA-ViT | 31.5 | 12.9 | 精度+，效率≈ |
> | DyFo | 33.0 | **0.49** | 精度+，效率崩 |
> | iGVLM | **34.7** | 11.1 | 精度++，效率−18% |
>
> DyFo 的吞吐量从 13.5 降至 0.49（约 27x 下降），因为每个问题需要多轮 MCTS 搜索。iGVLM 只需一次前向传播（多了一个 CLIP text encoder + AdaLN 的计算），效率损失可接受。

In addition to accuracy, iGVLM maintains a favorable efficiency profile. Despite achieving higher overall performance, iGVLM preserves throughput comparable to LLaVA-1.5, with only a modest reduction (13.5→11.1 for Vicuna-7B and 9.8→8.6 for Vicuna-13B). In contrast, DyFo incurs a severe efficiency penalty due to repeated expertguided search, reducing throughput by more than 20× (13.5→0.49 for Vicuna-7B and 9.8→0.47 for Vicuna-13B). QA-ViT maintains efficiency similar to LLaVA-1.5, but achieves only limited accuracy gains, highlighting the tradeoff between conditioning strength and computational cost in existing approaches.

Results on Qwen2.5-3B further confirm the generality of iGVLM. Compared to LLaVA-1.5, iGVLM improves the average MMStar score from 16.8 to 21.3 (+4.5), with consistent gains across all capability dimensions. Taken together, these results demonstrate that iGVLM strikes a more effective balance between instruction-aware reasoning and computational efficiency than prior dynamic modulation methods, validating the advantages of decoupled instructionguided visual encoding on a general-purpose multimodal benchmark.

> 💡 **Qwen2.5-3B 的结果值得关注**: iGVLM 在 3B 模型上提升 +4.5，绝对分数 21.3 仍然远低于 Vicuna-7B 的 34.7。这说明：instruction-guided visual modulation 能带来**一致的相对提升**（无论语言模型大小），但**绝对性能的上限由语言模型决定**。视觉调制解决的是"如何利用视觉特征"，但"理解和推理视觉特征"的能力来自 LLM。

---

### 4.3. Results on MM4

We evaluate iGVLM on the proposed MM4 benchmark, which is specifically designed to assess question-aware and multi-query visual reasoning under shared visual inputs. Unlike general-purpose benchmarks that evaluate each query in isolation, MM4 requires models to adapt visual perception consistently across multiple, semantically distinct questions grounded in the same image.

**Quantitative Results.** As shown in Table 2, closed-source systems such as GPT-4o (OpenAI, 2024) and Qwen2.5-vlmax (Qwen Team, 2025) achieve the highest absolute scores, reflecting the advantages of large-scale proprietary models. Among open-source systems, iGVLM consistently outperforms its corresponding LLaVA-1.5 baselines under the same backbone. In particular, iGVLM-3B achieves the best performance among open-source models, improving over LLaVA-1.5-3B despite having the same parameter scale. Notably, iGVLM-3B also outperforms the larger iGVLM-13B, indicating that MM4 performance is driven more by instruction-aware visual utilization than by parameter count alone. These results suggest that the proposed decoupled visual modulation can effectively leverage stronger language backbones without architectural modification.

> 💡 **MM4 的关键洞察 — LLM 质量 > LLM 规模**:
> - iGVLM-3B (Qwen2.5-3B, n=4: 29) > iGVLM-13B (Vicuna-13B, n=4: 23)
> - 3B 模型超越 13B 模型，这在传统 benchmark 上几乎不可能出现
> - 原因：Qwen2.5 系列的语言理解和指令跟随能力优于 Vicuna 系列（尽管参数更少）。MM4 评测的是 instruction-conditioned perception，语言模型的指令理解质量比参数规模更关键。

**Multi-Query Consistency Analysis.** MM4 adopts an increasingly strict evaluation protocol in which a model receives credit only if it correctly answers at least n out of four questions per image (n = 1, 2, 3, 4). As reported in Table 2, performance decreases monotonically as n increases for all models, reflecting the growing difficulty of maintaining consistent reasoning across multiple queries. However, iGVLM exhibits a noticeably slower performance degradation compared to baseline methods, particularly at higher consistency thresholds (n = 3 and n = 4). This behavior indicates that instruction-guided visual modulation enables more stable adaptation of visual attention across different questions, rather than relying on isolated correct predictions.

> 💡 **性能退化速度分析 — Table 2 关键对比**:
>
> 从 n=1 到 n=4 的得分下降：
> - GPT-4o: 170 → 58 (保留 34%)
> - iGVLM-13B: 161 → 23 (保留 14%)
> - LLaVA-1.5-13B: 161 → 18 (保留 11%)
> - iGVLM-3B: 164 → 29 (保留 18%)
> - LLaVA-1.5-3B: 165 → 27 (保留 16%)
>
> 在开源模型中，iGVLM 在高严格度下的相对优势最明显。这说明：不是 iGVLM 答对了更多"简单题"，而是 iGVLM 能更好地**在同一个图上对多个不同问题给出都正确的答案**——这正是 instruction-conditioned perception 的核心价值。

To contextualize these results, we note that random guessing yields an expected score of approximately 0.7 at n = 4, derived from a per-question accuracy of 0.25 over four independent questions. All evaluated models perform substantially above this baseline, confirming that MM4 provides a discriminative evaluation of multi-query reasoning. Importantly, the relative advantage of iGVLM becomes more pronounced under stricter consistency requirements, aligning with the intended diagnostic goal of MM4.

**Qualitative Analysis.** We further visualize representative examples in Figure 3 to illustrate how instruction-guided visual modulation affects model behavior. Compared with LLaVA-1.5-13B and QA-ViT-13B, iGVLM-13B more accurately localizes instruction-relevant regions under different queries. In the science diagram example, iGVLM distinguishes semantically similar stages such as evaporation and transpiration, while baseline models attend to ambiguous regions. In the food scene example, iGVLM demonstrates stronger compositional reasoning by correctly identifying missing objects and spatial relationships across multiple questions. Together, these qualitative and quantitative results confirm that iGVLM enhances question-aware visual perception by enabling consistent, instruction-conditioned feature utilization.

---

### 4.4. Other Benchmarks

To assess whether instruction-guided visual modulation affects general-purpose multimodal capabilities, we further evaluate iGVLM on a diverse set of established benchmarks, including VQAv2, GQA, POPE, VizWiz, and ScienceQA-IMG, as summarized in Table 3. These benchmarks cover complementary aspects of vision–language understanding, ranging from open-ended visual reasoning (VQAv2, GQA), hallucination robustness (POPE), real-world visual grounding (VizWiz), to domain-specific scientific reasoning (ScienceQA-IMG). In contrast, DyFo relies heavily on Monte Carlo Tree Search (MCTS) and lacks specialized search strategies for non-selective benchmarks such as VQAv2, GQA, and VizWiz, which significantly limits its generalizability compared to QA-ViT and our proposed iGVLM.

As shown in Table 3, iGVLM consistently maintains comparable or improved performance relative to LLaVA-1.5 across different model scales and backbones. For Vicunabased models, iGVLM yields modest but consistent gains on most benchmarks. In particular, iGVLM improves POPE accuracy from 85.4 to 85.9 on Vicuna-7B and from 85.4 to 86.1 on Vicuna-13B, indicating enhanced robustness against visual hallucination. Notable improvements are also observed on VizWiz, where iGVLM raises accuracy from 50.0 to 52.5 for Vicuna-7B and from 53.6 to 55.3 for Vicuna-13B, suggesting more reliable visual grounding under real-world conditions. On ScienceQA-IMG, iGVLM achieves clear gains for Vicuna-7B (+3.1), while maintaining comparable performance for Vicuna-13B.

Under the Qwen2.5-3B backbone, iGVLM shows a similar trend. While performance on VQAv2 and GQA remains comparable to LLaVA-1.5-3B, iGVLM substantially improves VizWiz accuracy from 50.7 to 53.4 and ScienceQA-IMG accuracy from 72.2 to 73.0. Across all evaluated benchmarks, no systematic performance degradation is observed, indicating that instruction-guided visual modulation does not compromise general-purpose multimodal reasoning. Overall, these results suggest that iGVLM serves as a drop-in enhancement to existing vision–language models, improving instruction-aware visual utilization while preserving broad applicability across diverse multimodal tasks.

> 💡 **"No Performance Degradation" 的重要性**: 很多方法在提升某一维度能力时会损害其他能力（经典的 trade-off）。iGVLM 在所有 5 个额外 benchmark 上均未出现系统性性能退化，说明解耦双分支设计中的 frozen static branch 确实起到了"底线保护"作用——最坏情况下 Zero-FFN 退化为零，模型退化为 LLaVA-1.5。

> 💡 **DyFo 的泛化性缺陷**: DyFo 依赖 MCTS 搜索，但在 VQAv2/GQA/VizWiz 等非选择题场景下缺少专门的搜索策略，导致无法评估。这与 iGVLM 的"通用 modulation"形成鲜明对比：iGVLM 不需要问题类型特定的策略，AdaLN 调制对所有问题类型通用。

---

### 4.5. Ablation Study

**Effect of Architectural Components.** We first examine the contribution of the key design components in iGVLM by ablating (i) instruction-conditioned adaptive layer normalization (AdaLN), (ii) the Zero-FFN adapter used for feature fusion, and (iii) the static branch in the dual-branch architecture. As shown in Table 4, removing AdaLN (w/o AdaLN) leads to consistent performance drops on both MMStar and MM4, indicating that layer-wise, instructionconditioned normalization is critical for effective visual modulation. Eliminating the Zero-FFN adapter (w/o FFN) further degrades performance, suggesting that controlled, gradual integration of dynamic features is necessary to avoid disrupting pre-trained visual representations. The most significant degradation is observed when the static branch is removed (w/o Pure), highlighting the importance of preserving task-agnostic visual priors alongside instruction-guided adaptation. Together, these results support the hypothesis that instruction-aware perception benefits from explicitly decoupling representation preservation from task-specific modulation.

> 💡 **Table 4 消融解读 — 组件重要性排序**:
>
> | 移除的组件 | MMStar | MM4 | VQAv2 | VizWiz | 影响程度 |
> |-----------|--------|-----|-------|--------|---------|
> | None (完整 iGVLM) | 36.4 | 23 | 80.2 | 55.3 | — |
> | -w/o AdaLN | 35.1 (-1.3) | 22 (-1) | 80.2 | 53.5 | 中等 |
> | -w/o FFN | 34.1 (-2.3) | 17 (-6) | 80.1 | 54.7 | 较大 |
> | -w/o Pure | **27.3** (-9.1) | **5** (-18) | **60.2** | **37.9** | 致命 |
>
> **关键发现**:
> 1. **Static Branch 是最关键的组件**（w/o Pure 导致灾难性退化）：没有静态分支，模型完全失去了预训练的视觉感知能力，只剩指令调制分支无法提供稳定的视觉特征
> 2. **Zero-FFN 对 MM4 的影响特别大**（-6）：没有渐进融合机制，指令特征直接覆盖静态特征，破坏了一致性推理
> 3. **AdaLN 的移除影响相对温和**（-1.3）：说明即使没有指令调制，双分支 + Zero-FFN 也有一定效果；但加上 AdaLN 后才达到最优

**Comparison with Alternative Modulation Designs.** We further compare iGVLM with two representative variants that adopt different strategies for integrating instruction signals into visual features. iGVLM-MoF (Tong et al., 2024) interleaves static and dynamic tokens, while iGVLM-Cross (Peebles & Xie, 2023) replaces AdaLN with crossattention-based interaction. As reported in Table 5, both variants underperform the original iGVLM. MoF weakens the explicit separation between static and dynamic representations, while cross-attention introduces additional computational overhead and optimization noise without improving instruction consistency. These comparisons suggest that AdaLN-based modulation offers a more effective and efficient mechanism for conditioning visual representations on textual instructions.

> 💡 **Table 5 — 替代调制设计的失效原因**:
>
> | 方法 | MMStar | 核心问题 |
> |------|--------|---------|
> | iGVLM (AdaLN) | 34.7 | baseline |
> | iGVLM-MoF (交织) | 32.0 | 破坏静态/动态的显式分离，混合 token 导致特征空间混乱 |
> | iGVLM-Cross (Cross-Attn) | 33.0 | 引入额外计算和优化噪声，但未带来指令一致性提升 |
>
> **MoF (Mixture of Features) 的问题**: 将静态和动态 token 交织后送入 LLM，二者在同一个 token 序列中难以被区分，LLM 不知道哪个是"稳定的通用特征"、哪个是"指令敏感的调制特征"，导致信息混淆。
>
> **Cross-Attn 的问题**: 看似更直接的交互方式，但视觉 token 参照文本做 cross-attention 引入了不稳定的优化信号——每次文本不同，cross-attention 的行为差异很大，训练难以收敛到稳定状态。

**Scaling Behavior.** Finally, we analyze how model capacity influences instruction-aware reasoning by training iGVLM-1.5B based on the Qwen2.5-1.5B backbone and comparing it with larger variants. As shown in Table 6, even the smallest iGVLM model improves over its LLaVA-1.5B counterpart on MMStar (19.7 vs. 17.1), indicating that instruction-guided visual modulation is beneficial across model scales. However, the lower MM4 score of iGVLM-1.5B (16 vs. 19) reveals that consistent multi-instruction reasoning requires sufficient language modeling capacity. Performance improves monotonically from 1.5B to 3B, 7B, and 13B, and notably, iGVLM-3B outperforms the larger Vicuna-13B variant on MM4. This trend suggests a strong synergy between the proposed dual-branch vision encoder and modern language backbones, and highlights that instruction-aware visual reasoning is jointly constrained by visual modulation and language capacity.

> 💡 **Scaling 分析 — Table 6 揭示的"双重约束"**:
>
> | Model | MMStar (Avg.) | MM4 (n=4) | 洞察 |
> |-------|--------------|-----------|------|
> | iGVLM-1.5B | 19.7 | 16 | 太小，MM4 甚至不如 LLaVA-1.5-1.5B |
> | iGVLM-3B | 21.3 | **29** | MM4 最优（开源），MMStar 较低 |
> | iGVLM-7B | **34.7** | 17 | MMStar 最优，MM4 一般 |
> | iGVLM-13B | **36.4** | 23 | MMStar 最优，MM4 次优 |
>
> **核心洞察**: MMStar 和 MM4 对模型能力的要求不同：
> - MMStar（通用多模态推理）：随 LLM 规模单调增长
> - MM4（多查询一致性）：存在"甜点区"——太小的 LLM 无法处理复杂的指令切换（1.5B），太大的 LLM（Vicuna-13B）虽然能力强但不如 Qwen2.5-3B 的架构先进
> - **结论**: instruction-conditioned perception 的性能同时受 (i) 视觉调制质量 (ii) LLM 的指令理解和推理能力 双重约束

---

## 三、Summary

| 实验维度 | 核心结论 |
|---------|---------|
| **MMStar 主结果** | iGVLM 在所有 backbone 上一致提升（+4.4/+3.6/+4.5），超越 QA-ViT 和 DyFo，吞吐量仅下降 18% |
| **MM4 一致性** | iGVLM 在高严格度（n=3, n=4）下退化更慢，iGVLM-3B 开源最优（29），证明 LLM 质量 > 规模 |
| **泛化性** | 5 个额外 benchmark 上无系统退化，POPP/VizWiz 有增益，证明是 drop-in enhancement |
| **消融 — 组件重要性** | Static Branch > Zero-FFN > AdaLN，但三者缺一不可 |
| **消融 — 调制设计** | AdaLN > Cross-Attn > MoF，验证了 AdaLN 是最优的条件化方式 |
| **Scaling** | MMStar（通用推理）随规模增长，MM4（一致性推理）存在甜点区，受双重约束 |
