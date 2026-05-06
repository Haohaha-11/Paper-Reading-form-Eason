[← 返回 README](../README.md)

# 3. Preliminary and Motivation

## 📌 预览
这一节是 DMLR 的证据地基：先证明视觉依赖在 token/chain 级别都不均匀，再证明 confidence 和正确性、reasoning quality、visual grounding 有统计相关。

---

# 3. Preliminary and Motivation

As shown in Figure 12, existing reasoning paradigms commonly suffer from insufficient visual grounding, unstable tool invocation, and high computational overhead. These limitations motivate a fundamental question: why can’t MLLMs reason like humans do, dynamically deciding how to reason and which visual information to pay attention on during the thinking process? To this end, we organize the section around two research questions: (RQ1) Whether multimodal models require visual perception at every step of reasoning? (RQ2) If not, can their internal representations indicate when visual perception and reasoning is required?

> 💡 **研究问题拆解**: RQ1 对应 dynamic visual injection：如果不是每步都需要视觉，就不该全量注入。RQ2 对应 confidence reward：如果内部状态能提示何时需要视觉，就能用它指导 latent token 更新。

# 3.1 Dynamic Perception-Reasoning is Necessary

Definition 3.1 (Visual Dependency Score). Let the visual input be denoted as I, and its perturbed version as $\tilde { I } .$ Given a query $q _ { \ast }$ , the model’s dependence on visual information can be quantified by measuring the output discrepancy between the original and perturbed visual inputs. Specifically, for the i-th generated sequence $\mathcal { X } _ { i } = \{ x _ { i , 1 } , x _ { i , 2 } , . . . , x _ { i , t } \} _ { \mathrm { { \scriptsize { : \scriptsize ~ } } } }$ , the visual dependency score at position $t$ is defined as:

> 💡 **Visual Dependency Score 动机**: 这个定义用“原图 vs 扰动图”下同一 token 的 log-prob 差来衡量视觉依赖。它不是看 attention 大小，而是看视觉输入改变后 token 概率是否变，这更接近因果敏感性。

$$
S _ { i , t } = \log \pi _ { \theta } ( x _ { i , t } \mid x _ { i , < t } , I , q ) - \log \pi _ { \theta } ( x _ { i , t } \mid x _ { i , < t } , \tilde { I } , q )
$$

> 💡 **公式批读**: $S_{i,t}$ 越大，说明当前位置 token 更依赖真实图像。这里的关键不是绝对 accuracy，而是沿 reasoning chain 找出哪些 token 真正在“看图”。这为后面“只在需要时注入 visual patches”提供依据。

where $\pi _ { \theta } ( \cdot )$ denotes the token-level conditional probability distribution of the model. A larger $S _ { i , t }$ indicates a stronger dependency of the generated token on visual information. Building upon the above metric, we analyze visual dependency on the Math-Vision benchmark using the Qwen2.5-VL-7B [42] at two levels. First, for individual reasoning chains, we compute token-level visual dependency scores, capturing how much each generated token relies on visual information, as illustrated in Figure 2(a). Second, as shown in Figure 2(b), we aggregate these

![Figure 2](../images/7f9abab2feb7d772dc4bb2d00ceb66838534ac87a159fd2562e3310834768535.jpg)
*Figure 2: Analysis of visual dependency in reasoning. (A) Token-level distribution shows visual sensitivity is concentrated in a few tokens. (B) Chain-level distribution reveals large variation in visual reliance across reasoning trajectories.*

> 💡 **Figure 2 批读**: 左图说明 token-level visual sensitivity 是尖峰状的，不是每个 reasoning token 都同等需要视觉；右图说明不同推理链之间的视觉依赖强度也差异很大。DMLR 因此不采用固定注入位置，而是让 latent state 自己选择 patch。

scores across full reasoning trajectories to obtain chain-level visual dependency, which reveals how different reasoning paths vary in their reliance on visual perception. These results reveal that:

> 💡 **段落衔接**: MinerU 把这一段被 Figure 2 切开了，这里按论文语义合并阅读：作者先定义 token-level score，再汇总成 chain-level dependency。两个层级共同支持“动态视觉使用”。

Takeaway 1. The dependency on visual input across the reasoning process is highly uneven: only a small subset of tokens show strong sensitivity to visual features, while the majority operate independently of the image.

> 💡 **Takeaway 1 批读**: 如果大多数 token 不强依赖图像，全程反复注入视觉信息就是冗余计算，还可能把无关视觉噪声带进推理。

$\blacktriangle$ Takeaway 2. Across reasoning chains sampled from the same model, visual dependency varies substantially.
Chains exhibiting stronger visual reliance consistently yield higher accuracy.

> 💡 **Takeaway 2 批读**: 这里进一步说“更会看图”的链更准，但不等于“看越多越准”。后文 Table 2 会显示全量注入反而不稳，真正需要的是相关 patch，而不是更多 patch。

# 3.2 Internal Confidence Affects Multimodal Reasoning

Definition 3.2 (Confidence Gain). Let $I$ denote the visual input, $q$ the query, and $\mathcal { T } _ { t }$ denotes the reasoning at step t. The Confidence Gain at step $t$ is defined as the change in the probability of the ground-truth answer $Y _ { g t }$ after adding step $x _ { t }$ . $A$ positive $\mathcal { G } _ { t }$ suggests that step $x _ { t }$ strengthens the confidence, whereas a negative value indicates the opposite.

> 💡 **Confidence Gain 定义**: 这不是推理时直接可用的 reward，因为定义里用了 ground-truth answer。它更像 motivation analysis，用来验证“confidence gain 与好推理相关”。真正方法里的 reward 后面改成 top-k entropy 的无标签 confidence。

$$
\mathcal { G } _ { t } = \log \pi _ { \theta } ( Y _ { g t } \mid I , q , \mathcal { T } _ { \leq t } ) - \log \pi _ { \theta } ( Y _ { g t } \mid I , q , \mathcal { T } _ { < t } )
$$

> 💡 **公式批读**: Eq. Confidence Gain 衡量“加入当前 reasoning step 后，模型对真答案更有把握还是更没把握”。它把 reasoning step 的价值转成概率增量，后续 reward 设计则用熵近似这种把握程度。

❖ Observation 1: Higher Confidence Tends to Indicate Higher Reasoning Accuracy. We analyze reasoning chains generated by various reasoning models across four benchmarks, where all chains are partitioned into a correct set $\tau ^ { + }$ and an incorrect set $\mathcal { T } ^ { - }$ based on their answer correctness. We then compute the proportion of reasoning steps for each chain that obtain a positive confidence reward. As shown in Figure 3(a), reasoning chains in ${ { \mathcal { T } } ^ { + } }$ exhibit a substantially higher proportion of positive confidence increments compared to those in $\mathcal { T } ^ { - }$ , indicating that the reasoning leading to correct answers tends to exhibit more stable and higher confidence.

> 💡 **Observation 1 批读**: 这里证明的是 answer-level correlation：正确链更常出现正 confidence gain。它支持“优化 confidence 可能提升答案正确率”，但仍是相关性，不保证每次 confidence 增加都是真的推理进步。

❖ Observation 2: Confidence Reflects Reasoning Chains Quality. We investigate whether confidence dynamics reflect reasoning quality by evaluating reasoning chains within the correct set ${ { \mathcal { T } } ^ { + } }$ using the evaluator GPT-4o [43]. Each chain is assessed for logical validity and factual consistency, and categorized into Faithful and Spurious groups. As shown in Figure 3(b), faithful reasoning chains exhibit a higher proportion of positive confidence increments, suggesting that confidence improvement not only correlates with answer accuracy but also reveals the intrinsic quality of the reasoning process.

> 💡 **Observation 2 批读**: 作者没有只看最终答案，而是在正确答案内部区分 faithful/spurious。这个设计很重要，因为多模态模型可能“猜对但理由错”。confidence 若能区分 faithful 与 spurious，才更适合作为 latent optimization 的目标。

![Figure 3](../images/8b03a5807361e0b58c300c41c7dbfc5cca41a235ad632a27017ee71c48207f0a.jpg)
*Figure 3: Analysis of the relationship between confidence and reasoning quality. (A) Correct reasoning chains exhibit substantially higher frequencies of positive confidence gains than incorrect ones. (B) Faithful reasoning shows consistently stronger confidence improvement than spurious reasoning.*

> 💡 **Figure 3 批读**: Figure 3 是 confidence reward 的核心背书。A 看最终正确性，B 看 reasoning faithfulness；两者同时成立，才说明 confidence 不只是“模型自信地胡说”的信号。

![Figure 4](../images/85e39c41b187ca14ae0b522f50e94baa4d3591ad7bda213e48f99e6a1edec8e5.jpg)
*Figure 4: Analysis of the relationship between confidence and visual grounding. (A) Hallucinated steps show lower confidence than non-hallucinated ones. (B) Hallucinated steps exhibit weaker image relevancy than their counterparts.*

> 💡 **Figure 4 批读**: Figure 4 把 confidence 和视觉 grounding 接上：hallucinated steps 的 confidence 与 image relevancy 都更弱。对 DMLR 来说，这说明低 confidence 时回看视觉 patch 有合理性。

❖ Observation 3: High Confidence Aligns with Stronger Visual Grounding. We further evaluate various reasoning models on the perception benchmark to analyze the relationship between confidence and visual grounding. Each step in the reasoning chain is categorized as hallucinated or non-hallucination based on whether it refers to an object actually present in the image. As shown in Figure 4, hallucinated steps exhibit lower confidence and weaker visual grounding, while non-hallucinatory steps maintain higher and more stable confidence with stronger visual alignment. The results indicate that confidence acts as an intrinsic signal of visual faithfulness, with higher confidence consistently associated with more reliable reasoning.

> 💡 **Observation 3 批读**: 这一观察把 DMLR 的两个模块连起来：confidence 不仅指导 latent token 更新，也用于判断视觉证据是否有帮助。风险是 hallucination 标注依赖对象是否存在，复杂数学/图表推理中的 grounding 可能更难判定。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 / 结论 |
|------|-------------|
| Visual Dependency Score | 原图与扰动图下 token log-prob 差 |
| 分析模型 | Qwen2.5-VL-7B |
| 分析 benchmark | MathVision |
| Confidence Gain | 加入 step 前后 ground-truth answer log-prob 差 |

### 核心洞察
1. 视觉依赖是稀疏且链间差异大的，所以固定注入/全量注入都不理想。
2. confidence 同时关联 answer correctness、reasoning faithfulness 和 visual grounding。
3. 这节的证据大多是相关性分析，后文方法需要通过消融证明它能转化为实际性能收益。

### Q&A 批注记录
- **Q: 推理时没有 ground truth，Confidence Gain 怎么用？**
  A: 第 3 节的 Confidence Gain 是分析工具；第 4 节实际 reward 使用 top-k truncated entropy，不需要 ground truth。
