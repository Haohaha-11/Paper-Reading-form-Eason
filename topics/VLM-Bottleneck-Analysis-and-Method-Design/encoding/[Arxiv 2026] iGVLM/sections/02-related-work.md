[← 返回 README](../README.md)

# 2. Related Work

## 一、Preview

Related work 分为两条主线：(1) VLM 中的视觉编码器设计，从 CLIP 出发，经历了多编码器融合和查询感知调制两个方向，iGVLM 定位在后者但采用了解耦设计；(2) VLM 评测基准的演进，从 VQA/MS-COCO 到 MMStar，再到本文提出的 MM4——专门评测多查询多指令下的视觉感知一致性。

---

## 二、原始文本

**Vision Encoders in Vision–Language Models.** Recent years have witnessed rapid progress in Vision–Language Models (VLMs) (Chen et al., 2025; Zhou et al., 2024; Wu et al., 2024b), driven by advances in both large-scale multimodal pretraining and architectural design. A foundational line of work, exemplified by CLIP (Radford et al., 2021), demonstrates that contrastive learning on large-scale image–text pairs can effectively align visual and textual representations, forming the basis of many modern VLMs. Subsequent studies have explored how to enhance the visual encoding component within VLMs to better support downstream multimodal reasoing.

One line of research focuses on strengthening visual representations by aggregating information from multiple encoders or pretrained visual models. For example, Ranzinger et al. (Ranzinger et al., 2024) fuse features from multiple vision encoders, while Tong et al. (Tong et al., 2024) augment CLIP features with representations from DINOv2 (Oquab et al., 2023), leading to improved visual grounding. Monkey (Li et al., 2024) further explores fine-tuning multiple vision encoders to support high-resolution image understanding. These approaches primarily aim to improve the capacity and coverage of visual representations, but typically rely on static encoders whose outputs are invariant to task-specific instructions.

A complementary line of work investigates how to introduce query- or instruction-awareness into the vision encoder. QA-ViT (Ganz et al., 2024) incorporates query-aware cross-attention to modulate visual features based on textual prompts, enabling more effective integration of visual and linguistic information for question answering. While such designs provide a degree of instruction-dependent adaptation, they often operate within a single encoder pathway and offer limited control over how pretrained visual representations are preserved or modified. In contrast to these approaches, our work focuses on explicitly decoupling representation preservation from instruction-conditioned modulation within the vision encoder.

> 💡 **Related Work 结构拆解 — 两条研究主线**:
>
> | 方向 | 代表方法 | 核心思路 | 与 iGVLM 的关系 |
> |------|---------|---------|----------------|
> | 增强视觉表征能力 | AM-RADIO, DINOv2+CLIP, Monkey | 融合多编码器，提升特征覆盖度和分辨率 | 互补：iGVLM 可以叠加在这些编码器之上 |
> | 引入查询/指令感知 | QA-ViT | 通过 cross-attention 用文本调制视觉特征 | 对比：QA-ViT 在单一路径内调制，iGVLM 采用解耦双路径 |

> 💡 **关键区分**: "improving the **capacity** of visual representations" vs. "introducing instruction **awareness** into the vision encoder"。这是两种不同维度上的改进：前者是让编码器"看得更多/更清楚"，后者是让编码器"根据指令看不同的东西"。iGVLM 属于后者，但超越了 QA-ViT 的单路径设计。

> 💡 **解耦 vs 单路径的关键论证**: QA-ViT 在单个编码器路径内做 cross-attention 注入文本信息，这带来一个问题——你无法控制预训练的视觉先验是否被破坏。iGVLM 的解法：让两个分支独立运作，一个永远保持原始视觉特征，一个负责指令调制，最后通过 Zero-FFN 融合。这保证了最坏情况下（Zero-FFN 输出为零）模型退化为原始 LLaVA-1.5。

**Evaluating Vision–Language Models.** Evaluating the capabilities of VLMs has been an active area of research, leading to the development of a diverse set of multimodal benchmarks. Early benchmarks, such as VQA (Goyal et al., 2017), MS-COCO (Sharma et al., 2018), and OK-VQA (Schwenk et al., 2022), provide task-specific assessments of multimodal perception and reasoing. More recent efforts aim to offer broader and more challenging evaluations of multimodal understanding and instruction following (Wu et al., 2024a; Fu et al., 2023; Cheng et al., 2023). MMStar (Chen et al., 2024b) further consolidates existing benchmarks and introduces a carefully curated, vision-dependent evaluation suite designed to mitigate data leakage and spurious correlations.

Despite these advances, most existing benchmarks primarily assess general-purpose multimodal capabilities and evaluate each query in isolation. As a result, they provide limited insight into whether a model can consistently adapt its visual perception to different instructions grounded in the same image. To address this gap, we introduce MM4, a controlled diagnostic benchmark specifically designed to evaluate question-aware visual understanding. MM4 challenges models to answer multiple, semantically distinct queries associated with a single image, enabling more fine-grained analysis of instruction-conditioned visual perception and multi-query consistency.

> 💡 **评测基准演进路线**:
> ```
> VQA/MS-COCO/OK-VQA (任务特定)
>     → MME/Q-Bench (更广的多模态理解+指令跟随)
>         → MMStar (精选题目，防数据泄露)
>             → MM4 (本文：多查询、同图多问、一致性评测)
> ```

> 💡 **MM4 的设计哲学**: 现有 benchmark 评测的是"模型能答对多少个独立问题"，而 MM4 评测的是"模型能否对同一张图的不同问题给出**一致的、问题感知的**答案"。这两者的区别类似于：前者测视力表的单个字母识别，后者测的是"看清之后能否灵活切换关注点"。这对于评测 instruction-conditioned perception 是本质性的。

---

## 三、Summary

| 维度 | 内容 |
|------|------|
| **视觉编码器方向 1** | 多编码器融合增强表征容量（AM-RADIO, DINOv2+CLIP, Monkey）——静态编码 |
| **视觉编码器方向 2** | 查询感知调制（QA-ViT）——单路径内做，条件化控制有限 |
| **iGVLM 定位** | 在方向 2 上，但采用**解耦双分支**，显式分离表征保留与调制 |
| **评测演进** | 独立问题评测 → MMStar（防泄露） → MM4（同图多问一致性） |
| **MM4 独特价值** | 评测 instruction-conditioned perception 的一致性，而非孤立的正确率 |
