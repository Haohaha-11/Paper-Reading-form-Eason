# 2 Related Work

[← 返回 README](../README.md)

---

Large vision-language models (LVLMs) are built on foundation architectures such as CLIP-style image encoders and large language backbones, enabling strong instruction-following and open-ended reasoning Radford et al. (2021); Alayrac et al. (2022); Li et al. (2022); Liu et al. (2023); Dai et al. (2023). Reliability and grounding concerns emerge when these models generate fluent but incorrect outputs, which has motivated benchmark-centric studies of hallucination in captioning and VQA Rohrbach et al. (2018); Li et al. (2023b). Beyond LLaVA-Bench Zhou et al. (2023), recent evaluation suites such as MME Fu et al. (2023), SEED Bench Li et al. (2023a), and MM-Vet Yu et al. (2023) broaden coverage across multimodal skills and stress-test visual grounding in diverse settings. In parallel, interpretability work debates whether attention is a faithful explanation signal Jain & Wallace (2019); Wiegreffe & Pinter (2019). Relatedly, recent work on faithfulness and behavioral reliability shows that surface-level explanations can decouple from the internal determinants of outputs, including scenario-dependent shifts Chaudhury & Shiromani (2025); Shiromani et al. (2026). For VLMs specifically, recent evidence also reports the "see-but-not-believe" phenomenon, i.e., correct localization without correct reasoning Liu et al. (2025). Our contribution is therefore not the generic claim that attention alone is insufficient, but a cross-family, layerwise reliability analysis centered on early locking/symbolic detachment and on hidden-state reliability probes.

> 💡 **文献脉络梳理**: 相关工作的三个层次：
> 1. **架构层**: CLIP → LLaVA/InstructBLIP/Flamingo 等VLM架构的演进
> 2. **评估层**: 幻觉检测基准从POPE到MME/SEED-Bench/MM-Vet的扩展
> 3. **可解释性层**: NLP中的"Attention is not Explanation"辩论 (Jain & Wallace vs Wiegreffe & Pinter) 向多模态的延伸
> 本文的定位是在这三个层次之上添加第四层：**跨架构的、逐层的可靠性因果分析**。

Recent work on language prior highlights a core evaluation tension: should we assess whether the model gives the correct answer, or whether it truly integrates visual evidence? Long et al. (2025) asks a more representation-centric question and contrasts hidden trajectories with and without images to identify a Visual Integration Point (VIP) and define Total Visual Integration (TVI), a metric that quantifies how strongly visual evidence shapes representations. This reveals when models start "seeing" and how visual influence accumulates, addressing a gap left by output-only probes. Our study complements this line of inquiry but targets a different blind spot: we ask whether spatial attention structure itself is predictive of correctness, and whether reliability signals live in the generation dynamics rather than in the visual attention maps. In contrast to VIP/TVI, which measure representational shift induced by the image, we show that even when attention appears structurally grounded, it can be statistically decoupled from truthfulness; the strongest signals instead emerge from agreement across sampled reasoning paths and from hidden-state probes. This clarifies what our work addresses that prior representation analyses do not: reliability prediction and calibration, not just visual integration. Complementary benchmark and mitigation work further suggests that reliability is evaluation and decoding-dependent, motivating our focus on generation dynamics as a readout for correctness Thomas et al. (2026); Sahay et al. (2025).

> 💡 **机制拆解 — VIP/TVI vs VRP的区别**: 
> - VIP (Visual Integration Point): 测量视觉信息何时开始影响表示（"模型什么时候开始'看'"）
> - TVI (Total Visual Integration): 测量视觉证据对表示的总影响力（"模型'看'了多深"）
> - VRP (VLM Reliability Probe): 测量视觉注意力结构是否预测正确性（"看了对地方"是否等于"想对了答案"）
> 三者的区别在于：VIP/TVI关注的是视觉影响的**程度和时间**，VRP关注的是视觉注意力结构与输出可信度的**统计和因果关系**。本文的发现暗示：即使模型"看"了很多（高TVI），但如果"看"的位置与"想"的答案是脱钩的（Symbolic Detachment），TVI仍然不能保证可靠性。

To make the contribution boundary explicit: we do not claim to newly discover that attention can be unfaithful or that self-consistency helps; those are established in prior NLP/VLM literature. Our contribution is a unified, cross-family reliability study that links early-locking/symbolic-detachment dynamics to downstream correctness and shows that hidden-state probes provide the strongest single-pass reliability signals.

> 💡 **Q&A 批注记录**:
> *Q: VIP/TVI 和本文的Symbolic Detachment是什么关系？*
> A: 互补关系。VIP/TVI回答"视觉何时/多大程度影响表示"，适合研究模型是否真正使用了视觉。Symbolic Detachment回答"视觉注意力的空间质量是否反映答案可信度"，适合研究可靠性的预测。一个可能的综合图景是：高TVI + 高Symbolic Detachment = 模型大量使用了视觉，但视觉使用的好坏（而非数量）决定了答案正确性。
