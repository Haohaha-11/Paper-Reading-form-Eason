[← 返回 README](../README.md)

---

## 📌 Preview

MedMO 被定位为当前最佳的全开源多模态医学基础模型，统一了 VQA、QA、报告生成和 grounding 能力。未来工作将聚焦于灾难性遗忘缓解和更广泛的模态覆盖。

---

## 5. Conclusion

We introduced MedMO, a general-purpose medical multimodal foundation model that unifies visual grounding, clinical reasoning, and language understanding across diverse medical modalities. MedMO is trained with a scalable fourstage post-training pipeline that includes large-scale alignment, high-resolution fine-tuning, instruction tuning, and reinforcement learning with verifiable rewards. This design enables robust multimodal comprehension and precise spatial localization. Experimental results show substantial gains over strong open-source baselines across VQA, text QA, report generation, and grounding benchmarks, establishing MedMO as the best fully open-source medical multimodal foundation model to date. As an open medical MLLM, MedMO provides a scalable path toward reliable and transparent medical vision language systems. Future work could explore strategies to better retain SFT knowledge within reinforcement learning frameworks.

**Limitation.** MedMO's stage-wise training introduces minor task-level performance shifts, as shown in Figures 5 and 6, a typical behavior in large multimodal models due to catastrophic forgetting [55]. Future work will focus on improving cross-task retention while expanding coverage across additional medical imaging modalities.

---

## 🔖 Summary

MedMO 代表了当前开源医学 VLM 的前沿水平：全开源（模型 + 数据 + 训练方案）、多任务（VQA + QA + 报告 + grounding）、多模态（8 种成像模态）。明确承认灾难性遗忘为首要局限性既诚实又具有前瞻性——这为医学基础模型的持续/终身学习指明了未来研究方向，在医疗领域新疾病、新成像协议和新临床指南不断涌现的背景下尤为重要。

> 💡 **问题动机 -- "Retain SFT knowledge within RL" 核心挑战**: 这个future work方向直指当前RLHF/RLVR范式的根本矛盾：RL奖励信号（bounding box quality, label accuracy）可能与SFT阶段学到的 rich clinical knowledge 产生冲突。表现为表1中MedMO-8B在QA上（61.3%）优于Next版本（60.1%）。未来的解决方案可能包括：(1) SFT+RL联合训练而非串行, (2) 在RL reward中加入knowledge retention信号, (3) 使用更强的KL正则化约束policy偏离。

> 💡 **开放性评估**: MedMO 的"fully open-source"声明包括：(a) 模型权重（HuggingFace collection）, (b) 代码（GitHub）, (c) 训练数据和配方（论文中详细列出）。但需注意 MedTrinity 的18.5M instruction-following pairs 是自动生成的，其生成代码是否开源、generation pipeline 是否可复现，会影响 MedMO 的"完全可复现性"。

> 💡 **Q&A 批注记录**: 
> - **Q**: "MedMO 4B 和 8B 的差异是否支持 scaling law？"
> - **A**: 从Table 1的VQA avg来看, MedMO-4B (45.4%) → MedMO-8B (63.2%) 提升 +17.8%, 说明 scale 对医学VQA的贡献显著。但有趣的是 MedMO-4B-Next (68.5%) 已接近 MedMO-8B-Next (72.7%), 仅差4.2%, 说明 RL post-training 可以在一定程度上弥补 scale 的不足 -- 高质量的训练策略比纯模型大小更重要。
> - **Q**: "该论文的局限性是否被充分讨论？"
> - **A**: 论文只提到了 catastrophic forgetting 这一个 limitation。没有讨论的潜在局限包括：(a) 缺乏 clinical validation/reader study, (b) 没有 fairness/bias 分析, (c) 只在英语环境下评估, (d) 没有推理延迟和部署效率的讨论, (e) RL 阶段只用了 300K 数据（vs SFT 的 26M），RL 的 scaling 潜力未探索。

[← 返回 README](../README.md)
