[← 返回 README](../README.md)

# 04 — Discussion & Conclusion

## 4 Conclusion and future direction

> **原文**:

In this work we found that SiMLP, a simple fine-tuning method, enables pathology foundation models to effectively adapt to slide-level tasks. Extensive experiments demonstrate that SiMLP outperforms widely used MIL-based weakly supervised learning, confirming its strong performance and generalization ability.

Our findings provide four key insights for the future of computational pathology in the foundation model era:

1. **Patch-level foundation model development remains crucial**. While existing pretrained encoders enhance WSI analysis, balancing data redundancy and model complexity is essential. For instance, ViT-Base (CONCH) performed competitively against ViT-Giant (Prov-GigaPath). We encourage future research to explore efficient architectures, diverse multimodal models, and improved data-driven preprocessing strategies.

2. **Task-agnostic slide representation learning may be more impactful than weakly supervised learning**. Such representations improve generalization and stability while enabling broader applications like slide embedding retrieval and convenient multimodal integration.

3. **Advancing slide-level foundation models enhances clinical performance**. Pretraining slide encoders on large-scale datasets not only supports task-agnostic representation learning but also allows for performance improvements through diverse fine-tuning strategies.

4. **Tailored weakly supervised learning remains necessary for slide-level tasks**. SiMLP performs well broadly, however, weakly supervised learning still holds advantages in specific tasks, highlighting its effectiveness for clinically tailored applications. For example, it remains valuable for biomarker prediction, hierarchical classification of rare diseases [10,15], and long-tailed data analysis.

In summary, as pathology foundation models continue to evolve, simplifying traditional weakly supervised learning paradigms and pioneering a new generation of research directions will be key to further enhancing performance and enabling broader real-world applications in computational pathology.

> 💡 **四个洞察的层次**: Hao 批注 — 这四个洞察实际上形成了一个完整的范式递进：Insight 1（patch encoder 最重要）→ Insight 2（task-agnostic 表示更有价值）→ Insight 3（slide encoder 预训练有用但非必需）→ Insight 4（MIL 在特定场景仍有价值）。这是一个平衡且成熟的论述——没有过度宣称"SiMLP 替代一切 MIL"，而是精确指出了 task-agnostic pooling 的优势边界。

> 💡 **Insight 1 的隐含含义**: Hao 批注 — "ViT-Base (CONCH) 与 ViT-Giant (Prov-GigaPath) 竞争性表现"是一个重要的发现——说明模型规模不是唯一决定因素，预训练策略（CONCH 是 vision-language，GigaPath 是 vision-only DINOv2）和架构选择同样重要。这对我们的 ReadySlide 方向有直接影响：选择哪个 PFM 做 feature extractor 可能比 compression 方法的选择更重要。

> 💡 **Insight 2 与我们方向的一致性**: Hao 批注 — "task-agnostic slide representation may be more impactful than weakly supervised learning"——这正是我们 analysis-ready compression 的核心理念："压缩一次，任意 FM+task 都能用"。SiMLP 在 task-agnostic 方向上的成功增强了我们 pursuit 的信心。但我们需要注意：SiMLP 的 task-agnostic 是"mean pooling 不依赖任务标签"，而我们的 task-agnostic 是"compressed WSI 不依赖特定下游任务"——两者的 task-agnostic 含义不完全相同。

> 💡 **Insight 4 的保守性**: Hao 批注 — 作者承认 MIL 在特定场景仍有价值，列举了 biomarker prediction、hierarchical classification、long-tailed data。但我们在这篇论文中看到的 HER2 预测反例其实已经属于 "biomarker prediction" 范畴——说明 MIL 在这些场景的具体优势边界还需更多实验来界定。对 ReadySlide 的启示：我们的 allocator 可能在这些"MIL 仍有优势"的场景中获得最大的相对提升——当 mean pooling 不够用时，智能 retention（类似 attention 但更符合信息论）可能弥补 gap。

> 💡 **论文的最大贡献是方法论而非算法**: Hao 批注 — SiMLP 作为一个算法并不复杂（mean pooling + 2-layer MLP 可以在一页代码内实现）。这篇论文的真正价值在于：(1) 通过大规模系统性实验挑战了一个已被广泛接受的研究范式；(2) 为社区提供了一个强 baseline，促使未来工作在被接受前必须对比 SiMLP；(3) 重新将注意力从"如何更好地聚合 patches"转移到"如何获得更好的 patch 特征"——这与 PFM 时代的趋势一致。

> 💡 **与 TAPFM (Paper 7) 的对比**: Hao 批注 — 两篇论文从不同角度质疑了传统 MIL 范式。TAPFM 的解决方案是"用 ViT 内部注意力替代外部 MIL 聚合器"——仍然关注"如何更好地聚合"但用更优雅的方式；SiMLP 的解决方案是"不需要复杂聚合，mean pooling 就够了"——从根本上挑战聚合的必要性。从实验结果看，SiMLP 的论点更激进，但 TAPFM 在特定任务（BLCA FGFR3 TCGA 0.9021）上达到了更高的绝对性能——说明在某些任务上，"好的聚合"仍然优于"简单的聚合"。最值得探索的方向可能是：**在 task-agnostic mean pooling 的基础上，针对信息局部化严重的任务，用轻量级 attention 做 soft 补充**——类似 "mean pooling + residual attention correction"。

> 💡 **对我们 ReadySlide allocator learning 的直接影响**: Hao 批注 — SiMLP 验证了 (1) mean pooling 是强 baseline；(2) task-agnostic 表示优于 task-specific。这直接支持我们当前的 direction：**compression 应该是 task-agnostic 的**（不是为每个 downstream task 定制 compression policy），而 allocator learning 的目标不是替代 mean pooling，而是判断"哪些 patches 值得保留更多信息以最大化 task-agnostic 表示质量"。这篇论文进一步强化了"retention 分配的价值在于超越 mean pooling 的等权假设"这一 framing。但 SiMLP 也提醒我们：mean pooling 是一个极高的 baseline——allocator 需要在 BOTH "信息集中型"和"信息分散型"任务上都超越 mean pooling，难度不小。

## References (部分关键引用)

- [3] Chen et al. (2024) — UNI
- [8] Ilse et al. (2018) — ABMIL
- [18] Lu et al. (2024) — CONCH
- [21] Shao et al. (2021) — TransMIL
- [22] Song et al. (2024) — Morphological Prototyping (CVPR)
- [25] Tang et al. (2024) — RRTMIL (CVPR)
- [28] Vorontsov et al. (2024) — Virchow
- [30] Wang et al. (2024) — CHIEF (Nature)
- [33] Xu et al. (2024) — Prov-GigaPath (Nature)
- [34] Xu et al. (2025) — MIL meets FM (Med Image Analysis)
- [36] Zhang et al. (2022) — DTFD-MIL (CVPR)
- [38] Zhang et al. (2024) — ACMIL (ECCV)
