[← 返回 README](../README.md)

# 04 Discussion

## 4.1 Limitations

> 📄 **原文 - 4 Limitations**

We note several limitations. Our slide-level evaluation covers only two public classification benchmarks and does not assess transfer to other tasks such as survival analysis, retrieval, or treatment response predictions. It also uses one custom split, a single run per model, and a common five-epoch training recipe, which does not capture variability across splits or random seeds and may not optimally tune every model; scores are therefore not directly comparable with those reported under other protocols. GigaPath-Flash trades some absolute performance for efficiency, and efficiency measurements on a single NVIDIA A100 may not generalize across hardware or implementations. GigaTIME-Flash was evaluated with one LoRA configuration and a fixed 21-marker panel in cohorts with limited numbers of independent patients. Its in-silico predictions and windowed Pearson correlation do not establish cell-level accuracy or clinical utility. Broader multi-institutional and prospective validation across tasks, cohorts, scanners, and patient subgroups is needed before deployment.

> 💡 **Hao 批注 - 局限性评估 (客观批评)**:

**诚实的自评**:
1. **2 个基准太少**: PANDA + EBRAINS 确实无法代表病理 FM 的全部应用场景——缺少生存分析、MSI 预测、突变预测等关键临床任务
2. **单次运行无统计**: 无交叉验证、无 seed 平均、无置信区间——无法判断 0.826 vs 0.825 的差距是否统计显著
3. **5 epoch 可能不公平**: 大模型 (TITAN, UNI2-h) 可能需要更多轮次才能收敛——5 epoch 统一训练可能系统性地低估了大模型的真实能力
4. **窗口级 Pearson 不等于临床价值**: mIF 预测质量最终应由下游临床任务（如免疫亚型分类、治疗响应预测）验证

**未明确指出的局限**:
1. **LongNet 的 dilated attention 是否真的比 ABMIL 好**: 文中没有 ABMIL + GigaPath-Flash tile features 的对照实验来隔离 slide encoder 的收益
2. **蒸馏数据量未公开**: Providence 真实世界数据有多少 WSI、多少机构——同样是关键信息但缺失
3. **训练细节缺失**: 蒸馏温度、batch size、数据增强策略等超参数完全未提及——严格来说不可复现
4. **无多模态对比**: 为什么只比 tile-level vision-only 模型而不比 CONCH/MI-Zero 等 vision-language 模型？

## 4.2 Conclusion

> 📄 **原文 - 5 Conclusion**

We introduced GigaPath-Flash and GigaTIME-Flash, additions to the GigaPath/GigaTIME model family that provide an efficient foundation for whole-slide representation learning and spatial tumor immune microenvironment prediction. GigaPath-Flash retains competitive performance on the evaluated slide-level benchmarks with substantially reduced compute, while GigaTIME-Flash improves prediction quality over the original CNN-based GigaTIME with lower inference time and GPU memory requirements. Although broader evaluation and external validation remain necessary, these results highlight the potential of efficient foundation models to balance predictive performance with practical efficiency. By lowering computational barriers, we hope these models will enable research across increasingly large and diverse cohorts and help accelerate the shift from isolated tile-level analysis toward context-aware whole-slide modeling and scalable spatial characterization of the tumor immune microenvironment.

> 💡 **Hao 批注 - 论文价值总体评价**: 

**GigaPath-Flash 在病理 FM 生态中的位置**:
- 它填补了 "高效率 + 全切片上下文 + Apache-2.0" 的三元组空白。在此之前，高效率的模型 (Kaiko-S, Path Foundation) 没有 slide encoder；有 slide encoder 的 (GigaPath, TITAN, PRISM) 太高成本；许可开放的 (GigaPath) 又太贵。
- 对于资源受限的学术实验室和需要大规模推理的工业场景，GigaPath-Flash 是目前最实用的选择。

**对 ReadySlide 的关联**:
1. GigaPath-Flash 证明了 "压缩 + 保留性能" 在病理 FM 中是可行的——这为 ReadySlide 的压缩导向研究提供了实证支持
2. LongNet 的 slide-level 上下文编码是 tile→slide 信息聚合的一种高效方案，可以作为 ReadySlide 中 slide-level representation 的对标方法
3. GigaTIME-Flash 的跨模态预测 (H&E→mIF) 展示了蒸馏特征的多功能性——如果 ReadySlide 的压缩特征也能支持类似的多任务迁移，价值会进一步提升

**论文的软肋**:
1. 核心主张 "retains 97% of GigaPath's performance" 仅基于 2 个基准——如果在 10 个任务上平均可能是另一回事
2. 蒸馏方法几乎没有技术新颖性——就是 DINOv2 减 KoLeo——创新的重心在工程整合而非科学发现
3. 与 TITAN 和 PRISM 的对比受到 5-epoch 限制和自定义 split 的 confound——需要更严格的 head-to-head 比较
4. 缺少临床应用（如 Gleason 分级一致性 vs 病理医生、治疗决策影响）的验证——"降低计算门槛"对临床的影响仍是假设性的

---

> **论文标签**: #PathologyFoundationModel #KnowledgeDistillation #WholeSlideImaging #SpatialProteomics #EfficientFM #ApacheLicense #LongNet #GigaPath #Flash
> **与 CKMIL 主题关联**: GigaPath-Flash 提供了高效的全切片特征提取基线，其 LongNet dilated attention 与 CKMIL 跨尺度注意力共享 "高效建模长序列空间依赖" 的技术动机——两者可以形成互补（CKMIL 的 MIL 聚合器 + GigaPath-Flash 的瓦片特征）。
