[← 返回 README](../README.md)

# 04 Discussion / Conclusions

> 💡 **Hao 批注 - 讨论部分结构**: 4.7节用"问答"形式组织了讨论，回答了7个社区关心的问题。这种组织方式实用且易读。4.8节讨论了局限性，5节给出结论。

---

## 4.7 Discussion: Key Questions Answered

### Q1: Do we need complex embedding-based MIL?

> 💡 **Hao 批注**: 回答很明确：不需要。在4个数据集上，实例级MIL方法要么最优、要么次优，无论看最优组合还是平均AUC。

**Answer: No.** Instance-based MIL methods, coupled with good SSL feature extractors, are either the best or second best performing methods, both in terms of combinations and averaged AUC scores, across all four datasets.

### Q2: What is the best backbone?

> 💡 **Hao 批注**: CNNs比ViTs更鲁棒(在本文训练配置下)。性能随backbone增大而提升(ResNet50 > ResNet18)。建议：有计算资源用20x+大架构(ResNet50/ViT Large)，资源受限用10x+ResNet18。

**Answer:** CNNs seem more robust than ViTs. Performance improves with larger backbones: ResNet50 best AUC on Camelyon16 is 3 points above ResNet18. With great computational capability, use data at 20x and big architectures (ResNet50/ViT Large). With limited resources, use ResNet18.

### Q3: What is the best SSL?

> 💡 **Hao 批注**: SSL显著优于ImageNet。MoCoV3/DINO/Barlow Twins 效果最好，Barlow Twins因其简单架构和易调参被特别推荐。SimCLR效果较差(已知结论)。

**Answer:** SSL pre-training consistently outperforms ImageNet initialization. Best results with MoCoV3, DINO, or Barlow Twins. Barlow Twins is particularly advantageous due to its simple architecture and ease of tuning. SimCLR falls short.

### Q4: Do we benefit from pathology-specific SSL techniques?

> 💡 **Hao 批注**: 有帮助但有限。病理适应增强提升+1.2至+3.3 AUC (平均)。CluBYOL比BYOL高8.6 AUC (平均所有MIL方法)。但改进相对较小——这些方法主要只是调整增强策略，而非从根本上重新设计SSL。

**Answer:** Yes, pathology-adapted techniques improve performance of traditional SSL methods (+1.2 to +3.3 AUC on average). CluBYOL outperforms BYOL by 8.6 AUC points (averaged across all MIL methods). Using domain knowledge to adapt augmentations enhances generalization.

### Q5: Can we leverage foundation models?

> 💡 **Hao 批注**: 可以，但必须用病理领域适配的。DINOv2(通用视觉)效果最差——领域差异大。UNI效果最好(最大数据集+最新SSL+最大backbone)。注意UNI(ViT Large, 307M)参数量远超本文训练的任何backbone。

**Answer:** Yes, but it's important to use those specifically adapted to pathology. DINOv2 (general vision) performs significantly worse due to domain gap. UNI achieves highest performance (largest dataset: 100M+ patches, latest SSL framework, larger backbone).

### Q6: Should we simply use MaxMIL?

> 💡 **Hao 批注**: 不一定。MaxMIL在肿瘤小而分散时有效(Camelyon16)，但在肿瘤覆盖>80%组织区域时(TCGA-NSCLC, VisioMel)，MeanMIL更好。推荐：当肿瘤大小未知或病理标记微妙时，用MixMIL/AutoMIL等自适应方法。

**Answer:** Not always. In TCGA-NSCLC and VisioMel (tumor > 80% of tissue), MeanMIL achieves best results. When tumor size is unknown/variable, use adaptive methods (MixMIL, AutoMIL, etc.).

---

## 4.8 Limitations and Perspectives

> 💡 **Hao 批注 - 局限性**:

### 1. Multiple Magnification Levels
Not used due to computational complexity. Multi-scale approaches (like hierarchical DINO) known to improve performance (~4-5x more data per resolution level).

### 2. Other Types of MIL
Only 4 embedding-based MIL methods included. Missing: graph-based methods (PatchGCN), hierarchical methods (HIPT), multi-magnification methods, etc.

### 3. Influence of Number of Training Samples N
Instance-based methods (especially MaxMIL) theoretically require more WSIs (only 1 instance per slide used in decision). Embedding-based methods have more parameters and thus also need more data. Not systematically studied.

### 4. Better Pathology-Adapted SSL Methods
> 💡 **Hao 批注 - 最重要的未来方向**: 当前病理适应SSL方法只是对通用SSL的微小修改(增强调整/聚类策略)，真正需要的是"模仿病理学家推理过程"的SSL方法——利用多倍率、先验医学知识等。这也可以集成到UNI等基础模型中。

Current pathology-adapted SSL methods involve minor modifications (augmentation adjustments, positive/negative sampling, clustering strategies). More effort should be put into developing well-adapted SSL methods that leverage multiple magnification levels and prior medical knowledge of pathologists. An SSL approach that mimics the reasoning process of a pathologist could potentially be integrated into foundation models like UNI.

---

## 5 Conclusions

> 💡 **Hao 批注 - 结论**: 简洁地重申了核心发现。最后一句是整篇论文的方向性建议——"在WSI领域，应该把更多努力投入到好的SSL方法而非复杂的嵌入级MIL方法中"。这与Paper 1的结论形成了有趣的互补：Paper 1说"光加架构不够，还要确保架构被正确训练"，Paper 2说"其实不需要那么多架构"。

In this paper, we conducted a large-scale study using 6 SSL with 4 backbones, 4 foundation models and 10 MIL methods on four diverse datasets, covering binary and multi-class classification tasks with increasing clinical complexity. Our results demonstrate that simple instance-based MIL methods with very few parameters, combined with robust SSL feature extractors, can achieve competitive or superior performance than complex embedding-based MIL methods across different backbones. The newly proposed instance-based MIL methods achieve new SOTA results on BRACS and Camelyon16. By sharing our code, pre-trained models, and insights, we aim to provide valuable resources for future research in this domain.

---

## Appendix: Full Quantitative Results Tables

> 💡 **Hao 批注 - 主要结果表概览**: Tables 2, 3, 7, 8, 9 包含所有710次实验的AUC结果。这些表非常大(每个表 ~10行 x 13列)，是论文的主要实证贡献。Tables 4-6 是额外的消融/分析。

- **Table 2**: Camelyon16 x10, ViT-Tiny + ViT-Small + ResNet18, all SSL+MIL combinations
- **Table 3**: Camelyon16 x20, ResNet50, all SSL+MIL combinations
- **Table 4**: Camelyon16 x10, Foundation Models (CTransPath, PathAugFM, DINOv2, UNI)
- **Table 5**: Pathology-adapted SSL methods comparison (Barlow/MoCoV3/BYOL vs _path variants, CluBYOL)
- **Table 6**: ImageNet initialization impact (ResNet18, Barlow/SimCLR/DINO, 3 runs each)
- **Table 7**: TCGA-NSCLC x10, all backbones + SSL + MIL
- **Table 8**: BRACS x10 (multi-class), all backbones + SSL + MIL
- **Table 9**: VisioMel x10, all backbones + SSL + MIL
