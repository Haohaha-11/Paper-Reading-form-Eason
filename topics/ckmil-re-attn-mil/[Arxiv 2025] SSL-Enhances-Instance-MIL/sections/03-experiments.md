[← 返回 README](../README.md)

# 03 Experiments

> 💡 **Hao 批注 - 实验设计**: 4个数据集按临床复杂度递增排列：Camelyon16(转移检测，二分类)→TCGA-NSCLC(肺癌亚型，二分类)→BRACS(乳腺癌亚型，三分类，含非典型类)→VisioMel(黑色素瘤复发，高度不平衡二分类)。评估指标以AUC为主(对类别不平衡鲁棒)。

---

## 3.1 Datasets and Experimental Setup

> 💡 **Hao 批注 - 数据集概览**:

| Dataset | Classes | Slides | Clinical Complexity | Imbalance |
|---------|---------|--------|---------------------|-----------|
| Camelyon16 | 2 (Normal/Tumor) | 400 | Low (metastasis detection) | Moderate |
| TCGA-NSCLC | 2 (LUAD/LUSC) | 997 | Medium (cancer subtyping) | Balanced |
| BRACS | 3 (Benign/Atypical/Malignant) | 547 | High (fine-grained, rare atypical) | Imbalanced |
| VisioMel | 2 (Relapse/No) | 1342 | Very High (melanoma prognosis) | Highly imbalanced |

Validation strategy: model selection on validation set, report on test set (except VisioMel: validation set only, last-epoch model to avoid overfitting).

---

## 3.2 Pre-processing

All WSIs cut into 256x256 non-overlapping patches from foreground tissue at x10 magnification (Camelyon16 also at x20). Results:
- CAMx20: 4.6M patches
- CAMx10: 0.6M patches
- TCGA-NSCLC: 2.7M patches
- BRACS: 1.4M patches
- VisioMel: 2.7M patches

---

## 3.3-3.6 Training Details

> 💡 **Hao 批注 - 训练配置要点**: (1) SSL预训练: 200 epochs, LARS优化器(CNN)/AdamW(ViT), batch size = 1024 (4x256 or 8x128 across GPUs), cosine annealing, warm-up 10 epochs。(2) MIL训练: 100 epochs, batch size = 1 slide, Adam optimizer, cosine annealing, grid search for learning rate。

- SSL pre-training: solo-learn library, 200 epochs, cosine annealing, warm-up 10 epochs
- Backbone sizes: ResNet18 (11.7M), ResNet50 (25.5M), ViT-Tiny (5.8M), ViT-Small (22.2M)
- MIL training: 100 epochs per model, grid search for LR, batch size=1 (standard in MIL)

---

## 4.1 Camelyon16 Results

> 💡 **Hao 批注 - Camelyon16 核心发现**: Tables 2-3 (本文最大的结果表) 展示了所有组合的AUC。核心模式是：(1) 实例级MIL的红色标注(每列最佳平均AUC)频繁出现；(2) 嵌入级方法不占优；(3) ImageNet pretrain结果显著低于SSL。

**Table 2 (Camelyon16 x10)**: Instance-based LNPMIL achieves best average AUC with ViT-Tiny (76.4) and ResNet18 (90.6). MixMIL best average with ViT-Small (82.8).

**Table 3 (Camelyon16 x20 with ResNet50)**: MixMIL + DINO achieves new SOTA **99.1 AUC**. All methods with SSL > 92 AUC except DAMIL (~64).

**Table 4 (Foundation Models)**: UNI + CLAM = 99.1 AUC (same as SSL-trained SOTA, but UNI has 307M params vs 25.5M). DINOv2 significantly worse due to domain gap.

**Table 5 (Pathology-adapted SSL)**: Pathology-adapted augmentations improve performance by +1.2 to +3.3 AUC points on average. BYOL_path shows largest improvement (+7.0 on ABMIL). CluBYOL outperforms BYOL by 8.6 AUC points averaged.

**Table 6 (ImageNet Init)**: ImageNet initialization has negligible impact (+0.1/-0.1/-0.4 AUC). 200 epochs of SSL pre-training is sufficient regardless of initialization. Standard deviations generally < 2 AUC points.

![](../images/a3a732f67a10062beb9eaec2379d563409df54e4f27e162066a73d03c6f6d949.jpg)

> 💡 **Hao 批注 - Figure 6 (定性分析)**: 这是论文最有说服力的可视化。(1) MoCoV3 embeddings通过K-means聚类能比ImageNet更好地识别肿瘤区域(橙色箭头)；(2) 实例级MIL的patch scores准确高亮肿瘤区域，而DSMIL的attention scores不能识别整个肿瘤区域——embedding-based方法的"可解释性"实际不如instance-based的patch scores直接。

---

## 4.2 TCGA-NSCLC Results

> 💡 **Hao 批注 - TCGA-NSCLC**: 实例级和嵌入级持平。注意在这个数据集上ResNet18实际上优于ViT-Small(平均值约95 vs 85)——作者的结论是CNNs比ViTs更鲁棒。但也可能是因为ViT需要更多数据/epochs。

**Table 7**: Instance-based and embedding-based MIL on par. Best combinations evenly spread. Note: ResNet18 outperforms ViT-Small (avg ~95 vs ~85).

| Backbone | Best MIL | Best SSL | AUC |
|----------|---------|----------|-----|
| ViT-Small | DSMIL | MoCoV3 | 96.9 |
| ResNet18 | TransMIL | MoCoV3 | 97.6 |
| ResNet50 | TransMIL | DINO | 97.7 |

---

## 4.3 BRACS Results (Multi-class)

> 💡 **Hao 批注 - BRACS**: AttenMIL + DINO + ResNet18 实现新SOTA 89.4 AUC。BRACS整体AUC偏低(约70-90)，主要因为Atypical Tumor类别样本少且难以区分。

**Table 8**: AttenMIL + DINO + ResNet18 achieves new SOTA **89.4 AUC**. Overall lower scores due to limited Atypical Tumor cases.

---

## 4.4 VisioMel Results

> 💡 **Hao 批注 - VisioMel**: 最具挑战性的临床任务，AUC普遍最低(~60-77)。实例级和嵌入级方法表现相近。这是平衡性最差的数据集(213 pos / 1139 neg)，但AUC对imbalance鲁棒。

**Table 9**: Most challenging clinical task, lowest overall AUC (~60-77). Instance-based and embedding-based similar.

---

## 4.5-4.6 Training Ablations

> 💡 **Hao 批注 - 训练消融**: (1) SSL epoch数: 200 epochs是良好平衡，更长的训练对ViT backbone有额外收益。(2) Patch size: 224x224 vs 256x256影响不大，选择256以减少patch数量节省计算。

**Number of Epochs (Figure 10)**: 200 epochs chosen; extended pre-training enhances performance and stabilizes ViT backbones.

**Patch Size (Figure 11)**: 224x224 vs 256x256 comparable results. 256x256 chosen for computational reasons (fewer patches).

---

## Qualitative Results (Figures 6, 12)

> 💡 **Hao 批注 - 可解释性的关键对比**: Figure 6 的对比非常有力——左侧K-means聚类显示MoCoV3 embedding比ImageNet更好地分离肿瘤区域，右侧patch scores(MaxMIL)比attention scores(DSMIL)更准确地覆盖整个肿瘤区域。最后一行的误分类案例揭示了共同的失败模式：微小且变化微妙的肿瘤区域，无论哪种方法都难以捕捉——这也指出了更好SSL方法的方向。

![](../images/97fb1767f9a4dfc1e1c8a72b4bb2577d76715176baa914ef6fa0a0090ef99cea.jpg)

Figure 12: Qualitative results across all MIL methods on Camelyon16. Instance-based MILs (above blue line) show similar or better tumor localization than embedding-based MILs (below blue line). Red squares highlight patches with highest contribution to prediction.
