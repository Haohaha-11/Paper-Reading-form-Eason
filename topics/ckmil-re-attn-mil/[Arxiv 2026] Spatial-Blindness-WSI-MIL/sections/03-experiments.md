[← 返回 README](../README.md)

# 03 Experiments

> 💡 **Hao 批注 - 实验设计**: 实验回答三个递进问题：(1) 空间盲能否在受控条件下被隔离？(Spatial-MNIST-Bag) (2) 残差设计能否提升真实 WSI 预测？(4 分类 + 5 生存) (3) 提升是否伴随更强的空间行为而非仅更大容量？(消融 + shuffle sensitivity + 定位) 实验设计逻辑清晰，从受控到真实逐步推进。

---

## 5.1 Experimental Setup

> 💡 **Hao 批注 - 实验配置**: 9 个公开 benchmark，统一 20x 256x256 patches，统一 UNI 1024-d features，统一 patient-level stratified split，统一 5-seed mean±std。这个统一性对公平比较很重要。

Evaluation covers 9 public WSI benchmarks: BRACS, PANDA, TCGA-NSCLC, TCGA-BRCA, and five TCGA survival cohorts (KIRC, KIRP, LUAD, STAD, UCEC). Slides processed into non-overlapping 256x256 patches at 20x with 1024-d UNI features. All baselines use the same features, patient-level stratified splits, and evaluation protocol. Results are mean+/-std over 5 random seeds.

Baselines: AB-MIL, CLAM-SB, DTFD-MIL, DS-MIL, TransMIL, ILRA-MIL, MHIM-MIL, DGR-MIL, 2DMambaMIL.

---

## 5.2 Controlled Evidence for Spatial Blindness (Spatial-MNIST-Bag)

> 💡 **Hao 批注 - 受控诊断基准**: 这是论文最具说服力的实验之一。Dataset A 是纯组合(坐标随机，label 取决于是否包含 digit "9")，Dataset B 是纯拓扑(所有 bag 有相同 digit multiset，label 取决于 5 个关键 digit 是否形成紧凑空间 motif)。AB-MIL 和 TransMIL 在 Dataset A 上近乎完美(0.998/0.995 AUC)，但在 Dataset B 上崩溃(0.505/0.532)——这直接证明了空间盲的存在。

> 💡 **Hao 批注 - 联合训练 vs 残差解耦的关键对比**: Dataset B 上 ResTopoMIL(Joint) = 0.684 vs ResTopoMIL = 0.987。注意在这个受控环境中，统计流甚至没有真正的标签信号(组合完全相同)，但联合训练下模型仍无法有效利用拓扑——这强烈支持了优化偏置(而非信号强弱)是根本原因。

Dataset A (Pure Composition): coordinates are random, label depends only on whether digit "9" appears.
Dataset B (Pure Topology): every bag contains the same digit multiset (one each of "1,3,5,7,9" plus 45 even digits). Label depends only on whether the five key digits form a compact spatial motif (positive: clustered around centroid; negative: uniformly scattered).

| Method | A: Comp. AUC | B: Topo. AUC |
|--------|-------------|-------------|
| AB-MIL | 0.998 | 0.505 |
| TransMIL | 0.995 | 0.532 |
| ResTopoMIL (Joint) | 0.991 | 0.684 |
| ResTopoMIL | 0.994 | 0.987 |

---

## 5.3 WSI Classification (Table 2)

> 💡 **Hao 批注 - 分类结果**: ResTopoMIL 在所有 4 个数据集上 Accuracy 最优，3/4 AUC 最优(MHIM-MIL 在 NSCLC AUC 略高)。关键的是，效果最显著的 BRACS 和 PANDA 正是组织架构依赖最强的任务——BRACS 涉及非典型/恶性鉴别(PANDA 的 Gleason 分级依赖腺体形成)。TCGA-BRCA 上也保持优势(导管 vs 小叶的生长模式区分)。

> 💡 **Hao 批注 - 参数效率**: 1.15M 参数，比 DS-MIL(1.20M) 略小，远小于 TransMIL(2.67M)、MHIM-MIL(2.67M)、ILRA-MIL(3.68M)、DGR-MIL(4.35M)。排除了"更大容量"的替代解释。

| Method | BRACS Acc/AUC | PANDA Acc/AUC | NSCLC Acc/AUC | BRCA Acc/AUC |
|--------|--------------|--------------|--------------|-------------|
| AB-MIL | 0.7275/0.8806 | 0.7322/0.9306 | 0.8988/0.9569 | 0.9414/0.9727 |
| CLAM-SB | 0.7371/0.8840 | 0.7318/0.9215 | 0.8836/0.9635 | 0.9314/0.9814 |
| DS-MIL | 0.6460/0.8054 | 0.7394/0.9309 | 0.8836/0.9579 | 0.9409/0.9777 |
| TransMIL | 0.6506/0.8450 | 0.7090/0.9288 | 0.8933/0.9692 | 0.9409/0.9787 |
| MHIM-MIL | 0.6690/0.8340 | 0.6970/0.9155 | 0.8908/**0.9759** | 0.9465/0.9769 |
| **ResTopoMIL** | **0.7494/0.9006** | **0.7546/0.9426** | **0.9157**/0.9753 | **0.9568/0.9838** |

---

## 5.4 WSI Survival Prediction (Table 3)

> 💡 **Hao 批注 - 生存预测**: 5/5 数据集 C-index 全最优。LUAD 和 STAD 上优势最明显——这与生长模式和局部组织架构是预后因子一致。但需要注意 C-index 绝对值普遍不高(KIRC ~0.73, KIRP ~0.82, LUAD ~0.65)，反映了生存预测任务的固有难度。

| Method | KIRC | KIRP | LUAD | STAD | UCEC |
|--------|------|------|------|------|------|
| AB-MIL | 0.5694 | 0.7091 | 0.5942 | 0.5871 | 0.6220 |
| 2DMambaMIL | 0.7311 | 0.8027 | 0.6290 | 0.6515 | 0.7020 |
| **ResTopoMIL** | **0.7313** | **0.8182** | **0.6457** | **0.6807** | **0.7058** |

---

## 5.5 Mechanistic and Ablation Evidence

> 💡 **Hao 批注 - 梯度动态 (Figure 4)**: 这是证明"优化偏置"而非"架构不足"的关键证据。(a) 联合训练：AUC 早期上升后饱和，拓扑梯度衰减。(b) 分阶段训练：冻结统计流后拓扑梯度反弹，AUC 二次上升。(c) 移除 shuffle loss 后梯度再次崩溃。图(c)的对比尤其有力——同一个 GCN，有没有 L_texture 导致完全不同的梯度轨迹。

> 💡 **Hao 批注 - 核心消融 (Table 4)**: (1) Stat. Only 是强 baseline(PANDA 0.9027)，说明组合信号确实很强；(2) Topo. Only 0.9215 低于残差模型——因为图分支不应独立解决整个任务；(3) Joint Opt. 0.9299 < ResTopoMIL 0.9426——联合训练的梯度竞争确实损害了性能；(4) Multi-LR 0.9352——单纯增大图学习率有帮助但不够；(5) 移除 L_texture 0.9147——shuffle loss 对 PANDA 是最关键组件之一。

![](../images/1339a9d2107d26657a5f462be05a5d5dcd3066d5dbed69de28bb7088caa1a4f3.jpg)
![](../images/15e6f5ee34acc609670f7551bcd230676096eeabefc98fd76115796ebd660723.jpg)
![](../images/7d84ae2679742afff2044cc3af9491bc36d24abfb8d68fc2a69537a7a78d156f.jpg)
![](../images/430d38dd19a988f541bc0fdf6f04829001f1d594dc5d22a8f508d2cbd63db224.jpg)

**Core Strategy Ablation (Table 4):**

| Variant | PANDA AUC | BRCA AUC |
|---------|----------|----------|
| ResTopoMIL | **0.9426** | 0.9838 |
| Stat. Only | 0.9027 | 0.9486 |
| Topo. Only | 0.9215 | 0.9608 |
| Joint Opt. | 0.9299 | 0.9773 |
| Multi-LR | 0.9352 | 0.9786 |
| w/o L_tex | 0.9147 | 0.9762 |

**Architecture & Validity (Table 5):**

| Variant | PANDA AUC | BRCA AUC |
|---------|----------|----------|
| ResTopoMIL (GCN) | 0.9426 | 0.9838 |
| GAT | 0.9419 | **0.9889** |
| Random Graph | 0.8286 | 0.8563 |
| Fixed Proto. | 0.9349 | 0.9746 |

> 💡 **Hao 批注 - 架构消融**: GAT 在 TCGA-BRCA 上略高(0.9889 vs 0.9838)但在 PANDA 上持平——说明不是 graph operator 复杂度的贡献。Random Graph 大幅下降(PANDA 0.8286)，证明图结构而非图容量是关键。Fixed Proto. 比 learnable 略差，说明原型学习也有贡献但非主导。

**Extended Ablation (Appendix E):**

> 💡 **Hao 批注 - 软干预 vs 硬干预**: Table 9-12 的完整消融形成了一个有说服力的叙述——所有"软化"梯度竞争的 tricks (dropout, curriculum, hard mining, Multi-LR) 都有帮助但都不如 stop-gradient 解耦彻底。这支持了核心论点：空间盲是优化问题，需要训练策略层面的干预而非调参层面的修补。

- **Optimization (Table 9)**: Stat-Dropout (+PDL), Curriculum Scheduling, Hard Instance Mining, Multi-LR all improve over vanilla Joint Opt. but none match ResTopoMIL.
- **Fusion (Table 10)**: Gated Fusion underperforms (early adaptive mixing reintroduces shortcut); MoE strong on TCGA-BRCA but weaker on PANDA.
- **Hyperparameter Sensitivity (Table 12)**: K_proto (8-64), K_knn (4-16), margin m (0.1-0.5) all show broad stability. Removing L_tex is the strongest negative control.

---

## 5.6 Additional Analyses (Appendix G-K)

### Progressive Coordinate-Shuffling (Appendix H, Figure 5)

> 💡 **Hao 批注 - 渐进打乱**: 这是比单点 endpoint 更强的证据——AUC 随坐标置换比例单调下降，说明模型响应跟踪空间损伤的程度。如果模型只是"偶然"使用坐标，不会产生这种单调趋势。

![](../images/1ebb6eaff760ec1b6856f2529b7f9dbc318ee3ff58572c485749ff9f71599477.jpg)
![](../images/f62d0f6d8172129ece49d183d5d1b81a4a0bcfcdc5f3444b4a66519e124ba90f.jpg)
![](../images/bbbc579708b5902c695ef10dfca91c20a873133647ca1dfa1e9ca4780dec8ce2.jpg)
![](../images/aa5753d6be8d46ad8aa1eda5c9eee42eab359889213513e7202294f3f37016bc.jpg)

### Shuffle Endpoint Sensitivity (Table 15)

> 💡 **Hao 批注 - 打乱敏感性端点**: AB-MIL 不变(排列不变设计)，DS-MIL 完全不变(!)，TransMIL 仅有微弱下降。ResTopoMIL 在 BRACS 上下降 0.0656 AUC，PANDA 下降 0.0374——这些正是结构依赖最强的任务。高 AUC 不等于使用拓扑。

| Method | BRACS Orig->Shuff | PANDA Orig->Shuff | NSCLC Orig->Shuff |
|--------|-------------------|-------------------|-------------------|
| AB-MIL | 0.8806->0.8810 (+0.0004) | 0.9306->0.9302 (-0.0004) | 0.9569->0.9571 (+0.0002) |
| TransMIL | 0.8450->0.8425 (-0.0025) | 0.9288->0.9270 (-0.0018) | 0.9692->0.9675 (-0.0017) |
| DS-MIL | 0.8054->0.8054 (0.0000) | 0.9329->0.9329 (0.0000) | 0.9579->0.9579 (0.0000) |
| **ResTopoMIL** | **0.9006->0.8350 (-0.0656)** | **0.9426->0.9052 (-0.0374)** | **0.9753->0.9610 (-0.0143)** |

### CAMELYON-16 Localization (Appendix I, Table 16)

> 💡 **Hao 批注 - 定位评估**: ResTopoMIL Dice=0.624 显著超过最佳 baseline MHIM-MIL(0.548)，FROC=0.5483 也最高。值得注意的是 TransMIL：specificity 近乎完美(0.999)但 Dice=0.103——说明模型可以抑制假阳性但无法把空间证据分配到正确边界。这与优化偏置解释一致。

| Method | Dice | Specificity | FROC |
|--------|------|-------------|------|
| AB-MIL | 0.412 | 0.985 | 0.3952 |
| TransMIL | 0.103 | 0.999 | 0.4866 |
| DTFD-MIL | 0.525 | 0.999 | 0.4712 |
| MHIM-MIL | 0.548 | 0.992 | 0.4815 |
| **ResTopoMIL** | **0.624** | **0.999** | **0.5483** |

### Feature-Space Visualization (Appendix J, Figures 6-7)

> 💡 **Hao 批注 - 特征空间可视化**: PCA 和 t-SNE 在两个投影维度上都显示统计流特征高度混叠(负 silhouette, 高 DB index)，而拓扑流特征形成紧凑的分类簇(silhouette ~0.87-0.93, DB ~0.10-0.18)。星号标记的样本(被统计流误分类但被拓扑流纠正)从混叠区域移动到正确类别簇——直接可视化了残差修正的效果。

### Cross-Backbone Generalization (Appendix F, Tables 13-14)

> 💡 **Hao 批注 - CTransPath 结果**: ResTopoMIL 在 CTransPath 特征上也保持最优/接近最优，排除了 UNI encoder 依赖。BRACS F1 从 0.6525(best baseline) 提升到 0.6952，验证了拓扑分支捕捉的是与 patch encoder 互补的信息。
