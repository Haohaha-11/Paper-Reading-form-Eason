[← 返回 README](../README.md)

# Results 结果

## 📌 预览

结果分七块：数据与设计（Fig.1/8）→ biomarker 间显著互斥/共现且跨数据集变化（Fig.2）→ 模型确实训练良好（AUROC>0.8，Fig.3）→ **但按共依赖 biomarker 分层后 AUROC 大跌（Fig.4）** → **按 grade 分层也大跌（Fig.5）** → grade 本身就能预测很多 biomarker、ML 增量有限（Fig.6）→ **按 TMB 分层同样大跌（Fig.7）**。三条混杂路径全部坐实。

---

## Data and study design

We analysed the limitations of existing ML approaches for predicting molecular biomarkers from H&E stained WSIs. We hypothesize that interdependencies among biomarker statuses and clinicopathological variables in the training data bias ML models towards relying on aggregated influences of multiple factors in WSIs rather than patterns linked to individual biomarkers. We retrospectively analysed n = 8,221 patients with breast cancer (BRCA), colorectal cancer (CRC), endometrial cancer (UCEC) and lung cancer across four cohorts: TCGA (n = 2,683), METABRIC (n = 2,433), MSK (n = 2,486) and DFCI (n = 619). We performed four major steps: (1) interdependency analysis among biomarkers; (2) training deep learning models to predict biomarker status; (3) stratification analysis and permutation testing for bias; (4) analysis of added value of ML over pathologist-assigned grade.

To assess whether biomarker interdependencies introduce bias, we analysed three deep learning algorithms with different principles: attention-based (CLAM), graph neural network-based (SlideGraph∞) and a WSI-level multimodal foundation model (TITAN). CLAM and SlideGraph∞ were trained with two encoders: CTransPath (histology) and ShuffleNet (ImageNet). We evaluated on independent cohorts CPTAC and ABCTB.

![Fig 1](../images/16c8e23e2d1fcb271885c705fbbc4d2483a1c16cbd58e4d123b6c365638360ee.jpg)

*Fig. 1: ML 从 WSI 推断分子 biomarker 的概念框架。(a) 用已知 biomarker 状态的 WSI 训练，模型从 WSI 表征 X 预测 biomarker Y。(b) 理想预测器的输出 Z 应只依赖 Y 的组织学效应、独立于混杂因子 C；若 Z 还依赖 C（grade/TMB/其他 biomarker），则预测被混杂。*

> 💡 **Figure 1 批读**（因果图立框架）（Hao 批注）：(b) 的简化因果图是全文的理论骨架——**理想**：Y（目标 biomarker）→ 组织形态 → Z（预测）；**现实**：混杂 C（grade/TMB/共依赖 biomarker）也影响形态且与 Y 相关，模型走了 Z←C 这条捷径。分层分析就是"固定 C、看 Z 与 Y 的关系是否还成立"。三个被测模型（CLAM 注意力 / SlideGraph 图 / TITAN 基础模型）刻意覆盖不同范式 + 两种 encoder，就是为证明**混杂是范式无关的通病**，不是某个模型的锅。

## Biomarker statuses show significant interdependencies and variations

Our analysis revealed significant interdependencies (P ≪ 0.05) among biomarkers across cancer types. In BRCA, elevated ER and PR co-occur with mutations in CDH1, MAP3K1 and PIK3CA, but not with TP53. In CRC, MSI-high cases frequently carry BRAF, ATM, ARID1A and RNF43 mutations and are less likely to harbour KRAS mutations. Our analysis further showed that, within the same tissue type, biomarker associations can vary across datasets: in TCGA-BRCA, MAP3K1 mutations showed mutual exclusivity with AKT1 and ARID1A, whereas in METABRIC they showed co-occurrence. ER status and high TMB showed mild co-occurrence in TCGA-BRCA but mutual exclusivity in METABRIC.

![Fig 2a](../images/2c5137858ce3eb73b09182d73a34b75eddc267711d547e3b576fc67eb5527a00.jpg)
![Fig 2b](../images/0a9a0fd84cf33b203b2b8bcd54eee28df756c0c0eff73ba2c2e940938061c125.jpg)

*Fig. 2: biomarker 与基因突变状态在不同组织/数据集下的关联热图（红=共现，蓝=互斥，星号=显著）。同一组织类型的关联在不同数据集间会变（采样差异）。*

> 💡 **Figure 2 批读**（第一根证据链：依赖存在且会漂移）（Hao 批注）：两个发现都关键——(1) biomarker 间**显著**互斥/共现（有生物学根源，也有 spurious）；(2) 同组织的关联**跨数据集变化**（TCGA vs METABRIC 相反）。含义：模型在 TCGA 上学到的"复合表型"到了 METABRIC 就对不上 → 这就是"依赖结构漂移致 AUROC 虚高/虚低"的实证基础。这也警示压缩/迁移研究：**跨中心的关联结构本身在变**，单中心验证的"保真"结论可能不迁移。

## Prediction of biomarkers and gene alterations from WSIs

Different model configurations achieved AUROCs >0.80 for multiple biomarkers. In BRCA, CLAM with CTransPath predicts ER with AUROCs 0.87 (TCGA-BRCA CV) and 0.90 (ABCTB), CDH1 and TP53 mutations with 0.88 and 0.82. SlideGraph∞ (CTransPath) predicted MSI in CRC with AUROC 0.89 (TCGA-CRC) and 0.84 (CPTAC-CRC). TITAN single-output and multi-output models showed similar performance. These results confirm the proper training of these models.

![Fig 3](../images/95cc6401aca9756442d797f4c36e576cfad39e1c6126c79e1650d226422dc463.jpg)

*Fig. 3: 弱监督模型（CLAM、SlideGraph∞，各配 ShuffleNet/CTransPath 两 encoder）在各癌种预测 biomarker/突变的 AUROC。*

> 💡 **Figure 3 批读**（先证明"模型没训坏"）（Hao 批注）：这步很聪明——作者**先证明这些模型是达到 SOTA 水平的**（多 biomarker AUROC>0.8），排除"结论是因为模型训得差"的反驳。只有在"模型确实好"的前提下，后面分层暴露的崩塌才有说服力。这是批判性论文的严谨之处：先接受对方最强的结果，再拆解它。

## Interdependence leads to entangled histology phenotypes

Our confounding analysis reveals WSI-based predictors are strongly influenced by biomarker interdependencies. SlideGraph predicts CRC MSI status with AUROC 0.88; but when divided into hypermutated/non-hypermutated subgroups, the AUROC drops to 0.72 within each. In breast tumours, the ER predictor substantially declines in cases with GATA3, CDH1 and PIK3CA mutations. The AUROC of the ER predictor drops from 0.89 to 0.57 (single-output) / 0.88 to 0.58 (multi-output).

![Fig 4](../images/04e257e977dd91749149060efac89b13e37f1e5a1190922f872ce48b1388a85a.jpg)

*Fig. 4: 多个 WSI biomarker 预测器关于其他共依赖 biomarker 的分层分析。violin=全队列 100 次 bootstrap，doughnut=各分层组 AUROC，星号=显著。*

> 💡 **Figure 4 消融解读**（第二根证据链：共依赖 biomarker 混杂）（Hao 批注）：**最震撼的数字**——ER 预测器整体 AUROC 0.89，按共依赖 biomarker 分层后跌到 0.57（≈随机）。MSI 从 0.88 跌到 0.72。含义：模型的"高 AUROC"很大程度来自"猜对了共现的其他 biomarker"，一旦固定那个 biomarker（组内），ER 特异信号所剩无几。**这直接证伪了"模型学到了 ER 的形态"**——它学的是复合表型。分层内 AUROC≈0.57 意味着几乎没有 ER 特异判别力。

## WSI-based biomarker prediction is confounded by histology grade

Stratification by tumour grade reveals marked subgroup-level drops. The ER predictor AUROC drops to 0.76 for medium-grade cases; PR predictor drops to 0.59/0.69 for low/medium-grade. TP53 predictor drops from 0.81 to ~0.72-0.73 across grades. In high-grade UCEC, the TP53 predictor attains AUROC 0.70 in TCGA but only 0.36 in CPTAC — consistent with a shift in TP53-grade relationship from co-occurrence (train) to mutual exclusivity (test). Grade-specific models (trained per grade) attained lower AUROCs than the pooled model (TP53: ~0.73 vs 0.84 pooled).

![Fig 5](../images/bdf9c94e6f48d5a03d922f4e74eac8f6e69b3d071535ff6a21083076aa470312.jpg)

*Fig. 5: WSI biomarker 预测器在不同组织学 grade 分层下的偏差。violin=全队列，doughnut=各 grade 组 AUROC。*

> 💡 **Figure 5 消融解读**（第三根证据链：grade 混杂 + Simpson 悖论）（Hao 批注）：**UCEC TP53 的例子最致命**——TCGA 高 grade 组 AUROC 0.70，CPTAC 高 grade 组只有 **0.36（远低于随机！）**。原因：TP53-grade 关联在两个 cohort 里**方向相反**（TCGA 共现、CPTAC 互斥）。模型学的是 grade 形态当 TP53 的 proxy，关联一反就系统性错。而且"分 grade 单独训"比"合并训"更差 → 证明合并训的高分靠 grade 借力。**这是 Simpson 悖论的教科书级案例**：整体看很好，分层看崩甚至反向。

## The added predictive power beyond pathologist grade

Several biomarkers can be inferred with accuracy higher than expected from pathologist-assigned grade. Grade-based ER and PR classifiers achieved AUROCs 0.76 and 0.70 (TCGA-BRCA), 0.79 and 0.71 (ABCTB). Grade predicts TP53 with AUROC 0.75, nearly matching the 0.81 achieved by weakly supervised ML models.

![Fig 6](../images/15374da09c86d221846bcd4495d89a9b4e14acc92d88b5a3bad8d0de0f43aa3b.jpg)

*Fig. 6: 仅用病理 grade（one-hot + SVM）预测 biomarker/突变的 AUROC。*

> 💡 **Figure 6 批读**（"ML 到底比 grade 强多少？"）（Hao 批注）：这是给临床读者的暴击——**一个只用 grade 的 SVM 就能把 TP53 预测到 0.75，而复杂 ML 只有 0.81**。ER/PR 类似。含义：对某些 biomarker，昂贵的 WSI 深度模型相对"病理医生已能从 grade 读出的信息"**增量很小**。这直接质疑"用 ML 替代分子检测省钱"的商业叙事——很多"预测力"本就藏在 grade 里。

## WSI-based biomarker prediction is confounded by mutation density (TMB)

WSI models infer BRAF and TP53 in CRC with AUROCs 0.774 and 0.717. But for cases with low mutation density in genes other than BRAF (TMB_BRAF-bar), the BRAF predictor drops to 0.65; the TP53 predictor drops to 0.50 for high TMB cases. PTEN predictor in UCEC drops to 0.63 (TCGA) and 0.32 (CPTAC) for low TMB cases. KRAS predictor: AUROC 0.83 (CPTAC high TMB) vs 0.63 (TCGA high TMB), tracking the TMB-KRAS association shift.

![Fig 7](../images/234102fdb1c8411600065e6c86e21ab02ba331559c14c321fe3dc541f9072c2d.jpg)

*Fig. 7: WSI biomarker 预测器在不同 TMB 分层下的偏差（a）；grade/TMB 与突变关联结构的跨数据集漂移（b）。*

> 💡 **Figure 7 消融解读**（第三条混杂路径坐实 + 总结）（Hao 批注）：TMB 分层同样暴露崩塌——TP53 在高 TMB 组掉到 0.50（随机）。含义：模型部分在"数突变多不多"（TMB 的整体形态，如核异型/增殖）而非 TP53 特异信号。三条混杂路径（共依赖 biomarker / grade / TMB）**全部被同一套分层+permutation 协议坐实**，且在 CLAM/SlideGraph/TITAN 三范式 + 两 encoder 上一致 → 这是**范式无关的系统性问题**，不是个别模型缺陷。
