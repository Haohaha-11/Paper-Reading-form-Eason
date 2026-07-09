[← 返回 README](../README.md)

# 04 - Experiments

## 原文 Section: EXPERIMENTS

### Datasets and Evaluation Metrics

To validate the efficacy of our proposed CKMIL, we conducted extensive experiments on two representative downstream tasks across four public datasets.

**Survival Prediction** We selected three public cancer datasets from The Cancer Genome Atlas (TCGA) (Weinstein et al. 2013): BLCA, BRCA, and LUAD, which contain Whole Slide Images (WSIs) with corresponding survival time annotations. Following the experimental setup of GPFM (Ma et al. 2024), we employ a 5-fold cross-validation methodology to mitigate the impact of data partitioning on model evaluation, splitting the data into training and validation sets at a 4:1 ratio. We utilize the cross-validated Concordance Index (C-Index), with its standard deviation (std).

**Cancer Subtyping** We also conduct experiments on three challenging public datasets: BRACS (Brancati et al. 2022), and the NSCLC and BRCA cohorts from the TCGA database (Weinstein et al. 2013). For dataset partitioning, we follow the protocol from GPFM (Ma et al. 2024), splitting the data into training, validation, and testing sets at a 7:1:2 ratio. To ensure a robust evaluation, we generate 5 different random splits with this ratio for our experiments. For evaluation, we adopt the Area Under the Curve (AUC) and Accuracy (ACC) metrics, reporting their mean and standard deviation (std). Supplementary Material offers more details.

> **Hao 批注, 实验设置解读**: 两套评估协议：
> - **Survival Prediction**: 5-fold CV (4:1 split/train:val)，指标为 C-Index。C-Index 衡量的是预测风险顺序与实际生存时间排序的一致性（0.5 = 随机，1.0 = 完美），是生存分析的标准指标。
> - **Cancer Subtyping**: 5 random 7:1:2 splits (train:val:test)，指标为 AUC 和 ACC。多分类任务（BRACS-3 是三分类，BRCA-2/NSCLC-2 是二分类）。
>
> 两个协议都使用 5 次重复来报告 mean ± std，保证了统计可靠性。

---

### Comparison Methods and Training Details

**Comparison Methods** We compare our proposed methods CKMIL-base and CKMIL, against several categories of methods: (1) Simple Pooling Methods: Mean-Pooling and Max-Pooling; (2) Attention-Based Methods: ABMIL (Ilse, Tomczak, and Welling 2018), and its variants CLAM-MB (Lu et al. 2021) and DSMIL (Li, Li, and Eliceiri 2021); (3) Global Interaction Methods with Linear Complexity: TransMIL (Shao et al. 2021), MambaMIL (Yang, Wang, and Chen 2024), and RRTMIL (Tang et al. 2024).

**Comparison of CKMIL-Base and CKMIL** The distinction between our two models lies solely in the Q/K vector generation: CKMIL-Base uses conventional linear layers, while CKMIL incorporates our exploratory ICP module.

> **Hao 批注, 实验设计**: CKMIL-Base vs CKMIL 的设计很聪明——两者唯一差异是 ICP 模块，这为 ICP 的消融提供了自然的 baseline。Table 1-2 中两个变体的对比直接回答了"ICP 有没有用"这个问题。

**Training Details** Patches of size 256$\times$256 were cropped at 20$\times$ magnification WSIs without overlap. To extract patch features, we utilized two offline encoders: a ResNet50 (He et al. 2016) pre-trained on ImageNet (Deng et al. 2009) for general visual representations, and the UNI (Chen et al. 2024) model, which was self-supervised on a pancancer cohort to learn domain-specific pathology features. Supplementary Material offers more training details.

> **Hao 批注, 训练细节**: 关键实验设置：
> - Patch 大小: 256x256 at 20x（标准设置，pathology 文献中常见）
> - 两种特征提取器形成了"弱特征 vs 强特征"的对比——ResNet50 代表通用弱特征（对 pathology 没有领域适应），UNI 代表医学强特征（在大规模 pan-cancer 数据上自监督预训练）
> - 这一对比是论文实验设计的一大亮点——它验证了"好的聚合器可以让弱特征超越强特征+弱聚合器的组合"

---

### Results and Analysis

Tables 1 and 2 provide a comprehensive performance comparison of various MIL methods on cancer subtyping and survival prediction, utilizing two distinct feature extractors: the general-purpose ResNet50 (He et al. 2016) and the domain-specific UNI (Chen et al. 2024).

> **Hao 批注, 实验解读框架**: 主实验结果需要从两个维度解读：(1) 特征提取器维度——ResNet50 vs UNI；(2) 方法类别维度——简单池化 vs 独立注意力 vs 全局交互 vs CKMIL。最强 baseline 是 RRTMIL（ResNet50）和 ABMIL/CLAM（UNI）。

---

### Table 1: Cancer Subtyping 主实验结果

Methods | BRACS-3 (AUC) | BRACS-3 (ACC) | BRCA-2 (AUC) | BRCA-2 (ACC) | NSCLC-2 (AUC) | NSCLC-2 (ACC)

**ResNet-50**

Mean-Pooling | 0.8051$\pm$0.0319 | 0.6444$\pm$0.0337 | 0.9068$\pm$0.0276 | 0.8410$\pm$0.0262 | 0.8914$\pm$0.0203 | 0.8209$\pm$0.0282

Max-Pooling | 0.8064$\pm$0.0359 | 0.6907$\pm$0.0356 | 0.8372$\pm$0.0239 | 0.8152$\pm$0.0260 | 0.9163$\pm$0.0314 | 0.8342$\pm$0.0340

ABMIL (Ilse, Tomczak, and Welling 2018) | 0.8004$\pm$0.0382 | 0.6981$\pm$0.0368 | 0.8883$\pm$0.0190 | 0.8139$\pm$0.0401 | 0.9359$\pm$0.0276 | 0.8685$\pm$0.0463

CLAM-MB (Lu et al. 2021) | 0.8134$\pm$0.0287 | 0.6833$\pm$0.0280 | 0.8929$\pm$0.0177 | 0.8210$\pm$0.0232 | 0.9407$\pm$0.0207 | 0.8685$\pm$0.0266

DSMIL (Li, Li, and Eliceiri 2021) | 0.7950$\pm$0.0365 | 0.6481$\pm$0.0476 | 0.8196$\pm$0.0766 | 0.7809$\pm$0.0540 | 0.8491$\pm$0.0779 | 0.7561$\pm$0.0701

TransMIL (Shao et al. 2021) | 0.8160$\pm$0.0406 | 0.7111$\pm$0.0200 | 0.8774$\pm$0.0386 | 0.8145$\pm$0.0445 | 0.9348$\pm$0.0192 | 0.8495$\pm$0.0415

MambaMIL (Yang, Wang, and Chen 2024) | 0.8305$\pm$0.0427 | 0.7111$\pm$0.0553 | 0.8949$\pm$0.0375 | 0.8632$\pm$0.0273 | 0.9374$\pm$0.0190 | 0.8743$\pm$0.0302

RRTMIL (Tang et al. 2024) | 0.8160$\pm$0.0257 | 0.7129$\pm$0.0185 | 0.9163$\pm$0.0290 | 0.8484$\pm$0.0386 | 0.9421$\pm$0.0146 | 0.8723$\pm$0.0136

**CKMIL-Base (ours)** | 0.8483$\pm$0.0260 | 0.7130$\pm$0.0515 | 0.9269$\pm$0.0358 | 0.8716$\pm$0.0274 | 0.9439$\pm$0.0225 | 0.8752$\pm$0.0317

**CKMIL (ours)** | **0.8583$\pm$0.0297** | **0.7370$\pm$0.0427** | 0.9255$\pm$0.0261 | 0.8648$\pm$0.0252 | **0.9549$\pm$0.0148** | **0.8838$\pm$0.0253**

**UNI**

Mean-Pooling | 0.8771$\pm$0.0259 | 0.7203$\pm$0.0411 | 0.9552$\pm$0.0258 | 0.8943$\pm$0.0237 | 0.9746$\pm$0.0122 | 0.9257$\pm$0.0219

Max-Pooling | 0.8596$\pm$0.0285 | 0.7503$\pm$0.0101 | 0.9627$\pm$0.0190 | 0.9136$\pm$0.0109 | 0.9816$\pm$0.0109 | 0.9361$\pm$0.0246

ABMIL (Ilse, Tomczak, and Welling 2018) | 0.8901$\pm$0.0426 | 0.7635$\pm$0.0567 | **0.9671$\pm$0.0240** | **0.9187$\pm$0.0106** | 0.9796$\pm$0.0118 | 0.9485$\pm$0.0197

CLAM-MB (Lu et al. 2021) | 0.8862$\pm$0.0343 | 0.7629$\pm$0.0456 | 0.9625$\pm$0.0176 | 0.9291$\pm$0.0186 | 0.9825$\pm$0.0117 | 0.9409$\pm$0.0183

DSMIL (Li, Li, and Eliceiri 2021) | 0.8399$\pm$0.0169 | 0.7185$\pm$0.0266 | 0.9533$\pm$0.0124 | 0.8900$\pm$0.0129 | 0.9739$\pm$0.0129 | 0.9200$\pm$0.0278

TransMIL (Shao et al. 2021) | 0.8549$\pm$0.0226 | 0.7407$\pm$0.0340 | 0.9488$\pm$0.0293 | 0.9195$\pm$0.0172 | 0.9766$\pm$0.0124 | 0.9190$\pm$0.0187

MambaMIL (Yang, Wang, and Chen 2024) | 0.8842$\pm$0.0234 | 0.7645$\pm$0.0292 | 0.9568$\pm$0.0234 | 0.9099$\pm$0.0221 | 0.9791$\pm$0.0120 | 0.9352$\pm$0.0204

RRTMIL (Tang et al. 2024) | 0.8754$\pm$0.0284 | 0.7574$\pm$0.0583 | 0.9586$\pm$0.0221 | 0.9178$\pm$0.0153 | 0.9818$\pm$0.0115 | 0.9323$\pm$0.0182

**CKMIL-Base (ours)** | 0.8967$\pm$0.0275 | 0.7648$\pm$0.0274 | 0.9579$\pm$0.0192 | 0.9160$\pm$0.0253 | 0.9756$\pm$0.0086 | 0.9342$\pm$0.0169

**CKMIL (ours)** | 0.8952$\pm$0.0203 | 0.7648$\pm$0.0258 | 0.9556$\pm$0.0208 | 0.9125$\pm$0.0274 | 0.9836$\pm$0.0103 | 0.9361$\pm$0.0234

**Table 1**: Performance comparison on cancer subtyping tasks. Best results are in **bold**, and second-best results are underlined.

> **Hao 批注, Table 1 批读**: Table 1 提供了最多信息的主结果：
>
> **ResNet50 特征 — CKMIL 全面 SOTA**:
> - BRACS-3: CKMIL AUC 0.8583 vs RRTMIL 0.8160 (+4.23% relative)，ACC 0.7370 vs RRTMIL 0.7129 (+3.38%)
> - NSCLC-2: CKMIL AUC 0.9549 vs RRTMIL 0.9421 (+1.36%)，ACC 0.8838 vs MambaMIL 0.8743 (+1.09%)
> - BRCA-2: CKMIL-Base AUC 0.9269 最佳（CKMIL 0.9255 微低），ACC 0.8716 (CKMIL-Base)
>
> **UNI 特征 — 竞争但非全面最优**:
> - BRACS-3: CKMIL-Base AUC 0.8967 最佳，但 ACC 0.7648 与 MambaMIL 0.7645 持平
> - BRCA-2: ABMIL/CLAM AUC (0.9671/0.9625) 和 ACC (0.9187/0.9291) 反超 CKMIL
> - NSCLC-2: CKMIL AUC 0.9836 最佳，但 ACC 0.9361 略低于 Max-Pooling 0.9361（持平）
>
> **关键发现**:
> 1. ResNet50 下 CKMIL 的增益远超 UNI 下——说明聚合器质量对弱特征更关键
> 2. UNI 下 ABMIL/CLAM 在 BRCA 反超 CKMIL——"强特征 + 简单注意力"可能优于"强特征 + 复杂交互"
> 3. CKMIL-Base 和 CKMIL 差距不大，且方向不一——ICP 的效果不稳定

---

### Table 2: Survival Prediction 主实验结果

Methods | BLCA (C-index) ResNet-50 | BLCA UNI | BRCA ResNet-50 | BRCA UNI | LUAD ResNet-50 | LUAD UNI

Mean-Pooling | 0.5870$\pm$0.0583 | 0.5989$\pm$0.0129 | 0.6135$\pm$0.0631 | 0.6777$\pm$0.0602 | 0.6095$\pm$0.0820 | 0.6276$\pm$0.0623

Max-Pooling | 0.5589$\pm$0.0593 | 0.5742$\pm$0.0476 | 0.5754$\pm$0.0382 | 0.6119$\pm$0.0522 | 0.6063$\pm$0.0396 | 0.5951$\pm$0.0069

ABMIL (Ilse, Tomczak, and Welling 2018) | 0.5503$\pm$0.0986 | 0.6035$\pm$0.0491 | 0.6103$\pm$0.0739 | 0.6688$\pm$0.0534 | 0.6015$\pm$0.0767 | 0.6240$\pm$0.0762

CLAM-MB (Lu et al. 2021) | 0.5695$\pm$0.0951 | 0.5975$\pm$0.0445 | 0.5887$\pm$0.0592 | 0.6701$\pm$0.0413 | 0.6165$\pm$0.0761 | 0.6265$\pm$0.0490

DSMIL (Li, Li, and Eliceiri 2021) | 0.5774$\pm$0.0588 | 0.5885$\pm$0.0536 | 0.6199$\pm$0.0297 | 0.6460$\pm$0.0346 | 0.6147$\pm$0.0250 | 0.5496$\pm$0.0594

TransMIL (Shao et al. 2021) | 0.6055$\pm$0.0485 | 0.6119$\pm$0.0312 | 0.6158$\pm$0.0559 | 0.6163$\pm$0.0360 | 0.6335$\pm$0.0347 | 0.6222$\pm$0.0615

MambaMIL (Yang, Wang, and Chen 2024) | OOM | OOM | 0.6524$\pm$0.0494 | 0.6480$\pm$0.0399 | 0.6452$\pm$0.0168 | 0.6142$\pm$0.0580

RRTMIL (Tang et al. 2024) | OOM | OOM | 0.6445$\pm$0.0604 | 0.6500$\pm$0.0503 | 0.6231$\pm$0.0490 | 0.6303$\pm$0.0687

**CKMIL-Base (ours)** | 0.6287$\pm$0.0429 | 0.6038$\pm$0.0349 | 0.6440$\pm$0.0794 | **0.6920$\pm$0.0717** | **0.6820$\pm$0.0267** | 0.6300$\pm$0.0267

**CKMIL (ours)** | 0.6185$\pm$0.0406 | 0.6155$\pm$0.0429 | **0.6825$\pm$0.0887** | 0.6869$\pm$0.0661 | 0.6467$\pm$0.0402 | 0.6380$\pm$0.0640

**Table 2**: Performance comparison on survival prediction tasks. Best results are in **bold**, and second-best results are underlined. OOM denotes out of memory in the experiment settings.

> **Hao 批注, Table 2 批读**: Table 2 的生存预测结果揭示了一些与 subtyping 一致和不一致的模式：
>
> **ResNet50 — CKMIL SOTA**:
> - BLCA: CKMIL-Base 0.6287 (SOTA)，CKMIL 0.6185 (second)。注意只有 CKMIL-Base 和 TransMIL 在 BLCA 上没有 OOM。
> - BRCA: CKMIL 0.6825 (SOTA)，比 MambaMIL 0.6524 高 4.61% relative。CKMIL 的提升巨大（+6.67% vs CKMIL-Base 的 0.6440，归因于 ICP！）
> - LUAD: CKMIL-Base 0.6820 (SOTA)，CKMIL 0.6467 反而下降——ICP 在 LUAD 上有害
>
> **UNI — 混合结果**:
> - BLCA: CKMIL 0.6155 最佳但 TransMIL 0.6119 接近
> - BRCA: CKMIL-Base 0.6920 (SOTA)，CKMIL 0.6869 微降
> - LUAD: CKMIL 0.6380 最佳，但 TransMIL 0.6222 紧随其后
>
> **OOM 问题**: MambaMIL 和 RRTMIL 在 BLCA 上 OOM——暗示 BLCA 数据集可能有特别大的 bags，使这些方法的内存消耗不可接受。CKMIL 的 $O(n)$ Nystrom attention 在内存效率上有优势。
>
> **ICP 的不稳定性**: BRCA ResNet50 上 CKMIL > CKMIL-Base (+3.85% C-Index)，但 LUAD ResNet50 上 CKMIL < CKMIL-Base (-5.18%)。这是 ICP 最大的问题——效果方向不可预测。

---

### 主结果深度分析

When benchmarked with the ResNet50 feature extractor, our CKMIL models achieve state-of-the-art (SOTA) performance across all tasks and datasets. Notably, the full CKMIL model consistently outperforms all competing methods, with CKMIL-Base being the only exception. For instance, as shown in Table 1, our CKMIL model demonstrates significant improvements on the BRACS-3 subtyping task, outperforming the strong baseline RRTMIL with a 2.78% improvement in AUC and 2.01% in ACC. This superiority extends to survival prediction tasks. On the LUAD cohort, our CKMIL-Base model sets a new SOTA with a C-Index of 0.6820. Meanwhile, on the BRCA survival task, CKMIL achieves a C-Index of 0.6825, a substantial 3.81% improvement over the next-best comparable method. This finding is particularly significant, as it validates our core hypothesis that an effective aggregation mechanism can overcome the limitations of non-domain-specific features by effectively modeling instance correlations.

> **Hao 批注**: 作者将 ResNet50 上的 SOTA 解释为证明"好的聚合器可以弥补弱特征"。这是论文最重要的 take-home message——不是"我们的方法在所有场景下都是最好的"，而是"在特征不够好时，我们的方法是最有价值的"。

When using the pathology-specific UNI feature extractor, our models achieve new SOTA results across all survival prediction tasks. However, in certain subtyping tasks, such as on the BRCA dataset (for both AUC and ACC) and the NSCLC dataset (for ACC), the performance of methods that model inter-instance correlations, including ours, was surpassed by simpler approaches like ABMIL and CLAM. We hypothesize that this phenomenon occurs because UNI generates features that are already highly discriminative. For such strong features, explicitly modeling correlations might introduce noise from redundant instances, which inadvertently dilutes the weights or the features themselves of sparse, critical instances, and thus degrades performance. Conversely, our models' SOTA performance with the generic ResNet50 extractor corroborates the effectiveness of our correlation modeling, demonstrating its ability to adapt general-purpose features for specialized medical analysis through guided interaction.

> **Hao 批注, Q&A 批注记录**:
> - **Q**: 为什么 UNI 特征下 CKMIL 反而不如 ABMIL？
> - **A**: 作者假设 UNI 特征已经高度判别，进一步建模实例间相关性反而引入冗余实例的噪声，稀释了关键实例的权重或特征。这本质上是 bias-variance tradeoff——强特征下简单模型（ABMIL）的低方差优势超越了复杂模型（CKMIL）的低偏差优势。但这个假设需要更多实验支撑（如不同 bag 大小下的对比、不同正例比例下的对比等）。

---

### Ablation Study and Sensitivity Analysis

To rigorously validate the effectiveness of our proposed CKMIL framework, we conduct a series of ablation studies on its core components: the Subspace-Disentangled Attention (SDA), the Key-Instance Guided Global Attention (KGGA), and the Instance-Conv-Projection (ICP) module. We perform quantitative evaluations on the BRACS-3 cancer subtyping task (reporting mean AUC and ACC) and the TCGA-BRCA survival prediction task (reporting mean C-Index), using ResNet50 as the feature extractor and following the same experimental protocol as in the main experiments.

> **Hao 批注, 实验设计**: 消融实验的设计非常规范——每个消融都有明确的对比对：
> - SDA: ABMIL vs ABMIL+SDA, CKMIL(m=1) vs CKMIL
> - KGGA: ABMIL vs ABMIL+KGGA, TransMIL vs TransMIL+KGGA, CKMIL(Pooling) vs CKMIL
> - ICP: CKMIL-Base vs CKMIL (已在 Table 1-2 中)
>
> 消融实验只用 ResNet50 特征——因为作者已经知道在 UNI 下效果不稳定，用 ResNet50 最能体现各模块的独立贡献。

---

### Effectiveness of Subspace-Disentangled Attention (SDA)

The SDA module is designed to screen for key instances within multiple disentangled feature subspaces. To isolate its contribution, we conduct two sets of experiments:

* CKMIL vs. CKMIL ($m = 1$): We reduce the number of subspaces in the SDA module to one (i.e., $m = 1$). This variant, denoted as CKMIL ($m = 1$) or ABMIL+KGGA, replaces SDA with a single, shared attention layer akin to ABMIL, while keeping the KGGA module.

* ABMIL (Ilse, Tomczak, and Welling 2018) vs. ABMIL+SDA: To demonstrate that the multi-subspace scoring mechanism is inherently superior to a single attention layer, we integrate the SDA module into the standard ABMIL framework, creating ABMIL+SDA.

As presented in Table 3, ABMIL+SDA consistently surpasses ABMIL across all metrics. Similarly, CKMIL outperforms the original CKMIL ($m = 1$) across all metrics, further validating that the multi-subspace scoring design is a more effective strategy than a single shared attention layer.

> **Hao 批注, 消融解读**: SDA 消融的两个实验分别验证了不同的东西：
> - ABMIL vs ABMIL+SDA: 验证了多子空间独立打分优于单注意力层（在纯独立注意力框架下）
> - CKMIL(m=1) vs CKMIL: 验证了多子空间独立打分优于单注意力层（在 KGGA 框架下）
>
> 两个实验的一致性强化了结论的可靠性——多子空间打分的优势不依赖于 KGGA 的存在。

---

### Table 3: SDA 消融实验

Model | BRACS-3 (AUC $\uparrow$) | BRACS-3 (ACC $\uparrow$) | BRCA (C-Index $\uparrow$)

ABMIL | 0.8004 | 0.6981 | 0.6103

ABMIL+SDA | 0.8423 (+4.19%) | 0.7074 (+0.93%) | 0.6131 (+0.28%)

CKMIL ($m = 1$) | 0.8454 | 0.7240 | 0.6687

CKMIL (ours) | 0.8583 (+1.29%) | 0.7370 (+1.30%) | 0.6825 (+1.38%)

**Table 3**: Ablation study on the effectiveness of SDA.

> **Hao 批注, Table 3 批读**: SDA 的贡献在两个维度上都很显著：
>
> **ABMIL → ABMIL+SDA**: AUC +4.19% (BRACS-3)。这是一个巨大的提升，说明"多子空间独立打分"本身就比"单注意力层"在独立注意力场景下有本质优势——单一注意力可能被某一种特征模式主导，多子空间则能捕获更多维度的诊断信号。
>
> **CKMIL(m=1) → CKMIL**: AUC +1.29%, ACC +1.30%, C-Index +1.38%。增益虽然不如 ABMIL+SDA 大，但所有指标都一致提升。这说明多子空间设计为 KGGA 提供了更好的 landmarks 多样性。
>
> **有趣的现象**: ABMIL+SDA 的 BRCA C-Index 仅 +0.28%——在生存预测任务上，多子空间独立打分的增益很有限。但加上 KGGA 后 (CKMIL vs m=1: +1.38%)，效应变得显著。这暗示 SDA 和 KGGA 的协同作用大于各自独立贡献。

---

### Effectiveness of Key-Instance Guided Global Attention (KGGA)

The KGGA module is premised on the principle that global interaction should be guided by key instances. We validate its efficacy through the following experiments:

* CKMIL vs. CKMIL (Pooling): We replace the key-instance-guided landmark selection in KGGA with a conventional mean pooling strategy to select landmarks, a method similar to that used in TransMIL.

* ABMIL (Ilse, Tomczak, and Welling 2018) vs. ABMIL+KGGA: To demonstrate the importance of the global interaction mechanism itself, we augment the baseline ABMIL with our KGGA module which is equivalent to the CKMIL ($m=1$) variant.

* TransMIL (Shao et al. 2021) vs. TransMIL+KGGA: To show that our key-instance-guided approach is superior, we modify TransMIL by first adding an attention layer to score instances and then using the top-scoring instances as landmarks for its global interaction. We term this variant TransMIL+KGGA.

> **Hao 批注, 机制拆解**: KGGA 消融的三个实验设计非常精巧：
>
> 1. **CKMIL(Pooling) vs CKMIL**: 控制变量——在同一框架下，只改变 landmark 选择策略（pooling vs SDA 筛选）。直接验证"关键实例引导"的价值。
>
> 2. **ABMIL vs ABMIL+KGGA**: 验证全局交互本身（KGGA）的价值。ABMIL+KGGA = CKMIL(m=1)。这是一个"最小增量"实验——在一最简单的独立注意力 baseline 上只加 KGGA。
>
> 3. **TransMIL vs TransMIL+KGGA**: 验证关键实例引导策略可以"即插即用"地改进已有方法。这不仅是消融，也是对泛化性的验证——KGGA 的思想不绑定 CKMIL 框架。

As shown in Table 4, CKMIL significantly outperforms CKMIL (Pooling), confirming our hypothesis that using candidate key instances as landmarks is more effective than using landmarks derived from pooling. The comparison between ABMIL and ABMIL+KGGA shows that incorporating our KGGA module brings substantial performance gains across all tasks, underscoring the necessity of modeling global inter-instance correlations. Finally, TransMIL+KGGA surpasses the original TransMIL, further proving that a key-instance-guided strategy is a more powerful approach for global attention in MIL.

> **Hao 批注, 消融解读**: Table 4 中的三组对比分别回答三个问题：
> - CKMIL vs CKMIL(Pooling): 关键实例作为 landmarks 是否优于 pooling？→ Yes（BRCA C-Index +3.80%）
> - ABMIL vs ABMIL+KGGA: 全局交互是否有价值？→ Yes（BRCA C-Index +5.84%！）
> - TransMIL vs TransMIL+KGGA: KGGA 思想能否提升已有方法？→ Yes（所有指标一致提升）

---

### Table 4: KGGA 消融实验

Model | BRACS-3 (AUC $\uparrow$) | BRACS-3 (ACC $\uparrow$) | BRCA (C-Index $\uparrow$)

ABMIL | 0.8004 | 0.6981 | 0.6103

ABMIL+KGGA | 0.8454 (+4.50%) | 0.7240 (+2.59%) | 0.6687 (+5.84%)

TransMIL | 0.8160 | 0.7111 | 0.6158

TransMIL+KGGA | 0.8297 (+1.37%) | 0.7278 (+1.67%) | 0.6281 (+1.23%)

CKMIL (Pooling) | 0.8477 | 0.7185 | 0.6445

CKMIL (ours) | 0.8583 (+1.06%) | 0.7370 (+1.85%) | 0.6825 (+3.80%)

**Table 4**: Ablation study on the effectiveness of KGGA.

> **Hao 批注, Table 4 批读**: KGGA 消融的核心发现：
>
> **ABMIL → ABMIL+KGGA (+5.84% C-Index)**: 这是整篇论文最令人印象深刻的单个提升。但在 subtyping 上 AUC 仅 +4.50%、ACC +2.59%。生存预测对全局上下文的依赖远大于分类——因为预后判断需要综合考虑肿瘤微环境的空间组织，而不仅仅是"有没有癌"。
>
> **TransMIL → TransMIL+KGGA (+1.23%-1.67%)**: 提升温和但一致。这证明 KGGA 的关键实例引导策略可以改造已有方法。但提升幅度小于 ABMIL+KGGA，说明 TransMIL 本身已经从 Nystrom 全局交互中获得了一定收益，大头的增益已经被占用。
>
> **CKMIL(Pooling) vs CKMIL**: 这是最有说服力的对比——唯一变量是 landmark 选择策略。CKMIL (SDA landmarks) 在所有指标上超越 CKMIL(Pooling)，尤其在 BRCA C-Index 上 +3.80%。这个对比直接支持了论文的核心 claim："关键实例引导 > 无差别 pooling"。

---

### Effectiveness of Instance-Conv-Projection (ICP)

The ICP module is designed based on the hypothesis that local correlations exist among the components of an instance's feature vector. To investigate the feasibility of this exploratory module, we conducted a comprehensive comparison between the full CKMIL model and the CKMIL-Base model which uses standard linear projections across all tasks and datasets. The detailed results, presented in Table 1 and 2, reveal that the ICP module offers clear benefits in specific contexts. For instance, when using ResNet50 features for the BRACS subtyping task, CKMIL shows a 2.4% improvement in ACC and a 1.0% improvement in AUC over CKMIL-Base. Similarly, for survival prediction on the LUAD cohort with UNI features, CKMIL yields a 0.8% improvement in C-Index. On the TCGA-BRCA survival prediction task, the benefit is even more pronounced, with CKMIL delivering a 3.85% higher C-Index than CKMIL-Base. However, on other datasets, the impact of ICP is more varied and appears to be influenced by the choice of the upstream feature extractor. This suggests that while ICP can effectively capture latent intra-feature correlations, the prominence and utility of these correlations may depend on the specific dataset and the nature of the features generated by the encoder.

> **Hao 批注, 消融解读**: ICP 的消融总结（从 Table 1-2 提取）:
>
> | 场景 | CKMIL vs CKMIL-Base | 方向 |
> |------|---------------------|------|
> | BRACS-3 ResNet50 AUC/ACC | +1.0% / +2.4% | 正 |
> | BRCA-2 ResNet50 AUC/ACC | -0.15% / -0.79% | 负 |
> | NSCLC-2 ResNet50 AUC/ACC | +1.16% / +0.98% | 正 |
> | BLCA ResNet50 C-Index | -1.62% | 负 |
> | BRCA ResNet50 C-Index | +5.98% | 强正 |
> | LUAD ResNet50 C-Index | -5.18% | 强负 |
>
> ICP 的效果呈现出 "high variance, inconsistent direction" 的特征。这强烈暗示 ICP 的收益高度依赖于数据集-特征提取器的特定交互，不具有普遍适用性。作为论文的第三个 contribution，ICP 的说服力不如 SDA 和 KGGA。

---

### Visualization Results

Fundamentally, our proposed CKMIL is an Attention-Based method. To evaluate CKMIL's interpretability and localization capability, we visualize its attention heatmaps against baseline methods ABMIL and CLAM-MB, comparing them to ground truth (GT) annotations provided by pathologists.

As shown in the global view (Figure 5), the attention from ABMIL and CLAM-MB is diffuse and highlights non-diagnostic areas, failing to localize the scattered tumor regions indicated by the GT. In contrast, CKMIL produces precise, concentrated heatmaps that show high concordance with GT annotations, successfully identifying multiple key tumor clusters. This is due to the synergy between our SDA and KGGA modules, which suppresses non-critical regions.

> **Hao 批注, Figure 5 批读**: Figure 5（全局视图）是论文最重要的可视化证据。它直接展示了 CKMIL 的核心能力——注意力精确定位在 GT 标注的肿瘤区域上，而 ABMIL/CLAM 的注意力分散在非诊断区域。这验证了 paper 的核心 narrative："关键实例引导的全局交互不仅提升性能，还提升了注意力的定位精度。"

The superiority of CKMIL is apparent in the local view (Figure 6). While baseline methods fail to focus on core pathological cell structures, CKMIL's high-attention areas precisely cover the dense, diagnostically relevant cell regions, as confirmed by GT. This demonstrates the effectiveness of our key-instance-guided mechanism in identifying the most informative regions within a WSI.

> **Hao 批注, Figure 6 批读**: Figure 6（局部放大视图）进一步验证了 CKMIL 的注意力质量——在高倍镜下，CKMIL 的高注意力区域精确覆盖了细胞密集的诊断相关区域，而 ABMIL/CLAM 则无法聚焦于核心病理结构。两张图形成了从粗到细的证据链。

---

### Figure 5: 全局注意力热力图对比

![Figure 5](../images/page5_img1.jpeg)
![Figure 5 continued](../images/page5_img2.png)
![Figure 5 continued](../images/page5_img3.jpeg)
![Figure 5 continued](../images/page5_img4.png)
![Figure 5 continued](../images/page5_img5.jpeg)

**Figure 5**: Global attention heatmap comparison on a WSI from the BRACS dataset. (GT = Ground Truth annotations)

> **Hao 批注, Figure 5 批读**: 这张图包含四列（GT, CLAM, ABMIL, CKMIL），直观对比了四种注意力的全局分布。CKMIL 的注意力集中且精确覆盖肿瘤区域，与 GT 高度吻合。

---

### Figure 6: 局部注意力热力图对比

![Figure 6](../images/page5_img6.jpeg)
![Figure 6 continued](../images/page5_img7.png)
![Figure 6 continued](../images/page5_img8.jpeg)
![Figure 6 continued](../images/page5_img9.png)
![Figure 6 continued](../images/page5_img10.jpeg)
![Figure 6 continued](../images/page5_img11.png)

**Figure 6**: Local attention heatmap comparison on a WSI from the BRACS dataset. GT annotations show positive (tumor) and negative (normal) regions.

> **Hao 批注, Figure 6 批读**: 局部视图显示了高倍率下的细胞级注意力——CKMIL 的高注意力区域精确覆盖在 dense tumor cell regions，而 ABMIL/CLAM 的高注意力区域游离在非关键结构上。这个可视化对病理学家读者最有说服力——他们能直接从图中看出 CKMIL 在"看"正确的诊断区域。

---

## 🔖 Section 总结

### 关键数字速查

| 指标 | 数值 |
|------|------|
| 实验数据集数 | 4 (BRACS, TCGA-BLCA, TCGA-BRCA, TCGA-NSCLC/LUAD) |
| 下游任务数 | 2 (cancer subtyping, survival prediction) |
| 对比方法数 | 8 (含 Mean/Max Pooling) |
| 特征提取器数 | 2 (ResNet50, UNI) |
| ABMIL+KGGA BRCA C-Index 提升 | +5.84% |
| TransMIL+KGGA 平均提升 | ~+1.4% |
| CKMIL vs CKMIL(Pooling) BRCA C-Index | +3.80% |
| SDA 消融 (ABMIL+SDA) BRACS-3 AUC 提升 | +4.19% |
| MambaMIL/RRTMIL OOM 数据集 | TCGA-BLCA |

### 核心洞察

1. **实验设计的三层对比**: (a) 不同方法在相同特征下 (b) 相同方法在不同特征下 (c) 消融实验中控制变量的模块对比。三层对比形成了完整的证据链。

2. **"弱特征 + 强聚合 > 强特征 + 弱聚合"**: ResNet50 下 CKMIL 全面 SOTA，UNI 下 ABMIL 在某些任务上反超——这恰恰证明了好的聚合器在现实场景（通用特征更易获取）中的实际价值。

3. **KGGA 是贡献最大的单一模块**: ABMIL → ABMIL+KGGA 的 +5.84% BRCA C-Index 提升是整个实验中最显著的单个改进。KGGA 的泛化能力（TransMIL+KGGA 也有提升）进一步增加了说服力。

4. **可视化强化了方法论故事**: Figure 5-6 从全局到局部展示了 CKMIL 注意力的精准性，视觉上直接支撑了"关键实例引导"的 narrative。

### 可追问点

- 为什么 Table 2 中 MambaMIL 和 RRTMIL 在 BLCA 上 OOM 而 CKMIL 没有？BLCA 的 bags 平均有多大？是否可以量化 CKMIL 的内存优势？
- 可视化中 Figure 5-6 是否选了最好的 case？有没有失败的案例（CKMIL 注意力错误的 WSI）？
- ICP 在不同场景下效果如此 inconsistent，是否应该从 contribution 中降级或移除？
- 作者有没有做跨数据集的迁移实验（如 BRACS 训练的模型在 TCGA 上测试）？
