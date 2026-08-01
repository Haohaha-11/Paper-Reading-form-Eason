# Can We Simplify Slide-level Fine-tuning of Pathology Foundation Models?

Jiawen Li<sup>1,∗</sup>, Jiali Hu<sup>1,∗</sup>, Qiehe Sun<sup>1</sup>, Renao Yan<sup>4</sup>, Minxi Ouyang<sup>1</sup>, Tian Guan<sup>1</sup>, Anjia Han<sup>3</sup>, Chao He<sup>2,†</sup>, and Yonghong He<sup>1,†</sup>

Shenzhen International Graduate School, Tsinghua University 2 Department of Engineering Science, University of Oxford 3 Department of Pathology, The First Afiliated Hospital of Sun Yat-sen University 4 Department of Mechanical Engineering, University of Washington chao.he@eng.ox.ac.uk, heyh@sz.tsinghua.edu.cn

Abstract. The emergence of foundation models in computational pathol ogy has transformed histopathological image analysis, with whole slide imaging (WSI) diagnosis being a core application. Traditionally, weakly supervised fine-tuning via multiple instance learning (MIL) has been the primary method for adapting foundation models to WSIs. However, in this work we present a key experimental finding: a simple nonlinear mapping strategy combining mean pooling and a multilayer perceptron, called SiMLP, can efectively adapt patch-level foundation models to slide-level tasks without complex MIL-based learning. Through extensive experiments across diverse downstream tasks, we demonstrate the superior performance of SiMLP with state-of-the-art methods. For instance, on a large-scale pan-cancer classification task, SiMLP surpasses popular MIL-based methods by 3.52%. Furthermore, SiMLP shows strong learning ability in few-shot classification and remaining highly competitive with slide-level foundation models pretrained on tens of thousands of slides. Finally, SiMLP exhibits remarkable robustness and transferability in lung cancer subtyping. Overall, our findings challenge the conventional MIL-based fine-tuning paradigm, demonstrating that a taskagnostic representation strategy alone can efectively adapt foundation models to WSI analysis. These insights ofer a unique and meaningful perspective for future research in digital pathology, paving the way for more eficient and broadly applicable methodologies.

Keywords: Whole slide image · Fine-tuning · Foundation model

## 1 Introduction

With advancements in self-supervised learning and large-scale whole-slide digitization, foundation model-based pathology AI workflows are transforming computational pathology [23,37]. Self-distillation across millions of pathology images enhances region-of-interest representation [3,33,28], while contrastive learning with natural language descriptions enables multimodal pathology models to integrate semantic knowledge [18,31,7]. As foundation models evolve, their ability to generalize across clinical tasks is becoming a key driver of future pathology AI development.

Representing and analyzing gigapixel-level WSIs remains a critical challenge. Traditional visual models pretrained on low-resolution natural images struggle as WSI encoders [11]. A common approach is to extract tissue-containing patches and aggregate their patch-level features for slide representation finetuning [23,19,14,35]. Given its alignment with clinical needs, weakly supervised methods, particularly multiple instance learning (MIL), have become a widely adopted fine-tuning strategy [12,21,32,17]. While pathology foundation models enable direct histopathology image encoding, their adaptation to WSIs still largely depends on MIL or its variants [3,33,28].

Many MIL-based fine-tuning methods integrate complex feature transformations [2,39,4,21,13] or high-order aggregation strategies [38,36,24,35,17], yet their necessity in the foundation model era remains uncertain. In fact, the performance of MIL-based fine-tuning with foundation models is task-dependent [34,26]. For example, traditional MIL has outperformed more complex methods in metastasis detection [16] and breast morphological subtyping [9] but underperformed in lung cancer subtyping [17]; meanwhile, compared to unsupervised strategies [22], MIL-based fine-tuning have shown unstable generalization. Given the strong features extracted by pretrained models, the advantages of complex fine-tuning strategies may be limited. Thus, exploring simplified fine-tuning approaches could ofer greater eficiency, deployment flexibility, and enhanced generalization in adapting foundation models to WSIs.

In this work, we demonstrated the feasibility of simplifying slide-level finetuning for foundation models through extensive experiments. Using a simple combination of task-agnostic average pooling and a non-linear MLP, termed SiMLP, we seamlessly adapted foundation models to slide-level tasks. To comprehensively evaluate SiMLP, we fine-tuned three representative foundation models on six large-scale WSI classification tasks across TCGA, CPTAC, EBRAINS [20], and HEROHE [5] cohorts, achieving state-of-the-art performance. Few-shot experiments on pan-cancer tasks from TCGA and CPTAC further confirmed its superior feature representation capabilities over MIL-based methods. To assess competitiveness against pretrained slide-level foundation models, we tested it on two challenging tasks in the BRACS cohort [1], where it remained highly competitive despite other models being pretrained on tens of thousands of WSIs. Finally, transferability experiments on non-small cell lung cancer subtyping across three cohorts showed that SiMLP maintains stability with minimal standard deviation, making it well-suited for scaling to large external test cohorts.

![](images/802262e2b01d3e5f091a30d051c28ea43d21a1a7199806a96a7f744e816134b4.jpg)  
Fig. 1. Transition of slide-level adaption in pathology foundation models. a. Conventional fine-tuning strategy using task-specific supervised learning. b. Simplified fine-tuning strategy using task-agnostic pooling and nonlinear classifier (SiMLP). c. Comparison of SiMLP and other MIL-based fine-tuning methods across three pathology foundation models.

## 2 Methodology

## 2.1 Weakly supervised learning on fine-tuning slide-level tasks

Before the large-scale development of pathology foundation models, visual models pretrained on natural images struggled to extract efective features from pathology images due to their limited pathology domain understanding. Consequently, weakly supervised learning has been necessary to obtain slide representations from patch features (Fig.1a). Specifically, given a WSI with patch feature set denoted as $P = \{ p _ { 1 } , p _ { 2 } , . . . , p _ { n } \}$ , feature transformation F(·) and aggregation G(·) are proposed:

$$
\big ( \tilde { p _ { 1 } } , \tilde { p _ { 2 } } , . . . , \tilde { p _ { n } } \big ) = F \big ( p _ { 1 } , p _ { 2 } , . . . , p _ { n } \big ) , \quad s = G \big ( \{ \tilde { p _ { 1 } } , \tilde { p _ { 2 } } , . . . , \tilde { p _ { n } } \} \big ) ,\tag{1}
$$

where F and G respectively denote vector and scalar-valued functions. MILbased fine-tuning typically follows the composition of these two functions. For instance, in the classical ABMIL [8], F is the identity mapping and G is a gated attention mechanism; whereas in TransMIL [21], F applies a nonlinear selfattention transformation and G outputs a class token. Regardless of the specific method, the composite function invariably contains learnable parameters that must be optimized using the slide-level labels from downstream tasks:

$$
\hat { y } = \mathrm { S o f t m a x } ( W s ) , \quad L ( s , y ) = - \sum _ { k = 1 } ^ { K } y _ { k } l n ( \hat { y } _ { k } ) ,\tag{2}
$$

Table 1. Slide-level classification on TCGA and CPTAC cohort in terms of balanced accuracy. Best performing fine-tuning approach for each metric is bolded. 95% CI is included in parentheses.
<table><tr><td>Approach</td><td>TCGA (OncoTree) (30 classes, 2703 WSIs)</td><td>TCGA (Pan Cancer) (22 classes, 2703 WSIs)</td><td>CPTAC (Pan Cancer) (12 classes, 1772 WSIs)</td></tr><tr><td></td><td>Linear probe ABMIL [8] DTFD-MIL [36]</td><td>0.8090 (0.8032-0.8148) 0.8008 (0.7877-0.8140) (0.7720-0.7821) 0.8095 (0.8021-0.8169)</td><td>0.8702 (0.8642-0.8763) 0.9143 (0.9126-0.9160) 0.8539 (0.8457-0.862) 0.8988 (0.8918-0.9059) 0.8970 (0.8922-0.9017)</td></tr><tr><td>CN] I118]</td><td>0.7770 ACMIL [38] RRTMIL [25]</td><td>0.8423 (0.8381-0.8465) 0.8618 (0.8547-0.8690) 0.8725</td><td>0.9068 (0.9033-0.9103) 0.8116 (0.8079-0.8152)</td></tr><tr><td>DiffMIL</td><td>0.8221 (0.8157-0.8286) 0.8171 (0.8089-0.8253)</td><td>(0.8584-0.8865) 0.8720 (0.8669-0.8771)</td><td>0.8961 (0.8916-0.9007)</td></tr><tr><td>SiMLP</td><td>0.8273 (0.8250-0.8295)</td><td>0.8788 (0.8729-0.8847)</td><td>0.9251 (0.9203-0.9298)</td></tr><tr><td>Linear probe</td><td>(0.8229-0.8360)</td><td></td><td></td></tr><tr><td>ABMIL [8]</td><td>0.8295</td><td>0.8816 (0.8780-0.8851)</td><td>0.8997 (0.8965-0.9029)</td></tr><tr><td>[ε] INN</td><td>0.7906</td><td>0.8541 (0.8484-0.8598)</td><td>0.8770 (0.8712-0.8827)</td></tr><tr><td>DTFD-MIL [36]</td><td>(0.7842-0.7970)</td><td></td><td></td></tr><tr><td>ACMIL [38]</td><td>0.8127 (0.8090-0.8165)</td><td>0.8560 (0.8495-0.8626)</td><td>0.8595 (0.8286-0.8904)</td></tr><tr><td></td><td>0.8240 (0.8152-0.8329)</td><td>0.8712 (0.8651-0.8773)</td><td>0.8968 (0.8913-0.9023)</td></tr><tr><td>RRTMIL [25]</td><td>0.8342 (0.8198-0.8486)</td><td>0.8720 (0.8634-0.8806)</td><td>0.7801 (0.7713-0.7890)</td></tr><tr><td>DiffMIL</td><td>0.8346 (0.8318-0.8374)</td><td>0.8833 (0.8772-0.8895)</td><td>0.8790 (0.8736-0.8844)</td></tr><tr><td>SiMLP</td><td>0.8488 (0.8440-0.8537)</td><td>0.8846 (0.8821-0.8872)</td><td>0.9147 (0.9117-0.9176)</td></tr><tr><td>3</td><td>Linear probe</td><td>(0.7991-0.8087)</td><td>0.8959</td></tr><tr><td>rro-Gathl ABMIL [8]</td><td>0.8039 0.7738</td><td>0.8674 (0.8584-0.8764) 0.8389</td><td>(0.8885-0.9034)</td></tr><tr><td>DTFD-MIL [36]</td><td>(0.7669-0.7807) 0.7852 (0.7780-0.7924)</td><td>(0.8302-0.8475) 0.8352 (0.8256-0.8449)</td><td>0.8837 (0.8804-0.8869) (0.8810-0.8844)</td></tr><tr><td>ACMIL [38]</td><td>0.7996 (0.7936-0.8056)</td><td>0.8600 (0.8478-0.8721)</td><td>0.8827 0.8947</td></tr><tr><td>RRTMIL [25]</td><td>0.8147 (0.8062-0.8233)</td><td>0.8368 (0.8253-0.8483)</td><td>(0.8859-0.9035) 0.7849 (0.7788-0.7909)</td></tr><tr><td>DiffMIL</td><td>0.8237 (0.8167-0.8306)</td><td>0.8650 (0.8599-0.8701)</td><td></td></tr><tr><td>SiMLP</td><td>0.8247 (0.8190-0.8304)</td><td>0.8739 (0.8643-0.8835)</td><td>0.8814 (0.8780-0.8849) 0.9109 (0.9053-0.9164)</td></tr></table>

where W is a linear classifier and L represents the cross-entropy loss. While effective, this approach yields task-dependent slide representations, limiting generalizability and robustness to distributional shifts.

## 2.2 Slide representation with task-agnostic pooling

Pathology foundation models pretrained over millions of histopathology images provide the possibility of obtaining task-agnostic slide representation. For instance, by clustering patch features extracted from the foundation model, WSI features can be represented as a combination of morphological prototypes [22]. Additionally, further training a slide encoder with proxy tasks based on largescale patch features has been shown to be an efective aggregation strategy for generating generic slide-level features, both in visual [33,30] and multimodal [9,6,27] settings. Although these approaches have demonstrated promising results, they often rely on additional signals for guidance. In contrast, a more straightforward approach is to leverage pooling layers, which represent one of the simplest feature aggregation methods. Pooling has been widely adopted in fine-tuning modules across various vision tasks and requires no additional learnable parameters. Therefore, the aggregation capability of pooling-based methods is worth exploring as a baseline, providing a simplified solution for slide-level fine-tuning and validating its transferability across diverse tasks.

Table 2. Slide-level classification on EBRAINS and HEROHE cohort in terms of balanced accuracy. Best performing fine-tuning approach for each metric is bolded. 95% CI is included in parentheses.
<table><tr><td rowspan=1 colspan=2>Approach</td><td rowspan=1 colspan=8>EBRAINS (Subtyping)    EBRAINS (IDH Pred.)   HEROHE (HER2 Pred.)(27 classes, 649 WSIs)     (2 classes, 208 WSIs)      (2 classes, 149 WSIs)</td></tr><tr><td rowspan=1 colspan=2>Linear probeN] I118] ABMIL [8]DTFD-MIL [36]ACMIL [38]RRTMIL [25]DiffMILSiMLP</td><td rowspan=1 colspan=8>0.6391(0.6312-0.6471)     0.8456(0.8281-0.8632)   0.7578(0.7493-0.7663)0.6366 (0.6283-0.6449)     0.8398 (0.8246-0.8549)     0.7268(0.6885-0.7652)0.5323 (0.5251-0.5395)    0.6871 (0.6649-0.7093)     0.7036 (0.6707-0.7365)0.6620 (0.6474-0.6766)   0.8650 (0.8584-0.8716)   0.7518 (0.7438-0.7598)0.6084 (0.5875-0.6292)    0.8325 (0.8126-0.8524)    0.6770 (0.6379-0.7162)0.6628 (0.6411-0.6844)    0.8388 (0.8293-0.8482)     0.7433 (0.7227-0.7639)0.6763(0.6641-0.6884)   0.8567(0.8417-0.8716)     0.7149(0.6720-0.7578)</td></tr><tr><td rowspan=2 colspan=2>Linear probeABMIL [8]</td><td rowspan=9 colspan=8>0.6818(0.6728-0.6908)   0.8879 (0.8797-0.8961)  0.7325(0.7144-0.7507)0.8371 (0.8069-0.8672)     0.6829(0.6498-0.7161)0.7058 (0.6884-0.7233)     0.7090 (0.6874-0.7307)0.6873 (0.6713-0.7032)    0.8671 (0.8578-0.8764)     0.6920 (0.6403-0.7437)0.6189 (0.6035-0.6342)    0.8565 (0.8495-0.8634)     0.6776 (0.6449-0.7103)0.8263 (0.8173-0.8352)     0.7193 (0.7084-0.7302)0.6703 (0.6348-0.7057)</td></tr><tr><td rowspan=1 colspan=1>0.6501 (0.6312-0.6690</td></tr><tr><td></td><td></td><td rowspan=3 colspan=1>0.5843 (0.5737-0.5949</td></tr><tr><td></td><td></td><td rowspan=1 colspan=4></td></tr><tr><td rowspan=3 colspan=2>[ε] INΩ DTFD-MIL [36]ACMIL [38]RRTMIL [25]</td></tr><tr><td rowspan=1 colspan=1>0.6873 (0.6713-0.7032</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>RTMIL, [25]</td><td rowspan=1 colspan=1>0.6189 (0.6035-0.6342</td></tr><tr><td rowspan=2 colspan=2>DiffMILSiMLP</td><td rowspan=2 colspan=4></td><td rowspan=1 colspan=1>(0.7084-0.730</td></tr><tr><td rowspan=1 colspan=4>0.8790 (0.8660-0.8919)</td></tr><tr><td rowspan=5 colspan=2>wv] -3 Linear probeABMIL [8]DTFD-MIL [36]ACMIL [38]RRTMIL [25]</td><td rowspan=1 colspan=1>0.6915 (0.6859-0.6971)</td><td rowspan=8 colspan=7>0.6913 (0.6762-0.7064)0.8492 (0.8390-0.8594)     0.6869 (0.6312-0.7425)0.7692 (0.7190-0.8193)     0.6305 (0.6135-0.6476)0.8646 (0.8508-0.8785)     0.6576 (0.6326-0.6826)(0.5943-0.6794))                           0.7092(0.6800-0.7384)0.8726 (0.8545-0.8906)   0.6778 (0.6487-0.7069)</td></tr><tr><td rowspan=1 colspan=1>0.6717 (0.6593-0.6841)</td><td rowspan=1 colspan=3>0.8492 (0.8390-0.8594)</td></tr><tr><td rowspan=1 colspan=1>0.5456 (0.4963-0.5949)</td><td rowspan=1 colspan=3>0.7692 (0.7190-0.8193)</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.7069 (0.6916-0.7221)</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4>0.8646 (0.8508-0.8785)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.6576 (0.6326-0.6826</td></tr><tr><td rowspan=1 colspan=1>0.6171 (0.6050-0.6291)</td></tr><tr><td></td><td></td><td></td><td rowspan=2 colspan=2></td></tr><tr><td></td><td></td><td></td><td rowspan=1 colspan=4>0.8290 (0.8085-0.8496)</td><td rowspan=1 colspan=1>0.6368 (0.5943-0.6794</td></tr><tr><td rowspan=1 colspan=2>DiffMILSiMLP</td><td rowspan=1 colspan=1>0.7161 (0.7037-0.72850.6978 (0.6841-0.7116)</td><td rowspan=1 colspan=4>0.8570 (0.8352-0.8788)</td></tr></table>

## 2.3 Non-linear classification head

Using linear probe, a simple linear transformation, general-purpose slide representations can be widely adapted to various WSI-based clinical tasks. However, its linear nature limits its ability to efectively align representations with the lower-dimensional space of downstream tasks. To enhance the transferability of slide representations, we adopt a non-linear classifier based on a two-layer MLP. Notably, modern deep learning frameworks can eficiently optimize matrix multiplications and the additional activation layer, this adjustment strikes a balance between improving representation flexibility and maintaining eficiency. The overall of SiMLP is shown in Fig.1b.

## 3 Experiments

## 3.1 Datasets and experimental settings

We conducted extensive experiments on seven large-scale datasets, including TCGA (OncoTree, 30 classes), TCGA (Pan Cancer, 22 classes), CPTAC (Pan Cancer, 12 classes), EBRAINS (Subtyping, 27 classes), EBRAINS (IDH Prediction, 2 classes), HEROHE (HER2 Prediction, 2 classes) and BRACS (Coarse-grained, 3 classes; Fine-grained 7 classes). We trained all the fine-tuning approaches with AdamW optimizer (learning rate: 10<sup>−4</sup>, betas=[0.9, 0.98], weight decay: 10<sup>−4</sup>) and a batch size of 1 for 20 epochs. All approaches were trained on

![](images/10b5d9f36f2b6fee67111b578c112439737853ca23027736e011a95f1b70883f.jpg)

![](images/81a51fa2834e24b0f6da7f23d8510fb1c329cc1f7023979b9995d3a0d0fdcc18.jpg)  
Fig. 2. Few-shot slide-level performance on TCGA and CPTAC cohort with K ∈ {1, 5, 10, 20, 50} slides per class.

Table 3. Comparison with slide-level foundation models on BRACS cohort in terms of balanced accuracy, ROC AUC, and weighted F1 score (reported as averages). Best performing approach for each metric is bolded.
<table><tr><td>Approach</td><td colspan="3">BRACS (Coarse-grained, 3 classes)</td><td colspan="3">BRACS (Fine-grained, 7 classes)</td></tr><tr><td></td><td>Bal ACC</td><td>ROC AUC</td><td>Weighted F1</td><td>Bal ACC</td><td>ROC AUC</td><td>Weighted F1</td></tr><tr><td>CHIEF [30] with Linear probe</td><td>0.5438</td><td>0.8195</td><td>0.5089</td><td>0.2732</td><td>0.7353</td><td>0.2506</td></tr><tr><td>CHIEF [30] with Full tuning</td><td>0.5833</td><td>0.8249</td><td>0.5457</td><td>0.2780</td><td>0.7663</td><td>0.2665</td></tr><tr><td>SiMLP with CTransPath [29]</td><td>0.5155</td><td>0.7433</td><td>0.5250</td><td>0.2518</td><td>0.6534</td><td>0.2955</td></tr><tr><td>GigaPath [33] with Linear probe</td><td>0.3771</td><td>0.7298</td><td>0.3220</td><td>0.2289</td><td>0.6757</td><td>0.2393</td></tr><tr><td>GigaPath [33] with Full tuning</td><td>0.3333</td><td>0.4409</td><td>0.1978</td><td>0.1429</td><td>0.5047</td><td>0.0335</td></tr><tr><td>SiMLP with Prov-GigaPath [33]</td><td>0.5409</td><td>0.7419</td><td>0.5474</td><td>0.2516</td><td>0.6772</td><td>0.2959</td></tr></table>

1 × 24GB NVIDIA 4090 with 5 fixed random seeds. Additional details of implementation, datasets, and baselines will be available in the Github codebase.

## 3.2 SiMLP outperforms in diverse slide-level classification

To evaluate SiMLP across slide-level tasks, we selected three representative pathology foundation models: CONCH [18], UNI [3], and Prov-GigaPath [33]. We conducted experiments on six tasks across four cohorts and performed a fair comparison against linear probe, four popular MIL-based methods (ABMIL [8], DTFD-MIL [36], ACMIL [38], and RRT-MIL [25]), and a diferential attentionbased MIL method (DifMIL) that we specifically designed (Table 1-2). Overall, SiMLP achieved superior performance across all three foundation models (81.32%, 81.52%, 80.96% in Fig.1c), demonstrating stronger adaptability than task-specific weakly supervised learning. Notably, SiMLP achieved the best results in three pan-cancer tasks, improving upon ABMIL by 3.52% and ACMIL by 1.83% in TCGA OncoTree classification. While SiMLP underperformed in HER2 prediction, the linear probe, which also uses mean pooling, performed well, suggesting that task-agnostic simplified aggregation can still produce efective representations.

![](images/7d0043c6fb962c810bcac02326430a3f8d0c62cc561073af6e6fe91fc482d716.jpg)  
Fig. 3. Robustness and transfer testing evaluation on CPTAC, TCGA, and in-house NSCLC cohort by sweeping 10 random seeds.

Table 4. Ablation study with on pooling and activation functions on TCGA-OncoTree in tems of balanced accuracy, ROC AUC, and weighted F1 score (reported as averages). Best performing configuration for each metric is bolded.
<table><tr><td>Configuration</td><td>Bal ACC</td><td>ROC AUC</td><td>Weighted F1</td><td>Configuration</td><td>Bal ACC</td><td>ROC AUC</td><td>Weighted F1</td></tr><tr><td> $\mathrm { M e a n + R e L U }$ </td><td>0.8488</td><td>0.9946</td><td>0.8893</td><td> $\mathrm { M a x } + \mathrm { R e L U }$ </td><td>0.7456</td><td>0.9927</td><td>0.8173</td></tr><tr><td> $\mathrm { M e a n + G e L U }$ </td><td>0.8509</td><td>0.9949</td><td>0.8901</td><td> $\mathrm { M a x } + \mathrm { G e L U }$ </td><td>0.7140</td><td>0.9920</td><td>0.8027</td></tr><tr><td> $\mathrm { M e a n + S w i g L U }$ </td><td>0.8054</td><td>0.9833</td><td>0.8533</td><td> $\mathrm { M a x + S w i g L U }$ </td><td>0.5871</td><td>0.9645</td><td>0.7049</td></tr></table>

## 3.3 SiMLP outperforms in few-shot learning classification

To evaluate learning eficiency and generalization with limited data, we conducted few-shot classification on TCGA and CPTAC pan-cancer tasks using UNI (Fig.2). We trained SiMLP, ABMIL, and ACMIL with $K \in \{ 1 , 5 , 1 0 , 2 0 , 5 0 \}$ samples per class. The results show that SiMLP consistently outperformed other methods across nearly all shot settings while exhibiting lower variance across random seeds $( \mathrm { s t d . < 0 . 0 1 }$ per shot). These results highlight that SiMLP has potential for screening rare and underrepresented clinical conditions.

## 3.4 SiMLP is competitive with slide-level foundation models

We compared SiMLP with two pretrained slide-level foundation models, CHIEF [30] and GigaPath [33] (Table 3), using the BRACS cohort, a challenging breast cancer subtype classification dataset with coarse-grained (3-class) and fine-grained (7-class) tasks. CHIEF employs CTransPath [29] as its patch feature extractor, while GigaPath uses Prov-GigaPath. For fair comparison, we evaluated SiMLP under the same patch-level foundation model, applying both linear probing and full parameter fine-tuning. Results show that while SiMLP underperforms CHIEF overall, it achieves higher weighted F1 scores in fine-grained classification. Compared to GigaPath, SiMLP outperforms across all metrics in both tasks, likely due to the high computational complexity and large parameter size of GigaPath, which may hinder convergence during downstream fine-tuning. Given that CHIEF and GigaPath were pretrained on tens of thousands of WSIs, the competitive performance of SiMLP is particularly noteworthy.

## 3.5 SiMLP has a good transfer capability

We further evaluated the transferability across cohorts by constructing an NSCLC subtype classification task using LSCC and LUAD cases from CPTAC, TCGA, and an in-house (IH-LUNG) cohort. We used UNI to train ABMIL, DifMIL, and SiMLP on CPTAC with 10 random seeds, followed by CPTAC internal testing and TCGA, IH-LUNG external testing (Fig.3). Results show that SiMLP outperforms other methods in internal testing and exhibits greater stability than both baselines in external test sets. This highlights that SiMLP provides better generalization and robustness in transfer learning scenarios.

## 3.6 Ablation study

Finally, we conducted an ablation study on SiMLP. Specifically, we replaced mean pooling with max pooling and examined the efect of substituting the ReLU activation function with GeLU and SwigLU in diferent combinations. These modifications were evaluated on the TCGA-OncoTree task with UNI encoder (Table 4). The results show that slide representations generated using max pooling perform worse than those generated with mean pooling, indicating that capturing global features remains crucial for task-agnostic aggregation. Additionally, we observed that the combination of GeLU and mean pooling led to improved performance, suggesting that adjusting the non-linear classifier further enhances adaptation to downstream tasks.

## 4 Conclusion and future direction

In this work we found that SiMLP, a simple fine-tuning method, enables pathology foundation models to efectively adapt to slide-level tasks. Extensive experiments demonstrate that SiMLP outperforms widely used MIL-based weakly supervised learning, confirming its strong performance and generalization ability.

Our findings provide four key insights for the future of computational pathology in the foundation model era:

1. Patch-level foundation model development remains crucial. While existing pretrained encoders enhance WSI analysis, balancing data redundancy and model complexity is essential. For instance, ViT-Base (CONCH) performed competitively against ViT-Giant (Prov-GigaPath). We encourage future research to explore eficient architectures, diverse multimodal models, and improved datadriven preprocessing strategies.

2. Task-agnostic slide representation learning may be more impactful than weakly supervised learning. Such representations improve generalization and stability while enabling broader applications like slide embedding retrieval and convenient multimodal integration.

3. Advancing slide-level foundation models enhances clinical performance. Pretraining slide encoders on large-scale datasets not only supports task-agnostic representation learning but also allows for performance improvements through diverse fine-tuning strategies.

4. Tailored weakly supervised learning remains necessary for slidelevel tasks. SiMLP performs well broadly, however, weakly supervised learning still holds advantages in specific tasks, highlighting its efectiveness for clinically tailored applications. For example, it remains valuable for biomarker prediction, hierarchical classification of rare diseases [10,15], and long-tailed data analysis.

In summary, as pathology foundation models continue to evolve, simplifying traditional weakly supervised learning paradigms and pioneering a new generation of research directions will be key to further enhancing performance and enabling broader real-world applications in computational pathology.

## 5 Acknowledgement

This work was supported by the National Natural Science Foundation of China (NSFC) under Grant No.82430062, the Shenzhen Engineering Research Centre under Grant XMHT20230115004. We thank the Jilin FuyuanGuan Food Group Co., Ltd for their collaboration. C.H. was also supported by the St John’s College, the University of Oxford, and the Royal Society (URF\R1\241734). The authors have no competing interests to declare that are relevant to the content of this paper.

## References

1. Brancati, N., Anniciello, A.M., Pati, P., et al.: Bracs: A dataset for breast carcinoma subtyping in h&e histology images. Database 2022, baac093 (2022)

2. Chen, R.J., Chen, C., Li, Y., et al.: Scaling vision transformers to gigapixel images via hierarchical self-supervised learning. In: CVPR. pp. 16144–16155 (2022)

3. Chen, R.J., Ding, T., Lu, M.Y., et al.: Towards a general-purpose foundation model for computational pathology. Nature Medicine 30(3), 850–862 (2024)

4. Chu, H., Sun, Q., Li, J., et al.: Retmil: Retentive multiple instance learning for histopathological whole slide image classification. In: MICCAI. pp. 437–447. Springer (2024)

5. Conde-Sousa, E., Vale, J., Feng, M., et al.: Herohe challenge: predicting her2 status in breast cancer from hematoxylin–eosin whole-slide imaging. Journal of Imaging 8(8), 213 (2022)

6. Ding, T., Wagner, S.J., Song, A.H., et al.: Multimodal whole slide foundation model for pathology. arXiv preprint arXiv:2411.19666 (2024)

7. Ikezogwo, W., Seyfioglu, S., Ghezloo, F., et al.: Quilt-1m: One million image-text pairs for histopathology. NeurIPS 36, 37995–38017 (2023)

8. Ilse, M., Tomczak, J., Welling, M.: Attention-based deep multiple instance learning. In: ICML. pp. 2127–2136. PMLR (2018)

9. Jaume, G., Vaidya, A., Zhang, A., et al.: Multistain pretraining for slide representation learning in pathology. In: ECCV. pp. 19–37. Springer (2024)

10. Jin, C., Luo, L., Lin, H., et al.: Hmil: Hierarchical multi-instance learning for finegrained whole slide image classification. IEEE Transactions on Medical Imaging (2024)

11. Van der Laak, J., Litjens, G., Ciompi, F.: Deep learning in histopathology: the path to the clinic. Nature medicine 27(5), 775–784 (2021)

12. Li, B., Li, Y., Eliceiri, K.W.: Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. In: CVPR. pp. 14318–14328 (2021)

13. Li, J., Chen, Y., Chu, H., et al.: Dynamic graph representation with knowledgeaware attention for histopathology whole slide image analysis. In: CVPR. pp. 11323–11332 (2024)

14. Li, J., Cheng, J., Meng, L., et al.: Deeptree: Pathological image classification through imitating tree-like strategies of pathologists. IEEE Transactions on Medical Imaging 43(4), 1501–1512 (2023)

15. Li, J., Sun, Q., Yan, R., et al.: Diagnostic text-guided representation learning in hierarchical classification for pathological whole slide image. arXiv preprint arXiv:2411.10709 (2024)

16. Ling, X., Lei, Y., Li, J., et al.: Towards a comprehensive benchmark for pathological lymph node metastasis in breast cancer sections. arXiv preprint arXiv:2411.10752 (2024)

17. Ling, X., Ouyang, M., Wang, Y., et al.: Agent aggregator with mask denoise mechanism for histopathology whole slide image analysis. In: ACM Multimedia. pp. 2795–2803 (2024)

18. Lu, M.Y., Chen, B., Williamson, D.F., et al.: A visual-language foundation model for computational pathology. Nature Medicine 30(3), 863–874 (2024)

19. Lu, M.Y., Williamson, D.F., Chen, T.Y., et al.: Data-eficient and weakly supervised computational pathology on whole-slide images. Nature Biomedical Engineering 5(6), 555–570 (2021)

20. Roetzer-Pejrimovsky, T., Moser, A.C., Atli, B., et al.: The digital brain tumour atlas, an open histopathology resource. Scientific Data 9(1), 55 (2022)

21. Shao, Z., Bian, H., Chen, Y., et al.: Transmil: Transformer based correlated multiple instance learning for whole slide image classification. NeurIPS 34, 2136–2147 (2021)

22. Song, A.H., Chen, R.J., Ding, T., et al.: Morphological prototyping for unsupervised slide representation learning in computational pathology. In: CVPR. pp. 11566–11578 (2024)

23. Song, A.H., Jaume, G., Williamson, D.F., et al.: Artificial intelligence for digital and computational pathology. Nature Reviews Bioengineering 1(12), 930–949 (2023)

24. Sun, Q., Jiang, D., Li, J., Yan, R., He, Y., Guan, T., Cheng, Z.: Nciemil: Rethinking decoupled multiple instance learning framework for histopathological slide classification. In: MIDL (2024)

25. Tang, W., Zhou, F., Huang, S., et al.: Feature re-embedding: Towards foundation model-level performance in computational pathology. In: CVPR. pp. 11343–11352 (2024)

26. Vaidya, A., Chen, R.J., Williamson, D.F., et al.: Demographic bias in misdiagnosis by computational pathology models. Nature Medicine 30(4), 1174–1190 (2024)

27. Vaidya, A., Zhang, A., Jaume, G., et al.: Molecular-driven foundation model for oncologic pathology. arXiv preprint arXiv:2501.16652 (2025)

28. Vorontsov, E., Bozkurt, A., Casson, A., et al.: A foundation model for clinical-grade computational pathology and rare cancers detection. Nature Medicine 30(10), 2924–2935 (2024)

29. Wang, X., Yang, S., Zhang, J., et al.: Transformer-based unsupervised contrastive learning for histopathological image classification. Medical Image Analysis 81, 102559 (2022)

30. Wang, X., Zhao, J., Marostica, E., et al.: A pathology foundation model for cancer diagnosis and prognosis prediction. Nature 634(8035), 970–978 (2024)

31. Xiang, J., Wang, X., Zhang, X., et al.: A vision–language foundation model for precision oncology. Nature pp. 1–10 (2025)

32. Xiang, J., Zhang, J.: Exploring low-rank property in multiple instance learning for whole slide image classification. In: ICLR (2023)

33. Xu, H., Usuyama, N., Bagga, J., et al.: A whole-slide foundation model for digital pathology from real-world data. Nature pp. 1–8 (2024)

34. Xu, H., Wang, M., Shi, D., et al.: When multiple instance learning meets foundation models: advancing histological whole slide image analysis. Medical Image Analysis 101, 103456 (2025)

35. Yan, R., Sun, Q., Jin, C., et al.: Shapley values-enabled progressive pseudo bag augmentation for whole-slide image classification. IEEE Transactions on Medical Imaging (2024)

36. Zhang, H., Meng, Y., Zhao, Y., et al.: Dtfd-mil: Double-tier feature distillation multiple instance learning for histopathology whole slide image classification. In: CVPR. pp. 18802–18812 (2022)

37. Zhang, S., Metaxas, D.: On the challenges and perspectives of foundation models for medical image analysis. Medical Image Analysis 91, 102996 (2024)

38. Zhang, Y., Li, H., Sun, Y., et al.: Attention-challenging multiple instance learning for whole slide image classification. In: ECCV. pp. 125–143. Springer (2024)

39. Zheng, Y., Gindra, R.H., Green, E.J., et al.: A graph-transformer for whole slide image classification. IEEE Transactions on Medical Imaging 41(11), 3003–3015 (2022)