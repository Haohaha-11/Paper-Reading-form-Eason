# RetMIL: Retentive Multiple Instance Learning for Histopathological Whole Slide Image Classification

Hongbo Chu<sup>1</sup>, Qiehe Sun<sup>1</sup>, Jiawen Li<sup>1</sup>, Yuxuan Chen<sup>1</sup>, Lizhong Zhang<sup>1</sup>, Tian Guan<sup>1</sup>, Anjia Han<sup>2</sup>, and Yonghong He<sup>1</sup>

<sup>1</sup> Shenzhen International Graduate School, Tsinghua University, China {zhu-hb23,sunqh21,lijiawen21,chenyx23,zhanglz21}@mails.tsinghua.edu.cn {heyh, guantian}@sz.tsinghua.edu.cn

<sup>2</sup> department of Pathology, The First Afiliated Hospital of Sun Yat-sen University, China hananjia@mail.sysu.edu.cn

Abstract. Histopathological whole slide image (WSI) analysis with deep learning has become a research focus in computational pathology. The current paradigm is mainly based on multiple instance learning (MIL), in which approaches with Transformer as the backbone are well discussed. These methods convert WSI tasks into sequence tasks by representing patches as tokens in the WSI sequence. However, the feature complexity brought by high heterogeneity and the ultra-long sequences brought by gigapixel size makes Transformer-based MIL sufer from the challenges of high memory consumption, slow inference speed, and lack of performance. To this end, we propose a retentive MIL method called RetMIL, which processes WSI sequences through hierarchical feature propagation structure. At the local level, the WSI sequence is divided into multiple subsequences. Tokens of each subsequence are updated through a parallel linear retention mechanism and aggregated utilizing an attention layer. At the global level, subsequences are fused into a global sequence, then updated through a serial retention mechanism, and finally the slidelevel representation is obtained through a global attention pooling. We conduct experiments on two public CAMELYON and BRACS datasets and an public-internal LUNG dataset, confirming that RetMIL not only achieves state-of-the-art performance but also significantly reduces computational overhead. Our code will be accessed shortly.

Keywords: Histopathological Whole Slide Image · Multiple Instance Learning · Retention Mechanism.

## 1 Introduction

Pathological slide scanners store microscopic fields of view as the WSI, laying the foundation for automatic diagnostics based on deep learning [18]. However, the gigapixel-level resolution and the lack of pixel-level annotations pose significant challenges in developing such intelligent tools. In recent years, with the development of weakly-supervised technologies, MIL methods for WSI analysis have been well studied, which treats WSI as bags and cropped patches as instances. By embedding instances into high-dimensional space for aggregation, slide-level representations can be obtained. MIL methods are generally categorized into instance-level [4,7,11] and embedding-level [10,13,21] approaches. The former has been gradually replaced due to enormous data requirements and weak generalization.

Embedding-level MIL methods generally focus on proposing efective aggregation strategies to obtain more efective WSI representations. Although mean or max pooling is a direct corollary of the MIL theory, dynamically assigning importance scores to patches has proven more efective [10,16]. In addition, due to the wide application of Transformer[24], more research focuses on predicting WSI scores by modeling the correlation between patches through the self-attention mechanism, which helps describe the underlying tumor microenvironment patterns. Transformer-based MIL methods have shown better performance in many WSI analysis tasks[21,6,26]. However, the square complexity caused by the nonlinear mechanism of self-attention consumes more memory during training and inference, resulting in increased latency and reduced speed, which is not conducive to the actual deployment of algorithms in clinical scenarios.

To alleviate the above challenges, in this paper, we proposed a retentive multiple instance learning neural network called RetMIL, which introduces a retention mechanism to replace nonlinear self-attention, and efectively integrates subsequence information of WSI to obtain global representation with local features by building a hierarchical structure. We conduct experiments on public CAMELYON and BRACS datasets, as well as LUNG dataset for public data training and internal data testing. Results demonstrate that our proposed Ret-MIL achieves lower memory cost and higher throughput while exhibiting competitive performance.

## 2 Methodology

In this section, we introduce the methodology of RetMIL. First, WSI is processed into a sequence form. Then local subsequences and the global sequence are sequentially updated and aggregated through retention and attention pooling. Finally, the prediction score of WSI is obtained through the classification head. Fig 1 shows the overall framework of RetMIL.

## 2.1 From WSI to Sequence

We preprocess a WSI in four steps. First, we use the OTSU algorithm [19] to segment the WSI foreground, and then use the sliding window operation to crop patches under a fixed magnification. Secondly, ViT-S/16 [8], which is pretrained based on DINO [5] on large-scale WSIs [12], is used as a feature extractor to encode each patch into a high-dimensional feature embedding $x _ { i } \in$ $\mathbb { R } ^ { d \times 1 }$ . Next, we form all $x _ { i }$ into a sequence $X = \{ x _ { 1 } , \ldots , x _ { N } \}$ , and split it into multiple subsequences $\{ S _ { 1 } , \ldots , S _ { q } , R \}$ . Specifically, let $N = q l + r$ , where l is the length of each subsequence $S _ { j } ~ = ~ \{ x _ { ( j - 1 ) l + 1 } , \ldots , x _ { j l } \} , ~ j ~ = ~ 1 , 2 , \ldots , q$ and $r ~ = ~ | R |$ . Finally, to ensure that all subsequences have the same length and facilitate parallel calculation, we extend R to $S _ { q + 1 } = C o n c a t ( R , X _ { l - r } )$ , where $| X _ { l - r } | = l - r .$ , and there are three situations for $X _ { l - r }$ as follows.

![](images/134f66750ea24437813d06319c46736ebcc25cc90f9d35fd1d587806bdcfd622.jpg)  
Fig. 1. Overall framework of RetMIL

– If $r = 0 ,$ , then $X _ { l - r } = \theta$

$- \mathrm { ~ I f ~ } 0 < r < l / 2$ , let $l - r = a r + b$ and $A = \{ x _ { q l + 1 } , \dots , x _ { q l + r } \}$ , then $X _ { l - r } =$ $\{ \underset {  } { A , \dots , A } , x _ { q l + 1 } , \dots , x _ { q l + b } \}$

If $r \geq l / 2$ , then $X _ { l - r } = \{ x _ { q l + 1 } , \ldots , x _ { q l + ( l - r ) } \} ,$

The purpose of processing R in this way is to make each $x _ { i }$ exist in only one subsequence, ensuring that the mapping between feature embeddings and subsequences is satisfied.

## 2.2 Retention Mechanism

Inspired by applications of the retentive network in large language models [23], RetMIL updates and aggregates sequence tokens through retention mechanisms. Given the matrix form $\bar { S } \in \mathbb { R } ^ { | S | \times d }$ of an input sequence, we first use three linear

layers to project it into diferent feature spaces:

$$
Q = X W _ { Q } , K = X W _ { K } , V = X W _ { V } ,\tag{1}
$$

where $W _ { Q } , \ W _ { K }$ , and $W _ { V }$ are learnable transformation matrices respectively. Next, we split $Q , K$ and V into multiple heads $\{ Q _ { h } \} , \{ K _ { h } \}$ and $\left\{ V _ { h } \right\}$ and perform rotational position encoding [22] on each $Q _ { h }$ and $K _ { h }$ to obtain $Q _ { h }$ and $\tilde { K _ { h } }$ . Then we use the retention layer for processing, which is expressed as follows:

$$
R e t e n t i o n ( h , X ) = ( \tilde { Q } _ { h } \tilde { K } _ { h } ^ { \top } \odot D _ { h } ) V _ { h } ,\tag{2}
$$

where $D _ { h }$ is a relative distance decay matrix, and each element $D _ { h , n m }$ is expressed as:

$$
D _ { h , n m } = \left\{ \begin{array} { c c } { \gamma ^ { n - m } , n \geq m } \\ { 0 , n < m } \end{array} \right.\tag{3}
$$

Finally, we use GroupNorm [25] and swish gate [9,20] to normalize the output, and concatenate all retention head. The above mapping relationship can provide batch-level parallel calculation. When the input batch is $B ,$ , we denote the entire update operation as $M S R ( B ; S )$ ).

## 2.3 Hierarchical Retentive Aggregation Architecture

For any subsequence matrix $S _ { i } , i \in { 1 , . . . , q + 1 }$ in WSI, $M S R ( 1 ; S _ { i } )$ represents the result of $S _ { i }$ after passing through the retention mechanism. Our goal is to update all subsequences in parallel, which is expressed as follows:

$$
\begin{array} { r l r } {  { ( F _ { 1 } , \dots , F _ { q + 1 } ) = ( M S R ( 1 ; S _ { 1 } ) , \dots , M S R ( 1 ; S _ { q + 1 } ) ) } } \\ & { } & { = M S R ( q + 1 ; ( S _ { 1 } , \dots , S _ { q + 1 } ) ) , } \end{array}\tag{4}
$$

where $F _ { i } \in \mathbb { R } ^ { l \times d }$ represents the output embedding of subsequence $S _ { i }$ . Next, we use the attention pooling layer to aggregate the element features of each subsequence, which is expressed as:

$$
F _ { l o c a l , i } = \sum _ { k = 1 } ^ { l } { \alpha } _ { i , k } F _ { i , k } ,\tag{5}
$$

where $F _ { i , k }$ represents the kth element of $F _ { i } , F _ { l o c a l , i } \in \mathbb { R } ^ { d \times 1 }$ represents the feature embedding of subsequence $S _ { i } . ~ \alpha _ { k }$ is calculated through a nonlinear gating mechanism:

$$
\alpha _ { i , k } = \frac { \exp \{ \Gamma _ { l } \operatorname { t a n h } ( \mathrm { W } _ { l } F _ { i , k } ) \odot \mathrm { s i g m } ( \mathrm { U } _ { l } F _ { i , k } ) \} } { \sum _ { t = 1 } ^ { l } \exp \{ \Gamma _ { l } \operatorname { t a n h } ( \mathrm { W } _ { l } F _ { i , t } ) \odot \mathrm { s i g m } ( \mathrm { U } _ { l } F _ { i , t } ) \} } ,\tag{6}
$$

where $\Gamma _ { l } \in \mathbb { R } ^ { 1 \times M } , \mathrm { W } _ { l } , \mathrm { U } _ { l } \in \mathbb { R } ^ { M \times d }$ are learnable parameters, tanh(·), sigm(·) are nonlinear activation functions based on tanh and sigmoid respectively.

Next, we convert the feature embeddings of all subsequences into the local WSI feature matrix $F _ { l o c a l } = ( F _ { l o c a l , 1 } , \dots , F _ { l o c a l , q + 1 } ) ^ { \top } \in \mathbb { R } ^ { ( q + 1 ) \times d }$ , and utilize the retention mechanism to update:

$$
G = M S R ( 1 ; F _ { l o c a l } ) ,\tag{7}
$$

where $G \in \mathbb { R } ^ { ( q + 1 ) \times d }$ . Then attention pooling is used again to aggregate the (q + 1) dimension:

$$
F _ { g l o b a l } = \sum _ { p = 1 } ^ { q + 1 } \beta _ { p } G _ { p } ,\tag{8}
$$

where $G _ { p }$ represents the pth row element of G, and $\beta _ { p }$ represents as follows:

$$
\beta _ { p } = \frac { \exp \{ \Gamma _ { g l o b a l } \operatorname { t a n h } ( \mathrm { W } _ { g l o b a l } G _ { p } ) \odot \mathrm { s i g m } ( \mathrm { U } _ { g l o b a l } G _ { p } ) \} } { \sum _ { t = 1 } ^ { q + 1 } \exp \{ \Gamma _ { g l o b a l } \operatorname { t a n h } ( \mathrm { W } _ { g l o b a l } G _ { t } ) \odot \mathrm { s i g m } ( \mathrm { U } _ { g l o b a l } G _ { t } ) \} } ,\tag{9}
$$

where $\Gamma _ { g l o b a l } \in \mathbb { R } ^ { 1 \times M } , \mathrm { W } _ { g l o b a l } , \mathrm { U } _ { g l o b a l } \in \mathbb { R } ^ { M \times d }$ are learnable parameters. For the WSI classification task, $F _ { g l o b a l }$ is passed through a linear classifier to obtain the prediction score. The entire RetMIL is trained using the cross-entropy function as the objective loss.

## 3 Experiment

## 3.1 Datasets

CAMELYON: The CAMELYON dataset focuses on the binary classification task of lymph node metastases in breast cancer. It includes 399 WSIs from CAMELYON16 [2] and 500 WSIs from CAMELYON17 [1]. We use all data of CAMELYON16 to conduct four-fold cross-validation experiments, and choose the CAMELYON17 training set as our testing dataset.

BRACS: The BRACS dataset [3] focuses on multi-classification tasks aimed at subtype analysis of breast cancer. We conduct experiments based on the oficial classification. The dataset comprises 395 training samples, 65 validation samples, and 87 test samples. We use four diferent sets of model initialization parameters for training and testing.

LUNG: The LUNG dataset is a binary classification task focusing on non-small cell lung cancer subtypes. The training set is collected from TCGA repository [15], containing 541 WSIs of lung adenocarcinoma (LUAD) and 458 lung squamous cell carcinoma (LUSC). The test set is from the cooperative hospital, comprising 105 LUAD and 65 LUSC WSIs. We conduct four-fold cross-validation experiments on the training set and perform inference on the test set.

## 3.2 Experiment Setup and Evaluation Metrics

During the preprocessing stage, all WSIs are cropped into 224 × 224 patches at 20× magnification. The length of each subsequence is set to 512. The entire experiment is conducted on one NVIDIA RTX 4090, with 100 epochs, utilizing early stopping with 15 rounds. The batch size is set as 1, and the learning rate is 1e-4, with a weight decay of 1e-5 for the Adam optimizer. All other baseline methods adopt the same experimental settings. We record the Balanced Accuracy(B-Acc) and Weighted F1-score as evaluation metrics to comprehensively evaluate the performance.

## 3.3 Result

Performance Evaluation against SOTA: Table 1 displays the performance of our proposed RetMIL, and we compare it with the following six state-of-the-art methods: For attention-based MIL: ABMIL [10], DSMIL [14], CLAM-MB [16]. For Transformer-based MIL: TransMIL[21], HIPT [6] and HAG-MIL [26]. In the CAMELYON dataset, our RetMIL surpasses the second-ranked model TransMIL by 3.18% and 3.43% in F1-score and balanced accuracy. In the BRACS dataset, our model leads by 1.52% and 0.86% compared with the second-ranked CLAM-MB, while also achieving the minimum variance among all models. In the LUNG dataset, RetMIL outperforms by 0.13% in balanced accuracy.

Table 1. Mean and standard deviation of F1-score and Balanced accuracy (expressed in %) between RetMIL and current powerful MIL method. The best is in BOLD, and the second best is indicated with underline.
<table><tr><td rowspan="2">Methods</td><td colspan="2">CAMELYON</td><td colspan="2">BRACS</td><td colspan="2">LUNG</td></tr><tr><td>F1-score</td><td>B-Acc</td><td>F1-score</td><td>B-Acc</td><td>F1-score</td><td>B-Acc</td></tr><tr><td>ABMIL [10]</td><td> $8 1 . 2 7 _ { 3 . 1 1 }$ </td><td> $8 1 . 6 0 _ { 2 . 3 0 }$ </td><td> $6 4 . 1 1 _ { 5 . 2 4 }$ </td><td> $6 3 . 1 7 _ { 4 . 3 9 }$ </td><td> $8 8 . 6 8 _ { 3 . 9 8 }$ </td><td>90.713.26</td></tr><tr><td>CLAM-MB [16]</td><td> $8 3 . 0 6 _ { 4 . 5 9 }$ </td><td> $8 3 . 3 7 _ { 3 . 1 5 }$ </td><td> $\underline { { 6 6 . 9 9 _ { 4 . 0 2 } } }$ </td><td> $\underline { { 6 6 . 1 5 3 . 6 5 } }$ </td><td> $8 7 . 6 7 _ { 2 . 2 5 }$ </td><td> $8 9 . 7 3 \substack { 1 . 7 6 }$ </td></tr><tr><td>DSMIL [14]</td><td> $8 3 . 9 8 _ { 1 . 7 9 }$ </td><td> $8 3 . 7 7 _ { 1 . 3 0 }$ </td><td> $6 0 . 1 2 _ { 4 . 5 2 }$ </td><td> $5 9 . 2 2 _ { 3 . 2 3 }$ </td><td> $8 5 . 8 6 _ { 9 . 1 5 }$ </td><td> $8 6 . 5 7 _ { 8 . 1 8 }$ </td></tr><tr><td>TransMIL [21]</td><td> $\underline { { 8 4 . 0 6 _ { 8 . 1 9 } } }$ </td><td> $\underline { { 8 4 . 1 0 _ { 5 . 3 7 } } }$ </td><td> $6 2 . 8 3 _ { 3 . 9 7 }$ </td><td> $6 1 . 5 6 _ { 3 . 5 3 }$ </td><td> $\mathbf { 9 1 . 7 5 _ { 2 . 7 3 } }$ </td><td> $\underline { { 9 1 . 4 3 _ { 3 . 5 1 } } }$ </td></tr><tr><td>HIPT [6]</td><td> $7 8 . 9 2 _ { 8 . 1 1 }$ </td><td> $8 0 . 1 7 _ { 5 . 5 2 }$ </td><td> $6 6 . 1 9 _ { 8 . 9 7 }$ </td><td> $6 5 . 7 3 _ { 6 . 9 2 }$ </td><td> $8 1 . 5 5 _ { 6 . 3 7 }$ </td><td> $8 4 . 8 5 _ { 4 . 8 3 }$ </td></tr><tr><td>HAG-MIL [26]</td><td> $7 9 . 3 5 5 . 7 1 $ </td><td> $8 0 . 5 9 _ { 4 . 0 8 }$ </td><td> $6 6 . 2 6 _ { 4 . 5 2 }$ </td><td> $6 4 . 7 6 _ { 4 . 8 0 }$ </td><td> $8 5 . 4 7 _ { 4 . 4 2 }$ </td><td> $8 7 . 6 1 _ { 3 . 5 7 }$ </td></tr><tr><td>RetMIL (Ours)</td><td> $\mathbf { 8 7 . 2 4 _ { 4 . 2 2 } }$ </td><td> $\mathbf { 8 7 . 5 3 _ { 3 . 9 2 } }$ </td><td> $\mathbf { 6 8 . 5 1 _ { 0 . 5 4 } }$ </td><td> $\mathbf { 6 7 . 0 1 _ { 0 . 7 1 } }$ </td><td> $\underline { { 9 1 . 5 1 _ { 2 . 6 4 } } }$ </td><td> $\mathbf { 9 1 . 5 6 _ { 2 . 7 7 } }$ </td></tr></table>

We also compared RetMIL with Transformer-based models on the CAME-LYON dataset using AUC, which are shown in Fig 2.a. It can be observed that RetMIL achieves a 1.36% improvement in AUC compared to Transformer-based models. Additionally, Fig 2.b demonstrates the results of feature representations. All feature embeddings are reduced to a two-dimensional vector through the t-SNE algorithm [17]. Our observation reveals that RetMIL can better widen the gap between distinct categories while minimizing the separation among patches belonging to the same category compared with the TransMIL algorithm.

Performance at diferent lengths of sequences: In Transformer-based MIL methods, the length of the WSI sequence represents the number of cropped patches. We analyze model performance under diferent sequence lengths, and the result is shown in Table 2. Regardless of the length of the WSI sequence, our proposed method always significantly outperforms Transformer-based methods, especially for ultra-long sequences $( \mathrm { i . e . } ,$ , oversized WSI), which demonstrates the efectiveness of RetMIL in long sequence analysis.

![](images/ee7553be3790c584ebb4199c3fa490cc7de861b9b5804ae8514c1b141368907b.jpg)  
(a)  
(b)  
Fig. 2. (a) ROC curves and corresponding area under the curve(AUC) values for Ret-MIL and Transformer-based models (b)Visual analysis of feature dimensionality reduction between RetMIL and TransMIL.

Table 2. Performance comparison of RetMIL and Transformer-based models at diferent sequence lengths.
<table><tr><td rowspan="3">Methods</td><td colspan="6">Patch Number</td></tr><tr><td colspan="2">0-5000</td><td colspan="2">5001-10000</td><td colspan="2">10001-15000 15001-</td></tr><tr><td></td><td></td><td></td><td>F1-score B-Acc F1-score B-Acc F1-score B-Acc F1-score B-Acc</td><td></td><td></td></tr><tr><td>TransMIL [21]</td><td> $8 1 . 8 3 _ { 8 . 3 3 }$ </td><td> $8 1 . 6 9 _ { 5 . 5 5 }$ </td><td> $8 6 . 6 6 _ { 8 . 2 9 }$   $8 6 . 5 1 _ { 5 . 3 7 }$ </td><td> $\left| 8 4 . 4 7 _ { 1 0 . 4 6 } \right.$   $8 4 . 7 8 _ { 8 . 3 5 }$ </td><td> $7 9 . 2 9 _ { 5 . 8 3 }$ </td><td> $7 9 . 8 9 _ { 4 . 9 5 }$ </td></tr><tr><td>HIPT [6]</td><td> $7 7 . 9 8 _ { 6 . 8 9 }$ </td><td> $7 7 . 6 3 _ { 4 . 7 4 }$ </td><td> $8 3 . 8 6 _ { 8 . 5 9 }$   $8 4 . 3 4 _ { 5 . 9 5 }$ </td><td> $7 9 . 8 9 _ { 4 . 9 3 }$   $8 1 . 4 3 _ { 3 . 0 8 }$ </td><td> $7 3 . 5 7 _ { 4 . 4 5 }$ </td><td> $7 4 . 1 7 _ { 3 . 8 1 }$ </td></tr><tr><td>HAG-MIL [26]</td><td> $7 8 . 1 0 _ { 5 . 7 4 }$ </td><td> $7 8 . 1 8 _ { 4 . 4 1 }$ </td><td> $8 2 . 3 2 _ { 5 . 8 1 }$   $8 3 . 3 9 _ { 4 . 1 3 }$ </td><td> $7 8 . 7 4 6 . 2 0 $   $8 0 . 7 2 _ { 4 . 8 0 }$ </td><td> $6 8 . 7 3 \mathrm { _ { 7 . 1 5 } }$ </td><td> $7 1 . 1 4 5 . 1 3$ </td></tr><tr><td>RetMIL (Ours) 86</td><td> $\mathbf { . 6 7 _ { 2 . 2 6 } }$ </td><td>83.991.70 89</td><td> $\mathbf { 7 6 _ { 1 . 8 5 } }$  88  $\mathbf { . 1 9 1 . 5 8 }$ </td><td>88.590.6087.640.42</td><td>82.634.42 82.504.20</td><td></td></tr></table>

Inference performance: We also analyze the inference throughput and GPU memory usage under diferent sequence lengths with the Transformer-based models. As shown in Fig 3.b, the GPU memory consumption of HIPT and HAG-MIL almost linearly increases with increasing sequence lengths, except for the lightweight-designed TransMIL model. However, our RetMIL maintains almost constant GPU memory consumption. Fig 3.a shows that our retention model significantly improves model throughput. Even compared to the lightweightdesigned TransMIL, our model maintains a nearly 1.5× lead in throughput.

![](images/5dbaff5955587bc265c5c72505fc57d96e019b24724edfbe9b52504e4d2fd904.jpg)  
(a)

![](images/1b15f2c2fe42a93ab5c9f02a4f942546f5036d2ac0d71b7e56d9f46c33769b86.jpg)  
(b)  
Fig. 3. (a) Comparison of throughput between RetMIL and Transformer-based models at diferent sequence lengths. (b) Comparison of GPU memory consumption between RetMIL and Transformer-based models at diferent sequence lengths.

![](images/00971923460f1dc9e6b89c5da27750eff9e3a4ba3eb2564011779c888b2af011.jpg)  
Fig. 4. Heatmap visualization of WSI examples. In each pair of images, the left part displays the standard cancer regions outlined by pathologists (indicated by red contours), while the right side shows the heatmaps generated by our RetMIL.

Visulization: Fig 4 presents heatmap visualization results of our RetMIL. We select two macro-metastatic cancer slides and one micro-metastasis cancer slide from the CAMELYON17 to analyze the attention area of our model. For the kth element in subsequence $i ,$ the attention score $s c o r e _ { i , k }$ can be calculated as follow:

$$
s _ { i , k } = \alpha _ { i , k } \cdot \beta _ { i } ,\tag{10}
$$

For both macro-metastatic and micro-metastatic cancer, our model can accurately and comprehensively pay attention to the cancer area marked by the pathologist, which demonstrates the great interpretability of our model.

## 4 Conclusion

In this paper, we propose a retentive multiple instance learning approach called RetMIL, which uses linear retention mechanisms to reduce the computational overhead while modeling the correlation between patches. In addition, the hierarchical retentive aggregation architecture is designed to update local subsequences and characterize the global WSI sequence comprehensively. We demonstrate the superiority of RetMIL through comparative experiments on three histopathology WSI datasets. At the same time, we also compared the inference performance with the Transformer-based methods, and the results show that our proposed RetMIL has lower computational consumption.

## References

1. Bandi, P., Geessink, O., Manson, Q., Van Dijk, M., Balkenhol, M., Hermsen, M., Bejnordi, B.E., Lee, B., Paeng, K., Zhong, A., et al.: From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge. IEEE transactions on medical imaging 38(2), 550–560 (2018)

2. Bejnordi, B.E., Veta, M., Van Diest, P.J., Van Ginneken, B., Karssemeijer, N., Litjens, G., Van Der Laak, J.A., Hermsen, M., Manson, Q.F., Balkenhol, M., et al.: Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. Jama 318(22), 2199–2210 (2017)

3. Brancati, N., Anniciello, A.M., Pati, P., Riccio, D., Scognamiglio, G., Jaume, G., De Pietro, G., Di Bonito, M., Foncubierta, A., Botti, G., et al.: Bracs: A dataset for breast carcinoma subtyping in h&e histology images. Database 2022, baac093 (2022)

4. Campanella, G., Hanna, M.G., Geneslaw, L., Miraflor, A., Werneck Krauss Silva, V., Busam, K.J., Brogi, E., Reuter, V.E., Klimstra, D.S., Fuchs, T.J.: Clinicalgrade computational pathology using weakly supervised deep learning on whole slide images. Nature medicine 25(8), 1301–1309 (2019)

5. Caron, M., Touvron, H., Misra, I., J´egou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)

6. Chen, R.J., Chen, C., Li, Y., Chen, T.Y., Trister, A.D., Krishnan, R.G., Mahmood, F.: Scaling vision transformers to gigapixel images via hierarchical self-supervised learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16144–16155 (2022)

7. Chikontwe, P., Kim, M., Nam, S.J., Go, H., Park, S.H.: Multiple instance learning with center embeddings for histopathology classification. In: Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part V 23. pp. 519–528. Springer (2020)

8. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

9. Hendrycks, D., Gimpel, K.: Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415 (2016)

10. Ilse, M., Tomczak, J., Welling, M.: Attention-based deep multiple instance learning. In: International conference on machine learning. pp. 2127–2136. PMLR (2018)

11. Kanavati, F., Toyokawa, G., Momosaki, S., Rambeau, M., Kozuma, Y., Shoji, F., Yamazaki, K., Takeo, S., Iizuka, O., Tsuneki, M.: Weakly-supervised learning for lung carcinoma classification using deep learning. Scientific reports 10(1), 9297 (2020)

12. Kang, M., Song, H., Park, S., Yoo, D., Pereira, S.: Benchmarking self-supervised learning on diverse pathology datasets. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3344–3354 (2023)

13. Lerousseau, M., Vakalopoulou, M., Deutsch, E., Paragios, N.: Sparseconvmil: sparse convolutional context-aware multiple instance learning for whole slide image classification. In: MICCAI Workshop on Computational Pathology. pp. 129–139. PMLR (2021)

14. Li, B., Li, Y., Eliceiri, K.W.: Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 14318–14328 (2021)

15. Liu, J., Lichtenberg, T., Hoadley, K.A., Poisson, L.M., Lazar, A.J., Cherniack, A.D., Kovatich, A.J., Benz, C.C., Levine, D.A., Lee, A.V., et al.: An integrated tcga pan-cancer clinical data resource to drive high-quality survival outcome analytics. Cell 173(2), 400–416 (2018)

16. Lu, M.Y., Williamson, D.F., Chen, T.Y., Chen, R.J., Barbieri, M., Mahmood, F.: Data-eficient and weakly supervised computational pathology on whole-slide images. Nature biomedical engineering 5(6), 555–570 (2021)

17. Van der Maaten, L., Hinton, G.: Visualizing data using t-sne. Journal of machine learning research 9(11) (2008)

18. Madabhushi, A.: Digital pathology image analysis: opportunities and challenges. Imaging in medicine 1(1), 7 (2009)

19. Otsu, N.: A threshold selection method from gray-level histograms. IEEE transactions on systems, man, and cybernetics 9(1), 62–66 (1979)

20. Prajit Ramachandran, B.Z., Le, Q.V.: Swish: a self-gated activation function. arXiv: Neural and Evolutionary Computing, 2017 (2017)

21. Shao, Z., Bian, H., Chen, Y., Wang, Y., Zhang, J., Ji, X., et al.: Transmil: Transformer based correlated multiple instance learning for whole slide image classification. Advances in neural information processing systems 34, 2136–2147 (2021)

22. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063 (2024)

23. Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., Wei, F.: Retentive network: A successor to transformer for large language models (2023). URL http://arxiv. org/abs/2307.08621 v1

24. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

25. Wu, Y., He, K.: Group normalization. In: Proceedings of the European conference on computer vision (ECCV). pp. 3–19 (2018)

26. Xiong, C., Chen, H., Sung, J.J., King, I.: Diagnose like a pathologist: Transformerenabled hierarchical attention-guided multiple instance learning for whole slide image classification. arXiv preprint arXiv:2301.08125 (2023)