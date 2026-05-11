# Whole slide image redundancy reduction for efficient pathological diagnosis

> Extracted from local PDF `/data/share/hxd/haojiang/main.pdf` with PyMuPDF because MinerU credentials were not available in this shell. Page text is preserved in reading order as extracted from the PDF; figures and tables are cropped from the original PDF pages into `images/`.



## Page 1

Contents lists available at ScienceDirect
Biomedical Signal Processing and Control
journal homepage: www.elsevier.com/locate/bspc

Whole slide image redundancy reduction for efficient pathological diagnosis
Shijie Li a,b
, Zhineng Chen a
,∗, Feng-Jung Chen b
, Xieping Gao c
a College of Computer Science and Artificial Intelligence, Fudan University, Shanghai 200438, China
b Shanghai Key Laboratory of Metabolic Remodeling and Health, Institute of Metabolism and Integrative Biology, Department of Endocrinology and Metabolism,
Zhongshan Hospital, Fudan University, Shanghai 200438, China
c Hunan Provincial Key Laboratory of Intelligent Computing and Language Information Processing, Hunan Normal University, Changsha 410081, China
A R T I C L E  I N F O
Keywords:
Computational pathology
Image redundancy reduction
Efficient diagnosis

A B S T R A C T
The rapid advancement of Computational Pathology (CP) has revolutionized pathological diagnosis and
driven the development of intelligent diagnostic technologies. However, a major challenge in applying these
technologies is processing Whole Slide Images (WSIs) with resolutions that can exceeding tens of millions
of pixels (e.g., 10,000 × 10,000). While existing methods predominantly focus on WSI representation and
discrimination, they often overlook efficiency—an increasingly critical concern given the growing demand for
high-throughput diagnosis. This highlights the urgent need for efficient solutions. Notably, the recurrence
of biological features and patterns in pathological images indicates substantial redundancy within WSIs,
offering an opportunity for optimization. In this paper, we propose WSIR2, a WSI Redundancy Reduction-
based diagnostic framework. WSIR2 incorporates a novel token compression module that strategically selects
the top-𝑘 most informative patches and merges less informative ones, thereby reducing both the number
of tokens and their feature dimensions. Additionally, it employs the cross-attention mechanism with (𝑛)
complexity to construct a lightweight classifier for efficient classification. These innovations lead to a dramatic
reduction in computational cost – achieving up to 24× lower GFLOPs compared to leading MIL methods – while
maintaining diagnostic performance on par with state-of-the-art models. Experimental results demonstrate that
WSIR2 consistently delivers faster training and higher inference throughput, offering a compelling balance
between accuracy and efficiency for large-scale pathological diagnostic tasks.
1. Introduction
Whole Slide Images (WSIs) scanned by digital slide scanners have
profoundly changed the diagnostic paradigm of clinical pathology [1,
2]. The laborious operation of targeting biological patterns under the
classical microscope has been replaced by the ability to access the
patients’ pathological status via electronic devices, which has greatly
streamlined pathologists’ workflow. Concomitantly, the digitalization
of pathology has introduced the challenge of efficiently and effec-
tively processing the colossal volumes of ever-increasing WSIs, whose
gigapixel-level resolution further aggravates the computational burden.
In the field of computational pathology, numerous outstanding methods
have been proposed to aid pathological diagnosis [3–6]. It is worth not-
ing that although these methods have gained impressive performance
on many general tasks and scenarios [7–10], few of them have explored
the efficiency of WSI analysis. In the face of the growing demand
for pathological diagnosis, there is still an urgent need to efficiently
analyze WSIs to accelerate the associated diagnostic tasks and both
qualitative and quantitative biological analyses.
∗Corresponding author.
E-mail address: zhinchen@fudan.edu.cn (Z. Chen).
Due to the gigapixels of WSI and scarce pixel-level annotations,
previous works have assumed that WSI contains abundant information
and have developed various effective methods within the framework
of weakly-supervised learning (WSL) for pathological diagnosis, where
only slide-level labels are available, while patch-level labels are not.
Furthermore, based on multiple instance learning (MIL), which treats
a WSI as a bag and patches from it as instances, numerous methods
have constructed promising deep models to capture the relationship
between bags and their labels, achieving impressive accuracy on gen-
eral tasks [11–16]. However, most of these works overlook the practical
demand for efficiency in clinical tasks. Although several methods have
explored the efficiency of WSI diagnosis [17–19], few focus on opti-
mizing computational cost and reducing redundant information within
WSIs. Motivated by the ever-increasing volume of pathological images,
it is crucial to improve the efficiency of MIL-based models.
On the other hand, compared to natural images, pathological images
contain monotonous, similar biological patterns such as cells, tissues,
https://doi.org/10.1016/j.bspc.2025.109289
Received 29 March 2025; Received in revised form 22 August 2025; Accepted 29 November 2025
Available online 2 December 2025
1746-8094/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.


## Page 2

and sub-organ structures. As Fig. 1 illustrates, patches from patho-
logical images exhibit higher similarity with each other than patches
from natural images, which indicates that reducing redundant patches
can improve diagnostic efficiency while retaining critical information
within a single image. Additionally, during the diagnostic process,
pathologists focus on disease-related patterns or features against a
regular background, e.g. a partial tumor region in the tissue landscape.
This also suggests that only a few significant patches adequately repre-
sent the entire status of patients. All these cues provide evidence that
excessive redundancy exists in WSI for pathological diagnosis. Thus,
reducing redundancy in WSI could effectively tackle the limitations of
efficiently processing gigapixel WSIs and satisfy the increasing demands
of computer-aided diagnosis.
Motivated by the observations above, we propose a WSI redundancy
reduction method, i.e. WSIR2, for efficient WSI analysis and diag-
nosis. WSIR2 is characterized by two modules: token compression
and a lightweight classifier. In the token compression module, we
simultaneously prune and merge tokens within a single block. Based
on the efficient cross-attention mechanism with (𝑛) computational
complexity and its interpretation of attention values as indicators of
token significance, WSIR2 achieves redundancy reduction of WSIs. It
simultaneously reduces features along both the feature dimension and
the token dimension by selecting top-k tokens and fusing the non-
significant tokens. Then, a lightweight classifier is proposed to further
reduce computational costs of features aggregation for WSI classifica-
tion, which is constructed with several cross-attention blocks. Again,
benefiting from the (𝑛) computational complexity of the blocks, its
computational cost is further reduced.
To evaluate the efficiency and effectiveness of WSIR2, we conducted
extensive experiments on multiple datasets, including Camelyon16,
TCGA-NSCLC, and UniToPatho. The experimental results demonstrate
that WSIR2 significantly reduces FLOPs and model size to 1.02 or 0.29
GFLOPs and 225K parameters on these datasets, while maintaining
competitive performance compared to other MIL methods. Further-
more, incorporating the token compression module into other MIL
methods also significantly accelerates their inference, demonstrating
its broad applicability. These results also highlight that redundancy
within WSIs has been comprehensively neglected in previous works,
and the efficiency of WSI tasks should be seriously considered to meet
the increasing demands of computer-aided diagnosis. WSIR2 establishes
an efficient paradigm for efficient WSI diagnosis, which has built the
foundation for the further development.
Our contributions are concluded as follows:
• We propose WSIR2, an efficient MIL method for pathological
diagnosis and analysis. It is composed of a token compression
module and a lightweight classifier.
• We propose token compression, which utilizes cross-attention to
select top-k patches and merge the non-top-k patches, thereby
accelerating inference efficiency.
• Extensive experimental results have validated the effectiveness
and efficiency of WSIR2, demonstrating its potential in resolving
the bottleneck of high-throughput diagnoses.
2. Related work
2.1. Multiple instance learning for WSI classification
MIL-based pathological diagnosis utilizes readily available slide-
level diagnostic results as labels for model training and has gained
widespread attention in the computational pathology community. Early
MIL-based methods are rooted in an instance-based approach, in which
a model is first trained for patch-level prediction and then produces
the slide-level prediction by aggregating the patch-level predicted log-
its using mean or max pooling [20,21]. As this scheme introduces
erroneous pseudo-labels for patches and fails to capture robust rela-
tionships between contextual information and slide-level predictions,
it has gradually been superseded by embedding-based methods. The
embedding-based approach takes patch-level embeddings as input and
directly predicts the slide-level label. Because patch-level embeddings,
rather than only their labels, are available, this branch of methods gen-
erally achieves better results. For example, MIL-RNN [11] treats a WSI
as a sequence of patches and accounts for patch permutation variation.
CLAM [9] constructs an attention-based model to perform instance-
level clustering over representative regions, which are identified to
constrain and refine slide-level features. TransMIL [15] models rela-
tionships among patches via self-attention (SA) [22]. HIPT [23] builds
a contrastive learning framework with hierarchical features for multi-
scale fusion of WSIs. HAG-MIL [24] proposes a hierarchical gated at-
tention mechanism to leverage multi-scale features of WSIs. To address
the data deficiency of pathological images, DTFD-MIL [19] generates
various sub-bags from WSIs for data augmentation. Remix [17] gen-
erates slide-level samples based on clustering prototypes. MHIM [25]
employs a siamese structure with a consistency constraint to explore
potential hard instances. IBMIL [26] achieves de-confounded bag-level
prediction based on an interventional training framework. In this work,
we specifically explore redundancy reduction within these mechanisms
and introduce cross-attention with learnable queries. This design en-
hances scalability and generalization, enabling our method to inte-
grate seamlessly into a variety of attention-based architectures while
preserving efficiency.
2.2. Redundancy reduction for vision tasks
Redundancy reduction was first proposed in the context of neuro-
science [27]. It has been instrumental in explaining the organization of
the visual system and has led to numerous algorithms for supervised
and unsupervised learning [28–31]. In vision tasks, this principle has
been applied to vision model pre-training and efficient model construc-
tion [32–34]. For example, Rao et al. [35] employ sparse attention
in vision transformers and retrain a network to dynamically filter to-
kens; Yin et al. [36] propose an adaptive token reduction mechanism to
speed up inference without modifying the network architecture; Liang
et al. [37] leverage cross-attention scores between the class token and
other tokens to reorganize image tokens; and Bolya et al. [38] propose
a general and lightweight matching algorithm to gradually combine
similar tokens within a ViT. Nevertheless, these studies primarily focus
on subject-specific natural images with fixed sizes. In this paper, apply-
ing redundancy reduction to gigapixel WSIs is rarely explored, and the
prevalence of monotonous biological patterns within WSIs motivates
our WSIR2.
3. Methodology
3.1. MIL formulation
Firstly, we provide a brief explanation of embedding-based MIL.
Taking binary classification as an example, to predict a target value
𝑌𝑖 of the 𝑖th bag, given a bag containing n instances {𝑥𝑖,1, 𝑥𝑖,2, … , 𝑥𝑖,𝑛}
while the instance-level labels {𝑦𝑖,1, 𝑦𝑖,2, … , 𝑦𝑖,𝑛} are unknown, a binary
classification can be defined as:
Yi =
{0
∀𝑗, 𝑦𝑖,𝑗= 0,
1
∃𝑗, 𝑦𝑖,𝑗= 1,
(1)
where 𝑦𝑖,𝑗∈{0, 1} and 𝑗= 1, 2, … , 𝑛.
Furthermore, the embedding-based MIL framework adopts a trans-
formation function f to obtain instance representations and a MIL
classifier S to predict the label of the bag based on these features, which
can be expressed as:
̂Yi = S(𝑓(𝑥𝑖,1), 𝑓(𝑥𝑖,2), … , 𝑓(𝑥𝑖,𝑛)).
(2)


## Page 3

Fig. 1. Cosine-similarity of patches from natural image and pathological image. A general vision encoder (ViT-small, pre-trained with DINO on diverse data
containing pathological images) is used to separately extract features from patches of the natural image and the pathological image as shown. Patches (4 × 4)
from the pathological image have much larger similarity values with each other compared to natural image.
Fig. 2. Framework of WSIR2. Patches tiled from 20× magnification of the WSI are used for self-supervised learning. Pre-trained ViT-small is subsequently used to
extract patch features at optimal magnification and then concatenated as slide-level features. WSIR2 reduces the redundancy at patch-level and dimension-level,
i.e., reduces numbers and dimensions of patches. The MIL classifier could be any model including ours and others proposed.
3.2. WSI redundancy reduction
The frequent recurrence of similar biological patterns in WSIs sug-
gests that disease-related semantics are globally sparse and largely
redundant. Selecting informative patches for diagnosis offers an effi-
cient paradigm, allowing MIL aggregation to focus on fewer yet more
critical tokens. In this context, token compression is an effective means
of mitigating WSI redundancy. However, unlike token pruning and
merging in ViTs for natural images [30,38], token compression in WSIs
cannot be performed progressively across multiple self-attention blocks,
as the enormous number of tokens renders the direct application of a
vanilla ViT impractical.
To overcome the limitations, we proposed WSIR2 to achieve  a
balance between efficiency and accuracy. As illustrated in Fig. 2,
the extracted patch-level features are concatenated into a slide-level
feature, 𝐗∈R𝑛×𝑑, where 𝑛 represents the number of patches and 𝑑
is feature dimension. WSIR2 performs features compression at the two
dimensions: token pruning and merging to reduce the patch number
𝑛, and dimensionality reduction to reduce the feature dimension 𝑑.
Concretely, patch-level features are first transformed to 𝐊∈R𝑛×𝑑′,
which represents the compressed features that retain the most critical
information for each patch. To reduce the number of patches that need
to be aggregated, a learnable parameter 𝐪∗∈R𝑑′ is introduced to
select the informative patches and merge these non-significant patches
simultaneously via cross-attention mechanism. Specifically, 𝐪∗ not only
assigns attention weights 𝑎𝑖 to each token in 𝐊 but also compresses
redundant tokens into a single representative token. The computation
of attention values is formulated as follows:
𝐊= 𝐖𝑘𝐗= {𝐤𝑖∣𝑖= 0, 1 … , 𝑛}
(3)
𝐀= softmax( 𝐪∗𝐊⊤
√
𝑑′
) = {𝑎𝑖∣𝑖= 1, 2, … , 𝑛}
(4)
where 𝐖𝑘 is a learnable projection, and 𝑑′ is the shared dimensionality
of 𝐪∗ and 𝐤𝑖.
Then, the top-𝑘 tokens with the highest attention values are selected
from 𝐊 according to a predefined ratio 𝑟 (i.e., the top-𝑘 ratio). The top-𝑘
tokens are concatenated as the informative tokens 𝐊𝑡∈R𝑘×𝑑′, while

![Figure 1: Cosine-similarity of patches from natural image and pathological image.](images/figure1.png)

*Figure 1: Cosine-similarity of patches from natural image and pathological image.*

![Figure 2: Framework of WSIR2.](images/figure2.png)

*Figure 2: Framework of WSIR2.*


## Page 4

other tokens are less informative tokens, 𝐊𝑟∈R(𝑛−𝑘)×𝑑′:
𝑘= max(1, ⌈𝑛⋅𝑟⌉),
(5)
𝐊𝑡= {𝐤𝑖∣𝑖∈TopK(𝑎𝑖, 𝑘)},
(6)
𝐊𝑟= {𝐤𝑖∣𝑖∉TopK(𝑎𝑖, 𝑘)}
(7)
where ⌈⋅⌉ denotes the ceiling function and 𝑘 is the number of retained
tokens.
For the less informative tokens, we compress them into a single
redundant token 𝐱𝑟∈R𝑑′, which aims to preserve some background
information of the WSI:
𝐕𝑟= 𝐖𝑣𝐊𝑟,
(8)
𝐀𝑟= {𝑎𝑖∣𝑖∉TopK(𝑎𝑖, 𝑘)},
(9)
𝐱𝑟= 𝐀𝑟𝐕𝑟,
(10)
where 𝐕𝑟∈R(𝑛−𝑘)×𝑑′ is the linearly projected representation of 𝐊𝑟,
sharing the same dimensionality 𝑑′. To minimize computational cost,
we reuse the attention scores of 𝐪∗ to 𝐊𝑟 (i.e., 𝐀𝑟) in Eq. (4). And
a linear mapping is directly applied to 𝐊𝑟 to obtain 𝐕𝑟, where 𝐖𝑣∈
R𝑑′×𝑑′ introduces significantly less computational burden compared to
applying it to the raw features 𝐊. This simple yet effective operation
serves as an approximation and is further optimized during training.
Subsequently, the pruned tokens (𝐊𝑡) and the merged token (𝐱𝑟) are
concatenated and fed into a lightweight classifier for prediction. Again,
we employ the cross-attention mechanism as the basic building block
of the classifier. Concretely, a class token 𝐐 is initialized as the query,
which attends to the input features and compresses them into a single
global representation 𝐗𝑔. The predicted label ̂𝑌 is then obtained from
this representation as follows:
𝐗𝑘= concat(𝐊𝑡, 𝐱𝑟),
(11)
𝐗𝑔= softmax
(
𝐐(𝐖𝑘′𝐗𝑘)⊤
√
𝑑′
)
(𝐖𝑣′𝐗𝑘),
(12)
̂𝐘= S(𝐗𝑔)
(13)
where S(⋅) is a scoring function that maps the global representation
to the final prediction, and 𝐖𝑘′ and 𝐖𝑣′ are learnable projection
matrices applied to 𝐗𝑘. In summary, WSIR2 leverages token compres-
sion in conjunction with a lightweight classifier, substantially reducing
computational cost and thereby enabling efficient diagnosis.
4. Experiments
4.1. Datasets
WSIR2 is extensively evaluated on three public datasets: Came-
lyon16 [39], TCGA-NSCLC, and UniToPatho [40]. Camelyon16 is col-
lected for metastasis detection in breast cancer and contains two classes
(Normal/Tumor). It officially provides 270 training samples and 129
testing samples, which can be tiled into roughly 57.18 million patches
at 20× magnification. TCGA-NSCLC includes two subtypes of lung can-
cer: Lung Adenocarcinoma (LUAD) and Lung Squamous Cell Carcinoma
(LUSC). We obtained 1042 slides from https://www.cancer.gov/ccg/
access-data. All slides of TCGA can be tiled into 145.63 million patches
at 20× magnification. We further split this dataset into a training set and
a test set using an 8:2 ratio, and evaluate all methods on the test set.
UniToPatho is a public dataset comprising 292 WSIs, of which we uti-
lized 243 from https://ieee-dataport.org/open-access/unitopatho. All
slides of UniToPatho can be tiled into 9.87 million patches at 20×
magnification. This dataset includes four disease subtypes (NORM, HP,
HG, LG), and we follow the official train/test split.
Table 1
Classification accuracy and AUC comparison on Camelyon16, TCGA-NSCLC,
and UniToPatho datasets. Results are averaged over five independent runs.
The highest performance is shown in bold, and the second best is underlined.
𝑟 denotes the top-k ratio used in WSIR2.
 Datasets
Camelyon16
TCGA-NSCLC
UniToPatho
 Methods
Acc.
AUC
Acc.
AUC
Acc.
AUC
 Mean
0.6279
0.5904
0.8804
0.9460
0.6847
0.8353
 Max
0.8992
0.9491
0.9398
0.9830
0.6937
0.8390
 ABMIL
0.9302
0.9656
0.9450
0.9812
0.7849
0.8968
 DSMIL
0.9205
0.9526
0.9408
0.9808
0.7685
0.9009
 TransMIL
0.9238
0.9542
0.9450
0.9830
0.6904
0.8215
 CLAM-SB
0.9256
0.9545
0.9453
0.9831
0.7838
0.9156
 CLAM-MB
0.9302
0.9536
0.9458
0.9832
0.7827
0.9071
 DTFD-MIL
0.9390
0.9643
0.9446
0.9736
0.7973
0.9239
 MHIM-MIL
0.9264
0.9570
0.9366
0.9785
0.7854
0.8996

WSIR2
𝑟= 1.00
0.9349
0.9674
0.9466
0.9810
0.8108
0.9273

𝑟= 0.50
0.9315
0.9654
0.9446
0.9813
0.8074
0.9243

𝑟= 0.10
0.9202
0.9429
0.9394
0.9819
0.7854
0.9132

𝑟= 0.01
0.8815
0.9132
0.9387
0.9825
0.7838
0.9072
4.2. Implementation details
We construct the datasets by tiling the WSIs into non-overlapping
224 × 224 patches at a magnification of 20×, discarding background
patches. We then conduct self-supervised pre-training with DINO [41]
on the datasets. In the pre-training step, the backbone is ViT-small/16
(12 transformer blocks with a hidden dimension of 384, patch size of
16 × 16). For the pre-training setup, we follow the original training
settings and only adjust the batch size due to limited hardware re-
sources. After pre-training, features extracted by ViT-small are used in
subsequent WSI classification. For a fair comparison of multiple MIL
methods, the averaged accuracy and AUC over 8 random repetitions
are reported. In the classification training step, cross-entropy loss is
adopted. The AdamW [42] optimizer with an initial learning rate of
2 × 10−4, and a cosine annealing scheme without warm restarts [43]
for learning rate scheduling, is used to update the model weights. The
main hyperparameter of WSIR2 is 𝑡𝑜𝑝𝑘_𝑟𝑎𝑡𝑖𝑜, abbreviated as 𝑟, which is
separately set to [0.01, 0.1, 0.5, 1.0] to evaluate the redundancy in WSIs.
To evaluate model efficiency, we report the model parameter counts
and FLOPs. The FLOPs differ across datasets, and we report the counts
for processing 14,000 and 4000 patches separately—14,000 being the
average number of patches per WSI from Camelyon16 and TCGA-
NSCLC, and 4000 from UniToPatho. All experiments are conducted
with 8× NVIDIA GeForce RTX 3090 GPUs. The unit of model parameter
counts is K.
4.3. Main results
We summarize the main comparison of classification performance
and computational efficiency in Tables 1–3. WSIR2 consistently demon-
strates a favorable balance between diagnostic accuracy and efficiency
across three benchmark datasets (Camelyon16, TCGA-NSCLC, and Uni-
ToPatho).
Classification Performance. As shown in Table 1, WSIR2 achieves
competitive accuracy and AUC on all datasets. With a lightweight
classifier and token dimension 𝑑′ = 96, and without pruning (𝑟=
1.00), WSIR2 attains the competitive accuracy on Camelyon16 (0.9349),
TCGA-NSCLC (0.9466), and UniToPatho (0.8108). These results indi-
cate that WSIR2 can preserve critical diagnostic information while using
substantially fewer model parameters.
Efficiency and Trade-off Analysis. Table 2 compares WSIR2 with
representative MIL methods in terms of model size, computational
cost, and inference speed. WSIR2 maintains the lowest parameter count
(225K) among all methods. In terms of GFLOPs, even without pruning
(𝑟= 1.00), WSIR2 requires only 1.36 G for 14,000 patches, which
is 3–25× lower than other MIL methods. Correspondingly, inference

![Table 1: Classification accuracy and AUC comparison on Camelyon16, TCGA-NSCLC, and UniToPatho.](images/table1.png)

*Table 1: Classification accuracy and AUC comparison on Camelyon16, TCGA-NSCLC, and UniToPatho.*


## Page 5

Table 2
Efficiency comparison of WSIR2 and other mainstream MIL methods. Model parameters (Params) and floating-point operations
(GFLOPs) are reported in thousands (K) and billions (G), respectively. The left and right GFLOPs correspond to processing 14,000
patches (average in Camelyon16 and TCGA-NSCLC) and 4000 patches (UniToPatho), respectively. Inference speed (Infer. Speed),
training time per epoch (Train Time), and memory usage (Mem. Usage) are measured on TCGA-NSCLC.
 Methods
Params
GFLOPs
Infer. Speed
Train time
Mem. Usage
 ABMIL
300
4.24, 1.20
965 WSIs/s
0.75 s/epoch
0.50 GB

 DSMIL
298
4.24, 1.21
342 WSIs/s
1.04 s/epoch
0.49 GB

 TransMIL
2344
34.61, 10.27
62 WSIs/s
10.60 s/epoch
2.3 GB

 CLAM-MB
461
6.58, 1.86
574 WSIs/s
1.10 s/epoch
0.57 GB

 DTFD-MIL
459
4.69, 1.30
419 WSIs/s
2.50 s/epoch
1.04 GB

 MHIM-MIL
526
15.01, 4.25
173 WSIs/s
1.76 s/epoch
0.57 GB


WSIR2
r = 1.00
225
1.36, 0.39
985 WSIs/s
1.06 s/epoch
0.43 GB


r = 0.50
225
1.02, 0.29
1001 WSIs/s
1.04 s/epoch
0.43 GB


r = 0.10
225
0.75, 0.21
1046 WSIs/s
0.98 s/epoch
0.43 GB


r = 0.01
225
0.69, 0.20
1096 WSIs/s
0.95 s/epoch
0.43 GB

Table 3
Efficiency and performance comparison when integrating WSIR2 into other MIL methods. The results are evaluated on TCGA-NSCLC. The units of model parameters
and FLOPs are in thousands (K) and billions (G), respectively. 𝑟 denotes the top-k ratio. Accuracy (ACC.) and area under the ROC curve (AUC) reflect diagnostic
performance, while GFLOPs, inference speed, training time, and memory usage assess computational efficiency.
 Methods
top-k ratio
ACC.
AUC
Param.
GFLOPs
Infer. Speed
Train time
Mem. Usage

ABMIL
w/o WSIR2
0.9450
0.9812
300
4.24
965 WSIs/s
0.75 s/epoch
0.50 GB


𝑟= 0.50
0.9402
0.9743
75
0.75
1257 WSIs/s
0.92 s/epoch
0.43 GB


𝑟= 0.10
0.9366
0.9700
75
0.70
1372 WSIs/s
0.85 s/epoch
0.43 GB


𝑟= 0.01
0.9091
0.9584
75
0.68
1451 WSIs/s
0.84 s/epoch
0.43 GB


DSMIL
w/o WSIR2
0.9408
0.9808
298
4.24
342 WSIs/s
1.04 s/epoch
0.49 GB


𝑟= 0.50
0.9386
0.9746
75
0.75
406 WSIs/s
1.13 s/epoch
0.44 GB


𝑟= 0.10
0.9322
0.9738
75
0.69
426 WSIs/s
1.08 s/epoch
0.44 GB


𝑟= 0.01
0.9234
0.9674
75
0.68
450 WSIs/s
1.02 s/epoch
0.43 GB


TransMIL
w/o WSIR2
0.9450
0.9830
2344
34.61
62 WSIs/s
10.60 s/epoch
2.30 GB


𝑟= 0.50
0.9372
0.9795
2252
17.14
136 WSIs/s
8.71 s/epoch
1.38 GB


𝑟= 0.10
0.9221
0.9695
2252
4.08
153 WSIs/s
8.19 s/epoch
0.69 GB


𝑟= 0.01
0.9091
0.9627
2252
1.24
163 WSIs/s
8.02 s/epoch
0.57 GB

Table 4
Efficiency of WSIR2 under varying token dimensions (𝑑′) and top-k ratios (𝑟). 𝑑′
controls the local feature dimension, and 𝑟 controls the proportion of preserved
tokens.
 Hyperparamter
Param ↓
FLOPs ↓
 𝑑′
𝑟
[K]
[G]


96
1.00
225
1.36,0.39

0.50
1.02,0.29

0.10
0.75,0.21

0.01
0.69,0.20

192
1.00
818
4.25,1.21

0.50
2.94,0.83

0.10
1.87,0.53

0.01
1.63,0.46

384
1.00
3110
14.78,4.19

0.50
9.57,2.71

0.10
5.31,1.51

0.01
4.36,1.23
speed reaches 985 WSIs/s, significantly higher than other MIL methods.
Memory usage is also reduced (0.43 GB vs. 0.49–2.3 GB for other mod-
els), demonstrating the lightweight nature of WSIR2. When applying
token pruning (𝑟= 0.50, 0.10, 0.01), GFLOPs decrease further to 1.02 G,
0.75 G, and 0.69 G, respectively, while inference speed increases
to 1001–1096 WSIs/s. Accuracy remains competitive across datasets
(Table 1). These results indicate that WSIR2 can achieve substantial
computational savings and faster inference than existing MIL models,
while offering flexible trade-offs between efficiency and performance.
Integration with Existing MIL Models. We further evaluated
WSIR2 as a slide-level feature compression module for representa-
tive MIL models, including ABMIL, DSMIL, and TransMIL (Table 3).
Incorporating WSIR2 substantially reduces computational cost and ac-
celerates inference across all architectures. For instance, with ABMIL at
𝑟= 0.50, GFLOPs drop from 4.24 G to 0.75 G and throughput increases
from 965 to 1257 WSIs/s, while accuracy only slightly decreases
(0.9450 → 0.9402). TransMIL achieves similar gains, with throughput
more than doubling at 𝑟= 0.50. These results highlight the gener-
alizability of WSIR2 for reducing redundancy in WSIs and improving
the efficiency of diverse MIL methods without severely compromising
predictive performance.
Overall, WSIR2 provides a practical solution for high-throughput
WSI analysis. It maintains competitive accuracy, drastically reduces
computational cost and memory usage, and accelerates inference. These
advantages make it particularly suitable for clinical deployment, where
efficiency and speed are critical for timely diagnosis.
4.4. Ablation
We further investigate the effects of varying token dimensions on
token pruning and merging. We report the performance of WSIR2
with 𝑑′ = 96 in Table 1, and compare it to 𝑑′ = 196 and 𝑑′ =
384 in Fig. 3. The efficiency metric is also reported in Table 4. For
WSIs with abundant tokens, such as TCGA-NSCLC, employing higher
token dimensions requires a greater number of tokens to construct
complete WSI representations for effective classification. Conversely,
fewer feature dimensions (𝑑′ = 96) exhibit stable robustness as tokens
are pruned.
In UniToPatho, which contains fewer tokens on average (approx-
imately 4000) and sparse biological tissue-structure information, de-
ploying a larger model with more hidden dimensions introduces noise
to the MIL model. This often leads to severe overfitting and degraded
accuracy on the test set—a trend also observed in TransMIL (see
Table 1). In contrast, WSIR2 with 𝑑′ = 96 effectively suppresses noise
and achieves optimal performance. These observations validate the
presence of redundancy within WSIs and emphasize the robustness of
WSIR2 under varying token dimensionalities.

![Table 2: Efficiency comparison of WSIR2 and mainstream MIL methods.](images/table2.png)

*Table 2: Efficiency comparison of WSIR2 and mainstream MIL methods.*

![Table 3: Efficiency and performance when integrating WSIR2 into other MIL methods.](images/table3.png)

*Table 3: Efficiency and performance when integrating WSIR2 into other MIL methods.*

![Table 4: Efficiency under varying token dimensions and top-k ratios.](images/table4.png)

*Table 4: Efficiency under varying token dimensions and top-k ratios.*


## Page 6

Fig. 3. Effect of token dimension (𝑑′) and top-k ratio (𝑟) on WSIR2 performance. Accuracy and AUC are evaluated on TCGA-NSCLC and UniToPatho
datasets to illustrate the trade-off between computational efficiency and predictive performance. Experiments are conducted with 𝑑′ ∈{96, 192, 384} and
𝑟∈{0.01, 0.10, 0.50, 1.00}.
Fig. 4. Visualization of pruned and merged patches in WSIs using WSIR2. Two representative Camelyon16 test samples are shown, illustrating the effect of WSIR2
under varying top-k ratios.
4.5. Visualization
To examine the behavior of pruned and merged patches in WSIR2,
we visualize two Camelyon16 WSIs using top-k ratios of {0.01, 0.10,
0.50}. As shown in Fig. 4, patches with highly similar biological
morphology, such as parenchymal cells (background information), are
merged. WSIR2 consequently focuses more on indispensable diagnostic
regions, such as fat cells and tumor cells within the tissue.
5. Discussion
WSIR2 effectively addresses redundancy in WSIs by leveraging to-
ken pruning and merging alongside a lightweight classifier, achieving
substantial reductions in computational cost and model size while pre-
serving critical diagnostic information. Its plug-and-play design enables
acceleration of various MIL models, demonstrating strong generaliz-
ability, and the visualization results confirm its focus on diagnosti-
cally relevant regions. However, WSIR2 relies on a fixed top-k ratio,
which may be suboptimal for highly heterogeneous WSIs, and extreme
pruning can slightly degrade accuracy. Additionally, it depends on
pre-extracted patch-level features and does not explicitly capture lo-
cal spatial context, which may limit performance for tasks requiring
fine-grained detail. Future improvements could include adaptive token
selection, multi-scale or hierarchical attention to better model local
and global context, integration with end-to-end trainable backbones
to reduce preprocessing overhead, and extension to other pathological
tasks, thereby enhancing both efficiency and clinical applicability.
6. Conclusion
In this paper, we propose WSIR2, an efficient framework for WSI
diagnosis. WSIR2 introduces a token compression module leveraging
the cross-attention mechanism and a lightweight classifier. The concept
of diagnostic acceleration is incorporated throughout the design of
WSIR2. Extensive experiments on multiple datasets demonstrate that
WSIR2 achieves highly competitive classification performance while
maintaining remarkable efficiency. Extending WSIR2 to a broader range
of WSI analysis tasks warrants further investigation. Therefore, our
future work will focus on whole end-to-end pipeline of WSI diagnosis
and facilitating additional downstream tasks.
CRediT authorship contribution statement
Shijie Li: Writing – review & editing, Writing – original draft,
Visualization, Validation, Supervision, Software, Resources, Project ad-
ministration, Methodology, Formal analysis, Data curation, Concep-
tualization. Zhineng Chen: Writing – review & editing, Supervision,
Funding acquisition, Conceptualization. Feng-Jung Chen: Supervision,
Resources, Formal analysis, Data curation. Xieping Gao: Supervision,
Project administration, Investigation, Conceptualization.
Declaration of competing interest
The authors declare that they have no known competing finan-
cial interests or personal relationships that could have appeared to
influence the work reported in this paper.

![Figure 3: Effect of token dimension and top-k ratio on WSIR2 performance.](images/figure3.png)

*Figure 3: Effect of token dimension and top-k ratio on WSIR2 performance.*

![Figure 4: Visualization of pruned and merged patches in WSIs using WSIR2.](images/figure4.png)

*Figure 4: Visualization of pruned and merged patches in WSIs using WSIR2.*


## Page 7

Acknowledgments
This work was supported by the National Natural Science Founda-
tion of China NO.(62372170, 32341012, 62172103) and National Key
R&D Program of China NO.2024YFA1802800.
Data availability
Data will be made available on request.
References
[1] M. Cui, D.Y. Zhang, Artificial intelligence and computational pathology, Lab.
Invest. 101 (4) (2021) 412–422.
[2] K. Bera, et al., Artificial intelligence in digital pathology—new tools for diagnosis
and precision oncology, Nat. Rev. Clin. Oncol. 16 (11) (2019) 703–715.
[3] G. Tu, et al., GAMMIL: A graph attention-guided multi-scale fusion multiple
instance learning model for the WHO grading of meningioma in whole slide
images, Biomed. Signal Process. Control. 105 (2025) 107652.
[4] R. Wang, Y. Gu, J. Yang, Enhancing whole slide image classification through
label denoising in a multi-instance learning framework, Biomed. Signal Process.
Control. 105 (2025) 107599.
[5] Y. Qu, X. Zhou, P. Huang, Y. Liu, F. Mercaldo, A. Santone, P. Feng, CGAM: An
end-to-end causality graph attention Mamba network for esophageal pathology
grading, Biomed. Signal Process. Control. 103 (2025) 107452.
[6] M. Feng, K. Xu, N. Wu, W. Huang, Y. Bai, Y. Wang, C. Wang, H. Wang,
Trusted multi-scale classification framework for whole slide image, Biomed.
Signal Process. Control. 89 (2024) 105790.
[7] C.L. Srinidhi, O. Ciga, A.L. Martel, Deep neural network models for
computational histopathology: A survey, Med. Image Anal. 67 (2021) 101813.
[8] L. Qu, et al., Towards label-efficient automatic diagnosis and analysis: a
comprehensive survey of advanced deep learning-based weakly-supervised, semi-
supervised and self-supervised techniques in histopathological image analysis,
Phys. Med. Biol. (2022).
[9] M.Y. Lu, et al., Data-efficient and weakly supervised computational pathology
on whole-slide images, Nat. Biomed. Eng. 5 (6) (2021) 555–570.
[10] A. Abdul Salam, et al., Skin whole slide image segmentation using
lightweight-pruned transformer, Biomed. Signal Process. Control. 106 (2025)
107624.
[11] G. Campanella, et al., Clinical-grade computational pathology using weakly
supervised deep learning on whole slide images, Nature Med. 25 (8) (2019)
1301–1309.
[12] Y. Zheng, et al., A Graph-Transformer for Whole Slide Image Classification, IEEE
Transactions on Medical Imaging 41 (11) (2022) 3003–3015.
[13] M. Ilse, J. Tomczak, M. Welling, Attention-based deep multiple instance learning,
in: ICML, 2018.
[14] H. Li, et al., DT-MIL: deformable transformer for multi-instance learning on
histopathological image, in: MICCAI, 2021.
[15] Z. Shao, et al., Transmil: Transformer based correlated multiple instance learning
for whole slide image classification, in: NeurIPS, 2021.
[16] Z. Fu, Q. Chen, M. Wang, C. Huang, Whole slide images classification model
based on self-learning sampling, Biomed. Signal Process. Control. 90 (2024)
105826.
[17] J. Yang, et al., Remix: A general and efficient framework for multiple instance
learning based whole slide image classification, in: MICCAI, 2022.
[18] H. Li, et al., Long-MIL: Scaling long contextual multiple instance learning for
histopathology whole slide image analysis, 2023, arXiv preprint arXiv:2311.
12885.
[19] H. Zhang, et al., DTFD-MIL: Double-tier feature distillation multiple instance
learning for histopathology whole slide image classification, in: CVPR, 2022.
[20] Y. Xu, et al., Deep convolutional activation features for large scale Brain Tumor
histopathology image classification and segmentation, in: ICASSP, 2015.
[21] L. Hou, et al., Patch-based convolutional neural network for whole slide tissue
image classification, in: CVPR, 2016.
[22] A. Vaswani, et al., Attention is all you need, in: NeurIPS, 2017.
[23] R.J. Chen, et al., Scaling vision transformers to gigapixel images via hierarchical
self-supervised learning, in: CVPR, 2022.
[24] C. Xiong, et al., Diagnose like a pathologist: Transformer-enabled hierarchical
attention-guided multiple instance learning for whole slide image classification,
in: IJCAI, 2023.
[25] W. Tang, et al., Multiple instance learning framework with masked hard instance
mining for whole slide image classification, in: CVPR, 2023.
[26] T. Lin, et al., Interventional bag multi-instance learning on whole-slide
pathological images, in: CVPR, 2023.
[27] H.B. Barlow, Possible principles underlying the transformations of sensory
messages, in: Sensory Communication, 2012.
[28] J. Ballé, V. Laparra, E.P. Simoncelli, End-to-end optimized image compression,
2016, arXiv preprint arXiv:1611.01704.
[29] J. Lindsey, et al., A unified theory of early visual representations from retina
to cortex through anatomically constrained deep CNNs, 2019, arXiv preprint
arXiv:1901.00945.
[30] M. Chen, et al., Diffrate: Differentiable compression rate for efficient vision
transformers, in: ICCV, 2023.
[31] J. Mu, X. Li, N. Goodman, Learning to compress prompts with gist tokens, in:
NeurIPS, 2024.
[32] J. Zbontar, et al., Barlow twins: Self-supervised learning via redundancy
reduction, in: ICML, 2021.
[33] Y. Luo, et al., Learning to rank patches for unbiased image redundancy reduction,
in: CVPR, 2024.
[34] Y. Wang, et al., R2-trans: Fine-grained visual categorization with redundancy
reduction, 2022, arXiv preprint arXiv:2204.10095.
[35] Y. Rao, et al., Dynamicvit: Efficient vision transformers with dynamic token
sparsification, in: NeurIPS, 2021.
[36] H. Yin, et al., A-vit: Adaptive tokens for efficient vision transformer, in: CVPR,
2022.
[37] Y. Liang, et al., Not all patches are what you need: Expediting vision transformers
via token reorganizations, 2022, arXiv preprint arXiv:2202.07800.
[38] D. Bolya, et al., Token merging: Your vit but faster, 2022, arXiv preprint
arXiv:2210.09461.
[39] B.E. Bejnordi, et al., Diagnostic assessment of deep learning algorithms for
detection of lymph node metastases in women with breast cancer, JAMA 318
(22) (2017) 2199–2210.
[40] C.A. Barbano, et al., Unitopatho, a labeled histopathological dataset for colorectal
polyps classification and adenoma dysplasia grading, in: ICIP, 2021.
[41] M. Caron, et al., Emerging properties in self-supervised vision transformers, in:
CVPR, 2021.
[42] I. Loshchilov, F. Hutter, Fixing weight decay regularization in adam, 2018, arXiv
preprint arXiv:1711.05101.
[43] I. Loshchilov, F. Hutter, Sgdr: Stochastic gradient descent with warm restarts,
2016, arXiv preprint arXiv:1608.03983.
