CKMIL: Cascaded Key-Instance Attention Multiple Instance Learning for

Histopathology Whole Slide Image Analysis

Anonymous submission

Abstract

In computational pathology (CPath), the analysis of Whole
Slide Images (WSIs) using Multiple Instance Learning (MIL)
is a key technology for precision medicine. However, exist-
ing methods face a dilemma when modeling inter-instance
correlations: they either overlook the correlations entirely
or model them in a key-instance agnostic manner. Methods
based on the independent attention weighting ignore inter-
actions among instances, while the standard self-attention
mechanism is difﬁcult to apply to WSIs with massive num-
bers of instances due to its O(n2) computational complexity.
Although recent linear-complexity methods have addressed
the efﬁciency issue, they generally adopt a key-instance ag-
nostic strategy. This can dilute the sparse yet crucial diag-
nostic signals in WSIs, leading to suboptimal performance.
To address this challenge, we propose CKMIL, a novel
Cascaded Key-Instance Attention framework. CKMIL op-
erates via a two-stage cascaded process: ﬁrst, a Subspace-
Disentangled Attention (SDA) module identiﬁes candidate
key sub-instances with high discriminative scores within mul-
tiple feature subspaces. Subsequently, a Key-Instance Guided
Global Attention (KGGA) module utilizes these candidates
as landmarks for Nystr¨om attention. This achieves efﬁcient
global interaction guided by key information, effectively pre-
venting the dilution of diagnostic signals. Furthermore, pos-
tulating that local correlations exist among the components
within an instance’s feature vector, we introduce an Instance-
Conv-Projection (ICP) module to capture this internal feature
structure better. Extensive experiments for cancer subtyping
and survival prediction on public datasets, including BRACS
and the TCGA-BLCA/BRCA/NSCLC cohorts, demonstrate
that when used with feature extractors pre-trained on the gen-
eral domain, our proposed method surpasses existing main-
stream methods in performance.

Introduction
Computational pathology (CPath) (Cai et al. 2021; Cifci
et al. 2023), an interdisciplinary ﬁeld at the intersection of
pathology and computer science, has emerged as a fron-
tier with immense potential in precision medicine (Bera
et al. 2019). Unlike traditional pathology, which relies on
the visual assessment of tissue slides by pathologists—a pro-
cess that is costly, labor-intensive, and susceptible to inter-
observer variability (Elmore et al. 2015), computational
pathology leverages computational methods to analyze dig-
itized Whole Slide Images (WSIs) (Cui and Zhang 2021;

Stage2:Feature Aggregation

Downstream Tasks

Stage1:Feature Extraction

Feature Extractor

···

Whole Slide Image

Patching

Features

Original Features
Original Features

Attention Scores
Aggregation

Aggregated

Feature

Original Features
Adjusted Features Aggregated

Feature

Original Features

···

Independent Attention-Based Methods

Efficient Global Interaction Methods

Our Proposed CKMIL Method

MLP

Global 
Interaction

SDA

Initial Scores

Candidate 
Key-instances

Key Guided

Global 
Refinement

Final Scores

Original Features

Aggregated
   Feature
KGGA

Attention Scoring
Correlation
Keys Guided Global Interaction

Attention Scoring
Correlation
Keys Guided Global Interaction

Attention Scoring
Correlation
Keys Guided Global Interaction

···

···

···

···

···

···

···

Figure 1: The two-stage paradigm of MIL and a comparison
of different MIL methods. Top Methods: Generate attention
scores for each instance, but ignore the correlations. Middle
Methods: Model inter-instance correlations, but they cannot
generate attention scores for individual instances, and their
global interaction overlooks the critical role of sparse posi-
tive instances. Bottom(Our Method): Our method generates
attention scores for each instance and models their correla-
tions through a global interaction guided by key instances.
This approach effectively prevents the dilution of key diag-
nostic signals during the correlation modeling process.

Song et al. 2023). This provides decision support for early
diagnosis, prognosis prediction, and personalized treatment.

Although WSIs are considered the gold standard in com-
putational pathology due to their ability to capture compre-
hensive tumor microenvironment (Cai et al. 2021), their gi-
gapixel size (e.g.,80, 000 × 80, 000 pixels at 40× magniﬁ-
cation) and the scarcity of ﬁne-grained annotations present
signiﬁcant challenges for conventional deep learning models
(Campanella et al. 2019; Jin et al. 2023).

To address these challenges, Multiple Instance Learning
(MIL) has become the de facto paradigm for WSI analysis
(Maron and Lozano-P´erez 1997; Amores 2013; Campanella
et al. 2019; Lu et al. 2021). In this paradigm, each WSI
is treated as a bag, and the patches obtained by dividing it
are called instances. The prevalent MIL pipeline employs a
pre-trained feature extractor to encode instances into low-
dimensional features, followed by an aggregator that pools

instance features into a bag-level representation for down-
stream tasks such as cancer subtyping (Chen et al. 2013;
Coudray et al. 2018) and survival prediction (Yu et al. 2016).

While early MIL methods used simple pooling (Yu et al.
2016), attention-based approaches such as ABMIL (Ilse,
Tomczak, and Welling 2018) and CLAM (Lu et al. 2021)
were introduced to weight instances by their importance.
However, by treating instances as independent and identi-
cally distributed (i.i.d.), these models fundamentally ignore
the crucial contextual correlations among them. To capture
instance correlations, Transformer-based methods were ex-
plored, but they faced the prohibitive computational com-
plexity of O(n2). To overcome the computational complex-
ity, methods with linear complexity, such as MambaMIL
(Yang, Wang, and Chen 2024) and TransMIL (Shao et al.
2021), were proposed. However, these approaches often
failed to capture the most critical diagnostic information.
Their inherent simpliﬁcation strategies risked diluting the
signals from sparse but vital instances within a WSI, lead-
ing to suboptimal results.

Overall, existing methods for modeling instance correla-
tions are limited (as illustrated in Figure 1): independent
attention neglects instance interplay, while efﬁcient global
methods are key-instance agnostic, diluting critical diagnos-
tic signals.

In this paper, We propose Cascaded Key-Instance At-
tention Multiple Instance Learning (CKMIL), a framework
built on the principle that key instances should guide efﬁ-
cient global interaction. CKMIL materializes this through
a cascaded process. First, our Subspace-Disentangled At-
tention (SDA) module screens for candidate key instances
within feature subspaces. Crucially, the subsequent Key-
Instance Guided Global Attention (KGGA) module lever-
ages these very candidates as the landmarks for Nystr¨om
attention (Xiong et al. 2021). This design anchors the efﬁ-
cient global interaction directly to the most salient signals.
The resulting global context then reﬁnes the initial scores
from SDA via a gated fusion mechanism, tightly coupling
the screening and interaction stages. Additionally, we intro-
duce an exploratory Instance-Conv-Projection (ICP) mod-
ule to capture intra-feature correlations using convolutions
to replace conventional linear layers for generating Q and
K vectors. Our primary contributions are as follows:

• A novel cascaded attention framework, CKMIL, that ef-
ﬁciently models inter-instance dependencies in a key-
instance-guided manner.

• A Key-Instance Guided Global Attention (KGGA)
mechanism that uses key instances as landmarks to ad-
dress the information dilution problem in existing linear-
complexity methods.

• An Instance-Conv-Projection (ICP) module that lever-
ages convolutional fusion to capture latent intra-feature
correlations often missed by conventional linear layers.

• State-of-the-art
(SOTA)
performance
with
general-
purpose feature extractors and strong competitive perfor-
mance with domain-speciﬁc medical feature extractors
on cancer subtyping and survival prediction tasks.

Related Work
Multiple Instance Learning for WSI Analysis
The MIL paradigm addresses the challenge of gigapixel-
scale WSIs by treating each slide as a bag of instances
(Maron and Lozano-P´erez 1997), effectively leveraging bag-
level labels. Under this paradigm, the primary objective of
MIL becomes learning the relationships among instances
within a bag. A typical two-stage MIL approach involves
two steps (Lu et al. 2021). First, a feature encoder (e.g.,
Resnet50 (He et al. 2016)), often pre-trained on large-scale
image datasets (Deng et al. 2009), transforms instances into
low-dimensional feature vectors. Second, an aggregation
module is designed to aggregate instance-level features into
a bag-level representation for downstream tasks.

Attention as Independent Instance Weighting
To overcome the limitations of simple pooling aggregators
such as Mean-pooling and Max-pooling, attention mecha-
nisms were introduced to assign discriminative weights to
instances based on their importance (Ilse, Tomczak, and
Welling 2018). Foundational methods in this category, in-
cluding ABMIL (Ilse, Tomczak, and Welling 2018), CLAM
(Lu et al. 2021), and DSMIL (Li, Li, and Eliceiri 2021),
typically employ a shared attention network to score each
instance independently. However, these methods are funda-
mentally built on the independent and identically distributed
(i.i.d.) assumption, thereby neglecting the correlations be-
tween instances. This premise contradicts core pathology
principles, where interactions within the tumor microenvi-
ronment are often crucial for diagnosis. Consequently, by
treating each instance in isolation, these models cannot fully
model the broader tissue context and may over-focus on cy-
tologically salient but diagnostically redundant areas.

Global Interaction in MIL with Linear Complexity
To address the context-agnostic nature of independent
weighting, methods incorporating global self-attention were
explored. However, the standard self-attention mechanism,
with its prohibitive O(n2) computational complexity, is ill-
suited for the massive number of instances in a WSI. This
challenge motivated the development of several global in-
teraction methods with linear complexity. Prominent ex-
amples include TransMIL (Shao et al. 2021), which uses
Nystr¨om’s method to approximate the attention matrix;
MambaMIL (Yang, Wang, and Chen 2024), which lever-
ages the Mamba state space model (Gu and Dao 2023);
and RRTMIL (Tang et al. 2024), which adapts the Swin
Transformer (Liu et al. 2021). While computationally efﬁ-
cient, each carries its own limitations. TransMIL’s Nystr¨om
approximation with pooling-based landmarks risks diluting
key signals. MambaMIL is constrained by the ﬁxed 1D se-
quential processing of the Mamba architecture, and RRT-
MIL’s performance is sensitive to its window conﬁguration
and parameter count. Critically, these methods share a com-
mon ﬂaw: they are key-instance agnostic. By treating all in-
stances uniformly during interaction, they risk overlooking
the sparse yet crucial diagnostic signals present in WSIs,
leading to suboptimal outcomes.

Linear Projection

Partition

...

Gated MLP

Sort
Descending

�1 �2
��
��
...
...

...

KGGA
Candidate Key

Sub-Features

�1 �2
��
...

Initial Scores

...

�4

�1

�2
�3

�5

��

•
Candidate key sub-instances guide the global interaction. 
•
Final score derived via coupling global and initial scores.

KGGA

SDA

...
...
...

�1

�2

��−1

��

...
...

Feature 
Extractor

Whole Slide Image

...

...
�1

�2

��−1

��

...
...

Classifier

Linear Projection

Sub-feature with the k-th highest 
score

��
��

C
Concat

The k-th sub-space

The k-th highest score

��

The feature of a sub-instance (Sub-
feature)

The feature of an instance

...

Figure 2: Overview of our proposed CKMIL. CKMIL partitions instance features into multiple sub-spaces, where a sub-space
Discriminative Attention (SDA) module selects Candidate Key Sub-Features. These key candidates then drive a Global Interac-
tion with all sub-features in their respective space to generate an aggregated Sub-Feature, achieving efﬁcient and key-instance
guided global interaction (KGGA). Finally, all aggregated sub-features are concatenated to form the ﬁnal bag-level feature.

Methodology
The CKMIL framework is engineered to resolve the im-
passe where methods either neglect instance correlations or
are key-instance agnostic. It leverages a cascaded process
that uses key instances to guide global interaction, thereby
achieving robust correlation modeling and preventing the di-
lution of critical diagnostic signals in WSIs.

Problem formulation
Taking binary classiﬁcation in MIL as an example, to utilize
bag-level label Yi, for i = 1, 2, · · · , b, Yi ∈{0, 1}, we have
the corresponding instance feature set for each bag Xi ∈
Rn×D = {xi,1, · · · , xi,k, · · · , xi,n}, for i = 1, 2, · · · , b.
The MIL methodology can be represented as follows:

Yi =

0,
if Pn

k=1 yi,k = 0
1,
otherwise
(1)

ˆYi = f(Xi),
(2)

where yi,k ∈{0, 1} is the unknown instance-level label, ˆYi
in Eq.2. is the predicted value we obtain using bag Xi, b is
the number of WSIs, and n is the number of instances in
each bag (the value of n can vary for different bags). The
function f is what needs to be designed in MIL. Its main
component is the aggregator, whose role is to aggregate in-
stance features xi,1, · · · , xi,k, · · · , xi,n into a bag-level fea-
ture ˜xi. This feature is then fed into a classiﬁcation head
to obtain the prediction ˆYi, Unlike global interaction meth-
ods such as TransMIL (Shao et al. 2021), where the de-
signed function f causes instance-level features to change
after global interaction, our proposed CKMIL, despite hav-
ing global interaction, does not alter the instance-level fea-
tures themselves. As our comparative experiments show, our
approach, when used in two-stage MIL with feature en-
coders pre-trained on general domain images (like ResNet50
on ImageNet), avoids further distortion and loss of feature
information and outperforms other approaches.

Overview of CKMIL
The CKMIL framework, illustrated in Figure 2, operates
through a cascaded process designed to leverage sparse di-
agnostic signals for efﬁcient global attention. Initially, in-
stance features are partitioned into multiple subspaces where
the SDA module performs a screening to identify a set of
candidate key sub-instances with high discriminative scores.
These candidates are then utilized by the KGGA module as
landmarks for Nystr¨om-based attention (Xiong et al. 2021),
facilitating an efﬁcient global interaction explicitly guided
by high-value signals. Subsequently, the global context from
KGGA modulates the initial scores from SDA to obtain
the global scores, then via a gated fusion mechanism to
produce reﬁned ﬁnal scores. These ﬁnal scores guide the
weighted aggregation within each subspace, and the result-
ing sub-features are concatenated to form the ﬁnal bag-
level representation. Additionally, the framework includes
the Instance-Conv-Projection (ICP) module in an attempt to
capture local intra-feature correlations. This component ex-
plores using convolutional fusion instead of standard linear
projections for generating Query (Q) and Key (K) vectors.

Subspace-Disentangled Attention (SDA)
To mitigate the risk of attention focusing on non-critical re-
gions and to encourage feature diversity, inspired by the lo-
cal attention within multi-head spaces in ABMILX (Tang
et al. 2025), we propose SDA, the SDA module parti-
tions instance features and screens for key signals indepen-
dently within each subspace. Given a set of instances for
a bag X ∈Rn×D = {x1, · · · , xk, · · · , xn}, we ﬁrst par-
tition the features of the instances in the bag into m dif-
ferent low-dimensional feature subspaces, obtaining a col-
lection of bags in different feature subspaces, denoted as
{X1, · · · , Xh, · · · , Xm}, Xh ∈Rn× D

m , for h = 1, · · · , m.
For a given subspace Hh, an independent gated MLP layer:

AT

h = Gh · [Wh(tanh(EhXT

h )) ⊙σ(UhXT

h )] ∈R1×n,
(3)

Candidate Key

Sub-Features

...
Original 
Sub-Features

Initial Scores

�  Key Landmarks

�  Sub-Instances

Correlation

Matrix

Correlation

Matrix

Self-Attention

Matrix

K
Q

Global
Attention

means 
Moore-Penrose 
pseudoinverse.

Gate Fusion

Aggregatation

Aggregated 
Sub-Feature

Final Scores

...

Global Refined

Scores

Figure 3: Our proposed KGGA reﬁnes initial weights by
globally interacting with key candidate sub-instances from
the SDA module to embed instance correlation.

computes initial scores Ah for all sub-instances, where
Gh ∈R1× D

4m , Wh ∈R
D
4m × D
4m , Eh ∈R
D
4m × D
m , Uh ∈
R

D
4m × D
m are trainable matrices, and D is the dimension of
the instances. Sub-instances are then ranked by these scores,
and the top-r are selected to form the candidate key set
Lh ∈Rr×(D/m) for that subspace:

(˜xh,1, ˜ah,1), · · · , (˜xh,r, ˜ah,r), · · · , (˜xh,n, ˜ah,n) =

SortDescending((xh,1, ah,1), · · · , (xh,n, ah,n)),
(4)

Lh = {˜xh,1, ˜xh,2, . . . , ˜xh,r} ∈Rr×(D/m),
(5)

where xh,i represents the sub-instance feature of the i-th in-
stance in the h-th feature subspace, ah,i represents the inde-
pendent weight score of the i-th instance in the h-th feature
subspace, ˜xh,i represents the sub-instance feature with the
i-th highest score in the h-th feature subspace, and Lh is the
candidate key sub-instances in the h-th feature subspace.

Key-Instance Guided Global Attention (KGGA)

To efﬁciently model the correlations among the vast num-
ber of instances in a WSI, we adopt the Nystr¨om attention
mechanism (Xiong et al. 2021). This method achieves a lin-
ear O(n) complexity by constructing a low-rank approxima-
tion of the full attention matrix. The mathematical founda-
tion for this is the CUR matrix decomposition. This prin-
ciple approximates a large matrix by using a subset of its
actual columns (C) and rows (R), along with a smaller,
low-dimensional core matrix (U), to reconstruct an approx-
imation of the original matrix (i.e., A ≈CUR). However,
a critical challenge lies in the landmark selection strategy.
Conventional Nystr¨om Attention implementations typically
select these landmarks using pooling-based strategies. The
core matrix (approximating U) is then formed from the self-
attention matrix computed among these pooled landmarks.
While this process effectively reduces computational com-
plexity, the approach is fundamentally key-instance agnos-
tic, which risks diluting the sparse yet crucial diagnostic sig-
nals within the WSI.

To address the key-agnostic nature, the KGGA module
is designed (as illustrated in Figure 3). In contrast to the
method based on average pooling, it leverages the candidate
key sub-instances Lh from SDA as landmarks for Nystr¨om

attention, ensuring that global interaction is anchored by di-
agnostically relevant signals. The computation of the ap-
proximated global attention matrix ˆSh is described as:

ˆSh = softmax

QKT

Lh
p

D/m

(M)† softmax

QLhKT

D/m

,
(6)

M = softmax

QLhKT

Lh
p

D/m

,
(7)

where QLh and KLh are the query and key matrices corre-
sponding to the Lh landmarks, and M† denotes the Moore-
Penrose pseudoinverse of M.

The initial scores Ah obtained from the SDA module
fail to adequately consider the correlations among instances.
Therefore, to generate the global-aware scores Bh while
maintaining a computational complexity of O(n), we apply
the associative law of multiplication to left-multiply ˆSh by
the initial scores Ah, resulting in the following expression:

Φ1 = softmax

QKT

Lh
p

D/m

, Φ2 = softmax

QLhKT

D/m

,
(8)

Bh = ˆSh · Ah = [Φ1 (M)†]n×m [Φ2Ah]m×1.
(9)

To create a synergistic coupling between the screening
(SDA) and interaction (KGGA) stages, a gated mechanism
fuses the initial scores Ah and the global reﬁned scores Bh
into ﬁnal scores Ch :

g = σ (XhWg) ∈Rn×1,
(10)

Ch = (1 −g) ⊙Ah + g ⊙Bh,
(11)

where Wg is a trainable matrix and σ means the Sigma
function. These ﬁnal scores guide the weighted aggregation
of sub-features into a subspace representation Zh for down-
stream task analysis:

Zh = softmax

CT

Xh.
(12)

Finally, all subspace representations are concatenated to
form the bag-level feature Z:

Z = concat (Z1, . . . , Zh, . . . , Zm).
(13)

Instance-Conv-Projection (ICP)
Conventional attention mechanisms generate Query (Q) and
Key (K) vectors using linear projections, which have weak
capabilities in modeling the local, intra-feature correlations
crucial in pathology. To address this, the ICP module inte-
grates the local fusion capabilities of convolutions.

As shown in Figure 4, ICP implements a Reshape-
Convolution-Reshape-Projection pipeline. An input 1D in-
stance feature xi ∈R1×D is ﬁrst reshaped (R) into a 2D
pseudo-image. A lightweight convolutional layer then pro-
cesses this tensor, capturing local structural patterns imper-
ceptible to a standard linear layer. The tensor is then ﬂat-
tened back (F) to a 1D vector and projected to generate the
ﬁnal Qi or Ki vector:

Qi (Ki) = Linear (F (Conv (R (xi)))).
(14)

Methods
BRACS-3
BRCA-2
NSCLC-2

AUC
ACC
AUC
ACC
AUC
ACC

ResNet-50

Mean-Pooling
0.8051±0.0319
0.6444±0.0337
0.9068±0.0276
0.8410±0.0262
0.8914±0.0203
0.8209±0.0282
Max-Pooling
0.8064±0.0359
0.6907±0.0356
0.8372±0.0239
0.8152±0.0260
0.9163±0.0314
0.8342±0.0340

ABMIL (Ilse, Tomczak, and Welling 2018)
0.8004±0.0382
0.6981±0.0368
0.8883±0.0190
0.8139±0.0401
0.9359±0.0276
0.8685±0.0463
CLAM-MB (Lu et al. 2021)
0.8134±0.0287
0.6833±0.0280
0.8929±0.0177
0.8210±0.0232
0.9407±0.0207
0.8685±0.0266
DSMIL (Li, Li, and Eliceiri 2021)
0.7950±0.0365
0.6481±0.0476
0.8196±0.0766
0.7809±0.0540
0.8491±0.0779
0.7561±0.0701

TransMIL (Shao et al. 2021)
0.8160±0.0406
0.7111±0.0200
0.8774±0.0386
0.8145±0.0445
0.9348±0.0192
0.8495±0.0415
MambaMIL (Yang, Wang, and Chen 2024)
0.8305±0.0427
0.7111±0.0553
0.8949±0.0375
0.8632±0.0273
0.9374±0.0190
0.8743±0.0302
RRTMIL (Tang et al. 2024)
0.8160±0.0257
0.7129±0.0185
0.9163±0.0290
0.8484±0.0386
0.9421±0.0146
0.8723±0.0136

CKMIL-Base (ours)
0.8483±0.0260
0.7130±0.0515
0.9269±0.0358
0.8716±0.0274
0.9439±0.0225
0.8752±0.0317
CKMIL (ours)
0.8583±0.0297
0.7370±0.0427
0.9255±0.0261
0.8648±0.0252
0.9549±0.0148
0.8838±0.0253

UNI

Mean-Pooling
0.8771±0.0259
0.7203±0.0411
0.9552±0.0258
0.8943±0.0237
0.9746±0.0122
0.9257±0.0219
Max-Pooling
0.8596±0.0285
0.7503±0.0101
0.9627±0.0190
0.9136±0.0109
0.9816±0.0109
0.9361±0.0246

ABMIL (Ilse, Tomczak, and Welling 2018)
0.8901±0.0426
0.7635±0.0567
0.9671±0.0240
0.9187±0.0106
0.9796±0.0118
0.9485±0.0197
CLAM-MB (Lu et al. 2021)
0.8862±0.0343
0.7629±0.0456
0.9625±0.0176
0.9291±0.0186
0.9825±0.0117
0.9409±0.0183
DSMIL (Li, Li, and Eliceiri 2021)
0.8399±0.0169
0.7185±0.0266
0.9533±0.0124
0.8900±0.0129
0.9739±0.0129
0.9200±0.0278

TransMIL (Shao et al. 2021)
0.8549±0.0226
0.7407±0.0340
0.9488±0.0293
0.9195±0.0172
0.9766±0.0124
0.9190±0.0187
MambaMIL (Yang, Wang, and Chen 2024)
0.8842±0.0234
0.7645±0.0292
0.9568±0.0234
0.9099±0.0221
0.9791±0.0120
0.9352±0.0204
RRTMIL (Tang et al. 2024)
0.8754±0.0284
0.7574±0.0583
0.9586±0.0221
0.9178±0.0153
0.9818±0.0115
0.9323±0.0182

CKMIL-Base (ours)
0.8967±0.0275
0.7648±0.0274
0.9579±0.0192
0.9160±0.0253
0.9756±0.0086
0.9342±0.0169
CKMIL (ours)
0.8952±0.0203
0.7648±0.0258
0.9556±0.0208
0.9125±0.0274
0.9836±0.0103
0.9361±0.0234

Table 1: Performance comparison on cancer subtyping tasks. Best results are in bold, and second-best results are underlined.

Convolutional

Projection

...
reshape

pad

for each

instance

Bag 
Features

...
flatten

...
flatten

Key

Convolutional

Projection

Query
Linear

Projection

Linear

Projection

Instance

Feature

Figure 4: The framework of ICP following a Reshape-
Convolution-Reshape-Projection pipeline.

Experiments
Datasets and Evaluation Metrics
To validate the efﬁcacy of our proposed CKMIL, we con-
ducted extensive experiments on two representative down-
stream tasks across four public datasets.

Survival Prediction We selected three public cancer
datasets from The Cancer Genome Atlas (TCGA) (Wein-
stein et al. 2013): BLCA, BRCA, and LUAD, which con-
tain Whole Slide Images (WSIs) with corresponding sur-
vival time annotations. Following the experimental setup of
GPFM (Ma et al. 2024), we employ a 5-fold cross-validation
methodology to mitigate the impact of data partitioning on
model evaluation, splitting the data into training and valida-
tion sets at a 4:1 ratio. We utilize the cross-validated Con-
cordance Index (C-Index), with its standard deviation (std).

Cancer Subtyping We also conduct experiments on
three challenging public datasets: BRACS (Brancati et al.
2022), and the NSCLC and BRCA cohorts from the TCGA
database (Weinstein et al. 2013). For dataset partitioning, we
follow the protocol from GPFM (Ma et al. 2024), splitting
the data into training, validation, and testing sets at a 7:1:2

ratio. To ensure a robust evaluation, we generate 5 differ-
ent random splits with this ratio for our experiments. For
evaluation, we adopt the Area Under the Curve (AUC) and
Accuracy (ACC) metrics, reporting their mean and standard
deviation (std). Supplementary Material offers more details.

Comparison Methods and Training Details
Comparison Methods We compare our proposed meth-
ods CKMIL-base and CKMIL, against several categories
of methods:(1) Simple Pooling Methods: Mean-Pooling and
Max-Pooling; (2) Attention-Based Methods: ABMIL (Ilse,
Tomczak, and Welling 2018), and its variants CLAM-MB
(Lu et al. 2021) and DSMIL (Li, Li, and Eliceiri 2021); (3)
Global Interaction Methods with Linear Complexity: Trans-
MIL (Shao et al. 2021), MambaMIL (Yang, Wang, and Chen
2024), and RRTMIL (Tang et al. 2024).
Comparison of CKMIL-Base and CKMIL The distinc-
tion between our two models lies solely in the Q/K vec-
tor generation: CKMIL-Base uses conventional linear lay-
ers, while CKMIL incorporates our exploratory ICP module.

Training Details Patches of size 256×256 were cropped
at 20× magniﬁcation WSIs without overlap. To extract patch
features, we utilized two ofﬂine encoders: a ResNet50 (He
et al. 2016) pre-trained on ImageNet (Deng et al. 2009)
for general visual representations, and the UNI (Chen et al.
2024) model, which was self-supervised on a pancancer co-
hort to learn domain-speciﬁc pathology features. Supple-
mentary Material offers more training details.

Results and Analysis
Tables 1 and 2 provide a comprehensive performance com-
parison of various MIL methods on cancer subtyping and
survival prediction, utilizing two distinct feature extractors:
the general-purpose ResNet50 (He et al. 2016) and the
domain-speciﬁc UNI (Chen et al. 2024).

Methods
BLCA (C-index)
BRCA (C-index)
LUAD (C-index)

ResNet-50
UNI
ResNet-50
UNI
ResNet-50
UNI

Mean-Pooling
0.5870±0.0583
0.5989±0.0129
0.6135±0.0631
0.6777±0.0602
0.6095±0.0820
0.6276±0.0623
Max-Pooling
0.5589±0.0593
0.5742±0.0476
0.5754±0.0382
0.6119±0.0522
0.6063±0.0396
0.5951±0.0069

ABMIL (Ilse, Tomczak, and Welling 2018)
0.5503±0.0986
0.6035±0.0491
0.6103±0.0739
0.6688±0.0534
0.6015±0.0767
0.6240±0.0762
CLAM-MB (Lu et al. 2021)
0.5695±0.0951
0.5975±0.0445
0.5887±0.0592
0.6701±0.0413
0.6165±0.0761
0.6265±0.0490
DSMIL (Li, Li, and Eliceiri 2021)
0.5774±0.0588
0.5885±0.0536
0.6199±0.0297
0.6460±0.0346
0.6147±0.0250
0.5496±0.0594

TransMIL (Shao et al. 2021)
0.6055±0.0485
0.6119±0.0312
0.6158±0.0559
0.6163±0.0360
0.6335±0.0347
0.6222±0.0615
MambaMIL (Yang, Wang, and Chen 2024)
OOM
OOM
0.6524±0.0494
0.6480±0.0399
0.6452 ± 0.0168
0.6142±0.0580
RRTMIL (Tang et al. 2024)
OOM
OOM
0.6445±0.0604
0.6500±0.0503
0.6231±0.0490
0.6303±0.0687

CKMIL-Base (ours)
0.6287±0.0429
0.6038±0.0349
0.6440±0.0794
0.6920±0.0717
0.6820±0.0267
0.6300±0.0267
CKMIL (ours)
0.6185±0.0406
0.6155±0.0429
0.6825±0.0887
0.6869±0.0661
0.6467±0.0402
0.6380±0.0640

Table 2: Performance comparison on survival prediction tasks. Best results are in bold, and second-best results are underlined.
OOM denotes out of memory in the experiment settings.

When benchmarked with the ResNet50 feature extractor,
our CKMIL models achieve state-of-the-art (SOTA) perfor-
mance across all tasks and datasets. Notably, the full CK-
MIL model consistently outperforms all competing meth-
ods, with CKMIL-Base being the only exception. For in-
stance, as shown in Table 1, our CKMIL model demonstrates
signiﬁcant improvements on the BRACS-3 subtyping task,
outperforming the strong baseline RRTMIL with a 2.78%
improvement in AUC and 2.01% in ACC. This superiority
extends to survival prediction tasks. On the LUAD cohort,
our CKMIL-Base model sets a new SOTA with a C-Index
of 0.6820. Meanwhile, on the BRCA survival task, CKMIL
achieves a C-Index of 0.6825, a substantial 3.81% improve-
ment over the next-best comparable method. This ﬁnding is
particularly signiﬁcant, as it validates our core hypothesis
that an effective aggregation mechanism can overcome the
limitations of non-domain-speciﬁc features by effectively
modeling instance correlations.

When using the pathology-speciﬁc UNI feature extrac-
tor, our models achieve new SOTA results across all sur-
vival prediction tasks. However, in certain subtyping tasks,
such as on the BRCA dataset (for both AUC and ACC)
and the NSCLC dataset (for ACC), the performance of
methods that model inter-instance correlations, including
ours, was surpassed by simpler approaches like ABMIL
and CLAM. We hypothesize that this phenomenon occurs
because UNI generates features that are already highly
discriminative. For such strong features, explicitly model-
ing correlations might introduce noise from redundant in-
stances, which inadvertently dilutes the weights or the fea-
tures themselves of sparse, critical instances, and thus de-
grades performance. Conversely, our models’ SOTA per-
formance with the generic ResNet50 extractor corroborates
the effectiveness of our correlation modeling, demonstrating
its ability to adapt general-purpose features for specialized
medical analysis through guided interaction.

Ablation Study and Sensitivity Analysis

To rigorously validate the effectiveness of our proposed CK-
MIL framework, we conduct a series of ablation studies on
its core components: the Subspace-Disentangled Attention
(SDA), the Key-Instance Guided Global Attention (KGGA),
and the Instance-Conv-Projection (ICP) module. We per-
form quantitative evaluations on the BRACS-3 cancer sub-

typing task (reporting mean AUC and ACC) and the TCGA-
BRCA survival prediction task (reporting mean C-Index),
using ResNet50 as the feature extractor and following the
same experimental protocol as in the main experiments.

Effectiveness of Subspace-Disentangled Attention (SDA)
The SDA module is designed to screen for key instances
within multiple disentangled feature subspaces. To isolate
its contribution, we conduct two sets of experiments:

• CKMIL vs. CKMIL (m = 1): We reduce the number of
subspaces in the SDA module to one (i.e., m = 1). This
variant, denoted as CKMIL (m = 1) or ABMIL+KGGA,
replaces SDA with a single, shared attention layer akin to
ABMIL, while keeping the KGGA module.
• ABMIL (Ilse, Tomczak, and Welling 2018) vs. AB-
MIL+SDA: To demonstrate that the multi-subspace
scoring mechanism is inherently superior to a single at-
tention layer, we integrate the SDA module into the stan-
dard ABMIL framework, creating ABMIL+SDA.
As presented in Table 3, ABMIL+SDA consistently sur-
passes ABMIL across all metrics. Similarly, CKMIL out-
performs the original CKMIL (m = 1) across all metrics,
further validating that the multi-subspace scoring design is a
more effective strategy than a single shared attention layer.

Model
BRACS-3 (AUC ↑)
BRACS-3 (ACC ↑)
BRCA (C-Index ↑)

ABMIL
0.8004
0.6981
0.6103
ABMIL+SDA
0.8423 (+4.19%)
0.7074 (+0.93%)
0.6131 (+0.28%)

CKMIL (m = 1)
0.8454
0.7240
0.6687
CKMIL (ours)
0.8583 (+1.29%)
0.7370 (+1.30%)
0.6825 (+1.38%)

Table 3: Ablation study on the effectiveness of SDA.

Effectiveness of Key-Instance Guided Global Attention
(KGGA)
The KGGA module is premised on the principle
that global interaction should be guided by key instances.
We validate its efﬁcacy through the following experiments:

• CKMIL vs. CKMIL (Pooling): We replace the key-
instance-guided landmark selection in KGGA with a
conventional mean pooling strategy to select landmarks,
a method similar to that used in TransMIL.
• ABMIL (Ilse, Tomczak, and Welling 2018) vs. AB-
MIL+KGGA: To demonstrate the importance of the
global interaction mechanism itself, we augment the

GT
CLAM
ABMIL
CKMIL (ours)

Figure 5: Global attention heatmap comparison on a WSI
from the BRACS dataset.

baseline ABMIL with our KGGA module which is
equivalent to the CKMIL (m=1) variant.
• TransMIL (Shao et al. 2021) vs. TransMIL+KGGA:
To show that our key-instance-guided approach is supe-
rior, we modify TransMIL by ﬁrst adding an attention
layer to score instances and then using the top-scoring in-
stances as landmarks for its global interaction. We term
this variant TransMIL+KGGA.

As shown in Table 4, CKMIL signiﬁcantly outperforms CK-
MIL (Pooling), conﬁrming our hypothesis that using candi-
date key instances as landmarks is more effective than using
landmarks derived from pooling. The comparison between
ABMIL and ABMIL+KGGA shows that incorporating our
KGGA module brings substantial performance gains across
all tasks, underscoring the necessity of modeling global
inter-instance correlations. Finally, TransMIL+KGGA sur-
passes the original TransMIL, further proving that a key-
instance-guided strategy is a more powerful approach for
global attention in MIL.

Model
BRACS-3 (AUC ↑)
BRACS-3 (ACC ↑)
BRCA (C-Index ↑)

ABMIL
0.8004
0.6981
0.6103
ABMIL+KGGA
0.8454 (+4.50%)
0.7240 (+2.59%)
0.6687 (+5.84%)

TransMIL
0.8160
0.7111
0.6158
TransMIL+KGGA
0.8297 (+1.37%)
0.7278 (+1.67%)
0.6281 (+1.23%)

CKMIL (Pooling)
0.8477
0.7185
0.6445
CKMIL (ours)
0.8583 (+1.06%)
0.7370 (+1.85%)
0.6825 (+3.80%)

Table 4: Ablation study on the effectiveness of KGGA.

Effectiveness of Instance-Conv-Projection (ICP)
The
ICP module is designed based on the hypothesis that local
correlations exist among the components of an instance’s
feature vector. To investigate the feasibility of this ex-
ploratory module, we conducted a comprehensive compar-
ison between the full CKMIL model and the CKMIL-Base
model which uses standard linear projections across all tasks
and datasets. The detailed results, presented in Table 1 and
2, reveal that the ICP module offers clear beneﬁts in speciﬁc
contexts. For instance, when using ResNet50 features for the
BRACS subtyping task, CKMIL shows a 2.4% improvement
in ACC and a 1.0% improvement in AUC over CKMIL-
Base. Similarly, for survival prediction on the LUAD cohort
with UNI features, CKMIL yields a 0.8% improvement in
C-Index. On the TCGA-BRCA survival prediction task, the
beneﬁt is even more pronounced, with CKMIL delivering
a 3.85% higher C-Index than CKMIL-Base. However, on
other datasets, the impact of ICP is more varied and appears

ABMIL
CLAM-MB
CKMIL (ours)

GT

positive

negative

Figure 6: Local attention heatmap comparison on a WSI
from the BRACS dataset.

to be inﬂuenced by the choice of the upstream feature ex-
tractor. This suggests that while ICP can effectively capture
latent intra-feature correlations, the prominence and utility
of these correlations may depend on the speciﬁc dataset and
the nature of the features generated by the encoder.

The sensitivity analysis for key hyperparameters and the
attention heatmaps of the ablation study are provided in the
Supplementary Material.

Visualization Results
Fundamentally, our proposed CKMIL is an Attention-Based
method. To evaluate CKMIL’s interpretability and localiza-
tion capability, we visualize its attention heatmaps against
baseline methods ABMIL and CLAM-MB, comparing them
to ground truth (GT) annotations provided by pathologists.

As shown in the global view (Figure 5), the attention
from ABMIL and CLAM-MB is diffuse and highlights non-
diagnostic areas, failing to localize the scattered tumor re-
gions indicated by the GT. In contrast, CKMIL produces
precise, concentrated heatmaps that show high concordance
with GT annotations, successfully identifying multiple key
tumor clusters. This is due to the synergy between our SDA
and KGGA modules, which suppresses non-critical regions.

The superiority of CKMIL is apparent in the local view
(Figure 6), while baseline methods fail to focus on core
pathological cell structures, CKMIL’s high-attention areas
precisely cover the dense, diagnostically relevant cell re-
gions, as conﬁrmed by GT. This demonstrates the effective-
ness of our key-instance-guided mechanism in identifying
the most informative regions within a WSI.

Conclusion
In this work, we proposed CKMIL, a novel cascaded at-
tention framework for WSI analysis that addresses the key
information dilution problem in existing MIL methods. By
ﬁrst identifying key instances with a SDA module and then
using them to guide an efﬁcient global interaction via our
KGGA module, CKMIL achieves a more focused and ef-
fective aggregation. Extensive experiments demonstrate that
our approach sets a new state-of-the-art in cancer subtyping
and survival prediction, proving the effectiveness of a key-
instance-aware mechanism in computational pathology.

References

Amores, J. 2013. Multiple instance classiﬁcation: Review,
taxonomy and comparative study.
Artiﬁcial intelligence,
201: 81–105.
Bera, K.; Schalper, K. A.; Rimm, D. L.; Velcheti, V.; and
Madabhushi, A. 2019.
Artiﬁcial intelligence in digital
pathology—new tools for diagnosis and precision oncology.
Nature reviews Clinical oncology, 16(11): 703–715.
Brancati, N.; Anniciello, A. M.; Pati, P.; Riccio, D.; Scog-
namiglio, G.; Jaume, G.; De Pietro, G.; Di Bonito, M.; Fon-
cubierta, A.; Botti, G.; et al. 2022. Bracs: A dataset for breast
carcinoma subtyping in h&e histology images. Database,
2022: baac093.
Cai, Z.; Song, H.; Fingerhut, A.; Sun, J.; Ma, J.; Zhang, L.;
Li, S.; Yu, C.; Zheng, M.; and Zang, L. 2021. A greater
lymph node yield is required during pathological examina-
tion in microsatellite instability-high gastric cancer. BMC
cancer, 21(1): 319.
Campanella, G.; Hanna, M. G.; Geneslaw, L.; Miraﬂor, A.;
Werneck Krauss Silva, V.; Busam, K. J.; Brogi, E.; Reuter,
V. E.; Klimstra, D. S.; and Fuchs, T. J. 2019.
Clinical-
grade computational pathology using weakly supervised
deep learning on whole slide images.
Nature medicine,
25(8): 1301–1309.
Chen, R. J.; Ding, T.; Lu, M. Y.; Williamson, D. F.; Jaume,
G.; Chen, B.; Zhang, A.; Shao, D.; Song, A. H.; Shaban, M.;
et al. 2024. Towards a General-Purpose Foundation Model
for Computational Pathology. Nature Medicine.
Chen, Z.; Chi, Z.; Fu, H.; and Feng, D. 2013. Multi-instance
multi-label image classiﬁcation: A neural approach. Neuro-
computing, 99: 298–306.
Cifci, D.; Veldhuizen, G. P.; Foersch, S.; and Kather, J. N.
2023. AI in computational pathology of cancer: improving
diagnostic workﬂows and clinical outcomes? Annual Review
of Cancer Biology, 7(1): 57–71.
Coudray, N.; Ocampo, P. S.; Sakellaropoulos, T.; Narula, N.;
Snuderl, M.; Feny¨o, D.; Moreira, A. L.; Razavian, N.; and
Tsirigos, A. 2018. Classiﬁcation and mutation prediction
from non–small cell lung cancer histopathology images us-
ing deep learning. Nature medicine, 24(10): 1559–1567.
Cui, M.; and Zhang, D. Y. 2021. Artiﬁcial intelligence and
computational pathology. Laboratory Investigation, 101(4):
412–422.
Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and Fei-
Fei, L. 2009. Imagenet: A large-scale hierarchical image
database. In 2009 IEEE conference on computer vision and
pattern recognition, 248–255. IEEE.
Elmore, J. G.; Longton, G. M.; Carney, P. A.; Geller, B. M.;
Onega, T.; Tosteson, A. N.; Nelson, H. D.; Pepe, M. S.; Al-
lison, K. H.; Schnitt, S. J.; et al. 2015. Diagnostic concor-
dance among pathologists interpreting breast biopsy speci-
mens. Jama, 313(11): 1122–1132.
Gu, A.; and Dao, T. 2023.
Mamba: Linear-time se-
quence modeling with selective state spaces. arXiv preprint
arXiv:2312.00752.

He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep resid-
ual learning for image recognition. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, 770–778.
Ilse, M.; Tomczak, J.; and Welling, M. 2018.
Attention-
based deep multiple instance learning. In International con-
ference on machine learning, 2127–2136. PMLR.
Jin, C.; Guo, Z.; Lin, Y.; Luo, L.; and Chen, H. 2023.
Label-efﬁcient deep learning in medical image analy-
sis: Challenges and future directions.
arXiv preprint
arXiv:2303.12484.
Li, B.; Li, Y.; and Eliceiri, K. W. 2021. Dual-stream multiple
instance learning network for whole slide image classiﬁca-
tion with self-supervised contrastive learning. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, 14318–14328.
Liu, Z.; Lin, Y.; Cao, Y.; Hu, H.; Wei, Y.; Zhang, Z.; Lin,
S.; and Guo, B. 2021. Swin transformer: Hierarchical vi-
sion transformer using shifted windows. In Proceedings of
the IEEE/CVF international conference on computer vision,
10012–10022.
Lu, M. Y.; Williamson, D. F.; Chen, T. Y.; Chen, R. J.; Bar-
bieri, M.; and Mahmood, F. 2021. Data-efﬁcient and weakly
supervised computational pathology on whole-slide images.
Nature biomedical engineering, 5(6): 555–570.
Ma, J.; Guo, Z.; Zhou, F.; Wang, Y.; Xu, Y.; Li, J.; Yan, F.;
Cai, Y.; Zhu, Z.; Jin, C.; et al. 2024. Towards a generalizable
pathology foundation model via uniﬁed knowledge distilla-
tion. arXiv preprint arXiv:2407.18449.
Maron, O.; and Lozano-P´erez, T. 1997. A framework for
multiple-instance learning. Advances in neural information
processing systems, 10.
Shao, Z.; Bian, H.; Chen, Y.; Wang, Y.; Zhang, J.; Ji, X.;
et al. 2021. Transmil: Transformer based correlated multiple
instance learning for whole slide image classiﬁcation. Ad-
vances in neural information processing systems, 34: 2136–
2147.
Song, A. H.; Jaume, G.; Williamson, D. F.; Lu, M. Y.;
Vaidya, A.; Miller, T. R.; and Mahmood, F. 2023. Artiﬁcial
intelligence for digital and computational pathology. Nature
Reviews Bioengineering, 1(12): 930–949.
Tang, W.; Qin, R.; Fang, H.; Zhou, F.; Chen, H.; Li, X.; and
Cheng, M.-M. 2025. Revisiting End-to-End Learning with
Slide-level Supervision in Computational Pathology. arXiv
preprint arXiv:2506.02408.
Tang, W.; Zhou, F.; Huang, S.; Zhu, X.; Zhang, Y.; and Liu,
B. 2024. Feature re-embedding: Towards foundation model-
level performance in computational pathology. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, 11343–11352.
Weinstein, J. N.; Collisson, E. A.; Mills, G. B.; Shaw, K. R.;
Ozenberger, B. A.; Ellrott, K.; Shmulevich, I.; Sander, C.;
and Stuart, J. M. 2013. The cancer genome atlas pan-cancer
analysis project. Nature genetics, 45(10): 1113–1120.
Xiong, Y.; Zeng, Z.; Chakraborty, R.; Tan, M.; Fung, G.; Li,
Y.; and Singh, V. 2021. Nystr¨omformer: A nystr¨om-based

algorithm for approximating self-attention. In Proceedings
of the AAAI conference on artiﬁcial intelligence, volume 35,
14138–14148.
Yang, S.; Wang, Y.; and Chen, H. 2024. Mambamil: En-
hancing long sequence modeling with sequence reordering
in computational pathology. In International conference on
medical image computing and computer-assisted interven-
tion, 296–306. Springer.
Yu, K.-H.; Zhang, C.; Berry, G. J.; Altman, R. B.; R´e, C.;
Rubin, D. L.; and Snyder, M. 2016. Predicting non-small
cell lung cancer prognosis by fully automated microscopic
pathology image features.
Nature communications, 7(1):
12474.

Reproducibility Checklist

1. General Paper Structure

1.1. Includes a conceptual outline and/or pseudocode de-
scription of AI methods introduced (yes/partial/no/NA)
yes

1.2. Clearly delineates statements that are opinions, hypoth-
esis, and speculation from objective facts and results
(yes/no) yes

1.3. Provides well-marked pedagogical references for less-
familiar readers to gain background necessary to repli-
cate the paper (yes/no) yes

2. Theoretical Contributions

2.1. Does
this
paper
make
theoretical
contributions?
(yes/no) no

If yes, please address the following points:

2.2. All assumptions and restrictions are stated clearly
and formally (yes/partial/no)

2.3. All novel claims are stated formally (e.g., in theorem
statements) (yes/partial/no)

2.4. Proofs of all novel claims are included (yes/par-
tial/no)

2.5. Proof sketches or intuitions are given for complex
and/or novel results (yes/partial/no)

2.6. Appropriate citations to theoretical tools used are
given (yes/partial/no)

2.7. All theoretical claims are demonstrated empirically
to hold (yes/partial/no/NA)

2.8. All experimental code used to eliminate or disprove
claims is included (yes/no/NA)

3. Dataset Usage

3.1. Does this paper rely on one or more datasets? (yes/no)
yes

If yes, please address the following points:

3.2. A motivation is given for why the experiments
are conducted on the selected datasets (yes/par-
tial/no/NA) yes

3.3. All novel datasets introduced in this paper are in-
cluded in a data appendix (yes/partial/no/NA) NA

3.4. All novel datasets introduced in this paper will be
made publicly available upon publication of the pa-
per with a license that allows free usage for research
purposes (yes/partial/no/NA) NA

3.5. All datasets drawn from the existing literature (po-
tentially including authors’ own previously pub-
lished work) are accompanied by appropriate cita-
tions (yes/no/NA) yes

3.6. All datasets drawn from the existing literature
(potentially
including
authors’
own
previously
published work) are publicly available (yes/par-
tial/no/NA) yes

3.7. All datasets that are not publicly available are de-
scribed in detail, with explanation why publicly
available alternatives are not scientiﬁcally satisﬁcing
(yes/partial/no/NA) NA

4. Computational Experiments

4.1. Does this paper include computational experiments?
(yes/no) yes

If yes, please address the following points:

4.2. This paper states the number and range of values
tried per (hyper-) parameter during development of
the paper, along with the criterion used for selecting
the ﬁnal parameter setting (yes/partial/no/NA) par-
tial

4.3. Any code required for pre-processing data is in-
cluded in the appendix (yes/partial/no) no

4.4. All source code required for conducting and analyz-
ing the experiments is included in a code appendix
(yes/partial/no) no

4.5. All source code required for conducting and ana-
lyzing the experiments will be made publicly avail-
able upon publication of the paper with a license
that allows free usage for research purposes (yes/-
partial/no) yes

4.6. All source code implementing new methods have
comments detailing the implementation, with refer-
ences to the paper where each step comes from (yes/-
partial/no) partial

4.7. If an algorithm depends on randomness, then the
method used for setting seeds is described in a way
sufﬁcient to allow replication of results (yes/par-

tial/no/NA) yes

4.8. This paper speciﬁes the computing infrastructure
used for running experiments (hardware and soft-
ware), including GPU/CPU models; amount of
memory; operating system; names and versions of
relevant software libraries and frameworks (yes/par-
tial/no) partial

4.9. This paper formally describes evaluation metrics
used and explains the motivation for choosing these
metrics (yes/partial/no) yes

4.10. This paper states the number of algorithm runs used
to compute each reported result (yes/no) yes

4.11. Analysis
of
experiments
goes
beyond
single-
dimensional summaries of performance (e.g., aver-
age; median) to include measures of variation, con-
ﬁdence, or other distributional information (yes/no)
yes

4.12. The signiﬁcance of any improvement or decrease in
performance is judged using appropriate statistical
tests (e.g., Wilcoxon signed-rank) (yes/partial/no) no

4.13. This paper lists all ﬁnal (hyper-)parameters used
for each model/algorithm in the paper’s experiments
(yes/partial/no/NA) yes