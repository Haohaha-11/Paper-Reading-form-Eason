Contents lists available at ScienceDirect

Medical Image Analysis

journal homepage: www.elsevier.com/locate/media

Towards a universal JPEG lossless recompression foundation model for 
pathology images: A transformer context modeling approach$

Tao Song a,c,1, Rong Tao b,1, Chunyan Wu b, Mengmeng Zhao b, Jiajun Deng b, Yi Guo a, Feng Xu a, 
Chang Chen b,∗, Kun Qian d,e
,∗∗

a School of Information Science and Technology, Fudan university, Shanghai, China
b Department of Thoracic Surgery, Shanghai Pulmonary Hospital, Tongji University School of Medicine, Shanghai, China
c Shanghai Artificial Intelligence Lab, Shanghai, China
d SenseTime Research, Hong Kong Special Administrative Region of China
e CHUK MMLab, Hong Kong Special Administrative Region of China

A R T I C L E  I N F O

Keywords:
Pathology image
JPEG lossless recompression
Foundation model
Transformer-based

A B S T R A C T

Lossless recompression of JPEG images remains fundamentally constrained by the limited modeling capacity 
of traditional context-mixing entropy estimators, yielding suboptimal compression ratios. Recently, CNN-
based learned recompression methods have demonstrated improved entropy modeling by exploiting the strong 
representational capacity of deep networks. However, their reliance on local convolutional operations restricts 
long-range dependency modeling and limits generalization across diverse image domains. In this study, we 
introduce a Universal Pathology JPEG Lossless Recompression Foundation Model (ULRFM), a transformer-
based architecture explicitly designed to build long-range contextual dependencies within JPEG DCT coefficient 
streams. Leveraging a large-scale pathology dataset comprising more than nine million image tiles across 
multiple cancers and multiple organs, we systematically investigate the effects of model capacity and data 
quantity on lossless recompression performance. Extensive experiments demonstrate that ULRFM substantially 
outperforms existing CNN-based learned recompression approaches in both compression efficiency and cross-
distribution generalization. ULRFM provides a maximum file size reduction of 34.13% relative to the original 
JPEG format, highlighting its potential to markedly alleviate the growing storage burden in digital pathology 
infrastructures.

1. Introduction

Pathological whole-slide images (WSIs) are typically have gigapixel 
resolution, imposing substantial burdens on storage and data transmis-
sion infrastructure. To alleviate these demands, manufacturers com-
monly employ standard lossy JPEG compression under strict diagnostic 
quality constraints (Goode et al., 2013). Although this reduces storage 
to several extent, the rapid accumulation of pathology images continues 
to exert pressure on storage systems. As further lossy compression is 
unsuitable due to potential degradation of diagnostic fidelity, applying 
a secondary lossless recompression to already JPEG compressed images 
has become increasingly crucial in modern digital pathology pipelines.

$ This article is part of a Special issue entitled: ‘Foundation Models for Computational Pathology’ published in Medical Image Analysis.

∗Corresponding author.
∗∗Corresponding author at: SenseTime Research, Hong Kong Special Administrative Region of China.

E-mail addresses: changchenc@tongji.edu.cn (C. Chen), kqian@cuhk.edu.hk (K. Qian).
1 Equal Contribution.
2 https://www.byronknoll.com/cmix.html

JPEG compression (Wallace, 2002) quantizes luma and chroma 
components in the discrete cosine transform (DCT) domain and subse-
quently encodes the quantized coefficients using Huffman coding (Huff-
man, 2006). Two factors inherently limit the efficiency of this pipeline: 
(1) the quantization process introduces additional data redundancy, 
and (2) Huffman coding (Huffman, 2006) is less efficient than more 
advanced entropy coding schemes such as arithmetic coding (Witten 
et al., 1987) or asymmetric numeral systems (ANS) (Duda, 2013; 
Duda et al., 2015). Consequently, JPEG bitstreams retain considerable 
potential for further compression. Existing solutions attempt to exploit 
this potential through improved coding strategies or context-mixing 
mechanisms, such as Lepton (Horn et al., 2017), JPEG XL (Alakuijala 
et al., 2019, 2020), and CMIX.2 Among them, Lepton and JPEG XL are

https://doi.org/10.1016/j.media.2026.104152
Received 17 December 2025; Received in revised form 20 April 2026; Accepted 31 May 2026

Medical Image Analysis 113 (2026) 104152

Available online 5 June 2026 
1361-8415/© 2026 Published by Elsevier B.V.

T. Song et al.

specifically developed for JPEG recompression, whereas CMIX is a gen-
eral purpose compressor. Despite their effectiveness, these approaches 
rely heavily on hand crafted heuristics, fundamentally constraining 
their ultimate compression performance.

Recent advances in learned JPEG lossless recompression have been 
driven by convolutional neural networks (CNNs).  Guo et al. (2022) 
introduced a CNN-based Laplacian entropy model combined with mod-
ern coding frameworks, achieving measurable reductions in bitrates. 
Fan et al. (2022) employed an end-to-end learnable lossy transform 
coding architecture to reduce redundancy within the DCT domain. 
Eff-Net (Guo et al., 2023) proposed a multi-level parallel conditional 
modeling network that constructs Gaussian entropy models for luma 
and chroma components, thereby improving compression ratios while 
maintaining low latency. More broadly, large-scale foundation mod-
els H. Wang et al. (2025), Liu et al. (2024), Zhang and Metaxas 
(2024), Zhang et al. (2026), X. Wang et al. (2025, 2024) have achieved 
remarkable success across natural language processing, computer vi-
sion, and biomedical image analysis, demonstrating strong capabilities 
in representation learning, transferability, and cross-domain general-
ization. In computational pathology, such models have also shown 
promise in diagnostic and prognostic tasks by capturing morphology-
aware representations from gigapixel whole-slide images. Although 
CNN-based models outperform traditional approaches on natural image 
benchmarks, they remain limited by two major factors: (1) CNNs 
predominantly model local spatial dependencies and are less capable 
than transformer architectures in capturing long-range interactions, 
which are critical for accurate entropy modeling; and (2) the lack of 
investigation on large-scale, heterogeneous datasets limits the assess-
ment of generalization, thereby hindering their practical deployment 
in real-world scenario.

In response to these limitations, we introduce a Universe Pathology 
JPEG Lossless Recompression Foundation Model (ULRFM). The core 
innovation of ULRFM lies in its transformer-based context model, which 
explicitly models long-range dependencies among DCT quantized co-
efficients to enable substantially more accurate entropy estimation 
and, consequently, more effective compression. Built upon the previ-
ous framework (Guo et al., 2023), ULRFM reconstructs independent 
transformer-based context models for luma and chroma components. As 
show in Fig. 1, each component first passes through a dedicated hyper-
network to generate side information, which is subsequently utilized 
as conditional input to the corresponding context model. This paral-
lelized design reduces transformer sequence length and computational 
cost without compromising modeling power. Importantly, research on 
lossless recompression of pathological JPEG images remains scarce, 
and no existing work has explored this problem at scale. This gap 
limits the feasibility of learned compression techniques in clinical work-
flows. To the best of our knowledge, we present the first foundation 
model solution for large-scale pathology JPEG lossless recompression, 
systematically examine the impact of data quantity and model ca-
pacity on compression performance, and substantially surpass prior 
CNN-based approaches while demonstrating robust out-of-distribution 
generalization.

The main contributions of this work are summarized as follows:

(1) To the best of our knowledge, this is the first foundation model

specifically designed for pathology image JPEG lossless recom-
pression.
(2) We propose a transformer-based context modeling framework

that captures long-range dependencies among DCT coefficients, 
enabling more accurate entropy estimation and improved com-
pression efficiency.
(3) We provide the first systematic investigation into how data quan-

tity and model capacity influence the performance of transformer-
based JPEG recompression model.

(4) We conduct extensive experiments across multi-organ and multi-

cancer in-distribution pathology datasets as well as multiple 
out-of-distribution benchmarks. Results demonstrate that our ap-
proach significantly surpasses CNN-based learned methods, set-
ting a new state of the art while exhibiting strong out-of-
distribution generalization.

2. Related works

2.1. Overview of JPEG compression

The JPEG compression pipeline begins by converting an RGB image 
into the YCbCr color space, which comprises one luma component 
(Y) and two chroma components (Cb and Cr). Because chromatic 
information is generally less perceptually critical than luma detail, the 
chroma channels are commonly subsampled. The most widely used 
configuration is YCbCr 4:2:0, in which the Y component is preserved 
at full resolution, whereas the Cb and Cr components are downsampled 
to one quarter of their original spatial resolution (Wallace, 2002). This 
study focuses on this most commonly used YCbCr 4:2:0 subsampling 
configuration.

Following subsampling, each component is partitioned into nonover-
lapping 8 × 8 blocks, and each block is transformed into the frequency 
domain via the DCT, yielding an 8 × 8 matrix of DCT coefficients. These 
coefficients are subsequently quantized using specific quantization 
tables. Finally, the quantized DCT coefficients are coded by lossless 
Huffman coding (Huffman, 2006) to produce the final JPEG bitstream.

2.2. Lossless JPEG recompression

To address the issues of quantization redundancy and the limited 
efficiency of Huffman coding (Huffman, 2006) in the JPEG compres-
sion pipeline, researchers have developed several traditional JPEG 
lossless recompression methods, most notably Lepton (Horn et al., 
2017) and JPEG XL (Alakuijala et al., 2019, 2020). Lepton replaces 
JPEG’s Huffman coding (Huffman, 2006) with more efficient arithmetic 
coding (Witten et al., 1987) and employs a context adaptive probability 
model that is dynamically updated in real time based on the structure 
of the JPEG bitstream and previously coded data. This enables more 
accurate probability estimation and achieves an additional compression 
ratio of about 20%. JPEG XL substitutes the fixed 8 × 8 DCT blocks 
in JPEG with a variable-block DCT representation, thereby further 
reducing quantization redundancy, and integrates asymmetric numeral 
systems (ANS) (Duda, 2013; Duda et al., 2015) in place of Huffman 
coding. In addition, the general-purpose compressor CMIX can also 
perform secondary lossless recompression of JPEG images. CMIX is 
characterized by the use of a large number of contextual features for 
prediction. However, compared with Lepton and JPEG XL, it does not 
yield a substantial gain in compression ratio, while incurring signif-
icantly higher computational complexity and much slower running 
speed.

In recent years, CNN-based JPEG lossless recompression has at-
tracted increasing attention, leveraging the powerful modeling capacity 
of deep neural networks. Guo et al. (2022) were the first to propose 
a CNN-based entropy model directly in the DCT domain, learning the 
probability distribution of quantized DCT coefficients, thereby nar-
rowing the gap between the estimated and true data distributions. 
Their work demonstrated that directly compressing in the DCT domain 
has clear advantages over operating in the pixel domain, yielding an 
improvement in compression rate of approximately 30%. Fan et al. 
(2022) introduced an end-to-end learnable lossy transform that maps 
DCT coefficients to a more compact representation, effectively elimi-
nating redundancy introduced by DCT quantization. They then jointly 
encode the transformed representation and the residual between the 
lossy reconstruction and the original coefficients, achieving an average

Medical Image Analysis 113 (2026) 104152

T. Song et al.

Fig. 1. Overview of the Proposed Universe Pathology JPEG Lossless Recompression Foundation Model. For the luma (Y) and chroma (Cb and Cr) components, each 
component comprises a hyper-network and a Transformer-based context model. The hyper-network extracts side information to learn global correlation priors, 
while the Transformer context model establishes long-range dependencies among coefficients to capture fine-grained local details and further reduce redundancy. 
The primary workflow proceeds as follows: the hyper-network encodes the DCT coefficients and quantizes them into ̃𝐳, which is subsequently compressed into 
a bitstream using an Arithmetic Encoder (AE). Subsequently, an Arithmetic Decoder (AD) decompresses the bitstream and generates the side information 𝐡. The 
Transformer context model exploits the obtained side information to estimate the entropy of each DCT coefficient, and then performs lossless compression of the 
DCT coefficients using the estimated entropy in conjunction with an Asymmetric Numeral System Encoder (ANS-E).

compression ratio improvement of about 21.49% over standard JPEG. 
In Guo et al. (2023), the authors proposed a multi-level parallel con-
ditional modeling framework that enables parallel coding of the luma 
and chroma components, significantly reducing coding latency while 
maintaining a compression ratio gain of roughly 30%. Nonetheless, 
their evaluation metrics for compression rates are often compromised 
by their development on limited natural image datasets.

Furthermore, even though Convolutional Neural Networks (CNNs) 
have been leveraged for modeling DCT coefficient probability distribu-
tions, they predominantly capture local relationships. As a result, their 
representational capacity is generally inferior to that of transformer 
architectures, which are designed to model long-range dependencies. 
Moreover, these JPEG lossless recompression methods have not been 
thoroughly investigated on large-scale and heterogeneous datasets, 
making it difficult to rigorously validate their generalization ability 
and effectiveness, thereby impeding their deployment in real-world 
applications. In this study, we present a foundation model designed 
for lossless recompression of pathological images. Its performance is 
rigorously evaluated on a large-scale dataset to demonstrate its efficacy 
in real diagnostic workflows.

2.3. Pathology foundation model

The rapid advancement in pathology digitization has led to an 
exponential increase in the availability of Whole Slide Images (WSIs), 
thereby establishing a robust foundation for developing powerful and 
effective models to support computer-aided diagnosis and analysis. The 
current landscape of pathology foundation models is largely defined by 
two prominent paradigms: vision-centric models Ma et al. (2026), Zhou 
et al. (2026) and multimodal models Chen et al. (2025), Zhong et al. 
(2025), J. Lu et al. (2024), Hua et al. (2025).

Vision-centric foundation models are predominantly built upon Vi-
sion Transformers (ViTs) combined with self-supervised learning (SSL) 
on WSIs. Representative works such as Virchow (Vorontsov et al., 
2024) and its successors Virchow2 and Virchow2G (Zimmermann et al., 
2024) employ large ViT backbones with DINOv2 (Oquab et al., 2023) 
training and morphology-preserving augmentations. This approach ef-
fectively captures rich spatial dependencies crucial for pathological 
analysis. Subsequent models further refine this paradigm along sev-
eral key dimensions, including: (1) Enhancements at the SSL level, 
where methods like UNI (Chen et al., 2024), BROW (Wu et al., 2023),

Medical Image Analysis 113 (2026) 104152

T. Song et al.

Fig. 2. Data Preprocessing. Taking a 16 × 16 images as an example, four 8 × 8 DCT blocks are processed according to frequency, zigzag scanning, and inverse 
ordering.

Pathorchestra Yan et al. (2025) and Hibou (Nechaev et al., 2024) 
extend the DINO/DINOv2 framework with techniques such as self-
distillation, masked image modeling (MAE) (He et al., 2022), and 
stain-specific augmentations (e.g., RandStainNA Shen et al., 2022) to 
improve robustness against staining variability and generalize across 
tissues and stains; (2) Contributions at the data and domain knowledge 
level, exemplified by RudolfV (Dippel et al., 2024), which leverages 
expert-curated, heterogeneous datasets for tumor microenvironment 
analysis and biomarker prediction; and (3) Architectural and task-level 
innovations, such as Prov-GigaPath (H. Xu et al., 2024)’s two-stage 
design with tile-level pretraining and long-range slide-level encod-
ing, and Kaiko-ai (Aben et al., 2024)’s online patching for scalabil-
ity. Furthermore, aggregation modules like OmniScreen (Y.K. Wang 
et al., 2024) and COBRA (Lenz et al., 2025) utilize attention-based or 
Mamba-based (Gu and Dao, 2024; Rahman et al., 2024) mechanisms 
for effective slide-level representations.

Complementing these visual models, multimodal foundation models 
integrate visual and textual data, ushering in an era of more profound 
and comprehensive pathological analysis. Key developments include 
PathoDuet (Hua et al., 2024) and Madeleine (Vaidya et al., 2025), 
which are founded on SSL frameworks specifically engineered for H&E 
and IHC images. mSTAR (Y. Xu et al., 2024) seamlessly integrates visual 
and textual information to enrich analytical depth. While HistGen (Guo 
et al., 2024) focuses on report generation via Multiple Instance Learn-
ing, TITAN (Ding et al., 2025) employs a three-stage strategy leveraging 
iBOT (Zhou et al., 2021) for visual learning and CoCa (Yu et al., 
2022) for multimodal vision-language alignment. The integration of 
large language models with vision encoders is further exemplified by 
PathChat (M.Y. Lu et al., 2024) and PRISM (Shaikovski et al., 2024).

Despite these remarkable advancements across various downstream 
pathological analysis tasks, a critical research gap persists. None of 
the existing foundation models have been proposed specifically for 
lossless compression of pathological images. The ever-growing storage 
demands in digital pathology infrastructure necessitate efficient and 
lossless compression strategies to manage the enormous data volume. 
In this work, we introduce a novel foundation model explicitly designed 
for lossless compression of pathological images, trained on a large-scale 
dataset, thereby laying a crucial foundation for practical deployment in 
real-world clinical settings and directly addressing this unmet need for 
storage efficiency.

3. Method

The overall structure of the proposed ULRFM is illustrated in Fig. 
1. For the luma (Y) and chroma (Cb and Cr) components, each com-
ponent consists of a Hyper-Network and a Transformer Context Model,

Fig. 3. Comprehensive visualization of the distribution characteristics and 
sample quantities across the training and test datasets.

although the Context Models differ slightly between luma and chroma. 
The Hyper-Network extracts side information to learn a global correla-
tion prior, while the Transformer Context Model establishes long-range 
dependencies among coefficients in order to capture finer local details 
and further reduce redundancy. Specifically, the luma Transformer 
Context Model performs partitioning in both spatial and frequency 
directions, whereas the chroma Context Model employs a checker-
board spatial reassembly. It is important to note that the two chroma 
channels (Cb and Cr) are concatenated along the channel dimension 
and processed by the same network in parallel with the luma branch, 
which significantly lowers the computational overhead of the attention 
mechanism. Based on our foundation model, we have constructed a 
dataset of more than nine million image tiles, covering ten cancers from 
ten organs.

3.1. Preprocessing

JPEG encoders first partition an input image into non-overlapping 
8 × 8 pixel blocks and transform each block via the DCT into an 
8 × 8 coefficient matrix. In this matrix, each coefficient theoretically 
corresponds to a specific frequency: the top-left element is the DC 
component, and the remaining 63 elements are AC components. As 
illustrated in Fig. 2, for a 16 × 16 example image we collect the DCT 
outputs of its four blocks and reshape them into a single 4 × 8 × 8 
tensor. Because DCT coefficient matrices contain substantial redun-
dancy, we apply a zig-zag scan to each 8 × 8 block along the channel 
dimension, clustering zero-valued coefficients and emphasizing struc-
turally informative elements to aid network learning. Following Guo

Medical Image Analysis 113 (2026) 104152

T. Song et al.

Table 1
Summary of datasets, organ types, tissue types, the number of WSIs, and the 
corresponding number of 256 × 256 tiles.
 Dataset
Organ type
Tissue type
#WSIs
#Tiles
 BRCA
Breast
Breast Invasive Cancer
100
668,352
 KICH
Kidney
Kidney Chromophobe
115
1,431,679
 STAD
Stomach
Stomach Adenocarcinoma
91
897,867
 ACC
Adrenal gland
Adrenocortical Carcinoma
99
1,526,480
 PAAD
Pancreas
Pancreatic Adenocarcinoma
99
1,004,329
 BLCA
Bladder
Bladder Cancer
100
1,388,929
 LUAD
Bronchus and lung
Lung Adenocarcinoma
95
959,655
 LGG
Brain
Lower Grade Glioma
107
956,878
 TGCT
Testicular
Testicular Germ Cell Tumors
30
304,856
 UVM
Eye
Uveal Melanoma
30
198,565
 PANDA
Prostate
Prostate Cancer
100
33,797
 BRACS
Breast
Breast Cancer
10
76,197

et al. (2023), we then reshape this tensor so that coefficients of the 
same frequency are aggregated across spatial locations, while different 
frequencies are aligned along the channel dimension. An inverse re-
ordering along the frequency axis finally arranges the coefficients from 
high to low frequency. This frequency representation facilitates more 
accurate estimation of entropy model from the side information.

3.2. Hyper-network

The design of the hyper-network follows Guo et al. (2022, 2023), 
which is divided into a Hyper Encoder and a Hyper Decoder. The 
Hyper Encoder adopts the architecture: Conv → LeakyReLU → Conv 
→ LeakyReLU → Conv, where the first convolutional layer employs a 
stride of 1, while the remaining convolutional layers utilize a stride 
of 2. The Hyper Decoder comprises Conv → LeakyReLU → Deconv →
LeakyReLU → Deconv, where Deconv denotes a transposed convolution 
layer. Specifically, the first and second deconvolutional layers employ 
a stride of 2, while the final deconvolutional layer adopts a stride of 
1. As illustrated in Fig. 1, the overall computational pipeline proceeds 
as follows: the DCT coefficient array is encoded through the Hyper 
Encoder to obtain the latent feature 𝐳. Subsequently, 𝐳 is quantized 
via torch.round to yield ̃𝐳. Finally, the Hyper Decoder produces the 
side information 𝐡. To maintain unimpeded gradient flow during back-
propagation through the quantization operation, the quantize_STE
function is utilized, which approximates the gradient via the derivative 
of the identity function (straight-through estimator) (Theis et al., 2017). 
It is noteworthy that the side information must also be encoded as a 
bitstream to provide conditional information for subsequent decom-
pression. Specifically, a factorized entropy model (Ballé et al., 2018; 
Minnen et al., 2018) is employed to compress ̃𝐳 into a bitstream.

3.3. Transformer context model

Transformer Context Model for CbCr. We adopt the Checkboard 
Rearrangement context modeling approach proposed by (Guo et al., 
2023), which mitigates the spatial redundancy in computations inher-
ent to the original checkboard context model. To ensure the long-range 
dependency modeling capability of the context model, the entire con-
text model is implemented using pure Transformer architecture. As 
illustrated in Fig. 1, the CbCr coefficients are first partitioned into 
anchor and non-anchor regions according to their spatial positions. Sub-
sequently, the anchor and non-anchor regions are spatially rearranged 
to eliminate vacant regions. Specifically, the anchor region is condi-
tioned on 𝐡 (the side information of CbCr) and fed into a Transformer 
model to learn the mean and scale parameters of a Gaussian entropy 
model, thereby enabling the compression of the anchor region. The 
non-anchor region, in turn, is conditioned on both 𝐡 and the anchor 
region, and is processed by a separate Transformer model that learns a 
new Gaussian entropy model for compressing the non-anchor region.

Transformer Context Model for Y. We propose a transformer-
based context model to learn more powerful Gaussian entropy models 
for each subregion in both spatial and frequency directions. As illus-
trated in Fig. 1, we partition the DCT coefficient matrix 𝐘 into 𝑠× 𝑓
subregions by dividing it into 𝑠 rows in the spatial direction and 𝑓
columns in the frequency direction. For each subregion, we employ 
an independent transformer model to characterize the corresponding 
Gaussian entropy. Specifically, the first subregion 𝑦(1,1) utilizes the 
side information 𝐡 decoded from ̃𝐳 via the Hyper Decoder as condi-
tional input to a transformer model, which learns the mean and scale 
parameters of the Gaussian entropy for compression. Once 𝑦(1,1) is 
compressed, the subsequent subregion 𝑦(1,2) in the frequency direction 
takes both 𝑦(1,1) and the side information 𝐡 as conditional inputs to its 
corresponding transformer model for entropy modeling. This process 
continues sequentially, where each subsequent 𝑦(1,𝑓) conditions on both 
{𝑦(1,𝑗)}𝑓−1
𝑗=1  and 𝐡. Upon completion of entropy modeling for the first 
spatial subregion across all frequency directions, the remaining 𝑠−1
spatial subregions are modeled in a similar manner. Notably, 𝑦(𝑖,𝑗)
depends not only on the side information 𝐡 but also on the previously 
modeled coefficients 𝑦<(𝑖,𝑗), and this dependency pattern continues 
recursively for subsequent subregions. As illustrated in Algorithm 1, 
the decoding process reconstructs the DCT coefficient matrix 𝐘 from 
its compressed bitstream ̃𝐘 and the hyper-prior bitstream ̃𝐳. In this 
experiment, we configure 𝑠= 4 and 𝑓= 9. The spatial partition-
ing yields 4 rows, while the frequency partitioning yields 9 columns 
with lengths [28, 8, 7, 6, 5, 4, 3, 2, 1] respectively. The cumulative length 
across all frequency columns is 64, which corresponds to 64 frequency 
coefficients in total.

Algorithm 1 ULRFM: Decoding Process with 𝐘

1: Input: Side information bitstream ̃𝐳 and compressed bitstream ̃𝐘. 
2: Output: Reconstructed DCT coefficient matrix 𝐘. 
3: Requires: Hyper-Decoder, ANS Decoder and Transformer Context 
Model with {𝑇𝑀(𝑖,𝑗)}𝑠,𝑓
𝑖=1,𝑗=1.

4: # Step 1: Initialize side information 𝐡 and Condition .
5: 𝐡= 𝐻𝑦𝑝𝑒𝑟𝐷𝑒𝑐𝑜𝑑𝑒𝑟(̃𝐳)
6: = 𝐡

7: # Step 2: Sequentially process the first spatial row.
8: for 𝑗= 1 to 𝑓 do 
9:
(𝝁(1,𝑗), 𝝈(1,𝑗)) = 𝑇𝑀(1,𝑗)()
10:
𝑦(1,𝑗) = 𝐴𝑁𝑆𝐷𝑒𝑐𝑜𝑑𝑒𝑟((𝝁(1,𝑗), 𝝈(1,𝑗)), ̃𝐘)
11:
= ∪𝑦(1,𝑗)
12: end for

13: # Step 3: Process remaining spatial rows in parallel.
14: for 𝑗= 1 to 𝑓 do 
15:
{(𝝁(𝑖,𝑗), 𝝈(𝑖,𝑗))}𝑠
𝑖=2 = {𝑇𝑀(𝑖,𝑗)()}𝑠
𝑖=2
16:
{𝑦(𝑖,𝑗)}𝑠
𝑖=2 = {𝐴𝑁𝑆𝐷𝑒𝑐𝑜𝑑𝑒𝑟((𝝁(𝑖,𝑗), 𝝈(𝑖,𝑗)), ̃𝐘)}𝑠
𝑖=2
17:
= ∪{𝑦(𝑖,𝑗)}𝑠
𝑖=2
18: end for

19: 𝐘 = {𝑦(𝑖,𝑗)}𝑠,𝑓
𝑖=1,𝑗=1
20: return 𝐘

3.4. Large-scale database

An extensive dataset of image tiles forms the foundation of our 
model. In this study, leveraging the publicly available TCGA dataset,3 
PANDA (Bulten et al., 2022), and BRACS datasets (Brancati et al., 
2022), we have constructed, to our knowledge, the largest multi-organ 
and multi-cancer image tiles for JPEG lossless recompression. As sum-
marized in Table 1, the dataset comprises 976 WSIs spanning eleven 
cancer types from eleven organs, yielding approximately ten million

3 https://portal.gdc.cancer.gov/

Medical Image Analysis 113 (2026) 104152

T. Song et al.

Table 2
Comparison of existing methods on in-distribution and out-of-distribution datasets, evaluated using BPP and Compression Saving (%) under YCbCr 4:2:0 at quality 
75.

Metric
BPP and Compression Saving(%)

Dataset

Method 
JPEG
JPEG XL (Alakuijala et al., 2019)
Lepton (Horn et al., 2017)
Guo (Guo et al., 2022)
Eff-Net (Guo et al., 2023)
Ours

In-distribution

ACC
2.233
–
1.932
13.48%
1.885
15.58%
1.790
19.84%
1.670
25.21%
1.482
33.63%
BLCA
2.197
–
1.899
13.54%
1.846
15.93%
1.770
19.36%
1.646
25.08%
1.456
33.66%
BRCA
2.010
–
1.745
13.14%
1.676
16.57%
1.672
16.77%
1.531
23.83%
1.324
34.13%
KICH
1.976
–
1.716
13.24%
1.651
16.32%
1.620
17.98%
1.486
24.73%
1.310
33.68%
LGG
2.248
–
1.930
14.06%
1.881
16.24%
1.837
18.19%
1.716
23.55%
1.541
31.32%
LUAD
2.123
–
1.827
14.02%
1.767
16.77%
1.752
17.45%
1.618
23.73%
1.434
32.42%
PAAD
2.096
–
1.819
13.27%
1.763
15.94%
1.679
19.94%
1.560
25.48%
1.390
33.68%
STAD
2.088
–
1.800
13.85%
1.742
16.57%
1.708
18.25%
1.592
23.73%
1.415
32.39%

Out-of-distribution

TGCT
2.024
–
1.754
13.34%
1.692
16.39%
1.683
16.84%
1.550
23.37%
1.343
33.64%
UVM
2.039
–
1.760
13.53%
1.705
16.38%
1.721
15.69%
1.592
21.97%
1.384
32.17%
PANDA
1.780
–
1.544
13.26%
1.482
16.74%
1.326
25.51%
1.317
26.01%
1.216
31.69%
BRACS
1.867
–
1.648
11.73%
1.596
14.52%
1.366
26.83%
1.313
29.67%
1.244
33.37%

Fig. 4. Comparison of compression saving (%) on in-distribution and out-of-
distribution (OOD) datasets. Our proposed method consistently outperforms 
all competing methods.

foreground image tiles. For each tile, PNG images are paired with quan-
tized DCT coefficients extracted from their JPEG counterparts using the
torchjpeg.codec.quality module under various quality factors 
and chroma-subsampling configurations. Notably, we focus primarily 
on the industry-standard YCbCr 4:2:0 sampling at 75% quality. Data 
from eight organ types in the TCGA dataset were randomly split into 
in-distribution training and test sets with an 80:20 ratio. Meanwhile, 
data from the remaining two TCGA organ types, as well as images 
from non-TCGA datasets (PANDA and BRACS), were reserved as out-
of-distribution (OOD) test sets. Detailed partitioning information is 
provided in Fig. 3.

4. Experimental results

4.1. Experiment settings

4.1.1. Datasets

We leverage 806 whole-slide images (WSIs) obtained from the 
publicly accessible TCGA database for in-distribution training and eval-
uation, alongside 60 WSIs designated for out-of-distribution evaluation.

Table 3
With respect to the parameters and computational cost of Small, Medium, and 
Large model variants by varying the number of Transformer blocks (N).
 Model size
N
#Params(M)
#GFLOPs 
 Small
2
26.77
31.46
 
 Medium
3
43.34
39.94
 
 Large
4
76.48
56.90

Specifically, the in-distribution dataset comprises eight TCGA sub-
datasets (BRCA, KICH, STAD, ACC, PAAD, BLCA, LUAD, and LGG), each 
corresponding to a distinct cancer type from a particular anatomical 
site. Conversely, the out-of-distribution evaluation set consists of four 
sub-datasets (PANDA, BRACS, TGCT and UVM), as documented in Ta-
ble 1. Following the methodology in Ying et al. (2023), all foreground 
regions within the WSIs are tiled into non-overlapping 256 × 256 
patches, generating a total of over ten million image tiles. Furthermore, 
consistent with (Guo et al., 2022, 2023), we extract quantized DCT co-
efficients across diverse JPEG quality factors and chroma-subsampling 
schemes using torchjpeg.codec.quality. The extracted coeffi-
cients are subsequently processed via our preprocessing pipeline before 
being input to the model.

4.1.2. Implementation details

For training, we set the total number of epochs to 50, employing 
the Adam optimizer with a learning rate of 1 × 10−4 and a batch size of 
48. All experiments were conducted in a consistent environment using 
eight NVIDIA GeForce RTX 4090 GPUs. To further stabilize training, 
gradient clipping with a maximum norm of 1.0 is applied. To ensure a 
fair comparison, we also trained the two baseline methods, Guo et al. 
(2022) and Eff-Net (Guo et al., 2023), on our dataset from scratch. Com-
pression performance is evaluated using the widely adopted metrics: 
bits per pixel (BPP) and compression saving. The BPP for a given image 
is calculated as

BPP = Total bits for compressed image

𝐻× 𝑊
,
(1)

where 𝐻 and 𝑊 denote the spatial dimensions of the original image. 
The compression saving relative to a JPEG baseline at the same image 
quality level is then defined as

Compression Saving (%) = 100 × BPPJPEG −BPPmethod

BPPJPEG
,
(2)

where BPPJPEG and BPPmethod denote the bits per pixel of the JPEG 
baseline and the proposed method, respectively.

Medical Image Analysis 113 (2026) 104152

T. Song et al.

Table 4
Generalization performance under out-of-distribution (OOD) settings on the PANDA dataset: 
BPP and Compression Saving (%) across varying JPEG quality levels.

Quality

Method 
JPEG
JPEG XL
Lepton
Ours

BPP
BPP
Saving
BPP
Saving
BPP
Saving

85
2.3320
1.8103
22.37%
1.7785
23.73%
2.0300
12.95%
75
1.7804
1.5439
13.29%
1.4824
16.75%
1.2160
31.71%
65
1.6308
1.4096
13.57%
1.3461
17.46%
1.0960
32.79%
55
1.4993
1.2756
14.92%
1.2104
19.27%
0.9980
33.44%

4.2. Comparison with existing methods

To evaluate the effectiveness of our proposed method, we conducted 
a comprehensive quantitative comparison against several state-of-the-
art (SOTA) and classical compression methods, including JPEG, JPEG 
XL (Alakuijala et al., 2019, 2020), Lepton (Horn et al., 2017), Guo et al. 
(2022), and Eff-Net (Guo et al., 2023). The results, summarized in Table 
2, demonstrate the clear superiority of our approach.

On the in-distribution datasets, our method consistently achieves 
the lowest Bits Per Pixel (BPP) and consequently the highest compres-
sion savings. For instance, compared to the most recent competitor, Eff-
Net (Guo et al., 2023), our method boosts the compression saving from 
25% to an average of over 33%, marking a significant improvement 
in recompression performance. This consistent dominance is visually 
corroborated by the radar chart in Fig. 4, where our method’s perfor-
mance (purple) comprehensively envelops that of all other techniques, 
indicating its robust superiority across diverse data types. Empirical 
findings indicate a considerable performance drop for the methods 
proposed by  Guo et al. (2022) and Eff-Net (Guo et al., 2023) when 
subjected to large-scale and heterogeneous training and testing data, 
particularly when juxtaposed with their claimed performance on con-
strained natural image datasets. Among these, the approach by  Guo 
et al. (2022) suffered the most significant reduction in performance.

Crucially, our model exhibits remarkable generalization ability on 
diverse out-of-distribution (OOD) datasets. As shown in Table 2, our 
method maintains its lead across all OOD benchmarks. Specifically, it 
achieves compression savings of 33.64% and 32.17% on the TCGA-OOD 
datasets (TGCT and UVM), respectively. More importantly, on the chal-
lenging non-TCGA datasets PANDA and BRACS, characterized by signif-
icant domain shifts from different hospitals and scanners, our method 
continues to outperform all baselines, achieving substantial savings of 
31.69% and 33.37%, respectively. This sustained high performance on 
both intra-domain (TCGA) and cross-domain (non-TCGA) unseen data 
underscores the strong robustness and generalization capabilities of our 
model, a critical attribute for real-world clinical applications.

To further investigate the stability and fine-grained performance, 
we present a per-slice BPP comparison against the strongest recent 
baselines in Fig. 5. The bar chart clearly illustrates that our method 
(green bars) achieves a lower BPP on every single slice of the image vol-
ume. This consistent per-slice advantage, rather than just an on-average 
improvement, proves the robustness and reliability of our compression 
model. In summary, the collective evidence from these results validates 
the superior effectiveness, robustness, and generalization power of our 
proposed method.

4.3. Robustness to varying JPEG configurations

To evaluate the generalizability of our proposed method beyond 
the standard training configuration (YCbCr 4:2:0, Quality 75), we 
conducted extensive experiments on the PANDA dataset, which serves 
as a challenging out-of-distribution (OOD) benchmark due to its distinct 
domain characteristics (different hospitals and scanners) compared to 
the TCGA training data. Specifically, we assessed the compression 
performance across a range of JPEG quality levels: 85, 75, 65, and 
55. The results are summarized in Table 4. As shown in Table 4, our

Fig. 5. A detailed comparison of the Bits Per Pixel (BPP) for each of the 30 
slices in the out-of-distribution TGCT dataset. The results demonstrate that our 
method consistently outperforms other methods by achieving a lower bitrate 
for every slice.

method demonstrates robust generalization capabilities across all tested 
quality factors, consistently outperforming baseline methods (JPEG XL 
and Lepton) even when evaluated on unseen compression settings.

Performance at Standard and High Quality: At the standard 
clinical quality level (Quality 75), our method achieves a significant 
compression saving of 31.71% relative to JPEG. Even in higher-fidelity 
settings (Quality 85), where source images contain less redundancy, our 
model still delivers a 12.95% saving. Although this figure is lower than 
the savings achieved by JPEG XL (22.37%) and Lepton (23.73%), it 
demonstrates that our method remains effective. To further enhance 
compression performance for high-quality JPEGs, future work could 
involve incorporating high-quality samples into the training process.

Efficiency in Archival Scenarios: Notably, our method exhibits 
exceptional efficiency at lower quality factors (Quality 65 and 55), 
which are commonly used for long-term archival storage to minimize 
disk usage. In these scenarios, our approach achieves compression 
savings of 32.79% and 33.44%, respectively. This indicates that our 
probability modeling of DCT coefficients effectively captures statisti-
cal dependencies regardless of the quantization step size, making it 
particularly suitable for cost-effective medical data archiving.

4.4. Analysis of scaling laws

To systematically evaluate the scalability of our proposed architec-
ture, we conduct ablation studies on both model capacity and training 
data volume. These experiments are crucial for understanding the 
behavior of our model and its potential for future improvements.

Model Scaling. We first investigate the impact of model capacity 
on compression performance. We define Small, Medium, and Large 
by varying the number of Transformer blocks (N) from 2 to 4, as 
detailed in Table 3. The corresponding parameter counts and compu-
tational costs (GFLOPs) scale accordingly. The performance of these 
variants is presented in Fig. 6(a). As the model capacity increases

Medical Image Analysis 113 (2026) 104152

T. Song et al.

Fig. 6. Ablation studies on model scaling and data quantity. (a) Bits Per 
Pixel (BPP) for Small, Medium, and Large model variants, demonstrating 
the effectiveness of scaling model capacity. (b) The effect of training data 
volume on the Large model’s performance, showing a clear trend of improved 
compression with more data.

Table 5
Comparison of network efficiency on the PANDA dataset with JPEG 4:2:0 at 
QP 75. Evaluated on NVIDIA GeForce GTX 1660 Ti.
 Methods
GFLOPs
Encode(s)
Decode(s)
GPU memory(M) 
 JPEG XL
N/A
0.35
0.23
N/A
 
 Lepton
N/A
0.34
0.25
N/A
 
 Guo
102.85
2.69
2.5
965
 
 Eff-Net
44.66
1.52
1.44
853
 
 Ours
56.90
4.99
4.94
1065

from the Small to the Large configuration, a clear trend of improved 
compression performance emerges. Specifically, the BPP decreases from 
1.431 for the Small model to 1.406 for the Medium model, and cul-
minates at 1.399 for the Large model. This result demonstrates a 
positive correlation between model size and performance, confirming 
that our architecture effectively utilizes increased capacity to learn 
more efficient representations.

Data Scaling. In addition to model size, we analyze the effect of 
training data volume. For this study, we train our best-performing 
Large model on varying subsets of the training data, ranging from 10% 
to 100%. As illustrated in Fig. 6(b), a clear scaling law is observed: 
increasing the amount of training data leads to a monotonic decrease 
in BPP. The performance improves from a BPP of 1.437 when trained 
with only 10% of the data to 1.399 when the full dataset is utilized. 
This highlights that our model effectively leverages larger datasets 
to enhance its compression efficiency, and its performance is not yet 
saturated by the current data scale.

Collectively, these experiments confirm that our proposed model 
exhibits favorable scaling properties with respect to both model param-
eterization and data availability, which is a desirable characteristic for 
powerful neural compression models.

4.5. Computational efficiency and clinical deployability

To assess the practical applicability of our method in clinical set-
tings, we conducted a comprehensive efficiency analysis on the PANDA 
dataset. We evaluated encoding/decoding runtime, computational com-
plexity (GFLOPs), and GPU memory usage against baseline methods, 
including JPEG XL, Lepton, Eff-Net, and Guo et al. All experiments 
were performed under identical hardware conditions using an NVIDIA 
GeForce GTX 1660 Ti GPU, a mid-range consumer graphics card com-
monly found in general-purpose workstations, rather than high-end 
server clusters. The results are summarized in Table 5.

our method requires approximately 5 s for both encoding and 
decoding per tile, with a GPU memory footprint of ∼ 1 GB (1065 MB). 
While this latency is higher than that of traditional codecs (e.g., JPEG 
XL, ∼ 0.3s) and lightweight networks like Eff-Net (∼ 1.5s), it remains 
within a manageable range for offline processing workflows. Notably, 
our computational complexity (56.90 GFLOPs) is significantly lower

than the deep learning baseline Guo et al. (102.85 GFLOPs), indicating 
a more efficient architectural design despite the superior compression 
performance.

The primary clinical value of our strictly lossless method lies in 
long-term data archival. In such scenarios, compression is typically 
performed as an offline, batch process for ‘‘cold data’’ (infrequently 
accessed historical records). For these applications, the modest increase 
in inference time (∼ 5s/tile) is a negligible trade-off compared to 
the substantial benefits of reduced storage costs and bandwidth re-
quirements. It is important to note that these benchmarks represent a 
conservative estimate using older consumer hardware. With the rapid 
adoption of modern GPUs (e.g., RTX 30/40 series or professional A-
series cards) in medical imaging infrastructure, the inference speed of 
our transformer-based model is expected to improve drastically. Thus, 
our method offers a scalable and cost-effective solution for sustainable 
digital pathology archives.

4.6. Visualization

To gain a deeper understanding of our model’s internal workings 
and the source of its strong performance, we visualize the learned self-
attention maps from a representative Transformer block within our 
context model. Fig. 7 displays the attention patterns for all 8 heads, 
separately for the Y and CbCr components. The visualization reveals a 
remarkable degree of specialization among the heads, where each head 
learns a distinct and meaningful pattern for information aggregation.

A striking example is Head 2 for the CbCr component, which 
learns an almost perfect identity function (a sharp diagonal line). This 
indicates that the model has learned to selectively preserve information 
from certain tokens without modification, effectively acting as a resid-
ual connection. This is a powerful learned behavior, preventing feature 
degradation. And, Head 1 for both Y and CbCr components exhibits a 
strong focus on the main diagonal and its immediate neighbors. This 
pattern closely mimics the behavior of a convolutional kernel, focusing 
on capturing local dependencies and textures within a small receptive 
field. Other heads, such as Head 3, 4, and 7, display more complex 
and sparser patterns. They capture non-local, long-range dependencies 
across the image, linking spatially distant but contextually relevant 
regions. This capability is a key advantage of the Transformer architec-
ture over traditional CNNs, which struggle to model such relationships 
efficiently.

This observed specialization provides a clear rationale for the model’s 
excellent scaling properties, as demonstrated in our ablation studies. 
The multi-head attention mechanism is not a rigid structure but a 
flexible ensemble of experts. In essence, the ability of the attention 
mechanism to dynamically learn and combine a diverse set of fun-
damental operations makes it an incredibly powerful and scalable 
building block for a neural compression model.

5. Discussion and conclusion

In this work, we addressed the fundamental limitations of existing 
JPEG lossless recompression techniques, which often struggle with 
long-range dependency modeling and cross-domain generalization. We 
introduced a Universe Pathology JPEG Lossless Recompression Foun-
dation Model (ULRFM), a novel Transformer-based architecture specif-
ically designed to overcome these challenges by effectively modeling 
global contexts within JPEG DCT coefficient streams. Our extensive 
experiments, conducted on a massive-scale digital pathology dataset, 
provide compelling evidence for the superiority of our approach.

Our quantitative results demonstrate that ULRFM establishes a new 
state of the art, substantially outperforming both traditional meth-
ods like JPEG XL and recent CNN-based learned approaches such as 
Eff-Net. The significant increase in compression savings is a direct 
testament to the Transformer’s enhanced representational capacity.

Medical Image Analysis 113 (2026) 104152

T. Song et al.

Fig. 7. Visualization of the learned self-attention maps from a transformer 
block in our context model. The top and bottom rows display the patterns for 
the 8 attention heads corresponding to the CbCr and Y channels, respectively. 
The diversity in patterns highlights the specialization of each head. For 
example, Head 2 (CbCr) approximates an identity function, while other heads 
focus on local or long-range spatial dependencies.

Crucially, this superiority is not confined to in-distribution data. UL-
RFM’s strong performance out-of-distribution datasets underscores its 
robust generalization capabilities, a critical weakness in prior learned 
models that our approach successfully mitigates. Specifically, beyond 
the TCGA-based OOD sets (TGCT, UVM), our method demonstrates 
remarkable robustness on completely independent non-TCGA datasets, 
PANDA and BRACS. Despite significant domain shifts arising from 
different hospitals, scanners, and staining protocols, ULRFM achieves 
compression savings of 31.69% and 33.37%, respectively, consistently 
outperforming all baselines. This confirms that our model captures 
universal statistical patterns in pathology images rather than overfitting 
to specific dataset characteristics.

This study provides further insight into the potential of our model. 
We observed a clear and consistent improvement in compression per-
formance with increases in both model capacity and data quantity. 
This confirms that ULRFM is not a shallow model with quickly di-
minishing returns but a scalable foundation model architecture whose 
performance is not yet saturated. This scalability is a highly desirable 
property, suggesting that future performance gains are attainable with 
access to even larger datasets and greater computational resources. 
Furthermore, to validate the practical applicability of ULRFM across 
diverse clinical pipelines, we evaluated its generalization under varying 
JPEG configurations. Experiments on the PANDA dataset across quality 
factors ranging from 55 to 85 reveal that our method maintains superior 
efficiency regardless of the quantization level. Notably, at lower quality 
settings, commonly used for archival storage, our approach achieves 
savings exceeding 32%, demonstrating its particular suitability for 
cost-effective long-term data retention.

In conclusion, ULRFM represents a paradigm shift in JPEG lossless 
recompression. By leveraging the global context modeling power of 
Transformers and training on a large-scale, diverse dataset, it not only 
sets a new benchmark in compression efficiency and generalization but 
also provides a clear path for future scaling. The practical implications 
for digital pathology are significant, as our method can markedly 
alleviate the escalating storage and transmission burdens associated 
with whole-slide imaging.

Despite its strong performance, we acknowledge that the compu-
tational cost increases with model size. Future work could explore 
model compression techniques, such as knowledge distillation and 
quantization, to create more lightweight yet powerful variants.

CRediT authorship contribution statement

Tao Song: Writing – review & editing, Writing – original draft, 
Visualization, Validation, Methodology, Investigation, Conceptualiza-
tion. Rong Tao: Writing – original draft, Methodology, Investigation, 
Conceptualization. Chunyan Wu: Visualization, Validation, Data cura-
tion. Mengmeng Zhao: Visualization, Validation, Investigation, Data 
curation. Jiajun Deng: Writing – review & editing, Visualization, Vali-
dation, Formal analysis. Yi Guo: Writing – review & editing, Methodol-
ogy, Conceptualization. Feng Xu: Writing – review & editing, Method-
ology, Conceptualization. Chang Chen: Writing – review & editing, 
Supervision, Conceptualization. Kun Qian: Writing – review & editing, 
Supervision, Conceptualization.

Declaration of competing interest

The authors declare that they have no known competing finan-
cial interests or personal relationships that could have appeared to 
influence the work reported in this paper.

References

Aben, N., de Jong, E.D., Gatopoulos, I., Känzig, N., Karasikov, M., Lagré, A., Moser, R.,

van Doorn, J., Tang, F., et al., 2024. Towards large-scale training of pathology 
foundation models. arXiv preprint arXiv:2404.15217.
Alakuijala, J., Boukortt, S., Ebrahimi, T., Kliuchnikov, E., Sneyers, J., Upenik, E.,

Vandevenne, L., Versari, L., Wassenberg, J., 2020. Benchmarking JPEG XL im-
age compression. In: Optics, Photonics and Digital Technologies for Imaging 
Applications VI, vol. 11353, SPIE, pp. 187–206.
Alakuijala, J., Van Asseldonk, R., Boukortt, S., Bruse, M., Comşa, I.-M., Firsching, M.,

Fischbacher, T., Kliuchnikov, E., Gomez, S., Obryk, R., et al., 2019. JPEG XL next-
generation image compression architecture and coding tools. In: Applications of 
Digital Image Processing XLII, vol. 11137, SPIE, pp. 112–124.
Ballé, J., Minnen, D., Singh, S., Hwang, S.J., Johnston, N., 2018. Variational image

compression with a scale hyperprior. arXiv preprint arXiv:1802.01436.
Brancati, N., Anniciello, A.M., Pati, P., Riccio, D., Scognamiglio, G., Jaume, G.,

De Pietro, G., Di Bonito, M., Foncubierta, A., Botti, G., et al., 2022. Bracs: A dataset 
for breast carcinoma subtyping in h&e histology images. Database 2022, baac093.
Bulten, W., Kartasalo, K., Chen, P.-H.C., Ström, P., Pinckaers, H., Nagpal, K., Cai, Y.,

Steiner, D.F., Van Boven, H., Vink, R., et al., 2022. Artificial intelligence for 
diagnosis and gleason grading of prostate cancer: the PANDA challenge. Nature 
Med. 28 (1), 154–163.
Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F., Jaume, G., Song, A.H., Chen, B.,

Zhang, A., Shao, D., Shaban, M., et al., 2024. Towards a general-purpose foundation 
model for computational pathology. Nature Med. 30 (3), 850–862.
Chen, K., Liu, M., Yan, F., Ma, L., Shi, X., Wang, L., Wang, X., Zhu, L., Wang, Z.,

Zhou, M., et al., 2025. Cost-effective instruction learning for pathology vision and 
language analysis. Nature Comput. Sci. 5 (7), 524–533.
Ding, T., Wagner, S.J., Song, A.H., Chen, R.J., Lu, M.Y., Zhang, A., Vaidya, A.J.,

Jaume, G., Shaban, M., Kim, A., et al., 2025. A multimodal whole-slide foundation 
model for pathology. Nature Med. 1–13.
Dippel, J., Feulner, B., Winterhoff, T., Milbich, T., Tietz, S., Schallenberg, S., Dern-

bach, G., Kunft, A., Heinke, S., Eich, M.-L., et al., 2024. Rudolfv: a foundation 
model by pathologists for pathologists. arXiv preprint arXiv:2401.04079.
Duda, J., 2013. Asymmetric numeral systems: entropy coding combining speed of

huffman coding with compression rate of arithmetic coding. arXiv preprint arXiv:
1311.2540.
Duda, J., Tahboub, K., Gadgil, N.J., Delp, E.J., 2015. The use of asymmetric numeral

systems as an accurate replacement for huffman coding. In: 2015 Picture Coding 
Symposium. PCS, IEEE, pp. 65–69.
Fan, X., Li, X., Chen, Z., 2022. Learned lossless jpeg transcoding via joint lossy

and residual compression. In: 2022 IEEE International Conference on Visual 
Communications and Image Processing. VCIP, IEEE, pp. 1–5.
Goode, A., Gilbert, B., Harkes, J., Jukic, D., Satyanarayanan, M., 2013. OpenSlide: A

vendor-neutral software foundation for digital pathology. J. Pathol. Informatics 4 
(1), 27.
Gu, A., Dao, T., 2024. Mamba: Linear-time sequence modeling with selective state

spaces. In: First Conference on Language Modeling.

Medical Image Analysis 113 (2026) 104152

T. Song et al.

Guo, Z., Ma, J., Xu, Y., Wang, Y., Wang, L., Chen, H., 2024. Histgen: Histopathol-

ogy report generation via local-global feature encoding and cross-modal context 
interaction. In: International Conference on Medical Image Computing and 
Computer-Assisted Intervention. Springer, pp. 189–199.
Guo, L., Shi, X., He, D., Wang, Y., Ma, R., Qin, H., Wang, Y., 2022. Practical learned

lossless JPEG recompression with multi-level cross-channel entropy model in the 
DCT domain. In: Proceedings of the IEEE/CVF Conference on Computer Vision and 
Pattern Recognition. pp. 5862–5871.
Guo, L., Wang, Y., Xu, T., Luo, J., He, D., Ji, Z., Wang, S., Wang, Y., Qin, H., 2023.

Efficient learned lossless jpeg recompression. arXiv preprint arXiv:2308.13287.
He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R., 2022. Masked autoencoders are

scalable vision learners. In: Proceedings of the IEEE/CVF Conference on Computer 
Vision and Pattern Recognition. pp. 16000–16009.
Horn, D.R., Elkabany, K., Lesniewski-Lass, C., Winstein, K., 2017. The design, imple-

mentation, and deployment of a system to transparently compress hundreds of 
petabytes of image files for a {𝐹𝑖𝑙𝑒−𝑆𝑡𝑜𝑟𝑎𝑔𝑒} service. In: 14th USENIX Symposium 
on Networked Systems Design and Implementation. NSDI 17, pp. 1–15.
Hua, S., Wu, J., Shen, T., Hu, K., Huang, Z., Ni, S., Zhang, Z., Li, Y., Wang, Z., Zhang, X.,

2025. Pathfound: an agentic multimodal model activating evidence-seeking 
pathological diagnosis. arXiv preprint arXiv:2512.23545.
Hua, S., Yan, F., Shen, T., Ma, L., Zhang, X., 2024. PathoDuet: Foundation models for

pathological slide analysis of H&E and IHC stains. Med. Image Anal. 97, 103289.
Huffman, D.A., 2006. A method for the construction of minimum-redundancy codes.

Resonance 11 (2), 91–99.
Lenz, T., Neidlinger, P., Ligero, M., Wölflein, G., van Treeck, M., Kather, J.N., 2025.

Unsupervised foundation model-agnostic slide-level representation learning. In: 
Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 
30807–30817.
Liu, J., Yang, H., Zhou, H.-Y., Yu, L., Liang, Y., Yu, Y., Zhang, S., Zheng, H., Wang, S.,

2024. Swin-UMamba†: adapting mamba-based vision foundation models for medical 
image segmentation. IEEE Trans. Medical Imag. 44 (10), 3898–3908.
Lu, M.Y., Chen, B., Williamson, D.F., Chen, R.J., Zhao, M., Chow, A.K., Ikemura, K.,

Kim, A., Pouli, D., Patel, A., et al., 2024. A multimodal generative AI copilot for 
human pathology. Nature 634 (8033), 466–473.
Lu, J., Yan, F., Zhang, X., Gao, Y., Zhang, S., 2024. Pathotune: Adapting visual founda-

tion model to pathological specialists. In: In International Conference on Medical 
Image Computing and Computer-Assisted Intervention. Springer, pp. 395–406.
Ma, J., Guo, Z., Zhou, F., Wang, Y., Xu, Y., Li, J., Yan, F., Cai, Y., Zhu, Z., Jin, C., et

al., 2026. A generalizable pathology foundation model using a unified knowledge 
distillation pretraining framework. Nature Biomed. Eng. 10 (3), 545–564.
Minnen, D., Ballé, J., Toderici, G.D., 2018. Joint autoregressive and hierarchical priors

for learned image compression. Adv. Neural Inf. Process. Syst. 31.
Nechaev, D., Pchelnikov, A., Ivanova, E., 2024. Hibou: A family of foundational vision

transformers for pathology. arXiv preprint arXiv:2406.05074.
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernan-

dez, P., Haziza, D., Massa, F., El-Nouby, A., et al., 2023. Dinov2: Learning robust 
visual features without supervision. arXiv preprint arXiv:2304.07193.
Rahman, M.M., Tutul, A.A., Nath, A., Laishram, L., Jung, S.K., Hammond, T., 2024.

Mamba in vision: A comprehensive survey of techniques and applications. arXiv 
preprint arXiv:2410.03105.
Shaikovski, G., Casson, A., Severson, K., Zimmermann, E., Wang, Y.K., Kunz, J.D.,

Retamero, J.A., Oakley, G., Klimstra, D., Kanan, C., et al., 2024. Prism: A multi-
modal generative foundation model for slide-level histopathology. arXiv preprint 
arXiv:2405.10254.
Shen, Y., Luo, Y., Shen, D., Ke, J., 2022. Randstainna: Learning stain-agnostic features

from histology slides by bridging stain augmentation and normalization. In: 
International Conference on Medical Image Computing and Computer-Assisted 
Intervention. Springer, pp. 212–221.
Theis, L., Shi, W., Cunningham, A., Huszár, F., 2017. Lossy image compression with

compressive autoencoders. arXiv preprint arXiv:1703.00395.

Vaidya, A., Zhang, A., Jaume, G., Song, A.H., Ding, T., Wagner, S.J., Lu, M.Y.,

Doucet, P., Robertson, H., Almagro-Perez, C., et al., 2025. Molecular-driven 
foundation model for oncologic pathology. arXiv preprint arXiv:2501.16652.
Vorontsov, E., Bozkurt, A., Casson, A., Shaikovski, G., Zelechowski, M., Severson, K.,

Zimmermann, E., Hall, J., Tenenholtz, N., Fusi, N., et al., 2024. A foundation model 
for clinical-grade computational pathology and rare cancers detection. Nature Med. 
30 (10), 2924–2935.
Wallace, G.K., 2002. The JPEG still picture compression standard. IEEE Trans. Consum.

Electron. 38 (1), xviii–xxxiv.
Wang, H., Guo, S., Ye, J., Deng, Z., Cheng, J., Li, T., Chen, J., Su, Y., Huang, Z.,

Shen, Y., et al., 2025. SAM-Med3D: a vision foundation model for general-purpose 
segmentation on volumetric medical images. IEEE.
Wang, Y.K., Tydlitatova, L., Kunz, J.D., Oakley, G., Chow, B.K.B., Godrich, R.A.,

Lee, M.C., Aghdam, H., Bozkurt, A., Zelechowski, M., et al., 2024. Screen them 
all: high-throughput pan-cancer genetic and phenotypic biomarker screening from 
h&e whole slide images. arXiv preprint arXiv:2408.09554.
Wang, X., Wang, D., Li, X., Rittscher, J., Metaxas, D., Zhang, S., 2025. Editorial for

special issue on foundation models for medical image analysis. Medical Image Anal. 
100, 103389.
Wang, X., Zhang, X., Wang, G., He, J., Li, Z., Zhu, W., Guo, Y., Dou, Q., Li, X., Wang, D.,

et al., 2024. Openmedlab: An open-source platform for multi-modality foundation 
models in medicine. arXiv preprint arXiv:2402.18028.
Witten, I.H., Neal, R.M., Cleary, J.G., 1987. Arithmetic coding for data compression.

Commun. ACM 30 (6), 520–540.
Wu, Y., Li, S., Du, Z., Zhu, W., 2023. BROW: Better features for whole slide image

based on self-distillation. arXiv preprint arXiv:2309.08259.
Xu, H., Usuyama, N., Bagga, J., Zhang, S., Rao, R., Naumann, T., Wong, C., Gero, Z.,

González, J., Gu, Y., et al., 2024. A whole-slide foundation model for digital 
pathology from real-world data. Nature 630 (8015), 181–188.
Xu, Y., Wang, Y., Zhou, F., Ma, J., Jin, C., Yang, S., Li, J., Zhang, Z., Zhao, C., Zhou, H.,

et al., 2024. A multimodal knowledge-enhanced whole-slide pathology foundation 
model. arXiv preprint arXiv:2407.15362.
Yan, F., Wu, J., Li, J., Wang, W., Chen, Y., Wei, L., Lu, J., Chen, W., Gao, Z., Li, J.,

et al., 2025. Pathorchestra: A comprehensive foundation model for computational 
pathology with over 100 diverse clinical-grade tasks. npj Digital Medicine 8 (1), 
695.
Ying, N., Lei, Y., Zhang, T., Lyu, S., Li, C., Chen, S., Liu, Z., Zhao, Y., Zhang, G.,

2023. Cpia dataset: A comprehensive pathological image analysis dataset for 
self-supervised learning pre-training. arXiv preprint arXiv:2310.17902.
Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., Wu, Y., 2022. Coca:

Contrastive captioners are image-text foundation models. arXiv preprint arXiv:
2205.01917.
Zhang, Y., Gao, J., Tan, Z., Zhou, L., Ding, K., Zhou, M., Zhang, S., Wang, D.,

2026. Data-centric foundation models in computational healthcare: A survey. ACM 
Comput. Surveys 58 (11), 1–35.
Zhang, S., Metaxas, D., 2024. On the challenges and perspectives of foundation models

for medical image analysis. Medical Image Anal. 91, 102996.
Zhong, L., Huang, Z., Liu, Y., Liao, W., Zhang, S., Wang, G., Zhang, S., 2025. VLM-

CPL: Consensus pseudo-labels from vision-language models for annotation-free 
pathological image classification. IEEE Trans. Medical Imag..
Zhou, M., Gao, Y., Ding, K., Zhang, S., Metaxas, D.N., 2026. Ai methodologies for

multimodal pathology applications. In Less-supervised Segmentation with CNNs, 
pp. 311–320, Elsevier.
Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., Kong, T., 2021. Ibot: Image

bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832.
Zimmermann, E., Vorontsov, E., Viret, J., Casson, A., Zelechowski, M., Shaikovski, G.,

Tenenholtz, N., Hall, J., Klimstra, D., Yousfi, R., et al., 2024. Virchow2: Scaling 
self-supervised mixed magnification models in pathology. arXiv preprint arXiv:
2408.00738.

Medical Image Analysis 113 (2026) 104152

10