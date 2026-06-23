Contents lists available at ScienceDirect

Medical Image Analysis

journal homepage: www.elsevier.com/locate/media

A content-aware variable-rate framework for pathology learned image 
compression (PathoLIC)

Weiqi Li
a, Yonghao Li
a, Haoyuan Chen
a, Long Yang
a, Lin Wu
b, Zhenhui Li
c, 
Jing Ke
d,∗, Dinggang Shen
a,e,f,∗

a School of Biomedical Engineering & State Key Laboratory of Advanced Medical Materials & Devices, ShanghaiTech University, Shanghai, China
b Department of Pathology, The Third Aﬃliated Hospital of Kunming Medical University, Yunnan Cancer Hospital, Yunnan Cancer Center, Kunming, Yunnan, China
c Department of Radiology, The Third Aﬃliated Hospital of Kunming Medical University, Yunnan Cancer Hospital, Kunming, Yunnan, China
d School of Computer Science and Engineering, Shanghai Jiao Tong University, Shanghai, China
e Department of Research and Development, United Imaging Intelligence, Shanghai, China
f Shanghai Clinical Research and Trial Center, Shanghai, China

a r t i c l e  i n f o

Keywords:
Learned image compression
Whole slide image (WSI)
Variable-rate compression

a b s t r a c t

The substantial size of gigapixel whole slide images (WSIs) presents signiﬁcant challenges in terms of data stor-
age, transfer, and computational analysis. Existing image compression methods yield suboptimal compression 
ratios because they (1) overlook redundancy across neighboring/similar patches, and (2) apply uniform compres-
sion without considering content diﬀerences. To address these issues, we introduce PathoLIC (Pathology Learned 
Image Compression), a novel learning-based variable-rate compression framework tailored for WSI. Speciﬁcally, 
PathoLIC initially assigns a content score to each non-overlapping patch in the WSI, which reﬂects its diagnostic 
relevance. The compression level for each patch is determined based on the content scores, prioritizing detail 
preservation in diagnostically important regions, e.g., tumor area, while compressing more on less informative 
regions, e.g., stroma and background. Furthermore, PathoLIC employs attention mechanisms to capture relation-
ships between neighboring or similar patches, which minimize redundancy by compressing shared features. Ex-
perimental results demonstrate that PathoLIC achieves over 8× compression beyond the standard Aperio SVS for-
mat while preserving image details. Moreover, it maintains strong performance across various downstream tasks, 
such as patch-level (WSI-level) cancer subtyping and nuclei segmentation. These results demonstrate its potential 
for large-scale WSI data management. The source code will be released at https://github.com/wqli498/PathoLIC.

1.  Introduction

Digital pathology is an emerging ﬁeld in modern medicine that con-
verts histology slides into high-resolution whole slide images (WSIs) for 
computational analysis (Lu et al., 2021). With the rapid development of 
digital scanning technologies, clinical centers are now producing vast 
collections of WSIs for diagnostics, research, and long-term archiving. 
However, the enormous volume of digitized tissue slides presents ma-
jor challenges in data storage and management. A single WSI at 40×
magniﬁcation can reach resolutions of up to 80,000 × 80,000 pixels, with 
corresponding ﬁle sizes ranging from 1 to 4 GB (Van der Laak et al., 
2021). For institutions that archive tens of thousands of slides, total data 
storage requirements can scale to multiple petabytes, posing signiﬁcant 
challenges in both data storage and transfer (Association, 2019).

The Aperio ScanScope Virtual Slide (SVS) format is the most widely 
adopted standard for WSI storage. It represents WSIs as multi-resolution

∗Corresponding authors.
 
E-mail addresses: kejing@sjtu.edu.cn (J. Ke), dgshen@shanghaitech.edu.cn (D. Shen).

pyramids, compressed with either JPEG or JPEG2000. In particular, 
JPEG relies on block-wise discrete cosine transforms (DCT) with quan-
tization, while JPEG2000 employs wavelet transforms that support 
both lossy and lossless compression modes. JPEG enables fast com-
pression but often introduces visible artifacts (Wallace, 2002), whereas 
JPEG2000 achieves higher ﬁdelity at the cost of increased computa-
tional complexity (Taubman et al., 2002). Consequently, SVS ﬁles re-
main large and could beneﬁt from more advanced compression tech-
niques.

Given these limitations, there has been a growing research interest in 
learned image compression (LIC) (Ballé et al., 2018; Minnen et al., 2018; 
Cheng et al., 2020), which involves training artiﬁcial neural networks on 
large-scale image datasets. Speciﬁcally, LIC systems comprise two main 
components: an autoencoder and an entropy model (Ballé et al., 2018). 
The autoencoder compresses the input image into a compact latent rep-
resentation and then reconstructs it from this representation, while the

https://doi.org/10.1016/j.media.2026.104018
Received 4 November 2025; Received in revised form 12 February 2026; Accepted 2 March 2026

Medical Image Analysis 111 (2026) 104018

Available online 6 March 2026 
1361-8415/© 2026 Elsevier B.V. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

W. Li et al.

Fig. 1. The workﬂow of PathoLIC. Histology slides are digitized into WSIs in SVS format, which are partitioned into non-overlapping regions, each containing 
16 patches. By exploiting patch-level correlations, the network reduces redundancy and yields a binary ﬁle signiﬁcantly smaller than the original SVS. During 
decompression, PathoLIC restores the WSI with high ﬁdelity, ensuring that ﬁne-grained histological details are preserved.

entropy model estimates the probability distribution of the latent codes 
to facilitate eﬃcient compression. Values with higher probabilities are 
encoded with fewer bits. The framework is optimized to achieve a bal-
ance between compression rate and image ﬁdelity. Extensive evalua-
tions on natural image benchmarks have shown that LIC methods con-
sistently outperform conventional approaches in both perceptual quality 
and rate-distortion performance (Cheng et al., 2020).

Despite recent progress, existing image compression approaches re-
main suboptimal for WSI compression as they overlook key properties of 
WSIs. Given their enormous resolution, WSIs are generally divided into 
non-overlapping patches for data preprocessing and downstream analy-
sis (Xu et al., 2024; Wang et al., 2024). However, compressing patches 
independently using existing LIC methods ignores spatial redundancy 
across neighboring patches with similar morphology, resulting in insuf-
ﬁcient compression.

More importantly, most existing LIC models are optimized for a ﬁxed 
rate-distortion trade-oﬀ (Minnen et al., 2018; Cheng et al., 2020). There-
fore, they yield similar compression levels across diverse inputs, re-
gardless of their content variability. Nonetheless, diagnostic importance 
varies across diﬀerent regions within WSIs. For example, tumor regions 
are of higher diagnostic importance than normal or fatty tissue (Angell 
et al., 2013). Therefore, diagnostically relevant regions should be com-
pressed at higher ﬁdelity, while less informative areas can be more ag-
gressively compressed to improve overall eﬃciency. This characteristic 
of WSIs calls for content-aware compression frameworks with variable 
compression ratios. While recent variable-rate LIC models (Song et al., 
2021; Cai et al., 2024) support ﬂexible compression levels via introduc-
ing a global rate-distortion hyperparameter, they cannot automatically 
assign compression levels across WSI patches based on their contents.

To address these challenges, we propose PathoLIC, a content-aware, 
variable-rate compression framework tailored for WSIs. As illustrated 
in Fig. 1, our framework simultaneously processes 16 patches as input, 
considering (1) the diagnostic relevance of each patch and (2) the spatial 
correlations among neighboring or similar patches. PathoLIC produces a 
highly compressed binary representation, signiﬁcantly reducing storage 
requirements while preserving ﬁne-grained histological details. The key 
contributions of PathoLIC can be summarized as follows:

1. We present a learning-based variable-rate framework for WSI-level 
image compression, namely PathoLIC. It computes patch-level con-
tent scores to guide adaptive compression, preserving critical con-
tent at higher ﬁdelity while allocating fewer bits to less informative 
areas.
2. We leverage attention mechanisms to model correlations among 
neighboring or similar patches, thereby reducing redundancy 
through the compression of shared features.
3. We propose a region-wise WSI bitstream format to combine la-
tent bitstreams with content-score metadata, for enabling eﬃcient 
coordinate- and score-based decoding.

To the best of our knowledge, PathoLIC is the ﬁrst framework that 
leverages content-aware strategies in whole slide image compression. 
PathoLIC is validated using both compression metrics and clinically rel-
evant downstream tasks. Speciﬁcally, we benchmark its rate-distortion 
performance against conventional compression approaches and state-

of-the-art LIC methods. To assess its practical utility, we further test its 
robustness across a diverse set of diagnostic tasks using public and in-
house datasets covering multiple cancer types.

Extensive experiments demonstrate that PathoLIC achieves an aver-
age compression ratio 8× higher than the SVS format, while maintain-
ing comparable performance on various downstream tasks. These results 
highlight the potential of PathoLIC as a practical and scalable solution 
for eﬃcient WSI data management in digital pathology.

2.  Related work

2.1.  Digital pathology and computational tasks

WSI has reshaped computational pathology by facilitating auto-
mated analysis of histological specimens for applications such as can-
cer diagnosis, tumor grading, and cellular characterization (Madab-
hushi and Lee, 2016; Litjens et al., 2017; Komura and Ishikawa, 2018). 
These diagnostic tasks operate across multiple spatial resolutions, rang-
ing from individual patches to entire WSIs. At the cellular level, ﬁne-
grained analyses such as nucleus segmentation and phenotype classiﬁ-
cation are essential for quantifying the tumor microenvironment (Kumar 
et al., 2017; Graham et al., 2019; Gamper et al., 2019). Patch-level meth-
ods focus on performing mitosis detection, histological grading, and tis-
sue classiﬁcation (Aresta et al., 2019; Hou et al., 2016) with localized 
tissue regions as inputs. At the WSI level, models leverage global con-
textual information to perform cancer classiﬁcation and tumor subtyp-
ing (Campanella et al., 2019; Lu et al., 2021) without requiring ﬁne-
grained manual annotations.

2.2.  Traditional image compression

Traditional image compression algorithms such as JPEG (Wallace, 
2002) and JPEG2000 (Taubman et al., 2002) employ hand-crafted trans-
form coding pipelines based on the discrete cosine transform (DCT) or 
wavelet transforms, followed by quantization and entropy coding. In 
digital pathology, WSIs are typically stored using TIFF-based container 
formats (e.g., SVS, NDPI) that organize large images as pyramids. Lossy 
JPEG compression, which divides images into blocks and applies the 
DCT followed by quantization and entropy coding, is commonly used 
to balance ﬁle size and visual quality (Farahani et al., 2015). However, 
WSI ﬁles still occupy substantial storage space and can be further com-
pressed to reduce storage and transmission costs.

2.3.  Learned image compression

Learned image compression (LIC) frameworks employ end-to-end 
trainable autoencoder architectures, where the encoder, decoder, and 
entropy model are jointly optimized to balance compression rate and 
image ﬁdelity (Ballé et al., 2017; Toderici et al., 2017). The introduc-
tion of hyperprior models (Ballé et al., 2018), typically implemented 
with convolutional neural networks (CNNs), further enhances compres-
sion eﬃciency by enabling more accurate entropy estimation. Recently, 
transformer-based architectures have been investigated to further im-
prove compression performance (Zhu et al., 2022; Liu et al., 2023).

Medical Image Analysis 111 (2026) 104018

W. Li et al.

Fig. 2. An overview of the proposed framework for WSI compression. (a) Generation of content scores for WSI-level inference. A pretrained foundation model (e.g., 
CHIEF (Wang et al., 2024)) is used to produce an attention map, highlighting diagnostically salient areas. The map is normalized and converted into patch-level 
content scores. (b) Detailed architecture of PathoLIC. The framework consists of an encoder 𝑔𝑎, a hyper-encoder ℎ𝑎, a decoder 𝑔𝑠, a hyper-decoder ℎ𝑠, a channel-wise 
context model, and quality control modules (QCMs). QCMs are inserted at multiple stages to adapt feature representations based on input content scores. During 
training, AE and AD are excluded, and the rate is estimated using the entropy model (* means inference only).

Fig. 3. Architecture of the proposed QCM. Content scores 𝑄 are mapped by an 
MLP to generate scale (𝛼) and bias (𝛽) parameters, which modulate the input 
feature map via an aﬃne transformation for content-aware compression.

However, early LIC methods are trained for a ﬁxed compression level, 
which limits their ﬂexibility. To address this limitation, variable-rate 
frameworks have been proposed to adapt bitrates based on external 
control signals. For instance, QmapCompression (Song et al., 2021) 
and I2C (Cai et al., 2024) employ Spatially Feature Transforms (SFT) 
or Invertible Activation Transformations (IAT) using convolutional 
networks to perform pixel-wise modulation based on spatial quality
maps.

Unlike existing methods, our framework accounts for the unique 
characteristics of whole slide images (WSIs) to improve compres-
sion eﬃciency. First, it can automatically assign patch-level com-
pression rates according to the diagnostic importance of each patch.
Second, the model leverages a Transformer to capture spatial correla-
tions among neighboring or similar patches, thereby further reducing
redundancy.

3.  Method

We illustrate the overall framework of PathoLIC in Fig. 2. The input 
WSI is ﬁrst partitioned into non-overlapping patches, each assigned with 
a content score according to its clinical relevance. Neighboring patches 
are then grouped into ﬁxed-size regions of 𝐿× 𝐿 patches. PathoLIC com-
presses each region into a bitstream under the guidance of patch-level 
content scores. The compressed WSI is obtained by aggregating the bit-
streams generated from all regions, which can be decompressed to re-
construct the original WSI. Section 3.1 introduces the measurement of 
patch-level diagnostic value. The generated patch-level content scores 
guide the subsequent context-aware compression and decompression 
processes (see Sections 3.2 and 3.3), while the training and inference 
strategies of PathoLIC are detailed in Section 3.4.

3.1.  Assessment of patch-level diagnostic relevance

Our objective is to achieve variable-rate compression of WSI based 
on the patch-level diagnostic importance. Traditional measures, such as 
entropy, are eﬀective at capturing pixel-level statistics but fail to reﬂect 
the clinical relevance of diﬀerent regions within the WSI. In contrast, 
recent pathology foundation models can automatically extract disease-
relevant features to perform WSI-level predictive tasks (Xu et al., 2024; 
Wang et al., 2024). Thus, we adopt CHIEF, a pretrained foundation 
model (Wang et al., 2024), to estimate patch-level content scores. As 
illustrated in Fig. 2(a), patch-level features extracted by CHIEF are pro-
cessed through an attention module composed of multilayer perceptron 
(MLP) layers and a Tanh-Sigmoid gating mechanism, which produces 
an attention score for each patch. To obtain the ﬁnal content scores 
for variable-rate compression, attention values are projected using an

Medical Image Analysis 111 (2026) 104018

W. Li et al.

exponential function. In practice, high content scores align with diag-
nostically salient areas such as tumors or inﬂammatory regions, whereas 
low content scores are associated with less informative tissues such as 
adipose or stromal areas.

3.2.  Content-aware WSI compression

A distinctive characteristic of WSIs is that adjacent patches often 
share similar tissue morphology. Leveraging this property, our ﬁrst key 
innovation is to exploit the correlations among neighboring patches dur-
ing compression, thereby enhancing compression eﬃciency. As shown 
in Fig. 2(b), an encoder 𝑔𝑎(⋅) ﬁrst maps the input region 𝑥 of 𝐿× 𝐿
patches into latent features:

𝑦= 𝑔𝑎(𝑥),
(1)

where 𝑔𝑎 is composed of Residual Blocks with Stride (RBS) (Cheng et al., 
2020) and Transformer-CNN Mixture (TCM) blocks (Liu et al., 2023). 
Speciﬁcally, the RBS modules apply residual downsampling via stridden 
convolutions and progressively reduce the spatial resolution of feature 
maps. The TCM block employs a hybrid design that combines convolu-
tional layers with Transformer layers. Convolutional layers capture local 
patterns like nuclei boundaries and textures, while Transformer layers 
model long-range spatial dependencies, allowing patches to reﬁne their 
representations through global interactions. By combining the comple-
mentary strengths of convolution and self-attention, the proposed frame-
work achieves a balance between local detail preservation and global 
contextual representation.

Another key aspect of our framework is variable-rate compression 
guided by patch-level diagnostic relevance. In particular, the latent fea-
tures are modulated by the proposed Quality Control Module (QCM) 
under the guidance of content scores 𝑄 of the input region:

𝑦control = QCM𝑦(𝑦, 𝑄).
(2)

As illustrated in Fig. 3, QCM applies a feature-wise aﬃne transfor-
mation conditioned on 𝑄. Given an input feature 𝐹, QCM generates 
channel-wise scale and bias parameters using a lightweight MLP:

[𝛼, 𝛽] = MLP(𝑄), 𝛼, 𝛽∈ℝ𝐶,
(3)

𝐹modulated = (𝛼+ 1) ⊙𝐹+ 𝛽,
(4)

where 𝐶 is the channel dimensionality and ⊙ denotes the element-wise 
channel scaling. The residual-style scaling (𝛼+ 1) preserves the stability 
of the original features while allowing controlled ampliﬁcation or atten-
uation with respect to the target quality. The bias 𝛽 further introduces 
a ﬂexible shift, allowing the network to adjust feature distributions as 
needed. Together, these two operations provide ﬁne-grained modula-
tion of intermediate representations, allowing QCM to seamlessly adapt 
across diﬀerent compression levels.

To improve entropy modeling, a hyper-encoder ℎ𝑎(⋅) extracts side 
information from the modulated latent features:

𝑧= ℎ𝑎(𝑦control),
(5)

which is further processed by another QCM:

𝑧control = QCM𝑧(𝑧, 𝑄),
(6)

and then quantized:

̂𝑧←Quantize(𝑧control).
(7)

The hyperprior pathway, therefore, provides accurate distributional 
priors for entropy coding of the main latent information. Importantly, 
quantization (Q), arithmetic encoding (AE) and decoding (AD) are uti-
lized exclusively during inference. Speciﬁcally, AE converts ̂𝑦, ̂𝑧, and the 
corresponding content scores into binary streams for storage, while AD 
reconstructs features from these streams. Consequently, the ﬁnal com-
pressed WSI is represented as the aggregation of the binary streams from 
all input WSI regions.

3.3.  Content-aware WSI decompression

The decoder of PathoLIC mirrors the encoding process to reconstruct 
image patches, with reconstruction ﬁdelity explicitly guided by the con-
tent scores. In particular, the hyper-decoder ℎ𝑠(⋅) estimates the channel-
wise spatial statistics-speciﬁcally, the mean (𝜇) and variance (𝜎)-from

̂𝑧, which are subsequently modulated by the content scores:

𝜇, 𝜎= ℎ𝑠(̂𝑧),
(8)

(𝜇control, 𝜎control) = QCM𝜇&𝜎(𝜇, 𝜎, 𝑄).
(9)

The modulated latent 𝑦control is then quantized and passed through 
the channel-wise context model (CCM) for entropy coding:

̂𝑦= CCM(Quantize(𝑦control); 𝜇control, 𝜎control),
(10)

where the CCM employs masked convolutions and attention layers to 
capture both local and non-local dependencies across latent channels 
(see (Liu et al., 2023) for details). By leveraging these dependencies, the 
CCM produces more accurate probability estimates, further reducing the 
expected bits.

Finally, the quantized latent ̂𝑦 is modulated once more:

̂𝑦control = QCM ̂𝑦( ̂𝑦, 𝑄),
(11)

before being decoded by 𝑔𝑠(⋅) to reconstruct the input region:

̂𝑥= 𝑔𝑠( ̂𝑦control).
(12)

3.4.  Optimization and inference strategy

It is important to note that the content scores derived from founda-
tion models are applied only at the inference stage to guide compres-
sion. During training, the model instead relies on the content scores 
randomly sampled from a ﬁxed range, which enhances generalization 
of PathoLIC across diverse compression levels. For each patch 𝑥, a con-
tent score 𝑞 is sampled from a uniform distribution, 𝑞∼[0, 1], and 
mapped to the corresponding weighting factor 𝜆∈[0.0025, 0.04] via an

exponential projection: 𝜆(𝑞) = 0.0025
(
0.04
0.0025

)𝑞

. Crucially, we treat the

content score 𝑄 as a continuous ‘soft prior’ for rate allocation. The map-
ping is bounded (𝜆∈[0.0025, 0.04]) to establish a quality ﬂoor, prevent-
ing overly aggressive compression in low-score regions regardless of the 
foundation model’s bias. The model is then optimized end-to-end to min-
imize the rate-distortion loss:

= 𝔼𝑥,𝑄[𝑅(𝑥; 𝑄) + 𝜆(𝑄) 𝐷(𝑥, ̂𝑥(𝑄))],
(13)

where 𝑅(⋅) denotes the estimated rate term, computed as the sum of 
the negative log-likelihoods (expected bit cost) of the quantized latents 
and hyper-latents, 𝑄 denotes the set of content scores of all patches 
within the input region, 𝐷(⋅) is the distortion measured by MSE, and 𝜆(𝑄)
controls the trade-oﬀ between compression eﬃciency and reconstruc-
tion ﬁdelity. This setup forces the model to learn a continuous mapping 
from content scores to diﬀerent compression levels, thereby improving 
its generalization across diverse compression settings. During inference, 
each patch is assigned a content score derived from the attention values 
of the pathology foundation model (Wang et al., 2024), as outlined in 
Section 3.1. The same exponential mapping is then applied to compute 
𝜆(𝑄).

3.5.  WSI bitstream storage and decompression

For model deployment, we design a region-wise WSI bitstream for-
mat to enable eﬃcient storage and access of compressed WSIs in stan-
dard digital pathology settings. Speciﬁcally, the WSI is partitioned into 
non-overlapping regions, where every 4 × 4 adjacent patches are aggre-
gated into a single compressed bitstream segment. A lightweight index 
table is maintained in the ﬁle header to map the spatial coordinates 
of these units to their corresponding byte oﬀsets, enabling high-speed

Medical Image Analysis 111 (2026) 104018

W. Li et al.

Fig. 4. Visual comparison of the proposed framework guided by two foundation models (CHIEF vs. TITAN). Both models consistently highlight tumor regions (red) 
and suppress background.

Table 1 
Overview of dataset usage and data splits. The underlined values indicate the data used to train the 
PathoLIC compression model.

Dataset
 Unit
 Training
 Validation
 Test

Downstream
Evaluation

TCGA-BRCA
 WSI
623
 208
 209

WSI-level subtyping,
Survival prediction

TCGA-NSCLC
 WSI
620
 206
 207

WSI-level subtyping,
Survival prediction
 TCGA-RCC
 WSI
 561
 188
 188
 WSI-level subtyping

BACH

WSI
 18
 2
 8

Patch-level classiﬁcation
Patch (512 × 512)
 12,384
 4127
 4130
Patch (1,024 × 1,024)
 3060
 1019
 1023

In-house

WSI
18
 4
 8

Patch-level classiﬁcation
Patch (512 × 512)
 3940
 1313
 1315
Patch (1,024 × 1,024)
 122
 40
 43
 PanNuke
 Patch
 5179
 –
 2722
 Cell-level segmentation
 MNS
 Patch
 431
 –
 209
 Cell-level segmentation
 NCT-CRC-HE-100K
 Patch
 100,000
 –
 7180
 ROI retrieval

random access without global scanning. Based on the practical needs of 
clinicians during diagnosis, the decoding scheme is designed to provide 
two key functionalities:

• WSI-level decompression: It supports decoding the complete bit-
stream back into standard pyramidal SVS formats, ensuring com-
patibility with existing commercial slide viewers (e.g., Aperio Im-
ageScope, QuPath).

• Region-level decompression: Leveraging the spatial independence 
of our block-based compression, the interface allows users to de-
code speciﬁc regions based on coordinates or diagnostic scores with-
out processing the entire ﬁle. For example, clinicians can choose 
to decode only the top 10% diagnostically relevant patches for 
rapid assessment, signiﬁcantly reducing I/O latency in bandwidth-
constrained environments.

4.  Dataset preparation and experimental settings

4.1.  Datasets for training and evaluation

We ﬁrst introduce the training dataset of PathoLIC, followed by a 
description of diverse public and in-house datasets used to evaluate our 
framework across multiple tissue types and downstream tasks. We eval-
uate PathoLIC following a hierarchy of downstream clinical tasks, in-
cluding WSI-level tasks (subtyping and survival prediction), patch-level 
tasks (histology classiﬁcation and ROI retrieval), and cell-level task (nu-
clei segmentation).

4.1.1.  Datasets for compression

As shown in Table 1, PathoLIC is trained on 73,730 patches, 
with 1000 patches reserved for validation and 3694 for test. All 
patches (1, 024 × 1, 024 pixels at 40× magniﬁcation) are extracted from

Medical Image Analysis 111 (2026) 104018

W. Li et al.

Fig. 5. Rate-distortion comparison. PathoLIC achieves superior Multi-Scale Structural Similarity Index (MS-SSIM) and Peak Signal-to-Noise Ratio (PSNR) across a 
range of Bits-Per-Pixel (BPP) compared to traditional compression methods (JPEG, JPEG2000) and other LIC methods (QmapCompression, I2C).

Fig. 6. Region-level visualization across diﬀerent 𝜆 values using our model. 
The ﬁrst row shows original and reconstructed regions, the second row presents 
zoomed-in diagnostically relevant patches, and the third row depicts diﬀerence 
maps between original images and reconstruction images.

TCGA-BRCA (Weinstein et al., 2013), TCGA-NSCLC, as well as from 
an in-house dataset collected at Yunnan Cancer Hospital. Importantly, 
patches used to train the compression model are excluded from the val-
idation of downstream tasks.

4.1.2.  Datasets for WSI-level subtyping

We employ three datasets from The Cancer Genome Atlas (TCGA) for 
WSI-level evaluation. The detailed data splits for training, validation, 
and test are summarized in Table 1.

• TCGA-BRCA: Consists of invasive breast carcinoma WSIs utilized for 
tumor subtyping tasks.

• TCGA-NSCLC: Includes WSIs of lung adenocarcinoma (LUAD) and 
lung squamous cell carcinoma (LUSC) for lung cancer subtype clas-
siﬁcation.

• TCGA-RCC: Covers three renal cell carcinoma subtypes: kid-
ney renal clear cell carcinoma (KIRC), kidney renal papil-
lary cell carcinoma (KIRP), and kidney chromophobe carcinoma
(KICH).

4.1.3.  Datasets for WSI-level survival prediction

We conduct WSI-level survival prediction on TCGA-BRCA and TCGA-
NSCLC datasets. For each dataset, we use the same WSI set as in the 
subtyping task and adopt the available survival endpoints for evaluation. 
Model performance is evaluated using the Concordance Index (C-index).

4.1.4.  Datasets for patch-level classiﬁcation

We utilize two datasets for patch-level histology classiﬁcation. Non-
overlapping patches are cropped at two resolutions (512 × 512 and 
1, 024 × 1, 024) to evaluate performance across diﬀerent scales. The spe-
ciﬁc distribution of WSIs and patches is detailed in Table 1.

• BACH: Derived from the BACH dataset, this subset contains expert-
annotated WSIs processed for multi-class histology classiﬁcation.

• In-house dataset: This dataset comprises breast cancer WSIs col-
lected from Yunnan Cancer Hospital, curated for a binary classiﬁca-
tion task (normal vs. invasive).

4.1.5.  Datasets for ROI retrieval

We utilize the NCT-CRC-HE-100K dataset (Kather et al., 2018) to 
evaluate patch-level retrieval performance, strictly following the data 
distribution and experimental settings of UNI (Chen et al., 2024). This 
dataset consists of 107,180 non-overlapping patches (224 × 224 pixels 
at 0.5 mpp) extracted from H&E-stained colorectal cancer WSIs, cover-
ing 9 tissue classes (e.g., adipose, mucus, cancer-associated stroma). We 
use the oﬃcial case-stratiﬁed split, comprising a training set of 100,000 
patches (NCT-CRC-HE-100K) and a test set of 7180 patches (CRC-VAL-
HE-7K).

4.1.6.  Datasets for cell-level segmentation

We assess ﬁne-grained segmentation performance using two bench-
marks. The quantity of patches used for training and test is listed in 
Table 1.

• PanNuke: This dataset contains semi-automatically generated H&E-
stained patches sampled from 19 diﬀerent tissue types (Gamper 
et al., 2019).

• Merged nuclei segmentation benchmark (MNS): To rigorously 
evaluate generalization, we construct a uniﬁed benchmark merging 
four public datasets: CoNSeP (Graham et al., 2019), Lizard (Graham 
et al., 2021), MoNuSeg (Kumar et al., 2017), and MoNuSAC (Verma 
et al., 2021).

Medical Image Analysis 111 (2026) 104018

W. Li et al.

Table 2 
Storage statistics (GB) and compression ratios (Original/Compressed) for the TCGA-NSCLC dataset. PathoLIC (guided 
by CHIEF and TITAN) is benchmarked against the standard JPEG baseline across training, validation, and test splits to 
demonstrate overall storage eﬃciency.

Split
 Original (GB)
 PathoLICCHIEF (GB)
 RatioCHIEF
 PathoLICTITAN (GB)
 RatioTITAN
 JPEG (GB)
 RatioJPEG
 Training
 470.60
 88.36
 5.33×
 52.11
 9.03×
 98.07
 4.80×
 Validation
 138.95
 26.61
 5.22×
 15.63
 8.89×
 28.35
 4.90×
 Test
 148.92
 28.16
 5.29×
 16.60
 8.97×
 30.01
 4.96×
 All
 758.47
 143.13
 5.30×
 84.34
 8.99×
 150.58
 5.04×

Table 3 
Comparison of average encoding/decoding time (seconds per patch) 
and model size (MB).

Method
 Enc. Time (s)
 Dec. Time (s)
 Model Size (MB)
 JPEG
 0.035
 0.004
 –
 JPEG2000
 0.206
 0.001
 –
 QmapCompression
 0.207
 0.117
 316
 I2C
 9.114
 21.831
 576
 Ours
 0.293
 0.310
 879

4.2.  Implementation details

4.2.1.  Model architecture and training

We implement our LIC framework with the PyTorch library and con-
duct all experiments on a single NVIDIA A100 GPU with 80 GB mem-
ory under Python 3.10. The architecture follows the hybrid Transformer-
CNN framework of Liu et al. (2023), employing Swin Transformer blocks 
(with window sizes of 8 and 4 for main and hyperprior paths, respec-
tively) and latent dimensions of 320 and 192 for 𝑦 and 𝑧, respectively. 
Training is performed for 80,000 iterations using Kingma and Ba (2014) 
with a constant learning rate 4 × 10−5.

4.2.2.  WSI preprocessing for training

WSIs are ﬁrst partitioned into 256 × 256 patches at 40× magniﬁca-
tion. Foreground tissue is identiﬁed using the pipeline in CLAM (Lu 
et al., 2021). These foreground patches are then grouped into training 
regions of size 1, 024 × 1, 024, each containing 16 patches.

4.2.3.  Inference strategies

We employ two distinct inference modes tailored to diﬀerent down-
stream tasks:

• Fixed high-ﬁdelity compression: For patch- and cell-level tasks 
that demand maximal detail preservation (patch-level classiﬁcation, 
ROI retrieval, and nuclei segmentation), we uniformly assign the 
highest content score (𝑞= 1) to all patches to minimize information 
loss.

• Content-aware variable-rate compression: For WSI-level analy-
ses (subtyping and survival prediction), we employ the proposed 
content-aware strategy. We generate attention maps using pretrained 
foundation models, CHIEF (Wang et al., 2024) and TITAN (Ding 
et al., 2025).

4.3.  Experimental protocol

4.3.1.  Comparison methods

To benchmark PathoLIC, we compare it against four representative 
compression algorithms, including two conventional approaches and 
two learned image compression (LIC) methods:

• JPEG (Wallace, 2002): The widely used DCT-based codec, evaluated 
with quality settings from 10 to 90.

• JPEG2000 (Taubman et al., 2002): A wavelet-based standard oﬀer-
ing improved performance over JPEG.

• QmapCompression (Song et al., 2021): A LIC method using a spatial 
quality map for ﬁne-grained rate control.

• I2C (Cai et al., 2024): An invertible neural network-based codec en-
abling continuous rate control via normalizing ﬂows.

4.3.2.  Evaluation metrics

We assess performance using metrics for both reconstruction ﬁdelity 
and downstream task preservation.

• Reconstruction ﬁdelity: We measure standard image quality using 
Peak Signal-to-Noise Ratio (PSNR) and Multi-Scale Structural Simi-
larity (MS-SSIM).

• WSI-level classiﬁcation: Performance is evaluated using Accuracy, 
Balanced Accuracy (BACC), Area Under the Receiver Operating 
Characteristic Curve (AUROC), and Area Under the Precision-Recall 
Curve (AUPRC).

• Survival prediction: We employ the Concordance Index (C-index) 
to evaluate the model’s ability to correctly rank the survival times of 
pairs of patients.

• Patch-level classiﬁcation: We report patch-wise aggregated Accu-
racy, BACC, and AUROC.

• ROI retrieval: Following standard protocols used in UNI (Chen et al., 
2024), we measure retrieval performance using Top-𝐾 Accuracy 
(ACC@𝐾) for 𝐾∈{1, 3, 5} and majority vote of the top-5 retrieved 
images (MVACC@5).

• Cell-level segmentation: We use Dice coeﬃcient, Intersection over 
Union (IoU), precision, recall, and speciﬁcity.

5.  Experiments and results

We now present a comprehensive evaluation of the PathoLIC frame-
work. We ﬁrst assess its fundamental rate-distortion performance and 
computational eﬃciency, followed by an in-depth analysis of its impact 
on a hierarchy of downstream clinical tasks. Finally, we present ablation 
studies to validate our core architectural components.

5.1.  Rate-distortion performance and eﬃciency

As illustrated in Fig. 5 and summarized in Table 3, PathoLIC con-
sistently outperforms all comparison methods in the high-ﬁdelity set-
ting (0.23-0.46 BPP), which is the most critical for preserving diagnos-
tic information. For instance, at diagnostically relevant rate of 0.28 BPP, 
PathoLIC achieves a PSNR of 40.6 dB and an MS-SSIM of 0.990, surpass-
ing both QmapCompression (40.2 dB / 0.989) and I2C (39.7 dB / 0.989). 
Traditional codecs such as JPEG and JPEG2000 excel in speed but suf-
fer from lower reconstruction ﬁdelity. Among the learned approaches, 
PathoLIC achieves a strong balance of high ﬁdelity, eﬃcient runtime, 
and moderate model size, making it particularly suitable for real-world 
digital pathology workﬂows. QmapCompression oﬀers faster inference 
but lags behind in R-D performance, whereas I2C demonstrates consid-
erably lower speed due to the large parameter count in its invertible 
CNN architecture. By contrast, our lightweight transformer design en-
ables eﬃcient processing while maintaining competitive reconstruction 
quality.

Medical Image Analysis 111 (2026) 104018

W. Li et al.

Table 4 
WSI-level subtype classiﬁcation performance on TCGA-BRCA, TCGA-RCC, and TCGA-NSCLC. PathoLIC (guided by CHIEF and TITAN) is benchmarked 
against the JPEG baseline across diverse input conﬁgurations.

Compression Method
Training Input
Test Input
 BRCA
 RCC
 NSCLC
 BACC  ACC
 AUROC  AUPRC  BACC  ACC
 AUROC  AUPRC  BACC  ACC
 AUROC  AUPRC
 -
 Original
 Original
 0.907
 0.880  0.943
 0.898
 0.937
 0.952  0.985
 0.946
 0.961
 0.961  0.987
 0.979

PathoLICCHIEF

Original
 Compressed
 0.907
 0.880  0.949
 0.919
 0.957
 0.947  0.985
 0.942
 0.951
 0.952  0.982
 0.977
 Compressed
 Original
 0.937
 0.928  0.960
 0.930
 0.961
 0.957  0.995
 0.978
 0.957
 0.956  0.986
 0.979
 Compressed
 Compressed
 0.922
 0.933  0.956
 0.924
 0.964
 0.952  0.983
 0.947
 0.956
 0.956  0.987
 0.979

PathoLICTITAN

Original
 Compressed
 0.916
 0.909  0.951
 0.884
 0.961
 0.960  0.985
 0.917
 0.956
 0.957  0.990
 0.990
 Compressed
 Original
 0.905
 0.919  0.941
 0.883
 0.940
 0.946  0.992
 0.963
 0.952
 0.952  0.994
 0.994
 Compressed
 Compressed
 0.911
 0.928  0.943
 0.886
 0.959
 0.949  0.978
 0.921
 0.952
 0.952  0.991
 0.992

JPEG

Original
 Compressed
 0.893
 0.914  0.952
 0.910
 0.950
 0.962  0.986
 0.936
 0.943
 0.959  0.990
 0.992
 Compressed
 Original
 0.890
 0.895  0.950
 0.909
 0.930
 0.930  0.993
 0.971
 0.945
 0.937  0.995
 0.995
 Compressed
 Compressed
 0.895
 0.876  0.943
 0.903
 0.952
 0.911  0.978
 0.933
 0.944
 0.916  0.991
 0.993

Table 5 
WSI-level survival prediction performance on TCGA-BRCA and TCGA-NSCLC 
datasets. The C-Index (Mean ± Std) is reported to evaluate the robustness of 
prognostic signals under compression. PathoLIC (guided by CHIEF and TITAN) 
is benchmarked against the JPEG baseline across diverse input conﬁgurations.

Dataset  Compression Method  Training Input  Test Input
 C-Index (Mean ± Std)

BRCA

–
 Original
 Original
 0.665 ± 0.090

PathoLICCHIEF

Original
 Compressed  0.658 ± 0.104
 Compressed
 Original
 0.673 ± 0.084
 Compressed
 Compressed  0.696 ± 0.070

PathoLICTITAN

Original
 Compressed  0.650 ± 0.110
 Compressed
 Original
 0.666 ± 0.089
 Compressed
 Compressed  0.688 ± 0.074

JPEG

Original
 Compressed  0.642 ± 0.109
 Compressed
 Original
 0.654 ± 0.091
 Compressed
 Compressed  0.682 ± 0.069

NSCLC

–
 Original
 Original
 0.600 ± 0.058

PathoLICCHIEF

Original
 Compressed  0.587 ± 0.073
 Compressed
 Original
 0.591 ± 0.058
 Compressed
 Compressed  0.592 ± 0.041

PathoLICTITAN

Original
 Compressed  0.578 ± 0.078
 Compressed
 Original
 0.583 ± 0.062
 Compressed
 Compressed  0.583 ± 0.045

JPEG

Original
 Compressed  0.570 ± 0.078
 Compressed
 Original
 0.570 ± 0.063
 Compressed
 Compressed  0.577 ± 0.043

Fig. 7. Region-level comparison across methods. All models are shown at a 
similar bit-per-pixel (BPP) rate. Notably, even under a lower or comparable BPP, 
our method achieves higher perceptual quality and preserves more structural 
details, as reﬂected by the higher PSNR values and the reduced residuals in the 
diﬀerence maps.

Qualitative eﬀect of rate control. Complementing the quan-
titative curves, Fig. 6 illustrates the reconstruction quality across 
diﬀerent 𝜆 values. Higher 𝜆 values produce sharper reconstruc-
tions with ﬁner texture detail and reduced reconstruction er-
rors, conﬁrming that PathoLIC enables controlled variable-rate 
compression.

Qualitative comparison at matched bitrate. Fig. 7 compares 
PathoLIC with representative baselines at a matched bitrate budget. 
PathoLIC yields fewer visible structural deviations and blocking arti-
facts in the zoomed regions compared to baselines, which is consistent 
with the higher MS-SSIM and PSNR values reported in the quantitative 
analysis.

5.2.  Impact on downstream clinical tasks

Preserving diagnostic performance is essential in medical image 
compression. Consequently, we evaluate PathoLIC across a hierarchy of 
downstream tasks, including WSI-level subtyping and survival predic-
tion, patch-level classiﬁcation and ROI retrieval, and cell-level nuclei 
segmentation.

5.2.1.  WSI-level cancer subtyping

We assess the impact of our content-aware compression on cancer 
subtyping using the Prov-GigaPath foundation model (Xu et al., 2024). 
Prov-GigaPath employs a two-stage pretraining approach: a tile en-
coder based on DINOv2 captures local patterns at the patch level, while 
a slide encoder utilizing the LongNet architecture models global pat-
terns across the entire slide. We evaluate three compression schemes: 
PathoLIC guided by CHIEF scores (PathoLICCHIEF), PathoLIC guided by 
TITAN scores (PathoLICTITAN), and the standard JPEG codec. For each, 
we test four scenarios: training and test on original WSIs (i.e., original 
→ original), and three other combinations involving compressed data. 
Besides, Table 2 summarizes the original and compressed sizes of the 
test sets for TCGA-NSCLC datasets. Notably, PathoLICTITAN achieves the 
highest average compression ratio, outperforming both PathoLICCHIEF
and the JPEG baseline.

TCGA-BRCA results (Table 4): In the BRCA dataset, PathoLICCHIEF
demonstrates superior robustness, with the Compressed → Original set-
ting yielding improvements across all metrics compared to the un-
compressed baseline. PathoLICTITAN maintains competitive performance 
(AUROC 0.951) despite its aggressive compression, comparable to JPEG 
(AUROC 0.952).

TCGA-RCC results (Table 4): High robustness is observed across all 
methods for RCC subtyping. PathoLICCHIEF and PathoLICTITAN achieve 
AUROCs of 0.985 and 0.985, respectively, in the Original → Compressed 
setting, closely matching the uncompressed performance (0.985) and 
the JPEG baseline (0.986).

TCGA-NSCLC results (Table 4): For NSCLC, while models trained 
on original data show slight sensitivity to compression, training on

Medical Image Analysis 111 (2026) 104018

W. Li et al.

Table 6 
Patch-level classiﬁcation performance on the BACH dataset across diﬀerent patch resolutions (512 ×
512, 1, 024 × 1, 024) and backbone architectures (ResNet-18, ResNet-50). PathoLIC is benchmarked 
against the JPEG baseline to evaluate the impact of compression artifacts on diagnostic accuracy 
across diverse input conﬁgurations.

Model
(patch size)
 Compression Method
 Training Input
 Test Input
 ACC
 BACC
 AUROC

ResNet-18
(512 × 512)

–
 Original
 Original
 0.749
 0.734
 0.918

PathoLIC

Original
 Compressed
 0.648
 0.641
 0.864
 Compressed
 Original
 0.669
 0.650
 0.867
 Compressed
 Compressed
 0.684
 0.656
 0.880

JPEG

Original
 Compressed
 0.637
 0.604
 0.840
 Compressed
 Original
 0.642
 0.615
 0.843
 Compressed
 Compressed
 0.700
 0.691
 0.884

ResNet-50
(512 × 512)

–
 Original
 Original
 0.750
 0.741
 0.916

PathoLIC

Original
 Compressed
 0.631
 0.623
 0.871
 Compressed
 Original
 0.671
 0.634
 0.875
 Compressed
 Compressed
 0.696
 0.673
 0.886

JPEG

Original
 Compressed
 0.630
 0.622
 0.854
 Compressed
 Original
 0.662
 0.613
 0.862
 Compressed
 Compressed
 0.693
 0.684
 0.888

ResNet-18
(1,024 × 1,024)

–
 Original
 Original
 0.698
 0.666
 0.889

PathoLIC

Original
 Compressed
 0.625
 0.603
 0.845
 Compressed
 Original
 0.663
 0.604
 0.860
 Compressed
 Compressed
 0.685
 0.667
 0.877

JPEG

Original
 Compressed
 0.629
 0.606
 0.827
 Compressed
 Original
 0.636
 0.593
 0.811
 Compressed
 Compressed
 0.684
 0.648
 0.865

ResNet-50
(1,024 × 1,024)

–
 Original
 Original
 0.730
 0.716
 0.902

PathoLIC

Original
 Compressed
 0.635
 0.627
 0.847
 Compressed
 Original
 0.660
 0.629
 0.848
 Compressed
 Compressed
 0.689
 0.679
 0.879

JPEG

Original
 Compressed
 0.592
 0.593
 0.825
 Compressed
 Original
 0.644
 0.615
 0.836
 Compressed
 Compressed
 0.663
 0.659
 0.867

compressed 
data 
eﬀectively 
recovers 
performance. 
Notably, 
PathoLICTITAN achieves an AUROC of 0.990 in the Original →
Compressed setting, matching the performance of JPEG (0.990). 
This conﬁrms that PathoLIC can deliver high-ﬁdelity diagnostic 
features comparable to standard codecs but with reduced storage
overhead.

5.2.2.  Survival prediction

We further evaluate PathoLIC on WSI-level survival prediction, a 
task highly sensitive to global tissue context, using the TCGA-BRCA 
and TCGA-NSCLC datasets. Table 5 reports the C-index under diﬀer-
ent Training → Test input formats (Original vs. Compressed), enabling 
a transparent assessment of robustness.

TCGA-BRCA results (Table 5): In the BRCA dataset, the prognos-
tic signals captured by PathoLIC are highly robust. Models trained 
on original WSIs generalize eﬀectively to compressed slides. Speciﬁ-
cally, PathoLICCHIEF maintains a C-index of 0.696 (Compressed → Com-
pressed), which slightly outperforms the JPEG baseline (0.682). Fur-
thermore, PathoLICTITAN achieves a compelling eﬃciency-performance 
trade-oﬀ by retaining a comparable C-index (0.688 vs. 0.682) despite 
a higher compression ratio compared to JPEG. This indicates that our 
content-aware strategy preserves sparse, critical morphological features 
required for survival stratiﬁcation, with better eﬃciency than standard 
codecs.

TCGA-NSCLC results (Table 5): Similar trends are observed for the 
NSCLC dataset, where the advantage of PathoLIC becomes more pro-
nounced. PathoLICCHIEF achieves a C-index of 0.592 (Compressed →
Compressed), notably outperforming the JPEG baseline (0.577). This 
suggests that the blocking artifacts introduced by JPEG may disrupt sub-

tle prognostic features in lung tissue, whereas PathoLIC’s learned com-
pression eﬀectively retains the global contextual information necessary 
for prognostication.

5.2.3.  Patch-level histology classiﬁcation

We next focus on patch-level tasks, which demand ﬁdelity to ﬁne 
image details. All experiments are conducted with PathoLIC in the 
ﬁxed high-ﬁdelity setting (𝑞= 1). To assess practical utility, we bench-
mark PathoLIC against the standard JPEG codec, utilizing quality set-
tings calibrated to match the average bitrate of PathoLIC for a fair
comparison.

BACH dataset results (Table 6): On the four-class BACH dataset, 
models trained on original patches exhibit performance degradation 
when tested on compressed patches (original → compressed). However, 
PathoLIC consistently demonstrates classiﬁcation accuracy comparable 
to or exceeding the JPEG baseline across both ResNet-18 and ResNet-
50 architectures. Crucially, as detailed in Table 11, PathoLIC yields 
higher reconstruction quality (PSNR/MS-SSIM) than JPEG at equivalent 
or lower bitrates (e.g., 39.3 dB vs. 39.0 dB at 1.00 BPP). This indicates 
that our method preserves diagnostic features with greater ﬁdelity than 
the standard codec.

In-house dataset results (Table 7): For the simpler binary classiﬁca-
tion task on the in-house dataset, the impact of compression is less pro-
nounced. PathoLIC achieves robust performance, matching the ceiling 
accuracy of uncompressed data and performing on par with JPEG. No-
tably, for 1, 024 × 1, 024 patches, PathoLIC maintains this performance 
at a highly eﬃcient bitrate (0.59 BPP), underscoring its capability to 
handle large-context patches.

Medical Image Analysis 111 (2026) 104018

W. Li et al.

Fig. 8. ROI retrieval visualization on NCT-CRC-HE-100K. The ﬁgure displays query patches (left column) and their top-5 retrieved candidates. Rows 1 and 3 
present results using PathoLIC, while rows 2 and 4 show results using JPEG. The displayed tissue classes are: STR (Cancer-Associated Stroma), TUM (Colorectal 
Adenocarcinoma Epithelium), MUC (Mucus), and NORM (Normal Colon Mucosa). Class labels (CLS) and distance metrics (D) are provided for each patch. PathoLIC 
demonstrates stronger semantic consistency (e.g., in row 3), whereas JPEG exhibits semantic inconsistency (e.g., retrieving normal tissue for a tumor query in row 
4).

Fig. 9. Qualitative comparison of nuclei segmentation robustness against compression artifacts. The baseline prediction (on uncompressed input) is compared with 
results from PathoLIC, I2C, QmapCompression, and JPEG. PathoLIC yields segmentation masks closest to the Ground Truth (GT) with higher Dice scores (e.g., 0.927 
vs. 0.917 for JPEG).

5.2.4.  ROI retrieval

We evaluate patch-level ROI retrieval on the NCT-CRC-HE-100K 
dataset to verify feature consistency under compression. Given that re-
trieval relies on ﬁne-grained textural matching, we employ the ﬁxed 
high-ﬁdelity setting (𝑞= 1) for PathoLIC and benchmark it against JPEG.

NCT-CRC-HE-100K results (Tables 8 and 10): PathoLIC demon-
strates superior preservation of feature semantics essential for ROI re-
trieval. As shown in Table 8, PathoLIC achieves Top-1 and Top-5 re-

trieval accuracies nearly identical to the uncompressed upper bound, 
indicating negligible loss in discriminative feature power. Furthermore, 
regarding rate-distortion performance (Table 10), PathoLIC outperforms 
the JPEG baseline by achieving higher reconstruction ﬁdelity (PSNR) at 
equivalent or lower bitrates.

Visual analysis (Fig. 8): Qualitative results conﬁrm that PathoLIC 
retrieves semantically relevant patches (e.g., correctly matching speciﬁc 
tumor grades) with high visual ﬁdelity. In contrast, JPEG compression

Medical Image Analysis 111 (2026) 104018

10

W. Li et al.

Fig. 10. Component-wise ablation of QCM shows that removing any submodule (QCM𝑦 & QCM ̂𝑦, QCM𝑧, or QCM𝜇&𝜎) degrades reconstruction quality across all BPPs.

Fig. 11. Impact of removing the residual connection in the QCM. Disabling the residual path leads to a noticeable drop in both MS-SSIM and PSNR, conﬁrming its 
importance for stable and high-ﬁdelity feature modulation.

Fig. 12. Patch-level visual veriﬁcation of content scores. Representative patches 
with high vs. low attention scores derived from TITAN (top) and CHIEF (bot-
tom) are shown. High attention regions consistently correspond to diagnostic 
tumor nests and cellular areas. Low attention regions consistently correspond to 
adipose tissue, stroma, or background.

at matched bitrates introduces visible blocking artifacts that can obscure 
subtle histological textures.

5.2.5.  Cell-level nuclei segmentation

Finally, we assess the preservation of ﬁne-grained cellular structures 
via a nuclei segmentation task using the nnU-Net framework (Isensee 
et al., 2021). We also benchmark performance against JPEG under 
matched bitrate conditions.

PanNuke and MNS results (Table 9): The segmentation results re-
veal an exceptional level of robustness to our compression. The seg-
mentation results reveal an exceptional level of robustness to our com-
pression. PathoLIC achieves Dice and IoU scores nearly identical to the 
uncompressed baseline, outperforming JPEG on both PanNuke and the 
diverse Merged Nuclei Segmentation (MNS) benchmark. This advantage 
is particularly evident on MNS, where PathoLIC surpasses JPEG while 
using fewer bits.

Visual analysis (Fig. 9 and Table 12): The quantitative gain is 
supported by the rate-distortion analysis in Table 12, where PathoLIC 
demonstrates higher PSNR (e.g., +1.3 dB on MNS) compared to JPEG. 
Visually, segmentation masks produced from PathoLIC-compressed im-
ages are indistinguishable from original inputs (Fig. 9), whereas JPEG 
compression can introduce blocking artifacts that degrade nuclear 
boundary deﬁnition.

5.3.  Impact of foundation model

Our extensive validation conﬁrms that PathoLIC’s eﬃcacy is robust 
to the choice of guiding priors. As visualized in Figs. 4 and 12, although 
CHIEF and TITAN utilize distinct attention mechanisms, they yield se-
mantically consistent heatmaps, eﬀectively isolating tumor regions from

Medical Image Analysis 111 (2026) 104018

11

W. Li et al.

Table 7 
Patch-level classiﬁcation performance on the in-house dataset. PathoLIC is benchmarked against the 
JPEG baseline across varying patch resolutions (512 × 512, 1, 024 × 1, 024) and backbone architectures 
(ResNet-18, ResNet-50) to assess the robustness of learned representations under compression.

Model
(patch size)
 Compression Method
 Training Input
 Test Input
 ACC
 BACC
 AUROC

ResNet-18
(512 × 512)

–
 Original
 Original
 0.967
 0.957
 0.995

PathoLIC

Original
 Compressed
 0.875
 0.835
 0.942
 Compressed
 Original
 0.914
 0.896
 0.966
 Compressed
 Compressed
 0.947
 0.931
 0.987

JPEG

Original
 Compressed
 0.888
 0.874
 0.954
 Compressed
 Original
 0.908
 0.856
 0.972
 Compressed
 Compressed
 0.955
 0.949
 0.991

ResNet-50
(512 × 512)

–
 Original
 Original
 0.965
 0.958
 0.995

PathoLIC

Original
 Compressed
 0.901
 0.825
 0.963
 Compressed
 Original
 0.919
 0.912
 0.973
 Compressed
 Compressed
 0.941
 0.918
 0.988

JPEG

Original
 Compressed
 0.855
 0.852
 0.941
 Compressed
 Original
 0.922
 0.865
 0.981
 Compressed
 Compressed
 0.951
 0.948
 0.990

ResNet-18
(1,024 × 1,024)

–
 Original
 Original
 0.977
 0.967
 1.000

PathoLIC

Original
 Compressed
 0.977
 0.967
 1.000
 Compressed
 Original
 0.977
 0.967
 1.000
 Compressed
 Compressed
 0.977
 0.967
 1.000

JPEG

Original
 Compressed
 0.977
 0.967
 1.000
 Compressed
 Original
 1.000
 1.000
 1.000
 Compressed
 Compressed
 0.977
 0.967
 0.993

ResNet-50
(1,024 × 1,024)

–
 Original
 Original
 0.977
 0.967
 1.000

PathoLIC

Original
 Compressed
 1.000
 1.000
 1.000
 Compressed
 Original
 0.977
 0.967
 1.000
 Compressed
 Compressed
 0.977
 0.967
 1.000

JPEG

Original
 Compressed
 0.954
 0.933
 1.000
 Compressed
 Original
 0.907
 0.867
 1.000
 Compressed
 Compressed
 0.954
 0.933
 1.000

Table 8 
ROI retrieval accuracy on the NCT-CRC-HE-100K dataset using UNI (Chen 
et al., 2024). Top-𝐾 and Majority Vote (MVACC) accuracies are reported to 
assess feature space consistency under compression. PathoLIC is benchmarked 
against the JPEG baseline across diverse input conﬁgurations.

Compression
Method

Training
Input

Test
Input
 ACC@1  ACC@3  ACC@5  MVACC@5
 -
 Original
 Original
 0.957
 0.969
 0.972
 0.963

PathoLIC

Original
 Compressed  0.953
 0.971
 0.974
 0.967
 Compressed  Original
 0.963
 0.971
 0.975
 0.964
 Compressed  Compressed  0.968
 0.972
 0.977
 0.970

JPEG

Original
 Compressed  0.953
 0.973
 0.976
 0.965
 Compressed  Original
 0.963
 0.970
 0.973
 0.965
 Compressed  Compressed  0.967
 0.971
 0.974
 0.967

background stroma. Notably, the framework exhibits an adaptive capa-
bility: it automatically calibrates the average ﬁle size based on the spar-
sity of the attention map. The TITAN-driven version achieves higher 
compression without downstream accuracy loss, conﬁrming that our 
framework preserves diagnostic integrity across foundation models.

5.4.  Ablation study

To validate our architectural design, we further conduct ablation 
studies focusing on the Quality Control Module (QCM). The results in

Fig. 10 show that each QCM component enhances rate-distortion (R-
D) performance. The most pronounced degradation in both MS-SSIM 
and PSNR occurs when the QCMs modulating the primary latent space 
(𝑦 and ̂𝑦) are removed, underscoring the critical role of conditioning 
the main encoder and decoder. In addition, modulating the hyper-
prior (𝑧) and the entropy model parameters (𝜇, 𝜎) yields clear, comple-
mentary gains. We further examine the residual connection within the 
QCM’s aﬃne transform. As illustrated in Fig. 11, removing this residual 
path consistently degrades R-D performance, especially in low-bitrate 
regimes. This conﬁrms its importance for stable and eﬀective feature
modulation.

5.5.  Limitations

Despite the demonstrated robustness of PathoLIC, it currently lacks 
a fully integrated graphical user interface (GUI)-based software so-
lution that supports compression, decompression, and direct visual-
ization of WSIs. In addition, the current implementation does not 
support the direct integration or modiﬁcation of pathologist annota-
tions, such as tumor boundaries or tumor classiﬁcation labels, within 
the compressed ﬁles. Future work will focus on developing a com-
prehensive, user-friendly GUI platform that uniﬁes these capabili-
ties into a single, end-to-end system, enabling interactive visualiza-
tion, annotation management, and seamless deployment in clinical
workﬂows.

Medical Image Analysis 111 (2026) 104018

12

W. Li et al.

Table 9 
Cell-level nuclei segmentation performance on PanNuke and Merged Nuclei Segmentation (MNS) datasets using the nnU-Net 
framework. PathoLIC is benchmarked against the standard JPEG baseline across various training and test input conﬁgurations.

Dataset
 Compression Method
 Training Input
 Test Input
 Dice
 IoU
 Accuracy
 Precision
 Recall
 Speciﬁcity

PanNuke

–
 Original
 Original
 0.834
 0.732
 0.965
 0.854
 0.870
 0.970

PathoLIC

Original
 Compressed
 0.831
 0.727
 0.965
 0.856
 0.862
 0.971
 Compressed
 Original
 0.831
 0.728
 0.964
 0.852
 0.870
 0.970
 Compressed
 Compressed
 0.834
 0.731
 0.965
 0.853
 0.870
 0.970

JPEG

Original
 Compressed
 0.828
 0.723
 0.951
 0.854
 0.859
 0.970
 Compressed
 Original
 0.830
 0.726
 0.952
 0.856
 0.860
 0.971
 Compressed
 Compressed
 0.831
 0.725
 0.951
 0.852
 0.861
 0.969

MNS

–
 Original
 Original
 0.796
 0.669
 0.964
 0.820
 0.806
 0.964

PathoLIC

Original
 Compressed
 0.792
 0.663
 0.964
 0.816
 0.802
 0.963
 Compressed
 Original
 0.796
 0.669
 0.964
 0.818
 0.804
 0.963
 Compressed
 Compressed
 0.799
 0.673
 0.964
 0.816
 0.804
 0.963

JPEG

Original
 Compressed
 0.785
 0.653
 0.932
 0.807
 0.789
 0.961
 Compressed
 Original
 0.792
 0.664
 0.936
 0.819
 0.800
 0.964
 Compressed
 Compressed
 0.788
 0.658
 0.933
 0.815
 0.787
 0.963

Table 10 
Quantitative comparison of rate-distortion performance on the NCT-CRC-
HE-100K dataset. Reconstruction quality is evaluated using PSNR and MS-
SSIM at comparable BPP between PathoLIC and the JPEG baseline across 
diverse input conﬁgurations.

Split
 # Patches  Compression Method  BPP
 PSNR (dB)
 MS-SSIM (%)

Training
100,000
 PathoLIC
 1.50  32.5
 98.6
 JPEG
 1.52  31.6
 98.3

Test
7,180
 PathoLIC
 1.40  32.7
 98.5
 JPEG
 1.41  31.9
 98.0

Table 11 
Quantitative comparison of rate-distortion performance on the BACH and 
in-house datasets. Reconstruction quality is evaluated using PSNR and MS-
SSIM at comparable BPP between PathoLIC and the JPEG baseline across 
diﬀerent patch resolutions.

Setting
 # Patches

Compression
Method
 BPP

PSNR
(dB)

MS-SSIM
(%)

BACH
(512 × 512)
20,641
 PathoLIC
 1.00
 39.3
 99.1
 JPEG
 1.05
 39.0
 99.0

BACH
(1,024 × 1,024)
5,102
 PathoLIC
 1.25
 35.0
 98.9
 JPEG
 1.27
 34.8
 98.1

In-house
(512 × 512)
6,568
 PathoLIC
 0.99
 36.6
 99.0
 JPEG
 1.05
 36.1
 98.3

In-house
(1,024 × 1,024)
205
 PathoLIC
 0.59
 39.2
 99.3
 JPEG
 0.59
 39.0
 99.0

Table 12 
Quantitative comparison of rate-distortion performance on the Pan-
Nuke and MNS datasets. Reconstruction quality is evaluated using 
PSNR and MS-SSIM at comparable BPP for the images utilized in 
cell-level nuclei segmentation tasks.

Dataset
 # Patches

Compression
Method
 BPP

PSNR
(dB)

MS-SSIM
(%)

PanNuke
7,901
 PathoLIC
 1.10
 35.4
 98.9
 JPEG
 1.14
 34.7
 98.3

MNS
640
 PathoLIC
 1.26
 33.1
 91.9
 JPEG
 1.47
 31.8
 91.4

6.  Conclusion

We introduce PathoLIC, a novel content-aware variable-rate frame-
work tailored for whole slide image compression. As the ﬁrst deep 
learning-based method to perform content-aware compression on WSIs,

PathoLIC leverages content scores to modulate compression levels 
throughout the whole slide according to content scores. This approach 
reduces data redundancy eﬃciently while preserving ﬁne visual and 
structural details. Experimental results show that PathoLIC achieves 
over 8× compression beyond the standard Aperio SVS format, without 
noticeable loss of image details. Furthermore, it maintains strong per-
formance across various downstream tasks, including patch-level and 
WSI-level cancer subtyping as well as nuclei segmentation. Overall, 
PathoLIC provides an eﬃcient solution for managing large-scale pathol-
ogy archives, and also facilitates broader integration of AI in digital 
pathology workﬂows.

CRediT authorship contribution statement

Weiqi Li: Writing – review & editing, Writing – original draft, 
Methodology, Investigation, Conceptualization; Yonghao Li: Writing – 
review & editing, Methodology; Haoyuan Chen: Writing – review & 
editing; Long Yang: Writing – review & editing; Lin Wu: Data curation;
Zhenhui Li: Data curation; Jing Ke: Writing – review & editing, Super-
vision, Conceptualization; Dinggang Shen: Writing – review & editing, 
Supervision, Project administration, Funding acquisition, Conceptual-
ization.

Declaration of competing interest

The authors declare that they have no known competing ﬁnancial 
interests or personal relationships that could have appeared to inﬂuence 
the work reported in this paper.

Acknowledgement

This work is supported in part by National Natural Science Foun-
dation of China (grant numbers 82441023, U23A20295, 62131015, 
82394432), the China Ministry of Science and Technology (S20240085, 
STI2030-Major Projects-2022ZD0209000, STI2030-Major Projects-
2022ZD0213100), Shanghai Municipal Central Guided Local Science 
and Technology Development Fund (No. YDZX20233100001001), and 
HPC Platform of ShanghaiTech University.

References

Angell, H.K., Gray, N., Womack, C., Pritchard, D.I., Wilkinson, R.W., Cumberbatch, M., 
2013. Digital pattern recognition-based image analysis quantiﬁes immune inﬁltrates 
in distinct tissue regions of colorectal cancer and identiﬁes a metastatic phenotype. Br. 
J. Cancer 109 (6), 1618–1624.
Aresta, G., Araújo, T., Kwok, S., Chennamsetty, S., Safwan, M., Alex, V., Marami, B., 
Prastawa, M., Chan, M., Donovan, M., et al., 2019. Bach: grand challenge on breast 
cancer histology images. In: Medical Image Analysis. Vol. 56. Elsevier, pp. 122–139.

Medical Image Analysis 111 (2026) 104018

13

W. Li et al.

Association, 
D.P., 
2019. 
Digital 
pathology 
association 
whitepaper. 
https:
//digitalpathologyassociation.org/white-papers.
Ballé, J., Laparra, V., Simoncelli, E.P., 2017. End-to-end optimized image compression. 
In: 5th International Conference on Learning Representations, ICLR 2017.
Ballé, J., Minnen, D., Singh, S., Hwang, S.J., Johnston, N., 2018. Variational image com-
pression with a scale hyperprior. In: International Conference on Learning Represen-
tations (ICLR).
Cai, S., Chen, L., Zhang, Z., Zhao, X., Zhou, J., Peng, Y., Yan, L., Zhong, S., Zou, X., 
2024. I2c: Invertible continuous codec for high-ﬁdelity variable-rate image compres-
sion. IEEE Trans. Pattern Anal. Mach. Intell. 46 (6), 4262–4279.
Campanella, F., Hanna, M.G., Geneslaw, L., Miraﬂor, A., Werneck Krauss Silva, V., Busam, 
K.J., Brogi, E., Reuter, V.E., Klimstra, D.S., Fuchs, T.J., 2019. Clinical-grade computa-
tional pathology using weakly supervised deep learning on whole slide images. Nat. 
Med. 25 (8), 1301–1309.
Chen, R.J., Ding, T., Lu, M.Y., Williamson, D. F.K., Jaume, G., Song, A.H., Chen, B., Zhang, 
A., Shao, D., Shaban, M., et al., 2024. Towards a general-purpose foundation model 
for computational pathology. Nat. Med. 30 (3), 850–862.
Cheng, Z., Sun, H., Takeuchi, M., Katto, J., 2020. Learned image compression with dis-
cretized gaussian mixture likelihoods and attention modules. In: Proceedings of the 
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7939–7948.
Ding, T., Wagner, S.J., Song, A.H., Chen, R.J., Lu, M.Y., Zhang, A., Vaidya, A.J., Jaume, 
G., Shaban, M., Kim, A., et al., 2025. A multimodal whole-slide foundation model for 
pathology. Nat. Med. 1–13.
Farahani, N., Parwani, A.V., Pantanowitz, L., 2015. Whole slide imaging in pathology: 
advantages, limitations, and emerging perspectives. Pathol. Lab. Med. Int., 23–33.
Gamper, J., Alemi Koohbanani, N., Benet, K., Khuram, A., Rajpoot, N., 2019. Pannuke: an 
open pan-cancer histology dataset for nuclei instance segmentation and classiﬁcation. 
In: European Congress on Digital Pathology. Springer, pp. 11–19.
Graham, S., Jahanifar, M., Azam, A., Nimir, M., Tsang, Y.-W., Dodd, K., Hero, E., Sahota, 
H., Tank, A., Benes, K., et al., 2021. Lizard: a large-scale dataset for colonic nuclear in-
stance segmentation and classiﬁcation. In: Proceedings of the IEEE/CVF International 
Conference on Computer Vision, pp. 684–693.
Graham, S., Vu, Q.D., Raza, S. E.A., Azam, A., Tsang, Y.T., Kwak, J.T., Rajpoot, N.M., 
2019. Hover-net: simultaneous segmentation and classiﬁcation of nuclei in multi-
tissue histology images. In: International Conference on Medical Image Computing 
and Computer-Assisted Intervention (MICCAI). Springer, pp. 632–640.
Hou, L., Samaras, D., Kurc, T.M., Gao, Y., Davis, J.E., Saltz, J.H., 2016. Patch-based con-
volutional neural network for whole slide tissue image classiﬁcation. In: Proceedings 
of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2424–2433.
Isensee, F., Jaeger, P.F., Kohl, S.A.A., Petersen, J., Maier-Hein, K.H., 2021. nnU-Net: a 
self-conﬁguring method for deep learning-based biomedical image segmentation. Nat. 
Methods 18 (2), 203–211.
Kather, J.N., Halama, N., Marx, A., 2018. 100,000 histological images of human colorectal 
cancer and healthy tissue. (No Title).
Kingma, D.P., Ba, J., 2014. Adam: A method for stochastic optimization. arXiv preprint 
arXiv:1412.6980.
Komura, D., Ishikawa, S., 2018. Machine learning methods for histopathological image 
analysis. Comput. Struct. Biotechnol. J. 16, 34–42.

Kumar, N., Verma, R., Sharma, S., Bhargava, S., Vahadane, A., Sethi, A., 2017. A dataset 
and a technique for generalized nuclear segmentation for computational pathology. 
IEEE Trans. Med. Imaging 36 (7), 1550–1560.
Litjens, G., Kooi, T., Bejnordi, B.E., Setio, A. A.A., Ciompi, F., Ghafoorian, M., Van 
Der Laak, J.A., Van Ginneken, B., Sánchez, C.I., 2017. A survey on deep learning in 
medical image analysis. Med. Image Anal. 42, 60–88.
Liu, J., Sun, H., Katto, J., 2023. Learned image compression with mixed transformer-CNN 
architectures. In: Proceedings of the IEEE/CVF Conference on Computer Vision and 
Pattern Recognition, pp. 1–10.
Lu, M.Y., Williamson, D. F.K., Chen, T.Y., Chen, R.J., Barbieri, M., Mahmood, F., 2021. 
Data-eﬃcient and weakly supervised computational pathology on whole-slide images. 
Nat. Biomed. Eng. 5 (6), 555–570.
Madabhushi, A., Lee, G., 2016. Image analysis and machine learning in digital pathology: 
challenges and opportunities. Med. Image Anal. 33, 170–175.
Minnen, D., Ballé, J., Toderici, G., 2018. Joint autoregressive and hierarchical priors for 
learned image compression. In: Advances in Neural Information Processing Systems 
(NeurIPS), pp. 10771–10780.
Song, M., Choi, J., Han, B., 2021. Variable-rate deep image compression through spatially-
adaptive feature transform. In: Proceedings of the IEEE/CVF International Conference 
on Computer Vision, pp. 2380–2389.
Taubman, D.S., Marcellin, M.W., Rabbani, M., 2002. Jpeg2000: image compression fun-
damentals, standards and practice. J. Electron. Imaging 11 (2), 286–287.
Toderici, G., Vincent, D., Johnston, N., Jin Hwang, S., Minnen, D., Shor, J., Covell, M., 
2017. Full resolution image compression with recurrent neural networks. In: Pro-
ceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 
5306–5314.
Van der Laak, J. A.W.M., Litjens, G., Ciompi, F., 2021. Deep learning in histopathology: 
the path to the clinic. Nat. Rev. Gastroenterol. Hepatol. 18 (1), 11–24.
Verma, R., Kumar, N., Patil, A., Kurian, N.C., Rane, S., Graham, S., Vu, Q.D., Zwager, M., 
Raza, S. E.A., Rajpoot, N., et al., 2021. MonuSAC2020: a multi-organ nuclei segmen-
tation and classiﬁcation challenge. IEEE Trans. Med. Imaging 40 (12), 3413–3423.
Wallace, G.K., 2002. The JPEG still picture compression standard. IEEE Trans. Consum. 
Electron. 38 (1), xviii–xxxiv.
Wang, X., Zhao, J., Marostica, E., Yuan, W., Jin, J., Zhang, J., Li, R., Tang, H., Wang, K., 
Li, Y., et al., 2024. A pathology foundation model for cancer diagnosis and prognosis 
prediction. Nature 634 (8035), 970–978.
Weinstein, J.N., Collisson, E.A., Mills, G.B., Shaw, K.R., Ozenberger, B.A., Ellrott, K., 
Shmulevich, I., Sander, C., Stuart, J.M., 2013. The cancer genome atlas pan-cancer 
analysis project. Nat. Genet. 45 (10), 1113–1120.
Xu, H., Usuyama, N., Bagga, J., Zhang, S., Rao, R., Naumann, T., Wong, C., Gero, Z., 
González, J., Gu, Y., Xu, Y., Wei, M., Wang, W., Ma, S., Wei, F., Yang, J., Li, C., Gao, 
J., Rosemon, J., Bower, T., Lee, S., Weerasinghe, R., Wright, B.J., Robicsek, A., Piening, 
B., Bifulco, C., Wang, S., Poon, H., 2024. A whole-slide foundation model for digital 
pathology from real-world data. Nature 630 (8015), 181–188.
Zhu, Y., Yang, Y., Cohen, T., 2022. Transformer-based transform coding. In: International 
Conference on Learning Representations.

Medical Image Analysis 111 (2026) 104018

14