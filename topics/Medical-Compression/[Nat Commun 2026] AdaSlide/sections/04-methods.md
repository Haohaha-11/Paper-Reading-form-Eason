[← 返回 README](../README.md)

# 04 - Methods

## 预览

Methods 给出可复现的数据流：PanCancer 构建、FIE 训练、CDA 单步 REINFORCE reward、13 个下游任务配置、VTT、WSI 重建和评价指标。

# Methods

# PanCancer Datasets

The main objective of the FIE is to achieve robustness across multiple domains, including different organs, scanner variations, and scanned image quality. To address these challenges, we collected data from TCGA, which encompasses 31 projects. Thirty diagnostic slides were randomly sampled from each project, which yielded a training set of 930 WSIs. Although the entire TCGA collection contained over 20,000 WSIs, we sampled 5% to reduce computational costs and facilitate efficient training. Each WSI was tessellated into multiple patch images. We considered two key objectives to generate diverse patches:

1. Multi-magnification representation: Patches were extracted from both 40x (0.25 $\mu \mathrm { m } / \mathrm { p i x e l }$ ) and 20x (0.5 $\mu$ m/pixel) magnifications.   
2. Spatial diversity: To promote adequate coverage of different WSI regions, we sampled $1 0 \%$ of the patches at 40x and $4 0 \%$ at 20x. The WSIs exhibited strong spatial correlations, implying that nearby patches often shared similar patterns. We employed a sampling-based patch selection instead of a sliding-window-based selection to encourage diverse patch selection.

To exclude non-tissue areas, we generated binary tissue masks using the Otsu thresholding with a validity threshold of 70%. Each patch was 512x512 pixels in size. We generated 1.8 million patches, ensuring equal representation at 40x and 20x magnifications. The PanCancer dataset encompasses diverse organs and cancer types. We divided PanCancer into training (94.5%), validation ( $5 \%$ ), and test (0.5%) sets.

> 💡 **PanCancer 数据设计**: FIE 的目标是跨器官/倍率复原，所以训练集不是为某个下游标签构建，而是从 TCGA 31 projects 中采样。20x/40x 和空间采样降低了近邻 patch 重复，帮助 FIE 学更广的染色、组织和扫描分布。

# Datasets: Foundation Image Enhancer (FIE)

To train and validating the FIE, we started with high-resolution images (512x512 pixels) from the PanCancer dataset and artificially down-scaled them to 128x128 pixels using bilinear interpolation. This simplified setting focused solely on the challenge of upscaling without the additional noise complexities encountered in real-world lowresolution images. Patch images were generated at both 40x and 20x magnifications to train and validate robustness across varying scales. The same preprocessing pipeline was applied to generate an external validation dataset sourced from CPTAC.

# Datasets: Compress Decision Agent (CDA)

A validation split of the PanCancer dataset was used to train the CDA. For the pseudo ground truth, the HoverNet [? ] outputs were acquired from both the original image and the compressed-and-restored image.

> 💡 **CDA teacher 信号**: CDA 没有人类标注“这个 patch 应该压缩/保留”。它的 pseudo ground truth 是 HoverNet 在原图和复原图上的细胞分割输出差异，用来衡量压缩是否破坏细胞信息。

# Downstream Datasets

To validate the performance of the trained AdaSlide model, we leveraged several publicly available downstream datasets commonly used for slide-level and patch-level tasks. These datasets were specifically selected to evaluate the performance of AdaSlide in multi-magnification, multi-organ environments. A summary of each dataset is provided in Table 1.

The NCT-CRC dataset (National Center for Tumor Diseases Colorectal Cancer Type Classification) dataset consists of 100,000 training and 7,180 test images extracted from H&E-stained WSIs of human CRC and normal tissue [? ]. The dataset contains nine classes: adipose tissue, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma, and NCT-CRC epithelium. All images were color-normalized using the Macenko method [? ]. We split the original training set into a training and validation sets at a 7:3 ratio. The test dataset was then used according to a publicly available split set.

The MHIST (Minimalist Histopathology Image Analysis Dataset) dataset contains 2,175 training and 977 test patch images of H&E-stained colorectal polyps (extracted from 328 FFPE WSIs) from Dartmouth-Hitchcock [? ]. This dataset focused on the binary classification of benign hyperplastic polyps (HPs) vs. precancerous sessile serrated adenomas (SSAs). We split the original training set into a training and validation set at a 7:3 ratio. The test dataset was then used according to the publicly available split set.

The LI (Lymphatic Invasion Classification) dataset has 974 images extracted from H&E-stained WSIs of human stomach cancers [? ]. It specifically targets the binary classification of LI regions versus normal regions. Unlike other datasets, the LI dataset concentrates on small regions containing lymphocytes and the presence of tumor invasions within those lymphocyte areas. Although the original LI dataset includes weakly-labeled negative cases, we excluded them from this study. The LI dataset was divided into training, validation, and test datasets $\mathrm { a t  a }$ ratio of 5:3:2.

The SICAPv2 (Sistema de Interpretaci´on de Im´agenes Histopatol´ogicas para la Detecci´on de C´ancer de Pr´ostata) dataset comprises 155 biopsies from 95 patients, with labels based on primary Gleason grades ranging from non-cancerous (NC) to Grade 5 (GG5). The primary grade was used as the target label for classification. The public test split was utilized for validation, while the remaining data were divided into training and validation sets with a ratio of 0.8:0.2 based on the public training split set configuration.

The WSSS4LUAD (Weakly-supervised Semantic Segmentation Techniques for Histopathology Images of Lung Adenocarcinoma) dataset comprises LUAD image tiles ranging from 200 to 500 pixels in size, each annotated with labels such as tumor, tumorassociated stroma, or normal. For our analysis, we adopted the approach outlined in the UNI [? ], selecting only samples with a single ground-truth label. This resulted in a subset of 4,693 images from the official training split. In this study, the dataset was further divided into training, validation, and test sets in a ratio of 0.6:0.2:0.2.

The SNOW (Synthetic Dataset for Human Breast Cancer Nuclei Segmentation) dataset contains 20,000 images of breast cancer patch images [? ]. The patch images and nuclei annotation masks were generated from deep neural networks. We selected the first 5,000 images and their corresponding mask pairs to simplify the experiments. These 5,000 image-mask pairs were then divided into training, validation, and test datasets at a ratio of 9:0.5:0.5.

The NuInsSeg (Nuclei Instance Segmentation) dataset is a fully manually annotated dataset for nuclei segmentation in H&E images. It includes 665 image patches with over 30,000 manually segmented nuclei from 31 human and mouse organs. For this study, only the human H&E images (472 images) were used. The dataset was split into training, validation, and testing sets with a ratio of 0.6:0.2:0.2, respectively. Detailed instructions for generating the related segmentation masks are available in the public repository associated with the dataset.

The PanNuke (Pan-cancer Histology Dataset for Nuclei Instance Segmentation and Classification) dataset contains nuclei-level annotations derived from histopathology images spanning 19 tissue types. The dataset includes approximately 200,000 nuclei categorized into five clinically significant classes: neoplastic, inflammatory, connective tissue, necrotic/dead, and epithelial. In this study, we converted the dataset to binary segmentation masks for simplicity. The PanNuke dataset fold 1 was split into training, validation, and test datasets at a ratio of 0.6:0.2:0.2.

The TCGA-RCC dataset comprises 884 WSIs from TCGA’s kidney chromophobe (TCGA-KICH), kidney renal clear cell carcinoma (TCGA-KIRC), and kidney renal papillary cell carcinoma (TCGA-KIRP) projects. TCGA-KICH contains 111 images from 99 patients, TCGA-KIRC contains 489 images from 483 patients, and TCGA-KIRP contains 284 images from 264 patients. Patches were generated at 20x magnification with a size of 512x512 pixels. The average number of patches per slide was approximately 802.66, with a standard deviation of 435.81. The maximum and minimum number of patch images per slide were 4001 and 12, respectively.

The TCGA-NSCLC dataset contains 993 WSIs from TCGA, comprising 507 lung squamous cell carcinoma (TCGA-LUAD) and 486 lung adenocarcinoma (TCGA-LUSC) cases across 896 patients. Patches were generated at 20x magnification and are 512x512 pixels in size. The average number of patches per slide was approximately 694.63, with a standard deviation of 490.80. The maximum and minimum patch images per slide were 2933 and 10, respectively.

The TCGA-BRCA dataset includes various cancer types; however, for this analysis, we focused on invasive ductal carcinoma (IDC) and invasive lobular carcinoma (ILC) subtypes. Total 1,007 cases across 948 patients were employed. Patches were generated at 20x magnification and are 512x512 pixels in size. The average number of patches per slide was approximately 712.82, with a standard deviation of 452.17. The maximum and minimum patch images per slide were 4,252 and 12, respectively.

The Camelyon16 dataset (Cancer Metastases in Lymph Nodes Challenge 2016) contains 268 WSIs, including 111 slides from patients with lymph node metastases and 160 slides from normal lymph nodes. The dataset focuses on the binary classification of metastatic versus non-metastatic lymph nodes. Patches were extracted at 40x magnification with a size of 512x512 pixels. On average, each slide generated approximately 6021.86 patches, with a standard deviation of 4130.37. The maximum and minimum patch images per slide were 26,926 and 127, respectively.

The Children’s Brain Tumor Network (CBTN) [? ] contains 3,607 WSIs across 41 diagnostic categories derived from pediatric samples. For this study, we narrowed the classification task to only include Medulloblastoma and Ependymoma diagnoses. 150 WSIs were selected from each class and included 212 patients. Patches were generated at 20x magnification and are 512x512 pixels in size without overlaps. The average number of patches per slide was about 173, with a standard deviation of 127.12. The minimum and maximum number of patch images per slide were 4 and 721, respectively.

> 💡 **下游数据覆盖**: 数据集清单刻意覆盖器官、倍率、任务粒度和域偏移。Camelyon16 patch 数远多于 TCGA slide-level 数据，CBTN 是 pediatric domain，NuInsSeg 是外部 nuclei segmentation，这些都是检验 OOD 风险的关键点。

# Model Developments: Foundation Image Enhancer (FIE)

The FIE was trained using three backbone architectures: VAE [? ], VQVAE [? ], ESRGAN [? ], SwinIR [? ], and LDM [? ]. The FIE required two modules: an encoder $f ( \cdot )$ and a decoder $g ( \cdot )$ . The encoder maps the original images $x$ to the latent vectors, and the decoder maps the latent vectors to original images. Therefore, the FIE works as follows:

$$
{ \hat { x } } = g ( f ( x ) ) ,
$$

where the FIE was trained to minimize the difference between $x$ and $\hat { x }$

> 💡 **公式批读 - FIE**: 这个公式只是抽象接口。对 VAE/VQVAE，f 是 learned encoder；对 ESRGAN，f 实际是线性下采样，g 是超分复原网络。AdaSlide 只要求 FIE 能把压缩表示恢复到原尺寸，以便 CDA reward 计算和 WSI 重建。

For the VAE [? ], the original study used 64x64 pixel images. We trained it with images of size 224x224 and 512x512 pixels. The architecture consisted of hidden dimensions with various sizes depending on the inputs: 224x224 (32, 64, 128, 256, 256, 512, 512) and 512x512 (32, 64, 128, 256, 512, 512, 512, 1024, 1024, 1024). The batch size was set to 128, and training used the Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ . The training split of the PanCancer dataset were used and trained for five epochs. The checkpoints were selected based on the validation loss.

The VQVAE [? ] was proposed to replace the latent dimensions represented by a Gaussian distribution in VAEs. Instead, VQVAE employs quantized vector embeddings to represent the latent dimensions. In addition, VQVAE, particularly VQVAE-2, attempts to enhance the images in a two-stage manner. Similar to VAE, VQVAE exhibits low flexibility with respect to the input image shape. Therefore, we trained the VQVAE model twice, once for 224x224 pixels and again for 512x512 pixels. The architecture and hyperparameters are shared. The model hyperparameters and training parameters followed those of a previously proposed VQVAE model for histopathological image compression [? ]. The batch size was set to 128 and the learning rate to $1 \times 1 0 ^ { - 4 }$ , using the Adam optimizer with five epochs. Minimum validation loss was set as the checkpoint selection criterion.

Unlike the VQVAE models, the ESRGAN [? ] does not require trained encoders. The ESRGAN’s encoder $f ( \cdot )$ is a simple linearly downscaling method. Unlike other methods, ESRGAN attempts to minimize perceputal differences. As numerical metrics such as SSIM and PSNR diverged from human expert selection, the optimal checkpoint was determined through visual inspection in collaboration with a pathologist. To train the ESRGAN model, a batch size of 256 was used, and the model was initialized with a pretrained ESRGAN model. The Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ was used, and training lasted for a total of 50,000 steps, with the first 2,500 as a warm-up phase. All other hyperparameters were maintained at their default settings, as defined in the original ESRGAN model [? ].

The SwinIR [? ] is a Transformer-based image restoration model that utilizes the Swin Transformer architecture [? ]. Compared to CNN-based models such as ESR-GAN, SwinIR provides enhanced capability for modeling long-range dependencies and contextual information, which is particularly useful in reconstructing complex textures in histopathology images. In our implementation, we trained the SwinIR model to handle input sizes of 512x512 pixels. The architecture was initialized with a pretrained SwinIR model and fine-tuned using the PanCancer training split. A batch size of 12 and the Adam optimizer with a learning rate of $2 \times 1 0 ^ { - 4 }$ were used. The training was performed for 50,000 steps, with the multiple step scheduler with 5 milestones; 25000, 40000, 45000, 47500, 50000. The checkpoint was selected based on the lowest validation loss. All other hyperparameters followed the original SwinIR implementation [? ]. For training SwinIR, we employed KAIR library (v1.1, https://github.com/cszn/KAIR).

The LDM [? ] is a generative diffusion model that performs denoising in a lowerdimensional latent space, enabling both efficient training and high-quality image reconstruction. Unlike other models in this study, LDM consists of three main components: a first-stage VQVAE to map images to latent space, a UNet-based denoising network that operates in latent space, and a conditioning mechanism to guide reconstruction. In our implementation, we adopted a LDM architecture customized for histopathology image reconstruction. The pretrained first-stage VQVAE was employed. The denoising network was trained with a noise schedule following the cosine beta schedule, with 1,000 diffusion steps. We used a batch size of 8 and the Adam optimizer with a learning rate of $2 \times 1 0 ^ { - 6 }$ . The training was conducted for 7 epochs. The best checkpoint was selected based on the lowest validation loss computed on reconstructed images. Training the LDM required approximately 10 GPU days.

The VAE and VQVAE depended on a fixed input size; therefore, these models were trained twice to handle the input sizes of 224x224 and 512x512 pixels, respectively. The ESRGAN, SwinIR, LDM models were trained once to handle an input size of 128x128. Python (v3.8), PyTorch (v2.0.1), and a single RTX A6000 GPU were used to implement AdaSlide.

> 💡 **FIE 复现要点**: ESRGAN 的 checkpoint 选择依赖 pathologist visual inspection，这比纯 validation loss 更贴近 H&E 视觉质量，但也引入主观性。LDM 约 10 GPU days，说明 diffusion 质量不一定能抵消 WSI 级推理成本。

# Model Developments: Compression Decision Agent (CDA)

The CDA takes a single image patch as input and makes a binary compression decision $a \in \{ 0 , 1 \}$ , where $a = 1$ denotes Compress and $a = 0$ denotes Keep. The agent employs pretrained image encoders (ResNet-18, ResNet-50, ViT) and is optimized by policybased learning using the REINFORCE algorithm [? ] (see Supplementary Note 3). Unlike conventional multi-step REINFORCE, we adopt a single-step setting with one patch per decision, eliminating the computational overhead of episode sampling.

We balance compression efficiency and information preservation. Let $S _ { \mathrm { D i c e } } \in [ 0 , 1 ]$ be the Dice similarity between the segmentation mask derived from the ESRGANreconstructed (compressed) image and the reference mask obtained from the original image. The information penalty is

$$
S _ { \mathrm { i n f o } } = 1 - S _ { \mathrm { D i c e } } .
$$

To avoid slow convergence, we introduce a quality threshold $\tau \in ( 0 , 1 )$ on $S _ { \mathrm { D i c e } }$ . The final reward is

$$
S _ { \mathrm { R e w a r d } } = \left\{ \begin{array} { l l } { - 1 , } & { \mathrm { i f ~ } a = 1 \mathrm { ~ a n d ~ } S _ { \mathrm { D i c e } } < \tau , } \\ { 0 , } & { \mathrm { i f ~ } a = 0 , } \\ { 1 - \lambda S _ { \mathrm { i n f o } } , } & { \mathrm { i f ~ } a = 1 \mathrm { ~ a n d ~ } S _ { \mathrm { D i c e } } \geq \tau , } \end{array} \right.
$$

where $\lambda > 0$ controls the strength of the information-preservation term.

We tune $\lambda \in \{ 0 . 1 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 . 0 \}$ and the learning rate $\in \{ 1 \times 1 0 ^ { - 4 } , 5$ × $1 0 ^ { - 5 }$ , $1 \times 1 0 ^ { - 5 }$ , $5 \times 1 0 ^ { - 6 }$ , $1 \times 1 0 ^ { - 6 } \}$ via grid search. To avoid over/under-compression, models with CR outside $2 0 { - } 9 0 \%$ are discarded.

> 💡 **CDA reward 批读**: Keep 的 reward 是 0，相当于安全但没有压缩收益；Compress 如果 mask Dice 低于 tau 直接 -1；如果 Dice 过关，reward 是 1 - lambda * information loss。lambda 越大，同样的信息损失越不可接受。丢弃 CR 超出 20-90% 的模型，是为了避免 agent 学成全压或全不压。

# Downstream Tasks

To compare performance, we evaluated 13 downstream tasks for each condition: supervised training using the original image, using only the enhanced image processed by the foundation image enhancer (when $\lambda = 0$ ), and applying adaptive compression with the AdaSlide method.

Patch-level downstream tasks are divided into classification and segmentation. The classification tasks include five datasets (NCT-CRC, MHIST, LI, SICAPv2, WSSS4LUAD), while the segmentation tasks include three datasets (SNOW, NuInsSeg, PanNuke). All experiments were repeated using five random seeds ({42, 43, 44, 45, 46}). For consistency, the same split sets were used across experiments, and public split sets were utilized whenever available. The checkpoints were selected based on the lowest validation loss. The Adam optimizer, paired with a cosine annealing learning rate scheduler, was applied throughout.

For patch-level classification tasks, a class-balanced sampler was employed to address class imbalance. Only resize and normalization augmentations were applied. In contrast, patch-level segmentation tasks, which generally involve smaller datasets, utilized various image augmentations, including flipping, random cropping, and color jittering. Detailed configurations for each downstream task are summarized in Table 3.

The slide-level classification datasets included TCGA-RCC, TCGA-NSCLC, TCGA-BRCA, Camelyon16, and CBTN. We utilized the CLAM pipeline [? ] for preprocessing and training the MIL models, with the CLAM-SB backbone model. Feature extraction for each patch image was performed using the ResNet-50 model. The batch size was set to 1, with training conducted over 50 epochs at a learning rate of $1 \times 1 0 ^ { - 4 }$ . Early stopping was applied to prevent overfitting. The Adam optimizer was used without a learning rate scheduler.

![Table 3](../images/8b7fe9b261125c0fb03e01478bce2a3e06afbc3a217d8d0c4052bdfa72db899e.jpg)

*Table 3: Training configurations for patch-level tasks.*

> 💡 **实验控制**: 作者固定下游训练流程，只替换输入图像处理条件：原图、FIE-only、AdaSlide。这样性能差异主要来自压缩/复原造成的信息变化，而不是下游模型或训练技巧变化。

# Visual Turing Test (VTT)

Five expert pathologists participated in VTT. In each trial, participants were presented with two images: one original and the other restored using ESRGAN. They were instructed to identify which image appeared to be enhanced. A total of 30 questions were presented, and the chance level was set at $5 0 \%$ . The resulting accuracy scores were compared against the chance level using the Wilcoxon signed-rank test.

> 💡 **VTT 设计**: VTT 是 pairwise forced choice，不是让专家打质量分。统计检验对象是“能否显著高于 50% 识别复原图”；未显著高于 chance 支持 ESRGAN 复原图的感知真实性。

# Application Example of AdaSlide

Box 1 presents the AdaSlide pipeline. In this application, patch images were saved as JPEG for kept images and PNG for compressed images. The threshold $\tau$ was set to the optimal $\lambda$ determined by downstream performance.

# Box 1 — AdaSlide adaptive compression algorithm

Input: Patch images $P$ Output: Reconstructed whole-slide image $R$

1. Initialize empty sets for compressed $. S _ { P }$ ), non-compressed $\{ S _ { N } \}$ ), and enhanced ( $S _ { E }$ ) patches.

# 2. Encoding:

2.1. For each patch $p \in { \cal P }$ :   
2.1.a) Obtain compression decision $\hat { h } = \mathrm { C D A } ( p )$ .   
2.1.b) If $\hat { h } \ge \tau$ , set $p _ { c } = \mathrm { F I E } _ { \mathrm { e n c o d e r } } ( p )$ and add to $S _ { P }$ ; otherwise add $p$ to $S _ { N }$ .

# 3. Decoding:

3.1. For each $p _ { c } \in S _ { P }$ , compute $p _ { e } = \mathrm { F I E _ { d e c o d e r } } \left( p _ { c } \right)$ and add to $S _ { E }$ .

4. Reconstruct $R$ from $S _ { N } \cup S _ { E }$ : $R  \mathrm { R e c o n s t r u c t } ( S _ { N } \cup S _ { E } )$

Using CLAM toolbox [? ], we preprocessed the WSIs by selecting foreground tissue areas using binarized images and applying median blurring to smooth the edges. For the Camelyon16 sample, patches were generated at level 0 (0.25 µm/pixel), with a size of 512x512 pixels. For TCGA-BRCA, and TCGA-KIRC samples, the patch size was initially 1024x1024 at level 0, which was then resized to 512x512 pixels. The numbers of patches for Camelyon16, TCGA-BRCA, and TCGA-KIRC were 24,063, 9,278, and 3,720, respectively. Using the CDA, we made a patch-wise compression decision. Based on the decision made by the CDA, the patch images were either compressed to 128x128 pixels (compress) or left uncompressed (keep). In the decoding step, the compressed images are restored to their original pixel size and enhanced to the FIE. After the encoding-decoding step, the patch images were restored to their original WSI file format. The pyvips framework [? ] was used to generate a TIFF-format WSI file and JPEG compression with 75% quality. During reconstruction, the background images were not considered; therefore, the final size of the WSIs was reduced.

> 💡 **Box 1 数据流**: Box 1 把 AdaSlide 从论文模型变成文件系统 pipeline：S_P 保存压缩表示，S_N 保存原图 patch，S_E 保存 FIE 复原 patch，最终 S_N union S_E 重建 WSI。工程细节包括 pyvips、TIFF、JPEG quality 75% 和忽略背景区域。

# Evaluation Metrics: Foundation Image Enhancer (FIE)

The PSNR measures the extrinsic image quality based on the pixel values between the original and enhanced images. The PSNR is computed as follows:

$$
P S N R = 1 0 \log _ { 1 0 } \left( \frac { \mathrm { M A X } _ { I } ^ { 2 } } { \mathrm { M S E } } \right) ,
$$

where MAX $\boldsymbol { \mathit { 1 } }$ is the maximum pixel value of the image, and MSE is the mean squared error between the original and the enhanced images. Similarly, the SSIM measures the differences in structures and pixel values between the original and enhanced images.

The SSIM is computed as follows:

$$
S S I M = \frac { ( 2 \mu _ { x } \mu _ { y } + \epsilon ) ( 2 \sigma _ { x y } + \epsilon ) } { ( \mu _ { x } ^ { 2 } + \mu _ { y } ^ { 2 } + \epsilon ) ( \sigma _ { x } ^ { 2 } + \sigma _ { y } ^ { 2 } + \epsilon ) } ,
$$

where $\mu _ { x }$ and $\mu _ { y }$ are the average pixel values of the original and enhanced images, respectively. $\sigma _ { x }$ and $o _ { y }$ are the standard deviations of the original and enhanced images, respectively. $\sigma _ { x y }$ is the covariance between the original and enhanced images. $\epsilon$ is a small constant that avoids a zero denominator.

The LPIPS measures the dissimilarity between the original and enhanced images in feature space using pretrained image encoders, such as VGG, ResNet-50 models. The LPIPS is computed as follows:

$$
L P I P S = L _ { 1 } ( \phi ( x ) , \phi ( y ) ) ,
$$

where $L _ { 1 }$ represents the mean absolute difference, and $\phi ( x )$ and $\phi ( y )$ are the feature vectors extracted from the original image (x) and the enhanced image (y) by a pretrained deep learning model designed to capture perceptual similarity. In this study, we used the ImageNet pretrained VGG 16 model.

> 💡 **FIE 指标批读**: PSNR/SSIM 偏像素和结构，LPIPS 偏感知特征。本文选择 ESRGAN 时并没有只按这些指标排序，因为病理图像的视觉可读性和真实文件格式也影响工程选择。

# Evaluation Metrics: Compression Decision Agent (CDA)

CR refers to the theoretical amount of compression achieved through image compression. When the original image has an input size of 512x512, it is represented by a dimension of $\mathbb { R } ^ { 5 1 2 \times 5 1 2 \times 3 }$ . However, for $\mathrm { V A E _ { 5 1 2 } }$ , the representation is $\mathbb { R } ^ { 1 , 0 0 0 }$ , while VQVAE512 is represented by R128x128x2 + R64x64x2 , and ESRGAN by $\mathbb { R } ^ { 1 2 8 \mathbf { x } 1 2 8 \mathbf { x } 3 }$ . The size of the dimension required by each method is expressed as the ratio of the CR. When adaptive compression was determined, the adaptive compression ratio was calculated by assuming the condition of no compression as the baseline, set to a value of one. Therefore, if only the original image was used, it was $1 0 0 \%$ , and if both were compressed using ESRGAN, it was $6 . 2 5 \%$ , equivalent to the compression ratio achieved by SwinIR and LDM.

> 💡 **CR 注意事项**: Methods 的 CR 是 theoretical dimension ratio，不是磁盘上的 `.svs/.tiff/.jpg/.png` 文件大小。Discussion 中 VQVAE 文件反而可能更大，就是因为 raw latent 维度和实际存储编码不是同一个概念。

# Evaluation Metrics: Classification Performance

The performance of the classification models was evaluated using various widely accepted metrics, including AUROC, AUPRC, accuracy, and F1 score.

1. AUROC: The AUROC evaluates a classifier’s ability to differentiate between positive and negative classes across varying decision thresholds. This provides a comprehensive measure of the discriminatory ability of the model.   
2. AUPRC: The AUPRC summarizes the trade-off between precision and recall across different threshold values. This is particularly useful for imbalanced datasets where positive samples are scarce.   
3. Accuracy: The proportion of correctly classified instances out of the total number of instances. Although it offers a straightforward assessment of the overall model performance, it may not be suitable for imbalanced datasets.

4. F1 score: The F1 score considers precision and recall, providing a balanced evaluation of the classifier’s performance. It is calculated as the harmonic mean of these two metrics and is particularly valuable when precision and recall are equally important.

# Evaluation Metrics: Segmentation Performance

The performance of the segmentation models was assessed using commonly employed metrics, namely the Dice coefficient and the Jaccard index.

1. Dice score: The Dice score measures the spatial overlap between the predicted segmentation mask and the ground truth mask, expressed as the harmonic mean of precision and recall (Equation 7).   
2. Jaccard index: The Jaccard index evaluates the similarity between two sets by computing the ratio of intersections to unions, and is also known as the Intersection over Union (IoU) (Equation 8).

The Dice coefficient was computed as:

$$
{ \mathrm { D i c e } } = { \frac { 2 \cdot { \mathrm { T P } } } { 2 \cdot { \mathrm { T P } } + { \mathrm { F P } } + { \mathrm { F N } } } } .
$$

The Jaccard index was computed as:

$$
\mathrm { J a c c a r d } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } + \mathrm { F N } } } .
$$

where TP, FP, and FN denote true positives, false positives, and false negatives, respectively.

# Statistics and reproducibility

For pretraining the FIEs, 30 WSIs were randomly sampled from each TCGA project. During patch extraction, patches were also randomly selected within tissue regions to ensure representative sampling.

For the reconstruction performance of the FIE, SSIM, PSNR, and LPIPS were computed for each patch and then averaged across the evaluation set to obtain global scores. In the VTT, model performance was statistically compared against chance-level accuracy (0.5) using a two-sided Wilcoxon signed-rank test, with $P < 0 . 0 5$ considered statistically significant. No other formal statistical tests were applied.

For downstream classification and segmentation tasks, model performance was evaluated using AUROC, AUPRC, Dice score, and Jaccard index as the primary evaluation metrics. For patch-level analyses, these metrics were calculated per patch and averaged within slides, whereas for slide-level analyses, patch-level features were aggregated using attention weights to generate slide-level representations for classification. All downstream experiments were repeated five times with different random seeds (42, 43, 44, 45, and 46), and mean performance values were reported. All experiments were conducted under identical hyperparameter settings, except for the specific comparison variable being tested.

No statistical method was used to predetermine sample size, and no data were excluded from the analyses. The experiments were not randomized, and the investigators were not blinded to allocation during experiments or outcome assessment. All datasets used in this study are publicly available, and all codes, model configurations, and trained weights will be released to ensure reproducibility.

> 💡 **统计解读**: VTT 有显著性检验；下游任务主要报告 5 seeds 均值，没有对每个任务做 formal significance。读结果时应把小幅提升视为趋势或 regularization 可能性，而不是严格临床优越性结论。

## Section 总结

| 方法组件 | 关键复现点 |
|---|---|
| PanCancer | TCGA 31 projects, 930 WSIs, 1.8M patches, 20x/40x |
| FIE | HR 512x512 downscale to 128x128; ESRGAN/SwinIR/LDM 等候选 |
| CDA | single-step REINFORCE, action in {Keep, Compress}, HoverNet mask Dice penalty |
| Reward | Compress 过低 Dice = -1；通过阈值后 reward = 1 - lambda * S_info |
| Downstream | 13 tasks, fixed downstream training, 5 random seeds |
| VTT | 5 pathologists, 30 pairwise trials, chance-level test |
