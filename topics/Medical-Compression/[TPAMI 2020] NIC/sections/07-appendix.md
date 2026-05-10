[← 返回 README](../README.md)

# 07 - Appendix

## 📌 预览

Appendix 给出可复现细节：各 encoder 的网络层数、优化器、学习率策略，synthetic 版本的缩小架构，以及 image-level CNN/Grad-CAM 的具体实现。

---

# ACKNOWLEDGMENT

This study was supported by a Junior Researcher grant from the Radboud Institute of Health Sciences (RIHS), Nijmegen, The Netherlands; a grant from the Dutch Cancer Society (KUN 2015-7970); and another grant from the Dutch Cancer Society and the Alpe d’HuZes fund (KUN 2014-7032); this project has also been partially funded by the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825292. The authors would like to thank Dr. Mitko Veta for evaluating our predictions in the test set of the TUPAC16 dataset, and the developers of Keras [46], the open source tool that we used to run our deep learning experiments.

> 💡 **独立评估来源**: Dr. Mitko Veta 对 TUPAC16 test set 预测做评估，这对应正文中“third-party independent evaluation”的可信度来源。

# REFERENCES

[1] G. Litjens et al., “A survey on deep learning in medical image analysis,” Medical Image Analysis, vol. 42, pp. 60 – 88, 2017.   
[2] L. Zhang et al., “Deep learning for remote sensing data: A technical tutorial on the state of the art,” IEEE Geoscience and Remote Sensing Magazine, vol. 4, no. 2, pp. 22–40, 2016.   
[3] M. Veta et al., “Predicting breast tumor proliferation from wholeslide images: the TUPAC16 challenge,” Medical image analysis, vol. 54, pp. 111–121, 2019.   
[4] B. Ehteshami Bejnordi et al., “Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer,” JAMA, vol. 318, no. 22, pp. 2199–2210, 2017.   
[5] K. Sirinukunwattana et al., “Gland segmentation in colon histology images: The glas challenge contest,” Medical image analysis, vol. 35, pp. 489–502, 2017.   
[6] X. Wang et al., “Weakly supervised learning for whole slide lung cancer image classification,” in Medical Imaging with Deep Learning, 2018.   
[7] M. Ilse et al., “Attention-based deep multiple instance learning,” in International Conference on Machine Learning, 2018, pp. 2132–2141.   
[8] M. Combalia et al., “Monte-carlo sampling applied to multiple instance learning for whole slide image classification,” in Medical Imaging with Deep Learning, 2018.   
[9] J. M. Tomczak et al., “Histopathological classification of precursor lesions of esophageal adenocarcinoma: A deep multiple instance learning approach,” in Medical Imaging with Deep Learning, 2018.   
[10] L. Hou et al., “Patch-based convolutional neural network for whole slide tissue image classification,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.   
[11] N. Coudray et al., “Classification and mutation prediction from non–small cell lung cancer histopathology images using deep learning,” Nature medicine, vol. 24, no. 10, p. 1559, 2018.   
[12] I. Goodfellow et al., “Convolutional networks,” in Deep Learning. MIT Press, 2016, ch. 9.   
[13] L. Theis et al., “Lossy image compression with compressive autoencoders,” in International Conference on Learning Representations (ICLR), 2017.   
[14] J. Donahue et al., “Adversarial feature learning,” in International Conference on Learning Representations (ICLR), 2017.   
[15] V. Dumoulin et al., “Adversarially learned inference,” in International Conference on Learning Representations (ICLR), 2017.   
[16] A. van den Oord et al., “Representation learning with contrastive predictive coding,” in Advances in Neural Information Processing Systems, 2018. in Advances in Neural Information Processing Systems, 2017, pp. 6306–6315.   
[18] D. P. Kingma et al., “Auto-encoding variational bayes,” in International Conference on Learning Representations, 2013.   
[19] G. Koch et al., “Siamese neural networks for one-shot image recognition,” in ICML Deep Learning Workshop, vol. 2, 2015.   
[20] I. Melekhov et al., “Siamese network features for image matching,” in Pattern Recognition (ICPR), 2016 23rd International Conference on. IEEE, 2016, pp. 378–383.   
[21] A. Hyvarinen et al., “Unsupervised feature extraction by timecontrastive learning and nonlinear ica,” in Advances in Neural Information Processing Systems, 2016, pp. 3765–3773.   
[22] I. Goodfellow et al., “Generative adversarial nets,” in Advances in Neural Information Processing Systems 27, 2014, pp. 2672–2680.   
[23] X. Chen et al., “Infogan: Interpretable representation learning by information maximizing generative adversarial nets,” in Advances in neural information processing systems, 2016.   
[24] R. R. Selvaraju et al., “Grad-cam: Visual explanations from deep networks via gradient-based localization.” in ICCV, 2017, pp. 618– 626.   
[25] D. Tellez et al., “Gigapixel Whole-Slide Image Classification Using Unsupervised Image Compression And Contrastive Training,” in Medical Imaging with Deep Learning, 2018.   
[26] D. Tellez et al., “Whole-Slide Mitosis Detection in H&E Breast Histology Using PHH3 as a Reference to Train Distilled Stain-Invariant Convolutional Networks,” IEEE Transactions on Medical Imaging, vol. 37, no. 9, pp. 2126–2136, 2018.   
[27] A. Krizhevsky et al., “Imagenet classification with deep convolutional neural networks,” in Advances in neural information processing systems, 2012, pp. 1097–1105.   
[28] P. Bandi ´ et al., “Comparison of different methods for tissue segmentation in histopathological whole-slide images,” in 2017 IEEE 14th International Symposium on Biomedical Imaging (ISBI 2017). IEEE, 2017, pp. 591–595.   
[29] F. Chollet, “Xception: Deep learning with depthwise separable convolutions,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 1251–1258.   
[30] Y. LeCun et al., “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.   
[31] J. N. Weinstein et al., “The cancer genome atlas pan-cancer analysis project,” Nature genetics, vol. 45, no. 10, p. 1113, 2013.   
[32] T. O. Nielsen et al., “A comparison of pam50 intrinsic subtyping with immunohistochemistry and clinical prognostic factors in tamoxifen-treated estrogen receptor–positive breast cancer,” Clinical cancer research, pp. 1078–0432, 2010.   
[33] F. Ciompi et al., “The importance of stain normalization in colorectal tissue classification with convolutional networks,” in Biomedical Imaging (ISBI 2017), 2017 IEEE 14th International Symposium on. IEEE, 2017, pp. 160–163.   
[34] O. Sertel et al., “Computer-aided prognosis of neuroblastoma on whole-slide images: Classification of stromal development,” Pattern recognition, vol. 42, no. 6, pp. 1093–1103, 2009.   
[35] A. Tabesh et al., “Multifeature prostate cancer diagnosis and gleason grading of histological images,” IEEE Transactions on Medical Imaging, vol. 26, no. 10, pp. 1366–1378, 2007.   
[36] J. Kong et al., “Image analysis for automated assessment of grade of neuroblastic differentiation,” in 2007 4th IEEE International Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2007, pp. 61–64.   
[37] H.-C. Shin et al., “Learning to read chest x-rays: Recurrent neural cascade model for automated image annotation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.   
[38] S. A. Hasan et al., “Overview of the ImageCLEF 2018 medical domain visual question answering task,” in CLEF2018 Working Notes, ser. CEUR Workshop Proceedings, 2018.   
[39] Y. Anavi et al., $^ { \prime \prime } \mathrm { A }$ comparative study for chest radiograph image retrieval using binary texture and deep learning classification,” in Engineering in Medicine and Biology Society (EMBC), 2015 37th Annual International Conference of the IEEE, 2015.   
[40] T. Schlegl et al., “Unsupervised anomaly detection with generative adversarial networks to guide marker discovery,” in International Conference on Information Processing in Medical Imaging, 2017.   
[41] Q. Yang et al., “Low dose ct image denoising using a generative adversarial network with wasserstein distance and perceptual loss,” IEEE Transactions on Medical Imaging, 2018.   
[42] D. Bang et al., “High quality bidirectional generative adversarial networks,” arXiv preprint arXiv:1805.10717, 2018.   
[43] M. Caron et al., “Deep clustering for unsupervised learning of visual features,” in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 132–149.   
[44] S. Jetley et al., “Learn to pay attention,” in International Conference on Learning Representations (ICLR), 2018.   
[45] T. Chen et al., “Training deep nets with sublinear memory cost,” arXiv preprint arXiv:1604.06174, 2016.   
[46] F. Chollet et al., “Keras,” https://keras.io, 2015.

> 💡 **参考文献结构**: 参考文献覆盖五条线：medical image analysis/WSI benchmark、MIL 弱监督、representation learning、GAN/BiGAN、Grad-CAM/efficient CNN。

![Author photo](../images/690e4d24abbf99208c0838fbba4e2ffc37e48d6d4c7ab09e1851aa01ff42a2b9.jpg)

David Tellez received the BSc and MSc degrees in telecommunication engineering from the University of Seville, Spain, in 2014. He is a PhD candidate in the Computational Pathology Group of Jeroen van der Laak in the Radboud University Medical Center, The Netherlands. His research interests include computer vision and medical imaging.

![Author photo](../images/c4514402538c6f10e9bb7a0e269bf1a771a4035328e1618a2b34889c0f1209ae.jpg)

Geert Litjens completed his PhD thesis on ”Computerized detection of prostate cancer in multi-parametric MRI” at the Radboud University Medical Center. He is currently assistant professor in Computation Pathology at the same institution. His research focuses on applications of machine learning to improve oncology diagnostics.

![Author photo](../images/c9b3116fe3e929f244b36bbaf83f3250bc205a60b6f759a51bbfdf18fc40ef2f.jpg)

Jeroen van der Laak holds an MSc in computer science and acquired his PhD from the Radboud University Medical Center in Nijmegen, The Netherlands, where he currently is associate professor of Computational Pathology. His research focuses on deep learning algorithm development and validation for improved pathology diagnostics.

![Author photo](../images/5ff234a5429550d82ec82f4564b63f6cdfe5980aea1d63589ca2a1052243151e.jpg)

Francesco Ciompi received the MSc degree in Computer Vision and Artificial Intelligence from the Autonomous University of Barcelona in 2008. In July 2012 he obtained the PhD Cum Laude from the University of Barcelona. Since 2013, he is part of the Diagnostic Image Analysis Group of Radboud University Medical Center. He is Assistant Professor of Computational Pathology, working on Deep Learning for automatic analysis of digital pathology whole-slide images.

# APPENDIX

# ENCODERS FOR HISTOPATHOLOGICAL DATA

# Variational Autoencoder

Two networks are trained simultaneously, the encoder $E$ and the decoder $D$ . The task of $E$ is to map an input patch $\boldsymbol { x } ~ \in ~ \mathbb { R } ^ { P \times P \times 3 }$ to a compact embedded representation $\bar { \boldsymbol { e } } \in \mathbb { R } ^ { C } ,$ , and the task of $D$ is to reconstruct $x$ from $e ,$ producing $\begin{array} { r l r } { x ^ { \prime } } & { { } \in } & { \mathbb { R } ^ { P \times P \times 3 } } \end{array}$ . In this work, we used a more sophisticated version of AE, the variational autoencoder (VAE) [18], with $P = 1 2 8$ and $C = 1 2 8$ .

The encoder in the VAE model learns to describe $x$ with an entire probability distribution, in particular, given an input $x ,$ the encoder $E$ outputs $\boldsymbol { \mu } \in \mathbb { R } ^ { \tilde { C } }$ and $\sigma \in \ \mathbb { R } ^ { C } .$ , two embeddings representing the mean and standard deviation of a normal distribution so that:

$$
e = \mu + \sigma \odot n
$$

with $n \sim \mathcal { N } ( 0 , 1 )$ and $\odot$ denoting element-wise multiplication.

The architecture of $E$ consisted of 5 layers of strided convolutions with 32, 64, 128, 256 and 512 $3 \times 3$ filters, batch normalization (BN) and leaky-ReLU activation (LRA); followed by a dense layer with 512 units, BN and LRA; and a linear dense layer with $C$ units.

The architecture of the decoder $D$ consisted of a dense layer with 8192 units, BN and LRA, eventually reshaped to $( 4 \times 4 \times 5 1 2 )$ ; followed by 5 upsampling layers, each composed of a pair of nearest-neighbor upsampling and a convolutional operation [?], with 256, 128, 64, 32 and $1 6 3 \times 3$ filters, BN and LRA; finalized with a convolutional layer with $3 3 \times 3$ filters and tanh activation.

We trained the VAE model by optimizing the following objective:

$$
\begin{array} { r l } & { \mathcal { V } _ { \mathrm { V A E } } ( x , n , \theta _ { E } , \theta _ { D } ) = } \\ & { = \underset { E , D } { \operatorname* { m i n } } \left[ \underbrace { \left( x - D ( E ( x , n ) ) \right) ^ { 2 } } _ { \mathrm { R e c o n s t r u c t i o n ~ e r r o r } } + \underbrace { \gamma ( 1 + \log \sigma ^ { 2 } - \mu ^ { 2 } - \sigma ^ { 2 } ) } _ { \mathrm { K L \ d i v e r g e n c e } } \right] } \end{array}
$$

with $x$ representing a single data sample, $n$ a sample from $\mathcal { N } ( 0 , 1 )$ , $\gamma$ a scaling factor, and $\theta _ { E }$ and $\theta _ { D }$ as the parameters of $E$ and $D ,$ , respectively. Note that we optimized $\theta _ { E }$ and $\theta _ { D }$ to minimize both the reconstruction error between the input and output data distributions, and the KL divergence between the embedding distribution and the normal $\check { \mathcal { N } } ( 0 , 1 )$ distribution with $\gamma = \breve { 5 } \times 1 0 ^ { - 5 }$ .

We minimized $\mathcal { V } _ { \mathrm { V A E } }$ using stochastic gradient descent with Adam optimization and 64-sample mini-batch, decreasing the learning rate by a factor of 10 starting from $1 \times 1 0 ^ { - 3 }$ every time the validation loss plateaued until $1 \times 1 0 ^ { - 5 }$ . Finally, we selected the encoder $E$ corresponding to the VAE model with the lowest validation loss.

> 💡 **VAE 复现要点**: histopathology 版本用 5 层 strided conv 压到 $C=128$，优化选择最低 validation reconstruction/KL loss；下游不使用 decoder。

# Contrastive Training

We assembled a training dataset composed of pairs of patches ${ \pmb x } = \{ { \boldsymbol x } ^ { ( 1 ) } , { \boldsymbol x } ^ { ( 2 ) } \}$ with $\boldsymbol { x } ^ { ( i ) } ~ \in ~ \mathbb { R } ^ { \mathbf { \dot { P } } \times \mathbf { \mathit { P } } \times 3 }$ where each pair $_ { \textbf { \em x } }$ was associated with a binary label $y ,$ and $P = 1 2 8$

In order to solve this binary classification task, we trained a two-branch Siamese network [20] called $S$ . Both input branches shared weights and consisted of the same encoding architecture $E$ as the VAE model. After concatenation of the resulting embedding vectors, a MLP followed consisting of a dense layer with 256 units, BN and LRA; finalized by a single sigmoid unit.

We minimized the binary cross-entropy loss using stochastic gradient descent with Adam optimization and 64-sample mini-batch, decreasing the learning rate by a factor of 10 starting from $1 \times 1 0 ^ { - 2 }$ every time the validation classification accuracy plateaued until $\mathrm { i } \times 1 0 ^ { - 5 }$ . Finally, we selected the encoder $E$ corresponding to the $S$ with the highest validation classification accuracy.

> 💡 **Contrastive 复现要点**: Siamese 两支共享 VAE encoder 架构，训练完丢掉 pair classifier，只保留一支 encoder $E$。

# Bidirectional Generative Adversarial Network

We trained a BiGAN setup consisting of three networks: a generator $G ,$ a discriminator $D$ and an encoder $E , ~ G$ mapped a latent variable $z$ drawn from a normal distribution $\mathcal { N } ( 0 , 1 )$ into artificial images $x ^ { \prime }$ :

$$
z \sim \mathcal { N } ( 0 , 1 ) \in \mathbb { R } ^ { C } \overset { G } { \to } x ^ { \prime } \in \mathbb { R } ^ { P \times P \times 3 }
$$

whereas $E$ mapped images $x$ sampled from the true data distribution $\mathcal { X }$ into embeddings $e$ :

$$
x \sim \mathcal { X } \ \in \ \mathbb { R } ^ { P \times P \times 3 } \ \xrightarrow { E } e \in \ \mathbb { R } ^ { C }
$$

During training, the three networks played a minimax game where the discriminator $D$ tried to distinguish between actual and artificial image-embedding pairs, i.e. $\{ x , e \}$ and $\{ x ^ { \prime } , z \}$ respectively, while $G$ and $E$ tried to fool $D$ by producing increasingly more realistic images $x ^ { \prime }$ and embeddings $e _ { \cdot }$ , i.e. closer to $\mathcal { N } ( 0 , 1 )$ . We used $P = 1 2 8$ and $C = 1 2 8$ .

Given the difficulty of training a stable BiGAN model, we downsampled $x$ by a factor of 2 before feeding it to the model. The architecture of the encoder $E$ consisted of 4 layers of strided convolutions with $1 2 8 3 \times 3$ filters, BN and LRA; followed by a linear dense layer with $C$ units.

The architecture of the generator $G$ consisted of a dense layer with 1024 units, BN and LRA, eventually reshaped to $( 4 \times 4 \times 6 4 )$ ); followed by 4 upsampling layers, each composed of a pair of nearest-neighbor upsampling and a convolutional operation [?], with $1 2 8 \mathrm { ~ 3 ~ } \times \mathrm { ~ 3 ~ }$ filters, BN and LRA; finalized with a convolutional layer with $3 3 \times 3$ filters and tanh activation.

The discriminator $D$ had two inputs, a low-dimensional vector and an image. The image was fed through a network with an architecture equal to $E$ but different weights, and the resulting embedding vector concatenated to the input latent variable. This concatenation layer was followed by two dense layers with 1024 units, LRA and dropout (0.5 factor); finalized with a sigmoid unit.

We optimized the following objective function:

$$
\begin{array} { r l } & { \mathcal { V } _ { \mathrm { B i G A N } } ( x , z , \theta _ { G } , \theta _ { E } , \theta _ { D } ) = } \\ & { = \underset { G , E } { \operatorname* { m i n } } \underset { D } { \operatorname* { m a x } } \left[ \log \left[ D \big ( x , \underset { e } { \underbrace { E ( x ) } } \big ) \right] + \log \left[ 1 - D \big ( \underset { x ^ { \prime } } { \underbrace { G ( z ) } } , z \big ) \right] } \end{array}
$$

with $\theta _ { G } , \theta _ { E }$ and $\theta _ { D }$ representing the parameters of $G , E$ and $D$ , respectively.

We minimized $\nu _ { \mathrm { B i G A N } }$ using stochastic gradient descent with Adam optimization, 64-sample mini-batch, and fixed learning rate of $2 \times 1 0 ^ { - 4 }$ for a total of 200,000 epochs. Finally, we selected the encoder $E$ corresponding to the last epoch.

> 💡 **BiGAN 复现要点**: 为稳定训练，输入 patch 先下采样 2 倍；encoder 是 4 层 strided conv；论文选择最后 epoch 的 $E$，不是 validation-selected model。

# Mean-RGB Baseline

We extracted the embedding $e$ by averaging the pixel intensity across spatial dimensions from input RGB patches x ∈ RP ×P ×3:

$$
e ^ { ( c ) } = \frac { 1 } { P ^ { 2 } } \sum _ { j = 1 } ^ { P } \sum _ { k = 1 } ^ { P } x ^ { ( j , k , c ) }
$$

with $c$ indexing the three RGB color channels, and $j$ and $k$ indexing the two spatial dimensions.

> 💡 **Mean-RGB baseline**: 这是 $C=3$ 的颜色均值表示，用来检验复杂 encoder 是否真的超越染色/颜色 shortcut。

# Supervised-tumor Baseline

We trained an encoder $E$ identical to the one used in the VAE model, followed by a dense layer with 256 units, BN and LRA; and finalized by a single sigmoid unit.

We minimized the binary cross-entropy loss using stochastic gradient descent with Adam optimization and 64-sample mini-batch, decreasing the learning rate by a factor of 10 starting from $1 \times 1 0 ^ { - \widetilde { 2 } }$ every time the validation classification accuracy plateaued until $\mathrm { i } \times 1 0 ^ { - 5 }$ . Finally, we selected the encoder $E$ corresponding to the model with the highest validation classification accuracy.

# Supervised-tissue Baseline

We trained an encoder $E$ identical to the one used in the VAE model, followed by a dense layer with 256 units, BN and LRA; and finalized by a softmax layer with nine units.

We minimized the categorical cross-entropy loss using stochastic gradient descent with Adam optimization and 64-sample mini-batch, decreasing the learning rate by a factor of 10 starting from $1 \times 1 0 ^ { - \ Y }$ every time the validation classification accuracy plateaued until $\mathrm { i } \times 1 0 ^ { - 5 }$ . Finally, we selected the encoder $E$ corresponding to the model with the highest validation classification accuracy.

> 💡 **Supervised baselines**: 这两个 baseline 使用人工 patch labels，所以它们回答的是“如果有病灶/组织标注，上界大概怎样”，不是 NIC 的弱监督主设定。

# ENCODERS FOR SYNTHETIC DATA

We modified the network architectures to account for the smaller patch size selected in the synthetic dataset. We used grayscale patches $\begin{array} { r } { { \boldsymbol { x } } ~ \in ~ \mathbb { R } ^ { P \times P \times 1 } } \end{array}$ with $P = 9$ and $C = 1 6$ unless stated otherwise. Note that we did not modify the training protocol, e.g. the learning rate schedule.

The architecture of the encoder $E$ used in VAE, Contrastive and Supervised consisted of 2 layers of strided convolutions with 32 and $6 4 3 \times 3$ filters, BN and LRA; followed by a dense layer with 64 units, BN and LRA; and a linear dense layer with $C$ units.

The architecture of the decoder $D$ used in VAE consisted of a dense layer with 2048 units, BN and LRA, eventually reshaped to $( 4 \times 4 \times 1 2 8 )$ ; followed by 2 upsampling layers, each composed of a pair of nearest-neighbor upsampling and a convolutional operation [?], with 64 and $3 2 3 \times 3$ filters, BN and LRA; finalized with a convolutional layer with 1 $3 \times 3$ filters and tanh activation.

With respect to BiGAN, the architecture of the encoder $E$ consisted of 2 layers of strided convolutions with 32 $3 \times 3$ filters, BN and LRA; followed by a linear dense layer with $C$ units. The architecture of the generator $G$ consisted of a dense layer with 512 units, BN and LRA, eventually reshaped to $( 4 \times 4 \times 3 2 )$ ; followed by 2 upsampling layers, each composed of a pair of nearest-neighbor upsampling and a convolutional operation [?], with $3 2 \ 3 \times 3$ filters, BN and LRA; finalized with a convolutional layer with $1 \ 3 \times 3$ filters and tanh activation. The discriminator $D$ had two inputs, a low-dimensional vector and an image. The image was fed through a network with an architecture equal to $E$ but different weights, and the resulting embedding vector concatenated to the input latent variable. This concatenation layer was followed by two dense layers with 128 units, LRA and dropout (0.5 factor); finalized with a sigmoid unit.

> 💡 **Synthetic 架构缩放**: patch 从 128 缩到 9，所以 encoder 从 5/4 层缩到 2 层，保持“局部 patch → code vector”的接口一致。

# EXPERIMENTS

# Patch-level Classification

On top of each encoder with frozen weights, we trained an MLP consisting of a dense layer with 256 units, BN and LRA; followed by either a single sigmoid unit or a softmax layer with nine units, respectively for each classification task.

We minimized the cross-entropy loss using stochastic gradient descent with Adam optimization and 64-sample mini-batch, decreasing the learning rate by a factor of 10 starting from $1 \times 1 0 ^ { - 2 }$ every time the validation classification accuracy plateaued until $1 \times 1 0 ^ { - 5 }$ . Finally, we selected the model with the highest validation classification accuracy.

# Image-level Classification and Regression

We designed a CNN architecture consisting of 8 layers of strided depthwise separable convolutions [29] with $1 2 8 3 \times 3$ filters, BN, LRA, feature-wise $2 0 \%$ dropout, L2 regularization with $1 \times 1 0 ^ { - 5 }$ coefficient, and stride of 2 except for the 7-th and 8-th layers with no stride; followed by a dense layer with 128 units, BN and LRA; and a final output unit, with linear or sigmoid activation for regression or classification tasks, respectively.

We trained the CNN using stochastic gradient descent with Adam optimization and 16-sample mini-batch, decreasing the learning rate by a factor of 10 starting from $1 \times 1 0 ^ { - 2 }$ every time the validation metric plateaued until $1 \times 1 0 ^ { - 5 }$ . We minimized MSE for regression, and maximized binary cross-entropy for binary classification.

> 💡 **Image-level CNN 复现要点**: 主干是 8 层 depthwise separable conv，mini-batch 只有 16；classification 用 sigmoid，regression 用 linear，分别对应 Camelyon16 和 TUPAC16。

# Visualizing where the information is located

Given a compressed gigapixel image $\omega ^ { \prime }$ , its associated image-level label $y$ and a trained CNN $S ,$ we performed a forward pass over $\omega ^ { \prime }$ , producing a set of $J$ intermediate three-dimensional feature volumes $f _ { j } ^ { ( k ) }$ , with $j$ and $k$ indicating the $j$ -th and $k$ -th convolutional layer and feature map, respectively. Additionally, we computed the gradients of the feature volume $f _ { j } ^ { ( k ) }$ with respect to the class output $y ,$ , for a fixed convolutional layer. We averaged the gradients across the spatial dimensions and obtained a set of gradient coefficients $\dot { \gamma } _ { j } ^ { ( k ) }$ indicating how relevant each feature map was for the desired output $y$ . Finally, we performed a weighted sum of the feature map s f ( k ) using the gradient coefficients γ(kj

$$
h ^ { ( k ) } = \sum _ { j = 1 } ^ { J } f _ { j } ^ { ( k ) } \gamma _ { j } ^ { ( k ) }
$$

What we obtained was a two-dimensional heatmap $h ^ { ( k ) }$ that highlighted the regions of $\omega ^ { \prime }$ that were more relevant for $S$ to predict $y$ . In order to maximize the resolution of the generated heatmap, we selected $k = 1$ .

Grad-CAM heatmaps for Camelyon16 and TUPAC16 can be found here: https://drive.google.com/drive/folders/16E-06rFbGab6- pXfjpo9vXjBVyUQKUuc

> 💡 **Appendix 小结**: 若要复现 NIC，最关键的不是单个公式，而是接口一致性：所有 encoder 都输出 $C$ 维 patch embedding，所有 WSI 都变成 $M/S \times N/S \times C$，image-level CNN 和 Grad-CAM 都只在这个 compressed grid 上工作。
