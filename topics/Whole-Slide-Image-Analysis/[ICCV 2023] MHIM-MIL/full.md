# Multiple Instance Learning Framework with Masked Hard Instance Mining for Whole Slide Image Classification

Wenhao Tang<sup>1</sup> Sheng Huang<sup>1\*</sup> Xiaoxian Zhang<sup>1</sup> Fengtao Zhou<sup>2</sup> Yi Zhang<sup>1</sup> Bo Liu<sup>3</sup> <sup>1</sup> Chongqing University <sup>2</sup> The Hong Kong University of Science and Technology <sup>3</sup> Walmart Global Tech

{whtang, huangsheng, zhangxiaoxian, zhangyii}@cqu.edu.cn fzhouaf@connect.ust.hk, kfliubo@gmail.com

## Abstract

The whole slide image (WSI) classification is often formulated as a multiple instance learning (MIL) problem. Since the positive tissue is only a small fraction of the gigapixel WSI, existing MIL methods intuitively focus on identifying salient instances via attention mechanisms. However, this leads to a bias towards easy-to-classify instances while neglecting hard-to-classify instances. Some literature has revealed that hard examples are beneficial for modeling a discriminative boundary accurately. By applying such an idea at the instance level, we elaborate a novel MIL framework with masked hard instance mining (MHIM-MIL), which uses a Siamese structure (Teacher-Student) with a consistency constraint to explore the potential hard instances. With several instance masking strategies based on attention scores, MHIM-MIL employs a momentum teacher to implicitly mine hard instancesfor training the student model, which can be any attention-based MIL model. This counter-intuitive strategy essentially enables the student to learn a better discriminating boundary. Moreover, the student is used to update the teacher with an exponential moving average (EMA), which in turn identifies new hard instances for subsequent training iterations and stabilizes the optimization. Experimental results on the CAMELYON-16 and TCGA Lung Cancer datasets demonstrate that MHIM-MIL outperforms other latest methods in terms of performance and training cost. The code is available at: https://github.com/DearCaat/MHIM-MIL.

## 1. Introduction

Histopathological image analysis plays a crucial role in modern medicine, particularly in the treatment of cancer, where it serves as the gold standard for diagnosis [23, 25, 29,51]. Digitalizating pathological images into Whole Slide

![](images/73e17a69fae8ff43cd06320755a30157ce863f7da297366d622987fea548b822.jpg)  
Figure 1: Left: Previous MIL models focus on the more salient instances. Right: MHIM-MIL mines an amount of hard-to-classify instances to learn a better boundary.

Images (WSIs) through digital slide scanner has opened new avenues for computer-aided analysis [9, 31]. Due to the huge size of a WSI and the lack of pixel-level annotations, histopathological image analysis is commonly formulated as a multiple instance learning (MIL) task [11, 28, 36]. In MIL, each WSI (or slide) is a bag containing thousands of unlabeled instances (patches) cropped from the slide. With at least one instance being disease positive, the bag is deemed positive, otherwise negative.

However, the number of slides is limited and each slide contains a mass of instances with a low positive proportion. This imbalance would hinder the inference of bag labels [20, 49]. To alleviate this issue, several WSI classification methods [6, 20, 22, 23, 31] employ an attention mechanism to aggregate salient instance features into a bag-level feature for WSI classification. Furthermore, some MIL frameworks [22, 26, 46, 49] focus on the more salient instances in the bag and leverage them to facilitate WSI classification. For instance, existing frameworks [46, 49] propose to only select the instances that correspond to the top K highest or lowest attention scores [22, 46] or patch probabilities [49] for yielding high-quality bag embedding for both training and testing.

These salient instances are actually “easy-to-classify” instances, which are not optimal for training a discriminative WSI classification model. In conventional machine learning, such as Support Vector Machines (SVM) [17], samples near the category distribution boundary are more challenging to classify, but are more useful for depicting the classification boundary, as illustrated in Figure 1. Moreover, other deep learning works [30, 33, 38, 39] also reveal that mining hard samples for training can improve the generalization abilities of models. By applying such an idea at the instance level, we can better highlight the “hard-to-classify” instances that facilitate MIL model training, and benefit the final WSI classification. However, the lack of instance labels poses a challenge to the direct application of traditional hard sample mining strategies at the instance level.

To address this issue, we present a novel MIL framework based on masked hard instance mining strategies (MHIM) named MHIM-MIL. The main idea of MHIM is to mask out the instances with high attention scores to highlight the hard instances for model training. Based on this, we incorporate two other instance masking strategies to enhance training efficiency and mitigate the over-fitting risk. Another key design of MHIM-MIL is an instance attention generator based on a Siamese structure (Teacher-Student) [3, 8]. In MHIM-MIL, the MIL-based WSI classification model is the student network, which aggregates hard instances mined by a momentum teacher with different instance masking strategies. The momentum teacher is updated using an exponential moving average (EMA) of the student model. Moreover, the framework is optimized by inducing a consistency constraint that explores more supervised information beyond the limited slide label. Unlike the conventional MIL frameworks [46, 49], which adopt complex cascade gradient-updating structures, our method is more simple and does not require additional parameters. It not only improves efficiency but also provides improved performance stability. The contribution of this paper is summarized as follows,

• We propose a simple and efficient MIL framework with masked hard instance mining named MHIM-MIL. It implicitly mines hard instances with instance attention for training a more discriminative MIL model. Extensive experiments on two WSI datasets validate that MHIM boosts different MIL models and outperforms other latest methods in terms of performance and training cost.

• We propose several hybrid instance masking strategies for indirectly mining hard instances in MIL. These strategies not only address the reliance problem of conventional methods on instance-level supervision but also enhance the training efficiency of the model and mitigate the over-fitting risk.

• With the Siamese structure, we introduce a parameterfree momentum teacher to obtain instance attention scores more efficiently and stably. Moreover, we employ a consistency-based iterative optimization to improve the discriminability of both models progressively.

## 2. Related Work

## 2.1. Multiple Instance Learning in WSI Analysis

Multiple Instance Learning (MIL) [11] has been widely used in WSI analysis with its unique learning paradigm in recent years [22, 27, 31, 41, 46, 48]. MIL is a weakly supervised learning framework that utilizes coarse-grained bag labels for training instead of fine-grained instance annotations. Previous algorithms can be broadly categorized into two groups: instance-level [4, 14, 19, 46] and embeddinglevel [9, 32, 44, 45, 49]. The former obtain instance labels and aggregate them to obtain the bag label, whereas the latter aggregate all instance features into a high-level bag embedding for bag prediction. Most embedding-level methods share the basic idea of AB-MIL [20], which employs learnable weights to aggregate salient instance features into bag embedding. Furthermore, some MIL frameworks [22, 26, 46, 49] mine more salient instances making classification easier and facilitating classification. For example, Lu et al. selected the most salient instances based on their attention scores (e.g., maximum and minimum scores) to compute instance-level loss and improve performance [26]. Zhang et al. proposed a class activation map (CAM) based on the AB-MIL paradigm to better mine salient instances and used AB-MIL to aggregate them into bag embedding [49]. In addition, feature clustering methods [32,43,50] computed cluster centroids of all feature embeddings and used representative embeddings for the final prediction. However, all these methods focused excessively on salient instances in training, which are easy instances with high confidence scores and can be easily classified. As a result, they overlook the importance of hard instances for training. In this paper, we intend to mine hard instances for improving WSI classification performance.

## 2.2. Hard Sample Mining in Computer Vision

Hard sample mining is a popular technique to speed up convergence and enhance the discriminative power of the model in many deep learning areas, such as face recognition [30], object detection [34, 42], person reidentification [1, 33, 38, 39], and deep metric learning [35, 37]. The main idea behind this technique is to select the samples which are hard to classify correctly (i.e., hard negatives and hard positives) for alleviating the imbalance between positive and negative samples and facilitating model training. There are generally three groups of approaches for evaluating sample difficulty: loss-based [18], similaritybased [7], and learnable weight-based [47]. Typically, these strategies require complete sample supervision information. Drawing on the ideas of the above works, we propose a hard instance mining approach in MIL, mining hard examples at the instance level. In this, there are no complete instance labels, only the bag label is available. Similar to our approach, Li et al. utilized attention scores to identify salient instances from false negative bags to serve as hard negative instances and used them to compose the hard bags for improving classification performance [24]. A key difference is that we indirectly mine hard instances by masking out the most salient instances rather than directly locating hard negative instances.

## 3. Proposed Method

## 3.1. Background: MIL Formulation

In MIL, any input WSI X is considered as a bag with multiple instances, which can be represented as $\begin{array} { r l } { X } & { { } = } \end{array}$ $\{ x _ { i } \} _ { i = 1 } ^ { \tilde { N } } , x _ { i }$ is a patch collected from the WSI and considered as the i-th instance of X. N is the number of instances. For a classification task, there exists a known label $Y \in C$ for the bag and an unknown label $y _ { n } \in C$ for each instance, where C is the collection of category labels. The goal of a MIL model $\mathcal { M } ( \cdot )$ is to predict the bag label with all instances $\hat { Y } \gets { \mathcal { M } } ( \ddot { X } )$ . The popular solution is to learn a bag representation $F$ from the extracted features of instances $\bar { Z = } \{ z _ { i } \} _ { i = 1 } ^ { N }$ in a bag, which is also referred as the instance aggregation step. And a classifier $\mathcal { C } ( \cdot )$ , trained upon the $F ,$ can be used to predict the bag label $\hat { Y }  \mathcal { C } ( F )$ . There are two ways to aggregate instances for achieving bag embedding. One is the attention-based aggregation [20] denoted as follows,

$$
F = \sum _ { i = 1 } ^ { N } a _ { i } z _ { i } \in \mathbb { R } ^ { D } ,\tag{1}
$$

where $a _ { i }$ is the learnable scalar weight for $z _ { i }$ , and $D$ is the dimension of vector $F$ and $z _ { i }$ . Many works [22, 26, 49] follow this formulation but differ in the ways they generate the attention score $a _ { i }$

Another is the multi-head self-attention (MSA) based aggregation [31]. In this fashion, a class token $z _ { \mathrm { 0 } }$ is embedded with the instance features to get the initial input sequence $Z ^ { 0 } \ = \ [ z _ { 0 } , z _ { 1 } , \ldots , z _ { N } ] \ \in \ \mathbb { R } ^ { \left[ \bar { N } + 1 \right) \times D }$ for aggregating instance features. This can be formulated as,

$$
\begin{array} { r l r } & { \mathbf { h e a d } = A ^ { \ell } \left( Z ^ { \ell - 1 } W ^ { V } \right) \in \mathbb { R } ^ { N \times \frac { D } { H } } , } & { \ell = 1 \dots L } \\ & { Z ^ { \ell } = \mathrm { C o n c a t } \left( \mathrm { h e a d } _ { 1 } , \cdot \cdot \cdot , \mathrm { h e a d } _ { H } \right) W ^ { O } , } & { \ell = 1 \dots L } \end{array}\tag{2}
$$

where $W ^ { V } \in \mathbb { R } ^ { D \times \frac { D } { H } }$ and $W ^ { O } \in \mathbb { R } ^ { D \times D }$ are the learnable projection matrices of MSA. $A ^ { \ell } \ \in \ \mathbb { R } ^ { ( N + 1 ) \times ( N + 1 ) }$ is the attention matrix of the ℓ-th layer, L is the number of MSA

block, and H is the number of head in each MSA block. The bag embedding $F$ is the output class token at the final layer,

$$
F = Z _ { 0 } ^ { L } .\tag{3}
$$

The self-attention-based bag embedding is essentially a special case of attention-based bag embedding in the multiinstance learning setting. Collectively, these approaches can be referred to as the general attention-based MIL method.

## 3.2. MHIM-MIL for WSI Classification

In general attention-based MIL frameworks, the attention scores of instances indicate the contributions of instances to the bag classification. The salient instances with high scores are useful for classifying WSI in the testing phase but are not conducive to training a MIL model with good generalization ability. Although hard samples have been proven to enhance the generalization ability of the model in many computer vision scenarios [13, 37–39], previous MIL works focus more on exploiting the salient instances and neglecting the utilization of hard instances in model optimization.

In this paper, we propose a simple and efficient MIL framework with Masked Hard Instance Mining (MHIM-MIL) to boost the WSI classification. As illustrated in Figure 2, the MHIM-MIL framework employs a Siamese structure during the training phase. The main component of our framework is a general attention-based MIL model (Student), denoted as $\boldsymbol { \mathcal { S } } ( \cdot )$ , for aggregating instance features. To increase the discriminatory difficulty of the student model and force it to focus on hard instances, we introduce a momentum teacher, denoted as $\tau ( \cdot )$ , to score the instances with attention weights and then employ some masked hard instance mining strategies to mask the salient instances while preserving the hard instances. After hard instance mining, all the mined features are forwarded into the student model for the inference of the bag label. The teacher shares the same network structure as the student model but does not need gradient-based updates. It is worth mentioning that, due to the varying number of instances within each bag, the non-batch gradient descent algorithm (i.e., SGD with batch size 1) is typically employed to optimize the MIL model. Therefore, compared to the traditional MIL frameworks with two-tier gradient updating models [46, 49], this Siamese structure makes training more stable and efficient with fewer parameters. The proposed framework can be defined as,

$$
\hat { Y } = \mathcal { S } \left( \hat { Z } \right) = \mathcal { S } \left( M _ { \mathcal { T } } \left( Z \right) \right) ,\tag{4}
$$

where $M \tau ( \cdot )$ denotes a masked hard instance mining strategy through the teacher model and $\hat { Z }$ are the mined instances.

![](images/076be4bd7553b19e6ef7e4762ba36f478268151963b4c0f6daaa88fe827e5323.jpg)  
Figure 2: Overview of proposed MHIM-MIL. A momentum teacher is used to compute attention scores for all instances. We mask instances based on attention with hard mining strategies and feed the remaining to the student model. The student is updated by a consistency loss term $\mathcal { L } _ { c o n }$ and a label error loss term $\mathcal { L } _ { c l s }$ . The teacher parameters are updated with an Exponential Moving Average (EMA) of the student parameters without gradient updates. In the inference phase, we use the complete input instances and the student model only.

## 3.3. Masked Hard Instance Mining Strategy

Conventional hard sample mining strategies are difficult to apply without instance-level supervision. We address this challenge by proposing masked hard instance mining strategies that use attention scores to implicitly mine hard instances by masking out easy instances with high attention scores. More specifically, given a complete sequence of instance features $Z = \{ z _ { i } \} _ { i = 1 } ^ { N }$ as the input of the teacher model $\tau ( \cdot )$ , the teacher outputs the attention weight $a _ { i }$ for each instance as follow,

$$
A = \left[ a _ { 1 } , \ldots , a _ { i } , \ldots , a _ { N } \right] = \mathcal { T } \left( Z \right) .\tag{5}
$$

Then, we obtain the indices of the attention sequence in descending order by applying a sorting operation on A,

$$
I = [ i _ { 1 } , i _ { 2 } , \ldots , i _ { N } ] = { \mathrm { S o r t } } \left( A \right) ,\tag{6}
$$

where $i _ { 1 }$ is the index of the instance with the highest attention score while $i _ { N }$ is the index of the one with the lowest score. With this index collection I, we will present several masked hard instance mining strategies to select the hard instances. We define an N-dimensional binary vector $M = [ m _ { 1 } , \dots , m _ { i } , \dots , m _ { N } ]$ for encoding the mask flags of instances where $m _ { i } \in \{ 0 , 1 \}$ . If $m _ { i } = 1$ , the i-th instance is masked, otherwise, it is unmasked.

High Attention Masking: The simplest masked hard instance mining strategy is the High Attention Masking (HAM) strategy, which simply masks instances with the top $\beta _ { h }$ % highest attention scores. The instance mask flags under HAM are initialized as all zero vectors, $M _ { h } ( : ) = 0$

Then we collect the indices of the instances whose scores are ranked in the top $\beta _ { h } \% , I _ { h } = [ i _ { t } ] _ { t = 1 } ^ { \lceil \beta _ { h } \% \times N \rceil }$ . Finally, we set the mask flags with these indices, $M _ { h } ( I _ { h } ) ~ = ~ 1$ . To ensure that positive instances are preserved within the unmasked sequences, we also utilized techniques such as mask ratio decay.

Hybrid Masking: We combine HAM with several other instance masking strategies as hybrid masking strategies to achieve some specific properties in hard instance mining, as shown in Figure 3. We consider the obtained mask flags as a collection and employ the union operation for mask flag fusion. We design three hybrid masking strategies as follows:

• L-HAM: We use the same pipeline as HAM to generate the mask flags $M _ { l }$ for masking the instances with the top $\beta _ { l } \%$ lowest attention scores in order to filter out the redundant uninformative instances and improve efficiency. To endower this property to HAM, we union the mask flags obtained by two strategies to get the new mask flags, $\hat { M } = M _ { h } \cup M _ { l }$

• R-HAM: Randomness is beneficial to reduce the risk of over-fitting. We generate a random mask flag vector $M _ { \tau }$ with a given random ratio $\beta _ { r } \%$ , and combine it with $M _ { h }$ for introducing the randomness to the hard instance mining, $\hat { M } = M _ { h } \cup M _ { r }$

• LR-HAM: Combining the above strategies, we can obtain completely hybrid mask flags, $\hat { M } \ = \ M _ { h }$ ∪ $M _ { r } \cup M _ { l }$ , which is expected to achieve both of the mentioned desirable properties.

![](images/3831079fdbc14354534c7aca80893d2ff16bfd9384ab7c3b744df4fca2d2092f.jpg)  
Figure 3: Illustration of proposed hybrid masking strategy for hard instance mining.

Once the final mask flag $\hat { M }$ is produced, the masked instance sequence will be obtained:

$$
\hat { Z } = M _ { \mathcal { T } } \left( Z \right) = \mathbf { M a s k } \left( Z , \hat { M } \right) \in \mathbb { R } ^ { \hat { N } \times D } ,\tag{7}
$$

where the $\hat { N }$ is the number of unmasked instances.

## 3.4. Consistency-based Iterative Optimization

Under the Siamese structure, while the teacher model guides the training of the student model, the new knowledge learned by the student model will also update the teacher model. This iterative optimization process progressively improves the mining ability of the teacher and the discriminability of the student. To further facilitate this optimization and explore additional supervised information provided by the momentum teacher, we propose a consistency loss that constrains the classification results of both models.

Student Optimization: There are two losses in student optimization. One is the cross-entropy for measuring the bag label prediction loss,

$$
\mathcal { L } _ { c l s } = Y \log \hat { Y } + ( 1 - Y ) \log \left( 1 - \hat { Y } \right) .\tag{8}
$$

Another is a consistency loss between the bag representation of student $F _ { s }$ and momentum teacher $F _ { t } .$

$$
\mathcal { L } _ { c o n } = - \mathrm { s o f t m a x } \left( F _ { t } / \tau \right) \log F _ { s }\tag{9}
$$

where the $\tau > 0$ is a temperature parameter. Overall, the final optimization loss is as follows:

$$
\{ \hat { \theta _ { s } } \}  \arg \operatorname* { m i n } _ { \theta _ { s } } \mathcal { L } = \mathcal { L } _ { c l s } + \alpha \mathcal { L } _ { c o n }\tag{10}
$$

where $\theta _ { s }$ is the parameters of $\boldsymbol { \mathcal { S } } ( \cdot )$ , and α is scaling factor. Teacher Optimization: The parameters of momentum teacher $\theta _ { t }$ are updated by an exponential moving average (EMA) of the student parameters. The update rule is $\theta _ { t } \gets \lambda \theta _ { t } + ( 1 - \lambda ) \theta _ { s }$ , where λ is a hyperparameter. More importantly, the updated teacher is utilized in the next iteration of hard instance mining.

## 4. Experiments and Results

## 4.1. Datasets and Evaluation Metrics

CAMELYON-16 [2] is a WSI dataset proposed for metastasis detection in breast cancer. The dataset contains a total of 400 WSIs, which are officially split into 270 for training and 130 for testing, and the testing sample ratio is 13/40≈1/3. Following [6, 26, 50], we adopt 3-times 3-fold cross-validation on this dataset to ensure that each slide is used in training and testing, which can alleviate the impact of data split and random seed on the model evaluation. Each fold has approximately 133 slides. We report the mean and standard deviation of performance metrics over 3 runs.

TCGA Lung Cancer includes two sub-type of cancers, Lung Adenocarcinoma (LUAD) and Lung Squamous Cell Carcinoma (LUSC). There are diagnostic slides, LUAD with 541 slides from 478 cases, and LUSC with 512 slides from 478 cases. We randomly split the dataset into training, validation, and testing sets with a ratio of 65:10:25 on the patient level. 4-fold cross-validation is adopted, and the mean and standard deviation of performance metrics of the 4 test folders are reported.

We adopt the same data pre-processing as in the CLAM [26]. Following the previous work [26, 31] we leverage Accuracy, Area Under Curve (AUC), and F1-score to evaluate model performance. AUC is the primary performance metric in the binary classification task, and we only report AUC in ablation experiments. Please refer to the Supplementary Material for the details of these two datasets.

## 4.2. Implementation Details

The details on network architectures and training are described in Supplementary Material.

## 4.3. Performance Comparison with Exiting Works

We mainly compare with AB-MIL [20], DSMIL [22], CLAM-SB [26], CLAM-MB [26], TransMIL [31], and DTFD-MIL [49], all of which are attention-based MIL methods. In addition, we compared two traditional MIL pooling operations, Max-pooling and Mean-pooling. Due to the dataset differences, the results of all other methods are reproduced using the official code they provide under the same settings.

As shown in Table 1, max-pooling and mean-pooling perform poorly on two datasets compared to other methods. We attribute this to their insufficient modeling of key instance information. Simple pooling operations are prone to be misled by limited slides that contain numerous instances. This problem is especially severe on the CAMELYON-16 dataset, where the proportion of significant instances is extremely small. For example, max-pooling lags behind DTFD-MIL [49] by 13.87% on AUC. Attention-based methods achieve better performance on both datasets by focusing on salient instances. In particular, the representative MIL framework DTFD-MIL [49] benefits from the further exploration of significant instances and achieves the second-best performance on both datasets (95.15% AUC on CAMELYON-16 and 93.83% AUC on TCGA). However, it also suffers from overemphasizing salient instances during training, which limits its generalization. Our proposed MHIM-MIL achieves significant performance improvement on both datasets (+1.34% AUC on CAMELYON-16 and +1.70% AUC on TCGA) by mining hard instances during training, breaking the performance bottleneck. It is worth mentioning that we validate our framework on three representative MIL models, both of which can outperform the existing MIL methods.

<table><tr><td rowspan="2">Method</td><td colspan="3">CAMELYON-16</td><td colspan="3">TCGA Lung Cancer</td></tr><tr><td>Accuracy</td><td>AUC</td><td>F1-score</td><td>Accuracy</td><td>AUC</td><td>F1-score</td></tr><tr><td>Max-pooling</td><td> $7 8 . 9 5 { \pm } 2 . 2 8 $ </td><td> $8 1 . 2 8 { \pm } 3 . 7 4$ </td><td> $7 1 . 0 6 { \pm } 2 . 5 9$ </td><td> $8 1 . 4 9 { \pm } 1 . 2 4 $ </td><td> $8 6 . 4 5 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $8 0 . 5 6 { \pm } 1 . 0 9 $ </td></tr><tr><td>Mean-pooling</td><td> $7 6 . 6 9 { \pm } 0 . 2 0 $ </td><td> $8 0 . 0 7 { \pm } 0 . 7 8 $ </td><td> $7 0 . 4 1 { \pm } 0 . 1 6$ </td><td> $8 4 . 1 4 \pm 2 . 9 7$ </td><td> $9 0 . 1 3 { \pm } 2 . 4 0 $ </td><td> $8 3 . 3 9 { \pm } 3 . 1 4 $ </td></tr><tr><td>AB-MIL [20]</td><td> $9 0 . 0 6 { \pm } 0 . 6 0$ </td><td> $9 4 . 0 0 { \pm } 0 . 8 3 $ </td><td> $8 7 . 4 0 { \pm } 1 . 0 5 $ </td><td> $8 8 . 0 3 { \pm } 2 . 1 9$ </td><td> $9 3 . 1 7 { \pm } 2 . 0 5 $ </td><td> $8 7 . 4 1 { \pm } 2 . 4 2 $ </td></tr><tr><td>DSMIL [22]</td><td> $9 0 . 1 7 { \pm } 1 . 0 2 $ </td><td> $9 4 . 5 7 { \pm } 0 . 4 0 $ </td><td> $8 7 . 6 5 { \pm } 1 . 1 8 $ </td><td> $8 8 . 3 2 { \pm } 2 . 7 0 $ </td><td> $9 3 . 7 1 { \pm } 1 . 8 2 $ </td><td> $8 7 . 9 0 { \pm } 2 . 5 0 $ </td></tr><tr><td>CLAM-SB [26]</td><td> $9 0 . 3 1 { \pm } 0 . 1 2 $ </td><td> $9 4 . 6 5 { \pm } 0 . 3 0 $ </td><td> $8 7 . 8 9 { \pm } 0 . 5 9 $ </td><td> $8 7 . 7 4 { \pm } 2 . 2 2$ </td><td> $9 3 . 6 7 { \pm } 1 . 6 4 $ </td><td> $8 7 . 3 6 { \pm } 2 . 2 4 $ </td></tr><tr><td>CLAM-MB [26]</td><td> $9 0 . 1 4 { \pm } 0 . 8 5$ </td><td> $9 4 . 7 0 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $8 8 . 1 0 { \pm } 0 . 6 3 $ </td><td> $8 8 . 7 3 { \pm } 1 . 6 2 $ </td><td> $9 3 . 6 9 { \pm } 0 . 5 4 $ </td><td> $8 8 . 2 8 { \pm } 1 . 5 8 $ </td></tr><tr><td>TransMIL [31]</td><td> $8 9 . 2 2 { \pm } 2 . 3 2 $ </td><td> $9 3 . 5 1 { \pm } 2 . 1 3 $ </td><td> $8 5 . 1 0 { \pm } 4 . 3 3 $ </td><td> $8 7 . 0 8 { \pm } 1 . 9 7 $ </td><td> $9 2 . 5 1 { \pm } 1 . 7 6 $ </td><td> $8 6 . 4 0 { \scriptstyle \pm 2 . 0 8 }$ </td></tr><tr><td>DTFD-MIL [49]</td><td> $9 0 . 2 2 { \pm } 0 . 3 6 $ </td><td> $9 5 . 1 5 { \pm } 0 . 1 4$ </td><td> $8 7 . 6 2 { \pm } 0 . 5 9 $ </td><td> $8 8 . 2 3 { \pm } 2 . 1 2 $ </td><td> $9 3 . 8 3 { \pm } 1 . 3 9 $ </td><td> $8 7 . 7 1 { \pm } 2 . 0 4 $ </td></tr><tr><td>MHIM-MIL (AB-MIL)</td><td> $9 1 . 8 1 { \pm } 0 . 8 2 $ </td><td> $9 6 . 1 4 { \pm } 0 . 5 2 $ </td><td> $8 9 . 9 4 { \pm } 0 . 7 0 $ </td><td> $8 9 . 6 4 \pm 2 . 2 5 $ </td><td> $9 4 . 9 7 { \pm } 1 . 7 2 $ </td><td> $8 9 . 3 1 { \pm } 2 . 1 9 $ </td></tr><tr><td>MHIM-MIL (TransMIL)</td><td> $9 1 . 9 8 { \pm } 0 . 8 9$ </td><td> $\mathbf { 9 6 . 4 9 } \pm \mathbf { 0 . 4 8 }$ </td><td> $9 0 . 1 3 { \pm } 1 . 0 8 $ </td><td> $\mathbf { 9 0 . 0 2 } \pm 2 . 5 \mathbf { 9 }$ </td><td> $9 4 . 8 7 { \pm } 2 . 1 7 $ </td><td> $8 9 . 6 5 { \pm } 2 . 6 3 $ </td></tr><tr><td>MHIM-MIL (DSMIL)</td><td> $\mathbf { 9 2 . 4 8 { \pm } 0 . 3 5 }$ </td><td> $9 6 . 4 9 { \pm } 0 . 6 5$ </td><td> $\mathbf { 9 0 . 7 5 { \scriptstyle \pm 0 . 7 3 } }$ </td><td> $8 9 . 8 3 { \pm } 3 . 3 7 $ </td><td> $\mathbf { 9 5 . 5 3 \pm 1 . 7 4 }$ </td><td> $\mathbf { 8 9 . 7 1 } { \pm } 2 . 9 2$ </td></tr></table>

Table 1: The performance of different MIL approaches on CAMELYON-16 (C16) and TCGA Lung Cancer (TCGA). The highest performance is in bold. The Accuracy and F1-score are determined by the optimal threshold.

<table><tr><td>Model</td><td>C16</td><td>TCGA</td><td>Para.</td><td>Time</td><td>Mem.</td></tr><tr><td>AB-MIL</td><td>94.00</td><td>93.17</td><td>657K</td><td>4.0s</td><td>2.4G</td></tr><tr><td>CLAM-MB</td><td>94.70</td><td>93.69</td><td>789K</td><td>4.3s</td><td>2.7G</td></tr><tr><td>DTFD-MIL</td><td>95.15</td><td>93.83</td><td>987K</td><td>5.2s</td><td>2.1G</td></tr><tr><td>MHIM-MIL</td><td>96.14</td><td>94.97</td><td>657K</td><td>4.3s</td><td>2.3G</td></tr><tr><td>TransMIL</td><td>93.51</td><td>92.51</td><td>2.67M</td><td>13.1s</td><td>10.6G</td></tr><tr><td>MHIM-MIL</td><td>96.49</td><td>94.87</td><td>2.67M</td><td>10.1s</td><td>5.5G</td></tr></table>

Table 2: Comparison of time and memory requirements of different MIL methods. We report the model size (Para.), the training time per epoch (Time), and the peak memory usage (Mem.) on the CAMELYON-16 dataset (C16).

## 4.4. Computational Cost Analysis

In this section, we report the training time and GPU memory requirements for running different MIL models on a 3090 GPU. The upper part of Table 7 compares some MIL frameworks that use AB-MIL [20] as a baseline. We observe that traditional MIL frameworks typically introduce additional parameters and reduce efficiency due to their complex structures. For example, the state-of-the-art framework DTFD-MIL [49] increases the parameter size by nearly twice (657K vs. 987K) and the training time by 30%. In contrast, MHIM-MIL achieves the most significant performance improvement with almost no extra computational cost due to the momentum teacher. Moreover, existing Transformer-based MIL methods are usually plagued by high computing costs due to their large number of parameters and self-attention operations. For instance, TransMIL [31], which first applies a pure Transformer MIL model to solve WSI classification problems, has 4× more parameters than AB-MIL, 3× longer training time, and almost 4.5× higher memory consumption. Furthermore, the extremely long input sequences in WSI classification degrade the stability of such complex structures (2.13% AUC standard deviation on C16, which is the highest among all embedding-level MIL methods). With the masked hard instance mining strategy, the MHIM-MIL framework significantly reduces the computational cost (-24% training time and -48% memory usage) and enhances its stability (0.48% AUC standard deviation on C16). More details are provided in Supplementary Material.

## 4.5. Ablation Study

## 4.5.1 Importance of the Different Components

Table 3 shows the effect of different modules in MHIM-MIL on two datasets. The baseline methods are two representative attention-based MIL methods, namely AB-MIL [20] and TransMIL [31]. First, we introduce the naive masked hard instance mining strategy, which leverages the model itself to mine hard instances during training. This strategy improves AUC by 1.86% and 2.55% for the two MIL models on CAMELYON-16 respectively, indicating that concentrating on hard instances during training can assist mainstream MIL models in building better classification boundaries. Further discussion on the masked hard instance mining strategy is presented in Section 4.5.2. Compared with the naive MHIM strategy, the third row of the table suggests that a Siamese structure [3,5,8] based on a momentum teacher is beneficial for more stable and effective mining of hard instances. We elaborate more on choosing the teacher model in Section 4.5.3. After adding consistency loss term to the objective function, our full MHIM-MIL framework achieves the best performance (96.49% AUC on CAMELYON-16 and 94 .97% AUC on TCGA). For subsequent ablation experiments, we include consistency loss by default to facilitate the optimization of our framework.

<table><tr><td rowspan="2">Module</td><td colspan="2">CAMELYON-16</td><td colspan="2">TCGA</td></tr><tr><td>AB.</td><td>Trans.</td><td>AB.</td><td>Trans.</td></tr><tr><td>Baseline</td><td>94.00</td><td>93.51</td><td>93.17</td><td>92.51</td></tr><tr><td>+MHIM</td><td>95.86</td><td>96.06</td><td>94.14</td><td>93.75</td></tr><tr><td>+MHIM+Siam.</td><td>95.82</td><td>96.24</td><td>94.55</td><td>94.13</td></tr><tr><td>+MHIM+Siam.+Con.</td><td>96.14</td><td>96.49</td><td>94.97</td><td>94.87</td></tr></table>

Table 3: The effect of different components in MHIM-MIL with two MIL models: AB-MIL (AB.) and TransMIL (Trans.). MHIM denotes the masked hard instance mining strategy. Siam. refers to the Siamese framework. Con. represents consistency loss.
<table><tr><td rowspan="2">Strategy</td><td colspan="2">CAMELYON-16</td><td colspan="2">TCGA</td></tr><tr><td>AB.</td><td>Trans.</td><td>AB.</td><td>Trans.</td></tr><tr><td>Baseline</td><td>94.00</td><td>93.51</td><td>93.17</td><td>92.51</td></tr><tr><td>HAM</td><td>95.68</td><td>95.90</td><td>93.83</td><td>94.54</td></tr><tr><td>R-HAM</td><td>96.14</td><td>95.88</td><td>94.79</td><td>94.60</td></tr><tr><td>L-HAM</td><td>95.81</td><td>96.49</td><td>94.33</td><td>94.67</td></tr><tr><td>LR-HAM</td><td>95.92</td><td>96.33</td><td>94.97</td><td>94.87</td></tr></table>

Table 4: Comparison between different masked hard instance mining strategies. The three hybrid strategies show varying performance across the benchmarks.

## 4.5.2 Impact of the Different MHIM Strategies

The masked hard instance mining strategy is the core design of our method. The main idea of this strategy is masking the most salient instances to indirectly mine hard instances to facilitate model training. Based on this idea, we devise three hybrid strategies (R-HAM, L-HAM, and LR-HAM) and present their impact in Table 4. The basic strategy, High Attention Masking (HAM), already boosts performance significantly, leading to AUC improvements of 1.68% and 2.39% for two MIL models on the CAMELYON-16 dataset, respectively. After introducing the other two strategies, different MIL models achieve performance improvements on both datasets. Specifically, AB-MIL [20] shows more significant performance gains after introducing randomness (96.14% AUC on CAMELYON-16 with R-HAM) due to its better ability to filter out redundant information, while TransMIL [31] shows the reverse trend (96.49% AUC on CAMELYON-16 with L-HAM). Furthermore, the more complex three-hybrid strategy (LR-HAM) achieves the best performance on the TCGA dataset, which has a larger proportion of positive areas and more instances. Overall, our experiments validate the effectiveness of masked hard instance mining strategy, and the diversity of proposed strategies improves its applicability to different datasets and MIL models.

<table><tr><td rowspan="2">Teacher</td><td colspan="2">CAMELYON-16</td><td colspan="2">TCGA</td></tr><tr><td>AB.</td><td>Trans.</td><td>AB.</td><td>Trans.</td></tr><tr><td>Baseline</td><td>94.00</td><td>93.51</td><td>93.17</td><td>92.51</td></tr><tr><td>Student copy</td><td>95.84</td><td>95.86</td><td>93.68</td><td>93.45</td></tr><tr><td>Init.</td><td>95.88</td><td>96.12</td><td>94.66</td><td>94.15</td></tr><tr><td>Momentum</td><td>95.96</td><td>96.11</td><td>94.65</td><td>94.45</td></tr><tr><td>Init.+Momentum</td><td>96.14</td><td>96.49</td><td>94.97</td><td>94.87</td></tr></table>

![](images/e63d8e80f3a544403d2385c7837e23d57b1ef6d966db1be83b2afc184f46c6a4.jpg)  
Table 5: Comparison of different types of teachers. Momentum denotes the teacher is updated by EMA strategy. Init. indicates the initialization of the teacher with pretrained parameters. The bottom figure compares the stability of the momentum teacher and the non-batch gradient updated student during training.

## 4.5.3 Impact of the Choice of Teacher Network

In MHIM-MIL, we employ a Teacher model to mine hard instances and facilitate training of the Student model. In Table 5, we comprehensively investigate the effects of various choices of Teacher network. First, we utilize a single-model structure, which treats the Student model as the Teacher. The student conducts masked hard instance mining prior to training. Due to the non-batch gradient update, the unstable performance of the Student model makes the strategy susceptible to noise, so the performance is not optimal. Second, we adopt a momentum teacher, which shares the same network structure as the Student model and is updated with the EMA strategy. This updating strategy enhanced the stability of momentum teachers, as shown in the figure below, and enabled MHIM-MIL to achieve 0.97% and 1.00% performance improvement in TCGA under the two MIL models, respectively. With proper initialization, the momentum teacher achieves the best performance. However, a fixed initialization teacher fails to learn new knowledge, which emphasizes the significance of iterative optimization.

![](images/d5a6f174b35dc2fe3660356bc44cb16e31336aa4133d04a547a096b2cbbde837.jpg)  
Figure 4: Patch visualization produced by AB-MIL [20] (baseline) and MHIM-MIL. The blue lines outline the tumor regions. The brighter patch indicates higher attention scores. The cyan colors indicate high probabilities of being tumor for the corresponding locations. Ideally, the cyan patches should cover only the area within the blue lines. We show that focusing only on more salient regions reduces the generalization ability of the model and that hard instances can provide useful information for more accurate and comprehensive judgments.

## 4.6. Visualization

To more intuitively understand the effect of the masked hard instance mining, we visualize the attention scores (bright patch) and tumor probabilities (cyan patch) of patches produced by AB-MIL and MHIM-MIL, as illustrated in Figure 4. Here, MHIM-MIL employs AB-MIL as its baseline model. We note that attention scores only indicate the regions of interest of models and are infeasible to reflect tumor probabilities [22, 49]. First, as shown in Figure 4, AB-MIL often assigns high tumor probabilities to patches in non-tumor areas. We attribute this phenomenon to the low generalization capability of conventional attention-based MIL models, which tend to focus only on salient regions during training. In contrast, MHIM-MIL trained with hard instances shows a much better generalization ability than the baseline model for noise robustness (rows 2 and 3 on the right) and for precise detection of challenging subtle tumor areas (row 3 on the left). More significantly, we find that focusing only on tumor areas leads to missing most of them, expanding the view to include some “irrelevant areas” enables the model to make more complete judgments (rows 1 and 2 on the left). This phenomenon demonstrates how hard instances provide more useful information to help the model make more accurate and comprehensive judgments. We provide more details and an indepth analysis of this patch visualization in Supplementary Material.

## 5. Conclusion

This paper rethinks the impact of salient instances for MIL-based WSI classification algorithms. We demonstrate that attention-based MIL methods excessively prioritizing salient instances harm the generalization ability of the model. To address this issue, we have proposed several masked hard instance mining strategies that mask out salient patches and encourage the model to attend to informative regions for better discriminative learning. Through qualitative analysis, we have demonstrated that these strategies effectively alleviate the under-fitting problem of general AB-MIL to hard instances. We have also developed the MHIM-MIL framework that leverages momentum teacher and consistency loss to further enhance hard instance mining. Our experimental results demonstrate the superiority and generality of the MHIM-MIL framework over other latest methods. In future work, we plan to devise a more precise localization scheme for hard instances that can facilitate model training and convergence.

## 6. Acknowledgement

Reported research is partly supported by the National Natural Science Foundation of China under Grant 62176030, and the Natural Science Foundation of Chongqing under Grant cstc2021jcyj-msxmX0568.

## References

[1] Ejaz Ahmed, Michael Jones, and Tim K Marks. An improved deep learning architecture for person re-identification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3908–3916, 2015. 2

[2] Babak Ehteshami Bejnordi, Mitko Veta, Paul Johannes Van Diest, Bram Van Ginneken, Nico Karssemeijer, Geert Litjens, Jeroen AWM Van Der Laak, Meyke Hermsen, Quirine F Manson, Maschenka Balkenhol, et al. Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. JAMA, 318(22):2199–2210, 2017. 5

[3] Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard Sackinger, and Roopak Shah. Signature verification using¨ a” siamese” time delay neural network. Advances in neural information processing systems, 6, 1993. 2, 7

[4] Gabriele Campanella, Matthew G Hanna, Luke Geneslaw, Allen Miraflor, Vitor Werneck Krauss Silva, Klaus J Busam, Edi Brogi, Victor E Reuter, David S Klimstra, and Thomas J Fuchs. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nature Medicine, 25(8):1301–1309, 2019. 2

[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou,´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 7, 13

[6] Richard J Chen, Chengkuan Chen, Yicong Li, Tiffany Y Chen, Andrew D Trister, Rahul G Krishnan, and Faisal Mahmood. Scaling vision transformers to gigapixel images via hierarchical self-supervised learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16144–16155, 2022. 1, 5

[7] Weihua Chen, Xiaotang Chen, Jianguo Zhang, and Kaiqi Huang. Beyond triplet loss: a deep quadruplet network for person re-identification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 403–412, 2017. 3

[8] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15750–15758, 2021. 2, 7

[9] Philip Chikontwe, Miguel Luna, Myeongkyun Kang, Kyung Soo Hong, June Hong Ahn, and Sang Hyun Park. Dual attention multiple instance learning with unsupervised

complementary loss for covid-19 screening. Medical Image Analysis, 72:102105, 2021. 1, 2

[10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 15

[11] Thomas G Dietterich, Richard H Lathrop, and Tomas´ Lozano-Perez. Solving the multiple instance problem with´ axis-parallel rectangles. Artificial intelligence, 89(1-2):31– 71, 1997. 1, 2

[12] Thomas G Dietterich, Richard H Lathrop, and Tomas´ Lozano-Perez. Solving the multiple instance problem with´ axis-parallel rectangles. Artificial Intelligence, 89(1-2):31– 71, 1997.

[13] Qi Dong, Shaogang Gong, and Xiatian Zhu. Class rectification hard mining for imbalanced deep learning. In Proceedings of the IEEE international conference on computer vision, pages 1851–1860, 2017. 3

[14] Ji Feng and Zhi-Hua Zhou. Deep miml network. In Proceedings of the AAAI conference on artificial intelligence, volume 31, 2017. 2

[15] Ju He, Jie-Neng Chen, Shuai Liu, Adam Kortylewski, Cheng Yang, Yutong Bai, and Changhu Wang. Transfg: A transformer architecture for fine-grained recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 852–860, 2022. 14

[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 15

[17] Marti A. Hearst, Susan T Dumais, Edgar Osuna, John Platt, and Bernhard Scholkopf. Support vector machines. IEEE Intelligent Systems and their applications, 13(4):18–28, 1998. 2

[18] Alexander Hermans, Lucas Beyer, and Bastian Leibe. In defense of the triplet loss for person re-identification. arXiv preprint arXiv:1703.07737, 2017. 3

[19] Le Hou, Dimitris Samaras, Tahsin M Kurc, Yi Gao, James E Davis, and Joel H Saltz. Patch-based convolutional neural network for whole slide tissue image classification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2424–2433, 2016. 2

[20] Maximilian Ilse, Jakub Tomczak, and Max Welling. Attention-based deep multiple instance learning. In International conference on machine learning, pages 2127–2136. PMLR, 2018. 1, 2, 3, 5, 6, 7, 8, 11, 12

[21] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 15

[22] Bin Li, Yin Li, and Kevin W Eliceiri. Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. In CVPR, pages 14318–14328, 2021. 1, 2, 3, 5, 6, 8

[23] Hang Li, Fan Yang, Yu Zhao, Xiaohan Xing, Jun Zhang, Mingxuan Gao, Junzhou Huang, Liansheng Wang, and Jianhua Yao. Dt-mil: Deformable transformer for multiinstance learning on histopathological image. In International Conference on Medical Image Computing and

Computer-Assisted Intervention, pages 206–216. Springer, 2021. 1

[24] Meng Li, Lin Wu, Arnold Wiliem, Kun Zhao, Teng Zhang, and Brian Lovell. Deep instance-level hard negative mining model for histopathology images. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part I 22, pages 514–522. Springer, 2019. 3

[25] Ming Y Lu, Tiffany Y Chen, Drew FK Williamson, Melissa Zhao, Maha Shady, Jana Lipkova, and Faisal Mahmood. Aibased pathology predicts origins for cancers of unknown primary. Nature, 594(7861):106–110, 2021. 1

[26] Ming Y Lu, Drew FK Williamson, Tiffany Y Chen, Richard J Chen, Matteo Barbieri, and Faisal Mahmood. Data-efficient and weakly supervised computational pathology on wholeslide images. Nature Biomedical Engineering, 5(6):555– 570, 2021. 1, 2, 3, 5, 6, 15

[27] Siyamalan Manivannan, Caroline Cobb, Stephen Burgess, and Emanuele Trucco. Subcategory classifiers for multipleinstance learning and its application to retinal nerve fiber layer visibility classification. IEEE Transactions on Medical Imaging, 36(5):1140–1150, 2017. 2

[28] Oded Maron and Tomas Lozano-P´ erez. A framework for´ multiple-instance learning. Advances in neural information processing systems, 10, 1997. 1

[29] Hans Pinckaers, Bram Van Ginneken, and Geert Litjens. Streaming convolutional neural networks for end-to-end learning with multi-megapixel images. IEEE transactions on pattern analysis and machine intelligence, 44(3):1581– 1590, 2020. 1

[30] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 815–823, 2015. 2

[31] Zhuchen Shao, Hao Bian, Yang Chen, Yifeng Wang, Jian Zhang, Xiangyang Ji, et al. Transmil: Transformer based correlated multiple instance learning for whole slide image classification. NeurIPS, 34, 2021. 1, 2, 3, 5, 6, 7, 12, 13, 15

[32] Yash Sharma, Aman Shrivastava, Lubaina Ehsan, Christopher A Moskaluk, Sana Syed, and Donald E Brown. Clusterto-conquer: A framework for end-to-end multi-instance learning for whole slide image classification. arXiv preprint arXiv:2103.10626, 2021. 2

[33] Hao Sheng, Yanwei Zheng, Wei Ke, Dongxiao Yu, Xiuzhen Cheng, Weifeng Lyu, and Zhang Xiong. Mining hard samples globally and efficiently for person reidentification. IEEE Internet ofThings Journal, 7(10):9611–9622, 2020. 2

[34] Abhinav Shrivastava, Abhinav Gupta, and Ross Girshick. Training region-based object detectors with online hard example mining. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 761–769, 2016. 2

[35] Kihyuk Sohn. Improved deep metric learning with multiclass n-pair loss objective. Advances in neural information processing systems, 29, 2016. 2

[36] Chetan L Srinidhi, Ozan Ciga, and Anne L Martel. Deep neural network models for computational histopathology: A survey. Medical Image Analysis, 67:101813, 2021. 1

[37] Yumin Suh, Bohyung Han, Wonsik Kim, and Kyoung Mu Lee. Stochastic class-based hard example mining for deep metric learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7251–7259, 2019. 2, 3

[38] Han Sun, Zhiyuan Chen, Shiyang Yan, and Lin Xu. Mvp matching: A maximum-value perfect matching for mining hard samples, with application to person re-identification. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6737–6747, 2019. 2, 3

[39] Zichang Tan, Ajian Liu, Jun Wan, Hao Liu, Zhen Lei, Guodong Guo, and Stan Z Li. Cross-batch hard example mining with pseudo large batch for id vs. spot face recognition. IEEE Transactions on Image Processing, 31:3224– 3235, 2022. 2, 3

[40] David Tellez, Geert Litjens, Jeroen van der Laak, and Francesco Ciompi. Neural image compression for gigapixel histopathology image analysis. IEEE transactions on pattern analysis and machine intelligence, 43(2):567–578, 2019.

[41] Tong Tong, Robin Wolz, Qinquan Gao, Ricardo Guerrero, Joseph V Hajnal, Daniel Rueckert, Alzheimer’s Disease Neuroimaging Initiative, et al. Multiple instance learning for classification of dementia in brain mri. Medical Image Analysis, 18(5):808–818, 2014. 2

[42] Keze Wang, Xiaopeng Yan, Dongyu Zhang, Lei Zhang, and Liang Lin. Towards human-machine cooperation: Selfsupervised sample mining for object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1605–1613, 2018. 2

[43] Xi Wang, Hao Chen, Caixia Gan, Huangjing Lin, Qi Dou, Efstratios Tsougenis, Qitao Huang, Muyan Cai, and Pheng-Ann Heng. Weakly supervised deep learning for whole slide lung cancer image analysis. IEEE transactions on cybernetics, 50(9):3950–3962, 2019. 2

[44] Xinggang Wang, Yongluan Yan, Peng Tang, Xiang Bai, and Wenyu Liu. Revisiting multiple instance neural networks. Pattern Recognition, 74:15–24, 2018. 2

[45] Yunan Wu, Arne Schmidt, Enrique Hernandez-S ´ anchez,´ Rafael Molina, and Aggelos K Katsaggelos. Combining attention-based multiple instance learning and gaussian processes for ct hemorrhage detection. In MICCAI, pages 582– 591. Springer, 2021. 2

[46] Gang Xu, Zhigang Song, Zhuo Sun, Calvin Ku, Zhe Yang, Cancheng Liu, Shuhao Wang, Jianpeng Ma, and Wei Xu. Camel: A weakly supervised learning framework for histopathology image segmentation. In Proceedings of the IEEE/CVF International Conference on computer vision, pages 10682–10691, 2019. 1, 2, 3

[47] Lin Xu, Han Sun, and Yuai Liu. Learning with batch-wise optimal transport loss for 3d shape recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3333–3342, 2019. 3

[48] Yan Xu, Jun-Yan Zhu, I Eric, Chao Chang, Maode Lai, and Zhuowen Tu. Weakly supervised histopathology cancer im-

age segmentation and classification. Medical Image Analysis, 18(3):591–604, 2014. 2

[49] Hongrun Zhang, Yanda Meng, Yitian Zhao, Yihong Qiao, Xiaoyun Yang, Sarah E Coupland, and Yalin Zheng. Dtfdmil: Double-tier feature distillation multiple instance learning for histopathology whole slide image classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18802–18812, 2022. 1, 2, 3, 5, 6, 8, 15

[50] Xiaoxian Zhang, Sheng Huang, Yi Zhang, Xiaohong Zhang, Mingchen Gao, and Liu Chen. Dual space multiple instance representative learning for medical image classification. In 33rd British Machine Vision Conference 2022, BMVC 2022, London, UK, November 21-24, 2022. BMVA Press, 2022. 2, 5

[51] Yu Zhao, Zhenyu Lin, Kai Sun, Yidan Zhang, Junzhou Huang, Liansheng Wang, and Jianhua Yao. Setmil: spatial encoding transformer-based multiple instance learning for pathological image analysis. In Medical Image Computing and ComputerAssisted Intervention–MICCAI 2022: 25th International Conference, Singapore, September 18–22, 2022, Proceedings, Part II, pages 66–76. Springer, 2022. 1

## A. Additional Visualization

Here, we attempt to further analyze the impact of Masked Hard Instance Mining (Masked HIM) on WSI classification algorithms based on multiple instance learning. As shown in Figure 5, we visualize the masked instances (middle column), which we call the mined hard instances, to illustrate the relationship between the instance-level tumor prediction probability (cyan patch) and model attention (bright patch) before and after Masked HIM training.

First, thanks to the outstanding saliency patch mining ability of traditional attention-based MIL models, Masked HIM can effectively mask out the most salient regions to indirectly mine hard instances while using random masking to mitigate over-fitting problems. Moreover, as shown in the Figure 6, this discriminatory power improves gradually during the training. To ensure that the instance sequence after masking still retains key instance information related to the slide category, we propose a randomization technique, which will be explained in detail in the following subsection. Second, contrary to intuition, MIL models do not lose their discriminative power for key regions after masking out the most salient instances, due to the MHIM-MIL framework. Instead, they achieve a significant improvement. Figure 5 strongly proves that focusing only on salient instances during the training stage damages the discriminative power of MIL models, and verifies the huge help of hard instances for MIL model training. Moreover, we visualize the instance patch attention after softmax, which can be regarded as the contribution to the final bag embedding. We find that although traditional MIL models seem to pay attention to salient regions, they do not make reasonable use of this part of the information. They ignore most features and extremely focus on individual features in the feature aggregation process, damaging model discriminativeness. In contrast, MIL models trained with Masked HIM seem to put more attention on more “irrelevant regions”, but better utilize key region features to generate higher quality bag features and improve model performance.

![](images/bac10e6c8b82a8807d8afc701a72173dc1ec03756d6679a4410d0dc9baf65896.jpg)  
Figure 5: Comparison of patch visualization produced by AB-MIL [20] (baseline) and MHIM-MIL. The blue lines outline the tumor regions. The brighter patch indicates higher attention scores. The cyan colors indicate high probabilities of being tumor for the corresponding locations. Ideally, the cyan patches should cover only the area within the blue lines. In the middle column, the dark patches denote masked instances.

Figure 13 shows more patch visualizations on the CAMELYON-16 dataset.

## B. Additional Quantitative Experiments

## B.1. More on Masked Hard Instance Mining

Discussion on Mask Ratio. We explored how various mask ratios affect MHIM-MIL training in this section. We fixed other ratios $( \beta _ { r } , \beta _ { l } )$ and varied high attention mask ratio $\beta _ { h }$ alone in Figure 7. We fixed $\beta _ { h }$ and changed different $\beta _ { r }$ and $\beta _ { l }$ in Table 6. Our findings are: 1) A low $\beta _ { h }$ reduces the difficulty of mined instances, thereby diminishing the overall model performance. Moreover, the randomized trick ensures that the model training does not collapse even at high $\beta _ { h }$ . More details are provided in the following section. 2) Compared to AB-MIL [20], TransMIL [31] has lower discriminative power for salient instances. This is why TransMIL needs a bigger $\beta _ { h }$ . 3) MHIM-MIL training is less sensitive to $\beta _ { r }$ and $\beta _ { l }$ than to $\beta _ { h }$ . However, choosing an appropriate mask ratio is still crucial for optimal performance. Specifically, we observed that combining three strategies on the CAMELYON-16 dataset decreases classification performance. We attribute this to excessive instance masking losing important information on the CAMELYON-16 dataset.

![](images/7567be05ce164a5fa8891aced7941eaa830aff2be43adce06b1e46d9652e67c5.jpg)  
Figure 6: Patch visualization during iteration process.

![](images/892227ee92fc901d95730def422455766666a10642c55d06e3a4c0f6aefbd099.jpg)

![](images/095b63230f6e63cba4a0ec39c8478c6438d3530a16994ca8dad7110f0c528a29.jpg)  
Figure 7: The performances of MHIM-MIL under different high attention mask ratio $\beta _ { h }$

Computational Cost. Here, we comprehensively discuss the impact of different MHIM strategies on the computational cost of model training. Table 7 shows the efficiency gains brought by large-scale low-attention masking and random-attention masking. This is especially significant for TransMIL [31], a baseline with both spatial and temporal complexity quadratic to the number of instances. Large-scale masking greatly reduces the input of the student model, thereby reducing memory and time consumption. Although the input of the teacher model is still full length, due to the application of momentum teacher, it hardly introduces extra training cost. In addition, we also find that mixing multiple strategies further reduces the number of instances but also introduces additional computation, which is more obvious on AB-MIL [20] baseline.

<table><tr><td>random ratio</td><td>low ratio</td><td>AUC</td></tr><tr><td>AB-MIL</td><td></td><td></td></tr><tr><td>60%</td><td>20%</td><td>94.57</td></tr><tr><td>70%</td><td>10%</td><td>94.65</td></tr><tr><td>70%</td><td>20%</td><td>94.97</td></tr><tr><td>70%</td><td>30%</td><td>94.55</td></tr><tr><td>80%</td><td>20%</td><td>94.49</td></tr><tr><td>TransMIL</td><td></td><td></td></tr><tr><td>50%</td><td>20%</td><td>94.60</td></tr><tr><td>60%</td><td>10%</td><td>93.97</td></tr><tr><td>60%</td><td>20%</td><td>94.87</td></tr><tr><td>60%</td><td>30%</td><td>94.37</td></tr><tr><td>70%</td><td>20%</td><td>94.60</td></tr><tr><td>(a) TCGA Lung Cancer dataset</td><td></td><td></td></tr><tr><td>random ratio</td><td>low ratio</td><td>AUC</td></tr><tr><td>AB-MIL</td><td></td><td></td></tr><tr><td>40%</td><td>0%</td><td>95.90</td></tr><tr><td>50%</td><td>0%</td><td>96.14</td></tr><tr><td>50%</td><td>20%</td><td>95.92</td></tr><tr><td>60%</td><td>0%</td><td>96.13</td></tr><tr><td>TransMIL</td><td></td><td></td></tr><tr><td>0%</td><td>70%</td><td>96.36</td></tr><tr><td>0%</td><td>80%</td><td>96.49</td></tr><tr><td>20%</td><td>80%</td><td>96.33</td></tr><tr><td></td><td></td><td></td></tr><tr><td>0%</td><td>90%</td><td>96.10</td></tr></table>

(b) CAMELYON-16 dataset  
Table $6 { : }$ Comparison of different random attention mask ratio $\beta _ { r }$ and low attention mask ratio $\beta _ { l }$ on both datasets.

Mask Ratio Decay. The discriminative ability of the model improves and stabilizes as training goes on. We follow the learning rate decay idea and tune $\beta _ { h }$ based on training progress to prevent a high initial ratio from hurting later training. We name this technique mask ratio decay and adopt a classic cosine decay function to regulate decay speed. Table 8 demonstrates that this trick significantly boosts performance. We note that we apply the decay strategy only to $\beta _ { h }$ while maintaining initial values for the other two ratios during training.

Randomly High Attention Masking. MHIM faces a major challenge: it may mask all key information and turn into “error instance mining”. We apply the Randomly High Attention Masking technique to address this issue and make sure that mined hard instances include key instance information for the slide category. Figure 8 illustrates our approach: we select instances with the highest $2 \times \beta _ { h }$ % attention scores as candidate states and randomly mask half of them to keep some key information. Table 9 demonstrates this technique suffers from low training difficulty in the TCGA dataset, where the tumor area ratio is high (typically over 40% [31]), and impairs the discriminability of the model. On the other hand, this technique performs well on the CAMELYON-16 dataset, indicating that it can preserve key information in original instances.

<table><tr><td>Model</td><td>C16</td><td>TCGA</td><td>Para.</td><td>Time</td><td>Mem.</td></tr><tr><td>AB-MIL HAM R-HAM</td><td>94.00 95.68 96.14</td><td>93.17 93.83 94.79</td><td>657K 657K 657K</td><td>4.0s 4.0s 4.3s</td><td>2.4G 2.7G 2.3G</td></tr><tr><td>L-HAM LR-HAM TransMIL</td><td>95.81 95.92 93.51</td><td>94.33 94.97 92.51</td><td>657K 657K 2.67M</td><td>4.2s 4.4s 13.1s</td><td>2.3G 2.2G 10.6G</td></tr><tr><td>HAM R-HAM L-HAM</td><td>95.90 95.88 96.49</td><td>94.54 94.60 94.67</td><td>2.67M 2.67M 2.67M</td><td>15.9s 10.3s 10.1s</td><td>10.3G 5.5G 5.5G</td></tr></table>

Table 7: Comparison of time and memory requirements of different masked hard instance mining strategies. We report the model size (Para.), the training time per epoch (Time), and the peak memory usage (Mem.) on the CAMELYON-16 dataset (C16).
<table><tr><td rowspan="2">Strategy</td><td colspan="2">CAMELYON-16</td><td colspan="2">TCGA</td></tr><tr><td>AB.</td><td>Trans.</td><td>AB.</td><td>Trans.</td></tr><tr><td>βh%</td><td>96.04</td><td>96.07</td><td>94.34</td><td>94.56</td></tr><tr><td>βh% → 0%</td><td>96.14</td><td>96.49</td><td>94.97</td><td>94.87</td></tr></table>

Table 8: Comparison results of applying high attention mask ratio decay.

![](images/817e6f78d6e061349e4d8afb4d7dc0dce39a6375b3d31ffe98cd9cb3b05bf40e.jpg)  
Figure 8: Illustration of Randomly High Attention Masking (Randomly HAM).

## B.2. Initialization of Student Network

MIL models typically employ a fully connected layer to project original 1024-dimensional instance features into 512 dimensions as final instance representation. In MHIM-MIL, we initialize the fully connected layer of the student network with pre-trained parameters to reduce collapse risk from the Siamese structure. [5] elaborates on more details about collapse risk. Figure 9 illustrates how this initialization affects teacher model performance. An uninitialized student model has slow initial training which drags down teacher model performance and harms the iterative optimization of the framework. The upper part of Table 10 displays a large margin in final student model performance with and without this initialization. Moreover, we applied the same initialization to mainstream MIL models to investigate if this initialization boosts performance by aiding Siamese structure optimization. The upper part of Table 10 reveals that this initialization does not noticeably enhance the performance of existing mainstream MIL models and sometimes lowers it. Our experiments confirm that initializing the first fully connected layer of student facilitate the iterative optimization of the MHIM-MIL framework instead of being a universal trick for increasing MIL model performance.

<table><tr><td rowspan="2">Strategy</td><td colspan="2">CAMELYON-16</td><td colspan="2">TCGA</td></tr><tr><td>AB.</td><td>Trans.</td><td>AB.</td><td>Trans.</td></tr><tr><td>w/o Ran. HAM</td><td>95.71</td><td>96.37</td><td>94.97</td><td>94.87</td></tr><tr><td>w/ Ran. HAM</td><td>96.14</td><td>96.49</td><td>94.52</td><td>94.17</td></tr></table>

Table 9: Comparison results of applying randomly high attention masking (Ran. HAM).

<table><tr><td>Model</td><td>CAMELYON-16</td><td>TCGA</td></tr><tr><td>AB-MIL w/ init</td><td>93.98 (-0.02)</td><td>92.75 (-0.42)</td></tr><tr><td>MHIM-MIL w/ init</td><td>96.14 (+0.63)</td><td>94.97 (+0.49)</td></tr><tr><td>TransMIL w/ init</td><td>94.22 (+0.71)</td><td>93.36 (+0.85)</td></tr><tr><td>MHIM-MIL w/ init</td><td>96.49 (+0.90)</td><td>94.87 (+0.95)</td></tr><tr><td>w/ init</td><td></td><td></td></tr><tr><td>CLAM-SB</td><td>94.53 (-0.12)</td><td>93.43 (-0.24)</td></tr><tr><td>DSMIL</td><td>94.96 (+0.39)</td><td>93.93 (+0.22)</td></tr><tr><td>DTFD-MIL</td><td>95.23 (+0.08)</td><td>93.80 (-0.03)</td></tr></table>

Table 10: Comparison results of different initialized MIL models.

## B.3. Transformer Attention

Transformer typically consists of a multi-layer multihead structure where each head within each layer generates independent attention scores. Thus, extracting the most effective attention score among them is very challenging. In particular, the baseline model TransMIL [31] comprises two layers with eight heads per layer. We separately examined the effect of attention scores from different layers and various multi-head fusion strategies. The upper part of Table 11 demonstrates the advantage of attention scores from the first layer over those from the final layer. We attribute this to the first layer producing more accurate attention scores for identifying hard instances. This is because the multihead self-attention (MSA) operation modifies original features which causes a large deviation between hard instances mined by the last layer and the actual situation, while only the input of the first layer is the original instance features.

![](images/ebd7137b2bec415d9d1c32f602f9d995e756c7e03be74581ea57292660ac27f9.jpg)  
Figure 9: Performance comparison of teacher models under initialized or uninitialized student networks.

![](images/02b8d193858513328850b90c43ac4f8b929f85874403967486337ec7167ad71d.jpg)  
Figure 10: Illustration of averaging and voting multi-head attention fusion strategy.

Additionally, prior work [15] equalizes the contribution of each head and distributes the total mask count among different heads, which is called “averaging”. However, this strategy fails to prevent the effect of the invalid heads on MHIM. As shown in Figure 11, some heads of TransMIL lack discrimination ability for instances and produce identical attention scores which we term as invalid heads. Invalid heads dilute localization accuracy for hard instances under averaging strategy and impair the training of MHIM-MIL. To mitigate this issue, we suggest a voting strategy that employs majority rule to eliminate noise from invalid heads, as shown in Figure 10. The lower part of Table 11 proves the effectiveness of this strategy.

![](images/674d535f4fd15b5517fd9ebb621566dc4c95dee86bfa41a68b3681c928ea6a50.jpg)  
Figure 11: Attention visualization of different heads in TransMIL first layer.

<table><tr><td>case</td><td>CAMELYON-16</td><td>TCGA</td></tr><tr><td>first</td><td>96.49</td><td>94.87</td></tr><tr><td>last</td><td>95.58 (-0.91)</td><td>93.90 (-0.97)</td></tr><tr><td>averaging</td><td>96.38 (-0.11)</td><td>94.40 (-0.47)</td></tr><tr><td>voting</td><td>96.49</td><td>94.87</td></tr></table>

Table 11: Comparison results of variants of TransMIL attention.

## B.4. Discussion on Hyperparameter

Here, we provide a systematic discussion of an important hyperparameter α in our framework. It balances the impact of self-supervised and fully supervised information during model training. Figure 12 demonstrates that α affects the training of both models consistently, with values that are either too high or too low resulting in biased training. Particularly, when α is too high, it impairs the positive effect of slide labels on model learning. This effect is more pronounced on the CAMELYON-16 dataset, as the model frequently misclassifies some challenging slides, requiring supervision from slide labels.

![](images/89d96508f71c4cb082628a2bb7f179825eb56cd361d37baf2dfefe16e85553f8.jpg)

![](images/39db99dc2457b46333f2be7d8874cb27eb77867ed21239a6d5546eeea8473e0e.jpg)  
Figure 12: The performances of MHIM-MIL under different loss scaling factors α.

## C. Data Pre-processing

Following prior works [26, 31, 49], we crop each WSI into a series of non-overlapping patches of size $2 5 6 \times 2 5 6$ at 20X magnification and discard the background region, including holes, as in CLAM [26]. After pre-processing, we obtain a total of 3.6M patches from the CAMELYON-16 dataset, with an average of about 9000 patches per bag, and 10.8M patches from the TCGA Lung Cancer dataset, with an average of about 10300 patches per bag.

## D. Implementation Details

Following [26,31,49], we use the ResNet-50 model [16] pretrained with ImageNet [10] as the backbone network to extract an initial feature vector from each patch, which has a dimension of 1024. The last convolutional module of the ResNet-50 is removed, and a global average pooling is applied to the final feature maps to generate the initial feature vector. The initial feature vector is then reduced to a 512- dimensional feature vector by one fully-connected layer. The momentum rate of EMA is 0.9999 and the temperature of consistency loss is 0.1. An Adam optimizer [21] with learning rate of $2 \times 1 0 ^ { - 4 }$ and weight decay of $1 \times 1 0 ^ { - 5 }$ is used for the model training. The Cosine strategy is adopted to adjust the learning rate. All the models are trained for 200 epochs with an early-stopping strategy. The patience of CAMELYON-16 and TCGA Lung Cancer are 30 and 20, respectively. We do not use any trick to improve the model performance, such as gradient cropping or gradient accumulation. The batch size is set to 1. All the experiments were conducted with an NVIDIA RTX3090 GPU.

## E. Pseudocode

We present the PyTorch-style pseudocode for the training scheme of MHIM-MIL in Algorithm 1.

## F. Limitation

In this paper, we propose a Masked Hard Instance Mining MIL framework to indirectly mine hard instances in the absence of instance supervision information. Although this strategy can effectively alleviate the over-reliance problem of traditional MIL models on salient instances, it is also challenging to accurately assess the difficulty level of instances and mine the most helpful hard instances for training. Compared with traditional hard sample mining strategies based on supervision information, this sub-optimal and rough strategy affects the convergence speed and discriminability of the model. In future work, we will focus on how to accurately evaluate instance difficulty level in the absence of complete supervision and use the most beneficial instances to facilitate model training.

## G. Code and Data Availability

The source code of our project will be uploaded at https://github.com/DearCaat/MHIM-MIL.

CAMELYON-16 dataset can be found at https://camelyon16.grand-challenge.org.

TCGA Lung Cancer dataset can be found at https://portal.gdc.cancer.gov.

The script of slide pre-processing and patching can be found at https://github.com/mahmoodlab/CLAM.

Algorithm 1: PyTorch-style pseudocode for MHIM-MIL training scheme   
<sup>#</sup><sub>#</sub> f t, f s: teacher and student networks   
f p: the pretrained network   
# mrh: high attention mask ratio   
# mrl: low attention mask ratio   
# mrr: random attention mask ratio   
<sup>#</sup><sub>#</sub> m: momentum rates   
tp: temperatures   
# a: consistency loss scaling factor   
# initialize   
f t.params = f p.params   
f s.proj head.params = f p.proj head.params   
# teacher network not introduces any parameter   
f t = f t.eval()   
def mask fn(attn,mask ratio,largest):   
# sort attention score and get the topk index   
attn = sort(attn)   
topk ids = topk(attn,k=int(mask ratio<sub>\*</sub>attn.length),largest=largest)   
# init vote matrix   
vote = 0   
# voting and counting   
vote[topk ids] = 1   
vote = sum(vote)   
# get mask index   
mask ids = topk(vote,k=int(mask ratio attn.length))   
return mask ids   
for x,y in loader: # load a minibatch x,y with N slides   
# get attention scores from teacher   
,bag feats t,attn t = f t.forward(x)   
# stop gradient of teacher network   
bag feats t = bag feats t.detach()   
# get masked instance index   
# High Attention Masking   
mask h = mask fn(attn t,mrh,True)   
# Low Attention Masking   
mask l = mask fn(attn t,mrl,False)   
# Random Attention Masking   
mask r = random select(attn t,mrr)   
# Combine all index   
mask all = mask h & mask l & mask r   
# masked hard instance mining   
x hard = masking(x,mask all)   
logits s,bag feats s, = f s.forward(x hard)   
# consistency loss   
loss con = -softmax(bag feats t / tp) <sub>\*</sub> log softmax(bag feats s)   
# label prediction loss   
loss cls = CrossEntropy(logits s,y)   
loss all = loss cls + a<sub>\*</sub>loss con   
# Adam update: student network   
loss all.backward()   
update(f s.params)   
# EMA update: teacher network   
f t.params = m<sub>\*</sub>f t.params+(1-m)<sub>\*</sub>f s.params   
# high attention mask ratio decay   
CosineDecay(mrh)

![](images/1f8b24e3ec866c642b624383338777fac07a8649698ca5326d0edd682a4e0a72.jpg)  
Figure 13: More comparisons of patch visualization between AB-MIL (baseline) and MHIM-MIL. Best viewed in color.