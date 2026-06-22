# Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning

Wentao Huang<sup>1,2</sup>\* Weimin Lyu<sup>1</sup> Peiliang Lou<sup>2</sup> Qingqiao Hu<sup>1</sup> Xiaoling Hu<sup>3</sup> Shahira Abousamra<sup>4</sup> Wenchao Han<sup>2</sup> Ruifeng Guo<sup>2</sup> Jiawei Zhou<sup>1</sup> Chao Chen<sup>1</sup> Chen Wang<sup>2</sup> <sup>1</sup>Stony Brook University, NY, USA <sup>2</sup>Mayo Clinic, MN, USA <sup>3</sup>Harvard Medical School, MA, USA <sup>4</sup>Stanford University, CA, USA

## Abstract

Computational pathology has advanced rapidly in recent years, driven by domain-specific image encoders and growing interest in using vision–language models to answer natural-language questions about diseases. Yet, the core problem behind pathology question-answering remains unsolved, considering that a gigapixel slide contains far more information than necessary for a given question. Pathologists naturally navigate tissue and morphology complexity by scanning broadly, and zooming in selectively accord ing to the clinical questions. Current models, in contrast, rely on uniform patch sampling or broad attention maps, often attending equally to irrelevant regions while overlooking key visual evidence. In this work, we try to bring models closer to how humans actually examine slides. We propose a question-guided, tissue-aware, and coarse-tofine retrieval framework, HistoSelect, that consists of two key components: a group sampler that identifies questionrelevant tissue regions, followed by a patch selector that retrieves the most informative patches within those regions. By selecting only the most informative patches, our method becomes significantly more efficient: reducing visual token usage by 70% on average, while improving accuracy across three pathology QA tasks. Evaluated on 356,000 question–answer pairs, our approach outperforms existing methods and produces answers grounded in interpretable, pathologist-consistent regions. Our results suggest that bringing human-like search and attention patterns into WSI reasoning is a promising direction for building practical and reliable pathology VLMs. Code is available at https: //github.com/winston52/HistoSelect.

## 1. Introduction

Histopathology image analysis plays a critical role in cancer diagnosis and treatment planning [4, 24, 29]. A key data source in this domain is the Whole Slide Image (WSI), a giga-pixel digital scan that captures rich cellular and tissue morphology. The rapid development of computational pathology has enabled success in fundamental tasks, such as subtype classification [14, 17, 19, 31, 33, 39, 48, 50], segmentation [45, 46] and survival analysis [8, 35, 47, 51, 52]. With the emergence of multi-modal learning, more challenging tasks have been introduced. In Visual Question Answering (VQA) [7, 9, 16, 22, 38], models are required not only to predict a correct answer, but also to produce clinically trustworthy and interpretable answers.

![](images/9f7dd091146cd031cb5d97bb4d9841397de34b9af37ad5c6cdb54e8508ff793b.jpg)  
Figure 1. Illustration of our HistoSelect framework. (a) The baseline method feeds a large number of patches indiscriminately into the VLM, leading to high redundancy and question-irrelevance. (b) Our question-guided tissue-aware selection method. The question guides the model to select a relevant and sparse subset of informative patches, which are then fed to the VLM for reasoning.

To address the Pathology VQA task, the classic Multiple Instance Learning (MIL) approaches [7, 17, 19, 25] are insufficient due to their lack of language understanding power. Recent Multimodal Large Language Model (MLLM) based approaches [9, 22, 32] convert WSI patches into visual tokens, which are then concatenated with the textual question tokens and fed into large language models (LLMs) for multi-modal reasoning. While these approaches have demonstrated competitive performance, they still suffer from two major limitations stemming from the nature of WSIs. The first challenge is the lack of attributable explainability. Although VQA models can generate textual answers, most existing MLLM methods do not reveal which patches or regions in the WSI support the prediction. This absence of localized, patch-level attribution results in a “black-box” behavior that undermines clinical trustworthiness, since pathologists cannot verify the model’s reasoning by inspecting the corresponding image evidence. The second challenge is the redundancy and question-irrelevance of patches. A single WSI can have tens of thousands of patches, many of which are irrelevant to the question, e.g., depicting background tissue, benign structures, or regions unrelated to the clinical decision. Meanwhile, patches from the same tissue type are often redundant; we do not need all of them to make the decision. Furthermore, the strict token limits of current LLMs force existing methods to adopt question-agnostic strategies such as non-selective sampling [9] or pooling [22]. This results in treating all patch tokens equally, which unnecessarily overwhelms the downstream LLM with question-irrelevant visual information and risks degrading the model’s performance.

(a) Original WSI  
(b) Tissue Segmentation  
(c) Question-relevance Heatmap  
![](images/ce11aa6a5acbc895a68b5bc9b53bf85c72be7cbff46691ae3e45e6b1b1395065.jpg)

(d) F1 for tumor patches  
![](images/36cca6c851115e63788dddd2f5568cba816235ab7ea150db23c656633fe4ce12.jpg)  
Figure 2. Visualization and quantitative pre-analysis of patch relevance for a VQA sample (from TCGA-BRCA). (a) Reference WSI. (b) Tissue segmentation, with tumor region shown in red. (c) Patch-level relevance heatmap based on question-patch similarity. High-relevance regions (light region) align with the tumor region from (b). (d) F1 score comparison for retrieving tumor patches using different sampling methods. The question-guided (red) sampling strategy vastly outperforms question-agnostic methods like diversity sampling [3] (blue) and random sampling (gray), demonstrating the limited efficacy of non-guided selection.

To address the aforementioned limitations, we take inspiration from how pathologists reason over WSIs. Rather than examining every region exhaustively, pathologists work in a tissue-aware manner: they first identify the tissue regions relevant to the clinical question and then zoom into a small set of critical patches for verification. Following this principle, we first establish a coarse-grained tissue context. In collaboration with expert pathologists, we define a set of K prompts describing fundamental tissue types, enabling a CLIP-like tissue segmentation that automatically assigns each WSI patch to a semantic category (Figure 2b). This step mirrors the initial stage of locating diagnostically meaningful regions. We further quantitatively validate fine-grained tissue region selection is guided by the question. By calculating the cosine similarity between each patch embedding and the question embedding, we generate a patch-level relevance heatmap (Figure 2c), where lighter regions indicate high relevance to the tumor features specified by the question. A quantitative comparison (Figure 2d) shows that question-guided sampling dramatically outperforms question-agnostic strategies (such as diversity sampling or random sampling) in retrieving relevant tumor patches. This validates our second key observation: the necessity of a question-guided selection mechanism to efficiently identify high-value information within gigapixel WSIs and lead to more accurate answers.

Motivated by these observations, we introduce HistoSelect, a hierarchical, question-guided, and tissue-aware patch selection framework for the pathology VQA task, designed to mirror the coarse-to-fine diagnostic process of pathologists. As shown in Figure 1, we leverage the pathologistdefined prompts for fundamental tissue types and a pretrained patch-level vision–language model [26] to partition the WSI into semantically coherent tissue groups. This provides the coarse-grained structure upon which our method operates. Building on this, HistoSelect implements a twostage selection mechanism grounded in the Information Bottleneck (IB) principle [41]. The first stage, the group sampler, evaluates how relevant each tissue type region (group) is to the input question and determines the patch sampling rate from each group. The second stage, the patch selector, ranks the patches within each active group by relevance to the question and selects the most informative ones according to the allocated token budget. Together, these modules emulate the pathologist’s workflow: first identify the meaningful regions, then zoom in on the key evidence.

To ensure that the selected patches are both sparse and sufficient for answering the question, we formulate the training objective using a dual-level compression loss enforcing sparsity and relevance at both group- and patchlevel. At both levels, we encourage the model to keep only what is necessary by penalizing the divergence between the learned selection probabilities and a dynamic prior derived from question–image similarity. This guides the selectors toward semantically aligned evidence while preventing over-selection. By integrating this IB-driven objective with the VQA loss, HistoSelect produces a compact, questionaligned set of visual tokens that retains critical information and enhances interpretability.

In summary, our contributions are:

• We collaborate with pathologists to design a series of basic tissue type prompts, enabling us to partition the WSIs into distinct tissue regions.

• We introduce HistoSelect, a hierarchical, questionguided, and tissue-aware selection framework based on the IB theory, which effectively prunes questionirrelevant tokens to increase the proportion of questionrelevant tokens fed into the LLM for reasoning, thereby enhancing the model’s interpretability.

• We conduct a detailed pathologist evaluation to ensure that both our tissue segmentation and model selection results align with the expectations of clinical pathologists.

• We achieve state-of-the-art performance on two public datasets and one in-house dataset.

## 2. Related Work

Whole Slide Image Analysis. Traditional WSI analysis primarily focuses on slide-level classification [17, 19, 31, 33, 39, 48, 50] and survival analysis [8, 35, 47, 51, 52] using Multiple Instance Learning (MIL) [17, 19, 23, 30, 48, 49, 53]. More recently, the field has advanced to Pathology Visual Question Answering (VQA) [9, 22, 32], which is a more challenging task. Unlike the aggregation-focused objective of MIL, VQA demands fine-grained reasoning to answer queries ranging from global morphology to the identification of cellular features. Pathology VQA benchmarks include patch-level datasets, such as Quilt-LLaVA [32], and slide-level datasets like SlideChat [9] and WSI-LLaVA [22]. In this work, we focus on developing a hierarchical selection method for the slide-level VQA task.

Multi-Modal Histopathology Models. Recent advances in vision-language foundation models, such as CONCH [26], PLIP [15], MUSK [42], Gecko [18] and CPath-CLIP [36], have demonstrated significant efficacy in bridging visual morphology with clinical language for WSI analysis. Building upon these foundation models, Pathology VQA has emerged as a key task, with the most recent frameworks employing MLLMs to achieve complex reasoning. Initial efforts primarily address localized analysis at the patch or region level, as seen in models such as LLaVA-Med [20], Quilt-LLaVA [32], and PathChat [27]. More recently, the focus has shifted toward slide-level diagnostics, where frameworks such as SlideChat [9] and WSI-LLaVA [22] attempt to handle comprehensive queries by aggregating massive visual features from gigapixel images. To enhance reasoning logic, agent-based frameworks such as PathFinder [12], WSI-Agents [28] and CpathAgent [37] have been proposed to emulate a pathologist’s workflow via iterative reasoning. While their dynamic navigation ensures structured evidence gathering, this sequential process may incur significant inference latency. Unlike exhaustive aggregation or iterative agents, we focus on reducing token redundancy in slide-level VQA by distilling a questionaligned subset of patches for efficient diagnostic reasoning.

Information Bottleneck in Computational Pathology. The Information Bottleneck (IB) principle [41] is an information-theoretic framework for learning, positing that an optimal model should learn a “bottleneck” representation that is maximally compressive of the input while retaining the maximum possible information about the downstream task [2]. Due to its inherent ability to mitigate redundancy, the IB principle has been increasingly adopted in computational pathology to address domain-specific challenges [11, 21, 34, 51]. For example, [21] proposed a variational IB-based fine-tuning strategy to learn task-specific features for WSI classification. Concurrently, [51] employed prototypical IB and information disentanglement to tackle the massive redundancy issues present in multimodal cancer survival prediction. Despite its demonstrated potential in classification and survival analysis, to the best of our knowledge, the IB framework has not yet been explored for pathology VQA. This represents a significant gap, as the hierarchical and token-intensive nature of LLM-based VQA models, which must process a massive number of visual tokens from WSIs, presents a critical challenge of information redundancy and computational inefficiency that the IB principle is ideally suited to address.

## 3. Methods

Our HistoSelect framework is shown in Figure 3. It mainly consists of two core components designed to select question-related patches for downstream multimodal reasoning. The first part is tissue segmentation, where the WSI is partitioned into distinct spatial regions corresponding to M different tissue types (e.g., tumor, stromal, lymphocyte, as illustrated) using pathologist-designed prompts. The second part is the hierarchical selector, which enhances model explainability and token efficiency by selecting patch tokens most relevant to the question. This hierarchical selector includes a group sampler and a patch selector, both taking the question embedding as input. The group sampler predicts the sampling rate for each tissue group, while the patch selector calculates a selection probability for every patch within the group. By ranking these probabilities, the selector extracts the top K most relevant patch tokens. Subsequently, the selected patch tokens and the question tokens are passed to the LLM decoder for answer generation. We detail each component in the following sections.

![](images/6bc17b5f12b0b13a4346381a1784762b1f33b5192219952efcf99be90d42d248.jpg)  
Figure 3. Overview of HistoSelect. The framework operates in two stages: Tissue Segmentation partitions the WSI into M tissue types (e.g., tumor, stromal, lymphocyte) using pathologist-designed prompts. The Hierarchical Selector then uses the question feature to dynam ically select the top K most relevant patch tokens, which are subsequently passed to the LLM for multi-modal answer generation.

Preliminaries. Given a WSI and a question $Q ,$ the WSI is initially divided into a collection of N non-overlapping patches. Each patch is encoded by a pretrained vision encoder to produce a set of features $\mathbf { X } = \{ \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { N } \}$ where $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ is the d-dimensional feature vector for the i-th patch. The question is encoded by a text encoder into a question feature $\mathbf { q } \in \mathbb { R } ^ { d }$ . Furthermore, each patch i is associated with a distinct tissue label $l _ { i } ,$ obtained from the tissue segmentation stage. The collection of all tissue labels is denoted as $L = \{ l _ { 1 } , l _ { 2 } , \dots , l _ { N } \}$ , where $l _ { i } \in \{ 1 , \ldots , M \}$ represents one of the M defined tissue types.

## 3.1. Tissue Segmentation

To identify the tissue regions relevant to the clinical question, we consult with expert pathologists. We design M general prompts, denoted as $\mathcal { P } = \{ p _ { 1 } , p _ { 2 } , . . . , p _ { M } \}$ , where each prompt $P _ { j }$ is designed to generally represent a key histological component within the WSI. Specifically, we utilize the visual encoder from CONCH [26] to obtain the patch features $\mathbf { X } = \{ \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { N } \}$ , and its text encoder to embed the M prompts into a feature space, yielding the tissue prompt features $\mathcal { T } = \{ \mathbf { t } _ { 1 } , \mathbf { t } _ { 2 } , \dots , \mathbf { t } _ { M } \}$ , where $\mathbf { t } _ { j }$ is the feature for prompt $P _ { j }$ . The tissue label $l _ { i }$ for patch i is determined by the highest cosine similarity between the patch feature $\mathbf { x } _ { i }$ and all prompt features $\mathbf { t } _ { j }$

$$
l _ { i } = \underset { j \in \{ 1 , \dots , M \} } { \operatorname { a r g m a x } } \left( \frac { \mathbf { x } _ { i } \cdot \mathbf { t } _ { j } } { \| \mathbf { x } _ { i } \| \cdot \| \mathbf { t } _ { j } \| } \right)\tag{1}
$$

The resulting set of labels $L = \{ l _ { 1 } , l _ { 2 } , \ldots , l _ { N } \}$ effectively partitions the WSI into M distinct tissue regions, which form the basis for the hierarchical selection process.

## 3.2. Group Sampler

The hierarchical selection process begins with the group sampler, which determines the importance of each tissue region relative to the input question. For each of the M tissue groups, we first compute a group prototype feature ${ \bf { g } } _ { j }$ by applying average pooling over all patch features $x _ { i }$ belonging to that group j. Let $\mathcal { T } _ { j }$ be the set of indices for patches belonging to tissue group j. The group prototype ${ \bf { g } } _ { j }$ is defined as:

$$
{ \bf g } _ { j } = \frac { 1 } { N _ { j } } \sum _ { i \in \mathcal { T } _ { j } } { \bf x } _ { i }\tag{2}
$$

where $N _ { j }$ is the total number of patches in group $j .$ The group sampler $\mathcal { F } _ { \mathrm { g r o u p } }$ then predicts a sampling rate $r _ { j }$ for group $j ,$ indicating the importance of this group. This prediction is based on the concatenation of the group prototype ${ \bf { g } } _ { j }$ and the question feature ${ \bf q } ,$ ensuring the sampling is context-aware. The sampling rate $r _ { j }$ is constrained to (0, 1) using the sigmoid function $\sigma ( \cdot ) ;$

$$
r _ { j } = \sigma \left( \mathcal { F } _ { \mathrm { g r o u p } } ( [ \mathbf { g } _ { j } ; \mathbf { q } ] ) \right)\tag{3}
$$

where the group sampler $\mathcal { F } _ { \mathrm { g r o u p } }$ is implemented with two linear layers, and $r _ { j }$ dictates the proportion of patches to be sampled from group j.

## 3.3. Patch Selector

The patch selector performs the final, fine-grained selection of relevant patches within each tissue group. Similar to the group sampler, the selection mechanism is driven by the question context. For every patch i, the patch selector $\mathcal { F } _ { \mathrm { p a t c h } }$ predicts a selection probability $s _ { i } .$ . This prediction is based on the concatenation of the individual patch feature $\mathbf { x } _ { i }$ and the question feature q:

$$
s _ { i } = \sigma ( \mathcal { F } _ { \mathrm { p a t c h } } ( [ \mathbf { x } _ { i } ; \mathbf { q } ] ) )\tag{4}
$$

where $s _ { i } \in ( 0 , 1 )$ is the predicted probability that patch i is relevant to question Q, and $\mathcal { F } _ { \mathrm { p a t c h } }$ is implemented with two linear layers. For each tissue group j, the number of patches to be selected $k _ { j }$ is determined by multiplying the group’s predicted sampling rate $r _ { j }$ by its total size $N _ { j }$ , followed by rounding up:

$$
k _ { j } = \lceil r _ { j } \cdot N _ { j } \rceil\tag{5}
$$

Finally, we select the top $k _ { j }$ features, denoted as $\mathbf { Z } _ { j }$ , from group j by ranking all features x<sub>i</sub> belonging to $\mathcal { T } _ { j }$ based on their selection probability $s _ { i }$ . The complete set of selected patch features Z is the union of all selected features across the M groups: $\begin{array} { r } { \mathbf { Z } = \bigcup _ { j = 1 } ^ { M } \mathbf { Z } _ { j } } \end{array}$

## 3.4. Hierarchical IB for Patch Selection

Our learning objective is based on the IB theory [41], which seeks to learn an optimal compressed representation Z of the input features X that retains maximal information about the ground truth answer Y. Given the question feature q, this objective is formally written as:

$$
\mathcal { L } _ { \mathrm { I B } } = I ( \mathbf { Z } ; \mathbf { Y } \mid \mathbf { q } ) - \beta I ( \mathbf { Z } ; \mathbf { X } \mid \mathbf { q } )\tag{6}
$$

where $I ( \cdot ; \cdot )$ denotes the mutual information, and $\beta$ is a Lagrangian multiplier balancing the trade-off between relevance and compression. To accommodate the hierarchical structure of WSIs, we model the selection process via a joint latent variable $\boldsymbol { \mathrm { Z } } = ( Z _ { g } , Z _ { p } )$ , where $Z _ { g }$ and $Z _ { p }$ denote the group-level and patch-level selection variables, respectively. By applying the chain rule, the complexity term $I ( \mathbf { Z } , \mathbf { X } \mid \mathbf { q } )$ is decomposed into a marginal group-level term and a conditional patch-level term:

$$
I ( \boldsymbol { \mathrm { Z } } _ { g } , \boldsymbol { \mathrm { Z } } _ { p } ; \mathbf { X } \mid \mathbf { q } ) = I ( \boldsymbol { \mathrm { Z } } _ { g } ; \mathbf { X } \mid \mathbf { q } ) + I ( \boldsymbol { \mathrm { Z } } _ { p } ; \mathbf { X } \mid \boldsymbol { \mathrm { Z } } _ { g } , \mathbf { q } )\tag{7}
$$

Since mutual information is computationally intractable, we adopt the Variational Information Bottleneck (VIB) framework [41] to derive a tractable bound. Inspired by recent hierarchical IB frameworks [11, 34] designed for multi-scale WSIs, we introduce the hierarchical variational posteriors $p _ { \phi _ { g } } ( Z _ { g } \mid X , \mathbf { q } )$ and $p _ { \phi _ { p } } ( \mathsf { Z } _ { p } \mid X , \mathsf { Z } _ { g } , \mathbf { q } )$ to model the grouplevel sampling and patch-level selection, respectively. Furthermore, we utilize the LLM as the variational decoder $p _ { \theta } ( Y \mid Z , \mathbf { q } )$ to generate the final answer. By approximating the complexity terms in Equation (7) with their corresponding KL divergence upper bounds, the resulting hierarchical IB objective function is formulated as:

$$
\begin{array} { r l } { \mathcal { I } _ { \mathrm { H I B } } = \mathbb { E } _ { \mathcal { D } } \Big [ \mathbb { E } _ { Z \sim p _ { \phi } } [ \log p _ { \theta } ( Y \mid Z , \mathbf { q } ) ] } & { } \\ { - \beta _ { g } D _ { \mathrm { K L } } ( p _ { \phi _ { g } } ( Z _ { g } \mid X , \mathbf { q } ) \parallel p _ { g } ) } & { } \\ { - \beta _ { p } D _ { \mathrm { K L } } ( p _ { \phi _ { p } } ( Z _ { p } \mid X , Z _ { g } , \mathbf { q } ) \parallel p _ { p } ) \Big ] } & { } \end{array}\tag{8}
$$

where $p _ { g }$ and $p _ { p }$ are the prior distributions. The hyperparameters $\beta _ { g }$ and $\beta _ { p }$ independently regulate the information flow at the group-level and patch-level granularities, respectively. The complete derivation of this hierarchical decomposition is provided in the supplementary material.

## 3.5. Loss Function and Implementation

We define the loss function to be minimized as ${ \mathcal { L } } _ { \mathrm { f i n a l } } ~ =$ $- \mathcal { I } _ { \mathrm { H I B } }$ . Following the hierarchical decomposition in Equation (8), the total loss is formulated as a weighted sum of the task-specific VQA loss and the hierarchical compression terms:

$$
\mathcal { L } _ { \mathrm { f i n a l } } = \mathcal { L } _ { \mathrm { V Q A } } + \beta _ { g } \mathcal { L } _ { \mathrm { g r o u p } } + \beta _ { p } \mathcal { L } _ { \mathrm { p a t c h } }\tag{9}
$$

where $\mathcal { L } _ { \mathrm { g r o u p } }$ and ${ \mathcal { L } } _ { \mathrm { p a t c h } }$ are the compression loss terms derived from the group sampler and patch selector, respectively. The $\beta _ { g }$ and $\beta _ { p }$ control the trade-off between taskrelevant information and redundancy at each stage.

VQA Loss. The VQA loss is the negative log-likelihood over the answer sequence:

$$
\mathcal { L } _ { \mathrm { V Q A } } = - \sum _ { t = 1 } ^ { T } \log p _ { \boldsymbol { \theta } } ( y _ { t } \mid y _ { < t } , \mathbf { Z } , \mathbf { q } )
$$

where $\mathrm { ~ Y ~ } = ~ \{ y _ { 1 } , \dots , y _ { T } \}$ is the ground truth answer sequence, and the total loss is averaged over the dataset D.

Group-Level Compression Loss. Weighted by $\beta _ { g }$ , this term regularizes the group sampler by minimizing the deviation of the predicted group sampling rate $r _ { j }$ from a pseudoprior parameter $p _ { j } ^ { g }$ . In this context, $r _ { j }$ and $\bar { p _ { j } ^ { g } }$ are interpreted as the parameters of two Bernoulli distributions. The loss is averaged over the M tissue groups:

$$
\mathcal { L } _ { \mathrm { g r o u p } } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbf { D } _ { \mathrm { K L } } ( \mathbf { B } ( r _ { j } ) \parallel \mathbf { B } ( p _ { j } ^ { g } ) )\tag{10}
$$

In practice, the pseudo-prior parameter $p _ { j } ^ { g }$ is implemented as the cosine similarity between the group prototype $\mathbf { g } _ { j }$ and the question feature q.

Patch-Level Compression Loss. Weighted by $\beta _ { p } ,$ , this term regularizes the patch selector by minimizing the deviation of the patch selection probability $s _ { i }$ from a pseudo-prior parameter $p _ { i } ^ { p }$ . In this context, $s _ { i }$ and $p _ { i } ^ { p }$ are interpreted as the parameters of two Bernoulli distributions. The loss is averaged over the N patches:

$$
\mathcal { L } _ { \mathrm { p a t c h } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { D } _ { \mathrm { K L } } \big ( \mathbf { B } ( s _ { i } ) \parallel \mathbf { B } ( p _ { i } ^ { p } ) \big )\tag{11}
$$

Similarly, the pseudo-prior parameter $p _ { i } ^ { p }$ is the cosine similarity between the patch feature $\mathbf { x } _ { i }$ and the question feature q. The KL divergence between two Bernoulli distributions with parameters π and $p$ is defined as:

$$
\mathbf { D } _ { \mathrm { K L } } ( \mathbf { B } ( \pi ) \parallel \mathbf { B } ( p ) ) = \pi \log { \frac { \pi } { p } } + ( 1 - \pi ) \log { \frac { 1 - \pi } { 1 - p } }\tag{12}
$$

## 3.6. Differentiable Hard Selection

During training, the model must sample discrete patches to compute L<sub>VQA</sub>, yet this sampling operation is nondifferentiable. To overcome this, we adopt the Straight-Through Estimator (STE) [5]. Specifically, a hard binary mask is applied during the forward pass to select the top- $\cdot k _ { j }$ features for each group $j .$ . During the backward pass, gradients are propagated directly through the soft probabilities $( r _ { j }$ and $s _ { i } ) _ { \cdot }$ , bypassing the discrete sampling step. This technique enables the entire pipeline, including both the group sampler and the patch selector, to be optimized end-to-end under the guidance of the VQA loss.

## 4. Experiments and Results

Datasets and Preprocessing. We conduct experiments on three slide-level VQA datasets, including two public benchmarks and one private dataset. The public datasets include 1) SlideBench-VQA [9] comprises 4,560 WSIs and 176K VQA pairs, spanning 10 different cancer types and covering three different scenarios: Microscopy, Diagnosis, and Clinical. 2) WSI-Bench [22] contains 9,850 WSIs and 180K VQA pairs, with scenarios focusing on Morphological analysis, Diagnosis, and Treatment Planning. Finally, to assess the model’s generalizability and clinical robustness, we curate a 3) Private Ovarian Dataset. This dataset consists of 375 WSIs and 375 corresponding VQA pairs annotated by pathologists, focusing on key diagnostic features of ovarian cancer, and is used as an independent test set. We followed the CLAM [25] methodology for patch-cutting and feature extraction, processing all patches to a size of 224 × 224.

Baselines. We compare our method with several state-ofthe-art WSI-based MLLMs. These baselines include models specifically designed for pathological and medical VQA tasks, such as Quilt-LLaVA [32], WSI-VQA [7], LLaVA-Med [20], and SlideChat [9], as well as models specialized for WSI report generation, including MI-Gen [6] and

Hist-Gen [13]. We also include the general-purpose MLLM GPT-4o as a non-specialized baseline.

Evaluation Metrics. For the closed-ended tasks, we group the questions based on various clinical categories. Performance is evaluated using accuracy. For the open-ended answer generation tasks, we adopt text-generation metrics, including BLEU and ROUGE-L to measure semantic similarity between generated and reference answers. Following WSI-LLaVA [22], we additionally employ two LLM-as-ajudge metrics: WSI-Precision (WSI-P), which evaluates the factual correctness of model responses, and WSI-Relevance (WSI-R), which assesses how well each response aligns with the reference answer in a clinical context.

Implementation Details. Following SlideChat [9], we employ the CONCH encoder [26] to extract patch-level features and LongNet [10, 44] for slide-level features, utilizing Qwen2.5-7B-Instruct [40] as our LLM framework. We use the text encoder from CONCH to obtain question embeddings. For all experiments, we adhere to the official data splits of SlideBench-VQA [9] and WSI-Bench [22]. Our training is conducted in two stages: consistent with SlideChat [9], the first stage focuses on projector training for modality alignment. Subsequently, the second stage jointly fine-tunes the projector, the LLM (using LoRA), and our HistoSelect module. Detailed hyperparameter settings are provided in the supplementary material.

## 4.1. Quantitative Results

Close-ended Selection Performance. Table 1 presents the close-ended VQA performance of our model against several state-of-the-art baselines across three benchmarks: SlideBench-VQA (TCGA), WSI-Bench (Close), and our In-house Ovarian dataset. We compare our WSI-based method against both thumbnail-based models (such as GPT-4o, Quilt-LLaVA) and other WSI-based models (such as SlideChat). The results clearly demonstrate that our model achieves the best performance across all tested categories, attaining an average score of 83.80% and significantly outperforming all other baseline methods.

Open-ended Generation Performance. To evaluate the model’s open-ended text generation capabilities, we conducted tests on the WSI-Bench benchmark, with results detailed in Table 2. We use BLEU (1-4) and ROUGE-L metrics to assess the quality of Report Generation and WSI metrics to evaluate the domain-specific VQA performance. The experimental results demonstrate the significant advantages of our model. For Report Generation, our model achieves the highest scores across all five metrics, with its BLEU-4 (0.221) and ROUGE-L (0.463) scores notably surpassing other advanced models like Quilt-LLaVA and SlideChat. In the domain-specific VQA tasks, our model obtains the best results in 5 out of 6 metrics, proving its superior and wellbalanced open-ended VQA capabilities.

Table 1. Close-ended VQA accuracy (%) across three benchmarks: SlideBench-VQA, WSI-Bench, and our In-house Ovarian dataset. Ou method consistently achieves the highest accuracy across all task categories and obtains the best overall average performance.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Input</td><td colspan="3">SlideBench-VQA (TCGA)</td><td colspan="3">WSI-Bench (Close)</td><td>In-house Ovarian</td><td rowspan="2">Average</td></tr><tr><td>Microscopy</td><td>Diagnosis</td><td>Clinical</td><td>Morphology</td><td>Diagnosis</td><td>Treatment</td><td>Diagnosis</td></tr><tr><td>GPT-40</td><td>Thumbnail</td><td>39.24</td><td>24.12</td><td>44.67</td><td>47.07</td><td>53.06</td><td>87.50</td><td></td><td>49.28</td></tr><tr><td>Quilt-LLaVA[32]</td><td>Thumbnail</td><td>52.39</td><td>30.19</td><td>49.33</td><td>94.13</td><td>84.13</td><td>97.92</td><td>70.67</td><td>68.39</td></tr><tr><td>LLaVA-Med [20]</td><td>Thumbnail</td><td>52.15</td><td>29.97</td><td>47.33</td><td>91.04</td><td>81.32</td><td>95.83</td><td>70.67</td><td>66.90</td></tr><tr><td>SlideChat [9]</td><td>WSI</td><td>83.15</td><td>71.36</td><td>75.33</td><td>91.34</td><td>82.15</td><td>93.75</td><td>69.33</td><td>80.88</td></tr><tr><td>HistoSelect</td><td>WSI</td><td>84.62</td><td>73.09</td><td>77.30</td><td>94.57</td><td>85.79</td><td>97.92</td><td>73.33</td><td>83.80</td></tr></table>

Table 2. Open-ended VQA performance on the WSI-Bench dataset. We evaluate two tasks: Report Generation and domain-specific VQA. Our method achieves the highest performance across all report-generation metrics and the best results on 5 of 6 domain-specific metrics.
<table><tr><td rowspan="2"></td><td colspan="9">WSI-Bench</td><td colspan="2"></td></tr><tr><td colspan="6">Report Generation</td><td colspan="2">Morphology</td><td colspan="2">Diagnosis</td><td colspan="2">Treatment</td></tr><tr><td>Method</td><td>BLEU-1</td><td>BLEU-2</td><td>BLEU-3</td><td>BLEU-4</td><td>ROUGE-L</td><td>WSI-P</td><td>WSI-R</td><td>WSI-P</td><td>WSI-R</td><td>WSI-P</td><td></td><td>WSI-R</td></tr><tr><td>GPT-40</td><td>0.202</td><td>0.069</td><td>0.030</td><td>0.016</td><td>0.132</td><td>0.220</td><td>0.204</td><td>0.472</td><td>0.457</td><td>0.513</td><td></td><td>0.704</td></tr><tr><td>WSI-VQA [7]</td><td>0.301</td><td>0.225</td><td>0.181</td><td>0.155</td><td>0.343</td><td>0.395</td><td>0.462</td><td>0.436</td><td>0.525</td><td></td><td>0.591</td><td>0.595</td></tr><tr><td>MI-Gen [6]</td><td>0.403</td><td>0.306</td><td>0.248</td><td>0.209</td><td>0.446</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Histo-Gen [13]</td><td>0.406</td><td>0.307</td><td>0.248</td><td>0.208</td><td>0.448</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Quilt-LLaVA [32]</td><td>0.421</td><td>0.316</td><td>0.257</td><td>0.216</td><td>0.455</td><td>0.453</td><td>0.484</td><td>0.521</td><td></td><td>0.552</td><td>0.751</td><td>0.807</td></tr><tr><td>SlideChat [9]</td><td>0.413</td><td>0.312</td><td>0.254</td><td>0.215</td><td>0.450</td><td>0.512</td><td>0.541</td><td>0.501</td><td></td><td>0.522</td><td>0.745</td><td>0.712</td></tr><tr><td>HistoSelect</td><td>0.431</td><td>0.324</td><td>0.262</td><td>0.221</td><td>0.463</td><td>0.538</td><td>0.589</td><td>0.542</td><td></td><td>0.587</td><td>0.766</td><td>0.801</td></tr></table>

## 4.2. Qualitative Result

Visualization. To intuitively demonstrate the effectiveness of our question-aware selection mechanism, we provide qualitative visualizations in Figure 4. The figure illustrates the model’s workflow, proceeding from (a) WSI to (b) the tissue segmentation mask, (c) a subset of candidate patches extracted from the tissue regions, and (d) the sparse set of question-relevant patches after our model’s selection. As shown, our method successfully filters out a large number of background and diagnostically irrelevant patches, allowing the model to focus its computation and attention on the most salient regions to answer the VQA query.

Pathologist’s Evaluation. To validate the practical utility and interpretability of our model from a clinical perspective, we conducted a human evaluation survey with two independent pathologists. We mainly evaluate the model interpretability and performance from two aspects, with a detailed survey in the supplementary material:

1. Tissue segmentation, we presented the pathologists with the original slide and our generated tissue mask. They were asked to rate the following on a 5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree):

• Q1: “How accurate is the tissue segmentation?”

2. Patch selection, we showed the pathologists the visualizations of patches before and after selection for a given question. They were asked to rate:

• Q2.1: “Does the model filter out a lot of questionirrelevant patches?”

• Q2.2: “Are the selected patches sufficient to answer the question?”

The detailed average scores are presented in Table 3. We are encouraged to find that the average rating for all four questions exceeded 3.5. This strongly indicates that (1) pathologists find our tissue segmentation accurate, and (2) they confirm that our selection model effectively filters out irrelevant regions while preserving the necessary diagnostic information to answer the clinical question.

Table 3. Average scores from the pathologist evaluation survey.
<table><tr><td>Category</td><td>Question ID</td><td>Avg. Score (P1)</td><td>Avg. Score (P2)</td></tr><tr><td>Tissue Seg.</td><td>Q1 Accuracy</td><td>4.17</td><td>3.67</td></tr><tr><td rowspan="2">Patch Selection</td><td>Q2.1 Question-relevant</td><td>4.80</td><td>3.87</td></tr><tr><td>Q2.2 Answer-relevant</td><td>4.67</td><td>3.73</td></tr></table>

## 4.3. Ablation Studies

To validate the effectiveness of HistoSelect, we conduct a series of ablation studies. We analyze three key aspects of our model: (1) the superiority of our learned selection mechanism against alternative strategies, (2) the contribution of each component in our hierarchical framework, and (3) the impact of the token budget on model performance.

Selection Mechanism. In Table 4, we first compare our full model against alternative selection baselines, all constrained to the same token budget. The poor performance of random sampling serves as a lower bound, confirming that intelligent selection is critical. While the diversity-based method DivPrune [3] performs better, it still falls short of our model, indicating that selecting question-relevant patches is more important than simply selecting diverse ones. Most importantly, we evaluate a simple similarity baseline that replaces our learnable selectors $( \mathcal { F } _ { \mathrm { g r o u p } }$ and $\mathcal { F } _ { \mathrm { p a t c h } } )$ with the non-learnable pseudo-prior parameters $p _ { j } ^ { g }$ and $p _ { i } ^ { p }$ (derived from cosine similarity) directly. Its inferior performance strongly validates our core hypothesis: an endto-end learned policy, regularized by the IB objective, is essential for identifying the most salient visual evidence and significantly outperforms static, similarity-based heuristics.

![](images/3b0bf55681b2b18db1b60e3aec8f3843e1bd1fa3a041edd6b30b80e05a08088a.jpg)  
Figure 4. Visualization of tissue segmentation and selection process. (a) Original WSI. (b) Tissue segmentation mask. (c)Visualization before selection (a randomly selected subset is shown for clarity). (d) Visualization after selection. Compared to (c), the patches selected by our model in (d) significantly remove non-tumor patches, demonstrating an improved focus on informative tumor-related regions.  
Table 6. Ablation Study on # of Tokens.

Table 4. Selection Mechanism Ablation.  
Table 5. Model Component Ablation Study.
<table><tr><td>Accuracy</td><td>Morphology</td><td>Diagnosis</td><td>Treatment</td></tr><tr><td>Random Sampling</td><td>88.84</td><td>78.02</td><td>91.67</td></tr><tr><td>DivPrune</td><td>90.01</td><td>80.99</td><td>93.75</td></tr><tr><td>Simple Similarity</td><td>92.22</td><td>81.98</td><td>93.75</td></tr><tr><td>Ours</td><td>94.57</td><td>85.79</td><td>97.92</td></tr></table>

<table><tr><td>Accuracy</td><td>Morphology</td><td>Diagnosis</td><td>Treatment</td></tr><tr><td>Random Sampling</td><td>88.84</td><td>78.02</td><td>91.67</td></tr><tr><td>w/o Group Sampler</td><td>91.78</td><td>81.82</td><td>95.83</td></tr><tr><td>w/o Patch Selector</td><td>92.07</td><td>81.32</td><td>93.75</td></tr><tr><td>Ours</td><td>94.57</td><td>85.79</td><td>97.92</td></tr></table>

<table><tr><td>Accuracy</td><td>Morphology</td><td>Diagnosis</td><td>Treatment</td></tr><tr><td>10k Tokens</td><td>94.12</td><td>85.12</td><td>97.92</td></tr><tr><td>5k Tokens</td><td>94.57</td><td>85.79</td><td>97.92</td></tr><tr><td>2k Tokens</td><td>93.83</td><td>83.80</td><td>95.83</td></tr><tr><td>1k Tokens</td><td>91.19</td><td>82.15</td><td>95.83</td></tr></table>

Model Components. In Table 5, we dissect our hierarchical architecture to understand each component. The baseline model, which defaults to Random Sampling, yields the poorest results. Removing only the patch selector $( \mathcal { F } _ { \mathrm { p a t c h } } )$ forces the model to rely on coarse group selection, and the subsequent performance drop highlights the necessity of fine-grained selection for critical patches. Conversely, removing the group sampler $( \mathcal { F } _ { \mathrm { g r o u p } } )$ degrades the model to a “flat” selection mechanism, forcing the patch selector to search globally across all N patches. Its poor performance relative to our full model confirms the benefit of our coarseto-fine approach, as the group sampler effectively narrows the search space. The superior performance of our full model demonstrates that the group sampler and the patch selector work synergistically.

Impact of Token Budget. In Table 6, we analyze the performance of HistoSelect under different token budget limits. We observe that performance improves as the token count limit increases from 1k to 5k, indicating the benefit of more visual context. However, the model achieves its peak performance at 5k tokens. Notably, increasing the limit further to 10k tokens provides no additional performance gain and even shows a slight degradation, likely due to the introduction of redundant information. This result is significant: HistoSelect validates that a large portion of a WSI is redundant for a specific question, as it achieves its optimal accuracy by selecting a compact, sufficient subset of only 30% of the total patches. This translates to a 70% reduction in token computation while maximizing diagnostic accuracy.

## 5. Conclusion

In this work, we introduce HistoSelect, a question-aware framework addressing explainability and redundancy in current pathology VQA models. By mimicking the coarseto-fine diagnostic strategy of human pathologists, HistoSelect efficiently prunes question-irrelevant patches, thereby increasing the signal-to-noise ratio for the downstream MLLM. This targeted selection process provides transparent and attributable visual evidence for prediction. Extensive experiments across diverse datasets, complemented by rigorous evaluations from practicing pathologists, confirm that HistoSelect not only achieves state-of-the-art performance but also delivers the trustworthy, explainable reasoning necessary to bridge the gap between automated analysis and clinical adoption in computational pathology.

## Acknowledgment

This work has been supported by Mayo Clinic Center for Individualized Medicine, and by the generous support of Schmidt Sciences and the Susan Morrow Legacy Foundation. This work was also partially supported by grants NSF CCF-2144901, NIH R01NS143143, and R01CA297843.

## References

[1] Streamlit. https://streamlit.io/, 2025. 12

[2] Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy. Deep variational information bottleneck. In ICLR, 2017. 3, 11

[3] Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversity-based visual token pruning for large multimodal models. In CVPR, 2025. 2, 7

[4] Laura Barisoni, Kyle J Lafata, Stephen M Hewitt, Anant Madabhushi, and Ulysses GJ Balis. Digital pathology and computational image analysis in nephropathology. Nature Reviews Nephrology, 2020. 1

[5] Yoshua Bengio, Nicholas Leonard, and Aaron Courville.´ Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013. 6

[6] Pingyi Chen, Honglin Li, Chenglu Zhu, Sunyi Zheng, Zhongyi Shui, and Lin Yang. Wsicaption: Multiple instance generation of pathology reports for gigapixel whole-slide images. In MICCAI, 2024. 6, 7

[7] Pingyi Chen, Chenglu Zhu, Sunyi Zheng, Honglin Li, and Lin Yang. Wsi-vqa: Interpreting whole slide images by generative visual question answering. In ECCV, 2024. 1, 6, 7

[8] Richard J Chen, Ming Y Lu, Wei-Hung Weng, Tiffany Y Chen, Drew FK Williamson, Trevor Manz, Maha Shady, and Faisal Mahmood. Multimodal co-attention transformer for survival prediction in gigapixel whole slide images. In ICCV, 2021. 1, 3

[9] Ying Chen, Guoan Wang, Yuanfeng Ji, Yanjun Li, Jin Ye, Tianbin Li, Ming Hu, Rongshan Yu, Yu Qiao, and Junjun He. Slidechat: A large vision-language assistant for wholeslide pathology image understanding. In CVPR, 2025. 1, 2, 3, 6, 7, 14

[10] Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, and Furu Wei. Longnet: Scaling transformers to 1,000,000,000 tokens. In ICLR, 2023. 6

[11] Zeyu Gao, Anyu Mao, Kefei Wu, Yang Li, Liebin Zhao, Xianli Zhang, Jialun Wu, Lisha Yu, Chao Xing, Tieliang Gong, et al. Childhood leukemia classification via information bottleneck enhanced hierarchical multi-instance learning. TMI, 2023. 3, 5

[12] Fatemeh Ghezloo, Mehmet Saygin Seyfioglu, Rustin Soraki, Wisdom O Ikezogwo, Beibin Li, Tejoram Vivekanandan, Joann G Elmore, Ranjay Krishna, and Linda Shapiro. Pathfinder: A multi-modal multi-agent system for medical diagnostic decision-making applied to histopathology. In CVPR, 2025. 3

[13] Zhengrui Guo, Jiabo Ma, Yingxue Xu, Yihui Wang, Liansheng Wang, and Hao Chen. Histgen: Histopathology report generation via local-global feature encoding and crossmodal context interaction. In MICCAI, 2024. 6, 7

[14] Wentao Huang, Xiaoling Hu, Shahira Abousamra, Prateek Prasanna, and Chao Chen. Hard negative sample mining for whole slide image classification. In MICCAI, 2024. 1

[15] Zhi Huang, Federico Bianchi, Mert Yuksekgonul, Thomas J Montine, and James Zou. A visual–language foundation model for pathology image analysis using medical twitter. Nature medicine, 2023. 3

[16] Wisdom Ikezogwo, Saygin Seyfioglu, Fatemeh Ghezloo, Dylan Geva, Fatwir Sheikh Mohammed, Pavan Kumar Anand, Ranjay Krishna, and Linda Shapiro. Quilt-1m: One million image-text pairs for histopathology. In NeurIPS, 2023. 1

[17] Maximilian Ilse, Jakub Tomczak, and Max Welling. Attention-based deep multiple instance learning. In ICML, 2018. 1, 3

[18] Saarthak Kapse, Pushpak Pati, Srikar Yellapragada, Srijan Das, Rajarsi R Gupta, Joel Saltz, Dimitris Samaras, and Prateek Prasanna. Gecko: Gigapixel vision-concept contrastive pretraining in histopathology. In ICCV, 2025. 3

[19] Bin Li, Yin Li, and Kevin W Eliceiri. Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. In CVPR, 2021. 1, 3

[20] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large languageand-vision assistant for biomedicine in one day. In NeurIPS, 2023. 3, 6, 7, 14

[21] Honglin Li, Chenglu Zhu, Yunlong Zhang, Yuxuan Sun, Zhongyi Shui, Wenwei Kuang, Sunyi Zheng, and Lin Yang. Task-specific fine-tuning via variational information bottleneck for weakly-supervised pathology whole slide image classification. In CVPR, 2023. 3

[22] Yuci Liang, Xinheng Lyu, Wenting Chen, Meidan Ding, Jipeng Zhang, Xiangjian He, Song Wu, Xiaohan Xing, Sen Yang, Xiyue Wang, et al. Wsi-llava: a multimodal large lan guage model for whole slide image. In ICCV, 2025. 1, 2, 3, 6, 14, 15

[23] Tiancheng Lin, Zhimiao Yu, Hongyu Hu, Yi Xu, and Chang-Wen Chen. Interventional bag multi-instance learning on whole-slide pathological images. In CVPR, 2023. 3

[24] Ming Y Lu, Tiffany Y Chen, Drew FK Williamson, Melissa Zhao, Maha Shady, Jana Lipkova, and Faisal Mahmood. Aibased pathology predicts origins for cancers of unknown pri mary. Nature, 2021. 1

[25] Ming Y Lu, Drew FK Williamson, Tiffany Y Chen, Richard J Chen, Matteo Barbieri, and Faisal Mahmood. Data-efficient and weakly supervised computational pathology on whole slide images. Nature Biomedical Engineering, 2021. 1, 6

[26] Ming Y Lu, Bowen Chen, Drew FK Williamson, Richard J Chen, Ivy Liang, Tong Ding, Guillaume Jaume, Igor Odintsov, Long Phi Le, Georg Gerber, et al. A visuallanguage foundation model for computational pathology. Nature medicine, 2024. 2, 3, 4, 6

[27] Ming Y Lu, Bowen Chen, Drew FK Williamson, Richard J Chen, Melissa Zhao, Aaron K Chow, Kenji Ikemura, Ahrong Kim, Dimitra Pouli, Ankush Patel, et al. A multimodal generative ai copilot for human pathology. Nature, 2024. 3

[28] Xinheng Lyu, Yuci Liang, Wenting Chen, Meidan Ding, Jiaqi Yang, Guolin Huang, Daokun Zhang, Xiangjian He, and Linlin Shen. Wsi-agents: A collaborative multi-agent system for multi-modal whole slide image analysis. In MICCAI, 2025. 3

[29] Muhammad Khalid Khan Niazi, Anil V Parwani, and Metin N Gurcan. Digital pathology and artificial intelligence. The lancet oncology, 2019. 1

[30] Linhao Qu, Manning Wang, Zhijian Song, et al. Bidirectional weakly supervised knowledge distillation for whole slide image classification. In NeurIPS, 2022. 3

[31] Linhao Qu, Zhiwei Yang, Minghong Duan, Yingfan Ma, Shuo Wang, Manning Wang, and Zhijian Song. Boosting whole slide image classification from the perspectives of distribution, correlation and magnification. In CVPR, 2023. 1, 3

[32] Mehmet Saygin Seyfioglu, Wisdom O Ikezogwo, Fatemeh Ghezloo, Ranjay Krishna, and Linda Shapiro. Quilt-llava: Visual instruction tuning by extracting localized narratives from open-source histopathology videos. In CVPR, 2024. 1, 3, 6, 7, 14

[33] Zhuchen Shao, Hao Bian, Yang Chen, Yifeng Wang, Jian Zhang, Xiangyang Ji, et al. Transmil: Transformer based correlated multiple instance learning for whole slide image classification. In NeurIPS, 2021. 1, 3

[34] Jiangbo Shi, Lufei Tang, Zeyu Gao, Yang Li, Chunbao Wang, Tieliang Gong, Chen Li, and Huazhu Fu. Mg-trans: Multi-scale graph transformer with information bottleneck for whole slide image classification. TMI, 2023. 3, 5

[35] Andrew H Song, Richard J Chen, Guillaume Jaume, Anurag J Vaidya, Alexander S Baras, and Faisal Mahmood. Multimodal prototyping for cancer survival prediction. In ICML, 2024. 1, 3

[36] Yuxuan Sun, Yixuan Si, Chenglu Zhu, Xuan Gong, Kai Zhang, Pingyi Chen, Ye Zhang, Zhongyi Shui, Tao Lin, and Lin Yang. Cpath-omni: A unified multimodal foundation model for patch and whole slide image analysis in computational pathology. In CVPR, 2025. 3

[37] Yuxuan Sun, Yixuan Si, Chenglu Zhu, Kai Zhang, Zhongyi Shui, Bowen Ding, Tao Lin, and Lin Yang. Cpathagent: An agent-based foundation model for interpretable highresolution pathology image analysis mimicking pathologists diagnostic logic. In NeurIPS, 2025. 3

[38] Yuxuan Sun, Hao Wu, Chenglu Zhu, Yixuan Si, Qizi Chen, Yunlong Zhang, Kai Zhang, Jingxiong Li, Jiatong Cai, Yuhan Wang, et al. Pathbench: Advancing the benchmark of large multimodal models for pathology image understanding at patch and whole slide level. TMI, 2025. 1

[39] Wenhao Tang, Sheng Huang, Xiaoxian Zhang, Fengtao Zhou, Yi Zhang, and Bo Liu. Multiple instance learning framework with masked hard instance mining for whole slide image classification. In CVPR, 2023. 1, 3

[40] Qwen Team et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 6

[41] Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. arXiv preprint physics/0004057, 2000. 2, 3, 5

[42] Jinxi Xiang, Xiyue Wang, Xiaoming Zhang, Yinghua Xi, Feyisope Eweje, Yijiang Chen, Yuchen Li, Colin Bergstrom, Matthew Gopaulchan, Ted Kim, et al. A vision–language foundation model for precision oncology. Nature, 2025. 3

[43] Feng Xu, Chuang Zhu, Wenqi Tang, Ying Wang, Yu Zhang, Jie Li, Hongchuan Jiang, Zhongyue Shi, Jun Liu, and Mulan Jin. Predicting axillary lymph node metastasis in early breast cancer using deep learning on primary tumor biopsy slides. Frontiers in oncology, 2021. 18

[44] Hanwen Xu, Naoto Usuyama, Jaspreet Bagga, Sheng Zhang, Rajesh Rao, Tristan Naumann, Cliff Wong, Zelalem Gero, Javier Gonzalez, Yu Gu, Yanbo Xu, Mu Wei, Wenhui Wang,´ Shuming Ma, Furu Wei, Jianwei Yang, Chunyuan Li, Jian feng Gao, Jaylen Rosemon, Tucker Bower, Soohee Lee, Roshanthi Weerasinghe, Bill J. Wright, Ari Robicsek, Brian Piening, Carlo Bifulco, Sheng Wang, and Hoifung Poon. A whole-slide foundation model for digital pathology from real-world data. Nature, 2024. 6

[45] Meilong Xu, Xiaoling Hu, Saumya Gupta, Shahira Abousamra, and Chao Chen. Semi-supervised segmentation of histopathology images with noise-aware topological con sistency. In ECCV, 2024. 1

[46] Meilong Xu, Xiaoling Hu, Shahira Abousamra, Chen Li, and Chao Chen. Match: Multi-faceted adaptive topo consistency for semi-supervised histopathology segmentation. In NeurIPS, 2025. 1

[47] Yingxue Xu and Hao Chen. Multimodal optimal transport based co-attention transformer with global structure consistency for survival prediction. In ICCV, 2023. 1, 3

[48] Hongrun Zhang, Yanda Meng, Yitian Zhao, Yihong Qiao, Xiaoyun Yang, Sarah E Coupland, and Yalin Zheng. Dtfdmil: Double-tier feature distillation multiple instance learning for histopathology whole slide image classification. In CVPR, 2022. 1, 3

[49] Jingwei Zhang, Anh Tien Nguyen, Xi Han, Vincent Quoc Huy Trinh, Hong Qin, Dimitris Samaras, and Mahdi S Hos seini. 2dmamba: Efficient state space model for image representation with applications on giga-pixel whole slide image classification. In CVPR, 2025. 3

[50] Yunlong Zhang, Honglin Li, Yuxuan Sun, Sunyi Zheng, Chenglu Zhu, and Lin Yang. Attention-challenging multiple instance learning for whole slide image classification. In ECCV, 2024. 1, 3

[51] Yilan Zhang, Yingxue Xu, Jianqi Chen, Fengying Xie, and Hao Chen. Prototypical information bottlenecking and disentangling for multimodal cancer survival prediction. In ICLR, 2024. 1, 3

[52] Fengtao Zhou and Hao Chen. Cross-modal translation and alignment for survival analysis. In ICCV, 2023. 1, 3

[53] Wenhui Zhu, Xiwen Chen, Peijie Qiu, Aristeidis Sotiras, Abolfazl Razi, and Yalin Wang. Dgr-mil: Exploring diverse global representation in multiple instance learning for whole slide image classification. In ECCV, 2024. 3

# Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning — Supplementary Material —

In this supplementary material, we provide additional technical derivations, experimental details, and qualitative analyses to complement the main manuscript. First, Section 6 presents the mathematical derivation of our Hierarchical Information Bottleneck objective. Subsequently, we introduce the details of our in-house ovarian dataset and the specialized evaluation tool developed for pathologist assessment in Section 7 and Section 8, respectively. We then elaborate on the implementation details of our two-stage training strategy in Section 9. To further demonstrate the model’s performance, Section 10 provides additional quantitative results on public benchmarks, followed by extensive qualitative visualizations on both public and private datasets in Section 11. A comprehensive ablation study investigating various training configurations and sampling distributions is presented in Section 12. Finally, we discuss the limitations of our current approach and suggest future research directions in Section 13.

## 6. Hierarchical IB Objective Derivation

## 6.1. Variational Information Bound

By definition, the mutual information $I ( A ; B )$ between two random variables A and B is:

$$
I ( A ; B ) = \mathbb { E } _ { p ( A , B ) } \left[ \log { \frac { p ( A \mid B ) } { p ( A ) } } \right]\tag{13}
$$

In practice, the true marginal distribution $p ( A )$ is typically intractable. To derive a computable bound, we introduce a variational prior $q ( A )$ . Utilizing the non-negativity of the KL divergence:

$$
D _ { \mathrm { K L } } ( p ( A ) \parallel q ( A ) ) = \mathbb { E } _ { p ( A ) } \left[ \log { \frac { p ( A ) } { q ( A ) } } \right] \geq 0\tag{14}
$$

it follows that:

$$
\mathbb { E } _ { p ( A ) } [ \log p ( A ) ] \geq \mathbb { E } _ { p ( A ) } [ \log q ( A ) ]\tag{15}
$$

Consequently, by substituting this inequality into the definition of $I ( A ; B )$ , we arrive at the variational upper bound for the compression term, following the VIB framework [2]:

$$
I ( A ; B ) \leq \mathbb { E } _ { p ( B ) } [ D _ { \mathrm { K L } } ( p ( A \mid B ) \parallel q ( A ) ) ]\tag{16}
$$

## 6.2. Hierarchical Variational Decomposition

As stated in the main text, the total compression term $I ( Z ; X \mid { \mathbf { q } } )$ is decomposed into group-level and patch-level terms using the chain rule for mutual information:

$$
I ( Z _ { g } , Z _ { p } ; X \mid { \bf q } ) = I ( Z _ { g } ; X \mid { \bf q } ) + I ( Z _ { p } ; X \mid Z _ { g } , { \bf q } )\tag{17}
$$

By applying the variational bound from Equation (16) to each hierarchical component, we derive the tractable hierarchical constraints for our selection process. For the group-level complexity, we introduce the variational posterior $p _ { \phi _ { g } } ( Z _ { g } \mid X , \mathbf { q } )$ and the group-level prior $p _ { g } ( Z _ { g } \mid \mathbf { q } )$ The mutual information term is then bounded by:

$$
I ( Z _ { g } ; X \mid \mathbf { q } ) \leq \mathbb { E } _ { \mathcal { D } } [ D _ { \mathrm { K L } } ( p _ { \phi _ { g } } ( Z _ { g } \mid X , \mathbf { q } ) \parallel p _ { g } ) ]\tag{18}
$$

Similarly, for the patch-level complexity, we introduce the conditional posterior $p _ { \phi _ { p } } ( Z _ { p } \mid X , Z _ { g } , \mathbf { q } )$ and its corresponding conditional prior $p _ { p } ( Z _ { p } \mid Z _ { g } , \mathbf { q } )$ . The mutual information term is then bounded by:

$$
\begin{array} { r l } & { I ( Z _ { p } ; X \mid Z _ { g } , \mathbf { q } ) \leq \mathbb { E } _ { \mathcal { D } } \Big [ \mathbb { E } _ { Z _ { g } \sim p _ { \phi _ { g } } } [ D _ { \mathrm { K L } } ( p _ { \phi _ { p } } ( Z _ { p } \mid X , Z _ { g } , \mathbf { q } ) } \\ & { \qquad \parallel p _ { p } ( Z _ { p } \mid Z _ { g } , \mathbf { q } ) ) ] \Big ] } \end{array}\tag{19}
$$

## 6.3. Deriving the Final Objective

For the relevance term $I ( Z ; Y \mid \mathbf { q } )$ , which measures the predictive power of the selected features for the answer $Y _ { \textrm { \textrm { \textrm { \textrm { \textrm { \textrm { F } } } } } } }$ we utilize the LLM as a variational decoder $p _ { \theta } ( Y \mid Z , \mathbf { q } )$ to obtain a tractable lower bound:

$$
I ( Z ; Y \mid { \mathbf q } ) \ge \mathbb { E } _ { \mathcal { D } } \mathbb { E } _ { Z \sim p _ { \phi } } [ \log p _ { \theta } ( Y \mid Z , { \mathbf q } ) ]\tag{20}
$$

By substituting the complexity upper bounds and the relevance lower bound into the original IB objective ${ \mathcal { L } } _ { \mathrm { I B } }$ , we arrive at a tractable training objective. In the standard VIB [2] formulation, a single hyperparameter β typically regulates the entire compression term. To better accommodate the hierarchical nature of WSIs and provide greater flexibility in hyperparameter tuning, we extend this formulation by assigning independent Lagrange multipliers, $\beta _ { g }$ and $\beta _ { p } ,$ as group-level and patch-level regularizers, respectively. This decoupling allows the model to independently regulate the information bottleneck at each granularity. The resulting HIB objective is formulated as:

$$
\begin{array} { r l } & { \mathcal { T } _ { \mathrm { H I B } } = \mathbb { E } _ { \mathcal { D } } \Big [ \mathbb { E } _ { Z \sim p _ { \phi } } [ \log p _ { \theta } ( Y \mid Z , \mathbf { q } ) ] } \\ & { \phantom { \mathcal { T } _ { \mathrm { H I B } } = \mathbb { E } _ { \mathcal { D } } \Big [ \mathbb { E } _ { Z } \sim p _ { \phi } \left( Z _ { g } \mid X , \mathbf { q } \right) \left. p _ { g } \right) } } \\ & { \phantom { \mathcal { T } _ { \mathrm { H I B } } = \mathbb { E } _ { \mathcal { D } } \mathbb { E } _ { Z _ { g } \sim p _ { \phi _ { g } } } [ D _ { \mathrm { K L } } \big ( p _ { \phi _ { p } } \left( Z _ { p } \mid X , Z _ { g } , \mathbf { q } \right) \mid \mid p _ { p } \big ) \big ] \Big ] } } \end{array}\tag{21}
$$

In practice, the nested expectation over $Z _ { g }$ in the patch-level term is empirically estimated through the group-level sampling process. By using the specific sampling rate made by the group sampler during the forward pass, the theoretical objective reduces to the empirical loss function presented in Equation (8) of the main text.

![](images/85585e893271b4dea913aa75e21f38684c3e84444528d2e9d1b5ef6bd9ee0a74.jpg)  
Figure 5. The user interface for the Tissue Segmentation Survey. The central area shows a side-by-side comparison of the original WS and the tissue segmentation result. The right legend clarifies the tissue classes, and the bottom section collects the pathologists’ rating.

## 7. In-house Ovarian Dataset

To demonstrate the generalizability of our proposed model, we curated a small-scale, in-house ovarian dataset. This dataset is compiled from WSIs of ovarian tissues and formatted into question-answer pairs, focusing on distinct histological phenotypes visible within the WSIs. The dataset includes four primary diagnostic categories, based on the observed tumor morphology. In total, the dataset comprises 375 question-answer pairs. The distribution of samples across the four categories is as follows: endometrioid (n = 81), clear cell carcinoma (n = 82), high grade serous carcinoma (n = 123), and serous borderline carcinoma (n = 89).

A typical question within the dataset is structured as a multiple-choice classification task based on visual features observed in the WSI. An example is provided below:

Example Question-Answer Pair: Based on the observed features, what do you think is the correct histological classification of the tumor?

(a) endometrioid

(b) clear cell carcinoma

(c) high grade serous carcinoma

(d) serous borderline carcinoma

## 8. Evaluation Tool

To verify the reliability of our tissue segmentation and ensure that the model selects question-relevant patches for inference, we developed an interactive survey tool based on Streamlit [1]. This evaluation is twofold: it primarily validates the accuracy of tissue segmentation, followed by an assessment of the patch selection performance. We detail these two components in the following sections.

Tissue Segmentation Verification. In the first part, we focus on the tissue segmentation results. As illustrated in Figure 5, the user interface features a side-by-side comparison view: the original WSI is displayed on the left, while the corresponding model-generated segmentation mask is shown on the right. To assist pathologists in interpreting the results, a color-coded legend is provided on the sidebar, mapping specific colors to tissue types (e.g., red for malignant tissue, yellow for stroma).

Pathologists are asked to verify the alignment between the WSI and the mask using these visual cues. Based on their inspection, they rate the segmentation quality via a radio button selection:

• Q1: “How accurate is the tissue segmentation?” (1 = Strongly Disagree, 5 = Strongly Agree)

After the user clicks “Submit Answer”, the tool automati-

![](images/68393b9337623c5102a56f3c4a54667af75193e99403bf1a27a36434eecf64b3.jpg)  
Figure 6. The user interface for the Patch Selection Survey. The view visualizes the “Before” and “After” states of patch selection. Note that the input question and ground truth answer are displayed below the images to provide necessary context for the pathologists’ evaluation.

cally logs the feedback and advances to the next sample.

Patch Selection Assessment. In the second part, the tool adapts to evaluate the model’s coarse-to-fine selection capability. As shown in Figure 6, we visualize the patches in both their “Before” and “After” selection states.

To ensure sufficient diagnostic context for evaluation, the specific “Question” (highlighted in blue) and the “Ground Truth Answer” (highlighted in green) are explicitly displayed below the images. Pathologists assess whether the model successfully removes irrelevant regions while retaining diagnostic information by answering the following:

• Q2.1: “Does the model filter out a significant number of question-irrelevant patches?”

• Q2.2: “Are the selected patches sufficient to answer the question?”

To conduct a robust evaluation, we randomly sampled a total of 30 cases, comprising 20 from the public dataset and 10 from the private dataset. Two independent pathologists were invited to perform the annotations using the developed tool. To ensure the reliability and objectivity of the assessment, we calculated the average scores from both annotators as the final metric for pathologist-based evaluation.

Table 7. Quantitative results on SlideBench-VQA (TCGA). Performance is evaluated using accuracy and macro-averaged metrics.
<table><tr><td></td><td colspan="4">Microscopy</td><td colspan="4">Diagnosis</td><td colspan="4">Clinical</td></tr><tr><td>Method</td><td> $\operatorname { A c c } .$ </td><td>Macro-P</td><td>Macro-R</td><td>Macro-F1</td><td> $\operatorname { A c c } .$ </td><td>Macro-P</td><td>Macro-R</td><td>Macro-F1</td><td> $\operatorname { A c c } .$ </td><td>Macro-P</td><td>Macro-R</td><td>Macro-F1</td></tr><tr><td>GPT-40</td><td>39.24</td><td>36.12</td><td>39.45</td><td>35.54</td><td>24.12</td><td>22.83</td><td>24.34</td><td>21.72</td><td>44.67</td><td>42.93</td><td>44.95</td><td>43.58</td></tr><tr><td>Quilt-LLaVA [32]</td><td>52.39</td><td>46.75</td><td>50.09</td><td>46.84</td><td>30.19</td><td>27.94</td><td>30.34</td><td>27.15</td><td>49.33</td><td>47.01</td><td>48.91</td><td>47.54</td></tr><tr><td>LLaVA-Med [20]</td><td>52.15</td><td>46.30</td><td>49.80</td><td>46.51</td><td>29.97</td><td>27.84</td><td>29.93</td><td>26.86</td><td>47.33</td><td>45.12</td><td>47.19</td><td>45.58</td></tr><tr><td>SlideChat [9]</td><td>83.15</td><td>81.77</td><td>78.50</td><td>79.70</td><td>71.36</td><td>71.80</td><td>65.22</td><td>67.52</td><td>75.33</td><td>73.84</td><td>72.74</td><td>72.98</td></tr><tr><td>Ours</td><td>84.62</td><td>80.79</td><td>80.47</td><td>80.33</td><td>73.09</td><td>72.10</td><td>68.08</td><td>69.22</td><td>77.30</td><td>73.97</td><td>74.24</td><td>73.94</td></tr></table>

Table 8. Quantitative results on WSI-Bench (Close-ended). Performance is evaluated using accuracy and macro-averaged metrics.
<table><tr><td></td><td colspan="4">Morphology</td><td colspan="4">Diagnosis</td><td colspan="4">Treatment (Binary)</td></tr><tr><td>Method</td><td>Acc.</td><td>Macro-P</td><td>Macro-R</td><td>Macro-F1</td><td>Acc.</td><td>Macro-P</td><td>Macro-R</td><td>Macro-F1</td><td>Acc.</td><td>Macro-P</td><td>Macro-R</td><td>Macro-F1</td></tr><tr><td>GPT-40</td><td>47.07</td><td>42.89</td><td>47.29</td><td>43.19</td><td>53.06</td><td>49.02</td><td>53.37</td><td>49.27</td><td>87.50</td><td>79.12</td><td>83.76</td><td>81.05</td></tr><tr><td>Quilt-LLaVA [32]</td><td>94.13</td><td>91.74</td><td>91.19</td><td>91.42</td><td>84.13</td><td>81.35</td><td>78.68</td><td>79.63</td><td>97.92</td><td>98.75</td><td>94.44</td><td>96.42</td></tr><tr><td>LLaVA-Med [20]</td><td>91.04</td><td>86.21</td><td>87.59</td><td>86.83</td><td>81.32</td><td>77.80</td><td>74.81</td><td>76.02</td><td>95.83</td><td>93.16</td><td>93.16</td><td>93.16</td></tr><tr><td>SlideChat [9]</td><td>91.34</td><td>86.11</td><td>87.92</td><td>86.97</td><td>82.15</td><td>79.15</td><td>75.58</td><td>77.02</td><td>93.75</td><td>88.68</td><td>91.88</td><td>90.16</td></tr><tr><td>Ours</td><td>94.57</td><td>92.66</td><td>91.34</td><td>91.85</td><td>85.79</td><td>85.03</td><td>79.45</td><td>81.70</td><td>97.92</td><td>95.00</td><td>98.72</td><td>96.72</td></tr></table>

## 9. Implementation Details

In the first stage, following Slidechat [9], the projector and slide encoder are set to be trainable while the remaining components are frozen. This stage utilizes the WSI-caption data for initial alignment, employing a learning rate of 1e-3 for 3 epochs. In the second stage, we train the entire model for 2 epochs with a learning rate of 1e-4 and a batch size of 1. Specifically, we apply LoRA to the LLM to ensure efficient parameter updates. For the hyperparameters for our group sampler and patch selector, we employ a linear warmup schedule for the first 5000 iterations. During this warmup, the $\beta _ { g }$ weight increases from 0 to 0.1, and the $\beta _ { p }$ weight increases from 0 to 0.2, then they are held constant.

![](images/756281c36f86bf25087fe4e27a1ed15b464fa9f694be3f0c737c5e349422aa68.jpg)

![](images/9c382dd95bf0852784770e4b902b3ce6acded7d5588748c11cfe4da522b49a44.jpg)  
Figure 7. The sample distribution for the test set.

## 10. Additional Quantitative Results.

Figure 7 illustrates the sample distribution across different task categories in SlideBench-VQA and WSI-Bench. To further validate the reliability of HistoSelect, we report detailed macro-averaged metrics (Macro-Precision, Macro-Recall, and Macro-F1) across these two major benchmarks. As shown in Table 7 and Table 8, our method consistently achieves superior performance in these balanced metrics on both SlideBench-VQA and WSI-Bench. These improvements demonstrate that our approach effectively identifies task-relevant patches and generalizes well across various categories.

## 11. Additional Qualitative Results

To provide a more comprehensive evaluation of our proposed method, we present additional qualitative visualizations in this section. Due to the page constraints of the main manuscript, we extend our analysis here to demonstrate the model’s performance across different data distributions. Specifically, we visualize the effectiveness of our question-aware selection mechanism on both the public TCGA dataset and an in-house Ovarian dataset. These results further validate that our method can consistently filter out background noise and diagnostically irrelevant patches, enabling the model to focus efficiently on the regions most related to the VQA query.

## 11.1. Results on Public Dataset

Figure 8 showcases the visualization results on the public TCGA dataset from WSI-Bench [22]. Consistent with the findings in the main text, our model demonstrates strong generalization capabilities. It successfully identifies and retains key histological features required to answer the question while discarding a significant portion of irrelevant and redundant patches, thereby ensuring that the downstream reasoning is primarily driven by the most informative patches.

(c) Before Selection  
(b) Tissue Segmentation  
(a) Original WSI  
![](images/a42ccb07aef9329c2b3e198e9355aa09632148e7c542b362bb99e4ea2d130868.jpg)  
Figure 8. Additional visualization of the selection process on the public WSI-Bench dataset. (a) Original WSI. (b) The tissue segmentation mask. (c) A visualization of candidate patches extracted from tissue regions prior to selection. (d) The sparse set of patches retained by our model. As observed in (d), the model effectively suppresses irrelevant regions, focusing the attention solely on the informative patches required for the VQA task.

![](images/6385aa4de89d080a7c134c03ed63e6db3e909ea19530c9a46dad5a8ee36d0f5a.jpg)  
Figure 9. Visualization of the selection process on the private Ovarian dataset. The figure follows the same pipeline as the main manuscript and Figure 8: (a) Original WSI. (b) Tissue segmentation mask. (c) Patches before selection. (d) Patches after selection. These results demonstrate the robustness of our method against domain shifts common in private clinical data. The model successfully filters out noninformative tissue, preserving only the regions essential for accurate question answering.

## 11.2. Results on Private Dataset

To assess the robustness of our model in a real-world clinical setting, Figure 9 illustrates the selection process on our private ovarian dataset. Despite potential domain shifts such as variations in staining protocols and scanner properties compared to the public dataset, our question-aware selector maintains high precision. It effectively selects informative regions relevant to the query, verifying the method’s applicability to proprietary clinical workflows.

## 11.3. Sampling Rate Distribution Analysis

To investigate the sampling rate distribution across different questions, we conducted a quantitative analysis on the Diagnosis and Morphology subset of the WSI-Bench dataset [22]. Specifically, we represented the tissue group sampling rate for each question as a 13-dimensional vector, where each dimension corresponds to the normalized sampling rate of a specific tissue component. We then applied K-means clustering to these vectors and visualized the resulting groupings using t-SNE, as shown in Figure 11. The emergence of four distinct clusters (K = 4) demonstrates that our model generates diverse and structured sampling patterns in the feature space. This grouping behavior confirms that the selection mechanism effectively navigates the complex composition of WSIs by adaptively prioritizing different histological tissue types based on the semantic focus of the question, rather than collapsing into a fixed, question-agnostic distribution.

![](images/8e6eb9b8cacea91c9d0cb262e640dfe3572ed511ec5712bf73ff0f15bd01d7eb.jpg)  
Figure 10. Mean sampling rate distribution bar charts for identified clusters. We report the average 13-dimensional sampling vectors for the four clusters discovered in Fig. 11. Each cluster exhibits a unique sampling rate pattern.

![](images/a6bb414e2dda415225f8af6954b9e206c8415c7c21271aad15cd5ea1eda2fd19.jpg)  
Figure 11. Visualization of question-aware sampling distributions. We visualize the 13-dimensional sampling rate vectors from the WSI-Bench Diagnosis and Morphology test set using t-SNE.

The cluster-specific sampling distributions, as visualized in Fig. 10, illustrate that our model develops distinct, taskdriven sampling patterns. Each cluster corresponds to a specific clinical focus in the questions:

• Cluster 0 (Tumor Classification): Prioritizes malignant tissue, aligning with questions focused on histological tumor grading and classification.

Example: Based on the observed features, what do you think is the correct histological classification of the tumor? A) Adenocarcinoma B) Small cell carcinoma C) Squamous cell carcinoma D) Large cell carcinoma

• Cluster 1 (Cellular Morphology): Shifts focus toward smooth muscle and stromal components, providing the necessary context for evaluating cellular variability and mitotic activity.

Example: What are the notable features of the cellular morphology in this slide? A) Nuclei are uniform in appearance, showing no signs of active division. B) There is minimal variability in nuclear size, with a low rate of cell division. C) Nuclei appear extremely pleomorphic, with a very high rate of mitotic activity. D) There is moderate variability in nuclear size and shape, with a moderate rate of cell division and presence of single cells.

• Cluster 2 (Tissue Architecture): Shows a strong preference for benign tissue and surrounding structures, which is essential for assessing the overall microanatomy and glandular patterns.

Example: What observations can you make about the tissue architecture on this slide? A) The tumor forms wellorganized acinar structures with a clear glandular pattern. B) The tumor is characterized by prominent chickenwire vasculature providing stroma. C) Tumor cells create extensive solid sheets, with a completely homogeneous pattern. D) The tissue maintains normal microanatomy with minimal deviation.

• Cluster 3 (Tumor Infiltration): Concentrates on lymphocytes and extracellular components, capturing the critical interface where tumor cells infiltrate the stroma and adipose layers.

Example: What is the observed pattern of tumor infil tration in this specimen? A) Tumor cells are limited to the submucosal layer without muscularis propria involvement. B) Tumor cells infiltrate the stroma, extending into the muscularis propria and adipose tissue. C) Tumor cells remain within glandular structures without stromal invasion. D) There is only infiltration into the adipose tissue, sparing the submucosal layer.

This clear divergence confirms that our selection mechanism is question-aware. Instead of relying on a static saliency map, the model dynamically re-prioritizes different histological tissue types based on the semantic intent of the question, ensuring that the most relevant patches are selected for each specific question.

## 12. Ablation Study

In this section, we conduct extensive ablation studies to evaluate the effectiveness of the proposed components in HistoSelect. We first analyze the impact of hyperparameters $\beta _ { g }$ and $\beta _ { p }$ in our loss function, which control the information bottleneck at the group sampler and patch selector levels, respectively. Subsequently, we investigate the influence of different training strategies and assess the modelagnostic generalization of our selector modules across various base models. Finally, a group selection analysis is performed to demonstrate the critical role of group-level selection in handling complex multi-tissue reasoning tasks.

<table><tr><td> $\beta _ { g }$ </td><td>Morphology</td><td>Diagnosis</td><td>Treatment</td></tr><tr><td>0</td><td>91.78</td><td>81.82</td><td>95.83</td></tr><tr><td>0.1</td><td>93.39</td><td>84.13</td><td>95.83</td></tr><tr><td>0.2</td><td>94.57</td><td>85.79</td><td>97.92</td></tr><tr><td>0.3</td><td>93.10</td><td>83.25</td><td>93.75</td></tr></table>

Table 9. Ablation Study on the weight $\beta _ { g }$ for the group sampler.

Impact of $\beta _ { g }$ for the Group Sampler. Table 9 reports the model performance under different values of $\beta _ { g } ~ \in$ $\{ 0 , 0 . 1 , 0 . 2 , 0 . 3 \}$ while keeping $\beta _ { p }$ fixed. We observe that when $\beta _ { g } = 0$ , the group sampler lacks the necessary regularization to filter out irrelevant tissue groups, leading to a lower signal-to-noise ratio and suboptimal performance. As $\beta _ { g }$ increases to 0.2, the model effectively suppresses background noise at the group level, achieving the best overall accuracy across all tasks. However, further increasing $\beta _ { g }$ to 0.3 results in a performance drop. This suggests that an overly aggressive penalty causes the sampler to excessively reduce the sampling rate of tissue groups that contain necessary contextual information.

<table><tr><td> $\beta _ { p }$ </td><td>Morphology</td><td>Diagnosis</td><td>Treatment</td></tr><tr><td>0</td><td>92.07</td><td>81.32</td><td>93.75</td></tr><tr><td>0.05</td><td>93.25</td><td>84.23</td><td>95.83</td></tr><tr><td>0.10</td><td>94.57</td><td>85.79</td><td>97.92</td></tr><tr><td>0.15</td><td>93.83</td><td>83.74</td><td>95.83</td></tr></table>

Table 10. Ablation study on the weight $\beta _ { p }$ for the patch selector.

Impact of $\beta _ { p }$ for the Patch Selector. Table 10 examines the effect of the patch-level weight $\beta _ { p } \in \{ 0 , 0 . 0 5 , 0 . 1 0 , 0 . 1 5 \}$ Similar to the group level, setting $\beta _ { p } = 0$ tends to retain a large number of redundant patches, which introduces potential interference for the answer predictor. We find that setting $\beta _ { p } ~ = ~ 0 . 1 0$ yields the optimal balance, allowing the model to identify the most distinct and discriminative patches without losing critical information. Conversely, setting $\beta _ { p }$ too high (e.g., 0.15) leads to over-pruning, where the model is penalized for retaining informative patches, causing a loss of fine-grained details essential for accurate diagnosis and treatment prediction.

<table><tr><td>Training Strategy</td><td>Morphology</td><td>Diagnosis</td><td>Treatment</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>Joint Training</td><td>94.57</td><td>85.79</td><td>97.92</td><td>92.76</td></tr><tr><td>Joint + Patch Selector</td><td>94.71</td><td>85.95</td><td>97.92</td><td>92.86</td></tr><tr><td>Joint + Group Sampler</td><td>95.15</td><td>86.11</td><td>97.92</td><td>93.06</td></tr></table>

Table 11. Ablation on training strategies.

Impact of Training Strategies. Beyond hyperparameter tuning, we investigate whether extending the training procedure further impacts performance. As shown in Table 11, while the initial joint training yields strong results, conducting an additional epoch of training specifically for the sampler or selector modules proves beneficial. In particular, performing one extra epoch for the group sampler achieves the highest performance across most tasks. This suggests that after the joint training stage has established a solid foundation, the group sampler can further benefit from a dedicated optimization phase.

Ablation with Different Base Models. To verify the model-agnostic effectiveness of HistoSelect, we conduct an experiment using Gemini 3 Flash as a frozen reasoning engine. Specifically, we randomly sample 200 cases from WSI-Bench and compare two input strategies: (1) a baseline using 100 randomly sampled patches with the question, and (2) using the top-100 patches selected by HistoSelect with the same question. As shown in Table 12, our method yields a consistent performance boost even for a stronger foundation model. This demonstrates our model’s ability to filter redundant noise and identify task-relevant tokens independently of the base model’s reasoning capacity.

<table><tr><td>Method</td><td>ACC</td><td>Macro-P</td><td>Macro-R</td><td>Macro-F1</td></tr><tr><td>Gemini 3 Flash</td><td>59.5</td><td>57.0</td><td>60.0</td><td>57.0</td></tr><tr><td>Gemini 3 Flash + HistoSelect</td><td>62.5</td><td>58.0</td><td>64.0</td><td>59.0</td></tr></table>

Table 12. Performance comparison on WSI-Bench (200 samples).

Impact of Group Selection. We further conduct an analysis by comparing our base model with a version using “Ideal Group Selection” (i.e., perfect identification of relevant tissue regions). As reported in Table 13, better group selection consistently leads to higher performance. Notably, the gain is more significant for multi-tissue questions (e.g., tumor infiltration patterns) compared to single-tissue ones (e.g., tumor detection). This highlights that the group sampler is particularly essential for handling complex clinical reasoning that requires cross-tissue contextual integration.

<table><tr><td>Question Type</td><td>Base</td><td>Base + Ideal Group</td><td>Gain (∆)</td></tr><tr><td>Single-tissue (Easy)</td><td>85.0</td><td>87.0</td><td>+2.0</td></tr><tr><td>Multi-tissue (Hard)</td><td>73.0</td><td>79.0</td><td>+6.0</td></tr></table>

Table 13. Ablation on group selection.

## 13. Limitation

While our proposed method demonstrates promising results and improved efficiency in histopathology VQA, we acknowledge several limitations that outline directions for future research.

Evaluation on Other Datasets. First, our current experimental validation primarily focuses on the TCGA dataset and our in-house private dataset. While this covers a significant amount of variation, the heterogeneity of pathological data across different organs and scanning protocols is vast. To further verify the generalizability of our model, we intend to extend our training and testing to other large-scale public datasets, such as the BCNB (Early Breast Cancer Core-Needle Biopsy) [43] dataset. Evaluating on such diverse cohorts will help ensure our method remains robust across different cancer subtypes and data distributions.

Lack of Explicit Textual Reasoning. Second, while our method offers visual interpretability by highlighting the selected question-relevant patches, it does not currently generate explicit textual explanations justifying why these patches were selected. Providing a natural language rationale alongside the final VQA answer would further enhance trust in clinical decision-support systems. We aim to explore the integration of LLMs more deeply in future iterations to bridge this gap between visual attention and semantic reasoning.