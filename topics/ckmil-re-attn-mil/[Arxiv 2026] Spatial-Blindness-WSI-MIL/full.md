# Spatial Blindness in Whole-Slide Multiple Instance Learning

Xiangyu Li College of Intelligence and Computing Tianjin University Tianjin, China xiangyuli@tju.edu.cn

Ran Su<sup>∗</sup> College of Intelligence and Computing Tianjin University Tianjin, China ran.su@tju.edu.cn

## Abstract

Whole-slide MIL models are often called context-aware once graphs, Transformers, or state-space modules are placed above patch embeddings. We show that this label can be deceptive. On pathology tasks where tissue architecture is part of the diagnostic signal, several strong MIL baselines retain nearly unchanged slidelevel AUC after patch coordinates are permuted. Their predictions are accurate, but largely compositional. We refer to this failure mode as spatial blindness. Our explanation is optimization-based: dense appearance statistics are learned early under slide-level supervision, leaving weak gradients for sparse spatial relations. ResTopoMIL addresses the issue by first fitting a permutation-invariant prototype histogram and then freezing it while a lightweight graph branch learns the residual under a coordinate-shuffling constraint. The architecture is simple by design; the intervention is in how the spatial branch is trained. Across 9 public WSI benchmarks, ResTopoMIL improves classification and survival prediction with 1.15M parameters, restores sensitivity to coordinate perturbation, and gives stronger localization evidence on CAMELYON-16.

## 1 Introduction

Computational pathology increasingly learns diagnostic and prognostic models directly from digitized whole-slide images (WSIs), where cellular morphology, tissue architecture, and long-range context are preserved at gigapixel scale [Pantanowitz et al., 2011, Verghese et al., 2023, Song et al., 2023]. The weak supervision problem is severe: a slide may contain tens of thousands of relevant and irrelevant regions, but most clinical datasets provide only a slide-level label. Multiple instance learning (MIL) is therefore the standard formulation for WSI analysis [Dietterich et al., 1997, Maron and Lozano-Pérez, 1997, Campanella et al., 2019, Lu et al., 2021]: patches are treated as instances, the slide as a bag, and the model predicts the bag label without patch-level annotation.

MIL works remarkably well in pathology, but its success can hide a limitation. Many labels are not determined by local appearance alone. Gleason grading depends on gland formation; breast cancer subtyping distinguishes ductal growth from single-file strands; prognosis often reflects invasive fronts, immune aggregates, papillary cores, solid growth, or tumor-stroma organization [Quail and Joyce, 2013, Cheng et al., 2021, Wang et al., 2021]. These are spatial statements. They depend not just on which tissue components appear, but on how they are arranged.

![](images/96c6ce68f96152ed47fbcf9987b31bd590d9823b7346983151900d5ed0271c70.jpg)  
Figure 1: The ResTopoMIL Concept. (a) A standard MIL model may give similar predictions before and after spatial permutation, indicating that it mainly uses composition. (b) ResTopoMIL separates the problem into a statistical stream and a topological stream. (c) The statistical stream provides a base prediction, while the topological stream learns a residual correction from spatial organization.

At first glance, recent context-aware MIL methods should address this issue. Graph networks, Transformers, hierarchical models, and state-space models all process a slide as more than an unordered bag [Chen et al., 2021b, Pati et al., 2022, Adnan et al., 2020, Vaswani et al., 2017, Shao et al., 2021, Chen et al., 2022, Gu et al., 2021, Yang et al., 2024, Zhang et al., 2025a]. Architecture alone, however, does not tell us what the trained predictor uses. We use a simple stress test: keep every patch embedding fixed, and randomly permute the coordinates used to build spatial context. On tasks where architecture is label-relevant, a topology-using model should suffer. Several strong context aware baselines barely move. They have spatial machinery, but their learned decision rules behave much like bag-of-visual-words classifiers.

The question is then not whether a model contains a spatial operator, but why such an operator can remain unused. Our explanation is optimization-based. Tissue composition provides a dense, early signal: many patches contribute to the same slide label. Topological evidence is sparser and harder to align with slide-level supervision. Under joint training, the network can reduce the loss by fitting composition first; once that happens, little useful gradient remains for the spatial branch. We call this behavior optimization laziness, in the descriptive sense that the easiest explanatory signal dominates training while the harder structural signal is left undertrained. The phenomenon is related to simplicity bias, texture bias, and gradient starvation [Shah et al., 2020, Geirhos et al., 2018, Noroozi and Favaro, 2016, Pezeshki et al., 2021].

ResTopoMIL follows this diagnosis. Rather than asking one network to discover composition and topology at the same time, it learns the compositional explanation explicitly. A permutationinvariant statistical stream is trained first and frozen. A lightweight graph stream is then trained on the residual, with a shuffle-based loss that asks it to distinguish real tissue topology from coordinatepermuted topology. The graph module is intentionally small. The point is not to add a heavier context block, but to change the training problem faced by the spatial branch.

This shifts the evaluation away from architecture labels. “Context-aware” should mean that a model’s decision changes when clinically relevant spatial organization is removed. We therefore test topology-destroying coordinate perturbations, separate pure composition from pure topology in a controlled benchmark, and ask whether residual training changes both accuracy and spatial behav ior. Topology-preserving transformations are discussed only as expected graph invariances, not as an additional measured benchmark.

The paper makes four contributions:

• Spatial blindness is defined and tested as insensitivity to coordinate perturbations on structure-dependent MIL tasks.

• A controlled composition–topology diagnostic benchmark shows that strong MIL models can solve compositional tasks while failing on pure topology.

• ResTopoMIL learns composition first and topology as a residual correction, with design analysis deferred to the appendix.

• Experiments on 9 public pathology benchmarks show gains in prediction, spatial sensitivity, and localization quality with a compact 1.15M-parameter model.

## 2 Related Work

MIL has a long history as a weak-supervision framework for learning from bag labels without instance labels [Dietterich et al., 1997, Maron and Lozano-Pérez, 1997]. In pathology, this formulation is natural because WSIs are too large for end-to-end pixel-level training and dense annotation is expensive. Early WSI approaches used patch classifiers, clustering, global pooling, or multi-view CNN aggregation to move from local tiles to slide-level labels [Xu et al., 2014, Kraus et al., 2016, Das et al., 2017, 2018, Wang et al., 2019, Chen et al., 2021a]. Attention-based MIL then became a central baseline because it learns instance weights while remaining permutation-invariant and interpretable [Ilse et al., 2018, Lu et al., 2021]. Later methods select critical instances, split bags, mine hard examples, regularize attention, or improve instance-level classifiers [Li et al., 2021, Zhang et al., 2022, Tang et al., 2023, Lin et al., 2023, Zhang et al., 2024, 2025b, Qu et al., 2024, Shao et al., 2025, Zhu et al., 2025, 2023].

A second line goes beyond unordered aggregation by modeling context among patches. Graph models treat WSIs as point clouds or tissue graphs [Chen et al., 2021b, Adnan et al., 2020, Pati et al., 2022, Pal et al., 2022]; Transformer and low-rank attention models capture long-range bag dependencies [Vaswani et al., 2017, Shao et al., 2021, Xiong et al., 2021, Chen et al., 2022, Xiang and Zhang, 2023]; hierarchical and context-aware variants exploit WSI pyramids [Zhang et al., 2021, Guo et al., 2023, Buzzard et al., 2024, Tran et al., 2025, Fourkioti et al., 2024, Chen et al., 2024]; and state-space models provide efficient long-sequence modeling [Gu et al., 2021, Yang et al., 2024, Zhang et al., 2025a]. These methods matter because pathology is not i.i.d.; structured MIL already recognizes that bags with the same instances can have different labels when dependencies differ [Zhou et al., 2009, Zhang et al., 2011]. We ask a complementary question: after adding such modules, does the learned predictor rely on topology, or does it still mainly count visual words?

Foundation models have greatly improved pathology patch embeddings [Chen et al., 2023, Kang et al., 2023, Oquab et al., 2023, Caron et al., 2021, Lu et al., 2023, Kapse et al., 2025], while visionlanguage and prompt-based WSI methods broaden slide supervision [Shi et al., 2024, Han et al., 2025, Wong et al., 2025, Tomar et al., 2025, Gou et al., 2025]. These advances make the question more urgent: strong features can make compositional shortcuts easier to exploit. ResTopoMIL therefore fixes the UNI encoder for all methods and studies the slide-level aggregation problem.

Finally, the diagnosis is related to shortcut learning and optimization bias: neural networks often prefer simple, high-variance, or dense predictive features even when structured features are available [Shah et al., 2020, Geirhos et al., 2018, Pezeshki et al., 2021]. ResTopoMIL makes the compositional signal explicit through a prototype/statistical stream related to Deep Sets, histograms, and morphological prototypes [Zaheer et al., 2017, Peeples et al., 2021, Song et al., 2024, Wei et al., 2016], then trains topology as residual information. The information-theoretic connection [Shannon, 1948, Cover, 1999] is used as design analysis in Section 4.4 and Appendix B.

## 3 Preliminaries and Motivation

A WSI is written as a bag

$$
{ \cal X } = \{ ( { \bf h } _ { i } , { \bf p } _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{1}
$$

where $\mathbf { h } _ { i } \in \mathbb { R } ^ { d }$ is a patch embedding and $\mathbf { p } _ { i } \in \mathbb { R } ^ { 2 }$ is its slide coordinate. We distinguish two sources of label information. Composition is the empirical distribution of patch appearances, for example the abundance of tumor-like or stromal patches. Topology is the spatial relation among patches: clustering, gland formation, boundaries, invasive fronts, and similar architectural cues.

The distinction matters because the two signals optimize differently. Composition is dense: every patch contributes to a histogram-like summary, and slide labels are often partly predictable from the prevalence of visual phenotypes. Topology is sparse: a small set of spatial relations may carry the decisive evidence, and supervision arrives only after whole-slide aggregation. A jointly trained model is naturally drawn to the dense signal first. Once the loss has fallen, too little error may remain to train the spatial pathway. Standard validation can then look reassuring even when the explanation ignores architecture.

To test whether a model uses topology, a coordinateshuffling operator π keeps {h<sub>i</sub>} fixed and permutes {p<sub>i</sub>}. This preserves composition but destroys adjacency and tissue architecture. A model is defined as spatially blind on a structure-dependent task if its prediction or performance is nearly invariant under this perturbation. Robustness is desirable when a perturbation preserves semantics; here the perturbation removes part of the diagnostic evidence.

![](images/aadd2167d33d0a9b8901b64e75292da39bf34ea1682d3fd57b07d049bbe7391e.jpg)

Figure 2 shows the motivating observation. Trans-MIL has contextual machinery, and DS-MIL is a strong dual-stream MIL baseline; both show little AUC change after coordinate shuffling. This is not a failure of prediction. It is evidence that high slidelevel AUC can be obtained from composition alone. The controlled benchmark in Section 5.2 makes the separation explicit: strong MIL models solve a purecomposition task but fail when the label is defined only by a spatial motif.

Figure 2: A coordinate-shuffling stress test. Patch embeddings are fixed while coordinates are permuted. On TCGA-BRCA, several MIL models remain almost unchanged after complete spatial shuffling. ResTopoMIL is more sensitive to this topology-destroying perturbation, as expected when structure is labelrelevant.

The stress test does not imply that every pathology task must depend on topology. Some tasks are primarily compositional, and a permutation-invariant model may be the right tool. Our claim is narrower: when the label is known to depend on architecture, a model should not be invariant to a perturbation that destroys architecture while preserving patch appearances. This is the sense in which we use spatial blindness.

This motivates the additive view

$$
F ( X ) \approx F _ { s t a t } ( \{ \mathbf { h } _ { i } \} ) + F _ { t o p o } ( \{ ( \mathbf { h } _ { i } , \mathbf { p } _ { i } ) \} ) ,\tag{2}
$$

where $F _ { s t a t }$ captures permutation-invariant composition and $F _ { t o p o }$ captures the remaining structuredependent signal. The difficulty is not only architectural. If both terms are learned jointly, the first term is often easier to optimize and can reduce the loss before the second term learns. ResTopoMIL therefore makes Eq. (2) operational through staged residual training.

## 4 ResTopoMIL

ResTopoMIL first learns the compositional explanation and then trains topology as a residual correction (Figure 3). The design does not hide the problem behind a larger spatial module. If spatial blindness is an optimization problem, increasing graph capacity can leave the shortcut intact. We instead use a strong permutation-invariant stream as the compositional anchor and train the graph stream on the remaining error. The residual representation is then checked against a perturbation that preserves patch appearance but destroys coordinates, so the graph branch cannot quietly become another texture aggregator.

![](images/093b45899472d63690a48286d614aa103e04981da9c15c9450f6e2e3859ff186.jpg)  
Figure 3: Overview of the ResTopoMIL Framework. The architecture decouples WSI analysis into two parallel streams. Top: The Statistical Stream captures tissue composition via a learnable prototype-based soft histogram, providing a statistical baseline. Bottom: The Topological Stream models spatial structure using a simple GNN. To prevent degeneration, ResTopoMIL introduces a Structure-Aware Texture Loss $( \mathcal { L } _ { t e x t u r e } )$ that forces the model to distinguish between genuine tissue topology $\left( Z _ { c l e a n } \right)$ and spatially shuffled noise $( Z _ { s h u f f l e d } )$ . The final prediction is the residual summation of both streams.

## 4.1 Statistical Anchor

The first stream ignores coordinates by construction. Its role is to absorb the label signal explained by tissue composition, so the second stream is not rewarded for relearning the same shortcut. We use a soft prototype histogram: richer than mean pooling, but still permutation-invariant. Let $C =$ $\{ \mathbf { c } _ { k } \} _ { k = 1 } ^ { K }$ be a learnable codebook initialized by sampled MiniBatch K-Means on randomly sampled training patch embeddings, rather than by exact K-Means over all WSI patches. For each patch embedding h<sub>i</sub>, the assignment is

$$
a _ { i k } = \frac { \exp ( - \| { \bf h } _ { i } - { \bf c } _ { k } \| ^ { 2 } / \tau ) } { \sum _ { j = 1 } ^ { K } \exp ( - \| { \bf h } _ { i } - { \bf c } _ { j } \| ^ { 2 } / \tau ) } , \qquad { \bf a } _ { i } = [ a _ { i 1 } , \dots , a _ { i K } ] ^ { \top } , \qquad { \bf z } _ { s t a t } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \bf a } _ { i } .\tag{3}
$$

An MLP maps $\mathbf { z } _ { s t a t }$ to logits $f _ { s t a t }$ . Similar to codebook-based MIL encodings [Wei et al., 2016], this stream measures how far one can go by counting visual phenotypes alone. Soft assignment avoids the brittleness of hard clustering, while remaining more expressive than mean pooling. A good anchor is useful scientifically as well as computationally: if the residual branch improves over it, the gain is less easily dismissed as a better bag-level composition classifier.

The anchor also gives a concrete diagnostic baseline. On a task where tissue proportion is sufficient, the residual branch should have little to add. On a task where architecture matters, the remaining error after $f _ { s t a t }$ is precisely where gland formation, invasive fronts, or tumor-stroma organization can enter. For this reason, the graph branch is not trained as a second full classifier.

## 4.2 Topological Residual Branch

The second stream receives the same patch embeddings together with a spatial graph. A KNN graph $\mathcal { G } = ( \nu , \mathcal { E } )$ is built from coordinates $\mathbf { p } _ { i }$ and processed by a two-layer GCN:

$$
\mathbf { H } ^ { ( l + 1 ) } = \sigma \left( \tilde { \mathbf { D } } ^ { - \frac { 1 } { 2 } } \tilde { \mathbf { A } } \tilde { \mathbf { D } } ^ { - \frac { 1 } { 2 } } \mathbf { H } ^ { ( l ) } \mathbf { W } ^ { ( l ) } \right) ,\tag{4}
$$

where $\tilde { \mathbf { A } } = \mathbf { A } + \mathbf { I }$ and $\mathbf { H } ^ { ( 0 ) }$ is the matrix of patch embeddings. We use global mean pooling to obtain the graph-level representation

$$
{ \bf z } _ { t o p o } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \bf H } _ { i } ^ { ( 2 ) } , \qquad f _ { t o p o } = { \bf W } _ { t o p o } { \bf z } _ { t o p o } + { \bf b } _ { t o p o } .\tag{5}
$$

The graph is built from physical coordinates rather than feature similarity: two tumor patches can look similar while belonging to different glands or invasive fronts, whereas neighboring patches define local architecture. The branch is intentionally simple. A deeper graph model would make it harder to tell whether improvement comes from topology or from capacity; the two-layer GCN leaves the optimization question visible.

## 4.3 Residual Training Objective

Training both streams from scratch reintroduces the competition that caused spatial blindness. ResTopoMIL uses two stages. Stage 1 trains only the statistical stream with standard cross-entropy, producing $f _ { s t a t }$ . Stage 2 freezes this stream and optimizes the topological branch through the combined logits

$$
f ( X ) = \mathrm { s g } [ f _ { s t a t } ( X ) ] + f _ { t o p o } ( X ) ,\tag{6}
$$

where sg[·] stops gradients. The graph branch learns corrections to a fixed compositional predictor rather than acting as an independent classifier. The stop-gradient has two effects: the statistical stream cannot absorb the errors exposed in Stage 2, and the topological stream receives a stable residual target rather than a moving joint optimum. The residual logits can be read as structured corrections–positive when topology supports the statistical prediction, negative when spatial organization contradicts it.

To ensure that this residual is genuinely spatial, ResTopoMIL adds a shuffle-based constraint. Let $\tilde { X }$ be obtained by permuting coordinates while keeping all patch embeddings fixed. This operation preserves composition and destroys topology. Let $\tilde { \mathbf { z } } _ { t o p o }$ denote the graph-level representation computed from this shuffled-coordinate view, and let sim $( \cdot , \cdot )$ be cosine similarity. The graph representation of X is required to differ from that of $\tilde { X }$

$$
\mathcal { L } _ { t e x t u r e } = \operatorname* { m a x } ( 0 , m - [ 1 - \mathrm { s i m } ( \mathbf { z } _ { t o p o } , \tilde { \mathbf { z } } _ { t o p o } ) ] ) .\tag{7}
$$

This loss is not a generic contrastive regularizer. Its negative view keeps all patch appearances unchanged, all labels unchanged, and all bag-level composition unchanged; only the coordinateinduced graph is corrupted. Satisfying the margin therefore encourages the branch to encode spatial arrangement rather than another appearance summary. This is also why the loss does not require tumor masks or pathologist-drawn regions. The final Stage-2 objective is

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { c l s } ( \mathrm { s g } [ f _ { s t a t } ] + f _ { t o p o } , Y ) + \lambda \mathcal { L } _ { t e x t u r e } .\tag{8}
$$

Thus the statistical stream explains composition, the residual branch explains what remains, and the auxiliary loss prevents the residual branch from becoming another permutation-invariant texture encoder.

At inference, no shuffled view is constructed. The model computes the statistical logits and topological residual logits once, then sums them. The method changes the training signal, not the deployment protocol: no patch-level annotations, topology labels, or test-time augmentations are required.

## 4.4 Why Decoupling Helps

The analysis in Appendix B is not a new general theory of gradient starvation or mutual information. Its purpose is narrower: to show why the three design choices in ResTopoMIL–a statistical anchor, stop-gradient residual training, and coordinate shuffling–belong together.

Proposition 1 (Residual-error gating of the topological update). For an additive MIL logit $f =$ $f _ { s t a t } + f _ { t o p o }$ trained with cross-entropy, the topological update can be written as

$$
\hat { p } _ { \theta } ( X ) = P _ { \theta } ( Y = 1 \mid X ) = \sigma ( f ( X ) ) , \qquad \nabla _ { \theta _ { t } } \mathcal { L } = \mathbb { E } [ ( \hat { p } _ { \theta } ( X ) - Y ) \nabla _ { \theta _ { t } } f _ { t o p o } ( X ) ] .\tag{9}
$$

Hence its norm is bounded by the remaining prediction error after the current model has used the statistical shortcut:

$$
\| \nabla _ { \theta _ { t } } \mathcal { L } \| \leq \left( \mathbb { E } ( \hat { p } _ { \theta } ( X ) - Y ) ^ { 2 } \right) ^ { 1 / 2 } \big ( \mathbb { E } \| \nabla _ { \theta _ { t } } f _ { t o p o } ( X ) \| _ { F } ^ { 2 } \big ) ^ { 1 / 2 } .\tag{10}
$$

This proposition is deliberately modest, but it is the useful part for our setting. If composition quickly reduces the residual error, the graph branch can be present yet receive little informative supervision. Freezing $f _ { s t a t }$ changes the learning problem: the error left by the statistical anchor becomes a stable target for the topological branch instead of a moving target shared by both branches.

Proposition 2 (Residual branch as conditional label information). After $Z _ { s t a t }$ and $f _ { s t a t }$ are fixed, optimizing the Stage-2 decoder

$$
q _ { \phi } ( Y = y \mid Z _ { s t a t } , Z _ { t o p o } ) = [ \mathrm { S o f t m a x } ( f _ { s t a t } + f _ { t o p o , \phi } ) ] _ { y }\tag{11}
$$

minimizes a variational upper bound on $H ( Y \mid Z _ { s t a t } , Z _ { t o p o } )$ and therefore maximizes a lower bound on the additional label information carried by topology, $I ( Z _ { t o p o } ; \dot { Y } \mid Z _ { s t a t } )$ . In the binary derivation, the same statement uses the sigmoid probability $\sigma ( f _ { s t a t } + \mathrm { \bar { \it f } } _ { t o p o , \phi } ) f o r Y = 1$

Proposition 2 explains why the graph branch is trained as a residual correction rather than as another full classifier. It does not by itself make the residual spatial. That role is played by $\mathcal { L } _ { t e x t u r e } \mathrm { : }$ the negative view preserves all patch appearances and labels but corrupts the coordinate-induced graph, so a branch that ignores topology cannot reliably satisfy the margin. The theory therefore supports a specific method design rather than serving as a standalone theoretical contribution.

## 5 Experiments

## 5.1 Experimental Setup

Evaluation covers 9 public WSI benchmarks: BRACS [Brancati et al., 2022], PANDA, TCGA-NSCLC, TCGA-BRCA, and five TCGA survival cohorts (KIRC, KIRP, LUAD, STAD, UCEC). Slides are processed into non-overlapping 256 × 256 patches at 20× and represented with 1024- d UNI features. All baselines use the same features, patient-level stratified splits, preprocessing, optimizer family, and evaluation protocol; no patient appears in more than one of train, validation, and test splits. Unless otherwise noted, results are mean±std over 5 random seeds.

Baselines cover classic MIL (AB-MIL, CLAM-SB, DTFD-MIL), dual-stream and Transformer MIL (DS-MIL, TransMIL), and recent high-capacity methods including ILRA-MIL, MHIM-MIL, DGR-MIL, and 2DMambaMIL. Full setup details are in Appendix C.

The evaluation asks three questions: whether the phenomenon can be isolated under controlled composition/topology, whether the residual design improves real WSI prediction under one feature pipeline, and whether the gains come with stronger spatial behavior rather than only higher capacity.

## 5.2 Controlled Evidence for Spatial Blindness

Spatial-MNIST-Bag is first used to separate composition from topology. Dataset A is purely compositional: coordinates are random and the label depends only on whether digit “9” appears. Dataset B removes this shortcut: every bag contains the same digit multiset, and the label depends only on whether five key digits form a compact spatial motif. Full construction details are in Appendix C.2.

Table 1 gives the main diagnostic result. AB-MIL and TransMIL solve Dataset A, so their failure is not basic MIL capacity. The same models collapse on Dataset B, where composition is identical across classes. Joint training helps but remains far below residual decoupling. In this controlled setting, the issue is not an easier compositional rule; the statistical stream has no real label signal. The difficulty is that weak bag-level supervision must be assigned to a sparse spatial motif, while a jointly trained model can still fit non-informative statistical noise before the graph branch has learned the motif.

Table 1: Composition–Topology Diagnostic Benchmark. Dataset A isolates composition; Dataset B requires topology.
<table><tr><td>Method</td><td>A: Comp. AUC</td><td>B: Topo. AUĆ</td></tr><tr><td>AB-MIL</td><td>0.998</td><td>0.505</td></tr><tr><td>TransMIL</td><td>0.995</td><td>0.532</td></tr><tr><td>ResTopoMIL (Joint)</td><td>0.991</td><td>0.684</td></tr><tr><td>ResTopoMIL</td><td>0.994</td><td>0.987</td></tr></table>

## 5.3 WSI Classification

Table 2 combines the four classification benchmarks and reports Accuracy/AUC; F1 scores, parameter counts, and CTransPath results are in Appendices D and F.

Table 2: Classification Results on Four WSI Benchmarks. Accuracy/AUC are reported as mean ± std. F1 scores and parameter counts are separated into Appendix D.
<table><tr><td rowspan="2">Method</td><td colspan="2">BRACS</td><td colspan="2">PANDA</td><td colspan="2">TCGA-NSCLC</td><td colspan="2">TCGA-BRCA</td></tr><tr><td>Acc</td><td>AUC</td><td>Acc</td><td>AUC</td><td>Acc</td><td>AUC</td><td> $_ \mathrm { A c c }$ </td><td>AUC</td></tr><tr><td>AB-MIL</td><td> $0 . 7 2 7 5 { \scriptstyle \pm . 0 7 5 9 }$ </td><td> $0 . 8 8 0 6 _ { \pm . 0 0 9 1 }$ </td><td> $0 . 7 3 2 2 { \scriptstyle \pm . 0 0 5 9 }$ </td><td> $0 . 9 3 0 6 _ { \pm . 0 0 1 7 }$ </td><td> $0 . 8 9 8 8 { \scriptstyle \pm . 0 0 6 6 }$ </td><td> $0 . 9 5 6 9 { \scriptstyle \pm . 0 0 7 3 }$ </td><td> $0 . 9 4 1 4 _ { \pm . 0 0 6 2 }$ </td><td> $0 . 9 7 2 7 { \scriptstyle \pm . 0 0 1 9 }$ </td></tr><tr><td>CLAM-SB</td><td> $\underline { { 0 . 7 3 7 1 } } \pm . 0 1 8 2$ </td><td> $\underline { { 0 . 8 8 4 0 } } \pm . 0 1 3 1$ </td><td> $0 . 7 3 1 8 { \scriptstyle \pm . 0 0 4 1 }$ </td><td> $0 . 9 2 1 5 { \scriptstyle \pm . 0 0 0 9 }$ </td><td> $0 . 8 8 3 6 { \scriptstyle \pm . 0 0 8 5 }$ </td><td> $0 . 9 6 3 5 { \scriptstyle \pm . 0 0 6 4 }$ </td><td> $0 . 9 3 1 4 _ { \pm . 0 0 6 2 }$ </td><td> $\underline { { 0 . 9 8 1 4 } } \pm . 0 0 2 7$ </td></tr><tr><td>DS-MIL</td><td> $0 . 6 4 6 0 { \scriptstyle \pm . 0 1 8 9 }$ </td><td> $0 . 8 0 5 4 { \scriptstyle \pm . 0 2 2 5 }$ </td><td> $0 . 7 3 9 4 _ { \pm . 0 2 0 7 }$ </td><td>0.9309±.0070</td><td> $0 . 8 8 3 6 { \scriptstyle \pm . 0 0 8 5 }$ </td><td> $0 . 9 5 7 9 _ { \pm . 0 0 8 3 }$ </td><td> $0 . 9 4 0 9 { \scriptstyle \pm . 0 1 2 4 }$ </td><td> $0 . 9 7 7 7 { \scriptstyle \pm . 0 0 3 9 }$ </td></tr><tr><td>TransMIL</td><td> $0 . 6 5 0 6 { \scriptstyle \pm . 0 1 7 4 }$ </td><td> $0 . 8 4 5 0 { \scriptstyle \pm . 0 0 9 8 }$ </td><td> $0 . 7 0 9 0 { \scriptstyle \pm . 0 0 8 8 }$ </td><td> $0 . 9 2 8 8 { \scriptstyle \pm . 0 0 3 8 }$ </td><td> $0 . 8 9 3 3 { \scriptstyle \pm . 0 1 3 2 }$ </td><td> $0 . 9 6 9 2 { \scriptstyle \pm . 0 0 8 4 }$ </td><td> $0 . 9 4 0 9 { \scriptstyle \pm . 0 0 9 5 }$ </td><td> $0 . 9 7 8 7 { \scriptstyle \pm . 0 1 5 4 }$ </td></tr><tr><td>ILRA-MIL</td><td>0.6230±.0286</td><td> $0 . 8 0 1 2 { \scriptstyle \pm . 0 2 1 9 }$ </td><td> $0 . 7 4 0 2 { \scriptstyle \pm . 0 0 9 6 }$ </td><td> $0 . 9 2 6 6 { \scriptstyle \pm . 0 0 3 4 }$ </td><td> $0 . 8 9 1 2 { \scriptstyle \pm . 0 1 5 7 }$ </td><td> $0 . 9 6 2 8 { \scriptstyle \pm . 0 0 9 5 }$ </td><td> $0 . 9 4 5 5 { \scriptstyle \pm . 0 0 5 1 }$ </td><td> $0 . 9 6 2 1 { \scriptstyle \pm . 0 1 4 8 }$ </td></tr><tr><td>MHIM-MIL</td><td> $0 . 6 6 9 0 { \scriptstyle \pm . 0 3 5 8 }$ </td><td> $0 . 8 3 4 0 { \scriptstyle \pm . 0 1 2 8 }$ </td><td> $0 . 6 9 7 0 { \scriptstyle \pm . 0 1 2 3 }$ </td><td> $0 . 9 1 5 5 { \scriptstyle \pm . 0 0 2 0 }$ </td><td> $0 . 8 9 0 8 { \scriptstyle \pm . 0 1 8 3 }$ </td><td> $\mathbf { 0 . 9 7 5 9 } _ { \pm . 0 0 3 0 }$ </td><td> $\underline { { 0 . 9 4 6 5 } } \pm . 0 0 9 5$ </td><td> $0 . 9 7 6 9 { \scriptstyle \pm . 0 1 2 9 }$ </td></tr><tr><td>DGR-MIL</td><td> $0 . 7 1 2 6 { \scriptstyle \pm . 0 3 6 3 }$ </td><td> $0 . 8 2 5 8 { \scriptstyle \pm . 0 3 8 6 }$ </td><td> $0 . 6 9 6 4 { \scriptstyle \pm . 0 1 2 9 }$ </td><td> $0 . 9 0 4 3 _ { \pm . 0 0 8 2 }$ </td><td> $0 . 9 0 3 6 { \scriptstyle \pm . 0 2 2 5 }$ </td><td> $0 . 9 3 9 0 { \scriptstyle \pm . 0 2 2 4 }$ </td><td> $0 . 9 4 5 5 { \scriptstyle \pm . 0 0 5 1 }$ </td><td> $0 . 9 7 2 4 { \scriptstyle \pm . 0 1 9 7 }$ </td></tr><tr><td>2DMambaMIL</td><td> $0 . 7 1 8 5 { \scriptstyle \pm . 0 2 9 0 }$ </td><td> $0 . 8 3 1 5 { \scriptstyle \pm . 0 3 1 0 }$ </td><td> $0 . 7 0 1 2 { \scriptstyle \pm . 0 1 1 5 }$ </td><td> $0 . 9 1 0 5 { \scriptstyle \pm . 0 0 7 5 }$ </td><td> $\underline { { 0 . 9 0 0 8 } } \pm . 0 1 9 0$ </td><td> $0 . 9 4 5 0 { \scriptstyle \pm . 0 1 8 0 }$ </td><td> $0 . 9 4 9 0 { \scriptstyle \pm . 0 0 6 0 }$ </td><td> $0 . 9 7 5 5 _ { \pm . 0 1 8 0 }$ </td></tr><tr><td>ResTopoMIL</td><td> $\mathbf { 0 . 7 4 9 4 } _ { \pm . 0 2 8 6 }$ </td><td> $\mathbf { 0 . 9 0 0 6 } _ { \pm . 0 0 5 5 }$ </td><td> $\mathbf { 0 . 7 5 4 6 _ { \pm . 0 0 9 4 } }$ </td><td> $\mathbf { 0 . 9 4 2 6 _ { \pm . 0 0 1 0 } }$ </td><td> $\mathbf { 0 . 9 1 5 7 { \scriptstyle \pm . 0 0 8 5 } }$ </td><td> $\underline { { 0 . 9 7 5 3 } } \pm . 0 0 2 9$ </td><td> $\mathbf { 0 . 9 5 6 8 _ { \pm . 0 0 9 5 } }$ </td><td> $\mathbf { 0 . 9 8 3 8 _ { \pm . 0 0 4 9 } }$ </td></tr></table>

ResTopoMIL gives the strongest overall classification profile in Table 2: it has the best Accuracy on all four datasets and the best AUC on three of four datasets, with MHIM-MIL slightly higher on TCGA-NSCLC AUC. The differences are most revealing on BRACS and PANDA, where atypia and Gleason grading depend strongly on tissue organization. On TCGA-BRCA the accuracy margins are smaller, but AUC and F1 remain consistent with the ductal-versus-lobular growth-pattern distinction.

## 5.4 WSI Survival Prediction

Survival prediction is a useful stress test because labels are weaker and more heterogeneous than diagnostic labels. Table 3 shows the best C-index on all five cohorts.

Table 3: Survival Prediction Results (C-Index). Concordance Index is reported as mean ± std on 5 TCGA datasets. Parameter counts are listed separately in Appendix D. Higher C-Index is better.
<table><tr><td>Method</td><td>KIRC</td><td>KIRP</td><td>LUAD</td><td>STAD</td><td>UCEC</td></tr><tr><td>AB-MIL</td><td> $0 . 5 6 9 4 { \scriptstyle \pm . 0 2 7 9 }$ </td><td> $0 . 7 0 9 1 { \scriptstyle \pm . 0 5 9 2 }$ </td><td> $0 . 5 9 4 2 { \scriptstyle \pm . 0 1 2 0 }$ </td><td> $0 . 5 8 7 1 { \scriptstyle \pm . 0 0 9 0 }$ </td><td> $0 . 6 2 2 0 { \scriptstyle \pm . 0 2 4 1 }$ </td></tr><tr><td>CLAM-SB</td><td> $0 . 5 7 0 5 { \scriptstyle \pm . 0 1 0 1 }$ </td><td> $0 . 7 0 3 4 { \scriptstyle \pm . 0 5 6 5 }$ </td><td> $0 . 6 1 9 2 { \scriptstyle \pm . 0 1 0 8 }$ </td><td> $0 . 5 8 2 9 _ { \pm . 0 2 4 3 }$ </td><td> $0 . 6 0 3 4 { \scriptstyle \pm . 0 3 9 8 }$ </td></tr><tr><td>DS-MIL</td><td> $0 . 5 8 3 2 { \scriptstyle \pm . 0 2 0 7 }$ </td><td> $0 . 6 8 7 5 { \scriptstyle \pm . 0 4 5 6 }$ </td><td> $0 . 5 6 6 9 { \scriptstyle \pm . 0 1 4 5 }$ </td><td> $0 . 6 0 9 0 { \scriptstyle \pm . 0 1 8 7 }$ </td><td> $0 . 6 3 6 8 { \scriptstyle \pm . 0 4 1 7 }$ </td></tr><tr><td>TransMIL</td><td> $0 . 5 6 7 7 { \scriptstyle \pm . 0 1 6 2 }$ </td><td> $0 . 7 0 0 0 { \scriptstyle \pm . 0 2 3 8 }$ </td><td> $0 . 6 1 4 8 { \scriptstyle \pm . 0 1 2 4 }$ </td><td> $0 . 6 2 0 4 { \scriptstyle \pm . 0 3 5 9 }$ </td><td> $0 . 6 5 9 5 { \scriptstyle \pm . 0 4 8 6 }$ </td></tr><tr><td>ILRA-MIL</td><td> $0 . 7 2 7 6 { \scriptstyle \pm . 0 2 7 1 }$ </td><td> $0 . 7 2 7 2 { \scriptstyle \pm . 1 3 2 2 }$ </td><td> $0 . 6 0 6 7 { \scriptstyle \pm . 0 1 5 6 }$ </td><td> $0 . 6 5 8 9 { \scriptstyle \pm . 0 3 5 4 }$ </td><td> $0 . 6 7 6 7 { \scriptstyle \pm . 0 3 0 8 }$ </td></tr><tr><td>MHIM-MIL</td><td> $0 . 7 1 7 0 { \scriptstyle \pm . 0 1 4 3 }$ </td><td> $\underline { { 0 . 8 0 7 3 } } \pm . 0 5 4 5$ </td><td> $0 . 5 4 9 1 { \scriptstyle \pm . 0 2 0 5 }$ </td><td> $0 . 6 3 5 7 { \scriptstyle \pm . 0 3 2 1 }$ </td><td> $0 . 6 4 7 4 { \scriptstyle \pm . 0 7 0 4 }$ </td></tr><tr><td>DGR-MIL</td><td> $0 . 7 2 9 8 { \scriptstyle \pm . 0 0 6 8 }$ </td><td> $0 . 8 0 7 1 { \scriptstyle \pm . 0 2 6 5 }$ </td><td> $0 . 6 2 4 5 { \scriptstyle \pm . 0 1 1 7 }$ </td><td> $0 . 6 6 2 5 { \scriptstyle \pm . 0 1 2 2 }$ </td><td> $0 . 6 9 7 6 { \scriptstyle \pm . 0 1 0 4 }$ </td></tr><tr><td>2DMambaMIL</td><td> $\underline { { 0 . 7 3 1 1 } } \underline { { \pm } } . 0 1 1 0$ </td><td> $0 . 8 0 2 7 { \scriptstyle \pm . 0 2 5 0 }$ </td><td> $\underline { { 0 . 6 2 9 0 } } \pm . 0 0 9 5$ </td><td> $0 . 6 5 1 5 { \scriptstyle \pm . 0 1 5 0 }$ </td><td> $\underline { { 0 . 7 0 2 0 } } { \scriptstyle \pm . 0 1 4 0 }$ </td></tr><tr><td>ResTopoMIL</td><td> $\mathbf { 0 . 7 3 1 3 _ { \pm . 0 1 0 4 } }$ </td><td> $\mathbf { 0 . 8 1 8 2 } _ { \pm . 0 2 2 7 }$ </td><td> $\mathbf { 0 . 6 4 5 7 { \scriptstyle \pm . 0 0 7 0 } }$ </td><td> $\mathbf { 0 . 6 8 0 7 { \scriptstyle \pm . 0 0 8 6 } }$ </td><td> $\mathbf { 0 . 7 0 5 8 _ { \pm . 0 1 2 8 } }$ </td></tr></table>

The advantage is clearest on LUAD and STAD, where growth patterns and local tissue organization are prognostic factors. The consistent C-index gains suggest that the residual branch captures more than tissue proportions.

## 5.5 Mechanistic and Ablation Evidence

The mechanistic question is whether the gain comes from residual decoupling rather than a larger backbone, a training trick, or a favorable seed. Figure 4 and Tables 4–5 give the main evidence; Appendix E gives full metrics.

Figure 4 separates endpoint performance from training dynamics. In the joint run, slide-level AUC improves early and then saturates while the graph branch receives little gradient. In the stepwise run, unfreezing the graph branch produces a second rise, consistent with residual errors becoming available to topology after the statistical stream is fixed. The gradient trace is the sharper evidence: the topological gradient fades under joint optimization, rebounds after freezing, and collapses again when $\mathcal { L } _ { t e x t u r e }$ is removed. Thus the same GCN needs both a fixed residual target and a coordinatespecific constraint.

(d) PANDA: Impact of Texture Loss Removal  
![](images/15e6f5ee34acc609670f7551bcd230676096eeabefc98fd76115796ebd660723.jpg)

![](images/7d84ae2679742afff2044cc3af9491bc36d24abfb8d68fc2a69537a7a78d156f.jpg)

![](images/1339a9d2107d26657a5f462be05a5d5dcd3066d5dbed69de28bb7088caa1a4f3.jpg)

![](images/430d38dd19a988f541bc0fdf6f04829001f1d594dc5d22a8f508d2cbd63db224.jpg)  
Figure 4: Gradient dynamics. Stepwise training revives the topological gradient after freezing the statistical stream; joint optimization and the variant without $\mathcal { L } _ { t e x t u r e }$ both let it fade.

Table 4: Core Strategy. AUC mean±std.  
Table 5: Architecture & Validity. AUC mean±std.
<table><tr><td>Group</td><td>Variant</td><td>PANDA AUC</td><td>BRCA AUC</td></tr><tr><td rowspan="5">Opt.</td><td>ResTopoMIL</td><td> $\mathbf { 0 . 9 4 2 6 { \scriptstyle \pm . 0 0 1 0 } }$ </td><td> $0 . 9 8 3 8 { \scriptstyle \pm . 0 0 4 9 }$ </td></tr><tr><td>Stat. Only</td><td> $0 . 9 0 2 7 { \scriptstyle \pm . 0 0 5 1 }$ </td><td> $0 . 9 4 8 6 { \scriptstyle \pm . 0 0 3 5 }$ </td></tr><tr><td>Topo. Only</td><td> $0 . 9 2 1 5 { \scriptstyle \pm . 0 0 5 5 }$ </td><td> $0 . 9 6 0 8 { \scriptstyle \pm . 0 0 4 8 }$ </td></tr><tr><td>Joint Opt.</td><td> $0 . 9 2 9 9 { \scriptstyle \pm . 0 0 5 2 }$ </td><td> $0 . 9 7 7 3 { \scriptstyle \pm . 0 0 2 2 }$ </td></tr><tr><td>Multi-LR</td><td> $0 . 9 3 5 2 _ { \pm . 0 0 4 8 }$ </td><td> $0 . 9 7 8 6 _ { \pm . 0 0 1 8 }$ </td></tr><tr><td rowspan="3">Fusion</td><td>Gated Fusion</td><td> $0 . 9 2 2 5 { \scriptstyle \pm . 0 0 6 2 }$ </td><td> $0 . 9 8 0 2 { \scriptstyle \pm . 0 0 4 2 }$ </td></tr><tr><td>MoE</td><td> $0 . 9 3 0 5 { \scriptstyle \pm . 0 0 3 5 }$ </td><td> $\mathbf { 0 . 9 8 9 7 { \scriptstyle \pm . 0 0 2 8 } }$ </td></tr><tr><td>Indep. Clf</td><td> $0 . 9 2 5 6 _ { \pm . 0 0 6 2 }$ </td><td> $0 . 9 8 4 5 _ { \pm . 0 0 2 5 }$ </td></tr><tr><td>Constraint w/o</td><td> $\mathcal { L } _ { t e x }$ </td><td> $0 . 9 1 4 7 { \scriptstyle \pm . 0 0 6 5 }$ </td><td> $0 . 9 7 6 2 { \scriptstyle \pm . 0 0 3 8 }$ </td></tr></table>

<table><tr><td>Group</td><td>Variant</td><td>PANDA AUC</td><td>BRCA AUC</td></tr><tr><td>Backbone</td><td>ResTopoMIL (GCN) GAT</td><td> $\begin{array} { c } { { \mathbf { 0 . 9 4 2 6 { \scriptstyle \pm . 0 0 1 0 } } } } \\ { { 0 . 9 4 1 9 _ { \pm . 0 0 4 1 } } } \end{array}$ </td><td>0.9838±.0049  $\mathbf { 0 . 9 8 8 9 _ { \pm . 0 0 2 5 } }$ </td></tr><tr><td>Sanity</td><td>Random Graph Fixed Proto.</td><td> $0 . 8 2 8 6 _ { \pm . 0 1 5 0 }$   $0 . 9 3 4 9 _ { \pm . 0 0 5 5 } ^ { - }$ </td><td>0.8563±.0110 0.9746±.0035</td></tr></table>

The compact tables are best read together rather than as a single leaderboard. Stat.-Only and Topo.- Only show that neither branch alone explains the full behavior: composition is a strong baseline, but topology has independent signal. Joint Opt. and Multi-LR are the closest alternatives to our schedule; the latter helps on PANDA, but still does not match residual decoupling. MoE and GAT reach higher TCGA-BRCA AUC than the default GCN setting, yet they are weaker on PANDA and do not give the same controlled spatial behavior seen in the coordinate-shuffling diagnostic. The sanity controls are more decisive: a random graph breaks performance, fixed prototypes weaken the anchor, and removing $\mathcal { L } _ { t e x }$ is the strongest non-random negative control on PANDA AUC.

These results are not meant to argue that a two-layer GCN is intrinsically better than attention or state-space context modules. The point is narrower: once a strong compositional shortcut is available, a spatial module can be present but weakly used. Residual decoupling changes the assignment of error. It lets the statistical stream explain the easy part, then gives the remaining supervision to a branch explicitly tested against coordinate corruption. This is also why high slide-level AUC should not be taken as proof that a model has used tissue architecture; the model should predict well, but lose the right evidence when spatial structure is deliberately removed.

## 6 Conclusion

This paper identifies spatial blindness: context-aware MIL can perform well at slide level while using little tissue topology. ResTopoMIL addresses this failure mode by fitting composition first and then training topology as a residual correction under a shuffle-based spatial constraint. Across 9 public benchmarks it improves prediction, restores spatial sensitivity, and gives stronger localization evidence with only 1.15M parameters. The broader lesson is that adding context is not the same as learning from context; optimization can decide which signal is actually used. Future work should test this diagnosis with trainable pathology foundation encoders, prospective cohorts, and clinically realistic coordinate perturbations such as registration noise or rigid transformations.

## References

Mohammed Adnan, Shivam Kalra, and Hamid R Tizhoosh. Representation learning of histopathology images using graph neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 988–989, 2020.

Nadia Brancati, Anna Maria Anniciello, Pushpak Pati, Daniel Riccio, Giosuè Scognamiglio, Guillaume Jaume, Giuseppe De Pietro, Maurizio Di Bonito, Antonio Foncubierta, Gerardo Botti, et al. Bracs: A dataset for breast carcinoma subtyping in h&e histology images. Database, 2022: baac093, 2022.

Zak Buzzard, Konstantin Hemker, Nikola Simidjievski, and Mateja Jamnik. Paths: A hierarchical transformer for efficient whole slide image analysis. arXiv preprint arXiv:2411.18225, 2024.

Gabriele Campanella, Matthew G Hanna, Luke Geneslaw, Allen Miraflor, Vitor Werneck Krauss Silva, Klaus J Busam, Edi Brogi, Victor E Reuter, David S Klimstra, and Thomas J Fuchs. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nature medicine, 25(8):1301–1309, 2019.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

Chi-Long Chen, Chi-Chung Chen, Wei-Hsiang Yu, Szu-Hua Chen, Yu-Chan Chang, Tai-I Hsu, Michael Hsiao, Chao-Yuan Yeh, and Cheng-Yu Chen. An annotation-free whole-slide training approach to pathological classification of lung cancer types using deep learning. Nature communications, 12(1):1193, 2021a.

Kaitao Chen, Shiliang Sun, and Jing Zhao. Camil: Causal multiple instance learning for whole slide image classification. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 1120–1128, 2024.

Richard J Chen, Ming Y Lu, Muhammad Shaban, Chengkuan Chen, Tiffany Y Chen, Drew FK Williamson, and Faisal Mahmood. Whole slide images are 2d point clouds: Context-aware survival prediction using patch-based graph convolutional networks. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 339–349. Springer, 2021b.

Richard J Chen, Chengkuan Chen, Yicong Li, Tiffany Y Chen, Andrew D Trister, Rahul G Krishnan, and Faisal Mahmood. Scaling vision transformers to gigapixel images via hierarchical self-supervised learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16144–16155, 2022.

Richard J Chen, Tong Ding, Ming Y Lu, Drew FK Williamson, Guillaume Jaume, Bowen Chen, Andrew Zhang, Daniel Shao, Andrew H Song, Muhammad Shaban, et al. A general-purpose self-supervised model for computational pathology. arXiv preprint arXiv:2308.15474, 2023.

Jun Cheng, Yuting Liu, Wei Huang, Wenhui Hong, Lingling Wang, Xiaohui Zhan, Zhi Han, Dong Ni, Kun Huang, and Jie Zhang. Computational image analysis identifies histopathological image features associated with somatic mutations and patient survival in gastric adenocarcinoma. Frontiers in Oncology, 11:623382, 2021.

Thomas M Cover. Elements ofinformation theory. John Wiley & Sons, 1999.

Kausik Das, Sri Phani Krishna Karri, Abhijit Guha Roy, Jyotirmoy Chatterjee, and Debdoot Sheet. Classifying histopathology whole-slides using fusion of decisions from deep convolutional network on a collection of random multi-views at multi-magnification. In 2017 IEEE 14th Interna tional Symposium on Biomedical Imaging (ISBI 2017), pages 1024–1027. IEEE, 2017.

Kausik Das, Sailesh Conjeti, Abhijit Guha Roy, Jyotirmoy Chatterjee, and Debdoot Sheet. Multiple instance learning of deep convolutional neural networks for breast histopathology whole slide classification. In 2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018), pages 578–581. IEEE, 2018.

Thomas G Dietterich, Richard H Lathrop, and Tomás Lozano-Pérez. Solving the multiple instance problem with axis-parallel rectangles. Artificial intelligence, 89(1-2):31–71, 1997.

Babak Ehteshami Bejnordi, Mitko Veta, Paul Johannes van Diest, Bram Van Ginneken, Nico Karssemeijer, Geert Litjens, Jeroen AWM Van Der Laak, CAMELYON16 consortium, Meyke Hermsen, Quirine F Manson, et al. Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. Jama, 318(22):2199–2210, 2017.

Olga Fourkioti, Matt De Vries, and Chris Bakal. CAMIL: Context-aware multiple instance learning for cancer detection and subtyping in whole slide images. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=rzBskAEmoc.

Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A Wichmann, and Wieland Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. In International conference on learning representations, 2018.

Jiaxiang Gou, Luping Ji, Pei Liu, and Mao Ye. Queryable prototype multiple instance learning with vision-language models for incremental whole slide image classification. In Proceedings of the AAAI conference on artificial intelligence, volume 39, pages 3158–3166, 2025.

Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured state spaces. In International Conference on Learning Representations, 2021.

Ziyu Guo, Weiqin Zhao, Shujun Wang, and Lequan Yu. Higt: Hierarchical interaction graphtransformer for whole slide image analysis. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 755–764. Springer, 2023.

Minghao Han, Linhao Qu, Dingkang Yang, Xukun Zhang, Xiaoying Wang, and Lihua Zhang. Mscpt: Few-shot whole slide image classification with multi-scale and context-focused prompt tuning. IEEE Transactions on Medical Imaging, 2025.

Maximilian Ilse, Jakub Tomczak, and Max Welling. Attention-based deep multiple instance learning. In International conference on machine learning, pages 2127–2136. PMLR, 2018.

Mingu Kang, Heon Song, Seonwook Park, Donggeun Yoo, and Sérgio Pereira. Benchmarking selfsupervised learning on diverse pathology datasets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3344–3354, 2023.

Saarthak Kapse, Pushpak Pati, Srikar Yellapragada, Srijan Das, Rajarsi R Gupta, Joel Saltz, Dimitris Samaras, and Prateek Prasanna. Gecko: Gigapixel vision-concept contrastive pretraining in histopathology. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20020–20030, 2025.

Oren Z Kraus, Jimmy Lei Ba, and Brendan J Frey. Classifying and segmenting microscopy images with deep multiple instance learning. Bioinformatics, 32(12):i52–i59, 2016.

Bin Li, Yin Li, and Kevin W Eliceiri. Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14318–14328, 2021.

Tiancheng Lin, Zhimiao Yu, Hongyu Hu, Yi Xu, and Chang-Wen Chen. Interventional bag multiinstance learning on whole-slide pathological images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19830–19839, 2023.

Ming Y Lu, Drew FK Williamson, Tiffany Y Chen, Richard J Chen, Matteo Barbieri, and Faisal Mahmood. Data-efficient and weakly supervised computational pathology on whole-slide images. Nature biomedical engineering, 5(6):555–570, 2021.

Ming Y Lu, Bowen Chen, Andrew Zhang, Drew FK Williamson, Richard J Chen, Tong Ding, Long Phi Le, Yung-Sung Chuang, and Faisal Mahmood. Visual language pretrained multiple instance zero-shot transfer for histopathology images. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 19764–19775, 2023.

Oded Maron and Tomás Lozano-Pérez. A framework for multiple-instance learning. Advances in neural information processing systems, 10, 1997.

Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In European conference on computer vision, pages 69–84. Springer, 2016.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Soumyasundar Pal, Antonios Valkanas, Florence Regol, and Mark Coates. Bag graph: Multiple instance learning using bayesian graph neural networks. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pages 7922–7930, 2022.

Liron Pantanowitz, Paul N Valenstein, Andrew J Evans, Keith J Kaplan, John D Pfeifer, David C Wilbur, Laura C Collins, and Terence J Colgan. Review of the current state of whole slide imaging in pathology. Journal ofpathology informatics, 2(1):36, 2011.

Pushpak Pati, Guillaume Jaume, Antonio Foncubierta-Rodriguez, Florinda Feroce, Anna Maria Anniciello, Giosue Scognamiglio, Nadia Brancati, Maryse Fiche, Estelle Dubruc, Daniel Riccio, et al. Hierarchical graph representations in digital pathology. Medical image analysis, 75:102264, 2022.

Joshua Peeples, Weihuang Xu, and Alina Zare. Histogram layers for texture analysis. IEEE Transactions on Artificial Intelligence, 3(4):541–552, 2021.

Mohammad Pezeshki, Oumar Kaba, Yoshua Bengio, Aaron C Courville, Doina Precup, and Guillaume Lajoie. Gradient starvation: A learning proclivity in neural networks. Advances in Neural Information Processing Systems, 34:1256–1272, 2021.

Linhao Qu, Yingfan Ma, Xiaoyuan Luo, Qinhao Guo, Manning Wang, and Zhijian Song. Rethinking multiple instance learning for whole slide image classification: A good instance classifier is all you need. IEEE Transactions on Circuits and Systems for Video Technology, 34(10):9732–9744, 2024.

Daniela F Quail and Johanna A Joyce. Microenvironmental regulation of tumor progression and metastasis. Nature medicine, 19(11):1423–1437, 2013.

Harshay Shah, Kaustav Tamuly, Aditi Raghunathan, Prateek Jain, and Praneeth Netrapalli. The pitfalls of simplicity bias in neural networks. Advances in Neural Information Processing Systems, 33:9573–9585, 2020.

Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):379–423, 1948.

Daniel Shao, Richard J Chen, Andrew H Song, Joel Runevic, Ming Y Lu, Tong Ding, and Faisal Mahmood. Do multiple instance learning models transfer? Proceedings of Machine Learning Research, 267:54219–54238, 2025.

Zhuchen Shao, Hao Bian, Yang Chen, Yifeng Wang, Jian Zhang, Xiangyang Ji, et al. Transmil: Transformer based correlated multiple instance learning for whole slide image classification. Advances in neural information processing systems, 34:2136–2147, 2021.

Jiangbo Shi, Chen Li, Tieliang Gong, Yefeng Zheng, and Huazhu Fu. Vila-mil: Dual-scale visionlanguage multiple instance learning for whole slide image classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11248–11258, 2024.

Andrew H Song, Guillaume Jaume, Drew FK Williamson, Ming Y Lu, Anurag Vaidya, Tiffany R Miller, and Faisal Mahmood. Artificial intelligence for digital and computational pathology. Nature Reviews Bioengineering, 1(12):930–949, 2023.

Andrew H Song, Richard J Chen, Tong Ding, Drew FK Williamson, Guillaume Jaume, and Faisal Mahmood. Morphological prototyping for unsupervised slide representation learning in computational pathology. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11566–11578, 2024.

Wenhao Tang, Sheng Huang, Xiaoxian Zhang, Fengtao Zhou, Yi Zhang, and Bo Liu. Multiple instance learning framework with masked hard instance mining for whole slide image classification. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 4078–4087, 2023.

Devavrat Tomar, Guillaume Vray, Dwarikanath Mahapatra, Sudipta Roy, Jean-Philippe Thiran, and Behzad Bozorgtabar. Slide-level prompt learning with vision language models for few-shot multiple instance learning in histopathology. In 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI), pages 1–5. IEEE, 2025.

Manuel Tran, Sophia Wagner, Wilko Weichert, Christian Matek, Melanie Boxberg, and Tingying Peng. Navigating through whole slide images with hierarchy, multi-object, and multi-scale data. IEEE transactions on medical imaging, 44(5):2002–2015, 2025.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Gregory Verghese, Jochen K Lennerz, Danny Ruta, Wen Ng, Selvam Thavaraj, Kalliopi P Siziopikou, Threnesan Naidoo, Swapnil Rane, Roberto Salgado, Sarah E Pinder, et al. Computational pathology in cancer diagnosis, prognosis, and prediction–present day and prospects. The Journal ofpathology, 260(5):551–563, 2023.

Xi Wang, Hao Chen, Caixia Gan, Huangjing Lin, Qi Dou, Efstratios Tsougenis, Qitao Huang, Muyan Cai, and Pheng-Ann Heng. Weakly supervised deep learning for whole slide lung cancer image analysis. IEEE transactions on cybernetics, 50(9):3950–3962, 2019.

Xiaodong Wang, Ying Chen, Yunshu Gao, Huiqing Zhang, Zehui Guan, Zhou Dong, Yuxuan Zheng, Jiarui Jiang, Haoqing Yang, Liming Wang, et al. Predicting gastric cancer outcome from resected lymph node histopathology images using deep learning. Nature communications, 12(1):1637, 2021.

Xiu-Shen Wei, Jianxin Wu, and Zhi-Hua Zhou. Scalable algorithms for multi-instance learning. IEEE transactions on neural networks and learning systems, 28(4):975–987, 2016.

Bryan Wong, Jongwoo Kim, Huazhu Fu, and Mun Yong Yi. Few-shot learning from gigapixel images via hierarchical vision-language alignment and modeling. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Jinxi Xiang and Jun Zhang. Exploring low-rank property in multiple instance learning for whole slide image classification. In The Eleventh International Conference on Learning Representations, 2023.

Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, and Vikas Singh. Nyströmformer: A nyström-based algorithm for approximating self-attention. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 14138–14148, 2021.

Yan Xu, Jun-Yan Zhu, I Eric, Chao Chang, Maode Lai, and Zhuowen Tu. Weakly supervised histopathology cancer image segmentation and classification. Medical image analysis, 18(3): 591–604, 2014.

Shu Yang, Yihui Wang, and Hao Chen. Mambamil: Enhancing long sequence modeling with sequence reordering in computational pathology. In International conference on medical image computing and computer-assisted intervention, pages 296–306. Springer, 2024.

Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov, and Alexander J Smola. Deep sets. Advances in neural information processing systems, 30, 2017.

Dan Zhang, Yan Liu, Luo Si, Jian Zhang, and Richard Lawrence. Multiple instance learning on structured data. Advances in Neural Information Processing Systems, 24, 2011.

Hongrun Zhang, Yanda Meng, Yitian Zhao, Yihong Qiao, Xiaoyun Yang, Sarah E Coupland, and Yalin Zheng. Dtfd-mil: Double-tier feature distillation multiple instance learning for histopathology whole slide image classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18802–18812, 2022.

Jingwei Zhang, Ke Ma, John Van Arnam, Rajarsi Gupta, Joel Saltz, Maria Vakalopoulou, and Dimitris Samaras. A joint spatial and magnification based attention framework for large scale histopathology classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3776–3784, 2021.

Jingwei Zhang, Anh Tien Nguyen, Xi Han, Vincent Quoc-Huy Trinh, Hong Qin, Dimitris Samaras, and Mahdi S Hosseini. 2dmamba: Efficient state space model for image representation with applications on giga-pixel whole slide image classification. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3583–3592, 2025a.

Yunlong Zhang, Honglin Li, Yunxuan Sun, Sunyi Zheng, Chenglu Zhu, and Lin Yang. Attentionchallenging multiple instance learning for whole slide image classification. In European conference on computer vision, pages 125–143. Springer, 2024.

Yunlong Zhang, Honglin Li, Yuxuan Sun, Zhongyi Shui, Jingxiong Li, Chenglu Zhu, and Lin Yang. Aem: attention entropy maximization for multiple instance learning based whole slide image classification. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 45–55. Springer, 2025b.

Zhi-Hua Zhou, Yu-Yin Sun, and Yu-Feng Li. Multi-instance learning by treating instances as non-iid samples. In Proceedings of the 26th annual international conference on machine learning, pages 1249–1256, 2009.

Wenhui Zhu, Peijie Qiu, Xiwen Chen, Oana M Dumitrascu, and Yalin Wang. Pdl: Regularizing multiple instance learning with progressive dropout layers. arXiv preprint arXiv:2308.10112, 2023.

Wenhui Zhu, Peijie Qiu, Xiwen Chen, Zhangsihao Yang, Aristeidis Sotiras, Abolfazl Razi, and Yalin Wang. How effective can dropout be in multiple instance learning? In International Conference on Machine Learning, pages 80090–80106. PMLR, 2025.

## A Overview

This supplement provides the full technical and experimental material behind the main paper. Appendix B presents the design analysis of residual decoupling, including the residual-error view of graph gradients and the conditional-information interpretation of the topological branch. Appendix C gives the experimental configuration, covering the WSI datasets, survival cohorts, Spatial-MNIST-Bag construction, patch processing, implementation details, and training protocol. Appendix D reports additional classification metrics and model sizes. Appendix E provides the extended ablation analysis for optimization strategy, fusion design, graph validity, and hyperparameter sensitivity. Appendix F evaluates cross-backbone generalization with CTransPath features. Appendix G reports additional shuffle sensitivity on WSI benchmarks, and Appendix H gives the progressive coordinate shuffling analysis. Appendix I details the CAMELYON-16 localization protocol and quantitative localization results. Appendix J visualizes the feature spaces learned by the statistical and topological streams. Appendix K provides qualitative heatmap analysis, and Appendix L discusses additional limitations and negative-result scope.

## B Design Analysis of Residual Decoupling

This appendix derives the two propositions used in Section 4.4. The scope is deliberately narrow. We do not claim that every MIL model must suffer from gradient starvation, or that conditional mutual information is new. The aim is to spell out what the stop-gradient residual objective does in ResTopoMIL: it separates a compositional anchor from a spatial correction.

## B.1 Residual-Error Gating of the Graph Gradient

For binary classification, ResTopoMIL uses an additive logit

$$
f ( X ) = f _ { s t a t } ( X ; \theta _ { s } ) + f _ { t o p o } ( X ; \theta _ { t } ) , \qquad \hat { p } _ { \theta } ( X ) = P _ { \theta } ( Y = 1 \mid X ) = \sigma ( f ( X ) ) .\tag{12}
$$

With cross-entropy loss

$$
\mathcal { L } = \mathbb { E } \big [ - Y \log \hat { p } _ { \theta } ( X ) - ( 1 - Y ) \log ( 1 - \hat { p } _ { \theta } ( X ) ) \big ] ,\tag{13}
$$

the gradient received by the topological branch is

$$
\nabla _ { \theta _ { t } } \mathcal { L } = \mathbb { E } \bigl [ ( \hat { p } _ { \theta } ( X ) - Y ) \nabla _ { \theta _ { t } } f _ { t o p o } ( X ; \theta _ { t } ) \bigr ] .\tag{14}
$$

Let $\boldsymbol { r } _ { \theta } ( X , Y ) = \boldsymbol { \hat { p } _ { \theta } } ( X ) - Y$ denote the remaining prediction error. Applying Cauchy–Schwarz to Eq. (14) gives

$$
\begin{array} { r } { \| \nabla _ { \theta _ { t } } \mathcal { L } \| \leq \left( \mathbb { E } r _ { \theta } ( X , Y ) ^ { 2 } \right) ^ { 1 / 2 } \left( \mathbb { E } \| \nabla _ { \theta _ { t } } f _ { t o p o } ( X ; \theta _ { t } ) \| _ { F } ^ { 2 } \right) ^ { 1 / 2 } . } \end{array}\tag{15}
$$

The bound gives the method-specific point: the graph update is gated by the residual error of the current additive predictor. If the statistical stream rapidly explains much of the slide label, the magnitude of $r _ { \theta } ( X , Y )$ becomes small before the graph branch has learned a useful spatial rule. The graph module may be present in the architecture but weakly trained in practice.

Why the stop-gradient matters. Stage 2 replaces the moving joint target with

$$
f ( X ) = \operatorname { s g } [ f _ { s t a t } ( X ) ] + f _ { t o p o } ( X ; \theta _ { t } ) .\tag{16}
$$

The residual error is now computed against a fixed compositional anchor. The statistical stream can no longer reduce this residual during Stage 2, so the remaining errors are assigned to the topological branch. This is the specific optimization change in ResTopoMIL. Figure 4 is the empirical counterpart: the graph gradient rebounds after unfreezing in the stepwise run, while it decays under joint training. The ablations give additional checks: GAT does not rescue joint training, Multi-LR improves only modestly, and removing $\mathcal { L } _ { t e x }$ again weakens the graph branch.

## B.2 What Information the Residual Branch Is Asked to Learn

Let $Z _ { s t a t } = g _ { s } ( X )$ and $Z _ { t o p o } = g _ { t } ( X )$ denote the statistical and topological representations. The information that the two streams jointly provide about the label decomposes by the chain rule:

$$
\begin{array} { r } { I ( Z _ { s t a t } , Z _ { t o p o } ; Y ) = I ( Z _ { s t a t } ; Y ) + I ( Z _ { t o p o } ; Y \mid Z _ { s t a t } ) . } \end{array}\tag{17}
$$

The first term corresponds to label information already captured by the statistical anchor. The second term is the extra information that remains after conditioning on that anchor. ResTopoMIL is designed around this second term, but with an important practical detail: the topology representation is trained through logits added to a frozen statistical predictor.

In Stage 1, the statistical branch is optimized to obtain $Z _ { s t a t }$ and a statistical logit $f _ { s t a t }$ . In Stage $2 , \ Z _ { s t a t }$ and $f _ { s t a t }$ are frozen, and the topological branch parameterizes a variational conditional posterior

$$
q _ { \phi } ( Y = y \mid Z _ { s t a t } , Z _ { t o p o } ) = \left[ \mathrm { S o f t m a x } ( f _ { s t a t } ( Z _ { s t a t } ) + f _ { t o p o } ( Z _ { t o p o } ; \phi ) ) \right] _ { y } .\tag{18}
$$

For binary classification, this reduces to the sigmoid probability of the positive class, $q _ { \phi } ( Y = 1$ | $Z _ { s t a t } , Z _ { t o p o } ) = \sigma ( f _ { s t a t } + f _ { t o p o } )$ . The conditional mutual information can be written as

$$
I ( Z _ { t o p o } ; Y \mid Z _ { s t a t } ) = H ( Y \mid Z _ { s t a t } ) - H ( Y \mid Z _ { s t a t } , Z _ { t o p o } ) .\tag{19}
$$

Since $Z _ { s t a t }$ is fixed in Stage 2, $, H ( Y \mid Z _ { s t a t } )$ is constant with respect to $\phi .$ It remains to minimize the second term. For any variational decoder $q _ { \phi }$

$$
H ( Y \mid Z _ { s t a t } , Z _ { t o p o } ) = \operatorname { \mathbb { E } } { \big [ } - \log p ( Y \mid Z _ { s t a t } , Z _ { t o p o } ) { \big ] }\tag{20}
$$

$$
\leq \mathbb { E } \big [ - \log q _ { \phi } ( Y \mid Z _ { s t a t } , Z _ { t o p o } ) \big ] ,\tag{21}
$$

where the inequality follows from the non-negativity of

$$
\mathbb { E } _ { Z _ { s t a t } , Z _ { t o p o } } \mathrm { K L } \big ( p ( Y \mid Z _ { s t a t } , Z _ { t o p o } ) \| q _ { \phi } ( Y \mid Z _ { s t a t } , Z _ { t o p o } ) \big ) .\tag{22}
$$

Therefore,

$$
I ( Z _ { t o p o } ; Y \mid Z _ { s t a t } ) \ge H ( Y \mid Z _ { s t a t } ) + \mathbb { E } \big [ \log q _ { \phi } ( Y \mid Z _ { s t a t } , Z _ { t o p o } ) \big ] .\tag{23}
$$

Maximizing the right-hand side is equivalent, up to the fixed constant $H ( Y \mid Z _ { s t a t } )$ , to minimizing the Stage-2 cross-entropy loss

$$
\mathcal { L } _ { r e s } = \mathbb { E } \left[ - \log q _ { \phi } ( Y \mid Z _ { s t a t } , Z _ { t o p o } ) \right] .\tag{24}
$$

Thus the residual objective maximizes a variational lower bound on $I ( Z _ { t o p o } ; Y \mid Z _ { s t a t } )$ with the statistical stream held fixed. This does not claim that the learned representation is automatically spatial. It only states what label information the residual decoder is optimized to extract once the anchor is frozen.

Why conditional information is not enough. The chain rule alone would still allow $Z _ { t o p o }$ to encode another compositional statistic not captured by $Z _ { s t a t }$ . ResTopoMIL therefore adds a coordinate-specific constraint. In the shuffled view, the patch multiset and slide label are unchanged, but the KNN graph induced by physical coordinates is corrupted. A residual branch that ignores coordinates will map the clean and shuffled views to similar representations and will be penalized by $\mathcal { L } _ { t e x t u r e }$ . This is the part of the method that turns a generic residual objective into a topology-seeking residual objective.

## B.3 Why Common Optimization Heuristics Are Not Equivalent

The residual view also clarifies why simple training tricks help only partially. Let

$$
r _ { \theta _ { s } , \theta _ { t } } ( X , Y ) = \hat { p } _ { \theta _ { s } , \theta _ { t } } ( X ) - Y\tag{25}
$$

be the error term multiplying the graph gradient in Eq. (14). In joint optimization, both streams reduce the same residual:

$$
\mathbb { E } [ \Delta \theta _ { t } ] = - \eta _ { t } \mathbb { E } [ r _ { \theta _ { s } , \theta _ { t } } ( X , Y ) \nabla _ { \theta _ { t } } f _ { t o p o } ( X ) ] .\tag{26}
$$

If the statistical stream quickly reduces the magnitude of $r _ { \theta _ { s } , \theta _ { t } }$ , the update in Eq. (26) becomes small even when the graph branch has not learned a meaningful spatial rule. This is the optimization form of spatial blindness.

A larger graph learning rate changes Eq. (26) only by a scalar. With a multiplier $\alpha > 1$

$$
\mathbb { E } [ \Delta \theta _ { t } ] = - \alpha \eta _ { t } \mathbb { E } [ r _ { \theta _ { s } , \theta _ { t } } ( X , Y ) \nabla _ { \theta _ { t } } f _ { t o p o } ( X ) ] .\tag{27}
$$

The graph branch moves faster while the residual is nonzero, but the update is still gated by the same shrinking residual. This explains why Multi-LR improves over vanilla joint training in Table 9, but remains below the frozen-anchor schedule.

Statistical dropout attacks the same problem from the opposite side. By corrupting the statistical stream, it can increase the residual visible to the graph branch:

$$
r _ { \mathrm { d r o p } } ( X , Y ) = \hat { p } ( Y = 1 \mid D ( f _ { s t a t } ) , f _ { t o p o } ) - Y ,\tag{28}
$$

where $D ( \cdot )$ denotes dropout. This may expose topology, but it also injects noise into useful compositional evidence. The higher variance of the Stat-Dropout row is consistent with this trade-off. Curriculum scheduling is a softer decoupling: it changes the relative speed of the two streams, but unless the statistical branch is frozen, the residual target remains moving. Hard instance mining reweights samples,

$$
\mathbb { E } [ \Delta \theta _ { t } ] = - \eta _ { t } \mathbb { E } [ w ( X ) r _ { \theta _ { s } , \theta _ { t } } ( X , Y ) \nabla _ { \theta _ { t } } f _ { t o p o } ( X ) ] ,\tag{29}
$$

and can help by emphasizing difficult slides; it still does not prevent the easy stream from absorbing part of the residual.

ResTopoMIL makes a stronger intervention:

$$
r _ { \mathrm { r e s } } ( X , Y ) = \hat { p } ( Y = 1 \mid \mathrm { s g } [ f _ { s t a t } ( X ) ] , f _ { t o p o } ( X ) ) - Y ,\tag{30}
$$

so the residual is measured against a fixed compositional anchor. The graph branch no longer competes with a moving statistical predictor for the same error term. The shuffle loss then restricts the residual representation to information that changes when the coordinate-induced graph is damaged. This is why the method should be read as an optimization design rather than as a more complicated graph architecture.

## C Experimental Configuration

## C.1 Datasets

Evaluation was conducted on nine datasets spanning three tasks. TCGA-NSCLC contains 1,053 WSIs (LUAD: 541, LUSC: 512) for non-small cell lung cancer subtyping. TCGA-BRCA includes 1,021 WSIs distinguishing Invasive Ductal Carcinoma (IDC) from Invasive Lobular Carcinoma (ILC), with structural differences (ductal vs. single-file strands) serving as key discriminators. BRACS comprises 525 WSIs covering 7 fine-grained categories, where differentiation of atypical classes (ADH, FEA) relies on subtle microenvironmental changes. Finally, PANDA contains 10,616 WSIs for Gleason grading, a task highly dependent on structure $( \mathrm { e . g . }$ , gland formation vs. fused glands) and serving as a primary topological benchmark. All WSI splits are stratified at the patient level whenever patient identifiers are available; slides from the same patient are never shared across train, validation, and test partitions.

## C.1.1 Survival Prediction

Prognosis performance was evaluated on five TCGA cohorts selected for their structural prognostic factors. TCGA-LUAD (516 WSIs) prognosis correlates with histological growth patterns (lepidic, acinar, papillary, micropapillary, solid). TCGA-STAD (441 WSIs) utilizes differentiation grade and structural anomalies like signet-ring cells as indicators. TCGA-UCEC (537 WSIs) prognosis links to FIGO grade, defined by the glandular-to-solid component ratio. TCGA-KIRC (519 WSIs) survival outcomes associate with Fuhrman grade (nuclear characteristics within clear cell architecture), while TCGA-KIRP (259 WSIs) depends on the structural integrity and arrangement of papillary cores. The task formulation is discrete time-to-event prediction across 4 non-overlapping intervals.

## C.2 Spatial-MNIST-Bag Construction

To rigorously separate statistical composition from spatial topology, a controlled synthetic benchmark is constructed from MNIST, termed Spatial-MNIST-Bag. Raw MNIST images are first encoded into 512-dimensional embeddings using a pre-trained ResNet-18. Each bag is represented as

$$
X = \{ ( h _ { i } , p _ { i } ) \} _ { i = 1 } ^ { N } , \qquad N = 5 0 , \qquad p _ { i } \in [ 0 , 1 ] ^ { 2 } ,\tag{31}
$$

where $h _ { i }$ is the digit embedding and $p _ { i }$ is its assigned two-dimensional coordinate. The benchmark contains two complementary datasets.

Dataset A: Pure Composition. Dataset A follows the standard i.i.d. MIL assumption and evaluates whether a model can detect a statistical anomaly without relying on structural cues. For every bag, all coordinates are sampled independently from $U ( 0 , 1 ) \times \bar { U } ( 0 , \bar { 1 } )$ ), so the spatial graph is random noise. The bag label depends only on digit composition: $Y = 1$ if at least one instance is digit $" 9 "$ , and $Y = 0$ otherwise. A successful model should behave as a permutation-invariant instance detector.

Dataset B: Pure Topology. Dataset B removes compositional shortcuts and forces the model to use spatial reasoning. Every bag contains the same predefined digit multiset: exactly one “1”, one ${ \bf \ddot { \Delta } } ^ { 6 } 3 { \bf \ddot { \Delta } } .$ , one “5”, one “7”, one “9”, and 45 even digits sampled from $\{ 0 , 2 , 4 , 6 , 8 \}$ . Therefore, a purely statistical MIL model observes the same bag-level composition for positive and negative samples and should remain close to chance.

The label in Dataset B is determined only by the coordinates of the five key odd digits. For positive bags $( Y ~ = ~ 1 )$ , a centroid $c \sim U ( [ 0 . 2 , { \bf \bar { 0 . 8 } } ] ^ { 2 } )$ is sampled and the coordinates of digits $^ { \mathrm { 6 6 } } \mathrm { 1 } ^ { \mathrm { 7 } } , \ ^ { \mathrm { 6 6 } } 3 ^ { \mathrm { 9 } }$ $^ { 6 6 5 ^ { 3 } } , ^ { 6 6 } 7 ^ { 3 }$ , and $" 9 "$ are drawn from $\bar { \mathcal { N } } ( c , \sigma ^ { 2 } I )$ with $\sigma = 0 . 0 5$ , creating a compact topological motif that forms a dense clique in the KNN graph. The remaining 45 background digits are uniformly scattered in $[ 0 , 1 ] ^ { 2 }$ . For negative bags $( Y = 0 )$ , all 50 instances, including the five key odd digits, are sampled uniformly in $[ 0 , \bar { 1 } ] ^ { 2 }$ with no spatial correlation. The task is therefore impossible to solve from composition alone and directly tests whether a model can detect a non-i.i.d. spatial motif.

Table 6: Spatial-MNIST-Bag Generation Parameters. Dataset A isolates composition, while Dataset B neutralizes composition and makes the label depend only on a spatial motif.
<table><tr><td>Item</td><td>Dataset A: Pure Composition</td><td>Dataset B: Pure Topology</td></tr><tr><td>Instance feature Bag size</td><td colspan="2">512-d ResNet-18 embedding of raw MNIST digits</td></tr><tr><td>Coordinate domain</td><td>N = 50 instances</td><td> $p _ { i } \in [ 0 , 1 ] ^ { 2 }$ </td></tr><tr><td>Composition</td><td>Random MNIST digits; positive if digit “9&quot; appears</td><td>Fixed multiset: one each of “1,3,5,7,9” plus 45 even digits</td></tr><tr><td>Positive rule</td><td>At least one digit “9&quot; in the bag</td><td>Key odd digits form a compact cluster</td></tr><tr><td>Negative rule</td><td>No digit “9” in the bag</td><td>All instances, including key digits, are uniformly scattered</td></tr><tr><td>Key motif Required capability</td><td>None; coordinates are nuisance noise Permutation-invariant instance detection</td><td>Centroid  $c \sim \dot { U } ( [ 0 . 2 , 0 . \check { 8 } ] ^ { 2 } ) ,$  key coordinates~  $\dot { N } ( c , 0 . 0 5 ^ { 2 } I )$  Detection of a non-i.i.d. spatial motif</td></tr></table>

This construction removes a common ambiguity in real WSI data. Dataset A checks whether the model still behaves like a standard MIL classifier when topology is irrelevant. Dataset B keeps the instance composition fixed and makes spatial arrangement the only usable signal. The two settings separate questions that are usually entangled in pathology benchmarks: whether a method recognizes discriminative patches, and whether it uses the physical arrangement of those patches once composition is no longer enough.

## C.3 Preprocessing and Feature Extraction

Data processing followed standard MIL protocols (e.g., CLAM, TransMIL). Tissue extraction was performed via Otsu’s thresholding in HSV space, followed by the generation of non-overlapping $2 5 6 \times 2 5 6$ patches at 20× magnification. Feature extraction used the UNI foundation model, a ViT-Large architecture pre-trained via self-supervised learning (DINOv2) on Mass-100K $( > 1 0 ^ { 8 }$ patches). This yields robust 1024-dimensional embeddings; all baselines used identical UNI features.

## C.4 Implementation Details

Experiments were conducted using PyTorch 1.13 and PyTorch Geometric on a single NVIDIA RTX 3090 (24GB).

Architecture. The statistical stream utilizes a codebook size of $K = 3 2$ with a learnable temperature τ (init 1.0). Prototype initialization uses MiniBatch K-Means on a random subset of training patch embeddings, which makes the initialization scalable to WSI collections with millions of patches. The topological stream employs a KNN graph $( K _ { k n n } = 8 )$ processed by a 2-layer GCN with ReLU and Dropout $( p \ : = \ : 0 . 2 5 )$ . Both streams map to a 512-dim hidden space, and node embeddings are aggregated by global mean pooling before the residual classifier.

Training Protocol. A decoupled two-stage strategy was employed. Stage 1 (Warmup) optimizes the Statistical Stream for 10 epochs. Stage 2 (Refinement) freezes the Statistical Stream and optimizes the Topological Stream (30 epochs for classification, 20 for survival). Optimization used Lookahead (Adam base, weight decay $1 \times 1 0 ^ { - 4 } )$ with a Cosine Annealing scheduler $( \mathrm { L R 2 \times 1 0 ^ { - 4 } } )$ Loss scaling factors were set to margin $m = 0 . 3$ and weight $\lambda = 1 . 0 .$

## D Additional Classification Metrics and Model Size

The main text reports Accuracy and AUC for compactness. Table 7 lists the model sizes used in the same comparison, and Table 8 reports the corresponding F1 scores on the four classification datasets.

Table 7: Parameter Counts. Model sizes are reported in millions of trainable parameters under the unified implementation.
<table><tr><td>Method</td><td>Params (M)</td></tr><tr><td>AB-MIL</td><td>0.59</td></tr><tr><td>CLAM-SB</td><td>0.79</td></tr><tr><td>DS-MIL</td><td>1.20</td></tr><tr><td>TransMIL</td><td>2.67</td></tr><tr><td>ILRA-MIL</td><td>3.68</td></tr><tr><td>MHIM-MIL</td><td>2.67</td></tr><tr><td>DGR-MIL</td><td>4.35</td></tr><tr><td>2DMambaMIL</td><td>1.27</td></tr><tr><td>ResTopoMIL</td><td>1.15</td></tr></table>

The parameter comparison rules out a simple capacity explanation. ResTopoMIL has 1.15M trainable parameters, close to DS-MIL and smaller than TransMIL, ILRA-MIL, MHIM-MIL, DGR-MIL, and 2DMambaMIL under the same implementation. The gains in the main tables therefore do not come from a larger model budget, but from how the statistical and topological streams are trained and constrained.

Table 8: F1 Scores on Four Classification Benchmarks. Mean ± std over 5 random seeds.
<table><tr><td>Method</td><td>BRACS</td><td>PANDA</td><td>TCGA-NSCLC</td><td>TCGA-BRCA</td></tr><tr><td>AB-MIL</td><td> $0 . 6 7 4 8 { \scriptstyle \pm . 0 1 8 5 }$ </td><td> $0 . 6 8 7 6 { \scriptstyle \pm . 0 0 7 9 }$ </td><td> $0 . 8 6 8 0 { \scriptstyle \pm . 0 0 6 8 }$ </td><td> $0 . 9 1 8 7 { \scriptstyle \pm . 0 1 0 6 }$ </td></tr><tr><td>CLAM-SB</td><td> $\underline { { 0 . 7 0 5 1 } } \pm . 0 2 8 7$ </td><td> $0 . 6 8 1 4 { \scriptstyle \pm . 0 0 5 0 }$ </td><td> $0 . 8 5 2 9 _ { \pm . 0 0 8 4 }$ </td><td> $\underline { { 0 . 9 2 6 8 } } { \pm . 0 1 1 2 }$ </td></tr><tr><td>DS-MIL</td><td> $0 . 6 0 6 8 { \scriptstyle \pm . 0 4 3 6 }$ </td><td> $\underline { { 0 . 6 9 1 1 } } \underline { { \pm } } . 0 2 3 1$ </td><td> $0 . 8 6 2 7 { \scriptstyle \pm . 0 0 9 9 }$ </td><td> $0 . 9 0 5 8 { \scriptstyle \pm . 0 2 0 8 }$ </td></tr><tr><td>TransMIL</td><td> $0 . 5 5 9 2 { \scriptstyle \pm . 0 3 9 8 }$ </td><td> $0 . 6 6 6 7 { \scriptstyle \pm . 0 1 3 3 }$ </td><td> $0 . 8 8 2 2 { \scriptstyle \pm . 0 1 3 2 }$ </td><td> $0 . 9 0 1 6 { \scriptstyle \pm . 0 1 6 3 }$ </td></tr><tr><td>ILRA-MIL</td><td> $0 . 5 5 2 6 { \scriptstyle \pm . 0 3 6 8 }$ </td><td> $0 . 6 8 8 0 { \scriptstyle \pm . 0 0 9 4 }$ </td><td> $0 . 8 8 2 9 { \scriptstyle \pm . 0 1 5 9 }$ </td><td> $0 . 9 1 0 8 { \scriptstyle \pm . 0 0 9 1 }$ </td></tr><tr><td>MHIM-MIL</td><td> $0 . 5 8 7 4 _ { \pm . 0 5 0 8 }$ </td><td> $0 . 6 3 7 2 { \scriptstyle \pm . 0 1 5 8 }$ </td><td> $0 . 8 7 0 3 { \scriptstyle \pm . 0 1 8 3 }$ </td><td> $0 . 9 1 1 7 { \scriptstyle \pm . 0 1 5 5 }$ </td></tr><tr><td>DGR-MIL</td><td> $0 . 6 6 8 4 { \scriptstyle \pm . 0 5 0 9 }$ </td><td> $0 . 6 3 9 9 { \scriptstyle \pm . 0 1 7 8 }$ </td><td> $0 . 8 8 2 9 { \scriptstyle \pm . 0 2 2 5 }$ </td><td> $0 . 9 0 9 9 { \scriptstyle \pm . 0 0 9 8 }$ </td></tr><tr><td>2DMambaMIL</td><td> $0 . 6 7 1 0 { \scriptstyle \pm . 0 4 5 0 }$ </td><td> $0 . 6 4 5 0 { \scriptstyle \pm . 0 1 6 0 }$ </td><td> $\underline { { 0 . 8 8 5 5 } } { \pm . 0 2 1 0 }$ </td><td> $0 . 9 1 5 0 { \scriptstyle \pm . 0 1 1 0 }$ </td></tr><tr><td>ResTopoMIL</td><td> $\mathbf { 0 . 7 1 4 2 { \scriptstyle \pm . 0 4 2 0 } }$ </td><td> $\mathbf { 0 . 7 0 9 7 { \scriptstyle \pm . 0 0 9 8 } }$ </td><td> $\mathbf { 0 . 9 1 3 5 } _ { \pm . 0 0 8 7 }$ </td><td> $\mathbf { 0 . 9 3 0 8 _ { \pm . 0 1 6 6 } }$ </td></tr></table>

The F1 scores are consistent with Accuracy and AUC. ResTopoMIL obtains the best F1 on all four classification datasets, including fine-grained BRACS and structure-heavy PANDA. Since F1 is more sensitive to class imbalance and minority-class errors than accuracy, the improvement is not only a ranking effect in AUC; it also reflects more balanced classification behavior.

## E Extended Ablation Analysis

The main text reports compact AUC-only ablations. Here we provide the full Accuracy/F1/AUC results with standard deviations, but split them into smaller tables so that each block answers a specific question.

Table 9: Optimization Ablation. Full metrics for residual decoupling and softer optimization alternatives.
<table><tr><td rowspan="2">Variant</td><td colspan="3">PANDA</td><td colspan="3">TCGA-BRCA</td></tr><tr><td>Acc</td><td>F1</td><td>AUC</td><td>Acc</td><td>F1</td><td>AUC</td></tr><tr><td>Stat. Only</td><td> $0 . 6 7 6 1 { \scriptstyle \pm . 0 0 6 2 }$ </td><td> $0 . 6 1 1 3 { \scriptstyle \pm . 0 0 7 5 }$ </td><td> $0 . 9 0 2 7 { \scriptstyle \pm . 0 0 5 1 }$ </td><td> $0 . 9 1 3 8 { \scriptstyle \pm . 0 0 6 9 }$ </td><td> $0 . 8 8 9 6 _ { \pm . 0 1 1 2 }$ </td><td> $0 . 9 4 8 6 _ { \pm . 0 0 3 5 }$ </td></tr><tr><td>Topo. Only</td><td> $0 . 7 0 4 2 { \scriptstyle \pm . 0 0 7 2 }$ </td><td> $0 . 6 5 9 8 { \scriptstyle \pm . 0 0 8 5 }$ </td><td> $0 . 9 2 1 5 { \scriptstyle \pm . 0 0 5 5 }$ </td><td> $0 . 9 3 8 2 { \scriptstyle \pm . 0 0 6 8 }$ </td><td> $0 . 8 9 4 5 { \scriptstyle \pm . 0 1 0 2 }$ </td><td> $0 . 9 6 0 8 { \scriptstyle \pm . 0 0 4 8 }$ </td></tr><tr><td>Joint Opt.</td><td> $0 . 7 3 9 7 { \scriptstyle \pm . 0 0 5 5 }$ </td><td> $0 . 7 0 2 5 { \scriptstyle \pm . 0 0 6 5 }$ </td><td> $0 . 9 2 9 9 _ { \pm . 0 0 5 2 }$ </td><td> $0 . 9 4 3 2 { \scriptstyle \pm . 0 0 5 5 }$ </td><td> $0 . 9 3 1 8 { \scriptstyle \pm . 0 0 6 1 }$ </td><td> $0 . 9 7 7 3 { \scriptstyle \pm . 0 0 2 2 }$ </td></tr><tr><td>+ Stat-Dropout (PDL)</td><td> $0 . 7 4 1 5 { \scriptstyle \pm . 0 0 7 0 }$ </td><td> $0 . 7 0 3 8 { \scriptstyle \pm . 0 0 7 5 }$ </td><td> $0 . 9 3 2 5 { \scriptstyle \pm . 0 0 6 5 }$ </td><td> $0 . 9 4 4 8 _ { \pm . 0 0 4 8 }$ </td><td> $0 . 9 3 0 2 { \scriptstyle \pm . 0 0 6 8 }$ </td><td> $0 . 9 7 8 0 { \scriptstyle \pm . 0 0 3 0 }$ </td></tr><tr><td> $+ \mathrm { C u r r i c u l u m } \mathrm { S c h e d } .$ </td><td> $0 . 7 4 4 2 { \scriptstyle \pm . 0 0 5 2 }$ </td><td> $0 . 7 0 5 5 { \scriptstyle \pm . 0 0 5 8 }$ </td><td> $0 . 9 3 4 8 { \scriptstyle \pm . 0 0 4 5 }$ </td><td> $0 . 9 4 6 5 { \scriptstyle \pm . 0 0 4 5 }$ </td><td> $0 . 9 3 1 5 { \scriptstyle \pm . 0 0 5 5 }$ </td><td> $0 . 9 7 8 5 { \scriptstyle \pm . 0 0 1 8 }$ </td></tr><tr><td>+ Hard Instance Mining</td><td> $0 . 7 4 8 5 { \scriptstyle \pm . 0 0 6 2 }$ </td><td> $0 . 7 0 9 1 { \scriptstyle \pm . 0 0 6 5 }$ </td><td> $0 . 9 3 7 2 _ { \pm . 0 0 5 0 }$ </td><td> $0 . 9 4 9 2 _ { \pm . 0 0 5 2 }$ </td><td> $0 . 9 2 8 8 _ { \pm . 0 0 6 5 }$ </td><td> $0 . 9 7 9 2 { \scriptstyle \pm . 0 0 2 1 }$ </td></tr><tr><td>Joint Opt. (Multi-LR)</td><td> $0 . 7 4 6 8 { \scriptstyle \pm . 0 0 5 1 }$ </td><td> $0 . 7 0 8 4 _ { \pm . 0 0 4 5 }$ </td><td> $0 . 9 3 5 2 _ { \pm . 0 0 4 8 }$ </td><td> $0 . 9 4 7 5 _ { \pm . 0 0 4 2 }$ </td><td> $0 . 9 2 3 5 _ { \pm . 0 0 5 4 }$ </td><td> $0 . 9 7 8 6 _ { \pm . 0 0 1 8 }$ </td></tr><tr><td>ResTopoMIL</td><td> $\mathbf { 0 . 7 5 4 6 { \scriptstyle \pm . 0 0 9 4 } }$ </td><td> $\mathbf { 0 . 7 0 9 7 { \scriptstyle \pm . 0 0 9 8 } }$ </td><td> $\mathbf { 0 . 9 4 2 6 { \scriptstyle \pm . 0 0 1 0 } }$ </td><td> $\mathbf { 0 . 9 5 6 8 _ { \pm . 0 0 9 5 } }$ </td><td> $\mathbf { 0 . 9 3 0 8 _ { \pm . 0 1 6 6 } }$ </td><td> $\mathbf { 0 . 9 8 3 8 _ { \pm . 0 0 4 9 } }$ </td></tr></table>

Table 9 asks whether easier training heuristics can fix spatial blindness. Topo. Only is not a replacement for the statistical stream: it is close to Stat. Only on PANDA and higher on TCGA-BRCA, but does not provide the same balanced behavior as the residually decoupled model. The graph branch should model the structural residual left by a frozen compositional predictor, not solve the whole task alone. Stat-Dropout, curriculum scheduling, hard instance mining, and Multi-LR improve over vanilla joint training in several columns, but they are softer interventions: they rescale, perturb, schedule, or reweight the joint residual, whereas ResTopoMIL fixes the compositional anchor and changes what error the graph branch is asked to explain. The benefit is clearest when these endpoint metrics are read together with the shuffle and localization analyses, which test whether high AUC i accompanied by coordinate-dependent behavior.

## E.1 Training Dynamics and Computational Cost

The two-stage schedule introduces a real training constraint, so it should not be treated as free. In our implementation, Stage 1 trains the statistical stream for 10 epochs. Stage 2 then freezes this stream and trains the graph branch for 30 epochs on classification tasks and 20 epochs on survival tasks. Compared with a single joint run using the same total epoch budget, the main extra work in Stage 2 is the shuffled-graph forward pass required by $\mathcal { L } _ { t e x } .$ This cost is incurred only during training. At test time, the shuffled view is not constructed; inference uses one statistical forward pass, one graph forward pass, and logit summation.

The benefit of the schedule is visible in both convergence behavior and final metrics. Figure 4 shows that the topological gradient decays under joint training but rebounds after the statistical stream is frozen. This rebound is important: it means Stage 2 is not merely continuing the same optimization path with a different learning rate, but exposing an error signal that joint training had largely suppressed. The ablation table gives the endpoint check. Joint Opt. reaches 0.9299 AUC on PANDA and 0.9773 on TCGA-BRCA. A 10× topological learning-rate multiplier improves these values to 0.9352 and 0.9786, but the broader table shows that learning-rate balancing alone does not give the same stable, spatially sensitive solution. Curriculum scheduling and hard instance mining show the same pattern: they reduce the problem, but do not remove the competition between the streams.

The schedule also affects stability. ResTopoMIL uses a fixed residual target in Stage 2, whereas dropout and curriculum variants create a less stable target: the statistical signal is either randomly weakened or still changing while the graph branch is being trained. This does not mean two-stage training is always easier to tune. It adds one schedule boundary, requires choosing the warmup length, and depends on the statistical anchor being strong enough to remove genuine compositional signal. The main result is therefore not that two-stage training reduces engineering complexity, but that the added constraint buys a better optimization problem for topology.

Table 10: Fusion and Interaction Ablation. Full metrics for different ways of combining statistical and topological evidence.
<table><tr><td rowspan="2">Variant</td><td colspan="3">PANDA</td><td colspan="3">TCGA-BRCA</td></tr><tr><td>Acc</td><td>F1</td><td>AUC</td><td>Acc</td><td>F1</td><td>AUC</td></tr><tr><td>Gated Fusion</td><td> $0 . 7 3 8 9 { \scriptstyle \pm . 0 0 6 8 }$ </td><td> $0 . 6 9 9 1 { \scriptstyle \pm . 0 0 7 5 }$ </td><td> $0 . 9 2 2 5 { \scriptstyle \pm . 0 0 6 2 }$ </td><td> $0 . 9 4 3 2 { \scriptstyle \pm . 0 0 5 1 }$ </td><td> $0 . 9 0 6 8 _ { \pm . 0 0 6 5 }$ </td><td> $0 . 9 8 0 2 _ { \pm . 0 0 4 2 }$ </td></tr><tr><td>MoE</td><td> $0 . 7 4 7 1 { \scriptstyle \pm . 0 0 5 2 }$ </td><td> $0 . 6 9 1 8 { \scriptstyle \pm . 0 0 4 8 }$ </td><td> $0 . 9 3 0 5 _ { \pm . 0 0 3 5 }$ </td><td> $0 . 9 4 3 2 { \scriptstyle \pm . 0 0 4 5 }$ </td><td> $0 . 9 1 0 8 { \scriptstyle \pm . 0 0 5 5 }$ </td><td> $\mathbf { 0 . 9 8 9 7 { \scriptstyle \pm . 0 0 2 8 } }$ </td></tr><tr><td>Indep. Clf</td><td> $0 . 7 3 5 2 { \scriptstyle \pm . 0 0 8 5 }$ </td><td> $0 . 6 8 4 5 { \scriptstyle \pm . 0 0 8 1 }$ </td><td> $0 . 9 2 5 6 _ { \pm . 0 0 6 2 }$ </td><td> $0 . 9 4 6 5 { \scriptstyle \pm . 0 0 5 8 }$ </td><td> $0 . 9 1 1 2 { \scriptstyle \pm . 0 0 9 5 }$ </td><td> $0 . 9 8 4 5 _ { \pm . 0 0 2 5 }$ </td></tr><tr><td>ResTopoMIL</td><td> $\mathbf { 0 . 7 5 4 6 { \scriptstyle \pm . 0 0 9 4 } }$ </td><td> $\mathbf { 0 . 7 0 9 7 { \scriptstyle \pm . 0 0 9 8 } }$ </td><td> $\mathbf { 0 . 9 4 2 6 { \scriptstyle \pm . 0 0 1 0 } }$ </td><td> $\mathbf { 0 . 9 5 6 8 _ { \pm . 0 0 9 5 } }$ </td><td> $\mathbf { 0 . 9 3 0 8 _ { \pm . 0 1 6 6 } }$ </td><td> $0 . 9 8 3 8 { \scriptstyle \pm . 0 0 4 9 }$ </td></tr></table>

Table 10 shows that the benefit is not obtained by adding a generic interaction head. Gated Fusion performs poorly, suggesting that early adaptive mixing can reintroduce the same shortcut: the fusion gate may favor the stream that already explains the label. MoE is stronger in AUC on TCGA-BRCA, but it is weaker on PANDA AUC and TCGA-BRCA Accuracy/F1. Independent classifiers are also insufficient. The result supports the design choice that the topological stream should be trained as a residual correction, not as another branch that competes with statistics from the start.

Table 11: Architecture and Validity Ablation. Full metrics for graph-backbone and sanity-control variants.
<table><tr><td rowspan="2">Variant</td><td colspan="3">PANDA</td><td colspan="3">TCGA-BRCA</td></tr><tr><td>Acc</td><td>F1</td><td>AUC</td><td>Acc</td><td>F1</td><td>AUC</td></tr><tr><td>GAT (Complex)</td><td> $0 . 7 4 8 9 _ { \pm . 0 0 6 2 }$ </td><td> $0 . 7 0 7 4 { \scriptstyle \pm . 0 0 5 8 }$ </td><td> $0 . 9 4 1 9 _ { \pm . 0 0 4 1 }$ </td><td> $0 . 9 4 3 2 { \scriptstyle \pm . 0 0 5 2 }$ </td><td> $0 . 9 0 6 8 _ { \pm . 0 0 6 1 }$ </td><td> $\mathbf { 0 . 9 8 8 9 } _ { \pm . 0 0 2 5 }$ </td></tr><tr><td>Random Graph</td><td> $0 . 6 8 1 5 { \scriptstyle \pm . 0 1 4 8 }$ </td><td> $0 . 5 3 6 2 { \scriptstyle \pm . 0 2 6 0 }$ </td><td> $0 . 8 2 8 6 { \scriptstyle \pm . 0 1 5 0 }$ </td><td> $0 . 8 5 2 3 { \scriptstyle \pm . 0 1 8 0 }$ </td><td> $0 . 6 7 4 9 { \scriptstyle \pm . 0 2 5 0 }$ </td><td> $0 . 8 5 6 3 { \scriptstyle \pm . 0 1 1 0 }$ </td></tr><tr><td>Fixed Proto.</td><td> $0 . 7 3 3 5 { \scriptstyle \pm . 0 0 6 5 }$ </td><td> $0 . 6 7 8 1 { \scriptstyle \pm . 0 0 7 2 }$ </td><td> $0 . 9 3 4 9 _ { \pm . 0 0 5 5 }$ </td><td> $0 . 9 4 4 6 _ { \pm . 0 0 5 6 }$ </td><td> $0 . 9 0 9 8 { \scriptstyle \pm . 0 0 9 4 }$ </td><td> $0 . 9 7 4 6 { \scriptstyle \pm . 0 0 3 5 }$ </td></tr><tr><td>ResTopoMIL</td><td> $\mathbf { 0 . 7 5 4 6 { \scriptstyle \pm . 0 0 9 4 } }$ </td><td> $\mathbf { 0 . 7 0 9 } 7 _ { \pm . 0 0 9 8 }$ </td><td> $\mathbf { 0 . 9 4 2 6 { \scriptstyle \pm . 0 0 1 0 } }$ </td><td> $\mathbf { 0 . 9 5 6 8 { \scriptstyle \pm . 0 0 9 5 } }$ </td><td> $\mathbf { 0 . 9 3 0 8 _ { \pm . 0 1 6 6 } }$ </td><td> $0 . 9 8 3 8 { \scriptstyle \pm . 0 0 4 9 }$ </td></tr></table>

Table 11 separates architectural complexity from the proposed training logic. Replacing the 2-layer GCN with a GAT does not improve PANDA and only modestly changes TCGA-BRCA, so the gain is not explained by a more expressive graph operator. Random Graph trains the model after replacing the true neighborhood graph with random edges; its large drop in F1 and AUC shows that merely adding noisy graph connectivity is not enough. Fixed Proto. keeps the prototype assignment fixed rather than learned, and remains below ResTopoMIL on PANDA and on TCGA-BRCA Accuracy/F1. These controls support the paper’s position that the contribution is not a complex graph module; it is the explicit residual formulation and the way topology is forced to remain spatial.

Table 12: Hyperparameter and Constraint Sensitivity. Full metrics for prototype number, graph degree, margin, and texture-loss removal.
<table><tr><td rowspan="2">Variant</td><td colspan="3">PANDA</td><td colspan="3">TCGA-BRCA</td></tr><tr><td>Acc</td><td>F1</td><td>AUC</td><td>Acc</td><td>F1</td><td>AUC</td></tr><tr><td> $K _ { p r o t o } = 8$ </td><td> $0 . 7 4 0 5 _ { \pm . 0 0 7 1 }$ </td><td> $0 . 6 9 1 2 { \scriptstyle \pm . 0 0 6 3 }$ </td><td> $0 . 9 2 8 5 _ { \pm . 0 0 5 1 }$ </td><td> $0 . 9 4 8 2 { \scriptstyle \pm . 0 0 5 5 }$ </td><td> $0 . 9 1 4 5 _ { \pm . 0 0 8 9 }$ </td><td> $0 . 9 8 5 2 { \scriptstyle \pm . 0 0 2 1 }$ </td></tr><tr><td> $K _ { p r o t o } = 1 6$ </td><td> $0 . 7 5 1 2 { \scriptstyle \pm . 0 0 5 8 }$ </td><td> $0 . 7 0 6 5 { \scriptstyle \pm . 0 0 4 2 }$ </td><td> $0 . 9 3 9 2 _ { \pm . 0 0 3 4 }$ </td><td> $0 . 9 5 1 5 { \scriptstyle \pm . 0 0 3 8 }$ </td><td> $0 . 9 2 2 4 _ { \pm . 0 0 6 1 }$ </td><td> $0 . 9 8 8 9 { \scriptstyle \pm . 0 0 1 5 }$ </td></tr><tr><td> $K _ { p r o t o } = 6 4$ </td><td> $0 . 7 5 4 4 { \scriptstyle \pm . 0 0 5 2 }$ </td><td> $\mathbf { 0 . 7 0 9 8 _ { \pm . 0 0 4 8 } }$ </td><td> $0 . 9 4 1 5 _ { \pm . 0 0 3 1 }$ </td><td> $0 . 9 5 2 8 { \scriptstyle \pm . 0 0 4 2 }$ </td><td> $0 . 9 2 4 5 _ { \pm . 0 0 6 5 }$ </td><td> $0 . 9 8 9 2 _ { \pm . 0 0 1 8 }$ </td></tr><tr><td> $K _ { k n n } = 4$ </td><td> $0 . 7 4 3 8 { \scriptstyle \pm . 0 0 6 2 }$ </td><td> $0 . 6 9 5 5 { \scriptstyle \pm . 0 0 5 9 }$ </td><td> $0 . 9 3 1 2 { \scriptstyle \pm . 0 0 4 5 }$ </td><td> $0 . 9 5 0 2 { \scriptstyle \pm . 0 0 4 9 }$ </td><td> $0 . 9 1 9 2 { \scriptstyle \pm . 0 0 7 1 }$ </td><td> $0 . 9 8 7 5 { \scriptstyle \pm . 0 0 1 9 }$ </td></tr><tr><td> $K _ { k n n } = 1 6$ </td><td> $0 . 7 4 9 5 { \scriptstyle \pm . 0 0 5 5 }$ </td><td> $0 . 7 0 2 4 { \scriptstyle \pm . 0 0 5 1 }$ </td><td> $0 . 9 3 6 8 { \scriptstyle \pm . 0 0 3 7 }$ </td><td> $0 . 9 5 1 8 { \scriptstyle \pm . 0 0 4 4 }$ </td><td> $0 . 9 2 1 8 { \scriptstyle \pm . 0 0 6 8 }$ </td><td> $0 . 9 8 8 2 { \scriptstyle \pm . 0 0 2 2 }$ </td></tr><tr><td> $\mathbf { M a r g i n } m = 0 . 1$ </td><td> $0 . 7 4 6 2 { \scriptstyle \pm . 0 0 6 0 }$ </td><td> $0 . 6 9 8 8 { \scriptstyle \pm . 0 0 5 5 }$ </td><td> $0 . 9 3 2 5 { \scriptstyle \pm . 0 0 4 8 }$ </td><td> $0 . 9 5 1 0 { \scriptstyle \pm . 0 0 4 0 }$ </td><td> $0 . 9 2 0 5 _ { \pm . 0 0 6 2 }$ </td><td> $0 . 9 8 8 0 { \scriptstyle \pm . 0 0 1 8 }$ </td></tr><tr><td> $\mathrm { M a r g i n } m = 0 . 2$ </td><td> $0 . 7 5 3 5 { \scriptstyle \pm . 0 0 4 8 }$ </td><td> $0 . 7 0 8 2 { \scriptstyle \pm . 0 0 4 1 }$ </td><td> $0 . 9 4 0 8 { \scriptstyle \pm . 0 0 3 5 }$ </td><td> $0 . 9 5 3 2 { \scriptstyle \pm . 0 0 3 5 }$ </td><td> $0 . 9 2 4 8 { \scriptstyle \pm . 0 0 5 5 }$ </td><td> $0 . 9 8 9 6 _ { \pm . 0 0 1 4 }$ </td></tr><tr><td> $\mathbf { M a r g i n } m = 0 . 4$ </td><td> $\mathbf { 0 . 7 5 4 8 _ { \pm . 0 0 4 9 } }$ </td><td> $0 . 7 0 9 4 { \scriptstyle \pm . 0 0 4 2 }$ </td><td> $0 . 9 4 1 7 { \scriptstyle \pm . 0 0 3 3 }$ </td><td> $0 . 9 5 3 6 { \scriptstyle \pm . 0 0 3 6 }$ </td><td> $0 . 9 2 5 1 { \scriptstyle \pm . 0 0 5 3 }$ </td><td> $\mathbf { 0 . 9 8 9 8 } _ { \pm . 0 0 1 4 }$ </td></tr><tr><td> $\mathrm { M a r g i n } m = 0 . 5$ </td><td> $0 . 7 5 0 5 { \scriptstyle \pm . 0 0 5 2 }$ </td><td> $0 . 7 0 4 2 { \scriptstyle \pm . 0 0 4 9 }$ </td><td> $0 . 9 3 8 5 _ { \pm . 0 0 3 9 }$ </td><td> $0 . 9 5 2 5 { \scriptstyle \pm . 0 0 3 8 }$ </td><td> $0 . 9 2 3 0 { \scriptstyle \pm . 0 0 5 8 }$ </td><td> $0 . 9 8 9 0 { \scriptstyle \pm . 0 0 1 6 }$ </td></tr><tr><td>w/o  $\mathcal { L } _ { t e x }$ </td><td> $0 . 7 1 5 5 { \scriptstyle \pm . 0 0 7 2 }$ </td><td> $0 . 6 9 7 0 { \scriptstyle \pm . 0 0 8 1 }$ </td><td> $0 . 9 1 4 7 { \scriptstyle \pm . 0 0 6 5 }$ </td><td> $0 . 9 4 1 8 { \scriptstyle \pm . 0 0 5 4 }$ </td><td> $0 . 9 0 6 5 _ { \pm . 0 0 9 8 }$ </td><td> $0 . 9 7 6 2 { \scriptstyle \pm . 0 0 3 8 }$ </td></tr><tr><td>ResTopoMIL</td><td> $0 . 7 5 4 6 { \scriptstyle \pm . 0 0 9 4 }$ </td><td> $0 . 7 0 9 7 { \scriptstyle \pm . 0 0 9 8 }$ </td><td> $\mathbf { 0 . 9 4 2 6 _ { \pm . 0 0 1 0 } }$ </td><td> $\mathbf { 0 . 9 5 6 8 _ { \pm . 0 0 9 5 } }$ </td><td> $\mathbf { 0 . 9 3 0 8 _ { \pm . 0 1 6 6 } }$ </td><td> $0 . 9 8 3 8 { \scriptstyle \pm . 0 0 4 9 }$ </td></tr></table>

Table 12 shows a broad but interpretable stability pattern. Very small prototype dictionaries and sparse KNN graphs underfit the statistical or spatial summaries, while larger settings approach the default and occasionally edge it in individual columns. The margin sweep is not a fragile optimum search: $m = 0 . 2$ and $m = 0 . 4$ are close to the default, whereas too small or too large a margin weakens residual separation. Removing $\mathcal { L } _ { t e x }$ is the strongest negative control among these rows,

which supports the role of the shuffle constraint in keeping the residual branch aligned with spatial arrangement.

## F Cross-Backbone Generalization on CTransPath

To check whether the ranking depends on the UNI encoder, we repeat the classification benchmarks with CTransPath features under the same implementation. We report full comparisons on NSCLC/TCGA-BRCA and BRACS/PANDA. ResTopoMIL remains strongest or near-strongest across the reported metrics, suggesting that the gain is not tied to a particular patch encoder.

Table 13: CTransPath Results on NSCLC and TCGA-BRCA. Accuracy, F1, and AUC are reported as mean ± std.
<table><tr><td>Method</td><td></td><td></td><td></td><td></td><td></td><td>NSCLC (Acc) NSCLC (F1) NSCLC (AUC) TCGA-BRCA (Acc) TCGA-BRCA (F1) TCGA-BRCA (AUC)</td></tr><tr><td>AB-MIL</td><td> $0 . 8 8 1 0 { \scriptstyle \pm . 0 0 7 5 }$ </td><td> $0 . 8 5 1 0 { \scriptstyle \pm . 0 0 8 2 }$ </td><td> $0 . 9 4 1 2 { \scriptstyle \pm . 0 0 8 5 }$ </td><td> $0 . 9 2 5 0 { \scriptstyle \pm . 0 0 7 5 }$ </td><td> $0 . 9 0 1 0 { \scriptstyle \pm . 0 1 1 5 }$ </td><td> $0 . 9 6 0 5 { \scriptstyle \pm . 0 0 3 5 }$ </td></tr><tr><td>CLAM-SB</td><td> $0 . 8 8 5 5 _ { \pm . 0 0 9 2 }$ </td><td> $0 . 8 5 8 0 _ { \pm . 0 0 9 5 }$ </td><td> $0 . 9 4 8 5 _ { \pm . 0 0 7 5 }$ </td><td> $0 . 9 1 5 0 _ { \pm . 0 0 8 5 }$ </td><td> $0 . 9 0 8 5 _ { \pm . 0 1 2 5 }$ </td><td> $0 . 9 6 8 5 _ { \pm . 0 0 4 5 }$ </td></tr><tr><td>DS-MIL</td><td> $0 . 8 8 5 0 { \scriptstyle \pm . 0 1 1 5 }$ </td><td> $0 . 8 5 3 0 { \scriptstyle \pm . 0 0 9 8 }$ </td><td> $0 . 9 4 3 0 { \scriptstyle \pm . 0 0 9 2 }$ </td><td> $0 . 8 9 1 0 { \scriptstyle \pm . 0 2 2 5 }$ </td><td> $0 . 9 1 1 0 { \scriptstyle \pm . 0 1 3 5 }$ </td><td> $0 . 9 6 2 2 { \scriptstyle \pm . 0 0 4 1 }$ </td></tr><tr><td>TransMIL</td><td> $0 . 8 9 5 0 { \scriptstyle \pm . 0 1 4 5 }$ </td><td> $0 . 8 6 5 5 { \scriptstyle \pm . 0 1 4 0 }$ </td><td> $0 . 9 5 1 5 { \scriptstyle \pm . 0 0 7 5 }$ </td><td> $0 . 9 2 5 0 { \scriptstyle \pm . 0 1 1 5 }$ </td><td> $0 . 8 8 5 0 { \scriptstyle \pm . 0 1 7 5 }$ </td><td> $0 . 9 6 4 5 { \scriptstyle \pm . 0 1 6 0 }$ </td></tr><tr><td>ILRA-MIL</td><td> $0 . 8 8 2 0 { \scriptstyle \pm . 0 1 6 5 }$ </td><td> $0 . 8 7 1 0 { \scriptstyle \pm . 0 1 6 2 }$ </td><td> $0 . 9 4 8 5 { \scriptstyle \pm . 0 0 8 2 }$ </td><td> $0 . 9 3 1 0 { \scriptstyle \pm . 0 0 6 5 }$ </td><td> $0 . 8 9 5 0 { \scriptstyle \pm . 0 1 0 5 }$ </td><td> $0 . 9 5 2 5 { \scriptstyle \pm . 0 1 1 5 }$ </td></tr><tr><td>MHIM-MIL</td><td> $0 . 8 9 1 0 { \scriptstyle \pm . 0 1 9 5 }$ </td><td> $0 . 8 5 5 0 { \scriptstyle \pm . 0 1 9 5 }$ </td><td> $0 . 9 6 1 0 { \scriptstyle \pm . 0 0 4 8 }$ </td><td> $0 . 9 3 1 0 { \scriptstyle \pm . 0 1 0 5 }$ </td><td> $0 . 8 9 5 0 { \scriptstyle \pm . 0 1 6 5 }$ </td><td> $0 . 9 6 5 8 { \scriptstyle \pm . 0 1 4 2 }$ </td></tr><tr><td>DGR-MIL</td><td> $0 . 8 8 5 0 { \scriptstyle \pm . 0 2 3 5 }$ </td><td> $0 . 8 6 5 0 { \scriptstyle \pm . 0 2 4 5 }$ </td><td> $0 . 9 2 2 0 { \scriptstyle \pm . 0 2 5 0 }$ </td><td> $0 . 8 9 5 0 { \scriptstyle \pm . 0 1 1 5 }$ </td><td> $0 . 8 9 5 0 { \scriptstyle \pm . 0 1 5 5 }$ </td><td> $0 . 9 6 0 5 { \scriptstyle \pm . 0 2 1 0 }$ </td></tr><tr><td>2DMambaMIL</td><td> $0 . 8 9 1 0 { \scriptstyle \pm . 0 1 8 5 }$ </td><td> $0 . 8 7 5 0 { \scriptstyle \pm . 0 2 1 5 }$ </td><td> $0 . 9 3 1 5 { \scriptstyle \pm . 0 1 9 5 }$ </td><td> $0 . 9 3 5 0 { \scriptstyle \pm . 0 0 7 5 }$ </td><td> $0 . 9 0 1 0 { \scriptstyle \pm . 0 1 2 5 }$ </td><td> $0 . 9 6 4 0 { \scriptstyle \pm . 0 1 8 5 }$ </td></tr><tr><td>ResTopoMIL (Ours)</td><td> $\mathbf { 0 . 9 0 1 0 _ { \pm . 0 0 9 5 } }$ </td><td> $\mathbf { 0 . 8 9 5 5 _ { \pm . 0 0 9 5 } }$ </td><td> $\mathbf { 0 . 9 6 8 5 _ { \pm . 0 0 3 2 } }$ </td><td> $\mathbf { 0 . 9 4 2 0 { \scriptstyle \pm . 0 1 0 5 } }$ </td><td> $\mathbf { 0 . 9 1 5 0 } _ { \pm . 0 1 7 5 }$ </td><td> $\mathbf { 0 . 9 7 5 0 } _ { \pm . 0 0 5 2 }$ </td></tr></table>

Table 14: CTransPath Results on BRACS and PANDA. Accuracy, F1, and AUC are reported as mean ± std.
<table><tr><td>Method</td><td></td><td></td><td></td><td>BRACS (Acc) BRACS (F1) BRACS (AUC) PANDA (Acc) PANDA (F1) PANDA (AUC)</td><td></td><td></td></tr><tr><td>AB-MIL</td><td> $0 . 7 2 5 0 _ { \pm . 0 2 1 0 }$ </td><td> $0 . 6 4 1 5 _ { \pm . 0 2 2 5 }$ </td><td> $0 . 8 5 1 2 _ { \pm . 0 1 0 5 }$ </td><td> $0 . 6 8 5 4 _ { \pm . 0 0 9 2 }$ </td><td> $0 . 6 6 5 2 _ { \pm . 0 0 8 5 }$ </td><td> $0 . 9 1 8 5 _ { \pm . 0 0 4 2 }$ </td></tr><tr><td>CLAM-SB</td><td> $0 . 7 1 8 5 { \scriptstyle \pm . 0 1 9 5 }$ </td><td> $0 . 6 1 2 0 { \scriptstyle \pm . 0 2 1 5 }$ </td><td> $0 . 8 5 8 0 { \scriptstyle \pm . 0 1 2 5 }$ </td><td> $0 . 6 8 2 0 { \scriptstyle \pm . 0 0 8 8 }$ </td><td> $0 . 6 6 1 0 { \scriptstyle \pm . 0 0 6 5 }$ </td><td> $0 . 9 1 5 0 { \scriptstyle \pm . 0 0 3 8 }$ </td></tr><tr><td>DS-MIL</td><td> $0 . 7 3 0 5 { \scriptstyle \pm . 0 2 8 5 }$ </td><td> $0 . 6 5 2 5 { \scriptstyle \pm . 0 3 2 0 }$ </td><td> $0 . 8 5 5 0 { \scriptstyle \pm . 0 1 5 2 }$ </td><td> $0 . 6 8 1 5 { \scriptstyle \pm . 0 0 6 5 }$ </td><td> $0 . 6 6 1 2 { \scriptstyle \pm . 0 0 5 2 }$ </td><td> $0 . 9 2 0 5 { \scriptstyle \pm . 0 0 6 8 }$ </td></tr><tr><td>TransMIL</td><td> $0 . 6 1 5 0 { \scriptstyle \pm . 0 1 8 5 }$ </td><td> $0 . 5 2 5 5 _ { \pm . 0 4 1 5 }$ </td><td> $0 . 8 1 2 5 _ { \pm . 0 1 2 8 }$ </td><td> $0 . 6 8 5 5 { \scriptstyle \pm . 0 1 0 5 }$ </td><td> $0 . 6 4 2 0 { \scriptstyle \pm . 0 1 5 0 }$ </td><td> $0 . 9 0 2 4 { \scriptstyle \pm . 0 0 5 5 }$ </td></tr><tr><td>ILRA-MIL</td><td> $0 . 5 9 0 0 { \scriptstyle \pm . 0 2 5 0 }$ </td><td> $0 . 5 1 0 5 { \scriptstyle \pm . 0 3 8 5 }$ </td><td> $0 . 7 8 5 0 { \scriptstyle \pm . 0 2 1 0 }$ </td><td> $0 . 7 2 5 5 _ { \pm . 0 0 8 0 }$ </td><td> $0 . 6 7 1 0 { \scriptstyle \pm . 0 1 1 5 }$ </td><td> $0 . 9 1 2 0 { \scriptstyle \pm . 0 0 4 5 }$ </td></tr><tr><td>MHIM-MIL</td><td> $0 . 6 3 5 0 { \scriptstyle \pm . 0 3 8 0 }$ </td><td> $0 . 5 5 1 0 { \scriptstyle \pm . 0 5 2 0 }$ </td><td> $0 . 8 0 1 5 { \scriptstyle \pm . 0 1 6 5 }$ </td><td> $0 . 6 7 5 0 { \scriptstyle \pm . 0 1 3 5 }$ </td><td> $0 . 6 1 5 5 { \scriptstyle \pm . 0 1 8 5 }$ </td><td> $0 . 8 9 8 5 { \scriptstyle \pm . 0 0 3 5 }$ </td></tr><tr><td>DGR-MIL</td><td> $0 . 6 8 0 0 { \scriptstyle \pm . 0 3 5 0 }$ </td><td> $0 . 6 3 5 0 { \scriptstyle \pm . 0 4 8 5 }$ </td><td> $0 . 7 9 5 0 { \scriptstyle \pm . 0 3 0 5 }$ </td><td> $0 . 6 7 2 0 { \scriptstyle \pm . 0 1 4 5 }$ </td><td> $0 . 6 1 8 5 _ { \pm . 0 1 9 5 }$ </td><td> $0 . 8 8 4 5 _ { \pm . 0 0 9 5 }$ </td></tr><tr><td>2DMambaMIL</td><td> $0 . 6 9 0 5 { \scriptstyle \pm . 0 2 8 5 }$ </td><td> $0 . 6 4 2 0 { \scriptstyle \pm . 0 4 6 5 }$ </td><td> $0 . 8 0 8 0 { \scriptstyle \pm . 0 2 5 5 }$ </td><td> $0 . 6 2 5 5 { \scriptstyle \pm . 0 1 8 5 }$ </td><td> $0 . 6 8 1 5 { \scriptstyle \pm . 0 1 2 5 }$ </td><td> $0 . 8 9 1 2 { \scriptstyle \pm . 0 0 8 2 }$ </td></tr><tr><td>ResTopoMIL (Ours)</td><td> $\mathbf { 0 . 7 3 8 5 _ { \pm . 0 2 5 5 } }$ </td><td> $\mathbf { 0 . 6 9 5 2 } _ { \pm . 0 4 3 5 }$ </td><td> $\mathbf { 0 . 8 7 1 5 _ { \pm . 0 0 8 5 } }$ </td><td> $\mathbf { 0 . 7 3 5 0 } _ { \pm . 0 1 1 0 }$ </td><td> $\mathbf { 0 . 6 8 8 5 _ { \pm . 0 1 0 5 } }$ </td><td> $\mathbf { 0 . 9 2 5 8 _ { \pm . 0 0 2 5 } }$ </td></tr></table>

The cross-backbone result is consistent with the UNI-based main table, but it is also useful for a more specific reason. CTransPath changes the patch representation while leaving the MIL aggregator comparison unchanged. Table 13 shows that ResTopoMIL remains best on NSCLC and TCGA-BRCA across Accuracy, F1, and AUC, so the improvement is not an artifact of one particular feature extractor.

Table 14 is even more diagnostic because BRACS and PANDA rely more heavily on tissue architec ture. ResTopoMIL improves BRACS AUC from the strongest baseline value of 0.8580 to 0.8715, and improves PANDA AUC from 0.9205 to 0.9258. The BRACS F1 gain is also large, moving from 0.6525 for the strongest baseline to 0.6952. These improvements are not huge in every column, but they are consistent across accuracy, F1, and AUC, which is the expected pattern if topology provides complementary evidence rather than replacing the patch encoder.

## G Additional Shuffle Sensitivity Across WSI Benchmarks

Before showing the full progressive shuffling curves, we report the end-point sensitivity under complete coordinate permutation on representative WSI benchmarks. Patch embeddings are fixed, and only coordinates are permuted; a small AUC change therefore means that the model prediction is largely insensitive to the spatial graph.

Table 15: Additional Shuffle Sensitivity on WSI Benchmarks. Patch embeddings are fixed; only coordinates are permuted. Endpoint values are shown for representative runs unless otherwise stated; tables in the main text report 5-seed mean±std.
<table><tr><td>Method</td><td>BRACS  $\mathrm { O r i g }  \mathrm { S h u f f } ( \Delta )$ </td><td>PANDA Orig → Shuff (∆)</td><td>NSCLC Orig → Shuff (∆)</td></tr><tr><td>AB-MIL</td><td>0.8806→0.8810 (+0.0004)</td><td>0.9306→0.9302 (-0.0004)</td><td>0.9569→0.9571 (+0.0002)</td></tr><tr><td>TransMIL</td><td>0.8450→0.8425 (-0.0025)</td><td>0.9288→0.9270 (-0.0018)</td><td>0.9692→0.9675 (-0.0017)</td></tr><tr><td>DS-MIL</td><td>0.8054→0.8054 (0.0000)</td><td>0.9329→0.9329 (0.0000)</td><td>0.9579→0.9579 (0.0000)</td></tr><tr><td>ResTopoMIL</td><td>0.9006→0.8350 (-0.0656)</td><td>0.9426→0.9052 (-0.0374)</td><td>0.9753→0.9610 (-0.0143)</td></tr></table>

Table 15 gives a compact view of spatial sensitivity at the 100% shuffle endpoint. AB-MIL is almost unchanged, as expected for a permutation-invariant model. DS-MIL is exactly unchanged on the listed benchmarks, and TransMIL shows only mild degradation despite modeling contextual relations. ResTopoMIL drops more clearly when coordinates are destroyed, especially on BRACS and PANDA. Together with the progressive curves below, this table separates slide-level predictive strength from actual coordinate dependence: high AUC alone does not imply that the decision rule is using tissue arrangement.

## H Progressive Coordinate-Shuffling Analysis

Figure 5 reports the full progressive coordinate-shuffling analysis. This experiment keeps patch embeddings fixed and gradually corrupts only the spatial coordinates used to construct context. The resulting monotonic degradation provides a behavioral check that complements the ablations in the main text: the residual branch depends on preserved spatial arrangement rather than only on extra capacity or a favorable optimization schedule.

![](images/1ebb6eaff760ec1b6856f2529b7f9dbc318ee3ff58572c485749ff9f71599477.jpg)

![](images/f62d0f6d8172129ece49d183d5d1b81a4a0bcfcdc5f3444b4a66519e124ba90f.jpg)

![](images/bbbc579708b5902c695ef10dfca91c20a873133647ca1dfa1e9ca4780dec8ce2.jpg)

![](images/aa5753d6be8d46ad8aa1eda5c9eee42eab359889213513e7202294f3f37016bc.jpg)  
Figure 5: Progressive Shuffle Sensitivity. AUC decreases as an increasing fraction of patch coordinates is permuted, while the patch composition remains unchanged. Curves show representative-seed trajectories for trend visualization; numeric tables report 5-seed mean±std unless explicitly marked otherwise.

Figure 5 adds a dose-response view to the endpoint table. The important property is not only that the fully shuffled model performs worse, but that the degradation increases as a larger fraction of coordinates is permuted. This monotonic trend is a stronger check than a single shuffle point because it shows that the model response tracks the amount of spatial damage. In contrast, a model that uses coordinates only incidentally would not be expected to produce a smooth decline as the coordinate field is progressively corrupted.

The shuffling test should not be read as evidence that the model is fragile to ordinary coordinate noise. ResTopoMIL operates on the coordinate-induced KNN graph, not on absolute slide coordinates. Rigid translation, rotation, reflection, and uniform scaling preserve pairwise distances and therefore leave the unweighted KNN graph unchanged. Mild coordinate jitter is also unlikely to affect the prediction when the local neighbor identities remain stable. The expected failure mode is different: performance should drop when the perturbation is large enough to rewire local neighborhoods and erase the tissue arrangement that supports the diagnosis. Figure 5 is consistent with this selective behavior. Small topology-preserving changes should be tolerated, whereas progressive coordinate permutation degrades performance because it increasingly destroys the graph structure. A dedicated invariance study over registration noise and rigid transformations remains useful future work.

## I Quantitative Localization Assessment on CAMELYON-16

To further examine whether overcoming spatial blindness translates into more useful spatial evidence, patch-level lesion localization is evaluated on CAMELYON-16 [Ehteshami Bejnordi et al., 2017]. The official tumor annotations are used only for evaluation; training remains slide-level, and tumor masks are not used to supervise any MIL model. Because patch-level localization scores are sensitive to mask conversion and thresholding, we specify the evaluation protocol before reporting the numbers.

CAMELYON-16 is used here as an additional localization benchmark and is not counted among the 9 primary WSI benchmarks used for classification and survival prediction.

Patch-level labels. Slides are tiled with the same non-overlapping 256 × 256 patches at 20× magnification used elsewhere in the paper. Each patch footprint is mapped to the coordinate system of the CAMELYON-16 annotation mask. A patch is labeled positive if at least 1% of its area overlaps an annotated tumor region, and negative otherwise. This small-overlap rule includes boundary patches while avoiding isolated mask-contact artifacts. Patches outside the tissue mask produced by Otsu thresholding are ignored for both prediction and evaluation. The official annotations are therefore converted into a binary patch grid, rather than into pixel-level supervision.

Patch-level scores. For ResTopoMIL, patch evidence is extracted from the node-level topological residual before graph pooling. Let $\mathbf { H } _ { i } ^ { ( 2 ) }$ be the final GCN embedding of patch i and let $\mathbf { W } _ { t o p o , y }$ be the residual-classifier weight for the target or positive class. We use

$$
s _ { i } = ( \mathbf { W } _ { t o p o , y } ) ^ { \top } \mathbf { H } _ { i } ^ { ( 2 ) }\tag{32}
$$

as the raw patch-level localization score, then min–max normalize scores within each slide for heatmap rendering and threshold-based evaluation. Baseline heatmaps use their native attention or instance-score outputs under the same normalization and evaluation protocol.

Dice threshold. Dice is computed on the patch grid. The binarization threshold is selected on the validation split by maximizing mean Dice for each method, and the selected threshold is then fixed for test slides. We do not tune the threshold on test slides or choose slide-specific thresholds.

FROC computation. FROC is threshold-swept rather than evaluated at a single operating point. For each threshold, connected components in the binarized patch grid are treated as lesion candidates, and the candidate score is the maximum patch score inside the component. A candidate is counted as a true positive if its maximum-score patch falls inside an annotated tumor region; otherwise it is counted as a false positive. The reported FROC is the average sensitivity at the standard

CAMELYON operating points of 1/8, 1/4, 1/2, 1, 2, 4, and 8 false positives per slide. All methods use the same patch labels, threshold-selection protocol, component rule, and FROC operating points.

With this fixed protocol, Table 16 reports Dice, specificity, and FROC. TransMIL illustrates a mismatch between false-positive control and lesion overlap: it achieves near-perfect specificity (0.999), but its localization performance is weak by Dice (0.103) despite a moderate FROC (0.4866). This is consistent with the optimization-laziness interpretation: a model can suppress false positives while still failing to assign spatial evidence to the correct tumor boundary.

Table 16: Quantitative Localization on CAMELYON-16. Patch-level Dice, specificity, and FROC are reported. Higher is better for all metrics.
<table><tr><td>Method</td><td>Dice</td><td>Specificity</td><td>FROC</td></tr><tr><td>AB-MIL</td><td>0.412</td><td>0.985</td><td>0.3952</td></tr><tr><td>CLAM-SB</td><td>0.459</td><td>0.987</td><td>0.4257</td></tr><tr><td>DS-MIL</td><td>0.259</td><td>0.863</td><td>0.4506</td></tr><tr><td>TransMIL</td><td>0.103</td><td>0.999</td><td>0.4866</td></tr><tr><td>DTFD-MIL</td><td>0.525</td><td>0.999</td><td>0.4712</td></tr><tr><td>MHIM-MIL</td><td>0.548</td><td>0.992</td><td>0.4815</td></tr><tr><td>2DMambaMIL</td><td>0.475</td><td>0.995</td><td>0.4520</td></tr><tr><td>ResTopoMIL</td><td>0.624</td><td>0.999</td><td>0.5483</td></tr></table>

ResTopoMIL achieves the best localization performance while maintaining near-perfect specificity. Its Dice score reaches 0.624 and FROC reaches 0.5483. By offloading compositional statistics to the statistical stream, the topological branch acts as a spatial regularizer and is encouraged to respect local adjacency in the tumor microenvironment. This result also clarifies why specificity alone is insufficient for evaluating structure-aware pathology models: avoiding false-positive activations can coexist with poor spatial overlap. The main conclusion from Table 16 should therefore be read as evidence that ResTopoMIL produces more spatially coherent weakly supervised evidence under a fixed protocol, not as a claim that it is a fully validated tumor segmentation system.

## J Feature-Space Visualization of Statistical and Topological Streams

The feature spaces learned by the statistical and topological streams on BRACS are visualized using both principal component analysis (PCA) and t-SNE. These two projections provide complementary views. PCA is a linear projection that preserves the dominant global variance directions, and is therefore useful for checking whether the separation between streams is visible without relying on a nonlinear embedding method. t-SNE, by contrast, emphasizes local neighborhood structure and is better suited for inspecting whether samples from the same class form compact local clusters and whether difficult samples are reorganized by the topological stream.

![](images/b1b0a4e2f07db2d89b03e4d58538efc3797ea680d5bd08519ba8d9fa83cf51ab.jpg)

![](images/3caa502d73e58e163cb5f8dc5fd3ec2d077353349d2bd93b618c1ecb8dfd1ae7.jpg)

![](images/432e023ce83fb64a4aafc149177833e702d5f9388f8d76ef756a6d090687d8b9.jpg)

![](images/af0efb7aa022285beba6a7123cffd385e3c02e9c09e9ae143e4b28698a45d226.jpg)

![](images/831cbf3184a254fc6172c264c52f38c6db44361370e9f60954c83ab964f0c4bc.jpg)

![](images/8086abe7ece37f54aed8f909f63d79a67ee1ab77735b97c5584c605782963b1c.jpg)

![](images/09a857732c08870aed161b840b9efb1e5020143222d3f27eaa18b71720516f8d.jpg)

![](images/da9f1aa76b33612a54f4a6736f3a5feaca299f75189cb02b728c17673032f401.jpg)

![](images/c750273d3fa7cef6fb882f6f69fd8b0c44e9cca01a8f0b9054e9abc4046c0f9c.jpg)  
ATBTMTstat-hard☆ stat-wrong, topo-correct

![](images/fb280eccf29d45ae1faad933ab3446a3372831d90319d1c222995dcd083232c2.jpg)  
Figure 6: PCA Visualization of Statistical and Topological Streams. The statistical stream produces heavily mixed class distributions, with negative silhouette scores in the shown projections (e.g., around −0.05 to −0.09) and large Davies–Bouldin (DB) indices. In contrast, the topological stream forms much clearer classseparated structures, with silhouette scores around 0.87–0.89 and DB indices around 0.16–0.18. Hollow circles denote statistically hard samples, and stars denote samples misclassified by the statistical stream but corrected by the topological stream.

Although PCA is only a two-dimensional linear projection, Figure 6 already separates the two streams clearly. The statistical stream contains large overlaps among BRACS classes, matching the intuition that patch-composition summaries struggle with atypical and borderline lesions. The topological stream, by contrast, forms separated class regions with much better silhouette and DB indices. The corrected samples marked by stars are especially important: they show that the topological stream is not merely producing a prettier embedding, but reorganizing cases that are difficult for the statistical stream.

![](images/d54440f701c96f01b047025af4d6c8000b668c0455f15ddc623ac8e85339c671.jpg)

![](images/d5c9962550ef72c72308b29facfff840d107e92b1c6824bd387077e330f3fd83.jpg)

![](images/d6f3a945267f1c2bf4fd22baf55836ef2599a0d87b38a87138fed10bf565a7db.jpg)

![](images/f5dac4f41b8b27126ab633c672d018c2a2e2ee9fc11fe7f9fc3e70fd996c6230.jpg)

![](images/d5e6f6c26af23a77861775b05245c5adc8892ac3093fe7ef5a303b6747f3275c.jpg)

![](images/e56ffa8fbe6fedb839e7f63c7a124e3b4d053153b45a25eb1aa4b9304abf631a.jpg)

![](images/48c54ecafef987c75815691d56dc275834a27adee1196857b62c362ee43facf4.jpg)

![](images/041cca2f7e8c749b5d6249f5938f06143bdc279bc2434cce4713ea9585c03604.jpg)  
ATBTMTstat-hard☆ stat-wrong, topo-correct

![](images/efcd212c80cdab78eea2f7f2004a8492837e9afb5bfcc9cf8bff5b1ff7b33149.jpg)

![](images/0a2c5b284004e7e42f2bd80e555307f1e8485dde0387953d25cbb2198a5e68b5.jpg)  
Figure 7: t-SNE Visualization of Statistical and Topological Streams. The nonlinear projection shows the same qualitative pattern as PCA. Statistical-stream features remain entangled, with negative silhouette scores and large DB indices, whereas topological-stream features form compact class-specific clusters with silhouette scores around 0.92–0.93 and DB indices close to 0.10. Many star-marked cases move from mixed regions in the statistical stream to the corresponding class clusters in the topological stream.

Figure 7 confirms the same separation from a local-neighborhood perspective. The t-SNE projection is useful here because it emphasizes whether samples with similar learned representations form compact neighborhoods. The statistical stream again leaves many classes entangled, while the topo logical stream places corrected samples near their true-class clusters. Since the PCA figure shows a compatible trend, the conclusion does not rely only on nonlinear t-SNE visualization.

Together, the two visualizations indicate that the statistical stream and the topological stream encode different information. In the statistical stream, AT, BT, and MT samples are substantially intermixed, and the negative silhouette scores together with high DB values indicate poor class separation. This is expected because the statistical stream mainly captures compositional and texture-distribution information, which can be insufficient for fine-grained BRACS categories whose distinction depends on subtle tissue organization.

The topological stream shows a markedly different structure. Across both PCA and t-SNE, the topological features yield compact and well-separated clusters, and the quantitative clustering indicators improve sharply. Importantly, the star-marked samples, which correspond to cases misclassified by the statistical stream but corrected by the topological stream, tend to move from ambiguous mixed regions into the neighborhoods of their true classes. The hollow markers also highlight statistically hard samples that are better organized after incorporating spatial topology.

These observations support the intended decoupling behavior of ResTopoMIL. PCA shows that the separation is visible even from a global linear perspective, while t-SNE confirms that the local neighborhood structure is also improved. Therefore, the apparent class separation is not merely an artifact of a nonlinear visualization method; rather, both projections suggest that the topological stream learns complementary spatial-organization cues that help resolve samples that are ambiguous under purely statistical representations.

## K Qualitative Analysis

Attention heatmaps are used as qualitative, pathology-reviewed evidence rather than pixel-level validation, since the model is trained with slide-level labels. The visual findings in this section were reviewed and confirmed by pathologists as qualitatively consistent with tumor-relevant regions and diagnostically meaningful tissue organization.

Heatmap scores and post-processing. For each method, the attention or localization score assigned to a retained patch is linearly normalized to [0, 1] within each slide before visualization. We do not apply CRF refinement, Gaussian smoothing, morphological closing, or tumor-shape priors. The only post-processing step is the removal of non-tissue patches using the same tissue mask as preprocessing. This choice is intentionally conservative: the displayed heatmaps should reflect the MIL scoring function rather than a separate segmentation pipeline.

Figure 8 collects the representative comparison discussed in the main text and three additional TCGA examples. The representative case (Fig. 8a) shows more contiguous tumor attention with less diffuse background activation. TCGA-A2-A3XY (Fig. 8b) constrains high attention to the tumor core, avoiding stromal leakage. TCGA-B6-A0IA (Fig. 8c) gives sharper boundary delineation than TransMIL. TCGA-LL-A442 (Fig. 8d) localizes invasive-front regions more consistently, suggesting that the topological branch captures architectural arrangement beyond local texture.

![](images/76f5a1a5cefa05e3fd991cd346bda5af0a67ba3f8ca1071df25dc1c16eddfd94.jpg)  
(c) TCGA-B6-A0IA  
(d) TCGA-LL-A442  
Figure 8: Attention Heatmap Visualization. Representative and additional pathology-reviewed heatmaps. Warmer colors indicate higher attention weights. Compared with TransMIL, ResTopoMIL shows less background leakage and more contiguous attention over tumor-relevant regions.

## L Additional Limitations and Negative-Result Scope

The experiments deliberately emphasize structure-dependent WSI tasks, because those are the settings where spatial blindness is most clinically and methodologically relevant. This focus leaves a weaker view of the opposite regime: real pathology tasks that are almost purely compositional. In such tasks, a strong statistical stream should be close to optimal, and a residual topological branch could plausibly add noise or overfit incidental tissue layout.

Spatial-MNIST-Bag Dataset A provides a controlled synthetic check of this regime, showing that ResTopoMIL can still solve a pure-composition MIL problem. However, it is not a realistic WSI benchmark. We currently lack a public real WSI dataset whose label is known to be determined primarily by composition while being insensitive to tissue arrangement. Because of this missing negative-control dataset, our exploration of composition-dominant failure modes remains limited. A useful future benchmark would contain real WSI labels for which pathologists agree that topology is largely irrelevant; such a dataset would test whether ResTopoMIL correctly defaults to statistical performance without introducing unnecessary residual variance.

The two-stage training protocol is another limitation. It makes the optimization target cleaner for the graph branch, but it also introduces a schedule choice: the statistical anchor must be trained long enough to capture composition, yet not treated as a perfect model. If the anchor is weak, Stage 2 may be asked to correct errors that are not truly spatial. If the anchor overfits, the residual available to the graph branch may be noisy or too small. The present experiments use a fixed 10-epoch warmup and then 30/20 epochs of refinement, and the ablations show that this choice is stable on the evaluated benchmarks. Still, the method is less plug-and-play than a single end-to-end MIL baseline, and future work should study adaptive stopping rules or validation criteria for deciding when to freeze the statistical stream.