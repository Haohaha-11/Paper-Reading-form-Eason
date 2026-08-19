# Are Compact Rationales Free? Measuring Tile Selection Headroom in Frozen WSI-MIL

Hyun Do Jung<sup>1</sup>, Jungwon Choi<sup>2</sup>, Soojung Choi<sup>3</sup>, Yujin Oh<sup>†,4</sup>, and Hwiyoung Kim<sup>†,5</sup>

<sup>1</sup>Department of Artificial Intelligence, Yonsei University, Seoul, South Korea <sup>2</sup>Kim Jaechul Graduate School of AI, KAIST, Daejeon, South Korea

<sup>3</sup>Department of Integrative Medicine, College of Medicine, Yonsei University, Seoul, South Korea <sup>4</sup>Department of Biomedical Systems Informatics, College of Medicine, Yonsei University, Seoul, South Korea <sup>5</sup>H-Data Strategy Center, Hallym University Chuncheon Sacred Heart Hospital, Chuncheon, South Korea <sup>†</sup>Co-corresponding authors: yujinoh@yuhs.ac, hykim@hallym.or.kr

## Abstract

Whole-slide image (WSI) multiple instance learning (MIL) classifiers can achieve strong slide-level AUC while leaving the full-bag prediction opaque. Attention scores are widely reused as post-hoc explanations, but high attention can reflect aggregation preference rather than a compact, model-sufficient rationale. We study post-hoc rationale highlighting for frozen WSI-MIL: given a trained classifier, can its slide-level prediction be recovered from a compact, output-consistent tile subset without retraining the backbone? We instantiate this question with Finding Optimal Contextual Instances (FOCI), a lightweight rationale-readout layer over a frozen MIL backbone. FOCI is trained with model-output sufficiency and exclusion objectives over keep/drop tile subsets, evaluated with an insertion-style Sequential Reveal Protocol (SRP) adapted to WSI-MIL, and summarized by the Selection Headroom Index (SHI). Across three WSI benchmarks and seven MIL backbones, FOCI reveals that compact rationales are selection-headroom dependent rather than universally available: transformer and multi-branch attention aggregators can admit compact rationales, near-minimal attention-pooling baselines enter a selection-saturation regime, and hard-selection backbones can conflict with an external readout. For TransMIL, relative to its documented CLS-proxy ranking, FOCI reduces the Minimum Sufficient K (MSK) tile count by 32–56% across the three benchmarks, while ACMIL+FOCI attains the highest mean SHI (+0.465). Deletion-based perturbation and selected-only downstream evaluation provide complementary checks. These results position FOCI as a model-level interpretability and audit layer: selected tiles are not claims of clinical or pathologist-level diagnostic sufficiency, but candidate rationales that offer a compact, reviewable view of when a frozen MIL prediction can be localized to a small output-consistent subset.

## 1 Introduction

Whole-slide image (WSI) classification plays a central role in computational pathology, supporting cancer subtyping, grading, and prognosis. The dominant approach extracts patch features with a frozen foundation encoder, aggregates them through a MIL backbone, and trains on slide-level labels alone [1, 2, 3, 4]. This pipeline reaches competitive diagnostic accuracy across several benchmarks [5, 6], but the full-bag prediction remains opaque: it gives a single slide label without surfacing the tiles that support it.

Attention scores are commonly repurposed as post-hoc explanations [7, 8, 9], but high attention does not by itself answer whether a compact subset can recover the model output; in some settings, attention can reflect aggregation or training dynamics rather than an output-consistent rationale [10, 11]. Such rationale highlighting may support downstream review by surfacing candidate regions for inspection, but we do not evaluate reader performance or claim clinical sufficiency.

![](images/da9488cf363e4bb4c7ce30223f81b02124a85379d7bfd9daa684adb7a5bafdaf.jpg)  
Figure 1: Selection headroom for post-hoc rationale highlighting in frozen WSI-MIL: a frozen MIL classifier produces an opaque slide-level prediction, FOCI selects a compact output-consistent tile subset that recovers it, and selection headroom across backbones determines when such compact rationales exist. On TransMIL, relative to its documented CLS-proxy ranking, FOCI reduces MSK by 32–56% across the three benchmarks while leaving the full-bag classifier unchanged.

To address this gap, we study post-hoc rationale highlighting for frozen WSI-MIL classifiers: given a trained classifier, can its slide-level prediction be recovered from a compact, output-consistent tile subset without retraining the backbone? Figure 1 summarizes this audit setting. We instantiate this question with Finding Optimal Contextual Instances (FOCI), a lightweight rationale-readout layer attached to any backbone exposing per-tile features without modifying the existing inference pipeline.

We adapt perturbation-curve evaluation to WSI-MIL through an insertion-style Sequential Reveal Protocol (SRP): tiles are revealed in rank order and the frozen classifier’s confidence is tracked as a function of K. We summarize this curve with AUKC, Minimum Sufficient K (MSK; the smallest K that reaches κ), and Reach (fraction of slides reaching κ). SRP applies to any per-tile ranking, making it a backbone-agnostic operating-point analysis. We further introduce the Selection Headroom Index (SHI) to quantify per-backbone compression of FOCI relative to the frozen backbone’s own ranking, and we triangulate compactness with deletion-based perturbation and selected-only downstream evaluation (§4).

Across three datasets—TCGA-NSCLC, TCGA-BRCA [12], and PANDA [13]—and seven MIL backbones, FOCI reveals that compact post-hoc rationales are selection-headroom dependent rather than universally available. Soft-aggregation backbones with rationale-compression headroom can be high lighted with a small tile subset, near-minimal attention-pooling baselines enter a selection-saturation regime, and hard-selection backbones can conflict with an external readout. This architecturedependent pattern is not captured by slide-level AUC alone.

In summary, we present three contributions:

• We formulate post-hoc rationale highlighting as a model-level audit layer for frozen WSI-MIL classifiers: the full-bag classifier remains unchanged, and sufficiency is used strictly in the model-output sense rather than as clinical or pathologist-level diagnostic sufficiency.

• We introduce FOCI, a lightweight rationale-readout module trained with keep/drop modeloutput sufficiency and exclusion objectives, and evaluate ranked subsets with SRP, MSK, AUKC, Reach, and SHI.

• We show that compact rationale highlighting is architecture-dependent: soft-aggregation backbones can admit compact rationales, selection-saturation regimes leave little room to improve, and hard-selection backbones can conflict with an external selector. Deletion-based perturbation and selected-only downstream evaluation provide complementary checks, with per-backbone SHI values reported in §4.3.

Although we do not evaluate reader-level performance, the resulting candidate rationales provide a compact, reviewable view of when frozen MIL predictions can be localized to small output-consistent subsets.

## 2 Related Work

## 2.1 Multiple instance learning for whole-slide images

The standard MIL recipe treats each slide as a bag of patch features with a single slide-level label and no per-patch annotation [14, 1]. Attention-based pooling [2] became the default aggregator, with CLAM [3] adding class-specific attention branches and instance-level clustering. Later WSI-MIL backbones replace or augment this pooling mechanism with transformer self-attention [15], hierarchical representations [16], hard instance mining [17], attribution-based selection [18], or multibranch masked attention [19]. In parallel, frozen pathology foundation encoders such as UNI [4], CONCH [20], and Prov-GigaPath [21] now provide patch features for slide-level MIL. FOCI fits this frozen-feature pipeline: the encoder and MIL backbone remain fixed, and only a lightweight rationale-readout module is trained to score which tiles to keep.

## 2.2 Interpretability and faithfulness in MIL

Attention weights are commonly reused as explanations in MIL [2, 3, 22], but attention scores do not directly answer whether a compact tile subset can recover the model output [7, 8]. Other explanation approaches include instance-level classifiers in CLAM [3], concept-based models [23], and gradient-based localization such as GradCAM [24]. These methods surface regions or concepts, but they typically do not report the operating-point question central to our study: how many tiles are sufficient for the frozen model to recover its prediction?

Interpretable-by-design MIL methods address related goals from a different angle. Additive MIL [25] decomposes slide predictions into region-wise additive contributions, and SI-MIL [26] introduces a self-interpretable MIL framework with feature-level explanations. Rather than designing a new intrinsically interpretable MIL architecture, we audit frozen WSI-MIL classifiers that expose per-tile features, and we report where post-hoc rationale highlighting has selection headroom rather than claiming superiority over intrinsically interpretable models. Perturbation-based evaluation [27] and MIL-specific patch-dropping metrics such as xMIL/AUPC [28] measure how predictions change when ranked regions are removed. Our SRP is complementary: it evaluates the insertion direction, measuring how quickly confidence is recovered as ranked tiles are progressively revealed.

## 2.3 Token selection and frozen rationale readouts

Selecting inputs that support a prediction has been studied extensively in NLP rationalization [29]. Differentiable selectors identify token subsets that recover the full-input prediction [30, 31, 32]. Related work on selective prediction [33], early-exit networks [34], and cooperative rationalization [35] studies adjacent questions of confidence, computation, and complement control under different training assumptions. In vision and MIL, straight-through estimators [36] enable hard sparse selection, and ASMIL [37] jointly trains a selector-like mechanism with the MIL backbone.

FOCI differs from joint selector-training approaches in its frozen setting. The MIL classifier is already trained and remains fixed; the selector is a post-hoc readout head over a stable feature space, trained with keep/drop sufficiency and exclusion losses and evaluated through insertion-style SRP. This also separates FOCI from ReaMIL [38], a concurrent evidence-aware MIL training method in whole-slide histopathology. In ReaMIL, the selector and backbone share gradient flow after warmup to train a compact-rationale classifier. FOCI does not train a new evidence-aware classifier. It asks whether the decisions of already-trained MIL backbones are post-hoc readable from compact tile subsets, and uses this readout to measure selection headroom and architecture-dependent failure modes.

![](images/386659d7c1d4faef56b081a81266af84f885acbc03cc49e538b71c0244415aea.jpg)  
Figure 2: FOCI as a frozen rationale-readout probe. The frozen encoder maps WSI tiles to features $x _ { 1 } , \ldots , x _ { N }$ , and the frozen MIL backbone produces the primary full-bag prediction. Only the lightweight FOCI selector trains; keep/drop subsets are re-forwarded through the same frozen backbone for training and evaluation.

## 3 Method

## 3.1 Frozen WSI-MIL setup

Following standard weakly supervised MIL, each slide s is a bag of patch features $X _ { s } = \{ x _ { s , i } \} _ { i = 1 } ^ { N _ { s } }$ extracted by a frozen encoder, together with spatial coordinates $C _ { s } ~ = ~ \{ c _ { s , i } \} _ { i = 1 } ^ { N _ { s } }$ where $c _ { s , i } =$ $( u _ { s , i } , v _ { s , i } )$ is the pixel location of patch i. We use UNI2-h [4] to extract d=1536-dimensional features. The slide has a single label $y _ { s } \in \{ 1 , \ldots , C \}$ with no patch-level supervision.

Patch features are projected into a shared token space and processed by the MIL backbone (e.g., TransMIL with a learned [CLS] token through L transformer layers); the final representation maps to class logits $\boldsymbol { \ell _ { s } } \in \mathbb { R } ^ { C }$ . Full implementation and backbone details are provided in Appendix L.

Frozen backbone. The backbone remains fully frozen during FOCI training; rationale losses update only the lightweight selection head (∼130K parameters, under 1% of the primary TransMIL pipeline). Joint training conflicts with the classification objective and collapses validation AUC by more than 15 points within two epochs (see Appendix G.4).

## 3.2 Output-consistent rationale selection

Given the frozen slide classifier $f$ above, mapping a bag of tile features $X = \{ x _ { i } \} _ { i = 1 } ^ { N }$ to a class probability p for target class y, we seek a binary mask $z \in \{ 0 , 1 \} ^ { N }$ with $\| z \| _ { 0 } = K$ satisfying two output-consistency conditions:

$$
p _ { y } \big ( f ( z \odot X ) \big ) \ge \tau \quad \mathrm { ( s u f f i c i e n c y ) } , \qquad p _ { y } \big ( f ( ( 1 - z ) \odot X ) \big ) \le \beta \quad \mathrm { ( e x c l u s i o n ) , }\tag{1}
$$

where $\tau , \beta$ are confidence thresholds. We call any such K-tile subset a model-sufficient rationale.

Pipeline preservation. The frozen backbone f continues to produce the primary slide-level prediction unchanged. FOCI does not replace the full-bag forward pass, retrain the backbone, or require pathologist annotation; it learns a per-tile scoring head that partitions each slide into a keep set (candidate rationale) and a drop set (complement). During training, the keep, drop, and full-bag views pass through the same frozen backbone with separate loss terms; at test time, tiles are ranked by the selector score and evaluated under SRP. Figure 2 shows the architecture.

## 3.3 Rationale selection module

Given the token representations from Section 3.1, a small MLP computes a scalar selection logit $a _ { s , i } = \mathbf { M L P _ { s e l } } ( t _ { s , i } ) \in$ R for each token $t _ { s , i }$ . We consider two variants for turning these logits into selection decisions.

Soft gate (FOCI-Soft). The first variant applies the Concrete (Gumbel–sigmoid) relaxation [39, 40]. Sampling $\epsilon _ { s , i } \sim$ Uniform(0, 1):

$$
z _ { s , i } = \sigma \bigg ( \frac { a _ { s , i } + \log \epsilon _ { s , i } - \log ( 1 - \epsilon _ { s , i } ) } { T } \bigg ) ,\tag{2}
$$

where $T > 0$ is temperature. The scores $z _ { s , i } \in ( 0 , 1 )$ approach binary values as $T  0$

Hard top-K with straight-through (FOCI-STE). FOCI-STE replaces the soft Concrete gate with an exactly K-sparse binary mask in the forward pass while routing the backward gradient through a sigmoid surrogate [36], eliminating the soft-vs-hard cardinality mismatch between training and SRP evaluation. Although hard top-K fixes the forward-pass cardinality, we retain a small per-bag budget/scale regularizer $\mathsf { \bar { ( } } \lambda _ { \mathrm { b u d g e t } } = 5 \times 1 0 ^ { - 3 } )$ to stabilize selector scores near the rank-K boundary. FOCI-STE is one of two parameterizations of the same audit framework (the other being FOCI-Soft); the central object of study is whether the frozen classifier exhibits selection headroom under a consistent ranking, not the choice of gate parameterization. Full STE derivation, surrogate-gradient mechanics, and forward/backward analysis are in Appendix J.

Three-view inference. Both variants produce the same three views of the slide, namely the original bag $X _ { \mathrm { f u l l } } = X _ { s }$ , the keep bag $X _ { \mathrm { k e e p } } = z _ { s } \odot X _ { s }$ <sub>s</sub> (or $m _ { s } \odot X _ { s }$ <sub>s</sub> for FOCI-STE), and the drop bag $X _ { \mathrm { d r o p } } = ( 1 - z _ { s } ) \odot X _ { s }$ . In FOCI-Soft, all tokens stay in the sequence but non-selected patches are down-weighted by $z _ { s } ,$ , whereas in FOCI-STE the mask is binary. Each view passes through the frozen backbone to produce logits $\ell _ { \mathrm { f u l l } } , \ell _ { \mathrm { k e e p } } ,$ , and $\ell _ { \mathrm { d r o p } }$

## 3.4 Rationale-aware training objectives

In addition to slide-level accuracy, we design a training objective that explicitly shapes how the selector partitions tiles into a rationale subset within each bag. We compute the full-bag cross-entropy only as a preservation monitor since the full-bag forward pass bypasses the selector and the backbone is frozen. The selector itself is optimized only through losses on the keep/drop views, each enforcing a distinct property of the selection: (i) sufficiency, where the selected tiles alone support a highconfidence prediction; (ii) exclusion, where the remaining tiles do not support the true class; (iii) spatial compactness, where selected tiles form a coherent region on the slide rather than scattering across it; and (iv) a small budget/scale regularizer that controls selection mass in FOCI-Soft and stabilizes selector scores in FOCI-STE.

Let $p _ { y } ( \ell ) = \mathrm { s o f t m a x } ( \ell ) [ y _ { s } ]$ denote the true-class probability. We first define one full-bag preservation monitor and four keep/drop selector losses:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { f u l l } } = \mathrm { C E } ( \ell _ { \mathrm { f u l l } } , y _ { s } ) , } \end{array}\tag{3}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s u f f } } = \mathrm { C E } ( \ell _ { \mathrm { k e e p } } , y _ { s } ) , } \end{array}\tag{4}
$$

$$
\mathcal { L } _ { \mathrm { h i n g e } } = \operatorname* { m a x } \bigl ( \tau - p _ { y } ( \ell _ { \mathrm { k e e p } } ) , 0 \bigr ) ,\tag{5}
$$

$$
\mathcal { L } _ { \mathrm { e x c l } } = \operatorname* { m a x } \big ( p _ { y } ( \ell _ { \mathrm { d r o p } } ) - \beta , 0 \big ) ,\tag{6}
$$

$$
\mathcal { L } _ { \mathrm { c o n t i g } } = \frac { \sum _ { i } z _ { s , i } \| c _ { s , i } - \mu _ { s } \| _ { 2 } ^ { 2 } } { \sum _ { i } z _ { s , i } } ,\tag{7}
$$

where $\begin{array} { r } { \mu _ { s } = \sum _ { i } z _ { s , i } c _ { s , i } / \sum _ { i } z _ { s , i } } \end{array}$ is the selection-weighted centroid, $\tau \in ( 0 , 1 )$ ) is the target confidence for the keep bag, and $\beta \in ( \bar { 0 } , 1 )$ is the tolerance for the drop bag. We separate the keep-bag CE term $( \mathcal { L } _ { \mathrm { s u f f } } )$ from the confidence hinge $( \mathcal { L } _ { \mathrm { h i n g e } } )$ because they receive different weights in the total loss.

Because the full-bag forward pass bypasses the selector and the backbone is frozen, ${ \mathcal { L } } _ { \mathrm { f u l l } }$ contributes no gradient to the selector parameters; we monitor it as a preservation check rather than as a selector training term. The selector objective excludes ${ \mathcal { L } } _ { \mathrm { f u l l } }$ and uses the keep/drop terms plus the budget

regularizer:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s e l e c t o r } } = \lambda _ { \mathrm { s u f f } } \mathcal { L } _ { \mathrm { s u f f } } + \lambda _ { \mathrm { h i n g e } } \mathcal { L } _ { \mathrm { h i n g e } } + \lambda _ { \mathrm { e x c l } } \mathcal { L } _ { \mathrm { e x c l } } + \lambda _ { \mathrm { c o n i g } } \mathcal { L } _ { \mathrm { c o n i g } } + \lambda _ { \mathrm { b u d g e t } } \mathcal { L } _ { \mathrm { b u d g e t } } . } \end{array}\tag{8}
$$

where $\mathcal { L } _ { \mathrm { b u d g e t } }$ is a small per-bag budget/scale regularizer $( \lambda _ { \mathrm { b u d g e t } } = 5 \times 1 0 ^ { - 3 } )$ . Full details for $\mathcal { L } _ { \mathrm { b u d g e t } }$ , the FOCI-Soft entropy term, and the “sufficiency objective” shorthand are provided in Appendix K.

Contiguity caveat. $\mathcal { L } _ { \mathrm { { c o n t i g } } }$ is a small-weighted optimization stabilizer against scattered masks, not a clinical or morphological prior. Its ablation reduces training stability (Appendix G.4), so we interpret it as part of the selector parameterization rather than evidence that diagnostic tissue must be spatially contiguous.

## 3.5 Sequential Reveal Protocol and rationale metrics

Standard metrics such as AUC, accuracy, and F1 evaluate whether a model predicts the correct slide label, but they do not capture how much of the slide the model needed to see. Two models with identical AUC may differ in rationale compactness if one requires hundreds of tiles while the other needs only a handful.

To quantify this gap, the Sequential Reveal Protocol (SRP) ranks tiles by a per-tile score $( a _ { s , i }$ for FOCI, or the method-specific native/proxy ranking score in Appendix C) and reveals them in descending order; after each tile is added we record the true-class probability $p _ { y } ( K )$ to trace a confidence–count K-curve. SRP is an insertion-style adaptation of perturbation-curve evaluation [41] for WSI-MIL, with three complementary operating-point summaries (MSK, Reach, AUKC) reported per slide. Cross-method SRP reveal curves are shown in Appendix G.1.

SRP metrics. For each slide s, we summarize the K-curve $p _ { y _ { s } } ^ { ( s ) } ( K )$ with three metrics at operating confidence κ. MSK and Reach use the threshold κ:

$$
\mathrm { M S K } _ { s } ( \kappa ) = \operatorname* { m i n } \{ K \leq K _ { \operatorname* { m a x } } : \arg \operatorname* { m a x } _ { c } p _ { c } ^ { ( s ) } ( K ) = y _ { s } \land p _ { y _ { s } } ^ { ( s ) } ( K ) \geq \kappa \} ,\tag{9}
$$

$$
\mathrm { R e a c h } ( \kappa ) = | S _ { \mathrm { t e s t } } | ^ { - 1 } \sum _ { s \in S _ { \mathrm { t e s t } } } \mathbf { 1 } \big [ \mathrm { M S K } _ { s } ( \kappa ) \mathrm { e x i s t s } \big ] .\tag{10}
$$

AUKC is unthresholded. Let $K _ { s , j }$ denote the j-th reveal count evaluated for slide $s ,$ let $m _ { s }$ be the number of reveal steps, and let $\rho _ { s , j } = K _ { s , j } / N _ { s } ^ { \mathrm { r e a l } }$ be the normalized reveal fraction, where N<sup>real</sup> is the unpadded number of candidate tiles. We compute AUKC as the trapezoidal area under the confidence curve, normalized by the maximum reveal fraction:

$$
\mathrm { A U K C } = | S _ { \mathrm { t e s t } } | ^ { - 1 } \sum _ { s \in S _ { \mathrm { t e s t } } } \frac { 1 } { \rho _ { s , m _ { s } } } \sum _ { j = 1 } ^ { m _ { s } - 1 } \frac { p _ { y _ { s } } ^ { ( s ) } ( K _ { s , j } ) + p _ { y _ { s } } ^ { ( s ) } ( K _ { s , j + 1 } ) } { 2 } \big ( \rho _ { s , j + 1 } - \rho _ { s , j } \big ) .\tag{11}
$$

$\mathrm { M S K } _ { \mathrm { c o n d } }$ denotes the conditional mean over slides reaching κ. Thus, AUKC summarizes the full reveal curve, while MSK and Reach depend on the operating threshold. The three metrics expose different failure modes: AUKC can be high while Reach is low (failure on hard slides) or MSK is high (inefficient compression). We report all three at $\kappa = 0 . 9$ unless stated otherwise.

Selection Headroom Index (SHI). A frozen MIL classifier already produces a tile ranking through its internal attention or aggregation weights, which itself induces a K-curve. To quantify how much further FOCI compresses the sufficient set relative to that internal ranking, we define the Selection Headroom Index:

$$
\mathrm { S H I } ( f ) = \frac { \mathrm { M S K } _ { \mathrm { b a s e } } ( f ) - \mathrm { M S K } _ { \mathrm { F O C I } } ( f ) } { \mathrm { M S K } _ { \mathrm { b a s e } } ( f ) + \epsilon } ,\tag{12}
$$

where $\operatorname { M S K } _ { \mathrm { b a s e } } ( f )$ is computed by ranking tiles with the frozen backbone’s own attention or aggregation weights, $\operatorname { M S K } _ { \operatorname { F O C I } } ( f )$ uses a FOCI-trained selector attached to the same frozen $f ,$ and ϵ is a small stabilizing constant. We read the sign directly: positive SHI indicates that FOCI compresses the rationale beyond what the backbone’s internal ranking already provides, near-zero SHI indicates a selection-saturation regime where the backbone ranking is already near-minimal, and negative SHI indicates that the external selector conflicts with the backbone’s internal ranking enough to inflate the sufficient set. Unlike AUC, which measures slide-level discrimination, SHI measures whether the frozen decision can be compressed into a smaller sufficient subset, and is therefore a property of the trained model and feature encoder rather than of classification performance. We report SHI alongside MSK and AUKC in §4. For backbones without an explicit native tile score, SHI is computed relative to a documented proxy ranking (Appendix C) and measures improvement over the available backbone ranking, not a ranking-independent property. Our main SRP uses the ground-truth class to jointly assess confidence recovery and correctness on labeled test sets; an audit-time variant tracks the predicted class $\hat { y } = \arg \operatorname* { m a x } _ { c } f _ { c } ( X )$ instead of y, reported in Appendix N.

## 4 Experiments

We evaluate FOCI on three public benchmarks along three axes: compact rationale recovery while preserving the frozen full-bag classifier, consistency across backbone architectures and failure modes, and the contribution of each loss component (with FOCI-Soft vs FOCI-STE in Appendix G.4). Primary compactness results use the SRP metrics defined in Section 3.5; deletion-based perturbation and selected-only AUC provide complementary checks.

Hypotheses tested. We test four linked hypotheses: (H1) slide-level AUC decouples from selection headroom; (H2) non-minimal baseline MSK leaves room for FOCI compression; (H3) deletion and selected-only AUC provide complementary checks rather than interchangeable explanation metrics; and (H4) FOCI-STE mainly improves hard-cardinality alignment rather than serving as the central contribution. The corresponding evidence is reported in §4.3, Table 1, §4.5, and Appendix G.4.

## 4.1 Setup

Datasets. TCGA-NSCLC (LUAD vs. LUSC, 1,043 slides) and TCGA-BRCA (IDC vs. other subtypes, 1,126 slides) from TCGA/GDC [12], and PANDA (benign vs. malignant, ISUP grade ≥ 1, 10,615 slides) [13]. We use 70/15/15 train/val/test splits matching the appendix counts; full split details and per-class counts are in Appendix L.

Features and backbones. Patches are extracted at 20×, embedded with frozen UNI2-h [4] (d=1536). The primary FOCI-STE backbone is a 4-layer TransMIL (d=512, 8 heads) pretrained for 20 epochs with cross-entropy. For cross-backbone experiments, FOCI is additionally applied to ABMIL, CLAM-SB, AttriMIL, ACMIL, ASMIL, and MHIM-MIL, each pretrained independently with its original objective; in all cases the backbone is frozen during FOCI training. Full architecture, optimizer, and hyperparameter settings are in Appendix L.

Baselines and SRP scores. For each backbone, SRP uses its native attention or aggregation logits as the ranking score; FOCI ranks tiles by its own selector head. TransMIL has no native attention head, so we use the post-encoder CLS-dot-product score as a documented proxy ranking (Appendix C); SHI for TransMIL therefore measures improvement over this proxy.

## 4.2 Main results

Per-dataset SRP results for all seven backbones with and without FOCI are reported in Appendix H (Tables 12–14). Across these tables, FOCI reduces MSK when the frozen backbone has rationalecompression headroom and inflates MSK when the native ranking is already near-minimal or conflicts with an external selector. A paired Wilcoxon test on the nine TransMIL (dataset, seed) observations confirms a significant MSK reduction (p=0.008, median ∆MSK=−4.14) but no significant AUKC change (p=0.13); per-dataset tests are underpowered (n=3), so we use the appendix tables as direction-of-effect summaries.

## 4.3 Selection headroom analysis

To quantify the per-backbone effect of attaching FOCI to a frozen MIL classifier, we compute the Selection Headroom Index (SHI, defined in §3.5) for every (backbone, dataset) pair. Table 1 summarizes per-dataset and mean SHI for each backbone family. The raw baseline MSK, FOCI MSK, ∆MSK, Reach, and AUKC values are reported in Appendix H (Tables 12–14). All values are 3-seed means at $\kappa = 0 . 9$ . SHI should be read as a signed normalized effect size rather than an absolute ranking of rationale quality: when the baseline MSK is already near one tile, small absolute MSK changes can produce large negative ratios. We therefore interpret SHI together with the raw MSK and ∆MSK values in Appendix H, using it to identify headroom, saturation, and conflict regimes rather than to rank backbones in isolation.

Table 1: Selection Headroom Index (SHI) per backbone and its per-dataset breakdown. SHI is normalized by the baseline Minimum Sufficient K (MSK) tile count, so extreme values may occur when the baseline MSK is small; see Appendix H for raw MSK.
<table><tr><td colspan="2">Backbone</td><td colspan="4">Dataset</td></tr><tr><td>Family</td><td></td><td>NSCLC</td><td>BRCA</td><td>PANDA</td><td>Mean</td></tr><tr><td colspan="6">Soft-aggregation (positive headroom)</td></tr><tr><td>Transformer</td><td>TransMIL</td><td>+0.562</td><td>+0.317</td><td>+0.357</td><td>+0.412</td></tr><tr><td>Multi-branch attention</td><td>ACMIL</td><td>+0.705</td><td>+0.337</td><td>+0.354</td><td>+0.465</td></tr><tr><td colspan="6">Attention pooling (dataset-dependent saturation)</td></tr><tr><td>Attention pool</td><td>ABMIL</td><td>+0.425</td><td>-1.914</td><td>-0.319</td><td>-0.603</td></tr><tr><td>Attention pool</td><td>CLAM-SB</td><td>+0.248</td><td>-2.542</td><td>+0.016</td><td>-0.760</td></tr><tr><td>Attribution-based</td><td>AttriMIL</td><td>+0.327</td><td>-0.931</td><td>-0.336</td><td>-0.313</td></tr><tr><td colspan="6">Hard selection (architectural conflict)</td></tr><tr><td>Hard selection</td><td>ASMIL</td><td>-2.055</td><td>+0.191</td><td>-1.441</td><td>-1.102</td></tr><tr><td>Hard instance mining</td><td>MHIM-MIL</td><td>-0.309</td><td>-0.229</td><td>-0.179</td><td>-0.239</td></tr></table>

Note. SHI > 0: FOCI compresses beyond the baseline ranking, SHI ≈ 0: little room to compress, SHI < 0: FOCI conflicts with the backbone’s native selection; bold: best mean SHI within family.

![](images/9d2924cd773f296e16adc3e7bdd43a5d32deef65d3e1e9804f2fba38ebc55476.jpg)  
Figure 3: Slide-level AUC and SHI are decoupled. Each point is a (backbone, dataset) pair; color denotes backbone and marker shape denotes dataset. High-AUC backbones can have near-zero or negative SHI when their native ranking saturates or conflicts with an external readout, while TransMIL and ACMIL maintain positive SHI without necessarily being the best full-bag classifiers.

Three patterns emerge: (1) TransMIL and ACMIL show consistently positive SHI across all three datasets $( + 0 . 3 2 \mathrm { t o } + 0 . 7 1 ) ;$ (2) attention-pooling backbones (ABMIL, CLAM-SB, AttriMIL) show a dataset-dependent saturation regime, improving NSCLC baselines but inflating the sufficient set on BRCA, on which baseline MSK is already near-minimal (≈ 1.1 for ABMIL/CLAM-SB); and (3) hard-selection backbones (ASMIL, MHIM-MIL) mostly inflate under FOCI, which reflects architectural conflict between native instance selection and an external selector.

Figure 3 visualizes this decoupling: selection headroom is not predicted by slide-level AUC alone. Predicted-class SRP (Appendix N) follows the same qualitative MSK-compression pattern on TransMIL±FOCI, which supports the interpretation that the readout recovers the frozen model’s own decision rather than exploiting label-specific evaluation.

## 4.4 Selected-only downstream triangulation

A complementary check asks whether the compact rationale preserves the downstream TransMIL prediction. Table 2 reports full-bag AUC and selected-only test AUC for random K=32, FOCI fixed-K=32, and FOCI adaptive-K within the same TransMIL pipeline; full per-seed, per-K, and ABMIL-pipeline reference results are in Appendix G.3.

Table 2: Selected-only downstream AUC within the TransMIL predictor pipeline (3-seed mean). Random and fixed-FOCI rows use K=32; FOCI adaptive uses $\begin{array} { r } { \dot { K _ { s } } = \operatorname* { m a x } ( \dot { 1 } \dot { 6 } , \lfloor 0 . 0 3 N _ { s } \rfloor ) } \end{array}$ . ABMIL native top-K comparison and full per-seed / per-K results are in Appendix G.3.
<table><tr><td>Dataset</td><td>Full bag</td><td>Random  $K { = } 3 2$ </td><td>FOCI K=32</td><td>FOCI adaptive  $K _ { s }$ </td></tr><tr><td>NSCLC</td><td>0.974</td><td>0.969</td><td>0.954</td><td>0.963</td></tr><tr><td>BRCA</td><td>0.907</td><td>0.881</td><td>0.885</td><td>0.907</td></tr><tr><td>PANDA</td><td>0.989</td><td>0.931</td><td>0.945</td><td>0.934</td></tr></table>

This mini-table is a preservation check rather than a universal dominance claim. It tests whether compact FOCI-selected subsets preserve the frozen TransMIL decision and supports the headroom framing: BRCA shows a clear adaptive-K preservation signal (matching the full-bag AUC 0.907), whereas NSCLC and PANDA expose selection-saturation regimes where random subsets already preserve much of the prediction, which leaves little operating margin for any external selector. Thus, selected-only AUC serves as a triangulation check rather than the primary evidence of FOCI superiority: it reveals when a dataset/backbone pair has enough operating margin for a learned selector to improve over random compact subsets.

## 4.5 Deletion-based perturbation faithfulness

Deletion-based perturbation [41, 27] complements SRP by asking whether top-ranked tiles are load-bearing for the model output when removed, rather than how quickly confidence is recovered when they are inserted. Attention-pooling methods often score strongly on this metric because their ranking is part of the aggregation mechanism itself. On TransMIL, where the native ranking is only a CLS-dot-product proxy, FOCI increases NSCLC deletion-AUC from 0.0003 to 0.0274, indicating that the readout extracts a more load-bearing ranking than the frozen proxy. Cross-dataset deletion-AUC results are reported in Appendix I.

We therefore treat SRP, deletion, and selected-only AUC as complementary rather than interchangeable rationale-quality axes: SRP measures insertion sufficiency, deletion measures load-bearing removal, and selected-only AUC measures downstream prediction preservation under masked input. No single ranking dominates all three, which is why we use deletion as a faithfulness check rather than as the sole explanation metric.

## 5 Conclusion

FOCI is a post-hoc rationale-highlighting layer for frozen WSI-MIL classifiers: the full-bag prediction is preserved while FOCI selects a compact, output-consistent tile subset that recovers it, evaluated through SRP and the Selection Headroom Index. Across three benchmarks and seven backbones, compact rationales are selection-headroom dependent — TransMIL and ACMIL admit consistent compression, attention-pooling backbones saturate on near-minimal baselines, and hard-selection backbones conflict with an external readout. High slide-level AUC does not by itself imply that a frozen MIL classifier admits a compact rationale; SHI motivates a selection-headroom audit before treating selected tiles as faithful explanations.

Limitations. We measure model-output sufficiency only: selected tiles are candidate rationales for the frozen classifier, not annotation-validated clinical evidence. Our experiments use UNI2-h features and binary WSI tasks, so broader encoder evaluation, multiclass settings, external clinical cohorts, and multi-reader validation remain future work; details are in Appendix M.

## References

[1] Gabriele Campanella, Matthew G. Hanna, Luke Geneslaw, Allen Miraflor, Vitor Werneck Krauss Silva, Klaus J. Busam, Edi Brogi, Victor E. Reuter, David S. Klimstra, and Thomas J. Fuchs. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nature Medicine, 25(8):1301–1309, 2019.

[2] Maximilian Ilse, Jakub Tomczak, and Max Welling. Attention-based deep multiple instance learning. In Proceedings of the 35th International Conference on Machine Learning, pages 2127–2136. PMLR, 2018.

[3] Ming Y Lu, Drew FK Williamson, Tiffany Y Chen, Richard J Chen, Matteo Barbieri, and Faisal Mahmood. Data-efficient and weakly supervised computational pathology on whole-slide images. Nature biomedical engineering, 5(6):555–570, 2021.

[4] Richard J Chen, Tong Ding, Ming Y Lu, Drew FK Williamson, Guillaume Jaume, Andrew H Song, Bowen Chen, Andrew Zhang, Daniel Shao, Muhammad Shaban, et al. Towards a general-purpose foundation model for computational pathology. Nature medicine, 30(3):850–862, 2024.

[5] Neofytos Dimitriou, Ognjen Arandjelovic, and Peter D Caie. Deep learning for whole slide image analysis:´ an overview. Frontiers in medicine, 6:264, 2019.

[6] Michael Gadermayr and Maximilian Tschuchnig. Multiple instance learning for digital pathology: A review of the state-of-the-art, limitations & future potential. Computerized Medical Imaging and Graphics, 112:102337, 2024.

[7] Sofia Serrano and Noah A. Smith. Is attention interpretable? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 2931–2951, Florence, Italy, July 2019. Association for Computational Linguistics.

[8] Danish Pruthi, Mansi Gupta, Bhuwan Dhingra, Graham Neubig, and Zachary C. Lipton. Learning to deceive with attention-based explanations. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics, pages 4782–4793. Association for Computational Linguistics, July 2020.

[9] Martim Afonso, Praphulla MS Bhawsar, Monjoy Saha, Jonas S Almeida, and Arlindo L Oliveira. Multiple instance learning for wsi: A comparative analysis of attention-based approaches. Journal of Pathology Informatics, 15:100403, 2024.

[10] Supriyo Chakraborty, Richard Tomsett, Ramya Raghavendra, Daniel Harborne, Moustafa Alzantot, Federico Cerutti, Mani Srivastava, Alun Preece, Simon Julier, Raghuveer M Rao, et al. Interpretability of deep learning models: A survey of results. In 2017 IEEE smartworld, ubiquitous intelligence & computing, advanced & trusted computed, scalable computing & communications, cloud & big data computing, Internet of people and smart city innovation (smartworld/SCALCOM/UIC/ATC/CBDcom/IOP/SCI), pages 1–6. IEEE, 2017.

[11] Wenhui Zhu, Peijie Qiu, Xiwen Chen, Zhangsihao Yang, Aristeidis Sotiras, Abolfazl Razi, and Yalin Wang. How effective can dropout be in multiple instance learning ? In Forty-second International Conference on Machine Learning, 2025.

[12] John N Weinstein, Eric A Collisson, Gordon B Mills, Kenna R Shaw, Brad A Ozenberger, Kyle Ellrott, Ilya Shmulevich, Chris Sander, and Joshua M Stuart. The cancer genome atlas pan-cancer analysis project. Nature genetics, 45(10):1113–1120, 2013.

[13] Wouter Bulten, Kimmo Kartasalo, Po-Hsuan Cameron Chen, Peter Ström, Hans Pinckaers, Kunal Nagpal, Yuannan Cai, David F Steiner, Hester Van Boven, Robert Vink, et al. Artificial intelligence for diagnosis and gleason grading of prostate cancer: the panda challenge. Nature medicine, 28(1):154–163, 2022.

[14] Thomas G. Dietterich, Richard H. Lathrop, and Tomás Lozano-Pérez. Solving the multiple instance problem with axis-parallel rectangles. Artificial Intelligence, 89(1):31–71, 1997.

[15] Zhuchen Shao, Hao Bian, Yang Chen, Yifeng Wang, Jian Zhang, Xiangyang Ji, and Yongbing Zhang. Transmil: Transformer based correlated multiple instance learning for whole slide image classification. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 2136–2147. Curran Associates, Inc., 2021.

[16] Richard J. Chen, Chengkuan Chen, Yicong Li, Tiffany Y. Chen, Andrew D. Trister, Rahul G. Krishnan, and Faisal Mahmood. Scaling vision transformers to gigapixel images via hierarchical self-supervised learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16144–16155, June 2022.

[17] Wenhao Tang, Sheng Huang, Xiaoxian Zhang, Fengtao Zhou, Yi Zhang, and Bo Liu. Multiple instance learning framework with masked hard instance mining for whole slide image classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4078–4087, October 2023.

[18] Linghan Cai, Shenjin Huang, Ye Zhang, Jinpeng Lu, and Yongbing Zhang. Attrimil: Revisiting attention based multiple instance learning for whole-slide pathological image classification from a perspective of instance attributes. Medical Image Analysis, 103:103631, 2025.

[19] Yunlong Zhang, Honglin Li, Yunxuan Sun, Sunyi Zheng, Chenglu Zhu, and Lin Yang. Attentionchallenging multiple instance learning for whole slide image classification. In European conference on computer vision, pages 125–143. Springer, 2024.

[20] Ming Y. Lu, Bowen Chen, Drew F. K. Williamson, Richard J. Chen, Ivy Liang, Tong Ding, Guillaume Jaume, Igor Odintsov, Long Phi Le, Georg Gerber, Anil V. Parwani, Andrew Zhang, and Faisal Mahmood. A visual-language foundation model for computational pathology. Nature Medicine, 30(3):863–874, 2024.

[21] Hanwen Xu, Naoto Usuyama, Jaspreet Bagga, Sheng Zhang, Rajesh Rao, Tristan Naumann, Cliff Wong, Zelalem Gero, Javier González, Yu Gu, Yanbo Xu, Mu Wei, Wenhui Wang, Shuming Ma, Furu Wei, Jianwei Yang, Chunyuan Li, Jianfeng Gao, Jaylen Rosemon, Tucker Bower, Soohee Lee, Roshanthi Weerasinghe, Bill J. Wright, Ari Robicsek, Brian Piening, Carlo Bifulco, Sheng Wang, and Hoifung Poon. A whole-slide foundation model for digital pathology from real-world data. Nature, 630(8015):181–188, 2024.

[22] Hongyi Wang, Luyang Luo, Fang Wang, Ruofeng Tong, Yen-Wei Chen, Hongjie Hu, Lanfen Lin, and Hao Chen. Rethinking multiple instance learning for whole slide image classification: A bag-level classifier is a good instance-level teacher. IEEE Transactions on Medical Imaging, 43(11):3964–3976, 2024.

[23] Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings ofMachine Learning Research, pages 5338–5348. PMLR, 2020.

[24] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings ofthe IEEE international conference on computer vision, pages 618–626, 2017.

[25] Syed Ashar Javed, Dinkar Juyal, Harshith Padigela, Amaro Taylor-Weiner, Limin Yu, and aaditya prakash. Additive MIL: Intrinsically interpretable multiple instance learning for pathology. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022.

[26] Saarthak Kapse, Pushpak Pati, Srijan Das, Jingwei Zhang, Chao Chen, Maria Vakalopoulou, Joel Saltz, Dimitris Samaras, Rajarsi R. Gupta, and Prateek Prasanna. SI-MIL: Taming deep MIL for selfinterpretability in gigapixel histopathology. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11226–11237, 2024.

[27] Wojciech Samek, Alexander Binder, Grégoire Montavon, Sebastian Lapuschkin, and Klaus-Robert Müller. Evaluating the visualization of what a deep neural network has learned. IEEE Transactions on Neural Networks and Learning Systems, 28(11):2660–2673, 2017.

[28] Julius Hense, Mina Jamshidi Idaji, Oliver Eberle, Thomas Schnake, Jonas Dippel, Laure Ciernik, Oliver Buchstab, Andreas Mock, Frederick Klauschen, and Klaus Robert Müller. xMIL: Insightful explanations for multiple instance learning in histopathology. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[29] Sai Gurrapu, Ajay Kulkarni, Lifu Huang, Ismini Lourentzou, and Feras A. Batarseh. Rationalization for explainable nlp: a survey. Frontiers in Artificial Intelligence, Volume 6 - 2023, 2023.

[30] Tao Lei, Regina Barzilay, and Tommi Jaakkola. Rationalizing neural predictions. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 107–117, Austin, Texas, November 2016. Association for Computational Linguistics.

[31] Jasmijn Bastings, Wilker Aziz, and Ivan Titov. Interpretable neural predictions with differentiable binary variables. In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 2963–2977, Florence, Italy, July 2019. Association for Computational Linguistics.

[32] Libing Yuan, Shuaibo Hu, Kui Yu, and Le Wu. Boosting explainability through selective rationalization in pre-trained language models. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1, page 1867–1878, 2025.

[33] Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

[34] Surat Teerapittayanon, Bradley McDanel, and H. T. Kung. BranchyNet: Fast inference via early exiting from deep neural networks. In 2016 23rd International Conference on Pattern Recognition (ICPR), pages 2464–2469. IEEE, 2016.

[35] Mo Yu, Shiyu Chang, Yang Zhang, and Tommi Jaakkola. Rethinking cooperative rationalization: Introspective extraction and complement control. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4094–4103, Hong Kong, China, November 2019. Association for Computational Linguistics.

[36] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation, 2013.

[37] Linfeng Ye, Shayan Mohajer Hamidi, Zhixiang Chi, Guang Li, Mert Pilanci, Takahiro Ogawa, Miki Haseyama, and Konstantinos N. Plataniotis. ASMIL: Attention-stabilized multiple instance learning for whole-slide imaging. In The Fourteenth International Conference on Learning Representations, 2026.

[38] Hyun Do Jung, Jungwon Choi, and Hwiyoung Kim. Reamil: Reasoning- and evidence-aware multiple instance learning for whole-slide histopathology. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops, pages 40–45, March 2026.

[39] Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete random variables. In International Conference on Learning Representations, 2017.

[40] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In International Conference on Learning Representations, 2017.

[41] Vitali Petsiuk, Abir Das, and Kate Saenko. RISE: Randomized input sampling for explanation of black-box models. In British Machine Vision Conference (BMVC), 2018.

## A Qualitative Illustration

This appendix shows where FOCI-selected tiles appear, in WSI context, relative to two attention/selection baselines on the same input bag. The figure is illustrative and not a claim of clinical sufficiency. Informal pathologist feedback suggested that isolated patch-only review is not aligned with clinical slide review; we therefore present selected tiles only in WSI context. Figure 4 compares FOCI against the TransMIL CLS-proxy ranking and ASMIL hard-selection ranking on two LUSC slides from the TCGA-NSCLC test set; each ranking uses its own scoring source, with FOCI and the TransMIL proxy sharing the frozen TransMIL backbone and ASMIL using its native model.

TCGA-33-4582 (compact case, MSK=1): a single patch is enough to cross 90% confidence. The top row shows FOCI’s top-32 selections concentrated in a small densely cellular region, with three highlighted zoom-ins. The bottom row contrasts FOCI’s selection against the TransMIL CLS-proxy ranking and ASMIL hard selection on the same bag; all three methods cluster in similar regions, consistent with NSCLC’s selection-saturation regime where many tile subsets recover the model’s confidence.

TCGA-NK-A5D1 (multi-fragment case, MSK=103): the tissue spans multiple fragments with considerable morphological variation, and the model requires 103 patches before crossing the confidence threshold. Here the three methods diverge: FOCI concentrates on a single tissue fragment while attention and hard-selection rankings spread across multiple fragments. The high MSK reflects the slide’s genuine complexity rather than a failure of any single ranker.

Informal visual inspection note. On informal visual inspection with WSI context, a subset of highlighted FOCI regions appeared plausibly compatible with squamous histology in WSI context. This qualitative review was not blinded, was not systematic, was performed on n=2 slides for illustration only, and is reported as informal feedback rather than a reader study.

These two cases illustrate that compact low-MSK selections and diffuse high-MSK selections can correspond to visibly different tissue patterns, and that FOCI’s selections can differ from attention-based or hard-selection baselines on the same input. They do not establish clinical sufficiency.

![](images/b1e68a722327bb92857f095802b065f588b6077b7ab4c476c22ba83a1840576e.jpg)  
Top of each pair: WSI thumbnail with FOCI's top-32 selected tiles (yel ow) and the top-3 highlighted (orange) shown as zoom-in crops at 20× magnification. Bottom: same WSI rendered three times with each method's top-32 ranked tiles outlined.  
Figure 4: Qualitative illustration of FOCI selections on two LUSC slides. Each slide is shown twice. Top row ofeach pair: WSI thumbnail with FOCI’s top-32 selected tiles outlined in yellow and the top-3 highlighted in orange (#1, #2, #3), plus three zoom-in crops at 20× magnification. Bottom row ofeach pair: same WSI rendered three times with each method’s top-32 ranked tiles outlined; cyan = TransMIL CLS-proxy ranking, lime = ASMIL hard selection, yellow = FOCI selector. Top slide (TCGA-33-4582, MSK=1, compact): all three methods cluster in similar regions, consistent with NSCLC’s selection-saturation regime. Bottom slide (TCGA-NK-A5D1, MSK=103, multifragment): FOCI concentrates on a single tissue fragment while attention/hard-selection rankings spread across fragments. These examples are illustrative and do not establish clinical sufficiency.

## B Classification Preservation

FOCI freezes the encoder and MIL backbone and trains only the lightweight selector. Therefore, under standard full-bag inference without masking, the FOCI-augmented model produces the same logits as the standalone backbone. Slide-level AUC is preserved by construction for the primary full-bag prediction; FOCI changes only the post-hoc tile ranking and masked keep/drop evaluations. We verified this empirically across all configurations, which confirms full-bag AUC equivalence to four decimal places in every case (7 backbones × 3 datasets × 3 seeds).

Table 3: Per-tile ranking scores used by each method under SRP. All scores are extracted from the trained model on the same pre-filtered bag and produce a per-tile real number; higher scores are revealed earlier.
<table><tr><td>Method</td><td>SRP ranking score</td></tr><tr><td>TransMIL [15]</td><td>Post-encoder CLS-dot-product score  $\langle h _ { \mathrm { t o k } _ { i } } , h _ { \mathrm { c l s } } \rangle$  (documented CLS-proxy; sole TransMIL ranking source).</td></tr><tr><td>ABMIL [2]</td><td>Pre-softmax gated-attention logits  $\mathbf { w } ^ { \top } ( \operatorname { t a n h } ( \mathbf { V } x _ { i } ) \odot \sigma ( \mathbf { U } x _ { i } ) )$ </td></tr><tr><td>CLAM-SB [3]</td><td>Pre-softmax gated-attention logits (same gated form as ABMIL); class-specific branch shared with the predicted class.</td></tr><tr><td>ACMIL [19]</td><td>Mean over the multiple attention branches of the per-tile branch-attention logits.</td></tr><tr><td>AttriMIL [18]</td><td>Per-class attribute score inst  $\mathbf { \nabla } _ { : } ( x _ { i } ) \cdot \exp ( a _ { c } ^ { \mathrm { l o g i t } } ( x _ { i } ) )$  , taken as the maximum over classes.</td></tr><tr><td>ASMIL [37]</td><td>Online model&#x27;s pre-softmax gated-attention logits; the anchor-EMA path is used only during training.</td></tr><tr><td>MHIM-MIL [17]</td><td>Last-layer encoder attention logits exposed through the wrapper&#x27;s return_attn=True forward pass.</td></tr><tr><td>FOCI (ours)</td><td>Lightweight selector head  $a _ { i } = \mathrm { M L P } ( x _ { i } )$  trained with sufficiency, exclusion, and contiguity losses.</td></tr></table>

## C Per-method SRP ranking-score extraction

Each baseline contributes a per-tile ranking score to SRP. Table 3 lists the score used for every method in the 7-backbone matrix. All scores are computed from the trained model on the same pre-filtered bag (top $n _ { \mathrm { c a p } } { = } 1 0 2 4$ tokens by feature L2 norm; see Appendix E.2); a higher score means earlier reveal. TransMIL does not expose a native attention head, so we use a post-encoder CLS-dot-product score $\langle h _ { \mathrm { t o k } _ { i } } , h _ { \mathrm { c l s } } \rangle$ as a documented CLS-proxy ranking. For the other six backbones, the ranking score is the model’s own pre-softmax attention logit or attribute-attention aggregate, exposed through an attn\_logits interface. FOCI ranks tiles by its selector head, $a _ { i } = \mathrm { M L P } ( x _ { i } )$

This table is intentionally explicit because SRP is sensitive to ranking quality. The same masking and reveal procedure is applied regardless of how each score is obtained. Methods that perform hard instance selection during training, such as ASMIL and MHIM-MIL, still expose continuous attention logits through this interface; these logits are what we rank for side-by-side SRP comparison in Tables 12–14.

## D Equal-Budget Comparison

Table 4 evaluates all methods under a fixed reveal budget of $K _ { \mathrm { m a x } } = 3 2$ , which matches the FOCI-STE training target. This equal-budget setting provides a complementary view to the main SRP results at $K _ { \operatorname* { m a x } } = 2 5 6$ (Tables 12–14) by isolating early-ranking quality under a small tile budget. The same headroom-vs-saturation pattern remains visible: FOCI helps when the baseline ranking has room to compress, while near-minimal attention-pooling baselines leave little margin for further reduction.

## E Sensitivity Analysis

We analyze the sensitivity of FOCI-STE to hyperparameters that govern the SRP evaluation protocol and the selector budget.

## E.1 Operating confidence threshold κ

The operating confidence threshold κ determines when a classifier is deemed sufficiently confident during sequential reveal. Table 5 reports MSK, Reach, and AUKC for FOCI-STE at $\kappa \in \{ 0 . 7 , 0 . 8 , 0 . 9 , 0 . 9 5 \}$ across all three datasets. AUKC is invariant to κ by construction: it is computed from the full reveal-probability curve, so the threshold affects only which slides reach the operating point and how many tiles are counted toward MSK. Reach generally decreases as κ rises. $\mathrm { M S K } _ { \mathrm { c o n d } }$ can shift non-monotonically because it is averaged only over reachable slides: higher thresholds require more tiles for reachable slides, while the hardest slides may drop out of the conditional set. The default κ=0.9 provides a conservative operating point for the main results.

Table 4: Equal K-budget comparison at $K _ { \mathrm { m a x } } = 3 2 ( \kappa = 0 . 9 , n _ { \mathrm { c a p } } = 1 0 2 4 _ { \mathrm { \ell } }$ , 3-seed mean±std). Paired rows show each backbone before and after attaching the FOCI selector. All methods are evaluated under SRP with the same reveal budget, which matches FOCI-STE’s training target. Bold: best per column. Underline: second best.
<table><tr><td rowspan="2">Method</td><td colspan="3">NSCLC</td><td colspan="3">BRCA</td><td colspan="3">PANDA</td></tr><tr><td> $\overline { { { \bf M S K } _ { \mathrm { c o n d } } \downarrow } }$ </td><td>Reach (%) ↑</td><td>AUKC↑</td><td> $\overline { { { \bf M S K } _ { \mathrm { c o n d } } \downarrow } }$ </td><td>Reach (%) ↑</td><td>AUKC↑</td><td> $\overline { { { \bf M S K } _ { \mathrm { c o n d } } \downarrow } }$ </td><td> $\overline { { \mathrm { R e a c h } \left( \% \right) \uparrow } }$ </td><td>AUKC ↑</td></tr><tr><td colspan="10">Soft-aggregation backbones</td></tr><tr><td>TransMIL</td><td>2.1 ± 0.4</td><td>87.7 ± 0.5</td><td> $0 . 8 2 9 \pm 0 . 0 0 6$ </td><td>1.4 ± 0.3</td><td>81.3 ± 4.3</td><td> $0 . 8 0 9 \pm 0 . 0 1 1$ </td><td>3.1 ± 0.2</td><td>81.0 ± 2.4</td><td> $0 . 7 6 7 \pm 0 . 0 2 7$ </td></tr><tr><td>+ FOCI</td><td> $1 . 7 \pm 0 . 1$ </td><td> $8 8 . 5 \pm 3 . 8 $ </td><td> $0 . { \dot { 8 } } 4 { \dot { 6 } } \pm { \dot { 0 } } . 0 1 { \dot { 6 } }$ </td><td> $1 . 6 \pm 0 . { \overset { . } { 4 } }$ </td><td> $8 3 . 7 \pm 3 . 3$ </td><td> $0 . 8 2 2 \pm 0 . 0 1 1$ </td><td> $2 . 6 \pm 0 . 6$ </td><td> $8 4 . 3 \pm 3 . 9$ </td><td> $0 . 8 2 4 \pm 0 . 0 3 3$ </td></tr><tr><td>ABMIL</td><td> $1 . 2 \pm 0 . 0$ </td><td> $9 1 . 1 \pm 0 . 5$ </td><td> $0 . 8 9 1 \pm 0 . 0 0 5$ </td><td>1.1 ± 0.1</td><td>85.4 ± 2.6</td><td> $0 . 8 3 6 \pm 0 . 0 2 1$ </td><td> $1 . 5 \pm 0 . 2$ </td><td>88.1 ± 3.2</td><td></td></tr><tr><td>+ FOCI</td><td> $\underline { { 1 . 2 \pm 0 . 1 } }$ </td><td> $9 2 . 8 \pm 1 . 8$ </td><td> $\underline { { 0 . 8 9 9 } } \pm 0 . 0 0 9$ </td><td>1.2 ± 0.1</td><td> $8 4 . 5 \pm 4 . 9$ </td><td> $0 . 8 2 6 \pm 0 . 0 2 7$ </td><td> $\overline { { 1 . 7 \pm 0 . 7 } }$ </td><td> $9 0 . 9 \pm 3 . 0$ </td><td> $\frac { 0 . 8 9 7 \pm 0 . 0 1 7 } { 0 . 8 7 3 \pm 0 . 0 5 0 }$ </td></tr><tr><td></td><td>1.3 ± 0.1</td><td> $9 0 . 9 \pm 1 . 0$ </td><td> $0 . 8 8 9 \pm 0 . 0 0 7$ </td><td> ${ \bf 1 . 0 \pm 0 . 0 }$ </td><td> $8 7 . 0 \pm 1 . 0$ </td><td></td><td> $1 . 6 \pm 0 . 2$ </td><td> $8 8 . 2 \pm 1 . 6 $ </td><td></td></tr><tr><td> $^ \mathrm { C L A M - S B } _ { + \mathrm { F O C I } }$ </td><td> $1 . 3 \pm 0 . 3$ </td><td> $9 2 . 5 \pm 1 . 0$ </td><td> $0 . 8 8 9 \pm 0 . 0 1 3$ </td><td> $1 . 8 \pm 0 . 6$ </td><td> $8 7 . 7 \pm 2 . 9$ </td><td> $\frac { 0 . 8 4 8 \pm 0 . 0 0 3 } { 0 . 8 3 2 \pm 0 . 0 2 8 }$ </td><td> $1 . 6 \pm 0 . 6$ </td><td> $_ { 9 1 . 5 \pm 2 . 1 } ^ { 8 8 . 2 }$ </td><td> $\frac { 0 . 8 9 7 \pm 0 . 0 0 9 } { 0 . 8 7 5 \pm 0 . 0 3 9 }$ </td></tr><tr><td></td><td> $\underline { { 1 . 2 \pm 0 . 1 } }$ </td><td> $9 0 . 4 \pm 1 . 4$ </td><td></td><td> $1 . 4 \pm 0 . 3$ </td><td> $8 3 . 6 \pm 3 . 1$ </td><td></td><td> ${ \bf 1 . 3 \pm 0 . 1 }$ </td><td></td><td></td></tr><tr><td> $\begin{array} { c } { { \bf A u t t r i M I L } } \\ { { \bf + F O C I } } \end{array}$ </td><td> $1 . 4 \pm 0 . 2$ </td><td> $9 1 . 4 \pm 0 . 4$ </td><td> $\begin{array} { c } { 0 . 8 8 2 \pm 0 . 0 1 2 } \\ { 0 . 8 8 6 \pm 0 . 0 1 0 } \end{array}$ </td><td> $1 . 8 \pm 0 . { \overset { . } { 4 } }$ </td><td> $\stackrel { \mathrm { \scriptsize { > } } \mathrm { . } \mathrm { 0 } } { 7 } \stackrel { \pm } { 8 } 4 . 6$ </td><td> $\begin{array} { c } { 0 . 8 1 8 \pm 0 . 0 2 2 } \\ { 0 . 8 0 9 \pm 0 . 0 2 8 } \end{array}$ </td><td> $1 . 7 \pm 0 . 2$ </td><td> $^ { 9 0 . 0 \pm 1 . 7 } _ { 9 1 . 8 \pm 0 . 4 }$ </td><td> $\frac { 0 . 8 9 7 \pm 0 . 0 0 6 } { 0 . 8 7 7 \pm 0 . 0 0 7 }$ </td></tr><tr><td></td><td> $1 . 8 \pm 0 . 1$ </td><td> $8 8 . 8 \pm 3 . 0$ </td><td> $0 . 8 5 2 \pm 0 . 0 2 0$ </td><td></td><td></td><td> $0 . 8 4 0 \pm 0 . 0 0 4$ </td><td> $1 . 8 \pm 0 . 0$ </td><td> ${ \bf 9 3 . 8 \pm 0 . 4 }$ </td><td> $0 . 8 8 1 \pm 0 . 0 0 4$ </td></tr><tr><td> $\begin{array} { c } { { \tt A C M I L } } \\ { { + \tt F O C I } } \end{array}$ </td><td>1.1 ± 0.1</td><td> $9 0 . 9 \pm 0 . 7$ </td><td> $0 . 8 8 7 \pm 0 . 0 0 3$ </td><td> $^ { 1 . 3 \pm 0 . 1 } _ { 1 . 3 \pm 0 . 1 }$ </td><td> $\frac { 8 8 . 5 \pm 1 . 5 } { 8 8 . 8 \pm 1 . 6 }$ </td><td> $\mathbf { 0 . 8 5 6 \pm 0 . 0 0 5 }$ </td><td> $\underline { { 1 . 5 \pm 0 . 2 } }$ </td><td> $\underline { { 9 3 . 2 \pm 1 . 8 } }$ </td><td> $\mathbf { 0 . 9 0 5 \pm 0 . 0 1 0 }$ </td></tr><tr><td colspan="10">Hard-selection backbones</td></tr><tr><td>ASMIL</td><td> $\underline { { 1 . 2 \pm 0 . 1 } }$ </td><td> ${ \bf 9 3 . 8 \pm 1 . 0 }$ </td><td>0.908 ± 0.003</td><td>2.4 ± 0.5</td><td> $7 0 . 9 \pm 1 2 . 0$ </td><td> $0 . 7 0 3 \pm 0 . 0 8 2$ </td><td>1.7 ± 0.1</td><td> $8 2 . 3 \pm 1 . 8$ </td><td>0.817 ± 0.014</td></tr><tr><td> $+ \operatorname { F O C I }$ </td><td> $\overline { { 1 . 4 \pm 0 . 3 } }$ </td><td> $8 7 . 7 \pm 1 . 9$ </td><td> $0 . 8 4 0 \pm 0 . 0 2 5$ </td><td>2.3 ± 0.7</td><td> $7 2 . 9 \pm 8 . 9$ </td><td> $0 . 7 3 4 \pm 0 . 0 6 9$ </td><td>3.4 ± 1.1</td><td> $7 0 . 6 \pm 1 4 . 8$ </td><td> $0 . 6 7 1 \pm 0 . 1 2 5$ </td></tr><tr><td> $\mathbf { M H I M - M I L }$ </td><td> $1 . 5 \pm 0 . 0$ </td><td>83.1 ± 4.0</td><td> $0 . 8 4 3 \pm 0 . 0 1 5$ </td><td>2.2 ± 0.6</td><td>55.9 ± 5.7</td><td> $0 . 7 5 8 \pm 0 . 0 2 0$ </td><td> $1 . 7 \pm 0 . 2$ </td><td>88.0 ± 1.7</td><td> $0 . 8 9 0 \pm 0 . 0 0 9$ </td></tr><tr><td>+ FOCI</td><td> $1 . 5 \pm 0 . 3$ </td><td> $8 1 . 8 \pm 2 . 9$ </td><td> $0 . 8 2 \bar { 7 } \pm 0 . 0 0 \bar { 6 }$ </td><td> $3 . 0 \pm 0 . 7$ </td><td> $5 2 . 5 \pm 8 . 9$ </td><td> $0 . 7 5 5 \pm 0 . 0 2 2$ </td><td> $\underline { { 1 . 5 \pm 0 . 0 } }$ </td><td> $8 7 . 8 \pm 2 . 1$ </td><td> $0 . 8 8 4 \pm 0 . 0 0 9$ </td></tr></table>

<table><tr><td rowspan="2">κ</td><td colspan="3">NSCLC</td><td colspan="3">BRCA</td><td colspan="3">PANDA</td></tr><tr><td> $\overline { { \mathrm { { M S K } } _ { \mathrm { { c o n d } } } \downarrow } }$ </td><td>Reach (%) ↑</td><td> $\overline { { \mathrm { { A U K C } \uparrow } } }$ </td><td> $\overline { { { \mathrm { \mathbf { M S K } } } _ { \mathrm { c o n d } } \downarrow } }$ </td><td> $\mathrm { R e a c h } \left( \% \right) \uparrow$ </td><td> $\overline { { { \bf A U K C \uparrow } } }$ </td><td> $\overline { { \mathrm { \mathbf { M S K } } _ { \mathrm { c o n d } } \downarrow } }$ </td><td> $\mathrm { R e a c h } \left( \% \right) \uparrow$ </td><td> $\overline { { \mathbf { A U K C \uparrow } } }$ </td></tr><tr><td>0.70</td><td>3.68 ± 0.59</td><td> $\overline { { 9 4 . 4 \pm 1 . 6 } }$ </td><td> $0 . 8 9 3 \pm 0 . 0 1 6$ </td><td> $3 . 6 8 \pm 1 . 2 8 $ </td><td> $8 9 . 8 \pm 1 . 1$ </td><td> $0 . 8 5 6 \pm 0 . 0 1 3$ </td><td> $9 . 0 1 \pm 2 . 7 9$ </td><td> $\overline { { 9 6 . 3 \pm 0 . 7 } }$ </td><td> $0 . 9 0 3 \pm 0 . 0 1 8$ </td></tr><tr><td>0.80</td><td> $3 . 8 2 \pm 0 . 8 4$ </td><td> $9 4 . 1 \pm 1 . 6 $ </td><td> $0 . 8 9 3 \pm 0 . 0 1 6$ </td><td> $3 . 2 9 \pm 0 . 9 6$ </td><td> $8 8 . 8 \pm 1 . 1$ </td><td> $0 . 8 5 6 \pm 0 . 0 1 3$ </td><td> $9 . 5 7 \pm 3 . 2 1$ </td><td> $9 5 . 2 \pm 0 . 6$ </td><td> $0 . 9 0 3 \pm 0 . 0 1 8$ </td></tr><tr><td>0.90</td><td> $3 . 2 1 \pm 0 . 3 8$ </td><td> $9 0 . 1 \pm 4 . 3 $ </td><td> $0 . 8 9 3 \pm 0 . 0 1 6$ </td><td> $3 . 8 6 \pm 0 . 9 5$ </td><td> $8 5 . 7 \pm 2 . 5$ </td><td> $0 . 8 5 6 \pm 0 . 0 1 3$ </td><td> $1 0 . 6 2 \pm 3 . 4 5$ </td><td> $9 2 . 0 \pm 1 . 4$ </td><td> $0 . 9 0 3 \pm 0 . 0 1 8$ </td></tr><tr><td>0.95</td><td> $2 . 9 6 \pm 1 . 2 7$ </td><td> $7 5 . 8 \pm 2 4 . 2$ </td><td> $0 . 8 9 3 \pm 0 . 0 1 6$ </td><td> $5 . 1 3 \pm 0 . 8 6$ </td><td> $7 7 . 0 \pm 1 0 . 3$ </td><td> $0 . 8 5 6 \pm 0 . 0 1 \dot { 3 }$ </td><td> $1 1 . 2 5 \pm 3 . 8 7$ </td><td> $8 8 . 8 \pm 2 . 0$ </td><td> $0 . 9 0 3 \pm 0 . 0 1 8$ </td></tr></table>

## E.2 Pre-filter budget $n _ { \mathrm { c a p } }$

Table 5: Sensitivity to operating confidence threshold κ on FOCI-STE. All results use $K _ { \operatorname* { m a x } } = 2 5 6$ and $n _ { \mathrm { c a p } } = 1 0 2 4$ . Bold κ: default used in all main results. AUKC is unthresholded and therefore invariant to κ for a fixed reveal curve.

The pre-filter budget $n _ { \mathrm { c a p } }$ controls how many patches are retained by L2-norm pre-filtering before FOCI re-ranks them. Table 6 varies $n _ { \mathrm { c a p } } \in \{ 2 5 6 , 5 1 2 , 1 0 2 4 , 2 0 4 8 \}$ while keeping the SRP threshold fixed at $\kappa = 0 . 9$ . AUKC varies by less than 0.006 across all settings and datasets, which indicates that the FOCI-STE ranking is not sharply sensitive to this pre-filter budget. $\mathrm { M S K } _ { \mathrm { c o n d } }$ can shift, especially on BRCA, because changing the candidate pool alters the marginal patch distribution even when the overall ranking signal remains stable.

## E.3 Adaptive K training schedule $( \alpha , K _ { \operatorname* { m i n } } )$

Section G.3 evaluates a separate adaptive-K training schedule, $K _ { s } = \operatorname* { m a x } ( K _ { \mathrm { m i n } } , \lfloor \alpha N _ { s } ^ { \mathrm { r e a l } } \rfloor )$ , for the selectedonly downstream analysis. This schedule is separate from the fixed- $K { = } 3 2$ FOCI-STE configuration used in the main SRP experiments; $N _ { s } ^ { \mathrm { r e a l } }$ is the unpadded token count of slide s. The default setting $( \alpha = 0 . 0 3 ,$ $K _ { \operatorname* { m i n } } = 1 6 )$ corresponds to approximately $K \approx 3 0$ at the pre-filter cap $n _ { \mathrm { c a p } } = 1 0 2 4$ and uses $K = 1 6$ for short slides.

Table 7 reports validation AUKC for the default adaptive schedule on all three datasets, averaged over three seeds, and a single-seed α sweep on NSCLC. The sweep is intended as a sensitivity check rather than a separate model-selection procedure. AUKC varies by roughly 0.01–0.02 across $\alpha \in \left. 0 . 0 1 , 0 . 0 3 , 0 . 0 5 \right.$ , comparable to seed variation, which suggests that the adaptive rule is not sharply sensitive to the budget coefficient. Eval-time K-sensitivity at the default-trained selector is reported in §G.3.

Table 6: Sensitivity to pre-filter budget $n _ { \mathrm { c a p } }$ on FOCI-STE. All results use $\kappa = 0 . 9$ and $K _ { \operatorname* { m a x } } = 2 5 6$ Bold $n _ { \mathrm { c a p } } { \mathrm { : } }$ default.
<table><tr><td></td><td colspan="3">NSCLC</td><td colspan="3">BRCA</td><td colspan="3">PANDA</td></tr><tr><td> $n _ { \mathrm { c a p } }$ </td><td> $\mathbf { M S K } _ { \mathrm { c o n d } } \downarrow$ </td><td>Reach (%) ↑</td><td>AUKC↑</td><td> $\mathbf { M S K } _ { \mathrm { c o n d } } \downarrow$ </td><td>Reach (%) ↑</td><td> $\mathbf { A U K C \uparrow }$ </td><td> $\mathbf { M S K } _ { \mathrm { c o n d } } \downarrow$ </td><td>Reach (%) ↑</td><td>AUKC↑</td></tr><tr><td>256</td><td> $3 . 2 4 \pm 1 . 4 2$ </td><td> $9 1 . 1 \pm 2 . 2 $ </td><td> $0 . 8 9 2 \pm 0 . 0 1 2$ </td><td>2.36 ± 0.86</td><td> $8 4 . 9 \pm 2 . 4$ </td><td> $0 . 8 5 0 \pm 0 . 0 1 2$ </td><td> $8 . 9 8 \pm 3 . 4 5$ </td><td> $9 2 . 2 \pm 0 . 8$ </td><td> $0 . 9 0 8 \pm 0 . 0 1 5$ </td></tr><tr><td>512</td><td> $4 . 1 6 \pm 1 . 3 6$ </td><td> $9 1 . 4 \pm 2 . 2 $ </td><td> $0 . 8 9 3 \pm 0 . 0 1 0$ </td><td> $2 . 7 2 \pm 1 . 2 7$ </td><td> $8 5 . 1 \pm 2 . 1$ </td><td> $0 . 8 5 3 \pm 0 . 0 1 0$ </td><td> $1 0 . 6 0 \pm 3 . 4 7$   $\begin{array} { l } { { \scriptstyle 1 0 . 0 0 \pm \delta . 4 } } \\ { { \scriptstyle 1 0 \ 6 0 \pm 2 \ A 5 } } \end{array}$ </td><td> $9 2 . 3 \pm 1 . 2$ </td><td> $0 . 9 0 5 \pm 0 . 0 1 7$ </td></tr><tr><td>1024</td><td> $3 . 2 1 \pm 0 . 3 8$ </td><td> $9 0 . 1 \pm 4 . 3 $ </td><td> $0 . 8 9 3 \pm 0 . 0 1 6$ </td><td> $3 . 8 6 \pm 0 . 9 5$ </td><td> $8 5 . 7 \pm 2 . 5$ </td><td> $0 . 8 5 6 \pm 0 . 0 1 3$ </td><td> $1 0 . 6 2 \pm 3 . 4 5$ </td><td> $9 2 . 0 \pm 1 . 4$ </td><td> $0 . 9 0 3 \pm 0 . 0 1 8$ </td></tr><tr><td>2048</td><td> $4 . 1 6 \pm 1 . 4 2$ </td><td> $9 0 . 0 \pm 4 . 9$ </td><td> $0 . 8 8 9 \pm 0 . 0 1 9$ </td><td> $5 . 2 9 \pm 2 . 1 6$ </td><td> $8 6 . 2 \pm 2 . 5$ </td><td> $0 . 8 5 6 \pm 0 . 0 1 1$ </td><td> $1 0 . 6 9 \pm 3 . 4 7$ </td><td> $9 2 . 1 \pm 1 . 4$ </td><td> $0 . 9 0 3 \pm 0 . 0 1 8$ </td></tr></table>

Table 7: Adaptive K training schedule sensitivity. Top block: default $\alpha = 0 . 0 3$ with $K _ { \mathrm { m i n } } = 1 6 .$ mean±std over three seeds. Bottom block: single-seed (seed 42) α sweep on NSCLC for sensitivity characterization.
<table><tr><td>Dataset</td><td>α</td><td> $K _ { \mathrm { m i n } }$ </td><td>Validation AUKC</td></tr><tr><td colspan="4">Default schedule, three seeds</td></tr><tr><td>NSCLC</td><td>0.03</td><td>16</td><td> $0 . 8 5 4 \pm 0 . 0 1 1$ </td></tr><tr><td>BRCA PANDA</td><td>0.03 0.03</td><td>16 16</td><td> $0 . 8 0 1 \pm 0 . 0 0 5$   $0 . 8 1 7 \pm 0 . 0 2 7$ </td></tr><tr><td colspan="4"></td></tr><tr><td>α sweep, NSCLC</td><td>0.01</td><td>NSCLC seed 42 only 16</td><td>0.853</td></tr><tr><td>NSCLC</td><td>0.03</td><td>16</td><td>0.845</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>NSCLC</td><td>0.05</td><td>16</td><td>0.863</td></tr></table>

## F E<sub>x</sub>t<sub>e</sub>nd<sub>e</sub>d SRP C<sub>u</sub>r<sub>ves</sub>

![](images/1c1b354faeb825b3a91a04f105d00f48e0f945e880dbacd37cd5eb2cd536a467.jpg)  
Figure 5 : Extended SRP reveal curves for seven backbones ± FOCI across NSCLC BRCA and PANDA<sub>.</sub> Curves show the full confidence–K trajectories behind T<sub>a</sub>bl<sub>es</sub> $1 2 { - } 1 4 ;$ bl<sub>ue so</sub>lid = <sub>s</sub>t<sub>an</sub>d<sub>a</sub>l<sub>one</sub> b<sub>ac</sub>kb<sub>one re</sub>d d<sub>as</sub>h<sub>e</sub>d = FOCI-<sub>augmen</sub>t<sub>e</sub>d <sub>s</sub>h<sub>a</sub>d<sub>e</sub>d = ± 1 <sub>s</sub>td <sub>over</sub> th<sub>ree see</sub>d<sub>s</sub> d<sub>o</sub>tt<sub>e</sub>d li<sub>ne</sub> = <sub>κ</sub> = 0 <sub>.</sub> 9 <sub>.</sub>

## G Additional analyses

## G.1 Cross-method SRP reveal curves

Figure 6 shows cross-method SRP confidence curves on TCGA-NSCLC, TCGA-BRCA, and PANDA. True-class probability $p _ { y } ( K )$ is averaged over test slides and three seeds as tiles are revealed in descending score order. SRP applies uniformly to any method’s tile ranking. ASMIL ranks strongly on NSCLC but collapses on BRCA, a failure mode not visible from slide-level AUC alone. FOCI-STE improves the TransMIL SRP footprint across datasets without the cross-dataset collapse observed in some hard-selection backbones.

![](images/dc266e10664031ffc960fc9d3bd4430ce775a65e1db5cb302b8c10243d421db9.jpg)  
Figure 6: Cross-method SRP confidence curves on TCGA-NSCLC, TCGA-BRCA, and PANDA. True-class probability $p _ { y } ( K )$ is averaged over test slides and three seeds as tiles are revealed in descending score order.

## G.2 Inference efficiency

FOCI adds only a lightweight selector on top of the frozen backbone. Table 8 reports standard full-bag inference latency and memory, together with the offline SRP evaluation cost. Adding FOCI introduces negligible measured overhead for standard inference at the reported precision; SRP is a separate offline analysis which requires repeated masked forward passes.

Table 8: Inference efficiency on NSCLC (mean over 20 slides, RTX 6000 Ada). Paired rows show each backbone before and after attaching FOCI. Standard inference uses the full bag once; SRP columns report the one-time offline cost of sequential reveal evaluation.
<table><tr><td>Method</td><td>Params</td><td>Infer (ms)</td><td>∆ Infer</td><td>VRAM (MB)</td><td>SRP (ms)</td><td>SRP VRAM (MB)</td></tr><tr><td>TransMIL</td><td>13.5M</td><td>1.4</td><td></td><td>264</td><td>127.2</td><td>3,841</td></tr><tr><td>+ FOCI-STE</td><td>13.7M</td><td>1.4</td><td>+0.0</td><td>263</td><td>128.1</td><td>3,841</td></tr><tr><td>ABMIL</td><td>0.49M</td><td>0.6</td><td></td><td>176</td><td>2.1</td><td>559</td></tr><tr><td>+ FOCI-STE</td><td>0.62M</td><td>0.6</td><td>+0.0</td><td>177</td><td>2.1</td><td>559</td></tr><tr><td>ACMIL</td><td>0.92M</td><td>1.1</td><td></td><td>178</td><td>107.6</td><td>179</td></tr><tr><td>+ FOCI-STE</td><td>1.05M</td><td>1.1</td><td>+0.0</td><td>179</td><td>107.6</td><td>179</td></tr><tr><td>ASMIL</td><td>0.56M</td><td>0.4</td><td>一</td><td>177</td><td>2.1</td><td>560</td></tr><tr><td>CLAM-SB</td><td>0.49M</td><td>0.4</td><td>一</td><td>176</td><td>2.1</td><td>559</td></tr><tr><td>MHIM-MIL</td><td>1.18M</td><td>0.3</td><td>一</td><td>184</td><td>3.3</td><td>820</td></tr><tr><td>AttriMIL</td><td>1.71M</td><td>0.7</td><td></td><td>185</td><td>13.8</td><td>1,209</td></tr></table>

## G.3 Selected-only downstream performance

SRP and deletion-based faithfulness characterize rationale ranking quality, but they do not directly answer an audit-relevant question: ifthe model only sees the top-K selected tiles, does the slide-level prediction still hold? Because the backbone is frozen, the primary full-bag prediction is preserved by construction; the meaningful test is whether the selected K-tile subset alone preserves the prediction. Table 9 reports top-K classification metrics in which each (predictor, ranking) pipeline is restricted to its top-K tiles via key\_padding\_mask exclusion of the rest.

Table 9 supports the headroom interpretation rather than a universal dominance claim. On BRCA, adaptive-K FOCI preserves the full-bag TransMIL AUC (0.907), which outperforms fixed-K=32 FOCI and random K=32. On NSCLC and PANDA, random top-32 subsets already retain much of the full-bag prediction, which indicates selection-saturation regimes with limited operating margin for any external selector. ABMIL is reported as a native-pipeline reference only, since its backbone and full-bag baseline differ from TransMIL. Adaptive-K robustness on BRCA was further checked by an eval-time sweep over $K \in \{ 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ : selected-only AUC reaches 0.906 ± 0.015 even at K=16, which supports the $K _ { \mathrm { m i n } } \mathrm { = } 1 6$ floor used in the adaptive rule.

Table 9: Selected-only downstream performance. Each row is a (predictor, ranking) pipeline restricted to top-K tiles. Adaptive K uses $K _ { s } ^ { ' } = \operatorname* { m a x } ( K _ { \mathrm { m i n } } { = } 1 6 , \lfloor \alpha N _ { s } ^ { \mathrm { r e a l } } \rfloor )$ with $\alpha = 0 . 0 3$ (average K ≈ 30 at $n _ { \mathrm { c a p } } { = } 1 0 2 4 )$ . Random $K { = } 3 2$ uses TransMIL as the predictor for a random-control baseline. ABMIL is reported under its own native backbone and attention ranking.
<table><tr><td>Dataset</td><td>Method (Predictor + Ranking)</td><td>Top-K AUC</td><td>∆ vs. full-bag</td></tr><tr><td rowspan="5">NSCLC</td><td>TransMIL (full bag, K=1024)</td><td> $0 . 9 7 4 \pm 0 . 0 0 3$ </td><td>(baseline)</td></tr><tr><td>TransMIL + FOCI (fixed K=32)</td><td> $0 . 9 5 4 \pm 0 . 0 1 7$ </td><td>-0.020</td></tr><tr><td> $\mathrm { T r a n s M I L } + \mathrm { F O C I } \left( \mathrm { a d a p t i v e } \ : K _ { s } , \bar { K } { \approx } 3 0 \right)$ </td><td> $0 . 9 6 3 \pm 0 . 0 0 4$ </td><td>-0.011</td></tr><tr><td>TransMIL + Random K=32</td><td> $0 . 9 6 9 \pm 0 . 0 0 4$ </td><td>-0.005</td></tr><tr><td>ABMIL native (top-32)</td><td> $0 . 9 7 6 \pm 0 . 0 0 2$ </td><td>(own pipeline)</td></tr><tr><td rowspan="5">BRCA</td><td>TransMIL (full bag, K=1024)</td><td> $0 . 9 0 7 \pm 0 . 0 0 2$ </td><td>(baseline)</td></tr><tr><td>TransMIL + FOCI (fixed K=32)</td><td> $0 . 8 8 5 \pm 0 . 0 1 2$ </td><td>-0.022</td></tr><tr><td>TransMIL + FOCI (adaptive Ks, K≈30)</td><td> $0 . 9 0 7 \pm 0 . 0 1 1$ </td><td>±0.000</td></tr><tr><td>TransMIL + Random K=32</td><td> $0 . 8 8 1 \pm 0 . 0 0 7$ </td><td>-0.026</td></tr><tr><td>ABMIL native (top-32)</td><td> $0 . 8 6 5 \pm 0 . 0 4 0$ </td><td>(own pipeline)</td></tr><tr><td rowspan="5">PANDA</td><td>TransMIL (full bag, K=1024)</td><td> $0 . 9 8 9 \pm 0 . 0 0 2$ </td><td>(baseline)</td></tr><tr><td>TransMIL + FOCI (fixed K=32)</td><td> $0 . 9 4 5 \pm 0 . 0 0 6$ </td><td>-0.044</td></tr><tr><td>TransMIL + FOCI (adaptive  $K _ { s } , \bar { K } { \approx } 3 0 )$ </td><td> $0 . 9 3 4 \pm 0 . 0 1 2$ </td><td>-0.055</td></tr><tr><td>TransMIL + Random  $\bar { K } \mathrm { = } 3 2$ </td><td> $0 . 9 3 1 \pm 0 . 0 1 0$ </td><td>-0.058</td></tr><tr><td>ABMIL native (top-32)</td><td> $0 . 9 8 9 \pm 0 . 0 0 1$ </td><td>(own pipeline)</td></tr></table>

## G.4 Ablation study

STE vs. soft gate. Table 10 compares FOCI-STE and FOCI-Soft on NSCLC under the same frozen backbone, losses, and hyperparameters. FOCI-Soft uses a Gumbel-sigmoid relaxation $( T { = } 0 . 5 )$ with an entropy regularizer $( \lambda _ { \mathrm { { e n t } } } \mathrm { { = } } 0 . 1 )$ , whereas FOCI-STE uses a hard top-K forward mask with a sigmoid surrogate gradient. FOCI-Soft underperforms the freeze-only control on MSK, consistent with a soft-vs-hard cardinality gap: training uses continuous nonzero gates, whereas SRP evaluates hard top-K subsets. FOCI-STE narrows this mismatch, reduces MSK from 8.03 to 3.21, and preserves the same frozen classifier.

Frozen vs. joint training. In a pilot NSCLC/TransMIL run, unfreezing the backbone and training jointly with the rationale losses reduced validation AUC by more than 15 percentage points within two epochs. We therefore freeze the trained MIL backbone and use FOCI as a readout head over a stable feature space rather than as a jointly trained classifier component.

Loss components. Table 11 ablates each loss term by setting it to zero. The main failure mode is Reach collapse: full FOCI-STE reaches κ=0.9 on 90.1% of slides, whereas removing any single term reduces Reach to 20–34%. The sufficiency and exclusion terms are both needed to define the keep/drop contrast, and the contiguity term improves stability. Compared with the freeze-only selector, full FOCI-STE trades a small decrease in Reach/AUKC for a large MSK reduction, consistent with optimizing compact rationale recovery rather than uniformly improving every SRP metric.

Table 10: Gate formulation ablation on NSCLC (κ=0.9). Values are 3-seed mean except the freezeonly diagnostic control, which uses two seeds.
<table><tr><td>Method</td><td> $\mathbf { M S K } _ { \mathrm { c o n d } } \downarrow$ </td><td>Reach (%) ↑</td><td>AUKC ↑</td></tr><tr><td>TransMIL (frozen, no selection)</td><td> $7 . 3 3 \pm 0 . 3 9$ </td><td> $9 1 . 7 \pm 0 . 9$ </td><td> $0 . 8 9 0 \pm 0 . 0 0 8$ </td></tr><tr><td>Freeze-only (no rationale loss)</td><td> $7 . 6 0 \pm 0 . 1 8$ </td><td> ${ \bf 9 3 . 8 \pm 1 . 0 }$ </td><td> $\mathbf { 0 . 9 0 6 \pm 0 . 0 0 6 }$ </td></tr><tr><td>FOCI-Soft (Gumbel gate)</td><td> $8 . 0 3 \pm 0 . 4 5$ </td><td> $8 7 . 1 \pm 5 . 2$ </td><td> $0 . 8 6 6 \pm 0 . 0 0 8$ </td></tr><tr><td>FOCI-STE (hard top-K)</td><td> ${ \bf 3 . 2 1 \pm 0 . 3 8 }$ </td><td> $9 0 . 1 \pm 4 . 3 $ </td><td> $0 . 8 9 3 \pm 0 . 0 1 6$ </td></tr></table>

Table 11: Loss component ablation on NSCLC (κ=0.9, 3-seed mean±std). Each row zeroes one loss while keeping the others at their tuned values.
<table><tr><td>Method</td><td> $\mathbf { M S K } _ { \mathrm { c o n d } } \downarrow$ </td><td> $\mathrm { R e a c h } \left( \% \right) \uparrow$ </td><td> $\mathbf { A U K C \uparrow }$ </td></tr><tr><td>Freeze-only (no rationale loss)</td><td> $7 . 6 0 \pm 0 . 1 8$ </td><td> ${ \bf 9 3 . 8 \pm 1 . 0 }$ </td><td> $\mathbf { 0 . 9 0 6 \pm 0 . 0 0 6 }$ </td></tr><tr><td>Full FOCI-STE</td><td> ${ \bf 3 . 2 1 \pm 0 . 3 8 }$ </td><td> $9 0 . 1 \pm 4 . 3$ </td><td> $0 . 8 9 3 \pm 0 . 0 1 6$ </td></tr><tr><td> $\lambda _ { \mathrm { s u f f } } { = } 0$ </td><td> $1 . 5 2 \pm 0 . 6 5$ </td><td> $3 4 . 4 \pm 1 4 . 7$ </td><td> $0 . 5 1 9 \pm 0 . 0 1 2$ </td></tr><tr><td> $\lambda _ { \mathrm { e x c l } } { = } 0$ </td><td> $1 . 7 8 \pm 0 . 6 5$ </td><td> $3 0 . 1 \pm 2 1 . 5$ </td><td> $0 . 5 0 8 \pm 0 . 0 0 5$ </td></tr><tr><td> $\lambda _ { \mathrm { c o n t i g } } { = } 0$ </td><td> $2 7 . 4 1 \pm 2 1 . 7 3$ </td><td> $2 0 . 3 \pm 1 3 . 7$ </td><td> $0 . 5 1 6 \pm 0 . 0 0 5$ </td></tr></table>

## H Per-dataset SRP main result tables

Tables 12–14 report per-dataset SRP results at κ=0.9 for each frozen backbone and its FOCI-augmented counterpart. The Selection Headroom Index summary in Table 1 is computed from the same per-seed runs. These tables provide the raw per-dataset breakdown behind the family-wise SHI analysis in the main text.

Table 12: SRP results on NSCLC comparing each backbone with and without FOCI (3-seed mean±std). Bold/underline: best/second best. Deltas show change from baseline (improved, degraded).
<table><tr><td>Method</td><td> $\mathbf { M S K } _ { \mathrm { c o n d } } \downarrow$ </td><td>Reach (%) ↑</td><td>AUKC↑</td></tr><tr><td colspan="4">Soft-aggregation backbones</td></tr><tr><td> $\mathrm { T r a n s M L }$ </td><td> $7 . 3 3 \pm 0 . 3 9$ </td><td> $9 1 . 7 \pm 0 . 9$ </td><td> $0 . 8 9 0 \pm 0 . 0 0 8$ </td></tr><tr><td>+ FOCI</td><td> $3 . 2 1 \pm 0 . 3 8 \ ( - 4 . 1 2 )$ </td><td> $9 0 . 1 \pm 4 . 3 \ : ( - 1 . 6 )$ </td><td> $0 . 8 9 3 \pm 0 . 0 1 6 _ { ( + . 0 0 3 ) }$ </td></tr><tr><td> $\mathbf { A B M L }$ </td><td> $2 . 6 5 \pm 0 . 7 3$ </td><td> $9 2 . 2 \pm 0 . 5$ </td><td> $0 . 9 2 5 \pm 0 . 0 0 2$ </td></tr><tr><td>+FOCI</td><td> $1 . 5 2 \pm 0 . 0 8 ( - 1 . 1 3 )$ </td><td> $9 3 . 5 \pm 1 . 6 \ ( + 1 . 3 ) $ </td><td> $\underline { { 0 . 9 2 8 \pm 0 . 0 0 5 } } \ : ( + . 0 0 3 )$ </td></tr><tr><td> $\mathrm { C L A M - S B }$ </td><td> $2 . 0 9 \pm 0 . 5 6$ </td><td> $9 1 . 7 \pm 0 . 5$ </td><td> $0 . 9 2 3 \pm 0 . 0 0 3$ </td></tr><tr><td> $+ \operatorname { F O C I }$ </td><td> $1 . 5 7 \pm 0 . 3 0 \ _ { ( - 0 . 5 2 ) }$ </td><td> $9 2 . 7 \pm 0 . 8 ~ ( + 1 . 0 )$ </td><td> $0 . 9 2 1 \pm 0 . 0 0 9 \ : ( . . 0 0 2 )$ </td></tr><tr><td> $\operatorname { A t t r i M L }$ </td><td> $3 . 2 1 \pm 0 . 8 1$ </td><td> $9 2 . 0 \pm 1 . 8$ </td><td> $0 . 9 2 0 \pm 0 . 0 1 0$ </td></tr><tr><td> $+ \operatorname { F O C I }$ </td><td> $2 . 1 6 \pm 1 . 0 4 ( - 1 . 0 5 )$ </td><td> $9 2 . 2 \pm 0 . 8 ~ ( + 0 . 2 )$ </td><td> $0 . 9 1 7 \pm 0 . 0 1 0 \ ( \div 0 0 3 )$ </td></tr><tr><td> $\mathbf { A C M L }$ </td><td></td><td></td><td></td></tr><tr><td>+FOCI</td><td> $6 . 0 8 \pm 1 . 0 9$   $1 . 7 9 \pm 0 . 0 2 \ ( - 4 . 2 9 )$ </td><td>92.4 ± 1.4  $9 1 . 5 \pm 0 . 8 ( - 0 . 9 ) $ </td><td>0.907 ± 0.009  $0 . 9 1 7 \pm 0 . 0 0 1 \ : ( + . 0 1 0 )$ </td></tr><tr><td colspan="4">Hard-selection backbones</td></tr><tr><td>ASMIL</td><td> ${ \bf 1 . 3 6 \pm 0 . 3 7 }$ </td><td> ${ \bf 9 4 . 1 \pm 0 . 6 }$ </td><td> $\mathbf { 0 . 9 3 5 \pm 0 . 0 0 0 }$ </td></tr><tr><td>+ FOCI</td><td> $4 . 1 6 \pm 1 . 4 4 \ : ( + 2 . 8 0 )$ </td><td> $9 0 . 9 \pm 0 . 7 \ : ( - 3 . 2 )$ </td><td> $0 . 8 9 6 \pm 0 . 0 1 1 ~ ( - . 0 3 9 )$ </td></tr><tr><td> $\mathbf { M H I M - M I L }$ </td><td> $2 . 7 7 \pm 1 . 4 7$ </td><td> $8 4 . 4 \pm 3 . 0$ </td><td> $0 . 8 7 2 \pm 0 . 0 1 4$ </td></tr><tr><td>+ FOCI</td><td> $3 . 6 3 \pm 1 . 1 8 ~ ( + 0 . 8 6 )$ </td><td> $8 4 . 2 \pm 2 . 2 \ : ( - 0 . 2 ) $ </td><td> $0 . 8 6 5 \pm 0 . 0 0 2 \ ( \AA . 0 0 7 )$ </td></tr></table>

Table 13: SRP results on BRCA comparing each backbone with and without FOCI (3-seed mean±std). Bold/underline: best/second best. Deltas show change from baseline (improved, degraded).

$$
\mathbf { M S K } _ { \mathrm { c o n d } } \downarrow
$$

$$
b a c k b o n e s
$$

$$
5 . 6 5 \pm 3 . 4 6
$$

$$
8 4 . 5 \pm 2 . 6
$$

$$
0 . 8 4 0 \pm 0 . 0 1 0
$$

$$
3 . 8 6 \pm 0 . 9 5 ~ ( - 1 . 7 9 )
$$

$$
8 5 . 7 \pm 2 . 5 \ ( + 1 . 2 )
$$

$$
0 . 8 5 6 \pm 0 . 0 1 3 ~ ( + . 0 1 6 )
$$

$$
{ \bf 1 . 1 0 \pm 0 . 1 4 }
$$

$$
8 5 . 4 \pm 2 . 6
$$

$$
3 . 2 1 \pm 1 . 4 9 \ ( + 2 . 1 1 )
$$

$$
0 . 8 6 2 \pm 0 . 0 1 8
$$

$$
8 6 . 1 \pm 3 . 8 ~ ( + 0 . 7 )
$$

$$
0 . 8 5 9 \pm 0 . 0 2 1 \ ( \cdot . 0 0 3 )
$$

$$
\mathbf { C L A M - S B }
$$

$$
1 . 1 7 \pm 0 . 2 2
$$

$$
4 . 1 6 \pm 3 . 8 7 ( + 2 . 9 9 )
$$

$$
8 7 . 1 \pm 0 . 9
$$

$$
0 . 8 7 1 \pm 0 . 0 0 5
$$

$$
9 0 . 1 \pm 1 . 9 ( + 3 . 0 ) 
$$

$$
\underline { { 0 . 8 7 3 \pm 0 . 0 0 8 } } ( + . 0 0 2 )
$$

$$
\mathbf { A t t r i M L }
$$

$$
3 . 5 2 \pm 1 . 8 2
$$

$$
8 5 . 7 \pm 1 . 6
$$

$$
6 . 7 9 \pm 3 . 7 8 ~ ( + 3 . 2 7 )
$$

$$
0 . 8 5 7 \pm 0 . 0 1 0
$$

$$
8 2 . 5 \pm 3 . 6 ( - 3 . 2 )
$$

$$
0 . 8 4 9 \pm 0 . 0 1 7 \ : ( - . 0 0 8 )
$$

$$
\mathbf { A C M L }
$$

$$
3 . 3 9 \pm 0 . 5 7
$$

$$
2 . 2 5 \pm 0 . 5 5 ( - 1 . 1 4 )
$$

$$
{ \bf 9 0 . 3 \pm 1 . 1 }
$$

$$
0 . 8 6 7 \pm 0 . 0 0 7
$$

$$
9 0 . 0 \pm 1 . 8 \ : ( - 0 . 3 ) 
$$

$$
\mathbf { 0 . 8 8 0 \pm 0 . 0 0 5 \ ( + . 0 1 3 ) }
$$

$$
b a c k b o n e s
$$

$$
\mathbf { A S M L }
$$

$$
1 5 . 8 3 \pm 6 . 6 3
$$

$$
+ \operatorname { F O C I }
$$

$$
1 2 . 8 1 \pm 5 . 9 6 ~ ( - 3 . 0 2 )
$$

$$
8 0 . 7 \pm 7 . 9
$$

$$
0 . 7 9 6 \pm 0 . 0 3 8
$$

$$
\mathbf { M H I M - M I L }
$$

$$
8 0 . 3 \pm 5 . 8 ( - 0 . 4 )
$$

$$
0 . 8 1 4 \pm 0 . 0 3 7 ~ ( + . 0 1 8 )
$$

$$
1 2 . 2 \pm 5 . 7 8
$$

$$
+ \operatorname { F O C I }
$$

$$
1 4 . 9 4 \pm 1 1 . 5 7 \ : ( + 2 . 7 4 )
$$

$$
6 0 . 4 \pm 3 . 0
$$

$$
5 7 . 8 \pm 4 . 8 ( - 2 . 6 )
$$

$$
0 . 7 9 7 \pm 0 . 0 0 7
$$

$$
0 . 7 9 5 \pm 0 . 0 1 3 \ ( . 0 0 2 )
$$

Table 14: SRP results on PANDA comparing each backbone with and without FOCI (3-seed mean±std). Bold/underline: best/second best. Deltas show change from baseline (improved, degraded).
<table><tr><td>Method</td><td> $\mathbf { M S K } _ { \mathrm { c o n d } } \downarrow$ </td><td>Reach (%) ↑</td><td>AUKC↑</td></tr><tr><td colspan="4">Soft-aggregation backbones</td></tr><tr><td> $\mathrm { T r a n s M I L }$ </td><td> $1 6 . 5 \pm 2 . 0$ </td><td> $9 3 . 4 \pm 1 . 4$ </td><td> $0 . 8 8 1 \pm 0 . 0 1 7$ </td></tr><tr><td>+ FOCI</td><td> $1 0 . 6 2 \pm 3 . 4 5 ~ ( - 5 . 8 8 )$ </td><td> $9 2 . 0 \pm 1 . 4 ( - 1 . 4 )$ </td><td> $0 . 9 0 3 \pm 0 . 0 1 8 _ { ( + . 0 2 2 ) }$ </td></tr><tr><td>ABMIL</td><td> $4 . 0 8 \pm 1 . 6 4$ </td><td> $9 0 . 5 \pm 2 . 0$ </td><td> $0 . 9 3 3 \pm 0 . 0 1 1$ </td></tr><tr><td> $+ \mathrm { F O C I }$ </td><td> $5 . 3 7 \pm 4 . 1 8 ( + 1 . 2 9 )$ </td><td> $9 4 . 6 \pm 1 . 1 ~ ( + 4 . 1 )$ </td><td> $0 . 9 2 7 \pm 0 . 0 1 9 \ : ( . . 0 0 6 )$ </td></tr><tr><td> $\mathbf { C L A M - S B }$ </td><td> $4 . 8 9 \pm 1 . 4 6$ </td><td> $9 1 . 1 \pm 0 . 8$ </td><td> $\mathbf { 0 . 9 3 4 \pm 0 . 0 0 5 }$ </td></tr><tr><td>+ FOCI</td><td> $4 . 8 1 \pm 2 . 8 9 \ ( - 0 . 0 8 )$ </td><td> $9 5 . 0 \pm 1 . 5 ~ ( + 3 . 9 )$ </td><td> $0 . 9 3 1 \pm 0 . 0 1 1 \ : ( - . 0 0 3 )$ </td></tr><tr><td> $\mathbf { A t t r i M L }$ </td><td> $3 . 2 5 \pm 0 . 8 7$ </td><td> $9 1 . 8 \pm 1 . 2$ </td><td> $0 . 9 3 4 \pm 0 . 0 0 4$ </td></tr><tr><td> $+ \mathrm { F O C I }$ </td><td> $4 . 3 5 \pm 1 . 1 8 ( + 1 . 1 0 )$ </td><td> $9 4 . 5 \pm 1 . 0 ~ ( + 2 . 7 ) $ </td><td> $\overline { { 0 . 9 2 6 \pm 0 . 0 0 1 } } \ : ( - . 0 0 8 )$ </td></tr><tr><td> $\mathbf { A C M L }$ </td><td> $4 . 7 8 \pm 0 . 7 5$ </td><td> ${ \bf 9 7 . 2 \pm 0 . 1 }$ </td><td> $0 . 9 3 0 \pm 0 . 0 0 5$ </td></tr><tr><td>+FOCI</td><td> $\mathbf { 3 . 0 9 \pm 0 . 5 0 \ } _ { ( - 1 . 6 9 ) }$ </td><td> $9 5 . 0 \pm 1 . 5 \ : ( - 2 . 2 )$ </td><td> $0 . 9 3 3 \pm 0 . 0 0 9 \ : ( + . 0 0 3 )$ </td></tr><tr><td colspan="4">Hard-selection backbones</td></tr><tr><td>ASMIL</td><td> $1 1 . 2 0 \pm 0 . 4 4$ </td><td> $8 9 . 6 \pm 2 . 7$ </td><td> $0 . 9 0 3 \pm 0 . 0 1 0$ </td></tr><tr><td>+FOCI</td><td> $2 7 . 3 5 \pm 1 7 . 2 5 ( + 1 6 . 1 5 )$ </td><td> $8 8 . 3 \pm 5 . 8 ( - 1 . 3 )$ </td><td> $0 . 8 1 7 \pm 0 . 0 8 1 \ ( \cdot . 0 8 6 )$ </td></tr><tr><td>MHIM-MIL</td><td> $5 . 3 0 \pm 0 . 9 5$ </td><td> $9 1 . 6 \pm 1 . 1$ </td><td> $0 . 9 3 0 \pm 0 . 0 0 7$ </td></tr><tr><td>+FOCI</td><td> $6 . 2 5 \pm 1 . 2 4 ( + 0 . 9 5 )$ </td><td> $9 2 . 0 \pm 1 . 2 \ ( + 0 . 4 ) $ </td><td> $0 . 9 2 7 \pm 0 . 0 0 7 \ : ( - . 0 0 3 )$ </td></tr></table>

## H.1 Brief per-dataset interpretation

NSCLC. NSCLC shows the clearest positive-headroom pattern among soft-aggregation backbones. FOCI reduces MSK for all five soft-aggregation backbones, including TransMIL (7.33 → 3.21), ABMIL (2.65 → 1.52), and ACMIL (6.08 → 1.79). Hard-selection backbones behave differently: ASMIL has the best native MSK and AUKC on NSCLC, but attaching an external FOCI selector inflates its sufficient set, consistent with an architectural conflict mode.

BRCA. BRCA exposes the selection-saturation regime most clearly for attention-pooling backbones. ABMIL and CLAM-SB already reach near-single-tile MSK (1.10 and 1.17), which leaves little room for an external selector; ABMIL+FOCI and CLAM-SB+FOCI therefore inflate MSK despite small AUKC changes. In contrast, TransMIL and ACMIL retain headroom: FOCI reduces TransMIL MSK from 5.65 to 3.86, and ACMIL+FOCI achieves the highest BRCA AUKC (0.880) with reduced MSK.

PANDA. PANDA has higher baseline MSK for several backbones, consistent with a more distributed prediction footprint under the binarized prostate grading task. FOCI reduces TransMIL MSK from 16.5 to 10.62 and ACMIL MSK from 4.78 to 3.09. However, ABMIL+FOCI and AttriMIL+FOCI are less stable, and ASMIL+FOCI strongly degrades, which again shows that FOCI is a readout probe whose usefulness depends on backbone selection headroom rather than a universal improvement module.

Takeaway. Across datasets, the appendix tables support the main SHI result: compact post-hoc rationales are available when the frozen backbone has selection headroom, the regime saturates when the native ranking is already near-minimal, and the readout can conflict with hard-selection backbones. These tables provide the raw per-dataset breakdown for the family-wise summary in Table 1.

## I Deletion-based perturbation details

Deletion-based perturbation asks whether top-ranked tiles are load-bearing when removed from the input. This is complementary to SRP insertion: deletion measures removal impact, whereas SRP measures confidence recovery as ranked tiles are inserted. Negative values indicate that deleting the ranked tiles increases the true-class probability on average, which usually reflects saturation or noisy ranking under that perturbation protocol.

Table 15: Deletion-based perturbation faithfulness on TCGA-NSCLC (3-seed mean±std). Faithfulness AUC summarizes the drop in true-class probability when the top-K ranked tiles are deleted, averaged over $K \in \{ 1 6 , 3 2 , 6 4 , \bar { 1 } 2 8 , 2 5 6 \}$ [41]. Higher values indicate a more load-bearing ranking.
<table><tr><td>Method</td><td>Faith. AUC ↑</td><td> $\Delta p _ { y } @ \mathrm { K } = 1 6$ </td><td> $\scriptstyle ( a ) \mathrm { K } = 3 2$ </td><td> $\scriptstyle ( { \vec { \omega } } \operatorname K = 6 4$ </td><td> $\scriptstyle ( \omega \mathrm { K } = 1 2 8$ </td><td> $\scriptstyle { \circledcirc } \mathrm { K } = 2 5 6$ </td></tr><tr><td>TransMIL</td><td> $0 . 0 0 0 3 \pm 0 . 0 0 0 7$ </td><td> $- 0 . 0 0 0$ </td><td>-0.000</td><td>0.000</td><td>0.000</td><td>0.001</td></tr><tr><td>ASMIL</td><td> $0 . 0 0 8 7 \pm 0 . 0 0 1 3$ </td><td>0.002</td><td>0.003</td><td>0.006</td><td>0.010</td><td>0.015</td></tr><tr><td>FOCI (ours, on TransMIL)</td><td> $0 . 0 2 7 4 \pm 0 . 0 2 1 3$ </td><td>0.008</td><td>0.013</td><td>0.020</td><td>0.029</td><td>0.046</td></tr><tr><td>AttriMIL</td><td> $0 . 0 4 4 4 \pm 0 . 0 0 6 9$ </td><td>0.004</td><td>0.014</td><td>0.026</td><td>0.046</td><td>0.084</td></tr><tr><td>MHIM-MIL</td><td> $0 . 0 5 3 7 \pm 0 . 0 0 7 2$ </td><td>0.016</td><td>0.023</td><td>0.036</td><td>0.058</td><td>0.090</td></tr><tr><td>CLAM-SB</td><td> $0 . 0 5 5 1 \pm 0 . 0 1 1 6$ </td><td>0.010</td><td>0.025</td><td>0.039</td><td>0.061</td><td>0.089</td></tr><tr><td>ABMIL</td><td> $\mathbf { 0 . 0 7 3 6 \pm 0 . 0 1 4 2 }$ </td><td>0.023</td><td>0.035</td><td>0.057</td><td>0.080</td><td>0.116</td></tr></table>

Table 16: Cross-dataset deletion-based faithfulness for the three methods with comparable per-tile scoring (3-seed mean±std, normalized by $K _ { \mathrm { m a x } } { = } 2 5 6$ to match Table 15). Higher values indicate a more load-bearing ranking; negative values indicate that deleting the ranked tiles increases the true-class probability on average. Absolute scale differs across datasets because of saturation, so comparisons are within-dataset. Bold: best within each dataset.
<table><tr><td>Method</td><td>NSCLC</td><td>BRCA</td><td>PANDA</td></tr><tr><td>TransMIL</td><td> $0 . 0 0 0 3 \pm 0 . 0 0 0 7$ </td><td> $0 . 0 0 1 0 \pm 0 . 0 0 1 0$ </td><td> $0 . 0 4 9 1 \pm 0 . 0 1 2 6$ </td></tr><tr><td>ASMIL</td><td> $0 . 0 0 8 7 \pm 0 . 0 0 1 3$ </td><td> $- 0 . 0 0 5 4 \pm 0 . 0 0 1 3$ </td><td> $\mathbf { 0 . 1 6 8 1 \pm 0 . 0 1 6 8 }$ </td></tr><tr><td>FOCI (ours, on TransMIL)</td><td> $\mathbf { 0 . 0 2 7 4 \pm 0 . 0 2 1 3 }$ </td><td> $\mathbf { 0 . 0 0 9 7 \pm 0 . 0 0 8 9 }$ </td><td> $0 . 0 7 4 7 \pm 0 . 0 1 3 1$ </td></tr></table>

## J FOCI-STE: full technical details

Hard top-K with straight-through (FOCI-STE). FOCI-Soft uses continuous Concrete gates during training, whereas SRP evaluates hard ranked subsets at test time. FOCI-STE reduces this soft-vs-hard cardinality mismatch by enforcing an exactly K-sparse binary mask in the forward pass while routing gradients through a sigmoid surrogate [36]. Given selector logits as i $a _ { s , i }$ for slide $s ,$ we define

$$
m _ { s , i } = \mathbf { 1 } [ a _ { s , i } \in \mathrm { t o p } { - } K ( a _ { s } ) ] , \qquad \sum _ { i } m _ { s , i } = K ,\tag{13}
$$

and use the straight-through gate

$$
\tilde { m } _ { s , i } = m _ { s , i } + \sigma ( a _ { s , i } ) - \mathrm { s t o p g r a d } ( \sigma ( a _ { s , i } ) ) .\tag{14}
$$

The forward value of $\tilde { m } _ { s , i }$ equals the binary mask $m _ { s , i } ,$ , while the backward gradient follows the sigmoid surrogate,

$$
\frac { \partial \tilde { m } _ { s , i } } { \partial a _ { s , i } } = \sigma ^ { \prime } ( a _ { s , i } ) .\tag{15}
$$

Thus, exactly K tiles are selected in every forward pass, but the selector logits still receive dense surrogate gradients during optimization.

During training, the auxiliary keep/drop views are realized through multiplicative straight-through gating. During SRP evaluation, unrevealed tokens are excluded through the model’s masking interface (key\_padding\_mask). These two operations are not pointwise identical for every MIL pooling architecture, but they impose the same hard cardinality constraint and use the same tile ranking. This is the relevant alignment for our audit setting: the selector is trained to produce a compact ordered subset, and SRP evaluates the resulting order under hard reveal.

Although the hard top-K operator fixes the forward-pass cardinality, we retain a small per-bag budget regularizer,

$$
\mathcal { L } _ { \mathrm { b u d g e t } } = \sum _ { i } \tilde { m } _ { s , i } , \qquad \lambda _ { \mathrm { b u d g e t } } = 5 \times 1 0 ^ { - 3 } .\tag{16}
$$

Its forward value is constant at $K ,$ but its backward pass provides a small stabilizing gradient to the underlying selector scores through $\sigma ( a _ { s } )$ . This term is therefore not used to enforce sparsity in FOCI-STE (top-K already does that), but to regularize the score scale around the hard selection boundary. The FOCI-Soft counterpart uses continuous gates $z _ { s }$ and is described in §3.4.

FOCI-STE is one parameterization of the same frozen-backbone audit framework as FOCI-Soft. The central object of study is not the gate parameterization itself, but whether a frozen WSI-MIL classifier exhibits selection headroom under a consistent tile ranking.

## K Loss term details

This appendix clarifies three implementation details that are abbreviated in the main method: the budget regularizer, the FOCI-Soft entropy term, and the shorthand “sufficiency objective.”

Budget regularizer. For FOCI-Soft, the budget term is applied to the continuous Gumbel-sigmoid gates:

$$
{ \mathcal { L } } _ { \mathrm { b u d g e t } } = \sum _ { i } z _ { s , i } ,
$$

where $z _ { s , i } \in ( 0 , 1 )$ is the soft gate for tile i in slide s. This term discourages diffuse high-mass gates and encourages the selector to use a compact subset.

For FOCI-STE, the hard top-K forward mask already fixes the selected cardinality exactly, $\begin{array} { r } { \sum _ { i } m _ { s , i } = K } \end{array}$ . We therefore apply the budget term to the straight-through gate $\tilde { m } _ { s }$ defined in Appendix J:

$$
\mathcal { L } _ { \mathrm { b u d g e t } } = \sum _ { i } \tilde { m } _ { s , i } .
$$

Its forward value is constant at $K ,$ , but its backward pass provides a small stabilizing gradient through the sigmoid surrogate. Thus, in FOCI-STE, $\mathcal { L } _ { \mathrm { b u d g e t } }$ regularizes score scale near the rank-K boundary rather than enforcing sparsity. We use $\lambda _ { \mathrm { b u d g e t } } = 5 \times 1 0 ^ { - 3 }$ in both FOCI-Soft and FOCI-STE.

Sufficiency objective shorthand. We use “sufficiency objective” as shorthand for the keep-bag terms $\mathcal { L } _ { \mathrm { s u f f } }$ and $\mathcal { L } _ { \mathrm { h i n g e } }$ . Both are computed on the keep bag, but they enter the total objective with different weights: $\mathcal { L } _ { \mathrm { s u f f } }$ encourages recovery of the target-class output, while $\dot { \mathcal { L } } _ { \mathrm { h i n g e } }$ enforces the operating-confidence margin used during selector training.

FOCI-Soft entropy term. FOCI-Soft uses continuous gates and therefore does not impose an exact tile count during training. To prevent diffuse fractional masks, we add an entropy penalty

$$
\lambda _ { \mathrm { e n t } } \mathcal { H } ( z _ { s } ) , \qquad \mathcal { H } ( z _ { s } ) = - \frac { 1 } { N _ { s } } \sum _ { i } \left[ z _ { s , i } \log z _ { s , i } + ( 1 - z _ { s , i } ) \log ( 1 - z _ { s , i } ) \right] ,
$$

which pushes the continuous gates toward binary values. This entropy term is used only for FOCI-Soft;   
FOCI-STE obtains exact hard cardinality through the top-K forward mask.

## L Full experimental setup details

## L.1 Datasets and features

We evaluate on three public WSI benchmarks (TCGA-NSCLC, TCGA-BRCA, and PANDA), all with slide-level labels and no patch-level annotation.

NSCLC. The non-small-cell lung cancer cohort comprises 1,043 slides (729 train / 105 validation / 209 test).   
Each slide is labeled LUAD (lung adenocarcinoma) or LUSC (lung squamous cell carcinoma).

BRCA. The breast cancer cohort contains 1,126 slides (724 train / 179 validation / 223 test), labeled as invasive ductal carcinoma (IDC) versus other subtypes. The class distribution is skewed (≈ 73% IDC), and the boundary between IDC and the rarer subtypes is histologically subtle.

PANDA. The PANDA prostate grading dataset [13] has 10,615 slides (6,793 train / 1,699 validation / 2,123 test). Labels are binarized as benign (ISUP grade 0) versus malignant (ISUP grade ≥ 1).

Features. Slides are tiled at 20× magnification into 256×256 patches. We extract d=1536-dimensional features using frozen UNI2-h [4], a vision transformer pretrained on a large-scale histology corpus. Features are extracted once and stored as HDF5 files, and the encoder is never updated. For both FOCI training and SRP evaluation, slides with more than $n _ { \mathrm { c a p } } { = } 1 0 2 4$ patches are pre-filtered to the top $n _ { \mathrm { c a p } }$ tokens by feature L2 norm before MIL aggregation; see Appendix E.2 for sensitivity.

## L.2 Implementation details

Backbone. The primary FOCI-STE backbone is a four-layer TransMIL [15] with $d _ { \mathrm { m o d e l } } { = } 5 1 2$ , eight attention heads, and a learned [CLS] token, pretrained for 20 epochs with cross-entropy on the full bag. For cross-backbone experiments, FOCI is additionally applied post-hoc to ABMIL (d=512), CLAM-SB (d=256), AttriMIL (d=512), ACMIL (d=512), ASMIL (d=256), and MHIM-MIL (d=512), each pretrained independently with its original objective. In all cases, the encoder and MIL backbone are fully frozen during FOCI training.

FOCI-STE training. The selection module is a two-layer MLP $( d { = } 5 1 2  2 5 6  1 )$ with 132,609 parameters, which is under 1% of the primary TransMIL pipeline. We train the selector for 30 epochs with a 5-epoch linear warmup using AdamW with cosine annealing from $1 0 ^ { - 4 } ~ \mathrm { t o } ~ 1 0 ^ { - 5 } ;$ the selector uses a $5 \times$ learning-rate multiplier relative to the base schedule and an AdamW weight decay of 0.3. The frozen encoder and backbone are not optimized. Batch size is 2, and slides are padded to $n _ { \mathrm { c a p } }$ tokens. FOCI-STE selects exactly K=32 tiles per slide in the forward pass.

The sufficiency cross-entropy and exclusion losses are weighted equally $( \lambda _ { \mathrm { s u f f } } = \lambda _ { \mathrm { e x c l } } = 0 . 5 ) .$ , with the keepbag confidence hinge added at $\lambda _ { \mathrm { h i n g e } } { = } 1 . 0$ and a light spatial compactness term $( \lambda _ { \mathrm { c o n t i g } } { = } 0 . 0 1 )$ ) to encourage contiguous selections without dominating the rationale losses. The training sufficiency target is set to $\tau { = } 0 . 9 .$ numerically matching the SRP operating threshold $\kappa { = } 0 . 9$ used in the main evaluation; the drop-bag tolerance in the exclusion loss $( \ S \bar { \ S } . \ 4 )$ is set to β=0.2. For FOCI-Soft, we add budget and entropy penalties $( \mathrm { \hat { \lambda } _ { b u d g e t } } = 5 \times 1 0 ^ { - 3 }$ $\lambda _ { \mathrm { e n t } } { = } 0 . 1 )$ to push the continuous gates toward binary values.

Unless otherwise stated, FOCI-STE is trained with the fixed K=32 budget above. The adaptive-K schedule $K _ { s } = \operatorname* { m a x } ( 1 6 , \lfloor 0 . 0 3 N _ { s } ^ { \mathrm { r e a l } } \rfloor )$ is evaluated separately in §G.3 for selected-only downstream analysis and in Appendix E.3 for budget-sensitivity analysis. All reported results are averaged over three seeds and evaluated at $\kappa { = } 0 . 9$

## L.3 Baselines

We compare seven MIL methods, all trained with the same frozen UNI2-h features. TransMIL [15] is the primary frozen backbone without rationale selection. ABMIL [2] uses scalar attention weights. CLAM-SB [3] adds instance-level discrimination via an auxiliary loss. AttriMIL [18] decomposes attention across attribute heads. ACMIL [19] uses multiple attention branches with masked patch training to capture complementary diagnostic regions. ASMIL [37] trains with top-K attention sampling. MHIM-MIL [17] uses masked hard instance mining.

Each baseline is re-evaluated under SRP using its own native per-tile ranking score, summarized in Table 3 (Appendix C).

## M Limitations and future work

Limitations. SRP measures model-output sufficiency, not annotation-validated clinical sufficiency. The selected tiles are candidate rationales for a frozen classifier and do not establish pathologist-level diagnostic sufficiency or clinical utility. Our main experiments use UNI2-h features; because the current FOCI-STE pipeline has not been evaluated across a broad set of pathology encoders, we do not claim universal encoder agnosticism. Extending FOCI to ground-truth tumor annotations (e.g., CAMELYON16/17), broader encoder benchmarks, external clinical cohorts, and multi-reader pathologist studies would be needed to argue clinical relevance beyond model-sufficient rationale highlighting; we leave these directions to future work.

## N Predicted-class SRP variant (audit-time view)

The main SRP analysis tracks $p _ { y } ( K )$ against the ground-truth label $_ { y , }$ which jointly assesses confidence recovery and correctness on labeled test sets. In an audit-time setting, however, explanations are often requested for the model’s own predicted class ${ \hat { y } } = \operatorname { a r g }$ max<sub>c</sub> $f _ { c } ( X )$ . We therefore report a predicted-class SRP variant using the same insertion-style reveal protocol, but with K-curves tracking $p _ { \hat { y } } ( K )$

For the binary classification tasks studied here, this variant can be recovered from the stored K-curves without retraining or re-evaluating any model: for slides where $\hat { y } = y , p _ { \hat { y } } ( K ) = p _ { y } ( K )$ , and for slides where $\hat { y } \neq y ,$ $p _ { \hat { y } } ( K ) = 1 - p _ { y } ( K )$ . We report this audit-time view for the TransMIL baseline and TransMIL+FOCI on all three datasets.

Table 17: Predicted-class SRP variant on TransMIL baseline vs. TransMIL+FOCI (3-seed mean±std at $\kappa { = } 0 . 9 )$ . Reach, $\mathrm { M S K } _ { \mathrm { c o n d } }$ , and AUKC are computed against the model’s own predicted class yˆ rather than the ground-truth label $y .$ . The qualitative pattern matches ground-truth SRP: FOCI compresses MSK on all three datasets, which supports the interpretation that the readout recovers the frozen model’s own decision rather than exploiting label-specific evaluation.
<table><tr><td>Dataset</td><td>Method</td><td> $\operatorname { R e a c h } _ { \hat { y } }$ </td><td> $\mathbf { M S K } _ { \mathrm { c o n d } } ^ { \hat { y } } \downarrow$ </td><td>AUKC ↑</td></tr><tr><td rowspan="2">NSCLC</td><td>TransMIL baseline</td><td>0.984</td><td> $7 . 5 5 \pm 0 . 7 7$ </td><td> $0 . 9 5 6 \pm 0 . 0 0 5$ </td></tr><tr><td>TransMIL+FOCI</td><td>0.951</td><td> ${ \bf 3 . 5 2 \pm 0 . 5 4 }$ </td><td> $0 . 9 4 6 \pm 0 . 0 3 4$ </td></tr><tr><td rowspan="2">BRCA</td><td>TransMIL baseline</td><td>0.934</td><td> $6 . 1 1 \pm 3 . 4 8$ </td><td> $0 . 9 3 2 \pm 0 . 0 2 0$ </td></tr><tr><td>TransMIL+FOCI</td><td>0.928</td><td> ${ \bf 4 . 1 7 \pm 1 . 2 3 }$ </td><td> $0 . 9 3 6 \pm 0 . 0 2 0$ </td></tr><tr><td>PANDA</td><td>TransMIL baseline TransMIL+FOCI</td><td>0.966 0.940</td><td> $1 6 . 8 2 \pm 1 . 3 2$   ${ \bf 1 0 . 8 1 \pm 3 . 3 7 }$ </td><td> $0 . 9 1 4 \pm 0 . 0 0 6$   $0 . 9 3 1 \pm 0 . 0 1 5$ </td></tr></table>

Predicted-class SRP changes both the reachable set and the target probability being tracked, so its MSK is not expected to match ground-truth SRP exactly. In our binary tasks, the qualitative compression pattern remains the same: TransMIL+FOCI reduces predicted-class MSK on all three datasets. Reach is generally higher under predicted-class SRP because correctness is no longer required. We report predicted-class SRP as a complementary audit view; ground-truth SRP remains the appropriate benchmark for rationale quality on labelled test sets. For binary classification, this analysis also characterizes recovery of the model’s own decision on incorrectly classified slides. Multiclass extensions require tracking $p _ { \hat { y } } ( K )$ directly and are not addressed in this paper.