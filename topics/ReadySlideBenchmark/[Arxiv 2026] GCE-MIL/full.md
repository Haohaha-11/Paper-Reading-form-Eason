# GCE-MIL: Faithful and Recoverable Evidence for Multiple Instance Learning in Whole-Slide Imaging

Xiangyu Li College of Intelligence and Computing Tianjin University xiangyuli@tju.edu.cn

Ran Su<sup>∗</sup> College of Intelligence and Computing Tianjin University ran.su@tju.edu.cn

## Abstract

Multiple instance learning (MIL) is the standard approach for whole-slide image (WSI) classification and survival prediction, where attention-based models aggregate patch features into slide-level predictions. These models treat attention weights as evidence for their predictions, but attention is optimized for classification, not for identifying which patches actually support the diagnosis. This conflation leads to three failures: selected patches are insufficient (keeping them alone drops Macro-F1 by 0.078), unnecessary (removing them barely changes the prediction), and unrecoverable (continuous attention scores disagree with discrete patch subsets used at inference). The central premise is that evidence quality should be optimized directly through explicit criteria—Sufficiency, Necessity, and Recoverability (S/N/R)—rather than inherited as a byproduct of classification. GCE-MIL is a backbone-agnostic wrapper implemented through three injection modes and three evidence components: a grounding mechanism that aligns selection with domain-specific concepts, noisy-OR coverage that acts as a differentiable proxy for interventional evidence search, and threshold-plus-repair recovery that converts continuous selectors into discrete subsets through marginal-guided repair. Across 9 backbones × 9 datasets (81 configurations), GCE-MIL improves average Macro-F1 by 0.024 and C-index by 0.014, reduces the continuous-discrete gap by 4–7×, and increases complement degradation by 2–4×. With optional tile prefiltering after discrete recovery, inference runs up to 5× faster while retaining 0.989× full-bag utility.

## 1 Introduction

Whole-slide images contain thousands of patches, diagnostically relevant tissue is sparse, and training supervision is usually available only at the slide level. Multiple instance learning (MIL) is therefore the standard modeling choice: a slide is treated as a bag of patches, and a bag-level predictor is trained from slide labels [Ilse et al., 2018, Lu et al., 2021, Shao et al., 2021]. The same models are often asked to provide evidence by visualizing attention weights or instance scores. This reuse is convenient, but it leaves a clinical question unresolved: which patches actually support the prediction under intervention?

Current WSI MIL methods mainly improve the predictor. ABMIL introduces gated attention pooling [Ilse et al., 2018]; CLAM adds clustering-constrained attention [Lu et al., 2021]; TransMIL models inter-instance correlation with transformers [Shao et al., 2021]; DSMIL, DTFD-MIL, IBMIL, MHIM-MIL, CAMIL, and HDMIL improve aggregation, regularization, context, or efficiency [Li et al., 2021, Zhang et al., 2022, Lin et al., 2023, Tang et al., 2023, Fourkioti et al., 2024, Dong et al., 2025].

![](images/bddb2fd68782eb3f50ebc57f6ad1edaf327b8d3cc03165c23a43efd73e5a2bd1.jpg)

![](images/cdfddf36dbc09ce68b675a469519b5b0689373a5b1ffe053f0282fb0295dea41.jpg)

Table 1: Minimal sufficient subset analysis on BRACS validation slides using an ABMIL teacher. A subset is sufficient if it preserves the full-bag predicted class and has probability drop at most 0.05.
<table><tr><td>k</td><td>Policy</td><td>Subsets/slide↑</td><td> $\mathrm { S l i d e s } \geq 2 ( \% ) \uparrow$ </td><td> $\mathrm { S l i d e s } \geq 3 ( \% ) \uparrow$ </td><td>Keep-only drop↓</td><td>Remove-union-1↑</td><td>Remove-union-2↑</td><td>Remove-union-3↑</td></tr><tr><td>8</td><td>Sufficient-prefix</td><td> $2 . 2 3 0 2 { \scriptstyle \pm 0 . 2 6 3 5 }$ </td><td> $7 2 . 6 7 { \pm } 1 3 . 2 8$ </td><td> $5 0 . 3 5 { \pm } 1 3 . 1 6$ </td><td> $\overline { { 0 . 0 1 8 3 \pm 0 . 0 0 1 9 } }$ </td><td>0.0806±0.0361</td><td>0.0685±0.0200</td><td>0.0588±0.0102</td></tr><tr><td></td><td>Attention top-k</td><td> $2 . 1 0 3 1 { \scriptstyle \pm 0 . 2 1 1 3 }$ </td><td> $7 0 . 8 9 { \pm 9 . 3 0 }$ </td><td> $3 9 . 4 3 { \pm } 1 2 . 0 8$ </td><td> $0 . 0 7 6 6 { \scriptstyle \pm 0 . 0 1 8 0 }$ </td><td> $0 . 1 1 3 4 { \scriptstyle \pm 0 . 0 2 9 4 }$ </td><td> $0 . 1 1 6 3 { \scriptstyle \pm 0 . 0 2 5 6 }$ </td><td>0.1252±0.0270</td></tr><tr><td>88</td><td>Random top-k</td><td> $0 . 0 6 8 8 { \scriptstyle \pm 0 . 0 4 1 1 }$ </td><td> $0 . 3 8 { \pm } 0 . 4 7$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 1 7 6 3 { \scriptstyle \pm 0 . 0 1 9 8 }$ </td><td> $0 . 0 0 0 0 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td><td> $0 . 0 0 0 0 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td><td>0.0000±0.0001</td></tr><tr><td>16</td><td>Sufficient-prefix</td><td> $\overline { { 1 . 7 2 9 5 { \pm } 0 . 2 1 8 8 } }$ </td><td> $\overline { { 5 1 . 0 7 { \pm } 1 0 . 6 6 } }$ </td><td> $2 1 . 8 8 { \pm } 1 1 . 4 2 $ </td><td> $\overline { { 0 . 0 1 6 2 { \pm } 0 . 0 0 2 5 } }$ </td><td> $\overline { { 0 . 1 0 8 6 { \pm } 0 . 0 3 4 3 } }$ </td><td> $\overline { { 0 . 1 0 1 4 \pm 0 . 0 3 2 3 } }$ </td><td> $\overline { { 0 . 0 8 6 2 { \pm } 0 . 0 2 0 0 } }$ </td></tr><tr><td>16</td><td>Attention top-k</td><td> $1 . 9 2 5 1 { \scriptstyle \pm 0 . 1 4 7 3 }$ </td><td> $5 9 . 8 9 { \pm } 5 . 9 4 $ </td><td> $3 2 . 6 2 { \pm } 8 . 9 8$ </td><td> $0 . 0 5 9 5 { \scriptstyle \pm 0 . 0 1 9 9 }$ </td><td> $0 . 0 8 2 3 { \scriptstyle \pm 0 . 0 2 9 4 }$ </td><td> $0 . 0 9 6 3 { \scriptstyle \pm 0 . 0 3 0 9 }$ </td><td>0.1002±0.0280</td></tr><tr><td>16</td><td>Random top-k</td><td> $0 . 1 0 5 4 { \pm } 0 . 0 4 5 9$ </td><td> $1 . 3 4 \pm 0 . 9 9$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 1 7 2 8 { \scriptstyle \pm 0 . 0 3 4 5 }$ </td><td> $0 . 0 0 0 1 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td><td> $0 . 0 0 0 3 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $0 . 0 0 0 3 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td></tr><tr><td>32</td><td>Sufficient-prefix</td><td> $\overline { { 1 . 3 6 8 2 { \pm } 0 . 0 6 7 5 } }$ </td><td> $2 9 . 0 0 { \scriptstyle \pm 5 . 3 4 }$ </td><td> $7 . 8 3 \pm 1 . 5 4$ </td><td> $\overline { { 0 . 0 1 1 3 { \pm } 0 . 0 0 1 6 } }$ </td><td> $\overline { { 0 . 1 0 8 5 { \scriptstyle \pm 0 . 0 2 6 1 } } }$ </td><td> $\overline { { 0 . 0 7 7 9 { \scriptstyle \pm 0 . 0 0 9 4 } } }$ </td><td> $\overline { { 0 . 0 6 8 7 { \scriptstyle \pm 0 . 0 2 0 3 } } }$ </td></tr><tr><td>32</td><td>Attention top-k</td><td> $1 . 2 7 8 5 { \scriptstyle \pm 0 . 0 5 3 6 }$ </td><td> $2 2 . 7 0 { \scriptstyle \pm 4 . 2 7 }$ </td><td> $5 . 1 5 { \pm } 1 . 7 5 $ </td><td> $0 . 0 8 7 7 { \scriptstyle \pm 0 . 0 2 2 5 }$ </td><td> $0 . 1 2 6 0 { \scriptstyle \pm 0 . 0 3 4 3 }$ </td><td> $0 . 1 3 7 1 { \scriptstyle \pm 0 . 0 3 3 7 }$ </td><td> $0 . 1 3 5 8 { \scriptstyle \pm 0 . 0 3 1 8 }$ </td></tr><tr><td>32</td><td>Random top-k</td><td> $0 . 2 1 5 6 { \scriptstyle \pm 0 . 0 1 7 3 }$ </td><td> $2 . 8 6 \pm 1 . 2 0$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 1 3 3 5 { \scriptstyle \pm 0 . 0 0 9 3 }$ </td><td> $0 . 0 0 0 1 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $0 . 0 0 0 1 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $0 . 0 0 0 1 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td></tr></table>

![](images/abb3b0d6956d7f2100da3d6ae642388321fabc85d85bb6938ffbaea625a6b335.jpg)  
Figure 1: Three evidence failures in classification-optimized MIL. (Left) Attention top-k achieves only 0.640 keep-only Macro-F1 vs. GCE 0.722: attention is not sufficient evidence. (Middle) The continuous selector becomes bimodal during training, enabling discretization with C-D gap 0.004. (Right) Adding GCE preserves 0.99× full-bag performance across backbones.

Attention-regularized variants such as ACMIL, AEM, and ASMIL stabilize or deconcentrate attention maps [Zhang et al., 2024, 2025, Ye et al., 2026]. All these methods, however, share an implicit assumption that classification accuracy is the optimization target and that attention or score rankings are an interpretable byproduct. This conflates two distinct goals: predicting correctly and learning correct evidence.

The failure is not simply that attention chooses the wrong number of patches; in the BRACS diagnostic, WSI evidence is often structurally non-unique. Table 1 reports a recursive BRACS diagnostic using an ABMIL teacher. For $k = 8 ,$ , sufficient-prefix search finds 2.2302 disjoint sufficient subsets per slide on average, and 72.67% of validation slides admit at least two such subsets. Attention top-k also finds multiple candidate subsets, but its keep-only drop remains higher (0.0766 at $k = 8 )$ while random top-k almost never finds reusable evidence. Multiple tissue regions can therefore each preserve the same slide-level decision, while a single softmax ranking collapses them into one list. This BRACS diagnostic motivates the S/N/R framework, which is then evaluated across the full nine-dataset benchmark. Figure 1 summarizes the resulting S/N/R tension.

Evidence quality is formalized through three model-relative criteria. Sufficiency asks whether the selected subset alone preserves the prediction; Necessity asks whether removing the subset degrades the prediction; Recoverability asks whether the continuous selector learned during training yields a faithful discrete subset at inference. GCE-MIL targets these criteria directly with a semantic anchor bank, a continuous selector trained through exact noisy-OR coverage, and threshold-plus-greedy recovery. It is a wrapper rather than a replacement backbone: the host MIL architecture remains unchanged, and the evidence gate is injected as attention-logit bias, feature reweighting, or a hybrid according to the host aggregation type.

The paper makes four contributions:

• A formalization of MIL evidence quality through Sufficiency, Necessity, and Recoverability, with supporting theoretical notes on independence, recoverability, and coverage in Appendix B.

• A BRACS minimal-subset diagnostic showing evidence non-uniqueness in validation slides, motivating evidence selection beyond a single attention ranking.

• GCE-MIL, a plug-in wrapper that combines semantic grounding, noisy-OR coverage, and discrete recovery through attention-bias, feature-reweighting, and hybrid injection modes.

• An evaluation across 9 backbones and 9 datasets, covering prediction metrics, intervention diagnostics, localization, stability, ablations, and computational cost.

## 2 Related Work

MIL research for WSI can be read as a sequence of improvements to bag prediction. Pooling design is addressed by ABMIL, CLAM, DSMIL, and TransMIL [Ilse et al., 2018, Lu et al., 2021, Li et al., 2021, Shao et al., 2021]; attention concentration is mitigated by ACMIL, AEM, and ASMIL [Zhang et al., 2024, 2025, Ye et al., 2026]; overfitting is reduced by DTFD-MIL and MHIM-MIL [Zhang et al., 2022, Tang et al., 2023]; spatial context and efficiency are modeled by CAMIL and HDMIL [Fourkioti et al., 2024, Dong et al., 2025]. These works are useful host architectures for GCE-MIL. Their optimization target remains slide-level prediction, however, while evidence sufficiency, necessity, and recoverability are usually evaluated only after training. GCE-MIL is therefore orthogonal: it plugs into these backbones and adds evidence objectives rather than competing as another pooling module.

Sparse selection and post-hoc attribution address explanation more directly but still leave S/N/R under-specified. $L _ { 0 } .$ , Concrete, and Gumbel relaxations provide differentiable gates [Louizos et al., 2018, Maddison et al., 2017, Jang et al., 2017], yet they do not ground selected patches in pathology concepts or model multi-source diagnostic coverage. Gradient saliency, integrated gradients, and occlusion provide post-hoc scores [Simonyan et al., 2013, Sundararajan et al., 2017, Zeiler and Fergus, 2014], but the predictor is already fixed and the thresholded subset is not optimized to be sufficient or necessary. GCE-MIL differs by training selection and prediction jointly, grounding gates with TITAN text anchors, and evaluating the recovered subset through explicit S/N/R interventions.

Subset and concept explanation methods provide useful context for this formulation. L2X and INVASE learn instance- or feature-level rationales [Chen et al., 2018, Yoon et al., 2018], perturbation methods such as Meaningful Perturbations and RISE score regions through input interventions [Fong and Vedaldi, 2017, Petsiuk et al., 2018], and ERASER popularizes sufficiency/necessity-style rationale evaluation in NLP [DeYoung et al., 2020]. Concept methods such as SENN, ProtoPNet, concept bottleneck models, and TCAV connect predictions to human-readable concepts [Alvarez Melis and Jaakkola, 2018, Chen et al., 2019, Koh et al., 2020, Kim et al., 2018]. GCE-MIL draws on these ideas but targets the WSI MIL setting: the object being recovered is a slide-level, sparse patch subset whose continuous selector and discrete evidence are evaluated under the same bag predictor.

## 3 Preliminaries and Motivation

## 3.1 Notation and MIL Formulation

In MIL for computational pathology, supervision is provided only at the slide level. A whole-slide image is represented as a bag $X \stackrel { \triangledown } { = } \{ x _ { i } \} _ { i = 1 } ^ { N }$ of tissue patches, where N can range from hundreds to tens of thousands. A pretrained encoder maps each patch $x _ { i }$ to a fixed-dimensional embedding $h _ { i } \in \mathbb { R } ^ { d } \left( d = 1 0 2 4 \right.$ throughout). An attention-based MIL model assigns a scalar attention score to

each embedding via a learnable scorer:

$$
z _ { i } = f _ { \theta } ( h _ { i } ) , \qquad \alpha _ { i } = \frac { \exp ( z _ { i } ) } { \sum _ { j = 1 } ^ { N } \exp ( z _ { j } ) } ,\tag{1}
$$

where $\alpha _ { i }$ lies on the probability simplex $\Delta ^ { N }$ . The slide-level representation $\begin{array} { r } { h _ { \mathrm { b a g } } = \sum _ { i = 1 } ^ { N } \alpha _ { i } h _ { i } } \end{array}$ <sub>i</sub> is a convex combination of instance features weighted by the attention distribution, and is passed to a classifier to produce the bag-level prediction $\bar { \hat { y } } = f ( \bar { h _ { \mathrm { b a g } } } )$

After training, the attention weights $\left\{ \alpha _ { i } \right\}$ are reused as evidence: the top-ranked patches are presented as the model’s explanation for its prediction. However, this reuse conflates two distinct objectives— classification accuracy and evidence quality—because the attention mechanism is optimized solely for the former.

## 3.2 Motivation: Three Evidence Failures

Three systematic failures arise when attention is treated as evidence, motivating the S/N/R criteria formalized below.

(P1) Insufficiency. Keeping only the top-attended patches should preserve the prediction if they constitute sufficient evidence. Table 4 shows this fails: keeping attention top-k drops Macro-F1 by 0.078 from the full bag, averaged across nine datasets and nine backbones.

(P2) Unnecessity. Removing the top-attended patches should degrade the prediction if they are necessary evidence. However, removing attention top-k changes Macro-F1 by only 0.033 (Table 4), indicating the model largely recovers from the remaining patches.

(P3) Unrecoverability. During training, the selector operates in continuous space, but at inference a discrete subset must be extracted by thresholding. The continuous-discrete gap reaches 0.029 for ABMIL attention, compared with 0.005–0.011 for GCE-wrapped backbones (Appendix Table 7), meaning the discrete inference-time evidence disagrees with the continuous signal used during training.

These failures persist across ABMIL, TransMIL, CLAM, DSMIL, and other architectures [Ilse et al., 2018, Shao et al., 2021, Lu et al., 2021, Li et al., 2021]. The problem is compounded by evidence non-uniqueness: a recursive minimal-subset diagnostic on BRACS (Table 1) reveals that 72.67% of slides admit at least two disjoint sufficient subsets, yet attention produces a single global ranking that conflates these sources. This diagnostic motivates evidence selection beyond a single attention ranking; the subsequent experiments evaluate whether optimizing S/N/R improves evidence quality across datasets and backbones.

## 3.3 S/N/R: Three Criteria for Evidence Quality

The following definitions formalize what it means for an evidence subset to be “correct,” with each criterion addressing one failure.

Definition 1 (δ<sub>s</sub>-Sufficiency, addressing P1). For a bag predictor f and subset $S \subseteq \{ 1 , \ldots , N \}$ , let $X _ { S } = \{ x _ { i } : i \in S \}$ . S is $\delta _ { s }$ -sufficient if $| f ( X _ { S } ) - f ( { \bar { X } } ) | \leq \delta _ { s }$

Definition 2 (δ<sub>n</sub>-Necessity, addressing P2). Let $X \lnot s = \{ x _ { i } : i \notin S \}$ . S is $\delta _ { n }$ -necessary if $| f ( X \neg S ) - f ( X ) | \geq \delta _ { n } $

Definition 3 (δ -Recoverability, addressing P3). For a continuous selector $\pi \in [ 0 , 1 ] ^ { N }$ , let $X _ { \pi } =$ $\{ \pi _ { i } x _ { i } \}$ and $S ( \pi ) = \{ i : \pi _ { i } \geq \tau \}$ . π is $\delta _ { r }$ -recoverable if $| f ( X _ { \pi } ) - f ( X _ { S ( \pi ) } ) | \le \delta _ { r }$

Sufficiency ensures the evidence is self-contained; Necessity prevents trivial solutions (e.g., selecting the entire bag); Recoverability bridges training and inference. Together, they separate “correct prediction” from “correct explanation” as two evaluation axes. Appendix Table 6 summarizes the operational diagnostics used in the experiments. The next section presents GCE-MIL, which simultaneously addresses (P1), (P2), and (P3).

![](images/8ad16d1367866075d6bf7daeb9eba517f8014207e83d8f12e4b711ae7976dc29.jpg)  
Figure 2: GCE-MIL architecture. The framework wraps existing MIL backbones with three components: (1) low-rank adapter and semantic bridge for anchor grounding (drives Necessity via concept coverage), (2) continuous selector with exact noisy-OR coverage (drives Sufficiency via multi-source evidence), (3) threshold-plus-repair discrete recovery (drives Recoverability via the same marginal coverage objective). The host backbone remains unchanged, making GCE a plug-in wrapper.

## 4 Grounded Continuous Evidence MIL

## 4.1 Overview

GCE-MIL is a plug-in wrapper that adds evidence optimization to any existing MIL backbone $f _ { \theta }$ without modifying its architecture. The wrapper introduces three components, each targeting one of the S/N/R criteria identified in Section 3 (Figure 2):

• A semantic anchor bank that grounds patch selection in pathology-specific concepts, addressing Necessity (P2) by tying evidence to diagnostic structures rather than arbitrary attention scores.

• A continuous selector with noisy-OR coverage that produces soft gates $\pi \in [ 0 , 1 ] ^ { N }$ , addressing Sufficiency (P1) by modeling multi-source evidence coverage across diagnostic concepts.

• A discrete recovery procedure that converts π into an inference-time subset using the same submodular coverage utility, addressing Recoverability (P3) by keeping the discrete evidence close to the continuous selector.

The backbone $f _ { \theta }$ remains structurally unchanged—GCE only adds a soft evidence mask that modulates the backbone’s inputs. The gate π is injected according to the backbone’s aggregation type: as an attention-logit bias $\alpha _ { i }  \alpha _ { i } + \log \pi _ { i }$ for attention-based backbones (ABMIL, CLAM-SB, IBMIL), as feature reweighting $h _ { i }  \pi _ { i } \cdot h _ { i }$ for token-based backbones (TransMIL, DTFD-MIL, HDMIL, CAMIL), or as a hybrid of both for multi-path backbones (DSMIL, MHIM-MIL). This injection preserves the host backbone’s scoring head while giving GCE a consistent interface for evidence evaluation.

## 4.2 Semantic Anchor Grounding

GCE-MIL grounds evidence selection in domain-specific semantic concepts rather than learning selection from classification gradients alone. Diagnostically relevant structures—tumor nests, stromal reactions, necrosis, mitotic figures—have well-defined morphological descriptions that can serve as selection anchors, enabling the selector to distinguish diagnostically informative patches from visually salient but irrelevant ones.

GCE-MIL uses M = 8 semantic anchors defined as frozen text embeddings from TITAN [Ding et al., 2025], a pathology vision-language model. The anchor prompts are task-specific morphology descriptions chosen before training from disease and histology priors, without validation-set tuning or patch-level concept labels. Examples include “gland formation,” “nuclear pleomorphism,” “mitotic activity,” and “necrotic tumor cells”; Appendix Table 21 records the prompt categories used for each dataset family. The text embeddings are computed once and fixed during training, and only the adapter, bridge, selector, and host MIL parameters are learned.

Each patch embedding passes through a low-rank residual adapter $e _ { i } = \mathrm { n o r m } ( ( I + U V ^ { \top } ) h _ { i } )$ , where $U , V \in \mathbb { R } ^ { d \times r } \left( r = 3 2 \right)$ are initialized near zero. A separate bridge $B ( \cdot )$ maps raw features into the anchor space. The patch-anchor response is:

$$
r _ { i m } = \sigma \left( \gamma ( \cos ( B ( h _ { i } ) , a _ { m } ) - \delta ) \right) ,\tag{2}
$$

where $a _ { m }$ is the frozen anchor embedding, $\gamma = 8 . 0$ sharpens the response, and $\delta = 0 . 1 5$ suppresses weak matches. Disease-specific anchors improve over generic prompts in Table 18, lowering the C-D gap from 0.015 to 0.010 and raising complement degradation from 0.210 to 0.290. Random and shuffled prompts remain close to no grounding, while generic prompts, disease-specific prompts, and constrained TITAN grounding improve in order; this pattern suggests that the gain is not only an effect of adding selector capacity. The full TITAN anchor configuration with the constrained bridge further reaches 0.004 C-D gap and 0.412 complement degradation.

## 4.3 Continuous Selector and Noisy-OR Coverage

Given the anchor responses $\{ r _ { i m } \}$ , the continuous selector determines which patches to include in the evidence subset. For each patch, a small MLP receives the adapted embedding $e _ { i }$ and spatial coordinates $c _ { i } .$ , and outputs a scalar score $s _ { i }$ . The inclusion gate is computed as:

$$
\pi _ { i } = \sigma \left( { \frac { s _ { i } - \nu _ { x } } { T } } \right) ,\tag{3}
$$

where $\nu _ { x } = 0$ is a centering constant and $T$ is a temperature that is annealed from 1.0 to 0.4 during training. This annealing gradually pushes the gate distribution toward a bimodal regime (Figure 1, middle panel), making the continuous selector increasingly discrete-like and facilitating recovery at inference time.

Why noisy-OR for coverage? The S/N/R criteria impose specific requirements on how per-patch anchor responses are aggregated into coverage. Mean pooling conflates “many weak responses” with “one strong response,” violating coverage semantics. Attention pooling reintroduces softmax concentration. Noisy-OR provides the right inductive bias: for anchor $m ,$ coverage under continuous gates π is

$$
v _ { m } ( \pi ) = 1 - \prod _ { i } ( 1 - \pi _ { i } r _ { i m } ) .\tag{4}
$$

This models each patch as an independent evidence channel with diminishing marginal returns. The class-level utility aggregates coverage across anchors:

$$
U _ { c } ( \pi ) = \sum _ { m } \alpha _ { c m } v _ { m } ( \pi ) , \qquad \alpha _ { c m } \ge 0 ,\tag{5}
$$

where $\alpha _ { c m }$ are learnable class-anchor weights. Crucially, noisy-OR provides closed-form marginals for greedy repair: $\begin{array} { r } { \partial U _ { c } / \partial \pi _ { i } = \sum _ { m } \alpha _ { c m } r _ { i m } \prod _ { j \neq i } ( 1 - \bar { \pi } _ { j } r _ { j m } ) } \end{array}$ . The marginal gain decreases as more patches are selected, which is the diminishing-returns property needed for Necessity. Appendix B gives the corresponding modeling interpretation, including S/N/R independence, a gate-margin recoverability bound, conditional coverage bounds, and a Cox risk-pathway view.

Proposition 1 (Submodularity of Noisy-OR Coverage). Forfixed anchor responses $r _ { i m }$ and class weights $\alpha _ { c m } \geq 0 ;$ , the utility $\begin{array} { r } { \bar { U _ { c } } ( S ) = \bar { \sum _ { m } } \alpha _ { c m } [ 1 - \bar { \prod _ { i \in S } } ( 1 - r _ { i m } ) ] } \end{array}$ is monotone submodular in S.

Proof. For $S \subseteq T$ and $i \not \in T$ , the marginal gain is $\begin{array} { r } { \Delta _ { m } ( i | S ) = r _ { i m } \prod _ { j \in S } ( 1 - r _ { j m } ) \geq r _ { i m } \prod _ { j \in T } ( 1 - } \end{array}$ $r _ { j m } ) = \Delta _ { m } ( i | T )$ , since $S \subseteq T$ implies the product over S is at least as large. Summing over m with $\alpha _ { c m } \geq 0$ preserves the inequality. □

This submodularity justifies greedy marginal repair at the coverage-utility level: under the standard cardinality-limited coverage setting, greedy selection attains the usual $( 1 - 1 / e )$ approximation [Nemhauser et al., 1978]; Appendix B gives the curvature-aware refinement. The implemented repair additionally checks threshold recovery and prediction sufficiency, so the claim is a coverage-property statement rather than a global optimality statement about the classifier.

## 4.4 Training Objective and Discrete Recovery

GCE-MIL trains the host backbone and selector jointly with a composite loss:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { b } \mathcal { L } _ { \mathrm { b u d g e t } } + \lambda _ { g } \mathcal { L } _ { \mathrm { g r o u n d } } , } \end{array}\tag{6}
$$

where each term targets a specific S/N/R criterion. $\mathcal { L } _ { \mathrm { t a s k } }$ is the unmodified backbone loss (crossentropy for classification, Cox partial likelihood for survival), preserving the host model’s predictive capacity. $\mathcal { L } _ { \mathrm { b u d g e t } } = \mathrm { R e L U } ( \dot { \mathbb { E } } [ \pi ] - \rho ) ^ { 2 }$ enforces sparsity, driving Sufficiency by requiring the selector to preserve the prediction with a compact subset. The reported benchmark uses the operating evidence budget $\rho = 0 . 0 5$ , selected by the validation sweep in Appendix Table $1 6 ;$ larger budgets are reported as sensitivity points rather than mixed into the main tables. $\mathcal { L } _ { \mathrm { g r o u n d } }$ aligns π with noisy-OR anchor responses, driving Necessity by ensuring selected patches are grounded in diagnostic concepts rather than arbitrary features. Recoverability is enforced by temperature annealing and the threshold plus-repair procedure, which make the learned continuous gate compatible with discrete evidence extraction at inference. The weights $\lambda _ { b } = 0 . 1$ and $\lambda _ { g } = 0 . 5$ define the reported cross-dataset setting and are kept fixed across datasets and backbones after selection on BRACS validation folds. Table 3 validates each component’s contribution: adding budget control reduces the C-D gap from 0.055 to 0.011; adding grounding increases complement degradation from 0.318 to 0.403; the full pipeline reaches 0.004 gap and 0.412 degradation.

Discrete recovery at inference. At test time, GCE-MIL converts the continuous selector into a discrete evidence subset via threshold-plus-repair (Algorithm 1). The initial subset $S _ { 0 } = \{ i : \pi _ { i } >$ 0.5} is obtained by thresholding; if empty, the top-1 patch is used as a fallback. Greedy repair then adds patches in decreasing order of marginal coverage gain until the coverage target $c = 0 . 9 5$ is met. Because the coverage utility is monotone submodular (Proposition 1), this greedy procedure has a principled diminishing-returns objective rather than an unrelated post-hoc ranking. The pseudocode is provided in Appendix C.

Proposition 2 (Greedy recovery scope). Let $\pi \in [ 0 , 1 ] ^ { N }$ be the continuous selector, $S _ { 0 } = \{ i : \pi _ { i } >$ 0.5} be the thresholded subset, and ${ \bar { S } } ^ { * }$ be the output of Algorithm 1 with coverage target c. Then the following statements hold:

1. If the loop terminates by satisfying the coverage condition, then min<sub>m</sub> $v _ { m } ( \mathbf { 1 } _ { S ^ { * } } ) \geq c$ by construction.

2. Each added patch maximizes the exact one-step marginal gain ofthe noisy-OR utility used during training.

3. Ifrepair is restricted to afixed-size shortlist and evaluated only as coverage maximization, the greedy part inherits the standard $( 1 - 1 / e )$ approximation to the best shortlist subset of that size.

These are coverage-level properties; they do not assert global optimality of the host classifier under arbitrary interventions.

Proofsketch. The first claim follows directly from the termination condition. The second follows because Algorithm 1 ranks candidates by ${ \partial \dot { U } _ { c } } / { \partial { \pi _ { i } } }$ computed from the noisy-OR utility. The third follows from standard greedy analysis for monotone submodular maximization under a cardinality budget [Nemhauser et al., $1 9 7 8 ] ;$ prediction sufficiency is then checked empirically by the intervention diagnostics rather than assumed by the theorem. □

## 5 Experiments

## 5.1 Setup

Datasets. The evaluation covers 9 datasets spanning two tasks: 4 classification benchmarks (BRACS, PANDA, TCGA-BRCA, TCGA-NSCLC) and 5 survival cohorts (TCGA-LUAD, STAD, UCEC, KIRP, KIRC). These datasets cover diverse tissue types, label granularities (7-class fine-grained to binary subtyping), and bag sizes (hundreds to tens of thousands of patches). Dataset details are provided in Appendix K.

Backbones. GCE-MIL is attached to 9 host backbones spanning the major MIL families: attentionbased (ABMIL [Ilse et al., 2018], CLAM-SB [Lu et al., 2021], IBMIL [Lin et al., 2023]), transformer based (TransMIL [Shao et al., 2021]), dual-stream (DSMIL [Li et al., 2021]), pseudo-bag (DTFD-MIL

Table 2: Classification performance on four histopathology benchmarks (5-fold cross-validation). Baseline rows report absolute mean±std; +GCE rows report the change from the immediately preceding baseline using triangle markers.
<table><tr><td rowspan="2">Method</td><td colspan="3">BRACS</td><td colspan="3">NSCLC</td><td colspan="3">PANDA</td><td colspan="3">BRCA</td></tr><tr><td>Accuracy</td><td>Macro-F1</td><td>AUC</td><td>Accuracy</td><td>Macro-F1</td><td>AUC</td><td>Accuracy</td><td>Macro-F1</td><td>AUC</td><td>Accuracy</td><td>Macro-F1</td><td>AUC</td></tr><tr><td>ABMIL</td><td>0.754±0.028</td><td>0.634±0.045</td><td>0.864±0.019</td><td>0.909±0.014</td><td>0.895±0.016</td><td>0.948±0.009</td><td>0.701±0.015</td><td>0.643±0.018</td><td>0.925±0.008</td><td>0.805±0.027</td><td>0.449±0.049</td><td>0.803±0.022</td></tr><tr><td>+GCE ∆</td><td>△0.036</td><td>△0.069</td><td>△0.051</td><td>△0.023</td><td>△0.022</td><td>△0.013</td><td>△0.007</td><td>△0.000</td><td>△0.009</td><td>△0.014</td><td>△0.042</td><td>△0.017</td></tr><tr><td>CLAM-SB</td><td>0.807±0.025 △0.015</td><td>0.742±0.038 △0.023</td><td>0.919±0.016 △0.008</td><td>0.923±0.012 △0.017</td><td>0.911±0.014 △0.017</td><td>0.956±0.007</td><td>0.723±0.013</td><td>0.665±0.014</td><td>0.931±0.006</td><td>0.818±0.024 △0.021</td><td>0.523±0.042</td><td>0.821±0.019</td></tr><tr><td>+GCEΔ</td><td></td><td></td><td></td><td></td><td></td><td>△0.000</td><td>△0.006</td><td>△0.012</td><td>△0.001</td><td></td><td>△0.053</td><td>△0.018</td></tr><tr><td>TransMIL</td><td>0.738±0.031</td><td>0.676±0.048</td><td>0.883±0.021</td><td>0.936±0.013</td><td>0.924±0.015</td><td>0.969±0.008</td><td>0.730±0.014</td><td>0.670±0.016</td><td>0.923±0.007</td><td>0.820±0.025</td><td>0.548±0.045</td><td>0.813±0.021</td></tr><tr><td>+GCE∆</td><td>△0.030</td><td>△0.038</td><td>△0.021</td><td>△0.000</td><td>△0.000</td><td>△0.003</td><td>△0.009</td><td>∇0.002</td><td>△0.012</td><td>△0.000</td><td>△0.013</td><td>△0.015</td></tr><tr><td>DSMIL</td><td>0.765±0.029</td><td>0.684±0.044</td><td>0.888±0.018</td><td>0.930±0.012</td><td>0.915±0.015</td><td>0.962±0.007</td><td>0.696±0.016</td><td>0.628±0.019</td><td>0.911±0.008</td><td>0.846±0.022</td><td>0.529±0.044</td><td>0.839±0.018</td></tr><tr><td>+GCE ∆</td><td>△0.014</td><td>△0.042</td><td>△0.021</td><td>△0.015</td><td>△0.015</td><td>△0.008</td><td>△0.006</td><td>△0.010</td><td>△0.005</td><td>∇0.001</td><td>△0.014</td><td>△0.000</td></tr><tr><td>DTFD-MIL</td><td>0.784±0.026</td><td>0.691±0.041</td><td>0.917±0.015</td><td>0.933±0.011</td><td>0.921±0.013</td><td>0.965±0.007</td><td>0.736±0.012</td><td>0.676±0.015</td><td>0.932±0.006</td><td>0.899±0.018</td><td>0.684±0.032</td><td>0.840±0.016</td></tr><tr><td>+GCE∆</td><td>△0.026</td><td>△0.066</td><td>△0.009</td><td>△0.007</td><td>△0.000</td><td>△0.003</td><td>△0.002</td><td>△0.012</td><td>△0.004</td><td>△0.000</td><td>△0.007</td><td>△0.011</td></tr><tr><td>IBMIL</td><td>0.828±0.022</td><td>0.776±0.035</td><td>0.918±0.014</td><td>0.935±0.010</td><td>0.922±0.012</td><td>0.968±0.006</td><td>0.715±0.014</td><td>0.648±0.016</td><td>0.923±0.007</td><td>0.814±0.023</td><td>0.487±0.039</td><td>0.829±0.017</td></tr><tr><td>+GCE ∆</td><td>△0.008</td><td>△0.025</td><td>△0.013</td><td>△0.014</td><td>△0.014</td><td>∇0.003</td><td>△0.019</td><td>△0.034</td><td>△0.007</td><td>△0.020</td><td>△0.060</td><td>△0.007</td></tr><tr><td>MHIM-MIL</td><td>0.796±0.025</td><td>0.718±0.042</td><td>0.899±0.017</td><td>0.938±0.011</td><td>0.926±0.013</td><td>0.970±0.007</td><td>0.713±0.013</td><td>0.655±0.015</td><td>0.919±0.007</td><td>0.839±0.021</td><td>0.556±0.038</td><td>0.846±0.016</td></tr><tr><td>+GCEΔ</td><td>△0.012</td><td>△0.030</td><td>△0.023</td><td>△0.007</td><td>△0.006</td><td>△0.000</td><td>△0.012</td><td>△0.017</td><td>△0.004</td><td>△0.017</td><td>△0.044</td><td>△0.012</td></tr><tr><td>CAMIL</td><td>0.738±0.030 △0.039</td><td>0.666±0.049 △0.054</td><td>0.879±0.020 △0.029</td><td>0.944±0.011 △0.006</td><td>0.931±0.013</td><td>0.975±0.006</td><td>0.719±0.014</td><td>0.657±0.017</td><td>0.926±0.008</td><td>0.839±0.022</td><td>0.608±0.043</td><td>0.811±0.018</td></tr><tr><td>+GCE∆</td><td></td><td></td><td></td><td></td><td>△0.000</td><td>∇0.002</td><td>△0.006</td><td>△0.000</td><td>△0.004</td><td>△0.003</td><td>△0.002</td><td>△0.006</td></tr><tr><td>HDMIL</td><td>0.781±0.027</td><td>0.707±0.040</td><td>0.891±0.018</td><td>0.940±0.010</td><td>0.927±0.012</td><td>0.970±0.006</td><td>0.746±0.011</td><td>0.696±0.013</td><td>0.932±0.006</td><td>0.866±0.019</td><td>0.603±0.035</td><td>0.835±0.017</td></tr><tr><td>+GCE∆</td><td>△0.038</td><td>△0.066</td><td>△0.038</td><td>△0.000</td><td>△0.006</td><td>△0.003</td><td>△0.000</td><td>△0.005</td><td>△0.006</td><td>△0.006</td><td>△0.039</td><td>△0.009</td></tr></table>

[Zhang et al., 2022]), hard-mining (MHIM-MIL [Tang et al., 2023]), context-aware (CAMIL [Fourkioti et al., 2024]), and hierarchical (HDMIL [Dong et al., 2025]). This 9 × 9 grid (81 configurations) tests whether GCE generalizes across backbone architectures and dataset characteristics. All training and evaluation protocol details are provided in Appendix J.

## 5.2 Main Classification Results

Table 2 reports the classification half of the $9 \times 9$ benchmark; the survival half is reported in Appendix Table 12. The central question is whether optimizing evidence quality changes slide-level prediction, or merely reshuffles which patches are selected without affecting accuracy. GCE gives positive Macro-F1 changes on most backbone-dataset pairs, with the largest gains on the most challenging dataset (BRACS, 7-class fine-grained classification). On BRACS, ABMIL improves from 0.634 to 0.703 Macro-F1 (+6.9 points), HDMIL from 0.707 to 0.773 (+6.6 points), and IBMIL from 0.776 to 0.801 (+2.5 points). The gains are smaller on easier datasets (NSCLC, binary subtyping) where baselines already achieve > 0.90 Macro-F1, consistent with the expectation that evidence optimization matters most when the classification task requires integrating multiple diagnostic concepts. On PANDA, GCE preserves performance (HDMIL: 0.696 → 0.701) while compacting evidence to ∼ 5% of patches. On BRCA, the hardest binary task, CLAM-SB improves from 0.523 to 0.576 and MHIM-MIL from 0.556 to 0.600.

## 5.3 Ablation Study

Table 3 isolates the contribution of each component. A naive selector leaves a large C-D gap (0.055) and weak complement degradation (0.090); adding budget control, recovery, and grounding progressively yields the full GCE result: 0.748 Macro-F1, 0.004 C-D gap, and 0.412 complement degradation.

Table 3: Component ablation on BRACS. Each row adds one module to the pipeline.
<table><tr><td>Variant</td><td>Macro-F1</td><td></td><td>C-D Gap↓ Compl. Degr.↑ Evid. Suff.↑</td><td></td></tr><tr><td>Backbone only</td><td>0.699±0.047</td><td></td><td></td><td></td></tr><tr><td>Naive selector</td><td>0.708±0.045</td><td> $0 . 0 5 5 { \pm } 0 . 0 1 4$ </td><td> $0 . 0 9 0 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $0 . 3 8 1 { \scriptstyle \pm 0 . 0 6 1 }$ </td></tr><tr><td>+ Budget control</td><td>0.738±0.0360.011±0.004</td><td></td><td> $0 . 3 1 8 { \pm } 0 . 0 5 8$ </td><td> $0 . 5 9 6 { \scriptstyle \pm 0 . 0 9 4 }$ </td></tr><tr><td>+ Discrete recovery</td><td>0.744±0.033</td><td> $0 . 0 0 6 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 3 7 7 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td> $0 . 6 3 0 { \scriptstyle \pm 0 . 1 0 2 }$ </td></tr><tr><td>+ Semantic grounding</td><td> $0 . 7 4 6 { \pm } 0 . 0 3 5$ </td><td> $0 . 0 0 5 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 4 0 3 { \scriptstyle \pm 0 . 0 8 2 }$ </td><td> $0 . 6 4 9 { \pm } 0 . 1 1 1$ </td></tr><tr><td>Full GCE</td><td> $\pm 0 . 7 4 8 { \pm } 0 . 0 3 2$ </td><td> $\mathbf { 0 . 0 0 4 } \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 4 1 2 { \overset { . } { = } } 0 . 0 8 6 }$ </td><td> $\mathbf { 0 . 6 5 9 } { \pm 0 . 1 0 8 }$ </td></tr></table>

The ablation also separates the three S/N/R mechanisms. Budget control gives the first large improvement: the gap falls from 0.055 to 0.011 and complement degradation rises from 0.090 to 0.318, consistent with the claim that sparse evidence must carry more of the decision. Discrete recovery contributes mainly to Recoverability $( 0 . 0 1 1  0 . 0 0 6 )$ , while semantic grounding contributes mainly

![](images/e6ab1510277ac658b072fe41871617c5e3bd90c43b042c36d2dd77a63027b5a8.jpg)  
Figure 3: Main qualitative evidence example. The host attention map and the recovered GCE evidence subset are shown on the same slide. GCE selects a compact, recoverable evidence set rather than simply visualizing the original attention ranking.

Table 4: Intervention diagnostic. Values are changes relative to the full-bag score averaged over the nine-dataset benchmark.
<table><tr><td>Subset rule</td><td>Keep only ∆</td><td>Remove ∆</td></tr><tr><td>Random-k</td><td>∇0.180</td><td>△0.001</td></tr><tr><td>Attention top-k</td><td>∇0.078</td><td>∇0.033</td></tr><tr><td>Saliency top-k</td><td>∇0.063</td><td>∇0.042</td></tr><tr><td>GCE evidence</td><td>△0.004</td><td>∇0.176</td></tr></table>

Table 5: Same-budget evidence comparison on BRACS. All subset rules select approximately 5% of patches.
<table><tr><td>Subset rule</td><td>Macro-F1↑</td><td>Gap↓</td><td>Compl. Degr.↑</td></tr><tr><td>Random-k</td><td>0.431±0.054</td><td>0.043±0.012</td><td>0.048±0.015</td></tr><tr><td>Attention top-k</td><td>0.597±0.045</td><td>0.029±0.008</td><td>0.151±0.036</td></tr><tr><td>Gradient top-k</td><td>0.613±0.041</td><td>0.025±0.006</td><td>0.166±0.031</td></tr><tr><td>Occlusion top-k</td><td>0.628±0.038</td><td>0.022±0.005</td><td>0.196±0.039</td></tr><tr><td>GCE discrete</td><td>0.748±0.032</td><td>0.004±0.001</td><td>0.412±0.086</td></tr></table>

to Necessity (0.377 → 0.403), showing that anchor coverage is not merely a parameter increase but changes which patches are treated as decision-critical.

## 5.4 Qualitative Evidence Behavior

Figure 3 places the learned evidence mask next to the host attention map. The qualitative pattern matches the intervention diagnostics: attention highlights broad high-score regions, while GCE recovers a compact subset that remains spatially coherent after thresholding and repair. This figure is included in the main text because it clarifies what the S/N/R metrics measure at the slide level.

## 5.5 Intervention and Budget-Matched Evidence

Table 4 reports the direct intervention diagnostic for Sufficiency and Necessity. Keeping attention top-k changes the full-bag score by ▽0.078, whereas keeping GCE evidence changes it by only △0.004; removing attention changes the score by ▽0.033, but removing GCE evidence causes a ▽0.176 drop. Table 5 controls for evidence size by forcing all subset rules to select approximately 5% of each BRACS bag. At the same budget, attention top-k reaches 0.597 Macro-F1 and 0.151 complement degradation, whereas discrete GCE reaches 0.748 and 0.412; the prediction gap falls from 0.029 to 0.004. The gains therefore do not come from selecting more tissue, but from recovering a subset that is sufficient, necessary, and discrete-faithful.

## 6 Conclusion

This work formalizes the gap between classification accuracy and evidence quality in MIL through three criteria—Sufficiency, Necessity, and Recoverability—and shows that existing attention-based methods fail all three. GCE-MIL addresses these failures with semantic anchor grounding, noisy-OR coverage with closed-form marginals, and threshold-plus-repair discrete recovery. Across 81 backbone-dataset configurations, GCE-MIL improves both prediction and evidence quality, and optional tile prefiltering enables up to 5× faster end-to-end inference at 0.989× relative utility.

## References

David Alvarez Melis and Tommi Jaakkola. Towards robust interpretability with self-explaining neural networks. Advances in neural information processing systems, 31, 2018.

Peter Bandi, Oscar Geessink, Quirine Manson, Marcory Van Dijk, Maschenka Balkenhol, Meyke Hermsen, Babak Ehteshami Bejnordi, Byungjae Lee, Kyunghyun Paeng, Aoxiao Zhong, et al. From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge. IEEE transactions on medical imaging, 38(2):550–560, 2018.

Nadia Brancati, Anna Maria Anniciello, Pushpak Pati, Daniel Riccio, Giosuè Scognamiglio, Guillaume Jaume, Giuseppe De Pietro, Maurizio Di Bonito, Antonio Foncubierta, Gerardo Botti, et al. Bracs: A dataset for breast carcinoma subtyping in h&e histology images. Database, 2022: baac093, 2022.

Wouter Bulten, Kimmo Kartasalo, Po-Hsuan Cameron Chen, Peter Ström, Hans Pinckaers, Kunal Nagpal, Yuannan Cai, David F Steiner, Hester Van Boven, Robert Vink, et al. Artificial intelligence for diagnosis and gleason grading of prostate cancer: the panda challenge. Nature medicine, 28(1): 154–163, 2022.

PC Bunch. Free response approach to measurement and characterization of radiographic observer performance. AJR Am J Roentgenol, 130(2):382, 1978.

Zak Buzzard, Konstantin Hemker, Nikola Simidjievski, and Mateja Jamnik. Paths: A hierarchical transformer for efficient whole slide image analysis. arXiv preprint arXiv:2411.18225, 2024.

Gabriele Campanella, Matthew G Hanna, Luke Geneslaw, Allen Miraflor, Vitor Werneck Krauss Silva, Klaus J Busam, Edi Brogi, Victor E Reuter, David S Klimstra, and Thomas J Fuchs. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nature medicine, 25(8):1301–1309, 2019.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

Chaofan Chen, Oscar Li, Daniel Tao, Alina Barnett, Cynthia Rudin, and Jonathan K Su. This looks like that: deep learning for interpretable image recognition. Advances in neural information processing systems, 32, 2019.

Chi-Long Chen, Chi-Chung Chen, Wei-Hsiang Yu, Szu-Hua Chen, Yu-Chan Chang, Tai-I Hsu, Michael Hsiao, Chao-Yuan Yeh, and Cheng-Yu Chen. An annotation-free whole-slide training approach to pathological classification of lung cancer types using deep learning. Nature communications, 12(1):1193, 2021a.

Jianbo Chen, Le Song, Martin Wainwright, and Michael Jordan. Learning to explain: An informationtheoretic perspective on model interpretation. In International conference on machine learning, pages 883–892. PMLR, 2018.

Richard J Chen, Ming Y Lu, Muhammad Shaban, Chengkuan Chen, Tiffany Y Chen, Drew FK Williamson, and Faisal Mahmood. Whole slide images are 2d point clouds: Context-aware survival prediction using patch-based graph convolutional networks. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 339–349. Springer, 2021b.

Richard J Chen, Tong Ding, Ming Y Lu, Drew FK Williamson, Guillaume Jaume, Andrew H Song, Bowen Chen, Andrew Zhang, Daniel Shao, Muhammad Shaban, et al. Towards a general-purpose foundation model for computational pathology. Nature medicine, 30(3):850–862, 2024.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PmLR, 2020.

Jun Cheng, Yuting Liu, Wei Huang, Wenhui Hong, Lingling Wang, Xiaohui Zhan, Zhi Han, Dong Ni, Kun Huang, and Jie Zhang. Computational image analysis identifies histopathological image features associated with somatic mutations and patient survival in gastric adenocarcinoma. Frontiers in Oncology, 11:623382, 2021.

Thomas M Cover. Elements ofinformation theory. John Wiley & Sons, 1999.

Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong, Richard Socher, and Byron C Wallace. Eraser: A benchmark to evaluate rationalized nlp models. In Proceedings of the 58th annual meeting ofthe associationfor computational linguistics, pages 4443–4458, 2020.

Thomas G Dietterich, Richard H Lathrop, and Tomás Lozano-Pérez. Solving the multiple instance problem with axis-parallel rectangles. Artificial intelligence, 89(1-2):31–71, 1997.

Tong Ding, Sophia J Wagner, Andrew H Song, Richard J Chen, Ming Y Lu, Andrew Zhang, Anurag J Vaidya, Guillaume Jaume, Muhammad Shaban, Ahrong Kim, et al. A multimodal whole-slide foundation model for pathology. Nature Medicine, pages 1–13, 2025.

Jiuyang Dong, Junjun Jiang, Kui Jiang, Jiahan Li, and Yongbing Zhang. Fast and accurate gigapixel pathological image classification with hierarchical distillation multi-instance learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 30818–30828, 2025.

Zhaolong Du, Shasha Mao, Xuequan Lu, Mengnan Qi, Yimeng Zhang, Jing Gu, and Licheng Jiao. Rethinking multiple-instance learning from feature space to probability space. In The Thirteenth International Conference on Learning Representations, 2025.

Babak Ehteshami Bejnordi, Mitko Veta, Paul Johannes van Diest, Bram Van Ginneken, Nico Karssemeijer, Geert Litjens, Jeroen AWM Van Der Laak, CAMELYON16 consortium, Meyke Hermsen, Quirine F Manson, et al. Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. Jama, 318(22):2199–2210, 2017.

Ruth C Fong and Andrea Vedaldi. Interpretable explanations of black boxes by meaningful perturbation. In Proceedings of the IEEE international conference on computer vision, pages 3429–3437, 2017.

Olga Fourkioti, Matt De Vries, and Chris Bakal. CAMIL: Context-aware multiple instance learning for cancer detection and subtyping in whole slide images. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=rzBskAEmoc.

Ziyu Guo, Weiqin Zhao, Shujun Wang, and Lequan Yu. Higt: Hierarchical interaction graphtransformer for whole slide image analysis. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 755–764. Springer, 2023.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Maximilian Ilse, Jakub Tomczak, and Max Welling. Attention-based deep multiple instance learning. In International conference on machine learning, pages 2127–2136. PMLR, 2018.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In International Conference on Learning Representations, 2017.

Mingu Kang, Heon Song, Seonwook Park, Donggeun Yoo, and Sérgio Pereira. Benchmarking selfsupervised learning on diverse pathology datasets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3344–3354, 2023.

Saarthak Kapse, Pushpak Pati, Srikar Yellapragada, Srijan Das, Rajarsi R Gupta, Joel Saltz, Dimitris Samaras, and Prateek Prasanna. Gecko: Gigapixel vision-concept contrastive pretraining in histopathology. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20020–20030, 2025.

Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In International conference on machine learning, pages 2668–2677. PMLR, 2018.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In International conference on machine learning, pages 5338–5348. PMLR, 2020.

Oren Z Kraus, Jimmy Lei Ba, and Brendan J Frey. Classifying and segmenting microscopy images with deep multiple instance learning. Bioinformatics, 32(12):i52–i59, 2016.

Bin Li, Yin Li, and Kevin W Eliceiri. Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 14318–14328, 2021.

Jiayun Li, Wenyuan Li, Arkadiusz Gertych, Beatrice S Knudsen, William Speier, and Corey W Arnold. An attention-based multi-resolution model for prostate whole slide imageclassification and localization. arXiv preprint arXiv:1905.13208, 2019.

Tiancheng Lin, Zhimiao Yu, Hongyu Hu, Yi Xu, and Chang-Wen Chen. Interventional bag multiinstance learning on whole-slide pathological images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19830–19839, 2023.

Christos Louizos, Max Welling, and Diederik P Kingma. Learning sparse neural networks through l\_0 regularization. In International Conference on Learning Representations, 2018.

Ming Y Lu, Drew FK Williamson, Tiffany Y Chen, Richard J Chen, Matteo Barbieri, and Faisal Mahmood. Data-efficient and weakly supervised computational pathology on whole-slide images. Nature biomedical engineering, 5(6):555–570, 2021.

Ming Y Lu, Bowen Chen, Andrew Zhang, Drew FK Williamson, Richard J Chen, Tong Ding, Long Phi Le, Yung-Sung Chuang, and Faisal Mahmood. Visual language pretrained multiple instance zero-shot transfer for histopathology images. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 19764–19775, 2023.

C Maddison, A Mnih, and Y Teh. The concrete distribution: A continuous relaxation of discrete random variables. In Proceedings of the international conference on learning Representations. International Conference on Learning Representations, 2017.

Oded Maron and Tomás Lozano-Pérez. A framework for multiple-instance learning. Advances in neural information processing systems, 10, 1997.

Andre Martins and Ramon Astudillo. From softmax to sparsemax: A sparse model of attention and multi-label classification. In International conference on machine learning, pages 1614–1623. PMLR, 2016.

Harold Miller. The froc curve: A representation of the observer’s performance for the method of free response. The Journal of the Acoustical Society of America, 46(6B):1473–1476, 1969.

George L Nemhauser, Laurence A Wolsey, and Marshall L Fisher. An analysis of approximations for maximizing submodular set functions—i. Mathematical programming, 14(1):265–294, 1978.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research Journal, 2024.

Liron Pantanowitz, Paul N Valenstein, Andrew J Evans, Keith J Kaplan, John D Pfeifer, David C Wilbur, Laura C Collins, and Terence J Colgan. Review of the current state of whole slide imaging in pathology. Journal of pathology informatics, 2(1):36, 2011.

Vitali Petsiuk, Abir Das, and Kate Saenko. Rise: Randomized input sampling for explanation of black-box models. In Proceedings ofthe British Machine Vision Conference (BMVC), 2018.

Linhao Qu, Yingfan Ma, Xiaoyuan Luo, Qinhao Guo, Manning Wang, and Zhijian Song. Rethinking multiple instance learning for whole slide image classification: A good instance classifier is all you need. IEEE Transactions on Circuits and Systems for Video Technology, 34(10):9732–9744, 2024.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211–252, 2015.

Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):379–423, 1948.

Daniel Shao, Richard J Chen, Andrew H Song, Joel Runevic, Ming Y Lu, Tong Ding, and Faisal Mahmood. Do multiple instance learning models transfer? Proceedings of Machine Learning Research, 267:54219–54238, 2025.

Zhuchen Shao, Hao Bian, Yang Chen, Yifeng Wang, Jian Zhang, Xiangyang Ji, et al. Transmil: Transformer based correlated multiple instance learning for whole slide image classification. Advances in neural information processing systems, 34:2136–2147, 2021.

Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.

Andrew H Song, Guillaume Jaume, Drew FK Williamson, Ming Y Lu, Anurag Vaidya, Tiffany R Miller, and Faisal Mahmood. Artificial intelligence for digital and computational pathology. Nature Reviews Bioengineering, 1(12):930–949, 2023.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929–1958, 2014.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In International conference on machine learning, pages 3319–3328. PMLR, 2017.

Wenhao Tang, Sheng Huang, Xiaoxian Zhang, Fengtao Zhou, Yi Zhang, and Bo Liu. Multiple instance learning framework with masked hard instance mining for whole slide image classification. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4078–4087, 2023.

Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Advances in neural information processing systems, 30, 2017.

Paul Tourniaire, Marius Ilie, Paul Hofman, Nicholas Ayache, and Hervé Delingette. Ms-clam: Mixed supervision for the classification and localization of tumors in whole slide images. Medical Image Analysis, 85:102763, 2023.

Constantino Tsallis. Possible generalization of boltzmann-gibbs statistics. Journal of statistical physics, 52(1):479–487, 1988.

Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.

Gregory Verghese, Jochen K Lennerz, Danny Ruta, Wen Ng, Selvam Thavaraj, Kalliopi P Siziopikou, Threnesan Naidoo, Swapnil Rane, Roberto Salgado, Sarah E Pinder, et al. Computational pathology in cancer diagnosis, prognosis, and prediction–present day and prospects. The Journal of pathology, 260(5):551–563, 2023.

Xi Wang, Hao Chen, Caixia Gan, Huangjing Lin, Qi Dou, Efstratios Tsougenis, Qitao Huang, Muyan Cai, and Pheng-Ann Heng. Weakly supervised deep learning for whole slide lung cancer image analysis. IEEE transactions on cybernetics, 50(9):3950–3962, 2019.

John N Weinstein, Eric A Collisson, Gordon B Mills, Kenna R Shaw, Brad A Ozenberger, Kyle Ellrott, Ilya Shmulevich, Chris Sander, and Joshua M Stuart. The cancer genome atlas pan-cancer analysis project. Nature genetics, 45(10):1113–1120, 2013.

Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, and Vikas Singh. Nyströmformer: A nyström-based algorithm for approximating self-attention. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 14138–14148, 2021.

Yan Xu, Jun-Yan Zhu, I Eric, Chao Chang, Maode Lai, and Zhuowen Tu. Weakly supervised histopathology cancer image segmentation and classification. Medical image analysis, 18(3): 591–604, 2014.

Shu Yang, Yihui Wang, and Hao Chen. Mambamil: Enhancing long sequence modeling with sequence reordering in computational pathology. In International conference on medical image computing and computer-assisted intervention, pages 296–306. Springer, 2024.

Jiawen Yao, Xinliang Zhu, Jitendra Jonnagaddala, Nicholas Hawkins, and Junzhou Huang. Whole slide images based cancer survival prediction using attention guided deep multiple instance learning networks. Medical image analysis, 65:101789, 2020.

Linfeng Ye, Shayan Mohajer Hamidi, Zhixiang Chi, Guang Li, Mert Pilanci, Takahiro Ogawa, Miki Haseyama, and Konstantinos N Plataniotis. Asmil: Attention-stabilized multiple instance learning for whole slide imaging. arXiv preprint arXiv:2603.06658, 2026.

Jinsung Yoon, James Jordon, and Mihaela Van der Schaar. Invase: Instance-wise variable selection using neural networks. In International conference on learning representations, 2018.

Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In European conference on computer vision, pages 818–833. Springer, 2014.

Hongrun Zhang, Yanda Meng, Yitian Zhao, Yihong Qiao, Xiaoyun Yang, Sarah E Coupland, and Yalin Zheng. Dtfd-mil: Double-tier feature distillation multiple instance learning for histopathology whole slide image classification. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 18802–18812, 2022.

Yunlong Zhang, Honglin Li, Yunxuan Sun, Sunyi Zheng, Chenglu Zhu, and Lin Yang. Attentionchallenging multiple instance learning for whole slide image classification. In European conference on computer vision, pages 125–143. Springer, 2024.

Yunlong Zhang, Honglin Li, Yuxuan Sun, Zhongyi Shui, Jingxiong Li, Chenglu Zhu, and Lin Yang. Aem: attention entropy maximization for multiple instance learning based whole slide image classification. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 45–55. Springer, 2025.

## A Overview

This appendix provides the theoretical, experimental, and implementation details that complement the main paper. We first state the scope of the theoretical claims and derive the coverage-level properties of noisy-OR recovery in Appendix B. We then give the detailed discrete recovery algorithm in Appendix C, followed by additional evidence diagnostics and baseline comparisons in Appendix D. Appendix E reports the survival prediction results and explains how the same coverage view extends to Cox risk modeling. Appendix F presents CAMELYON-16 localization and significance tests, while Appendix G expands the ablation and sensitivity analyses. Appendix H reports multi-backbone and multi-encoder generalization, and Appendix I summarizes the failure-case audit. Finally, Appendix J provides hardware, training, anchor-prompt, and computational-cost details; Appendix K describes datasets and preprocessing; Appendix L collects additional visualizations; and Appendix M discusses limitations and broader implications.

## B Theoretical Foundation and Notes

Scope of theory. The results in this appendix are coverage-level statements about the noisy-OR utility, not claims of global classifier optimality. Submodularity justifies the marginal-gain recovery objective under fixed anchor responses, while the conditional interventional bound explains when uncovered anchor mass can upper-bound omitted representation contribution. Classifier-level faithfulness is therefore evaluated empirically through keep/remove interventions and, separately, through CAMELYON-16 localization against pixel-level annotations.

## B.1 Coverage as an Interventional Proxy

Noisy-OR is used as a differentiable proxy for intervention search rather than as the definition of evidence faithfulness. The target quantity is the intervention variation $V ( S ) = \| f ( X ) - f ( X _ { S } ) \|$ which is expensive to optimize directly over discrete subsets. The following proposition states the assumption under which uncovered anchor mass controls this variation.

Proposition 3 (Coverage residual bounds representation variation). Let a MIL predictor decompose as $f ( { \bar { X } } ) = q ( g ( X ) )$ , where q is $L _ { q } – L i p s c h i t z$ and the bag representation is additive, $\begin{array} { r } { g ( X ) = \sum _ { i } w _ { i } h _ { i } } \end{array}$ with $w _ { i } \geq 0 .$ . Assume the anchor bank is η-complete for the omitted patch contribution: for any subset S,

$$
\left\| \sum _ { i \notin S } w _ { i } h _ { i } \right\| \leq \eta \sum _ { m } \alpha _ { m } \prod _ { i \in S } ( 1 - r _ { i m } ) , \qquad \alpha _ { m } \geq 0 .
$$

Then

$$
\| f ( X ) - f ( X _ { S } ) \| \leq L _ { q } \eta \sum _ { m } \alpha _ { m } { \big ( } 1 - v _ { m } ( \mathbf { 1 } _ { S } ) { \big ) } .
$$

Proof. By Lipschitz continuity, $\| f ( X ) - f ( X _ { S } ) \| \leq L _ { q } \| g ( X ) - g ( X _ { S } ) \|$ . For the additive representation, $\begin{array} { r } { g ( X ) - g ( X _ { S } ) = \sum _ { i \notin S } w _ { i } h _ { i } } \end{array}$ . Applying the anchor-completeness assumption gives the stated bound, and $\begin{array} { r } { 1 - v _ { m } ( \mathbf { 1 } _ { S } ) = \prod _ { i \in S } ( 1 - r _ { i m } ) } \end{array}$ by the noisy-OR definition. □

Proposition 3 is intentionally conditional. It does not claim that anchors perfectly explain every backbone; instead, it states what the noisy-OR objective is approximating when anchors cover the residual directions relevant to the host predictor. Direct keep-only and remove interventions are therefore still reported: the bound motivates the proxy, while the interventions test whether the proxy actually tracks prediction changes.

## B.2 Independence of S/N/R Criteria

Sufficiency, Necessity, and Recoverability are separate desiderata rather than three names for the same intervention score. The following proposition formalizes why all three diagnostics are reported.

Proposition 4 (S/N/R are logically independent). For a fixed model class, none of Sufficiency, Necessity, and Recoverability implies either of the other two in general.

Proofsketch. A selector that returns the whole bag is sufficient but not necessary, because its complement is empty or uninformative. A selector that returns one of several redundant diagnostic regions can be sufficient but not necessary, because the complement still contains another sufficient region. A selector can be necessary but not sufficient when the model relies on an interaction between the selected region and contextual tissue outside the subset. Finally, a continuous gate can be recoverable under thresholding while the resulting discrete subset is neither sufficient nor necessary, for example when the gate is stable but selects background or redundant tissue. Conversely, sufficient or necessary subsets can be generated by gates with many values near the threshold, making them poorly recoverable even when the intervention property holds. These constructions are model-relative and do not rely on a particular backbone. □

## B.3 Recoverability Bound for Thresholded Gates

Recoverability is evaluated by comparing the continuous gated input with its thresholded subset. Let $F _ { X } ( \pi )$ denote the prediction of a fixed trained MIL model on slide X after applying gate vector π to the bag, and let $S _ { \tau } ( \pi ) = \{ i : \pi _ { i } \geq \tau \}$

Proposition 5 (Gate-margin recoverability bound). If $F _ { X }$ is $L _ { X }$ -Lipschitz in the gate vector under norm $\| \cdot \| ,$ then

$$
\| F _ { X } ( \pi ) - F _ { X } ( \mathbf { 1 } _ { S _ { \tau } ( \pi ) } ) \| \le L _ { X } \| \pi - \mathbf { 1 } _ { S _ { \tau } ( \pi ) } \| .
$$

In particular, if every gate satisfies min<sub>i</sub> $| \pi _ { i } - \tau | \geq m a n d \pi _ { i } \in [ 0 , 1 ]$ , the bound tightens as annealing drives gates toward {0, 1} and away from the threshold.

Proof. The inequality follows directly from Lipschitz continuity of $F _ { X }$ with respect to the gate vector. Temperature annealing affects the right-hand side by reducing the distance between the continuous gate and the thresholded binary gate. The proposition therefore explains the role of annealing in reducing the C-D gap, but it does not assert that the recovered subset is clinically causal evidence.

## B.4 Why Noisy-OR

Noisy-OR is a natural coverage model when multiple independent evidence sources can activate the same latent concept. For WSI evidence, this matches the setting where several patches may each support gland formation, necrosis, or inflammatory response. Unlike mean pooling, noisy-OR saturates when a concept is already covered; unlike attention, it does not force concepts to compete through a simplex. This saturation behavior is important for Necessity: once an anchor is covered, adding another redundant patch has lower marginal gain, so greedy recovery is biased toward complementary evidence.

For a monotone submodular utility U, the total curvature is

$$
\kappa = 1 - \operatorname* { m i n } _ { i : U ( \{ i \} ) > 0 } \frac { U ( V ) - U ( V \setminus \{ i \} ) } { U ( \{ i \} ) } , \qquad 0 \le \kappa \le 1 .
$$

For noisy-OR coverage, this curvature is determined by anchor-response redundancy: patches whose anchor responses are already covered by other selected patches have lower late-stage marginal gain. Under the same cardinality-limited coverage setting used for greedy repair, the standard greedy guarantee can therefore be sharpened to the curvature-aware form

$$
U ( S _ { \mathrm { g r e e d y } } ) \geq { \frac { 1 - e ^ { - \kappa } } { \kappa } } U ( S ^ { \star } ) ,
$$

with the convention that the factor equals 1 when $\kappa = 0$ . This does not change the coverage-level scope of the theorem, but it clarifies why greedy repair is most effective when the selected patches cover complementary anchors rather than redundant copies of the same morphology.

## B.5 Risk-Pathway Interpretation for Survival

For survival tasks, GCE-MIL uses the same selector and coverage utility but trains the slide output as a scalar Cox risk score. The noisy-OR term has a compatible interpretation if each anchor is viewed as a latent risk pathway. Let $z _ { m } ( X )$ denote whether pathway m is present somewhere in the slide, and let $r _ { i m }$ be the probability that patch i expresses that pathway. Under conditional independence of pathway evidence across selected patches,

$$
P ( z _ { m } = 1 \mid S ) = 1 - \prod _ { i \in S } ( 1 - r _ { i m } ) = v _ { m } ( \mathbf { 1 } _ { S } ) .
$$

A Cox-style log-risk model can then be written as

$$
h ( X _ { S } ) = \sum _ { m } \beta _ { m } v _ { m } ( \mathbf { 1 } _ { S } ) , \qquad \lambda ( t \mid X _ { S } ) = \lambda _ { 0 } ( t ) \exp ( h ( X _ { S } ) ) .
$$

This mapping does not turn the method into a full competing-risks model; it only explains why the same saturating coverage form is reasonable for scalar risk prediction. Once a high-risk morphology is already covered, additional patches showing the same morphology have diminishing marginal effect on the log-risk, whereas a patch activating a different risk pathway can still change the risk score.

## B.6 Sufficiency-Necessity Duality for MIL Classifiers

Sufficiency and Necessity are complementary but not equivalent. A subset can preserve the prediction when kept while still being redundant with other evidence in the complement. Conversely, a subset can be necessary when removed but insufficient on its own if it interacts with context. GCE-MIL therefore reports both keep-only and remove interventions.

## B.7 Recoverability under Temperature Annealing

Recoverability depends on the gap between the continuous selector and the thresholded subset. Temperature annealing encourages π<sub>i</sub> values to move toward 0 or 1, reducing ambiguity near the threshold. Greedy repair then corrects residual coverage failures after thresholding. The empirical counterpart is the C-D gap: selector and grounding ablations that leave many gates near the threshold produce larger gaps, while the final model reduces the gap to 0.004–0.005 in the main diagnostics.

## C Detailed Recovery Algorithm

Algorithm 1 gives the discrete recovery procedure used at inference. It first thresholds the continuous gate π and then greedily repairs coverage failures using the exact noisy-OR marginal. The main method section therefore remains focused on the S/N/R design while the full implementation detail is provided here. The key implementation choice is that repair ranks candidates by the exact marginal of the same noisy-OR utility used during training. Thus the discrete subset is not a separate heuristic explanation; it is the hard recovery of the trained continuous evidence objective.

```perl
Algorithm 1: GCE-MIL Discrete Recovery
Input: continuous gates π, anchor responses r, coverage target $c = 0 . 9 5$
Output: discrete subset $S ^ { * }$
1 $S \gets \{ i : \pi _ { i } > 0 . 5 \}$ ;
2 if $S = \emptyset$ then
3 S ← {arg ma $\mathrm { x } _ { i } \pi _ { i } \}$ ;
4 while min<sub>m</sub> $v _ { m } ( \mathbf { 1 } _ { S } ) < c$ do
5 i<sup>⋆</sup> ← arg max<sub>i/∈S</sub> $\partial U _ { c } ( { \bf 1 } _ { S } ) / \partial \pi _ { i } ;$
6 $S \gets S \cup \{ i ^ { \star } \} ;$
7 return $S ^ { * } = S ;$
```

## D Evidence Diagnostics and Baseline Comparisons

Table 6 defines the operational diagnostics used throughout the evidence experiments. Let $\Phi ( \cdot )$ denote the intervention output compared within a slide (class probability for classification, normalized risk for survival), and let $M ( \cdot )$ denote the aggregate task metric (Macro-F1 or C-index) evaluated after replacing the full bag by the specified subset.

Table 6: Operational S/N/R diagnostics. The C-D gap is an optimization-aligned recoverability diagnostic; keep/remove interventions and CAMELYON localization provide additional modelrelative and spatial checks.
<table><tr><td>Diagnostic</td><td>Definition</td><td>Evidence role</td></tr><tr><td>Keep-only drop</td><td> $M ( X ) - M ( X _ { S } )$  after retaining only S</td><td>Sufficiency; lower is bet- ter</td></tr><tr><td>Evidence sufficiency</td><td> $M ( X _ { S } )$  , the task metric of the evidence-only bag</td><td>Sufficiency; higher is better</td></tr><tr><td>Complement degradation</td><td> $\bar { M ( X ) } - M ( X { \lrcorner } s )$  after removing S</td><td>Necessity; higher is bet- ter</td></tr><tr><td>C-D gap</td><td> $\mathbb { E } _ { X } \left[ | \Phi ( X _ { \pi } ) - \Phi ( X _ { S ( \pi ) } ) | \right]$ </td><td>Recoverability; lower is better</td></tr></table>

Table 7 provides a compact BRACS diagnostic for the S/N/R metrics. This appendix groups the evidence diagnostics with budget-matched and post-hoc comparisons, so each S/N/R claim is accompanied by the table used to support it. On BRACS, DTFD-MIL+GCE reaches 0.484 complement degradation and HDMIL+GCE has the highest evidence sufficiency (0.685), while the C-D gap remains below 0.012 for every host backbone. This compact table gives single-dataset intuition before the larger three-dataset diagnostic table: different backbones trade off Necessity and Sufficiency, but all recovered subsets remain close to their continuous selectors.

Table 7: BRACS evidence diagnostics. Single-column S/N/R summary for GCE-wrapped backbones.
<table><tr><td>Method</td><td>C-D Gap↓</td><td>Compl. Degr.↑</td><td>Evid. Suff.↑</td></tr><tr><td> $\mathbf { A B M I L } { + } G C E$  CLAM-SB+GCE</td><td> $0 . 0 1 1 { \scriptstyle \pm 0 . 0 0 4 }$   $0 . 0 0 6 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 2 3 0 { \pm } 0 . 0 4 2$   $0 . 3 3 2 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $0 . 4 3 0 { \scriptstyle \pm 0 . 0 6 8 }$   $0 . 5 6 4 { \scriptstyle \pm 0 . 0 7 7 }$ </td></tr><tr><td>TransMIL+GCE</td><td> $0 . 0 0 6 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 2 2 1 { \scriptstyle \pm 0 . 0 4 7 }$ </td><td> $0 . 4 8 3 { \pm } 0 . 0 8 4$ </td></tr><tr><td>DSMIL+GCE</td><td> $0 . 0 0 8 { \pm } 0 . 0 0 3$ </td><td> $0 . 2 4 3 { \pm } 0 . 0 4 6$ </td><td> $0 . 4 6 8 { \pm } 0 . 0 8 2$ </td></tr><tr><td>DTFD-MIL+GCE</td><td> $0 . 0 0 5 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 4 8 4 } \pm 0 . 0 8 2$ </td><td>0.645±0.091</td></tr><tr><td> $\mathbf { I B M L + } G C E$ </td><td> $0 . 0 0 5 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $\underline { { 0 . 4 5 0 } } \pm 0 . 0 7 6$ </td><td>0.660±0.088</td></tr><tr><td> $\mathbf { M H I M - M I L + } G C E$ </td><td> $\mathbf { 0 . 0 0 4 } 2 0 . 0 0 1$ </td><td> $0 . 2 3 6 { \pm } 0 . 0 4 4$ </td><td> $0 . 4 7 5 { \scriptstyle \pm 0 . 0 8 1 }$ </td></tr><tr><td> $\mathbf { C A M I L + } G C E$ </td><td> $0 . 0 0 6 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 3 4 6 { \pm } 0 . 0 6 5$ </td><td> $0 . 5 7 0 { \scriptstyle \pm 0 . 0 9 5 }$ </td></tr><tr><td> $\mathrm { H D M L } { + } G C E$ </td><td> $\mathbf { 0 . 0 0 4 } { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 4 4 1 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td> $\mathbf { 0 . 6 8 5 { \scriptstyle \pm 0 . 0 9 4 } }$ </td></tr></table>

Table 8 reports the three S/N/R metrics for each backbone after GCE training and discrete recovery. The C-D gap is an optimization-aligned recoverability diagnostic: HDMIL achieves 0.004 on BRACS and 0.002 on NSCLC, indicating that the continuous selector and recovered discrete subset remain close under the reported recovery protocol. Complement degradation (Necessity) reaches 0.484 for DTFD-MIL on BRACS, meaning removing the evidence subset drops performance by nearly half—the selected patches are genuinely informative. Evidence sufficiency reaches 0.685 for HDMIL on BRACS, indicating that the evidence subset alone retains most of the full-bag prediction. The pattern is consistent across datasets: backbones with stronger attention mechanisms (DTFD-MIL, IBMIL, HDMIL) achieve higher Necessity and Sufficiency scores, suggesting that GCE amplifies the backbone’s existing ability to identify informative patches. The important point is not that one host backbone dominates every metric. Instead, every host receives an explicit evidence interface, allowing S/N/R diagnostics to be compared across architectures that otherwise expose different attention or instance scores.

Table 8: GCE evidence diagnostics per backbone. C-D Gap: continuous–discrete prediction gap (Recoverability); Complement Degradation: performance drop when evidence is removed (Necessity); Evidence Sufficiency: performance retained by evidence alone (Sufficiency).
<table><tr><td></td><td colspan="3">BRACS</td><td colspan="3">NSCLC</td><td colspan="3">PANDA</td></tr><tr><td>Method</td><td>C-D Gap</td><td>Compl. Degr.</td><td>Evid. Suff.</td><td>C-D Gap</td><td>Compl. Degr.</td><td>Evid. Suff.</td><td>C-D Gap</td><td>Compl. Degr.</td><td>Evid. Suff.</td></tr><tr><td>ABMIL</td><td>0.011±0.004</td><td>0.230±0.042</td><td>0.430±0.068</td><td>0.008±0.003</td><td>0.280±0.054</td><td>0.500±0.075</td><td>0.012±0.004</td><td>0.153±0.038</td><td>0.338±0.059</td></tr><tr><td>CLAM-SB</td><td>0.006±0.003</td><td>0.332±0.051</td><td>0.564±0.077</td><td>0.004±0.001</td><td>0.313±0.058</td><td>0.550±0.081</td><td>0.006±0.002</td><td>0.263±0.044</td><td>0.496±0.071</td></tr><tr><td>TransMIL</td><td>0.006±0.002</td><td>0.221±0.047</td><td>0.483±0.084</td><td>0.002±0.001</td><td>0.271±0.050</td><td>0.537±0.092</td><td>0.006±0.002</td><td>0.144±0.033</td><td>0.370±0.065</td></tr><tr><td>DSMIL</td><td>0.008±0.003</td><td>0.243±0.046</td><td>0.468±0.082</td><td>0.004±0.001</td><td>0.310±0.049</td><td>0.559±0.088</td><td>0.008±0.003</td><td>0.150±0.031</td><td>0.370±0.058</td></tr><tr><td>DTFD-MIL</td><td>0.005±0.002</td><td>0.484±0.082</td><td>0.645±0.091</td><td>0.003±0.001</td><td>0.440±0.081</td><td>0.612±0.097</td><td>0.005±0.001</td><td>0.437±0.074</td><td>0.603±0.086</td></tr><tr><td>IBMIL</td><td>0.005±0.002</td><td>0.450±0.076</td><td>0.660±0.088</td><td>0.003±0.001</td><td>0.431±0.075</td><td>0.651±0.102</td><td>0.005±0.002</td><td>0.362±0.058</td><td>0.575±0.079</td></tr><tr><td>MHIM-MIL</td><td>0.004±0.001</td><td>0.236±0.044</td><td>0.475±0.081</td><td>0.002±0.001</td><td>0.273±0.052</td><td>0.517±0.086</td><td>0.004±0.001</td><td>0.180±0.035</td><td>0.422±0.066</td></tr><tr><td>CAMIL</td><td>0.006±0.002</td><td>0.346±0.065</td><td>0.570±0.095</td><td>0.002±0.001</td><td>0.408±0.078</td><td>0.651±0.114</td><td>0.005±0.002</td><td>0.285±0.051</td><td>0.507±0.082</td></tr><tr><td>HDMIL</td><td>0.004±0.002</td><td>0.441±0.071</td><td>0.685±0.094</td><td>0.002±0.001</td><td>0.418±0.074</td><td>0.666±0.110</td><td>0.003±0.001</td><td>0.368±0.062</td><td>0.611±0.095</td></tr></table>

## D.1 Same-Budget Comparison

Table 9 compares selection methods under a fixed 5% budget on BRACS. Attention top-k achieves 0.597 Macro-F1 and 0.151 complement degradation; discrete GCE reaches 0.748 and 0.412. The prediction gap drops from 0.029 to 0.004. Under a fixed budget, GCE improves all three S/N/R criteria rather than simply selecting more patches. This is the fairest control for sparse evidence methods because it removes subset size as a confounder. The remaining gap is therefore attributable to how the subset is selected and recovered, not to using more tissue.

## D.2 Stability and Post-hoc Comparison

GCE evidence is more stable than attention under stochastic perturbation: Jaccard overlap between evidence sets across random seeds rises from 0.329 (attention) to 0.606 (GCE), and prediction flips fall from 0.128 to 0.045 (Tables 10 and 11). Compared with post-hoc attribution methods at matched subset sizes, GCE reaches 0.722 keep-only Macro-F1 and 0.176 remove drop, vs. 0.662/0.048 for integrated gradients and 0.668/0.052 for occlusion. The joint training of selection and prediction in GCE produces evidence that is both more faithful and more stable than post-hoc methods.

Table 10 separates two forms of stability. The Jaccard score measures whether the selected evidence set itself is stable, while prediction flip measures whether stochastic perturbations change the slide-

Table 9: Same-budget comparison on BRACS. All methods select approximately 5% of each bag, except the soft GCE row which uses the continuous selector before discretization.
<table><tr><td>Subset Rule</td><td>Macro-F1</td><td>Prediction Gap↓ Compl. Degr.↑</td><td></td><td> $\mathrm { E v i d . } \mathrm { S u f f . } \uparrow$ </td></tr><tr><td>Random-k</td><td> $0 . 4 3 1 { \pm } 0 . 0 5 4$ </td><td> $0 . 0 4 3 { \pm } 0 . 0 1 2$ </td><td> $0 . 0 4 8 { \pm } 0 . 0 1 5$ </td><td> $0 . 2 9 5 { \scriptstyle \pm 0 . 0 4 7 }$ </td></tr><tr><td>Attention top-k</td><td> $0 . 5 9 7 { \scriptstyle \pm 0 . 0 4 5 }$ </td><td> $0 . 0 2 9 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 1 5 1 { \scriptstyle \pm 0 . 0 3 6 }$ </td><td> $0 . 4 3 4 { \pm } 0 . 0 6 9$ </td></tr><tr><td>Gradient top-k</td><td> $0 . 6 1 3 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $0 . 0 2 5 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 1 6 6 { \pm } 0 . 0 3 1$ </td><td> $0 . 4 5 4 { \scriptstyle \pm 0 . 0 7 4 }$ </td></tr><tr><td>Occlusion top-k</td><td> $0 . 6 2 8 { \pm } 0 . 0 3 8$ </td><td> $0 . 0 2 2 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 1 9 6 { \pm } 0 . 0 3 9$ </td><td> $0 . 4 7 8 { \pm } 0 . 0 8 1$ </td></tr><tr><td>GCE soft evidence</td><td> $\mathbf { 0 . 7 5 4 } \pm 0 . 0 2 9$ </td><td> $\mathbf { 0 . 0 0 3 } { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $\underline { { 0 . 4 0 5 } } \pm 0 . 0 7 8$ </td><td> $\mathbf { 0 . 6 7 0 { \overset { . } { \bot } } 0 . 0 9 5 }$ </td></tr><tr><td>GCE discrete evidence</td><td> $\underline { { 0 . 7 4 8 } } \pm 0 . 0 3 2$ </td><td> $0 . 0 0 4 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 4 1 2 { \overset { . } { = } } 0 . 0 8 6 }$ </td><td> $\underline { { 0 . 6 5 9 } } \pm 0 . 1 0 8$ </td></tr></table>

Table 10: Evidence stability under stochastic perturbation. Jaccard measures overlap between evidence sets across runs.
<table><tr><td>Evidence Rule</td><td>Jaccard↑</td><td>Prediction Flip↓</td><td>C-D Gap↓</td></tr><tr><td>Attention top-k</td><td> $0 . 3 2 9 { \pm } 0 . 0 4 3$ </td><td> $0 . 1 2 8 { \pm } 0 . 0 1 5$ </td><td> $0 . 0 2 9 { \scriptstyle \pm 0 . 0 0 8 }$ </td></tr><tr><td>Gradient top-k</td><td> $0 . 3 5 7 { \scriptstyle \pm 0 . 0 4 6 }$ </td><td> $0 . 1 1 2 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 0 2 5 { \scriptstyle \pm 0 . 0 0 6 }$ </td></tr><tr><td>Occlusion top-k</td><td> $0 . 3 8 9 { \pm } 0 . 0 4 9$ </td><td> $0 . 0 9 8 { \pm } 0 . 0 1 1$ </td><td> $0 . 0 2 2 { \scriptstyle \pm 0 . 0 0 5 }$ </td></tr><tr><td>GCE soft thresholded</td><td> $\mathbf { 0 . 6 2 7 { \scriptstyle \pm 0 . 0 7 6 } }$ </td><td> $\mathbf { 0 . 0 4 1 } { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 0 0 3 } { \scriptstyle \pm 0 . 0 0 1 }$ </td></tr><tr><td>GCE discrete recovered</td><td> $\underline { { 0 . 6 0 6 } } \pm 0 . 0 7 9$ </td><td> $\underline { { 0 . 0 4 5 } } \pm 0 . 0 0 6$ </td><td> $\underline { { 0 . 0 0 5 } } \pm 0 . 0 0 2$ </td></tr></table>

level decision. GCE improves both: the recovered discrete subset has 0.606 Jaccard overlap and 0.045 prediction flip rate, compared with 0.329 and 0.128 for attention top-k. This directly supports Recoverability because the discrete subset remains close to the continuous selector across stochastic runs.

Table 11: Comparison with post-hoc explainability methods averaged across nine datasets. All methods use the same evidence fraction.
<table><tr><td>Method</td><td>Keep-only Macro-F1↑</td><td>Remove Drop↑</td><td>Prediction Gap↓</td><td> $\mathrm { C o m p l . D e g r . } \uparrow$ </td><td>Evid. Suff.↑</td><td>Runtime</td></tr><tr><td>Attention top-k</td><td> $0 . 6 4 0 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 0 3 3 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 0 2 9 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 1 0 7 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 3 6 7 { \scriptstyle \pm 0 . 0 5 4 }$ </td><td>1.00×</td></tr><tr><td>CLAM attention</td><td> $0 . 6 4 5 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 0 3 5 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 0 2 7 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 1 1 2 { \pm } 0 . 0 2 5$ </td><td> $0 . 3 7 4 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td>1.10×</td></tr><tr><td>Gradient saliency</td><td> $0 . 6 5 5 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 0 4 2 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 0 2 5 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 1 1 8 { \pm } 0 . 0 2 7$ </td><td> $0 . 3 8 6 { \scriptstyle \pm 0 . 0 6 2 }$ </td><td>1.42×</td></tr><tr><td>Integrated gradients</td><td> $0 . 6 6 2 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 0 4 8 { \pm } 0 . 0 1 3$ </td><td> $0 . 0 2 3 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 1 2 8 { \pm } 0 . 0 3 1$ </td><td> $0 . 3 9 6 { \scriptstyle \pm 0 . 0 6 5 }$ </td><td>3.80×</td></tr><tr><td>Occlusion top-k</td><td> $0 . 6 6 8 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 0 5 2 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 0 2 2 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 1 3 9 { \pm } 0 . 0 3 4$ </td><td> $0 . 4 0 5 { \scriptstyle \pm 0 . 0 6 9 }$ </td><td>8.70×</td></tr><tr><td>GCE discrete</td><td> $\mathbf { 0 . 7 2 2 { \scriptstyle \pm 0 . 0 2 2 } }$ </td><td> $\mathbf { 0 . 1 7 6 { \overset { . } { = } } 0 . 0 3 8 }$ </td><td> $\mathbf { 0 . 0 0 5 } { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 2 7 7 { \scriptstyle \pm 0 . 0 5 6 } }$ </td><td> $\mathbf { 0 . 5 3 3 { \scriptstyle \pm 0 . 0 8 3 } }$ </td><td>1.08×</td></tr></table>

Table 11 shows that post-hoc explanations improve over raw attention but do not close the S/N/R gap. Occlusion reaches 0.668 keep-only Macro-F1 and 0.052 remove drop, whereas GCE reaches 0.722 and 0.176 at comparable subset size. The runtime column also matters: integrated gradients and occlusion require repeated backward or forward passes, while GCE obtains its evidence during the normal model pass. The result is evidence that is both intervention-faithful and cheaper to recover.

## E Survival Prediction and Task Generalization

Table 12 reports the five TCGA survival cohorts evaluated with C-index.

Table 12: Survival prediction on five TCGA cohorts (C-index, 5-fold cross-validation). Baseline rows report absolute mean±std; +GCE rows report the change from the preceding baseline.
<table><tr><td>Method</td><td>KIRC</td><td>KIRP</td><td>LUAD</td><td>STAD</td><td>UCEC</td></tr><tr><td>ABMIL +GCE ∆</td><td>0.724±0.031 △0.000</td><td>0.779±0.035 △0.007</td><td>0.641±0.039 △0.013</td><td>0.576±0.028 △0.018</td><td>0.696±0.033 △0.007</td></tr><tr><td>CLAM-SB +GCE ∆</td><td>0.715±0.028 △0.008</td><td>0.758±0.032 △0.000</td><td>0.644±0.035 △0.022</td><td>0.603±0.025 △0.000</td><td>0.746±0.030 △0.006</td></tr><tr><td>TransMIL +GCE∆</td><td>0.669±0.029 △0.012</td><td>0.818±0.034 △0.008</td><td>0.633±0.037 △0.015</td><td>0.597±0.026 △0.016</td><td>0.721±0.032 △0.019</td></tr><tr><td>DSMIL +GCE∆</td><td>0.667±0.028 △0.031</td><td>0.743±0.032 △0.023</td><td>0.612±0.035 △0.008</td><td>0.588±0.025 △0.019</td><td>0.662±0.030 △0.037</td></tr><tr><td>DTFD-MIL +GCE ∆</td><td>0.707±0.027 △0.007</td><td>0.777±0.030 △0.007</td><td>0.549±0.033 △0.018</td><td>0.614±0.024 △0.008</td><td>0.706±0.029 △0.000</td></tr><tr><td>IBMIL +GCE∆</td><td>0.680±0.025 △0.014</td><td>0.753±0.029 △0.008</td><td>0.625±0.032 △0.030</td><td>0.595±0.023 △0.010</td><td>0.643±0.027 △0.048</td></tr><tr><td>MHIM-MIL +GCE∆</td><td>0.733±0.027 △0.003</td><td>0.758±0.030 △0.021</td><td>0.576±0.033 △0.038</td><td>0.622±0.024 △0.024</td><td>0.712±0.029 △0.011</td></tr><tr><td>CAMIL +GCE∆</td><td>0.665±0.029 △0.006</td><td>0.732±0.034 △0.017</td><td>0.661±0.037 △0.008</td><td>0.590±0.026 ∇0.003</td><td>0.702±0.032 △0.006</td></tr><tr><td>HDMIL +GCE∆</td><td>0.682±0.025 △0.010</td><td>0.753±0.029 △0.023</td><td>0.592±0.032 △0.044</td><td>0.610±0.023 △0.008</td><td>0.750±0.027 △0.006</td></tr></table>

Survival prediction results are reported here to keep the main text focused on the classification benchmark and intervention evidence. GCE improves average C-index by 0.014 across the survival grid, with visible gains on LUAD (HDMIL △0.044, MHIM-MIL △0.038) and UCEC (DSMIL △0.037, IBMIL △0.048). The survival results are consistent with the risk-pathway interpretation in Appendix B: a selected patch subset can activate one or more morphology-linked risk pathways, and the Cox head maps these activations to scalar risk. Evidence-oriented training therefore does not require changing the host backbone and remains useful beyond categorical slide labels. The gains are not uniform across cohorts, which is expected because survival labels are noisier and censoring reduces supervision strength. Nevertheless, the table shows that optimizing S/N/R does not harm risk prediction: only one cell has a small negative change (CAMIL on STAD, ▽0.003), while the remaining cells are non-negative.

## F Localization and Additional Experiments

The experiments in this section provide supporting checks that are not part of the main nine-dataset prediction grid. CAMELYON-16 localization uses pixel-level annotations as an external spatial validation of evidence maps, while the paired tests summarize fold-level uncertainty for the reported prediction gains.

## F.1 Localization on CAMELYON-16

Table 13 reports localization quality on CAMELYON-16 using pixel-level annotations. The comparison evaluates whether evidence gates improve the spatial agreement between model evidence maps and annotated tumor regions. Delta rows are placed immediately below each host backbone to make the effect of adding GCE visible without a separate explanatory table.

Table 13: Localization metrics on CAMELYON-16 with pixel-level annotations. Higher Dice, specificity, and FROC indicate better spatial agreement between evidence maps and annotated tumor regions. Delta rows report the gain after adding GCE to the same host backbone.
<table><tr><td>Method</td><td>Dice↑</td><td>Specificity↑</td><td>FROC↑</td></tr><tr><td>ABMIL  $\mathbf { A B M I L } { + } G C E$   $\Delta$ </td><td> $0 . 4 2 1 { \scriptstyle \pm 0 . 0 3 4 }$   $0 . 5 4 7 { \scriptstyle \pm 0 . 0 2 9 }$   $\uparrow 0 . 1 2 6 { \pm } 0 . 0 1 8$ </td><td> $0 . 9 7 6 { \scriptstyle \pm 0 . 0 0 8 }$   $0 . 9 9 2 { \scriptstyle \pm 0 . 0 0 5 }$   $\uparrow 0 . 0 1 6 { \pm } 0 . 0 0 6$ </td><td> $0 . 3 9 8 { \pm } 0 . 0 2 9$   $0 . 4 8 7 { \scriptstyle \pm 0 . 0 2 4 }$   $\uparrow 0 . 0 8 9 { \pm } 0 . 0 1 6$ </td></tr><tr><td>CLAM-SB  $\mathrm { C L A M - S B } { + } G C E$   $\Delta$ </td><td> $0 . 4 5 9 { \pm } 0 . 0 3 1$   $0 . 5 7 2 { \scriptstyle \pm 0 . 0 2 6 }$   $\uparrow 0 . 1 1 3 { \pm } 0 . 0 1 7$ </td><td> $0 . 9 8 7 { \scriptstyle \pm 0 . 0 0 6 }$   $0 . 9 9 4 { \scriptstyle \pm 0 . 0 0 4 }$   $\uparrow 0 . 0 0 7 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 4 2 6 { \scriptstyle \pm 0 . 0 2 7 }$   $0 . 5 0 3 { \scriptstyle \pm 0 . 0 2 2 }$   $\uparrow 0 . 0 7 7 { \scriptstyle \pm 0 . 0 1 5 }$ </td></tr><tr><td>TransMIL  $\mathrm { T r a n s M I L } { + } G C E$   $\Delta$ </td><td> $0 . 1 0 3 { \pm } 0 . 0 2 4$   $0 . 3 9 8 { \pm } 0 . 0 3 4$   $\uparrow 0 . 2 9 5 { \pm } 0 . 0 2 6$ </td><td> $0 . 9 9 9 { \scriptstyle \pm 0 . 0 0 2 }$   $0 . 9 9 9 { \scriptstyle \pm 0 . 0 0 2 }$   $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 4 8 7 { \pm } 0 . 0 1 9$   $0 . 5 4 7 { \scriptstyle \pm 0 . 0 2 8 }$   $\uparrow 0 . 0 6 0 { \pm } 0 . 0 2 4$ </td></tr><tr><td>DSMIL  $\mathrm { D S M I L } { + } G C E$   $\Delta$ </td><td> $0 . 2 5 9 { \pm } 0 . 0 3 2$   $0 . 4 7 6 { \scriptstyle \pm 0 . 0 2 7 }$   $\uparrow 0 . 2 1 7 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 8 6 3 { \scriptstyle \pm 0 . 0 2 4 }$   $0 . 9 5 4 { \pm } 0 . 0 1 2$   $\uparrow 0 . 0 9 1 { \pm } 0 . 0 1 8$ </td><td> $0 . 4 5 1 { \scriptstyle \pm 0 . 0 3 1 }$   $0 . 5 2 7 { \scriptstyle \pm 0 . 0 2 4 }$   $\uparrow 0 . 0 7 6 { \pm } 0 . 0 1 8$ </td></tr><tr><td>DTFD-MIL  $\mathrm { D T F D - M I L } { + } G C E$   $\Delta$ </td><td> $0 . 5 2 5 { \scriptstyle \pm 0 . 0 2 9 }$   $0 . 6 0 4 { \scriptstyle \pm 0 . 0 2 4 }$   $\uparrow 0 . 0 7 9 { \pm } 0 . 0 1 5$ </td><td> $0 . 9 9 9 { \scriptstyle \pm 0 . 0 0 2 }$   $0 . 9 9 9 { \scriptstyle \pm 0 . 0 0 2 }$   $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 4 7 1 { \scriptstyle \pm 0 . 0 2 6 }$   $0 . 5 1 8 { \pm } 0 . 0 2 2$   $\uparrow 0 . 0 4 7 { \pm } 0 . 0 1 3$ </td></tr><tr><td>CAMIL  $\mathbf { C A M I L + } G C E$   $\Delta$ </td><td> $0 . 5 1 5 { \scriptstyle \pm 0 . 0 3 0 }$   $0 . 5 8 9 { \pm } 0 . 0 2 5$   $\uparrow 0 . 0 7 4 { \pm } 0 . 0 1 4$ </td><td> $0 . 9 8 0 { \scriptstyle \pm 0 . 0 0 8 }$   $0 . 9 9 4 { \scriptstyle \pm 0 . 0 0 5 }$   $\uparrow 0 . 0 1 4 { \pm } 0 . 0 0 6$ </td><td> $0 . 4 6 1 { \scriptstyle \pm 0 . 0 2 7 }$   $0 . 5 1 2 { \scriptstyle \pm 0 . 0 2 3 }$   $\uparrow 0 . 0 5 1 { \pm } 0 . 0 1 4$ </td></tr></table>

The localization results provide an external check on the evidence maps. TransMIL has high specificity and relatively strong baseline FROC (0.487) but poor baseline Dice (0.103), suggesting that its detections are conservative and spatially sparse; adding GCE raises Dice to 0.398 and further improves FROC to 0.547 without reducing specificity. For stronger localization baselines such as DTFD-MIL and CAMIL, GCE still increases Dice by 0.079 and 0.074, respectively. These gains suggest that anchor-grounded recovery improves spatial alignment rather than simply selecting more tissue.

## F.2 Significance Tests

Tables 14 and 15 report paired tests for classification and survival benchmarks.

Table 14: Paired significance tests for classification benchmarks comparing each baseline with its +GCE variant.
<table><tr><td>Method</td><td>BRACS</td><td>NSCLC</td><td>PANDA</td><td>BRCA</td></tr><tr><td> $\mathbf { A B M I L } { + } G C E$ </td><td> $p = 0 . 0 0 3 ^ { \ast \ast }$ </td><td> $p = 0 . 0 1 8 ^ { * }$ </td><td> $p = 0 . 4 1 2$ </td><td> $p = 0 . 0 2 4 ^ { * }$ </td></tr><tr><td> $\mathrm { C L A M - S B } { + } G C E$ </td><td> $p = 0 . 0 4 1 ^ { * }$ </td><td> $p = 0 . 0 2 9 ^ { \ast }$ </td><td> $p = 0 . 0 3 8 ^ { \ast }$ </td><td> $p = 0 . 0 1 1 ^ { * }$ </td></tr><tr><td> $\mathrm { T r a n s M I L } { + } G C E$ </td><td> $p = 0 . 0 1 2 ^ { \ast }$ </td><td> $p = 0 . 4 9 8$ </td><td> $p = 0 . 3 4 7$ </td><td> $p = 0 . 1 0 6$ </td></tr><tr><td> $\mathrm { D S M I L } { + } G C E$ </td><td> $p = 0 . 0 2 7 ^ { \ast }$ </td><td> $p = 0 . 0 3 4 ^ { \ast }$ </td><td> $p = 0 . 0 5 8$ </td><td> $p = 0 . 0 4 3 ^ { * }$ </td></tr><tr><td> $\mathrm { D T F D - M I L } { + } G C E$ </td><td> $p = 0 . 0 0 4 ^ { \ast \ast }$ </td><td> $p = 0 . 5 2 3$ </td><td> $p = 0 . 0 2 9 ^ { \ast }$ </td><td> $p = 0 . 1 8 7$ </td></tr><tr><td> $\mathrm { I B M I L } { + } G C E$ </td><td> $p = 0 . 0 7 8$ </td><td> $p = 0 . 0 4 6 ^ { \ast }$ </td><td> $p = 0 . 0 0 8 ^ { \ast \ast }$ </td><td> $p = 0 . 0 0 6 ^ { \ast \ast }$ </td></tr><tr><td> $\mathbf { M H M - M I L + } G C E$ </td><td> $p = 0 . 0 2 1 ^ { \ast }$ </td><td> $p = 0 . 0 9 2$ </td><td> $p = 0 . 0 1 4 ^ { * }$ </td><td> $p = 0 . 0 1 8 ^ { * }$ </td></tr><tr><td> $\mathbf { C A M I L + } G C E$ </td><td> $p = 0 . 0 0 9 ^ { \ast \ast }$ </td><td> $p = 0 . 2 2 4$ </td><td> $p = 0 . 4 1 2$ </td><td> $p = 0 . 3 6 7$ </td></tr><tr><td> $\mathrm { H D M L } { + } G C E$ </td><td> $p = 0 . 0 0 5 ^ { \ast \ast }$ </td><td> $p = 0 . 0 6 7$ </td><td> $p = 0 . 4 2 8$ </td><td> $p = 0 . 0 3 9 ^ { \ast }$ </td></tr></table>

The classification significance table serves as a robustness check rather than the main evidence. Most BRACS and TCGA-BRCA comparisons pass the paired threshold, while some PANDA and NSCLC cells do not because the baseline variance is higher or the absolute gain is smaller. This is consistent with the main table: GCE gives positive average gains, but the size of the prediction gain depends on the host backbone and dataset.

Table 15: Paired significance tests for survival benchmarks comparing each baseline with its +GCE variant.
<table><tr><td>Method</td><td>KIRC</td><td>KIRP</td><td>LUAD</td><td>STAD</td><td>UCEC</td></tr><tr><td>ABMIL+GCE</td><td> $p = 0 . 5 2 3$ </td><td> $p = 0 . 0 8 7$ </td><td> $p = 0 . 0 4 1 ^ { * }$ </td><td> $p = 0 . 0 1 8 ^ { * }$ </td><td> $p = 0 . 0 9 4$ </td></tr><tr><td>CLAM-SB+GCE</td><td> $p = 0 . 0 6 2$ </td><td> $p = 0 . 4 9 8$ </td><td> $p = 0 . 0 1 3 ^ { * }$ </td><td> $p = 0 . 4 6 7$ </td><td> $p = 0 . 1 0 3$ </td></tr><tr><td>TransMIL+GCE</td><td> $p = 0 . 0 3 8 ^ { \ast }$ </td><td> $p = 0 . 0 7 3$ </td><td> $p = 0 . 0 2 9 ^ { \ast }$ </td><td> $p = 0 . 0 2 4 ^ { \ast }$ </td><td> $p = 0 . 0 1 1 ^ { \ast }$ </td></tr><tr><td>DSMIL+GCE</td><td> $p = 0 . 0 0 4 ^ { \ast \ast }$ </td><td> $p = 0 . 0 1 4 ^ { \ast }$ </td><td> $p = 0 . 0 8 7$ </td><td> $p = 0 . 0 1 9 ^ { \ast }$ </td><td> $p = 0 . 0 0 3 ^ { \ast \ast }$ </td></tr><tr><td>DTFD-MIL+GCE</td><td> $p = 0 . 0 8 2$ </td><td> $p = 0 . 0 9 4$ </td><td> $p = 0 . 0 2 2 ^ { \ast }$ </td><td> $p = 0 . 0 6 9$ </td><td> $p = 0 . 5 1 2$ </td></tr><tr><td>IBMIL+GCE</td><td> $p = 0 . 0 2 9 ^ { \ast }$ </td><td> $p = 0 . 0 8 7$ </td><td> $p = 0 . 0 0 6 ^ { \ast \ast }$ </td><td> $p = 0 . 0 4 1 ^ { * }$ </td><td> $p = 0 . 0 0 2 ^ { \ast \ast }$ </td></tr><tr><td>MHIM-MIL+GCE</td><td> $p = 0 . 3 9 8$ </td><td> $p = 0 . 0 1 8 ^ { \ast }$ </td><td> $p = 0 . 0 0 3 ^ { \ast \ast }$ </td><td> $p = 0 . 0 1 1 ^ { \ast }$ </td><td> $p = 0 . 0 4 6 ^ { \ast }$ </td></tr><tr><td> $\mathbf { C A M I L } { + } G C E$ </td><td> $p = 0 . 0 9 4$ </td><td> $p = 0 . 0 3 4 ^ { \ast }$ </td><td> $p = 0 . 0 8 7$ </td><td> $p = 0 . 4 1 2$ </td><td> $p = 0 . 1 0 3$ </td></tr><tr><td> $\mathrm { H D M L } { + } G C E$ </td><td> $p = 0 . 0 6 4$ </td><td> $p = 0 . 0 1 9 ^ { \ast }$ </td><td> $p = 0 . 0 0 1 ^ { \ast \ast }$ </td><td> $p = 0 . 0 6 7$ </td><td> $p = 0 . 1 0 8$ </td></tr></table>

The survival significance table shows the same mixed but directional pattern. The strongest evidence appears on LUAD, UCEC, and several DSMIL/IBMIL/MHIM-MIL cells, where the C-index gains are larger. KIRC and KIRP contain more cells above the paired-test threshold, which matches the smaller absolute gains reported in the survival table. The survival results therefore support the method’s portability, while the evidence diagnostics remain the primary evidence-quality analysis.

## G Ablation Studies and Sensitivity Analysis

This section groups the sensitivity analyses that explain how the evidence selector behaves under different budgets, selector parameterizations, and grounding variants. The tables are placed next to their interpretation so that each design choice is tied back to one of Sufficiency, Necessity, or Recoverability.

## G.1 Budget Sweep

Table 16 reports the full budget sweep used to choose $\rho = 0 . 0 5 .$

Table 16 explains why the main configuration uses $\rho = 0 . 0 5$ . Increasing the budget from 0.01 to 0.05 improves Macro-F1 from 0.711 to 0.748 and raises evidence sufficiency from 0.440 to 0.659. Larger budgets produce almost no prediction gain and only marginal increases in complement degradation, while the evidence fraction grows substantially. Thus $\rho = 0 . 0 5$ is the knee point: it preserves Sufficiency and Necessity without drifting toward full-bag evidence.

## G.2 Recovery Hyperparameter Scope

The reported discrete recovery protocol fixes the threshold at $\tau = 0 . 5$ and the coverage target at $c = 0 . 9 5$ for all datasets. These values are used as operating rules rather than per-dataset tuning knobs: thresholding first makes the continuous selector discrete, and repair then adds patches only when the

Table 16: Effect of evidence budget $\rho$ on BRACS. Gains plateau beyond $\rho = 0 . 0 5$ while the evidence fraction continues to increase.
<table><tr><td> $\rho$ </td><td>Macro-F1</td><td> $\mathrm { E v i d . } \mathrm { F r a c . }$ </td><td> $\mathrm { C - D \ G a p { \downarrow } }$ </td><td>Compl. Degr.↑</td><td> $\mathrm { E v i d . } \mathrm { S u f f . } \uparrow$ </td></tr><tr><td>0.01</td><td> $0 . 7 1 1 { \pm } 0 . 0 4 8$ </td><td> $0 . 0 1 3 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 0 1 4 { \pm } 0 . 0 0 5$ </td><td> $0 . 2 5 3 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $0 . 4 4 0 { \scriptstyle \pm 0 . 0 6 4 }$ </td></tr><tr><td>0.02</td><td> $0 . 7 3 0 { \scriptstyle \pm 0 . 0 4 2 }$ </td><td> $0 . 0 2 4 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 0 0 9 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 3 3 2 { \scriptstyle \pm 0 . 0 6 7 }$ </td><td> $0 . 5 5 5 { \pm } 0 . 0 8 9$ </td></tr><tr><td>0.05</td><td> $\pm 0 . 7 4 8 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $0 . 0 5 1 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $\mathbf { 0 . 0 0 4 } 2 0 . 0 0 1$ </td><td> $0 . 4 1 2 { \scriptstyle \pm 0 . 0 8 6 }$ </td><td> $0 . 6 5 9 { \pm } 0 . 1 0 8$ </td></tr><tr><td>0.10</td><td> $\pm 0 . 7 4 8 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $0 . 0 9 4 { \pm } 0 . 0 1 5$ </td><td> $\mathbf { 0 . 0 0 4 } 2 0 . 0 0 1$ </td><td> $0 . 4 2 0 { \scriptstyle \pm 0 . 0 9 1 }$ </td><td> $0 . 6 6 4 { \scriptstyle \pm 0 . 1 1 2 }$ </td></tr><tr><td>0.20</td><td> $0 . 7 4 7 { \scriptstyle \pm 0 . 0 3 5 }$ </td><td> $0 . 1 8 2 { \pm } 0 . 0 2 2$ </td><td> $\mathbf { 0 . 0 0 4 } 2 0 . 0 0 1$ </td><td> $\pm 0 . 4 2 2 { \scriptstyle \pm 0 . 0 8 9 }$ </td><td> $\mathbf { 0 . 6 6 5 { \scriptstyle \pm 0 . 1 1 0 } }$ </td></tr><tr><td>1.00</td><td> $0 . 7 4 5 { \scriptstyle \pm 0 . 0 3 6 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td></td><td></td><td></td></tr></table>

recovered subset fails the anchor-coverage target. The retained aggregate outputs support the budget and selector sensitivity analyses in Tables 16 and 17, but they do not contain a complete per-slide cache of gates and logits for a post-hoc $\tau \times c$ sweep across all folds. Accordingly, no additional numeric threshold/coverage table is reported. The available sensitivity evidence still addresses the main concern: hard top-k selection at the exact 5% budget has a larger C-D gap (0.031) than the recovered selector (0.005), so Recoverability is not obtained merely by fixing the evidence fraction.

## G.3 Selector Architecture Variants

Table 17 compares selector parameterizations under the same training and recovery protocol. The consecutive selector used by GCE-MIL keeps the evidence fraction close to the target while reducing the continuous-discrete gap to 0.005.

Table 17: Selector architecture ablation on BRACS. The consecutive selector attains the lowest continuous-discrete gap while maintaining a compact evidence fraction.
<table><tr><td>Selector type</td><td>Macro-F1↑</td><td>Evid. Frac.</td><td> $\mathrm { C - D } \mathrm { G a p } \downarrow$ </td><td>Compl. Degr.↑</td></tr><tr><td>Sigmoid + threshold</td><td> $0 . 7 1 9 { \scriptstyle \pm 0 . 0 5 4 }$ </td><td> $0 . 0 6 4 { \pm } 0 . 0 1 8$ </td><td> $0 . 0 2 4 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 2 8 7 { \scriptstyle \pm 0 . 0 6 3 }$ </td></tr><tr><td>Top-k hard  $( k = 5 \% )$ </td><td> $0 . 7 2 3 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td> $0 . 0 5 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 3 1 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 2 5 4 { \pm } 0 . 0 5 8$ </td></tr><tr><td>Gumbel-sigmoid  $( \tau = 1 . 0 )$ </td><td> $0 . 7 2 8 { \pm } 0 . 0 5 3$ </td><td> $0 . 0 7 1 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 0 1 7 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 3 1 2 { \scriptstyle \pm 0 . 0 6 7 }$ </td></tr><tr><td>Concrete relaxation</td><td> $0 . 7 3 1 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td> $0 . 0 6 8 { \pm } 0 . 0 2 0$ </td><td> $0 . 0 1 5 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 3 2 9 { \scriptstyle \pm 0 . 0 6 8 }$ </td></tr><tr><td>Sparsemax</td><td> $0 . 7 2 6 { \scriptstyle \pm 0 . 0 5 4 }$ </td><td> $0 . 0 8 3 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 0 2 0 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 2 9 8 { \pm } 0 . 0 6 6$ </td></tr><tr><td>Entmax-1.5</td><td> $0 . 7 2 9 { \pm } 0 . 0 5 3$ </td><td> $0 . 0 7 6 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 0 1 8 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 3 1 4 { \scriptstyle \pm 0 . 0 6 7 }$ </td></tr><tr><td>Consecutive selector (ours)</td><td> $\mathbf { 0 . 7 4 5 { \scriptstyle \pm 0 . 0 5 5 } }$ </td><td> $0 . 0 5 1 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $\mathbf { 0 . 0 0 5 } { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 4 1 2 } { \scriptstyle \pm 0 . 0 9 1 }$ </td></tr></table>

Table 17 isolates the selector design from the grounding design. Hard top-k enforces the target evidence fraction exactly but has the largest C-D gap (0.031), showing that sparsity alone does not imply recoverability. Continuous relaxations such as Concrete and Gumbel-sigmoid reduce the gap, but the consecutive selector combines a near-target evidence fraction (0.051) with the smallest gap (0.005) and the largest complement degradation (0.412). This is the selector-side evidence for the Recoverability component of S/N/R.

## G.4 Grounding Variant Ablation

The main BRACS configuration uses eight frozen TITAN text anchors. Table 18 compares semantic grounding variants under the same budget and repair settings.

Table 18: Effect of semantic grounding variants averaged across nine datasets.
<table><tr><td>Grounding Variant</td><td>Macro-F1 / C-index</td><td>C-D Gap↓</td><td>Compl. Degr.↑</td><td>Evid. Suff.↑</td></tr><tr><td>No grounding</td><td> $0 . 6 8 5 { \scriptstyle \pm 0 . 0 3 5 }$ </td><td> $0 . 0 2 5 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 1 2 0 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 3 5 0 { \scriptstyle \pm 0 . 0 4 5 }$ </td></tr><tr><td>Random anchors</td><td> $0 . 6 8 2 { \pm } 0 . 0 3 8$ </td><td> $0 . 0 2 8 { \pm } 0 . 0 0 9$ </td><td> $0 . 1 1 5 { \pm } 0 . 0 3 2$ </td><td> $0 . 3 4 5 { \pm } 0 . 0 4 8$ </td></tr><tr><td>Shuffled prompts</td><td> $0 . 6 8 4 { \pm } 0 . 0 3 6$ </td><td> $0 . 0 2 6 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 1 1 8 { \pm } 0 . 0 3 0$ </td><td> $0 . 3 4 8 { \pm } 0 . 0 4 6$ </td></tr><tr><td>Generic pathology prompts</td><td> $0 . 7 0 8 { \pm } 0 . 0 2 2$ </td><td> $0 . 0 1 5 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 2 1 0 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $0 . 4 8 0 { \pm } 0 . 0 3 5$ </td></tr><tr><td>Disease-specific prompts</td><td> $0 . 7 2 5 { \pm } 0 . 0 1 8$ </td><td> $0 . 0 1 0 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 2 9 0 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 5 6 0 { \scriptstyle \pm 0 . 0 2 8 }$ </td></tr><tr><td>TITAN + unconstrained bridge</td><td> $0 . 7 3 2 { \pm } 0 . 0 1 5$ </td><td> $\underline { { 0 . 0 0 8 } } \pm 0 . 0 0 2$ </td><td> $0 . 3 2 0 { \pm } 0 . 0 1 8$ </td><td> $\underline { { 0 . 6 1 0 } } \pm 0 . 0 2 5$ </td></tr><tr><td> $\mathrm { T I T A N } + \mathrm { c o n s t r a i n e d \ b r i d g e \ ( F u l l ) }$ </td><td> $\mathbf { 0 . 7 4 8 { \scriptstyle \pm 0 . 0 1 1 } }$ </td><td> $\mathbf { 0 . 0 0 4 } 2 0 . 0 0 1$ </td><td> $\mathbf { 0 . 4 1 2 { \overset { . } { \bot } } 0 . 0 1 5 }$ </td><td> $\mathbf { 0 . 6 5 9 } { \scriptstyle \pm 0 . 0 1 8 }$ </td></tr></table>

Table 18 shows that grounding quality, not only sparsity, determines evidence quality. Removing grounding or replacing anchors with random/shuffled prompts leaves Macro-F1/C-index around 0.682–0.685 and complement degradation near 0.115–0.120, indicating weak Necessity. Generic

pathology prompts help, but disease-specific prompts give a larger jump in both prediction (0.725) and evidence sufficiency (0.560). The full TITAN plus constrained-bridge variant reaches 0.748 Macro-F1/C-index, 0.004 C-D gap, 0.412 complement degradation, and 0.659 evidence sufficiency; this is the clearest ablation evidence that semantic grounding supports all three S/N/R criteria.

## H Multi-Backbone and Multi-Encoder Generalization

The main text evaluates nine MIL backbones with UNI features. This section evaluates whether the GCE wrapper remains useful when the patch encoder changes, using ResNet-50, ViT-S, and UNI features on BRACS and LUAD.

## H.1 Multi-Backbone Generalization

Table 19 reports the full encoder/backbone generalization study referenced in Appendix H. Each encoder block is placed adjacent to the corresponding BRACS and LUAD results, enabling direct comparison of feature extractors within the same table.

Table 19: Backbone generalization on BRACS classification and LUAD survival prediction.
<table><tr><td>Dataset</td><td>Backbone</td><td>Method</td><td>Baseline</td><td>+GCE</td><td> $\Delta$ </td></tr><tr><td colspan="6">BRACS Macro-F1</td></tr><tr><td>BRACS</td><td>ResNet-50 (ImageNet)</td><td>ABMIL</td><td> $0 . 5 8 1 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td> $0 . 6 5 2 { \pm } 0 . 0 3 8$ </td><td> $0 . 0 7 1 { \scriptstyle \pm 0 . 0 2 2 }$ </td></tr><tr><td>BRACS</td><td>ResNet-50 (ImageNet)</td><td>CLAM-SB</td><td> $0 . 6 4 3 { \pm } 0 . 0 3 8$ </td><td> $0 . 7 0 1 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $0 . 0 5 8 { \pm } 0 . 0 1 9$ </td></tr><tr><td>BRACS</td><td>ResNet-50 (ImageNet)</td><td>TransMIL</td><td> $0 . 6 0 8 { \pm } 0 . 0 4 1$ </td><td> $0 . 6 6 8 { \scriptstyle \pm 0 . 0 3 6 }$ </td><td> $0 . 0 6 0 { \scriptstyle \pm 0 . 0 2 0 }$ </td></tr><tr><td>BRACS</td><td>ResNet-50 (ImageNet)</td><td>DTFD-MIL</td><td> $0 . 6 2 7 { \scriptstyle \pm 0 . 0 3 9 }$ </td><td> $0 . 6 9 5 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $0 . 0 6 8 { \pm } 0 . 0 1 8$ </td></tr><tr><td>BRACS</td><td>ViT-S (SSL pathology)</td><td>ABMIL</td><td> $0 . 6 3 4 { \pm } 0 . 0 5 0$ </td><td> $0 . 7 0 3 { \scriptstyle \pm 0 . 0 4 4 }$ </td><td> $0 . 0 6 9 { \pm } 0 . 0 1 8$ </td></tr><tr><td>BRACS</td><td>ViT-S (SSL pathology)</td><td>CLAM-SB</td><td> $0 . 7 4 2 { \scriptstyle \pm 0 . 0 4 5 }$ </td><td> $0 . 7 6 5 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $0 . 0 2 3 { \pm } 0 . 0 1 4$ </td></tr><tr><td>BRACS</td><td>ViT-S (SSL pathology)</td><td>TransMIL</td><td> $0 . 6 7 6 { \scriptstyle \pm 0 . 0 4 7 }$ </td><td> $0 . 7 1 4 { \pm } 0 . 0 4 3$ </td><td> $0 . 0 3 8 { \pm } 0 . 0 1 6$ </td></tr><tr><td>BRACS</td><td>ViT-S (SSL pathology)</td><td>DTFD-MIL</td><td> $0 . 6 9 1 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td> $0 . 7 5 7 { \pm } 0 . 0 3 8$ </td><td> $0 . 0 6 6 { \pm } 0 . 0 1 7$ </td></tr><tr><td>BRACS</td><td></td><td>ABMIL</td><td> $0 . 7 2 8 { \pm } 0 . 0 3 4$ </td><td></td><td></td></tr><tr><td>BRACS</td><td>UNI (foundation)</td><td>CLAM-SB</td><td> $0 . 7 7 8 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 7 6 4 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 0 3 6 { \pm } 0 . 0 1 3$ </td></tr><tr><td>BRACS</td><td>UNI (foundation)</td><td>TransMIL</td><td> $0 . 7 4 4 { \pm } 0 . 0 3 3$ </td><td> $0 . 7 9 8 { \pm } 0 . 0 2 7$ </td><td> $0 . 0 2 0 { \scriptstyle \pm 0 . 0 1 1 }$ </td></tr><tr><td>BRACS</td><td>UNI (foundation)</td><td>DTFD-MIL</td><td> $0 . 7 6 3 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 7 7 2 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 0 2 8 { \pm } 0 . 0 1 2$ </td></tr><tr><td>LUAD C-index</td><td>UNI (foundation)</td><td></td><td></td><td> $0 . 7 9 5 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 0 3 2 { \scriptstyle \pm 0 . 0 1 2 }$ </td></tr><tr><td colspan="6"></td></tr><tr><td>LUAD</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>ResNet-50 (ImageNet)</td><td>ABMIL</td><td> $0 . 5 9 8 { \pm } 0 . 0 4 3$ </td><td> $0 . 6 2 7 { \scriptstyle \pm 0 . 0 4 0 }$ </td><td>0.029±0.015</td></tr><tr><td>LUAD LUAD</td><td>ResNet-50 (ImageNet)</td><td>CLAM-SB</td><td> $0 . 6 1 3 { \scriptstyle \pm 0 . 0 3 9 }$ </td><td> $0 . 6 4 4 { \pm } 0 . 0 3 6$ </td><td> $0 . 0 3 1 { \pm } 0 . 0 1 4$ </td></tr><tr><td>LUAD</td><td>ResNet-50 (ImageNet)</td><td>TransMIL</td><td> $0 . 6 0 4 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $0 . 6 3 1 { \pm } 0 . 0 3 8$ </td><td> $0 . 0 2 7 { \scriptstyle \pm 0 . 0 1 5 }$ </td></tr><tr><td></td><td>ResNet-50 (ImageNet)</td><td>DTFD-MIL</td><td> $0 . 5 2 7 { \scriptstyle \pm 0 . 0 3 6 }$ </td><td> $0 . 5 6 1 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 0 3 4 { \pm } 0 . 0 1 3$ </td></tr><tr><td>LUAD</td><td>ViT-S (SSL pathology)</td><td>ABMIL</td><td> $0 . 6 4 1 { \pm } 0 . 0 3 9$ </td><td> $0 . 6 5 4 { \pm } 0 . 0 3 6$ </td><td> $0 . 0 1 3 { \pm } 0 . 0 1 3$ </td></tr><tr><td>LUAD</td><td>ViT-S (SSL pathology)</td><td>CLAM-SB</td><td> $0 . 6 4 4 { \pm } 0 . 0 3 5$ </td><td> $0 . 6 6 6 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $0 . 0 2 2 { \scriptstyle \pm 0 . 0 1 2 }$ </td></tr><tr><td>LUAD</td><td>ViT-S (SSL pathology)</td><td>TransMIL</td><td> $0 . 6 3 3 { \scriptstyle \pm 0 . 0 3 7 }$ </td><td> $0 . 6 4 8 { \pm } 0 . 0 3 4$ </td><td> $0 . 0 1 5 { \pm } 0 . 0 1 3$ </td></tr><tr><td>LUAD</td><td>ViT-S (SSL pathology)</td><td>DTFD-MIL</td><td> $0 . 5 4 9 { \pm } 0 . 0 3 3$ </td><td> $0 . 5 6 7 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 0 1 8 { \pm } 0 . 0 1 2$ </td></tr><tr><td>LUAD</td><td>UNI (foundation)</td><td>ABMIL</td><td> $0 . 6 8 2 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 6 9 1 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $0 . 0 0 9 { \scriptstyle \pm 0 . 0 1 0 }$ </td></tr><tr><td>LUAD</td><td>UNI (foundation)</td><td>CLAM-SB</td><td> $0 . 6 8 5 { \pm } 0 . 0 2 8$ </td><td> $0 . 6 9 7 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 0 1 2 { \scriptstyle \pm 0 . 0 1 0 }$ </td></tr><tr><td>LUAD</td><td>UNI (foundation)</td><td>TransMIL</td><td> $0 . 6 7 1 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 6 8 2 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 0 1 1 { \scriptstyle \pm 0 . 0 1 1 }$ </td></tr><tr><td>LUAD</td><td>UNI (foundation)</td><td>DTFD-MIL</td><td> $0 . 5 8 3 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 5 9 9 { \pm } 0 . 0 2 5$ </td><td> $0 . 0 1 6 { \pm } 0 . 0 1 0$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 19 shows that the gain is not tied to UNI alone. On BRACS, GCE improves all four listed backbones for ResNet-50, ViT-S, and UNI features; the gains are largest for weaker encoders such as ResNet-50, where ABMIL improves by 0.071 and DTFD-MIL by 0.068. On LUAD, every encoder/backbone pair also improves, but the gains are smaller with UNI because the baseline risk models are already stronger. This pattern supports the plug-in interpretation: GCE changes the evidence interface and remains useful across representation quality levels.

## I Failure Case Audit

The audit below records where S/N/R evidence selection is least reliable. The taxonomy, quantitative rates, and qualitative interpretation are kept together so that failure modes remain adjacent to the numbers used to define them.

## I.1 Failure Taxonomy

The audit uses six failure types: case-level prediction failure, artifact-driven evidence, low sufficiency, low necessity, semantic mismatch, and severe failure. Case-level prediction failure means the full-bag prediction is incorrect or unreliable. Artifact-driven evidence means selected patches contain scanner artifacts, tissue folds, pen marks, or background. Low sufficiency means the recovered subset fails the keep-only criterion, while low necessity means removing the subset does not meaningfully degrade prediction. Semantic mismatch means selected evidence does not align with the intended anchor concept. Severe failure denotes cases where multiple failure modes occur together.

## I.2 Per-Dataset Failure Rates

Table 20 reports the failure audit across all nine datasets.

Table 20: Failure-case audit across all nine datasets.
<table><tr><td>Dataset</td><td>Slides</td><td>Case Failure (%)</td><td>Artifact Evidence (%)</td><td>Low Sufficiency (%)</td><td>Low Necessity (%)</td><td>Semantic Mismatch (%)</td><td>Severe Failure (%)</td></tr><tr><td>BRACS</td><td>200</td><td>8.0</td><td>3.5</td><td>5.0</td><td>7.0</td><td>4.5</td><td>4.0</td></tr><tr><td>NSCLC</td><td>220</td><td>5.0</td><td>2.3</td><td>3.2</td><td>5.0</td><td>3.2</td><td>2.7</td></tr><tr><td>PANDA</td><td>300</td><td>9.3</td><td>2.7</td><td>6.7</td><td>8.3</td><td>5.0</td><td>4.7</td></tr><tr><td>BRCA</td><td>180</td><td>14.4</td><td>4.4</td><td>9.4</td><td>10.6</td><td>6.7</td><td>6.1</td></tr><tr><td>LUAD</td><td>230</td><td>18.3</td><td>3.9</td><td>7.8</td><td>11.3</td><td>6.1</td><td>7.0</td></tr><tr><td>STAD</td><td>190</td><td>21.6</td><td>4.7</td><td>9.5</td><td>13.2</td><td>6.3</td><td>7.4</td></tr><tr><td>UCEC</td><td>205</td><td>14.6</td><td>3.4</td><td>7.3</td><td>9.8</td><td>5.4</td><td>5.9</td></tr><tr><td>KIRP</td><td>170</td><td>11.8</td><td>2.9</td><td>5.9</td><td>8.2</td><td>4.7</td><td>4.7</td></tr><tr><td>KIRC</td><td>240</td><td>16.3</td><td>3.3</td><td>7.9</td><td>10.4</td><td>5.8</td><td>6.3</td></tr><tr><td>All</td><td>1935</td><td>13.1</td><td>3.4</td><td>6.9</td><td>9.3</td><td>5.3</td><td>5.4</td></tr></table>

Table 20 identifies where the S/N/R assumptions are most fragile. The largest failure rates appear in STAD and LUAD, where prognostic morphology is heterogeneous and full-bag risk prediction is itself harder. Low necessity is more common than low sufficiency in most cohorts, meaning GCE often finds a subset that can predict but the complement may still contain redundant evidence. This reinforces the paper’s motivation: WSI evidence is often multi-source, so evidence methods should report both keep-only and remove interventions rather than only attention localization.

## I.3 Qualitative Failure Examples

Qualitative failure examples are included in Figures 12–17. The most common qualitative failures are artifact-adjacent evidence and semantically plausible but incomplete evidence sets. These cases usually preserve Sufficiency but weaken Necessity, because other regions in the slide contain similar diagnostic patterns.

## J Implementation and Computational Cost

This section combines reproducibility details with the runtime analysis. The cost numbers are interpreted together with the implementation setup because all primary experiments use cached patch features, whereas optional tile prefiltering changes the end-to-end deployment path.

## J.1 Hardware Setup

All experiments use pre-extracted patch features, so the dominant training cost comes from MIL aggregation, selector scoring, anchor response computation, and intervention evaluation. The implementation is written in PyTorch and uses mixed CPU/GPU data loading with list-valued batches. Experiments were run with PyTorch 1.13 on a single NVIDIA RTX 3090 GPU with 24GB memory. Each fold job uses one GPU; patch features are precomputed, so the GPU workload is dominated by the MIL head, selector network, anchor-response computation, and intervention evaluation rather than by patch encoding. This hardware setting matters for interpreting the cost table: the reported speedups come from reducing the number of patches processed by the MIL aggregation stage, not from changing the offline UNI feature extractor.

## J.2 Training Hyperparameters per Dataset

All nine primary datasets use five-fold cross-validation. All results use UNI features [Chen et al., 2024] and identical train/validation splits for each baseline and its GCE wrapper. Unless otherwise stated, tables report mean±standard deviation over validation folds; repeated-seed analyses are used only for stability and perturbation experiments where seed sensitivity is the measured quantity. The BRACS main GCE configuration trains for 15 epochs with AdamW, learning rate $2 \times \bar { 1 0 } ^ { - 4 }$ , weight decay $1 0 ^ { - 5 }$ , gradient clipping at 5.0, and cosine learning-rate scheduling. The same reported loss weights $( \lambda _ { b } = 0 . 1 , \lambda _ { g } = 0 . 5 )$ and operating evidence budget $( \rho = 0 . 0 5 )$ are used across datasets without per-dataset retuning; budget sensitivity is reported in Table 16. The selector temperature is annealed from 1.0 to 0.4, training bags are randomly sampled to at most 512 patches, and validation bags use all available patches in deterministic order. The same split and feature files are used for each host backbone and its GCE wrapper. The most important reproducibility detail is that the task and evidence paths see the same sampled bag during training. This design keeps evidence supervision focused on selector behavior rather than on stochastic differences in sampled patches.

## J.3 Data Preprocessing Pipeline

Each slide is represented by one HDF5 file containing a feature matrix and patch coordinates. The feature matrix has shape N ×1024 for UNI features, and coordinates have shape N ×2. Classification splits are resolved at the slide level, while survival splits are resolved at the case level to avoid leakage across slides from the same patient. The collate function does not stack bags; it returns a list of samples so that each bag can be processed with its own number of patches. This list-valued batching is also why the evidence selector is applied per slide rather than through a fixed-size tensor batch. It preserves each WSI’s natural patch count and avoids padding artifacts in evidence ratios.

## J.4 Patch Sampling Strategy

Training uses random patch sampling when a bag exceeds the configured training maximum. For the BRACS main configuration, at most 512 patches are sampled for training, while validation uses all available patches and deterministic ordering. The same sampled bag is used for the task and evidence paths within a forward pass. This avoids conflating selector quality with stochastic changes in the input bag.

## J.5 Random Seeds and Reproducibility

The minimal-subset motivation analysis uses fixed seed 42 for random top-k baselines. Model training uses fold-specific splits, and repeated-seed estimates are reserved for the stability tables. The code, split CSV files, configuration YAML files, fold-level evaluation scripts, and JSON summaries for intervention metrics will be released after acceptance. The seed-controlled random top-k baseline is included only as a negative control: it measures how often sufficiency appears by chance at the same subset size.

## J.6 Anchor Prompt Protocol

Table 21 records how anchor prompts are specified. Each dataset family uses eight task-specific morphology anchors chosen before training from standard disease and histology concepts. The prompts are embedded once with TITAN and then frozen; they are not selected by validation performance and do not use patch-level concept annotations. This protocol is meant to provide a reproducible semantic prior while avoiding prompt tuning on the evaluation folds.

Table 21: Anchor prompt protocol. Entries list prompt categories rather than all wording variants;   
each row uses eight frozen TITAN text anchors selected before model training.

Dataset family Anchor source Prompt categories / examples Selection rule   
BRACS / TCGA-BRCA Breast pathology prior Gland formation; nuclear pleomorphism; mitotic activity; stromal reaction; necrosis; lymphocytic infiltration; invasive epithelial nests; benign ducts Fixed before folds; no validation tuning   
PANDA Prostate grading prior Benign glands; gland fusion; cribriform architecture; poorly formed glands; solid growth; stromal/background tissue; inflammation; necrosis Fixed before folds; no patch labels   
TCGA-NSCLC / LUAD Lung tumor morphology Malignant epithelial nests; acinar growth; papillary growth; solid growth; necrotic tumor cells; stromal lymphocytes; lepidic-like regions; mucinous regions   
TCGA-STAD Gastric adenocarcinoma morphology Tubular glands; poorly cohesive cells; signet-ring morphology; necrosis; desmoplastic stroma; lymphocytic infiltration; mucin pools; ulcerated tumor surface Fixed before folds; no case outcome tuning   
TCGA-UCEC Endometrial carcinoma morphology Glandular architecture; solid sheets; nuclear atypia; squamous differentiation; necrosis; stromal reaction; mitotic activity; inflammatory infiltrate Fixed before folds; no validation tuning   
TCGA-KIRP / KIRC Renal-cell carcinoma morphology Fixed before folds; no patch labels

## J.7 Added Capacity and Training Overhead

GCE adds trainable capacity through the low-rank adapter, anchor bridge, selector MLP, and classanchor weights. The aggregate experiment artifacts report relative runtime and memory rather than architecture-specific parameter counts; an exact parameter table is therefore omitted because the count varies with the host backbone implementation. Table 22 summarizes the measured overhead and the controls used to separate added capacity from useful grounding.

Table 22: Added capacity and overhead summary. Runtime and memory values are relative to the full-bag host backbone.

Ouestion Supporting evidence Key value Interpretation   
Does the soft wrapper add runtime? Table 23, GCE soft selector + head End-to-end 1.02×; peak memory 1.05× Training/inference overhead from selector and anchor scoring is small with cached features   
Does discrete recovery reduce aggregation cost? Table 23, discrete cached features MIL aggregation 0.22×; peak memory 0.18× Compact subsets reduce the MIL aggregation stage after recovery   
Is the gain only extra capacity? Table 18, random/shuffled anchors C-D gap 0.028/0.026; complement degradation 0.115/0.118 Capacity without meaningful grounding does not close the evidence gap   
Does grounding quality matter? Table 18, generic to full TITAN C-D gap 0.015 → 0.004; degradation 0.210 → 0.412 Better semantic grounding and constrained bridge improve S/N/R diagnostics

## J.8 Cost Summary

Discrete recovery enables efficient aggregation: with cached features, GCE uses 0.22× aggregation time and 0.18× peak memory while retaining 1.002× relative Macro-F1 (Table 23). End-to-end acceleration requires optional tile prefiltering, which skips many non-selected patches before feature extraction; in that mode, end-to-end time falls to 0.20× at 0.989× utility. Thus the headline speedup is an inference-mode option, while the cached-feature setting isolates the savings from MIL aggregation and memory. The result should be interpreted as an inference-mode option rather than the default validation protocol: the main accuracy tables use the standard evaluation path, while Table 23 asks what happens after evidence has been recovered.

## J.9 Full Inference Cost Table

Table 23 reports inference cost relative to the full-bag backbone.

Table 23: Inference cost comparison. Values are relative to the full-bag backbone forward pass.
<table><tr><td>Inference Mode</td><td>Feature Extraction</td><td>MIL Aggregation</td><td>End-to-end Time</td><td>Peak Memory</td><td>Patch Ratio</td><td>Relative Macro-F1</td></tr><tr><td>Full-bag backbone</td><td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.000</td><td>1.000</td></tr><tr><td>GCE soft selector + head</td><td>1.00×</td><td>1.08×</td><td>1.02×</td><td>1.05×</td><td>1.000</td><td>1.027</td></tr><tr><td>GCE discrete, cached features</td><td>1.00×</td><td>0.22×</td><td>0.92×</td><td>0.18×</td><td>0.056</td><td>1.002</td></tr><tr><td>GCE discrete + tile prefilter</td><td>0.18×</td><td>0.22×</td><td>0.20×</td><td>0.18×</td><td>0.056</td><td>0.989</td></tr><tr><td>Attention top-k re-score</td><td>1.00×</td><td>0.21×</td><td>0.95×</td><td>0.18×</td><td>0.056</td><td>0.902</td></tr><tr><td>Occlusion explanation</td><td>1.00×</td><td>8.70×</td><td>1.75×</td><td>1.00×</td><td>0.056</td><td>0.915</td></tr></table>

Table 23 separates two deployment regimes. If all patch features are already cached, discrete GCE mainly reduces MIL aggregation cost, lowering aggregation time to 0.22× and memory to 0.18× while preserving relative Macro-F1. If tile prefiltering is available, the system can skip many patch computations and reach 0.20× end-to-end time, but with a small utility loss (0.989×). This tradeoff is useful clinically: cached-feature inference favors fidelity, whereas prefiltering favors throughput.

## J.10 Runtime Compared with Post-hoc Methods

Table 11 reports post-hoc runtime ratios. Occlusion requires 8.70× runtime, while GCE discrete evidence costs 1.08× in the same comparison. The difference arises because post-hoc methods explain a fixed trained predictor after the fact, while GCE amortizes evidence selection during training and exposes the selector during inference.

## K Dataset Details

This section records the source, task definition, and preprocessing assumptions for every dataset used in the benchmark. The nine primary datasets define the prediction benchmark; CAMELYON-16 is used only for external localization analysis.

## K.1 Dataset Sources, Splits, and Labels

Table 24 records the dataset card for the nine-dataset benchmark. The four classification datasets are evaluated with Macro-F1, and the five survival cohorts are evaluated with C-index. Slides are split at the patient level whenever patient identifiers are available; slides from the same patient are not shared across train, validation, and test partitions. The dataset mix is intentionally heterogeneous: BRACS and PANDA test diagnostic grading, TCGA-BRCA and TCGA-NSCLC test subtype classification, and the five TCGA survival cohorts test whether evidence selection remains compatible with scalar risk prediction.

Table 24: Dataset details for the nine-dataset benchmark. Counts and diagnostic descriptions summarize the benchmark protocol.
<table><tr><td>Dataset</td><td>Source</td><td>Task</td><td>Split Unit</td><td>Size / Label Distribution</td></tr><tr><td>BRACS</td><td>BRACS public benchmark</td><td>Classification</td><td>Slide</td><td>525 WSIs; 7 fine-grained breast lesion categories</td></tr><tr><td>PANDA</td><td>PANDA challenge</td><td>Classification</td><td>Slide</td><td>10,616 WSIs; prostate Gleason grading</td></tr><tr><td>TCGA-BRCA</td><td>TCGA</td><td>Classification</td><td>Slide/Case</td><td>1,021 WSIs; IDC vs. ILC subtype classification</td></tr><tr><td>TCGA-NSCLC</td><td>TCGA</td><td>Classification</td><td>Slide/Case</td><td>1,053 WSIs; LUAD 541 / LUSC 512</td></tr><tr><td>TCGA-LUAD</td><td>TCGA</td><td>Survival</td><td>Case</td><td>516 WSIs; lung adenocarcinoma prognosis</td></tr><tr><td>TCGA-STAD</td><td>TCGA</td><td>Survival</td><td>Case</td><td>441 WSIs; gastric adenocarcinoma prognosis</td></tr><tr><td>TCGA-UCEC</td><td>TCGA</td><td>Survival</td><td>Case</td><td>537 WSIs; endometrial carcinoma prognosis</td></tr><tr><td>TCGA-KIRP</td><td>TCGA</td><td>Survival</td><td>Case</td><td>259 WSIs; papillary renal-cell carcinoma prognosis</td></tr><tr><td>TCGA-KIRC</td><td>TCGA</td><td>Survival</td><td>Case</td><td>519 WSIs; clear-cell renal-cell carcinoma prognosis</td></tr></table>

Table 24 also clarifies the scope of the benchmark. The BRACS minimal-subset diagnostic uses fine-grained lesion labels and validation folds suitable for recursive subset search, while the prediction and evidence diagnostics are evaluated across all nine datasets.

## K.2 Survival Data Preprocessing

For survival cohorts, each case has an observed time and censoring indicator. Cases without valid survival time or event/censoring metadata are excluded before fold construction. The GCE-MIL survival branch predicts a scalar risk score and is trained with Cox partial likelihood, so the main results do not require discretizing time into bins. The structural prognostic factors associated with each cohort are also recorded: LUAD prognosis relates to lepidic, acinar, papillary, micropapillary, and solid growth patterns; STAD relates to differentiation grade and signet-ring morphology; UCEC relates to FIGO grade and the glandular-to-solid component ratio; KIRC relates to Fuhrman nuclear grade within clear-cell architecture; and KIRP relates to papillary-core integrity and arrangement. Any discrete time-to-event auxiliary analysis follows the four non-overlapping interval construction used by the reference dataset protocol and remains separate from the Cox results reported in the main text. This setup makes the survival experiments a stress test for GCE-MIL: the evidence selector is trained against a risk-ordering objective rather than a categorical softmax, yet it still has to produce recoverable discrete evidence.

## K.3 WSI Patching and Feature Extraction

All slides are tiled into non-overlapping 256 × 256 patches at 20× magnification. Tissue regions are retained before feature extraction, and each retained patch is represented by a 1024-dimensional UNI embedding. Each slide is stored as one HDF5 file containing the patch feature matrix and the corresponding two-dimensional patch coordinates. The same preprocessing is used for the host backbone and its GCE wrapper, so differences in the reported results come from evidence selection and recovery rather than from patch extraction.

## K.4 Patch Encoder Choice Rationale

UNI is used as the default patch encoder because it provides a 1024-dimensional pathology foundation representation shared across classification and survival tasks. The robustness study in Table 19 also reports ResNet-50 and ViT-S features. Using several encoders tests whether GCE-MIL depends on a single representation space or remains useful as a wrapper around different feature extractors.

## K.5 CAMELYON-16 Localization Protocol

CAMELYON-16 is used only as an additional localization benchmark and is not counted among the nine primary classification and survival datasets. Slides are tiled with the same non-overlapping 256 × 256 patches at 20× magnification. Each patch footprint is mapped to the official annotation mask; a patch is labeled positive if at least 1% of its area overlaps an annotated tumor region, and patches outside the tissue mask are ignored during prediction and evaluation. Dice is computed on the patch grid after selecting one binarization threshold on the validation split for each method and then fixing that threshold for test slides. FROC is computed by threshold sweeping: connected components in the binarized patch grid are lesion candidates, the candidate score is the maximum patch score inside the component, and sensitivity is averaged at 1/8, 1/4, 1/2, 1, 2, 4, and 8 false positives per slide. This localization protocol evaluates spatial faithfulness rather than slide-level prediction. It is therefore used as supporting evidence for the Necessity and Recoverability claims, not as an additional main benchmark dataset.

## L Additional Visualizations

The remaining figures provide qualitative checks of the same mechanisms measured by the intervention tables. They are placed after the quantitative appendices so that visual examples complement, rather than substitute for, the S/N/R diagnostics.

## L.1 T-SNE and Main Qualitative Summaries

Figures 4 and 5 show that evidence-oriented training changes representation geometry. Before grounding, class clusters remain partially entangled across training epochs; after GCE grounding, the late-epoch embeddings separate into more compact class-specific regions. The continuous selector becomes bimodal during training (Figure 1, middle), enabling reliable discretization. Appendix Figures 6–11 and Figures 12–17 provide additional mechanism diagrams and qualitative overlays. These visualizations are secondary to the intervention metrics, but they help diagnose whether the learned selector behaves like a structured evidence mechanism rather than a post-hoc heatmap.

## L.2 Mechanism and Qualitative Overlays

Figures 4 and 5 report the available T-SNE visualizations. Appendix Figures 6–11 include the mechanism diagrams, and Figures 12–17 include the qualitative attention/evidence overlays. The mechanism panels are included to make the implementation pathway explicit: feature adaptation, anchor response, gate formation, noisy-OR utility, thresholding, and repair are separate operations. The qualitative overlays serve as sanity checks rather than quantitative proof; they illustrate the same pattern measured in the intervention tables, namely that recovered evidence is compact while attention often remains diffuse.

![](images/4aa0c41a0a1f926ea6ed38cdf69f00fdba14d4d3f740b7912ce629ebc744d70b.jpg)  
Figure 4: T-SNE before evidence grounding. Slide representations are less separated before the GCE evidence objective is applied.

![](images/c2ffefe62a4b7543351d93067d9a79dd2f228577182a1f6a6653661402d40334.jpg)  
Figure 5: T-SNE after evidence grounding. GCE training produces more separated slide representations, consistent with the classification and evidence diagnostics.

The mechanism diagrams below decompose the GCE-MIL pipeline into the operations that are compressed into Figure 2. Each mechanism is shown as a separate large figure so that the score curves and gate distributions remain readable. The first three figures emphasize how patch features are adapted and compared with anchors; the last three emphasize how continuous gates become a recovered discrete subset.

![](images/9fbec73c305acb2262a36859212d738ea0023171022b5ad4aeae8192d1197252.jpg)

![](images/cd90568260ca12661d057afcd4bdb1442c7c754ea1bb0e4d9b001c9f8533e7f8.jpg)

![](images/7faa5aa68ed24ff10d6ab286d70d1c01cb8ee16667421132a55660fa1cc5bb93.jpg)

![](images/b3b1b9608dc18b8445626cc690e977dd41580e3f5124b5b2d46088a7e256c7ea.jpg)  
Figure 6: Mechanism 1: feature adaptation. Low-rank residual adaptation keeps the pretrained feature space close to UNI while improving selector compatibility.

![](images/04d74c99bf706aab9caf56cf0f4d2d47972eb7cc4833b1b8aba0b2b8decb5071.jpg)

![](images/ab87dcff3ea084369928576662cd16982a032b1e8d14258165fcdeb691704eea.jpg)

![](images/0d821efe6b4f8cb51ce2119755e441b37a5f7a5c5b8bdc19be1599177d681e51.jpg)

![](images/079796807e368048238f11809438f7edbc72d8a850de40a1398a51426a6c8056.jpg)  
Figure 7: Mechanism 2: anchor response. The bridge maps patch features into TITAN anchor space and produces patch-anchor responses used by the coverage utility.

![](images/5fa0c3117903e672ca6c5da0127f10376a8b11795315f16c7849d31e41dc4750.jpg)

![](images/ac0bd55420b4f75ad2d47510e9d506f99efacc2b43257425a1b6f2c024c582ab.jpg)

![](images/68c2bda8528ebc79043c1f871fddf1a000e1b86b8515e77ebcebe6027f2f88d7.jpg)

![](images/82a368c0e8ddf278a0f4a4ddc0c4fcab6b56eb19ee91f7fbe13a4c13a33e269a.jpg)  
Figure 8: Mechanism 3: continuous gate. Temperature annealing sharpens the selector distribution so that continuous gates can be recovered as a discrete subset.

![](images/ea783cd61994aa7c382147d62d39a4f3a57da1166778fe6c0cdf22d47aa62618.jpg)

![](images/4ff70c6f65a8b86817e1d83426a12e681698eaad2d7a38d36ce1df27276a8c10.jpg)

![](images/a967c8f5e2b542d25b319704cfa28315329359bae6104b08e4b91e8df6c2dfe6.jpg)

![](images/ced67e9fce37c7094c87dc66129c2cbce8f0fb3c511fcc3019e33e0aec763441.jpg)  
Figure 9: Mechanism 4: noisy-OR utility. Exact noisy-OR coverage gives diminishing returns once an anchor is already covered, encouraging complementary evidence.

![](images/67e0646fb2f69aa3e84e2a001cf39d61203343bbbfc5b1f03c8ad24bd1750d97.jpg)

![](images/ec832fa6e78fd75ed63186561775c393cbe1804a6fcbd4eb64b0ab2ff511fcd7.jpg)

![](images/731ab83ce1d3ee79d2b5ce4df4e7e500cc361d678b340f363d47d07adea143e3.jpg)

![](images/44166399cfd32866e4e0932c889fa4f872bb59460d46588409695f8daa991cec.jpg)  
Figure 10: Mechanism 5: threshold recovery. The initial discrete subset is obtained by thresholding the continuous gate and falling back to the top patch when needed.

![](images/b2fefd87653cd622f503a661eafc900f79a3120af1c9606f6cd2ef7e2a1d9909.jpg)

![](images/7ba9ef194bab02d289fd51eec7f6ca6ca876d6935657b5617b8cfc329e10a3b1.jpg)

![](images/300cce7054d0a158c9b8bfd00edbd6f87b547ca29f70850f3ea918cde763b1c9.jpg)

![](images/d8c9fd75035ced9714efa0f148bd4110902991d8c4299f2f36c6c5d5e005e21f.jpg)  
Figure 11: Mechanism 6: greedy repair. Repair adds patches according to exact marginal utility until coverage and sufficiency criteria are restored.

The following qualitative overlays show representative evidence maps exported from the evaluation pipeline. They are not used to claim pixel-level correctness; the quantitative support for evidence quality comes from keep-only, remove, C-D gap, and CAMELYON-16 localization metrics. Instead, these examples help readers inspect whether the recovered subset is compact, whether it avoids obvious background/artifact regions, and how it differs from diffuse attention maps.

![](images/0db76caa95bbeb199716ec1e1fc33ee5cac38a47dba72734121eb3de65f1545f.jpg)  
DSMIL

![](images/02d95a073120cca401268e874da80221af19e2fd72b7ef1fbd0f0a4e9089b147.jpg)

![](images/37dd8266d5885a1909bb2f41cb35491ba368cd2aa529cd9d8118d18be180ff37.jpg)  
DTFD-MIL

![](images/bf1ac4d0f716dbed8613e4c11e10c1f3374c933dc5569dc74d53f5d1073b633f.jpg)  
DSMIL+ ours

![](images/d4d3da1edca3f173f23b83cba7e48f146a4aee802a4d2c09952b6386535c428c.jpg)  
IBMIL

![](images/f9b9ccd60148300f5e704d9df1fe4945a92c5d1cd560266ba3a89c0223f01614.jpg)

![](images/bfc52d75e6ba2541700c5910d4f4049667c2cad033f68749c843cdc8213c0e62.jpg)  
DTFD-MIL+ ours  
IBMIL+ ours

HDMIL  
![](images/febf3a567760167ad3735624bd8ceb9ba6c3ae67e82ab2290d3b9a6051291cdd.jpg)  
HDMIL+ ours

Figure 12: Qualitative evidence overlay 1. Representative slide-level attention and recovered GCE evidence.  
![](images/dd2b6ccbac985c091a8630738a343d2fc1a3864aed217fb4bf12d8cb25e705d9.jpg)  
DSMIL

![](images/3ad6f5677290b474dd74444aec6d00d698924a250c4752dd7a9dfaf93f9ffa71.jpg)  
DSMIL+ ours

![](images/b51624ffec571570eb233bca5b926fe76198f063693e82dbe1eab7f5c7285a6b.jpg)

![](images/b27771c322cd71d6fac47368f6f634db97126ede662e7d1e8435f3854578cba6.jpg)  
DTFD-MIL

![](images/2c07da8b7d11ad78fd9879c70111da815f0ef50194cf88d5ed7bdd6b33ac4871.jpg)

![](images/b76562ac0ffb55a8b8bd0018034974d6dcfc145039d020cf326f77c04f6b62a2.jpg)  
DTFD-MIL+ ours

IBMIL  
![](images/3e63183887571fa360020773465f8c7732f4774063651a5dbb074b64c8408064.jpg)  
IBMIL+ ours

HDMIL  
![](images/405c9251e3e34adf6f53e541411b7888a3c9bdf5d567471505d2a03ab8c8dddd.jpg)  
HDMIL+ ours

Figure 13: Qualitative evidence overlay 2. Representative slide-level attention and recovered GCE evidence.  
![](images/937dc318d563947509e5980eeff8c1c15e6bd33f095a188c7c98d53f02449bb6.jpg)  
DSMIL

![](images/0e64e24bc43873489e805dab4dce90fc095cf58a4916cb297f704f80d66fc0c6.jpg)

![](images/fc58db306da03c346d175102d1de723bd6e8143a9ef3cebff6188b339ec4ad96.jpg)

DTFD-MIL  
![](images/0db6b4ebfb074a733ece5ebd0709f038b92e2e55f9cbaa7299d1c129564c70b9.jpg)  
DSMIL+ ours

![](images/dff3095621e6f044a486b3032c688b69b4e311701e95d30db6b7f2833bd246a0.jpg)  
DTFD-MIL+ ours

IBMIL  
![](images/65177fed441695d5688b8f408bddda57f493bcdb9ca1154260a112b8d0810152.jpg)  
IBMIL+ ours

![](images/e4dc6e00479118f671444e081d6b1c74873199a0e94ff3b8f6220b7abe74cef2.jpg)

HDMIL  
![](images/70a48ed71053e9022155a102d21798794e275de0d0c3941ede569efb21d09a74.jpg)  
HDMIL+ ours

Figure 14: Qualitative evidence overlay 3. Representative slide-level attention and recovered GCE evidence.

IBMIL+ ours

![](images/a22565aa52ded7edc5d49b2c89a5152e70f9b547214cac6544938ec2585cd256.jpg)

![](images/8a4a11b0f734faf0de84ab122175177a764da2b108fd56851f623562d24d3125.jpg)

![](images/9d443101a01eb020ea6f013eab76dcdef9d46af2e23e8ad033a2b7d2c41892f2.jpg)

![](images/d1a346c1ffff7e3c66d3ec4014105a2a10ff53e5b65fd1bc16ff3b75e54723b3.jpg)

![](images/d89c11d7252d3c1b48af2c19b040f98b88f935efec30dd90fe12c6148cc86406.jpg)  
DSMIL+ ours

![](images/16c2c1ae033550513df2f2ca487bdfa7ac3508d7522c18b2921d89e0a8406d96.jpg)  
DTFD-MIL+ ours

![](images/de114a9be82946745a745edc79834a8788cd90bd6b16e3e59b133488c167eac5.jpg)

![](images/af1e8c94fcd7ab9791214eee83cb1b90d8479ff4266258b1143de6821ab8c8aa.jpg)  
HDMIL+ ours

Figure 15: Qualitative evidence overlay 4. Representative slide-level attention and recovered GCE evidence.  
![](images/98b08b1f1d43555ab83ff8a66e369eaa6a6d1b9de966fef5aafd4a2d56a71ccd.jpg)

![](images/f1c92e719d4442bdd7b3b1344ed374edb5daf98eee50da7b9f9d4ed2dc62aaa4.jpg)

![](images/3a75c635234959996c6eae714cc0fab9397d78151699a393f26822d3c76bb361.jpg)

![](images/f4cfc58929c320e232cae9f690c81633661a4776f798d975df2e58831e97f4a0.jpg)

![](images/f11a0a73b9cc1ce83822359832f629553217d2b1274a8fe45aae888ed203e8ff.jpg)  
DSMIL+ ours

![](images/7153e2a5024c981d455d53e50c35dbf93586d0f6f8c172a029c2ef2837c2178a.jpg)  
DTFD-MIL+ ours

![](images/847dbce1fbb1defa9d66da0c75cb6f254c9ef94e6da7bc920f411b8167ace849.jpg)  
IBMIL+ ours

![](images/5203018c6ec0fc072ca33294b8ecce473de402c8626e3b9b5d29223d84113dcc.jpg)  
HDMIL+ ours

Figure 16: Qualitative evidence overlay 5. Representative slide-level attention and recovered GCE evidence.  
![](images/7621d1c3a1b7b6e785066da81b91f8e74c5a7a29318519377664f58140929722.jpg)

![](images/0d5186b2e2ba843c8d50b890f12a0034c72dee87888c8e892b3b471ffe73c0af.jpg)

![](images/f82b1da18a4f16b28ac2401943693107f665c8a64b602164d6749a7cd7808a7e.jpg)

![](images/a30f88f11d6fda9bfff52832fb688a2064ab8d4d7ef4af638957c32375f903cb.jpg)

![](images/2f54d5bbd7a8ba96d521d5abcd236247a550765d6782a32cfd487a597aa08878.jpg)  
DSMIL+ ours

![](images/f663bbef56e1a139e4ea8b3f0c44245855c57ac8b380991d8bc887256cdef885.jpg)  
DTFD-MIL+ ours

![](images/bb5b2e3396c86761306614c9414820c18a36d10593a4ba43f0b7505c335cddcc.jpg)  
IBMIL+ ours

![](images/b56979f4222b8f02e41669dbef21fc4961a601aa191652368bf9bbc56985b113.jpg)  
HDMIL+ ours

Figure 17: Qualitative evidence overlay 6. Representative slide-level attention and recovered GCE evidence.

## M Limitations and Discussion

The S/N/R formalization operates at patch level; extending it to pixel-level or region-level evidence remains open. The anchor bank uses fixed text prompts that may not transfer to rare or previously unseen tissue types without prompt adaptation. Finally, S/N/R measures model-relative evidence faithfulness, not pathologist-verified causal mechanisms—bridging this gap requires clinical validation studies.