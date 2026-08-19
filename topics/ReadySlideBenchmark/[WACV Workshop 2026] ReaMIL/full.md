# ReaMIL: Reasoning- and Evidence-Aware Multiple Instance Learning for Whole-Slide Histopathology

Hyun Do Jung<sup>1</sup> Jungwon Choi<sup>2</sup> Hwiyoung Kim<sup>1\*</sup> <sup>1</sup>Yonsei University <sup>2</sup>KAIST

## Abstract

We introduce ReaMIL (Reasoning- and Evidence-Aware MIL), a multiple instance learning approachfor whole-slide histopathology that adds a light selection head to a strong MIL backbone. The head produces soft per-tile gates and is trained with a budgeted-sufficiency objective: a hinge loss that enforces the true-class probability to be ≥ τ using only the kept evidence, under a sparsity budget on the number ofselected tiles. The budgeted-sufficiency objective yields small, spatially compact evidence sets without sacrificing baseline performance. Across TCGA-NSCLC (LUAD vs. LUSC), TCGA-BRCA (IDC vs. Others), and PANDA, ReaMIL matches or slightly improves baseline AUC and provides quantitative evidence-efficiency diagnostics. On NSCLC, it attains AUC 0.983 with a mean minimal sufficient K (MSK) ≈ 8.2 tiles at τ = 0.90 and AUKC ≈ 0.864, showing that class confidence rises sharply and stabilizes once a small set of tiles is kept. The method requires no extra supervision, integrates seamlessly with standard MIL training, and naturally yields slide-level overlays. We report accuracy alongside MSK, AUKC, and contiguity for rigorous evaluation ofmodel behavior on WSIs.

## 1. Introduction

Whole-slide histopathology has become a standard testbed for weakly supervised learning [7]. Modern scanners produce gigapixel slides, but in most clinical datasets only slide-level labels are available: tumor subtype, grade, or outcome, without any pixel- or patch-level annotations [12]. Multiple instance learning (MIL) provides a natural framework for this setting, treating each slide as a bag of tiles that are encoded and aggregated into a single prediction [2, 10, 13]. Despite the weak supervision, these models can reach pathologist-level performance on some benchmarks and are now being deployed in early-stage clinical decision support tools.

However, standard MIL training focuses on bag-level accuracy: the model is rewarded for predicting the correct slide label, with no explicit notion of which tiles actually constitute the “evidence” for that prediction. Attention weights are often interpreted as explanations, but they are a side effect of training, not a primary objective [15, 16]. This gap between slide-level performance and tile-level reasoning becomes critical when models are meant to support clinical decisions. In practice, pathologists justify diagnoses by pointing to specific regions—glands with certain architecture, nests of atypical cells, or characteristic tumor– stroma interfaces. Computational models should ideally do the same: highlight a compact set of tiles sufficient to support the predicted label, while showing that the rest of the slide does not drive the decision.

Recent advances in representation learning have shifted the landscape toward foundation models pretrained on millions of tiles across sites and organs [4, 6]. We leverage pre-extracted UNI2-h [4] features as patch-level representations, allowing us to focus on the reasoning layer. On top of these frozen features, transformer-based MIL backbones such as TransMIL [17] already achieve competitive performance on multiple WSI benchmarks. Yet this “foundation MIL” stack does not address interpretability [3]: we have powerful encoders and backbones, but how they use evidence inside the bag remains opaque.

Our work treats evidence selection as a first-class objective in MIL rather than an afterthought. We attach a lightweight selection head on top of a strong MIL backbone to produce soft selection scores over tiles. These scores define three views of each slide: a full bag, a keep bag retaining only evidence tiles, and a drop bag containing the complement. By feeding these three bags through a shared backbone, we explicitly shape how the model uses evidence through a budgeted sufficiency objective: the keep bag should reach a target confidence τ for the true class while the drop bag does not support the true label (its trueclass probability remains low). We regularize evidence to be spatially compact and penalize selecting too many tiles, yielding four concrete properties: sufficiency, exclusion, contiguity, and budget. We call this framework ReaMIL: reasoning- and evidence-aware MIL.

To measure these properties, we introduce diagnostics that probe how the model’s true-class probability grows as we reveal more top-scoring tiles. The area under this “Kcurve” (AUKC) and the minimal sufficient K (MSK) at a chosen confidence threshold summarize how quickly the model’s belief saturates. Across TCGA-NSCLC, TCGA-BRCA [8, 18], and PANDA [1], we show that ReaMIL preserves or improves baseline AUC while substantially reducing MSK and improving AUKC, indicating that highconfidence decisions can be supported by small, spatially compact sets of tiles.

In summary, the main contributions of this work are summarized as follows:

• We present ReaMIL, a reasoning- and evidence-aware MIL framework that integrates sufficiency, exclusion, spatial contiguity, and evidence sparsity.

• We introduce quantitative evidence-efficiency metrics, including minimal sufficient K (MSK) and the area under the K-curve (AUKC), which measure how quickly confidence emerges as diagnostic tiles are revealed.

• We demonstrate that our ReaMIL maintains or even improves slide-level performance while producing highly compact and spatially coherent evidence sets across TCGA-NSCLC, TCGA-BRCA, and PANDA.

## 2. Related Work

## 2.1. Multiple instance learning for whole-slide histopathology

Multiple instance learning (MIL) treats a digital slide as a bag of tiles with a single slide-level label and no supervision for individual tiles. Attention-based pooling, introduced by Ilse et al. [10], replaced fixed max- or mean-pooling with a learned weighted combination of tile features and became the standard aggregation strategy. Subsequent architectures incorporated class-specific attention and clustering constraints (CLAM [13]) or transformer-based selfattention to model long-range context between tiles (Trans-MIL [17]). More recently, feature extraction has been decoupled from MIL aggregation: large self-supervised or multimodal encoders pre-trained on millions of histology tiles are frozen, and MIL models operate on pre-extracted features [5]. This reduces training cost and improves robustness across cohorts. We follow this strategy, using UNI2-h as the feature backbone while the MIL component focuses on aggregating and selecting evidence at the tile level.

## 2.2. Interpretability of attention in MIL

Interpretability in MIL for pathology has largely relied on visualizing attention weights as heatmaps or displaying top-attended tiles [10, 13]. However, attention as explanation has well-known limitations [16]: attention scores are shaped by end-to-end training and may not reflect causal importance; high-attention tiles can be redundant or partially spurious; and there is no guarantee that the attended subset alone suffices to recover the correct prediction, nor that the complement is non-predictive. Various remedies—instance-level regularization, auxiliary classifiers, multiple attention heads, or region proposals from slide labels—can make heatmaps more visually convincing, but they typically lack a quantitative framework for measuring how much evidence is actually needed for a decision.

## 2.3. Budgeted evidence and selective prediction

The idea of constraining a model to rely on a small subset of inputs appears in selective prediction [9], budgeted or early-exit models, and rationalization methods that train differentiable selectors to pick a few tokens or patches so that a downstream predictor matches the full model using only the selected subset. Our work adapts this perspective to MIL: we attach a small selection head on top of a fixed MIL encoder and train it with losses enforcing sufficiency of the kept bag, exclusion of the dropped bag, spatial contiguity, and an explicit budget on selection rate (normalized selection mass). We quantify the resulting behavior with K-curves, minimal sufficient K (MSK), and area under the K-curve (AUKC)—metrics that capture how quickly confidence rises as diagnostic regions are added and how small a subset suffices for a diagnosis.

## 3. Methodology

We build ReaMIL on top of a transformer-based MIL backbone, adding a lightweight evidence head that learns which patches suffice for the slide-level prediction. Figure 1 illustrates the overall architecture.

## 3.1. Problem setup and backbone

Following standard weakly supervised MIL, each slide s consists of a bag of patch features $X _ { s } = \{ x _ { s , i } \} _ { i = 1 } ^ { N _ { s } }$ extracted by a frozen encoder, along with spatial coordinates $C _ { s } =$ $\{ \bar { c } _ { s , i } \} _ { i = 1 } ^ { N _ { s } }$ where $c _ { s , i } = ( u _ { s , i } , v _ { s , i } )$ is the pixel location of patch i. We use UNI2-h [4] to extract d=1536 dimensional features. The slide has a single label $y _ { s } \in \{ 1 , \ldots , C \}$ but no patch-level supervision.

Patch features are projected into a token space via $\tilde { x } _ { s , i } = W _ { \mathrm { f e a t } } x _ { s , i } + b _ { \mathrm { f e a t } } .$ , with optional positional embeddings $t _ { s , i } = \tilde { x } _ { s , i } + \mathrm { M L P } _ { \mathrm { p o s } } ( \mathrm { n o r m } ( c _ { s , i } ) )$ . The resulting tokens $T _ { s } ~ = ~ [ t _ { s , 1 } , \ldots , t _ { s , N _ { s } } ]$ are processed by a TransMIL backbone [17]: a learned [CLS] token is prepended to the sequence and passed through L transformer layers. The final CLS representation $h _ { \mathrm { C L S } } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } }$ is mapped to class logits $\ell _ { s } = W _ { \mathrm { c l s } } h _ { \mathrm { C L S } } + b _ { \mathrm { c l s } } \in \mathbb { R } ^ { C }$ , and baseline training uses cross-entropy $\mathcal { L } _ { \mathrm { f u l l } } = \mathbf { C } \mathbf { E } ( \ell _ { s } , y _ { s } )$

![](images/ba5397be14a471f1eae75b65046e96bfe706e24d4bb067cdbb4c6540446a68de.jpg)  
Figure 1. Overview of ReaMIL. Frozen UNI2-h features and patch coordinates are extracted from each WSI and mapped to tokens with positional embeddings. An evidence head produces soft selection scores $z \in ( 0 , 1 ) ^ { N }$ via a Concrete (Gumbel–sigmoid) gate, and defines three bags: the full bag x, a keep bag $z \cdot x .$ and a drop bag $( 1 - z ) \cdot x .$ All three bags are processed by a shared TransMIL encoder and slide head. Losses encourage (i) correct predictions on the full and keep bags (cross-entropy on $\ell _ { \mathrm { f u l l } }$ and $\ell _ { \mathrm { k e e p } }$ plus a sufficiency hinge at confidence \tau ), (ii) low true-class probability on the drop bag (exclusion), (iii) spatially compact selections (contiguity on coordinates), and (iv) a small evidence budget via an $\ell _ { 1 }$ penalty on z. At test time, the model outputs both slide predictions and ranked evidence coordinates. Reasoning metrics are computed by probing the top-K curve of true-class probability $p _ { y } ( K )$ : AUKC summarizes the area under this curve, and \protect \mathrm {MSK}@\tau measures the minimal number of tiles required to reach confidence \tau .

## 3.2. Evidence selection head

For each token $t _ { s , i } ,$ a small MLP computes a selection logit $a _ { s , i } = \mathrm { M L P } _ { \mathrm { s e l } } ( t _ { s , i } ) \ \in \ \mathbb { R }$ . To enable differentiable selection, we apply the Concrete (Gumbel-sigmoid) relaxation [11, 14]. We sample $\epsilon _ { s , i } \sim$ Uniform(0, 1) and compute:

$$
z _ { s , i } = \sigma \bigg ( \frac { a _ { s , i } + \log \epsilon _ { s , i } - \log ( 1 - \epsilon _ { s , i } ) } { T } \bigg )\tag{1}
$$

where $T > 0$ is the temperature. This yields soft selection scores $z _ { s , i } \in ( 0 , 1 )$ that approach binary values as $T  0$

The scores define three views of each slide: the original bag $X _ { \mathrm { f u l l } } = X _ { s }$ , the evidence bag $X _ { \mathrm { k e e p } } = z _ { s } \odot X _ { s } ,$ and its complement $X _ { \mathrm { d r o p } } = ( 1 - z _ { s } ) \odot X _ { s }$ , where ⊙ denotes element-wise scaling. Since hard selection is nondifferentiable, we retain all tokens in the sequence but down-weight non-selected patches via soft masking. Each view is processed by the shared backbone to produce logits $\ell _ { \mathrm { f u l l } } , \ell _ { \mathrm { k e e p } } ,$ and $\ell _ { \mathrm { d r o p } } .$

## 3.3. Evidence-aware training objectives

Our goal is not only to achieve high slide-level accuracy, but also to explicitly shape how the model uses evidence inside each bag. To this end, we design an evidence-aware training objective that couples a standard classification loss with four additional terms, each enforcing a distinct property of the selector. Together, these losses encourage decisions that are (i) sufficient, with a small subset of selected patches supporting high-confidence predictions; (ii) exclusive, with the remaining patches not supporting the true label (low true-class probability); (iii) spatially contiguous, so that evidence forms coherent regions on the slide; and (iv) budgeted, limiting the amount of selected evidence.

Let $p _ { y } ( \ell ) = \mathrm { s o f t m a x } ( \ell ) [ y _ { s } ]$ denote the true-class probability. We combine five losses:

$$
\mathcal { L } _ { \mathrm { f u l l } } = \mathrm { C E } ( \ell _ { \mathrm { f u l l } } , y _ { s } ) ,\tag{2}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s u f f } } = \mathrm { C E } ( \ell _ { \mathrm { k e e p } } , y _ { s } ) + \operatorname* { m a x } \big ( \tau - p _ { y } ( \ell _ { \mathrm { k e e p } } ) , 0 \big ) , } \end{array}\tag{3}
$$

$$
\mathcal { L } _ { \mathrm { e x c l } } = \operatorname* { m a x } \big ( p _ { y } ( \ell _ { \mathrm { d r o p } } ) - \beta , 0 \big ) ,\tag{4}
$$

$$
\mathcal { L } _ { \mathrm { c o n t i g } } = \frac { \sum _ { i } z _ { s , i } \| c _ { s , i } - \mu _ { s } \| _ { 2 } ^ { 2 } } { \sum _ { i } z _ { s , i } } ,\tag{5}
$$

$$
\mathcal { L } _ { \mathrm { b u d g e t } } = \frac { 1 } { N _ { s } } \sum _ { i } z _ { s , i } ,\tag{6}
$$

where $z _ { s , i }$ are selection scores, $\begin{array} { r } { \mu _ { s } = \sum _ { i } z _ { s , i } c _ { s , i } / \sum _ { i } z _ { s , i } } \end{array}$ is the z-weighted centroid, and $\tau , \beta \in ( 0 , 1 )$ are hyperparameters, with τ used as a confidence threshold on the true-class probability $p _ { y } ( . )$ and T in (1) serving as the temperature of the Concrete gate. $\mathcal { L } _ { \mathrm { b u d g e t } }$ is the average selection rate (normalized $\ell _ { 1 }$ norm of $z _ { s } )$ and acts as an explicit sparsity penalty.

The total loss is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { f u l l } } + \lambda _ { \mathrm { s u f f } } \mathcal { L } _ { \mathrm { s u f f } } + \lambda _ { \mathrm { e x c l } } \mathcal { L } _ { \mathrm { e x c l } } + \lambda _ { \mathrm { c o n t i g } } \mathcal { L } _ { \mathrm { c o n t i g } } + \lambda _ { \mathrm { b u d g e t } } \mathcal { L } _ { \mathrm { b u d g e t } } . } \end{array}\tag{7}
$$

Here, the weights $\lambda _ { \mathrm { s u f f } } , \lambda _ { \mathrm { e x c l } } , \lambda _ { \mathrm { c o n t i g } } , \lambda _ { \mathrm { b u d g e t } }$ balance fidelity against the strength of the evidence-aware constraints.

## 3.4. Evidence-efficiency metrics

Conventional metrics such as AUC, accuracy, or F1 summarize how often a model predicts the correct slide label, but they are insensitive to how much evidence the model needs to make those predictions. To evaluate whether ReaMIL actually learns to rely on small, sufficient evidence sets, we introduce a family of evidence-efficiency metrics based on the behavior of the model as top-ranked tiles are gradually revealed. To quantify evidence efficiency, we probe the relationship between revealed patches and model confidence. At test time, we rank patches by their selection logits $a _ { s , i }$ (Gumbel noise is used only during training) and construct a K-curve that records the true-class probability $p _ { y } ( K )$ as a function of the number of revealed patches K.

Minimal Sufficient K (MSK). For each slide s and confidence threshold τ, we define

$$
\mathrm { M S K } _ { s } ( \tau ) = \operatorname* { m i n } \{ K : p _ { y } ( K ) \geq \tau \} .\tag{8}
$$

MSK measures how many top-ranked patches are needed for the model to reach confidence τ.

Area Under K-Curve (AUKC). We also define the area under the K-curve in terms of the normalized evidence fraction $\kappa = K / N _ { s } \in [ 0 , 1 ]$

$$
\mathrm { A U K C } _ { s } = \int _ { 0 } ^ { 1 } p _ { y } ( \kappa ) d \kappa ,\tag{9}
$$

where $p _ { y } ( \kappa )$ denotes the true-class probability when the top $\kappa \cdot N _ { s }$ tiles are kept.

## 4. Experiments

## 4.1. Datasets and setup

We evaluate on three binary WSI classification tasks: TCGA-NSCLC (LUAD vs. LUSC), TCGA-BRCA (IDC vs. Others), and PANDA (clinically significant vs. nonsignificant prostate cancer). For each dataset, we construct patient-disjoint train/validation/test splits with class stratification. All slides are processed into frozen UNI2-h features (d=1536) and tile coordinates; the encoder is never finetuned. Details on label mappings and patient counts are in the supplement.

The backbone is a TransMIL-style transformer $( d _ { \mathrm { m o d e l } } { = } 5 1 2 .$ 8 heads, 4 layers). We first train a baseline with standard cross-entropy, then attach the evidence head and continue training with the combined loss (Section 3), warm-starting from the baseline checkpoint. All models use AdamW with cosine decay and mixed-precision on two RTX 6000 Ada GPUs. We report mean±std over three seeds.

![](images/5828bb00fcc808348fee9038a70b4761050be0aa442db96dfe72ed82f444ce7a.jpg)  
Figure 2. K-curve on NSCLC (test set). True-class probability $p _ { y } ( K )$ as top-K tiles (ranked by selection score) are revealed. Solid line: mean across slides; shaded region: ±1 std. Vertical dashed line: mean MSK@τ = 0.9. MSK is computed per-slide before averaging, so individual slides may cross τ even when the mean curve does not.

## 4.2. Slide-level performance

Table 1 compares the baseline (TransMIL + UNI2-h, no evidence head) against ReaMIL with the full budgeted objective, showing that adding the evidence head and reasoning losses extends standard MIL pipelines without trading accuracy for interpretability.

<table><tr><td>Dataset</td><td>Method</td><td>AUC</td><td>Acc</td><td>F1macro</td></tr><tr><td>BRCA</td><td>Baseline +ReaMIL</td><td>0.897±0.019 0.904±0.011</td><td>0.877±0.006 0.819±0.022 0.888±0.010 0.827±0.019</td><td rowspan="5"></td></tr><tr><td rowspan="2">NSCLC</td><td>Baseline</td><td>0.969±0.006</td><td>0.935±0.006 0.935±0.006</td></tr><tr><td>+ReaMIL</td><td>0.983±0.004 0.927±0.025</td><td>0.927±0.026</td></tr><tr><td>PANDA</td><td>Baseline +ReaMIL 0.989±0.003</td><td>0.985±0.002</td><td>0.955±0.004 0.945±0.004</td></tr></table>

Table 1. Slide-level performance (mean±std, 3 seeds). ReaMIL uses the full budgeted objective.

## 4.3. Evidence efficiency

Figure 2 shows K-curves on NSCLC: for each slide, tiles are ranked by selection score and the true-class probability $p _ { y } ( K )$ is recorded as the top-K tiles are revealed. Table 2 reports MSK@τ=0.90 (minimal tiles to reach 90% confidence) and AUKC across all datasets. Note that these metrics require an explicit selector to rank tiles and are therefore defined only for ReaMIL, not for vanilla MIL baselines.

![](images/42e1128ca85ca430f3ee09639ec223d59c282ebbc40ba10f301fe95e80ce824a.jpg)  
Figure 3. Evidence visualization on TCGA-NSCLC. Left: LUSC (squamous cell carcinoma) case with relatively compact evidence clusters over squamous tumor nests. Right: LUAD (adenocarcinoma) case with more diffuse selection over gland-forming tumor regions. Each panel shows selected tile locations (green boxes) and the corresponding top-K zoomed patches. For visualization, we show zoomed-in regions (left: $8 1 9 2 \times 8 1 9 2 ;$ right: 16384 × 16384 pixels), where the selected tiles (size 256 × 256) are outlined in green.

<table><tr><td>Dataset</td><td> $\mathrm { M S K } @ \tau = 0 . 9 0 \left( \downarrow \right)$ </td><td>AUKC (↑)</td></tr><tr><td>BRCA</td><td> $1 6 . 0 { \pm } 1 1 . 8 $ </td><td> $0 . 8 3 3 { \pm } 0 . 0 1 8$ </td></tr><tr><td>NSCLC</td><td> $8 . 2 { \pm } 2 . 1$ </td><td> $0 . 8 6 4 { \scriptstyle \pm 0 . 0 6 9 }$ </td></tr><tr><td>PANDA</td><td> $7 . 2 \pm 3 . 6$ </td><td> $0 . 8 1 1 { \scriptstyle \pm 0 . 0 5 5 }$ </td></tr></table>

Table 2. Evidence efficiency metrics for ReaMIL (mean±std, 3 seeds). MSK@0.9: minimal tiles to reach 90% confidence. AUKC: area under the K-curve. These metrics require an explicit selector and are not defined for vanilla MIL baselines.

On NSCLC, ReaMIL achieves MSK@0.9 of approximately 8.2 tiles—fewer than 0.1% of the average bag size $( \sim 6 , 0 0 0$ tiles)—demonstrating that the selector concentrates evidence into a small, sufficient subset.

## 4.4. Ablations

Table 3 isolates each loss component on NSCLC. Without the full objective, ablated models select nearly all tiles (mean $\| z \| _ { 1 } ~ > ~ 0 . 8 5$ vs. 0.002 for ReaMIL), causing the keep bag to approximate the full bag. This yields trivially low suff. gap and contig. values—not because evidence is well-selected, but because almost nothing is excluded. In contrast, ReaMIL (full) achieves true sparse selection: $p _ { y } ( \mathrm { d r o p } ) \approx 0$ shows the complement is non-predictive for the true class, confirming that the small selected set genuinely captures the diagnostic signal.

## 4.5. Qualitative results

Figure 3 shows evidence overlays on representative NSCLC slides. The LUSC case (left) exhibits relatively compact evidence clusters over squamous tumor nests. The LUAD case (right) shows a more diffuse pattern of selected tiles across gland-forming adenocarcinoma regions. In both cases, ReaMIL concentrates its evidence on morphologically relevant tumor areas while largely ignoring background tissue, consistent with the quantitative findings.

<table><tr><td>Variant AUC Suff. gap</td><td colspan="4"> $( \downarrow ) p _ { y } ( \mathrm { d r o p } )$  (↓) Contig. (↓) ∥z∥1 (↓)</td></tr><tr><td>ReaMIL (full)</td><td>0.984</td><td>0.119</td><td>0.000</td><td>0.137 0.002</td></tr><tr><td>w/o sufficiency 0.981</td><td></td><td>0.039 0.167</td><td>0.106</td><td>0.847</td></tr><tr><td>w/o exclusion 0.981</td><td>0.000</td><td>0.414</td><td>0.128</td><td>0.923</td></tr><tr><td>w/o contiguity 0.985</td><td>0.001</td><td>0.339</td><td>0.127</td><td>0.891</td></tr></table>

Table 3. Ablations on NSCLC. Suff. gap: confidence drop using only kept tiles. $p _ { y } ( \mathrm { d r o p } ) { \mathrm { : } }$ true-class probability of the drop bag (lower = the drop bag alone does not support the true label). Contig.: spatial dispersion. $\| z \| _ { 1 } { \mathrm { : } }$ mean selection rate (normalized $\ell _ { 1 } ;$ lower = sparser). Ablations select nearly all tiles, yielding trivially low suff. gap but defeating the goal of compact evidence; only ReaMIL (full) achieves true sparse selection.

## 5. Conclusion

We presented ReaMIL, a method that transforms wholeslide classification into an evidence-seeking problem by adding a budgeted selection head to standard MIL backbones. Training the selector so that a small, spatially compact subset suffices for prediction while forcing complementary tiles to be non-predictive for the true class preserves baseline AUC while producing compact evidence— on TCGA-NSCLC, AUC 0.983 with MSK ≈ 8.2 at τ = 0.90 and AUKC ≈ 0.864. The framework requires only slide-level supervision, fits existing pipelines, and shows that accurate yet interpretable MIL is achievable without extra annotation—critical as computational pathology moves toward clinical deployment.

Limitations. Our approach relies on pre-extracted features from a single foundation model (UNI2-h) and has been evaluated on relatively balanced research datasets. Validation on more diverse clinical cohorts with class imbalance and domain shift, as well as user studies with pathologists to assess clinical utility, remain important directions for future work.

## Acknowledgement

This work was supported by the Bio-industrial Technology Development Program (RS-2025-02220286, (Division 2) Development of large language AI model-based techniques and platforms for nursery record generation and task automation) funded by the Ministry of Trade, Industry & Resources (MOTIR, Korea)

## References

[1] Wouter Bulten, Kimmo Kartasalo, Po-Hsuan Cameron Chen, Peter Strom, Hans Pinckaers, Kunal Nagpal, Yuannan Cai,¨ David F Steiner, Hester Van Boven, Robert Vink, et al. Artificial intelligence for diagnosis and gleason grading of prostate cancer: the panda challenge. Nature Medicine, 28(1):154– 163, 2022. 2

[2] Gabriele Campanella, Matthew G. Hanna, Luke Geneslaw, Andrew Miraflor, Vitor Werneck Krauss Silva, Klaus J. Busam, Edi Brogi, Victor E. Reuter, David S. Klimstra, and Thomas J. Fuchs. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nature Medicine, 25(8):1301–1309, 2019. 1

[3] Supriyo Chakraborty, Richard J. Tomsett, R. Raghavendra, Daniel Harborne, M. Alzantot, F. Cerutti, M. Srivastava, A. Preece, S. Julier, R. Rao, T. Kelley, Dave Braines, M. Sensoy, C. Willis, and Prudhvi K. Gurram. Interpretability of deep learning models: A survey of results. 2017 IEEE SmartWorld, Ubiquitous Intelligence & Computing, Advanced & Trusted Computed, Scalable Computing & Communications, Cloud & Big Data Computing, Internet of People and Smart City Innovation (Smart-World/SCALCOM/UIC/ATC/CBDCom/IOP/SCI), 2017. 1

[4] Richard J Chen, Tong Ding, Ming Y Lu, Drew F K Williamson, Guillaume Jaume, Bowen Chen, Andrew Zhang, Daniel Shao, Andrew H Song, Muhammad Shaban, et al. Towards a general-purpose foundation model for computational pathology. Nature Medicine, 30:154–165, 2024. 1, 2

[5] Zhengying Chen, Xiao Wang, Alexander Kolesnikov, Dirk Weissenborn, Xiaojuan Qi, Jakob Verbeek, Neil Houlsby, Xiaohua Zhai, and Lucas Beyer. Scaling vision transformers to 22 billion parameters. arXiv preprint, 2022. 2

[6] Ozan Ciga and Anne L Martel. Overcoming data scarcity in biomedical imaging with a foundational multi-task model. arXiv preprint arXiv:2010.07964, 2022. 1

[7] Neofytos Dimitriou, Ognjen Arandjelovic, and P. Caie. Deep learning for whole slide image analysis: An overview. Frontiers in Medicine, 2019. 1

[8] Nathan J. Edwards, Mauricio Oberti, Ratna R. Thangudu, Shuang Cai, Peter B. McGarvey, Shine Jacob, Subha Madhavan, and Karen A. Ketchum. The CPTAC Data Portal: A Resource for Cancer Proteomics Research. Journal of Proteome Research, 14(6):2707–2713, 2015. 2

[9] Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems, 2017. 2

[10] Maximilian Ilse, Jakub M. Tomczak, and Max Welling. Attention-based deep multiple instance learning. In Proceed ings ofthe 35th International Conference on Machine Learning, pages 2127–2136. PMLR, 2018. 1, 2

[11] Eric Jang, Shixiang Gu, and Ben Poole. Categorical repa rameterization with Gumbel-softmax. In International Con ference on Learning Representations, 2017. 3

[12] G. Litjens, Peter B´ andi, Babak Ehteshami Bejnordi, Oscar´ G. F. Geessink, M. Balkenhol, P. Bult, A. Halilovic, M. Hermsen, Rob van de Loo, R. Vogels, Quirine F. Manson, N. Stathonikos, A. Baidoshvili, Paul van Diest, C. Wauters, Marcory van Dijk, and Jeroen van der Laak. 1399 h&estained sentinel lymph node sections of breast cancer patients: the camelyon dataset. GigaScience, 2018. 1

[13] Ming Y. Lu, Drew F. K. Williamson, Tiffany Y. Liu, Richard J. Chen, Matteo Barbieri, and Faisal Mahmood. Data-efficient and weakly supervised computational pathology on whole-slide images. Nature Biomedical Engineering, 5(6):555–570, 2021. 1, 2

[14] Chris J Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete random variables. In International Conference on Learning Representations, 2017. 3

[15] Danish Pruthi, Mansi Gupta, Bhuwan Dhingra, Graham Neubig, and Zachary Chase Lipton. Learning to deceive with attention-based explanations. In Proceedings ofthe 58th An nual Meeting of the Association for Computational Linguistics, pages 4782–4793, Online, 2020. 1

[16] Sofia Serrano and Noah A Smith. Is attention interpretable? In Proceedings ofthe 57th Annual Meeting ofthe Association for Computational Linguistics, pages 2931–2951, 2019. 1, 2

[17] Zhuchen Shao, Hao Bian, Yang Chen, Yifeng Wang, Jian Zhang, Xiangyang Ji, and Yongbing Zhang. Transmil: Transformer based correlated multiple instance learning for whole slide image classification. In Advances in Neural In formation Processing Systems, pages 2136–2147, 2021. 1, 2

[18] John N Weinstein, Eric A Collisson, Gordon B Mills, Kenna R Shaw, Brad A Ozenberger, Kyle Ellrott, Ilya Shmulevich, Chris Sander, and Joshua M Stuart. The cancer genome atlas pan-cancer analysis project. Nature Genetics, 45(10):1113–1120, 2013. 2