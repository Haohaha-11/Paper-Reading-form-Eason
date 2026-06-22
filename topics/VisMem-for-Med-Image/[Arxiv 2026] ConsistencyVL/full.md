# Visuals Lie, Consistency Speaks: Disentangling Spatial Attention from Reliability in Vision-Language Models

Logan Mann<sup>2</sup> Yi Xia<sup>1</sup> Ajit Saravanan<sup>2</sup> Ishan Dave<sup>3</sup> Saadullah Ismail<sup>1</sup> Shikhar Shiromani<sup>4</sup> Emily Huang<sup>1</sup> Ruizhe Li<sup>1</sup> Kevin Zhu<sup>1</sup>

<sup>1</sup>University of California, Santa Barbara <sup>2</sup>Algoverse AI Research

<sup>3</sup>University of California, Berkeley <sup>4</sup>Independent Researcher

## Abstract

Multimodal Foundation Models (MFMs) are rapidly evolving from simple pattern matchers into reasoning agents. As these systems are used in higher-stakes settings, reliability, or knowing when a model may hallucinate, becomes critical. A common intuition in the field, which we call the Attention-Confidence Assumption, is that reliability follows from “structural” visual perception: if a model focuses tightly on relevant image regions, its subsequent answer should be trustworthy. Conversely, scattered attention is often assumed to signal confusion.

We challenge this assumption through VLM Reliability Probe (VRP), a systematic cross-family investigation into reliability signals in contemporary Vision-Language Models (VLMs). We introduce “structural attention” metrics, including cluster counts (C ) and spatial entropy (H<sub>s</sub>) to quantify the coherence of the visual encoder’s gaze. To capture the dynamics of this gaze, we further track attention evolution (∆H<sub>s</sub>) across all layers. This analysis reveals a critical “Symbolic Detachment”: models often exhibit “Early Locking” of visual features only to diffuse attention in later layers, effectively severing the link between early perception and final generation. Contrary to the grounding hypothesis, our results demonstrate a “Cluster Failure”: spatial attention patterns possess near-zero correlation (R ≈ 0.001) with model accuracy. Instead, we find that reliability is fundamentally a phenomenon of generation dynamics and internal state distributions. Self-Consistency (SC), the agreement rate across sampled reasoning paths, emerges as the dominant predictor of truth (R = 0.429). By aggressively scaling causal interventions, we further demonstrate a massive architectural divergence: LLaVA “locks” its prediction in a fragile late-stage structural bottleneck, whereas PaliGemma and Qwen2-VL distribute reliability globally, showing extreme resilience even when ∼ 50% or more of their most predictive layer is destroyed. These findings suggest that for current VLMs, reliability signals are detached from visual grounding maps, and are best inferred from generation-time dynamics and hidden-state probes.

## 1 Introduction

The integration of vision and language into Multimodal Foundation Models (MFMs) promises a future where AI agents can perceive and reason about the physical world. However, this promise is threatened by hallucination, the tendency of models to generate confident but factually incorrect assertions. To deploy these models in safety-critical domains (e.g., robotics, medical imaging), we must be able to quantify their reliability.

Traditionally, interpretability research has looked to the ”Attention Mechanism” as a window into the model’s mind Jain & Wallace (2019). In Vision-Language Models (VLMs), this manifests as the Attention-Confidence Assumption: the belief that a model’s reliability is correlated with the quality of its visual grounding. If a model is asked, ”Is there a dog?”

and it focuses sharply on the dog, we assume that it ”knows” the answer. If its attention is diffuse or focuses on the background, we assume that it is hallucinating.

In this work, we rigorously test this assumption across three representative VLM families (LLaVA-1.5, PaliGemma, and Qwen2-VL) Liu et al. (2023); Beyer et al. (2024); Wang et al. (2024). We perform a comprehensive analysis of reliability signals by comparing ”structural” metrics derived from visual cross-attention against ”linguistic” metrics derived from generation dynamics. We explicitly position novelty at the hidden-state reliability probe and cross-family layer-wise analysis; attention-failure and self-consistency are treated as important prior findings that we extend and calibrate in the VLM setting.

Terminology note. We use VLM as the default term throughout; MFM and LVLM are used only when matching prior-work phrasing.

Reproducibility. Code and evaluation scripts are available at https://github.com/ itsloganmann/VLM-Reliability-Probe (prompts, split definitions, and probe training pipeline).

## 2 Related Work

Large vision-language models (LVLMs) are built on foundation architectures such as CLIPstyle image encoders and large language backbones, enabling strong instruction-following and open-ended reasoning Radford et al. (2021); Alayrac et al. (2022); Li et al. (2022); Liu et al. (2023); Dai et al. (2023). Reliability and grounding concerns emerge when these models generate fluent but incorrect outputs, which has motivated benchmark-centric studies of hallucination in captioning and VQA Rohrbach et al. (2018); Li et al. (2023b). Beyond LLaVA-Bench Zhou et al. (2023), recent evaluation suites such as MME Fu et al. (2023), SEED Bench Li et al. (2023a), and MM-Vet Yu et al. (2023) broaden coverage across multimodal skills and stress-test visual grounding in diverse settings. In parallel, interpretability work debates whether attention is a faithful explanation signal Jain & Wallace (2019); Wiegreffe & Pinter (2019). Relatedly, recent work on faithfulness and behavioral reliability shows that surface-level explanations can decouple from the internal determinants of outputs, including scenario-dependent shifts Chaudhury & Shiromani (2025); Shiromani et al. (2026). For VLMs specifically, recent evidence also reports the “see-but-not-believe” phenomenon, i.e., correct localization without correct reasoning Liu et al. (2025). Our contribution is therefore not the generic claim that attention alone is insufficient, but a cross-family, layerwise reliability analysis centered on early locking/symbolic detachment and on hidden-state reliability probes.

Recent work on language prior highlights a core evaluation tension: should we assess whether the model gives the correct answer, or whether it truly integrates visual evidence? Long et al. (2025) asks a more representation-centric question and contrasts hidden trajectories with and without images to identify a Visual Integration Point (VIP) and define Total Visual Integration (TVI), a metric that quantifies how strongly visual evidence shapes representations. This reveals when models start “seeing” and how visual influence accumulates, addressing a gap left by output-only probes. Our study complements this line of inquiry but targets a different blind spot: we ask whether spatial attention structure itself is predictive of correctness, and whether reliability signals live in the generation dynamics rather than in the visual attention maps. In contrast to VIP/TVI, which measure representational shift induced by the image, we show that even when attention appears structurally grounded, it can be statistically decoupled from truthfulness; the strongest signals instead emerge from agreement across sampled reasoning paths and from hidden-state probes. This clarifies what our work addresses that prior representation analyses do not: reliability prediction and calibration, not just visual integration. Complementary benchmark and mitigation work further suggests that reliability is evaluation and decoding-dependent, motivating our focus on generation dynamics as a readout for correctness Thomas et al. (2026); Sahay et al. (2025).

To make the contribution boundary explicit: we do not claim to newly discover that attention can be unfaithful or that self-consistency helps; those are established in prior NLP/VLM literature. Our contribution is a unified, cross-family reliability study that links earlylocking/symbolic-detachment dynamics to downstream correctness and shows that hiddenstate probes provide the strongest single-pass reliability signals.

![](images/6c689bbc60f9563b69fc85ecb584b784f243f5593aaed878450aa02829a1fd1e.jpg)  
Figure 1: VLM Reliability Probe (VRP) Framework. We instrument three computational stages: Stage 1 extracts cross-attention maps from the visual encoder, yielding Structural metrics (entropy $H _ { s } ,$ clusters $C _ { k } ) ;$ we aggregate $A ^ { ( l , h ) }$ by averaging over heads and answertoken positions to form one per-layer spatial vector in $\mathbb { R } ^ { S }$ . Stage 2 probes hidden states via logit lens plus dense MLP and sparse L -logistic probe variants, providing Mechanistic signals; Stage 3 samples K=10 outputs for Behavioral metrics (self-consistency). Key finding: Structural metrics fail $( R ^ { 2 } < 0 . 0 8 )$ , while Mechanistic probes succeed (AUROC > 0.95). Red indicates causal intervention points.

## Our findings reveal a disconnect:

1. Visuals Lie: The spatial structure of attention (entropy, clustering, focus) has almost no statistical relationship with correctness $( R \approx 0 )$ . A model can hallucinate while attending to the right region, or answer correctly with diffuse attention.

2. Consistency Speaks: The most reliable behavioral signal of truth is not found in pixel-space attention, but rather in the stability of linguistic generation. Self-Consistency Wang et al. (2022) outperforms all visual metrics, achieving $R = 0 . 4 2 9$

3. Causal Architectures Diverge: Hidden-state representations house the most powerful predictive indicators (AUROC > 0.95). Crucially, massive scaling of neural ablations proves reliability paths are architecturally dependent. LLaVA centralizes truth in a sparse, fragile late-stage bottleneck, while PaliGemma and Qwen2-VL dynamically distribute these functions, remaining computationally robust even when ∼ 50% or more of their most predictive subnetwork is bypassed.

## 3 Methodology

We introduce VLM Reliability Probe (VRP), a comprehensive analysis pipeline designed to extract, quantify and correlate the internal state of the model with the correctness of the output (Figure 1). Our primary investigative goal is to disentangle two competing hypotheses regarding VLM reliability:

1. The Structural Hypothesis: Reliability is grounded in the spatial coherence of the visual encoder’s attention (i.e., how the model “looks”).

Table 1: Cross-Model Summary I: Reliability and attention structure. Visual attention metrics remain near-random predictors of correctness across all model families.
<table><tr><td>Model</td><td>Model Accuracy</td><td>Top-K Attention  $R ^ { 2 }$  (max)</td><td>Supervised Classifier Acc</td></tr><tr><td>LLaVA-1.5-7B</td><td>67.6%</td><td>0.008</td><td>53.0%</td></tr><tr><td>PaliGemma-3B</td><td>78.6%</td><td>0.080</td><td>55.0%</td></tr><tr><td>Qwen2-VL-7B</td><td>28.8%</td><td>0.007</td><td>52.0%</td></tr></table>

Table 2: Cross-Model Summary II: Logit-lens dynamics. Integration layer location and margin formation differ by family but remain strongly predictive in hidden states.
<table><tr><td>Model</td><td>Peak visual-integration layer  $( l _ { \mathrm { v i s } } ^ { \star } )$ </td><td>Peak final-margin value  $( \Delta \mathcal { M } _ { l _ { \mathrm { f i n a l } } ^ { \star } } )$ </td><td>MLP Contribution</td></tr><tr><td>LLaVA-1.5-7B</td><td>L24</td><td>+9.20 (L31)</td><td>82.1%</td></tr><tr><td>PaliGemma-3B</td><td>L14</td><td>+10.85 (L14)</td><td>47.6%</td></tr><tr><td>Qwen2-VL-7B</td><td>L27</td><td>+8.40 (L27)</td><td>68.2%</td></tr></table>

2. The Consistency Hypothesis: Reliability is a product of the generation dynamics and latent linguistic stability (i.e., how the model “speaks”).

## 3.1 Method Summary (Main Text)

We instrument VLMs with forward hooks to capture cross-attention maps and hidden states during generation, then compare structural signals $( C _ { k } , H _ { s } )$ against linguistic/mechanistic signals (self-consistency, token confidence, and learned probe scores from dense MLP and sparse L -logistic variants). In the main paper, we focus on the core reliability findings and cross-model comparisons.

## 4 Experimental Setup

We evaluate LLaVA-1.5-7B, PaliGemma-3B, and Qwen2-VL-7B Liu et al. (2023); Beyer et al. (2024); Wang et al. (2024) across POPE Li et al. (2023b) (Adversarial split, 1,000 samples), LLaVA-Bench Zhou et al. (2023) (90 open-ended questions), custom counting/spatial tasks, and the new VQA v2 and TextVQA evaluations. This setup allows us to compare reliability behavior on hallucination stress tests, open-ended reasoning, scene understanding, and OCR-heavy question answering using correlation and AUROC metrics; it is complementary to broader multimodal suites such as MME Fu et al. (2023), SEED-Bench Li et al. (2023a), and MM-Vet Yu et al. (2023). We provide sample accounting and uncertainty intervals for headline claims in Table 4.

## 5 Results

We present empirical evaluation across three VLMs: LLaVA-1.5-7B, PaliGemma-3B, and Qwen2-VL-7B. Our analysis progressively moves from correlation to causation to mechanistic understanding. Tables 1–3 summarize key findings; extended results are in Appendix A.3. Table 1 reports reliability and attention-structure failures, Table 2 summarizes layer-wise logit-lens dynamics, and Table 3 reports benchmark-level reliability prediction.

## 5.1 Visual Attention Does Not Predict Reliability

Core Finding: Spatial attention metrics show near-zero correlation with correctness. On the pooled 3,090-sample structural-analysis set (Table 4), cluster count $( C _ { k } )$ achieves $R =$ 0.001 (95% CI: $[ - 0 . 0 \dot { 3 } 4 , 0 . 0 3 6 ] )$ and spatial entropy $( H _ { s } )$ achieves $R = - 0 . 0 1 2$ (95% CI: [−0.047, 0.024]), both statistically indistinguishable from random noise $( p > 0 . 0 5 )$ . This “Cluster Failure” persists regardless of attention head selection: even when filtering to the top-k heads by logit contribution, $R ^ { 2 } \leq 0 . 0 8$ (Table 1).

Table 3: Cross-Model Summary III: Reliability prediction across benchmarks. Hiddenstate probes are strongest on POPE/LLaVA-Bench and show task-dependent gains over raw output confidence on VQA v2/TextVQA.
<table><tr><td>Model</td><td>POPE Probe</td><td>LLaVA-Bench Probe</td><td>VQA v2 Output</td><td>VQA v2 Probe</td><td>TextVQA Output</td><td>TextVQA Probe</td></tr><tr><td>LLaVA-1.5-7B</td><td>0.956</td><td>0.956</td><td>0.559</td><td>0.745</td><td>0.563</td><td>0.721</td></tr><tr><td>PaliGemma-3B</td><td>0.738</td><td>0.738</td><td>0.892</td><td>0.795</td><td>0.859</td><td>0.806</td></tr><tr><td>Qwen2-VL-7B</td><td>0.971</td><td>0.971</td><td>0.892</td><td>0.778</td><td>0.774</td><td>0.852</td></tr></table>

Table 4: Sample accounting and uncertainty summary for headline reliability claims. Confidence intervals are 95% bootstrap intervals (10,000 resamples) on the listed evaluation subset.
<table><tr><td>Quantity</td><td>Value</td><td>Subset / 95% CI</td></tr><tr><td>POPE (Adversarial) sample count</td><td> $n = 1 , 0 0 0$ </td><td>fixed evaluation split</td></tr><tr><td>LLaVA-Bench sample count</td><td> $n = 9 0$ </td><td>fixed evaluation split</td></tr><tr><td>Custom counting + spatial sample count</td><td> $n = 2 , 0 0 0$ </td><td> $1 , 0 0 0 + 1 , 0 0 0$ </td></tr><tr><td>Pooled structural-analysis set</td><td> $n = 3 , 0 9 0$ </td><td>used for  $R ( C _ { k } , y ) , R ( H _ { s } , y )$ </td></tr><tr><td> $R ( C _ { k } , y )$ </td><td>0.001</td><td>95%  $\operatorname { C I } \left[ - 0 . 0 \dot { 3 } 4 , 0 . 0 3 6 \right]$ </td></tr><tr><td> $R \big ( H _ { s } , \dot { y } \big )$ </td><td>-0.012</td><td>95%  $\mathrm { C I } \left[ - 0 . 0 4 7 , 0 . 0 2 4 \right]$ </td></tr><tr><td>Precision at  $\mathrm { S C } { = } 1$ </td><td>90.8%</td><td> $9 5 \% \mathrm { C i } [ 8 8 . 4 , 9 2 . 8 ] \%$ </td></tr></table>

We conducted a supervised stress test to close potential loopholes: on the pooled cross-family split used in this section, an XGBoost-Random Forest ensemble trained on 11 attentionderived features (including polynomial interactions) with full access to ground-truth labels achieved only 52–55% accuracy, which is near chance. In a separate architecture-specific setting (Appendix Table 8), a deeper supervised attention probe reaches AUROC 0.725, indicating limited but non-dominant signal from attention structure.

Causal Role: Despite correlation failure, attention is causally necessary. Masking the top 30% attended patches reduces LLaVA accuracy by 8.2pp and PaliGemma by 11.3pp $( p < \dot { 0 } . 0 0 1 )$ This reveals a critical distinction: attention patterns enable feature extraction but do not encode uncertainty about those features.

## 5.2 Logit Lens: Tracing the Emergence of Reliability

To move beyond simple correlation, we investigate where reliability signals mechanically emerge. We apply the Logit Lens technique Nostalgebraist (2020), projecting the hidden state $h _ { l }$ of layer l directly into the vocabulary space. We define the Truth Margin $\Delta \mathcal { M } _ { l }$ as the logit difference between the correct token and the top incorrect token. Cross-family peak layers, final margins, and MLP contributions are summarized in Table 2.

Visual Integration is Late and MLP-Dominated. Tracking $\Delta \mathcal { M } _ { l }$ reveals a distinct ”Silent Phase” in some families (Figure 2, Left). Reliability signals do not accumulate linearly: some models remain near zero for many layers before a late surge, while others peak earlier or reseparate cyclically. To avoid terminology drift, we report two peak definitions throughout: a visual-integration peak $l _ { \mathrm { v i s } } ^ { \star }$ (maximum correct-vs-incorrect separation) and a final-margin peak $l _ { \mathrm { f i n a l } } ^ { \star }$ (maximum absolute $\Delta \mathcal { M } _ { l } )$

1. MLP vs. Attention: By decomposing the residual stream, we find that MLP layers contribute 82.1% of the margin growth at the peak. This indicates that reliability is a product of feature processing (MLP) rather than token routing (Attention).

(a) Layer-wise Truth Margin Trajectory  
![](images/3029e9295dd77868815f5a00607bb612b4e102c77bb673f208e0ae91717e83ec.jpg)

(b) Sparse Circuit: Neuron Distribution  
![](images/a291ff44bb05d44d09f409894c621fc38b3a4a724924dae26461b36aa526baa0.jpg)  
Figure 2: Mechanistic analysis of reliability emergence. (a) Left panel: Transformer layer index l (x-axis) vs. truth margin $\Delta \mathcal { M } _ { l } \ \mathrm { ( y { - } a x i s ) }$ . Model families display distinct temporal integration profiles: late-emergent $( \mathbf { L L a V } \dot { \mathbf { A } } ,$ , solid blue), earlier-peaking (PaliGemma, dashed red), and cyclical (Qwen2-VL, dotted green). (b) Right panel: Probe neuron activation shift (x-axis) vs. population density (y-axis). The distribution highlights a dense near-zero bulk (most neurons are inactive for truth prediction), alongside sparse, highly predictive outliers (green = success neurons, red = failure neurons) that drive probe discrimination.

2. Architecture Divergence: While LLaVA delays integration, PaliGemma integrates early (Peak L14), validating that ”Symbolic Detachment” is an architectural choice, not a universal law.

## 5.3 Sparse Reliability Circuits: Localizing Reliability-Associated Neurons

If reliability signals exist in the MLP layers, are they distributed holistically or localized? We trained $L _ { 1 } .$ -regularized sparse logistic regression probes $( \lambda = 0 . 1 )$ on the internal activations.

Layer Specificity Analysis. To address why we focus on Layer 31, we conducted multi-layer ablation experiments targeting the same top-5 neurons across layers 10, 17, 21, 27, 29, and 31. Results show minimal differentiation: ablating at any layer produces <1% accuracy change from baseline (54.5%). Critically, single-neuron ablation of all five reliability-associated neurons — including extreme activation clamping (±100) — produced zero measurable accuracy change $( \Delta = 0 . 0 \mathrm { p p } , p = 1 .$ 00 for all neurons). Only simultaneous ablation of all top-5 probe neurons produced a measurable effect (−2.0pp overall, −8.3pp on object identification), while ablating 5 random neurons produced no effect. This confirms two things: (1) no single neuron is a causal bottleneck, and (2) reliability is encoded in a localized circuit across a handful of neurons rather than a single isolated unit. across multiple neurons rather than isolated units.

Table 5: Causal Ablation Results (LLaVA-1.5, Layer $\mathbf { 3 1 } , n { = } 2 \mathbf { 0 0 } )$ . Ablating probe-identified neurons causes measurable accuracy drops, while random neurons show no effect. Effect is strongest for object identification questions.
<table><tr><td>Ablation Condition</td><td>Overall Acc.</td><td>Object ID Acc.</td><td>∆ Overall / Object-ID (pp)</td></tr><tr><td>Baseline (no ablation)</td><td>54.5%</td><td>100.0%</td><td>N/A</td></tr><tr><td>Single neuron (N1512)</td><td>54.5%</td><td>100.0%</td><td>0.0/0.0</td></tr><tr><td>Top 5 probe neurons</td><td>52.5%</td><td>91.7%</td><td>-2.0/-8.3</td></tr><tr><td>Random 5 neurons (control)</td><td>54.5%</td><td>100.0%</td><td>0.0/0.0</td></tr></table>

## 5.4 Architectural Robustness: Late Bottlenecks vs. Distributed Circuits

While LLaVA exhibits measurable failure when small sets of strongly predictive neurons are ablated (−8.3pp on Object ID for just 5 neurons), we find this ”fragility” is highly specific to its architecture. To determine if this bottleneck phenomenon holds across modern VLM families, we extend our causal interventions to PaliGemma (Layer 15) and Qwen2-VL (Layer 25).

Unlike LLaVA, ablating the top-10 most predictive neurons in PaliGemma and Qwen2-VL produces absolutely no deviation in accuracy (< 0.7pp). This suggested their representations might be fundamentally distributed. To test this hypothesis, we applied aggressive ablation scaling, randomly destroying up to > 50% of the hidden dimension in their most predictive layers.

Remarkably, PaliGemma suffers only a 1.0% accuracy drop even when 1,000 neurons (∼ 50% of the layer’s 2048 hidden size) are destroyed. Similarly, Qwen2-VL shows extreme resilience: ablating up to 2,000 neurons (> 55% of its 3584 residual dimension) causes zero measurable degradation (∆ bounds of ±2.0pp). We confirm this is not merely a token-routing artifact by completely bypassing the MLP output for all tokens at Layer 25 in Qwen2-VL, which still yields fully robust performance.

Table 6: Large-Scale Causal Ablation Results. Unlike LLaVA’s localized fragility, PaliGemma and Qwen2-VL exhibit extreme causal robustness. Ablating up to half of their most predictive layers produces negligible impact on generation accuracy, highlighting highly distributed internal circuits. Accuracies are reported on an n = 100 causal validation split, with ∆ showing the deviation from the architecture’s local baseline.
<table><tr><td>Model</td><td>Ablation Condition</td><td>Split Acc.</td><td>∆ from Baseline (pp)</td></tr><tr><td rowspan="4">PaliGemma (Layer 15)</td><td>Baseline</td><td>97.0%</td><td></td></tr><tr><td>Top-10 Predictive Neurons</td><td>96.3%</td><td>-0.7</td></tr><tr><td>500 Random Neurons (24%)</td><td>97.0%</td><td>0.0</td></tr><tr><td>1,000 Random Neurons (49%)</td><td>96.0%</td><td>-1.0</td></tr><tr><td rowspan="5">Qwen2-VL (Layer 25)</td><td>Baseline</td><td>55.0%</td><td></td></tr><tr><td>500 Random Neurons (14%)</td><td>58.0%</td><td>+3.0</td></tr><tr><td>1,000 Random Neurons (28%)</td><td>56.0%</td><td>+1.0</td></tr><tr><td>2,000 Random Neurons (56%)</td><td>57.0%</td><td>+2.0</td></tr><tr><td>Complete MLP Bypass (All Tokens)</td><td>65.0%</td><td>+5.0 (valid. split var.)</td></tr></table>

This confirms our logit lens analysis: Qwen2-VL’s “Cyclical Refinement” and PaliGemma’s early visual integration represent fundamentally different architectural strategies than LLaVA. They distribute the reliability computation across a wide manifold of subsequent layers, allowing the residual stream to effortlessly patch missing representational lobes. In contrast, LLaVA “locks” its prediction unrecoverably at a late-stage bottleneck, rendering its reasoning structurally fragile.

## 5.5 Reliability Prediction: Probes Outperform Attention

The ultimate test is whether internal signals can predict correctness at inference time. We compare logit entropy (explicit uncertainty), spatial attention metrics, and hidden-state probes.

Finding: Standard uncertainty baselines fail. Logit entropy achieves AUROC ≈ 0.50, confirming poor calibration, and spatial attention remains near random (AUROC = 0.50). Probe gains are strongest on POPE/LLaVA-Bench and mixed on the added VQA tasks: for VQA v2/TextVQA cells in Table 3, probe outperforms output confidence in 3 of 6 model-task comparisons (both LLaVA tasks and Qwen2-VL on TextVQA), while output confidence is stronger for PaliGemma on both tasks and Qwen2-VL on VQA v2. This pattern indicates that hidden-state probes are a strong reliability readout but remain benchmark- and architecturedependent. Self-consistency achieves R = 0.429, substantially outperforming all visual metrics but requiring 10× inference cost.

PaliGemma shows lower POPE/LLaVA-Bench probe performance (0.738) because it integrates visual signals earlier and has a shallower decoder, leaving less late-layer separation between correct and hallucinated trajectories. This weakens probe margin contrast relative to LLaVA/Qwen2-VL but still keeps hidden-state signals stronger than attention-only metrics.

Table 7: Reliability Prediction: Method Comparison (POPE Adversarial split). AUROC scores for predicting answer correctness across signal sources. Spatial attention is near random, while hidden-state probes provide the strongest single-pass reliability signal. Selfconsistency provides good signal but requires 10× inference cost.
<table><tr><td>Method</td><td>LLaVA-1.5</td><td>PaliGemma</td><td>Qwen2-VL</td></tr><tr><td colspan="4">Baseline Metrics</td></tr><tr><td>Spatial Attention  $( H _ { s } , C _ { k } )$ </td><td>0.50</td><td>0.50</td><td>0.50</td></tr><tr><td>Logit Entropy</td><td>0.50</td><td>0.52</td><td>0.51</td></tr><tr><td>Output Confidence</td><td>0.54</td><td>0.55</td><td>0.53</td></tr><tr><td colspan="4">Our Probes</td></tr><tr><td>Margin-only (∆Ml)</td><td>0.72</td><td>0.70</td><td>0.63</td></tr><tr><td>Hidden-State Probe (Best Layer)</td><td>0.956</td><td>0.738</td><td>0.971</td></tr><tr><td>Combined (Last 5 Layers)</td><td>0.956</td><td>0.738</td><td>0.970</td></tr><tr><td colspan="4">Behavioral (10× cost)</td></tr><tr><td>Self-Consistency (K=10)</td><td>0.78</td><td>0.81</td><td>0.79</td></tr></table>

## 5.6 Symbolic Detachment: Why Attention Fails

Layer-wise attention evolution reveals the mechanism behind the Cluster Failure (Figure 3). LLaVA exhibits “Early Locking”: attention sharpens dramatically at Layer 2 (∆H ≈ −2.5), then stagnates for 28 layers before diffusing at the final layer $( \bar { \Delta } H _ { s } \approx + 1 . 0 )$ . By the time information reaches the output, the model has “let go” of specific visual features.

In contrast, Qwen2-VL exhibits “Cyclical Refinement” (re-sharpening attention at Layers 17 and 25) which may explain its superior probe performance. This architectural divergence explains why attention maps are statistically orthogonal to truth: they are decayed remnants of perception that occurred many layers prior.

Architectural Drivers of Early Locking: Late-Stage Forcing. To investigate family-specific attention dynamics, we measured the layer-wise residual update magnitude $( \lVert h ^ { ( l ) } - \dot { h } ^ { ( \bar { l } - 1 ) } \rVert _ { 2 } )$ on visual tokens. As shown in Appendix Figure 4, some architectures exhibit relatively low and stable updates through middle layers followed by a sharp late-stage increase. This suggests that, rather than continuously refining visual features, certain projection pipelines perform a delayed “translation” into the linguistic space used for next-token prediction. More broadly, this supports our central claim: alignment between visual evidence and final verbal output is architecture-dependent and may be introduced late in the stack.

## 6 Discussion

The results above challenge the intuition that reliable multimodal generation is directly readable from visual attention maps.

## 6.1 The Illusion of Grounding

Across models, structural attention metrics are weak predictors of correctness $( R ( C _ { k } , y ) =$ 0.001, $R ( H _ { s } , y ) = - 0 . 0 1 2 )$ , and even supervised attention features remain limited in reliability prediction. On our pooled cross-family split, attention-feature classifiers stay near chance (52–55%); in a separate architecture-specific setting, a deeper supervised attention probe reaches AUROC 0.725 but still trails hidden-state probes and self-consistency. The

![](images/a7c5ac7e259f46285116b46d437e6a114c31e4665a3d03a038adde759fdcc303.jpg)  
LLaVA-1.5 (solid blue): Early lock, late diffuse Qwen2-VL (dashed green): Cyclical refinement PaliGemma (dotted red): Steady decay

Figure 3: Symbolic Detachment: Attention Evolution Across Layers. We track the relative change in spatial attention entropy (∆H , y-axis) across normalized transformer layers (l/L, x-axis) for three VLM families. LLaVA (solid blue circles) exhibits “Early Locking,” where entropy drops sharply at Layer 2 and stagnates before diffusing rapidly at the final layer. Qwen2-VL (dashed green triangles) shows “Cyclical Refinement,” continuously resharpening its attention in deeper layers. PaliGemma (dotted red squares) shows a steady decay. This architectural divergence explains why early spatial attention is decorrelated from final reliability: for prefix-based models like LLaVA, visual attention patterns become “stale” long before the final reasoning and decision-making step occurs.

practical takeaway is that spatial attention is functionally important for feature extraction, yet poorly calibrated as an uncertainty signal.

## 7 Conclusion

This study reveals that reliability and causal robustness in current VLMs are highly architecture-dependent and not well captured by attention-map structure alone. We find a stark architectural divergence: early-fusion and cyclically refining models (PaliGemma, Qwen2-VL) distribute their truth representations, remaining resilient even when ∼ 50% or more of their peak informational neurons are destroyed. Conversely, late-fusion models like LLaVA rely on localized, fragile late-stage bottlenecks.

For reliability prediction, stronger signals come from generation dynamics and internal-state probes: self-consistency provides the best behavioral proxy for correctness (R = 0.429), and hidden-state probes achieve high discrimination (AUROC > 0.95 on our strongest settings). Ultimately, these findings support a practical direction for trustworthy multimodal systems: use latent-state and consistency-based monitors rather than heatmap sharpness, and favor distributed, early-fusion architectures for causally robust multimodal reasoning.

## References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, et al. Flamingo: A visual language model for few-shot learning. In NeurIPS, 2022.

Lucas Beyer, Andreas Steiner, Alejandro Pinto, et al. Paligemma: A versatile 3b visionlanguage model for transfer. arXiv preprint arXiv:2407.07726, 2024.

Archie Chaudhury and Shikhar Shiromani. Chameleonbench: Quantifying alignment faking in large language models. In Proceedings of Machine Learning Research (ACML 2025), 2025. PMLR 304.

Wenliang Dai, Junnan Li, Dongxu Dong, Anthony Meng Huat Tiong, Steven Li, Silvio Savarese, and Steven CH Hoi. Instructblip: Towards general-purpose vision-language models. In NeurIPS, 2023.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

Sarthak Jain and Byron C Wallace. Attention is not explanation. NAACL, 2019.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seedbench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023a.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven C. H. Hoi. Blip: Bootstrapping languageimage pre-training for unified vision-language understanding and generation. In ICML, 2022.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In EMNLP, 2023b.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.

Yifan Liu, Zhen Chen, Rui Wang, and Wayne Xin Zhao. Seeing but not believing: Vision-language models can attend correctly yet reason incorrectly. arXiv preprint arXiv:2510.17771, 2025.

Lin Long, Changdae Oh, Seongheon Park, and Sharon Li. Understanding language prior of lvlms by contrasting chain-of-embedding. arXiv preprint arXiv:2509.23050, 2025.

Nostalgebraist. The logit lens: Understanding hidden state dynamics in language models. arXiv preprint arXiv:2012.08981, 2020.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. EMNLP, 2018.

Kenji Sahay, Snigdha Pandya, Rohan Nagale, Anna Lin, Shikhar Shiromani, Kevin Zhu, and Dev Sunishchal. Compass: Context-modulated pid attention steering system for hallucination mitigation. arXiv preprint arXiv:2511.14776, 2025.

Shikhar Shiromani, Archie Chaudhury, and Sri Pranav Kunda. The hypocrisy gap: Quantifying divergence between internal belief and chain-of-thought explanation via sparse autoencoders. arXiv preprint arXiv:2602.02496, 2026.

Rohan Subramanian Thomas, Shikhar Shiromani, Abdullah Chaudhry, Ruizhe Li, Vasu Sharma, Kevin Zhu, and Sunishchal Dev. Promoral-bench: Evaluating prompting strategies for moral reasoning and safety in llms. arXiv preprint arXiv:2602.13274, 2026.

Peng Wang, Xue Li, et al. Qwen2-vl: Enhancing vision-language model perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning. arXiv preprint arXiv:2203.11171, 2022.

Sarah Wiegreffe and Yuval Pinter. Attention is not not explanation. In EMNLP, 2019.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

Li Zhou, Wenhui Fu, Yujie Chen, Wei Liu, Zongjing Lin, Shuicheng Yan, and Weiyang Chen. Llava-bench: A benchmark for visual instruction following. In arXiv preprint arXiv:2308.13692, 2023.

## A Appendix

## A.1 Detailed Methodology and Metric Definitions

## A.2 Detailed Experimental Setup

Models: We evaluate three VLM architectures: LLaVA-1.5-7B (32 layers, CLIP ViT-L/14 encoder), PaliGemma-3B (18 layers, SigLIP encoder), and Qwen2-VL-7B-Instruct (28 layers, native multimodal) Liu et al. (2023); Beyer et al. (2024); Wang et al. (2024). All experiments use HuggingFace implementations on NVIDIA A100 GPUs.

Datasets: We evaluate on: (1) POPE Li et al. (2023b) for object hallucination (Adversarial split, 1,000 samples), (2) LLaVA-Bench Zhou et al. (2023) for open-ended reasoning (90 questions), and (3) Custom Counting & Spatial Tasks (2,000 samples total: 1,000 counting + 1,000 spatial-relation prompts). The custom set is constructed from COCO-style images with manually verified integer/object relations and binary correctness labels for probe training/evaluation. To test probe generalization beyond these splits, we further expand evaluation to VQA v2 (scene-understanding questions) and TextVQA (OCR-heavy questions), and report task-specific reliability AUROC in Table 3.

Metrics: For reliability prediction, we report Point-Biserial Correlation $( R _ { p b } )$ with binary correctness and AUROC. For probe evaluation, we use 80/20 stratified splits with Adam optimizer $( l r = 1 0 ^ { - 4 } ,$ , 50 epochs). Self-consistency uses $K = 1 0$ samples with nucleus sampling $( p = 0 . 9 , T = 0 . 7 )$ . For structural concentration, we build a binary attention mask from the top-30% attention mass and compute connected components on the patch grid. We report both total component count $K _ { \mathrm { t o t a l } }$ and secondary-component count $\dot { C } _ { k } \equiv \breve { K } _ { \mathrm { t o t a l } } - 1$ after removing the dominant component; thus $C _ { k } = 0$ indicates a single dominant focus. We additionally verified robustness with a DBSCAN variant $( \epsilon = 1 . 5 )$ , min samples= 3). Full implementation details are in Appendix A.11.

## A.3 Extended Analysis: The Ensemble Attention Probe

In Section 5.1, we briefly introduced the “Ensemble Attention Probe.” Here, we provide a detailed breakdown of its architecture and performance relative to other methods.

Motivation: The failure of unsupervised metrics (Cluster Count $C _ { k } )$ suggested that reliability is not encoded in simple geometric properties of the attention map (e.g., “is it sharp?”). However, we hypothesized that reliability might be encoded in high-dimensional patterns across multiple layers, patterns too complex for human inspection but accessible to a nonlinear classifier.

Architecture: We extracted cross-attention tensors $A ^ { ( l , h ) } \in \mathbb { R } ^ { T \times S }$ from all $L = 3 2$ layers of the Vicuna-7B backbone, then averaged over heads h and answer-token indices t to obtain a per-layer spatial vector $m ^ { ( l ) } \in \mathbb { R } ^ { S }$

• Input: A concatenated vector of per-layer spatial vectors:

$$
\boldsymbol { x } = \mathrm { C o n c a t } ( m ^ { ( 1 ) } , \dots , m ^ { ( 3 2 ) } )\tag{1}
$$

• Model: A 3-layer Multi-Layer Perceptron (MLP) with ReLU activations and Dropout $( p = 0 . { \dot { 1 } } )$ .

• Dimensions: Input $d _ { i n } = 3 2 \times 5 7 6 = 1 8 , 4 3 2  1 0 2 4  5 1 2  1$ (Binary Classification).

Results & Comparison: Table 8 details the performance of various probes. While the Ensemble Attention Probe significantly outperforms random chance and simple visual entropy, it remains inferior to Self-Consistency. This reinforces our main finding: generation dynamics (consistency) are a stronger signal than internal state snapshots.

Table 8: Probe Performance Comparison. The Supervised Ensemble (Idea 4) extracts some signal, but Consistency (Behavioral) remains superior.
<table><tr><td>Method</td><td>Type</td><td>AUROC</td><td>Cost (Inference)</td></tr><tr><td>Random Baseline</td><td>Statistical</td><td>0.500</td><td>1x</td></tr><tr><td>Focus Entropy (Hs)</td><td>Unsupervised Visual</td><td>0.504</td><td>1x</td></tr><tr><td>Cluster Count  $( C _ { k } )$ </td><td>Unsupervised Visual</td><td>0.501</td><td>1x</td></tr><tr><td>Linear Probe  $( h _ { l a s t } )$ </td><td>Supervised Ling.</td><td>0.620</td><td>1x</td></tr><tr><td>Ensemble Probe</td><td>Supervised Attn.</td><td>0.725</td><td>1x</td></tr><tr><td>Self-Consistency (SC)</td><td>Behavioral</td><td>0.784</td><td>10x</td></tr></table>

## A.4 The Counting Anomaly: Severe Miscalibration

A critical discovery in our baseline testing was the model’s behavior on quantitative reasoning tasks. We refer to this as the “Counting Anomaly.”

The Phenomenon: On tasks asking “How many [objects] are in the image?”, the evaluated VLM families exhibit severe miscalibration. As shown in our data, the model often assigns extremely high probability (> 90%) to incorrect integers.

Case Study: Consider an image with 3 baseball players.

• Ground Truth: 3

• Model Prediction: ${ } ^ { \prime \prime } \mathrm { F o u r } ^ { \prime \prime }$

• Token Confidence $( P _ { t o k } ) \colon 9 2 \% \ ( \mathrm { V e r y \ : H i g h } )$

• Total Visual Clusters $( K _ { \mathrm { t o t a l } } ) \colon 3$ distinct clusters (equivalently $C _ { k } = 2$ after removing the dominant component).

This dissociation highlights a “Symbolic Detachment.” The visual encoder correctly identifies 3 regions (verified by $K _ { \mathrm { t o t a l } } = 3$ , hence $C _ { k } = 2 )$ , but the projection into the language space maps these features to the token ${ } ^ { \prime \prime } \mathrm { F o u r } . { } ^ { \prime \prime }$ Because the language model is autoregressively coherent, it assigns high probability to the token “Four” despite being factually grounded in “Three” visual features.

Conclusion: Token probability measures the model’s fluency, not its grounding. Self-Consistency mitigates this because, in the miscalibrated state, the model is likely to oscillate between “Four” and “Three” across different sampling temperatures, lowering the SC score.

## A.5 Architectural Drivers of Early Locking: Residual Update Analysis

To investigate the architectural drivers behind LLaVA’s “Early Locking” and “Symbolic Detachment” discussed in Section 5.6, we extracted the hidden states of the 576 visual tokens at every layer of the LLaVA-1.5-7B architecture. We then computed the average L<sub>2</sub> norm of the residual updates $( \lVert h ^ { ( l ) } - h ^ { ( l - 1 ) } \rVert _ { 2 } )$ to measure how actively the model processes visual features at each depth.

As shown in Figure 4, the visual token representations remain remarkably dormant across the middle 25 layers of the network. Because the visual representations are not actively updated during these middle layers, the spatial attention maps naturally stagnate (the “Early Locking” phenomenon). The model applies massive non-linear transformations to these features only in the final three layers to extract confidence and generate text, directly corroborating our Logit Lens findings that true visual-linguistic grounding occurs at the end of the network.

![](images/16d00016ca3bb95fc5f79cfdb5e0053c1212e9f5a879856a8c2c577c729068dc.jpg)  
Figure 4: Visual Token Updates: Late-Stage Transformation in LLaVA. We plot the average L norm of the residual updates $( \lVert h ^ { ( l ) } - h ^ { ( l - 1 ) } \rVert _ { 2 } )$ for the 576 visual tokens across all 32 transformer layers. The representations remain largely dormant across the middle layers (Layers 5–28), explaining the stagnation of early attention maps. A massive non-linear transformation occurs only in the final layers (Layers 30–32), forcing the alignment between visual perception and linguistic output.

## A.6 Qualitative Failure Analysis

We analyzed specific instances where the “Attention-Confidence Assumption” broke down.

False Negatives (Good Attention, Bad Answer): In 15% of failure cases, the attention map was “perfect” (low entropy, high clustering on relevant objects). For example, in a POPE object-existence query, the model attended solely to a chair while answering “No” to “Is there a chair?”. This suggests that the attention mechanism acted as a retrieval query that successfully found the feature, but the LLM decoder failed to interpret the retrieved feature as “existence.”

False Positives (Bad Attention, Good Answer): In 22% of correct cases, the model exhibited “scattered” attention (high entropy, $H _ { s } > 4 . 5 )$ . This frequently occurred in background scene questions (e.g., “Is this a rainy day?”). The model likely relied on global texture features pooled from the entire image rather than specific object attention, yet standard interpretability metrics would penalize this as “unfocused.”

## A.7 Extended Case Study: Why Attention Fails and Consistency Succeeds

To concretely illustrate the disconnect between visual attention and reliability, we present an actual failure case from our VQAv2 experiments (Figure 5).

Why Attention Fails: This example starkly illustrates the “Cluster Failure.” The model’s attention exhibits ideal structural properties: entropy $H _ { s } = 0 . 3 2 1$ places it in the bottom 15% (highly focused), and a single dominant focus $\overset { \bullet } { ( } \overset { \ j } { C } _ { k } = 0$ under our definition) suggests the model is “looking” at a specific region. By all attention-based metrics, this should be a reliable prediction. Yet the model hallucinates the absence of a collar that is clearly visible. The failure occurs because attention captures where features were extracted, not whether those features were correctly interpreted. The visual encoder successfully attends to the dog, but the downstream LLM fails to bind the “collar” concept to the perceived visual features.

![](images/34fa0310c6f44e8e9fcb00fd24c67bbe3bcf959f16696beceaef38072aa57fdb.jpg)  
Figure 5: Case Study: High-Quality Attention, Wrong Answer (PaliGemma, Sample #31). The image shows a dog on a surfboard clearly wearing a red collar. The model answers “No” despite exhibiting excellent attention: very low entropy (H<sub>s</sub> = 0.321, bottom 15% of dataset) and a single dominant focus (C = 0 under our connected-component definition in Appendix A.2). Attention-based metrics would classify this as trustworthy. However, the logit lens reveals that the correct token “Yes” is suppressed at layer 14, correctly identifying unreliability.

Why Logit Lens Succeeds: Probing the hidden states reveals the failure mechanism. The correct token “Yes” gains probability through layers 0–10 as visual features are processed, but is sharply suppressed at layer 14: the peak visual integration point (∆margin = +9.57). This suppression pattern, detectable by our hidden-state probes, correctly flags the prediction as unreliable. The model’s internal trajectory reveals uncertainty that the final output masks.

This case exemplifies our core finding: looking well is not knowing well. A model can attend perfectly to the right region and still hallucinate.

## A.8 Cross-Model Experiment Details

## A.9 Family-Specific Reliability Patterns

To complement the per-model deep dives, we summarize family-specific behaviors: LLaVA-1.5 shows a long early-lock plateau followed by late diffusion and strong final-layer probe signal; PaliGemma-3B shows earlier integration and weaker late-layer margin separation; and Qwen2-VL-7B shows iterative re-integration cycles with strong late reliability separation.

We conducted extensive experiments across three VLM architectures to validate generality. Model Architectures:

• LLaVA-1.5-7B: 32 transformer layers, 32 attention heads per layer. Uses frozen CLIP ViT-L/14 visual encoder with Vicuna-7B language backbone. Visual tokens projected via 2-layer MLP.

• PaliGemma-3B (Google): 18 transformer layers, 8 attention heads per layer. Uses SigLIP visual encoder with Gemma language backbone. Visual tokens projected via linear layer.

• Qwen2-VL-7B-Instruct (Alibaba): 28 transformer layers with Grouped Query Attention (28 heads, 4 KV heads). Native multimodal architecture with interleaved visual tokens and dynamic resolution support.

## A.10 Model-Specific Deep Dive: LLaVA

Key Insight: Correctness emerges before final answer selection. Margin trajectories diverge at Layer 21 and peak at Layer 24, suggesting reliability is determined in mid-layers, not at the final output. In our notation, this corresponds to $l _ { \mathrm { v i s } } ^ { \star } = 2 4$ , while the maximum absolute final margin occurs at $l _ { \mathrm { f i n a l } } ^ { \star } = 3 1$ . Table 9 presents the complete LLaVA analysis.

Table 9: Model-Specific Complete Analysis (LLaVA-1.5-7B). Layer-wise computational pipeline, neuron-level findings, and causal validation.
<table><tr><td colspan="2">Layer-wise Computational Pipeline</td><td rowspan="2">∆margin</td><td rowspan="2">Dominant Component</td></tr><tr><td>Layers</td><td>Role</td></tr><tr><td>0-16</td><td>Feature extraction</td><td>Low variance</td><td>N/A</td></tr><tr><td>17</td><td>Early prediction</td><td>N/A</td><td>82.3% probe accuracy</td></tr><tr><td>19</td><td>Early boosting</td><td>+0.53</td><td>MLP</td></tr><tr><td>21-28</td><td>Suppression</td><td>-0.85 to -2.27</td><td>Attention (72%)</td></tr><tr><td>24</td><td>Max separation</td><td>N/A</td><td>Largest correct/incorrect gap</td></tr><tr><td>29</td><td>Neuron commitment</td><td>N/A</td><td>86.3% probe, 5.7% sparse</td></tr><tr><td>30</td><td>Answer boosting</td><td>+2.61</td><td>MLP</td></tr><tr><td>31</td><td>Final decision</td><td>+9.20</td><td>MLP (72%)</td></tr><tr><td colspan="2">Key Neurons (Layer 31)</td><td></td><td></td></tr><tr><td>Neuron ID</td><td>Type</td><td>∆activation</td><td>Functional Role</td></tr><tr><td>1512</td><td>Success</td><td>+27.23</td><td>Answer confidence</td></tr><tr><td>1360</td><td>Failure</td><td>-3.11</td><td>Failure detection</td></tr><tr><td>3839</td><td>Failure</td><td>-3.08</td><td>Failure detection</td></tr><tr><td>2660</td><td>Failure</td><td>-2.95</td><td>Failure detection</td></tr></table>

## A.11 Implementation and Hardware Details

All experiments were conducted on compute clusters provided by RunPod and Lambda Labs, using NVIDIA A100 GPUs (80GB VRAM), AMD EPYC 7742 64-Core CPUs, and 512 GB system memory. The software stack used PyTorch 2.1.0 with CUDA 12.1 and the HuggingFace transformers library with official checkpoints for LLaVA, PaliGemma, and Qwen2-VL Liu et al. (2023); Beyer et al. (2024); Wang et al. (2024). Attention extraction was implemented via PyTorch register forward hook hooks on decoder MultiheadAttention modules in each family’s multimodal-integration regime (e.g., late layers for LLaVA and architecture-adjusted regions for PaliGemma and Qwen2-VL).

## A.12 Discussion Extensions: Cross-Family Interpretation and Efficiency Trade-offs

## A.13 Cross-Family Interpretation

Across all three families, the same reliability taxonomy appears with model-specific signatures. LLaVA-1.5 exhibits the strongest symbolic-detachment gap (early lock, late diffusion), which aligns with high probe separability in late layers. PaliGemma-3B integrates visual evidence earlier and more smoothly, yielding weaker late-layer separability and lower probe AUROC (0.738). Qwen2-VL-7B shows cyclical refinement and strong late-stage re-separation, consistent with high probe AUROC (0.971).

These differences suggest that reliability probing should be architecturally adaptive (e.g., layer selection and probe capacity per family), rather than assuming a one-size-fits-all late-layer template.

## A.14 Reliability vs. Efficiency Trade-offs

While Self-Consistency (SC) is the gold standard for reliability (R = 0.43), it comes at a high computational cost: it requires K = 10 forward passes. For real-time applications (e.g., robotics), this is often prohibitive.

Our Hidden State Probe offers a compelling alternative:

• Self-Consistency: High Accuracy (AUROC = 0.78), High Cost (10× inference).

• Learned Probe: Moderate to High Accuracy (up to AUROC = 0.96 on familyspecific splits), Zero Cost (overhead of a single linear layer).

• Visual Metrics: Low Accuracy (AUROC = 0.50), Low Cost.

The success of the Hidden State Probe confirms that the model’s reliability is encoded in the linear subspace of the final residual stream. This aligns with recent work in “Lie Detection” for LLMs, extending it to the multimodal domain. Future work should focus on distilling the signal from Self-Consistency into a single-pass value head, effectively training the model to predict its own consistency score.

## A.15 Limitations and Future Work

Model Scale: Our study focuses on three mid-scale open VLMs. It is possible that larger models (e.g., LLaVA-34B or GPT-4V) exhibit stronger alignment between attention and truthfulness due to better reinforcement learning from human feedback (RLHF).

Computational Cost: The most reliable metric found, Self-Consistency, requires K = 10 inference passes. This is prohibitively expensive for low-latency edge applications.

Causal Evidence Scope: While our ablation experiments demonstrate causal effects of probe-identified neurons (8.3% accuracy drop for top-5 vs. 0% for random), the effect requires ablating multiple neurons simultaneously, suggesting a localized circuit rather than individual “truth units.” The effect is also moderate in magnitude, indicating these neurons are contributors to reliability rather than sole determinants. Future work should explore activation patching and interchange interventions to further characterize the causal mechanism.

Future Direction: We propose that future work should focus on distillation. Since Self-Consistency provides a high-quality “silver label” for reliability (R = 0.43), we can curate a dataset of (Image, Question, Answer, SC-Score) and fine-tune a value head on top of the VLM to predict the SC-Score in a single pass. This would combine the accuracy of consistency with the efficiency of a probe.