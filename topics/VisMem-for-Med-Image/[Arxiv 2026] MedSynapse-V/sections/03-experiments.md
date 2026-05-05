[← 返回 README](../README.md)

# Experiments

## 📌 预览
本文件合并 Experiments/Results/Analysis/Ablation，重点看方法是否真的提升视觉 grounding、医学诊断、鲁棒性和效率。

---

# 3 Experiments

# 3.1 Experimental Setup

Datasets Training data: We conduct comprehensive experiments on seven medical multimodal benchmarks spanning diverse task types and difficulty levels. Stage I (MQPM warmup) uses 50K image–text pairs from PubMedVision [6] covering radiology and pathology. Stage II (CCR) constructs a mixed-modality RL set: 3K closed-ended VQA samples from OmniMedVQA [18] training split (8 modalities: CT, MRI, X-ray, dermoscopy, fundus, OCT, pathology, ultrasound) plus 1K open-ended samples from SLAKE [29] and PathVQA [16] training sets, totaling ∼4K samples. Region masks for $r _ { c a u s a l }$ are provided by Med-SAM3 [28]. Stage III (IMT) reuses the Stage II data. Evaluation benchmarks: (i) Closed-ended VQA: VQA-RAD [20], SLAKE [29], PathVQA [16], PMC-VQA [64]; (ii) Clinical reasoning: MMMU Health & Medicine [60] (denoted MMMU $^ *$ ); (iii) Expert-level reasoning: MedXpertQA-MM [69] (Total score); (iv) Multi-granularity: GMAI-MMBench [57].

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

Baselines We compare against four categories of methods: (1) General VLMs: Qwen3-VL-8B [2] (our base model), InternVL3-8B [67]; (2) Medical-specific VLMs: RadFM [51], LLaVA-Med [24], GMAI-VL [26], HuatuoGPT-Vision [6], BiMediX2- 8B [33], MedMO-8B [9]; (3) RL-enhanced medical reasoning: MedVLM-R1- 2B [37], Med-R1-3B [19], MediX-R1-8B [32], MMedExpert-R1-7B [11]; (4) Latentspace reasoning: Coconut $^ \dagger$ [15], MCOUT-Multi $\dagger$ [38], IVT-LR $^ { \dag }$ [5] ( $^ \dagger$ : adapted with identical Qwen3-VL-8B backbone and training data). We additionally report MedSynapse-V-4B on the Qwen3-VL-4B backbone to assess scalability.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

Implementation Details Our framework builds upon Qwen3-VL-8B-Instruct [2]. The frozen anatomical encoder $\mathcal { E } _ { a n a }$ employs MedSAM3 [28] pre-trained on largescale multi-organ segmentation datasets. Stage I freezes both VLM and $\mathcal { E } _ { a n a }$ , training only the Diagnostic Memory Sampler $\mathcal { P } _ { \phi }$ with $\mathrm { l r } = 2 \times 1 0 ^ { - 4 }$ for 3 epochs. The diagnostic probe count is $N { = } 1 6$ ; $\mathcal { P } _ { \phi }$ is a 2-layer cross-attention Transformer with output dimension $d _ { m } { = } 4 0 9 6$ (matching Qwen3-VL-8B). Images are processed at native dynamic resolution following Qwen3-VL’s default configuration. Stage $\mathrm { I I }$ freezes $\mathcal { P } _ { \phi }$ and adapts VLM via LoRA [17] (rank=64, applied to all attention layers). GRPO generates $G { = } 4$ candidate trajectories per sample, with clipping coefficient $\varepsilon { = } 0 . 2$ , reward weights $\lambda _ { a c c }$ =1.0 and $\lambda _ { c a u s a l } { = } 0 . 5$ , training for 200 steps with a rollout batch size of 32. Max generation length is 1024 tokens. Stage III introduces the Autonomous Memory Module $\mathcal { A } _ { \psi }$ (2-layer MLP + LayerNorm, input from VLM’s visual encoder features), with JSD coefficient $\beta$ =0.5, $\mathrm { l r } = 1 \times 1 0 ^ { - 4 }$ , 3 epochs. For each sample we draw one on-policy trajectory $\hat { y } \sim \pi ^ { - }$ per gradient step; the Stage II data is reused with identical preprocessing. Closed-ended VQA tasks report overall accuracy ( $\%$ ). For GMAI-MMBench and MedXpertQA-MM, we follow their respective official evaluation protocols. Inference efficiency is measured quantitatively as ms/sample and peak GPU memory (GB). More details are provided in the supplementary material.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Table 1: Comprehensive comparison on seven medical benchmarks. Base model: Qwen3-VL-8B (unless otherwise noted). MMMU\* = Health & Medicine track. †: general latent-space methods adapted to medical VQA with identical backbone and training data. $\varDelta$ : absolute gap (pp) to MedSynapse-V (w/ ${ \dot { \varepsilon } } _ { a n a }$ ) average. All results are averaged over five independent runs. Bold: best; underline: second best.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

![Table 4](../images/5b7101edd7a13777d8b79ee831851caad1287b525873c0f14ddeb283ca846986.jpg)
*Table 4: Table 4: Per-modality accuracy ( $\%$ ) on OmniMedVQA. $\varDelta$ : improvement over Qwen3- VL-8B zero-shot baseline.*

> 💡 **Table 4 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

# 3.2 Main Results

As shown in Table 1, MedSynapse-V (w/ $\mathcal { E } _ { a n a }$ ) achieves the highest average of ${ \bf 6 1 . 4 \% }$ , and the encoder-free MedSynapse-V (IMT) retains ${ \bf 5 9 . 6 \% }$ , surpassing all baselines. Compared to the strongest RL baseline MMedExpert-R1 $( 5 5 . 7 \%$ ), MedSynapse-V (IMT) leads by ${ \bf + 3 . 9 }$ pp without any auxiliary module at inference, with the largest margins on visual-grounding benchmarks (VQA-RAD ${ \bf + 9 . 0 }$ , SLAKE +7.0, PathVQA +6.7), where discrete CoT tokens are prone to attenuating early visual evidence across long reasoning chains. On GMAI-MMBench spanning 38 modalities, MedSynapse-V scores $5 4 . 8 \%$ , confirming that the anatomical priors generalize beyond the training distribution.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

RL baselines reveal a specialization dilemma. MediX-R1 benefits from multilingual pretraining and leads on PMC-VQA (56.2%), yet this breadth dilutes radiology-specific precision (VQA-RAD: 56.4%); MMedExpert-R1 achieves the most balanced profile by leveraging guideline-based reward. Small-scale models (MedVLM-R1 2B, Med-R1 3B) collapse on out-of-domain tasks (MedXpert below $1 9 \%$ ), confirming that parameter capacity sets a hard ceiling RL alone cannot raise. n contrast, MedSynapse-V sidesteps this dilemma by injecting latent priors that benefit all task types uniformly, achieving the top performance on every benchmark without task-specific tuning.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

Latent methods require domain priors. Among adapted latent baselines, the hierarchy Coconut (44.5%) $<$ < MCOUT-Multi (47.9%) $<$ < IVT-LR $( 5 0 . 5 \% )$ ) tracks optimization sophistication, yet even IVT-LR barely exceeds zero-shot Qwen3-VL-8B (48.6%). This inversion reveals that latent compression without clinical grounding encodes statistical shortcuts rather than diagnostic logic; the

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

![Figure 4](../images/6143166381cb21b1252cda3a97706934b40f744fdfde3a9aac24b4110202f03e.jpg)
*Figure 4: Fig. 4: Effect of diagnostic probe count $N$ . Performance peaks around $N { = } 1 6$ across benchmarks; further increasing $N$ dilutes diagnostically relevant signals.*

> 💡 **Figure 4 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Table 2: Ablation study on the 8B backbone. All variants undergo the full threestage pipeline including IMT (§2.4); results reflect inference without the anatomical encoder unless stated otherwise. MQPM: Meta Query for Prior Memorization (§2.2); CCR: Causal Counterfactual Refinement (§2.3); $\mathcal { M }$ : diagnostic implicit memory. Best per group in bold.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

![Table 5](../images/ce73b6a94b6f145b107eb3ee8b620cb018c348798fa73fc9c7c3edac3745da80.jpg)
*Table 5: Table 5: Inference latency (single A100, batch $= 1$ ) and parameter count. Prefill: time to first token. End-to-end : full autoregressive decoding (max 128 tokens, greedy).*

> 💡 **Table 5 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

10.9 pp gap to MedSynapse-V confirms that prior injection and causal calibration are prerequisites for effective latent reasoning in medicine.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Scaling efficiency. MedSynapse-V-4B (w/ ${ \mathscr E } _ { a n a }$ ) reaches $5 4 . 7 \%$ with roughly half the parameters of 7B baselines; after encoder removal the IMT variant still achieves $5 2 . 7 \%$ at $8 5 \mathrm { m s }$ and 10.8 GB, surpassing MediX-R1-8B $( 4 9 . 9 \% )$ ). This efficiency stems from a structural advantage: diagnostic expertise is distilled into 16 compact memory vectors consumed in a single forward pass, rather than spread across $1 5 0 +$ verbose reasoning tokens.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

# 3.3 Ablation Study

Table 2 reports comprehensive ablation study results. Specifically, (i) Progressive training stages. MQPM warmup is indispensable: skipping it collapses Avg to 52.9, barely above zero-shot (54.2), because randomly initialized memory destabilizes early RL training. Replacing CCR with SFT reaches 59.2 but lags by $8 . 5 \ \mathrm { p p }$ due to limited out-of-distribution generalization. The full pipeline (Avg 67.7) confirms non-redundant contributions: MQPM grounds semantics, CCR refines via exploration, IMT compresses into an autonomous pathway. (ii) Reward design. rcausal is the dominant reward component ( $^ +$ 4.1 pp, $6 3 . 6 \to 6 7 . 7$ ). Without causal pressure the model bypasses $\mathcal { M }$ via direct shortcuts, treating memory as inert padding; the counterfactual intervention penalizes trajectories insensitive to diagnostic regions. The effect concentrates on radiology benchmarks and persists after IMT, indicating stronger memory utilization transfers more faithfully through distillation. (iii) Encoder retention vs. removal. IMT achieves near-lossless removal: only $1 . 4 ~ \mathrm { p p }$ degradation $6 9 . 1  6 7 . 7 $ ) while latency drops 39% and memory decreases 6.3 GB. The gap is not uniform: core VQA metrics degrade minimally, whereas MedXpert and GMAI suffer more, suggesting complex reasoning depends more on encoder-derived priors than closed-ended recognition. (iv) Anatomical encoder choice. MedSAM3 outperforms SAM-Med2D by 4.4 pp (67.7 vs. 63.3), reflecting richer spatial representations from multi-organ segmentation pretraining. Random initialization yields only 52.0, confirming that gains originate from what the encoder knows, rather than how memory is aggregated. (v) Probe count $N$ . As shown in Fig. 4, $N { = } 1 6$ balances expressiveness against redundancy. The CCR to SFT gap widens with $N$ (3.5 pp at $N { = } 4$ vs. $7 . 2 \mathrm { \ p p }$ at $N { = } 1 6$ ), revealing that larger memory pools amplify bypass shortcuts and therefore benefit disproportionately from causal refinement.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Figure 5](../images/953e6d177886aadcd5d391a32a867c5e8bb39982f95e976011318be129cc4637.jpg)
*Figure 5: Fig. 5: Qualitative comparison across CT, MRI, and Ultrasound cases. MedSynapse-V produces concise, correct diagnoses, while Med-R1 and MMedExpert-R1 generate verbose CoT with hallucinated findings (red) leading to misdiagnoses.*

> 💡 **Figure 5 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

# 3.4 In-Depth Case Analysis

As illustrated in Fig. 5, we compare MedSynapse-V with Med-R1 [19] and MMedExpert-R1 [11] across three distinct imaging modalities. Both baselines produce verbose CoT reasoning ( $\sim$ 185–238 tokens) yet arrive at incorrect diagnoses due to hallucinated observations erroneously propagating through the chain. In the CT case, Med-R1 fabricates pleural thickening in the left upper lobe, while MMedExpert-R1 hallucinates a laminated calcification pattern and mischaracterizes the nodule as a benign granuloma. In the MRI case, Med-R1 misidentifies the extra-axial mass as intra-axial and concludes glioblastoma, whereas MMedExpert-R1 fabricates ring enhancement with central necrosis, both missing the classic meningioma presentation. In the ultrasound case, Med-R1 hallucinates gallbladder wall thickening to over-diagnose acute cholecystitis, while MMedExpert-R1 denies posterior acoustic shadowing and misdiagnoses a gallbladder polyp. In contrast, MedSynapse-V generates concise, correct answers ( $\sim$ 34–44 tokens) without explicit CoT, demonstrating that diagnostic implicit memory provides sufficient latent guidance while avoiding the hallucination cascades inherent in token-level CoT.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 3.5 Efficiency, RL Dynamics, and Latent Space

Performance–efficiency trade-off. As shown in Fig. 6, MedSynapse-V (IMT) achieves $5 9 . 6 \%$ at 2.6 s/sample, comparable to zero-shot Qwen3-VL-8B $4 8 . 6 \%$ , 2.8 s) since both share the same backbone and the 16 memory vectors add negligible overhead. Full-scale 7–8B CoT methods (MediX-R1, MMedExpert-R1) require 5.8 s each due to 300–400 autoregressive reasoning tokens, while smaller CoT models (MedVLM-R1 2B, Med-R1 3B) offset verbosity with faster per-token speed yet remain 18–21 pp below MedSynapse-V. This confirms that compact latent memory provides diagnostic grounding without the token-generation overhead of full-scale CoT.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Training dynamics. Fig. 7 shows the full model $\textit { \textbf { ( w / } \textit { r } _ { c a u s a l } ) }$ improving steadily to ${ \sim } 0 . 8 8$ with a transient exploration dip near step 900, where the policy sacrifices reward to explore memory-reliant generation strategies, while the $w / o \ r _ { c a u s a l }$ ablation plateaus at ${ \sim } 0 . 4 8$ throughout the training. This confirms that accuracy-only reward cannot distinguish memory-dependent from shortcut trajectories; without causal pressure the model bypasses $\mathcal { M }$ entirely, treating injected memory as inert padding.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Figure 6](../images/f0ae86ba491fcc2f228e071dbbd5703f42d7fca1bbd2094bd7e14578efe414d6.jpg)
*Figure 6: Fig. 6: Accuracy–latency trade-off across compared VLM categories.*

> 💡 **Figure 6 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 7](../images/be97594aa2d2e0ada142aae3c720a5fde92bde89e8b8752d67d7fe11321b0a71.jpg)
*Figure 7: Fig. 7: The RL training reward dynamics with and without $r _ { c a u s a l }$ .*

> 💡 **Figure 7 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 8](../images/68f08dc8e3f2f097bb675cbefa34e55f9899509661a3f4321132f92fa0037508.jpg)
*Figure 8: Fig. 8: t-SNE visualization of implicit memory $\mathcal { M } _ { a u t o }$ after CCR. (a) Eight imaging modalities form well-separated clusters with clinically coherent proximity. $( \mathrm { b } , \mathrm { c } )$ Within CT and Pathology, disease subtypes further segregate into distinct regions.*

> 💡 **Figure 8 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Latent space structure. Fig. 8 visualizes the evolved memory $\mathcal { M } _ { a u t o }$ via t-SNE across three granularities. At the cross-modality level (a), eight imaging types form compact clusters with clinically coherent proximity (e.g., CT and MRI lie adjacent; dermoscopy and fundus form a nearby pair). Within individual modalities $\displaystyle ( \mathrm { b } , \mathrm { c } )$ , disease subtypes further segregate: CT memory separates lung nodules, liver lesions, kidney cysts, pneumonia, and aortic aneurysms, while pathology memory distinguishes adenocarcinoma, squamous cell carcinoma, normal tissue, lymphoma, and melanoma. This hierarchical organization confirms that $r _ { c a u s a l }$ reshapes the latent space into a diagnostically meaningful manifold rather than merely boosting task accuracy.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Why latent memory evolution works. Our ablations pinpoint two necessary conditions that general latent methods lack. First, structured priors are indispensable: replacing MedSAM3 with a random encoder collapses Avg from $6 7 . 7 \%$ to $5 2 . 0 \%$ (Table 2d). Second, causal calibration activates the priors: rcausal lifts accuracy by 4.1 pp (Table 2b) and reorganizes memory into the hierarchical diagnostic manifold shown in Fig. 8. Neither condition alone suffices, and their synergy is precisely what general latent methods lack.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

# 6 Training Dynamics Analysis

Figure 10 extends the Stage II reward summary in Fig. 7 of the main text with additional monitoring metrics across all three stages.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

Stage I (panel d): the NTP loss of $\mathcal { P } _ { \phi }$ drops sharply from ${ \sim } 2 . 6$ to ${ \sim } 0 . 4 5$ within the first epoch and converges smoothly across three epochs, with minor jumps at epoch boundaries due to learning-rate scheduling. This confirms that the semantic alignment warmup provides a well-conditioned initialization for subsequent RL. Stage II (panels a–c, e): the full model reward (panel a) climbs to ${ \sim } 0 . 8 8$ with a transient exploration dip near step 150, while the ablation (w/o rcausal) stalls at ${ \sim } 0 . 4 8$ . Panel (b) shows $r _ { c a u s a l }$ rising from near-zero to ${ \sim } 0 . 3 5$ , confirming progressive memory utilization. Panel (c) reveals that $r _ { c a u s a l }$ stabilizes gradient norms within [0.2, 0.6], whereas removing it produces frequent spikes exceeding the clip threshold, indicating an unstable optimization surface. Panel (e) shows monotonically decreasing policy loss with well-controlled KL divergence (<0.02). Stage III (panel f): the JSD between teacher and student branches drops from ${ \sim } 0 . 4 2$ to ${ \sim } 0 . 0 3 5$ , while output agreement rises from $7 2 \%$ to ${ \sim } 9 7 \%$ , consistent with the near-lossless encoder removal ( $\varDelta$ =1.4 pp) reported in Table 2c.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

# 7 Benchmark Dataset Statistics

We evaluate our model across below medical benchmarks, using official test splits where available. Our evaluation suite covers both closed-ended (CE) and multichoice (MC) formats: (1) CE tasks include VQA-RAD [20] (451 radiology questions), SLAKE [29] (1,061 mixed-modality samples), and PathVQA [16] (6,719 pathology samples); (2) MC tasks comprise PMC-VQA [64] (10,000 samples), the Health & Medicine track of MMMU [60] (150 samples), and the expertlevel MedXpertQA-MM [69] (960 samples). Additionally, we evaluate on GMAI-MMBench [57], a multi-granularity benchmark spanning 38 distinct modalities with 2,847 questions. Accuracy is the primary metric across all benchmarks, except for MedXpertQA-MM which uses a Total Score.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

Table 4: Per-modality accuracy ( $\%$ ) on OmniMedVQA. $\varDelta$ : improvement over Qwen3- VL-8B zero-shot baseline.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

![Table 6](../images/0b7872378a2d62950746d74926b8c505ed5b9150f926284dc7e2532348139d92.jpg)
*Table 6: Table 6: Memory synthesis design ablations. Left: meta-query sampling vs. simpler injection (full MQPM CCR IMT pipeline, encoder-free inference). Right: effect of including question tokens $\mathbf { q }$ in $\mathcal { A } _ { \psi }$ input. Default configurations are highlighted.*

> 💡 **Table 6 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

The training pipeline follows three progressive phases to enhance the medical grounding of the model. Stage I involves large scale pre training with 50K image text pairs from PubMedVision [6]. These samples were rigorously curated by medical experts to ensure accurate alignment across radiology modalities like CT, MRI, and X ray along with pathology. Stage II constructs a specialized mixed modality reinforcement learning set of 4K samples. These instances were also selected by clinical professionals to prioritize high diagnostic value, including 3K closed ended VQA samples from expert annotated OmniMedVQA [18] and 1K open ended samples from SLAKE and PathVQA. Stage III refines the model by reusing the Stage II data with identical preprocessing to ensure consistent optimization. Importantly, rigorous filtering was applied across all dataset splits to ensure that no evaluation test samples overlap with any training data, maintaining the absolute integrity of our zero shot assessment.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

# 8 Additional Analysis

# 8.1 Per-Modality Breakdown on OmniMedVQA

Table 4 summarizes performance across the eight imaging modalities in OmniMedVQA. MedSynapse-V achieves consistent performance gains across all eight OmniMedVQA modalities, with the largest gains on radiology-centric modalities (CT: $+ 1 4 . 4$ , MRI: $+ 1 4 . 9$ , X-ray: +13.1) where structured anatomical priors are most informative. Notably, although MMedExpert-R1 improves over the Qwen3-VL-8B zero-shot baseline across all modalities through guideline-based RL rewards, the margin remains modest on challenging modalities such as OCT $\left( + 5 . 6 \right)$ and Fundus (+5.6), where explicit CoT reasoning struggles to capture subtle spatial patterns. In contrast, MedSynapse-V’s latent memory mechanism yields substantially larger gains on these same modalities ( $+ 1 1 . 9$ and +11.8), confirming that continuous diagnostic memory encodes fine-grained anatomical features more effectively than discrete token reasoning.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Figure 11](../images/5e72027899d9413025f89830e9e59a6a77541af5496e8739e0c048e035db9d65.jpg)
*Figure 11: Fig. 11: Causal intervention visualization on fundus (left group) and dermoscopy (right group). Each group: original image, MedSAM3 region mask $\mathbf { B }$ , and post-CCR memory attention map. After refinement, memory attention concentrates on diagnostically critical structures while suppressing background.*

> 💡 **Figure 11 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

∼2.6 s. CoT baselines require 5.2–5.8 s due to 300–400 autoregressive reasoning

# 8.3 Inference Latency and Parameter Count

The prefill latency of MedSynapse-V (IMT) is 102 ms, identical to the vanilla baseline; $\mathcal { A } _ { \psi }$ contributes only 4 ms. Although prefill overhead is negligible, the 16 memory vectors injected during this stage play a pivotal role in overall efficiency: they become part of the KV cache built at prefill time, so every subsequent decoding step can attend to condensed diagnostic priors at no extra cost. This latent conditioning steers the model toward shorter, more decisive outputs ( $\sim$ 34–44 answer tokens vs. ${ \sim } 5 0 { - } 8 0$ for the zero-shot baseline), reducing endto-end sample latency from ${ \sim } 2 . 8 \mathrm { s }$ to

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Table 5: Inference latency (single A100, batch $= 1$ ) and parameter count. Prefill: time to first token. End-to-end : full autoregressive decoding (max 128 tokens, greedy).

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

![Table extracted](../images/5ae667aa0106c047344170d7c22b810805f7045852b0da524f62f66475d80f47.jpg)
*Table extracted: Table extracted by MinerU. Aψφ Input VQA- RAD SLAKE Avg. Visual only 72.8 78.1 67.0 Visual + q (default) 74.2 79.8 67.7*

> 💡 **Table extracted 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

![Figure 12](../images/ba27b9e72b9b4c0708eacbaffa963f6c1f6d99eb94c6714c300255d6f13ac0ad.jpg)
*Figure 12: Fig. 12: Memory evolution across training stages. Visualization of diagnostic memory vectors colored by modality. Contours denote kernel density estimates per modality to highlight the clustering density across different clinical domains.*

> 💡 **Figure 12 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

tokens. The inference footprint is 8.41B parameters ( $1 . 4 \%$ over the bare backbone), as the $\mathcal { E } _ { a n a }$ is entirely removed.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

Note on the “ms/token” column in Table 2 of the main text. The ms/token values in the main paper’s ablation table measure average per-token decoding latency (total decode wall time / number of generated tokens). The zeroshot Qwen3-VL-8B baseline shows a higher value (126 ms/token) than MedSynapse-V ( $\sim$ 102 ms/token) because, without compact memory conditioning in the KV cache, the model’s attention must scatter across the full visual token sequence at each decode step, yielding broader attention patterns and slower per-step computation. These per-token values should not be confused with the prefill latency (102 ms, Table 5) or the end-to-end sample latency (∼2.6 s, Fig. 5), which cover the complete inference pipeline.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

# 8.6 Robustness of Causal Intervention to Mask Quality

The causal counterfactual reward $r _ { c a u s a l }$ relies on region masks $\mathbf { B }$ from Med-SAM3. We verify robustness via two ablations: (i) varying the binarization threshold $\tau$ , and (ii) replacing the top-1 mask with lower-ranked candidates.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Table 8: Effect of mask rank selection. Rank-1: highest confidence (default); Rank-2: second-highest; Random: uniformly sampled candidate.

> 💡 **批注**: 这是因果/反事实修正：关注它如何剔除伪相关、突出关键病灶或视觉证据。

![Table 7](../images/6046e77e96a0937d1ce8e881e39d5e418f5b9dc52fcf85eafe4ef9fd89eb81f7.jpg)
*Table 7: Table 7: Effect of mask confidence threshold $\tau$ . Full MQPM CCR IMT pipeline; encoder-free inference. $| \mathbf { B } | / H W$ : average masked fraction.*

> 💡 **Table 7 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

(i) Confidence threshold $\tau$ . Table 7 reports accuracy under five thresholds (default $\tau { = } 0 . 7$ ). All thresholds outperform the no-mask baseline (Avg $6 3 . 6 \%$ ). Performance is stable across $\tau \in [ 0 . 5 , 0 . 8 ]$ (spread only $1 . 1 \mathrm { p p }$ ), confirming that $r _ { c a u s a l }$ does not require pixel-perfect boundaries. Extreme values degrade: $\tau { = } 0 . 3$ masks $4 2 . 6 \%$ of the image (intervention too destructive); $\tau { = } 0 . 9$ masks only $5 . 4 \%$ (intervention too weak). (ii) Mask rank selection. We replace MedSAM3’s top-1 mask with its second-highest confidence candidate (Rank-2) or a random candidate. Rank-2 retains most of the gain ( $- 1 . 2 \mathrm { p p }$ vs. Rank-1), and even random masks outperform the no-mask baseline by $1 . 6 \mathrm { p p }$ . The monotonic ordering Rank- $1 >$ Rank- $2 >$ Random $>$ None confirms that mask quality helps but is not critical: the causal reward exploits the relative contrast between masked and unmasked conditions rather than relying on pixel-precise delineation.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

# 9 Additional Qualitative Results

# 9.1 Additional Representative Cases

Figure 13 illustrates comparative evaluations between MedSynapse-V and two competitive RL-CoT baselines across diverse modalities. While Med-R1 and MMedExpert-R1 generate extensive reasoning chains, they frequently yield erroneous diagnoses as a result of hallucinatory observations that propagate and amplify throughout the inference process. In the chest X-ray case, Med-R1 fabricates bilateral interstitial opacities and claims sharp costophrenic angles, missing the obvious pleural effusion; MMedExpert-R1 hallucinates a convex border with cavitation and misdiagnoses a lung abscess. In the pathology case, Med-R1 incorrectly describes preserved polarity and intact basement membranes to conclude fibroadenoma, while MMedExpert-R1 fabricates lymphovascular invasion and comedonecrosis to misclassify as invasive lobular carcinoma. In the head CT case, Med-R1 denies the presence of a hyperdense lesion and diagnoses ischemic infarct, whereas MMedExpert-R1 hallucinates ring enhancement with central necrosis and concludes cerebral abscess. In contrast, MedSynapse-V directly identifies the correct findings in 38–43 tokens without explicit CoT, demonstrating that latent diagnostic memory provides sufficient guidance while avoiding hallucination cascades.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Figure 13](../images/7c50709fbdae932425ff387e0cdeb717ce0bf6ea93dde73ecfab7f85d27c5974.jpg)
*Figure 13: Fig. 13: Qualitative comparison across Chest X-ray, Pathology, and Head CT cases. MedSynapse-V produces concise, correct diagnoses $\sim 3 8 – 4 3$ tokens), while other methods generate verbose CoT (∼195–215 tokens) with hallucinated findings (red).*

> 💡 **Figure 13 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

# 9.2 Failure Case Analysis

Figure 14 illustrates three primary failure modes. (a) Rare modality under-representation: OCT, constituting the smallest training proportion $( \sim 2 5 \% )$ , exhibits the lowest per-modality accuracy, indicating that memory quality degrades when prior exposure is insufficient. (b) Multi-lesion ambiguity: accuracy drops from $7 8 \%$ on single-lesion images to $5 2 \%$ on multi-lesion cases, as the

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Figure 14](../images/32cd467f0ab0c323eced98b24879e572f00f30e58f06776007b7bbab1176e089.jpg)
*Figure 14: Fig. 14: Three representative challenging modes.*

> 💡 **Figure 14 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

fixed $N { = } 1 6$ memory pool becomes saturated when multiple co-occurring pathologies compete for representational capacity. (c) Subtle feature discrimination: each scatter point represents one evaluation sample from the dermoscopy subset, where the $x$ -axis is the model’s confidence defined as the mean token-level generation probability $\begin{array} { r } { \mathrm { c o n f } ( \mathbf { o } ) = \frac { 1 } { \left| \mathbf { o } \right| } \sum _ { t } \pi _ { \theta } ( \mathbf { o } _ { t } \ \mid \ X , q , \mathcal { M } , \mathbf { o } _ { < t } ) } \end{array}$ , and the $y$ -axis is binary diagnostic correctness (1=correct, 0=incorrect; vertical jitter applied for visibility). While high-confidence predictions are predominantly correct, a notable cluster at conf $< 0 . 3$ with correctness $= 0$ reveals that borderline cases (e.g., benign vs. dysplastic nevi) fall below the memory’s discriminative granularity. These modes point to future directions including balanced modality sampling, adaptive memory pool sizing, and calibrated uncertainty estimation.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Figure 15](../images/f1b9ce5d1db44ac432ac2df68105f310b2352ce9a01a9a9c7f94b0b6bedd559e.jpg)
*Figure 15: Fig. 15: Prompt template for closed-ended multi-choice VQA (VQA-RAD, SLAKE, PathVQA, PMC-VQA, MMMU\*, MedXpertQA-MM, GMAI-MMBench). The number of options varies by dataset (2–5); the template adapts accordingly.*

> 💡 **Figure 15 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 16](../images/304cf67f28390cebd3daaa0a382d3acbe83ee679fe56b26b99b72099de349018.jpg)
*Figure 16: Fig. 16: Prompt template. Notably, $\mathcal { M } _ { a u t o }$ is autonomously generated and injected in the hidden stream without altering the text prompt.*

> 💡 **Figure 16 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

# 11 Extended Ablation Studies

Table 9 and Table 10 present four complementary design analyses that further validate the key design choices of MedSynapse-V. All results use the full threestage pipeline with IMT inference; default configurations are highlighted.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

(i) Memory injection position. Table 9 (left) examines the effect of injecting diagnostic memory $\mathcal { M }$ at different positions in the input sequence, where $\mathbf { V }$ denotes visual tokens and q denotes question tokens. Placing $\mathcal { M }$ after $\mathbf { q }$ and before answer generation (our default) yields the best average of ${ \bf 6 9 . 3 \% }$ , as answer tokens can attend to both visual features and diagnostic memory simultaneously. Injection before $\mathbf { V }$ degrades performance to $6 5 . 5 \%$ because self-attention cannot condition memory on the question context; interleaving with q $( 6 8 . 2 \% )$ partially recovers but still disrupts the natural query encoding flow.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Table 10: Left: comparison of divergence measures for IMT distillation ( $\beta$ controls JSD interpolation weight). Right: sensitivity analysis of causal reward weight $\lambda _ { c a u s a l }$ . Default configurations are highlighted.

> 💡 **批注**: 这是因果/反事实修正：关注它如何剔除伪相关、突出关键病灶或视觉证据。

![Table 8](../images/5956ccb924a62cdb55f19a67600d9248d3f5b13e772590187fe702bcaa8627e0.jpg)
*Table 8: Table 8: Effect of mask rank selection. Rank-1: highest confidence (default); Rank-2: second-highest; Random: uniformly sampled candidate.*

> 💡 **Table 8 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

![Table 9](../images/4c77f29e1c8fee1ac68d3023bbab852cbb7405f62d975ac18e84393165b31964.jpg)
*Table 9: Table 9: Left: effect of memory injection position on diagnostic accuracy ( $\%$ ), where $\mathbf { V }$ and q denote visual and question tokens. Right: effect of GRPO group size $G$ on accuracy ( $\%$ ) and training cost. All results use the full three-stage pipeline with IMT inference. Default configurations are highlighted.*

> 💡 **Table 9 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

(ii) GRPO group size $G$ . Table 9 (right) shows that $G { = } 4$ achieves the optimal accuracy–cost balance: smaller groups ( $G { = } 2$ ) yield noisy advantage estimates $( 6 7 . 3 \% )$ , while $G { = } 6$ and $G { = } 8$ provide only marginal gains $( + 0 . 1 -$ $0 . 3 \mathrm { p p }$ ) at $1 . 4 \mathrm { - } 2 \times$ additional GPU hours. The diminishing returns beyond $G { = } 4$ confirm that four trajectories suffice for stable advantage estimation under our composite reward.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

(iii) IMT divergence function. Table 10 (left) compares divergence measures for the IMT distillation objective. Jensen–Shannon divergence with $\beta$ =0.5 outperforms both forward KL $( 6 7 . 2 \% )$ and reverse KL (66.8%). Forward KL causes mode-covering behavior that dilutes diagnostic specificity; reverse KL leads to mode-seeking collapse. The symmetric JSD provides a balanced learning signal, and performance remains stable across $\beta \in [ 0 . 3 , 0 .$ 7].

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

(iv) Causal reward weight $\lambda _ { c a u s a l }$ . Table 10 (right) reveals that performance is robust within $\lambda _ { c a u s a l } \in \mathsf { \Omega } [ 0 . 3 , 0 . 7 ]$ , peaking at 0.5. Setting $\lambda _ { c a u s a l } { = } 0$ causes the model to bypass memory via direct shortcuts (65.4%); excessively high values $\geq 1 . 0 )$ over-penalize trajectories and destabilize training $( 6 7 . 7 \% )$ .

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

---

## 🔖 Section 总结

### 核心洞察
1. 检查 benchmark 是否覆盖医学/跨域/鲁棒性。
2. 关注消融、效率和失败案例。
