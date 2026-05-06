[← 返回 README](../README.md)

# 3 Experiments

## 📌 预览
本节验证三件事：latent memory 是否提升诊断准确率，CCR/IMT 等模块是否真的不可替代，以及相较显式 CoT 是否能同时减少幻觉链和推理开销。

---

# 3.1 Experimental Setup

Datasets Training data: We conduct comprehensive experiments on seven medical multimodal benchmarks spanning diverse task types and difficulty levels. Stage I (MQPM warmup) uses 50K image–text pairs from PubMedVision [6] covering radiology and pathology. Stage II (CCR) constructs a mixed-modality RL set: 3K closed-ended VQA samples from OmniMedVQA [18] training split (8 modalities: CT, MRI, X-ray, dermoscopy, fundus, OCT, pathology, ultrasound) plus 1K open-ended samples from SLAKE [29] and PathVQA [16] training sets, totaling ∼4K samples. Region masks for $r _ { c a u s a l }$ are provided by Med-SAM3 [28]. Stage III (IMT) reuses the Stage II data. Evaluation benchmarks: (i) Closed-ended VQA: VQA-RAD [20], SLAKE [29], PathVQA [16], PMC-VQA [64]; (ii) Clinical reasoning: MMMU Health & Medicine [60] (denoted MMMU $^ *$ ); (iii) Expert-level reasoning: MedXpertQA-MM [69] (Total score); (iv) Multi-granularity: GMAI-MMBench [57].

> 💡 **数据设置批读**: 训练数据被拆成三个阶段：50K PubMedVision 用于 MQPM 语义 warmup；约 4K mixed-modality 样本用于 CCR RL 和 IMT。评估覆盖 VQA-RAD、SLAKE、PathVQA、PMC-VQA、MMMU Health & Medicine、MedXpertQA-MM、GMAI-MMBench，既有常规 VQA，也有专家级和多模态泛化。

Baselines We compare against four categories of methods: (1) General VLMs: Qwen3-VL-8B [2] (our base model), InternVL3-8B [67]; (2) Medical-specific VLMs: RadFM [51], LLaVA-Med [24], GMAI-VL [26], HuatuoGPT-Vision [6], BiMediX2- 8B [33], MedMO-8B [9]; (3) RL-enhanced medical reasoning: MedVLM-R1- 2B [37], Med-R1-3B [19], MediX-R1-8B [32], MMedExpert-R1-7B [11]; (4) Latentspace reasoning: Coconut $^ \dagger$ [15], MCOUT-Multi $\dagger$ [38], IVT-LR $^ { \dag }$ [5] ( $^ \dagger$ : adapted with identical Qwen3-VL-8B backbone and training data). We additionally report MedSynapse-V-4B on the Qwen3-VL-4B backbone to assess scalability.

> 💡 **Baseline 设计**: baseline 分四类，能回答不同问题：general VLM 测 base 能力；medical VLM 测领域预训练；RL-CoT 测显式推理增强；latent-space reasoning 测“只做 latent reasoning 不加解剖先验”是否足够。最后一类对本文 claim 最关键。

Implementation Details Our framework builds upon Qwen3-VL-8B-Instruct [2]. The frozen anatomical encoder $\mathcal { E } _ { a n a }$ employs MedSAM3 [28] pre-trained on largescale multi-organ segmentation datasets. Stage I freezes both VLM and $\mathcal { E } _ { a n a }$ , training only the Diagnostic Memory Sampler $\mathcal { P } _ { \phi }$ with $\mathrm { l r } = 2 \times 1 0 ^ { - 4 }$ for 3 epochs. The diagnostic probe count is $N { = } 1 6$ ; $\mathcal { P } _ { \phi }$ is a 2-layer cross-attention Transformer with output dimension $d _ { m } { = } 4 0 9 6$ (matching Qwen3-VL-8B). Images are processed at native dynamic resolution following Qwen3-VL’s default configuration. Stage $\mathrm { I I }$ freezes $\mathcal { P } _ { \phi }$ and adapts VLM via LoRA [17] (rank=64, applied to all attention layers). GRPO generates $G { = } 4$ candidate trajectories per sample, with clipping coefficient $\varepsilon { = } 0 . 2$ , reward weights $\lambda _ { a c c }$ =1.0 and $\lambda _ { c a u s a l } { = } 0 . 5$ , training for 200 steps with a rollout batch size of 32. Max generation length is 1024 tokens. Stage III introduces the Autonomous Memory Module $\mathcal { A } _ { \psi }$ (2-layer MLP + LayerNorm, input from VLM’s visual encoder features), with JSD coefficient $\beta$ =0.5, $\mathrm { l r } = 1 \times 1 0 ^ { - 4 }$ , 3 epochs. For each sample we draw one on-policy trajectory $\hat { y } \sim \pi ^ { - }$ per gradient step; the Stage II data is reused with identical preprocessing. Closed-ended VQA tasks report overall accuracy ( $\%$ ). For GMAI-MMBench and MedXpertQA-MM, we follow their respective official evaluation protocols. Inference efficiency is measured quantitatively as ms/sample and peak GPU memory (GB). More details are provided in the supplementary material.

> 💡 **实现细节批读**: 关键超参集中在这里：$N=16$ probes，$\mathcal P_\phi$ 是 2-layer cross-attention Transformer；Stage II 用 LoRA rank 64 和 GRPO $G=4$；reward 权重为 accuracy 1.0、causal 0.5；Stage III 用 $\beta=0.5$ 的 JSD。复现时这些决定 memory 容量、RL 稳定性和 IMT 损失。

Table 1: Comprehensive comparison on seven medical benchmarks. Base model: Qwen3-VL-8B (unless otherwise noted). MMMU\* = Health & Medicine track. †: general latent-space methods adapted to medical VQA with identical backbone and training data. $\varDelta$ : absolute gap (pp) to MedSynapse-V (w/ ${ \dot { \varepsilon } } _ { a n a }$ ) average. All results are averaged over five independent runs. Bold: best; underline: second best.

> 💡 **Table 1 说明**: Table 1 是主结果表，重点看平均分、IMT 去编码器后的分数、与最强 RL-CoT 的差距，以及不同 benchmark 上是否一致提升。这里还标注了五次独立运行均值，降低单次采样偶然性。

![Table 1](../images/6262963dba1cd3c9eec425a48578107b11636a235ac0f868dfa076c7e9195f46.jpg)
*Table 1: Table 1: Comprehensive comparison on seven medical benchmarks. Base model: Qwen3-VL-8B (unless otherwise noted). MMMU\* = Health & Medicine track. †: general latent-space methods adapted to medical VQA with identical backbone and training data. $\varDelta$ : absolute gap (pp) to MedSynapse-V (w/ ${ \dot { \varepsilon } } _ { a n a }$ ) average. All results are averaged over five independent runs. Bold: best; underline: second best.*

> 💡 **Table 1 批读**: 带 $\mathcal E_{ana}$ 的 MedSynapse-V 平均 61.4%，IMT 去掉外部 encoder 后 59.6%，仍高于 MMedExpert-R1 的 55.7%。最大增益出现在 VQA-RAD、SLAKE、PathVQA 等视觉 grounding 更强的任务，支持“连续诊断 memory 比冗长 CoT 更保留视觉证据”的 claim。

# 3.2 Main Results

As shown in Table 1, MedSynapse-V (w/ $\mathcal { E } _ { a n a }$ ) achieves the highest average of ${ \bf 6 1 . 4 \% }$ , and the encoder-free MedSynapse-V (IMT) retains ${ \bf 5 9 . 6 \% }$ , surpassing all baselines. Compared to the strongest RL baseline MMedExpert-R1 $( 5 5 . 7 \%$ ), MedSynapse-V (IMT) leads by ${ \bf + 3 . 9 }$ pp without any auxiliary module at inference, with the largest margins on visual-grounding benchmarks (VQA-RAD ${ \bf + 9 . 0 }$ , SLAKE +7.0, PathVQA +6.7), where discrete CoT tokens are prone to attenuating early visual evidence across long reasoning chains. On GMAI-MMBench spanning 38 modalities, MedSynapse-V scores $5 4 . 8 \%$ , confirming that the anatomical priors generalize beyond the training distribution.

> 💡 **主结果读数**: 两个数字最关键：61.4% 是训练/推理都保留解剖 encoder 的 privileged 上限，59.6% 是部署形态 IMT。和 MMedExpert-R1 相比 +3.9 pp，说明优势不是靠推理时多跑专家 encoder 得来的。

RL baselines reveal a specialization dilemma. MediX-R1 benefits from multilingual pretraining and leads on PMC-VQA (56.2%), yet this breadth dilutes radiology-specific precision (VQA-RAD: 56.4%); MMedExpert-R1 achieves the most balanced profile by leveraging guideline-based reward. Small-scale models (MedVLM-R1 2B, Med-R1 3B) collapse on out-of-domain tasks (MedXpert below $1 9 \%$ ), confirming that parameter capacity sets a hard ceiling RL alone cannot raise. n contrast, MedSynapse-V sidesteps this dilemma by injecting latent priors that benefit all task types uniformly, achieving the top performance on every benchmark without task-specific tuning.

> 💡 **RL baseline 对比**: RL-CoT 方法在某些数据集上有专长，但容易出现“任务专门化”或小模型容量上限。MedSynapse-V 的优势在于 latent priors 是跨任务注入的，不需要每个 benchmark 单独设计推理链或奖励模板。

Latent methods require domain priors. Among adapted latent baselines, the hierarchy Coconut (44.5%) $<$ < MCOUT-Multi (47.9%) $<$ < IVT-LR $( 5 0 . 5 \% )$ ) tracks optimization sophistication, yet even IVT-LR barely exceeds zero-shot Qwen3-VL-8B (48.6%). This inversion reveals that latent compression without clinical grounding encodes statistical shortcuts rather than diagnostic logic; the

> 💡 **latent baseline 对比**: Coconut/MCOUT/IVT-LR 的层级说明普通 latent computation 确实有优化收益，但缺少 anatomical grounding 时很难超过医学任务需求。IVT-LR 只到 50.5%，和 MedSynapse-V 差 10.9 pp，直接支撑“domain prior + causal calibration 才是关键”。

![Figure 4](../images/6143166381cb21b1252cda3a97706934b40f744fdfde3a9aac24b4110202f03e.jpg)
*Figure 4: Fig. 4: Effect of diagnostic probe count $N$ . Performance peaks around $N { = } 1 6$ across benchmarks; further increasing $N$ dilutes diagnostically relevant signals.*

> 💡 **Figure 4 批读**: probe count $N$ 的曲线显示 16 附近最好：太少表达力不足，太多会稀释诊断信号并放大绕过 memory 的 shortcut。它也解释了为什么后面失败案例会提到多病灶时固定 $N=16$ 可能容量不够。

Table 2: Ablation study on the 8B backbone. All variants undergo the full threestage pipeline including IMT (§2.4); results reflect inference without the anatomical encoder unless stated otherwise. MQPM: Meta Query for Prior Memorization (§2.2); CCR: Causal Counterfactual Refinement (§2.3); $\mathcal { M }$ : diagnostic implicit memory. Best per group in bold.

> 💡 **Table 2 说明**: Table 2 是模块消融，不只看某个模块有没有用，还要看三阶段之间的依赖关系：没有 MQPM，RL 起点不稳；没有 CCR，memory 容易被绕过；没有 IMT，推理成本高；encoder 质量差，memory 没有医学先验。

![Table 2](../images/8ff7ded8edad82f133252449c956da5811b7893efb6fa44c870b0e833c316c77.jpg)
*Table 2: Table 2: Ablation study on the 8B backbone. All variants undergo the full threestage pipeline including IMT (§2.4); results reflect inference without the anatomical encoder unless stated otherwise. MQPM: Meta Query for Prior Memorization (§2.2); CCR: Causal Counterfactual Refinement (§2.3); $\mathcal { M }$ : diagnostic implicit memory. Best per group in bold.*

> 💡 **Table 2 批读**: 全流程的平均分 67.7 是消融设置下的最好值；跳过 MQPM 掉到 52.9，CCR 换 SFT 只有 59.2，去 causal reward 从 67.7 掉到 63.6。IMT 去 encoder 只损失约 1.4 pp，但 latency 降 39%、memory 降 6.3GB，是部署价值的核心证据。

10.9 pp gap to MedSynapse-V confirms that prior injection and causal calibration are prerequisites for effective latent reasoning in medicine.

> 💡 **10.9 pp 的含义**: 这句收束 latent baseline 对比：不是“latent space 天然有效”，而是“带医学先验且经过因果校准的 latent memory 有效”。这对同 topic 下其他 VisMem/latent memory 论文也很有参考价值。

Scaling efficiency. MedSynapse-V-4B (w/ ${ \mathscr E } _ { a n a }$ ) reaches $5 4 . 7 \%$ with roughly half the parameters of 7B baselines; after encoder removal the IMT variant still achieves $5 2 . 7 \%$ at $8 5 \mathrm { m s }$ and 10.8 GB, surpassing MediX-R1-8B $( 4 9 . 9 \% )$ ). This efficiency stems from a structural advantage: diagnostic expertise is distilled into 16 compact memory vectors consumed in a single forward pass, rather than spread across $1 5 0 +$ verbose reasoning tokens.

> 💡 **Scaling 批读**: 4B 版本显示方法不是 8B 特例。IMT 4B 仍有 52.7%，延迟 85ms、显存 10.8GB，并超过 MediX-R1-8B 的 49.9%。这说明 compact memory 可以部分弥补参数规模差距。

# 3.3 Ablation Study

Table 2 reports comprehensive ablation study results. Specifically, (i) Progressive training stages. MQPM warmup is indispensable: skipping it collapses Avg to 52.9, barely above zero-shot (54.2), because randomly initialized memory destabilizes early RL training. Replacing CCR with SFT reaches 59.2 but lags by $8 . 5 \ \mathrm { p p }$ due to limited out-of-distribution generalization. The full pipeline (Avg 67.7) confirms non-redundant contributions: MQPM grounds semantics, CCR refines via exploration, IMT compresses into an autonomous pathway. (ii) Reward design. rcausal is the dominant reward component ( $^ +$ 4.1 pp, $6 3 . 6 \to 6 7 . 7$ ). Without causal pressure the model bypasses $\mathcal { M }$ via direct shortcuts, treating memory as inert padding; the counterfactual intervention penalizes trajectories insensitive to diagnostic regions. The effect concentrates on radiology benchmarks and persists after IMT, indicating stronger memory utilization transfers more faithfully through distillation. (iii) Encoder retention vs. removal. IMT achieves near-lossless removal: only $1 . 4 ~ \mathrm { p p }$ degradation $6 9 . 1  6 7 . 7 $ ) while latency drops 39% and memory decreases 6.3 GB. The gap is not uniform: core VQA metrics degrade minimally, whereas MedXpert and GMAI suffer more, suggesting complex reasoning depends more on encoder-derived priors than closed-ended recognition. (iv) Anatomical encoder choice. MedSAM3 outperforms SAM-Med2D by 4.4 pp (67.7 vs. 63.3), reflecting richer spatial representations from multi-organ segmentation pretraining. Random initialization yields only 52.0, confirming that gains originate from what the encoder knows, rather than how memory is aggregated. (v) Probe count $N$ . As shown in Fig. 4, $N { = } 1 6$ balances expressiveness against redundancy. The CCR to SFT gap widens with $N$ (3.5 pp at $N { = } 4$ vs. $7 . 2 \mathrm { \ p p }$ at $N { = } 1 6$ ), revealing that larger memory pools amplify bypass shortcuts and therefore benefit disproportionately from causal refinement.

> 💡 **消融结论**: 五组消融共同形成因果链：MQPM 给可读 memory，CCR 给 causally grounded memory，IMT 给 deployable memory，MedSAM3 给高质量 spatial prior，$N=16$ 给容量/冗余平衡。单看最终平均分会漏掉这些机制分工。

![Figure 5](../images/953e6d177886aadcd5d391a32a867c5e8bb39982f95e976011318be129cc4637.jpg)
*Figure 5: Fig. 5: Qualitative comparison across CT, MRI, and Ultrasound cases. MedSynapse-V produces concise, correct diagnoses, while Med-R1 and MMedExpert-R1 generate verbose CoT with hallucinated findings (red) leading to misdiagnoses.*

> 💡 **Figure 5 批读**: 图 5 是定性证据：CT/MRI/Ultrasound 中，RL-CoT baseline 生成更长解释但引入不存在的影像征象，导致错误诊断；MedSynapse-V 输出短而准。它支撑的是“少说但 latent memory 对齐视觉证据”而不是“解释更漂亮”。

# 3.4 In-Depth Case Analysis

As illustrated in Fig. 5, we compare MedSynapse-V with Med-R1 [19] and MMedExpert-R1 [11] across three distinct imaging modalities. Both baselines produce verbose CoT reasoning ( $\sim$ 185–238 tokens) yet arrive at incorrect diagnoses due to hallucinated observations erroneously propagating through the chain. In the CT case, Med-R1 fabricates pleural thickening in the left upper lobe, while MMedExpert-R1 hallucinates a laminated calcification pattern and mischaracterizes the nodule as a benign granuloma. In the MRI case, Med-R1 misidentifies the extra-axial mass as intra-axial and concludes glioblastoma, whereas MMedExpert-R1 fabricates ring enhancement with central necrosis, both missing the classic meningioma presentation. In the ultrasound case, Med-R1 hallucinates gallbladder wall thickening to over-diagnose acute cholecystitis, while MMedExpert-R1 denies posterior acoustic shadowing and misdiagnoses a gallbladder polyp. In contrast, MedSynapse-V generates concise, correct answers ( $\sim$ 34–44 tokens) without explicit CoT, demonstrating that diagnostic implicit memory provides sufficient latent guidance while avoiding the hallucination cascades inherent in token-level CoT.

> 💡 **病例分析批读**: 这些 case 的共同模式是 hallucinated observation 先出现，再沿 CoT 传播成错误结论。MedSynapse-V 的 34-44 token 回答减少了可见推理链，但隐式 memory 提供足够诊断锚点，因此避免了“越推越错”的 cascade。

# 3.5 Efficiency, RL Dynamics, and Latent Space

Performance–efficiency trade-off. As shown in Fig. 6, MedSynapse-V (IMT) achieves $5 9 . 6 \%$ at 2.6 s/sample, comparable to zero-shot Qwen3-VL-8B $4 8 . 6 \%$ , 2.8 s) since both share the same backbone and the 16 memory vectors add negligible overhead. Full-scale 7–8B CoT methods (MediX-R1, MMedExpert-R1) require 5.8 s each due to 300–400 autoregressive reasoning tokens, while smaller CoT models (MedVLM-R1 2B, Med-R1 3B) offset verbosity with faster per-token speed yet remain 18–21 pp below MedSynapse-V. This confirms that compact latent memory provides diagnostic grounding without the token-generation overhead of full-scale CoT.

> 💡 **效率批读**: Fig. 6 的重点是 Pareto 位置：IMT 59.6%、2.6s/sample，和 zero-shot Qwen3-VL 延迟接近，但准确率高 11 pp；CoT 方法 5.8s 左右，因为要生成 300-400 个推理 token。latent memory 把诊断先验放进 KV cache，比生成文本链便宜。

Training dynamics. Fig. 7 shows the full model $\textit { \textbf { ( w / } \textit { r } _ { c a u s a l } ) }$ improving steadily to ${ \sim } 0 . 8 8$ with a transient exploration dip near step 900, where the policy sacrifices reward to explore memory-reliant generation strategies, while the $w / o \ r _ { c a u s a l }$ ablation plateaus at ${ \sim } 0 . 4 8$ throughout the training. This confirms that accuracy-only reward cannot distinguish memory-dependent from shortcut trajectories; without causal pressure the model bypasses $\mathcal { M }$ entirely, treating injected memory as inert padding.

> 💡 **训练动态批读**: Fig. 7 展示 with causal reward 的曲线能涨到约 0.88，而无 causal reward 卡在约 0.48。短暂 exploration dip 不是失败，而是策略探索 memory-dependent generation；没有 $r_{causal}$ 时模型直接绕过 memory。

![Figure 6](../images/f0ae86ba491fcc2f228e071dbbd5703f42d7fca1bbd2094bd7e14578efe414d6.jpg)
*Figure 6: Fig. 6: Accuracy–latency trade-off across compared VLM categories.*

> 💡 **Figure 6 批读**: 这张 accuracy-latency 图用来回答部署问题。MedSynapse-V 不在最高延迟区，而在接近 vanilla VLM 的位置拿到明显更高准确率；这是 IMT 的直接收益。

![Figure 7](../images/be97594aa2d2e0ada142aae3c720a5fde92bde89e8b8752d67d7fe11321b0a71.jpg)
*Figure 7: Fig. 7: The RL training reward dynamics with and without $r _ { c a u s a l }$ .*

> 💡 **Figure 7 批读**: reward dynamics 是 CCR 有效性的过程证据：$r_{causal}$ 不只是最终表格上 +4.1 pp，而是在训练中持续拉开 with/without causal reward 的学习轨迹。

![Figure 8](../images/68f08dc8e3f2f097bb675cbefa34e55f9899509661a3f4321132f92fa0037508.jpg)
*Figure 8: Fig. 8: t-SNE visualization of implicit memory $\mathcal { M } _ { a u t o }$ after CCR. (a) Eight imaging modalities form well-separated clusters with clinically coherent proximity. $( \mathrm { b } , \mathrm { c } )$ Within CT and Pathology, disease subtypes further segregate into distinct regions.*

> 💡 **Figure 8 批读**: t-SNE 图展示 $\mathcal M_{auto}$ 已经按模态和疾病亚型形成层次结构。需要谨慎读：t-SNE 不是因果证明，但结合消融和 case，它说明 CCR/IMT 后的 memory 不只是随机向量，而有可视化的诊断 manifold。

Latent space structure. Fig. 8 visualizes the evolved memory $\mathcal { M } _ { a u t o }$ via t-SNE across three granularities. At the cross-modality level (a), eight imaging types form compact clusters with clinically coherent proximity (e.g., CT and MRI lie adjacent; dermoscopy and fundus form a nearby pair). Within individual modalities $\displaystyle ( \mathrm { b } , \mathrm { c } )$ , disease subtypes further segregate: CT memory separates lung nodules, liver lesions, kidney cysts, pneumonia, and aortic aneurysms, while pathology memory distinguishes adenocarcinoma, squamous cell carcinoma, normal tissue, lymphoma, and melanoma. This hierarchical organization confirms that $r _ { c a u s a l }$ reshapes the latent space into a diagnostically meaningful manifold rather than merely boosting task accuracy.

> 💡 **Latent space 结构**: cross-modality 聚类说明 memory 捕获模态差异；CT/pathology 内部子类分离说明它还捕获诊断类别差异。这个结果补强 Table 2：accuracy 提升背后确实有 representation re-organization。

Why latent memory evolution works. Our ablations pinpoint two necessary conditions that general latent methods lack. First, structured priors are indispensable: replacing MedSAM3 with a random encoder collapses Avg from $6 7 . 7 \%$ to $5 2 . 0 \%$ (Table 2d). Second, causal calibration activates the priors: rcausal lifts accuracy by 4.1 pp (Table 2b) and reorganizes memory into the hierarchical diagnostic manifold shown in Fig. 8. Neither condition alone suffices, and their synergy is precisely what general latent methods lack.

> 💡 **机制总结**: 作者最后把必要条件说清楚：结构化 prior 让 memory 有临床语义，因果校准让 prior 被正确使用。少任何一个，latent memory 都可能退化成普通压缩或任务 shortcut。

---

## 🔖 Section 总结

### 核心洞察
1. 主结果证明 IMT 部署形态仍高于强 RL-CoT baseline，而 privileged encoder 只提供约 1.8 pp 的额外上限。
2. 消融证明三阶段分工清晰：MQPM 提供可读 prior，CCR 激活因果使用，IMT 几乎无损移除外部 encoder。
3. 效率和病例分析共同说明：MedSynapse-V 不是生成更长解释，而是用 compact latent memory 减少 CoT 幻觉链和解码成本。
