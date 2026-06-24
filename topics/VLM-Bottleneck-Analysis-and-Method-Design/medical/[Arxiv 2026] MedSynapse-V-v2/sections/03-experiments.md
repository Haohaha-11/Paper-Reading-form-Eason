[← 返回 README](../README.md)

# 3. Experiments

## 一、Preview

实验部分覆盖了 7 个医学多模态 benchmark、17 个 baseline（4 类）、系统消融研究、案例分析和效率-性能权衡分析。核心发现：MedSynapse-V (IMT) 以近乎标准 VLM 的推理开销，在所有 benchmark 上一致超越所有 RL-CoT baseline；解剖先验 + 因果校准是实现有效医学潜空间推理的两个必要条件。

---

## 二、原始文本

### 3.1 Experimental Setup

**Datasets.** Training data: Stage I (MQPM warmup) uses 50K image-text pairs from PubMedVision [7] covering radiology and pathology. Stage II (CCR) constructs a mixed-modality RL set: 3K closed-ended VQA samples from OmniMedVQA [35] training split (8 modalities: CT, MRI, X-ray, dermoscopy, fundus, OCT, pathology, ultrasound) plus 1K open-ended samples from SLAKE [71] and PathVQA [29] training sets, totaling ~4K samples. Region masks for $r_{causal}$ are provided by Med-SAM3 [70]. Stage III (IMT) reuses the Stage II data. Evaluation benchmarks: (i) Closed-ended VQA: VQA-RAD [48], SLAKE [71], PathVQA [29], PMC-VQA [137]; (ii) Clinical reasoning: MMMU Health & Medicine [132] (denoted MMMU*); (iii) Expert-level reasoning: MedXpertQA-MM [160] (Total score); (iv) Multi-granularity: GMAI-MMBench [127].

> **数据策略解读**:
> - Stage I: 50K 大规模 image-text pairs（覆盖放射学和病理学）做语义对齐——数据量大但只需要 NTP loss
> - Stage II/III: ~4K 高质量 mixed-modality RL samples——数据量小但 label 质量高，且 MedSAM3 提供 region mask
> - 关键设计：Stage II 的 4K 数据覆盖 8 种成像模态，确保 causal reward 在多模态上都有意义

**Baselines.** We compare against four categories of methods: (1) General VLMs: Qwen3-VL-8B [2] (our base model), InternVL3-8B [157]; (2) Medical-specific VLMs: RadFM [117], LLaVA-Med [53], GMAI-VL [59], HuatuoGPT-Vision [7], BiMediX2-8B [81], MedMO-8B [14]; (3) RL-enhanced medical reasoning: MedVLM-R1-2B [84], Med-R1-3B [47], MediX-R1-8B [80], MMedExpert-R1-7B [15]; (4) Latent-space reasoning: Coconut$^{\dagger}$ [28], MCOUT-Multi$^{\dagger}$ [85], IVT-LR$^{\dagger}$ [4] ($^{\dagger}$: adapted with identical Qwen3-VL-8B backbone and training data). We additionally report MedSynapse-V-4B on the Qwen3-VL-4B backbone to assess scalability.

> **Baseline 选择解读**: 四类 baseline 各有侧重——(1) 通用 VLM 提供 zero-shot 基线，(2) 医学专用 VLM 提供 domain-specific 对比，(3) RL-CoT 提供"当前最优离散推理"对比，(4) 通用潜空间方法（统一 backbone + 训练数据适配）提供"不加医学先验的纯潜空间推理"对比。这种全覆盖的 baseline 设计使得每个组件（先验、因果精炼、蒸馏）的贡献可以清晰归因。

**Implementation Details.** Our framework builds upon Qwen3-VL-8B-Instruct [2]. The frozen anatomical encoder $\mathcal{E}_{ana}$ employs MedSAM3 [70] pre-trained on large-scale multi-organ segmentation datasets. Stage I freezes both VLM and $\mathcal{E}_{ana}$, training only the Diagnostic Memory Sampler $\mathcal{P}_{\phi}$ with lr = $2\times 10^{-4}$ for 3 epochs. The diagnostic probe count is N = 16; $\mathcal{P}_{\phi}$ is a 2-layer cross-attention Transformer with output dimension $d_m = 4096$ (matching Qwen3-VL-8B). Images are processed at native dynamic resolution following Qwen3-VL's default configuration. Stage II freezes $\mathcal{P}_{\phi}$ and adapts VLM via LoRA [31] (rank=64, applied to all attention layers). GRPO generates G = 4 candidate trajectories per sample, with clipping coefficient $\varepsilon$ = 0.2, reward weights $\lambda_{acc} = 1.0$ and $\lambda_{causal} = 0.5$, training for 200 steps with a rollout batch size of 32. Max generation length is 1024 tokens. Stage III introduces the Autonomous Memory Module $\mathcal{A}_{\psi}$ (2-layer MLP + LayerNorm, input from VLM's visual encoder features), with JSD coefficient $\beta = 0.5$, lr = $1\times 10^{-4}$, 3 epochs. For each sample we draw one on-policy trajectory $\hat{y} \sim \pi^{-}$ per gradient step; the Stage II data is reused with identical preprocessing. Closed-ended VQA tasks report overall accuracy (%). For GMAI-MMBench and MedXpertQA-MM, we follow their respective official evaluation protocols. Inference efficiency is measured quantitatively as ms/sample and peak GPU memory (GB). More details are provided in the supplementary material.

> **实现关键点**: (1) Stage II 仅训练 LoRA (83.9M params, ~1.0% backbone)，保持预训练知识完整性；(2) GRPO 的 G=4 平衡了梯度估计稳定性和计算成本；(3) Stage III 每 gradient step 从 student 采样一条 on-policy 轨迹——这是 on-policy distillation 的关键，确保训练分布与推理分布一致。

---

### 3.2 Main Results

As shown in Table 1, MedSynapse-V (w/ $\mathcal{E}_{ana}$) achieves the highest average of 61.4%, and the encoder-free MedSynapse-V (IMT) retains 59.6%, surpassing all baselines. Compared to the strongest RL baseline MMedExpert-R1 (55.7%), MedSynapse-V (IMT) leads by +3.9 pp without any auxiliary module at inference, with the largest margins on visual-grounding benchmarks (VQA-RAD +9.0, SLAKE +7.0, PathVQA +6.7), where discrete CoT tokens are prone to attenuating early visual evidence across long reasoning chains. On GMAI-MMBench spanning 38 modalities, MedSynapse-V scores 54.8%, confirming that the anatomical priors generalize beyond the training distribution.

> **主结果解读 — 三个关键发现**:
>
> | 发现 | 数据 | 含义 |
> |------|------|------|
> | IMT 近乎无损 | w/ encoder 61.4% -> IMT 59.6%, 仅损失 1.8 pp | 蒸馏效率极高，encoder 的知识被有效内化 |
> | Visual-grounding benchmark 上最大优势 | VQA-RAD +9.0, SLAKE +7.0, PathVQA +6.7 vs. MMedExpert-R1 | 离散 CoT 在长推理链中视觉证据衰减，latent memory 避免了这个问题 |
> | 解剖先验泛化 | GMAI-MMBench (38 modalities) 54.8% | 虽然训练仅 8 种模态，但结构化先验具有跨模态迁移能力 |

RL baselines reveal a specialization dilemma. MediX-R1 benefits from multilingual pretraining and leads on PMC-VQA (56.2%), yet this breadth dilutes radiology-specific precision (VQA-RAD: 56.4%); MMedExpert-R1 achieves the most balanced profile by leveraging guideline-based reward. Small-scale models (MedVLM-R1 2B, Med-R1 3B) collapse on out-of-domain tasks (MedXpert below 19%), confirming that parameter capacity sets a hard ceiling RL alone cannot raise. In contrast, MedSynapse-V sidesteps this dilemma by injecting latent priors that benefit all task types uniformly, achieving the top performance on every benchmark without task-specific tuning.

> **RL-CoT 的专业化困境 vs. MedSynapse-V 的统一优势**: RL-CoT 方法在不同 benchmark 上表现不均衡——MediX-R1 在 PMC-VQA 领先但放射学精度被稀释，小模型在域外任务崩盘。MedSynapse-V 通过注入隐空间先验**统一地**提升所有任务类型，无需任务特定调优。核心原因：latent memory 编码的是"如何看图像"的诊断基础能力，而非某个特定 benchmark 的解题策略。

Latent methods require domain priors. Among adapted latent baselines, the hierarchy Coconut (44.5%) < MCOUT-Multi (47.9%) < IVT-LR (50.5%) tracks optimization sophistication, yet even IVT-LR barely exceeds zero-shot Qwen3-VL-8B (48.6%). This inversion reveals that latent compression without clinical grounding encodes statistical shortcuts rather than diagnostic logic; the 10.9 pp gap to MedSynapse-V confirms that prior injection and causal calibration are prerequisites for effective latent reasoning in medicine.

> **关键实验发现 — 纯潜空间推理在医学场景的反转**: IVT-LR 是最复杂的通用潜空间方法，但仅略高于 zero-shot baseline (50.5% vs. 48.6%)。这说明在医学领域，**没有领域先验的潜空间压缩编码的是统计捷径而非诊断逻辑**。10.9 pp 的差距证明了先验注入和因果校准是医学潜空间推理的必要前提。

**Scaling efficiency.** MedSynapse-V-4B (w/ $\mathcal{E}_{ana}$) reaches 54.7% with roughly half the parameters of 7B baselines; after encoder removal the IMT variant still achieves 52.7%, surpassing MediX-R1-8B (49.9%). This efficiency stems from a structural advantage: diagnostic expertise is distilled into 16 compact memory vectors consumed in a single forward pass, rather than spread across 150+ verbose reasoning tokens.

> **Scaling Eficiency 解读**: 4B 参数的 IMT 版本 (52.7%) 超过 8B 的 MediX-R1 (49.9%)。优势来自结构设计而非参数规模——16 个 memory vector 在一次前向传播中被消费，而非扩散在 150+ 冗余推理 token 中。

---

### 3.3 Ablation Study

Table 2 reports comprehensive ablation study results.

**(i) Progressive training stages.** MQPM warmup is indispensable: skipping it collapses Avg to 52.9, barely above zero-shot (54.2%), because randomly initialized memory destabilizes early RL training. Replacing CCR with SFT reaches 59.2 but lags by 8.5 pp due to limited out-of-distribution generalization. The full pipeline (Avg 67.7) confirms non-redundant contributions: MQPM grounds semantics, CCR refines via exploration, IMT compresses into an autonomous pathway.

> **消融解读 — 三阶段不可替代性**:
>
> | 配置 | Avg | 关键发现 |
> |------|-----|---------|
> | Qwen3-VL-8B (zero-shot) | 54.2 | 基线 |
> | MQPM -> IMT (skip CCR) | 55.4 | 仅 +1.2 pp，warmup 本身贡献有限 |
> | MQPM -> SFT -> IMT (replace CCR) | 59.2 | SFT 比 RL 弱 8.5 pp，且缺乏分布外泛化 |
> | Direct RL -> IMT (skip MQPM) | 52.9 | **低于 zero-shot**！随机 memory 导致 RL 训练崩溃 |
> | MQPM -> CCR -> IMT (full) | 67.7 | 完整 pipeline，三阶段缺一不可 |

**(ii) Reward design.** $r_{causal}$ is the dominant reward component (+4.1 pp, 63.6 -> 67.7). Without causal pressure the model bypasses $M$ via direct shortcuts, treating memory as inert padding; the counterfactual intervention penalizes trajectories insensitive to diagnostic regions. The effect concentrates on radiology benchmarks and persists after IMT, indicating stronger memory utilization transfers more faithfully through distillation.

> **消融解读 — $r_{causal}$ 的核心作用**:
> - $r_{acc}$ only: 63.6% -> 模型 bypass M 直接走捷径
> - $r_{acc} + r_{causal}$: 67.7% -> +4.1 pp，因果压力迫使模型真正利用 memory
> - 效果集中在放射学 benchmark：因为这些任务最依赖视觉 grounding
> - 效果在 IMT 后仍然保持：**更强的 memory 利用模式通过蒸馏更忠实地传递**

**(iii) Encoder retention vs. removal.** IMT achieves near-lossless removal: only 1.4 pp degradation (69.1 -> 67.7) while latency drops 39% and memory decreases 6.3 GB. The gap is not uniform: core VQA metrics degrade minimally, whereas MedXpert and GMAI suffer more, suggesting complex reasoning depends more on encoder-derived priors than closed-ended recognition.

**(iv) Anatomical encoder choice.** MedSAM3 outperforms SAM-Med2D by 4.4 pp (67.7 vs. 63.3), reflecting richer spatial representations from multi-organ segmentation pretraining. Random initialization yields only 52.0, confirming that gains originate from what the encoder knows, rather than how memory is aggregated.

> **消融解读 — Encoder 质量决定 memory 上限**: MedSAM3 (multi-organ segmentation pretraining) > SAM-Med2D > Random init (52.0%, 甚至低于 zero-shot)。这确认了一个核心命题：**收益来自 encoder 知道什么，而非 memory 如何被聚合**。

**(v) Probe count N.** As shown in Fig. 4, N=16 balances expressiveness against redundancy. The CCR to SFT gap widens with N (3.5 pp at N=4 vs. 7.2 pp at N=16), revealing that larger memory pools amplify bypass shortcuts and therefore benefit disproportionately from causal refinement.

> **N=16 的最优性**: N 太小 -> 容量不足；N 太大 -> 引入无关信号稀释诊断信息。更重要的是：N 越大，CCR 与 SFT 的差距越大（3.5 pp -> 7.2 pp），说明更大的 memory pool 放大了 shortcut 风险，因此更需要因果精炼。

---

### 3.4 In-Depth Case Analysis

As illustrated in Fig. 5, we compare MedSynapse-V with Med-R1 [47] and MMedExpert-R1 [15] across three distinct imaging modalities. Both baselines produce verbose CoT reasoning (~185-238 tokens) yet arrive at incorrect diagnoses due to hallucinated observations erroneously propagating through the chain. In the CT case, Med-R1 fabricates pleural thickening in the left upper lobe, while MMedExpert-R1 hallucinates a laminated calcification pattern and mischaracterizes the nodule as a benign granuloma. In the MRI case, Med-R1 misidentifies the extra-axial mass as intra-axial and concludes glioblastoma, whereas MMedExpert-R1 fabricates ring enhancement with central necrosis, both missing the classic meningioma presentation. In the ultrasound case, Med-R1 hallucinates gallbladder wall thickening to over-diagnose acute cholecystitis, while MMedExpert-R1 denies posterior acoustic shadowing and misdiagnoses a gallbladder polyp. In contrast, MedSynapse-V generates concise, correct answers (~34-44 tokens) without explicit CoT, demonstrating that diagnostic implicit memory provides sufficient latent guidance while avoiding the hallucination cascades inherent in token-level CoT.

> **案例分析解读 — CoT 幻觉级联 vs. Latent Memory 准确简洁**:
>
> | 案例 | Med-R1 幻觉 | MMedExpert-R1 幻觉 | MedSynapse-V 正确诊断 |
> |------|-----------|-------------------|---------------------|
> | CT | 左上叶胸膜增厚 | 层状钙化+良性肉芽肿 | 简洁正确 (34-44 tokens) |
> | MRI | 轴内肿块->胶质母细胞瘤 | 环形强化+中央坏死 | 脑膜瘤经典表现 |
> | 超声 | 胆囊壁增厚->急性胆囊炎 | 否认后方声影->胆囊息肉 | 简洁正确 |
>
> 关键洞察：CoT 方法在推理链早期产生一个幻觉观察，后续推理步骤基于这个错误前提进一步放大错误——形成"幻觉级联"。Latent memory 避免了这种级联，因为诊断信号在连续隐空间中一次性编码，而非逐步离散展开。

---

### 3.5 Efficiency, RL Dynamics, and Latent Space

**Performance-efficiency trade-off.** As shown in Fig. 6, MedSynapse-V (IMT) achieves 59.6% at 2.6 s/sample, comparable to zero-shot Qwen3-VL-8B (48.6%, 2.8 s) since both share the same backbone and the 16 memory vectors add negligible overhead. Full-scale 7-8B CoT methods (MediX-R1, MMedExpert-R1) require 5.8 s each due to 300-400 autoregressive reasoning tokens, while smaller CoT models (MedVLM-R1 2B, Med-R1 3B) offset verbosity with faster per-token speed yet remain 18-21 pp below MedSynapse-V. This confirms that compact latent memory provides diagnostic grounding without the token-generation overhead of full-scale CoT.

> **效率-性能权衡的核心对比**:
> - 标准 VLM zero-shot: 48.6% @ 2.8s
> - MedSynapse-V IMT: 59.6% @ 2.6s (更快！因为 memory conditioning 使输出更简洁)
> - 大规模 CoT (8B): 49.9-55.7% @ 5.8s
> - 小规模 CoT (2-3B): 38.9-40.2% @ 更快但性能差距 18-21 pp

**Training dynamics.** Fig. 7 shows the full model (w/ $r_{causal}$) improving steadily to ~0.88 with a transient exploration dip near step 900, where the policy sacrifices reward to explore memory-reliant generation strategies, while the w/o $r_{causal}$ ablation plateaus at ~0.48 throughout the training. This confirms that accuracy-only reward cannot distinguish memory-dependent from shortcut trajectories; without causal pressure the model bypasses $M$ entirely, treating injected memory as inert padding.

> **训练动态解读**: w/ $r_{causal}$ 的 reward 从 ~0 爬升至 ~0.88，期间出现 ~step 900 的 transient exploration dip——策略牺牲短期 reward 探索 memory-reliant 生成策略。w/o $r_{causal}$ 的 reward 始终停滞在 ~0.48，因为 accuracy-only reward 无法区分"利用 memory"和"绕过 memory"的轨迹。这直观验证了 causal reward 的必要性。

**Latent space structure.** Fig. 8 visualizes the evolved memory $M_{auto}$ via t-SNE across three granularities. At the cross-modality level (a), eight imaging types form compact clusters with clinically coherent proximity (e.g., CT and MRI lie adjacent; dermoscopy and fundus form a nearby pair). Within individual modalities (b, c), disease subtypes further segregate: CT memory separates lung nodules, liver lesions, kidney cysts, pneumonia, and aortic aneurysms, while pathology memory distinguishes adenocarcinoma, squamous cell carcinoma, normal tissue, lymphoma, and melanoma. This hierarchical organization confirms that $r_{causal}$ reshapes the latent space into a diagnostically meaningful manifold rather than merely boosting task accuracy.

> **t-SNE 可视化 — 三层结构证明了 $r_{causal}$ 的 manifold shaping 效果**:
> 1. **跨模态层**: 8 种成像模态形成分离紧凑的聚类，具有临床一致的邻近关系（CT 和 MRI 相邻；皮肤镜和眼底镜相邻）
> 2. **疾病亚型层**: 在 CT 内，肺结节/肝脏病变/肾囊肿/肺炎/主动脉瘤各自成簇；在病理内，腺癌/鳞癌/正常组织/淋巴瘤/黑色素瘤各自成簇
> 3. **本质**: $r_{causal}$ 不仅提分，更将隐空间重塑为具有诊断意义的流形结构

**Why latent memory evolution works.** Our ablations pinpoint two necessary conditions that general latent methods lack. First, structured priors are indispensable: replacing MedSAM3 with a random encoder collapses Avg from 67.7% to 52.0% (Table 2d). Second, causal calibration activates the priors: $r_{causal}$ lifts accuracy by 4.1 pp (Table 2b) and reorganizes memory into the hierarchical diagnostic manifold shown in Fig. 8. Neither condition alone suffices, and their synergy is precisely what general latent methods lack.

> **核心结论**: 医学潜空间推理的两个必要条件——(1) 结构化先验 (structured priors) 提供表示基础，(2) 因果校准 (causal calibration) 激活先验。两者缺一不可，其协同正是通用潜空间方法所缺乏的。

---

## 三、Summary

- **主结果**: MedSynapse-V (IMT) 59.6% Avg，超越最强 RL baseline MMedExpert-R1 (55.7%) 3.9 pp
- **消融核心发现**: (1) 三阶段缺一不可，(2) $r_{causal}$ 贡献 +4.1 pp，(3) IMT 仅损失 1.4 pp 但延迟降 39%、显存降 6.3 GB，(4) Encoder 知识决定上界，(5) N=16 最优
- **效率**: IMT 2.6s/sample vs. CoT 5.8s/sample vs. zero-shot 2.8s/sample
- **Latent Space 结构**: $r_{causal}$ 将 memory 空间重塑为分层诊断流形
- **必要条件**: 结构化先验 + 因果校准，两者缺一不可
