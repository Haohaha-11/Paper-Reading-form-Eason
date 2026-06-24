[← 返回 README](../README.md)

# 5--12. Supplementary Material

## 一、Preview

附录部分覆盖：(5) 实现细节与超参数，(6) 训练动态分析，(7) Benchmark 数据集统计，(8) 额外分析（per-modality 分解、因果干预可视化、延迟分析、memory 演化、设计选择、mask 鲁棒性），(9) 额外定性结果与失败案例分析，(10) 评估 Prompt 模板，(11) 扩展消融，(12) 相关工作综述。

---

## 二、原始文本与批注

### 5. Implementation Details

**Training Configuration.** Table 3 provides the hyperparameter configuration across all three training stages for reproducibility. Key configurations:

| Stage | Trainable Module | Key Hyperparams |
|-------|-----------------|-----------------|
| Stage I (MQPM) | $\mathcal{P}_{\phi}$ only (~12.6M) | lr=$2\times10^{-4}$, 3 epochs, batch=32, AdamW |
| Stage II (CCR) | LoRA adapters (~83.9M) | rank=64, alpha=128, G=4, $\varepsilon$=0.2, lr=$1\times10^{-5}$, 200 steps |
| Stage III (IMT) | $\mathcal{A}_{\psi}$ only (~33.6M) | 2-layer MLP+LayerNorm, $\beta$=0.5, lr=$1\times10^{-4}$, 3 epochs |

Infrastructure: 4x A100 80GB, bf16 + FlashAttention-2, ~38 hours total training, 5-fold cross-validation.

**Architectural Details.** $\mathcal{P}_{\phi}$: 2-layer Transformer, 8 heads (head dim=128), 16 meta-query probes initialized via truncated normal ($\sigma=0.02$), final linear projection to 4096-dim. $\mathcal{A}_{\psi}$: pooled visual features -> 2x 4096-dim linear layers with GELU + LayerNorm -> N x $d_h$. MedSAM3 ViT-B backbone, $64\times64\times1024$ spatial features (flattened to M=4096 tokens), segmentation head threshold 0.7 for region masks. LoRA: applied to all attention projection matrices across 32 layers of Qwen3-VL-8B.

**Evaluation Details.** Closed-ended VQA: extract first option letter (A/B/C/D/E) by regex, fallback to fuzzy string matching. Open-ended: exact match after lowercasing + stripping punctuation. Greedy decoding (temperature=0, top-p=1.0), max 512 generation tokens.

> **实现关键点**: (1) $\mathcal{P}_{\phi}$ 仅 12.6M 参数——轻量到足以高效训练；(2) LoRA 覆盖所有 attention 层——确保 RL 梯度能充分塑造 memory 利用模式；(3) 评估用 greedy decoding——确保结果可复现。

---

### 6. Training Dynamics Analysis

Figure 10 extends the Stage II reward summary with additional monitoring metrics:

- **Stage I (panel d)**: NTP loss 从 ~2.6 快速降至 ~0.45 (epoch 1)，三轮平滑收敛。Epoch 边界的小跳变来自 lr schedule，不影响最终收敛。
- **Stage II (panels a-c, e)**: Full model reward 爬升至 ~0.88（step ~150 出现 transient exploration dip）。$r_{causal}$ 从 near-zero 升至 ~0.35，确认 memory 利用率逐步提升。$r_{causal}$ 稳定梯度范数在 [0.2, 0.6]，去除后频繁出现超 clip threshold 的 spike。Policy loss 单调下降，KL divergence 控制在 <0.02。
- **Stage III (panel f)**: JSD 从 ~0.42 降至 ~0.035，output agreement 从 72% 升至 ~97%。与 Table 2c 的 near-lossless encoder removal ($\Delta$=1.4 pp) 一致。

> **训练动态的三个关键观察**:
> 1. **Transient exploration dip (step ~150)**: 策略暂时牺牲 reward 探索 memory-reliant 生成策略——这是 RL 中的典型 exploration-exploitation 现象，w/o $r_{causal}$ 无此现象
> 2. **梯度稳定性**: $r_{causal}$ 是梯度稳定器——有它时梯度范数稳定在 [0.2, 0.6]，无它时频繁出现超阈值 spike
> 3. **蒸馏保真度**: JSD 降至 0.035 + agreement 97%，表明 student 已在分布层面几乎完美复制 teacher 的诊断行为

---

### 7. Benchmark Dataset Statistics

Evaluation suite covers: (1) CE tasks: VQA-RAD (451 radiology Qs), SLAKE (1,061 mixed-modality), PathVQA (6,719 pathology); (2) MC tasks: PMC-VQA (10,000), MMMU* Health & Medicine (150), MedXpertQA-MM (960); (3) Multi-granularity: GMAI-MMBench (38 modalities, 2,847 Qs). Primary metric: accuracy (except MedXpertQA-MM: Total Score).

Training pipeline: Stage I: 50K PubMedVision image-text pairs (radiology + pathology); Stage II: ~4K mixed-modality RL set (3K OmniMedVQA closed-ended + 1K SLAKE/PathVQA open-ended); Stage III: reuse Stage II data. Rigorous filtering ensures no test-train overlap.

---

### 8. Additional Analysis

#### 8.1 Per-Modality Breakdown on OmniMedVQA

Table 4: MedSynapse-V achieves consistent gains across all 8 OmniMedVQA modalities, with largest gains on radiology-centric modalities (CT: +14.4, MRI: +14.9, X-ray: +13.1). MMedExpert-R1 gains are modest on challenging modalities (OCT +5.6, Fundus +5.6) where explicit CoT struggles with subtle spatial patterns. MedSynapse-V's latent memory yields substantially larger gains (OCT +11.9, Fundus +11.8), confirming continuous diagnostic memory encodes fine-grained anatomical features more effectively than discrete token reasoning.

> **按模态分解的关键发现**: 放射学中心模态 (CT/MRI/X-ray) 获益最大，因为结构化解剖先验在这些模态上最信息丰富。OCT/眼底等细粒度空间模式模态上 CoT 的提升有限(+5.6)，而 latent memory 提升显著(+11.8-11.9)，证实了连续隐空间记忆编码细粒度特征的优越性。

#### 8.2 Visualization of Causal Counterfactual Intervention

Figure 11: CCR 重塑 memory attention 的空间分布。眼底案例中，post-CCR 的 memory attention 紧密对齐视网膜病灶 (微动脉瘤+硬性渗出)，而视盘和健康血管仅获最小激活。皮肤镜案例中，post-CCR attention 集中在病变边缘 (不对称性、边界不规则、颜色异质性最显著处)，符合临床 ABCD 标准。这些可视化提供了 $r_{causal}$ 成功建立 memory 与病理相关区域间因果依赖的直接证据。

#### 8.3 Inference Latency and Parameter Count

Table 5: Prefill latency 102ms (与 vanilla 完全一致)，$\mathcal{A}_{\psi}$ 仅贡献 4ms。关键机制：16 个 memory vector 在 prefill 时被构建进 KV cache，之后每个 decode step 可以以零额外代价 attend 到诊断先验。这种 latent conditioning 引导模型生成更短更果断的输出 (~34-44 answer tokens vs. zero-shot 的 ~50-80)，将 end-to-end 延迟从 ~2.8s 降至 ~2.6s。

注意 "ms/token" 列 (Table 2) 测量 per-token 解码延迟。zero-shot baseline 的 ms/token 更高 (126 vs. 102)，因为缺乏 memory conditioning 时 attention 必须分散到完整 visual token 序列，产生更宽的 attention pattern 和更慢的 per-step 计算。

> **延迟分析关键洞察**: Memory vectors 在 KV cache 中的作用是一种"注意力引导"——它们不是增加了计算量，而是浓缩了信息使 attention 更高效，从而反直觉地降低了 per-token 延迟。

#### 8.4 Memory Evolution Across Training Stages

Figure 12: Before MQPM -> 混沌分布，8 种模态完全混合。Stage I (MQPM) -> 引入共享表示基础，但 CT/MRI 仍有重叠，OCT 划分不佳。Stage II (CCR) -> $r_{causal}$ 将流形重塑为紧凑、分离的聚类，放射模态和表面成像形成不同邻域。Stage III (IMT) -> $\mathcal{M}_{auto}$ 忠实内化了这种精炼结构，拓扑结构几乎与 panel (c) 一致，验证了 near-lossless distillation ($\Delta$=1.4 pp)。

#### 8.5 Memory Synthesis Design Choices

Table 6 (left): 聚合策略比较——Avg-pool concat (+4.2 pp over zero-shot)、Linear projector (+6.0 pp)、Meta-query $\mathcal{P}_{\phi}$ (+13.5 pp)。Meta-query 的巨大优势表明通过 cross-attention 进行选择性、输入条件的空间特征聚合至关重要——静态压缩丢弃了可学习 probe 能选择性保留的细粒度空间线索。

Table 6 (right): Question conditioning in $\mathcal{A}_{\psi}$——包含 question tokens 带来 +0.7 pp 的一致提升。增益虽小但一致：因为 VLM self-attention 已将 answer 条件于 q；额外 query signal 主要帮助 $\mathcal{A}_{\psi}$ 区分多个诊断假设竞争相同视觉特征的情况。

> **设计选择解读**: Cross-attention 的 meta-query aggregation 是核心性能来源——不是简单的维度压缩，而是允许每个 probe 学习关注特定病理语义模式（边界不规则性、密度异质性、血管-组织空间关系等）。

#### 8.6 Robustness of Causal Intervention to Mask Quality

Table 7 (threshold $\tau$): 所有阈值均优于 no-mask baseline (63.6%)。$\tau \in [0.5, 0.8]$ 内性能稳定 (仅 1.1 pp spread)。极端值退化：$\tau$=0.3 时 mask 42.6% 图像 (干预过于破坏性)，$\tau$=0.9 时 mask 仅 5.4% (干预过弱)。

Table 8 (mask rank): Rank-2 保留大部分收益 (-1.2 pp vs. Rank-1)，random mask 仍优于 no-mask baseline (+1.6 pp)。单调排序 Rank-1 > Rank-2 > Random > None 确认 mask 质量有帮助但非关键：causal reward 利用的是 masked vs. unmasked 条件下的相对对比，而非依赖像素精确分割。

> **Mask 鲁棒性的深层含义**: $r_{causal}$ 对 mask 精度不敏感的关键在于其本质是"相对对比"信号——即使 mask 不完全精确，只要它大致覆盖了诊断相关区域的一部分，原始与干预条件之间仍存在概率差异。这使得该方法在实际部署中对分割模型的噪声具有一定容忍度。

---

### 9. Additional Qualitative Results

#### 9.1 Additional Representative Cases

Figure 13: 胸片案例——Med-R1 编造双侧间质混浊影并声称肋膈角锐利，忽略了明显的胸腔积液；MMedExpert-R1 幻象凸面边界伴空洞形成，误诊为肺脓肿。病理案例——Med-R1 错误描述细胞极性保留和完整基底膜，倾向纤维腺瘤；MMedExpert-R1 编造淋巴血管侵犯和粉刺样坏死，误分类为浸润性小叶癌。头部 CT——Med-R1 否认高密度病灶存在，诊断为缺血性脑梗死；MMedExpert-R1 幻象环形强化伴中央坏死，诊断为脑脓肿。MedSynapse-V 在 38-43 tokens 内直接给出正确发现，无显式 CoT。

> **定性分析的核心信息**: 所有 CoT 方法的错误遵循同一模式——在推理链中早期产生幻觉观察，后续步骤基于错误前提进一步放大。MedSynapse-V 避免了这种级联，因为诊断信号在连续隐空间中一次性编码，而非逐步离散展开。

#### 9.2 Failure Case Analysis

Figure 14 揭示三种主要失败模式:

(a) **稀有模态欠表征**: OCT 训练占比最小 (~25%)，per-modality 准确率最低，说明 memory 质量在先验暴露不足时退化。

(b) **多病灶歧义**: 单病灶准确率 78% -> 多病灶 52%，固定 N=16 的 memory pool 在多发病变竞争表示容量时饱和。

(c) **细微特征区分**: 皮肤镜 borderline cases (良性 vs. 非典型痣) 中，confidence < 0.3 且 correctness = 0 的样本簇表明 memory 的判别粒度不足以区分细微差异。

> **失败案例的未来方向提示**: (1) 均衡模态采样策略，(2) 自适应 memory pool 大小，(3) 校准化不确定性估计——这些都是论文自身指出的未来改进方向，也成为该方向的开放研究问题。

---

### 10. Evaluation Prompt Templates

Closed-ended template (Fig. 15): "You are a helpful medical assistant. Answer the question based on the image." + image + question + options + "Please answer with the option letter only."

Open-ended template (Fig. 16): "You are a helpful medical assistant. Provide a concise answer to the question." + image + question + "Answer the question using a single word or phrase."

> **关键**: MedSynapse-V 的 $\mathcal{M}_{auto}$ 在隐空间中自主生成并注入，无需修改 surface-level prompt——无需额外 text token、特殊标记或推理引导指令。这与 CoT 需要添加 "Let's think step by step" 以及 Coconut 需要 `<bot>`/`<eot>` 标记形成对比。

---

### 11. Extended Ablation Studies

**Table 9 (left) — Memory Injection Position**:

| Position | Avg |
|----------|-----|
| Before visual tokens | 65.5 |
| V - M - q | 67.6 |
| After q (default) | 69.3 |
| Interleaved w/ q | 68.2 |

Default position (after q) 最优——answer token 可同时 attend visual features 和 memory。Before V 最差——self-attention 无法根据 question context 来 condition memory。

**Table 9 (right) — GRPO Group Size G**:

G=4 实现最优精度-成本平衡。G=2 噪声优势估计 (67.3%)，G=6/8 仅 +0.1-0.3 pp 但 GPU 时间增加 1.4-2x。收益递减确认 4 条轨迹对复合 reward 下的稳定优势估计已足够。

**Table 10 (left) — IMT Divergence Function**:

| Divergence | Avg |
|------------|-----|
| Forward KL | 67.2 |
| Reverse KL | 66.8 |
| JSD (β=0.3) | 68.6 |
| JSD (β=0.5) | 69.3 |
| JSD (β=0.7) | 68.9 |

JSD (β=0.5) 最优。Forward KL 的 mode-covering 行为稀释诊断特异性，Reverse KL 导致 mode-seeking collapse。对称 JSD 提供平衡学习信号，且对 β ∈ [0.3, 0.7] 鲁棒。

**Table 10 (right) — Causal Reward Weight $\lambda_{causal}$**:

$\lambda_{causal}$ ∈ [0.3, 0.7] 内性能鲁棒，峰值在 0.5。$\lambda_{causal}=0$ 导致模型 bypass memory (65.4%)，≥1.0 过度惩罚轨迹导致训练不稳定 (67.7%)。

---

### 12. Related Works

论文将相关工作分为四个领域，覆盖了非常广泛的文献：

1. **Latent Computation and Memory-Augmented Reasoning**: COCONUT, CoDi, Heima, latent regulated generation (Seek in the Dark, Deliberation in Latent Space), memory evolution (RGMem), prompt-based continual learning, counterfactual explanation, causal dynamics of modality arbitration.

2. **RL for VLMs**: PPO, DPO, GRPO + medical applications (MedVLM-R1, Med-R1, MediX-R1, MMedExpert-R1), spatial reasoning, cognitive supersensing, hallucination mitigation, ofine RL methods (federated, conservative estimation, collapse suppression).

3. **Medical Image Understanding and Efficient Deployment**: medical segmentation (annotation efficient, semi-supervised, federated), medical diagnosis (knowledge enhanced, domain adaptive), clinical workflow analysis, efficient inference (speculative decoding, quantization, NPU co-design).

4. **VLM Backbones and Benchmarks**: Qwen3-VL, InternVL3, MedSAM3, SAM-Med2D, OmniMedVQA, MMMU etc.

> **相关工作映射 — MedSynapse-V 在文献中的位置**:
> - 与 Latent Computation 的区别：引入了领域先验注入 + 因果校准
> - 与 RL for VLM 的区别：reward 设计从 accuracy-only 升级到 accuracy + causal counterfactual
> - 与 Medical VLM 的区别：从外部知识注入 -> 内生 memory 演化
> - 与 Efficient Deployment 的共鸣：16 memory vectors 的核心设计哲学——"不是所有计算都需要同等投入"

---

## 三、Summary

- **实现关键**: LoRA rank=64 覆盖所有 attention 层，GRPO G=4 稳定高效，IMT on-policy sampling 确保分布一致
- **训练动态**: $r_{causal}$ 既是因果校准器也是梯度稳定器，transient exploration dip 是 RL 的正常探索行为
- **编码器选择**: 收益来自 encoder 知道什么 (MedSAM3 > SAM-Med2D > Random)，而非 memory 聚合方式
- **Mask 鲁棒性**: $r_{causal}$ 对 mask 精度有较好鲁棒性，利用相对对比而非像素精确分割
- **失败模式**: 稀有模态欠表征、多病灶饱和、细微特征区分不足——指向自适应 memory 容量和校准化不确定性估计
- **相关工作的核心差异**: MedSynapse-V 独有"先验注入 + 因果校准 + 内生蒸馏"三阶段组合
