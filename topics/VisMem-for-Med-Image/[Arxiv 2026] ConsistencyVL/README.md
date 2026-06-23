# Visuals Lie, Consistency Speaks: Disentangling Spatial Attention from Reliability in Vision-Language Models

**Authors:** Logan Mann, Yi Xia, Ajit Saravanan, Ishan Dave, Saadullah Ismail, Shikhar Shiromani, Emily Huang, Ruizhe Li, Kevin Zhu

**Affiliations:** UC Santa Barbara, Algoverse AI Research, UC Berkeley, Independent Researcher

**Venue:** arXiv 2026 (2606.17389)

**Code:** https://github.com/itsloganmann/VLM-Reliability-Probe

---

## 一句话总结

本文系统性地证明了视觉语言模型（VLM）中的空间注意力模式与输出正确性之间近乎零相关（R ≈ 0.001）；相反，可信度信号最优的捕获方式是生成时动态（Self-Consistency，R = 0.429）和隐藏状态探针（AUROC > 0.95），揭示出视觉定位与真实生成之间存在根本性的"Symbolic Detachment"（符号性脱钩）。

---

## 核心贡献

1. **"Cluster Failure" 发现：** 空间注意力指标（聚类数 C_k、空间熵 H_s）在三族 VLM（LLaVA-1.5、PaliGemma、Qwen2-VL）中与正确性的相关性接近零（R ≈ 0.001），直接反驳了被广泛接受的"Attention-Confidence Assumption"（注意力-置信度假定）。

2. **"Symbolic Detachment" 机制：** 逐层注意力演化分析揭示了"Early Locking"——模型在早期就锐化了视觉注意力，但在后期却将其扩散，切断了感知与生成之间的联系。这解释了为什么注意力图在统计上与正确答案正交。

3. **隐藏状态探针作为可信度检测器：** 在内部隐藏状态上训练的探针在单次推理中实现了 AUROC > 0.95 的答案正确性预测，大幅超越基于注意力的指标（AUROC ≈ 0.50）和输出置信度（AUROC ≈ 0.54）。

4. **因果鲁棒性的架构分化：** 大规模消融实验揭示，LLaVA 依赖脆弱、局部的后期瓶颈神经元（仅消融 5 个神经元，对象识别准确率即下降 8.3pp），而 PaliGemma 和 Qwen2-VL 将可信度分布到全局，即使在预测层中摧毁超过 50% 的神经元仍保持鲁棒。

5. **Self-Consistency 作为金标准行为信号：** K=10 条采样推理路径的一致性（R = 0.429，AUROC = 0.78-0.81）是最强的行为可信度信号，尽管推理成本为 10 倍。隐藏状态探针在单次推理成本下实现了更高的 AUROC（最高 0.971）。

---

## 📖 批读导航

| 章节 | 文件 | 说明 |
|---------|------|-------------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | 带批注的完整摘要 |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | 问题动机与注意力-置信度假定 |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) | VLM、幻觉、可解释性、语言先验 |
| 3. Methodology | [03-methodology.md](sections/03-methodology.md) | VRP框架：结构假设 vs 一致性假设 |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) | 完整结果：注意力失败、logit lens、稀疏电路、可信度预测 |
| 5. Conclusion | [05-conclusion.md](sections/05-conclusion.md) | 总结、局限性、未来工作 |

---

## 关键数字

| 指标 | 数值 | 上下文 |
|--------|-------|---------|
| R(C_k, y) -- 聚类数 vs 正确性 | 0.001 (95% CI [-0.034, 0.036]) | 近零相关，p > 0.05 |
| R(H_s, y) -- 空间熵 vs 正确性 | -0.012 (95% CI [-0.047, 0.024]) | 统计上无法区分于噪声 |
| Self-Consistency (SC) R | 0.429 | 最强的行为可信度信号 |
| SC=1 时的精度 | 90.8% (95% CI [88.4, 92.8]%) | 10条样本全部一致时的高精度 |
| 隐藏状态探针 AUROC (LLaVA) | 0.956 (POPE) | 近乎完美的可信度鉴别 |
| 隐藏状态探针 AUROC (Qwen2-VL) | 0.971 (POPE) | 最佳单次推理可信度检测器 |
| 监督注意力探针 AUROC | 0.725 | 注意力携带有限信号 |
| 空间注意力 AUROC | 0.50 | 随机猜测水平 |
| Logit Entropy AUROC | 0.50-0.52 | 校准能力差，作为基线 |
| 输出置信度 AUROC | 0.54 | 略高于随机 |
| MLP 对 Margin 增长的贡献 (LLaVA) | 82.1% | 可信度由特征处理驱动，而非路由 |
| LLaVA 准确率下降 (消融 top-5 神经元，对象识别) | -8.3pp | 脆弱的局部瓶颈 |
| PaliGemma 准确率下降 (消融1000/2048神经元) | -1.0pp | 高度分布式，鲁棒 |
| Qwen2-VL 准确率变化 (消融2000/3584神经元) | +2.0pp (在噪声范围内) | 极端弹性 |
| Qwen2-VL 在 POPE 上 | 28.8% 准确率 | 严重失校准（注意：低模型准确率可能夸大探针 AUROC） |
| 聚合结构分析集 | n = 3,090 | POPE + LLaVA-Bench + 自定义任务 |
| Self-Consistency 成本 | 10x 推理 | K=10 条样本，核采样 (p=0.9, T=0.7) |

---

## 数据流：输入 → 中间表示 → 输出

```
[图像 + 问题]
       |
       v
[阶段 1：结构指标]
  - 从视觉编码器提取交叉注意力图 A^{(l,h)}
  - 对注意力头及答案token位置取平均 → 逐层空间向量 m^{(l)} ∈ R^S
  - 计算：聚类数 (C_k)、空间熵 (H_s)、注意力演化 (ΔH_s)
  - 输出：结构可信度分数（失败：R^2 < 0.08）
       |
       v
[阶段 2：机制探针]
  - Logit Lens：将隐藏状态 h_l 投影到词表空间
  - 计算 Truth Margin ΔM_l = logit(正确答案) - logit(最高错误答案)
  - 训练：密集 MLP 探针 + 稀疏 L1-logistic 探针（作用于隐藏状态）
  - 识别预测性神经元（成功/失败神经元）
  - 因果消融：将识别的神经元置零，测量准确率影响
  - 输出：可信度预测分数（成功：AUROC 最高 0.971）
       |
       v
[阶段 3：行为指标]
  - 采样 K=10 条推理路径（核采样 p=0.9, T=0.7）
  - 计算 Self-Consistency = 各样本之间的回答一致率
  - 输出：行为可信度分数（成功：R = 0.429, AUROC 0.78-0.81）
       |
       v
[最终判断：该回答可信吗？]
  - 最佳单次推理：隐藏状态探针（AUROC 0.95+）
  - 最佳行为信号：Self-Consistency（10倍成本，R = 0.429）
  - 不要使用：注意力图锐度 / 聚类数（AUROC = 0.50）
```

---

## 优缺点与还能做什么

### 优点
- **严谨的跨架构设计：** 测试了三种架构各异的 VLM 家族（prefix-based 的 LLaVA、early-fusion 的 PaliGemma、native-multimodal 的 Qwen2-VL），增强了结论的泛化性
- **多层次分析：** 从相关性（注意力 vs 正确性）推进到因果性（神经元消融）再到机制（logit lens、稀疏电路），形成了完整的科学叙事
- **实践意义明确：** 隐藏状态探针以几乎零额外开销提供单次推理可信度评估，可直接部署落地
- **负结果记录详实：** "Cluster Failure"和注意力失败通过置信区间和监督压力测试获得统计验证
- **诚实的学术定位：** 作者明确指出注意力失败和 self-consistency 是已有发现；创新在于统一的跨架构可信度研究和隐藏状态探针

### 局限 / 风险
- **仅限中等规模模型（7B、3B、7B）：** 更大模型（LLaVA-34B、GPT-4V）可能由于更好的 RLHF 而表现出不同的注意力-可信度关系
- **Qwen2-VL 的低准确率（POPE 上 28.8%）混淆了探针 AUROC：** 在一个大多时候出错的模型上，高探针 AUROC（0.971）引出了探针到底在检测什么的问题
- **神经元消融的效应量中等（对象识别 -8.3pp）：** 因果证据显示了相关性但机制控制有限；神经元是"贡献者"而非"真理单元"
- **主要分析中的基准多样性有限：** POPE 和 LLaVA-Bench 占主导；VQA v2 和 TextVQA 的结果更加混杂
- **Self-Consistency 需要 10 倍推理成本：** 对实时应用不实用；蒸馏方案被提出但未实现
- **未考察校准技术：** 微调或 RLHF 可能使注意力与可信度对齐
- **探针需要标注正确性数据用于训练：** 非零样本方法；需要按模型家族进行分布内校准

---

## 阅读 Q&A 记录

> **Q1:** 如果注意力是因果必要的（遮盖 top 30% 的高注意力 patch 会使准确率下降 8-11pp），为什么它与正确性不相关？
> **A:** 本文给出了一个关键区分：注意力支持*特征提取*（因果上对任务性能必要），但并不编码*关于这些特征的不确定性*（与正确性不相关）。可以这样理解：注意力是"看哪里"的机制，但知道"看了哪里"并不等于知道"是否正确地理解了所见内容"。Symbolic Detachment 现象（Early Locking + Late Diffusion）意味着早期注意力模式在 LLM 解码器做出决策时早已过时。

> **Q2:** 为什么 Qwen2-VL 的 POPE 准确率如此之低（28.8%），却拥有最高的探针 AUROC（0.971）？
> **A:** 这是本文未完全解决的一个潜在混淆因素。当一个模型大多数时候出错时，一个学会检测"模型何时违背了自身偏见"的探针，可能仅仅因为检测到了罕见的正确答案而获得了人为的高 AUROC。作者将此作为一个局限性（模型规模/架构特定效应），但鉴于低基线准确率，高 AUROC 应谨慎解读。

> **Q3:** "self-consistency" 与简单的多数投票有何不同？
> **A:** 这里的 Self-Consistency（SC）是 K=10 条核采样（p=0.9, T=0.7）推理路径之间的回答一致率。本质上是通过基于温度的多样性进行多数投票。关键洞察在于：当10条多样化样本全部一致时，答案高度可信（SC=1 时精度为 90.8%）。

> **Q4:** 为什么 PaliGemma 的探针 AUROC（0.738）低于 LLaVA/Qwen2-VL？
> **A:** PaliGemma 更早地集成视觉证据（峰值在 L14），且解码器更浅（18层），导致正确与幻觉轨迹在后期层面的分离空间较小，削弱了探针的 margin 对比度。LLaVA 将集成延迟到 L24-31，创造了探针可利用的巨大分离间隙。

> **Q5:** 我们可以跨不同模型以零样本方式使用隐藏状态探针吗？
> **A:** 不能。本文推荐架构自适应的探测（每个家族不同的层选择和探针容量）。探针的跨家族泛化能力未被测试。

> **Q6:** 什么是 Counting Anomaly，为什么它很重要？
> **A:** Counting Anomaly 是指视觉编码器正确识别了图像中的 3 个独立聚类，但 LLM 解码器以 92% 的置信度输出"Four"的案例。这生动地展示了 Symbolic Detachment：视觉系统工作正常，但语言投影失败。Token 概率反映的是流畅度而非视觉基础。

---

## Citation Landscape

Connected Papers: https://www.connectedpapers.com/main/2606.17389

Key related works cited:
- **CLIP** (Radford et al., 2021): Foundation for vision-language alignment
- **LLaVA** (Liu et al., 2023): Visual instruction tuning, prefix-based VLM architecture
- **PaliGemma** (Beyer et al., 2024): Google's 3B vision-language model with SigLIP encoder
- **Qwen2-VL** (Wang et al., 2024): Alibaba's native multimodal architecture
- **Self-Consistency** (Wang et al., 2022): Agreement across sampled reasoning paths
- **Logit Lens** (Nostalgebraist, 2020): Hidden state projection to vocabulary
- **"Attention is not Explanation"** (Jain & Wallace, 2019): NLP interpretability debate
- **POPE** (Li et al., 2023b): Object hallucination evaluation benchmark
- **VIP/TVI** (Long et al., 2025): Visual Integration Point for LVLMs, measuring representational shift from images
- **"See but not believe"** (Liu et al., 2025): Correct localization without correct reasoning in VLMs
- **MME, SEED-Bench, MM-Vet:** Broader multimodal evaluation suites
