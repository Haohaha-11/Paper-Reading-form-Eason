# 4. Experiments

[← 返回 README](../README.md)

## 📌 预览

在 SafeBench 和 HADES 两个数据集上，对 7 个 LVLM（4 闭源+3 开源）进行全面评估。PIS 在几乎所有设置下取得最优 ASR，并通过消融和可视化分析验证各组件的有效性。

---

## 4.1. Experimental Setup

**Datasets.** SafeBench (350 samples, 7 categories) and HADES (750 samples, 5 scenarios: Violence, Finance, Privacy, Self-Harm, Animals).

**Target Models.** 4 closed-source: GPT-4V, GPT-4o, Gemini 2.5 Flash, Qwen3-VL-Plus; 3 open-source: Qwen3-VL-8B, InternVL3-9B, GLM-4.5V (106B, API).

**Comparison Methods.** FigStep, MM-SafetyBench, HADES, SI-Attack, Vanilla baseline. Supplementary: FigSteppro, HIMRD.

**Evaluation Metrics.** ASR (Attack Success Rate) and Toxic Score (1-5 scale, Deepseek-V3 as judge). Attack successful if score $\geq S_{harm} = 3$.

$$\text{ASR} = \frac{\sum \mathbb{I}\{R(t, M_{\theta}(x_t, x_v)) \geq S_{harm}\}}{N_{total}}$$

**Implementation.** Single RTX 4090 GPU. PROMPTER/PAINTER agents use GPT-3.5-Turbo; GUIDER uses GPT-4.1; Image generation: Qwen-Image-Plus.

## 4.2. Main Results

**Closed-Source Models (Tab. 1).** On SafeBench, PIS achieves 60.85% (GPT-4o), 71.14% (Qwen3-VL-Plus), 58.57% (Gemini 2.5 Flash). On HADES: 48.67% (GPT-4o), 62.53% (Qwen3-VL-Plus), 58.27% (Gemini 2.5 Flash).

**Open-Source Models (Tab. 1).** On SafeBench: GLM-4.5V 66.00%, Qwen3-VL-8B 63.14%, InternVL3-9B 67.71%. On HADES: GLM-4.5V 65.73%, Qwen3-VL-8B 53.60%, InternVL3-9B 62.00%.

> 💡 **消融解读 — ASR 结果分析**: 最显著的提升在 GPT-4o 上——PIS 60.85% vs HADES 34.28%（+26.57%）。Qwen3-VL-Plus 上的 71.14% 说明 PIS 对中文优化模型也有效。Gemini 2.5 Flash 上 58.57% 虽然相对较低，但仍比第二名 FigStep 高 14.17%。

> 💡 **Figure 4 & 5 批读**: 跨类别 Toxic Score 分析显示 PIS 在所有 5 个 HADES 类别上均取得最高平均毒性分数，证明 PIS 不依赖特定攻击类别。

## 4.3. Universality Analysis

Tab. 2 compares PIS with FigSteppro and HIMRD. Key finding: HIMRD achieves 81.43%/81.95% on open-source InternVL3-9B/GLM-4.5V but 0% on GPT-4V/GPT-4o. Removing mandatory suffix: HIMRD ASR on GPT-4o rises to 26.57% but drops to 21.71% on GLM-4.5V (-60.24%). PIS achieves consistent 50.28-67.71% across all models.

> 💡 **消融解读 — Mandatory Suffix 双刃剑**: 这是本文最有价值的发现。强制后缀在开源模型上帮助格式引导（+60%），但在闭源模型上被安全检测识别为攻击特征（降至 0%）。PIS 不依赖强制后缀，因此具有跨架构通用性。

## 4.4. Ablation Study (SafeBench-Small, Tab. 3)

| 变体 | GPT-4o ASR | 关键洞察 |
|------|------------|----------|
| HADES (baseline) | 31.88% | 原方法 |
| HADES (+PAINTER) | 55.71% | PAINTER 贡献 +23.83% |
| HADES_harm (+PROMPTER) | 31.43% | PROMPTER 贡献 +11.43% |
| PIS (full) | 61.43% | 三队协同 +29.55% |

> 💡 **消融解读**: PAINTER 的单独贡献 (+23.83%) 大于 PROMPTER (+11.43%)，说明图像自然性对绕过视觉安全检测更重要。但 GUIDER 的协同效应最大（full PIS vs HADES_harm+PROMPTER+PAINTER = +24.29%），验证了跨模态对齐的关键作用。

**Supervisor Impact (Fig. 7).** 大多数样本在首轮通过 Supervisor，最多需 5 轮。有 Supervisor 的 ASR 显著高于无 Supervisor 版本。

## 4.5. Visual Analysis (Fig. 6)

Representation distribution visualization on Qwen3-VL-8B: (a) PROMPTER alone shifts representations toward harmful anchor; (b) +PAINTER further clusters toward harmful; (c) Full PIS converges completely into harmful distribution.

> 💡 **Figure 6 批读**: 这是对 PIS 组件效果的直观验证——三个阶段的表示分布逐步向有害区域移动，证明每个 Agent 团队都有增量贡献。

## 🔖 Section 总结

| 维度 | 关键结果 |
|------|----------|
| 闭源模型最优 ASR | Qwen3-VL-Plus 71.14% (SafeBench) |
| GPT-4o 提升幅度 | +26.57% vs HADES |
| PAINTER 贡献 | +23.83% ASR |
| Mandatory Suffix 效应 | 开源+60% / 闭源降至 0% |
| Supervisor 作用 | 最多 5 轮迭代，ASR 显著提升 |
