# Imagine Before You Predict: Interleaved Latent Visual Reasoning for Video Event Prediction (FUTURE-L1)

## Paper Metadata

| 项目 | 内容 |
|------|------|
| **Title** | Imagine Before You Predict: Interleaved Latent Visual Reasoning for Video Event Prediction |
| **Authors** | Tianxiang Jiang\*, Linquan Wu\*, Sheng Xia, Songze Li, Ziang Yan, Haoyu Yang, Yu Qiao, Yi Wang\* |
| **Affiliations** | USTC, Shanghai AI Laboratory, CityU HK, Nanjing University, Fudan University, Zhejiang University, UESTC |
| **Venue** | arXiv 2026 |
| **Code** | https://github.com/OpenGVLab/Future-L1 |
| **Backbone** | Qwen3-VL-8B-Instruct |

## One-Sentence Summary

FUTURE-L1 提出了一种**交错式潜空间视觉推理 (Interleaved Latent Visual Reasoning)** 框架，在自回归解码过程中交替生成文本 token 和连续潜视觉 span，并通过 visual-gain 数据筛选 (FUTURE-L1-50K) 和潜空间感知 RL (LA-DAPO) 进行两阶段训练，在 FutureBench 和 TwiFF-Bench 上大幅刷新 SOTA——核心思想是：**对未来动态视觉结构进行推理时，不应将每一个中间步骤都转换为文本，而应在潜空间中保留连续的动态视觉语义**。

## Core Contributions

1. **Visual-Gain 数据筛选与 FUTURE-L1-50K 构建** (Section 3.2): 提出 visual-gain 指标 (pv - pt)，从 TwiFF-2.7M 中筛选出未来视觉线索对预测有实质性帮助的 50K 高质量样本，用于监督潜空间未来视觉推理的冷启动。

2. **交错式潜空间视觉推理框架** (Section 3.1): 首次在 VEP 任务中引入交替文本-潜视觉的自回归解码范式，通过 `<|latent_start|>` / `<|latent|>` / `<|latent_end|>` 三个特殊 token 控制潜视觉 span 的边界，让语言组织推理结构、潜空间保留动态视觉结构。

3. **LA-DAPO: 潜空间感知的 RL 优化** (Section 3.3): 在 DAPO 基础上引入两类轨迹级潜空间奖励——outcome-contrastive reward (Rctr, 跨 rollout 对齐正负例潜轨迹) 和 temporal-diversity reward (Rdiv, 惩罚相邻 span 的视觉思维重复)，无需 RL 阶段的中间帧标注即可优化潜轨迹。

4. **SOTA 实验验证** (Section 4): FutureBench 上 Qwen3-VL-8B 从 61.0 提升至 85.4 (+24.4)，超越前 SOTA Video-CoE 10.4 分；TwiFF-Bench 平均分从 2.44 提升至 3.04；尤其在 3-Hop 和 Interp. 等长链/非连续未来事件推理上增益最大。

## Section Navigation

| 章节 | 文件 | 核心内容 |
|------|------|---------|
| Abstract & Figure 1 | [00-abstract.md](sections/00-abstract.md) | 论文概述、潜视觉推理动机 |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | VEP 难点、文本推理瓶颈、三阶段总结 |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) | MLLM、潜空间推理、VEP 三条脉络 |
| 3. Method | [03-method.md](sections/03-method.md) | 交错推理、FUTURE-L1-50K 构建、LA-DAPO |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) | 主实验、消融、潜推理分析 |
| 5. Conclusion | [05-conclusion.md](sections/05-conclusion.md) | 总结与关键附录要点 |

## Key Numbers

| 指标 | 数值 |
|------|------|
| Backbone | Qwen3-VL-8B-Instruct |
| SFT 数据量 | FUTURE-L1-50K (从 TwiFF-2.7M 筛选) |
| FutureBench 零样本基线 | 61.0 (Qwen3-VL-8B) |
| FutureBench SFT (文本-only) | 65.0 |
| FutureBench SFT (交错潜空间) | 73.2 (+8.2 over text-only) |
| FutureBench RL (FUTURE-L1-RL) | 85.4 (+24.4 over backbone) |
| FutureBench 前 SOTA (Video-CoE) | 75.0 |
| TwiFF-Bench 零样本 | 2.44 |
| TwiFF-Bench RL | 3.04 |
| 潜在 span 长度上限 Lmax | 4 |
| 潜空间 MSE 权重 λ | 0.1 |
| RL 奖励系数 (Rctr / Rdiv) | λc=0.2, λd=0.1 |
| GPU | 8x NVIDIA H200 |

## Data Flow: Input → Intermediate → Output

```
| 阶段 | 描述 |
|------|------|
| 1. FUTURE-L1 Data Flow |  |
| 2. [Stage 1 | Data Curation] |
| 3. TwiFF-2.7M → Visual-Gain Probe (pv - pt) → Top 50K |  |
| 4. → FUTURE-L1-50K (interleaved text-latent format) |  |
| 5. [Stage 2 | SFT] |
| 6. Input | Observed Video V + Question q |
| 7. Training Format |  |
| 8. <reason>[Text CoT 0]</reason> |  |
| 9. <|latent_start|>[Latent States 1]<|latent_end|> |  |
| 10. <reason>[Text CoT 1]</reason> ... |  |
| 11. <answer>[Prediction]</answer> |  |
| 12. Loss | L_CE (text tokens) + λ * L_Latent (MSE to future-frame |
| 13. visual embeddings) |  |
| 14. [Stage 3 | LA-DAPO RL] |
| 15. Group of G=8 rollouts → |  |
| 16. Rewards | λa*Racc + λf*Rfmt + λc*Rctr + λd*Rdiv |
| 17. Rctr | hardest-positive InfoNCE across correct/incorrect |
| 18. Rdiv | -mean(cos²(b_m, b_{m+1})) across latent spans |
| 19. [Inference] |  |
| 20. Autoregressive decoding alternates |  |
| 21. text tokens → <|latent_start|> → h_t (fed back to input) |  |
| 22. → ... → <|latent_end|> → text tokens → final answer |  |

┘
```

## Pros/Cons & Future Work

### Strengths

1. **模态匹配**: 潜空间连续表示天然适合保留动态视觉语义（运动、几何、交互），避免了文本化造成的信息损失——这是 VEP 任务的根本性改进方向。

2. **两阶段训练设计合理**: SFT 阶段的未来帧嵌入对齐提供了必要的冷启动（无此则模型不会使用潜空间），RL 阶段的潜空间感知奖励进一步优化采样轨迹——两阶段分工清晰，互为补充。

3. **数据筛选方法可复用**: visual-gain 指标 (pv - pt) 是一种通用的"视觉线索效用"度量，不限于 VEP 任务，可推广到任何需要评估视觉信息增量贡献的场景。

4. **推理效率高**: 潜空间计算不产生显式 token 生成开销，FUTURE-L1-RL 仅需 195.3 tokens / 0.91s，显著优于文本重推理方法（Video-R1: 398.5 tokens, Video-o3: 348.6 tokens）。

5. **自适应潜空间分配**: 模型根据推理难度动态调整潜 span 数量（1-Hop: 1.79 spans → 3-Hop: 2.52 spans），而非固定模板。

### Weaknesses / Limitations

1. **潜空间可控性有限**: 潜状态是不可解释的连续向量，无法像文本 CoT 那样直接检查中间推理是否正确，调试和可信度验证困难。

2. **依赖未来帧标注**: SFT 阶段需要未来推理帧的 visual embedding 作为监督信号，这限制了训练数据的来源（需要视频中未来的帧）。虽然 RL 阶段不需要，但冷启动仍受此约束。

3. **仅验证在单一 backbone**: 所有实验基于 Qwen3-VL-8B，虽然 8B 规模已有代表性，但未在更大模型（如 30B/72B）或其他架构（如 InternVL、LLaVA）上验证泛化性。

4. **失败案例分析 (Figure 18)**: 当场景需要细粒度的视觉事件身份保持时，潜轨迹可能漂移到"看似合理"但具体细节错误的通用延续——说明仅仅调用潜 space 不够，还需要更精细的视觉约束。

5. **潜空间奖励设计的敏感性**: 两阶段训练需要仔细调节多个超参数（λ, Lmax, λc, λd），且不同任务（FutureBench vs TwiFF）需要不同的 RL 数据量（2K vs 20K）。

### Future Work

1. 将潜空间视觉推理扩展至更多 backbone 架构和更大规模模型
2. 探索更精细的潜状态约束机制（如细粒度事件身份保持）以解决 failure case 中的漂移问题
3. 将 visual-gain 数据筛选方法推广到其他需要视觉想象的任务
4. 探索纯 RL（跳过 SFT 冷启动）或更弱的监督信号来训练潜空间推理能力
5. 研究潜空间状态的可解释性方法，使中间视觉思维也能被人类理解

## Reading Q&A Record

| # | 问题 | 答案位置 | 解答 |
|---|------|---------|------|
| 1 | 为什么文本化中间推理对 VEP 特别不利？ | Section 1, Figure 1 | VEP 需要推理未观察到的动态视觉状态（物体移动、实体交互、场景演变）。一旦视觉证据被转换为文字，细粒度的运动、几何、相对位置和交互信息就会丢失。文本推理可能"听起来合理"但偏离视觉语义，尤其是当正确答案依赖微妙的未来动态时。这本质上是一个 representation bottleneck：文本是离散的、符号化的，而未来视觉动态是连续的、空间化的。 |
| 2 | FUTURE-L1 的潜空间 span 和 COCONUT 的 latent thought 有什么区别？ | Section 2, Section 3.1 | COCONUT 将显式 CoT 压缩为一组连续的 latent thought tokens；FUTURE-L1 的潜空间 span 不同：(1) 是交错式而非全部替换——文本仍组织推理结构；(2) 由特殊 token 显式控制边界；(3) 通过未来帧嵌入进行视觉对齐（SFT），而非仅用语言建模目标；(4) 可包含多个 span 支持多步视觉更新。 |
| 3 | 为什么需要 visual-gain 筛选？随机 50K 不行吗？ | Section 3.2, Table 6 | 随机 50K 也能提升性能（68.4 vs 61.0），但比 visual-gain 筛选的 50K (73.2) 低 4.8 分。原因：并非所有 TwiFF 样本对未来视觉推理都有同等价值——简单样本和模糊样本会稀释训练信号。visual-gain 筛选确保训练样本中的未来视觉线索确实提供了可测量的预测效用。 |
| 4 | LA-DAPO 的两类潜空间奖励分别解决什么问题？ | Section 3.3, Table 4 | Rctr（outcome-contrastive）：SFT 后每个潜状态 teacher-forced 匹配到特定未来帧，但采样轨迹中的潜状态没有直接的预测正确/错误反馈。Rctr 通过 InfoNCE 使正确 rollout 的潜轨迹聚拢、错误 rollout 的推开，在 sequence level 给予潜状态正确性信号。Rdiv（temporal-diversity）：SFT 中的 frame-distinct 监督在 RL 阶段不再存在，模型可能坍缩为重复相同的视觉思维。Rdiv 惩罚相邻 span 余弦相似度，保持时间维度的多样性。 |
| 5 | 为什么潜空间 span 越长反而不一定好？ | Section 4.2, Table 3 | Lmax 从 4 增加到 64 时，性能从 73.2 降到 67.4。可能的解释：(1) 过长 span 中后续潜状态缺乏对应的未来帧监督；(2) 过多潜 tokens 增加了优化维度，信号被稀释；(3) 潜状态可能开始编码与未来预测无关的信息。这支持了"短而精"的潜 span 设计原则。 |
| 6 | FUTURE-L1 和 LVR/Monet/SwimBird 等静态潜视觉推理方法的本质区别是什么？ | Section 2, Table 1 | LVR/Monet 等方法的潜视觉思维锚定在**静态图像**（辅助图像、草图、已有场景），而 VEP 需要推理**尚未观察到的动态未来帧**。这些方法直接迁移到 VEP 效果很差：LVR 仅 21.0（甚至常因密集视频 token 输入导致崩溃），Monet 47.9。FUTURE-L1 的关键设计——用未来帧嵌入而非静态图像 embedding 监督潜状态——正是针对这一差异。 |
| 7 | 推理效率提升是怎么实现的？ | Section 4.3, Table 7 | 文本 CoT 方法需要逐 token 生成完整的文本推理链（Video-R1: 398.5 tokens），而潜空间计算不产生显式 token——模型在潜状态下进行内部计算，仅文本部分产生 tokens。这直接将 token 数压缩到 195.3（不到一半），同时每个 token 仍被有效利用（accuracy per second: 93.8 vs Video-R1 的 19.3）。 |

## Citation Landscape

### Reference Grouping by Topic

**MLLM Backbones & Architectures**:
- Qwen2.5/3-VL [Bai et al., 2025b,a], InternVL3/3.5 [Zhu et al., 2025; Wang et al., 2025d], GLM-4.1V/5V [Team et al., 2025; Hong et al., 2026], LLaVA-OneVision-2 [An et al., 2026], MiMo-VL [Xiaomi, 2025], Kimi K2.5 [Team et al., 2026]

**Video Understanding & Benchmarks**:
- MVBench [Li et al., 2024], Video-MME [Fu et al., 2024], Video-MME-v2 [Fu et al., 2026], TempCompass [Liu et al., 2024], ExpVid [Xu et al., 2025], River [Shi et al., 2026]
- InternVideo2 [Wang et al., 2024], Flexible Video Models [Wang et al., 2025a]

**Latent Space Reasoning (LLM)**:
- COCONUT [Hao et al., 2024], CODI [Shen et al., 2025], SIM-CoT [Wei et al., 2025], The Latent Space Survey [Yu et al., 2026b]

**Latent Visual Reasoning (MLLM)**:
- LVR [Li et al., 2025a], Mirage/Machine Mental Imagery [Yang et al., 2025b], Monet [Wang et al., 2025c], SkiLa [Tong et al., 2025], SwimBird [Tong et al., 2026], LaViT [Wu et al., 2026], OneVL [Lu et al., 2026a]
- Chain-of-Visual-Thought [Qin et al., 2025], Multimodal CoCT [Pham and Ngo, 2025]
- Hybrid Latent Reasoning with DePO [Cheng et al., 2026]

**Video Event Prediction (VEP)**:
- NEP [Wang et al., 2025b], Video-CoE [Su et al., 2026], TwiFF (Think with Future Frames) [Liu et al., 2026a]
- TEMPURA [Cheng et al., 2025a], Video-as-Answer [Cheng et al., 2025b]
- Action Anticipation [Lan et al., 2014; Gammulle et al., 2019]
- Future Frame Prediction [Ranzato et al., 2014; Vondrick et al., 2016b]
- Next-Event Prediction [Lei et al., 2020; Jiang et al., 2025; Liang et al., 2025; Su et al., 2025]

**Video Reasoning (Text-CoT)**:
- Video-RFT [Wang et al., 2026], Video-R1 [Feng et al., 2026], VideoAuto-R1 [Liu et al., 2026b], Video-o3 [Zeng et al., 2026], VideoChat-R1 [Li et al., 2025d]
- VideoEspresso [Han et al., 2025], DeepEyes [Zheng et al., 2025b]
- Thinking with Visual Primitives [Lu et al., 2026b]

**RL Methods**:
- DAPO [Yu et al., 2026a], Easy-R1 [Zheng et al., 2025a], GRPO

**Evaluation Tools**:
- lmms-eval [Zhang et al., 2024a]

---

*Batch reading created on 2026-06-24*
