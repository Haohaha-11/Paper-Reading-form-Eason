# Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously

> **Authors**: Yiran Guan<sup>1\*</sup>, Liang Yin<sup>1\*</sup>, Dingkang Liang<sup>1</sup>, Jianzhong Ju<sup>2</sup>, Zhenbo Luo<sup>2</sup>, Jian Luan<sup>2</sup>, Yuliang Liu<sup>1</sup>, Xiang Bai<sup>1B</sup>
> **Affiliations**: <sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>MiLM Plus, Xiaomi Inc.
> **Venue**: arXiv 2026 (2603.12262)
> **Links**: [arXiv](https://arxiv.org/abs/2603.12262) | [GitHub](https://github.com/1ranGuan/VST) | [Connected Papers](https://www.connectedpapers.com/main/2603.12262)
> \* Equal contribution. <sup>B</sup> Corresponding author.

---

## 一句话总结

**VST 提出了一种面向流式视频理解的"边看边思考"范式，通过在视频播放间隙将 Chain-of-Thought 推理成本摊销到查询前阶段，在保持实时响应的同时实现 state-of-the-art 的在线精度，核心依靠两阶段后训练流程（VST-SFT + VST-RL）和基于知识图谱的数据合成引擎。**

---

## 核心贡献

1. **VST 范式**：一种新颖的流式视频理解范式，将主动、显式的 CoT 生成与连续视频流交织进行，将 LLM 后端从被动等待转变为在视频播放过程中主动间歇推理。这实现了推理成本的摊销式 test-time scaling，同时保持实时问答响应。

2. **两阶段后训练流程**：
   - **VST-SFT**：通过流式注意力掩码强制时序因果约束，将离线 VideoLLM 结构性地适配到因果流式推理，从 off-policy 示范中引导"边看边思考"的基础能力。
   - **VST-RL**：在多轮视频交互环境中通过 GRPO 进行端到端强化学习，reward 仅基于最终答案的正确性，从而鼓励模型生成对下游问答有用的中间流式思考。

3. **基于知识图谱的数据合成**：一套自动化 pipeline，构建时序一致的视频知识图谱，通过 DFS 采样多样化的多跳证据链，并利用 Gemini 3.0 flash 生成 100K 高质量的流式思维问答对，严格满足时序因果约束。

4. **全面的实验验证**：在线基准测试的 SOTA 结果（StreamingBench 79.5%，OVO-Bench 59.3%），离线场景的竞争力表现（VideoHolmes 41.9%，比 Video-R1 高 +5.4%），QA 延迟比 Video-R1 快 15.7 倍，且在 3B/7B/32B 三个模型规模上均有一致的增益。

---

## 📖 批读导航

| 章节 | 文件 | 核心内容 |
|---------|------|-------------|
| 00 - Abstract | [sections/00-abstract.md](sections/00-abstract.md) | 论文摘要，问题动机，关键数字 |
| 01 - Introduction | [sections/01-introduction.md](sections/01-introduction.md) | 范式对比（Fig.1），人类认知启发，核心贡献 |
| 02 - Related Work | [sections/02-related-work.md](sections/02-related-work.md) | 流式视频理解，VideoLLM Test-Time Scaling |
| 03 - Methodology | [sections/03-methodology.md](sections/03-methodology.md) | VST 范式，双记忆系统，VST-SFT/RL 训练，数据合成 |
| 04 - Experiments | [sections/04-experiments.md](sections/04-experiments.md) | 基准测试结果，消融实验，效率分析，案例研究 |
| 05 - Conclusion | [sections/05-conclusion.md](sections/05-conclusion.md) | 总结，局限性，未来工作，附录亮点 |

---

## 关键数字

### 性能亮点
| 指标 | VST-7B | 此前 SOTA | 提升 |
|--------|--------|---------------|------|
| StreamingBench | **79.5%** | 77.3% (StreamForest) | +2.2% |
| OVO-Bench Overall | **59.3%** | 57.9% (Streamo) | +1.4% |
| OVO-Bench Backward Tracing | **56.7%** | 52.0% (StreamForest) | +4.7% |
| VideoHolmes | **41.9%** | 36.5% (Video-R1) | +5.4% |
| LongVideoBench | **58.0%** | 58.0% (LongVILA-R1) | 并列 SOTA |

### 延迟对比 (VideoHolmes)
| 方法 | QA 延迟 | 对比 VST-7B |
|--------|-----------|-----------|
| Video-R1 w/CoT | 8.80s | **慢 15.7 倍** |
| Qwen2.5-VL w/CoT | 5.30s | **慢 9.5 倍** |
| Qwen2.5-VL direct | 0.54s | 相近 |
| **VST-7B** | **0.56s** | -- |

### 训练规模
| 阶段 | 数据 | 算力 |
|-------|------|---------|
| VST-SFT | 100K 流式思维 + 50K QA | 32 x 80GB GPU，1 epoch |
| VST-RL | 11K 问题（GRPO，N=8） | 32 x 80GB GPU，1 epoch |

### 消融：训练阶段贡献 (OVO-Bench)
| 配置 | Backward | Forward | Overall |
|---------------|----------|---------|---------|
| Base (Qwen2.5-VL-7B) | 47.5% | 41.9% | 50.5% |
| +仅有 VST-SFT | **56.7%** (+9.2) | 48.5% (+6.6) | 57.4% |
| +仅有 VST-RL | 49.3% (+1.8) | **54.6%** (+12.7) | 56.8% |
| +VST-SFT & VST-RL | **56.7%** | 54.0% | **59.3%** |

---

## 数据流：输入 → 中间表示 → 输出

```
输入: 连续视频流（无限长度，2 fps 采样）
  │
  ├─▶ [PySceneDetect] 场景分割成 N 个片段
  │
  ▼
中间表示 1: 知识图谱构建（仅用于数据合成）
  ├─ 滑动窗口实体提取（Gemini 3.0 flash）
  │   └─ 输出：每个片段的 (头实体, 关系, 尾实体) 三元组
  ├─ 实体库维护（FIFO，W 窗口大小）
  ├─ 噪声过滤（去重、字幕移除）
  ├─ NetworkX 图谱构建
  └─ DFS 证据链采样（实体重叠率 < 10%）
        │
        ▼
中间表示 2: 流式思维问答生成（VST-SFT 数据）
  ├─ 基于知识图谱的 CoT 推理链生成
  ├─ 从证据链合成多跳问答对
  └─ 5 重过滤（世界知识检查、格式对齐、逻辑一致性、重复检查、思维验证）
        │
        ▼
中间表示 3: VST-SFT 训练
  ├─ 输入：交织的 (video_clip, streaming_thought) 序列
  ├─ 流式注意力掩码：visual token 仅在滑动窗口 L 内可见
  ├─ 时序分段：长序列拆分为连续的段落
  └─ Next-token loss 仅作用于 {streaming thoughts} + {final answer}
        │
        ▼
中间表示 4: VST-RL 训练（Agentic Loop）
  ├─ Rollout：Policy 模型与流式环境交互
  │   └─ 生成轨迹：(z^1, z^2, ..., z^{K-1}, y)
  ├─ Reward：仅基于最终答案的正确性（可验证）
  ├─ GRPO：advantage 分配给轨迹中的全部 token
  └─ DAPO clipping + KL 惩罚（β=0.001）
        │
        ▼
输出（推理）:
  ┌──────────────────────────────────────────────────────┐
  │ 双记忆系统                                             │
  │                                                       │
  │ 短期视觉缓冲 [c^k]: 最近 L 个 visual token            │
  │   │                                                   │
  │   ▼                                                   │
  │ 长期文本记忆 [m^k]: 累积的思考                         │
  │   │  （容量超限时 FIFO 淘汰）                          │
  │   │                                                   │
  │ 流式思维 [z^k]: 自回归生成                             │
  │   每个片段: z^k ~ p(z | c^k, m^{k-1})                │
  │                                                       │
  │ 用户提问 [q] 时:                                       │
  │   最终回答: y ~ p(y | q, c^K, m^K)                   │
  │   QA 延迟: ~0.56s（思考已摊销）                        │
  └──────────────────────────────────────────────────────┘
```

### 概率分解（核心公式）
```
p(y | q, V) = p(y | q, c^K, m^K)           ← 直接回答（查询后）
              × ∏_{k=1}^{K-1} p(z^k | c^k, m^{k-1})  ← 流式思维（查询前）
```

---

## 优缺点与还能做什么

### 优点

1. **延迟-质量 Pareto 改进**：VST 相比离线 CoT 方法同时实现了更高的准确率和更低的延迟（例如 VideoHolmes 上比 Video-R1 高 +5.4%，同时响应快 15.7 倍）。这种"质量更高、速度更快"的双赢局面在系统优化中非常罕见——通常准确率和延迟是此消彼长的 trade-off。

2. **有理论原则的双记忆架构**：短期视觉缓冲（原始感知）+ 长期文本记忆（语义压缩）的设计优雅地解决了无限长视频流在固定上下文窗口约束下的记忆问题。FIFO 淘汰机制的文本记忆自然地遗忘旧事件，模仿了人类的工作记忆机制。

3. **训练-推理对齐**：流式注意力掩码（Eq.3）确保模型在训练时学习的因果约束与推理时完全一致，消除了训练和推理之间 attention pattern 的不匹配。

4. **跨模型规模可扩展**：在 3B/7B/32B 三个规模上均取得一致的性能增益（StreamingBench +7.7% 到 +9.2%），证明 VST 范式具有参数可扩展性，而非仅在 7B 规模有效的"特技"。

5. **认知科学理论支撑强**："边看边思考"的设计以神经耦合（neural coupling）研究 [16, 36] 为理论基础，使该范式在纯工程考量之外，还具有认知科学层面的合理性。

6. **数据合成具有因果保证**：基于知识图谱的 pipeline 确保生成的流式思维永远不会"泄漏"未来信息，这种时序因果性保证是朴素的 video-to-QA 生成方法无法做到的。

### 局限 / 风险

1. **额外的 Token 消耗**：虽然 QA 延迟低，但流式思维本身消耗了大量 LLM token 预算（每个视频片段约 4 次思考步骤）。论文承认了这一问题，并将 latent reasoning 作为未来工作方向。

2. **反事实推理是弱项**：VST 在 StreamingBench 的 CT（反事实思维）任务上仅得 47.3%，明显低于 TimeChatOnline 的 58.0%。因果性（仅前向）的流式思维在需要考量替代事件序列的反事实推理方面可能存在本质局限。

3. **思维质量缺乏直接监督**：VST-RL 仅对最终答案做 reward，依赖优化过程隐式地发现有用的思维模式。这无法保证每一步流式思维都是事实准确的——幻觉性思维可能被存入文本记忆并传播错误。

4. **单次查询假设**：当前范式假设每个视频流只有一次用户查询。在多用户或多轮对话场景下，流式思维需要适配多样化的下游问题，可能需要更通用或条件于具体查询的思维。

5. **依赖场景检测质量**：数据合成 pipeline 依赖 PySceneDetect 进行片段分割。糟糕的场景边界可能导致连贯事件被拆分到不同片段中，降低流式思维生成的质量。

6. **与视觉记忆方法的对比有限**：VST 的文本记忆未与视觉记忆方法（StreamForest、TimeChatOnline）进行直接对比或组合测试。论文将这一点列为未来工作，但缺少组合系统意味着"文本 + 视觉记忆是否具有协同效应"的问题尚未回答。

---

## 引用全景

**Connected Papers**： [https://www.connectedpapers.com/main/2603.12262](https://www.connectedpapers.com/main/2603.12262)

### 核心相关工作

**流式视频理解**：
- **VideoLLM-online** (Chen et al., CVPR 2024): Early work on online VideoLLM for streaming video, focused on efficient visual token processing.
- **Dispider** (Qian et al., CVPR 2025): Disentangled perception, decision, and reaction for active real-time interaction via external memory retrieval.
- **TimeChat-Online** (Yao et al., MM 2025): Shows 80% visual tokens are redundant in streaming, uses compression-based streaming.
- **StreamForest** (Zeng et al., NeurIPS 2025): Persistent event memory for efficient online video understanding, previous SOTA on StreamingBench (77.3%).
- **Streamo** (Xia et al., CVPR 2026): Streaming video instruction tuning method, previous SOTA on OVO-Bench (57.9%).
- **Flash-VStream** (Zhang et al., ICCV 2025): Real-time understanding for long video streams with efficient token management.
- **LiveVLM** (Ning et al., 2025): Streaming-oriented KV cache and retrieval for efficient online video understanding.
- **StreamMem** (Yang et al., 2025): Query-agnostic KV cache memory for streaming video.

**VideoLLM Test-Time Scaling / 推理**：
- **Video-R1** (Feng et al., NeurIPS 2025): R1-style RL for video reasoning, the main CoT baseline VST compares against on latency.
- **LongVILA-R1** (Chen et al., NeurIPS 2025): Scaling RL to long videos with paralleled encoding strategy (adopted by VST).
- **REVISOR** (Li et al., CVPR 2026): Multimodal introspective reasoning for long-form video, competitive offline baseline.
- **VideoEspresso** (Han et al., CVPR 2025): Large-scale CoT dataset via core frame selection for fine-grained video reasoning.
- **Video-RFT** (Wang et al., NeurIPS 2025): Reinforced fine-tuning for video reasoning capabilities.
- **ThinkOmni** (Guan et al., ICLR 2026): Lifting textual reasoning to omni-modal scenarios (same first author).
- **StreamingThinker** (Tong et al., 2025): Text-only "think while reading" for LLMs, conceptual precursor to VST in text domain.

**基础模型与基准测试**：
- **Qwen2.5-VL** (Bai et al., 2025): Base VideoLLM used by VST.
- **StreamingBench** (Lin et al., 2024): Online streaming video understanding benchmark.
- **OVO-Bench** (Niu et al., CVPR 2025): Real-world online video understanding benchmark with Backward/Forward task decomposition.
- **VideoHolmes** (Cheng et al., 2025): Complex video reasoning benchmark.
- **VideoMME** (Fu et al., CVPR 2025): Comprehensive offline video understanding benchmark.
- **LongVideoBench** (Wu et al., NeurIPS 2024): Long-context interleaved video-language benchmark.

---

## 阅读 Q&A 记录

> **Q1: VST 的"边看边思考"与在视频播放期间定期运行 CoT 有何本质区别？**
>
> A: 关键区别在于训练。VST-SFT 在严格的时序因果约束（流式注意力掩码（Eq.3））下训练模型生成流式思维，因此模型在思考时永远不会"看到"未来帧。若朴素地对离线模型做 prompt 以定期生成 CoT，会因为离线模型使用全局注意力（global attention）训练而泄漏未来信息。此外，VST-RL 提供了端到端优化，使得思维专为下游 QA 的有用性而优化，而不仅仅是生成听起来合理的文本。

> **Q2: 为什么 VST-SFT 主要提升 Backward 记忆（+9.2%），而 VST-RL 主要提升 Forward 预测（+12.7%）？**
>
> A: VST-SFT 教会了模型*记录什么*到文本记忆中——内容摘要、事件记录、实体追踪——这些直接有利于向后回溯检索。VST-RL 的 reward 基于最终答案的正确性，由于许多问题需要基于过去的观察来预测或推理未来事件，RL 优化自然地鼓励模型生成有助于前向推断的思维。两个阶段是互补关系而非冗余关系。

> **Q3: 比 Video-R1 快 15.7 倍的延迟对比是否公平？**
>
> A: 延迟测量（QA latency = 从提问到回答的时间）是公平的，因为它测量的是用户实际体验到的。总计算量（流式思维 + 最终回答）可能与 Video-R1 的查询后 CoT 相当，但 VST 将成本提前到视频播放的空闲时段。这类似于视频流媒体在播放时缓冲内容以避免卡顿——总传输数据量相同，但用户体验截然不同。

> **Q4: VST 能否处理多轮对话或同一视频流上的多个查询？**
>
> A: 当前公式假设每个视频流只有一次查询（Eq.1），但文本记忆机制本身是与查询无关的。原则上，累积的流式思维如果能捕捉到足够通用的视频推理，就可以服务于多个查询。然而，论文未对此进行验证，需要进一步研究，特别是围绕"在没有特定问题导向的情况下生成的流式思维是否对多样化的下游查询足够有用"这一问题。

> **Q5: 知识图谱数据合成 pipeline 如何防止信息泄漏？**
>
> A: Pipeline 以增量方式构建知识图谱——每当新的片段到来时（按时序顺序），Gemini 仅从该片段（加上一个小的重叠窗口 W-1）中提取实体和关系。这意味着在为第 k 个片段生成流式思维时，可用知识严格受限于到该时间点为止已观察到的内容。证据链随后从完整图谱中采样，但 QA 合成 prompt 强制每个中间推理步骤仅引用该时间点可用的信息。

> **Q6: 如果流式思维生成了错误或幻觉内容会怎样？**
>
> A: 这是一个真实存在的风险。与最终答案有 ground truth 用于 RL reward 计算不同，单个流式思维没有直接的监督信号。文本记忆中的幻觉内容可能传播到最终答案中。VST-RL 对此提供间接压力（无用/有害的思维导致错误答案，从而降低 reward），但并没有显式的幻觉检测机制。这是"仅对最终答案做 reward"方法的固有局限。

> **Q7: 为什么 VST 在 StreamingBench 的 CT 任务（Counterfactual Thinking，47.3%）上不如 TimeChatOnline（58.0%）？**
>
> A: 反事实推理（"如果 X 以不同方式发生会怎样"）需要对替代事件序列进行推理，这本质上需要一种"后见之明"的全局视角，与 VST 严格的因果前向思维范式存在冲突。TimeChatOnline 尽管缺乏显式推理，但可能受益于所有 visual token 的可用性（仅做压缩）而非 VST 依赖的更为抽象的文本记忆。这暗示了视觉记忆和文本记忆方法之间存在潜在的互补性。

---

## 图表索引

| 图表 | 文件 | 描述 |
|--------|------|-------------|
| Fig. 1 | [images/21a7b95c...747a.jpg](images/21a7b95c3be0763213271697c0fed7b2418600a3c655a42ee43e5723c032747a.jpg) | Benchmark 结果与范式对比 (a-d) |
| Fig. 2 | [images/be382a30c...90e7.jpg](images/be382a30cb04d65a2fe4e291fdae034f6992b6b0cf2eab532ba363a16c6a90e7.jpg) | VST pipeline 与双记忆系统 |
| Fig. 3 | [images/6d5d820fb...3ae2.jpg](images/6d5d820fb427f8f6743c58065f984a4b81fb1573ff73c0f106a8afaea49a3ae2.jpg) | 训练 pipeline：VST-SFT + VST-RL |
| Fig. 4 | [images/c1db74b26...ce1.jpg](images/c1db74b26259d4b40ea898936c4410b7d4a94da0e3c331bdbad2839913c72ce1.jpg) | 数据合成 pipeline（知识图谱 → QA） |
| Fig. 5a | [images/47939b2c3...6bd7.jpg](images/47939b2c36734b9a28ac0e72a1d3f2c12dabaa449c4e585f1c44fc1646dd6bd7.jpg) | 消融实验：最大思考次数 vs 准确率 |
| Fig. 5b | [images/e3d59d79d...e3fa.jpg](images/e3d59d79d63338f100b4c24e012c77788cfa020819a6915358a43dbc2154e3fa.jpg) | 消融实验：最大思考次数（续） |
| Fig. 6 | [images/19df46057...27ab.jpg](images/19df46057ddbbb359c016f60107f222f2e65916f3063280f7bb11ae9e62d27ab.jpg) | 案例研究：VST vs Video-R1 在 VideoHolmes 上 |
| Fig. 7 | [images/fd22e5567...a44d.jpg](images/fd22e55679939455aa35996dc302cf95f4ec1651398efb4e08840bbf4318a44d.jpg) | 流式推理 pipeline（附录） |

---

*批读完成于 2026-06-22。所有原文内容保持原样；批注以 > 💡 标记。*
