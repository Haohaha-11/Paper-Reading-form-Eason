[← 返回 README](../README.md)

# 04 - Experiments

📌 **Preview**: 在 5 个 benchmark 上的全面评测（StreamingBench, OVO-Bench, VideoMME, LongVideoBench, VideoHolmes），展示 VST 在 online 和 offline 场景下的性能。包含消融实验（训练阶段、thinking times、模型规模）、效率分析（延迟对比）和案例分析。

---

## 3 Experiment

### 3.1 Implementation Details

We adopt Qwen2.5-VL [1] as our base offline VideoLLM, processing input videos at 2 fps. Both VST-SFT and VST-RL (7B model) training stages are conducted on **32 x 80GB VRAM GPUs**, utilizing the datasets detailed in Sec. 2.3. The visual encoder and projection layer are frozen throughout the entire training process. For VST-SFT, each training sample follows a 128-second time limit, and overlong raw videos are segmented into clips following Eq. (4). For VST-RL, we employ verl [34] with vLLM [19] and FSDP [57] backend. We configure the rollout batch size to 256 with a group size of N = 8, and define the reward function based on the correctness of the final answer. Additionally, following LongVILA-R1 [4], we leverage the paralleled encoding strategy during rollout to pre-compute video embeddings.

During testing, following StreamingForest [51], we cap each inference step (including streaming-think and the final answer) at **8,192 video tokens** and limit the **max thinking times to 4** for efficient evaluation. We conduct all evaluations using the lmms-eval framework [54].

> 💡 **实现细节批读**:
> - 基座模型：Qwen2.5-VL（2 fps 采样）
> - 视觉端冻结：视觉编码器和投影层全程不训练，确保视觉表征的稳定性
> - RL 框架：verl + vLLM + FSDP，rollout batch size=256, group size=8（GRPO 标准配置）
> - 推理配置：每步最多 8192 visual tokens，最多 4 次 thinking（基于消融实验优化）

### 3.2 Benchmarks

To demonstrate the effectiveness of our method, we conducted a comprehensive evaluation across five video understanding benchmarks:

- **StreamingBench** [26]: Online video understanding, focusing on real-time reasoning and temporal awareness
- **OVO-Bench** [29]: Online video understanding, with Backward Tracing and Forward Prediction tasks
- **VideoMME** [10]: Comprehensive offline benchmark covering diverse domains and varying video durations
- **LongVideoBench** [42]: Long-form video understanding capabilities
- **VideoHolmes** [5]: Logical reasoning within video content

### 3.3 Online Video Benchmark Results

**Table 1**: Comparison of offline and online VideoLLMs on **StreamingBench** Real-Time understanding tasks.

| Model | Venue | OP | CR | CS | ATP | EU | TR | PR | SU | ACP | CT | Overall |
|-------|-------|----|----|----|----|----|----|----|----|----|----|----|
| **Proprietary Models** | | | | | | | | | | | | |
| Gemini 1.5 pro [37] | - | 79.0 | 80.5 | 83.5 | 79.7 | 80.0 | 84.7 | 77.8 | 64.2 | 72.0 | 48.7 | 75.7 |
| GPT-4o [30] | - | 77.1 | 80.5 | 83.9 | 76.5 | 70.2 | 83.8 | 66.7 | 62.2 | 69.1 | 49.2 | 73.3 |
| **Open-source Offline Models** | | | | | | | | | | | | |
| VILA-1.5-8B [24] | CVPR'24 | 53.7 | 71.0 | 56.9 | 53.4 | 53.9 | 54.6 | 48.8 | 54.7 | 50.1 | 17.6 | 52.3 |
| LongVA-7B [55] | TMLR'25 | 49.2 | 63.3 | 70.9 | 62.7 | 59.5 | 61.1 | 53.7 | 54.7 | 34.7 | 60.0 | |
| MiniCPM-v2.6-7B [18] | COLM'24 | 71.9 | 71.1 | 77.9 | 75.8 | 64.6 | 65.7 | 70.4 | 56.1 | 62.3 | 53.4 | 67.4 |
| LLaVA-OV-7B [20] | TMLR'25 | 80.4 | 74.2 | 76.0 | 80.7 | 72.7 | 71.7 | 67.6 | 65.5 | 65.7 | 45.1 | 71.1 |
| Qwen2.5-VL-7B [1] | - | 78.3 | 80.5 | 78.9 | 80.5 | 76.7 | 78.5 | 79.6 | 63.4 | 66.2 | 53.2 | 73.7 |
| **Open-source Online Models** | | | | | | | | | | | | |
| Flash-VStream-7B [53] | ICCV'25 | 25.9 | 43.6 | 24.9 | 23.9 | 27.3 | 13.1 | 18.5 | 25.2 | 23.9 | 48.7 | 23.2 |
| VideoLLM-online-8B [2] | CVPR'24 | 39.1 | 40.1 | 34.5 | 31.1 | 46.0 | 32.4 | 31.5 | 34.2 | 42.5 | 27.9 | 36.0 |
| Dispider-8B [31] | CVPR'25 | 74.9 | 75.5 | 74.1 | 73.1 | 74.4 | 59.9 | 76.1 | 62.9 | 62.2 | 45.8 | 67.6 |
| TimeChatOnline-7B [48] | MM'25 | 80.2 | 82.0 | 79.5 | 83.3 | 76.1 | 78.5 | 78.7 | 64.6 | 69.6 | 58.0 | 75.4 |
| Streamforest-7B [51] | NeurIPS'25 | 83.1 | 82.8 | 82.7 | 84.3 | 77.5 | 78.2 | 76.9 | 69.1 | 75.6 | 54.4 | 77.3 |
| **VST-7B (ours)** | - | **85.4** | 82.0 | **86.4** | **89.1** | 74.2 | **87.2** | **82.4** | **73.1** | **73.9** | 47.3 | **79.5** |

As shown in Tabs. 1 and 2, we evaluate our model on two online benchmarks, StreamingBench and OVO-Bench. VST-7B achieves **79.5%** on StreamingBench and **59.3%** on OVO-Bench, clearly outperforming prior open-source streaming SOTA models, including Streamforest [51] (77.3%) on StreamingBench and Streamo [43] (57.9%) on OVO-Bench. Notably, despite being much smaller than proprietary models, our method surpasses GPT-4o and Gemini 1.5 pro on StreamingBench by **+6.2%** and **+3.8%**, respectively, and achieves comparable performance with GPT-4o on OVO-Bench.

Beyond the overall scores, VST-7B is particularly strong on OVO-Bench's **Backward Tracing task**, where it achieves **56.7%**, outperforming Streamforest by +4.7%. This result indicates that our model can retain and retrieve historical information effectively, supporting sustained memory over streaming inputs.

> 💡 **实验结果批读（StreamingBench）**:
> - VST-7B 在 10 个子任务中的 8 个取得最优（OP, CR, CS, ATP, TR, PR, SU, ACP）
> - 相比 base model Qwen2.5-VL-7B（73.7%），提升 +5.8%
> - 相比 streaming SOTA Streamforest（77.3%），提升 +2.2%
> - 相比 proprietary GPT-4o（73.3%），提升 +6.2%
> - 唯一明显的弱项是 CT（47.3% vs TimeChatOnline 58.0%），CT=Counterfactual Thinking，说明 VST 的因果反事实推理能力还有提升空间

---

**Table 2**: Comparison of offline and online VideoLLMs on **OVO-Bench**.

| Model | Real-Time Avg. | Backward Avg. | Forward Avg. | **Overall** |
|-------|---------------|---------------|--------------|-------------|
| **Proprietary Models** | | | | |
| Gemini 1.5 pro [37] | 62.5 | 35.5 | 74.2 | 63.0 |
| GPT-4o [30] | 59.8 | 27.6 | 73.2 | 59.5 |
| **Open-source Online Models** | | | | |
| VideoLLM-online-8B [2] | 17.7 | 18.1 | 37.4 | 41.8 |
| Dispider-8B [31] | 36.1 | 18.1 | 37.4 | 48.8 |
| TimeChatOnline-7B [48] | 41.7 | 31.6 | 38.5 | 46.7 |
| Streamforest-7B [51] | 52.0 | 32.8 | 70.6 | 55.6 |
| Streamo-7B [43] | 49.2 | 30.8 | 57.6 | 57.9 |
| **VST-7B (ours)** | **56.7** | **33.0** | 66.9 | **59.3** |

> 💡 **实验结果批读（OVO-Bench）**:
> - Overall 59.3%，超过 Streamo (57.9%) 和 Streamforest (55.6%)
> - Backward Tracing 56.7%，比 Streamforest 高 +4.7%，证明 textual memory 在历史信息保留上优于纯 visual token 管理
> - Forward Prediction 66.9%，略低于 Streamforest (70.6%)，但 Forward 的 ASI (Action Sequence Inference) 和 HLD (High-Level Decision) 子任务 VST 表现不错
> - Real-Time 56.7%，大幅领先所有 open-source online models

### 3.4 Offline Video Benchmark Results

**Table 3**: Comparison of offline and online VideoLLMs on **VideoMME** (without subtitles), **LongVideoBench**, and **VideoHolmes**.

| Model | Venue | VideoMME Long | VideoMME Overall | LongVideoBench | VideoHolmes |
|-------|-------|---------------|------------------|----------------|-------------|
| **Proprietary Models** | | | | | |
| Gemini 1.5 pro [37] | - | 67.4 | 75.0 | 64.0 | 45.7 |
| GPT-4o [30] | - | 65.3 | 71.9 | 66.7 | 42.0 |
| **Open-source Offline Models** | | | | | |
| LongVA-7B [55] | TMLR'25 | 47.6 | 54.3 | 56.3 | - |
| Video-R1-7B [8] | NeurIPS'25 | - | 61.4 | - | 36.5 |
| LongVILA-R1-7B [4] | NeurIPS'25 | 55.2 | 65.1 | 58.0 | - |
| REVISOR-7B [21] | CVPR'26 | 56.2 | 65.7 | 57.5 | - |
| **Open-source Online Models** | | | | | |
| Dispider-7B [31] | CVPR'25 | - | 57.2 | - | - |
| Streamforest-7B [51] | NeurIPS'25 | - | 61.4 | - | - |
| TimeChatOnline-7B [48] | MM'25 | 48.4 | 62.4 | 55.4 | - |
| **VST-7B (Ours)** | - | 55.3 | 64.9 | **58.0** | **41.9** |

In Tab. 3, we evaluate VST-7B on three offline video benchmarks. The results show that VST-7B delivers competitive performance across all three datasets, with particularly strong gains on long-video understanding and complex reasoning:

- On long-video benchmarks, VST-7B achieves **55.3%** on VideoMME-long, outperforming TimeChat-Online by +6.9%, and **58.0%** on LongVideoBench, exceeding it by +2.6%.
- On the reasoning benchmark VideoHolmes, VST-7B reaches **41.9%**, surpassing Video-R1 by **+5.4%**.
- We attribute these improvements to our streaming thinking framework, which enables dynamic thinking over long videos to build long-term memory, and leverages both historical memory and current visual context for deep reasoning.

> 💡 **关键数字批读**:
> - VST 在 offline benchmarks 上的表现证明其既擅长 online（流式实时理解）也擅长 offline（长视频深度推理），这是一种"通才"能力
> - VideoHolmes 上超越 Video-R1 (+5.4%) 非常关键——Video-R1 是专门为 video reasoning 设计的 RL 方法，而 VST 在推理上超越它的同时，还保持了 15.7x 的响应速度优势
> - 长视频上的优势（VideoMME-long 55.3% vs TimeChatOnline 48.4%）说明 streaming thinking 建立的 textual memory 对理解超长时序依赖是有效的

### 3.5 Ablation Study

**Table 4**: Ablation study on VST training schedule.

| Model & Config | OVO-Bench Backward | OVO-Bench Forward | OVO-Bench Overall | VideoMME |
|---------------|-------------------|-------------------|-------------------|----------|
| Qwen2.5-VL-7B (Base model) | 47.5 | 41.9 | 50.5 | 62.9 |
| **Ablation on VST-SFT training data** | | | | |
| +LLaVA-Vid (50K) | 49.9 | 42.4 | 52.3 | 61.8 |
| +LLaVA-Vid (30K) & VST (20K) | 52.0 | 50.1 | 56.8 | 62.5 |
| +LLaVA-Vid (20K) & VST (30K) | 53.3 | 50.0 | 57.1 | 63.1 |
| **Ablation on different training stage** | | | | |
| +VST-SFT | **56.7** | 48.5 | 57.4 | 63.0 |
| +VST-RL | 49.3 | **54.6** | 56.8 | 62.8 |
| +VST-SFT & VST-RL | **56.7** | 54.0 | **59.3** | **64.9** |

> 💡 **消融解读（Table 4）**:
> **数据混合比例消融**：
> - 仅用 LLaVA-Vid (50K) → OVO-Bench 52.3%（仅比 base model +1.8%），说明通用 QA 数据对 streaming understanding 帮助有限
> - 加入 VST data 后：20K VST → 56.8%（+4.5%），30K VST → 57.1%（+4.8%），证明 VST streaming-thought 数据是关键贡献
> - VST 数据从 20K 到 30K 的边际收益递减，说明 20-30K 是合适的配比
>
> **训练阶段消融**：
> - **VST-SFT 主要负责 Backward**：Backward 从 47.5% → 56.7%（+9.2%），而 Forward 仅从 41.9% → 48.5%（+6.6%）。SFT 教会模型"记录历史"。
> - **VST-RL 主要负责 Forward**：Forward 从 41.9% → 54.6%（+12.7%），而 Backward 仅从 47.5% → 49.3%（+1.8%）。RL 教会模型"预测未来"。
> - **两者结合达到最佳**：Backward 56.7% + Forward 54.0% + Overall 59.3%。两个阶段的贡献是互补的（complementary），而非替代的（substitutive）。

**Ablation on Streaming Thinking Times at Inference.** Figure 5 analyzes the impact of maximum streaming thinking times on OVO-Bench.

![Figure 5a](../images/47939b2c36734b9a28ac0e72a1d3f2c12dabaa449c4e585f1c44fc1646dd6bd7.jpg)
![Figure 5b](../images/e3d59d79d63338f100b4c24e012c77788cfa020819a6915358a43dbc2154e3fa.jpg)

*Figure 5: Ablation study on max thinking times.*

For the Backward task, accuracy increases from 53.3% and grows continuously from 1 to 16 steps, ultimately reaching 57.5%. This demonstrates that additional thinking steps help generate precise memories for backward tracing. For the Real-Time and Forward tasks, initial thinking steps significantly aid in understanding visual information. However, performance reaches a plateau for >= 4 steps, as excessive memory details introduce redundancy.

> 💡 **消融解读（Figure 5）**:
> - **Backward 随 thinking steps 单调增长**：更多思考 → 更丰富的 textual memory → 更好的回溯能力。这是一个非常符合直觉的结果——历史记录越详细，回溯越准确。
> - **Forward/Real-Time 在 4 steps 处饱和**：说明对于需要即时理解和预测的任务，过多的中间思考反而引入噪声。4 步是 practical 的最优配置，这也是论文选择 max thinking times=4 的原因。
> - **启示**：Backward 和 Forward 对 thinking steps 的需求不同，理想情况下可以对不同任务动态调整 thinking budget。

**Table 5**: Ablation on different base offline VideoLLM's size.

| Size | Model | OVO-Bench Overall | StreamingBench Realtime | VideoMME Overall | LongVideoBench | VideoHolmes |
|------|-------|-------------------|------------------------|------------------|----------------|-------------|
| 3B | Qwen2.5-VL | 53.1 | 67.8 | 57.9 | 53.3 | 30.7 |
| 3B | VST | 56.2 (+3.1) | 75.5 (+7.7) | 59.5 (+1.6) | 54.1 (+0.8) | 36.1 (+5.4) |
| 7B | Qwen2.5-VL | 55.0 | 71.7 | 62.3 | 54.7 | 32.9 |
| 7B | VST | 59.3 (+4.3) | 79.5 (+7.8) | 64.9 (+2.6) | 58.0 (+3.3) | 41.9 (+9.0) |
| 32B | Qwen2.5-VL | 60.1 | 71.5 | 65.8 | 59.8 | 40.1 |
| 32B | VST | 63.5 (+3.4) | 80.7 (+9.2) | 67.2 (+1.4) | 60.7 (+0.9) | 45.1 (+5.0) |

Table 5 examines the impact of the base model capacity. We apply our two-stage training recipe (VST-SFT and VST-RL) to the Qwen2.5-VL-Instruct models at 3B, 7B, and 32B scales. Evaluated under identical inference configurations, the Video Stream Thinking paradigm yields consistent improvements across all online and offline benchmarks regardless of the model size. For instance, on StreamingBench Realtime, VST achieves absolute accuracy gains of +7.7%, +7.8%, and +9.2% over the 3B, 7B, and 32B base models, respectively. Similar consistent enhancements are observed on complex tasks like VideoHolmes (+5.4%, +9.0%, and +5.0%). These results demonstrate that our proposed method is **highly parameter-scalable**.

> 💡 **消融解读（Table 5）**:
> - **跨参数规模的泛化能力**：VST 对 3B/7B/32B 三个规模的 Qwen2.5-VL 都能带来一致的提升，证明 VST 不是对特定模型规模的"过拟合"
> - **StreamingBench Realtime 增益最显著**（+7.7~9.2%），说明 VST 对 online 场景的提升是普适的
> - **32B 的 LongVideoBench 增益较小**（+0.9%），可能是因为 base model 本身已经很强（59.8%），或者长视频理解的增益已达瓶颈
> - **VideoHolmes 在 7B 增益最大**（+9.0%），可能是因为 7B 恰好是 VST 训练的最佳规模，而 3B 容量不足、32B 基数较高

### 3.6 Analysis

#### Efficiency Analysis

**Table 6**: Inference Latency.

| Method | QA Latency |
|--------|-----------|
| **Offline** | |
| Qwen2.5-VL-7B | 0.54s |
| Qwen2.5-VL-7B w/CoT | 5.30s |
| Video-R1 w/CoT | 8.80s |
| **Online** | |
| VideoLLM-online-8B | 0.38s |
| Dispider-7B | 1.10s |
| VST-3B (Ours) | 0.53s |
| VST-7B (Ours) | 0.56s |
| VST-32B (Ours) | 1.40s |

We compare the QA latency of several offline and online methods under the same experimental setup. All measurements are conducted on VideoHolmes, as shown in Tab. 6. Models without CoT directly output the final answer without generating intermediate reasoning. Benefiting from our query-ahead streaming think mechanism, VST maintains significantly lower response latency.

Moreover, streaming think is executed asynchronously before the query and finishes within the clip inter-arrival interval, so its computation is amortized over playback rather than added after the query. As a result, it does not increase the real-world end-to-end inference time.

> 💡 **效率分析批读**:
> - **VST-7B QA Latency: 0.56s** vs Video-R1: 8.80s（快 **15.7x**）vs Qwen2.5-VL w/CoT: 5.30s（快 **9.5x**）
> - VST-7B 的 QA latency（0.56s）与 Qwen2.5-VL 直接回答（0.54s）几乎相同，说明 streaming thinking 的推理成本完全被摊销
> - VST-32B 延迟为 1.40s，虽然比 7B 高，但仍远低于 CoT 方法
> - 关键洞察：VST 将延迟的主导部分从"可见的用户等待时间"转移到了"不可见的视频播放间隙"

#### Case Study

![Figure 6](../images/19df46057ddbbb359c016f60107f222f2e65916f3063280f7bb11ae9e62d27ab.jpg)

*Figure 6: Case Study from VideoHolmes. We compare VST-7B with Video-R1-7B. VST-7B processes the video stream and performs streaming thinking before the query, then answers directly once the query arrives. In contrast, Video-R1-7B generates CoT after the query, resulting in higher QA latency. VST-7B achieves better performance with lower QA latency in this example.*

Fig. 6 presents a case study from the VideoHolmes benchmark. The query requires temporal reasoning over disjoint segments, specifically aligning repeated visualizations of a wall clock with the subsequent appearance of a "blurred-face man". The baseline Video-R1-7B, which relies on post-query thinking, fails to capture these dispersed temporal cues due to the difficulty of attending to specific evidence across a long context. Consequently, it hallucinates a spurious correlation involving object interactions, leading to a logical error. Furthermore, this retrospective reasoning incurs a significant latency of 9.53s.

In contrast, VST-7B employs streaming thinking to continuously update its evidence (e.g., timestamps and event triggers) as the video memories. This pre-query evidence accumulation allows VST to correctly deduce the time-based rule and, by shifting the reasoning burden to the streaming phase, drastically reduces the response latency to 0.51s. This comparison demonstrates that pre-query streaming thinking simultaneously enhances reasoning robustness and system responsiveness.

> 💡 **Figure 6 批读 (Case Study)**:
> - **Video-R1 为什么失败？** 因为它需要在 query 后从整个长上下文中定位分散的时间线索（wall clock + blurred-face man），而长上下文中的 evidence retrieval 本身就很困难，导致 hallucination
> - **VST 为什么成功？** 因为它在视频播放过程中已经持续记录了关键事件（时间戳、事件触发），这些信息以 textual memory 的形式被"索引"好了——查询时不再需要从头扫描长视频上下文
> - **核心洞察**：VST 的本质是"用存储换检索"——streaming thinking 阶段的额外计算代价换来的是 query 时免于长上下文检索的收益。这个 trade-off 在 streaming 场景下是非常划算的，因为 streaming thinking 的时间可以被视频播放间隙覆盖。

---

## Annotations

> 💡 **Q&A 批注记录**:
> - Q: VST 在 StreamingBench CT（Counterfactual Thinking）任务上表现较弱（47.3%），可能的原因是什么？
> - A: CT 任务要求模型对"如果事件以不同方式发生会怎样"进行推理，这种反事实推理通常需要全局视角（看到整个事件链后再做假设）。VST 的 streaming thinking 是因果性的（只能基于过去推理），对反事实推理的建模可能不够充分。这也是 limitation 中提到的"latent reasoning"未来方向可以探索的问题。
>
> - Q: 为什么 VST-32B 的增益在部分任务上反而比 7B 小（如 VideoHolmes +5.0% vs +9.0%）？
> - A: 可能有两个原因：1) Qwen2.5-VL-32B 的 base performance 已经较高（40.1%），提升空间有限；2) VST 的训练数据和超参数可能对 7B 规模做了更多优化。论文未详细讨论不同规模的 training recipe 差异，这可能是未来工作可以探索的方向。
>
> - Q: Video-R1 和 VST 的 QA Latency 对比是否公平？Video-R1 的 CoT 可以视为 VST 的 streaming thinking 的总和吗？
> - A: 不完全等价。Video-R1 的 post-query CoT 是一次性生成长段推理文本，而 VST 的 streaming thinking 是多次短段生成且被视频播放间隙覆盖。从"总计算量"角度看两者可能相近（都需要生成推理 token），但从"用户体感延迟"角度看 VST 远优于 Video-R1。这个对比的目的是证明"计算前置"的工程设计价值。

🔖 **Summary**: VST-7B achieves SOTA on online benchmarks (StreamingBench 79.5%, OVO-Bench 59.3%) while maintaining competitive offline performance (VideoHolmes 41.9%, +5.4% over Video-R1). Ablations reveal VST-SFT primarily benefits backward memory (+9.2%) and VST-RL benefits forward prediction (+12.7%), with their combination yielding best overall results. VST responds 15.7x faster than Video-R1 and scales effectively from 3B to 32B parameters.
