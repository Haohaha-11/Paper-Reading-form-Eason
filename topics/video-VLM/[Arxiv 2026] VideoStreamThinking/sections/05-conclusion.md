[← 返回 README](../README.md)

# 05 - Conclusion

📌 **Preview**: 总结 VST 的核心贡献——提出 streaming thinking 范式、发展 VST-SFT+VST-RL 训练框架、基于知识图谱的数据合成 pipeline，并讨论当前局限（token 消耗、与 visual memory 机制的整合）和未来方向。

---

## 5 Conclusion

In this paper, we propose **Video Streaming Thinking (VST)**, a new paradigm for streaming video understanding that introduces a synchronized stream of logical inference with real-time responsiveness. VST enables a *thinking-while-watching* mechanism that performs reasoning over incoming clips during streaming.

We further develop a post-training recipe (VST-SFT and VST-RL) and an automated data synthesis pipeline based on video knowledge graphs to produce streaming-thought supervision. Empirically, VST not only delivers robust performance across multiple online and offline video understanding benchmarks but also scales seamlessly to VideoLLMs ranging from 3B to 32B parameters, demonstrating exceptional generalization and broad applicability.

Overall, our study establishes VST as a practical test-time scaling approach for streaming scenarios, simultaneously enabling explicit CoT generation and real-time responsiveness.

### Limitation and Future Works

While the computation of streaming thoughts can be scheduled in parallel with incoming video clips, the additional LLM token consumption is still non-negligible. A promising direction is to explore **latent reasoning** to enable more token-efficient streaming thinking. Moreover, VST primarily focuses on text-guided memory management, which is orthogonal to existing streaming visual memory mechanisms. Investigating their combination and potential synergy is an interesting avenue for future work.

> 💡 **局限性与未来方向 批读**:
> 1. **Token 消耗（Latent Reasoning）**: 当前 streaming thinking 生成了大量文本 tokens（每步 thinking 都会消耗 token budget），虽然不增加用户感知延迟，但 GPU 计算开销不可忽略。作者提出 latent reasoning（在隐空间中推理而非显式文本生成）是合理的后续方向——类似于 Coconut [2024] 的连续 latent space reasoning 思路。
> 2. **与 Visual Memory 的整合**: VST 的 textual memory 和 StreamForest/TimeChatOnline 的 visual memory 是正交的，各有优势。Textual memory 提供语义压缩，visual memory 保留细粒度感知。作者认为两者结合是 promising 的方向，这意味着未来可能出现"dual-stream memory"的架构：既有 textual 的语义记忆流，也有 visual 的特征记忆流。
> 3. **尚未讨论的问题**：论文未讨论 multi-user / multi-query 场景——如果多个用户在视频流的不同时间点提问，streaming thinking 是否需要对不同用户的问题进行差异化？这是实际部署中的重要挑战。

> 💡 **Q&A 批注记录**:
> - Q: VST 的方法论是否适合扩展到其他流式模态（如音频流、传感器流）？
> - A: 从原理上讲是的。VST 的 dual-memory system（textual memory + modality-specific buffer）和 "thinking while streaming" 范式是模态无关的。对于音频流，可以将 visual buffer 替换为 audio buffer，streaming thinking 机制保持不变。事实上，LiveCC [3] 等工作的 speech transcription streaming 就是一个潜在的应用场景。
>
> - Q: 为什么 limitation 只提了 token consumption，没有提 thinking quality 的控制？
> - A: VST-RL 的设计已经通过"仅 reward final answer"部分解决了 thinking quality 问题。但确实存在一个隐含的局限性：streaming thinking 的内容是 LLM 自主生成的，可能存在"离题思考"(tangential reasoning) 或"hallucinated memory"（记录了不准确的事件）的风险。这是"autonomous memory management"的双刃剑——你可以自己管理记忆，但也可能管理出错。后续可以考虑加入 consistency check 机制。

---

## Appendix (Key Details)

### A.1 Inference Prompt

The LLM generates two distinct types of responses:

1. **Intermittent streaming inference** (as video progresses): conditioned primarily on past memory and the current video clip
2. **Final answer generation** (upon receiving user query): conditioned on accumulated memory, current video clip, and the specific question

**Prompt Template**:
```
[System]
You are a Streaming Video Analyst.
{Memory}
{TimeStamp} {VideoClip}

[System]
You are a Streaming Video Analyst.
{Memory}
{TimeStamp} {VideoClip}
{QueryTime} Based on the provided Video Memory and the Current Video
Clip, answer the following Problem.
{Problem}
Output the final answer in \boxed{}
Your answer:
```

### A.2 Streaming Inference Pipeline

![Figure 7](../images/fd22e55679939455aa35996dc302cf95f4ec1651398efb4e08840bbf4318a44d.jpg)

*Figure 7: The streaming inference pipeline of VST. By generating stream thoughts for incoming video clips before a user query arrives, VST effectively hides reasoning latency and enables rapid QA responses.*

Prior to receiving the user query, we conduct a streaming thinking process for each video clip, ensuring the output is generated before the subsequent clip arrives. Consequently, our method effectively utilizes the natural waiting time inherent in real-world video streams. This enables a rapid response once the user poses a question, where the QA latency is defined as the time elapsed from the user's query submission to the LLM's response.

> 💡 **Figure 7 批读**: 推理 pipeline 的时间线清晰展示了 VST 的 latency hiding 机制：
> - Clip 1 arrives → Streaming Think 1 (completed before Clip 2 arrives)
> - Clip 2 arrives → Streaming Think 2
> - ...
> - Clip K arrives → Streaming Think K
> - User Query arrives → Direct Answer (0.56s!)
> - 关键假设：每个 clip 的处理时间 < clip 之间的间隔时间。如果视频帧率很高导致间隔极短，可能需要优化 thinking 的生成速度或使用更小的 thinking budget。

### B. Training Hyperparameters

**VST-SFT**: Up to 384 frames per video, max 19,267,584 video pixels, capped at 24K video tokens with 8K reserved for language/reasoning. 1 epoch, lr=5e-6, 8 gradient accumulation steps.

**VST-RL**: DAPO algorithm, max prompt length 11,000 tokens, 1,000 reserved for generation. Rollout: 8 candidates per prompt, temperature=1.0, top-p=0.98. 1 epoch, global batch size=256, PPO mini-batch=64, lr=5e-7 with 20 warmup steps. Frozen vision tower, KL penalty coefficient=0.001, FSDP with parameter and optimizer offloading.

> 💡 **超参数批读**:
> - VST-SFT 的 visual token cap (24K) + language reserve (8K) = 32K context，对应当前 VideoLLM 的主流配置
> - VST-RL 的 rollout candidates (8 per prompt) + group size=8 是 GRPO 的标准配置
> - KL penalty=0.001 较小，说明作者对 policy divergence 的容忍度较高，可能是因为 SFT 已经提供了较好的初始化

### C. Data Generation Templates

The data generation pipeline uses three distinct prompt templates:

1. **Video Knowledge Graph Generation**: Maps all physical relationships (object-to-object, containment, spatial relations) in video segments, with strict anti-"hub-and-spoke" bias (avoid making humans the subject of every relation)
2. **Intermediate CoT Generation**: Focuses on dynamics and incremental updates, strictly avoiding restating previous segments' content
3. **QA Generation**: Synthesizes multi-hop reasoning QA pairs requiring integration of information from multiple video segments, with natural language constraints (no direct segment/clip references)

> 💡 **Data Generation 设计批读**:
> - "NO HUB-AND-SPOKE BIAS" 约束非常关键：避免知识图谱退化为以人类主角为中心的星形结构，强制模型建模物体间关系（如"桌子上的杯子"而非"人看着杯子"）
> - "No restate" 约束确保每个 streaming thought 只描述增量信息，避免冗余——这直接对应了 streaming 场景中的效率需求
> - 生成 QA 时强制使用时间区间（如"03:15 to 04:30"）而非 clip index，使 QA 更自然、更具泛化性

🔖 **Summary**: VST 建立了一个实用的流式 test-time scaling 框架，明确了两个未来方向：(1) token 高效的 latent reasoning 以降低计算开销，(2) 与现有人视觉记忆机制整合以构建统一的"双流记忆"架构。推理 pipeline 将查询前思维与查询后回答干净地解耦，实现了实时响应。
