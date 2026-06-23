[← 返回 README](../README.md)

# 02 - Related Work

📌 **Preview**: 讨论两个与 VST 直接相关的研究方向——Streaming Video Understanding 和 VideoLLMs Test-Time Scaling，以及 VST 如何在这两条线的交叉点上填补空白。

---

## 4 Related Work

### Streaming Video Understanding

Streaming video understanding processes continuous visual inputs of indeterminate length. Unlike offline methods, the lack of global sampling and restricted context windows poses significant challenges for VideoLLMs. Some existing methods attempt to retain extended video information within limited context lengths through real-time visual token compression [2, 32, 35, 45, 48, 51]. Others incorporate external memory mechanisms to recall historical information via query-relevant retrieval [6, 31, 47, 53].

However, these methods rely on static heuristics, lacking autonomous memory management and the ability to perform complex, multi-step reasoning. To bridge this gap, we propose Video Streaming Thinking (VST), which introduces an online thinking process that evolves with the video stream. By coupling autonomous memory management with in-depth instruction analysis, VST enables models to transcend short-range perception and achieve robust streaming intelligence.

> 💡 **问题动机 批读**: 作者对现有流式方法的批评非常精准——"static heuristics"和"lack of autonomous memory management"。现有方法（如 StreamForest 的 persistent event memory、Dispider 的 disentangled perception 等）本质上是用固定规则来管理 visual tokens，而 VST 的创新在于让 LLM 自己决定"记住什么"和"怎么推理"。这是从被动记忆管理到主动认知管理的范式转变。

### VideoLLMs Test-Time Scaling

Following the breakthrough of test-time scaling and chain-of-thought in LLMs [12, 38, 41], recent VideoLLMs have adopted supervised fine-tuning (SFT) to mimic expert reasoning trajectories [14, 15] or utilized R1-style reinforcement learning (RL) to enhance task performance [4, 8, 21, 40, 46].

Despite these advances, existing post-training research remains predominantly confined to offline video understanding. The exploration of reasoning within streaming contexts, particularly regarding long-horizon cognitive capabilities, remains a critical yet neglected frontier. In this paper, we introduce a unified SFT and RL framework for streaming video understanding. Our method achieves a synergistic balance between real-time responsiveness and sophisticated reasoning, enabling autonomous memory management and in-depth analysis of evolving video streams.

> 💡 **机制拆解 批读**: Test-time scaling 在 VideoLLM 领域的主要路线：1) Video-R1 [8] / LongVILA-R1 [4]：使用 RL 增强推理；2) VideoEspresso [14] / ReVisionLLM [15]：SFT 模仿专家推理轨迹。但这些工作的共同问题是：它们都是 offline 的，推理发生在看完全部视频之后。VST 首次将 test-time scaling 的思想引入 streaming 场景，同时解决了"怎么在流式场景下做 CoT"和"怎么不让 CoT 增加延迟"这两个问题。
>
> 💡 **定位批读**: VST 的定位非常清晰——它不想取代 offline VideoLLM（实验显示 offline 性能与 SOTA 持平），而是为 online/streaming 场景补充缺失的"推理"能力。这是对现有 VideoLLM test-time scaling 研究的重要补充，而非替代。

---

## Annotations

> 💡 **对比总结 批读**: 下表梳理了 streaming video understanding 中三类方法的对比：

| 维度 | Token Compression (StreamForest等) | KV Retrieval (Dispider, LiveVLM等) | **VST (Ours)** |
|------|--------------------------------------|-------------------------------------|----------------|
| 记忆管理方式 | 静态启发式（保留/丢弃） | 查询相关检索 | **LLM自主管理**（textual memory） |
| 是否有推理 | 无 | 无 | **有**（streaming CoT） |
| LLM参与程度 | 低（仅编码/解码） | 低（仅检索后生成） | **高**（全程参与推理） |
| 长时依赖 | 依赖压缩质量 | 依赖检索精度 | **依赖持续推理链** |

> 💡 **Q&A 批注记录**:
> - Q: VST 和 StreamingThinker [38] 有什么区别？
> - A: StreamingThinker [38] 是 text-only LLM 的 "think while reading" 方案，在阅读长文本时边读边思考。VST 可以看作其在多模态 video 领域的延伸。但 VST 的挑战更大：video tokens 的信息密度和顺序依赖性远超文本，且 visual context window 的约束更为严格。VST 的训练方法（streaming attention mask、VST-RL 的 agentic rollout）都是针对 video 模态专门设计的。
>
> - Q: 为什么现有流式方法不直接加 CoT？
> - A: 根本原因有两个：1) 如果 query 后再做 CoT，响应延迟不可接受（Video-R1 的 QA latency 达 8.80s vs VST 的 0.56s，差距 15.7x）；2) 流式场景的因果约束意味着你无法"回顾未来"来修正思考。VST 通过前移 CoT 到查询前解决了延迟问题，通过 streaming attention mask 解决了因果性问题。

🔖 **Summary**: VST 桥接了此前互不相干的两个研究方向：(1) 流式视频理解（缺乏推理能力）和 (2) VideoLLM test-time scaling（缺乏流式支持）。这是首个在在线视频场景中实现显式 CoT 推理的统一框架。
