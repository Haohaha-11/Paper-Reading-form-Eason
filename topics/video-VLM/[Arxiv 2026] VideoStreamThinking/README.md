# Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously

> **Authors**: Yiran Guan<sup>1\*</sup>, Liang Yin<sup>1\*</sup>, Dingkang Liang<sup>1</sup>, Jianzhong Ju<sup>2</sup>, Zhenbo Luo<sup>2</sup>, Jian Luan<sup>2</sup>, Yuliang Liu<sup>1</sup>, Xiang Bai<sup>1B</sup>
> **Affiliations**: <sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>MiLM Plus, Xiaomi Inc.
> **Venue**: arXiv 2026 (2603.12262)
> **Links**: [arXiv](https://arxiv.org/abs/2603.12262) | [GitHub](https://github.com/1ranGuan/VST) | [Connected Papers](https://www.connectedpapers.com/main/2603.12262)
> \* Equal contribution. <sup>B</sup> Corresponding author.

---

## One-Sentence Summary

**VST introduces a "thinking while watching" paradigm for streaming video understanding that amortizes Chain-of-Thought reasoning over video playback intervals, achieving real-time responsiveness and state-of-the-art online accuracy via a two-stage post-training pipeline (VST-SFT + VST-RL) and a knowledge-graph-grounded data synthesis engine.**

---

## Contributions

1. **VST Paradigm**: A novel streaming video understanding paradigm that interleaves active, explicit CoT generation with continuous video streams, shifting the LLM backend from passive waiting to proactive intermittent reasoning during video consumption. This achieves amortized test-time scaling while preserving real-time QA responsiveness.

2. **Two-Stage Post-Training Pipeline**:
   - **VST-SFT**: Structurally adapts an offline VideoLLM to causal streaming reasoning via a streaming attention mask that enforces temporal causality, bootstrapping the thinking-while-watching capability from off-policy demonstrations.
   - **VST-RL**: End-to-end reinforcement learning via GRPO in a multi-turn video interaction environment, where rewards are computed solely from the final answer correctness to encourage useful intermediate streaming thoughts.

3. **Knowledge-Graph-Grounded Data Synthesis**: An automated pipeline that builds temporally consistent video knowledge graphs, samples diverse multi-hop evidence chains via DFS, and uses Gemini 3.0 flash to generate 100K high-quality streaming-thought QA pairs with strict temporal causality constraints.

4. **Comprehensive Empirical Validation**: SOTA results on online benchmarks (StreamingBench 79.5%, OVO-Bench 59.3%), competitive offline performance (VideoHolmes 41.9%, +5.4% over Video-R1), 15.7x faster QA latency than Video-R1, and consistent gains across 3B/7B/32B model scales.

---

## Section Navigation

| Section | File | Key Content |
|---------|------|-------------|
| 00 - Abstract | [sections/00-abstract.md](sections/00-abstract.md) | Paper abstract, problem motivation, key numbers |
| 01 - Introduction | [sections/01-introduction.md](sections/01-introduction.md) | Paradigm comparison (Fig.1), human cognition inspiration, contributions |
| 02 - Related Work | [sections/02-related-work.md](sections/02-related-work.md) | Streaming Video Understanding, VideoLLMs Test-Time Scaling |
| 03 - Methodology | [sections/03-methodology.md](sections/03-methodology.md) | VST paradigm, dual-memory system, VST-SFT/RL training, data synthesis |
| 04 - Experiments | [sections/04-experiments.md](sections/04-experiments.md) | Benchmark results, ablation studies, efficiency analysis, case study |
| 05 - Conclusion | [sections/05-conclusion.md](sections/05-conclusion.md) | Summary, limitations, future work, appendix highlights |

---

## Key Numbers

### Performance Highlights
| Metric | VST-7B | Previous SOTA | Gain |
|--------|--------|---------------|------|
| StreamingBench | **79.5%** | 77.3% (StreamForest) | +2.2% |
| OVO-Bench Overall | **59.3%** | 57.9% (Streamo) | +1.4% |
| OVO-Bench Backward Tracing | **56.7%** | 52.0% (StreamForest) | +4.7% |
| VideoHolmes | **41.9%** | 36.5% (Video-R1) | +5.4% |
| LongVideoBench | **58.0%** | 58.0% (LongVILA-R1) | tied SOTA |

### Latency Comparison (VideoHolmes)
| Method | QA Latency | vs VST-7B |
|--------|-----------|-----------|
| Video-R1 w/CoT | 8.80s | **15.7x slower** |
| Qwen2.5-VL w/CoT | 5.30s | **9.5x slower** |
| Qwen2.5-VL direct | 0.54s | similar |
| **VST-7B** | **0.56s** | -- |

### Training Scale
| Stage | Data | Compute |
|-------|------|---------|
| VST-SFT | 100K streaming thoughts + 50K QA | 32 x 80GB GPUs, 1 epoch |
| VST-RL | 11K questions (GRPO, N=8) | 32 x 80GB GPUs, 1 epoch |

### Ablation: Training Stage Contribution (OVO-Bench)
| Configuration | Backward | Forward | Overall |
|---------------|----------|---------|---------|
| Base (Qwen2.5-VL-7B) | 47.5% | 41.9% | 50.5% |
| +VST-SFT only | **56.7%** (+9.2) | 48.5% (+6.6) | 57.4% |
| +VST-RL only | 49.3% (+1.8) | **54.6%** (+12.7) | 56.8% |
| +VST-SFT & VST-RL | **56.7%** | 54.0% | **59.3%** |

---

## Data Flow: Input -> Intermediate -> Output

```
INPUT: Continuous Video Stream (indefinite length, 2 fps sampling)
  │
  ├─▶ [PySceneDetect] Scene segmentation into N clips
  │
  ▼
INTERMEDIATE 1: Knowledge Graph Construction (Data Synthesis only)
  ├─ Sliding window entity extraction (Gemini 3.0 flash)
  │   └─ Output: (head, relation, tail) triples per clip
  ├─ Entity bank maintenance (FIFO, W-window)
  ├─ Noise filtering (duplicates, subtitles removal)
  ├─ NetworkX graph construction
  └─ DFS evidence chain sampling (entity overlap < 10%)
        │
        ▼
INTERMEDIATE 2: Streaming Thought QA Generation (VST-SFT data)
  ├─ CoT rationale generation conditioned on knowledge graph
  ├─ Multi-hop QA pair synthesis from evidence chain
  └─ 5-stage filtering (world-knowledge, format, logic, repetition, thought validation)
        │
        ▼
INTERMEDIATE 3: VST-SFT Training
  ├─ Input: Interleaved (video_clip, streaming_thought) sequences
  ├─ Streaming attention mask: visual tokens visible only within sliding window L
  ├─ Temporal segmentation: long sequences split into consecutive segments
  └─ Next-token loss on {streaming thoughts} + {final answer} only
        │
        ▼
INTERMEDIATE 4: VST-RL Training (Agentic Loop)
  ├─ Rollout: Policy model interacts with streaming environment
  │   └─ Generates trajectory: (z^1, z^2, ..., z^{K-1}, y)
  ├─ Reward: based solely on final answer correctness (verifiable)
  ├─ GRPO: advantage assigned to ALL tokens in trajectory
  └─ DAPO clipping with KL penalty (β=0.001)
        │
        ▼
OUTPUT (Inference):
  ┌──────────────────────────────────────────────────────┐
  │ DUAL-MEMORY SYSTEM                                    │
  │                                                       │
  │ Short-term Visual Buffer [c^k]: last L visual tokens  │
  │   │                                                   │
  │   ▼                                                   │
  │ Long-term Textual Memory [m^k]: accumulated thoughts  │
  │   │  (FIFO eviction when capacity exceeded)           │
  │   │                                                   │
  │ Streaming Thinking [z^k]: autoregressive generation   │
  │   per clip: z^k ~ p(z | c^k, m^{k-1})                │
  │                                                       │
  │ Upon User Query [q]:                                  │
  │   Final Answer: y ~ p(y | q, c^K, m^K)               │
  │   QA Latency: ~0.56s (thinking amortized)             │
  └──────────────────────────────────────────────────────┘
```

### Probability Decomposition (Core Equation)
```
p(y | q, V) = p(y | q, c^K, m^K)           ← Direct Answer (post-query)
              × ∏_{k=1}^{K-1} p(z^k | c^k, m^{k-1})  ← Streaming Thinking (pre-query)
```

---

## Pros and Cons

### Pros

1. **Latency-Quality Pareto Improvement**: VST achieves BOTH higher accuracy AND lower latency compared to offline CoT methods (e.g., +5.4% on VideoHolmes with 15.7x faster response vs Video-R1). This is rare -- typically accuracy and latency trade off against each other.

2. **Principled Dual-Memory Architecture**: The short-term visual buffer (raw perception) + long-term textual memory (semantic compression) design elegantly solves the fixed-context-window problem for indefinite video streams. The FIFO textual memory naturally forgets old events, mimicking human working memory.

3. **Training-Inference Alignment**: The streaming attention mask (Eq.3) ensures the model learns under the exact same causal constraints it faces at inference time. No train-test mismatch in attention patterns.

4. **Scalable Across Model Sizes**: Consistent performance gains across 3B/7B/32B scales (+7.7% to +9.2% on StreamingBench) demonstrate the paradigm's parameter-scalability, not just a 7B-specific trick.

5. **Well-Motivated by Cognitive Science**: The "thinking while watching" design is grounded in neural coupling research [16, 36], giving the paradigm a strong cognitive foundation beyond pure engineering.

6. **Data Synthesis with Causal Guarantees**: The knowledge-graph-grounded pipeline ensures generated streaming thoughts never "leak" future information, a critical property that naive video-to-QA generation methods cannot guarantee.

### Cons

1. **Additional Token Consumption**: While QA latency is low, the streaming thinking itself consumes significant LLM token budget (~4 thinking steps per video segment). The paper acknowledges this and suggests latent reasoning as future work.

2. **Counterfactual Reasoning Weakness**: VST's CT (Counterfactual Thinking) score on StreamingBench is 47.3%, notably lower than TimeChatOnline's 58.0%. Causal (forward-only) streaming thinking may be inherently limited for counterfactual reasoning that requires considering alternative event sequences.

3. **Thinking Quality Not Directly Supervised**: VST-RL only rewards the final answer, relying on the optimization to implicitly discover useful thinking patterns. There is no guarantee that every generated streaming thought is factually accurate -- hallucinated thoughts could be stored in textual memory and propagate errors.

4. **Single-Query Assumption**: The paradigm assumes one user query per video stream. In multi-user or multi-turn scenarios, the streaming thinking may need to serve diverse downstream questions, which may require more generic or query-conditioned thinking.

5. **Dependence on Scene Detection Quality**: The data synthesis pipeline relies on PySceneDetect for clip segmentation. Poor scene boundaries could fragment coherent events across different clips, degrading the quality of generated streaming thoughts.

6. **Limited Comparison with Visual Memory Methods**: VST's textual memory is not directly compared with or combined with visual memory methods (StreamForest, TimeChatOnline). The paper acknowledges this as future work, but the absence of a combined system leaves open the question of whether textual + visual memory would be synergistic.

---

## Citation Landscape

**Connected Papers**: [https://www.connectedpapers.com/main/2603.12262](https://www.connectedpapers.com/main/2603.12262)

### Key Related Works

**Streaming Video Understanding**:
- **VideoLLM-online** (Chen et al., CVPR 2024): Early work on online VideoLLM for streaming video, focused on efficient visual token processing.
- **Dispider** (Qian et al., CVPR 2025): Disentangled perception, decision, and reaction for active real-time interaction via external memory retrieval.
- **TimeChat-Online** (Yao et al., MM 2025): Shows 80% visual tokens are redundant in streaming, uses compression-based streaming.
- **StreamForest** (Zeng et al., NeurIPS 2025): Persistent event memory for efficient online video understanding, previous SOTA on StreamingBench (77.3%).
- **Streamo** (Xia et al., CVPR 2026): Streaming video instruction tuning method, previous SOTA on OVO-Bench (57.9%).
- **Flash-VStream** (Zhang et al., ICCV 2025): Real-time understanding for long video streams with efficient token management.
- **LiveVLM** (Ning et al., 2025): Streaming-oriented KV cache and retrieval for efficient online video understanding.
- **StreamMem** (Yang et al., 2025): Query-agnostic KV cache memory for streaming video.

**VideoLLM Test-Time Scaling / Reasoning**:
- **Video-R1** (Feng et al., NeurIPS 2025): R1-style RL for video reasoning, the main CoT baseline VST compares against on latency.
- **LongVILA-R1** (Chen et al., NeurIPS 2025): Scaling RL to long videos with paralleled encoding strategy (adopted by VST).
- **REVISOR** (Li et al., CVPR 2026): Multimodal introspective reasoning for long-form video, competitive offline baseline.
- **VideoEspresso** (Han et al., CVPR 2025): Large-scale CoT dataset via core frame selection for fine-grained video reasoning.
- **Video-RFT** (Wang et al., NeurIPS 2025): Reinforced fine-tuning for video reasoning capabilities.
- **ThinkOmni** (Guan et al., ICLR 2026): Lifting textual reasoning to omni-modal scenarios (same first author).
- **StreamingThinker** (Tong et al., 2025): Text-only "think while reading" for LLMs, conceptual precursor to VST in text domain.

**Foundation Models & Benchmarks**:
- **Qwen2.5-VL** (Bai et al., 2025): Base VideoLLM used by VST.
- **StreamingBench** (Lin et al., 2024): Online streaming video understanding benchmark.
- **OVO-Bench** (Niu et al., CVPR 2025): Real-world online video understanding benchmark with Backward/Forward task decomposition.
- **VideoHolmes** (Cheng et al., 2025): Complex video reasoning benchmark.
- **VideoMME** (Fu et al., CVPR 2025): Comprehensive offline video understanding benchmark.
- **LongVideoBench** (Wu et al., NeurIPS 2024): Long-context interleaved video-language benchmark.

---

## Q&A Record

> **Q1: How does VST's "thinking while watching" differ from simply running CoT periodically during video playback?**
>
> A: The key difference is in training. VST-SFT trains the model to generate streaming thoughts under strict temporal causality constraints (streaming attention mask, Eq.3) so the model never "sees" future frames when thinking. Naively prompting an offline model to generate CoT at intervals would leak future information because offline models are trained with global attention. Additionally, VST-RL provides end-to-end optimization so the thoughts are optimized for downstream QA usefulness, not just for generating plausible-sounding text.

> **Q2: Why does VST-SFT improve Backward memory (+9.2%) while VST-RL improves Forward prediction (+12.7%)?**
>
> A: VST-SFT teaches the model *what* to record in textual memory -- content summarization, event logging, entity tracking -- which directly benefits backward retrieval. VST-RL's reward is based on final answer correctness, and since many questions require predicting or reasoning about future events based on past observations, the RL optimization naturally encourages the model to generate thoughts that facilitate forward-looking inference. The two stages are complementary rather than redundant.

> **Q3: Is the 15.7x latency advantage over Video-R1 a fair comparison?**
>
> A: The latency measurement (QA latency = time from query to answer) is fair because it measures what users actually experience. The total computation (streaming thinking + final answer) may be comparable to Video-R1's post-query CoT, but VST frontloads the cost to idle periods during video playback. This is analogous to how video streaming buffers content during playback to avoid buffering interruptions -- the total data transferred is the same, but the user experience differs dramatically.

> **Q4: Can VST handle multi-turn conversations or multiple queries on the same stream?**
>
> A: The current formulation assumes one query per stream (Eq.1), but the textual memory mechanism is query-agnostic. In principle, accumulated streaming thoughts could serve multiple queries if they capture sufficiently general reasoning about the video. However, this is not evaluated in the paper and would require additional investigation, particularly around whether the streaming thoughts generated without a specific question in mind are sufficiently useful for diverse downstream queries.

> **Q5: How does the knowledge graph data synthesis pipeline prevent information leakage?**
>
> A: The pipeline builds the knowledge graph incrementally -- as each clip arrives (in temporal order), Gemini extracts entities and relations from ONLY that clip (plus a small overlap window W-1). This means when generating a streaming thought for clip k, the available knowledge is strictly bounded by what has been observed up to that point. The evidence chains are then sampled from the complete graph but the QA synthesis prompt enforces that each intermediate reasoning step references only information available at that timestamp.

> **Q6: What happens if streaming thinking generates incorrect or hallucinated content?**
>
> A: This is a genuine risk. Unlike the final answer which has ground truth for RL reward computation, individual streaming thoughts have no direct supervision signal. Hallucinated content in textual memory could propagate to the final answer. VST-RL provides indirect pressure against this (useless/harmful thoughts lead to wrong answers, which reduces reward), but there is no explicit hallucination detection mechanism. This is an implicit limitation of the "reward only final answer" approach.

> **Q7: Why does VST underperform on StreamingBench CT (Counterfactual Thinking, 47.3%) compared to TimeChatOnline (58.0%)?**
>
> A: Counterfactual reasoning ("what if X happened differently") requires reasoning about alternative event sequences, which inherently requires a kind of "hindsight" perspective that is at odds with VST's strict forward-causal thinking paradigm. TimeChatOnline, despite lacking explicit reasoning, may benefit from having all visual tokens available (with compression) rather than the more abstract textual memory that VST relies on. This suggests a potential complementarity between visual and textual memory approaches.

---

## Figures Index

| Figure | File | Description |
|--------|------|-------------|
| Fig. 1 | [images/21a7b95c...747a.jpg](images/21a7b95c3be0763213271697c0fed7b2418600a3c655a42ee43e5723c032747a.jpg) | Benchmark results and paradigm comparison (a-d) |
| Fig. 2 | [images/be382a30c...90e7.jpg](images/be382a30cb04d65a2fe4e291fdae034f6992b6b0cf2eab532ba363a16c6a90e7.jpg) | VST pipeline with dual-memory system |
| Fig. 3 | [images/6d5d820fb...3ae2.jpg](images/6d5d820fb427f8f6743c58065f984a4b81fb1573ff73c0f106a8afaea49a3ae2.jpg) | Training pipeline: VST-SFT + VST-RL |
| Fig. 4 | [images/c1db74b26...ce1.jpg](images/c1db74b26259d4b40ea898936c4410b7d4a94da0e3c331bdbad2839913c72ce1.jpg) | Data synthesis pipeline (knowledge graph -> QA) |
| Fig. 5a | [images/47939b2c3...6bd7.jpg](images/47939b2c36734b9a28ac0e72a1d3f2c12dabaa449c4e585f1c44fc1646dd6bd7.jpg) | Ablation: max thinking times vs accuracy |
| Fig. 5b | [images/e3d59d79d...e3fa.jpg](images/e3d59d79d63338f100b4c24e012c77788cfa020819a6915358a43dbc2154e3fa.jpg) | Ablation: max thinking times (continued) |
| Fig. 6 | [images/19df46057...27ab.jpg](images/19df46057ddbbb359c016f60107f222f2e65916f3063280f7bb11ae9e62d27ab.jpg) | Case study: VST vs Video-R1 on VideoHolmes |
| Fig. 7 | [images/fd22e5567...a44d.jpg](images/fd22e55679939455aa35996dc302cf95f4ec1651398efb4e08840bbf4318a44d.jpg) | Streaming inference pipeline (Appendix) |

---

*Batch reading completed on 2026-06-22. All original text preserved verbatim; annotations marked with > 💡.*
