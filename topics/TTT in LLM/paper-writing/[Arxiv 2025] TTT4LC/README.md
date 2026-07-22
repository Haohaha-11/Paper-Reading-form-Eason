# Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs

## Paper Metadata

| Field | Details |
|-------|---------|
| **Full Title** | Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs |
| **Authors** | Rachit Bansal, Aston Zhang, Rishabh Tiwari, Lovish Madaan, Sai Surya Duvvuri, Devvrit Khatri, David Brandfonbrener, David Alvarez-Melis, Prajjwal Bhargava, Mihir Sanjay Kale, Samy Jelassi |
| **Affiliations** | Meta, Harvard University, Kempner Institute at Harvard, OpenAI, UC Berkeley, UT Austin |
| **Venue** | arXiv 2025 (2025/12/15) |
| **ArXiv ID** | 2512.13898 |
| **Correspondence** | rachitbansal@g.harvard.edu, az@astonzhang.com |

## One-Sentence Summary

Long-context LLMs suffer from "score dilution" -- a phenomenon in static self-attention where distractors inflate the softmax denominator and bury the target signal -- and instead of generating more "thinking" tokens, a few gradient updates to query projections at test time (query-only TTT) provably lifts the target-distractor margin and yields 12.6/14.1 pp gains on LongBench-v2/ZeroScrolls under FLOP-matched budgets.

## Core Contributions

1. **Empirical diagnosis**: Controlled sandbox tasks (code bug localization + transaction log anomaly detection) expose that in-context accuracy drops sharply with context length T, and "thinking" tokens show rapidly diminishing returns.

2. **Theoretical formalization of score dilution**: Proved that in static, finite-precision self-attention, the target-distractor logit margin must scale as Omega(log T) to avoid vanishing target probability, and that autoregressively generating thinking tokens cannot circumvent this failure.

3. **Query-Only TTT (qTTT)**: A compute-efficient test-time training variant -- one prefill to cache K/V, then gradient updates only on query projection matrices over short spans -- that provably increases target-distractor separation.

4. **Strong empirical validation**: On 15+ datasets from LongBench-v2 and ZeroScrolls across Qwen3-{1.7B, 4B, 8B}, qTTT consistently outperforms both vanilla in-context and FLOP-matched thinking tokens, with largest gains on retrieval/multi-hop tasks.

5. **Practical inference-time reallocation**: Under matched FLOPs, T_think ~ 2 * N_qTTT * k, meaning the same compute budget spent on query updates rather than thinking tokens is a strictly better use of inference-time resources.

## Section Navigation

| Section | File | Description |
|---------|------|-------------|
| Abstract | [sections/00-abstract.md](sections/00-abstract.md) | Paper abstract |
| 1. Introduction | [sections/01-introduction.md](sections/01-introduction.md) | Motivation, failure diagnosis, qTTT overview, contributions |
| 2. Failure Analysis | [sections/02-failure-analysis.md](sections/02-failure-analysis.md) | Synthetic tasks, score dilution theory, thinking-token limits |
| 3. Methodology | [sections/03-methodology.md](sections/03-methodology.md) | Query-only TTT algorithm, theory of margin improvement, FLOP equivalence |
| 4. Experiments | [sections/04-experiments.md](sections/04-experiments.md) | LongBench-v2, ZeroScrolls, additional baselines, latency analysis |
| 5. Discussion | [sections/05-discussion.md](sections/05-discussion.md) | Prior work, discussion, future directions, appendix highlights |

## Key Numbers

| Metric | Value |
|--------|-------|
| LongBench-v2 avg improvement (Qwen3-4B, qTTT vs in-context) | +12.6 pp (27.0 -> 39.6) |
| ZeroScrolls avg improvement (Qwen3-4B, qTTT vs in-context) | +14.1 pp (18.4 -> 32.5) |
| LongBench-v2 avg improvement (Qwen3-8B, qTTT vs in-context) | +16.5 pp (32.3 -> 48.8) |
| Code Repositories improvement (Qwen3-8B) | 30.0 -> 44.0 (thinking) -> 52.0 (qTTT) |
| Long Dialogue History improvement (Qwen3-4B) | 30.8 -> 43.6 (qTTT) |
| Multi-Document QA improvement (Qwen3-4B) | 40.0 -> 46.0 (qTTT) |
| QAuLITY (ZeroScrolls) improvement (Qwen3-4B) | 40.5 -> 76.2 (thinking) -> 87.0 (qTTT) |
| MuSiQue improvement (Qwen3-4B) | 17.1 -> 7.5 (thinking drops!) -> 30.5 (qTTT) |
| Default hyperparams | k=128, N_qTTT=32, T_think=8192 |
| FLOP equivalence | T_think ~ 2 * N_qTTT * k |
| Models evaluated | Qwen3-{1.7B, 4B, 8B, 32B} |
| Datasets | 15+ from LongBench-v2 (6 subsets) + ZeroScrolls (8 datasets) |
| Context lengths tested | O(10^2) to O(10^4) tokens (synthetic), up to 128K (latency) |

## Data Flow: Input -> Intermediate -> Output

```
[Long Context x_{1:T}]
    |
    v
[Single Prefill] -- caches {K, V} at each layer
    |
    v
[Query-Only TTT Loop] -- N_qTTT iterations:
    |-- Sample random span x_{t:t+k}
    |-- Compute next-token prediction loss L_TTT on span (using frozen K/V)
    |-- Compute gradient ONLY w.r.t. {W_Q} matrices
    |-- Update {W_Q} <- {W_Q} - eta * grad
    |
    v
[Adapted Model f_{theta'}]
    |
    v
[Final Answer Generation] -- using updated W_Q + frozen K/V cache
```

**Key insight**: The K/V cache is computed only once. Query updates do NOT invalidate the cache, so each TTT step is cheap (only recomputes attention for the short span). Gradients flow only through {W_Q}, reshaping how queries access the evidence rather than changing the evidence itself.

## Pros/Cons & Future Work

### Strengths
- **Theoretically grounded**: Score dilution is formalized and proven; margin improvement from query updates is proven monotonic.
- **Compute-efficient**: Uses inference-time budget more effectively than thinking tokens; FLOP equivalence formula is simple and actionable.
- **Model-agnostic**: Works on top of any pre-trained transformer; evaluated on Qwen3-{1.7B, 4B, 8B, 32B} without any retraining or architecture changes.
- **Consistent gains**: Improvements hold across diverse tasks (code, dialogue, multi-doc QA, structured data) and both benchmarks.
- **Cache-friendly**: Single prefill, no KV cache invalidation, practical wall-clock times.

### Limitations
- **Task-dependent gains**: Summarization tasks benefit less -- when generation quality (not retrieval) is the bottleneck, reweighting attention yields limited returns.
- **Learning rate sensitivity**: Performance degrades at extreme LR values (too high causes instability, too low insufficient adaptation); optimal range is narrow (1e-6 to 1e-5).
- **Single (k, N_TTT) configuration**: Only one point on the budget trade-off surface was evaluated.
- **Thinking-token baseline only**: Main comparison is to chain-of-thought thinking tokens; self-consistency and best-of-N evaluated only in appendix (show mixed results).
- **Cost of prefill dominates**: For very long contexts, prefill cost dominates total latency regardless of method.

### Future Work (from paper)
1. Explore budget schedules across span size k and number of steps N_TTT.
2. Extend compute-matched comparisons to self-consistency and best-of-N within the same framework.
3. Develop simple predictors for when to prefer qTTT vs. decoding-based scaling.

## Reading Q&A Record

> 💡 **Q&A 批注记录**: Questions and answers collected during batch reading.

**Q1: Why does qTTT update only Q matrices and not K or V?**

A1: Updating K or V would invalidate the KV cache, forcing a full re-prefill on every step. Since the KV cache is the dominant cost for long contexts, this would make TTT prohibitively expensive. Q-only updates preserve the cache while still reshaping how queries attend to the evidence. Theoretically, Proposition 3.1 shows that moving q_i toward k_j* increases the target-distractor margin directly.

**Q2: Why do thinking tokens fail for long contexts?**

A2: Proposition 2.4 shows that the needle signal carried by any generated thinking token is bounded by its own attention mass on the needle. Under score dilution (small margin), this mass is provably tiny (Corollary 2.5). So thinking tokens cannot amplify the signal -- they just propagate the dilution. You cannot retrieve what you cannot attend to.

**Q3: Is qTTT compatible with existing long-context techniques like RoPE scaling?**

A3: Yes. Since qTTT operates at inference time and only modifies W_Q, it can be applied on top of any existing architecture/training strategy. The paper explicitly mentions compatibility with sliding window attention, adaptive positional encoding, training tweaks for longer windows, and RAG.

**Q4: What is the wall-clock time comparison between qTTT and thinking tokens?**

A4: Under FLOP-matched budgets, wall-clock times are very similar across all methods (Tables 10-12 in Appendix H). The prefill dominates most of the decoding time for longer sequences, which is why the frozen K/V design is critical.

**Q5: When does qTTT NOT help?**

A5: On summarization tasks (e.g., QMSum, SummScreen-FD), qTTT provides little to no gain over in-context. These tasks are more about generation quality than retrieval precision, so reweighting attention does not address the primary bottleneck. This suggests a "when-to-use" predictor would be valuable.

## Citation Landscape

Connected Papers: https://www.connectedpapers.com/main/2512.13898

Key related works cited:
- **Long-Context LLMs**: Reid et al. 2024 (Gemini 1.5), Ding et al. 2024 (LongRoPE), Anthropic 2024 (Claude 3)
- **Score Dilution / Lost in the Middle**: Liu et al. 2024, Kamradt 2024 (Needle-in-a-Haystack)
- **Test-Time Training**: Sun et al. 2020, Hardt and Sun 2024, Akyurek et al. 2024
- **Inference-Time Compute Scaling**: Wei et al. 2022 (CoT), Snell et al. 2024, Zelikman et al. 2024 (Quiet-STaR)
- **Long-Context Benchmarks**: LongBench-v2 (Bai et al. 2023b), ZeroScrolls (Shaham et al. 2023)
- **Positional Encoding**: RoPE (Su et al. 2024), ALiBi (Press et al. 2022), YaRN (Peng et al. 2024)
