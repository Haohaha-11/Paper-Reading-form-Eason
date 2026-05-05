[← 返回 README](../README.md)

# Experiments

## 📌 预览
本文件合并 Experiments/Results/Analysis/Ablation，重点看方法是否真的提升视觉 grounding、医学诊断、鲁棒性和效率。

---

# 4. Experiments

# 4.2. Main Results

The main experimental results demonstrate that our proposed memory system VisMem unlocks the untapped potentials with three key enhancements: $I E n h . I J$ advanced visual capabilities, [Enh.2] cross-domain generalization, [Enh.3] catastrophic forgetting alleviation.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

[Enh.1] VisMem enables advanced and comprehensive visual capabilities. As presented in Tab. 1, our proposed method demonstrates distinct superiority over other baseline models. Compared with the vanilla model, Vis-Mem achieves a notable average improvement of $1 1 . 0 \%$ across all benchmarks. When compared with the top three baselines (i.e., Vision-R1 [26], VLM-R1 [44], and Open-ThinkImg [49]), our method still maintains improvements of $3 . 0 \%$ , $4 . 2 \%$ , and $4 . 9 \%$ , respectively. Furthermore, it consistently enhances performance across the three core domains of visual tasks, namely, understanding, reasoning, and generation. Our latent vision memory mechanism yields comprehensive enhancements in visual capabilities, with specific gains of $+ 8 . 9 \%$ in visual understanding, $+ 1 4 . 4 \%$ in reasoning, and $+ 1 0 . 6 \%$ in generation, relative to the vanilla model. It is also noteworthy that direct RL-based methods (e.g., VLM-R1 [44] and Vision-R1 [26]) also achieve relatively better performance than most other paradigms. However, this approach of directly modifying parameters relies on incremental parameter updates, which may lead to the overwriting of prior general knowledge and result in catastrophic forgetting.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Figure 3.](../images/2f9f9cbf762c37b6cb1521fb2b9d575bce8d0f3aaa3a47e021f5bbc8caf4f751.jpg)
*Figure 3.: Figure 3. Results of the cross-domain generalization study. Models are only trained on Visual CoT [42] and Mulberry [71]. Dashed bar indicates the results with full training data.*

> 💡 **Figure 3. 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

As illustrated in Tab. 5 and 6, we conduct additional evaluations on selected subsets of MuirBench [57] and LogicVista [67]. Endowed with short- and long-term vision memory, our VisMem outperforms all baseline methods by a substantial margin in tasks demanding fine-grained visual evidence, such as counting $( + 7 . 0 \% )$ , visual retrieval $( + 9 . 4 \% )$ , and grounding $( 1 3 . 1 \% )$ , while also yielding notable improvements in visual reasoning tasks, including inductive $( + 5 . 7 \% )$ and deductive $( + 7 . 1 \% )$ learning.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

[Enh.2] VisMem showcases great cross-domain generalization. To evaluate the cross-domain generalization capability of our model, specifically whether its stored latent visual memory can transfer across diverse unseen tasks, we exclusively train our VisMem and comparative baseline models on two datasets: Visual CoT [42] and Mulberry [71], then subsequently assess their performance on four unseen target benchmarks. As demonstrated in Fig. 3, 7, and Tab. 7, VisMem not only consistently achieves significant performance gains on out-of-domain tasks $( + 6 . 9 \%$ on MMVet [76], $+ 9 . 1 \%$ on MuirBench [57], $+ 2 0 . 2 \%$ on MV-Math [62], and $+ 9 . 9 \%$ on MultiTrust [82]), but also maintains leading performance relative to all baselines. Notably, our method outperforms the second-ranked model by a substantial margin of $2 . 7 \mathrm { - } 6 . 8 \%$ across all four benchmarks, while narrowing the performance gap relative to results obtained with full training data. This observation underscores its robust cross-domain knowledge transfer capability.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Table 2. Results on nine base models with various sizes and sources, including Qwen2.5-VL-3B/7B/32B [4], LLaVA-OV-1.5-4B/8B [1], InternVL-3.5-4B/8B/14B/38B [63]. $\uparrow$ indicates the performance enhancement compared with the base model.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

![Table 6.](../images/91b363ac799ee7782387bca515d7b008e03e7f1579eda85f0ec0d8ac76826dd3.jpg)
*Table 6.: Table 6. Results on 10 selected subsets (5 reasoning skills and 5 capabilities) of LogicVista [67]. We compare our VisMem with the second and third best scored counterparts, and separately use the short or long latent memory to assess the improvements of each.*

> 💡 **Table 6. 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

![Figure 4.](../images/f9b367394263fee2b8e04e0d6ea3a987faf0ba52875dd5879fe4e306c1e32352.jpg)
*Figure 4.: Figure 4. Results of four-stage continual learning on MMVet [76]. Stage 0 only includes itself, while stage 1, 2, 3 sequentially train models on different additional training data combinations.*

> 💡 **Figure 4. 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

[Enh.3] VisMem alleviates catastrophic forgetting. As illustrated in Fig. 4, 8, and Tab. 8, we conduct sequential training of the models across four stages, with performance assessed on MMVet [76] after each stage. At stage 0, the model was trained exclusively on the base task, and in subsequent stages, we incrementally incorporated selected benchmarks into the training process. From the continual learning results, our VisMem demonstrates significantly stronger knowledge retention capabilities. Although direct training paradigms yield relatively excellent overall performance in offline learning tasks with once-off training, they suffer from severe catastrophic forgetting. For instance, SFT exhibits over $10 \%$ performance degradation throughout the training process, the highest among all baselines. Additionally, at stage 0, VLM-R1 [44] and Vision-R1 [35] achieve performance improvements of $1 1 . 8 \%$ and $1 0 . 9 \%$ respectively compared to the vanilla model, however, these improvements are retained by less than $0 . 5 \%$ at stage 4. In contrast, our method effectively mitigates catastrophic forgetting, exhibiting the smallest performance gap relative to original full-data training among all baselines. It is further worth noting that our latent vision memory enhances performance at stages 1 and 3 without any degradation, reflecting superior cross-task generalization.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

# 8. Experiment Details

# 8.2. Benchmarks

To comprehensively evaluate the performance of the selected baselines, we involve 12 benchmarks, consisting of 5 benchmarks of understanding, 4 benchmarks of reasoning, and 3 benchmarks of generation:

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

LogicVista [67] evaluates general logical cognition abilities across 5 logical reasoning tasks, which encompass 9 distinct capabilities. Each question is annotated with the correct answer and the human-written reasoning behind the selection.

HallBench [19] consists of images paired with questions, designed by human experts to assess the hallucination level of generation.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

• MMStar [7] is a high-quality vision-centric benchmark meticulously curated by human experts. This benchmark assesses 6 core capabilities across 18 detailed axes of visual understanding.   
• MMVet [76] establishes 6 core visual understanding capabilities and investigates 16 critical integrations derived from their combinations. It uses an evaluator tailored for open-ended outputs.   
• MMT [73] consists of carefully curated multi-choice visual questions, covering 32 core meta-tasks and 162 subtasks within the field of visual understanding.   
• BLINK [15] reconstructs 14 classic computer vision tasks into multiple-choice questions. Each question is paired with either single or multiple images and supplemented with visual prompting.   
• MuirBench [57] covers 12 diverse multi-image tasks, which involve 10 categories of multi-image relations. Each standard instance is paired with an unanswerable variant that differs only minimally in semantics.   
• MMMU [79] comprises meticulously curated visual questions sourced from college exams, quizzes, and textbooks spanning 30 subjects and 183 subfields, which focus on advanced reasoning grounded in domain-specific knowledge.   
• MathVista [59] unifies the challenges of heterogeneous mathematical and visual tasks, which are curated from math-oriented multimodal datasets.   
• MV-Math [62] is a dataset comprising mathematical problems, integrating multiple images interleaved with text, and detailed annotations. It features multiple-choice, free-form, and multi-step questions across 11 subject areas at 3 difficulty levels.   
• MultiTrust [82] covers five primary aspects: truthfulness, safety, robustness, fairness, and privacy, evaluating the trustworthiness of generation.   
• MMVU [34] encompasses 12 categories, and designs evaluation metrics that measure the quality and error degree of generation.

> 💡 **列表批读**: 这组条目通常对应贡献、设置、发现或模块；建议逐条映射到 visual memory / latent reasoning / medical diagnosis 的主线。

# 9. Additional Results

# 9.1. Benchmark Subset Results towards Visual Subcapacities

To precisely identify the capability boundaries and advantages of our VisMem, rather than relying solely on overall scores to judge its quality, we evaluate the results of subsets of MuirBench [57] and LogicVista [67] benchmarks. We select 9 subsets of the former benchmark, including: counting, grounding, matching, scene, difference, cartoon, diagram, geographic, and retrieval. While in the latter benchmark, we also select 10 subsets, including 5 reasoning skills: inductive, deductive, numerical, spatial, and mechanical, and 5 capacities: patterns, puzzles, OCR, graphs, and tables. It is worth noting that the selected subsets are only part of the benchmark, thus, the average values of the 10 subsets are not the results of the benchmarks.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Table 4. Configurations of parameters.

![Table 7.](../images/0db04ebf092289144e5076be75b49c79a661d1e45481680b1e8985483f87f7ce.jpg)
*Table 7.: Table 7. Results of various models with full training datasets and partial datasets (Visual CoT [42] and Mulberry [71]), and evaluated across four benchmarks.*

> 💡 **Table 7. 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

Table 5. Results on 9 selected subsets of MuirBench [57]. We compare our VisMem with the second and third best scored counterparts, and separately use the short or long latent memory to assess the improvements of each.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Table 8.](../images/0d4ef835faab2ebe588b7955ea0de3cb774ec0c2613b713391bff985de952b7f.jpg)
*Table 8.: Table 8. Results of various models on MMVet [76] with four-stage continual learning. Stage 0: MMVet [76]; Stage 1: BLINK [15], and MuirBench [57]; Stage 2: LogicVista [67], and Math-V [59]; Stage 3: MultiTrust [82], and MMVU [34].*

> 💡 **Table 8. 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

As listed in Tab. 5, compared with VLM-R1 [44] and Vision-R1 [26], our VisMem achieves the best results on 7 subsets and ranks second on the remaining two subsets. Specifically, it has a generalized enhancement of at least $5 \%$ over the base model. Besides, VisMem improves the performance the vanilla model by $1 6 . 7 \% / 1 8 . 2 \% / 1 1 . 8 \% / 1 3 . 7 \%$ on the counting, grounding, geographic, and retrieval subtasks, vastly exceeding the second-best counterpart by 7.0- $1 3 . 1 \%$ . These results indicate that our latent vision memory system significantly promote the fine-grained visual cognition and perception of the base VLMs.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

As presented in Tab. 6, our VisMem outperforms two baseline models, i.e., VLM-R1 [44] and Vision-R1 [26], by achieving the top performance across 8 subsets. Specifically, it delivers a generalized improvement of no less than $7 \%$ over the base model. Notably, on inductive, deductive, graph-based, and table-based sub-tasks, VisMem surpasses the vanilla model by $1 4 . 8 \%$ , $1 4 . 8 \%$ , $1 8 . 4 \%$ , and $2 1 . 1 \%$ , respectively, which exceeds the second-ranked model by a substantial margin of $5 . 3 \mathrm { - } 7 . 1 \%$ . These results demonstrate that our latent visual memory system delivers contextualized semantic knowledge, thereby enhancing visual reasoning and robust generation capabilities.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Table 6. Results on 10 selected subsets (5 reasoning skills and 5 capabilities) of LogicVista [67]. We compare our VisMem with the second and third best scored counterparts, and separately use the short or long latent memory to assess the improvements of each.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Table extracted](../images/daa8c548224cc4f987898df95ca1d6a0aa2662f857a330390e24c4fef1aeaf3e.jpg)
*Table extracted: Table extracted by MinerU. Ablation MMVet MuirBench MV-Math MultiTrust Time Speed Perf. Time Speed Perf. Time Speed Perf. Time Speed Perf. Vanilla 0.76 1.32 66.0 3.79 0.26 57.4 5.47 0.18 18.9 3.62 0.28 64.8*

> 💡 **Table extracted 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

# 9.5. Ablation Study

The vanilla model establishes a baseline characterized by the shortest inference time and highest speed across all benchmarks, yet exhibits the lowest performance. This confirms that latent vision memory is indispensable for enhancing task performance. For the random memory invocation variants, increasing the invocation probability $( 2 5 \% - 1 0 0 \% )$ results in longer inference time and reduced speed. Performance peaks at a $7 5 \%$ probability before declining, indicating that excessive memory invocation impairs efficiency without yielding additional performance benefits. Ablation studies of the short-term and long-term memory components reveal task-specific advantages: the short-term memory component outperforms on MuirBench [57] and MultiTrust [82], while the long-term component demonstrates superior performance on MV-Math [62]. Notably, the complete VisMem framework achieves the highest performance across all benchmarks, validating the value of integrating dual-component vision memory for balanced and robust visual capacities.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Figure 7.](../images/935456cab9e457ca8c498bac3558c11b82e259f998a902d7f03f12aa0e2cbb2b.jpg)
*Figure 7.: Figure 7. Results of various models of the cross-domain generalization study. Models are only trained on Visual CoT [42] and Mulberry [71], and are evaluated on four benchmarks.*

> 💡 **Figure 7. 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 8.](../images/90cdfb12671cf06cc0932619e72bdf2ab681e68e9bb8f46f06ab436fc9bafffb.jpg)
*Figure 8.: Figure 8. Results of four-stage continual learning on MMVet [76]. The model is sequentially trained on each training data combination (Stage $0 $ Stage $1 $ Stage $2  \mathrm { S t a g e } 3$ ). Stage 0 only includes MMVet [76] as training data, while Stage 1, 2, 3 add data targeting visual understanding [15, 57], reasoning [59, 67], and generation [34, 82].*

> 💡 **Figure 8. 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Table 9. Ablations of latent vision memory invocation and dual vision memory formation. Following [81], “Random Invocation” denotes that the latent memory is inserted into the output sequence with a certain probability when outputting delimiter symbol tokens, and short or long latent memory is inserted with equal probability. When only utilizing short or long latent memory, we directly skip the formation of the specific memory if invocation tokens are predicted and continue the process of decoding.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Table 10.](../images/4a9fb190d745ba3276736c742da38da5820cd87d9d7a8613b82c799f78ad5e8b.jpg)
*Table 10.: Table 10. Results of different length of memory query $K$ .*

> 💡 **Table 10. 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

# 9.6. Analysis of Latent Vision Memory

We visualize the invocation ratio and relative invocation position, as presented in Fig. 5 and 9: the former illustrates benchmark-specific differences between the two memory components, while the latter depicts type-specific variations across the four benchmarks. In addition, as reported in Tab. 5 and 6, the short- and long-term latent visual memory components exhibit task-specific advantages for different visual sub-tasks. For instance, the short-term memory provides supplementary visual information to support enhanced visual understanding, such as counting, grounding, and visual retrieval. By contrast, the long-term memory encodes contextualized semantic knowledge, which strengthens complex visual reasoning. These results reveal that our proposed VisMem dynamically adjusts invocation position and frequency according to task characteristics, thereby balancing efficiency and performance.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Figure 9.](../images/0469fa78da88af59e7984c4815ab0a81b0c92c878208e177b114d2e8f150e7cd.jpg)
*Figure 9.: Figure 9. Results of memory invocation ratio and relative position across four benchmarks. The former denotes the proportion of invoked samples to all samples, while the relative position denotes the position in the whole output sequence when the invocation occurred. We apply gaussian smoothing to the curves to highlight their main trends.*

> 💡 **Figure 9. 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 10.](../images/558207ea6479ed202949460a83c99891c6f94cfcf34f1f7c033df9a22ab073fe.jpg)
*Figure 10.: Figure 10. Results of sensitivity analysis on the sequence length of memory query $K$ , short- and long-term memory $N _ { s }$ and $N _ { l }$ .*

> 💡 **Figure 10. 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Table 10. Results of different length of memory query $K$ .

![Table 11.](../images/5c45b592dc82eb1cae86c901c5462703ba1edf913349510a7a8ae227a19a5e4d.jpg)
*Table 11.: Table 11. Results of different length of short latent vision memory $N _ { s }$ and the length of long latent vision memory $N _ { l }$ across four benchmarks.*

> 💡 **Table 11. 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

# 9.7. Sensitive Analysis of Sequence Lengths

We conduct an analysis on MMVet [76] focused on the lengths of three key sequences: the memory query $K$ , the short-term latent visual memory $N _ { s }$ , and the long-term latent visual memory $N _ { l }$ . It is observed that as the lengths of these three sequences increase from 2 to 32, model performance improves accordingly, but this is accompanied by increased computational costs.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Table 11. Results of different length of short latent vision memory $N _ { s }$ and the length of long latent vision memory $N _ { l }$ across four benchmarks.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Table 12.](../images/18ad46c3f408453c239af77e49d6ab56e78aad03cdff530821676f8a01846aa9.jpg)
*Table 12.: Table 12. Average inference time per sample (seconds), average inference speed (samples / seconds), and task performances across four benchmarks on various methods. Perf. indicates Performance.*

> 💡 **Table 12. 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

---

## 🔖 Section 总结

### 核心洞察
1. 检查 benchmark 是否覆盖医学/跨域/鲁棒性。
2. 关注消融、效率和失败案例。
