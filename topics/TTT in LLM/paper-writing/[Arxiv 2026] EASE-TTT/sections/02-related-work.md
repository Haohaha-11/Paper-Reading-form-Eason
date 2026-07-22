[← 返回 README](../README.md)

# 2. Related Work（相关工作）

## 📌 预览

相关工作分三块：(1) Within-Context Retrieval and Evidence Selection——检索派现状与短板；(2) Test-Time Training——TTT/qTTT 的进展与局限；(3) Gap and Motivation——把两条线并置，点明各自解决问题的一半，EASE-TTT 用"检索结果当 TTT 监督"来桥接。Figure 2（方法总览图）虽排版出现在这一页，实际归属方法章节，已移至 03-methodology.md。

---

### Within-Context Retrieval and Evidence Selection.

Within-Context Retrieval and Evidence Selection. A common approach to long-context question answering is to localize question-relevant evidence within the input context before generation (Li et al., 2024a; Qiu et al., 2025; Lee et al., 2024). Unlike standard retrieval-augmented generation (Lewis et al., 2020), which retrieves passages from an external corpus, within-context retrieval treats the given long input itself as the retrieval source (Qian et al., 2024; Taguchi et al., 2025). Prior work has explored related strategies such as prompt compression, context pruning, discourse-based document selection, and hierarchical retrieval to reduce distractors and expose useful evidence to the model (Jiang et al., 2023; Zhao et al., 2024; Yoon et al., 2024). Efficiency-oriented variants also rely on selecting, compressing, or reorganizing input passages before generation (Xu et al., 2023; Pan et al., 2024). However, these methods treat evidence access mainly as an input-level operation: retrieved chunks are used to replace, shorten, reorder, or prepend to the original context. As a result, the model's parameters and context-access behavior remain unchanged. This is limiting when answer-bearing evidence is already present in the context window but is still not reliably accessed by the model. Moreover, hard selection can introduce a new bottleneck: selected chunks may omit useful surrounding context, separate evidence distributed across distant regions, or remove information needed to interpret the retrieved span (Günther et al., 2024; Tian et al., 2025). Thus, retrieval and prompt editing can change what the model sees, but they do not change how the model attends to and uses evidence in the full context.

> 💡 **机制拆解**: 这段把 within-context retrieval 与标准 RAG 划清界限——检索源是给定长输入本身，不查外部语料库。列举的技术谱系包括 prompt compression（LLMLingua 系）、context pruning（Provence）、discourse-based selection、hierarchical retrieval（RAPTOR 等）。作者用一句话点睛全派通病：**change what the model sees, but do not change how the model attends**（改的是"看到什么"而非"怎么看"）。加上 hard selection 的三宗罪——省略邻近上下文、切断远距离分布的证据、移除解读所需信息——共同构成 EASE-TTT 拒绝硬替换的理论依据。

### Test-Time Training.

Test-Time Training. Test-time training (TTT) improves model behavior at inference time by updating parameters using self-supervised signals derived from the test input itself (Hu et al., 2025; Zhang et al., 2025). These approaches have been explored in settings such as distribution shift, domain adaptation, and reasoning-time adaptation, where fixed pretrained parameters may be insufficient for the input at hand (Hübotter et al., 2025; Agarwal et al., 2025; Li et al., 2025). In the long-context setting, TTT is especially relevant because each test instance may exhibit different local structures, evidence layouts, and distraction patterns (Muhtar et al., 2024). However, parameterlevel adaptation alone does not solve evidence access unless the training signal is aligned with the evidence required by the current question. Applying TTT to long contexts is therefore nontrivial: the adaptation signal is often local, partial, and potentially noisy, while broad parameter updates may introduce instability or unnecessary computational overhead (Su et al., 2023; Zhang et al., 2024). These challenges make targeted and evidence-aligned test-time training important for long-context inference. Query-only test-time training (qTTT) narrows the update to the query projections in self-attention rather than adapting the full model (Bansal et al., 2025). However, qTTT still relies on generic self-supervised objectives rather than explicit supervision from question-relevant evidence. As a result, it can update query-side attention parameters, but it does not specify which full-context positions should guide the update. This creates a mismatch for long-context QA: the model is adapted, but the adaptation is not anchored to the evidence needed to answer the question.

> 💡 **机制拆解**: 这段梳理 TTT 谱系并把矛头精准指向 qTTT。关键逻辑链：(1) TTT 用测试输入自身派生的自监督信号更新参数，适合处理每个实例不同的 local structure / evidence layout / distraction pattern；(2) 但 broad parameter update 有 instability 和 overhead 风险——这是 qTTT 只更新 query projection 的动机；(3) qTTT（Bansal 2025）把更新收窄到 self-attention 的 query 投影，效率高，但仍用 generic self-supervised objective，**does not specify which full-context positions should guide the update**。作者用 "the model is adapted, but the adaptation is not anchored to the evidence" 一句锁定 qTTT 的缺口——这正是 EASE-TTT 要补的 anchor。

### Gap and Motivation.

Gap and Motivation. These two lines of work address different sides of the long-context evidenceaccess problem, but neither resolves it alone. Within-context retrieval and prompt editing operate at the input level: they can localize or expose candidate evidence, but they leave the model's contextaccess behavior unchanged. This is insufficient when the relevant content is already inside the context window but the model fails to attend to it. Query-only test-time training operates at the parameter level: it can adapt query-side attention behavior, but its objectives are not tied to the evidence positions required by the current question. Consequently, existing methods either select evidence without adapting the model, or adapt the model without explicit evidence guidance. Our method bridges this gap by using retrieved evidence chunks not as a replacement for the full context, but as supervision for query-side test-time training. The final answer is still generated from the original full context, while the retrieved evidence guides how the model updates its attention behavior.

> 💡 **问题动机**: 这段是全文 gap 论证的收口，用对偶结构把两派并置：检索派 operate at input level（选证据但不改模型）、qTTT operate at parameter level（改模型但不锚定证据位置）。结论精炼成一句 "either select evidence without adapting the model, or adapt the model without explicit evidence guidance"。EASE-TTT 的桥接方式——把检索块当 supervision 而非 replacement，最终仍从 full context 生成——在这里第三次被强调（摘要、引言、相关工作各出现一次），说明作者把"不替换 + 全上下文生成"当作最核心的卖点。

## 🔖 Section 总结

### 核心洞察
1. **两派各解决一半**：检索派改"看到什么"，qTTT 改"参数"，但都没做到"用问题证据锚定参数更新"。
2. **hard selection 三宗罪**：省略邻近上下文、切断分布式证据、移除解读信息——这是 soft target 存在的必要性。
3. **qTTT 是最近的对标**：同为 query-side adaptation，唯一差异是监督信号（generic span NTP vs evidence-aligned attention KL）——所以效率分析（Table 2）和主消融（Figure 3）都拿 qTTT 当对照。

### 可追问点
- qTTT 的 generic 目标具体长什么样？→ 03 节 Eq.(公式1) 的 span NLL loss。
- "改注意力行为"如何量化验证？→ 论文没有直接可视化注意力图，主要靠下游任务分数间接支撑。
