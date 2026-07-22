[← 返回 README](../README.md)

# 0. Abstract（摘要）

## 📌 预览

这篇摘要要解决的核心痛点：**长上下文 QA 里答案证据其实已经在输入中，但小模型仍然读不到**。作者主张把"检索"从"改输入"升级为"改模型注意力参数"，方法叫 EASE-TTT——用检索到的证据块构造一个**软注意力监督目标**，在测试时只更新 query 侧的轻量适配器，最终答案仍从**完整原始上下文**生成。

---

Long-context question answering (QA) remains challenging for smaller language models even when answer-bearing evidence is already present in the input. Existing withincontext retrieval methods localize and expose candidate evidence chunks for the question, but they stop at input-level evidence exposure rather than adapting the query-side attention pa rameters that control how the model allocates attention over full-context positions. In con trast, lightweight test-time adaptation methods, such as query-only test-time training (qTTT), leave evidence localization unresolved because their generic span-level self-supervised objectives do not identify which context positions support the current answer. In this paper, we propose Evidence-Aligned SElective Test-Time Training (EASE-TTT), a within-context retrieval-augmented test-time training framework that converts selected evidence chunks into a soft attention supervision target over their token positions. Instead of replacing the full context with retrieved chunks, EASE-TTT uses the resulting attention target to guide queryside adaptation, with the adapted model generating the final answer from the original full context. Experiments on six LongBench QA tasks and three small decoder-only language models show that EASE-TTT achieves the strongest macro-average performance among full-context inference, retrieval-only baselines, and qTTT, supporting evidence-aligned testtime adaptation in long-context QA.

> 💡 **问题动机**: 摘要开门见山点出一个反直觉现象——答案证据 already present in the input，但小模型 still fails。这把矛盾从"上下文窗口不够长"转移到"证据访问不可靠"。作者随后把已有方案分成两派并各打五十大板：
> - **within-context retrieval（检索派）**：只做 input-level evidence exposure（定位/暴露候选块），但不改 query-side attention 参数——即不改变模型"怎么分配注意力"。
> - **qTTT（测试时训练派）**：能改 query 侧参数，但用的是 generic span-level self-supervised 目标（随机采样片段做下一个词预测），不知道哪些位置真正支撑当前答案，所以 evidence localization 仍未解决。

> 💡 **机制拆解**: EASE-TTT 的核心动作是"把检索结果翻译成监督信号"。具体三步：(1) 选出与问题最相关的证据块；(2) 把这些块覆盖的 token 位置转成一个 **soft attention target**——给证据位置分配大部分概率质量，同时对其余上下文保留非零质量（这是"soft"的关键，避免硬丢弃）；(3) 冻结 base model，只更新 query 侧适配器，让真实注意力去逼近这个目标分布。关键设计：**Instead of replacing the full context with retrieved chunks**——检索块只当监督信号，最终生成仍用完整上下文，因此不会因硬截断丢掉分散在别处的证据。

> 💡 **证据支撑**: 摘要给出的实验规模是 six LongBench QA tasks × three small decoder-only LMs，声称拿到 strongest macro-average，超过三类对手：full-context inference、retrieval-only baselines、qTTT。注意这里 claim 的是 **macro-average 最强**，不是每个任务都最强——正文 Table 1 里确实存在个别任务被 baseline 反超的情况（如 Llama 上的 MuSiQue、NarrativeQA），批读实验章节会具体拆。
