[← 返回 README](../README.md)

# 2 Preliminary

## 📌 预览

Preliminary 提供三部分基础：(1) Agentic RL 的数学形式化定义（目标函数 + 轨迹分解）；(2) Token 熵计算与最关键的先导实验——证明工具调用后熵急剧上升；(3) 本文使用的三类工具介绍。这个 Section 的核心价值在于用实验证据（Figure 2）支撑 ARPO 的核心设计动机。

---

## 2 PRELIMINARY

Before introducing ARPO, we first provide a brief overview of key concepts and review preliminary entropy-based experiments on LLM reasoning with tools.

---

## 2.1 AGENTIC REINFORCEMENT LEARNING

In this section, we formulate the agentic RL training objective as:

$$
\operatorname* { m a x } _ { \pi _ { \theta } } \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot | x ; T ) } [ r _ { \phi } ( x , y ) ] - \beta \mathbb { D } _ { \mathrm { K L } } [ \pi _ { \theta } ( y \mid x ; T ) ] \mid \pi _ { \mathrm { r e f } } ( y \mid x ; T ) ]
$$

where $T$ denotes the set of available tools, $\pi_{\theta}$ represents the policy LLM, $\pi _ { \mathrm { r e f } }$ is the reference LLM, $r _ { \phi }$ and $\mathbb { D } _ { \mathrm { K L } }$ denotes the reward function and KL divergence respectively. The input x is sampled from dataset $\mathcal { D }$, and $y$ is the corresponding output, possibly interleaved with tool-call feedback.

Unlike conventional RL methods that rely solely on LLM rollouts, agentic RL incorporates tool-call feedback during the reasoning process (Chen et al., 2023; Gou et al., 2024; Li et al., 2025f; Wu et al., 2025c; Li et al., 2024a). The rollout sampling can be decomposed as:

$$
P _ { \theta } ( \mathcal { R } , y \mid x ; T ) = \underbrace { \prod _ { t = 1 } ^ { t _ { \mathcal { R } } } P _ { \theta } ( \mathcal { R } _ { t } \mid \mathcal { R } _ { \lt t } , x ; T ) } _ { \mathrm { A g e n t i c ~ R e a s o n i n g } } \cdot \underbrace { \prod _ { t = 1 } ^ { t _ { y } } P _ { \theta } ( y _ { t } \mid y _ { \lt t } , \mathcal { R } , x ; T ) } _ { \mathrm { A n s w e r ~ G e n e r a t i o n } }
$$

where $\mathcal { R }$ is the reasoning trajectory of length $t _ { \mathcal { R } }$, interleaved with tool-call feedback, and $y$ is the final answer with length $t _ { y }$. Our ARPO is built upon rule-based RL algorithm (e.g. GRPO (Shao et al., 2024), Reinforce++ (Hu, 2025)) designed to optimize LLM-based agents.

> 💡 **公式批读**：
> - **Eq. 1 (目标函数)**：标准 RL objective，但注意条件分布中显式加入了工具集 $T$——这是 Agentic RL 和标准 RL 的关键区别。与单轮 RL 不同，这里的 $y$ 不是纯粹的模型输出，而是**与工具反馈交织**的产物。
> - **Eq. 2 (轨迹分解)**：将 rollout 分解为两个阶段——(1) Agentic Reasoning（模型生成指令 + 接收工具反馈），长度为 $t_{\mathcal{R}}$；(2) Answer Generation（基于收集到的信息生成最终答案），长度为 $t_y$。这个分解对理解 ARPO 的分支采样位置至关重要：ARPO 主要在 Agentic Reasoning 阶段进行熵监控和分支。
> - 重要约束：ARPO 基于 rule-based RL 算法（如 GRPO/Reinforce++），即使用可验证的 reward 而非 reward model。

---

## 2.2 ANALYZING TOKEN ENTROPY IN AGENTIC REASONING

Token Entropy Calculation. Following recent entropy-based RL studies (Wang et al., 2025c;d; Cheng et al., 2025; Zheng et al., 2025), we compute the token-level generation entropy at step t as:

$$
H _ { t } = - \sum _ { j = 1 } ^ { V } p _ { t , j } \log p _ { t , j } , \quad \mathrm { w h e r e } p _ { t } = \pi _ { \theta } \left( \cdot \mid \mathcal { R } _ { \lt t } , x ; T \right) = \mathrm { S o f t m a x } \left( \frac { z _ { t } } { \tau } \right)
$$

Here, $V$ is the vocabulary size, $z _ { t } \in \mathbb { R } ^ { V }$ is the pre-softmax logits, and $\tau$ is the decoding temperature. Note that this entropy reflects the uncertainty in the token generation distribution, rather than the uncertainty of any particular token.

> 💡 **公式批读**：
> - Eq. 3 是标准的分类分布熵公式。关键参数是温度 $\tau$：$\tau$ 越高，分布越平滑，熵越大。这里计算的是**整个词表上的分布熵**，而不是"下一个 token 的确定程度"——这很重要，因为高熵意味着模型在多个可能的后续 token 间犹豫，这正是探索的价值所在。
> - 特别留意：$p_t$ 的条件是 $\mathcal{R}_{\lt t}, x, T$，即**考虑了之前的推理轨迹和可用工具集**。在工具调用后，这个条件分布会因外部信息的注入而发生剧烈变化。

Pilot Experiment on Token Entropy. To gain deeper insights into the reasoning process of LLM-based tool-use agents, we conduct a pilot study with two types of agents: one using a search engine for knowledge-intensive tasks and another using a Python interpreter for computational tasks. We measure token entropy variations throughout the reasoning process to assess uncertainty.

As shown in Figure 2, our key observations are: (1) Entropy rises sharply in the first 10-50 tokens following each tool call; (2) Entropy tends to increase during early reasoning stages, but remains lower than after receiving tool-call feedback; (3) Search feedback introduces more uncertainty than Python feedback.

> 💡 **Figure 2 批读**：
> - 三个观察逐条解读：
>   - **Ob.1 (工具调用后 10-50 token 熵急剧上升)**：这是 ARPO 最核心的经验依据。10-50 token 大约对应模型"消化"工具返回信息并规划下一步行动的阶段。
>   - **Ob.2 (早期推理阶段熵也升高，但低于工具反馈后)**：说明输入问题本身就引入了一定不确定性，但外部工具反馈带来的分布偏移远超原始输入。
>   - **Ob.3 (搜索反馈 > Python 反馈)**：搜索返回的是丰富、多变的自然语言文本，Python 返回的是确定性数值或错误信息。这解释了为什么知识密集型搜索任务的 ARPO 收益更大。

We attribute these effects to the distributional shift between external feedback and the model's internal reasoning (Ob.1), which introduces uncertainty often exceeding that of the original input (Ob.2). Furthermore, search engines typically return informative textual content, whereas Python outputs consist of deterministic numbers, resulting in greater entropy fluctuations in the former case (Ob.3).

These findings highlight a limitation of trajectory-level RL methods, which focus on initial reasoning while overlooking the uncertainty introduced by tool-call feedback. Our proposed ARPO algorithm addresses this by incorporating entropy-based exploration tailored to LLM agent training.

![Figure 2: Analysis of token entropy variations and token frequency statistics](../images/c6939f16bf431d69516a6b8e4e0238a194802205e3128ae9867a3183a8153fd8.jpg)

*Figure 2: Analysis of token entropy variations and token frequency statistics of LLM-based tool-use agent across different datasets.*

> 💡 **Figure 2 批读**：
> - 上图展示了不同数据集上的 token 熵变化模式。横轴是推理步骤（时间），纵轴是熵值。每次工具调用后紧跟的尖峰清晰可见。
> - 注意不同颜色的曲线代表不同的数据集/任务，普遍在工具调用后出现熵峰——证明这不是某个特定任务的巧合，而是多轮 Agent 推理的普遍特性。

---

## 2.3 AGENTIC TOOL DESIGN

In this work, we mainly focus on optimizing the training algorithms of LLM-based tool-use agents. After a comprehensive review of agentic RL studies (Dong et al., 2025; Feng et al., 2025a; Jin et al., 2025a), we identify three representative tools to empirically evaluate the effectiveness of ARPO:

- **Search Engine**: Retrieves relevant information by executing queries across the web.

- **Web Browser Agent**: Accesses and parses relevant web links returned by the search engine, extracting and summarizing key content.

- **Code Interpreter**: Automatically executes code generated by the language model, returning execution results if successful, or compiler error messages otherwise.

> 💡 **问题动机**：
> - 三类工具的选择不是随意的。搜索返回非结构化文本（高熵变化），浏览器进一步解析网页内容（中等熵变化），代码解释器返回确定性结果（低熵变化）。这三类工具覆盖了不同程度的"工具反馈不确定性"梯度，为验证 ARPO 的熵自适应机制提供了理想的测试场景。
> - 注意：ARPO 不假设特定的工具实现，工具的设计和实现是独立的。ARPO 只关注训练算法层面。

---

> 💡 **Section 2 总结**：
> - **关键公式**：Eq. 1（Agentic RL 目标）、Eq. 2（轨迹分解）、Eq. 3（token 熵计算）。
> - **核心洞察**：
>   1. Agentic RL 的 rollout 可分解为 Agentic Reasoning + Answer Generation 两阶段，ARPO 主要在第一个阶段进行熵监控和分支。
>   2. 工具调用后的熵增是普遍现象，搜索引擎的熵增比 Python 更大。
>   3. Trajectory-level RL 只关注初始推理，忽视工具交互引入的不确定性——这是 ARPO 要解决的核心问题。
> - **可追问点**：为什么搜索反馈比 Python 反馈熵增更大？这对 ARPO 的超参数设置有什么影响？
