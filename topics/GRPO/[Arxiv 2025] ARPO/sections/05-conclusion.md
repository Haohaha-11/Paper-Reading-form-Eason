[← 返回 README](../README.md)

# 5 Related Work & 6 Conclusion

## 📌 预览

Related Work 从两个主线梳理：(1) RLVR 在推理任务中的应用与发展；(2) Agentic RL 的演进与局限。Conclusion 简洁回顾 ARPO 的核心贡献与实验结论。

---

## 5 RELATED WORK

**Reinforcement Learning with Verifiable Reward**. Recently, Reinforcement Learning with Verifiable Rewards (RLVR) (Lambert et al., 2024; Kaufmann et al., 2025) has become a leading approach in Reinforcement Learning through Human Feedback (RLHF), particularly excelling in enhancing mathematical and programming reasoning (Shao et al., 2024; DeepSeek-AI et al., 2025; Yang et al., 2025; 2024; Team, 2024b;a; Dong et al., 2024c; Qiao et al., 2024). OpenAI o1 (OpenAI, 2024) first showcased RL's effectiveness in large-scale reasoning tasks. Building on this, models like DeepSeek R1 (DeepSeek-AI et al., 2025), QwQ (Team, 2024c), and Kimi k1.5 (Team et al., 2025) aim to replicate and surpass its performance. To improve RL algorithms' performance and stability, researchers have developed models like DAPO (Yu et al., 2025) and SimpleRLZoo (Zeng et al., 2025), exploring algorithm design across various RL modules (Hu et al., 2025; Yue et al., 2025; Feng et al., 2025b; Liu et al., 2025; Kool et al., 2019; Ahmadian et al., 2024; Dong et al., 2024a; Hu, 2025). Lin et al. identified key tokens affecting errors and showed that replacing them can alter model behavior. Studies (Gandhi et al., 2025; Li et al., 2025b) found RLVR primarily learns format over content, while several works (Vassoyan et al., 2025; Wang et al., 2025c; Cheng et al., 2025; Wang et al., 2025d) pointed out key tokens to high-entropy tokens to explore RL learning's essence. However, RLVR algorithms specifically for LLM agents remain underexplored. This paper uses entropy as a criterion to investigate reinforcement learning algorithms suited for LLM agent behavior.

> 💡 **问题动机**：
> - 这段 Related Work 确立了 ARPO 的学术定位：RLVR 领域已经发展出丰富的算法（GRPO、DAPO、REINFORCE++ 等），并积累了关于关键 token、格式学习 vs 内容学习、高熵 token 等洞察，但**专门针对 LLM Agent 的 RLVR 算法仍然不足**。
> - 与已有高熵 token 研究的区别：Wang et al. (2025c), Cheng et al. (2025) 等也研究过高熵 token，但他们关注的是"哪些 token 对学习更重要"，ARPO 的核心创新在于**利用熵变化作为动态采样信号**——这是从"分析和理解"到"设计和优化"的飞跃。

**Agentic Reinforcement Learning**. Reinforcement learning (RL) is essential for enabling LLM agents to adapt to dynamic and open environments (Lu et al., 2025; Shridhar et al., 2020; Mialon et al., 2024). Foundational works like DQN (Mnih et al., 2015) and AlphaZero (Silver et al., 2017) demonstrate that self-play-based RL can equip agents with capabilities from natural language understanding to strategic gameplay (Narasimhan et al., 2015). Building on this, value-based RL approaches have been employed to enhance embodied intelligence in hardware control and complex gaming tasks (Tan et al., 2024; Zhai et al., 2024; Bai et al., 2024; Wang et al., 2024; Schulman et al., 2017; Peng et al., 2019). Recent efforts, exemplified by RAGEN (Wang et al., 2025e; Zhou et al., 2024), integrates reasoning states and environmental interactions into turn-level responses using trajectory-level RL. To improve tool-integrated reasoning, studies (Jin et al., 2025a; Feng et al., 2025a; Song et al., 2025; Jin et al., 2025a; Chen et al., 2025b; Feng et al., 2025a; Li et al., 2025f; Sun et al., 2025a; Li et al., 2025e; Singh et al., 2025a) employ rule-based RL to teach LLMs how to autonomously invoke external tools (e.g. search engines, Python compilers) to boost reasoning accuracy. Further research, including ToolRL (Qian et al., 2025a), Tool-Star (Dong et al., 2025), and OTC (Wang et al., 2025b) explores multi-tool integration and tool-use efficiency. A series of works led by Kimi Deepresearcher and Websailor (Li et al., 2025c) optimize RL algorithms to better adapt to deepsearch's long context scenarios. While most works improve tool invocation through reward shaping and rollout mechanisms, simply applying trajectory-level RL fails to effectively capture the multi-turn, long-horizon characteristics of LLM-based agent behavior. This motivates the proposal of ARPO to attempt learning step-level tool-use behavior patterns.

> 💡 **机制拆解**：
> - Agentic RL 的研究路线可以归纳为三个阶段：
>   1. **基础 RL 赋能 Agent**（DQN, AlphaZero → LLM Agent）
>   2. **Rule-based RL 教 LLM 用工具**（Search-R1, Tool-Star, R1-Searcher, Search-o1 等）
>   3. **优化 RL 算法适应多轮 Agent 场景**（RAGEN, OTC, ToolRL 等）
> - ARPO 的独特贡献：大多数工作在第二阶段（让 LLM 学会用工具）或第三阶段（优化 reward shaping），但 ARPO 从根本上改变了**采样策略**——通过熵引导的分支采样，在算法层面适配了多轮工具交互的动态特性。
> - 与同类工作的区别：
>   - RAGEN：整合推理状态和环境交互到 turn-level 响应，但仍是 trajectory-level RL。
>   - Tool-Star：设计了多工具协作的 reward 机制（ARPO 沿用了这部分），但采样策略仍是传统的。
>   - OTC：优化工具调用效率，但通过 reward 设计而非采样策略。

---

## 6 CONCLUSION

In conclusion, we present Agentic Reinforced Policy Optimization (ARPO), an innovative reinforcement learning algorithm tailored for training multi-turn, LLM-based agents. Our experiments reveal that LLMs exhibit high token entropy after tool usage. ARPO leverages this by incorporating an entropy-based adaptive rollout mechanism, balancing global and step-level sampling to encourage diverse exploration in high-entropy tool-use phases. By integrating Advantage Attribution Estimation, ARPO enables LLMs to internalize advantage differences in stepwise tool-use interactions. Across 13 challenging benchmarks in computational reasoning, knowledge reasoning, and deep search domains, ARPO consistently outperforms traditional trajectory-level RL algorithms. Remarkably, it achieves great performance with only half the tool-use budget of other methods, offering a scalable solution for aligning LLM-based agents with dynamic environments.

> 💡 **Section 5-6 总结**：
> - **核心洞察**：
>   1. ARPO 填补了 RLVR 领域的一个重要空白：专门针对 LLM Agent 多轮工具交互的 RL 算法。
>   2. 与已有 Agentic RL 工作的关键区别：ARPO 从**采样策略**层面创新，而非 reward shaping 或 prompt 工程。
>   3. 从熵分析到算法设计再到理论证明，ARPO 提供了完整的 "observation → mechanism → theory" 闭环。
> - **可追问点**：
>   1. ARPO 能否与 reward shaping 方法（如 Tool-Star 的 multi-tool reward）进一步结合？
>   2. ARPO 的理论框架（GPG Theorem）能否推广到非 Transformer 架构？
>   3. 实际生产中，熵监控的计算开销是否成为瓶颈？
> - **关系定位**：本文在 GRPO 的基础上，利用工具调用后的熵增信号实现自适应分支采样，属于 GRPO 在 Agent 场景的增强版本，也是 GRPO topic 下的重要论文。
