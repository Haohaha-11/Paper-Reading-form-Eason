[← 返回 README](../README.md)

# Abstract

## 📌 预览

本文提出 ARPO（Agentic Reinforced Policy Optimization），面向多轮 LLM Agent 训练的强化学习算法。核心动机：工具调用后 token 熵剧烈上升，trajectory-level RL 忽视了这一现象；解决方案：熵自适应 rollout + Advantage Attribution Estimation。

---

AGENTIC REINFORCED POLICY OPTIMIZATION

Guanting Dong<sup>1*</sup>, Hangyu Mao<sup>2</sup>, Kai Ma<sup>2</sup>, Licheng Bao<sup>2*</sup>, Yifei Chen<sup>1</sup>, Zhongyuan Wang<sup>2*</sup>   
Zhongxia Chen<sup>2</sup>, Jiazhen Du<sup>2</sup>, Huiyang Wang<sup>2*</sup>, Fuzheng Zhang<sup>2</sup>, Guorui Zhou<sup>2+</sup>   
Yutao Zhu<sup>1</sup>, Ji-Rong Wen<sup>1</sup>, Zhicheng Dou<sup>1+</sup>   
<sup>1</sup>Renmin University of China, <sup>2</sup>Kuaishou Technology   
{dongguanting, dou}@ruc.edu.cn

GitHub: https://github.com/dongguanting/ARPO

---

## ABSTRACT

Large-scale reinforcement learning with verifiable rewards (RLVR) has demonstrated its effectiveness in harnessing the potential of large language models (LLMs) for single-turn reasoning tasks. In realistic reasoning scenarios, LLMs can often utilize external tools to assist in task-solving processes. However, current RL algorithms inadequately balance the models' intrinsic long-horizon reasoning capabilities and their proficiency in multi-turn tool interactions. To bridge this gap, we propose Agentic Reinforced Policy Optimization (ARPO), a novel agentic RL algorithm tailored for training multi-turn LLM-based agents. Through preliminary experiments, we observe that LLMs tend to exhibit highly uncertain behavior, characterized by an increase in the entropy distribution of generated tokens, immediately following interactions with external tools. Motivated by this observation, ARPO incorporates an entropy-based adaptive rollout mechanism, dynamically balancing global trajectory sampling and step-level sampling, thereby promoting exploration at steps with high uncertainty after tool usage. By integrating an advantage attribution estimation, ARPO enables LLMs to internalize advantage differences in stepwise tool-use interactions. Our experiments across 13 challenging benchmarks in computational reasoning, knowledge reasoning, and deep search domains demonstrate ARPO's superiority over trajectory-level RL algorithms. Remarkably, ARPO achieves improved performance using only half of the tool-use budget required by existing methods, offering a scalable solution for aligning LLM-based agents with real-time dynamic environments.

> 💡 **问题动机**：
> - 核心矛盾：单轮推理 RL（如 GRPO 训练数学推理）已非常成功，但多轮 Agent 场景下 LLM 需要与工具环境进行**动态、多轮交互**，传统 trajectory-level RL 方法（如 GRPO、DAPO）直接套用会忽视 step-level 工具使用行为的学习。
> - 本文的核心观察：工具调用反馈后，LLM 生成的 token 熵**急剧上升**，说明工具交互引入了显著的不确定性，而这恰恰是有价值的探索信号。
> - ARPO 的两个核心组件：(1) 熵自适应 rollout：在高熵步骤动态分支采样；(2) Advantage Attribution Estimation：让模型感知共享 token 和分叉 token 的优势差异。
> - 结果亮点：13 个 benchmark 全面超越 trajectory-level RL，且仅用一半的工具调用预算。

---

Token Entropy Visualization of GAIA  
![Figure 1: Token Entropy Visualization of GAIA](../images/a4a8068d20994c695b5b5df603e35dba33534934ca83975a6162766db001f11b.jpg)

Performance on Deepsearch Benchmarks  
![Figure 1: Performance on Deepsearch Benchmarks](../images/7a87b183504b5712ffbbcc5c2efc185c44052502685e1a7c5d46724a1c2d5801.jpg)

Tool-Call Efficiency Analysis in RL Stage  
![Figure 1: Tool-Call Efficiency Analysis in RL Stage](../images/da112b43d6d80e174f72d692201820fd7dce139b2a5fd18e2e6d6ed4dbb75f39.jpg)  

*Figure 1: Overview of tool-use token entropy exploration and ARPO algorithm performance. Left: High entropy observed in the LLM following tool usage. Right: LLM performance comparison on deep search tasks using only 1k RL samples, along with a comparison of training tool-use budgets.*

> 💡 **Figure 1 批读**：
> - **左图**：GAIA benchmark 上的 token 熵可视化。注意每次 `<tool_call>` 标记后紧接着的 token 位置（红框标注）熵值明显升高，证明工具反馈引入了不确定性。
> - **中图**：Deep Search benchmark 上的性能对比。ARPO 在仅使用 1K RL 样本的情况下显著优于 GRPO，且在 Qwen3-8B 和 Qwen3-14B 上均有稳定提升。
> - **右图**：工具调用效率分析。ARPO 用 GRPO 一半的工具调用次数就达到了更高的准确率，x 轴是训练中 tool-call 次数，y 轴是整体准确率。
> - 这三张图分别对应本文的三个核心 claim：(1) 工具交互后有熵增；(2) ARPO 性能优于 trajectory-level RL；(3) ARPO 工具调用更高效。

---

> 💡 **Abstract 总结**：
> - **关键数字**：13 个 benchmark、50% 工具预算节省。
> - **核心洞察**：工具调用后的熵增是关键信号，trajectory-level RL 忽略了这个信号。
> - **可追问点**：ARPO 的熵阈值如何选择？分支采样的计算开销如何？理论保证是什么（答案见 Section 3.3 和 Appendix D.2）。
