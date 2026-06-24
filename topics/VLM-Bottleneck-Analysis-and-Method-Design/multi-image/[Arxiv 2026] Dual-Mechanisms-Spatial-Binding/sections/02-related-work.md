[← 返回 README](../README.md)

# 2. Related Works

## 一、Preview

本文的相关工作围绕三条主线展开：(1) 空间推理基准 —— VLMs 在空间推理上仍面临困难；(2) VLM 内部机制研究 —— 跨模态注意力、表征几何等提供有价值的工具但未聚焦空间绑定；(3) LMs/VLMs 中的符号表征 —— LM 中已知变量绑定依赖排序标识符，VLM 中该机制的起源是开放问题。

---

## 二、原始文本

Benchmarking Spatial Reasoning Many benchmarks have been proposed to evaluate spatial reasoning in VLMs, including synthetic datasets designed to test compositional relations and naturalistic benchmarks derived from real images and human annotations [22, 4, 14, 26]. These benchmarks have revealed that spatial reasoning remains challenging for current VLMs, particularly as relational complexity increases or when reasoning must generalize beyond object-centric cues.

Understanding Inner Working of VLMs A growing body of work has investigated the internal workings of VLMs, examining cross-modal attention patterns, token alignment, representational geometry, and information flow between vision encoders and LM backbones [29, 32, 1, 20, 28, 19, 21, 33, 9]. These studies have provided valuable insights into how visual and textual information is integrated and how multimodal representations emerge. Our work builds on this line of research by focusing specifically on the mechanisms underlying spatial association and reasoning.

Symbolic Representation in LMs & VLMs Several studies have shown that LMs implement variable binding by forming content-independent ordering identifiers over important tokens [16, 31, 10, 30, 12, 11]. These mechanisms underlying variable binding are fundamental to many in-context reasoning tasks and have been presented as evidence of symbol-like processing in neural networks. Recent work has extended this line of inquiry to VLMs, investigating similar symbolic representations [24, 1, 34, 17]. Our findings relate to this body of work while revealing an important distinction in the multimodal setting. We show that although VLMs form symbolic representations that encode ordering information within the LM backbone, they only act as a supporting mechanism. Instead, symbolic representations originating from the vision encoder play a dominant role.

> **本文在三类相关工作中的差异化定位**:
>
> | 相关工作线 | 已有认知 | 本文的新发现 |
> |-----------|---------|-------------|
> | 空间推理基准 | VLMs 在空间推理上表现不佳 | 不仅评估"好不好"，还揭示了"为什么不好"的机制原因 |
> | VLM 内部机制 | 跨模态集成如何发生 | 具体揭示了空间排序信息的双重来源和 flow |
> | LM/VLM 符号表征 | LM 中用排序标识符做变量绑定；VLM 中也存在类似机制 [1, 24] | VLM 中排序表征的**起源是双重的**：视觉编码器提供主导信号（全局分布），LM backbone 只是辅助（局部） |
>
> **与 Assouel et al. (2025) [1] 的关键区别**: [1] 证明了 VLM 在 visual variable binding 中使用符号表征（存在性），但没有回答表征的**来源**。本文填补了这一空白：揭示了符号表征既来自视觉编码器又来自 LM backbone，且视觉编码器是主导来源。

> **方法论文献的关键工具**:
>
> | 方法 | 来源 | 本文中的用途 |
> |------|------|-------------|
> | Interchange Intervention (Activation Patching) | Vig et al. 2020 [35], Meng et al. 2022 [27] | 通过替换 counterfactual 样本的中间表征来测试因果性 |
> | Interchange Intervention Accuracy (IIA) | Geiger et al. 2022 [13] | 量化干预效果（用目标 token 的平均概率替代原定义的二元一致） |
> | Linear Probing | Belinkov 2022 [2] | 验证排序信息是否在 visual embedding 中线性可解码 |

---

## 三、Summary

- **关键文献链**: LM 变量绑定机制 (Prakash/Dai/Feng) → VLM 符号表征存在性 (Assouel/Kang) → **本文**: VLM 符号表征的双重起源
- **本文的核心区分**: 视觉编码器产生的符号表征是全局分布式的，LM backbone 产生的符号表征是局部化的，这与纯 LM 场景既相似又不同
- **方法基础**: interchange intervention + probing 的组合是因果分析的标准范式
