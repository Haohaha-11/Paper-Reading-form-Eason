[← 返回 README](../README.md)

# 5. Conclusion

## 一、Preview

结论部分简洁但有力：SIEVE 证明了"有效的视觉回访不需要外部工具编排"，通过谨慎利用内部多模态表示可以更高效、更可扩展地改善 image-grounded reasoing。同时隐含了一个更大的主张：VLM 的内部信号可能比我们想象的更丰富，值得被更系统地挖掘。

---

## 二、原始文本

In this work, we propose a training framework, SIEVE, that enables vision–language models (VLMs) to revisit and leverage image information during inference without relying on external tools. Unlike existing approaches that depend on tool invocation, retrieval systems, or additional visual processing modules, SIEVE extracts and utilizes the intrinsic signals already present within the VLM's hidden states, using them as structured hints to guide image-grounded reasoing. By exploiting these internal representations, SIEVE encourages the model to dynamically refine its understanding of visual content in a lightweight and self-contained manner. Despite its simplicity, SIEVE demonstrates substantial performance improvements over both the vanilla baseline model and training-free toolbased methods. These results suggest that effective visual revisiting does not necessarily require external tool orchestration; rather, carefully harnessing internal multimodal representations can provide a more efficient and scalable solution for improving image-grounded reasoing.

> 💡 **结论的三个层次解读**:
>
> **层次 1 — 方法贡献**: SIEVE 是一个免外部工具的视觉回访框架，用 ~1.5k 样本训练即可让 VLM 学会 "何时回头看"和"看哪里"，在多个 benchmark 上取得一致提升。
>
> **层次 2 — 范式启示**: "有效的视觉回访不需要外部工具编排"——这是一个对 "Thinking with Images" 潮流的根本性质疑。SIEVE 的实验结果表明，内部信号挖掘的增益可以匹敌甚至超越外部工具调用，且成本更低、更稳定。这暗示着研究重点应该从 "设计更精巧的外部工具" 转向 "更深入地理解 VLM 内部表示"。
>
> **层次 3 — 未来方向**: SIEVE 的成功暗示 VLM 的 hidden states 可能包含比我们意识到的更丰富的信息——不仅包括视觉证据，还可能包括空间推理模式、对象关系、甚至多步推理的隐式 memory。这开启了一个新的研究方向：**"内部信号挖掘"作为 VLM 能力增强的通用范式**。

> 💡 **批判性总结 — SIEVE 的定位与未解决的问题**:
>
> **SIEVE 解决了什么**:
> - 证明了 "不需要外部工具就能做视觉回访"，且性能有竞争力
> - 提出了一套完整的 "自引导 evidence discovery + RL training" 管线
> - 揭示了内部 embedding 复用比生成新 view 更高效、更稳定
>
> **SIEVE 没解决什么**:
> - 证据发现的质量仍依赖基础模型的校准度——如果模型完全不理解问题，锚点本身就是噪声
> - K=1 的单区域假设在需要多对象联合推理时可能不够
> - 仅在 Qwen3 系列上验证，跨架构泛化性待证
> - Evidence cache 的周期性刷新增加了训练的工程复杂度
> - 没有与更强 RL-based 工具增强方法（如 DeepEyes, OpenThinkImg）直接对比
>
> **SIEVE 打开的方向**:
> - 从 "生成新信息" 到 "复用已有信息" 的范式转换
> - 内部信号挖掘作为 VLM 能力增强的第三条路径（不同于 model scaling 和 data scaling）
> - 轻量级 evidence 机制的消融研究可能启发更紧凑的 VLM 推理架构设计

---

## 三、Summary

- **核心结论**: 有效视觉回访可以完全通过内部多模态表示的复用实现，无需外部工具
- **核心证据**: SIEVE 在 2 个模型规模 × 8 个 benchmark 上的一致提升
- **范式意义**: "从重新看到重新用"——VLM 内部信号的挖掘是一个被严重低估的研究方向
