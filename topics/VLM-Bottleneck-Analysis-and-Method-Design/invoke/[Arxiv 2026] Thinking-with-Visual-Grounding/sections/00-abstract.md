[← 返回 README](../README.md)

# Abstract & Paper Overview

## 一、论文信息速览

| 项目 | 内容 |
|------|------|
| **标题** | Thinking with Visual Grounding |
| **作者** | Junkai Zhang, Yihe Deng, Kai-Wei Chang, Wei Wang |
| **单位** | University of California, Los Angeles (UCLA) |
| **发表** | arXiv 2026 |

---

## 二、原始文本

Visual thinking should not only sound right; it should show its evidence. While recent vision language models (VLMs) can produce natural language reasoing traces, these traces often leave the supporting image regions implicit, making them hard to verify and difficult to supervise. We introduce visually grounded thinking, a reasoing process in which models interleave natural-language thoughts with explicit point or box groundings of the visual evidence used at each step. This lets the model express intermediate reasoing in language while grounding key objects in the image regions they refer to. To train this behavior, we construct a scalable synthesis pipeline that distills correct visual reasoing traces, extracts the visual objects required by the traces, grounds them with a SAM3-based agent, and derives aligned point and box supervision from the resulting masks. We further propose grounding-aware reinforcement learning, which combines answer correctness rewards with dense grounding rewards that score whether generated object references match the correct image evidence. Across two counting benchmarks and four spatial reasoing benchmarks, adding visually grounded thinking to Gemma3-4B-IT consistently improves performance over the original model and the non-grounded thinking baseline. On spatial reasoing, the visually grounded thinking 4B models match, and in some cases surpass, Gemma3-27B-IT from the same model family. Our analysis shows that point grounding is well suited to counting, while box grounding benefits most from explicit grounding rewards on spatial tasks. Overall, our results show that VLMs think better when their intermediate thoughts are tied to the image regions that make them true.

> 💡 **一句话概括**: 本文的核心命题是"视觉思考不应只是文本上听起来合理，还应能展示出它依赖的图像证据"。为此提出 visually grounded thinking——让 VLM 在推理过程中显式地用坐标（点或框）标注每一步思考所对应的图像区域。通过 SAM3-based 自动合成 pipeline 构建训练数据 + grounding-aware RL reward 联合优化答案正确性和 grounding 精度，使 4B 模型在计数和空间推理上达到甚至超越 27B 模型。

---

> 💡 **核心问题拆解**:
>
> - **现状痛点**: 当前 VLM 能产生自然语言推理链（如 R1-style thinking），但这些推理链的视觉证据是隐式的——模型说"有三个人撑着伞"，但没有指出图像中哪个区域支持这个判断。这使得：
>   1. 推理难以验证（可能答案正确但并非基于图像）
>   2. 推理难以监督（trace 看起来合理但实际缺少视觉 grounding）
>   3. 可能出现 "MIRAGE" 现象：答案对了但模型根本没看对地方
>
> - **解决方案**: 在自然语言 thinking 中交错插入 `<obj> name | [coordinates] </obj>` 的 grounding tag，用 box 或 point 坐标显式定位每一步推理依赖的视觉对象。
>
> - **关键挑战**: 如何大规模获取带有精确 grounding 标注的推理数据？答案：自动合成 pipeline = VLM 蒸馏推理链 + SAM3 agentic grounding + 对齐的 box/point 标注。

---

## 三、论文结构速览

```
Section 1: Introduction     → 问题动机 + visually grounded thinking 定义
Section 2: Related Work      → Visual CoT 到 Grounded Thinking 的演进
Section 3: Data Synthesis    → 6 阶段自动合成 pipeline（核心工程贡献）
Section 4: RL with Grounding → Box IoU / Point F1 reward 设计
Section 5: Experiments       → 6 个 benchmark 上的全面验证
Section 6: Conclusion        → 总结与展望
```

---

## 三、Summary

- **核心问题**: 视觉推理需要 explicit visual evidence，纯文本 thinking 隐去了证据来源
- **核心方案**: Visually grounded thinking = language thinking + coordinate grounding
- **工程支柱**: SAM3-based agentic grounding pipeline（无需人工标注）
- **训练方法**: SFT cold-start + GRPO with grounding-aware reward
- **关键结论**: 4B grounded 模型在空间推理上可比肩 27B；point 适合计数、box 受益于 grounding reward
