[← 返回 README](../README.md)

# 1. Introduction

## 一、Preview

Introduction 沿着"纯文本推理的局限 → 视觉推理需要 grounding 的原因 → visually grounded thinking 的提出 → 数据与训练挑战 → 贡献总结"的递进逻辑展开。核心论点是：视觉推理的 evidence 位于图像中，无法完全用文字表达，因此需要在 thinking 中显式地链接到图像区域。

---

## 二、原始文本

Language models have made strong progress on complex problem solving by producing explicit natural-language reasoing traces. In particular, R1-style reinforcement learning has shown that models can improve their ability to solve math, coding, and general reasoing problems through long textual thinking (Guo et al., 2025). This success has motivated analogous reasoing methods for vision-language models (VLMs): given an image and a question, the model can think in text before giving the final answer. Such a strategy has been shown to be effective for visual question answering (Deng et al., 2025; Hu et al., 2026).

However, visual thinking differs from purely textual thinking because the evidence needed to solve a visual question is located in the image (Zhu et al., 2016) and cannot be fully expressed in words. When humans answer visual questions, we often link our thoughts to concrete image regions, such as the person on the left, the cup near the table edge, or the object being counted (Das et al., 2016). These visual references guide where attention should be directed and what task-specific information should be extracted (Hayhoe and Ballard, 2005). In contrast, a pure natural-language reasoing trace may state that "the red car is near the entrance" or that "there are three people holding umbrellas," but it does not identify which image regions support these claims. This makes the thinking hard to verify and supervise: a final answer may be correct even without the image, while the reasoing trace can still appear coherent and image-based (Asadi et al., 2026). Thus, visual thinking requires not only step-by-step reasoing, but also explicit links between important reasoing steps and the correct visual evidence.

> 💡 **机制拆解 — 纯文本 thinking 的三个根本缺陷**:
>
> | 缺陷 | 说明 | 对应现象 |
> |------|------|---------|
> | **不可验证 (unverifiable)** | 模型说"有三个人撑着伞"，但无法检查它是真的从图中数出来的，还是语言先验猜出来的 | MIRAGE -- 答案正确但视觉理解不存在 |
> | **难以监督 (hard to supervise)** | 推理链文本看起来很合理，但可能根本没有用到正确的图像区域 | 无法区分"真 reasoing"和"空洞 rhetoric" |
> | **证据缺失 (missing evidence)** | 图像中的空间关系 (left/right, near/far)、遮挡、重叠等信息难以用纯文字完整表达 | 文字是离散符号，图像是连续信号 |

We propose visually grounded thinking to address this issue, as shown in Figure 1. In this format, the model interleaves natural-language reasoing with explicit visual grounding. Whenever a reasoing step refers to an important visual object, the model outputs a coordinate-based grounding tag, using either a bounding box or a point, to identify the referenced object in the image. Natural language and spatial coordinates are combined in the thinking process: language describes the thoughts, while the coordinates specify the visual evidence that supports each step.

> 💡 **Visually Grounded Thinking 的形式定义**:
>
> ```
> <think>
>   I can see <obj> the red car near the entrance | [120,300,450,620] </obj>
>   and <obj> a blue truck on the right | [500,280,850,640] </obj>.
>   There are <obj> three pedestrians | [100,680,160,780]; [170,670,230,770]; [240,685,300,785] </obj>.
>   Counting the vehicles on the left side, there are two.
> </think>
> \boxed{2}
> ```
>
> 关键设计:
> - **自然语言** 描述思考过程（如 "the red car near the entrance"）
> - **坐标 grounding** 指出证据位置（box 或 point）
> - **多实例支持**：同一 `<obj>` tag 内用 `;` 分隔多个实例
> - **两种模式**：box mode (`[x1,y1,x2,y2]`) vs point mode (`[x,y]`)

Visual thinking should not only sound plausible in natural language; it should point to the evidence it uses. Our work turns that idea into a training recipe for visually grounded thinking, where models interleave natural-language thinking with point or box groundings of the image regions that support each step. By combining a scalable SAM3-based synthesis pipeline with an RL grounding reward, we train VLMs to optimize both answer correctness and the accurate grounding of visual objects referenced during thinking. The results show that visually grounded thinking substantially improves counting and spatial reasoing, with 4B grounded models matching, and sometimes exceeding, much larger 27B models on spatial benchmarks. Overall, our work suggests that the next step for visual thinking is not simply longer thinking, but thinking that is tied to the image in a form that can be checked, supervised, and improved.

> 💡 **关键类比**: 本文从人类认知角度切入：
> - 人类在回答视觉问题时，注视点会动态地在关键物体之间切换
> - 人类的"内部思维"会将对物体的描述与物体的空间位置绑定
> - Visually grounded thinking 就是让 VLM 模拟这个过程：说一个 object 的时候，同时指出它在哪里

Visually grounded thinking requires training data that supports both SFT and RL. Coordinate-annotated reasoing traces can teach the model to interleave language and grounding during SFT, but RL needs supervision at the level of each visual reference, since a rollout may rename objects, reorder reasoing steps, skip supervised entities, or ground additional useful evidence. We therefore build an automatic data synthesis pipeline around a SAM3-based grounding agent (Carion et al., 2026). Starting from visual questions, the pipeline obtains correct reasoing traces, extracts the visual objects used in the reasoing, represents each object with a name and a disambiguating scene context, and grounds it with a run-length encoding (RLE) mask. These masks are used to construct both point and box-mode grounded reasoing traces, while the corresponding grounded objects are kept as structured supervision for grounding-aware RL.

> 💡 **数据合成的核心难题 — 为什么需要 RL-level 的结构化监督？**:
>
> SFT 只需要带标注的完整推理链（模型模仿即可），但 RL 需要评估推理链中每个 grounding object 的质量。问题在于：
> - 模型 rollout 中可能重新命名 object（"red car" → "the automobile"）
> - 重新排序推理步骤
> - 跳过 supervised entity（不需要那个 object 也能推理）
> - 增加额外的合理 grounding（发现了 pipeline 没提取的视觉线索）
>
> 因此不能简单用文本匹配来评估 grounding，而是需要：
> 1. 保留每个 ground-truth object 的 (name, context, mask) 作为结构化监督
> 2. RL 时用 object router 将 rollout 中的 grounding objects 匹配到 ground-truth objects
> 3. 在 matched pairs 上计算几何质量（box IoU 或 point F1）

---

**Figure 1** (images/a0695a50f352ffff2ef1d669ac7595a7d7924a7b7c014fe97a45cb3dd54ab191.jpg): 对比三种模式：
- Pure natural language thinking：只有文本描述，没有空间指引
- Box-mode grounded thinking：文本 + bounding box 坐标
- Point-mode grounded thinking：文本 + point 坐标

**Figure 2** (images/084645ca2187996989903a8e2931993d865da5878683ce63d5839df94918e22c.jpg): 真实评测中 visually grounded thinking 模型的输出示例。

---

## 三、Summary

- **问题定义**: 纯文本视觉推理缺乏 explicit visual evidence，推理链不可验证、难以监督
- **核心洞察**: 视觉推理的 evidence 在图像中，无法完全用文字表达；需要像人类一样将"想什么"和"看哪里"绑定
- **方案定义**: Visually grounded thinking = interleaved language thinking + coordinate grounding tags
- **三大贡献**: (1) SAM3-based 自动数据合成 pipeline; (2) Grounding-aware RL reward; (3) 6 个 benchmark 上验证 grounding 的有效性及 box/point 的差异化优势
