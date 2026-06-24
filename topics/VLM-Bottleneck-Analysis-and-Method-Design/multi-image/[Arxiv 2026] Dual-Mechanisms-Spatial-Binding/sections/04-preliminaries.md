[← 返回 README](../README.md)

# 4. Preliminaries

## 一、Preview

本节建立概念基础：首先回顾 LM 中符号表征（排序信息）的已知机制，然后过渡到 VLM 中的已知发现与关键开放问题。核心论点是：虽然已有工作证明 VLM 使用排序表征做空间变量绑定，但这些表征的**来源**（视觉编码器 vs LM backbone）完全未知。

---

## 二、原始文本

### 4.1 Ordering Information in LMs

Variable binding is the process of associating attributes with their respective entities. For instance, given a red, green, and blue square, the process of assigning the color feature to each square involves variable binding. Owing to its fundamental role in reasoning, recent works have investigated the underlying mechanisms that enable it in LMs [31, 10]. A common finding among these works is the reliance of LMs on content-independent ordering representations to bind entities to their attributes. These representations encode the order of entities independently of their content and align this ordering with a corresponding order over attributes in the context, enabling the model to retrieve entity-specific information when queried. For instance, given a prompt such as "Box A contains an apple, box B contains a banana, and box C contains cherries. What is the color of the fruit in box B?", the model uses ordering information corresponding to the second entity (Box B) to retrieve the attributes of the corresponding second fruit (banana). Prior analyses suggest that such ordering information is formed in intermediate layers on top of entity-related token positions and is subsequently transferred to the final token position when answering a query, where it is used to retrieve the relevant property (e.g., the color "yellow" of the banana). In this work, we find that VLMs similarly form ordering information over visual tokens corresponding to objects in an image and use this information to reason about the spatial locations of objects and their attributes. However, as we show in the following sections, this LM-style ordering mechanism plays only a secondary role in multimodal spatial variable binding.

> **LM 中变量绑定的执行流程（两阶段）**:
>
> ```
> 阶段 1: 排序编码 → 阶段 2: 属性检索
> ────────────────────────────────────────
> "Box A: apple,   Box B: banana,   Box C: cherries"
>        │               │                │
>    (第1个实体)    (第2个实体)     (第3个实体)
>        │               │                │
>    [中间层形成排序标识符: 1st, 2nd, 3rd]
>        │               │                │
>        └───────────────┼────────────────┘
>                        ↓
>        Query: "...fruit in box B?"
>                        ↓
>        [Final token 接收排序标识符 "2nd"]
>                        ↓
>        [检索第2个属性: banana → "yellow"]
>        Output: "yellow"
> ```
>
> **核心概念**: "内容无关" (content-independent) 意味着排序标识符不编码 "apple/banana/cherries" 的内容，只编码 "第1/第2/第3" 的顺序。这样即使换一组水果，排序机制仍然工作。

### 4.2 Ordering Information in VLMs

In VLMs, spatial variable binding is a fundamental capability central to many spatial reasoning tasks. Recently, [1] demonstrates that when VLMs are queried about objects within images, they retrieve symbolic representations that encode their ordering information. While this work establishes the existence of such ordering representations for visual variable binding tasks, it does not characterize the source of it. Specifically, the work mainly shows that for variable binding, the model uses this symbolic representation at the last token position, and that it comes from visual tokens; however it remains unclear where and how VLMs create ordering information. In particular, it remains unclear whether ordering representations are constructed within the LM backbone, inherited directly from the vision encoder, or emerge through interactions between the two components. In this work, we show that the source of this is two-fold: the vision encoder encodes positional information to capture spatial layout, while the LM backbone further enhances this information. Understanding the origin of these representations enables a simple and principled intervention to improve spatial reasoning.

> **本文在已知基础上的增量**:
>
> | 已知（Assouel et al. 2025 [1]） | 未知 → 本文回答 |
> |--------------------------------|----------------|
> | VLMs 在 final token 位置使用排序表征解决 visual variable binding | 排序表征的**来源**是什么？ |
> | 排序表征来自 visual tokens | 是在视觉编码器中编码的，还是 LM backbone 从 visual tokens 中提取的？ |
> | 排序表征是符号化的（内容无关） | 这种符号化是在哪个阶段完成的？ |
>
> **本文的回答画布**:
> ```
> Image → [Vision Encoder] → visual embeddings (含全局排序信息)
>                                   ↓
>                              [LM Backbone]
>                          ├─ 消费 + 增强视觉排序 (主导路径)
>                          └─ 在物体 token 上独立形成排序 (辅助路径)
>                                   ↓
>                          Final token: 排序 → 属性检索 → Output
> ```

---

## 三、Summary

- **LM 已知**: 变量绑定通过中间层形成内容无关排序标识符 + final token 检索属性，分两阶段执行
- **VLM 已知**: [1] 证明 VLM 在 visual variable binding 中也使用排序表征，且来自 visual tokens
- **核心 gap**: 排序表征的起源不明 —— 是视觉编码器直接编码，还是 LM backbone 构建，还是交互产生？
- **本文答案**: 双重来源，视觉主导 (全局分布式) + LM 辅助 (局部)
