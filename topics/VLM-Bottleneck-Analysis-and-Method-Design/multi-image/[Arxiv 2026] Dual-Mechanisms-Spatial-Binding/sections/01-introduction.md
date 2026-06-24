[← 返回 README](../README.md)

# 1. Introduction

## 一、Preview

本文从空间变量绑定是 VLM 基本但困难的能力出发，指出 LM 领域已知变量绑定依赖排序表征，但 VLM 中这些表征的**起源**完全未知。作者提出核心问题：排序表征是由视觉编码器继承而来，还是 LM backbone 内部构建，还是两者的交互？回答这个问题不仅有助于理解 VLM 的内部机制，还能指导针对性的干预来改善推理性能。

---

## 二、原始文本

Spatial variable binding, the ability to associate objects with their properties and their relations to other elements, is a fundamental component of multimodal reasining [8, 6, 38, 5, 15, 37]. Tasks such as image captioning, visual question answering, and visual navigation require vision–language models (VLMs) to bind objects to spatial relations and reason about their relative arrangement within the scene. Although recent advances have made VLMs highly capable across many tasks, spatial variable binding remains a significant challenge [3, 7, 36, 23]. In this work, we investigate the internal mechanisms underlying spatial variable binding in VLMs and show how understanding these mechanisms enables targeted interventions to improve reasoning performance in complex scenes.

In language models (LMs), recent studies have shown that associating entities with their attributes (often referred to as variable binding) is done by forming symbolic representations for each entity in context. This representation encodes content-independent information about the order of entities in context, and allows the model to distinguish between entities and retrieve entity-specific information when needed [10, 30, 31]. Follow-up studies suggest that VLMs similarly rely on ordering information when reasoning about objects in images [1, 24]. However, while these findings establish the use of ordering-based representations during multimodal reasoning, they do not reveal their origin. In particular, it remains unclear whether such representations are constructed within the LM backbone, inherited from the vision encoder, or emerge from interactions between the two. Identifying the source of these representations is essential for understanding spatial variable binding failures in VLMs and for developing principled methods to diagnose and correct them.

> **机制拆解 — 变量绑定在 LMs 中的已知机制**:
>
> | 概念 | 含义 | LM 中的执行方式 |
> |------|------|----------------|
> | Variable Binding | 将属性与实体关联（如 "第二个盒子的水果是香蕉"） | 形成**内容无关的排序标识符**，用序号 (ordinal position) 而非内容来区分实体 |
> | Ordering Representation | 编码实体在上下文中出现顺序的表征 | 在中间层形成于实体相关 token 位置，随后传递到 final token 用于属性检索 |
> | Content-Independent | 排序表征不依赖于实体内容（颜色、形状等） | 确保即使实体内容被替换，排序信号仍然有效 |

> **核心 gap — 本文要回答的问题**:
>
> VLM 中的排序表征有三个可能的来源：
> 1. **Vision Encoder Origin**: 排序信息由视觉编码器直接编码在 visual embedding 中，LM backbone 只是消费它
> 2. **LM Backbone Origin**: 排序信息在 LM backbone 内部独立生成，与视觉编码器无关
> 3. **Interactive Origin**: 排序信息由视觉编码器和 LM backbone 的交互产生
>
> 本文通过因果干预实验回答：实际上是 **(1) + (2) 并存**，且 (1) 占主导，(2) 是辅助。

We show that the ordering-based representations underlying spatial variable binding in VLMs arise from two concurrent sources: The vision encoder represents the global layout of objects in the image, encoding ordering information that is directly projected into the embedding space of the LM backbone. Then, the LM backbone can further augment these representations by forming ordering information over object-associated visual tokens. While the vision encoder provides the primary source of ordering information, the LM-side mechanism plays a secondary role, augmenting spatial ordering when that information in the vision embeddings is degraded or completely removed. Notably, the ordering information provided by the vision encoder is distributed globally across visual tokens, extending beyond object regions into surrounding background areas. This is in contrast to the LM backbone, where information is formed locally over object-related tokens, similar to the process found in language processing [31, 1].

> **关键对比 — 视觉编码器 vs LM backbone 的排序表征差异**:
>
> | 维度 | 视觉编码器 | LM Backbone |
> |------|-----------|-------------|
> | 角色 | **主导** (primary) | **辅助** (secondary) |
> | 信息范围 | **全局分布式** (物体 + 背景 strip) | **局部** (仅在物体相关 token) |
> | 编码方式 | 视觉 embedding 直接编码空间布局 | 中间层在物体 token 上形成排序标识符 |
> | 与纯 LM 的类比 | 无（视觉特有） | 类似于 NLP 中的变量绑定机制 |
> | 消融后影响 | 准确率从 1.00 降至 0.43-0.64 | 可部分补偿（仍高于 chance） |

We establish these results through a series of controlled interchange intervention experiments [35, 27, 13] using carefully constructed counterfactual samples across three synthetic datasets and one controlled naturalistic dataset [23]. We validate our findings across three transformer-based VLMs, Qwen2-VL-7B-Instruct and Gemma-3-4b-it, showing that they rely on similar mechanisms to solve spatial variable binding. Leveraging this mechanistic understanding, we introduce a simple global intervention on vision embeddings that amplifies ordering information across all image tokens. On complex scenes from the COCO dataset [25, 23], this intervention corrects up to 55% of previously incorrect predictions, yielding improvements of up to 22 percentage points in overall accuracy across tested models.

> **实验路线图的一页总结**:
>
> ```
> 因果验证 (受控)→ 来源定位 (视觉 vs LM)→ 机制理解 → 干预纠正 (自然)
>      │                    │                    │              │
>  Sec 5.1            Sec 5.2-5.3           Sec 5.4        Sec 5.4
>  interchange        probe +              ablation       probe
>  intervention       patching             + patching     amplification
>  on final token     on vision tokens     on LM layers   on COCO
> ```

Overall, our results elucidate the mechanisms underlying spatial association in VLMs, connect binding mechanisms identified in LMs to multimodal reasoning, and demonstrate the central role of vision encoders in supporting spatial variable binding in VLMs. All code and data supporting this study are available at https://spatial.baulab.info

---

## 三、Summary

- **问题定义**: VLM 中空间变量绑定的排序表征从何而来？是继承自视觉编码器还是 LM backbone 生成的？
- **核心洞察**: 双重来源 —— 视觉编码器（主导，全局分布式）+ LM backbone（辅助，局部）
- **方法**: 受控 interchange intervention + linear probe + 消融实验 + probe 放大
- **贡献**: 回答了排序表征的起源问题 + 建立了从机制理解到性能改善的完整链路
