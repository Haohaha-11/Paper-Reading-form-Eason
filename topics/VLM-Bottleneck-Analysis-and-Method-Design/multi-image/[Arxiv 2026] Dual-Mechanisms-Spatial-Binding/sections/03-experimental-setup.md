[← 返回 README](../README.md)

# 3. Experimental Setup

## 一、Preview

本节定义了实验的三要素：数据（从合成到自然的四层递进）、模型（两个 transformer-based VLMs）、方法（probe + interchange intervention）。受控设置支持严格因果分析，自然场景验证生态效度。特别重要的是三类合成设置中"只改变视觉内容而非空间结构"的设计，使得可以精确构建用于因果干预的 counterfactual 配对。

---

## 二、原始文本

### 3.1 Data

We study spatial variable binding across increasing visual complexity, from synthetic and controlled settings to complex natural images (Fig. 1). The controlled settings are necessary to enable causal analysis, which requires constructing original-counterfactual input pairs that differ in exactly one factor. The naturalistic settings then provide evidence that the findings apply to real-world conditions.

Synthetic Settings We consider three synthetic settings: Squares, Shapes, and Objects. In all settings, three equal-sized items are placed at fixed, equal distances and arranged either horizontally or vertically. The settings differ only in visual content: Squares uses colored squares, Shapes uses colored geometric shapes, and Objects uses real-world object images. All images are generated programmatically from predefined collections of colors, shapes, and objects (see full list at App. A), enabling precise control over attributes and spatial configuration. For each image, we query the VLM to predict an attribute (color, shape, or object name) of a target entity (square, shape, or object) based on its relative position to a reference entity. For our intervention experiments (see Sec. 5), we sample 50 pairs of clean-counterfactual images from each setting. We validate generalization of results to synthetic data with real background in App. C.8.

> **数据集设计的关键约束**:
>
> | 约束 | 原因 |
> |------|------|
> | 三个物体等大小、等距排列 | 确保空间关系是唯一变化因素，不受大小/距离等混淆变量影响 |
> | 仅水平或垂直排列 | 简化空间关系为 left/right 或 above/below 二元选择 |
> | Clean-counterfactual 仅在一个因素上不同 | interchange intervention 的因果推断前提：如果替换一个中间表征改变了输出，说明该表征对输出有因果作用 |
> | 合成图片 + 真实背景对照 | App. C.8 验证排序信息分布不受背景类型影响 |

Naturalistic Setting For mechanistic analysis, we use the control subset of What'sUp [23] which consists of images containing pairs of objects: a central reference object (e.g., a chair) and a secondary object positioned adjacent to it on either side. To adapt this dataset to our intervention experiments (Sec. 5), which require images containing three objects, we construct composite scenes by merging pairs of What'sUp images into a single image with a shared reference object and two surrounding objects, resulting in 1074 images. This construction supports only horizontal spatial relations. In Sec. 5.4, we evaluate generalization to natural images from COCO-spatial [23, 25], curated by filtering images from the COCO validation set that exhibit spatial relations, resulting in 2687 images with complex scenes containing multiple objects, varied spatial arrangements, and realistic backgrounds.

> **数据集递进逻辑**:
>
> ```
> Squares → Shapes → Objects → What'sUp → COCO-spatial
>   └─完全受控─┘      └─受控自然─┘   └──完全自然──┘
>   因果分析           因果分析          生态效度验证
> ```

### 3.2 Models

The mechanistic study is performed with three transformer-based VLMs: Qwen2-VL-7B-Instruct and Gemma-3-4b-it. These models differ in scale and training data, allowing us to assess whether the identified mechanisms generalize across diverse VLM designs. The behavioral performance of the models on the spatial variable binding tasks is summarized in Table 1. For causal experiments, we restrict analysis to examples where the model answers correctly, ensuring the identified mechanisms reflect genuine behavior rather than noise. Nevertheless, we show generalization of the findings to failure cases in Sec. 5.4.

Table 1: Models' accuracy on studied tasks.

| | Squares | Shapes | Objects | What'sUp |
|---|---------|--------|---------|----------|
| Qwen2-VL-7B-Instruct | 1.00 | 1.00 | 1.00 | 0.73 |
| Gemma-3-4b-it | 0.98 | 0.99 | 0.99 | 0.62 |

> **模型选择逻辑**:
> - Qwen2-VL-7B: 7B 参数，在合成任务上完美解决，适合研究"正确时的机制"
> - Gemma-3-4B: 4B 参数，在 What'sUp 上准确率较低（0.62），展现不同能力水平
> - 两个模型架构和训练数据不同，验证机制的跨模型通用性
> - 因果分析只在正确样本上进行（排除噪声），但 Sec 5.4 在失败样本上验证

### 3.3 Methods

Representation Probing Probing methods are widely used to analyze what information is encoded in the internal representations of neural networks [2]. In this framework, a probe is typically a lightweight classifier trained to predict a target property y from an internal representation h ∈ R^d extracted from the model. In its simplest form, a linear probe takes the form

ŷ = Wh + b

where W and b are learned parameters. Successful probing that generalizes to test cases indicates that the target information is linearly decodable from h, suggesting that it is present in the model's representation. However, probe performance alone does not establish that the probed representation plays a causal role in the model's behavior [18]. As a result, probing is best interpreted as a diagnostic tool for localizing candidate representations, and is often combined with causal interventions to distinguish correlational encoding from functional relevance.

> **Probing 的局限性与本文的应对**:
>
> | 问题 | 应对 |
> |------|------|
> | Probe 准确率高 ≠ 有因果作用 | 本文将 probe 作为定位工具，然后用 interchange intervention 验证因果性 |
> | Probe 可能学到 shortcut | 在训练集外的背景 token 上测试 probe 的泛化，发现 strip 模式，说明 probe 的泛化模式是有意义的 |

Causal Mediation Analysis Interchange intervention is a technique for testing causal relationships between a model's internal representations and its behavior [35, 27]. Let f(·) denote the model, and let $h_{ℓ}$(x) denote the internal representation at component or layer ℓ produced during a forward pass on input x. Given an original input x and a corresponding counterfactual input x', an interchange intervention replaces $h_{ℓ}$(x) with h̄_ℓ(x') while keeping the remainder of the computation unchanged. This yields an intervened output

ŷ_int = f(x; $h_{ℓ}$ ← $h_{ℓ}$(x'))

a procedure commonly referred to as activation patching. If the intervened output ŷ_int matches the target outcome associated with the counterfactual input, this provides evidence that the intervened representation h plays a causal role in the model's computation. We quantify this effect using interchange intervention accuracy (IIA) [13]. In our setting, rather than measuring binary agreement with the intervention target as in the original definition of IIA, we measure the average probability assigned to the target intervention outcome token.

> **本文 IIA 变体的含义**:
>
> 原始 IIA: 输出是否精确等于 counterfactual 的 target token（二元）
> 本文 IIA: 输出 token 的平均概率（连续值），更细粒度地测量干预效果的强度
>
> 例如，在 Fig 2 的实验中：
> - 如果模型输出 "Blue" 的概率从 0.05 变为 0.85 → 排序信息被成功转移
> - 如果模型输出 "Black" 的概率从 0.05 变为 0.85 → 颜色属性被直接转移

---

## 三、Summary

- **数据递进**: 合成（3 类）→ 受控自然（What'sUp）→ 完全自然（COCO-spatial），兼顾因果推断和生态效度
- **模型**: Qwen2-VL-7B 和 Gemma-3-4B，验证机制的跨模型泛化
- **方法**: Linear probe（定位候选表征）+ Interchange intervention（验证因果性），probe 不建立因果但引导干预目标
