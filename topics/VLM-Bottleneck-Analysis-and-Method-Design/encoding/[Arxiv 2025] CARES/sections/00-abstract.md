[← 返回 README](../README.md)

# Abstract

## 📌 预览

摘要是全篇压缩版：VLM 普遍用高分辨率处理所有图像，导致 visual tokens 膨胀到占 prefill 阶段 99% 的 token——但大多数 query 根本不需要这么多细节。CARES 是一个轻量预处理模块，用一个小 VLM 看一眼低分辨率图像就能预测当前 query 需要多高分辨率，从而在 tokenization 之前削减计算。在 9 个 benchmark 上平均降低 78% 计算量，精度几乎不变。

---

# CARES: Context-Aware Resolution Selector for VLMs

Moshe Kimhi<sup>1,2</sup>\* Nimrod Shabtay<sup>2,3</sup>\* Raja Giryes<sup>3</sup> Chaim Baskin<sup>4†</sup> Eli Schwartz<sup>2†</sup>

<sup>1</sup>Technion <sup>2</sup>IBM Research

<sup>3</sup>Tel-Aviv University <sup>4</sup>Ben-Gurion University

Project Page: https://mkimhi.github.io/CARES/

## Abstract

Large vision–language models (VLMs) commonly process images at native or high resolution to remain effective across tasks. This inflates visual tokens up to to 99% of total tokens of the prefill stage, resulting in high compute and latency, even when lowresolution images would suffice. We introduce CARES—a Context-Aware Resolution Selector, a lightweight preprocessing module that, given an image–query pair, predicts the minimal sufficient input resolution. CARES uses a compact VLM (350M) to extract features and predict when a target pretrained VLM's response converges to its peak ability to answer correctly. Though trained as a discrete classifier over a set of optional resolutions, CARES interpolates continuous resolutions at inference for fine-grained control. Across nine multimodal benchmarks spanning documents and natural images, as well as diverse target VLMs, CARES preserves task performance while reducing compute by up to 78% on average across 9 benchmarks.

> 💡 **摘要批读**: 摘要中有四个关键 claim。第一，"visual tokens 可以占到 99%"——这个数字来自高分辨率下的 token 统计（Appx A.1），需要核实它的计算假设（100 文本 token + AnyRes/Qwen2.5-VL tokenization）。第二，"预训练 VLM 的 response 收敛到最佳能力"——这里 CARES 用 ANLS >= tau 且高分辨率无显著提升来定义"收敛"，tau=0.85 的选择会影响标注质量。第三，"离散训练 + 连续推理"——这是 CARES 的一个设计精巧处：训练用 K-way 分类避免连续标注的成本，推理用 softmax 期望插值实现细粒度控制。第四，"up to 78%"——需要看这 78% 是否在某些 benchmark 上特别高（如 MMMU 85%），另一些 benchmark 特别低（如 MathVista 22%），以及差异的原因。

> 💡 **批注**: 关键词 "minimal sufficient input resolution" 是本文的核心定义。注意它用的是"充分"（sufficient）而不是"最优"（optimal），因为多分辨率 rollout 只能提供离散的充分性检查，不一定能给出真正连续意义上的最优值。后面的 continuous inference 正是为了弥补这个 gap。

> 💡 **批注**: CARES 的 "lightweight preprocessing module" 定位值得注意——它不是 VLM 的一部分，而是 VLM 前面的一个独立组件。这意味着它可以和任何 VLM 一起使用（包括 API 模型如 GPT-4o），且和所有 post-tokenization 的效率优化方法正交互补。

---

## 🔖 Section 总结

### 核心洞察
1. CARES 的核心对象是"最小充分输入分辨率"，不是"最优分辨率"。充分性由 ANLS 阈值和收敛条件共同定义。
2. "离散训练 + 连续推理"的设计解决了标注成本（只需标注 K 个离散分辨率）和部署精度（需要连续控制）之间的矛盾。
3. 摘要的 +78% compute reduction 是全局 claim，后面要看这 78% 在不同 benchmark 和不同 target VLM 上的分布是否均衡。
