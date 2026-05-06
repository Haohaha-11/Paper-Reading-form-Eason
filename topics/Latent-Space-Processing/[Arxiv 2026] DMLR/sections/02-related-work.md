[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
Related Work 很短，重点把 DMLR 放在 explicit reasoning 和 latent reasoning 的交叉处：它既关心视觉 grounding，也关心 latent-space 内部推理。

---

# 2. Related Work

Explicit Reasoning. Many prior works have explored visual reasoning. Early approaches mainly relied on semantic CoT, where the model performs all inference in the text space after a one-time visual encoding [5, 4, 20, 21]. However, this separation between perception and reasoning often leads to misalignment and hallucination [6, 22, 23, 24, 22, 25]. To address these limitations, recent studies adopt a Thinking-with-Images paradigm, where the model can draw auxiliary elements [13, 26, 27], zoom or crop regions [11, 8, 28, 29], or generate intermediate visual cues [30, 10, 9], enabling it to reason directly over visual structures.

> 💡 **显式推理脉络**: 作者把 explicit reasoning 的缺陷定义成“先看一次图，再在 text space 里想”。这解释了为什么长 CoT 反而可能 hallucinate：后续 token 主要沿语言先验滚动，视觉证据没有被持续校正。Thinking-with-Images 修正这个问题，但代价是工具/图像操作的稳定性和 latency。

Latent Reasoning. Recently, an increasing number of studies have begun to shift reasoning from the explicit token space to the model’s latent representation space. Some methods introduce dedicated training frameworks that optimize latent representations to support more effective internal reasoning [31, 14, 32, 33, 34, 35], while others propose training-free approaches that manipulate latent activations during inference to refine the reasoning process [15, 36, 37, 38, 39]. In addition, several recent works explore injecting visual information into the latent space [16, 17, 40, 41, 18], enabling models to iteratively operate over both latent semantic features and latent visual cues, thereby supporting a more flexible form of interleaved multimodal reasoning.

> 💡 **latent reasoning 定位**: DMLR 属于 training-free / test-time latent manipulation，更接近 LatentSeek、test-time policy optimization 一类，而不是 CoCoNut 那种训练阶段塑形。它和视觉 latent 注入工作的区别在于注入不是固定规则，而是按 reward 更新 best visual patches。

---

## 🔖 Section 总结

### 关键数字速查
| 类别 | 代表问题 |
|------|----------|
| Explicit text CoT | 语言偏置、视觉 grounding 弱 |
| Thinking-with-Images | 工具调用不稳定、开销高 |
| Latent reasoning | 常需训练或固定触发位置 |
| DMLR 位置 | training-free latent optimization + dynamic visual injection |

### 核心洞察
1. 相关工作段落虽短，但已经给出 DMLR 的差异化：不训练、动态、视觉和语义 latent 交错。
2. 对 Latent-Space-Processing 课题，DMLR 是从纯语言 latent reasoning 走向多模态 latent interleaving 的一篇。
