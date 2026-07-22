[← 返回 README](../README.md)

## 📌 预览
TTT、长上下文架构和记忆增强相关工作。

---

# 5 Related Work

Test-Time Training (TTT). Test-Time Training (TTT) is a paradigm that enables a model to adapt dynamically in response to continuous streams of data at inference by updating a small subset of its parameters, known as fast weights [2]. Initially demonstrating success in computer vision [47, 53], TTT has since been extended to numerous other modalities—including language [48], video [15], and audio [18]—underscoring its broad applicability. Research in this area has largely focused on two avenues for improving TTT’s effectiveness: the design of more sophisticated test-time optimizers [4] and the formulation of novel, self-supervised online learning objectives [3, 32]. However, the computational efficiency of TTT remained a critical bottleneck due to its inherently sequential, per-token update process. The chunk wise Test-Time Training framework was the first to directly address this challenge by introducing a chunk-wise update mechanism to better leverage parallel hardware [3, 30, 36, 49, 63, 67]. Despite these advances, TTT’s function as the primary token mixer necessitates reliance on small chunks to preserve performance—this in turn creates a bottleneck that limits the massive parallelism needed to fully utilize modern accelerators. Furthermore, prior work has not addressed how to seamlessly integrate TTT into large, pre-trained models, nor developed learning objectives specifically tailored for the autoregressive nature of LLMs, which are gaps our work directly addresses.

> 💡 **相关工作定位**: 作者承认 chunk-wise TTT 已有，但强调既有方法仍常作为 primary token mixer，因此受小 chunk 限制；In-Place TTT 的差异是补充 attention、原位更新 MLP。

Efficient Long-Context Architectures. A parallel line of research seeks to extend the effective context window of LLMs by mitigating the quadratic complexity of the standard attention mechanism. Major approaches include: 1) Sparse attention methods, which restrict the range of token-to-token interactions via fixed patterns like sliding or strided windows [6, 10, 65]; 2) Linear-time variants, which approximate the attention mechanism or replace it with efficient recurrent or gated formulations, such as linear attention [33, 43] Gated Linear Attention (GLA) [58]; and 3) State-Space Models (SSMs), which compress sequence history into a compact latent state, enabling processing with linear complexity [16, 17]. Recently, the delta rule has emerged as a popular design choice for linear attention and SSMs, enabling better experessivity and highly parallelizable implementations [61]. These architectural advances are complementary to our framework. While they focus on efficiently processing long contexts, TTT provides a mechanism for online adaptation to the information within that context. Our In-Place TTT can be also naturally integrated with these efficient backbones, as they also have MLP blocks. And we leave these as the future work.

> 💡 **与长上下文架构关系**: Sparse/linear attention 和 SSM 解决计算复杂度，In-Place TTT 解决在线适应/记忆写入；两者不是替代关系。

Memory Design and Augmentation. A related domain of research involves augmenting neural architectures with explicit memory modules to enhance their reasoning and contextual understanding capabilities. These approaches can be broadly distinguished by their function: some are designed to store persistent, task-agnostic knowledge in an external memory bank, while others focus on capturing transient, data-dependent information from the immediate context [25, 34, 35, 55, 64]. The latter, contextual memories, have been implemented using various mechanisms, including recurrent state transitions, attention-based context aggregation in Transformers [14], and rapid, gradient-based updates to fast weights [61]. Test-Time Training (TTT) represents a powerful instance of this latter category, which conceptually extends the notion of a hidden state found in Recurrent Neural Networks (RNNs). Rather than compressing contextual history into a fixed-size activation vector, TTT designates a subset of the model’s own parameters—the fast weights—to function as a high-capacity, dynamic memory [2, 43]. These weights are updated on the fly at inference time, allowing the model to continuously internalize evolving contextual information and thereby function as an expressive, online evolving state.

> 💡 **记忆视角**: 这篇论文把 MLP 参数从静态知识库扩展为上下文动态记忆。它不像 RAG 外接显式库，也不像 RNN 压缩到 hidden state，而是直接写模型内部投影矩阵。

---

## 🔖 Section 总结

> 💡 **Section 小结**: Related Work 把 In-Place TTT 放在 TTT、efficient long-context 和 memory augmentation 交叉点：它不是替代 attention，而是给标准 MLP 加在线记忆写入。
