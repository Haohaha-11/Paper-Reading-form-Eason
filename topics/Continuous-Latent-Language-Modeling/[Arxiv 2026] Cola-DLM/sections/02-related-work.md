[← 返回 README](../README.md)

# 2 Related Work

> 📌 **Preview**: 此节系统回顾了 AR 语言模型、离散扩散语言模型、连续扩散语言模型三类工作，每类子节以"In contrast, Cola DLM..."结尾，明确给出了 Cola DLM 与各类方法的本质区别。

---

## 2.1 Autoregressive Language Models

Autoregressive language models [56, 77, 92, 101] factorize the discrete text distribution by the chain rule and are trained with token-level maximum likelihood, making them the most widely adopted paradigm for text modeling. Their limitations are that generation is constrained by a fixed left-to-right order, inference is inherently sequential, and they are less suitable for non-monotonic generation tasks such as infilling, local editing, and global reorganization. In contrast, Cola DLM first models a global semantic prior in a continuous latent space and then performs conditional decoding, thereby alleviating token-level ordering bias and improving generation efficiency with a block-causal DiT.

> 💡 **对比分析**: AR 的核心优势是直接的 token 级别似然训练和评估，但代价是 hard-coded left-to-right inductive bias。Cola DLM 的关键改进不是"不用 AR"，而是"将顺序 bias 从全局语义层移除"：语义组织在潜空间进行（无 token 顺序约束），文本实现由 conditional decoder 完成（可以使用 AR decoder）。

## 2.2 Discrete Diffusion Language Models

Discrete diffusion language models mainly fall into two categories. The first category is based on discrete transition kernels [2, 10, 88], which define forward perturbation and reverse recovery in discrete token space and achieve non-autoregressive generation through multi-step denoising; however, sampling is usually slow and these methods cannot easily exploit the smooth semantic structure of continuous spaces. The second category is based on masking or absorbing states [69, 70, 80, 81, 84, 105, 113, 114, 117, 118], which construct training objectives by progressively mapping tokens to masks or absorbing states and then recovering the original text; however, information loss in intermediate states limits global semantic planning and fine-grained control. In contrast, Cola DLM moves the diffusion process to a continuous latent space, where compressible latent variables carry global semantics, thus combining the manipulability of continuous spaces with hierarchical semantic modeling.

> 💡 **对比分析**: 离散扩散的两类方法（transition kernel 和 masking）虽然移除了 left-to-right 顺序，但仍有根本局限：(1) 扩散仍在离散 token 空间进行——中间状态是离散的，不适合表示平滑的语义结构；(2) 目标仍是 observation recovery——从噪声/掩码状态恢复到原始 token。Cola DLM 将扩散移至连续潜空间，潜变量携带的是压缩后的全局语义，而非 token 级别的身份信息。

## 2.3 Continuous Diffusion Language Models

Continuous diffusion language models can be broadly divided into three categories. The first category consists of high-dimensional vocabulary-aligned continuous methods [31, 43, 59, 79], which perform continuous diffusion or flow modeling directly on one-hot vectors, logit simplexes, or probability simplexes to preserve alignment with discrete vocabularies; however, their representation dimension scales with vocabulary size, which limits scalability. The second category consists of token-embedding-based continuous methods [14, 21, 24, 27, 29, 52, 54, 87], which first map text into continuous embedding spaces and then apply diffusion or flow modeling to improve generation flexibility; however, their generation process remains essentially the recovery of noisy target representations, lacking an explicit hierarchical latent-variable interpretation and a unified marginal-likelihood view of text distributions. The third category consists of latent-space continuous methods [44, 58, 63, 109], which compress text into latent spaces with autoencoders or VAEs and then perform diffusion modeling. These methods typically rely on latent-space design and autoregressive decoders, and usually treat the latent space as a fixed representation rather than modeling it under a hierarchical latent-variable framework. In contrast, Cola DLM explicitly separates global semantics from local realization through hierarchical latent-variable modeling, and learns a semantic prior in a dynamic continuous latent space, thereby better balancing modeling flexibility, inference efficiency, and theoretical interpretability.

> 💡 **对比分析**: 连续扩散的三个子类：(1) vocabulary-aligned —— 维度与词表大小成比例，不可扩展；(2) token-embedding-based —— 虽然使用了连续空间，但扩散目标仍然是恢复 noisy target representations（如 Plaid），缺乏显式的分层潜变量解释；(3) latent-space —— 使用 autoencoder 压缩，但通常将潜空间视为固定表示，而非在一个分层概率框架中建模。Cola DLM 与第三类的区别在于：Cola DLM 的潜空间不是固定的，而是与 DiT 共同演化的（dynamic latent space），同时整个框架由 ELBO 严格推导，具有完整的分层概率解释。

---

🔖 **Summary**: This section positions Cola DLM against three existing paradigms. AR models suffer from fixed-order inductive bias. Discrete diffusion removes order but keeps diffusion in token space, unsuitable for semantic representation. Continuous diffusion (vocabulary-aligned, embedding-based, latent-space) improves on continuity but still performs observation recovery or treats latent space as fixed. Cola DLM uniquely combines hierarchical latent-variable modeling, continuous latent prior transport, and dynamic co-evolving latent space, achieving a principled balance of flexibility, efficiency, and theoretical rigor.
