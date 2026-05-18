[← 返回 README](../README.md)

# Abstract

> 📌 **Preview**: Cola DLM proposes a hierarchical latent diffusion language model that decomposes text generation into continuous latent prior modeling and conditional decoding, achieving strong scaling behavior as a principled alternative to token-level language modeling.

---

Large language models have achieved remarkable success under the autoregressive paradigm, yet high-quality text generation need not be tied to a fixed left-to-right order. Existing alternatives still struggle to jointly achieve generation efficiency, scalable representation learning, and effective global semantic modeling. We propose Cola DLM, a hierarchical latent diffusion language model that frames text generation through hierarchical information decomposition. Cola DLM first learns a stable text-to-latent mapping with a Text VAE, then models a global semantic prior in continuous latent space with a block-causal DiT, and finally generates text through conditional decoding. From a unified Markov-path perspective, its diffusion process performs latent prior transport rather than token-level observation recovery, thereby separating global semantic organization from local textual realization. This design yields a more flexible non-autoregressive inductive bias, supports semantic compression and prior fitting in continuous space, and naturally extends to other continuous modalities. Through experiments spanning 4 research questions, 8 benchmarks, strictly matched ~2B-parameter autoregressive and LLaDA baselines, and scaling curves up to about 2000 EFLOPs, we identify an effective overall configuration of Cola DLM and verify its strong scaling behavior for text generation. Taken together, the results establish hierarchical continuous latent prior modeling as a principled alternative to strictly token-level language modeling, where generation quality and scaling behavior may better reflect model capability than likelihood, while also suggesting a concrete path toward unified modeling across discrete text and continuous modalities.

> 💡 **问题动机**: 该文的核心洞察是：现有语言模型（无论是AR、离散扩散还是连续扩散）本质上都在做 token 级别的观测恢复（observation recovery），而不是显式地建模高层语义先验。Cola DLM 的动机是将"全局语义组织"和"局部文本实现"这两个层次解耦，让扩散过程只负责 latent prior transport，而非 token 级别的去噪。

> 💡 **机制拆解**: Abstract 中提出了三个关键组件：(1) Text VAE 建立 text-to-latent 映射；(2) block-causal DiT 在连续潜空间中对全局语义先验建模；(3) conditional decoder 实现文本生成。从 Markov-path 角度看，扩散过程的作用是"潜空间先验传输"而非"token 观测恢复"，这是与 LLaDA/Plaid 等方法的本质区别。

> 💡 **Q&A 批注记录**: Abstract 声称 "generation quality and scaling behavior may better reflect model capability than likelihood" -- 这是一个 strong claim。为什么 likelihood（PPL）不能反映模型能力？因为 Cola DLM 不是直接最大化 token 级别的 likelihood，而是通过 ELBO 优化一个分层目标（重建 + 压缩 + 先验匹配）。因此 PPL 衡量的是局部概率密度校准，而实际生成质量取决于先验是否覆盖了语义有效的潜空间区域。这个 mismatch 在 Section 5.1 中有详细的理论和实证分析。

---

🔖 **Summary**: This paper introduces Cola DLM, a hierarchical latent-variable language model that uses a Text VAE for text-to-latent mapping, a block-causal DiT for continuous latent prior modeling via flow matching, and a conditional decoder for text generation. The key innovation is using diffusion for latent prior transport rather than token-level observation recovery, validated through extensive experiments on 8 benchmarks at ~2B scale showing strong scaling behavior.
