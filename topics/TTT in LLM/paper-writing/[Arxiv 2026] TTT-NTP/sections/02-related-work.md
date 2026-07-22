[← 返回 README](../README.md)

# 2. Related Work（相关工作）

## 📌 预览

三条相关工作线，每条都在为 TTT-NTP 的定位铺路：
- **§2.1 TTT for LLMs**：fast-weight 的谱系（fast-weight programmers → linear transformers → SSM/DeltaNet → in-place TTT），指出「placement 和 update 已标准化，target 是开放轴」。
- **§2.2 NTP 与长上下文**：扩窗口、改 attention、continual pretraining 三条路都不能保证长上下文利用；TTT-NTP 与它们互补（保持 backbone 和窗口不变，只更新 fast-weight memory）。
- **§2.3 表示空间的预测目标**：BYOL/I-JEPA/DINOv2/MAE 的共识——predictive target 不必是 raw observation，representation-space target 更稳；TTT-NTP 把这个思想搬到 LLM fast-weight。

---

## 2.1 Test-Time Training for Language Models

Test-time training adapts a model on each test input using a self-supervised objective (Sun et al., 2020). In later architectural forms, the hidden state of a recurrent layer is itself a set of fast weights updated by an online learner (Sun et al., 2024). This view connects naturally to fast-weight programmers (Schmidhuber, 1992; Ba et al., 2016), the fast-weight interpretation of linear transformers (Schlag et al., 2021), modern Hopfield networks (Ramsauer et al., 2020), and linear-state sequence models such as linear attention (Katharopoulos et al., 2020), RetNet (Sun et al., 2023), RWKV (Peng et al., 2023), DeltaNet (Yang et al., 2024; 2025b), and state-space models (Gu & Dao, 2023; Dao & Gu, 2024). All maintain a compact state that is written by earlier tokens and read by later tokens.

> 💡 **机制拆解**（Hao 批注）: 这段建立了「fast weight = 被 earlier token 写、被 later token 读的紧凑 state」这条统一视角。它把 TTT、linear attention、SSM、DeltaNet 全部收编到「dynamic memory」框架下。对 TTT-NTP 的意义：它继承了这套 read/write 机制，但把注意力从「state 怎么更新」转到「state 里应该存什么值」。

Recent work makes TTT practical for long-context language models through large-chunk or in-place fast-weight updates (Zhang et al., 2025; Tandon et al., 2025; Behrouz et al., 2026); most directly, In-Place TTT (Feng et al., 2026) updates existing MLP down-projections with chunk-parallel rank-one writes. These methods largely fix the fast-weight location and the efficient update mechanics. Because the write mechanism and the placement are now largely standardized across recent in-place TTT recipes, the supervisory target is the open design axis where method choices most directly determine downstream performance. Prior in-place recipes use the current representation or a locally constructed value proxy, often built from neighboring activations by a small auxiliary network (Behrouz et al., 2026). TTT-NTP changes this supervision target. Instead of asking the fast weight to reconstruct a local proxy, we use the next-position contextual hidden state at the same layer: a model-native representation produced by the causal forward pass after observing the next prompt token.

> 💡 **机制拆解**（Hao 批注）: 这段是本文对自己「novelty」最直白的界定。作者主动承认：write mechanism（rank-one）和 placement（MLP down-proj）都是沿用 In-Place TTT（Feng et al. 2026），**不是我们的创新**。真正开放、且直接决定下游表现的轴是 supervisory target。baseline 用「current representation 或 locally constructed value proxy（邻域激活 + 小辅助网络）」；TTT-NTP 用「同层下一位置的 contextual hidden state」。这句话是理解全文贡献边界的关键——它把创新精准锁定在 target 一个维度。

## 2.2 Next-Token Prediction and Long Context

Next-token prediction is the standard objective for autoregressive language modeling, but optimizing the objective on long documents does not by itself ensure reliable long-context utilization. One line of work extends effective context windows through rotary position embeddings (Su et al., 2024), positional interpolation (Chen et al., 2023), frequency-aware rescaling (Peng et al., 2024; Ding et al., 2024), or ALiBi (Press et al., 2021), often combined with efficient attention kernels (Dao et al., 2022). A second line modifies attention with local or sparse patterns (Beltagy et al., 2020; Zaheer et al., 2021), or replaces attention with recurrent or linear-state mechanisms (Katharopoulos et al., 2020; Gu & Dao, 2023; Dao & Gu, 2024; Sun et al., 2023; Peng et al., 2023; Yang et al., 2024). A third performs continual pretraining on long documents with tuned data mixtures and long-context recipes (Chen et al., 2024; Fu et al., 2024; Grattafiori et al., 2024; Yang et al., 2025a).

> 💡 **机制拆解**（Hao 批注）: 三条已有路线的分类——(1) 扩 effective window（RoPE/PI/YaRN/ALiBi）；(2) 改 attention 结构（local/sparse 或换成 recurrent/linear-state）；(3) 长文档 continual pretraining。作者列这三条是为了说明 TTT-NTP 不属于任何一条：它不扩窗口、不换 attention、不靠更多长文档数据，而是「保持 backbone 和窗口不变，用 prompt 自己的 next-token 对更新 fast-weight memory」——这是下一段要点出的互补定位。

Despite these advances, evaluation suites such as RULER and LongBench show substantial degradation inside the nominal context window (Hsieh et al., 2024; Bai et al., 2024; Liu et al., 2024). The relevant evidence is often in the prompt, but the model fails to keep it available for the future next-token decisions that need it. TTT-NTP is complementary to context-window extension and data selection (Wang et al., 2026): it keeps the pretrained backbone and context window fixed, then uses the prompt’s own next-token pairs to update a compact fast-weight memory.

> 💡 **机制拆解**（Hao 批注）: 这里再次强调病灶（「证据在 prompt 里，但模型没把它 keep available 给需要它的未来 next-token 决策」）并给出 TTT-NTP 的互补性论点。「complementary」很重要：意味着 TTT-NTP 可以叠加在扩窗口/数据选择方法之上，而不是替代它们——这也是 Future Work 里「和 native long-context window、RAG pipeline 结合」的伏笔。

## 2.3 Predictive Targets in Representation Space

TTT-NTP uses an NTP-aligned signal but implements it through a representation-space target. This follows a broader lesson from self-supervised learning: predictive targets need not be raw observations. BYOL predicts another view’s embedding (Grill et al., 2020); I-JEPA predicts masked image regions in latent space (Assran et al., 2023); and DINOv2 uses large-scale latent-space distillation for visual representation learning (Oquab et al., 2024).

Masked autoencoders (He et al., 2022) reconstruct through a decoder bottleneck, sharing the idea that an intermediate representation can carry richer and more stable supervision than the input alone.

> 💡 **机制拆解**（Hao 批注）: 这是 TTT-NTP 最重要的「哲学辩护」：predictive target 不必是 raw observation。BYOL 预测另一视角的 embedding、I-JEPA 在 latent space 预测被 mask 的区域、DINOv2 用 latent-space distillation、MAE 通过 decoder bottleneck 重建——它们的共识是「intermediate representation 能比原始输入携带更丰富、更稳定的监督」。TTT-NTP 借此论证：与其把 vocabulary-level 的 raw target 写进 fast weight，不如写 representation-space 的 next hidden state。下一段解释为什么这对 MLP down-projection 尤其合适。

For LLM TTT, directly writing vocabulary-level targets into a layer-local fast weight is poorly matched to the MLP down-projection where the fast weight lives. The next contextual hidden state is a better interface: it is dense, same-dimensional as the MLP output stream, and causally determined by the next observed token. Unlike hidden-state distillation losses that only shape static parameters during training, TTT-NTP uses the next hidden state as the value target for a fast weight that is rewritten on each prompt.

> 💡 **公式批读**（Hao 批注）: 这段给出「为什么用 hidden state 而不是 vocab target」的三条工程理由，和 §3.2 的 fast-weight placement 直接呼应：
> 1. **维度匹配**：fast weight 住在 MLP down-projection，输出的是 $d$ 维的 MLP output stream；vocab-level target 是 $|V|$ 维的，接口不匹配。next hidden state $h_{\ell,t+1}$ 恰好是 $d$ 维，same-dimensional。
> 2. **dense**：每个位置都有，不像 vocab target 那样稀疏。
> 3. **causally determined**：由观测到的下一个 token 因果地决定，是 model-native 的。
> 关键区别：hidden-state distillation loss 只塑造训练时的 static 参数；TTT-NTP 把 next hidden state 当作**每个 prompt 都会重写的 fast weight 的 value target**——这是 dynamic 的、per-prompt 的。

---

## 🔖 Section 总结

### 核心洞察

1. **创新边界**：write mechanism 和 placement 沿用 In-Place TTT，唯一创新在 supervisory target（从 learned local proxy → model-native next-position state）。
2. **互补定位**：不扩窗口、不换 attention、不靠更多数据，而是固定 backbone 和窗口、用 prompt 的 next-token 对更新 fast-weight memory，可与其他长上下文方法叠加。
3. **表示空间监督的传承**：继承 BYOL/JEPA/DINOv2/MAE 的思想（predictive target 用 representation 而非 raw input），并论证 hidden state 与 MLP down-projection 的维度/密度/因果性都匹配。

### 可追问点

- In-Place TTT 的 learned convolution proxy 具体长什么样？→ §4.5.1 的 Past-5/Next-5/Bi-dir-5 就是这类 proxy 的对照实现。
- 「hidden-state distillation」和 TTT-NTP 的区别到底在哪？→ 前者改 static 参数，后者改 per-prompt 重写的 fast weight。
