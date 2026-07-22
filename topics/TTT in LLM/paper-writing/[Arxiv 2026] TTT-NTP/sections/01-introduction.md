[← 返回 README](../README.md)

# 1. Introduction（引言）

## 📌 预览

Introduction 的论证结构：(1) NTP 是天然的 test-time 训练信号；(2) 长上下文场景最需要这种 adaptation——模型有大窗口却用不好窗口里的信息；(3) TTT 通过写 fast weight 改变 prompt 信息的表示方式；(4) 但现有 in-place TTT 的 inner-loop objective 和模型自己的预测目标 misaligned；(5) 引出 TTT-NTP，并说明为什么不能直接对全 NTP loss 求梯度（太贵、破坏 chunk-parallel），而是用 layer-local 的 next-position target。最后列出四条贡献。

---

Language models are trained by next-token prediction (NTP): given a prefix, predict the next observed token. The same self-supervised signal is present at test time for every token in a prompt. Before the model generates a response, the prompt itself provides many prefix–next-token pairs that can adapt the model to the current document, topic, style, or retrieval problem. This makes NTP a natural training signal for test-time training (TTT) in language models.

> 💡 **问题动机**（Hao 批注）: 这段是整篇文章的立论基石——「同一个信号，训练时和测试时都在」。训练时用 NTP 学 slow weights；测试时 prompt 里的每个 token 依然提供 (prefix, next-token) 对，只是这次拿来更新 fast weights。作者要说服你：既然模型就是靠 NTP 学出来的，那 test-time adapt 也应该用 NTP，而不是另造一个 proxy 目标。

The need for such adaptation is clearest in long-context settings. Modern large language models (LLMs) support long context windows (Grattafiori et al., 2024), and positionalextension methods (Su et al., 2024; Peng et al., 2024; Ding et al., 2024) and long-document continual pretraining (Chen et al., 2024; Fu et al., 2024) have substantially increased their nominal context lengths. Yet long-context benchmarks show that models still fail to use information that lies inside the available window (Liu et al., 2024; Hsieh et al., 2024; Bai et al., 2024). These failures are not simply failures of context capacity: the evidence is present, but the model’s computation does not carry it forward reliably enough to affect the next-token distribution.

> 💡 **机制拆解**（Hao 批注）: 关键判断——「不是窗口容量的问题，是 computation 没把证据 carry forward」。这句定义了本文要解决的病灶：信息就在窗口里（capacity 够），但模型的前向计算没能把它稳定地传递到影响 next-token 分布的地方。这正好给出了 fast-weight 的切入点：与其扩窗口，不如改「表示信息的方式」，让证据真的进到后续 token 能读到的 memory 里。

Test-time training offers a direct mechanism for changing how prompt information is represented during inference. Instead of relying only on attention over a fixed key–value cache, the model writes observations into fast weights (Schmidhuber, 1992; Ba et al., 2016; Ramsauer et al., 2020) that are read by later tokens. This dynamic-memory view is closely related to linear and recurrent sequence models (Katharopoulos et al., 2020; Gu & Dao, 2023; Dao & Gu, 2024; Sun et al., 2023; Peng et al., 2023; Yang et al., 2024) and to broader test-time memory systems (Behrouz et al., 2026). Recent TTT methods make fast-weight adaptation practical for pretrained LLMs by placing fast weights at selected MLP down-projections and accumulating chunk-parallel rank-one writes (Feng et al., 2026). However, the objective that drives these writes remains misaligned with the model’s own prediction objective. Existing recipes are trained with an outer language-modeling loss, but the inner loop writes a learned local activation proxy into the fast weight. The proxy is useful as an engineering device, but it is not the representation target that the model itself uses when performing next-token prediction.

> 💡 **机制拆解**（Hao 批注）: 这段是「fast weight = dynamic memory」的世界观，也是和 baseline 的分界线。
> - **fast weight 的角色**：不再只靠 attention 读固定的 KV cache，而是把观测「写」进一组会被后续 token「读」的快速权重。这与 linear attention / SSM / DeltaNet 等 linear-state 序列模型同源（都是 earlier tokens 写 state、later tokens 读 state）。
> - **In-Place TTT 已定的两件事**：placement 放在 MLP down-projection；update 用 chunk-parallel rank-one write。
> - **本文要改的一件事**：inner loop 写进去的是「learned local activation proxy」，和 outer 的 language-modeling loss 是两套目标——misaligned。proxy 是个 engineering trick，不是模型做 NTP 时真正用的表示。这句 misalignment 就是 TTT-NTP 的立足点。

We introduce Test-Time Training with Next-Token Prediction (TTT-NTP). TTT-NTP aligns the fast-weight inner loop with the same self-supervised prediction problem used to train the backbone. Directly taking full next-token cross-entropy gradients through every adapted layer at test time would be expensive and would destroy the chunk-parallel structure that makes in-place TTT attractive. TTT-NTP therefore uses a layer-local form of the NTP signal: at an adapted MLP block, the key is the current gated MLP activation, and the value target is the same layer’s contextual hidden state at the next position. This state is produced by the causal forward pass on the observed next token. Because it lies on the representation trajectory used to form later next-token distributions, it provides a dense, model-native target without aggregating over a neighborhood of positions.

> 💡 **机制拆解**（Hao 批注）: 这段是全文最关键的「设计取舍」，务必看清 TTT-NTP 为什么是 layer-local 而不是 end-to-end。
> - **为什么不直接对全 NTP cross-entropy 求梯度？** 两个致命问题：(1) 大 decoder 上贵；(2) 会破坏 chunk-parallel 结构——而 chunk-parallel 正是 in-place TTT 效率的来源。
> - **layer-local 近似**：在某个 adapted MLP block 上，key = 当前 gated MLP 激活 $z_{\ell,t}$；value target = 同一层在下一位置的 contextual state $h_{\ell,t+1}$。这个 state 是「观测到下一个 token 后，因果前向传播自然产生」的，无需反传整个网络。
> - **为什么这是好 target？** 它就在「形成后续 next-token 分布」的表示轨迹上，是 dense（每个位置都有）且 model-native（模型自己算出来的）的，**不需要在邻域上做聚合**（对比 baseline 的卷积 proxy）。

During continual pretraining, TTT-NTP learns this chunk-parallel causal write. At inference time, the same next-position rule yields a closed-form fast-weight write computed from cached gated activations, which we apply before decoding. On RULER Full-13 (averaged over 4k, 8k, 16k, and 32k context lengths), TTT-NTP is the only method that consistently improves the released backbone across four models from three families spanning 0.6–8B, and on the real-world LongBench-v2 long-document QA benchmark it is again the only method that improves over the base model on both Llama-3.1-8B and Mistral-7B-v0.3. Under matched data, compute, fast-weight placement, and chunk size, the prior in-place TTT method does not improve over the released backbone, supporting the importance of the next-token-prediction-aligned target. General capability on ARC (Clark et al., 2018), PIQA (Bisk et al., 2020), MMLU (Hendrycks et al., 2020), and HellaSwag (Zellers et al., 2019) remains comparable in aggregate.

> 💡 **机制拆解**（Hao 批注）: 这段点出「两阶段」结构——这是理解全文的骨架。
> - **训练阶段（CPT）**：把 chunk-parallel causal write 学出来（尤其学 $W_{\ell}^{\text{proj}}$）。
> - **推理阶段**：同一个 next-position 规则变成一个**闭式解（closed-form）**的 fast-weight write，从缓存的 gated 激活里算出来，在 decode 之前一次性打进 down-projection。
> - **最强的对照实验预告**：在 matched data/compute/placement/chunk size 下，In-Place TTT（差别只在 target）**不能**超过 released backbone。这就把提升干净地归因到「NTP-aligned target」本身，而不是数据、算力或 rank-one 机制。

## Contributions.

• We formulate TTT-NTP, a test-time-training framework that aligns fast-weight writes in pretrained LLMs with next-token prediction, instantiated with a layerlocal next-position target that pairs each current MLP activation with the next contextual hidden state produced by the causal forward pass.

• Across four backbones from three families spanning 0.6–8B, TTT-NTP is the only method that consistently improves RULER Full-13 over the released model—Llama-3.1-8B (+3.9), Mistral-7B-v0.3 (+3.0, an already-strong long-context baseline on which other baselines regress), Qwen3-4B (+4.1), and Qwen3-0.6B (+2.9)—while keeping aggregate commonsense and knowledge performance comparable.

• On the real-world LongBench-v2 long-document QA benchmark (medium split, 32k context budget), TTT-NTP attains the best overall accuracy on both backbones and is the only method that improves over Base on both—Llama-3.1-8B (+5.6) and Mistral-7B-v0.3 (+3.7).

• A target ablation isolates the supervisory signal from the rank-one machinery: under matched placement, update mechanics, data, and compute, next-position supervision outperforms Past-5, Next-5, and Bi-dir-5 convolutional targets by at least five RULER points at every evaluated length.

> 💡 **消融解读**（Hao 批注）: 四条贡献其实对应四类证据，读实验时可以按这个映射对号入座：
> 1. **方法（formulation）**：layer-local next-position target = 「当前 MLP 激活」配「下一位置 contextual state」。
> 2. **主结果（RULER）**：唯一全 backbone 提升；特别标注 Mistral 是「已经很强的长上下文 baseline，其他 baseline 在它上面会退步」——反衬 TTT-NTP 的鲁棒性。
> 3. **迁移（LongBench-v2）**：真实长文档 QA 上也是唯一双 backbone 提升，证明不是只对合成 needle 检索有效。
> 4. **target 消融**：把 supervisory signal 从 rank-one 机制里剥离出来——next-position 比 Past-5/Next-5/Bi-dir-5 三种卷积 target 在每个长度上至少高 5 个 RULER 点。这是全文最硬的因果归因。

---

## 🔖 Section 总结

### 核心洞察

1. **病灶诊断**：长上下文失败不是容量问题，而是「计算没把证据 carry forward 到影响 next-token 分布」。fast weight 提供了改「信息表示方式」的直接手段。
2. **misalignment 是切入点**：in-place TTT 的 outer loss 是 LM，inner loop 写的却是 learned proxy——两套目标。TTT-NTP 把 inner target 换成模型自己做 NTP 时的 next-position state。
3. **layer-local 是效率妥协**：直接反传全 NTP loss 太贵且破坏 chunk-parallel，所以用「同层下一位置 hidden state」作为 NTP 的 layer-local 代理。

### 可追问点

- 「下一位置的 hidden state」是否会泄露未来信息（因为它由 $x_{t+1}$ 算出）？→ §3.3 用 $\mathcal{T}_c^-$（排除 chunk 最后一个 token）+ 因果 chunk 结构保证 write 不影响 $t+1$ 自己的预测。
- 为什么 In-Place TTT 在强 baseline（Mistral）上会退步？→ §4.3 会给出「naive adaptation 反而 backfire」的解释。
