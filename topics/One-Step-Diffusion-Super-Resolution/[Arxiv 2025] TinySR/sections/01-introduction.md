[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
本节建立动机链：Real-ISR 需要高质量，单步方法解决效率，但 TinySR 还要补上本文关注的关键短板。
---

> 💡 **Q&A 批注记录**:
> - Q: TinySR 的核心不是新的生成先验吗？
> - A: 对，它更像系统压缩论文：核心贡献在剪枝和模块移除，让已有 one-step diffusion SR 更小更快。


Real-world image super-resolution (Real-ISR) [42, 54] aims to reconstruct high-fidelity images from low-quality observations corrupted by compound degradations including noise contamination, nonlinear blur, and compression artifacts. Recently, diffusion models (DMs) have demonstrated significant promise for Real-ISR by leveraging their powerful priors to effectively address complex degradation patterns while recovering realistic textures and details [19, 34]. Their impressive performance over GAN-based Real-ISR methods [30] has driven their widespread adoption in practical downstream applications. However, for resource-constrained environments, the iterative sampling nature and high computational demands of DMs fundamentally limit their deployment [8].
> 💡 **动机批读**: 这段建立 Real-ISR 的基本矛盾：diffusion prior 能补真实纹理，但多步采样和大模型让部署困难。TinySR 接下来不是提升生成先验，而是把已有 one-step diffusion SR 压到实时。


![Figure 1.](../images/dabe78bdfd7b5ed6fda4c34c8a0b6d75760ea013455ecdeb7c63ea6577d408e2.jpg)
*Figure 1.: Figure 1. A comprehensive comparison of recent Real-ISR models in terms of visual quality, inference time, computational cost (MACs), and parameter count, highlighting the superior efficiency and performance of our proposed method.*
> 💡 **Figure 批读**: Fig. 1 同时展示视觉质量、耗时、MACs 和参数量，是引言里最重要的 claim 图。读它时不要只看 TinySR 最快，还要看相对 teacher TSD-SR 的质量是否仍接近，以及是否出现过度生成纹理。


Significant efforts in reducing diffusion model sampling steps have dramatically improved the inference latency of DM-based Real-ISR approaches. Recent advancements in efficient Real-ISR models, such as OSEDiff [48] and TSD-SR [13] seek to condense the denoising process into a single step by carefully designed distillation while preserving high-quality restoration performance. However, these methods still depend on large pre-trained models, which require significant computational resources and footprint, posing challenges for real-time applications and edgedevice deployment. There is a critical demand for more efficient and compact models that can be readily achieved while maintaining competitive performance.
> 💡 **定位批读**: one-step distillation 已经解决了“步数慢”，但没有解决“网络大”。TinySR 的问题定义比 OSEDiff/TSD-SR 更工程化：在单步前提下继续压 backbone、VAE 和条件分支。


Diffusion compression primarily involves several key techniques, including operator optimization [11, 38], precision quantization [17, 27], width pruning [4, 8, 45], and depth pruning [10, 15, 25, 28]. Depth pruning serves as a simple but effective technique, favored for its linear acceleration and straightforward implementation. The conventional paradigm for depth pruning is dependent upon empirical [25] or importance-based metrics [16, 33] to guide layer selection, yet it overlooks the recoverability of the model’s performance on its target task after being pruned. Instead of relying on static importance scores, probability-based mask learning aims to iteratively refine this sampling distribution, such that layers with greater performance recoverability are more likely to be sampled. However, in ultra-deep networks, mask learning confronts a combinatorial explosion in the search space, which results in prohibitive optimization complexity and slow convergency [15].
> 💡 **剪枝动机**: 作者选择 depth pruning 是因为它能直接跳过 block，速度收益更接近线性。难点在于：静态重要性分数不等于剪完可恢复，因此引出后面的 mask learning、DIA 和 Expansion-Corrosion。


Beyond deep architecture limitation, DMs-based Real-ISR models exhibit two other computational bottlenecks: high overhead from VAE (Variational Auto-Encoder) [26] and inefficiencies from redundant condition modules during super-resolution process. AdcSR [5] eliminates the VAE encoder by employing a PixelUnshuffle operation [37]. However, fine-grained channel pruning is required in denoising networks to align channel dimensions, which significantly increases overall complexity and coupling between its components. Regarding prompts and time embeddings, recent studies [5, 13] indicate that these conditions contribute minimally to the one-step Real-ISR model, suggesting that a more efficient model can be achieved by eliminating these inputs and related modules.
> 💡 **系统瓶颈**: 这段很关键：TinySR 不满足于剪 denoiser，因为 latent diffusion 的 VAE 和条件模块也是推理图的一部分。time/prompt 在单步 SR 中贡献小，说明它们可以作为部署冗余被移除。


Based on these analyses, we propose TinySR, a compact yet effective DMs-based Real-SR model that eliminates computational redundancies in TSD-SR while maintaining restoration quality. Following mask learning, we propose identifying candidate layers that exhibit highperformance recoverability. We partition the network into non-overlapping blocks and introduce sets of learnable probabilities $p ( m )$ for each blcok to constrain search space. We propose Dynamic Inter-block Activation, a method that leverages the learnable probability $q ( t )$ for soft boundary exploration, and introduce a novel Expansion-Corrosion Strategy to determine the optimal pruning scheme. These proposed techniques involve a strategic trade-off between optimization complexity and exploratory potential. Furthermore, we propose several strategies to further compress the Real-ISR model. To lighten the VAE, we perform channel-wise pruning, remove its computationally intensive attention modules, and replace its standard convolutions with depthwise separable convolutions [20]. We eliminate time- and prompt-related modules to further enhance computational efficiency. Extensive experiments on standard Real-ISR benchmarks demonstrate that our method is up to $\pmb { 5 . 6 8 } \times$ faster, with $84 \%$ MAC and $83 \%$ parameter reductions compared to its teacher, TSD-SR, yet preserves strong perceptual quality (Fig. 1) and comparable quantitative results (Fig. 2) .
> 💡 **贡献批读**: 这里把方法拆成两组：剪枝算法负责 shallow denoiser，component streamlining 负责 VAE/condition/cache。DIA 的 $q(t)$ 是跨 block 探索，Expansion-Corrosion 是最终 mask 决策；两者都服务于“同等剪枝率下更可恢复”。


![Figure 2.](../images/c9674c1c807a86f0bc79aab15930fc9cf2952c7ec2be5dab37831ab0ee58f534.jpg)
*Figure 2.: Figure 2. Performance and efficiency comparison among DMsbased Real-ISR methods on an NVIDIA V100 GPU. All metrics are evaluated on the RealSR benchmark. TinySR achieves the fastest inference, lightest computation (low MACs) and commendable performance (high CLIPIQA).*
> 💡 **Figure 批读**: Fig. 2 是速度-质量散点读法：TinySR 要证明自己不是“快但画质差”，而是在 RealSR 上用更低 MACs/latency 取得可竞争的 CLIPIQA。这个图也提醒我们，评价重点是 trade-off 位置而非单指标第一。


Overall, our contribution is summarized as follows: • A Real-ISR model called TinySR that achieves $\pmb { 5 . 6 8 } \times$ speedup and $83 \%$ parameter reduction compared to its teacher, while maintaining a strong perceptual quality. • A novel depth pruning method that incorporates Dynamic Inter-block Activation and Expansion-Corrosion Strategy to enable more effective pruning decisions with preserved model performance. • Component streamlining strategies incorporate lightweight VAE, redundant conditional structures pruning, and modulation parameters pre-caching.
> 💡 **贡献核对**: 后面实验应分别对应三类证据：完整 TinySR 对 teacher/baseline 的速度质量对比、剪枝方法对其他 pruning baseline 的 recoverability、VAE 与 time/prompt/cache 的消融。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 任务 | Real-world image super-resolution |
| 主矛盾 | 效率 vs 质量 vs 可恢复性 |
| 本文回答 | 把 one-step diffusion SR teacher 系统性压缩成更小、更快的 student，目标是实时 Real-ISR。 |

### 核心洞察
1. Introduction 的逻辑链是：Real-ISR 需要 generative prior → 多步 diffusion 慢 → one-step 模型仍然太大 → TinySR 做系统级压缩。
2. TinySR 的差异化不是提出新 SR 先验，而是把 TSD-SR 的冗余结构分层清掉，并用 recoverability 约束剪枝。
3. 贡献列表要和实验模块一一对应：主表验证总体 trade-off，剪枝表验证 DIA/Expansion-Corrosion，消融验证 VAE 与条件模块。

### 可追问点
- TinySR 的核心不是新的生成先验吗？
- 为什么要同时压缩 VAE？
- 深度剪枝相比 width pruning 的线性加速优势，在不同 GPU/移动端上是否仍成立？
