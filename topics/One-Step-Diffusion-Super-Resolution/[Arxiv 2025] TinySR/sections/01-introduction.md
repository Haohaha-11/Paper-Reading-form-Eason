[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
本节建立研究动机、现有方法缺口和贡献列表，是理解论文叙事的入口。

---

Real-world image super-resolution (Real-ISR) [42, 54] aims to reconstruct high-fidelity images from low-quality observations corrupted by compound degradations including noise contamination, nonlinear blur, and compression artifacts. Recently, diffusion models (DMs) have demonstrated significant promise for Real-ISR by leveraging their powerful priors to effectively address complex degradation patterns while recovering realistic textures and details [19, 34]. Their impressive performance over GAN-based Real-ISR methods [30] has driven their widespread adoption in practical downstream applications. However, for resource-constrained environments, the iterative sampling nature and high computational demands of DMs fundamentally limit their deployment [8].

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Figure 1.](../images/dabe78bdfd7b5ed6fda4c34c8a0b6d75760ea013455ecdeb7c63ea6577d408e2.jpg)
*Figure 1.: Figure 1. A comprehensive comparison of recent Real-ISR models in terms of visual quality, inference time, computational cost (MACs), and parameter count, highlighting the superior efficiency and performance of our proposed method.*

> 💡 **Figure 1. 批读**: 这张图通常承担方法框架、动机或视觉对比作用；重点看它支撑的是机制、效果还是局限。

Significant efforts in reducing diffusion model sampling steps have dramatically improved the inference latency of DM-based Real-ISR approaches. Recent advancements in efficient Real-ISR models, such as OSEDiff [48] and TSD-SR [13] seek to condense the denoising process into a single step by carefully designed distillation while preserving high-quality restoration performance. However, these methods still depend on large pre-trained models, which require significant computational resources and footprint, posing challenges for real-time applications and edgedevice deployment. There is a critical demand for more efficient and compact models that can be readily achieved while maintaining competitive performance.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Diffusion compression primarily involves several key techniques, including operator optimization [11, 38], precision quantization [17, 27], width pruning [4, 8, 45], and depth pruning [10, 15, 25, 28]. Depth pruning serves as a simple but effective technique, favored for its linear acceleration and straightforward implementation. The conventional paradigm for depth pruning is dependent upon empirical [25] or importance-based metrics [16, 33] to guide layer selection, yet it overlooks the recoverability of the model’s performance on its target task after being pruned. Instead of relying on static importance scores, probability-based mask learning aims to iteratively refine this sampling distribution, such that layers with greater performance recoverability are more likely to be sampled. However, in ultra-deep networks, mask learning confronts a combinatorial explosion in the search space, which results in prohibitive optimization complexity and slow convergency [15].

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Beyond deep architecture limitation, DMs-based Real-ISR models exhibit two other computational bottlenecks: high overhead from VAE (Variational Auto-Encoder) [26] and inefficiencies from redundant condition modules during super-resolution process. AdcSR [5] eliminates the VAE encoder by employing a PixelUnshuffle operation [37]. However, fine-grained channel pruning is required in denoising networks to align channel dimensions, which significantly increases overall complexity and coupling between its components. Regarding prompts and time embeddings, recent studies [5, 13] indicate that these conditions contribute minimally to the one-step Real-ISR model, suggesting that a more efficient model can be achieved by eliminating these inputs and related modules.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Based on these analyses, we propose TinySR, a compact yet effective DMs-based Real-SR model that eliminates computational redundancies in TSD-SR while maintaining restoration quality. Following mask learning, we propose identifying candidate layers that exhibit highperformance recoverability. We partition the network into non-overlapping blocks and introduce sets of learnable probabilities $p ( m )$ for each blcok to constrain search space. We propose Dynamic Inter-block Activation, a method that leverages the learnable probability $q ( t )$ for soft boundary exploration, and introduce a novel Expansion-Corrosion Strategy to determine the optimal pruning scheme. These proposed techniques involve a strategic trade-off between optimization complexity and exploratory potential. Furthermore, we propose several strategies to further compress the Real-ISR model. To lighten the VAE, we perform channel-wise pruning, remove its computationally intensive attention modules, and replace its standard convolutions with depthwise separable convolutions [20]. We eliminate time- and prompt-related modules to further enhance computational efficiency. Extensive experiments on standard Real-ISR benchmarks demonstrate that our method is up to $\pmb { 5 . 6 8 } \times$ faster, with $84 \%$ MAC and $83 \%$ parameter reductions compared to its teacher, TSD-SR, yet preserves strong perceptual quality (Fig. 1) and comparable quantitative results (Fig. 2) .

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Figure 2.](../images/c9674c1c807a86f0bc79aab15930fc9cf2952c7ec2be5dab37831ab0ee58f534.jpg)
*Figure 2.: Figure 2. Performance and efficiency comparison among DMsbased Real-ISR methods on an NVIDIA V100 GPU. All metrics are evaluated on the RealSR benchmark. TinySR achieves the fastest inference, lightest computation (low MACs) and commendable performance (high CLIPIQA).*

> 💡 **Figure 2. 批读**: 这张图通常承担方法框架、动机或视觉对比作用；重点看它支撑的是机制、效果还是局限。

Overall, our contribution is summarized as follows: • A Real-ISR model called TinySR that achieves $\pmb { 5 . 6 8 } \times$ speedup and $83 \%$ parameter reduction compared to its teacher, while maintaining a strong perceptual quality. • A novel depth pruning method that incorporates Dynamic Inter-block Activation and Expansion-Corrosion Strategy to enable more effective pruning decisions with preserved model performance. • Component streamlining strategies incorporate lightweight VAE, redundant conditional structures pruning, and modulation parameters pre-caching.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
