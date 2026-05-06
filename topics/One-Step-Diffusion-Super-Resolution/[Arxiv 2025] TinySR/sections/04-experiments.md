[← 返回 README](../README.md)

# 4. Experiments

## 📌 预览
本节验证方法是否成立，重点看主结果、消融、效率和视觉/病例案例。

---

# 4.1. Experimental Settings

Datasets. We utilize DIV2K [1], Flickr2K [39], LSDIR [29], and FFHQ [23] for training. To synthesize lowresolution and high-resolution image pairs, we employ the same degradation pipeline as described in Real-ESRGAN [42]. We evaluate the performance of our model on the synthetic DIV2K-Val [1] dataset, alongside two real-world datasets, RealSR [3] and DRealSR [46]. The datasets consist of paired images with $1 2 8 \mathrm { x } 1 2 8$ low-quality and 512x512 high-quality resolutions.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Evaluation Metrics. For evaluating our method, we apply both full-reference and no-reference metrics. Full-reference metrics include PSNR and SSIM [44] (calculated on the Y channel in YCbCr space) for fidelity, LPIPS [56] and DISTS [12] for perceptual quality, and FID [18] for distribution comparison. No-reference metrics include NIQE [55], MUSIQ [24], MANIQA [51], CLIPIQA [40], TOPIQ [7] and Q-Align [47].

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

# 4.2. Comparison with Existing SR Methods

We categorize existing outstanding SR models into two groups, single-step and multi-step diffusion models. Singlestep models include SinSR [43], OSEDiff [48], TSD-SR [13], and AdcSR [5], and multi-step models include StableSR [41], DiffBIR [31], SeeSR [49], and ResShift [53]. Additional details of GAN-based Real-ISR methods [6, 30, 42, 54] are given in the supplementary material.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Quality Comparison. Tab. 1 shows a comparison with DMs-based baselines in Real-ISR tasks. For the first four full-reference metrics, our model achieves performance comparable to its teacher TSD-SR and outperforms most other models, securing the second rank for LPIPS and DISTS. Furthermore, our model achieves the best result for the distribution metric FID. For the latter three no-reference metrics, it demonstrates competitive performance: outperforming all other models for NIQE, and ranking second for both MUSIQ and CLIPIQA, thereby surpassing most other methods.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Table 1. Quantitative comparison of various methods on the DIV2K-Val dataset, with all efficiency metrics benchmarked on a NVIDIA V100 GPU. The best and second-best results are highlighted in bold, italic, respectively.

> 💡 **批注**: 这是实验证据段：同时看主指标、消融、效率和案例，判断 claim 是否被支撑。

![Table 1.](../images/4caebd55230a5d11f456f4c411ee5698bac5c5d6e5606b81d03368efca4266cc.jpg)
*Table 1.: Table 1. Quantitative comparison of various methods on the DIV2K-Val dataset, with all efficiency metrics benchmarked on a NVIDIA V100 GPU. The best and second-best results are highlighted in bold, italic, respectively.*

> 💡 **Table 1. 批读**: 表格要看主指标、次指标与效率/鲁棒性是否一致支持论文 claim。

As illustrated in Fig. 1 and Fig. 8, TinySR achieves competitive performance in recovering high-quality, sharp, and photorealistic images. Visual artifacts, specifically the generation of fake textures, are frequently observed in outputs from multi-step models such as StableSR, SeeSR, DiffBIR, and ResShift due to their propensity for over-generation. A clear example, demonstrated in Fig. 8 (top), is the spurious generation of hair patterns in regions where they should not exist. OSEDiff and AdcSR consistently demonstrate suboptimal restoration performance, frequently yielding outputs with discernible blur. While TSD-SR exhibits excellent generation capabilities, it also tends to produce fake textures, as shown in Fig. 8 (bottom). TinySR, by contrast, demonstrates robust capabilities in reconstructing natural textures, notably encompassing structural integrity, botanical patterns, and sculpted surface details.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Efficiency Comparison. As demonstrated by the last four columns of Tab. 1, the proposed TinySR exhibits superior efficiency in terms of step number, inference time, and computational cost. By leveraging one-step inference and advanced compression, our model achieves dramatic efficiency gains over leading multi-step Real-ISR methods while maintaining comparable performance. Compared to StableSR, SeeSR, DiffBIR, and ResShift, our model delivers significant speed improvements $4 8 9 \times$ , $2 2 0 \times$ , $2 4 7 \times$ , $2 9 \times$ ) with corresponding MAC reductions $1 8 7 \times$ , $1 5 4 \times$ , $5 7 \times$ , and $1 2 \times$ ). Compared to the one-step model SinSR and OSEDiff, it achieves $8 . 1 \times$ and $6 . 4 \times$ acceleration, respectively. Compared to its teacher, TSD-SR, it achieves a $5 . 6 8 \times$ acceleration, a $84 \%$ reduction in computation, and a

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Table 2. Performance comparison of depth pruning methods on the DIV2K-Val dataset. Our method exhibits superior recovery performance relative to other pruning strategies.

> 💡 **批注**: 这是实验证据段：同时看主指标、消融、效率和案例，判断 claim 是否被支撑。

![Table 2.](../images/fc649d2f3dd84d2d55fc64fa1601da1149277bb11fa0cba21f7c98ef6a50bd37.jpg)
*Table 2.: Table 2. Performance comparison of depth pruning methods on the DIV2K-Val dataset. Our method exhibits superior recovery performance relative to other pruning strategies.*

> 💡 **Table 2. 批读**: 表格要看主指标、次指标与效率/鲁棒性是否一致支持论文 claim。

$83 \%$ decrease in total parameters, as shown in Fig. 5. Notably, In a direct comparison with the current state-of-theart SR compression model, AdcSR, our model also demonstrates superior efficiency, with a $1 . 8 \times$ speedup and a $2 . 4 5 \times$ computation reduction.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

# 4.3. Comparison with Depth Pruning Methods

We evaluate the depth pruning methods following these baselines: (1) Perturbation-based – We randomly prune models and select one with minimal task loss (LPIPS) for training; (2) Similarity-based – Typically, these methods base their decisions on an analysis of the similarity between each layer’s input and output, such as Flux-Lite [10] and ShortGPT [33]; (3) Metric-based – Decision-making through metric in general, such as Sensitivity Analysis [16] and SnapFusion [28]. (4) Experience-based - We follow the design of BK-SDM [25] for corresponding pruning; (5) Probability-based – Decision by optimized probability parameters, such as TinyFusion [15] and our method. Randomly generating and then selecting the minimum loss yields a low initial loss, but exhibits extremely weak recovery ability after training. Similarity-based methods demonstrate suboptimal fidelity (poor DISTS) and pronounced inconsistencies across various no-reference metrics. For example, ShortGPT performs well on CLIPIQA but struggles considerably on MANIQA. Metric-based methods often exhibit a bias towards the metrics they optimize. For instance, while Sensitivity Analysis performs well on reference metrics, and SnapFusion excels on CLIPIQA due to its pruning scheme’s relation to it, both approaches demonstrate shortcomings in other metrics. Although BK-SDM and Tiny-Fusion show some effectiveness, our method exhibits enhanced recoverability over all other approaches, performing favorably across both full-reference and no-reference evaluation metrics.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Figure 8.](../images/d58ff4644c0fd5de45ad57b085de82bf6306d2d3c0d620419ceffb111fec6d5d.jpg)
*Figure 8.: Figure 8. Qualitative comparisons of different DMs-based Real-ISR methods. Please zoom in for a better view.*

> 💡 **Figure 8. 批读**: 这张图通常承担方法框架、动机或视觉对比作用；重点看它支撑的是机制、效果还是局限。

Table 3. Ablation study of VAE compression on DrealSR.

![Table 3.](../images/3a7fc9ac0df56fed68b7149205dbab53fb3016093831482b3e44877a21232a19.jpg)
*Table 3.: Table 3. Ablation study of VAE compression on DrealSR.*

> 💡 **Table 3. 批读**: 表格要看主指标、次指标与效率/鲁棒性是否一致支持论文 claim。

# 4.4. Ablation Study

Effect of VAE Compression. Tab. 3 presents the ablation studies of VAE compression process. Channel pruning offers a significant reduction in computational overhead, with only a minor compromise to perceptual reconstruction fidelity. While removing the attention module effectively doubles inference speed, it can somewhat impact reconstruction quality. For lightweight convolution, an alternative approach, employing SnapGen’s [8] strategy of expanding channels in SepConv intermediate layers, yields performance comparable to our proposed solution. However, our method notably exhibit reduced computational overhead and superior inference efficiency. Our efforts culminated in a lightweight VAE that delivers reconstruction quality on par with the teacher, concurrently achieving a $1 0 \times$ increase in inference speed and a $2 2 \times$ reduction in MACs.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Table 4. Ablation study of removing the text embeddings, time embeddings, and related modules on RealSR.

> 💡 **批注**: 这是实验证据段：同时看主指标、消融、效率和案例，判断 claim 是否被支撑。

![Table 4.](../images/fae10dce475dfde160c82a5587e693d9e18d422bfbf8057d503a72b74f797fa3.jpg)
*Table 4.: Table 4. Ablation study of removing the text embeddings, time embeddings, and related modules on RealSR.*

> 💡 **Table 4. 批读**: 表格要看主指标、次指标与效率/鲁棒性是否一致支持论文 claim。

Effect of Removing the Text and Time Modules. Tab. 4 presents the ablation study on the elimination of text and time conditions, which demonstrates that our approach achieves a highly favorable trade-off between efficiency and quality. Excising the text embeddings and related context modules yields a 486M parameter, 108G MACs, and 8ms time reduction, while only marginally decreasing the CLIP-IQA score by 0.0046 and the MANIQA score by 0.0048. Subsequent removal of the time modules further reduces parameters by 8M with a negligible impact on final quality.

> 💡 **批注**: 这是实验证据段：同时看主指标、消融、效率和案例，判断 claim 是否被支撑。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
