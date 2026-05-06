[← 返回 README](../README.md)

# 5. Analysis and Discussion

## 📌 预览
本节是 CODSR 最值得细读的部分：它逐个拆 RGPA、LQFM、TMG 和 VSD，说明哪些模块负责真实感、哪些模块负责保真，以及 timestep 如何显式改变 fidelity-realism trade-off。
---

> 💡 **Q&A 批注记录**:
> - Q: TMG 是否只是锦上添花？
> - A: 如果只追求基础重建，RGPA+LQFM 已经覆盖主要路径；TMG 的价值在语义纹理归位，比如 bird 的 attention 和羽毛生成，不是让所有像素都更接近 GT。


Effect of region-adaptive generative prior activation. To validate the effectiveness of the proposed RGPA, we first compare a baseline method stripped of all the proposed modules of RGPA, LQFM, and TMG (Base for short) against a variant incorporating only the RGPA $( B a s e _ { w / R G P A }$ for short). Table 2 shows the quantitative comparison, where we find that employing RGPA facilitates the release of generative priors for enhanced perceptual quality, cf. 65.89/0.5038 for Base vs. 66.27/0.5109 for $B a s e _ { w / R G P A }$ in terms of MUSIQ/CLIPIQA+.
> 💡 **RGPA 消融**: Base → Base_w/RGPA 主要提升 MUSIQ/CLIPIQA+，说明加区域噪声确实释放了更多生成先验。这里的提升偏 perceptual，而不是单纯重建误差下降。


Table 2. Effectiveness of each module in our proposed network. All methods are trained using the same settings as the proposed framework for fair comparison.   
Table 3. Quantitvative comparison of different noise strategies on the DrealSR [49] benchmark.
> 💡 **表格定位**: Table 2 看模块增量，Table 3 看噪声策略。两个表合起来回答“是否需要噪声”和“为什么必须是 latent-space adaptive noise”。


![Table 2.](../images/b72fd172a10779904878480779dc5859ba07f8db5abb75ba03ac56ae3076c852.jpg)
*Table 2.: Table 2. Effectiveness of each module in our proposed network. All methods are trained using the same settings as the proposed framework for fair comparison. Table 3. Quantitvative comparison of different noise strategies on the DrealSR [49] benchmark.*
> 💡 **表格批读**: Table 2 应按 Base、+RGPA、+LQFM、+TMG 读，每加一个模块都问它主要推高哪类指标。若某模块只提升 no-reference，说明它更偏 realism；若同时改善 full-reference，说明它也补了 fidelity。


To further analyze the effect of RGPA, we compare $B a s e _ { w / R G P A }$ with two baselines that individually add an adaptive noise $\epsilon _ { a }$ to the input LQ image $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ $( B a s e _ { w / \pmb { x } _ { L } + \pmb { \epsilon } _ { a } }$ for short) or replace the adaptive noise with standard Gaussian noise $\epsilon$ in the latent space $( B a s e _ { w / z _ { L } + \epsilon }$ for short). As shown in Table 3, adding noise $\epsilon _ { a }$ to $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ fails to enhance the generative capability, since perturbing the image space does not align the noise pathway between the forward and reverse processes in the latent space. Although adding standard Gaussian noise $\epsilon$ to the latent code $z _ { L }$ can enhance generative prior utilization, it significantly compromises fidelity, introducing artifacts in flat regions, as shown in Figure 4(b). In contrast, the method with our proposed RGPA achieves a superior balance, enabling region-adaptive activation of generative priors while faithfully preserving local structures (Figure 4(c)).
> 💡 **噪声策略批读**: 这段是 RGPA 最强的论证：像素空间加 adaptive noise 不对齐 latent reverse pathway；latent 标准噪声能增强生成但牺牲平坦区保真；latent adaptive noise 才同时满足“激活 prior”和“保护局部结构”。


We additionally investigate the impact of adaptive noise $\epsilon _ { a }$ intensity on reconstruction by adjusting the timestep $t _ { s }$ during the forward process. The trend in Figure 6 demonstrates a clear trade-off between fidelity and quality: higher $t _ { s }$ values (i.e., greater noise intensity) enhance the release of generative priors, as illustrated in the increasing MUSIQ values, but concurrently degrade fidelity, as evidenced by the decrease in PSNR. This property is characterized by a transition from fidelity-preserving reconstruction at lower noise levels to perceptually-oriented generation at higher levels. To balance these competing objectives, we set $t _ { s } =$ 100 as the default setting for subsequent experiments.
> 💡 **trade-off 批读**: Figure 6 把控制变量讲透了：t_s 越高，MUSIQ 越好但 PSNR 越差。这不是副作用，而是 CODSR 明确承认的 realism-fidelity 旋钮；默认 t_s=100 是经验折中点。


Effect of LQ-guided feature modulation. To validate the effectiveness of our LQFM, we compare $B a s e _ { w / R G P A }$ with a variant that is augmented with the LQFM module $( B a s e _ { w / R G P A \& L Q F M }$ for short). As shown in Table 4, using LQFM yields substantial improvements in both fidelity and perceptual metrics. We further compare $B a s e _ { w / R G P A \& L Q F M }$ with two variants that respectively replace the modulation in (4) with the element-wise addition ${ \pmb f } ^ { m } + \mathrm { M L P } ( { \pmb x } _ { L } )$ $( B a s e _ { w / R G P A \mathcal { E } f ^ { m } + M L P ( \widetilde { \pmb { x } } _ { L } ) }$ for short) or $\mathrm { S F T } ( z _ { L } \quad \vert \quad \widetilde { \pmb { x } } _ { L } )$
> 💡 **LQFM 消融**: LQFM 的关键不是“有 LQ 条件”这么简单，而是条件怎么融合。它提升 fidelity 的同时维持 perceptual metrics，说明中间 feature 的 SFT 调制比直接相加或调 latent 更稳。


![Figure 5.](../images/36585fda7887f040e9139409883803668254f190fc362e4e65c694ce05bea213.jpg)
*Figure 5.: Figure 5. Visualization comparison of DAAMs [41] for the query word “bird”. (b) and (c) are DAAMs for OSEDiff [50] and our method. Compared to [50], our method achieves more precise text–image interaction in the semantic region corresponding to the bird, resulting in more realistic feather generation.*
> 💡 **Figure 批读**: Figure 5 不是普通视觉对比，而是 attention 证据：query word “bird” 在 CODSR 中更集中到鸟区域，因此羽毛纹理生成更有语义归属。它直接支撑 TMG，而不是支撑 LQFM/RGPA。


![Figure 6.](../images/1c85919eda2d1d27a1d008c1c33d630f10163b029a2b6f5df268741fe3e13051.jpg)
*Figure 6.: Figure 6. PSNR and MUSIQ metrics variation with different timestep on the DrealSR [49] benchmark.*
> 💡 **Figure 批读**: Figure 6 是局部控制之外的全局控制证据：t_s 越大，模型越从“忠实复原”转向“感知生成”。如果应用场景对文字/身份敏感，应选择更低 t_s 或额外加结构约束。


Table 4. Comparison of different modulation strategies on the DrealSR [49] benchmark.
> 💡 **调制策略批读**: Table 4 的问题不是 LQ 信息有没有用，而是插在哪里、怎么插。直接 addition 受 domain gap 影响，调 z_L 破坏 latent 分布，调 U-Net 中间 f_m 是本文选择的折中。


![Table 4.](../images/d8140dd6bdf3922c8269774397364aa48c5eb832b6b1b56a567a22de9e480148.jpg)
*Table 4.: Table 4. Comparison of different modulation strategies on the DrealSR [49] benchmark.*
> 💡 **表格批读**: 看 Table 4 时重点比较三种融合方式对 no-reference metrics 的影响：如果保真提升但感知崩，说明 LQ 条件压制了 diffusion prior；CODSR 的 LQFM 目标就是避免这种崩塌。


$( B a s e _ { w / R G P A \& S F T } ( z _ { L } | \widetilde { \pmb { x } } _ { L } )$ for short). The comparison results ein Table 4 show that using element-wise addition performs suboptimally, even degrading no-reference metrics compared to $B a s e _ { w / R G P A }$ . We attribute this to the domain gap between the LQ input and U-Net features, which introduces instability into the denoising process and can lead to its eventual collapse of the denoising paradigm. Moreover, modulating the latent feature $z _ { L }$ causes a more pronounced drop in no-reference metrics than modulating the intermediate feature $\pmb { f } ^ { m }$ , as it disrupts the Gaussian diagonal distribution of $z _ { L }$ and consequently limits the effective utilization of generative priors during the reverse process. In contrast, our proposed LQFM effectively enhances restoration fidelity while maintaining better generative performance.
> 💡 **机制解释**: 作者把失败归因到 domain gap 和 latent 分布破坏，这与方法设计一致：LQ feature 与 U-Net feature 不是同分布，不能粗暴相加；z_L 也不是普通特征图，不能随意做 SFT。


Effect of text-matching guidance. To demonstrate the effectiveness of the TMG, we compare our full model with $B a s e _ { w / R G P A \& L Q F M } .$ . As shown in Table 2, our approach with TMG yields further improvements across all non-reference metrics, underscoring its critical role in enhancing the generation capability of the model. Figure 5 visualizes diffusion attentive attribution maps (DAAMs) of our method and [50], where the method [50] suffers from text misalignment even with the regularization of pre-trained models. In contrast, our proposed TMG effectively alleviates the misalignment issue, leading to more accurate and consistent text–image interactions.
> 💡 **TMG 消融**: TMG 加上后主要推动 no-reference 指标，符合它的职责：不是重建更多 GT 像素，而是让语义纹理更自然、更落在正确区域。Figure 5 的 DAAM 是比普通指标更直接的证据。


Table 5. Comparison of different semantic enhancement strategies on the DrealSR [49] benchmark.
> 💡 **VSD 表格定位**: Table 5 专门看第二阶段语义增强策略。它回答的是：继续用 GAN loss 是否足够，还是必须从 pretrained diffusion teacher 蒸馏语义 prior。


![Table 5.](../images/7ad6dd87e03f217a2f4af149f846b6eac82473492964a803f0f84327205d90a7.jpg)
*Table 5.: Table 5. Comparison of different semantic enhancement strategies on the DrealSR [49] benchmark.*
> 💡 **表格批读**: GAN loss 偏局部真实纹理，VSD 更像从大 diffusion model 借语义分布梯度。若 Table 5 中 VSD 明显改善 no-reference/语义指标，就说明第二阶段不是重复第一阶段，而是补 semantic generation。


To further investigate the influence of semantic knowledge distillation in the second training stage, we compare our CODSR with a baseline that replaces the VSD loss in (5) with the GAN loss $\left( O u r s _ { w / o } \ : V S D \ : l o s s \right.$ for short). As shown in Table 5, using GAN Loss fails to fully unleash the semantic generation capability due to the lack of effective semantic guidance in the second-stage training, whereas the VSD Loss enables semantic knowledge distillation from the pretrained model, thereby enhancing the generative ability.
> 💡 **VSD 批注**: 这段也暴露风险：VSD 释放的是 teacher 的语义生成能力，不天然保证忠实于 LQ。因此它需要和 LQFM/TMG 搭配，否则可能变成“更会画”但不一定“更会复原”。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 读表顺序 | Table 2 模块增量 → Table 3 噪声策略 → Figure 6 t_s trade-off → Table 4 调制策略 → Table 5 VSD |
| 核心检查 | 每个模块是否对应一个明确短板，并有指标/图像证据 |
| 风险 | 生成先验越强，越需要 LQFM/TMG 防止结构和语义漂移 |

### 核心洞察
1. RGPA 的证据链最完整：模块增量、噪声策略对比、平坦区视觉案例、t_s 曲线都有覆盖。
2. LQFM 的关键证据在“调制中间特征优于直接相加/调 latent”，这比简单加一个 LQ 分支更有说服力。
3. TMG/VSD 主要证明语义生成能力，但也最依赖外部 prompt、mask 和 teacher prior，属于收益与风险并存的部分。

### 可追问点
- RGPA 和普通加噪有什么区别？
- 为什么还要 LQFM？
- t_s=100 是否对所有图像和区域都合适？
