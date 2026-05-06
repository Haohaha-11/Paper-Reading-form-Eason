[← 返回 README](../README.md)

# 5. Analysis and Discussion

## 📌 预览
本节验证方法是否成立，重点看主结果、消融、效率和视觉/病例案例。

---

Effect of region-adaptive generative prior activation. To validate the effectiveness of the proposed RGPA, we first compare a baseline method stripped of all the proposed modules of RGPA, LQFM, and TMG (Base for short) against a variant incorporating only the RGPA $( B a s e _ { w / R G P A }$ for short). Table 2 shows the quantitative comparison, where we find that employing RGPA facilitates the release of generative priors for enhanced perceptual quality, cf. 65.89/0.5038 for Base vs. 66.27/0.5109 for $B a s e _ { w / R G P A }$ in terms of MUSIQ/CLIPIQA+.

> 💡 **批注**: 这是实验证据段：同时看主指标、消融、效率和案例，判断 claim 是否被支撑。

Table 2. Effectiveness of each module in our proposed network. All methods are trained using the same settings as the proposed framework for fair comparison.   
Table 3. Quantitvative comparison of different noise strategies on the DrealSR [49] benchmark.

> 💡 **批注**: 这是实验证据段：同时看主指标、消融、效率和案例，判断 claim 是否被支撑。

![Table 2.](../images/b72fd172a10779904878480779dc5859ba07f8db5abb75ba03ac56ae3076c852.jpg)
*Table 2.: Table 2. Effectiveness of each module in our proposed network. All methods are trained using the same settings as the proposed framework for fair comparison. Table 3. Quantitvative comparison of different noise strategies on the DrealSR [49] benchmark.*

> 💡 **Table 2. 批读**: 表格要看主指标、次指标与效率/鲁棒性是否一致支持论文 claim。

To further analyze the effect of RGPA, we compare $B a s e _ { w / R G P A }$ with two baselines that individually add an adaptive noise $\epsilon _ { a }$ to the input LQ image $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ $( B a s e _ { w / \pmb { x } _ { L } + \pmb { \epsilon } _ { a } }$ for short) or replace the adaptive noise with standard Gaussian noise $\epsilon$ in the latent space $( B a s e _ { w / z _ { L } + \epsilon }$ for short). As shown in Table 3, adding noise $\epsilon _ { a }$ to $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ fails to enhance the generative capability, since perturbing the image space does not align the noise pathway between the forward and reverse processes in the latent space. Although adding standard Gaussian noise $\epsilon$ to the latent code $z _ { L }$ can enhance generative prior utilization, it significantly compromises fidelity, introducing artifacts in flat regions, as shown in Figure 4(b). In contrast, the method with our proposed RGPA achieves a superior balance, enabling region-adaptive activation of generative priors while faithfully preserving local structures (Figure 4(c)).

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

We additionally investigate the impact of adaptive noise $\epsilon _ { a }$ intensity on reconstruction by adjusting the timestep $t _ { s }$ during the forward process. The trend in Figure 6 demonstrates a clear trade-off between fidelity and quality: higher $t _ { s }$ values (i.e., greater noise intensity) enhance the release of generative priors, as illustrated in the increasing MUSIQ values, but concurrently degrade fidelity, as evidenced by the decrease in PSNR. This property is characterized by a transition from fidelity-preserving reconstruction at lower noise levels to perceptually-oriented generation at higher levels. To balance these competing objectives, we set $t _ { s } =$ 100 as the default setting for subsequent experiments.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Effect of LQ-guided feature modulation. To validate the effectiveness of our LQFM, we compare $B a s e _ { w / R G P A }$ with a variant that is augmented with the LQFM module $( B a s e _ { w / R G P A \& L Q F M }$ for short). As shown in Table 4, using LQFM yields substantial improvements in both fidelity and perceptual metrics. We further compare $B a s e _ { w / R G P A \& L Q F M }$ with two variants that respectively replace the modulation in (4) with the element-wise addition ${ \pmb f } ^ { m } + \mathrm { M L P } ( { \pmb x } _ { L } )$ $( B a s e _ { w / R G P A \mathcal { E } f ^ { m } + M L P ( \widetilde { \pmb { x } } _ { L } ) }$ for short) or $\mathrm { S F T } ( z _ { L } \quad \vert \quad \widetilde { \pmb { x } } _ { L } )$

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Figure 5.](../images/36585fda7887f040e9139409883803668254f190fc362e4e65c694ce05bea213.jpg)
*Figure 5.: Figure 5. Visualization comparison of DAAMs [41] for the query word “bird”. (b) and (c) are DAAMs for OSEDiff [50] and our method. Compared to [50], our method achieves more precise text–image interaction in the semantic region corresponding to the bird, resulting in more realistic feather generation.*

> 💡 **Figure 5. 批读**: 这张图通常承担方法框架、动机或视觉对比作用；重点看它支撑的是机制、效果还是局限。

![Figure 6.](../images/1c85919eda2d1d27a1d008c1c33d630f10163b029a2b6f5df268741fe3e13051.jpg)
*Figure 6.: Figure 6. PSNR and MUSIQ metrics variation with different timestep on the DrealSR [49] benchmark.*

> 💡 **Figure 6. 批读**: 这张图通常承担方法框架、动机或视觉对比作用；重点看它支撑的是机制、效果还是局限。

Table 4. Comparison of different modulation strategies on the DrealSR [49] benchmark.

![Table 4.](../images/d8140dd6bdf3922c8269774397364aa48c5eb832b6b1b56a567a22de9e480148.jpg)
*Table 4.: Table 4. Comparison of different modulation strategies on the DrealSR [49] benchmark.*

> 💡 **Table 4. 批读**: 表格要看主指标、次指标与效率/鲁棒性是否一致支持论文 claim。

$( B a s e _ { w / R G P A \& S F T } ( z _ { L } | \widetilde { \pmb { x } } _ { L } )$ for short). The comparison results ein Table 4 show that using element-wise addition performs suboptimally, even degrading no-reference metrics compared to $B a s e _ { w / R G P A }$ . We attribute this to the domain gap between the LQ input and U-Net features, which introduces instability into the denoising process and can lead to its eventual collapse of the denoising paradigm. Moreover, modulating the latent feature $z _ { L }$ causes a more pronounced drop in no-reference metrics than modulating the intermediate feature $\pmb { f } ^ { m }$ , as it disrupts the Gaussian diagonal distribution of $z _ { L }$ and consequently limits the effective utilization of generative priors during the reverse process. In contrast, our proposed LQFM effectively enhances restoration fidelity while maintaining better generative performance.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Effect of text-matching guidance. To demonstrate the effectiveness of the TMG, we compare our full model with $B a s e _ { w / R G P A \& L Q F M } .$ . As shown in Table 2, our approach with TMG yields further improvements across all non-reference metrics, underscoring its critical role in enhancing the generation capability of the model. Figure 5 visualizes diffusion attentive attribution maps (DAAMs) of our method and [50], where the method [50] suffers from text misalignment even with the regularization of pre-trained models. In contrast, our proposed TMG effectively alleviates the misalignment issue, leading to more accurate and consistent text–image interactions.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Table 5. Comparison of different semantic enhancement strategies on the DrealSR [49] benchmark.

![Table 5.](../images/7ad6dd87e03f217a2f4af149f846b6eac82473492964a803f0f84327205d90a7.jpg)
*Table 5.: Table 5. Comparison of different semantic enhancement strategies on the DrealSR [49] benchmark.*

> 💡 **Table 5. 批读**: 表格要看主指标、次指标与效率/鲁棒性是否一致支持论文 claim。

To further investigate the influence of semantic knowledge distillation in the second training stage, we compare our CODSR with a baseline that replaces the VSD loss in (5) with the GAN loss $\left( O u r s _ { w / o } \ : V S D \ : l o s s \right.$ for short). As shown in Table 5, using GAN Loss fails to fully unleash the semantic generation capability due to the lack of effective semantic guidance in the second-stage training, whereas the VSD Loss enables semantic knowledge distillation from the pretrained model, thereby enhancing the generative ability.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
