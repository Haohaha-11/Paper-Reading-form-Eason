[← 返回 README](../README.md)

# 4. Experimental Results

## 📌 预览
本节验证方法是否成立，重点看主结果、消融、效率和视觉/病例案例。

---

# 4.1. Experimental settings

Training datasets. Following the protocols [40, 50, 51], we collect a training dataset including the images from LS-DIR [25], DIV2K [1], Flicker2K [29], DIV8K [16] and the first 10K images from FFHQ [20]. The corresponding LQ images are synthesized from their HQ counterparts through the complex degradation model in Real-ESRGAN [45].

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Testing datasets. We evaluate our method on four realworld datasets, including RealSR [4], DrealSR [49], RealPhoto60 [56] and RealDeg [6]. For RealSR and DRealSR, the LQ-HQ pairs involve a $\times 4$ super-resolution task from $1 2 8 \times 1 2 8$ to $5 1 2 \times 5 1 2$ . While for RealPhoto60, the LQ images $( 5 1 2 \times 5 1 2 )$ are upscaled by $\times 2$ to $1 0 2 4 \times 1 0 2 4$ . In addition, RealDeg [6] contains 238 images of old photographs, classic film stills, and social media photos under diverse real-world degradation types.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Compared methods. We compare our method with representative approaches, including full-step diffusion-based methods (i.e., DiffBIR [30], SeeSR [51], SUPIR [56], and FaithDiff [6]) and one-step diffusion-based methods (i.e., SinSR [46], OSEDiff [50], TVT [55], PiSA-SR [40], and HYPIR [31]). In addition, we also include GAN-based super-resolution methods such as RealESRGAN [45] and BSRGAN [60] for comprehensive comparison, as detailed in the supplemental material. For a fair comparison, all reported results for competing methods are produced using their publicly available official implementations and pretrained models.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Evaluation metrics. We evaluate all methods using fullreference and no-reference image quality metrics. For fullreference assessment, we employ PSNR and SSIM [47], computed on the Y channel in YCbCr space to evaluate reconstruction fidelity, alongside LPIPS [64] and DISTS [11] to measure perceptual similarity. The no-reference metrics include NIQE [61], MUSIQ [23], MANIQA-pipal [53], and CLIPIQA $^ +$ [42], which estimate perceptual quality without references.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Figure 3.](../images/f0ffe075fcc994178ff94bc23b038ac393a443a2be95b212d951ef7ba12c3044.jpg)
*Figure 3.: Figure 3. Qualitative comparisons of different methods on the RealSR [4] dataset, RealLQ250 [2] dataset and RealLR200 [51] dataset. Compared to competing methods, our approach generates a more realistic image with fine-scale structures and details. Please zoom in for a better view.*

> 💡 **Figure 3. 批读**: 这张图通常承担方法框架、动机或视觉对比作用；重点看它支撑的是机制、效果还是局限。

Implementation details. Our implementation is built upon the SD 2.1-base [37]. Similar to OSEDiff [50], we optimize LoRA [18] adapters within the VAE encoder and the U-Net network while keeping the VAE decoder fixed. The LoRA ranks are configured as 4 for the VAE encoder and 16 for the U-Net, respectively. We use RAM [68] for prompt extraction during training and DAPE [51] at inference. Our model is trained using 4 NVIDIA 4090 GPUs with a batch size of

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

16, using the AdamW optimizer [32] with a learning rate of $5 e ^ { - 5 }$ . In the first-stage training, we employ the GAN loss used in S3Diff [59]. In the second stage, the VSD loss is incorporated with a weighting coefficient of 2. More experimental results are included in the supplemental material.

# 4.2. Comparison with the state of the art

Quantitative evaluations against diffusion-based fullstep methods. We first compare our approach with diffusion-based full-step methods [6, 30, 51, 56] in Table 1. Our method performs the best in terms of the all full-reference metrics (PSNR, SSIM, LPIPS, and DISTS) as well as the no-reference metrics NIQE and MUSIQ while delivering comparable results on other metrics. Notably, our method improves the MUSIQ by at least 1.96 and 5.15 on the datasets of DRealSR and RealDeg, respectively.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Figure 4.](../images/73b8c250fcc74e8c2d7a2cf273ead777e3f9e9562456111284beaae7e707814e.jpg)
*Figure 4.: Figure 4. Comparisons between $B a s e _ { w / \ z _ { L } + \epsilon }$ and $B a s e _ { w / R G P A }$ on the DrealSR [49] benchmark. While $B a s e _ { w / \ z _ { L } + \epsilon }$ introduces unpleasant details in smooth areas, our method achieves more faithful reconstructions.*

> 💡 **Figure 4. 批读**: 这张图通常承担方法框架、动机或视觉对比作用；重点看它支撑的是机制、效果还是局限。

Quantitative evaluations against diffusion-based onestep methods. We then evaluate the proposed approach against diffusion-based one-step methods [31, 40, 46, 50, 55]. The quantitative results in Table 1 show that our method outperforms all competing approaches across all no-reference metrics. The NIQE, MUSIQ, and CLIPIQA+ of our method is at least 0.15, 0.39, and 0.0133 higher than the competing methods across all datasets.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Qualitative results. Figure 3 shows the visual comparisons. For the example from RealSR [4], the full-step methods [6, 30, 56] cannot effectively recover a clear figure viewed from the behind that is consistent with the LQ input. The result generated by [46] exhibits obvious artificial textures. Compared to these competing methods, our approach yields a clearer result with a more realistic background and better-defined foreground. Figure 3 also shows an example from RealLQ250 [2], our method further demonstrates a superior capability for storing high-quality images, e.g., clear and natural flowers and antlers. For the example from RealLR200 [51], the methods [45, 50, 56] struggle to recover characters from the LQ input, while [30, 40, 55] introduce background noise or structurally distorted characters, leading to compromised readability. In contrast, our method faithfully restores text with sharper edges, continuous strokes, and minimal background artifacts, yielding a highly legible result.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
