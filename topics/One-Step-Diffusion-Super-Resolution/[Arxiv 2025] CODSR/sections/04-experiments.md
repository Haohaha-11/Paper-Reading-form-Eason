[← 返回 README](../README.md)

# 4. Experimental Results

## 📌 预览
本节验证 CODSR 是否真的“快而不糊、真而不飘”。读法是先看数据集和指标是否覆盖真实退化，再看 full-step/one-step/GAN baseline 的比较，最后回到视觉案例检查文字、平坦区域和细纹理。
---

> 💡 **Q&A 批注记录**:
> - Q: TMG 是否只是锦上添花？
> - A: TMG 不是基础重建模块，它主要负责让语义纹理生成落在正确对象上；如果只看 PSNR/SSIM 可能不显著，但在 DAAM、鸟羽毛、文字/物体语义区域上更能体现作用。


# 4.1. Experimental settings
> 💡 **小节预览**: 设置部分要确认三件事：训练退化是否接近 real-world，测试集是否覆盖真实退化，指标是否同时包含 full-reference fidelity 与 no-reference realism。


Training datasets. Following the protocols [40, 50, 51], we collect a training dataset including the images from LS-DIR [25], DIV2K [1], Flicker2K [29], DIV8K [16] and the first 10K images from FFHQ [20]. The corresponding LQ images are synthesized from their HQ counterparts through the complex degradation model in Real-ESRGAN [45].
> 💡 **训练数据批读**: 训练集规模和来源比较常规，关键是 LQ 由 Real-ESRGAN 复杂退化合成。这保证可训练性，但也意味着模型对真实未知退化的泛化仍需要 RealSR/DRealSR/RealDeg 等测试集来验证。


Testing datasets. We evaluate our method on four realworld datasets, including RealSR [4], DrealSR [49], RealPhoto60 [56] and RealDeg [6]. For RealSR and DRealSR, the LQ-HQ pairs involve a $\times 4$ super-resolution task from $1 2 8 \times 1 2 8$ to $5 1 2 \times 5 1 2$ . While for RealPhoto60, the LQ images $( 5 1 2 \times 5 1 2 )$ are upscaled by $\times 2$ to $1 0 2 4 \times 1 0 2 4$ . In addition, RealDeg [6] contains 238 images of old photographs, classic film stills, and social media photos under diverse real-world degradation types.
> 💡 **测试集批读**: 四个测试集覆盖 paired real SR、真实照片和多类型老照片/社媒退化。RealSR/DRealSR 能看 full-reference fidelity，RealDeg/RealPhoto60 更接近无参考真实观感。


Compared methods. We compare our method with representative approaches, including full-step diffusion-based methods (i.e., DiffBIR [30], SeeSR [51], SUPIR [56], and FaithDiff [6]) and one-step diffusion-based methods (i.e., SinSR [46], OSEDiff [50], TVT [55], PiSA-SR [40], and HYPIR [31]). In addition, we also include GAN-based super-resolution methods such as RealESRGAN [45] and BSRGAN [60] for comprehensive comparison, as detailed in the supplemental material. For a fair comparison, all reported results for competing methods are produced using their publicly available official implementations and pretrained models.
> 💡 **baseline 批读**: baseline 选择覆盖了三类竞争者：full-step diffusion 代表质量强但慢，one-step diffusion 是直接竞品，GAN SR 是传统高效生成基线。CODSR 要证明的不是单点 SOTA，而是在这三类之间占到更好的速度-质量位置。


Evaluation metrics. We evaluate all methods using fullreference and no-reference image quality metrics. For fullreference assessment, we employ PSNR and SSIM [47], computed on the Y channel in YCbCr space to evaluate reconstruction fidelity, alongside LPIPS [64] and DISTS [11] to measure perceptual similarity. The no-reference metrics include NIQE [61], MUSIQ [23], MANIQA-pipal [53], and CLIPIQA $^ +$ [42], which estimate perceptual quality without references.
> 💡 **指标批读**: PSNR/SSIM 看像素/结构保真，LPIPS/DISTS 看有参考感知差异，NIQE/MUSIQ/MANIQA/CLIPIQA+ 看无参考自然度。CODSR 的 claim 必须在这些指标之间保持平衡，不能只靠无参考指标变好。


![Figure 3.](../images/f0ffe075fcc994178ff94bc23b038ac393a443a2be95b212d951ef7ba12c3044.jpg)
*Figure 3.: Figure 3. Qualitative comparisons of different methods on the RealSR [4] dataset, RealLQ250 [2] dataset and RealLR200 [51] dataset. Compared to competing methods, our approach generates a more realistic image with fine-scale structures and details. Please zoom in for a better view.*
> 💡 **Figure 批读**: Figure 3 要专门看三类区域：人物/物体轮廓是否跟 LQ 一致，花/鹿角等细纹理是否自然，文字笔画是否连续可读。CODSR 的优势如果成立，应该是细节更自然且结构不明显漂移。


Implementation details. Our implementation is built upon the SD 2.1-base [37]. Similar to OSEDiff [50], we optimize LoRA [18] adapters within the VAE encoder and the U-Net network while keeping the VAE decoder fixed. The LoRA ranks are configured as 4 for the VAE encoder and 16 for the U-Net, respectively. We use RAM [68] for prompt extraction during training and DAPE [51] at inference. Our model is trained using 4 NVIDIA 4090 GPUs with a batch size of
> 💡 **实现批读**: CODSR 基于 SD 2.1-base，只训练 VAE encoder 和 U-Net 的 LoRA，decoder 固定。这个设计保留 pretrained decoder 的图像先验，也意味着新增模块主要是在 latent/attention 层调控，而不是重训整套 diffusion。


16, using the AdamW optimizer [32] with a learning rate of $5 e ^ { - 5 }$ . In the first-stage training, we employ the GAN loss used in S3Diff [59]. In the second stage, the VSD loss is incorporated with a weighting coefficient of 2. More experimental results are included in the supplemental material.
> 💡 **训练细节批读**: 4 张 4090、batch 16、lr 5e-5 给出复现成本量级。第二阶段 VSD 权重为 2，说明语义蒸馏强度不弱；复现时如果出现过度生成，首先要检查 VSD 权重、t_s 和 TMG mask。


# 4.2. Comparison with the state of the art
> 💡 **小节预览**: SOTA 比较要分两步读：先看 full-step diffusion，确认 CODSR 是否在低延迟下接近或超过慢模型；再看 one-step diffusion，确认它不是只靠多模块堆叠换来边际提升。


Quantitative evaluations against diffusion-based fullstep methods. We first compare our approach with diffusion-based full-step methods [6, 30, 51, 56] in Table 1. Our method performs the best in terms of the all full-reference metrics (PSNR, SSIM, LPIPS, and DISTS) as well as the no-reference metrics NIQE and MUSIQ while delivering comparable results on other metrics. Notably, our method improves the MUSIQ by at least 1.96 and 5.15 on the datasets of DRealSR and RealDeg, respectively.
> 💡 **主结果批读**: 这段说 CODSR 在 full-reference 指标和部分 no-reference 指标上都强，支撑“fidelity 与 perceptual quality 同时好”的主张。要注意 CLIPIQA+/MANIQA 若只是 comparable，就不能夸成所有感知指标全胜。


![Figure 4.](../images/73b8c250fcc74e8c2d7a2cf273ead777e3f9e9562456111284beaae7e707814e.jpg)
*Figure 4.: Figure 4. Comparisons between $B a s e _ { w / \ z _ { L } + \epsilon }$ and $B a s e _ { w / R G P A }$ on the DrealSR [49] benchmark. While $B a s e _ { w / \ z _ { L } + \epsilon }$ introduces unpleasant details in smooth areas, our method achieves more faithful reconstructions.*
> 💡 **Figure 批读**: Figure 4 是 RGPA 的关键视觉证据：全图标准噪声能激活生成先验，但会在 smooth area 产生难看的伪细节；RGPA 的价值就在于同样释放 prior，却把平坦区保护住。


Quantitative evaluations against diffusion-based onestep methods. We then evaluate the proposed approach against diffusion-based one-step methods [31, 40, 46, 50, 55]. The quantitative results in Table 1 show that our method outperforms all competing approaches across all no-reference metrics. The NIQE, MUSIQ, and CLIPIQA+ of our method is at least 0.15, 0.39, and 0.0133 higher than the competing methods across all datasets.
> 💡 **one-step 对比**: 对 one-step baseline 全部 no-reference 指标领先，说明 CODSR 的生成观感更强；但这里也要回看 full-reference 指标，防止把“更像自然图”误读成“更忠实于输入”。


Qualitative results. Figure 3 shows the visual comparisons. For the example from RealSR [4], the full-step methods [6, 30, 56] cannot effectively recover a clear figure viewed from the behind that is consistent with the LQ input. The result generated by [46] exhibits obvious artificial textures. Compared to these competing methods, our approach yields a clearer result with a more realistic background and better-defined foreground. Figure 3 also shows an example from RealLQ250 [2], our method further demonstrates a superior capability for storing high-quality images, e.g., clear and natural flowers and antlers. For the example from RealLR200 [51], the methods [45, 50, 56] struggle to recover characters from the LQ input, while [30, 40, 55] introduce background noise or structurally distorted characters, leading to compromised readability. In contrast, our method faithfully restores text with sharper edges, continuous strokes, and minimal background artifacts, yielding a highly legible result.
> 💡 **视觉结果批读**: 文字恢复案例尤其重要，因为它同时考验 LQFM 的结构保真和 TMG/生成先验的克制。能把笔画补连续但不乱造字符，才是真正有用的 SR，而不是单纯锐化。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 读表顺序 | 数据集/指标 → full-step 对比 → one-step 对比 → 视觉案例 |
| 核心检查 | 低延迟下 fidelity 与 perceptual quality 是否同时合理 |
| 风险 | no-reference 指标提升不等于 LQ 条件忠实 |

### 核心洞察
1. CODSR 的主结果强在“低延迟 + 多指标平衡”，不是单纯刷新一个感知指标。
2. 视觉案例要专门找 failure-prone 区域：文字、平坦墙面、重复纹理、细小边缘。
3. 实验设置仍依赖合成退化训练，真实退化泛化要看跨数据集结果而不是只看训练协议。

### 可追问点
- RGPA 和普通加噪有什么区别？
- 为什么还要 LQFM？
- 哪些指标最能暴露 CODSR 的 hallucination 风险？
