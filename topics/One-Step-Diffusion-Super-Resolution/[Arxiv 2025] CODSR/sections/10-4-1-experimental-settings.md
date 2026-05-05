[← 返回 README](../README.md)

# 4.1. Experimental settings

## 📌 预览
Experiments 验证方法是否真的改善质量、速度和可控性；注意 full-reference/no-reference 指标之间的张力。

> 💡 **与 CODSR 主线的关系**: CODSR 从 LQ 信息损失、区域差异化先验激活和文本-区域错配三方面补强 controllable one-step diffusion SR，在单步推理下兼顾感知质量与局部保真。

---

Training datasets. Following the protocols [40, 50, 51], we collect a training dataset including the images from LS-DIR [25], DIV2K [1], Flicker2K [29], DIV8K [16] and the first 10K images from FFHQ [20]. The corresponding LQ images are synthesized from their HQ counterparts through the complex degradation model in Real-ESRGAN [45].

> 💡 **批注**: 这是控制机制：作者试图把退化强度、区域差异或 timestep 映射成可操作的生成强度。

Testing datasets. We evaluate our method on four realworld datasets, including RealSR [4], DrealSR [49], RealPhoto60 [56] and RealDeg [6]. For RealSR and DRealSR, the LQ-HQ pairs involve a $\times 4$ super-resolution task from $1 2 8 \times 1 2 8$ to $5 1 2 \times 5 1 2$ . While for RealPhoto60, the LQ images $( 5 1 2 \times 5 1 2 )$ are upscaled by $\times 2$ to $1 0 2 4 \times 1 0 2 4$ . In addition, RealDeg [6] contains 238 images of old photographs, classic film stills, and social media photos under diverse real-world degradation types.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

Compared methods. We compare our method with representative approaches, including full-step diffusion-based methods (i.e., DiffBIR [30], SeeSR [51], SUPIR [56], and FaithDiff [6]) and one-step diffusion-based methods (i.e., SinSR [46], OSEDiff [50], TVT [55], PiSA-SR [40], and HYPIR [31]). In addition, we also include GAN-based super-resolution methods such as RealESRGAN [45] and BSRGAN [60] for comprehensive comparison, as detailed in the supplemental material. For a fair comparison, all reported results for competing methods are produced using their publicly available official implementations and pretrained models.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

Evaluation metrics. We evaluate all methods using fullreference and no-reference image quality metrics. For fullreference assessment, we employ PSNR and SSIM [47], computed on the Y channel in YCbCr space to evaluate reconstruction fidelity, alongside LPIPS [64] and DISTS [11] to measure perceptual similarity. The no-reference metrics include NIQE [61], MUSIQ [23], MANIQA-pipal [53], and CLIPIQA $^ +$ [42], which estimate perceptual quality without references.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

![Figure 3.](../images/f0ffe075fcc994178ff94bc23b038ac393a443a2be95b212d951ef7ba12c3044.jpg)
*Figure 3.: Figure 3. Qualitative comparisons of different methods on the RealSR [4] dataset, RealLQ250 [2] dataset and RealLR200 [51] dataset. Compared to competing methods, our approach generates a more realistic image with fine-scale structures and details. Please zoom in for a better view.*

> 💡 **Figure 3. 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

Implementation details. Our implementation is built upon the SD 2.1-base [37]. Similar to OSEDiff [50], we optimize LoRA [18] adapters within the VAE encoder and the U-Net network while keeping the VAE decoder fixed. The LoRA ranks are configured as 4 for the VAE encoder and 16 for the U-Net, respectively. We use RAM [68] for prompt extraction during training and DAPE [51] at inference. Our model is trained using 4 NVIDIA 4090 GPUs with a batch size of

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

16, using the AdamW optimizer [32] with a learning rate of $5 e ^ { - 5 }$ . In the first-stage training, we employ the GAN loss used in S3Diff [59]. In the second stage, the VSD loss is incorporated with a weighting coefficient of 2. More experimental results are included in the supplemental material.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

---

## 🔖 Section 总结

### 核心洞察

1. 同时看 PSNR/SSIM 与 LPIPS/FID/NIQE/MUSIQ 等指标。
2. 检查 one-step 是否真的在 latency/参数量上有优势。
3. 优先关注 ablation 是否支持论文的主模块设计。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Core controls | region-adaptive prior activation + text-matching guidance |
| Project | https://github.com/Chanson94/CODSR |
| Benchmarks | RealSR/DrealSR and other real-world SR test sets in paper tables |
