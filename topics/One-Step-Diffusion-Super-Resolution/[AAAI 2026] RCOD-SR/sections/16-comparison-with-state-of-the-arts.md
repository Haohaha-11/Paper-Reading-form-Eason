[← 返回 README](../README.md)

# Comparison with State-of-the-Arts

## 📌 预览
Experiments 验证方法是否真的改善质量、速度和可控性；注意 full-reference/no-reference 指标之间的张力。

> 💡 **与 RCOD-SR 主线的关系**: RCOD-SR 在 one-step diffusion Real-ISR 中引入 latent domain grouping、退化感知采样蒸馏和视觉 prompt 注入，让单步模型在推理时可控地调节 realism。

---

Quantitative Comparisons: We can observe in Table S2: $i$ ) By applying RCOD, on each dataset, the “-Fid.” versions $( t = 2 5 0$ ) have better full-reference (FR) metrics such as PSNR, SSIM, LPIPS, and FID, while keeping the noreference (NR) metrics relatively ordinary. In contrast, the “- Real.” versions $t = 7 5 0$ ) achieve obviously higher NR metrics like MANIQA, MUSIQ, and CLIPIQA. Most “-Neu.” versions $( t = 5 0 0 )$ ) fall within the middle range of the previous two versions. This illustrates that we can effectively and simply control the realism level (usually measured by perceptual NR metrics) during the inference stage, and that the realism level increases monotonically with the timestep. $i i \ "$ ) $\mathrm { R C O D } _ { \mathrm { S } }$ -Adap. has relatively balanced metrics between $\mathrm { R C O D } _ { \mathrm { S } }$ -Fid. and ${ \mathrm { R C O D } } _ { \mathrm { S } }$ -Neu.. This indicates that most estimated $M _ { L }$ values are closer to 1 than to 0. This roughly matches the cosine similarity value distributions in Fig. S4 (b). iii) When applying RCOD, $\mathrm { R C O D _ { O } }$ -Fid. and $\operatorname { R C O D } _ { \operatorname { S } }$ - Adap. perform better than their original methods (OSEDiff and S3Diff, respectively) on most metrics, including PSNR, SSIM, LPIPS, MANIQA, and MUSIQ. Even with a preference for fidelity in FR metrics, $\mathrm { R C O D _ { O } }$ -Fid. shows superior performance on some NR metrics, such as MANIQA and MUSIQ, compared to the original OSEDiff method on realworld data. Additionally, ${ \mathrm { R C O D } } _ { \mathrm { S } }$ -Real. usually achieves the best NR metrics (NIQE and CLIPIQA). iv) S3Diff shows better performance on the perceptual quality metric DISTS. This may arise from the negative online prompting (NOP) used in training, which provides a more accurate concept of high quality. However, since the text encoder and text prompt are replaced by VPIM in our $\mathrm { R C O D } _ { \mathrm { S } }$ , the NOP is also removed. Despite this, our ${ \mathrm { R C O D } } _ { \mathrm { S } }$ -Adap. performs better on other NR metrics.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

![Figure S5](../images/72d8efd35cce6ef0c1bfc4449499b8142caa4f5ffa766b45c5845a8f55c870b8.jpg)
*Figure S5: Figure S5: Visual comparison $( \times 4 )$ of $\mathrm { R C O D _ { O } }$ -Real. with other methods on DRealSR data.*

> 💡 **Figure S5 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

Time Efficiency: In Table S3, we compare inference time and trainable parameters. All methods are tested on an A100 GPU with a $5 1 2 \times 5 1 2$ input image. RCOD keeps similar time efficiency and trainable parameters as the original methods while have higher PSNR and MANIQA. $\mathrm { R C O D _ { O } }$ even inferences faster as the text extractor is removed.

> 💡 **批注**: 这里涉及条件信号：prompt 是否准确、是否退化感知，会直接影响生成细节与语义一致性。

Qualitative Comparisons: Fig. S5 compares the visual qualities of different Real-ISR methods. $\mathrm { R C O D _ { O } }$ -Real. demonstrates its ability to recover more detailed and natural textures. Fig. S6 illustrates the changes in visual effects as $t$ increases, i.e, from $\mathrm { R C O D } _ { \mathrm { S } }$ -Fid. $( t ~ = ~ 2 5 0 )$ t o ${ \mathrm { R C O D } } _ { \mathrm { S } }$ -Real. $( t ~ = ~ 7 5 0 )$ . As $t$ increases during the inference stage, more skin texture and wrinkles are recovered. ${ \mathrm { R C O D } } _ { \mathrm { S } }$ -Adap. chooses a proper $t = 5 0 0$ in this case, where the $M _ { L }$ range is [0.5, 0.75) according to Eq. 5. The $M _ { L }$ of the LR image (0.621) falls within this range.

> 💡 **批注**: 这里涉及条件信号：prompt 是否准确、是否退化感知，会直接影响生成细节与语义一致性。

Ablation Study We performed a series of ablation studies of the RCOD framework, the details can be found in SM.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标，避免被单一数字误导。

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
| Control variable | latent-domain group / denoising degree |
| Accepted venue | AAAI 2026 Oral according to arXiv comment |
| Training data change | minimal paradigm modification and original training data claimed |
