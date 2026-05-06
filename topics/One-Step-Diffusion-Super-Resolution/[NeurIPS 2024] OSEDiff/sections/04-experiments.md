[← 返回 README](../README.md)

# 4 Experiments

## 📌 预览
本节验证 OSEDiff 的 claim，重点看主结果、消融、效率、视觉案例和 trade-off 是否一致。
---

> 💡 **Q&A 批注记录**:
> - Q: OSEDiff 为什么成为后续 baseline？
> - A: 它给出了清晰的 one-step SD prior distillation 范式，后续工作主要围绕可控性、局部保真和压缩继续改。


# 4.1 Experimental Settings
> 💡 **小节预览**: 实验要同时看三件事：质量指标是否赢、速度是否保持单步优势、可控/消融是否支撑核心 claim。


Training and Testing Datasets. Prior works [42, 57, 31, 52] employed different training datasets, making unified training standards for Real-ISR difficult to establish. For simplicity, we adopt SeeSR’s setup [52] and train OSEDiff using the LSDIR [26] dataset and the first 10K face images from FFHQ [19]. The degradation pipeline of Real-ESRGAN [45] is used to synthesize LQ-HQ training pairs. We evaluate OSEDiff and compare it with competing methods using the test set provided by StableSR [42], including both synthetic and real-world data. The synthetic data includes 3000 images of size $5 1 2 \times 5 1 2$ , whose GT are randomly cropped from DIV2K-Val [2] and degraded using the Real-ESRGAN pipeline [45]. The real-world data include LQ-HQ pairs from RealSR [3] and DRealSR [51], with sizes of $1 2 8 \times 1 2 8$ and $5 1 2 \times 5 1 2$ , respectively.
> 💡 **数据设置批读**: 训练跟 SeeSR 对齐，用 LSDIR + FFHQ face 前 10K，并用 Real-ESRGAN 退化合成 LQ-HQ。好处是比较方便，风险是模型仍会继承合成退化 pipeline 的偏差。


Compared Methods. We compare OSEDiff with state-of-the-art DM-based Real-ISR methods, including StableSR [42], ResShift [60], PASD [57], DiffBIR [31], SeeSR [52] and SinSR [48]. Among them, StableSR, PASD, DiffBIR, and SeeSR are all based on the pre-trained SD model. ResShift trains a DM from scratch in the pixel domain, while SinSR is a one-step model distilled from ResShift. Note that we do not compare with the recent method SUPIR [59] because it tends to generate rich yet excessive details, which are however unfaithful to the input image.
> 💡 **baseline 选择**: 对照组覆盖多步 SD 方法、pixel-domain diffusion、以及 one-step SinSR。未比较 SUPIR 的理由是作者认为其细节过量且不忠实输入，这也提示 OSEDiff 的评价重点不是单纯“越锐越好”。


Table 1: Quantitative comparison with state-of-the-art methods on both synthetic and real-world benchmarks. ‘s’ denotes the number of diffusion reverse steps in the method. The best and second best results of each metric are highlighted in red and blue, respectively.
> 💡 **Table 1 导读**: 这张主表要按三类指标读：PSNR/SSIM 看保真，LPIPS/DISTS/FID 看分布和感知相似，NIQE/MUSIQ/MANIQA/CLIPIQA 看无参考质量。OSEDiff 的强项主要落在感知和语义质量，不是所有 fidelity 指标第一。


![Table 1](../images/a236a580e95d4b33c1e2db66f2985b919d218318e50ffc850d0e4c768ce9bf42.jpg)
*Table 1: Table 1: Quantitative comparison with state-of-the-art methods on both synthetic and real-world benchmarks. ‘s’ denotes the number of diffusion reverse steps in the method. The best and second best results of each metric are highlighted in red and blue, respectively.*
> 💡 **Table 1 批读**: 表里 OSEDiff 在 DRealSR/RealSR 上的 LPIPS、FID、CLIPIQA 等表现支撑“SD prior + VSD 带来真实感”；ResShift/SinSR 的 PSNR 更好则说明 restoration 专用扩散仍更保守、更偏保真。


Table 2: Complexity comparison among different methods. All methods are tested with an input image of size $5 1 2 \times 5 1 2$ , and the inference time is measured on an A100 GPU.
> 💡 **Table 2 导读**: 复杂度表是 OSEDiff 的硬证据：1 step、0.11s/A100/512x512、8.5M trainable parameters。这里要把“快”归因于推理只跑 generator，而不是 VSD regularizer。


![Table 2](../images/d194b2d67bcbeb864681ee086090e39d129a94a01ec496f7b7fa2ffaff4f4eaa.jpg)
*Table 2: Table 2: Complexity comparison among different methods. All methods are tested with an input image of size $5 1 2 \times 5 1 2$ , and the inference time is measured on an A100 GPU.*
> 💡 **Table 2 批读**: 和 StableSR/SeeSR 相比，OSEDiff 的优势来自去掉几十到数百步采样；和 SinSR 相比，优势是同为一步但复用 SD prior 和 prompt/VSD，质量更高。


For those GAN-based Real-ISR methods, including BSRGAN [61], Real-ESRGAN [45], LDL [27], and FeMaSR [4], we put their results into the Appendix.
> 💡 **附录交叉引用**: GAN-based baseline 被放到 Appendix，说明正文主战场是 diffusion-based Real-ISR。读完整证据时仍要看 Appendix，因为它能暴露 OSEDiff 相对 GAN 的 fidelity/perception trade-off。


Evaluation Metrics. To provide a comprehensive and holistic assessment on the performance of different methods, we employ a range of full-reference and no-reference metrics. PSNR and SSIM [50] (calculated on the Y channel in YCbCr space) are reference-based fidelity measures, while LPIPS [64], DISTS [12] are reference-based perceptual quality measures. FID [15] evaluates the distance of distributions between GT and restored images. NIQE [62], MANIQA-pipal [55], MUSIQ [22], and CLIPIQA [41] are no-reference image quality measures. We also conduct a user study, which is presented in the Appendix.
> 💡 **指标批读**: 指标覆盖较全，但 no-reference 指标会偏好丰富纹理，不一定保证内容忠实。作者补 user study 是必要的，因为 OSEDiff 的核心 claim 同时包含 perceptual quality 和 content consistency。


Implementation Details. We train OSEDiff with the AdamW optimizer [33] at a learning rate of 5e-5. The entire training process took approximately 1 day on 4 NVIDIA A100 GPUs with a batch size of 16. The rank of LoRA in the VAE Encoder, diffusion network, and finetuned regularizer is set to 4. For the text prompt extractor, although advanced multimodal language models [32] can provide detailed text descriptions, they come at a high inference cost. We adopt the degradation-aware prompt extraction (DAPE) module in SeeSR [52] to extract text prompts. The SD 2.1-base is used as the pre-trained T2I model. The weighting scalars $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are set to 2 and 1, respectively.
> 💡 **实现细节**: 关键超参：AdamW 5e-5、4*A100 约 1 天、batch 16、LoRA rank=4、SD 2.1-base、$\lambda_1=2,\lambda_2=1$。DAPE 替代 LLaVA 是质量/成本折中。


# 4.2 Comparison with State-of-the-Arts
> 💡 **小节预览**: 实验要同时看三件事：质量指标是否赢、速度是否保持单步优势、可控/消融是否支撑核心 claim。


Quantitative Comparisons. The quantitative comparisons among the competing methods on the three datasets are presented in Table 1. We can have the following observations. (1) First, OSEDiff exhibits clear advantages over competing methods in full-reference perceptual quality metrics LPIPS and DISTS, distribution alignment metric FID, and semantic quality metric CLIPIQA, especially on the two real-world datasets DrealSR and RealSR. (2) Second, SeeSR and PASD show better no-reference metrics like NIQE, MUSIQ and MANIQA. This is because these multi-step methods can produce rich image details in the diffusion process, which are preferred by no-reference metrics. (3) Third, ResShift and its distilled version SinSR show better full-reference fidelity metrics such as PSNR. This is mainly because they train a DM from scratch specifically for the restoration purpose, instead of exploring the pre-trained T2I model such as SD. However, ResShift and SinSR show poor perceptual quality metrics than other methods.
> 💡 **定量结果批读**: 作者自己承认了 trade-off：OSEDiff 在 LPIPS/DISTS/FID/CLIPIQA 更强，SeeSR/PASD 在部分 no-reference 指标更强，ResShift/SinSR 在 PSNR 更强。它不是全指标碾压，而是质量/速度综合更优。


![Figure 3](../images/5826ad2dd8b383b675665cc1f5b07a970462d3ccda5b9bcfd0ec2e834aaec2d0.jpg)
*Figure 3: Figure 3: Qualitative comparisons of different Real-ISR methods. Please zoom in for a better view.*
> 💡 **Figure 3 批读**: 视觉案例主要验证两个点：脸部细节和叶脉纹理。读图时要警惕“更丰富”是否等于“更真实”，尤其叶脉、文字、规则结构容易被 SD 先验合理化但不忠实。


Qualitative Comparisons. Fig. 3 presents visual comparisons of different Real-ISR methods. As illustrated in the first example, ResShift and SinSR severely blur the facial details due to the lack of pre-trained image priors. StableSR, DiffBIR and SeeSR reconstruct more facial details by exploring the image prior in pre-trained SD model. PASD generates excessive yet unnatural details. Though OSEDiff performs only one step forward propagation, it reproduces realistic and superior facial details to other methods. Similar conclusion can be drawn from the second example. StableSR and DiffBIR are limited in generating rich textures due to the lack of text prompts. PASD suffers from incorrect semantic generation because its prompt extraction module is not robust to degradation. While SeeSR utilizes degradation-aware semantic cues to stimulate image generation priors, the generated leaf veins are unnatural, which may be influenced by its random noise sampling. In contrast, OSEDiff can generate detailed and natural leaf veins. More visualization comparisons and the results of subjective user study can be found in the Appendix.
> 💡 **视觉结论**: 作者把 OSEDiff 的优势归因于三点：SD prior 带细节、DAPE prompt 激活语义、LQ 起点减少随机噪声带来的错误生成。这里也暴露 prompt extractor 鲁棒性会直接影响输出语义。


Complexity Comparisons. We further compare the complexity of competing DM-based Real-ISR models in Table 2, including the number of inference steps, inference time, and trainable parameters. All methods are tested on an A100 GPU with an input image of size $5 1 2 \times 5 1 2$ . OSEDiff has the fewest trainable parameters, and the trained LoRA layers can be merged into the original SD to further reduce the computational cost. By using only one forward pass, OSEDiff has significant advantage in inference time over multi-step methods. Specifically, OSEDiff demonstrates a substantial speed advantage, being approximately 105 times faster than StableSR, 39 times faster than SeeSR, and 6 times faster than ResShift. When compared to the single-step method SinSR, OSEDiff not only achieves faster inference but also delivers significantly higher output quality. In terms of complexity, OSEDiff requires the lowest MACs at just 2265G, as it operates with only a single diffusion step. In contrast, methods like StableSR, which require 200 steps, incur substantially higher MACs (e.g., 79940G). Regarding trainable parameters, OSEDiff is highly parameter-efficient, requiring only 8.5M parameters (LoRA layers), compared to models such as SeeSR, which necessitates 749.9M parameters. This highlights the efficiency of OSEDiff during the training process.
> 💡 **复杂度结论**: 0.11s 和 8.5M trainable parameters 是 OSEDiff 最强卖点。注意这是 A100/512x512 条件，不能直接外推到端侧；LoRA 可合并只降低训练/部署参数管理复杂度，不等于 UNet 本体很小。


Table 3: Comparison of different losses on the RealSR benchmark.

![Table 3](../images/327baa64007b5c4a08af6bccbd3bcf2ca4be1ca5b6d75dbbb99468227479c2d7.jpg)
*Table 3: Table 3: Comparison of different losses on the RealSR benchmark.*
> 💡 **Table 3 导读**: 这是 VSD 的核心消融：w/o VSD、GAN loss、image-domain VSD、latent-domain VSD。读它要看 latent VSD 是否同时改善感知指标和 CLIPIQA，而不是只比 GAN loss 多一点。


Table 4: Comparison of different text prompt extractors on the DrealSR benchmark.
> 💡 **Table 4 导读**: prompt extractor 消融揭示一个典型 trade-off：无 prompt 的 full-reference 指标更好，有 prompt 的无参考/语义质量更好。DAPE 被选中不是因为最强，而是因为接近 LLaVA 的生成收益且只需 0.02s。


![Table 4](../images/9e125af3fce24144b6db9331675590410c770298e08ea197bc2daf7e859f3656.jpg)
*Table 4: Table 4: Comparison of different text prompt extractors on the DrealSR benchmark.*
> 💡 **Table 5 批读**: VAE encoder LoRA rank=2 不收敛，rank=8 有过拟合退化估计、丢细节风险，rank=4 是稳定性和表达力折中。这支撑“少量参数足够，但不是越小越好”。


Table 5: Comparison of LoRA in VAE encoder with different ranks.

![Table 5](../images/d26c6c349c36d69fcd6e498c51096ef5eafe4da969229305419c8a4db7012981.jpg)
*Table 5: Table 5: Comparison of LoRA in VAE encoder with different ranks.*
> 💡 **Table 6 批读**: UNet LoRA rank 也选 4，说明生成主干的适配容量需要控制。rank 增大未带来稳定提升，可能破坏预训练 SD prior 或导致数据集过拟合。


Table 6: Comparison of LoRA in UNet with different ranks.

![Table 6](../images/417d342a2d82f46964ea5a9882f720620dd19acf62b7f9912ef6a7ee3b3bed5b.jpg)
*Table 6: Table 6: Comparison of LoRA in UNet with different ranks.*
> 💡 **LoRA rank 小结**: Tables 5-6 合起来说明 rank=4 是 encoder 和 UNet 的共同折中点；rank 太低容量不足，rank 太高不一定提升，还可能削弱原 SD prior 的泛化。


Table 7: Ablation studies on finetuning the VAE encoder and decoder on the RealSR benchmark.
> 💡 **Table 7 导读**: 这张表回答“为什么训 encoder、不训 decoder”。训 encoder 能去退化、改善感知；冻结 decoder 保持 latent space 和 regularizer 对齐，因此 CLIPIQA/MUSIQ 更好。


![Table 7](../images/fd92547225f54515bd73d429ab16d9a859198f9da48eac552297979fcebd0261.jpg)
*Table 7: Table 7: Ablation studies on finetuning the VAE encoder and decoder on the RealSR benchmark.*
> 💡 **Table 7 批读**: 同时训 decoder 虽可能提高 PSNR，却会破坏和 SD VAE latent 空间的一致性，使 latent VSD 难以发挥。OSEDiff 的架构选择是为 regularization 服务的。


# 4.3 Ablation Study

Effectiveness of VSD Loss. To validate the effectiveness of our VSD loss in latent space, we perform ablation studies by removing the VSD loss, replacing it with the GAN loss used in [35], and applying VSD loss in the image domain. The results on the RealSR test set are shown in Table 3. We can see that without using the VSD loss, the perceptual quality metrics are significantly degraded because it is hard to ensure good visual quality using only MSE loss and even LPIPS loss [64]. Using GAN loss and VSD loss in the image domain can improve the performance, but the results are not as good as applying VSD loss in the latent domain. Our proposed OSEDiff can effectively align the distribution of Real-ISR outputs by performing VSD regularization in the latent domain.
> 💡 **VSD 消融结论**: latent-domain VSD 比 image-domain VSD 更有效，说明“在 SD 工作的 latent 空间做分布匹配”是必要设计，而不是工程加速小技巧。


Comparison on Text Prompt Extractors. We then conduct experiments to evaluate the effect of different text prompt extractors on the Real-ISR results. We test three options. The first option does not employ text prompts. The second option uses the DAPE module in SeeSR [52] to extract degradation-aware tag-style prompts, as we used in our main experiments. The third option uses LLaVA-v1.5 [32] to extract long text descriptions after removing the degradation of input LQ images, as used in SUPIR [59]. We retrain the models based on different prompt extraction methods. The ablation results are shown in Table 4.
> 💡 **prompt 消融设置**: 三组 prompt 对照能区分“没有语义激活”“轻量 tag-style prompt”“昂贵长 caption”。这直接关系到 one-step 模型能从 SD 先验里拿到多少可用细节。


One can see that without using text prompts as inputs, those full-reference metrics such PSNR, SSIM, LPIPS, DISTS and even FID can improve, while those no-reference metrics such as MUSIQ, MANIQA and CLIPIQA become worse. By using DAPE and LLaVA to extract text prompts, the generation capability of the pre-trained T2I SD model can be triggered, resulting in richer synthesized details, which however can reduce the full-reference indices. A visual example is shown in Figure 4. We see that while LLaVA extracts significantly longer text prompts than DAPE, they produce a similar amount of visual details. However, it is worth mentioning that the MLLM model LLaVA is very costly, increasing the inference time of DAPE by 170 times. Considering the cost-effectiveness, we ultimately choose DAPE as the text prompt extractor in OSEDiff.
> 💡 **prompt trade-off**: prompt 会提升 MUSIQ/MANIQA/CLIPIQA，但牺牲 PSNR/SSIM/LPIPS/DISTS/FID 等 full-reference 指标。原因是 prompt 激活了更多生成细节，也更可能偏离 GT 像素。


![Figure 4](../images/c2aaef75504cc6098835517df040d25731ac78ecd293200b75dcd5aad1d235e0.jpg)
*Figure 4: Figure 4: The impact of different prompt extraction methods. Please zoom in for a better view.*
> 💡 **Figure 4 批读**: 这张图要看 DAPE 和 LLaVA 是否真的产生明显不同细节。若视觉差距不大，DAPE 的 0.02s 成本优势就足以支持作者选择。


Setting of LoRA Rank. When finetuning the VAE encoder and the UNet, we need to set the rank of LoRA layers. Here we evaluate the effect of different LoRA ranks on the Real-ISR performance by using the RealSR benchmark [3]. The results are shown in Tables 5 and 6, respectively. As shown in Table 5, if a too small LoRA rank, such as 2, is set for the VAE encoder, the training will be unstable and cannot converge. On the other hand, if a higher LoRA rank, such as 8, is used for the VAE encoder, it may overfit in estimating image degradation, losing some image details in the output, as evidenced by the PSNR, DISTS, MUSIQ and NIQE indices. We find that setting the rank to 4 can achieve a balanced result for the VAE encoder. Similar conclusions can be made for the setting of LoRA rank on UNet. As can be seen from Table 6, a rank of 4 strikes a good balance. Therefore, we set the rank as 4 for both the VAE encoder and UNet LoRA layers.
> 💡 **LoRA rank 结论**: rank=4 是经验折中，不是理论最优。后续如果换 SD 底座、数据集或退化强度，这个 rank 仍需要重新扫。


The Finetuning on the VAE Encoder and Decoder. We conducted ablation studies to examine the impact of finetuning the VAE encoder and decoder, as shown Table 7. In the first row, where neither the VAE encoder nor decoder is finetuned, the results show poor perception performance. Comparing with OSEDiff, where only the VAE encoder is finetuned, we observe significant improvements in perceptual quality (e.g., MUSIQ improves from 58.99 to 69.09). This demonstrates that finetuning the VAE encoder is important for removing degradation and enhancing overall performance. When comparing the third row, where both the VAE encoder and decoder are finetuned, with OSEDiff, where only the encoder is trained and the decoder is fixed, we note that OSEDiff also achieves better perceptual quality (CLIPIQ improves from 0.5778 to 0.6693). This indicates that fixing the VAE decoder is important to ensure that the UNet output remains in the original VAE latent space, which helps minimizing the VSD loss more effectively. Thus, finetuning the VAE encoder is important to remove degradation, while fixing the VAE decoder helps maintaining stability in the latent space, leading to better perceptual quality.
> 💡 **VAE 消融结论**: 只训 encoder、冻 decoder 的设计同时服务两件事：让输入 latent 适配退化恢复，保持输出 latent 可被 frozen SD/VSD regularizer 正确解释。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 读表顺序 | 主结果 → 复杂度 → VSD/prompt/LoRA/VAE 消融 → 视觉案例 |
| 核心检查 | 感知质量、保真、速度、训练/推理分支是否分离 |
| 风险 | no-reference 指标提升不等于结构真实 |

### 核心洞察
1. OSEDiff 的证据不是“全指标第一”，而是感知质量接近/超过多步 SD，同时速度显著领先。
2. 消融支撑了四个关键设计：latent VSD、DAPE prompt、LoRA rank=4、train encoder/freeze decoder。
3. 视觉案例要特别看文字、边缘、平坦区域和重复纹理，因为这些最容易暴露 SD prior 的幻觉风险。

### 可追问点
- 为什么从 LQ latent 而不是纯噪声开始？
- VSD 在这里起什么作用？
- 为什么有 prompt 会牺牲 full-reference 指标？
- Table 7 为什么支持冻结 VAE decoder？
