[← 返回 README](../README.md)

# 4 EXPERIMENTS

## 📌 预览
本节验证四个 claim：first-stage teacher 是否强，one-step student 是否接近 teacher，$t$ 是否真的调 fidelity-realism，distillation 是否比 BOOT/PINN/SinSR 等更高效。
---

> 💡 **Q&A 批注记录**:
> - Q: 主要风险是什么？
> - A: teacher 能力和退化分布是上限；如果 teacher 对真实退化、开放域纹理或高倍率场景覆盖不足，student 的轨迹对齐只会稳定复刻这个不足。


In this section, we provide experimental details and empirical evaluation of OFTSR and compare it with prior works.
> 💡 **实验总览**: 读实验先分清 teacher 来源：自训 Guided Diffusion/ResShift 架构用于证明框架，off-the-shelf DiT4SR 用于证明能蒸馏强 SD prior。


![Table extracted](../images/cbdfc8779ab1b331d89ffbfb34cc5277ce189448573d0f9ed39bbb881f865656.jpg)
*Table extracted: Table extracted by MinerU. DIV2K Method NFEs (↓) PSNR (↑) LPIPS (↓) FID (↓) Training- free DPS (Chung et al., 2022) 1000 23.05 0.447 109.35 DDRM (Kawar et al., 2022) 20 27.87 0.285 23.38 DDNM (Wang et al., 2*
> 💡 **表格批读**: DIV2K 表要横向读三列：NFEs 代表速度，PSNR 代表 fidelity，LPIPS/FID 代表 realism。OFTSR 的价值在于 1 NFE 下仍能给出合理的两端指标，而不是单项第一。


Table 1: Noiseless quantitative results on DIV2K. We compute the average PSNR (dB), LPIPS and FID of different methods on $4 \times$ SR. The best and second best results are highlighted in bold and underline. The distilled model produces superior performance in terms of trade-off metrics through adjustment of the hyperparameter $t$ .   
Table 2: Noiseless (top) and noisy (bottom) quantitative results on FFHQ $\pmb { 2 5 6 } \times 2 5 6$ . We compute the average PSNR (dB), LPIPS and FID of different methods on $4 \times$ SR. The best and second best results are highlighted in bold and underline.
> 💡 **指标批读**: 作者把 noisy/noiseless FFHQ 和 DIV2K 放一起，是为了说明方法不只适合人脸或干净 bicubic。注意 noisy setting 下 PSNR 与 LPIPS 的排序可能更能暴露保真-真实感冲突。


![Table 1](../images/a12df72f12d83fb80b6dec976e4c6e326796677e345a8f2d2c11c00c62af6cd6.jpg)
*Table 1: Table 1: Noiseless quantitative results on DIV2K. We compute the average PSNR (dB), LPIPS and FID of different methods on $4 \times$ SR. The best and second best results are highlighted in bold and underline. The distilled model produces superior performance in terms of trade-off metrics through adjustment of the hyperparameter $t$ . Table 2: Noiseless (top) and noisy (bottom) quantitative results on FFHQ $\pmb { 2 5 6 } \times 2 5 6$ . We compute the average PSNR (dB), LPIPS and FID of different methods on $4 \times$ SR. The best and second best results are highlighted in bold and underline.*
> 💡 **表格批读**: 重点看 distilled OFTSR 在不同 $t$ 下是否形成一组点：高 PSNR 档、高 LPIPS/FID 档都能被同一个 one-step 模型覆盖，才支撑“tunable”。


# 4.1 EXPERIMENTAL SETUP
> 💡 **小节预览**: 这一节决定公平性：数据集、退化模型、teacher 来源、指标和 baseline 分类都要读清，否则后面的 SOTA claim 不好判断。


Datasets. We perform extensive SR experiments on the FFHQ $2 5 6 \times 2 5 6$ (Karras et al., 2019), DIV2K (Agustsson & Timofte, 2017) and ImageNet $2 5 6 \times 2 5 6$ (Russakovsky et al., 2015) datasets to assess the bicubic SR performance of OFTSR on faces and natural images. For each dataset, we evaluate on 100 hold-out validation images without cherry-picking. For evaluating real SR, we use both synthetic set and real world set. Synthetic set includes 100 images from Imagenet for $6 4  2 5 6$ SR and 100 images from DIV2K for $1 2 8 \to 5 1 2$ SR, both degraded using RealESRGAN pipeline. Real world test set includes RealSR (Cai et al., 2019), RealSet80 (Yue et al., 2024b) and RealLQ250 (Ai et al., 2025). For distilling DiT4SR, we construct the training set using a combination of images from DIV2K (Agustsson & Timofte, 2017), DIV8K (Gu et al., 2019), Flickr2K (Timofte et al., 2017), LSDIR (Li et al., 2023b) and the first 10K images from FFHQ (Karras et al., 2019).
> 💡 **数据批读**: 100 hold-out images 适合快速比较但统计方差不可忽略；real-world 结果更依赖退化覆盖，DiT4SR 蒸馏又用了更大混合训练集，和自训 teacher 结果不要混为一个证据。


Teacher Models. We employ three types of teacher models in our experiments: (1) Self-trained teachers (2 types of backbones: Guided Diffusion (Dhariwal & Nichol, 2021) for bicubic SR (Tabs. 1 to 3) and ResShift (Yue et al., 2024b) for real SR (Tabs. 4 and 5)) using the noise-augmented conditional flow strategy in Sec. 3.1, which showcases the effectiveness of our training scheme; (2) An off-the-shelf DiT4SR teacher (built on Stable Diffusion (SD) 3.5). Since SD-based models possess significantly stronger generative priors and are prohibitively expensive for us to pre-train, we distill DiT4SR with our method to enable fair comparison with the latest SOTA approaches for realSR (Tab. 6); (3) An off-the-shelf ResShift teacher, allowing direct comparison with SinSR (which is distilled from ResShift) and fair computational cost comparison (Tabs. 9 and 10).
> 💡 **teacher 批读**: 这是最容易误读的地方：OFTSR 既是一个 teacher training recipe，也是一个 distillation recipe。DiT4SR 结果更多证明 distillation 可迁移，不能全部归功于 OFTSR 自训 flow prior。


Evaluation Metrics. The metrics we use for comparison are Peak Signal-to-Noise Ratio (PSNR), Frechet Inception Distance (FID), and Learned Perceptual Image Patch Similarity (LPIPS) (Zhang ´ et al., 2018) distance. The FID evaluates the visual quality by calculating the feature distance between two image distributions. In our experiments, we calculate the FID using the HR images and the restored images from the 100 hold-out validation set with Clean-FID (Parmar et al., 2022). LPIPS measures the average perceptual similarity between the restored images and their corresponding HR images. PSNR measures the restoration faithfulness between two images. And LPIPS and PSNR are the two main metrics we use to measure the perceptual-fidelity trade-offs. For real SR task, we also use no-reference Image Quality Assessment (IQA) including NIQE Zhang et al. (2015), CLIPIQA Wang et al. (2023b), MUSIQ Ke et al. (2021), and MANIQA Yang et al. (2022).
> 💡 **实验设置**: 这里要核对三点：训练退化是否公平、测试集是否真实、指标是否同时覆盖 fidelity 与 perceptual quality。


Compared Methods. We conduct comprehensive comparisons against state-of-the-art diffusionbased image super-resolution methods, which can be categorized into two groups: (1) Training-free methods, including DPS (Chung et al., 2022), DDRM (Kawar et al., 2022), DDNM (Wang et al., 2022b), DiffPIR (Zhu et al., 2023), CDDB (Chung et al., 2024), and SITCOM (Alkhouri et al., 2024); (2) Training-based methods: GOUB (Yue et al., 2023), ECDB (Yue et al., 2024a), InDI (Delbracio & Milanfar, 2023), DAVI (Lee et al., 2024), I2SB (Liu et al., 2023a), DDC (Chen et al., 2024), ResShift (Yue et al., 2024b), SinSR (Wang et al., 2024c) and CTMSR (You et al., 2025). And large scale Stable Diffusion based methods such as OSEDiff (Wu et al., 2024), AddSR (Xie et al., 2024) and TSDSR (Dong et al., 2025). It is noteworthy that SITCOM requires K inner-iterations to evaluate and differentiate the score function at each sampling step. To further validate the effectiveness of our method, we conduct experiment on real-world image super-resolution. Following (Yue et al., 2024b; Wang et al., 2024c), we use Imagenet $2 5 6 \times 2 5 6$ as HR training data and synthesize LR images using degradation pipeline of RealESRGAN (Wang et al., 2021).
> 💡 **baseline 批读**: training-free 方法常用很多 NFEs，training-based/SD 方法可能有更强先验或更多训练数据。公平比较时要把 NFE、teacher/backbone、训练数据和是否 one-step 一起看。


Table 4: Quantitative results of real-world image superresolution on ImageNet $2 5 6 \times 2 5 6$ and RealSR (Cai et al., 2019). The best and second best results are highlighted in bold and underline. The number of inference steps is indicated by ‘s’, which is the same as NFE when not use CFG.
> 💡 **实验批读**: 本节证据要反问：OFTSR 的主模块是否分别被消融证明，速度/质量/可控性是否同时成立。


Table 3: Noiseless quantitative results on ImageNet $2 5 6 \times 2 5 6$ . We compute the average PSNR (dB), LPIPS and FID of different methods on $4 \times \mathrm { S R }$ . The best and second best results are highlighted in bold and underline.
> 💡 **指标解读**: PSNR/SSIM 偏结构保真，LPIPS/DISTS/NIQE/MUSIQ/CLIPIQA 偏感知质量；one-step SR 的 claim 通常要看两类指标是否同步成立。


![Table 3](../images/5007da874231420f4acdcf905607c2eebd9245d5e9970520314f886806244e08.jpg)
*Table 3: Table 3: Noiseless quantitative results on ImageNet $2 5 6 \times 2 5 6$ . We compute the average PSNR (dB), LPIPS and FID of different methods on $4 \times \mathrm { S R }$ . The best and second best results are highlighted in bold and underline.*
> 💡 **表格批读**: ImageNet 比 FFHQ 更开放，若 OFTSR 仍能在 LPIPS/FID 与 PSNR 间形成合理档位，说明 $t$ 控制不只是人脸域特例。


![Table extracted](../images/eedfca804729351cbb5b3cd83aa19dfde443fb5a456bb2d8eb26d9a36bc19a2d.jpg)
*Table extracted: Table extracted by MinerU. ImageNet PSNR (↑) LPIPS (↓) FID (↓) NIQE (↓) MUSIQ (↑) MANIQA (↑) CLIPIQA (↑) SwinIR (Liang et al., 2021) 22.24 0.320 207.75 6.0084 47.46 0.5527 0.5544 NAFNet (Chen et al., 2022) 2*
> 💡 **表格批读**: 表格的价值在于把 claim 数值化：重点看是否有同等设置、同等步数、同等参数/延迟下的公平比较。


![Figure 5](../images/68ab2b8212b6cddb3d7ff0663cd0b487376ad9a355c276a14e52ee388633d0d6.jpg)
*Figure 5: Figure 5: Qualitative comparison with training-based methods. The first two columns demonstrate $4 \times \mathrm { S R }$ results on the FFHQ dataset with noise level $\sigma _ { n } = 0 . 0 5$ . The remaining columns show noiseless $4 \times \mathrm { S R }$ results on the ImageNet dataset. Numbers next to the method names represent the required NFEs.*
> 💡 **Figure 批读**: Fig. 5 对 training-based 方法，重点看 OFTSR 是否在 1 NFE 下同时保住 noisy FFHQ 的身份结构和 ImageNet 的自然纹理；方法名旁的 NFE 是速度证据。


Training Details. We do experiments for both noisy and noiseless SR. For noiseless SR, bicubic downsampling is performed on all three datasets. For noisy SR, we conduct experiment only on FFHQ $2 5 6 \times 2 5 6$ dataset with average-pooling downsampling and Gaussian noise with a standard deviation $\sigma _ { y } = 0 . 0 5$ . All images are normalized to the range of $[ - 1 , 1 ]$ . For experiments on FFHQ $2 5 6 \times 2 5 6$ and DIV2K, we adopt the same model architecture used for FFHQ in (Chung et al., 2022); and for experiment on ImageNet $2 5 6 \times 2 5 6$ , we use the same model architecture as the pretrained unconditional model used in (Dhariwal & Nichol, 2021). We modify the input convolution layer to accept concatenated image input. The first stage models are trained from scratch and are sampled with RK45 sampler by default. The one-step model is initialized from the teacher model for distillation. We use the Adam optimizer with a linear warmup schedule over 1k training steps, followed by a learning rate of $1 e - 4$ for both stages.
> 💡 **复现批读**: student 从 teacher 初始化很重要，这降低蒸馏难度；teacher 默认 RK45 sampling 说明 first-stage 仍是多步高质量模型，one-step 只发生在蒸馏后的 student。


# 4.2 RESULTS
> 💡 **小节预览**: 主结果要分三层读：teacher 多步效果、student 单步效果、不同 $t$ 的 trade-off 点。


Quantitative Results. We present comprehensive quantitative evaluations on several benchmark datasets: DIV2K, FFHQ, ImageNet and real world test set and different tasks (including noiseless SR, noisy SR and real world SR) (Tabs. 1 to 6). Our analysis reveals several findings: (i) The first-stage OFTSR achieves superior performance in perceptual metrics (FID and LPIPS) while requiring fewer than 32 NFEs. (ii) Our distillation
> 💡 **结果批读**: “teacher <32 NFEs” 与 “student 1 NFE”是两个不同速度层级。评价 one-step claim 时应主要看 distilled rows，而不是 first-stage teacher。


Table 5: Quantitative comparison on real world sets. The best and second best results are in bold and underline.
> 💡 **real-world 批读**: real-world sets 更依赖 no-reference IQA，读表时要把 NIQE/MUSIQ/MANIQA/CLIPIQA 与视觉案例合看，避免只被单个无参考指标带偏。


![Table 5](../images/d3dcf78bde94293d8677498a06f09b4137e7999f0a5c85f33fc161bf8e4cc215.jpg)
*Table 5: Table 5: Quantitative comparison on real world sets. The best and second best results are in bold and underline.*
> 💡 **表格批读**: Tab. 5 是真实退化泛化证据，但没有 GT 的指标更偏感知质量；如果高 realism 档提升 IQA，也仍需检查是否改变了原图语义结构。


algorithm is versatile, when applied to ResShift (Yue et al., 2024b) teacher, our distilled model achieved better one-step performance than SinSR (Wang et al., 2024c) (see Tab. 9). (iii) Our distilled version of OFTSR demonstrates remarkable versatility, achieving either the highest PSNR
> 💡 **通用性批读**: 蒸馏 ResShift 并优于 SinSR 是方法通用性的关键证据，因为它把 teacher 固定住，更能比较 distillation loss 本身。


Table 6: Quantitative comparison of state-of-the-art one-step SR methods on synthetic (DIV2K-Val) and real-world (RealLQ250) benchmarks. Best results are in bold, second best are underlined. Our method is tested under $t = 1$ . ResShift∗ means we train our noise-augmented conditional flow in Sec. 3.1 using the ResShift model architecture then distillation; and DiT4SR means we use the pretrained DiT4SR model as the teacher model for distillation.
> 💡 **表格批读**: Tab. 6 固定 $t=1$，所以它展示的是高 realism 档的一步性能，不是完整 trade-off 曲线。ResShift* 与 DiT4SR 两行要分开读：前者验证自训 flow，后者验证强 teacher 蒸馏。


![Table extracted](../images/2646708c729bf5fb49dbb1dc98fb0b02c729c70aa247642de828365f1a20c8db.jpg)
*Table extracted: Table extracted by MinerU. Dataset Metric SinSR-1s CTMSR-1s AddSR-1s OSEDiff-1s TSDSR-1s Ours (ResShift*)-1s Ours (DiT4SR)-1s DIV2K-Val PSNR ↑ 24.50 24.87 22.39 23.86 22.17 23.91 22.80 SSIM ↑ 0.6136 0.6349 0*
> 💡 **表格批读**: 这张表直接对 one-step SOTA。OFTSR(DiT4SR) 和 OFTSR(ResShift*) 不应合并读：一个体现强 SD teacher 的上限，一个更能看 OFTSR 自训 conditional flow 的能力。


scores or ranking among the top two methods for FID and LPIPS metrics in one step. This indicates minimal performance degradation between the teacher and student models. (iv) Our experiments suggest that FID serves as a more reliable indicator of perceptual quality and better captures the performance gap between teacher and student models during distillation. (v) When applied to a powerful SD-based SR model (DiT4SR), our distillation algorithm produces a one-step generator whose performance is competitive with other leading SOTA distillation methods. This also validates the versatility of our distillation algorithm.
> 💡 **指标风险**: FID 对分布感知质量敏感，但不保证单张图结构忠实；LPIPS 也可能偏好纹理。真实应用仍要把 PSNR/结构检查和 no-reference IQA 一起看。


Visual Results. Our experimental results demonstrate that OFTSR achieves high-quality image reconstructions. We evaluate OFTSR against leading training-free methods for $4 \times \mathrm { S R }$ , as shown in Fig. 4. While DPS can produce sharp reconstructions, it requires 1000 NFEs and often introduces significant distortions. In contrast, OFTSR successfully preserves structural information from lowresolution inputs while reconstructing fine details. Notably, our distilled version of OFTSR requires only one NFE, as other training-free methods suffer from severe error accumulation when using less than 10 NFEs. As illustrated in Fig. 5, we also compare OFTSR against state-of-the-art SR methods that require training. The results show that our approach generates patterns with rich, natural details. Furthermore, our distilled model enables flexible control over the fidelity-realism trade-offs in the generated high-resolution images. Fig. 3 demonstrates this capability through examples of noisy $4 \times$ SR with varying degrees of realism and fidelity. More qualitative comparison and visual examples can be found in Sec. K
> 💡 **视觉批读**: 视觉结果要专门看 hallucination：人脸身份、物体边缘、文字/纹理方向、平坦区域噪声。高 realism 档如果创造了 LR 不支持的细节，在 PSNR 上可能会吃亏。


# 4.3 ABLATIONS

Perturbation Strength $\sigma _ { p }$ . In Tab. 7, we evaluate the design choices in the simple conditional flow training stage. All experiments in this ablation study are conducted under identical training conditions, with performance metrics measured using the RK45 solver. The most critical hyperparameter in this ablation is the strength of the perturbation $\sigma _ { p }$ . Consistent with previous works, we confirm that perturbation is essential for generating perceptually compelling images from LR inputs. Notably, we discover that increasing perturbation strength does not necessarily improve perceptual quality but instead leads to more curved PF-ODE, requiring additional NFEs to solve (see Tab. 7). Furthermore, our experiments demonstrate that conditioning on $\mathbf { x } _ { \mathrm { L R } }$ is crucial to compensate for information loss during perturbation. We also find that $\ell _ { 1 }$ loss outperforms $\ell _ { 2 }$ loss for our specific task. While (Kim et al., 2024) previously highlighted the significance of Gaussian perturbation, our work is the first to systematically analyze the relationship between noise perturbation and the trade-off between generation quality and efficiency in flow-based models.
> 💡 **消融批读**: $\sigma_p$ 的结论很关键：更随机不等于更好。过大 perturbation 让 flow 更弯、solver 更贵，也可能让 LR condition 承担过多信息恢复压力。


Distillation Design Space. In Tab. 8, we evaluate several crucial design choices for the distillation stage, including the distillation loss type, solver type, $\mathrm { d } t$ value, and the weighting of alignment and boundary losses. Since learning ${ \mathbf { v } } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , 0 )$ is considerably easier than learning $\mathbf { v } _ { \phi } ( \mathbf { x } _ { 0 , \mathrm { L R } } , 1 )$ , we utilize metrics from the latter to decide our distillation hyperparameters. Our analysis of the step size
> 💡 **消融批读**: 用 $t=1$ 选超参是合理的压力测试，因为高 realism 端最难学；但这也意味着调参可能偏向感知端，保真端表现要另看曲线。


Table 7: Ablation on noiseless FFHQ $2 5 6 \times 2 5 6$ first stage. The default training setting is ${ \mathfrak { b s } } = 3 2$ $\mathrm { ~ k ~ } = \ 0 . 0 0 0 1$ ; loss type $\mathit { \Theta } = \mathit { \Theta } \ell _ { 1 }$ ; with condition; all experiments are trained for $1 0 0 \mathrm { k }$ steps. The final choice is highlighted to balance the performance and efficiency.
> 💡 **消融批读**: Tab. 7 是 stage-1 teacher 的设计依据，尤其要看 no condition、不同 $\sigma_p$、loss type 对 FID/LPIPS/NFE 的影响。


![Table 7](../images/6797bd1a22bec3d2e94e483c9c065869a955f2281342589e7fe61a13747d6bad.jpg)
*Table 7: Table 7: Ablation on noiseless FFHQ $2 5 6 \times 2 5 6$ first stage. The default training setting is ${ \mathfrak { b s } } = 3 2$ $\mathrm { ~ k ~ } = \ 0 . 0 0 0 1$ ; loss type $\mathit { \Theta } = \mathit { \Theta } \ell _ { 1 }$ ; with condition; all experiments are trained for $1 0 0 \mathrm { k }$ steps. The final choice is highlighted to balance the performance and efficiency.*
> 💡 **表格批读**: 这张表支撑“LR condition 必不可少、$\sigma_p$ 要折中”的结论；如果只看最高 perceptual 指标，可能会忽略对应 NFE/flow curvature 成本。


Table 8: Ablation on noiseless FFHQ $2 5 6 \times 2 5 6$ distillation stage. The default training setting is $\ b s \ = \ 8$ ; $\sigma _ { p } = 0 . 1$ , $\mathrm { l r } = 0 . 0 0 0 1$ ; loss type $= \ell _ { 2 }$ ; with LR condition; all experiments are trained for 20k steps; And the one-step metrics are calculated with $t = 1$ . Ablations in subgroups can be ordered as $\mathrm { d } t \  \ \lambda _ { \mathrm { B C } } \  \ \lambda _ { \mathrm { a l i g n } } \ $ dt reveals that smaller values do not necessarily yield better results, leading us to select $\mathrm { d } t = 0 . 0 5$ for subsequent experiments. Our proposed loss function demonstrates substantial improvement over both the original BOOT (Gu et al., 2023) loss and PINN (Tee et al., 2024) distillation loss, achieving a significant LPIPS score improvement of more than 0.1. Further experimentation shows that both the alignment loss (Eq. (10)) and boundary loss (Eq. (11)) contribute to enhanced performance. By combining these losses with a Midpoint 2-order solver, we achieve additional improvements in our one-step model’s performance at $t = 1$ .
> 💡 **消融结论**: Tab. 8 是支撑方法 novelty 的关键表：OFTSR loss 要优于 BOOT/PINN，alignment 和 boundary 都要有增益，solver 选择也影响最终 one-step realism。


Solver , and $\mathrm { d } t \ $ Distillation Loss .

![Table 8](../images/f7518f32363a4dc32c1c99ced8af2132a3656553e67de2414f482150567d4da9.jpg)
*Table 8: Solver , and $\mathrm { d } t \ $ Distillation Loss .*
> 💡 **表格批读**: Tab. 8 要读四组因素：loss 形式、solver、$dt$、alignment/boundary 权重。它直接回答“为什么不是 BOOT/PINN，为什么要这三个 loss”。


# 4.4 COMPUTATIONAL OVERHEAD

Training Cost Comparison. Our distillation algorithm is highly flexible and can be easily applied to any pre-trained diffusion/flow-based conditional model. As shown in Tab. 9, we applied our distillation algorithm to the ResShift(Yue et al., 2024b) pre-trained model and achieved teacherlevel performance in one step, surpassing SinSR(Wang et al., 2024c) in FID with much less training compute. Even taking the training stage into account with a larger model, our method remains more efficient than ResShift. We use $t = 1$ for OFTSR.
> 💡 **成本批读**: 这里的公平性较强，因为同样围绕 ResShift teacher 对比 SinSR。若 OFTSR 训练成本更低且 FID 更好，说明 trajectory loss 有实际工程价值。


Table 9: Comparison of training cost on single NVIDIA A100.

![Table 9](../images/dc921e452b879de5adc6c67378902cd0c50049348d96a6d3f9ffa98638c97738.jpg)
*Table 9: Table 9: Comparison of training cost on single NVIDIA A100.*
> 💡 **表格批读**: Tab. 9 关注训练成本而非最终画质；同样以 ResShift 为 teacher 时，OFTSR 若更省训练 compute，说明它的 trajectory alignment 比模拟完整 teacher 轨迹更经济。


Inference Cost Comparison. We have included a detailed comparison of the inference cost in Tab. 10, using FLOPS and MAC to measure model complexity. We use $t = 1$ for OFTSR.
> 💡 **部署批读**: 1 NFE 不等于绝对最快，还要看 backbone 参数量、FLOPs/MAC、分辨率切 patch 开销。DiT4SR teacher 蒸馏出的 student 可能仍比轻量 CNN/SwinIR 重。


Table 10: Comparison of inference cost on single NVIDIA A100.

![Table 10](../images/acb01dc280931857b3c1d9debfad33ed63a5cbfa0ba3f543cc3265c0a50bd670.jpg)
*Table 10: Table 10: Comparison of inference cost on single NVIDIA A100.*
> 💡 **表格批读**: Tab. 10 用 FLOPs/MAC 补充 NFE 之外的部署成本；one-step 的速度优势还要受 backbone 和输出分辨率影响。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 读表顺序 | teacher 来源 → one-step student → $t$ 档位 → cost |
| 核心检查 | 1 NFE、PSNR/LPIPS/FID、真实退化、消融是否同时支撑 |
| 风险 | SD teacher 结果与 OFTSR 自身贡献要区分；no-reference 指标不保证结构真实 |

### 核心洞察
1. 主结果证明“能做”，Tab. 7 证明 noise-augmented teacher 的必要性，Tab. 8 证明 distillation loss 的必要性。
2. $t=1$ 常代表高 realism 档，不等于所有档位都优；可调性要看曲线和视觉序列。
3. 成本比较显示 one-step 有部署吸引力，但 teacher 训练和强 backbone 仍是门槛。

### 可追问点
- OFTSR 和 diffusion one-step 的差别在哪里？
- 为什么 flow 适合可调 trade-off？
- DiT4SR 蒸馏结果中，多少提升来自 teacher prior，多少来自 OFTSR loss？
