[← 返回 README](../README.md)

# Appendix

## 📌 预览
附录补充 RCOD 的训练配置、伪代码、完整结果和消融。重点查四件事：LoRA/训练成本是否可复现，推理时如何用 $t$ 控制 realism，VPIM/DAS 是否各自有效，$n=3$ 的分组选择是否有数据支撑。
---

> 💡 **Q&A 批注记录**:
> - Q: 为什么单步 SR 难以控制 realism？
> - A: 因为多数 one-step 模型在固定 timestep/固定蒸馏强度下学习一个确定映射，推理时没有多步采样那样的 noise schedule 可调自由度。
> - Q: 附录最关键的消融是哪两个？
> - A: Table S5 看 VPIM 与 DAS 是否有效，Table S6 看 group 数量 $n=3/4$ 对控制和性能的影响。
> - Q: 推理阶段还需要计算 $M_L$ 吗？
> - A: 手动控制时不需要，直接指定 $t$；自动控制才需要 MEM 估计 LR 的退化 group。


# Supplementary Material for “Realism Control One-step Diffusion for Real-World Image Super-Resolution”

# In the supplementary material, we provide the following contents:

• Algorithm Details: Pseudo-code of our RCODO training strategy (referring to Section Methodology in the main paper).   
• Implement Details: Hyper-parameters setting and experimental environments (As a complement to Section Implement Details in the main paper).   
• Experiments: Full table of quantitative comparisons and more visual Comparisons (as a complement to Section Experiments in the main paper).   
• Ablation Study (referring to Section Ablation Study in the main paper).
> 💡 **附录地图**: 这四项正好补正文的证据缺口：伪代码说明训练数据流，超参说明成本，完整实验看泛化，消融确认 LDG/DAS/VPIM 不是装饰模块。


# Implement Details

Training: The training takes over $3 0 \mathrm { k }$ iterations, with a batch size of 4 and a learning rate of $\mathrm { 2 e ^ { - 5 } }$ . We adopted LoRA for fine-tuning. Follow the original setting $\mathrm { W u }$ et al. 2024a; Zhang et al. 2024), for $\mathrm { R C O D } _ { \mathrm { S } }$ , the rank parameter in LoRA is set as 16 for the VAE encoder and 32 for the diffusion UNet, respectively. For $\mathrm { R C O D _ { O } }$ , all rank is set to 4.
> 💡 **训练配置**: 30k iter、batch 4、lr 2e-5、LoRA fine-tune。RCOD_S 的 VAE/UNet LoRA rank 更高，RCOD_O 全部 rank=4；这说明两条底座的可训练容量不同，横向比较时要注意不只是模块差异。


Hardware and Software Environments: We train and test our model on NVIDIA A100 GPU. The amount of training memory is about $2 8 \ \mathrm { G B }$ with batch size 1 using $\mathrm { R C O D _ { O } }$ . We use a Linux system with pytorch version 2.4.
> 💡 **复现成本**: A100、PyTorch 2.4、RCOD_O batch 1 约 28GB 显存。单步推理快不代表训练便宜，部署前还要看目标硬件上的 VAE/UNet/VPIM 实际延迟。


# Algorithm Details

Pseudo-code The pseudo-code of our RC-OSD training strategy is summarized as Algorithm 1. In inference stage, we do not need to compute $M _ { L }$ . Instead, we directly calculate $\hat { z } _ { H } = F _ { \theta } ( z _ { L } ; t , c _ { y } )$ , where $t$ a user-specified parameter that controls the level of realism for SR output. The choice of $t$ depends on the different application scenarios and is discussed in Subsection “Latent Domain Grouping”.
> 💡 **伪代码重点**: 训练时需要 HR 来算 $M_L$ 并分组；推理时手动控制不需要 $M_L$，直接把用户指定的 $t$ 送入 $F_\theta(z_L;t,c_y)$。这解释了 RCOD 为什么能提供 slider 式控制。


# Experiments

Quantitative Comparisons: We quantitatively compare recent SOTA SR methods in Table S4 including synthetic and real-world data, multi-step diffusion, and GAN-based methods. Compared with multi-step diffusion methods, our methods demonstrate better NR metrics and perceptual FR metrics across three datasets. Furthermore, their FR fidelity metrics are more competitive with those of multi-step diffusion methods when compared to other OSD methods. On the DrealSR dataset, $\mathrm { R C O D _ { O } }$ -Fid. can even surpass some multistep diffusion methods in many FR metrics (PSNR, SSIM, and LPIPS) and NR metrics (MUSIQ, MANIQA, and CLIP-IQA).
> 💡 **完整结果批读**: Table S4 是主表扩展版，加入 synthetic/real-world、多步 diffusion、GAN。这里要看 RCOD 是否只是 OSD 内部领先，还是在 NR/perceptual FR 上能接近甚至超过多步方法。


Qualitative Comparisons: We conduct more qualitative comparisons in Fig. S7. On Sony 0049 (DRealSR), $\mathrm { R C O D _ { O } }$ -Fid. shows fewer artefacts of handrail than other OSD methods. On Nikon 047 (RealSR) and 0802 pch 00007 (DIV2K), $\mathrm { R C O D _ { O } }$ -Real. and $\mathrm { R C O D } _ { \mathrm { S } }$ - Real. both generate more detailed textures than other OSD methods.
> 💡 **视觉补充**: 附录视觉图给出两个方向：Fid. 档减少结构伪影，Real. 档增强纹理。批读时要把它们当成两个不同使用场景，而不是简单判断哪个“更好看”。


# Ablation Study

Effectiveness of VPIM and DAS: To validate the effectiveness of VPIM, we perform ablation studies by replacing the VLM and text encoder (“Text Model”) with VPIM in Table S5 for the “w/o VPIM-Fid.” and “w/o VPIM-Real.” configurations. We observe that both configurations achieve better PSNR, LPIPS, and MANIQA scores than the baseline (the original OSEDiff). This indicates that VPIM provides not only sufficient semantic information but also fidelity information from LR images. By removing DAS during distillation, as implemented in the w/o DAS-Fid.” and w/o DAS-Real.” configurations, we observe that DAS amplifies the performance gap between the fidelity-oriented and realism-oriented configurations. The fidelity-oriented variant achieves higher PSNR and lower LPIPS, while the realism-oriented variant yields higher MANIQA scores—indicating improved realism. This divergence in evaluation metrics demonstrates that DAS effectively enhances LDG’s ability to decouple fidelity preservation from realism generation, enabling finer control over the fidelity–realism trade-off through adaptive timestep regularization.
> 💡 **消融批读**: Table S5 是 VPIM/DAS 的关键证据。VPIM 替换 text model 后 PSNR、LPIPS、MANIQA 都改善，说明 LR visual token 不只是语义替代，也给了 fidelity 线索。DAS 的作用不是单纯提升一个数，而是拉开 Fid. 与 Real. 的指标差异，让 LDG 的可控性更明显。


Choice of $n$ : As mentioned in section “Latent Domain Grouping”, $n$ can only be $\leq 4$ . For sufficient flexibility, we
> 💡 **分组数问题**: $n$ 是控制档位数量。更大的 $n$ 看似更灵活，但若某些 $M_L$ 区间样本太少，高 $t$ 档会训练不足，反而损害 Real. 档质量。


![Table extracted](../images/5f1f0392764f9c069d52998f9af923330b55fc189a58668662e9a932c70d1e67.jpg)
*Table extracted: Table extracted by MinerU. Datasets Type Methods |PSNR↑ SSIM↑ LPIPS↓ DISTS↓ NIQE↓ MUSIQ↑ MANIQA↑ CLIPIQA↑ DIV2K-Val Non-Diff. BSRGANReal-ESRGAN 24.580.62690.33510.22754.752 61.20 0.5071 0.524724.290.63710.31*
> 💡 **表格批读**: MinerU 抽取的表格可读性较差，建议以图片 Table S4 为准。批读重点仍是分数据集、分方法类别比较 FR/NR 指标，而不是逐个 OCR 数字硬读。


Table S4: Quantitative comparison with state-of-the-art methods on both synthetic and real-world benchmarks. The best and second best results within both multi-step and one-step diffusion-based methods are highlighted in red and blue, respectively.
> 💡 **Table S4 批读**: 红/蓝标注区分 multi-step 与 one-step 内部排名，避免把计算成本不同的方法混成一个绝对榜。RCOD 如果在 one-step 里同时保持速度和更好 FR/NR，论证就比较完整。


![Figure S7](../images/34d7ff87219a623c0c149df08176c7431fb0335c17e995b35facda46b3359b84.jpg)
*Figure S7: Figure S7: Visual comparison $( \times 4 )$ on RealSR (Nikon 047), DRealSR (Sony 49), and DIV2K (0802 pch 00007) data.*
> 💡 **Figure 批读**: Fig. S7 跨 RealSR/DRealSR/DIV2K 展示泛化。重点看真实相机数据上的栏杆、细线、重复纹理：这些最容易暴露“感知增强”是否变成伪细节。


![Table extracted](../images/ef55c7b28f74c0e25ab3a0355fa0fa990342d1797964bd3844f9df25f995aa09.jpg)
*Table extracted: Table extracted by MinerU. TOhal0 VPIM Y , training iteration N iz ← LRA;←RA ← parameters Yθ ← Yφ; random initialize trainable parameters MLPθ 2 for i ← 1 to N do 3 Sample low-resolution image x L and corres*
> 💡 **算法批读**: 这个 OCR 片段对应 Algorithm 1。即使文字损坏，也能看出训练循环包含采样 LR/HR、计算 latent、计算 $M_L$/group、更新 LoRA/MLP 等步骤；完整理解应结合方法段。


16 end Output: Trained generator $G _ { \theta }$

consider $n \geq 3 .$ . Therefore, we compare the training results for $n = 3$ and $n = 4$ in Table S6. We find that $\mathrm { R C O D _ { O } }$ -Real. $\mathit { n } = 4$ ) performs much worse than $\mathrm { R C O D _ { O } }$ -Real. $( n = 3$ ). To explain this phenomenon, we visualize the training data distribution for the two choices in Fig. S8. We observe that there is too little data for training at large $t$ (when $M _ { L }$ is small). Thus, we finally choose $n = 3$ for our experiment.
> 💡 **n 的消融**: $n=4$ 的 Real. 档明显变差，原因是小 $M_L$/大 $t$ 区间样本太少。这个结果很重要：RCOD 的可控性不是档位越多越好，而是受训练数据退化分布约束。


Robustness of $M _ { L }$ : Due to the encoder of the VAE being trainable in most SD-based method settings, we consider a domain shift phenomenon during training: the distribution of $M _ { L }$ may change after training. Therefore, we compare the $M _ { L }$ values of the entire training dataset before and after training in main manuscript Fig. 4 (b). The histogram shows a slight shift in the metric, with values becoming closer to larger similarities, but the overall distribution remains similar. This indicates that even though the model aligns the feature domains of LR and HR, $M _ { L }$ retains its robustness in measuring the divergence in the latent domain.
> 💡 **鲁棒性批读**: 由于 VAE encoder 可训练，$M_L$ 分布可能随训练漂移。作者观察到整体分布仍类似，这是 LDG 可持续使用的必要条件；但这仍主要是训练集统计，跨真实退化域还需要更多验证。


Correlation of $M _ { L }$ and image quality: In Table 1(b) in main manuscript, we calculate |Spearman coefficient| $\uparrow$ of different $M _ { L }$ and image quality metrics (which are calculated by LR and HR images). Fig. S9 visualize the correlation of randomly selected 600 points. We can find that cosine similarity exhibits a higher correlation coefficient with image quality metrics compared to other distances.
> 💡 **相关性批读**: Fig. S9 进一步说明 cosine similarity 的排序能力强于 L1/MSE。这里的证据支撑“分组大致合理”，但不能证明 $M_L$ 是完美退化估计器。


![Figure S8](../images/64844ef206aa08b6696a45d228629c314abdc97853f4e85f5964f0357b659caa.jpg)
*Figure S8: Figure S8: Distribution of (a) mean value of LR and HR training images in the latent domain, (b) the $M _ { L }$ metric in the latent domain of VAE before and after training.*
> 💡 **Figure 批读**: Fig. S8 用分布解释 $n=4$ 失败：高 $t$ 档训练样本稀疏，导致 Real. 档不稳定。这也是 README 里应写入的局限。


# References

Agustsson, E.; and Timofte, R. 2017. Ntire 2017 challenge on single image super-resolution: Dataset and study. In CVPRW, 126–135.
> 💡 **参考文献批读**: DIV2K 是合成测试来源之一，主要检验受控 degradation pipeline 下的 SR 表现。


Cai, J.; Zeng, H.; Yong, H.; Cao, Z.; and Zhang, L. 2019. Toward real-world single image super-resolution: A new benchmark and a new model. In ICCV, 3086–3095.
> 💡 **参考文献批读**: RealSR 是真实 LR-HR 配对测试集，能比纯合成数据更好检验 Real-ISR 泛化。


Chen, B.; Li, G.; Wu, R.; Zhang, X.; Chen, J.; Zhang, J.; and Zhang, L. 2025. Adversarial diffusion compression for real-world image super-resolution. In CVPR, 28208–28220.
> 💡 **参考文献批读**: ADCSR 属于提升 OSD 性能/效率的近邻工作，可作为 RCOD 是否通用的潜在对照。


Chen, J.; Pan, J.; and Dong, J. 2025. Faithdiff: Unleashing diffusion priors for faithful image super-resolution. In CVPR, 28188–28197.
> 💡 **参考文献批读**: FaithDiff 代表“faithful SR”方向，提醒 RCOD 的 Real. 档需要持续关注结构忠实性。


Table S5: Ablation study of modules. The best results are highlighted in bold.
> 💡 **消融入口**: Table S5 是判断 VPIM/DAS 贡献的核心表，优先看去掉模块后 Fid./Real. 两档差异是否缩小、指标是否退化。


![Table S5](../images/7e16e77c629b4eef969d47b07d75e812f25f21806040cfd7aaa2acf946f53d47.jpg)
*Table S5: Table S5: Ablation study of modules. The best results are highlighted in bold.*
> 💡 **Table S5 批读**: VPIM 的价值看“替代文本后是否同时有 fidelity 和 realism 收益”；DAS 的价值看“是否增强 Fid. 与 Real. 的分离度”。这比只看 bold 数字更符合 RCOD 的主 claim。


![Table extracted](../images/f13efd3f309aa98cbf2da671df93877cc4f15592f142c2952553632d1ffe1b5e.jpg)
*Table extracted: Table extracted by MinerU. PSNR↑ DISTS↓ LPIPS↓ MANIQA↑ RCODo-Fid. (n = 4) 26.00 0.2111 0.2805 0.6490 RCODo-Fid. (n = 3) 26.01 0.2103 0.2796 0.6647 RCODo-Real. (n = 4) 24.02 0.2590 0.3588 0.6400 RCODo-Real. (*
> 💡 **Table S6 批读**: OCR 已经能看出 $n=4$ 的 Real. 档 PSNR/LPIPS/MANIQA 更差。结论是 RCOD 当前更适合少数稳定档位，而不是细粒度连续控制。


Table S6: Ablation study of grouping strategy. The best results are highlighted in bold.
> 💡 **分组消融**: Table S6 支撑 $n=3$ 的选择，也暴露局限：控制粒度受数据分布限制，需要更均衡采样或连续估计才能扩展到更多档。


Chen, T.; Kornblith, S.; Norouzi, M.; and Hinton, G. 2020. A simple framework for contrastive learning of visual representations. In ICML, 1597–1607.
> 💡 **参考文献批读**: SimCLR 代表用 cosine/representation similarity 衡量高维特征关系的背景，支撑作者选择 cosine similarity 做 $M_L$。


Ding, K.; Ma, K.; Wang, S.; and Simoncelli, E. P. 2020. Image quality assessment: Unifying structure and texture similarity. IEEE PAMI, 44(5): 2567–2581.
> 💡 **参考文献批读**: DISTS 是 RCOD 用来评估 perceptual FR 质量的重要指标，能补 PSNR/SSIM 对纹理感知不足的问题。


Dong, C.; Loy, C. C.; He, K.; and Tang, X. 2015. Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence, 38(2): 295–307.
> 💡 **参考文献批读**: SRCNN 是经典 SR 起点，用来说明 RCOD 所处任务从早期 supervised SR 演进到真实退化 + diffusion prior。


Dong, L.; Fan, Q.; Guo, Y.; Wang, Z.; Zhang, Q.; Chen, J.; Luo, Y.; and Zou, C. 2025. Tsd-sr: One-step diffusion with target score distillation for real-world image superresolution. In CVPR, 23174–23184.
> 💡 **参考文献批读**: TSD-SR 是 one-step distillation 近邻，和 RCOD 的差别在于是否提供退化感知的推理控制。


Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. In NeurIPS.
> 💡 **参考文献批读**: DDPM 是 timestep/noise schedule 语义的源头，RCOD 对 $t$ 的控制解释依赖这套扩散过程。


Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. Lora: Low-rank adaptation of large language models. In ICLR.
> 💡 **参考文献批读**: LoRA 是 RCOD 控制训练成本的关键实现手段，但也意味着方法表现会明显依赖预训练 SD/SD-Turbo 底座。


Ignatov, A.; Kobyshev, N.; Timofte, R.; Vanhoey, K.; and Van Gool, L. 2017. Dslr-quality photos on mobile devices with deep convolutional networks. In ICCV.
> 💡 **参考文献批读**: 早期真实图像增强/SR 工作说明 Real-ISR 的应用牵引来自真实拍摄设备，而不只是合成退化。


Ji, X.; Cao, Y.; Tai, Y.; Wang, C.; Li, J.; and Huang, F. 2020. Real-world super-resolution via kernel estimation and noise injection. In CVPRW.
> 💡 **参考文献批读**: kernel/noise 建模是传统 Real-ISR 处理未知退化的路线，RCOD 则把退化程度映射到 latent/timestep 控制。


Karras, T.; Laine, S.; and Aila, T. 2019. A style-based generator architecture for generative adversarial networks. In CVPR.
> 💡 **参考文献批读**: FFHQ 用作训练中的人脸数据补充，人脸场景对 hallucination/身份一致性尤其敏感。


Ke, J.; Wang, Q.; Wang, Y.; Milanfar, P.; and Yang, F. 2021. Musiq: Multi-scale image quality transformer. In ICCV. Kim, J.; Kwon Lee, J.; and Mu Lee, K. 2016. Accurate image super-resolution using very deep convolutional networks. In CVPR.   
Ledig, C.; Theis, L.; Huszar, F.; Caballero, J.; Cunningham, ´ A.; Acosta, A.; Aitken, A.; Tejani, A.; Totz, J.; Wang, Z.; et al. 2017. Photo-realistic single image super-resolution using a generative adversarial network. In CVPR, 4681–4690. Li, Y.; Zhang, K.; Liang, J.; Cao, J.; Liu, C.; Gong, R.; Zhang, Y.; Tang, H.; Liu, Y.; Demandolx, D.; et al. 2023. Lsdir: A large scale dataset for image restoration. In CVPR, 1775–1787.   
Liang, J.; Cao, J.; Sun, G.; Zhang, K.; Van Gool, L.; and Timofte, R. 2021. Swinir: Image restoration using swin transformer. In ICCV.   
Lin, X.; He, J.; Chen, Z.; Lyu, Z.; Fei, B.; Dai, B.; Ouyang, W.; Qiao, Y.; and Dong, C. 2023. Diffbir: Towards blind image restoration with generative diffusion prior. arXiv preprint arXiv:2308.15070.   
Liu, A.; Liu, Y.; Gu, J.; Qiao, Y.; and Dong, C. 2022. Blind image super-resolution: A survey and beyond. IEEE PAMI, 45(5): 5461–5480.   
Meng, C.; Rombach, R.; Gao, R.; Kingma, D.; Ermon, S.;
> 💡 **参考文献批读**: 这一串主要是数据集、IQA 指标、多步/单步扩散和 Real-ISR baseline，支撑实验设置与方法谱系。


![Figure S9](../images/11f21785280c3c454b280825a03f9d821a8b3ce4e73a06352d5647639736e52f.jpg)
*Figure S9: Figure S9: Visualization of |Spearman coefficient| $\uparrow$ of different $M _ { L }$ distances and image quality metrics.*
> 💡 **Figure 批读**: Fig. S9 可视化 $M_L$ 与质量指标的相关性。它支撑 cosine similarity 作为分组 proxy，但也提醒这只是相关性，不是因果退化估计。


Ho, J.; and Salimans, T. 2023. On distillation of guided diffusion models. In CVPR, 14297–14306.
> 💡 **参考文献批读**: diffusion distillation 是 OSD 的速度来源，也是固定 timestep 可控性问题出现的背景。


Q Nguyen, T. H.; and Tran, A. 2024. Swiftbrush: One-step Ptext-to-image diffusion model with variational score distil-Llation. In CVPR.   
Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In CVPR, 10684–10695.   
Salimans, T.; and Ho, J. 2022. Progressive distillation ?1for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512.   
Song, Y.; Dhariwal, P.; Chen, M.; and Sutskever, I. 2023. Consistency models. In ICML.   
SSIMSong, Y.; Sohl-Dickstein, J.; Kingma, D. P.; Kumar, A.; Ermon, S.; and Poole, B. 2020. Score-based generative model-后ing through stochastic differential equations. arXiv preprint arXiv:2011.13456.   
Sun, L.; Wu, R.; Ma, Z.; Liu, S.; Yi, Q.; and Zhang, L. 2025. Pixel-level and semantic-level adjustable super-resolution: A dual-lora approach. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2333–2343. Wang, J.; Chan, K. C.; and Loy, C. C. 2023. Exploring clip for assessing the look and feel of images. In AAAI, vol-Correlatioume 37, 2555–2563.   
0.0521 Correlation between cs_Wang, J.; Yue, Z.; Zhou, S.; Chan, K. C.; and Loy, C. C. 2024a. Exploiting diffusion prior for real-world image super-resolution. International Journal of Computer Vision, 后：1–21. Wang, X.; Xie, L.; Dong, C.; and Shan, Y. 2021. Realesrgan: Training real-world blind super-resolution with pure synthetic data. In ICCV.   
Wang, Y.; Yang, W.; Chen, X.; Wang, Y.; Guo, L.; Chau, L.- P.; Liu, Z.; Qiao, Y.; Kot, A. C.; and Wen, B. 2024b. SinSR: Diffusion-Based Image Super-Resolution in a Single Step. vs.In CVPR.   
? Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. Cosine2004. Image quality assessment: from error visibility to Similaritystructural similarity. IEEE TIP, 13(4): 600–612.   
Wang, Z.; Lu, C.; Wang, Y.; Bao, F.; Li, C.; Su, H.; and Zhu, ?? J. 2024c. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In NeurIPS. Wei, P.; Xie, Z.; Lu, H.; Zhan, Z.; Ye, Q.; Zuo, W.; and Lin, L. 2020. Component divide-and-conquer for real-world image super-resolution. In ECCV, 101–117. Springer.   
Wu, R.; Sun, L.; Ma, Z.; and Zhang, L. 2024a. Onestep effective diffusion network for real-world image superresolution. NeurIPS, 37: 92529–92553.   
e and psnr: -0.6121Wu, R.; Yang, T.; Sun, L.; Zhang, Z.; Li, S.; and Zhang, L. 2024b. Seesr: Towards semantics-aware real-world image super-resolution. In CVPR.   
Yang, S.; Wu, T.; Shi, S.; Lao, S.; Gong, Y.; Cao, M.; Wang, J.; and Yang, Y. 2022. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In snr: 0.066CVPR, 1191–1200.   
e and psnr: -0.2718Yin, T.; Gharbi, M.; Park, T.; Zhang, R.; Shechtman, E.; Durand, F.; and Freeman, W. T. 2024a. Improved Distribution Matching Distillation for Fast Image Synthesis. arXiv preprint arXiv:2405.14867. Yin, T.; Gharbi, M.; Zhang, R.; Shechtman, E.; Durand, F.; Freeman, W. T.; and Park, T. 2024b. One-step diffusion with distribution matching distillation. In CVPR, 6613–6623. Yue, Z.; Liao, K.; and Loy, C. C. 2025. Arbitrary-steps image super-resolution via diffusion inversion. In CVPR, 23153–23163.   
Yue, Z.; Wang, J.; and Loy, C. C. 2023. Resshift: Efficient diffusion model for image super-resolution by residual shifting. In NeurIPS.   
Zhang, A.; Yue, Z.; Pei, R.; Ren, W.; and Cao, X. 2024. Degradation-guided one-step image super-resolution with diffusion priors. arXiv preprint arXiv:2409.17058.   
Zhang, K.; Liang, J.; Van Gool, L.; and Timofte, R. 2021. Designing a practical degradation model for deep blind image super-resolution. In ICCV, 4791–4800.   
Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding conditional control to text-to-image diffusion models. In ICCV, 3836–3847.   
Zhang, L.; Zhang, L.; and Bovik, A. C. 2015. A featureenriched completely blind image quality evaluator. IEEE TIP, 24(8): 2579–2591.   
Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018a. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 586–595.   
Zhang, Y.; Li, K.; Li, K.; Wang, L.; Zhong, B.; and Fu, Y. 2018b. Image super-resolution using very deep residual channel attention networks. In ECCV.   
Zhu, Y.; Wang, R.; Lu, S.; Li, J.; Yan, H.; and Zhang, K. 2024. Oftsr: One-step flow for image super-resolution with tunable fidelity-realism trade-offs. arXiv preprint arXiv:2412.09465.
> 💡 **参考文献批读**: OFTSR 是“可调 fidelity-realism”的近邻工作。RCOD 与它的主要区别是把真实退化程度显式纳入 LDG/DAS，而不是只做轨迹/flow 式 trade-off 控制。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 训练 | 30k iter, batch 4, lr 2e-5, LoRA |
| 环境 | A100, PyTorch 2.4, RCOD_O batch 1 约 28GB |
| 关键消融 | VPIM/DAS 有效性；$n=3$ 优于 $n=4$ |

### 核心洞察
1. 推理时手动控制不需要计算 $M_L$，直接指定 $t$；自动控制才依赖 MEM。
2. DAS 的价值是增强 Fid./Real. 两档分离度，VPIM 的价值是用 LR 视觉 token 替代不稳定文本条件。
3. $n=4$ 失败说明控制粒度受训练数据分布约束，RCOD 当前更像三档控制而非连续 slider。
4. 附录没有系统失败案例，尤其缺少文字/身份/结构敏感场景下的 hallucination 分析。

### 可追问点
- 为什么单步 SR 难以控制 realism？
- LDG 真正控制的是什么？
- 为什么 $n=4$ 的 Real. 档反而更差？
- VPIM 和 text prompt 的取舍是否依赖图像类型？
