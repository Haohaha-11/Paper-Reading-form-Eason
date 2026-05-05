[← 返回 README](../README.md)

# Discussion

## 📌 预览
本文件合并 Conclusion/Appendix/References，重点看实现细节、局限、额外结果和可追踪文献。

---

# Conclusion

We propose RCOD, a framework that enhances one-step diffusion methods for Real-ISR through flexible realism control. RCOD employs latent grouping with degradationaware sampling during distillation and introduces a robust latent metric enabling denoising networks to assess degradation levels. Applied to two distinct one-step diffusion methods, RCOD achieves superior super-resolution performance across most FR and NR metrics while maintaining computational efficiency. We believe RCOD holds promise for diverse Real-ISR scenarios with varying requirements.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

Table S3: Efficiency comparison on an NVIDIA A100 GPU. The best results are highlighted in bold.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标。

![Table extracted](../images/ef55c7b28f74c0e25ab3a0355fa0fa990342d1797964bd3844f9df25f995aa09.jpg)
*Table extracted: Table extracted by MinerU. TOhal0 VPIM Y , training iteration N iz ← LRA;←RA ← parameters Yθ ← Yφ; random initialize trainable parameters MLPθ 2 for i ← 1 to N do 3 Sample low-resolution image x L and corres*

> 💡 **Table extracted 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

![Figure S6](../images/a9912a5174992b3fe3e14658665d54dfe002cca2fce7dafe937ab9dd63a8c1c6.jpg)
*Figure S6: Figure S6: Visual comparison $( \times 4 )$ of realism controls during $\operatorname { R C O D } _ { \operatorname { S } }$ inference stage. The best results are in bold.*

> 💡 **Figure S6 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

# Algorithm Details

Pseudo-code The pseudo-code of our RC-OSD training strategy is summarized as Algorithm 1. In inference stage, we do not need to compute $M _ { L }$ . Instead, we directly calculate $\hat { z } _ { H } = F _ { \theta } ( z _ { L } ; t , c _ { y } )$ , where $t$ a user-specified parameter that controls the level of realism for SR output. The choice of $t$ depends on the different application scenarios and is discussed in Subsection “Latent Domain Grouping”.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近结构，又要生成自然高频细节。

# References

Agustsson, E.; and Timofte, R. 2017. Ntire 2017 challenge on single image super-resolution: Dataset and study. In CVPRW, 126–135.

Cai, J.; Zeng, H.; Yong, H.; Cao, Z.; and Zhang, L. 2019. Toward real-world single image super-resolution: A new benchmark and a new model. In ICCV, 3086–3095.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标。

Chen, B.; Li, G.; Wu, R.; Zhang, X.; Chen, J.; Zhang, J.; and Zhang, L. 2025. Adversarial diffusion compression for real-world image super-resolution. In CVPR, 28208–28220.

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

Chen, J.; Pan, J.; and Dong, J. 2025. Faithdiff: Unleashing diffusion priors for faithful image super-resolution. In CVPR, 28188–28197.

Table S5: Ablation study of modules. The best results are highlighted in bold.

![Table S5](../images/7e16e77c629b4eef969d47b07d75e812f25f21806040cfd7aaa2acf946f53d47.jpg)
*Table S5: Table S5: Ablation study of modules. The best results are highlighted in bold.*

> 💡 **Table S5 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

![Table extracted](../images/f13efd3f309aa98cbf2da671df93877cc4f15592f142c2952553632d1ffe1b5e.jpg)
*Table extracted: Table extracted by MinerU. PSNR↑ DISTS↓ LPIPS↓ MANIQA↑ RCODo-Fid. (n = 4) 26.00 0.2111 0.2805 0.6490 RCODo-Fid. (n = 3) 26.01 0.2103 0.2796 0.6647 RCODo-Real. (n = 4) 24.02 0.2590 0.3588 0.6400 RCODo-Real. (*

> 💡 **Table extracted 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

Table S6: Ablation study of grouping strategy. The best results are highlighted in bold.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标。

Chen, T.; Kornblith, S.; Norouzi, M.; and Hinton, G. 2020. A simple framework for contrastive learning of visual representations. In ICML, 1597–1607.

Ding, K.; Ma, K.; Wang, S.; and Simoncelli, E. P. 2020. Image quality assessment: Unifying structure and texture similarity. IEEE PAMI, 44(5): 2567–2581.

> 💡 **批注**: 这里涉及条件信号：prompt 是否准确、是否退化感知，会影响生成细节与语义一致性。

Dong, C.; Loy, C. C.; He, K.; and Tang, X. 2015. Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence, 38(2): 295–307.

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

Dong, L.; Fan, Q.; Guo, Y.; Wang, Z.; Zhang, Q.; Chen, J.; Luo, Y.; and Zou, C. 2025. Tsd-sr: One-step diffusion with target score distillation for real-world image superresolution. In CVPR, 23174–23184.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. In NeurIPS.

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐。

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. Lora: Low-rank adaptation of large language models. In ICLR.

> 💡 **批注**: 这是 flow/ODE 视角：重点是 student 是否沿 teacher 的连续采样轨迹对齐。

Ignatov, A.; Kobyshev, N.; Timofte, R.; Vanhoey, K.; and Van Gool, L. 2017. Dslr-quality photos on mobile devices with deep convolutional networks. In ICCV.

Ji, X.; Cao, Y.; Tai, Y.; Wang, C.; Li, J.; and Huang, F. 2020. Real-world super-resolution via kernel estimation and noise injection. In CVPRW.

Karras, T.; Laine, S.; and Aila, T. 2019. A style-based generator architecture for generative adversarial networks. In CVPR.

Ke, J.; Wang, Q.; Wang, Y.; Milanfar, P.; and Yang, F. 2021. Musiq: Multi-scale image quality transformer. In ICCV. Kim, J.; Kwon Lee, J.; and Mu Lee, K. 2016. Accurate image super-resolution using very deep convolutional networks. In CVPR.   
Ledig, C.; Theis, L.; Huszar, F.; Caballero, J.; Cunningham, ´ A.; Acosta, A.; Aitken, A.; Tejani, A.; Totz, J.; Wang, Z.; et al. 2017. Photo-realistic single image super-resolution using a generative adversarial network. In CVPR, 4681–4690. Li, Y.; Zhang, K.; Liang, J.; Cao, J.; Liu, C.; Gong, R.; Zhang, Y.; Tang, H.; Liu, Y.; Demandolx, D.; et al. 2023. Lsdir: A large scale dataset for image restoration. In CVPR, 1775–1787.   
Liang, J.; Cao, J.; Sun, G.; Zhang, K.; Van Gool, L.; and Timofte, R. 2021. Swinir: Image restoration using swin transformer. In ICCV.   
Lin, X.; He, J.; Chen, Z.; Lyu, Z.; Fei, B.; Dai, B.; Ouyang, W.; Qiao, Y.; and Dong, C. 2023. Diffbir: Towards blind image restoration with generative diffusion prior. arXiv preprint arXiv:2308.15070.   
Liu, A.; Liu, Y.; Gu, J.; Qiao, Y.; and Dong, C. 2022. Blind image super-resolution: A survey and beyond. IEEE PAMI, 45(5): 5461–5480.   
Meng, C.; Rombach, R.; Gao, R.; Kingma, D.; Ermon, S.;

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

![Figure S9](../images/11f21785280c3c454b280825a03f9d821a8b3ce4e73a06352d5647639736e52f.jpg)
*Figure S9: Figure S9: Visualization of |Spearman coefficient| $\uparrow$ of different $M _ { L }$ distances and image quality metrics.*

> 💡 **Figure S9 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

Ho, J.; and Salimans, T. 2023. On distillation of guided diffusion models. In CVPR, 14297–14306.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

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

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

---

## 🔖 Section 总结

### 核心洞察
1. 总结可复用设计和局限。
2. References 可用于扩展本课题文献图谱。
