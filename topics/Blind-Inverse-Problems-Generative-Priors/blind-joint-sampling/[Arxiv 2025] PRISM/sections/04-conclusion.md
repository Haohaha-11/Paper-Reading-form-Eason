[← 返回 README](../README.md)

# 4. Conclusion

## 📌 预览

结论回收全文：PRISM = split Gibbs sampling + 四步更新（条件核先验步、图像似然步、图像先验步、核似然步），在 FFHQ 盲去模糊上图像与核恢复都超 SOTA，且对随机初始化鲁棒、能做 UQ。References 附在本节末尾。

---

In this work, we introduced PRISM as a novel method for solving blind inverse problems with diffusion models. The proposed method is based on split Gibbs sampling, and the resulting algorithm consists of four sampling steps: a measurement-conditioned kernel prior step, an image likelihood step, an image prior step, and a kernel likelihood step. The likelihood steps involve closed-form updates that are readily computed, while the image prior step employs a pre-trained image diffusion model and the kernel prior step uses a measurement-conditioned kernel diffusion model. We empirically validated the effectiveness of PRISM on the blind deblurring task using the FFHQ dataset. Experimental results show that PRISM offers improved recovery of both the image and blur kernel over existing state-of-the-art methods. Furthermore, PRISM demonstrates strong robustness to initialization. Stable convergence to a high-quality image solution is observed even with fully random initialization. Additional experiments on UQ further corroborate PRISM's capability as a sampling method to generate reliable samples for both the image and kernel.

> 💡 **结论批读：三条 claim 的兑现程度** (Hao 批注): 结论声明了三点，逐条对照证据看含金量：
> - **重建更好**（image+kernel 超 SOTA）：证据扎实——Table 1/2 五项指标全占优，Fig.1 视觉支撑。✔
> - **对初始化鲁棒**（随机初始化也稳定收敛）：证据是 Fig.2 的 conditional vs unconditional 对比，逻辑清楚但只在一个任务、只看 PSNR/SSIM 曲线。✔（有限）
> - **UQ 生成可靠样本**：这是 claim 最"软"的一条。正文用了"competitive"和"corroborate"这类保守措辞，实际 Table 3 显示 $\sigma=0.02$ 下 NLL 输给 BlindDPS。所以准确说法是"**能做定性 UQ**"，而非"校准可靠"。⚠
>
> **对本课题的定位小结**：PRISM 是我们最直接的竞品——同样联合估 $(x,\varphi)$、同样自称后验采样、同样报告像素 SD/NLL/覆盖。但它 (1) 不估 $\sigma$；(2) 没有 gauge 处理；(3) 缺 SBC/coverage 曲线/CRPS 这类严格校准检验；(4) 只验证单一任务。我们的贡献点应锁定"**联合估计 $x,\varphi,\sigma$ + gauge-aware + 系统性校准诊断**"，并在与 PRISM 同等的 FFHQ 去模糊设置上正面比校准。

---

## References

[1] K. P. Pruessmann, M. Weiger, M. B. Scheidegger, and P. Boesiger, "SENSE: Sensitivity encoding for fast MRI," Magn. Reson. Med., vol. 42, no. 5, pp. 952–962, Nov. 1999.

[2] Y. Hu, W. Gan, C. Ying, T. Wang, C. Eldeniz, J. Liu, Y. Chen, H. An, and U. S. Kamilov, "Spicer: Self-supervised learning for mri with automatic coil sensitivity estimation and reconstruction," Magnetic resonance in medicine, vol. 92, no. 3, pp. 1048–1063, 2024.

[3] S. Basu and Y. Bresler, "Uniqueness of tomography with unknown view angles," IEEE transactions on image processing, vol. 9, no. 6, pp. 1094–1106, 2000.

[4] M. Xie, J. Liu, Y. Sun, W. Gan, B. Wohlberg, and U. S. Kamilov, "Joint reconstruction and calibration using regularization by denoising with application to computed tomography," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 4028–4037.

[5] R. Fergus, B. Singh, A. Hertzmann, S. T. Roweis, and W. T. Freeman, "Removing camera shake from a single photograph," in ACM SIGGRAPH 2006 Papers, ser. SIGGRAPH '06. New York, NY, USA: Association for Computing Machinery, 2006, p. 787–794.

[6] L. Chen, F. Fang, T. Wang, and G. Zhang, "Blind image deblurring with local maximum gradient prior," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 1742–1750.

[7] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, and J. C. Ye, "Diffusion posterior sampling for general noisy inverse problems," in International Conference on Learning Representations, 2023.

[8] B. Kawar, M. Elad, S. Ermon, and J. Song, "Denoising diffusion restoration models," in Advances in Neural Information Processing Systems, 2022.

[9] Y. Wang, J. Yu, and J. Zhang, "Zero-shot image restoration using denoising diffusion null-space model," The Eleventh International Conference on Learning Representations, 2023.

[10] H. Chung, J. Kim, S. Kim, and J. C. Ye, "Parallel diffusion models of operator and image for blind inverse problems," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 6059–6069.

[11] N. Murata, K. Saito, C.-H. Lai, Y. Takida, T. Uesaka, Y. Mitsufuji, and S. Ermon, "Gibbsddrm: A partially collapsed gibbs sampler for solving blind inverse problems with denoising diffusion restoration," in International conference on machine learning. PMLR, 2023, pp. 25 501–25 522.

[12] Y. Sanghvi, Y. Chi, and S. H. Chan, "Kernel diffusion: An alternate approach to blind deconvolution," in European Conference on Computer Vision. Springer, 2024, pp. 1–20.

[13] Z. Wu, Y. Sun, Y. Chen, B. Zhang, Y. Yue, and K. L. Bouman, "Principled probabilistic imaging using diffusion models as plug-and-play priors," in Advances in Neural Information Processing Systems, 2024.

[14] A. Li, W. Gan, and U. S. Kamilov, "Plug-and-play posterior sampling for blind inverse problems," arXiv preprint arXiv:2505.22923, 2025.

[15] M. Vono, N. Dobigeon, and P. Chainais, "Split-and-augmented gibbs sampler—application to large-scale inference problems," IEEE Transactions on Signal Processing, vol. 67, no. 6, pp. 1648–1661, 2019.

[16] D. Geman and C. Yang, "Nonlinear image recovery with half-quadratic regularization," IEEE transactions on Image Processing, vol. 4, no. 7, pp. 932–946, 1995.

[17] S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein et al., "Distributed optimization and statistical learning via the alternating direction method of multipliers," Foundations and Trends in Machine learning, vol. 3, no. 1, pp. 1–122, 2011.

[18] Y. Wang, W. Yin, and J. Zeng, "Global convergence of admm in nonconvex nonsmooth optimization," Journal of Scientific Computing, vol. 78, no. 1, pp. 29–63, 2019.

[19] R. Laumont, V. D. Bortoli, A. Almansa, J. Delon, A. Durmus, and M. Pereyra, "Bayesian imaging using plug & play priors: When langevin meets tweedie," SIAM J. Imaging Sci., vol. 15, no. 2, pp. 701–737, 2022.

[20] Y. Sun, Z. Wu, Y. Chen, B. T. Feng, and K. L. Bouman, "Provable probabilistic imaging using score-based generative priors," IEEE Transactions on Computational Imaging, vol. 10, pp. 1290–1305, 2024.

[21] T. Karras, S. Laine, and T. Aila, "A style-based generator architecture for generative adversarial networks," in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 4396–4405.

[22] C. Saharia, J. Ho, W. Chan, T. Salimans, D. J. Fleet, and M. Norouzi, "Image super-resolution via iterative refinement," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 4, pp. 4713–4726, 2023.

[23] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," Advances in neural information processing systems, vol. 30, 2017.

> 💡 **参考文献批读：谱系定位** (Hao 批注): 三个直接 baseline 的原始出处——BlindDPS [10]、GibbsDDRM [11]、Kernel-Diff [12]；方法底座 PnP-DM [13]（Wu et al., NeurIPS 2024，split Gibbs 即插即用扩散后验采样，PRISM 的直接前身）；最近邻竞品 Blind-PnPDM [14]（Li et al., arXiv 2505.22923, 2025）。核扩散架构借自 SR3 [22]。UQ 的 NLL 指标出处是 Deep Ensembles [23]（Lakshminarayanan et al., NeurIPS 2017）——注意 PRISM 的 UQ 评价工具是 2017 年的 NLL，而非近年更严格的校准工具（SBC/PIT/CRPS），这从文献选择上也印证了它校准检验的保守。
