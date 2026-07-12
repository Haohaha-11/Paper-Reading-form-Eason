[← 返回 README](../README.md)

# 7. Discussion and Conclusion

## 📌 预览

结论很短，但把全篇的"元观点"点破：DIS 领域没有银弹，所有方法都在 **速度 × 保真 × 精确性（是否真的采后验）** 的三角里权衡，选谁取决于应用约束。作者也坦白划了两条本文故意没覆盖的线（LDM 专用解法、diffusion bridges）。本节末尾附完整参考文献（原文逐字保留）。

---

In this chapter, we gave a comprehensive overview of using diffusion models for inverse problems, primarily focused on the general zero-shot solvers that does not involve task-specific training, and thus can be adapted to various applications. Our survey deliberately omitted two related areas to maintain this focus: solvers designed explicitly for Latent Diffusion Models (LDMs) (Rout et al. 2023, Rout, Chen, Kumar, Caramanis, Shakkottai & Chu 2024, Raphaeli et al. 2025)—whose underlying principles largely align with the methods discussed—and approaches based on diffusion bridges (Delbracio & Milanfar 2023, Luo et al. 2023, Liu, Vahdat, Huang, Theodorou, Nie & Anandkumar 2023, Chung, Kim & Ye 2023), which necessitate supervised training.

A key takeaway is the inherent trade-off among the surveyed methods. Solvers present a spectrum of design choices, balancing computational speed against reconstruction fidelity and exactness. The selection of an appropriate method is therefore contingent upon the specific constraints and goals of the target application.

The diversity of these powerful techniques signifies a rapidly maturing field. As these tools become more robust, they offer practitioners a versatile and adaptable toolkit for a wide range of scientific and creative applications. Future work will likely focus on reconciling the trade-offs between speed and accuracy, pushing the boundaries of what is achievable in unsupervised inverse problem-solving.

> 💡 **结论批读：作者亲手确认的"三角权衡"与本课题的缺口** (Hao 批注):
> - **三角权衡**: computational speed ↔ reconstruction fidelity ↔ **exactness**。这个 "exactness" 正是本课题最在意的维度——一个方法是否真在采后验（vs 塌向 MAP/MMSE）。全篇的隐性排序：explicit approximation（快、有偏）< decoupled/VI（中、较准）< SMC（慢、渐近精确）。
> - **作者的诚实**: 本文只覆盖 zero-shot（免任务训练）解法，故意跳过 LDM 专用解法和 diffusion bridges（后者需监督训练）。所以本文的"地图"是**免训练后验采样**这一支的地图，读时别把它当成全部 DIS。
> - **对本课题的最终落点**: 全篇 40+ 方法，(i) 无一给出严格的 $\nabla_{x_t}\log p(y|x_t)$——"数据一致性修正 ≠ 严格后验 score" 得到全景确认；(ii) 盲设置（Sec. 5.1）的联合后验方法全部近似化、且无一做 $\varphi$ 的 gauge 处理或 SBC/coverage 校准。future work 提到的"speed vs accuracy"没提"calibration"这第三条轴——**后验校准是这张地图上尚未标注的空白大陆**，正是本课题要填的。

---

## References

Aggarwal, H. K., Pramanik, A., John, M. & Jacob, M. (2022), ‘Ensure: A general approach for unsupervised training of deep image reconstruction algorithms’, IEEE transactions on medical imaging 42(4), 1133–1144.

Akiyama, K., Alberdi, A., Alef, W., Asada, K., Azulay, R., Baczko, A.-K., Ball, D., Balokovic, M., Barrett, J., Bintley, D. et al. (2019), ‘First m87 event horizon telescope results. iv. imaging the central supermassive black hole’, The Astrophysical Journal Letters 875(1), L4.

Alkhouri, I., Liang, S., Huang, C.-H., Dai, J., Qu, Q., Ravishankar, S. & Wang, R. (2024), ‘Sitcom: Step-wise triple-consistent diffusion sampling for inverse problems’, arXiv preprint arXiv:2410.04479 .

Anderson, B. D. (1982), ‘Reverse-time diffusion equation models’, Stochastic Processes and their Applications 12(3), 313–326.

Bai, W., Wang, Y., Chen, W. & Sun, H. (2024), ‘An expectation-maximization algorithm for training clean diffusion models from corrupted observations’, Advances in Neural Information Processing Systems 37, 19447–19471.

Barbano, R., Denker, A., Chung, H., Roh, T. H., Arridge, S., Maass, P., Jin, B. & Ye, J. C. (2025), ‘Steerable conditional diffusion for out-of-distribution adaptation in medical image reconstruction’, IEEE Transactions on Medical Imaging

Blau, Y. & Michaeli, T. (2018), The perception-distortion tradeoff, in ‘Proceedings of the IEEE conference on computer vision and pattern recognition’, pp. 6228–6237.

Bora, A., Jalal, A., Price, E. & Dimakis, A. G. (2017), Compressed sensing using generative models, in ‘International conference on machine learning’, PMLR, pp. 537–546.

Boyd, S., Parikh, N. & Chu, E. (2011), Distributed optimization and statistical learning via the alternating direction method of multipliers, Now Publishers Inc.

Cardoso, G., el idrissi, Y. J., Corff, S. L. & Moulines, E. (2024), Monte carlo guided denoising diffusion models for bayesian linear inverse problems., in ‘The Twelfth International Conference on Learning Representations’.

Chen, R. T. Q., Rubanova, Y., Bettencourt, J. & Duvenaud, D. K. (2018), Neural ordinary differential equations, in ‘Advances in Neural Information Processing Systems’, Vol. 31.

Chung, H., Kim, J., Kim, S. & Ye, J. C. (2023), ‘Parallel diffusion models of operator and image for blind inverse problems’, IEEE/CVF Conference on Computer Vision and Pattern Recognition .

Chung, H., Kim, J., Mccann, M. T., Klasky, M. L. & Ye, J. C. (2023), Diffusion posterior sampling for general noisy inverse problems, in ‘International Conference on Learning Representations’.

Chung, H., Kim, J. & Ye, J. C. (2023), ‘Direct diffusion bridge using data consistency for inverse problems’, Advances in Neural Information Processing Systems 36, 7158–7169.

Chung, H., Lee, D., Wu, Z., Kim, B.-H., Bouman, K. L. & Ye, J. C. (2025), ‘Contextmri: Enhancing compressed sensing mri through metadata conditioning’, arXiv preprint arXiv:2501.04284 .

Chung, H., Lee, S. & Ye, J. C. (2024), Decomposed diffusion sampler for accelerating large-scale inverse problems, in ‘The Twelfth International Conference on Learning Representations’.

Chung, H., Ryu, D., Mccann, M. T., Klasky, M. L. & Ye, J. C. (2023), ‘Solving 3d inverse problems using pre-trained 2d diffusion models’, IEEE/CVF Conference on Computer Vision and Pattern Recognition .

Chung, H., Sim, B., Ryu, D. & Ye, J. C. (2022), Improving diffusion models for inverse problems using manifold constraints, in ‘Advances in Neural Information Processing Systems’.

Chung, H., Sim, B. & Ye, J. C. (2022), Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction, in ‘Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition’.

Chung, H. & Ye, J. C. (2024), Deep diffusion image prior for efficient ood adaptation in 3d inverse problems, in ‘European Conference on Computer Vision’, Springer, pp. 432–455.

Chung, H., Ye, J. C., Milanfar, P. & Delbracio, M. (2024), Prompt-tuning latent diffusion models for inverse problems, in ‘Forty-first International Conference on Machine Learning’.

Daras, G., Chung, H., Lai, C.-H., Mitsufuji, Y., Ye, J. C., Milanfar, P., Dimakis, A. G. & Delbracio, M. (2024), ‘A survey on diffusion models for inverse problems’, arXiv preprint arXiv:2410.00083 .

Daras, G., Dagan, Y., Dimakis, A. & Daskalakis, C. (2023), ‘Consistent diffusion models: Mitigating sampling drift by learning to be consistent’, Advances in Neural Information Processing Systems 36, 42038–42063.

Daras, G., Dimakis, A. & Daskalakis, C. C. (2024), Consistent diffusion meets tweedie: Training exact ambient diffusion models with noisy data, in ‘Forty-first International Conference on Machine Learning’.

Daras, G., Rodriguez-Munoz, A., Klivans, A., Torralba, A. & Daskalakis, C. (2025), ‘Ambient diffusion omni: Training good models with bad data’, arXiv preprint arXiv:2506.10038 .

Daras, G., Shah, K., Dagan, Y., Gollakota, A., Dimakis, A. & Klivans, A. (2023), ‘Ambient diffusion: Learning clean distributions from corrupted data’, Advances in Neural Information Processing Systems 36, 288–313.

Delbracio, M. & Milanfar, P. (2023), ‘Inversion by direct iteration: An alternative to denoising diffusion for image restoration’, arXiv preprint arXiv:2303.11435 .

Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2017), Density estimation using real NVP, in ‘International Conference on Learning Representations’.

Dou, Z. & Song, Y. (2024), Diffusion posterior sampling for linear inverse problem solving: A filtering perspective, in ‘The Twelfth International Conference on Learning Representations’.

Efron, B. (2011), ‘Tweedie’s formula and selection bias’, Journal of the American Statistical Association 106(496), 1602–1614.

Eldar, Y. C. (2008), ‘Generalized sure for exponential families: Applications to regularization’, IEEE Transactions on Signal Processing 57(2), 471–481.

Erbach, J., Narnhofer, D., Dombos, A., Schiele, B., Lenssen, J. E. & Schindler, K. (2025), ‘Solving inverse problems with flair’, arXiv preprint arXiv:2506.02680 .

Feng, B. & Bouman, K. (2024), ‘Variational bayesian imaging with an efficient surrogate score-based prior’, Transactions on Machine Learning Research .

Feng, B. T., Smith, J., Rubinstein, M., Chang, H., Bouman, K. L. & Freeman, W. T. (2023), Score-based diffusion models as principled priors for inverse imaging, in ‘Proceedings of the IEEE/CVF International Conference on Computer Vision’, pp. 10520–10531.

Gao, R., Hoogeboom, E., Heek, J., Bortoli, V. D., Murphy, K. P. & Salimans, T. (2024), Diffusion meets flow matching: Two sides of the same coin.

Gupta, H., McCann, M. T., Donati, L. & Unser, M. (2021), ‘Cryogan: A new reconstruction paradigm for single-particle cryo-em via deep adversarial learning’, IEEE Transactions on Computational Imaging 7, 759–774.

He, Y., Murata, N., Lai, C.-H., Takida, Y., Uesaka, T., Kim, D., Liao, W.-H., Mitsufuji, Y., Kolter, J. Z., Salakhutdinov, R. & Ermon, S. (2024), Manifold preserving guided diffusion, in ‘The Twelfth International Conference on Learning Representations’.

Ho, J., Jain, A. & Abbeel, P. (2020), ‘Denoising diffusion probabilistic models’, Advances in Neural Information Processing Systems 33, 6840–6851.

Hu, J., Song, B., Fessler, J. A. & Shen, L. (2025), ‘Test-time adaptation improves inverse problem solving with patch-based diffusion models’, IEEE Transactions on Computational Imaging .

Hu, J., Song, B., Xu, X., Shen, L. & Fessler, J. A. (2024), ‘Learning image priors through patch-based diffusion models for solving inverse problems’, Advances in Neural Information Processing Systems 37, 1625–1660.

Huang, C.-W., Lim, J. H. & Courville, A. (2021), ‘A variational perspective on diffusion-based generative models and score matching’, arXiv preprint arXiv:2106.02808 .

Hyvärinen, A. & Dayan, P. (2005), ‘Estimation of non-normalized statistical models by score matching.’, Journal of Machine Learning Research 6(4).

Jalal, A., Arvinte, M., Daras, G., Price, E., Dimakis, A. G. & Tamir, J. (2021), ‘Robust compressed sensing mri with deep generative priors’, Advances in Neural Information Processing Systems 34.

Kadkhodaie, Z. & Simoncelli, E. P. (2021), Stochastic solutions for linear inverse problems using the prior implicit in a denoiser, in ‘Advances in Neural Information Processing Systems’.

Kawar, B., Elad, M., Ermon, S. & Song, J. (2022), Denoising diffusion restoration models, in ‘Advances in Neural Information Processing Systems’.

Kawar, B., Elata, N., Michaeli, T. & Elad, M. (2024), ‘GSURE-based diffusion model training with corrupted data’, Transactions on Machine Learning Research .

Kawar, B., Vaksman, G. & Elad, M. (2021), ‘Snips: Solving noisy inverse problems stochastically’, Advances in Neural Information Processing Systems 34, 21757–21769.

Kim, J., Kim, B. S. & Ye, J. C. (2025), ‘Flowdps: Flow-driven posterior sampling for inverse problems’, arXiv preprint arXiv:2503.08136 .

Kim, J., Park, G. Y., Chung, H. & Ye, J. C. (2025), Regularization by texts for latent diffusion inverse solvers, in ‘The Thirteenth International Conference on Learning Representations’.

Kim, K. & Ye, J. C. (2021), ‘Noise2Score: Tweedie’s Approach to Self-Supervised Image Denoising without Clean Images’, Advances in Neural Information Processing Systems 34.

Kingma, D. P. & Welling, M. (2013), ‘Auto-encoding variational bayes’, arXiv preprint arXiv:1312.6114 .

Laroche, C., Almansa, A. & Coupete, E. (2024), Fast diffusion em: a diffusion model for blind inverse problems with application to deconvolution, in ‘Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision’, pp. 5271–5281.

Lee, S., Chung, H., Park, M., Park, J., Ryu, W.-S. & Ye, J. C. (2023), Improving 3d imaging with pre-trained perpendicular 2d diffusion models, in ‘Proceedings of the IEEE/CVF International Conference on Computer Vision’, pp. 10710–10720.

Lee, S., Park, D., Kong, I. & Kim, H. J. (2024), Diffusion prior-based amortized variational inference for noisy inverse problems, in ‘European Conference on Computer Vision’, Springer, pp. 288–304.

Li, X., Kwon, S. M., Liang, S., Alkhouri, I. R., Ravishankar, S. & Qu, Q. (2024), ‘Decoupled data consistency with diffusion purification for image restoration’, arXiv preprint arXiv:2403.06054 .

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. & Le, M. (2023), Flow matching for generative modeling, in ‘The Eleventh International Conference on Learning Representations’.

Liu, G.-H., Vahdat, A., Huang, D.-A., Theodorou, E. A., Nie, W. & Anandkumar, A. (2023), ‘I2sb: Image-to-image schrödinger bridge’, arXiv preprint arXiv:2302.05872 .

Liu, X., Gong, C. & qiang liu (2023), Flow straight and fast: Learning to generate and transfer data with rectified flow, in ‘The Eleventh International Conference on Learning Representations’.

Luo, Z., Gustafsson, F. K., Zhao, Z., Sjölund, J. & Schön, T. B. (2023), ‘Image restoration with mean-reverting stochastic differential equations’, arXiv preprint arXiv:2301.11699 .

Mammadov, A., Chung, H. & Ye, J. C. (2024), ‘Amortized posterior sampling with diffusion prior distillation’, arXiv preprint arXiv:2407.17907 .

Man, S., Ohayon, G., Raphaeli, R. & Elad, M. (2025), ‘Proxies for distortion and consistency with applications for real-world image restoration’, arXiv preprint arXiv:2501.12102 .

Mardani, M., Song, J., Kautz, J. & Vahdat, A. (2023), ‘A variational perspective on solving inverse problems with diffusion models’, arXiv preprint arXiv:2305.04391 .

Murata, N., Saito, K., Lai, C.-H., Takida, Y., Uesaka, T., Mitsufuji, Y. & Ermon, S. (2023), Gibbsddrm: A partially collapsed gibbs sampler for solving blind inverse problems with denoising diffusion restoration, in ‘International conference on machine learning’, PMLR, pp. 25501–25522.

Park, B., Go, H., Nam, H., Kim, B.-H., Chung, H. & Kim, C. (2025), ‘Steerx: Creating any camera-free 3d and 4d scenes with geometric steering’, arXiv preprint arXiv:2503.12024 .

Patel, M., Wen, S., Metaxas, D. N. & Yang, Y. (2024), ‘Steering rectified flow models in the vector field for controlled image generation’, arXiv preprint arXiv:2412.00100 .

Peng, X., Zheng, Z., Dai, W., Xiao, N., Li, C., Zou, J. & Xiong, H. (2024), ‘Improving diffusion models for inverse problems using optimal posterior covariance’, arXiv preprint arXiv:2402.02149 .

Raphaeli, R., Man, S. & Elad, M. (2025), ‘Silo: Solving inverse problems with latent operators’, arXiv preprint arXiv:2501.11746 .

Romano, Y., Elad, M. & Milanfar, P. (2017), ‘The little engine that could: Regularization by denoising (red)’, SIAM journal on imaging sciences 10(4), 1804–1844.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P. & Ommer, B. (2022), High-resolution image synthesis with latent diffusion models, in ‘Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition’, pp. 10684–10695.

Rout, L., Chen, Y., Kumar, A., Caramanis, C., Shakkottai, S. & Chu, W.-S. (2024), Beyond first-order tweedie: Solving inverse problems using latent diffusion, in ‘Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition’, pp. 9472–9481.

Rout, L., Raoof, N., Daras, G., Caramanis, C., Dimakis, A. & Shakkottai, S. (2023), ‘Solving linear inverse problems provably via posterior sampling with latent diffusion models’, Advances in Neural Information Processing Systems 36, 49960–49990.

Rout, L., Raoof, N., Daras, G., Caramanis, C., Dimakis, A. & Shakkottai, S. (2024), ‘Solving linear inverse problems provably via posterior sampling with latent diffusion models’, Advances in Neural Information Processing Systems 36.

Rozet, F., Andry, G., Lanusse, F. & Louppe, G. (2024), ‘Learning diffusion priors from observations by expectation maximization’, Advances in Neural Information Processing Systems 37, 87647–87682.

Singhal, R., Horvitz, Z., Teehan, R., Ren, M., Yu, Z., McKeown, K. & Ranganath, R. (2025), ‘A general framework for inference-time scaling and steering of diffusion models’, arXiv preprint arXiv:2501.06848 .

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. & Ganguli, S. (2015), Deep unsupervised learning using nonequilibrium thermodynamics, in ‘International Conference on Machine Learning’, PMLR, pp. 2256–2265.

Song, J., Vahdat, A., Mardani, M. & Kautz, J. (2023), Pseudoinverse-guided diffusion models for inverse problems, in ‘International Conference on Learning Representations’.

Song, Y., Dhariwal, P., Chen, M. & Sutskever, I. (2023), Consistency models, in ‘Proceedings of the 40th International Conference on Machine Learning’, pp. 32211–32252.

Song, Y., Durkan, C., Murray, I. & Ermon, S. (2021), ‘Maximum likelihood training of score-based diffusion models’, Advances in Neural Information Processing Systems 34.

Song, Y. & Ermon, S. (2019), Generative modeling by estimating gradients of the data distribution, in ‘Advances in Neural Information Processing Systems’, Vol. 32.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S. & Poole, B. (2021), Score-based generative modeling through stochastic differential equations, in ‘9th International Conference on Learning Representations, ICLR’.

Stein, C. M. (1981), ‘Estimation of the mean of a multivariate normal distribution’, The annals of Statistics pp. 1135–1151.

Tarantola, A. (2005), Inverse problem theory and methods for model parameter estimation, SIAM.

Trippe, B. L., Yim, J., Tischer, D., Baker, D., Broderick, T., Barzilay, R. & Jaakkola, T. S. (2023), Diffusion probabilistic modeling of protein backbones in 3d for the motif-scaffolding problem, in ‘The Eleventh International Conference on Learning Representations’.

Ulyanov, D., Vedaldi, A. & Lempitsky, V. (2018), Deep image prior, in ‘Proceedings of the IEEE conference on computer vision and pattern recognition’, pp. 9446–9454.

Vahdat, A., Kreis, K. & Kautz, J. (2021), ‘Score-based generative modeling in latent space’, Advances in neural information processing systems 34, 11287–11302.

Venkatakrishnan, S. V., Bouman, C. A. & Willett, R. M. (2013), Plug-and-play priors for model based reconstruction, in ‘2013 IEEE Global Conference on Signal and Information Processing (GlobalSIP)’, IEEE, pp. 945–948.

Vincent, P. (2011), ‘A connection between score matching and denoising autoencoders’, Neural computation 23(7), 1661–1674.

Wu, H., He, L., Zhang, M., Chen, D., Luo, K., Luo, M., Zhou, J.-Z., Chen, H. & Lv, J. (2024), Diffusion posterior proximal sampling for image restoration, in ‘Proceedings of the 32nd ACM International Conference on Multimedia’, pp. 214–223.

Xu, T., Cai, X., Zhang, X., Ge, X., He, D., Sun, M., Liu, J., Zhang, Y.-Q., Li, J. & Wang, Y. (2025), ‘Rethinking diffusion posterior sampling: From conditional score estimator to maximizing a posterior’, arXiv preprint arXiv:2501.18913 .

Yang, L., Ding, S., Cai, Y., Yu, J., Wang, J. & Shi, Y. (2024), ‘Guidance with spherical gaussian constraint for conditional diffusion’, arXiv preprint arXiv:2402.03201 .

Zhang, B., Chu, W., Berner, J., Meng, C., Anandkumar, A. & Song, Y. (2025), Improving diffusion inverse problem solving with decoupled noise annealing, in ‘Proceedings of the Computer Vision and Pattern Recognition Conference’, pp. 20895–20905.

Zhang, K., Zuo, W., Chen, Y., Meng, D. & Zhang, L. (2017), ‘Beyond a gaussian denoiser: Residual learning of deep CNN for image denoising’, IEEE transactions on image processing 26(7), 3142–3155.

Zilberstein, N., Mardani, M. & Segarra, S. (2025), Repulsive latent score distillation for solving inverse problems, in ‘The Thirteenth International Conference on Learning Representations’.
