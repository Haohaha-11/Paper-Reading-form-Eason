[← 返回 README](../README.md)

# 6. Conclusion

## 📌 预览
本节收束三条贡献：spatially adaptive noise 激活区域先验，LQFM 补 VAE 压缩造成的信息损失，TMG 提供文本区域 grounding。结论本身不写局限，需要从前文消融和 trade-off 曲线反推后续问题。
---

> 💡 **Q&A 批注记录**:
> - Q: RGPA 和普通加噪有什么区别？
> - A: 普通 latent 加噪只调全局生成强度；RGPA 用 LQ 梯度构造 spatially adaptive noise，把生成先验更多分配给纹理区域，并尽量保护平坦/结构区域。


We present CODSR, a diffusion-based one-step SR framework in order to achieve a favorable balance between structural fidelity and perceptual quality. We introduce spatially adaptive noise to the LQ latent, enabling regiondiscriminative activation of generative priors. We further develop a lightweight LQ-guided feature modulation module to preserve information from the LQ image and mitigate the LQ information loss caused by compression encoding, thus enhancing fidelity without compromising generative capability. In addition, we propose a text-matching guidance strategy to provide explicit spatial guidance for text-interactive regions, achieving precise semantic grounding. Extensive experiments on benchmarks demonstrate that CODSR outperforms state-of-the-art methods in terms of structural fidelity and visual quality.
> 💡 **结论批读**: 结论里的“balance”不是抽象口号，而是由三个约束共同实现：RGPA 控制生成先验释放位置，LQFM 把结构保真拉回 denoising，TMG 把文本语义落到正确区域。真正未解决的是这些控制仍依赖 Sobel、prompt/mask 和经验 t_s。


# References

[1] Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In CVPRW, 2017. 5   
[2] Yuang Ai, Xiaoqiang Zhou, Huaibo Huang, Xiaotian Han, Zhengyu Chen, Quanzeng You, and Hongxia Yang. Dreamclear: High-capacity real-world image restoration with privacy-safe dataset curation. NeurIPS, 37:55443–55469, 2024. 6, 7   
[3] Steven Bird. Nltk: the natural language toolkit. In Proceedings of the COLING/ACL 2006 interactive presentation sessions, pages 69–72, 2006. 4   
[4] Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, and Lei Zhang. Toward real-world single image super-resolution: A new benchmark and a new model. In ICCV, 2019. 5, 6, 7   
[5] Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, and Wen Gao. Pre-trained image processing transformer. In CVPR, 2021. 1, 2   
[6] Junyang Chen, Jinshan Pan, and Jiangxin Dong. Faithdiff: Unleashing diffusion priors for faithful image superresolution. In CVPR, 2025. 1, 5, 6, 7   
[7] Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, and Chao Dong. Activating more pixels in image superresolution transformer. In CVPR, 2023. 1, 2   
[8] Zheng Chen, Yulun Zhang, Jinjin Gu, Linghe Kong, Xiaokang Yang, and Fisher Yu. Dual aggregation transformer for image super-resolution. In ICCV, 2023. 1, 2   
[9] Tao Dai, Jianrui Cai, Yongbing Zhang, Shu-Tao Xia, and Lei Zhang. Second-order attention network for single image super-resolution. In CVPR, 2019. 1, 2   
[10] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 2   
[11] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE TPAMI, 2020. 6   
[12] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Learning a deep convolutional network for image super-resolution. In ECCV, 2014. 1, 2   
[13] Linwei Dong, Qingnan Fan, Yihong Guo, Zhonghao Wang, Qi Zhang, Jinwei Chen, Yawei Luo, and Changqing Zou. Tsd-sr: One-step diffusion with target score distillation for real-world image super-resolution. In CVPR, 2025. 1, 3   
[14] Zheng-Peng Duan, Jiawei Zhang, Xin Jin, Ziheng Zhang, Zheng Xiong, Dongqing Zou, Jimmy S Ren, Chunle Guo, and Chongyi Li. Dit4sr: Taming diffusion transformer for real-world image super-resolution. In ICCV, 2025. 1   
[15] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS, 2014. 2   
[16] Shuhang Gu, Andreas Lugmayr, Martin Danelljan, Manuel Fritsche, Julien Lamour, and Radu Timofte. Div8k: Diverse 8k resolution image dataset. In ICCVW, 2019. 5   
[17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2   
[18] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In ICLR, 2022. 6   
[19] Dongzhi Jiang, Guanglu Song, Xiaoshi Wu, Renrui Zhang, Dazhong Shen, Zhuofan Zong, Yu Liu, and Hongsheng Li. Comat: Aligning text-to-image diffusion model with imageto-text concept matching. In NeurIPS, 2024. 4   
[20] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019. 5   
[21] Bahjat Kawar, Gregory Vaksman, and Michael Elad. Snips: Solving noisy inverse problems stochastically. In NeurIPS, 2021. 2   
[22] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. In NeurIPS, 2022. 2   
[23] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In ICCV, 2021. 6   
[24] Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, ´ Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, et al. Photorealistic single image super-resolution using a generative adversarial network. In CVPR, 2017. 1, 2   
[25] Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, et al. Lsdir: A large scale dataset for image restoration. In CVPR, 2023. 5   
[26] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In ICCV, 2021. 1, 2   
[27] Jie Liang, Hui Zeng, and Lei Zhang. Efficient and degradation-adaptive network for real-world image superresolution. In ECCV, 2022. 2   
[28] Jie Liang, Hui Zeng, and Lei Zhang. Details or artifacts: A locally discriminative learning approach to realistic image super-resolution. In CVPR, 2022.   
[29] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. Enhanced deep residual networks for single image super-resolution. In CVPRW, 2017. 5   
[30] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Yu Qiao, Wanli Ouyang, and Chao Dong. Diffbir: Toward blind image restoration with generative diffusion prior. In ECCV, 2024. 1, 2, 5, 6, 7   
[31] Xinqi Lin, Fanghua Yu, Jinfan Hu, Zhiyuan You, Wu Shi, Jimmy S Ren, Jinjin Gu, and Chao Dong. Harnessing diffusion-yielded score priors for image restoration. arXiv preprint arXiv:2507.20590, 2025. 3, 5, 6, 7   
[32] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6   
[33] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In AAAI, 2024. 2   
[34] IDEA Research. Grounded sam 2: Ground and track anything in videos. https://github.com/IDEA-Research/Grounded-SAM-2, 2024. 2, 3, 4 [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn- ¨ thesis with latent diffusion models. In CVPR, 2022. 1 [36] Yukai Shi, Hao Li, Sen Zhang, Zhijing Yang, and Xiao Wang. Criteria comparative learning for real-scene image super-resolution. IEEE TCSVT, 2022. 1, 2 [37] Stability.ai. Sd, 2021. https://stability.ai/stablediffusion. 2, 6 [38] Long Sun, Jinshan Pan, and Jinhui Tang. Shufflemixer: An efficient convnet for image super-resolution. In NeurIPS,   
2022. 1, 2 [39] Long Sun, Jiangxin Dong, Jinhui Tang, and Jinshan Pan. Spatially-adaptive feature modulation for efficient image super-resolution. In ICCV, 2023. 1, 2 [40] Lingchen Sun, Rongyuan Wu, Zhiyuan Ma, Shuaizheng Liu, Qiaosi Yi, and Lei Zhang. Pixel-level and semantic-level adjustable super-resolution: A dual-lora approach. In CVPR,   
2025. 1, 3, 4, 5, 6, 7 [41] Raphael Tang, Linqing Liu, Akshat Pandey, Zhiying Jiang, Gefei Yang, Karun Kumar, Pontus Stenetorp, Jimmy Lin, and Ferhan Ture. What the daam: Interpreting stable diffusion using cross attention. arXiv preprint arXiv:2210.04885,   
2022. 8 [42] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In AAAI, 2023. 6 [43] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. IJCV, 2024. 1, 2 [44] Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao, and Chen Change Loy. Esrgan: Enhanced super-resolution generative adversarial networks. In ECCVW, 2018. 2 [45] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In ICCV, 2021. 1, 2, 5, 6, 7 [46] Yufei Wang, Wenhan Yang, Xinyuan Chen, Yaohui Wang, Lanqing Guo, Lap-Pui Chau, Ziwei Liu, Yu Qiao, Alex C Kot, and Bihan Wen. Sinsr: diffusion-based image superresolution in a single step. In CVPR, 2024. 2, 5, 6, 7 [47] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 2004. 6 [48] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In NeurIPS, 2023. 3 [49] Pengxu Wei, Ziwei Xie, Hannan Lu, Zongyuan Zhan, Qixiang Ye, Wangmeng Zuo, and Liang Lin. Component divide-and-conquer for real-world image super-resolution. In ECCV, 2020. 1, 5, 7, 8 [50] Rongyuan Wu, Lingchen Sun, Zhiyuan Ma, and Lei Zhang. One-step effective diffusion network for real-world image super-resolution. In NeurIPS, 2024. 1, 2, 3, 4, 5, 6, 7, 8 [51] Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. Seesr: Towards semantics-aware real-world image super-resolution. In CVPR, 2024. 2, 5, 6,   
[52] Rui Xie, Chen Zhao, Kai Zhang, Zhenyu Zhang, Jun Zhou, Jian Yang, and Ying Tai. Addsr: Accelerating diffusionbased blind super-resolution with adversarial diffusion distillation. arXiv preprint arXiv:2404.01717, 2024. 2   
[53] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In CVPR, 2022. 6   
[54] Tao Yang, Rongyuan Wu, Peiran Ren, Xuansong Xie, and Lei Zhang. Pixel-aware stable diffusion for realistic image super-resolution and personalized stylization. In ECCV, 2024. 2   
[55] Qiaosi Yi, Shuai Li, Rongyuan Wu, Lingchen Sun, Yuhui Wu, and Lei Zhang. Fine-structure preserved real-world image super-resolution via transfer vae training. arXiv preprint arXiv:2507.20291, 2025. 1, 2, 3, 5, 6, 7   
[56] Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photorealistic image restoration in the wild. In CVPR, 2024. 1, 5, 6, 7   
[57] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Resshift: Efficient diffusion model for image superresolution by residual shifting. In NeurIPS, 2023. 2   
[58] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In CVPR, 2022. 1, 2   
[59] Aiping Zhang, Zongsheng Yue, Renjing Pei, Wenqi Ren, and Xiaochun Cao. Degradation-guided one-step image super-resolution with diffusion priors. arXiv preprint arXiv:2409.17058, 2024. 2, 4, 6   
[60] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. In ICCV, 2021. 1, 2, 5   
[61] Lin Zhang, Lei Zhang, and Alan C Bovik. A feature-enriched completely blind image quality evaluator. IEEE TIP, 2015. 6   
[62] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2   
[63] Leheng Zhang, Weiyi You, Kexuan Shi, and Shuhang Gu. Uncertainty-guided perturbation for image super-resolution diffusion model. In CVPR, 2025. 4   
[64] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 4, 6   
[65] Xindong Zhang, Hui Zeng, Shi Guo, and Lei Zhang. Efficient long-range attention network for image superresolution. In ECCV, 2022. 1, 2   
[66] Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu. Image super-resolution using very deep residual channel attention networks. In ECCV, 2018. 1, 2   
[67] Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, and Yun Fu. Residual dense network for image super-resolution. In CVPR, 2018. 1, 2
> 💡 **参考文献批读**: References 里能看到 CODSR 的知识来源：OSEDiff/VSD 提供 one-step 蒸馏范式，PiSA-SR 提供 dual-LoRA 思路，UPSR 启发 adaptive perturbation，Grounded-SAM2/CoMat 支撑 TMG 的区域语义约束。


[68] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. In CVPR, 2024. 4, 6
> 💡 **工具链风险**: RAM/DAPE/Grounded-SAM2/NLTK 都出现在方法链路里，说明 CODSR 的语义对齐不是端到端自足能力。部署时 prompt 抽取或分割失败，可能直接影响最终纹理生成。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 本节作用 | 收束贡献，并从未明说处反推限制 |
| 主线 | one-step SR 中用区域噪声、LQ 调制、文本区域约束来桥接 fidelity 与 realism |
| 后续关注 | t_s 自适应、RGPA 权重学习、TMG 语义链路鲁棒性、结构敏感评估 |

### 核心洞察
1. CODSR 的贡献可以概括为“局部释放 prior，同时局部约束 prior”，这是它区别于全图 one-step SR 的地方。
2. 结论没有展开失败模式，但 Figure 6 已经说明 fidelity-realism 不可能消失，只是变得可调。
3. 后续工作最自然的是让 t_s 和 RGPA 权重从图像内容自适应学习，而不是依赖默认经验值。

### 可追问点
- RGPA 和普通加噪有什么区别？
- 为什么还要 LQFM？
- TMG 的外部分割和 prompt 链路失败时，模型是否有 fallback？
