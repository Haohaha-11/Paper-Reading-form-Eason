[← 返回 README](../README.md)

# 9. Authors and Contributors + References（Appendix）

## 📌 预览

本节归并论文末尾的**作者与贡献者名单**(原文第 9 节)与**参考文献**。StarVLA 采用"核心作者 + 社区贡献者"双层贡献者模式,强调这是一个持续维护、开放共建的开源项目。参考文献完整逐字保留,便于溯源本文引用的 VLA / world model / benchmark 工作。

---

## 9. Authors and Contributors for StarVLA v1.0

StarVLA thrives on the synergy between its dedicated core team and a vibrant open-source community. To accurately reflect the nature of involvement, we list contributors in two categories: Authors and Community Contributors. We extend our deepest gratitude to everyone who has helped shape and scale StarVLA.

Authors. Jinhui Ye, Ning Gao, Yilun Chen<sup>†</sup>, Weiyu Guo, Zixuan Wang, Yuxing Chen, Fangjing Wang, Senqiao Yang, Chengyao Wang, Yuqi Liu, Meng Chu, Changsheng Lu, Pengguang Chen, Shu Liu<sup>†</sup>, Jiaya Jia<sup>†∗</sup>

Community Contributors. Junqiu Yu, Shuang Zeng, Shijie Lian, Hanwen Wan, Changjiu Zhang, Zhijie Song, Mingsheng Li, Qiuyue Wang, Sicheng Xie, Jinliang Zheng, Deyu Zhou, Jiaming Zhou, Lu Dai, Xiaorui Zhao

Contributor Policy. Authors constitute the core team of StarVLA. This group is responsible for continuously iterating on core features, maintaining the foundational framework, and providing ongoing, long-term support for the project. Researchers and developers who are interested in making sustained, structural contributions and wish to join the core author team are highly encouraged to contact us. Community Contributors are the vital force behind the project’s broader ecosystem. We continuously receive invaluable support from the open-source community—ranging from new feature implementations (pull requests) and bug fixes to constructive feedback. We deeply appreciate these efforts, which allow StarVLA to evolve rapidly. The full and actively updated contributor history is maintained at starvla.github.io/contributors.

<sup>†</sup>Corresponding authors. ∗ Von Neumann Institute, HKUST

> 💡 **机制拆解** (claude 批注): 双层贡献者模式(Authors = 核心团队,负责核心功能迭代/框架维护/长期支持;Community Contributors = 生态贡献者,提 PR/修 bug/给反馈)不是客套,而是**开源可持续性的治理设计**。它明确了"谁对框架长期负责"与"谁可以低门槛参与",并主动招募愿意做结构性贡献的人加入核心团队。这与全文"降低门槛、开放共建"的平台定位一致——论文本身就是一个持续演进的项目报告(摘要也说"we will update this report as the project evolves")。通讯作者为 Yilun Chen、Shu Liu、Jiaya Jia(HKUST / Von Neumann Institute)。

---

## References

> 💡 **参考文献批读** (claude 批注): 引用结构清晰反映本文的三条技术根系,读参考文献时可按此归类:(1) **VLA 方法**——OpenVLA(Kim 2024/2025)、π0(Black 2024)、π0-FAST(Pertsch 2025)、GR00T(Bjorck 2025)、RT-1/RT-2(Brohan 2022/2023)、CogACT、SpatialVLA、X-VLA 等,是四大 head 与对比 baseline 的来源;(2) **World model / 视频建模**——Cosmos policy(Kim 2026)、V-JEPA2(Assran 2025)、DreamGen(Jang 2025)、GigaWorld/GigaBrain 等,支撑 world-model backbone 路线;(3) **Benchmark**——LIBERO(Liu 2024a)、LIBERO-Plus(Fei 2025)、SimplerEnv(Li 2024b)、RoboCasa(Nasiriany 2024)、RoboTwin 2.0(Chen 2025b)、BEHAVIOR-1K(Li 2023)、CALVIN(Mees 2022)。特别注意 ST4VLA(Ye et al. 2026a)——第 6 节 co-training 案例的原始出处。以下为原文完整参考文献,逐字保留。

1X World Model Team (2025). 1x world model: Evaluating bits, not atoms. Supplementary technical progress report. Contributed by Daniel Ho, Jack Monas, Juntao Ren, Christina Yu.

AgiBot (2025). Agibot official website. https://www.agibot.com/.

Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Komeili, M., Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., Arnaud, S., Gejji, A., Martin, A., Hogan, F. R., Dugas, D., Bojanowski, P., Khalidov, V., Labatut, P., Massa, F., Szafraniec, M., Krishnakumar, K., Li, Y., Ma, X., Chandar, S., Meier, F., LeCun, Y., Rabbat, M., and Ballas, N. (2025). V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985.

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al. (2025a). Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M.-H., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. (2025b). Qwen2.5-VL technical report. CoRR, abs/2502.13923.

Bjorck, J., Castañeda, F., Cherniadev, N., Da, X., Ding, R., Fan, L., Fang, Y., Fox, D., Hu, F., Huang, S., et al. (2025). Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734.

Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al. (2024). pi<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., Ding, T., Driess, D., Dubey, A., Finn, C., et al. (2023). Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al. (2022). Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817.

Cai, J., Cai, Z., Cao, J., Chen, Y., He, Z., Jiang, L., Li, H., Li, H., Li, Y., Liu, Y., et al. (2026). Internvla-a1: Unifying understanding, generation and action for robotic manipulation. arXiv preprint arXiv:2601.02456.

Chen, B., Zhang, T., Geng, H., Song, K., Zhang, C., Li, P., Freeman, W. T., Malik, J., Abbeel, P., Tedrake, R., Sitzmann, V., and Du, Y. (2025a). Large video planner enables generalizable robot control. arXiv preprint arXiv:2512.15840.

Chen, D., Zhang, J., Mu, T., Tan, Q., Li, Y., Mao, J., Liu, X., Li, K., Qiao, Y., Xiao, F., Ling, Z., and Su, H. (2025b). Robotwin 2.0: Towards general robot policies with active data generation. arXiv preprint arXiv:2504.13059.

Chen, X., Chen, Y., Fu, Y., Gao, N., Jia, J., Jin, W., Li, H., Mu, Y., Pang, J., Qiao, Y., et al. (2025c). Internvla-m1: A spatially guided vision-language-action framework for generalist robot policy. arXiv preprint arXiv:2510.13778.

Chen, X., Djolonga, J., Padlewski, P., Mustafa, B., Changpinyo, S., Wu, J., Riquelme Ruiz, C., Goodman, S., Wang, X., Tay, Y., Shakeri, S., Dehghani, M., Salz, D., Lucic, M., Tschannen, M., Nagrani, A., Hu, H., Joshi, M., Pang, B., Montgomery, C., Pietrzyk, P., Ritter, M., Piergiovanni, A., Minderer, M., Pavetic, F., Waters, A., Li, G., Alabdulmohsin, I., Beyer, L., Amelot, J., Lee, K., Steiner, A. P., Li, Y., Keysers, D., Arnab, A., Xu, Y., Rong, K., Kolesnikov, A., Seyedhosseini, M., Angelova, A., Zhai, X., Houlsby, N., and Soricut, R. (2023). PaLI-X: On scaling up a multilingual vision and language model.

Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., and Song, S. (2024a). Diffusion policy: Visuomotor policy learning via action diffusion.

Chi, C., Xu, Z., Pan, C., Cousineau, E., Burchfiel, B., Feng, S., Tedrake, R., and Song, S. (2024b). Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. In Proceedings of Robotics: Science and Systems (RSS).

Collaboration, O. X.-E., O’Neill, A., Rehman, A., Gupta, A., Maddukuri, A., Gupta, A., Padalkar, A., Lee, A., Pooley, A., Gupta, A., Mandlekar, A., Jain, A., Tung, A., Bewley, A., Herzog, A., Irpan, A., Khazatsky, A., Rai, A., Gupta, A., Wang, A., Kolobov, A., Singh, A., Garg, A., Kembhavi, A., Xie, A., Brohan, A., Raffin, A., Sharma, A., Yavary, A., Jain, A., Balakrishna, A., Wahid, A., Burgess-Limerick, B., Kim, B., Schölkopf, B., Wulfe, B., Ichter, B., Lu, C., Xu, C., Le, C., Finn, C., Wang, C., Xu, C., Chi, C., Huang, C., Chan, C., Agia, C., Pan, C., Fu, C., Devin, C., Xu, D., Morton, D., Driess, D., Chen, D., Pathak, D., Shah, D., Büchler, D., Jayaraman, D., Kalashnikov, D., Sadigh, D., Johns, E., Foster, E., Liu, F., Ceola, F., Xia, F., Zhao, F., Frujeri, F. V., Stulp, F., Zhou, G., Sukhatme, G. S., Salhotra, G., Yan, G., Feng, G., Schiavi, G., Berseth, G., Kahn, G., Yang, G., Wang, G., Su, H., Fang, H.-S., Shi, H., Bao, H., Amor, H. B., Christensen, H. I., Furuta, H., Bharadhwaj, H., Walke, H., Fang, H., Ha, H., Mordatch, I., Radosavovic, I., Leal, I., Liang, J., Abou-Chakra, J., Kim, J., Drake, J., Peters, J., Schneider, J., Hsu, J., Vakil, J., Bohg, J., Bingham, J., Wu, J., Gao, J., Hu, J., Wu, J., Wu, J., Sun, J., Luo, J., Gu, J., Tan, J., Oh, J., Wu, J., Lu, J., Yang, J., Malik, J., Silvério, J., Hejna, J., Booher, J., Tompson, J., Yang, J., Salvador, J., Lim, J. J., Han, J., Wang, K., Rao, K., Pertsch, K., Hausman, K., Go, K., Gopalakrishnan, K., Goldberg, K., Byrne, K., Oslund, K., Kawaharazuka, K., Black, K., Lin, K., Zhang, K., Ehsani, K., Lekkala, K., Ellis, K., Rana, K., Srinivasan, K., Fang, K., Singh, K. P., Zeng, K.-H., Hatch, K., Hsu, K., Itti, L., Chen, L. Y., Pinto, L., Fei-Fei, L., Tan, L., Fan, L. J., Ott, L., Lee, L., Weihs, L., Chen, M., Lepert, M., Memmel, M., Tomizuka, M., Itkina, M., Castro, M. G., Spero, M., Du, M., Ahn, M., Yip, M. C., Zhang, M., Ding, M., Heo, M., Srirama, M. K., Sharma, M., Kim, M. J., Kanazawa, N., Hansen, N., Heess, N., Joshi, N. J., Suenderhauf, N., Liu, N., Palo, N. D., Shafiullah, N. M. M., Mees, O., Kroemer, O., Bastani, O., Sanketi, P. R., Miller, P. T., Yin, P., Wohlhart, P., Xu, P., Fagan, P. D., Mitrano, P., Sermanet, P., Abbeel, P., Sundaresan, P., Chen, Q., Vuong, Q., Rafailov, R., Tian, R., Doshi, R., Mart’in-Mart’in, R., Baijal, R., Scalise, R., Hendrix, R., Lin, R., Qian, R., Zhang, R., Mendonca, R., Shah, R., Hoque, R., Julian, R., Bustamante, S., Kirmani, S., Levine, S., Lin, S., Moore, S., Bahl, S., Dass, S., Sonawani, S., Tulsiani, S., Song, S., Xu, S., Haldar, S., Karamcheti, S., Adebola, S., Guist, S., Nasiriany, S., Schaal, S., Welker, S., Tian, S., Ramamoorthy, S., Dasari, S., Belkhale, S., Park, S., Nair, S., Mirchandani, S., Osa, T., Gupta, T., Harada, T., Matsushima, T., Xiao, T., Kollar, T., Yu, T., Ding, T., Davchev, T., Zhao, T. Z., Armstrong, T., Darrell, T., Chung, T., Jain, V., Kumar, V., Vanhoucke, V., Zhan, W., Zhou, W., Burgard, W., Chen, X., Chen, X., Wang, X., Zhu, X., Geng, X., Liu, X., Liangwei, X., Li, X., Pang, Y., Lu, Y., Ma, Y. J., Kim, Y., Chebotar, Y., Zhou, Y., Zhu, Y., Wu, Y., Xu, Y., Wang, Y., Bisk, Y., Dou, Y., Cho, Y., Lee, Y., Cui, Y., Cao, Y., Wu, Y.-H., Tang, Y., Zhu, Y., Zhang, Y., Jiang, Y., Li, Y., Li, Y., Iwasawa, Y., Matsuo, Y., Ma, Z., Xu, Z., Cui, Z. J., Zhang, Z., Fu, Z., and Lin, Z. (2023). Open X-Embodiment: Robotic learning datasets and RT-X models. https://arxiv.org/abs/2310.08864.

Contributors, D. (2025). Dexbotic: Open-source vision-language-action toolbox. arXiv preprint arXiv:2510.23511.

Dhariwal, P. and Nichol, A. (2021). Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations (ICLR 2021). OpenReview.net.

Driess, D., Springenberg, J. T., Ichter, B., Yu, L., Li-Bell, A., Pertsch, K., Ren, A. Z., Walke, H., Vuong, Q., Shi, L. X., et al. (2025). Knowledge insulating vision-language-action models: Train fast, run fast, generalize better. arXiv preprint arXiv:2505.23705.

Du, Y., Yang, M., Dai, B., Dai, H., Nachum, O., Tenenbaum, J. B., Schuurmans, D., and Abbeel, P. (2023). Learning universal policies via text-guided video generation. In Advances in Neural Information Processing Systems (NeurIPS).

Duan, J., Yuan, W., Pumacay, W., Wang, Y. R., Ehsani, K., Fox, D., and Krishna, R. (2024). Manipulate-anything: Automating real-world robots using vision-language models. arXiv preprint arXiv:2406.18915.

Ebert, F., Yang, Y., Schmeckpeper, K., Bucher, B., Georgakis, G., Daniilidis, K., Finn, C., and Levine, S. (2021). Bridge data: Boosting generalization of robotic skills with cross-domain datasets. arXiv preprint arXiv:2109.13396.

Fei, S., Wang, S., Shi, J., Dai, Z., Cai, J., Qian, P., Ji, L., He, X., Zhang, S., Fei, Z., Fu, J., Gong, J., and Qiu, X. (2025). Libero-plus: In-depth robustness analysis of vision-language-action models.

Gao, S., Liang, W., Zheng, K., Malik, A., Ye, S., Yu, S., Tseng, W.-C., Dong, Y., Mo, K., Lin, C.-H., Ma, Q., Nah, S., Magne, L., Xiang, J., Xie, Y., Zheng, R., Niu, D., Tan, Y. L., Zentner, K. R., Kurian, G., Indupuru, S., Jannaty, P., Gu, J., Zhang, J., Malik, J., Abbeel, P., Liu, M.-Y., Zhu, Y., and Linxi "Jim" Fan (2026). Dreamdojo: A generalist robot world model from large-scale human videos. arXiv preprint arXiv:2602.06949.

Gao, Y., Guo, H., Hoang, T., Huang, W., Jiang, L., Kong, F., Li, H., Li, J., Li, L., Li, X., Li, X., Li, Y., Lin, S., Lin, Z., Liu, J., Liu, S., Nie, X., Qing, Z., Ren, Y., Sun, L., Tian, Z., Wang, R., Wang, S., Wei, G., Wu, G., Wu, J., Xia, R., Xiao, F., Xiao, X., Yan, J., Yang, C., Yang, J., Yang, R., Yang, T., Yang, Y., Ye, Z., Zeng, X., Zeng, Y., Zhang, H., Zhao, Y., Zheng, X., Zhu, P., Zou, J., and Zuo, F. (2025). Seedance 1.0: Exploring the boundaries of video generation models. CoRR, abs/2506.09113.

Gemini Team, Google (2024). Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context.

Generalist AI (2025). Gen-0: Embodied foundation models that scale with physical interaction. https://generalistai. com/blog/nov-04-2025-GEN-0. Generalist AI Blog.

GigaBrain Team, Wang, B., Li, B., Ni, C., Huang, G., Zhao, G., Li, H., Li, J., Lv, J., Liu, J., Feng, L., Yu, M., Li, P., Deng, Q., Liu, T., Zhou, X., Chen, X., Wang, X., Wang, Y., Li, Y., Nie, Y., Li, Y., Zhou, Y., Ye, Y., Liu, Z., and Zhu, Z. (2026). Gigabrain-0.5m\*: a vla that learns from world model-based reinforcement learning. arXiv preprint arXiv:2602.12099.

GigaWorld Team, Ye, A., Wang, B., Ni, C., Huang, G., Zhao, G., Li, H., Zhu, J., Li, K., Xu, M., Deng, Q., Wang, S., Qin, W., Chen, X., Wang, X., Wang, Y., Cao, Y., Chang, Y., Xu, Y., Ye, Y., Wang, Y., Zhou, Y., Zhang, Z., Dong, Z., and Zhu, Z. (2025). Gigaworld-0: World models as data engine to empower embodied ai. arXiv preprint arXiv:2511.19861.

Google DeepMind (2025). Veo 3 model card. Technical report, Google DeepMind. Published May 23, 2025.

Guo, Y., Shi, L. X., Chen, J., and Finn, C. (2025). Ctrl-world: A controllable generative world model for robot manipulation. arXiv preprint arXiv:2510.10125.

He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778.

Hoque, R., Huang, P., Yoon, D. J., Sivapurapu, M., and Zhang, J. (2025). Egodex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709.

Intelligence, P., Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., et al. (2025a). pi<sub>0.5</sub>: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054.

Intelligence, P., Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., et al. (2025b). Pi0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054.

Jang, J., Ye, S., Lin, Z., Xiang, J., Bjorck, J., Fang, Y., Hu, F., Huang, S., Kundalia, K., Lin, Y.-C., Magne, L., Mandlekar, A., Narayan, A., Tan, Y. L., Wang, G., Wang, J., Wang, Q., Xu, Y., Zeng, X., Zheng, K., Zheng, R., Liu, M.-Y., Zettlemoyer, L., Fox, D., Kautz, J., Reed, S., Zhu, Y., and Fan, L. (2025). Dreamgen: Unlocking generalization in robot learning through neural trajectories. arXiv preprint arXiv:2505.12705.

Jiang, Z., Zhou, S., Jiang, Y., Huang, Z., Wei, M., Chen, Y., Zhou, T., Guo, Z., Lin, H., Zhang, Q., Wang, Y., Li, H., Yu, C., and Zhao, D. (2026). Wovr: World models as reliable simulators for post-training vla policies with rl. arXiv preprint arXiv:2602.13977.

Karamcheti, S., Nair, S., Balakrishna, A., et al. (2024). Prismatic: A (nearly) universal vision-language model with fine-grained visual representations. In International Conference on Machine Learning (ICML).

Khazatsky, A., Pertsch, K., Nair, S., Balakrishna, A., Dasari, S., Karamcheti, S., Nasiriany, S., Srirama, M. K., Chen, L. Y., Ellis, K., et al. (2024). Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945.

Kim, M. J., Finn, C., and Liang, P. (2025). Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645.

Kim, M. J., Gao, Y., Lin, T.-Y., Lin, Y.-C., Ge, Y., Lam, G., Liang, P., Song, S., Liu, M.-Y., Finn, C., and Gu, J. (2026). Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al. (2024). Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246.

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollar, P., and Girshick, R. (2023). Segment anything. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 4015–4026.

Ko, P.-C., Mao, J., Du, Y., Sun, S.-H., and Tenenbaum, J. B. (2024). Learning to act from actionless videos through dense correspondences. In International Conference on Learning Representations (ICLR).

Li, C., Zhang, R., Wong, J., Gokmen, C., Srivastava, S., Martín-Martín, R., Wang, C., Levine, G., Lingelbach, M., Sun, J., et al. (2023). Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation. In Conference on Robot Learning, pages 80–93. PMLR.

Li, L., Zhang, Q., Luo, Y., Yang, S., Wang, R., Han, F., Yu, M., Gao, Z., Xue, N., Zhu, X., Shen, Y., and Xu, Y. (2026). Causal world modeling for robot control. arXiv preprint arXiv:2601.21998.

Li, Q., Liang, Y., Wang, Z., Luo, L., Chen, X., Liao, M., Wei, F., Deng, Y., Xu, S., Zhang, Y., et al. (2024a). Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650.

Li, X., Hsu, K., Gu, J., Pertsch, K., Mees, O., Walke, H. R., Fu, C., Lunawat, I., Sieh, I., Kirmani, S., et al. (2024b). Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941.

Liu, B., Zhu, Y., Gao, C., Feng, Y., Liu, Q., Zhu, Y., and Stone, P. (2024a). Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. (2023a). Visual instruction tuning. CoRR, abs/2304.08485.

Liu, S., Wu, L., Li, B., Tan, H., Chen, H., Wang, Z., Xu, K., Su, H., and Zhu, J. (2024b). Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864.

Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., Zhu, J., and Zhang, L. (2023b). Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection.

Mees, O., Hermann, L., Rosete-Beas, E., and Burgard, W. (2022). Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334.

Nair, S., Rajeswaran, A., Kumar, V., Finn, C., and Gupta, A. (2022). R3M: A universal visual representation for robot manipulation.

Nasiriany, S., Maddukuri, A., Zhang, L., Parikh, A., Lo, A., Joshi, A., Mandlekar, A., and Zhu, Y. (2024). Robocasa: Large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523.

Octo Model Team, Ghosh, D., Walke, H., Pertsch, K., Black, K., Mees, O., Dasari, S., Hejna, J., Xu, C., Luo, J., Kreiman, T., Tan, Y., Sanketi, P., Vuong, Q., Xiao, T., Sadigh, D., Finn, C., and Levine, S. (2024). Octo: An open-source generalist robot policy. In Proceedings ofRobotics: Science and Systems, Delft, Netherlands.

OpenAI (2023). Gpt-4 technical report. arXiv:2303.08774.

OpenAI (2024). GPT-4o system card. CoRR, abs/2410.21276.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., and Bojanowski, P. (2023). DINOv2: Learning robust visual features without supervision.

Pai, J., Achenbach, L., Montesinos, V., Forrai, B., Mees, O., and Nava, E. (2025). mimic-video: Video-action models for generalizable robot control beyond vlas. arXiv preprint arXiv:2512.15692.

Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., and Levine, S. (2025). Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747.

Qiu, Y., Zhao, Z., Li, W., Ziser, Y., Korhonen, A., Cohen, S. B., and Ponti, E. M. (2026). Self-improving world modelling with latent actions. arXiv preprint arXiv:2602.06130.

Qu, D., Song, H., Chen, Q., Yao, Y., Ye, X., Ding, Y., Wang, Z., Gu, J., Zhao, B., Wang, D., et al. (2025). Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. (2021). Learning transferable visual models from natural language supervision. In Meila, M. and Zhang, T., editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 8748–8763. PMLR.

Shi, L. X., Ichter, B., Equi, M., Ke, L., Pertsch, K., Vuong, Q., Tanner, J., Walling, A., Wang, H., Fusai, N., et al. (2025). Hi robot: Open-ended instruction following with hierarchical vision-language-action models. arXiv preprint arXiv:2502.19417.

Team, G. R., Choromanski, K., Devin, C., Du, Y., Dwibedi, D., Gao, R., Jindal, A., Kipf, T., Kirmani, S., Leal, I., Liu, F., Majumdar, A., Marmon, A., Parada, C., Rubanova, Y., Shah, D., Sindhwani, V., Tan, J., Xia, F., Xiao, T., Yang, S., Yu, W., and Zhou, A. (2025). Evaluating gemini robotics policies in a veo world simulator.

Wei, S., Jing, H., Li, B., Zhao, Z., Mao, J., Ni, Z., He, S., Liu, J., Liu, X., Kang, K., Zang, S., Yuan, W., Pavone, M., Huang, D., and Wang, Y. (2026). Ψ<sub>0</sub>: An open foundation model towards universal humanoid loco-manipulation. arXiv preprint arXiv:2603.12263.

Wu, K., Hou, C., Liu, J., Che, Z., Ju, X., Yang, Z., Li, M., Zhao, Y., Xu, Z., Yang, G., Fan, S., Wang, X., Liao, F., Zhao, Z., Li, G., Jin, Z., Wang, L., Mao, J., Liu, N., Ren, P., Zhang, Q., Lyu, Y., Liu, M., He, J., Luo, Y., Gao, Z., Li, C., Gu, C., Fu, Y., Wu, D., Wang, X., Chen, S., Wang, Z., An, P., Qian, S., Zhang, S., and Tang, J. (2024). Robomind: Benchmark on multi-embodiment intelligence normative data for robot manipulation. arXiv preprint arXiv:2412.13877.

Wu, P., Escontrela, A., Hafner, D., Abbeel, P., and Goldberg, K. (2023). Daydreamer: World models for physical robot learning. In Proceedings ofThe 6th Conference on Robot Learning, volume 205 of Proceedings ofMachine Learning Research, pages 2226–2240. PMLR.

Wu, W., Lu, F., Wang, Y., Yang, S., Liu, S., Wang, F., Zhu, Q., Sun, H., Wang, Y., Ma, S., et al. (2026). A pragmatic vla foundation model. arXiv preprint arXiv:2601.18692.

Xiao, T., Radosavovic, I., Darrell, T., and Malik, J. (2022). Masked visual pre-training for motor control.

Yang, J., Tan, R., Wu, Q., Zheng, R., Peng, B., Liang, Y., Gu, Y., Cai, M., Ye, S., Jang, J., et al. (2025a). Magma: A foundation model for multimodal ai agents. arXiv preprint arXiv:2502.13130.

Yang, R., Yu, Q., Wu, Y., Yan, R., Li, B., Cheng, A.-C., Zou, X., Fang, Y., Cheng, X., Qiu, R.-Z., Yin, H., Liu, S., Han, S., Lu, Y., and Wang, X. (2025b). Egovla: Learning vision-language-action models from egocentric human videos. arXiv preprint arXiv:2507.12440.

Yang, S., Li, H., Chen, Y., Wang, B., Tian, Y., Wang, T., Wang, H., Zhao, F., Liao, Y., and Pang, J. (2025c). Instructvla: Vision-language-action instruction tuning from understanding to manipulation. arXiv preprint arXiv:2507.17520.

Ye, J., Wang, F., Gao, N., Yu, J., Zhu, Y., Wang, B., Zhang, J., Jin, W., Fu, Y., Zheng, F., et al. (2026a). St4vla: Spatially guided training for vision-language-action models. arXiv preprint arXiv:2602.10109.

Ye, S., Ge, Y., Zheng, K., Gao, S., Yu, S., Kurian, G., Indupuru, S., Tan, Y. L., Zhu, C., Xiang, J., Malik, A., Lee, K., Liang, W., Ranawaka, N., Gu, J., Xu, Y., Wang, G., Hu, F., Narayan, A., Bjorck, J., Wang, J., Kim, G., Niu, D., Zheng, R., Xie, Y., Wu, J., Wang, Q., Julian, R., Xu, D., Du, Y., Chebotar, Y., Reed, S., Kautz, J., Zhu, Y., Fan, L., and Jang, J. (2026b). World action models are zero-shot policies. arXiv preprint arXiv:2602.15922.

Ye, S., Jang, J., Jeon, B., Joo, S., Yang, J., Peng, B., Mandlekar, A., Tan, R., Chao, Y.-W., Lin, B. Y., Liden, L., Lee, K., Gao, J., Zettlemoyer, L., Fox, D., and Seo, M. (2025). Latent action pretraining from videos. In The Thirteenth International Conference on Learning Representations (ICLR).

Yuan, C., Zhou, R., Liu, M., Hu, Y., Wang, S., Yi, L., Wen, C., Zhang, S., and Gao, Y. (2025). Motiontrans: Human vr data enable motion-level learning for robotic manipulation policies. arXiv preprint arXiv:2509.17759.

Yuan, T., Dong, Z., Liu, Y., and Zhao, H. (2026). Fast-wam: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666.

Ze, Y., Zhang, G., Zhang, K., Hu, C., Wang, M., and Xu, H. (2024). 3d diffusion policy. arXiv preprint arXiv:2403.03954.

Zeng, A., Florence, P., Yang, M., Du, Y., et al. (2024). Molmoact: Vision-language-action model for robotic manipulation. arXiv preprint arXiv:2403.03368.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. (2023). Sigmoid loss for language image pre-training. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 11975–11986.

Zhao, T. Z., Kumar, V., Levine, S., and Finn, C. (2023). Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705.

Zheng, J., Li, J., Wang, Z., Liu, D., Kang, X., Feng, Y., Zheng, Y., Zou, J., Chen, Y., Zeng, J., et al. (2025a). X-vla: Softprompted transformer as scalable cross-embodiment vision-language-action model. arXiv preprint arXiv:2510.10274.

Zheng, R., Niu, D., Xie, Y., Wang, J., Xu, M., Jiang, Y., Castañeda, F., Hu, F., Tan, Y. L., Fu, L., Darrell, T., Huang, F., Zhu, Y., Xu, D., and Fan, L. (2026). Egoscale: Scaling dexterous manipulation with diverse egocentric human data. arXiv preprint arXiv:2602.16710.

Zheng, R., Wang, J., Reed, S., Bjorck, J., Fang, Y., Hu, F., Jang, J., Kundalia, K., Lin, Z., Magne, L., Narayan, A., Tan, Y. L., Wang, G., Wang, Q., Xiang, J., Xu, Y., Ye, S., Kautz, J., Huang, F., Zhu, Y., and Fan, L. (2025b). Flare: Robot learning with implicit world modeling. arXiv preprint arXiv:2505.15659.

Zhou, Z., Zhu, Y., Zhu, M., Wen, J., Liu, N., Xu, Z., Meng, W., Cheng, R., Peng, Y., Shen, C., et al. (2025). Chatvla: Unified multimodal understanding and robot control with vision-language-action model. arXiv preprint arXiv:2502.14420.

Zhu, L. Y., Kuppili, P., Punamiya, R., Aphiwetsa, P., Patel, D., Kareer, S., Ha, S., and Xu, D. (2025). Emma: Scaling mobile manipulation via egocentric human data. arXiv preprint arXiv:2509.04443.

> 💡 **Section 总结** (claude 批注):
> - **贡献者治理**: Authors(核心团队,长期维护)+ Community Contributors(生态贡献)双层模式;通讯作者 Yilun Chen / Shu Liu / Jiaya Jia。
> - **引用三根系**: VLA 方法(π0/GR00T/OpenVLA/FAST…)、world model(Cosmos/V-JEPA2/GigaWorld…)、benchmark(LIBERO/SimplerEnv/RoboTwin/RoboCasa/BEHAVIOR/CALVIN)。
> - **关键内部引用**: ST4VLA(Ye 2026a)是第 6 节 co-training 案例出处;Qwen3-VL(Bai 2025a)、Cosmos policy(Kim 2026)分别是两类 backbone 出处。
> - **可复用点**: 想复现某一 head,直接查其对标方法的引用(FAST→Pertsch 2025,OFT→Kim 2025,π→Black 2024,GR00T→Bjorck 2025)。
