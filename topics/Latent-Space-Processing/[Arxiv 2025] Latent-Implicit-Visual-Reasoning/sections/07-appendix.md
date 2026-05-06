[← 返回 README](../README.md)

## 📌 预览
Appendix 补充九个任务的数据来源、prompt、训练设置和更多 latent attention 可视化，是复现实验和判断数据泄漏风险的关键。

---

# References

> 💡 **References 入口**: 参考文献显示 LIVR 同时借鉴 LLaVA-style LMM、latent reasoning、视觉 CoT 和 BLINK/PixMo 等视觉任务体系。

[1] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, Huajie Tan, Chunyuan Li, Jing Yang, Jie Yu, Xiyao Wang, Bin Qin, Yumeng Wang, Zizhen Yan, Ziyong Feng, Ziwei Liu, Bo Li, and Jiankang Deng. Llava-onevision-1.5: Fully open framework for democratized multimodal training, 2025. 5
[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. 4
[3] Vassileios Balntas, Karel Lenc, Andrea Vedaldi, and Krystian Mikolajczyk. Hpatches: A benchmark and evaluation of handcrafted and learned local descriptors. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 4, 3
[4] Sean Bell, Kavita Bala, and Noah Snavely. Intrinsic images in the wild. ACM Trans. Graph., 33(4), 2014. 6
[5] Mahtab Bigverdi, Zelun Luo, Cheng-Yu Hsieh, Ethan Shen, Dongping Chen, Linda G. Shapiro, and Ranjay Krishna. Perception tokens enhance visual reasoning in multimodal language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3836–3845, 2025. 3
[6] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 91–104, 2025. 4, 1
[7] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. In Advances in Neural Information Processing Systems, pages 50742–50768, 2023. 4, 6
[8] Xingyu Fu, Ben Zhou, Ishaan Preetam Chandratreya, Carl Vondrick, and Dan Roth. There is a time and place for reasoning beyond the image, 2022. 2
[9] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In Computer Vision – ECCV 2024, pages 148–166, Cham, 2025. Springer Nature Switzerland. 4
[10] Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with pause tokens, 2024. 3
[11] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space, 2025. 3
[12] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. 4
[13] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models, 2025. 2
[14] Zihang Lai, Senthil Purushwalkam, and Abhinav Gupta. The functional correspondence problem. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15772–15781, 2021. 4, 5
[15] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. 7
[16] Bangzheng Li, Ximeng Sun, Jiang Liu, Ze Wang, Jialian Wu, Xiaodong Yu, Hao Chen, Emad Barsoum, Muhao Chen, and Zicheng Liu. Latent visual reasoning, 2025. 3
[17] Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vulic, and Furu Wei. Imag- ´ ine while reasoning in space: Multimodal visualization-ofthought, 2025. 3
[18] Peiyuan Liao, Xiuyu Li, Xihui Liu, and Kurt Keutzer. The artbench dataset: Benchmarking generative models with artworks, 2022. 4, 3
[19] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C. Lawrence ´ Zitnick. Microsoft coco: Common objects in context. In Computer Vision – ECCV 2014, pages 740–755, Cham, 2014. Springer International Publishing. 4, 1, 2
[20] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, pages 34892–34916. Curran Associates, Inc., 2023. 1
[21] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visualrft: Visual reinforcement fine-tuning, 2025. 2
[22] Yunze Man, De-An Huang, Guilin Liu, Shiwei Sheng, Shilong Liu, Liang-Yan Gui, Jan Kautz, Yu-Xiong Wang, and Zhiding Yu. Argus: Vision-centric reasoning with grounded chain-of-thought. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14268–14280, 2025. 2
[23] Juhong Min, Jongmin Lee, Jean Ponce, and Minsu Cho. Spair-71k: A large-scale benchmark for semantic correspondence, 2019. 4
[24] Lukas Murmann, Michael Gharbi, Miika Aittala, and Fredo Durand. A dataset of multi-illumination images in the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019. 4, 6
[25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 1
[26] John Schulman and Thinking Machines Lab. Lora without regret. Thinking Machines Lab: Connectionism, 2025. https://thinkingmachines.ai/blog/lora/. 4
[27] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. In Advances in Neural Information Processing Systems, pages 8612–8642. Curran Associates, Inc., 2024. 2
[28] Charlie Snell, Jaehoon Lee, Kelvin $\mathrm { X u }$ , and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters, 2024. 2
[29] Qwen Team. Qwen3 technical report, 2025. 4
[30] Omkar Thawakar, Dinura Dissanayake, Ketan More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, Yuhao Li, Mohammed Zumri, Jean Lahoud, Rao Muhammad Anwer, Hisham Cholakkal, Ivan Laptev, Mubarak Shah, Fahad Shahbaz Khan, and Salman Khan. Llamav-o1: Rethinking step-bystep visual reasoning in llms, 2025. 2
[31] Jiacong Wang, Zijian Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, and Jun Xiao. Vgr: Visual grounded reasoning, 2025. 2
[32] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600–612, 2004. 1
[33] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, pages 24824–24837. Curran Associates, Inc., 2022. 2
[34] Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2087–2098, 2025. 2
[35] Zeyuan Yang, Xueyang Yu, Delin Chen, Maohao Shen, and Chuang Gan. Machine mental imagery: Empower multimodal reasoning with latent visual tokens, 2025. 3, 5, 6
[36] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, and Dacheng Tao. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search, 2024. 2
[37] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems, pages 11809–11822. Curran Associates, Inc., 2023. 2
[38] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via stepwise group relative policy optimization, 2025. 2
[39] Kesen Zhao, Beier Zhu, Qianru Sun, and Hanwang Zhang. Unsupervised visual chain-of-thought reasoning via preference optimization, 2025. 2
[40] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Tsung-Yi Lin, Gordon Wetzstein, Ming-Yu Liu, and Donglai Xiang. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1702– 1713, 2025. 3

# Latent Implicit Visual Reasoning

Supplementary Material

Here, we provide additional details on datasets (Section A), training setup (Section B), and visualizations (Section C).

# A. Datasets

> 💡 **Appendix 数据总览**: 附录 A 是实验可信度关键，因为主文声称“不需要中间监督”，这里逐项说明训练数据确实是 QA 样本和 prompt template。

We evaluate LIVR on nine complementary, perceptionheavy tasks that together span low-level (visual correspondence, relative reflectance), mid-level (jigsaw, art style classification), and higher-level (localization, counting, visual similarity, semantic correspondence, functional correspondence) visual reasoning. Building on the BLINK benchmark and PixMo-Count, we derive evaluation splits that represent a broad and diverse testbed that stresses the model across heterogeneous task types and difficulty levels.

We provide detailed descriptions of the datasets and task setups used in our experiments. For each of the nine perception-heavy tasks - counting, jigsaw, object localization, visual correspondence, art style, semantic correspondence, functional correspondence, relative reflectance, and visual similarity - we describe: (i) the data sources and train/validation/test splits, and (ii) the VQA-style prompt templates used during training and evaluation.

# A.1. Counting

> 💡 **Counting 任务**: Counting 是唯一 open-ended exact-match 任务，能避免 multiple-choice 选项先验，但也更考验输出格式和数数能力。

# A.1.1. Data Sources and Splits

We use the PixMo-Count dataset [6], available as allenai/pixmo-count on HuggingFace, which provides an image URL, an object label (e.g., “people”, “cars”), and an integer count for each example, with official train, validation, and test splits.

For training, we construct a 1,000-example subset of the PixMo-Count train split by first discarding any examples whose remote image URLs no longer resolve and then restricting to images whose ground-truth counts lie in the range $c \in \{ 2 , 3 , . . . , 1 0 \}$ and sampling examples so that these counts are approximately uniformly represented, matching the range we evaluate on. The PixMo authors note that the official splits may contain overlapping images, so we perform an additional visual de-duplication step between our train $^ +$ validation images and the official PixMo-Count test images: using CLIP [25] embeddings together with perceptual hashing and SSIM-based image similarity [32], we flag near-duplicate pairs and remove any training/validation example whose image is a near-duplicate of a test image. From the remaining pool, we obtain 1,000 training instances.

For validation and test, we use the official PixMo-Count validation and test splits, discarding any examples whose remote image URLs no longer resolve; these contain 534 and 528 examples, respectively.

# A.1.2. Prompt Template

We phrase Counting as an open-ended task, using the following prompt template:

# Counting

![](../images/5f07c4871e940d1db491d1a3916a0cbb0df1dfd1a4f6bc27728cec05d127a6c6.jpg)

Prompt: How many {airplanes} are there in

Gold: 9

# A.2. Jigsaw

> 💡 **Jigsaw 任务**: Jigsaw 需要模型理解整体结构与局部拼接关系，是文本描述特别吃力的视觉抽象任务。

# A.2.1. Data Sources and Splits

We construct BLINK-style Jigsaw training and validation sets from COCO [19]. Starting from COCO2017 train2017 and val2017, we sample 1,000 and $2 5 0 \mathrm { i m } \cdot$ - ages for our train and val splits, respectively. For each image, we first sample a horizontal canvas of fixed width $4 0 0 \ \mathsf { p x }$ and random height in [170, 230] px, then crop a $4 0 0 \times h$ region from the original image. We partition this canvas into four equal quadrants and treat the bottom-right quadrant as the ground-truth patch. The input image is obtained by blacking out this quadrant, while the correct option is the original bottom-right patch. We then sample a distractor patch of the same size from the same COCO image, enforcing that it (i) intersects the canvas, (ii) does not overlap the ground-truth patch, and (iii) has its center at least a fixed fraction of the canvas size away from the ground-truth center. Finally, we randomly assign the ground-truth patch to option A or B, and use the other as the distractor. We ensure that each COCO image (identified by its file stem) appears in at most one of our Jigsaw train/val splits, so there is no cross-split image overlap within our generated data.

For our test split, we use the official BLINK Jigsaw val split (150 examples), which is constructed from the TARA dataset [8] rather than COCO. Because our COCObased Jigsaw training/validation sets and the BLINK Jigsaw benchmark come from disjoint source datasets, we do not perform any additional train–test de-duplication across them.

# A.2.2. Prompt Template

We phrase Jigsaw as a two-way multiple-choice question over candidate patches, using the following prompt template:

![](../images/29fbebf5df1cd4e5c8cb4d202f93bb7ebfa169422c378bd6405cddf215da992d.jpg)
Jigsaw

Prompt: Given the first image with the image is the missing part? Imagine which Select from the following choices.

Gold: A

# A.3. Object Localization

> 💡 **Localization 任务**: Localization 与 attention 可视化关系最直接，适合验证 latent 是否聚焦目标区域。

# A.3.1. Data Sources and Splits

We construct a 1,000-example training set and 250-example validation set for object localization from the COCO 2017 detection splits [19]. Starting from the official train2017 and $\mathtt { v a l } 2 0 1 7$ annotations, we first filter for non-crowd instances whose bounding boxes cover between $15 \%$ and $50 \%$ of the image area and whose segmentation masks fill at least $60 \%$ of the box area. For each selected instance, we treat its COCO bounding box as the “gold” localization and generate a distractor box by jittering the four box corners until the resulting box attains an IoU in [0.2, 0.5] with the gold box. We then render both boxes onto the image, label them as A and B, and phrase the task as a two-way multiple-choice question asking which box more accurately localizes the object. Because both our COCO-based data and the BLINK Object Localization benchmark are derived from the same underlying COCO imagery and overlay annotated bounding boxes, we perform an explicit visual deduplication step between our COCO candidates (train $^ +$ validation) and the BLINK Object Localization val images: using CLIP embeddings together with perceptual hashing and SSIM-based image similarity (on both raw and blurred grayscale images), we flag near-duplicate pairs and remove any COCO example whose image is a near-duplicate of a BLINK image. We also enforce that each COCO image (identified by its image ID) appears in at most one of our object-localization train/val splits, so there is no crosssplit image overlap within our generated data. From the remaining pool of candidates, we obtain 1,000 training and 250 validation instances.

For our test split, we use the BLINK Object Localization val split (122 examples).

# A.3.2. Prompt Template

We phrase Object Localization as a two-way multiplechoice question over candidate bounding boxes, using the following prompt template:

![](../images/9e1d8aee739fbe3ae263d3e3a7b7fc5ed7be277bc0d7ffa2eeeaa911cb7661b2.jpg)
Object Localization

Prompt: A bounding box is an annotated rectangle surrounding an object. The edges of bounding boxes being labeled. Given the two bounding boxes on the image, labeled by A and B, which bounding box more accurately localizes and encloses the {sandwich}?

Gold: A

# A.4. Visual Correspondence

> 💡 **Visual Correspondence 任务**: 对应关系强调低层几何/纹理匹配，不容易用语言中间步骤完整表达。

# A.4.1. Data Sources and Splits

We construct BLINK-style visual correspondence data from the HPatches sequences dataset [3]. We download the official HPatches sequence release (including homographies) and treat each sequence of six aligned images as a unit. Viewpoint and Illumination sequences are shuffled separately and then split 80/10/10 into train, val, and test subsets, after which we merge the viewpoint and illumination partitions for each split. This ensures that each HPatches sequence, and hence each source/target image pair, appears in exactly one split.

Within each sequence, we consider all forward image pairs $( i , j )$ with $1 \ \leq \ i \ < \ j \ \leq \ 6$ and $j \ : - \ : i \ : \geq \ : 2$ , following the six-image HPatches protocol. For a given pair, we use the provided homographies to map points from the source image $i$ to the target image $j$ . On the source image, we repeatedly sample reference points away from the image boundary and from one another, warp them to the target using the $i  j$ homography, and retain only those whose projections lie safely inside the target image. For each valid reference point, we create a single multiple-choice example: the source image shows the reference point annotated as “REF”, and the target image shows four candidate locations (labeled A–D), one of which is the true correspondence and three of which are distractors sampled to be far from the true location and from each other. We randomly assign the correct correspondence to one of the four labels, yielding BLINK-style MCQs. Across the HPatches sequences, we generate 1,000 training, 500 validation, and 700 test examples.

Because both the BLINK Visual Correspondence benchmark and our training data are derived from HPatches, using the official BLINK split together with strict de-duplication would leave too little training and validation data. Instead, we adopt the BLINK task format but evaluate on the HPatches-based test split constructed above.

# A.4.2. Prompt Template

We phrase Visual Correspondence as a four-way multiplechoice question over candidate correspondence points, using the following prompt template:

# Visual Correspondence

![](../images/4777b64ab12a13addace4f9e52b282e2732d45136df088a0319dc8986d793f1c.jpg)

Prompt: A point is circled on the first image, labeled with REF. We change the camera position or lighting

(B) Point B

Gold: D

# A.5. Art Style

> 💡 **Art Style 任务**: 风格判断需要全局统计特征，显式 box/crop 监督未必合适，因此是 LIVR 隐式抽象的自然场景。

# A.5.1. Data Sources and Splits

We construct a 1,000-example training set and 250-example validation set for binary art-style classification from the ArtBench-10 dataset [18]. Starting from the official $2 5 6 \times$ 256 ImageFolder variant with train/test splits, we first build a style-balanced pool of paintings from the ArtBench train split. Each example is then converted into a binary multiple-choice question: given a reference painting and two candidate paintings (A and B), the model must decide which candidate shares the same style as the reference. For each reference, we sample a positive candidate from the same ArtBench style (but a different image) and a negative candidate from a different style, and we randomize whether the positive candidate appears as option A or B. We ensure that each underlying ArtBench image appears in at most one of our train/val splits, so there is no cross-split image overlap within our generated data. Because ArtBench-10 also draws from large online art repositories (including WikiArt), the ArtBench and BLINK corpora can share overlapping images. To prevent train–test leakage, we perform an explicit crossdataset de-duplication between our ArtBench-based training/validation pool (considering all three images per example: reference and both candidates) and the BLINK Art Style val images. Using CLIP image embeddings to retrieve high-similarity pairs, followed by perceptual hashing and SSIM-based image similarity checks, we flag nearduplicate pairs and remove any training/validation example whose images are near-duplicates of BLINK Art Style examples. From the remaining pool of candidates, we obtain 1,000 training and 250 validation examples.

For our test split, we use the BLINK Art Style val split (117 examples).

# A.5.2. Prompt Template

We phrase Art Style as a two-way multiple-choice question over candidate images, using the following prompt template:

# Art Style

![](../images/31966532c5cd66ee37913a79b5cb665826fc88961c0b903163d9998ea354d139.jpg)

Prompt: Some most common art painting styles

Given the following images of art paintings, use the first image as the reference image, and determine which one

# A.6. Semantic Correspondence

> 💡 **Semantic Correspondence 任务**: 语义对应需要跨图定位同一部件，latent token 的任务条件化摘要在这里特别有意义。

# A.6.1. Data Sources and Splits

For semantic correspondence, we construct a 1,000- example training set and 250-example validation set from SPair-71k [23], which provides object-category image pairs with dense keypoint annotations. We use the official training split (trn) and filter to pairs with at least four valid keypoint correspondences (finite, non-negative coordinates in both source and target). From this pool, we construct BLINK-style four-way MCQs: for each selected pair, we mark a single reference point on the source image at one annotated keypoint (labeled “REF”), and on the target image we place four candidate circles (A–D), consisting of the true corresponding keypoint and three distractor keypoints sampled from the remaining annotations. All annotations are drawn directly on the original-resolution images, and we randomly assign the correct correspondence to one of the four labels. From the resulting candidates, we obtain 1,000 training and 250 validation examples. Because both our custom subset and BLINK’s Semantic Correspondence task are derived from SPair-71k, we explicitly de-duplicate our SPair-based train/validation pool against the BLINK Semantic Correspondence val split. We perform a pair-aware similarity check: for each BLINK pair, we retrieve a small set of nearest custom pairs under CLIP-based image similarity, consider both aligned and swapped orientations, and treat a pair as a duplicate only if both images pass strict perceptual-hash and SSIM criteria. Any flagged custom examples are removed. We obtain a total of 1,000 training and 250 validation instances.

For our test split, we use the BLINK Semantic Correspondence val split (139 examples) as the held-out test set, without further resampling or modification.

# A.6.2. Prompt Template

We phrase Semantic Correspondence as a four-way multiple-choice question over candidate correspondence points, using the following prompt template:

![](../images/799b15a4e22038dc0b0cc8ab4eea3ecfdd82619c9daccd33bdd52ad03ed48a56.jpg)
Semantic Correspondence

Prompt: Humans can find corresponding points for and the right front paw of one cat corresponds to the right front paw of the other cat.

Given the following two images, a reference point is annotated on the first image, labeled with REF. You are ${ } ^ { \prime \prime } \mathrm { A } .$ $\mathrm { D ^ { \prime \prime } }$ are drawn beside each circle.

Select between the choices on the second image and find the corresponding point for the reference point. Which point is corresponding to the reference point?

(B) Point B

Gold: D

# A.7. Functional Correspondence

> 💡 **Functional Correspondence 任务**: 功能对应是 Table 1 增益最明显的方向之一，说明 LIVR 对“看起来不同但用途相同”的抽象关系有帮助。

# A.7.1. Data Sources and Splits

For functional correspondence, we construct BLINK-style multiple-choice questions from the FunKPoint dataset [14], which provides per-image action labels (e.g., “pour”, “scoop”) together with five normalized functional keypoints. We first form a global, disjoint 80/10/10 split of FunKPoint images into train, val, and test, using an action-aware balancing scheme so that the per-action image counts are approximately preserved across splits. Each underlying FunKPoint image appears in exactly one split.

Within each split and action, we then pair images that share the same action using a one-use-per-image policy: images are randomly shuffled and grouped into disjoint pairs, ensuring that no image is reused within that split and action. For a given pair, we treat one image as the left (source) and the other as the right (target). On the left image, we sample a reference keypoint index $k \in \{ 1 , \ldots , 5 \}$ and mark the corresponding functional keypoint with a red circle labeled “REF”. On the right image, we annotate four candidate keypoints (A–D): the keypoint with the same index $k$ (the true correspondence) and three distractor keypoints sampled from the remaining indices $\{ 1 , \ldots , 5 \} \setminus \{ k \}$ . We randomly assign the correct correspondence to one of the four labels and phrase the task as a BLINK-style MCQ: given the action and the left “REF” point, select which candidate (A–D) on the right matches it. Across all actions, this procedure yields 1,000 training, 144 validation, and 146 test examples.

Because both our functional-correspondence data and the BLINK Functional Correspondence benchmark are derived from FunKPoint, using the official BLINK split together with strict de-duplication would leave too little training and validation data. As in our visual-correspondence setup, we therefore adopt the BLINK task format but evaluate on the FunKPoint-based test split constructed above, rather than on BLINK’s Functional Correspondence split.

# A.7.2. Prompt Template

We phrase Functional Correspondence as a four-way multiple-choice question over candidate correspondence points, using the following prompt template:

# Functional Correspondence

![](../images/1853dad276705ea0f3d359f9b62e9de107e33e617e048227aace4489fcaa4661.jpg)

Prompt: Humans can find corresponding points for the same action between different objects. For instance, if a person uses a

the first image, labeled with REF.

$\mathrm { D ^ { \prime \prime } }$

(B) Point B

Gold: D

# A.8. Relative Reflectance

# A.8.1. Data Sources and Splits

For relative reflectance, we construct three-way multiplechoice questions from the Multi-Illumination Dataset (MID) [24], which provides images of indoor scenes under multiple point-light directions together with per-pixel diffuse albedo. Starting from the official MID training split, we select a subset of scenes and 25 illumination directions per scene, download sRGB images at a fixed mip level, and attach the corresponding albedo images from the released archives. For each RGB–albedo pair, we generate a single example by sampling two spatially separated points on the image, converting the albedo to linear RGB, and computing the local disk-averaged luminance at each point. Let $Y _ { A }$ and $Y _ { B }$ be the luminance values at the two locations; we define the relative difference

$$
\mathrm { r e l } = \frac { \left| Y _ { A } - Y _ { B } \right| } { \operatorname* { m a x } ( Y _ { A } , Y _ { B } , 1 0 ^ { - 8 } ) } .
$$

If rel $\leq 0 . 1 0$ , we assign label (C) “About the same”; otherwise we assign (A) “A is darker” or (B) “B is darker” according to which point has lower luminance. We control the sampling schedule so that the “About the same” class constitutes roughly one quarter of the data. From this generated pool, we randomly subsample 1,000 training and 250 validation examples.

For our test split, we use the BLINK Relative Reflectance val split (134 examples). Because our training and validation data are derived from MID while BLINK Relative Reflectance is built on IIW [4], no additional crossdataset de-duplication is required.

# A.8.2. Prompt Template

We phrase Relative Reflectance as a three-way multiplechoice question over relative surface brightness, using the following prompt template:

# Relative Reflectance

![](../images/e27d69a3c270e78a09a9977019f3ca56aebdcb374182eb73afe2abbae6f0782d.jpg)

Prompt: Two points are annotated on the image, labeled by A darker surface color, or the colors is about the same?

(B) B is darker

Gold: B

# A.9. Visual Similarity

# A.9.1. Data Sources and Splits

For visual similarity, we use the NIGHTS dataset introduced in the DreamSim work [7] as our training and validation source, NIGHTS provides human-tested triplets consisting of a reference image and two candidates, together with votes indicating which candidate is perceptually closer to the reference. Each triplet is converted into a threeimage example (image 1, image 2, image 3), where image 1 is the reference, image 2 and image 3 are the two candidates in randomized order, and the label is “A” or “B” depending on whether the second or third image is judged more similar to the reference. Because both our NIGHTS triads and BLINK Visual Similarity are derived from the same underlying source, we enforce strict data deduplication. We deduplicate against BLINK by dropping any triad whose reference or candidate image is a near-duplicate of a BLINK Visual Similarity image, using a CLIP-based pre-filter followed by perceptual hashing and SSIM to confirm matches. After de-duplication, we retain 1,000 training and 250 validation triads from NIGHTS.

For our test split, we use the BLINK Visual Similarity val split (135 examples).

# A.9.2. Prompt Template

We phrase Visual Similarity as a two-way multiple-choice question over candidate images, using the following prompt template:

# Visual Similarity

![](../images/be41d665718d69ffa9323f1102a223ae39f55c4faa1c2c15f099acaaebd5ba5b.jpg)

Prompt: Given three similar but different images, take the first images is most similar to the first one?

Gold: B

# B. Training Setup

> 💡 **训练复现点**: LoRA rank 16、alpha 32、dropout 0.05，vision encoder/projector frozen；LIVR 额外训练 latent embedding rows。这个设置让方法开销集中在语言侧适配。

Single-task experiments. Single-task experiments use the following protocol. For each of the nine perceptionheavy tasks, we construct an independent training set of 1,000 examples and fine-tune a separate model per task. For a given task, the Direct SFT baseline is trained for 10 epochs on the 1,000 training examples. LIVR uses a twostage schedule with 4 epochs of Stage 1 (masked bottleneck training) followed by 6 epochs of Stage 2 (unmasked fine-tuning) on the same per-task training set, with $K = 1 6$ latent tokens. We select the checkpoint with the highest validation accuracy for reporting.

Multi-task experiments. For the multi-task setting with Qwen3-VL-4B-Instruct, we train on a combined dataset of six tasks (counting, localization, visual correspondence, semantic correspondence, functional correspondence, relative reflectance), using 1,000 training examples per task (6,000 total). Direct SFT is trained for 5 epochs, while LIVR is trained for 2 epochs of Stage 1 and 3 epochs of Stage 2, with $K = 1 6$ latent tokens. We report performance using the final checkpoint.

Optimization details. We adopt LoRA for both attention and MLP modules of the language backbone, with rank $r = 1 6$ , $\alpha = 3 2$ , and dropout 0.05. We keep the vision encoder and projector frozen. For Direct SFT, only the LoRA parameters are trainable; for LIVR we additionally unfreeze the rows of the embedding table corresponding to the $K$ latent tokens, while all other embeddings remain frozen. We use AdamW with learning rate $1 \times 1 0 ^ { - 4 }$ , weight decay 0.01, betas (0.9, 0.999), and $\epsilon = 1 0 ^ { - 8 }$ . Training uses a per-device batch size of 1 with 8 gradient-accumulation steps, giving an effective batch size of 8. The learning-rate schedule uses a $5 \%$ linear warmup followed by cosine decay; LIVR applies a separate schedule to each stage with the same warmup and cosine decay.

Mirage comparison. For the head-to-head comparison with Mirage, we align the number of stages and epochs but keep each method’s own two-stage objective. For LIVR, we set $K \ = \ 4$ latent tokens and train for 10 epochs in Stage 1 (LIVR’s masked bottleneck) and 10 epochs in Stage 2 (LIVR’s unmasked fine-tuning). For Mirage, we likewise use $K = 4$ latent tokens, training for 10 epochs with Mirage’s Stage 1 objective and 10 epochs with Mirage’s Stage 2 objective. In this comparison we use full finetuning of the language backbone (while keeping the vision encoder and projector frozen), with AdamW (learning rate $1 \times 1 0 ^ { - 5 }$ , weight decay 0.01, betas (0.9, 0.999), $\epsilon = 1 0 ^ { - 8 }$ ) and the same effective batch size of 8 (per-device batch size 1 with 8 gradient-accumulation steps). The learning-rate schedule again uses a $5 \%$ linear warmup followed by cosine decay for each stage.

Compute resources. All experiments are run with a fixed random seed of 42 on a single node equipped with 8 NVIDIA RTX 6000 Ada GPUs, using single-GPU training for each run.

# C. Additional Visualizations

> 💡 **补充可视化**: Figure 5 扩展了 attention 例子，尤其展示 Visual Similarity/Art Style 中正确答案 attention map 更接近参考图，补足主文 Figure 3。

# C.1. Additional Latent to Image Attentions

We show in Figure 5 some additional examples of latent-toimage attentions for Qwen-3-VL-4B. For the Visual Similarity and Art Style tasks, we display the raw attention maps instead. For question types where images serve as the choices (Jigsaw, Visual Similarity, and Art Style), we display a check mark symbol next to the correct answer. We can see that in the Visual Correspondence example, the latent tokens focus correctly on the parking line that is indicated by the REF point. In the Semantic Correspondence task, the model focuses correctly on the toothbrush handle. In the Jigsaw task, the latent tokens actually focus on diagonal ledge of the table in the masked out reference image, and finds this feature in Choice B, which is the correct answer. In Relative Reflectance, the latent tokens focus on the 2 points that it needs to compare. Finally, in the Visual Similarity and the Art Style tasks, the attention maps for the latent tokens of the correct answers are much more similar to attention maps of the reference images. For these tasks, it may be hard for humans to define explicit representations, but our approach allows latent tokens to learn these useful representations implicitly.

![](../images/147df4d2eb648e6b679dbb1aa352cd1d009fc1b049777a2b2be2b2388d1d9c1e.jpg)
Figure 5. Additional visualizations of latent token to image attention.

---

## 🔖 Section 总结

> 💡 **Section 小结**:
> - 附录价值: 数据来源、prompt、LoRA 配置和额外可视化支撑复现。
> - 关键检查点: 每任务 1k 训练样本、去重策略、vision encoder/projector frozen、latent embedding rows 额外解冻。
> - 可追问点: 自构训练集与 BLINK/Mirage 任务之间的分布差异是否会影响结论外推。
