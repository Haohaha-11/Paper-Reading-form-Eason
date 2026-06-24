# iGVLM: Dynamic Instruction-Guided Vision Encoding for Question-Aware Multimodal Understanding

Hanpeng Liu <sup>1</sup> <sup>2</sup> Yaqian Li <sup>2</sup> Zidan Wang <sup>1</sup> Shuoxi Zhang <sup>3</sup> Zihao Bo <sup>2</sup> Rinyoichi Takezoe <sup>2</sup> Kaiwen Long <sup>2</sup> Kun He <sup>1</sup>

## Abstract

Despite the success of Large Vision–Language Models (LVLMs), most existing architectures suffer from a representation bottleneck: they rely on static, instruction-agnostic vision encoders whose visual representations are utilized in an invariant manner across different textual tasks. This rigidity hinders fine-grained reasoning where taskspecific visual cues are critical. To address this issue, we propose iGVLM, a general framework for instruction-guided visual modulation. iGVLM introduces a decoupled dual-branch architecture: a frozen representation branch that preserves taskagnostic visual representations learned during pretraining, and a dynamic conditioning branch that performs affine feature modulation via Adaptive Layer Normalization (AdaLN). This design enables a smooth transition from general-purpose perception to instruction-aware reasoning while maintaining the structural integrity and stability of pre-trained visual priors. Beyond standard benchmarks, we introduce MM4, a controlled diagnostic probe for quantifying logical consistency under multi-query, multi-instruction settings. Extensive results show that iGVLM consistently enhances instruction sensitivity across diverse language backbones, offering a plug-andplay paradigm for bridging passive perception and active reasoning.

## 1. Introduction

In recent years, advances in computer vision (Zhang et al., 2024; Liu et al., 2021) and natural language processing (Vaswani et al., 2017; Radford et al., 2019; Brown et al.,

![](images/96137fc811beeb264db3ad9176d8e30299502643957a793b1940f980fdfe1b60.jpg)  
What color is the truck in the image? (Red)  
Figure 1. Visualization of vision features. We employ Grad-CAM to visualize the vision encoders of the two branches in iGVLM, highlighting the regions most relevant to the correct answer. As shown in the figure, the instruction-guided branch distinctly focuses on areas that are more closely associated with the correct answer.

2020) have driven remarkable progress in Vision–Language Models (VLMs) (Chen et al., 2024a; Lu et al., 2024; Chen et al., 2023; Jiang et al., 2024). By jointly modeling visual perception and linguistic understanding, these models achieve strong performance on multimodal tasks such as image captioning, visual question answering, and grounded dialogue, representing an important step toward generalpurpose multimodal intelligence. Despite this progress, a fundamental challenge remains: how to condition visual perception on task-specific linguistic instructions in a principled and efficient manner.

Most existing VLMs rely on static, instruction-agnostic vision encoders, such as CLIP-ViT (Radford et al., 2021), which extract visual representations independently of the downstream textual query. As a result, visual features are reused across different instructions in an invariant manner, limiting the model’s ability to emphasize task-relevant cues and perform fine-grained, question-aware reasoning. This limitation is qualitatively illustrated in Figure 1, where static visual representations fail to highlight instruction-dependent regions that are critical for answering different questions grounded in the same image. These observations suggest that the core difficulty lies not in relearning visual perception itself, but in conditioning the utilization of visual features on linguistic instructions.

Consequently, recent work has explored lightweight mechanisms to introduce instruction awareness while preserving the perceptual strength of pretrained vision encoders. QA-ViT (Ganz et al., 2024) injects textual representations into upper layers of a frozen vision transformer, enabling limited instruction-dependent adaptation with high efficiency. However, such partial integration provides relatively weak conditioning and may still perturb pretrained visual representations. In contrast, DyFo (Li et al., 2025) formulates visual reasoning as a sequential decision process guided by external expert models and Monte Carlo Tree Search, allowing more flexible, instruction-aware attention shifts at the cost of substantial inference overhead and reliance on expert quality. Taken together, existing approaches highlight the challenge of achieving effective instruction conditioning while maintaining both computational efficiency and representation stability.

Motivated by this observation, we propose iGVLM, a decoupled instruction-guided vision encoder for Vision–Language Models. iGVLM adopts a dual-branch architecture that separates static and dynamic perception pathways: a frozen static branch preserves task-agnostic visual representations learned during pre-training, while a dynamic branch integrates lightweight, instruction-conditioned adapter modules that modulate feature utilization under textual guidance. This design enables flexible, instruction-aware visual reasoning without retraining the backbone, achieving a favorable balance between adaptability, efficiency, and representation stability. We evaluate iGVLM on the MMStar (Chen et al., 2024b) benchmark for fine-grained multimodal reasoning, and further introduce MM4, a controlled diagnostic benchmark for assessing question-aware visual reasoning under multi-query, multi-instruction settings.

Our main contributions are summarized as follows:

• We propose iGVLM, a decoupled instruction-guided vision encoder that separates representation preservation from instruction-conditioned adaptation via a dualbranch architecture.

• We introduce MM4, a controlled diagnostic benchmark for evaluating question-aware visual perception under multi-query, multi-instruction scenarios.

• We demonstrate through extensive experiments on MMStar and other multimodal benchmarks that iGVLM improves instruction sensitivity and finegrained reasoning while maintaining efficiency and general-purpose multimodal capability.

## 2. Related work

Vision Encoders in Vision–Language Models. Recent years have witnessed rapid progress in Vision–Language Models (VLMs) (Chen et al., 2025; Zhou et al., 2024; Wu et al., 2024b), driven by advances in both large-scale multimodal pretraining and architectural design. A foundational line of work, exemplified by CLIP (Radford et al., 2021), demonstrates that contrastive learning on large-scale image– text pairs can effectively align visual and textual representations, forming the basis of many modern VLMs. Subsequent studies have explored how to enhance the visual encoding component within VLMs to better support downstream multimodal reasoning.

One line of research focuses on strengthening visual representations by aggregating information from multiple encoders or pretrained visual models. For example, Ranzinger et al. (Ranzinger et al., 2024) fuse features from multiple vision encoders, while Tong et al. (Tong et al., 2024) augment CLIP features with representations from DINOv2 (Oquab et al., 2023), leading to improved visual grounding. Monkey (Li et al., 2024) further explores fine-tuning multiple vision encoders to support high-resolution image understanding. These approaches primarily aim to improve the capacity and coverage of visual representations, but typically rely on static encoders whose outputs are invariant to task-specific instructions. A complementary line of work investigates how to introduce query- or instruction-awareness into the vision encoder. QA-ViT (Ganz et al., 2024) incorporates query-aware cross-attention to modulate visual features based on textual prompts, enabling more effective integration of visual and linguistic information for question answering. While such designs provide a degree of instruction-dependent adaptation, they often operate within a single encoder pathway and offer limited control over how pretrained visual representations are preserved or modified. In contrast to these approaches, our work focuses on explicitly decoupling representation preservation from instruction-conditioned modulation within the vision en-

![](images/397bd620036db749e93ae3cb963ff7359d73014947879d3210b84b3f4875a668.jpg)  
Figure 2. (a): The proposed iGVLM architecture. The Text Encoder extracts features from the input instructions to guide the Vision Encoder, enabling dynamic modulation of visual representations. These instruction-conditioned features are then fused with static visual features. The fused representation is aligned with a Large Language Model (LLM) to generate responses. The illustrated example comes from a real-world VQA scenario rather than the MM4 benchmark. (b): AdaLN-Modified ViT. We leverage textual information to modulate the multi-head attention and MLP modules within the ViT the AdaLN adapter, enabling instruction-aware adjustment of visua attention.

coder.

Evaluating Vision–Language Models. Evaluating the capabilities of VLMs has been an active area of research, leading to the development of a diverse set of multimodal benchmarks. Early benchmarks, such as VQA (Goyal et al., 2017), MS-COCO (Sharma et al., 2018), and OK-VQA (Schwenk et al., 2022), provide task-specific assessments of multimodal perception and reasoning. More recent efforts aim to offer broader and more challenging evaluations of multimodal understanding and instruction following (Wu et al., 2024a; Fu et al., 2023; Cheng et al., 2023). MMStar (Chen et al., 2024b) further consolidates existing benchmarks and introduces a carefully curated, vision-dependent evaluation suite designed to mitigate data leakage and spurious correlations.

Despite these advances, most existing benchmarks primarily assess general-purpose multimodal capabilities and evaluate each query in isolation. As a result, they provide limited insight into whether a model can consistently adapt its visual perception to different instructions grounded in the same image. To address this gap, we introduce MM4, a controlled diagnostic benchmark specifically designed to evaluate question-aware visual understanding. MM4 challenges models to answer multiple, semantically distinct queries associated with a single image, enabling more fine-grained analysis of instruction-conditioned visual perception and multi-query consistency.

## 3. Method

In this section, we introduce iGVLM, a decoupled instruction-guided vision encoder designed to condition the utilization of visual features on linguistic instructions while preserving pretrained visual representations. We first present an overview of the overall architecture, followed by detailed descriptions of (i) instruction-guided visual feature modulation and (ii) dual-branch feature fusion. Finally, we introduce MM4, a controlled diagnostic benchmark for evaluating question-aware visual perception in Vision–Language Models (VLMs).

## 3.1. Overall Architecture

An overview of the proposed framework is illustrated in Figure 2(a). Given an image–text pair, iGVLM conditions visual feature generation on the textual instruction through a dedicated conditioning pathway, while preserving the original perceptual capacity of the pretrained vision backbone. Specifically, the textual instruction is first encoded into a compact semantic representation, which serves as a global guidance signal for visual modulation. This instruction embedding conditions a pretrained vision encoder, enabling visual features to be selectively modulated according to task-specific linguistic cues.

To explicitly separate representation preservation from instruction-conditioned adaptation, iGVLM adopts a dualbranch architecture. A static branch retains a frozen vision encoder to preserve task-agnostic visual priors, while a dynamic branch generates instruction-adapted visual features through lightweight modulation modules. The outputs of these two branches are fused to obtain a balanced visual representation that combines general-purpose perceptual semantics with task-specific adaptation. The fused visual features are subsequently projected into the language embedding space and provided, together with the instruction tokens, to a large language model (LLM) for multimodal reasoning and response generation.

## 3.2. Instruction-Guided Visual Feature Modulation

To enable instruction-conditioned visual perception, we derive a global textual guidance signal from the instruction. We adopt the text encoder from a pretrained CLIP model and truncate the input text to a maximum length of 77 tokens. The resulting [CLS] token embedding summarizes the semantic intent of the instruction and is mapped into the vision latent space through a lightweight linear projection:

$$
c _ { t } = \mathscr { F } _ { T } ( T _ { \leq 7 7 } ) , \quad \hat { c } _ { t } = \mathscr { H } _ { t } ( \mathrm { N o r m } ( c _ { t } ) ) ,\tag{1}
$$

where $\mathcal { F } _ { T } ( \cdot )$ denotes the CLIP text encoder, and $\mathcal { H } _ { t } ( \cdot )$ aligns the text embedding with the vision feature space.

We incorporate Adaptive Layer Normalization (AdaLN) (Perez et al., 2018) into each transformer block of the CLIP vision encoder to inject textual conditioning in a stable and localized manner. The projected instruction embedding $\hat { c } _ { t }$ is transformed into layer-wise modulation parameters that control feature scaling and shifting within both the self-attention and feedforward submodules. By integrating AdaLN across all transformer layers, iGVLM enables hierarchical instruction-conditioned modulation while preserving the pretrained weights of the vision backbone.

Formally, given an input image I and instruction embedding $\hat { c } _ { t } ,$ the instruction-guided vision encoder produces:

$$
y _ { c t } = \mathscr { F } _ { c t } ( I , \hat { c } _ { t } ; \Theta _ { \mathrm { C L I P } } ) ,\tag{2}
$$

where $y _ { c t } \in \mathbb { R } ^ { N _ { I } \times D _ { I } }$ denotes the instruction-conditioned visual features and $\Theta _ { \mathrm { C L I P } }$ represents the frozen pretrained parameters.

## 3.3. Dual-Branch Feature Fusion

While instruction-conditioned modulation enables taskspecific adaptation, preserving the original perceptual semantics is essential for stable and generalizable visual understanding. To this end, iGVLM employs a dual-branch fusion mechanism that explicitly combines instruction-guided features with the original frozen visual representations.

Let $y _ { c t } \in \mathbb { R } ^ { N _ { I } \times D _ { I } }$ denote the instruction-guided features from $\mathcal { F } _ { c t }$ , and let $y _ { 0 } = \mathcal { F } _ { I } ( I ; \Theta _ { \mathrm { C L I P } } )$ denote the corresponding frozen features from the original vision encoder. The fused visual representation is computed as:

$$
y _ { I } = \mathcal { Z } ( \mathrm { N o r m } ( y _ { c t } ) ) + y _ { 0 } ,\tag{3}
$$

where Z is a learnable linear projection initialized to zero. This initialization ensures that the fused representation initially matches the pretrained visual features, allowing instruction-conditioned adaptation to be introduced gradually and safely during training.

Following the LLaVA-1.5 framework, the fused visual features $y _ { I }$ are projected into the input embedding space of the LLM via a learnable linear transformation. Training proceeds in two stages: first, the instruction-guided vision encoder and projection layers are optimized while keeping the pretrained vision backbone and LLM frozen; second, all components are jointly optimized to enable coherent multimodal reasoning.

## 3.4. MM4: A Diagnostic Benchmark for Question-Aware Visual Perception

To complement existing multimodal benchmarks such as MMStar (Chen et al., 2024b), which primarily assess general-purpose multimodal reasoning, we introduce MM4, a controlled diagnostic benchmark designed to evaluate question-aware and multi-query visual perception. MM4 consists of 180 images and 720 manually verified question– answer pairs, with annotations curated by domain experts to ensure quality and consistency.

Each image in MM4 is associated with four semantically distinct questions, constructed according to three design principles: (i) robustness through answer reversal, (ii) multiperspective semantic diversity, and (iii) balanced answer distribution. This design enables MM4 to jointly assess intra-image consistency and inter-question diversity. To evaluate multi-query reasoning, MM4 adopts a hierarchical scoring protocol that credits a model only if it correctly answers at least n of the four questions per image, encouraging consistent instruction-aware reasoning rather than isolated accuracy.

## 4. Experiments

## 4.1. Experimental Settings

Training Setup. All experiments are conducted within the LLaVA-1.5 training framework to ensure a controlled and fair comparison. We use the same open-source training data as LLaVA-1.5, including 558K image–text pairs for alignment pretraining and 665K samples for instruction tuning, without introducing any additional data. All models share the same visual backbone, CLIP-Large-336 (Radford et al., 2021), and differ only in how visual features are conditioned on textual instructions. We evaluate iGVLM across multiple language backbones, including Vicuna-7B, Vicuna-13B (Chiang et al., 2023), and Qwen2.5-3B (Yang et al., 2024), enabling analysis of both intra-family scaling and cross-architecture generalization. All experiments are run on 8 NVIDIA A100 GPUs under identical hardware and software configurations. Compared to LLaVA-1.5, training a 7B version of iGVLM incurs a moderate computational overhead of approximately 1.1× GPU hours, reflecting the lightweight nature of the proposed instruction-guided visual modulation.

Evaluation Benchmarks. Our primary evaluation is conducted on MMStar (Chen et al., 2024b), a vision-dependent multimodal benchmark designed to assess fine-grained reasoning while minimizing data leakage, and we report MM-Star results as the main indicator of general-purpose multimodal performance. To further examine instruction sensitivity and generalization, we additionally evaluate on a range of established benchmarks, including VQAv2 (Goyal et al., 2017), GQA (Hudson & Manning, 2019), POPE (Li et al., 2023), VizWiz (Gurari et al., 2018), and ScienceQA (Lu et al., 2022), which collectively assess open-ended visual understanding, robustness to hallucination, zero-shot generalization, and scientific reasoning.

Baselines. We adopt LLaVA-1.5 (Liu et al., 2024) as the primary baseline, as it provides fully open-source data, model weights, and training code. All baseline models are trained using the same data, optimization settings, and vision backbone as iGVLM ensuring that performance differences arise solely from differences in visual encoding strategies. For Vicuna-based backbones, we additionally compare with representative instruction-aware modulation methods, including QA-ViT (Ganz et al., 2024) and DyFo (Li et al., 2025), which introduce query-aware modulation and expertguided search, respectively. For Qwen2.5-3B, we compare against LLaVA-1.5 with the same backbone due to architectural compatibility, ensuring consistent evaluation across different model families and instruction-conditioning strategies.

## 4.2. Results on MMStar

We evaluate iGVLM on MMStar under three representative settings. For Vicuna-based models (7B and 13B), we compare iGVLM with both the static baseline LLaVA-1.5 and two instruction-aware modulation methods, QA-ViT and DyFo. For the Qwen2.5-3B backbone, we compare against the corresponding LLaVA-1.5 model due to architectural compatibility. All methods are evaluated using identical training data, vision backbones, and inference configurations to ensure a controlled comparison.

As shown in Table 1, iGVLM achieves the best overall performance across all backbones. On Vicuna-7B, iGVLM improves the average MMStar score by +4.4 points over LLaVA-1.5, outperforming both QA-ViT (+1.2) and DyFo (+2.7). Notably, the gains are most pronounced on instruction-sensitive and fine-grained reasoning dimensions, including Instance Reasoning (IR), Logical Reasoning (LR), and Science & Technology (ST), where iGVLM consistently surpasses both baselines. A similar trend is observed on Vicuna-13B: iGVLM achieves an average improvement of +3.6 points, exceeding QA-ViT (+3.1) and DyFo (+1.8), with particularly strong gains on Fine-grained Perception (FP) and ST. These results indicate that explicitly conditioning the utilization of visual features enables more effective instruction-aware reasoning than single-path modulation (QA-ViT) or expert-guided search (DyFo).

In addition to accuracy, iGVLM maintains a favorable efficiency profile. Despite achieving higher overall performance, iGVLM preserves throughput comparable to LLaVA-1.5, with only a modest reduction (13.5→11.1 for Vicuna-7B and 9.8→8.6 for Vicuna-13B). In contrast, DyFo incurs a severe efficiency penalty due to repeated expertguided search, reducing throughput by more than 20× (13.5→0.49 for Vicuna-7B and 9.8→0.47 for Vicuna-13B). QA-ViT maintains efficiency similar to LLaVA-1.5, but achieves only limited accuracy gains, highlighting the tradeoff between conditioning strength and computational cost in existing approaches.

Results on Qwen2.5-3B further confirm the generality of iGVLM. Compared to LLaVA-1.5, iGVLM improves the average MMStar score from 16.8 to 21.3 (+4.5), with consistent gains across all capability dimensions. Taken together, these results demonstrate that iGVLM strikes a more effective balance between instruction-aware reasoning and computational efficiency than prior dynamic modulation methods, validating the advantages of decoupled instructionguided visual encoding on a general-purpose multimodal benchmark.

## 4.3. Results on MM4

We evaluate iGVLM on the proposed MM4 benchmark, which is specifically designed to assess question-aware and multi-query visual reasoning under shared visual inputs. Unlike general-purpose benchmarks that evaluate each query in isolation, MM4 requires models to adapt visual perception consistently across multiple, semantically distinct questions grounded in the same image.

Quantitative Results. As shown in Table 2, closed-source systems such as GPT-4o (OpenAI, 2024) and Qwen2.5-vlmax (Qwen Team, 2025) achieve the highest absolute scores, reflecting the advantages of large-scale proprietary models. Among open-source systems, iGVLM consistently outperforms its corresponding LLaVA-1.5 baselines under the same backbone. In particular, iGVLM-3B achieves the best performance among open-source models, improving over LLaVA-1.5-3B despite having the same parameter scale. Notably, iGVLM-3B also outperforms the larger iGVLM-13B, indicating that MM4 performance is driven more by instruction-aware visual utilization than by parameter count alone. These results suggest that the proposed decoupled visual modulation can effectively leverage stronger language backbones without architectural modification.

Table 1. Comparison among dynamic visual perception methods on MMStar, evaluated across six perception dimensions (CP, FP, IR, LR, ST, MA), average accuracy (Avg.), and throughput (it/s). Our iGVLM consistently outperforms prior approaches under both Vicuna and Qwen2.5 backbones, achieving the best overall balance between accuracy and efficiency. ∆ indicates improvement over LLaVA-1.5 under the same backbone. Bold and underlined values indicate the best and second-best results, respectively.
<table><tr><td>Method</td><td>Venue</td><td>CP</td><td>FP</td><td>IR</td><td>LR</td><td>ST</td><td>MA</td><td>Avg.</td><td>∆</td><td>Thpt</td></tr><tr><td colspan="9">Vicuna-7B Backbone</td></tr><tr><td>LLaVA-1.5</td><td>CVPR&#x27;24</td><td>58.8</td><td>24.0</td><td>38.8</td><td>24.0</td><td>13.6</td><td>22.8</td><td>30.3</td><td></td><td>13.5</td></tr><tr><td>QA-ViT</td><td>CVPR&#x27;24</td><td>60.0</td><td>28.0</td><td>40.4</td><td>25.6</td><td>14.0</td><td>21.2</td><td>31.5</td><td>+1.2</td><td>12.9</td></tr><tr><td>DyFo</td><td>CVPR&#x27;25</td><td>54.8</td><td>28.4</td><td>38.0</td><td>25.6</td><td>24.0</td><td>27.2</td><td>33.0</td><td>+2.7</td><td>0.49</td></tr><tr><td>iGVLM</td><td>Ours</td><td>59.6</td><td>28.0</td><td>41.2</td><td>32.8</td><td>19.6</td><td>26.8</td><td>34.7</td><td>+4.4</td><td>11.1</td></tr><tr><td colspan="9">Vicuna-13B Backbone</td></tr><tr><td>LLaVA-1.5</td><td>CVPR&#x27;24</td><td>58.8</td><td>28.0</td><td>41.6</td><td>24.4</td><td>18.4</td><td>25.6</td><td>32.8</td><td></td><td>9.8</td></tr><tr><td>QA-ViT</td><td>CVPR&#x27;24</td><td>57.6</td><td>32.8</td><td>43.2</td><td>28.8</td><td>21.2</td><td>31.6</td><td>35.9</td><td>+3.1</td><td>10.2</td></tr><tr><td>DyFo</td><td>CVPR’25</td><td>55.6</td><td>32.0</td><td>44.4</td><td>27.2</td><td>22.0</td><td>26.4</td><td>34.6</td><td>+1.8</td><td>0.47</td></tr><tr><td>iGVLM</td><td>Ours</td><td>57.6</td><td>37.6</td><td>43.2</td><td>27.2</td><td>22.4</td><td>30.8</td><td>36.4</td><td>+3.6</td><td>8.6</td></tr><tr><td colspan="10">Qwen2.5-3B Backbone</td></tr><tr><td>LLaVA-1.5</td><td>CVPR&#x27;24</td><td>8.4</td><td>8.4</td><td>16.4</td><td>24.4</td><td>16.8</td><td>26.4</td><td>16.8</td><td></td><td>12.6</td></tr><tr><td>iGVLM</td><td>Ours</td><td>20.0</td><td>13.2</td><td>19.2</td><td>28.4</td><td>21.6</td><td>25.6</td><td>21.3</td><td>+4.5</td><td>10.9</td></tr></table>

Multi-Query Consistency Analysis. MM4 adopts an increasingly strict evaluation protocol in which a model receives credit only if it correctly answers at least n out of four questions per image $( n = 1 , 2 , 3 , 4 )$ . As reported in Table 2, performance decreases monotonically as n increases for all models, reflecting the growing difficulty of maintaining consistent reasoning across multiple queries. However, iGVLM exhibits a noticeably slower performance degradation compared to baseline methods, particularly at higher consistency thresholds (n = 3 and n = 4). This behavior indicates that instruction-guided visual modulation enables more stable adaptation of visual attention across different questions, rather than relying on isolated correct predictions.

To contextualize these results, we note that random guessing yields an expected score of approximately 0.7 at n = 4, derived from a per-question accuracy of 0.25 over four independent questions. All evaluated models perform substantially above this baseline, confirming that MM4 provides a discriminative evaluation of multi-query reasoning. Importantly, the relative advantage of iGVLM becomes more pronounced under stricter consistency requirements, aligning with the intended diagnostic goal of MM4.

Qualitative Analysis. We further visualize representative examples in Figure 3 to illustrate how instruction-guided visual modulation affects model behavior. Compared with LLaVA-1.5-13B and QA-ViT-13B, iGVLM-13B more accurately localizes instruction-relevant regions under different queries. In the science diagram example, iGVLM distinguishes semantically similar stages such as evaporation and transpiration, while baseline models attend to ambiguous regions. In the food scene example, iGVLM demonstrates stronger compositional reasoning by correctly identifying missing objects and spatial relationships across multiple questions. Together, these qualitative and quantitative results confirm that iGVLM enhances question-aware visual perception by enabling consistent, instruction-conditioned feature utilization.

## 4.4. Other Benchmarks

To assess whether instruction-guided visual modulation affects general-purpose multimodal capabilities, we further evaluate iGVLM on a diverse set of established benchmarks, including VQAv2, GQA, POPE, VizWiz, and ScienceQA-IMG, as summarized in Table 3. These benchmarks cover complementary aspects of vision–language understanding, ranging from open-ended visual reasoning (VQAv2, GQA), hallucination robustness (POPE), real-world visual grounding (VizWiz), to domain-specific scientific reasoning (ScienceQA-IMG). In contrast, DyFo relies heavily on Monte Carlo Tree Search (MCTS) and lacks specialized search strategies for non-selective benchmarks such as VQAv2, GQA, and VizWiz, which significantly limits its generalizability compared to QA-ViT and our proposed iGVLM.

As shown in Table 3, iGVLM consistently maintains comparable or improved performance relative to LLaVA-1.5 across different model scales and backbones. For Vicunabased models, iGVLM yields modest but consistent gains on most benchmarks. In particular, iGVLM improves POPE accuracy from 85.4 to 85.9 on Vicuna-7B and from 85.4 to 86.1 on Vicuna-13B, indicating enhanced robustness against visual hallucination. Notable improvements are also observed on VizWiz, where iGVLM raises accuracy from 50.0 to 52.5 for Vicuna-7B and from 53.6 to 55.3 for Vicuna-13B, suggesting more reliable visual grounding under real-world conditions. On ScienceQA-IMG, iGVLM achieves clear gains for Vicuna-7B (+3.1), while maintaining comparable performance for Vicuna-13B.

![](images/b30c2d53d5a90b24813e991dbe3beed72adecb982d9156a6e2bfca27f4554285.jpg)  
Figure 3. Representative Examples from MM4. Deceptively simple questions demand multiperspective reasoning. Correct answers (highlighted) are uniformly distributed across options to prevent positional bias. More examples can be found in Section C.

Under the Qwen2.5-3B backbone, iGVLM shows a similar trend. While performance on VQAv2 and GQA remains comparable to LLaVA-1.5-3B, iGVLM substantially improves VizWiz accuracy from 50.7 to 53.4 and ScienceQA-IMG accuracy from 72.2 to 73.0. Across all evaluated benchmarks, no systematic performance degradation is observed, indicating that instruction-guided visual modulation does not compromise general-purpose multimodal reasoning. Overall, these results suggest that iGVLM serves as a drop-in enhancement to existing vision–language models, improving instruction-aware visual utilization while preserving broad applicability across diverse multimodal tasks.

## 4.5. Ablation Study

Effect of Architectural Components. We first examine the contribution of the key design components in iGVLM by ablating (i) instruction-conditioned adaptive layer normalization (AdaLN), (ii) the Zero-FFN adapter used for feature fusion, and (iii) the static branch in the dual-branch architecture. As shown in Table 4, removing AdaLN (w/o AdaLN) leads to consistent performance drops on both MMStar and MM4, indicating that layer-wise, instructionconditioned normalization is critical for effective visual modulation. Eliminating the Zero-FFN adapter (w/o FFN) further degrades performance, suggesting that controlled, gradual integration of dynamic features is necessary to avoid disrupting pre-trained visual representations. The most significant degradation is observed when the static branch is removed (w/o Pure), highlighting the importance of preserving task-agnostic visual priors alongside instruction-guided adaptation. Together, these results support the hypothesis that instruction-aware perception benefits from explicitly decoupling representation preservation from task-specific modulation.

Comparison with Alternative Modulation Designs. We further compare iGVLM with two representative variants that adopt different strategies for integrating instruction signals into visual features. iGVLM-MoF (Tong et al., 2024) interleaves static and dynamic tokens, while iGVLM-Cross (Peebles & Xie, 2023) replaces AdaLN with crossattention-based interaction. As reported in Table 5, both variants underperform the original iGVLM. MoF weakens the explicit separation between static and dynamic representations, while cross-attention introduces additional computational overhead and optimization noise without improving instruction consistency. These comparisons suggest that AdaLN-based modulation offers a more effective and efficient mechanism for conditioning visual representations on textual instructions.

Table 2. Scores of different methods on MM4. Different values of n indicate different scoring schemes.
<table><tr><td>n</td><td>GPT-40</td><td>Qwen2.5-vl-max</td><td>GPT-4v</td><td>iGVLM-13B</td><td>QA-ViT-13B</td><td>LLaVA1.5-13B</td><td>iGVLM-7B</td><td>LLaVA1.5-7B</td><td>QA-ViT-7B</td><td>LLaVA-1.5-3B</td><td>iGVLM-3B</td></tr><tr><td>1</td><td>170</td><td>172</td><td>161</td><td>161</td><td>163</td><td>161</td><td>161</td><td>157</td><td>162</td><td>165</td><td>164</td></tr><tr><td>2</td><td>138</td><td>149</td><td>120</td><td>117</td><td>115</td><td>113</td><td>109</td><td>113</td><td>108</td><td>130</td><td>124</td></tr><tr><td>3</td><td>107</td><td>114</td><td>66</td><td>71</td><td>68</td><td>77</td><td>58</td><td>61</td><td>59</td><td>82</td><td>78</td></tr><tr><td>4</td><td>58</td><td>52</td><td>24</td><td>23</td><td>21</td><td>18</td><td>17</td><td>13</td><td>11</td><td>27</td><td>29</td></tr></table>

Table 3. Comparison with existing VLMs on other benchmarks including VQAv2, GQA, POPE, VisWiz, and SQA. Our iGVLM consistently achieves higher accuracy across all backbones, confirming its strong generalization beyond MMStar and MM4.
<table><tr><td>Method</td><td>LLM</td><td>VQAv2</td><td>GQA</td><td>POPE</td><td>VisWiz</td><td>SQA</td></tr><tr><td>LLaVA-1.5</td><td>Vicuna-7B</td><td>78.5</td><td>62.0</td><td>85.4</td><td>50.0</td><td>66.8</td></tr><tr><td>QA-ViT</td><td>Vicuna-7B</td><td>79.0</td><td>62.2</td><td>85.7</td><td>51.3</td><td>68.1</td></tr><tr><td>DyFo</td><td>Vicuna-7B</td><td></td><td></td><td>84.8</td><td></td><td>65.5</td></tr><tr><td>iGVLM</td><td>Vicuna-7B</td><td>79.1</td><td>62.8</td><td>85.9</td><td>52.5</td><td>69.9</td></tr><tr><td>LLaVA-1.5</td><td>Vicuna-13B</td><td>80.0</td><td>63.3</td><td>85.4</td><td>53.6</td><td>71.6</td></tr><tr><td>QA-ViT</td><td>Vicuna-13B</td><td>79.9</td><td>63.2</td><td>85.8</td><td>56.0</td><td>71.3</td></tr><tr><td>DyFo</td><td>Vicuna-13B</td><td></td><td></td><td>84.6</td><td></td><td>71.6</td></tr><tr><td>iGVLM</td><td>Vicuna-13B</td><td>80.2</td><td>63.3</td><td>86.1</td><td>55.3</td><td>71.5</td></tr><tr><td>LLaVA-1.5</td><td>Qwen2.5-3B</td><td>77.7</td><td>62.4</td><td>84.5</td><td>50.7</td><td>72.2</td></tr><tr><td>iGVLM</td><td>Qwen2.5-3B</td><td>77.2</td><td>61.7</td><td>84.6</td><td>53.4</td><td>73.0</td></tr></table>

Table 4. Component ablation of iGVLM. We analyze the effect of removing key modules including AdaLN, Zero-FFN, and the static branch. Performance consistently drops, confirming that instruction-guided modulation, Zero-FFN fusion, and dual-branch structure are all critical for iGVLM’s effectiveness.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Params(B)</td><td rowspan=1 colspan=1>MM4</td><td rowspan=1 colspan=1>MMStar</td><td rowspan=1 colspan=1>VQAv2 Viswiz</td></tr><tr><td rowspan=1 colspan=1>LLaVA1.5-13B</td><td rowspan=1 colspan=1>13.35</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>32.8</td><td rowspan=1 colspan=1>80.0   53.6</td></tr><tr><td rowspan=1 colspan=1>iGVLM-13B</td><td rowspan=1 colspan=1>13.78</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>36.4</td><td rowspan=1 colspan=1>80.2   55.3</td></tr><tr><td rowspan=1 colspan=1>-w/o AdaLN</td><td rowspan=1 colspan=1>13.78</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>35.1</td><td rowspan=1 colspan=1>80.2   53.5</td></tr><tr><td rowspan=1 colspan=1>-w/o FFN</td><td rowspan=1 colspan=1>13.78</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>34.1</td><td rowspan=1 colspan=1>80.1   54.7</td></tr><tr><td rowspan=1 colspan=1>-w/o Pure</td><td rowspan=1 colspan=1>13.48</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>27.3</td><td rowspan=1 colspan=1>60.2   37.9</td></tr></table>

Table 5. Comparison of iGVLM variants on multiple benchmarks. iGVLM-MoF and iGVLM-Cross replace AdaLN with interleaved or cross-attention modulation, but both yield lower accuracy, highlighting the efficiency of our AdaLN-based design.
<table><tr><td>Method</td><td>MMStar</td><td>VQAv2</td><td>GQA</td><td>VisWiz</td><td>POPE</td></tr><tr><td>iGVLM</td><td>34.7</td><td>79.1</td><td>62.8</td><td>69.9</td><td>85.9</td></tr><tr><td>iGVLM-MoF</td><td>32.0</td><td>79.3</td><td>62.7</td><td>68.8</td><td>85.4</td></tr><tr><td>iGVLM-Cross</td><td>33.0</td><td>79.4</td><td>62.8</td><td>68.6</td><td>85.8</td></tr></table>

Table 6. Scaling performance of iGVLM on MMStar and MM4 benchmarks under different model sizes and LLM backbones.
<table><tr><td>Model</td><td>LLM</td><td>MMStar (Avg.)</td><td>MM4 (Score)</td></tr><tr><td>LLaVA-1.5B</td><td>Qwen2.5-1.5B</td><td>17.1</td><td>19</td></tr><tr><td>iGVLM-1.5B</td><td>Qwen2.5-1.5B</td><td>19.7</td><td>16</td></tr><tr><td>iGVLM-3B</td><td>Qwen2.5-3B</td><td>21.3</td><td>29</td></tr><tr><td>iGVLM-7B</td><td>Vicuna-7B</td><td>34.7</td><td>17</td></tr><tr><td>iGVLM-13B</td><td>Vicuna-13B</td><td>36.4</td><td>23</td></tr></table>

Scaling Behavior. Finally, we analyze how model capacity influences instruction-aware reasoning by training iGVLM-1.5B based on the Qwen2.5-1.5B backbone and comparing it with larger variants. As shown in Table 6, even the smallest iGVLM model improves over its LLaVA-1.5B counterpart on MMStar (19.7 vs. 17.1), indicating that instruction-guided visual modulation is beneficial across model scales. However, the lower MM4 score of iGVLM-1.5B (16 vs. 19) reveals that consistent multi-instruction reasoning requires sufficient language modeling capacity. Performance improves monotonically from 1.5B to 3B, 7B, and 13B, and notably, iGVLM-3B outperforms the larger Vicuna-13B variant on MM4. This trend suggests a strong synergy between the proposed dual-branch vision encoder and modern language backbones, and highlights that instruction-aware visual reasoning is jointly constrained by visual modulation and language capacity.

## 5. Conclusion

We presented iGVLM, a decoupled instruction-guided vision encoder that enables visual representations to be modulated according to textual instructions without retraining the visual backbone. By explicitly separating representation preservation from instruction-conditioned adaptation, iGVLM provides an efficient and stable mechanism for question-aware visual perception in vision–language models. Extensive experiments across diverse benchmarks demonstrate that iGVLM consistently improves instruction sensitivity and fine-grained multimodal reasoning while maintaining strong general-purpose performance across model scales from 3B to 13B parameters. In addition, we introduced MM4, a controlled diagnostic benchmark for evaluating multi-instruction, multi-query visual reasoning, enabling targeted analysis of instruction-conditioned perception. Overall, our results highlight the importance of explicitly conditioning the utilization of visual features on linguistic instructions, and suggest decoupled visual modulation as a principled design direction for instruction-aware multimodal models.

## References

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. NeurIPS, 33:1877–1901, 2020.

Chen, L., Li, J., Dong, X., Zhang, P., He, C., Wang, J., Zhao, F., and Lin, D. Sharegpt4v: Improving large multi-modal models with better captions. In ECCV, volume 15075, pp. 370–387, 2024a.

Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., and Zhao, F. Are we on the right way for evaluating large vision-language models? In NeurIPS, 2024b.

Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., and Ruan, C. Janus-pro: Unified multimodal understanding and generation with data and model scaling. CoRR, abs/2501.17811, 2025.

Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., Li, B., Luo, P., Lu, T., Qiao, Y., and Dai, J. Internvl: Scaling up vision foundation models and aligning for generic visuallinguistic tasks. CoRR, abs/2312.14238, 2023.

Cheng, S., Guo, Z., Wu, J., Fang, K., Li, P., Liu, H., and Liu, Y. Can vision-language models think from a first-person perspective? CoRR, abs/2311.15596, 2023.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I., and Xing, E. P. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality, March 2023. URL https://lmsys.org/blog/ 2023-03-30-vicuna/.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Qiu, Z., Lin, W., Yang, J., Zheng, X., Li, K., Sun, X., and Ji, R. MME: A comprehensive evaluation benchmark for multimodal large language models. CoRR, abs/2306.13394, 2023.

Ganz, R., Kittenplon, Y., Aberdam, A., Ben-Avraham, E., Nuriel, O., Mazor, S., and Litman, R. Question aware vision transformer for multimodal reasoning. In CVPR, pp. 13861–13871, 2024.

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In CVPR, pp. 6325–6334, 2017.

Gurari, D., Li, Q., Stangl, A. J., Guo, A., Lin, C., Grauman, K., Luo, J., and Bigham, J. P. Vizwiz grand challenge: Answering visual questions from blind people. In CVPR, pp. 3608–3617, 2018.

Hudson, D. A. and Manning, C. D. GQA: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, pp. 6700–6709, 2019.

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., de Las Casas, D., Hanna, E. B., Bressand, F., Lengyel, G., Bour, G., Lample, G., Lavaud, L. R., Saulnier, L., Lachaux, M., Stock, P., Subramanian, S., Yang, S., Antoniak, S., Scao, T. L., Gervet, T., Lavril, T., Wang, T., Lacroix, T., and Sayed, W. E. Mixtral of experts. CoRR, abs/2401.04088, 2024.

Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.

Li, G., Xu, J., Zhao, Y., and Peng, Y. Dyfo: A training-free dynamic focus visual search for enhancing lmms in finegrained visual understanding. In CVPR, pp. 9098–9108, 2025.

Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J. Evaluating object hallucination in large vision-language models. In EMNLP, pp. 292–305, 2023.

Li, Z., Yang, B., Liu, Q., Ma, Z., Zhang, S., Yang, J., Sun, Y., Liu, Y., and Bai, X. Monkey: Image resolution and text label are important things for large multi-modal models. In CVPR, pp. 26763–26773, 2024.

Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In CVPR, pp. 26286–26296, 2024.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, pp. 9992– 10002, 2021.

Lu, H., Liu, W., Zhang, B., Wang, B., Dong, K., Liu, B., Sun, J., Ren, T., Li, Z., Yang, H., Sun, Y., Deng, C., Xu, H., Xie, Z., and Ruan, C. Deepseek-vl: Towards real-world vision-language understanding. CoRR, abs/2403.05525, 2024.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K., Zhu, S., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS, 2022.

OpenAI. Gpt-4o system card. https://cdn.openai. com/gpt-4o-system-card.pdf, 2024.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In CVPR, pp. 4195–4205, 2023.

Perez, E., Strub, F., de Vries, H., Dumoulin, V., and Courville, A. C. Film: Visual reasoning with a general conditioning layer. In AAAI, pp. 3942–3951, 2018.

Qwen Team, A. G. Qwen2.5-vl technical report. CoRR, abs/2502.13923, 2025.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In ICML, volume 139, pp. 8748–8763, 2021.

Ranzinger, M., Heinrich, G., Kautz, J., and Molchanov, P. Am-radio: Agglomerative vision foundation model reduce all domains into one. In CVPR, pp. 12490–12500, 2024.

Schwenk, D., Khandelwal, A., Clark, C., Marino, K., and Mottaghi, R. A-OKVQA: A benchmark for visual question answering using world knowledge. In ECCV, volume 13668, pp. 146–162, 2022.

Sharma, P., Ding, N., Goodman, S., and Soricut, R. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL, pp. 2556– 2565, 2018.

Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., and Xie, S. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In CVPR, pp. 9568–9578, 2024.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In NeurIPS, pp. 5998–6008, 2017.

Wu, H., Zhang, Z., Zhang, E., Chen, C., Liao, L., Wang, A., Li, C., Sun, W., Yan, Q., Zhai, G., and Lin, W. Q-bench: A benchmark for general-purpose foundation models on low-level vision. In ICLR, 2024a.

Wu, Z., Chen, X., Pan, Z., Liu, X., Liu, W., Dai, D., Gao, H., Ma, Y., Wu, C., Wang, B., Xie, Z., Wu, Y., Hu, K., Wang, J., Sun, Y., Li, Y., Piao, Y., Guan, K., Liu, A., Xie, X., You, Y., Dong, K., Yu, X., Zhang, H., Zhao, L., Wang, Y.,

and Ruan, C. Deepseek-vl2: Mixture-of-experts visionlanguage models for advanced multimodal understanding. CoRR, abs/2412.10302, 2024b.

Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report. CoRR, abs/2412.15115, 2024.

Zhang, S., Liu, H., Lin, S., and He, K. You only need less attention at each stage in vision transformers. In CVPR, pp. 6057–6066, 2024.

Zhou, B., Hu, Y., Weng, X., Jia, J., Luo, J., Liu, X., Wu, J., and Huang, L. Tinyllava: A framework of small-scale large multimodal models. CoRR, abs/2402.14289, 2024.

Table 7. Hyperparameters of iGVLM are the same as the original LLaVA-1.5, except the learning rate in pretraining.
<table><tr><td>Hyperparameter</td><td>Pretrain</td><td>Finetune</td></tr><tr><td>Batch size</td><td>256</td><td>128</td></tr><tr><td>Lr</td><td>6e-4</td><td>2e-5</td></tr><tr><td>Lr schedule</td><td>cosine decay</td><td></td></tr><tr><td>Lr warmup ratio</td><td>0.03</td><td></td></tr><tr><td>Weight decay</td><td>0</td><td></td></tr><tr><td>Epoch</td><td>1</td><td></td></tr><tr><td>Optimizer</td><td>AdamW</td><td></td></tr><tr><td>DeepSpeed stage</td><td>2</td><td>3</td></tr></table>

## A. Hyperparameters

Our iGVLM follows the hyperparameter configuration of LLaVA-1.5. Since iGVLM employs a dual-branch vision encoder, we set the learning rate to 6e-4 during the pre-training stage. The hyperparameters for the first-stage vision–language alignment pre-training and the second-stage instruction tuning are provided in Table 7.

## B. Analyzing Question Diversity via CLIP Text Embeddings

During the construction of the MM4 benchmark, we enforce strict criteria to ensure sufficient distinction between question pairs $( Q _ { 1 } , Q _ { 2 } )$ and $( Q _ { 3 } , Q _ { 4 } )$ . To further verify the semantic separation achieved by these rules, we extract feature representations for all questions using the CLIP text encoder, compute pairwise cosine similarities, and visualize the results. We randomly select nine representative images for illustration. As shown in Figure 4, each heatmap consistently exhibits three distinct color clusters along both rows and columns, indicating that our rule-based construction effectively maintains semantic diversity and minimizes redundancy among questions.

## C. More Cases of MM4

We present additional examples from MM4 to provide a more intuitive demonstration of the importance of uestion-aware understanding for vision-language models.

Image 20 4 Questions Similarity Heatmap  
![](images/539337bdd2de245fb5a4439b593532bd6e78bff6b07c4ba667890ae971c32221.jpg)  
Image 78 4 Questions Similarity Heatmap

Image 45 4 Questions Similarity Heatmap  
![](images/8083d8cb7e05baaef79e9c29d4d701886dcad83821fd95f0e778119a3e3b7ff3.jpg)

Image 135 4 Questions Similarity Heatmap  
![](images/88f8d1a318144d93b8d9adb5e9acc9d5727ff5637434c94958e779ac079c3260.jpg)  
Image 88 4 Questions Similarity Heatmap

![](images/8ed2785f106683baa56087995524217fa59f7b786aceb7a70e1ebe773bf97f59.jpg)  
Image 95 4 Questions Similarity Heatmap

![](images/104e6b537ea4ce28b8a82c3fdb785b9c642bdafd7c88027fce1e4c5b1da1f259.jpg)

![](images/7373bcfb6882da36b003d7b23c23099192d61937c98b2d2ce8560aeafb1f5086.jpg)

![](images/6d427f3ad6254fb12dae06339ffe742b55db1850489d40ebefacc5040120e3c9.jpg)

Image 136 4 Questions Similarity Heatmap  
![](images/b90ea13bbad06ec36a518cc8f52dae5f3da4cb9adef9160c55ee9708199340be.jpg)  
Figure 4. Different questions similarity heatmap.

![](images/e5e42859f8c7dc55e7f0cc902e9d9742367ab568c549f954cff58fe4a6658215.jpg)

## Case-1:

![](images/87a7efb51f9a494f6fd72c065f7651c17070784dcfb961e741824ea2ac80650d.jpg)

What color dominates the flowers in the foreground? (Type: CP)

Options: A: Blue, B: Yellow, C: Red,

What color dominates the flowers in the foreground? (Type: CP)

Options: A: Yellow, , C: Red, D: Blue

What detail is noticeable about the ground surrounding the flowers? (Type: FP)

Options: A: It is fully paved, B: It is covered in grass, , D: It is sandy

What might be a practical reaso for the arrangement of flowers? (Type: IR)

Options: , B: To prevent erosion, C: To block noise, D: To create shade

##

GPT-4o: D, B, C, A QWen2.5-vl-max: D, B, C, A QWen2.5-vl-plus: C, B, C, A Doubao-vision-pro-32k: C, A, B, A GPT-4v: C, B, B, A iGVLM-13B: D, B, C, A QA-ViT-13B: D, B, C, A LLaVA-1.5-13B: D, B, C, A

## Case-2:

![](images/b4ed1476321c78d4305b432fa7dfa3df1c307afd30db531efbd90e6e27a39ea4.jpg)

What is the age range of the person wearing the yellow T-shirt? (Type: IR)

Options: A: 0-2 years old, B: 20-25 years old, , D: 60-65 years old

What is the age range of the person wearing the yellow T-shirt? (Type: IR)

Options: , B: 20-25 years old, C: 0-2 years old, D: 60-65 years old

What's the person in the white T-shirt doing? (Type: IR)

Options: A: Reading a book, , C: Drawing, D: Writing

What's on the wall in the upper right corner of the picture? (Type: FP)

Options: A: Star posters, B: People photos, C: Calendar,

##

GPT-4o: C, A, B, D QWen2.5-vl-max: C, A, B, D QWen2.5-vl-plus: C, A, B, D Doubao-vision-pro-32k: C, A, B, A GPT-4v: B, B, B, B iGVLM-13B: C, A, B, D QA-ViT-13B: C, A, B, C LLaVA-1.5-13B: C, C, B, D

## Case-3:

![](images/7c90bece6d76310237543894a02af049f988928a2d66f8786d4fa16b136fcfdd.jpg)

What is the man in the picture doing? (Type: IR)

Options: A: Running, B: Cycling, C: Skating,

What is the man in the picture doing? (Type: IR)

Options: A: Running, B: Cycling, , D: Skating

What is the direction of motion of the dog in the diagram? (Type: IR)

Options: , B: In the opposite direction of the man's movement, C: Movement toward the camera, D: Unable to determine direction

Which option does not match the man's appearance? (Type: FP)

Options: A: He's wearing gloves, , C: He's carrying a backpack, D: He's wearing blue pants

##

GPT-4o: D, C, A, B QWen2.5-vl-max: D, C, A, B QWen2.5-vl-plus: D, C, B, B Doubao-vision-pro-32k: D, C, A, A GPT-4v: D, C, A, B iGVLM-13B: D, C, A, B QA-ViT-13B: D, C, B, B LLaVA-1.5-13B: D, C, B, B

## Case-4:

![](images/ece55d4735a2721db4fa15a35808999255843ea7948afba2df6258b03ed6ec73.jpg)

Which place is the green area on the map? (Type: FP)

Options: , B: New Mexico, C: Arkansas, D: Mississippi

Which place is the green area on the map? (Type: FP)

Options: A: Mississippi, B: New Mexico, C: Arkansas,

Which country is this? (Type: CP)

Options: A: Canada, B: Mexico, , D: China

Where is the green area marked on the map located? (Type: IR)

Options: A: North, , C: Southwest, D: East

##

GPT-4o: A, D, C, C QWen2.5-vl-max: C, C, C, C QWen2.5-vl-plus: C, C, C, C Doubao-vision-pro-32k: C, C, C, C GPT-4v: B, C, C, C iGVLM-13B: A, D, C, C QA-ViT-13B: B, D, C, C LLaVA-1.5-13B: B, B, C, C

## Case-5:

![](images/95f18dcd7c741266561b887d4712aac56c0973206d957f1be383be4d63e4923f.jpg)

What is the answer in the picture? (Type: MA)

Options: , B: 40, C: 60, D: 10

What is the answer in the picture? (Type: MA)

Options: A: 10, B: 40, C: 60,

How many obtuse angles are in the picture? (Type: MA)

Options: A: 4, B: 3, , D: 1

How many triangles are in the picture? (Type: MA)

Options: A: 5, , C: 6, D: 3

##

GPT-4o: C, C, B, A QWen2.5-vl-max: C, C, C, C QWen2.5-vl-plus: C, C, D, C Doubao-vision-pro-32k: A, B, A, A GPT-4v: C, D, D, A iGVLM-13B: B, B, C, B QA-ViT-13B: B, B, C, D LLaVA-1.5-13B: A, D, D, D

## Case-6:

![](images/2d1d5fa6f584c65dd8f20b6dc674fb6846f1cc8d8e7acbb143705dcf8109a372.jpg)

What is the range of this function? (Type: MA)

Options: , B: [0, 2], C: [0, 1], D: [-1, 0]

What is the range of this function? (Type: MA)

Options: A: [0, 2], , C: [0, 1], D: [-1, 0]

What is the parity of this function? (Type: MA)

Options: A: Neither odd nor even, B: Even function, , D: Both even and odd function

What is the monotonicity of this function? (Type: MA)

Options: A: Monotonically increasing, B: Monotonically decreasing, C: First monotonically increasing, then decreasing, and finally increasing again,

##

GPT-4o: A, B, B, C QWen2.5-vl-max: A, B, B, C QWen2.5-vl-plus: A, B, A, C Doubao-vision-pro-32k: C, C, A, C GPT-4v: A, B, B, C iGVLM-13B: A, B, A, C QA-ViT-13B: C, B, C, C LLaVA-1.5-13B: C, B, C, C