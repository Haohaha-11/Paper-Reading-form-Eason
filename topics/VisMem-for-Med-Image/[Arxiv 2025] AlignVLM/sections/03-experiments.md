[← 返回 README](../README.md)

# Experiments

## 📌 预览
本文件合并 Experiments/Results/Analysis/Ablation，重点看方法是否真的提升视觉 grounding、医学诊断、鲁棒性和效率。

---

# 4 Experimental Setup

Setup. We conduct all experiments using 8 nodes of H100 GPUs, totaling 64 GPUs. For model training, we leverage the MS-Swift framework [Zhao et al., 2024] for its flexibility. Additionally, we utilize the DeepSpeed framework [Aminabadi et al., 2022], specifically the ZeRO-3 configuration, to optimize efficient parallel training across multiple nodes. Detailed hyperparameters are outlined in Appendix A.1.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

Table 1: Main Results on General Document Benchmarks. We compare ALIGNVLM (ours) with state-of-the-art (SOTA) open and closed-source instructed models, and with base models that we trained using the process described in Section 3.3. ALIGNVLM models outperform all Base VLM models trained in the same data regime. Our models also perform competitively across document benchmarks even compared with SOTA models, in which the data regime is more targeted and optimized. Color coding for comparison: closed-source models ,

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

![Table 4](../images/4e7ee4ac8c665535e9fe9bba4752642e2fb742c5f43ad41bb1594a2f85a748f7.jpg)
*Table 4: Table 4: Performance comparison when evaluating ALIGN with the full text embedding vocabulary (128K) versus the reduced subset of 3.4K high-probability embeddings. The results show negligible performance degradation, indicating that ALIGN relies primarily on a small subset of embeddings.*

> 💡 **Table 4 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

open-source models below 7B parameters , open-source models between 7-12B parameters .

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

Baselines. Our work focuses on architectural innovations, so we ensure that all baselines are trained on the same datasets. To enable fair comparisons, we evaluate our models against a set of Base VLMs fine-tuned on the same instruction-tuning tasks (Stages 2 and 3) as our models, using the BigDocs-7.5M and BigDocs-DocDownstream datasets. This approach ensures consistent training data, avoiding biases introduced by the Instruct versions of VLMs, which are often trained on undisclosed instruction-tuning datasets. Due to the scarcity of recently released publicly available Base VLMs, we primarily compare our model against the following Base VLMs of varying sizes: Qwen2-VL-2B [Wang et al., 2024], DocOwl1.5-8B [Hu et al., 2024], and LLama 3.2-11B [Grattafiori et al., 2024].

> 💡 **批注**: 这是一段信息密度较高的论述：建议拆成“现象/问题 → 机制 → 证据/影响”三层读。

For additional context, we also include results from the Instruct versions of recent VLMs of different sizes: Phi3.5-Vision-4B [Abdin et al., 2024], Qwen2-VL-2B and 7B [Wang et al., 2024], Qwen2.5- VL-7B [Qwen et al., 2025], LLaVA-NeXT-7B [Liu et al., 2024], InternVL2.5-2B and 8B [Chen et al., 2024b], InternVL3-2B and 8B [Zhu et al., 2025], Janus-1.3B [Wu et al., 2024a], DeepSeek-VL2-Tiny [Wu et al., 2024b], Ovis1.6-Gemma-9B [Lu et al., 2024], Llama3.2-11B [Grattafiori et al., 2024], DocOwl1.5-8B [Hu et al., 2024], and Pixtral-12B [Agrawal et al., 2024].

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

Evaluation Benchmarks. We evaluate our models on a diverse range of document understanding benchmarks that assess the model’s capabilities in OCR, chart reasoning, table processing, or form comprehension. In particular, we employ the VLMEvalKit [Duan et al., 2024] framework and report the results on the following popular benchmarks: DocVQA [Mathew et al., 2021b], InfoVQA [Mathew et al., 2021a], DeepForm [Svetlichnaya, 2020], KLC [Stanisławek et al., 2021], WTQ [Pasupat and Liang, 2015], TabFact [Chen et al., 2020], ChartQA [Masry et al., 2022], TextVQA [Singh et al., 2019], and TableVQA [Kim et al., 2024].

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

Table 2: Impact of Connector Designs on VLM Performance: We present the results of experiments evaluating different connector designs for conditioning LLMs on visual features. Our proposed ALIGN connector is compared against a basic Multi-Layer Perceptron (MLP), the Perceiver Resampler, and Ovis. The results demonstrate that ALIGN consistently outperforms these alternatives across all benchmarks.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

![Table 5](../images/6177f99a17a0b6d0f6827710489fa30fee5346587206894256e412406a5387a0.jpg)
*Table 5: Table 5: Robustness to Noise. Comparison of Avg. Scores with and without Gaussian noise $( \sigma = 3$ ), including performance drop $( \Delta )$ .*

> 💡 **Table 5 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

# 5 Results

# 5.1 Main Results

Table 1 presents the performance of ALIGNVLM compared to state-of-the-art (SOTA) open- and closed-source instructed models, as well as baseline Base VLMs fine-tuned in the same instructiontuning setup. The results demonstrate that ALIGNVLM consistently outperforms all Base VLMs within the same size category and achieves competitive performance against SOTA Instruct VLMs despite being trained on a more limited data regime. Below, we provide a detailed analysis.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

ALIGNVLM vs. Base VLMs. Our ALIGNVLM models, based on Llama 3.2-1B and Llama 3.2- 3B, significantly outperform the corresponding Base VLM, Qwen2-VL-2B, by up to $9 . 2 2 \%$ . Notably, ALIGNVLM-Llama-3.2-3B surpasses DocOwl1.5-8B, which has 4B more parameters, demonstrating the effectiveness of ALIGN in enhancing multimodal capabilities compared to traditional shallow fusion methods (e.g., MLPs). Furthermore, our 8B model achieves a $2 . 6 2 \%$ improvement over Llama3.2-11B despite sharing the same Base LLM, Llama3.1-8B. Since all models in this comparison were trained on the same instruction-tuning setup, this experiment provides a controlled evaluation, isolating the impact of architectural differences rather than dataset biases. Consequently, these results suggest that ALIGNVLM outperforms VLMs with shallow fusion techniques and surpasses parameter-heavy deep fusion VLMs, such as Llama3.2-11B, while maintaining a more efficient architecture.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

ALIGNVLM vs. Instruct VLMs. Even as open-source Instruct models are trained on significantly larger, often undisclosed instruction-tuning datasets, ALIGNVLM achieves competitive performance. For example, ALIGNVLM-Llama-3.2-3B $( 5 8 . 8 I \% )$ outperforms other strong instruction-tuned VLMs in its size class, such as Qwen2-VL-2B and InternVL-3-2B, by considerable margins $2 . 9 7 \%$ and $1 . 1 7 \%$ , respectively). While it falls slightly behind Qwen2.5-VL-3B, a direct comparison is not entirely fair, as the latter was trained on a proprietary instruction-tuning dataset.

Additionally, our 8B model outperforms significantly larger models such as Llama 3.2-11B and PixTral-12B by substantial margins. It also surpasses InternVL-2.5-8B and performs competitively with Qwen2.5-VL-7B, though a direct comparison may not be entirely fair since Qwen2.5-VL-7B was trained on an undisclosed instruction-tuning dataset. Finally, ALIGNVLM also exhibits comparable performance to closed-source models like GeminiPro-1.5 and GPT4o.

Overall, these results validate the effectiveness of ALIGN and establish ALIGNVLM as a state-of-theart model for multimodal document understanding.

# 5.3 Probability Distribution over Text Tokens Analysis

To better understand the behavior of ALIGN, we examine the probability distribution, $\mathbf { P _ { \mathrm { v o c a b } } }$ in Eq (1), over the LLM’s text vocabulary generated from visual features. Specifically, we process 100 document images through the vision encoder and ALIGN, then average the resulting probability distributions across all image patches. The final distribution is shown in Figure 3. As illustrated, the distribution is dense (rather than sparse), with the highest probability assigned to a single token being 0.0118. This can be explained by the vision feature space being continuous and of much higher cardinality than the discrete text space. Indeed, while the LLM has 128K distinct vocabulary tokens, an image patch (e.g., $1 4 \times 1 4$ pixels) contains continuous, high-dimensional information that cannot be effectively mapped to a single or a few discrete tokens.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

We conducted a deeper analysis of the token probability distributions produced by the ALIGN connector. Our observations show that ALIGN consistently assigns high probabilities to approximately 3.4K tokens from the entire vocabulary, while the remaining tokens receive negligible probabilities (below $1 0 ^ { - 6 }$ ). To better understand this behavior, we applied Principal Component Analysis (PCA) to reduce the dimensionality of the embeddings and visualized them in a two-dimensional space, as shown in Figure 4. The visualization reveals that these 3.4K tokens densely and comprehensively span the latent space of the LLM’s text embeddings. To validate this finding, we conducted additional evaluation experiments in which we retained only these 3.4K high-probability embeddings in the ALIGN connector, entirely removing the rest during evaluation. As shown in Table 4, the performance difference compared to using the full embedding set (128K) was negligible. This confirms that ALIGN effectively leverages and combines a compact subset of embeddings to map visual features into semantically meaningful regions within the LLM’s latent text space. Moreover, this suggests that ALIGN can be further optimized through targeted embedding pruning to improve computational efficiency without sacrificing performance.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# 5.4 Robustness to Noise Analysis

To evaluate the robustness of our ALIGN connector to noisy visual features, we conduct an experiment where random Gaussian noise is added to the visual features produced by the vision encoder before passing them into the connector. Specifically, given the visual features $\mathbf { F } \in \mathbb { R } ^ { N \times d }$ output by the vision encoder (where $N$ is the number of feature vectors and $d$ is their dimensionality), we perturbed them as

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

![Equation 10](../images/ec209c66564eda910f63fb3517a617e9e4e6f4c507460d016693a917f3610080.jpg)
*Equation 10: Equation extracted by MinerU.*

> 💡 **Equation 10 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

Table 5: Robustness to Noise. Comparison of Avg. Scores with and without Gaussian noise $( \sigma = 3$ ), including performance drop $( \Delta )$ .

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

![Table 6](../images/bd8359cdcf91bf8906a3d50792e1dc171fa7e757e3b2dbe74d027afea592a2e2.jpg)
*Table 6: Table 6: Detailed hyperparameters for each training stage across different LLM backbones.*

> 💡 **Table 6 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

![Figure 3](../images/451c44aadc02913db60e60a9a0d1bb213d955c53c923137741400ba123aa3ce3.jpg)
*Figure 3: Figure 3: Probability distribution over LLM tokens, highlighting dense probabilities for whitespace tokens.*

> 💡 **Figure 3 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 4](../images/ceebd133bc9b53b1e054a38238fe2fc40590c9428471c0c78896830f2f04cefd.jpg)
*Figure 4: Figure 4: PCA of ALIGN Embeddings: The principal components of the most influential embeddings in the Align Connector span most of the feature space represented by all embeddings.*

> 💡 **Figure 4 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

As shown in Table 5, our ALIGN connector demonstrates high robustness to noise, with only a $1 . 6 7 \%$ average drop in performance. In contrast, the widely adopted MLP connector suffers a significant performance degradation of $2 5 . 5 4 \%$ , highlighting its vulnerability to noisy inputs. Furthermore, we measured the average cosine distance between the original and noise-perturbed visual embeddings using both the ALIGN and MLP connectors. ALIGN showed significantly lower distances (0.0036) than MLP (0.3938), further validating its robustness to noise. These empirical results support our hypothesis that leveraging the knowledge encoded in the LLM’s text embeddings and constraining the visual features within the convex hull of the text latent space act as a regularization mechanism, reducing the model’s sensitivity to noisy visual features.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# A.1 Experimental Setup

We provide detailed hyperparameters of our experiments in Table 6.

Table 6: Detailed hyperparameters for each training stage across different LLM backbones.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

![Table 7](../images/8121b250a8a05589b7bb5d89ac036f815e5a1d85dc0b36e032e8d0e77bdf7ea9.jpg)
*Table 7: Table 7: Runtime and memory comparison between different connector designs. The results show that ALIGN introduces negligible computational overhead compared to other connectors.*

> 💡 **Table 7 批读**: 表格要看不同任务/模态/模型规模下是否一致提升；医学场景尤其关注 per-modality 和失败案例。

# A.3 Pixel-Level Tasks Analysis

To rigorously evaluate the ability of vision-language models to integrate fine-grained visual and textual pixellevel cues, we test our model on the VCR benchmark [Zhang et al., 2024], which requires the model to recover partially occluded texts with pixel-level hints from the revealed parts of the text. This task challenges VLM’s alignment of text and image in extreme situations. Current state-of-the-art models like GPT-4V OpenAI et al. [2023], Claude 3.5 Sonnet Anthropic [2024], and Llama-3.2 Dubey et al. [2024] significantly underperform humans on hard VCR task due to their inability to process subtle pixel-level cues in occluded text regions. These models frequently discard critical visual tokens during image tokenization on semantic priors, overlooking the interplay between partial character strokes and contextual visual scenes. To evaluate performance on VCR, we modify our Stage 3 SFT dataset composition by replacing the exclusive use of DocDownstream with a 5:1 blended ratio of DocDownstream and VCR training data. This adjustment enables direct evaluation of our architecture ALIGN’s ability to leverage pixel-level character cues.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

From the experimental outcomes, it is evident that ALIGNVLM consistently outperforms the MLP Connector Model across both easy and hard settings of the pixel-level VCR task (see Figure 5), with improvements ranging from $1 0 . 1 8 \%$ on the hard setting to $1 4 . 4 1 \%$ on the easy setting.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

We provide a case study on VCR in Figure 6, featuring four representative examples. In Figure 6a, it is evident that the MLP connector model fails to capture semantic consistency as effectively as ALIGNVLM. The phrase “The commune first census in written history in” (where the words in italics are generated by the model while the rest are in the image) is not as semantically coherent as the phrase generated by ALIGN “The commune first appears in written history in”.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

![Figure 5](../images/61347f42b2c200e4673e326746b975f9c95bd4bc20c9b4c778000f38c9b10040.jpg)
*Figure 5: Figure 5: Comparison of Llama-3.2-3b-ALIGN and Llama-3.2-3B-MLP on the Easy and Hard VCR tasks.*

> 💡 **Figure 5 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Beyond the issue of semantic fluency, in Figure 6b we also observe that ALIGNVLM successfully identifies the uncovered portion of the letter “g” in “accounting” and uses it as a pixel-level hint to infer the correct word. In contrast, the MLP model fails to effectively attend to this crucial detail.

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

Figures 6c and 6d show examples where ALIGNVLM fails on the VCR task. These carefully picked instances show that our method mistakes names of landmarks with common words when the two are very similar. As seen in the examples, ALIGNVLM mistakes “Llanengan" for “Llanongan" and “Gorden" for “Garden”. In both instances, the pairs differ by one character, indicating perhaps that ALIGNVLM tends to align vision representations to more common tokens in the vocabulary. One approach that would potentially mitigate such errors would be to train ALIGNVLM with more contextually-relevant data.

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

# A.4 Case Studies

In this section, we provide case studies for the experiments in Section 5.1. Specifically, we provide examples of our Llama-3.2-3B-ALIGN, and its counterpart model with alternative connectors Llama-3.2-3B-MLP and Llama-3.2-3B-Ovis on three different datasets: KLC [Stanisławek et al., 2021], DocVQA [Mathew et al., 2021b], and TextVQA [Singh et al., 2019]. The examples are shown in Figure 7, 8, and 9.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

![Figure 6](../images/a9b0c14ded967ebe428fc2530d4fda17ac4e5264ae6c3cddc0ae9181ac5165f5.jpg)
*Figure 6: Figure 6: Case Study for Pixel-Level Tasks. We provide examples of our proposed ALIGN connector compared with a the Multi-Layer Perceptron (MLP) connector. The ALIGN connector tends to better map visual elements to common words. GT is the ground truth.*

> 💡 **Figure 6 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 0](../images/f474ff2f1777e57266670e9b5bf01d580f90a0f764f3e8b00805dab456a9fa90.jpg)
*Figure 0: MinerU 原始图片*

> 💡 **Figure 0 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 0](../images/c7f39fff13bc3db1cc2f772d2d07ba8d1a8415ad9864909c9acd5b2a76cd7ebd.jpg)
*Figure 0: MinerU 原始图片*

> 💡 **Figure 0 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Question: What is the value for the charity name?   
GT: (Ardingly College Ltd.)   
MLP: (Ardington College Ltd.) ✗   
Ovis: (Ardington College Ltd.) $\cdot$   
ALIGN: (Ardingly College Ltd.) $\checkmark$ (a) Positive Example #1   
Question: What is the value for the address postcode?   
GT: (SW2 2QP)   
MLP: (SW22 0PQ) ✗   
Ovis: (SW2 2OP) ✗   
ALIGN: (SW2 2QP) ✓ (b) Positive Example #2   
Question: What is the value for the charity name?   
GT: (Human Appeal)   
MLP: (Humanitarian Agenda) ✗   
Ovis: (Human Appeal)   
ALIGN: (Human Rightsappeal) ✗ (c) Negative Example #1   
Question: What is the value for the post town address?   
GT: (Bishop’s Stortford)   
MLP: (Stortford) ✗   
Ovis: (Bishop’s Stortford)   
ALIGN: (Stortford) ✗ (d) Negative Example #2   
Question: What does the afternoon session begin on June 29?   
GT: (1:00)   
MLP: (2:45) ✗   
Ovis: (3:30) ✗   
ALIGN: (1:00) ✓ (a) Positive Example #1   
Question: What levels does the second table indicate?   
GT: (hematocrit data - Massachusetts)   
MLP: (SATISFACTORY) ✗   
Ovis: (Females) ✗   
ALIGN: (hematocrit data - Massachusetts) (b) Positive Example #2   
Question: What was the diet fed to the #1 group?   
GT: (basal diet)   
MLP: (basel diet) ✓   
Ovis: (Whole blood) ✗   
ALIGN: (control diet) ✗ (d) Negative Example #2   
Question: What greeting is written on the letter?   
GT: (good bye)   
MLP: (good) ✗   
Ovis: (good buy) ✗   
ALIGN: (good bye) ✓ (a) Positive Example #1   
Question: What indoor temperature is shown?   
GT: (68.4)   
MLP: $( \quad )$   
Ovis: $( 4 0 . 0 ) x$   
ALIGN: $( 6 8 . 4 ) \times$ (b) Positive Example #2   
Question: What type of club is advertised?   
GT: (health club)   
MLP: (topnote health club) ✗   
Ovis: (health club)   
ALIGN: (professional passionate personal) ✗ (c) Negative Example #1   
Question: What credit card is this?   
GT: (hadiah plus)   
MLP: (hadiah plus)   
Ovis: (american big loyalty program) ✗   
ALIGN: (hadia plus) ✗ (d) Negative Example #2

> 💡 **批注**: 这是实验证据：要同时看任务指标、鲁棒性、效率和消融。

![Figure 0](../images/bb4ae64f3a3ec69f8e31e8e7c34110ca4f515febc767820e5e18118da60d4fd2.jpg)
*Figure 0: MinerU 原始图片*

> 💡 **Figure 0 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 7](../images/c7f7aa67d099349cd9217bcf1c818a94eb3727daab52552e5064af9f4520b5f6.jpg)
*Figure 7: Figure 7: Case Study for Connector Comparison on the KLC dataset [Stanisławek et al., 2021]. We show four qualitative examples (including two correct and two incorrect examples) comparing Llama-3.2-3B-ALIGN to the same architecture with different connectors, Llama-3.2-3B-MLP and Llama-3.2-3B-Ovis. “GT” denotes the ground truth.*

> 💡 **Figure 7 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 0](../images/8d45b618a261b088c7852b4cdd6b16a0e5bf7bdd325cfc0cb69403fe96ef2f03.jpg)
*Figure 0: MinerU 原始图片*

> 💡 **Figure 0 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 0](../images/e66a8b3f68a6731a82ac476ed5b5359803ad846f034fd7f300be5a1a05a8b275.jpg)
*Figure 0: MinerU 原始图片*

> 💡 **Figure 0 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 0](../images/c293206317f004ab0afee584af0721b9b29d40005037488fedcb20163c621ccf.jpg)
*Figure 0: MinerU 原始图片*

> 💡 **Figure 0 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 8](../images/f95a06fc1fac83d8dda4efd17bf40ca34f5d50f841942be067d3317d880542f3.jpg)
*Figure 8: Figure 8: Case Study for Connector Comparison on the DocVQA dataset [Mathew et al., 2021b]. We show four qualitative examples (including two correct and two incorrect examples) comparing Llama-3.2-3B-ALIGN to the same architecture with different connectors, Llama-3.2-3B-MLP and Llama-3.2-3B-Ovis. “GT” denotes the ground truth.*

> 💡 **Figure 8 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 0](../images/6133652db17747e0ea1e53a6d0006d4ed1fde56399a350dec0db4b1c495638d1.jpg)
*Figure 0: MinerU 原始图片*

> 💡 **Figure 0 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 0](../images/b254dbcca6b550671899405d11a49fb40c0f3219542d4793aa5236ae5717ec6c.jpg)
*Figure 0: MinerU 原始图片*

> 💡 **Figure 0 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 0](../images/906263cf2b56e94aae6b0030fd5049756478b61a6e567d663acdea30cf05d696.jpg)
*Figure 0: MinerU 原始图片*

> 💡 **Figure 0 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

![Figure 9](../images/f867ff2fcd53ddd6f8040af178ed7134de4fa3a1a07367321937827ece8c002f.jpg)
*Figure 9: Figure 9: Case Study for Connector Comparison on the TextVQA dataset [Singh et al., 2019]. We show four qualitative examples (including two correct and two incorrect examples) comparing Llama-3.2-3B-ALIGN to the same architecture with different connectors, Llama-3.2-3B-MLP and Llama-3.2-3B-Ovis. “GT” denotes the ground truth.*

> 💡 **Figure 9 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

---

## 🔖 Section 总结

### 核心洞察
1. 检查 benchmark 是否覆盖医学/跨域/鲁棒性。
2. 关注消融、效率和失败案例。
