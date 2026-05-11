[← 返回 README](../README.md)

# 4. Experiments and Results

> 💡 **实验读法**: Sec. 4 要回答三件事：FOCUS 是否优于 MIL/FSWL baseline，三阶段模块是否逐步贡献，FM/LLM 选择是否会影响 few-shot 结果。

# 4.1. Experimental Settings

> 💡 **设置关键点**: 三个数据集覆盖 lung、breast、ovarian；实现用 CLAM 切 512x512 patch，CONCH 编码图文，Claude-3.5-Sonnet 生成 prompt，K 只取 4/8/16。

Datasets. To demonstrate the effectiveness of our proposed framework FOCUS, we conduct extensive experiments on three publicly available pathology datasets, i.e., TCGA-NSCLC1 [39], CAMELYON2,3 [3, 4], and UBC-OCEAN4, covering lung, breast, and ovarian cancers. TCGA-NSCLC comprises two lung cancer subtypes: lung adenocarcinoma (LUAD, 541 slides) and lung squamous cell carcinoma (LUSC, 512 slides). CAMELYON contains 577 normal slides and 341 slides with breast cancer metastasis. UBC-OCEAN encompasses five ovarian cancer subtypes: clear cell carcinoma (CC, 98 slides), endometrioid carcinoma (EC, 122 slides), high-grade serous carcinoma (HGSC, 221 slides), low-grade serous carcinoma (LGSC, 43 slides), and mucinous carcinoma (MC, 43 slides). Each dataset is split into train/validation/test sets with a ratio of 6:2:2, and we randomly sample $K$ samples per class under the $K$ -shot FSWL setting $( K = 4 , 8 , 1 6$ in our implementation).

Implementation Details. We utlize CLAM toolset [27] for WSI pre-processing, cropping the high-resolution $2 0 \times$ or $4 0 \times$ ) WSIs into patches of $5 1 2 \times 5 1 2$ pixels for feature extraction. For the frozen LLM that answers the pathology knowledge query, we use Claude-3.5-Sonnet. Both visual and textual features are encoded using CONCH [29], with a feature dimension $d$ of 512 for both modalities. Key hyperparameters of FOCUS include: a window size $w$ of 32 (Sec. 3.2.1), a compression ratio $\gamma$ of 0.8 (Sec. 3.2.2), and a similarity threshold $\theta _ { \mathrm { b a s e } }$ of 0.7 (Sec. 3.3). The model is optimized using AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ and trained for up to 80 epochs with early stopping, selecting the best checkpoint based on validation performance.

Evaluation Metrics. We employ three evaluation metrics: balanced accuracy (Balanced ACC), F1 score, and the area under the curve (AUC). To ensure robust evaluation, we conduct ten-fold cross-validation, where each fold contains a unique sampling of train/validation/test data. The mean and standard deviation across all folds are reported.

# 4.2. FSWL Results

> 💡 **主表证据链**: 所有方法都用 CONCH visual features，对比重点就落在 FOCUS 的 token compression + cross-modal aggregation，而不是特征提取器不公平。

To evaluate the effectiveness of FOCUS, we compare our framework against SOTA methods in general pathology image analysis approaches including TransMIL [34], DS-MIL [23], DTFD-MIL [54], and WiKG-MIL [25], as well as SOTA methods in few-shot pathology image analysis like TOP-MIL [32] and ViLa-MIL [35]. Additionally, we establish strong baselines using foundation model features with Max Pooling, Mean Pooling, and Attentive Pooling as linear probing methods. CONCH [29] is applied to extract visual features for all methods. We compare FOCUS with these methods across three datasets under three few-shot settings (4-shot, 8-shot, and 16-shot), as shown in Table 1.

> 💡 **TCGA-NSCLC 结果**: 4-shot Balanced ACC 81.9%、AUC 91.5%，比 ViLa-MIL 分别高 1.2 和 3.0 个百分点；这里最能体现文本提示和压缩对二分类低样本的帮助。

TCGA-NSCLC (2 classes). FOCUS achieves the best performance across all metrics and shot settings on this dataset. Notably, in the 4-shot setting, our method achieves a Balanced ACC of $8 1 . 9 \%$ , outperforming the second-best method (ViLa-MIL) by $1 . 2 \%$ . The improvement is particularly significant in AUC, where FOCUS reaches $9 1 . 5 \%$ , surpassing ViLa-MIL’s $8 8 . 5 \%$ by a margin of $3 \%$ . This performance advantage is maintained in higher-shot settings, with FOCUS achieving $9 7 . 2 \%$ AUC in the 16-shot scenario, demonstrating that our method can match or exceed the performance of fully-supervised approaches while requiring only a fraction of the labeled training data.

> 💡 **CAMELYON 结果**: 4-shot ACC 70.1% 对 ViLa-MIL 65.8%，提升 4.3 个百分点，是三组数据里低样本 Balanced ACC 提升最明显的场景。

CAMELYON (2 classes). FOCUS shows remarkable improvement in low-shot scenarios on this dataset. In the challenging 4-shot setting, our method achieves an ACC of $7 0 . 1 \%$ , significantly outperforming all baseline methods, with the next best method ViLa-MIL achieving only $6 5 . 8 \%$ . As the number of shots increases, FOCUS maintains competitive performance, achieving the best AUC scores in both 8-shot $( 8 5 . 6 \% )$ and 16-shot $( 9 4 . 3 \% )$ ) settings.

> 💡 **UBC-OCEAN 结果**: 五分类更依赖细粒度形态描述，FOCUS 在 4-shot ACC/AUC 达 70.4%/91.1%，说明 prompt 对 subtype 形态差异的编码确实进入了筛选和聚合链路。

UBC-OCEAN (5 classes). FOCUS shows strong performance across all few-shot settings on this more complex dataset. In the 4-shot scenario, our method achieves the highest ACC $( 7 0 . 4 \% )$ and AUC $( 9 1 . 1 \% )$ , significantly outperforming other methods. This advantage persists in higher-shot settings, with FOCUS achieving state-of-the-art AUC of $9 6 . 7 \%$ in the 16-shot setting, surpassing the previous best (DS- and TOP-MIL tied at $9 5 . 6 \%$ ) by $1 . 1 \%$ .

The extensive experimental results across three pathology datasets demonstrate that FOCUS consistently outperforms previous SOTA methods, especially in the most challenging 4-shot scenarios, in which our method achieves absolute improvements of $1 . 2 \%$ , $4 . 3 \%$ , and $3 . 3 \%$ in Balanced ACC over the second-best methods on the TCGA-NSCLC, CAMELYON, and UBC-OCEAN datasets, respectively. The performance advantages are maintained as the number of available shots increases, with FOCUS achieving the highest AUC scores of $9 7 . 2 \%$ , $9 4 . 3 \%$ , and $9 6 . 7 \%$ in the 16-shot setting across the three datasets.

# 4.3. Ablation Studies

# 4.3.1 Effects of Key Modules in FOCUS

> 💡 **消融解释**: BaseMIL→Prompt→KAVTC→SVTC→CrossAgg 的递增实验对应论文的设计论点：不是单靠 prompt，也不是单靠压缩，而是语义筛选和邻域过滤一起提高 few-shot 表现。

To evaluate the contribution of each module within FOCUS to the overall performance, we conduct an ablation study on the UBC-OCEAN dataset using 5 variants under 4-, 8-, and 16-shot settings. The experimental results, measured by Balanced ACC with ten-fold cross-validation, are reported in Table 2.

Table 2. Ablation study of key modules of FOCUS on UBC-OCEAN under 4-, 8-, and 16-shot settings (via Balanced ACC).   

<table><tr><td></td><td></td><td>BaseMIL Prompt KAVTC SVTC CrossAgg</td><td></td><td>Balanced ACC</td></tr><tr><td>✓ ✓</td><td>✓</td><td></td><td></td><td>0.623±0.086 0.638±0.074</td></tr><tr><td>20 ✓</td><td>✓</td><td>✓</td><td></td><td>0.673±0.082</td></tr><tr><td>✓ ✓</td><td>✓</td><td>✓ ✓</td><td>✓ ✓ √</td><td>0.688±0.056 0.704±0.088</td></tr><tr><td>✓ ✓</td><td>✓</td><td></td><td></td><td>0.682±0.034</td></tr><tr><td>20</td><td>✓</td><td></td><td></td><td>0.698±0.049</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td></td><td>0.731±0.037</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>0.769±0.045</td></tr><tr><td>✓</td><td>✓</td><td>√</td><td>✓ ✓</td><td>0.773±0.054</td></tr><tr><td>20 ✓ ✓ ✓</td><td>✓ ✓ ✓ ✓</td><td></td><td></td><td>0.779±0.040 0.798±0.038 0.827±0.030 0.849±0.048</td></tr></table>

We systematically analyze these variants by first establishing a baseline BaseMIL that performs only patch feature aggregation. We then incrementally added components: the Prompt variant incorporates a language prompt branch and computes similarity scores between slide-level features and class-specific text features for prediction. Next, we progressively integrated the knowledge-enhanced adaptive visual token compression (KAVTC, Figure 1(a)), sequential visual token compression (SVTC, Figure 1(b)), and crossmodal aggregation (CrossAgg, Figure 1(c)) modules. The experimental results demonstrate consistent performance improvements with each additional component, indicating the effectiveness of our proposed three-stage progressive and adaptive visual compression strategy.

# 4.3.2 Effects of Pathology FM Encoders

> 💡 **FM 选择结论**: CONCH 在 4/8/16-shot 上分别达到 70.4%、77.3%、86.4% Balanced ACC；作者将优势归因于大规模病理图文预训练带来的图文对齐。

To evaluate the effect of using different pathology FM as feature extractors for FOCUS, we employ 5 pathology FMs, i.e., UNI [8], GPFM [30], Virchow [41], PLIP [19], and CONCH [29] for comparison. Figure 2 shows the implications of different FMs as feature extractors across various few-shot settings, measured by Balanced ACC via ten-fold cross-validation. CONCH consistently outperforms other FMs, achieving the highest Balanced ACC across all settings $( 7 0 . 4 \%$ , $7 7 . 3 \%$ , and $8 6 . 4 \%$ for 4-shot, 8-shot, and 16-shot, respectively). This can be attributed to the fact that CONCH is pre-trained with large-scale pathological image-caption pairs, potentially generating more aligned visual and textual features. Interestingly, despite being a vision-language model, PLIP yields the lowest performance among both multi-modal and uni-modal FMs. Besides, all FMs show substantial improvement as the number of shots increases, with the most significant gains observed in 4- and 8-shot settings. The performance gap between FMs narrows under 16-shots, with UNI $( 8 4 . 3 \% )$ , GPFM $( 8 3 . 2 \% )$ , and Virchow $( 8 3 . 2 \% )$ achieving comparable results.

![](../images/d7e7b06d1475920b6e4bc5d468c34a5b3ded0f58a3274fe3b41ac33c66ed048e.jpg)  
Figure 2. Performance comparison of different foundation models (UNI, GPFM, Virchow, PLIP, and CONCH) on UBC-OCEAN under few-shot settings (measured by Balanced ACC).

# 4.3.3 Effects of LLMs for Prompt Generation

> 💡 **LLM prompt 结论**: 所有 LLM prompt 在 16-shot 都超过 83%，Claude-3.5-Sonnet 最高 86.4%；这说明 prompt 质量会影响上限，但 FOCUS 对不同 LLM 并非极端脆弱。

We also explore the effects of employing different LLMs for pathology knowledge query answering and the candidates include ChatGPT3.5-Turbo, ChatGPT-4o [1], OpenAI-o1- mini, Claude-3.5-Sonnet, and Llama3.1-405B [12]. Each LLM is prompted with identical instructions (detailed in Supplementary Material) to generate the visual descriptions used in FOCUS. We conduct this ablation study on the UBC-OCEAN dataset under 16-shots, with the results shown in Figure 3. The experimental results demonstrate the strong capability of LLMs in generating effective pathology description prompts, with all variants achieving accuracies above $8 3 \%$ . Among the evaluated models, Claude3.5-Sonnet achieved the highest Balanced ACC of $8 6 . 4 \%$ , followed by ChatGPT3.5-Turbo $( 8 4 . 8 \% )$ and OpenAI-o1-mini $( 8 4 . 6 \% )$ ). These observations suggest that LLMs with enhanced language modeling capabilities and domain-specific knowledge can potentially further boost the performance of FOCUS.

![](../images/aa1da8fc591bbb515598cae8a5401ec7b113a429aac064b79b1e062633f424ff.jpg)  
Figure 3. Performance comparison of prompts from different LLMs on UBC-OCEAN under 16-shot settings (Balanced ACC).
