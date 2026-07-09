[← 返回 README](../README.md)

# 05 - Conclusion, References & Reproducibility

## 原文 Section: CONCLUSION

In this work, we proposed CKMIL, a novel cascaded attention framework for WSI analysis that addresses the key information dilution problem in existing MIL methods. By first identifying key instances with a SDA module and then using them to guide an efficient global interaction via our KGGA module, CKMIL achieves a more focused and effective aggregation. Extensive experiments demonstrate that our approach sets a new state-of-the-art in cancer subtyping and survival prediction, proving the effectiveness of a key-instance-aware mechanism in computational pathology.

> **Hao 批注**: Conclusion 非常简洁——重申了核心问题（关键信息稀释）、核心方法（SDA + KGGA 级联）、核心结果（SOTA）。但没有讨论 limitation 或 future work，这在投稿阶段可能是一个弱点。审稿人可能期望在 Conclusion 中看到对方法限制的诚实讨论（如 UNI 特征下的退化、ICP 的不稳定性等）。

---

## 原文: REFERENCES

Amores, J. 2013. Multiple instance classification: Review, taxonomy and comparative study. *Artificial intelligence*, 201: 81-105.

Bera, K.; Schalper, K. A.; Rimm, D. L.; Velcheti, V.; and Madabhushi, A. 2019. Artificial intelligence in digital pathology -- new tools for diagnosis and precision oncology. *Nature reviews Clinical oncology*, 16(11): 703-715.

Brancati, N.; Anniciello, A. M.; Pati, P.; Riccio, D.; Scognamiglio, G.; Jaume, G.; De Pietro, G.; Di Bonito, M.; Foncubierta, A.; Botti, G.; et al. 2022. Bracs: A dataset for breast carcinoma subtyping in h&e histology images. *Database*, 2022: baac093.

Cai, Z.; Song, H.; Fingerhut, A.; Sun, J.; Ma, J.; Zhang, L.; Li, S.; Yu, C.; Zheng, M.; and Zang, L. 2021. A greater lymph node yield is required during pathological examination in microsatellite instability-high gastric cancer. *BMC cancer*, 21(1): 319.

Campanella, G.; Hanna, M. G.; Geneslaw, L.; Miraflor, A.; Werneck Krauss Silva, V.; Busam, K. J.; Brogi, E.; Reuter, V. E.; Klimstra, D. S.; and Fuchs, T. J. 2019. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. *Nature medicine*, 25(8): 1301-1309.

Chen, R. J.; Ding, T.; Lu, M. Y.; Williamson, D. F.; Jaume, G.; Chen, B.; Zhang, A.; Shao, D.; Song, A. H.; Shaban, M.; et al. 2024. Towards a General-Purpose Foundation Model for Computational Pathology. *Nature Medicine*.

Chen, Z.; Chi, Z.; Fu, H.; and Feng, D. 2013. Multi-instance multi-label image classification: A neural approach. *Neurocomputing*, 99: 298-306.

Cifci, D.; Veldhuizen, G. P.; Foersch, S.; and Kather, J. N. 2023. AI in computational pathology of cancer: improving diagnostic workflows and clinical outcomes? *Annual Review of Cancer Biology*, 7(1): 57-71.

Coudray, N.; Ocampo, P. S.; Sakellaropoulos, T.; Narula, N.; Snuderl, M.; Fenyo, D.; Moreira, A. L.; Razavian, N.; and Tsirigos, A. 2018. Classification and mutation prediction from non-small cell lung cancer histopathology images using deep learning. *Nature medicine*, 24(10): 1559-1567.

Cui, M.; and Zhang, D. Y. 2021. Artificial intelligence and computational pathology. *Laboratory Investigation*, 101(4): 412-422.

Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and Fei-Fei, L. 2009. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, 248-255. IEEE.

Elmore, J. G.; Longton, G. M.; Carney, P. A.; Geller, B. M.; Onega, T.; Tosteson, A. N.; Nelson, H. D.; Pepe, M. S.; Allison, K. H.; Schnitt, S. J.; et al. 2015. Diagnostic concordance among pathologists interpreting breast biopsy specimens. *Jama*, 313(11): 1122-1132.

Gu, A.; and Dao, T. 2023. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*.

He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 770-778.

Ilse, M.; Tomczak, J.; and Welling, M. 2018. Attention-based deep multiple instance learning. In *International conference on machine learning*, 2127-2136. PMLR.

Jin, C.; Guo, Z.; Lin, Y.; Luo, L.; and Chen, H. 2023. Label-efficient deep learning in medical image analysis: Challenges and future directions. *arXiv preprint arXiv:2303.12484*.

Li, B.; Li, Y.; and Eliceiri, K. W. 2021. Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 14318-14328.

Liu, Z.; Lin, Y.; Cao, Y.; Hu, H.; Wei, Y.; Zhang, Z.; Lin, S.; and Guo, B. 2021. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF international conference on computer vision*, 10012-10022.

Lu, M. Y.; Williamson, D. F.; Chen, T. Y.; Chen, R. J.; Barbieri, M.; and Mahmood, F. 2021. Data-efficient and weakly supervised computational pathology on whole-slide images. *Nature biomedical engineering*, 5(6): 555-570.

Ma, J.; Guo, Z.; Zhou, F.; Wang, Y.; Xu, Y.; Li, J.; Yan, F.; Cai, Y.; Zhu, Z.; Jin, C.; et al. 2024. Towards a generalizable pathology foundation model via unified knowledge distillation. *arXiv preprint arXiv:2407.18449*.

Maron, O.; and Lozano-Perez, T. 1997. A framework for multiple-instance learning. *Advances in neural information processing systems*, 10.

Shao, Z.; Bian, H.; Chen, Y.; Wang, Y.; Zhang, J.; Ji, X.; et al. 2021. Transmil: Transformer based correlated multiple instance learning for whole slide image classification. *Advances in neural information processing systems*, 34: 2136-2147.

Song, A. H.; Jaume, G.; Williamson, D. F.; Lu, M. Y.; Vaidya, A.; Miller, T. R.; and Mahmood, F. 2023. Artificial intelligence for digital and computational pathology. *Nature Reviews Bioengineering*, 1(12): 930-949.

Tang, W.; Qin, R.; Fang, H.; Zhou, F.; Chen, H.; Li, X.; and Cheng, M.-M. 2025. Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology. *arXiv preprint arXiv:2506.02408*.

Tang, W.; Zhou, F.; Huang, S.; Zhu, X.; Zhang, Y.; and Liu, B. 2024. Feature re-embedding: Towards foundation model-level performance in computational pathology. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 11343-11352.

Weinstein, J. N.; Collisson, E. A.; Mills, G. B.; Shaw, K. R.; Ozenberger, B. A.; Ellrott, K.; Shmulevich, I.; Sander, C.; and Stuart, J. M. 2013. The cancer genome atlas pan-cancer analysis project. *Nature genetics*, 45(10): 1113-1120.

Xiong, Y.; Zeng, Z.; Chakraborty, R.; Tan, M.; Fung, G.; Li, Y.; and Singh, V. 2021. Nystromformer: A Nystrom-based algorithm for approximating self-attention. In *Proceedings of the AAAI conference on artificial intelligence*, volume 35, 14138-14148.

Yang, S.; Wang, Y.; and Chen, H. 2024. Mambamil: Enhancing long sequence modeling with sequence reordering in computational pathology. In *International conference on medical image computing and computer-assisted intervention*, 296-306. Springer.

Yu, K.-H.; Zhang, C.; Berry, G. J.; Altman, R. B.; Re, C.; Rubin, D. L.; and Snyder, M. 2016. Predicting non-small cell lung cancer prognosis by fully automated microscopic pathology image features. *Nature communications*, 7(1): 12474.

> **Hao 批注**: 共 36 篇参考文献，涵盖四大类：
> - **MIL 方法论**: ABMIL, CLAM, DSMIL, TransMIL, MambaMIL, RRTMIL, ABMILX
> - **Nystrom Attention**: Nystromformer (Xiong et al. 2021)
> - **特征提取器与基础模型**: ResNet50, UNI, GPFM, Swin Transformer, Mamba
> - **计算病理综述与数据集**: BRACS, TCGA cohorts, 多篇 CPath 综述
>
> 参考文献覆盖面合理，涵盖了 MIL 方法链上的所有关键节点。Nystromformer 是唯一一篇非病理领域的核心引用——CKMIL 的 KGGA 直接依赖于该论文的 Nystrom attention 机制。

---

## 原文: REPRODUCIBILITY CHECKLIST

### 1. General Paper Structure

1.1. Includes a conceptual outline and/or pseudocode description of AI methods introduced (yes/partial/no/NA) **yes**

1.2. Clearly delineates statements that are opinions, hypothesis, and speculation from objective facts and results (yes/no) **yes**

1.3. Provides well-marked pedagogical references for less-familiar readers to gain background necessary to replicate the paper (yes/no) **yes**

### 2. Theoretical Contributions

2.1. Does this paper make theoretical contributions? (yes/no) **no**

### 3. Dataset Usage

3.1. Does this paper rely on one or more datasets? (yes/no) **yes**

3.2. A motivation is given for why the experiments are conducted on the selected datasets (yes/partial/no/NA) **yes**

3.3. All novel datasets introduced in this paper are included in a data appendix (yes/partial/no/NA) **NA**

3.4. All novel datasets introduced in this paper will be made publicly available upon publication of the paper with a license that allows free usage for research purposes (yes/partial/no/NA) **NA**

3.5. All datasets drawn from the existing literature (potentially including authors' own previously published work) are accompanied by appropriate citations (yes/no/NA) **yes**

3.6. All datasets drawn from the existing literature (potentially including authors' own previously published work) are publicly available (yes/partial/no/NA) **yes**

3.7. All datasets that are not publicly available are described in detail, with explanation why publicly available alternatives are not scientifically satisfying (yes/partial/no/NA) **NA**

### 4. Computational Experiments

4.1. Does this paper include computational experiments? (yes/no) **yes**

4.2. This paper states the number and range of values tried per (hyper-) parameter during development of the paper, along with the criterion used for selecting the final parameter setting (yes/partial/no/NA) **partial**

4.3. Any code required for pre-processing data is included in the appendix (yes/partial/no) **no**

4.4. All source code required for conducting and analyzing the experiments is included in a code appendix (yes/partial/no) **no**

4.5. All source code required for conducting and analyzing the experiments will be made publicly available upon publication of the paper with a license that allows free usage for research purposes (yes/partial/no) **yes**

4.6. All source code implementing new methods have comments detailing the implementation, with references to the paper where each step comes from (yes/partial/no) **partial**

4.7. If an algorithm depends on randomness, then the method used for setting seeds is described in a way sufficient to allow replication of results (yes/partial/no/NA) **yes**

4.8. This paper specifies the computing infrastructure used for running experiments (hardware and software), including GPU/CPU models; amount of memory; operating system; names and versions of relevant software libraries and frameworks (yes/partial/no) **partial**

4.9. This paper formally describes evaluation metrics used and explains the motivation for choosing these metrics (yes/partial/no) **yes**

4.10. This paper states the number of algorithm runs used to compute each reported result (yes/no) **yes**

4.11. Analysis of experiments goes beyond single-dimensional summaries of performance (e.g., average; median) to include measures of variation, confidence, or other distributional information (yes/no) **yes**

4.12. The significance of any improvement or decrease in performance is judged using appropriate statistical tests (e.g., Wilcoxon signed-rank) (yes/partial/no) **no**

4.13. This paper lists all final (hyper-)parameters used for each model/algorithm in the paper's experiments (yes/partial/no/NA) **yes**

> **Hao 批注, 可复现性评估**: 这项检查清单暴露了几个可复现性问题：
> - **4.2 (partial)**: 超参数搜索范围和最终选择标准未在主文中详细说明
> - **4.3-4.4 (no)**: 代码未提供（投稿阶段常见，但承诺发表后公开）
> - **4.6 (partial)**: 代码注释不完全
> - **4.8 (partial)**: 计算基础设施描述不完整（但 Supplementary Material 可能包含）
> - **4.12 (no)**: 未使用统计显著性检验（如 Wilcoxon signed-rank）来验证改进是否统计显著——这是潜在的弱点
>
> 其中 4.12 的缺失可能受到审稿人质疑——特别是对于 CKMIL vs ABMIL 这样标准差有重叠的场景。

---

## 🔖 Section 总结

### 核心洞察

1. **Conclusion 过于简短**: 没有讨论 limitations（UNI 退化、ICP 不稳定）、没有 future work、没有 broader impact。这在会议投稿阶段是常见的"省空间"策略，但可能被审稿人标记为不完整。

2. **参考文献的覆盖率好但深度有待提升**: 36 篇参考文献对于 10 页的会议论文是合理的，但 MIL 领域近年有大量相关工作（如 DAS-MIL、MHIM-MIL、DTFD-MIL 等），未在 Related Work 中提及可能是知识覆盖的遗漏。

3. **可复现性 checklist 的 self-report 诚实度高**: 作者坦诚地标记了 "partial" 和 "no"，没有过度声称。但 4.12（无统计检验）和 4.3-4.4（无代码）是实际弱点。

### 可追问点

- Supplementary Material 中包含哪些额外信息？超参数敏感性分析、更多可视化、训练细节是否在其中？
- 作者是否计划在 camera-ready 版本中补充 statistical significance test？
- 为什么没有在 Conclusion 中讨论 limitations？是篇幅限制还是有意识的选择？
