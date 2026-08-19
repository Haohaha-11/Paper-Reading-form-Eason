[← 返回 README](../README.md)

# 4 结论、局限与参考文献

> 💡 **claude 批注｜结论预览**: 作者把 tile benchmark 定位为第一阶段候选筛选，把真实 slide 任务保留为最终验证。结论成立的层级是“跨任务平均的编码器排名”，且本文在 Mean Pooling 下观察到比 ABMIL 更高的相关；这支持对目标 slide 任务保留复验，但不能从两个 aggregator 设置推出一般的复杂度—重排单调关系。

## 4 Conclusion

In this paper, we show that tile-level benchmarking is a strong proxy for slidelevel model selection in digital pathology. Across 19 foundation models, tilelevel linear probing correlates strongly with slide-level performance, with higher correlation for mean pooling and slightly lower (but still clear) correlation for ABMIL, where learned aggregation introduces additional variability. Our sensitivity analyses indicate that the ABMIL correlation is stable and not driven by a single model, and that transferability is more sensitive to dataset properties (e.g., cohort size and number of tiles per slide) than to average task dificulty. The top-5 overlap analysis shows that tile-level benchmarks can help identify promising models for a new clinical slide-level task, even when the exact best models difer because tile and slide-level tasks may rely on diferent types of information. Overall, tile-level benchmarks provide an eficient and practical first step for pathology FM selection, while slide-level validation remains important for context-heavy tasks and for distinguishing between closely ranked models.

> 💡 **claude 批注｜可操作结论**: 推荐流程是“tile 线性探测粗筛 → 少量强候选跑完整 WSI 管线 → 在目标临床 cohort 上定案”。前一阶段节省重复特征提取与聚合器训练，后一阶段专门处理 tile benchmark 无法覆盖的上下文信息与中游模型重排。作者没有给出应保留几个候选的统一阈值，top-5 只是本实验的决策示例。

Limitations — (i) Our tile-level tasks are predominantly oriented toward local morphology, which may explain the lower top-5 overlap observed for immune profiling tasks that rely on broader spatial tissue architecture. (ii) Beyond dataset scope, more expressive aggregators such as ABMIL introduce learned task-specific parameters that reshape the frozen FM’s representation space, introducing aggregation-specific variability that may afect transferability for certain tasks. (iii) Tile-level benchmarking requires laborious manual annotations. However, this remains a fixed, one-time cost (already absorbed by public benchmarks like THUNDER). In contrast, the recurring hardware cost of evaluating new FMs or conducting hyperparameter searches is a bottleneck that tile-level benchmarking can substantially reduce across development cycles.

> 💡 **claude 批注｜局限拆解**:
> 1. **监督覆盖偏差**：THUNDER 的局部形态标签不充分代表空间组织、稀有病灶和免疫微环境，所以代理端可能系统性漏掉 slide 端所需信号。
> 2. **aggregation 依赖**：本文在 ABMIL 下观察到 Spearman 0.814，低于 Mean Pooling 的 0.925，作者解释为任务特异可学习聚合带来额外变异；其他 aggregator 或端到端微调是否产生更多、更少或不同位置的重排仍需实测。
> 3. **成本口径**：tile 人工标注昂贵但可复用，slide 全管线的算力成本会随模型／超参数反复发生。该结论依赖已有公共 tile benchmark，若新域没有局部标注，前期固定成本仍可能很高。

We view these remaining gaps as opportunities, and would like to emphasize that data quality might be a major factor to consider: richer, context-aware tilelevel datasets will strengthen tile-to-slide transferability, further consolidating tile-level benchmarking as the first step for FM selection in digital pathology.

> 💡 **claude 批注｜对 ReadySlide 的研究议程**: “更 context-aware 的 tile 数据”只是一个方向；ReadySlide 还应做 selector × consumer × budget 交叉，但分成两个可配对问题：固定 selector／budget，测同一组 encoder–consumer pipelines 的 full-bag→budgeted 排名迁移；固定 encoder／consumer，测同一组 selectors 在非退化预算间的排名稳定性，并以 retention／regret 对照 full-bag。再按病种、阳性区域稀有度、空间分散度、scanner／center shift 分层，定位各自失效边界。

Acknowledgments — This work has been supported by the Agence Nationale de la Recherche through ANR-23-IAHU-0002, ANR-21-CE45-0007, ANR-23- CE45-0029, ANR-23-IACL-0003 (DATAIA CLUSTER) and the Health Data Hub as part of the second edition of the France-Québec call for projects Intelligence Artificielle en santé. This work was performed using HPC resources from GENCI-IDRIS (Grant 2025-AD011015593R1 ).

Disclosure of Interests — The authors declare no competing interests.

## References

1. Aben, N., de Jong, E.D., Gatopoulos, I., et al.: Towards large-scale training of pathology foundation models. arXiv (2024)

2. Alfasly, S., Alabtah, G., Hemati, S., et al.: Validation of histopathology foundation models through whole slide image retrieval. Scientific Reports (2025)

3. Alfasly, S., Nejat, P., Hemati, S., et al.: Foundation models for histopathology—fanfare or flair. Mayo Clinic Proceedings: Digital Health (2024)

4. Brancati, N., Anniciello, A.M., Pati, P., et al.: Bracs: A dataset for breast carcinoma subtyping in h&e histology images. Database (2022)

5. Breen, J., Allen, K., Zucker, K., et al.: A comprehensive evaluation of histopathology foundation models for ovarian cancer subtype classification. NPJ Precision Oncology (2025)

6. Campanella, G., Chen, S., Singh, M., et al.: A clinical benchmark of public selfsupervised pathology foundation models. Nature Communications (2025)

7. Chen, R.J., Ding, T., Lu, M.Y., et al.: Towards a general-purpose foundation model for computational pathology. Nature Medicine (2024)

8. Ding, T., Wagner, S.J., Song, A.H., et al.: Multimodal whole slide foundation model for pathology. arXiv (2024)

9. Edwards, N.J., Oberti, M., Thangudu, R.R., et al.: The cptac data portal: a resource for cancer proteomics research. Journal of proteome research (2015)

10. Ehteshami Bejnordi, B., Veta, M., Johannes van Diest, P., et al.: Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. Jama 318(22), 2199–2210 (2017)

11. Filiot, A., Dop, N., Tchita, O., et al.: Distilling foundation models for robust and eficient models in digital pathology. In: MICCAI (2025)

12. Filiot, A., Ghermi, R., Olivier, A., et al.: Scaling self-supervised learning for histopathology with masked image modeling. medRxiv (2023)

13. Filiot, A., Jacob, P., Mac Kain, A., et al.: Phikon-v2, a large and public feature extractor for biomarker prediction. arXiv (2024)

14. Gatopoulos, I., Känzig, N., Moser, R., et al.: eva: Evaluation framework for pathology foundation models. In: MIDL (2024)

15. Gustafsson, F.K., Rantalainen, M.: Evaluating computational pathology foundation models for prostate cancer grading under distribution shifts. arXiv (2024)

16. Huang, Z., Bianchi, F., Yuksekgonul, M., et al.: A visual–language foundation model for pathology image analysis using medical twitter. Nature medicine (2023)

17. Ikezogwo, W., Seyfioglu, S., Ghezloo, F., et al.: Quilt-1m: One million image-text pairs for histopathology. NeurIPS (2023)

18. Ilse, M., Tomczak, J., Welling, M.: Attention-based deep multiple instance learning. In: ICML (2018)

19. Kang, M., Song, H., Park, S., et al.: Benchmarking self-supervised learning on diverse pathology datasets. In: CVPR (2023)

20. Karasikov, M., van Doorn, J., Känzig, N., et al.: Training state-of-the-art pathology foundation models with orders of magnitude less data. arXiv (2025)

21. Lee, J., Lim, J., Byeon, K., et al.: Benchmarking pathology foundation models: Adaptation strategies and scenarios. Computers in Biology and Medicine (2025)

22. Lu, M.Y., Chen, B., Williamson, D.F., et al.: A visual-language foundation model for computational pathology. Nature Medicine (2024)

23. Majzoub, R.A., Malik, H., Naseer, M., et al.: How good is my histopathology vision-language foundation model? a holistic benchmark. arXiv (2025)

24. Marza, P., Fillioux, L., Boutaj, S., et al.: THUNDER: Tile-level histopathology image understanding benchmark. Neural Information Processing Systems (NeurIPS) D&B Track (2025)

25. Nechaev, D., Pchelnikov, A., Ivanova, E.: Hibou: A family of foundational vision transformers for pathology. arXiv (2024)

26. Neidlinger, P., El Nahhas, O.S., Muti, H.S., et al.: Benchmarking foundation models as feature extractors for weakly-supervised computational pathology. arXiv (2024)

27. Oquab, M., Darcet, T., Moutakanni, T., et al.: Dinov2: Learning robust visual features without supervision. arXiv (2023)

28. Radford, A., Kim, J.W., Hallacy, C., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021)

29. Saillard, C., Jenatton, R., Llinares-López, F.o.: H-optimus-0 (2024), https:// github.com/bioptimus/releases/tree/main/models/h-optimus/v0

30. Shaikovski, G., Casson, A., Severson, K., et al.: Prism: A multi-modal generative foundation model for slide-level histopathology. arXiv (2024)

31. Shao, D., Chen, R.J., Song, A.H., et al.: Do multiple instance learning models transfer? arXiv preprint arXiv:2506.09022 (2025)

32. Tomczak, K., Czerwińska, P., Wiznerowicz, M.: Review the cancer genome atlas (tcga): an immeasurable source of knowledge. Contemporary Oncology (2015)

33. Vaidya, A., Zhang, A., Jaume, G., et al.: Molecular-driven foundation model for oncologic pathology. arXiv (2025)

34. Vorontsov, E., Bozkurt, A., Casson, A., et al.: A foundation model for clinical-grade computational pathology and rare cancers detection. Nature medicine (2024)

35. Wang, X., Zhao, J., Marostica, E., et al.: A pathology foundation model for cancer diagnosis and prognosis prediction. Nature (2024)

36. Wölflein, G., Ferber, D., Meneghetti, A.R., et al.: Benchmarking pathology feature extractors for whole slide image classification. arXiv (2023)

37. Xiang, J., Wang, X., Zhang, X., et al.: A vision–language foundation model for precision oncology. Nature (2025)

38. Xu, H., Usuyama, N., Bagga, J., et al.: A whole-slide foundation model for digital pathology from real-world data. Nature (2024)

39. Zhang, A., Jaume, G., Vaidya, A., et al.: Accelerating data processing and benchmarking of ai models for pathology. arXiv (2025)

40. Zhou, X., Sun, L., He, D., et al.: A knowledge-enhanced pathology vision-language foundation model for cancer diagnosis. arXiv (2024)

41. Zimmermann, E., Vorontsov, E., Viret, J., et al.: Virchow2: Scaling self-supervised mixed magnification models in pathology. arXiv (2024)

> 💡 **claude 批注｜Q&A 批注记录**:
> - Q: 作者最终是否主张跳过 slide-level evaluation？
> - A: 否。结论第一段明确把 tile benchmark 定位为 first step，并要求在 context-heavy tasks 和区分近邻模型时做 slide-level validation。
> - Q: 增加更多 tile 标签就一定改善迁移吗？
> - A: 不一定。作者强调的是 richer、context-aware 的数据质量；如果新增样本仍只覆盖局部形态，免疫／空间任务的监督错配仍在。需要按任务信号类型和病例结构做分层验证。

## 本节小结

- 最可靠用途：低成本 shortlist 编码器。
- 不可替代环节：目标 cohort 上的 slide-level 验证，尤其是上下文密集任务和名次接近的模型。
- ReadySlide 下一步：分别验证同一组 pipelines 的 full-bag→budgeted 迁移，以及同一组 selectors 跨非退化预算的排名稳定性，不能混用候选对象。
