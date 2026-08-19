[← 返回 README](../README.md)

# 数据、代码、参考文献与出版后记

## 📌 预览

本节完整保留 Data availability、Code availability、51 条参考文献，以及致谢、作者贡献、基金、利益冲突、补充材料、开放许可和作者单位后记。公开数据与受限临床队列必须分开理解。

---

## Data availability

WSIs from TCGA are publicly available through the Genomic Data Commons Data Portal (https://portal.gdc.cancer.gov/). CPTAC data are publicly available through the National Cancer Institute CPTAC resources and associated data portals (https://proteomics.cancer.gov/ data-portal). Molecular data for TCGA and CPTAC can be accessed through cBioPortal (https://www.cbioportal.org/). The PathoBench dataset is publicly available on Hugging Face (https://huggingface.co/ datasets/MahmoodLab/Patho-Bench). Patient-level data from DACHS, Kiel, Bern, IEO, and GECCO are third-party clinical datasets and are not publicly deposited because redistribution is constrained by the origi nal ethics approvals, consent conditions, applicable privacy law, and institutional or consortium data-use and transfer agreements. The restricted data comprise H&E whole-slide images and linked clinicopathological, biomarker, and/or molecular variables used in this study. Access is limited to qualified researchers acting through institutions that can enter the required agreements, and proposed use must be compatible with the approved scientific purpose and any cohort-specific data-use limitations. Initial responses to data access requests can generally be expected within several weeks; the total time to execute a data use or transfer agreement may be longer depending on institutional review processes at the requesting and data-holding institutions. Access duration is determined case by case by the data holder during review and contracting. The slides and biomarker data for DACHS were generated for prior studies<sup>46–48</sup> with restricted access.

DACHS biomarker and genotype data can be requested through dbGaP Authorized Access via the GECCO top-level study phs001078, with DACHS represented as sub-study phs001113.v1.p1 [https://www. ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study\_id=phs001113. v1.p1]. Applications for access to DACHS biomarker data are reserved for Senior Investigators and NIH Investigators as defined in https:// dbgap.ncbi.nlm.nih.gov/aa/wga.cgi, and upon successful application grants access to the data for 1 year with the option to renew access. The slides for DACHS can only be requested directly through the DACHS principal investigators. The contact details are listed at http://dachs. dkfz.org/dachs/kontakt.html. Kiel WSIs and linked clinicopathological data are held by the Department of Pathology, University Hospital Schleswig-Holstein, Kiel, Germany. Requests should be directed to the department through its official contact page (https://www.medizin. uni-kiel.de/en/institutes-departments/institutes-of-clinical-theory/ department-of-pathology). Bern whole-slide and linked clinicopathological data are held by the Institute of Tissue Medicine and Pathology, University of Bern, Switzerland; individual patient-level data are not publicly shared, and requests should be directed to the institute (contact.igmp@unibe.ch) in reference to ref. 49. IEO wholeslide and linked clinicopathological data are held by the European Institute of Oncology, Milan, Italy; requests are evaluated case by case under institutional policies and patient-privacy obligations and should be submitted through the institute’s official contact route (https:// www.ieo.it/en/contact\_us/). GECCO H&E WSIs and associated clinicopathological and molecular data used in this study are coordinated through the GECCO consortium at Fred Hutchinson Cancer Center; requests should be directed to the GECCO coordinating center (gecco@fredhutch.org) and may require approval consistent with the policies of the contributing studies (CORSA, EPIC, CRA, WHI, IWHS). Source data are provided with this paper.

## Code availability

All benchmarking experiments build upon the open-source STAMP-Benchmark software, available at https://github.com/KatherLab/ STAMP-Benchmark and archived at Zenodo under DOI 10.5281/ zenodo.15749283 (ref. 50), released under the MIT License. The implementation of EAGLE, including the tested slide encoders, the MLP and linear-probing classifiers, the GPT-4o in-context-learning evaluation, UMAP visualization, top-tile extraction, and slide-search analyses, is available at https://github.com/KatherLab/EAGLE and is archived at Zenodo under https://doi.org/10.5281/zenodo.19799127 (ref. 51), released under the GNU General Public License v3.0 (GPL-3.0). The repository includes the software license and citation metadata. Third-party foundation models used in this study are distributed under their respective licenses and access conditions. All experiments were conducted using NVIDIA RTX A6000, L40 or H100 GPUs.


> 💡 **数据与代码可复现性（claude 批注）**：TCGA、CPTAC 与 PathoBench 可公开获取；DACHS、Kiel、Bern、IEO、GECCO 的 WSI/临床变量受伦理、隐私与机构协议限制，申请可能需要数周或更久。代码同时给出 GitHub 与 Zenodo 固化版本：STAMP-Benchmark 为 MIT，EAGLE 为 GPL-3.0；“代码开放”不意味着所有第三方 foundation model 权重与临床数据自动具有相同许可。

## References

1. Coudray, N. et al. Classification and mutation prediction from nonsmall cell lung cancer histopathology images using deep learning. Nat. Med. 24, 1559–1567 (2018).

2. Kather, J. N. et al. Deep learning can predict microsatellite instability directly from histology in gastrointestinal cancer. Nat. Med. 25, 1054–1056 (2019).

3. Lu, M. Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat. Biomed. Eng. 5, 555–570 (2021).

4. Campanella, G. et al. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nat. Med. 25, 1301–1309 (2019).

5. Lipkova, J. et al. Artificial intelligence for multimodal data integration in oncology. Cancer Cell 40, 1095–1110 (2022).

6. Song, A. H. et al. Artificial intelligence for digital and computational pathology. Nat. Rev. Bioeng. 1, 930–949 (2023).

7. Chen, R. J. et al. Towards a general-purpose foundation model for computational pathology. Nat. Med. 30, 850–862 (2024).

8. Wang, X. et al. Transformer-based unsupervised contrastive learning for histopathological image classification. Med. Image Anal. 81, 102559 (2022).

9. Xu, H. et al. A whole-slide foundation model for digital pathology from real-world data. Nature 630, 181–188 (2024).

10. Vorontsov, E. et al. A foundation model for clinical-grade computational pathology and rare cancers detection. Nat. Med. 30, 2924–2935 (2024).

11. Lipkova, J. & Kather, J. N. The age of foundation models. Nat. Rev. Clin. Oncol. 21, 769–770 (2024).

12. Lu, M. Y. et al. A multimodal generative AI copilot for human pathology. Nature 634, 466–473 (2024).

13. Derraz, B. et al. New regulatory thinking is needed for AI-based personalised drug and cell therapies in precision oncology. NPJ Precis. Oncol. 8, 23 (2024).

14. Tran, M. et al. Generating dermatopathology reports from gigapixel whole slide images with HistoGPT. Nat. Commun. 16, 4886 (2025).

15. Ferber, D. et al. In-context learning enables multimodal large language models to classify cancer pathology images. Nat. Commun. 15, 10104 (2024).

16. Clusmann, J. et al. The future landscape of large language models in medicine. Commun. Med. 3, 141 (2023).

17. Chen, R. J. et al. Scaling vision Transformers to gigapixel images via hierarchical self-supervised learning. In Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 16123–16134 (IEEE Computer Society Conference Publishing Services, 2022).

18. El Nahhas, O. S. M. et al. From whole-slide image to biomarker prediction: end-to-end weakly supervised deep learning in computational pathology. Nat. Protoc. 20, 293–316 (2025).

19. Frascarelli, C. et al. Deep learning algorithm on H&E whole slide images to characterize TP53 alterations frequency and spatial distribution in breast cancer. Comput. Struct. Biotechnol. J. 23, 4252–4259 (2024).

20. Shmatko, A., Ghaffari Laleh, N., Gerstung, M. & Kather, J. N. Artificial intelligence in histopathology: enhancing cancer research and clinical oncology. Nat. Cancer 3, 1026–1038 (2022).

21. Ilse, M., Tomczak, J. M. & Welling, M. Attention-based deep multiple instance learning. In Proc. 35th International Conference on Machine Learning (eds Dy, J. & Krause, A.) 2127–2136 (PMLR, 2018).

22. Neidlinger, P. et al. Benchmarking foundation models as feature extractors for weakly supervised computational pathology. Nat. Biomed. Eng. https://doi.org/10.1038/s41551-025-01516-3 (2025).

23. Wang, X. et al. A pathology foundation model for cancer diagnosis and prognosis prediction. Nature 634, 970–978 (2024).

24. Zimmermann, E. et al. Virchow2: Scaling self-supervised mixed magnification models in pathology. arXiv Preprint https://doi.org/ 10.48550/arXiv.2408.00738 (2024).

25. Kather, J. N. et al. Pan-cancer image-based detection of clinically actionable genetic alterations. Nat. Cancer 1, 789–799 (2020).

26. Shaikovski, G. et al. PRISM: a multi-modal generative foundation model for slide-level histopathology. arXiv Preprint https://doi.org/ 10.48550/arXiv.2405.10254 (2024).

27. Jaume, G. et al. Multistain pretraining for slide representation learning in pathology. In Computer Vision—ECCV 2024 19–37 (Springer, 2024).

28. Lu, M. Y. et al. A visual-language foundation model for computational pathology. Nat. Med. 30, 863–874 (2024).

29. Lenz, T. et al. Unsupervised foundation model-agnostic slide-level representation learning. In Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 30807–30817 (IEEE Computer Society Conference Publishing Services, 2025).

30. Ding, T. et al. A multimodal whole-slide foundation model for pathology. Nat. Med. 31, 3749–3761 (2025).

31. Vaidya, A. et al. Molecular-driven foundation model for oncologic pathology. arXiv Preprint https://doi.org/10.48550/arXiv.2501. 16652 (2025).

32. Zhang, A., Jaume, G., Vaidya, A., Ding, T. & Mahmood, F. Accelerating data processing and benchmarking of AI models for pathology. arXiv Preprint https://doi.org/10.48550/arXiv.2502. 06750 (2025).

33. McInnes, L., Healy, J., Saul, N. & Großberger, L. UMAP: Uniform Manifold Approximation and Projection. J. Open Source Softw. 3, 861 (2018).

34. Shang, H. H. et al. Histopathology slide indexing and search—are we there yet? NEJM AI 1, AIcs2300019 (2024).

35. Wang, X. et al. RetCCL: clustering-guided contrastive learning for whole-slide image retrieval. Med. Image Anal. 83, 102645 (2023).

36. Clusmann, J. et al. Prompt injection attacks on vision language models in oncology. Nat. Commun. 16, 1239 (2025).

37. Carr, P. R. et al. Estimation of absolute risk of colorectal cancer based on healthy lifestyle, genetic risk, and colonoscopy status in a population-based study. Gastroenterology 159, 129–138.e9 (2020).

38. Hoffmeister, M. et al. Colonoscopy and reduction of colorectal cancer risk by molecular tumor subtypes: a population-based casecontrol study. Am. J. Gastroenterol. 115, 2007–2016 (2020).

39. Brenner, H., Chang-Claude, J., Seiler, C. M., Stürmer, T. & Hoffmeister, M. Does a negative screening colonoscopy ever need to be repeated? Gut 55, 1145–1150 (2006).

40. dbGaP Study. https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/ study.cgi?study\_id=phs001078.v1.p1.

41. Gustav, M. et al. Assessing genotype-phenotype correlations in colorectal cancer with deep learning: a multicentre cohort study. Lancet Digit. Health 7, 100891 (2025).

42. Canny, J. A computational approach to edge detection. IEEE Trans. Pattern Anal. Mach. Intell. 8, 679–698 (1986).

43. Oquab, M. et al. DINOv2: Learning Robust Visual Features without Supervision. Trans. Mach. Learn. Res. https://openreview.net/ forum?id=a68SUt6zFt (2024).

44. Yu, J. et al. CoCa: Contrastive Captioners are Image-Text Foundation Models. Trans. Mach. Learn. Res. https://openreview.net/ forum?id=Ee277P3AYC (2022).

45. Selvaraju, R. R. et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. Int. J. Comput. Vis. 128, 336–359 (2020).

46. Lilla, C. et al. Effect of NAT1 and NAT2 genetic polymorphisms on colorectal cancer risk associated with exposure to tobacco smoke and meat consumption. Cancer Epidemiol. Biomark. Prev. 15, 99–107 (2006).

47. Brenner, H., Chang-Claude, J., Seiler, C. M. & Hoffmeister, M. Longterm risk of colorectal cancer after negative colonoscopy. J. Clin. Oncol. 29, 3761–3767 (2011).

48. Hoffmeister, M. et al. Statin use and survival after colorectal cancer: the importance of comprehensive confounder adjustment. J. Natl. Cancer Inst. 107, djv045 (2015).

49. Dislich, B., Blaser, N., Berger, M. D., Gloor, B. & Langer, R. Preservation of Epstein-Barr virus status and mismatch repair protein status along the metastatic course of gastric cancer. Histopathology 76, 740–747 (2020).

50. Neidlinger, P. et al. STAMP-Benchmark. Source code. Zenodo https://doi.org/10.5281/zenodo.15749283 (2025).

51. Neidlinger, P. et al. EAGLE. Source code. Zenodo https://doi.org/10. 5281/zenodo.19799127 (2026).


> 💡 **参考文献脉络（claude 批注）**：方法主线由弱监督 WSI（refs. 1–4, 17–21）、病理 foundation model（refs. 7–10, 23–30）、PathoBench（refs. 31–32）、可解释与检索（refs. 33–36, 45）以及队列/数据来源（refs. 37–41, 46–49）组成。refs. 50–51 是可执行复现入口；与只按标题相似度排序相比，这组功能分组更能解释 EAGLE 的 selector、consumer、benchmark 和审计协议来自哪里。

## Acknowledgements

We kindly thank all individuals who agreed to participate in the CORSA study. Furthermore, we thank all cooperating physicians and students and the Biobank Graz of the Medical University of Graz. We also acknowledge the TCGA Research Network and the Clinical Proteomic

Tumor Analysis Consortium (CPTAC), which generated the data on which some of the results shown in this study are based. The authors thank the WHI investigators and staff for their dedication, and the study participants for making the program possible. A full listing of WHI investigators can be found at: https://s3-us-west-2.amazonaws.com/ www-whi-org/wp-content/uploads/WHI-Investigator-Long-List.pdf. The authors gratefully acknowledge GWK support through computing time provided by the Center for Information Services and HPC (ZIH) at TU Dresden. The authors gratefully acknowledge the Gauss Centre for Supercomputing e.V. for computing time provided through the John von Neumann Institute for Computing (NIC) on the GCS supercomputer JUWELS at Jülich Supercomputing Centre (JSC).

## Author contributions

P.N., T.L. and J.N.K. designed the study. P.N. and T.L. developed the software. P.N., M.G., R.L., B.D., L.A.B., A.J.F., E.L.G., A.G., S.B., M.J.G., R.S., H.M.B., C.R., T.H., U.P., A.I.P., G.C., N.F., A.M., M.H., H.B. and J.N.K. contributed to data collection and assembly. P.N., T.L., S.F., C.M.L.L., J.C., L.A.S. and J.N.K. analyzed and interpreted the data. P.N., T.L. and J.N.K. drafted the manuscript. All authors reviewed the manuscript, approved the final version for submission, and agree to be accountable for their own contributions and to ensure that questions related to the accuracy or integrity of any part of the work are appropriately investigated, resolved, and documented.

## Funding

J.N.K. is supported by the German Cancer Aid DKH (DECADE, 70115166), the German Federal Ministry of Research, Technology and Space BMFTR (PEARL, 01KD2104C; CAMINO, 01EO2101; TRANSFORM LIVER, 031L0312A; TANGERINE, 01KT2302 through ERA-NET Transcan; Come2Data, 16DKZ2044A; DEEP-HCC, 031L0315A; DECIPHER-M, 01KD2420A; NextBIG, 01ZU2402A; PROSURV, 01KD2509C), the German Research Foundation (DFG, Deutsche Forschungsgemeinschaft) as part of Germany’s Excellence Strategy, EXC 2050/2, Project ID 390696704, Cluster of Excellence “Centre for Tactile Internet with Human-in-the-Loop” (CeTI) of Technische Universität Dresden, as well as through DFG-funded collaborative research projects (TRR 412/1, 535081457; SFB 1709/1 2025, 533056198), the German Academic Exchange Service DAAD (SECAI, 57616814), the German Federal Joint Committee G-BA (TransplantKI, 01VSF21048), the European Union Horizon Europe research and innovation programme (ODELIA, 101057091; GENIAL, 101096312), the European Research Council ERC (NADIR, 101114631), the Breast Cancer Research Foundation (BELLA-DONNA, BCRF-25-225), and the National Institute for Health and Care Research NIHR (Leeds Biomedical Research Centre, NIHR203331). The views expressed are those of the author(s) and not necessarily those of the NHS, the NIHR, or the Department of Health and Social Care. This work was funded by the European Union. Views and opinions expressed are, however, those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them. S.F. is supported by the German Federal Ministry of Education and Research (SWAG, 01KD2215C), the German Cancer Aid (DECADE, 70115166; TargHet, 70115995), and the German Research Foundation (504101714). The DACHS study was supported by the German Research Council (BR 1704/6-1, BR 1704/6-3, BR 1704/6-4, CH 117/1-1, HO 5117/2-1, HO 5117/2-2, HE 5998/2-1, HE 5998/2-2, KL 2354/3-1, KL 2354/3-2, RO 2270/8-1, RO 2270/8-2, BR 1704/17-1, BR 1704/17-2); the Interdisciplinary Research Program of the National Center for Tumor Diseases (NCT), Germany; and the German Federal Ministry of Education and Research (01KH0404, 01ER0814, 01ER0815, 01ER1505A, 01ER1505B, and 01KD2104A). J.C. is supported by the Mildred-Scheel-Postdoktorandenprogramm of the German Cancer Aid (70115730). A.M. is supported by the European Society for Medical Oncology José Baselga Fellowship for Clinician Scientists founded by AstraZeneca

(2023–2025). The Genetics and Epidemiology of Colorectal Cancer Consortium (GECCO) is funded by the National Cancer Institute, National Institutes of Health, U.S. Department of Health and Human Services (U01 CA137088, R01 CA488857, P20 CA252733, P50 CA285275). Genotyping and sequencing services were provided by the Center for Inherited Disease Research (CIDR) under contract HHSN268201700006I. This research was funded in part through the NIH/NCI Cancer Center Support Grant P30 CA015704. Scientific computing infrastructure at Fred Hutch was funded by ORIP grant S10OD028685. The CORSA study was funded by Austrian Research Funding Agency (FFG) BRIDGE (grant 829675, to A.G.), the Herzfel der’sche Familienstiftung (grant to A.G.), and COST Action BM1206 CRA was supported by the National Institutes of Health grant R01 CA068535. The coordination of EPIC is financially supported by the International Agency for Research on Cancer (IARC) and by the Department of Epidemiology and Biostatistics, School of Public Health Imperial College London, which has additional infrastructure support from the NIHR Imperial Biomedical Research Centre. The nationa cohorts are supported by the Danish Cancer Society (Denmark); Ligue Contre le Cancer, Institut Gustave Roussy, Mutuelle Générale de l’Education Nationale, and Institut National de la Santé et de la Recherche Médicale (INSERM) (France); German Cancer Aid, German Cancer Research Center (DKFZ), German Institute of Human Nutrition Potsdam-Rehbruecke (DIfE), and the Federal Ministry of Education and Research (BMBF) (Germany); Associazione Italiana per la Ricerca sul Cancro, Compagnia di SanPaolo, and the National Research Council (Italy); the Dutch Ministry of Public Health, Welfare and Sports (VWS) Netherlands Cancer Registry (NKR), LK Research Funds, Dutch Prevention Funds, Dutch ZON (Zorg Onderzoek Nederland), World Cance Research Fund (WCRF), and Statistics Netherlands (The Netherlands); Health Research Fund (FIS), Instituto de Salud Carlos III (ISCIII), Regional Governments of Andalucía, Asturias, Basque Country, Murcia, and Navarra, and the Catalan Institute of Oncology (ICO) (Spain) Swedish Cancer Society, Swedish Research Council, Region Skåne, and Region Västerbotten (Sweden); and Cancer Research UK (14136 to EPIC-Norfolk; C8221/A29017 to EPIC-Oxford) and the Medical Research Council (1000143 to EPIC-Norfolk: MR/M012190/1 to EPIC-Oxford (United Kingdom). The IWHS study was supported by NIH grants CA107333 (R01 grant awarded to P.J. Limburg) and HHSN261201000032C (N01 contract awarded to the University of Iowa). The WHI program is funded by the National Heart, Lung, and Blood Institute, National Institutes of Health, U.S. Department of Health and Human Services through contracts 75N92021D00001 75N92021D00002, 75N92021D00003, 75N92021D00004, and 75N92021D00005. This work was partially supported by the Italian Ministry of Health through Ricerca Corrente 5 × 1000 funds; the Italian Ministry of Innovations via the Sustainable Growth Fund, Innovation Agreements under the Ministerial Decree of December 31, 2021, and the Director’s Decree of November 14, 2022 (2nd Call), Project No. F/ 350104/01-02/X60; and the Italian Ministry of University and Research (MUR) 2023 through the Future Artificial Intelligence Research (FAIR) program, PE0000013, CUP D53C22002380006, within the National Recovery and Resilience Plan (PNRR), Mission 4, Component 2, Investment 1.3, funded by the European Union, NextGenerationEU. Project: AIDH, Adaptive AI Methods for Digital Health. Where authors are identified as personnel of the International Agency for Research on Cancer or World Health Organization, the authors alone are responsible for the views expressed in this article and they do not necessarily represent the decisions, policy, or views of the International Agency for Research on Cancer or World Health Organization. Open Access funding enabled and organized by Projekt DEAL.

## Competing interests

J.N.K. reports ongoing consulting for AstraZeneca and Bioptimus; equity in StratifAI, Synagen, Tremont, Saterra and Spira Labs; institutional research grants from GSK and AstraZeneca; and honoraria from Astra-Zeneca, Bayer, Daiichi Sankyo, Eisai, Janssen, Merck, MSD, BMS, Roche, Pfizer, and Fresenius. S.F. reports honoraria from MSD and BMS. M.G. reports honoraria for lectures sponsored by Techniker Krankenkasse and AstraZeneca. R.L. reports consulting and honoraria from MSD, Janssen, AstraZeneca, Astellas, and Roche. U.P. reports consulting for AbbVie; in addition, her husband holds individual stocks in BioNTech SE ADR, Amazon, CureVac, NanoString Technologies, Alphabet, NVIDIA, and Microsoft. A.M. reports honoraria as a consultant, advisor, or speaker from Roche, Lilly, and Menarini/Stemline, and support for accommodation and travel from AstraZeneca. N.F. reports consulting or advisory roles for MSD, Merck, Novartis, AstraZeneca, Sysmex, Roche, Menarini Group, Gilead, Veracyte, Sakura, and AbbVie; speaker fees from MSD, Novartis, AstraZeneca, Daiichi Sankyo, Sysmex, GSK, Gilead, Roche, Menarini, Leica Biosystems, Thermo Fisher, Genomic Health, Veracyte, and Lilly; research grants from Novartis, Gilead, AstraZeneca, GSK, and Pfizer; and travel grants from Roche and Novartis. All other authors declare no competing interests.

## Additional information

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s41467-026-74918-9.

Correspondence and requests for materials should be addressed to Jakob Nikolas Kather.

Peer review information Nature Communications thanks the anonymous reviewer(s) for their contribution to the peer review of this work. A peer review file is available.

Reprints and permissions information is available at http://www.nature.com/reprints

Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/ licenses/by/4.0/.

© The Author(s) 2026

> 💡 **MinerU 作者单位修复（claude 批注）**：`full.md` 只保留了第 5–23 单位的跨页续段；下面先补入 `content_list.json` 的 page footnote 原文，再与续段衔接，从而恢复第 1–23 单位。

A full list of affiliations appears at the end of the paper. e-mail: jakob-nikolas.kather@alumni.dkfz.de

<sup>1</sup>Else Kroener Fresenius Center for Digital Health, Faculty of Medicine and University Hospital Carl Gustav Carus, TUD Dresden University of Technology, Dresden, Germany. <sup>2</sup>Institute of Pathology, University Medical Center Mainz, Mainz, Germany. <sup>3</sup>Department of Medicine I, Faculty of Medicine and University Hospital Carl Gustav Carus, TUD Dresden University of Technology, Dresden, Germany. <sup>4</sup>National Center for Tumor Diseases Dresden (NCT/UCC),

Dresden, Germany. <sup>5</sup>Department of Medicine III, University Hospital RWTH Aachen, Aachen, Germany. <sup>6</sup>Institute of Pathology and Molecular Pathology, Kepler University Hospital, Johannes Kepler University Linz, Linz, Austria. <sup>7</sup>Institute of Tissue Medicine and Pathology, University of Bern, Bern, Switzerland. <sup>8</sup>Division of Gastroenterology and Hepatology, Mayo Clinic, Rochester, MN, USA. <sup>9</sup>Division of Laboratory Genetics, Department of Laboratory Medicine and Pathology, Mayo Clinic, Rochester, MN, USA. <sup>10</sup>Department of Quantitative Health Sciences, Division of Epidemiology, Mayo Clinic, Rochester, MN, USA. <sup>11</sup>Center for Cancer Research, Medical University of Vienna, Vienna, Austria. <sup>12</sup>Nutrition and Metabolism Branch, International Agency for Research on Cancer, World Health Organization, Lyon, France. <sup>13</sup>Cancer Epidemiology and Prevention Research Unit, School of Public Health, Imperial College London, London, UK. <sup>14</sup>Division of Public Health Sciences, Fred Hutchinson Cancer Center, Seattle, WA, USA. <sup>15</sup>Department of Pathology, University Hospital Schleswig-Holstein, Kiel, Germany. <sup>16</sup>Department of Epidemiology, University of Washington, Seattle, WA, USA. <sup>17</sup>Division of New Drugs and Early Drug Development, European Institute of Oncology IRCCS, Milan, Italy. <sup>18</sup>Department of Oncology and Hemato-Oncology, University of Milan, Milan, Italy. <sup>19</sup>Division of Pathology, European Institute of Oncology IRCCS, Milan, Italy. <sup>20</sup>Division of Clinical Epidemiology and Aging Research, German Cancer Research Center (DKFZ), Heidelberg, Germany. <sup>21</sup>German Cancer Consortium (DKTK), German Cancer Research Center (DKFZ), Heidelberg, Germany. <sup>22</sup>Medical Oncology, National Center for Tumor Diseases (NCT), University Hospital Heidelberg, Heidelberg, Germany. <sup>23</sup>Pathology & Data Analytics, Leeds Institute of Medical Research at St James’s, University of Leeds, Leeds, UK. e-mail: jakob-nikolas.kather@alumni.dkfz.de

> 💡 **利益冲突与使用边界（claude 批注）**：作者披露多项咨询、股权、研究资助和讲者关系；这不自动否定结果，但临床转化判断应结合外部独立复现。CC BY 4.0 允许在署名与标注修改的条件下复用论文内容，第三方材料仍需检查独立权利说明。

## 🔖 本节小结

- **公开入口**：TCGA、CPTAC、PathoBench、STAMP-Benchmark、EAGLE GitHub/Zenodo。
- **受限入口**：DACHS、Kiel、Bern、IEO、GECCO 需按队列单独申请。
- **参考文献数**：正文列出 51 条。
- **出版信息**：Nature Communications 17:5740，2026 年 7 月 1 日在线发表，CC BY 4.0。
