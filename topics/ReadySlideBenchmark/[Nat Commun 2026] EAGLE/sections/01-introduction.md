[← 返回 README](../README.md)

# 引言

## 📌 预览

引言从三项现实摩擦出发：每张 WSI 平均约 18,000 个 tile 的计算负担、每个任务都要重训聚合器的扩展成本，以及弱监督小样本下不稳定的区域选择。EAGLE 的回答是先用跨任务预训练先验给全图排序，再只让强 tile encoder 看最有价值的 25 个区域。

---

Artificial intelligence (AI) has significantly advanced computational pathology (CPath) by enabling the extraction of clinically relevant information from gigapixel-scale whole-slide images (WSIs)<sup>1–6</sup>. Existing methods use resource-intensive vision transformers trained with selfsupervised learning (SSL) to encode detailed morphological features essential for diagnosis, prognosis, and treatment planning in oncology<sup>7–11</sup>. While these approaches have shown great promise across a wide range of tasks, their inefficiencies and limited scalability highlight the need for solutions that better align with real-world diagnostic workflows. Recently, pathology-specific multimodal large language models (MLLMs) have emerged as AI copilots for clinical decision making, but they often underperform in biomarker prediction and the regulatory pathway for approving such models as medical devices remains uncertain<sup>12–16</sup>.

Current methods predominantly operate at the tile level, requiring the extraction and analysis of thousands of tiles per WSI, with datasets in this study averaging approximately 18,000 tiles per slide at a resolution of 0.5 µm/pixel (MPP). This computationally intensive process deviates from how pathologists evaluate slides, as they selectively focus on regions ofinterest<sup>17–19</sup>. Moreover, tile-wise features are aggregated into slide-level predictions using models trained separately for each task, limiting scalability and interpretability<sup>8,20,21</sup>. The complexity of these models often obscures the decision-making process, making it challenging to understand how predictions are derived and which tissue regions are influential. These systems also struggle in data-scarce scenarios, where tile selection often fails to identify the most relevant regions, leading to suboptimal predictions<sup>22</sup>. Such scenarios are often a clinical reality, for example during the evaluation of small biopsy specimens.

To address these limitations, we developed EAGLE (Efficient Approach for Guided Local Examination), a framework that emulates the diagnostic strategy of pathologists by focusing on a small, informative subset of tiles within WSIs. EAGLE combines CHIEF<sup>23</sup>, a pretrained and task-agnostic model used for global tissue representation and guided tile selection, with Virchow2<sup>24</sup>, for detailed feature extraction from selected tiles. This combination substantially reduces computational demands while increasing performance (Fig. 1a–c). By selecting a small, reproducible subset of regions, EAGLE enhances auditability and scalability, particularly in biomarker prediction tasks where subtle morphological features are critical<sup>25</sup>. Unlike MLLMs, which emphasize multimodal interaction, EAGLE prioritizes efficient high-quality WSI analysis. Still, it can integrate with MLLMs to provide valuable inputs for enhanced contextual analysis. Through comprehensive evaluation against state-of-the-art models, including multiple instance learning (MIL) and slide-encoder approaches, we demonstrate the efficacy and robustness of EAGLE across 43 tasks spanning nine cancer types<sup>8–10,23,24,26–30</sup>.

> 💡 **痛点到设计的映射（claude 批注）**：计算低效对应“只让 Virchow2 看 25 个 tile”；逐任务聚合对应“一次生成 task-agnostic slide embedding”；弱监督注意力在小样本下不稳对应“使用冻结 CHIEF 的预训练显著性”。这三项不是并列装饰，而是同一数据流上粗筛、精提、复用三个阶段。

![Figure 1](../images/5a53db1170c9759376b0b8728d6db6cf4eb55a725b74e99159ab19d2da330b41.jpg)

*Fig. 1 | EAGLE framework. a Main benchmark overview: 9,528 whole-slide images (WSIs) from 13 cohorts spanning four cancer types. b Workflow comparison of EAGLE and conventional supervised pipelines. After tessellation, EAGLE applies CTransPath feature extraction, CHIEF tile selection, and Virchow2 encoding of 25 selected tiles to generate one averaged WSI embedding per patient. Supervised pipelines encode all tiles and aggregate them with task-specific models. Processing time per WSI is shown for EAGLE at 2 microns per pixel (MPP) and conventional supervised models at 0.5 MPP. c Mean AUROC across 31 computational pathology tasks for EAGLE, CHIEF, Prov-GigaPath, CTransPath, and Virchow2. Axes are normalized from 0.5 to the best AUROC for each task. d Example applications ofWSI or patient embeddings, including classification, biomarker prediction, prognosis, retrieval, and multi-omics integration. Source data are provided as a Source Data file.*

> 💡 **Figure 1 批读（claude 批注）**：图 1b 是整篇论文最重要的成本边界。常规管线对全部 tile 运行强编码器并为每个任务训练聚合器；EAGLE 先以 CTransPath@2 MPP 扫全图，再用 CHIEF 的 attention ranking 选 top 25，最后才调用 Virchow2。图 1c 显示组合表示优于 CHIEF、CTransPath 和 Virchow2 单独使用，但这仍是固定的 CHIEF→Virchow2 配对证据，而不是角色互换或全交叉结果。图 1d 则说明统一 embedding 的价值不止分类，还包括检索、多组学和预后。

> 💡 **Selector–Consumer 专项（claude 批注）**：selector 并非直接读取原始 WSI，它消费 CTransPath 粗特征；consumer 也不是接收 CHIEF embedding，而是重新读取 CHIEF 排名前 25 的原始区域并用 Virchow2 编码。因而 selector 成本必须包括 CTransPath 的 2.01 秒，不能只报 CHIEF 的 0.36 毫秒。

## 🔖 本节小结

- **输入规模**：约 18,000 tile/WSI（0.5 MPP）。
- **核心中间量**：CHIEF attention vector 与 top-25 tile 坐标。
- **输出**：25 个 Virchow2 embedding 的等权均值，可供多个下游任务复用。
