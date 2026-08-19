# ReadySlideBenchmark 论文批读设计

## 目标

在 `topics/ReadySlideBenchmark/` 建立面向病理 FM Selector–Consumer 角色解耦 benchmark 的五篇核心论文批读集，并将论文证据组织为 ReadySlide 的写作、实验和 novelty 参照。

## 范围

- 完整复制现有 EAGLE 批读资产到新 topic，并补充 Selector–Consumer 专项分析。
- 完整批读 arXiv:2606.10778、2605.12575、2601.10073、2605.17456。
- 每篇包含 MinerU 原始资产、按原论文大分节拆分的原文与中文 `claude 批注`、中文 README、Mermaid 数据流、Q&A 和 Citation Landscape。
- 新建 topic README，综合五篇论文的角色矩阵、novelty 约束、阅读顺序和 ReadySlide 实验启示。
- 更新根 README，完成链接、图片、公式、原文完整性和修改范围 QA 后推送 GitHub `main`。

## 组织方式

```text
topics/ReadySlideBenchmark/
├── README.md
├── [MICCAI 2026] From-Patches-to-Patients/
├── [Nat Commun 2026] EAGLE/
├── [Arxiv 2026] FOCI/
├── [WACV Workshop 2026] ReaMIL/
└── [Arxiv 2026] GCE-MIL/
```

EAGLE 保留完整独立副本，使新 topic 可独立阅读；新增批注聚焦 CHIEF selector、Virchow2 consumer、固定配对、预算和部署效率。其余论文从 arXiv PDF 经 MinerU 解析后完整批读。

## 质量与发布

- 不修改其他 topic 的既有论文内容。
- 子任务按论文目录隔离；公共 README、QA、提交和推送由主线程统一完成。
- 分论文提交，最终综合 README 单独提交；远端 `main` SHA 必须与本地 HEAD 一致。
