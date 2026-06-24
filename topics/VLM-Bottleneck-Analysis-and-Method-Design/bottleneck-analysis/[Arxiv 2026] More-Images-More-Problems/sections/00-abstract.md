[<- 返回 README](../README.md)

# Abstract

## 一、Preview

摘要精炼地交代了论文的三层逻辑：(1) 多图 LVLM 缺乏深度诊断 -> 提出 MIMIC benchmark；(2) MIMIC 诊断揭示 6 大 failure mode -> 根源是"单图模型行为"；(3) 提出数据侧 + 优化侧两种微调策略解决 -> 刷新多图 benchmark SOTA。核心叙事："发现问题 -> 分析根因 -> 针对性解决"。

---

## 二、原始文本

Large Vision Language Models (LVLMs) have demonstrated remarkable capabilities, yet their proficiency in understanding and reasoing over multiple images remains largely unexplored. While existing benchmarks have initiated the evaluation of multi-image models, a comprehensive analysis of their core weaknesses and their causes is still lacking. In this work, we introduce MIMIC (Multi-Image Model Insights and Challenges), a new benchmark designed to rigorously evaluate the multi-image capabilities of LVLMs. Using MIMIC, we conduct a series of diagnostic experiments that reveal pervasive issues: LVLMs often fail to aggregate information across images and struggle to track or attend to multiple concepts simultaneously. To address these failures, we propose two novel complementary remedies. On the data side, we present a procedural data-generation strategy that composes single-image annotations into rich, targeted multi-image training examples. On the optimization side, we analyze layer-wise attention patterns and derive an attention-masking scheme tailored for multi-image inputs. Experiments substantially improved cross-image aggregation, while also enhancing performance on existing multi-image benchmarks, outperforming prior state of the art across tasks. Data and code will be made available at https://github.com/anurag-198/MIMIC.

> **一句话概括**: MIMIC 通过程序化可控的多图评测诊断出 LVLM"无法跨图聚合信息 + 无法追踪多概念"的核心缺陷，进而提出合成多图训练数据（数据侧）和层间注意力掩码（优化侧）两种互补微调策略，在多个 benchmark 上刷新 SOTA。

---

## 三、Summary

- **核心问题**: 当前 LVLM 在多图场景的能力缺乏系统诊断，现有 benchmark 未控制混淆变量
- **核心方案**: MIMIC benchmark（诊断） + 合成数据微调（数据侧） + 注意力掩码微调（优化侧）
- **核心发现**: LVLM 在多图场景下的本质问题是"单图模型行为"
- **核心贡献**: 首次对多图 LVLM 进行可控、解耦的系统性 failure mode 分析 + 两种有效的针对性解决方案
