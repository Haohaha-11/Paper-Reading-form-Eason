[← 返回 README](../README.md)

# 摘要与论文元信息

## 📌 预览

EAGLE 把全切片计算拆成“全图廉价粗筛”和“少量区域昂贵精提”：固定用 CHIEF 从 CTransPath 特征中选出 25 个 tile，再交给 Virchow2 编码并等权平均。摘要给出的主张是 43 个任务、9 个癌种、单张 WSI 2.27 秒，以及相对既有模型超过 99% 的时间缩减；后文的负对照、注意力集中分析和外部验证决定这些主张是否站得住。

---

Article

# A deep learning framework for efficient pathology image analysis

https://doi.org/10.1038/s41467-026-74918-9

Received: 14 October 2025

Accepted: 16 June 2026

Published online: 01 July 2026

Check for updates

Peter Neidlinger <sup>1</sup>, Tim Lenz<sup>1</sup>, Sebastian Foersch <sup>2</sup>, Chiara M. L. Loeffler<sup>1,3,4</sup>, Jan Clusmann<sup>1,5</sup>, Marco Gustav <sup>1</sup>, Lawrence A. Shaktah <sup>1</sup>, Rupert Langer<sup>6</sup>, Bastian Dislich<sup>7</sup>, Lisa A. Boardman <sup>8</sup>, Amy J. French<sup>9</sup>, Ellen L. Goode <sup>10</sup>, Andrea Gsur <sup>11</sup>, Stefanie Brezina <sup>11</sup>, Marc J. Gunter<sup>12,13</sup>, Robert Steinfelder <sup>14</sup>, Hans-Michael Behrens<sup>15</sup>, Christoph Röcken <sup>15</sup>, Tabitha Harrison <sup>14,16</sup>, Ulrike Peters <sup>14,16</sup>, Amanda I. Phipps<sup>14,16</sup>, Giuseppe Curigliano <sup>17,18</sup>, Nicola Fusco <sup>18,19</sup>, Antonio Marra <sup>17</sup>, Michael Hoffmeister <sup>20</sup>, Hermann Brenner <sup>20,21</sup> & Jakob Nikolas Kather <sup>1,3,22,23</sup>

## 摘要原文

Artificial intelligence has transformed digital pathology by enabling biomarker prediction from high-resolution whole-slide images. However, current methods are computationally inefficient, processing thousands of redundant tiles per slide and requiring complex aggregation models. We introduce EAGLE (Efficient Approach for Guided Local Examination), a deep learning framework that emulates pathologists by selectively analyzing informative regions. EAGLE combines task-agnostic tile selection with detailed feature extraction and is benchmarked against leading slide- and tile-level foundation models across 43 tasks from nine cancer types spanning morphology, biomarker prediction, treatment response and prognosis. EAGLE outperforms patch aggregation methods by up to 23% and achieves the highest overall classification performance. It processes one slide in 2.27 s, reducing computational time by more than 99% compared with existing models. This efficiency supports rapid and auditable workflows by enabling review of the exact tiles used for each prediction and reducing dependence on high-performance computing. By reliably identifying informative regions and minimizing artifacts, EAGLE provides robust and auditable outputs, supported by systematic negative controls and attention concentration analyses. Its unified embedding enables rapid slide search, integration into multi-omics pipelines and emerging clinical foundation models.

> 💡 **问题动机与研究主张（claude 批注）**：摘要把三个目标绑在一起：准确性、效率和可审计性。真正的新意不是单独再训练一个更大的编码器，而是把“哪里值得看”与“如何精细表征”分给两个冻结基础模型。对 ReadySlideBenchmark 而言，这相当于把可传输预算直接压到 25 个离散区域；它证明选择式压缩可能同时减少冗余与提升判别信号，但并没有证明该 selector 对任意 consumer 都最优。

> 💡 **关键数字核对（claude 批注）**：43 个任务由主 benchmark 的 31 个任务与 PathoBench 的 12 个任务组成；9 个癌种来自主 benchmark 的 4 种加 PathoBench 扩展癌种。2.27 秒由 2.01 秒 CTransPath 粗提、0.36 毫秒 CHIEF 排序和 0.26 秒 Virchow2 top-25 精提近似相加，口径是 GPU 模型推理，不等于包含切片读取、tessellation 与磁盘 I/O 的完整临床周转时间。

## 🔖 本节小结

- **核心对象**：9,528 张 WSI、43 个任务、9 个癌种。
- **关键接口**：CHIEF selector → Virchow2 consumer，预算固定为 top 25。
- **需要追问**：超过 99% 的节省是否在统一 I/O、缓存与预处理口径下仍成立；固定配对能否推广到 selector×consumer 全交叉。
