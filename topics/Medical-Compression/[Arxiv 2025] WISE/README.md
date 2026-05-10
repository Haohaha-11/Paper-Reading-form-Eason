# WISE: A Framework for Gigapixel Whole-Slide-Image Lossless Compression

**作者**: Yu Mao; Jun Wang; Nan Guan; Chun Jason Xue  
**机构**: MBZUAI; City University of Hong Kong  
**会议/期刊**: arXiv | **年份**: 2025  
**链接**: [arXiv Abs](https://arxiv.org/abs/2503.18074) | [PDF](https://arxiv.org/pdf/2503.18074)

## 一句话总结

WISE 是一个面向 gigapixel whole-slide image 的无损压缩框架：先编码空白区域，再用层次投影、bitmap 重排和 LZW 字典编码，把 WSI 中高频、强波动、难预测的信息重新组织成更适合字典压缩的 byte stream。

## 核心贡献

1. 首次系统研究 WSI 场景下的无损压缩问题，指出通用 PNG、Huffman、Arithmetic、Pfor 以及部分 learned lossless compressor 在 WSI informative region 上压缩率不足。
2. 将 WSI 无损压缩失败原因归结为 **information irregularity**：高频信号分布广、局部极值多、相邻像素/频率模式波动大，使 entropy/prediction-based 方法难以稳定建模。
3. 提出 WISE 复合编码流程：空白区域预处理、hierarchical projection coding、bitmap encoding、LZW dictionary coding，核心不是学习重建，而是重排信息使字典更容易收集重复模式。
4. 在 C16、C17、TCGA、BNCB、DigestPath、IHC4BC 等 WSI 数据上验证，报告 gigapixel WSI 平均 36x、最高 136x 的压缩效果。
5. 额外给出吞吐与内存对比：在 CPU 上 WISE 对 patch 的吞吐为 15.1 MB/s、内存 0.1 GB，而 ArIB-BPS 为 18.73 KB/s、3.81 GB。

## 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题、information irregularity、WISE 主流程和 36x/136x 结果 |
| [01 - Introduction](sections/01-introduction.md) | WSI 存储/传输压力、为什么必须 lossless、贡献列表 |
| [02 - Related Work](sections/02-related-work.md) | lossy medical compression、patch-based lossless、byte-stream compressors 的边界 |
| [03 - WISE Framework](sections/03-wise-framework.md) | information irregularity、空白区预处理、层次投影、bitmap encoding、LZW |
| [04 - Experiments](sections/04-experiments.md) | 主结果、NN compressor 对比、效率、消融和 bitmap PSNR 经验验证 |
| [05 - Conclusion](sections/05-conclusion.md) | 论文结论、部署意义与未展开的问题 |
| [06 - Appendix / References](sections/06-appendix-references.md) | 参考文献与可追溯背景 |

## 关键数字

| 指标 | 数值 |
|------|------|
| WSI 原始规模 | 约 1-5 GB/slide；论文图注称未压缩约 3 GB/slide |
| WSI 分辨率示例 | 0.25 microns per pixel |
| gigapixel WSI 压缩率 | 平均 36x，最高 136x |
| 通用 learned lossless 在 non-empty area 的压缩 | 约 1.6x-2x |
| Table 1 中 TCGA WSI 样本 LZMA | 1.93x，高于 PNG 1.01x |
| patch 级 WISE vs ArIB-BPS | C16 2.82 vs 2.63；C17 3.20 vs 2.72；TCGA 2.50 vs 2.38；IHC 3.85 vs 3.81 |
| throughput | WISE 15.1 MB/s；ArIB-BPS 18.73 KB/s |
| memory | WISE 0.1 GB；ArIB-BPS 3.81 GB |
| ablation: LZMA baseline → +hierarchical → +bitmap | C16 1.25→2.12→2.82；C17 1.39→2.46→3.20；TCGA 1.17→2.03→2.50 |
| bitmap PSNR empirical study | before bitmap coding 约 3-18；after hierarchical/bitmap coding 可升至 48，部分 higher-bit bitmap 为 infinite PSNR |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. WSI base level | multi-resolution pyramid 中的 base-level WSI | 以 patch-based 方式处理；其他 pyramid layers 可由 base level downsampling 得到 | WSI patch stream |
| 2. Empty-area preprocessing | RGBA WSI patch / slide | 删除 row/column sum 为 0 的区域；丢弃诊断信息很少的 alpha channel | tissue/informative RGB region + 空白区域坐标表示 |
| 3. Hierarchical projection coding | RGB informative patch | 按 row、column、channel 存储一阶差分，而不是原始像素值 | 数值范围缩短、entropy 降低的 delta-coded patch |
| 4. Bitmap encoding | delta-coded byte stream | 将同一 bit position 的 bits 跨相邻 bytes 分组并重新打包，分离 effective / ineffective bits | 更有局部重复模式的 bitmap-repacked byte stream |
| 5. LZW dictionary coding | bitmap-repacked stream | 用 LZW 将频繁出现的长模式替换为字典索引 | WISE compressed bitstream |
| 6. 解码 | compressed bitstream + 必要元数据 | LZW 解码、bitmap 反变换、hierarchical projection 逆变换、空白区域恢复 | 与原 WSI 完全一致的 lossless reconstruction |

## 优缺点与还能做什么

### 优点

- 针对 WSI 的两个结构性特点设计：大面积空白区域用坐标/区域编码解决，informative region 的不规则高频信息用差分与 bitmap 重排解决。
- 不依赖训练数据或神经网络推理，CPU throughput 和 memory 对实践部署更友好。
- 方法解释性强：Table 1 说明为什么 dictionary 更适合 WSI，Table 5 说明 hierarchical projection 和 bitmap encoding 各自贡献。

### 局限 / 风险

- 空白区域搜索算法被明确排除在论文重点之外，真实医院系统中扫描仪格式、背景值、组织边界和 metadata 处理会影响端到端效果。
- LZW 字典收集带来输入尺寸越大 memory 越高的问题；论文也指出 8192x8192x3 时 memory 已接近数 GB。
- 主文没有详细展开文件格式封装、随机访问、streaming 解码、pyramid 多层 metadata 兼容等工程问题。
- Table 2 的 MinerU 提取文本存在错位，阅读时应优先看表格图片和 PDF。

### 还能做什么

- 将空白区检测、坐标编码、patch 顺序、metadata 封装补成可复现实验协议。
- 研究可随机访问的 block-wise WISE，让病理 viewer 不必解压整张 slide。
- 替换或增强 LZW，例如用 Zstd/LZMA 作为 final backend，测试 bitmap 重排是否对不同字典编码器同样有效。
- 在更多扫描仪、染色、组织类型和医院真实归档格式上验证 36x 平均压缩率是否稳定。

## 阅读 Q&A 记录

- **Q: WISE 为什么不是直接把 WSI 交给 PNG、Zstd 或 learned compressor？**  
  A: Introduction 和 Sec. 3.1 说明，WSI informative region 不是普通自然图像的平滑局部结构，而是高频、局部极值多、波动强。PNG/prediction/entropy 方法难以建模这种局部性；dictionary 方法更鲁棒，但裸 LZMA 仍只有约 2x，因此需要先重排信息。

- **Q: empty-area preprocessing 是否就是主要压缩来源？**  
  A: 不是。作者承认空白区可显著节省空间，但 Sec. 3.3 和 Sec. 4.4 明确把核心挑战放在 non-empty informative areas。Table 3/4 的 patch 级实验也专门验证非空 patch 上的压缩率。

- **Q: hierarchical projection coding 改变了什么中间表示？**  
  A: 它把原始 RGB 像素改成 row、column、channel 三个方向的一阶差分。这样数值范围缩短、entropy 降低，后续 bitplane/bitmap 中 higher-position bits 会更空、更相似。

- **Q: bitmap encoding 的关键不是压缩，而是什么？**  
  A: 它主要是重排 byte stream：把相同 bit position 的 bits 聚在一起，使 effective bits 和 ineffective bits 的模式更集中。真正的压缩由后续 LZW 利用这些长重复模式完成。

- **Q: 论文中 PSNR 为什么出现在 lossless paper 里？**  
  A: Sec. 4.1 说明 decoded image 不用 PSNR/SSIM，因为无损重建没有失真；Sec. 4.6 的 PSNR 是用来比较 bitmap 之间的局部相似性，作为“更容易被字典收集”的证据。

## Citation Landscape

**Semantic Scholar 状态**: 本目录已有 `semantic-scholar/references.json` 和空的 `recommendations.json`；`detail.json` 返回 429 rate limit。因此 TLDR、被引次数、influential citation count 无法可靠读取，以下使用本地已保存 references 数据做 graceful fallback。

| 项目 | 数值 / 状态 |
|------|-------------|
| TLDR | unavailable，Semantic Scholar detail API rate-limited |
| reference count | 本地 references.json 含 37 条 |
| citation count | unavailable，detail.json 为 429 |
| influential citation count | unavailable，detail.json 为 429 |
| recommendations | recommendations.json 中 `recommendedPapers` 为空 |
| Semantic Scholar 查询键 | ArXiv:2503.18074 |
| Connected Papers | https://www.connectedpapers.com/main/2503.18074 |

### 参考文献分组（按本地 Semantic Scholar references）

| 主题 | 高相关参考文献 |
|------|----------------|
| WSI / digital pathology datasets and analysis | Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer；1399 H&E-stained sentinel lymph node sections of breast cancer patients: the CAMELYON dataset；Whole Slide Imaging (WSI) in Pathology: Current Perspectives and Future Directions；DigestPath；TCGA pan-cancer analysis |
| Medical / WSI compression | Optimized JPEG 2000 Compression for Efficient Storage of Histopathological Whole-Slide Images；Digital pathology whole slide image compression with vector quantized variational autoencoders；A deep learning-based compression and classification technique for whole slide histopathology images；Optimizing Storage and Computational Efficiency |
| Learned lossless image compression | Learned Lossless Image Compression Based on Bit Plane Slicing；LC-FDNet；Integer Discrete Flows；IDF++；Bit-Swap |
| General byte-stream / entropy / dictionary compression | TRACE；PAC；Huffman；Arithmetic coding；GZIP；LZW/LZMA；Zstd；Pfor |

### 推荐继续读

1. **Learned Lossless Image Compression Based on Bit Plane Slicing** - WISE 对比的 ArIB-BPS neural baseline，和 bitmap/bit-plane 思路最接近。
2. **LC-FDNet: Learned Lossless Image Compression with Frequency Decomposition Network** - 用频域分解处理 lossless image compression，可对照 WISE 的 information irregularity 论证。
3. **Optimized JPEG 2000 Compression for Efficient Storage of Histopathological Whole-Slide Images** - WSI 压缩中 lossy/near-lossless 医学风险背景。
4. **TRACE / PAC** - 作者引用的 general-purpose lossless compressor，可对照 WISE 为什么不用预测高波动 WSI。
5. **CAMELYON16 / CAMELYON17 dataset papers** - 理解 C16/C17 作为 WSI benchmark 的来源和 slide 规模。

## BibTeX

```bibtex
@article{mao2025wise,
  title={WISE: A Framework for Gigapixel Whole-Slide-Image Lossless Compression},
  author={Mao, Yu and Wang, Jun and Guan, Nan and Xue, Chun Jason},
  journal={arXiv preprint arXiv:2503.18074},
  year={2025}
}
```
