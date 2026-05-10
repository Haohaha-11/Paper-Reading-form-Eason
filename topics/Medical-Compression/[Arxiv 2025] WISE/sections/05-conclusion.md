[← 返回 README](../README.md)

# 5. Conclusions

## 预览

Conclusion 重申 WISE 解决 WSI 的 information sparsity 与 information irregularity，并把方法定位成可部署的 lossless compressor。

In this work, we explored applying compression techniques to WSI images. Through the study of existing compression techniques on WSI images, we identified several unique challenges in compressing WSI images, including how to deal with information sparsity and information irregularity. To resolve those challenges, we developed a simple yet effective WSI specialized compressor called WISE. WISE is built upon a hierarchical encoding strategy and a dictionarybased compression method, which can be further combined with various entropy-based compression methods to further improve the compression ratios. Extensive experiments on realistic WSI benchmarks show that WISE achieves up to $1 3 6 \times$ and on average $3 6 \times$ compression over prior methods, while retaining low decoding complexity and preserving critical diagnostic content, making it highly practical for deployment in real-world digital pathology workflows across diverse tissue types and scanning conditions.

> 💡 **结论批读**: 论文最后把 WISE 的两类挑战说清楚：sparsity 对应空白区，irregularity 对应 informative tissue。真正可迁移的想法是“先把领域数据的冗余结构显式化，再交给成熟无损压缩器”，而不是为 WSI 训练一个重模型。

> 💡 **部署追问**: “highly practical” 还需要工程层面的补充验证：WSI 文件格式封装、随机 tile 访问、metadata 保真、scanner 差异、并行压缩、失败恢复。这些不影响论文主方法，但会影响医院系统落地。

## Section 总结

| 结论 | 含义 |
|------|------|
| information sparsity | WSI 大量空白区域不应逐像素存储 |
| information irregularity | informative region 高频/波动强，需要重排信息结构 |
| WISE | hierarchical encoding + dictionary coding 的 WSI-specialized lossless compressor |
| 未来重点 | 随机访问、streaming、格式兼容和更强 backend codec |

