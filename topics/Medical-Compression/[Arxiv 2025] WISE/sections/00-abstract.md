[← 返回 README](../README.md)

# Abstract

## 预览

摘要把 WISE 的主线压得很紧：WSI 很大、必须 lossless、通用无损压缩失败、失败原因是 information irregularity，WISE 用 hierarchical encoding + effective-bit extraction + dictionary coding 得到平均 36x、最高 136x。

Whole-Slide Images (WSIs) have revolutionized medical analysis by presenting high-resolution images of the whole tissue slide. Despite avoiding the physical storage of the slides, WSIs require considerable data volume, which makes the storage and maintenance of WSI records costly and unsustainable. To this end, this work presents the first investigation of lossless compression of WSI images. Interestingly, we find that most existing compression methods fail to compress the WSI images effectively. Furthermore, our analysis reveals that the failure of existing compressors is mainly due to information irregularity in WSI images. To resolve this issue, we develop a simple yet effective lossless compressor called WISE, specifically designed for WSI images. WISE employs a hierarchical encoding strategy to extract effective bits, reducing the entropy of the image and then adopting a dictionary-based method to handle the irregular frequency patterns. Through extensive experiments, we show that WISE can effectively compress the gigapixel WSI images to 36 times on average and up to 136 times.

> 💡 **摘要问题链**: 这段不是泛泛说“大图压缩”，而是把目标锁定为 WSI 的 lossless compression。医学图像的关键约束是不能引入诊断相关失真，所以高压缩率的 lossy WSI 方法不能直接替代。WISE 的 claim 也很具体：不是发明新的神经 codec，而是先改变 byte stream 的信息组织方式，再交给字典压缩。

> 💡 **关键术语 - information irregularity**: 摘要中的 information irregularity 后面在 Sec. 3.1 展开，指 WSI informative region 中高频信息占比高、局部极值多、波动大。这个性质会破坏 PNG、entropy coding、prediction-based compressor 依赖的局部平滑/可预测性。

> 💡 **结果解读**: “36 times on average and up to 136 times” 是整张 gigapixel WSI 的强结果；非空 patch 上的结果更保守，Table 3/4 中 WISE 大约 1.63x-3.85x。阅读时要区分“整张 slide 含大量空白区域”的压缩率和“informative patch”的压缩率。

## Section 总结

| 要点 | 说明 |
|------|------|
| 任务 | WSI lossless compression |
| 核心困难 | information irregularity 使通用无损压缩器效果差 |
| 方法 | hierarchical projection coding + bitmap encoding + LZW dictionary coding |
| 结果 | gigapixel WSI 平均 36x，最高 136x |

