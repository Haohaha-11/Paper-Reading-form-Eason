[← 返回 README](../README.md)

# 2. Related Work

## 预览

Related Work 按三类压缩路径组织：医学图像 lossy compression、patch-based lossless compression、byte stream-based compression。WISE 的位置是在第三类里，但它先对 WSI byte stream 做结构化重排，再用 dictionary coding。

Medical Image Compression has attracted great interest in the community. Several existing works have adopted lossy compression techniques such as JPEG-2000 or VQ-

VAE [5, 11, 12, 15], and compress WSI patches into compact embeddings [7, 19, 27]. Reconstruction from such embeddings has also gained popularity [35, 37]; however, these approaches inevitably suffer from information loss inherent in the compressed representations.

> 💡 **医学压缩路线**: 这里的 lossy 方法适合节省存储或下游任务表示，但与 WISE 的目标不同。WISE 针对的是归档级原图恢复，不能接受 embedding 重建带来的不可逆误差。

Patch-based Lossless Compression Approaches typically divide an image into multiple small patches, which are then compressed individually. Typical patch-based compression methods include traditional image compression techniques, such as PNG [3] and TIFF [2]. In recent years, there has been emerging interest in applying deep learning methods to image compression and it has intrigued multiple advanced compression technologies, such as Bit-Swap [16], IDF [13], $\mathrm { I D F + + }$ [28], and ArIB-BPS [36]. However, deep learning methods are still based on entropy encoding and thus does not perform well on WSI slides.

> 💡 **patch-based 的限制**: patch 化可以降低单次处理内存，但不能自动解决 WSI 的信息不规则性。作者对 ArIB-BPS 的比较很重要：即使是 bit-plane slicing 的 learned lossless 方法，在 WSI patch 上吞吐和内存也不适合实践。

Byte Stream-based Image Compression Approaches treats the image as a byte stream input, onto which to perform the compression operations to reduce the storage requirements. These methods can be further divided into entropy-based and dictionary-based methods. Common entropy-based methods include Huffman coding [14], Arithmetic coding [33], and PixelCNN [25, 29]. Huffman coding [14] is a classical entropy-based coding method that assigns shorter codes to more frequently occurring symbols. Arithmetic coding [33] is a more sophisticated method that assigns a unique floating-point range to the entire input sequence, often offering higher compression rates than Huffman coding. PixelCNN [25, 29] uses convolutional neural networks to model the conditional dependencies between pixels, enabling efficient compression by learning the image's probability distribution. TRACE [22] and PAC [23] utilize lightweight transformers and MLP to model dependencies between adjacent pixels, respectively. Common dictionary-based methods include Gzip [10], LZMA [38], and Zstandard [8]. Gzip [10] is widely used in file compression, which is built upon the LZ77 [17] compression algorithm combined with Huffman coding for further optimization. LZMA [38] enhances efficiency by using bitfieldspecific contexts to represent literals or phrases. Zstandard [8] combines a dictionary-matching stage (LZ77) with a large search window and a dual entropy-coding stage using Huffman coding for literals and Finite State Entropy (FSE) for sequences. However, although these methods can achieve a certain amount of lossless compression ratio on WSI images, the compression ratio is not satisfied.

> 💡 **WISE 与 byte-stream compressor 的差异**: WISE 没有否定 dictionary compressor，反而把它作为最后一步。关键改动在输入：把原始 pixel byte stream 变成 delta-coded、bitmap-repacked stream，让 LZW 看到更多长重复模式。

> 💡 **Q&A 批注记录**:
> - Q: 既然 Zstd/LZMA 已经是成熟字典压缩器，WISE 为什么还要自己加 hierarchical/bitmap 编码？
> - A: Related Work 说这些方法直接压 WSI 有一定效果但不满意；Sec. 3.1 的 Table 1 显示 LZMA 在 WSI 样本只有 1.93x。WISE 的前处理等于为字典编码器“制造更好的局部模式”。

## Section 总结

| 类别 | 与 WISE 的关系 |
|------|----------------|
| lossy WSI compression | 压缩率高但不能保证诊断信息无损 |
| patch-based lossless | 可控内存，但 entropy/prediction 假设在 WSI 上弱 |
| byte-stream entropy methods | 依赖符号分布或预测，遇到高波动 WSI 受限 |
| byte-stream dictionary methods | 更鲁棒，是 WISE 最终采用并增强的方向 |

