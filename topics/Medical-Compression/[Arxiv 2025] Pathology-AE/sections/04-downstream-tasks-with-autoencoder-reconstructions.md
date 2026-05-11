[← 返回 README](../README.md)

# 4 Downstream tasks with autoencoder reconstructions

## 预览

本节回答“embedding similarity 高是否真的有用”：作者把压缩后重建图直接替换原图，测试 slide-level MIL、patch classification 和 segmentation。

We benchmark the downstream performance of all 3 pre-trained and fine-tuned AEs on dense pixel-level and image-level tasks. For those benchmarks, we encode the images into latents using the AEs and reconstruct the image from the compressed representations. We compare the performance of the downstream methods using the original images vs using the reconstructed images.

> 💡 **验证方式**: 下游模型吃的是 reconstructed image，而不是 latent 本身。这说明本文目标仍是图像压缩/重建文件格式替代，而不是让每个下游模型都改成直接消费 AE latent。

# 4.1 Image-level tasks

For slide-level classification on TCGA-BRCA, we perform subtyping (Invasive Ductal Carcinoma vs. Invasive Lobular Carcinoma) using ABMIL [15]. For patchlevel classification, we classify images from the NCT-CRC dataset [17] into nine tissue classes. In both cases, we perform 10-fold cross validation.

> 💡 **MIL 与 patch 分类互补**: BRCA subtyping 是 slide-level MIL，测试压缩对 WSI 级聚合诊断信号的影响；NCT-CRC 是 patch-level 9 类组织分类，测试局部纹理/组织类别是否被重建保留。

Table 2 shows that existing AE reconstructions already achieve strong classification performance, with minimal drop compared to original images. Finetuning further closes the gap, particularly for SD-1.5 and DC-AE-f32, bringing them closer to the original image results. This highlights the effectiveness of our approach and suggests that high representation similarity in the foundation model embedding space translates to functional similarity in predictive tasks.

> 💡 **证据链成立点**: Table 1 的 embedding similarity 不是孤立指标；Table 2 显示 fine-tuning 后分类性能确实回升。尤其 DC-AE-f32 在 NCT-CRC 从 89.23±0.61 提升到 94.65±0.29，说明 pathology-aware decoder 对局部类别特征很关键。

Table 2. Classification using original (uncompressed) images and AE-reconstructed images. Fine-tuning improves performance, bringing it closer to that of original images.   

<table><tr><td>Compression</td><td>[Fine-tuned</td><td>BRCA subtyping NCT-CRC</td></tr><tr><td>Original images</td><td>−</td><td>94.89 ± 2.67 96.31 ± 0.20</td></tr><tr><td rowspan="2">SD-1.5</td><td>X</td><td>92.89 ± 2.51 94.36 ± 0.48</td></tr><tr><td>✓</td><td>93.96 ± 2.77 95.73 ± 0.21</td></tr><tr><td rowspan="2">SD-3</td><td>×</td><td>94.44 ± 2.65 95.99 ± 0.16</td></tr><tr><td>✓</td><td>94.05 ± 3.40 96.41</td></tr><tr><td rowspan="2">DC-AE-f32</td><td>X</td><td>92.82 ± 1.81 89.23 ± 0.61</td></tr><tr><td>✓</td><td>93.30 ± 2.15 94.65 ± 0.29</td></tr></table>

> 💡 **Table 2 读数**: SD-3 原本就接近原图，fine-tuning 增益小；DC-AE-f32 压缩最强，vanilla patch classification 掉得最多，fine-tuning 恢复明显。这和 Table 1 中 DC-AE embedding similarity 改善最大相互印证。

# 4.2 Pixel-level tasks

We perform segmentation using SAM-path [27] on two datasets – the Breast Cancer Semantic Segmentation (BCSS) dataset [1] and the Colorectal Adenocarcinoma Gland (CRAG) dataset [13]. BCSS contains patches sampled from 151

TCGA-BRCA WSIs on which we perform tissue region segmentation. CRAG contains patches sampled from 38 colon cancer WSIs, and we perform semantic segmentation of colorectal adenocarcinoma and benign glands.

> 💡 **为什么要 segmentation**: 分类可以容忍局部轻微形变，但 segmentation 要求边界、腺体结构、组织区域空间位置准确。这里是对 AE reconstruction locality 的更强测试。

In Table 3 we showcase the average Dice score, intersection over union (IoU) and F1 score for the original images, JPEG-compressed images and autoencoder reconstructions. For the BCSS dataset, we perform comparable compression to JPEG-10 (32 vs 57 KB) when using our K-means quantization, while only having ${ < } 1 \%$ performance drop for all metrics. With JPEG-10 the performance decreases by ${ > } 1 0 \%$ . Similarly, on CRAG we apply similar compression (72 vs 93 KB) without altering the result while JPEG-10 reduces the Dice score by $>$ 3%.

> 💡 **segmentation 关键对比**: DC-AE-f32 + K-means 在 BCSS 32KB 下 Dice 71.20，原图 71.57，几乎不掉；JPEG-10 57KB 下 Dice 60.56，掉幅巨大。这个结果强力支持“AE 压缩伪影比 JPEG 高压缩伪影更少伤害病理任务”。

Table 3. Segmentation results using original, JPEG and AE-compressed images. The original CRAG images are uncompressed whereas BCSS are compressed with JPEG-75. Heavy JPEG compression corrupts task-important features while AE compression (with and without fine-tuning) with quantization, does not sacrifice performance.   

<table><tr><td rowspan="2">Compression</td><td rowspan="2">FT</td><td rowspan="2">Quant</td><td colspan="3">BCSS</td><td colspan="2">CRAG</td></tr><tr><td>Size (KB)]</td><td>Dice ↑</td><td>F1 ↑ IoU ↑</td><td>Size (KB)</td><td>Dice ↑ F1 ↑ IoU ↑</td></tr><tr><td>Original images</td><td>−</td><td></td><td>545</td><td>71.57 80.38</td><td>67.20</td><td>3935</td><td>87.16 85.24 85.24</td></tr><tr><td>JPEG 50</td><td></td><td></td><td>154</td><td>71.16 79.92</td><td>66.56</td><td>246</td><td>86.87 85.38 85.38</td></tr><tr><td>JPEG 10</td><td>=</td><td></td><td>57</td><td>60.56 68.02</td><td>51.54</td><td>93</td><td>83.45 84.87 84.87</td></tr><tr><td rowspan="3">SD-1.5</td><td>×&gt;&gt;</td><td>×</td><td>256</td><td>71.34 80.12</td><td>66.84</td><td>576</td><td>86.57 84.34 84.34</td></tr><tr><td></td><td>X</td><td>256</td><td>71.41 80.20</td><td>66.95</td><td>576</td><td>86.80 85.48 85.48</td></tr><tr><td></td><td>✓</td><td>64</td><td>71.34 80.13</td><td>66.84</td><td>144</td><td>86.82 85.50 85.50</td></tr><tr><td rowspan="3">SD-3</td><td></td><td>×</td><td>512</td><td>71.52 80.33</td><td>67.12</td><td>1152</td><td>86.85 84.49 84.49</td></tr><tr><td></td><td>X</td><td>512</td><td>71.46 80.26</td><td>67.02</td><td>1152</td><td>86.90 84.80 84.80</td></tr><tr><td></td><td>✓</td><td>128</td><td>71.47 80.27</td><td>67.04</td><td>288</td><td>86.90 84.81 84.81</td></tr><tr><td rowspan="3">DC-AE-f32</td><td></td><td>×</td><td>128</td><td>71.25 80.02</td><td>66.70</td><td>288</td><td>87.27 87.62 87.62</td></tr><tr><td></td><td>X</td><td>128</td><td>71.21 79.97</td><td>66.63</td><td>288</td><td>87.20 88.06 88.06</td></tr><tr><td>*</td><td>✓</td><td>32</td><td>71.20 79.96</td><td>66.62</td><td>72</td><td>87.20 88.08 88.08</td></tr></table>

> 💡 **fine-tuning 在 segmentation 中不总是显著**: Table 3 里 AE reconstruction 本身已经很接近原图，fine-tuning 的增益不如分类明显。这提示 UNI embedding loss 更直接帮助语义/表征任务，pixel-level 边界任务还可能需要边界感知或 stain-aware 的额外约束。

## Section 总结

| 下游证据 | 数据/模型 | 结论 |
|----------|-----------|------|
| MIL slide classification | TCGA-BRCA + ABMIL | fine-tuned AE reconstruction 接近原图 |
| Patch classification | NCT-CRC 9 classes | DC-AE fine-tuning 恢复明显，89.23→94.65 |
| Segmentation | BCSS / CRAG + SAM-path | K-means AE reconstruction 在强压缩下远优于 JPEG-10 |
