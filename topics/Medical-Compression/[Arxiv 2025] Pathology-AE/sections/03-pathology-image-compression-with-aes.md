[← 返回 README](../README.md)

# 3 Pathology image compression with AEs

## 预览

本节是方法核心：先比较 SD-1.5、SD-3、DC-AE 与 JPEG 的压缩/重建质量，再说明 decoder-only pathology fine-tuning，最后给出 K-means latent quantization。

# 3.1 Compression ratio

The starting point is the comparison of storage requirements and reconstruction fidelity between JPEG compression, with different quality settings, and the learned autoencoder compression schemes. We utilize 1000 256 $\times$ 256 px image patches from the TCGA dataset to measure reconstruction fidelity. For the autoencoders we choose SD-1.5 [21], SD-3 [7] and DC-AE [4]. The SD-1.5 and SD-3 autoencoders downsample the image by a factor of 8, with SD-1.5 using 4 channels in the embedding whereas SD-3 uses 16 channels. The DC-AE f32 variant downsamples the image by a factor of 32 and uses 32 latent channels.

> 💡 **压缩率由 latent 形状决定**: 对 256x256 patch，SD-1.5/SD-3 是 8x spatial downsample，但 channel 数不同；DC-AE-f32 是 32x spatial downsample。SD-3 latent 大、保真高；DC-AE latent 小、压缩强，是后续“2KB K-means latent”结果的来源。

In Table 1, we compare the compression rates of JPEG at different quality settings against the autoencoder models. To assess reconstruction fidelity, we employ three pathology foundation models—UNI [5], Phicon-v2 [9], and Gigapath [25]. Foundation pathology models [5,9] train an image encoder that embeds images into a learned embedding vector in $\mathbb { R } ^ { k }$ . These encoders maximize the similarity of the k-dimensional embeddings for images that are semantically and visually similar while minimizing it for different images.

> 💡 **foundation-model reconstruction evaluation**: UNI、Phikon-v2、Gigapath 是评价器而不是压缩器。用它们的 embedding cosine similarity，是为了问“重建图在病理表征空间是否还像原图”，比像素误差更贴近 AI 使用场景。

In our assessment, we extract embeddings from both the original and reconstructed images and compute their cosine similarity. Traditional image quality metrics, such as SSIM and PSNR capture pixel-level differences and are inadequate for accurately assessing the quality of pathology images [26]. Embedding similarity however provides a more task-relevant evaluation, as it correlates better with downstream performance. Our results show that existing autoencoders already achieve adequate reconstruction fidelity; even the aggressive compression performed by DC-AE maintains better embedding similarity than JPEG while requiring half the storage.

> 💡 **不要误读 SSIM/PSNR**: Table 1 中 AE 的 SSIM/PSNR 低于 JPEG-75/50，但 embedding similarity 和下游任务更稳。作者借此反驳“像素指标越高越好”的简单判断；病理图像里保住表征特征比逐像素接近更关键。

Table 1. Compression metrics for JPEG and different LDM autoencoders. Fine-tuning boosts reconstruction fidelity. Employing a K-means-based quantization allows for minimal storage requirements while preserving important pathology image features.   

<table><tr><td>Compression</td><td>Fine-tuned</td><td>Quant</td><td>Size (KB)</td><td>Embedding similarity UNI Phicon-v2 Gigapath</td><td></td><td>Image quality SSIM PSNR</td></tr><tr><td>JPEG - 75</td><td></td><td>—</td><td>18</td><td>0.988 0.985</td><td>0.985</td><td>0.994 41.93</td></tr><tr><td>JPEG - 50</td><td></td><td></td><td>15</td><td>0.904 0.855</td><td>0.866</td><td>0.964 32.97</td></tr><tr><td>JPEG - 20</td><td></td><td></td><td>7</td><td>0.734 0.623</td><td>0.645</td><td>0.870 27.69</td></tr><tr><td>JPEG - 10</td><td></td><td></td><td>5</td><td>|0.512 0.456</td><td>0.407</td><td>0.788 25.13</td></tr><tr><td rowspan="4">SD-1.5</td><td>X</td><td>×</td><td>16</td><td>0.837 0.825</td><td>0.796</td><td>0.651 22.55</td></tr><tr><td>✓</td><td>X</td><td>16</td><td>0.932 0.935</td><td>0.909</td><td>0.649 22.50</td></tr><tr><td>V</td><td>static-int8</td><td>4</td><td>0.912 0.921</td><td>0.878</td><td>0.639 21.99</td></tr><tr><td>✓</td><td>K-means</td><td>4</td><td>0.932 0.935</td><td>0.909</td><td>0.649 22.49</td></tr><tr><td rowspan="4">SD-3</td><td>X</td><td>×</td><td>64</td><td>0.959 0.931 0.978</td><td>0.947</td><td>0.894 28.25</td></tr><tr><td>√</td><td>X</td><td>64</td><td>0.967</td><td>0.972</td><td>0.877 27.42</td></tr><tr><td>✓</td><td>static-int8</td><td>16</td><td>0.944 0.934</td><td>0.932</td><td>0.862 26.67</td></tr><tr><td>✓</td><td>K-means</td><td>16</td><td>|0.978 0.967</td><td>0.972</td><td>0.877 27.41</td></tr><tr><td rowspan="4">DC-AE-f32</td><td>×</td><td>×</td><td>8</td><td>0.733 0.747</td><td>0.659</td><td>0.536 20.82</td></tr><tr><td>✓</td><td>X</td><td>8</td><td>|0.906 0.925</td><td>0.868</td><td>0.538 21.03</td></tr><tr><td>V</td><td>static-int8</td><td>2</td><td>0.900 0.921</td><td>0.861</td><td>0.537 21.06</td></tr><tr><td>✓</td><td>K-means</td><td>2</td><td>0.906 0.926</td><td>0.867</td><td>0.538 21.03</td></tr></table>

> 💡 **Table 1 关键读数**: JPEG-10 5KB 时 UNI similarity 约 0.512；fine-tuned DC-AE-f32 K-means 2KB 时 UNI/Phikon/Gigapath 约 0.906/0.926/0.867。也就是说 DC-AE 在更小表示下保住了更多病理 foundation features。

# 3.2 Pathology fine-tuning for AEs

Although existing autoencoders produce faithful reconstructions, they also make non-negligible changes to specific pathology features that may be critical in downstream tasks. In Fig. 1 we demonstrate one such example, where the DC-AE model changes the cell contents and structures in its reconstructions. We develop a simple fine-tuning scheme for pre-trained autoencoders to better align the image reconstructions, retaining important pathology features.

> 💡 **fine-tuning 的目标**: vanilla AE 已能压缩，但对病理显微结构没有显式忠诚约束。微调不是为了提升自然图像美观度，而是减少细胞内容、腺体边界、组织纹理这类病理特征的重建偏移。

To perform this alignment, we utilize foundation pathology models and finetune only the decoder with an additional loss that maximizes the similarity of the reconstruction and the original image in the learned embedding space of a foundation model. Specifically, for an image $x$ and its reconstruction $y$ , we use UNI [5] as the foundation model and minimize the L1 distance between the embeddings produced by the UNI encoder for them -

$$
\mathcal { L } _ { \mathrm { U N I } } ( x , y ) = | | \mathrm { U N I } ( x ) - \mathrm { U N I } ( y ) | | _ { 1 } .
$$

> 💡 **pathology-specific learned perceptual metric**: 这相当于把 LPIPS 式“深特征感知距离”换成病理 foundation encoder 的特征距离。UNI 被冻结，只提供 loss；它把重建优化方向从像素/纹理转向病理语义表征。

Following the training scheme of previous autoencoders [21], we use an L1 pixel reconstruction loss and a learned PatchGAN discriminator loss [16]. We select 2500 WSIs from TCGA Breast, Colon, and Prostate. Using DSMIL [19], we extract $2 5 6 \times 2 5 6$ patches at $2 0 \times$ magnification, yielding 24 million patches.

> 💡 **训练数据规模**: 2500 WSIs / 24M patches 看似大，但相对从头训练 AE 仍小。作者因此选择 decoder-only fine-tuning，把新信息限制在“如何从既有 latent 解码成更像病理图”的映射上。

We utilize the codebase of [21] to finetune the autoencoders. We set the learning rate at $5 \times 1 0 ^ { - 5 }$ with a warmup of 10,000 steps. We train for 120,000 iterations on 8 NVIDIA A100 GPUs, with a batch size of 12 per GPU. The loss used is

$$
\mathcal { L } ( x , y ) = w _ { \mathrm { L 1 } } \mathcal { L } _ { \mathrm { L 1 } } ( x , y ) + w _ { \mathrm { G A N } } \mathcal { L } _ { \mathrm { G A N } } ( x , y ) + w _ { \mathrm { U N I } } \mathcal { L } _ { \mathrm { U N I } } ( x , y )
$$

where we choose $w _ { \mathrm { L 1 } } = 1$ , $w _ { \mathrm { G A N } } = 0 . 5$ and $w _ { \mathrm { U N I } } = 1$ .

> 💡 **loss 组合**: L1 约束低频/颜色和像素一致性，PatchGAN 约束局部真实感，UNI loss 约束病理表征一致性。三个 loss 的平衡避免 decoder 为了追求 UNI embedding 而产生不自然的纹理。

We find that training only the decoder is enough to improve the pathologyspecific reconstruction metrics, even when using a relatively small dataset. By preserving the encoder, we ensure that the AE latent space is unchanged and we only learn to interpret it differently during reconstruction. We validate the assumption that the existing autoencoders can already compress pathology images adequately and that we only need to slightly alter the reconstruction part. Fine-tuning the full autoencoder model would require significantly more data to ensure that the learned latent would not overfit to the fine-tuning dataset.

> 💡 **为什么不动 encoder**: 压缩系统里 encoder 决定 latent 分布、局部性和可量化性。若 full fine-tuning，latent 可能变成 TCGA-specific，反而破坏 LDM AE 的泛化与后续 K-means quantization 的稳定性。

The results presented in Table 1 show that fine-tuning leads to a significant boost in embedding similarity across all models. Notably, DC-AE improves from 0.733 to 0.906, demonstrating that fine-tuning enables even highly compressed representations to retain critical information. Additionally, fine-tuning enables AEs to surpass JPEG in embedding similarity at comparable compression ratios. For example, fine-tuned SD-1.5 achieves a UNI similarity of 0.932, whereas JPEG-50 – despite having a similar compression ratio – only reaches 0.904, indicating that learned compression better preserves meaningful information.

> 💡 **最强数值证据**: DC-AE-f32 从 0.733 到 0.906 的 UNI similarity 增幅最大，说明高压缩 AE 的主要短板不是 latent 完全丢失信息，而是 decoder 没把病理相关信息正确重建出来。

# 3.3 Quantization

To further reduce storage, we apply quantization to the AE latents, converting the continuous-valued representations into more compact discrete values. The straightforward approach is static int8 quantization, where the range of the latent representations is split into equally spaced bins and mapped to 8- bit integers. However, this method does not consider the statistics of the latent representations; we find that the values are not uniformly distributed but have higher concentration near 0, which can be partially attributed to the regularization applied to the AEs (e.g. KL). This naive approach noticeably degrades reconstruction quality due to the misalignment between chosen quantization bins and the actual latent distribution.

> 💡 **static int8 的问题**: 等宽 bin 假设 latent 值均匀覆盖范围，但 LDM latent 往往集中在 0 附近。均匀量化会把大量高密度区域用不够细的 bin 表示，造成 decoder 输入偏移和局部重建伪影。

To mitigate this, we propose a clustering-based quantization strategy that adapts to the distribution of latent values. Instead of mapping values to fixed bins, we first learn a set of representative centroids using K-means clustering and then quantize the latents based on these learned clusters. The process consists of the following steps: (1) Cluster learning: We extract latents from randomly sampled TCGA images and apply K-means clustering on the values, learning 256 clusters (8-bit). (2) Compression: Each latent value is assigned to its closest cluster center, only storing the cluster index instead of the floating-point value. (3) Decompression: Stored indices are replaced with their respective floatvalued cluster centers, followed by passing the latent through the AE decoder.

> 💡 **K-means latent quantization 数据流**: 训练时学 256 个 centroid；压缩时只存每个 latent value 的 cluster index；解压时查表还原成 centroid float，再交给 decoder。它不是向量量化整块 latent token，而是按 latent value 分布做标量聚类量化。

Table 1 shows that static int8 quantization leads to a non-trivial drop in embedding similarity across all models. In contrast, K-means clustering-based quantization attains nearly the same embedding similarity as the non-quantized representations while considerably reducing storage requirements.

> 💡 **量化结论**: SD-1.5 K-means 从 16KB 降到 4KB，DC-AE-f32 从 8KB 降到 2KB，同时 embedding similarity 几乎不变。这是本文相对“直接用 AE reconstruction”的重要存储增益。

## Section 总结

| 模块 | 输入 | 操作 | 输出 / 证据 |
|------|------|------|-------------|
| AE benchmark | TCGA 1000 个 256x256 patches | SD-1.5 / SD-3 / DC-AE 重建，与 JPEG 比较 | DC-AE 8KB 已优于重压 JPEG 的 embedding similarity |
| Pathology fine-tuning | 原图与 AE 重建图 | decoder-only，L1 + GAN + UNI embedding L1 | DC-AE UNI 0.733→0.906 |
| K-means quantization | 连续 latent values | K=256 centroid，存 8-bit index | DC-AE-f32 8KB→2KB，similarity 基本保持 |
