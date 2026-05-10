# Learned Image Downscaling for Upscaling using Content Adaptive Resampler

**作者**: Wanjie Sun; Zhenzhong Chen  
**会议/期刊**: IEEE Transactions on Image Processing (TIP) | **年份**: 2020  
**链接**: [arXiv](https://arxiv.org/abs/1907.12904) / [PDF](https://arxiv.org/pdf/1907.12904) / [Code](https://github.com/sunwj/CAR)  
**主题定位**: Medical-Compression topic paper 4；自然图像 learned downscaling/SR 方法，可作为医学压缩中“低分辨率传输 + 高质量恢复”的机制参考。

## 一句话总结
CAR 用 ResamplerNet 为每个 LR 像素预测内容自适应 kernel weights 与 horizontal/vertical offsets，并通过可微下采样和 EDSR 的 L1 重建误差端到端训练，使 LR 图像在保持接近 bicubic 可视/压缩性质的同时显著提升后续 SR 恢复质量。

## 核心贡献
1. 提出 SR-guided learned image downscaling：下采样器不以固定 bicubic 为目标，而由后续 SR reconstruction error 反向指导。
2. 用 resampling 而非直接合成 LR：输出来自 HR 像素插值，避免无 LR 监督时生成不像图像的 latent-like 结果。
3. 为每个 LR 像素学习 content adaptive kernels 和 offsets：kernel 控制加权，offset 控制非均匀采样位置，能沿结构保留 SR 所需信息。
4. 设计 L1 reconstruction + offset distance + partial TV objective：在 SR gain、offset 拓扑稳定和 LR jaggies 之间做可调权衡。
5. 通过 PSNR/SSIM、跨 SRNet、消融、频谱/bpp、用户研究证明：CAR 的 LR 更适合 SR，同时 LR 视觉质量与压缩性接近 bicubic。

## 📖 批读导航

| Section | 内容 |
|---|---|
| [00 - Abstract](sections/00-abstract.md) | 摘要：CAR 目标、ResamplerNet、SRNet guidance |
| [01 - Introduction](sections/01-introduction.md) | 存储/传输动机、固定下采样不足、贡献 |
| [02 - Related Work](sections/02-related-work.md) | task independent 与 task specific downscaling |
| [03 - Model Architecture](sections/03-model-architecture.md) | CAR 数据流、kernel/offset、可微下采样、EDSR、loss |
| [04 - Experiments](sections/04-experiments.md) | 主结果、跨网络泛化、消融、频谱/bpp、用户研究 |
| [05 - Conclusion](sections/05-conclusion.md) | 结论与医学压缩启发 |
| [06 - Appendix / References](sections/06-appendix-references.md) | References 与后续阅读脉络 |

## 关键数字

| 指标 | 数值 |
|---|---|
| 训练数据 | DIV2K: 800 train / 100 validation / 100 test |
| 测试集 | Set5, Set14, BSD100, Urban100, DIV2K validation |
| ResamplerNet | 5 residual blocks, 128 channels |
| SRNet | EDSR, 32 residual blocks, 256 features |
| Kernel size | 3x3 in LR space, actual HR support is (3s)x(3s) |
| Batch / patch | batch 16; 192x192 for 4x, 96x96 for 2x |
| Main metric | PSNR / SSIM on Y channel in YCbCr |
| User study | 29 participants; 29 valid SR records, 28 valid downscaling records |
| Preference | CAR SR images get at least 73% preference over compared methods; >98% vs Perceptually in SR task |
| Citation count | Semantic Scholar: 163 citations, 23 influential citations, 63 references |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|---|---|---|---|
| 1. HR context modeling | HR image | Downscaling blocks + residual blocks extract context | feature maps |
| 2. Resampler generation | feature maps | Two upscaling branches predict kernel weights and offsets | K, ΔX, ΔY for each LR pixel |
| 3. Differentiable downscaling | HR image + K/offsets | Project LR coordinate to HR, apply offset sampling, bilinear interpolation, weighted sum, soft-round gradient | LR image |
| 4. SR guidance | LR image | EDSR/SRNet reconstructs HR | SR image |
| 5. Optimization | SR image + original HR | L1 reconstruction + offset distance + partial TV | updated ResamplerNet and SRNet |
| 6. Deployment intuition | HR image for storage/transmission | CAR downscales once; later SRNet restores details | compressed/transmitted LR + recovered HR |

## 优缺点与还能做什么

### 优点
- 下采样器直接为后续 SR 服务，比固定 bicubic 更能保留可恢复信息。
- Resampling 形式让无监督 LR 仍是图像，适合存储、传输和预览。
- Offsets 提供结构对齐能力，在 Urban100 等边缘丰富数据上收益明显。
- Table 5 显示 bpp 接近 bicubic，避免把收益完全建立在不可压缩高频隐藏信息上。

### 局限 / 风险
- LR 直接视觉质量与 SR 指标存在冲突；无 TV 时可能出现规律 jaggies。
- CAR 学到的是“对指定 SRNet/数据分布友好”的下采样，换域到医学图像需要重新验证诊断可读性。
- 使用 EDSR 作为 teacher 时，LR 可能编码 EDSR 偏好的结构，和其他重建器仍有适配差距。
- 论文主要用自然图像 PSNR/SSIM 和用户偏好，缺少医学任务指标或临床风险评估。

### 还能做什么
- 在医学压缩中加入病灶/组织结构感知 loss，避免 SR gain 掩盖诊断关键细节。
- 把 TV/offset 正则做成码率或可视质量条件控制，按“预览优先”或“重建优先”切换。
- 联合真实压缩 codec/quantization/noise 训练，而不只用 JPEG-LS bpp 做事后分析。
- 测试跨 SRNet、跨模态、跨医院分布的泛化，确认 CAR-LR 不是过拟合单个恢复模型。

## 阅读 Q&A 记录

- **Q: CAR 和普通 learned compression 的区别是什么？**  
  A: CAR 不学习 entropy model 或 bitstream，而是学习 downscaling resampler；它的 LR 是普通图像，可用现有图像存储/传输流程处理，再由 SRNet 恢复。

- **Q: 为什么不用 bicubic LR 作为监督？**  
  A: 论文认为 bicubic 对 SR 是次优目标。CAR 通过 resampling 保证 LR 合法，因此可以不用 bicubic guidance，让 SR reconstruction error 直接决定哪些信息该保留。

- **Q: Offsets 真正解决什么问题？**  
  A: Kernel weights 只能在固定网格上加权，offsets 让 kernel 元素移动到更符合边缘/纹理结构的非整数 HR 位置，提升可恢复信息密度。

- **Q: 为什么 partial TV loss 会降低 SR 指标？**  
  A: Figure 5 和 Table 3 显示，规律 jaggies 有时是 SRNet 易解码的信息载体；TV 平滑了 LR 观感，但也打断了这种最优信息保留方式。

- **Q: 对医学压缩最值得借鉴的点是什么？**  
  A: 不是直接套 EDSR，而是把下采样器做成可微、内容自适应、受下游重建/诊断任务指导，同时显式约束 LR 的可视质量和可压缩性。

## 📊 Citation Landscape

**Semantic Scholar TLDR**: This paper proposes a learned image downscaling method based on content adaptive resampler (CAR) with consideration on the upscaling process and achieves a new state-of-the-art SR performance through upscaled guided image resamplers which adaptively preserve detailed information that is essential to theUpscaling.

| 统计项 | 数值 |
|---|---:|
| Semantic Scholar Paper ID | `810c1980beb8c715f6a93274f73e8d0630245f50` |
| References | 63 |
| Citations | 163 |
| Influential citations | 23 |

**外部链接**: [Semantic Scholar](https://www.semanticscholar.org/paper/810c1980beb8c715f6a93274f73e8d0630245f50) / [Connected Papers](https://www.connectedpapers.com/main/1907.12904)

### Super-Resolution Backbone / Survey
- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (2016, Computer Vision and Pattern Recognition, citations: 12005)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) (2016, European Conference on Computer Vision, citations: 11379)
- [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092) (2014, IEEE Transactions on Pattern Analysis and Machine Intelligence, citations: 9198)
- [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921) (2017, 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), citations: 7107)
- [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://arxiv.org/abs/1511.04587) (2015, Computer Vision and Pattern Recognition, citations: 6941)

### Learned / Task-aware Downscaling
- Adaptive downsampling to improve image compression at low bit rates (2006, IEEE Transactions on Image Processing, citations: 139)
- Learning a Convolutional Neural Network for Image Compact-Resolution (2019, IEEE Transactions on Image Processing, citations: 132)
- Interpolation-Dependent Image Downsampling (2011, IEEE Transactions on Image Processing, citations: 106)
- Task-Aware Image Downscaling (2018, European Conference on Computer Vision, citations: 93)
- Content-adaptive image downscaling (2013, ACM Transactions on Graphics, citations: 92)

### Sampling / Dynamic Filter / Deformable
- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) (2015, International Conference on Learning Representations, citations: 9337)
- [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211) (2017, IEEE International Conference on Computer Vision, citations: 6459)
- Stochastic sampling in computer graphics (1986, ACM Transactions on Graphics, citations: 1128)
- [Dynamic Filter Networks](https://arxiv.org/abs/1605.09673) (2016, Neural Information Processing Systems, citations: 1081)
- Adaptive downsampling to improve image compression at low bit rates (2006, IEEE Transactions on Image Processing, citations: 139)

### Compression / Quantization
- The LOCO-I lossless image compression algorithm: principles and standardization into JPEG-LS (2000, IEEE Transactions on Image Processing, citations: 1827)
- Down-Scaling for Better Transform Compression (2001, Scale-Space, citations: 189)
- Adaptive downsampling to improve image compression at low bit rates (2006, IEEE Transactions on Image Processing, citations: 139)
- [Neural Multi-scale Image Compression](https://arxiv.org/abs/1805.06386) (2018, Asian Conference on Computer Vision, citations: 42)

### Recommended Papers
- Image Quality Recovery from Bicubic Upscaling on Symmetric Scales via Deep Neural Networks (2026, Proceedings of the 18th International Conference on Agents and Artificial Intelligence, citations: 0)
- [SRGAN-CKAN: Expressive Super-Resolution with Nonlinear Functional Operators under Minimal Resources](https://arxiv.org/abs/2605.01459) (2026, venue n/a, citations: 0)
- [Faithful Extreme Image Rescaling with Learnable Reversible Transformation and Semantic Priors](https://arxiv.org/abs/2605.00605) (2026, venue n/a, citations: 0)
- DA-CycleGAN: Degradation-Adaptive Unpaired Super-Resolution for Historical Image Restoration (2026, Journal of Imaging, citations: 0)
- SITFFormer: A Blind Super-Resolution Framework Preserving Structural Integrity and Texture Fidelity (2026, ACM Transactions on Multimedia Computing, Communications, and Applications (TOMCCAP), citations: 0)
- [Enhanced Self-Supervised Multi-Image Super-Resolution for Camera Array Images](https://arxiv.org/abs/2604.06816) (2026, venue n/a, citations: 0)
- Learned Super-Resolution as a Low-Bitrate Image Compression Mechanism for Multimedia Systems (2026, IEEE Systems Conference, citations: 0)
- Wavelet Query Multihead Attention Generative Adversarial Network for Remote Sensing Image Super-Resolution Reconstruction (2026, IEEE Transactions on Geoscience and Remote Sensing, citations: 0)
- A Unified Framework for Noisy Image Super-Resolution (2026, ACM Transactions on Autonomous and Adaptive Systems, citations: 0)
- [CV-HoloSR: Hologram to hologram super-resolution through volume-upsampling three-dimensional scenes](https://arxiv.org/abs/2604.10393) (2026, venue n/a, citations: 0)

## BibTeX

```bibtex
@article{sun2020learned,
  title={Learned Image Downscaling for Upscaling Using Content Adaptive Resampler},
  author={Sun, Wanjie and Chen, Zhenzhong},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4027--4040},
  year={2020},
  doi={10.1109/TIP.2020.2970248}
}
```
