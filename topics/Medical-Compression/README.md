# Medical-Compression

医学影像压缩 / whole-slide image storage / adaptive downscaling：围绕病理 WSI 的无损压缩、诊断保持型有损压缩、few-shot visual token compression、latent autoencoder 表征压缩和可上采样的内容自适应下采样。

## 论文列表

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [AdaSlide](./%5BNat%20Commun%202026%5D%20AdaSlide/) | Nature Communications 2026 | AdaSlide 用 RL Compression Decision Agent 按 patch 临床信息量选择保留或压缩，再用 Foundational Image Enhancer 复原，在 WSI 存储成本与诊断完整性之间做自适应权衡。 |
| [WISE](./%5BArxiv%202025%5D%20WISE/) | arXiv 2025 | WISE 面向 WSI 无损压缩，利用空白区编码、层次投影、bitmap 重排和字典编码处理高频且不规则的病理图像信息。 |
| [FOCUS](./%5BCVPR%202025%5D%20FOCUS/) | CVPR 2025 | FOCUS 用 pathology foundation model 与语言先验做三阶段 visual token compression，在 few-shot WSI 分类中保留诊断相关 patch 并过滤冗余视觉 token。 |
| [Pathology-AE](./%5BArxiv%202025%5D%20Pathology-AE/) | arXiv 2025 | Pathology-AE 将 latent diffusion autoencoder 复用于病理压缩，用 pathology foundation model 感知指标微调，并用 K-means latent quantization 提升存储效率。 |
| [NIC](./%5BTPAMI%202020%5D%20NIC/) | IEEE TPAMI 2020 | NIC 把 gigapixel WSI patch 编成低维 embedding 网格，让 CNN 在压缩表示上用弱图像级标签完成分类、回归和可视化定位。 |
| [CAR](./%5BTIP%202020%5D%20CAR/) | IEEE TIP 2020 | CAR 学习内容自适应下采样 kernel 与 offsets，并用 SRNet 反向指导低分辨率图像保留可恢复细节。 |

## 课题主线

1. **存储目标不同**：WISE 坚持 bit-exact lossless；AdaSlide 接受诊断保持型有损压缩；FOCUS 压缩 few-shot WSI 分类中的视觉 token；Pathology-AE 压缩 AE latent 并重建图像；NIC 将压缩表示直接服务于弱监督分析；CAR 则从“低分辨率可恢复”角度讨论下采样。
2. **医学风险约束不同**：WISE 避免任何像素失真；AdaSlide 用信息惩罚、病理专家 VTT 和 13 个下游任务验证诊断完整性；FOCUS 用 few-shot 分类收益验证 token filtering 没有丢掉关键诊断 patch；Pathology-AE 用 foundation-model embedding similarity 与下游 segmentation/classification/MIL 评估重建是否可用；NIC 关注图像级任务信号是否保留；CAR 不是医学专用，但提供可学习 resampling 的通用机制。
3. **核心中间表示不同**：WISE 是 delta/bitmap/dictionary byte stream；AdaSlide 是 keep/compress action + FIE reconstruction；FOCUS 是 FM feature、language prompt relevance 和 sequential token subset；Pathology-AE 是 LDM AE latent 与 K-means code；NIC 是 spatial embedding grid；CAR 是 per-pixel resampling kernel weights 和 offsets。

## 关键问题

- WSI 的“空白区域多”和“组织区域高频不规则”分别应该由工程编码还是学习模型处理？
- 诊断保持型压缩的评价应优先看像素指标、病理专家感知、还是下游任务性能？
- 压缩策略能否从任务相关区域自动转向多任务/多器官泛化？
- visual token / latent code 的压缩率应该由码率、下游准确率、还是 pathology foundation model 的语义距离共同约束？
- 低分辨率预览与高分辨率恢复之间，是否可以形成可审计、可控的医学影像存储协议？
