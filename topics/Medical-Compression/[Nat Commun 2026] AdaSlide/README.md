# Adaptive compression framework for giga-pixel whole slide images

**简称**: AdaSlide  
**作者**: Jonghyun Lee; Lina Takemaru; D. M. Bappy; Ye Sul Jeong; Won-Ki Jeong; Derek Oldridge; Dokyoon Kim; Sangjeong Ahn; Sung Hak Lee  
**期刊**: Nature Communications | **年份**: 2026  
**链接**: [Nature](https://www.nature.com/articles/s41467-025-66889-0) | [PDF](https://www.nature.com/articles/s41467-025-66889-0_reference.pdf)  
**本地材料**: [full.md](full.md), [paper.pdf](paper.pdf), [content_list.json](content_list.json)

## 一句话总结

AdaSlide 用强化学习 CDA 对 WSI patch 做 keep/compress 二元决策，再用 FIE 复原被压缩 patch，在 13 个病理下游任务上把存储降到原始大小的约 10-35% 同时尽量保住诊断性能。

## 核心贡献

1. 提出 information disequilibrium：WSI 中不同 ROI 的临床信息密度和复原难度不均衡，因此不应统一压缩。
2. 设计 RL-based Compression Decision Agent (CDA)：以 patch 为单位选择 Keep 或 Compress，用 HoverNet 细胞分割一致性作为信息损失惩罚，不需要人工标注压缩标签。
3. 引入 Foundational Image Enhancer (FIE)：在多器官、多倍率 PanCancer 数据上训练 VAE/VQVAE/ESRGAN/SwinIR/LDM 等复原骨干，并在 AdaSlide 演示中采用更实用的 ESRGAN。
4. 用 lambda 控制压缩倾向：lambda 越高，信息保存惩罚越强，CDA 越保守，压缩率通常降低。
5. 在 13 个 patch/slide-level 任务、VTT 病理专家测试、WSI 级应用中验证压缩-性能权衡，报告 65-90% 以上存储降低的归档潜力。

## 批读导航

| Section | 内容 |
|---|---|
| [00 - Abstract](sections/00-abstract.md) | 摘要主张、关键数字、阅读入口 |
| [01 - Introduction](sections/01-introduction.md) | WSI 存储瓶颈、uniform compression 问题、AdaSlide 约束 |
| [02 - Results](sections/02-results.md) | AdaSlide 总览、information disequilibrium、FIE、CDA、13 个下游任务、lambda 选择、WSI 应用 |
| [03 - Discussion](sections/03-discussion.md) | 与 uniform/MIL/foreground cropping 方法的差异、局限、未来方向 |
| [04 - Methods](sections/04-methods.md) | 数据集、FIE/CDA 训练、reward、下游任务、VTT、评价指标 |
| [05 - Appendix](sections/05-appendix.md) | 数据/代码可用性、利益冲突、表格、图注、BibTeX |

## 关键数字

| 指标 | 数值 |
|---|---|
| FIE 预训练数据 | TCGA 31 projects, 930 WSIs, 1.8M patches |
| PanCancer split | train 1,766,502; val 93,307; test 9,393 patches |
| 外部 FIE 评估 | CPTAC 110 WSIs, 29,861 patches |
| 下游任务 | 13 个：5 patch classification, 3 patch segmentation, 5 slide classification |
| VTT | 5 位病理专家, 30 questions, 约 55-56% accuracy, not significant vs chance |
| 摘要报告存储 | 原始大小的 10-35% |
| 贡献段落报告 | 65-90% storage reduction |
| AdaSlide lambda grid | 0.1, 0.25, 0.5, 0.75, 1.0 |
| 默认 lambda 建议 | 0.5；细粒度/OOD 风险高时 1.0 |
| NuInsSeg 例子 | AdaSlide1.00SwinIR Dice 0.7068 vs baseline 0.7221；uniform models near 0 |
| WSI 应用例子 | lambda 0.50, CR 21.74%, tumor/non-tumor AUROC 0.6288 |
| Hybrid 存储演示 | 平均 363.1 MB, 11.19% |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|---|---|---|---|
| 1. WSI tessellation | gigapixel WSI | foreground/tissue selection, patch extraction | 512x512 patch set |
| 2. FIE training | HR patches x | downscale/encode to low-res or latent, decode/enhance | restored patch x_hat; fixed FIE for CDA reward |
| 3. CDA training | patch p plus original/restored HoverNet masks | REINFORCE single-step action, reward = compression gain minus lambda-weighted information penalty | policy deciding Keep/Compress |
| 4. Encoding | each WSI patch | CDA predicts action; Compress patches become 128x128 or latent; Keep patches stay original | compressed set S_P and non-compressed set S_N |
| 5. Decoding | compressed set S_P | FIE decoder/enhancer reconstructs original pixel size | enhanced set S_E |
| 6. WSI reconstruction | S_N union S_E | pyvips rebuilds TIFF/WSI grid; background omitted in demo | smaller reconstructed WSI for diagnosis or AI |

## 优缺点与还能做什么

### 优点

- 不是盲目追求最低文件大小，而是把细胞信息和复原难度写进 reward，直接对准病理诊断风险。
- CDA 是任务无关的 patch-level agent，比依赖某个 MIL classifier attention map 的 adaptive compression 更容易跨任务复用。
- lambda 提供可解释控制杆：更高 lambda 对应更保守的压缩策略，适合 OOD 或细粒度任务。
- 实验覆盖 patch classification、patch segmentation、slide classification，并用 VTT 补充人类视觉可感知质量。

### 局限 / 风险

- CDA 只有 Keep/Compress 二元动作，不能直接学习多档压缩等级。
- reward 当前把细胞密度/细胞分割一致性作为信息代理，对低细胞性但临床重要的脂肪性肿瘤等场景可能失灵。
- CDA 训练偶尔不稳定，lambda 与最终 CR/下游性能不是严格单调可控。
- CR 主要按 raw pixel 维度计算，和真实 JPEG/PNG/SVS 存储大小存在差异；VQVAE 的浮点 latent 在真实文件格式下可能反而更大。
- FIE 和 CDA 都有 OOD 风险：NuInsSeg、CBTN、Camelyon16 的结果说明域偏移会改变最佳策略。

### 还能做什么

- 把 CDA 从二元动作扩展为多档压缩动作，并在 reward 中加入显式 slide/group-level CR 约束。
- 用 PPO 等更稳定的 policy optimization 替代单步 REINFORCE，减少 CDA 不收敛。
- 用 VLM/病理语义模型替代或补充 HoverNet 细胞 mask，让信息惩罚覆盖脂肪、神经周围浸润、血管侵犯等任务。
- 训练能直接处理 JPEG 压缩低分辨率 patch 的 FIE，减少 PNG-only 压缩分支的存储开销。
- 在 slide-level CDA 中联合多个 ROI，避免每个 patch 独立决策忽视空间上下文。

## 阅读 Q&A 记录

- **Q: AdaSlide 真正改变的是压缩算法本身，还是压缩调度策略？**  
  A: 主要是压缩调度策略。FIE 可以换成 ESRGAN/VQVAE/SwinIR/LDM 等复原模型，CDA 决定每个 patch 是否进入 FIE 压缩-复原路径。

- **Q: information disequilibrium 在本文中如何落地？**  
  A: 它被落到两个可计算代理上：细胞丰富程度代表 tumor-related informativeness，压缩前后 HoverNet mask Dice 代表复原难度/信息损失。

- **Q: lambda 控制了什么？**  
  A: lambda 放大信息保存惩罚。lambda 越大，Compress 动作更难拿到正 reward，CDA 更倾向 Keep，CR 下降但信息风险更低。

- **Q: 为什么 VTT 近 chance level 是好结果？**  
  A: VTT 问病理专家区分原图和 ESRGAN 复原图；55-56% 接近 50% 且不显著，说明专家不容易看出复原痕迹。

- **Q: 13 个任务的证据链最关键看什么？**  
  A: 分类可接受 global pattern 的压缩，分割更依赖 local morphology。NuInsSeg uniform compression near-zero Dice 是强反例，说明自适应保留难复原细节很重要。

## Citation Landscape

Semantic Scholar API 在本次生成时返回 429 rate limit，因此这里采用 graceful fallback：基于本地 `full.md`、`paper-meta.json` 和论文正文可见引用脉络整理，不填造 citationCount / recommendations。

| 项目 | 当前记录 |
|---|---|
| Semantic Scholar TLDR | 未获取到，API rate limited |
| Reference count | 未获取到，MinerU 正文 references 未展开 |
| Citation count | 未获取到 |
| Influential citation count | 未获取到 |
| Semantic Scholar / Connected Papers | 可用标题检索：`Adaptive compression framework for giga-pixel whole slide images` |

### 参考文献脉络分组（正文可见）

| 主题 | 论文脉络 | 与 AdaSlide 的关系 |
|---|---|---|
| Digital pathology / CPath foundation models | 大规模 WSI 支撑诊断、预后、分割和 foundation model | 提供存储压力的背景，而非只优化预测精度 |
| Neural image compression for pathology | VAE, BiGAN, VQVAE-2 等 encoder-decoder 压缩复原 | AdaSlide 借用 FIE 思路，但拒绝 uniform compression |
| Super-resolution | ESRGAN, SwinIR, LDM 等从低分辨率恢复高分辨率 | FIE 候选骨干，ESRGAN 因效率和视觉质量用于演示 |
| MIL / attention-based adaptive decompression | 用 slide label 训练 MIL attention 找重要 ROI | AdaSlide 的差异是任务无关 CDA，不依赖某个下游监督标签 |
| Cell segmentation / pathology VLM | HoverNet, PLIP | HoverNet 构造信息损失惩罚，PLIP 检验 CDA 是否保留 cell-rich patch |
| Slide-level MIL downstream | CLAM | 评估 AdaSlide 复原后的 slide-level 分类可用性 |

## BibTeX

```bibtex
@article{lee2026adaslide,
  title = {Adaptive compression framework for giga-pixel whole slide images},
  author = {Lee, Jonghyun and Takemaru, Lina and Bappy, D. M. and Jeong, Ye Sul and Jeong, Won-Ki and Oldridge, Derek and Kim, Dokyoon and Ahn, Sangjeong and Lee, Sung Hak},
  journal = {Nature Communications},
  year = {2026},
  url = {https://www.nature.com/articles/s41467-025-66889-0}
}
```
