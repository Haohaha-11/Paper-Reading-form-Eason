# Realism Control One-step Diffusion for Real-World Image Super-Resolution

**作者**: Zongliang Wu; Siming Zheng; Peng-Tao Jiang; Xin Yuan
**会议**: AAAI 2026
**链接**: [arXiv 2509.10122](https://arxiv.org/abs/2509.10122) | [PDF](https://arxiv.org/pdf/2509.10122v2)

## 一句话总结

RCOD-SR 把真实退化程度映射到 diffusion timestep/denoising degree，并用 DAS 与 VPIM 保住这个控制链，让 one-step Real-ISR 在单步推理下也能按 $t$ 调节 fidelity-realism trade-off。

## 核心贡献

1. 诊断 fixed timestep 问题：现有 OSD 常用单一 $T$ 训练/蒸馏，把多种未知退化压成一个平均恢复偏好，推理时缺少可调 realism。
2. 提出 LDG：用 latent cosine similarity $M_L$ 衡量 LR-HR latent 偏离程度，将样本分到不同 group/timestep，使 U-Net 学到不同 denoising degree。
3. 提出 DAS：让 teacher/regularization network 的 timestep 采样与 LDG 对齐，避免蒸馏正则把 Fid./Real. 档重新拉回同一个平均风格。
4. 提出 VPIM：用 LR 图像驱动的 visual tokens 替代 VLM/text prompt，降低条件不匹配和额外文本提取开销。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 总览 claim |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 贡献 |
| [02 - Related Work](sections/02-related-work.md) | 相关工作谱系 |
| [03 - Methodology](sections/03-methodology.md) | 核心方法 |
| [04 - Experiments](sections/04-experiments.md) | 实验设置与结果 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结与局限 |
| [06 - Appendix](sections/06-appendix.md) | 附录补充 |


## 关键数字

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| 控制变量 | latent-domain group / denoising degree |
| 会议 | AAAI 2026 Oral |
| 核心问题 | realism controllability |
| 推理档位 | Fid./Neu./Real. 对应 $t=250/500/750$ |
| 训练配置 | 30k iter, batch 4, lr 2e-5, LoRA |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. 训练样本 | HR image + Real-ESRGAN pipeline 合成的多退化 LR | 构造不同类型/强度退化的 LR-HR pairs | 可比较退化差异的训练对 |
| 2. Latent 表示 | LR/HR image | VAE encoder 得到 $z_L, z_H$ | latent-domain LR/HR features |
| 3. 退化度量 | $z_L, z_H$ | 用 cosine similarity 计算 $M_L$，表示 LR latent 偏离 HR latent 的程度 | sample-specific degradation proxy |
| 4. LDG 控制 | $M_L$ | 按 Eq. 4 离散成 group，并映射到 timestep $t$ | Fid./Neu./Real. 等 denoising degree |
| 5. OSD 学习 | $z_L$, $t$, 条件 token | U-Net 单步预测 $\hat z_H$，不同 $t$ 学不同生成强度 | 可按 $t$ 调节的 HR latent |
| 6. DAS 蒸馏 | LDG 的 $t$ + teacher regularization | teacher timestep $t_r$ 按退化 group 对齐采样 | 不抹平 group 差异的蒸馏正则 |
| 7. VPIM 条件 | LR image | CLIP vision + MLP 生成 visual tokens，注入 U-Net cross-attention | 与当前 LR 内容/退化对齐的条件 |
| 8. 推理输出 | LR image + 用户指定 $t$ 或 MEM 自动估计 | 单步得到 $\hat z_H$，VAE decoder 解码 | 不同 fidelity/realism 档位的 SR image |

**整体输入**: 真实退化 LR image + 可选 realism 控制强度
**整体输出**: 可按 realism level 调节的 SR image

## 优缺点与还能做什么

### 优点
- 控制变量落在 diffusion 本来就有的 timestep 上，不是额外训练多个模型或另加重条件分支，解释性较强。
- LDG 把 fixed timestep 的静态映射改成 sample/group-specific denoising degree，使 one-step 模型能保留 Fid./Neu./Real. 多个工作点。
- DAS 补上蒸馏侧的一致性：teacher regularization 随 group 采样，避免把 LDG 的分组差异抹平。
- VPIM 用 LR visual tokens 替代 text prompt/VLM，减少 prompt 与图像不匹配，也可能降低 OSEDiff 路线的 text extractor 开销。
- 可插到 OSEDiff 和 S3Diff 两类 OSD 底座上，实验上比只服务单一模型更有框架意义。

### 局限 / 风险
- 控制效果依赖 $M_L$ 是否能单调排序退化严重度；当前证据主要是 Spearman 相关和训练集分布稳定性。
- $M_L$ 是全图标量/分组，复杂局部退化会被压缩；一张图内“平坦区域要保真、纹理区域要增强”的空间差异还处理不了。
- $n=4$ 消融变差说明档位数量受训练数据分布限制，高 $t$/低 $M_L$ 区间样本不足会让 Real. 档不稳。
- Real. 档提升 NR 指标不等于结构更真实，文字、身份、人脸、细线纹理等区域仍有 hallucination 风险。
- 用户指定 $t$ 的语义尚未校准：同一个 $t=750$ 对不同内容/退化可能对应不同观感强度。
- VPIM 虽减少文本依赖，但引入 CLIP vision + MLP 条件路径；移动端部署仍要实测延迟、显存和鲁棒性。

### 还能做什么
- 学习可校准的 degradation/realism estimator，把用户 slider 映射成跨图像稳定的 perceptual strength，而不是固定 $t$。
- 从全图 group 扩展到空间控制图：平坦区域保真、纹理区域增强、文字/人脸区域保守处理。
- 改进采样或重加权，让高 $t$ group 拥有足够训练样本，从三档控制走向更细粒度连续控制。
- 加入 identity/OCR/structure consistency 或人类偏好评估，补上 NR 指标无法反映的 hallucination 风险。
- 研究 VPIM 与 text prompt 的混合条件：当 LR 本身语义不足时，是否可以安全引入外部语义而不破坏 fidelity。

## 阅读 Q&A 记录

- **Q: 为什么单步 SR 难以控制 realism？**
  A: 因为多数 one-step 模型在固定 timestep/固定蒸馏强度下学习一个确定映射，推理时没有多步采样那样的 noise schedule 可调自由度。
- **Q: LDG 真正控制的是什么？**
  A: 不是直接控制像素纹理，而是控制 denoising degree：退化越重或选择越偏 realism，模型被引导释放更多生成先验。
- **Q: fixed timestep 为什么会造成“平均退化”输出？**
  A: 不同退化强度的 LR 样本都用同一个 $T$ 训练，U-Net 没有条件区分该保守恢复还是大胆生成，只能学一个在训练集上平均最优的恢复风格。
- **Q: DAS 为什么必要？**
  A: LDG 只改变 student 的 timestep 条件；如果 teacher/regularization 仍大范围随机采样，就可能把不同 group 的输出拉近。DAS 让 teacher 采样跟 LDG 对齐，保持 Fid./Real. 分离。
- **Q: VPIM 解决什么痛点？**
  A: Real-ISR 的 text prompt 往往粗糙或外部生成，VPIM 把当前 LR 图像退化线索编码成 visual token，减少 prompt 与图像内容不匹配。
- **Q: 推理时还要算 $M_L$ 吗？**
  A: 手动选档不用，直接指定 $t=250/500/750$；自动档才用 MEM 根据 LR 特征估计 group。
- **Q: 为什么 $n=4$ 没有比 $n=3$ 更好？**
  A: 附录显示高 $t$/低 $M_L$ 区间训练样本太少，导致 Real. 档训练不足。控制粒度受数据分布限制，不是越多档越好。

## 📊 Citation Landscape

> 数据来源: Semantic Scholar detail 暂缺或限流 | [Connected Papers](https://www.connectedpapers.com/main/2509.10122)

### TLDR (Semantic Scholar 自动生成)

> 暂缺。

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献数 | 暂缺 |
| 被引次数 | 暂缺 |
| Influential Citations | 暂缺 |

### 参考文献分组 (Top 5 per category, by citations)

- 参考文献 API 暂缺或限流。

### 🔗 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | 引用数 |
|------|------|--------|
| Realism Control One-step Diffusion for Real-world Image Super Resolution | 2026 | 0 |
| [MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution](https://arxiv.org/abs/2603.20690) | 2026 | 0 |
| [Bridging Restoration and Generation Manifolds in One-Step Diffusion for Real-World Super-Resolution](https://arxiv.org/abs/2604.24136) | 2026 | 0 |
| [Degradation-Aware and Structure-Preserving Diffusion for Real-World Image Super-Resolution](https://arxiv.org/abs/2604.11470) | 2026 | 0 |
| Selective Diffusion Distillation for Real-World High-Scale Image Super-Resolution | 2026 | 0 |
| [GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution](https://arxiv.org/abs/2603.16769) | 2026 | 0 |
| [GramSR: Visual Feature Conditioning for Diffusion-Based Super-Resolution](https://arxiv.org/abs/2604.25457) | 2026 | 0 |
| [QUSR: Quality-Aware and Uncertainty-Guided Image Super-Resolution Diffusion Model](https://arxiv.org/abs/2603.09125) | 2026 | 0 |
| [Revisiting the Perception-Distortion Trade-off with Spatial-Semantic Guided Super-Resolution](https://arxiv.org/abs/2603.14112) | 2026 | 0 |
| [VARestorer: One-Step VAR Distillation for Real-World Image Super-Resolution](https://arxiv.org/abs/2604.21450) | 2026 | 0 |

## BibTeX

```bibtex
@article{wu2026rcodsr,
  title={ Realism Control One-step Diffusion for Real-World Image Super-Resolution },
  author={ Zongliang Wu and Siming Zheng and Peng-Tao Jiang and Xin Yuan },
  journal={arXiv preprint arXiv:2509.10122},
  year={ 2026 }
}
```
