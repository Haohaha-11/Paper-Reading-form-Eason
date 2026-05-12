# Time-Aware One Step Diffusion Network for Real-World Image Super-Resolution

**作者**: Tianyi Zhang; Zheng-Peng Duan; Peng-Tao Jiang; Bo Li; Ming-Ming Cheng; Chun-Le Guo; Chongyi Li
**会议**: CVPR 2026 | **年份**: 2026
**链接**: [arXiv 2508.16557](https://arxiv.org/abs/2508.16557) | [PDF](https://arxiv.org/pdf/2508.16557) | [Code](https://github.com/zty557/TADSR)

## 一句话总结
TADSR 把 timestep 从 one-step diffusion SR 中被固定忽略的条件，变成同时控制 latent 编码和 VSD teacher guidance 的变量，从而单次推理即可调节 fidelity-realism trade-off。

## 核心贡献
1. 指出现有 one-step VSD SR 的结构性错位：student 固定 timestep，teacher 随机 timestep，导致 SD 在不同时间步的 generative prior 没有被一致使用。
2. 提出 Time-Aware VAE Encoder (TAE)，让同一 LQ 图像在不同 $t_s$ 下编码成不同 latent 分布，而不是只给 UNet 一个时间标量。
3. 提出 Time-Aware VSD (TAVSD)，把 student timestep $t_s$ 映射到 teacher/LoRA timestep $t_v$，让 score guidance 随时间条件一致变化。
4. 用 timestep 自然控制保真与真实感：小 $t_s$ 偏结构保真，大 $t_s$ 偏语义真实感和纹理生成。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 总览 claim |
| [01 - Introduction](sections/01-introduction.md) | 固定 timestep 问题与贡献 |
| [02 - Related Work](sections/02-related-work.md) | Real-ISR 与 diffusion one-step 背景 |
| [03 - Methodology](sections/03-methodology.md) | TAE、TAVSD、训练损失 |
| [04 - Experiments](sections/04-experiments.md) | 主结果、可控性和消融 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 |
| [06 - Appendix](sections/06-appendix.md) | 推导、更多消融、效率和视觉补充 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Default quantitative timestep | $t_s=500$ |
| Training data | LSDIR, 512x512 patches |
| Degradation | Real-ESRGAN pipeline |
| Pretrained model | SD 2.1-base |
| Training | 2k iterations, 8 NVIDIA A40, batch size 24 |
| LoRA rank | 4 |
| DRealSR example | $t_s=200$ gives 26.61dB PSNR, QALIGN higher than SinSR |
| Semantic Scholar status | API 429 in this session |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. Time sampling | HQ-LQ pair | sample student timestep $t_s \sim U(0,999)$ | 时间条件 $t_s$ |
| 2. Time-aware encoding | LQ image + $t_s$ | TAE $E_\theta(x_L,t_s)$ 编码同一图像的不同 latent 分布 | $z_L$ |
| 3. One-step SR | $z_L$ + $t_s$ | LoRA UNet $F_\theta$ 输出 reconstructed latent | $\hat z_0$ |
| 4. Teacher guidance | $\hat z_0$ + mapped $t_v$ | 加噪得到 $\hat z_{t_v}$，teacher/LoRA 计算 TAVSD residual | time-consistent generative gradient |
| 5. Student update | blurred MSE + LPIPS + TAVSD | 低频保真与高频真实感共同训练 | TADSR student |
| 6. Inference | LQ image + chosen $t_s$ | 单次前向 | 可调真实感的 HR image |

## 优缺点与还能做什么

### 优点
- 把 timestep 变成可解释控制变量，比固定一步映射更灵活。
- TAE 解决“同 latent + 不同 timestep”难以激活不同 prior 的问题。
- TAVSD 让 teacher/LoRA guidance 与 student condition 一致，减少随机 timestep 造成的冲突梯度。
- 实验同时给出主结果、timestep 曲线、TAE/TAVSD 消融和效率对比。

### 局限 / 风险
- 大 timestep 会更强生成语义细节，可能带来 hallucination，尤其在文字、人脸身份和规则结构上要谨慎。
- Semantic Scholar 本次限流，Citation Landscape 暂无法读取实时引用数。
- $t_s \rightarrow t_v$ 映射与超参可能依赖退化类型，跨数据集部署需要校准。
- 仍依赖 SD2.1-base、DAPE 和 LoRA 蒸馏，上限受 teacher prior 与训练退化分布限制。

### 还能做什么
- 学习退化感知 timestep policy，根据输入自动选择 $t_s$。
- 与 TSD-SR 的 target score / DASM 结合，让 time-aware guidance 方向更可靠。
- 给可控 SR 增加结构一致性检测，避免大 timestep 下的过度语义补全。
- 扩展到视频 SR，验证 timestep 控制是否会造成 temporal flicker。

## 阅读 Q&A 记录

- **Q: 为什么只随机采样 timestep 不够？**
  A: 原 VAE encoder 对同一图只给一个 latent，UNet 的主要视觉输入不变；TAE 让 latent 分布也随 $t_s$ 变化，才更接近 SD 原始扩散过程。
- **Q: TAVSD 中 $t_s$ 和 $t_v$ 分别是什么？**
  A: $t_s$ 是 student 接收的控制时间，$t_v$ 是映射后用于 teacher/LoRA score guidance 的时间；二者关联保证 guidance 与 student 输出状态一致。
- **Q: timestep 怎么控制 fidelity-realism？**
  A: 小 $t_s$ 时 TAVSD 梯度主要强化纹理细节，保真约束更强；大 $t_s$ 时 teacher 激活更多语义 prior，真实感增强但 PSNR 往往下降。
- **Q: TADSR 和 PisaSR 的可控性区别是什么？**
  A: PisaSR 调 semantic weight 主要增加锐度/语义强度，TADSR 调 timestep 会同步改变 latent 和 teacher prior，更接近 SD 原生时间语义。

## 📊 Citation Landscape

> Semantic Scholar detail endpoint 本次返回 HTTP 429 rate limit；本目录保留了 `semantic-scholar/*.json` fallback，不编造 TLDR、citation count 或 influential citation count。
> [Semantic Scholar 查询](https://www.semanticscholar.org/search?q=Time-Aware%20One%20Step%20Diffusion%20Network%20for%20Real-World%20Image%20Super-Resolution) | [Connected Papers](https://www.connectedpapers.com/main/2508.16557)

### TLDR (Semantic Scholar 自动生成)

> 暂缺，API 当前限流。

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献数 | 暂缺 |
| 被引次数 | 暂缺 |
| Influential Citations | 暂缺 |
| Semantic Scholar 状态 | HTTP 429 Too Many Requests |

### 参考文献分组（基于正文 references 脉络）

#### Time-aware / Diffusion SR 前置方法
- **TSD-SR** (CVPR 2025): TADSR 直接引用的 one-step target score distillation baseline。
- **OSEDiff** (NeurIPS 2024): VSD one-step Real-ISR 的关键前作。
- **PisaSR** (CVPR 2025): pixel/semantic dual-LoRA 可控 SR baseline。
- **AdcSR** (CVPR 2025): adversarial diffusion compression one-step baseline。
- **S3Diff** (2025): degradation-guided LoRA 与 adversarial one-step SR baseline。

#### Generative Priors / Evaluation
- **Stable Diffusion / Latent Diffusion**: TADSR 的 teacher prior 来源。
- **Real-ESRGAN**: 训练退化 pipeline 来源。
- **LSDIR**: 训练数据来源。
- **CLIPIQA / MUSIQ / MANIQA / TOPIQ / QALIGN**: no-reference perceptual quality evaluation。

### 🔗 推荐论文 (fallback)

Semantic Scholar recommendations 本次限流；优先继续读同 topic 中的 TSD-SR、OSEDiff、OFTSR、PisaSR、AdcSR，以及后续的 time-aware distillation / realism-control one-step SR 工作。

## BibTeX

```bibtex
@article{zhang2025tadsr,
  title={Time-Aware One Step Diffusion Network for Real-World Image Super-Resolution},
  author={Zhang, Tianyi and Duan, Zheng-Peng and Jiang, Peng-Tao and Li, Bo and Cheng, Ming-Ming and Guo, Chun-Le and Li, Chongyi},
  journal={arXiv preprint arXiv:2508.16557},
  year={2025}
}
```
