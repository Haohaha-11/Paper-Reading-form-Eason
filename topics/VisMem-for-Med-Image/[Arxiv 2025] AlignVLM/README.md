# AlignVLM: Bridging Vision and Language Latent Spaces for Multimodal Document Understanding

**作者**: Ahmed Masry; Juan A. Rodriguez; Tianyu Zhang; Suyuchen Wang; Chao Wang; Aarash Feizi; Akshay Kalkunte Suresh; Abhay Puri; Xiangru Jian; Pierre-André Noël; Sathwik Tejaswi Madhusudhan; Marco Pedersoli; Bang Liu; Nicolas Chapados; Yoshua Bengio; Enamul Hoque; Christopher Pal; Issam H. Laradji; David Vazquez; Perouz Taslakian; Spandana Gella; Sai Rajeswar
**会议**: Arxiv 2025
**链接**: [arXiv 2502.01341](https://arxiv.org/abs/2502.01341) | [PDF](https://arxiv.org/pdf/2502.01341v2)

## 一句话总结

AlignVLM 把 VLM connector 从“任意投影到 LLM embedding space”改成“视觉特征对 LLM 词表 embedding 做加权平均”，用语言模型已有的 text latent prior 约束视觉 token 落在 LLM 更容易解释的区域。

更具体地说，ALIGN 先把每个视觉 patch 特征映射成 LLM 词表上的概率分布，再用该分布加权求和 text embedding matrix，得到视觉输入向量；这样视觉 token 被限制在文本 embedding 的 convex hull 内，减少 MLP connector 常见的 OOD latent 和跨模态错位。

## 核心贡献

1. **ALIGN connector**: 视觉特征 $\breve F$ 先经 $W_1$ 进入 LLM hidden 维度，再经由初始化自 LM head 的 $W_2$ 产生词表概率 $P_{\mathrm{vocab}}$，最后与 LLM text embedding matrix 做加权和得到 $F'_{\mathrm{align}}$。
2. **Convex-hull alignment bias**: 每个视觉 token 都是已有 text embeddings 的凸组合，因此不会像无约束 MLP 一样自由漂到 LLM 未见过的 embedding 区域。
3. **Document VLM 训练范式**: 基于 SigLIP-400M + Llama 3.2/3.1，三阶段训练覆盖通用图文预训练、BigDocs 文档理解和 DocDownstream 指令调优。
4. **Connector 对照充分**: 同训练数据/超参下比较 MLP、Perceiver Resampler、Ovis 与 ALIGN，高资源下 ALIGN avg. score 58.81，高于 Ovis 54.72、MLP 53.06、Perceiver 50.68。
5. **机制分析有信息量**: 低资源实验、3.4K 高概率 token 子集、Gaussian noise perturbation、runtime/memory 和 VCR pixel-level case study 都在验证“对齐归纳偏置”不是只靠规模取胜。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 总览 claim |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 贡献 |
| [02 - Related Work](sections/02-related-work.md) | 相关工作谱系 |
| [03 - Methodology](sections/03-methodology.md) | 核心方法 |
| [04 - Experimental Setup](sections/04-experiments.md) | 实验设置与结果 |
| [05 - Results](sections/05-experiments.md) | 实验结果 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结与局限 |
| [07 - Appendix](sections/07-appendix.md) | 附录补充 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 核心问题 | vision-language latent alignment for VLM connector |
| Vision encoder | SigLIP-400M |
| LLM backbones | Llama 3.2-1B / 3B, Llama 3.1-8B |
| 图像切分 | predefined aspect ratios，最多 9 tiles，每 tile 为 14×14 patches |
| 训练阶段 | CC-12M 可取回 8.1M pairs → BigDocs-7.5M → DocDownstream |
| 高资源 connector avg. | ALIGN 58.81 vs Ovis 54.72 / MLP 53.06 / Perceiver 50.68 |
| 低资源 document avg. | ALIGN 59.32 vs Ovis 48.56 / HoneyBee 47.16 / MLP 40.61 |
| 词表分析 | 128K vocabulary 中约 3.4K high-probability embeddings 已几乎保持性能 |
| 噪声鲁棒性 | $\sigma=3$ feature noise 下 ALIGN drop 1.67，MLP drop 25.54 |
| Runtime 对照 | ALIGN 0.165s/sample, 115.4 tokens/s, 10.9GB；与 MLP/Ovis/Perceiver 接近 |
| 课题作用 | 医学/文档类 VLM 记忆模块之前的视觉证据入 LLM latent space 基础 |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. 文档图像切分 | scanned document / chart / table image | 按覆盖率选择 aspect-ratio grid，最多 9 tiles；每 tile 切成 $14\times14$ patches | patch 序列 $\{\mathbf p_{t,i}\}$ |
| 2. 视觉编码 | patch 序列 | SigLIP-400M vision encoder 提取上下文化视觉特征 | tile features $\mathbf F_t$，拼接成 $\breve{\mathbf F}\in\mathbb R^{T\cdot N_t\times d}$ |
| 3. ALIGN 概率映射 | $\breve{\mathbf F}$ | $W_1\in\mathbb R^{D\times d}$ 投到 LLM hidden 维度；$W_2\in\mathbb R^{V\times D}$ 从 LM head 初始化并接 softmax | 每个视觉 token 的词表分布 $\mathbf P_{\mathrm{vocab}}\in\Delta^{V-1}$ |
| 4. Text embedding 加权 | $\mathbf P_{\mathrm{vocab}}$ + LLM embedding matrix $\mathbf E_{\mathrm{text}}\in\mathbb R^{V\times D}$ | 用概率分布对 text embeddings 做 weighted sum | 对齐后的视觉向量 $\mathbf F'_{\mathrm{align}}=\mathbf P_{\mathrm{vocab}}\mathbf E_{\mathrm{text}}$ |
| 5. 多模态输入拼接 | $\mathbf F'_{\mathrm{align}}$ + 文本问题 token embeddings $\mathbf E_{\mathrm{text}}(\mathbf x)$ | 在 LLM input embedding 序列层面 concat | $\mathbf H_{\mathrm{input}}=[\mathbf F'_{\mathrm{align}};\mathbf E_{\mathrm{text}}(\mathbf x)]$ |
| 6. LLM 解码 | $\mathbf H_{\mathrm{input}}$ | Llama 3.2/3.1 自回归生成 | 文档问答、表格推理、OCR/信息抽取答案 |
| 7. 三阶段训练 | CC-12M / BigDocs-7.5M / DocDownstream | Stage 1-2 训练 full model，Stage 3 冻结 vision encoder、训练 LLM + ALIGN | ALIGNVLM 系列模型 |

**整体输入**: 文档图像 + 用户文本问题/指令 + 预训练 LLM text embedding matrix
**关键中间表示**: 视觉 patch features、词表概率分布 $P_{\mathrm{vocab}}$、text-embedding convex combination、拼接后的 LLM input embeddings
**整体输出**: 可解释文档内容的自然语言答案；训练后得到一个用 ALIGN connector 接入 LLM 的 VLM。

## 优缺点与还能做什么

### 优点
- 思路很直接：不再让 connector 自己学一块任意 latent，而是借 LLM 已有词表 embedding 做视觉 token 的坐标系。
- 归纳偏置明确：convex hull 约束把视觉输入限制在 LLM 训练时熟悉的 text embedding manifold 附近，尤其适合文字、表格、票据、表单这类视觉-文本高度相关场景。
- 对照公平性较好：高资源 connector 对照使用相同数据、训练阶段和超参，能较干净地隔离 connector 设计的影响。
- 低资源收益明显：LLaVA-NeXT 小数据设置下，ALIGN 在 document avg. 上比 Ovis/HoneyBee/MLP 高一截，说明该 bias 确实能省数据。
- 分析不只停在主表：3.4K token subset、PCA、噪声扰动、runtime/memory、VCR 像素级案例都帮助理解 connector 的行为。

### 局限 / 风险
- 凸组合约束可能偏向高频/常见 text tokens；附录 VCR 失败例中专名被替换成更常见词，正是这个风险。
- ALIGN 的优势最自然地出现在 document understanding；对自然图像、医学影像、遥感等“视觉概念不直接对应词表局部”的场景，效果需要重新验证。
- $V=128K$ 词表 softmax 和 embedding 加权仍是额外开销；虽然当前 runtime 接近其他 connector，但高分辨率、多图或长上下文场景可能需要 pruning/approximation。
- 主表把自训 base+ 模型和外部 Instruct 模型放在一起，外部模型训练数据不透明，SOTA 对比只能作参考，不能当完全公平结论。
- 论文没有充分讨论 embedding convex hull 是否会限制细粒度数值、罕见术语、医学缩写或跨语言 token 的表达。

### 还能做什么
- 把 3.4K high-probability token subset 做成训练/推理一体化的可学习 vocabulary router，减少 softmax 与矩阵乘开销。
- 在医学 VLM 中测试：报告图像、病理切片、影像文字标注、结构化化验单是否同样受益于 text embedding convex combination。
- 加入 domain vocabulary 或 ontology embeddings，例如 UMLS、RadLex、药品名、检查项目名，缓解常见词偏置和专名错误。
- 分析不同 token 类型的贡献：数字、空白、标点、OCR 字符、医学缩写是否被 ALIGN 高概率使用。
- 与 memory 模块结合：先用 ALIGN 得到 LLM 可解释的视觉 token，再把高置信视觉-文本片段写入外部记忆或病例级 longitudinal memory。

## 阅读 Q&A 记录

- **Q: ALIGN 和普通 MLP connector 最本质的差别是什么？**
  A: MLP 直接输出任意 $D$ 维向量；ALIGN 先输出词表概率，再用 text embedding matrix 加权求和。因此 ALIGN 的输出被限制在 text embeddings 的 convex hull 内。
- **Q: 为什么这种约束对文档理解特别有用？**
  A: 文档图像中大量视觉证据本来就是文字、数字、表格结构和符号，和 LLM 词表空间高度相关；把视觉 token 拉回 text latent space 更容易被 LLM 解码。
- **Q: $W_2$ 为什么要从 LM head 初始化？**
  A: LM head 已经学习了 hidden state 到词表语义的映射，用它初始化 $W_2$ 等于把视觉特征分类到 LLM 熟悉的 token 语义坐标上，减少从零学习连接器的负担。
- **Q: 3.4K token 分析说明了什么？**
  A: ALIGN 并不是平均使用 128K 词表；它稳定使用约 3.4K 个高概率 embedding，而且只保留这些 embedding 时 avg. score 58.69，几乎等于 full embedding 的 58.81。
- **Q: 噪声鲁棒性为什么能支持论文 claim？**
  A: 加 $\sigma=3$ Gaussian noise 后，MLP avg. 从 53.06 掉到 27.52，而 ALIGN 从 58.81 只掉到 57.14；这说明 text embedding convex hull 起到了正则化作用。
- **Q: ALIGN 会不会过度语言化视觉特征？**
  A: 有这个风险。VCR case 中专名一字之差时模型会偏向更常见词，说明凸组合和词表 prior 可能牺牲罕见视觉细节。
- **Q: 这篇和 VisMem / Med Image 主题怎么连接？**
  A: 它不是医学论文，但解决的是“视觉证据如何进入 LLM latent space”。医学 VLM 的记忆、检索、病例更新都依赖这个入口是否稳定、可解释、抗噪。

## 📊 Citation Landscape

> 数据来源: Semantic Scholar detail 暂缺或限流 | [Connected Papers](https://www.connectedpapers.com/main/2502.01341)

### TLDR (Semantic Scholar 自动生成)

> 暂缺。

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献数 | 暂缺 |
| 被引次数 | 暂缺 |
| Influential Citations | 暂缺 |

### 参考文献分组 (Top 5 per category, by citations)

#### VLM / LLM Foundations
- **Learning Transferable Visual Models From Natural Language Supervision** (2021) — 47720 citations [arXiv](https://arxiv.org/abs/2103.00020)
- **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** (2019) — 25419 citations [arXiv](https://arxiv.org/abs/1910.10683)
- **GPT-4 Technical Report** (2023) — 24033 citations [arXiv](https://arxiv.org/abs/2303.08774)
- **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution** (2024) — 3923 citations [arXiv](https://arxiv.org/abs/2409.12191)
- **Sigmoid Loss for Language Image Pre-Training** (2023) — 2908 citations [arXiv](https://arxiv.org/abs/2303.15343)

#### Multimodal Instruction / General VQA
- **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** (2023) — 7835 citations [arXiv](https://arxiv.org/abs/2301.12597)
- **Improved Baselines with Visual Instruction Tuning** (2023) — 4959 citations [arXiv](https://arxiv.org/abs/2310.03744)
- **GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering** (2019) — 3005 citations
- **MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI** (2023) — 2023 citations [arXiv](https://arxiv.org/abs/2311.16502)
- **Towards VQA Models That Can Read** (2019) — 1977 citations [arXiv](https://arxiv.org/abs/1904.08920)

#### Document / Table / Chart Reasoning
- **Language Models are Few-Shot Learners** (2020) — 57399 citations [arXiv](https://arxiv.org/abs/2005.14165)
- **The Llama 3 Herd of Models** (2024) — 14697 citations [arXiv](https://arxiv.org/abs/2407.21783)
- **ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning** (2022) — 1420 citations [arXiv](https://arxiv.org/abs/2203.10244)
- **MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities** (2023) — 1188 citations [arXiv](https://arxiv.org/abs/2308.02490)
- **TabFact: A Large-scale Dataset for Table-based Fact Verification** (2019) — 671 citations [arXiv](https://arxiv.org/abs/1909.02164)

#### Other Foundations
- **Deep Residual Learning for Image Recognition** (2015) — 226621 citations [arXiv](https://arxiv.org/abs/1512.03385)
- **Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts** (2021) — 1458 citations [arXiv](https://arxiv.org/abs/2102.08981)
- **VCR: Visual Caption Restoration** (2024) — 10 citations
- **Making transparency meaningful: A framework for policymakers** (2021) — None citations
- **1.5: Unified structure learning** (None) — None citations

#### Document Understanding
- **Compositional Semantic Parsing on Semi-Structured Tables** (2015) — 997 citations [arXiv](https://arxiv.org/abs/1508.00305)
- **FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents** (2019) — 485 citations [arXiv](https://arxiv.org/abs/1905.13538)
- **CORD: A Consolidated Receipt Dataset for Post-OCR Parsing** (2019) — 323 citations
- **Deepform: Understand structured documents at scale** (2020) — None citations

#### Vision-Language Alignment
- **Open frontier-class multimodal llms** (None) — None citations
- **Locality-enhanced projector for multimodal** (None) — None citations
- **: Decoupling visual encoding for unified multimodal understanding and** (None) — None citations
- **A versatile 3b vlm** (None) — None citations

#### Memory / Agent Memory
- **DeepSpeed- Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale** (2022) — 578 citations [arXiv](https://arxiv.org/abs/2207.00032)

### 🔗 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | 引用数 |
|------|------|--------|
| [One Token per Highly Selective Frame: Towards Extreme Compression for Long Video Understanding](https://arxiv.org/abs/2604.14149) | 2026 | 0 |
| [Hierarchical Pre-Training of Vision Encoders with Large Language Models](https://arxiv.org/abs/2604.00086) | 2026 | 0 |
| [Firebolt-VL: Efficient Vision-Language Understanding with Cross-Modality Modulation](https://arxiv.org/abs/2604.04579) | 2026 | 0 |
| [Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation](https://arxiv.org/abs/2604.24763) | 2026 | 0 |
| [MMCORE: MultiModal COnnection with Representation Aligned Latent Embeddings](https://arxiv.org/abs/2604.19902) | 2026 | 0 |
| [CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](https://arxiv.org/abs/2603.21077) | 2026 | 0 |
| [How to Utilize Complementary Vision-Text Information for 2D Structure Understanding](https://arxiv.org/abs/2603.16245) | 2026 | 0 |
| [Point Cloud as a Foreign Language for Multi-modal Large Language Model](https://arxiv.org/abs/2603.09173) | 2026 | 0 |
| [LinguDistill: Recovering Linguistic Ability in Vision-Language Models via Selective Cross-Modal Distillation](https://arxiv.org/abs/2604.00829) | 2026 | 0 |
| [CoME-VL: Scaling Complementary Multi-Encoder Vision-Language Learning](https://arxiv.org/abs/2604.03231) | 2026 | 0 |

## BibTeX

```bibtex
@article{masry2025alignvlm,
  title={ AlignVLM: Bridging Vision and Language Latent Spaces for Multimodal Document Understanding },
  author={ Ahmed Masry and Juan A. Rodriguez and Tianyu Zhang and Suyuchen Wang and Chao Wang and Aarash Feizi and Akshay Kalkunte Suresh and Abhay Puri and Xiangru Jian and Pierre-André Noël and Sathwik Tejaswi Madhusudhan and Marco Pedersoli and Bang Liu and Nicolas Chapados and Yoshua Bengio and Enamul Hoque and Christopher Pal and Issam H. Laradji and David Vazquez and Perouz Taslakian and Spandana Gella and Sai Rajeswar },
  journal={arXiv preprint arXiv:2502.01341},
  year={ 2025 }
}
```
