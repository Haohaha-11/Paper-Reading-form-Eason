![Paper Reading Banner](./banner.svg)

# Paper Reading 📚

Haojiang 的文献阅读仓库，按课题组织。每篇论文都有「批读格式」阅读笔记：原文完整保留 + 内嵌批注。

---

## 课题列表

### ⚡ One-Step Diffusion Super-Resolution
单步 diffusion / flow 超分辨率：在保持生成先验感知质量的同时解决推理效率、保真-真实感权衡和可控性。

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [RCOD-SR](./topics/One-Step-Diffusion-Super-Resolution/%5BAAAI%202026%5D%20RCOD-SR/) | AAAI 2026 | RCOD-SR 用 latent domain grouping、退化感知蒸馏和视觉 prompt 注入，让 one-step diffusion SR 具备推理时 realism 控制能力。 |
| [CODSR](./topics/One-Step-Diffusion-Super-Resolution/%5BArxiv%202025%5D%20CODSR/) | Arxiv 2025 | CODSR 通过 LQ-guided feature modulation、区域自适应生成先验激活和 text-matching guidance，在单步 SR 中兼顾局部保真与感知质量。 |
| [TinySR](./topics/One-Step-Diffusion-Super-Resolution/%5BArxiv%202025%5D%20TinySR/) | Arxiv 2025 | TinySR 面向实时 Real-ISR，把 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存压成轻量单步模型。 |
| [OFTSR](./topics/One-Step-Diffusion-Super-Resolution/%5BICLR%202026%5D%20OFTSR/) | ICLR 2026 | OFTSR 用 conditional flow teacher 和 ODE trajectory alignment distillation 构建 one-step SR，并保留可调 fidelity-realism trade-off。 |
| [OSEDiff](./topics/One-Step-Diffusion-Super-Resolution/%5BNeurIPS%202024%5D%20OSEDiff/) | NeurIPS 2024 | OSEDiff 将低质量图像直接作为扩散起点，用 latent-space VSD 把 Stable Diffusion 先验压缩到单步 Real-ISR 推理。 |

📖 [One-Step Diffusion Super-Resolution 详细总结](./topics/One-Step-Diffusion-Super-Resolution/README.md)

---

### 🧪 Diffision_Gzy
单步 diffusion Real-ISR：聚焦 score distillation、timestep control 和 fidelity-realism trade-off，比较如何把多步生成先验压缩到一次前向推理中。

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [TSD-SR](./topics/Diffision_Gzy/%5BCVPR%202025%5D%20TSD-SR/) | CVPR 2025 | TSD-SR 用 Target Score Distillation 和 DASM 修正 one-step VSD 的 score direction 与细节梯度，在 0.1362s 单步推理中提升 Real-ISR 感知质量。 |
| [TADSR](./topics/Diffision_Gzy/%5BCVPR%202026%5D%20TADSR/) | CVPR 2026 | TADSR 用 Time-Aware VAE Encoder 和 Time-Aware VSD 把 timestep 变成 one-step SR 的 fidelity-realism 控制变量。 |

📖 [Diffision_Gzy 详细总结](./topics/Diffision_Gzy/README.md)

---

### 🧠 VisMem for Med Image
latent vision memory / latent reasoning / medical image VLM：让医学影像模型持续保持视觉证据并调用隐式临床经验。

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [AlignVLM](./topics/VisMem-for-Med-Image/%5BArxiv%202025%5D%20AlignVLM/) | Arxiv 2025 | AlignVLM 把 connector 视为 vision-language latent alignment 模块，用更强归纳偏置改善多模态文档理解。 |
| [VisMem](./topics/VisMem-for-Med-Image/%5BArxiv%202025%5D%20VisMem/) | Arxiv 2025 | VisMem 提出 latent vision memory，通过 memory invocation 和 formation 缓解 VLM 长生成中的视觉 grounding 丢失。 |
| [Latent-Space-Survey](./topics/VisMem-for-Med-Image/%5BArxiv%202026%5D%20Latent-Space-Survey/) | Arxiv 2026 | 这篇综述系统梳理 latent space 的基础、演进、机制、能力和展望，是理解 latent memory/reasoning 的概念地图。 |
| [MedSynapse-V](./topics/VisMem-for-Med-Image/%5BArxiv%202026%5D%20MedSynapse-V/) | Arxiv 2026 | MedSynapse-V 面向医学 VLM，提出 latent diagnostic memory evolution，模拟临床专家在诊断时调用和演化隐式经验。 |
| [Visual-Enhanced-Depth-Scaling](./topics/VisMem-for-Med-Image/%5BArxiv%202026%5D%20Visual-Enhanced-Depth-Scaling/) | Arxiv 2026 | Visual Enhanced Depth Scaling 针对 multimodal latent reasoning 中视觉 token 梯度不稳定与语言偏置，提出视觉重放和动态深度扩展。 |

📖 [VisMem for Med Image 详细总结](./topics/VisMem-for-Med-Image/README.md)

---

### 🧩 Latent-Space Processing
LLM latent reasoning / cache augmentation / CoT compression：把额外推理、记忆整理或思维链压缩放到连续表示、hidden state 或 kv-cache 中完成。

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [Differentiable-Cache-Augmentation](./topics/Latent-Space-Processing/%5BArxiv%202024%5D%20Differentiable-Cache-Augmentation/) | Arxiv 2024 | DCA 用离线 coprocessor 读取 frozen LLM 的 kv-cache，生成动态 latent embeddings 追加回 cache，实现 cache-level latent deliberation。 |
| [Compressed-Chain-of-Thought](./topics/Latent-Space-Processing/%5BArxiv%202024%5D%20Compressed-Chain-of-Thought/) | Arxiv 2024 | CCoT 把显式 CoT hidden states 压缩成 contentful continuous contemplation tokens，用 compression ratio 控制准确率/延迟折中。 |
| [Latent-Implicit-Visual-Reasoning](./topics/Latent-Space-Processing/%5BArxiv%202025%5D%20Latent-Implicit-Visual-Reasoning/) | Arxiv 2025 | LIVR 用 latent visual reasoning tokens 和 visual bottleneck masking，让 LMM 在无显式视觉 CoT 监督下学习视觉中心推理。 |
| [DMLR](./topics/Latent-Space-Processing/%5BArxiv%202026%5D%20DMLR/) | Arxiv 2026 | DMLR 在测试时优化 latent think tokens，并动态注入相关视觉 patches，提升多模态数学、视觉和组合推理。 |

📖 [Latent-Space Processing 详细总结](./topics/Latent-Space-Processing/README.md)

---

### 🩺 Medical Compression
医学影像压缩 / whole-slide image storage / adaptive downscaling：围绕病理 WSI 的无损压缩、诊断保持型有损压缩、few-shot visual token compression、latent 表征压缩和可上采样的内容自适应下采样。

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [AdaSlide](./topics/Medical-Compression/%5BNat%20Commun%202026%5D%20AdaSlide/) | Nature Communications 2026 | AdaSlide 用 RL Compression Decision Agent 按 patch 临床信息量选择保留或压缩，再用 Foundational Image Enhancer 复原，在 WSI 存储成本与诊断完整性之间做自适应权衡。 |
| [DTC-WSI](./topics/Medical-Compression/%5BMIDL%202026%5D%20DTC-WSI/) | MIDL 2026 | DTC-WSI 用 importance network 和动态多阶段 token compression，把 bipartite soft matching token fusion 与 importance-guided pruning 结合，降低 WSI MIL 的 token aggregation 开销。 |
| [WSIR2](./topics/Medical-Compression/%5BBSPC%202026%5D%20WSIR2/) | Biomedical Signal Processing and Control 2026 | WSIR2 用 cross-attention 重要性选择 top-k patches、合并非关键 token 并配轻量 classifier，面向 WSI MIL 诊断加速；已基于本地 PDF 补齐全文批读和图表。 |
| [WISE](./topics/Medical-Compression/%5BArxiv%202025%5D%20WISE/) | Arxiv 2025 | WISE 面向 WSI 无损压缩，利用空白区编码、层次投影、bitmap 重排和字典编码处理高频且不规则的病理图像信息。 |
| [FOCUS](./topics/Medical-Compression/%5BCVPR%202025%5D%20FOCUS/) | CVPR 2025 | FOCUS 用 pathology foundation model 与语言先验做三阶段 visual token compression，在 few-shot WSI 分类中保留诊断相关 patch 并过滤冗余视觉 token。 |
| [Pathology-AE](./topics/Medical-Compression/%5BArxiv%202025%5D%20Pathology-AE/) | Arxiv 2025 | Pathology-AE 将 latent diffusion autoencoder 复用于病理压缩，用 pathology foundation model 感知指标微调，并用 K-means latent quantization 提升存储效率。 |
| [NIC](./topics/Medical-Compression/%5BTPAMI%202020%5D%20NIC/) | IEEE TPAMI 2020 | NIC 把 gigapixel WSI patch 编成低维 embedding 网格，让 CNN 在压缩表示上用弱图像级标签完成分类、回归和可视化定位。 |
| [CAR](./topics/Medical-Compression/%5BTIP%202020%5D%20CAR/) | IEEE TIP 2020 | CAR 学习内容自适应下采样 kernel 与 offsets，并用 SRNet 反向指导低分辨率图像保留可恢复细节。 |

📖 [Medical Compression 详细总结](./topics/Medical-Compression/README.md)

---

### 🔁 TTT in LLM
Test-Time Training / Test-Time Learning / long-context continual adaptation：让 LLM 在推理时用上下文或无标签测试数据更新 fast weights、LoRA 或部分权重，从而适应新信息和分布变化。

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [Absorber-LLM](./topics/TTT%20in%20LLM/%5BArxiv%202026%5D%20Absorber-LLM/) | Arxiv 2026 | Absorber LLM 用 self-supervised causal synchronization 把历史上下文吸收到参数中，让无上下文更新模型在未来生成上匹配有完整上下文的原模型。 |
| [Layer-Wise-Dynamic-TTA](./topics/TTT%20in%20LLM/%5BArxiv%202026%5D%20Layer-Wise-Dynamic-TTA/) | Arxiv 2026 | Layer-Wise Dynamic TTA 在无监督 sample-specific TTA 中用 layer-wise hypernetwork 动态调节各层 LoRA 更新强度，缓解固定学习率带来的过拟合与分布漂移。 |
| [In-Place-TTT](./topics/TTT%20in%20LLM/%5BICLR%202026%5D%20In-Place-TTT/) | ICLR 2026 Oral | In-Place TTT 把 LLM MLP block 的 final projection matrix 当作 fast weights，用 LM-aligned objective 和 chunk-wise update 做可插拔 test-time training。 |
| [TTT-E2E](./topics/TTT%20in%20LLM/%5BArxiv%202025%5D%20TTT-E2E/) | Arxiv 2025 | TTT-E2E 将 long-context LM 视作 continual learning，用测试时 next-token prediction 更新权重，并用训练时 meta-learning 学适合测试时学习的初始化。 |
| [TLM](./topics/TTT%20in%20LLM/%5BICML%202025%5D%20TLM/) | ICML 2025 | TLM 把 LLM test-time learning 表述为无标签测试数据的 input perplexity minimization，用高困惑度样本选择和 LoRA 更新做域适应。 |

📖 [TTT in LLM 详细总结](./topics/TTT%20in%20LLM/README.md)

---

### 🏭 Tech-repoorts
工业界技术报告 / frontier model system reports：聚焦大模型架构、训练基础设施、推理系统、长上下文工程和评测协议。

| 报告 | 来源 | 方法特点 |
|------|------|----------|
| [Claude Opus 4.7 System Card](./topics/Tech-repoorts/%5BSystem%20Card%202026%5D%20Claude-Opus-4.7/) | Anthropic System Card 2026 | Anthropic 2026 年 4 月 Opus 系统卡，适合看 frontier Claude 的 coding/agentic/search/long-context 能力、安全评估、RSP/ASL 部署边界和 agentic misuse / prompt injection 风险。 |
| [Mythos Preview System Card](./topics/Tech-repoorts/%5BSystem%20Card%202026%5D%20Mythos-Preview/) | Anthropic System Card 2026 | Anthropic 2026 年 4 月 Mythos Preview 系统卡，补充 Claude 产品线中非标准命名模型的能力定位、安全边界和部署策略。 |
| [Claude Sonnet 4.6 System Card](./topics/Tech-repoorts/%5BSystem%20Card%202026%5D%20Claude-Sonnet-4.6/) | Anthropic System Card 2026 | Anthropic 2026 年 2 月 Sonnet 线系统卡，覆盖 coding、agentic/search、long context、multimodal、harmlessness、alignment、agentic safety 与 ASL-3 部署判断。 |
| [DeepSeek-V4-Pro](./topics/Tech-repoorts/%5BTech%20Report%202026%5D%20DeepSeek-V4-Pro/) | Hugging Face Technical Report 2026 | DeepSeek-V4 系列报告覆盖 1M context MoE 模型的 CSA/HCA hybrid attention、mHC、Muon、>32T token 训练、post-training 与工业推理基础设施。 |
| [Attention-Residuals](./topics/Tech-repoorts/%5BTech%20Report%202026%5D%20Attention-Residuals/) | arXiv Technical Report 2026 | Kimi Team 将残差连接视为深度维度的信息聚合问题，用 Full/Block Attention Residuals 替换固定残差累加，并讨论可规模化训练/推理实现。 |

📖 [Tech-repoorts 详细总结](./topics/Tech-repoorts/README.md)

---

## 论文文件夹结构

```text
[会议 年份] 论文名/
├── README.md           # 论文概览 + Section 导航
├── sections/           # 按论文大分节生成的批读笔记
│   ├── 00-abstract.md
│   ├── 01-introduction.md
│   └── ...
├── full.md             # MinerU 解析全文
├── images/             # MinerU 图片/公式/表格
├── content_list.json   # 结构化内容
└── paper.pdf           # 原始 PDF
```

## 命名规范

- 文件夹命名：`[会议 年份] 论文名`
- Section 命名：按论文原始大分节生成，如 `00-abstract.md`、`01-introduction.md`、`02-related-work.md`、`03-methodology.md`。
- 批读格式：原文完整保留，批注使用 `> 💡 **标题**: ...`。
