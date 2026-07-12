![Paper Reading Banner](./banner.svg)

# Paper Reading 📚

Haojiang 的文献阅读仓库，按课题组织。每篇论文都有「批读格式」阅读笔记：原文完整保留 + 内嵌批注。

---

## 📅 每日待读

| 日期 | 论文 | 课题 | 状态 | 反馈 |
|------|------|------|------|------|
| 2025-06-23 | [SAMPath](./topics/Medical-Compression/%5BArxiv%202025%5D%20SAMPath/) | Medical Compression | ⏳ 待读 | - |
| 2025-06-23 | [VideoStreamThinking](./topics/video-VLM/%5BArxiv%202026%5D%20VideoStreamThinking/) | Video VLM | ⏳ 待读 | - |
| 2025-06-23 | [ConsistencyVL](./topics/VisMem-for-Med-Image/%5BArxiv%202026%5D%20ConsistencyVL/) | VisMem for Med Image | ⏳ 待读 | - |
| 2025-06-23 | [MedMO](./topics/VisMem-for-Med-Image/%5BArxiv%202026%5D%20MedMO/) | VisMem for Med Image | ⏳ 待读 | - |
| 2025-06-23 | [TissueWSI](./topics/Whole-Slide-Image-Analysis/%5BArxiv%202026%5D%20TissueWSI/) | Whole Slide Image Analysis | ⏳ 待读 | - |
| 2025-06-23 | [FlashMemory-DSV4](./topics/TTT%20in%20LLM/%5BArxiv%202026%5D%20FlashMemory-DSV4/) | TTT in LLM | ⏳ 待读 | - |
| 2025-06-23 | [ULRFM](./topics/Medical-Compression/%5BArxiv%202026%5D%20ULRFM/) | Medical Compression | ⏳ 待读 | - |
| 2025-06-23 | [PathoLIC](./topics/Medical-Compression/%5BArxiv%202026%5D%20PathoLIC/) | Medical Compression | ⏳ 待读 | - |
| 2025-06-24 | [Q-Zoom](./topics/VLM-Bottleneck-Analysis-and-Method-Design/encoding/%5BArxiv%202025%5D%20Q-Zoom/) +14 more | VLM Bottleneck Analysis | ⏳ 待读 | - |
| 2025-06-28 | [ARPO](./topics/GRPO/%5BArxiv%202025%5D%20ARPO/) | GRPO | ⏳ 待读 | - |

---

## 课题列表

### 🎯 Blind Inverse Problems with Generative Priors
生成先验下的参数化盲逆问题：算子未知时联合恢复图像 $x$、低维算子参数 $\varphi$ 与噪声 $\sigma$ 的后验，主线是 gauge-aware 联合后验采样与 SBC/coverage/CRPS 校准——追问何时真的得到联合贝叶斯后验，何时只是一组看似合理的样本。

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [BlindDPS](./topics/Blind-Inverse-Problems-Generative-Priors/blind-joint-sampling/%5BCVPR%202023%5D%20BlindDPS/) | CVPR 2023 | 图像与核各建扩散模型、同一数据一致项并行 DPS 引导；偏差入口为 Jensen 点估计 + 独立先验，无 gauge/σ 联合/校准。核心基线。 |
| [GibbsDDRM](./topics/Blind-Inverse-Problems-Generative-Priors/blind-joint-sampling/%5BICML%202023%5D%20GibbsDDRM/) | ICML 2023 | 部分塌缩 Gibbs：图像块 DDRM 谱域 + 算子块 Langevin，Prop 3.1 保平稳分布；与本课题联合核结构最接近。 |
| [PRISM](./topics/Blind-Inverse-Problems-Generative-Priors/blind-joint-sampling/%5BArxiv%202025%5D%20PRISM/) | Arxiv 2025 | measurement-conditioned diffusion prior + split-Gibbs，报告像素级 SD/NLL/覆盖；不确定性最直接竞品，但低噪声过自信、缺 SBC/CRPS。 |
| [Fast-Diffusion-EM](./topics/Blind-Inverse-Problems-Generative-Priors/blind-joint-sampling/%5BWACV%202024%5D%20Fast-Diffusion-EM/) | WACV 2024 | 扩散 E 步 + 快速 EM，核为共享单核 MAP 点估计；"用了不确定性却只输出点估计"的对照锚点。 |
| [Survey-Diffusion-Inverse](./topics/Blind-Inverse-Problems-Generative-Priors/survey/%5BArxiv%202024%5D%20Survey-Diffusion-Inverse/) | Arxiv 2024 | Daras 等综述：条件采样器家族图谱，渐近精确家族是校准最有意义的落点。 |
| [Diffusion-Models-for-Inverse-Problems](./topics/Blind-Inverse-Problems-Generative-Priors/survey/%5BArxiv%202025%5D%20Diffusion-Models-for-Inverse-Problems/) | Arxiv 2025 | Chung 等综述：把每个方法归结为"对似然 score 做了什么近似"，梳理盲逆三法谱系。 |
| [Beyond-Accuracy-Posterior-Fidelity](./topics/Blind-Inverse-Problems-Generative-Priors/posterior-calibration/%5BArxiv%202026%5D%20Beyond-Accuracy-Posterior-Fidelity/) | Arxiv 2026 | Qiu 等：精度≠分布一致，提出无需真后验的 score-KSD 诊断。 |
| [Simulation-Based-Calibration](./topics/Blind-Inverse-Problems-Generative-Priors/posterior-calibration/%5BArxiv%202018%5D%20Simulation-Based-Calibration/) | Arxiv 2018 | Talts 等 SBC：rank histogram 检验模型内校准，"算法错误 vs 模型错误"的第一道闸门。 |
| [Feynman-Kac-Bias-Stability](./topics/Blind-Inverse-Problems-Generative-Priors/posterior-calibration/%5BArxiv%202026%5D%20Feynman-Kac-Bias-Stability/) | Arxiv 2026 | Delgadino 等：Feynman–Kac 证明即便 prior score 精确 DPS 仍系统偏差、漏模态。 |
| [Principled-Posterior-Matching](./topics/Blind-Inverse-Problems-Generative-Priors/posterior-calibration/%5BArxiv%202026%5D%20Principled-Posterior-Matching/) | Arxiv 2026 | Bai 等：近似目标致 mode collapse，提出目标级无偏 PPM；"样本离散≠已校准"。 |
| [Exact-Posterior-Score](./topics/Blind-Inverse-Problems-Generative-Priors/posterior-calibration/%5BArxiv%202026%5D%20Exact-Posterior-Score/) | Arxiv 2026 | Mammadov 等：低维+可解析先验+已知线性算子下后验解析可得，作 SBC/coverage/CRPS 的 gold-standard 参考后验。 |

📖 [Blind Inverse Problems with Generative Priors 详细总结](./topics/Blind-Inverse-Problems-Generative-Priors/README.md)

---

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
| [DMLR](./topics/VisMem-for-Med-Image/%5BArxiv%202025%5D%20DMLR/) | Arxiv 2025 | DMLR 提出 training-free 测试时动态潜空间推理：置信度引导的策略梯度优化 + 动态视觉注入，实现推理-感知自适应交错。 |
| [MedVLThinker](./topics/VisMem-for-Med-Image/%5BArxiv%202025%5D%20MedVLThinker/) | Arxiv 2025 | MedVLThinker 是首个全开源医学多模态推理配方，揭示 RLVR > SFT 且 text-only > image-text 的反直觉发现，32B 匹配 GPT-4o。 |
| [ConsistencyVL](./topics/VisMem-for-Med-Image/%5BArxiv%202026%5D%20ConsistencyVL/) | Arxiv 2026 | ConsistencyVL 将空间注意力与可靠性解耦，用跨视图一致性信号识别 VLM 中哪些视觉区域可信。 |
| [MedMO](./topics/VisMem-for-Med-Image/%5BArxiv%202026%5D%20MedMO/) | Arxiv 2026 | MedMO 面向医学影像的 MLLM ground 与理解，增强模型在医学图像上的定位和诊断解释能力。 |

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
| [SAMPath](./topics/Medical-Compression/%5BArxiv%202025%5D%20SAMPath/) | Arxiv 2025 | SAMPath 用自然语言驱动病理图像分割，将 SAM 扩展到病理领域实现文本提示的任意分割。 |
| [ULRFM](./topics/Medical-Compression/%5BArxiv%202026%5D%20ULRFM/) | MedIA 2026 | ULRFM 用 Transformer 上下文建模对病理 JPEG 做无损再压缩，9M+ tile 训练，最高减容 34.13%。 |
| [PathoLIC](./topics/Medical-Compression/%5BArxiv%202026%5D%20PathoLIC/) | MedIA 2026 | PathoLIC 内容感知可变码率病理压缩，诊断区高保真、背景区高压缩，超 Aperio SVS 8×。 |

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
| [TTT4LC](./topics/TTT%20in%20LLM/%5BArxiv%202025%5D%20TTT4LC/) | arXiv 2025 | TTT4LC 揭示长上下文 static attention 的 score dilution 导致 thinking tokens 失效，提出 Query-Only TTT 用梯度更新打破注意力静态限制。 |
| [FlashMemory-DSV4](./topics/TTT%20in%20LLM/%5BArxiv%202026%5D%20FlashMemory-DSV4/) | arXiv 2026 | FlashMemory-DSV4 用 lookahead sparse attention 实现超长上下文的闪电索引，提升长序列处理效率。 |

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

### 🌊 Continuous Latent Language Modeling
连续潜空间语言建模：用 diffusion/flow 在连续嵌入空间中替代 token-level autoregressive，探索全局语义先验建模和文本-连续模态统一。

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [Cola DLM](./topics/Continuous-Latent-Language-Modeling/%5BArxiv%202026%5D%20Cola-DLM/) | Arxiv 2026 | Cola DLM 提出层级化连续潜扩散语言模型：Text VAE → block-causal DiT prior → conditional decoding，将 diffusion 定义为 latent prior transport。 |
| [ELF](./topics/Continuous-Latent-Language-Modeling/%5BArxiv%202026%5D%20ELF/) | Arxiv 2026 | ELF 用 Flow Matching 在连续嵌入空间建模语言，全程留在连续空间直到最后一步映射到 token，天然支持 CFG。 |

📖 [Continuous Latent Language Modeling 详细总结](./topics/Continuous-Latent-Language-Modeling/README.md)

---

### 🎬 Video VLM
视频大语言模型：让 VLM 在视频流中同时观看和思考，实现流式视觉感知与推理的实时交错。

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [VideoStreamThinking](./topics/video-VLM/%5BArxiv%202026%5D%20VideoStreamThinking/) | Arxiv 2026 | Video Streaming Thinking 让 VideoLLM 在视频流中边看边想，实现视觉感知与推理的同步交错。 |

📖 [Video VLM 详细总结](./topics/video-VLM/README.md)

---

### 🔬 Whole Slide Image Analysis
全切片病理图像分析：让模型像病理学家一样在 gigapixel 组织图像中进行空间感知推理，从定位关键区域到构建诊断证据链。

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [TissueWSI](./topics/Whole-Slide-Image-Analysis/%5BArxiv%202026%5D%20TissueWSI/) | Arxiv 2026 | Tissue-Aware WSI Reasoning 让模型像病理学家一样在组织空间上下文中推理，实现端到端的空间定位与诊断。 |

📖 [Whole Slide Image Analysis 详细总结](./topics/Whole-Slide-Image-Analysis/README.md)

---

### 🔍 VLM Bottleneck Analysis and Method Design
VLM 瓶颈分析与方法设计：系统诊断视觉编码、推理过程、奖励设计、多图理解等环节的 failure mode，并针对性设计新方法。

| 子类别 | 论文数 | 代表论文 |
|--------|--------|----------|
| encoding | 4 | Q-Zoom, CARES, iGVLM, Perceptual-Bandwidth-Bottleneck |
| grounding | 1 | Vision-aligned Latent Reasoning |
| hierarchical | 1 | Hierarchical Visual Cues Injection |
| invoke | 2 | Iterative Evidence Refinement, Thinking with Visual Grounding |
| reward | 3 | RegionReasoner, VisualPRM, Perception-centric PRM |
| long-reasoning | 3 | Imagine Before Predict, VisMem, DMLR |
| bottleneck | 1 | More Images More Problems |
| multi-image | 1 | Dual Mechanisms Spatial Binding |
| medical | 1 | MedSynapse-V |

📖 [VLM Bottleneck Analysis 详细总结](./topics/VLM-Bottleneck-Analysis-and-Method-Design/README.md)

---

### 🎯 GRPO
GRPO (Group Relative Policy Optimization) 及其变体在 LLM/VLM 强化学习训练中的应用与改进。

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [ARPO](./topics/GRPO/%5BArxiv%202025%5D%20ARPO/) | Arxiv 2025 | Agentic Reinforced Policy Optimization：将 GRPO 与大模型 Agent 范式结合，用于强化学习策略优化。 |

📖 [GRPO 详细总结](./topics/GRPO/README.md)

---

### 📝 Rebuttal
论文审稿 rebuttal 与学术论证：多 Agent 协同攻击 LVLM 安全边界。

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [Persuade-in-scene](./topics/Rebuttal/%5BArxiv%202025%5D%20Persuade-in-scene/) | CVPR 2026 | 多 Agent 排版越狱攻击 LVLM，视觉排版 + 文本协同绕过安全检测。 |
| [CKMIL](./topics/Rebuttal/%5BArxiv%202025%5D%20CKMIL/) | Anonymous | 级联关键实例注意力 MIL，解决 WSI 实例间相关性与 O(n²) 复杂度的两难。 |

📖 [Rebuttal 详细总结](./topics/Rebuttal/README.md)

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
