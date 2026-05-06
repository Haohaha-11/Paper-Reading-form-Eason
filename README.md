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
