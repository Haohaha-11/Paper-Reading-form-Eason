[← 返回 README](../README.md)

# Abstract

## 📌 预览

这个 Section 用一段话给出全文主张：VLA(视觉-语言-动作)研究正被"架构、代码库、评测协议互不兼容"割裂，导致无法公平对比与复现。StarVLA 提出一个**开源代码库**，从三方面解决:(1) 模块化的 backbone–action-head 架构,backbone(VLM 或 world model)与 action head 可各自独立替换;(2) 可复用、与范式无关的训练策略(跨本体学习 + 多模态联合训练);(3) 统一评测接口整合 5 大 benchmark,并支持仿真与真机部署。核心卖点是"极简数据工程即可 match/surpass 已有方法",且同时支持 VLM 与 world-model 两类 backbone。

---

Building generalist embodied agents requires integrating perception, language understanding, and action, which are core capabilities addressed by Vision-Language-Action (VLA) approaches based on multimodal foundation models, including recent advances in vision-language models and world models. Despite rapid progress, VLA methods remain fragmented across incompatible architectures, codebases, and evaluation protocols, hindering principled comparison and reproducibility. We present StarVLA, an open-source codebase for VLA research. StarVLA addresses these challenges in three aspects. First, it provides a modular backbone–action-head architecture that supports both VLM backbones (e.g., Qwen-VL) and world-model backbones (e.g., Cosmos) alongside four representative action-decoding paradigms, all under a shared abstraction in which backbone and action head can each be swapped independently. Second, it provides reusable training strategies, including crossembodiment learning and multimodal co-training, that apply consistently across supported paradigms. Third, it integrates major benchmarks, including LIBERO, SimplerEnv, RoboTwin 2.0, RoboCasa-GR1, and BEHAVIOR-1K, through a unified evaluation interface that supports both simulation and real-robot deployment. StarVLA also ships simple, fully reproducible single-benchmark training recipes that, despite minimal data engineering, already match or surpass prior methods on multiple benchmarks with both VLM and world-model backbones. To our best knowledge, StarVLA is one of the most comprehensive open-source VLA frameworks available, and we expect it to lower the barrier for reproducing existing methods and prototyping new ones. StarVLA is being actively maintained and expanded; we will update this report as the project evolves. The code and documentation are available at github.com/starVLA/starVLA.

> 💡 **问题动机** (claude 批注): 摘要把痛点精确定位在三个层次的"碎片化",这正是后文三大贡献一一对应的:
> - **架构层碎片化** → 用 backbone–action-head 解耦解决(第 2 节)。不同论文的 action decoding(自回归 tokenize / 并行回归 / diffusion / flow matching)本来各写各的,StarVLA 把它们统一成"共享 backbone + 可插拔 head"。
> - **系统层碎片化** → 用统一训练/推理 I/O 接口 + 可复用训练配方解决(第 3 节)。
> - **评测层碎片化** → 用 server-client 统一评测接口整合 5 个 benchmark 解决(第 4 节)。
>
> 关键差异点在于:StarVLA 不只是又一个 VLA 方法,而是一个**平台/代码库**。它的贡献不是某个 SOTA 数字,而是"让 VLM-based 与 world-model-based 两条技术路线能在完全相同的数据管线、训练循环、评测协议下直接对比"。这是本文反复强调的实验可比性主张。

> 💡 **机制拆解** (claude 批注): 摘要里埋了一个后文会展开的核心洞察——"backbone 与 action head 可各自独立替换"。这意味着一次实验只改动一个变量(要么换 backbone,要么换 head),其余全部固定,从而能干净地隔离单个设计选择的效果。这就是第 1 节提出的 **generalized VLA perspective(广义 VLA 视角)** 的实证基础:VLM-based 和 world-model-based 方法或许不是本质不同的范式,而是同一结构框架下辅助学习信号($\mathcal{L}_{\text{aux}}$)不同的变体。

> 💡 **Q&A 批注记录** (claude 批注):
> - Q: 摘要说"minimal data engineering 就能 match/surpass prior methods",这个 claim 靠什么支撑?
> - A: 靠第 5 节的单 benchmark SFT 结果。例如 LIBERO 上 StarVLA-OFT 用 30K steps(约 9.5 epochs)达到 96.6%,而 OpenVLA-OFT 需 175K steps(223 epochs)才到 97.1%——用 6× 更少的 step、23× 更少的 epoch 达到相近水平(见 Table 2)。这不是刷 SOTA,而是证明"干净的极简 baseline 已经很强",从而为社区提供可复现的锚点。

Date: April 2026

Project Page: https://starvla.github.io

> 💡 **Section 小结** (claude 批注):
> - **一句话**: StarVLA 是一个把 VLM-based 与 world-model-based VLA 统一到"可插拔 backbone + 可插拔 action head"抽象下的开源平台。
> - **关键数字**: 4 种 action-decoding 范式、2 类 backbone(VLM / world model)、5 个整合 benchmark、极简配方即可 match/surpass 已有方法。
> - **核心洞察**: 平台不仅是工具,还是"强且易复现的 baseline 提供者"。
> - **可追问点**: 4 种范式具体是什么?统一 I/O 接口如何做到 train/test 一致?这些在第 2、3 节展开。
