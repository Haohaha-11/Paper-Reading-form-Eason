# DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

**作者**: DeepSeek-AI  
**报告类型**: Technical Report | **年份**: 2026  
**来源**: [Hugging Face model collection](https://huggingface.co/collections/deepseek-ai/deepseek-v4) | [PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf)  
**本地材料**: `paper.pdf`, `full.md`, `content_list.json`, `images/`, `paper-meta.json`  
**阅读顺序**: Tech-repoorts report 1  
**MinerU task**: `341a6da6-cdd4-43f1-a50e-db89c737ec53`

## 一句话总结

DeepSeek-V4 系列把 1M token 上下文从“能塞进去”推进到“工程上可常规服务”：用 CSA/HCA 混合注意力压缩 KV 与 FLOPs，用 mHC 稳定超深 MoE 残差流，用 Muon、>32T token 预训练、OPD 后训练和大规模工业基础设施共同支撑 Pro/Flash 两档开放模型。

## 核心贡献

1. **两档 MoE 模型族**: DeepSeek-V4-Pro 为 1.6T 总参数 / 49B activated，DeepSeek-V4-Flash 为 284B 总参数 / 13B activated，二者都原生支持 1M token 上下文。
2. **CSA + HCA 混合注意力**: CSA 先按每 `m` 个 token 压缩 KV 再做稀疏选择，HCA 以更大 `m'` 做重压缩并保持 dense attention，组合后降低长上下文 attention FLOPs 和 KV cache。
3. **mHC 残差稳定化**: Manifold-Constrained Hyper-Connections 将残差映射约束到双随机矩阵流形，使谱范数受控，目标是在扩展 residual width 的同时避免深层堆叠不稳定。
4. **Muon 与训练基础设施协同**: 大部分模块用 Muon，配套 hybrid ZeRO、BF16 Newton-Schulz、mHC recomputation、两阶段 Context Parallelism、tensor-level checkpointing，解决 trillion-scale MoE 训练效率和稳定性。
5. **后训练与服务闭环**: SFT/RL 训练 domain specialists，再用 full-vocabulary OPD 合并；FP4 QAT、preemptible rollout、sandbox、Quick Instruction、on-disk KV cache 直接面向大规模工业部署。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要、模型族定位、1M context 效率 claim |
| [01 - Introduction](sections/01-introduction.md) | 长上下文动机、架构/基础设施总览、核心评估结论 |
| [02 - Architecture](sections/02-architecture.md) | DeepSeekMoE/MTP 继承、mHC、CSA/HCA、Muon |
| [03 - General Infrastructures](sections/03-general-infrastructures.md) | EP mega-kernel、TileLang、确定性 kernel、训练/推理框架、KV cache |
| [04 - Pre-Training](sections/04-pre-training.md) | >32T tokens 数据、Pro/Flash 配置、训练日程、稳定性技巧、base model 评测 |
| [05 - Post-Training](sections/05-post-training.md) | Specialist training、reasoning modes、GRM、OPD、FP4 QAT、rollout/sandbox infra |
| [06 - Evaluation and Real-World Tasks](sections/06-evaluation-and-real-world-tasks.md) | 标准 benchmark、1M context、agent、中文写作/搜索/白领任务/code agent |
| [07 - Conclusion, Limitations, and Appendix](sections/07-conclusion-limitations-and-appendix.md) | 结论、局限、未来方向、引用脉络、附录评测细节 |

## 关键数字

| 指标 | 数值 |
|------|------|
| DeepSeek-V4-Pro 参数 | 1.6T total / 49B activated |
| DeepSeek-V4-Flash 参数 | 284B total / 13B activated |
| 上下文长度 | 1M tokens |
| 预训练 token | Flash 32T；Pro 33T；报告摘要称 both >32T |
| Pro 1M context 效率 vs DeepSeek-V3.2 | 27% single-token inference FLOPs；10% KV cache |
| Flash 1M context 效率 vs DeepSeek-V3.2 | 10% single-token inference FLOPs；7% KV cache |
| CSA 压缩 | 每 `m=4` 个 token 压缩为一个 KV entry；Pro top-k 1024，Flash top-k 512 |
| HCA 压缩 | `m'=128` 重压缩 KV entry |
| Sliding window | `n_win=128` |
| mHC | expansion factor `n_hc=4`；Sinkhorn-Knopp iterations `t_max=20` |
| Pro 架构 | 61 layers；hidden dim 7168；384 routed experts + 1 shared expert；6 routed experts activated |
| Flash 架构 | 43 layers；hidden dim 4096；256 routed experts + 1 shared expert；6 routed experts activated |
| Pre-training batch | Flash max 75.5M tokens；Pro max 94.4M tokens |
| Sequence curriculum | 4K → 16K → 64K → 1M |
| FP4 QAT | MoE expert weights + CSA indexer QK path；top-k selector 2x speedup with 99.7% KV recall |
| EP mega-kernel | 1.50-1.73x general inference speedup；up to 1.96x latency-sensitive rollout/agent serving |
| mHC overhead | 约 6.7% overlapped 1F1B pipeline stage wall-time overhead |
| Code agent internal benchmark | ~200 candidate tasks from 50+ engineers，quality filtering 后 30 tasks |
| Developer survey | N=85；52% yes、39% lean yes，认为 V4-Pro 可作为默认/主力 coding model |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 中间表示 / 机制 | 输出 |
|------|------|------------------|------|
| 1. 数据构建 | Web、math、code、多语、长文档、agentic data | 过滤模板/批量生成内容，文档 packing，sample-level attention mask，128K vocab + special tokens | >32T 高质量预训练 token |
| 2. 模型骨架 | Token 序列与 MoE routing | DeepSeekMoE + MTP；早期 Hash routing；Sqrt(Softplus) affinity；aux-loss-free balance | Pro/Flash 两档 sparse-activated Transformer |
| 3. 残差路径 | 每层 residual stream `X_l` | mHC 扩宽 residual width，并把 `B_l` 投影到双随机矩阵流形 | 更稳定的深层信号传播 |
| 4. 长上下文注意力 | Hidden states `H` 与历史 KV | CSA: `m=4` 压缩 + lightning indexer top-k；HCA: `m'=128` 重压缩；SWA 保留局部细节 | 低 KV cache / 低 FLOPs 的 1M attention |
| 5. 预训练优化 | 4K→1M curriculum、Muon/AdamW 混合优化 | Hybrid Newton-Schulz、dense-to-sparse warmup、Anticipatory Routing、SwiGLU clamping | Flash-Base / Pro-Base |
| 6. 后训练 | Domain-specific SFT/RL specialists | Non-think / Think High / Think Max；GRPO；GRM；full-vocabulary OPD with >10 teachers | DeepSeek-V4-Pro/Flash 多 reasoning effort 模式 |
| 7. 工业推理 | 用户请求、共享 prefix、tool/agent 环境 | Heterogeneous KV layout、on-disk KV、Quick Instruction、preemptible rollout、DSec sandbox | API/Chatbot/agent/coding 等服务输出 |
| 8. 评测反馈 | Standard benchmarks + real-world internal tasks | Knowledge/reasoning/agent/1M context/中文写作/搜索/白领任务/code agent | 能力、成本、局限与下一轮数据/系统改进信号 |

## 优点与局限

### 优点

- 技术路线完整：不是单点 attention trick，而是架构、优化器、kernel、训练框架、推理 KV、后训练与 sandbox 的系统级组合。
- 长上下文 claim 有明确成本指标：Pro 在 1M context 下相对 V3.2 为 27% FLOPs / 10% KV cache，Flash 进一步到 10% / 7%。
- MoE 性价比清晰：Flash 以 13B activated 覆盖较强推理和长上下文，Pro 以 49B activated 保留知识与高难 agent 能力。
- 后训练方案面向多模式产品：Non-think、High、Max、tool calling、Quick Instruction、agentic search 都服务实际 API/Chatbot 使用。
- 工业基础设施披露较细：EP overlap、TileLang、batch-invariant deterministic kernel、WAL rollout、DSec sandbox 等机制可复用。

### 局限 / 风险

- 架构复杂度高：CSA/HCA/SWA/mHC/Muon/FP4/QAT/heterogeneous KV cache 叠加，复现和排障门槛明显高于标准 Transformer。
- 稳定性仍有经验成分：Anticipatory Routing 与 SwiGLU clamping 有效但机理未完全解释，报告也承认需要更 principled 的稳定性研究。
- 评测中含大量内部设置：Codeforces、R&D coding、白领任务、搜索、中文写作等结果很有工程价值，但外部复验难度较高。
- 表格 OCR 有损：`full.md` 中若干表格和公式存在 MinerU 解析噪声，本批读优先用原始图片保留证据。
- 1M context 不是所有任务的充分条件：报告中的 MRCR/CorpusQA 和 agent 设置说明长上下文仍要配合 retrieval、tool use、reasoning budget 与 cache 策略。

### 还能做什么

- 对 CSA/HCA/mHC 做公开可复现实验：拆分 KV 压缩、top-k、SWA、mHC、Muon 的独立增益和交互增益。
- 建立外部 1M context agent benchmark：覆盖代码仓库、法律/财务长文档、多轮工具调用、共享 prefix 高并发场景。
- 简化架构版本：围绕报告提到的 future direction，寻找低复杂度但保留大部分效率收益的 CSA/HCA/mHC 子集。
- 强化安全与可靠性评估：长上下文 prompt injection、tool-call schema 逃逸、sandbox 隔离、on-disk KV 数据边界都需要公开协议。

## Q&A

- **Q: V4 的核心瓶颈到底是什么？**  
  A: 报告把瓶颈定位为 vanilla attention 的二次复杂度和 1M context 下的 KV cache / FLOPs 成本。CSA/HCA 是正面解决这个成本项，mHC/Muon/infra 是保证大模型在这个架构下能稳定训练和部署。

- **Q: CSA 和 HCA 为什么要混合，而不是只用一种？**  
  A: CSA 保留稀疏选择能力，适合在压缩后仍按 query 找关键历史块；HCA 以更强压缩降低全局成本，但不做 sparse selection。混合后能在可检索性、局部细节、全局成本之间折中。

- **Q: mHC 改变的是注意力还是残差？**  
  A: mHC 改的是 residual stream。它把残差宽度扩展成 `n_hc` 路，并约束 residual mapping `B_l` 为双随机矩阵，目标是稳定深层信号传播，而不是替代 attention。

- **Q: Muon 在这里为什么重要？**  
  A: 报告称 Muon 带来更快收敛和更好训练稳定性；工程上难点在于它需要完整梯度矩阵，因此 Section 3.4 专门设计 hybrid ZeRO、bucket assignment、BF16 Newton-Schulz 与通信优化。

- **Q: 后训练为什么用 OPD 替换 mixed RL？**  
  A: 他们先训练多个 domain specialists，再让统一 student 在自己生成的 trajectories 上对齐 teacher 分布。full-vocabulary reverse KL 比 token-level 近似更稳定，但需要 teacher scheduling、hidden-state cache 与专门 kernel 支撑。

- **Q: 1M context 对产品的直接收益是什么？**  
  A: 报告给出三类直接收益：长文档/多文档理解，agentic workflow 中跨轮保留 reasoning/tool history，共享 prefix 场景下通过 on-disk KV cache 降低重复 prefill。

## 📊 Citation Landscape

> 这是 2026 技术报告，不是常规 arXiv/会议论文；截至本次批读，公开搜索未稳定定位到 Semantic Scholar 条目，因此不编造 TLDR、citation count 或 influential citation count。下面按报告 `References` 与正文脉络整理引用景观。

### TLDR

暂无 Semantic Scholar TLDR；本地批读 TLDR：DeepSeek-V4 是一份系统级长上下文大模型报告，核心是用 CSA/HCA/mHC/Muon 与工业训练推理基础设施把开放 MoE 模型推进到可常规服务的 1M token context。

### 引用统计

| 指标 | 数值 |
|------|------|
| Semantic Scholar detail | 未检索到稳定条目 |
| 报告参考文献 | `full.md` References 约 170 行，覆盖 LLM、MoE、long context、kernel、benchmark、agent infra |
| 外部主要来源 | Hugging Face 模型集合 / PDF |
| 本批读 citation stance | 不使用未验证被引次数；只按报告引用主题分组 |

### 参考文献分组

- **LLM/Transformer 基础**: Transformer, GQA, RoPE, MTP, AdamW, LLaMA/Qwen/MiniMax/Kimi 等作为架构与开放模型背景。
- **DeepSeek 系列与 MoE**: DeepSeekMoE、DeepSeek-V2/V3/V3.2、DeepSeek-R1、auxiliary-loss-free load balancing、DeepGEMM、3FS 构成 V4 的直接工程谱系。
- **长上下文与注意力效率**: LongBench-V2、MRCR、CorpusQA、Flash-decoding、Hymba、Jenga、attention sink 等连接到 CSA/HCA 和 KV cache 管理。
- **优化与稳定性**: Muon、Muon is scalable for LLM training、ZeRO、Newton-Schulz、SwiGLU clamping、Anticipatory Routing 相关机制。
- **后训练与评测**: GRPO/RLHF/OPD、SimpleQA、Chinese-SimpleQA、MMLU-Pro、HLE、GPQA、LiveCodeBench、Terminal-Bench、SWE 系列、BrowseComp、GDPval-AA、Tool-Decathlon、MCPAtlas。
- **系统与 sandbox**: TileLang、TVM、Z3、Firecracker、QEMU、EROFS、overlaybd、Comet、MegaMoE/DeepGEMM 等支撑训练、推理与 agent execution。

### 推荐继续读

1. DeepSeek-V3 Technical Report: 读 V4 继承的 DeepSeekMoE、MTP、训练框架背景。
2. DeepSeek-V3.2 Technical Report: 对照 V4 的 FLOPs/KV cache、interleaved thinking 与 post-training 改动。
3. DeepSeekMoE: 理解 fine-grained routed experts 与 shared experts。
4. HyperConnections / mHC paper: 深入 residual width scaling 与双随机矩阵约束。
5. Muon 与 Muon is scalable for LLM training: 理解 Newton-Schulz optimizer 在 LLM 训练中的可扩展实现。
6. LongBench-V2、MRCR、CorpusQA: 评估 1M context 能力时需要区分 synthetic retrieval 与真实 corpus reasoning。
7. Comet、DeepGEMM、TileLang: 复用 V4 报告里 EP overlap 与 fused kernel 的系统思路。

## BibTeX

```bibtex
@techreport{deepseek2026v4,
  title = {DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
  author = {DeepSeek-AI},
  institution = {DeepSeek-AI},
  year = {2026},
  url = {https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf}
}
```
