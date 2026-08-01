[← 返回 README](../README.md)

# 04 Discussion

## 原文

### 5 Conclusion

The lack of well-adapted offline features and disjointly optimized models has become a performance bottleneck in CPath. While slide-level supervised E2E learning presents a fundamental solution, it remains underexplored due to efficiency and performance challenges. Our work revisits slide-level supervised E2E learning in CPath from the MIL perspective. We demonstrate the impact of sparse-attention MIL on E2E optimization. After addressing optimization challenges through the proposed ABMILX, we show that E2E-trained ResNet achieves comparable performance to foundation models with lower computational costs. We believe E2E learning has the potential to benefit upstream pre-training and achieve further breakthroughs with increased computational resources. Revisiting the role of MIL in E2E learning may be key to realizing its potential.

### F Limitation & Broader Impacts

This work pioneered the exploration of end-to-end (E2E) optimization challenges in computational pathology and effectively mitigated them. It demonstrated the potential and advantages of E2E learning in this domain. However, our current full-training approach makes direct fine-tuning of large foundation models challenging under limited computational resources. Investigating the effectiveness of our proposed method for fine-tuning foundation models is a direction for future work. Furthermore, as this work focuses on computational pathology, it is directly relevant to tasks such as multi-cancer diagnosis and prognosis. This work has the potential to inspire and facilitate the deployment of more accurate and efficient clinical diagnosis and prognosis algorithms.

### Appendix C.1: More about E2E Methods

**Table 6: Comparison between E2E methods and two-stage methods.**

| Encoder | Method | TTime | Grad. | Sub. | Surv. |
|---------|--------|-------|-------|------|-------|
| **Latest Two-stage** | | | | | |
| R50 | WIKG | 3h | 62.72 | 88.37 | 60.65 |
| R50 | RRT | 3h | 60.42 | 89.35 | 63.03 |
| FM | WIKG | 24h | 74.97 | 94.76 | 66.97 |
| FM | RRT | 24h | 74.00 | 94.84 | 67.30 |
| **E2E Training** | | | | | |
| R18 | C2C | 84h | 62.91 | 91.13 | - |
| R50 | FT | 45h | 66.06 | 86.48 | - |
| R18 | ABMILX | 9h | 78.34 | 93.97 | 67.78 |
| R50 | ABMILX | 22h | 78.83 | 95.17 | 67.20 |

### Appendix C.2: More about MILs in E2E Learning

**Table 7: More MIL aggregators in E2E training.**

| Aggregator | Aggr. Type | Grad. | Sub. | Surv. |
|------------|------------|-------|------|-------|
| Best in FMs | - | 74.97 | 94.84 | 67.30 |
| ABMIL | S.A. | 75.46 | 89.23 | 62.70 |
| RRTMIL | S.A. | 17.99 | 61.82 | 53.42 |
| QAMIL | Trans. | 75.12 | 90.65 | 64.29 |
| TransMIL | Trans. | 75.08 | 91.44 | 63.42 |
| DSMIL | Trans. | 76.28 | 91.09 | 64.32 |
| ViTMIL | Trans. | 76.98 | 92.61 | 63.67 |
| ABMILX | S.A. | 78.34 | 93.97 | 67.78 |

### Appendix C.3: ABMILX in Two-Stage Framework

*(Full Table 8 in appendix, showing ABMILX can also be used as a general MIL aggregator in two-stage settings, performing well particularly under FM features)*

### Appendix C.4: Ablation

**Table 9: Ablation studies on various components.**

(a) FFN: w/o FFN Grad.77.04 vs w/FFN Grad.78.34 — FFN helps PANDA, hurts smaller datasets

(b) Projection Dim: optimal = 256 for Sub-typing, 512 for Grading

(c) Head Number: optimal = 8 for Sub/Surv, 4 for Grading

(d) Multi-scale Ratio: optimal = 4 for Grad/Sub, 10 for Survival

(e) Sampling Number: optimal = 128 (Grad), 512 (Sub), 768 (Surv)

(f) Sampling Strategy: vanilla random > regional random

---

> 💡 **Hao 批注：论文的贡献层级与局限**
>
> **贡献层级**:
> 1. **洞察级贡献**（最高）: 首次将 E2E 训练失败归因于 MIL 的稀疏注意力，而不是数据采样或计算量不足。这是一个"换个视角看问题"的贡献。
> 2. **方法级贡献**: ABMILX 设计（MHLA + A+）有理论支撑（Appendix A 优化风险分析），且消融实验完整。
> 3. **工程级贡献**: MRIS 多尺度随机采样简单但有效，大幅降低训练成本。
>
> **局限**:
> 1. **未扩展到 ViT/大模型**: 论文坦承"full-training approach makes direct fine-tuning of large FMs challenging"。ABMILX 是否对 ViT 架构（如 UNI、GigaPath 的 encoder）同样有效未知。
> 2. **只测试了 ResNet**: R18 和 R50，缺乏 ViT backbone 的实验。这可能是因为 ViT 的 E2E 训练对显存要求更高。
> 3. **采样数量 vs 性能的 trade-off**: 存活分析需要大量 patches（768），但 PANDA 用 128 个 patch 最佳——采样策略与任务的交互尚不明确。
> 4. **数据集规模**: 只用了 TCGA (~1K slides) 和 PANDA (~10K)，没有在更大的真实世界数据集上验证。
> 5. **跨中心泛化只做了 CPTAC**: 只有一组外部验证，不够充分。

---

> 💡 **Hao 批注：与同期论文 EXAONE Path 2.0 的对比**
>
> | 维度 | Revisiting-E2E | EXAONE Path 2.0 |
> |------|---------------|-----------------|
> | **问题视角** | MIL 设计导致 E2E 优化坍塌 | SSL 预训练不捕获 biomarker 特征 |
> | **技术路线** | 改良 MIL 聚合器 | 改良训练框架（curriculum + 多任务） |
> | **编码器** | ResNet (26M/12M) | HIPT 三层 ViT |
> | **训练数据** | 下游任务数据 (~1K-10K WSIs) | 37K WSIs + 多任务标签 |
> | **预训练** | ImageNet-1K (自然图像) | 无额外预训练（但用了 DINO curriculum） |
> | **关键创新** | MHLA + A+ 注意力机制 | 三层架构 E2E + curriculum learning |
> | **推理速度** | 1.7s/slide | 未报告 |
> | **实验结果** | PANDA Acc 78.83 (超 FM) | 10-task avg AUROC 0.784 (SOTA) |
>
> 两篇论文互补：
> - Revisiting-E2E 回答"E2E 中 MIL 应该怎么设计"
> - EXAONE Path 2.0 回答"E2E 训练中如何让编码器学到临床相关特征"
>
> 两者都论证了 E2E 的必要性和有效性，但切入角度不同：一个是微观的 MIL 机制优化，一个是宏观的训练框架创新。

---

> 💡 **Hao 批注：对 ReadySlide 项目的启示**
>
> 1. **MIL 设计比编码器 size 更重要**: ResNet-18 + ABMILX (PANDA 78.34) 超过 UNI(ViT-L) + ABMIL (74.69)。这暗示 ReadySlide 的 allocator（本质上是编码器选择哪些 patch 给多少 bit 的策略）的设计可能比使用哪个 FM 更重要。
>
> 2. **E2E 训练 vs 冻结编码器**: Table 4 的 freeze encoder ablation 显示冻结编码器导致性能急剧下降，说明 E2E 中编码器适配是关键——但如果 ReadySlide 要保持编码器冻结（为了跨 FM 的兼容性），那么 ABMILX 的"E2E-friendly" MIL 设计优势可能无法完全发挥。
>
> 3. **MIL 的抽象杠杆**: 本文证明在 E2E 设置中 MIL 的选择比采样策略重要一个数量级（4.7pp vs 0.5pp）。ReadySlide 中的 allocator policy 承担了类似 MIL 的角色——它决定了哪些 patch 获得更多 bit → 哪些 patch 的梯度更强 → 类似反向传播中的"软实例选择"。allocator_learn.py 的设计应当考虑这种优化循环机制。
>
> 4. **ABMILX 可作为 ReadySlide 的聚合器**: 如果 ReadySlide 需要 end-to-end 训练（压缩 + 下游任务联合优化），ABMILX 的设计（多头部 + 全局相关性传播）可能比标准 ABMIL 更适合，因为它天然能够容忍压缩带来的特征噪声。

---

> 💡 **Hao 批注：论文影响力的评估**
>
> **优点**:
> - 问题定义清晰且有一般性（不局限于病理，任何 MIL-based E2E 都可能受稀疏注意力困扰）
> - 理论-实验闭环完整：定义 R → 推导 MHLA/A+ 降低 R 的条件 → MAX-N 实验验证
> - 计算效率和性能同时改善，而非 trade-off
>
> **潜在问题**:
> - ABMILX 虽然有效，但其设计与多头自注意力的边界有些模糊——如果不仔细读附录的理论分析，容易认为只是"多头 ABMIL + self-attention"
> - 实验数据集都是公开的经典 benchmark，缺乏真实世界多中心验证
> - "surpass SOTA FMs" 的宣称需要谨慎解读——这只在特定任务（PANDA, BRCA）上成立，在 NSCLC/CAMELYON 上 FM 仍占优
