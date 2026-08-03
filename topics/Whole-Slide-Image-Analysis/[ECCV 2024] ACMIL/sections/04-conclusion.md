[← 返回 README](../README.md)

# 5. Conclusion & Appendix 结论与附录要点

## 📌 预览

结论重申"注意力过度集中 ↔ 过拟合"的发现与 MBA/STKIM 两药。附录补充：MBA/STKIM 可加到 GA/MHA 两种注意力上都提升；STKIM 训练开销≈ABMIL（MHIM 则显著更高）；诚实列出三条局限（超参需调、未建 instance 相关、定位 FROC 仍远低于全监督）。

---

## 5 Conclusion

Due to the intrinsic properties of WSI, MIL methods have often led to overfitting, limiting their applications. This paper reveals that the overly concentrated attention values in the heatmap are closely related to overfitting. To address this, we propose ACMIL, which is underpinned by two novel techniques: MBA and STKIM. Our experimental results on three datasets demonstrate that ACMIL significantly surpasses SOTA methods. Moreover, this paper provides comprehensive experiments confirming the efectiveness of ACMIL in suppressing the attention value concentration and alleviating overfitting. We hope that our work can inspire future exploration into leveraging attention values for a comprehensive analysis of attention mechanisms. We also hope that our ACMIL can be applied to a broader spectrum of WSI analysis tasks.

> 💡 **机制拆解**（一句话复述全文逻辑链）（Hao 批注）：**发现（注意力熵↓↔泛化↓）→ 归因（两种集中成因：模式多样性不足 + 少数 instance 垄断）→ 对症（MBA 补多样性 + STKIM 抑垄断）→ 验证（12 项 10 优 + 热图/UMAP/Top-K 三重可视化）**。这是"诊断驱动方法设计"的范式，比"又提一个新架构"更有说服力，也是本主题最值得学习的写作骨架。

## 附录要点（Appendix 9–10）

**GA 与 MHA 都能受益（Sec. 9.1, Tab. 4）**：把 ACMIL 加到门控注意力（GA）上平均 +4.4pp；加到多头注意力（MHA）上平均 +2.5pp。说明 ACMIL 是**与注意力公式解耦的通用外挂**。

**计算开销（Sec. 9.6, Tab. 6/7）**：STKIM 训练时间/显存与 ABMIL 几乎相同（因只加了一个排序操作），而 MHIM-MIL 因需 teacher + 两次前向，显存 0.3G→1.9G、单 epoch 8s→20.8s。MBA 因引入 $\mathcal{L}_d$ 使训练时间略增（8s→11.6s），FLOPs/显存增幅很小。推理阶段三者 FLOPs 相同（STKIM/MHIM 的遮蔽都在推理移除）。

> 💡 **消融解读**（效率账 = STKIM 的隐藏卖点）（Hao 批注）：Tab. 6 是 STKIM vs MHIM 的关键对比——**同等抗集中效果下，STKIM 训练成本≈ABMIL，MHIM 则贵一个量级（2.4× 时间、6× 显存）**。这对"要不要上 teacher-student 来挑战注意力"给了明确答案：如果只是想分散注意力抗过拟合，STKIM 的性价比碾压。

## 10 Limitations

Although ACMIL enhances the generalization ability and interpretability of MIL methods in WSI analysis, certain limitations necessitate further exploration. Firstly, the selection of hyperparameters M and K significantly impacts performance, and the optimal choice depends on the dataset, requiring practitioners to determine the best value through trial and error. In the future, how to simplify the framework should be considered. Secondly, our paper does not account for the correlation between instances, which is crucial for understanding the complex tumor structure. This aspect will be a focus of future investigations. Thirdly, ACMIL significantly reduces the need for instance annotations compared to instance-supervised approaches and achieves comparable WSI classification performance (AUC: ACMIL 0.974 vs. Full supervised 0.992), but it performs poorly in tumor localization tasks. Tab. 2 shows ACMIL achieves an FROC score of 0.4322 on the Camelyon16 tumor localization task, lower than the top-performing supervised approach with a score of 0.8074.

> 💡 **局限解读 + 对本主题的启示**（Hao 批注）：作者难得地诚实，列了三条硬伤：
> 1. **超参敏感**（M、K 需按数据集试错）——与 [SiMLP](../../ckmil-re-attn-mil/)"简单方法更稳"的批评呼应。
> 2. **未建 instance 相关性**——MBA/STKIM 都是"独立处理 instance"，没用空间/结构关系，这正是 [Spatial-Blindness](../../ckmil-re-attn-mil/) 那条线关心的。
> 3. **定位仍差**（FROC 0.43 vs 全监督 0.81）——分类友好 ≠ 定位友好。
>
> **对压缩/保留研究**：ACMIL 的核心贡献（Top-10 占 85% 注意力、真实阳性 instance 有上千个）是一个强证据——**基于单一注意力的 patch importance 会系统性漏掉大量判别 patch**。若做内容自适应保留，应当用 MBA 式的多视角重要性、或 STKIM 式的"别只信 Top-K"，否则压缩会把判别信息一起丢掉。这与 ReadySlide 项目"retention 是杠杆、per-patch importance 不可尽信"的主结论一致。
