[← 返回 README](../README.md)

# 5. Conclusion & Appendix 结论与附录要点

## 📌 预览

结论重申"过度依赖显著 instance 伤泛化 → 遮显著挖难样本"的主线。附录补充多个实用技巧与分析：mask ratio decay、Randomly HAM（防"错误挖掘"）、TransMIL 首层注意力比末层更适合挖难样本、voting 优于 averaging 处理无效头、student 首层初始化的必要性、超参 $\alpha$ 的平衡作用。局限：难样本难精确评估、收敛稍慢。

---

## 5. Conclusion

This paper rethinks the impact of salient instances for MIL-based WSI classification algorithms. We demonstrate that attention-based MIL methods excessively prioritizing salient instances harm the generalization ability of the model. To address this issue, we have proposed several masked hard instance mining strategies that mask out salient patches and encourage the model to attend to informative regions for better discriminative learning. Through qualitative analysis, we have demonstrated that these strategies effectively alleviate the under-fitting problem of general AB-MIL to hard instances. We have also developed the MHIM-MIL framework that leverages momentum teacher and consistency loss to further enhance hard instance mining. Our experimental results demonstrate the superiority and generality of the MHIM-MIL framework over other latest methods. In future work, we plan to devise a more precise localization scheme for hard instances that can facilitate model training and convergence.

> 💡 **机制拆解**（一句话复述逻辑链）（Hao 批注）：**发现（盯显著 instance 伤泛化）→ 对策（遮显著、挖难样本）→ 稳定器（动量 teacher + 一致性损失）→ 通用性（三 backbone 均涨）+ 高效（更省更稳）**。与 [ACMIL](../../%5BECCV%202024%5D%20ACMIL/) 合读：两文从"过拟合/难样本"两个角度得出同一操作——**别让注意力只盯 Top-K**。ACMIL 无 teacher（STKIM 更轻），MHIM 有 teacher（一致性正则 + 更强通用性）。

## 附录要点（Appendix A–F）

**Randomly HAM 防"错误挖掘"（Sec. B.1）**：MHIM 最大风险是把关键信息全遮 → 变"error mining"。解法：在 top-$2\beta_h$% 高注意力里当候选，**随机遮一半**，保证难样本序列里仍含关键 instance。Tab. 9：该技巧在 CAMELYON16（阳性稀疏）有效，在 TCGA（阳性占比高）反而降（那里不需保护关键信息）。

**Mask Ratio Decay（Sec. B.1）**：$\beta_h$ 随训练余弦衰减——前期模型弱、多遮逼学难；后期模型强、少遮保信息。Tab. 8 显示 $\beta_h\%\to0\%$ 衰减显著提升（C16 TransMIL 96.07→96.49）。只对 $\beta_h$ 衰减，$\beta_r,\beta_l$ 保持。

**TransMIL 首层 vs 末层注意力（Sec. B.3）**：用**首层**注意力挖难样本优于末层（C16 96.49 vs 95.58）。原因：MSA 会改变原始特征，末层注意力偏离真实 instance 显著性；只有首层输入是原始 instance 特征，打分更准。

**Voting vs Averaging 多头融合（Sec. B.3）**：TransMIL 有"无效头"（产生几乎相同的注意力、无判别力）。averaging 会让无效头稀释难样本定位；**voting（多数表决）**能滤除无效头噪声，更优。

**Student 首层初始化（Sec. B.2）**：用预训练参数初始化 student 的首个 FC 层，降低 Siamese 结构的塌缩风险、加速 teacher 早期训练。Tab. 10 显示这对 MHIM 框架的迭代优化重要，但对普通 MIL 模型并非普适涨点技巧（有时反降）。

> 💡 **附录技巧总评**（Hao 批注）：这些附录技巧暴露了 MHIM 的"工程复杂度"——要跑好需要：选对遮蔽策略（数据依赖）、调 $\beta_h/\beta_r/\beta_l$ 三个比率、开 mask decay、开 Randomly HAM、TransMIL 还要用首层 + voting、student 首层初始化。**相比 [ACMIL](../../%5BECCV%202024%5D%20ACMIL/) 的 STKIM（只调 K/p 两个、无 teacher），MHIM 的调参负担明显更重**。这印证了 ACMIL 论文里"STKIM 更简单高效"的对比。对复现/落地：MHIM 上限更高、更通用，但需要更多调试耐心。

## F. Limitation

In this paper, we propose a Masked Hard Instance Mining MIL framework to indirectly mine hard instances in the absence of instance supervision information. Although this strategy can effectively alleviate the over-reliance problem of traditional MIL models on salient instances, it is also challenging to accurately assess the difficulty level of instances and mine the most helpful hard instances for training. Compared with traditional hard sample mining strategies based on supervision information, this sub-optimal and rough strategy affects the convergence speed and discriminability of the model. In future work, we will focus on how to accurately evaluate instance difficulty level in the absence of complete supervision and use the most beneficial instances to facilitate model training.

> 💡 **局限解读 + 对本主题/压缩研究的启示**（Hao 批注）：作者坦承"用注意力间接挖难样本"是**粗糙的代理**——无 instance 标签下无法精确评估难度，可能挖到不是最有用的难样本，拖慢收敛。
> - **对 WSI Analysis 主题**：MHIM 与 ACMIL 共同确立了"注意力会过度集中→需要主动挑战/分散"的认识。但两者都停在"启发式遮蔽"，没有原理性地定义"哪些 instance 真该保留"。
> - **对压缩/保留研究（ReadySlide）**：MHIM 的核心洞察——"训练时只留难 instance 反而更好，且注意力≠肿瘤概率"——是对"按注意力 Top-K 保留 patch"的又一警告。若做内容自适应压缩，"保留高注意力 patch"可能既漏掉难样本（判别边界信息）、又误信非肿瘤高注意力区。**retention 策略需要比单一注意力更鲁棒的重要性度量**，这与 ReadySlide "per-patch importance 不可尽信、retention 才是杠杆"的结论一致。
