[← 返回 README](../README.md)

# 7. Conclusion

## 📌 预览

总结 SBC 的贡献（可识别 + 可解释错误的稳健工作流一环），并诚实列出三条局限与对应未来方向：**(1)** 依赖可视化，参数多时吃不消 → 需自动化数值摘要；**(2)** 全局 $\chi^2$ 检验效果不好（丢了形状信息）→ 需针对特定偏离形状设计的敏感摘要；**(3)** 只能查一维摘要 → 需引入多元校准。

---

In this paper, we introduce simulation-based calibration (SBC), a readily-implemented procedure that can identify sources of poorly implemented analyses, including biased computational algorithms or incorrect model specifications. The visualizations produced by the procedure allow us to not only identify that a problem exists but also learn how the problem will affect resulting inferences. The ability to both identify and interpret these issues makes SBC an important step in a robust Bayesian workflow.

> 💡 **机制拆解 (Hao 批注)**: 收尾重申 SBC 的双重能力——不只"发现有问题"（identify），还能"看懂问题怎么影响推断"（interpret，靠形状字典）。这个 "identify + interpret" 组合是它超越前人纯检验统计量的根本。再次点名它抓两类源头：biased computational algorithms（算法偏，如 ADVI/centered HMC）+ incorrect model specifications（模型指定错，如先验写错）——始终限于"假设模型内部"。

Our reliance on interpreting the SBC diagnostic through visualization, however, can be a limitation in practice, especially when dealing with models featuring a large number of parameters. One immediate direction for future work is to develop reliable numerical summaries that quantify deviations from uniformity of each SBC histogram and provide automated diagnostics that can flag certain parameters for closer inspection.

> 💡 **局限批注 (Hao 批注)**: **局限一：靠肉眼看图，参数多时不 scalable。** 你不可能对上千个参数逐一手工审视 rank histogram。未来方向是自动化数值摘要，能给每张 histogram 打一个"偏离均匀度"的分并自动标红可疑参数。**对我们课题极其相关**：盲逆问题的 $x$ 是百万维图像，即使投影成一组 $f$（ROI/边缘/高频）也可能有很多个，必须有自动化打分 + 排序机制，否则 SBC 无法规模化到图像后验。这也是后来 rank-ECDF 检验（Säilynoja et al. 2022）等自动化统计量工作要补的坑。

Global summaries, such as a $\chi^2$ goodness-of-fit test of the SBC histogram with respect to a uniform response, are natural options, but we found they did not perform particularly well in the above examples. The reason for this is that the deviation from uniformity tends to occur in only a few systematic ways, as discussed in Section 4.2, whereas these tests consider only global behavior and hence do not exploit these known failure modes. A potential alternative is to report a number of summaries that are designed to be sensitive to the specific types of deviation from uniformity we might expect to see.

> 💡 **局限批注 (Hao 批注)**: **局限二：全局 $\chi^2$ 检验不好用**——这是个很深刻的反直觉点。$\chi^2$ 只回答"整体偏不偏离均匀"，但**丢掉了形状信息**，而形状（∩/∪/倾斜/尖峰）恰恰是 SBC 最值钱的诊断维度（第 4.2 节）。因为真实偏离只发生在少数几种系统性模式上，针对这些模式**定向**设计的摘要（比如专测"两端 vs 中间"的 ∩∪ 摘要、专测单调倾斜的 bias 摘要）会比笼统的 $\chi^2$ 灵敏得多。启示：我们做联合后验校准时，别只报一个总体 p 值，要报"过宽/过窄/有偏"分方向的诊断量，才不浪费 rank histogram 的信息。

Another future direction is deriving the expected behavior of the SBC histograms in the presence of autocorrelation and dropping the thinning requirement of SBC. This could even be done empirically, using the output of chains with known autocorrelations to calibrate the deviations in the rank histograms. These calibrated deviations could be used to define a sense of effective sample size for any algorithm capable of generating samples, not just Markov chain Monte Carlo.

> 💡 **机制拆解 (Hao 批注)**: 一个有想象力的方向——与其 thinning 掉自相关（丢样本、贵），不如**直接推导自相关下 rank histogram 的期望形状**，从而免除 thinning 要求。更进一步：用已知自相关的链去经验校准这种形状偏离，反过来为**任何**能采样的算法（不止 MCMC）定义一种"有效样本量"概念。对我们课题里那些没有天然 $N_{\text{eff}}$ 定义的摊还/扩散类采样器，这个思路提供了一条量化其样本"真实信息量"的可能路径。

Finally, the SBC histograms are only able to assess the calibration of one-dimensional posterior summaries. This is a limitation, especially in situations where the quantities of interest are naturally multivariate. An interesting extension of this methodology would be to incorporate some of the advances in multivariate calibration of probabilistic forecasts (Gneiting et al., 2008; Thorarinsdottir, Scheuerer and Heinz, 2013).

> 💡 **局限批注 (Hao 批注)**: **局限三：只能查一维摘要，天然多元的量抓不住。** rank 统计量对每个 $f$ 分别检验，忽略了参数间的**联合**校准——比如 $\varphi$ 和 $\sigma$ 各自 rank 均匀，但二者的联合后验相关结构可能仍错。未来方向是借鉴气象概率预报的**多元校准**（Gneiting 2008、Thorarinsdottir 2013）。**这对我们盲逆问题是核心痛点**：我们要的是 $(x,\varphi,\sigma)$ 的**联合**后验校准，尤其 gauge 自由度会在 $x$ 和 $\varphi$ 之间制造强耦合——逐维 SBC 可能全绿，联合却错。所以我们的 gauge-aware 校准协议不能止步于逐维 rank histogram，要向多元校准（如多元 CRPS / copula 校准）延伸，这正是本文留下、我们要接的口子。

> 💡 **Section 总结 (Hao 批注)**:
> - **贡献**: SBC = 可识别（identify）+ 可解释（interpret）错误的稳健贝叶斯工作流一环，抓算法偏 + 模型指定错，限于假设内部。
> - **三大局限 → 未来方向**: ①靠肉眼、参数多不 scalable → 自动化数值摘要；②全局 $\chi^2$ 丢形状信息 → 分方向定向摘要；③只查一维 → 多元校准。
> - **对本课题的接力点**: 局限三（多元校准）直击我们 $(x,\varphi,\sigma)$ 联合 + gauge 耦合的校准需求——逐维 SBC 是必要非充分，需向多元校准延伸。
