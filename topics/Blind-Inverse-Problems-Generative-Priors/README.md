# Blind Inverse Problems with Generative Priors 🎯

> 生成先验（扩散 / 流模型）下的**参数化盲逆问题**：在退化算子未知或仅部分已知时，联合恢复图像 $x$、低维算子参数 $\varphi$（运动模糊长度/角度、PSF 宽度、CT 中心偏移等）与噪声 $\sigma$，并追问**采样分布是否正确、区间是否校准、误差来自哪里**。

本 topic 服务于课题 **《生成先验下的盲逆问题》**（工作标题 *Gauge-Calibrated Joint Posterior Sampling for Parametric Blind Inverse Problems*）。主线不是"再刷一次去模糊 PSNR"，而是研究：在算子未知、图像先验由生成模型给出时，我们**何时真的得到了联合贝叶斯后验，何时只是得到一组看起来合理的样本**。围绕这个可证伪问题，文献按三条线组织。

---

## 📚 阅读顺序（三遍法）

1. **建立全局地图** → `survey/` 两篇综述，把条件采样器家族、prior score 与 posterior score 的差距搞清楚。
2. **对齐直接竞品** → `blind-joint-sampling/` 四篇盲联合方法，这是我们要复现和正面比较的基线。
3. **进入后验正确性与校准** → `posterior-calibration/` 六篇，理解偏差从哪进入、如何构造参考后验、如何用 SBC/coverage/CRPS 检验校准。

---

## 一、全局地图 · survey

| 论文 | 会议 | 与本课题的关系 |
|------|------|----------------|
| [Survey-Diffusion-Inverse](./survey/%5BArxiv%202024%5D%20Survey-Diffusion-Inverse/) | Arxiv 2024 | Daras 等综述：条件采样器家族图谱；点出渐近精确家族是校准最有意义的落点。 |
| [Diffusion-Models-for-Inverse-Problems](./survey/%5BArxiv%202025%5D%20Diffusion-Models-for-Inverse-Problems/) | Arxiv 2025 | Chung 等综述：把每个方法翻译成"对似然 score 做了什么近似"；梳理盲逆三法谱系（BlindDPS→GibbsDDRM→FastDiffEM，$\varphi$ 不确定性递减、无一做 gauge/校准）。 |

## 二、图像—算子联合采样 · blind-joint-sampling（核心基线）

| 论文 | 会议 | 方法特点 / 定位 |
|------|------|----------------|
| [BlindDPS](./blind-joint-sampling/%5BCVPR%202023%5D%20BlindDPS/) | CVPR 2023 | 图像与核各建扩散模型，用同一数据一致项并行 DPS 引导。**偏差入口**：DPS 的 Jensen 点估计 + 独立先验假设；核用稀疏正则但**无 gauge 商空间处理**、无 $\sigma$ 联合推断、无 SBC/coverage/CRPS。 |
| [GibbsDDRM](./blind-joint-sampling/%5BICML%202023%5D%20GibbsDDRM/) | ICML 2023 | 部分塌缩 Gibbs（PCGS）：图像块=DDRM 谱域，算子块=Langevin，Prop 3.1 保平稳分布。**与本课题联合核结构最接近**，必比基线。 |
| [PRISM](./blind-joint-sampling/%5BArxiv%202025%5D%20PRISM/) | Arxiv 2025 | measurement-conditioned diffusion prior + split-Gibbs 后验采样；报告像素级 SD / NLL / 3-SD 覆盖。**不确定性最直接竞品**；已暴露弱点：低噪声过自信、缺 SBC/CRPS/reliability 曲线、$\sigma$ 当已知。 |
| [Fast-Diffusion-EM](./blind-joint-sampling/%5BWACV%202024%5D%20Fast-Diffusion-EM/) | WACV 2024 | 扩散 E 步 + 快速 EM，核为 $\arg\max$ MAP 点估计、所有粒子共享单核。**"用了不确定性却只输出点估计"的对照锚点**。 |
| [Instrument-Noise-Estimation](./blind-joint-sampling/%5BArxiv%202026%5D%20Instrument-Noise-Estimation/) | Arxiv 2026 | Giovannelli 的 **Hyper-G-DPS**：Gibbs 大循环联合估图像 + PSF 宽度 $\iota$ + 噪声偏置 $m_e$/方差 $v_e$，低维参数走共轭直采、图像块走 G-DPS（一次迭代一次网络）。**比 GibbsDDRM 多估噪声参数**、给 ±2 PSD 覆盖式 UQ；但收敛"保证"含 forward≈backward 近似（未量化）、校准仅到单场景个案覆盖，是本课题该用 SBC/参考后验去测的对象。 |

## 三、后验正确性 · 可辨识性 · 校准 · posterior-calibration

| 论文 | 会议 | 与本课题的关系 |
|------|------|----------------|
| [Beyond-Accuracy-Posterior-Fidelity](./posterior-calibration/%5BArxiv%202026%5D%20Beyond-Accuracy-Posterior-Fidelity/) | Arxiv 2026 | Qiu 等：精度≠分布一致；提出无需真后验的 **score-KSD** 诊断。迁移到盲设置有"评价 score field 依赖待估 $\varphi/\sigma$"的鸡生蛋问题 → 建议 $x$ 维用 score-KSD、$\varphi/\sigma$ 维用 SBC/coverage/CRPS 互补。 |
| [Simulation-Based-Calibration](./posterior-calibration/%5BArxiv%202018%5D%20Simulation-Based-Calibration/) | Arxiv 2018 | Talts 等 **SBC**：rank histogram 检验模型内校准。是"算法错误 vs 模型错误"两层结论的第一道闸门；高维图像需选可解释函数量 $f$（ROI 均值/边缘位置/高频能量）做 SBC。 |
| [Feynman-Kac-Bias-Stability](./posterior-calibration/%5BArxiv%202026%5D%20Feynman-Kac-Bias-Stability/) | Arxiv 2026 | Delgadino 等：Feynman–Kac 表示证明**即便 prior score 精确，DPS 仍系统偏差**（丢掉 reaction 项 $c_{DPS}$），在"流形宽×reward 敏感"处放大 → 漏模态。理论依据："用扩散模型 ≠ 得到贝叶斯后验"。 |
| [Principled-Posterior-Matching](./posterior-calibration/%5BArxiv%202026%5D%20Principled-Posterior-Matching/) | Arxiv 2026 | Bai 等：近似目标 → mode collapse / 不可靠 UQ；提出目标级无偏的 PPM。支撑"样本离散≠不确定性已校准"。 |
| [Exact-Posterior-Score](./posterior-calibration/%5BArxiv%202026%5D%20Exact-Posterior-Score/) | Arxiv 2026 | Mammadov 等：低维+高斯/GP 可解析先验+已知线性算子下 $p(x_0\mid y)$ 是**解析高斯后验** → 可作 SBC/coverage/CRPS 的 **gold-standard 参考后验**；$p(A)\to p(A\mid\varphi)$ 时 pivot 需用当前 $\varphi$ 构造，是 gauge-aware 联合估计的必要性来源。 |
| [Diffusion-Annealing-Benchmark](./posterior-calibration/%5BArxiv%202025%5D%20Diffusion-Annealing-Benchmark/) | Arxiv 2025 | Crafts & Villa：系统 benchmark 扩散退火式贝叶斯逆问题求解器能否给出严格 UQ；与本课题同问题域，确认"诊断+可迁移修复"是增量贡献。 |

---

## 🧭 对本课题的可复用结论

- **参考后验从哪来**：Exact-Posterior-Score 给出低维解析高斯后验的构造，作第 1 周"可知真后验"单元测试的 gold standard。
- **校准怎么测**：SBC（rank histogram）+ coverage/CRPS/spread–skill，$x$ 维再叠 score-KSD（Qiu）。
- **偏差为什么存在**：Feynman-Kac / PPM 从理论与变分两侧证明近似采样器会偏置、漏模态、过窄——这是我们诊断章的骨架。
- **基线差在哪**：BlindDPS/GibbsDDRM/PRISM 均无 gauge 商空间处理、$\sigma$ 联合推断不完整、缺分布级校准证据——正是主线三条贡献（诊断 / gauge-aware 联合采样 / 校准）要补的缺口。

*每篇论文目录内含「批读格式」笔记：原文完整保留 + 内嵌 `Hao 批注`，README 含数据流、关键数字、Citation Landscape 与阅读 Q&A。*
