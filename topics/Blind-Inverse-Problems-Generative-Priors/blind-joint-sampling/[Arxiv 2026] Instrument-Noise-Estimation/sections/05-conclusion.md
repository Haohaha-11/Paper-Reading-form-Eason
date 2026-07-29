[← 返回 README](../README.md)

# V. Conclusion 结论

## 📌 预览

总结：在"线性算子 + 加性高斯噪声 + 扩散先验"的逆问题里，本文用 **Hyper-G-DPS**（基于作者自己的 G-DPS [1]）**同时估计 PSF 宽度、噪声均值与方差、以及图像**，并给出 UQ。核心机制是"Gibbs 大循环把难题拆成好采的条件子问题"。展望：模型选择（从候选列表里选仪器/噪声模型）。

---

This paper deals with numerical methods for solving inverse problems when the observation model is linear with additive Gauss noise. The focus is on the delicate issue of estimating multiple parameters of the observation system: width of the point spread function and also mean and variance of noise / error. This issue has already been addressed in the literature and several solutions have been proposed, but not really (see Remark 1) in cases where the prior for the image is defined on the basis of a diffusion model. It is the specificity of the proposed method to estimate them in addition to the image of interest in that case. To this end, our recent contribution [1] allows for proper handling of conditional distributions for images, thereby enabling the inclusion of the conditional posterior for parameters and posterior given these parameters. More precisely, a Gibbs loop splits the overall problem in far simpler sub-problems: iteratively sample each parameter and each image under its conditional posterior.

> 💡 **机制拆解**（一句话定位贡献）（Hao 批注）：作者把贡献锚定在"**扩散先验下同时估仪器 + 噪声参数**"这个空位上——文献里参数估计做得很多，但都不是在扩散先验下（Remark 1 排除了 GibbsDDRM：它不估噪声、不在隐变量间交替）。使这件事可行的**唯一使能条件**是 G-DPS [1] 对图像条件分布的"正确处理"（把扩散链拆成可精确采样的高斯块），从而参数的条件后验能被干净地插进同一个 Gibbs 循环。**方法学骨架 = Gibbs 分块**，一句话可复述。

The simulation study focuses on parameter estimation issue and it is based on the MNIST example set. The proposed method provides accurate and coherent elements for uncertainty quantification, as well as accurate parameters estimation and image restoration. The numerical study also confirms the remarkable computational efficiency.

> 💡 **claim 核对**（Hao 批注）：三条结论 claim 与实验对应——"准确参数估计"（Tab. I 相对误差 1–4%）、"自洽 UQ"（±2 PSD 覆盖 + Fig. 5 像素带）、"高效"（62 秒/一次迭代一次网络）。**都被单场景实验支撑，但都停在"个案"层面**：没有跨场景覆盖率统计，"coherent UQ" 仍是弱意义（区间盖住了真值）而非强意义（频率覆盖率≈名义值）。本课题若引用它，应把它定位为"方法可行性 + 定性 UQ"而非"已校准后验"。

Conclusively, the paper addresses the crucial question of estimating instrument and noise parameters, in addition to the unknown image, in inverse problem based on a diffusion prior. It provides a novel solution referred to as Hyper-G-DPS, that is shown to be accurate and efficient.

To go further, it would be interesting to address model selection [27], [28], especially selection of a model for instrument an / or noise from a given list [29].

> 💡 **展望批读**（模型选择这条线的分量）（Hao 批注）：作者把下一步指向**模型选择**——不只是估某个 PSF 的宽度，而是从候选模型列表里选"哪种 PSF / 哪种噪声模型"（[29] 是作者自己无真值下比较 Wiener-Hunt 去卷积模型的工作）。这在贝叶斯框架里天然（用证据/边缘似然比较模型），且 Gibbs 输出的样本可用于估边缘似然。对本课题的启发：**参数估计 → 模型选择**是"从点/参数不确定性上升到结构不确定性"的自然延伸，也是校准可以进一步施力的地方。

> 💡 **Section 小结**（Hao 批注）：
> - **核心洞察**：本文是一篇"**方法迁移 + 能力扩展**"的短文——把 G-DPS 从"给定参数采图像"扩到"联合采图像 + 仪器 + 噪声参数"，靠的是 Gibbs 分块而非新近似。
> - **相对本课题的位置**：它坐在 [GibbsDDRM](../%5BICML%202023%5D%20GibbsDDRM/) 与 [PRISM](../%5BArxiv%202025%5D%20PRISM/) 之间——比 GibbsDDRM 多估噪声、UQ 更全；比 PRISM 更强调 MCMC 收敛性与共轭直采，但校准验证的严格度都还不到 SBC/CRPS。
> - **可追问点**：forward≈backward 近似对"真后验"的影响、多参数 PSF 下 MH 的可扩展性、以及展望里的模型选择如何与校准结合。
