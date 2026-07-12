[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览

结论重申三点：PCGS 采样 + 平稳分布不变的理论保证、两任务实验（尤其保真度）、problem-agnostic（单个预训练扩散跨任务）。并诚实点出唯一局限：SVD 计算不可行的算子不适用。

---

We have proposed GibbsDDRM, a method for solving blind linear inverse problems by sampling data and the parameters of a linear operator from a posterior distribution by using a PCGS. The PCGS procedure ensures that the stationary distribution is unchanged from that of the original Gibbs sampler. GibbsDDRM performed well in experiments on blind image deblurring and vocal dereverberation, particularly in terms of preserving the original data, despite its use of a simple prior distribution for the parameters. Additionally, GibbsDDRM has problem-agnostic characteristics, which means that a single pre-trained diffusion model can be used for various tasks. One limitation of the proposed method is that it is not easily applicable to problems involving linear operators for which the SVD is computationally infeasible.

> 💡 **结论批读 (Hao 批注)**: 三个卖点收束干净——（1）PCGS 联合后验采样 + 平稳分布不变（理论）；（2）两任务强保真（实证）；（3）简单先验 + problem-agnostic（实用性）。**唯一承认的局限**：算子 SVD 不可行时不适用。这条局限很实：本文所有效率优势都建立在"卷积算子可用 FFT 高效做 SVD"上（附录 B），一旦算子非结构化（如一般非线性退化、大尺度断层扫描算子），SVD 就成瓶颈。

> 💡 **Q&A 批注记录 — 与本课题的差距盘点 (Hao 批注)**:
> - Q: GibbsDDRM 作为本课题最近基线，还有哪些没做、正好是我们的切入点？
> - A: 结合全文，四个缺口：（1）**噪声未联合估**——$\sigma_\mathbf{y}$ 全程已知常数，我们要把 $\sigma$ 纳入联合后验；（2）**无校准检验**——只报点指标（PSNR/LPIPS/FID/FAD），从不检验后验是否被正确采样，我们用 SBC/coverage/CRPS 补；（3）**gauge 冗余未显式处理**——模糊核尺度靠归一化硬约束消掉，未做 gauge-aware 后验；（4）**局限本身**——SVD 依赖限制了算子类型，也限制了可扩展到的参数化形式。GibbsDDRM 的价值在于它已经把"块式联合采样（图像块 DDRM + 算子块 Langevin，PCGS 交替）"这条路走通并给了理论保证，是我们块式联合核最直接的对照与出发点。

---

## 🔖 Section 总结

### 核心洞察
1. **一句话定位**：GibbsDDRM = 用 PCGS 从 $p(\mathbf{x}_0,\varphi\mid\mathbf{y})$ 联合采样解盲线性逆问题，数据侧强扩散先验 + 算子侧通用简单先验，平稳分布 = 真后验。
2. **相对点估计的优势**：$\varphi$ 是 Langevin 采样而非 MAP，两块高频交替逼近联合贝叶斯，附录 D 证明比 MAP 更稳定。
3. **局限**：SVD 不可行的算子不适用；采样慢（56 s/图）；$\sigma$ 已知、无后验校准。

### 可追问点（本课题接续）
- 把 $\sigma$ 纳入联合后验 + gauge-aware 处理 + SBC/coverage/CRPS 校准 = 相对 GibbsDDRM 的直接增量。
- 突破 SVD 依赖（如用可学习提议 / latent 空间算子）以扩展算子类型。
