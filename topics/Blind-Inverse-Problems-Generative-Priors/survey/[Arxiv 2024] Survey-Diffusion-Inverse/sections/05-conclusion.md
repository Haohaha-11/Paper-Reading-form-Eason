[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览

结论重申四大家族划分，以及它们的共同敌人：后验分布的 intractability（Eq. 1.3 / 2.20 那个积分）。

---

## 5 Conclusion

In this survey, we discussed different types of inverse problems and different approaches that have been developed to solve them using diffusion priors. We identified four distinct families: methods that propose explicit approximations for the measurement score, variational inference methods, CSGM-type frameworks and finally approaches that asymptotically guarantee exact sampling (at the cost of increased computation). The different frameworks and the works therein are all trying to address the fundamental problem of the intractability of the posterior distribution. In this survey, we tried to unify seemingly different approaches and explain the trade-offs of different methods. We hope that this survey will serve as a reference point for the vibrant field of diffusion models for inverse problems.

> 💡 **机制拆解 (Hao 批注)**: 一句话收束全文的因果链——**所有方法的根源难题是 $p_t(y|x_t)=\int p(y|x_0)p(x_0|x_t)dx_0$ 这个积分不可算**（Eq. 1.3/2.20）。四家族是对付它的四种策略：Explicit 用 $\delta$ 或高斯近似 $p(x_0|x_t)$；Variational 换一个可算的 $q$；CSGM 绕开积分改优化 latent noise；Asymptotically Exact 用 MC 数值逼近积分。**记住这个"一因四果"结构，就抓住了整篇地图的骨架。**

## Acknowledgments

This research has been supported by NSF Grants AF 1901292, CNS 2148141, Tripods CCF 1934932, IFML CCF 2019844 and research gifts by Western Digital, Amazon, WNCG IAP, UT Austin Machine Learning Lab (MLL), Cisco and the Stanly P. Finch Centennial Professorship in Engineering. Giannis Daras has been supported by the Onassis Fellowship (Scholarship ID: F ZS 012-1/2022- 2023), the Bodossaki Fellowship and the Leventis Fellowship. The authors would like to thank our colleagues Viraj Shah, Miki Rubinstein, Murata Naoki, Yutong He, and Stefano Ermon for helpful discussions.

---

## 🔖 Section 总结

### 核心洞察
- 四家族 = 对付"后验 intractability"的四条技术路线，Gupta et al. [136] 甚至证明存在实例使 posterior sampling 需超多项式时间（见 02 节）——这是理论上的硬下界，也解释了为何没有"银弹"方法。
