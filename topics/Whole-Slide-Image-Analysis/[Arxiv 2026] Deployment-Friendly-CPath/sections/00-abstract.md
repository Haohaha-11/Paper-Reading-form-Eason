[← 返回 README](../README.md)

# Abstract 摘要

## 📌 预览

**LitePath** 是一个"部署友好"的病理基础模型框架，同时治两种低效：**(1) 模型过参数化**——用 **LiteFM**（从 Virchow2 + H-Optimus-1 + UNI2 三个大 PFM 蒸馏、190M patch、ViT-Small、22.5M 参数）；**(2) patch 级冗余**——用 **APS（Adaptive Patch Selector）** 混合"均匀采样 + 注意力采样"选 patch。相对 Virchow2 参数减 28×、FLOPs 减 403.5×，能跑在 $249 的 Jetson Orin Nano 边缘设备上（208 slide/h，快 104.5×，能耗低 171×），37 cohort 上排名第 2/19、保留 99.71% AUC。

---

## ABSTRACT

Pathology foundation models (PFMs) have enabled robust generalization in computational pathology through large-scale datasets and expansive architectures. However, the substantial computational cost of these models, particularly when analyzing gigapixel whole slide images, limits clinical accessibility and scalability. Here, we present LitePath, a deployment-friendly foundational framework designed to mitigate model over-parameterization and patch-level redundancy. LitePath integrates LiteFM, a compact model distilled from three large PFMs (Virchow2, H-Optimus-1 and UNI2) using 190 million patches, and the Adaptive Patch Selector (APS), a lightweight modular component for task-specific patch selection. The framework reduces model parameters by 28× and lowers FLOPs by 403.5× relative to Virchow2, enabling deployment on low-power edge hardware such as the NVIDIA Jetson Orin Nano Super. On this device, LitePath achieves a processing speed of 208 slides per hour, 104.5× faster than Virchow2, and consumes 0.36 kWh per 3,000 slides, 171× lower than Virchow2 on a standard RTX3090 GPU. We validated accuracy using 37 cohorts across four organs and 26 tasks (26 internal, 9 external, and 2 prospective cohorts), comprising 15,672 slides from 9,808 patients that are disjoint from the pretraining data. LitePath ranks second among 19 evaluated models in average ranking scores, outperforming larger models including H-Optimus-1, mSTAR, UNI2 and GPFM. Compared to the top-performing model (Virchow2), LitePath retains 99.71% of the AUC on average. To quantify the balance between accuracy and efficiency, we propose the Deployability Score (D-Score), defined as the weighted geometric mean of normalized AUC and normalized FLOPs scores, where LitePath achieves the highest value, surpassing Virchow2 by 10.64%. These results demonstrate that LitePath enables rapid, cost-effective and energy-efficient pathology image analysis on accessible hardware while maintaining accuracy comparable to state-of-the-art PFMs, facilitating accessible precision oncology and reducing the carbon footprint of AI deployment.

> 💡 **问题动机 + 双低效框架**（Hao 批注）：LitePath（HKUST Hao Chen 组，与 [PathBench](../%5BArxiv%202025%5D%20PathBench/) 同源）攻击 PFM 落地的两个正交低效：
> 1. **模型过参数化**（横向）：PathBench 发现十亿参数模型（H-Optimus-0/Prov-GigaPath）未必超过紧凑的 Virchow2 → 小模型可行 → **LiteFM 蒸馏**（22.5M 参数，28× 小）。
> 2. **patch 级冗余**（纵向）：诊断只依赖少数 ROI，但现有方法无差别处理所有 patch → **APS 自适应选 patch**（10.4× 少）。
> 两者相乘 = 403.5× FLOP 削减。**这与 [EAGLE](../%5BNat%20Commun%202026%5D%20DL-Efficient-Pathology/) 是同一主题的两种解法**：EAGLE 选 25 tile + 现成 FM；LitePath 蒸馏小 FM + APS 选 patch。EAGLE 只压 patch 数，LitePath 同时压模型和 patch。

> 💡 **机制拆解 + 对 ReadySlide 的相关性**（Hao 批注）：
> - **APS = patch retention 的另一种实现**：混合"均匀采样（保覆盖）+ 注意力采样（保信息区）"，模仿病理学家"先广筛后聚焦"。这与 EAGLE 的 CHIEF 选择、ReadySlide 的 allocator 是同一族。**关键差异**：APS 显式加了"均匀采样保覆盖"，防止纯注意力采样漏掉分散信号——这对 ReadySlide 有启发（纯 importance 保留可能漏覆盖）。
> - **D-Score（可部署性分数）**：归一化 AUC 与归一化 FLOPs 的加权几何均值——**一个显式的"精度-效率"Pareto 标量**。ReadySlide 评估压缩方法时正需要这种复合指标（不能只报 AUC 或只报 CR）。
> - **蒸馏多个 FM**：LiteFM 从 3 个互补 PFM（组织学/分子/预后专长）蒸馏 → FM-agnostic substrate 的另一形态（呼应 memory 里"P3 FM-agnostic substrate"）。
