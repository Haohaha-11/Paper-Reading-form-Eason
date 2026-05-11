[← 返回 README](../README.md)

# 5. Discussion

## 📌 预览

Discussion 承认 WSIR2 的主要价值和边界：它通过 token pruning、token merging 和 lightweight classifier 降低成本，plug-and-play 能加速多种 MIL 模型；但 fixed top-k ratio、极端 pruning、预提取 patch features、缺少显式局部空间上下文，都会限制临床部署和任务泛化。

## 原文要点

WSIR2 effectively addresses redundancy in WSIs by leveraging token pruning and merging alongside a lightweight classifier, achieving substantial reductions in computational cost and model size while preserving critical diagnostic information.

> 💡 **主 claim 回收**: 这句话把全文贡献闭环：redundancy -> pruning/merging -> lower cost/model size -> diagnostic information retained。每个箭头都在 Method 和 Experiments 中有对应证据。

Its plug-and-play design enables acceleration of various MIL models, demonstrating strong generalizability, and the visualization results confirm its focus on diagnostically relevant regions.

> 💡 **证据对应**: plug-and-play 对应 Table 3；visualization 对应 Figure 4。强 generalizability 目前主要限于 ABMIL/DSMIL/TransMIL 和三个分类数据集，不能直接扩展到所有病理任务。

However, WSIR2 relies on a fixed top-k ratio, which may be suboptimal for highly heterogeneous WSIs, and extreme pruning can slightly degrade accuracy.

> 💡 **核心风险**: fixed r 是最实际的临床风险。大面积正常组织、低肿瘤含量、小病灶、坏死区域、染色异常 slide 需要不同 token budget；固定比例既可能浪费，也可能漏掉稀疏关键证据。

Additionally, it depends on pre-extracted patch-level features and does not explicitly capture local spatial context, which may limit performance for tasks requiring fine-grained detail.

> 💡 **pipeline 风险**: 预提取 feature 让 MIL 阶段更高效，但端到端系统仍要承担 patch extraction 和 feature encoding 成本。没有显式空间上下文也意味着相邻组织结构、腺体形态、边界模式可能被 feature bag 表达削弱。

Future improvements could include adaptive token selection, multi-scale or hierarchical attention to better model local and global context, integration with end-to-end trainable backbones to reduce preprocessing overhead, and extension to other pathological tasks.

> 💡 **未来工作批注**: 这四个方向基本覆盖 WSIR2 的短板：adaptive budget 解决 fixed r；multi-scale/hierarchical attention 解决空间和尺度；end-to-end backbone 解决预处理断裂；other tasks 验证泛化。

## 读后风险清单

| 风险 | 为什么重要 | 可验证方式 |
|---|---|---|
| fixed top-k ratio | 不同 slide 病灶面积差异大 | 按 tumor purity / tissue area / WSI size 分层评估 |
| attention significance | attention score 不必然等于临床重要性 | lesion-level recall、expert review、mask overlap |
| single redundant token | 多种背景组织可能被压得过粗 | multi-prototype merging 消融 |
| pre-extracted features | classifier FLOPs 不等于端到端成本 | 统计 tiling + encoder + MIL + I/O runtime |
| classification-only evidence | 细粒度任务可能更依赖局部结构 | survival/retrieval/VQA/localization 实验 |

## Q&A 批注记录

- Q: fixed r 能不能通过 validation set 选择解决?
  A: 只能部分解决。validation set 给出全局最优 r，但每张 WSI 的有效组织面积和病灶稀疏度不同，最优 token budget 可能是 slide-specific。

- Q: WSIR2 会不会删掉临床罕见但关键的区域?
  A: 有可能，特别是 r=0.01 这类极端压缩。作者用 Figure 4 展示 top-k 区域与肿瘤区域对齐，但需要更多失败案例和 region-level recall。

- Q: 为什么说预提取 feature 是限制?
  A: 它让 compression 只能在 frozen patch features 上工作，不能端到端修正 patch encoder；同时真实系统的 encoder 计算可能远大于 MIL classifier。

## 🔖 Section 总结

### 核心洞察

1. Discussion 的最大价值是主动承认 fixed top-k ratio 和 spatial context 两个临床关键问题。
2. WSIR2 当前最适合被看作 feature-level MIL 加速器，而不是完整 WSI 端到端加速系统。
3. 下一步最自然的改进是 adaptive token budget + multi-prototype merging + multi-scale spatial modeling。
