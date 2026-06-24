# 02 - Related Work

## 原文 Section II: RELATED WORKS

---

## A. General Perception in MLLMs

### 早期架构：固定低分辨率

经典 MLLM 架构将视觉输入标准化为固定分辨率：

- **Q-Former 路线** [38]（如 BLIP-2）: 将视觉表示压缩进严格的预设 token 预算
- **LLaVA 路线** [1, 14]: 直接采用视觉编码器的密集未压缩 token 序列，投影到 LLM 特征空间

后者因架构简单和经验有效逐渐成为主流范式。

> **Hao 批注**: 这两条路线的分歧本质是"压缩 vs 保留"。Q-Former 试图用少量 query token 压缩视觉信息，但压缩损失严重；LLaVA 保留全部 token 但效率低。Q-Zoom 的选择性 RoI 重编码其实是在两者之间找到了第三条路——不全局压缩也不全量保留，而是精准裁切。

### 高分辨率适配的三条演化路径

**路径一：辅助高分辨率视觉编码器** [41-45]
- 集成 SAM [46] 或 ConvNeXt [47] 等，补偿独立低分辨率 ViT 的空间缺陷

**路径二：空间分割多 patch 编码（AnyRes 路线）** [8, 15, 19, 21, 48-51]
- 将高分辨率输入空间分割为多个局部 patch
- 独立编码后拼接送入 LLM
- 被 AnyRes [21, 49] 机制广泛普及

**路径三：原生动态分辨率（Native Dynamic Resolution）** [2, 20, 52]
- 视觉编码器原生适配处理更高、可变分辨率
- 如 Qwen2-VL 采用 NaViT 架构 [22] 无缝处理任意宽高比

> **Hao 批注**: 这三条路径代表了解决分辨率问题的不同思路。路径一是"换更好的眼睛"，路径二是"多看几次"，路径三是"训练眼睛看更清楚"。Q-Zoom 可以叠加在路径二和路径三之上——它不改变底层的分辨率编码方式，而是在 token 进入 LLM 之前就做筛选和增强。

---

## B. Query-aware Perception in MLLMs

Query-aware 设计的核心原则：先用粗粒度低分辨率输入识别任务相关 RoI，再以高分辨率重编码这些裁剪区域。

> **Hao 批注**: 这是 Q-Zoom 所属的研究子方向。下表整理了三类方法的本质差异。

| 维度 | Training-Free | SFT-Based | RL-Based | **Q-Zoom** |
|------|--------------|-----------|----------|-------------|
| 优化方式 | 无训练 | 有监督微调 | 强化学习 | 轻量分支自蒸馏 |
| RoI 定位方式 | 启发式规则 | 模型预测 heatmap | 模型生成 code/坐标 | 中间特征 heatmap |
| 瓶颈 | 多次 prefill pass | 数据标注昂贵 + 灾难性遗忘 | CoT 解码延迟 + RL 训练昂贵 | 需冻结 backbone |
| 代表方法 | ViCrop [23], FoCus [24], ZoomEye [25] | Token-Efficient VLM [56] | Thyme [5], DeepEyes [4, 70], Mini-o3 [6] | Q-Zoom |

### Training-Free Methods [23, 25, 53-55]

- 依赖手工启发式规则，不更新模型权重
- ViCrop [23]: 计算通用/任务特定文本提示之间的对比交叉注意力来定位相关视觉证据
- **核心问题**: 获取 attention 信号内在需要多次冗余 prefill pass 或计算密集的自回归解码

### SFT-Based Methods [56-58]

- 教导 MLLM 显式预测 RoI heatmap 或调用外部工具
- **核心问题**:
  1. 需要大规模配对 question-annotation 坐标数据集
  2. 全量微调 LLM backbone 计算禁止且存在灾难性遗忘风险

### RL-Based Methods [4-6, 59, 60]

- 将细粒度感知重新表述为自主"Think-with-Image"范式
- 通过 RL 优化模型迭代推断视觉充分性并定位 RoI
- **核心问题**:
  1. 优化整个 MLLM 的 RL 消耗巨大 GPU 显存，训练不稳定
  2. 严重依赖大规模专有教师模型生成可靠 reward 信号
  3. 推理时将计算负担从视觉编码器转移到语言模型——依赖冗长 CoT 解码
- **最新进展**: Latent Thinking 范式 [61, 62] 尝试在潜空间压缩推理轨迹，但不可避免地为模型最终感知性能施加了严格上限

> **Hao 批注**: 这一段对 RL 方法的批判非常到位。"将计算负担从视觉编码器转移到语言模型"是一个关键洞察——虽然 RL 方法减少了 visual token，但 CoT 解码产生的 text token 可能远远超过节省的 visual token，导致端到端延迟反而增加。Q-Zoom 避开 LLM 解码，直接在视觉特征空间操作，是这个思路的差异化优势。

---

### Table I: Resolution-Throughput Trade-off（Qwen2.5-VL 7B）

原文 Table I 展示了 visual token 约束对推理速度的影响：

| Max Visual Tokens | Throughput (samples/s) | DocVQA | OCRBench | InfoVQA | V* | Ave. |
|-------------------|----------------------|--------|----------|---------|-----|------|
| 512 | 4.6 | 90.9 | 83.3 | 69.1 | 62.8 | 76.5 |
| 2048 | 2.2 | 94.7 | 84.8 | 80.4 | 72.3 | 83.1 |

> **Hao 批注**: 这个表极其关键。从 2048 降到 512 tokens，吞吐翻倍（4.6 vs 2.2），精度只掉了 6.6%。这直接证明：**大多数 query 用粗粒度上下文就能解决**，全域高分辨率处理是巨大的浪费。这也是 Dynamic Gating Network 存在的基础。

---

## 本工作的定位

Q-Zoom 不属于上述任何单一范式：
- **不是 training-free**: 需要训练轻量分支（但数据来自自蒸馏，不需要人工标注）
- **不是 SFT**: 不微调整个 LLM backbone（Post-SFT 除外，且仅用 ~7K 硬样本）
- **不是 RL**: 不通过强化学习优化

Q-Zoom 的核心创新在于将 query-aware 感知完全转移到中间特征空间，在单次 prefill pass 中完成，避开了所有现有范式的瓶颈。
