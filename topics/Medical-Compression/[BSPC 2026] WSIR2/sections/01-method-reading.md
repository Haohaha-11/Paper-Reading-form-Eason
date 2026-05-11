[← 返回 README](../README.md)

# Method Reading

## 📌 预览

WSIR2 的方法主线是：先承认 WSI bag 内存在大量重复组织模式，再用 token compression module 对 patch tokens 做 top-k 保留和非关键 token 融合，最后用 cross-attention blocks 组成轻量分类器完成 slide-level diagnosis。

> 💡 **问题动机**: WSI MIL 常把上万 patches 当作 bag 处理，模型越强，聚合阶段越贵。WSIR2 的切入点不是“bag 里每个 token 都同等重要”，而是“病理图像有重复结构，许多 token 对诊断是冗余或可合并的”。

## Token compression module

公开摘要页描述该模块会“选择 top-k most informative patches，并融合 less informative ones”，同时降低 token 数量和 feature dimensions。

> 💡 **机制拆解**: top-k selection 解决 token 数问题，merging 解决“低分 token 直接丢弃可能丢上下文”的问题，feature-dimension reduction 则让压缩收益不止发生在 attention 复杂度上，也发生在后续 MLP/classifier 计算上。

## Cross-attention importance signal

摘要页和 Introduction 片段都把 cross-attention 的 attention values 解释为 token significance 指标，并强调该机制具备 O(n) 复杂度。

> 💡 **公式/复杂度批读**: O(n) 的意义在于 WSI token count 很大时，标准 self-attention 的 O(n^2) 聚合会迅速成为瓶颈。cross-attention 用较小 query 集合读取 token bag，理论上更适合做轻量 slide-level aggregation。

## Lightweight classifier

WSIR2 在压缩后继续用若干 cross-attention blocks 构造 lightweight classifier，进一步降低 feature aggregation cost。

> 💡 **数据流批注**: 这一步说明作者没有把 compression 当成独立预处理，而是把“压缩后的 token set”直接送入轻量聚合器。若压缩模块可插入其他 MIL 方法，则这个 classifier 是 WSIR2 自身高效版本；token compression module 则可能是可迁移组件。

## 与 topic 中其他论文的关系

- WISE 是无损 byte-stream / WSI 文件压缩，目标是存储 bit-exact。
- AdaSlide 是诊断保持型有损压缩，关注 patch keep/compress 决策和重建。
- FOCUS 是 few-shot WSI 分类中的 visual token filtering。
- DTC-WSI 是动态多阶段 token fusion/pruning。
- WSIR2 是 MIL 诊断 pipeline 内的 token redundancy reduction。

> 💡 **横向比较批注**: WSIR2 最接近 DTC-WSI 和 FOCUS，但它的公开来源更强调 cross-attention importance、top-k patches 和轻量 classifier；这使它更像“将 MIL 聚合器做薄”的工程路线。
