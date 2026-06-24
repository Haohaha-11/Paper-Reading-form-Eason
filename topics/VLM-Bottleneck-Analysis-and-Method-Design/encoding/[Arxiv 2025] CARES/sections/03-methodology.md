[← 返回 README](../README.md)

# 3. CARES

## 📌 预览

本节是核心方法，分四个子节：(3.1) 问题定义——给定 (x, q)，找最小充分分辨率；(3.2) 标注策略——用目标 VLM 的多分辨率 rollout + ANLS 收敛规则生成离散监督标签（Algorithm 1）；(3.3) 模型架构——冻结 SmolVLM 中间层 + 轻量分类器（350M 参数）作为 discriminative instantiation，以及 Granite-Docling + LoRA 作为 autoregressive instantiation；(3.4) 从离散到连续——训练时 K-way 分类，推理时 softmax 期望插值得到连续分辨率（Algorithm 2）。

---

## 3.1 Problem Definition

Given an image x and query q, let R = [$r_{min}$, $r_{max}$] subset of R+ denote the range of valid input resolutions and let F be a fixed VLM. For any resolution r in R, we denote by x^(r) the image x resized such that its largest dimension equals r. Feeding x^(r) and q into F yields an output y = F(x^(r), q). The VLM forms T(r) visual tokens at resolution r (including AnyRes/tiling effects). Our goal is to learn a selector $f_{theta}$ that predicts, from a single inexpensive low-resolution pass at $r_{min}$, the minimal sufficient resolution $r_{s}$ in R for accurately answering the query q given image x.

> 💡 **批注**: 问题定义为后续所有设计选择提供了 formal ground。关键约束：(1) 只需要一次 cheap low-res pass 就能做预测；(2) 目标是"最小充分分辨率"（minimal sufficient），不是"使 accuracy 最高的分辨率"；(3) T(r) 考虑了 AnyRes/tiling 效应——不同 VLM 在相同 r 下可能产生不同数量的 visual tokens。

> 💡 **批注**: 注意定义中的 "VLM forms T(r) visual tokens"——这意味着 visual token count 是 resolution 的函数，但不一定是简单的线性关系（AnyRes 会饱和，Qwen2.5-VL 近似平方增长）。这个细节影响 CARES 的实际节省量。

## 3.2 Labeling Strategy for Training CARES

Since searching for the optimal r* in R is prohibitively expensive, we chose to use a small, discrete set of valid resolutions for the annotation Rd = {r_1, ..., $r_{K}$} subset of R. For each sample, we render the image at the fixed resolutions Rd, and use a pretrained VLM to generate predictions at each resolution. The predictions are evaluated against the ground-truth annotations using the ANLS metric. The supervision label is assigned as the lowest resolution whose ANLS score exceeds a threshold, without significant improvement at higher resolutions. The procedure yields a discrete sufficiency label r* in Rd per example. We emphasize that discretization is only used for supervision efficiency; at inference, we deploy a continuous finer-grained selector (§3.4). Algorithm 1 outlines the data generation process, and Table 1 visualizes the concept.

> 💡 **批注**: 标注策略的三个关键设计：(1) 离散化只为降低标注成本（否则需要搜索连续空间）；(2) 用目标 VLM 自身来定义"充分性"——这保证了标签和部署的一致性；(3) "无显著提升"（delta <= 0.1）避免为 trivial improvement 升级分辨率。

Formally, we compute the ANLS score for each resolution:

$u_{k}$ = ANLS(F(x^($r_{k}$), q), gt) in [0, 1]

and select the minimal sufficient resolution as:

r* = min{ $r_{k}$ | $u_{k}$ >= tau, max_{l > k}($u_{l}$ - $u_{k}$) <= delta }

where we default to $r_{K}$ if no resolution satisfies the condition. We set tau = 0.85 and use a small margin delta (e.g., 0.1) to prevent rewarding negligible performance improvements. We define the full resolution range as R = [384, 1024], and use a discrete set Rd = {384, 768, 1024} for annotation.

> 💡 **公式批读**: 收敛规则的数学表达有两部分：(1) $u_{k}$ >= tau——当前分辨率已足够好；(2) max_{l>k}($u_{l}$ - $u_{k}$) <= delta——更高分辨率没有显著提升。两个条件缺一不可：条件(1)保证充分性，条件(2)保证"最小"性。

> 💡 **批注**: tau=0.85 和 delta=0.1 的选择值得关注。tau 太低会过早标记为"充分"（可能导致 CARES 过于激进地降分辨率），tau 太高会导致大多数样本都被标注为 $r_{K}$（失去 routing 效果）。delta 控制对微小提升的容忍度。这两个超参数在论文中没有做详细消融，是潜在的风险点。

**Algorithm 1: Labeling via multi-resolution sufficiency rollouts.**

```
Input: (x, q); resolutions R; VLM F; utility U; threshold tau; margin delta
Output: Label r* in R
for k <- 1 to K do
    $y_{k}$ <- F(x^($r_{k}$), q); $u_{k}$ <- U($y_{k}$, gt)
for k <- 1 to K do
    if $u_{k}$ >= tau and max_{l > k}($u_{l}$ - $u_{k}$) <= delta then
        return r* = $r_{k}$
return r* = $r_{K}$
```

> 💡 **批注**: Algorithm 1 简单但优雅。它体现了"first-satisfaction"原则——从低分辨率开始扫描，第一个满足条件的就被选中。这意味着在低分辨率下已经能正确回答的样本不会被标注为高分辨率。

> 💡 **Table 1 批读**: Table 1 用两个具体例子说明标注过程。第一个例子（"What is the contact person name?"）在 384 下 ANLS=1.0 就已经 perfect，所以标注为 384。第二个例子（"One variable that has implicitly not been controlled?"）在 384 下 ANLS=0.0（完全错误），768 下 ANLS=0.65（不够好），1024 下 ANLS=0.93（充分且收敛），所以标注为 1024。这个对比很好地说明了不同 query 对分辨率的依赖程度。

## 3.3 Model Instantiations

Unless otherwise stated, all main experiments in this paper use the following discriminative instantiation of CARES.

We design CARES as a lightweight resolution selector that can be deployed in front of any vision–language model (VLM) to improve efficiency. Its behavior is governed by three core principles:

1. Compactness: minimal overhead in computation and memory.
2. Preprocessing role: determines resolution directly from raw inputs before invoking the VLM.
3. VLM-agnosticism: works with any VLM, whether run locally or accessed via API, with no architecture changes or retraining required.

> 💡 **批注**: 三个设计原则值得逐一推敲。Compactness 要求 proxy VLM 足够小（~350M），但又要能提取足够的信息来预测分辨率需求——这里有个 trade-off。Preprocessing role 是最根本的差异化定位。VLM-agnosticism 体现为不访问 VLM 内部状态，只控制输入分辨率。

To implement these principles, we use a compact frozen VLM backbone as a joint vision–text feature extractor, followed by a lightweight classifier head.

Specifically, we adopt the pretrained SmolVLM-500M model (Marafioti et al., 2025), with layers 17–32 removed, as the backbone. Given an image at resolution $r_{min}$ and a text query, we feed both into the model and extract the hidden state of the final token at layer 16. This representation encodes the joint image–query context and is passed to a classifier that outputs a soft distribution over target resolutions. This design is motivated by recent findings showing that intermediate layer activations in LLMs and VLMs encode rich perceptual and semantic information that may not be surfaced at the output layer (Orgad et al., 2024; Zhang et al., 2025a). In addition to being more informative, as also evidenced by the performance gap in Table 3 where using intermediate features outperforms last-layer features by about 1%, this choice substantially reduces computation since only roughly half of the LLM is used for feature extraction.

> 💡 **批注**: 使用中间层（layer 16）而非最后层（layer 32）是 CARES 的一个关键设计选择。两个理由：(1) 中间层包含更丰富的感知信息（已有文献支持）；(2) 计算量减半。Table 3 的消融显示中间层比最后层高约 1% 准确率，同时少用约 150M 参数。这是一个用更少计算获得更好性能的纯增益。

> 💡 **批注**: SmolVLM-500M 的选择体现了实用主义——不是最大最强的特征提取器（Qwen2.5-3B 准确率更高，达到 67.2%），而是 performance/size/efficiency trade-off 最优的（350M，63.3%）。

The resulting CARES module has approximately 350M parameters and is trained with supervision over discrete resolution labels (see §3.2).

**Autoregressive document-specialized instantiation.** In addition to the discriminative selector above, we also instantiate CARES using an autoregressive vision-language model. Concretely, we start from Granite-Docling-258M (Auer et al., 2024) and fine-tune it with LoRA (rank 8) on the same resolution-selection training set. Given the low-resolution image and the query, the model is prompted to predict one resolution label from the discrete set Rd = {384, 768, 1024}. To avoid tokenization ambiguity, we map these labels to dedicated tokens <1>, <2> and <3>.

At inference time, we read the first-step logits over the resolution tokens, apply a softmax to obtain class probabilities, and use the same expectation-based interpolation described in Eq. 3 to produce a continuous resolution. This preserves the deployment rule of CARES while replacing the discriminative head with an autoregressive predictor.

> 💡 **批注**: AR instantiation 的存在有两个意义：(1) 展示 CARES 不是一个特定架构，而是一个通用范式；(2) Granite-Docling 专门针对文档任务，所以在文档 benchmark 上可能更激进。实际结果（Table 2）显示 AR 变体确实在 document 任务上更激进地降分辨率（-80% to -88% FLOPs），但精度也略低于 discriminative 变体。

> 💡 **批注**: 专用的 resolution tokens（<1>, <2>, <3>）避免了 tokenization ambiguity——如果直接用数字 token，384/768/1024 可能被分词器切成多个 token，导致预测困难。

## 3.4 From Discrete Supervision to a Continuous Resolution

Although CARES is trained as a K-way classifier over a discrete set of resolutions Rd = {r_1 < ... < $r_{K}$}, we deploy it as a continuous selector over R = [$r_{min}$, $r_{max}$]. Given features z from the low-resolution image and query, compute logits l(z) in R^K and class probabilities

p = softmax(l)

We use the probability-weighted expectation over Rd:

$r_{tilde}$ = SUM_{k=1}^{|Rd|} $p_{k}$ * $r_{k}$

This yields a continuous resolution that varies smoothly with confidence and is insensitive to the specific discretization used for labeling. In practice, $r_{tilde}$ preserves the routing behavior of the classifier while allowing finer control.

> 💡 **公式批读**: Eq. 3 是整个 continuous inference 的核心。预期值比 argmax 有更多信息——当模型在两个分辨率之间犹豫时（如 p_384=0.6, p_768=0.4），预期值会给一个中间值（~538），实现更平滑的 routing。

> 💡 **批注**: "insensitive to the specific discretization" 是一个重要的鲁棒性声明。即使训练用的 Rd 很粗糙（如 {384, 1024}），推理时通过插值仍能获得 384-1024 之间的任意分辨率值。

**Algorithm 2: Continuous resolution selection.**

```
Input: (x, q); low-res r_1; logits l.
Output: Continuous resolution $r_{tilde}$ in [r_1, $r_{K}$].
z <- features from proxy VLM at r_1
p <- softmax(l(z))
$r_{tilde}$ <- SUM_{k=1}^{K} $p_{k}$ * $r_{k}$
return $r_{tilde}$
```

**Deployment.** The target VLM receives x with the largest dimension resized to $r_{tilde}$ (or to the nearest supported side length to avoid under-allocation). For backbones that only accept a discrete set of input sizes, we round up to the next supported size.

> 💡 **批注**: "round up to the next supported size" 是一个实用的工程决策——避免因分辨率不足导致性能退化。但这个 rounding 也会削弱连续分辨率的优势，尤其是在离散支持的尺寸之间有大 gap 的模型中。

---

## 🔖 Section 总结

### 核心洞察
1. 标注策略是 CARES 的基石——如果没有合理的"充分分辨率"定义，训练出来的 selector 不会有意义。tau=0.85 和 delta=0.1 的选择决定了标注的保守程度。
2. SmolVLM 中间层特征（layer 16）的选择体现了 "less is more"——半层模型提取的特征反而比完整模型更好，因为中间层保留了更多感知信息。
3. "离散训练 + 连续推理" 的设计解决了标注效率（离散便宜）和部署精度（连续更优）之间的矛盾，label smoothing 是连接两者的桥梁。
4. AR instantiation 表明 CARES 是一个范式而非特定架构，但 discriminative instantiation 在大多数 benchmark 上优于 AR。
