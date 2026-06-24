[← 返回 README](../README.md)

# 3. Methodology

## 一、Preview

Method 分为三个子模块：(3.1) Recurrent Visual-Language Backbone——基于 Huginn 构建多模态 loop transformer，定义文本/视觉的嵌入、拼接、初始化与迭代公式；(3.2) Hierarchical Visual Injection——从 ViT 的 {6,12,18,24} 层提取多尺度特征，通过 Patch Merger 对齐后按"课程"顺序注入到 recurrent block 的前 K 次迭代中（含自适应降采样策略）；(3.3) Training Objective——Poisson 分布采样训练深度 + CE loss，使模型解耦推理步数与视觉融合。

---

## 二、原始文本

### 3.1. Recurrent Visual-Language Backbone

We denote the input sequence length as n, the hidden dimension of the model as h, and the vocabulary as V. Given a recurrent depth R, an iteration step t, an input text sequence x ∈ V^n, and a sequence of flattened image patches X_v, we process the textual and visual modalities separately. For the visual components, we denote the vision transformer and its associated projector as ViT(.) and Proj(.), respectively. The feature extraction and fusion process can be written as:

$e = [ e _ { v } ; e _ { t } ] = \text{concat}( e _ { v } , e _ { t } ), $

where

$\left\{ { \begin{array} { l l } { e _ { v } = \text{Proj}( \text{ViT}( X _ { v } ) ) , } \\ { e _ { t } = E ( x ) , } \end{array} } \right. $

e_v and e_t represent the visual and textual embeddings, respectively.

> 💡 **公式批读 — Eq.1-2（输入构造）**:
> - 文本嵌入 $e_t$ 由 Huginn 的 Embedding Block E 生成（这是 Huginn 三元结构的第一元）
> - 视觉嵌入 $e_v$ 由 ViT 最后一层 + Proj 生成（标准 MLLM 做法）
> - 两者 concat 为 e，这是送入 Recurrent Block 的输入序列
> - **注意**: 这里的 $e_v$ 只是 ViT 最后一层的投影，层级视觉特征（Section 3.2 中的 $v_l$）是额外的注入信号，二者并存。

We denote s_t as the hidden states after t iterations. To stabilize the recurrent iterations, Huginn utilizes a random vector:

$s _ { 0 } \sim N( 0 , \sigma ^ { 2 } I _ { n \cdot h } ). $

In the initial iteration, this vector is concatenated with the input embeddings along the channel dimension, which is subsequently mapped back to the original dimensionality by an adapter within the recurrent block R-Block. In subsequent iterations, the hidden states derived from the preceding block are concatenated with the input embeddings. Let hat(e)_v be the fused visual cues:

$s _ { r + 1 } = R\text{-}Block\left( e , \hat { e } _ { v } ; s _ { r } \right). $

> 💡 **公式批读 — Eq.3-4（递归迭代）**:
> - **$s_0$**: 随机初始化 hidden state（从高斯分布采样），这是循环的起点。随机性提供探索能力。
> - **初始迭代 (t=0)**: 将随机向量 $s_0$ 与输入嵌入 e 沿 channel 维 concat → adapter 映射回原始维度 → 进入 R-Block
> - **后续迭代 (t>0)**: 将前一迭代的 hidden state $s_r$ 与输入嵌入 e 拼接 → 继续 R-Block 计算
> - **关键设计**: 每次迭代都将**原始输入 e** 重新注入——这是一个 skip connection，防止深层迭代中的信号衰减
> - **hat(e)_v**: 注入的视觉线索（如 Eq.6 定义），在特定迭代步注入，其余步为 0

### 3.2. Hierarchical Visual Injection

To empower Huginn with the ability to perceive both structural details and high-level semantics, we move beyond the conventional practice of utilizing only the last layer of the vision encoder. Instead, we introduce a hierarchical visual injection strategy.

Specifically, we extract hidden states from a set of representative layers L = {6, 12, 18, 24}. This selection is motivated by the inherent hierarchical nature of vision transformers:

- **Lower-level Layers** (e.g., Layer 6) retain high-resolution spatial information and primitive visual patterns such as textures and edges, which are beneficial for grounding tasks.
- **Intermediate and Higher-level Layers** (e.g., Layer 12 to 24) gradually aggregate these primitives into complex semantic concepts and global context, providing the model with a holistic understanding of the scene.

> 💡 **机制拆解 — ViT 层级选择的设计逻辑**:
>
> | 层级 | Layer | 特征类型 | 适合任务 |
> |------|-------|---------|---------|
> | 浅层 | 6 | 高分辨率空间信息、纹理、边缘（primitive patterns） | Grounding, localization |
> | 中层 | 12 | 中级语义概念（object parts, shapes） | Object recognition, attribute reasoning |
> | 中高层 | 18 | 复杂语义聚合（objects, relations） | Relation reasoning, scene parsing |
> | 深层 | 24 | 全局上下文和抽象语义（scene-level semantics） | Holistic understanding, reasoning |
>
> **课程式注入的认知基础**: 先看细节（纹理/边缘→定位），再理解语义（概念→关系→全局）。这与人类视觉认知的 coarse-to-fine 或 bottom-up attention 机制有类比性。

To bridge the modality gap and align the dimensionalities, we employ a set of patch mergers inspired by Qwen3-VL: M = {m_l}_{l∈L} (Bai et al., 2025b). For each selected layer l, the visual features h_v^l are projected as:

$v _ { l } = m _ { l } ( h _ { v } ^ { l } ), \quad l \in \{ 6 , 1 2 , 1 8 , 2 4 \}, $

where v_l ∈ R^{n×h} represents the projected visual cues ready for recurrent injection. By progressively injecting these features from fine-grained semantics to coarse-grained textures into the initial recurrent iterations, we provide the language backbone with a "curriculum" of visual understanding, stabilizing the hidden state transition during the early stages of reasoning.

> 💡 **公式批读 — Eq.5（Patch Merger）**:
> - 每个选定的 ViT 层有一组独立的 patch merger m_l（轻量投影模块）
> - 作用：将 ViT 中间层特征 $h_v^l$ 从 ViT 的特征空间映射到 LLM 的 embedding 空间
> - "Curriculum of visual understanding": 从 fine-grained（低层）到 coarse-grained（高层）的渐进式注入，稳定早期推理阶段的 hidden state transition
> - **灵感来源**: Qwen3-VL 的 patch merger，但用法不同——Qwen3-VL 是静态注入到不同 LLM 层，HIVE 是动态注入到不同迭代步

The injection schedule is defined as:

${ \hat { e } } _ { v } = { \left\{ \begin{array} { l l } { v _ { i } } & { \text{if } t < K , } \\ { 0 } & { \text{if } t \geq K . } \end{array} \right. } \quad \text{where } i = L[ t ] $

> 💡 **公式批读 — Eq.6（注入调度）**:
> - 只在前 K 次迭代中注入视觉线索（K 通常是 4，因为视觉层级数为 4）
> - t ≥ K 后：hat(e)_v = 0，模型进入纯语言推理模式（纯隐空间 refinement）
> - i = L[t]：第 t 次迭代注入第 L[t] 层的视觉特征
> - **设计哲学**: 视觉信息只用于初始化/引导推理方向，后续的深度推理在隐空间内自主完成——视觉是"向导"而非"拐杖"

To enhance the robustness of the recurrent reasoning process, the recurrent depth is randomly sampled from a Poisson distribution during the training stage of Huginn. This stochasticity forces our model to decouple the visual-language fusion from a fixed step count.

We introduce an adaptive injection schedule. The core challenge lies in aligning the t iterations with the 4 available visual tiers. We define the injection at step t as follows:

1. **Case I: Sufficient Iterations (R ≥ 4)**. The visual cues are injected in a "top-down" hierarchical order during the initial 4 steps. For t > 4, the model performs pure language modeling to refine the reasoning output.

2. **Case II: Constrained Iterations (R < 4)**. When the sampled depth is shallower than the visual hierarchy, we perform progressive downsampling of the visual cues. Specifically, we select a subset of V with an interval of floor(4 / R) to ensure that even in shallow reasoning, the model still receives a representative spectrum of visual information (e.g., if R = 2, the model integrates {v_1, v_2}).

> 💡 **机制拆解 — 自适应注入策略**:
>
> | 场景 | R 值 | 注入策略 | 具体行为 |
> |------|------|---------|---------|
> | 充分迭代 | R ≥ 4 | Top-down 顺序注入 | t=0: v_6（浅层-纹理）, t=1: $v_12$, t=2: $v_18$, t=3: v_24（深层-语义）, t≥4: 纯语言推理 |
> | 受限迭代 | R = 3 | 降采样 | 间隔 floor(4/3)=1，选 3 层: {$v_6$, $v_12$, v_18} 或 {$v_6$, $v_12$, v_24}... |
> | 受限迭代 | R = 2 | 降采样 | 间隔 floor(4/2)=2，选 2 层: {$v_6$, v_18} 或 {$v_6$, v_24}... |
> | 受限迭代 | R = 1 | 最简 | 只注入 1 层 |
>
> **设计精妙处**: Poisson 分布采样 + 自适应降采样 = 模型学会在任何 recurrency depth 下都能有效利用视觉信息。这是训练和推理之间 decouple 的关键——训练时不固定步数，推理时才能灵活调整。

**Algorithm (Pseudo-code)**:

```python
def core_block_forward(x_in, embd):
    ... # Model expand recurrent blocks here.
    return x_out

def iterate_forward(x, embeds, vis_features):
    n_no_grad, n_grad = random_sampler()

    def get_input(i):
        if i < len(vis_features):
            # Vision features are injected into embeds
            return func(embeds, vis_features[i])
        else:
            return embeds

    with torch.no_grad():
        for i in range(n_no_grad):
            core_block_forward(x, get_input(i))

    for i in range(n_no_grad, n_no_grad + n_grad):
        core_block_forward(x, get_input(i))
```

> 💡 **伪代码批读**:
> - **n_no_grad / n_grad**: 随机采样决定多少步不做梯度回传（truncated BPTT）。前 n_no_grad 步用 `torch.no_grad()`，后 n_grad 步正常计算梯度
> - **get_input(i)**: 核心的分发逻辑——当 i < len(vis_features) 时注入视觉特征，否则只传原始 embed。这是层级注入在代码层面的实际体现
> - **设计要点**: 视觉特征通过 `func(embeds, vis_features[i])` 与基础嵌入融合（可能是 concat 或 add 后过 adapter），具体实现在 `core_block_forward` 的注释 "Model expand recurrent blocks here" 中，论文未详细展开

Finally, after r recurrent iterations, the model decodes the hidden state s_r to produce the output probabilities:

$p = H ( s _ { r } ). $

> 💡 **公式批读 — Eq.7（解码）**: H 是 Huginn 的 Language Head。经过 r 次迭代后，最终的 hidden state $s_r$ 被送入 Head 生成词表上的概率分布。整个过程是标准的自回归语言建模——HIVE 的"新意"不在于解码方式，而在于解码前 hidden state 是如何通过递归 + 层级视觉注入被构建的。

### 3.3. Training Objective

Given an input text sequence x and a set of hierarchical visual features V_hier = {v_(1), v_(2), ..., v_(L)} extracted from multiple encoder layers, the training loss is defined as:

$L( \theta ) = E _ { ( x , V _ { hier } ) \in X } \; E _ { r \sim \Lambda } \left[ L _ { CE } \left( \Gamma _ { \theta } ( x , V _ { hier } , r ) , x ^ { \prime } \right) \right],$

where θ denotes the trainable parameters, and Γ_θ(x, V_hier, r) represents the model output at the r-th recurrence step. The hierarchical features V_hier are selectively injected into the early layers of the transformer during each recurrent pass. This ensures that the model progressively refines its latent representations by anchoring them to multi-scale visual cues. The recurrence depth r is sampled from a log-normal Poisson distribution Λ with a targeted mean r_bar+1. This stochastic supervision forces the model to maintain semantic consistency across varied computational paths, facilitating the adaptive early-exit mechanism during inference.

> 💡 **公式批读 — Training Objective**:
> - **双层期望**：外层在数据分布上求期望，内层在 recurrency depth r 的分布上求期望
> - **CE Loss**: 标准的交叉熵，预测目标是 x'（ground truth continuation）
> - **Γ_θ(x, $V_hier$, r)**: 模型在第 r 步的输出——意味着在训练时模型的输出会受 recurrency depth 影响
> - **Log-Normal Poisson Distribution Λ**: 目标均值为 $r_bar$+1（$r_bar$ 是期望的额外迭代次数）。Log-Normal+Poisson 的组合确保：(1) 正数采样值；(2) 均值可控；(3) 方差适度
> - **Stochastic Supervision 的核心作用**: 因为训练时 r 是随机的，模型必须学会在任何 depth 下都输出正确结果。这迫使模型将"推理能力"与"推理步数"解耦，天然适配推理时的自适应早停

---

## 三、Summary

- **3.1 Recurrent V-L Backbone**: Huginn 三元结构扩展——文本嵌入 E(x) + 视觉嵌入 ViT+Proj → concat → 随机初始化 s_0 → R-Block 迭代
- **3.2 Hierarchical Visual Injection**: 核心创新——从 ViT {6,12,18,24} 提取多尺度特征 → Patch Merger 对齐 → 前 K 次迭代按课程顺序注入 → R 不足时降采样
- **3.3 Training Objective**: Poisson 分布采样深度 + CE Loss → 解耦推理深度与视觉融合 → 支持自适应早停
- **关键设计原则**:
  - 视觉信息是"引导"而非"驱动"——前几步提供多尺度感知信号，后续纯隐空间推理
  - 随机训练深度是自适应推理的前提——如果训练时只用固定深度，推理时无法灵活调整
  - 层级顺序是从细节到语义（bottom-up attention），符合认知规律
