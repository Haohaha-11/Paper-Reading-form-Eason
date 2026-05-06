[← 返回 README](../README.md)

# 3. Method

## 📌 预览
本节是核心方法，重点看模块输入输出、训练目标、推理路径和与 baseline 的差异。

---

# 3.1. Preliminary: Implicit and Explicit Decoding

Given a multimodal input comprising a question $\mathcal { Q }$ and an image $\nu$ , we first tokenize them into a sequence of text embeddings $\mathbf { Q } = \{ \mathbf { q } _ { i } \} _ { i = 1 } ^ { N _ { q } }$ and visual features $\mathbf { V } = \{ \mathbf { v } _ { i } \} _ { i = 1 } ^ { N _ { v } }$ via a word embedding matrix $\mathbf { E } \in \mathbb { R } ^ { | \mathcal { W } | \times d }$ and a pretrained visual encoder, respectively, where $| \mathcal { W } |$ denotes the vocabulary size and $d$ is the hidden dimension. Subsequently, we employ an autoregressive MLLM $\mathcal { F } _ { \theta }$ to encode the concatenated input into an initial hidden state ${ \bf H } ^ { ( 0 ) } \in \mathbb { R } ^ { P \times d }$ , where $P = N _ { q } + N _ { v }$ represents the total number of tokens in the prefilling phase. During the decoding phase, we predetermine the number of implicit reasoning steps $T _ { r }$ and explicit answer tokens $T _ { a }$ . The generation process consists of two following distinct stages.

> 💡 **预备设定**: 这里先把输入拆成问题 token $\mathbf Q$ 和图像 visual tokens $\mathbf V$，再拼成 prefilling 序列。后文所有 replay 和 routing 都发生在这串多模态 token 的 hidden states 上，所以先记住 $P=N_q+N_v$ 是初始上下文长度。

Implicit Reasoning Phase. In the $t$ -th implicit reasoning step, the model generates a continuous latent representation $\mathbf { z } _ { t }$ conditioned on the original input sequence $\mathbf { X } = \left[ \mathbf { Q } \parallel \mathbf { V } \right]$ and all preceding latent states. This iterative process is formulated as:

> 💡 **隐式推理阶段**: 显式 CoT 每一步输出文本；本文每一步输出连续 latent $\mathbf z_t$。这让推理步骤仍然存在，但不再消耗大量自回归文本 token，也减少“长文本链条越写越脱离图像”的风险。

![Equation 1](../images/c15c8a7dbd41f40455bedb7178d8167bf503c615bc227df5b6f36ee823d65f40.jpg)
*Equation 1: Equation extracted by MinerU.*

> 💡 **Equation 1 批读**: Eq. 1 定义 latent step 的递推：模型 $\mathcal F_\theta$ 读取原始多模态输入 $\mathbf X$ 和已有 latents $\mathbf Z_{<t}$，再用 $\tau(\cdot)$ 取最后位置 hidden state 作为新的 $\mathbf z_t$。这相当于把一个 reasoning step 写进 hidden vector。

where $\mathbf Z _ { < t } = \{ \mathbf z ^ { ( 1 ) } , \dots , \mathbf z ^ { ( t - 1 ) } \}$ denotes the sequence of latent tokens generated in previous steps, and $\tau ( \cdot )$ extracts the final vector representation from the model output (i.e., the hidden state corresponding to the last generated token). Unlike vanilla explicit reasoning, each step yields a informative continuous latent vector rather than a discrete token. The resulting latent sequence $\mathbf { Z } _ { 1 : T _ { r } } = \{ \mathbf { z } ^ { ( t ) } \} _ { t = 1 } ^ { T _ { r } }$ is concatenated to the context before transitioning to the explicit answer decoding phase.

> 💡 **latent 序列作用**: $\mathbf Z_{1:T_r}$ 会被拼回上下文，再进入答案解码。也就是说 latent 不是旁路特征，而是和原始图文 token 一起成为后续 answer decoding 的条件。

Explicit Answer Decoding. In the explicit phase, the model generates discrete answer tokens by sampling from the vocabulary distribution. The conditional probability of the $t$ -th answer token $a _ { t }$ is given by:

> 💡 **显式答案阶段**: 隐式推理完成后，模型仍按普通语言模型生成最终答案 token。区别在于答案前的“思考过程”主要由 $\mathbf Z$ 承担，而不是输出成可见 CoT。

![Equation 2](../images/0708040c17ebd8db627ccf41eeb42f3cecc1d136d01ffa072426e7a84b5df021.jpg)
*Equation 2: Equation extracted by MinerU.*

> 💡 **Equation 2 批读**: Eq. 2 是标准 LM head：decoder hidden state $\mathbf h_t^{dec}$ 经 $W_o$ 投到词表 logits，再 softmax 得到下一个 answer token 分布。latent reasoning 没有改变最终答案的生成形式。

where $\mathbf { h } _ { t } ^ { \mathrm { d e c } }$ is the decoder hidden state at step $t$ , and ${ \bf W } _ { o } \in $ $\mathbb { R } ^ { | \mathcal { W } | \times d }$ is the output projection matrix that maps hidden states to vocabulary logits.

> 💡 **符号补充**: 这里的 $W_o$ 是输出投影矩阵，不是视觉模块。视觉增强发生在前面的 hidden/latent 构造阶段，最终仍由语言头读出答案。

The likelihood of the complete answer sequence $\mathbf { a } _ { 1 : T _ { a } }$ is factorized as an autoregressive product:

![Equation 3](../images/1c1c93a314999a51efd277fc242211e5eb27d69e55bf2405812d34798f9f3da3.jpg)
*Equation 3: Equation extracted by MinerU.*

> 💡 **Equation 3 批读**: Eq. 3 把完整答案 likelihood 分解为自回归乘积。关键条件是 $\mathbf X$ 和 $\mathbf Z_{1:T_r}$ 一起进入每个 token 的上下文，因此 latent 序列承担了压缩 CoT 的角色。

Here, the full context input for the decoder is defined as $\mathbf { U } _ { t } = \left[ \mathbf { X } \parallel \mathbf { Z } _ { 1 : T _ { r } } \parallel \mathbf { a } _ { < t } \right]$ , where $\parallel$ denotes sequence concatenation along the token dimension.

> 💡 **数据流节点**: $\mathbf U_t=[\mathbf X\parallel\mathbf Z_{1:T_r}\parallel\mathbf a_{<t}]$ 是推理到答案的汇合点：原图文输入、隐式 reasoning latents 和已生成答案历史共同决定下一个 token。

# 3.2. Spatially-Coherent Finer Visual Replay

As mentioned earlier, we empirically reveal the gradient disparities between visual and textual tokens throughout the learning dynamics. Specifically, visual tokens consistently exhibit substantially larger gradient norms and fluctuations compared to textual tokens, indicating that visual representations remain under-optimized despite their critical role in multimodal reasoning. Motivated by these insights, we introduce the visual replay module to reinforce the engagement of visual cues via salient region detection, while enhancing fine-grained spatially-coherent perception capabilities via self-distillation supervision at each reasoning step.

> 💡 **SCF-VR 动机**: Visual replay 针对 Figure 1 的视觉 token 梯度不稳。思路不是简单增加视觉 token 数量，而是在每个 latent step 重新找当前推理最需要的视觉区域，并用自蒸馏让这些 replay tokens 保持细节。

Attention-Guided Region Focus. Several works [33, 76] have demonstrated that LLMs exhibit fundamental visual grounding capabilities. To identify visually salient regions, we aggregate attention weights across all transformer layers and attention heads to obtain a consolidated spatial focus map. Specifically, given the $l$ -th layer and the $h$ -th attention head, we compute the mean attention map $\bar { \mathbf { A } } ^ { ( t ) }$ at reasoning step $t$ :

> 💡 **区域选择**: 作者用模型内部 causal attention 找 salient visual regions，避免额外区域标注。这里的假设是：最近生成的 latent/hidden token 对历史视觉 token 的注意力，可以近似反映当前推理需要看哪里。

![Equation 4](../images/27322e749e4c48792a7b1f0c5a29e1e7dadf7f5ba1fc2e63054c8fc4632dd474.jpg)
*Equation 4: Equation extracted by MinerU.*

> 💡 **Equation 4 批读**: Eq. 4 对所有层和 attention head 的矩阵求平均，得到 reasoning step $t$ 的整体注意力图 $\bar{\mathbf A}^{(t)}$。这一步把局部 head 的噪声平滑成一个全模型的空间关注估计。

where A(l,h,t) ∈ RP (t)×P (t) denotes the attention matrix for layer $l$ and head $h$ at iteration $t$ $\begin{array} { r } { 1 \leq t \leq T _ { r } } \end{array}$ ), with $P ^ { ( t ) }$ representing the number of input tokens at iteration $t$ . Here, $L$ and $H$ represent the total number of layers and heads, respectively. To obtain token-level attention scores, we extract the attention distribution from the most recently generated token to all preceding tokens via column-wise summation, i.e., a(t)all = colsum(A¯ (t)) ∈ RP (t) . Subsequently, we extract only the visual token attention scores from $\mathbf { a } _ { a l l } ^ { ( t ) }$ using the image mask, denoted as $\mathbf { a } ^ { ( t ) } \in \mathbb { R } ^ { N _ { v } }$ , where $N _ { v }$ is the number of visual tokens. This visual attention vector effectively captures which visual tokens are most relevant to the current reasoning context.

> 💡 **visual score 提取**: 作者只取“最新 token 指向历史 token”的注意力，再用 image mask 抽出 visual token 分数。这个设计让 replay 跟当前 reasoning step 绑定，而不是静态选择全局最显眼区域。

As visual focus evolves across reasoning steps, we iteratively select the top- $K$ attended visual tokens $\{ \mathbf { v } _ { i } ^ { ( t ) } \} _ { i = 1 } ^ { K }$ as visual latents, which are integrated with hidden states $\mathbf { z } _ { t }$ . To prevent redundant re-selection and promote diverse exploration, we maintain a visited token set V (t)visite d, ensuring comprehensive visual coverage:

> 💡 **去重机制**: visited token set 防止每一步反复选同一批 patch，有利于覆盖更多相关证据。但也有风险：如果模型第一步误选，后续排除机制可能让真正关键区域被延后甚至错过。

![Equation 5](../images/071216ad9b39ab0c49e7d492ef650708de87d8838831640e0addd50c4125df5b.jpg)
*Equation 5: Equation extracted by MinerU.*

> 💡 **Equation 5 批读**: Eq. 5 定义 top-$K$ visual token 的索引集合 $\mathcal T^{(t)}$，并把已访问 token 排除。它是 replay 的离散选择步骤，决定哪些图像 patch 会被重新注入 latent 推理。

where $\boldsymbol { \mathcal { T } ^ { ( t ) } }$ represents the indices of the $K$ visual tokens with the highest attention scores at step t, and V (t)visite d is updated after each selection. The original embeddings of the selected tokens $\mathbf { V } _ { \mathcal { T } ^ { ( t ) } } = \{ \mathbf { v } _ { i } \ | \ i \in \mathcal { T } ^ { ( t ) } \}$ are further weighted by normalized attention scores:

> 💡 **加权 visual latents**: 选中的原始 visual embeddings 再按归一化 attention score 加权。这样 replay token 不只是 top-k 集合，还携带“当前 step 更看重哪个 patch”的强弱信息。

![Equation 6](../images/bc557e70a2a327e90d609893c05fc925023dc3a99f848cb27881876e4aabaaf1.jpg)
*Equation 6: Equation extracted by MinerU.*

> 💡 **Equation 6 批读**: Eq. 6 用对角矩阵把 attention 权重乘到 selected visual embeddings 上，得到 $\mathbf B^{(t)}$。这个 $\mathbf B^{(t)}$ 就是后续与 $\mathbf z_t$ 交织传播的 visual replay 表示。

where $\mathrm { D i a g ( \cdot ) }$ constructs a diagonal matrix from a vector.

Spatially-Coherent Regularization. Although leveraging learned attention within Transformers provides explainable visual locations, it often suffers from limited spatial continuity due to the scattered nature of selected tokens and introduces noise associated with the attention sink phenomenon [65]. To mitigate these issues and enhance finegrained perception without external annotations, we introduce self-distillation supervision. This mechanism involves cropping the visual regions exhibiting spatial coherence, reencoding them, and supervising the visual latents with these high-fidelity features.

> 💡 **空间一致正则**: 只按 attention 选 patch 容易得到分散 token，并受 attention sink 干扰。作者因此加入 self-distillation：从原图裁出空间连续窗口、重新编码为高保真 reference，用它监督 replay latent。

Specifically, we first search for a $W \times W$ sub-grid patch that maximizes the density of attended visual tokens. Formally, we find the optimal top-left corner $( r ^ { * } , c ^ { * } )$ within the valid grid bounds:

> 💡 **crop 选择**: $W\times W$ sub-grid 不是随便裁，而是找包含最多 attended tokens 的连续区域。这样能把“点状注意力”转成“局部证据块”，更适合保留物体部件、文字、病灶这类局部结构。

![Equation 7](../images/e64b5669f5fd965070d2bfbd2e0ed38c391b5fc1ab89b2c7181979fb41bc91b0.jpg)
*Equation 7: Equation extracted by MinerU.*

> 💡 **Equation 7 批读**: Eq. 7 是密度最大化：在视觉 token 网格上枚举窗口左上角 $(r,c)$，选择 attended token 数量最多的窗口。它把 replay 的训练目标从 scattered patches 收敛到一个空间连续 crop。

![Figure 2.](../images/b9a52a98a28fabf6368e5ab3db96524beb8ac11db360df696dca730e523b14a9.jpg)
*Figure 2.: Figure 2. Left Panel: Schematic illustration of our framework. Input images are encoded into visual tokens by a pretrained visual encoder and projected into a text-centric semantic space aligned with the LLM, while questions are tokenized by the text tokenizer. Our method enhances a standard latent MLLM with two synergistic components during the implicit reasoning phase: (1) Spatially-Coherent Finer Visual Replay (SCF-VR): Attention-guided selection identifies salient visual regions at each implicit step, generating fine-grained visual latents enhanced by spatially-coherent constraints. (2) Routing Depth Scaling (RDS): A lightweight learnable router adaptively allocates additional reasoning steps for high-difficulty tokens or latents. Refined hidden states and visual latents are interleaved and propagated to subsequent reasoning steps. The overall objective is jointly optimized via standard cross-entropy loss and self-distillation loss. Right Panel: Detailed architecture of the SCF-VR module and token-wise depth scaling mechanism. TokenSR denotes the token super-resolution module, and Pool is the average pooling.*

> 💡 **Figure 2 批读**: 左图给总数据流：图像/问题输入后，在 implicit reasoning 阶段插入 SCF-VR 和 RDS；右图展开两模块细节。注意 TokenSR 和 self-distillation 是训练期约束，RDS 是 layer 内 token 选择性重算。

where the density function $\mathcal { N } ( r , c )$ counts the visited tokens falling within the window $\begin{array} { r } { \mathcal { N } ( r , c ) = \sum _ { i \in \mathcal { T } ^ { ( t ) } } \mathbb { I } \big [ r \leq r _ { i } < } \end{array}$ $r + W , c \leq c _ { i } < c + W \big ]$ . Here, $( r _ { i } , c _ { i } )$ denotes the rowcolumn position of token $i$ on the $G \times G$ grid, and $\mathbb { I } [ \cdot ]$ is the indicator function. This greedy selection ensures the cropped region captures a coherent visual context rather than scattered details. Then, the selected window is projected to pixel coordinates in the original image via mathematical transformation, cropped, and resized to the standard encoder input resolution using bilinear interpolation. We then reencode this refined patch through the same visual encoder to obtain fine-grained representations $\{ \mathbf { f } _ { i } \} _ { i = 1 } ^ { N _ { v } ^ { \prime } }$ . A global pooling operation $\operatorname { P o o l } ( { \mathord { \cdot } } )$ is applied to yield a robust reference token uref:

> 💡 **reference 构造**: 选出窗口后映射回原图像素、裁剪、resize，再经过同一个视觉 encoder 得到细粒度特征。这相当于让局部高分辨视觉证据给 replay latent 当 teacher。

![Equation 8](../images/22601ebbf93e51350c9ccb21ae09d33c628642cf67c916cf3527a6517238601a.jpg)
*Equation 8: Equation extracted by MinerU.*

> 💡 **Equation 8 批读**: Eq. 8 对重编码 patch features 做 pooling 得到 $\mathbf u_{ref}$。这个 reference token 是空间连续 crop 的压缩表示，用来监督 coarse replay token 是否保留了局部视觉语义。

Finally, we align the coarse-grained global token $\mathbf { b } ^ { ( t ) } =$ $\mathrm { P o o l } ( \mathbf { B } ^ { ( t ) } )$ with this high-fidelity reference using a lightweight token super-resolution module $\mathcal { F } _ { \mathrm { S R } } : \mathbb { R } ^ { D } \to \mathbb { R } ^ { D }$ We minimize the reconstruction error as follows:

> 💡 **TokenSR 作用**: $\mathcal F_{SR}$ 不是图像超分网络，而是一个 feature-space mapping，用粗 replay latent 预测高保真 reference。名字容易误导，实际作用是提升 token 表示的细粒度视觉信息。

![Equation 9](../images/2913b7fb28dda5128bfd4df2a4efbc57c62b4a4468b3713efe0188f2fd23badd.jpg)
*Equation 9: Equation extracted by MinerU.*

> 💡 **Equation 9 批读**: Eq. 9 是 self-distillation reconstruction loss，约束 $\mathcal F_{SR}(\mathbf b^{(t)})$ 接近 $\mathbf u_{ref}$。它不需要外部区域标注，但依赖模型 attention 能大致定位相关区域。

This supervisory signal enables the model to prioritize spatially coherent and semantically intact visual contexts during latent generation through self-distillation.

> 💡 **监督信号意义**: 这个 loss 让 latent generation 更偏向空间连续、语义完整的视觉上下文。对医学图像可复用的点是：用局部 crop 重编码约束 hidden evidence，而不是只靠答案监督。

visual latents, existing methods typically allocate uniform computational budgets across all tokens during contextual refinement. Our empirical analysis reveals a fixed-depth optimization dilemma, demonstrating that token representations exhibit heterogeneous optimization complexities during latent training, which necessitates adaptive reasoning depths. To address this limitation without modifying the pretrained VLM architecture, while effectively leveraging its inherent knowledge, we introduce a lightweight router that dynamically allocates additional reasoning steps exclusively to critical tokens at each iteration. Router Network. Let the input token sequence at the $t { \cdot }$ -th reasoning step be denoted as ${ \bf U } ^ { ( t ) } = [ { \bf X } \| { \bf B } ^ { ( 1 ) } \| { \bf z } ^ { ( 1 ) } \| \cdot \cdot \cdot \| { \bf B } ^ { ( t ) } \| { \bf z } ^ { ( t ) } ] \in \mathbb { R } ^ { P ^ { ( t ) } \times d }$ , where $P ^ { ( t ) }$ represents the total sequence length and $d$ is the hidden dimension. To facilitate the illustration of our depth scaling mechanism, we decompose the forward pass into layer-wise computations. Specifically, within the $l$ -th transformer layer, we first compute the intermediate feature $\mathbf { H } ^ { ( l , t ) } = f ( \mathbf { U } ^ { ( t ) } ) \in \mathbb { R } ^ { P ^ { ( t ) } \times d }$ , where $f ( \cdot )$ denotes the transformation of the $l$ -th transformer layer in LLM. Subsequently, a lightweight router network computes a scalar importance score for each token based on its corresponding hidden representation:

> 💡 **RDS 入口**: 这里开始从“补视觉证据”转向“给复杂 token 更多深度”。核心问题是 uniform computation budget 不合理：简单 token 早已收敛，复杂视觉/latent token 仍需要上下文 refinement。

While the visual replay mechanism significantly enhances fine-grained grounding by introducing spatially-coherent

> 💡 **router 输入**: $\mathbf U^{(t)}$ 包含原始图文 token、此前 replay 的 visual latents 和 reasoning latents。router 因此可以基于当前完整上下文判断哪个 token 值得额外计算，而不是只看原始图像 patch。

![Equation 10](../images/1e628710764009bb713787da12b93eb1643520607f527a90061b69d9d047adff.jpg)
*Equation 10: Equation extracted by MinerU.*

> 💡 **Equation 10 批读**: Eq. 10 用每层独立 router $\mathcal F_{router}^{(l)}$ 给 token hidden representation 输出重要性分数 $s_i^{(l,t)}$。这是 RDS 的 gating 信号，后续只让 top-$\alpha$ token 进入额外 refinement。

# 3.3. Routing Depth Scaling

where the $i$ -th element $s _ { i } ^ { ( l , t ) }$ quantifies the importance of the i-th token in the l-th layer, and F (l)route $\mathcal { F } _ { \mathrm { r o u t e r } } ^ { ( l ) } ( \cdot )$ denotes the independent router network in the $l$ -th layer. We define $T _ { \alpha } ( \mathbf { s } ^ { ( l , t ) } )$

as the index set of the top- $\alpha$ tokens with the highest scores, where $\alpha$ serves as a predefined hyper-parameter. In practice, $\alpha$ can be formulated as a layer-dependent function $\alpha ( l )$ to enable adaptive resource allocation at different layers.

> 💡 **top-$\alpha$ 选择**: $\alpha$ 是计算预算超参，决定每层有多少 token 继续“深想”。作者后面消融显示 $\alpha=32$ 最稳，说明预算太小会丢上下文，太大收益有限且可能引入无关 token。

Depth Scaling Computation. Obtaining the router weight, we select the top- $\alpha$ tokens for depth scaling, i.e., repeat iteration in one transformer block. For a given depth scaling step $d \in \{ 1 , \ldots , D \}$ , the token-wise representation update rule within a transformer layer can be generally formulated as follows:

![Equation 11](../images/298844f0824d4459ec2a84ed163843f0cd2f2b415acf0f5e11c10de043d56f7f.jpg)
*Equation 11: Equation extracted by MinerU.*

> 💡 **Equation 11 批读**: Eq. 11 描述额外 depth step 的 token update。只有满足 $i\in T_\alpha(\mathbf s^{(d)})$ 的 token 进入 refinement，其余 token 保持原表示，这就是 token-wise depth scaling 的计算节省来源。

For notational brevity, we omit the layer index $l$ and reasoning step $t$ in the following formulation, where $d$ denotes the refinement depth. For instance, the score vector $\mathbf { s } ^ { ( l , t ) }$ is simplified to $\mathbf { s } ^ { ( d ) }$ . The function $f _ { i \in T _ { \alpha } ( \mathbf { s } ^ { ( d ) } ) } ( \cdot )$ indicates that the attention operation is restricted to the token subset $T _ { \alpha } ( \mathbf { s } ^ { ( d ) } )$ Here, ${ \bf h } _ { i } ^ { ( 0 ) }$ represents the initial hidden state of the $i$ -th token after the first forward pass, and m(d) ∈ RP (t)×P (t) denotes the attention mask corresponding to refinement step $d$ . The condition $i \in T _ { \alpha } ( \mathbf { s } ^ { ( d ) } )$ functions as a binary gating mechanism, ensuring that only tokens with importance scores exceeding the threshold undergo additional computation.

> 💡 **mask 与 gating**: 这里的 attention mask $m^{(d)}$ 和 top-$\alpha$ 条件共同限制额外计算范围。读法上抓住一句话：不是整层重复跑所有 token，而是对关键 token 做受控重算。

The router network dynamically determines whether to apply depth scaling based on the current contextual representations in a data-driven manner. Tokens identified as critical are iteratively processed through the transformation function for $d$ additional refinement steps, while non-critical tokens retain their prior representations to preserve computational efficiency. Finally, we aggregate the depth-wise refined representation with step-aware positional encoding to form the final hidden state incrementally:

![Equation 12](../images/9e95891a06271606b493993c16b3472658fe10ae26875be9c6ecefa912c0896a.jpg)
*Equation 12: Equation extracted by MinerU.*

> 💡 **Equation 12 批读**: Eq. 12 把不同 refinement depth 的结果与 step-aware positional encoding 聚合，得到最终 hidden state。保留位置信息很重要，后面 position encoding 消融显示重排位置会破坏跨 step 的结构一致性。

This adaptive scaling strategy effectively allocates deeper reasoning pathways to critical tokens, enabling the model to capture complex visual contexts and facilitate more profound reasoning capabilities.

> 💡 **RDS 总结**: RDS 让模型对复杂视觉上下文有更深处理路径，但仍限制在关键 token 上。它适合小目标、局部关系、多对象组合这类“不是所有 token 都难，但某些 token 非常关键”的场景。

# 3.4. Training Procedure

Curriculum Latent Training. To mitigate annotation overhead and avoid the risk of human priors constraining model learning, we depart from prior approaches that rely on intermediate supervision for latent representations. Instead, we design a curriculum that facilitates the contextual and logical dependencies between implicit latents and explicit reasoning chains. In the initial stage, the model is trained with standard Chain-of-Thought (CoT) supervision, generating all reasoning steps explicitly to establish foundational reasoning capabilities. Subsequently, as training progresses, latent tokens are incrementally introduced. Specifically, one explicit reasoning step is progressively encapsulated into an informative ⟨latent⟩ token. Through this curriculum, each latent token progressively learns to ground the relevant contextual cues, effectively internalizing explicit reasoning chains into compact latent representations.

> 💡 **curriculum 动机**: 直接训练 latent token 可能没有稳定语义，完全依赖显式 CoT 又慢。课程学习先让模型掌握显式步骤，再逐步把步骤压缩成 `<latent>`，相当于从可见推理过渡到隐式推理。

Training Objective. The overall training objective combines the standard language modeling loss with the self-distillation loss in the VR-SCF module,

> 💡 **训练目标概览**: 总目标由答案/文本的 CE loss 与 SCF-VR 的自蒸馏 loss 组成。CE 保证最终任务正确，$\mathcal L_{SCF}$ 约束 latent 推理过程中保留视觉细节。

![Equation 13](../images/dc8350622edd27670032fd6f475d80db3c740bb56fbb55cd20a161561060d99b.jpg)
*Equation 13: Equation extracted by MinerU.*

> 💡 **Equation 13 批读**: Eq. 13 是联合 loss：$\mathcal L=\mathcal L_{CE}+\lambda\mathcal L_{SCF}$。$\lambda$ 控制视觉自蒸馏对主任务训练的影响，过大可能让模型过度依赖局部视觉，过小则 replay 约束不足。

where $\mathcal { L } _ { \mathrm { C E } }$ is the standard cross-entropy loss over the language modeling task,

> 💡 **CE loss 角色**: $\mathcal L_{CE}$ 仍是最终答案 token 的标准监督，保证模型不会只学会定位/重放视觉区域而忘记完成语言答案生成。

![Equation 14](../images/08bd42b3c6459beb91593d5ce8ca9604a0822acd0648dcdda2469fbe436a3c4c.jpg)
*Equation 14: Equation extracted by MinerU.*

> 💡 **Equation 14 批读**: Eq. 14 展开 CE 的自回归形式，监督 answer sequence。结合 Eq. 13 看，本文没有给 latent token 单独人工标签，而是通过最终答案和视觉自蒸馏共同塑形 latent space。

where $\lambda$ is a hyperparameter balancing the primary task and the auxiliary reconstruction objective. During inference, only the LLM component is active, and the visual play module is disabled, ensuring no additional computational overhead at test time.

> 💡 **推理开销**: 作者强调推理时只启用 LLM component，visual play/replay module disabled，因此 SCF-VR 的额外 crop/重编码主要是训练期成本。部署时仍需确认 RDS 是否启用以及 eager attention 相关实现开销。

---

## 🔖 Section 总结

### 核心洞察
1. 基础数据流是：图像/问题 → visual/text tokens → latent reasoning tokens → explicit answer tokens。
2. SCF-VR 用 attention 选视觉 patch，再用空间连续 crop 的重编码特征做 self-distillation，目标是让 latent 推理保留局部视觉证据。
3. RDS 用 layer-wise router 只给 top-$\alpha$ 关键 token 追加 refinement，解决复杂 token 固定深度不足的问题。
4. Curriculum training 把显式 CoT 逐步内化成 `<latent>`，避免大量直接 latent supervision。

### 可追问点
- Attention 选出的 visual patch 是否真是因果证据，还是只是相关区域？
- 推理时关闭 visual replay 后，训练期学到的视觉保持能力能否在分布外图像上稳定保留？
- 医学场景中是否需要把 RDS 的 router 与病灶不确定性、风险等级或专家标注联动？
