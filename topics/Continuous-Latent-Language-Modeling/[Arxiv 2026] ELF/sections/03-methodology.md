[← 返回 README](../README.md)

# 3 Embedded Language Flows

> 📌 **Preview**: Core methodology covering (3.1) the ELF framework — embedding encoding, Flow Matching with x-prediction, final-step discretization with shared-weight network; (3.2) training and inference pseudocode; (3.3) self-conditioning, training-time CFG, and extension to conditional generation.

In this section, we present our flow-based formulation for language modeling (Fig. 3). Our method leverages the iterative nature of flow models to perform denoising primarily in continuous embedding space, converting clean embeddings back to discrete tokens only at the final step. Following prior work [56, 57, 30, 10], we describe our method in the simpler setting of unconditional generation. The framework can be extended to conditional generation, as discussed in Sec. 3.3.

## 3.1 The ELF Framework

### From Discrete Tokens to Continuous Embeddings

From discrete tokens to continuous embeddings. To apply continuous diffusion to language, we first map discrete tokens to continuous representations. Given a sentence, we tokenize it into a sequence of tokens **s** = [s_1, ..., s_L] in V^L, where each s_i is drawn from the vocabulary V and L denotes the sequence length. We then map the discrete token sequence into a continuous embedding space. The choice of the embedding method is flexible. By default, we use a pretrained T5 encoder [53] for bidirectional contextual embeddings. We also explore other jointly trained and randomized embeddings (see Sec. 4.1). The encoder is only used during training, which does not incur additional modules at inference.

> 💡 **机制拆解**: 嵌入方法的选择是一个关键设计空间。ELF默认使用frozen pretrained T5 encoder（35M参数），产生512维的上下文嵌入。这与其他方法的区别：
> - Diffusion-LM等：使用联合训练的可学习token embedding（非上下文）
> - LD4LG等：使用frozen encoder但需要单独训练decoder
> - FLM：使用one-hot编码（维度=词汇表大小，极其高维）
> - LangFlow：使用可学习的嵌入层（联合训练）
> ELF的frozen encoder方案的优势：(1)不需要训练encoder/decoder；(2)上下文嵌入提供更丰富的语言表示；(3)encoder只在训练时使用，推理时不需要。

### Flow Matching on Continuous Embeddings

Flow Matching on continuous embeddings. After obtaining continuous language representations, we formulate the denoising process in the resulting embedding space using Flow Matching [37, 38, 3]. Flow Matching defines a continuous flow path from noise to data in this space. Let **x** ~ p_data(x) denote the embedding distribution and ε ~ p_noise(ε) denote the noise distribution (e.g., ε ~ N(0, I)). The noisy latent variable is defined by linear interpolation ("rectified flows"): z_t = t x + (1 - t) ε, where t in [0, 1], and z_0 ~ p_noise and z_1 ~ p_data. In continuous time, the flow velocity **v** is defined as the time derivative of z, that is, **v** = dz/dt = **x** - ε.

While standard Flow Matching directly parameterizes **v** via a neural network, ELF follows recent advances on image generation and instead parameterizes **x** (x-prediction). Specifically, let x_θ = net_θ(z_t, t) denote the network's immediate output. We train the model by minimizing the mean squared error (MSE) between the predicted velocity and the target velocity:

$$ \mathcal{L}_{\mathrm{MSE}} = \mathbb{E}_{t, x, \epsilon} \| v_{\theta}(z_t, t) - v \|^2 = \mathbb{E}_{t, x, \epsilon} \frac{1}{(1-t)^2} \| x_{\theta}(z_t, t) - x \|^2, $$

where we leverage the relation v(z_t, t) = (x - z_t) / (1 - t) [32].

> 💡 **公式批读**: L_MSE的两个等价形式揭示了x-prediction和v-prediction之间的数学关系：(1)v-prediction的MSE loss等价于x-prediction的MSE除以(1-t)^2；(2)当t接近1时，(1-t)^2 → 0，权重1/(1-t)^2 → ∞。这意味着在t接近1(数据端)时，loss对预测误差极度敏感。这解释了为什么x-prediction比v-prediction更稳定——因为网络直接预测x(数值范围有限)，而非v(在t→1时可能被放大)。

The x-prediction parameterization is important for ELF. First, it enables Flow Matching to perform effectively on high-dimensional representations (e.g., 768-d per-token embeddings), consistent with observations in [32] (see Appendix C.1 for ELF's ablations on prediction targets). Second, predicting clean embeddings (i.e., **x**) aligns naturally with the objective of predicting clean discrete tokens at the final step (discussed next), whereas the standard **v**-prediction in Flow Matching does not. Although **v** can be predicted by a network and transformed into **x**, the weight sharing that ties the denoising (MSE loss) and decoding (cross-entropy loss) objectives is compromised. Empirically, we observe that **v**-prediction works poorly when weights are shared with the final discretization step.

> 💡 **机制拆解**: x-prediction是ELF设计中最重要的技术决策之一，原因有三：(1)高维嵌入空间中，干净数据x位于低维流形上，直接预测x比预测v或ε更稳定(见Fig. 10)；(2)共享权重的denoiser和decoder都需要预测干净嵌入x——denoiser用x̂计算v̂（用于MSE loss），decoder用x̂通过unembedding矩阵W得到logits（用于CE loss）；(3)如果用v-prediction，需要额外转换v̂→x̂→logits，权重共享的意义被削弱，因为网络学习的是不同的表示。实验证明v-prediction+共享权重的效果很差。

### Back to Discrete Tokens

Back to discrete tokens. As the generation output consists of discrete tokens, we convert the clean embeddings back into tokens at the final time step (i.e., at t = 1). By considering the final time step of ELF naturally as continuous-to-discrete decoding, our method does not require a separate decoder (or equivalently, it can be thought of as a decoder sharing weights with the denoiser).

The network input at this time step should be z_t in the limit t → 1. But because z_t → x as t → 1, we introduce a token-level corruption process at this final step to create a nontrivial training input, denoted as z̃ (detailed in Appendix B.1). The same network net_θ maps z̃ to a clean embedding x_θ(z̃), which is subsequently projected by a learnable "unembedding" matrix W to obtain logits. We minimize a per-token cross-entropy (CE) loss against the ground-truth token **s**:

$$ \mathcal{L}_{\mathrm{CE}} = \mathbb{E}_{\tilde{z}} \left[ \mathrm{CrossEnt}(W \mathbf{x}_{\theta}(\tilde{z}), s) \right], $$

The network x_θ shares weights with that in Eq. (1) and is conditioned on a binary "mode" token (denoise or decode) in addition to the time condition t = 1. At inference time, we evaluate W x_θ(z_t) only at the final step t = 1, and apply argmax to obtain a discrete token.

> 💡 **机制拆解**: Decode branch训练时的关键设计——per-token corruption level。由于t=1时z_t → x(干净嵌入)，如果不加corruption，decoder的输入就是x本身，训练信号太简单。ELF的解决方案是：对每个token独立采样corruption level p ~ logit-normal(P_mean=0.8, P_std=0.8)，构造z̃ = p*x + (1-p)*ε。这样：(1)同一序列中不同token的噪声程度不同，decoder必须从上下文中推理被破坏token的原始值；(2)decoder学会处理不完美的嵌入(推理时denoiser的输出可能不完全等于x)；(3)noise scale设置不同(OWT用5，条件生成用1)，控制decoder对imperfection的容忍度。

## 3.2 Pseudocode

The core concepts of ELF are summarized in Alg. 1 and Alg. 2 (detailed in Appendix Fig. 9).

Training. As in standard Flow Matching, ELF employs a single network net_θ to model all time steps, conditioned on t. This includes the final time step t = 1, which uses different pre-processing (corruption) and post-processing (loss computation). For clarity, we illustrate this distinction using an explicit "if" branch in Alg. 1. In practice, samples from both branches are processed within a single batch, and masking is used to selectively apply the appropriate corruption and unembedding operations as well as the corresponding loss terms. The network is further conditioned on a binary "mode" token that indicates whether the operation is "denoise" or "decode".

Inference. During inference, ELF iteratively transforms noisy samples into clean embeddings. Starting from z_0 ~ N(0, I), ELF solves the ODE: dz_t/dt = v_θ(z_t, t), which is approximated with

**Algorithm 1 ELF: training.**
Two-branch computation is batched, adding no extra training cost.

```
x = encode(s)
if uniform(0, 1) < threshold:              # denoising branch
    t = sample_t()
    e = randn_like(x)
    z = t * x + (1 - t) * e
    v = x - e
    x_pred = net(z, t, mode="denoise")
    v_pred = (x_pred - z) / (1 - t)
    loss = mse_loss(v_pred, v)
else:                                       # decoding branch
    t = 1
    z = corrupt(x)
    x_pred = net(z, t=1, mode="decode")
    s_pred = unembed(x_pred)
    loss = ce_loss(s_pred, s)
```

**Algorithm 2 ELF: inference.**
We show ODE for simplicity. SDE sampler is also applicable.

```
# shape: shape of embedded sequences
# ts: sampling time schedule, from 0 to 1
z = randn(shape)
for i in range(len(ts) - 1):
    t = ts[i]
    dt = ts[i + 1] - ts[i]
    x_pred = net(z, t, mode="denoise")
    # convert x prediction to velocity
    v = (x_pred - z) / (1 - t)
    z = z + dt * v
# final step
h = net(z, t=1, mode="decode")
# unembedding
token_logits = unembed(h)
tokens = argmax(token_logits)
```

a numerical (e.g., Euler) solver. At the final time step t = 1, we apply the network under the "decode" mode and perform unembedding and discretization.

Besides the ODE formulation, our method also supports an SDE-inspired sampler. The underlying SDE associated with Flow Matching can be derived following [43], where the dynamics can be interpreted as injecting infinitesimal noise at each step. In practice, we adopt a simpler approximation to emulate this behavior: we inject small noise at each step while correspondingly shifting the time variable t toward the noise regime (detailed in Appendix, Alg. 6). For brevity, we refer to the resulting SDE-inspired sampler as the "SDE" variant, while noting that it primarily captures the per-step stochastic behavior. We experimentally compare the ODE formulation with this SDE variant.

> 💡 **机制拆解**: SDE sampler的核心公式(详见Alg. 6)：每步先re-inject噪声，将状态"拉回"噪声区间，然后重新去噪——α = 1 - γ*dt, t_back = α*t, z_back = α*z + (1-α)*ε。关键是：噪声注入的幅度由γ和dt共同控制(大步长注入更多噪声)。相比于ODE的确定性轨迹，SDE通过随机性"修正"早期去噪误差——如果denoiser在早期产生了不完美的x̂，ODE会沿着这个不完美轨迹继续放大误差，而SDE通过重新注入噪声给模型第二次机会。

## 3.3 Conditioning and Guidance

Controlling model generation is an important aspect of generative modeling. In image diffusion models, classifier-free guidance (CFG) [25] has been established as a highly effective technique for steering the generated output. CFG also enables a trade-off between generation quality and diversity. Because CFG was originally formulated for continuous quantities (e.g., score functions or velocity fields), it is naturally applicable to ELF. This stands in contrast to discrete counterparts, where CFG remains largely unexplored and has been shown less effective [30, 51].

In the absence of class labels, we employ self-conditioning [9] to construct the conditioning signals required for CFG. Given that self-conditioning is already a standard component in DLMs [79, 13, 66, 41, 44, 59, 60], incorporating CFG introduces only marginal computational overhead. In what follows, we first describe the self-conditioning used in ELF and then introduce CFG.

### Self-Conditioning

Self-conditioning. In a standard Flow Matching model (i.e., without self-conditioning), a forward pass at a given time step yields a single prediction. We denote this prediction by x̂' in our case, indicating that it corresponds to a prediction of the clean embedding **x**. During training, self-conditioning [9] performs a second forward pass, conditioned on x̂', which serves as an intermediate prediction. The output of the second pass, denoted as x̂, can be written as x̂ = net_θ(z_t | x̂', t). This is implemented by concatenating [z_t, x̂'] as the network input [9]. During training, the model is conditioned on x̂' with probability 50%, and uses a null condition 0 otherwise (see Appendix, Fig. 9 for details). During inference, the model conditions on the prediction from the previous time step, thus introducing no extra forward passes for inference.

The intermediate prediction x̂' serves as a condition for the network. As such, it can be treated as the conditioning signal c in the application of CFG, introduced next.

### CFG with Self-Conditioning

CFG with self-conditioning. CFG [25] combines the unconditional and conditional predictions through a linear extrapolation. Formally, given a conditioning signal c, CFG in Flow Matching defines a velocity field as v_cfg(z_t | c) = ω v(z_t | c) + (1 - ω) v(z_t | ∅), where ∅ denotes the unconditional counterpart and ω is the guidance scale. As discussed, our conditioning signal c is obtained from self-conditioning. In its original form [25], CFG is applied at inference time, requiring two forward passes per step.

To avoid inference-time overhead, we adopt training-time CFG techniques [8, 69, 16, 17] previously developed for image generation. These methods use a single network pass to model v_cfg instead of v (in our case, x_cfg instead of x). Because ELF is formulated similarly to its image-generation counterpart, adapting it to training-time CFG is straightforward, further illustrating the advantages of our continuous-based formulation. The implementation details, following the form in [16, 17], are in Appendix (Alg. 3, 4, & 5).

> 💡 **机制拆解**: Training-time CFG的核心思想：训练时就让网络直接学习v_cfg(组合后的速度场)，而非v(组合前的速度场)。回归目标变成：v_target = x - ε + (1-1/ω)(v_sc - v_no_sc)。关键点：(1)ω=1时退化为标准Flow Matching(无CFG)；(2)训练时每个样本随机采样ω∈[0.5,5.0]，偏向小值；(3)推理时只需一次前向传播，ω作为输入token传给网络。这与inference-time CFG (需要两次前向，一次条件一次无条件)相比，推理成本减半。

### Extension to Conditional Generation

Extension to conditional generation. Thus far, we have presented our method in the setting of unconditional generation, as in prior work [56, 57, 30, 10]. Our method can be naturally extended to conditional generation, in which outputs are conditioned on an input sequence (e.g., a prompt). In this setting, we prepend the clean embeddings of the conditioning sequence to the model input and preserve them without corruption during both training and inference. The model can then condition on them through self-attention.

CFG remains applicable in the conditional setting. The conditioning c now consists of both the self-conditioning and the prefix clean embeddings; the unconditional counterpart is obtained by zeroing out c. Analogous to text-to-image generation [14], CFG is effective in controlling generation quality in our scenario, which can be viewed as "text-to-text" generation.

> 💡 **机制拆解**: 条件生成的实现非常简洁——将source序列的干净嵌入(不经过噪声破坏)直接prepend到目标序列前，模型通过双向self-attention访问条件信息。CFG两个信号：(1)self-conditioning(自身前一步的预测)；(2)input-condition(source序列嵌入)。训练时以10%概率将source嵌入置零来学习无条件生成，推理时组合两者。

🔖 **Summary**: Section 3 establishes ELF's three-pillar design: (1) Flow Matching in continuous embedding space with x-prediction for stable high-dimensional denoising; (2) shared-weight denoiser-decoder network with final-step discretization only; (3) training-time CFG with self-conditioning for efficient quality-diversity control. The entire framework is modular and naturally extends to conditional generation.
