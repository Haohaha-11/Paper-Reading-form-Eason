[← 返回 README](../README.md)

## 📌 预览
理论证明、context parallel algorithm、训练/评估/初始化细节。

---

# Appendix

# A Proof of theorem 1

For completeness, we first restate the theorem with the precise bounds derived from the assumptions.

Theorem 1 (Logit-wise Effect of LM-Aligned Target v.s. Reconstruction Target (Restated)). Under the specified setting and assumptions, for a learning rate $\lambda _ { l r } > 0$ , the expected change in logits $\Delta \ell _ { n }$ after one update step using the NTP-aligned target satisfies:

> 💡 **附录作用**: Appendix A 展开主文 theorem 的代数证明，核心仍是 embedding 近似正交下，next-token target 才能把正确 value 的 logit 往上推。

$$
\begin{array} { r l } { ( \mathsf { C o r r e c t : l o g i t i n c r e a s e s } ) } & { \mathbb { E } \left[ \Delta \ell _ { n } [ v ^ { * } ] \right] \geq \lambda _ { l r } \cdot c _ { n o r m } ^ { 2 } \cdot c _ { a l i g n } , } \\ { ( \mathsf { O t h e r \ l o g i t s \ c a l m o s t \ u n c h a n g e d } ) } & { \left| \mathbb { E } \left[ \Delta \ell _ { n } [ w ] \right] \right| \leq \lambda _ { l r } \cdot \epsilon \cdot c _ { a l i g n } , \quad \forall w \neq v ^ { * } . } \end{array}
$$

In contrast, for the reconstruction target, the expected change in logits is negligible for the correct token:

$$
\begin{array} { r } { | \mathbb { E } \left[ \Delta \ell _ { n } [ v ^ { * } ] \right] | \leq \lambda _ { l r } \cdot \epsilon \cdot c _ { a l i g n } . } \end{array}
$$

Proof. We begin from the setup defined in section 3.3. The change to the fast weights $\mathbf { W } _ { \mathrm { d o w n } }$ from all prior context tokens is given by $\begin{array} { r } { \Delta \mathbf { W } _ { \mathrm { d o w n } } = \lambda _ { \mathrm { l r } } \sum _ { t \in \mathrm { p r i o r } } \mathbf { v } _ { t } \mathbf { z } _ { t } ^ { \top } } \end{array}$ , where we use $\lambda _ { \mathrm { l r } }$ to denote the learning rate $\eta$ for consistency with the theorem statement. The resulting change in the logit for an arbitrary token $w$ at the query position $n$ is:

$$
\Delta \ell _ { n } [ w ] = \mathbf { e } _ { w } ^ { \top } ( \Delta \mathbf { W } _ { \mathrm { d o w n } } ) \mathbf { z } _ { n } = \lambda _ { \mathrm { l r } } \sum _ { t \in \mathrm { p r i o r } } \mathbf { e } _ { w } ^ { \top } ( { \mathbf { v } _ { t } } \mathbf { z } _ { t } ^ { \top } ) \mathbf { z } _ { n } .
$$

Since $\mathbf { e } _ { w } ^ { \top } \mathbf { v } _ { t }$ and ${ \bf z } _ { t } ^ { \top } { \bf z } _ { n }$ are scalars, we can rearrange the terms to get:

$$
\Delta \ell _ { n } [ w ] = \lambda _ { \mathrm { l r } } \sum _ { t \in \mathrm { p r i o r } } ( \mathbf { e } _ { w } ^ { \top } \mathbf { v } _ { t } ) ( \mathbf { z } _ { t } ^ { \top } \mathbf { z } _ { n } ) .
$$

To analyze the expected change, we take the expectation over the representations. Applying the linearity of expectation, we have:

$$
\mathbb { E } [ \Delta \ell _ { n } [ w ] ] = \lambda _ { \mathrm { l r } } \sum _ { t \in \mathrm { p r i o r } } \mathbb { E } [ ( { \bf e } _ { w } ^ { \top } { \bf v } _ { t } ) ( { \bf z } _ { t } ^ { \top } { \bf z } _ { n } ) ] .
$$

Per our setup, the target vectors $\mathbf { v } _ { t }$ (e.g., ${ \bf e } _ { x _ { t } }$ or $\mathbf { e } _ { x _ { t + 1 } }$ ) are treated as determined. Thus, we can factor them out of the expectation:

$$
\mathbb { E } [ \Delta \ell _ { n } [ w ] ] = \lambda _ { \mathrm { l r } } \sum _ { t \in \mathrm { p r i o r } } \mathbf { e } _ { w } ^ { \top } \mathbb { E } [ \mathbf { v } _ { t } \mathbf { z } _ { t } ^ { \top } \mathbf { z } _ { n } ] .
$$

Now, we invoke Assumption 2. It states that for the unique key position $t ^ { * }$ , we have $\mathbb { E } [ { \bf z } _ { t ^ { * } } ^ { \top } { \bf z } _ { n } ] = c _ { \mathrm { a l i g n } }$ , and for all other prior positions $t \neq t ^ { * }$ , the updates provide no information gain, which implies $\mathbb { E } [ { \bf v } _ { t } { \bf z } _ { t } ^ { \mathrm { ~ \tiny ~ | ~ } } { \bf z } _ { n } ] = { \bf 0 }$ . This simplifies the summation to a single term corresponding to the key-value pair $( k ^ { * } , v ^ { * } )$ at position $t ^ { * }$ :

$$
\mathbb { E } \left[ \Delta \boldsymbol { \ell } _ { n } [ w ] \right] = \lambda _ { \mathrm { l r } } \cdot \mathbb { E } \left[ \left( \mathbf { e } _ { w } ^ { \top } \mathbf { v } _ { t ^ { * } } \right) \cdot \left( \mathbf { z } _ { t ^ { * } } ^ { \top } \mathbf { z } _ { n } \right) \right] .
$$

We now analyze this simplified expression for the two target choices.

# Case 1: NTP-Aligned Target $\left\{ \mathbf { v } _ { t ^ { * } } = \mathbf { e } _ { x _ { t ^ { * } + 1 } } \right\}$

First, we consider the logit of the correct token, $w = v ^ { * }$ . Substituting the target and the token into equation (11) yields:

$$
\mathbb { E } \left[ \Delta \boldsymbol { \ell } _ { n } [ \boldsymbol { v } ^ { * } ] \right] = \lambda _ { \mathrm { l r } } \cdot \mathbb { E } \left[ ( \mathbf { e } _ { v ^ { * } } ^ { \top } \mathbf { e } _ { v ^ { * } } ) \cdot ( \mathbf { z } _ { t ^ { * } } ^ { \top } \mathbf { z } _ { n } ) \right] .
$$

By Assumption 1, token embeddings have a non-trivial magnitude, $\lVert \mathbf { e } _ { v ^ { \ast } } \rVert ^ { 2 } \geq c _ { \mathrm { n o r m } } ^ { 2 }$ . This gives us the lower bound in equation (4):

$$
\begin{array} { r } { \mathbb { E } \left[ \Delta \ell _ { n } [ v ^ { * } ] \right] \geq \lambda _ { \mathrm { l r } } \cdot c _ { \mathrm { a l i g n } } \cdot c _ { \mathrm { n o r m } } ^ { 2 } . } \end{array}
$$

Next, for any incorrect token $w \neq v ^ { * }$ , the expected change is $\mathbb { E } \left[ \Delta \boldsymbol { \ell } _ { n } [ w ] \right] = \mathbb { E } \left[ ( { \mathbf { e } } _ { w } ^ { \top } { \mathbf { e } } _ { v ^ { * } } ) \cdot ( { \mathbf { z } } _ { t ^ { * } } ^ { \top } { \mathbf { z } } _ { n } ) \right]$ . Taking the absolute value and applying Assumption 1, which states that distinct embeddings are nearly orthogonal $( | \mathbf { e } _ { w } ^ { \top } \mathbf { e } _ { v ^ { * } } | \leq \epsilon )$ , we obtain the bound in equation (5):

$$
| \mathbb { E } \left[ \Delta \ell _ { n } [ w ] \right] | \leq \lambda _ { \mathrm { l r } } \cdot c _ { \mathrm { a l i g n } } \cdot \epsilon .
$$

# Case 2: Reconstruction Target $\mathbf { \{ v }  _ { t ^ { * } } = \mathbf { e } _ { x _ { t ^ { * } } }$ )

Here, we analyze the effect on the correct logit $w = v ^ { * }$ . The expected change is:

$$
\mathbb { E } \left[ \Delta \boldsymbol { \ell } _ { n } [ v ^ { * } ] \right] = \mathbb { E } \left[ ( { \bf e } _ { k ^ { * } } ^ { \top } { \bf e } _ { v ^ { * } } ) \cdot ( { \bf z } _ { t ^ { * } } ^ { \top } { \bf z } _ { n } ) \right] .
$$

In an induction task, the key $k ^ { * }$ is distinct from the value $v ^ { * }$ . We again invoke Assumption 1 for these distinct tokens. Taking the absolute value gives the bound in equation (6):

$$
| \mathbb { E } \left[ \Delta \ell _ { n } [ v ^ { * } ] \right] | \leq \lambda _ { \mathrm { l r } } \cdot c _ { \mathrm { a l i g n } } \cdot \epsilon .
$$

This confirms that the reconstruction target has a negligible expected effect on the logit of the correct answer $v ^ { * }$ .

The results from these two cases establish the claims in theorem 1, providing a clear theoretical basis for the superiority of the NTP-aligned objective in the context of in-context learning. This completes the proof. $\boxed { \begin{array} { r l } \end{array} }$

# B Context Parallel Algorithm for In-Place TTT

For more clarity, we list the pseudocode of the context parallel implementation of our In-Place TTT here in algorithm 1.

> 💡 **算法细节**: Appendix B 的伪代码说明 CP 版本先并行算每段 delta，再做 cumulative sum，最后并行 apply；这是“严格因果但可并行”的实现基础。

# Algorithm 1 In-Place TTT with Context Parallelism (Single Layer)

<table><tr><td colspan="4">Require: Pre-trained weights θ (incl. Wup, Wgate, I η.</td></tr><tr><td>1: Input: Sequence chunks {X (i)}T=1.</td><td></td><td> Sequence partitioned for Context Parallelism (CP).</td><td></td></tr><tr><td></td><td>2: for all i  {1, . . . , T} in parallel do</td><td>&gt; Step 1: Compute update deltas.</td><td></td></tr><tr><td></td><td>Hi ← AttentionBlock(X (i); θ)</td><td> Standard attention, no changes required.</td><td></td></tr><tr><td>4:</td><td>, HiW Tt gate</td><td></td><td></td></tr><tr><td>5: Zi ← φ(Gi)  Ui</td><td>Vi ← Conv1DK(X (i)Wtarget</td><td> Compute NTP-aligned target with causal padding.</td><td></td></tr><tr><td>6: ∆Wi ← VT Zi</td><td></td><td>&gt; Compute gradient for the fast weight update.</td><td></td></tr><tr><td>7: 8: end for</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td> Step 2: Aggregate deltas associatively.</td><td></td></tr><tr><td></td><td>9: {Si}T=1 ← CUMSUM({∆Wi}T=1)</td><td></td><td></td></tr><tr><td>10:</td><td>for all i  {1, . . . , T } in parallel do</td><td> Step 3: Apply updates and compute outputs.</td><td></td></tr><tr><td>11: down</td><td>← +ηSi</td><td>&gt; Effective weight for chunk i uses updates from chunks &lt; i.</td><td></td></tr><tr><td></td><td>r(i−1))T</td><td></td><td></td></tr><tr><td>12:</td><td>Oi ← Zi(W (i−1)</td><td></td><td></td></tr><tr><td>13: end for</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

# C Experiment Details

This appendix provides all details of the experimental settings, datasets, model configurations, and training hyperparameters used for the results presented in Section 4. The following subsections detail the setups

for our three primary sets of experiments: the continual pre-training of Qwen3-4B-Base, the from-scratch pre-training of models at multiple scales (500M, 1.5B, and 4B), and the targeted ablation studies. Our goal is to provide sufficient detail to ensure the reproducibility of our findings.

# C.1 Details of Datasets

For the large scale pretraining, continual pretraining, and ablation study, we use the dataset collected by ourselves, we give the details of these datasets as follows.

From Scratch Pretraining Dataset. The pretraining dataset mainly includes general English and Chinese text, along with high knowledge- or reasoning-density data, code, mathematics data, and multilingual text, forming a balanced mixture of linguistic diversity, knowledge and reasoning-rich content, programming material, and mathematical reasoning.

Continual Pretraining Dataset. The continual pretraining dataset is designed to enhance long-context modeling: its short-document portion follows a distribution similar to Pretrain Data, while the long-document portion combines natural data such as books and repository-level code with synthetic data including retrievalaugmented and long-context-QA style constructions, ensuring both consistency with pretraining and coverage of challenging long-context scenarios. The data is organized into subsets with maximum sequence lengths of $3 2 \mathrm { k }$ and 128k for our two-stage training curriculum.

> 💡 **数据设置**: continual pretraining 数据混合短文档分布和长文档/合成长上下文 QA，用来让模型在保持预训练分布的同时适应 32k/128k 序列。

# C.2 Details of Training and Evaluation

Training Details. All models are trained on Nvidia H800 GPUs, with the detailed training hyperparameters listed in Tables 4 through 6.

Table 4 Training hyperparameters for 500M and 1.5B models.   

<table><tr><td>Hyperparameter</td><td>500M Model</td><td>1.5B Model</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>5e-4</td><td>3e-4</td></tr><tr><td>Batch Size</td><td>2M tokens</td><td>4M tokens</td></tr><tr><td>Weight Decay</td><td>0.1</td><td>0.1</td></tr><tr><td>Gradient Clipping</td><td>1.0</td><td>1.0</td></tr><tr><td>Warmup Steps</td><td>1024</td><td>1024</td></tr><tr><td>Sequence Length</td><td>32,768</td><td>32,768</td></tr><tr><td>Tokens Trained</td><td>20B</td><td>60B</td></tr><tr><td>Sliding Window Size</td><td>2,048</td><td>4,096</td></tr></table>

Table 5 Training hyperparameters for 1.7B models and 4B models pretraining   

<table><tr><td>Hyperparameter</td><td>value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>3e-4</td></tr><tr><td>Batch Size</td><td>8M tokens</td></tr><tr><td>Weight Decay</td><td>0.1</td></tr><tr><td>Gradient Clipping</td><td>1.0</td></tr><tr><td>Warm-up Tokens1</td><td>1.6B</td></tr><tr><td>Sequence Length</td><td>8,192</td></tr><tr><td>Tokens Trained</td><td>120B</td></tr></table>

Evaluation Details. We employ the evaluation framework lm-evaluation-harness [21] to evaluate the models on the common sense reasoning benchmarks and employ the evaluation framework opencompass [13] to evaluate the models on the long context benchmarks. All evaluation are conducted on Nvidia H800 GPUs.

Table 6 Hyperparameters for two-stage continual pre-training.   

<table><tr><td>Hyperparameter</td><td>Stage 1 (32k Context)</td><td>Stage 2 (128k Context)</td></tr><tr><td>Base Model</td><td>Qwen3-4B-Base</td><td>Qwen3-4B-Base</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>5e-6</td><td>5e-6</td></tr><tr><td>Weight Decay</td><td>0.1</td><td>0.1</td></tr><tr><td>Sequence Length</td><td>32,768</td><td>131,072</td></tr><tr><td>Tokens Trained</td><td>~20B</td><td>~15B</td></tr><tr><td>RoPE Extension</td><td>None</td><td>YaRN</td></tr><tr><td>Conv Size</td><td>5</td><td>5</td></tr></table>

Table 7 Hyperparameters for continual pre-training of LLaMA-3.1-8B and Qwen3-14B-Base.   

<table><tr><td>Hyperparameter</td><td>LLaMA-3.1-8B</td><td>Qwen3-14B-Base</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>5e-6</td><td>5e-6</td></tr><tr><td>Weight Decay</td><td>0.1</td><td>0.1</td></tr><tr><td>Sequence Length</td><td>32,768</td><td>32,768</td></tr><tr><td>Tokens Trained</td><td>∼20B</td><td>∼20B</td></tr><tr><td>RoPE Extension</td><td>None</td><td>None</td></tr><tr><td>Conv Size</td><td>5</td><td>5</td></tr></table>

In the evaluation of our continual pretrained Qwen3-4B model, we apply a clipping mechanism at inference time to ensure stable fast-weight updates. Specifically, if the Frobenius norm of an update delta $\lVert \Delta \mathbf { W } _ { \mathrm { d o w n } } ^ { ( i ) } \rVert _ { F }$ exceeds a predefined threshold $\tau$ , the delta matrix is rescaled to have norm $\tau$ before being applied, i.e., ∆W(i)dow $\Delta \mathbf { W } _ { \mathrm { d o w n } } ^ { ( i ) }  \tau \cdot \Delta \mathbf { W } _ { \mathrm { d o w n } } ^ { ( i ) } / \| \Delta \mathbf { W } _ { \mathrm { d o w n } } ^ { ( i ) } \| _ { F }$ . This prevents the accumulated updates from growing unboundedly as the sequence length increases, thereby maintaining numerical stability for long-context inference. For all reported evaluations of the Qwen3-4B model, this threshold was set to $\tau = 1 \mathrm { e } { - } 5$ .

> 💡 **稳定性细节**: Qwen3-4B 评估时对 $\Delta W_{down}$ 做 Frobenius norm clipping，阈值 $1e-5$。这提示长上下文累积更新有数值稳定风险，不能只看公式的线性累加。

To evaluate the efficiency of our In-Place TTT, we evaluate the prefill throughput and peak memory for sequence length ranging from 8k to $1 2 8 \mathrm { k }$ . We run the inference for our continual pretrained checkpoints based on Qwen3-4B-Base model. For the setting of sliding window, we set change the attention mechanism of these pretrained checkpoints to sliding window of 1024 tokens manually. We run the inference on Nvidia H800 GPUs with batch size of 1.

# C.3 Details of Model Configuration

This section details the architectural configurations of the models used in our experiments. All models are decoder-only Transformer architectures featuring standard components, including SwiGLU activations and Rotary Position Embeddings (RoPE) [46]. The key architectural parameters for all models trained from scratch are summarized in Table 8.

The models trained from scratch employ different attention mechanisms based on their experimental purpose. The 500M, 1.5B utilize sliding-window attention and we list the model configuration in table 8. The 4B-scale experiments and ablation study evaluate two variants: one with full attention and another with sliding-window attention. The backbone architectures for the 4B models and 1.7B models are identical to the Qwen3-4B-Base model and the Qwen3-1.7B-Base model.

For the continual pre-training experiments described in section 4.1, we start directly from publicly available pre-trained models—Qwen3-4B-Base [57], LLaMA-3.1-8B [24], and Qwen3-14B-Base [57]—inheriting their architectures without modification. In experiments featuring our method, the In-Place TTT module is integrated into the MLP blocks and applied to every sixth layer. For the ablation studies, this frequency is varied as described in the main paper. The training hyperparameters for Qwen3-4B-Base are listed in table 6, and those for LLaMA-3.1-8B and Qwen3-14B-Base are listed in table 7.

Table 8 Model architectural configurations for 500M and 1.5B Model.   

<table><tr><td>Parameter</td><td>500M</td><td>1.5B</td></tr><tr><td>Parameters (Approx.)</td><td>500M</td><td>1.5B</td></tr><tr><td>Hidden Size (dmodel)</td><td>1024</td><td>2048</td></tr><tr><td>Num Layers</td><td>24</td><td>24</td></tr><tr><td>Num Attention Heads</td><td>8</td><td>16</td></tr><tr><td>FFN Hidden Size (dff)</td><td>3072</td><td>6144</td></tr><tr><td>Window Size</td><td>2048</td><td>4096</td></tr><tr><td>Vocabulary Size</td><td>32,000</td><td>32,000</td></tr><tr><td>Rope Base</td><td>1e6</td><td>1e6</td></tr></table>

Initialization of In-Place TTT Modules. When integrating In-Place TTT into a pre-trained model for continual training, careful initialization is essential to preserve the model’s pre-trained capabilities at the start of training. Specifically, we initialize the newly introduced TTT components—the Conv1D operator and the projection matrix $\mathbf { W }$ target—such that the TTT update $\Delta \mathbf { W } _ { \mathrm { d o w n } }$ is negligible at initialization, ensuring the model begins from its original pre-trained behavior. Concretely, the depthwise Conv1D (kernel size 5, causal padding, no bias) is zero-initialized, so the target $\hat { \textbf { V } }$ is zero at initialization. The projection matrix $\mathbf { W _ { \mathrm { t a r g e t } } } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { \mathrm { m o d e l } } }$ is initialized as a sparse diagonal matrix, where all off-diagonal entries are zero and the diagonal entries are drawn from ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ with $\sigma$ being the model’s standard initializer range. This near-zero initialization of both components guarantees that the initial fast-weight update $\eta \hat { \mathbf { V } } _ { [ i ] } ^ { \top } \mathbf { Z } _ { [ i ] } \approx \mathbf { 0 }$ , and consequently the effective $\mathbf { W } _ { \mathrm { d o w n } }$ remains identical to its pre-trained value. As training progresses, the Conv1D and projection gradually learn to produce meaningful NTP-aligned targets, allowing the TTT mechanism to smoothly emerge without disrupting the pre-trained model.

> 💡 **warm-start 关键**: Conv1D 零初始化、$W_{target}$ 稀疏对角初始化，使初始 $\Delta W_{down} \approx 0$，因此插入模块时保持原预训练行为，再在 continual training 中逐渐学会写入。

# D Usage of LLMs

During the preparation of this manuscript, LLMs was used to check grammar and improve readability. All authors have reviewed, edited, and take full responsibility for the paper’s final version.

---

## 🔖 Section 总结

> 💡 **Section 小结**: 附录补足 theorem 证明、CP 伪代码和训练细节，尤其说明初始化与 clipping 对 drop-in 和 128k 稳定性很关键。
