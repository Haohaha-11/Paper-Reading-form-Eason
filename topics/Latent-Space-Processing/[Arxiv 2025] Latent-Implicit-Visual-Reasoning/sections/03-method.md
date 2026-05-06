[← 返回 README](../README.md)

## 📌 预览
Method 是全篇核心：添加 K 个 latent token，通过视觉瓶颈 attention mask 强迫视觉信息经 latent 流动，再用两阶段训练把 latent 从临时通道变成可用视觉表示。

---

# 3. Method

We begin by describing some background on the LMM architectures (Section 3.1), then introduce our method (Section 3.2) and implementation details (Section 3.3). An illus-

> 💡 **Section 路线图**: Method 的数据流是：图像 I 经视觉编码器和 projector 进入 LM embedding；prompt 后追加 latent token；Stage 1 用 attention mask 建瓶颈；Stage 2 恢复标准 mask 共同答题。

tration of our method is shown in Figure 2.

# 3.1. Preliminaries

Large Multimodal Models. LMMs are generative models that process both visual and textual inputs to perform various tasks. They typically consist of three parts, a visual encoder, a language model decoder, and a projector that projects outputs from the visual encoder into the embedding space of the language model. To be more precise, given a text prompt $Q$ and visual input $I$ , the prompt $Q$ is first encoded by a language encoder $l$ . The image $I$ is encoded using a visual encoder $v$ , then projected into the language model’s embedding space via a projector $p$ . Finally, the language model $M$ processes these embeddings to output a textual response $R$ :

> 💡 **基础符号**: 这里的 $p(v(I))$ 是视觉 token 进入语言模型的入口。LIVR 不改视觉 encoder/projector，而是在 LM 侧加 latent token 和 mask。

$$
R = M \left( p ( v ( I ) ) , l ( Q ) \right) )
$$

> 💡 **公式批读**: 公式表达的是标准 LMM 的单通路：视觉 embedding + 文本 prompt 直接进入 decoder 生成 response。LIVR 后面改的是 token 交互图，不是换掉这个主干。

Visual Question Answering. In Visual Question Answering (VQA), the LMM is provided with a set of images and tasked with answer questions about the images. Typically, the model is evaluated through top- $I$ accuracy. In the openended setting, the model is provided prompt $Q$ and visual inputs $I$ , and is asked to output the best answer $a$ . In the multiple choice setting, the model is additionally provided with a set of possible choices $a _ { 1 } , a _ { 2 } , \cdots , a _ { n }$ and tasked with selecting the best $a _ { i }$ . Next, we introduce our method.

> 💡 **任务形式**: 论文同时处理 open-ended counting 和 multiple-choice 视觉任务，因此 loss 都可落到 answer token 上；这也是它不需要中间监督的前提。

# 3.2. Latent Implicit Visual Reasoning

Current LMMs are trained to autoregressively generate text tokens for visual tasks. Visual information is projected once into the language space at the beginning of the inputs, after which the LMM reasons in the text space. We hypothesize that LMMs’ abilities can be improved by providing extra visual compute space, allowing them to implicitly learn how to best utilize visual information. We do this by equipping the LMM with latent tokens and training the model to use them through a novel visual bottlenecking approach.

> 💡 **机制动机**: 作者假设瓶颈不在视觉 encoder，而在“视觉信息进入语言空间后缺少额外视觉计算槽”。latent tokens 就是给模型一个可学习的视觉工作区。

Latent Tokens. To provide the LMM more expressivity to reason beyond the discrete text space, we equip it with latent tokens. Specifically, we introduce $K$ new special tokens, $L = \left\{ l _ { 1 } , l _ { 2 } , \cdots , l _ { K } \right\}$ , to the model’s existing vocabulary, $V$ . The new vocabulary becomes $V \cup L$ , with a total size of $| V | + K$ . During training, we append these latent tokens to the input. Thus, given an original prompt $Q$ , the new prompt $Q ^ { \prime }$ becomes ${ \textit { Q } } + L$ . While these tokens are randomly initialized, their corresponding rows in the embedding table remain unfrozen during training. Crucially, the model does not need to learn how to generate these latent tokens. Instead, it only needs to learn how to use them to represent important visual information.

> 💡 **Latent token 设计**: 这组 $L={l_1,...,l_K}$ 是新增特殊 token，不需要模型在输出里生成。它们被固定插入 prompt，训练时只解冻这些 embedding rows，让 token 本身学会变成视觉信息容器。

Visual Bottlenecking. In order to train our latent tokens, we introduce a bottlenecking approach where we force visual information to pass through the latent tokens. We do this by modifying the attention mask so that the answer tokens can only attend to the prompt tokens $Q$ and the latent tokens $L$ , but cannot attend to the visual inputs $I$ . To avoid residual visual leakage to the answer tokens, we also prevent the prompt tokens $Q$ from attending to the visual inputs $I$ . In this setup, the model can only “see” visual information through the latent tokens, which serve as the bottleneck.

> 💡 **瓶颈 attention mask**: 关键约束是 answer token 不能直接看 image token，prompt token 也不能看 image token，避免视觉信息绕过 latent 泄漏。这样正确回答的梯度会逼 latent 承载视觉证据。

This bottlenecking may help for a few reasons. Firstly, it forces the latents to carry visual information, providing extra “visual computation” that may be more expressive than pre-trained text tokens. Secondly, the model must focus on these visual latents to answer the question correctly, which may reduce existing language biases for the model.

> 💡 **为什么可能有效**: 第一，latent 是额外视觉计算容量；第二，瓶颈削弱语言捷径。注意这不是“多几个 token 就行”，后文 latents-only 消融正是验证这一点。

Multi-Stage Training. We utilize a 2-stage approach to train our model. In Stage 1, we apply the masking described above and train the model using the standard negative log likelihood (NLL) objective:

> 💡 **两阶段训练**: Stage 1 是强制学习 latent 视觉通道；Stage 2 让模型在正常可见图像 token 的条件下联合使用原图 token 和 enriched latent。没有 Stage 2，模型可能只适应受限 mask；没有 Stage 1，latent 容易被忽略。

$$
L = - { \frac { 1 } { | x | } } \sum _ { i = 1 } ^ { | x | } \log P ( x _ { i } | x _ { < i } )
$$

> 💡 **Loss 批读**: NLL 只算 answer tokens，说明监督信号完全来自最终答案。latent 是否学到视觉抽象，不靠人工标签，而靠“经由 latent 才能降低答案 loss”这个训练压力。

where we compute the loss only on the answer tokens. By doing so, our objective directly optimizes the latent tokens to capture the most useful visual information for solving the question. Moreover, this approach allows the model to implicitly discover optimal ways to use latent tokens without requiring explicit supervision or additional data.

After the latent tokens are trained to contain useful visual information in Stage 1, we revert to a standard attention mask that allows the answer tokens to attend to both the original image tokens and the latent tokens. The loss remains the same and is still computed only on the answer tokens. The goal of this stage is to train the model to jointly use both the original image tokens and the now-enriched latent tokens to answer the question.

# 3.3. Implementation Details

We fine-tune the language backbone using LoRA (applied to attention and MLP blocks) [12, 26], while keeping the vision encoder and projector frozen. In addition to LoRA parameters, we unfreeze only the embedding rows corresponding to the $K$ new latent tokens. Full optimization and schedule details are provided in Appendix B.

> 💡 **参数更新范围**: vision encoder/projector 冻结，LoRA 调 language backbone，同时解冻 K 个 latent embedding rows。这个选择让方法成本接近 PEFT，也隔离了收益来源。

---

## 🔖 Section 总结

> 💡 **Section 小结**:
> - 数据流: image/prompt → append K latent tokens → Stage 1 bottleneck mask → answer NLL → Stage 2 standard mask。
> - 核心机制: 用 attention mask 制造 image→latent→answer 的必经路径。
> - 可追问点: bottleneck mask 是否会伤害原始图像 token 的利用，Stage 2 如何恢复协作?
