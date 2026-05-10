[← 返回 README](../README.md)

# 02 - Neural Image Compression

## 📌 预览

本节定义 NIC 的压缩算子：WSI $\omega$ 被切成 patch，encoder $E$ 把每个 patch 变成 $C$ 维 embedding，再按空间位置组成 $\omega'$。随后分别介绍 VAE、contrastive Siamese、BiGAN 三种训练 $E$ 的方式。

---

# 2 NEURAL IMAGE COMPRESSION

Let us define $\boldsymbol { \omega } \in \mathbb { R } ^ { M \times N \times 3 }$ as the gigapixel image (e.g., a WSI) to be compressed, with $M$ rows, $N$ columns, and three color channels (RGB). In order to compress $\omega$ into a more compact representation $\omega ^ { \prime }$ , two steps were taken. First, $\omega$ was divided into a set of high-resolution patches $X =$ $\{ x _ { i j } \}$ with $x _ { i j } \in \mathbb { R } ^ { P \times P \times 3 } .$ , sampled from the $i$ -th row and $j$ -th column of an uniform grid of square patches of size $P$ using a stride of $S$ throughout $\omega$ . Second, each $\boldsymbol { x } _ { i j }$ was compressed independently from each other, generating a set of low-dimensional embedding vectors of length $C$ at each spatial location on the grid: $Y = \{ e _ { i j } \}$ with $e _ { i j } \in \mathbb { R } ^ { C }$ .

> 💡 **压缩对象定义**: $E$ 不看整张 WSI，只看单个 $P \times P$ patch；空间关系不是 encoder 学出来的，而是在把 $e_{ij}$ 放回网格时被保留下来。

We formulated the task of mapping high-entropy $X$ into low-entropy $Y$ as an instance of an unsupervised representation learning problem, and parameterized this mapping function with a neural network $E$ so that $X \xrightarrow { E } Y$ . By sliding $E$ throughout all $i j$ spatial locations, $\omega$ was compressed into $\omega ^ { \prime }$ with a total volume reduction of $F = 3 { \frac { S ^ { 2 } } { C } }$ . More formally:

> 💡 **压缩率读法**: $F=3S^2/C$ 来自原 patch 的 $S \times S \times 3$ 像素被替换成 $C$ 维向量。论文常用 $S=128,C=128$，理论体积约压缩 $3 \times 128=384$ 倍。

![Figure 2](../images/ea9e7bfeb8b9250a0396030a2a996b77283e01c3b2908b20610b16140a6e8027.jpg)

*Fig. 2: Variational Autoencoder. Top: the encoder maps a patch to an embedding vector depending on a noise vector while the decoder reconstructs the original patch from the embedding vector. Bottom: pairs of real and reconstructed patch samples using $C = 128$.*

> 💡 **Figure 2 批读**: VAE 图展示了 reconstruction family 的训练信号。它能产生连续 latent，但底部重建样本也提示其目标主要是像素分布相似，不一定优先保留肿瘤/组织判别语义。

$$
\omega \in \mathbb { R } ^ { M \times N \times 3 } \xrightarrow { E } \omega ^ { \prime } \in \mathbb { R } ^ { \frac { M } { S } \times \frac { N } { S } \times C }
$$

We investigated several unsupervised encoding strategies for learning $E$ . Three of the most well-known and accessible methods in unsupervised image representation learning were selected. In all cases, neural networks were trained to solve an auxiliary task and learn $E$ as a byproduct of the training process. Note that none of the studied methods required the use of manual annotations. Network architectures and training protocols are detailed in the Supplementary Material accompanying this paper.

> 💡 **辅助任务的共同输出**: 三种策略差异只在训练目标，最终交给 NIC-CNN 的接口一致：固定 encoder，把每个 patch 映射成 $C$ 维向量。

# 2.1 Variational Autoencoder

Two networks are trained simultaneously, the encoder $E$ and the decoder $D$ . The task of $E$ is to map an input patch $x$ into a compact embedded representation $e ,$ and the task of $D$ is to reconstruct $x$ from $e _ { \cdot }$ , producing $x ^ { \prime }$ . In this work, we used a more sophisticated version of AE, the variational autoencoder (VAE) [18]. The encoder in the VAE model learns to describe $x$ with an entire probability distribution instead of a single vector (Fig. 2). More formally, $E$ outputs $\mu \in \mathbb { R } ^ { C }$ and $\overset { \mathbf { \omega } } { \boldsymbol { \sigma } } \in \ \mathbb { R } ^ { C }$ , two embeddings representing the mean and standard deviation of a normal distribution such that:

$$
e = \mu + \sigma \odot n
$$

with $n \sim \mathcal { N } ( 0 , 1 )$ and $\odot$ denoting element-wise multiplication.

We trained the VAE model by optimizing the following objective:

$$
\begin{array} { r l } & { \mathcal { V } _ { \mathrm { V A E } } ( x , n , \theta _ { E } , \theta _ { D } ) = } \\ & { = \underset { E , D } { \operatorname* { m i n } } \left[ \underbrace { \left( x - D ( E ( x , n ) ) \right) ^ { 2 } } _ { \mathrm { R e c o n s t r u c t i o n ~ e r r o r } } + \underbrace { \gamma ( 1 + \log \sigma ^ { 2 } - \mu ^ { 2 } - \sigma ^ { 2 } ) } _ { \mathrm { K L \ d i v e r g e n c e } } \right] } \end{array}
$$

with $\gamma$ as a scaling factor, and $\theta _ { E }$ and $\theta _ { D }$ as the parameters of $E$ and $D$ , respectively. Note that we optimized $\theta _ { E }$ and $\theta _ { D }$ to minimize both the reconstruction error between the input and output data distributions, and the KL divergence between the embedding distribution and the normal $\mathcal { N } ( 0 , 1 )$ distribution.

> 💡 **VAE 公式批读**: reconstruction error 要求 $e$ 保留足够像素信息，KL 项把 embedding 约束到标准正态附近。对 NIC 来说，这种“可重建”不等于“可判别”，后文真实病理结果正体现这个差距。

![Figure 3](../images/e31113ba36f725c7f699b59212c287c8b20d03807c8beb89357415988e13abf8.jpg)

*Fig. 3: Contrastive training. Top: pairs of patches are extracted from gigapixel images. Pairs labeled as same originate from the same spatial location whereas different are extracted from either adjacent locations or different images. Bottom: scheme of a Siamese network trained for binary classification using the previous pairs.*

> 💡 **Figure 3 批读**: “same” 是同一空间位置经强增强后的两张 patch，“different” 主要是邻近非重叠 patch。邻近负样本故意很难，迫使 encoder 不只记颜色和粗纹理。

This procedure results in a continuous latent space where changes in the embedding vectors are proportional to changes in the input data and vice-versa, effectively retaining semantic knowledge present in the input space.

# 2.2 Contrastive Training

We assembled a training dataset composed of pairs of patches ${ \pmb x } = \{ { \boldsymbol x } ^ { ( 1 ) } , { \boldsymbol x } ^ { ( 2 ) } \}$ where each pair $_ { \textbf { \em x } }$ was associated with a binary label $y$ . Each label described whether the patches had been extracted from the same or a different location in a given gigapixel image, with $y = 1$ and $y = 0 .$ , respectively. We trained a two-branch Siamese network [20] to solve this classification problem (Fig. 3).

> 💡 **contrastive 的标签来源**: 这里的 binary label 不是人工病理标签，而是由采样策略自动生成的位置关系标签。因此它仍属于无监督/自监督 encoder 训练。

We applied heavy data augmentation on all patches as indicated in [26], i.e., rotation, color augmentation, brightness, contrast, zooming, elastic deformation, and added Gaussian noise. Due to the strong augmentation, patches from the same location looked substantially different in a highly non-linear fashion while keeping a similar overall structure (semantic), see examples in Fig. 3. Patches from the different class were extracted from two distributions: $7 5 \%$ of them corresponded to non-overlapping adjacent locations (i.e., neighboring patches) where most of the visual features were shared, and the remaining $2 5 \%$ were sampled from different WSIs. Note that we included more of the neighboring different pairs to increase the difficulty of the classification task, forcing the network to extract higherlevel features. The same data augmentation was applied to other encoders as well to ensure a fair comparison.

> 💡 **难负样本设计**: 75% negative 来自 adjacent locations 是关键，因为相邻 patch 颜色、染色、组织背景都接近。若 encoder 仍能分辨，就更可能编码细粒度结构而不是 slide-level shortcut。

![Figure 4](../images/b18f2b5ab16c1186afb1db25a3fb43104b6091698ad330a008193adb4453c263.jpg)

*Fig. 4: Adversarial Feature Learning. Top: three networks play a minimax game where the discriminator distinguishes between actual or generated image-embedding pairs, while the generator and the encoder fool the discriminator by producing increasingly more realistic images and embeddings. Bottom: real and generated patch samples using $C = 128$.*

> 💡 **Figure 4 批读**: BiGAN 和普通 GAN 的差别是多了 $E$，让真实 patch 也能映射到 latent。NIC 真正需要的是这个反向映射，而不是 generator 生成图像本身。

# 2.3 Bidirectional Generative Adversarial Network

The BiGAN setup consists of three networks: a generator $G$ , a discriminator $D$ , and an encoder $E$ (Fig 4). $G$ maps a latent variable $z \sim \mathcal { N } ( 0 , 1 )$ to generated images $x ^ { \prime }$ :

$$
z \sim \mathcal { N } ( 0 , 1 ) \in \mathbb { R } ^ { C } \overset { G } { \to } x ^ { \prime } \in \mathbb { R } ^ { P \times P \times 3 }
$$

whereas $E$ maps images $x$ sampled from the true data distribution $\mathcal { X }$ to embeddings $e$ :

$$
\boldsymbol { x } \sim \boldsymbol { \mathcal { X } } \in \mathbb { R } ^ { P \times P \times 3 } \xrightarrow { E } \boldsymbol { e } \in \mathbb { R } ^ { C }
$$

During training, the three networks play a minimax game where the discriminator $D$ tries to distinguish between actual or generated image-embedding pairs, i.e., $\{ x , e \}$ and $\{ x ^ { \prime } , z \}$ respectively, while $G$ and $E$ try to fool $D$ by producing increasingly more realistic images $x ^ { \prime }$ and embeddings $e$ closer to $\mathcal { N } ( 0 , 1 )$ . More formally, we optimized the following objective function:

$$
\begin{array} { r l } & { \mathcal { V } _ { \mathrm { B i G A N } } ( x , z , \theta _ { G } , \theta _ { E } , \theta _ { D } ) = } \\ & { = \underset { G , E } { \operatorname* { m i n } } \underset { D } { \operatorname* { m a x } } \left[ \log \left[ D \big ( x , \underset { e } { \underbrace { E ( x ) } } \big ) \right] + \log \left[ 1 - D \big ( \underset { x ^ { \prime } } { \underbrace { G ( z ) } } , z \big ) \right] \right] } \end{array}
$$

with $\theta _ { G } , \theta _ { E } ,$ and $\theta _ { D }$ representing the parameters of $G$ , $E ,$ and $D _ { \cdot }$ , respectively.

The authors of BiGAN theoretically and experimentally demonstrate that $G$ and $E$ learn an approximate inverse mapping function from each other, producing an encoding network $E$ that learns a powerful low-dimensional representation of the image world inherited from $G$ , suitable for downstream tasks such as supervised classification [14].

> 💡 **BiGAN 目标批读**: Discriminator 判断的是 pair 是否来自真实 joint $(x,E(x))$ 或生成 joint $(G(z),z)$。为了骗过它，$E(x)$ 必须像真实 latent，且与 $x$ 匹配；这比单纯重建更偏向建模数据流形的高层结构。

## 🔖 Section 总结

| Encoder | 辅助任务 | NIC 里可能学到的东西 | 主要风险 |
|---|---|---|---|
| VAE | 重建 patch + KL | 连续、可压缩的 patch 表示 | 容量被低级像素重建消耗 |
| Contrastive | same/different patch pair 分类 | 对增强不变、对邻近 patch 可分的局部语义 | 依赖手工 pair 设计 |
| BiGAN | image-latent pair adversarial matching | generator latent 中的高层数据流形 | 训练不稳定、计算更重 |
