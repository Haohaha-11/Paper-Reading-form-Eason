[← 返回 README](../README.md)

# 3 Model Architecture

## 📌 预览
本节是 CAR 的机制核心：ResamplerNet 预测 kernel/offset，Downscaling module 通过双线性插值和 soft round 保持可微，SRNet/EDSR 提供 L1 重建目标，同时用 offset distance 和 partial TV 正则控制采样自由度。

---

# 3 Model Architecture

> 💡 **方法总览**: 数据流是 HR -> ResamplerNet 预测每个 LR 像素的 kernel/offset -> differentiable downscaling 生成 LR -> SRNet/EDSR 重建 HR -> L1 与正则 loss 反传。ResamplerNet 是被 SRNet 教出来的下采样器。

This section introduces the architecture and formulation of the content adaptive resampler (CAR) model for image downscaling. As shown in Fig. 1, the framework is composed of two major components, i.e., the resampler generation network (Resampler-Net) and the SR network (SRNet). The ResamplerNet is responsible for estimating the content adaptive resampling kernels according to its input HR image, later the resampling kernels are applied to the input HR image to produce the downscaled image. The SRNet takes the resulting downscaled image as input and tries to restore the underlying HR image. The entire framework is trained end-to-end in an unsupervised manner where the primary objective we need to optimize is the SR reconstruction error with respect to the input HR image. By back-propagating the gradient of the reconstruction error through the SRNet and ResamplerNet, the gradient descent algorithm adjusts the parameters of the resampler generation network to produce better resampling kernels which make the downscaled image can be super-resolved more easily.

![Figure 1](../images/8ed0595cd09f9fde7b315d7c2060843f080a39b0944e8cbdcd5b0fc65cc5c92e.jpg)  
*Figure 1: Network architecture.*

> 💡 **Figure 1 框架批读**: 图中的蓝色虚线块说明 CAR 不是单独训练一个 compressor，而是把下采样模块夹在 HR 和 SRNet 之间端到端优化。关键梯度路径从 SR reconstruction error 穿过 SRNet、soft rounding/重采样，再更新 ResamplerNet 的 kernel 和 offsets。

Figure 1: Network architecture. It consists of three parts, the ResamplerNet, the Downscaling module, and the SRNet. The ResamplerNet is designed to estimate content adaptive resampling kernels and its corresponding offsets for each pixel in the downscaled image. The SRNet, can be any form of differentiable upsampling operations, is employed to guide the training of the ResamplerNet by simply minimizing the SR error. The entire framework is trained end-to-end by back-propagating error signals through a the differentiable downscaling module. The composition of each building block is detailed on the blue dashed frame.

# 3.1 Content adaptive image downscaling resampler

> 💡 **Resampling 而非直接生成**: 直接生成 LR 容易学出不自然的像素分布，所以作者让网络只预测采样规则。这样即使没有 LR ground truth，输出仍由原 HR 像素插值得到，保留了“可被当作普通图像存储/传输”的性质。

We design the proposed content adaptive image downscaling model that is trained using the unsupervised strategy. Methods presented in [20, 21, 4] synthesize the downscaled image by combining latent representations of the HR image extracted by the CNN and proper constraints are required to make sure that the result is a meaningful image. In this paper, we propose to obtain the downscaled image using the idea of resampling the HR image, which effectively makes the downscaled result look like the original HR image without any constraints.

> 💡 **Kernel + offset 机制**: 每个 LR 像素都有自己的 kernel weights，并且 kernel 中每个元素都有独立 offset。这比动态滤波多了空间自由度：不仅决定邻域像素怎么加权，还决定去 HR 的哪个非整数位置采样，类似可学习的非均匀 atrous/deformable resampling。

The filters for traditional bilinear or bicubic downscaling are basically fixed, with the only variation being the shift of the kernel according to the location of the newly created pixel in the downscaled image. Contrary to this, we propose to use dynamic downscaling filters inspired by the dynamic filter networks [22]. The downscaling kernels are generated for each pixel in the downscaled image depending on the effective resampling region on the HR image. It can be considered as one type of metalearning [23] that learns how to resample. However, filter-based image resampling methods generally require a certain minimum kernel size to be effective [19]. We alleviate this issue by taking the idea from the deformable convolutional networks [24]. In addition to estimating the content adaptive resampling kernel weights, we also associate spatial offset with each element in the resampling kernel. The content adaptive resampling kernels with position offsets can be considered as learnable dilated (atrous) convolutions [25] with the learned dilation rate. Besides, the offset for each kernel element can be different in both magnitude and direction, it can perform non-uniform sampling according to the content structure of the input HR image.

> 💡 **张量形状读法**: 对输入 HR 尺寸 h x w、缩放因子 s，CAR 输出 (h/s) x (w/s) 个 LR 像素所需的 m x n kernel，以及对应的 ΔX/ΔY offsets。kernel sum-to-1 保证局部加权平均的亮度尺度稳定，offset 则承担结构对齐。

We use a convolutional neural network with residual connections [26] to estimate the weights and offsets for each resampling kernel. The ResamplerNet consists of downscaling blocks, residual blocks and upscaling blocks. The downscaling and residual blocks are trained to model the context of the input HR image as a set of feature maps. Then, two upscaling blocks are used to learn the content adaptive resampling kernel weights $\mathbf { K } \ \in \ \mathbb { R } ^ { ( h / s ) \times ( w / s ) \times m \times n }$ , offsets in the horizontal direction $\Delta \mathbf { Y } \in \mathbb { R } ^ { ( h / s ) \times ( w / s ) \times m \times n }$ and offsets in the vertical direction $\Delta \mathbf { X } \in \mathbb { R } ^ { ( h / s ) \times ( w / s ) \times m \times n }$ , respectively. $h$ and $w$ are the height and width of the input HR image, $s$ is the downscaling factor, and m, $n$ represent the size of the content adaptive resampling kernel. Each kernel is normalized so that elements of a kernel are summed up to be 1.

# 3.2 Image downscaling

> 💡 **下采样前向过程**: 一个 LR 像素不是固定取 HR 网格点，而是先把 LR 坐标投影到 HR 坐标，再围绕该位置用 kernel weights 和 offsets 采样。三通道共用同一个 resampling kernel，避免 RGB 通道采样位置不一致造成颜色错位。

The estimated content adaptive resampling kernels are applied to the corresponding positions of the input HR image to construct the pixel in the downscaled image. For each output pixel, the same resampling kernel is simultaneously applied to three channels of the RGB image. As illustrated by Fig. 2, pixels covered by the resampling kernel are weights summed to obtain the pixel value in the downscaled image.

Forward pass. Firstly, we need to position each resampling kernel, associated with the pixel of the downscaled image, on the HR image. It can be achieved using the projection operation defined as:

![Figure 2](../images/e83f62f070b3e63750c34e2c3568bc8e248b324b7d5b237e8043a113a973eeb6.jpg)  
*Figure 2: Non-uniform resampling.*

> 💡 **Figure 2 偏移批读**: 黑色点是 HR 像素中心，红色点是实际采样位置。蓝色箭头就是 offset，它让 kernel 元素从规则网格移动到更适合当前结构的位置；因此 CAR 可以沿边缘或纹理方向采样，而不是被固定 bicubic 网格限制。

Figure 2: Non-uniform resampling. The black $^ +$ represents the center of a pixel in the HR image and the red $^ +$ indicate the position of the resampling kernel element. The blue dashed arrow shows the offset direction and magnitude of the resampling kernel element relative to its corresponding pixel center.

$$
( u , \nu ) = ( x + 0 . 5 , y + 0 . 5 ) \times s c a l e - 0 . 5
$$

where $( x , y )$ is the indices of a pixel at the $x$ -th row and $y .$ -th ,column in the downscaled image, scale represents the downscaling factor, and the resulting $( u , \nu )$ is the center of downscaled pixel $p _ { x , y } ^ { \mathrm { L R } }$ ,projected on the input HR image. Equation 1 assumes ,that pixels have a nonzero size, and the distance between two neighboring samples is one pixel.

Then, each pixel in the downscaled image is created by local filtering on pixels in the input HR image with the corresponding content adaptive resampling kernel as follows:

$$
\begin{array} { l } { \pmb { p } _ { x , y } ^ { \mathrm { L R } } = \displaystyle \sum _ { i = 0 } ^ { m - 1 } \sum _ { j = 0 } ^ { n - 1 } \mathbf { K } _ { x , y } ( i , j ) \cdot s ^ { \mathrm { H R } } \left( u + i - \frac { m } { 2 } + \Delta \mathbf { X } _ { x , y } ( i , j ) , \right. } \\ { \displaystyle \left. \nu + j - \frac { n } { 2 } + \Delta \mathbf { Y } _ { x , y } ( i , j ) \right) } \end{array}
$$

where $\mathbf { K } _ { x , y } \in \mathbb { R } ^ { m \times n }$ is the resampling kernel associated with the , downscaled pixel at location $( x , y )$ , $\Delta \mathbf { X } _ { x , y } \in \mathbb { R } ^ { m \times n }$ and $\Delta \mathbf { Y } _ { x , y } \in$ $\mathbb { R } ^ { m \times n }$ , ,are the spatial offset for each element in the $\mathbf { K } _ { x , y }$ . The $s ^ { \mathrm { H R } }$ ,is the sample point value of the input HR image. Due to the location projection and fractional offsets, $s ^ { \mathrm { H R } }$ can refer to nonlattice point on the HR image, therefore, $s ^ { \mathrm { H R } }$ is computed by bilinear interpolating the nearby four pixels around the fractional position:

$$
\begin{array} { r l } & { s _ { u ^ { \prime } , \nu ^ { \prime } } ^ { \mathrm { H R } } = ( 1 - \alpha ) ( 1 - \beta ) \cdot p _ { \lfloor u ^ { \prime } \rfloor , \lfloor \nu ^ { \prime } \rfloor } ^ { \mathrm { H R } } + \alpha ( 1 - \beta ) \cdot p _ { \lfloor u ^ { \prime } \rfloor , \lfloor \nu ^ { \prime } \rfloor + 1 } ^ { \mathrm { H R } } } \\ & { \qquad + ( 1 - \alpha ) \beta \cdot p _ { \lfloor u ^ { \prime } \rfloor + 1 , \lfloor \nu ^ { \prime } \rfloor } ^ { \mathrm { H R } } + \alpha \beta \cdot p _ { \lfloor u ^ { \prime } \rfloor + 1 , \lfloor \nu ^ { \prime } \rfloor + 1 } ^ { \mathrm { H R } } } \end{array}
$$

where $u ^ { \prime }$ and $\nu ^ { \prime }$ are fractional position on the HR image, $\alpha =$ $u ^ { \prime } - \lfloor u ^ { \prime } \rfloor$ and $\beta = \nu ^ { \prime } - \lfloor \nu ^ { \prime } \rfloor$ αare the bilinear interpolation weights.

> 💡 **可微性关键**: 公式 4-9 的目的不是提出新 SR 网络，而是证明下采样操作能把 SRNet 的梯度传回 kernel weights 和 offsets。对 weight 的梯度就是采样到的 HR 值；对 offset 的梯度依赖双线性插值的局部像素差，因此 offsets 会被边缘/纹理梯度牵引。

Backward pass. The ResamplerNet is trained using the gradient descent technique and we need to back-propagate gradients

from the SRNet through the resampling operation. The partial derivative of the downscaled pixel with respect to the resampling kernel weight can be formulated as:

$$
\begin{array} { r } { \frac { \partial p _ { x , y } ^ { \mathrm { L R } } } { \partial \mathbf { K } _ { x , y } ( i , j ) } = s ^ { \mathrm { H R } } \left( u + i - \frac { m } { 2 } + \Delta \mathbf { X } _ { x , y } ( i , j ) , \right. } \\ { \left. \nu + j - \frac { n } { 2 } + \Delta \mathbf { Y } _ { x , y } ( i , j ) \right) \qquad } \end{array}
$$

the partial derivative of downscaled pixel with respect to the element in the resampling kernel is simply the interpolated pixel value. Equation 4 is derived with a single channel image and can be generalized to the color image by summing up values calculated separately on the R, G and B channels.

The partial derivative of downscaled pixel with respect to the kernel element offset is computed as:

$$
\begin{array} { r l } & { \frac { \partial { p } _ { x , y } ^ { \mathrm { L R } } } { \partial \Delta \mathbf { X } _ { x , y } ( i , j ) } = \frac { \partial { p } _ { x , y } ^ { \mathrm { L R } } } { \partial s _ { u ^ { \prime } , v ^ { \prime } } ^ { \mathrm { H R } } } \cdot \frac { \partial s _ { u ^ { \prime } , v ^ { \prime } } ^ { \mathrm { H R } } } { \partial \Delta \mathbf { X } _ { x , y } ( i , j ) } } \\ & { \frac { \partial { p } _ { x , y } ^ { \mathrm { L R } } } { \partial \Delta \mathbf { Y } _ { x , y } ( i , j ) } = \frac { \partial { p } _ { x , y } ^ { \mathrm { L R } } } { \partial s _ { u ^ { \prime } , v ^ { \prime } } ^ { \mathrm { H R } } } \cdot \frac { \partial s _ { u ^ { \prime } , v ^ { \prime } } ^ { \mathrm { H R } } } { \partial \Delta \mathbf { Y } _ { x , y } ( i , j ) } } \end{array}
$$

because we employ bilinear interpolation to compute $\pmb { s } _ { u ^ { \prime } , \nu ^ { \prime } } ^ { \mathrm { H R } }$ u 0 v 0 , there-,fore, the partial derivative of downscaled pixel with respect to the kernel offset is defined as:

$$
\begin{array} { r l } & { \frac { \partial p _ { x y } ^ { \mathrm { I R } } } { \partial \Delta \mathbf { X } _ { x y } ( i , j ) } = } \\ & { \mathbf { K } _ { x y } ( i , j ) \cdot ( 1 - \boldsymbol { \it \beta } ) \cdot \left( p _ { | v | , | v | ^ { r } | + 1 } ^ { \mathrm { H R } } - p _ { | v | , | v ^ { r } | } ^ { \mathrm { H R } } \right) + } \\ & { \boldsymbol { \beta } \cdot \left( p _ { | v | + 1 , | v ^ { r } | + 1 } ^ { \mathrm { H R } } - p _ { | v ^ { r } | + 1 , | v ^ { r } | } ^ { \mathrm { H R } } \right) } \\ & { \frac { \partial p _ { x y } ^ { \mathrm { I R } } } { \partial \Delta \mathbf { X } _ { x y } ( i , j ) } = } \\ & { \mathbf { K } _ { x y } ( i , j ) \cdot ( 1 - \boldsymbol { \alpha } ) \cdot \left( p _ { | v | + 1 , | v ^ { r } | } ^ { \mathrm { H R } } - p _ { | v ^ { r } | , | v ^ { r } | } ^ { \mathrm { H R } } \right) + } \\ & { \boldsymbol { \alpha } \cdot \left( p _ { | v ^ { r } | + 1 , | v ^ { r } | + 1 } ^ { \mathrm { H R } } - p _ { | v ^ { r } | , | v ^ { r } | + 1 } ^ { \mathrm { H R } } \right) } \end{array}
$$

also because Equation 9 is defined with a single color image, we need to sum up partial derivative of each color component with respect to the offset to obtain the final partial derivative of downscaled pixel with respect to the kernel element offset.

> 💡 **Soft round 的位置**: 训练时需要模拟真实 8-bit 图像的取整，但 round 不可导。CAR 前向仍用普通 round 得到整数 LR，反向用 soft round 近似梯度，这保证模型学到的 LR 更接近实际可存储图像，而不是只在浮点空间有效。

The pixel value of the downscaled image created using Equation 2 is inherently continuous floating point number. However, common image representation describes the color using integer number ranging from 0 to 255, thus, a quantization step is required. Since simply rounding the floating point number to its nearest integer number is not differentiable, we utilize the soft round method proposed by Nakanishi et al.[27] to derive the gradient in the back-propagation during the training phase. The soft round function is defined as:

$$
\mathrm { r o u n d } _ { \mathrm { s o f t } } ( x ) = x - \alpha \frac { \sin 2 \pi x } { 2 \pi }
$$

where $\alpha$ is a tuning parameter used to adjust the gradient around αthe integer position. Note that in the forward propagation, the non-differentiable round function is used to get the nearest integer value.

# 3.3 Image upscaling

> 💡 **SRNet 角色**: SRNet 在这里是 teacher/optimizer target，不一定必须是 EDSR；只要上采样操作可微，就能给 ResamplerNet 提供恢复 HR 的误差信号。论文选择 EDSR 是为了和当时强 SR baseline 公平对齐。

The proposed CAR model is trained using back-propagation to maximize the SR performance. The image upscaling module can be of any form of SR networks, even the differentiable bilinear or bicubic upscaling operations. After the seminal super-resolution model using deep learning proposed by Dong et al.[28], i.e., the SRCNN, many state-of-the-art neural SR models have been proposed [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]. More reviews and discussion about single image SR using deep learning can be referred to [8]. This paper employs state-of-the-art SR model EDSR [33] as the image upscaling module to guide the training of the proposed CAR model.

The EDSR is one of the state-of-the-art deep SR networks and its superior performance is benefited from the powerful residual learning techniques [26]. The EDSR is built on the success of the SRResNet [32] which achieved good performance by simply employing the ResNet [26] architecture on the SR task. The EDSR enhanced SR performance by removing unnecessary parts (Batch Normalization layer [41]) from the SRResNet and also applying a number of tweaks, such as adding residual scaling operation to stable the training process [33]. The SRNet part of Fig. 1 shows the architecture of the EDSR. It is composed of convolution layers converting the RGB image into feature spaces and a group of residual blocks refining feature maps. A global residual connection is employed to improve the efficiency of the gradient back-propagation. Finally, sub-pixel layers are utilized to upsample and transform features into the target SR image.

# 3.4 Training objectives

> 💡 **Loss 设计**: 主 loss 是 EDSR 常用的 L1 重建，避免因为加入 SSIM/perceptual loss 改变比较条件。offset distance regularization 限制单个 kernel 元素乱跑，partial TV loss 限制相邻 kernel 的 offset 不连续；二者分别处理 topology 和 LR jaggies。

One of the main contributions of our work is that we propose a model to learn image downscaling without any supervision signifing that no constraint is applied to the downscaled image. The only objective guiding the generation of the downscaled image is the SR restoration error. The most common loss function generally defaults to the SR network training is the mean squared error (MSE) or the L2 norm loss, but it tends to lead to poor image quality as perceived by human observers [42]. Lim et al.[33] found that using another local metric, e.g., L1 norm, can speed up the training process and produce better visual results. In order to improve the visual fidelity of the super-resolved image, perceptually-motivated metrics, such as SSIM [16], MS-SSIM [43] and perceptual loss [44] are usually incorporated in the SR network training. To do fair comparisons with the EDSR, we only adopt the L1 norm loss as the restoration metric as suggested by [33]. The L1 norm loss defined for SR is:

$$
\mathcal { L } ^ { \mathrm { L } _ { 1 } } ( \hat { \mathbf { I } } ) = \frac { 1 } { N } \sum _ { p \in \mathbf { I } } | p - \hat { p } |
$$

where $\hat { \bf I }$ is the SR result, $\pmb { p }$ and $\hat { p }$ represent the ground-truth and reconstructed pixel value, $N$ indicates the number of pixels times the number of color channels.

We associate spatial offset for each element in the resampling kernel, and the offset is estimated without taking the neighborhoods of the kernel element into account. Independent kernel element offset may break the topology of the resampling kernel. To alleviate this problem, we suggest using the total offset distance of all kernel elements as a regularization which encourages kernel elements to stay in their rest position (avoid unnecessary movements). Additionally, since the pixels indexed by the kernel elements that are far from the central position may have less correlation to the resampling result, we assign different weights to their corresponding offset distance in terms of their position relative to the central position. The offset distance regularization term for a single resampling kernel is thus formulated as:

$$
\mathcal { L } _ { x , y } ^ { \mathrm { o f f s e t } } = \sum _ { i = 0 } ^ { m - 1 } \sum _ { j = 0 } ^ { n - 1 } \eta + \sqrt { \Delta \mathbf { X } _ { x , y } ( i , j ) ^ { 2 } + \Delta \mathbf { Y } _ { x , y } ( i , j ) ^ { 2 } } \cdot w ( i , j )
$$

where $w ( i , j ) = \ \sqrt { ( i - { \textstyle { \frac { m } { 2 } } } ) ^ { 2 } + ( j - { \textstyle { \frac { n } { 2 } } } ) ^ { 2 } } \ / \ \sqrt { { \textstyle { \frac { m } { 2 } } } ^ { 2 } + { \textstyle { \frac { n } { 2 } } } ^ { 2 } }$ is the normalized distance weight, the $\eta$ is introduced to act as the offset distance weight regulator.

> 💡 **TV 正则的取舍**: TV loss 不是为了提升 PSNR，而是为了减少 LR 图像中由 offset 不连续引起的 jaggies。实验后面显示它会牺牲部分 SR 指标，这正好暴露 CAR 的核心张力：LR 视觉质量和 SR 可恢复性并不完全一致。

The inconsistent resampling kernel offset of spatially neighboring resampling kernels may cause pixel phase shift on the resulting LR images, which is manifested as jaggies, especially on the vertical and horizontal sharp edges (e.g., the LR image in Fig. 5 (b)). To alleviate this phenomenon, we introduce the total variation (TV) loss [45] to constrain the movement of spatially neighboring resampling kernels. Instead of constraining the offsets on both vertical and horizontal directions, we only regularize vertical offsets on the horizontal direction and horizontal offsets on the vertical direction, which we name it the partial TV loss. Besides, variations of each resampling kernel offsets are weighted by its corresponding resampling kernel weights, leading to the following formula:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } ^ { \mathrm { T V } } = \sum _ { x , y } \left( \sum _ { i , j } | \Delta \mathbf { X } _ { \cdot , y + 1 } ( i , j ) - \Delta \mathbf { X } _ { \cdot , y } ( i , j ) | \cdot \mathbf { K } ( i , j ) \right. } } \\ { { \displaystyle \left. + \sum _ { i , j } | \Delta \mathbf { Y } _ { x + 1 , \cdot } ( i , j ) - \Delta \mathbf { Y } _ { x , \cdot } ( i , j ) | \cdot \mathbf { K } ( i , j ) \right) } } \end{array}
$$

Finally, the optimization objective of the entire framework is defined as:

$$
\mathcal { L } = \mathcal { L } ^ { \mathrm { L } _ { 1 } } + \lambda \overline { { \mathcal { L } ^ { \mathrm { o f f s e t } } } } + \gamma \mathcal { L } ^ { \mathrm { T V } }
$$

where the $\overline { { \mathcal { L } ^ { \mathrm { o f f s e t } } } }$ is the mean offset distance regularization term of all the resampling kernels, and $\lambda$ is a scalar introduced to λcontrol the strength of offset distance regularization. $\gamma$ is also a γscalar used to tune the contribution of the partial TV loss to the final optimization objective.

## 🔖 Section 总结

### 数据流速查

| 阶段 | 输入 | 中间变量 | 输出 |
|---|---|---|---|
| ResamplerNet | HR image | feature maps | K, ΔX, ΔY |
| Downscaling | HR + K/offsets | projected HR coordinates + bilinear interpolation | integer LR image |
| SRNet | LR image | EDSR residual features | SR/HR reconstruction |
| Optimization | SR vs HR | L1 + offset + TV losses | updated ResamplerNet/SRNet |

### 核心洞察
1. Kernel weights 决定“混合哪些像素”，offsets 决定“到哪里取样”。
2. Soft round 让训练贴近实际 8-bit LR 存储，而不是只在浮点图像上成立。
3. Offset regularization 保拓扑，partial TV 管 LR jaggies，二者服务不同风险。
