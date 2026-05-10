[← 返回 README](../README.md)

# 03 - Gigapixel Image Analysis

## 📌 预览

本节说明压缩后的 $\omega'$ 如何进入 CNN：输入层 filter depth 改成 code size $C$，空间维度对应 WSI patch 网格；训练时用 crop sampling、8-fold augmentation、depthwise separable conv 控制过拟合；最后用第一层 Grad-CAM 保持较高定位分辨率。

---

# 3 GIGAPIXEL IMAGE ANALYSIS

In this section, we describe a method to train a CNN to predict image-level labels directly from compressed gigapixel images. Furthermore, we analyzed the location of visual cues associated with the image-level labels.

> 💡 **Section 路线图**: 前半节回答“压缩 WSI 怎么喂 CNN”，后半节回答“CNN 学到的位置证据怎么可视化”。

# 3.1 Feeding a CNN with compressed gigapixel images

We consider a dataset of gigapixel images $\begin{array} { r } { \Omega = \{ \omega _ { i } \} _ { , i = 1 } ^ { Q } } \end{array}$ that were compressed into Ω0 = {ω0i}Qi=1 with ω0i ∈ R MS × NS ×C using Eq. 1. In order to train a standard CNN on a dataset like $\Omega ^ { \prime }$ , we set the depth of the convolutional filters of the input layer to be equal to the code size $C$ used to compress the images.

> 💡 **CNN 输入适配**: 普通 RGB CNN 第一层深度是 3；NIC-CNN 第一层深度是 $C$。这意味着每个空间位置已经是一段 patch descriptor，卷积在 descriptor grid 上学习邻域组合。

We hypothesized that such a CNN can learn to detect highly discriminative features by exploiting two complementary sources of information from $\Omega ^ { \prime }$ : (1) the global context encoded within the spatial arrangement of embedding vectors, and (2) the local high-resolution information encoded within the features of each embedding vector.

> 💡 **局部与全局的接口**: 局部信息藏在每个 $e_{ij}$ 的通道里；全局信息来自 $e_{ij}$ 在二维网格中的排列。这个设计正对应 Introduction 的 what/where。

# 3.2 Preventing overfitting

Note that in this setup, despite its gigapixel nature, each compressed image $\omega _ { i } ^ { \prime }$ constitutes a single training data point. Most public datasets with gigapixel images and their respective image-level labels consist only of a few hundred data points [3], [4], increasing the risk of overfitting. The steps taken to prevent this effect are enumerated below.

> 💡 **样本数悖论**: WSI 有十亿像素，但监督样本数仍是几百张 slide。压缩不能自动增加 label 数，所以 CNN 训练仍然容易过拟合。

First, we extended the training dataset $\Omega ^ { \prime }$ by taking spatial crops of size $R \times R \times C$ from $\omega _ { i } ^ { \prime } ,$ , drastically increasing the total number and variability of the samples presented to the CNN [27]. During training, we randomly sampled the location of the center pixel of these crops. During testing, we selected $T$ crops uniformly distributed along the spatial dimensions of $\omega _ { i } ^ { \prime }$ and averaged the predictions of the CNN across them [27]. Without any loss of generality, we applied this method to histopathology WSIs. As WSIs often contain large empty areas with no tissue, we detected the tissue regions [28] and sampled crops proportionally to the distance to background to accelerate the training, so that areas with higher tissue density were sampled more often. Similarly, test crops were sampled from locations where tissue was present.

> 💡 **crop sampling 的作用**: 这不是 patch-level label 训练，而是从同一 compressed WSI 采样多个 $R \times R \times C$ 子图，仍继承 slide label。tissue-aware sampling 避免大量空白区域稀释训练信号。

The second measure taken to prevent overfitting was a simple augmentation at image level (i.e., 90-degree rotation and mirroring), encoding each image 8 times. This augmentation was carried out during testing as well, averaging the predictions of the CNN across them.

> 💡 **8 次编码的含义**: 旋转/镜像发生在 image level，测试时也做 TTA 平均。这样能减少 WSI 朝向和组织摆放对预测的偶然影响。

Finally, we designed a CNN architecture aimed at reducing the number of parameters present in the model. In particular, all convolutional layers were set to use depthwise separable convolutions, a type of convolution that reduces the number of parameters while maintaining a similar level of performance [29].

> 💡 **轻量 CNN 的必要性**: 输入虽被压缩，但仍是较大的 embedding grid；depthwise separable conv 同时减少参数和过拟合风险，后面 appendix 给出 8 层结构。

![Figure 5](../images/d8c01d643b594428dde33cb7243a36a713f0715d950d838c7737a8609718e44e.jpg)

*Fig. 5: Example of an image from the synthetic dataset. Left: ground truth mask depicting the tilted and non-tilted rectangles that simulate lesions in grey and white, respectively. Center: image containing instances of MNIST digits; classes are defined by the rectangles or selected randomly. Right: all digits within the tilted rectangle boundary (in green) belong to the same class (number two), which corresponds to the image label as well.*

> 💡 **Figure 5 批读**: synthetic task 专门构造了“局部 digit class + 全局 tilted rectangle”组合。模型必须找到倾斜矩形区域并读出其中 digit 类别，不能只看单个 patch 或整图颜色。

The problem of feature localization is of utmost relevance for gigapixel image analysis: visual cues related to the image-level labels are often sparse and positioned in arbitrary locations within the image. For the purpose of identifying the location of these visual cues, we applied the Gradient-weighted Class-Activation Map (Grad-CAM) algorithm [24] to our trained CNN.

> 💡 **定位为何困难**: 图像级 label 不告诉模型病灶在哪里；Grad-CAM 只能在训练后反推模型关注区域，因此是弱监督定位证据，不是显式监督分割。

# 3.3 Visualizing visual cues related to image-level labels

Given a compressed gigapixel image $\omega ^ { \prime }$ , its associated image-level label $y ,$ , and a trained CNN, Grad-CAM performs a forward pass over $\omega ^ { \prime }$ to produce a set of $J$ intermediate three-dimensional feature volumes $f _ { j } ^ { ( k ) }$ , with $j$ and $k$

indicating the $j$ -th and $k$ -th convolutional layer and feature map, respectively. Subsequently, it computes the gradients of f (k)j with respect to $y$ for a fixed convolutional layer. It averages the gradients across the spatial dimensions and obtains a set of gradient coefficients $\gamma _ { j } ^ { ( k ) } .$ , indicating how relevant each feature map is for the desired output $y$ . Finally, it performs a weighted sum of the feature maps $f _ { j } ^ { ( k ) }$ using the gradient coefficients $\gamma _ { j } ^ { ( k ) }$ :

$$
h ^ { ( k ) } = \sum _ { j = 1 } ^ { J } f _ { j } ^ { ( k ) } \gamma _ { j } ^ { ( k ) }
$$

We applied the visualization method to the first convolutional layer $k = 1$ ) in order to maximize the heatmap resolution.

> 💡 **Grad-CAM 公式批读**: $\gamma_j^{(k)}$ 是“某个 feature map 对当前输出的梯度重要性”，$h^{(k)}$ 是 feature map 的加权和。选第一层是为了少丢空间分辨率，更适合映射回 WSI 上看 where。

## 🔖 Section 总结

| 组件 | 解决的问题 |
|---|---|
| $C$-channel input CNN | 让压缩 WSI 可直接卷积 |
| Spatial crops $R \times R \times C$ | 增加训练样本变化，降低 slide 数少带来的过拟合 |
| Tissue-aware sampling | 避开空白背景，提升训练效率 |
| 8-fold rotation/mirroring | 降低组织朝向依赖 |
| First-layer Grad-CAM | 在压缩网格上尽量保持定位分辨率 |
