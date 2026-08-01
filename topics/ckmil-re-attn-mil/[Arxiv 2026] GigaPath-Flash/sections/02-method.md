[← 返回 README](../README.md)

# 02 Method

## 2.1 GigaPath-Flash: 高效全切片编码

> 📄 **原文 - 2.1 Model**

GigaPath-Flash is an efficient foundation model that turns a gigapixel pathology slide into a spatially contextualized whole-slide representation. It has two components: a ViT-S/16 tile encoder (22M parameters) that maps each 224x224 px tile to a 384-dimensional embedding, and a 12-layer LongNet [4] slide encoder (21M parameters, 384 dimensions) that jointly contextualizes all tile embeddings via dilated attention, scaling linearly with the number of tiles. The tile encoder is distilled from the frozen GigaPath ViT-g (1B) teacher with a DINOv2 [5] objective on whole-slide images from the Providence real-world cohort, transferring the representational capacity of the billion-parameter model into a backbone an order of magnitude smaller. We follow the standard DINOv2 recipe but omit the KoLeo regularization term, which we found to destabilize training when distilling into this compact student. The slide encoder is then pretrained on the resulting tile features with a masked autoencoding objective, learning to predict a tile's representation from the rest of the slide and thereby encoding each location in the context of the whole.

> 💡 **Hao 批注 - DINOv2 蒸馏的技术选择**: 
> 
> DINOv2 蒸馏的核心思想：让 student ViT-S 学习模仿 teacher ViT-g 在相同输入上的特征输出。DINOv2 使用以下几个 loss 项：
> - **CLS token 对比损失**（student 和 teacher 的 CLS token 应相似）
> - **Patch token 对比损失**（对应位置的 patch 特征应匹配）
> - **KoLeo 正则化**（鼓励特征在球面上均匀分布）
> 
> 去掉 KoLeo 的原因可能是：小型 student 的表示能力受限，强行要求特征均匀分布会挤压任务相关维度的表示质量。但这是一个经验发现而非理论推导，在其他蒸馏场景（如蒸馏到 ViT-Tiny）可能需要重新评估。
>
> **Masked Autoencoding 的 slide encoder 预训练**：随机 mask 一些 tile 的特征，让 LongNet 从其他 tile 预测这些 masked tile 的特征——这迫使 LongNet 学习"每个位置在整张切片语境下的意义"，而非简单的特征聚合。这与 MAE 的思路一致但作用在预提取的 tile 特征上（更快，因为不需要反复前向 tile encoder）。

## 2.2 GigaTIME-Flash: 高效空间蛋白质组预测

> 📄 **原文 - 3.1 Model**

GigaTIME follows a UNet++ architecture [16] which takes in a H&E patch as an input and predicts 21 channel multiplex immunofluorescence maps across protein markers - as output. UNet++ has a convolutional encoder and decoder framework with skip connections. In the supplementary material of [3], we had shown that when we replace the encoder of GigaTIME with GigaPath weights and train it for mIF prediction we do get an improvement in performance with a trade-off in the number of parameters and inference time which would be inefficient while doing any population-scale inference with the model.

As GigaPath-Flash is an efficient distilled model compared to the full GigaPath encoder, we introduce GigaTIME-Flash, where the encoder is initialized from the distilled GigaPath-Flash ViT-small checkpoint and paired with a lightweight convolutional decoder for H&E-to-mIF translation. The encoder is a 12-layer ViT-small backbone with patch size 16, embedding dimension 384, and 6 attention heads. For a 256x256 H&E input tile, this produces a 16x16 grid of patch tokens. Skip features are extracted from the 4th, 6th, 9th, and 12th transformer blocks. The deepest feature from block 12 initializes the decoder, while features from blocks 9, 6, and 4 are injected into successive decoder stages through learned transposed-convolution skip projections. The decoder follows a U-Net-style convolutional design. The block-12 token map is reshaped into a spatial feature map and passed through four decoder stages with channel dimensions 384→192→96→48→24. Each decoder stage contains two 3x3 convolution layers with ReLU activations, followed by bilinear upsampling by a factor of 2. The skip paths project encoder features using transposed convolutions: the block-9 feature is projected to 192 channels, the block-6 feature to 96 channels, and the block-4 feature to 48 channels before being added to the corresponding decoder activations. A final 1x1 convolution maps the 24-channel decoder output to the 21 mIF output channels.

To adapt the encoder while limiting catastrophic forgetting, we fine-tune using LoRA [17] adapters on the transformer attention qkv and output projection proj modules. The LoRA configuration uses rank r=8, scaling parameter α=16, and dropout 0.1. Non-LoRA encoder parameters are frozen, while the LoRA adapters and the convolutional decoder are trainable. Overall, GigaTIME-Flash contains 23,806,559 parameters. The model is trained for 21-channel mIF prediction using BCEDice loss, Adam optimization with learning rate 10^-4 and weight decay 10^-4, and a cosine annealing learning-rate schedule with minimum learning rate 10^-5. GigaTIME-Flash was trained for 300 epochs on a NVIDIA A100 GPU node with batch size of 64 per GPU on the same GigaTIME training data for fairness [3].

> 💡 **Hao 批注 - ViT-S 作为 UNet 编码器的 skip connection**: 这是 GigaTIME-Flash 最有技术含量的设计细节。传统 ViT 是序列化的无层次结构，但 ViT-S 的 12 层 Transformer 内部仍然保留了空间结构——因为从 patch embedding 层开始每个 token 就与特定空间位置绑定。作者选择 block 4/6/9/12（均匀分布在浅层到深层）作为 skip 连接点，使 decoder 可以融合多级语义：浅层保留细粒度纹理（核形态），深层编码语义概念（组织类型）。
>
> 💡 **Hao 批注 - LoRA 的选择**: r=8 在 384 维特征空间中相对较小（约 2%），意味着仅对 encoder 做非常轻微的 task-specific 适配——这合理，因为基础 ViT-S 已经在蒸馏时学到了丰富的组织形态表示。如果 r 过大，可能破坏预训练表示中的通用特征（灾难性遗忘），如果过小则适配不足。

> 💡 **Hao 批注 - 方法总结**: GigaPath-Flash 的技术栈可以分解为三个独立但协同的组件：
> 1. **Tile-level**: 蒸馏 ViT-S (DINOv2, 无 KoLeo) —— 提供廉价且高质量的瓦片特征
> 2. **Slide-level**: LongNet MAE pretrain —— 在瓦片特征之上学习全切片空间上下文
> 3. **TME-level**: ViT-S + LoRA + Conv Decoder —— 复用蒸馏编码器做跨模态预测 (H&E→mIF)
>
> 这三个组件的共同使能者是：蒸馏的高质量 ViT-S 编码器。这是典型的 "build once, use many" 设计哲学。
