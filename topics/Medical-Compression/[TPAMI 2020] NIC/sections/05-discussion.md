[← 返回 README](../README.md)

# 05 - Discussion

## 📌 预览

Discussion 回到机制解释：为什么 BiGAN 在真实病理任务中更强，NIC 对未知图像级标签意味着什么，以及它在小病灶、I/O 成本、训练样本数上的边界。

---

# 5 DISCUSSION

Our experimental results support the hypothesis that visual cues associated with weak image-level labels can be exploited by our method, integrating information from global structure and local high-resolution visual cues. Furthermore, we have shown that this methodology is flexible and completely label-agnostic, delivering relevant results for both classification and regression tasks in synthetic as well as histopathological data. It emerges as a promising strategy to tackle the analysis of more challenging imagelevel labels that are closely related to patient outcome, e.g., overall survival and recurrence-free survival. Gigapixel NIC paves the way for leveraging existing computer vision algorithms that could not be applied in the gigapixel domain until now, such as image captioning (useful to generate written clinical reports), visual question answering, image retrieval (to find similar pathologies), anomaly detection, and generative modeling [37]–[41].

> 💡 **claim 范围**: 作者最强的主张不是“超过有像素标注的专用系统”，而是“任意 image-level label 可以先投到 latent WSI，再用标准 CNN/视觉算法处理”。

A key assumption in our method was that highresolution image patches could be represented by lowdimensional highly compressed embedding vectors. We analyzed several unsupervised strategies to achieve such a compression and found that the BiGAN encoder, trained using adversarial feature learning, was superior to all other methods across all experiments with histopathological data. We believe that this relative improvement with respect to the VAE and contrastive methods is explained by intrinsic algorithmic differences among the methods. In particular, the VAE model relies on minimizing the MSE objective, which is a unimodal function that fails to capture highlevel semantics; it focuses on reconstructing low-level pixel information instead, wasting embedding capacity. On the other hand, the contrastive encoder uses the embedding capacity more efficiently, but its performance is driven by the design of the hand-engineered contrastive task. Remarkably, the BiGAN model learns an encoder that fully automatically inverts a complex mapping between the latent space and the image space. By doing so, the encoder benefits from all the high-level features and semantics already discovered by the generator, producing very effective discriminative embedding vectors. Furthermore, BiGAN achieved the best classification accuracy on the challenging blood, mucus, and necrotic tissue classes that rarely appear in the Camelyon16 and TUPAC16 WSIs. We hypothesize that the adversarial method can model these rare data modes more effectively than the contrastive or VAE approaches. Nevertheless, we believe that the choice of encoder may be data-dependent, since the contrastive encoder outperformed the other approaches in the synthetic dataset.

> 💡 **BiGAN 优势解释**: 这段是全文最重要的 encoder 讨论。VAE 的瓶颈被 MSE 重建占用；contrastive 的语义来自手工 pair 规则；BiGAN 通过反演 generator latent，可能更自然地保留组织模式和罕见类别。

We trained a CNN to predict the breast tumor proliferation speed based on gene-expression profiling, a label associated with unknown visual cues. Our method succeeded in finding and exploiting these patterns in order to predict expected tumor proliferation speed, surpassing the current state-of-the-art for image-level based methods. This shows that our method constitutes an effective solution to deal with gigapixel image-level labels with unknown associated visual cues. Moreover, our method could be used in future works to effectively mine datasets with thousands of gigapixel images [11]; other automatically generated labels from immunohistochemistry, genomics, or proteomics can be targeted, and visual patterns beyond the knowledge of human pathologists may be discovered.

> 💡 **未知视觉线索的意义**: TUPAC16 的标签不是“某个 pathologist 能画出来的对象”。NIC 的价值在于把 genomics/proteomics 等 slide-level 标签接到 WSI 视觉空间里，用模型找潜在形态学 correlates。

For the first time, the regions of a gigapixel image that a trained CNN attends to when predicting image-level labels were visualized, and the effect of different encoding methods was compared. We discovered that only the CNNs trained with images compressed with the BiGAN encoder and the supervised-tumor baseline were able to attend to regions of the image where tumor cells were present. The fact that the BiGAN model simultaneously learned to delimit metastatic lesions and identify tumor features within the patch embeddings validates our hypothesis that CNNs are an effective method for analyzing gigapixel images, i.e., since they can exploit both global and local context.

> 💡 **where claim 的证据边界**: “attend to tumor regions” 是定性 Grad-CAM 证据，不等同于可部署分割。它验证模型利用了合理区域，但 failure case 说明还不能把热图当诊断标注。

We targeted the presence of tumor metastasis in breast lymph nodes and showed that the BiGAN setup performed similarly to the supervised baseline. However, our bestperforming algorithm was still inferior to that of the Camelyon16 leadingboard (0.9935 AUC using accurate pixel-level annotations). This performance gap is likely due to two factors. First, the majority of the images marked as positive contain tumor lesions comprised of only a few tumor cells (i.e., micro-metastasis), becoming almost undetectable with the compression setup tested in this work (see Fig. 8). Second, the lack of training data (only a few hundred training images) may lead the CNN into the overfitting regime.

> 💡 **失败机制**: micro-metastasis 对 NIC 是双重难题：压缩时病灶可能只影响少数 patch embedding；训练时图像级 positive signal 又被整张 WSI 的海量背景稀释。

We acknowledge several limitations of our method. For one, it requires a substantial amount of I/O throughput and storage due to the need to write compressed WSI representations to disk before training, and repetitively read them to assemble mini-batches during training. This computational burden prevented us from performing a wide hyper-parameter value search, which may have resulted in a suboptimal parameter selection. Second, it was also observed that the method’s performance was proportional to lesion size. In particular, it struggled to detect micrometastasis in Camelyon16 data, i.e., tumor lesions smaller than $2 \mathrm { m m }$ , limiting the applicability of NIC to tasks with large lesions.

> 💡 **工程瓶颈**: NIC 把 GPU 显存问题转成了存储/I/O 问题。compressed WSI 虽比原图小得多，但训练时仍需频繁读大网格 crop，限制了超参搜索。

This method can be extended in multiple ways. More sophisticated encoders may improve the low-dimensional representation of the image patches [16], [42], [43]. Incorporating attention mechanisms may make it easier for the CNN to attend to relevant regions for the image-level labels [44], improving the detection of small lesions. Finally, gradient checkpointing [45] could be used to backpropagate the training signal from the image-level labels towards the encoder weights.

> 💡 **未来方向具体化**: 三个方向分别对应 encoder、aggregator、end-to-end training。今天复现或扩展时，最自然的是用现代 self-supervised histopathology encoder 替换 BiGAN，并加入 attention 处理稀疏小病灶。

## 🔖 Section 总结

1. BiGAN 的优势来自 adversarial latent inversion 更偏高层语义，尤其可能覆盖罕见组织模式。
2. NIC 更适合大区域或全局形态相关标签；对 micrometastasis 这种极稀疏信号，压缩和弱监督都会损失灵敏度。
3. 论文的未来路线已经很清楚：更强 encoder、attention、更端到端的 image-level supervision。
