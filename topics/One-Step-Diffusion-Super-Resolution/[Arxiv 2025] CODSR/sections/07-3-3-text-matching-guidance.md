[← 返回 README](../README.md)

# 3.3. Text-matching guidance

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 CODSR 主线的关系**: CODSR 从 LQ 信息损失、区域差异化先验激活和文本-区域错配三方面补强 controllable one-step diffusion SR，在单步推理下兼顾感知质量与局部保真。

---

To strengthen text-image alignment and effectively leverage the conditioning potential of text prompts, we propose a text-matching guidance (TMG) strategy. It employs Grounded-SAM2 [34], an open-vocabulary segmentation model, to derive region maps corresponding to text prompts. These maps provide explicit spatial guidance for modulating the interaction of text conditions with the latent features within the U-Net, ensuring that the textual conditioning is applied to semantically relevant image areas.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

Specifically, we employ NLTK [3] to remove adjectives from the prompt extracted by RAM [68], as they are too abstract for precise spatial guidance. The remaining nouns $\{ n ^ { 1 } , \ldots , n ^ { N } \}$ , together with $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ , are then fed into Grounded-SAM2 [34] to generate $\{ M ^ { 1 } , \ldots , M ^ { N } \}$ , where $M ^ { i }$ indicates the binary region mask corresponding to the noun $n ^ { i }$ . To ensure the effectiveness of the guidance, we perform a validity check on $M ^ { i }$ and discard it if the number of active pixels is below a predefined threshold. These masks explicitly define the regions for text interaction during the reverse process. We then achieve the text-matching interaction between $n ^ { i }$ and $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ via each cross-attention layer in the U-Net, and produce the averaged attention map $A ^ { i }$ across all layers that correspond to the noun $n ^ { i }$ , supervised by the positive area loss from CoMat [19].

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

---

## 🔖 Section 总结

### 核心洞察

1. 明确输入、输出、teacher/student 或控制变量。
2. 把每个 loss/模块对应到 fidelity、realism、speed 或 controllability。
3. 关注哪些组件是训练时使用，哪些是推理时仍有成本。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Core controls | region-adaptive prior activation + text-matching guidance |
| Project | https://github.com/Chanson94/CODSR |
| Benchmarks | RealSR/DrealSR and other real-world SR test sets in paper tables |
