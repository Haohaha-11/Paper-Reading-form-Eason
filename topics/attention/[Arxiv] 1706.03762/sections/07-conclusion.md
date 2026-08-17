[← 返回 README](../README.md)

# 7 Conclusion

## 📌 预览

结论收束全文：Transformer 是**第一个完全基于注意力**的序列转换模型，用多头自注意力替代了 encoder-decoder 里最常用的循环层；翻译上更快更强、双料 SOTA；并展望把注意力推广到图像/音频/视频等其他模态，以及让生成不那么串行。

---

## 7 Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

> 💡 **贡献重述**: 一句话锁定核心贡献——Transformer 是**第一个完全基于注意力**的序列转换模型，把 encoder-decoder 里最常见的循环层替换成多头自注意力。呼应第 2 节的 novelty claim，首尾闭环。

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

> 💡 **结果重述**: 把"快"和"强"两个卖点再钉一遍——训练显著快于 RNN/CNN 架构；英德、英法双双 SOTA；英德甚至超过所有已报告的 ensemble。这正是摘要与 Table 2 的浓缩。

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

> 💡 **未来展望（回看极具预见性）**: 作者列了三个方向：① 推广到**文本以外的模态**（图像/音频/视频）；② 研究**局部/受限注意力**以高效处理超大输入输出（呼应第 4 节的 $O(n^2)$ 软肋与 restricted attention）；③ 让**生成不那么串行**（非自回归生成）。从 2026 年回看，这三条几乎全部成真——ViT、语音/视频 Transformer、稀疏注意力、非自回归解码分别对应，这段展望的命中率极高。

The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor.

> 💡 **可复现性**: 开源在 tensor2tensor 仓库。这在 2017 年是加分项，也是 Transformer 能迅速被社区复现、扩散的重要原因之一。

Acknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.

---

## 🔖 Section 总结

### 核心洞察
1. **首尾闭环**：结论把"第一个纯注意力模型 + 双 SOTA + 更快"三点收束，与摘要/引言完全一致。
2. **展望极准**：跨模态、稀疏/局部注意力、非自回归——三条未来方向在此后数年逐一实现。
3. **开源助推扩散**：tensor2tensor 的开源降低了复现门槛。

### 可追问点
- "restricted attention" 未来方向如何演化？（→ Sparse Transformer、Longformer、Linear Attention 等一整条研究线）
- 论文只在翻译 + 句法分析验证，为何能扩散到几乎所有序列任务？（QKV + 位置编码是与领域无关的通用组件，这是可迁移性的根源）
