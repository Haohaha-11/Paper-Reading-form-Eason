[← 返回 README](../README.md)

# 4 Experimental Setup

## 📌 预览
本节验证方法是否成立，重点看主结果、消融、效率和视觉/病例案例。

---

Setup. We conduct all experiments using 8 nodes of H100 GPUs, totaling 64 GPUs. For model training, we leverage the MS-Swift framework [Zhao et al., 2024] for its flexibility. Additionally, we utilize the DeepSpeed framework [Aminabadi et al., 2022], specifically the ZeRO-3 configuration, to optimize efficient parallel training across multiple nodes. Detailed hyperparameters are outlined in Appendix A.1.

> 💡 **批注**: 这段是 vision-language latent alignment 主线：关注视觉特征如何经 connector 进入 LLM 可解释的文本嵌入区域，以及这种约束如何影响文档理解、低资源训练和噪声鲁棒性。

Table 1: Main Results on General Document Benchmarks. We compare ALIGNVLM (ours) with state-of-the-art (SOTA) open and closed-source instructed models, and with base models that we trained using the process described in Section 3.3. ALIGNVLM models outperform all Base VLM models trained in the same data regime. Our models also perform competitively across document benchmarks even compared with SOTA models, in which the data regime is more targeted and optimized. Color coding for comparison: closed-source models ,

> 💡 **批注**: 这段是 vision-language latent alignment 主线：关注视觉特征如何经 connector 进入 LLM 可解释的文本嵌入区域，以及这种约束如何影响文档理解、低资源训练和噪声鲁棒性。

![Table 1](../images/2bc8157c77e9da75dd4033cd701b72fc2a86ebce9014ae26b134056ea091042a.jpg)
*Table 1: Table 1: Main Results on General Document Benchmarks. We compare ALIGNVLM (ours) with state-of-the-art (SOTA) open and closed-source instructed models, and with base models that we trained using the process described in Section 3.3. ALIGNVLM models outperform all Base VLM models trained in the same data regime. Our models also perform competitively across document benchmarks even compared with SOTA models, in which the data regime is more targeted and optimized. Color coding for comparison: closed-source models ,*

> 💡 **Table 1 批读**: 表格要看主指标、次指标与效率/鲁棒性是否一致支持论文 claim。

open-source models below 7B parameters , open-source models between 7-12B parameters .

Baselines. Our work focuses on architectural innovations, so we ensure that all baselines are trained on the same datasets. To enable fair comparisons, we evaluate our models against a set of Base VLMs fine-tuned on the same instruction-tuning tasks (Stages 2 and 3) as our models, using the BigDocs-7.5M and BigDocs-DocDownstream datasets. This approach ensures consistent training data, avoiding biases introduced by the Instruct versions of VLMs, which are often trained on undisclosed instruction-tuning datasets. Due to the scarcity of recently released publicly available Base VLMs, we primarily compare our model against the following Base VLMs of varying sizes: Qwen2-VL-2B [Wang et al., 2024], DocOwl1.5-8B [Hu et al., 2024], and LLama 3.2-11B [Grattafiori et al., 2024].

> 💡 **批注**: 这段按 AlignVLM 的 latent alignment 主线读：视觉 token 不是被普通 MLP 任意投影，而是被约束为 LLM 文本嵌入的加权组合；关键是这种语言先验是否提升文档元素、表格结构和 OCR 相关推理的可解释性。

For additional context, we also include results from the Instruct versions of recent VLMs of different sizes: Phi3.5-Vision-4B [Abdin et al., 2024], Qwen2-VL-2B and 7B [Wang et al., 2024], Qwen2.5- VL-7B [Qwen et al., 2025], LLaVA-NeXT-7B [Liu et al., 2024], InternVL2.5-2B and 8B [Chen et al., 2024b], InternVL3-2B and 8B [Zhu et al., 2025], Janus-1.3B [Wu et al., 2024a], DeepSeek-VL2-Tiny [Wu et al., 2024b], Ovis1.6-Gemma-9B [Lu et al., 2024], Llama3.2-11B [Grattafiori et al., 2024], DocOwl1.5-8B [Hu et al., 2024], and Pixtral-12B [Agrawal et al., 2024].

> 💡 **批注**: 这段按 AlignVLM 的 latent alignment 主线读：视觉 token 不是被普通 MLP 任意投影，而是被约束为 LLM 文本嵌入的加权组合；关键是这种语言先验是否提升文档元素、表格结构和 OCR 相关推理的可解释性。

Evaluation Benchmarks. We evaluate our models on a diverse range of document understanding benchmarks that assess the model’s capabilities in OCR, chart reasoning, table processing, or form comprehension. In particular, we employ the VLMEvalKit [Duan et al., 2024] framework and report the results on the following popular benchmarks: DocVQA [Mathew et al., 2021b], InfoVQA [Mathew et al., 2021a], DeepForm [Svetlichnaya, 2020], KLC [Stanisławek et al., 2021], WTQ [Pasupat and Liang, 2015], TabFact [Chen et al., 2020], ChartQA [Masry et al., 2022], TextVQA [Singh et al., 2019], and TableVQA [Kim et al., 2024].

> 💡 **批注**: 这段是 vision-language latent alignment 主线：关注视觉特征如何经 connector 进入 LLM 可解释的文本嵌入区域，以及这种约束如何影响文档理解、低资源训练和噪声鲁棒性。

Table 2: Impact of Connector Designs on VLM Performance: We present the results of experiments evaluating different connector designs for conditioning LLMs on visual features. Our proposed ALIGN connector is compared against a basic Multi-Layer Perceptron (MLP), the Perceiver Resampler, and Ovis. The results demonstrate that ALIGN consistently outperforms these alternatives across all benchmarks.

> 💡 **批注**: 这段按 AlignVLM 的 latent alignment 主线读：视觉 token 不是被普通 MLP 任意投影，而是被约束为 LLM 文本嵌入的加权组合；关键是这种语言先验是否提升文档元素、表格结构和 OCR 相关推理的可解释性。

![Table 2](../images/d34468df26187864fbfe15b1edccc79e6e5021d933c8d7dc426fe4897d7bf7c3.jpg)
*Table 2: Table 2: Impact of Connector Designs on VLM Performance: We present the results of experiments evaluating different connector designs for conditioning LLMs on visual features. Our proposed ALIGN connector is compared against a basic Multi-Layer Perceptron (MLP), the Perceiver Resampler, and Ovis. The results demonstrate that ALIGN consistently outperforms these alternatives across all benchmarks.*

> 💡 **Table 2 批读**: 表格要看主指标、次指标与效率/鲁棒性是否一致支持论文 claim。

---

## 🔖 Section 总结

### 核心洞察
1. 本节精读重点：把 AlignVLM 的 connector 设计、实验结论和文档理解场景联系起来看，尤其关注“对齐到语言嵌入凸组合”带来的收益与边界。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
