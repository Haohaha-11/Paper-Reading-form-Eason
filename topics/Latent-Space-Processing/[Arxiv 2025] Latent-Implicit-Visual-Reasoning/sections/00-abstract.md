[← 返回 README](../README.md)

## 📌 预览
摘要把 LIVR 的问题设定讲清楚：LMM 过度依赖文本推理，显式视觉中间监督成本高且带人工先验；LIVR 用无需中间标签的 latent visual reasoning tokens 做任务自适应视觉重编码。

---

# Latent Implicit Visual Reasoning

Kelvin Li1∗ Chuyi Shang1∗ Leonid Karlinsky2 Rogerio Feris3 Trevor Darrell1 Roei Herzig1,3 1University of California, Berkeley 2 Xero 3MIT-IBM Watson AI Lab

# Abstract

While Large Multimodal Models (LMMs) have made significant progress, they remain largely text-centric, relying on language as their core reasoning modality. As a result, they are limited in their ability to handle reasoning tasks that are predominantly visual. Recent approaches have sought to address this by supervising intermediate visual steps with helper images, depth maps, or image crops. However, these strategies impose restrictive priors on what “useful” visual abstractions look like, add heavy annotation costs, and struggle to generalize across tasks. To address this critical limitation, we propose a task-agnostic mechanism that trains LMMs to discover and use visual reasoning tokens without explicit supervision. These tokens attend globally and re-encode the image in a task-adaptive way, enabling the model to extract relevant visual information without hand-crafted supervision. Our approach outperforms direct fine-tuning and achieves state-of-the-art results on a diverse range of vision-centric tasks – including those where intermediate abstractions are hard to specify – while also generalizing to multi-task instruction tuning.

> 💡 **问题动机**: 这段把 LIVR 的核心矛盾压缩成一句话：LMM 虽然看图，但推理通道主要还是语言。对视觉相似、对应、拼图这类任务，很多中间结构不是一句自然语言能稳定表达的。

---

## 🔖 Section 总结

> 💡 **Section 小结**:
> - 核心变量: visual reasoning tokens / latent tokens、task-adaptive re-encoding、end-task answer loss。
> - 核心洞察: LIVR 把视觉推理从“写出中间文字”转为“学习不可见但可优化的视觉 latent”。
> - 可追问点: latent token 是否真的有因果作用，还是只是额外参数容量?
