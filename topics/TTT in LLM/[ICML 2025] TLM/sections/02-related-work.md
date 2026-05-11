[← 返回 README](../README.md)

# 2. Related Work

The adaptability of deep learning models to dynamic and diverse real-world environments has emerged as a prominent focus in recent research. Various methods have been proposed to enhance model performance under distributional shifts, as summarized in Table 1.

Fine-Tuning adapts pre-trained models to specific tasks or domains by updating their parameters, such as LoRA, with labeled data (Hu et al., 2022; Thirunavukarasu et al., 2023; Chen et al., 2024c; Wang et al., 2025b). This approach allows models to specialize in domain-specific tasks by leveraging transfer learning to refine their capabilities.

> 💡 **与 fine-tuning 的边界**: TLM 不引入 labeled target pairs，因此不能期待它学习新任务标签映射；它主要让模型参数贴近当前无标签输入分布。

However, it is often constrained by the need for extensive labeled datasets and high computational costs, which limit its practicality in dynamic environments where data distributions are continuously evolving. In contrast, our work aims to dynamically update the model at test-time using unlabeled input data, eliminating the need for extensive labeled datasets and addressing the challenges posed by evolving data distributions.

Retrieval-Augmented Generation (RAG) incorporates external knowledge by retrieving relevant information from a knowledge base during inference (Jiang et al., 2024; Qian et al., 2024; Asai et al., 2024). This allows the model to generate more accurate and contextually grounded responses without requiring parameter updates. Qian et al. (2024) propose MemoRAG, a retrieval-augmented generation framework enhanced by long-term memory for improved task performance. RAG is effective for tasks requiring up-todate or domain knowledge but relies heavily on the quality of retrieved information and incurs additional computational latency, limiting its suitability for time-sensitive tasks.

> 💡 **与 RAG 的边界**: RAG 改上下文、TLM 改参数；TLM 不依赖外部知识库，但也因此无法凭空补充测试输入中没有出现的新事实。

Test-Time Adaptation (TTA) dynamically updates model parameters during inference by utilizing unlabeled test data (Wang et al., 2021; Niu et al., 2022a; 2023; Chen et al., 2024b;a; Liang et al., 2024; Yi et al., 2024). This approach enables real-time adaptation to distributional shifts, making it suitable for scenarios where labeled data is unavailable or the test data distribution deviates significantly from the training distribution. Wang et al. (2021) propose the test entropy minimization method, which improves model confidence by minimizing prediction entropy through online updates of normalization statistics and affine transformations. Most TTA methods rely on entropy minimization, but this approach is not well-suited for the dynamic updates required by LLMs, as shown in Figure 1a. To address this issue, we propose minimizing the perplexity of test samples as the optimization objective, which effectively enhances the performance of LLMs in dynamic environments.

Test-Time Training (TTT) retrieves relevant data from the training set or a knowledge base during inference and uses it to fine-tune the models (Niu et al., 2022b; Hardt & Sun, 2024; Hubotter et al.¨ , 2024). This allows the model to leverage adjacent data to better adapt to current test inputs, improving its performance in dynamic scenarios. Hardt & Sun (2024) propose a test-time training approach for LLMs by fine-tuning the model on retrieved nearest neighbors from a large-scale text embedding index. Hubotter et al. ¨ (2024) propose SIFT, a data selection algorithm that optimizes information gain to outperform Nearest Neighbor retrieval for test-time fine-tuning with low computational overhead. However, TTT assumes that the model’s training data or knowledge base is accessible during deployment, and the retrieval process introduces computational overhead, which is often impractical. Unlike TTT, our focus is on TTL, which dynamically updates LLMs during test time using only unlabeled test data.

> 💡 **与 TTA 的关键差异**: 论文把 LLM 的自回归建模作为反驳 entropy minimization 的依据，因此选择 perplexity 而非分类置信度作为 test-time loss。

> 💡 **与 TTT 的关键差异**: Hardt & Sun / SIFT 这类 TTT 需要检索训练邻居或知识库，TLM 的约束更强：只使用当前测试域的 unlabeled input。
