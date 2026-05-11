[← 返回 README](../README.md)

# 6. Conclusion

In this paper, we propose a novel Test-Time Learning (TTL) method for Large Language Models (LLMs), named TLM, to address the challenges posed by dynamic and diverse data distributions in real-world environments. By leveraging only unlabeled test data, TLM efficiently adapts LLMs and improves their robustness in target domains. Specifically, through observation and theoretical analysis, we argue that reducing output perplexity can be achieved by minimizing the input perplexity of unlabeled test data. Based on this insight, we adopt input perplexity minimization as the optimization objective for test-time LLM updates. Moreover, we find that high-perplexity test samples play a more crucial role in model updates than low-perplexity samples. This insight motivates the development of our Sample Efficient Learning Strategy, which actively selects and emphasizes high-perplexity test samples for backpropagation, optimizing the learning process. Lastly, we reveal that Low-Rank Adaptation is more effective than full parameter updates in mitigating catastrophic forgetting, and we utilize it for parameter updates during TTL, enabling lightweight model adaptation while effectively preserving prior knowledge.

> 💡 **结论边界**: 论文证明的是无标签测试数据可用于轻量域适应，并非说明所有部署场景都应该反向传播；安全性、延迟和数据污染仍是实际系统门槛。
