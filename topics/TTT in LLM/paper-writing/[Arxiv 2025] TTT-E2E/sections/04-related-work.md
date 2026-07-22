[← 返回 README](../README.md)

# 4 Related Work

## 📌 预览
本节把 TTT-E2E 放进 continual learning、test-time training、fast weights 和 meta-learning 谱系，并澄清它不是单纯 KV memorization。


# 4.1 Continual Learning

Most of today’s AI systems remain static after deployment, even though the world keeps changing. The high-level goal of continual learning is to enable AI systems to keep changing with the world, similar to how humans improve throughout their lives [36, 22].
> 💡 **相关工作定位**: 作者先从 continual learning 起笔，是为了强调本文不是纯架构论文：模型在测试序列上持续改变，long context 被视作部署后的连续经验流。


Conventionally, continual learning as a research field has focused on learning from a distribution that gradually changes over time [68, 96, 33]. For example, one could update a chatbot model every hour using new knowledge from the Internet, while typical use cases of the model may require knowledge from both the past and the present [80, 56, 100]. More formally, at each timestep, we sample new training and test data from the current distribution, update our model using the new training data, and then evaluate it on all the test data up to the current timestep. Under this setting, most algorithms focus on not forgetting the past when learning from the present [75, 64, 57, 30].

# 4.2 Test-Time Training

The algorithmic framework of test-time training has the same high-level goal as continual learning, but it focuses on two aspects where human learning stands out from the forms of continual learning in the conventional literature.

First, each person has a unique brain that learns within the context of their individual life. This personalized form of continual learning is quite different from, for example, the chatbot model that is fine-tuned hourly using the latest information available worldwide. While such a model does change over time, it is still the same at any given moment for every user and every problem instance.

Second, most human learning happens without a boundary between training and testing. Consider your commute to work this morning. It is both "testing" because you did care about getting to work this very morning, and "training" because you were also gaining experience for future commutes. But in machine learning, the train-test split has always been a fundamental concept.

The concept of test-time training is introduced to realize these two special aspects of human learning. Training typically involves formulating a learning problem (such as empirical risk minimization) and then solving it. Following [86], test-time training is defined as any kind of training that formulates a potentially different learning problem based on each individual test instance.
> 💡 **TTT 定义**: 这里的 TTT 不是 prompt-time trick，而是真正构造 test-instance-specific learning problem 并更新模型参数。TTT-E2E 的学习问题就是给定 context 上的 next-token prediction。


This concept has a rich history in AI. A well-known example in NLP is dynamic evaluation, pioneered by Mikolov et al. [72] and extended by Krause et al. [60], which our Subsection 2.1 builds upon. In computer vision, early examples have also emerged in applications such as face detection [52], video segmentation [73], super-resolution [83], and 3D reconstruction [70]. Next, we discuss three popular forms of test-time training today, with an emphasis on their connections to each other and to historical examples.

# 4.2.1 TTT on Nearest Neighbors: Larger Effective Capacity

One simple form of test-time training was called locally weighted regression in the 1970s [85, 18], local learning in the 1990s [12], and KNN-SVM in the 2000s [109]: Given a test instance, find its nearest neighbors in the training set, and then train (or fine-tune) the model on these neighbors before making a prediction. This procedure can significantly increase the effective capacity of the model; for example, it allows a linear model to fit a highly nonlinear ground truth [85].

This simple form captures one of the key intuitions of test-time training. In the conventional view of machine learning, a model, once trained, no longer changes at test time. As a consequence, it must prepare to be good at all possible inputs in the future. This task can be very hard, because being good at all possible futures limits the model’s capacity to be good at any particular one. But only one future is actually going to happen. So why not train our model once this future happens?

Recently, [35] extended this idea to modern language models and observed a similar benefit of larger effective model capacity after test-time training, and [45] further improved these results through better strategies for neighbor selection. In addition, [46] showed that test-time training on neighbors from the training set is also effective with RL for reasoning tasks, and [5] developed the same idea for visual-motor tasks.

# 4.2.2 TTT for Novel Instances: Better Generalization

As models become larger today, their competence is often limited not by their capacity, but by the amount of available training data, especially when they need to generalize to novel test instances that are “out-of-distribution”. In this case, it is even harder to prepare for all possible test instances in the future, especially the novel ones, with a static model. But once a specific test instance is given, we can use it to generate relevant data, which we can then use for training [88]. In other words, the “neighbors” for TTT do not have to come from the training set; they can also be generated on-the-fly.

Since the test instance is unlabeled, one way to make it useful for training is through self-supervision, which generates new pairs of inputs and labels for an auxiliary task such as masked reconstruction (e.g., BERT [23] and MAE[37]). While the auxiliary task is different from the main prediction task, improving performance in one can help the other through their shared representations. This form of TTT can significantly improve generalization under distribution shifts [88, 28].

Recently, TTT has been an important part of AlphaProof [44], which achieved IMO silver-medal standard in 2024. Given each test problem, their system first generates a targeted curriculum of easier problems by prompting a language model, and then performs reinforcement learning on the generated data. Another recent work, Akyurek et al. [2], found TTT effective for few-shot reasoning tasks such as ARC-AGI. Their system generates augmentations of the few-shot demonstrations in the test problem then performs supervised learning.

# 4.2.3 TTT on Sequences: Longer Memory

In all the forms of TTT discussed so far, the model is reset after each prediction because the test instances are independent. However, humans do not constantly reset their minds. Our memory of how to solve the previous learning problem often helps with the current one, because our experience in the world is much closer to a correlated sequence of data than independent ones.
> 💡 **序列 TTT 的特殊性**: 常见 TTT 每个样本后 reset；长文本序列不能 reset，因为前面 token 的学习状态正是后面预测要用的上下文压缩。


Sequential applications, such as videos and robotics, offer a playground that bridges this difference. For example, [34] extended TTT with self-supervision to a manipulation policy whose input is a video stream of the robot’s workstation, and found that no reset leads to a much larger improvement. Recently, [101] extended the same idea to video segmentation using a model trained with only images. In this case, TTT can be viewed as compressing the context from previous frames into the weights of the model without learning to learn, similar to the naive version of our method in Subsection 2.1.

TTT-KVB. Text, like videos, is a form of sequence. In Subsection 2.4, we have discussed TTT-KVB as the most relevant line of prior work [87, 110, 19], which includes variants such as MesaNet [98], Titans [7], and Nested Learning [6]. The popularity of TTT-KVB has two side effects:

• Because the KVB objective is inspired by self-attention, which stores the keys and values, many think that long-context TTT is about memorization instead of generalization. • Because TTT(-KVB) layers are drop-in replacements for self-attention layers, many also think of long-context TTT as an approach to architecture design.

Our work shows that long-context TTT does not need to memorize the association between the keys and values. In addition, our method is derived purely under the formulation of a continual learning problem, with minimal changes to the architecture.
> 💡 **反 KVB 误读**: 作者特别澄清：long-context TTT 不一定是 KV memorization，也不一定是替代 attention layer 的 architecture design；TTT-E2E 是 continual learning formulation。


# 4.3 Fast Weights and Fast Weight Programmers

The general idea of fast weights is to update the parameters of a “fast” model on only the most relevant data, as opposed to the conventional practice of updating a “slow” model on all data [94]. This idea has existed since the 1980s [26, 38, 97]. Because the most relevant data can often include the test instance itself, test-time training can be viewed as a special case of fast weights, with a heavier emphasis on the formulation of an explicit learning problem.

The general idea of fast weight programmers (FWPs) is to update the fast weights at test time with a “slow” model (as a programmer) that, in turn, is updated less frequently, if at all [79]. In our method, the inner-loop weights W can be viewed as “fast” and the outer-loop weights $\theta$ as “slow”. Therefore, our method can be viewed as a special case of FWPs [58]. Next, we briefly review some of the literature on FWPs in the order of relevance.
> 💡 **fast weights 视角**: TTT-E2E 可读作 fast weights programmer：测试时更新的 MLP 是 fast weights，训练时学到的初始化/外层参数是 slow weights。这个视角解释了“weights as context compression”。


Clark et al. [17]. This work is the most relevant to ours in methodology. Given a Transformer baseline with full attention, they add an MLP layer as fast weights, whose initialization is trained as slow weights along with the rest of the model. Similar to ours, their method updates the fast weights by taking a gradient step on the next-token prediction loss computed over each chunk (mini-batch) of tokens. Their method significantly improves perplexity compared to the baseline but does not improve efficiency, since their combined architecture does not have linear complexity. In addition, their design adds the fast weights only to the end of the model instead of interleaving them with attention layers. In our experiments, interleaving proves to be critical for maintaining the performance gain on top of larger baselines. Nevertheless, we find Clark et al. to be a valuable inspiration. An earlier work [102] also contains sketches of a similar idea with limited experiments.
> 💡 **最接近前作差异**: Clark et al. 也用 next-token loss 更新 fast weights，但叠在 full attention 上，因此没有线性复杂度；TTT-E2E 把 fast weights 与 SWA 交织，目标是同时拿到性能和效率。


FWPs for long context. Many methods addressing the problem of long context have roots in the literature of FWPs. In particular, [79] (Schmidhuber, 1992) has been a major source of inspiration for modern RNN layers, such as linear attention [55, 77], DeltaNet [76, 106], and Gated DeltaNet [105], one of our baselines. In addition, some of the work on TTT for long context [87, 110] (discussed in Subsection 4.2) can also be viewed as FWPs, due to the connection between TTT and fast weights. Notably, one instantiation in Irie et al. [49] uses MLPs as layer-wise fast weights for long context, preceding the similar instantiation in [87].

Other FWPs. While the FWPs above can be interpreted through TTT, many other varieties cannot. For example, [50] designs the fast weights to be programmed by themselves, [51] builds an image generator using the images as fast weights, [48] applies continuous-time extensions of FWPs to time-series classification, while [47] and [31] demonstrate how the choice of update rules affects the expressiveness of FWPs on formal language recognition tasks. In fact, all networks with some gating mechanism, such as Transformers with SwiGLU blocks [82], can also be viewed as FWPs [32].

# 4.4 Learning to Learn

For decades, researchers have been arguing that learning to learn, also known as meta-learning or bi-level optimization, should be an important component of intelligence [78, 9, 93, 61]. Perhaps the most relevant work in this field is MAML [27]. Similar to our work, MAML also has an outer loop that learns the initialization of the inner loop through gradients of gradients. The main difference between MAML and our work lies in the problem setting. Specifically, their inner loop learns from an entire dataset at a time, so the outer loop requires a large collection of datasets. In contrast, our work addresses the problem of language modeling by casting it as learning to learn. In principle, any supervised learning problem can be cast into our problem formulation.

> 💡 **meta-learning 差异**: 与 MAML 相同点是通过二阶梯度学习初始化；差异是这里没有一组显式小任务数据集，而是把语言序列本身切成内外循环的 continual learning 轨迹。

---

## 🔖 Section 总结

- **谱系**: continual learning 提供问题表述，TTT 提供测试实例内训练框架，fast weights 提供权重作为动态记忆的解释，meta-learning 提供初始化学习机制。
- **与前作差异**: 不把 long-context TTT 等同于 KVB memorization，也不把它主要包装成替代 attention 的新架构。
