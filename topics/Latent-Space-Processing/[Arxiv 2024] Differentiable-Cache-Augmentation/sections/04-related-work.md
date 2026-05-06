[← 返回 README](../README.md)

## 📌 预览
相关工作把 DCA 放在 CoT、latent reasoning、kv-cache compression、external modules 和 hypernetworks 的交叉处。

---

# 4. Related Work

# 4.1. Chain-of-Thought Reasoning in LLMs

Limitations in eliciting complex reasoning from LLMs through standard prompting have motivated research into prompting strategies that encourage intermediate reasoning steps. Chain-of-Thought (CoT) prompting (Wei et al., 2022) significantly improved reasoning performance by prompting LLMs to “think step by step”. Subsequent work explored zero-shot CoT (Kojima et al., 2022; Zhou et al., 2023), aggregating multiple reasoning paths (Wang et al., 2022), internalizing the intermediate reasoning steps (Deng et al., 2024), verifying generation steps (Lightman et al., 2023), and broader search spaces for reasoning trajectories, such as Tree-of-Thought (Wang and Zhou, 2024; Yao et al., 2024). Other approaches leverage reinforcement learning (RL) to optimize the reasoning process based on final answer accuracy or target text likelihood (e.g., StaR (Zelikman et al., 2022), TRICE (Hoffman et al., 2024), Quiet-STaR (Zelikman et al., 2024)). While effective, these methods are often constrained by the expressiveness of natural language and can be computationally expensive due to the sequential generation of reasoning steps, both during training and inference.

> 💡 **CoT 对比**: DCA 与 CoT 的差别不是“有没有中间步骤”，而是中间步骤在什么空间里发生。CoT 在 token space、可读但慢；DCA 在 cache/latent space、不可读但可端到端训练。


# 4.2. Latent Space Reasoning

Previous research has investigated the role of latent transformer computations in LLMs’ reasoning abilities. As demonstrated in (Biran et al., 2024), a sequential latent reasoning pathway in LLMs is identified for multi-hop reasoning problems. (Shalev et al., 2024) revealed that the middle layers of

LLMs produce highly interpretable embeddings, representing a set of potential intermediate answers for multi-hop queries. To improve LLMs’ latent reasoning ability, researchers have proposed to augment LLMs with meta tokens. The Pause Token method (Goyal et al., 2023), closely related to our work, introduces trainable embeddings inserted between input and output sequences to encourage latent “thinking”. Similarly, a recent work (Pfau et al., 2024) also studied the circumstances under which causal transformers are able to learn to utilize intermediate dummy tokens (e.g. the filler token used in this work). Unlike these studies which employ pretrained embeddings, our work generates latent tokens dynamically based on the input. More recently, COCONUT (Hao et al., 2024) introduced a new reasoning paradigm by utilizing LLMs’ hidden states as input embeddings in latent space. In contrast to COCONUT, which requires multi-stage training to internalize reasoning, our approach only trains the coprocessor, thereby avoiding limitations on broader applicability.

# 4.3. KV-Cache Compression

KV-cache compression is a technique used to reduce the size of the transformer’s kv-cache, the memory storing past activations, for efficient storage and faster computation. Ge et al. (2024) propose an in-context autoencoder (ICAE) to compress the context into concise embeddings for the kv-cache. Mu et al. (2024) introduce the concept of "gist tokens" to learn compressed representations of the kv-cache. Alternatively, prior work (Li et al., 2023) also improves the efficiency of LLMs by identifying and removing redundant information from the input, making it more compact and easier to process.

While our approach also leverages the kv-cache, our motivation is fundamentally different. Rather than focusing on compression, we aim to augment the kv-cache with latent embeddings produced by an offline coprocessor, thereby enhancing the transformer’s reasoning capabilities without modifying its architecture. This allows us to improve the fidelity of further decoding and boost performance on reasoning-intensive tasks. Our work is inspired by the idea of deliberation, where the coprocessor can "think" in the latent space by processing the kv-cache and generating meaningful embeddings that guide the transformer’s subsequent generation.

> 💡 **KV-cache compression 对比**: 虽然输入都是 kv-cache，但目标相反：compression 是少存少传，DCA 是多加 latent 让后续预测更好。这个区别决定了评价指标从 serving memory/latency 转到 perplexity/reasoning accuracy。


# 4.4. Augmenting LLMs with External Modules

Extensive research has focused on augmenting pretrained LLMs with external modules for improved efficiency and performance (Pfeiffer et al., 2023). Parameter-efficient fine-tuning methods like prompt tuning (Lester et al., 2021), prefix tuning (Li and Liang, 2021), and adapters have also been explored. Adapters, first introduced for computer vision by Rebuffi et al. (2017, 2018) and later popularized in NLP by Houlsby et al. (2019), insert small, trainable modules into the LLM. LoRA (Hu et al., 2021) further improves adapter efficiency by decomposing weight updates into low-rank matrices. Furthermore, multimodal models like Flamingo (Alayrac et al., 2022), CoCa (Yu et al., 2022), PaLI (Chen et al., 2022), and PaLM-E (Driess et al., 2023) leverage cross-attention or soft prompts to incorporate information from other modalities. Building upon these techniques, recent work has explored augmenting LLMs with modules specifically designed for reasoning. For example, CALM (Bansal et al., 2024) employs cross-attention between specialized and general models to enhance the general model’s capabilities.

# 4.5. Hypernetworks for Parameter Generation

Our work shares conceptual similarities with the concept of hypernetworks, where a separate network (the hypernetwork) generates the parameters of another network. In the context of LLM augmentation, rather than learning a fixed set of parameters for a module, a hypernetwork could generate these parameters conditioned on embeddings representing different tasks, inputs, or contexts (Ha et al.,

2017; Platanios et al., 2018). This allows for a form of parameter sharing and "entanglement" between modules that are otherwise disjoint in their parameters (Goyal et al., 2021). Hypernetworks have been used to condition parameter generation on inputs as well. Examples include conditional batch normalization (de Vries et al., 2017), feature-wise linear modulation (FiLM) for text-and-vision tasks (Perez et al., 2018), and self-modulation in GANs (Chen et al., 2019). Bertinetto et al. (2016) even conditioned parameter generation on individual examples for one-shot learning. In the context of LLMs, hypernetworks have generated diverse module parameters, such as classifier heads (Ponti et al., 2021), continuous prompts (He et al., 2022), and adapter layers (Ansell et al., 2021; Mahabadi et al., 2021; Üstün et al., 2020), conditioned on task or language embeddings. These embeddings can be learned or fixed, incorporating side information about task or language relationships.

Importantly, our approach can be viewed through the lens of hypernetworks, with the coprocessor itself acting as a hypernetwork that is conditioned on the kv-cache of the frozen LLM. Instead of generating parameters for a separate module, the coprocessor generates latent embeddings that augment the kv-cache. However, the core principle remains similar: a network dynamically generating outputs based on a rich contextual input. In our case, the kv-cache serves as a highly informative representation of the input sequence, allowing the coprocessor (hypernetwork) to generate augmentations tailored to the specific context.

> 💡 **hypernetwork 视角**: coprocessor 不生成参数，而是生成 cache-conditioned latent context。这个类比有用，因为它强调输出是按输入动态生成的，不是固定 adapter 或固定 prompt。

---

## 🔖 Section 总结

> 💡 **Section 小结**:
> - DCA 横跨 CoT、latent reasoning、kv-cache、external modules 和 hypernetwork。
> - 最贴切的类比是“cache-conditioned network 生成 soft context”。
> - 它的独特性在于 frozen decoder + dynamic cache augmentation，而不是单纯增加 prompt token。
