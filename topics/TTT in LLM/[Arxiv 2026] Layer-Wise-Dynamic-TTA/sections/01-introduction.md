[← 返回 README](../README.md)

# 1. Introduction

Large language models (LLMs) have become a generalpurpose backbone for natural language understanding and generation, powering applications ranging from open-ended assistants to domain-specific writing and reasoning tools (Chowdhery et al., 2022; Singh et al., 2025). Despite strong overall performance, its real-world usage often differs from the training regime: prompts vary wildly in style, length and content, domain-specific terminology appears frequently, and user instructions can deviate from patterns seen during pretraining and alignment. This motivates test-time optimization (Sun et al., 2020) methods that adjust model behavior to bridge the gap between deployment context and the training corpus.

> 💡 **批注**: 这里把 TTA 的动机放在 deployment context shift 上，但本文不是做稳定目标域的 continual adaptation；它处理的是单个 prompt 到来时的即时适应，因此后续必须强调 reset，否则容易把不同用户/不同任务的噪声串起来。

A second, orthogonal challenge for LLMs is instance-level objective mismatch (Krause et al., 2019; Rannen-Triki et al., 2024). Standard training minimizes the expected (or average) loss over the entire training set, producing a single parameter setting that is necessarily a global compromise across heterogeneous samples. At test time, however, performance is determined by the loss on the current prompt instance. Even when the test distribution matches training in aggregate, the globally optimal parameters need not be optimal for a particular prompt, leaving room for per-instance specialization.

> 💡 **批注**: “instance-level objective mismatch” 是这篇比普通 domain adaptation 更细的立足点：即使 test distribution 没有整体 OOD，单个 prompt 也可能希望不同的局部参数。这个论点支撑 sample-specific TTA，而不是批量目标域 fine-tuning。

During inference, LLM behavior can be adjusted in two different ways. First, prompting-based methods—such as in-context learning (Dong et al., 2024) and retrievalaugmented generation (RAG) (Lewis et al., 2021)—steer LLM by modifying the input context while keeping model parameters frozen. Second, test-time adaptation (TTA) (Hu et al., 2025a) modifies the model itself by performing parameter updates during inference. A practically dominant deployment regime is unsupervised, sample-specific TTA. “Unsupervised” means that we typically observe only the prompt $x$ and rarely have access to the gold continuation/answer $y$ (or any other supervision). “Sample-specific” means per-instance updates: for each prompt $x$ , the model takes a small number of gradient steps on $x$ itself (e.g., minimizing prompt negative log-likelihood), then generates $y$ and resets before the next query. This adapt-and-reset protocol matches real-world usage, where prompt streams are heterogeneous and often non-stationary; otherwise, offline (or continual) fine-tuning would be more appropriate. Unsupervised, sample-specific adaptation is also attractive operationally because it requires no labels or external data, can be applied on-the-fly, and can leverage parameter updates to encode information beyond the finite context window length of LLMs (Dai et al., 2019).

> 💡 **设定批注**: 这段明确了核心协议：prompt NLL 更新 → 生成 answer → discard adaptation state。它和 TLM 的 input perplexity minimization 一脉相承，但本文更强调“每个样本独立”的短程优化风险；这也是 fixed LR 很难兼顾不同 prompt、layer 和 step 的原因。

![](../images/a9ee865a7030c12b59cb2ee381a227a73332acb72f72daa428f785dc669f99f3.jpg)
Figure 1. Unsupervised layer-wise dynamic TTA training pipeline (right side).

> 💡 **Figure 1 批注**: 图中右侧训练 pipeline 的关键是“训练时能看 gold answer 来训练 SCALENET，推理时不能看 answer”。所以这不是完全无监督地学控制器，而是 supervised-learned controller for unsupervised inference-time adaptation。

However, na¨ıve unsupervised TTA which runs a few gradient steps on the prompt (Hu et al., 2025b) with a fixed learning rate is often brittle for LLMs. Because the optimization signal comes solely from the observed prompt, updates can over-amplify prompt-specific statistics and deviate from the desired continuation/answer distribution, degrading generation quality. This procedure is also highly sensitive to adaptation strength: small learning rates produce negligible improvements within a handful of steps, whereas large learning rates can cause destructive parameter changes. More fundamentally, sample-specific TTA is driven by a single instance, so its gradients have high-variance and are easily dominated by idiosyncratic prompt features. In our experiments, effective adaptation calls for an aggressive first update (often unevenly distributed across layers) followed by rapid damping, and the appropriate scale can even depend on the LoRA initialization. In summary, a single fixed global learning rate cannot accommodate these step- and layer-dependent dynamics.

> 💡 **fixed LR instability**: 这是本文的核心观察。少步 TTA 需要第一步足够大，否则 LoRA 几乎不动；但继续用同样 LR 多走几步就会过拟合 prompt statistics，偏离 answer distribution。后文 Llama70B AdaptEval fixed LR 的 NLL 从 2.1907 一步后暴涨到多步 5.8/9.7/11+，正是这段话的实证版本。

In this paper we propose the unsupervised layer-wise dynamic TTA framework aimed at controlling the most direct and influential hyperparameters in TTA: the learning rate. The central idea is that the appropriate adaptation strength should depend on both the input prompt and the adaptation state, and that a single global TTA learning rate is too coarse for deep transformers. Gradient magnitudes and update sensitivity can vary substantially across layers and across TTA steps; as a result, uniform updates may be either too small to matter or large enough to be harmful. We therefore introduce a learnable, fine-grained lightweight scaling hypernetwork, called SCALENET, which predicts per-layer, per-step learning rate multipliers for test-time updates. Intuitively, it serves as a control mechanism: it suppresses potentially harmful updates in sensitive layers or for delicate stages, while allowing stronger adaptation where it is beneficial.

> 💡 **动态控制批注**: 学习率在这里被当成 control signal，而不是 optimizer 超参。SCALENET 的假设是“深层 Transformer 的不同 block 对同一 prompt 梯度有不同可塑性”，所以 scaler 必须随 prompt representation、adaptation step 和 layer 改变。

In our implementation, test-time adaptation updates only LoRA (Hu et al., 2021) parameters in the attention query/value projections while keeping the pretrained backbone frozen; before TTA backpropagation, SCALENET predicts multipliers that rescale the base learning rate for LoRA.

> 💡 **LoRA 更新批注**: 只更新 Q/V LoRA 是一个稳健性和成本折中：attention projection 影响 token mixing 和条件化，LoRA 限制更新自由度，backbone 冻结降低 catastrophic drift。但它也意味着容量受限，复杂推理任务未必能靠 prompt-only LoRA 修好。

Our method can also be viewed through the lens of testtime scaling (Muennighoff et al., 2025; Zeng et al., 2025). Rather than improving inference by increasing sampling, reranking candidates, or performing multi-pass generation, our framework scales adaptation compute: it uses a small budget of test-time gradient steps and learns how to allocate update magnitudes across layers and steps on a per-sample basis.

> 💡 **定位批注**: 这里的 test-time scaling 不是采样更多答案，而是把算力花在临时反向传播上。实际部署时要把它和多采样/自一致性/RAG 比较延迟收益比，论文主要验证质量，还没有完整系统成本曲线。

Our contributions can be summarized as:

• Layer-wise dynamic learning rate for unsupervised, sample-specific test time adaptation (TTA). We propose a dynamic TTA scheme that replaces a single global learning rate with learned, step- and layerdependent learning rates, improving stability and effectiveness under a small adaptation budget.

• Lightweight per-layer (Q/V) scaling framework. A hypernetwork predicts non-negative multipliers for each transformer layer (separately for query/value LoRA matrices) and each TTA step, enabling finegrained control of update magnitudes during test-time learning.

• Efficient training via first-order approximation. We train the framework by unrolling the same TTA procedure used at inference and optimizing it with a firstorder approximation that avoids expensive secondorder derivatives.

> 💡 **贡献核对**: 三个贡献对应三个应读证据：1) fixed vs dynamic 的稳定性；2) step-wise vs layer-wise 的增量；3) first-order approximation 是否让训练可行。本文对前两个给了较多实验图表，对第三个更多是方法合理性而非效率 benchmark。
