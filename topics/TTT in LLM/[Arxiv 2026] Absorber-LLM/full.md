# Absorber LLM: Harnessing Causal Synchronization for Test-Time Training

Zhixin Zhang \* 1 Shabo Zhang \* 1 Chengcan $\mathbf { W _ { u } } ^ { * 1 }$ Zeming Wei 1 Meng Sun 1

# Abstract

Transformers suffer from a high computational cost that grows with sequence length for selfattention, making inference in long streams prohibited by memory consumption. Constantmemory alternatives such as RNNs and SSMs compress history into states with fixed size and thus lose long-tail dependencies, while methods that memorize contexts into parameters, such as Test-Time Training (TTT), are prone to overfitting token-level projection and fail to preserve the causal effect of context in pretrained LLMs. We propose Absorber LLM, which formulates longcontext retention as a self-supervised causal synchronization: after absorbing historical contexts into parameters, a contextless model should match the original model with full context on future generations. We optimize this objective by synchronizing internal behaviors of the updated model with the original one, ensuring context absorption and generalization. Experiments on long-context and streaming benchmarks show that Absorber LLM reduces inference memory and improves accuracy over prior parameter-as-memory baselines.

erate texts. Although variants like sparse (Beltagy et al., 2020) and linearized attention (Katharopoulos et al., 2020) mitigate this overhead, they merely delay the inevitable exhaustion of cache and computing resources in long-stream scenarios.

To resolve these memory constraints, prior works have proposed linear-time models, including Recurrent Neural Networks (RNNs) (Peng et al., 2023) and State Space Models (SSMs) (Gu et al., 2021; Gu & Dao, 2024). These models achieve efficiency by compressing history into fixed-size hidden states. However, recent theoretical and empirical analyses reveal that this efficiency comes at the cost of model expressive power, leading to significant information loss compared to full-attention transformers (Merrill et al., 2024; Hsieh et al., 2024).

Alternatively, Test-Time Training (TTT) (Sun et al., 2025; Feng et al., 2026) methods utilize model parameters for context storage to overcome state capacity limits. Specifically, given a sequence $X$ to be stored in the parameters of a model, TTT methods train the model to project $X$ into a specifically constructed target $V$ . Such methods optimize the model’s performance on $X$ by writing it directly into the model parameters. Yet, when learning $X$ , they ignore the model’s performance on subsequent sequences, failing to help the model absorb $X$ in a way that contributes to future inferences. Crucially, such reconstruction methods overlook the causal mechanisms between historical contexts and subsequent inferences, which are essential for effective model deduction (Geiger et al., 2024).

# 1. Introduction

Transformers (Vaswani et al., 2017) underpin modern AI. However, in real-world deployments, transformer models struggle to satisfy further expectations, such as maintaining lifelong memories of user interactions across multiple domains (Park et al., 2023; Wang et al., 2024; Jimenez et al., 2024; Bairi et al., 2024), and achieving the ultimate goal of continuously absorbing a never-ending stream of data without resetting their state. This bottleneck stems from their inability to effectively process super-long contextual streams. Specifically, the self-attention mechanism of transformers (Vaswani et al., 2017; Dao et al., 2022) incurs a quadratic computational complexity $\mathcal { O } ( N ^ { 2 } )$ to gen-

In this paper, we draw inspiration from TTT and further propose to preserve the causal relation between historical contexts and the subsequent generation process. We introduce Absorber LLM (Fig. 1), which ensures causal effect preservation by training the contextless model to behave identically to the full-context model in future inferences. Note that compared to prior TTT methods, our approach focuses on causal effect synchronization rather than simple target projection. We maintain the contribution of removed historical sequences to future inferences, which is crucial for more effective deductions.

Our main contributions are as follows:

• We move beyond directly memorizing contexts to preserving their causal effects on subsequent deductions, and formally define such causal effect preservation as a functional equivalence problem, providing a standard for information transfer from contexts to parameters.

![](images/59df0826c67f5ec5e8ac4741ee064ae1b26db416a12a8c964c60ee909512d42f.jpg)
Figure 1. A brief overview of our method. We propose to absorb historical contexts and move beyond simple context memorization, focusing on preserving the causal relationship between historical context and future generations rather than directly reconstructing historical tokens. Specifically, we fine-tune the context-less model to behave identically to the original model with full contexts. This is achieved through a self-supervised optimization that synchronizes the hidden states or output behaviors of the two models. This example shows the context absorption process, where the model absorbs 3 historical tokens by synchronization on 5 future tokens. Our method ensures that the absorbed context retains its semantic and causal influence on future deductions.

• We introduce an update mechanism to satisfy the equivalence condition in transformer models, successfully synchronizing the behavior of the updated contextless model with the ideal full-context model, and realizing causal effect preservation and context absorption.

• We validate the efficiency and effectiveness of our method on long-context benchmarks. Our results demonstrate that Absorber LLM not only outperforms traditional transformer models in computational efficiency but also achieves superior performance compared to prior linear-time and parameter memory methods, paving the way for scalable, infinite-stream inference in real-world deployments.

# 2. Motivation

# 2.1. Memory issue of Transformers

Transformers exhibit quadratic computational complexity $\mathcal { O } ( N ^ { 2 } )$ due to their attention mechanism (Vaswani et al., 2017). Specifically, given a pretrained LLM $f$ with parameter $W$ , generating a single token $x _ { n + 1 } = f _ { W } ( x _ { 1 } , \ldots , x _ { n } )$ from an input sequence $X = ( x _ { 0 } , \ldots , x _ { n } )$ costs ${ \mathcal { O } } ( n )$ with

KV cache. Autoregressively, producing $N$ tokens results in a cumulative $\mathcal { O } ( N ^ { 2 } )$ computational overhead and $\mathcal { O } ( N )$ memory footprint (Fig. 2). Numerous variants (Child, 2019; Beltagy et al., 2020; Katharopoulos et al., 2020; Choromanski et al., 2021; Liu et al., 2024) have been proposed to resolve this memory issue, but they merely delay the inevitable exhaustion of cache memory and computing resources.

# 2.2. Superiority of parameter memory

The primary cause of the high computational cost of transformers in long contexts is their attention mechanism, which must process all historical contexts during deduction. To address this memory issue, prior works have proposed lineartime sequence models (Peng et al., 2023; Gu et al., 2021; Gu & Dao, 2024) that reduce the computational cost to $\mathcal { O } ( N )$ . Note that the linear-time models gain efficiency by sacrificing model capacity (Merrill et al., 2024), as they compress arbitrarily rich history into a fixed-size hidden state $h _ { t } \in \mathbb { R } ^ { d }$ . By contrast, the paradigm that restores contexts into the model parameters is more promising, since the parameter space $\boldsymbol { \theta } \in \mathbb { R } ^ { d \times d }$ is orders of magnitude larger than the activation space $h \in \mathbb { R } ^ { d }$ , offering a vastly superior capacity for encoding history. Consequently, computationally costly contexts can be safely omitted from long-stream inferences.

Formally, let $X$ denote the historical contexts and $Y$ denote the subsequent inferences. In the parameter memory paradigm, rather than performing standard inference over the full context $f _ { W } ( X Y )$ , we compute $f _ { W ^ { * } } ( Y )$ without the explicit context $X$ , having stored the information of $X$ directly into the updated parameters $W ^ { * }$ . Such a parameter memory paradigm has been proposed by (Sun et al., 2025; Behrouz et al.).

![](images/99313db5380dd90e701613cc6f3d783e2f18d4b6be00a0e8fbf6b410802b2714.jpg)
Figure 2. A demonstration of the necessity of inference with a limited number of contexts. Normally, transformers compute $f _ { W } ( x _ { 0 } x _ { 1 } \dots x _ { n } )$ to generate the next token $x _ { n + 1 }$ with a computational complexity of ${ \mathcal { O } } ( n )$ . If we absorb history into the parameters to get $f _ { W ^ { * } }$ and deduct with a limited context length, the deduction with discarded history has a reduced complexity of $\mathcal { O } ( 1 )$ , which suits scenarios with long sequence lengths.

# 2.3. Analysis of existing parameter memory methods

Test-Time Training (TTT) (Sun et al., 2025; Feng et al., 2026), the state-of-the-art parameter memory method, treats historical contexts as a continuous learning loop, updating model weights to memorize the incoming stream. That is:

$$
W ^ { * } = \arg \operatorname* { m i n } _ { W ^ { * } } \| f _ { W ^ { * } } ( x _ { \mathrm { k } } ) - x _ { \mathrm { v } } \|
$$

where $x _ { \mathrm { k } }$ and $x _ { \mathrm { v } }$ are conversions of input $x$ , which build target projections. By training the model on target projections, TTT effectively converts context into training data and leverages the plasticity of neural networks to extend memory capacity, and reveals optimum performance in comprehensive context streams as batched training data. Overall, TTT is more suitable for scenarios of effective training, as it relies on adequate inputs to stabilize its training and has limitations when there are insufficient contexts that cannot form batch data, which is more common in the prevalent scenario of model inference in user interactions. Specifically, in the model inference process, TTT memorizes contexts by training the model to project them, and simply memorizing such inadequate data tokens often leads to the retention of irrelevant noise while failing to capture the high-level semantic dependencies required for reasoning (Cao et al., 2025). Furthermore, when projecting contexts, TTT does not specify the model’s performance on subsequent inferences. This cannot invoke the causal mechanisms in LLM, as it does not maintain the contribution of removed historical sequences to future inferences, which is crucial for model deduction (Geiger et al., 2021; 2024). Overall, incorporating historical contexts is not just data summarization, but learning the causal representation (Scholkopf et al. ¨ , 2021) that maintains the intervention of history tokens to future ones.

Motivation summary To summarize, we aim to reduce the $\mathcal { O } ( N ^ { 2 } )$ computational cost of full-attention transformers by absorbing contexts into model parameters. Based on analyses of prior works, we propose to preserve the causal effects of historical context on subsequent sequences when building parameter memory.

# 3. Proposed Method

# 3.1. Context absorption with causal relation synchronization

Inspired by the parameter memory paradigm and the necessity of invoking causal mechanisms, we propose to move beyond context memorization to the preservation of the causal relation between historical contexts and the subsequent generation process. We define the required causal relation preservation in context memorization as the following self-supervised learning target: The updated contextless model $f _ { W ^ { \ast } }$ , with historical contexts $X$ absorbed into its parameters and access only to the subsequent texts $Y$ , should behave identically to the original pretrained model $f _ { W }$ with full contexts $X Y$ . That is:

$$
f _ { W ^ { * } } ( Y ) \equiv f _ { W } ( X Y )
$$

where $Y$ denotes the subsequent text following $X$ , i.e., successive input tokens or model-generated inferences, such that the distribution of $Y$ is conditionally dependent on $X$ Note that such equivalence should be kept throughout the inference process of $Y$ :

$$
f _ { W ^ { * } } ( Y [ : i ] ) = f _ { W } ( X Y [ : i ] ) \quad i = 1 , \dots , | Y |
$$

Then we get an optimization target:

$$
\boldsymbol { W } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { W } ^ { * } } \sum _ { i = 1 } ^ { | \boldsymbol { Y } | } \parallel f _ { \boldsymbol { W } ^ { * } } ( \boldsymbol { Y } [ : i ] ) - f _ { \boldsymbol { W } } ( \boldsymbol { X } \boldsymbol { Y } [ : i ] ) \parallel
$$

Note that compared to prior works, our optimization is guided by the causal effect synchronization objective rather than simple reconstruction in Equation 1. Instead of trying to reconstruct the exact history $X$ token-by-token, our goal is to find a parameter configuration $W _ { * }$ that preserves the causal effect of $X$ on the subsequent generation process of $Y$ . This ensures that the model absorbs existing contextual information and preserves its effects on further deduction, rather than simply memorizing it. Moreover, by applying semantic dependencies between $X$ and $Y$ , such absorption only retains the relevant information in historical contexts $X$ for subsequent inference in $Y$ , which filters noise in contextual data. By absorbing only the relevant information, our method relies less on high-quality contexts, and is more suitable for usual scenarios of model inferences.

# Algorithm 1 Context Absorption

Inputs: LLM $f _ { W }$ ; Historical context $X = ( x _ { 1 } , x _ { 2 } , \ldots , x _ { n } )$ to be absorbed into parameters $W$ ; Subsequent text segment $Y = ( x _ { n + 1 } , x _ { n + 1 } , \dots , x _ { n + m } )$ ; Learning rate $\eta$ ; Maximum fine-tuning step number $K$ ; Synchronization threshold $\epsilon$

Outputs: Updated model parameters $W ^ { * }$ with context $X$ being absorbed.

1: Forward $f _ { W } ( X Y )$ , get full-context hidden states
$H _ { 1 } , \dots , H _ { n } , H _ { n + 1 } , \dots , H _ { n + m }$
2: Initialize $W ^ { * }  W$
3: for $k \gets 1$ to $K$ do
4: Forward $f _ { W ^ { * } } ( Y )$ to get contextless hidden states:
$\tilde { H } _ { n + 1 } , \ldots , \tilde { H } _ { n + m }$
5: Compute synchronization loss:
$\begin{array} { r } { \mathcal { L }  \frac { 1 } { m } \sum _ { p = n + 1 } ^ { \cdot n + m } \parallel \tilde { H } _ { p } - H _ { p } \parallel } \end{array}$
6: if L < ϵ then
7: ReturnW ∗
8: end if
9: $\nabla _ { W ^ { * } } { \mathcal { L } } \gets \mathrm { B a c k w a r d } ( { \mathcal { L } } )$
10: $W ^ { * }  W ^ { * } - \eta \cdot \nabla _ { W ^ { * } } \mathcal { L }$
11: end for
12: ReturnW ∗

# 3.2. Absorb contexts through hidden states synchronization

The goal of our method is to realize equations 2 through optimization in equation 4 for causal effect synchronization. To ensure a more thorough synchronization, we force the hidden states of $f _ { W ^ { * } } ( Y )$ to be identical to $f _ { W ^ { * } } ( X Y )$ , which guarantees that they conduct identical deductions in the same internal manner. Specifically, let historical contexts $X ~ = ~ ( x _ { 1 } , x _ { 2 } , . . . , x _ { n } )$ be absorbed into model parameters, and we refer to future inferences $Y = ( x _ { n + 1 } , x _ { n + 1 } , \dots , x _ { n + m } )$ to synchronize the behavior of the contextless model to the original full-context model; let $H _ { 1 } , \dots , H _ { n } , H _ { n + 1 } , \dots , H _ { n + m }$ be the hidden states in propagating $f _ { W } ( X Y ) , { \tilde { H } } _ { n + 1 } , \dots , { \tilde { H } } _ { m }$ be the hidden states in propagating $f _ { W ^ { * } } ( Y )$ , where $H _ { p } = ( h _ { p } ^ { 0 } , \ldots , h _ { p } ^ { L } ) , p =$ $1 , \ldots , n + m$ , and $\tilde { { \cal H } } _ { p } \ = \ ( \tilde { h } _ { p } ^ { 0 } , \tilde { h } _ { p } ^ { 1 } , \dots , \tilde { h } _ { p } ^ { L } ) , p \ = \ n + $ $1 , \ldots , n + m$ , and $L$ is the number of layers in LLM $f$ Then the hidden states synchronization is:

$$
\tilde { H } _ { p } = H _ { p } \quad p = n + 1 , \ldots , n + m
$$

which ensures the output synchronization in equation 3 by synchronizing the propagation process more thoroughly. Then we get the optimization target to ensure equation 4:

$$
W ^ { * } = \arg \operatorname* { m i n } _ { W } \frac { 1 } { m } \sum _ { p = n + 1 } ^ { n + m } \parallel \tilde { H } _ { p } - H _ { p } \parallel
$$

The above process that absorbs contexts $X$ into parameters $W$ is summarized in Algorithm 1. Note that the absorption requires fine-tuning, and we employ LoRA (Hu et al., 2022) for efficiency.

# 3.3. Conduct absorption in the flowing sequence

In a continual workflow where new texts, i.e., model generations or subsequent user prompts, are continuously absorbed into the model parameters, the synchronization process becomes iterative: Where new incoming contexts $X$ keep being absorbed into parameters $W$ by synchronization on subsequent texts $Y$ . Specifically, we maintain a dynamic absorption window $Z = X Y$ . After the absorption of $X$ , we move the window to absorb the segment subsequent to $X$ . Such a continual absorption is detailed in Algorithm 2.

To summarize, we synchronize $f _ { W ^ { * } } ( Y )$ with $f _ { W } ( X Y )$ using a fine-tuning strategy. By minimizing the divergence between the updated contextless model behavior and the ideal full-context behavior within the local segment, we treat the parameter update as a continuous error-correction process, which ensures the context absorption and generalization. Please refer to Figure 1 for a brief demonstration. Note that the computational cost of context absorption in Algorithm 1 is fixed given $m$ and $n$ ; in Algorithm 2, we absorb and generate $n$ tokens in each round, so that the computational costs grow linearly with sequence length. Overall, the computational cost of our method is $\mathcal { O } ( N )$ .

# 4. Experiments

# 4.1. Set up

In this section, we evaluate the performance of Absorber LLM across four critical dimensions: computational efficiency, information retention in many-shot in-context learning, logical consistency in long-chain reasoning, and long text summarization.

Settings. Our experiments utilize the LLaMA2-7B (Touvron et al., 2023) architecture as the primary backbone. All evaluations are conducted on an NVIDIA GeForce RTX

# Algorithm 2 Absorber Deduction

#

Original pretrained LLM $f _ { W }$ ; Input texts $I$ ; absorbed token number at one time: $n$ ; synchronization token number $m$

#

Inference texts $S$

1: Initialize $t \gets 0$ , absorption window $Z \gets I$ , Inference
texts $S = \varepsilon$
2: while True do
3: if $| Z | < m + n$ then
4: for $i \gets 1$ to $m + n - | Z |$ do
5: Forward $f _ { W } ( Z )$ to generate the next token $z$
6: if $z = \mathrm { e o s } ,$ token then
7: Return $S$
8: end if
9: $Z  Z z , S  S z$
10: end for
11: end if
12: $X  Z [ : n ] , Y  Z [ n + 1 : n + m ]$
13: Absorb $X$ into $W$ to get $W ^ { * }$ by synchronizing
$f _ { W ^ { * } } ( Y )$ with $f _ { W } ( X Y )$ (Algorithm 1)
14: $Z \gets Z [ n + 1$ :]; $W  W ^ { * }$
15: end while

4080 SUPER GPU. In Algorithm 1, we set $n = 1 0 2 4$ and $m = 2 0 4 8$ . We employ the $L 1$ norm for the loss function and use the AdamW (Loshchilov & Hutter, 2019) optimizer with a learning rate of $\eta = 5 \times 1 0 ^ { - 4 }$ . For LoRA, we set the rank to $r = 6 4$ , and the synchronization threshold is set to $\epsilon = 2$ .

Baselines. We compare our method against standard LLaMA2-7B as a representative of full-attention transformers, Mamba (Gu & Dao, 2024) as a representative SSM, and TTT (Sun et al., 2025) as a parameter-memory baseline.

Datasets and Evaluation. For each of the four aforementioned tasks, we selected current popular datasets for evaluation. To assess computational efficiency, we employed a popular subset of the Pile called Books3 (Gao et al., 2020) and used the text generation time of the model as the evaluation metric. For information retention in many-shot in-context learning, we utilized the Agnews (Zhang et al., 2015) dataset, which requires the model to classify different types of news articles. In our experiments, we provided the model with several news classification examples as the many-shot context, forming a long text input, and then asked the model to classify a new article. The classification accuracy served as the evaluation score for this task. To evaluate logical consistency in long-chain reasoning, we adopted the Musique (Bai et al., 2023) dataset, a collection of long-text reasoning problems. The model’s task is to identify several relevant points from a lengthy passage and perform reasoning to derive the answer. The reasoning accuracy was used as the performance metric. Finally, for long text summarization, we employed the well-known SamSum (Gliwa et al., 2019) dataset, which similarly tests the model’s ability to comprehend and summarize extended texts. We computed the similarity between the model-generated summary and the reference summary using the Bleurt (Sellam et al., 2020) model, and this similarity score served as the final evaluation metric.

# 4.2. Inference Complexity and Latency Scaling

We use Books3, a popular subset of the Pile. We assess the inference efficiency by measuring the average per-token latency $( \mathcal { L } )$ , which we define as the amortized time to generate $K = 1 2 8$ tokens following a prefix of length $N$ :

$$
\mathcal { L } ( N ) = \frac { 1 } { K } \left( \mathcal { T } _ { g e n } ( N + K ) - \mathcal { T } _ { p r e f i l l } ( N ) \right)
$$

where $\tau$ denotes wall-clock time. We vary the historical prefix $N$ from $1 0 ^ { 3 }$ to $1 . 6 \times 1 0 ^ { 4 }$ tokens.

Table 1. Comparison of amortized per-token latency $\mathcal { L } ( N )$ across varying historical prefix lengths $N$ . $K = 1 2 8$ tokens are generated for each measurement.

<table><tr><td>Method(10−3s)</td><td>N = 1k ∼ 2k</td><td>N = 2k ∼ 4k</td><td>N = 4k ∼ 8k</td><td>N = 8k ∼ 16k</td></tr><tr><td>Standard</td><td>19.23</td><td>39.07</td><td>76.32</td><td>OOM</td></tr><tr><td>Mamba</td><td>4.14</td><td>4.52</td><td>3.88</td><td>3.69</td></tr><tr><td>TTT</td><td>20.33</td><td>20.39</td><td>19.16</td><td>19.31</td></tr><tr><td>Absorber</td><td>29.73</td><td>30.22</td><td>30.38</td><td>29.98</td></tr></table>

As evidenced by the empirical data in Table 1, the Standard Transformer exhibits a sharp, non-linear increase in inference latency as the context length scales, jumping from $1 9 . 2 3 \mathrm { m s }$ to $7 6 . 3 2 \mathrm { m s }$ , and eventually leading to an Out-of-Memory error beyond 8K tokens. This bottleneck stems from the quadratic complexity of self-attention and the mounting overhead of KV cache management. In contrast, Absorber LLM maintains a remarkably stable latency profile. While it carries a slightly higher initial overhead at short contexts, its latency remains near-constant regardless of the internalized context length. This demonstrates a practical $O ( 1 )$ inference complexity.

# 4.3. Long-Context In-Context Learning

We evaluate many-shot In-Context Learning (ICL) performance using the Agnews dataset, focusing on tasks that require integrating multiple demonstration samples across long sequences. Table 2 summarizes the averaged scores across different sequence lengths.

Table 2. Performance comparison on Agnews. We report macroaveraged Accuracy.

<table><tr><td>Text Length</td><td>1k ~2k</td><td>2k ~4k</td><td>4k ~8k</td></tr><tr><td>Standard</td><td>54.2</td><td>49.8</td><td>43.7</td></tr><tr><td>Mamba</td><td>28.7</td><td>28.5</td><td>26.2</td></tr><tr><td>TTT</td><td>30.6</td><td>30.2</td><td>29.7</td></tr><tr><td>Absorber LLM</td><td>37.9</td><td>38.2</td><td>35.5</td></tr></table>

As shown in Table 2, Absorber LLM consistently outperforms other linear-complexity baselines, namely Mamba and TTT, across all evaluated context windows. While the standard transformer maintains higher accuracy in shorter contexts ${ \left( { < 4 \mathrm { K } } \right) }$ , it imposes a heavy memory footprint within the $4 \mathrm { K } \mathrm { \sim } 8 \mathrm { K }$ range, limiting its practical deployment. Absorber LLM demonstrates a robust and stable performance profile, maintaining an accuracy of approximately $3 5 \% \sim 3 8 \%$ , which is higher than the $2 8 \% \sim 3 1 \%$ range achieved by Mamba and TTT. These results indicate that our proposed functional alignment is more effective at distilling salient task-specific logic into model parameters than traditional reconstruction-based objectives.

# 4.4. Logical Integrity in Multi-Step Reasoning

In this part, we focus on multi-step deduction where the model must connect multiple factual links scattered across a long context. Specifically, we provide the model with the question and the initial reasoning steps, and task it with inferring the final answer based on this context. The accuracy of the answer is the score of the testing model.

Table 3. Performance comparison on Musique. We report macroaveraged Accuracy.

<table><tr><td>Method</td><td>8k ~12k</td><td>12k ~16k</td><td>16k ~20k</td></tr><tr><td>Standard</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>Standard-Truncation</td><td>19.2</td><td>18.5</td><td>17.9</td></tr><tr><td>Mamba</td><td>31.5</td><td>30.4</td><td>27.6</td></tr><tr><td>TTT</td><td>34.3</td><td>30.9</td><td>28.4</td></tr><tr><td>Absorber LLM</td><td>33.8</td><td>31.6</td><td>29.5</td></tr></table>

As illustrated in Table 3, the standard transformer fails to process ultra-long sequences $( > 8 K )$ due to its quadratic memory complexity, resulting in OOM errors. To provide a viable baseline, we introduce standard-truncation, a method that constrains the standard model by truncating the input to its maximum supported context window, effectively discarding earlier historical information to avoid memory exhaustion. The experimental results demonstrate that as the sequence length scales to the ultra-long regime $( 1 6 K \sim 2 0 K )$ , the performance of other linear-complexity models like Mamba and TTT degrades significantly, dropping to 27.6 and 28.4, respectively. Notably, Absorber LLM exhibits superior scalability, maintaining a higher accuracy of 29.5.

# 4.5. Long Text Summary

In this experiment, we selected three different lengths of text, namely less than $2 \mathrm { k } \mathord { \sim } 4 \mathrm { k }$ , $4 \mathrm { k } \mathrm { \sim } 8 \mathrm { k }$ , and $8 \mathrm { k } \sim 1 6 \mathrm { k }$ . For each length of text, we used different methods to input the text and extract the model output. Finally, we compared the model output with the standard answers in the Samsum dataset, and used the Bleurt model (Sellam et al., 2020) to calculate the similarity between the two as the final score.

Table 4. Performance of the model on Samsum of different text lengths

<table><tr><td>Text Length</td><td>2k~4k</td><td>4k~8k</td><td>8k~16k</td></tr><tr><td>Standard</td><td>0.433</td><td>0.428</td><td>OOM</td></tr><tr><td>Mamba</td><td>0.428</td><td>0.413</td><td>0.395</td></tr><tr><td>TTT</td><td>0.415</td><td>0.406</td><td>0.391</td></tr><tr><td>Absorber LLM</td><td>0.423</td><td>0.417</td><td>0.408</td></tr></table>

In the summarization task, we evaluate the model’s ability to capture salient information from long dialogues. As shown in Table 4, the standard transformer serves as the performance upper bound in shorter contexts; however, its memory requirements become prohibitive as the sequence scales, eventually leading to an OOM error in the $8 K \sim 1 6 K$ range.In the $2 K \sim 4 K$ range, while Absorber LLM (0.423) still trails the standard model (0.433), it significantly outperforms the other linear-complexity baselines, exceeding Mamba and TTT by 0.008 and 0.005 points, respectively. This suggests that in the early stages of context extension, functional alignment provides a more effective compression mechanism for dialogue structures than the hidden state updates in Mamba or the reconstruction objective in TTT. As the text length increases to $8 K \sim 1 6 K$ , the advantage of Absorber LLM as a scalable alternative becomes decisive. While the Standard model is no longer functional due to memory exhaustion, Absorber LLM maintains a stable score of 0.408, outperforming Mamba (0.395) by a clear margin of 0.013. This indicates that Absorber LLM successfully internalizes global dialogue dependencies into its weights.

# 4.6. Ablation Study

# 4.6.1. F1 SCORE

The F1 score is defined as the harmonic mean of precision and recall:

$$
F _ { 1 } = 2 \cdot \frac { \mathrm { P r e c i s i o n } \cdot \mathrm { R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } }
$$

In this context, Precision is the ratio of the number of shared tokens to the total number of tokens in the predicted sequence, while Recall is the ratio of the number of shared tokens to the total number of tokens in the ground truth sequence. By treating both the prediction and the reference as bags of tokens, the F1 score effectively captures the semantic proximity and the quality of the generated reasoning steps.

# 4.6.2. HIDDEN STATES AND TOKEN

To verify the necessity of layer-wise functional alignment, we conduct an ablation study comparing our approach, aligning internal hidden states, against a baseline that aligns only the token-level output distributions, i.e., logits. In the tokenlevel alignment variant, we replace the state-wise MSE loss with the KL divergence loss between the next-token probabilities of the contextless model and the full-context oracle.

Table 5. Ablation results on alignment granularity using Samsum. Token Alignment constraints only the output layer, while Hidden State Alignment constraints the entire representational space.

<table><tr><td>Alignment Target</td><td>F1 Score</td><td>BRT Score</td></tr><tr><td>Token Distribution</td><td>37.4</td><td>0.324</td></tr><tr><td>Hidden States</td><td>57.5</td><td>0.415</td></tr></table>

The results in Table 5 reveal a significant performance gap. We observe that token-level alignment alone is insufficient for Samsum. We believe that this shallow supervision only optimizes the final predictive head, failing to internalize the complex causal structures. In contrast, by forcing the contextless model’s hidden states to match the oracle, we ensure that the high-dimensional semantic space, which has a vastly superior capacity for encoding history compared to surface-level tokens, is fully utilized. This deep alignment effectively preserves the deductive mechanisms required for reasoning and summarizing, preventing the model from collapsing into simple statistical pattern matching.

# 4.6.3. HYPERPARAMETER EXPERIMENT ON $n$ AND $m$

We investigate the impact of the absorption segment length $n$ and the behavioral alignment window $m$ on the model’s performance. Table 6 presents the accuracy on a validation subset of Samsum. We observe that increasing the reference length $m$ consistently improves alignment quality by providing a broader supervision signal for the model’s behavior. Our results indicate that performance generally improves as the absorption segment length $n$ increases, because a larger $n$ enables the model to capture more holistic contextual dependencies and internalize comprehensive semantic structures within each update. Consequently, the configuration with a larger $n$ combined with a larger $m$ provides an optimal balance for maintaining high alignment fidelity during long-stream inference.

Table 6. Ablation study on internalization segment length $n$ and behavioral alignment window $m$ . Scores represent Accuracy $( \% )$ .

<table><tr><td>Accuracy (%)</td><td>m = 512</td><td>m = 1024</td><td>m = 2048</td></tr><tr><td>n = 256</td><td>35.1</td><td>57.9</td><td>42.3</td></tr><tr><td>n = 512</td><td>41.9</td><td>46.5</td><td>55.1</td></tr><tr><td>n = 1024</td><td>47.8</td><td>50.8</td><td>61.9</td></tr></table>

# 4.6.4. IMPACT OF REGULARIZATION

We investigate the influence of different regularization strategies on the parameter absorption process. In our framework, the weight update involves a trade-off between minimizing the functional alignment loss and maintaining the structural integrity of the pretrained parameters. We compare two configurations: (1) alignment with L1-norm regularization to promote sparsity in updates, and (2) alignment with L2-norm regularization to encourage smooth parameter transitions. The experiments are conducted on Samsum dataset to assess how these constraints affect long-range logical deduction.

Table 7. Ablation study on regularization methods for context absorption. Results are reported on the Samsum dataset.

<table><tr><td>Regularization</td><td>F1 Score</td><td>BRT Score</td></tr><tr><td>L1 Regularization</td><td>57.5</td><td>0.415</td></tr><tr><td>L2 Regularization</td><td>42.3</td><td>0.383</td></tr></table>

The empirical findings presented in Table 7 indicate that L1 regularization is superior for the context absorption paradigm. We observe that L2 regularization, by imposing a uniform penalty on all weight magnitudes, tends to diffusely suppress updates, which may hinder the formation of the sharp, specific parameter changes required to encode discrete logical relations. This leads to a noticeable drop in both F1 and BRT scores. L1 regularization, by inducing sparsity, acts as a selective constraint. It effectively preserves the structural integrity of the pretrained representational space while providing sufficient plasticity to internalize sparse causal dependencies. This balance is crucial for maintaining the deductive capability of transformers.

# 5. Related Work

Efficient Transformers and Linear Attention. Standard self-attention computes the output as √ $O \_ { } \qquad =$ softmax $( Q K ^ { T } / \sqrt { d } ) V$ , incurring $\mathcal { O } ( N ^ { 2 } )$ computational complexity due to the $N \times N$ attention matrix. Linear attention methods (Katharopoulos et al., 2020; Choromanski et al., 2021) bypass this by employing a kernel trick. By approximating the softmax operation with feature maps $\phi ( \cdot )$ and exploiting the associativity of matrix multiplication, the attention mechanism becomes:

$$
O _ { i } = \phi ( Q _ { i } ) ^ { T } \sum _ { j = 1 } ^ { i } \left( \phi ( K _ { j } ) V _ { j } ^ { T } \right)
$$

Here, the term $\sum \phi ( K _ { j } ) V _ { j } ^ { T }$ acts as a matrix-valued hidden state $S _ { i } \in \mathbb { R } ^ { d \times d }$ , updated recurrently as $S _ { i } = S _ { i - 1 } +$ $\phi ( K _ { i } ) V _ { i } ^ { T }$ . This reduces inference complexity to $\mathcal { O } ( N )$ . However, strictly adhering to this linear recurrence degrades retrieval capabilities compared to full attention, primarily due to its deviation from the exact softmax mechanism.

State Space Models (SSMs) and Modern RNNs. Recent architectures like Mamba (Gu & Dao, 2024) and RWKV (Peng et al., 2023) discretize continuous-time state space equations into a recurrent form. The core mechanism involves compressing history into a hidden state $h _ { t } \in \mathbb { R } ^ { d }$ by:

$$
h _ { t } = \bar { A } _ { t } h _ { t - 1 } + \bar { B } _ { t } x _ { t } , \quad y _ { t } = C h _ { t }
$$

where ${ \bar { A } } _ { t }$ and ${ \bar { B } } _ { t }$ are discretized parameters (which, in the case of Mamba, are data-dependent). While efficient $\mathcal { O } ( 1 )$ inference state), these models are constrained by the Fixed-Capacity Hypothesis (Merrill et al., 2024). The state $h _ { t }$ must encode the entire context $x _ { 1 : t }$ . Since $\dim ( h _ { t } ) \ll \dim ( x _ { 1 : t } )$ for long sequences, this inevitably results in lossy compression and the catastrophic forgetting of long-tail dependencies that cannot be fit into the bounded vector space $\mathbb { R } ^ { d }$ .

Parametric Memory and Test-Time Training (TTT). To expand memory capacity beyond fixed vectors, TTT (Sun et al., 2025) adopts model parameters as the storage medium. TTT formulates memory retrieval as an online self-supervised learning problem. Given a history sequence or a new token $x _ { t }$ ), it updates the internal weights $W _ { t - 1 }$ to $W _ { t }$ by minimizing a reconstruction objective $\ell$ in Equation 1:

$$
W _ { t } = W _ { t - 1 } - \eta \nabla _ { W } \ell ( W _ { t - 1 } , x _ { t } )
$$

This process effectively embeds the context into the highdimensional parameter space $\mathbb { R } ^ { d \times d }$ , theoretically offering vastly superior memory capacity compared to vector states. However, this approach relies on iterative approximation through Stochastic Gradient Descent (SGD). Consequently, the quality of memory retention becomes heavily dependent on proper optimization hyperparameters and adequate training data. Furthermore, when projecting contexts, TTT does not specify the model’s performance on subsequent inferences. This cannot invoke the causal mechanisms in LLM, as it does not maintain the contribution of removed historical sequences to future inferences.

Predictive Information and Causal Alignment. Our formulation of parameter memory moves beyond simple target projection toward functional alignment, proposing a concept that bridges information theory and causal inference. From an information-theoretic perspective, the information bottleneck principle (Tishby et al., 2000; Bialek et al., 2001) suggests that an optimal representation should discard historical details irrelevant to predicting the future. More fundamentally, this aligns with recent advances in causal abstraction (Geiger et al., 2021; 2024), which posit that a simpler compressed model serves as a valid surrogate for a complex full system only if it preserves the causal mechanisms governing the output. In this view, context compression is not merely data summarization, but rather the learning of an invariant causal representation (Scholkopf et al.¨ , 2021) that maintains the intervention-effect relationship between historical and future tokens. While prior works in prompt compression (Chevalier et al., 2023; Mu et al., 2023) apply this philosophy at the token level, Absorber LLM extends causal alignment to the parameter space. By enforcing our equivalence condition, we ensure that the compressed parameters functionally substitute the full history, preserving the causal dynamics of generation without retaining the raw data.

# 6. Future Work

In this paper, we presented Absorber LLM, a framework that internalizes historical context into model parameters by enforcing functional alignment. Our proposed framework can be extended along three main directions.

First, we will explore how context absorption can be integrated with continual learning, enabling models to incrementally internalize new data while avoiding catastrophic forgetting of previously absorbed memories. Second, we will investigate the causal editability of absorbed representations to support selective forgetting, allowing specific memories to be removed for privacy or factual updates without full retraining. Third, we aim to extend the absorption paradigm to multimodal long-context settings, such as long-form video and sensor streams, where functional alignment could offer a scalable alternative to expanding the KV cache.

These extensions will advance the development of more efficient, controllable, and broadly capable long-context models. We plan to optimize and expand the framework to make it applicable to longer texts and larger-parameter models.

# 7. Conclusion

This paper introduced Absorber LLM, which internalizes long contexts into model parameters through functional alignment, moving beyond the computational constraints of conventional transformer KV-caches. By aligning layerwise representations between a contextless model and a full-context oracle, our approach encodes semantic dependencies directly into the parameter space, enabling causal and coherent reasoning without expanding external memory. Experiments demonstrate that Absorber LLM achieves constant-time inference and outperforms existing methods on long-context benchmarks, while robustly handling dynamic updates.

# References

Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., Dong, Y., Tang, J., and Li, J. Longbench: A bilingual, multitask benchmark for long context understanding, 2023.

Bairi, R., Sonwane, A., Kanade, A., C, V. D., Iyer, A., Parthasarathy, S., Rajamani, S., Ashok, B., and Shet, S. Codeplan: Repository-level coding using llms and planning. Proceedings of the ACM on Software Engineering, 1(FSE):675–698, 2024.

Behrouz, A., Razaviyayn, M., Zhong, P., and Mirrokni, V. Nested learning: The illusion of deep learning architectures. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

Bialek, W., Nemenman, I., and Tishby, N. Predictability, complexity, and learning. Neural computation, 13(11): 2409–2463, 2001.

Cao, Z., Schooler, L., and Zafarani, R. Analyzing memory effects in large language models through the lens of cognitive psychology. arXiv preprint arXiv:2509.17138, 2025.

Chevalier, A., Wettig, A., Ajith, A., and Chen, D. Adapting language models to compress contexts. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 3829–3846, 2023.

Child, R. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

Choromanski, K. M., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J. Q., Mohiuddin, A., Kaiser, L., Belanger, D. B., Colwell, L. J., and Weller, A. Rethinking attention with performers. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id $=$ Ua6zuk0WRH.

Dao, T., Fu, D., Ermon, S., Rudra, A., and Re, C. Flashat-´ tention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

Feng, G., Luo, S., Hua, K., Zhang, G., Huang, W., He, D., and Cai, T. In-place test-time training. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id $=$ dTWfCLSoyl.

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., Presser, S., and Leahy, C. The Pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Geiger, A., Lu, H., Icard, T., and Potts, C. Causal abstractions of neural networks. Advances in Neural Information Processing Systems, 34:9574–9586, 2021.

Geiger, A., Wu, Z., Potts, C., Icard, T., and Goodman, N. Finding alignments between interpretable causal variables and distributed neural representations. In Causal Learning and Reasoning, pp. 160–187. PMLR, 2024.

Gliwa, B., Mochol, I., Biesek, M., and Wawer, A. SAM-Sum corpus: A human-annotated dialogue dataset for abstractive summarization. In Proceedings of the 2nd Workshop on New Frontiers in Summarization, pp. 70–79, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/ v1/D19-5409. URL https://www.aclweb.org/ anthology/D19-5409.

Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. In First conference on language modeling, 2024.

Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., and Re, C. Combining recurrent, convolutional, and ´ continuous-time models with linear state space layers. Advances in neural information processing systems, 34: 572–585, 2021.

Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., and Ginsburg, B. RULER: What’s the real context size of your long-context language models? In First Conference on Language Modeling, 2024. URL https: //openreview.net/forum?id=kIoBbc76Sy.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. R. SWE-bench: Can language models resolve real-world github issues? In The Twelfth

International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id $\ c =$ VTF8yNQM66.

Tishby, N., Pereira, F. C., and Bialek, W. The information bottleneck method. arXiv preprint physics/0004057, 2000.

Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.

Liu, H., Zaharia, M., and Abbeel, P. Ringattention with blockwise transformers for near-infinite context. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/ forum?id $\underline { { \underline { { \mathbf { \Pi } } } } } =$ WsRHpHH4s0.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., \` Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview. net/forum?id $=$ Bkg6RiCqY7.

Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024.

Merrill, W., Petty, J., and Sabharwal, A. The illusion of state in state-space models. In Forty-first International Conference on Machine Learning, 2024. URL https: //openreview.net/forum?id $\underline { { \underline { { \mathbf { \Pi } } } } } =$ QZgo9JZpLq.

Zhang, X., Zhao, J. J., and LeCun, Y. Character-level convolutional networks for text classification. In NIPS, 2015.

Mu, J., Li, X., and Goodman, N. Learning to compress prompts with gist tokens. Advances in Neural Information Processing Systems, 36:19327–19352, 2023.

Park, J. S., O’Brien, J., Cai, C. J., Morris, M. R., Liang, P., and Bernstein, M. S. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, pp. 1–22, 2023.

Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S., Cao, H., Cheng, X., Chung, M., Grella, M., et al. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048, 2023.

Scholkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalch-¨ brenner, N., Goyal, A., and Bengio, Y. Toward causal representation learning. Proceedings of the IEEE, 109(5): 612–634, 2021.

Sellam, T., Das, D., and Parikh, A. Bleurt: Learning robust metrics for text generation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2020.

Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., Dubois, Y., Chen, X., Wang, X., Koyejo, S., Hashimoto, T., and Guestrin, C. Learning to (learn at test time): RNNs with expressive hidden states. In Fortysecond International Conference on Machine Learning, 2025. URL https://openreview.net/forum? id $=$ wXfuOj9C7L.
