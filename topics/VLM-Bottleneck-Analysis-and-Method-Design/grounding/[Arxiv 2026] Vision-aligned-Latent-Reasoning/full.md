# Vision-aligned Latent Reasoning for Multi-modal Large Language Model

Byungwoo Jeon <sup>1</sup> Yoonwoo Jeong <sup>2</sup> Hyunseok Lee <sup>1</sup> Minsu Cho <sup>2</sup> <sup>3</sup> <sup>\*</sup> Jinwoo Shin <sup>1</sup> <sup>3</sup> <sup>\*</sup>

## Abstract

Despite recent advancements in Multi-modal Large Language Models (MLLMs) on diverse understanding tasks, these models struggle to solve problems which require extensive multistep reasoning. This is primarily due to the progressive dilution of visual information during long-context generation, which hinders their ability to fully exploit test-time scaling. To address this issue, we introduce Vision-aligned Latent Reasoning (VaLR), a simple, yet effective reasoning framework that dynamically generates vision-aligned latent tokens before each Chainof-Thought reasoning step, guiding the model to reaso based on perceptual cues in the latent space. Specifically, VaLR is trained to preserve visual knowledge during reasoning by aligning intermediate embeddings of MLLM with those from vision encoders. Empirical results demonstrate that VaLR consistently outperforms existing approaches across a wide range of benchmarks requiring long-context understanding or precise visual perception, while exhibiting test-time scaling behavior not observed in prior MLLMs. In particular, VaLR improves the performance significantly from 33.0% to 52.9% on VSI-Bench, achieving a 19.9%p gain over Qwen2.5-VL. Code is available at project page.

## 1. Introduction

Multi-modal Large Language Models (MLLMs) have achieved remarkable success in various multi-modal tasks such as image captioning (Zhang et al., 2024; Cheng et al., 2025) and visual question answering (Manmadhan & Kovoor, 2020; Huynh et al., 2025). Beyond these tasks, there is a growing demand to deploy MLLMs in more complex applications that require multi-step reasoning and longhorizon planning, such as computer-use agents (CUA) (Anthropic, 2024a;b) and Vision-Language-Action (VLA) models (Kim et al., 2024; Black et al., 2024; Lee et al., 2025; Bjorck et al., 2025). A core challenge in such applications lies in integrating perceptual information into multi-step logical reasoning within MLLM architectures.

In the language domain, Chain-of-Thought (CoT) (Wei et al., 2022) has emerged as a cornerstone for improving reasoning capabilities of LLMs, enabling LLMs to decompose intricate tasks into intermediate linguistic steps. Building on the success of CoT, recent studies (Zheng et al., 2025b; Li et al., 2025e) have extended this approach from LLMs to MLLMs. However, in contrast to the test-time scaling law (Snell et al., 2024) of LLMs, MLLMs frequently struggle with long-context reasoning due to the attenuation of visual signals as the generated sequence length increases.

To address this issue, recent research in MLLMs focuses on enhancing the long-context reasoning of MLLMs. For instance, a line of work strengthens text reasoning of MLLMs via supervised fine-tuning (Yue et al., 2023; Yu et al., 2023) or reinforcement learning (Wang et al., 2024b; Havrilla et al., 2024; Shao et al., 2024b; Yu et al., 2024). While these text-centric methods have shown significant progress, they still suffer from diminishing visual signals when generating long text sequences. Alternatively, several studies explicitly re-introduce visual information by interleaving visual tokens (Zheng et al., 2025b; Yang et al., 2025d; Yoon et al., 2025) or generating images (Wang et al., 2025a; Li et al., 2025e). Yet, these approaches rely on static single-view visual features and use them only as a fixed initial context. Throughout this work, we demonstrate that utilizing static visual features leads to the gradual loss of visual context, whereas dynamically allocating visual details at each reasoning stage ensures information preservation, thereby enabling robust long-context reasoning in MLLMs.

In this paper, we introduce Vision-aligned Latent Reasoning (VaLR), a novel multi-modal reasoning framework that generates vision-aligned latent tokens during the reasoning process, which is inspired by the latent reasoning LLM approach (Hao et al., 2024b). The core idea of VaLR is to inject learnable latent tokens before each text-based reasoning step, creating “visual checkpoints” that keep the reasoning process grounded in image details. Unlike standard text tokens, these latent tokens are explicitly supervised to learn consistency with the dense visual representations of the input image which is highly correlated with the subsequent reasoning step. Specifically, we introduce a two-stage curriculum learning framework to gradually equip MLLMs with latent reasoning capabilities. The first stage involves supervised fine-tuning on general vision question-answering datasets to learn fundamental multi-modal reasoning ability. In the second stage, we incorporate a new group of latent tokens before every CoT step. We then apply representation alignment (Yu et al., 2025) to these latent tokens with dense features extracted from the corresponding image frame by vision encoders, e.g., DINOv2/v3 (Oquab et al., 2023; Simeoni et al.´ , 2025), CLIP (Radford et al., 2021), and SigLIPv2 (Tschannen et al., 2025).

We demonstrate the effectiveness of VaLR through extensive evaluations on multiple Vision Question-Answering (VQA) datasets. Overall, VaLR exhibits superior performance over existing baselines on multiple VQA benchmarks. Specifically, on VSI-Bench (Yang et al., 2025b), VaLR boosts the accuracy of Qwen2.5-VL from 33.0% to 52.9%. Notably, as shown in Figure 2, VaLR successfully follows the testtime scaling law: the performance of VaLR improves in cases requiring longer reasoning, whereas baselines degrade under similar conditions. Furthermore, ablation studies suggest that VaLR can be used agnostically on several vision encoders, e.g., DINO, SigLIP, CLIP and even works with the standalone vision encoders of the original MLLM, i.e., Qwen2.5-VL encoder (Bai et al., 2025).

## 2. Related Works

Multi-modal Large Language Models (MLLMs). Recent advancements in MLLMs harness the inherent reasoning proficiency of LLMs to establish unified architectures designed to handle multiple modalities within a single framework. Pioneering studies integrate visual information into LLMs, predominantly utilizing either resamplers (Alayrac et al., 2022; Awadalla et al., 2023; Li et al., 2025d; Cha et al., 2024) or Q-Former (Li et al., 2023; Dai et al., 2023; Zhu et al., 2023; Lin et al., 2024). Despite the effectiveness of these specialized architectures, LLaVA (Liu et al., 2023; 2024a) and its successors (Chen et al., 2024a; Liu et al., 2024b; Chu et al., 2023; 2024; Bai et al., 2023a;b; Yang et al., 2024; 2025a) demonstrate that aligning each modality through a trainable lightweight projector is sufficient when paired with visual instruction tuning. Nevertheless, these models suffer from solving problems that require comprehensive reasoning, falling short of the reasoning capabilities exhibited by Chain-of-Thought (CoT).

Chain-of-Thought (CoT) and Latent Reasoning. The emergence of Chain-of-Thought (CoT) prompting has significantly enhanced the reasoning capabilities of large language models (LLMs) by decomposing complex problems into intermediate linguistic steps. While early works (Wei et al., 2022; Khot et al., 2022; Zhou et al., 2022) primarily relied on explicit prompting to derive these chains, subsequent research has focused on intrinsic enhancement through supervised fine-tuning (Yue et al., 2023; Yu et al., 2023) or reinforcement learning (Wang et al., 2024b; Havrilla et al., 2024; Shao et al., 2024b; Yu et al., 2024). To expand the search space of reasoning chains during inference, extensive studies have introduced tree-based (Xie et al., 2023; Yao et al., 2023; Hao et al., 2024a) and trajectory-based (Lehnert et al., 2024; Gandhi et al., 2024; Su et al., 2024) exploring algorithms. Motivated by the insight that natural language cannot encapsulate all forms of reasoning, recent paradigms have shifted toward operating directly within latent space (Hao et al., 2024b; Wang et al., 2025c; Li et al., 2025b) or learning to generate visual information from latent reasoning features (He et al., 2024; Li et al., 2025e). Orthogonally, several approaches (Zheng et al., 2025b; Yang et al., 2025d) propose to interleave visual tokens in reasoning trajectories to empower multi-modal reasoning. In this work, we align the latent reasoning tokens with features from vision encoders to facilitate visual reasoning within the latent space.

Leveraging External Vision Encoders in MLLMs. Recent MLLMs incorporate rich visual features, e.g., CLIP (Radford et al., 2021), DINO (Oquab et al., 2023; Simeoni et al.´ , 2025), SigLIP (Zhai et al., 2023), and VGGT (Wang et al., 2025b), to enhance their visual and spatial reasoning capabilities. For instance, PrismaticVLM (Karamcheti et al., 2024) integrates CLIP and DINO features through trainable projection layers to leverage rich visual representations. Similarly, PaliGemma (Beyer et al., 2024) exploits the dense features of SigLIP to enable comprehensive visual understanding with fewer parameters. To enhance the spatial awareness of MLLMs, several studies (Zheng et al., 2025a; Wu et al., 2025; Huang et al., 2025) leverage VGGT to inject token-wise spatial information. Concurrent with our work, CoVT (Qin et al., 2025a) and Monet (Wang et al., 2025c) enhance the visual understanding of MLLMs by leveraging rich visual features to perform reasoning directly within the visual space. However, as the reasoning chain lengthens, they suffer from diminishing visual signals since the visual information is only utilized as a fixed initial context. To alleviate this attenuation, our VaLR aligns latent tokens at each reasoning step with vision encoders, thereby enabling long-context visual reasoning.

## 3. Vision-aligned Latent Reasoning

We propose VaLR, an approach that aligns latent reasoning tokens with visual features to prevent visual signal decay, thereby enabling effective test-time scaling in MLLMs.

![](images/5e7b72b79adf6a7cee846f2899e0de536f7a10aa786acb938e6f1df97a3c2222.jpg)  
Figure 1. Overview of VaLR. Our framework, VaLR, generates vision-aligned latent tokens and language tokens throughout reasoning process. (a) During latent token generation, the last hidden states of MLLM becomes input embedding for the next token prediction. (b) To train the latent token generation, we align the intermediate features of MLLM with pre-trained visual representation extracted from external vision encoders. Note that we do not use the external vision encoder at test-time.

In Section 3.1, we first revisit the concept of latent reasoning in MLLMs. Then, in Section 3.2, we discuss how multimodal reasoning can be enhanced through representation alignment between MLLMs and vision encoders. Finally, Section 3.3 presents VaLR, a two-stage supervised finetuning (SFT) pipeline designed to gradually equip MLLMs with latent multi-modal reasoning capabilities. The overall pipeline of VaLR is illustrated in Figure 1.

## 3.1. Latent Reasoning in MLLMs

Formally, given an input text sequence $\boldsymbol { x } = ( x _ { 1 } , \dots , x _ { T } )$ and images I, we formulate the task as generating a corresponding text response. During inference with latent reasoning, the model iteratively switches between two distinct modes: latent and language. In detail, in the latent mode, the model produces latent reasoning tokens that are not directly shown as text, while in the language mode, it generates the response with text tokens.

Specifically, the native vision encoder first extracts image tokens from images I, i.e., $v = ( v _ { 1 } , v _ { 2 } , \cdot \cdot \cdot v _ { S } ) = \mathrm { V i T } ( \mathcal { T } )$ Subsequently, the transformer decoder processes input texttoken embeddings, $E _ { T } = [ v _ { 1 } , \cdot \cdot \cdot , v _ { S } , e ( x _ { 1 } ) , . . . , e ( x _ { T } ) ]$ , to yield the last hidden state $H _ { T } = \mathrm { T r a n s f o r m e r } ( E _ { T } )$ , where e is the token embedding function. During inference, the model enters the latent mode by predicting a special token <latent> and reverts to the language mode by predicting another special token $< \backslash ]$ latent>. In the latent mode, the model leverages the previous hidden state, $h _ { t } = H _ { t } [ t , : ]$ , as input for the next prediction, whereas in the language mode, the model uses the token embedding, $e ( x _ { t + 1 } )$ , as input for the next prediction, as formulated below:

$$
\begin{array} { r l } & { E _ { t + 1 } = \left\{ \begin{array} { l l } { [ E _ { t } ; h _ { t } ] } & { \mathrm { i f ~ l a t e n t ~ m o d e } , } \\ { [ E _ { t } ; e ( x _ { t + 1 } ) ] } & { \mathrm { i f ~ l a n g u a g e ~ m o d e } , } \end{array} \right. } \\ & { H _ { t + 1 } = \mathrm { T r a n s f o r m e r } ( E _ { t + 1 } ) , } \end{array}
$$

where $t > T$ . This recursive process repeats until the model predicts the <EOS> token. Upon entering the latent mode, the model is constrained to remain in this state for a fixed number K of steps. After K latent steps, the model reverts to the language mode and resumes generating text tokens from the current hidden state $h _ { t }$ , using the language model head, LM-Head:

$$
\begin{array} { r } { \mathcal { M } \left( x _ { t } | \boldsymbol { v } , \boldsymbol { x } _ { < t } \right) = \mathrm { L M } \mathrm { - } \mathrm { H e a d } \left( h _ { t } \right) , } \end{array}
$$

where $\mathcal { M }$ denotes the standard MLLM. This alternation strategy allows MLLMs to broaden its reasoning capability without explicit linguistic reasoning steps.

## 3.2. Latent Reasoning with Representation Alignment

To effectively leverage latent reasoning for visual grounding, we align hidden states of MLLM with visual features from pre-trained vision encoders during the latent mode. This alignment encourages the MLLM to maintain visual information throughout the recurrent reasoning process.

Alignment objective. For each reasoning stage $i ,$ we first select an image $I ^ { ( i ) } \in \mathcal { T }$ (details in Appendix B). We then extract patch-wise visual features from pre-trained vision encoder, ϕ, i.e., $\mathbf { F } _ { \phi } ^ { ( i ) } = \phi ( I ^ { ( i ) } ) \in \mathbb { R } ^ { P \times D }$ , where $P$ is the number of patches and D is the feature dimension. Afterward, we extract features from the intermediate layer of MLLM, i.e., $\mathbf { F } _ { \mathrm { M L L M } } ^ { ( i ) } = [ f _ { 1 } ^ { ( i ) } , \cdot \cdot \cdot , f _ { K } ^ { ( i ) } ]$ . We project these intermediate features through a learnable MLP ψ to match the dimension of vision encoder features:

![](images/62cedb62d5986d113156235b113b9720a0de322e6d16e679055d640cf401165d.jpg)  
Figure 2. Reasoning length-wise analysis. We investigate the effect of reasoning length on model performance across different MLLMs. We report hallucination rate on MMhalu (Sun et al., 2024) benchmark and accuracy (%) on MathVista (Lu et al., 2023), MathVision (Wang et al., 2024a), and MMVP (Tong et al., 2024b) benchmark. For MMhalu, lower is better. We observe that VaLR is the only method that exhibits consistent performance improvements as reasoning length increases, while remaining robust on long-horizon tasks.

$$
\hat { \mathbf { F } } _ { \mathrm { M L L M } } ^ { \left( i \right) } = \psi \left( { \mathrm { U p s a m p l e } \left( { \mathbf { F } } _ { \mathrm { M L L M } } ^ { \left( i \right) } \right) } \right) \in \mathbb { R } ^ { P \times D } ,
$$

where the ‘Upsample’ denotes an operation that aligns the image feature resolution of the MLLM with that of the pre-trained vision encoder. The representation alignment loss, $i . e . , \mathcal { L } _ { \mathrm { { R E P A } } }$ , encourages these projected latent features to align with the visual features using patch-wise cosine similarity throughout all latent reasoning stages:

$$
\mathcal { L } _ { \mathtt { R E P A } } : = - \frac { 1 } { N P } \sum _ { i = 1 } ^ { N } \sum _ { p = 1 } ^ { P } \sin \left( \hat { \mathbf { F } } _ { \mathrm { M L L M } } ^ { ( i ) } [ p , : ] , \mathbf { F } _ { \phi } ^ { ( i ) } [ p , : ] \right) ,
$$

where $\sin ( \cdot , \cdot )$ denotes the conventional cosine similarity function. By aligning with visual features, each latent token learns to encode visual information inherent in the image, thereby enabling comprehensive visual reasoning. Note that the alignment is applied only during training, while at inference time the model performs latent mode reasoning without REPA supervision, relying on learned visual grounding.

Multi-encoder Alignment. While alignment with a single vision encoder provides a robust visual foundation, we observe that leveraging multiple vision encoders enables the model to capture complementary visual representations. For instance, CLIP (Radford et al., 2021) and SigLIP (Tschannen et al., 2025) excel at semantic understanding, DINO (Oquab et al., 2023; Simeoni et al. ´ , 2025) capture finegrained appearance and spatial relationships, and $\pi ^ { 3 }$ (Wang et al., 2025d) encode 3D spatial structure. To leverage these complementary strengths, we extend our framework to incorporate multiple vision encoders simultaneously.

Let $\{ \phi _ { 1 } , \cdot \cdot \cdot , \phi _ { M } \}$ denotes a set of M frozen vision encoders. We extract features from each vision encoder for

each reasoning stage i:

$$
\mathbf { F } _ { \phi _ { m } } ^ { ( i ) } = \phi _ { m } ( I ^ { ( i ) } ) \in \mathbb { R } ^ { P _ { m } \times D _ { m } } \quad \mathrm { f o r } m = 1 , \cdots , M ,
$$

where $P _ { m }$ and $D _ { m }$ denote the varying number of patches and feature dimension across different vision encoders, respectively. For each vision encoder, we employ a separate learnable projection head $\psi _ { m }$ to match its feature dimension. The multi-encoder alignment loss is computed as the average of individual REPA losses:

$$
\mathcal { L } _ { \mathrm { R E P A } } ^ { \mathrm { m u l t i } } : = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathcal { L } _ { \mathrm { R E P A } } ^ { ( m ) } ,
$$

where each $\mathcal { L } _ { \mathrm { R E P A } } ^ { ( m ) }$ follows the same formulation as the single-encoder case but uses features from the m-th vision encoder, $\phi _ { m } .$ , and its corresponding projection head $\psi _ { m }$ . This multi-encoder approach allows the model to distill diverse visual knowledge into its latent reasoning space, enhancing both spatial awareness and general visual understanding.

## 3.3. Training Pipeline

We adopt a two-stage curriculum learning strategy to progressively foster latent reasoning in MLLMs. In the first stage, we perform standard supervised fine-tuning (SFT) on Chain-of-Thought (CoT) visual question-answering (VQA) datasets to establish foundational multi-modal reasoning capabilities. Subsequently, in the second stage, we decompose the reasoning into step-by-step phases and interleave latent reasoning tokens, allowing the model to reaso within the latent representations. Crucially, we employ representation alignment (REPA) to align the intermediate hidden states of the MLLM with features extracted from vision encoders such as DINO (Oquab et al., 2023; Simeoni et al.´ , 2025), CLIP (Radford et al., 2021), or SigLIP (Tschannen et al., 2025). This alignment empowers MLLMs to retain visual information required for reasoning, thereby enabling robust long-context reasoning.

Table 1. Main results on long-context evaluation. Accuracy (%) on multi-view VQA Benchmark, VSI-Bench (Yang et al., 2025b), for VaLR (Ours) and other baselines including several reasoning models, the base model, and latent reasoning models. We report 8 different sub-task accuracy and average (Avg.) accuracy. VaLR-S and VaLR-M denotes single encoder (DINOv3)-aligned model and multiple encoder (DINOv3, SigLIPv2, π<sup>3</sup>)-aligned model, respectively. The bold indicates the best result and underlined indicates the second best result within the group.
<table><tr><td>Method</td><td> $\operatorname { A v g } .$ </td><td>Obj. Cnt.</td><td>Abs. Dist.</td><td>Obj. Size</td><td>Room size</td><td>Rel. Dist.</td><td>Rel. Dir.</td><td>Route plan</td><td>Appr. Order</td></tr><tr><td colspan="10">Other Models</td></tr><tr><td>GPT-40</td><td>34.0</td><td>46.2</td><td>5.3</td><td>43.8</td><td>38.2</td><td>37.0</td><td>41.3</td><td>31.5</td><td>28.5</td></tr><tr><td>LLaVA-NeXT-Video-7B</td><td>35.6</td><td>48.5</td><td>14.0</td><td>47.8</td><td>24.2</td><td>43.5</td><td>42.4</td><td>34.0</td><td>30.6</td></tr><tr><td colspan="10">Reasoning Models</td></tr><tr><td>R1-OneVision-7B (Yang et al., 2025c)</td><td>16.1</td><td>15.0</td><td>1.7</td><td>0.5</td><td>2.8</td><td>26.5</td><td>40.0</td><td>24.2</td><td>14.7</td></tr><tr><td>Ocean-R1-7B (Lingfeng et al., 2025)</td><td>30.5</td><td>16.5</td><td>14.6</td><td>38.9</td><td>40.1</td><td>38.0</td><td>36.1</td><td>30.9</td><td>30.1</td></tr><tr><td colspan="10">Base Model</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al., 2025)</td><td>33.0</td><td>40.9</td><td>14.8</td><td>43.4</td><td>20.7</td><td>38.6</td><td>38.5</td><td>33.0</td><td>29.8</td></tr><tr><td>+ vanilla SFT</td><td>33.7</td><td>42.3</td><td>14.7</td><td>44.1</td><td>20.8</td><td>39.4</td><td>34.7</td><td>32.5</td><td>33.5</td></tr><tr><td colspan="10">Latent Reasoning Models</td></tr><tr><td>+ LVR (Li et al., 2025c)</td><td>18.4</td><td>21.4</td><td>3.6</td><td>1.4</td><td>9.0</td><td>35.1</td><td>30.9</td><td>32.0</td><td>23.1</td></tr><tr><td>+ CoVT (Qin et al., 2025b)</td><td>18.6</td><td>16.5</td><td>2.3</td><td>1.0</td><td>7.3</td><td>35.9</td><td>33.0</td><td>25.8</td><td>30.4</td></tr><tr><td>+ Monet (Wang et al., 2025c)</td><td>14.0</td><td>1.9</td><td>0.1</td><td>0.0</td><td>0.0</td><td>38.0</td><td>20.5</td><td>24.2</td><td>31.2</td></tr><tr><td>+ VaLR-S (Ours)</td><td>41.5</td><td>49.0</td><td>24.5</td><td>53.9</td><td>38.2</td><td>43.9</td><td>41.9</td><td>34.0</td><td>39.2</td></tr><tr><td>+ VaLR-M (Ours)</td><td>52.9</td><td>66.4</td><td>40.6</td><td>64.2</td><td>56.6</td><td>50.0</td><td>51.8</td><td>35.1</td><td>48.9</td></tr></table>

Stage 1: Standard SFT on CoT datasets. We perform standard SFT on pre-trained MLLMs using 450K samples from existing CoT datasets, endowing MLLMs with languagebased reasoning capabilities. Concretely, given a training sample with an input image set I, a question $q ,$ and groundtruth language CoT reasoning $\textbf { y } = ~ [ r ^ { 1 } , r ^ { 2 } , \cdot \cdot \cdot , r ^ { N }$ , a] where $r ^ { i }$ represents the i-th reasoning step and a is the final answer, we optimize the model using the standard autoregressive language modeling objective:

$$
\mathcal { L } _ { \mathrm { C E } } : = - \mathbb { E } _ { ( \mathcal { T } , q , y ) } \left[ \sum _ { t } \log \mathcal { M } ( y _ { t } | v , q , y _ { < t } ) \right] ,
$$

where $y _ { t }$ denotes the t-th token in the reasoning sequence. This stage establishes the fundamental ability to decompose complex visual questions into intermediate linguistic reasoning steps. During this stage, we only train the decoder of MLLM while freezing the native vision encoder.

Stage 2: Latent token training with REPA. Building on the standard CoT reasoning capabilities established in Stage 1, we introduce latent reasoning supervised by vision encoders in this stage. We first tailor existing CoT datasets for latent reasoning and then train the model on the tailored datasets using representation alignment (REPA) (Yu et al., 2025).

Specifically, each sample from existing CoT datasets consists of visual information $v ,$ a question q conditioned on visual input, a sequence of intermediate reasoning steps $\{ r ^ { ( i ) } \} _ { i = 1 } ^ { N }$ , where $N$ denotes the number of reasoning steps, and the corresponding answer a, $i . e . ,$

$$
v , q \to \left( r ^ { ( i ) } \right) _ { i = 1 } ^ { N } \to a .
$$

To adapt these datasets for latent reasoning, we insert $K$ latent tokens, $\{ \ell _ { k } ^ { ( i ) } \} _ { k = 1 } ^ { K }$ , before each language reasoning step $r ^ { ( i ) }$ . To inform the model when the latent mode should be initialized or terminated, we set the first and last tokens of each latent segment to special control tokens, i.e., $\ell _ { 1 } ^ { ( i ) } =$ <latent> and $\ell _ { K } ^ { ( i ) } = < / 1$ atent>. This transformation yields a latent-augmented reasoning sequence, which can be expressed as follows:

$$
v , q \to \big ( \ell _ { [ 1 : K ] } ^ { ( i ) } , r ^ { ( i ) } \big ) _ { i = 1 } ^ { N } \to a .
$$

In this stage, we extend the Stage 1 training objective with a REPA loss, i.e., $\mathcal { L } : = \mathcal { L } _ { \mathrm { C E } } + \lambda \mathcal { L } _ { \mathrm { R E P A } }$ . When we use multiple encoders for training, we apply the multi-REPA loss instead of the single-REPA loss, i.e., $\mathcal { L } : = \mathcal { L } _ { \mathrm { C E } } + \lambda \mathcal { L } _ { \mathrm { R E P A } } ^ { \mathrm { m u l t i } }$ We freeze the vision encoder and train only the MLLM decoder. Remark that the REPA loss ensures that the hidden states remain grounded in visual information.

## 4. Experiment

We provide an empirical evaluation of Vision-aligned Latent Reasoning (VaLR) by investigating following questions:

• Does VaLR improve the performance on VQA datasets? (Table 1, Table 2)

• Does VaLR retain performance during long-context reasoning? (Figure 2)

• Does the latent token component really contribute to longcontext reasoning? (Table 3)

• Can VaLR be adapted to various vision models and tasks in a model-agnostic manner? (Table 4, Table 5)

Table 2. Main results on perception evaluation. Accuracy (%) for VaLR and other baselines. including several reasoning models, base model and latent reasoning models. We consider perception evaluation benchmarks, including BLINK, MMVP, MM-Star, $\mathbf { V } ^ { \ast }$ , and CVBench. VaLR-S and VaLR-M denotes single encoder (DINOv3)-aligned model and multiple encoder (DINOv3, SigLIPv2, $\pi ^ { 3 } )$ )-aligned model, respectively. The bold indicates the best result and underlined indicates the second best result within the group.
<table><tr><td>Method</td><td>BLINK</td><td>MMVP</td><td>MMStar</td><td>V*</td><td>CVBench</td></tr><tr><td colspan="6">API models</td></tr><tr><td>GPT-40</td><td>63.0</td><td>68.7</td><td>65.2</td><td>42.9</td><td>79.2</td></tr><tr><td>Claude-4-Sonnet</td><td>39.6</td><td>48.7</td><td>58.8</td><td>15.2</td><td>76.3</td></tr><tr><td colspan="6">Reasoning models</td></tr><tr><td>R1-OneVision-7B (Yang et al.)</td><td>50.1</td><td>48.7</td><td>55.6</td><td>59.2</td><td>67.2</td></tr><tr><td>Ocean-R1-7B (Lingfeng et al.)</td><td>56.8</td><td>58.0</td><td>62.6</td><td>78.0</td><td>78.1</td></tr><tr><td colspan="6">Base model</td></tr><tr><td>Qwen2.5-VL-7B (Bai e al.)</td><td>55.7</td><td>56.0</td><td>67.1</td><td>76.4</td><td>74.5</td></tr><tr><td>+ vanilla SFT</td><td>56.6</td><td>58.7</td><td>67.5</td><td>78.0</td><td>77.0</td></tr><tr><td colspan="6">Latent Reasoning Models</td></tr><tr><td>+ LVR (Li et al.)</td><td>52.8</td><td>59.3</td><td>64.4</td><td>81.7</td><td>76.9</td></tr><tr><td>+ CoVT (Qin et al.)</td><td>56.0</td><td>58.7</td><td>69.2</td><td>78.0</td><td>80.0</td></tr><tr><td>+ Monet (Wang et al.)</td><td>49.1</td><td>50.0</td><td>53.3</td><td>83.3</td><td>71.1</td></tr><tr><td>+ VaLR-S (Ours)</td><td>63.1</td><td>60.3</td><td>70.8</td><td>86.4</td><td>83.1</td></tr><tr><td>+ VaLR-M (Ours)</td><td>64.7</td><td>60.3</td><td>72.3</td><td>86.9</td><td>87.6</td></tr></table>

## 4.1. Experimental Setup

Training Setup. For the main experiment, we trained VaLR on Qwen2.5-VL-7B (Bai et al., 2025). Unless mentioned otherwise, we use DINOv3 (Simeoni et al.´ , 2025) as the aligning vision encoder. For the analysis and multi-encoder alignment training setting, we additionally consider alternative vision encoders, e.g., DINOv2 (Oquab et al., 2023), CLIP (Radford et al., 2021), SigLIPv2 (Tschannen et al., 2025)), and $\pi ^ { 3 }$ (Wang et al., 2025d). We perform training on 450K scale of Chain-of-Thought (CoT) dataset for both training stages. We construct the dataset with the mixture of several open-source datasets, e.g., Zebra-CoT (Li et al., 2025a), CogCoM (Qi et al., 2025), ReFocus (Fu et al., 2025), Visual-CoT (Shao et al., 2024a), OneThinker-SFT (Feng et al., 2025), and GCoT (Chen et al., 2025). Further details are provided in Appendix A.1.

Evaluation Setup. Following the evaluation setup of previous benchmarks, we mainly report the accuracy (%) across all benchmarks. For response generation, we apply greedy sampling. The models are evaluated on various VQA benchmarks, including VSI-Bench, BLINK, MMVP, MMStar, MathVision, and more. In Appendix A.2, we provide additional results under specific evaluation settings.

Baselines. We compare VaLR with API, reasoning, supervised finetuned, and latent reasoning models in MLLMs, namely, GPT-4o, Claude-4-Sonnet, R1-OneVision-7B (Yang et al., 2025c), Ocean-R1-7B (Lingfeng et al., 2025), LVT (Li et al., 2025c), CoVT (Qin et al., 2025b), and Monet (Wang et al., 2025c).

Table 3. Effect of representation alignment component. We ablate the latent alignment training component of VaLR. We compare the VaLR without visual alignment (VA) and visual alignment with Qwen encoder (QE) and DINOv3 (Ours). We report accuracy (%) on VSI-Bench, BLINK, MMVP, V<sup>∗</sup>, and CVBench. The bold indicates the best result within the group.
<table><tr><td>Method</td><td>VSI-Bench</td><td>BLINK</td><td>MMVP</td><td>V*</td><td>CVBench</td></tr><tr><td>Qwen2.5-VL-7B</td><td>33.0</td><td>55.7</td><td>56.0</td><td>76.4</td><td>74.5</td></tr><tr><td>+ vanilla SFT</td><td>33.7</td><td>56.6</td><td>58.7</td><td>78.0</td><td>77.0</td></tr><tr><td>+ VaLR w/o VA</td><td>34.0</td><td>57.1</td><td>56.7</td><td>75.9</td><td>73.4</td></tr><tr><td>+ VaLR w/ QE</td><td>39.6</td><td>58.9</td><td>60.0</td><td>81.7</td><td>81.6</td></tr><tr><td>+ VaLR (Ours)</td><td>41.5</td><td>63.1</td><td>60.3</td><td>86.4</td><td>83.1</td></tr></table>

## 4.2. 3D Spatial Reasoning Tasks

We examine VaLR’s effectiveness in long-context reasoning by comparing performance on 3D multi-view benchmark, VSI-Bench (Yang et al., 2025b), which requires long-context reasoning ability to integrate spatial information across multiple viewpoints. We report accuracy on 8 sub-tasks and the average accuracy. We train the VaLR with two different setups: (i) VaLR-S, which aligns a single encoder using DI-NOv3 (Simeoni et al.´ , 2025) and (ii) VaLR-M, which aligns multiple encoders using DINOv3, SigLIPv2 (Tschannen et al., 2025), and $\pi ^ { 3 }$ (Wang et al., 2025d).

Results. As shown in Table 1, VaLR-S achieves an average accuracy of 41.5%, substantially outperforming the base model, Qwen2.5-VL (33.0%). In contrast, previous latent reasoning methods struggle on this benchmark requiring multi-view understanding. For example, Monet (Wang et al., 2025c) reaches 14% in average accuracy, and other models (Li et al., 2025c; Qin et al., 2025b) also collapse (see more details in Appendix C.1). This performance gap between VaLR-S and other latent reasoning methods provides strong evidence that latent reasoning without visual recall fails to maintain visual grounding during long reasoning traces, confirming the effectiveness of dynamic visual re-injection.

In addition, VaLR-M even achieves state-of-the-art performance (52.9%) over previous baselines, e.g., GPT-4o (34.0%) and Ocean-R1 (30.5%), highlighting that the combination of different vision encoders produces the synergistic effect. In particular, VaLR-M achieves remarkable performance on spatial understanding tasks, such as relative (50.0%) and absolute (40.6%) distance prediction. These results validate our hypothesis that latent reasoning with visual alignment prevents the visual information decay observed in standard reasoning approaches.

## 4.3. Perception Tasks

We further present that VaLR also improves performance on moderate-length reasoning tasks beyond long-context by evaluating it on five perception benchmarks. We report accuracy on BLINK (Fu et al., 2024), MMVP (Tong et al., 2024b), MMStar (Chen et al., 2024b), $\mathrm { V } ^ { \ast }$ (Wu & Xie, 2024), and CVBench (Tong et al., 2024a). We train the model with two different setups: (i) VaLR-S, which aligns a single encoder using DINOv3 (Simeoni et al.´ , 2025) and (ii) VaLR-M, which aligns multiple encoders using DINOv3, SigLIPv2 (Tschannen et al., 2025), and $\pi ^ { 3 }$ (Wang et al., 2025d).

Table 4. Ablation study for multi-encoder alignment. Accuracy (%) for VaLR ablation of different combinations of vision encoder. We consider DINOv3 (Simeoni et al.´ , 2025) and SigLIPv2 (Tschannen et al., 2025) for self-supervised vision encoders and $\pi ^ { 3 }$ (Wang et al., 2025d) for a geometry foundation model. We evaluate the model on 3D benchmark, $i . e .$ , VSI-Bench (Yang et al., 2025b), and perception benchmarks including BLINK (Fu et al., 2024), MMVP (Tong et al., 2024b), MMStar (Chen et al., 2024b), $\mathbf { V } ^ { * }$ (Wu & Xie, 2024), and CVBench (Tong et al., 2024a). Top row is a baseline, i.e., Qwen2.5-VL-7B. The bold indicates the best result within the group.
<table><tr><td colspan="3">Target Encoder</td><td rowspan="2">VSI-Bench</td><td rowspan="2">BLINK</td><td rowspan="2">MMVP</td><td rowspan="2">MMStar</td><td rowspan="2"> $\mathbf { V } ^ { * }$ </td><td rowspan="2">CVBench</td></tr><tr><td> $\pi ^ { 3 }$ </td><td>DINOv3</td><td>SigLIPv2</td></tr><tr><td>X</td><td>X</td><td>X</td><td>33.0</td><td>55.7</td><td>56.0</td><td>67.1</td><td>76.4</td><td>74.5</td></tr><tr><td> $\boldsymbol { v }$ </td><td>V</td><td>X</td><td>52.4</td><td>64.6</td><td>60.0</td><td>68.9</td><td>85.8</td><td>87.2</td></tr><tr><td>V</td><td>X</td><td>V</td><td>50.5</td><td>63.8</td><td>60.0</td><td>70.5</td><td>85.3</td><td>87.2</td></tr><tr><td>X</td><td>V</td><td>V</td><td>41.9</td><td>62.5</td><td>60.7</td><td>72.0</td><td>86.9</td><td>84.5</td></tr><tr><td>L</td><td>V</td><td>L</td><td>52.9</td><td>64.7</td><td>60.3</td><td>72.3</td><td>86.9</td><td>87.6</td></tr></table>

Results. As shown in Table 2, VaLR achieves substantial improvements over the base model. These results reveal that the learned visual grounding capability generalizes to improve short-context perception as well. The consistent advantages over other latent reasoning methods are particularly informative: VaLR-M outperforms CoVT by 8.7%p on BLINK and 8.9%p on $\mathbf { V } ^ { * }$ , and significantly surpasses Monet and LVR across all benchmarks. Interestingly, reasoning models such as R1-OneVision and Ocean-R1 show inconsistent results, with Ocean-R1 achieving strong $\mathbf { V } ^ { * }$ performance (78.0%) while underperforming on BLINK (56.8%) and MMVP (58.0%), suggesting that their reasoning enhancement overfit to specific task patterns rather than developing robust visual understanding. In contrast, VaLR’s consistent improvements across diverse perception tasks validate that our visual alignment strategy during latent reasoning provides a general mechanism for maintaining high-quality visual representations throughout the reasoning process, regardless of reasoning length.

## 4.4. Reasoning Length Analysis

To investigate whether VaLR follows the test-time scaling law, we analyze the performance as a function of reasoning length. We consider Ocean-R1 (Lingfeng et al., 2025), and LVR (Li et al., 2025c) as baselines and evaluate on Math-Vista (Lu et al., 2023), MathVision (Wang et al., 2024a), MMhalu (Sun et al., 2024), and MMVP (Tong et al., 2024b), grouping samples by generated reasoning length to observe performance trends. Note that Ocean-R1 is a reasoning model trained with standard CoT data.

Results. As illustrated in Figure 2, while all baseline methods peak at intermediate reasoning lengths and subsequently degrade, VaLR shows monotonic improvement across all benchmarks. In particular, on MMVP, VaLR sustains strong performance across all reasoning lengths while Ocean-R1 dramatically collapses from 62.7% to 56.5% at 300 tokens. This divergent behavior provides compelling evidence that models trained for language reasoning or naive latent reasoning progressively lose visual priors as they generate longer reasoning chains. These results validate that VaLR successfully maintain visual grounding during extended reasoning, enabling the model to benefit from longer thinking time rather than suffer from it. We thereby achieve true test-time scaling in the multi-modal domain as widely demonstrated for language models.

Table 5. Ablation study for single encoder alignment. Accuracy (%) for VaLR ablation of varying vision foundation model including CLIP, DINOv2/v3 and SigLIPv2 during training. We consider several VQA benchmarks including BLINK, MMVP, MMStar, $\mathbf { V } ^ { * }$ and CVBench. The bold indicates the best result and underlined indicates the second best result within the group.
<table><tr><td>Method</td><td>BLINK</td><td>MMVP</td><td>MMStar</td><td>V*</td><td>CVBench</td></tr><tr><td>Qwen2.5-VL-7B</td><td>55.7</td><td>56.0</td><td>67.1</td><td>76.4</td><td>74.5</td></tr><tr><td>+ CLIP</td><td>62.3</td><td>59.3</td><td>71.0</td><td>83.2</td><td>79.1</td></tr><tr><td>+ SigLIPv2</td><td>62.8</td><td>59.7</td><td>71.3</td><td>83.2</td><td>81.9</td></tr><tr><td>+ DINOv2</td><td>62.7</td><td>60.0</td><td>70.7</td><td>83.8</td><td>81.8</td></tr><tr><td>+ DINOv3</td><td>63.1</td><td>60.3</td><td>70.8</td><td>86.4</td><td>83.1</td></tr></table>

## 4.5. Ablation Study and Analysis

Effect of representation alignment. To verify the contribution of representation alignment (REPA) to VaLR’s performance, we conduct ablation studies to test if VaLR functions effectively without external vision encoders and REPA. Specifically, during training, we replaced DINOv3 (Simeoni´ et al., 2025) with Qwen’s native vision encoder (VaLR w/QE) and also trained VaLR without REPA (VaLR w/o VA). As shown in Table 3, VaLR trained with Qwen’s native encoder still consistently outperforms other baselines even without external alignment. These results indicate that VaLR is not reliant on external vision encoders, while incorporating them further enhances the performance. Additional results are provided in Appendix C.2.

![](images/3f6a771c8918cf643bc6d6f818960d04d673eb7f568d4ef71613183a8d37dc34.jpg)

![](images/8e243f2a4993cc07040bf3e9fca36caa5b98097528f3ec3ead43240219da9836.jpg)

![](images/085e9fd9178a54108827a8b2aa6dcef617f15149bcbc72b03a00e2c3d77939da.jpg)  
Figure 3. Effect of data scalability. We investigate the effect of the size of data and evaluate on VSI-Bench, BLINK, and $\mathbf { V } ^ { * }$ benchmark. Results are marked 10K, 50K, 100K, 200K, and 450K sample size with fixed iterations. The result show consistent and scalable performance improvements with increased data size across all benchmarks. Notably, VaLR achieves >20x faster convergence than vanilla SFT model on $\grave { \mathrm { V } } ^ { * }$ benchmark.

Alignment to different vision encoders. We further analyze whether VaLR can extend to other vision encoders for representation alignment, not limited to DINOv3. Specifically, we train the Qwen2.5-VL-7B model using VaLR with the self-supervised vision encoders including CLIP (Radford et al., 2021), SigLIPv2 (Tschannen et al., 2025), and DINOv2 (Oquab et al., 2023). As shown in Table 5, VaLR consistently outperforms the base model regardless of the vision encoder choice. We observe that VaLR consistently improves performance in a encoder-agnostic manner, and yields larger gains when paired with stronger vision encoders such as DINOv3.

Multi-encoder analysis. To extend our analysis from the observation in Section 4.2, we investigate whether the model can align with the distinct representational characteristics of each encoder. To verify this, we conduct a control experiment on various multi-encoder variations including $\pi ^ { 3 }$ DINOv3, and SigLIPv2 (more details in Section 3.3). As shown in Table 4, incorporating additional encoders consistently leads to performance gains. Notably, these improvements are closely aligned with the specific characteristics of each encoder’s representation. In detail, integrating the 3D-specialized encoder, $\pi ^ { 3 }$ significantly improves results on the 3D multi-view benchmark, VSI-Bench (Yang et al., 2025b). Moreover, adding 2D encoders, such as DINOv3 or SigLIPv2, enhances the performance across several perception benchmarks. Finally, integrating all three encoders achieves the best performance across all tasks. These results indicate that VaLR successfully aligns with distinct encoder representations by effectively leveraging their domain-specific strengths.

Alignment layer analysis. We investigate which intermediate layer of MLLMs is most effective for alignment via REPA. Specifically, we vary the layer index across three settings—Front (4th), Middle (12th), and Last (27th). As shown in Table 6, while all settings improve performance, REPA applied at the middle layer achieves the strongest results. This observation is consistent with prior studies (Yu et al., 2025; Kang et al., 2025; Jiang et al., 2025) indicating that visual information is most prominently represented in the middle layers of MLLMs.

Table 6. Ablation study for alignment layer. Accuracy (%) for VaLR ablation of different alignment layer of MLLM. We consider perception benchmarks consist with BLINK, MMVP, MMStar, V<sup>∗</sup>, and CVBench. Front, Middle, and Last denotes 4, 12, 27-th layer index in Qwen2.5-VL-7B, respectively.
<table><tr><td>Method</td><td>BLINK</td><td>MMVP</td><td>MMStar</td><td>V*</td><td>CVBench</td></tr><tr><td>Qwen2.5-VL-7B</td><td>55.7</td><td>56.0</td><td>67.1</td><td>76.4</td><td>74.5</td></tr><tr><td>Front</td><td>59.2</td><td>55.7</td><td>68.5</td><td>83.8</td><td>78.6</td></tr><tr><td>Middle</td><td>63.1</td><td>60.3</td><td>70.8</td><td>86.4</td><td>83.1</td></tr><tr><td>Last</td><td>62.8</td><td>60.0</td><td>70.8</td><td>85.3</td><td>82.5</td></tr></table>

Data Scalability. We investigate the data scalability of VaLR by tracking the performance of the checkpoints trained on different numbers of samples, e.g., 10K, 50K, 100K, 200K, and 450K. We compare the performance of Vanilla-SFT, VaLR-S, and VaLR-M on three benchmarks: VSI-Bench, BLINK, and $\mathbf { V } ^ { * }$ . As shown in Figure 3, both VaLR variants consistently improve performance as the sample size grows, while Vanilla-SFT saturates beyond 200K samples. Notably, our best model VaLR-M achieves > 20× faster training to reach comparable performance on $\mathbf { V } ^ { * }$ . This suggests that aligning with encoders facilitates learning richer features from the training data and improves data scalability.

## 5. Conclusion

In this paper, we have presented VaLR, a multi-modal reasoning framework that generates vision-aligned latent tokens during the reasoning process. Our experiments showed that VaLR performs test-time scaling behavior and consistently improves performance on various benchmarks that require a long or short context. We hope our work will facilitate future research on reasoning in multi-modal large language models.

## Impact Statement

Recent advancements of Multi-modal Large Language Models (MLLMs) have enabled remarkable performance in vision question answering. However, these models suffer from dilution of visual information during autoregressive text generation. This phenomenon is emphasized during long-context reasoning. Consequently, it prevents the use of MLLM in domains that require long-context reasoning, such as the Vision Language Action (VLA) model and the Computer Use Agent (CUA).

Our work addresses these challenges by proposing an effective way to inject a visual checkpoint with a latent token. Our approach improves long-context reasoning with testtime scalability and general VQA performance. We believe VaLR suggests future directions for mitigating the dilution of vision information.

In context of applications, alleviating the long-context reasoning reveals the use of MLLM in much more complex tasks, especially when visual information is involved, such as a robot with VLA or proactive CUA. This automatic agentic system can facilitate the innovation of human society.

## Acknowledgments

This work was partly supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2019-II190075, Artificial Intelligence Graduate School Program (KAIST); No. RS-2025-02653113, High-Performance Research AI Computing Infrastructure Support at the 2 PFLOPS Scale; No. RS-2024-00509279, Global AI Frontier Lab). We are grateful to the RLWRLD Inc. for generously providing compute resources that supported a significant portion of the experiments conducted in this work. We also thank Min-Hung Chen and Ryo Hachiuma for providing constructive feedback and suggestions in conducting the experiments.

## References

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for fewshot learning. In NeurIPS, volume 35, pp. 23716–23736, 2022.

Anthropic. The claude 3 model family: Opus, sonnet, haiku. Technical report, Anthropic, 2024a. URL https://ww w-cdn.anthropic.com/de8ba9b01c9ab7cba bf5c33b80b7bbc618857627/Model\_Card\_C laude\_3.pdf.

Anthropic. Claude 3.5 sonnet model card. Technical report,

Anthropic, 2024b. URL https://www-cdn.anthr opic.com/fed9cc193a14b84131812372d8d 5857f8f304c52/Model\_Card\_Claude\_3\_Ad dendum.pdf.

Awadalla, A., Gao, I., Gardner, J., Hessel, J., Hanafy, Y., Zhu, W., Marathe, K., Bitton, Y., Gadre, S., Sagawa, S., et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023.

Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., Hui, B., Ji, L., Li, M., Lin, J., Lin, R., Liu, D., Liu, G., Lu, C., Lu, K., Ma, J., Men, R., Ren, X., Ren, X., Tan, C., Tan, S., Tu, J., Wang, P., Wang, S., Wang, W., Wu, S., Xu, B., Xu, J., Yang, A., Yang, H., Yang, J., Yang, S., Yao, Y., Yu, B., Yuan, H., Yuan, Z., Zhang, J., Zhang, X., Zhang, Y., Zhang, Z., Zhou, C., Zhou, J., Zhou, X., and Zhu, T. Qwen technical report, 2023a. URL https://arxiv.org/abs/23 09.16609.

Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023b. URL https://ar xiv.org/abs/2308.12966.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Beyer, L., Steiner, A., Pinto, A. S., Kolesnikov, A., Wang, X., Salz, D., Neumann, M., Alabdulmohsin, I., Tschannen, M., Bugliarello, E., et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.

Bjorck, J., Castaneda, F., Cherniadev, N., Da, X., Ding, R.,˜ Fan, L., Fang, Y., Fox, D., Hu, F., Huang, S., et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Cha, J., Kang, W., Mun, J., and Roh, B. Honeybee: Localityenhanced projector for multimodal llm. In CVPR, pp. 13817–13827, 2024.

Chen, L., Li, J., Dong, X., Zhang, P., He, C., Wang, J., Zhao, F., and Lin, D. Sharegpt4v: Improving large multi-modal models with better captions. In ECCV, pp. 370–387. Springer, 2024a.

Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., et al. Are we on the right way for evaluating large vision-language models? In NeurIPS, 2024b.

Chen, Y., Qi, Z., Zhang, W., Jin, X., Zhang, L., and Liu, P. Reasoning in space via grounding in the world. arXiv preprint arXiv:2510.13800, 2025.

Cheng, K., Song, W., Fan, J., Ma, Z., Sun, Q., Xu, F., Yan, C., Chen, N., Zhang, J., and Chen, J. Caparena: Benchmarking and analyzing detailed image captioning in the llm era. arXiv preprint arXiv:2503.12329, 2025.

Chu, X., Qiao, L., Lin, X., Xu, S., Yang, Y., Hu, Y., Wei, F., Zhang, X., Zhang, B., Wei, X., et al. Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886, 2(6):7, 2023.

Chu, X., Qiao, L., Zhang, X., Xu, S., Wei, F., Yang, Y., Sun, X., Hu, Y., Lin, X., Zhang, B., et al. Mobilevlm v2: Faster and stronger baseline for vision language model. arXiv preprint arXiv:2402.03766, 2024.

Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P. N., and Hoi, S. Instructblip: Towards generalpurpose vision-language models with instruction tuning. In NeurIPS, volume 36, pp. 49250–49267, 2023.

Duan, H., Yang, J., Qiao, Y., Fang, X., Chen, L., Liu, Y., Dong, X., Zang, Y., Zhang, P., Wang, J., et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pp. 11198– 11201, 2024.

Feng, K., Zhang, M., Li, H., Fan, K., Chen, S., Jiang, Y., Zheng, D., Sun, P., Zhang, Y., Sun, H., et al. Onethinker: All-in-one reasoning model for image and video. arXiv preprint arXiv:2512.03043, 2025.

Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N. A., Ma, W.-C., and Krishna, R. Blink: Multimodal large language models can see but not perceive. In ECCV, pp. 148–166. Springer, 2024.

Fu, X., Liu, M., Yang, Z., Corring, J., Lu, Y., Yang, J., Roth, D., Florencio, D., and Zhang, C. Refocus: Visual editing as a chain of thought for structured image understanding. In ICML, 2025.

Gandhi, K., Lee, D., Grand, G., Liu, M., Cheng, W., Sharma, A., and Goodman, N. D. Stream of search (sos): Learning to search in language. arXiv preprint arXiv:2404.03683, 2024.

Hao, S., Gu, Y., Luo, H., Liu, T., Shao, X., Wang, X., Xie, S., Ma, H., Samavedhi, A., Gao, Q., et al. Llm reasoners: New evaluation, library, and analysis of stepby-step reasoning with large language models. arXiv preprint arXiv:2404.05221, 2024a.

Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., and Tian, Y. Training large language models to reaso in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024b.

Havrilla, A., Du, Y., Raparthy, S. C., Nalmpantis, C., Dwivedi-Yu, J., Zhuravinskyi, M., Hambro, E., Sukhbaatar, S., and Raileanu, R. Teaching large language models to reaso with reinforcement learning. arXiv preprint arXiv:2403.04642, 2024.

He, L., Li, Z., Cai, X., and Wang, P. Multi-modal latent space learning for chain-of-thought reasoning in language models. In AAAI, volume 38, pp. 18180–18187, 2024.

Huang, X., Wu, J., Xie, Q., and Han, K. 3drs: Mllms need 3d-aware representation supervision for scene understanding. In NeurIPS, 2025.

Hurst, A., Lerer, A., Goucher, A. P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Huynh, N. D., Bouadjenek, M. R., Aryal, S., Razzak, I., and Hacid, H. Visual question answering: from early developments to recent advances–a survey. arXiv preprint arXiv:2501.03939, 2025.

Jiang, Z., Chen, J., Zhu, B., Luo, T., Shen, Y., and Yang, X. Devils in middle layers of large vision-language models: Interpreting, detecting and mitigating object hallucinations via attention lens. In CVPR, pp. 25004–25014, 2025.

Kang, S., Kim, J., Kim, J., and Hwang, S. J. Your large vision-language model only needs a few attention heads for visual grounding. In CVPR, pp. 9339–9350, 2025.

Karamcheti, S., Nair, S., Balakrishna, A., Liang, P., Kollar, T., and Sadigh, D. Prismatic vlms: Investigating the design space of visually-conditioned language models. In Forty-first International Conference on Machine Learning, 2024.

Khot, T., Trivedi, H., Finlayson, M., Fu, Y., Richardson, K., Clark, P., and Sabharwal, A. Decomposed prompting: A modular approach for solving complex tasks. arXiv preprint arXiv:2210.02406, 2022.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246, 2024.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Lee, J., Duan, J., Fang, H., Deng, Y., Liu, S., Li, B., Fang, B., Zhang, J., Wang, Y. R., Lee, S., et al. Molmoact: Action reasoning models that can reaso in space. arXiv preprint arXiv:2508.07917, 2025.

Lehnert, L., Sukhbaatar, S., Su, D., Zheng, Q., Mcvay, P., Rabbat, M., and Tian, Y. Beyond a\*: Better planning with transformers via search dynamics bootstrapping. arXiv preprint arXiv:2402.14083, 2024.

Li, A., Wang, C., Fu, D., Yue, K., Cai, Z., Zhu, W. B., Liu, O., Guo, P., Neiswanger, W., Huang, F., et al. Zebracot: A dataset for interleaved vision language reasoning. arXiv preprint arXiv:2507.16746, 2025a.

Li, B., Sun, X., Liu, J., Wang, Z., Wu, J., Yu, X., Chen, H., Barsoum, E., Chen, M., and Liu, Z. Latent visual reasoning. arXiv preprint arXiv:2509.24251, 2025b.

Li, B., Sun, X., Liu, J., Wang, Z., Wu, J., Yu, X., Chen, H., Barsoum, E., Chen, M., and Liu, Z. Latent visual reasoning. arXiv preprint arXiv:2509.24251, 2025c.

Li, B., Zhang, Y., Chen, L., Wang, J., Pu, F., Cahyono, J. A., Yang, J., Li, C., and Liu, Z. Otter: A multi-modal model with in-context instruction tuning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025d.

Li, C., Wu, W., Zhang, H., Xia, Y., Mao, S., Dong, L., Vulic, I., and Wei, F. Imagine while reasoning in space:´ Multimodal visualization-of-thought. arXiv preprint arXiv:2501.07542, 2025e.

Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pp. 19730–19742. PMLR, 2023.

Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., and Yuan, L. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the 2024 conference on empirical methods in natural language processing, pp. 5971–5984, 2024.

Lingfeng, M., Yadong, L., Song, C., Jianhua, X., Zenan, Z., and Chen, W. Ocean-r1: An open and generalizable large vision-language model enhanced by reinforcement

learning. https://github.com/VLM-RL/Ocea n-R1, 2025. Accessed: 2025-04-03.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. In NeurIPS, volume 36, pp. 34892–34916, 2023.

Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In CVPR, pp. 26296–26306, 2024a.

Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., and Lee, Y. J. Llavanext: Improved reasoning, ocr, and world knowledge, 2024b.

Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

Manmadhan, S. and Kovoor, B. C. Visual question answering: a state-of-the-art review. Artificial Intelligence Review, 53(8):5705–5745, 2020.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Qi, J., Ding, M., Wang, W., Bai, Y., Lv, Q., Hong, W., Xu, B., Hou, L., Li, J., Dong, Y., et al. Cogcom: A visual language model with chain-of-manipulations reasoning. In ICLR, 2025.

Qin, Y., Wei, B., Ge, J., Kallidromitis, K., Fu, S., Darrell, T., and Wang, X. Chain-of-visual-thought: Teaching vlms to see and think better with continuous visual tokens. arXiv preprint arXiv:2511.19418, 2025a.

Qin, Y., Wei, B., Ge, J., Kallidromitis, K., Fu, S., Darrell, T., and Wang, X. Chain-of-visual-thought: Teaching vlms to see and think better with continuous visual tokens. arXiv preprint arXiv:2511.19418, 2025b.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In ICML, pp. 8748–8763. PmLR, 2021.

Shao, H., Qian, S., Xiao, H., Song, G., Zong, Z., Wang, L., Liu, Y., and Li, H. Visual cot: Unleashing chain-ofthought reasoning in multi-modal language models. In NeurIPS, 2024a.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024b.

Simeoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab,´ M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm testtime compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Su, D., Sukhbaatar, S., Rabbat, M., Tian, Y., and Zheng, Q. Dualformer: Controllable fast and slow thinking by learning with randomized reasoning traces. arXiv preprint arXiv:2410.09918, 2024.

Sun, Z., Shen, S., Cao, S., Liu, H., Li, C., Shen, Y., Gan, C., Gui, L., Wang, Y.-X., Yang, Y., et al. Aligning large multimodal models with factually augmented rlhf. In ACL, pp. 13088–13110, 2024.

Tong, P., Brown, E., Wu, P., Woo, S., IYER, A. J. V., Akula, S. C., Yang, S., Yang, J., Middepogu, M., Wang, Z., et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In NeurIPS, 2024a.

Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., and Xie, S. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In CVPR, pp. 9568–9578, 2024b.

Tschannen, M., Gritsenko, A., Wang, X., Naeem, M. F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Wang, H., Zheng, A., Zhao, Y., Wang, T., Zheng, G., Zhang, X., and Zhang, Z. Reconstructive visual instruction tuning. In ICLR, 2025a.

Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., and Novotny, D. Vggt: Visual geometry grounded transformer. In CVPR, pp. 5294–5306, 2025b.

Wang, K., Pan, J., Shi, W., Lu, Z., Ren, H., Zhou, A., Zhan, M., and Li, H. Measuring multimodal mathematical reasoning with math-vision dataset. In NeurIPS, volume 37, pp. 95095–95169, 2024a.

Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., and Sui, Z. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In ACL, pp. 9426–9439, 2024b.

Wang, Q., Shi, Y., Wang, Y., Zhang, Y., Wan, P., Gai, K., Ying, X., and Wang, Y. Monet: Reasoning in latent visual space beyond images and language. arXiv preprint arXiv:2511.21395, 2025c.

Wang, Y., Zhou, J., Zhu, H., Chang, W., Zhou, Y., Li, Z., Chen, J., Pang, J., Shen, C., and He, T. π : Permutationequivariant visual geometry learning. arXiv preprint arXiv:2507.13347, 2025d.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, volume 35, pp. 24824–24837, 2022.

Wu, D., Liu, F., Hung, Y.-H., and Duan, Y. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747, 2025.

Wu, P. and Xie, S. V<sup>∗</sup>: Guided visual search as a core mechanism in multimodal llms. In CVPR, pp. 13084– 13094, 2024.

Xie, Y., Kawaguchi, K., Zhao, Y., Zhao, J. X., Kan, M.-Y., He, J., and Xie, M. Self-evaluation guided beam search for reasoning. In NeurIPS, volume 36, pp. 41618–41650, 2023.

Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., Liu, D., Huang, F., Dong, G., Wei, H., Lin, H., Tang, J., Wang, J., Yang, J., Tu, J., Zhang, J., Ma, J., Yang, J., Xu, J., Zhou, J., Bai, J., He, J., Lin, J., Dang, K., Lu, K., Chen, K., Yang, K., Li, M., Xue, M., Ni, N., Zhang, P., Wang, P., Peng, R., Men, R., Gao, R., Lin, R., Wang, S., Bai, S., Tan, S., Zhu, T., Li, T., Liu, T., Ge, W., Deng, X., Zhou, X., Ren, X., Zhang, X., Wei, X., Ren, X., Liu, X., Fan, Y., Yao, Y., Zhang, Y., Wan, Y., Chu, Y., Liu, Y., Cui, Z., Zhang, Z., Guo, Z., and Fan, Z. Qwen2 technical report, 2024. URL https://arxiv.org/abs/2407.10671.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z. Qwen3 technical report, 2025a. URL https: //arxiv.org/abs/2505.09388.

Yang, J., Yang, S., Gupta, A. W., Han, R., Fei-Fei, L., and Xie, S. Thinking in space: How multimodal large language models see, remember, and recall spaces. In CVPR, pp. 10632–10643, 2025b.

Yang, Y., He, X., Pan, H., Jiang, X., Deng, Y., Yang, X., Lu, H., Yin, D., Rao, F., Zhu, M., et al. R1-onevision: Advancing generalized multimodal reasoning through cross-

modal formalization. arXiv preprint arXiv:2503.10615, 2025c.

Yang, Z., Yu, X., Chen, D., Shen, M., and Gan, C. Machine mental imagery: Empower multimodal reasoning with latent visual tokens. arXiv preprint arXiv:2506.17218, 2025d.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. In NeurIPS, volume 36, pp. 11809–11822, 2023.

Yoon, H., Jung, J., Kim, J., Choi, H., Shin, H., Lim, S., An, H., Kim, C., Han, J., Kim, D., et al. Visual representation alignment for multimodal large language models. arXiv preprint arXiv:2509.07979, 2025.

Yu, F., Jiang, L., Kang, H., Hao, S., and Qin, L. Flow of reasoning: Efficient training of llm policy with divergent thinking. arXiv preprint arXiv:2406.05673, 1(2):6, 2024.

Yu, L., Jiang, W., Shi, H., Yu, J., Liu, Z., Zhang, Y., Kwok, J. T., Li, Z., Weller, A., and Liu, W. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.

Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., and Xie, S. Representation alignment for generation: Training diffusion transformers is easier than you think. In ICLR, 2025.

Yue, X., Qu, X., Zhang, G., Fu, Y., Huang, W., Sun, H., Su, Y., and Chen, W. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653, 2023.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In ICCV, pp. 11975–11986, 2023.

Zhang, J., Zhang, D., He, X., Gao, F., and Li, X. An overview of image captioning generation based on large language models. In International Conference on Haptics and Virtual Reality, pp. 50–56. Springer, 2024.

Zheng, D., Huang, S., Li, Y., and Wang, L. Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. arXiv preprint arXiv:2505.24625, 2025a.

Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., and Yu, X. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025b.

Zhou, D., Scharli, N., Hou, L., Wei, J., Scales, N., Wang,¨ X., Schuurmans, D., Cui, C., Bousquet, O., Le, Q., et al. Least-to-most prompting enables complex reasoning in

large language models. arXiv preprint arXiv:2205.10625, 2022.

Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

## A. Implementation Details

## A.1. Training Details

We adopt Qwen2.5-VL-7B (Bai et al., 2025) as the base model and perform the supervised fine-tuning. In stage 1, we freeze the vision encoder and train only the language model backbone. In stage 2, we continue to freeze the vision encoder while jointly training the language model and the MLP for alignment. Detailed hyperparameters are provided in Table 7. In both stages, we train the model with the same Chain-of-Thought (CoT) datasets. Details of dataset construction are provided in the Appendix B. All experiments are conducted with 4x NVIDIA Tesla A100s.

Table 7. Hyperparameters for Stage 1 and 2.
<table><tr><td>Hyperparameter</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>optimizer</td><td colspan="2">AdamW</td></tr><tr><td>deepspeed</td><td colspan="2">Zero-2</td></tr><tr><td>learning rate</td><td>1e-5</td><td>2e-6</td></tr><tr><td>MLP ψ learning rate</td><td></td><td>1e-5</td></tr><tr><td>per-GPU batch size</td><td>2</td><td></td></tr><tr><td>gradient accumulation steps</td><td>16</td><td></td></tr><tr><td>weight decay</td><td>0.01</td><td></td></tr><tr><td>epoch</td><td>1</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>warm-up ratio</td><td>0.03</td><td></td></tr><tr><td>latent tokens (K)</td><td></td><td>16</td></tr><tr><td>alignment weight (λ)</td><td></td><td>0.5</td></tr></table>

During training, we select CLIP (Radford et al., 2021), SigLIP (Tschannen et al., 2025), DINO (Oquab et al., 2023; Simeon´ et al., 2025) and $\pi ^ { 3 }$ (Wang et al., 2025d) for target of representation alignment. All vision encoders are ViT-L.

## A.2. Evaluation Details

We use VLMEvalKit (Duan et al., 2024) for all evaluations. We adapt LLM-as-a-Judge in our evaluation process and use GPT-4o (Hurst et al., 2024) as the judge. To evaluate Monet (Wang et al., 2025c), we follow the system prompt proposed by the Monet authors. For CoVT (Qin et al., 2025b), we use CoVT-7B-depth-seg-dino. We evaluate various models on VSI-Bench (Yang et al., 2025b) for 3D spatial reasoning tasks, BLINK (Fu et al., 2024), MMVP (Tong et al., 2024b), MMStar (Chen et al., 2024b), V<sup>∗</sup> (Wu & Xie, 2024), and CVBench (Tong et al., 2024a) for perception tasks, and MathVista (Lu et al., 2023), MathVision (Wang et al., 2024a), and MMhalu (Sun et al., 2024) for reasoning length-wise analysis. We follow the evaluation protocol specified by each benchmark. Specifically, for VSI-Bench, we select the number of frames for multi-view images as summarized in Table 8. In addition, we report the model versions used for API-based evaluation as follows:

• openai/gpt-4o-2024-08-06

• Claude/claude-sonnet-4-20250514

Table 8. Number of frames used in VSI-Bench evaluation.
<table><tr><td colspan="2">Methods # of Frames</td></tr><tr><td>GPT-40</td><td>16</td></tr><tr><td>LLaVA-NeXT-Video-7B</td><td>32</td></tr><tr><td>R1-OneVision-7B</td><td>32</td></tr><tr><td>Ocean-R1-7B</td><td>32</td></tr><tr><td>Qwen2.5-VL-7B</td><td>32</td></tr><tr><td>LVR</td><td>32</td></tr><tr><td>CoVT</td><td>32</td></tr><tr><td>Monet</td><td>32</td></tr><tr><td>VaLR (Ours)</td><td>32</td></tr></table>

## B. Dataset Construction

To train VaLR, we use a collection of existing Chain-of-Thought (CoT) datasets including interleaved and non-interleaved datasets. We provide detailed statistics for our datasets in Appendix B.1. To enable latent reasoning, we tailor the existing datasets following the procedure outlined in the main paper. We elaborate more details for non-interleaved dataset in Appendix B.2, and for interleaved dataset in Appendix B.3.

## B.1. Data Statistics

We collect 450K samples from the mixture of several open-source datasets. Specifically, for the 125K interleaved CoT samples, we select Zebra-CoT (Li et al., 2025a), CogCoM (Qi et al., 2025), ReFocus (Fu et al., 2025), and Visual-CoT (Shao et al., 2024a). For the remaining 325K non-interleaved CoT data, we choose GCoT (Chen et al., 2025) with 170K samples from OneThinker-SFT (Feng et al., 2025).

## B.2. Non-interleaved CoT Data

Let an input image set be $\mathcal { T } = \{ I _ { 1 } , \cdots , I _ { Q } \}$ where $Q$ is the number of input images, and the ground-truth language CoT reasoning be $\mathbf { y } = [ r ^ { 1 } , r ^ { 2 } , \cdot \cdot \cdot , r ^ { N } , a ]$ where $r ^ { i }$ is the i-th reasoning step and a is the final answer.

Single-view VQA dataset. For single-view data where only one input image is given, we extract a visual representation from the input image $I _ { 1 }$ using a vision encoder. Before the reasoning step $r ^ { 1 }$ , the model learns latent token generation with representation alignment (REPA) between the MLLM and the extracted features. This ensures that the latent reasoning tokens are grounded in the visual features from the beginning of the reasoning process.

Multi-view VQA dataset. For CoT data with multiple input images, standard CoT datasets often do not explicitly specify which image should be recalled at each reasoning step. To address this issue, we employ GPT-4o (Hurst et al., 2024) to identify which image is most relevant for each reasoning step $r ^ { ( i ) }$ in the ground-truth CoT reasoning. Specifically, we process GPT-4o with the set of input images $\mathcal { T }$ and the CoT reasoning chain $\mathbf { y }$ , and ask it to match each reasoning step with its corresponding target image. After obtaining the target image $I _ { \mathrm { t a r g e t } }$ for each reasoning step $r ^ { ( i ) }$ , we apply REPA in the same manner as for single-view data, aligning the latent tokens with the visual features of the identified target image. The prompt used for multi-view data curation is provided below.

## Prompt for Multi-view Data Curation

This is a Chain-of-Thought (CoT) VQA data including multiple input images and corresponding CoT. Please divide the CoT step-by-step. Then find a useful and proper target image for each step. Lastly, place the target image in front of each reasoning step.

## B.3. Interleaved CoT Data

Let $I _ { i }$ be an i-th initial input image and $I _ { i } ^ { \mathrm { i n t e r } }$ be an interleaved input image. The input image set can be expressed as $\mathcal { T } = \{ I _ { 1 } , \cdot \cdot \cdot , I _ { Q } , I _ { 1 } ^ { \mathrm { i n t e r } } , \cdot \cdot \cdot , I _ { R } ^ { \mathrm { i n t e r } } \}$ where $Q$ and R be the number of initial input image and interleaved image, respectively. Additionally, let the ground-truth language CoT reasoning be $\mathbf { y } = [ r ^ { 1 } , r ^ { 2 } , \cdot \cdot \cdot , r ^ { N } , \bar { a } ]$ where $r ^ { j }$ is the j-th reasoning step and a is the final answer.

For interleaved data, images are inserted at specific positions within the CoT reasoning process. These interleaved images naturally indicate critical points in reasoning where visual information is needed. Therefore, we initiate the latent mode at the position where each interleaved image appears. Specifically, when an interleaved image $I _ { i } ^ { \mathrm { i n t e r } }$ is encountered before each reasoning step $r ^ { j }$ , we use the image as the target image for representation alignment. This approach allows the model to dynamically recall and integrate visual information at the exact moments specified by the dataset, effectively leveraging the explicit visual checkpoints provided in interleaved CoT data. By aligning the latent tokens with the features from $I _ { i } ^ { \mathrm { i n t e r } }$ at these designated positions, VaLR learns to naturally incorporate visual information when transitioning between reasoning steps.

## C. Additional Experimental Results

## C.1. Detailed Analysis in Table 1

In Table 1, we observe that latent reasoning baselines including LVR (Li et al., 2025c), CoVT (Qin et al., 2025b), and Monet (Wang et al., 2025c) collapse on the multi-view benchmark (Yang et al., 2025b). This occurs because existing approaches can only cover single-view scenarios or have limited extension to multi-view settings. As a result, these baselines show vulnerable performance on tasks requiring long-term visual memory. In contrast, our method, VaLR, is applicable to both single-view and multi-view scenarios and demonstrates a robust framework for long-context reasoning, achieving state-of-the-art performance on the multi-view benchmark.

## C.2. Detailed Analysis in Table 4

In Table 4, we report representation alignment (REPA) results for external vision encoders, $e . g . , \pi ^ { 3 }$ (Wang et al., 2025d), DINOv3 (Simeoni et al.´ , 2025), and SigLIPv2 (Tschannen et al., 2025). In Table 9, we additionally apply REPA to the base model’s native encoder, i.e., Qwen’s encoder. Similar to the findings in Table 4, the model that aligns with all encoders including Qwen’s encoder achieves the best overall performance, demonstrating the potential benefit obtained from additional alignment with stronger encoders.

Table 9. Detailed ablation study for multi-encoder alignment. Accuracy (%) for VaLR ablation of different combinations of vision encoder. We consider $\pi ^ { 3 }$ for a geometry foundation model, DINOv3 and SigLIPv2 for self-supervised vision encoder, and Qwen encoder. We evaluate the model on multi-view benchmark, i.e., VSI-Bench, and perception benchmarks including BLINK, MMVP, MMStar, V<sup>∗</sup>, and CVBench. Top row is a baseline, i.e., Qwen2.5-VL-7B. The bold indicates the best result within the group.
<table><tr><td colspan="4">Target Encoder</td><td rowspan="2">VSI-Bench</td><td rowspan="2">BLINK</td><td rowspan="2">MMVP</td><td rowspan="2">MMStar</td><td rowspan="2"> $\mathbf { V } ^ { * }$ </td><td rowspan="2">CVBench</td></tr><tr><td> $\pi ^ { 3 }$ </td><td>DINOv3</td><td>SigLIPv2</td><td>Qwen Enc.</td></tr><tr><td>X</td><td>x</td><td>X</td><td>X</td><td>33.0</td><td>55.7</td><td>56.0</td><td>67.1</td><td>76.4</td><td>74.5</td></tr><tr><td>V</td><td>V</td><td>X</td><td>x</td><td>52.4</td><td>64.6</td><td>60.0</td><td>68.9</td><td>85.8</td><td>87.2</td></tr><tr><td>V</td><td>X</td><td>V</td><td>x</td><td>50.5</td><td>63.8</td><td>60.0</td><td>70.5</td><td>85.3</td><td>87.2</td></tr><tr><td>V</td><td>x</td><td>X</td><td>V</td><td>51.6</td><td>63.5</td><td>60.3</td><td>67.6</td><td>84.7</td><td>85.9</td></tr><tr><td>x</td><td>V</td><td>V</td><td>x</td><td>42.0</td><td>62.5</td><td>60.7</td><td>72.0</td><td>86.9</td><td>84.5</td></tr><tr><td>x</td><td>√</td><td>X</td><td>V</td><td>41.4</td><td>63.4</td><td>60.3</td><td>71.1</td><td>87.4</td><td>85.0</td></tr><tr><td>x</td><td>X</td><td>V</td><td>V</td><td>40.9</td><td>60.7</td><td>61.3</td><td>72.3</td><td>84.4</td><td>83.0</td></tr><tr><td>V</td><td>V</td><td>V</td><td>X</td><td>52.9</td><td>64.7</td><td>60.3</td><td>72.3</td><td>86.9</td><td>87.6</td></tr><tr><td>x</td><td>V</td><td>V</td><td>V</td><td>41.7</td><td>63.1</td><td>61.0</td><td>72.6</td><td>88.0</td><td>86.0</td></tr><tr><td>V</td><td>V</td><td>V</td><td>L</td><td>53.3</td><td>65.1</td><td>61.0</td><td>72.3</td><td>88.5</td><td>87.4</td></tr></table>

## C.3. Clock-time Reasoning Analysis

In this section, we analyze the computational cost incurred by the proposed additional components. We measure clock-time with batch size 1 on a single NVIDIA Tesla A100 GPU using vLLM (Kwon et al., 2023). We evaluate LVR (Li et al., 2025c), Monet (Wang et al., 2025c), and VaLR on 32-view and 1-view scenarios from VSI-Bench (Yang et al., 2025b) and CVBench (Tong et al., 2024a), respectively.

Table 10. Clock-time reasoning analysis.
<table><tr><td>Method</td><td>32-view</td><td>1-view</td></tr><tr><td>Qwen2.5-VL</td><td>1.21</td><td>0.64</td></tr><tr><td>+ vanilla SFT</td><td>1.43</td><td>0.68</td></tr><tr><td>+LVR</td><td>1.49</td><td>0.66</td></tr><tr><td>+ Monet</td><td>1.51</td><td>0.79</td></tr><tr><td>+VaLR (Ours)</td><td>1.55</td><td>0.80</td></tr></table>

![](images/e32ef716062229004097c03c8d1d9ffcbc4bbdd2270332d4a45a921e3086f520.jpg)

## C.4. REPA vs. Input Tokens for Visual Representations

We investigate how effectively our method utilizes external vision encoders. Specifically, we compare our representation alignment (REPA) approach with the method using DINOv3 (Simeoni et al.´ , 2025) features as input tokens to the LLM backbone. As shown in Figure 4, REPA outperforms the input token method on VSI-Bench (Yang et al., 2025b), BLINK (Fu et al., 2024), and $\mathbf { V } ^ { * }$ (Wu & Xie, 2024) benchmark. In particular, unlike the input token method, VaLR does not require vision encoder at test-time, indicating that our method is highly efficient and effective.1) 2)

![](images/c89b2caf44161815088357507d1960fb2102e52558a9a90e6a6e38b8330997a8.jpg)

![](images/c2d6e5fe1bc0ab7b99a873710ff07b24b33621634e7af309fc0a2ac0ddf98551.jpg)  
Figure 4. Comparison between methods using visual representations. We compare two methods using DINOv3 features: (a) Using visual features as input visual tokens of MLLM (Green), (b) Aligning visual features with MLLM embeddings (Red). We report accuracy (%) on VSI-Bench, BLINK, and $\mathrm { V } ^ { \ast }$ benchmark.

## C.5. Feature Visualization

We visualize the changes in MLLM intermediate features through representation alignment. Features of VaLR are extracted from 12-th layer.

![](images/16647426ed2480b02be06e2c7afb9102a518e061ddfc4d073ac0b172bd2d403e.jpg)  
RGB image

![](images/5a321af041086c8d065ce23b9e9e75de3ecff8b9882ac1306ad1306b129d7055.jpg)  
DINOv3

![](images/49f108d7f7bad42df9a2700ae943c099731856b602db85f7800ef9777d326e09.jpg)  
Qwen2.5-VL

![](images/68ba3439f92846d0e9b04bd466ef5e6a7fce6344179725a6c86396a83fc08022.jpg)  
VaLR (Ours)  
Figure 5. Feature visualization.

## C.6. Additional Ablation Study

We provide additional ablation studies for VaLR. All experiments are conducted on the aligned model with a single encoder (Simeoni et al.´ , 2025).

Ablation study for λ. During training, we use the standard autoregressive loss and the representation alignment (REPA) loss, where λ is the weight for the REPA loss. We conduct an ablation study to examine the effect of λ. As shown in Table 11, VaLR achieves the best performance at $\lambda = 0 . 5$ , demonstrating a good balance where language semantics are preserved while maintaining visual alignment.

Table 11. Ablation study for λ.
<table><tr><td>λ</td><td>VSI-Bench</td><td>BLINK</td><td> $\mathbf { V } ^ { * }$ </td></tr><tr><td>0.0</td><td>34.0</td><td>57.1</td><td>75.9</td></tr><tr><td>0.25</td><td>38.5</td><td>61.7</td><td>78.5</td></tr><tr><td>0.5</td><td>41.5</td><td>63.1</td><td>86.4</td></tr><tr><td>0.75</td><td>40.4</td><td>62.8</td><td>84.8</td></tr><tr><td>1.0</td><td>40.0</td><td>60.9</td><td>84.3</td></tr></table>

Ablation study for K. We investigate the effect of the number of latent tokens K, which determines the resolution of vision-aligned features in the MLLM’s latent mode. As shown in Table 12, we observe that increasing the number of latent tokens has clear advantage.

Table 12. Ablation study for K.
<table><tr><td>K</td><td>VSI-Bench</td><td>BLINK</td><td> $\mathrm { V } ^ { \ast }$ </td></tr><tr><td>1</td><td>33.9</td><td>58.6</td><td>78.0</td></tr><tr><td>4</td><td>34.5</td><td>62.0</td><td>85.3</td></tr><tr><td>9</td><td>39.7</td><td>62.8</td><td>85.3</td></tr><tr><td>16</td><td>41.5</td><td>63.1</td><td>86.4</td></tr><tr><td>25</td><td>41.7</td><td>63.2</td><td>87.4</td></tr></table>