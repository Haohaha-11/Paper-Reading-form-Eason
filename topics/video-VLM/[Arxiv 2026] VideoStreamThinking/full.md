# Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously

Yiran Guan<sup>1∗</sup> Liang Yin<sup>1∗</sup> Dingkang Liang<sup>1</sup> Jianzhong Ju<sup>2</sup> Zhenbo Luo<sup>2</sup> Jian Luan<sup>2</sup> Yuliang Liu<sup>1</sup> Xiang Bai<sup>1B</sup>

<sup>1</sup>Huazhong University of Science and Technology <sup>2</sup>MiLM Plus, Xiaomi Inc. \* Equal contribution. <sup>B</sup> Corresponding author. {yiranguan,liangyin,dkliang,xbai}@hust.edu.cn

Abstract. Online Video Large Language Models (VideoLLMs) play a critical role in supporting responsive, real-time interaction. Existing methods focus on streaming perception, lacking a synchronized logical reasoning stream. However, directly applying test-time scaling methods incurs unacceptable response latency. To address this trade-of, we pro pose Video Streaming Thinking (VST), a novel paradigm for streaming video understanding. It supports a thinking while watching mechanism, which activates reasoning over incoming video clips during streaming. This design improves timely comprehension and coherent cognition while preserving real-time responsiveness by amortizing LLM reasoning latency over video playback. Furthermore, we introduce a comprehensive posttraining pipeline that integrates VST-SFT, which structurally adapts the ofline VideoLLM to causal streaming reasoning, and VST-RL, which provides end-to-end improvement through self-exploration in a multi-turn video interaction environment. Additionally, we devise an automated training-data synthesis pipeline that uses video knowledge graphs to generate high-quality streaming QA pairs, with an entity–relation grounded streaming Chain-of-Thought to enforce multi-evidence reasoning and sustained attention to the video stream. Extensive evaluations show that VST-7B performs strongly on online benchmarks, e.g. 79.5% on StreamingBench and 59.3% on OVO-Bench. Meanwhile, VST remains competitive on ofline long-form or reasoning benchmarks. Compared with Video-R1, VST responds 15.7× faster and achieves +5.4% improvement on VideoHolmes, demonstrating higher eficiency and strong generalization across diverse video understanding tasks. Code, data, and models will be released at https://github.com/1ranGuan/VST.

Keywords: Streaming Video Understanding · CoT · VideoLLM

## 1 Introduction

Online video understanding enables Video Large Language Models (VideoLLMs) to interpret streaming visual inputs and respond in real time, making it particularly valuable for embodied intelligence and interactive AI assistants [3, 7]. Unlike ofline methods that benefit from post-hoc global access to the entire video [1, 22, 44], the core challenges of online video understanding lie in strict temporal causality, real-time processing, and a finite context window.

![](images/21a7b95c3be0763213271697c0fed7b2418600a3c655a42ee43e5723c032747a.jpg)  
Fig. 1: Benchmark results and paradigm comparison. (a) VST-7B delivers strong performance on online and ofline video understanding benchmarks while maintaining low QA latency. (b) Existing streaming VideoLLMs focus on eficient streaming processing, but lack explicit analytical reasoning. (c) VideoLLM with CoT performs heavy postquery step-by-step reasoning to improve performance, but incurs high QA latency. (d) Our Video Streaming Thinking introduces proactive pre-query reasoning, interleaving it with video consumption to achieve both strong performance and eficient responsiveness.

Several prior methods have been proposed to address the challenges of online video understanding. As shown in Fig. 1(b), they primarily improve contextwindow eficiency by explicitly managing visual tokens for compression [35,48,51] or by retrieving from the KV cache [6, 28, 47]. However, these methods primarily focus on streaming perception and treat the management of visual features as a form of memory, with limited involvement of the LLM itself and no explicit reasoning or analytical deliberation. To fill this missing piece, one promising direction inspired by ofline video understanding is to apply test-time scaling via Chain-of-Thought (CoT) to elicit stronger reasoning ability [4, 8, 11, 12, 23, 52, 58], as shown in Fig. 1(c). Nevertheless, directly performing step-by-step reasoning after the user query can significantly increase QA response latency, making it dificult to meet strict real-time requirements in online scenarios.

In this paper, we introduce the Video Streaming Thinking (VST) to resolve the trade-of between explicit reasoning and real-time responsiveness, shifting the LLM backend from passive waiting to active, intermittent reasoning during video consumption. This design is inspired by insights from human cognition. Findings on neural coupling [16, 36] suggest that the logical flow in the brain synchronizes closely with the influx of external information, fostering the perception of current signals and their synthesis into a coherent understanding. Similarly, as illustrated in Fig. 1(d), our method continuously processes incoming video clips and produces intermediate thoughts in real time. This eliminates the need to defer heavy computation until the query arrives, which is a common limitation of ofline VideoLLMs with CoT [4,8,40]. This thinking while watching mechanism maintains a coherent internal state over the stream, ensuring that the final response is grounded in a deeply processed understanding of the historical context. By frontloading and amortizing the reasoning cost ahead of query arrival, VST preserves the low QA latency required in streaming scenarios.

We instantiate this paradigm with a dedicated post-training pipeline that combines supervised fine-tuning (VST-SFT) and reinforcement learning (VST-RL). Concretely, we cast streaming thinking as a multi-turn conversation, where the model incrementally writes textual thoughts to an external memory while observing incoming video clips under a constrained visual context window. In the VST-SFT stage, we align the model with the desired streaming reasoning protocol by learning from of-policy demonstrations that strictly respect temporal causality, thereby bootstrapping its basic thinking-while-watching capability. Building upon this initialization, the VST-RL stage performs end-to-end reinforcement learning with verifiable rewards, encouraging the model to make intermediate reasoning steps that improve downstream question answering under realistic online conditions.

Due to the scarcity of existing data for video streaming thinking, we develop an automated synthesis pipeline to support our training, particularly the VST-SFT stage that requires high-quality reasoning demonstrations. Specifically, we model entities and their temporal relationships within long videos as knowledge graphs. By sampling paths from these graphs to form evidence chains, we prompt an ofline VideoLLM to generate complex QA pairs and their corresponding intermediate CoTs. This design enforces multi-hop reasoning across diverse visual evidence while ensuring strict alignment between the generated thoughts and the video context. Ultimately, we synthesize a large-scale dataset comprising 100K high-quality streaming reasoning samples.

We conducted extensive evaluations across multiple online and ofline video understanding benchmarks (see Fig. 1(a)). The results show that our method achieves state-of-the-art performance compared to existing online VideoLLMs, while remaining competitive on ofline video understanding benchmarks. Notably, VST performs particularly well on long-form videos that require comprehensive plot comprehension and multi-step reasoning. Moreover, compared to Video-R1, our method achieves higher accuracy while significantly reducing QA latency, demonstrating that VST is a viable test-time scaling approach that meets the requirements of streaming scenarios.

In summary, our main contributions are as follows:

• We propose the VST paradigm to interleave active explicit CoT generation with continuous video streams, enabling amortized test-time scaling with real-time responsiveness.

• A knowledge-graph-based data synthesis pipeline and a dedicated post-training recipe (VST-SFT and VST-RL) are introduced to adapt an ofline VideoLLM to streaming settings with strong streaming reasoning capabilities.

• Extensive evaluations across multiple online and ofline video understanding benchmarks demonstrate state-of-the-art performance. In addition, compared to ofline CoT VideoLLM, our method provides significantly lower QA latency.

![](images/be382a30cb04d65a2fe4e291fdae034f6992b6b0cf2eab532ba363a16c6a90e7.jpg)  
Fig. 2: Illustration of the Video Stream Thinking pipeline. The model employs a streaming thought mechanism to compress visual dynamics into a long-term textual memory. Combined with the short-term visual bufer, this enables eficient reasoning over indefinite video streams with fixed memory budgets.

## 2 Method

## 2.1 The Video Streaming Thinking (VST) Paradigm

We formulate VST as a multi-round video conversation task operating within a constrained context window, as illustrated in Fig. 2. Unlike previous online VideoLLMs, our model leverages streaming intervals before a user query to proactively reason about the content via autoregressive textual generation. This process synthesizes key visual details and event dynamics into a dual-memory system: maintaining a short-term native video memory for the current visual context, while accumulating a long-term textual semantic memory of past events.

Formally, given a video stream $\nu ,$ , let $\mathbf { v } _ { i }$ denote the visual features for the i-th frame. We accumulate these incoming features into discrete clips $\mathbf { c } ^ { k } \mathbf { \Sigma } = \mathbf { \Sigma }$ $\{ \mathbf { v } _ { i } \} _ { i = \tau _ { k - 1 } + 1 } ^ { \tau _ { k } }$ , where the boundary $\tau _ { k }$ is set when the accumulated visual tokens reach the preset capacity L. At each interval $k ,$ conditioned on the current clip $\mathbf { c } ^ { k }$ and the accumulated memory $\mathbf { m } ^ { k - 1 }$ , the LLM generates a streaming thought $\mathbf { z } ^ { k }$ by sampling from the distribution $\mathbf { z } ^ { k } \sim p ( \mathbf { z } \mid \mathbf { c } ^ { k } , \mathbf { \bar { m } } ^ { k - 1 } )$ . Here, $\mathbf { z } ^ { k }$ summarizes the essential semantics of the current video segment, preserving the continuity of the overall thought process. For the long-term textual memory, we employ a memory update function $\mathbf { \bar { m } } ^ { k } = \mathrm { U p d a t e } ( \mathbf { m } ^ { k - 1 } , \mathbf { z } ^ { k } )$ , which adopts a simple first-in-first-out strategy to evict the earliest memory entries.

This iterative reasoning process continues until step K, when a user query q is received. Upon this trigger, the LLM generates the final response y based on the accumulated previous thoughts and the latest visual context. Consequently, the joint probability is decomposed as:

$$
p ( \mathbf { y } \mid \mathbf { q } , \boldsymbol { \mathcal { V } } ) = \underbrace { p ( \mathbf { y } \mid \mathbf { q } , \mathbf { c } ^ { K } , \mathbf { m } ^ { K } ) } _ { \mathrm { D i r e c t ~ A n s w e r } } \prod _ { k = 1 } ^ { K - 1 } \underbrace { p ( \mathbf { z } ^ { k } \mid \mathbf { c } ^ { k } , \mathbf { m } ^ { k - 1 } ) } _ { \mathrm { S t r e a m i n g ~ T h i n k i n g } } .\tag{1}
$$

![](images/6d5d820fb427f8f6743c58065f984a4b81fb1573ff73c0f106a8afaea49a3ae2.jpg)  
Fig. 3: Overview of the training pipeline. (a) VST-SFT applies a streaming attention mask to enforce temporal causality, restricting attention to the current visual bufer and history textual context. (b) VST-RL performs on-policy optimization via an agentic loop, improving the quality of streaming thoughts through verifiable rewards computed solely from the final answer.

This formulation yields two distinct advantages. 1) It amortizes the computational cost of Chain-of-Thought (CoT) generation over the pre-query phase. This strategy efectively achieves test-time scaling to boost performance without incurring additional latency at the moment of user interaction. 2) The sequential generation of thoughts naturally aligns with the temporal causality inherent in streaming videos. This structure facilitates the adaptation of ofline models to online scenarios by mirroring the progressive nature of the video stream.

## 2.2 Training Method for VST

To instantiate the VST paradigm introduced in Sec. 2.1, we develop a two-stage post-training pipeline that combines supervised fine-tuning (VST-SFT) and reinforcement learning (VST-RL), progressively endowing an ofline VideoLLM with streaming thinking capabilities. The VST-SFT stage adapts the ofline model to the temporal causality of streaming video, while learning reasoning capabilities from of-policy expert data. Subsequently, VST-RL transitions the model from of-policy imitation to on-policy RL, and refines these learned capabilities for further end-to-end improvement.

Stage 1: VST-SFT. We initiate the training pipeline with SFT to instill the streaming thought mechanism into the ofline VideoLLM. For a training instance, we explicitly formulate the sequence as:

$$
\begin{array} { r } { \boldsymbol { \mathcal { S } } = \Big ( \mathbf { m } ^ { 0 } , ( { \mathbf { c } } ^ { 1 } , { \mathbf { z } } ^ { 1 } ) , \ldots , ( { \mathbf { c } } ^ { K - 1 } , { \mathbf { z } } ^ { K - 1 } ) , { \mathbf { c } } ^ { K } , \mathbf { q } , \mathbf { y } \Big ) . } \end{array}\tag{2}
$$

Here, $\mathbf { m } ^ { 0 }$ denotes the initial memory, and $( \mathbf { c } ^ { k } , \mathbf { z } ^ { k } )$ represent the interleaved video clips and streaming thoughts. The sequence concludes with the final clip $\mathbf { c } ^ { K }$ , user query q, and ground truth response y.

To align with the streaming inference architecture, we apply a streaming video attention mask. As depicted in Fig. 3(a), this mask restricts the model’s attention to a fixed-size window of recent visual tokens, mirroring the short-term visual bufer used during inference. Specifically, let M be the additive attention mask. Let $\mathbb { I } _ { v } ( j ) \in \{ 0 , 1 \}$ indicate whether the j-th token is a visual token, and let L denote the visual bufer size. Therefore, the attention mask can be written as:

$$
M _ { i , j } = \left\{ \begin{array} { l l } { 0 , } & { j \leq i \mathrm { ~ a n d ~ } \left( \mathbb { I } _ { v } ( j ) = 0 \mathrm { ~ o r ~ } \sum _ { t = j + 1 } ^ { i } \mathbb { I } _ { v } ( t ) < L \right) } \\ { - \infty , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{3}
$$

In this way, the model can only access a sliding window of the latest $L$ visual tokens, while all non-visual tokens remain fully visible under the causal constraint. Furthermore, to accommodate context length constraints while handling long-form videos, we implement a temporal segmentation strategy. The original sequence $s$ is sliced into consecutive segments $\{ \mathbf { s } _ { n } \} _ { n = 1 } ^ { M }$ , defined as:

$$
\mathbf { s } _ { n } = { \left\{ \begin{array} { l l } { { \Big ( } \mathbf { m } ^ { n - 1 } , \{ ( \mathbf { c } ^ { k } , \mathbf { z } ^ { k } ) \} _ { k = T _ { n - 1 } + 1 } ^ { T _ { n } } { \Big ) } , } & { n < M } \\ { { \Big ( } \mathbf { m } ^ { n - 1 } , \{ ( \mathbf { c } ^ { k } , \mathbf { z } ^ { k } ) \} _ { k = T _ { n - 1 } + 1 } ^ { K - 1 } , \mathbf { c } ^ { K } , \mathbf { q } , \mathbf { y } { \Big ) } , } & { n = M } \end{array} \right. }\tag{4}
$$

where $T _ { n }$ denotes the cut-of index for the n-th segment. The memory state is updated recursively across segments following $\mathbf { m } ^ { n } = \mathrm { U p d a t e } ( \mathbf { m } ^ { n - 1 } , \{ \mathbf { z } ^ { k } \} _ { k = T _ { n - 1 } + 1 } ^ { T _ { n } } )$ During SFT, we apply the standard next-token prediction loss exclusively to the streaming thoughts $\{ { \bf z } ^ { k } \} _ { k = 1 } ^ { K - 1 }$ and the final response $\mathbf { y } ,$ , treating visual tokens and historical memory as conditioning inputs.

Stage 2: VST-RL. Building upon the supervised foundation, we introduce VST-RL to transition the model from of-policy imitation to on-policy selfimprovement. The RL training process consists of two main phases: trajectory rollout and policy gradient optimization.

As shown in the upper part of $\mathrm { F i g . \ 3 ( b ) }$ , the rollout phase operates as an agentic loop. The policy model interacts with the streaming environment to generate a trajectory $\check { \tau }$ following the predefined joint probability in Eq. (1), where the streaming thoughts $\hat { \mathbf { z } } ^ { k }$ and the final response $\hat { \mathbf { y } }$ are sequentially sampled from the sampling policy $\pi _ { \theta ^ { \prime } }$ . After collecting a group of N trajectories $\lbrace \breve { T } _ { i } \rbrace _ { i = 1 } ^ { N } ,$ we employ a GRPO [12,27,49,50] strategy to optimize the policy model. We compute the reward $\mathbf { r _ { i } }$ solely based on the final answer $\mathbf { y _ { i } }$ via verifiable reward functions. To encourage the model to generate useful streaming thoughts, the calculated advantage is assigned to all generated tokens within the entire trajectory $\mathcal { T } _ { i }$ . The policy gradient objective is calculated as:

$$
\mathcal { I } _ { \mathrm { R L } } ( \theta ) = \mathbb { E } _ { q \sim \mathcal { D } , \{ \mathcal { T } _ { i } \} _ { i = 1 } ^ { N } \sim \pi _ { \theta ^ { \prime } } ( \cdot | q ) } \left[ \frac { 1 } { \sum _ { i = 1 } ^ { N } | \mathcal { T } _ { i } | } \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { | \mathcal { T } _ { i } | } \left( \mathcal { L } _ { i , t } ^ { \mathrm { c l i p } } ( \theta ) - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } | | \pi _ { \mathrm { r e f } } ) \right) \right]\tag{5}
$$

$$
\mathcal { L } _ { i , t } ^ { \mathrm { c l i p } } ( \theta ) = \operatorname* { m i n } \left[ \gamma _ { t } ( \theta ) \hat { A } _ { i } , \mathrm { c l i p } \left( \gamma _ { t } ( \theta ) , 1 - \epsilon _ { \mathrm { l o w } } , 1 + \epsilon _ { \mathrm { h i g h } } \right) \hat { A } _ { i } \right] .\tag{6}
$$

![](images/c1db74b26259d4b40ea898936c4410b7d4a94da0e3c331bdbad2839913c72ce1.jpg)  
Fig. 4: Stream-Thought QA data curation pipeline. We incrementally extract video entities and relations to build a knowledge graph, sample multi-hop evidence chains, and use Gemini to generate streaming QA pairs with grounded streaming thoughts, followed by automatic filtering.

Where $| \mathcal { T } _ { i } |$ denotes the total number of generated tokens in trajectory $\mathcal { T } _ { i } , \gamma _ { t } ( \theta )$ represents the probability ratio between π<sub>θ</sub> and the sampling policy π<sub>θ</sub>′ at step $t ,$ $\hat { A } _ { i } = r _ { i } - \mathrm { m e a n } ( R )$ is the group relative advantage, and $\epsilon _ { \mathrm { l o w } } , \epsilon _ { \mathrm { h i g h } }$ are the clipping hyperparameters follow DAPO [50].

## 2.3 Data Synthesis Pipeline for VST

We generate a set of video streaming thought data to support VST training, motivated by the fact that most existing chain-of-thought (CoT) datasets target ofline VideoLLMs with a global, hindsight view of the entire video, making it dificult to avoid information leakage under causal streaming constraints. To this end, we introduce an automated data generation pipeline grounded in knowledge graphs. As illustrated in Fig. 4, the pipeline produces high-quality training examples with explicit reasoning paths through streaming video entity extraction, evidence chain sampling, and streaming thought QA synthesis.

Streaming Video Entity Extraction. To build a temporally consistent knowledge graph, we maintain an entity bank and extract triples from a sliding window over the video stream. We segment the video into N scene clips with PySceneDetect. For each incoming clip, an ofline VideoLLM (e.g., Gemini 3.0 flash) updates the entity bank by adding newly observed entities and relations as  head, relation, tail. When the window exceeds size W , we drop the oldest clip and retain the most recent W − 1 overlapping clips to preserve temporal continuity. The entity bank thus serves as a lightweight memory for consistent entity tracking and timeline-aligned graph construction.

Evidence Chain Sampling. After processing the whole video, the complete entity bank is refined using an LLM to filter out noise entities, such as duplicates and subtitles. Subsequently, NetworkX [13] is used to construct the knowledge graph, which represents the logical relationships between events in the video. To mine long-term causal dependencies, an initial node is randomly selected, and a depth-first search (DFS) is used to extract evidence chains. Each node in these chains contains detailed information about the head and tail entities, their relationship, timestamps, and scene descriptions, facilitating comprehensive reasoning over the video content. For each video, we sample multiple evidence chains, enforcing that the entity overlap between any two chains is below 10% to promote diversity.

Stream Thought QA Synthesis. The final phase leverages Gemini 3.0 flash as a data synthesizer. Conditioned on the video knowledge graph, the model first generates a streaming CoT rationale to actively reason over video events and dynamic content. Subsequently, aligned with a sampled evidence chain $\{ \mathbf { z } ^ { k } \} _ { k = 1 } ^ { K } ,$ it synthesizes a query q and the final answer y, necessitating multi-evidence reasoning that integrates the CoT with visual context. To ensure data fidelity, we apply a strict post-generation filtering rubric, including: world-knowledge check, format alignment, logical consistency, repetition check, and thought validation.

Curation of VST training set. Following the above procedure, we generate 100K streaming-thought examples with videos from LLaVA-Vid [56] and Video-Marathon [25]. In addition, our full supervised fine-tuning corpus for VST-SFT includes 50K open-ended QA instances randomly sampled from LLaVA-Vid. For VST-RL, we train on 11K sampled questions, including multiple-choice questions from LLaVA-Vid, Video-Marathon, and Onethinker [9], as well as counting questions from RepCount [17].

## 3 Experiment

## 3.1 Implementation Details

We adopt Qwen2.5-VL [1] as our base ofline VideoLLM, processing input videos at 2 fps. Both VST-SFT and VST-RL (7B model) training stages are conducted on 32 × 80GB VRAM GPUs, utilizing the datasets detailed in Sec. 2.3. The visual encoder and projection layer are frozen throughout the entire training process. For VST-SFT, each training sample follows a 128 second time limit, and overlong raw videos are segmented into clips following Eq. (4). For VST-RL, we employ verl [34] with vLLM [19] and FSDP [57] backend. We configure the rollout batch size to 256 with a group size of N = 8, and define the reward function based on the correctness of the final answer. Additionally, following LongVILA-R1 [4], we leverage the paralleled encoding strategy during rollout to pre-compute video embeddings. During testing, following StreamingForest [51], we cap each inference step (including streaming-think and the final answer) at 8,192 video tokens and limit the max thinking times to 4 for eficient evaluation. We conduct all evaluations using the lmms-eval framework [54].

## 3.2 Benchmarks

To demonstrate the efectiveness of our method, we conducted a comprehensive evaluation across five video understanding benchmarks. Specifically, Streaming-Bench [26] and OVO-Bench [29] are utilized for online video understanding, focusing on the model’s online reasoning capabilities and temporal awareness. VideoMME [10] serves as a comprehensive ofline benchmark covering diverse domains and varying video durations. LongVideoBench [42] is designed to evaluate the long-form video understanding capabilities, while Video-Holmes [5] emphasizes logical reasoning within video content.

Table 1: Comparison of ofline and online VideoLLMs on StreamingBench Real-Time understanding tasks.
<table><tr><td>Model</td><td>Venue</td><td>OP</td><td>CR</td><td>CS</td><td>ATP</td><td>EU</td><td>TR PR</td><td>SU</td><td>ACP</td><td>CT |</td><td>|Overall</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Gemini 1.5 pro [37]</td><td></td><td></td><td>79.0 80.5</td><td>83.5 79.7</td><td>80.0</td><td>84.7</td><td>77.8</td><td>64.2</td><td>72.0</td><td>48.7</td><td>75.7</td></tr><tr><td>GPT-4o [30]</td><td></td><td></td><td>77.1 80.5</td><td>83.9 76.5</td><td>70.2</td><td>83.8</td><td>66.7</td><td>62.2</td><td>69.1</td><td>49.2</td><td>73.3</td></tr><tr><td colspan="10">Open-source Offline Models</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>50.1</td><td>17.6</td><td>52.3</td></tr><tr><td>VILA-1.5-8B [24] LongVA-7B [55]</td><td>CVPR&#x27;24 TMLR&#x27;25</td><td>53.749.2 70.0</td><td>71.0 63.3 61.2</td><td>56.9 70.9</td><td>53.4 62.7</td><td>53.9 59.5</td><td>54.6 61.1</td><td>48.8 53.7</td><td>54.7</td><td>34.7</td><td>60.0</td></tr><tr><td>MiniCPM-v2.6-7B [18]</td><td>COLM&#x27;24</td><td>71.9</td><td>71.1</td><td>77.9 75.8</td><td>64.6</td><td>65.7</td><td>70.4</td><td>56.1</td><td>62.3</td><td>53.4</td><td>67.4</td></tr><tr><td>LLaVA-OV-7B [20]</td><td>TMLR&#x27;25</td><td>80.4</td><td>74.2</td><td>76.0 80.7</td><td>72.7</td><td>71.7</td><td>67.6</td><td>65.5</td><td>65.7</td><td>45.1</td><td>71.1</td></tr><tr><td>Qwen2.5-VL-7B [1]</td><td></td><td>78.3</td><td>80.5</td><td>78.9 80.5</td><td>76.7</td><td>78.5</td><td>79.6</td><td>63.4</td><td>66.2</td><td>53.2</td><td>73.7</td></tr><tr><td colspan="10">Open-source Online Models</td></tr><tr><td>Flash-VStream-7B [53]</td><td>ICCV&#x27;25</td><td>25.9</td><td>43.6 24.9</td><td>23.9</td><td>27.3</td><td>13.1</td><td>18.5</td><td>25.2</td><td>23.9</td><td>48.7</td><td>23.2</td></tr><tr><td>VideoLLM-online-8B [2]</td><td>CVPR&#x27;24</td><td>39.1 40.1</td><td>34.5</td><td>31.1</td><td>46.0</td><td>32.4</td><td>31.5</td><td>34.2</td><td>42.5</td><td>27.9</td><td>36.0</td></tr><tr><td>Dispider-8B [31]</td><td>CVPR&#x27;25</td><td>74.9</td><td>75.5 74.1</td><td>73.1</td><td>74.4</td><td>59.9</td><td>76.1</td><td>62.9</td><td>62.2</td><td>45.8</td><td>67.6</td></tr><tr><td>TimeChatOnline-7B [48]</td><td>MM&#x27;25</td><td>80.2</td><td>82.0</td><td>79.5 83.3</td><td>76.1</td><td>78.5</td><td>78.7</td><td>64.6</td><td>69.6</td><td>58.0</td><td>75.4</td></tr><tr><td>Streamforest-7B [51]</td><td>NeurIPS&#x27;25</td><td>83.182.8</td><td></td><td>82.7 84.3</td><td>77.5</td><td>78.2</td><td>76.9</td><td>69.1</td><td>75.6</td><td>54.4</td><td>77.3</td></tr><tr><td>VST-7B (ours)</td><td></td><td>85.4 82.0</td><td></td><td>86.4</td><td>89.1</td><td>74.2 87.2</td><td></td><td>82.4 73.1</td><td>73.9</td><td>47.3</td><td>79.5</td></tr></table>

## 3.3 Online Video Benchmark Results

As shown in Tabs. 1 and 2, we evaluate our model on two online benchmarks, StreamingBench and OVO-Bench. VST-7B achieves 79.5% on StreamingBench and 59.3% on OVO-Bench, clearly outperforming prior open-source streaming SOTA models, including Streamforest [51] (77.3%) on StreamingBench and Streamo [43] (57.9%) on OVO-Bench. Notably, despite being much smaller than proprietary models, our method surpasses GPT-4o and Gemini 1.5 pro on StreamingBench by 6.2% and 3.8%, respectively, and achieves comparable performance with GPT-4o on OVO-Bench. Beyond the overall scores, VST 7B is particularly strong on OVO-Bench’s Backward Tracing task, where it achieves 56.7%, outperforming Streamforest by +4.7%. This result indicates that our model can retain and retrieve historical information efectively, supporting sustained memory over streaming inputs. These results highlight the strength of our approach for streaming video understanding. We believe the gains stem from our VST paradigm and a tailored post-training recipe, which together improve the model’s ability.

Table 2: Comparison of ofline and online VideoLLMs on OVO-Bench.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Venue</td><td rowspan="2"></td><td colspan="4">Real-Time</td><td colspan="2">Backward</td><td colspan="2">Forward REC SSR CRR|Avg.|</td><td rowspan="2">Overall</td></tr><tr><td colspan="2">[OCR ACR ATR STU FPD OJR|Avg.|</td><td colspan="2"></td><td colspan="2">EPM ASI HLD| |Avg.</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Gemini 1.5 pro [37]</td><td></td><td>85.9 67.0</td><td>79.3</td><td>58.4 63.4</td><td>62.0</td><td>69.3 58.6</td><td>76.4 52.6</td><td>62.5 35.5 74.2</td><td>61.7</td><td>|57.2|</td><td>63.0</td></tr><tr><td>GPT-4o [30]</td><td></td><td>69.8 64.2</td><td>71.6</td><td>51.1 70.3</td><td>59.8</td><td>64.5 57.9</td><td>75.7 48.7</td><td>60.8</td><td>27.6 73.2 59.4</td><td>53.4</td><td>59.5</td></tr><tr><td colspan="10">Open-source Offline Models</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2-VL-72B [39] LLaVA-Video-7B [56]</td><td>TMLR&#x27;25</td><td>65.8 60.6</td><td>69.8</td><td>51.7 69.3</td><td>54.4</td><td>61.9 52.5</td><td>60.8 57.5 7.5</td><td>57.0</td><td>38.864.1 45.0 60.4</td><td>49.3 54.8</td><td>56.3 52.9</td></tr><tr><td>LLaVA-OV-7B [20]</td><td>TMLR&#x27;25</td><td>69.1 58.7 66.4 57.8</td><td>68.8</td><td>49.4 74.3</td><td>59.8</td><td>63.5 56.2</td><td>57.4 55.4 21.5</td><td>40.4 43.7</td><td>34.1 70.0 25.6 67.1 58.8</td><td>50.5</td><td>52.7</td></tr><tr><td>LongVU-7B [33]</td><td>ICML&#x27;25</td><td>53.7 53.2</td><td>73.3 62.9</td><td>53.4 47.8 68.3</td><td>71.3 62.0 59.8</td><td>64.0 54.2 57.6 40.7</td><td>59.5 4.8</td><td>35.0</td><td>12.2 69.5</td><td>60.8 47.5</td><td>46.7</td></tr><tr><td colspan="10">Open-source Online Models</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VideoLLM-online-8B [2] Dispider-8B [31]</td><td>CVPR&#x27;24 CVPR&#x27;25</td><td>8.1 23.9 57.7 49.5</td><td>12.1</td><td>14.0 45.5</td><td>21.2</td><td>20.8 22.2 48.5</td><td>18.8 12.2 55.4 4.3</td><td>17.7 36.1</td><td>18.1 37.4 48.8</td><td>- 34.7</td><td>41.8</td></tr><tr><td>TimeChatOnline-7B [48]</td><td>MM&#x27;25</td><td>75.2 46.8</td><td>62.1 70.7</td><td>44.9 61.4 47.8 69.3</td><td>51.6 61.4</td><td>54.6 61.9</td><td>55.9 59.5 9.7</td><td>41.7</td><td>31.6 38.5 40.0</td><td>36.7</td><td>46.7</td></tr><tr><td>Streamforest-7B [51]</td><td>NeurIPS&#x27;25</td><td>68.5 53.2</td><td>71.6</td><td>47.8 65.4</td><td>60.9</td><td>61.2</td><td>58.9 64.9 32.3</td><td>52.0</td><td>32.8 70.6 57.1</td><td>52.5</td><td>55.6</td></tr><tr><td>Streamo-7B [43]</td><td>CVPR&#x27;26</td><td>77.2 66.1</td><td>76.7</td><td>45.5 66.3</td><td>72.8</td><td>67.4</td><td>55.6 58.1 33.9</td><td>49.2</td><td>30.8 57.6 82.5</td><td>57.0</td><td>57.9</td></tr><tr><td>VST-7B (ours)</td><td></td><td>80.5 55.1</td><td>72.4</td><td>55.1 76.2</td><td>64.1</td><td>67.2</td><td>56.9 64.9 48.4</td><td>56.7</td><td>33.0 66.9 62.1</td><td>54.0</td><td>59.3</td></tr></table>

Table 3: Comparison of ofline and online VideoLLMs on VideoMME (without subtitles), LongVideoBench, and VideoHolmes.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Venue</td><td colspan="2">VideoMME w/o sub.</td><td rowspan="2">LongVideoBench</td><td rowspan="2">VideoHolmes</td></tr><tr><td>Long</td><td>Overall</td></tr><tr><td colspan="6">Proprietary Models</td></tr><tr><td>Gemini 1.5 pro [37]</td><td></td><td>67.4</td><td>75.0</td><td>64.0</td><td>45.7</td></tr><tr><td>GPT-4o [30]</td><td></td><td>65.3</td><td>71.9</td><td>66.7</td><td>42.0</td></tr><tr><td colspan="6">Open-source Offline Models</td></tr><tr><td>LongVA-7B [55]</td><td>TMLR&#x27;25</td><td>47.6</td><td>54.3</td><td>56.3</td><td></td></tr><tr><td>Video-R1-7B [8]</td><td>NeurIPS&#x27;25</td><td></td><td>61.4</td><td></td><td>36.5</td></tr><tr><td>LongVILA-R1-7B [4]</td><td>NeurIPS&#x27;25</td><td>55.2</td><td>65.1</td><td>58.0</td><td></td></tr><tr><td>REVISOR-7B [21]</td><td>CVPR&#x27;26</td><td>56.2</td><td>65.7</td><td>57.5</td><td></td></tr><tr><td colspan="6">Open-source Online Models</td></tr><tr><td>Dispider-7B [31]</td><td>CVPR&#x27;25</td><td>-</td><td>57.2</td><td>一</td><td>-</td></tr><tr><td>Streamforest-7B [51]</td><td>NeurIPS&#x27;25</td><td>1</td><td>61.4</td><td></td><td></td></tr><tr><td>TimeChatOnline-7B [48]</td><td>MM&#x27;25</td><td>48.4</td><td>62.4</td><td>55.4</td><td></td></tr><tr><td>VST-7B (Ours)</td><td></td><td>55.3</td><td>64.9</td><td>58.0</td><td>41.9</td></tr></table>

## 3.4 Ofline Video Benchmark Results

In Tab. 3, we evaluate VST-7B on three ofline video benchmarks, including VideoMME, LongVideoBench, and VideoHolmes. The results show that VST-7B delivers competitive performance across all three datasets, with particularly strong gains on long-video understanding and complex reasoning. On longvideo benchmarks, VST-7B achieves 55.3% on VideoMME-long, outperforming TimeChat-Online by +6.9%, and 58.0% on LongVideoBench, exceeding it by +2.6%. On the reasoning benchmark VideoHolmes, VST-7B reaches 41.9%, surpassing Video-R1 by +5.4%. We attribute these improvements to our streaming thinking framework, which enables dynamic thinking over long videos to build long-term memory, and leverages both historical memory and current visual context for deep reasoning.

Table 4: Ablation study on VST training schedule.
<table><tr><td rowspan="2">Model &amp; Config</td><td colspan="3">OVO-Bench</td><td rowspan="2">VideoMME</td></tr><tr><td>Backward</td><td>Forward</td><td>Overall</td></tr><tr><td>Qwen2.5-VL-7B (Base model)</td><td>47.5</td><td>41.9</td><td>50.5</td><td>62.9</td></tr><tr><td colspan="5">Ablation on VST-SFT training data</td></tr><tr><td>+LLava-Vid (50K)</td><td>49.9</td><td>42.4</td><td>52.3</td><td>61.8</td></tr><tr><td>+LLava-Vid (30K) &amp; VST (20K)</td><td>52.0</td><td>50.1</td><td>56.8</td><td>62.5</td></tr><tr><td>+LLava-Vid (20K) &amp; VST (30K)</td><td>53.3</td><td>50.0</td><td>57.1</td><td>63.1</td></tr><tr><td colspan="5">Ablation on different training stage</td></tr><tr><td>+VST-SFT</td><td>56.7</td><td>48.5</td><td>57.4</td><td>63.0</td></tr><tr><td>+VST-RL</td><td>49.3</td><td>54.6</td><td>56.8</td><td>62.8</td></tr><tr><td>+VST-SFT &amp; VST-RL</td><td>56.7</td><td>54.0</td><td>59.3</td><td>64.9</td></tr></table>

## 3.5 Ablation Study

Ablation on training schedule. As shown in Tab. 4, we first analyze the composition of the SFT training data. Mixing our VST data with the LLaVA-Vid QA dataset significantly improves online video understanding. Specifically, compared to using 50K LLaVA-Vid data alone, the mix of 20K LLaVA-Vid and 30K VST data achieves a +6.6% gain on the OVO-Bench. Furthermore, the ablation on diferent training stages demonstrates that our training strategies efectively enhance online video capabilities. Interestingly, we find that VST-SFT primarily benefits the model’s backward memory capacity (+9.2% improvement on Backward track), while VST-RL is advantageous for forward prediction capabilities (improving the Forward score of +12.7%). Finally, combining both stages (VST-SFT & VST-RL) yields the highest overall performance on both OVO-Bench (59.3%) and VideoMME (64.9%).

Ablation on Streaming Thinking Times at Inference. Figure 5 analyzes the impact of maximum streaming thinking times on OVO-Bench. For the Backward task, accuracy increases from 53.3% and grows continuously from 1 to 16 steps, ultimately reaching 57.5%. This demonstrates that additional thinking steps help generate precise memories for backward tracing. For the Real-Time and

![](images/47939b2c36734b9a28ac0e72a1d3f2c12dabaa449c4e585f1c44fc1646dd6bd7.jpg)

![](images/e3d59d79d63338f100b4c24e012c77788cfa020819a6915358a43dbc2154e3fa.jpg)  
Fig. 5: Ablation study on max thinking times.

![](images/c62723a34a15603d1121e1e62be7cb392a3f5c7a67424a60e983255e3bb86ac0.jpg)

Table 5: Ablation Study on diferent base ofline VideoLLM’s size.
<table><tr><td>Size Model</td><td></td><td>Overall</td><td>OVOB. StreamingB. Realtime</td><td>V-MME w/o sub. Overall</td><td>LongVideoB. VideoHolmes</td><td></td></tr><tr><td rowspan="2">3B</td><td>Qwen2.5-VL</td><td>53.1</td><td>67.8</td><td>57.9</td><td>53.3</td><td>30.7</td></tr><tr><td>VST</td><td>56.2</td><td>75.5</td><td>59.5</td><td>54.1</td><td>36.1</td></tr><tr><td rowspan="2">7B</td><td>Qwen2.5-VL</td><td>55.0</td><td>71.7</td><td>62.3</td><td>54.7</td><td>32.9</td></tr><tr><td>VST</td><td>59.3</td><td>79.5</td><td>64.9</td><td>58.0</td><td>41.9</td></tr><tr><td rowspan="2">32B</td><td>Qwen2.5-VL</td><td>60.1</td><td>71.5</td><td>65.8</td><td>59.8</td><td>40.1</td></tr><tr><td>VST</td><td>63.5</td><td>80.7</td><td>67.2</td><td>60.7</td><td>45.1</td></tr></table>

Forward tasks, initial thinking steps significantly aid in understanding visual information. However, performance reaches a plateau for ≥ 4 steps, as excessive memory details introduce redundancy.

Ablation on Base Model Size. Table 5 examines the impact of the base model capacity. We apply our two-stage training recipe (VST-SFT and VST-RL) to the Qwen2.5-VL-Instruct models at 3B, 7B, and 32B scales. Evaluated under identical inference configurations, the Video Stream Think paradigm yields consistent improvements across all online and ofline benchmarks regardless of the model size. For instance, on StreamingBench Realtime, VST achieves absolute accuracy gains of +7.7%, +7.8%, and +9.2% over the 3B, 7B, and 32B base models, respectively. Similar consistent enhancements are observed on complex tasks like VideoHolmes (+5.4%, +9.0%, and +5.0%). These results demonstrate that our proposed method is highly parameter-scalable.

## 3.6 Analysis

Eficiency Analysis. We compare the QA latency of several ofline and online methods under the same experimental setup. All measurements are conducted on VideoHolmes, as shown in Tab. 6. Models without CoT directly output the final answer without generating intermediate reasoning. Benefiting from our query-ahead streaming think mechanism, VST maintains significantly lower response latency.

Table 6: Inference Latency.
<table><tr><td>Online Offline Method</td><td>QA Latency</td></tr><tr><td>Qwen2.5-VL-7B Qwen2.5-VL-7B w/CoT Video-R1 w/CoT</td><td>0.54s 5.30s 8.80s</td></tr><tr><td>VideoLLM-online-8B</td><td>0.38s</td></tr><tr><td>Dispider-7B</td><td>1.10s</td></tr><tr><td>VST-3B (Ours)</td><td>0.53s</td></tr><tr><td>VST-7B (Ours)</td><td>0.56s</td></tr><tr><td>VST-32B (Ours)</td><td>1.40s</td></tr></table>

Moreover, streaming think is executed asynchronously before the query and finishes within the clip inter-arrival interval, so its computation is amortized over playback rather than added after the query. As a result, it does not increase the real-world end-to-end inference time.

![](images/19df46057ddbbb359c016f60107f222f2e65916f3063280f7bb11ae9e62d27ab.jpg)  
Fig. 6: Case Study from VideoHolmes. We compare VST-7B with Video-R1-7B. VST-7B processes the video stream and performs streaming thinking before the query, then answers directly once the query arrives. In contrast, Video-R1-7B generates CoT after the query, resulting in higher QA latency. VST-7B achieves better performance with lower QA latency in this example.

Case Study. Fig. 6 presents a case study from the VideoHolmes benchmark. The query requires temporal reasoning over disjoint segments, specifically aligning repeated visualizations of a wall clock with the subsequent appearance of a “blurred-face man”. The baseline Video-R1-7B, which relies on post-query thinking, fails to capture these dispersed temporal cues due to the dificulty of attending to specific evidence across a long context. Consequently, it hallucinates a spurious correlation involving object interactions, leading to a logical error. Furthermore, this retrospective reasoning incurs a significant latency of 9.53s. In contrast, VST-7B employs streaming thinking to continuously update its evidence (e.g., timestamps and event triggers) as the video memories. This pre-query evidence accumulation allows VST to correctly deduce the time-based rule and, by shifting the reasoning burden to the streaming phase, drastically reduces the response latency to 0.51s. This comparison demonstrates that pre-query streaming thinking simultaneously enhances reasoning robustness and system responsiveness.

## 4 Related Work

Streaming Video Understanding. Streaming video understanding processes continuous visual inputs of indeterminate length. Unlike ofline methods, the lack of global sampling and restricted context windows poses significant challenges for

VideoLLMs. Some existing methods attempt to retain extended video information within limited context lengths through real-time visual token compression [2, 32, 35,45,48,51]. Others incorporate external memory mechanisms to recall historical information via query-relevant retrieval [6,31,47,53]. However, these methods rely on static heuristics, lacking autonomous memory management and the ability to perform complex, multi-step reasoning. To bridge this gap, we propose Video Streaming Thinking (VST), which introduces an online thinking process that evolves with the video stream. By coupling autonomous memory management with in-depth instruction analysis, VST enables models to transcend short-range perception and achieve robust streaming intelligence.

VideoLLMs Test-Time Scaling. Following the breakthrough of test-time scaling and chain-of-thought in LLMs [12,38,41], recent VideoLLMs have adopted supervised fine-tuning (SFT) to mimic expert reasoning trajectories [14, 15] or utilized R1-style reinforcement learning (RL) to enhance task performance [4, 8, 21, 40, 46]. Despite these advances, existing post-training research remains predominantly confined to ofline video understanding. The exploration of reasoning within streaming contexts, particularly regarding long-horizon cognitive capabilities, remains a critical yet neglected frontier. In this paper, we introduce a unified SFT and RL framework for streaming video understanding. Our method achieves a synergistic balance between real-time responsiveness and sophisticated reasoning, enabling autonomous memory management and in-depth analysis of evolving video streams.

## 5 Conclusion

In this paper, we propose Video Streaming Thinking (VST), a new paradigm for streaming video understanding that introduces a synchronized stream of logical inference with real-time responsiveness. VST enables a thinking-while-watching mechanism that performs reasoning over incoming clips during streaming. We further develop a post-training recipe (VST-SFT and VST-RL) and an automated data synthesis pipeline based on video knowledge graphs to produce streaming-thought supervision. Empirically, VST not only delivers robust performance across multiple online and ofline video understanding benchmarks but also scales seamlessly to VideoLLMs ranging from 3B to 32B parameters, demonstrating exceptional generalization and broad applicability. Overall, our study establishes VST as a practical test-time scaling approach for streaming scenarios, simultaneously enabling explicit CoT generation and real-time responsiveness.

Limitation and Future Works. While the computation of streaming thoughts can be scheduled in parallel with incoming video clips, the additional LLM token consumption is still non-negligible. A promising direction is to explore latent reasoning to enable more token-eficient streaming thinking. Moreover, VST primarily focuses on text-guided memory management, which is orthogonal to existing streaming visual memory mechanisms. Investigating their combination and potential synergy is an interesting avenue for future work.

## References

1. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 (2025) 2, 8, 9

2. Chen, J., Lv, Z., Wu, S., Lin, K.Q., Song, C., Gao, D., Liu, J.W., Gao, Z., Mao, D., Shou, M.Z.: Videollm-online: Online video large language model for streaming video. In: Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition. pp. 18407–18418 (2024) 9, 10, 14

3. Chen, J., Zeng, Z., Lin, Y., Li, W., Ma, Z., Shou, M.Z.: Livecc: Learning video llm with streaming speech transcription at scale. In: Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition. pp. 29083–29095 (2025) 1

4. Chen, Y., Huang, W., Shi, B., Hu, Q., Ye, H., Zhu, L., Liu, Z., Molchanov, P., Kautz, J., Qi, X., et al.: Scaling rl to long videos. In: Proc. of Advances in Neural Information Processing Systems (2025) 2, 8, 10, 14

5. Cheng, J., Ge, Y., Wang, T., Ge, Y., Liao, J., Shan, Y.: Video-holmes: Can mllm think like holmes for complex video reasoning? arXiv preprint arXiv:2505.21374 (2025) 9

6. Di, S., Yu, Z., Zhang, G., Li, H., Cheng, H., Li, B., He, W., Shu, F., Jiang, H., et al.: Streaming video question-answering with in-context video kv-cache retrieval. In: Proc. of Intl. Conf. on Learning Representations (2025) 2, 14

7. Driess, D., Xia, F., Sajjadi, M.S., Lynch, C., Chowdhery, A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T., et al.: Palm-e: an embodied multimodal language model. In: Proc. of Intl. Conf. on Machine Learning. pp. 8469–8488 (2023) 1

8. Feng, K., Gong, K., Li, B., Guo, Z., Wang, Y., Peng, T., Wu, J., Zhang, X., Wang, B., Yue, X.: Video-r1: Reinforcing video reasoning in mllms. In: Proc. of Advances in Neural Information Processing Systems (2025) 2, 10, 14

9. Feng, K., Zhang, M., Li, H., Fan, K., Chen, S., Jiang, Y., Zheng, D., Sun, P., Zhang, Y., Sun, H., et al.: Onethinker: All-in-one reasoning model for image and video. In: Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition (2026) 8

10. Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., et al.: Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In: Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition. pp. 24108–24118 (2025) 9

11. Guan, Y., Tu, S., Liang, D., Zhu, L., Ju, J., Luo, Z., Luan, J., Liu, Y., Bai, X.: Thinkomni: Lifting textual reasoning to omni-modal scenarios via guidance decoding. In: Proc. of Intl. Conf. on Learning Representations (2026) 2

12. Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., et al.: Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature 645(8081), 633–638 (2025) 2, 6, 14

13. Hagberg, A., Swart, P.J., Schult, D.A.: Exploring network structure, dynamics, and function using networkx. Tech. rep., Los Alamos National Laboratory (LANL) (2007) 7

14. Han, S., Huang, W., Shi, H., Zhuo, L., Su, X., Zhang, S., Zhou, X., Qi, X., Liao, Y., Liu, S.: Videoespresso: A large-scale chain-of-thought dataset for fine-grained video reasoning via core frame selection. In: Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition. pp. 26181–26191 (2025) 14

15. Hannan, T., Islam, M.M., Gu, J., Seidl, T., Bertasius, G.: Revisionllm: Recursive vision-language model for temporal grounding in hour-long videos. In: Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition. pp. 19012–19022 (2025) 14

16. Hasson, U., Nir, Y., Levy, I., Fuhrmann, G., Malach, R.: Intersubject synchronization of cortical activity during natural vision. Science 303(5664), 1634–1640 (2004) 2

17. Hu, H., Dong, S., Zhao, Y., Lian, D., Li, Z., Gao, S.: Transrac: Encoding multi-scale temporal correlation with transformers for repetitive action counting. arXiv preprint arXiv:2204.01018 (2022) 8

18. Hu, S., Tu, Y., Han, X., Cui, G., He, C., Zhao, W., Long, X., Zheng, Z., Fang, Y., Huang, Y., Zhang, X., Thai, Z.L., Wang, C., Yao, Y., Zhao, C., Zhou, J., Cai, J., Zhai, Z., Ding, N., Jia, C., Zeng, G., dahai li, Liu, Z., Sun, M.: MiniCPM: Unveiling the potential of small language models with scalable training strategies. In: Conference on Language Modeling (2024) 9

19. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., Stoica, I.: Eficient memory management for large language model serving with pagedattention. In: Proc. of the ACM SIGOPS 29th Symposium on Operating Systems Principles (2023) 8

20. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., Li, C.: LLaVA-onevision: Easy visual task transfer. Transactions on Machine Learning Research (2025) 9, 10

21. Li, J., Yin, H., Tan, W., Chen, J., Xu, B., Qu, Y., Chen, Y., Ju, J., Luo, Z., Luan, J.: Revisor: Beyond textual reflection, towards multimodal introspective reasoning in long-form video understanding. In: Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition (2026) 10, 14

22. Li, Z., Yang, B., Liu, Q., Ma, Z., Zhang, S., Yang, J., Sun, Y., Liu, Y., Bai, X.: Monkey: Image resolution and text label are important things for large multi-modal models. In: Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition. pp. 26763–26773 (2024) 2

23. Liang, D., Zhang, C., Xu, X., Ju, J., Luo, Z., Bai, X.: Cook and clean together: Teaching embodied agents for parallel task execution. In: Proc. of the AAAI Conf. on Artificial Intelligence (2025) 2

24. Lin, J., Yin, H., Ping, W., Molchanov, P., Shoeybi, M., Han, S.: Vila: On pretraining for visual language models. In: Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition. pp. 26689–26699 (2024) 9

25. Lin, J., Wu, J., Sun, X., Wang, Z., Liu, J., Su, Y., Yu, X., Chen, H., Luo, J., Liu, Z., et al.: Unleashing hour-scale video training for long video-language understanding. arXiv preprint arXiv:2506.05332 (2025) 8

26. Lin, J., Fang, Z., Chen, C., Wan, Z., Luo, F., Li, P., Liu, Y., Sun, M.: Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. arXiv preprint arXiv:2411.03628 (2024) 8

27. Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W.S., Lin, M.: Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783 (2025) 6

28. Ning, Z., Liu, G., Jin, Q., Ding, W., Guo, M., Zhao, J.: Livevlm: Eficient online video understanding via streaming-oriented kv cache and retrieval. arXiv preprint arXiv:2505.15269 (2025) 2

29. Niu, J., Li, Y., Miao, Z., Ge, C., Zhou, Y., He, Q., Dong, X., Duan, H., Ding, S., Qian, R., et al.: Ovo-bench: How far is your video-llms from real-world online video understanding? In: Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition. pp. 18902–18913 (2025) 8

30. OpenAI: Gpt-4o system card (2024) 9, 10

31. Qian, R., Ding, S., Dong, X., Zhang, P., Zang, Y., Cao, Y., Lin, D., Wang, J.: Dispider: Enabling video llms with active real-time interaction via disentangled

perception, decision, and reaction. In: Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition. pp. 24045–24055 (2025) 9, 10, 14

32. Qian, R., Dong, X., Zhang, P., Zang, Y., Ding, S., Lin, D., Wang, J.: Streaming long video understanding with large language models. In: Proc. of Advances in Neural Information Processing Systems. vol. 37, pp. 119336–119360 (2024) 14

33. Shen, X., Xiong, Y., Zhao, C., Wu, L., Chen, J., Zhu, C., Liu, Z., Xiao, F., Varadarajan, B., Bordes, F., et al.: Longvu: Spatiotemporal adaptive compression for long video-language understanding. In: Proc. of Intl. Conf. on Machine Learning (2025) 10

34. Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., Wu, C.: Hybridflow: A flexible and eficient rlhf framework. arXiv preprint arXiv: 2409.19256 (2024) 8

35. Song, E., Chai, W., Wang, G., Zhang, Y., Zhou, H., Wu, F., Chi, H., Guo, X., Ye, T., Zhang, Y., et al.: Moviechat: From dense token to sparse memory for long video understanding. In: Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition. pp. 18221–18232 (2024) 2, 14

36. Stephens, G.J., Silbert, L.J., Hasson, U.: Speaker–listener neural coupling underlies successful communication. Proc. of the National Academy of Sciences 107(32), 14425–14430 (2010) 2

37. Team, G., Georgiev, P., Lei, V.I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., et al.: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530 (2024) 9, 10

38. Tong, J., Fan, Y., Zhao, A., Ma, Y., Shen, X.: Streamingthinker: Large language models can think while reading. arXiv preprint arXiv:2510.17238 (2025) 14

39. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024) 10

40. Wang, Q., Yu, Y., Yuan, Y., Mao, R., Zhou, T.: VideoRFT: Incentivizing video reasoning capability in MLLMs via reinforced fine-tuning. In: Proc. of Advances in Neural Information Processing Systems (2025) 2, 14

41. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. In: Proc. of Advances in Neural Information Processing Systems. vol. 35, pp. 24824–24837 (2022) 14

42. Wu, H., Li, D., Chen, B., Li, J.: Longvideobench: A benchmark for long-context interleaved video-language understanding. In: Proc. of Advances in Neural Information Processing Systems. vol. 37, pp. 28828–28857 (2024) 9

43. Xia, J., Chen, P., Zhang, M., Sun, X., Zhou, K.: Streaming video instruction tuning. arXiv preprint arXiv:2512.21334 (2025) 9, 10

44. Xiaomi, L.C.T.: Mimo-vl technical report (2025), https://arxiv.org/abs/2506. 03569 2

45. Xu, R., Xiao, G., Chen, Y., He, L., Peng, K., Lu, Y., Han, S.: StreamingVLM: Real-time understanding for infinite video streams. In: Proc. of Intl. Conf. on Learning Representations (2026) 14

46. Yan, Z., He, Y., Li, X., Yue, Z., Zeng, X., Wang, Y., Qiao, Y., Wang, L., Wang, Y.: Videochat-r1.5: Visual test-time scaling to reinforce multimodal reasoning by iterative perception. In: Proc. of Advances in Neural Information Processing Systems (2025) 14

47. Yang, Y., Zhao, Z., Shukla, S.N., Singh, A., Mishra, S.K., Zhang, L., Ren, M.: Streammem: Query-agnostic kv cache memory for streaming video understanding. arXiv preprint arXiv:2508.15717 (2025) 2, 14

48. Yao, L., Li, Y., Wei, Y., Li, L., Ren, S., Liu, Y., Ouyang, K., Wang, L., Li, S., Li, S., et al.: Timechat-online: 80% visual tokens are naturally redundant in streaming videos. In: Proc. of ACM Intl. Conf. on Multimedia. pp. 10807–10816 (2025) 2, 9, 10, 14

49. Yu, H., Chen, T., Feng, J., Chen, J., Dai, W., Yu, Q., Zhang, Y.Q., Ma, W.Y., Liu, J., Wang, M., et al.: Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. arXiv preprint arXiv:2507.02259 (2025) 6

50. Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al.: Dapo: An open-source llm reinforcement learning system at scale. In: Proc. of Advances in Neural Information Processing Systems (2025) 6, 7

51. Zeng, X., Qiu, K., Zhang, Q., Li, X., Wang, J., Li, J., Yan, Z., Tian, K., Tian, M., Zhao, X., et al.: Streamforest: Eficient online video understanding with persistent event memory. In: Proc. of Advances in Neural Information Processing Systems (2025) 2, 8, 9, 10, 14

52. Zeng, X., Zhang, Z., Zhu, Y., Li, X., Wang, Z., Ma, C., Zhang, Q., Huang, Z., Ouyang, K., Jiang, T., et al.: Video-o3: Native interleaved clue seeking for long video multi-hop reasoning. arXiv preprint arXiv:2601.23224 (2026) 2

53. Zhang, H., Wang, Y., Tang, Y., Liu, Y., Feng, J., Jin, X.: Flash-vstream: Eficient real-time understanding for long video streams. In: Proc. of IEEE Intl. Conf. on Computer Vision. pp. 21059–21069 (2025) 9, 14

54. Zhang, K., Li, B., Zhang, P., Pu, F., Cahyono, J.A., Hu, K., Liu, S., Zhang, Y., Yang, J., Li, C., Liu, Z.: Lmms-eval: Reality check on the evaluation of large multimodal models (2024), https://arxiv.org/abs/2407.12772 8

55. Zhang, P., Zhang, K., Li, B., Zeng, G., Yang, J., Zhang, Y., Wang, Z., Tan, H., Li, C., Liu, Z.: Long context transfer from language to vision. Transactions on Machine Learning Research (2025) 9, 10

56. Zhang, Y., Wu, J., Li, W., Li, B., MA, Z., Liu, Z., Li, C.: LLaVA-video: Video instruction tuning with synthetic data. Transactions on Machine Learning Research (2025) 8, 10

57. Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C.C., Xu, M., Wright, L., Shojanazeri, H., Ott, M., Shleifer, S., et al.: Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277 (2023) 8

58. Zhu, L., Guan, Y., Liang, D., Ju, J., Luo, Z., Qin, B., Luan, J., Liu, Y., Bai, X.: Shufle-r1: Eficient rl framework for multimodal large language models via data-centric dynamic shufle. In: Proc. of Intl. Conf. on Learning Representations (2026) 2

## Appendix

## A Details of VST Inference

## A.1 Inference Prompt

We detail the inference prompts used forVideo Streaming Thinking (VST). Both the training and inference phases strictly follow this format. As discussed in the Method part, the LLM generates two distinct types of responses: 1) intermittent streaming inference as the video progresses, which is conditioned primarily on past memory and the current video clip (Tab. 7, top); and 2) final answer generation upon receiving the user query, which is conditioned on the accumulated memory, the current video clip, and the specific question (Tab. 7, bottom).

Table 7: Template of VST inference for streaming thinking (top part) and final answer generation (bottom). Curly-brace placeholders {} will be replaced with actual content. {Memory} denotes the historical outputs from previous streaming thinking, comprising the corresponding timestamps and generated content, while {VideoClip} represents the incoming video stream.  
```ini
[System]
You are a Streaming Video Analyst.
{Memory}
{TimeStamp} {VideoClip}
[System]
You are a Streaming Video Analyst.
{Memory}
{TimeStamp} {VideoClip}
{QueryTime} Based on the provided Video Memory and the Current Video
Clip, answer the following Problem.
{Problem}
Output the final answer in \boxed{}
Your answer:
```

## A.2 Streaming Inference Pipeline

Fig. 7 illustrates the streaming reasoning pipeline of VST. Prior to receiving the user query, we conduct a streaming thinking process for each video clip, ensuring the output is generated before the subsequent clip arrives. Consequently, our method efectively utilizes the natural waiting time inherent in real-world video streams. This enables a rapid response once the user poses a question, where the QA latency is defined as the time elapsed from the user’s query submission to the LLM’s response.

![](images/fd22e55679939455aa35996dc302cf95f4ec1651398efb4e08840bbf4318a44d.jpg)  
Fig. 7: The streaming inference pipeline of VST. By generating stream thoughts for incoming video clips before a user query arrives, VST efectively hides reasoning latency and enables rapid QA responses.

## B Details of VST-SFT / RL Training

VST-SFT Hyperparameters. To train a streaming model capable of continuous thinking and ensure reproducibility, we report the hyperparameters for the VST-SFT phase. We sample up to 384 frames per video and limit the maximum video pixels to 19,267,584, capping video tokens at 24K to reserve 8K context for language and reasoning. The model is trained for 1 epoch with a learning rate of 5e-6 and 8 gradient accumulation steps.

VST-RL Hyperparameters. For the reinforcement learning phase, we employ the DAPO algorithm to optimize the model. For training eficiency, the maximum prompt length is limited to 11,000 tokens, with 1,000 reserved for the maximum generated response length. During the rollout phase, we generate 8 candidate responses per prompt using vLLM with a temperature of 1.0 and a top-p of 0.98. The model is trained for 1 epoch with a global training batch size of 256 and a PPO mini-batch size of 64. We use a learning rate of 5e-7 with 20 warmup steps for the actor model. To ensure stable optimization and manage memory eficiently, we freeze the vision tower, set the KL penalty coeficient to 0.001, and leverage Fully Sharded Data Parallel (FSDP) with both parameter and optimizer ofloading.

## C Details of VST Data Generation

Table 8 presents the prompt templates utilized for our data generation pipeline. The entire data synthesis process strictly adheres to these formats to ensure high-quality, consistent annotations. As discussed in the Data Generation section, the LLM generates three distinct types of outputs: 1) video knowledge graph construction, which is conditioned on the current video segment and known entities to map dense physical relationships and object states (Tab. 8, top); 2) intermediate chain-of-thought (CoT) generation, which is conditioned on the specific time range and focused entities to provide incremental updates of the visual progress (Tab. 8, middle); and 3) multi-hop QA generation, which is conditioned on the constructed event reasoning path and visual evidence to synthesize practical reasoning question-answer pairs (Tab. 8, bottom).

Table 8: Templates of prompts used for data generation. The table presents three distinct prompts: Video Knowledge Graph Generation (top), Intermediate Chain-of-Thought (CoT) Generation (middle), and Question-Answer (QA) Generation (bottom). Curly-brace placeholders {...} will be replaced with actual segment-specific content during generation.

You are a Visual Scene Analyst specializing in dense scene graph generation. Your   
goal is to map ALL physical relationships in the video segment, not just human   
actions.   
[CURRENT TIMELINE] Segment {#step<sup>\_</sup>index}: {start<sup>\_</sup>time}s to {end<sup>\_</sup>time}s.   
[CONTEXTUAL DATA] 1. Entity Registry: {known<sup>\_</sup>entities<sup>\_</sup>str}   
[CRITICAL VISUAL EXTRACTION RULES] 1. NO "HUB-AND-SPOKE" BIAS (CRITICAL): Do not make   
the human protagonist the subject of every relation. Extract edges where neither the   
subject nor the object is a person. 2. Object-to-Object Relationships (MANDATORY):   
Look for Spatial Relations, Containment, and Passive Interactions. 3. Visual Entity   
Identification: Identify objects using specific visual descriptors. NO PRONOUNS. 4.   
Action & State Verbs: Use active verbs for humans and spatial/state verbs for objects.   
5. Description: Describe the scene layout and object states, not just the human’s   
movement.   
[OUTPUT FORMAT] Return ONLY a JSON object: {"events": [{"subject": "...", "relation":   
.", "object": "...", "description": "..."}]}   
Current Video Segment: {current<sup>\_</sup>time<sup>\_</sup>range}   
Task: Provide an incremental update of the visual scene/action/details in this   
segment. Focus on Progress (Delta) of visual elements.   
Context & Entities: Focused Entities: {entity<sup>\_</sup>text} (Integrate these naturally if   
they are actively involved in the current action. No full repetition each time;   
pronouns prohibited.)   
Strict Constraints: 1. Focus on Dynamics: Describe WHAT is happening now with details.   
If the scene is static, be very concise. 2. NO restate: Do NOT restate the   
action/scene/details from the previous segment’s end. Skip the action stated   
previously or state the changes. NEVER repeat. 3. Minimalist Reference: For entities   
already present/mentioned, use pronouns or minimal descriptors. 4. Language Variety:   
Avoid repetitive sentence structures. 5. No Redundancy: Do not repeat the timestamp   
or information from the previous context.   
Output:   
You are a Cognitive Video Intelligence Engine. Your task is to synthesize a Practical   
Deep Reasoning Question-Answer Pair based on the provided visual information from   
video segments and the Event Reasoning Path.   
=== CORE DEFINITION (MUST FOLLOW FIRST) === 1. Event Reasoning Path Nature: The   
provided video segments are ordered by logical reasoning relevance, NOT by their   
original chronological time sequence. 2. Time Reference Rule: All time mentions must   
be based on the explicit time interval of each node. 3. Multi-hop Foundation: The   
reasoning must rely on logical connections between segments.   
=== MANDATORY INSTRUCTIONS === 1. Strict Multi-hop Reasoning: The question MUST   
require integrating information from multiple video segments in the reasoning path. 2.   
Natural Language Constraint (CRITICAL): DO NOT use the words "Step", "Clip index",   
"Path node". Refer to segments using their time intervals or event descriptions. 3.   
Reasoning Dimensions: Construct the reasoning chain using one or more of the   
following logic types: {dimensions<sup>\_</sup>str} 4. Practicality Requirement: The question   
must be meaningful for understanding the visual content of the video’s narrative,   
intent, or physical logic.   
=== OUTPUT FORMAT (JSON ONLY, NO EXTRA TEXT) === {"question": "...", "cot": "...",   
"answer": "...", "reasoning<sup>\_</sup>type": "..."}