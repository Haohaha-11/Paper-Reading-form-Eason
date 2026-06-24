# Improving Vision-language Models with Perception-centric Process Reward Models

Yingqian Min<sup>1,2\*</sup>, Kun Zhou<sup>3\*</sup>, Yifan Li<sup>1,2\*</sup>, Yuhuan Wu<sup>4</sup>, Han Peng<sup>1</sup> Yifan Du<sup>1</sup>, Wayne Xin Zhao<sup>1†</sup>, Min Yang<sup>2</sup>, Ji-Rong Wen<sup>1</sup>

<sup>1</sup>Gaoling School of Artificial Intelligence, Renmin University of China.

<sup>2</sup>Bytedance. <sup>3</sup>University of California, San Diego.

<sup>4</sup>The Hong Kong University of Science and Technology. yingqianm@ruc.edu.cn, batmanfly@gmail.com

## Abstract

Recent advancements in reinforcement learning with verifiable rewards (RLVR) have significantly improved the complex reasoning ability of vision-language models (VLMs). However, its outcome-level supervision is too coarse to diagnose and correct errors within the reasoning chain. To this end, we propose Perceval, a process reward model (PRM) that enables token-level error grounding, which can extract image-related claims from the response and compare them one by one with the visual evidence in the image, ultimately returning claims that contain perceptual errors. Perceval is trained with perception-intensive supervised training data. We then integrate Perceval into the RL training process to train the policy models. Specifically, compared to traditional GRPO, which applies sequencelevel advantages, we apply token-level advantages by targeting penalties on hallucinated spans identified by Perceval, thus enabling fine-grained supervision signals. In addition to augmenting the training process, Perceval can also assist VLMs during the inference stage. Using Perceval, we can truncate the erroneous portions of the model’s response, and then either have the model regenerate the response directly or induce the model to reflect on its previous output. This process can be repeated multiple times to achieve test-time scaling. Experiments show significant improvements on benchmarks from various domains across multiple reasoning VLMs trained with RL, highlighting the promise of perception-centric supervision as a generalpurpose strategy. For test-time scaling, it also demonstrates consistent performance gains over other strategies, such as major voting. Our code and data will be publicly released at https://github.com/RUCAIBox/Perceval.

## 1. Introduction

Vision–language models (VLMs) [3, 7, 12] deliver strong results across tasks such as multimodal mathematics [26, 38], chart analysis [24, 27], and general VQA [50]. However, they still falter on complex visual reasoning tasks, where multi-step chains of thought can be brittle and produce perceptual or logical mistakes [6, 11, 42]. To improve the performance, reinforcement learning with verifiable rewards (RLVR) [13, 33, 35] has become a widely used post-training strategy. Built on policy-gradient methods like PPO and GRPO, RLVR assigns outcome-level rewards to explicit reasoning traces and optimizes the policy toward more consistent, robust multi-step visual reasoning.

Despite these advances, outcome-level supervision in RLVR is poorly matched to the inherently multi-step nature of visual reasoning. In fact, sequence-level rewards are too coarse to identify which perception or reasoning steps went wrong, creating a hard credit-assignment problem. In practice, VLMs often insert hallucinated objects or spatial relations and drift from the image context midchain [1, 19, 20, 22, 53], but only the final reward offers little guidance about whether the failure arose from visual grounding or subsequent logic. Thus, the sparse-reward regime ultimately bottlenecks RLVR’s gains on VLMs [48].

To overcome the sparse-reward limitation, we introduce a process reward model (PRM) that supervises intermediate steps rather than only the final outcome [39]. Prior work shows that PRMs can effectively guide both training and inference by rewarding stepwise, chain-of-thought correctness [21, 55]. However, building a high-quality PRM is difficult because step-level annotations are expensive and some steps are only verifiable after later derivations, complicating labeling and consistency [17, 54]. Fortunately, in visual reasoning many intermediate steps are perceptual claims (e.g., objects, attributes, or spatial relations) that can be grounded directly in the image, enabling automatic checks for “image–text misalignment” (hallucination). Therefore, it is promising to develop a perceptioncentric PRM that detects and explains such misalignments to provide fine-grained feedback, alleviating sparse-reward issue and improving learning of the reasoning ability.

To operationalize this, we first define a perceptionlevel error-finding schema for a perception-centric PRM. We curate training queries from perception-intensive settings—such as goal-directed visual search and referringexpression grounding—and use a strong LLM to produce structured annotations that mark image–text misalignments (hallucinatory spans and their visual counter-evidence). After supervised fine-tuning on this corpus, the PRM can reliably flag hallucinations that arise within multi-step rationales and return well-structured feedback. Building on this, we integrate the PRM into RLVR by decomposing the sequence-level advantage and assigning fine-grained, token-level penalties to spans identified as hallucinatory, yielding more precise credit assignment than GRPO alone. Finally, based on PRM’s structured outputs, we employ a simple Truncation–Regeneration loop at inference. In this way, suspect spans are pruned and regenerated, trading a bit more compute for stronger factual grounding.

Experimental results demonstrate that, compared to direct GRPO, our training method significantly enhances the model’s perceptual capabilities, boosting performance on perception-centric tasks. Furthermore, we observe a surprising and significant generalization effect: even without applying PRM supervision during the training for complex reasoning tasks, this foundational improvement in perception nonetheless generalizes, leading to a comprehensive enhancement of the model’s overall reasoning abilities.

Our main contributions are as follows:

• We propose a novel, perception-centric process reward model (PRM) that can explicitly identify perception errors in the reasoning process.

• We introduce a fine-grained, token-level advantage reallocation framework that integrates our PRM with GRPO, to solve the sparse reward issue.

• We design a test-time iterative refinement strategy that leverages our PRM to actively detect and correct perceptual errors from the policy model.

## 2. Preliminary

We introduce foundational concepts and notations used throughout this paper: the architecture of vision-language models (VLMs), the reinforcement-learning framework with verifiable rewards (RLVR) which our method builds on, and our problem statement for designing a perceptioncentric process reward model.

Vision-Language Models. A vision-language model (VLM) accepts multimodal input, typically an image v and a text query q, and generates the text output o, denoted as $\pi _ { \boldsymbol { \theta } } ( o | q , \boldsymbol { v } )$ . For reasoning tasks, the text output is generally a chain of language reasoning steps. Typical architecture combines a visual encoder (e.g. ViT) to embed I and a large language model (LLM) to decode the output. Typically, the two modalities are linked via a connection layer.

Reinforcement Learning with Verifiable Rewards. RL with verifiable rewards (RLVR) has become the key technique to improve the performance of VLMs in reasoning tasks [45]. It aims to train the VLM to not only generate plausible outputs but also satisfy measurable criteria $( e . g .$ correctness, spatial consistency). One algorithm is Group Relative Policy Optimization (GRPO) [33]: given the input prompt q and image v, a reference policy $\pi _ { \boldsymbol { \theta } } ( o | q , \boldsymbol { v } )$ samples multiple responses $\left\{ o _ { i } \right\}$ . Each response will be assigned with a scalar reward $R _ { i }$ from the verified function or reward model. The advantage of the i-th response is calculated by normalizing its reward relative to the group:

$$
\hat { A } _ { i } = \frac { R _ { i } - \mathrm { m e a n } ( \{ R _ { j } \} _ { j = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R _ { j } \} _ { j = 1 } ^ { G } ) }\tag{1}
$$

Note that this advantage ${ \hat { A } } _ { i }$ is a sequence-level signal, which is constant for all tokens within the i-th response. Hence, GRPO optimizes a clipped surrogate objective to update the policy π<sub>θ</sub> based on the advantage:

$$
\begin{array} { r l } & { J ( \theta ) = \mathbb { E } _ { ( q , \{ o _ { i } \} ) \sim \pi _ { \theta } } \Bigg [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \sum _ { t = 1 } ^ { | o _ { i } | } \operatorname* { m i n } \Big ( r _ { i , t } ( \theta ) \hat { A } _ { i } , } \\ & { ~ \mathrm { c l i p } ( r _ { i , t } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { i , t } ^ { \prime } \Big ) - \beta D _ { K L } ( \pi _ { \theta } | | \pi _ { r e f } ) \Bigg ] } \end{array}\tag{2}
$$

where ϵ is the clipping hyperparameter and $r _ { i , t } ( \theta )$ is the importance sampling ratio for token t.

Problem Statement. A key limitation of reinforcement learning with verifiable rewards (RLVR) is reward sparsity: conventional approaches provide a single scalar reward only at the end of the reasoning chain, so each token or step is credited equally regardless of its individual correctness or contribution. This coarse, sequence-level feedback makes it difficult to correct localized errors in perception or reasoning and undermines the model’s ability to generalize robustly. To overcome this, we propose training a perceptioncentric process reward model (PRM) that evaluates intermediate perceptual outputs and produces step-wise feedback. Concretly, the PRM checks whether the model’s perception content in response (e.g., a grounding, visual feature, or intermediate state) is correct relative to the input $v , q ,$ and generate structured outputs that can be used to provide fine-grained supervision. During inference, the PRM can be used to guide the selection of intermediate steps. During training, by designing proper learning objective with the PRM, we encourage correct intermediate perceptual reasoning, enabling more fine-grained supervision for effective learning.

![](images/e6d34e680e78e5b8eabee16f33367a14094c14205407fb8e5171d815719081a4.jpg)  
Figure 1. An overview of our Process-Supervised GRPO framework. For each generated response, we use the Perceval to create a token level penalty mask. This mask is used to calculate a fine-grained token-level advantage, which is then incorporated into the GRPO objective to penalize hallucinatory tokens and improve the model’s perceptual grounding.

## 3. Methodology

In this section, we devise our perception-centric process reward model for providing fine-grained, process-level supervision to guide VLMs. We first introduce the design and how to train the PRM, and then present how to integrate it with RLVR during training and how to perform test-time scaling with PRM guidance.

## 3.1. Perception-Centric Process Reward Model

To overcome the sparse supervision issue, we propose PERCEVAL (Perception-centric process reward evaluation model), which serves as an external, fine-grained, and interpretable critic for guiding VLM policy.

Error-finding Schema Design. Given a tuple of image, text query, and model’s response ⟨v, q, o⟩, PERCEVAL generates a structured verification V to assess the factual consistency with respect to v (conditioned on q). To improve reliability, PERCEVAL follows the well-known think-thenanswer paradigm [13]: it first analyzes each claim and outputs the thought process within <think>...</think>, where each statement in o is evaluated for consistency with the visual evidence in v. Based on these analyses, PERCEVAL provides the final decision wrapped in <answer>...</answer>. If no perceptual errors are found, the final answer is simply "The response is correct."; otherwise, the answer is formatted as a Python list containing the exact strings from o that are identified as errors.

Process Reward Model Training. We train PERCEVAL using a dataset constructed via a four-stage pipeline:

• Query selection: to emphasize perceptual grounding, we primarily source the images and queries from visual search datasets [42, 56] that require locating specific objects in large images, and we include a small proportion from other domains (e.g., mathematical reasoning and general understanding [10]) to preserve breadth;

• Rollout generation: based on the images and queries, we use an open-source VLM (e.g., Qwen2.5-VL-7B) to produce responses, whose imperfect perceptual alignment yields realistic hallucinations as negative examples;

• Automated annotation and verification: for each response, we adopt a strong models (e.g., Gemini-2.5-Pro) to perform hallucination-focused, step-by-step checks. The generated annotations follow our designed format.

• Supervised fine-tuning: we fine-tune the PERCEVAL backbone with a standard SFT objective on the aggregated data to emulate detailed, perception-centric verification and produce the prescribed structured output.

## 3.2. RLVR with Process-level Supervision

Building on PERCEVAL, we revise the GRPO objective to support process-level supervision by replacing the coarse sequence-level advantage $\hat { A } _ { i }$ (Eq. 1) with a token-level advantage $\hat { A } _ { i , t } ^ { \prime } .$ The key change is to let advantage computation accept per-token signals so that perceptual errors within a response are directly penalized during learning. To achieve it, for each response, we use PERCEVAL to identify the token spans that realize perception-induced hallucinations, and then re-assign advantages for those tokens to provide a reduced (or more negative) learning signal.

Given a response $o _ { i }$ of length $L _ { i }$ and the PERCEVAL verification, we parse the <answer> content and select the identified problematic substrings. We locate each substring in $o _ { i }$ via exact string match to obtain its token span $[ j _ { k } , l _ { k } ]$ and define $\begin{array} { r } { U _ { i } = \bigcup _ { k = 1 } ^ { K } [ j _ { k } , l _ { k } ] } \end{array}$ . From $U _ { i }$ we construct a binary mask $M _ { i } = [ m _ { i , 1 } , \dots , m _ { i , L _ { i } } ]$ with $m _ { i , t } = 1 { \mathrm { i f } } t \in U _ { i }$ and 0 otherwise. Then, we modulate the sequence-level signal with this mask to form the token-level advantage:

$$
\hat { A } _ { i , t } ^ { \prime } : = \hat { A } _ { i } - \alpha \cdot m _ { i , t } \cdot | \hat { A } _ { i } | ,\tag{3}
$$

where $\alpha ~ \in ~ [ 0 , 1 ]$ controls penalty strength. Thus, correct tokens $( m _ { i , t } = 0 )$ keep $\hat { A } _ { i , t } ^ { \prime } = \hat { A } _ { i }$ , while hallucination tokens $( m _ { i , t } = 1 )$ are downweighted: when $\hat { A } _ { i } > 0 .$ $\hat { A } _ { i , t } ^ { \prime } = \hat { A } _ { i } ( 1 - \alpha )$ ; when $\hat { A } _ { i } < 0 , \hat { A } _ { i , t } ^ { \prime } = \hat { A } _ { i } ( 1 + \alpha )$ , making the penalty stronger. Finally, we substitute $\hat { A } _ { i , t } ^ { \prime }$ into the GRPO objective in Eq. 2 to add the process supervision. Such a way injects direct, token-level corrective pressure into GRPO, which preserve sequence-level preferences while explicitly suppressing ungrounded content.

## 3.3. Test-Time Scaling with PRM Guidance

Beyond training-time use, PERCEVAL (our perceptioncentric PRM) enables test-time scaling by supplying targeted error-correction during inference. We introduce two pragmatic refinement loops:

Truncate–then–Regenerate. When PERCEVAL detects an erroneous claim, it returns the offending span in the model’s rationale. We truncate the hypothesis before the first token of that span, preserving only the verified prefix as context. The policy model then continues to regenerate the answer following this cleaned prefix. As the original image and question are given, the VLM just needs to resample the detected hallucinated part, without rewriting verified content. This truncate–continue cycle repeats until no new errors are flagged or a maximum of k iterations is reached. The iteration cap k bounds latency while typically yielding large accuracy gains with only a few refinement steps.

Truncate–Thinking–then–Regenerate. To further encourage self-correction, we augment the above method with a lightweight guidance for thinking. After truncating at the error, we append a brief thinking prompt in PERCE-VAL’s output, e.g., “Wait, I need to reconsider this reasoning more carefully: the mug is not on the brick in the image.”, which guides the model to think and then regenerate from the augmented context. The added thinking process enables self-reflection on the failure mode (object/attribute/spatial mismatch), improving the likelihood that the continuation repairs the specific misalignment. As with Truncate–then–Regenerate, we iterate up to k times or stop early when no further errors are found, trading modest extra compute for stronger factual grounding.

## 4. Experiment

## 4.1. Experimental Setup

Benchmarks. We select multiple visual reasoning benchmarks, covering visual search, perception-intensive reasoning, mathematical and chart-based reasoning.

1. V\* (V-Star) [42]: introduces an LLM-guided visual search mechanism and a dedicated benchmark, to assess models’ ability to localize and reaso about small, target objects within information-dense images. It contains 191 high-resolution images with two subtasks, i.e. attribute recognition and spatial-relation reasoning that require precise grounding before reasoning.

2. MME-RealWorld [50]: targets practical applications across five domains (OCR-in-the-wild, remote sensing, diagrams/tables, monitoring, autonomous driving). We use its subset MME-RealWorld-Lite for testing.

3. BLINK [11]: reframes 14 classic computer-vision tasks (e.g. relative depth, visual correspondence, image forensics, multi-view reasoning) into 3,807 multiple-choice items to probe foundational perceptual skills that resist purely linguistic mediation.

4. MMStar [6]: compiles 1,500 carefully selected, humancurated samples to probe six core capability areas along 18 fine-grained axes, focusing on cases where vision is indispensable (rather than solvable by text priors).

5. RealWorldQA [43]: contains 700 images captured from vehicles and other real-world settings, each paired with a question and an easily verifiable answer.

6. MathVista [26]: aggregates 6,141 examples from 28 existing multimodal sources and three new sets (IQTest, FunctionQA, PaperQA) to test numeracy, geometry/- diagram understanding, tables/plots, and compositional visual-math reasoning.

7. MATH-Vision [38]: offers 3,040 problems sourced from real competitions, spanning 16 mathematical disciplines and five difficulty levels, each embedded in a visual context (figures, diagrams, plots).

8. ChartQA[27]: contains 9.6K human-written and 23.1K generated questions over diverse chart types, requiring both visual parsing and table/logic operations.

Baselines. We compare our methods with multiple reasoning-oriented VLMs.

1. VLM-R1 [34]: extends R1-style RLVR to VLMs by leveraging tasks with deterministic visual ground truth.

2. LMM-R1 [30]:leverages text-only data with rule-based RL and multimodal generalization training to transfer gains to vision reasoning task.

3. R1-VL [47]: proposes StepGRPO, replacing sequencelevel rewards with dense step-wise rule-based rewards to stabilize visual reasoning ability learning.

4. Perception-R1 [45]: targets perception-heavy tasks and utilizes GRPO with perception-oriented rewards.

5. Jigsaw-R1 [41]: is first trained on jigsaw puzzles data to improve generation, and then visual reasoning datasets.

6. DeepEyes [56]: is end-to-end trained with RL to think with images and interleaves the visual grounding step inside the whole reasoning process.

7. PixelReasoner [37]: adopts pixel-space reasoning (e.g. zoom and crop) and a two-phase training: fine-tuning on synthesized data, then curiosity-driven RL.

8. Vision-R1 [14]: cold-starts via a synthetic dataset, then applies GRPO with a hard formatting reward and a progressive thinking suppression training strategy.

9. VL-Rethinker [36]: uses GRPO with selective sample replay to mitigate vanishing advantages and adds forced rethinking triggers to elicit reflection.

10. VLAA-Thinker [5]: is trained using mixed verifiable rewards on multimodal CoT dataset with GRPO.

11. OpenVLThinker [8]: iterate fine-tuning on distilled data and RL for improvement until convergence.

12. MM-Eureka [28]: scales up the training data for rulebased RL in multimodal settings.

Implementation Details. We select Qwen2.5-VL as the backbone for both reward and policy models. We first train two versions of PERCEVAL of 3B and 7B sizes, following the procedures outlined in section 3.1, and then correspondingly train two policy models of the same sizes using the proposed method. As for the training data, the supervised fine-tuning data are collected from DeepEyes [56] and SophiaVL-R1 [10], each of which is rolled out 3 times using the backbone models. The RL training data is also derived from [56], with the primary objective of enhancing the model’s perception capabilities, while also containing a subset of general-purpose reasoning data. Consequently, during the RL training phase, we implement a conditional strategy: PERCEVAL is used only on perception-related data to perform fine-grained advantage rescaling. For all other training data (e.g., mathematical reasoning), no additiona intervention is applied, and we revert to using direct GRPO. This experimental design allows us to investigate whether fine-grained supervision focused on perception tasks can generalize and yield performance gains in other domains.

Evaluation Setup. To ensure fair and reproducible evaluation, we establish a unified evaluation pipeline. We employ greedy decoding for all models and utilize the same prompt template to collect responses. We then extract the final answer following the official procedures of each benchmark. Finally, the accuracy is determined through a twostage judging process: we first apply an exact match (EM) judge for each extracted answer against the ground truth. For any answer that does not match, a robust judge mode (i.e. GPT-4o-mini) is utilized for a final verification to account for minor formatting variations. Additionally, we report the relaxed accuracy for ChartQA [27], aligned with the official evaluation of the benchmark, which uses the methodology of PlotQA [29].

## 4.2. Main Results

RL Training with PRM. As shown in Table 1, our method significantly and consistently outperforms the GRPO baseline across both 3B and 7B model scales. Specifically, for the 3B model, our approach achieves average improvements of approximately 4% in the Visual Search category, 3% in Math and Chart reasoning, and 1% in Perception-intensive Reasoning relative to the GRPO baseline. This result strongly demonstrates that our method provides richer and more fine-grained supervision. A deeper analysis of the Visual Search sub-tasks reveals that (Positional Perception), particularly at the 3B scale (e.g., improving from 86.95 to 90.43). This strongly suggests that our finegrained process supervision has successfully guided the model to enhance its precise spatial localization capabilities. Concurrently, the improvements on benchmarks like BLINK and MMStar also indicate that this enhanced perception leads to higher fidelity and fewer hallucinations. A crucial finding is the model’s strong generalization ability. As discussed in Section 4.1, although our PRM training and RL intervention were predominantly focused on Visual Search tasks, the model still exhibits consistent performance gains across all other domains, including general perception and math reasoning. We attribute this “capability transfer” to the fact that tasks in Math & Chart (such as MathVision and ChartQA) are fundamentally reliant on precise, fine-grained perceptual abilities (e.g., localizing data points on a chart, reading text). By strengthening the model’s foundational perceptual accuracy, our method successfully generalizes this improvement to broader and more complex reasoning tasks. Furthermore, our 7B model trained with our method also surpasses Pixel-Reasoner and achieves performance competitive with DeepEyes on Visual

Table 1. Main results on multimodal benchmarks regarding visual search, perception-intensive reasoning and math&chart tasks. MRW and RWQA denote MME-RealWorld and RealWorldQA, respectively. Best and second best results in each group are highlighted in bold and underlined, respectively. <sup>∗</sup> indicates models capable of calling tools.
<table><tr><td rowspan="2">Models</td><td rowspan="2">#Param</td><td colspan="3">Visual Search</td><td colspan="4">Perception-intensive Reasoning</td><td colspan="3">Math &amp; Chart</td></tr><tr><td> $\mathbf { V _ { a t t r } ^ { * } }$ </td><td> $\mathbf { V _ { p o s } ^ { * } }$ </td><td>Vl1</td><td>BLINK</td><td>MMStar</td><td>MRW</td><td>RWQA</td><td>MathVision</td><td>MathVista</td><td>ChartQA</td></tr><tr><td>VLM-R1 [34]</td><td>3B</td><td>75.65</td><td>67.11</td><td>72.25</td><td>46.25</td><td>56.7</td><td>42.3</td><td>61.5</td><td>21.71</td><td>65.1</td><td>83.48</td></tr><tr><td>LMM-R1 [31]</td><td>3B</td><td>46.09</td><td>53.95</td><td>49.21</td><td>46.60</td><td>56.7</td><td>35.8</td><td>58.7</td><td>24.47</td><td>63.5</td><td>85.04</td></tr><tr><td>R1-VL [47]</td><td>2B</td><td>59.13</td><td>57.89</td><td>58.64</td><td>42.81</td><td>38.9</td><td>32.4</td><td>49.2</td><td>10.20</td><td>47.0</td><td>68.08</td></tr><tr><td>Perception-R1 [45]</td><td>3B</td><td>57.39</td><td>48.68</td><td>53.92</td><td>46.44</td><td>54.8</td><td>37.5</td><td>55.8</td><td>22.03</td><td>58.1</td><td>81.60</td></tr><tr><td>Jigsaw-R1 [41]</td><td>3B</td><td>72.17</td><td>65.79</td><td>69.63</td><td>45.01</td><td>54.4</td><td>42.2</td><td>57.9</td><td>19.40</td><td>61.0</td><td>84.60</td></tr><tr><td>Qwen2.5-VL [4]</td><td>3B</td><td>57.39</td><td>65.79</td><td>60.73</td><td>46.94</td><td>52.1</td><td>41.7</td><td>63.4</td><td>21.40</td><td>61.6</td><td>83.12</td></tr><tr><td>+ GRPO</td><td>3B</td><td>86.95</td><td>69.73</td><td>80.10</td><td>49.13</td><td>55.3</td><td>46.8</td><td>62.1</td><td>23.36</td><td>65.1</td><td>83.32</td></tr><tr><td>+ Ours</td><td>3B</td><td>90.43</td><td>72.37</td><td>83.25</td><td>48.75</td><td>55.8</td><td>47.6</td><td>64.9</td><td>26.32</td><td>65.6</td><td>86.48</td></tr><tr><td>DeepEyes [56]*</td><td>7B</td><td>91.30</td><td>81.58</td><td>87.43</td><td>50.98</td><td>62.7</td><td>46.5</td><td>67.0</td><td>12.50</td><td>69.9</td><td>75.84</td></tr><tr><td>Pixel-Reasoner [31]*</td><td>7B</td><td></td><td></td><td>84.30</td><td>51.10</td><td>63.1</td><td>43.5</td><td>64.0</td><td>22.03</td><td>69.4</td><td>77.36</td></tr><tr><td>Vision-R1 [14]</td><td>7B</td><td></td><td></td><td></td><td>49.72</td><td>55.4</td><td>49.6</td><td>64.1</td><td>36.18</td><td>71.3</td><td>83.36</td></tr><tr><td>VL-Rethinker [36]</td><td>7B</td><td>54.78</td><td>59.21</td><td>56.54</td><td>49.91</td><td>64.0</td><td>38.8</td><td>64.0</td><td>31.91</td><td>72.6</td><td>85.60</td></tr><tr><td>VLAA-Thinker [5]</td><td>7B</td><td>43.47</td><td>52.63</td><td>47.12</td><td>49.38</td><td>64.0</td><td>48.0</td><td>62.0</td><td>27.96</td><td>70.3</td><td>85.36</td></tr><tr><td>R1-VL [47]</td><td>7B</td><td>47.83</td><td>67.11</td><td>55.50</td><td>47.19</td><td>55.5</td><td>40.5</td><td>59.5</td><td>22.37</td><td>64.1</td><td>82.80</td></tr><tr><td>OpenVLThinker [8]</td><td>7B</td><td>76.52</td><td>80.26</td><td>78.01</td><td>51.36</td><td>62.8</td><td>59.1</td><td>66.5</td><td>32.57</td><td>71.1</td><td>89.00</td></tr><tr><td>MM-Eureka [28]</td><td>7B</td><td>42.61</td><td>56.58</td><td>48.17</td><td>50.23</td><td>63.4</td><td>46.4</td><td>62.3</td><td>32.23</td><td>72.4</td><td>82.36</td></tr><tr><td>Qwen2.5-VL [4]</td><td>7B</td><td>60.87</td><td>64.47</td><td>62.30</td><td>48.56</td><td>62.3</td><td>43.0</td><td>60.6</td><td>26.97</td><td>70.2</td><td>84.28</td></tr><tr><td>+ GRPO</td><td>7B</td><td>85.22</td><td>82.89</td><td>84.29</td><td>53.55</td><td>62.0</td><td>49.5</td><td>66.4</td><td>27.96</td><td>71.7</td><td>85.16</td></tr><tr><td>+ Ours</td><td>7B</td><td>86.09</td><td>86.84</td><td>86.39</td><td>54.49</td><td>63.8</td><td>50.0</td><td>67.4</td><td>30.92</td><td>72.0</td><td>84.44</td></tr></table>

Table 2. Comparison of different test-time scaling strategies, where Truncate and Truncate-Thinking denote our proposed Truncate–then–Regenerate and Truncate–Thinking–then–Regenerate methods, respectively.
<table><tr><td rowspan="2">Sample</td><td rowspan="2">Method</td><td colspan="3">V*</td><td rowspan="2">BLINK</td></tr><tr><td>Attr</td><td>Pos</td><td>All</td></tr><tr><td rowspan="3">k=4</td><td>Major voting</td><td>91.30</td><td>76.32</td><td>85.34</td><td>48.24</td></tr><tr><td>Truncate</td><td>93.04</td><td>77.63</td><td>87.96</td><td>49.13</td></tr><tr><td>Truncate-Thinking</td><td>94.78</td><td>76.32</td><td>86.91</td><td>48.85</td></tr><tr><td rowspan="3">k=8</td><td>Major voting</td><td>92.17</td><td>76.32</td><td>85.86</td><td>48.41</td></tr><tr><td>Truncate</td><td>93.91</td><td>78.95</td><td>87.96</td><td>49.25</td></tr><tr><td>Truncate-Thinking</td><td>94.78</td><td>77.63</td><td>87.96</td><td>49.25</td></tr><tr><td rowspan="3">k=16</td><td>Major voting</td><td>92.17</td><td>76.32</td><td>85.86</td><td>48.41</td></tr><tr><td>Truncate</td><td>94.78</td><td>81.57</td><td>89.53</td><td>49.45</td></tr><tr><td>Truncate-Thinking</td><td>94.78</td><td>78.95</td><td>88.48</td><td>49.38</td></tr></table>

Search tasks. It is noteworthy that the latter two models both rely on external tool manipulation to assist in object grounding. This result indicates that enhancing the intrinsic perceptual abilities of multimodal base models is a highly promising research direction, capable of rivaling the performance of tool-augmented SOTA methods.

Test-time Scaling with PRM. As mentioned earlier, PERCEVAL has the potential to assist in the test-time scaling of policy models with the Truncate or Feedback strategies. To validate their effectiveness, we compare them with the major voting strategy, a classic test-time scaling method, where the policy model generate responses for multiple times and selects the most common answer as the final response. We conducted the experiment on the 3B policy model and present the results in Table 2. With different sampling times k, the PRM-based strategies consistently outperform major voting on ${ \mathrm { V } } ^ { * }$ and BLINK. The Truncate strategy, in particular, shows a more significant improvement compared to the Feedback strategy. We hypothesize that the model’s training data does not contain sufficient reflective data, which results in poorer instruction-following quality when the reflective prompts are inserted in the Feedback strategy. In contrast, the Truncate strategy allows the model to regenerate the response based on its own generated context, aligning more closely with the model’s original distribution, thus producing more stable and reliable outputs. Another observation is that the major voting strategy quickly converges on difficult tasks (e.g., the Pos subset of V\*) and fails to show further improvement. This suggests that without external intervention, the model’s inherent capabilities are insufficient to rectify its errors.

![](images/a5da04ebf7dfb58bb6d77345634fb3809f780ba59fe0ce804672979a95873ff6.jpg)  
Figure 2. The proportion of responses identified by PERCEVAL as containing hallucinations during training.

## 4.3. Further Analysis

Reward Hacking Test. A critical challenge in reinforcement learning with reward models (RMs) is reward hacking, where the policy overfits the RM’s scoring function. This issue is particularly pronounced with traditional RMs that output a single scalar reward for an entire response. Such a direct and holistic score, which is often influenced by the RM’s own intrinsic biases, provides a simple signal for the policy to exploit, leading to score inflation without genuine quality improvement. Our proposed PERCEVAL is designed to mitigate this specific vulnerability. Instead of providing a direct scalar reward, PERCEVAL intervenes during the advantage calculation stage. Specifically, it reduces the advantage values of only those tokens within a response that are identified as contributing to a hallucination. This fine-grained, indirect guidance mechanism is inherently more difficult for the policy to overfit and simultaneously enhances the contrast between correct and incorrect tokens within the same sequence. The effectiveness of this approach is demonstrated in Figure 2, which plots the proportion of responses identified by PERCEVAL as containing hallucinations during training. The curve initially shows a decline, indicating that the policy is successfully learning to reduce hallucinations. Crucially, the rate then stabilizes rather than continuing to drop. A continuously decreasing curve would suggest that the policy is learning to deceive the PRM—a clear sign of reward hacking. The observed stability therefore confirms that our proposed PERCEVAL effectively guides the policy toward genuine improvement while avoiding significant reward hacking.

Hyperparameter Tuning. Our proposed RL training with PRM framework introduces the hyperparameter α (Equation 3), which governs the penalty strength applied

Table 3. Ablation study on the penalty strength hyperparameter α.
<table><tr><td>α</td><td>V*</td><td>RealWorldQA</td><td>MathVision</td><td>ChartQA</td></tr><tr><td>0.0</td><td>80.10</td><td>62.17</td><td>23.36</td><td>83.32</td></tr><tr><td>0.03</td><td>81.68</td><td>63.09</td><td>22.70</td><td>84.44</td></tr><tr><td>0.1</td><td>83.25</td><td>64.92</td><td>26.32</td><td>85.04</td></tr><tr><td>0.3</td><td>78.53</td><td>61.78</td><td>22.04</td><td>84.56</td></tr></table>

![](images/c548dbe5c3f1522ba04a763f57cba273d72cb700b6f3fa13cbc144e5ef2c6430.jpg)  
Question: Is the blue truck on the left or right side of the white vehicle?

GRPO: <think>The blue truck is visible in the image, positioned to the left of the white vehicle. This can be determined by observing the relative positions of the vehicles in the image.</think> <answer>B. The blue truck is on the left side of the white vehicle.</answer>"

Ours: <think>To determine the correct position of the blue truck, I need to identify the white vehicle first. The white vehicle appears to be a car parked near the building with the red roof in the center of the image. The blue truck is positioned to the right of this white vehicle.</think> <answer>A. The blue truck is on the right side of the white vehicle.</answer>

Figure 3. Case study of the visual reasoning process from models trained with GRPO and our method.

to tokens identified as hallucinatory. The selection of an optimal α is critical, as it requires balancing the suppression of hallucinations against the preservation of overall response quality. To quantitatively determine this optimal value, we conduct a series of experiments, varying α across {0.03, 0.1, 0.3} and benchmarking against a standard GRPO baseline $( \alpha ~ = ~ 0 )$ The results, summarized in Table 3, reveal a distinct non-monotonic trend. A minimal value of α = 0.03 provides an insufficient corrective gradient. While offering a marginal improvement over the baseline, the penalty is too subtle to effectively steer the model away from ingrained hallucinatory patterns. Conversely, an excessively large α of 0.3 proves counterproductive. We attribute this to collateral “penalization”: since the PERCEVAL flags entire substrings, a high penalty indiscriminately punishes all tokens within that span, including syntactically necessary but factually benign words (e.g., articles, prepositions). This introduces significant training noise and degrades overall performance. The analysis reveals that α = 0.1 strikes the optimal balance. It is potent enough to achieve a substantial reduction in hallucinations while avoiding the destabilizing effects of overpenalization. Therefore, we adopt α = 0.1 as the canonical value for all other experiments.

Qualitative Analysis. To clearly demonstrate the efficacy of our method, we present a qualitative analysis of model outputs in Figure 3. This case study compares the outputs from a model trained with direct GRPO against one trained with our method on an identical query. The task necessitates locating two minuscule objects (i.e., a blue vehicle and a vehicle car) to determine their spatial relationship. The baseline model, trained with direct GRPO, bypasses the perceptual task and directly outputs a relative position (“left”). This is a classic example of hallucination, as the model provides an answer without seemingly grounding its response in the visual evidence. In sharp contrast, our model exhibits a deliberate, step-by-step process. It first attempts to locate the white car, subsequently searches for the blue car, and then correctly deduces their relative positions. This case study demonstrates that our RL training process signif icantly enhances the model’s perceptual capabilities, compelling its responses to be faithfully grounded in the visual content.

## 5. Related Work

Vision-language Models The field of vision-language models (VLMs) has evolved from foundational representation alignment to complex multimodal reasoning. Early breakthroughs such as CLIP [32] and ALIGN [16] demonstrate that contrastive pre-training on web-scale image-text pairs yields powerful, transferable representations, setting the stage for Large Vision Language Models (LVLMs) that bridge pre-trained visual encoders with LLMs [2, 18, 23]. “Visual Instruction Tuning” [23] emerges as a critical paradigm for unlocking multimodal instruction-following, rapidly scaled in open-source models like Qwen-VL [3] and InternVL [57]. By incorporating large-scale SFT and RL, advanced VLMs [12, 44] achieve strong performance on complex reasoning tasks. However, perceptual capabilities remain a critical bottleneck: models frequently exhibit hallucinations [19, 20] or are unduly dominated by textual priors [1, 22, 53], highlighting a persistent gap in reliable, fine-grained visual perception.

Reinforcement Learning for VLMs The application of RL to VLMs has rapidly evolved toward capability incentivization for complex multimodal reasoning. This shift was catalyzed by breakthroughs in LLMs demonstrating that large-scale RL can elicit emergent “slow-thinking” behaviors [13, 15, 35], inspiring a new wave of VLM research that optimizes the synergy between visual perception and logical deliberation [14, 30, 34]. Beyond adapting LLM strategies, researchers explore reflection techniques tailored to the visual domain and “thinking with images” paradigms that leverage image manipulation tools to support reasoning. However, a critical limitation persists: methods based on RLVR predominantly rely on GRPO, which provides only coarse, outcome-level supervision and lacks the finegrained signals necessary for improving complex, step-bystep reasoning.

Multimodal Reward Models Multimodal reward models [40, 46, 52] play a pivotal role in Reinforcement Learning from Human Feedback (RLHF) by aligning model outputs with human preferences. With the recent proliferation of reinforcement learning in complex reasoning tasks, RMs are also increasingly employed to supplement methods like Reinforcement Learning with Verifiable Rewards (RLVR). This becomes particularly crucial in domains where verifiable ground truth is inaccessible, such as open-ended creative tasks [25], which are environments where methods reliant on verifiable rewards consequently struggle. The predominant approach for these RMs involves training them to directly output a single scalar score, which represents the overall quality of a given trajectory [39, 46]. Recognizing the limitations of this direct scoring, more recent research efforts have sought to integrate “slow thinking” or deliberate reasoning paradigms into reward modeling [49, 51]. These approaches enable the RM to generate a rationale or critique before assigning the final score, aiming for more meticulous and robust evaluations [9]. However, a fundamental limitation persists: whether generated directly or after deliberation, the feedback from existing RMs ultimately collapses into a single scalar reward. This offers only sparse, outcome-level supervision for algorithms like GRPO. We propose a perception-centric reward model that provides a more fine-grained signal, which enabling tokenlevel adjustments of advantages, thereby offering a more precise supervision.

## 6. Conclusion

In this work, we introduced PERCEVAL, a perceptioncentric process reward model (PRM) that addresses the sparse reward issue in RLVR by enabling token-level error grounding. Unlike traditional outcome-level supervision, PERCEVAL detects image–text misalignments within the model’s reasoning process and provides grounded, stepaware feedback. We trained PERCEVAL with perceptionintensive data and integrate it into both the training and inference stages of VLMs. At the training stage, we leverage PERCEVAL to apply token-level penalties to hallucinated spans, improving fine-grained credit assignment and surpassing the capabilities of sequence-level methods like GRPO. During inference, PERCEVAL enables a Truncation–Regeneration loop that prunes erroneous responses and induces model reflection. Our experiments demonstrate that PERCEVAL substantially improves visual grounding on perception-heavy benchmarks and facilitates better transfer to multi-step reasoning tasks. This method represents a significant advancement in fine-tuning the reasoning capabilities of VLMs, with the potential to generalize across domains and tasks.

## 7. Acknowledge

This work was partially supported by the National Natural Science Foundation of China No. 92470205 and Beijing Major Science and Technology Project under Contract No. Z251100008425002.

## References

[1] Aakriti Agrawal, Gouthaman KV, Rohith Aralikatti, Gauri Jagatap, Jiaxin Yuan, Vijay Kamarshi, Andrea Fanelli, and Furong Huang. Towards mitigating hallucinations in large vision-language models by refining textual embeddings. arXiv preprint arXiv:2511.05017, 2025. 1, 8

[2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022. 8

[3] Jinze Bai, Shuai Bai, Keqin Chen, Mengfei Du, Yang Fan, Zhihao Fan, Wenbin Ge, Dayiheng Liu, Rui Men, Xuancheng Ren, et al. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 1, 8

[4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 6

[5] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large vision-language models, 2025. 5, 6

[6] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024. 1, 4

[7] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024. 1

[8] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: Complex visionlanguage reasoning via iterative sft-rl cycles, 2025. 5, 6

[9] Kaixuan Fan, Kaituo Feng, Haoming Lyu, Dongzhan Zhou, and Xiangyu Yue. Sophiavl-r1: Reinforcing mllms reasoning with thinking reward, 2025. 8

[10] Kaixuan Fan, Kaituo Feng, Haoming Lyu, Dongzhan Zhou, and Xiangyu Yue. Sophiavl-r1: Reinforcing mllms reasoning with thinking reward, 2025. 3, 5

[11] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024. 1, 4

[12] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, et al. Gemini: A family of highly capable multimodal mod els. arXiv preprint arXiv:2508.11630, 2025. 1, 8

[13] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1, 3, 8

[14] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025. 5, 6, 8

[15] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024. 8

[16] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR, 2021. 8

[17] Hadi Khalaf, Claudio Mayrink Verdun, Alex Oesterling, Himabindu Lakkaraju, and Flavio du Pin Calmon. Inference time reward hacking in large language models, 2025. 1

[18] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In In ternational conference on machine learning, pages 19730– 19742. PMLR, 2023. 8

[19] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023. 1, 8

[20] Yifan Li, Kun Zhou, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. Analyzing and mitigating object hallucination: A training bias perspective. arXiv preprint arXiv:2508.04567, 2025. 1, 8

[21] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Repre sentations, 2023. 1

[22] Chengzhi Liu, Zhongxing Xu, Qingyue Wei, Juncheng Wu, James Zou, Xin Eric Wang, Yuyin Zhou, and Sheng Liu. More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models. arXiv preprint arXiv:2505.21523, 2025. 1, 8

[23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.

Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 8

[24] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024. 1

[25] Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. Inference-time scaling for generalist reward modeling, 2025. 8

[26] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024. 1, 5

[27] Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, 2022. Association for Computational Linguistics. 1, 5

[28] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning, 2025. 5, 6

[29] Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In The IEEE Winter Conference on Applications of Computer Vision (WACV), 2020. 5

[30] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025. 5, 8

[31] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl, 2025. 6

[32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 8

[33] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. 1, 2

[34] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025. 5, 6, 8

[35] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025. 1, 8

[36] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025. 5, 6

[37] Haozhe Wang, Alex Su, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning, 2025. 5

[38] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In The Thirty-eight Conference on Neural Infor mation Processing Systems Datasets and Benchmarks Track, 2024. 1, 5

[39] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math shepherd: Verify and reinforce llms step-by-step without hu man annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9426–9439, 2024. 1, 8

[40] Yibin Wang, Zhimin Li, Yuhang Zang, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Unified multimodal chain-of-thought reward model through reinforcement finetuning, 2025. 8

[41] Zifu Wang, Junyi Zhu, Bo Tang, Zhiyu Li, Feiyu Xiong, Ji aqian Yu, and Matthew B. Blaschko. Jigsaw-r1: A study of rule-based visual reinforcement learning with jigsaw puzzles, 2025. 5, 6

[42] Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 2023. 1, 3, 4

[43] xAI. Grok-1.5 vision preview. https://x.ai/news/ grok-1.5v, 2024. Accessed: 2024-08-27. 4

[44] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 8

[45] En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Jingyu Wang, and Wen bing Tao. Perception-r1: Pioneering perception policy with reinforcement learning, 2025. 2, 5, 6

[46] Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Ziyu Liu, Shengyuan Ding, Shenxi Wu, Yubo Ma, Haodong Duan, Wenwei Zhang, Kai Chen, Dahua Lin, and Jiaq Wang. Internlm-xcomposer2.5-reward: A simple yet effec tive multi-modal reward model, 2025. 8

[47] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reaso with multimodal large language models via step wise group relative policy optimization, 2025. 5, 6

[48] Jipeng Zhang, Kehao Miao, Renjie Pi, Zhaowei Wang, Run tao Liu, Rui Pan, and Tong Zhang. Vl-genrm: Enhancing vision-language verification via vision experts and iterative training. arXiv preprint arXiv:2506.13888, 2025. 1

[49] Xiangxiang Zhang, Jingxuan Wei, Donghong Zhong, Qi Chen, Caijun Jia, Cheng Tan, Jinming Gu, Xiaobo Qin, Zhiping Liu, Liang Hu, Tong Sun, Yuchen Wu, Zewei Sun, Chenwei Lou, Hua Zheng, Tianyang Zhan, Changbao Wang, Shuangzhi Wu, Zefa Lin, Chang Guo, Sihang Yuan, Riwei Chen, Shixiong Zhao, Yingping Zhang, Gaowei Wu, Bihui Yu, Jiahui Wu, Zhehui Zhao, Qianqian Liu, Ruofeng Tang, Xingyue Huang, Bing Zhao, Mengyang Zhang, and Youqiang Zhou. Structvrm: Aligning multimodal reasoning with structured and verifiable reward models, 2025. 8

[50] Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? arXiv preprint arXiv:2408.13257, 2024. 1, 4

[51] Yi-Fan Zhang, Xingyu Lu, Xiao Hu, Chaoyou Fu, Bin Wen, Tianke Zhang, Changyi Liu, Kaiyu Jiang, Kaibing Chen, Kaiyu Tang, Haojie Ding, Jiankang Chen, Fan Yang, Zhang Zhang, Tingting Gao, and Liang Wang. R1-reward: Training multimodal reward model through stable reinforcement learning, 2025. 8

[52] Yi-Fan Zhang, Haihua Yang, Huanyu Zhang, Yang Shi, Zezhou Chen, Haochen Tian, Chaoyou Fu, Haotian Wang, Kai Wu, Bo Cui, Xu Wang, Jianfei Pan, Haotian Wang, Zhang Zhang, and Liang Wang. Basereward: A strong baseline for multimodal reward model, 2025. 8

[53] Zhuoran Zhang, Tengyue Wang, Xilin Gong, Yang Shi, Haotian Wang, Di Wang, and Lijie Hu. When modalities conflict: How unimodal reasoning uncertainty governs preference dynamics in mllms. arXiv preprint arXiv:2511.02243, 2025. 1, 8

[54] Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning, 2025. 1

[55] Congming Zheng, Jiachen Zhu, Zhuoying Ou, Yuxiang Chen, Kangning Zhang, Rong Shan, Zeyu Zheng, Mengyue Yang, Jianghao Lin, Yong Yu, et al. A survey of process reward models: From outcome signals to process supervisions for large language models. arXiv preprint arXiv:2510.08049, 2025. 1

[56] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing "thinking with images" via reinforcement learning, 2025. 3, 5, 6

[57] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 8