[← 返回 README](../README.md)

# 4. Multiple Benchmark Integration

## 📌 预览

这节讲怎么把多个 benchmark 统一整合进来。4.1 给出统一整合接口——每个 benchmark 用三件套(checkpoint 包 / 可运行训练入口 / 可运行评测流程)对齐,并统一暴露 WebSocket 服务。4.2 逐一介绍支持的 benchmark 套件:LIBERO、LIBERO-Plus、SimplerEnv、RoboCasa-GR1、RoboTwin 2.0、BEHAVIOR-1K、CALVIN,给出各自的任务数、demo 规模、评测协议。

---

Recent vision-language-action (VLA) research has made rapid progress across a wide range of benchmarks. However, most existing methods are evaluated on only a limited number of environments, and their implementations often differ substantially in preprocessing pipelines, policy interfaces, and evaluation protocols. These inconsistencies hinder fair cross-paper comparison and weaken reproducibility.

> 💡 **问题动机** (claude 批注): 这段把评测碎片化再强调一遍,作为第 4 节的动机。三个不一致源:预处理管线、policy 接口、评测协议。前两个已被第 3 节的统一 I/O + server-client 解决,第 4 节要解决第三个——把各 benchmark 的官方评测协议以标准化方式接进来。

---

## 4.1 Unified Benchmark Integration Interface

> 💡 **4.1 要点预览** (claude 批注): 整合原则有两条:(i) 尽量贴近每个 benchmark 的官方训练/评测流程,最小化数据工程和环境改动;(ii) 标准化 policy 侧接口。落地成"每个 benchmark = 三件对齐组件"。

StarVLA aims to provide simple and reproducible baselines across a diverse benchmark suite by: (i) adhering as closely as possible to the official training and evaluation workflows of each benchmark, with minimal data engineering and environment-specific modifications, and (ii) standardizing the policy-side interface. Concretely, all StarVLA variants expose a unified lightweight WebSocket service, enabling different benchmark runners to interact with a shared inference endpoint. This design facilitates seamless integration and simplifies scaling to additional benchmarks.

To facilitate reproducibility, StarVLA defines a unified integration interface for benchmark onboarding. Specifically, each benchmark integration is structured around three aligned components: (i) a checkpoint package containing the saved config.yaml and dataset\_statistics.json, (ii) a runnable training entry (YAML configuration and launch script under examples/<BENCH>/train\_files/), and (iii) a runnable evaluation workflow that launches a policy server and invokes the official benchmark evaluator (typically under examples/<BENCH>/eval\_files/). This design ensures that benchmark-specific workflows remain reproducible while maintaining a consistent policy interface across environments.

> 💡 **机制拆解** (claude 批注): "benchmark onboarding = 三件对齐组件"是本文可复现性的工程契约,值得记住:
> - **(i) checkpoint 包**: 含 `config.yaml`(还原模型配置)+ `dataset_statistics.json`(动作反归一化统计量)。有了它,评测端才能把归一化动作还原成真实量纲。
> - **(ii) 可运行训练入口**: `examples/<BENCH>/train_files/` 下的 YAML + 启动脚本。
> - **(iii) 可运行评测流程**: `examples/<BENCH>/eval_files/`,启动 policy server 并调用**官方** benchmark 评测器。
>
> 关键在"调用官方评测器"——不是自己重写评测逻辑,而是保留官方协议以保证数字可比。这是"贴近官方 + 标准化 policy 接口"两条原则的落地形态,也是加新 benchmark 只需填这三件套的原因。

---

## 4.2 Supported Benchmark Suite

> 💡 **4.2 要点预览** (claude 批注): 逐个介绍 benchmark。读时抓三个量:任务数、每任务 demo 数(训练规模)、评测协议(episode 数/成功率聚合方式)。这些直接决定第 5 节结果的可比性。下表先做速查:
>
> | Benchmark | 任务数 | 训练 demo | 评测规模/协议要点 |
> |-----------|-------|-----------|------------------|
> | LIBERO | 130(4 套件) | 50/任务(约 6.5K 轨迹) | 每套件 10 任务 × 50 episode |
> | LIBERO-Plus | 10,030(test-only) | 无(纯测试) | 7 种扰动因子鲁棒性 |
> | SimplerEnv | 不固定 | 用真实数据(Bridge/Fractal) | Visual Matching / Variant Aggregation |
> | RoboCasa-GR1 | 24 | ~1000/任务(约 24K) | 人形桌面操作 |
> | RoboTwin 2.0 | 50 | 50 clean+500 rand(约 27.5K) | 50×2×100=10,000 trial |
> | BEHAVIOR-1K | 1000 活动(挑 50 任务) | 200/任务 | 长时序,部分得分 |
> | CALVIN | 34 | A–C 训 D 测 | ABC→D,长度 5 序列 |

StarVLA integrates a diverse set of manipulation benchmarks spanning different simulators, embodiments, and protocols, including LIBERO, SimplerEnv, RoboTwin 2.0, RoboCasa GR1 Tabletop Tasks, and BEHAVIOR-1K. The experiments section reports detailed results and comparisons under each benchmark’s official evaluation protocols.

LIBERO. LIBERO (Liu et al., 2024a) is a widely used benchmark for language-conditioned robot manipulation and lifelong robot learning. It contains 130 manipulation tasks organized into four suites: Spatial, Object, Goal, and Long, each targeting a different form of generalization, including spatial variation, object-centric manipulation, goal-conditioned execution, and long-horizon dependencies. A standard training protocol uses 50 demonstrations per task, resulting in approximately 6.5K trajectories in total. LIBERO provides a standardized evaluation protocol and serves as a comprehensive testbed for instruction following, compositional generalization, and multi-task policy learning.

> 💡 **机制拆解** (claude 批注): LIBERO 的四套件各测一种泛化维度——Spatial(空间变化)、Object(物体中心)、Goal(目标条件)、Long(长时序依赖)。记住 Long 是最难的:第 5 节里几乎所有方法在 Long 上都掉分最多(如 π0+FAST 在 Long 只有 60.2%,而 Spatial/Object 都 96+)。这解释了为什么 LIBERO 被当作"instruction following + 组合泛化 + 多任务"的综合测试床。

LIBERO-Plus. LIBERO-Plus (Fei et al., 2025) is a robustness-oriented benchmark built on top of LIBERO for systematically evaluating the generalization ability of vision-language-action models under distribution shifts. It expands the original benchmark by introducing perturbations over seven factors, including object layout, camera viewpoints, robot initial states, language instructions, lighting, background textures, and sensor noise. The final benchmark is a test-only evaluation set with 10,030 tasks spanning 7 perturbation factors and 21 low-level components.

> 💡 **机制拆解** (claude 批注): LIBERO-Plus 是**纯测试集**(test-only,10,030 任务),专门做鲁棒性/分布偏移评估。7 种扰动因子(物体布局、相机视角、机器人初始状态、语言指令、光照、背景纹理、传感噪声)覆盖了真机部署常见的 domain shift。它的定位是"训练还在原 LIBERO,测试换成扰动版",用来量化模型对分布偏移的敏感度。

SimplerEnv. SimplerEnv (Li et al., 2024b) is a simulation-based evaluation benchmark designed as a scalable proxy for real-world robot evaluation. It provides standardized simulated environments corresponding to common real-robot platforms, including the WidowX (BridgeData V2) and the Google Robot (RT-series) setups. The benchmark defines fixed evaluation protocols such as Visual Matching and Variant Aggregation, as well as standardized success-rate aggregation rules. Although it does not specify a fixed number of tasks or dataset size, it is widely used to evaluate policies trained on real-world data under reproducible simulated conditions, with prior work showing strong correlation between simulated and real-world performance.

> 💡 **机制拆解** (claude 批注): SimplerEnv 的独特之处是"用仿真当真机评测的可扩展代理"——它复现了 WidowX(对应 BridgeData V2)和 Google Robot(对应 RT 系列)两个真实平台,且已被证明仿真与真机成绩强相关。两个评测协议要区分:**Visual Matching(VM)** 是让仿真视觉尽量贴近真实数据分布,**Variant Aggregation(VA)** 是在多种视觉变体上聚合成绩测鲁棒性。第 5.2 节 Table 3/4 就分这两个协议报数。

RoboCasa-GR1. RoboCasa-GR1 (Nasiriany et al., 2024; Bjorck et al., 2025) is a tabletop manipulation benchmark built on the RoboCasa simulation framework, commonly used to evaluate humanoid-style manipulation policies. Compared with standard single-arm setups, it introduces more complex embodiments and household interaction scenarios involving articulated objects and multi-stage tasks. The benchmark contains 24 tasks, with approximately 1,000 demonstrations per task, resulting in around 24K trajectories in total.

> 💡 **机制拆解** (claude 批注): RoboCasa-GR1 是**人形(humanoid-style)桌面操作**,比单臂设置更复杂(铰接物体、多阶段任务)。24 任务、每任务约 1000 demo。第 5.3 节明说它"比 LIBERO/SimplerEnv 难得多,且 action head 的选择影响更大"——离散 FAST 只有 39%,连续变体到 48.8%。这个 benchmark 是本文区分不同 head 优劣的关键战场。

RoboTwin 2.0. RoboTwin 2.0 (Chen et al., 2025b) is a large-scale benchmark for bimanual robotic manipulation, focusing on dual-arm coordination across diverse scenarios. It contains 50 tasks with two evaluation setups: clean and randomized. Each task includes 50 clean demonstrations together with 500 randomized demonstrations, resulting in approximately 550 trajectories per task and 27.5K trajectories in total. The randomized data is generated via structured domain randomization, including variations in scene clutter, backgrounds, table height, and lighting, providing a challenging testbed for both coordination and robustness. For evaluation, each task is tested for 100 episodes under each setup. In total, this results in 50 tasks × 2 setups × 100 episodes, equals to 10,000 evaluation trials.

> 💡 **机制拆解** (claude 批注): RoboTwin 2.0 是**双臂协调**benchmark,50 任务,每任务 50 clean + 500 randomized demo。clean/randomized 双设置的设计很聪明:randomized 用结构化域随机(场景杂乱、背景、桌高、光照)专门测鲁棒性。评测规模 50×2×100=10,000 trial。第 5.4 节 Table 7 里 StarVLA 变体在 randomized 上甚至常高于 clean,说明域随机训练数据反而帮助了泛化。

BEHAVIOR-1K. BEHAVIOR-1K (Li et al., 2023) is a large-scale benchmark for human-centered embodied AI, built around everyday activities. It defines 1,000 activities across 50 interactive scenes with more than 9,000 objects, covering environments such as homes, offices, and restaurants. Built on OmniGibson, it supports realistic physics for rigid, deformable, and liquid objects, and emphasizes long-horizon interaction requiring perception, navigation, and manipulation. An active evaluation setting is the BEHAVIOR Challenge, which selects 50 household tasks from the activity set and provides 10,000 teleoperated demonstrations (over 1,200 hours), with 200 demonstrations per task released for training. For evaluation, each task includes 20 additional instances with varying initial conditions, of which 10 are used for reporting, and each instance is evaluated once with a fixed timeout. Performance is measured by the average task success rate across all tasks, with partial credit based on goal completion.

> 💡 **机制拆解** (claude 批注): BEHAVIOR-1K 是规模最大、最贴近真实家务的 benchmark:1000 活动 / 50 场景 / 9000+ 物体,基于 OmniGibson 支持刚体/可变形/液体的真实物理。它强调**长时序 + 感知+导航+操作三合一**,且用"部分得分(partial credit,按目标完成度)"而非二元成功率——这对超长任务更合理。注意:虽然摘要列了 BEHAVIOR-1K 为整合 benchmark,但第 5 节的详细结果表里并未报 BEHAVIOR-1K 的数字,说明它是已接入但结果待补的 benchmark。

CALVIN. CALVIN (Mees et al., 2022) is a benchmark for long-horizon language-conditioned manipulation, designed to evaluate whether a single policy can execute sequences of natural-language instructions from visual observations. It comprises four environments (A, B, C, and D) and 34 manipulation tasks involving articulated objects and stateful scene elements such as drawers, sliding doors, lights, and switches. The standard evaluation follows the ABC→D setting, where policies are trained on A–C and tested on D using 1,000 task sequences of length 5. Performance is reported by the average length of successfully completed subtask sequences.

> 💡 **机制拆解** (claude 批注): CALVIN 测的是"长时序语言指令**序列**执行"——ABC→D 设置是标准的跨环境泛化(A–C 训练、D 测试),指标是"成功完成的连续子任务平均长度"(而非单任务成功率),因为它要考核连贯执行 5 步指令链的能力。这个指标对"中途断链"很敏感。

> 💡 **Q&A 批注记录** (claude 批注):
> - Q: 摘要说整合 5 个 benchmark,Table 1 说 #Bench=7,这节却列了 7 个,到底几个?
> - A: 主打的核心 5 个(LIBERO、SimplerEnv、RoboTwin 2.0、RoboCasa-GR1、BEHAVIOR-1K)是摘要口径;Table 1 的 #Bench=7 与本节把 LIBERO-Plus、CALVIN 也算进"整合套件"一致(7 = 5 核心 + LIBERO-Plus + CALVIN)。差异来自"主打 benchmark"与"全部已接入 benchmark"两种计数口径,并非矛盾。

> 💡 **Section 总结** (claude 批注):
> - **整合契约**: 每个 benchmark = checkpoint 包 + 训练入口 + 评测流程三件套,且调用**官方**评测器保证可比。
> - **难度梯度**: LIBERO/SimplerEnv(较易)< RoboCasa-GR1 / RoboTwin 2.0(较难,能区分 head 优劣)< BEHAVIOR-1K(最大最难,结果待补)。
> - **协议速查**: SimplerEnv 分 VM/VA;RoboTwin 分 clean/randomized;LIBERO 分 4 套件;CALVIN 用子任务链长度;BEHAVIOR 用部分得分。
> - **可追问点**: 这些 benchmark 上四个 StarVLA 变体各自表现如何?见第 5 节。
