[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览

Related Work 分四条线定位 RoboTwin：①机器人数据集与基准（遥操作 / 算法轨迹生成器 / MimicGen·RoboCasa / ManiSkill）；②双臂操作（Peract2 / HumanoidBench / BiGym 的局限）；③机器人操作学习方法（BC、Offline RL、ACT、Diffusion Policy、DP3）；④LLM 做机器人代码生成（Code as Policies、RoboCodeX、ReKep 的不足）。每条线末尾都点出 RoboTwin 相对它的增量。

> 💡 **Figure 2 归属说明（claude 批注）**: MinerU 把 Figure 2 提取在 Related Work 这一页（page 2），但按其 caption（real-to-sim transfer + expert data generation），它实际描述的是方法主图，已按 skill 规则移入 [03-methodology](03-methodology.md)。本节不放图。

---

## 2.1. Datasets and Benchmarks for Robotics

To collect effective demonstrations for robotic tasks, human teleoperation is the most common approach, where human manually guides a robot across various tasks [18, 31, 50, 51, 53, 57]. Recent advancements have extended this methodology by employing teams of human operators over prolonged periods to assemble substantial real-world datasets [2, 7, 18, 31]. An alternative method employs algorithmic trajectory generators within simulators [15, 23, 30, 34, 74]. Nevertheless, such approaches typically demand manual, task-specific design for individual tasks. Recent initiatives like MimicGen [54] and RoboCaca [59] generate simulated expert data by adapting actions to new object poses, but remain limited to fixed scenarios and predefined task configurations. Furthermore, their reliance on fixed 3D objects limits the diversity of interacting objects and shapes. Besides, Maniskill [23, 69] provides diverse simulation scenarios but lakes automated data collection mechanism.

> 💡 **机制拆解（数据来源三分法）（claude 批注）**: 这段把数据来源分成三档并各自点出软肋：①遥操作（贵、需长期人力）；②仿真算法轨迹生成器（每任务手工设计）；③MimicGen/RoboCasa（能按新物体位姿适配动作放大数据，但**依赖固定 3D 物体**，物体/形状多样性受限）。最后一句点名 ManiSkill：场景多样但**缺自动数据采集机制**。注意 RoboTwin 恰恰用 ManiSkill3 当物理引擎（见 [04-benchmark](04-benchmark.md)），等于是"给 ManiSkill 补上自动采数据这块"。

In contrast, RoboTwin leverages 3D generative foundation models and LLMs to autonomously create both task variations and corresponding expert demonstrations. From 3D assets, it generates task scenarios and executable code via spatial reasoning, minimizing human intervention and supporting diverse object appearances.

> 💡 **与 baseline 的差异（claude 批注）**: 这段是本小节的"定位句"。相对上面三档方法，RoboTwin 的增量是同时自动生成**任务变体**和**对应专家演示**，且靠"3D 生成资产 + 空间推理"支撑物体外观多样性——把"固定 3D 物体"这个限制直接拿掉。

## 2.2. Dual-arm Manipulation

While significant advances have been made in single-arm manipulation, coordinated multi-arm manipulation remains largely unexplored. Peract2 [22] offers benchmarks for bimanual tasks with separated arms, but its setup lacks the complexity of integrated dual-arm systems. Humanoid-Bench [64] evaluates dexterous, whole-body manipulation with a humanoid robot in a fixed reinforcement learning benchmark, while BiGym [13] provides a bimanual benchmark but is constrained by VR teleoperation, limiting their scalability in data collection and evaluation. As a benchmark for dual-arm tasks, RoboTwin enables automatic and large-scale coordinated manipulation data generation with comprehensive policy evaluation.

> 💡 **机制拆解（双臂基准的空白）（claude 批注）**: 逐一对标同类双臂基准：Peract2 是"分离双臂"、缺一体化协调复杂度；HumanoidBench 是固定 RL 环境的全身操作；BiGym 靠 VR 遥操作采数据、规模受限。RoboTwin 的卖点是"**自动 + 大规模**的协调操作数据生成 + 全面策略评测"，正好补 scalability 这块。这与 [01-introduction](01-introduction.md) 里"可扩展 + 标准化双臂基准"的目标呼应。

## 2.3. Robot Manipulation Learning Methods

The adoption of human demonstrations to instruct robots in manipulation skills is a prevalent method in Robot Manipulation Learning [5, 6, 11, 14, 19, 32, 48, 49, 66, 73]. Among the techniques, Behavioral Cloning stands out for learning policies offline from these demonstrations. It replicates observed actions from a curated dataset [7, 15, 18, 31, 34, 52, 61, 75]. Conversely, Offline Reinforcement Learning enhances policy learning by optimizing actions based on a predefined reward function and exploiting large datasets [8, 24, 36–39, 47]. The Action Chunking with Transformers (ACT) technique integrates a Transformerbased visuomotor policy with a conditional variational autoencoder to structure the learning of action sequences [67, 70, 76]. Diffusion models have been introduced into robot imitation learning and are gradually becoming a mainstream approach due to their excellent generative capabil ities [3, 33, 43–45, 60]. Recently, the Diffusion Policy method has gained prominence. It employs a conditional denoising diffusion process for visuomotor policy representation, effectively reducing the accumulative error in trajectory generation that is often observed in Transformerbased visuomotor policies [14]. The 3D Diffusion Policy [73] uses point clouds for environmental observations, enhancing spatial information utilization and managing various robotic tasks in both simulated and real environments with only a small number of demonstrations.

> 💡 **机制拆解（为什么后面用 DP/DP3 当 baseline）（claude 批注）**: 这段梳理了模仿学习家谱：BC（离线复刻动作）→ Offline RL（按奖励优化）→ ACT（Transformer + CVAE 建模动作块）→ Diffusion 系（DP 用条件去噪扩散建模 visuomotor 策略，缓解 Transformer 的累积误差；DP3 用点云增强空间信息、少量演示即可）。这解释了 [05-experiments](05-experiments.md) 为何选 **DP（2D）与 DP3（3D，含 XYZ 与 XYZ+RGB 两种输入）** 作为基准算法——它们是当前主流且对"数据量/空间信息"敏感，正好用来检验 RoboTwin 数据的价值。

## 2.4. LLM for Robotic Code Generation.

With their remarkable ability in natural language understanding and code generation, Large Language Models (LLMs) have revolutionized numerous domains in artificial intelligence. In robotics, these models have shown exceptional capabilities in bridging the gap between natural language commands and executable robot actions [9, 10, 17, 21, 25–29, 42, 46, 58, 65, 71]. Code as Policies [41] and RoboCodeX [10, 56] established that LLMs can effectively translate high-level task descriptions into functional robot control programs. While Rekep [29] advances spatial reasoning between key points, it has limitations in handling functional axis constraints and fails to account for spatial relationships between object functional axes and the table surface during code generation. Furthermore, existing code generation approaches predominantly focus on single-arm robots, overlooking crucial aspects of dual-arm collaboration and active collision avoidance strategies.

> 💡 **与 baseline 的差异（本文技术缝隙）（claude 批注）**: 这段直指 RoboTwin 代码生成框架要补的两个洞。①ReKep 做了关键点间的空间推理，但**处理不了 functional axis 约束**、也不考虑"物体功能轴与桌面"的空间关系；RoboTwin 的空间标注恰恰引入 function/approach/lateral 三轴（见 [03-methodology](03-methodology.md) §3.2），就是冲这个来的。②已有代码生成基本只做单臂，忽略**双臂协作与主动避碰**；RoboTwin 的 API 里专门有 `get_avoid_collision_pose` 等双臂避碰接口（见 [07-appendix](07-appendix.md) D 节）。

---

## 🔖 Section 总结

### 核心洞察
1. RoboTwin 在四条线上都定位为"补空白"：数据（物体多样性 + 自动采集）、双臂基准（scalability）、学习方法（选 DP/DP3 当被测对象）、LLM 代码生成（functional axis 约束 + 双臂避碰）。
2. 它与 ManiSkill 是"用 + 补"的关系：借 ManiSkill3 当引擎，补上自动数据采集。
3. 选 DP/DP3 作 baseline 不是随意——它们对数据规模与空间信息敏感，正好检验生成数据的价值。

### 可追问点
- ReKep 的"functional axis 缺失"具体怎么被三轴标注解决？→ [03-methodology](03-methodology.md) §3.2 Axes 定义。
- 双臂避碰在代码层如何实现？→ [07-appendix](07-appendix.md) D.2 `get_avoid_collision_pose` 与 D.3 示例。

