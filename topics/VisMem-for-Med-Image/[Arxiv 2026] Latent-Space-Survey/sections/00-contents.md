[← 返回 README](../README.md)

# Contents

## 📌 预览
本节保留原文并穿插批注，重点提炼与课题主线相关的机制和证据。

---

# The Latent Space: Foundation, Evolution, Mechanism, Ability, and Outlook

Xinlei $\forall \mathbf { u } ^ { * , \star }$ , Zhangquan Chen∗, Yongbo $\boldsymbol { \mathsf { H e } } ^ { * }$ , Tianyu $\boldsymbol { \mathsf { F u } } ^ { * }$ , Cheng Yang∗, Chengming $\mathbf { \times u } ^ { \ast }$ , Yue $\boldsymbol { \mathsf { M } } \boldsymbol { \mathsf { a } } ^ { * }$ , Xiaobin $\boldsymbol { \mathsf { H } } \boldsymbol { \mathsf { u } } ^ { * , \dagger }$ , Zhe Cao, Jie Xu, Guibin Zhang, Jiale Tao, Jiayi Zhang, Siyuan Ma, Kaituo Feng, Haojie Huang, Youxing Li, Ronghao Chen, Huacan Wang, Chenglin Wu, Zikun Su, Xiaogang Xu, Kelu Yao, Kun Wang, Chen Gao, Yue Liao, Ruqi Huang, Tao Jin, Zhucun Xue, Cheng Tan†, Jiangning Zhang†, Wenqi Ren, Yanwei Fu, Yong Liu, Yu Wang, Xiangyu Yue†, Yu-Gang Jiang†, Shuicheng Yan†

∗ Core Contributors $\dagger$ Core Supervisors $\star$ Organizer

National University of Singapore, Fudan University, Tsinghua University, Zhejiang University, Shanghai Artificial Intelligence Laboratory, Renmin University of China, The Chinese University of Hong Kong, The Hong Kong University of Science and Technology, DeepWisdom, Nanjing University, Shanghai Jiatong University, Nanyang Technological University, Tencent Hunyuan, QuantaAlpha, Beijing University of Posts and Telecommunications, Zhejiang Lab, University of Chinese Academy of Sciences, Hong Kong University of Science and Technology (Guangzhou), Sun Yat-sen University

https://github.com/YU-deep/Awesome-Latent-Space

Latent space is rapidly emerging as a native substrate for language-based models. While modern systems are still commonly understood through explicit token-level generation, an increasing body of work shows that many critical internal processes are more naturally carried out in continuous latent space than in human-readable verbal traces. This shift is driven by the structural limitations of explicit-space computation, including linguistic redundancy, discretization bottlenecks, sequential inefficiency, and semantic loss. As a result, research on latent space has expanded from early latent reasoning into a broader landscape spanning planning, modeling, perception, memory, collaboration, and embodiment. However, the literature remains fragmented across mechanisms, modalities, and tasks, lacking a unified perspective on how latent space is defined, classified, and studied.

> 💡 **批注**: 开篇就把 latent space 从“隐藏状态的副产品”提升为 machine-native substrate。对这个 topic 来说，重点不是它能否替代所有文本推理，而是它是否能更稳定地承载视觉证据、记忆和跨步骤状态。

This survey aims to provide a unified and up-to-date landscape of latent space in language-based models. We organize the survey into five sequential perspectives: Foundation, Evolution, Mechanism, Ability, and Outlook. We begin by delineating the scope of latent space, distinguishing it from explicit or verbal space and from the latent spaces commonly studied in generative visual models. We then trace the field’s evolution from early exploratory efforts to the current large-scale expansion. To organize the technical landscape, we examine existing work through the complementary lenses of mechanism and ability. From the perspective of Mechanism, we identify four major lines of development: Architecture, Representation, Computation, and Optimization. From the perspective of Ability, we show how latent space supports a broad capability spectrum spanning Reasoning, Planning, Modeling, Perception, Memory, Collaboration, and Embodiment. Beyond consolidation, we discuss the key open challenges, and outline promising directions for future research. We hope this survey serves not only as a reference for existing work, but also as a foundation for understanding latent space as a general computational and systems paradigm for next-generation intelligence.

> 💡 **批注**: 五段式结构是这篇综述最重要的价值。Foundation 负责边界，Evolution 负责时间线，Mechanism 负责“怎么做”，Ability 负责“做成了什么”，Outlook 负责风险和下一步。

# Contents

---

## 🔖 Section 总结

### 核心洞察
1. 这篇 survey 不是在罗列方法，而是在给 latent space 建一个统一坐标系。
2. 对当前 `VisMem-for-Med-Image` 课题，最有用的是它把 memory、perception、reasoning 放进同一连续计算框架里。
3. 后续阅读时应一直带着两个问题：某篇论文在 mechanism 上属于哪一类，它最终增强的是哪种 ability。
