[← 返回 README](../README.md)

# Abstract

## 📌 预览
本文件保留论文题名、作者、摘要等开场信息，先抓住作者定义的问题、方法名和主要 claim。

---

# The Latent Space: Foundation, Evolution, Mechanism, Ability, and Outlook

Xinlei $\forall \mathbf { u } ^ { * , \star }$ , Zhangquan Chen∗, Yongbo $\boldsymbol { \mathsf { H e } } ^ { * }$ , Tianyu $\boldsymbol { \mathsf { F u } } ^ { * }$ , Cheng Yang∗, Chengming $\mathbf { \times u } ^ { \ast }$ , Yue $\boldsymbol { \mathsf { M } } \boldsymbol { \mathsf { a } } ^ { * }$ , Xiaobin $\boldsymbol { \mathsf { H } } \boldsymbol { \mathsf { u } } ^ { * , \dagger }$ , Zhe Cao, Jie Xu, Guibin Zhang, Jiale Tao, Jiayi Zhang, Siyuan Ma, Kaituo Feng, Haojie Huang, Youxing Li, Ronghao Chen, Huacan Wang, Chenglin Wu, Zikun Su, Xiaogang Xu, Kelu Yao, Kun Wang, Chen Gao, Yue Liao, Ruqi Huang, Tao Jin, Zhucun Xue, Cheng Tan†, Jiangning Zhang†, Wenqi Ren, Yanwei Fu, Yong Liu, Yu Wang, Xiangyu Yue†, Yu-Gang Jiang†, Shuicheng Yan†

> 💡 **批注**: 这是一段信息密度较高的论述：建议拆成“现象/问题 → 机制 → 证据/影响”三层读。

∗ Core Contributors $\dagger$ Core Supervisors $\star$ Organizer

National University of Singapore, Fudan University, Tsinghua University, Zhejiang University, Shanghai Artificial Intelligence Laboratory, Renmin University of China, The Chinese University of Hong Kong, The Hong Kong University of Science and Technology, DeepWisdom, Nanjing University, Shanghai Jiatong University, Nanyang Technological University, Tencent Hunyuan, QuantaAlpha, Beijing University of Posts and Telecommunications, Zhejiang Lab, University of Chinese Academy of Sciences, Hong Kong University of Science and Technology (Guangzhou), Sun Yat-sen University

https://github.com/YU-deep/Awesome-Latent-Space

Latent space is rapidly emerging as a native substrate for language-based models. While modern systems are still commonly understood through explicit token-level generation, an increasing body of work shows that many critical internal processes are more naturally carried out in continuous latent space than in human-readable verbal traces. This shift is driven by the structural limitations of explicit-space computation, including linguistic redundancy, discretization bottlenecks, sequential inefficiency, and semantic loss. As a result, research on latent space has expanded from early latent reasoning into a broader landscape spanning planning, modeling, perception, memory, collaboration, and embodiment. However, the literature remains fragmented across mechanisms, modalities, and tasks, lacking a unified perspective on how latent space is defined, classified, and studied.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

This survey aims to provide a unified and up-to-date landscape of latent space in language-based models. We organize the survey into five sequential perspectives: Foundation, Evolution, Mechanism, Ability, and Outlook. We begin by delineating the scope of latent space, distinguishing it from explicit or verbal space and from the latent spaces commonly studied in generative visual models. We then trace the field’s evolution from early exploratory efforts to the current large-scale expansion. To organize the technical landscape, we examine existing work through the complementary lenses of mechanism and ability. From the perspective of Mechanism, we identify four major lines of development: Architecture, Representation, Computation, and Optimization. From the perspective of Ability, we show how latent space supports a broad capability spectrum spanning Reasoning, Planning, Modeling, Perception, Memory, Collaboration, and Embodiment. Beyond consolidation, we discuss the key open challenges, and outline promising directions for future research. We hope this survey serves not only as a reference for existing work, but also as a foundation for understanding latent space as a general computational and systems paradigm for next-generation intelligence.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

# Contents

---

## 🔖 Section 总结

### 核心洞察
1. 先记住本文的问题定义、方法名和主要 claim。
2. 后续阅读时回看摘要里的 claim 是否被实验支持。
