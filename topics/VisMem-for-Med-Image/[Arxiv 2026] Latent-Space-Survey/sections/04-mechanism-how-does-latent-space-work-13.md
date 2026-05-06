[← 返回 README](../README.md)

# Mechanism: How Does Latent Space Work? 13

## 📌 预览
本节保留原文并穿插批注，重点提炼与课题主线相关的机制和证据。

---

# 4.1 Architecture 15

4.1.1 Backbone 15   
4.1.2 Component 16   
4.1.3 Auxiliary Model 19

# .2 Representation 20

4.2.1 Internal 21   
4.2.2 External . 24   
4.2.3 Learnable 25   
4.2.4 Hybrid . 26

# .3 Computation 27

4.3.1 Compressed 27   
4.3.2 Expanded 29   
4.3.3 Adaptive 30   
4.3.4 Interleaved 32

# .4 Optimization 33

4.4.1 Pre-training . 33   
4.4.2 Post-training 34   
4.4.3 Inference 36

> 💡 **导读**: 这一节是全篇最关键的技术骨架。作者不按任务分方法，而是按 latent 在系统里“以什么形式存在、怎么被计算、什么时候被塑形”来分。

> 💡 **阅读重点**: Architecture 看 latent 放在哪；Representation 看 latent 是什么；Computation 看 latent 如何流动；Optimization 看 latent 在训练或推理哪个阶段被塑形。后面读单篇论文时，基本都可以沿这四问做拆解。

---

## 🔖 Section 总结

### 核心洞察
1. Mechanism 四分法比按任务分法更稳定，因为它抓的是系统设计决策而不是数据集名称。
2. 对 `VisMem-for-Med-Image` 这个 topic，`AlignVLM` 更偏 alignment/representation，`VisMem` 和 `MedSynapse-V` 更偏 storage/control，`Visual-Enhanced-Depth-Scaling` 更偏 computation/optimization。
3. 后续做批读时，最值得追问的是：论文到底新增了新的 latent 载体、计算路径，还是只是换了 loss 或训练日程。
