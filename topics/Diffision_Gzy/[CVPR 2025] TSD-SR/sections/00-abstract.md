[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要直接给出 TSD-SR 的问题定位：已有 one-step SR 虽快，但 VSD 梯度和细节恢复不足；本文用 Target Score Distillation 与 DASM 改造蒸馏信号。


Pre-trained text-to-image diffusion models are increasingly applied to real-world image super-resolution (Real-ISR) tasks. Given the iterative refinement nature of diffusion models, most existing approaches are computationally expensive. While methods such as SinSR and OSEDiff have emerged to condense inference steps via distillation, their performance in image restoration or details recovery is not satisfactory. To address this, we propose TSD-SR, a novel distillation framework specifically designed for real-world image super-resolution, aiming to construct an efficient and effective one-step model. We first introduce the Target Score Distillation, which leverages the priors of diffusion models and real image references to achieve more realistic image restoration. Secondly, we propose a Distribution-Aware Sampling Module to make detail-oriented gradients more readily accessible, addressing the challenge of recovering fine details. Extensive experiments demonstrate that our TSD-SR has superior restoration results (most of the metrics perform the best) and the fastest inference speed (e.g. 40 times faster than SeeSR) compared to the past Real-ISR approaches based on pre-trained diffusion priors. Our code is released at https://github.com/Microtreei/TSD-SR.

> 💡 **问题动机**: Real-ISR 已经转向利用 SD 这类大规模 T2I prior，但多步采样成本高；TSD-SR 关注的是把 prior 压到一步，同时不让细节恢复掉队。


---

## 🔖 Section 总结
- TSD-SR 的一句话目标是把 SD prior 蒸馏到 one-step Real-ISR，同时修正 VSD 在细节恢复中的方向问题。
- 核心变量是 student output、teacher score、LoRA fake score、target score 和 DASM noisy latent。
- 可追问：target score 如何比普通 VSD 更可靠？DASM 具体把 noisy latent 推向哪条轨迹？
