[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要给出全文的压缩版问题链：OSD 已经把 diffusion SR 做到单步高效，但固定 timestep/固定蒸馏强度会把输出锁在一个静态 fidelity-realism 折中点。RCOD-SR 的核心是把退化程度映射到 latent domain group 与 timestep，再用 DAS 和 VPIM 让这个控制在训练、蒸馏和条件输入中都成立。
---

> 💡 **Q&A 批注记录**:
> - Q: 为什么单步 SR 难以控制 realism？
> - A: 因为多数 one-step 模型在固定 timestep/固定蒸馏强度下学习一个确定映射，推理时没有多步采样那样的 noise schedule 可调自由度。
> - Q: 摘要里的 minimal training paradigm modifications 指什么？
> - A: 主要指 LDG 利用 diffusion 本来就有的 timestep condition 做分组控制，不需要为 realism slider 另起一个大模块；额外模块集中在 DAS 的采样策略和 VPIM 的条件替换上。


# Realism Control One-step Diffusion for Real-World Image Super-Resolution

Zongliang Wu1, 2, 3, \*, Siming Zheng 3, \*, Peng-Tao Jiang3, #, Xin Yuan2, #
> 💡 **基本信息**: 作者来自高校与 vivo，说明这篇的应用目标不是纯 benchmark，而是偏真实移动端/部署场景：单步效率和可控输出同等重要。


1 Zhejiang University, Hangzhou, China 2 School of Engineering, Westlake University, Hangzhou, China 3 vivo Mobile Communication Co., Ltd \* Equal Contribution # Corresponding Author {wuzongliang, xyuan}@westlake.edu.cn, {zhengsiming, pt.jiang}@vivo.com
> 💡 **读法提示**: 工业作者参与时，后面实验里的 A100 延迟、参数量、去掉 VLM/text extractor 的开销收益都要认真看，不能只看感知指标。


# Abstract

Pre-trained diffusion models have shown great potential in real-world image super-resolution (Real-ISR) tasks by enabling high-resolution reconstructions. While one-step diffusion (OSD) methods significantly improve efficiency compared to traditional multi-step approaches, they still have limitations in balancing fidelity and realism across diverse scenarios. Since the OSDs for SR are usually trained or distilled by a single timestep, they lack flexible control mechanisms to adaptively prioritize these competing objectives, which are inherently manageable in multi-step methods through adjusting sampling steps. To address this challenge, we propose a Realism Controlled One-step Diffusion (RCOD) framework for Real-ISR. RCOD provides a latent domain grouping strategy that enables explicit control over fidelity-realism trade-offs during the noise prediction phase with minimal training paradigm modifications and original training data. A degradation-aware sampling strategy is also introduced to align distillation regularization with the grouping strategy and enhance the controlling of trade-offs. Moreover, a visual prompt injection module is used to replace conventional text prompts with degradation-aware visual tokens, enhancing both restoration accuracy and semantic consistency. Our method achieves superior fidelity and perceptual quality while maintaining computational efficiency. Extensive experiments demonstrate that RCOD outperforms state-of-the-art OSD methods quantitatively and visually, with flexible realism control capabilities in the inference stage. Code is available at https://zongliang-wu.github.io/RCOD-SR.
> 💡 **摘要主线**: 这段有三个关键判断。第一，固定 timestep 是 OSD 可控性不足的根因：模型只能学到一个平均退化偏好。第二，LDG 不直接“生成更多纹理”，而是把退化程度转成 denoising degree，让推理时的 timestep 变成 realism 控制旋钮。第三，DAS 和 VPIM 分别补两处容易失效的环节：蒸馏正则不能抹平分组差异，prompt 条件不能和当前 LR 图像脱节。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 一句话 | 把退化严重度映射到 latent domain group / timestep，使单步 Real-ISR 也能在推理时调 fidelity-realism trade-off。 |
| 输入 | 真实退化 LR image + 可选 realism 控制强度 |
| 输出 | 可按 realism level 调节的 SR image |

### 核心洞察
1. RCOD-SR 的 novelty 不在“又一个 one-step SR”，而在让 one-step 模型恢复一部分多步采样里的可调 denoising 自由度。
2. LDG 是控制入口，DAS 是蒸馏对齐，VPIM 是条件质量；三者分别对应 timestep、teacher regularization、cross-attention condition。
3. 摘要的强 claim 是“更好指标 + 单步效率 + 推理可控”同时成立，后文实验需要逐项核验。

### 可追问点
- 为什么单步 SR 难以控制 realism？
- LDG 真正控制的是什么？
- VPIM 替换文本 prompt 后，语义能力来自哪里？
