[← 返回 README](../README.md)

# 1 INTRODUCTION

## 📌 预览
本节建立动机链：diffusion/flow SR 质量高但依赖多步采样；已有单步蒸馏提升效率却把 fidelity-realism trade-off 固定住；OFTSR 的切入点是把 teacher 的 ODE 轨迹控制能力蒸馏给 one-step student。
---

> 💡 **Q&A 批注记录**:
> - Q: OFTSR 和 diffusion one-step 的差别在哪里？
> - A: 它把“可调”建在连续 flow/ODE 轨迹上，再要求 student 的不同 $t$ 输出落在 teacher 同一条轨迹上；普通 one-step 往往只学一个固定输出分布或固定 timestep 行为。


Recently, diffusion and flow-based generative models have demonstrated the ability to generate images with higher quality (Ramesh et al., 2022; Nichol & Dhariwal, 2021; Dhariwal & Nichol, 2021) than earlier generative models such as Generative Adversarial Networks (GANs) (Goodfellow et al., 2020; Karras et al., 2019), Normalizing Flows (NFs) (Dinh et al., 2016) and Variational Autoencoders (VAEs) (Kingma & Welling, 2013; Razavi et al., 2019). Beyond visual generation, diffusion models have shown remarkable success across a variety of tasks, including image editing (Hertz et al., 2022; Brooks et al., 2023; Kawar et al., 2023), 3D content generation (Poole et al., 2022; Wang et al., 2023a; Liu et al., 2023b; Wu et al., 2022; Wang et al., 2024a), and image restoration (Kawar et al., 2022; Chung et al., 2022; Wang et al., 2022b; Zhu et al., 2023; Delbracio & Milanfar, 2023; Lin et al., 2023), with particularly notable advancements in image super-resolution (SR) (Saharia et al., 2021; Chen et al., 2023; Yue et al., 2024b; Wang et al., 2024b).
> 💡 **动机批读**: 这段是背景铺垫：生成先验已经能带来高感知质量，SR 的问题不再只是“能不能生成细节”，而是如何在观测一致性、真实感和速度之间取舍。


Existing diffusion and flow-based SR methods can be broadly divided into two approaches: trainingfree methods (Zhu et al., 2023; Kawar et al., 2022; Wang et al., 2022b; Chung et al., 2022; Alkhouri et al., 2024; Mardani et al., 2023; Song et al., 2023a), and training-based methods (Saharia et al., 2021; Luo et al., 2023b; Liu et al., 2023a; Yue et al., 2023; Wang et al., $2 0 2 4 \mathrm { c }$ ; Yue et al., 2024b; Liu et al., 2024; Delbracio & Milanfar, 2023). Training-free methods decompose the conditional probability into a prior term and a likelihood term, with each term associating directly to a specific subproblem (Zhu et al., 2023). During iterative sampling, the prior subproblem is naturally handled by pre-trained unconditional diffusion models, which serve as powerful regularizers to guide the solution toward realistic High Resolution (HR) images. Meanwhile, the likelihood subproblem is addressed through specialized optimization techniques or analytical approximations to ensure fidelity to the observed Low Resolution (LR) image. On the other hand, training-based methods directly model the conditional probability using paired data, either by training from scratch (Saharia et al., 2021; Delbracio & Milanfar, 2023) or by incorporating additional control modules into existing generative priors (Wang et al., 2024b; Yu et al., 2024; Lin et al., 2023; Rombach et al., 2022). Several other bridge-based methods (Luo et al., 2023b; Liu et al., 2023a; Yue et al., 2023; Chung et al., 2024) have also been proposed for general image-to-image translation tasks, sharing similarities with direct learning approaches.
> 💡 **文献批读**: 作者把 prior+likelihood 的 training-free 路线和 paired-data training 路线并列，是为了引出本文的中间位置：训练一个条件 teacher，但仍保留生成轨迹的连续采样/控制语义。


![Figure 1](../images/9fc51ba765fa36d9fc9e5c103d757bcff5c74a738b1cabd535b1154ba2b9b062.jpg)
*Figure 1: Figure 1: (a) Our final model takes the concatenation of a low-resolution image with its noise-augmented version as input, and is able to generate high-resolution outputs with either high realism or high fidelity by adjusting the interpolation parameter $t$ . We indicate the PSNR and LPIPS value on the output images. (b) Comparison of different diffusion and flow based image super-resolution methods on the ImageNet $2 5 6 \times 2 5 6$ dataset. Bubble radius indicates the NFEs used by the methods.*
> 💡 **Figure 批读**: Fig. 1 同时画了机制和卖点：(a) $t$ 是推理时旋钮，PSNR/LPIPS 随输出风格变化；(b) bubble 半径对应 NFEs，OFTSR 想站在“1 step + 高感知质量”的左上区域。读图时要区分“可调”是模型真实能力还是只展示两个 cherry-picked 档位。


Despite the promising results of the above methods, they require many iterative sampling steps to achieve high perceptual quality, and reducing the number of iterations often results in higher fidelity but lower perceptual quality. In this sense, their fidelity-realism trade-offs are achieved at the cost of more sampling steps. In order to achieve high perceptual quality with fewer sampling steps, some attempts (Wang et al., 2024c; Lee et al., 2024; Wu et al., 2024; Xie et al., 2024; Li et al., 2024) have been made to distill the diffusion sampling process into a single step with diffusion distillation approaches (Luhman & Luhman, 2021; Salimans & Ho, 2022; Liu et al., 2022; Song et al., 2023b; Yan et al., 2024; Yin et al., 2024b;a; Sauer et al., 2025). However, while these methods improve efficiency, they sacrifice flexibility by limiting control over the fidelity-realism trade-off, reducing their applicability in domains where different tasks require varying levels of fidelity and realism, such as medical imaging, remote sensing and film upscaling (Greenspan, 2009; Li et al., 2023a; Wang et al., 2022a; Mentzer et al., 2020; Joshi et al., 2025).
> 💡 **问题定义**: 这里把本文的缺口说清楚了：多步数本身可以调 trade-off，但代价是慢；单步蒸馏快，但通常变成固定风格。OFTSR 要证明“可调”不必依赖多步 sampling。


In this paper, we propose OFTSR that achieves one-step image SR and preserves the capability to produce outputs with tunable fidelity-realism trade-offs. Specifically, OFTSR uses a two-stage pipeline. In stage one we train a noise-augmented conditional rectified flow to expand the support of the initial distribution: noise-perturbed LR images form the initial distribution while the LR images are used as conditions, enabling diverse HR reconstructions from a single LR. In the second stage, a distillation strategy is proposed to restrict the student model’s predictions to match the same Ordinary Differential Equation (ODE) induced by the teacher model from the first stage.
> 💡 **方法预告**: 两阶段各解决一个痛点：noise-augmented conditional flow 解决 LR 到 HR 的一对多支持集问题；ODE alignment distillation 解决把多步轨迹压成一步后仍能沿 $t$ 控制的问题。


Our main contributions can be summarized as follows:

• Noise-augmented Conditional Rectified Flow for Image Restoration: We introduce an enhanced conditional rectified flow model for image restoration. By leveraging a noiseaugmented LR conditioning strategy, our approach enables more effective LR-conditioned diffusion restoration, serving as both a general restoration framework and the foundational stage for our proposed distillation algorithm.
> 💡 **贡献批读**: 这不是简单给 LR 加噪声；关键是“噪声增强版本作初态，干净 LR 作条件”。一个负责多样性，一个负责不丢观测约束。


• One-Step Diffusion Distillation with Flexible Fidelity-Realism Trade-off: We introduce a distillation strategy applicable to empirical probability flow ODEs of any pre-trained conditional diffusion or flow model. Unlike prior methods that limit flexibility, ours enables one-step sampling while preserving control over fidelity and perceptual realism for SR.
> 💡 **贡献批读**: “any pre-trained conditional diffusion or flow model” 是一个很强的通用性 claim，后面必须看它蒸馏 ResShift、DiT4SR 的结果是否支持，而不是只在自训 teacher 上成立。


• State-of-the-Art (SOTA) Performance on Benchmark Datasets: Extensive experiments on DIV2K (Agustsson & Timofte, 2017), FFHQ (Karras et al., 2019), ImageNet (Deng et al., 2009) and several real world SR dataset including RealSR (Cai et al., 2019), RealSet80 (Yue et al., 2024b) and RealLQ250 (Ai et al., 2025) show that OFTSR achieves competitive one-step reconstruction, surpassing recent SOTA methods in both perceptual quality and fidelity.
> 💡 **证据批读**: 这类结果段不能只看“赢了”，还要看赢在哪个指标、哪个数据集、是否牺牲了另一侧 trade-off。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 任务 | Image super-resolution, including bicubic/noisy/real-world settings |
| 主矛盾 | 效率 vs 质量 vs 可控性 |
| 本文回答 | 先训练可调 conditional flow teacher，再用 ODE trajectory alignment 蒸馏 one-step student。 |

### 核心洞察
1. 本文不是否定 diffusion/flow SR，而是把它们的“多步可控”能力迁移到单步推理。
2. 噪声增强、LR condition、trajectory alignment、alignment/boundary loss 都应在消融中找到证据。
3. “可调”必须同时看数值曲线、视觉样例和 $t$ 越界/边界行为，不能只看一个最优档位。

### 可追问点
- OFTSR 和 diffusion one-step 的差别在哪里？
- 为什么 flow 适合可调 trade-off？
- 如果 $t$ 调大带来 hallucination，论文如何度量或限制？
