[← 返回 README](../README.md)

# ABSTRACT

## 📌 预览
摘要给出全文最短链路：多步 diffusion/flow SR 质量高但慢，普通 one-step distillation 快但 trade-off 固定；OFTSR 用 conditional flow teacher + ODE trajectory alignment distillation，把可调轨迹压到 one-step student。
---

> 💡 **Q&A 批注记录**:
> - Q: OFTSR 和 diffusion one-step 的差别在哪里？
> - A: OFTSR 的监督不是只让 student 复现 teacher 终点，而是让 student 在不同 $t$ 的单步预测满足 teacher PF-ODE 的局部轨迹关系。


# ONE-STEP FLOW FOR IMAGE SUPER-RESOLUTION WITH TUNABLE FIDELITY-REALISM TRADE-OFFS
> 💡 **标题批读**: 题目里四个关键词都要保留：one-step 是效率目标，flow 是建模形式，SR 是任务边界，tunable fidelity-realism 是相对其他单步方法的差异点。


Yuanzhi ${ \bf Z } { \bf h } { \bf u } ^ { \ast \dagger { 1 } , 2 }$ Ruiqing Wang\*1 Shilin ${ \bf L } { \bf u } ^ { 3 }$ Junnan Li2 Hanshu Yan2 Kai Zhang‡1 1Nanjing University 2Rhymes.AI 3Nanyang Technological University
> 💡 **阅读定位**: 作者信息本身不贡献技术论证，但 Kai Zhang 组在图像恢复/SR 上积累很深，后面实验要特别看与 ResShift、Real-ESRGAN 退化管线等既有 SR 体系的关系。


# ABSTRACT

Recent advances in diffusion and flow-based generative models have demonstrated remarkable success in image restoration tasks, achieving superior perceptual quality compared to traditional deep learning approaches. However, these methods either require numerous sampling steps to generate high-quality images, resulting in significant computational overhead, or rely on common model distillation, which usually imposes a fixed fidelity-realism trade-off and thus lacks flexibility. In this paper, we introduce OFTSR, a novel flow-based framework for one-step image super-resolution that can produce outputs with tunable levels of fidelity and realism. Our approach first trains a conditional flow-based superresolution model to serve as a teacher model. We then distill this teacher model by applying a specialized constraint. Specifically, we force the predictions from our one-step student model for same input to lie on the same sampling ODE trajectory of the teacher model. This alignment ensures that the student model’s singlestep predictions from initial states match the teacher’s predictions from a closer intermediate state. Through extensive experiments on datasets including FFHQ $( 2 5 6 \times 2 5 6 )$ , DIV2K, ImageNet $( 2 5 6 \times 2 5 6 )$ ) and real world SR datasets, we demonstrate that OFTSR achieves state-of-the-art performance for one-step image superresolution, while having the ability to flexibly tune the fidelity-realism tradeoff. Code and pre-trained models are available at https://github.com/yuanzhizhu/OFTSR and https://huggingface.co/Yuanzhi/OFTSR, respectively.
> 💡 **摘要批读**: 摘要的关键不是“又一个单步 SR”，而是把两个通常冲突的目标绑在一起：1 NFE 的速度，以及由 $t$ 控制的 fidelity-realism 曲线。最需要后文验证的是：student 是否真的继承 teacher 的轨迹结构，而不是只学到一个固定风格的锐化器。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 一句话 | conditional flow teacher 负责生成可调轨迹，ODE trajectory alignment 负责把这条轨迹压给 one-step student。 |
| 输入 | LR condition + noise-augmented LR initial state + trade-off parameter $t$ |
| 输出 | 单步 SR image；小 $t$ 偏 fidelity，大 $t$ 偏 realism |

### 核心洞察
1. OFTSR 的 novelty 主要在“单步还可调”，不是单步本身。
2. conditional flow teacher 的质量决定 student 上限，distillation loss 决定可调轨迹能保留多少。
3. 摘要提到 real-world SR，但泛化强弱要看退化设置、teacher 来源和 no-reference 指标。

### 可追问点
- OFTSR 和 diffusion one-step 的差别在哪里？
- 为什么 flow 适合可调 trade-off？
- teacher trajectory 与 student 单步输出之间的 gap 有多大？
