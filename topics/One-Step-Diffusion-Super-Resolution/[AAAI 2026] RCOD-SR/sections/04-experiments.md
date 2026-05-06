[← 返回 README](../README.md)

# Experiments

## 📌 预览
本节验证三个 claim 是否同时成立：RCOD 是否比 OSEDiff/S3Diff 等 OSD baseline 指标更好，是否保持 one-step 的延迟优势，是否能随 $t=250/500/750$ 单调调节 fidelity-realism。
---

> 💡 **Q&A 批注记录**:
> - Q: VPIM 解决什么痛点？
> - A: Real-ISR 的 text prompt 往往粗糙或外部生成，VPIM 把当前 LR 图像退化线索编码成 visual token，减少 prompt 与图像内容不匹配。
> - Q: 实验中怎么判断 RCOD 真的可控？
> - A: 不只看最好指标，还要看 Fid./Neu./Real. 三档是否随 timestep 形成稳定趋势：Fid. 更好 FR 指标，Real. 更好 NR/感知指标，Neu. 居中。


# Experiments Setups

Datasets: Following the training and testing settings of prior works(Wu et al. 2024b,a; Zhang et al. 2024), we employ LSDIR (Li et al. 2023) and the first 10K face images from FFHQ (Karras, Laine, and Aila 2019) for training and degradation pipeline of Real-ESRGAN (Wang et al. 2021) for LR image synthesizing. The synthetic test data use cropped $5 1 2 \times 5 1 2$ synthetic data from DIV2K-Val (Agustsson and Timofte 2017) and degraded using the Real-ESRGAN pipeline (Wang et al. 2021). The real-world data include LR-HR pairs from RealSR (Cai et al. 2019) and DRealSR (Wei et al. 2020), both with sizes of $1 2 8 \times 1 2 8 .$ - $5 1 2 \times 5 1 2$ for LR-HR pairs.
> 💡 **数据设置**: 训练使用 LSDIR + FFHQ，并用 Real-ESRGAN degradation pipeline 合成 LR；测试同时看 synthetic DIV2K-Val 和真实配对 RealSR/DRealSR。这里要注意：训练退化仍是合成 pipeline，真实泛化主要靠 RealSR/DRealSR 检验。


Evaluation Metrics: We employ widely used FR and NR metrics. FR metrics include PSNR, SSIM (Wang et al. 2004), LPIPS (Zhang et al. 2018a), and DISTS (Ding et al. 2020). NR metrics include NIQE (Zhang, Zhang, and Bovik
> 💡 **指标读法**: FR 指标需要 HR 参考，主要看保真/结构；NR 指标不需要参考，偏真实感和观感。RCOD 的 trade-off claim 必须用两类指标一起读，不能只按单一分数排序。


Table S2: Quantitative comparison with state-of-the-art methods on both synthetic and real-world benchmarks. The best and second best results are highlighted in bold and underline, respectively.
> 💡 **表格入口**: Table S2 是主结果表。读表顺序建议先看同一底座加 RCOD 前后变化，再看 Fid./Neu./Real. 三档趋势，最后才和 multi-step/GAN 方法横比。


![Table S2](../images/18e7075787af7a46cf69ef1bb61230009e0da77b644bb7e2ea833535d8f1ecc5.jpg)
*Table S2: Table S2: Quantitative comparison with state-of-the-art methods on both synthetic and real-world benchmarks. The best and second best results are highlighted in bold and underline, respectively.*
> 💡 **Table 批读**: 这张表的关键不是找到“唯一最好模型”，而是看 RCOD 是否把一个底座展开成多个可选工作点。若 Real. 档 NR 提升但 PSNR/SSIM 下降，这是预期 trade-off；若三档没有顺序，LDG 的控制 claim 就站不稳。


2015), MANIQA-pipal (Yang et al. 2022), MUSIQ (Ke et al.   
2021), and CLIPIQA (Wang, Chan, and Loy 2023).
> 💡 **指标补充**: MANIQA/MUSIQ/CLIPIQA 高通常说明图像更“像自然图”，但不保证细节真实来自输入；因此 Fig. S5/S6 的局部视觉检查很重要。


Method Comparison: We compare our method with stateof-the-art methods (SOTA) in three categories: multi-step diffusion Real-ISR methods, including StableSR (Wang et al. 2024a), ResShift (Yue, Wang, and Loy 2023), Diff-BIR (Lin et al. 2023), and SeeSR (Wu et al. 2024b); onestep diffusion Real-ISR methods, such as SinSR (Wang et al. 2024b), OSEDiff (Wu et al. 2024a), S3Diff (Zhang et al. 2024), TSD-SR (Dong et al. 2025), PiSA-SR (default version) (Sun et al. 2025), and InvSR (Yue, Liao, and Loy 2025); and GAN-based methods, including BSR-GAN (Zhang et al. 2021) and Real-ESRGAN (Wang et al. 2021). We quantitatively compare recent SOTA OSD methods in Table S2 on real-world data. More qualitative comparisons and the full table including synthetic data, multi-step diffusion, and GAN-based methods are in SM.
> 💡 **Baseline 批读**: baseline 覆盖 multi-step、one-step、GAN 三类，其中最该盯的是 OSEDiff 和 S3Diff，因为 RCOD 直接套在它们上面；PiSA-SR/OFTSR 则是可调 trade-off 的近邻对照。


# Implement Details

Model Setting: To verify our framework, we select two types of recent SOTA OSD Real-ISR methods: OSEDiff (Wu et al. 2024a), which is distilled from a pre-trained multi-step Stable Diffusion (SD) model, and S3Diff (Zhang et al. 2024), which directly uses SD-T (a distilled version of SD), as our base models. When RCOD is applied to them, they are named $\mathrm { R C O D _ { O } }$ and $\mathrm { R C O D } _ { \mathrm { S } }$ , respectively. The choice of $n = 3$ corresponds to three distinct generation levels in the inference stage: Fidelity, Neutral, and Realism. i.e, $t \ : = \ : 2 5 0$ , 500, and 750 during inference. Since the S3Diff is not directly distilled from a multi-step diffusion, we do not apply DAS on it. ${ \mathrm { R C O D } } _ { \mathrm { S } }$ -Adap. employs MEM during inference. The input to the MEM in this case is $z _ { L }$ and the features in the last layer degradation estimation model used in (Zhang et al. 2024). The MEM is trained after $\mathrm { R C O D } _ { \mathrm { S } }$ training, utilizing the same training data. More details please refer to SM.
> 💡 **实现批读**: 这里有两个细节。第一，$n=3$ 对应 Fid./Neu./Real. 三档，推理 timestep 是 250/500/750。第二，DAS 只用于 OSEDiff 路线，因为 S3Diff 直接用 SD-Turbo，不是同样的 multi-step teacher 蒸馏设定；这会影响消融解读。


# Comparison with State-of-the-Arts
> 💡 **小节预览**: 实验要同时看三件事：质量指标是否赢、速度是否保持单步优势、可控/消融是否支撑核心 claim。


Quantitative Comparisons: We can observe in Table S2: $i$ ) By applying RCOD, on each dataset, the “-Fid.” versions $( t = 2 5 0$ ) have better full-reference (FR) metrics such as PSNR, SSIM, LPIPS, and FID, while keeping the noreference (NR) metrics relatively ordinary. In contrast, the “- Real.” versions $t = 7 5 0$ ) achieve obviously higher NR metrics like MANIQA, MUSIQ, and CLIPIQA. Most “-Neu.” versions $( t = 5 0 0 )$ ) fall within the middle range of the previous two versions. This illustrates that we can effectively and simply control the realism level (usually measured by perceptual NR metrics) during the inference stage, and that the realism level increases monotonically with the timestep. $i i \ "$ ) $\mathrm { R C O D } _ { \mathrm { S } }$ -Adap. has relatively balanced metrics between $\mathrm { R C O D } _ { \mathrm { S } }$ -Fid. and ${ \mathrm { R C O D } } _ { \mathrm { S } }$ -Neu.. This indicates that most estimated $M _ { L }$ values are closer to 1 than to 0. This roughly matches the cosine similarity value distributions in Fig. S4 (b). iii) When applying RCOD, $\mathrm { R C O D _ { O } }$ -Fid. and $\operatorname { R C O D } _ { \operatorname { S } }$ - Adap. perform better than their original methods (OSEDiff and S3Diff, respectively) on most metrics, including PSNR, SSIM, LPIPS, MANIQA, and MUSIQ. Even with a preference for fidelity in FR metrics, $\mathrm { R C O D _ { O } }$ -Fid. shows superior performance on some NR metrics, such as MANIQA and MUSIQ, compared to the original OSEDiff method on realworld data. Additionally, ${ \mathrm { R C O D } } _ { \mathrm { S } }$ -Real. usually achieves the best NR metrics (NIQE and CLIPIQA). iv) S3Diff shows better performance on the perceptual quality metric DISTS. This may arise from the negative online prompting (NOP) used in training, which provides a more accurate concept of high quality. However, since the text encoder and text prompt are replaced by VPIM in our $\mathrm { R C O D } _ { \mathrm { S } }$ , the NOP is also removed. Despite this, our ${ \mathrm { R C O D } } _ { \mathrm { S } }$ -Adap. performs better on other NR metrics.
> 💡 **结果批读**: 作者声称三档趋势基本成立：Fid. 更偏 FR，Real. 更偏 NR，Neu. 介于中间。这里最有价值的是“同一模型同一输入只改 $t$”带来的工作点变化；但也要注意 DISTS 上 S3Diff 仍强，说明 VPIM 去掉 NOP 后并非所有感知指标都占优。


![Figure S5](../images/72d8efd35cce6ef0c1bfc4449499b8142caa4f5ffa766b45c5845a8f55c870b8.jpg)
*Figure S5: Figure S5: Visual comparison $( \times 4 )$ of $\mathrm { R C O D _ { O } }$ -Real. with other methods on DRealSR data.*
> 💡 **Figure 批读**: Fig. S5 用 Real. 档展示更丰富纹理。批读时要放大看栏杆、文字、皮肤/织物等区域：纹理更自然是收益，但若结构边界被改写，就是 realism 档的风险。


Time Efficiency: In Table S3, we compare inference time and trainable parameters. All methods are tested on an A100 GPU with a $5 1 2 \times 5 1 2$ input image. RCOD keeps similar time efficiency and trainable parameters as the original methods while have higher PSNR and MANIQA. $\mathrm { R C O D _ { O } }$ even inferences faster as the text extractor is removed.
> 💡 **效率批读**: 这是 RCOD 相对 PiSA-SR/多步方法的重要卖点：仍是 one-step，参数量接近底座；RCOD_O 去掉 text extractor 后甚至更快。需要注意这里是 A100 上 512 输入的测量，移动端真实延迟还要另测。


Qualitative Comparisons: Fig. S5 compares the visual qualities of different Real-ISR methods. $\mathrm { R C O D _ { O } }$ -Real. demonstrates its ability to recover more detailed and natural textures. Fig. S6 illustrates the changes in visual effects as $t$ increases, i.e, from $\mathrm { R C O D } _ { \mathrm { S } }$ -Fid. $( t ~ = ~ 2 5 0 )$ t o ${ \mathrm { R C O D } } _ { \mathrm { S } }$ -Real. $( t ~ = ~ 7 5 0 )$ . As $t$ increases during the inference stage, more skin texture and wrinkles are recovered. ${ \mathrm { R C O D } } _ { \mathrm { S } }$ -Adap. chooses a proper $t = 5 0 0$ in this case, where the $M _ { L }$ range is [0.5, 0.75) according to Eq. 5. The $M _ { L }$ of the LR image (0.621) falls within this range.
> 💡 **可控性批读**: Fig. S6 是控制 claim 的视觉证据：$t$ 从 250 到 750，皮肤纹理/皱纹等生成细节增加。Adap. 选到 $t=500$，说明 MEM 的自动档落在中间，但这也提示用户手动档和自动档需要有清晰产品语义。


Ablation Study We performed a series of ablation studies of the RCOD framework, the details can be found in SM.
> 💡 **消融入口**: 正文只提示消融在附录，真正要去 Table S5/S6 看 VPIM、DAS 和 group 数量。没有这些消融，主结果很难说明提升来自 RCOD 设计而不是底座/训练波动。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 读表顺序 | 主结果 → 消融 → 效率 → 视觉案例 |
| 核心检查 | fidelity 与 perceptual quality 是否同时合理 |
| 风险 | no-reference 指标提升不等于结构真实 |

### 核心洞察
1. 证明 RCOD 有效的核心不是单点 SOTA，而是同一底座在不同 $t$ 下形成可解释工作点。
2. Fid./Neu./Real. 三档分别对应保真、折中、真实感，指标趋势比绝对排名更能支撑 LDG。
3. 速度表说明 RCOD 没有破坏 one-step 优势；视觉图则需要检查 realism 档是否引入幻觉细节。

### 可追问点
- 为什么单步 SR 难以控制 realism？
- LDG 真正控制的是什么？
- RCOD 的 Real. 档在哪些区域最容易 hallucinate？
- MEM 自动选档和用户手动选档如何取舍？
