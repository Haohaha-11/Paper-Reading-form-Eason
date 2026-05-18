# ELF: Embedded Language Flows

## 1. Paper Metadata

- **Title**: ELF: Embedded Language Flows
- **Authors**: Keya Hu\*, Linlu Qiu\*, Yiyang Lu, Hanhong Zhao, Tianhong Li, Yoon Kim, Jacob Andreas, Kaiming He (\*Equal contribution; order decided by coin flip)
- **Affiliation**: MIT
- **Venue**: arXiv 2026 (2026/05/11)
- **ArXiv ID**: 2605.10938
- **Code**: https://github.com/lillian039/ELF
- **Connected Papers**: https://www.connectedpapers.com/main/2605.10938

## 2. One-Sentence Summary

ELF formulates language generation as continuous-time Flow Matching entirely in the embedding space, applying discrete token mapping only at the final time step, which enables straightforward adoption of classifier-free guidance and achieves SOTA generation quality with fewer sampling steps and training tokens.

## 3. Core Contributions

1. **Continuous-time Flow Matching for language**: Formulates DLMs in continuous embedding space with Flow Matching (rectified flows), using x-prediction parameterization for high-dimensional representations.

2. **Single-step discretization**: Unlike prior DLMs that apply per-step token-level cross-entropy supervision along the flow trajectory, ELF stays continuous until the final time step, using a shared-weight network that serves as both denoiser (MSE loss) and decoder (CE loss).

3. **Training-time Classifier-Free Guidance (CFG)**: Adapts training-time CFG from image generation to language, combining it with self-conditioning -- naturally applicable because ELF operates in continuous space.

4. **Minimalist design**: No separate decoder, no per-step rounding/CE, no simplex constraints. Uses frozen pretrained T5 contextual embeddings with a bottleneck projection.

5. **Strong empirical results**: Outperforms leading discrete DLMs (MDLM, Duo) and continuous DLMs (FLM, LangFlow, DFM) with 10x fewer training tokens (45B vs 500B+), achieving Gen. PPL of 24 with only 32 sampling steps.

## 4. Section Navigation

| Section | File | Description |
|---------|------|-------------|
| Abstract | [sections/00-abstract.md](sections/00-abstract.md) | Paper abstract |
| 1. Introduction | [sections/01-introduction.md](sections/01-introduction.md) | Motivation and high-level approach |
| 2. Background & Related Work | [sections/02-background-related-work.md](sections/02-background-related-work.md) | Diffusion/Flow models, continuous & discrete DLMs |
| 3. Embedded Language Flows | [sections/03-methodology.md](sections/03-methodology.md) | Framework (3.1), Pseudocode (3.2), Conditioning & Guidance (3.3) |
| 4. Experiments | [sections/04-experiments.md](sections/04-experiments.md) | Ablations (4.1), Unconditional (4.2), Conditional (4.3) |
| 5. Conclusion + Appendix | [sections/05-conclusion.md](sections/05-conclusion.md) | Conclusion + Appendix highlights |

## 5. Key Numbers

| Metric | Value |
|--------|-------|
| ELF Model Sizes | ELF-B (105M), ELF-M (342M), ELF-L (652M) |
| Encoder | Frozen T5-small (35M), embedding dim 512, bottleneck 128 |
| Embedding Dimension | 512 (from T5-small); bottleneck projects to 128 |
| Training Dataset (Uncond) | OpenWebText (~9B tokens, 5 epochs = 45B effective tokens) |
| Sequence Length | L=1024 (uncond), L=128 (MT), L=1088 (XSum) |
| Best Gen. PPL (ELF-B) | 24.08 at 32 SDE steps (CFG=3, $\gamma$=1.5) |
| Baseline Gen. PPL (MDLM) | ~23 at 1000+ steps |
| WMT14 De-En BLEU | 26.4 (ELF-B, 64-step ODE) |
| XSum ROUGE-L | 27.8 (ELF-B, 64-step ODE) |
| Training Optimizer | Muon, LR=0.002, batch 512 |
| Sampling | ODE (deterministic) or SDE-inspired (stochastic); logit-normal time schedule |
| CFG Scale | 3 (uncond SDE), 2 (cond ODE) |
| Denoising:Decoding Ratio | 80% MSE : 20% CE during training |

## 6. Data Flow: Input -> Intermediate -> Output

```
Input: Discrete tokens s = [s_1, ..., s_L]
    |
    v
Encoder: Frozen T5-small -> contextual embeddings x (512-dim per token)
    |
    v
Bottleneck: Linear projection 512 -> 128 -> model hidden 768 (optional compression)
    |
    v
Training (two branches, shared-weight network net_θ):
  Branch 1 - Denoising (80%): z_t = t*x + (1-t)*ε, predict x̂ → MSE loss L_MSE
  Branch 2 - Decoding (20%): z̃ = p*x + (1-p)*ε at t=1, predict x̂ → W·x̂ → CE loss L_CE
    |
    v
Self-Conditioning: Concatenate [z_t, x̂'] → project back → condition on previous prediction
    |
    v
In-Context Conditioning: Prepend time tokens [0,1], CFG-scale tokens [0.5,5], mode tokens (denoise/decode)
    |
    v
Training-Time CFG: Single-pass modeling of v_cfg = ω·v_sc + (1-ω)·v_no_sc
    |
    v
Inference: z_0 ~ N(0,I) → ODE/SDE solver for N steps → z → decode mode at t=1 → W·net_θ(z,t=1) → argmax → tokens
```

## 7. Pros/Cons & Future Work

### Strengths
- **Simplicity**: Minimal discretization (only at t=1), shared-weight denoiser-decoder, no extra decoder training.
- **Efficiency**: 10x fewer training tokens than prior DLMs; 32 SDE steps achieve strong results (vs 1000+ for discrete DLMs).
- **CFG compatibility**: Continuous formulation makes CFG naturally applicable; training-time CFG avoids inference overhead.
- **Scalability**: Consistent improvements with larger models (B, M, L).
- **Versatility**: Works for both unconditional and conditional (MT, summarization) generation.
- **SDE sampler**: Re-injecting noise during sampling effectively corrects early errors in few-step regime.

### Limitations
- **Encoder dependency**: Requires a pretrained encoder (T5) for contextual embeddings; performance drops with non-contextual or random embeddings.
- **CFG sensitivity**: Self-conditioning CFG scales beyond 3-4 degrade performance for smaller models.
- **Bottleneck tuning**: Bottleneck dimension matters significantly; 128 found optimal for 512-dim embeddings.
- **Limited conditional tasks**: Only evaluated on MT and summarization; broader NLG tasks untested.
- **Compute**: Uses TPU v5p x 64; ~1.5h per epoch on OWT.

### Future Work
- Scaling to larger models and datasets beyond 652M parameters.
- Exploring other embedding sources beyond T5 encoders.
- Extending to code generation, multimodal generation, and other conditional tasks.
- Investigating distilled/few-step variants for faster inference.
- Studying the theoretical properties of continuous embedding space flow dynamics.

## 8. Reading Q&A Record

**Q1: Why does ELF use x-prediction instead of v-prediction or ε-prediction?**

> **Answer**: x-prediction is essential for three reasons. (1) It enables the shared-weight design between denoising (MSE) and decoding (CE) since both predict clean embeddings. v-prediction compromises this weight sharing when converted to x. (2) High-dimensional clean data lies on a low-dimensional manifold [32], making x-prediction more stable -- v-prediction degrades at higher dimensions (768, 1024), while ε-prediction collapses entirely. (3) Empirically, x-prediction maintains a stable Gen. PPL-entropy trade-off across all embedding dimensions. See Appendix C.1 and Fig. 10.

**Q2: How does ELF differ from latent diffusion LMs (LD4LG, TEncDM, Cosmos)?**

> **Answer**: Key differences: (1) ELF uses Flow Matching (continuous-time) vs DDPM/score-based (discrete-time) in latent diffusion LMs. (2) ELF has no separate decoder -- the shared-weight network handles both denoising and decoding. Latent diffusion LMs require separately trained autoregressive or non-autoregressive decoders. (3) ELF directly operates in high-dimensional embedding space (768 hidden dims) rather than a compressed latent bottleneck. See Section 2 and Tab. 2 in Appendix A.

**Q3: How does the SDE-inspired sampler work and why is it better in the few-step regime?**

> **Answer**: The SDE sampler re-injects Gaussian noise at each sampling step while shifting the time variable backward toward the noise regime, controlled by γ. Specifically: α = 1 - γ*dt, t_back = α*t, z_back = α*z + (1-α)*ε. The denoiser is evaluated on this perturbed state, and the clean-embedding prediction updates the original state. This stochastic correction prevents error accumulation from imperfect trajectories (which ODE deterministically amplifies). See Sec. 3.2 and Appendix B.2, Algorithm 6.

**Q4: Why is the bottleneck dimension 128 optimal for 512-dim embeddings?**

> **Answer**: This is consistent with the manifold hypothesis in [32]: natural language embeddings lie on a low-dimensional manifold within the 512-dim space. A bottleneck of 128 provides the right compression -- 32 is too restrictive (loses diversity → low entropy), while 512 is too high-dimensional (suffers from poor Gen. PPL). The 128-dim bottleneck achieves the best Gen. PPL-entropy balance under both ODE and SDE sampling. See Fig. 11.

**Q5: How does training-time CFG differ from inference-time CFG and why does ELF use the former?**

> **Answer**: Training-time CFG [16, 17] trains the network to directly model v_cfg (the post-combination velocity) instead of v (the pre-combination velocity). The regression target becomes: v_target = x - ε + (1 - 1/ω)(v_sc - v_no_sc). At inference time, no second forward pass is needed -- the CFG scale ω is provided as an input token to the network. This avoids the 2x inference cost of standard CFG. ELF adopts this because its continuous formulation is structurally identical to image diffusion models where training-time CFG was developed.

## 9. Citation Landscape

Connected Papers: https://www.connectedpapers.com/main/2605.10938

### Upstream (foundational):
- Diffusion models: DDPM [Ho et al. 2020], Score-SDE [Song et al. 2021], D3PM [Austin et al. 2021]
- Flow Matching: [Lipman et al. 2023], Rectified Flow [Liu et al. 2023], Stochastic Interpolants [Albergo et al. 2025]
- x-prediction parameterization: [Li & He 2025]
- CFG: [Ho & Salimans 2021], Training-time CFG [Geng et al. 2025]

### Peer (contemporary DLMs):
- Discrete: MDLM [Sahoo et al. 2024], Duo [Sahoo et al. 2025], E2D2 [Arriola et al. 2025]
- Continuous: Diffusion-LM [Li et al. 2022], CDCD [Dieleman et al. 2022], DiffuSeq [Gong et al. 2023]
- Flow-based concurrent: FLM [Lee et al. 2026], LangFlow [Chen et al. 2026], DFM [Potaptchik et al. 2026], CFM [Roos et al. 2026]
- Latent diffusion: LD4LG [Lovelace et al. 2023], TEncDM [Shabalin et al. 2025], Cosmos [Meshchaninov et al. 2025]

### Downstream (extensions):
- Discrete scaling: LLaDA [Nie et al. 2025], Dream 7B [Ye et al. 2025], Seed Diffusion [Song et al. 2025]
- Multimodal: Omni-Diffusion [Li et al. 2026], MMADA [Yang et al. 2025], LLaDA-V [You et al. 2025]
- Fast inference: Fast-DLLM [Wu et al. 2026], Remasking [Wang et al. 2025]
