# One-Step Diffusion Super-Resolution

本课题聚焦 **one-step diffusion / flow based image super-resolution**：在尽量保留扩散/生成模型感知质量的同时，把推理压缩到单步，并进一步解决可控性、保真度和实时部署问题。

## 论文列表

| 论文 | 会议/年份 | 核心问题 | 方法特点 |
|------|-----------|----------|----------|
| [OSEDiff](./%5BNeurIPS%202024%5D%20OSEDiff/) | NeurIPS 2024 | 多步 SD-based Real-ISR 慢且随机 | LQ latent 直接作为起点 + latent-space VSD，单步 Real-ISR 基线 |
| [OFTSR](./%5BICLR%202026%5D%20OFTSR/) | ICLR 2026 | 常规蒸馏固定 fidelity-realism trade-off | conditional flow teacher + ODE trajectory alignment distillation |
| [TinySR](./%5BArxiv%202025%5D%20TinySR/) | arXiv 2025 | one-step 模型仍过大、实时性不足 | depth pruning + VAE compression + prompt/time module removal + cache |
| [RCOD-SR](./%5BAAAI%202026%5D%20RCOD-SR/) | AAAI 2026 | 单步模型缺少 realism 控制 | latent domain grouping + degradation-aware sampling distillation + visual prompt |
| [CODSR](./%5BArxiv%202025%5D%20CODSR/) | arXiv 2025 | LQ 压缩信息损失、区域先验激活不足 | LQ-guided feature modulation + region-adaptive prior activation + text matching |

## 推荐阅读顺序

1. **OSEDiff**：先理解 one-step diffusion Real-ISR 的基本范式。
2. **OFTSR**：再看 flow/ODE 视角如何保留可调 fidelity-realism trade-off。
3. **RCOD-SR**：重点读单步模型如何恢复多步方法中的 realism 控制能力。
4. **CODSR**：关注区域级控制和 LQ 原始信息如何补保真。
5. **TinySR**：最后看系统压缩与实时部署，把“单步”推进到“轻量实时”。

## 方法谱系

```text
Multi-step diffusion SR
        │
        ├── OSEDiff: one-step distillation baseline
        │       └── gap: fixed output / heavy SD backbone / fidelity-realism trade-off hard to control
        │
        ├── RCOD-SR: global realism control via latent grouping
        ├── CODSR: region-adaptive control + LQ feature fidelity compensation
        ├── TinySR: architecture/VAE/prompt streamlining for real-time inference
        └── OFTSR: flow teacher + trajectory alignment for tunable trade-off
```

## 横向比较

| 维度 | OSEDiff | OFTSR | TinySR | RCOD-SR | CODSR |
|------|---------|-------|--------|---------|-------|
| 基础范式 | SD one-step diffusion | conditional flow / ODE | pruned one-step diffusion | controllable one-step diffusion | controllable one-step diffusion |
| 主要优化目标 | 单步有效性 | 可调 fidelity-realism | 实时轻量化 | 推理时 realism 控制 | 区域级保真与先验激活 |
| 核心控制量 | LQ latent 起点 + VSD | teacher ODE trajectory | pruning ratio / cached modules | latent domain group / timestep degree | region activation + text matching |
| 解决的痛点 | 多步慢、随机性 | fixed trade-off | 模型过大 | 单步不可控 | LQ 信息损失和局部错配 |
| 适合关注 | baseline | distillation objective | deployment | controllability | fidelity-realism localization |

## 可复用 Insight

- **单步不是终点**：OSEDiff/SinSR 之后，研究问题转向“单步但可控”“单步但轻量”“单步但保真”。
- **fidelity-realism 是主轴**：这组论文几乎都在不同层面处理 perception-distortion trade-off。
- **控制可以全局也可以局部**：RCOD-SR 偏全局/退化强度控制，CODSR 偏区域级先验激活。
- **蒸馏目标很关键**：OSEDiff 用 VSD，OFTSR 用 ODE trajectory alignment，二者体现 diffusion 与 flow 两条路线。
- **部署要看系统瓶颈**：TinySR 说明采样步数降到 1 之后，backbone、VAE、prompt/time 模块仍可能是主要开销。

## 待读问题

- one-step SR 的可控性是否能统一成一个连续 control parameter，而不是方法各自定义？
- fidelity-realism trade-off 是否应该区分全局图像风格、局部纹理、身份/文字等不同维度？
- flow-based one-step SR 和 diffusion-distilled one-step SR 在真实复杂退化上的泛化差异是什么？
- 轻量化模型是否会削弱对 out-of-distribution degradation 的鲁棒性？
