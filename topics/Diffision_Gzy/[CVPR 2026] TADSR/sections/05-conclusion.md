[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览
结论强调 TADSR 的核心不是更复杂采样，而是让 one-step student 使用可变 timestep，并用 TAE/TAVSD 让时间条件真的生效。


In this paper, we propose TADSR, a one-step SD-based Real-ISR method. TADSR introduces a variable timestep $t _ { s }$ into the student model and uses a Time-Aware VAE Encoder to fully utilize the generative priors in SD at different timesteps. To further distill the priors at different timesteps in SD to achieve varied SR effects, TADSR leverages the Time-Aware Variational Score Distillation to provide more consistent generative guidance condition on $t _ { s }$ . As a result, TADSR fully leverages the generative priors in SD and naturally achieves a controllable trade-off between fidelity and realism condition on $t _ { s }$ . Our experiments demonstrate that TADSR achieves state-of-the-art performance among all Real-ISR methods.


---

## 🔖 Section 总结
- 结论强调可变 timestep 是 TADSR 的核心控制变量。
- TAE/TAVSD 让单步模型既可控又保持效率。
- 可追问：如果部署时退化很轻，是否应自动选择小 timestep？
