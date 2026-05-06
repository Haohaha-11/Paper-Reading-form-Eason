[← 返回 README](../README.md)

# 5 CONCLUSION

## 📌 预览
结论很短，只重申效率与质量；真正的 limitation 在附录 C 和 D：student 受 teacher 上限约束，distilled curve 与 teacher frontier 存在轻微 $t$ shift。
---

> 💡 **Q&A 批注记录**:
> - Q: OFTSR 和 diffusion one-step 的差别在哪里？
> - A: 它把 teacher 的 PF-ODE 轨迹关系蒸馏给 student，使 one-step 模型仍能用 $t$ 选择 fidelity-realism 档位。


In this paper, we introduced OFTSR, a novel approach to developing efficient one-step image superresolution models. Our extensive experiments on FFHQ, DIV2K, ImageNet and real world SR datasets demonstrate that our method significantly improves computational efficiency while maintaining high-quality image restoration capabilities. The proposed framework represents a promising direction in efficient image SR, effectively addressing the perception-distortion trade-off.
> 💡 **结论批读**: 结论把“addressing the trade-off”说得较强，准确理解应是“提供可调路径”，不是消除 perception-distortion 下界。未来问题应落到 teacher 上限、$t$ 校准和真实退化泛化。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 本节作用 | 收束主 claim |
| 主线 | one-step SR + tunable fidelity-realism |
| 后续关注 | teacher 上限、$t$ 校准、真实退化、hallucination 风险 |

### 核心洞察
1. 论文贡献集中在蒸馏可调轨迹，而不是重新定义 SR 评价。
2. 结论没有展开失败案例，必须回附录看 $t$ 越界、frontier shift 和 teacher limitation。
3. 部署时不能只选 $t=1$，要按任务决定是否允许更强生成纹理。

### 可追问点
- OFTSR 和 diffusion one-step 的差别在哪里？
- 为什么 flow 适合可调 trade-off？
- 真实应用应该如何选择 $t$ 才能避免过度 hallucination？
