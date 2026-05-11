[← 返回 README](../README.md)

# 02 - Cyber

## 📌 预览

Cyber 是 Mythos Preview 限制发布的最直接原因。报告把它描述为 Anthropic 已发布模型中 cyber 能力最强的模型，并明确承认这些能力具有防御/进攻双用性。

## 主要结论

报告称 Claude Mythos Preview 在内部 cyber evaluation suite 上超过所有先前模型，几乎 saturating 现有内部和已知外部 capability evaluations。Anthropic 因此把评估哲学从 CTF-style static benchmark 转向 real-world cybersecurity tasks。

最关键描述是：在 minimal human steering 的 agentic harness 中，模型能够在授权披露项目或安排下，自主发现 open-source 和 closed-source software 的 zero-day vulnerabilities，并在许多情况下发展为 working proof-of-concept exploits。

> 💡 **部署批注**: 这解释了为什么模型不是一般开放。漏洞发现和 exploit development 对防御者有巨大价值，但如果广泛开放也会加速 offensive exploitation。

## 缓解措施

Anthropic 的 cyber misuse mitigation 依赖两类机制：

- **Restricted access**: 只给 carefully vetted partners，优先用于 Project Glasswing 等防御性用途。
- **Probe classifiers / monitoring**: 监控 prohibited use、high risk dual use、dual use 三类潜在 misuse。

报告特别说明，由于本次 release 非常有限且目标是 trusted cyber defenders，他们不会仅因 classifier trigger 就 block exchange；但对未来 general-release 强 cyber 模型，计划 block prohibited use，并在许多或多数情况下 block high-risk dual-use prompts。

## Benchmark 结果

| 评估 | 结果 | 批读 |
|---|---:|---|
| Cybench subset | 100% pass@1 | 35 challenges, 10 trials/challenge；报告认为此类 CTF benchmark 已不够区分 frontier capability |
| CyberGym | 0.83 | Opus 4.6 为 0.67，Sonnet 4.6 为 0.65；测试真实开源项目中已知漏洞的 targeted reproduction |
| Firefox 147 exploitation | Mythos 明显领先 | 给定 50 crash categories、250 trials，任务是发展 exploit 读取并复制 secret；Opus 4.6 之前只有数百次中 2 次能开发 exploit |

> 💡 **Firefox 147 批注**: 这个评估比 CTF 更接近真实攻防链路：模型要 triage crash、识别 exploitable primitive、再构造 proof-of-concept。报告还做了去除 top-2 bugs 的 variant，说明不是单一漏洞记忆即可解释。

## 小结

- Cyber 章节支撑了“restricted partner deployment”的核心论证。
- Mythos Preview 的能力不是只会回答安全知识题，而是具备 agentic vulnerability discovery/exploitation 能力。
- 报告也承认现有 CTF benchmark 接近饱和，未来评估需要更贴近真实软件和真实防御场景。
