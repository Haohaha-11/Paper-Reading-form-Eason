[← 返回 README](../README.md)

# 07. Appendix

## A. Contributions

The authors are listed in order of the significance of their contributions, with those in project leadership roles appearing last.

Guangyu Chen*  
Yu Zhang*  
Jianlin Su*  
Weixin Xu  
Siyuan Pan  
Yaoyu Wang  
Yucheng Wang  
Guanduo Chen  
Bohong Yin  
Yutian Chen  
Junjie Yan  
Ming Wei  
Y. Zhang  
Fanqing Meng  
Chao Hong  
Xiaotong Xie  
Shaowei Liu  
Enzhe Lu  
Yunpeng Tai  
Yanru Chen  
Xin Men  
Haiqing Guo  
Y. Charles  
Haoyu Lu  
Lin Sui  
Jinguo Zhu  
Zaida Zhou  
Weiran He  
Weixiao Huang  
Xinran Xu  
Yuzhi Wang  
Guokun Lai  
Yulun Du  
Yuxin Wu  
Zhilin Yang  
Xinyu Zhou

> 💡 **作者列表读法**: 技术报告明确说明按贡献显著性排序，项目领导角色靠后。这对追踪后续代码、复现或相关技术路线有用。

## B. Optimized Inference I/O for Full Attention Residuals

A naïve implementation of Full AttnRes scans all preceding layer outputs at every layer, so memory traffic scales linearly with depth. As noted in §4.2, however, the pseudo-query $\boldsymbol w_l$ is a learned parameter independent of both the input and the hidden state. We can therefore batch inter-block accesses across layers in a two-phase schedule, bringing total I/O well below the naïve bound.

Note that the block partition introduced below is purely an inference scheduling device. Unlike Block AttnRes, it leaves the model architecture unchanged and does not replace per-layer sources with block summaries; it simply makes the amortization argument concrete.

Setup. Let the model have $L$ layers and hidden dimension $d$, partitioned into $N$ contiguous blocks of size $S=L/N$. Inference proceeds one block at a time: Phase 1 jointly computes inter-block attention for all $S$ layers in the block against all preceding blocks, and Phase 2 walks through intra-block dependencies sequentially.

> 💡 **重要区分**: Appendix B 的 block partition 是 Full AttnRes 的推理调度技巧，不是 Block AttnRes 的结构压缩。Full 仍然保留 layer-level source，只是把 I/O 批处理。

## Phase 1: Batched Inter-block Attention

Consider block $n$ with its $S$ layers. The queries $\{w_l\}_{l\in B_n}$ are all known before execution begins, so the $(n-1)S$ preceding key-value pairs need only be read once from HBM and reused across all $S$ queries. The read cost for block $n$ is therefore

$$
\mathrm{Read}_{\mathrm{inter}}^{(n)}=2(n-1)Sd,
$$

where the factor of 2 accounts for both keys and values. Summing over all $N$ blocks and using $SN=L$:

$$
\mathrm{Read}_{\mathrm{inter}}
=
\sum_{n=1}^{N}2(n-1)Sd
=
2Sd\cdot\frac{N(N-1)}{2}
=
dL(N-1).
$$

Phase 1 also writes one $d$-dimensional output per layer, giving $\mathrm{Write}_{\mathrm{inter}}^{(n)}=Sd$ per block and

$$
\mathrm{Write}_{\mathrm{inter}}=Ld
$$

in total.

> 💡 **Phase 1 省在哪里**: 朴素做法中 block 内 $S$ 层各自扫一遍前面 $(n-1)S$ 个 source；批处理后这些 source 只从 HBM 读一次，给 $S$ 个 query 共用。

## Phase 2: Sequential Intra-block Attention

Phase 1 covers all sources before the current block. Within the block, however, each layer depends on those before it, so these must be handled in order. Layer $t$ $(1\leq t\leq S)$ reads $t-1$ intra-block key-value pairs at a cost of $2(t-1)d$. Summing over one block:

$$
\mathrm{Read}_{\mathrm{intra}}^{(n)}
=
\sum_{t=1}^{S}2(t-1)d
=
S(S-1)d.
$$

The intra-block write cost is:

$$
\mathrm{Write}_{\mathrm{intra}}^{(n)}=Sd.
$$

> 💡 **Phase 2 不能完全并行的原因**: 当前 block 内第 $t$ 层的 source 包含前面层刚产生的输出或 partial sum，所以必须按层顺序推进。

## Total Amortized I/O per Layer

Summing both phases over all $N$ blocks:

$$
\mathrm{Read}_{\mathrm{total}}
=
dL(N-1)+N\cdot S(S-1)d,\qquad
\mathrm{Write}_{\mathrm{total}}=2Ld.
$$

Dividing by $L$ and using $SN=L$:

$$
\mathrm{Read\ per\ layer}
=(N-1)d+(S-1)d
=(S+N-2)d,
\qquad
\mathrm{Write\ per\ layer}=2d.
$$

Batching inter-block reads thus brings per-layer I/O from $\mathcal O(L)$ down to $\mathcal O(S+N)$. The schedule follows the same two-phase split as Block AttnRes: inter-block attention accounts for the bulk of the traffic, while sequential computation stays local within each block.

## 中文批读

> 💡 **为什么不是 $O(L)$**: 选择合适的 $S$ 和 $N$ 后，per-layer read 变为 $(S+N-2)d$。例如把 $L$ 层分成 $\sqrt L$ 量级的 block，调度层面的 Full AttnRes I/O 可以明显低于逐层扫描。

> 💡 **和主文 Table 1 对齐**: 主文把 typical $L=128,N=8,S=16$ 放进 I/O 表。Full AttnRes two-phase 的 symbolic cost 约是 $(S+N)d$，Block AttnRes 由于只存 block summaries，典型值更低。

> 💡 **工程限制仍在**: Appendix B 解决的是本地推理 I/O，不直接消除训练 pipeline 下所有 layer outputs 跨 stage 通信的问题；这也是主文仍需要 Block AttnRes 的原因。
