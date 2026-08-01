[← 返回 README](../README.md)

# 04 Discussion / Conclusion / Appendix

> 💡 **Hao 批注 - 结论质量**: 结论简洁有力，重申了核心洞察——"adding context is not the same as learning from context; optimization can decide which signal is actually used." 这句话值得作为论文的灵魂句。

---

## 6 Conclusion

This paper identifies spatial blindness: context-aware MIL can perform well at slide level while using little tissue topology. ResTopoMIL addresses this failure mode by fitting composition first and then training topology as a residual correction under a shuffle-based spatial constraint. Across 9 public benchmarks it improves prediction, restores spatial sensitivity, and gives stronger localization evidence with only 1.15M parameters. The broader lesson is that adding context is not the same as learning from context; optimization can decide which signal is actually used. Future work should test this diagnosis with trainable pathology foundation encoders, prospective cohorts, and clinically realistic coordinate perturbations such as registration noise or rigid transformations.

---

## Appendix L: Additional Limitations and Negative-Result Scope

> 💡 **Hao 批注 - 坦诚的局限性讨论**: 这是论文可圈可点的部分——作者坦诚承认了两个重要局限：

> 💡 **Hao 批注 - 局限1: composition-dominant 任务未验证**: 论文聚焦于结构依赖任务，但对于纯组合任务(拓扑无关的病理任务)，统计流应接近最优，拓扑残差分支可能增加噪声或过拟合。Spatial-MNIST-Bag Dataset A 提供了合成控制，但缺乏真实 WSI 的"负控制"数据集。

> 💡 **Hao 批注 - 局限2: 两阶段训练不即插即用**: 需要选择 warmup 长度(固定 10 epoch)，统计锚点必须足够强但又不能过拟合。如果锚点弱，Stage 2 可能被要求修正非空间错误；如果锚点过拟合，留给图分支的残差可能充满噪声或太小。缺少自适应停止规则。

The experiments deliberately emphasize structure-dependent WSI tasks, because those are the settings where spatial blindness is most clinically and methodologically relevant. This focus leaves a weaker view of the opposite regime: real pathology tasks that are almost purely compositional. In such tasks, a strong statistical stream should be close to optimal, and a residual topological branch could plausibly add noise or overfit incidental tissue layout.

Spatial-MNIST-Bag Dataset A provides a controlled synthetic check of this regime, showing that ResTopoMIL can still solve a pure-composition MIL problem. However, it is not a realistic WSI benchmark. We currently lack a public real WSI dataset whose label is known to be determined primarily by composition while being insensitive to tissue arrangement. Because of this missing negative-control dataset, our exploration of composition-dominant failure modes remains limited. A useful future benchmark would contain real WSI labels for which pathologists agree that topology is largely irrelevant; such a dataset would test whether ResTopoMIL correctly defaults to statistical performance without introducing unnecessary residual variance.

The two-stage training protocol is another limitation. It makes the optimization target cleaner for the graph branch, but it also introduces a schedule choice: the statistical anchor must be trained long enough to capture composition, yet not treated as a perfect model. If the anchor is weak, Stage 2 may be asked to correct errors that are not truly spatial. If the anchor overfits, the residual available to the graph branch may be noisy or too small. The present experiments use a fixed 10-epoch warmup and then 30/20 epochs of refinement, and the ablations show that this choice is stable on the evaluated benchmarks. Still, the method is less plug-and-play than a single end-to-end MIL baseline, and future work should study adaptive stopping rules or validation criteria for deciding when to freeze the statistical stream.

---

## Appendix B: Design Analysis (详细分析)

> 💡 **Hao 批注 - 理论分析定位**: 作者明确声明附录 B 的理论分析不声称是新的通用理论贡献——其目的是说明 ResTopoMIL 的三个设计选择(统计锚点、stop-gradient 残差训练、坐标打乱约束)为何应该一起使用。这种诚实的定位值得赞赏。

### B.1 Residual-Error Gating of the Graph Gradient

For binary classification with cross-entropy loss, the gradient received by the topological branch is:

$$
\nabla _ { \theta _ { t } } \mathcal { L } = \mathbb { E } \bigl [ ( \hat { p } _ { \theta } ( X ) - Y ) \nabla _ { \theta _ { t } } f _ { t o p o } ( X ; \theta _ { t } ) \bigr ] .
$$

The bound: $\| \nabla _ { \theta _ { t } } \mathcal { L } \| \leq ( \mathbb { E } r _ { \theta } ( X , Y ) ^ { 2 } ) ^ { 1 / 2 } ( \mathbb { E } \| \nabla _ { \theta _ { t } } f _ { t o p o } ( X ; \theta _ { t } ) \| _ { F } ^ { 2 } ) ^ { 1 / 2 }$.

The stop-gradient in Stage 2 replaces the moving joint target with $f(X) = sg[f_{stat}(X)] + f_{topo}(X; \theta_t)$. The residual error is now computed against a fixed compositional anchor; the statistical stream can no longer reduce this residual during Stage 2.

### B.2 What Information the Residual Branch Is Asked to Learn

The information decomposition: $I(Z_{stat}, Z_{topo}; Y) = I(Z_{stat}; Y) + I(Z_{topo}; Y | Z_{stat})$.

The Stage-2 decoder $q_{\phi}(Y=y|Z_{stat}, Z_{topo}) = [Softmax(f_{stat}(Z_{stat}) + f_{topo}(Z_{topo}; \phi))]_y$ minimizes a variational upper bound on $H(Y|Z_{stat}, Z_{topo})$, thereby maximizing a lower bound on $I(Z_{topo}; Y | Z_{stat})$ with the statistical stream held fixed.

The chain rule alone would still allow $Z_{topo}$ to encode another compositional statistic not captured by $Z_{stat}$. ResTopoMIL therefore adds the coordinate-specific constraint $\mathcal{L}_{texture}$: the shuffled view preserves all patch appearances and labels but corrupts the coordinate-induced graph, so a branch that ignores coordinates cannot reliably satisfy the margin.

### B.3 Why Common Optimization Heuristics Are Not Equivalent

| Heuristic | Mechanism | Why Insufficient |
|-----------|-----------|-----------------|
| Multi-LR | Scales graph update by alpha>1 | Update still gated by shrinking residual |
| Statistical Dropout | Corrupts stat stream to increase residual | Injects noise into useful compositional evidence |
| Curriculum Scheduling | Changes relative speed of two streams | Residual target remains moving |
| Hard Instance Mining | Reweights difficult samples | Easy stream still absorbs part of residual |
| **ResTopoMIL (stop-gradient)** | Fixes compositional anchor | Graph branch no longer competes with moving stat predictor |

---

## Appendix I: CAMELYON-16 Localization Protocol

> 💡 **Hao 批注 - 定位协议细节**: 值得注意的细节：(1) 使用 node-level topological residual $s_i = W_{topo,y}^T H_i^{(2)}$ 而非 attention weight 作为定位分数；(2) 阈值在 validation split 上统一选择(非 per-slide)；(3) FROC 使用标准 CAMELYON operating points (1/8, 1/4, 1/2, 1, 2, 4, 8 FPs/slide)。

Patch-level labels: patches with >=1% overlap with annotated tumor region labeled positive.
Scores: $s_i = (W_{topo,y})^T H_i^{(2)}$ as raw patch-level localization score.
Threshold: selected on validation split by maximizing mean Dice, fixed for test slides.
FROC: threshold-swept, connected components as lesion candidates, candidate score = max patch score inside component.
