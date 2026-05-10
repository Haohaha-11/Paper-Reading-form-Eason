[← 返回 README](../README.md)

# 5 Conclusion

## 📌 预览
Conclusion 重申 CAR 的定位：无 LR 监督、由 SR performance 指导的内容自适应下采样器，在保持 LR 可视质量接近 bicubic 的同时提升 SR。

---

# 5 Conclusion

This paper introduces the CAR model for image downscaling, which is an end-to-end system trained by maximizing the SR performance. It simultaneously learns a mapping for resolution reduction and SR performance improvement. One major contribution of our work is that the CAR model is trained in an unsupervised manner meaning that there is no assumption on how the original HR image will be downscaled, which helps the image downscale model to learn to keep essential information for SR task in a more optimal way. This is achieved by the content adaptive resampling kernel generation network which estimates spatial non-uniform resampling kernels for each pixel in the downscaled image according to the input HR image. The downscaled pixel value is obtained by decimating HR pixels covered by the resampling kernel. Our experimental results illustrate that the CAR model trained jointly with the SR networks achieves a new state-of-the-art SR performance while produces downscaled images whose quality are comparable to that of the widely adopted image downscaling method.

> 💡 **结论批读**: CAR 的可复用思想是把“压缩/下采样”设计成被后续任务反向指导的可微采样器。它适合需要低分辨率存储或传输、再由确定上游模型恢复的场景；风险是 LR 可能学习到对机器友好但对人眼不完全自然的结构，需要 TV/感知约束控制。


## 🔖 Section 总结

- 论文最终 claim：CAR 同时学习 resolution reduction 和 SR improvement。
- 方法边界：依赖下游 SRNet 和训练数据；LR 质量需要正则权衡。
- 对 Medical-Compression 的启发：可把“压缩预览图”设计成服务诊断重建模型的可微采样结果，但必须加入医学可读性和安全约束。
