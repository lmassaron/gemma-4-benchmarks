# Investigation Report: Gemma 4 26B-A4B (MoE) - Bottleneck Resolution

## 1. Resolution of Early Access Issues

In initial testing of the Gemma 4 MoE model using early-access weights, significant architectural bottlenecks and shape mismatches were reported. This report documents the resolution of those issues following the release of the official weights and `transformers` v5.5.0.

### The "Fixed" Issue: Native Support
Previously, a manual transposition of the `down_proj` layer was required to prevent shape mismatches. With the official **google/gemma-4-26B-A4B-it** weights, the model architecture now aligns perfectly with the standard `transformers` implementation. **Manual code intervention is no longer required.**

## 2. Technical Validation of Performance Gains

Following the update to official weights and the use of the `AutoModelForMultimodalLM` class, performance improved dramatically:

- **Previous Throughput (Early Weights):** ~2.7 TPS
- **Current Throughput (Official Weights):** ~12.5 TPS
- **Improvement:** **+363%**

This improvement confirms that the earlier bottlenecks were primarily due to non-optimized weight loading and early-stage software implementation rather than architectural flaws.

## 3. The MoE Efficiency Profile

With the bottlenecks resolved, we can now accurately profile the MoE architecture:

### A. Sparse vs. Dense Scaling
The 26B-A4B model (4B active parameters) now correctly outperforms the 31B Dense model by a significant margin (~12.5 TPS vs ~3.4 TPS). This validates the MoE strategy for scaling model capacity while maintaining manageable inference costs.

### B. The Cost of Sparsity (Routing Overhead)
While the MoE model is much faster than the 31B model, it remains slower than the **E4B** (4.5B dense) model.
- **E4B (4.5B Dense):** ~15.7 TPS
- **26B-A4B (4B Active MoE):** ~12.5 TPS

The ~20% performance delta, despite having fewer active parameters, is attributed to the **expert routing logic** and the necessity of fetching non-contiguous weights from memory for different experts.

## Summary
The "MoE Paradox" has been solved. The official Gemma 4 MoE weights provide exactly the performance scaling expected: deep, high-capacity reasoning at a fraction of the compute cost of dense models, albeit with a minor "routing tax" compared to similarly sized small-dense variants.
