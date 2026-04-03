# Deep Dive: Hands-on with the New Gemma 4 Series (Official Weights)

The open-weight AI landscape is moving at breakneck speed, and the latest iteration from Google—**Gemma 4**—has just transitioned from early-access to official release with a major performance-optimized weight update. 

But how do these official models actually perform in a production-grade environment? I put the final weights to the test on a **DGX Spark** system, and the results for the larger models have changed the narrative.

---

## The Gemma 4 Lineup: Speed Meets Sparse Reasoning

Gemma 4 introduces three distinct architectures across four sizes, each optimized for specific hardware tiers. The most anticipated change in this update is the native optimization of the Mixture of Experts (MoE) variant:

1. **Gemma 4 E2B (2.3B effective)**: Optimized for mobile, featuring Per-Layer Embeddings (PLE).
2. **Gemma 4 E4B (4.5B effective)**: The sweet spot for laptops, balancing LLM-level reasoning with edge speeds.
3. **Gemma 4 26B-A4B (MoE)**: A sparse model that activates only 4B parameters, now optimized for 4x higher throughput.
4. **Gemma 4 31B**: The flagship dense model, designed for maximum reasoning depth.

---

## The Benchmarking Rig: DGX Spark

To see the final capabilities, I ran the tests on a **DGX Spark** equipped with an **NVIDIA GB10 GPU (121.69 GB of VRAM)** using official `transformers` v5.5.0 support.

| Task | E2B-it (TPS) | E4B-it (TPS) | 26B-A4B MoE (TPS) | 31B Dense (TPS) |
| :--- | :--- | :--- | :--- | :--- |
| Text Generation | 28.00 | 15.69 | 12.52 | 3.41 |
| Thinking Mode (CoT) | 28.73 | 15.93 | 12.48 | ~3.4 |
| Vision (Multimodal) | 27.69 | 15.67 | 12.24 | ~3.4 |
| Function Calling | 28.76 | 16.10 | 11.55 | ~3.4 |

---

## Key Takeaways: The MoE Breakthrough

### 1. MoE Performance is Unlocked
In early-access testing, the 26B-A4B MoE model was plagued by architectural bottlenecks, barely hitting 2.7 TPS. With the official weights and optimized native support in `transformers`, that performance has skyrocketed to **~12.5 TPS**. It is now nearly **4x faster than the 31B Dense model**, proving that sparse activation is the superior strategy for those balancing reasoning depth with speed.

### 2. The "Small Model" Performance Ceiling
The **E2B** and **E4B** models remain incredibly responsive, consistently hitting **28 TPS** and **16 TPS**. Interestingly, the **E4B (Dense)** remains slightly faster than the **MoE (Sparse)** model, even though the MoE activates fewer parameters (4B vs 4.5B). This highlights the "routing tax" of sparse architectures—the overhead of choosing and fetching experts still costs about 20% in raw throughput.

### 3. Native "Thinking" is Built-In
One of the most impressive findings is that enabling **Thinking Mode** (native Chain-of-Thought) has virtually no impact on performance. The throughput during the reasoning phase is identical to standard generation, allowing developers to implement transparent AI logic without any latency penalty.

---

## Conclusion: A New Standard for Open Models

The transition to official weights has cemented Gemma 4 as a high-performance powerhouse. By resolving the early bottlenecks in the MoE architecture, Google has provided a clear path for developers: use the **E4B** for maximum edge speed, or the **26B-A4B MoE** for a nearly perfect balance of 31B-level reasoning at nearly 4.5B-level speeds. The era of the agent-ready, multimodal open model is officially here.
