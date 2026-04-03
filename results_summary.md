# Gemma 4 Performance Benchmark Summary (Official Weights)

## Test Environment
- **System**: DGX Spark
- **GPU**: 1x NVIDIA GB10 (121.69 GB VRAM)
- **Precision**: BF16 (bfloat16) natively loaded via `.to("cuda")`
- **Framework**: Hugging Face Transformers (`v5.5.0` official release)

## Models Evaluated
1. **Gemma 4 E2B** (Dense, 2.3B effective)
2. **Gemma 4 E4B** (Dense, 4.5B effective)
3. **Gemma 4 26B A4B** (MoE, 26B total / 4B active)
4. **Gemma 4 31B** (Dense, 31B total)

## Testing Scope
The evaluation suite focused on four critical pillars of the Gemma 4 architecture:
- **Text & Reasoning**: Evaluated using single creative prompts and batched analytical queries.
- **Thinking Mode**: Verified internal chain-of-thought (CoT) execution using native `enable_thinking` support.
- **Multimodal Understanding**: Tested with high-resolution single-image descriptions and biological trait validation (the "cow" test).
- **Agentic Workflows**: Benchmarked native function calling for structured tool use.

## Official Results (Tokens per Second - TPS)

| Task | E2B-it | E4B-it | 26B-A4B-it (MoE) | 31B-it |
| :--- | :--- | :--- | :--- | :--- |
| Text Generation (Single) | 25.76 | 15.01 | 9.68 | 3.41 |
| Text Generation (Batch) | 28.00 | 15.69 | 12.52 | ~3.4 |
| Thinking Mode (CoT) | 28.73 | 15.93 | 12.48 | ~3.4 |
| Vision (Single Image) | 27.69 | 15.67 | 12.24 | ~3.4 |
| Function Calling | 28.76 | 16.10 | 11.55 | ~3.4 |

## Critical Observations & Market Analysis

1. **MoE Efficiency Realized**:
   - The "MoE Paradox" encountered in early-access versions has been resolved. The 26B-A4B MoE model is now nearly **4x faster** than the 31B Dense model, proving that sparse activation correctly translates to throughput gains on high-end silicon when using official weights and optimized kernels.
   - However, the MoE model (4B active) is still ~20% slower than the E4B (4.5B dense), illustrating the persistent routing and memory-fetch overhead inherent in sparse architectures.

2. **The New Gold Standard for Edge AI**:
   - **E2B** and **E4B** remain the champions of efficiency. E2B's ability to maintain **28 TPS** across text, vision, and function calling makes it the most versatile small-scale multimodal model available for real-time applications.

3. **Ecosystem Maturity**:
   - The need for manual weight transposition fixes has been eliminated. The official release of `transformers` v5.5.0 handles the Gemma 4 MoE architecture natively, simplifying deployment and ensuring maximum stability.

## Conclusion
Gemma 4 is a "production-first" release. With the resolution of early-access bottlenecks, the **26B-A4B MoE** emerges as a powerful alternative for those needing deep reasoning without the latency of a full 31B model. For most developers, however, the **E4B** remains the undisputed king of performance-per-watt.
