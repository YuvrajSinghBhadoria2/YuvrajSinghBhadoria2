# Yuvraj Singh Bhadoria
LLM Inference Optimization Engineer — kernels, serving, and quantization for production LLM systems.

---

## Work That Matters

| Project | What It Is | Stack | Result |
|---------|------------|-------|--------|
| **[FP4 KV-Cache for llama.cpp](https://github.com/YuvrajSinghBhadoria2/fp4-kvcache-llama-cpp)** | 4× KV memory compression on Blackwell (RTX 5090) via custom FP4 E2M1 + MXFP4 conversion for native FP4 tensor cores | CUDA, PTX, MXFP4, llama.cpp internals | 1152 vs 4096 bytes/token; targeting 230+ tok/s decode (vs 207 FP16 baseline) — upstream PR in progress |
| **[Triton Attention Kernels](https://github.com/YuvrajSinghBhadoria2/triton-attention-kernels)** | FA2 prefill, split-KV flash decode, paged attention — roofline-validated on T4 | Triton, CUDA, NCU | 10× prefill memory reduction (4.46 GB → 0.44 GB); 2.8× decode speedup at 16K context |
| **[Speculative Decoding Engine](https://github.com/YuvrajSinghBhadoria2/speculative-decoding-engine)** | From-scratch n-gram + EAGLE-style drafters with distribution-preserving verification | Python, PyTorch | 1.6× speedup train-free; 2–3× with trained drafter |
| **[inferlast](https://github.com/YuvrajSinghBhadoria2/inferlast)** | CPU-first optimizer with honest benchmarking, quantization quality auditing, GPU-worthiness estimation | Python, llama.cpp, vLLM | Catches false wins from noise; emits deploy-ready configs |
| **[Flash Attention Benchmark](https://github.com/YuvrajSinghBhadoria2/flash-attention-triton-cuda-benchmark)** | Reproducible roofline-grounded comparison: Triton vs CUDA vs SDPA | Triton, CUDA, NCU | Methodology used across projects |

---

## Measured Results

| Metric | Baseline | Optimized | Method |
|--------|----------|-----------|--------|
| KV cache memory | 4096 B/token (FP16) | **1152 B/token (4×)** | FP4 E2M1 quantization |
| Prefill memory (4K ctx) | 4.46 GB (naive) | **0.44 GB (10× less)** | Triton FA2 online softmax |
| Decode latency (split-KV 32) | 1.14 ms | **0.40 ms (2.8×)** | Split-KV flash decoding |
| Speculative decoding speedup | 1.0× | **1.6×** | n-gram drafter + rejection sampling |
| CPU llama.cpp decode | baseline | **4× faster** | Thread/quant/batch tuning |
| Production serving latency | 800 ms | **<300 ms** | vLLM + FastAPI + K8s |
| Inference cost | baseline | **-25%** | Quantization + batching + routing |

---

## Technical Stack

**Kernels & Quantization**  
CUDA • PTX (MMA) • Triton • FP4 E2M1 • MXFP4 • UE8M0 • FlashAttention-2/3 • Paged Attention • Split-KV

**Frameworks & Serving**  
vLLM • SGLang • TensorRT-LLM • llama.cpp • Ray Serve • FastAPI • Docker • Kubernetes

**Profiling & Methodology**  
NCU • NSYS • Roofline Analysis • Memory Bandwidth Optimization • Quantization Quality Auditing • Honest Benchmarking

**Core ML**  
PyTorch • Speculative Decoding (EAGLE, Medusa) • GQA • KV Cache Management • Prefix Caching

---

## Writing

- [FP4 E2M1 → MXFP4 Conversion Mathematics](https://github.com/YuvrajSinghBhadoria2/fp4-kvcache-llama-cpp/blob/main/upstream-pr/MXFP4_CONVERSION_MATH.md) — Why Blackwell requires MXFP4 for native FP4 MMA, UE8M0 scale encoding, conversion algorithms
- [Triton Flash Attention Kernel Walkthrough](https://github.com/YuvrajSinghBhadoria2/triton-attention-kernels) — Kernel structure, online softmax, split-KV parallelism, paged attention block tables
- [Honest Benchmarking Methodology](https://github.com/YuvrajSinghBhadoria2/inferlast) — Framework overhead measurement, quantization quality via logit cosine, trust-check for false wins

---

## Experience

**Bank of America** — LLM Inference Engineer (Current)  
Optimizing LLM serving infrastructure: vLLM/TensorRT-LLM deployment, quantization pipelines, latency/cost reduction.

**Policybazaar** — ML Engineer  
ML systems for insurance: recommendation engines, fraud detection, NLP pipelines.

**Infosys** — Software Engineer  
Full-stack development and cloud migration.

---

## Seeking

**Senior / Staff Inference Optimization Engineer** at companies building LLM serving infrastructure, custom AI silicon, or compiler stacks.

Remote / Hybrid US — available immediately.

---

## Connect

[LinkedIn](https://linkedin.com/in/yuvraj-singh-bhadoria) • [GitHub](https://github.com/YuvrajSinghBhadoria2) • [Email](mailto:yuvrajsinghbhado2030@gmail.com)