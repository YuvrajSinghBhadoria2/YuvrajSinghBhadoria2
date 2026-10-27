# Yuvraj Singh Bhadoria
**LLM Inference Optimization Engineer** — Building kernels & serving systems that make LLMs faster, cheaper, and longer-context.

---

## 🔥 Flagship Projects

| Project | What It Does | Tech Stack | Measured Impact |
|---------|--------------|------------|-----------------|
| **[FP4 KV-Cache for llama.cpp](https://github.com/YuvrajSinghBhadoria2/fp4-kvcache-llama-cpp)** | 4× KV memory compression on Blackwell (RTX 5090) via custom FP4 E2M1 quantization + MXFP4 conversion for native FP4 MMA | CUDA, PTX, MXFP4, llama.cpp internals | **Target: 230+ tok/s decode** (vs 207 FP16 baseline) — *Upstream PR in progress* |
| **[Triton Attention Kernels](https://github.com/YuvrajSinghBhadoria2/triton-attention-kernels)** | FA2 prefill, split-KV flash decode, paged attention — roofline-validated on T4 | Triton, CUDA, NCU profiling | **10× prefill memory reduction**, **2.8× decode speedup** at 16K context |
| **[Speculative Decoding Engine](https://github.com/YuvrajSinghBhadoria2/speculative-decoding-engine)** | From-scratch n-gram + EAGLE-style drafters with distribution-preserving verification | Python, PyTorch | **1.2–1.6× speedup** (train-free), **2–3×** with trained drafter |
| **[inferlast](https://github.com/YuvrajSinghBhadoria2/inferlast)** | CPU-first optimizer with honest benchmarking, quantization quality auditing, GPU-worthiness estimation | Python, llama.cpp, vLLM | Catches **false wins from noise**; emits deploy-ready configs |
| **[Flash Attention Benchmark](https://github.com/YuvrajSinghBhadoria2/flash-attention-triton-cuda-benchmark)** | Reproducible roofline-grounded comparison: Triton vs CUDA vs SDPA | Triton, CUDA, NCU | Methodology used across projects |

---

## 📊 Measured Results

| Metric | Baseline | Optimized | Method |
|--------|----------|-----------|--------|
| **KV cache memory** | 4096 bytes/token (FP16) | **1152 bytes/token (4×)** | FP4 E2M1 quantization |
| **Prefill memory (4K ctx)** | 4.46 GB (naive) | **0.44 GB (10× less)** | Triton FA2 online softmax |
| **Decode latency (split-KV 32)** | 1.14 ms | **0.40 ms (2.8×)** | Split-KV flash decoding |
| **Speculative decoding speedup** | 1.0× (autoregressive) | **1.6×** | n-gram drafter + rejection sampling |
| **CPU llama.cpp decode** | Baseline | **4× faster** | Thread/quant/batch tuning |
| **Production serving latency** | 800 ms | **<300 ms (-62%)** | vLLM + FastAPI + K8s optimization |
| **Inference cost** | Baseline | **-25%** | Quantization + batching + routing |

---

## 🛠 Technical Stack

**Kernels & Quantization**  
`CUDA` `PTX (MMA)` `Triton` `FP4 E2M1` `MXFP4` `UE8M0` `FlashAttention-2/3` `Paged Attention` `Split-KV`

**Frameworks & Serving**  
`vLLM` `SGLang` `TensorRT-LLM` `llama.cpp` `Ray Serve` `FastAPI` `Docker` `Kubernetes`

**Profiling & Methodology**  
`NCU` `NSYS` `Roofline Analysis` `Memory Bandwidth Optimization` `Quantization Quality Auditing` `Honest Benchmarking`

**Core ML**  
`PyTorch` `Speculative Decoding` `EAGLE/Medusa` `GQA` `KV Cache Management` `Prefix Caching`

---

## 📝 Technical Writing

- **[FP4 E2M1 → MXFP4 Conversion Mathematics](https://github.com/YuvrajSinghBhadoria2/fp4-kvcache-llama-cpp/blob/main/upstream-pr/MXFP4_CONVERSION_MATH.md)** — Why Blackwell requires MXFP4 for native FP4 MMA, UE8M0 scale encoding, conversion algorithms
- **[Triton Flash Attention Kernel Walkthrough](https://github.com/YuvrajSinghBhadoria2/triton-attention-kernels)** — Kernel structure, online softmax, split-KV parallelism, paged attention block tables
- **[Honest Benchmarking Methodology](https://github.com/YuvrajSinghBhadoria2/inferlast)** — Framework overhead measurement, quantization quality via logit cosine, trust-check for false wins

---

## 💼 Experience

**Bank of America** — *LLM Inference Engineer* (Current)  
Optimizing LLM serving infrastructure: vLLM/TensorRT-LLM deployment, quantization pipelines, latency/cost reduction.

**Policybazaar** — *ML Engineer*  
Built ML systems for insurance domain: recommendation engines, fraud detection, NLP pipelines.

**Infosys** — *Software Engineer*  
Full-stack development and cloud migration projects.

---

## 🎯 Currently Seeking

**Senior / Staff Inference Optimization Engineer** roles at companies building:
- LLM serving infrastructure (Together AI, Fireworks, Anyscale, Baseten, Modal, vLLM core)
- Custom AI silicon & compilers (Groq, Cerebras, Tenstorrent, SambaNova, NVIDIA TensorRT-LLM, AMD MLIR/IREE)
- Applied research serving stacks (xAI, Anthropic, Mistral, Cohere)

**Location:** Remote / Hybrid US  
**Availability:** Immediate

---

## 📫 Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/yuvraj-singh-bhadoria)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/YuvrajSinghBhadoria2)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:yuvraj.bhadoria@gmail.com)

---

*Last updated: October 2026*