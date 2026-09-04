<h2> Hi, I'm Yuvraj Singh Bhadoria! <img src="https://media.giphy.com/media/mGcNjsfWAjY5AEZNw6/giphy.gif" width="50"></h2>

<img align='right' src="https://media.giphy.com/media/l3vR16pONs5pOqyAM/giphy.gif" width="230">

<p><em>LLM Inference Engineer at <a href="https://www.bankofamerica.com/">Bank of America</a><img src="https://media.giphy.com/media/fYSnHlufseco8Fh93Z/giphy.gif" width="30"></br>Ex-Policybazaar · Ex-Infosys<img src="https://media.giphy.com/media/WUlplcMpOCEmTGBtBW/giphy.gif" width="30"> 
</em></p>

[![Linkedin: yuvraj-singh-bhadoria](https://img.shields.io/badge/-yuvraj--singh--bhadoria-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/yuvraj-singh-bhadoria/)](https://www.linkedin.com/in/yuvraj-singh-bhadoria/)
[![GitHub YuvrajSinghBhadoria2](https://img.shields.io/github/followers/YuvrajSinghBhadoria2?label=follow&style=social)](https://github.com/YuvrajSinghBhadoria2)
[![Email](https://img.shields.io/badge/-yuvrajsinghbhado2030@gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:yuvrajsinghbhado2030@gmail.com)

---

### <img src="https://media.giphy.com/media/VgCDAzcKvsR6OM0uWg/giphy.gif" width="50"> A little more about me...  

```python
class LLMInferenceEngineer:
    """
    LLM Inference Engineer @ Bank of America
    Making models faster and cheaper in production
    """
    
    def __init__(self):
        self.name = "Yuvraj Singh Bhadoria"
        self.pronouns = ["he", "him"]
        self.role = "LLM Inference Engineer @ Bank of America"
        
    # Core focus — inference optimization
    focus = [
        "LLM Serving",            # vLLM, SGLang, TensorRT-LLM
        "Kernel Optimization",    # Triton, CUDA, custom attention kernels
        "Speculative Decoding",   # draft-verify, EAGLE, n-gram drafters
        "Quantization & Cost",    # INT4/INT8, speed vs accuracy tradeoffs
    ]
    
    # Tech stack — inference-first
    serving = ["vLLM", "SGLang", "TensorRT-LLM", "Triton Inference Server"]
    kernels = ["CUDA", "Triton", "FlashAttention", "PagedAttention"]
    frameworks = ["PyTorch", "JAX", "Transformers", "llama.cpp"]
    profiling = ["Nsight Systems", "PyTorch Profiler", "ncu", "torch.cuda.Event"]
    infra = ["Docker", "Kubernetes", "NVIDIA GPU", "Ray Serve"]
    
    # What I do daily
    @property
    def work(self):
        return {
            "kernel_dev": "Writing Triton/CUDA kernels for fused ops",
            "serving": "Tuning vLLM/SGLang for throughput and latency",
            "profiling": "Finding bottlenecks with NSight and torch profiler",
            "optimization": "Quantization, KV-cache, speculative decoding",
        }
    
    # Shipped results
    @property
    def metrics(self):
        return {
            "inference_latency": "800ms → <300ms",   # -62%
            "cpu_inference": "~4x faster decoding",   # llama.cpp
            "inference_cost": "-25%",                 # saved
            "pricing_error": "-18%",                  # reduced
        }
    
    # Experience
    @property
    def experience(self):
        return {
            "Bank of America": "LLM inference infrastructure & optimization",
            "Policybazaar": "RAG systems & document intelligence",
            "Infosys": "ML solutions at scale",
        }
    
    # Currently building
    exploring = "SGLang RadixAttention & CUDA graph capture"
    building = "High-throughput LLM serving pipelines"
    writing = "LLM inference blog series"
```

---

### <img src="https://media.giphy.com/media/LnQjpWaON8nhr21vNW/giphy.gif" width="60"> <em><b>I make LLMs faster and cheaper in production</b> — inference optimization, GPU kernels, and the systems around them. Let's connect!</em>

---

<div align="center">

### `inference optimization work`

</div>

<table>
<tr>

<td width="50%" valign="top">

### triton-attention-kernels

**Portable Triton attention kernels for LLM inference.**

`Triton` `CUDA` `FlashAttention` `PagedAttention`

Fused prefill · split-KV flash-decoding · paged KV-cache attention — validated against PyTorch SDPA on NVIDIA T4

**→ [explore repository](https://github.com/YuvrajSinghBhadoria2/triton-attention-kernels)**

</td>

<td width="50%" valign="top">

### speculative-decoding-engine

**From-scratch speculative decoding for LLM inference.**

`Python` `Speculative Decoding` `LLM`

N-gram drafter + distribution-preserving verification · measured speedup vs autoregressive

**→ [explore repository](https://github.com/YuvrajSinghBhadoria2/speculative-decoding-engine)**

</td>

</tr>
<tr>

<td width="50%" valign="top">

### production-llm-platform

**Production LLM serving infrastructure.**

`vLLM` `FastAPI` `PostgreSQL` `Redis` `Kubernetes`

PagedAttention · continuous batching · multi-tenant routing · circuit breakers

**→ [explore repository](https://github.com/YuvrajSinghBhadoria2/production-llm-platform)**

</td>

<td width="50%" valign="top">

### llm-quant-serve

**Quantize + serve LLMs with vLLM.**

`Python` `vLLM` `Quantization` `Benchmarking`

Speed vs accuracy measurement with 2-pt guard (USE/AVOID)

**→ [explore repository](https://github.com/YuvrajSinghBhadoria2/llm-quant-serve)**

</td>

</tr>
<tr>

<td width="50%" valign="top">

### llm-cpu-inference-optimization

**Measured CPU LLM inference with llama.cpp.**

`Python` `llama.cpp` `Quantization` `CPU`

Thread-topology · quantization · batching · ~4x faster decoding, identical output quality

**→ [explore repository](https://github.com/YuvrajSinghBhadoria2/llm-cpu-inference-optimization)**

</td>

<td width="50%" valign="top">

### inferlast

**CPU-first LLM inference optimizer.**

`Python` `Benchmarking` `Quantization` `CPU`

Profiles the true bottleneck, picks a change, refuses to call it a win unless the benchmark is honestly real

**→ [explore repository](https://github.com/YuvrajSinghBhadoria2/inferlast)**

</td>

</tr>
<tr>

<td width="50%" valign="top">

### vllm-inference-tradeoff-benchmark

**Speed-vs-accuracy tradeoffs of vLLM serving configs.**

`Jupyter Notebook` `vLLM` `Benchmarking`

Measured on a real model + free GPU · USE/AVOID decision rule with accuracy guard

**→ [explore repository](https://github.com/YuvrajSinghBhadoria2/vllm-inference-tradeoff-benchmark)**

</td>

<td width="50%" valign="top">

### llm-inference-blog

**LLM inference optimization blog series.**

`HTML` `Writing`

Day 1+ posts on serving, kernels, and practical optimization

**→ [explore repository](https://github.com/YuvrajSinghBhadoria2/llm-inference-blog)**

</td>

</tr>
</table>

---

<div align="center">

## `the stack`

<br>

<img src="https://skillicons.dev/icons?i=python,cpp,pytorch,jax,linux,docker,kubernetes,cuda,nvidia&perline=8" />

<br><br>

`vLLM` · `SGLang` · `Triton` · `CUDA` · `FlashAttention` · `TensorRT-LLM` · `PyTorch` · `llama.cpp` · `Ray Serve`

</div>

---

## `where I'm going deeper`

```text
                INFERENCE OPTIMIZATION

     ┌──────────────┬──────────────┬──────────────┐
     │              │              │              │
   SERVING       KERNELS       MEMORY        DECODING
     │              │              │              │
   vLLM          Triton        KV-cache       Speculative
   SGLang        CUDA          PagedAttn      EAGLE
   TRT-LLM      FlashAttn     Quantization   N-gram draft
     │              │              │              │
     └──────────────┼──────────────┼──────────────┘
                    │
                    ▼
              GPU SYSTEMS
                    │
             profiling / tuning
             latency / cost / $/token
                    │
                    ▼
              PRODUCTION
```

<br>

I'm focused on **inference optimization — making LLMs faster and cheaper to serve**.  
Core areas: **vLLM, SGLang, Triton kernels, CUDA, speculative decoding, quantization, and GPU systems**.  
Always happy to connect with others working on the same problems.