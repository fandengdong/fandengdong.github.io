---
title: "SGLang"
description: "Notes on SGLang,  a fast serving framework for large language models and vision language models."
---

SGLang is a high-performance serving framework for large language models and vision-language models. It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters. Its core features include:

1. **RadixAttention for Prefix Caching**
    - Implements a specialized attention mechanism for efficient prefix caching
    - More advanced than standard paged attention used in vLLM
2. **Zero-Overhead CPU Scheduler**
    - Unique scheduling approach that eliminates CPU overhead
    - Unlike traditional schedulers that may introduce latency
3. **Extensive Hardware Support**
    - Runs on NVIDIA GPUs (GB200/B300/H100/A100/Spark), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, **Ascend NPUs**, and more.
4. **Multi-Modal and Multi-LoRA Support**
    - Native support for vision-language models
    - Advanced multi-LoRA batching for serving multiple adapted models simultaneously

# useful links

- [github](https://github.com/sgl-project/sglang)
