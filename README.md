# RAGO_RAGCache_LLM
This method optimises RAG Pipelines to use less inference cost and reduce latency. We will be hosting the LLM in a GPU Environment to run this.

# RAG Optimizer + Cache (Prototype)
A small, runnable prototype that **combines ideas from RAGO (performance/config optimization) + RAGCache (multi-level caching + overlap/pipelining)**, with an optional “query-level controller” hook inspired by RAGServe/METIS.

This repo is **not** the official implementation of those papers.  
It’s a **clean-room, minimal prototype** that demonstrates the *core mechanics* in a way you can extend into a real RAG serving stack.

---

## What you get

### 1) `rag_rago_cache.cpp` (C++)
A single-file “server-ish” loop that demonstrates:
- **Retrieval cache**: `query -> doc_ids` (LRU)
- **Block cache**: `doc_id -> text block` (LRU)
- **Overlap/pipelining**: retrieval runs async while we do a small “draft” step (toy), similar in spirit to overlapping retrieval and generation.
- **RAGO-like knob selection**: a tiny cost model/heuristic picks `top_k`, `batch`, and a “cheap_mode” to keep latency within a target budget.

### 2) `rag_rago_cache_cuda.py` (Python + CUDA via CuPy)
A runnable GPU retrieval loop that demonstrates:
- A “document embedding bank” on GPU
- **Top-k retrieval** with GPU matrix multiply + `argpartition`
- **Retrieval caching** and “context budgeting”
- **RAGO-like runtime tuning**: chooses `top_k`, `token_budget`, and an `int8_store` toggle depending on observed bottlenecks.

---

## Why this is “RAGO + RAGCache” and not something else

### RAGCache-style parts (caching + overlap)
- **Multi-level caching**:
  - retrieval results cache (`query -> doc_ids`)
  - context block cache (`doc_id -> block`)
- **Overlap/pipelining**:
  - start retrieval in background, do small work while it runs (prototype version of overlapping retrieval and generation).

The RAGCache paper explicitly targets caching intermediate states and overlapping retrieval and inference to reduce end-to-end latency and TTFT. :contentReference[oaicite:0]{index=0}

### RAGO-style parts (system config optimization)
- A small controller that adapts:
  - retrieval depth (`top_k`)
  - context/token budget
  - execution mode (“cheap_mode” / “int8_store” as placeholders for quantization/placement/batching knobs)

RAGO proposes a system optimization framework that searches/chooses system configurations based on workload + hardware for efficient RAG serving. :contentReference[oaicite:1]{index=1}

### Optional RAGServe/METIS-style extension (query-level adaptation)
If you later add a query classifier (easy/hard), you can adapt `top_k`/compression per query in the spirit of METIS/RAGServe’s per-query configuration adaptation. :contentReference[oaicite:2]{index=2}

---

## Quick start

### Build and run (C++)
```bash
g++ -O2 -std=c++17 rag_rago_cache.cpp -o rag_demo
./rag_demo
