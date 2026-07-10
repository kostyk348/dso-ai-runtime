# DSO-AI Runtime

High-performance, **zero-malloc-in-loop** C++ inference engine for transformer LLMs
(Qwen2/Qwen2.5) and GGUF models (incl. hybrid SSM+Attention `qwen35`), with
**layer-by-layer disk streaming**, a static memory **arena**, **INT8 weight-only PTQ**,
AVX2 SIMD GEMM and async page-cache windowing.

This is a from-scratch runtime: weights are `mmap`'d from disk, exactly **one weight
matrix is resident in RAM at a time**, and the OS page-cache holds only a sliding
window of ~1–2 layers (the rest is evicted with `madvise(MADV_DONTNEED)` after use).
A background thread prefetches the next layer (`MADV_WILLNEED`) so I/O overlaps compute.

## Features

- **Streaming from disk** — no need to fit the whole model in RAM. RAM usage is
  independent of model size.
- **INT8 weight-only PTQ** (per-row symmetric scale) — ~2× smaller on disk, 4×
  smaller active weight buffer vs FP32, minimal accuracy loss.
- **AVX2 INT8 GEMM** with per-row dynamic activation quantization.
- **Static arenas** — no heap allocation inside the token generation loop.
- **OpenMP** row/column parallel GEMM with SMT-aware thread cap (`OMP_NUM_THREADS`).
- **Tiled lm_head** — logits computed by streaming vocab rows in blocks.
- **GGUF support (`g_mode=2`)** — native GGUF v3 loader: `Q8_0`/`BF16`/`F16`/`F32`
  dequant, `mmap` streaming, HF ↔ llama.cpp tensor-name resolution, AVX2 lm_head
  dot products. No conversion step — point `DSO_MODEL` straight at a `.gguf`.

## Build

```bash
g++ -O3 -fopenmp -std=c++17 dso_runtime.cpp -o dso_runtime
```

## Get a model

```bash
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='model', \
  allow_patterns=['config.json','tokenizer*.json','vocab.json','merges.txt','*.safetensors'])"
```

Then quantize to INT8 `.dso` (optional but recommended):

```bash
python3 quantize.py        # -> model/model.dso  (~2x smaller than safetensors)
```

## Run

`run.py` tokenizes the prompt (Qwen BPE, pure-Python), calls the engine, decodes output.

```bash
# INT8, cool (low CPU) — ~20 tok/s on a laptop
OMP_NUM_THREADS=4 DSO_MODEL=model/model.dso python3 run.py "The capital of France is" 64

# INT8, max speed
OMP_NUM_THREADS=16 DSO_MODEL=model/model.dso python3 run.py "Hello" 64

# BF16 reference (no .dso needed)
OMP_NUM_THREADS=16 python3 run.py "Hello" 64
```

### Direct engine usage

The engine reads prompt token ids (space-separated) from a file:

```bash
DSO_MODEL=model/model.dso ./dso_runtime prompt.tok 64
# set DSO_NOEOS=1 to disable early stop (benchmarking)
```

### GGUF models (g_mode=2)

Point `DSO_MODEL` at any `.gguf` — the loader auto-detects the format and switches
to `g_mode=2` (no `.dso` quantize step needed). `GGUF_DEBUG=1` dumps the full
tensor inventory (names, shapes, dtypes, byte offsets):

```bash
# bare engine
DSO_MODEL=/path/Qwen3.5-4B-gabliterated.q8_0.gguf ./dso_runtime prompt.tok 64

# dump tensor inventory
GGUF_DEBUG=1 DSO_MODEL=/path/Qwen3.5-4B-gabliterated.q8_0.gguf ./dso_runtime prompt.tok 1
```

Verified against `Qwen3.5-4B-gabliterated.q8_0.gguf` (arch `qwen35`, 426 tensors):
the v3 parser, `Q8_0`/`BF16`/`F16`/`F32` dequant, and lm_head dot products all run
without crashes. Full generation for that model is pending the hybrid
SSM+Attention (Qwen3.5) architecture port — see Architecture notes.

## Benchmarks (Qwen2.5-0.5B-Instruct, this machine — 16-core x86, 14 GB RAM)

| Mode | Threads | tok/s | CPU load |
|------|---------|-------|----------|
| BF16 (safetensors) | 16 | 4.7 | 100% |
| INT8 (scalar GEMM) | 16 | 18.8 | 100% |
| INT8 + AVX2 | 4 | 20.5 | ~25% |
| INT8 + AVX2 | 16 | 24.6 | 100% |

### GGUF load (Qwen3.5-4B-gabliterated.q8_0.gguf, 426 tensors)

| Stage | Result |
|-------|--------|
| GGUF v3 header + tensor index parse | OK (no crash) |
| `Q8_0`/`BF16`/`F16`/`F32` dequant | OK |
| lm_head Q8_0 dot product | OK |
| Full generation | blocked — `qwen35` hybrid SSM+Attention not yet ported |

## Architecture notes

- `dso_runtime.cpp` — engine: mmap loader, arenas, RMSNorm / RoPE / GQA attention /
  SwiGLU, INT8 + AVX2 GEMM, async streaming worker. GGUF `g_mode=2` path (v3 parser,
  `Q8_0`/`BF16`/`F16`/`F32` dequant, HF↔llama.cpp name map, lm_head dot products).
  Full `qwen35` hybrid SSM+Attention forward pass is the next port (selective-scan
  SSM + sparse attention every 4th layer + multi-section RoPE).
- `tok.py` — Qwen2 ByteLevel BPE tokenizer (pure Python, no `transformers` needed).
- `quantize.py` — produces the custom `.dso` format (int8 weights + per-row fp32 scale).
- `run.py` — CLI wrapper (tokenize → engine → decode).

The `.dso` format: 8-byte header length + JSON header (per-tensor `kind`/`shape`/`off`/
`nbytes`) + concatenated blobs. INT8 tensors store `int8[q]` then `float32[scale]`,
dequantized as `value ≈ scale[j] * q[j]`.
