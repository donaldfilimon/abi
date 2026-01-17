# Project TODO List

> **Developer Guide**: See [AGENTS.md](AGENTS.md) for coding patterns and [CLAUDE.md](CLAUDE.md) for development guidelines.
>
> This file tracks incomplete or placeholder implementations in the codebase. Each entry includes a file path, line number, and a brief description of the pending work.

## Zig TODO/FIXME markers

All previously tracked code-level TODOs have been completed:
- ✅ `embed.zig` - Custom model support implemented
- ✅ `explore.zig` - Help text placeholder fixed
- ✅ `attention.zig` - Multi-head splitting implemented
- ✅ `converters.zig` - GGUF and NPZ write support added
- ✅ `explore_test.zig` - Placeholder query string fixed
- ✅ `streaming.zig` - Metadata counting implemented

Note: `src/features/ai/explore/query.zig` contains intentional TODO/FIXME pattern strings used for code search functionality.

## Roadmap TODOs

The following high-level items are still open in **[ROADMAP.md](ROADMAP.md)**. They are added here to surface future work for the team.

| File | Line | Description |
|------|------|-------------|
| `ROADMAP.md` | 86-89 | Tooling: Debugger integration, Performance profiler, Memory leak detector |
| `ROADMAP.md` | 92-95 | Documentation: Comprehensive API docs (auto-generated, tutorials, videos) |
| `ROADMAP.md` | 96-99 | Documentation: Architecture diagrams (system, component, data flow) |
| `ROADMAP.md` | 109 | Testing: Competitive benchmarks |
| `ROADMAP.md` | 125-129 | High Availability: Failover mechanisms (automatic failover, health checks, circuit breakers) |
| `ROADMAP.md` | 130-133 | High Availability: Disaster recovery (backup orchestration, point-in-time recovery, multi-region support) |
| `ROADMAP.md` | 140-143 | Ecosystem: Package manager integration (Zig registry, Homebrew formula, Docker images) |
| `ROADMAP.md` | 148-151 | Research & Innovation: Experimental hardware acceleration (FPGA, ASIC), novel index structures, AI-optimized workloads |
| `ROADMAP.md` | 152-155 | Academic collaborations (research partnerships, paper publications, conference presentations) |
| `ROADMAP.md` | 158-161 | Community governance: RFC process, voting mechanism, contribution recognition |
| `ROADMAP.md` | 162-165 | Education: Training courses, certification program, university partnerships |
| `ROADMAP.md` | 168-171 | Commercial support: SLA offerings, priority support, custom development |
| `ROADMAP.md` | 172-175 | Cloud integration: AWS Lambda, Google Cloud Functions, Azure Functions |

## Llama-CPP Parity Tasks (Zig 0.16)

| Area | Status | Description | Target File(s) |
|------|--------|-------------|----------------|
| Model I/O | ✅ | GGUF loader, metadata parsing, `load_model` API. | `src/features/ai/llm/io/gguf.zig` |
| Quantization | ✅ | Q4_0, Q4_1, Q8_0 tensor decoders with roundtrip tests. | `src/features/ai/llm/tensor/quantized.zig` |
| Tokenizer | ✅ | BPE and SentencePiece (Viterbi) implemented. | `src/features/ai/llm/tokenizer/` |
| CPU Inference | ✅ | MatMul, attention, RMSNorm kernels with SIMD. | `src/features/ai/llm/ops/` |
| GPU Backends | ✅ | CUDA/cuBLAS matmul + activation kernels (softmax, RMSNorm, SiLU). | `src/features/ai/llm/ops/gpu.zig` |
| Sampling | ✅ | Top-k, top-p, temperature, tail-free, mirostat (v1/v2). | `src/features/ai/llm/generation/sampler.zig` |
| Streaming | ✅ | Async streaming with SSE support, callbacks, cancellation. | `src/features/ai/llm/generation/streaming.zig` |
| CLI | ✅ | Full llama-cpp CLI parity (info, generate, chat, bench). | `tools/cli/commands/llm.zig` |
| Library API | ✅ | C-compatible API (llama_model_*, llama_context_*, tokenize, generate). | `bindings/c/abi_llm.zig` |
| Tests & Benchmarks | ✅ | Reference vectors for Q4/Q8, softmax, RMSNorm, SiLU, MatMul, attention. | `src/tests/llm_reference_vectors.zig` |
| Training | ✅ | Backward ops, loss, trainable model, LoRA, mixed precision. | `src/features/ai/training/` |
| Gradient Checkpointing | ✅ | Memory-efficient training with selective activation storage. | `src/features/ai/training/trainable_model.zig` |

**Legend:** ✅ Complete | ⚠️ Partial | ❌ Not Started

## Remaining Work

All major implementation tasks are complete. See ROADMAP.md for future enhancements.

## Recently Completed

| Area | Description | Target File(s) |
|------|-------------|----------------|
| Flash Attention | Memory-efficient tiled attention with online softmax normalization. O(N) memory. | `src/features/ai/llm/ops/attention.zig` |
| Fused Attention Kernel | Single CUDA kernel for Q*K^T, softmax, V in one pass. Includes tiled variant. | `src/compute/gpu/backends/cuda/llm_kernels.zig` |
| Interactive TUI CLI | Cross-platform terminal UI for selecting CLI commands. | `tools/cli/tui/`, `tools/cli/commands/tui.zig` |
| Paged Attention | Block-based KV cache with on-demand allocation, sequence forking, prefix sharing. | `src/features/ai/llm/cache/paged_kv_cache.zig` |
| GPU Backward Ops | cuBLAS-accelerated matmul backward with auto CPU fallback. | `src/features/ai/llm/ops/backward/gpu_backward.zig` |
| GPU Unified Inference | CUDA kernels wired into GpuOpsContext with auto CPU fallback. | `src/features/ai/llm/ops/gpu.zig` |
| Q5_0/Q5_1 Quantization | 5-bit quantization with symmetric/asymmetric modes. | `src/features/ai/llm/tensor/quantized.zig` |
| GGUF Export | Export trained weights to llama.cpp-compatible GGUF. | `src/features/ai/llm/io/gguf_writer.zig` |
| CUDA Kernels | Softmax, RMSNorm, SiLU, elementwise ops, fused attention for GPU. | `src/compute/gpu/backends/cuda/llm_kernels.zig` |
| Reference Vectors | MatMul, attention, GeLU, LayerNorm, cross-entropy tests. | `src/tests/llm_reference_vectors.zig` |

Developers should prioritize these items to achieve functional parity with llama-cpp while maintaining Zig 0.16 conventions.
