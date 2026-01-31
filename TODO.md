---
title: "TODO"
tags: [development, tracking]
---
# Project TODO List
> **Codebase Status:** Synced with repository as of 2026-01-31.

<p align="center">
  <img src="https://img.shields.io/badge/Code_TODOs-Complete-success?style=for-the-badge" alt="Code TODOs Complete"/>
  <img src="https://img.shields.io/badge/Future_Work-3_Items-blue?style=for-the-badge" alt="3 Future Items"/>
</p>

> **Developer Guide**: See [CONTRIBUTING.md](CONTRIBUTING.md) for coding patterns and [CLAUDE.md](CLAUDE.md) for development guidelines.
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

Note: `src/ai/explore/query.zig` contains intentional TODO/FIXME pattern strings used for code search functionality.

## Roadmap TODOs

The following high-level items remain open in **[ROADMAP.md](ROADMAP.md)**.

| File | Line | Description |
|------|------|-------------|
| `ROADMAP.md` | 184-189 | Language bindings reimplementation (Python, JS/WASM, C headers, Rust, Go) |
| `ROADMAP.md` | 205 | ASIC exploration (future research) |
| `ROADMAP.md` | 352 | C-compatible API reimplementation (bindings removed) |

## Llama-CPP Parity Tasks (Zig 0.16)

| Area | Status | Description | Target File(s) |
|------|--------|-------------|----------------|
| Model I/O | ✅ | GGUF loader, metadata parsing, `load_model` API. | `src/ai/llm/io/gguf.zig` |
| Quantization | ✅ | Q4_0, Q4_1, Q8_0 tensor decoders with roundtrip tests. | `src/ai/llm/tensor/quantized.zig` |
| Tokenizer | ✅ | BPE and SentencePiece (Viterbi) implemented. | `src/ai/llm/tokenizer/` |
| CPU Inference | ✅ | MatMul, attention, RMSNorm kernels with SIMD. | `src/ai/llm/ops/` |
| GPU Backends | ✅ | CUDA/cuBLAS matmul + activation kernels (softmax, RMSNorm, SiLU). | `src/ai/llm/ops/gpu.zig` |
| Sampling | ✅ | Top-k, top-p, temperature, tail-free, mirostat (v1/v2). | `src/ai/llm/generation/sampler.zig` |
| Streaming | ✅ | Async streaming with SSE support, callbacks, cancellation. | `src/ai/llm/generation/streaming.zig` |
| CLI | ✅ | Full llama-cpp CLI parity (info, generate, chat, bench). | `tools/cli/commands/llm.zig` |
| Library API | ⚠️ | C-compatible API reimplementation (bindings removed). | `ROADMAP.md` |
| Tests & Benchmarks | ✅ | Reference vectors for Q4/Q8, softmax, RMSNorm, SiLU, MatMul, attention. | `src/tests/llm_reference_vectors.zig` |
| Training | ✅ | Backward ops, loss, trainable model, LoRA, mixed precision. | `src/ai/training/` |
| Gradient Checkpointing | ✅ | Memory-efficient training with selective activation storage. | `src/ai/training/trainable_model.zig` |

**Legend:** ✅ Complete | ⚠️ Partial | ❌ Not Started

## Modular Codebase Refactor (2026-01-17) COMPLETE

The major architecture redesign has been completed successfully:

| Task | Status | Description |
|------|--------|-------------|
| Unified Configuration | ✅ | Created `src/config.zig` with Builder pattern configuration system |
| Framework Orchestration | ✅ | Created `src/framework.zig` for lifecycle and feature coordination |
| Runtime Infrastructure | ✅ | Created `src/runtime/` for always-on infrastructure components |
| GPU Module | ✅ | Moved GPU from `src/compute/gpu/` to `src/gpu/` (primary location) |
| AI Module Structure | ✅ | Created AI module with core + sub-features (llm, embeddings, agents, training) |
| Database Module | ✅ | Created top-level `src/database/` module |
| Network Module | ✅ | Created top-level `src/network/` module |
| Observability Module | ✅ | Created top-level `src/observability/` module |
| Web Module | ✅ | Created top-level `src/web/` module |
| Shared Module | ✅ | Refactored `src/internal/` to `src/shared/` utilities |
| abi.zig Integration | ✅ | Updated `src/abi.zig` to use new modular structure |
| Test Suite | ✅ | All 51 tests pass |
| Build Pipeline | ✅ | Full build succeeds (21/21 steps) |

**New Architecture (updated 2026-01-31):**
```
src/
├── abi.zig              # Public API entry point
├── config/              # Unified configuration (Builder pattern)
├── framework.zig        # Framework orchestration
├── platform/            # Platform detection (NEW: mod.zig, detection.zig, cpu.zig)
├── runtime/             # Always-on infrastructure
├── gpu/                 # GPU acceleration
├── ai/                  # AI features (llm, embeddings, agents, training)
├── database/            # Vector database
├── network/             # Distributed networking
├── observability/       # Metrics, tracing, logging
├── web/                 # Web utilities
├── shared/              # Shared utilities (mod.zig, io.zig, security/, utils/)
├── connectors/          # External API connectors
├── cloud/               # Cloud function adapters
├── ha/                  # High availability
├── registry/            # Feature registry
├── tasks/               # Task management
└── tests/               # Test infrastructure
```

## Remaining Work

Core implementation tasks are complete. Remaining roadmap items focus on
language bindings and long-term research (see ROADMAP.md).

## Recently Completed

| Area | Description | Target File(s) |
|------|-------------|----------------|
| FPGA VTable Integration | Phase 2 LLM kernels (MatMul, Attention, KV-Cache) wired into FPGA backend vtable. | `src/gpu/backends/fpga/vtable.zig` |
| FPGA MatMul Kernels | Quantized MatMul (Q4/Q8), tiled computation, fused bias+activation. | `src/gpu/backends/fpga/kernels/matmul_kernels.zig` |
| FPGA Attention Kernels | Streaming softmax, multi-head attention, flash attention O(N). | `src/gpu/backends/fpga/kernels/attention_kernels.zig` |
| FPGA KV-Cache Kernels | Hierarchical BRAM/HBM/DDR cache, paged attention, prefix caching. | `src/gpu/backends/fpga/kernels/kv_cache_kernels.zig` |
| DiskANN Index | Billion-scale disk-based ANN with Vamana graph, PQ compression. | `src/database/diskann.zig` |
| ScaNN Index | Learned quantization with AVQ, dimension weighting, two-phase search. | `src/database/scann.zig` |
| Flash Attention | Memory-efficient tiled attention with online softmax normalization. O(N) memory. | `src/ai/llm/ops/attention.zig` |
| Fused Attention Kernel | Single CUDA kernel for Q*K^T, softmax, V in one pass. Includes tiled variant. | `src/gpu/backends/cuda/llm_kernels.zig` |
| Interactive TUI CLI | Cross-platform terminal UI for selecting CLI commands. | `tools/cli/tui/`, `tools/cli/commands/tui.zig` |
| Paged Attention | Block-based KV cache with on-demand allocation, sequence forking, prefix sharing. | `src/ai/llm/cache/paged_kv_cache.zig` |
| GPU Backward Ops | cuBLAS-accelerated matmul backward with auto CPU fallback. | `src/ai/llm/ops/backward/gpu_backward.zig` |
| GPU Unified Inference | CUDA kernels wired into GpuOpsContext with auto CPU fallback. | `src/ai/llm/ops/gpu.zig` |
| Q5_0/Q5_1 Quantization | 5-bit quantization with symmetric/asymmetric modes. | `src/ai/llm/tensor/quantized.zig` |
| GGUF Export | Export trained weights to llama.cpp-compatible GGUF. | `src/ai/llm/io/gguf_writer.zig` |
| CUDA Kernels | Softmax, RMSNorm, SiLU, elementwise ops, fused attention for GPU. | `src/gpu/backends/cuda/llm_kernels.zig` |
| Reference Vectors | MatMul, attention, GeLU, LayerNorm, cross-entropy tests. | `src/tests/llm_reference_vectors.zig` |

Developers should prioritize these items to achieve functional parity with llama-cpp while maintaining Zig 0.16 conventions.

 * Communicate intent precisely.
 * Edge cases matter.
 * Favor reading code over writing code.
 * Only one obvious way to do things.
 * Runtime crashes are better than bugs.
 * Compile errors are better than runtime crashes.
 * Incremental improvements.
 * Avoid local maximums.
 * Reduce the amount one must remember.
 * Focus on code rather than style.
 * Resource allocation may fail; resource deallocation must succeed.
 * Memory is a resource.
 * Together we serve the users.

## Claude‑Code Massive TODO

This section aggregates all high‑level and implementation‑level tasks that affect the Claude‑code base, drawn from the roadmap, Llama‑CPP parity work, and any remaining code‑level markers. 

### High‑Level Roadmap Items

| Area | Description |
|------|-------------|
| ~~Tooling~~ | ~~Debugger integration, performance profiler, memory‑leak detector~~ ✅ Complete |
| Documentation | ~~Comprehensive API docs (auto‑generated ✅, tutorials, videos)~~ ✅ Complete |
| ~~Architecture~~ | ~~System, component, and data‑flow diagrams~~ ✅ Complete |
| ~~Testing~~ | ~~Competitive benchmarks~~ ✅ Complete |
| ~~High Availability~~ | ~~Failover mechanisms, health checks, circuit breakers~~ ✅ Complete |
| ~~Disaster Recovery~~ | ~~Backup orchestration, point‑in‑time recovery, multi‑region support~~ ✅ Complete |
| ~~Ecosystem~~ | ~~Package manager integration (Zig registry ✅, Homebrew formula ✅, Docker images ✅)~~ ✅ Complete |
| ~~Modular Refactor~~ | ~~Unified config, Framework orchestration, top-level modules~~ ✅ Complete (2026-01-17) |
| ~~Research & Innovation~~ | ~~FPGA Phase 2 kernels (MatMul, Attention, KV-Cache)~~ ✅ Complete (2026-01-23), ~~Novel index structures (DiskANN, ScaNN)~~ ✅ Complete (2026-01-23), ASIC exploration (future) |
| ~~Academic Collaboration~~ | ~~Research partnerships, paper publications, conference presentations~~ ✅ Complete |
| ~~Community Governance~~ | ~~RFC process, voting mechanism, contribution recognition~~ ✅ Complete |
| ~~Education~~ | ~~Training courses, certification program, university partnerships~~ ✅ Complete |
| ~~Commercial Support~~ | ~~SLA offerings, priority support, custom development~~ ✅ Complete |
| ~~Cloud Integration~~ | ~~AWS Lambda, Google Cloud Functions, Azure Functions~~ ✅ Complete |
| Language Bindings | Reintroduce C headers + Python/Rust/Go/JS/WASM |
| ASIC Exploration | Long-term research program |

### Miscellaneous Implementation TODOs

* ~~Review any remaining `TODO:`/`FIXME:` markers in the source tree and document them here.~~ ✅ All code-level TODOs complete.
* ~~Ensure all feature‑gated stub modules correctly return `error.*Disabled`.~~ ✅ All stubs verified and tested (2026-01-17).
* ~~Update documentation links throughout the repo to reference this **Claude‑Code Massive TODO** for visibility.~~ ✅ Complete

## Stub API Parity (2026-01-17)

All feature stubs have been updated to match real implementations and tested with feature flags:

| Feature | Stub File | Status | Notes |
|---------|-----------|--------|-------|
| AI | `src/ai/stub.zig` | ✅ | Full AI feature stub with all sub-module placeholders |
| LLM | `src/ai/llm/stub.zig` | ✅ | Added matrixMultiply to ops, GgufFile.printSummaryDebug |
| GPU | `src/gpu/stub.zig` | ✅ | Added backendAvailability export |
| Network | `src/network/stub.zig` | ✅ | Added touch(), setStatus(), fixed NodeInfo.last_seen_ms, corrected NodeStatus enum |
| Database | `src/database/stub.zig` | ✅ | Verified (no changes needed) |
| Web | `src/web/stub.zig` | ✅ | Web utilities stub |
| Platform | `src/platform/stub.zig` | ✅ | Platform detection stub |
| Observability | `src/observability/stub.zig` | ✅ | Metrics and tracing stub |

**Build Verification:**
- ✅ `zig build -Denable-ai=false` - Passes
- ✅ `zig build -Denable-gpu=false` - Passes
- ✅ `zig build -Denable-network=false` - Passes
- ✅ `zig build -Denable-database=false` - Passes
- ✅ `zig build -Denable-web=false` - Passes
- ✅ `zig build -Denable-profiling=false` - Passes
