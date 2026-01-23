---
title: "TODO"
tags: []
---
# Project TODO List
> **Codebase Status:** Synced with repository as of 2026-01-23.

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

The following items are tracked in **[ROADMAP.md](ROADMAP.md)**:

### Next Release (v0.8.0)

| Category | Item | Status |
|----------|------|--------|
| Performance | Benchmark baseline refresh | ⬜ Not Started |
| Ecosystem | Python bindings expansion | ⬜ Not Started |
| Ecosystem | npm package for WASM bindings | ⬜ Not Started |
| Features | Streaming inference API improvements | ⬜ Not Started |
| Features | Multi-model orchestration | ⬜ Not Started |
| Tooling | VS Code extension for ABI development | ⬜ Not Started |
| Tooling | Interactive benchmark dashboard | ⬜ Not Started |

### Later (2026-2027)

| Category | Item | Status |
|----------|------|--------|
| Research | Hardware acceleration (FPGA, ASIC exploration) | ⬜ Future |
| Research | Novel vector index structures | ⬜ Future |
| Research | Zig std.gpu native integration (when stable) | ⬜ Future |
| Ecosystem | Cloud function adapters (AWS Lambda, GCP, Azure) | ⬜ Future |
| Ecosystem | Kubernetes operator | ⬜ Future |
| Community | RFC process formalization | ⬜ Future |
| Community | Contributor certification program | ⬜ Future |

## Multi-Persona System TODOs

See **[docs/architecture/multi-persona-roadmap.md](docs/architecture/multi-persona-roadmap.md)** for detailed implementation tasks:

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Core persona types and registry | ✅ Complete |
| Phase 2 | ABI persona (sentiment, policy, routing) | ✅ Complete |
| Phase 3 | Persona embeddings and learning | ✅ Complete |
| Phase 4 | Abbey persona (emotion, empathy) | ✅ Complete |
| Phase 5 | Aviva persona (knowledge, code, facts) | ✅ Complete |
| Phase 6 | Metrics & Observability | ✅ Complete |
| Phase 7 | Load Balancing & Resilience | ✅ Complete |
| Phase 8 | API & Integration | ✅ Complete |
| Phase 9 | Testing & Validation | ✅ Complete |
| Phase 10 | Documentation & Release | ✅ Complete |

**Implementation complete as of 2026-01-23.** All persona modules, API endpoints, tests, and documentation are in place.

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
| Library API | ✅ | C-compatible API (llama_model_*, llama_context_*, tokenize, generate). | `bindings/c/abi_llm.zig` |
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

**New Architecture:**
```
src/
├── abi.zig              # Public API entry point
├── config.zig           # Unified configuration (Builder pattern)
├── framework.zig        # Framework orchestration
├── runtime/             # Always-on infrastructure
├── gpu/                 # GPU acceleration (moved from compute/)
├── ai/                  # AI features (llm, embeddings, agents, training)
├── database/            # Vector database
├── network/             # Distributed networking
├── observability/       # Metrics, tracing, logging
├── web/                 # Web utilities
├── internal/            # Internal utilities (from shared/)
├── core/                # Core I/O and collections
├── compute/             # Runtime and concurrency (GPU removed)
└── features/            # Legacy feature modules (being migrated)
```

## Remaining Work

All major implementation tasks are complete. See ROADMAP.md for future enhancements.

## Recently Completed

| Area | Description | Target File(s) |
|------|-------------|----------------|
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

| Area | Description | Status |
|------|-------------|--------|
| Tooling | Debugger integration, performance profiler, memory‑leak detector | ✅ Complete |
| Documentation | Comprehensive API docs, tutorials, architecture diagrams | ✅ Complete |
| Testing | Competitive benchmarks | ✅ Complete |
| High Availability | Failover mechanisms, health checks, circuit breakers | ✅ Complete |
| Disaster Recovery | Backup orchestration, point‑in‑time recovery, multi‑region support | ✅ Complete |
| Ecosystem | Package manager integration (Zig registry, Homebrew, Docker) | ✅ Complete |
| Modular Refactor | Unified config, Framework orchestration, top-level modules | ✅ Complete |
| Performance | Benchmark baseline refresh after consolidation | ⬜ Next |
| Ecosystem | Python bindings expansion, npm WASM package | ⬜ Next |
| Tooling | VS Code extension, interactive benchmark dashboard | ⬜ Next |
| Research | Hardware acceleration (FPGA, ASIC), novel index structures | ⬜ Future |
| Cloud | AWS Lambda, GCP Functions, Azure Functions, Kubernetes | ⬜ Future |
| Community | RFC process, contributor certification | ⬜ Future |

### Llama‑CPP Parity Overview (✅)

| Area | Description |
|------|-------------|
| Model I/O | GGUF loader, metadata parsing, `load_model` API |
| Quantization | Q4_0, Q4_1, Q8_0, Q5_0, Q5_1 tensor decoders |
| Tokenizer | BPE and SentencePiece (Viterbi) |
| CPU Inference | MatMul, attention, RMSNorm kernels with SIMD |
| GPU Backends | CUDA/cuBLAS kernels for matmul, activation, softmax, RMSNorm, SiLU |
| Sampling | Top‑k, top‑p, temperature, tail‑free, mirostat (v1/v2) |
| Streaming | Async SSE streaming, callbacks, cancellation |
| CLI | Full llama‑cpp CLI parity (info, generate, chat, bench) |
| Library API | C‑compatible API (llama_model_*, llama_context_*, tokenize, generate) |
| Tests & Benchmarks | Reference vectors for Q4/Q5/Q8, softmax, RMSNorm, SiLU, MatMul, attention |
| Training | Backward ops, loss, trainable model, LoRA, mixed precision |
| Gradient Checkpointing | Memory‑efficient training with selective activation storage |

### Miscellaneous Implementation TODOs

* ~~Review any remaining `TODO:`/`FIXME:` markers in the source tree and document them here.~~ ✅ All code-level TODOs complete.
* ~~Ensure all feature‑gated stub modules correctly return `error.*Disabled`.~~ ✅ All stubs verified and tested (2026-01-17).
* ~~Update documentation links throughout the repo to reference this **Claude‑Code Massive TODO** for visibility.~~ ✅ Complete

## Stub API Parity (2026-01-17)

All feature stubs have been updated to match real implementations and tested with feature flags:

| Feature | Stub File | Status | Notes |
|---------|-----------|--------|-------|
| AI | `src/ai/stub.zig` | ✅ | Fixed SessionData, SessionMeta, PromptBuilder, TrainingConfig, TrainingReport, TrainingResult, Checkpoint, TrainableModelConfig, TrainableModel |
| LLM | `src/ai/llm/stub.zig` | ✅ | Added matrixMultiply to ops, GgufFile.printSummaryDebug |
| GPU | `src/gpu/stub.zig` | ✅ | Added backendAvailability export |
| Network | `src/network/stub.zig` | ✅ | Added touch(), setStatus(), fixed NodeInfo.last_seen_ms, corrected NodeStatus enum |
| Database | `src/database/stub.zig` | ✅ | Verified (no changes needed) |
| Web | `src/shared/web/stub.zig` | ✅ | Verified (no changes needed) |
| Profiling | `src/compute/profiling/stub.zig` | ✅ | Verified (no changes needed) |

**Build Verification:**
- ✅ `zig build -Denable-ai=true` - Passes
- ✅ `zig build -Denable-gpu=true` - Passes
- ✅ `zig build -Denable-network=true` - Passes
- ✅ `zig build -Denable-database=true` - Passes
- ✅ `zig build -Denable-web=true` - Passes
- ✅ `zig build -Denable-profiling=true` - Passes

