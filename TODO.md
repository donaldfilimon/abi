# Project TODO List

This file tracks incomplete or placeholder implementations in the codebase. Each entry includes the file path, line number, and a brief description of the pending work.

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

The following high‑level items are still open in **[ROADMAP.md](ROADMAP.md)**. They are added here to surface future work for the team.

| File | Line | Description |
|------|------|-------------|
| `ROADMAP.md` | 107‑113 | Tooling: Debugger integration, Performance profiler, Memory leak detector |
| `ROADMAP.md` | 119‑124 | Documentation: Comprehensive API docs (auto‑generated, tutorials, videos) |
| `ROADMAP.md` | 125‑127 | Documentation: Architecture diagrams (system, component, data flow) |
| `ROADMAP.md` | 139‑140 | Testing: Competitive benchmarks |
| `ROADMAP.md` | 152‑155 | High Availability: Failover mechanisms (automatic failover, health checks, circuit breakers) |
| `ROADMAP.md` | 156‑158 | High Availability: Disaster recovery (backup orchestration, point‑in‑time recovery, multi‑region support) |
| `ROADMAP.md` | 168‑170 | Ecosystem: Package manager integration (Zig registry, Homebrew formula, Docker images) |
| `ROADMAP.md` | 184‑186 | Research & Innovation: Experimental hardware acceleration (FPGA, ASIC), novel index structures, AI‑optimized workloads |
| `ROADMAP.md` | 187‑189 | Academic collaborations (research partnerships, paper publications, conference presentations) |
| `ROADMAP.md` | 199‑201 | Community governance: RFC process, voting mechanism, contribution recognition |
| `ROADMAP.md` | 202‑204 | Education: Training courses, certification program, university partnerships |
| `ROADMAP.md` | 216‑218 | Commercial support: SLA offerings, priority support, custom development |
 | `ROADMAP.md` | 219‑221 | Cloud integration: AWS Lambda, Google Cloud Functions, Azure Functions |
| `ROADMAP.md` | 222‑225 | Commercial support: SLA offerings, priority support, custom development |

## Llama‑CPP Parity Tasks (Zig 0.16)

| Area | Status | Description | Target File(s) |
|------|--------|-------------|----------------|
| Model I/O | ✅ | GGUF loader, metadata parsing, `load_model` API. | `src/features/ai/llm/io/gguf.zig` |
| Quantization | ✅ | Q4_0, Q4_1, Q8_0 tensor decoders with roundtrip tests. | `src/features/ai/llm/tensor/quantized.zig` |
| Tokenizer | ⚠️ | BPE implemented; SentencePiece pending. | `src/features/ai/llm/tokenizer/` |
| CPU Inference | ✅ | MatMul, attention, RMSNorm kernels with SIMD. | `src/features/ai/llm/ops/` |
| GPU Backends | ⚠️ | CUDA/Vulkan backends exist; unified inference path pending. | `src/compute/gpu/` |
| Sampling | ✅ | Top-k, top-p, temperature, tail-free, mirostat (v1/v2). | `src/features/ai/llm/generation/sampler.zig` |
| Streaming | ⚠️ | Basic generation implemented; async streaming pending. | `src/features/ai/llm/generation/` |
| CLI | ✅ | Full llama-cpp CLI parity (info, generate, chat, bench). | `tools/cli/commands/llm.zig` |
| Library API | ❌ | C-compatible functions pending. | `src/abi.zig` |
| Tests & Benchmarks | ⚠️ | Module tests exist; llama-cpp reference vectors pending. | `src/features/ai/llm/` |

**Legend:** ✅ Complete | ⚠️ Partial | ❌ Not Started

Developers should prioritize these items to achieve functional parity with llama‑cpp while maintaining Zig 0.16 conventions.
