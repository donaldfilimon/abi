# Project TODO List

This file tracks incomplete or placeholder implementations in the codebase. Each entry includes the file path, line number, and a brief description of the pending work.

## Zig TODO/FIXME markers

| File | Line | Description |
|------|------|-------------|
| `tools/cli/commands/embed.zig` | 193 | TODO: Use custom model when specified |
| `tools/cli/commands/explore.zig` | 220 | Placeholder "TODO" in help text |
| `src/features/ai/abbey/neural/attention.zig` | 159 | TODO: Implement proper multi‑head splitting |
| `src/features/database/formats/converters.zig` | 64‑65 | TODO: Implement write support for GGUF and NPZ formats |
| `src/features/ai/explore/explore_test.zig` | 101 | Test uses a "TODO" query string (placeholder) |
| `src/features/database/formats/streaming.zig` | 203 | TODO: Count metadata entries |
| `src/features/ai/explore/query.zig` | 176‑179 | TODO/FIXME patterns added to query list |

Developers should prioritize implementing these items to reduce stub behavior and improve feature completeness.

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

| Area | Description | Target File(s) |
|------|-------------|----------------|
| Model I/O | Implement full GGUF loader, parse metadata, and expose `load_model` API. | `src/features/ai/llm/loader.zig` |
| Quantization | Add Q4_0, Q4_1, Q8_0 tensor decoders and verification against llama‑cpp reference tensors. | `src/features/ai/llm/quant.zig` |
| Tokenizer | Port BPE/SentencePiece tokenizer with `tokenize` and `detokenize` functions. | `src/shared/utils/tokenizer.zig` |
| CPU Inference | Write matmul, attention, RMSNorm kernels using SIMD; match llama‑cpp accuracy and speed. | `src/features/ai/llm/infer_cpu.zig` |
| GPU Backends | Wrap existing CUDA/Vulkan kernels; expose unified GPU inference path. | `src/features/compute/gpu/` |
| Sampling | Implement top‑k, top‑p, temperature, and tail‑free sampling strategies. | `src/features/ai/llm/sampler.zig` |
| Streaming | Provide async token streaming via `std.Io.Threaded` channels. | `src/features/ai/llm/stream.zig` |
| CLI | Mirror llama‑cpp CLI options (model path, prompt, generation params). | `tools/cli/commands/llama.zig` |
| Library API | Export stable C‑compatible functions (`llama_cpp_*`) matching llama‑cpp signatures. | `src/abi.zig` |
| Tests \& Benchmarks | Port llama‑cpp test vectors; add performance regression suite. | `tests/llama/` |

Developers should prioritize these items to achieve functional parity with llama‑cpp while maintaining Zig 0.16 conventions.
