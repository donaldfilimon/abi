---
title: Roadmap
description: Version history and future plans
section: Reference
order: 6
---

# Roadmap

This page covers the version history of the ABI framework, the current state
of the project, and planned future work.

---

## Version History

### v0.1.0 -- Foundation

- Initial Zig framework with comptime feature gating
- GPU module with CUDA and Vulkan backends
- SIMD-accelerated vector operations
- WDBX vector database with HNSW indexing
- Basic AI module with LLM inference
- Framework lifecycle state machine

### v0.2.0 -- Expansion

- Added 6 new feature modules: cache, storage, search, messaging, gateway, pages
- Distributed network module with Raft consensus
- High availability: replication, backup, PITR
- Observability: metrics, tracing, profiling
- Analytics event tracking
- Cloud serverless adapters (AWS, GCP, Azure)
- Authentication and security infrastructure (17 modules)
- Mobile platform support (opt-in)

### v0.3.0 -- AI Split and Connectors

- Split monolithic AI module into 5 independent modules:
  - `ai` (monolith), `ai_core`, `inference`, `training`, `reasoning`
- Added 9 LLM provider connectors: OpenAI, Anthropic, Ollama, HuggingFace,
  Mistral, Cohere, LM Studio, vLLM, MLX
- Discord REST client integration
- Job scheduler connector
- Abbey advanced reasoning engine (meta-learning, self-reflection, theory of mind)
- MCP server (JSON-RPC 2.0 over stdio for WDBX)
- ACP server (agent communication protocol)
- Builder pattern for framework configuration
- 36 C API bindings

### v0.4.0 -- Current Release

- Full Zig 0.16 migration (compiles and passes all tests)
- vNext staged-compatibility API surface
- v2 runtime primitives: Channel, ThreadPool, DagPipeline
- v2 shared utilities: SwissMap, ArenaPool, structured errors
- Shared resilience module (parameterized circuit breakers)
- Consolidated rate limiters
- Radix tree routing shared between gateway and pages
- 10 GPU backends (added WebGL2, OpenGL, OpenGLES, FPGA, simulated)
- TPU backend support (runtime-linked)
- 30 CLI commands + 8 aliases
- 36 examples
- Ralph iterative agent with skill memory and multi-Ralph bus
- Baseline validation scripts and CI quality gates
- Full test suite: 1270 pass, 5 skip (main), 1535 pass (feature)

---

## Current State

| Metric | Value |
|--------|-------|
| Version | 0.4.0 |
| Zig version | 0.16.0-dev.2611+f996d2866 |
| Feature modules | 21 (17 core + 4 AI split) |
| Services | 9 always-available (runtime, platform, shared, connectors, ha, tasks, mcp, acp, simd) |
| GPU backends | 10 (CUDA, Vulkan, Metal, WebGPU, TPU, stdgpu, WebGL2, OpenGL, OpenGLES, FPGA) + simulated |
| LLM connectors | 9 (OpenAI, Anthropic, Ollama, HuggingFace, Mistral, Cohere, LM Studio, vLLM, MLX) |
| CLI commands | 30 + 8 aliases |
| C API exports | 36 |
| Examples | 36 |
| Main tests | 1270 pass, 5 skip |
| Feature tests | 1535 pass |
| Flag combos validated | 34 |

### Zig 0.16 Migration Status

The framework compiles cleanly and passes all tests on Zig 0.16. GPU and
database backend source files compile through the named `abi` module but
cannot yet be registered in `feature_test_root.zig` for direct inline
testing -- tracked for a dedicated migration pass in v0.5.0.

---

## Planned: v0.5.0

Goals for the next minor release:

- **vNext API promotion** -- Graduate `abi.vnext.App` to primary API; deprecation
  warnings on legacy `abi.init()`
- **GPU backend test coverage** -- Register GPU backends in feature test root
  after migration pass
- **Streaming improvements** -- Enhanced SSE/WebSocket resilience, backpressure
- **Training pipeline v2** -- Distributed training with runtime.ThreadPool,
  gradient sync via Channel
- **Plugin system** -- Dynamic feature loading beyond comptime gating
- **Documentation** -- Auto-generated API docs (`abi gendocs` or `zig build gendocs`) coverage
  for all 21 modules

## Planned: v0.6.0

Longer-term goals:

- **Stable 1.0 preparation** -- API freeze, backwards compatibility guarantees
- **WASM full support** -- Database, network, and GPU stubs for browser targets
- **Multi-node orchestration** -- Production-grade distributed compute with
  network module
- **Federated learning** -- Cross-node training with privacy guarantees
- **Package manager integration** -- Publish as a Zig package for `build.zig.zon`
- **Language bindings** -- Python, Rust, and Go wrappers beyond the C API
- **Benchmark suite** -- Publish reproducible performance baselines

---

## How to Contribute

See [Contributing](contributing.html) for the development workflow, style guide,
and PR checklist. All contributions -- bug fixes, new features, documentation
improvements -- are welcome.

---

## Related Pages

- [Architecture](architecture.html) -- Module hierarchy and design
- [Configuration](configuration.html) -- Feature flags and build options
- [Contributing](contributing.html) -- Development workflow

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
