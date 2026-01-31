---
title: "PLAN"
tags: [planning, sprint, development]
---
# Current Development Focus
> **Codebase Status:** Synced with repository as of 2026-01-31.

<p align="center">
  <img src="https://img.shields.io/badge/Sprint-Complete-success?style=for-the-badge" alt="Sprint Complete"/>
  <img src="https://img.shields.io/badge/Tests-787%2F792-success?style=for-the-badge" alt="Tests"/>
</p>

## This Sprint

**Focus: Documentation & Stream Reliability - COMPLETE**

### Completed This Sprint
- [x] **Stream error recovery** - Per-backend circuit breakers, exponential backoff retry, session caching, recovery events
- [x] **Streaming integration tests** - E2E tests with fault injection for circuit breaker, session cache, metrics
- [x] **Security hardening** - JWT none algorithm warning, master key requirement option, secure API key wiping
- [x] **Streaming documentation** - Comprehensive guide for SSE/WebSocket streaming API (`docs/content/api.html`)
- [x] **Model management guide** - Documentation for downloading, caching, hot-reload (`docs/content/cli.html`)
- [x] **Metal backend enhancements** - Accelerate framework (vBLAS/vDSP/vForce), unified memory manager, zero-copy tensors

---

## Blocked

Waiting on external dependencies:

| Item | Blocker | Workaround |
|------|---------|------------|
| Native HTTP downloads | Zig 0.16 `std.Io.File.Writer` API unstable | Falls back to curl/wget instructions |
| Toolchain CLI | Zig 0.16 API incompatibilities | Command disabled; manual zig installation |

**Note:** These will be re-evaluated when Zig 0.16.1+ releases with I/O API stabilization.

---

## Next Sprint Preview

Potential focus areas for upcoming work:

- [ ] Language bindings reimplementation (Python, Rust, Go, JS/WASM, C headers)
- [ ] ASIC exploration research (long-term)
- [ ] Additional competitive benchmarks
- [ ] Community contribution tooling

---

## Recently Completed

- **GPU backend test coverage complete** - Added inline tests to ALL GPU backends: WebGPU, OpenGL, OpenGL ES, Vulkan (17 error cases), Metal (10 error cases), WebGL2, stdgpu; Verified Metal backend works (emulated mode); All CLI commands functional including nested subcommands; Training pipeline tested; 787/792 tests passing (2026-01-30)
- **Documentation cleanup** - Removed 23 redundant files: 21 deprecated api_*.md redirects, performance.md stub, gpu-backends.md duplicate; Added standardized error module (src/shared/errors.zig) with ResourceError, IoError, FeatureError, ConfigError, AuthError sets; Added inline tests to config/loader.zig and platform/detection.zig; 787/792 tests passing (2026-01-30)
- **Zig 0.16 pattern modernization** - Replaced @tagName() with {t} format specifier in print statements, converted std.ArrayList to ArrayListUnmanaged in docgen, updated std.json.stringify to std.json.fmt API; 787/792 tests passing (2026-01-30)
- **Configuration loader with env vars** - New ConfigLoader for runtime configuration via environment variables (ABI_GPU_BACKEND, ABI_LLM_MODEL_PATH, etc.); documented in CLAUDE.md; 787/792 tests passing (2026-01-30)
- **Build system improvements** - Fixed pathExists() for Zig 0.16 using C stat(); synced package version to 0.4.0 across build.zig, build.zig.zon, and all source files; cross-platform cli-tests and full-check build steps; 787/792 tests passing (2026-01-30)
- **C-compatible library bindings** - Complete FFI layer with headers (abi.h, abi_types.h, abi_errors.h), Zig exports (errors, framework, simd, database, gpu, agent), shared/static library build, C examples; Llama-CPP Library API parity now 100% complete; 787/792 tests passing (2026-01-30)
- **Codebase cleanup** - Removed unnecessary files for fresh start: bindings/ (Rust, Go, Python, WASM, C), vscode-abi/, www/, models/, .serena/, migration scripts (probe_*.zig, fix_*.py, migrate*.sh), tools/migrate_0_16/; Removed legacy plan archives; Fixed WASM build targets to gracefully no-op when bindings missing; Updated .gitignore; Updated documentation (CLAUDE.md, AGENTS.md, ROADMAP.md, TODO.md); 787/792 tests passing (2026-01-30)
- **AI stub parity complete** - Full stub/real API parity for `-Denable-ai=false` builds; Added TrainableViTConfig, TrainableViTModel, CLIPTrainingConfig, TrainableCLIPModel, VisionTrainingError, MultimodalTrainingError stubs; Fixed DownloadResult.checksum type (`[64]u8` vs optional); All feature flag combinations now compile; 787/792 tests passing (2026-01-30)
- **src/ restructure (partial)** - Created `src/platform/` module with unified platform detection (mod.zig, detection.zig, cpu.zig, stub.zig), created `src/shared/mod.zig` to consolidate utilities, moved io.zig to shared/, updated CLAUDE.md architecture diagram; 787/792 tests passing (2026-01-30)
- **GPU platform detection** - Centralized platform detection for all GPU backends (`src/gpu/platform.zig`), PlatformCapabilities for runtime feature detection, BackendSupport for compile-time availability, isCudaSupported/isMetalSupported/isVulkanSupported helpers; 787/792 tests passing (2026-01-30)
- **CUDA Zig 0.16 compatibility** - Fixed CUDA loader to work without deprecated `std.process.getEnvVarOwned` API, added allocator parameter throughout CUDA initialization chain, updated memory/mod/vtable modules to pass allocators correctly; 787/792 tests passing (2026-01-30)
- **Metal backend enhancements** - Accelerate framework integration (vBLAS/vDSP/vForce for AMX-accelerated ops), unified memory manager for zero-copy CPU/GPU sharing, UnifiedTensor type, storage mode selection, neural network primitives (softmax, rmsnorm, silu, gelu); 787/792 tests passing (2026-01-30)
- **Stream error recovery implementation** - Per-backend circuit breakers (closed/open/half_open states), exponential backoff retry with jitter, LRU session token caching for reconnection, comprehensive streaming metrics, recovery event callbacks, BackendRouter with recovery-aware routing, 503 with Retry-After when circuit open; 787/792 tests passing (2026-01-30)
- **Security hardening** - JWT none algorithm runtime warning, require_master_key config option for production, secure API key wiping with secureZero(); Addresses security audit findings H-1, H-2, M-1 (2026-01-30)
- **Zig 0.16 compilation fixes** - Fixed std.time.sleep() with Timer-based busy-wait in tests, fixed linux.getpid()/getppid() with proper platform detection for macOS/BSD (2026-01-30)
- **Model download infrastructure** - Enhanced `abi model download` with progress display infrastructure; `DownloadResult` struct (path, checksum, was_resumed, verified); `DownloadConfig` with resume/checksum options; Detailed multi-line ANSI progress bar (size, speed, ETA); `--no-verify` flag for checksum skip; Graceful fallback to curl/wget instructions; Native HTTP deferred until Zig 0.16 File I/O stabilizes; 771/776 tests passing (2026-01-26)
- **Model management CLI** - Download, cache, and manage GGUF models locally; `abi model` command with list/info/download/remove/search/path subcommands; HuggingFace shorthand (`TheBloke/Model:Q4_K_M`); Resolves download URLs; Manager tracks cached models with metadata; Platform-aware cache directories (`~/.abi/models/`); Inline tests; 771/776 tests passing (2026-01-26)
- **Streaming benchmarks** - Performance tests for streaming inference pipeline; Measures TTFT (Time To First Token), inter-token latency (P50/P90/P99), throughput (tok/s), SSE encoding overhead, WebSocket framing overhead; MockTokenGenerator with 4 patterns (constant_rate, variable_rate, burst, warmup); `abi bench streaming` CLI command; Quick/standard/comprehensive presets; 771/776 tests passing (2026-01-26)
- **Model hot-reload** - Swap GGUF models without server restart via `POST /admin/reload`; Waits for active streams to drain (30s timeout); No authentication required; No rollback on failure (leaves server without model); Uses existing `Engine.loadModelImpl()` which handles unload-before-load; 771/776 tests passing (2026-01-26)
- **SSE heartbeat system** - Timer-based keep-alive heartbeats for long-running SSE connections; Configurable `heartbeat_interval_ms` (default 15s); SSE comment format (`: heartbeat\n\n`) prevents proxy timeouts; Both OpenAI-compatible and ABI endpoints supported; Uses `std.time.Timer` for precise timing; 771/776 tests passing (2026-01-26)
- **WebSocket streaming** - Bidirectional real-time communication for `/api/stream/ws` endpoint; RFC 6455 compliant frame encoding/decoding; Multiple requests per connection; Cancellation support via `{"type":"cancel"}` messages; ABI message format with start/token/end/error events; Bearer token auth; Concurrent stream limits; 771/776 tests passing (2026-01-26)
- **True SSE streaming** - Replaced non-streaming fallback with real Server-Sent Events streaming; ConnectionContext for writer passthrough; Incremental token delivery via `data: {json}\n\n` format; OpenAI-compatible `[DONE]` termination; Custom ABI endpoint with start/token/end events; 771/776 tests passing (2026-01-26)
- **Streaming Inference API** - Real-time token streaming for LLM responses with SSE/WebSocket support; OpenAI-compatible `/v1/chat/completions` endpoint; Backend routing for local GGUF, OpenAI, Ollama, Anthropic; Bearer token auth; Heartbeat keep-alive; 770/776 tests passing (2026-01-26)
- **ArrayList to ArrayListUnmanaged modernization** - Comprehensive migration across GPU (18 files), Database (8 files), Security (10 files), AI vision, and network test modules; 762/767 tests passing (2026-01-26)
- **Complete WASM support** - Fixed all getCpuCount calls across 9 files with WASM/freestanding guards; WASM build now passes (2026-01-26)
- **gendocs module paths fix** - Updated gendocs tool to use correct paths after config module refactoring; all 22 modules now generate docs (2026-01-26)
- **build.zig.zon version field fix** - Restored missing required `.version` field that was breaking all Zig 0.16 builds (2026-01-26)
- **Database search prefetching** - Added @prefetch hints to search loop for better cache performance on large datasets (2026-01-26)
- **Engine use-after-free fix** - Fixed critical bug in executeTask/executeTaskInline where node.id was used after node destruction (2026-01-26)
- **LockFreeStackEBR re-export** - Added ABA-safe lock-free stack re-export from epoch module for production use (2026-01-26)
- **HNSW SearchStatePool improvements** - Safe type casting with overflow error, exponential backoff in CAS loop to reduce CPU contention (2026-01-26)
- **Memory leak fix in QueryUnderstanding** - Fixed freeParsedQuery() to properly free target_paths strings and slices (2026-01-26)
- **SIMD performance optimizations** - Optimized vectorReduce with @reduce(), added batchCosineSimilarityPrecomputed() for pre-computed norms (2026-01-26)
- **Toolchain CLI fix** - Temporarily disabled toolchain command due to Zig 0.16 API incompatibilities (2026-01-26)
- **Docker Compose deployment** - Added docker-compose.yml with standard and GPU service variants, Ollama integration, health checks, and .dockerignore for optimized builds (2026-01-26)
- **Test coverage improvements** - Added inline tests for multi_agent coordinator, observability monitoring (alerting), web client, OpenAI connector, HuggingFace connector, logging, plugins, network registry, and network linking modules (2026-01-26)
- **Zig 0.16 format specifier compliance** - Replaced `@tagName()` with `{t}` format specifier in CLI, GPU modules, and examples for Zig 0.16 best practices (2026-01-26)
- **Vision and CLIP CLI training commands** - Added `abi train vision` for ViT image classification and `abi train clip` for CLIP multimodal training with full architecture configuration, training loops, and help documentation (2026-01-26)
- **abi-dev-agents Claude Code plugin** - Created 6 specialized agents for ABI development: abi-planner, abi-explorer, abi-architect, abi-code-explorer, abi-code-reviewer, abi-issue-analyzer with Zig 0.16 and ABI pattern expertise (2026-01-25)
- **AI architecture refinements** - Updated documentation with multi-model training (ViT, CLIP), gradient management APIs, training architecture diagrams (2026-01-25)
- **GPU memory pooling improvements** - Added best-fit allocation, buffer splitting, fragmentation tracking/statistics, auto-defragmentation, and manual defragment API (2026-01-25)
- **Stress test timing fixes** - Fixed timing-sensitive assertions in HA/database stress tests, added Windows sleep support, updated API calls (2026-01-25)
- **Multi-Model Training Infrastructure** - Complete forward/backward training loops for LLM, Vision (ViT), and Multimodal (CLIP) models with gradient clipping, mixed precision support, contrastive learning, and 744 passing tests (2026-01-25)
- **Parallel HNSW index building** - Work-stealing parallelization for HNSW construction using Chase-Lev deques, fine-grained locking, atomic entry point updates (2026-01-25)
- **WebGPU quantized kernels** - WGSL shaders for Q4/Q8 matmul, SwiGLU, RMSNorm, Softmax, SiLU for WASM-compatible inference (2026-01-25)
- **Metal quantized kernels** - Q4/Q8 matrix-vector multiplication, SwiGLU, RMSNorm, Softmax, SiLU kernels for Apple Silicon (2026-01-25)
- **Zig 0.16 comprehensive migration** - Fixed 55+ compilation errors across test files, updated ArrayList to ArrayListUnmanaged, fixed time APIs (2026-01-25)
- **Runtime concurrency documentation** - Comprehensive API docs for ChaseLevDeque, EpochReclamation, MpmcQueue, ResultCache, NumaStealPolicy (2026-01-25)
- **GPU module fixes** - Fixed LaunchConfig stream field, ExecutionResult gpu_executed field, unified_buffer memory copy (2026-01-25)
- **Build system fix** - Added build_options to buildTargets for benchmarks (2026-01-25)
- **CLAUDE.md concurrency example fix** - Corrected MpmcQueue API usage (2026-01-25)
- **Lock-free concurrency primitives** - Chase-Lev deque, epoch reclamation, MPMC queue, NUMA-aware work stealing (2026-01-25)
- **Quantized CUDA kernels** - Q4/Q8 matrix-vector multiplication with fused dequantization, SwiGLU, RMSNorm (2026-01-25)
- **Result caching** - Sharded LRU cache with TTL support for task memoization (2026-01-25)
- **Parallel search** - SIMD-accelerated batch HNSW queries with ParallelSearchExecutor (2026-01-25)
- **GPU memory pool** - LLM-optimized memory pooling with size classes (2026-01-25)
- **CLI Zig 0.16 fixes** - Environment variable access, plugins command, profile command (2026-01-25)
- **Rust bindings** - Complete FFI bindings with safe wrappers for Framework, SIMD, VectorDatabase, GPU modules (2026-01-24)
- **Go bindings** - cgo bindings with SIMD, database, GPU modules and examples (2026-01-24)
- **CLI improvements** - Plugin management, profile/settings command, PowerShell completions (2026-01-24)
- **VS Code extension enhancements** - Diagnostics provider, status bar with quick actions, 15 Zig snippets for ABI patterns (2026-01-24)
- **Python observability module** - Metrics (Counter/Gauge/Histogram), distributed tracing, profiler, health checks with 57 tests (2026-01-24)
- **E2E Testing** - Comprehensive tests for Python (149 tests), WASM (51 tests), VS Code extension (5 suites) (2026-01-24)
- **VS Code extension** - Build/test integration, AI chat sidebar webview, GPU status tree view, custom task provider (2026-01-24)
- **npm WASM package** - @abi-framework/wasm v0.4.0 with updated README (2026-01-24)
- **Python bindings expansion** - Streaming FFI layer, training API with context manager, pyproject.toml for PyPI (2026-01-24)
- **Mega GPU Orchestration + TUI + Learning Agent Upgrade** - Full Q-learning scheduler, cross-backend coordinator, TUI widgets, dashboard command (2026-01-24)
- **Vulkan backend consolidation** - Single `vulkan.zig` module (1,387 lines) with VTable, types, init, cache stubs (2026-01-24)
- **SIMD and std.gpu expansion** - Integer @Vector ops, FMA, element-wise ops, subgroup operations, vector type utilities (2026-01-24)
- **GPU performance refactor** - Memory pool best-fit allocation, lock-free metrics, adaptive tiling for matrix ops, auto-apply kernel fusion (2026-01-24)
- **Multi-Persona AI Assistant** - Full implementation of Abi/Abbey/Aviva personas with routing, embeddings, metrics, load balancing, API, and documentation (2026-01-23)
- **Benchmark baseline refresh** - Performance validation showing +33% average improvement (2026-01-23)
- Documentation update: Common Workflows section added to CLAUDE.md and AGENTS.md
- GPU codegen consolidation: WGSL, CUDA, MSL, GLSL all using generic module
- Observability module consolidation (unified metrics, tracing, profiling)
- Task management system (CLI + persistence)
- Runtime consolidation (2026-01-17)
- Modular codebase refactor (2026-01-17)

---

## Quick Links

- [ROADMAP.md](ROADMAP.md) - Full project roadmap
- [CLAUDE.md](CLAUDE.md) - Development guidelines
