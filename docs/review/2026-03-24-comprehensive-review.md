# ABI Framework Comprehensive Review — 2026-03-24

## Executive Summary

The ABI Zig framework is in excellent structural health after 4 waves of systematic work. The mod/stub architecture is uniformly applied, error handling discipline is strong, and test coverage is comprehensive (54 integration tests). However, three critical gaps were identified:

1. **Database Engine data race** — `db_lock` exists but is only acquired in 1 of 7+ public methods
2. **Stub parity drift** — 14 of 20 features are missing lifecycle functions (`isEnabled`/`isInitialized`) in stubs
3. **Engine-to-connector bridge is unwired** — the inference engine and 16 LLM connectors work independently but are not connected, blocking the core `abi chat` workflow

The decomposition wave is nearly complete for features (19/20 done) but protocols remain monolithic. The roadmap's highest-impact work is wiring the inference-to-connector bridge to enable end-to-end LLM responses.

---

## 1. Architecture Findings

### 1.1 Decomposition Status

**Features (19/20 decomposed):**

| Feature | mod.zig Lines | Status |
|---------|--------------|--------|
| mobile | 533 | Only undecomposed feature >300 lines |
| All others | varied | Already decomposed into subdirectories |

**Protocols (0/4 fully decomposed):**

| Protocol File | Lines | Notes |
|---------------|-------|-------|
| `protocols/acp/server.zig` | 839 | Monolithic server |
| `protocols/mcp/server.zig` | 747 | Partial — `server/` subdir exists with 5 files |
| `protocols/mcp/real.zig` | 723 | Monolithic |
| `protocols/lsp/client.zig` | 634 | Monolithic |
| `protocols/ha/backup.zig` | 816 | Monolithic |
| `protocols/ha/replication.zig` | 647 | Monolithic |
| `protocols/ha/stub.zig` | 717 | Large stub (HA has wide surface area) |

**Additional large files (>1000 lines, lower priority):**

| File | Lines |
|------|-------|
| `features/gpu/dsl/codegen/generic.zig` | 1161 |
| `features/gpu/dsl/codegen/spirv/generator.zig` | 1157 |
| `features/ai/training/llm_trainer.zig` | 1146 |
| `foundation/security/password.zig` | 1125 |
| `features/ai/training/vision_trainer.zig` | 1118 |
| `features/ai/streaming/server/mod.zig` | 1105 |
| `connectors/discord/rest.zig` | 1096 |
| `features/ai/llm/ops/gpu_memory_pool.zig` | 1084 |
| `features/gpu/fusion.zig` | 1066 |
| `features/ai/llm/generation/streaming.zig` | 1059 |
| `features/gpu/memory/lockfree.zig` | 1055 |
| `features/network/raft.zig` | 1053 |
| `features/gpu/device.zig` | 1002 |
| `core/database/hnsw/mod.zig` | 997 |
| `core/database/parallel_hnsw.zig` | 973 |
| `connectors/shared.zig` | 900 |

### 1.2 Stub Parity Drift (Critical)

14 of 20 features have stubs missing lifecycle functions that exist in mod.zig:

**Missing `isEnabled` + `isInitialized` (13 features):**
ai, analytics, cache, cloud, mobile, network, search, storage, web, auth, messaging, gateway (isEnabled only)

**Additionally missing `init` + `deinit` (5 features):**
auth, cache, messaging, search, storage

This means compiling with any of these features disabled and calling `abi.<feature>.isEnabled()` produces a compile error.

### 1.3 Ungated Modules

Three modules are imported unconditionally in `root.zig` with no feature gate:
- `inference` — engine, scheduler, sampler, KV cache
- `tasks` — task management, job queues
- `connectors` — 16 provider adapters, 11,521+ lines

All other major subsystems have `feat_*` gates.

### 1.4 Build System

- `build/validation.zig` (378 lines) contains 12+ repetitive test step definitions that could be a single helper function
- Missing `inference_async_test.zig` import in `test/mod.zig`

### 1.5 Clean Areas

- Cross-feature import hygiene: zero violations (no `@import("../../features/")` bypassing comptime gates)
- root.zig export consistency: all 20 features + 4 protocols follow identical pattern
- Feature structure: 20/20 features have mod.zig + stub.zig + types.zig

---

## 2. Quality Findings

### 2.1 Bugs

**BUG-1 (Critical): Database Engine data race**
- File: `src/core/database/engine.zig`
- `db_lock` declared at line 62 but only acquired in `dreamStatePrune` (line 104)
- 6 public methods access `vectors_array` and `hnsw_index` without the lock: `index`, `delete`, `search`, `searchByVector`, `rebuildHnswIndex`, `count`
- If `dreamStatePrune` runs on a background thread (as documented), concurrent calls cause data races
- Fix: acquire `db_lock` in all public methods. Readers take shared lock, writers take exclusive lock.

**BUG-2 (Important): HNSW search double-silent-swallow**
- File: `src/core/database/hnsw/mod.zig`, lines 340-370
- When GPU path fails allocation, fallback to `searchNeighborsSequential` is also `catch {}`'d
- Under memory pressure, search results silently degrade with no indication to caller
- Fix: propagate error or log warning when fallback also fails

**BUG-3 (Important): Compute Mesh drops discovered nodes**
- File: `src/features/compute/mesh.zig`, line 211
- `self.nodes.append(self.allocator, ...) catch {}` silently drops OOM
- User sees "Discovered new node" log but node is never added
- Fix: propagate error or log specific failure message

### 2.2 Warnings

**WARN-1: WAL sync silently ignored**
- File: `src/core/database/storage/wal.zig`, line 137
- `file.sync(io) catch {}` after write defeats durability guarantee
- Also in `src/protocols/ha/pitr/persistence.zig`, line 213

**WARN-2: Plugin manifest directory creation failure ignored**
- File: `src/features/ai/llm/providers/plugins/manifest.zig`, line 266

**WARN-3: Discord REST parsers use catch false/0 for required fields**
- File: `src/connectors/discord/rest_parsers.zig`, multiple lines
- Silently provides wrong defaults on malformed payloads

### 2.3 Clean Areas

- 0 problematic `@panic` in library code
- 5 `catch unreachable` (all crypto/csprng — justified)
- 0 `catch |_|` silent error discards
- Consistent allocator/defer patterns across all features
- Thread safety solid in cache, messaging, runtime, parallel_search (Engine is the exception)
- GPU backend dispatch and fallback: clean error propagation
- Connector error propagation: correct (HTTP status mapping, timeout handling)
- Most `catch {}` patterns are intentional (test cleanup, terminal restore, temp files, optional operations)

---

## 3. Roadmap Findings

### 3.1 GPU Backend Status

| Backend | Status | Lines | Notes |
|---------|--------|-------|-------|
| Metal | Partial | 2,212 | Compute works, missing MPS operators |
| CUDA | Partial | 6,080 | Comprehensive, untestable without hardware |
| Vulkan | Partial | 2,746 | Types + pipeline complete, needs hardware validation |
| stdgpu | Complete | 951 | CPU emulation fallback, works in CI |
| WebGPU | Partial | 660+ | API structure, function pointers null |
| OpenGL | Partial | 762+ | GL 4.3+ compute, needs external context |
| OpenGL ES | Partial | 843+ | GLES 3.1+, same as OpenGL |
| WebGL2 | Stub (correct) | 121 | Returns ComputeShadersUnavailable by design |
| DirectML | Stub | 81 | Returns BackendNotSupported, Windows-only |
| FPGA | Simulation | 4,444 | Substantial kernel code, no hardware wiring |
| TPU | Missing | 0 | Build flag exists, no implementation files |

### 3.2 Inference Engine

| Component | Status | Notes |
|-----------|--------|-------|
| Demo backend | Complete | Synthetic text from seeded PRNG |
| Connector backend | Stub | Echoes prompt, does NOT call any connector |
| Local backend | Partial | Transformer loop implemented, needs GGUF validation |
| Paged KV Cache | Complete | Block-based page management |
| Scheduler | Complete | Heap-based priority queue |
| Sampler | Complete | Temperature, top-k, top-p |

**Critical gap**: `generateConnector` (line 14 of `backends.zig`) echoes the prompt instead of calling any connector client. This is the single most impactful gap in the framework.

### 3.3 Connector Coverage

| Connector | Lines | Depth |
|-----------|-------|-------|
| OpenAI | 392 | Full |
| Anthropic | 508 | Full |
| Ollama | 584 | Full |
| Cohere | 710 | Full |
| Mistral | 521 | Full |
| MLX | 379 | Full |
| HuggingFace | 328 | Partial |
| llama.cpp | 213 | Partial |
| LM Studio | 211 | Partial |
| vLLM | 210 | Partial |
| Gemini | 238 | Partial |
| Discord | directory | Partial (bot integration) |
| Claude, Codex, OpenCode, ollama_passthrough | 83-97 | Thin wrappers |

### 3.4 Protocol Maturity

| Protocol | Maturity | Missing for Production |
|----------|----------|----------------------|
| MCP | Most mature | Auth, SSE transport, subscriptions, prompt templates |
| ACP | Functional | Streaming, push notifications, multi-agent federation, auth |
| LSP | ZLS wrapper | Not a standalone server, depends on external ZLS |
| HA | Framework | Multi-node transport, leader election, WAL shipping |

### 3.5 CLI Completeness

| Command | Status |
|---------|--------|
| version, doctor, features, platform, connectors, info, help | Complete |
| chat | Partial — routes but prints "Engine would execute here" |
| db | Partial — works when feat_database=true |
| serve / acp serve | Partial — starts ACP HTTP server |
| lsp | Partial — requires external ZLS |
| dashboard | Partial — read-only TUI, requires -Dfeat-tui=true |

### 3.6 TUI Dashboard

~1,512 lines. Framework primitives (render, layout, widgets, events, ANSI) are solid and tested. Dashboard is a read-only two-panel display showing feature flags and GPU backends. Missing: live metrics, log viewer, interactive chat, database browser.

### 3.7 Feature Catalog Assessment

| Status | Count | Features |
|--------|-------|----------|
| Complete | 3 | database (WDBX), stdgpu emulation, inference primitives (KV cache, scheduler, sampler) |
| Partial | 16 | gpu, ai, llm, agents, network, observability, web, auth, messaging, cache, search, gateway, reasoning, tui, mcp, acp |
| Stub/Aspirational | 9 | training, embeddings (standalone), cloud, analytics, storage, mobile, desktop, compute, ha |
| Missing | 1 | TPU (build flag only, no code) |

---

## 4. Actionable Task List

### P0 — Fix Now (bugs and broken contracts)

- [x] **Fix Database Engine data race** — Added `db_lock` to all public methods (1f01853)
- [x] **Fix stub parity drift** — Stubs already had lifecycle functions via StubFeature helpers; fixed orchestration + multi_agent parity gaps (354fa4e, da70e3e)
- [x] **Wire `inference_async_test.zig`** — Added import to `test/mod.zig` (1f01853)
- [x] **Fix HNSW double-silent-swallow** — Added log warnings on fallback failure (1f01853)
- [x] **Fix WAL sync discard** — Propagated file.sync() errors (1f01853, b19af6d)

### P1 — Wave 5 Decomposition

- [x] **Decompose `features/mobile/mod.zig`** → sensors, notifications, permissions, device (383b44d)
- [x] **Decompose `protocols/acp/server.zig`** → server/ with routing, sessions, tasks, agent_card (12e8491)
- [x] **Decompose `protocols/mcp/real.zig`** → handlers/, factories.zig (e9bc644)
- [x] **Decompose `protocols/lsp/client.zig`** → client/ with requests, notifications, transport (d6fbcf0)
- [x] **Decompose `protocols/ha/backup.zig`** → backup/ with config, execution, storage (353ea00)
- [x] **Decompose `protocols/ha/replication.zig`** → replication/ with state, sync, membership (353ea00)
- [x] **Reduce `protocols/ha/stub.zig`** — Imported shared types from types.zig (717→665 lines) (b61f234)
- [x] **Fix Compute Mesh node drop** — Propagated error with log message (b19af6d)
- [x] **Fix Plugin manifest dir creation** — Propagated error with log message (b19af6d)

### P2 — Feature Completeness (highest impact)

- [x] **Wire engine-to-connector bridge** — Provider resolution from model_id, env config loading, 12 providers (33d74bc)
- [x] **Complete `abi chat` end-to-end** — Profile router → Engine.generate() (33d74bc)
- [ ] **Add feature gates for inference, tasks, connectors** — Needs stub modules, deferred
- [ ] **Add MCP authentication** — Token/API key validation for MCP tool calls
- [ ] **Add MCP SSE transport** — For web-based clients (Claude Desktop over HTTP)
- [ ] **Add Metal MPS integration** — Wire MPS operators (GEMM, softmax, layer norm) for real GPU inference on macOS
- [x] **Refactor `build/validation.zig`** — Extracted addFeatureTestLane helper (406→240 lines) (958dba1)

### P3 — Backlog

- [ ] **Create TPU backend** — Build flag exists, no implementation files
- [ ] **Validate local inference with real GGUF** — Test llama.zig model loader against actual weights
- [ ] **Add TUI live dashboard panels** — Metrics, log viewer, interactive chat widget
- [ ] **Add ACP streaming responses** — SSE for long-running agent tasks
- [ ] **Add HA multi-node transport** — Leader election, WAL shipping
- [ ] **Add OAuth/JWT to protocols** — Currently no protocol has authentication
- [ ] **Decompose large leaf files** — raft.zig (1053), device.zig (1002), discord/rest.zig (1096), etc.
- [ ] **End-to-end inference integration test** — Test that exercises chat → routing → engine → connector → response
- [ ] **Cross-backend GPU test** — Switch between Metal/CUDA/stdgpu in same test

---

## Review Methodology

Three specialized agents were dispatched in parallel, each with a distinct focus:
1. **Architecture agent** — decomposition targets, import hygiene, stub parity, build system (37 tool calls)
2. **Quality agent** — catch {} audit, thread safety, GPU edge cases, connector errors (43 tool calls)
3. **Roadmap agent** — GPU backends, inference engine, connectors, protocols, CLI, TUI, catalog (58 tool calls)

Total analysis: 138 tool calls across ~2,500 source files, producing this unified report.
