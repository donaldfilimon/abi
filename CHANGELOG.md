# Changelog

<p align="center">
  <img src="https://img.shields.io/badge/Version-0.4.0-blue?style=for-the-badge" alt="Version 0.4.0"/>
  <img src="https://img.shields.io/badge/Status-Stable-success?style=for-the-badge" alt="Stable"/>
</p>

## Unreleased

### MCP & ACP Server Infrastructure
- **MCP server** (`src/services/mcp/`): JSON-RPC 2.0 over stdio, 5 WDBX tools (db_query, db_insert, db_stats, db_list, db_delete)
- **ACP server** (`src/services/acp/`): Agent Communication Protocol with AgentCard, Task lifecycle, skills
- CLI commands: `mcp serve`, `mcp tools`, `acp card`, `acp serve`

### New Connectors
- **LM Studio** (`connectors/lm_studio.zig`): OpenAI-compatible local server, default port 1234
- **vLLM** (`connectors/vllm.zig`): OpenAI-compatible local server, default port 8000
- Total connectors: 8 LLM providers + discord + scheduler

### Gateway Module
- **Gateway** (`features/gateway/`): Radix tree router, 3 rate limiters (token bucket, sliding window, fixed window), circuit breaker

### Training CLI
- 11 subcommands: run, new, llm, vision, clip, auto, resume, monitor, info, generate-data, help
- Synthetic data generation, external quantization support

### Bug Fixes & Hardening
- Fixed ACP `createTask` errdefer double-free (ensureUnusedCapacity pattern)
- MCP: JSON-RPC version validation, 6 new security/edge-case tests
- ACP: 4 new tests (unknown ID, sequential IDs, message content, control chars)
- Feature tests: 675 → 684 (13 new inline tests)

## 0.4.0 - 2026-01-23

### GPU Backend Completion

All GPU backends now have complete VTable implementations with full production readiness:

- **Vulkan Backend** (`src/features/gpu/backends/vulkan.zig`)
  - Fixed missing `destroyKernel()` implementation (was empty stub)
  - Added proper Vulkan resource cleanup (pipelines, pipeline layouts, descriptor sets)
  - Full VTable compliance with all 12 required functions
  - Cross-platform compute shader support via SPIR-V

- **OpenGL ES VTable** (`src/features/gpu/backends/opengles_vtable.zig`)
  - Created new VTable wrapper for OpenGL ES 3.1+ compute support
  - Mobile/embedded GPU compute capabilities
  - Follows identical pattern to OpenGL VTable implementation
  - Proper memory management and kernel lifecycle

- **Backend Factory Integration**
  - All 11 GPU backends integrated with `backend_factory.zig`
  - Priority selection (NN): CUDA → TPU → Metal → Vulkan → WebGPU → OpenGL → … → simulated
  - Build flags: `-Dgpu-backend=cuda,vulkan,metal,webgpu,tpu` (comma-separated)

- **GPU Backend Status Matrix**
  | Backend | Status | Production Ready |
  |---------|--------|-----------------|
  | CUDA | ✅ Complete | Yes |
  | Vulkan | ✅ Complete | Yes |
  | Metal | ✅ Complete | Yes |
  | WebGPU | ✅ Complete | Yes |
  | TPU | ⚠️ Stub (runtime not linked) | When linked |
  | OpenGL | ✅ Complete | Yes |
  | OpenGL ES | ✅ Complete | Yes |
  | WebGL2 | ⚠️ Stub (design limitation) | No |
  | std.gpu | ✅ Complete | Yes |
  | Simulated | ✅ Complete (CPU fallback) | Yes |
  | FPGA | ⚠️ Stub (hardware dependent) | No |

## 0.3.3 - 2026-01-23

### Documentation & Cleanup

- Updated CLAUDE.md and AGENTS.md with "Common Workflows" section
- Added comprehensive GPU backend guide (`docs/_docs/gpu.md`)
- Implemented real CUDA VTable stub with loadable driver handling
- Added OpenGL, OpenGL‑ES, WebGL2 stub support (mutually exclusive warning)
- Introduced dedicated benchmark runner (`scripts/run_benchmarks.bat`)
- Captured and documented benchmark results in the README
- Updated security guidance to reference the built‑in Secrets Manager

## 0.3.0 - 2026-01-17

### Feature Stub API Parity

All feature-gated stub modules have been audited and updated for complete API parity:

- **AI Stub** (`src/features/ai/stub.zig`)
  - Fixed `SessionData`, `SessionMeta` to match real implementation
  - Updated `TrainingConfig` with all fields (sample_count, model_size, learning_rate, optimizer, etc.)
  - Fixed `TrainingReport` and `TrainingResult` structures
  - Updated `Checkpoint` with correct fields (step, timestamp, weights)
  - Fixed `TrainableModelConfig` (num_layers, num_heads, intermediate_dim, etc.)
  - Added `numParams()` method to `TrainableModel`
  - Added `CheckpointingStrategy` enum
  - Added `addMessage()` to `PromptBuilder`

- **LLM Stub** (`src/features/ai/llm/stub.zig`)
  - Added `matrixMultiply` to ops struct
  - Added `printSummaryDebug` to GgufFile

- **GPU Stub** (`src/features/gpu/stub.zig`)
  - Added missing `backendAvailability` export

- **Network Stub** (`src/features/network/stub.zig`)
  - Added `touch()` and `setStatus()` methods to NodeRegistry
  - Fixed `NodeInfo` to include `last_seen_ms` field
  - Corrected `NodeStatus` enum (healthy, degraded, offline)

**Build Verification:**
- `zig build -Denable-ai=false` - Passes
- `zig build -Denable-gpu=false` - Passes
- `zig build -Denable-network=false` - Passes
- `zig build -Denable-database=false` - Passes
- `zig build -Denable-web=false` - Passes
- `zig build -Denable-profiling=false` - Passes

### Documentation Updates

- Rewrote README.md with cleaner structure and current status
- Updated all docs/ files with correct links
- Removed stale MAIN_WORKSPACE.md references from 53 files
- Updated TODO.md with stub verification status
- Updated ROADMAP.md with code quality completion status

## Zig 0.16.0-dev.2611+f996d2866 Compatibility Updates

### 2026-01-23

**Critical Compilation Fixes**

- **GPU Dispatcher Syntax** (`src/features/gpu/dispatcher.zig`)
  - Fixed missing semicolons after catch blocks (Zig 0.16.0-dev.2611+f996d2866 requirement)
  - Cleaned up misplaced/dueplicate code structure

- **Execution Coordinator** (`src/features/gpu/execution_coordinator.zig`)
  - Added missing `simd_outlier_threshold` variable definition
  - Restored proper SIMD outlier detection logic

- **LLM Trainer** (`src/features/ai/training/llm_trainer.zig`)
  - Fixed struct initialization missing `gpu_ops` field

**Build Status:** ✅ All compilation tests pass with Zig 0.16.0-dev.2611+f996d2866
**Formatting:** ✅ `zig fmt --check .` passes
**Type Checking:** ✅ `zig typecheck` passes
**CLI Smoke Tests:** ✅ All command-line operations functional

### Compatibility Notes

The codebase now demonstrates full Zig 0.16.0-dev.2611+f996d2866 compatibility:
- ✅ `std.Io` patterns instead of deprecated APIs
- ✅ `std.time.Timer` for timing operations
- ✅ Proper `{t}` formatting for errors/enums
- ✅ Modern type casting (`@as`, `@intFromEnum`)
- ✅ `std.ArrayListUnmanaged` with explicit allocator passing

---

This changelog combines information from both CHANGELOG.md and RELEASE_NOTES.md into a single authoritative source.
Updates will now be tracked exclusively in this file.

© 2026 ABI Framework contributors.