# AGENTS.md - Comprehensive Code-Centric Refactor Playbook

Overview
This document serves as a highly structured and code-oriented guide intended for all agents contributing to the ABI Repository. The playbook focuses on executing a comprehensive refactoring effort to align with Zig 0.16.0-dev.427+86077fe6b, ensuring reproducible, stable, and maintainable code practices. Each agent role has defined objectives and responsibilities to modernize the repository, enforce strong typing, and support a sustainable development lifecycle.

The goals of this playbook include:
Modernizing the build system and repository structure.
Enforcing safe and explicit I/O boundaries.
Building a typed, ergonomic, and testable CLI.
Implementing a high-performance, safe parsing subsystem.
Establishing reproducible, cross-platform CI/CD pipelines.

Following these guidelines will result in a fully modernized ABI repository with long-term maintainability and deterministic builds.

## 0) Project Purpose & Scope (ABI Agent + WDBX)

**What ABI is for:** A lightning-fast AI/ML stack in Zig that powers an ABI agent and a custom WDBX vector database. It targets GPU/WebGPU acceleration, SIMD-optimized kernels, and lock-free concurrency with a production-grade server and CLI surface. In short: low-latency inference and training orchestration with an embedded vector store, portable across Windows, macOS, and Linux.

**Primary capabilities (purpose-critical):**
- GPU acceleration (WebGPU backends plus CPU fallback), SIMD math, and zero-copy hot paths.
- WDBX vector database for embeddings (custom file and record layout; high-throughput insert and search).
- Agent runtime (CLI/TUI and server) with plugin and dynamic-loading hooks.
- Cross-platform build on Zig 0.16-dev, emphasizing reliability and reproducibility.

**Purpose-aware constraints for refactors:**
- No performance regression on hot paths: core kernels, I/O pipelines, vector search, and the agent message loop.
- Preserve WDBX on-disk layout and memory-layout assumptions (ABI and FFI expectations, alignment).
- Maintain lock-free invariants (do not introduce contention through cleanups).
- Keep GPU/WebGPU compatibility intact across supported platforms.

**Module map (inferred):**
- `agent/` (runtime and skills)
- `wdbx/` (vector database)
- `gpu/` (WebGPU and kernels)
- `server/` (HTTP and TCP surfaces)
- `cli/` (CLI and TUI)
- `tools/` (analyzer and gates)
- `build.zig` (top-level build orchestration)

_Source: repo tagline and README excerpt referenced from the Zigistry GitHub mirror._

---

Agent Roles & Responsibilities

1. Build Agent
Objective: Maintain a robust and modernized Zig build pipeline.

Core Responsibilities:
Refactor build.zig to leverage b.root_module and b.createModule for modularity.
Configure common target settings and optimization flags for Debug, ReleaseFast, and ReleaseSmall.
Provide standardized build steps for:
zig build run
zig build test
zig build docs
zig build fmt
zig build bench
Ensure deterministic builds with consistent artifact naming across platforms.
Integrate caching where possible to reduce build times without compromising reproducibility.

Deliverables:
Updated build.zig with modularized tasks.
Documented build commands in README.md.
Verified reproducibility on all supported platforms.

---

2. I/O Agent
Objective: Transition all I/O operations toward an explicit and recoverable boundary-based architecture.

Core Responsibilities:
Replace all direct stdout/stderr usage with injected writer objects.
Enforce strict separation between:
Human-readable logs -> stderr
Machine-readable outputs -> stdout
Use explicit formatting specifiers for all messages to avoid implicit conversions.
Implement file I/O via adapter layers to facilitate error handling and safe recovery.
Provide error-handling patterns for partial reads/writes and support graceful degradation.

Deliverables:
I/O abstraction layer with adapters.
Tests demonstrating recoverable I/O scenarios.
Logging and output behavior documented for developers and CI usage.

---

3. CLI Agent
Objective: Provide a clear, typed, and ergonomic command-line interface.

Core Responsibilities:
Implement entrypoints fully compatible with zig build run.
Add subcommands for structured operations:
parse
version
lint
check
Deliver comprehensive --help output with usage examples.
Generate meaningful typed error messages instead of raw strings.
Conduct smoke tests for all major CLI flows to ensure stability.

Deliverables:
A fully functional CLI with typed commands and structured options.
CLI integration tests for all subcommands.
Updated user documentation for command usage.

---

4. Parser Agent
Objective: Modernize the parsing subsystem for safety, performance, and maintainability.

Core Responsibilities:
Replace pointer arithmetic with safe, slice-based APIs.
Use std.ArrayList and streaming reads for efficient handling of large files.
Integrate a Diagnostics system to report recoverable parsing errors without premature termination.
Maintain golden tests to ensure behavior consistency during future refactors.
Run performance benchmarks on large inputs to validate optimizations.

Deliverables:
Refactored parser with slice-based data handling.
Diagnostics framework for error reporting.
Performance benchmark results and golden test snapshots.

---

5. CI/CD Agent
Objective: Guarantee reproducible, multi-platform automation and enforce repository consistency.

Core Responsibilities:
Configure CI pipelines for Linux, macOS, and Windows using Zig 0.16.
Automate the generation and publication of documentation artifacts.
Enforce formatting and style checks as part of the CI pipeline.
Provide reproducible build artifacts validated through CI.
Optionally deploy generated HTML documentation to GitHub Pages.

Deliverables:
Multi-OS CI pipeline configuration with reproducible builds.
Automated artifact uploads and optional documentation deployments.
Style and format enforcement integrated into CI checks.

---

Lifecycle & Collaboration

> **Acceptance Criteria:** All CI gates green, analyzer metrics at or better than targets; no public API diffs; no behavior changes in tests; zero shadowing and zero unused discards; canonical module order across the repository; no performance regression on purpose-critical paths (agent loop, WDBX operations, GPU kernels).

Phased Delivery
Refactor efforts are delivered in sequential phases:
Build Agent
I/O Agent
CLI Agent
Parser Agent
CI/CD Agent

Each phase must:
Be submitted as a separate PR.
Pass CI validation.
Include documentation updates and migration notes.

Rollback & Safety
Maintain fallback paths for each agent scope.
Provide adapter layers to bridge breaking changes during migration.
Document rollback procedures and known limitations.

Documentation & Artifacts
Keep README.md and CONTRIBUTING.md continuously updated.
Ensure consistent logging, reproducible tests, and generated documentation.
Maintain a clear artifacts directory for all CI outputs.

### 5.5 Metrics Summary

| Metric | Before | After (target) |
|---|---:|---:|
| Total lines | TBD | -15% |
| Avg function length (LOC) | TBD | -10% to -25% |
| Functions > 200 lines | TBD | 0 |
| Avg cyclomatic complexity | TBD | <= configured threshold |
| Max cyclomatic complexity | TBD | -30% to -50% |
| Files violating 100-char rule | TBD | 0 |
| Shadowing occurrences | TBD | 0 |
| Unused discards (`var _`) | TBD | 0 |
| Nested type depth > 3 | TBD | <= 3 |
| **Inference latency P50/P99 (ms)** | TBD | <= baseline |
| **Vector DB search throughput (qps)** | TBD | >= baseline |
| **GPU kernel throughput (GB/s or it/s)** | TBD | >= baseline |

*Note:* Keep this table updated as instrumentation improves. Ensure improvements do not sacrifice ABI and WDBX layout guarantees or hot-path performance.

### 5.7 PR Template (Quality Sprint)

**Checklist**
- [ ] zig fmt --check passes
- [ ] Analyzer clean vs `.code-quality-config.json`
- [ ] No variable shadowing
- [ ] No `var _ = ...` discards
- [ ] Functions >200 lines split
- [ ] Error handling explicit and consistent
- [ ] Canonical file layout (imports -> consts -> types -> funcs -> tests)
- [ ] Build.zig uses current APIs (modules, root modules, targets/opts)
- [ ] WDBX on-disk/memory layouts unchanged
- [ ] No perf regression on agent loop / vector search / GPU kernels (bench CI attached)

**Testing**
- Benchmarks: `zig build bench` (attach P50/P99 & throughput deltas)

---

Expected Outcomes
Fully modernized ABI repository compatible with Zig 0.16-dev.
Deterministic multi-platform builds with validated CI pipelines.
Strongly typed errors and explicit I/O boundaries.
Maintainable, code-focused collaboration model supporting long-term development.

---

References
Build Modernization: b.root_module, b.createModule
I/O Boundaries: Injected writers, adapter layers
CLI Ergonomics: Typed errors, stdout/stderr separation
Parser Internals: std.ArrayList, streaming APIs, diagnostics
CI/CD Best Practices: Multi-OS pipelines, docs generation, formatting enforcement
