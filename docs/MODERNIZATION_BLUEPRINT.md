# ABI Modernization Blueprint

## Purpose
This document distills the modernization charter for the ABI framework into an actionable reference. The goal is to refresh the project while keeping the existing public API contract intact and preserving ABI's focus on high performance.

## Guiding Objectives
- **Build parity with Zig 0.16** – adopt `b.createModule`, `b.addModule`, `b.standardTargetOptions`, and `b.standardOptimizeOption` so every build target follows the contemporary Zig build graph.
- **Clarity first** – group code by responsibility (framework core, WDBX, GPU, utilities, connectors) and converge on snake_case files with camelCase symbols.
- **Safety and speed** – treat allocators as explicit parameters, favour stack storage, add bounds checking in hot paths where it does not jeopardise throughput, and prefer error unions over panics.
- **Comprehensive testing** – split unit, integration, property, and benchmark suites; execute them across Linux, macOS, and Windows in debug, release-safe, and release-fast profiles.
- **Developer empathy** – ship with clear README guidance, doc comments, and contribution templates so new collaborators can confidently extend the system.

## Build Modernisation
1. Replace legacy module wiring with explicit `b.createModule` / `b.addModule` calls and wire consumers through `module.use`.
2. Expose `b.standardTargetOptions` and `b.standardOptimizeOption` so callers can request target/optimisation combinations.
3. Model each major component (CLI, agent tests, WDBX tests, documentation generators) as a distinct build step that CI can execute independently.
4. Separate installation outputs: binaries land in `zig-out/bin`, libraries in `zig-out/lib`, documentation artefacts in `zig-out/share`.

## Code Organisation & Readability
- Preserve backwards compatibility via `src/root.zig`, but reorganise modules beneath `src/framework`, `src/features`, and `src/util`.
- Remove dead code and replace TODO blocks with tracked issues.
- Adopt Zig doc comments (`///`) for all public entry points.
- Factor long functions into helpers with early exits to reduce nesting.

## Memory and Concurrency Strategy
- Use caller-provided `std.mem.Allocator` handles throughout the codebase and note ownership semantics in doc comments.
- Prefer slices to raw pointer arithmetic; when pointer math is unavoidable, document invariants inline.
- Apply Zig's `async`/`await` primitives for cooperative concurrency and hide synchronisation details behind small, composable abstractions.
- Offer deterministic single-thread execution for reproducibility by gating threading through build/runtime options.

## WDBX Database Expectations
- Publish an explicit API for opening, closing, and mutating WDBX stores along with documented error sets.
- Describe the on-disk record format so migrations can be versioned.
- Support concurrent readers/writers with locking or optimistic strategies and capture performance goals for record search, index maintenance, and SIMD paths.
- Prepare the storage layer for alternative backends (in-memory, external services) through a common interface.

## Error Handling, Logging, and Observability
- Define namespaced error sets per subsystem or a shared tagged error union.
- Provide a configurable logging facade that supports trace/debug/info/warn/error levels and avoids unnecessary allocations in hot loops.
- Integrate lightweight metrics hooks so latency and throughput of core paths can be measured without code churn.

## Documentation & Examples
- Refresh the top-level README with the project purpose, setup steps, and common workflows.
- Generate API references automatically and host them in `docs/` or via GitHub Pages.
- Maintain subsystem design notes (agents, WDBX, GPU) and focused code examples (record CRUD, agent pipeline run, CLI demos).

## Testing & CI Requirements
- Ensure unit and integration tests cover success and failure paths, including concurrency scenarios.
- Run CI on Linux, macOS, and Windows with debug, release-safe, and release-fast optimisations; optionally include sanitizer or fuzz configurations.
- Track benchmark outputs (latency, throughput) to detect performance regressions.

## Packaging & Distribution
- Keep executables separate from libraries during install; document how to consume `libabi` and any C bindings.
- Maintain semantic versioning with a changelog and release automation.
- Provide Dockerfiles, Helm charts, and SBOM generation steps for production rollouts.

## Community & Future-Proofing
- Supply `CONTRIBUTING.md`, issue templates, and a code of conduct; enforce formatting through `zig fmt` and other linters in CI.
- Abstract compute backends so GPU acceleration or distributed execution can be added later.
- Offer compile-time/runtime toggles for optional features and instrumentation hooks for profiling and monitoring.

