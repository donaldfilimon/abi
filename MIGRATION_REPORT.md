# Zig 0.16-dev Migration Playbook

## Strategic Objectives
- **Maintain feature parity** while adopting Zig 0.16-dev language, stdlib, and build-system changes.
- **Unlock GPU acceleration** across Vulkan, Metal, and CUDA backends using the refreshed async/event APIs.
- **Lay neural-network foundations** for tensor-core optimized kernels and training workflows.
- **De-risk future upgrades** by codifying playbooks for testing, observability, and security guardrails.

## Migration Strategy
1. **Stabilize toolchain alignment**: pin Zig 0.16-dev.254+6dd0270a1, upgrade build metadata, and validate cross-platform bootstrap scripts.
2. **Refactor core modules** in staged waves (allocators → stdlib updates → GPU → neural-network stack) with regression baselines between waves.
3. **Parallelize validation** using CI matrix expansion, nightly benchmarks, and canary deployments for GPU hardware pools.
4. **Institutionalize knowledge** via updated guides, reviewer checklists, and metrics dashboards to sustain the new baseline.

## Milestones & Owners
| Milestone | Target Date | Success Criteria | Owner |
| --- | --- | --- | --- |
| Toolchain pin + build graph audit | 2025-09-22 | `zig build`/`zig build test` clean on all dev OS images | Dev Infra (A. Singh) |
| Allocator + stdlib refactor | 2025-10-01 | Allocator policy tests pass; no regressions in memory diagnostics | Runtime Team (L. Gomez) |
| GPU backend enablement | 2025-10-12 | Vulkan/Metal/CUDA smoke tests green; shader transpile scripts updated | GPU Team (M. Ito) |
| Neural-network kernel uplift | 2025-10-20 | Tensor-core matmul 1.8× speedup; training loop convergence parity | ML Systems (R. Chen) |
| Observability & security sign-off | 2025-10-25 | Dashboards live; threat model & SBOM refreshed | SRE/Security (K. Patel) |
| Final migration review | 2025-10-27 | Reviewer checklist cleared; release notes drafted | Release Eng (S. Walker) |

## Risk Register
| Risk | Impact | Probability | Owner | Mitigation |
| --- | --- | --- | --- | --- |
| Upstream Zig breaking change lands mid-migration | Build failures across CI | Medium | Dev Infra | Track nightly Zig changelog, freeze snapshot after RC, mirror binaries internally |
| GPU driver/toolchain mismatch (Metal 4 / CUDA 12.5) | GPU tests fail on macOS/Linux | High | GPU Team | Maintain driver matrix, pre-warm CI AMIs, provide fallback software rasterizer mode |
| Allocator regression introduces leaks | Runtime instability | Medium | Runtime Team | Expand leak detectors, nightly fuzzing, require allocator-specific review |
| Neural-net kernels miss tensor-core utilization | Performance degradation | Medium | ML Systems | Profile via Nsight/Metal Perf HUD, compare against baseline kernels, escalate to vendor support |
| Observability gaps hide regressions | Slow incident response | Low | SRE | Enforce tracing spans, synthetic monitors per backend, alert fatigue review |

## Module Inventory Snapshot
- **Core Runtime**: `src/runtime/allocator.zig`, `src/runtime/context.zig`, `src/runtime/async.zig` (ownership: Runtime Team).
- **Compiler/Build Glue**: `build.zig`, `build.zig.zon`, `scripts/toolchain/*.sh` (ownership: Dev Infra).
- **GPU Backends**: `src/gpu/vulkan/*.zig`, `src/gpu/metal/*.zig`, `src/gpu/cuda/*.zig`, shader assets in `assets/shaders/` (ownership: GPU Team).
- **Neural Network Stack**: `src/nn/tensor.zig`, `src/nn/ops/*.zig`, `src/nn/train/*.zig` plus benchmarks under `benchmarks/nn/` (ownership: ML Systems).
- **Observability & Security**: `src/telemetry/*.zig`, `tools/security/*.py`, CI workflows in `.github/workflows/` (shared by SRE/Security).

## Technical Workstreams

### Build-System Deltas
- Adopt `std.Build.Step` graph updates: replace deprecated `.builder` field usage with `b` references and ensure custom steps emit `b.allocator` for scratch allocations.
- Normalize module registration via `b.addModule("abi", .{ .source_file = .{ .path = "src/mod.zig" } });` and migrate dependent packages to `module.createImport()` semantics.
- Update cross-compilation targets to rely on `std.zig.CrossTarget` with architecture triples, ensuring WASI and Android targets reflect new enum variants.
- Regenerate `build.zig.zon` with SHA256 checksums for third-party packages compatible with 0.16-dev; verify license metadata remains intact.

### Stdlib / API Updates
- Replace legacy `std.os.time.Timestamp` usage with `std.time.Timestamp` and adopt monotonic clock helpers.
- Migrate `std.json.Value` handling to the new `parseFromSlice` interface and update error propagation to typed `error{ ParseError, InvalidType }` sets.
- Refactor async runtime to align with `std.Channel` API adjustments (`.close()` explicit, `try channel.close();` semantics).
- Audit `std.meta` usage for renamed helpers (`std.meta.trait.fields` → `std.meta.fields`, etc.) and add compatibility shims where necessary.

### Allocator Policies
- Default allocator remains `std.heap.page_allocator`; introduce scoped `ArenaAllocator` for neural-net graph compilation to avoid fragmentation.
- Create profiling toggles that wrap allocators with `std.heap.LoggingAllocator` under debug builds and push telemetry counters through `telemetry/alloc.zig`.
- Enforce zero-cost abstractions by converting ad-hoc `Allocator` pointers to `anytype` generics when call sites are specialized.
- Document fallback strategy for systems without large page support: auto-detect and switch to `GeneralPurposeAllocator(.{})` with guard-page monitoring enabled.

### GPU Backend Enablement
- **Vulkan**: update descriptor set layouts to `std.vulkan.descriptor` helpers, migrate synchronization to timeline semaphores, and confirm SPIR-V generation via `tools/shaderc` using Zig build steps.
- **Metal**: adopt Zig 0.16 `@cImport` updates, refresh Metal shading language bindings, and ensure argument buffers map to new resource index macros.
- **CUDA**: align CUDA driver API bindings with `extern` calling convention changes; regenerate PTX kernels with tensor-core paths compiled for `sm_90a`.
- Shared requirements: integrate GPU backend selection into `zig build` options (`-Dgpu-backend=vulkan|metal|cuda`), ensure fallback CPU path remains functional, and capture backend metrics via observability pipeline.

### Neural-Network Foundations
- **Tensor Core Enablement**: use WMMA intrinsics where available; add architecture detection to pick FP8/BF16 kernels and guard fallback code paths.
- **Operator Library**: refactor `ops` module to expose fused kernels (conv+bias+activation), adopt Zig iterators for shape inference, and document error sets for invalid tensor layouts.
- **Training Loop**: rebuild optimizer modules to leverage async awaitables for gradient aggregation, introduce checkpoint/rollback support, and ensure determinism via seeded RNG wrappers.
- **Benchmark Harness**: update `benchmarks/nn/` to compare old vs. new kernels, capturing throughput, memory footprint, and convergence metrics.

## CI & Benchmark Plan
- Expand GitHub Actions matrix to cover Linux (x86_64, aarch64), macOS (ARM64, x86_64), and Windows (x86_64) with Zig 0.16-dev toolchain cache.
- Add GPU self-hosted runners tagged per backend; nightly workflow executes `zig build gpu-tests -Dgpu-backend=<backend>` and publishes flamegraphs.
- Introduce smoke benchmark job `zig build bench --summary all` with reduced dataset and weekly long-form benchmarks using `tools/run_benchmarks.sh`.
- Set promotion gates: migrations cannot merge without green CI matrix, benchmark regression <5%, and manual sign-off for GPU jobs.

## Observability Requirements
- Emit structured logs using `telemetry/log.zig` with new fields for Zig version, allocator policy, and GPU backend.
- Instrument async runtimes with tracing spans exported to OpenTelemetry collector; include GPU queue timelines and allocator stats.
- Update dashboards (Grafana + Loki + Tempo) with migration-specific panels: build/test duration, GPU kernel occupancy, allocator fragmentation.
- Configure synthetic probes hitting inference/training endpoints for each backend; alert when latency or error budget deviates >10%.

## Security Considerations
- Refresh threat model including GPU driver attack surface; validate sandboxing on macOS via system extensions and Linux via cgroups/SELinux profiles.
- Regenerate SBOM (CycloneDX) using Zig build metadata; ensure supply chain scanning covers new dependencies.
- Enforce secure coding guidelines: no unchecked pointer casts, validated FFI boundaries for CUDA driver, and secrets redaction in observability pipelines.
- Perform dependency audit for tooling scripts; ensure container images updated with security patches alongside Zig upgrade.

## Reviewer Checklist
1. Confirm `.zigversion` and `build.zig.zon` pin Zig 0.16-dev snapshot and match CI images.
2. Validate build graph changes: no deprecated API usage, custom steps compile, `zig fmt` clean.
3. Review allocator updates for leak detection hooks, debug toggles, and documented fallbacks.
4. Execute GPU backend smoke tests (all platforms) and inspect generated shaders/PTX artifacts.
5. Run neural-network benchmarks ensuring performance goals met and convergence plots attached to PR.
6. Verify CI and observability dashboards updated with new metrics, alerts configured, and runbooks linked.
7. Complete security review: SBOM regenerated, threat model updated, secrets scans green.
8. Approve documentation updates across guides, ensuring migration playbook references correct owners and timelines.

## Appendices
- **Reference Commands**:
  - `zig version`
  - `zig build --summary all`
  - `zig build test --summary all`
  - `zig build gpu-tests -Dgpu-backend=<vulkan|metal|cuda>`
  - `tools/run_benchmarks.sh --suite nn --compare-baseline`
- **Escalation Contacts**: #zig-migration Slack channel, on-call rotation (PagerDuty schedule "ABI Platform").
