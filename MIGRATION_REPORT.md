# Zig 0.16-dev Migration Playbook

This document is the authoritative playbook for the Zig 0.16-dev refactor. It defines strategy, ownership, deliverables, and the
technical depth required so every workstream can execute autonomously while staying aligned with the overall migration intent.

---

## 1. Strategy, Scope, and Governance

### 1.1 Strategic Objectives
- **Maintain feature parity** while adopting Zig 0.16-dev language, stdlib, and build-system changes across all supported
  products.
- **Unlock GPU acceleration** across Vulkan, Metal, and CUDA backends by aligning with the refreshed async/event APIs and build
  knobs.
- **Lay neural-network foundations** for tensor-core optimized kernels, fused ops, and resilient training workflows.
- **De-risk future upgrades** by codifying testing, observability, and security guardrails that become the default operating mode
  for the platform.

### 1.2 Guiding Principles
- **Wave-based execution**: deliver incremental, reviewable waves with regression baselines before advancing.
- **Owner accountability**: each milestone has a directly responsible individual (DRI) and an escalation deputy.
- **Fail-fast validation**: expand CI, benchmarks, and telemetry to surface breakage within the same working day.
- **Documentation-first**: every change is accompanied by updated runbooks, reviewer checklists, and migration notes.

### 1.3 Governance Cadence
- **Weekly steering sync** (Dev Infra, Runtime, GPU, ML Systems, SRE/Security) to unblock cross-cutting issues.
- **Daily async standup** in `#zig-migration` Slack with blockers, risk changes, and test status.
- **Change control**: high-risk merges (GPU, allocator) require dual approval from owning team + release engineering.

---

## 2. Milestones, Deliverables, and Owners

| Milestone | Target Date | Success Criteria | Primary Owner | Deputy |
| --- | --- | --- | --- | --- |
| Toolchain pin + build graph audit | 2025-09-22 | `zig build`/`zig build test` green on Linux/macOS/Windows dev images; build.zig graph documented | Dev Infra (A. Singh) | Release Eng (S. Walker) |
| Allocator + stdlib refactor | 2025-10-01 | Allocator policy tests pass; memory diagnostics regression ≤1%; stdlib API usage conforms to 0.16-dev | Runtime (L. Gomez) | Dev Infra (E. Chen) |
| GPU backend enablement | 2025-10-12 | Vulkan/Metal/CUDA smoke + performance tests green; shader toolchain refreshed; feature flags documented | GPU (M. Ito) | ML Systems (R. Chen) |
| Neural-network kernel uplift | 2025-10-20 | Tensor-core matmul ≥1.8× speedup; training convergence parity; ops library API locked | ML Systems (R. Chen) | Runtime (J. Patel) |
| Observability & security sign-off | 2025-10-25 | Dashboards live; tracing coverage checklist complete; SBOM & threat model updated | SRE/Security (K. Patel) | Dev Infra (A. Singh) |
| Final migration review | 2025-10-27 | Reviewer checklist cleared; release notes drafted; rollback plan rehearsed | Release Eng (S. Walker) | Program Mgmt (T. Rivera) |

Milestones are exit criteria for wave completion; each owner is responsible for ensuring regression baselines and documentation
are archived in the migration repository.

---

## 3. Module Inventory & Ownership

| Domain | Key Modules & Artifacts | Owner Team | Notes |
| --- | --- | --- | --- |
| Core Runtime | `src/runtime/allocator.zig`, `src/runtime/context.zig`, `src/runtime/async.zig` | Runtime | Align async primitives with new `std.Channel` semantics; ensure allocator instrumentation hooks emit telemetry. |
| Compiler / Build Glue | `build.zig`, `build.zig.zon`, `scripts/toolchain/*.sh`, `.zigversion` | Dev Infra | Responsible for toolchain pinning, build graph modernization, dependency integrity. |
| GPU Backends | `src/gpu/vulkan/*.zig`, `src/gpu/metal/*.zig`, `src/gpu/cuda/*.zig`, `assets/shaders/*`, `tools/shaderc/*` | GPU | Own shader transpilation, backend toggles, and driver validation scripts. |
| Neural Network Stack | `src/nn/tensor.zig`, `src/nn/ops/*.zig`, `src/nn/train/*.zig`, `benchmarks/nn/*`, `tests/nn/*` | ML Systems | Deliver tensor-core kernels, fused ops, and convergence benchmarks. |
| Observability | `src/telemetry/*.zig`, `tools/metrics/*`, dashboards in `reports/observability/*` | SRE | Ensure logging/tracing schema, exporters, and dashboards align with new metrics. |
| Security & Compliance | `tools/security/*.py`, `deploy/*`, container manifests, SBOM scripts | Security | Maintain secure defaults, patch cadence, and compliance reporting. |
| Release Tooling | `.github/workflows/*.yml`, `scripts/release/*`, `docs/release_notes.md` | Release Eng | Gate releases and maintain canary promotion scripts. |

Owners must keep module inventories updated in the runbook after each wave, noting new files or deprecations.

---

## 4. Risk Register and Contingencies

| Risk | Impact | Probability | Owner | Mitigation | Trigger / Contingency |
| --- | --- | --- | --- | --- | --- |
| Upstream Zig breaking change lands mid-migration | Build failures across CI pipelines | Medium | Dev Infra | Track nightly Zig changelog, freeze snapshot after RC, mirror binaries internally | If break detected, revert to last known-good snapshot and raise upstream issue within 24h. |
| GPU driver/toolchain mismatch (Metal 4 / CUDA 12.5) | GPU test failures on macOS/Linux | High | GPU | Maintain driver matrix, pre-warm CI AMIs, provide fallback software rasterizer | If driver incompatibility found, switch CI to fallback runner pool and block merges touching GPU code. |
| Allocator regression introduces leaks | Runtime instability in prod workloads | Medium | Runtime | Expand leak detectors, nightly fuzzing, allocator-focused reviews | On leak detection, freeze allocator merges and initiate bisect with instrumentation build. |
| Neural-net kernels miss tensor-core utilization | Performance degradation vs. baseline | Medium | ML Systems | Profile via Nsight/Metal Perf HUD, compare against baseline kernels, vendor escalation | If <1.5× uplift, trigger optimization tiger team to tune kernels before GA. |
| Observability gaps hide regressions | Slow incident response | Low | SRE | Enforce tracing spans, synthetic monitors per backend, alert fatigue review | If SLA breach occurs without alert, halt rollout until observability fixes land. |
| Security regressions from new dependencies | Compliance or vulnerability exposure | Low | Security | Automated SBOM diff scanning, container hardening, secure code review gates | If critical CVE found, revert dependency or apply hotfix within 48h. |

Owners update impact/probability weekly; mitigations must have associated tasks in the migration tracker.

---

## 5. Technical Workstreams

Each subsection captures the delta from pre-0.16 state, required tasks, validation steps, and deliverables.

### 5.1 Build-System Deltas
- **Graph Modernization**: Replace deprecated `.builder` references with the new `b` handle, ensure custom steps allocate scratch
  memory via `b.allocator`, and refactor step dependencies to explicit `dependOn` calls.
- **Module Registration**: Normalize module definitions using `b.addModule("abi", .{ .source_file = .{ .path = "src/mod.zig" } });`
  and migrate dependents to `module.dependOn()` / `createImport()` semantics. Document module graph in `docs/build_graph.md`.
- **Cross Compilation**: Adopt `std.zig.CrossTarget` for target resolution, update target triples (notably WASI preview2 and
  Android API level enums), and regenerate cached artifacts per target.
- **Dependency Metadata**: Regenerate `build.zig.zon` with SHA256 checksums, license fields, and compatibility notes; verify
  `zig fetch` works offline via mirrored tarballs.
- **Toolchain Automation**: Update `scripts/toolchain/*` to install Zig 0.16-dev snapshot, seed caches, and validate using `zig
  env`. Document bootstrap instructions in `DEPLOYMENT_GUIDE.md`.

**Validation:** `zig build`, `zig build test`, `zig fmt`, and dry-run of cross targets (`zig build -Dtarget=wasm32-wasi`). Capture
logs in CI artifact bucket.

### 5.2 Stdlib and API Updates
- **Time APIs**: Swap `std.os.time.Timestamp` for `std.time.Timestamp`, using monotonic clocks for scheduler logic; adjust tests
  to tolerate nanosecond precision.
- **JSON Handling**: Migrate to `std.json.Parser.parseFromSlice`, updating error propagation to typed `error{ParseError,
  InvalidType}` sets and adding fuzz tests for schema drift.
- **Async Primitives**: Refactor to new `std.Channel` behavior (`close()` explicit, iteration semantics changed), audit awaiting
  patterns, and ensure cancellation tokens propagate errors.
- **Reflection Helpers**: Update `std.meta` usages (`trait.fields` → `fields`, `Child` rename) and add compatibility wrappers
  where necessary until downstream consumers migrate.
- **Error Sets & Testing**: Document error-set changes in module docs and regenerate snapshots for API clients.

**Validation:** Run unit tests, API compatibility tests under `tests/api/*`, and contract tests with SDK consumers.

### 5.3 Allocator Policies and Instrumentation
- **Allocator Topology**: Retain `std.heap.page_allocator` as default, introduce scoped `ArenaAllocator` for neural-network graph
  builds, and evaluate `GeneralPurposeAllocator(.{})` for platforms lacking large pages.
- **Instrumentation Hooks**: Wrap allocators with `std.heap.LoggingAllocator` under debug builds; export counters via
  `telemetry/alloc.zig` (alloc/free rate, high-water marks) to dashboards.
- **API Contracts**: Convert ad-hoc `*Allocator` parameters to `anytype` generics where call sites specialize, improving
  optimization opportunities.
- **Fallback Strategy**: Auto-detect huge page support; if unavailable, enable guard-page monitoring and log warnings with
  remediation steps.
- **Testing**: Extend allocator-specific tests under `tests/runtime/allocator.zig`, add soak test nightly job `zig build
  allocator-soak`.

### 5.4 GPU Backend Enablement
- **Feature Flags**: Introduce `-Dgpu-backend=<vulkan|metal|cuda|cpu>` build option with compile-time dispatch tables and
  documented defaults in `docs/gpu_backends.md`.
- **Vulkan**: Adopt `std.vulkan.descriptor` helpers, migrate synchronization to timeline semaphores, validate descriptor set
  layouts via `vkconfig`, and ensure SPIR-V generation pipelines (via `tools/shaderc`) emit debug info toggles.
- **Metal**: Adjust `@cImport` bindings to Zig 0.16 rules, regenerate Metal headers, ensure argument buffers follow new resource
  index macros, and test on macOS ARM/x86 hardware.
- **CUDA**: Align driver API bindings with `extern` calling convention updates, regenerate PTX kernels optimized for `sm_90a`
  tensor cores, and run Nsight Compute regression scripts.
- **Shared Requirements**: Maintain CPU fallback path parity, expose backend telemetry (queue depth, kernel latency), and
  document driver prerequisites.

**Validation:** `zig build gpu-tests -Dgpu-backend=<backend>`, run shader compilation CI job, execute real-hardware canary tests
on staging clusters, and capture flamegraphs.

### 5.5 Neural-Network Foundations
- **Tensor Core Enablement**: Detect architecture capabilities (FP8/BF16/TF32) at runtime, select WMMA kernels accordingly, and
  provide CPU reference implementations for verification.
- **Operator Library**: Refactor `ops` module to expose fused kernels (conv+bias+activation, attention block), implement shape
  inference via iterators, and document tensor layout requirements and error sets.
- **Training Loop**: Rebuild optimizer modules to leverage async awaitables for gradient aggregation, add checkpoint/rollback
  support, deterministic seeding wrappers, and align logging with observability schema.
- **Data Pipeline Integration**: Ensure data loaders adapt to Zig 0.16 IO changes, provide streaming dataset interface, and add
  instrumentation for throughput / latency.
- **Benchmark Harness**: Update `benchmarks/nn/*` to compare pre/post kernels, capture throughput, memory footprint, and
  convergence metrics; surface results in Grafana panel.

**Validation:** `zig build nn-tests`, nightly benchmark suite via `tools/run_benchmarks.sh --suite nn --compare-baseline`, and
compare convergence plots stored in `reports/nn/`.

---

## 6. CI and Benchmark Execution Plan
- **Matrix Expansion**: GitHub Actions workflows cover Linux (x86_64, aarch64), macOS (ARM64, x86_64), and Windows (x86_64) using
  cached Zig 0.16-dev toolchains.
- **GPU Runners**: Add self-hosted runners tagged `gpu:vulkan`, `gpu:metal`, `gpu:cuda`; nightly workflow executes `zig build
  gpu-tests` for each backend and uploads flamegraphs + telemetry snapshots.
- **Benchmark Cadence**: Introduce smoke benchmark job `zig build bench --summary all` per PR; weekly long-form benchmarks via
  `tools/run_benchmarks.sh` with baseline diff reports archived in `reports/benchmarks/`.
- **Promotion Gates**: Merges blocked unless CI matrix green, benchmark regression <5%, GPU jobs manually signed off by owning
  team.
- **Alerting**: CI failure notifications routed to #zig-migration; autopage release engineering on repeated failures.

---

## 7. Observability Requirements
- **Logging**: Emit structured logs via `telemetry/log.zig`, including Zig version, allocator policy, GPU backend, and
  performance counters; ensure logs parse in Loki.
- **Tracing**: Instrument async runtimes with OpenTelemetry spans, capturing queue wait times, GPU submissions, and allocator
  events. Export to Tempo with 7-day retention during migration.
- **Metrics**: Update Grafana dashboards with migration panels (build/test duration, GPU kernel occupancy, allocator
  fragmentation, nn convergence). Provide runbooks linking metrics to remediation steps.
- **Synthetic Monitoring**: Deploy probes per backend hitting inference/training endpoints; configure alerts for >10% latency or
  error budget deviations.
- **Telemetry Validation**: Add CI job `zig build telemetry-test` to verify instrumentation compiles and emits expected schema.

---

## 8. Security and Compliance Considerations
- **Threat Modeling**: Refresh GPU driver attack surface analysis; document isolation strategies (macOS system extensions,
  Linux cgroups/SELinux profiles) in `docs/security/zig016.md`.
- **Supply Chain**: Regenerate CycloneDX SBOM via Zig build metadata, diff against previous release, and feed into dependency
  scanners; ensure new packages meet license policy.
- **Secure Coding**: Enforce guidelines—no unchecked pointer casts, validated FFI boundaries for CUDA driver, secrets redaction in
  telemetry exporters. Integrate with `tools/security/lint.py` pre-submit hook.
- **Container & Runtime Hardening**: Update base images with patched libraries, enable kernel lockdown on GPU nodes, and verify
  TLS certificates for remote shader compilation services.
- **Incident Response**: Define rollback plan and contact tree in `SECURITY.md`; run tabletop exercise before final rollout.

---

## 9. Reviewer Checklist and Exit Criteria
1. Confirm `.zigversion` and `build.zig.zon` pin the approved Zig 0.16-dev snapshot and match CI toolchains.
2. Validate build graph updates: no deprecated APIs remain, custom steps compile, `zig fmt` is clean.
3. Review allocator changes for leak detection hooks, debug toggles, and documented fallbacks (including guard-page logic).
4. Execute GPU backend smoke tests across platforms; inspect generated shaders/PTX artifacts into `reports/gpu/`.
5. Run neural-network benchmarks; ensure performance targets met and convergence plots attached to PR.
6. Confirm CI workflows and observability dashboards updated with new metrics and alerts; provide screenshots or links.
7. Complete security review: SBOM regenerated, threat model updated, secrets scans green, container images signed.
8. Verify documentation updates (guides, runbooks, module inventories) reflect latest ownership and timelines.
9. Ensure rollback procedure tested and logged in release checklist before sign-off.

---

## 10. Appendices
- **Reference Commands**:
  - `zig version`
  - `zig build --summary all`
  - `zig build test --summary all`
  - `zig build gpu-tests -Dgpu-backend=<vulkan|metal|cuda>`
  - `zig build nn-tests`
  - `tools/run_benchmarks.sh --suite nn --compare-baseline`
- **Artifact Repositories**:
  - CI logs: `s3://abi-ci-artifacts/zig016/`
  - Benchmarks: `reports/benchmarks/`
  - Observability dashboards: Grafana folder `Zig 0.16 Migration`
- **Escalation Contacts**: `#zig-migration` Slack channel, PagerDuty schedule “ABI Platform”, program manager T. Rivera.

This playbook should be treated as a living document. Update sections as deliverables close, risks evolve, or upstream Zig
changes introduce new constraints.
