# ABI Zig 0.16-dev Project — AGENTS.md (Production-Ready Finalized, Expanded)

> **Mission**: Ship a production-grade refactor of ABI—an experimental Zig framework providing a bootstrap runtime and curated feature modules for AI experiments—on Zig 0.16-dev.254+6dd0270a1 (pinned for reproducibility; monitor for 0.16 stable or 1.0 progression per Zig Software Foundation updates, with 0.15.1 as the latest stable release as of August 2025 per official downloads), preserving public APIs while optimizing performance, reliability, and developer ergonomics. This playbook standardizes build, I/O, CLI, GPU/CPU compute, database (WDBX), agents, tests, CI/CD, security, deployment, observability, and dependency management. Incorporate the current repository’s focus on reusable modules (e.g., `abi.ai`, `abi.database` for WDBX), bootstrap CLI for feature summary, and stubs for GPU utilities. Elevate to a lightning-fast, fully optimized AI and ML training stack with enhanced agent runtime, WDBX database (including concurrency), and inline prompts for automated code generation and review. The current executable initializes the framework, emits a textual summary of configured modules, and exits; expand to full CLI subcommands and demos. Non-Goals: Selecting a single ML architecture, locking to one GPU backend, or shipping proprietary model weights. This document defines a framework for rapid, safe iteration without regressions, aligned with production Zig best practices like explicit memory management, cross-compilation, comprehensive testing, and mixed Zig/C codebases for legacy integration.

---

## 0) Table of Contents

1. Guardrails (Hard Constraints)  
2. Versioning & Release Policy  
3. Architecture Overview  
4. Target Repository Layout (Authoritative)  
5. Build & Configuration (Zig 0.16-dev)  
6. Feature Flags & Framework Options  
7. I/O & Logging (Writers, Channels, JSON)  
8. CLI Specification (Subcommands, Error Codes)  
9. WDBX Database (Data Model, Concurrency, File Format)  
10. GPU Acceleration (Backends, Fallbacks, Kernels)  
11. Agents Runtime (Pipelines, Middleware, Timeouts)  
12. Web & Connectors (HTTP façade, Plugins)  
13. Dependency Management & Supply Chain  
14. Testing Strategy (Unit, Property, Fuzz, Soak)  
15. Benchmark Suite & Metrics Schema  
16. CI/CD (GitHub Actions, Gates, Artifacts)  
17. Security, Privacy & Supply Chain  
18. Performance Budgets & SLOs  
19. Risk Register & Mitigations  
20. LLM Collaboration (Codex/Tool-Calling Playbook)  
21. Agent-Specific Prompts (Inline)  
22. Patch Templates & PR Hygiene  
23. Migration Guide (Old → New Layout)  
24. Quick Reference Snippets  
25. Acceptance Checklist (Ship-Ready)  
26. Deployment & Orchestration  
27. Observability & Monitoring  
28. Contribution Guidelines & Code of Conduct  
29. Appendix A — Example Diffs & Golden Outputs  
30. Appendix B — Thresholds & Config Files  
31. Appendix C — Glossary  
32. Appendix D — Zig Best Practices Integration  

---

## 1) Guardrails (Hard Constraints)

- **API Stability**: Existing imports must keep working (e.g., `@import("abi").ai.agent.Agent`, `@import("abi").database`). Any rename ships a shim in `src/root.zig` + CHANGELOG deprecation. Maintain compatibility with `abi.wdbx` namespace for database helpers.
- **Semantic Preservation**: Examples/tests retain behavior unless an improvement is deliberately gated and documented.
- **No Perf Regressions**: Hot paths — WDBX insert/search, agent `process`, SIMD/GPU kernels — meet or beat baseline P50/P95.
- **Allocator Discipline**: No hidden allocations. Every heap use goes through a caller-provided `std.mem.Allocator`. Ownership is explicit.
- **Cross-Platform**: Linux, macOS, Windows. WASM/WebGPU behind `-Denable-web`.
- **Deterministic Builds**: Pinned `.zigversion` to 0.16.0-dev.254+6dd0270a1. Reproducible `zig build`. No net fetch at build.
- **Typed Errors**: Use `error{...}` sets; avoid sentinel returns.
- **Observability**: Structured logs at all layers; metrics opt-in without code changes.
- **Repository Alignment**: Build on existing modules (`abi.ai`, `abi.database`, `abi.gpu`, `abi.web`, `abi.monitoring`, `abi.connectors`, `abi.VectorOps` from `abi.simd`); expand GPU from CPU-backed stubs to full backends, inspired by production ML stacks like ZML for tensor ops.
- **Production Readiness**: Align with Zig's emphasis on robustness (no UB, explicit errors) and integrate ML-specific libs like ggml-zig for tensor acceleration always feasible; follow mixed Zig/C best practices for gradual migration of legacy code.

---

## 2) Versioning & Release Policy

- **SemVer (pre-release)**: 0.1.0-alpha.N → 0.1.0-beta.N → 0.1.0. Stable surface targets 0.2.x; plan for 1.0 alignment with Zig 1.0 (TBD, est. post-2025 per community discussions).
- **Deprecations**: Announce in CHANGELOG, provide shims for 1 minor, include migration notes.
- **Artifacts**: CLI binary, optional lib, docs site, benchmark JSON, SBOM, Docker images.
- **Zig Support**: `.zigversion` pins a known-good 0.16-dev snapshot (0.16.0-dev.254+6dd0270a1); automate updates via CI with regression tests; prepare for 0.15.1 stable upgrades as per 2025 patches.
- **Release Automation**: Use semantic-release in CI for tagging, changelog gen, and GitHub releases; include SBOM signing.

---

## 3) Architecture Overview

```
+--------------------+     +---------------------+     +------------------+
|   CLI / HTTP API   | --> |   Framework Runtime  | --> |   Feature Graph   |
| (human & JSON I/O) |     |  (init, toggles,    |     | (ai, db, gpu,     |
|                    |     |   plugin discovery)  |     |  web, monitoring) |
+--------------------+     +---------------------+     +------------------+
                                                    |           |
                                                    v           v
                                              +---------+   +-------+
                                              |  WDBX   |   |  GPU  |
                                              | Vector  |   | Kernels|
                                              |  Store  |   | (WGSL/ |
                                              +---------+   |  CPU)  |
                                                             +-------+
                                                             |
                                                             v
+--------------------+     +---------------------+     +------------------+
|   Observability    | <-- |   Deployment Layer  | <-- |   ML Integrations|
| (Prometheus/OTEL)  |     |  (Docker/K8s Helm)  |     | (ggml-zig/ZML)   |
+--------------------+     +---------------------+     +------------------+
                                |
                                v
+--------------------+
|   Dependencies     |
| (Managed via Zig)  |
+--------------------+
```

---

## 4) Target Repository Layout (Authoritative)

```
src/
├── mod.zig                   # Public surface: abi.framework, abi.features, re-exports
├── root.zig                  # Back-compat shims & deprecations
├── main.zig                  # Legacy CLI (compat; currently prints feature summary)
├── comprehensive_cli.zig     # Modern typed CLI (subcommands)
├── framework/
│   ├── mod.zig
│   ├── config.zig            # Feature enum + toggles + FrameworkOptions
│   ├── runtime.zig           # Lifecycle, registry, plugin discovery, summary writer
│   └── state.zig             # Shared runtime structs (optional)
├── features/
│   ├── mod.zig               # Aggregates feature namespaces
│   ├── ai/
│   │   ├── mod.zig
│   │   ├── agent.zig         # Baseline Agent API; expand existing lightweight prototype
│   │   ├── enhanced_agent.zig # Pipelines, middleware, logging hooks
│   │   └── transformer.zig   # NN adapters (stubs ok; integrate ggml-zig tensors)
│   ├── database/
│   │   ├── mod.zig
│   │   ├── database.zig      # WDBX: insert/search/update/delete; build on existing vector components
│   │   ├── config.zig        # Storage + index parameters
│   │   ├── file_format.zig   # On-disk layout & versioning
│   │   └── http.zig          # Optional façade (feature-gated; expand existing front-end)
│   ├── gpu/
│   │   ├── mod.zig
│   │   ├── compute/          # tensor ops + CPU fallback; expand from current stubs
│   │   ├── backends/         # vulkan/metal/webgpu adapters
│   │   └── optimizations.zig # SIMD/GPU optimizations; integrate abi.VectorOps
│   ├── web/
│   │   ├── mod.zig
│   │   ├── http_client.zig
│   │   └── web_server.zig    # Minimal HTTP scaffolding; align with existing WDBX demo
│   ├── monitoring/
│   │   ├── mod.zig           # metrics, tracing, sinks; build on logging helpers
│   │   └── sinks/*.zig       # Prometheus/OTEL exporters
│   └── connectors/
│       ├── mod.zig
│       └── plugin.zig        # Plugin interface; expand placeholder for integrations
├── shared/
│   ├── mod.zig               # Logging façade, plugin loader façade
│   ├── core/                 # lifecycle helpers, error types
│   ├── utils/                # json/math/crypto/net/http helpers
│   ├── logging/
│   ├── platform/
│   └── simd/                 # SIMD helpers (re-export as abi.VectorOps)
└── core/
    └── collections.zig       # containers & cleanup utils

examples/
├── quickstart_agent.zig
├── db_usage.zig
└── gpu_dense_demo.zig

benchmarks/
├── db_insert_search.zig
├── simd_vector_ops.zig
└── gpu_dense_forward.zig

deployment/
├── Dockerfile
├── docker-compose.yml
├── k8s/
│   ├── deployment.yaml
│   └── helm-chart/
└── terraform/

docs/
├── MODULE_ORGANIZATION.md    # Module structure overview (update with current modules)
├── MODULE_REFERENCE.md       # Public API reference (generated from Zig sources)
├── EXAMPLES.md               # Working code samples (include README example)
├── GPU_AI_ACCELERATION.md    # GPU setup and usage (note current CPU stubs)
├── DEPLOYMENT.md             # Containerization and orchestration
├── DEPENDENCY_MANAGEMENT.md  # Best practices for deps
└── TESTING_STRATEGY.md       # Test approach and coverage

scripts/
├── perf_summary.py
└── release.sh                # Semantic release automation

tests/
└── <mirror feature tree with *_test.zig files>

deps/
└── <managed via Zig package manager or git submodules>
```

- **Expansion Note**: Added `docs/DEPENDENCY_MANAGEMENT.md` and `deps/` for production dependency handling.

---

## 5) Build & Configuration (Zig 0.16-dev)

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const abi_mod = b.addModule("abi", .{ .root_source_file = .{ .path = "src/mod.zig" } });

    const enable_gpu = b.option(bool, "enable-gpu", "Enable GPU features") orelse false;
    const enable_web = b.option(bool, "enable-web", "Enable Web/WebGPU") orelse false;
    const enable_mon = b.option(bool, "enable-monitoring", "Enable metrics") orelse false;

    const exe = b.addExecutable(.{
        .name = "abi",
        .root_source_file = .{ .path = "src/comprehensive_cli.zig" },
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("abi", abi_mod);
    if (enable_gpu) { /* link gpu deps here */ }
    b.installArtifact(exe);

    const unit = b.addTest(.{ .root_source_file = .{ .path = "src/mod.zig" }, .target = target, .optimize = optimize });
    unit.root_module.addImport("abi", abi_mod);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&unit.step);

    const bench_step = b.step("bench", "Run benchmark suite");
    // add benchmark executables

    const docs_step = b.step("docs", "Generate docs");
    // attach doc generation

    const sbom_step = b.step("sbom", "Generate SBOM");
    // integrate syft or custom SBOM gen
}
```

- **Production Note**: Enable cross-compilation for deployment targets; use `zig build -Dtarget=aarch64-linux-gnu` for ARM servers; add SBOM generation step.

---

## 6) Feature Flags & Framework Options

```zig
pub const Feature = enum { ai, database, gpu, web, monitoring, connectors, simd };

pub const FeatureToggles = packed struct {
    ai: bool = true,
    database: bool = true,
    gpu: bool = false,
    web: bool = false,
    monitoring: bool = false,
    connectors: bool = false,
    simd: bool = true,
};

pub const FrameworkOptions = struct {
    features: FeatureToggles = .{},
    plugin_paths: []const []const u8 = &.{},
    log_level: enum { error, warn, info, debug, trace } = .info,
    metrics_export: ?[]const u8 = null, // e.g., "prometheus:9090"
    dep_versions: std.StringHashMap([]const u8) = .{}, // For pinned deps
};
```

Usage:

```zig
var fw = try abi.init(alloc, .{ .features = .{ .gpu = true, .database = true }, .log_level = .debug, .metrics_export = "prometheus:9090" });
```

- **Expansion Note**: Added `dep_versions` for production dep pinning.

---

## 7) I/O & Logging (Writers, Channels, JSON)

```zig
const std = @import("std");

pub const Channels = struct {
    out: std.io.AnyWriter, // machine (JSON)
    err: std.io.AnyWriter, // human logs
};

pub fn printJson(out: std.io.AnyWriter, comptime fmt: []const u8, args: anytype) !void {
    try out.print(fmt, args);
    try out.print("\n", .{});
}

pub const Logger = struct {
    level: u8,
    err: std.io.AnyWriter,
    pub fn info(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
        if (self.level >= 3) try self.err.print(fmt, args);
    }
};
```

- **Channel Separation**: stdout for machine-readable JSON; stderr for human logs. `--json` forces structured output.
- **Structured Events**: Prefer `{ "ts": ..., "level": ..., "msg": ..., "fields": { ... } }`; integrate OTEL spans for tracing.
- **Production Note**: Rate-limit logs to prevent floods; rotate files for long-running processes.

---

## 8) CLI Specification (Subcommands, Error Codes)

**Subcommands**
- `features list|enable|disable` — inspect/toggle runtime features.
- `agent run --name <str> [--json]` — run demo agent; read stdin, write stdout.
- `db insert --vec <file|inline> [--meta <json>]` — add vectors.
- `db search --vec <file|inline> -k <N>` — k-NN query.
- `gpu bench [--size MxN, ...]` — run dense-layer demo with CPU/GPU compare.
- `deps list|update` — manage dependencies (new for production).

**Error Codes**
- 0 success; 1 usage; 2 config; 3 runtime; 4 I/O; 5 backend missing; 6 dep conflict.

**Help Sketch**

```
abi 0.1.0-alpha — AI/ML framework (Zig 0.16-dev)

USAGE:
  abi <command> [options]

COMMANDS:
  features   list|enable|disable runtime features
  agent      run demo agents
  db         insert/search vectors (WDBX)
  gpu        run GPU/CPU demos and benchmarks
  deps       list|update dependencies

Run 'abi <command> --help' for details.
```

- **Production Note**: Validate inputs per OWASP API Security Top 10; rate-limit CLI for scripted abuse; add deps subcommand for production.

---

## 9) WDBX Database (Data Model, Concurrency, File Format)

**Data Model**
- **Vectors**: fixed f32 dim D (configurable).
- **Metadata**: optional UTF-8 bytes or JSON.
- **IDs**: monotonic u64.

**Concurrency**
- Sharded index maps (N shards by ID hash) with independent locks or lock-free queues (fast path).
- Writer appends to segment; readers snapshot per shard.

**On-Disk Header**

```zig
struct FileHeader {
    magic: [4]u8 = "WDBX",
    version: u32 = 1,
    dim: u32,
    index_kind: u32, // 0=IVF-Flat, 1=HNSW (reserved)
    reserved: [16]u8,
}
```

**API Surface**

```zig
pub const Database = struct {
    pub fn init(alloc: std.mem.Allocator, cfg: DatabaseConfig) !Database;
    pub fn insert(self: *Database, vec: []const f32, meta: ?[]const u8) !u64;
    pub fn search(self: *Database, query: []const f32, k: usize) ![]SearchResult;
    pub fn update(self: *Database, id: u64, vec: []const f32) !void;
    pub fn delete(self: *Database, id: u64) !void;
    pub fn deinit(self: *Database) void;
};
```

**Recall/Precision Gates**
- IVF-Flat: recall@10 ≥ 0.95 on Gaussian synthetic sets with D∈{64,128}, N∈{10k,100k}.

- **Production Note**: Encrypt sensitive metadata at rest; support sharding across nodes for horizontal scaling; integrate with distributed systems like Raft for replication.

---

## 10) GPU Acceleration (Backends, Fallbacks, Kernels)

- **Backends**: Vulkan, Metal, WebGPU (CUDA future work; integrate ZML for MLIR-based kernels).
- **Fallback**: Tuned CPU SIMD fallback always available.
- **Kernels**: Matmul, add/mul, activation (ReLU/GELU), reduce; add custom ML kernels like convolution.
- **Self-Check**: At init, dispatch tiny kernel, verify correctness, enable backend.

**CPU Fallback Sketch**

```zig
fn matmul(out: []f32, a: []const f32, b: []const f32, m: usize, n: usize, p: usize) void {
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var k: usize = 0;
        while (k < p) : (k += 1) {
            var sum: f32 = 0.0;
            var j: usize = 0;
            while (j < n) : (j += 1) sum += a[i*n + j] * b[j*p + k];
            out[i*p + k] = sum;
        }
    }
}
```

- **Production Note**: Resource limits (VRAM caps); graceful degradation on OOM; multi-GPU support via Zig's async.

---

## 11) Agents Runtime (Pipelines, Middleware, Timeouts)

- **API**: init/process/deinit with typed config; pass allocator.
- **Middleware**: Pre/post hooks for logging, metrics, auth, tracing; optional retries.
- **Cancellation & Timeouts**: Optional deadline/context handle.
- **Examples**: EchoAgent (echo), TransformAgent (uppercase), PipelineAgent (compose); extend with LLM inference via zig-ml; add distributed agents for scale.

```zig
pub const Agent = struct {
    name: []const u8,
    pub fn init(alloc: std.mem.Allocator, opts: struct{ name: []const u8 }) !Agent { _ = alloc; return .{ .name = opts.name }; }
    pub fn process(self: *Agent, input: []const u8, alloc: std.mem.Allocator) ![]u8 { _ = alloc; return input; }
    pub fn deinit(self: *Agent) void { _ = self; }
};
```

- **Production Note**: Async processing for concurrency; fault isolation with arenas.

---

## 12) Web & Connectors (HTTP façade, Plugins)

- **HTTP Server**: Feature-gated minimal server exposing `/db/insert`, `/db/search`, `/agent/run` with JSON payloads; use Zap for high-perf routing.
- **Plugins**: Load plugins via shared objects or Zig modules registered in `FrameworkOptions.plugin_paths`; support dynamic reloading for zero-downtime.

- **Production Note**: TLS termination, CORS, JWT auth; rate limiting with token buckets; gRPC support for connectors.

---

## 13) Dependency Management & Supply Chain

- **Zig Package Manager**: Use `zig fetch` for deps; pin in `build.zig.zon`.
- **Best Practices**: No global deps; explicit imports; audit with `zig ast-check`; follow mixed Zig/C guidelines for C libs (e.g., translate.h for headers).
- **Pinning**: Version locks in `build.zig`; CI verifies reproducibility.
- **Updates**: Semantic versioning checks; automated dep upgrades via dependabot.
- **Supply Chain**: SBOM for deps; vulnerability scans on PRs.

- **Expansion Note**: New section for production dep handling, per 2025 best practices.

---

## 14) Testing Strategy (Unit, Property, Fuzz, Soak)

- **Unit**: Each public fn has positive/negative tests; `std.testing.refAllDecls` anchors coverage.
- **Property**: Random vectors for DB; invariants (idempotent delete, monotonic IDs, recall bounds).
- **Fuzz**: CLI parsing, HTTP handlers, file parsers; use Zig's built-in fuzzing or honggfuzz integration.
- **Soak/Stress**: Concurrency (N writers, M readers) for WDBX; GPU self-checks in loop.
- **Golden Files**: Stable JSON outputs for CLI subcommands.
- **Conformance**: ML-specific tests against ONNX runtime benchmarks; end-to-end with Docker Compose.
- **Integration**: Dep injection tests; mock backends.

- **Production Note**: Achieve 90%+ branch coverage; mutate tests for resilience; add dep-related tests.

---

## 15) Benchmark Suite & Metrics Schema

- **DB**: insert N, search k → ops/s, P50/P95/P99, recall@k.
- **SIMD**: vector ops times; numeric epsilon checks; integrate with `abi.VectorOps`.
- **GPU**: dense forward grid; CPU vs GPU speedup; tolerance; start from current stubs.
- **Scalability**: Multi-node WDBX sharding benchmarks; dep impact measurements.

**Metrics JSON**

```json
{
  "suite": "db_insert_search",
  "commit": "<sha>",
  "zig": "0.16-dev",
  "config": {"dim": 128, "n": 100000, "k": 10},
  "results": {"ops_per_sec": 52341.2, "p50_ms": 1.9, "p95_ms": 6.2, "recall_at_10": 0.962}
}
```

- **Production Note**: CI-integrated; historical trending.

---

## 16) CI/CD (GitHub Actions, Gates, Artifacts)

Expanded YAML with dep scans and depabot.

```yaml
name: abi-ci
on: [push, pull_request]
jobs:
  build-test:
    strategy: { matrix: { os: [ubuntu-latest, macos-latest, windows-latest] } }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: mlugg/setup-zig@v2
        with: { version: master }
      - name: Build
        run: zig build -Doptimize=ReleaseSafe
      - name: Test
        run: zig build test
      - name: Lint
        run: zig fmt --check .
      - name: Security Scan
        uses: aquasecurity/trivy-action@master
        with: { scan-type: 'fs', format: 'sarif', output: 'trivy-results.sarif' }
      - name: Dep Scan
        run: zig deps check  # Custom script for dep auditing
  benchmarks:
    if: github.event_name == 'push'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: mlugg/setup-zig@v2
        with: { version: master }
      - name: Bench
        run: zig build bench || true
      - name: Upload metrics
        uses: actions/upload-artifact@v4
        with: { name: bench-metrics, path: benchmarks/out/*.json }
  release:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    steps:
      - uses: cycjimmy/semantic-release-action@v4
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
```

**Gates**: PRs may not regress perf beyond thresholds; fail on build/test/lint/security/dep scans.

- **Production Note**: Multi-stage Docker builds; vulnerability gating with Trivy; depabot for auto-updates.

---

## 17) Security, Privacy & Supply Chain

- **SBOM**: Generate SBOM per release via `syft` or Zig build hooks; no opaque blobs in repo.
- **Secrets**: No tokens in code; use env/CI secrets; scan with GitHub Secret Scanning.
- **HTTP Inputs**: Validate sizes and JSON schema; sane timeouts; OWASP API Sec Top 10 compliance.
- **Sandbox**: GPU backend runtime-check; deny by default when missing; seccomp profiles for containers.
- **Supply Chain**: SBOM for deps; vulnerability scans on PRs; pin deps in `build.zig.zon`.

- **Expansion Note**: Added dep pinning details.

---

## 18) Performance Budgets & SLOs

- **DB Insert**: ≥ 50k ops/s on 8-core desktop @ D=128.
- **Search P95**: ≤ 10 ms for k=10 @ N=100k (IVF-Flat baseline).
- **Agent process**: ≤ 1 ms echo; ≤ 5 ms with middleware.
- **GPU Dense Forward**: ≥ 5× CPU speedup for 32×784 • 784×128 on mid-tier GPU (or clean fallback).
- **SLOs**: 99.9% uptime for HTTP endpoints; <100ms p99 latency under load.
- **Dep Impact**: Benchmarks pre/post dep updates.

- **Production Note**: Profile with Zig's built-in tools or perf; set alerts on SLO breaches; monitor dep perf.

---

## 19) Risk Register & Mitigations

- **Zig dev churn** → Pin `.zigversion`; isolate std-dependent shims; CI canary builds on master.
- **GPU variance** → Mandatory CPU fallback; init self-test; clear errors. Expand from current stubs carefully.
- **Concurrency bugs** → Sharding; property tests; soak harness; contention profiling with perf.
- **API drift** → Shims + deprecations; doc sync; example compile checks in CI.
- **Perf regressions** → Bench gates; thresholds JSON; trend graphs in CI artifacts.
- **Supply Chain Attacks** → SBOM verification; dep pinning; reproducible builds.
- **Dep Conflicts** → Zon pinning; CI dep scans.

---

## 20) LLM Collaboration (Codex/Tool-Calling Playbook)

- **Operating Mode**: LLM acts as change proposer. Must emit unified diffs or full files plus tests & docs.
- **Safety Rails**:
  - Don’t delete public APIs without shims + tests.
  - Update docs/examples with code.
  - Include rationale/trade-offs.
  - Validate deps in patches.
- **Artifacts**: Patches, tests, docs, benchmark updates in one response per PR.

**Tool Schema (example)**

```json
{
  "tools": [
    {"name":"create_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}},
    {"name":"modify_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"patch":{"type":"string"}},"required":["path","patch"]}},
    {"name":"run_tests","parameters":{"type":"object","properties":{}}},
    {"name":"run_benchmarks","parameters":{"type":"object","properties":{}}},
    {"name":"lint_code","parameters":{"type":"object","properties":{"code":{"type":"string"}}}},
    {"name":"format_code","parameters":{"type":"object","properties":{"code":{"type":"string"}}}},
    {"name":"check_deps","parameters":{"type":"object","properties":{"code":{"type":"string"}}}}
  ]
}
```

**Master Orchestration Prompt**

**Role**: Senior Zig engineer.  
**Goal**: Apply Phases A–H with zero regression. Produce patches + tests + docs.  
**Inputs**: current tree, baselines, this AGENTS.md.  
**Rules**: explicit allocators; no public API breaks; JSON/human I/O split; add tests/docs; align with production practices (e.g., security scans, container builds, dep pinning).  
**Output**: unified diffs + new files; updated build.zig; CI YAML; rationale for every change.

---

## 21) Agent-Specific Prompts (Inline)

**Build Agent – build.zig**  

Refactor build.zig to Zig 0.16-dev. Define module imports for framework, features/*, shared. Add `-Denable-gpu`/`-web`/`-monitoring`. Produce abi CLI at zig-out/bin/abi and a unit-test step. Return the complete build.zig and a one-paragraph rationale. Integrate existing `abi.simd`; add Docker multi-stage support; pin deps in zon.

**I/O Agent – Writers**  

Introduce a Logger façade and channel separation (stdout JSON, stderr logs). Replace std.debug.print. Provide unit tests asserting no interleaving and correct JSON under `--json`. Integrate with existing monitoring helpers; add OTEL tracing hooks.

**CLI Agent – Subcommands**  

Create comprehensive_cli.zig with subcommands: features list|enable|disable, agent run, db insert|search, gpu bench, deps list|update. Implement `--help` autogen and robust validation errors with exit codes. Provide golden-output tests for human and JSON modes. Expand from current bootstrap summary; add rate-limiting.

**Database Agent (WDBX) – Concurrency & Persistence**  

Implement thread-safe insert/search/update/delete with typed errors. Provide a stable on-disk layout and version header. Add property tests (random vectors) measuring recall@k and latency. Include a minimal HTTP façade behind a feature flag. Build on existing vector components and front-ends; add encryption for metadata.

**GPU Agent – Tensor Core + Fallback**  

Add matmul/add/mul/activation with CPU SIMD fallback. Implement backend detection (vulkan/metal/webgpu) and a self-check routine. Provide a dense-forward demo and a test comparing CPU vs GPU outputs within epsilon. Expand from current CPU stubs; integrate ggml-zig for advanced tensors.

**AI Agent – Runtime & Middleware**  

Define Agent API (init/process/deinit) with typed config and middleware. Ship EchoAgent and TransformAgent. Add lifecycle tests and allocator-ownership documentation. Expand existing lightweight prototype; support LLM inference via zig-ml.

**Tests/CI Agent – Matrix + Gates**  

Add GH Actions matrix (linux/macos/windows). Cache Zig. Run zig fmt --check, build, test, and upload docs/bench artifacts. Set performance gates via a JSON thresholds file and fail PRs on regression; add Trivy security scans, SBOM gen, and dep checks.

**Docs Agent – Single Source of Truth**  

Update docs/EXAMPLES.md, MODULE_ORGANIZATION.md, and MODULE_REFERENCE.md so samples compile against final APIs. Add quick-start, configuration flags, troubleshooting, benchmark methodology, and dependency management sections. Ensure MODULE_REFERENCE.md matches public APIs; generate from Zig sources as in current repo; include deployment guides.

---

## 22) Patch Templates & PR Hygiene

**Conventional Commits**: feat:, fix:, perf:, refactor:, docs:, test:, chore:, ci:.

**PR Body Template**

### Summary  
<what changed>

### Motivation  
<why this is needed>

### Public API  
<unchanged | list changes + shims>

### Tests  
<list new/updated tests>

### Performance  
<before vs after numbers; attach JSON>

### Docs  
<updated files>

### Risks  
<what could go wrong + mitigations>

### Security/Compliance  
<scans passed; SBOM updates>

### Dep Changes  
<new/updated deps; rationale>

---

## 23) Migration Guide (Old → New Layout)

- **Imports**: `@import("abi").ai.agent.Agent` unchanged; if relocating types, add re-exports in src/mod.zig.
- **build.zig**: Legacy scripts continue to work; new flags optional.
- **CLI**: main.zig remains; comprehensive_cli.zig is the modern entry.
- **Docs**: Examples updated; old paths listed with deprecation badges.
- **Deployment**: New Docker/K8s configs; migrate via docker-compose up.
- **Deps**: Move to zon pinning; update scripts accordingly.

---

## 24) Quick Reference Snippets

**Agent usage** (From repository README)

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var framework = try abi.init(alloc, .{});
    defer abi.shutdown(&framework);

    const Agent = abi.ai.agent.Agent;
    var agent = try Agent.init(alloc, .{ .name = "EchoAgent" });
    defer agent.deinit();

    const reply = try agent.process("Hello, ABI!", alloc);
    defer alloc.free(@constCast(reply));

    try std.io.getStdOut().writeAll(reply);
}
```

**CLI outline**

```zig
// comprehensive_cli.zig
pub fn main() !void { /* parse args -> dispatch subcommands */ }
```

**Docker Build**

```dockerfile
FROM ziglang/zig:latest AS builder
WORKDIR /app
COPY . .
RUN zig build -Drelease-safe -Dtarget=x86_64-linux-gnu

FROM gcr.io/distroless/base:latest
COPY --from=builder /app/zig-out/bin/abi /usr/local/bin/abi
CMD ["abi", "features", "list"]
```

**Dep Pinning in build.zig.zon**

```zig
.{
    .name = "abi",
    .version = "0.1.0",
    .dependencies = .{
        .ggml_zig = .{
            .url = "git+https://github.com/example/ggml-zig#v1.2.3",
            .hash = "sha256-...",
        },
    },
}
```

---

## 25) Acceptance Checklist (Ship-Ready)

- [ ] zig build clean; CI matrix green; zig fmt passes; security scans clean.
- [ ] abi features list prints coherent enabled features in human/JSON modes, aligning with current repo summary.
- [ ] abi agent run executes demo agents; ownership rules documented.
- [ ] abi db insert/search passes property tests and meets latency thresholds under concurrent loads.
- [ ] GPU demo runs or cleanly falls back to CPU; expanded from stubs.
- [ ] Docs updated; examples compile; CHANGELOG includes refactor notes.
- [ ] Docker image builds/runs; K8s manifests validate.
- [ ] SBOM generated; release artifacts published.
- [ ] Deps pinned and scanned.

---

## 26) Deployment & Orchestration

- **Containerization**: Multi-stage Dockerfiles for slim images (<50MB); use distroless base for security.
- **Orchestration**: Helm charts for K8s; support auto-scaling based on CPU/GPU metrics.
- **CI Integration**: Build/push images on tag; scan with Trivy in pipeline.
- **Zero-Downtime**: Health checks (`/healthz` endpoint); rolling updates with readiness probes.
- **Configs**: Env vars for FrameworkOptions; secrets via K8s Secrets or Vault.
- **Scaling**: Horizontal pod autoscaler; GPU scheduling with node affinity.

**Example Helm Values**

```yaml
replicaCount: 3
image:
  repository: ghcr.io/donaldfilimon/abi
  tag: "0.1.0"
resources:
  limits:
    nvidia.com/gpu: 1  # For GPU workloads
service:
  type: LoadBalancer
  port: 80
```

- **Production Note**: Support serverless (e.g., Knative) for bursty ML inference; Terraform for infra as code.

---

## 27) Observability & Monitoring

- **Metrics**: Prometheus exporter in `monitoring/sinks/prometheus.zig`; expose `/metrics` endpoint.
- **Tracing**: OTEL integration for distributed traces; sample at 1% for prod.
- **Logging**: Structured JSON to stdout; aggregate with Fluentd/ELK.
- **Alerting**: SLO breaches trigger PagerDuty; dashboards in Grafana for latency/recall.
- **Profiling**: Continuous profiling with Zig's async I/O hooks; export to Pyroscope.
- **Dep Monitoring**: Track dep perf in benchmarks.

**Prometheus Metrics Example**

```
# HELP abi_db_insert_duration_seconds Insert latency
# TYPE abi_db_insert_duration_seconds histogram
abi_db_insert_duration_seconds_bucket{le="0.005"} 100
```

- **Production Note**: Golden signals (latency, traffic, errors, saturation); integrate with existing monitoring helpers; dep impact alerts.

---

## 28) Contribution Guidelines & Code of Conduct

- **Guidelines**: Fork/PR model; DCO 1.1 sign-off; small PRs (<400 LOC); dep updates in separate PRs.
- **Code of Conduct**: Adopt Contributor Covenant; enforce via CLA bot.
- **Onboarding**: CONTRIBUTING.md with setup, phases from this doc, and LLM prompt templates.
- **Reviews**: Require 2 approvals; lint/security gates; post-merge benchmarks.
- **Community**: Discord/Slack for discussions; monthly office hours.
- **Dep Contributions**: Propose new deps with SBOM and perf data.

- **Production Note**: Maintain audit trail for compliance (e.g., SOC 2 if enterprise).

---

## 29) Appendix A — Example Diffs & Golden Outputs

**Unified diff — writer channel separation (illustrative)**

```
--- a/src/old_cli.zig
+++ b/src/comprehensive_cli.zig
@@
- std.debug.print("{s}\n", .{msg});
+ if (json_mode) {
+     try printJson(ch.out, "{\"msg\":\"{s}\"}", .{msg});
+ } else {
+     try logger.info("{s}\n", .{msg});
+ }
```

**Golden JSON output — features list --json**

```json
{"features":{"ai":true,"database":true,"gpu":false,"web":false,"monitoring":false,"connectors":false,"simd":true}}
```

**Example SBOM Snippet (cyclonedx-json)**

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "components": [
    {"type": "library", "name": "zig-std", "version": "0.16-dev"},
    {"type": "library", "name": "ggml-zig", "version": "1.2.3"}
  ]
}
```

---

## 30) Appendix B — Thresholds & Config Files

**bench_thresholds.json**

```json
{
  "db_insert_search": { "p95_ms_max": 10.0, "ops_per_sec_min": 50000 },
  "gpu_dense_forward": { "speedup_vs_cpu_min": 5.0 },
  "simd_vector_ops": { "p95_ms_max": 2.0 }
}
```

**abi.config.example.json**

```json
{
  "features": { "ai": true, "database": true, "gpu": false, "web": false, "monitoring": false, "connectors": false, "simd": true },
  "plugin_paths": ["./plugins"],
  "log_level": "info",
  "metrics_export": "prometheus:9090"
}
```

**docker-compose.yml (Dev)**

```yaml
version: '3.8'
services:
  abi:
    build: .
    ports:
      - "8080:8080"
    environment:
      - ABI_METRICS_EXPORT=prometheus:9090
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
```

**build.zig.zon Example**

```zig
.{
    .name = "abi",
    .version = "0.1.0",
    .dependencies = .{
        .ggml_zig = .{
            .url = "git+https://github.com/example/ggml-zig#v1.2.3",
            .hash = "sha256-...",
        },
        .zap = .{
            .url = "git+https://github.com/zigzap/zap#v0.1.7-pre",
            .hash = "sha256-...",
        },
    },
}
```

---

## 31) Appendix C — Glossary

- **ABI**: Application Binary Interface (here: also name of the framework).
- **WDBX**: Native vector database component.
- **IVF-Flat/HNSW**: Vector search index types.
- **P50/P95**: Latency percentiles.
- **Shim**: Back-compat adapter layer.
- **Golden file**: Canonical expected output used by tests.
- **SLO**: Service Level Objective.
- **SBOM**: Software Bill of Materials.
- **OTEL**: OpenTelemetry.
- **Zon**: Zig package manifest file.

---

## 32) Appendix D — Zig Best Practices Integration

- **Structuring Projects**: Use modules for separation; explicit deps in zon; avoid globals.
- **Mixed Zig/C**: Use translate-c for headers; gradual migration with safety checks.
- **Error Handling**: Union error sets; exhaustive switches; no panics in prod.
- **Performance**: Comptime where possible; profile hot paths; SIMD opts.
- **Deps**: Pin hashes; audit regularly; prefer Zig-native libs.
- **2025 Trends**: Async I/O for servers; MLIR integration for ML; container-first deploys.

---

**End of AGENTS.md (Production-Ready Finalized, Expanded)**. Treat this as the SSOT for the refactor sprint. Assign owners per phase, stage PRs incrementally, and enforce quality gates. Default to allocator clarity, measurable performance, and API stability. For enterprise adoption, monitor Zig 1.0 progress and integrate with mature ML stacks like ZML.
