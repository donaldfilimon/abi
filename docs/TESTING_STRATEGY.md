# Testing Strategy

## Purpose
- Provide a unified approach for validating the Abi AI Framework across functionality, performance, security, and reliability.
- Ensure every refactoring milestone in `REFACTORING_PLAN.md` is backed by executable tests and measurable quality gates.
- Create actionable guidance for contributors and CI automation so the 95%+ coverage and performance SLAs remain enforceable.

## Guiding Principles
- **Shift-left validation**: run fast unit and contract tests locally by default.
- **Deterministic by design**: tests must not rely on external services unless explicitly tagged and isolated.
- **Measure everything**: track functional, performance, and security metrics per module and trend them in CI.
- **Automate gating**: no merge without green functional, coverage, lint, and security checks.
- **Document expectations**: every public API and refactoring task references the tests that protect it.

## Test Categories & Scope

### Unit Tests
- Location: `src/**` `test` blocks and mirrored files under `tests/`.
- Scope: pure functions, data structures, protocol adapters, error sets.
- Tooling: `zig test src/path/to/file.zig`, orchestrated via `zig build test` and `make test`.
- Requirements: >95% statement coverage for public APIs, exhaustive error-path assertions, property-based cases where feasible.
- Ownership: module maintainers; enforce new tests for each public symbol change.

### Component / Integration Tests
- Location: `tests/*integration*`, `tests/test_web_server_*.zig`, `tests/test_plugin_*.zig`.
- Scope: subsystem seams (agent routing + datastore, web server + plugins, GPU manager + kernels).
- Tooling: `zig test` with feature flags (`-Dgpu=true`, `-Dsimd=true`, `-Denable_metrics=true`).
- Requirements: cover cross-module behaviors, contract tests for connectors (`src/connectors`), regression harness for database sharding.
- Data: use in-repo fixtures under `tests/fixtures/` (create as needed) and deterministic mocks.

### Performance & Load Tests
- Location: `tests/test_performance_*`, `benchmarks/**`, `performance_reports/`.
- Scope: throughput/latency SLAs (agents, vector search, web API, plugin loading).
- Tooling: `zig build -Doptimize=ReleaseFast perf` (add step below), benchmark harnesses invoking `std.time.Timer` with percentile reporting, optional external load generators via `tools/` scripts.
- Metrics: record p50/p95 latency, max throughput, CPU%, memory footprint. Store outputs in `performance_reports/YYYYMMDD/`.
- Regression Policy: fail CI if deviation >5% from last accepted benchmark or target unmet.

### Security Tests
- Location: dedicated suite `tests/security/` (to be created) plus fuzzers under `tools/`.
- Scope: auth flows, schema validation, input sanitization, encryption, dependency scanning.
- Tooling: static analysis (`zig build security-scan` target), fuzzing via `zig test --fuzz`, third-party scanners integrated in CI (e.g., cargo `cargo-audit` style equivalent for Zig packages when available).
- Requirements: ensure every security-sensitive module (`src/framework`, `src/shared/enhanced_plugin_system.zig`, `src/features/web`) has both positive and negative tests; run dependency and secret scanners on every merge.

### End-to-End Tests
- Location: `tests/test_web_server_e2e.zig`, CLI workflow tests, scripted flows in `examples/` promoted to tests when stable.
- Scope: full agent lifecycle (request → routing → execution → response), plugin install + usage, deployment bootstrap.
- Tooling: orchestrated by `zig build e2e` (target to add), optionally backed by containerized environment via `deploy/compose.yml`.
- Requirements: use tagged scenarios (`test "agent handles vector search happy path" { ... }`) with synthetic data; assert telemetry events and persisted state.

## Coverage & Quality Metrics
- **Code coverage**: compile tests with `-fprofile-instr-generate -fcoverage-mapping`; aggregate via `llvm-profdata`/`llvm-cov` and publish HTML under `zig-out/coverage/`.
- **Branch coverage**: collect from `llvm-cov report`; enforce 90%+ for `src/shared/core`, `src/framework`, `src/features/web`, `src/features/database`, and `src/features/connectors`.
- **Mutation sampling**: quarterly run with `tools/mutagen.zig` (to implement) on critical modules.
- **Static checks**: treat `zig fmt --check .` and `zig build lint` (add target) as mandatory gates.

## Environments
- **Local**: developers run `zig build test`, targeted `zig test path`, and `zig build perf -- --quick` for smoke performance.
- **CI**: matrix across targets (Linux, Windows), feature flags (`-Dgpu`, `-Dsimd`, `-Dhot_reload`), nightly extended runs with `--fuzz` and performance benchmarks.
- **Pre-production**: weekly soak tests using `deploy/staging` scripts, capturing telemetry via Prometheus exporters in `src/monitoring`.

## Test Data & Fixtures
- Centralize fixtures under `tests/fixtures/` with subfolders (`agents/`, `plugins/`, `http/`, `database/`).
- Use `std.json` fixture loaders for HTTP responses, and deterministic seeded random sources for performance tests (`std.rand.DefaultPrng.init(1337)`).
- Secrets/config: store mock credentials in `.env.test`, never in repo; CI injects via secure variables.

## Automation & Tooling Backlog
1. Add `zig build perf`, `zig build e2e`, `zig build security-scan`, and `zig build lint` steps inside `build.zig` for unified orchestration.
2. Provide `make perf`, `make e2e`, `make security`, and `make coverage` wrappers.
3. Extend `scripts/ci/` with PowerShell + Bash entrypoints (`run-tests.ps1`, `run-tests.sh`) to normalize local/CI commands.
4. Generate coverage badges and trend charts in `docs/generated/` during nightly builds.

## Execution Cadence
- **Per commit / PR**: unit, integration, static checks, security scan, coverage threshold.
- **Nightly**: full matrix + performance benchmarks + fuzzing (6h budget).
- **Weekly**: soak tests, mutation sampling on rotating modules, long-running vector search load.
- **Release candidate**: full regression, disaster recovery drills, restore-from-backup verification.

## Reporting & Governance
- Capture results in CI artifacts with normalized JSON schema under `zig-out/reports/`.
- Publish dashboards in `docs/generated/PERFORMANCE_GUIDE.md` and `performance_report.md`.
- Triage failures within one business day; create tracking issues labeled `quality`, `security`, or `performance-regression`.
- Maintain a flaky test suppression list (`tests/flaky_allowlist.json`) with owner, rationale, and sunset date.

## Ownership & Responsibilities
- **Quality Lead**: approves changes to this strategy and signs off on release readiness.
- **Module Owners**: responsible for unit/integration coverage in their domains.
- **Performance Team**: maintains benchmarks, analyzes regressions, updates `performance_reports/`.
- **Security Team**: runs penetration scripts, reviews dependency reports, and validates fixes.
- **Dev Experience**: maintains tooling, CI scripts, and documentation updates.

## Quick Reference Commands
```
zig build test                          # run entire unit/integration suite
zig build test --test-filter agent      # run targeted subset
zig test tests/test_web_server.zig      # single module run
zig build -Doptimize=ReleaseFast perf   # performance suite (add step)
zig build e2e                           # end-to-end flows (add step)
zig build security-scan                 # security checks (add step)
llvm-cov show zig-out/coverage/...      # coverage inspection
```

---

Document owner: Quality Lead (assign in CODEOWNERS). Review quarterly or when architecture/modules change significantly.
