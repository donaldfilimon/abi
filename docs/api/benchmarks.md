---
title: benchmarks API
purpose: Generated API reference for benchmarks
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2934+47d2e5de9
---

# benchmarks

> Benchmarks Module

Performance benchmarking and timing utilities. Provides a Context for
running benchmark suites, recording results, and exporting metrics.

**Source:** [`src/features/benchmarks/mod.zig`](../../src/features/benchmarks/mod.zig)

**Build flag:** `-Dfeat_benchmarks=true`

---

## API

### <a id="pub-fn-addbenchmark-self-benchmarksuite-allocator-std-mem-allocator-name-const-u8-func-benchmarkfn-void"></a>`pub fn addBenchmark( self: *BenchmarkSuite, allocator: std.mem.Allocator, name: []const u8, func: BenchmarkFn, ) !void`

<sup>**fn**</sup> | [source](../../src/features/benchmarks/mod.zig#L86)

Register a benchmark function.

### <a id="pub-fn-run-self-benchmarksuite-allocator-std-mem-allocator-void"></a>`pub fn run(self: *BenchmarkSuite, allocator: std.mem.Allocator) !void`

<sup>**fn**</sup> | [source](../../src/features/benchmarks/mod.zig#L105)

Execute all registered benchmarks.

For each benchmark:
1. Run `config.warmup_iterations` warmup iterations (discarded).
2. Run `config.sample_iterations` timed iterations.
3. Compute min / max / mean / median from the samples.
4. Store in `results`.

### <a id="pub-fn-formatreport-self-const-benchmarksuite-allocator-std-mem-allocator-u8"></a>`pub fn formatReport(self: *const BenchmarkSuite, allocator: std.mem.Allocator) ![]u8`

<sup>**fn**</sup> | [source](../../src/features/benchmarks/mod.zig#L173)

Format a human-readable report of all results.

### <a id="pub-fn-formatjson-self-const-benchmarksuite-allocator-std-mem-allocator-u8"></a>`pub fn formatJson(self: *const BenchmarkSuite, allocator: std.mem.Allocator) ![]u8`

<sup>**fn**</sup> | [source](../../src/features/benchmarks/mod.zig#L202)

Format results as a JSON array.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence while replacing the toolchain.
