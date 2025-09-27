# Repository Guidelines

This repo is a Zig project with a Makefile wrapper. Source lives in `src/`, examples in `examples/`, benchmarks in `benchmarks/`, tests in `tests/`, and build artifacts in `zig-out/`. Build configuration is defined in `build.zig` and dependencies in `build.zig.zon`.

## Project Structure & Module Organization

- `src/` — library and app code; keep modules small and cohesive (e.g., `src/net/`, `src/util/`).
- `tests/` — unit and integration tests mirror `src/` layout (e.g., `tests/net/…`).
- `examples/` — minimal runnable samples demonstrating public APIs.
- `benchmarks/` — micro/throughput benchmarks; isolate external effects.
- `tools/` — helper scripts; keep cross‑platform where possible.

## Build, Test, and Development Commands

- `zig version` — confirm toolchain (also see `.zigversion`).
- `zig build` — default build; produces artifacts under `zig-out/`.
- `zig build test` — compile and run all tests.
- `zig build run` — run the default executable (if defined in `build.zig`).
- `zig build -Doptimize=ReleaseFast` — optimized build for benchmarks.

## Coding Style & Naming Conventions

- Indentation: 4 spaces; no tabs. Line length ~100 chars.
- Zig style: prefer explicit types at public boundaries; use `const` where possible.
- Naming: `PascalCase` for types, `camelCase` for functions/vars, `SCREAMING_SNAKE_CASE` for compile‑time constants.
- Errors: return typed error sets; avoid `catch |e|` that masks context.
- Formatting: run `zig fmt .` before committing.

## Testing Guidelines

- Framework: Zig’s built‑in test runner (`test "name" { … }`).
- Layout: place tests beside code with `test` blocks or under `tests/` mirroring `src/`.
- Naming: describe behavior, e.g., `test "parser handles empty input" {}`.
- Coverage: add tests for new features and bug fixes; include edge cases and error paths.
- Run: `zig build test` (CI expects zero failures).

## Commit & Pull Request Guidelines

- Commits: present‑tense, scope-first messages, e.g., `net: fix timeout handling`.
- Keep changes focused; include rationale in the body when non‑obvious.
- PRs: include summary, linked issues, usage notes, and before/after benchmarks when performance‑related. Add repro steps for fixes and update `examples/` when APIs change.

## Security & Configuration Tips

- Avoid unchecked `@ptrCast`/`@intCast`; validate sizes and alignment.
- Prefer bounded operations; assert invariants in debug builds.
- Respect platform differences; gate OS‑specific code via `std.builtin.os`.

## Extra Nodes

# Agents – Unified Reference

> *A concise, copy‑and‑paste reference for the **Agents** subsystem.  
>  This file is deliberately kept short, tab‑aligned, and style‑consistent so it can be dropped into any documentation or example project.*

---

## 1.  What is an **Agent**?

| Responsibility | What it does |
|----------------|--------------|
| **Data ingestion** | Loads, shreds, and pre‑processes batches. |
| **Model execution** | Runs forward / backward passes with autograd. |
| **Optimization** | Applies a user‑selected optimizer (SGD, Adam, …). |
| **Metrics** | Reports loss, accuracy, throughput, and wall‑time. |

> Agents are *lock‑free* and *arena‑first*: all hot‑path memory comes from a per‑frame `ArenaAllocator`.  All API functions are `!T` (typed error) and never silently swallow errors.

---

## 2.  Public API

The entire public surface is exposed through `src/agent/mod.zig`.  Below is the *complete* list of exported symbols.

| Symbol | Type | Short description |
|--------|------|-------------------|
| `Agent` | `struct` | Orchestrates training. |
| `AgentConfig` | `struct` | Configuration (batch size, learning rate, etc.). |
| `Metrics` | `struct` | Immutable snapshot of training stats. |
| `Agent.Error` | `error set` | Public error set. |
| `Agent.init(allocator, config) !Agent` | `fn` | Constructor. |
| `Agent.trainStep(batch: []Tensor) !Metrics` | `fn` | Trains one batch. |
| `Agent.run() !void` | `fn` | Blocking loop until `shutdown`. |
| `Agent.shutdown() void` | `fn` | Gracefully stops the agent. |

> **Naming** – Types use `PascalCase`, functions use `camelCase`, constants use `SCREAMING_SNAKE_CASE`.

---

### 2.1  Types

```zig
pub const AgentConfig = struct {
    batchSize: usize,        // # of samples per batch
    learningRate: f32,       // optimizer step size
    accumSteps: usize = 1,   // gradient accumulation
    gpu: ?usize = null,      // `null` → CPU; otherwise GPU device ID
};

pub const Metrics = struct {
    loss: f32,
    accuracy: f32,
    wallTimeNs: u64,
    throughput: f32, // samples/sec
};
```

### 2.2  Errors

```zig
pub const Error = error{
    EmptyBatch,
    InvalidConfig,
    GpuError,
    OutOfMemory,
    Internal,
};
```

### 2.3  Key Functions

```zig
pub fn init(allocator: std.mem.Allocator, cfg: AgentConfig) !Agent { … }

pub fn trainStep(self: *Agent, batch: []Tensor) !Metrics { … }

pub fn run(self: *Agent) !void { … }

pub fn shutdown(self: *Agent) void { … }
```

---

## 3.  Usage Example

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    const gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const cfg = abi.agent.AgentConfig{
        .batchSize = 256,
        .learningRate = 0.001,
        .gpu = 0,     // use GPU 0, `null` for CPU
    };

    var agent = try abi.agent.Agent.init(gpa.allocator(), cfg);
    defer agent.shutdown();

    // Create a dummy batch (replace with real data in your code)
    var batch = try abi.tensor.Tensor.init(gpa.allocator(), 256);
    defer batch.deinit();

    const metrics = try agent.trainStep(&batch);
    std.debug.print(
        "step: loss={d:.4} | acc={d:.2}%\n",
        .{ metrics.loss, metrics.accuracy * 100.0 },
    );

    // To run continuously:
    // try agent.run();    // blocks until `shutdown()` is called
}
```

---

## 4.  Testing

| Test | Purpose |
|------|---------|
| `agent/agent_test.zig` | Validates config handling, error propagation, single‑step training. |
| `agent/run_test.zig` | Integration test for `run()` with a dummy loader. |

> Run all tests: `zig build test`.  CI expects zero failures.

---

## 5.  Benchmarks

```zig
// benchmarks/agent/agent_bench.zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    const gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const cfg = abi.agent.AgentConfig{
        .batchSize = 1024,
        .learningRate = 0.0001,
        .gpu = 0,
    };

    var agent = try abi.agent.Agent.init(gpa.allocator(), cfg);
    defer agent.shutdown();

    const batch = try abi.tensor.Tensor.init(gpa.allocator(), 1024);
    defer batch.deinit();

    const timer = try std.time.Timer.start();
    const steps = 10_000;
    var i: usize = 0;
    while (i < steps) : (i += 1) {
        _ = try agent.trainStep(&batch);
    }
    const elapsed = timer.read();
    const nsPerStep = elapsed / steps;
    std.debug.print("ns per step: {d}\n", .{nsPerStep});
}
```

> Run with: `zig build bench`.  
> Baseline values live in `benchmarks/.baseline.json`; any > 5 % slowdown will fail CI.

---

## 6.  Cross‑Platform Notes

| Feature | Windows | Linux | macOS |
|---------|---------|-------|-------|
| GPU (Vulkan) | ✅ | ✅ | ✅ |
| CUDA | ✅ (requires NVIDIA driver) | ✅ | ❌ |
| Metal | ❌ | ❌ | ✅ |
| Threading | `std.Thread` | `std.Thread` | `std.Thread` |
| Async I/O | `std.fs.AsyncFile` | `std.fs.AsyncFile` | `std.fs.AsyncFile` |

> For portability, set `cfg.gpu = null` to use the CPU backend.

---

## 7.  Security & Safety

| Topic | Guideline |
|-------|-----------|
| `@ptrCast` | Only in internal modules; guarded by alignment checks in debug builds. |
| Size/Alignment | All slices validate length and element size before use. |
| Bounded Ops | `std.mem` helpers enforce bounds; no unchecked slices. |
| Error Propagation | No silent failures; all API functions return a typed error set. |

---

## 8.  Contributing

1. **Branch** – `git checkout -b feat/<short-name>`.  
2. **Edit** – update `agents.md`, add tests in `tests/agent/`.  
3. **Bench** – if you change performance, run `zig build bench` and commit the new baseline.  
4. **Commit** – style: `agent: <short description>`.  
5. **PR** – include a summary, related issues, and a “before/after” comparison for performance changes.

---

## 9.  Quick Commands

| Command | What it does |
|---------|--------------|
| `zig build` | Build library & demo. |
| `zig build test` | Run tests. |
| `zig build bench` | Run benchmarks. |
| `zig build run` | Run demo executable. |
| `zig fmt .` | Format the repo. |

--- 

*Happy training!*
