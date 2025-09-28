# ABI Repository – Comprehensive Refactor Playbook (Zig 0.16.0-dev.393+dd4be26f5)

This **comprehensive refactor playbook** is a full migration and modernization plan for the ABI repository targeting Zig **0.16.0-dev.393+dd4be26f5**. It provides a structured roadmap, moving far beyond a checklist into a detailed guide: covering build scripts, I/O redesign, CLI ergonomics, parser internals, CI/CD automation, documentation pipelines, testing philosophy, contributor guidelines, performance considerations, error handling, and long-term maintainability. The expanded edition emphasizes clarity and consistency, offering practical examples, rationale, and safeguards to ensure the upgrade is both successful and sustainable.

---

## Goals & Constraints

* **Primary Goal:** Upgrade ABI’s entire codebase to Zig 0.16-dev, adopting modern build APIs (`.root_module`, `b.createModule`), the updated I/O layer (`std.Io.Writer`/`Reader`), safer error handling, and explicit formatting semantics.
* **Secondary Goal:** Enhance developer experience through deterministic CI, automated documentation, consistent formatting enforcement, multi-platform coverage (Linux/macOS/Windows), and improved contributor resources.
* **Constraints:** Public API stability is prioritized. When breaking changes are unavoidable, provide adapters and migration notes to maintain external compatibility.
* **Expected Outcomes:** Green builds, consistent formatting, reproducible CI runs, documentation and log artifacts, and automatically generated HTML docs under `zig-out/docs` or via GitHub Pages.

---

## Migration Strategy (Phased Rollout)

1. **Phase 1 – Build System Upgrade**
   Transition to new APIs and modular design. Introduce `run`, `test`, `docs`, `fmt`, and optionally `bench`. Confirm reproducibility.

2. **Phase 2 – I/O Boundary Rework**
   Replace global stdout calls with injected writers. Use adapters and explicit formatting. Enforce separation between human-readable and machine output.

3. **Phase 3 – CLI & Parser Modernization**
   Update CLI entrypoints to support `zig build run`. Modernize parser internals with `std.ArrayList`, streaming reads, and diagnostics structures.

4. **Phase 4 – CI/CD & Documentation**
   Implement a multi-OS CI matrix, automated docs generation, format checks, optional GitHub Pages deployment, and artifact uploads.

5. **Phase 5 – Quality, Testing & Polish**
   Add integration and snapshot tests, refine error handling, update documentation (README, CONTRIBUTING.md), enforce formatting, and strengthen style consistency.

Each phase should be delivered in a separate PR for easier rollbacks. CI must validate each stage incrementally, and fallback paths should be documented.

---

## 1) Build System Modernization

### Targets

* Replace `root_source_file` with `.root_module = b.createModule(...)`.
* Standardize use of `b.standardTargetOptions(.{})` and `b.standardOptimizeOption(.{})`.
* Attach library and CLI executables as separate modules, wiring imports cleanly.
* Add steps for `run`, `test`, `docs`, `fmt`, and optionally `bench`.

### Example

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Core library
    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .name = "abi",
        .root_module = lib_mod,
        .linkage = .static,
    });
    b.installArtifact(lib);

    // CLI executable
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("abi", lib_mod);
    b.installArtifact(exe);

    // Run step
    const run_exe = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the ABI CLI");
    run_step.dependOn(&run_exe.step);

    // Tests
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/all_tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(unit_tests).step);

    // Documentation
    const install_docs = b.addInstallDirectory(.{
        .source_dir = lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    const docs_step = b.step("docs", "Generate documentation");
    docs_step.dependOn(&install_docs.step);

    // Formatting
    const fmt = b.addFmt(.{ .paths = &.{"src", "tests"} });
    const fmt_step = b.step("fmt", "Format source files");
    fmt_step.dependOn(&fmt.step);
}
```

### Considerations

* Add feature toggles (`-Denable-ansi`, `-Dstrict-io`, `-Dexperimental`) with clear defaults.
* Strip symbols for smaller release binaries with `exe.strip = true;`.
* Avoid absolute paths; use `b.path` for reproducibility.
* Keep artifact naming consistent.

---

## 2) I/O Migration & Boundaries

### Principles

* Inject writers into libraries instead of calling stdout directly.
* Let the CLI manage global stdout/stderr.
* Use adapters (`adaptToNewApi()`) to bridge legacy writers.
* Enforce explicit formatting specifiers (`{s}`, `{d}`, `{x}`, `{any}`).

### Example

```zig
// Before
std.debug.print("Processed {d} items\n", .{count});

// After
var adapter = std.io.getStdOut().writer().adaptToNewApi(&.{});
const w = adapter.new_interface;
try w.print("Processed {d} items\n", .{count});
```

**File Writing**

```zig
var file = try std.fs.cwd().createFile("out.txt", .{ .truncate = true });
defer file.close();
var wr = file.writer().adaptToNewApi(&.{}).new_interface;
try wr.print("{s}\n", .{"hello"});
```

**Safe Reads**

```zig
const max = 1 << 20;
var f = try std.fs.cwd().openFile(path, .{});
defer f.close();
const buf = try f.reader().readAllAlloc(allocator, max);
```

### Guidance

* Use `std.log` for filtering (`.info`, `.warn`, `.err`).
* Prefer `stderr` for human-readable messages.
* Reserve `stdout` for structured or machine output (e.g., JSON).
* Consider helper functions for structured logging.

---

## 3) CLI Tools & User Experience

### Example Skeleton

```zig
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var out_adapter = std.io.getStdOut().writer().adaptToNewApi(&.{});
    const w = out_adapter.new_interface;

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    try runCli(w, args);
}
```

### Guidelines

* Always support `--help` with concise usage information.
* Provide subcommands: `parse`, `version`, `lint`, `check`.
* Return typed errors: `error.UnknownCommand`, `error.BadArgument`, `error.Io`.
* Use `stderr` for human-readable output and `stdout` for machine output.
* Add smoke tests for common command flows.

---

## 4) Parser Subsystem

### Improvements

* Replace manual pointer arithmetic with slice-based APIs and `std.mem` helpers.
* Use `std.ArrayList` for dynamic buffers.
* Stream reads for large files to avoid unbounded memory usage.
* Add a `Diagnostics` struct (`line`, `col`, `message`, `kind`) for recoverable errors.

### Example

```zig
pub fn parseFile(alloc: std.mem.Allocator, path: []const u8) !void {
    var f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    var r = f.reader();

    var line = std.ArrayList(u8).init(alloc);
    defer line.deinit();

    while (true) {
        line.clearRetainingCapacity();
        const rc = try r.readUntilDelimiterArrayList(&line, '\n', 1 << 20);
        if (rc == .DelimiterNotFound and line.items.len == 0) break;
        // process line.items...
    }
}
```

### Notes

* Use explicit error sets (e.g., `error.ParseError`).
* Add golden tests for known cases.
* Benchmark parser with large inputs.

---

## 5) Continuous Integration

### Workflow Example

```y
```
