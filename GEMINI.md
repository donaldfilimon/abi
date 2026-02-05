# GEMINI.md

Quick reference for Gemini agents working in this repository.
Read `AGENTS.md` first for baseline rules. This file only highlights the essentials.

---

## Prerequisites

**Required:** Zig `0.16.0-dev.2471+e9eadee00` or later

```bash
zig version
export PATH="$HOME/.zvm/bin:$PATH"
zvm use master
```

---

## Before Making Changes

```bash
git status
git diff --stat
```

Ask about large or unclear diffs before proceeding.
Use package managers to add new dependencies.

---

## Quick Commands

```bash
zig build
zig build test --summary all
zig test src/path/to/file.zig --test-filter "pattern"
zig fmt .
zig build lint
zig build validate-flags          # verify all flag combos compile
```

Feature flags:

```bash
zig build -Denable-ai=true -Denable-gpu=false
zig build -Dgpu-backend=vulkan,cuda
```

---

## Must-Follow Rules

- Use `@import("abi")` for public API imports.
- Keep `mod.zig` and `stub.zig` signatures identical.
- Nested modules import via their parent `mod.zig`.
- Use Zig 0.16 APIs (no `std.fs.cwd()` or `std.time.Instant.now()`).

Zig 0.16 patterns at a glance:

```zig
const std = @import("std");
const abi = @import("abi");

// I/O backend (required)
var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
const io = io_backend.io();

// File system
const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024));

// Timing
var timer = std.time.Timer.start() catch return error.TimerFailed;

// Sleep (preferred)
abi.shared.time.sleepMs(10);

// ArrayListUnmanaged
var list = std.ArrayListUnmanaged(u8).empty;

// Format specifier
std.debug.print("State: {t}\n", .{state});

// Reserved keyword
const err = result.@"error";
```

---

## Post-Edit Checklist

```bash
zig fmt .
zig build test --summary all
zig build lint
```

---

## References

- `AGENTS.md` - Baseline rules
- `CLAUDE.md` - Full reference and examples
- `CONTRIBUTING.md` - Workflow
- `SECURITY.md` - Security practices
