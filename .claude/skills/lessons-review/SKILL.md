---
name: lessons-review
description: Review tasks/lessons.md for recurring pitfalls before starting work on Zig codebase changes
user-invocable: false
---

# Lessons Review Skill

At the start of every work session involving the ABI Zig codebase, automatically review `tasks/lessons.md` to prevent recurring mistakes.

## When to Activate

This skill activates automatically when beginning any task that involves:
- Editing Zig source files under `src/`, `build/`, `tools/`, or `benchmarks/`
- Modifying `build.zig` or build system configuration
- Running build or test commands
- Creating or modifying feature modules

## Procedure

### Step 1: Read the lessons file

Read `tasks/lessons.md` in the repository root. If the file does not exist or is empty, skip the remaining steps and proceed with the task.

### Step 2: Summarize recent relevant lessons

Identify the 3-5 most recent lessons that are relevant to the current task. Present a brief summary (one line per lesson) so the context is available without re-reading the full file each time.

### Step 3: Highlight critical categories

Pay special attention to and always surface lessons in these categories, regardless of recency:

**Zig 0.16 API changes:**
- `std.time.timestamp()` was removed; use `time.unixSeconds()`.
- `usingnamespace` was removed; pass parent context as parameters to submodules.
- `File.writeAll` was removed; use `file.writeStreamingAll(io, data)`.
- No `makeDirAbsolute*`; use `createDirPath(.cwd(), io, path)`.
- `addOptions`, `addTest`, `LazyPath` signatures changed from 0.15.
- ZON parsing requires arena-backed allocation with proper deinit.

**Darwin linker issues (macOS 25+/26+):**
- LLD has zero Mach-O support; never set `use_lld = true`.
- `zig build lint` fails on Darwin 25+ with undefined symbols; use `zig fmt --check` directly.
- Build runner links before `build.zig` runs, so build.zig workarounds cannot fix build runner link failures.
- Use `./tools/scripts/run_build.sh` to relink with Apple `/usr/bin/ld`.

**Build system patterns:**
- Files in the test manifest must compile standalone (no cross-directory relative imports above module root).
- Feature modules require both `mod.zig` and `stub.zig` with matching public signatures.
- Never run `zig fmt .` from repo root (vendored fixtures cause false positives); use the repo-safe format surface.
- Manifest-driven feature tests must share one module graph, not one module per entry.
- Version pin changes must update all sources atomically: `.zigversion`, `build.zig.zon`, `baseline.zig`, `README.md`, `.cel/config.sh`.

### Step 4: Emit preemptive warnings

Before the task begins, warn about any pitfall from the lessons file that directly applies to the planned work. Format warnings as:

```
Pitfall warning: [short description of the trap]
From lesson: [date and title from lessons.md]
Prevention: [the prevention rule from the lesson]
```

Only emit warnings for lessons that are concretely relevant to the current task. Do not repeat the entire lessons file.

## Output Format

Keep the review brief. A typical output is 5-10 lines: a short list of relevant lessons and any applicable warnings. Do not block or delay the user's task -- this review is informational context, not a gate.
