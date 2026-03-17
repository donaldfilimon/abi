---
name: lessons-review
description: This skill should be used when starting any work session involving the ABI Zig codebase, before editing source files, modifying build configuration, or running build/test commands. Automatically reviews tasks/lessons.md for recurring pitfalls to prevent repeat mistakes.
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

Identify the 3-5 most recent lessons relevant to the current task. Present a brief summary (one line per lesson) so the context is available without re-reading the full file each time.

### Step 3: Highlight critical categories

Pay special attention to and always surface lessons in these categories, regardless of recency:

- **Zig 0.16 API changes** — any lessons about deprecated or renamed APIs
- **Darwin linker issues (macOS 25+/26+)** — linker workarounds and build runner limitations
- **Build system patterns** — test manifest rules, module ownership, format check surfaces
- **mod/stub sync** — parity requirements, stub boilerplate, CLI sub-module re-exports
- **Import rules** — relative vs `@import("abi")`, cross-feature gating, explicit `.zig` extensions
- **foundation namespace** — not a separate module, relative path access within `src/`

Note: The actual lessons content lives in `tasks/lessons.md` and evolves over time. Always read the file rather than relying on cached knowledge.

### Step 4: Emit preemptive warnings

Before the task begins, warn about any pitfall from the lessons file that directly applies to the planned work. Format warnings as:

```
Pitfall warning: [short description of the trap]
From lesson: [category and topic from lessons.md]
Prevention: [the prevention rule from the lesson]
```

Only emit warnings for lessons concretely relevant to the current task. Do not repeat the entire lessons file.

## Output Format

Keep the review brief. A typical output is 5-10 lines: a short list of relevant lessons and any applicable warnings. Do not block or delay the task — this review is informational context, not a gate.
