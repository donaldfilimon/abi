---
title: Ralph Orchestrator Guide
description: Autonomous AI code agent with iterative improvement loops, skill memory, and multi-agent coordination
section: Guides
order: 91
permalink: /ralph-guide/
---

# Ralph Orchestrator Guide
## Summary
Autonomous AI code agent with iterative improvement loops, skill memory, and multi-agent coordination

## What is Ralph?

Ralph is an autonomous AI code agent built into the ABI CLI. It runs iterative
improvement loops: read the codebase, generate changes, verify with quality gates,
commit, and repeat. Ralph uses the Abbey reasoning engine under the hood and
persists learned **skills** across runs so each iteration gets smarter.

## Architecture

```
ralph init  ->  ralph.yml + .ralph/ + PROMPT.md
                       |
                ralph run / ralph super
                       |
         +-------------+-------------+
         |             |             |
    Read PROMPT    Generate plan   Execute changes
         |             |             |
         +-------> Verify (gate) ---+
                       |
                  Commit / Skip
                       |
                  Extract skill (optional)
```

## CLI Commands

### `abi ralph init`

Create the workspace scaffold:

```bash
abi ralph init
```

Creates:
- `ralph.yml` — configuration (provider, model, iteration limits, gate commands)
- `.ralph/` — working directory for state, skills, and logs
- `PROMPT.md` — task description consumed by `ralph run`

### `abi ralph run`

Execute the iterative improvement loop:

```bash
# Run task from PROMPT.md
abi ralph run

# Inline task
abi ralph run --task "Fix all type errors in src/features/gpu/"

# Run with automatic skill extraction
abi ralph run --auto-skill
```

### `abi ralph super`

Power one-shot: `init` (if needed) + `run` + optional gate:

```bash
abi ralph super --task "Refactor the cache module" --gate --auto-skill
```

### `abi ralph multi`

Zig-native multithreaded multi-agent execution using `ThreadPool` and a lock-free
`RalphBus` for inter-agent communication:

```bash
abi ralph multi -t "Fix GPU tests" -t "Update docs" -t "Add benchmarks"
```

Each task runs in its own thread with an isolated Ralph context. Agents communicate
results via the bus, and the orchestrator merges non-conflicting changes.

### `abi ralph gate`

Run the native quality gate (replaces `check_ralph_gate.sh`):

```bash
abi ralph gate
```

The gate runs the configured verification commands (default: `zig build full-check`)
and scores the result. A passing gate allows the loop to commit.

### `abi ralph improve`

Autonomous self-improvement loop that analyzes the codebase and applies
incremental fixes with per-iteration `verify-all` and commits:

```bash
abi ralph improve
abi ralph improve --apply   # Actually apply changes (default is analysis-only)
```

### `abi ralph skills`

Manage persisted skills stored in `.ralph/skills.jsonl`:

```bash
abi ralph skills                      # List all skills
abi ralph skills --add "lesson text"  # Add a skill manually
abi ralph skills --clear              # Remove all skills
```

## Configuration (`ralph.yml`)

```yaml
provider: ollama           # LLM backend (ollama, anthropic, openai, mlx, etc.)
model: llama3              # Model name or path
max_iterations: 5          # Maximum loop iterations per run
gate_commands:
  - zig build full-check   # Commands that must pass for gate success
auto_skill: true           # Extract skills after successful runs
prompt_file: PROMPT.md     # Task description file
```

### Provider Backend Selection

Ralph delegates LLM calls to the same provider router used by `abi llm`. Set
`provider` in `ralph.yml` or pass `--backend <name>` to override. The full
fallback chain is available (`--fallback mlx,ollama,anthropic`).

## Skills System

Skills are short lessons extracted from successful runs. They are stored in
`.ralph/skills.jsonl` (one JSON object per line) and injected into the system
prompt of subsequent runs. This gives Ralph cumulative memory across sessions.

Example skill entry:

```json
{"timestamp":1740230400,"skill":"When fixing Zig 0.16 compile errors, check std.Io vs std.io first"}
```

## Generated Reference
## Overview

This guide is generated from repository metadata for **Guides** coverage and stays deterministic across runs.

## Build Snapshot

- Zig pin: `0.16.0-dev.2623+27eec9bd6`
- Main tests: `1289` pass / `7` skip / `1296` total
- Feature tests: `2332` pass / `2337` total

## Feature Coverage

- **agents** — AI agent runtime
  - Build flag: `enable_ai`
  - Source: `src/features/ai/facades/core.zig`
  - Parent: `ai`
- **reasoning** — AI reasoning (Abbey, eval, RAG)
  - Build flag: `enable_reasoning`
  - Source: `src/features/ai/facades/reasoning.zig`
  - Parent: `ai`

## Module Coverage

- `src/services/connectors/mod.zig` ([api](../api/connectors.html))

## Command Entry Points

- `abi agent` — Run AI agent (interactive or one-shot)
- `abi embed` — Generate embeddings from text (openai, mistral, cohere, ollama)
- `abi llm` — LLM inference (run, session, serve, providers, plugins, discover)
- `abi model` — Model management (list, download, remove, search)
- `abi ralph` — Ralph orchestrator (init, run, super, multi, status, gate, improve, skills)
- `abi train` — Training pipeline (run, llm, vision, auto, self, resume, info)

## Validation Commands

- `zig build typecheck`
- `zig build check-docs`
- `zig build run -- gendocs --check`

## Navigation

- API Reference: [../api/](../api/)
- API App: [../api-app/](../api-app/)
- Plans Index: [../plans/index.md](../plans/index.md)
- Source Root: [GitHub src tree](https://github.com/donaldfilimon/abi/tree/master/src)

## Maintenance Notes
- This page is generated by `zig build gendocs`.
- Edit template source in `tools/gendocs/templates/docs/` for structural changes.
- Edit generator logic in `tools/gendocs/` for data model or rendering changes.


---

*Generated automatically by `zig build gendocs`*


## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
