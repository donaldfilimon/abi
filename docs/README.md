---
title: docs/
purpose: Documentation guide — what's maintained, what's generated, and how to add pages
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# Documentation Guide

This directory contains both hand-maintained documentation and auto-generated
API references. Understanding which is which prevents accidental overwrites.

## Maintained Docs (edit directly)

| File | Purpose |
|------|---------|
| `README.md` | This file — documentation guide |
| `STRUCTURE.md` | Full directory tree with annotations |
| `PATTERNS.md` | Zig 0.16 codebase patterns and conventions |
| `ABI_WDBX_ARCHITECTURE.md` | WDBX vector database design |
| `ZIG_MACOS_LINKER_RESEARCH.md` | Darwin linker bypass research |

## Generated Docs (do not hand-edit)

| Directory | Source | Regenerate |
|-----------|--------|------------|
| `api/` | `tools/gendocs/` + `build/module_catalog.zig` | `zig build gendocs` |
| `plans/` | `src/services/tasks/roadmap_catalog.zig` | `zig build gendocs` |
| `data/` | Structured data exports | `zig build gendocs` |

Generated files are overwritten each time `gendocs` runs. Do not hand-edit them;
instead, modify the source templates in `tools/gendocs/` or the catalog data.

## How to Regenerate

```bash
# Full regeneration
zig build gendocs

# With options (skip WASM, check untracked markdown)
zig build gendocs -- --no-wasm --untracked-md

# Check-only mode (verify determinism, no writes)
zig build gendocs -- --check --no-wasm --untracked-md

# On Darwin 25+ (compile-only — gendocs cannot run)
./tools/scripts/run_build.sh gendocs
```

## Markdown Allowlist Policy

The `.gitignore` ignores `*.md` globally, then explicitly allows specific paths.
This prevents accidental tracking of generated or scratch markdown.

**To add a new maintained doc:**

1. Create the file under `docs/`
2. Add `!/docs/<filename>.md` to `.gitignore` in the markdown allowlist section
3. Verify with `git status` that the file appears as untracked (not ignored)

Without step 2, git silently ignores the file.

## Directory Layout

```
docs/
├── README.md                 # This file
├── STRUCTURE.md              # Full project directory tree
├── PATTERNS.md               # Zig 0.16 patterns reference
├── ABI_WDBX_ARCHITECTURE.md  # WDBX architecture design
├── ZIG_MACOS_LINKER_RESEARCH.md  # Darwin linker notes
│
├── api/                      # Generated API reference pages
│   ├── index.md
│   ├── features.md
│   ├── services.md
│   └── ...
│
├── plans/                    # Generated roadmap docs
│   └── ...
│
└── data/                     # Structured exports (JSON, etc.)
```

## Related

- [CLAUDE.md](../CLAUDE.md) — Build commands and conventions
- [AGENTS.md](../AGENTS.md) — Contributor workflow contract
- [STRUCTURE.md](STRUCTURE.md) — Full project directory tree
- [PATTERNS.md](PATTERNS.md) — Zig 0.16 codebase patterns
