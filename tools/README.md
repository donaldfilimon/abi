---
title: "Tools Directory"
tags: [tools, cli, utilities]
---
# Tools Directory
> **Codebase Status:** Synced with repository as of 2026-01-30.

<p align="center">
  <img src="https://img.shields.io/badge/Tools-Developer-blue?style=for-the-badge" alt="Developer Tools"/>
  <img src="https://img.shields.io/badge/CLI-Interactive-purple?style=for-the-badge" alt="CLI"/>
  <img src="https://img.shields.io/badge/DocGen-Automated-green?style=for-the-badge" alt="DocGen"/>
</p>

Index of developer tools and internal utilities.

## Layout

| Path | Purpose |
| --- | --- |
| `tools/cli/` | ABI CLI implementation (commands, TUI, utils, tests) |
| `tools/docgen/` | Documentation generator templates + driver |
| `tools/gendocs/` | Doc generation entry point (zig build gendocs) |
| `tools/benchmark-dashboard/` | Benchmark visualization UI + data |
| `tools/perf/` | Perf checks and microbench helpers |

## Quick Commands

```bash
zig build run -- --help
zig build gendocs
zig build benchmarks
```

---

See also: `benchmarks/README.md`, `docs/docs-index.md`.
