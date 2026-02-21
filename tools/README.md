# Tools Directory

Developer tools and internal utilities for the ABI framework.

## Layout

| Path | Purpose | Build Step |
| --- | --- | --- |
| `cli/` | ABI CLI (30 commands + 8 aliases, TUI, utils) | `zig build run -- --help` |
| `gendocs/` | API documentation generator | `zig build gendocs` / `abi gendocs` |
| `perf/` | Performance KPI verification tool | `zig build check-perf` |

## CLI Command Subdirectories

Large commands are organized into subdirectories with one file per subcommand:

| Directory | Files | Subcommands |
| --- | --- | --- |
| `cli/commands/train/` | 10 | run, new, llm, vision, clip, auto, resume, monitor, info, generate-data |
| `cli/commands/llm/` | 9 | info, generate, chat, bench, list, list-local, demo, download, serve |
| `cli/commands/bench/` | 5 | suites, micro, output, training-comparison |
| `cli/commands/ralph/` | 10 | init, run, super, multi, status, gate, improve, skills, config |

Single-file commands remain directly in `cli/commands/`.

## Quick Commands

```bash
zig build run -- --help          # CLI help
zig build gendocs                # Generate API docs to docs/api/
zig build check-perf             # Build perf verification tool
```
