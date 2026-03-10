# Documentation Layout

The ABI documentation tree is organized into the following subdirectories and key files.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| [`api/`](./api) | Generated API reference Markdown (`index.md`, `coverage.md`, per-module pages) |
| [`guides/`](./guides) | Developer guides and IDE configuration (Cursor rules, etc.) |
| [`plans/`](./plans) | Project plans, roadmap history, and execution logs |
| [`data/`](./data) | Docs-site data artifacts and generated metadata |

## Key Files

| File | Description |
|------|-------------|
| [`ABI_WDBX_ARCHITECTURE.md`](./ABI_WDBX_ARCHITECTURE.md) | WDBX vector database architecture and on-disk format |
| [`ZIG_MACOS_LINKER_RESEARCH.md`](./ZIG_MACOS_LINKER_RESEARCH.md) | Darwin linker workarounds and LLD/Mach-O research |
| [`FAQ-agents.md`](./FAQ-agents.md) | Style rules and expanded command reference for agents |
| [`index.html`](./index.html) | Local docs site entrypoint |

## Generation Workflow

```bash
zig build gendocs       # Generate API reference docs from source
zig build check-docs    # Validate deterministic output and policy compliance
```

`zig build gendocs` is the canonical generator entrypoint. It produces Markdown pages under `api/`. `zig build check-docs` verifies that output is deterministic and conforms to documentation policy.
