# CLAUDE.md

This file provides quick-reference guidance to Claude Code (claude.ai/code) when
working with code in this repository.

**Canonical Reference: [AGENTS.md](AGENTS.md)**
(Read this first for build/test commands and repo guidelines.)

## Architecture Overview

ABI is a modular Zig 0.16 framework. Every feature module in `src/features/`
MUST maintain a `mod.zig` and a `stub.zig` with matching public signatures.
The build system selects the implementation based on `feat_<name>` flags.

### Key Source Layout
- `src/abi.zig`: Public API entry point.
- `src/features/`: Feature modules (agents, vectors, GPU, network).
- `build/`: Modular build system (options, flags, modules, test discovery).
- `tools/cli/`: Main CLI implementation and registry.
- `docs/`: Documentation and architecture guides.

## Essential Development Commands

Refer to [AGENTS.md](AGENTS.md) for the full canonical list.

```bash
# Confidence gates
zig build full-check              # Mandatory before any commit
zig build verify-all              # Full release gate

# Core lifecycle
zig build test --summary all      # Primary service tests
zig build feature-tests           # Manifest-driven feature coverage
zig build refresh-cli-registry    # Update generated CLI snapshot
zig build fix                     # Auto-format codebase
```

## Workflow Rules

- **Plugin**: Use `zig-abi-plugin/` for smart build routing and feature scaffolding.
- **Atomic Edits**: Plan multi-file changes with a detailed research report first.
- **Validation**: Never mark a task as complete without passing `zig build full-check`.
- **Lessons**: Review `tasks/lessons.md` at session start and update it after fixing any recurring pitfalls.

## Known Issues & Bypass (macOS 26+)

Due to upstream Zig linker issues on Darwin Tahoe, use the `addObject` bypass
mechanism provided by the build system. Host-executing scripts (doctors, registry
generators) use `addHostScriptStep` to maintain compile-only compatibility.

```bash
# Use the CEL toolchain (patched Zig) if full linking is required locally:
./.zig-bootstrap/build.sh && eval "$(./tools/scripts/use_zig_bootstrap.sh)"
```

*See [docs/FAQ-agents.md](docs/FAQ-agents.md) for detailed style rules and command documentation.*
