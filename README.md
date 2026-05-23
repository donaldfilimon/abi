# ABI Framework

ABI is a **Zig 0.17.0-dev.329+21b7ceb5e** framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime.

## Quick Start
```bash
zig version             # Confirm Zig 0.17.0-dev.329+ compatible toolchain
./build.sh check        # Primary validation gate on macOS/Darwin
./build.sh full-check   # Check + integration tests + benchmarks
./build.sh cli          # Build zig-out/bin/abi
./build.sh mcp          # Build zig-out/bin/abi-mcp
```

Plain `zig build` and `zig build check` are expected to work with the pinned toolchain. On macOS/Darwin, keep using `./build.sh ...` for the documented project workflow.

## Current Status

- ABI is modernization-complete for Zig 0.17.0.
- All core feature modules and MCP transport are stable.
- The framework is fully validated; `./build.sh full-check` passes locally with no known test failures.
- Documentation: `CLAUDE.md`, `GEMINI.md`, and `AGENTS.md` are updated for the 0.17+ development lifecycle.
- Build: `./build.sh check` builds CLI/MCP, runs module tests, connector tests, linting, and validates mod/stub parity.
- Full validation: `./build.sh full-check` executes all integration tests and benchmarks.
- Plugins: `tools/generate_plugin_registry.zig` automatically maintains `src/plugin_registry.zig` based on `abi-plugin.json` manifests.

See [docs/index.md](docs/index.md) for architecture, onboarding, and development guides.
