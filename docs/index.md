# ABI Framework Documentation Hub

Welcome to the ABI framework documentation. This hub provides central access to architecture, onboarding, and development guides for the ABI multi-AI orchestration system.

## Architecture
- [ABI Refactor Design](spec/abi-refactor-design.md)
- [ABI Master Specification](superpowers/specs/ABI-MASTER-SPEC.md)
- [Public API Contract](contracts/public-api.md)

## Project Metadata
- [AGENTS.md](../AGENTS.md)
- [CLAUDE.md](../CLAUDE.md)
- [GEMINI.md](../GEMINI.md)

## Current Build And Runtime Guides
- [README.md](../README.md) for quick-start commands and current validation status.
- [walkthrough.md](../walkthrough.md) for CLI, MCP, GPU, TUI, and verification examples.

On macOS/Darwin, prefer `./build.sh ...` for project validation even when plain `zig build` works locally. Use `./build.sh full-check` for the complete local gate: `check`, integration tests, and benchmarks.

Key executable contracts live in `tests/contracts/` and cover CLI/MCP surfaces plus generated plugin registry metadata.

*For any issues or questions, please refer to the project's issue tracker or consult the primary maintainers.*
