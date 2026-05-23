# ABI Framework Documentation Hub

Welcome to the ABI framework documentation. This hub provides central access to architecture, onboarding, and development guides for the ABI multi-AI orchestration system.

## Architecture
- [ABI Refactor Design](spec/abi-refactor-design.md)

## Project Metadata
- [AGENTS.md](../AGENTS.md)
- [CLAUDE.md](../CLAUDE.md)
- [GEMINI.md](../GEMINI.md)

## Current Build And Runtime Guides
- [README.md](../README.md) for quick-start commands and current validation status.
- [walkthrough.md](../walkthrough.md) for CLI, MCP, GPU, TUI, and verification examples.

On macOS/Darwin, prefer `./build.sh ...` for project validation even when plain `zig build` works locally. Use `./build.sh full-check` for the complete local gate: `check`, integration tests, and benchmarks.

*For any issues or questions, please refer to the project's issue tracker or consult the primary maintainers.*
