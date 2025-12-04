# ABI Framework - Agent Guidelines

## Project Overview
High-performance AI framework with vector database, GPU/TPU/NPU acceleration, and neural network training.

**Zig Version**: `0.16.0-dev` (use `std.debug.print` instead of `std.io.getStdOut`)

## Directory Structure
```
lib/           Core library (import as "abi")
├── core/      Types, errors, memory, I/O
├── features/  AI, GPU, database, web, monitoring
├── framework/ Runtime orchestration
└── shared/    Utils, platform, logging

tools/cli/     Command-line interface
tests/         Test suite
examples/      basic-usage.zig
```

## Build Commands
```bash
zig build              # Build library + CLI
zig build test         # Run tests
zig build run          # Run CLI
```

## Code Style
- **Indent**: 4 spaces
- **Types**: `PascalCase`
- **Functions/vars**: `snake_case`
- **Docs**: `//!` module, `///` public API

## Zig 0.16 Notes
- `std.io.getStdOut()` unavailable → use `std.debug.print`
- Use `vtable.*` not `rawAlloc/rawResize/rawFree` for custom allocators
- Async I/O coming in stable 0.16

## Key Modules
| Module | Purpose |
|--------|---------|
| `abi.gpu.accelerator` | Unified GPU/TPU/NPU/CPU |
| `abi.database` | Vector database |
| `abi.ai` | Neural networks |
| `abi.features` | Feature toggles |

## Feature Flags
`zig build -Denable-gpu=true -Denable-ai=true`
