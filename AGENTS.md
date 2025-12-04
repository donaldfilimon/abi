# ABI Framework – Agent Guidelines

## 1. Project Overview
High‑performance AI framework with a vector database, GPU/TPU/NPU acceleration, and neural‑network training.

**Zig version**: ``0.16.0-dev`` – use ``std.debug.print`` instead of ``std.io.getStdOut``.

## 2. Directory Layout
```
lib/           Core library (import as ``abi``)
    core/       Types, errors, memory, I/O
    features/   AI, GPU, database, web, monitoring
    framework/  Runtime orchestration
    shared/     Utils, platform, logging

tools/cli/     Command‑line interface
tests/         Test suite
examples/      basic‑usage.zig
```

## 3. Build Commands
```bash
zig build              # Build library + CLI
zig build test         # Run tests
zig build run          # Run CLI
```

## 4. Code‑Style Rules
* **Indentation** – 4 spaces.
* **Types** – PascalCase.
* **Variables / functions** – snake_case.
* **Documentation** – `//!` for modules, `///` for public API.

## 5. Zig 0.16 Specifics
* ``std.debug.print`` for stdout.
* Prefer ``vtable.*`` for custom allocators over legacy ``rawAlloc``/``rawResize``/``rawFree``.
* Async I/O primitives stabilized.

## 6. Key Modules
| Module | Purpose |
|--------|---------|
| ``abi.gpu.accelerator`` | Unified GPU/TPU/NPU/CPU abstraction |
| ``abi.database`` | Vector database implementation |
| ``abi.ai`` | Neural‑network training utilities |
| ``abi.features`` | Feature‑flag management |

## 7. Feature‑Flags
Build with optional GPU / AI support:
```bash
zig build -Denable-gpu=true -Denable-ai=true
```
---
This file defines the core conventions for writing code in the ABI Framework. Follow the guidelines to keep the repo consistent and buildable.
