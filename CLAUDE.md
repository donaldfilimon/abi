# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick‑Start Overview
```bash
# 1️⃣ Bootstrap Zig toolchain and dependencies
tools/zigly --bootstrap

# 2️⃣ Build the library (or any target) – macOS uses ./build.sh, other platforms use zig build
./build.sh               # macOS – auto‑links with Apple linker
zig build                # Linux / older macOS

# 3️⃣ Run the full validation gate (lint + tests + mod/stub parity)
./build.sh check
```

*Run a single test:* `zig build test -- --test-filter "<test name>"`

See `docs/index.md` for more onboarding details.

## Common CLI Commands (quick reference)
| Command | What it does |
| ------- | ------------ |
| `abi` | Smart status (feature count, enabled/disabled tags) |
| `abi version` | Show version and build info |
| `abi doctor` | Build‑config report (feature flags + GPU back‑ends) |
| `abi features` | List all feature flags |
| `abi platform` | Detect OS/arch/GPU back‑ends |
| `abi connectors` | List LLM provider connectors and env‑var status |
| `abi search <sub>` | Full‑text search index management |
| `abi info` | High‑level framework architecture summary |
| `abi chat …` | Route a message through the multi‑profile AI pipeline |
| `abi db <subcommand>` | Vector‑DB operations (add, query, serve, etc.) |
| `abi serve` | Start the ACP HTTP server (default 127.0.0.1:8080) |
| `abi dashboard` | Interactive diagnostics shell (requires `-Dfeat-tui=true`) |
| `abi help` | Full help reference |

## High‑Level Architecture
* **Entry point** – `src/root.zig` re‑exports the public API as `@import("abi")`.
* **Public wiring** – `src/public/` gathers feature exports; each feature follows a **mod/stub** pattern:
  * `mod.zig` – real implementation
  * `stub.zig` – compile‑time no‑op when the feature is disabled
  * `types.zig` – shared types
  * Conditional export example (GPU):
    ```zig
    pub const gpu = if (build_options.feat_gpu)
        @import("../features/gpu/mod.zig")
    else
        @import("../features/gpu/stub.zig");
    ```
* **Feature catalog** – `src/features/` contains ~21 domains (core, ai, gpu, etc.).
* **AI Multi‑Profile Pipeline** – The Abbey‑Aviva‑Abi stack is wired in `src/features/ai/profile/router.zig`:
  `User → Abi analysis → AdaptiveModulator → Routing → Profile (Abbey/Aviva/Abi) → Constitution → WDBX storage → Response`.
* **GPU back‑ends** – Metal, CUDA, Vulkan, stdgpu (default) are functional; WebGPU, OpenGL, WebGL2, FPGA/TPU are partial or stubs (see table in the original file).

## Build & Test Commands (condensed)
```bash
# Full library build (macOS)          ./build.sh
# Full library build (Linux/old macOS) zig build

# Run all tests & summary               ./build.sh test --summary all
# Run a single test (pattern)           zig build test -- --test-filter "<pattern>"

# Lint / auto‑format                    zig build lint
# Auto‑format (fix)                     zig build fix

# Verify mod/stub parity                zig build check-parity

# Full validation (lint + test + parity) ./build.sh check   # macOS
# Full validation (Linux)            zig build check
```

## Feature Flags & GPU Backend Selection
All features are enabled by default except `feat-mobile` and `feat-tui`. Override with `-D` flags, e.g.:
```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false          # disable GPU and AI
zig build -Dgpu-backend=metal                     # select Metal backend
zig build -Dgpu-backend=cuda,vulkan               # enable CUDA and Vulkan
```

## MCP Server
```bash
zig build mcp                     # builds abi-mcp
zig-out/bin/abi-mcp               # JSON‑RPC 2.0 MCP server (stdio or SSE mode)
```
Restart any MCP clients after rebuilding.

## Development Workflow (short checklist)
1. Run `tasks/lessons.md` at session start.
2. Keep `tasks/todo.md` up‑to‑date for non‑trivial work.
3. After any public API change, run `zig build check-parity`.
4. Before marking a change done, run the appropriate validation gate (`./build.sh check` or `zig build full-check`).
5. Use conventional commits.
6. Never use plain `rm`; prefer safe alternatives.

## Additional References
* **Docs index:** `docs/index.md`
* **Known test failures:** see the "Testing" section in this file (inference engine connectors, auth integration).
* **.codex/** – internal Claude Code metadata; do not modify unless instructed.

---
*This file is intentionally focused on the most useful guidance for future Claude Code instances. For deeper details, consult the linked documentation files.*
