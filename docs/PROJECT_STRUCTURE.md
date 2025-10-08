# Project Structure Overview

The **ABI** repository follows a clear, modular layout that supports
incremental development, easy testing, and production‑ready builds.  
All source files live under `src/`; documentation lives under `docs/`; and
build‑related configuration is in the repository root.

```
abi/
├── .github/                # CI/CD workflows, issue templates
├── .zed/                   # Zed editor workspace (optional)
├── .zig-cache/             # Zig build cache (auto‑generated)
├── benchmarks/             # Performance benchmark programs
├── config/                 # Optional runtime configuration files
├── docker/                 # Dockerfiles for container builds
├── docs/                   # Markdown documentation
│   ├── PROJECT_STRUCTURE.md   ← This file
│   ├── MODULE_ORGANIZATION.md
│   ├── MODULE_REFERENCE.md
│   └── … (other guides)
├── monitoring/             # Observability helpers (metrics, tracing)
├── scripts/                # Helper scripts (release, perf‑summary, etc.)
├── src/                    # Core source tree
│   ├── agent/              # Agent‑related prototypes and middleware
│   ├── cli/                # Legacy CLI helpers (now superseded by `comprehensive_cli.zig`)
│   ├── connectors/         # External service connectors (e.g., Ollama)
│   ├── core/               # Low‑level building blocks (collections, error types)
│   ├── examples/           # Small runnable examples for each feature
│   ├── features/           # Public feature modules (ai, database, gpu, web, …)
│   │   ├── ai/               # Agent, model, training utilities
│   │   ├── database/         # WDBX vector DB, CLI & HTTP front‑ends
│   │   ├── gpu/              # GPU back‑ends, SIMD fallbacks, kernels
│   │   ├── web/              # HTTP server/client scaffolding
│   │   ├── monitoring/       # Structured logging, metrics exporters
│   │   └── connectors/       # Third‑party integration stubs
│   ├── framework/          # Runtime orchestration, feature toggles, plugin loader
│   ├── ml/                 # Machine‑learning utilities (future extensions)
│   ├── shared/             # Cross‑cutting utilities (logging, platform, simd)
│   ├── tests/              # Test suite mirroring the feature tree (`*_test.zig`)
│   ├── tools/              # Auxiliary tools (benchmark harness, docs generator, etc.)
│   ├── comprehensive_cli.zig # Modern, sub‑command based CLI entry point
│   ├── mod.zig             # Public façade – re‑exports all public APIs
│   └── simd.zig            # SIMD helpers (`VectorOps`) re‑exported from `shared`
├── tests/                  # High‑level integration tests (imports `src/...`)
├── .gitattributes
├── .gitignore
├── .zigversion             # Pinned Zig version (0.16.0)
├── AGENTS.md               # High‑level production playbook
├── CHANGELOG.md            # Release notes
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── MODERNIZATION_REPORT.md
├── MODERNIZATION_STATUS.md
├── README.md               # Project overview and quick‑start guide
├── SECURITY.md
├── build.zig               # Zig build script (produces `abi` executable)
└── build.zig.zon           # Dependency lockfile (Zig package manager)
```

## Key Design Principles

| Area                     | Guideline |
|--------------------------|-----------|
| **Explicit imports**     | All public symbols are re‑exported from `src/mod.zig`. Users can write `@import("abi").ai` or `@import("abi").database` without navigating the tree. |
| **Feature flags**        | Compile‑time flags (`-Denable-gpu`, `-Denable-web`, …) are collected in `framework/config.zig`. The CLI can query/toggle them at runtime. |
| **Allocator discipline** | Every allocation receives an explicit `std.mem.Allocator`. Ownership is clear and deallocation is always performed. |
| **Cross‑platform**       | The layout works on Windows, macOS, Linux, and (optionally) WASM/WebGPU (`-Denable-web`). |
| **Testing parity**       | Each feature directory contains a matching `*_test.zig` in `src/tests/`. `zig build test` runs the full suite. |
| **Documentation**        | `docs/` contains generated API reference, module organization, and a quick‑start guide. The `PROJECT_STRUCTURE.md` you are reading lives here. |
| **Backward compatibility** | Legacy symbols are re‑exported under `abi.wdbx` to keep older examples working. |

## How to Navigate

* **Library consumption** – Import `abi` from any Zig project and access sub‑modules directly via the façade:
  ```zig
  const abi = @import("abi");
  const Agent = abi.ai.agent.Agent;
  ```
* **CLI usage** – The single executable (`src/comprehensive_cli.zig`) provides sub‑commands:
  ```
  abi features list
  abi agent run --name Echo
  abi db insert --vec <file>
  abi gpu bench
  abi deps list
  ```
* **Running examples** – Each example under `src/examples/` can be compiled with:
  ```
  zig build examples/<example_name>
  ```
* **Extending the framework** – Add new feature modules under `src/features/`, expose them in `src/features/mod.zig`, and they become automatically available via the root façade.

---

*This file is part of the ABI project’s documentation suite and should be kept up‑to‑date as the repository evolves.*