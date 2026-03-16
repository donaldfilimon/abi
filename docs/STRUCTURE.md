# ABI Directory Structure

Comprehensive directory tree for the ABI framework, annotated with the purpose
of each directory and key file patterns.

## Top-level Layout

```
abi/
├── build.zig                 # Build root — defines all steps, targets, gates
├── build.zig.zon             # Package manifest (Zig 0.16 format)
├── .zigversion               # Pinned Zig version (0.16.0-dev.2905+...)
│
├── src/                      # All framework source — single "abi" module
│   ├── root.zig              # Public package entrypoint (@import("abi"))
│   ├── abi.zig               # Internal composition layer
│   ├── core/                 # Always-on framework internals
│   ├── features/             # Comptime-gated feature modules (19 modules)
│   ├── services/             # Runtime services shared across features
│   └── inference/            # Sampler, scheduler, KV cache
│
├── build/                    # Modular build system (Zig source files)
├── tools/                    # CLI, docs generator, validation scripts
├── tests/                    # Integration and focused test roots
├── examples/                 # 36 standalone example programs
├── benchmarks/               # Performance benchmark suites
├── bindings/                 # C and WASM language bindings
├── docs/                     # Maintained + generated documentation
│
├── CLAUDE.md                 # AI agent instructions and conventions
├── AGENTS.md                 # Contributor workflow contract
├── GEMINI.md                 # Gemini CLI instructions
├── README.md                 # Project overview and quick start
└── SECURITY.md               # Vulnerability reporting
```

## Source Tree (`src/`)

```
src/
├── root.zig                  # Public API surface — what @import("abi") exposes
├── abi.zig                   # Internal wiring layer — composes features + services
│
├── core/                     # Always-on internals
│   ├── config/               # Configuration loading and validation
│   ├── feature_catalog.zig   # Source of truth for feature metadata
│   ├── registry/             # Service registry and lifecycle
│   └── stub_context.zig      # StubFeature / StubFeatureNoConfig helpers
│
├── features/                 # Comptime-gated modules (one dir per feature)
│   ├── ai/                   # Agents, profiles, training, reasoning, LLM
│   ├── gpu/                  # Unified GPU compute (CUDA/Vulkan/Metal/WebGPU)
│   ├── database/             # WDBX semantic store, HNSW, vector search
│   ├── network/              # TCP/UDP, HTTP, WebSocket
│   ├── web/                  # HTTP server, routing, templates
│   ├── auth/                 # Authentication and RBAC
│   ├── cache/                # In-memory and tiered caching
│   ├── cloud/                # Cloud provider integrations
│   ├── compute/              # General compute and tensor ops
│   ├── storage/              # File and object storage
│   ├── messaging/            # Message queues and pub/sub
│   ├── observability/        # Metrics, tracing, logging
│   ├── analytics/            # Event tracking and pipelines
│   ├── search/               # Full-text and semantic search
│   ├── mobile/               # Mobile platform bindings
│   ├── desktop/              # Desktop platform support
│   ├── gateway/              # API gateway and rate limiting
│   ├── documents/            # Document processing
│   └── benchmarks/           # In-framework benchmark support
│
├── services/                 # Shared runtime services
│   ├── shared/               # Common utilities, signal handling, security
│   ├── connectors/           # External service connectors
│   ├── lsp/                  # Language Server Protocol
│   ├── mcp/                  # Model Context Protocol
│   └── tests/                # Service test root
│
└── inference/                # ML inference runtime
    ├── sampler.zig
    ├── scheduler.zig
    └── kv_cache.zig
```

### The mod/stub/types.zig Pattern

Every feature module under `src/features/<name>/` follows a strict contract:

```
src/features/<name>/
├── mod.zig          # Real implementation (used when feature is enabled)
├── stub.zig         # API-compatible no-ops (used when feature is disabled)
└── types.zig        # Shared types imported by both mod and stub
```

- **`mod.zig`** — full implementation, imported when `-Dfeat-<name>=true` (default)
- **`stub.zig`** — returns `error.FeatureDisabled` or zero-values; must match `mod.zig`
  public signatures exactly
- **`types.zig`** — shared type definitions (enums, structs, error sets) that both
  `mod.zig` and `stub.zig` import, avoiding duplication

The build system selects between `mod.zig` and `stub.zig` at comptime via
`build_options` flags. See `src/core/stub_context.zig` for `StubFeature` and
`StubFeatureNoConfig` helpers that reduce stub boilerplate.

## Build System (`build/`)

```
build/
├── options.zig               # 25 feature flag definitions
├── flags.zig                 # 42 flag combination validations
├── modules.zig               # Module creation and import wiring
├── module_catalog.zig        # Module registry for gendocs
├── targets.zig               # Example/target tables, cross-compilation matrix
├── test_discovery.zig        # Feature test manifest
├── link.zig                  # Platform link flags (Metal, libc, etc.)
├── gpu.zig                   # GPU backend option parsing
├── gpu_policy.zig            # GPU policy validation
├── mobile.zig                # Mobile target support
├── wasm.zig                  # WebAssembly build targets
├── cli_tests.zig             # CLI smoke test definitions
├── cli_tui_tests_root.zig    # TUI test root
└── cli_tui_test_runner.zig   # Custom test runner for TUI tests
```

## Tools (`tools/`)

```
tools/
├── cli/                      # ABI CLI implementation
│   ├── main.zig              # CLI entrypoint
│   ├── mod.zig               # CLI module root
│   ├── commands/             # Command implementations
│   └── registry/             # CLI command registry snapshots
│
├── gendocs/                  # Documentation generator
│   ├── main.zig              # Gendocs entrypoint
│   ├── source_cli.zig        # CLI source discovery
│   └── render_api_md.zig     # Markdown renderer
│
├── scripts/                  # Validation and utility scripts
│   ├── run_build.sh          # Darwin 25+ build wrapper
│   ├── fmt_repo.sh           # Format check wrapper
│   ├── baseline.zig          # Version baseline checks
│   ├── check_*.zig           # Various consistency checks
│   └── util.zig              # Shared script utilities
│
└── server/                   # V3 server executable
    └── main.zig
```

## Tests (`tests/`)

```
tests/
├── zig/                      # Zig test roots
│   ├── mod.zig               # Test module root
│   ├── database_fast_tests_root.zig
│   └── database_wdbx_tests_root.zig
│
└── (future: integration/, e2e/)
```

Test roots in `tests/zig/` are thin wrappers that re-export test declarations
from `src/`. The build system references them via `b.path("tests/zig/...")`.

## Documentation (`docs/`)

```
docs/
├── README.md                 # Documentation guide (this system)
├── STRUCTURE.md              # This file — directory tree reference
├── PATTERNS.md               # Zig 0.16 codebase patterns
├── ABI_WDBX_ARCHITECTURE.md  # WDBX vector database architecture
├── ZIG_MACOS_LINKER_RESEARCH.md  # Darwin linker workarounds
│
├── api/                      # Generated API reference (zig build gendocs)
├── plans/                    # Generated roadmap plans
└── data/                     # Structured data exports
```

Maintained docs are edited directly. Generated docs (`api/`, `plans/`, `data/`)
are produced by `zig build gendocs` — do not hand-edit.

## Other Directories

| Directory | Purpose |
|-----------|---------|
| `bindings/c/` | C FFI bindings (`abi.h` + Zig wrapper) |
| `bindings/` | WASM and other language bindings |
| `benchmarks/` | Performance suites (see `benchmarks/README.md`) |
| `examples/` | Standalone example programs (see `examples/README.md`) |
| `tasks/` | Agent task tracking (`todo.md`, `lessons.md`) |
| `.claude/` | Claude Code configuration and skills |
