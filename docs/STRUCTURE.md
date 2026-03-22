# ABI Directory Structure

Comprehensive directory tree for the ABI framework, annotated with the purpose
of each directory and key file patterns.

## Top-level Layout

```
abi/
├── build.zig                 # Build root — defines all steps, targets, gates
├── build.zig.zon             # Package manifest (Zig 0.16 format)
├── .zigversion               # Pinned Zig version (0.16.0-dev.2962+...)
│
├── src/                      # All framework source — single "abi" module
│   ├── root.zig              # Public package entrypoint (@import("abi"))
│   ├── core/                 # Always-on framework internals
│   ├── features/             # Comptime-gated feature modules (19 directories)
│   ├── services/             # Runtime services shared across features
│   └── inference/            # Engine, scheduler, sampler, KV cache (Top-level API)
│
├── build/                    # Modular build system (Zig source files)
├── tools/                    # CLI, docs generator, validation scripts
├── tests/                    # Legacy standalone tests (build uses src/-rooted test entrypoints)
├── examples/                 # Standalone example programs
├── benchmarks/               # Performance benchmark suites
├── bindings/                 # C and WASM language bindings
├── lang/                     # High-level language bindings (Swift, Kotlin); wraps bindings/c/
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
│
├── core/                     # Always-on internals
│   ├── config/               # Configuration loading and validation
│   ├── database/             # Core database engine (WDBX, HNSW, vector search)
│   ├── framework/            # Framework abstractions
│   ├── registry/             # Service registry and lifecycle
│   ├── feature_catalog.zig   # Source of truth for feature metadata
│   ├── errors.zig            # Shared error types
│   ├── mod.zig               # Core module root
│   └── stub_context.zig      # StubFeature / StubFeatureNoConfig helpers
│
├── features/                 # Comptime-gated modules (one dir per feature)
│   ├── ai/                   # Agents, profiles, training, reasoning, LLM, embeddings
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
├── services/                 # Runtime services shared across features
│   ├── shared/               # Common utilities, signal handling, security
│   ├── acp/                  # Agent Communication Protocol
│   ├── connectors/           # External service connectors
│   ├── ha/                   # High-availability clustering
│   ├── lsp/                  # Language Server Protocol
│   ├── mcp/                  # Model Context Protocol
│   ├── platform/             # Platform abstraction layer
│   ├── runtime/              # Runtime scheduling and lifecycle
│   ├── tasks/                # Task execution engine
│   └── tests/                # Service test root
│
└── inference/                # Canonical inference runtime (abi.inference)
    ├── mod.zig
    ├── engine.zig
    ├── scheduler.zig
    ├── sampler.zig
    └── kv_cache.zig
```

### The mod/stub/types.zig Pattern

Every feature module under `src/features/<name>/` follows a strict contract:

```
src/features/<name>/
├── mod.zig          # Real implementation (used when feature is enabled)
├── stub.zig         # API-compatible no-ops (used when feature is disabled)
└── types.zig        # Shared types imported by both mod and stub (when needed)
```

`types.zig` is required when the module has public types that both `mod.zig`
and `stub.zig` must share. Thin modules without shared type contracts may omit
it. When in doubt, add `types.zig` to prevent mod/stub type drift.

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
├── options.zig               # 27 feature flag definitions
├── flags.zig                 # 58 flag combination validations
├── modules.zig               # Module creation, import wiring, version parsing
├── module_catalog.zig        # Module registry for gendocs
├── targets.zig               # Example/target tables, cross-compilation matrix
├── test_discovery.zig        # Unified abi-module test root (feature tests)
├── link.zig                  # Platform linking (macOS, Linux, Windows, BSD, Android, illumos, Haiku)
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
├── integration_test.zig
├── distributed_integration.zig
├── hnsw_test.zig
├── profiles_test.zig
└── simd_test.zig
```

These are legacy standalone test files. Primary test entrypoints are rooted in
`src/` for module path compliance. The build system references them directly
via `b.path("src/services/tests/mod.zig")` and similar `src/`-rooted paths.
Former shim wrappers in `tests/zig/` have been removed.

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
| `lang/` | High-level language bindings (Swift, Kotlin); wraps `bindings/c/` |
| `tasks/` | Agent task tracking (`todo.md`, `lessons.md`) |
| `.claude/` | Claude Code configuration and skills |
