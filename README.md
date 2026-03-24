# ABI Framework

ABI is a Zig 0.16 framework for AI services, semantic vector storage, GPU acceleration,
and distributed runtime. The public package entrypoint is `src/root.zig`, exposed to
consumers as `@import("abi")`.

## What ABI includes

- `abi.App` / `abi.AppBuilder` for framework setup and feature wiring
- `abi.database` for the semantic store and vector search surface
- `abi.ai` for agents, profiles, training, reasoning, and LLM support
- `abi.inference` for engine, scheduler, sampler, and paged KV cache primitives
- `abi.gpu` / `abi.Gpu` / `abi.GpuBackend` for unified compute backends
- `abi.runtime`, `abi.platform`, `abi.connectors`, `abi.mcp`, `abi.acp`, `abi.tasks` for services
- `abi.foundation` for shared utilities (SIMD, logging, security, time)

## Quick start

```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
./build.sh                  # macOS 26.4+ (auto-relinks with Apple ld)
zig build                   # Linux / older macOS
```

On macOS 26.4+ (Darwin 25.x), stock prebuilt Zig's LLD linker cannot link binaries.
Use `./build.sh` which auto-relinks with Apple's native linker. To install and symlink
the correct Zig version:

```bash
tools/zigup.sh --status     # Auto-install if missing
tools/zigup.sh --link       # Symlink to ~/.local/bin
```

## Build commands

```bash
./build.sh                         # Build (macOS 26.4+ auto-relinks with Apple ld)
./build.sh test --summary all      # Run tests via wrapper (macOS 26.4+)
zig build                          # Build static library (Linux / older macOS)
zig build test --summary all       # Run tests
zig build check                    # Lint + test + stub parity
zig build lint                     # Check formatting
zig build fix                      # Auto-format
zig build check-parity             # Verify mod/stub declaration parity
zig build feature-tests            # Run feature integration and parity tests
zig build mcp-tests                # Run MCP integration tests
zig build cli-tests                # Run CLI tests
zig build tui-tests                # Run TUI tests
zig build typecheck                # Compile-only validation for the current/selected target
zig build validate-flags           # Validate feature flags
zig build full-check               # Run full check
zig build verify-all               # Verify all components
zig build cross-check              # Verify cross-compilation (linux, wasi, x86_64)
zig build lib                      # Build static library artifact
zig build mcp                      # Build MCP stdio server (zig-out/bin/abi-mcp)
zig build doctor                   # Report build configuration and diagnostics
```

## CLI

Build the CLI binary with `zig build cli` (or `./build.sh cli` on macOS 26.4+).
The binary lands at `zig-out/bin/abi`. Entry point: `src/main.zig`.

```bash
abi                # Smart status (feature count + available commands)
abi version        # Print version and build info
abi doctor         # Run diagnostics (all feature flags + GPU backends)
abi features       # List all 32 features with [+]/[-] status
abi platform       # Show platform detection (OS, arch, CPU, GPU)
abi connectors     # List 16 LLM provider connectors
abi info           # Framework architecture summary
abi chat <message...>  # Route through multi-persona pipeline
abi db <cmd>       # Vector database (add, query, stats, diagnostics, optimize, backup, restore, serve)
abi dashboard      # Interactive TUI (requires -Dfeat-tui=true)
abi help           # Full help reference
```

## Pipeline architecture

The multi-persona pipeline (Abbey-Aviva-Abi) is wired end-to-end in `src/features/ai/persona/router.zig`:

```
User Input
  -> Abi Analysis        (sentiment + policy + rules)
  -> Modulation          (EMA user preference learning)
  -> Routing Decision    (single / parallel / consensus)
  -> Persona Execution   (Abbey / Aviva / Abi)
  -> Constitution        (6-principle ethical validation)
  -> WDBX Memory         (cryptographic block-chain storage)
  -> Response
```

## Feature flags

All features default to enabled except `feat-mobile` and `feat-tui` (both false). Disable with `-Dfeat-<name>=false`:

```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false    # Disable GPU and AI features
zig build -Dfeat-training=false                # Disable training sub-feature only
zig build -Dfeat-mobile=true                   # Enable mobile (off by default)
zig build -Dgpu-backend=metal                  # Single GPU backend
zig build -Dgpu-backend=cuda,vulkan            # Multiple GPU backends
```

## Public surface

| Surface | Purpose |
|---------|---------|
| `abi.App` / `abi.AppBuilder` | Framework lifecycle and feature orchestration |
| `abi.database` | Semantic store, search, backup, restore, diagnostics |
| `abi.ai` | Agents, profiles, LLM, training, reasoning |
| `abi.inference` | Engine, scheduler, sampler, and paged KV cache primitives |
| `abi.gpu` / `abi.Gpu` / `abi.GpuBackend` | GPU compute backends |
| `abi.foundation` / `abi.runtime` | Shared foundations, time/sync/SIMD, runtime primitives |
| `abi.search` | BM25 full-text search with inverted index persistence |
| `abi.connectors` / `abi.ha` / `abi.tasks` / `abi.lsp` / `abi.mcp` / `abi.acp` | Service and integration surfaces |

## Project structure

```
abi/
├── src/                  # Framework source (single "abi" module)
│   ├── root.zig          # Public package entrypoint (@import("abi"))
│   ├── core/             # Always-on framework internals
│   ├── features/         # 20 feature directories (32 features including AI sub-features)
│   ├── foundation/       # Shared utilities: logging, security, time, SIMD, sync
│   ├── runtime/          # Task scheduling, event loops, concurrency
│   ├── platform/         # OS detection, capabilities, environment
│   ├── connectors/       # External service adapters (OpenAI, Anthropic, Discord)
│   ├── protocols/        # MCP, LSP, ACP, HA protocol implementations
│   ├── tasks/            # Task management, async job queues
│   └── inference/        # ML inference: engine, scheduler, sampler, KV cache
├── build.zig             # Self-contained build root
├── build.zig.zon         # Package manifest
└── tools/                # zigup.sh, crossbuild.sh, auto_update.sh
```

Each feature under `src/features/<name>/` follows the **mod/stub contract**:
`mod.zig` (real implementation), `stub.zig` (API-compatible no-ops), and `types.zig`
(shared types). The stub provides the same public declarations so code compiles
regardless of which features are enabled.

## Tools

```bash
tools/zigup.sh --status    # Print zig path (auto-install if missing)
tools/zigup.sh --install   # Force re-download zig + ZLS
tools/zigup.sh --link      # Symlink zig + zls into ~/.local/bin
tools/zigup.sh --unlink    # Remove symlinks from local bin
tools/zigup.sh --update    # Check for newer zig and update if available
tools/zigup.sh --check     # Report if update available (no download)
tools/zigup.sh --clean     # Remove all cached versions
tools/crossbuild.sh        # Cross-compile for linux, wasi, x86_64 targets
tools/auto_update.sh       # Check and apply updates for zig + zls
```

Cache location: `~/.cache/abi-zig/<version>/bin/{zig,zls}`

## Toolchain

ABI is pinned to the Zig version in `.zigversion`. On macOS 26.4+, `./build.sh`
auto-relinks with Apple's native linker. On Linux / older macOS, `zig build` works directly.

## Testing

```bash
zig build test --summary all       # Run all unit + integration tests
./build.sh test --summary all      # Same, via macOS 26.4+ wrapper
zig build check                    # Full gate: lint + test + stub parity
zig build check-parity             # Verify mod/stub declaration parity only
```

Two test suites run under `zig build test`:
1. **Unit tests** (`src/root.zig`) -- `refAllDecls` walks the entire module tree.
2. **Integration tests** (`test/mod.zig`) -- 38 modules covering database, inference, persona pipeline, security, CLI, TUI, and all features.

## Specification

See [docs/spec/ABBEY-SPEC.md](docs/spec/ABBEY-SPEC.md) for the comprehensive mega spec covering
architecture, personas, behavioral model, math foundations, ethics, and benchmarks.

## License

See [LICENSE](LICENSE).
