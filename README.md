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
zig build
```

On macOS 26.4+ (Darwin 25.x), stock prebuilt Zig may be linker-blocked. Use a host-built
Zig matching `.zigversion` prepended to `PATH`:

```bash
export PATH="$HOME/.cache/abi-host-zig/$(cat .zigversion)/bin:$PATH"
hash -r
```

## Build commands

```bash
zig build                          # Build static library (default)
zig build test --summary all       # Run tests
zig build check                    # Lint + test + stub parity
zig build lint                     # Check formatting
zig build fix                      # Auto-format
zig build check-parity             # Verify mod/stub declaration parity
zig build lib                      # Build static library artifact
```

## Feature flags

All features default to enabled except `feat-mobile` (false). Disable with `-Dfeat-<name>=false`:

```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false
zig build -Dgpu-backend=metal
zig build -Dgpu-backend=cuda,vulkan
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
│   ├── features/         # 19 comptime-gated feature modules (mod/stub/types pattern)
│   ├── services/         # Runtime services, connectors, protocols
│   └── inference/        # ML inference: sampler, scheduler, KV cache
├── build.zig             # Self-contained build root
└── build.zig.zon         # Package manifest
```

Each feature under `src/features/<name>/` follows the **mod/stub contract**:
`mod.zig` (real implementation), `stub.zig` (API-compatible no-ops), and `types.zig`
(shared types). The stub provides the same public declarations so code compiles
regardless of which features are enabled.

## Toolchain

ABI is pinned to the Zig version in `.zigversion`. On macOS 26.4, the supported local
path is a host-built Zig matching the pin.

## License

See [LICENSE](LICENSE).
