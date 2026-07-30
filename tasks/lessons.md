# Lessons — ABI Framework

Session-start checklist and conventions for agents working on this repo.

## Session-Start Checklist

1. Read this file (`tasks/lessons.md`) at session start.
2. Read `tasks/todo.md` for current work items and priorities.
3. Run `./tools/check.sh` (or `./build.sh check`) to verify baseline before
   making changes.
4. Identify which crate(s) you are touching under `crates/`.
5. Update `tasks/todo.md` as you begin and complete work items.

## Key Conventions

### Toolchain
- **Nightly Rust** via `rust-toolchain.toml`.
- Homebrew stable `cargo` shadows rustup — **always** use `./tools/cargo.sh`.
- Primary gate: `./tools/check.sh` (fmt, clippy `-D warnings`, build, test, docs).

### Naming (Rust)
- Functions/variables/modules: `snake_case`
- Types/traits: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`

### Crate layout
- Live code under `crates/*` (`abi-foundation`, `abi-core`, `abi-ai`, `abi-sea`,
  `abi-nn`, `abi-gpu`, `abi-wdbx`, `abi-connectors`, `abi-plugins`,
  `abi-telemetry`, `abi-cli`, `abi-mcp`).
- Frozen CLI (13) and MCP (12) surfaces — see `AGENTS.md`.
- Golden fixtures under `tests/golden/` pin help/MCP contracts.

### Error handling
- No silent swallow on persistence, inference, or connector paths.
- Prefer typed `Result` / domain errors; log or propagate.

### Testing
- Prefer unit tests in the crate (`--lib`) plus golden/integration tests in
  `crates/*/tests/` and workspace fixtures.
- Focused run: `./tools/cargo.sh test -p <crate> --lib -- <filter>`
- Never open the user's real `~/.abi/` store — use scratch paths or
  `ABI_WDBX_PATH=:memory:` / `ABI_WDBX_PERSIST=0`.

## Build/Test Workflow

```bash
# Baseline check
./tools/check.sh
# compat: ./build.sh check

# Build binaries
./tools/cargo.sh build -p abi-cli
./tools/cargo.sh build -p abi-mcp

# Focused tests
./tools/cargo.sh test -p abi-wdbx --lib -- <filter>

# Format / lint
./tools/cargo.sh fmt --all
./tools/cargo.sh clippy --workspace --all-targets -- -D warnings
```

## Claims discipline

No unproven claims (production FHE/AES/RBAC, multi-host sharding, QPS/latency/
accuracy, K8s/H100, native CUDA/ANE kernels). GPU reports `accelerated=false`
when kernels are not linked. WDBX secure demos are reference-grade. Audit:
`docs/contracts/external-claims-audit.mdx`.

## Common Pitfalls to Avoid

1. **Bare `cargo`** — always `./tools/cargo.sh` (nightly pin + link environment).
2. **Opening `~/.abi/` in tests** — use scratch / in-memory env vars.
3. **Fake-completing residuals** — Metal/CUDA kernels, live Discord/Twilio TLS
   without proxy, production FHE/sharding stay disclosed Partial.
4. **Expanding frozen surfaces** without golden + contract updates.
5. **MCP launcher** — prefer `mcp/launcher.sh` (or run from repo root) so
   `target/{release,debug}/abi-mcp` and the FM dylib resolve.
6. **Historical Zig prose** — `docs/superpowers/archive/**` and rewrite audit
   notes may still say Zig; live gates are Rust only.
