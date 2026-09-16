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
- ABI-local workspace crates live under `crates/*`, including `abi-ai`,
  `abi-sea`, `abi-nn`, `abi-gpu`, `abi-wdbx-gateway`, `abi-connectors`,
  `abi-plugins`, `abi-cli`, and `abi-mcp`.
- `abi-foundation`, `abi-core`, `abi-compute`, `abi-wdbx`, and
  `abi-telemetry` are required path dependencies under sibling `../wdbx/crates`.
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
./tools/cargo.sh test --manifest-path ../wdbx/Cargo.toml -p abi-wdbx --lib -- <filter> < /dev/null

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
3. **Fake-completing residuals** — CUDA/Vulkan runtime, verified ANE residency,
   production FHE/sharding, and separate-host deployment stay disclosed Partial.
   Metal numerical paths and Discord/Twilio rustls clients are Current only at
   their locally tested boundaries.
4. **Expanding frozen surfaces** without golden + contract updates.
5. **MCP launcher** — prefer `mcp/launcher.sh` (or run from repo root) so
   `target/{release,debug}/abi-mcp` and the FM dylib resolve.
6. **Historical Zig prose** — `docs/superpowers/archive/**` and rewrite audit
   notes may still say Zig; live gates are Rust only.
7. **Rewriting a `target/<profile>/*.dylib` in place** — a build that
   `fs::copy`s or `cp`s over an existing dylib truncates the same inode; if a
   long-lived `abi-mcp stdio` still has it mapped, the kernel keeps the old code
   directory and every later exec of `abi`/`abi-mcp` dies `SIGKILL (Code
   Signature Invalid)` while `codesign -vv` says valid (2026-09-16). Build
   scripts copy to `.tmp` and `rename`; repair a broken tree with
   `cp X X.new && mv -f X.new X`, not a rebuild.
8. **Exec'ing the toolchain's `cargo` directly drops rustup's
   `DYLD_FALLBACK_LIBRARY_PATH`** — rustc on Apple targets strips release
   binaries with `lib/rustlib/<host>/bin/rust-objcopy`, whose
   `@rpath/libLLVM.dylib` only resolves through the fallback path the rustup
   proxy injects (`<toolchain>/lib` first). Without it every release link
   prints `warning: stripping debug info with rust-objcopy failed: signal: 6
   (SIGABRT)`, ships the binary unstripped, and writes a dyld crash report,
   while cargo still exits 0 (2026-09-16). `tools/cargo.sh` now exports the
   same list rustup uses; the rustc warning line is the evidence, because
   macOS throttles crash reports and their count under-reports aborts.
