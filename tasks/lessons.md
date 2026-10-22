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
- **Stable Rust 1.99.1** via `rust-toolchain.toml`.
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
- Frozen CLI (14) and MCP (12) surfaces — see `AGENTS.md`.
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

1. **Bare `cargo`** — always `./tools/cargo.sh` (repository pin + link environment).
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
9. **The gate's Python tooling needs 3.11+** — `tools/ci_contract.py`,
   `tools/toolchain_policy.py`, and several `tools/tests/*` import `tomllib`
   (3.11+). macOS still ships `/usr/bin/python3` at 3.9, and a bare `python3`
   can resolve to it, which killed the gate with
   `ModuleNotFoundError: No module named 'tomllib'`. `tools/check.sh` and
   `tools/cargo.sh` now resolve a capable interpreter explicitly (versioned
   `python3.X` first, then `python3`/`python`; override with `ABI_PYTHON`) and
   fail closed with an actionable message instead of a traceback. Note the
   related unittest change: Python 3.12+ exits **5** when discovery finds no
   tests, where 3.9 exited 0 — a fixture with an empty `tools/tests` therefore
   fails the gate on a modern interpreter.
10. **The required dependency scanner is not on the default PATH** —
    `cargo-audit`/`cargo-deny` live in `~/.cargo/bin`, so
    `ABI_DEP_SCAN_REQUIRE=1 ./tools/security/run-dep-scan.sh` failed closed
    with "no dependency scanner ran" on any host that did not export it. The
    scanner script and `tools/cargo.sh` now add `~/.cargo/bin` to PATH
    themselves, so the required scan no longer depends on the caller's
    environment. A SKIP is not evidence: the scan must actually RUN.
11. **`set -euo pipefail` turns a check into an abort** — a "read a value and
    report it" step written as `VAR="$(cmd | head -1)"` kills the whole script
    when `cmd` fails, *before* the intended diagnostic branch, and the script
    never reaches its summary. `tools/github-runner/harden-macos.sh` had this
    twice (missing `rust-toolchain.toml`; `df` on a missing install dir). In a
    checklist-style script, compute the value with `|| true` and let the
    explicit branch report the problem — otherwise a missing file reads as a
    silent crash instead of a diagnostic. The same trap applies to
    `$(compgen ... | grep ...)`: a `grep` that matches nothing is fine inside
    a process substitution (its status is not propagated), but a bare
    `$(grep ...)` in an assignment is not.
12. **A capability probe must use a DISTINGUISHING exit status, not "exit 0"** —
    `if "$candidate" -c '...'; then` accepts *any* program that ignores its
    argument and exits 0 (`/bin/echo`, `/bin/true`, a broken pyenv/asdf shim).
    When that candidate is the interpreter a check depends on, the check
    silently becomes a no-op: `ABI_PYTHON=/bin/echo` made the gate's Python
    policy tests and the Abbey Python oracle do nothing while the gate still
    reported "all green", and it disabled the compiler-substitution policy in
    `tools/cargo.sh` (which exists precisely to reject a substituted `RUSTC`).
    Use a unique status (`sys.exit(3 if capable else 4)`) and accept only
    that. Mutate the probe to the weak form and confirm the regression
    actually fails — a green test against a weak probe is not evidence.
13. **Never probe by mutating a repo file in place** — a probe that copies
    `tools/cargo.sh` aside and restores it with `cp` at the end will leave the
    repo holding the probe if the shell is killed before the restore (a
    timeout, an interrupt). Copy the file, mutate the copy, and run the copy;
    or restore in a `trap`.

## 2026-10-20 — owner integration and source qualification

- Host mocks and separately tested stores missed incompatible cancellation
  terminals and stage 11 denial audits. Exercise the actual RuntimeStore join,
  including reopen/replay and approval consumption, before accepting the slice.
- macOS protected shells remove inherited DYLD fallback variables. Restore them
  inside each Cargo invocation through tools/cargo.sh; an outer export is not
  evidence. Inspect strip warnings even when Cargo exits0.
- Bind every receipt to exact content/modes, compiler and feature scope. A later
  consumer fixture or moving sibling cannot inherit an older full-gate claim.
  Preserve intermediate failures and do not rerun unrelated producer gates when
  only a separately qualified consumer test/reporting delta changes.

## 2026-10-22 — metadata line contract and unguarded `$HOME`

- **The first output line of a CLI command is a contract, not a layout choice.**
  The checked-in SEA smoke reads only `out`'s first line, deliberately: the
  generated response text is free to repeat arbitrary markers, so a later line
  could otherwise make a failed learning write look healthy. Splitting the
  learn fields onto their own first line looked like tidier output and broke a
  healthy run with "missing metadata: provider=local". A Rust unit test that
  stubs the CLI cannot catch this class of drift — the real smoke (and its
  `.claude` mirror) must run, and the unit test should pin the exact first-line
  shape the script depends on. When a frozen surface changes, check every
  consumer of the bytes, not just the crate that emits them.
- **Under `set -u`, every `$HOME` dereference needs a guard.** `tools/cargo.sh`
  runs before every build and gate step, so an unguarded `${HOME}` aborts the
  whole toolchain in a minimal container or an `env -i` job. Both dereferences
  (the `~/.cargo/bin` scanner entry and rustup's `~/lib` dyld fallback) now
  come from one guarded `HOME_ROOT`; `tools/security/run-dep-scan.sh` already
  used `${HOME:-}` for the same reason.
