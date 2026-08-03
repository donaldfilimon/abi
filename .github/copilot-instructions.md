# Copilot Instructions - ABI Framework

Read `AGENTS.md` first. It is the canonical instruction file; this file is a
short editor-facing reminder.

## Runtime and toolchain

- ABI is a nightly-Rust workspace under `crates/*`.
- Always use `./tools/cargo.sh`. A Homebrew stable Cargo shadows rustup on
  this machine.
- Primary gate after edits: `./tools/check.sh`.
- Compatibility entrypoint: `./build.sh check`.
- Do not recreate the removed Zig tree, `build.zig`, `.zigversion`,
  `modernized/`, or `modern-refactor/`.

## Contracts

- Frozen CLI surface: 13 top-level commands, defined in
  `crates/abi-cli/src/usage.rs` and pinned by golden tests.
- Frozen MCP surface: 12 tools, implemented under `crates/abi-mcp/` and
  pinned by contract tests.
- Do not add or rename a frozen command/tool without updating its authoritative
  source, golden fixtures, and tests together.

## Store safety

Tests and smokes must never open `~/.abi/`. Use
`ABI_WDBX_PATH=:memory:`, `ABI_WDBX_PERSIST=0`, or
`abi_foundation::temp_path::temp_file_path()`.

## Claims

Do not claim production FHE, AES/RBAC storage, multi-host sharding, native
CUDA/Vulkan/ANE execution, benchmark superiority, or public-internet hardening
without current source, tests, and artifacts. Local scheduler work is not
distributed agents. Lexical constitution telemetry is not a general safety
classifier. See `docs/contracts/external-claims-audit.mdx`.

## Workflow

1. Read `tasks/lessons.md` and `tasks/todo.md`.
2. Inspect `git status --short --branch` and preserve unrelated dirty work.
3. Make a focused change in the owning crate.
4. Run a focused test through `./tools/cargo.sh`.
5. Run `./tools/check.sh` before commit or handoff.
