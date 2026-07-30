# ABI Framework Walkthrough

This walkthrough covers the **nightly Rust** ABI framework on the current
branch (post Zig teardown).

## Toolchain

- Nightly Rust via `rust-toolchain.toml`. Always use `./tools/cargo.sh` (never
  bare `cargo` — Homebrew stable cargo can shadow rustup).
- Primary validation is `./tools/check.sh`.

## Build Commands

```bash
./tools/check.sh
./tools/cargo.sh build -p abi-cli
./tools/cargo.sh build -p abi-mcp
./tools/cargo.sh test --workspace
./tools/cargo.sh clippy --workspace --all-targets -- -D warnings
```

Compatibility: `./build.sh check` → `./tools/check.sh`.

## Local surfaces

```bash
./tools/cargo.sh build -p abi-cli
ABI=./target/debug/abi

$ABI backends
$ABI scheduler status
$ABI dashboard --once --plain
$ABI complete "Summarize the current ABI runtime status"
$ABI complete --neural "hello"
$ABI train "example"
$ABI agent plan "stage a safe WDBX refactor"
$ABI agent train all
$ABI plugin list
$ABI wdbx stats
$ABI wdbx compute info
$ABI wdbx secure demo
$ABI wdbx cluster status
```

Non-interactive auth sign-in (does not touch the real `~/.abi/credentials.json`
when isolated with `ABI_CREDENTIALS_PATH`):

```bash
ABI_AUTH_TOKEN=sk-test $ABI auth signin anthropic
$ABI auth status
$ABI auth logout
```

## MCP

```bash
./tools/cargo.sh build -p abi-mcp
./target/debug/abi-mcp stdio
# or: mcp/launcher.sh
```

Frozen MCP tools: `ai_run`, `ai_complete`, `ai_learn`, `ai_train`, `wdbx_query`,
`scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`,
`plugin_list`, `wdbx_stats`, `plugin_run`.

## GPU honesty

`abi backends` and MCP `gpu_status` report preferred backend metadata with
`accelerated=false` when native kernels are not linked. Vector ops use
deterministic CPU SIMD fallback. CUDA/Vulkan/ANE execution is a non-claim.

## Validation

```bash
./tools/check.sh
```
