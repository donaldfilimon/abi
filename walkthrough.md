# ABI Framework Walkthrough

This walkthrough covers the **stable Rust 1.99.1** ABI framework on the current
branch (post Zig teardown).

## Toolchain

- Stable Rust 1.99.1 via `rust-toolchain.toml`. Always use `./tools/cargo.sh` (never
  bare `cargo` — Homebrew stable cargo can shadow rustup).
- Primary validation is `./tools/check.sh`.

## Build Commands

```bash
./tools/check.sh
./tools/cargo.sh build -p abi-cli
./tools/cargo.sh build -p abi-mcp
./tools/cargo.sh test --workspace < /dev/null
./tools/cargo.sh clippy --locked --workspace --all-targets -- -D warnings
```

The `< /dev/null` on the test command is required: `abi auth signin` reads a
secret from stdin when `ABI_AUTH_TOKEN` is unset and stdin is not a TTY, so a
test exercising the no-token path blocks forever on an inherited open stdin.

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
$ABI wdbx query "$(mktemp -d)/store"   # store stats as JSON (no `wdbx stats` command)
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
./mcp/launcher.sh stdio
```

Launch MCP through `mcp/launcher.sh` (it picks `target/release/abi-mcp`, then
`target/debug/abi-mcp`, and `cd`s to the repo root so the FoundationModels
shim resolves via `@loader_path`). A direct `./target/debug/abi-mcp stdio` can
hide a stale build or fail to resolve the dylib.

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
