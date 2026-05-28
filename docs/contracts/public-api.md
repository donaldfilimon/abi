# ABI Public API Contract

This document records the public surfaces that should remain stable across feature flags and refactors. Treat it as a human-readable companion to `tests/contracts/` and `zig build check-parity`.

## Root exports

`src/root.zig` exports these top-level namespaces:

- `interfaces`
- `foundation`
- `features`
- `registry`
- `config`
- `connectors`
- `memory`
- `scheduler`
- `plugins`

## Feature gates

Feature modules are selected by `src/features/mod.zig` from `build_options` and must keep `mod.zig` / `stub.zig` public API parity. `zig build check-parity` checks top-level public declaration names for feature/plugin pairs; it is not a complete signature/type equivalence checker.

- `ai`
- `accelerator`
- `gpu`
- `hash`
- `metrics`
- `mlir`
- `os_control`
- `shaders`
- `tui`
- `wdbx`
- `mobile`

Disabled features should return stable no-op or explicit disabled-feature responses rather than removing public symbols. `tools/check_feature_stubs.sh` now runs CLI compilation, focused `zig build test-feature-contracts -Dfeat-*=false`, and feature-aware `zig build test-contracts -Dfeat-*=false` coverage for every feature flag, plus `-Dfeat-mobile=true` smoke coverage for the normally-disabled mobile implementation.

The GPU feature has a stable runtime contract across real and disabled builds: all backends are listed, `detectBackend()` returns a safe backend status, vector operations remain deterministic via CPU fallback when native kernels are unavailable, and `backendStatusReport()` explains whether Metal/native kernels are active or falling back.

The WDBX store contract exposes key-value counts, vector counts, block counts, spatial record counts, vector dimensions, next vector ID, and acceleration status via `stats()` and `exportManifest()`. Default contract tests verify non-increasing vector search scores, block metadata round-tripping, manifest snapshots, and block-chain snapshot lookup. Disabled WDBX builds preserve the same manifest shape where possible and add `"disabled":true`; key-value/vector/search/block/spatial write paths return `error.FeatureDisabled` instead of fabricating successful persistence.

This contract does not prove distributed sharding, AES-256 encryption, RBAC, Swift/Python/TensorFlow implementations, Kubernetes/H100 deployments, regulatory certifications, production QPS/latency/accuracy, energy-use, or comparative model-benchmark claims. Keep those out of public collateral unless they are backed by new executable tests, benchmark artifacts, or implementation files.

## AI/WDBX completion contract

`abi.features.ai.CompletionRequest.store_result` defaults to `false`. `completeWithStore()` must not mutate WDBX when storage is not requested, when input is invalid, or when WDBX is disabled.

When WDBX is enabled and `store_result=true`, `completeWithStore()` stores JSON completion metadata under `completion:<query_vector_id>`, records query/response vectors, and appends a conversation block linked to the previous block. `CompletionResult.query_vector_id`, `CompletionResult.response_vector_id`, and `CompletionResult.block_id` expose those persisted IDs when storage succeeds; they remain `null` when persistence is skipped or unavailable. The metadata includes `kind`, `model`, `profile`, `audit_passed`, byte counts, and query/response vector IDs. `abi.features.wdbx.Store.verifyBlocks()` exposes public chain-integrity verification across real and disabled WDBX builds.

## CLI command contract

The CLI command surface is guarded by `tests/contracts/surface.zig` and currently includes the frozen `src/abi_cli/usage.zig` command array:

- `help`
- `complete`
- `train`
- `agent`
- `backends`
- `plugin`
- `auth`
- `twilio`
- `tui`
- `dashboard`

The top-level `abi --tui` shortcut is handled in `src/main.zig` outside the frozen `usage.commands` array. `abi complete` prints completion persistence observability (`persisted=`, WDBX counts, vector IDs, metadata key, and block hash when available) for its transient in-memory WDBX store.

## MCP contract

The MCP server exposes JSON-RPC 2.0 over stdio and loopback HTTP/SSE. Feature-backed tools must expose explicit degraded responses when AI or WDBX is disabled; for example, `ai_train` reports `training disabled` when the AI stub returns `accepted=false`, and `ai_complete` rejects empty input before touching WDBX. Contract tools are:

- `ai_run`
- `ai_complete`
- `ai_train`
- `wdbx_query`
- `scheduler_stats`
- `gpu_status`
- `wdbx_stats`
- `plugin_run`

HTTP defaults to `127.0.0.1:8080`; use `ABI_MCP_HTTP_PORT` to select another loopback port. Empty, invalid, zero, or out-of-range overrides fall back to `8080`; HTTP bind failure is non-fatal and leaves stdio running.

## Plugin registry contract

Plugin discovery is static and generated at build time from `src/plugins/*/abi-plugin.json`. Bundled plugin directories validated by `src/foundation/plugin_validator.zig` include `mod.zig`, `stub.zig`, and `abi-plugin.json`.

Required manifest fields:

- `name`
- `version`
- `description`
- `target_feature`
- `entry_point` (safe relative `.zig` path; no absolute paths, traversal, empty/`.`/`..` segments, Windows drive separators, or backslashes; the file must exist under the plugin directory)

`targetFeature` and `entryPoint` manifest aliases are accepted by the generator and plugin manager. `src/core/registry.zig` stores generated plugins as `PluginDescriptor` values containing all manifest metadata and exposes stable accessors such as `getPlugin()`, `pluginCount()`, `appendPluginNames()`, `snapshotPluginNames()`, `snapshotPlugins()`, and `formatPluginList()`. `abi plugin list` includes plugin count, version, target feature, entry point, and description. The legacy `Registry.register(name, info)` helper remains supported and maps `info` to `description` for compatibility.

`tests/contracts/plugin_registry.zig` verifies generated metadata for the bundled example plugins, including the WDBX-targeted fixture.

## Connector contract

Connectors provide deterministic local behavior and explicit live-transport boundaries.

Discord connector calls validate:

- non-empty printable ASCII credentials without whitespace
- numeric snowflake-like client/channel/author IDs
- non-empty message content no larger than 2000 bytes

Twilio connector calls validate:

- account SIDs shaped as `AC` plus 32 hex characters
- auth tokens shaped as 32 hex characters
- non-empty base URL and non-zero timeout
- ConversationRelay aliases (`event`, `callSid`, `from`, camelCase memory/intelligence fields) and wrong-typed payload rejection before building local responses or live TwiML/form payloads
- XML escaping for TwiML `<Say>` and `<Redirect>` text and URL-encoded form fields

OpenAI and Anthropic connectors validate shared connector config and keep local streaming responses deterministic. Live HTTP dispatch remains an explicit `.live` transport path, and connector tests validate malformed live-path inputs before network dispatch.

## Validation gates

For source changes:

```bash
./build.sh check
```

For release/readiness changes:

```bash
./build.sh full-check
```

`full-check` is `check` plus integration tests, benchmarks, and TUI smoke.

For public feature API changes:

```bash
zig build check-parity
```
