# Changelog

All notable ABI Framework changes are recorded here. The executable gates remain the source of truth for readiness: `./build.sh check`, `zig build check-parity`, and `./build.sh full-check`.

## Unreleased

### Fixed

- MCP shutdown use-after-free: the `shutdown` JSON-RPC method tore down shared state in-band from whichever transport thread handled it, so `deinitScheduler()` could free the `Scheduler` while the peer transport was inside a tool call holding a `*Scheduler`. The stdio (`server.zig`) and HTTP (`rpc.zig`) handlers now only *signal* shutdown; `main` runs `deinitScheduler()`/`deinitWdbxStore()` via LIFO `defer`s after the HTTP thread is joined, so teardown can't race an in-flight call (deinit is idempotent; the WAL covers crash recovery).
- MCP shared-state lazy-init TOCTOU race: `ensureScheduler`/`ensureWdbxStore` flipped the `initialized` atomic to `true` *before* constructing the backing optional, so a concurrent reader (HTTP/SSE thread + stdio loop both call `getScheduler`/`getWdbxStore`) could observe `initialized==true` while the optional was still `null` and panic on `.?`. Switched to double-checked locking under dedicated init spinlocks that publish the flag with release ordering only after construction. No contract/API/parity change.
- WDBX write-ahead-log double-frees: `Store.putVector` and `Store.store` could double-free / dangle a buffer when a WAL append IO error followed the in-memory commit. The `errdefer` now stays the sole owner across the fallible append — `putVector` moves the append above the padded-buffer free; `store` disarms the owned-key/value `errdefer`s with commit flags — preserving the deliberate memory-first / WAL-after ordering. (Latent on the persistent path; no default-build test reproduced them.)
- WDBX remote-compute reference listener (`serveOnce`) now guards the untrusted `dim * 2` against `usize` overflow via `std.math.mul` before allocating, looping, and slicing.
- `zig build check-parity` now fails when a feature/plugin leaf ships `mod.zig` with no sibling `stub.zig` (previously a silent pass); only the intentional `src/features/mod.zig` dispatcher is exempt.
- SEA learn loop logs (rather than swallows) a router weight-save failure on the durable persistence path; the CLI dashboard handler `defer`s `MemoryTracker.deinit()` to match the train/agent handlers.
- Reconciled stale docs against source: threat-model `src/abi_cli/` → `src/cli/` paths after the CLI rename; corrected the apple-fm `@c`-shim wording in CLAUDE/AGENTS/GEMINI (the shim exists and is linked; it is not a nonexistent `@_cdecl` shim) and softened the unbacked "runtime-verified on Apple-Intelligence hardware" claim per the external-claims policy.

### Added

- SEA `runLearnLoop` gains an optional `LearnLoopConfig.tracker` that makes adaptive persona-router weight persistence observable through a `MemoryTracker` (balanced, non-escaping; default off → no call-site change).
- `runCli` behavioral tests covering help/no-args (exit 0) and unknown-command / missing-required-positional (exit 2) dispatch paths.

- Hardened the modernization contract suite around root/feature namespaces, CLI/MCP tools, generated plugin registry metadata, and feature-off stub behavior.
- Added `ABI_MCP_HTTP_PORT` support for moving the MCP loopback HTTP/SSE transport off the default `127.0.0.1:8080` port.
- Expanded generated plugin metadata to include `name`, `version`, `description`, `target_feature`, and a safe relative `.zig` `entry_point`; added a second bundled plugin fixture for multi-plugin registry contract coverage.
- Added manifest validation coverage for unsafe plugin entry points, missing entry files, camelCase manifest aliases, and nested safe entry files.
- Tightened AI/WDBX completion semantics so `CompletionRequest.store_result=false` leaves WDBX unchanged, while persisted completions record query/response vectors, metadata, and chain blocks.
- Expanded WDBX store manifest output with spatial record counts and disabled-stub manifest fields that preserve the real manifest shape; added contract coverage for ordered vector search, block metadata round-tripping, and snapshot lookup.
- Hardened connector boundaries: Discord validates token shape, numeric IDs, inbound author IDs, and message size; Twilio validates credential shape, base URL, timeout, explicit `.live` transport selection, ConversationRelay aliases/wrong-typed fields, TwiML XML escaping, and URL-encoded form payloads before local/live dispatch.
- Tightened disabled-feature stubs: AI mirrors empty-input/training/agent validation while preserving requested completion models, MLIR/shader stubs validate inputs before disabled artifacts, and WDBX nested writes report disabled behavior without recording phantom vectors or blocks.
- Added AI/WDBX edge coverage for empty completion input, disabled-WDBX training degradation, append-linked completion blocks, and WDBX block-chain ownership/tamper detection.
- Added an external-claims audit doc and public-doc contract test so Drive/investor collateral can reuse only repo-backed ABI/WDBX claims.
- Added WDBX JSONL snapshot integrity, CRC32-framed write-ahead log replay/corruption detection, temporal/causal ranking primitives, and a frozen `abi wdbx` CLI namespace.
- Added honest in-process WDBX roadmap demonstrations for local consensus, backend selection with CPU fallback, int8 embedding quantization, additive aggregation, and loopback REST.
- Added the default-on `telemetry` feature surface for lightweight event/counter hooks with mod/stub parity.

### Performance

- Removed redundant work from WDBX HNSW search, WAL append/replay, and block-chain hot paths (no behavioral change; covered by existing gates).

### Tests

- Added second-pass audit coverage (additive only, no production change): `segments.readManifest` rejects a corrupt manifest (bad header / non-numeric `next_epoch` / non-numeric active token) with `InvalidManifest` instead of silently dropping live segments; SEA `gatherEvidence` populated-recall + `staticProfileLabel` attribution paths; and the SEA persist→recall round-trip (a persisted turn recalled as evidence on a later related turn).

### Validation

- Use `./build.sh check` as the baseline gate for source changes.
- Use `zig build check-parity` after public feature API changes.
- Use `./build.sh full-check` for release/readiness checks (`check` + integration tests + benchmarks + TUI smoke).
- Public docs intentionally avoid unproven external claims for distributed sharding, AES/RBAC, Swift/Python/TensorFlow implementations, Kubernetes/H100 deployments, regulatory certifications, QPS/latency/accuracy, energy use, and comparative model benchmarks.
