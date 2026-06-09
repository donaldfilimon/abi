# Changelog

All notable ABI Framework changes are recorded here. The executable gates remain the source of truth for readiness: `./build.sh check`, `zig build check-parity`, and `./build.sh full-check`.

## Unreleased

### Added

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

### Validation

- Use `./build.sh check` as the baseline gate for source changes.
- Use `zig build check-parity` after public feature API changes.
- Use `./build.sh full-check` for release/readiness checks (`check` + integration tests + benchmarks + TUI smoke).
- Public docs intentionally avoid unproven external claims for distributed sharding, AES/RBAC, Swift/Python/TensorFlow implementations, Kubernetes/H100 deployments, regulatory certifications, QPS/latency/accuracy, energy use, and comparative model benchmarks.
