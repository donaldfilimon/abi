# ABI Framework

ABI is a **Zig 0.17.0-dev.813+2153f8143** framework for local AI service orchestration, semantic vector storage, GPU capability reporting, and runtime primitives.

## Quick Start
```bash
zig version             # Confirm Zig 0.17.0-dev.813+ compatible toolchain
./build.sh check        # Primary validation gate on macOS/Darwin
./build.sh full-check   # Check + integration tests + benchmarks + TUI smoke
./build.sh cli          # Build zig-out/bin/abi
./build.sh mcp          # Build zig-out/bin/abi-mcp
```

Plain `zig build` and `zig build check` are expected to work with the pinned toolchain. On macOS/Darwin, keep using `./build.sh ...` for the documented project workflow.

## Current Status

- ABI currently targets the pinned Zig 0.17.0 development toolchain and should be validated from the live checkout before release or external reuse.
- Core feature modules and MCP transport have contract coverage; MCP HTTP can use `ABI_MCP_HTTP_PORT` when `127.0.0.1:8080` is occupied.
- Documentation: `CLAUDE.md`, `GEMINI.md`, and `AGENTS.md` describe the 0.17+ development lifecycle.
- Build: `./build.sh check` builds CLI/MCP, runs module tests, connector tests, contract tests, focused feature-off contracts, feature-aware public contracts for every `-Dfeat-*` stub, linting, and validates mod/stub parity.
- Full validation: `./build.sh full-check` executes all integration tests, benchmarks, and TUI smoke.
- GPU/modules: all feature modules keep real/stub parity; GPU status and vector operations fall back deterministically to CPU when native kernels are unavailable.
- Plugins: `tools/generate_plugin_registry.zig` automatically maintains `src/plugin_registry.zig` from bundled `abi-plugin.json` manifests, with multi-plugin metadata coverage for `name`, `version`, `description`, `target_feature`, and safe relative `.zig` `entry_point` paths. The generator and plugin manager accept `targetFeature` / `entryPoint` aliases and require entry files to exist under the plugin directory.
- AI/WDBX: API callers opt into persistence with `CompletionRequest.store_result=true`; CLI/MCP completion uses an ephemeral in-memory WDBX store for query/response vectors, metadata, and block-chain entries when WDBX is enabled.
- WDBX: contract coverage verifies ordered vector search results, block metadata round-tripping, and block-chain snapshot lookup while disabled builds return explicit `error.FeatureDisabled` operations for key-value/vector/search/block/spatial writes.
- Connectors: Discord validates printable non-whitespace credentials, numeric snowflake-like IDs, and message size; Twilio validates account SID/auth-token shape, base URL, timeout, explicit `.live` transport, TwiML/form escaping, and ConversationRelay payload aliases before local/live dispatch.
- External collateral should not cite distributed sharding, AES/RBAC, Swift, Python/TensorFlow stacks, Kubernetes/H100 deployments, regulatory certifications, QPS/latency/accuracy, energy-efficiency, or model-benchmark claims unless a repo test, benchmark artifact, or documented source file proves them; see [docs/contracts/external-claims-audit.md](docs/contracts/external-claims-audit.md).

See [docs/index.md](docs/index.md) for architecture, public API contracts, onboarding, and development guides, and [CHANGELOG.md](CHANGELOG.md) for release-note style modernization highlights.
