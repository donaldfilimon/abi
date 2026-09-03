# ABI Framework

> **Intelligence Without Limits — local runtime.**  
> Nightly Rust framework for agent orchestration, WDBX semantic storage,  
> and claim-honest capability reporting. Inspectable on your machine.


**Intelligence Without Limits.** Local AI orchestration with inspectable memory.

ABI is a **nightly Rust** framework for local AI service orchestration, semantic
vector storage, claim-honest GPU capability reporting, and runtime primitives.
abi is the runtime/WDBX twin; Abbey is the companion that will not claim what
the ledger cannot prove. Cloud backends are optional. Browser autonomy is not
Current. See [docs/brand.md](docs/brand.md).

Project site: <https://donaldfilimon.github.io/abi/> — deployed from `site/`
through the GitHub Pages Actions workflow when that source changes on `main`
(see `site/README.md`); its benchmark dashboard renders synthetic sample data,
not measurements.

Companion interface: [abbey](https://github.com/donaldfilimon/abbey) — Intelligence Without Limits with a claims ledger.

## Quick Start
```bash
./tools/cargo.sh --version   # rustup nightly via tools/cargo.sh (never bare cargo)
./tools/check.sh             # Primary validation gate
./tools/cargo.sh build -p abi-cli   # target/debug/abi
./tools/cargo.sh build -p abi-mcp   # target/debug/abi-mcp
```

Compatibility: `./build.sh check` → `./tools/check.sh`. The Zig tree has been
removed; see `RUST-REWRITE-PLAN.md` for the port history.

## Cross-compilation

Cross targets are available via standard Rust `cargo` target triples once the
corresponding toolchains are installed (`rustup target add …`). Prefer
`./tools/cargo.sh build -p abi-cli --target <triple>` for smoke builds.

## Local Walkthrough

Build the CLI, then exercise the local surfaces without live network credentials:

```bash
./tools/cargo.sh build -p abi-cli
ABI=./target/debug/abi
$ABI backends
$ABI scheduler status
$ABI dashboard --once --plain   # one-shot operational snapshot
$ABI complete "summarize ABI scheduler status"
$ABI complete --model fable-5 "summarize ABI scheduler status"
$ABI complete --neural "hello"  # in-process char-LM demo (not a production LLM)
# arm64 macOS + Apple Intelligence ready:
# $ABI complete --live --model apple-fm --confirm "hello"
$ABI complete --model llama/phi3 "hello"  # local OpenAI-compat bridge (falls back offline)
$ABI agent plan "stage a safe WDBX refactor"
$ABI agent train all
$ABI wdbx stats
$ABI wdbx compute info
$ABI wdbx secure demo
$ABI wdbx cluster status
```

`abi scheduler status` runs a one-shot scheduler probe and reports task/memory counters. `abi help --json [command] [subcommand]` emits typed command/subcommand plus shortcut and completion-shell metadata for automation, and `abi help --completion <bash|zsh|fish>` emits metadata-driven shell completions. `abi dashboard` / `abi tui` renders the operational diagnostics snapshot; `abi --tui` is the same dashboard shortcut. With non-TTY stdin or `--once` it exits after one frame; on a real TTY it enters raw-mode refresh until `q`/Esc. Use `--pane <pane>`, `--plain`/`--no-color`, `--compact`, `--once`, `--interval <ms>` (100-60000), `--json`, and `--list-panes` to choose the initial pane, log-safe styling, selected-pane-only rendering, forced one-shot rendering, refresh cadence, a machine-readable snapshot, or pane metadata; JSON snapshots include layout metadata (`compact`, color, visible panes, pane titles/hotkeys). `abi agent plan`, `train`, `multi`, and `spawn` use the local scheduler-backed AI helper surface; `abi agent tui` is the interactive REPL with `/status` session telemetry and validated printable model ids. `abi agent browser` emits a reviewed local plan and local planner output only—ABI does not embed or launch a browser, and real navigation remains an external MCP integration step. The `abi complete` and `abi agent train` CLI paths, plus MCP `ai_complete`, `ai_train`, and `ai_learn`, attempt the configured WDBX store, or `$HOME/.abi/wdbx` by default; operators opt out with `ABI_WDBX_PERSIST=0` or `ABI_WDBX_PATH=:memory:`. Skipped, unavailable, or failed persistence is reported rather than fabricated. The separate top-level `abi train` command only inspects the requested training configuration and does not open WDBX. `wdbx query <path> [text] [persona]` does store-stats, hybrid semantic search, or persona-isolated retrieval; `cluster serve`/`compute info`/`secure demo` expose the networked consensus RPC, accelerator selection, and security demos honestly as single-host / reference-scoped surfaces (see `docs/spec/wdbx-north-star.mdx` for the Current/Partial/Proposed mapping). Cluster RPC supports shared-secret frames via `ABI_WDBX_CLUSTER_TOKEN` and an optional node allowlist via `ABI_WDBX_CLUSTER_PEERS`; non-loopback binds refuse to start without the token. `complete --model <id>` records the catalog-canonical id (aliases such as `fable-5` -> `claude-fable-5`; unrecognized ids print a stderr warning and pass through); `complete --live` serves anthropic-provider models over the explicit live transport and therefore needs stored credentials (`ABI_AUTH_TOKEN=… abi auth signin anthropic`).

For MCP smoke testing, build the server and call the same contract tools through an MCP client:

```bash
./tools/cargo.sh build -p abi-mcp
# Prefer mcp/launcher.sh (or run from repo root) so @loader_path resolves
# libabi_fm_shim.dylib next to the binary on arm64 macOS.
./mcp/launcher.sh
# stdio is primary; optional custom loopback HTTP compatibility listener on
# ABI_MCP_HTTP_PORT (default 8080), with optional ABI_MCP_HTTP_TOKEN bearer auth.
# GET /sse only advertises POST /message; it is not persistent MCP HTTP+SSE.
```

Contract-covered MCP tool names are `ai_run`, `ai_complete`, `ai_train`, `ai_learn`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, and `plugin_run`. `wdbx_query` returns a local hybrid-ranked match from the configured WDBX store; `connector_test` uses deterministic local connector paths and does not perform live network dispatch.

## Current Status

- ABI targets **nightly Rust** (`rust-toolchain.toml`); validate with `./tools/check.sh`.
- Core crates and MCP transport have contract/golden coverage; MCP stdio is primary, with an optional custom loopback HTTP compatibility listener whose `GET /sse` only advertises `POST /message`, not a persistent spec-conforming MCP HTTP+SSE channel.
- Documentation: `CLAUDE.md`, `GEMINI.md`, and `AGENTS.md` describe the Rust lifecycle.
- Build gate: `./tools/check.sh` runs policy tests, `./tools/cargo.sh xtask ci verify` (judo #817), Abbey corpus (Python + `./tools/cargo.sh xtask abbey verify`), size limits, fmt, clippy (`-D warnings`), workspace build/tests, and docs.
- Local models: `abi-model-runtime` requires an exact registry model, accepting principal, external storage root, and device choice. Generated scratch fixtures prove its tiny `abi-bigram-v1` Candle path on CPU and locally exercised Metal; this is runtime-foundation evidence, not a Gemma, quality, placement, speedup, or CUDA-runtime claim.
- Workers: `abi-worker` provides authenticated, audience-bound, bounded offline admission/control contracts with finite leases, cooperative cancellation, replay resistance, and verifiable results. It reuses the gateway mTLS configuration but does not claim a listener, scheduler, model execution, production cluster, or separate-host deployment.
- GPU: capability table + preferred backend reporting with **honest `accelerated=false`** when native kernels are not linked; vector ops use deterministic CPU SIMD fallback. CUDA/Vulkan/ANE kernels remain non-claims.
- Plugins: sixteen bundled plugins ship as compiled-in `mod.rs`/`stub.rs` pairs under `crates/abi-plugins/plugins/` with `assert_plugin_parity!`.
- AI/WDBX: the sibling WDBX crates are required build dependencies. Pure `abi_ai::complete` is store-independent; the `abi complete` and `abi agent train` CLI paths and MCP `ai_complete`, `ai_train`, and `ai_learn` attempt the configured/default durable store unless persistence is disabled with `ABI_WDBX_PERSIST=0` or `ABI_WDBX_PATH=:memory:`. Top-level `abi train` does not open WDBX. Successful writes record query/response vectors, metadata, and block-chain entries. Scheduler-backed completion, training, and agent helpers expose live task/memory observability.
- Abbey identity: Abbey is the primary empathetic-polymath profile, Aviva is the direct expert mode, and ABI is the orchestration/governance layer. The preserved Primary Declaration and the claim-honest Current/Partial/Proposed mapping live in `docs/spec/abbey-core-identity.mdx`; local profile output is deterministic template generation, not a model-quality or distributed-AI claim.
- WDBX: contract coverage in the required sibling workspace verifies ordered vector search results, block metadata round-tripping, segment/WAL recovery and compaction, temporal graph snapshot restore, and MCP hybrid ranking. Runtime persistence can be skipped or unavailable without fabricating a successful write.
- Connectors: Discord validates printable non-whitespace credentials, numeric snowflake-like IDs, and message size; Twilio validates account SID/auth-token shape, base URL, timeout, explicit `.live` transport, TwiML/form escaping, and ConversationRelay payload aliases before local/live dispatch.
- External collateral should not cite distributed sharding, AES/RBAC, Python/TensorFlow stacks, Kubernetes/H100 deployments, regulatory certifications, QPS/latency/accuracy, energy-efficiency, or model-benchmark claims unless a repo test, benchmark artifact, or documented source file proves them. The Apple `FoundationModels` Swift shim is an on-device bridge only (not a general Swift product stack). See [docs/contracts/external-claims-audit.mdx](docs/contracts/external-claims-audit.mdx).

See [docs/index.mdx](docs/index.mdx) for architecture, public API contracts, onboarding, and development guides, and [CHANGELOG.md](CHANGELOG.md) for release-note style modernization highlights.
