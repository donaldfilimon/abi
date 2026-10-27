# ABI Framework

> **Intelligence Without Limits — local runtime.**
> Rust 2024 orchestration, routing, and governance with explicit capability evidence.

ABI is the local AI runtime and integration framework behind the existing Abbey
and WDBX ecosystem. It provides deterministic profile routing, agent and tool
contracts, provider connectors, model-artifact foundations, and a CLI/MCP
interface. The repository targets **stable Rust 1.99.1**.

In the **LYLX by The Donald Company** architecture, ABI's role is **dynamic
moderation, routing, context selection, and authorized execution**. That design
direction does not imply that every moderation, model, or distributed feature
is already implemented in this repository.

[Quick start](#quick-start) · [Architecture](#architecture-and-project-roles) ·
[Local walkthrough](#local-walkthrough) · [MCP](#mcp) ·
[Current status](#current-status) · [Documentation](#documentation)

Project site: <https://donaldfilimon.github.io/abi/>. It is built from `site/`
through the GitHub Pages workflow when that source changes on `main`; its
benchmark dashboard uses **synthetic sample data, not measurements**. See
[the site README](site/README.md) and [brand guidance](docs/brand.md).

## Architecture and project roles

| Component | Responsibility | Boundary |
| --- | --- | --- |
| **LYLX** | New, ground-up intelligence platform by The Donald Company. | A distinct implementation program, not a silent rename or replacement of this repository. |
| **Quesar** | Designated largest shared foundation model in the LYLX family. | Model manifests, checkpoints, training runs, and evaluations must establish actual capabilities. |
| **Abbey / Aviva** | Consolidated Quesar personality direction: primary collaborator and candid expert mode. | Existing deterministic profiles and application integrations do not establish trained or consolidated Quesar weights. |
| **ABI** | Dynamic moderation/routing and execution governance. | Current executable behavior is defined by this repository's source, contracts, and tests. |
| **WDBX 3D** | Neural memory/storage direction, including spatial, semantic, graph, and provenance structures. | The existing sibling store has its own implemented features and conformance limits. |

```text
User / Abbey / CLI / MCP
           |
     ABI routing and governed host composition
           |                    |
     Model/provider         WDBX memory
           |
     Tool proposal -> validation -> authorization -> permitted execution
```

The diagram shows responsibility boundaries, not a claim that every command
uses a fully integrated production agent host. `abi-ai` remains store-independent;
CLI/MCP wrappers and `abi-sea` own memory integration. The local scheduler is
single-process, and worker-control contracts do not by themselves provide
remote execution.

The existing Abbey/Aviva/Abi profile names remain compatibility contracts.
Aviva's requested expressive latitude changes style, not permissions, privacy,
or the standard of evidence. Model output and retrieved text cannot grant
execution authority.

Related repositories: [Abbey](https://github.com/donaldfilimon/abbey) ·
[WDBX](https://github.com/donaldfilimon/wdbx) ·
[quesar.cloud website](https://github.com/donaldfilimon/quesar.cloud).
The website is not the Quesar model checkpoint. LYLX's ground-up direction
preserves these repositories and their historical data as references; it does
not authorize replacing their source or storage formats.

## Quick Start

### Prerequisites

ABI requires the sibling WDBX checkout, not just this repository:

```text
workspace/
  abi/
  wdbx/
```

`Cargo.toml` references
`../wdbx/crates/{abi-compute,abi-foundation,abi-core,abi-telemetry,abi-wdbx}`.
Missing or incompatible sibling sources prevent workspace resolution. For
qualification, use the exact `WDBX_REVISION` recorded in
[the CI workflow](.github/workflows/ci.yml), rather than assuming both repositories'
latest commits are compatible.

The full gate also needs Python 3.11+ and an available dependency-security
scanner. Follow [AGENTS.md](AGENTS.md) for the complete requirements. Use the
Cargo wrapper: a separately installed Cargo can ignore the pinned toolchain or
select incompatible linker tools.

```bash
./tools/cargo.sh --version
./tools/cargo.sh build --locked -p abi-cli
./tools/cargo.sh build --locked -p abi-mcp
./tools/check.sh
```

`./tools/check.sh` is the primary validation gate, not a statement that the
current checkout has passed. `./build.sh check` is its compatibility alias.
The retired Zig port is documented in [RUST-REWRITE-PLAN.md](RUST-REWRITE-PLAN.md);
it is history, not the active toolchain.

## Account and node enrollment research service

`abi-control-plane` adds a separate Rust account authority and browser client:
verified OIDC login, tenant-scoped metadata, locally confirmed node pairing,
signed application credentials, revocation/deletion, and an encrypted outbound
inference relay with per-account admission quotas. See the
[setup and protocol guide](crates/abi-control-plane/README.md). This is locally
tested research infrastructure. Two enrolled tenants execute isolated local
probe jobs over TCP. Opted-in durable jobs retain exact encrypted output on
the user node and replay it after local worker/service restart under current
authority, with verified client imports and no recovery inference. See the
[recovery workflow and limits](crates/abi-control-plane/RELAY.md). Public operation,
separate-host/power-loss qualification, useful trained weights, general governed
platform adapters and cross-platform acceptance remain open. The governed-tool
continuation adds durable local Abbey approvals and a scratch-file execution
assembly; see [its contract and remaining acceptance](docs/spec/durable-governed-tools.md). The CLI/MCP catalogs are unchanged.

## Cross-compilation

Install the intended Rust target and use
`./tools/cargo.sh build --locked -p abi-cli --target <triple>` for a smoke build.
Target compilation, live OS execution, accelerator operation, and production
qualification are separate results. Do not infer all-platform runtime support
from a successful cross-build or a CI configuration.

## Local Walkthrough

The following Bash walkthrough uses an in-memory store so it does not change
an operator's existing `~/.abi/wdbx` data:

```bash
./tools/cargo.sh build --locked -p abi-cli
ABI=./target/debug/abi
(
  export ABI_WDBX_PATH=:memory:
  "$ABI" backends
  "$ABI" scheduler status
  "$ABI" dashboard --once --plain
  "$ABI" complete "summarize ABI scheduler status"
  "$ABI" complete --neural "hello"  # char-LM demonstration, not a production LLM
  "$ABI" agent plan "stage a safe WDBX refactor"
  "$ABI" wdbx compute info
)
```

Select a disposable explicit store before testing persistence. Production-facing
completion commands otherwise attempt the configured store or `$HOME/.abi/wdbx`.
Use `ABI_WDBX_PERSIST=0` or `ABI_WDBX_PATH=:memory:` to opt out. Skipped,
unavailable, and failed persistence must be reported, not replaced with a
fabricated successful write.

### Useful commands and their scope

| Command | What it does / does not prove |
| --- | --- |
| `abi help --json [command] [subcommand]` | Typed command metadata for automation. |
| `abi help --completion <bash|zsh|fish>` | Metadata-driven shell completions. |
| `abi scheduler status` | A one-shot scheduler probe and task/memory counters, not durable background jobs. |
| `abi dashboard` / `abi tui` / `abi --tui` | Operational diagnostics. Non-TTY input or `--once` exits after one frame. |
| `abi agent plan`, `train`, `multi`, `spawn` | Local scheduler-backed helpers; names do not establish trained production weights or remote agents. |
| `abi agent tui` | Interactive REPL with `/status` telemetry and validated model IDs. |
| `abi agent browser` | Reviewed local plan and planner output; no embedded browser or automatic navigation. |
| `abi wdbx query <path> [text] [persona]` | Store statistics, hybrid retrieval, or persona-isolated retrieval. There is no `abi wdbx stats` command. |
| `abi wdbx secure demo` | Reference-scoped demonstration, not production cryptographic certification. |
| `abi wdbx cluster status` | Cluster-related local diagnostics, not production separate-host proof. |

Dashboard options include `--pane`, `--plain` / `--no-color`, `--compact`,
`--once`, `--interval` (100–60000 ms), `--json`, and `--list-panes`.
A real terminal enters refresh mode until `q` or Escape.

`abi complete --learn` adds SEA evidence retrieval, persistence, and routing
adaptation; it is not equivalent to foundation-model weight training. Specimen
Studio's Agent desk does not call this path or write this store. The separate
top-level `abi train` command inspects its requested configuration and does not
open WDBX.

### Explicit providers

```bash
# Local OpenAI-compatible bridge; unavailable bridges can fall back offline:
ABI_WDBX_PATH=:memory: ./target/debug/abi complete --model llama/phi3 "hello"

# Supported arm64 macOS host with Apple Intelligence ready:
# ABI_WDBX_PATH=:memory: ./target/debug/abi complete --live --model apple-fm --confirm "hello"
```

Model aliases are canonicalized; unknown IDs warn and pass through. Live
Anthropic-provider requests require the explicit live transport and configured
credentials. The selected route, actual provider, and any fallback must remain
visible. Never commit credentials to examples or logs.

## MCP

Build the server and use its launcher so binary and macOS FoundationModels
shim selection remain consistent:

```bash
./tools/cargo.sh build --locked -p abi-mcp
ABI_WDBX_PATH=:memory: ./mcp/launcher.sh stdio
```

Stdio JSON-RPC is the primary contract. The server also attempts a custom
loopback HTTP compatibility listener on `ABI_MCP_HTTP_PORT` (default 8080),
with optional `ABI_MCP_HTTP_TOKEN` bearer enforcement. `GET /sse` opens an MCP
2024-11-05 HTTP+SSE session; `POST /message` without a session ID retains the
one-shot compatibility response. **This is not Streamable HTTP.** A loopback
listener and optional token do not establish production network exposure.

The frozen tools are `ai_run`, `ai_complete`, `ai_train`, `ai_learn`,
`wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`,
`plugin_list`, `wdbx_stats`, and `plugin_run`. `connector_test` uses deterministic
local paths, not live network dispatch. See the
[public API contract](docs/contracts/public-api.mdx) for schemas and boundaries.

## Current Status

The [architecture specification](docs/spec/agent-wdbx-architecture.mdx) and
[external claims audit](docs/contracts/external-claims-audit.mdx) distinguish
implemented behavior, partial capabilities, and proposals.

| Area | Repository-supported boundary |
| --- | --- |
| Runtime and contracts | Rust 2024 / 1.99.1, local scheduling, CLI/MCP golden and contract tests. |
| Agent host | Validated requests, explicit policy/audit composition, bounded runs, and cooperative cancellation. |
| Capability kernel | Deny-by-default request-scoped authorization and recording fixtures; not a general production shell/filesystem actuator. |
| Model runtime | Exact authorized manifests and explicit devices; the tiny `abi-bigram-v1` fixture is infrastructure evidence, not a production Quesar or Gemma checkpoint. |
| Training foundation | Approved-record, eligibility, revocation, snapshot, and receipt contracts. These do not establish actual trained weights or model quality. |
| Workers | Signed audience-bound admission/control with finite leases, cooperative cancellation, replay resistance, private Unix control history, and bounded in-process inference through an explicitly supplied provider at startup; listeners, scheduling, original trained weights, and separate-host production operation remain unproved, and Windows private-log ACL validation refuses explicitly. |
| GPU/compute | Evidence-scoped native paths and deterministic CPU fallback; preserve `accelerated=false` when acceleration is unavailable. Detection does not establish speedup or ANE residency. |
| Plugins | Sixteen compiled-in `mod.rs`/`stub.rs` pairs with parity checks. |
| Memory | Required sibling WDBX provides persistence, recovery, vectors, blocks, temporal/spatial records, and bounded retrieval; refer to its current contracts for exact scope. |
| Connectors | Explicit local/live transports and validated credential/request shapes. Live-service results require separate evidence. |

Pure `abi_ai::complete` remains store-independent. CLI `abi complete` and
`abi agent train`, and MCP `ai_complete`, `ai_train`, and `ai_learn`, attempt
configured or default persistence unless disabled. Successful writes record
query/response vectors, metadata, and linked blocks; a failed store write is
not success.

Do not promote production multi-host sharding, universal encryption/RBAC,
regulatory certification, CUDA/Vulkan execution, model benchmarks, energy
savings, or throughput/latency claims without supporting artifacts. The
FoundationModels Swift shim is an on-device bridge, not a general Swift app
stack. Website films and synthetic dashboards are not benchmarks.

## Verification and contribution

Read [AGENTS.md](AGENTS.md) before changing the project. The full gate includes
policy tests, Rust/Python Abbey contract checks, a required dependency-security
scan, size limits, formatting, Clippy, builds/tests, sibling WDBX conformance,
applicable Metal checks, benchmark regression checks, rustdoc, and release
binaries. A skipped scanner or unavailable hardware test is not passing evidence.

For documentation-only changes, the repository guidance names:

```bash
python3 -m unittest discover -s tools/tests -p 'test_docs*.py'
git diff --check
```

Run the applicable policy suites as well; changes under `docs/` additionally
use the documented docs-validation workflow. Keep existing CI pauses and
operator decisions intact. Tests and smoke runs must never open live user
stores. Documentation edits do not qualify a new compiler, model, platform,
network service, or deployment.

## Documentation

- [Documentation index](docs/index.mdx)
- [Agent and WDBX architecture](docs/spec/agent-wdbx-architecture.mdx)
- [Abbey identity and operating contract](docs/spec/abbey-core-identity.mdx)
- [WDBX north-star and status mapping](docs/spec/wdbx-north-star.mdx)
- [Public API contract](docs/contracts/public-api.mdx)
- [External claims audit](docs/contracts/external-claims-audit.mdx)
- [Repository instructions](AGENTS.md), [expanded guidance](CLAUDE.md), and [changelog](CHANGELOG.md)
