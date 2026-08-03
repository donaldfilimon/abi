# ABI Threat Model

Status: current for the nightly-Rust workspace. This document is security
guidance, not a certification or production-hardening claim.

Source of truth: `Cargo.toml`, `crates/`, `tests/golden/`,
`docs/contracts/external-claims-audit.mdx`, and `./tools/check.sh`.

## Scope

ABI is a local-first AI and WDBX framework with:

- the `abi` CLI;
- the `abi-mcp` JSON-RPC server over stdio and optional loopback HTTP/SSE;
- WDBX durable local storage, REST, and cluster RPC;
- explicit live-provider and local-bridge connectors;
- dry-run and confirmed OS-control paths;
- optional native Metal kernels with deterministic CPU fallback.

The model does not claim hardened public-internet deployment, production
multi-host sharding, audited FHE, complete authorization, or a general AI
safety classifier.

## Assets

| Asset | Security property |
| --- | --- |
| Provider, Discord, and Twilio credentials | Confidentiality; never print or persist in WDBX completion metadata |
| `~/.abi/` WDBX store | Integrity, confidentiality, and exclusive writer ownership |
| Completion and SEA records | Integrity, provenance, bounded retention, and honest partial-write status |
| OS-control policy and audit records | Integrity, fail-closed parsing, and complete execution provenance |
| MCP/WDBX bearer and cluster tokens | Confidentiality and fixed-work comparison where implemented |
| Model/provider attribution | Integrity: identify the transport and engine that actually generated output |
| Golden CLI/MCP contracts | Integrity and compatibility |

## Trust boundaries

| Boundary | Current control | Residual risk |
| --- | --- | --- |
| CLI input -> local runtime | Typed parsing, length/path checks where defined | Local caller can request expensive work |
| MCP client -> stdio server | 64 KiB request cap, JSON depth bound, typed dispatch | Host process controls the pipe; no per-client identity |
| Loopback HTTP/SSE -> MCP | Loopback bind and optional bearer | No native TLS or scoped authorization |
| Loopback/routable client -> WDBX REST | Bind policy, optional bearer, token bucket | Shared bearer, DoS, and proxy misconfiguration |
| Cluster peer -> cluster RPC | Token for non-loopback, optional peer allowlist | Cleartext without proxy TLS; no dynamic membership protocol |
| ABI -> live provider | Explicit live mode, credential lookup, HTTPS validation | Provider data handling and remote compromise remain external |
| ABI -> local bridge | Explicit endpoint and bounded parsing | A local service is a separate trust domain |
| ABI -> OS process | Dry-run default, compiled ceiling, narrow-only policy, `--confirm` | Allowed commands still execute with user privileges |
| Tests -> WDBX | In-memory or counter-based scratch paths | An unisolated child process could otherwise resolve `~/.abi` |

## Primary threats and mitigations

### Live-store corruption or disclosure

Threats:

- tests opening `~/.abi/`;
- concurrent writers;
- torn WAL/checkpoint publication;
- partial completion records presented as fully persisted.

Controls:

- tests use `ABI_WDBX_PATH=:memory:`, `ABI_WDBX_PERSIST=0`, or
  `abi_foundation::temp_path::temp_file_path()`;
- `DurableStore` holds a writer lock for its lifetime;
- WAL frames and segment/checkpoint integrity are verified;
- store-open and write failures must be surfaced.

Residual:

- completion persistence currently consists of multiple mutations unless a
  transactional API explicitly groups them. Callers must report partial
  failure and must not equate some successful writes with an atomic commit.

### Credential leakage

Threats:

- secrets in logs, CLI output, WDBX metadata, inherited child environments, or
  malformed connector diagnostics.

Controls:

- credential backends are explicit;
- OS-control clears the environment and restores only non-sensitive variables;
- metadata stores identifiers and counts rather than raw provider secrets;
- connector errors must redact credentials.

Residual:

- local users with the same OS account may access user-owned files and process
  memory; ABI does not provide a separate privilege boundary.

### Network exposure

Threats:

- public binding without TLS;
- bearer theft;
- request flooding;
- untrusted MCP dispatch reaching connector-capable code.

Controls:

- MCP HTTP/SSE is loopback-oriented;
- WDBX REST defaults to loopback and has token/rate-limit controls;
- cluster RPC requires a token for non-loopback binds and can restrict peers;
- request/body/depth bounds reject oversized or malformed input.

Residual:

- shared tokens are not RBAC or per-client authorization;
- native TLS is not a blanket property of these listeners;
- deliberate non-loopback exposure requires a reviewed TLS-terminating proxy,
  token rotation, rate limits, monitoring, and explicit threat acceptance.

See `docs/spec/non-loopback-rest-threat-review.mdx` and
`docs/spec/cluster-mtls-ops.mdx`.

### OS command execution

Threats:

- arbitrary command execution;
- policy widening;
- secret inheritance;
- hung or output-blocked children;
- missing audit records.

Controls:

- dry-run is the default and does not execute;
- execute requires `--confirm`;
- the operator policy can narrow but cannot widen the compiled command ceiling;
- unknown, duplicate, and malformed policy keys fail closed;
- timeout and pipe-draining prevent indefinite or output-induced hangs;
- executed commands, including timeouts, are recorded to the configured WDBX
  audit store when available.

Residual:

- an allowed command runs with the invoking user's privileges;
- audit-store failure is disclosed but cannot retroactively prevent an already
  executed command.

### AI provenance and policy telemetry

Threats:

- attributing a local template to a remote model;
- treating local scheduler accounting as distributed agents;
- treating a lexical phrase scan as a complete safety decision;
- streaming unreviewed provider bytes before post-generation policy handling.

Controls:

- local output must identify its requested model separately from actual
  provider/transport and generation engine;
- local worker instructions and tool hints are metadata unless a real executor
  is connected;
- lexical constitution results are telemetry and must not independently
  authorize side effects;
- browser planning reports `embedded_browser=false`.

Residual:

- live and streaming paths require a common typed disposition and buffering if
  ABI is to claim pre-output policy enforcement. Until then, no such claim is
  permitted.

### Reference cryptography and acceleration

Threats:

- presenting demos as production encryption;
- presenting capability detection or CPU fallback as native acceleration.

Controls:

- WDBX homomorphic-encryption and learned-compression surfaces are
  reference-scoped;
- backend reports distinguish linked native kernels from deterministic CPU
  fallback;
- benchmark output is local and in-memory unless a reproducible artifact says
  otherwise.

Residual:

- no blanket AES-at-rest, RBAC, audited FHE, CUDA/Vulkan/ANE execution, or
  speedup claim is supported.

## Verification

Use the narrowest relevant checks, followed by the primary gate:

```bash
./tools/check.sh
.agents/skills/run-abi/smoke.sh
.agents/skills/mcp-smoke/smoke.sh
.agents/skills/sea-learn-loop/learn.sh
.agents/skills/wdbx-bench/bench.sh 50
```

Network, credential, platform, and live-provider claims require additional
evidence from the actual target environment. A green local gate is not proof of
public deployment security, Windows ACL behavior, or a successful remote model
call.

## Review triggers

Re-review this document when changing:

- listener bind/auth/TLS behavior;
- connector credentials or live transport;
- OS-control policy or execution;
- WDBX persistence, recovery, or locking;
- completion provenance, streaming, or constitution disposition;
- frozen CLI/MCP surfaces;
- security or performance claims.
