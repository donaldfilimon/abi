# Program 6: Abbey API and Application Federation

> **Constitutional mapping.** This file was written against the seven-program
> list Donald gave in conversation. The ratified constitution's section 13
> "Delivery programs" uses a different numbering, and the constitution wins.
> In section 13's terms this document specifies **Program 7, Application federation and production profiles.**
>
> The filename is therefore name-based rather than numbered, so no numbering is
> asserted. Nothing in section 13 was renumbered: section 15 reserves amendment
> to Donald, and the collision is raised as one request covering the whole set
> rather than five independent ones.


Status: **proposed design. No implementation is authorized by this document.**
Evidence level: **C0 (specified)** per the Abbey System Constitution section 11.

Author date: 2026-08-22.
Governing document: `docs/superpowers/specs/2026-08-22-abbey-system-constitution.md`.
Companion measurement: `docs/superpowers/specs/2026-08-22-wdbx-conformance-gap-analysis.md`.

Scope in Donald's words:

> Give the Rust bot, Swift companion/server, Abbey runtime, and future
> applications one stable authenticated protocol rather than merging every
> codebase or duplicating cognitive behavior.

## 0. Two conflicts recorded before anything else

### 0.1 Program numbering conflict with the ratified constitution

The constitution section 13 assigns:

- **Program 6: Model registry and adaptive arbiter.**
- **Program 7: Application federation and production profiles.**

The content of this document maps onto constitutional **Program 7**, not
Program 6. This document is filed under the number Donald used when
commissioning it. Constitution section 15 states that a program which needs to
contradict the constitution must amend it first, and that silence is not
amendment.

Recommended resolution, requiring Donald's explicit approval:

1. Amend section 13 so that the federation program carries the number used in
   practice, or
2. Retitle this document to Program 7 and leave the constitution unchanged.

Until one of those happens, treat "Program 6" in this filename as a working
label and "application federation and production profiles" as the
constitutional identity of the work. Do not silently renumber the constitution.

### 0.2 Program 1 owns the schemas; this program owns the wire

Constitution section 2 states that the Abbey contracts package "is a Program 1
deliverable. It may later become a repository. Until created and qualified, it
must not be documented as an existing repository or published package."

That line binds this document. The seam is:

| Owner | Deliverable |
| --- | --- |
| **Program 1** | Principal, scope, grant, approval, consent, capability, episode, receipt, error, and claim **schemas**, plus the cross-language fixture corpus. |
| **Program 6 (this document)** | **Transport, encoding, identity, authentication, versioning operation, error transport, degraded-operation obligations, fixture distribution mechanism, and deployment topology.** |

This document therefore names message *families* and their required fields
where transport correctness depends on them. It does not attempt to be the
normative schema. Where the two disagree once Program 1 lands, Program 1 wins
on field shape and this document wins on framing, negotiation, and identity.

## 1. Current state, verified by reading source

Everything in this section was read on 2026-08-22 from the paths given. Nothing
here is inferred from documentation prose alone unless labeled as such.

### 1.1 Repositories and toolchains

| Repository | Toolchain, verified file | Consequence |
| --- | --- | --- |
| `dev/active/abi` | `rust-toolchain.toml` → `nightly-2026-08-20` | Nightly required |
| `dev/active/wdbx` | `rust-toolchain.toml` → `nightly` | Nightly required |
| `dev/active/abbey` | `rust-toolchain.toml` → `nightly-2026-08-19` | Nightly required |
| `dev/active/abbey-bot` | `rust-toolchain.toml` → `1.97.1`, `profile = "minimal"` | **Pinned stable** |
| `dev/active/AbbeyBot` | Swift 6.4, macOS 27, Xcode toolchain for desktop | Swift, not Rust |

`wdbx/crates/abi-compute/src/lib.rs:1` is `#![feature(portable_simd)]`.
`abi/Cargo.toml:15` declares `abi-compute = { path = "../wdbx/crates/abi-compute" }`
and `abi/Cargo.toml:27` declares `abi-wdbx = { path = "../wdbx/crates/abi-wdbx" }`.

**Observation.** No single Rust toolchain compiles both `abbey-bot` and the ABI
or WDBX crate graph. This is not a preference. It is a compile-time fact
established by a nightly-only language feature on one side and a pinned stable
channel on the other. Any design whose sharing mechanism is a Rust crate,
trait, or dev-dependency excludes `abbey-bot`, which is the most developed
adapter, and excludes Swift entirely.

### 1.2 The precedent already in the tree

`wdbx/crates/abi-wdbx/tests/abbey_bot_projection_conformance.rs` exists and its
header states the constraint and the chosen remedy directly:

- `abbey-bot/tests/fixtures/wdbx_v1_conformance.seg.jsonl` is asserted
  byte-identical to its writer's output, inside `abbey-bot`, on stable.
- `wdbx/crates/abi-wdbx/tests/golden/abbey-bot-projection.seg.jsonl` is asserted
  parseable by `abi_wdbx::format::parse_segment`, inside `wdbx`, on nightly.
- The fixture is duplicated in two repositories **on purpose**, each side
  pinning its own copy, so divergence fails a test rather than surfacing in
  production.

This is the pattern this program generalizes. It is already proven in-tree for
one format. It is not yet generalized to a protocol, to four adapters, or to
Swift.

`abbey-bot/src/wdbx.rs` describes itself as a WDBX-v1 **projection**: no
segments, no manifest, no audit chain, unknown record types preserved verbatim,
no `# checksum:` trailer maintained. That self-description matters for section
7.4 below.

### 1.3 `abi-wdbx-gateway`, read in full

Crate at `abi/crates/abi-wdbx-gateway`. Proto at `proto/gateway.proto`, package
`abi.wdbx.gateway.v1`, service `WdbxGateway`, eight RPCs:

`PutVector`, `Search`, `PutKv`, `GetKv`, `ResolveConflict`, `Stats`,
`MembershipChange`, `WatchMutations` (server-streaming).

Verified properties:

- **Auth** (`src/auth.rs`): one `BearerToken` per listener. The token file is
  opened through `open_validated_file` with symlink, ownership, and mode
  validation, hashed with SHA-256, compared with `subtle::ConstantTimeEq`,
  zeroized on drop, and its `Debug` prints `BearerToken([REDACTED])`. Duplicate
  `authorization` metadata entries are rejected in `GatewayService::admit`.
- **Rate limiting** (`src/service.rs`): `rate: Arc<Mutex<RateWindow>>` is a
  **single process-wide one-second window** incremented on every `admit` call
  regardless of caller. There is no per-caller attribution because there is no
  per-caller identity: there is exactly one token.
- **Bounds** (`src/config.rs`, `struct Limits`): `message_bytes`,
  `batch_items`, `key_bytes`, `value_bytes`, `requests_per_second`,
  `blocking_jobs`, `concurrent_streams`, `membership_entries`, `event_queue`,
  `idle_seconds`, `search_limit`. Defaults include `message_bytes = 1 MiB` and
  `batch_items = 128`.
- **Events** (`src/events.rs`): `MutationNotice` carries only `sequence`,
  `kind`, `transaction_id`, `item_count`, `unix_ms`. Its doc comment states it
  never contains keys, values, or vectors. Same payload on both gRPC
  `WatchMutations` and the WebSocket `/v1/events` route.
- **TLS** (`src/tls.rs`, `src/server.rs`): non-loopback binding requires a
  server certificate and an owner-protected key; supplying a client CA makes
  client certificates mandatory on both listeners.

**What the gateway is not.** `abi/CLAUDE.md` records, and the proto confirms,
that this surface is not the CSAPS `MemoryService` surface: there is no
`ProposeWrite` write gate and no `Verify`. `PutVector` and `PutKv` write
directly through `StoreExecutor::run(|state| state.store.commit_export(...))`
with only shape validation ahead of them. The README further scopes
`MembershipChange` as gateway-local durable signed lineage, explicitly not
separate-host consensus.

### 1.4 `abi-mcp`, the frozen 12-tool surface

Crate at `abi/crates/abi-mcp`. `src/handlers.rs` declares `const TOOLS` in
contract order, not alphabetical order:

`ai_run`, `ai_complete`, `ai_learn`, `ai_train`, `wdbx_query`,
`scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`,
`plugin_list`, `wdbx_stats`, `plugin_run`.

Verified properties:

- `handle_initialize` pins `protocolVersion: "2024-11-05"`, `serverInfo.name:
  "abi-mcp"`, `serverInfo.version: "0.2.0"`.
- `tools/list` output is asserted byte-for-byte against
  `tests/golden/mcp-tools-list.json`, **property order included**, per the
  module doc comment on `handlers.rs`.
- `src/stdio.rs` is line-delimited JSON-RPC with a byte-by-byte reader that
  drops an overlong line with a `-32700 Parse error` before it can exceed
  `protocol::MAX_REQUEST_SIZE`. **stdio is tokenless.**
- `src/protocol.rs` bounds nesting at `MAX_JSON_DEPTH = 32` and shares
  `MAX_REQUEST_SIZE` with the HTTP path.
- `src/http.rs` self-documents as a bounded loopback compatibility endpoint and
  explicitly disclaims being a conforming persistent MCP HTTP+SSE channel.
- `ToolError::message` returns fixed, non-leaking strings; every `tools/call`
  failure reports JSON-RPC `-32603` and only the message varies.

### 1.5 `abbey`, the product runtime host

`abbey/src/daemon/` already implements an authenticated bounded local
control plane for `abbeyd`:

- `protocol.rs` defines `RequestEnvelope { version, request_id, bearer, command }`
  and a **separate** `V3RequestEnvelope { version, schema_version, request_id,
  bearer, grants: V3CapabilitySet, command }`. Both derive
  `#[serde(deny_unknown_fields)]`. `Debug` redacts `bearer`.
- `ResponsePayload` is `#[serde(tag = "outcome", rename_all = "snake_case",
  deny_unknown_fields)]` with `Ok { event }` / `Error { code, message }`.
- `PROTOCOL_VERSION`, `CURRENT_PROTOCOL_VERSION`, and `SUPPORTED_PROTOCOL_VERSIONS`
  exist; the module doc states protocol v3 is deliberately a separate envelope
  so that a v3 command cannot be decoded as a legacy `AppCommand` and the exact
  v1/v2 wire fixture is preserved.
- `app_core/v3.rs` has `V3CapabilitySet::deny_all()` and `from_sorted`, which
  accepts a grant set **only when its supplied order is canonical**.
- `config.rs` defines `DaemonConfig { socket_path, bearer: BearerSecret,
  max_frame_len, read_timeout, write_timeout, accept_poll_interval,
  authenticated_rate_limit }` with `AuthenticatedRateLimit` defaulting to 64
  requests per 1 second. `server.rs` serves over `#[cfg(unix)]` Unix sockets.
- The module doc states requests never choose a program, argument recipe,
  environment, workspace, or memory backend, and that an approved effect needs
  an identical explicit resubmission after approval.

**Assessment.** This is already the closest thing on disk to the Abbey API:
JSON envelopes, a bearer, a request id, a version, a separately versioned
capability grant set, canonical ordering, bounded frames, deny-by-default
grants, and Unix-socket-only transport. Program 6 should generalize and
version this rather than invent a parallel mechanism. `abbey` is a nightly
crate, so its Rust types cannot be shared with `abbey-bot`; only its **wire
shape** can be.

### 1.6 `AbbeyBot`, the Swift adapter

From `AbbeyBot/AGENTS.md`, verified 2026-08-22:

- Products: `AbbeyBot` desktop (SwiftUI + SwiftData), `AbbeyServer` (Vapor +
  Fluent, Leaf status page, `/api/*`, `/dashboard` SPA), `abbey` CLI, all
  dispatched from one `AbbeyEntry` `@main`.
- Discord via **DiscordBM**; Twitch via IRC and EventSub with HMAC-SHA256
  signature verification and a Fluent-backed replay guard.
- Gate of record: `bash Scripts/verify-all.sh`. GitHub Actions is secondary and
  currently fails at startup with 0 jobs pending a billing unblock.
- Toolchain fragility that is load-bearing for this design: desktop **requires**
  the Xcode toolchain because swiftly DEVELOPMENT-SNAPSHOT breaks SwiftData
  macros; the server graph is separately checked under a snapshot toolchain
  (`Scripts/check-server-snapshot.sh`) and cross-compiled to Linux against a
  **musl** Static Linux SDK (`Scripts/check-server-linux.sh`) which ships
  neither `Testing` nor `XCTest`, so that path cannot compile test targets at
  all. A `glibc-divergence` static lint exists because musl and glibc disagree
  on `extern FILE *stderr` mutability under Swift 6 strict concurrency.
- **Auth today:** `ABBEY_API_TOKEN` is **optional**. When unset, the `/api/*`
  list and state routes are open, by design, so the local SPA tabs work without
  configuration. When set it is accepted as `Bearer` or `X-Abbey-Token` and
  gates those routes uniformly.
- Personas `AbbeyPersona`, `AvivaPersona`, `AbiPersona` live in
  `Sources/AbbeyCore/Personas/`, and `ABIRouter.route(intent:)` selects among
  them locally.

### 1.7 `abbey-bot`, the Rust Discord adapter

Single binary crate, pinned stable 1.97.1, serenity plus poise plus songbird.
Modules relevant here: `src/provider.rs`, `src/generation/`,
`src/provider_self_test.rs`, `src/wdbx/`, `src/memory/`, `src/persona.rs`,
`src/voice_session/`, `src/commands_voice.rs`, `src/perms.rs`, `src/guild.rs`,
`src/wyhash.rs`, `src/embedding.rs`.

`src/wdbx.rs` states the module takes no dependency on abi and that
`wyhash.rs`/`embedding.rs`/persona transcriptions are pinned by golden tests so
stores stay bit-compatible. Deduplicating them against ABI would reverse a
documented, tested decision.

### 1.8 Current-state gaps this program must not paper over

| Gap | Where | Consequence for federation |
| --- | --- | --- |
| One bearer token per gateway listener, no per-caller identity | `abi-wdbx-gateway/src/auth.rs`, `src/service.rs` | No adapter attribution, no per-adapter ceiling |
| One process-wide rate window shared by all callers | `abi-wdbx-gateway/src/service.rs` | One noisy adapter starves every other adapter; incompatible with constitution section 8 per-guild budgets |
| No write gate on the substrate transport | `proto/gateway.proto` has no `ProposeWrite`, no `Verify` | Retention classing and the selective write gate have nowhere to live on that path |
| `ABBEY_API_TOKEN` optional, routes open when unset | `AbbeyBot` Vapor `/api/*` | An unauthenticated local surface exists beside the authenticated one |
| MCP stdio is tokenless | `abi-mcp/src/stdio.rs` | Acceptable for a stdio-child developer tool, unacceptable as an adapter federation channel |
| `deny_unknown_fields` on every `abbey` daemon envelope | `abbey/src/daemon/protocol.rs` | Forecloses additive evolution unless a policy is chosen deliberately |
| Digest is `serde_json` over declaration order, parents unsorted, no episode signature | `wdbx/crates/abi-wdbx/src/versioned.rs:499` per the gap analysis section 6.4 | Any API field that carries a digest must not imply the digest is currently canonical |

## 2. Proposed: what the Abbey API is, and what it is not

**Proposed.** The Abbey API is one versioned, authenticated, bounded
request/response and event contract, expressed as **framed JSON over an
authenticated local IPC channel by default**, whose normative artifacts are
JSON Schema documents plus a golden fixture corpus, vendored by each adapter
and gated by a per-adapter conformance test.

Non-goals, stated so a reader does not assume them:

1. **Program 6 does not modify `abi/crates/abi-wdbx-gateway/proto/gateway.proto`
   and does not expose its eight RPCs to adapters.** The gateway remains a
   substrate transport reachable only by the canonical owner of durable memory.
   No adapter calls `PutVector`, `PutKv`, `Search`, or `MembershipChange`.
2. **Program 6 does not extend the MCP surface past 12 tools.** `tools/list` is
   golden-tested byte-for-byte including property order, and stdio is
   tokenless. MCP is a developer and agent tool surface, cited here as the
   compatibility discipline to imitate, not a surface to grow.
3. **AbbeyBot's `/api/*` Vapor surface stays AbbeyBot-local.** It is the
   operator UI and desktop sync channel. It is not the federation contract and
   is not renamed to look like one.
4. Program 6 does not define episode field semantics, evidence dimensions, or
   retention classes. Those are Program 1 schemas and Program 4 behavior.
5. Program 6 does not merge repositories and does not create a monorepo.
   Constitution section 2 rejected both.

## 3. Transport and encoding

### 3.1 The discriminating constraint

The choice is not JSON versus Protocol Buffers on general merit. The
discriminating question is: **does the contract force a code-generation step
into an adapter's gate of record?**

For `abbey-bot`: `prost`/`tonic` compile on stable 1.97.1, so protobuf is
technically available there. It would add `protoc` (vendored or system) and a
`build.rs` to a crate that currently has neither.

For `AbbeyBot`: protobuf means SwiftProtobuf plus, for streaming, grpc-swift,
plus their SwiftPM plugin codegen, added to a package whose build graph is
already the most fragile artifact in the federation. That package must satisfy
simultaneously: an Xcode toolchain for the SwiftData desktop target, a swiftly
snapshot toolchain for the server graph, and a musl Static Linux SDK that ships
no `Testing` or `XCTest` module. Its CI is billing-blocked and its gate of
record is a local script. Adding two codegen-bearing dependencies with plugin
build phases to that graph is a real and asymmetric cost.

JSON via `Codable` adds **nothing** to the Swift build graph. It is in the
standard library. On the Rust side both `abbey-bot` and `abbey` already depend
on `serde_json`. `abbey/src/daemon/protocol.rs` already speaks framed JSON
envelopes with `Serialize`/`Deserialize`.

**Decision.** The Abbey API wire encoding is **UTF-8 JSON**, one JSON object
per frame, with an explicit length or newline framing bound (section 3.3).

Secondary reasons, subordinate to the one above:

- Adapters are few, local, and low-rate. The default topology is a Unix socket
  on one machine. Byte efficiency is not the binding constraint; auditability
  is. A JSON frame can be logged, redacted, diffed in a fixture, and read by a
  human reviewing an authorization decision.
- The fixture corpus is the compatibility mechanism (section 6). Text fixtures
  can be reviewed in a pull request. Binary fixtures cannot.
- `abi-mcp` already demonstrates in-tree that a JSON-RPC surface can be frozen
  to byte-for-byte property order and defended by golden tests.

### 3.2 CSAPS section 6.4 does not apply to transport

The WDBX conformance gap analysis section 6.4 records that CSAPS "explicitly
rejects deterministic Protocol Buffers as a durable canonical representation
across schemas, builds, languages, and library versions," and notes
`serde_json` is weaker still because field order follows struct declaration
order and there is no canonical number or string form.

That finding is about **durable signed records**: the input to
`d_t = SHA256(c_t)` and `sigma_t = Sign(d_t)`. It is not about transport
encoding, and the two must not be conflated.

**Proposed invariant T1.** The Abbey API wire encoding is never the input to a
durable digest. An episode's canonical encoding is a separate,
Program-1-and-Program-4-owned form (CSAPS specifies canonical CBOR). Where the
API carries a digest, it carries it as an opaque, already-computed value
alongside an explicit `canonical_encoding` discriminator naming the form the
digest was taken over. Changing the wire encoding must never change a digest.

**Proposed invariant T2.** Adapters may **carry and echo** an episode digest and
signer identity. Adapters must never **compute** one, and must never treat a
digest they verified as evidence that the content is true. If an adapter can
mint a digest it can mint an episode, which violates constitution section 5's
"integrity is not truth," decision 26, and decision 73's single canonical
writer per domain. Digest computation is a canonical-owner operation only.

**Note on current honesty.** Per the gap analysis, today's `audit_hash` takes
`serde_json` bytes in declaration order and does not apply CSAPS's
`sort(parent_hashes)`, and `V2AuditBlock` carries no `signature` or
`signer_key_id`. Therefore the API's `digest` and `signer_key_id` fields must be
**optional and explicitly labeled non-canonical** until Program 4 replaces the
commitment function. Carrying them as if they were canonical would be a false
Current claim.

### 3.3 Framing, bounds, and streaming

**Proposed.**

- Default channel: `SOCK_STREAM` Unix domain socket, owner-only mode, path
  under the host's own state root. This matches `abbey`'s existing
  `DaemonConfig::socket_path` and constitution decision 74.
- Framing: a 4-byte big-endian unsigned length prefix followed by exactly that
  many bytes of UTF-8 JSON. Length-prefixed rather than newline-delimited
  because streaming events and large proposals should not be constrained to
  escape-free single lines, and because a length prefix makes the bound
  checkable before allocation, which is the property `abi-mcp/src/stdio.rs`
  achieves the hard way by reading byte-by-byte.
- `max_frame_bytes` is a required configured bound, defaulting to 1 MiB, which
  matches the gateway's `Limits::message_bytes` default. A frame declaring a
  larger length is rejected before any read of the body, and the connection is
  closed. Adopt `abi-mcp`'s `MAX_JSON_DEPTH = 32` nesting bound verbatim.
- Streaming: server-to-client streams are a sequence of frames sharing the
  request's `correlation_id`, each carrying a monotonic `sequence`, terminated
  by exactly one terminal frame (`complete`, `cancelled`, or `error`). This
  mirrors the existing `MutationNotice.sequence` discipline. Concurrent streams
  per connection are bounded, as `EventHub::try_stream_permit` already does.
- Ordering: responses may interleave. Every frame carries `correlation_id`.
  Frames on one stream are strictly ordered by `sequence`.

### 3.4 The non-loopback profile

**Proposed.** Non-loopback operation is a separately qualified deployment
profile, per constitution decision 75. It is the same JSON contract carried over
TLS, with:

- mandatory server certificate and owner-protected key, and mandatory client
  certificates via a configured client CA. `abi-wdbx-gateway/src/tls.rs`
  already implements exactly this shape for the substrate transport and is the
  reference implementation to copy, not to reuse in place.
- an explicit origin allowlist, matching the gateway's `allowed_origins`.
- separate qualification evidence per host profile. A local-Mac qualification
  never promotes a server profile.

Until that profile is qualified, non-loopback is refused at configuration
validation time, not at request time.

## 4. Service surface

**Proposed.** Namespace `abbey.v1`. Eight method families. Each entry states
where it maps today.

| Method | Direction | Maps to Current | Notes |
| --- | --- | --- | --- |
| `Hello` | request | new | Version negotiation, adapter identity assertion, contract digest exchange. Must be the first frame on a connection. |
| `Authorize` | request | new; nothing in `abi-wdbx-gateway` or `abi-mcp` performs typed authorization | Returns exactly one of `allow`, `approval_required`, `deny`, `pause`, per constitution section 3. |
| `Cognize` | request, optionally streaming | conceptually `abi-mcp` `ai_run` / `ai_complete` / `ai_learn`; those stay MCP-only | Produces user-facing content **or** a typed proposal, never a side effect. |
| `Propose` / `Execute` | request pair | partially `abbey` daemon v3 prepared-intent then identical resubmission | `Execute` requires a current grant plus revalidated preconditions. Recommend, propose, execute are three visible stages. |
| `Retrieve` | request | backed today by `abi-wdbx-gateway` `Search`, reached only by the canonical owner | SEA-bounded evidence selection. Returns evidence dimensions separately, never one collapsed score. |
| `ProposeWrite` | request | **no counterpart exists**; the gateway writes through `commit_export` with no gate | The selective write gate and retention classing live here. Program 4 implements the semantics. |
| `Capabilities` family | request | partially `abi-mcp` `plugin_list`; not equivalent | `List`, `Describe`, `PreviewManifest`, `ApplyManifest`. Manifest apply is previewed, approved, hashed, reversible. |
| `Consent` family | request | `abbey-bot/src/voice_session/` owns epochs today, entirely locally | `OpenEpoch`, `AttestParticipants`, `CloseEpoch`. Content-free. See section 5. |
| `WatchEvents` | server stream | `abi-wdbx-gateway` `WatchMutations` and `/v1/events` | Metadata-only, same discipline as `MutationNotice`: no keys, values, vectors, transcripts, or message content. |

### 4.1 Relationship to the frozen MCP 12

The MCP surface and the Abbey API are **different surfaces with different
audiences**, and this is deliberate.

| Aspect | `abi-mcp` | Abbey API |
| --- | --- | --- |
| Audience | Model clients and developer tooling | Product adapters |
| Transport | stdio JSON-RPC (tokenless) plus bounded loopback HTTP | Authenticated framed JSON over Unix socket |
| Authorization | Field validation only; no principal | Typed principal, delegation chain, capability grant |
| Stability mechanism | Byte-frozen golden `tools/list` | Versioned negotiation plus fixture corpus |
| Growth | **Frozen at 12** | Additive under section 7 |

`ai_run`, `ai_complete`, and `ai_learn` are conceptual ancestors of `Cognize`,
and `wdbx_query` of `Retrieve`. They are not migrated, wrapped, or deprecated by
this program. An `abbey.v1` implementation may internally reach the same ABI
code paths those tools reach. It must not reach them **through** MCP, because
MCP carries no principal and its stdio channel carries no credential.

### 4.2 Relationship to `abi-wdbx-gateway`

The gateway keeps its eight RPCs and its `abi.wdbx.gateway.v1` package
unchanged. Position in the topology:

```text
Discord / macOS / CLI / future adapter
        |  abbey.v1 over authenticated Unix socket   (Abbey API, northbound)
        v
Abbey product runtime host  (dev/active/abbey)
        |  abbey.v1                                   (Abbey API, southbound)
        v
ABI authorization and cognition  (dev/active/abi)
        |  abi.wdbx.gateway.v1 (protobuf/gRPC) or in-process
        v
canonical WDBX  (dev/active/wdbx)
```

**Proposed invariant H1 (host-as-proxy).** The runtime host speaks the same
`abbey.v1` contract northbound and southbound. It is a policy-enforcing proxy,
not a translator. On forwarding it may **add** authorization context: the
authenticated adapter principal, the channel identity, the delegation chain
link, and resource accounting. It may **never remove, rewrite, or downgrade**
context supplied by an authenticated caller, and it may never widen a grant.
This is the exact seam where "an adapter silently redefines authorization"
would creep in, so it is stated as an invariant and gets its own conformance
fixture.

This resolves the apparent tension in constitution section 10, which shows the
adapter-to-host hop as "authenticated Unix socket or owner-scoped loopback" and
the host-to-ABI hop as "versioned Abbey API." One contract, two hops, one
enforcement point per hop.

## 5. Cancellation, deadlines, and consent-epoch propagation

Constitution section 3 requires that closing a voice media epoch "cancels
in-flight STT, reasoning, synthesis, provider work, and playback," and section
10 requires that "ABI or provider failure must not weaken the media gate." An
error taxonomy alone cannot deliver that. The propagation contract is normative.

### 5.1 Required request fields

**Proposed.** Every `abbey.v1` request frame carries:

- `correlation_id`: unique per request, echoed on every response and stream
  frame.
- `idempotency_key`: caller-minted, stable across retries of the *same
  intended effect*. Required on `Execute` and `ProposeWrite`. Optional
  elsewhere.
- `deadline_unix_ms`: absolute, not a relative timeout, so it survives hops
  without accumulating skew-per-hop. A receiver that cannot complete before the
  deadline must fail fast with `deadline_exceeded` rather than starting work.
- `consent_epoch`: present when and only when the request is bound to an open
  voice media epoch. Carries the epoch id and the epoch's participant-set
  digest. Never the participant identities.
- `cancel_token`: server-assigned on first response, usable in `Cancel`.

### 5.2 Propagation rules

**Proposed.**

1. **Cancellation is a first-class frame, not a connection drop.** `Cancel`
   carries `correlation_id` and travels adapter to host to ABI to provider
   adapter. Each hop must forward within its own bounded budget and must not
   wait for the downstream hop to acknowledge before ceasing its own work.
2. **Consent-epoch closure cancels by class, not by enumeration.**
   `Consent.CloseEpoch` cancels every in-flight request bearing that
   `consent_epoch`, without the closer needing to know their correlation ids.
   This is required because a new, unidentified, or unattested participant must
   close media *immediately* and cannot be made to wait on a cancellation
   inventory.
3. **Connection loss is treated as cancellation of every in-flight request on
   that connection**, plus closure of every consent epoch opened on it. Failing
   open here would leave a media epoch alive with no supervising adapter.
4. **Deadline expiry is cancellation.** A hop whose `deadline_unix_ms` passes
   cancels downstream and returns `deadline_exceeded`.
5. **Cancellation never rolls back an already-committed platform effect.** It
   stops further work. What actually happened is reported by the receipt, not
   inferred from the cancellation.
6. **Barge-in is cancellation, not consent withdrawal.** Per constitution
   section 3, a barge-in cancels active playback and stale downstream work and
   does not by itself close the epoch. The API models these as two distinct
   calls precisely so an adapter cannot conflate them.

### 5.3 Cancellation racing a side effect

**Proposed.** When `Cancel` arrives after an actuator call has been dispatched
but before its outcome is known, the response is `cancellation_raced`, not
`cancelled`, and the accompanying receipt must enumerate, per step:
`completed`, `reverted`, or `unresolved`. It must not report a clean
cancellation. This is the transport-level expression of constitution section
10's rule that an incomplete rollback identifies completed, reverted, and
unresolved steps without exposing private content.

### 5.4 Idempotency and replay

**Proposed.**

- A repeated `Execute` with the same `idempotency_key` and the same request
  digest returns the **original** outcome and receipt. It does not re-execute.
- A repeated `Execute` with the same `idempotency_key` and a **different**
  request digest is `idempotency_conflict` and is refused. This prevents an
  approval obtained for one action being replayed against another.
- An approval reference is single-use and bound to `(idempotency_key, request
  digest, capability version, policy version)`. Constitution decision 16:
  repeated approval does not become standing authority.

## 6. Authentication and per-adapter identity

### 6.1 Current

One shared bearer per listener in `abi-wdbx-gateway`; one `BearerSecret` in
`abbey`'s `DaemonConfig`; tokenless MCP stdio; optional `ABBEY_API_TOKEN` in
AbbeyBot's Vapor routes. No surface on disk distinguishes *which* caller
presented a credential.

### 6.2 Proposed: two principals per request

**Proposed.** Every `abbey.v1` request carries two distinct principals and they
are never merged:

- **Channel principal (adapter identity).** Authenticates the *process*.
  Established at `Hello` and pinned for the connection's lifetime. Answers
  "which adapter is speaking."
- **Subject principal (human or platform actor) with delegation chain.**
  Answers "on whose behalf." Supplied per request. May be absent for
  adapter-internal maintenance calls, in which case no consequential capability
  is reachable.

**Proposed invariant A1.** The effective authority of a request is the
**intersection** of the channel principal's ceiling and the subject principal's
grants, evaluated deny-by-default. An adapter can never exceed its own ceiling
by presenting a more privileged subject, and a privileged subject can never
exceed the adapter's ceiling by choosing a weaker channel. Constitution
decisions 9, 10, and 11.

### 6.3 Credential material

**Proposed.**

- Each adapter identity gets its **own** credential file, not a shared one.
  Reuse `abi-wdbx-gateway/src/auth.rs`'s discipline verbatim: symlink,
  ownership, and mode validation via `open_validated_file`; SHA-256 digest held
  in memory; `subtle::ConstantTimeEq` comparison; `zeroize` on drop; `Debug`
  prints `[REDACTED]`. That code is the reference behavior, reimplemented per
  repository rather than shared as a crate, for the toolchain reason in 1.1.
- Rotation is per adapter identity and does not require restarting other
  adapters. A rotated-out credential is refused immediately; there is no grace
  window, because constitution decision 17 requires revocation to take effect
  before new work begins.
- On the non-loopback profile the channel principal is additionally bound to
  the client certificate subject, and a mismatch between certificate subject
  and asserted adapter identity is `unauthenticated`, not a warning.

### 6.4 Per-identity accounting

**Proposed.** Rate limits, concurrency limits, and budgets are keyed by
`(channel principal, guild scope)`, not by process. This is a direct correction
of the Current single-window limiter in
`abi-wdbx-gateway/src/service.rs`, and it is what makes constitution section
8's "separate budgets for speech, observation, planning, APIs, commands, and
changes" expressible at all. Exhaustion of one adapter's budget must not
produce `resource_exhausted` for another adapter.

Per-identity counters are metadata-only: identity, class, count, window. They
carry no request content and are subject to the same redaction rules as
`MutationNotice`.

### 6.5 What does not get a credential

MCP stdio stays tokenless and stays out of the federation path. A stdio child
process inherits its parent's trust by construction; adding a token there would
be theater. The correct statement is the one already in `abi/CLAUDE.md`: stdio
is the primary MCP transport and it is tokenless, therefore it is not an
adapter channel.

## 7. Versioning and compatibility policy

### 7.1 Three independent version axes

**Proposed.** Do not collapse these.

| Axis | Identifier | Changes when |
| --- | --- | --- |
| **Contract major** | `abbey.v1`, `abbey.v2` | A breaking change lands |
| **Contract revision** | `contract_revision`, monotonic integer | An additive change lands within a major |
| **Capability version** | per capability package semver | A capability's own schema, risk class, or policy changes |

A capability version drift is **not** an API version change and must not be
reported as one. Constitution decision 46: schema drift disables the affected
capability version and preserves the last approved version. It does not
renegotiate the transport.

### 7.2 Negotiation

**Proposed.** `Hello` is mandatory and first. It carries, from the client:
supported contract majors, highest known `contract_revision`, adapter identity,
adapter build identity, and the digest of the fixture corpus the adapter
vendored (section 8). The server replies with the selected major, its own
`contract_revision`, its corpus digest, and the capability set the channel
principal may reach.

Outcomes:

- **No common major** is `unsupported_contract_version` and the connection
  closes. There is no downgrade-by-guessing.
- **Corpus digest mismatch** is a **warning by default and a refusal under a
  strict profile.** It is a warning by default because an adapter legitimately
  lags a purely additive revision. It is a refusal under strict because
  constitution decision 79 makes shared fixtures the proof of cross-repository
  compatibility, and a mismatched corpus means that proof does not exist for
  this pairing.

### 7.3 Unknown-field policy, per message class

Constitution section 12 requires that unknown additive fields "round-trip **or**
fail according to the compatibility policy." Strict rejection is therefore a
legitimate policy, not a violation. The problem with a blanket strict rule is
that it forecloses additive evolution; the problem with a blanket tolerant rule
is that it lets an unknown field smuggle semantics past a security decision.

**Proposed: mixed policy by class.**

| Class | Policy | Rationale |
| --- | --- | --- |
| Grants, approvals, consent envelopes, capability manifests, principals, delegation chains | **Strict reject** on unknown fields | An unknown field here could carry authority the receiver does not evaluate. Fail closed. |
| Cognition content, proposals, receipts, evidence references, telemetry, events | **Tolerant round-trip**: preserve unknown fields verbatim on re-emission, ignore them for decisions | Allows additive evolution and lets an older intermediary forward a newer payload without lossy rewriting. |
| Digest-bearing envelopes | **Strict reject**, and unknown fields never enter a digest input | Prevents a tolerant path from silently altering what a digest covers. |

Impact on existing `abbey` code, stated so the migration is not a surprise:

- `RequestEnvelope` and `V3RequestEnvelope` carry `bearer` and `grants` and are
  correctly `deny_unknown_fields` today. **Keep strict.**
- `ResponsePayload::Ok { event }` currently inherits `deny_unknown_fields`.
  Under this policy the **event body** moves to tolerant round-trip while the
  outcome discriminator stays strict. That is a real, deliberate change to
  `abbey/src/daemon/protocol.rs` and it must be made through a
  `contract_revision` bump with a fixture, not incidentally.
- The v1/v2 legacy envelope fixture that `abbey`'s module doc says is
  deliberately preserved stays byte-exact. New behavior lands on a new
  envelope, exactly as v3 already did.

### 7.4 How a breaking change is introduced

**Proposed.** A breaking change is any of: removing a field, narrowing a field's
type or range, changing a field's meaning, changing an enum member's meaning,
making an optional field required, or changing a default that alters an
authorization outcome.

Procedure, in order, none skippable:

1. **Write the divergent prediction.** State what the old contract would do and
   what the new one will do for at least one fixture that distinguishes them.
   Constitution decision 65: thresholds and predictions before results.
2. **Open `abbey.v2` in the schema set, additively, alongside `abbey.v1`.** Both
   majors exist in the corpus simultaneously. `v1` fixtures are frozen at this
   point and never edited again.
3. **Serve both majors from the canonical owner.** `Hello` negotiation selects.
   Dual-serving is mandatory; there is no flag day.
4. **Migrate adapters one at a time**, each landing its own vendored corpus
   update and its own conformance-gate pass, on its own repository's schedule.
   This is the entire point of the federation: `abbey-bot` on stable, AbbeyBot
   on Swift 6.4, and `abbey` on nightly migrate independently.
5. **Deprecation window** begins only after every adapter in the current
   compatibility matrix reports `v2` at `Hello`. The window is recorded with a
   date, not "eventually."
6. **Retire `v1`** by refusing it at `Hello`, keeping its fixtures in the corpus
   permanently as historical evidence.

**Proposed invariant V1.** A single wire shape is never reinterpreted. If the
meaning changes, the field name or the message name changes. This is the rule
`abbey`'s v3 envelope already follows by being a separate type rather than an
extended `AppCommand`, and it is why that decision should be generalized rather
than treated as a one-off.

## 8. Sharing schemas across Rust and Swift without a repository merger

### 8.1 Mechanism: one normative corpus, N vendored copies, digest equality

**Proposed.** Source of truth is a directory in `dev/active/abi`, the canonical
governance runtime:

```text
abi/contracts/abbey/v1/
  schema/*.schema.json      JSON Schema draft 2020-12, one per message family
  fixtures/**/*.json        the golden corpus: valid, invalid, boundary, adversarial
  CORPUS_DIGEST             a single recorded digest over the corpus
  COMPATIBILITY.md          the section 7 policy, restated normatively
```

`abi/docs/contracts/` already exists and holds `external-claims-audit.mdx` and
`public-api.mdx`, so this is an addition beside an established location, not a
new convention.

Each adapter **vendors** a copy and asserts digest equality against the recorded
value, plus encode and decode round-trips against every fixture. Nobody
submodules. Nobody publishes a package. Nobody depends on a Rust crate across
the toolchain line.

This is exactly `abbey_bot_projection_conformance.rs` generalized. That test
already proves the pattern works across the stable/nightly boundary for a file
format. The two changes are that the corpus becomes a **hub** rather than a
pairwise pair, and that Swift joins.

### 8.2 Why a hub, not pairwise fixtures

The existing arrangement is two copies asserted independently against each
other's behavior. With four adapters, pairwise is six relationships and six
places for a two-way drift to hide. With a hub it is four relationships, each
of which is "does my copy match the recorded digest, and do I round-trip every
fixture." A drift is then attributable to exactly one adapter.

### 8.3 Where each conformance gate lands

**Proposed.** Named concretely, because constitution decision 79 makes shared
fixtures the proof and an unnamed gate leaves the requirement aspirational.

| Repository | Vendored path | Test | Runs under |
| --- | --- | --- | --- |
| `dev/active/abi` | `contracts/abbey/v1/` (the source) | new integration test in the crate that implements the surface | `./tools/check.sh` |
| `dev/active/abbey` | `contracts/abbey/v1/` | new integration test under `tests/` | `abbey`'s own `check.sh` |
| `dev/active/abbey-bot` | `tests/fixtures/abbey-api/v1/` beside the existing `wdbx_v1_conformance.seg.jsonl` | new module test using `include_str!`, matching the existing fixture idiom in `src/wdbx/tests.rs` and `src/wyhash.rs` | the crate's existing test gate, stable 1.97.1, no new dependency beyond `serde_json` |
| `dev/active/AbbeyBot` | `Tests/AbbeyCoreTests/Fixtures/abbey-api/v1/` | new `AbbeyCoreTests` cases using `Codable` | `bash Scripts/verify-all.sh`, and it must be in the four CI test filters so the Linux job covers it |

`AbbeyCoreTests` is the correct home rather than `AbbeyServerTests` because
`AbbeyCore` is shared by desktop, server, and CLI, and because
`AbbeyCoreTests` is already one of the four filters the Linux CI job runs.

**Caveat that must be respected:** the musl Static Linux SDK path
(`Scripts/check-server-linux.sh`) ships no `Testing` or `XCTest` module and
therefore **cannot** run this gate. Conformance evidence for Swift comes from
the macOS gate and the Docker Linux CI job, not from the cross-compile lint.
Recording this prevents a future claim that "Linux is covered" based on the
wrong script.

### 8.4 Required fixture categories

**Proposed.** The corpus is not just happy-path envelopes. Per constitution
section 12:

1. Valid envelopes for every message family, both majors once `v2` exists.
2. **Encode fixtures and decode fixtures separately.** A round-trip test alone
   can hide a symmetric bug where both sides are wrong the same way.
3. Unknown additive field, once per policy class from section 7.3, asserting
   strict rejection or tolerant preservation as that class requires.
4. Invalid, duplicate, stale, oversized, and contradictory envelopes, each
   asserted to fail **deterministically** with a specific error code.
5. Boundary frames: exactly `max_frame_bytes`, one byte over, depth 32, depth
   33.
6. Cancellation-race and `idempotency_conflict` fixtures, since section 5 is
   the part most likely to be implemented differently in four places.
7. Degraded-operation fixtures, one per obligation in section 10.

### 8.5 Recommendation on a shared schema repository

**Not now. Later, on a named trigger.**

The constitutional position is not a preference. Section 2 states the contracts
package "is a Program 1 deliverable. It may later become a repository. Until
created and qualified, it must not be documented as an existing repository or
published package." Creating a repository now would document a thing that does
not exist and would be its own violation.

The engineering position agrees. A repository buys atomic cross-adapter schema
changes and a single reviewable history. It costs a release process, a version
matrix, a publication mechanism per language, and one more thing to keep in
sync, all before there is a second consumer of `v2`. A directory in `abi` plus
vendored copies plus digest equality buys the same correctness guarantee at
roughly zero process cost, and the mechanism has already been demonstrated
in-tree.

**Triggers. Any one of these makes the repository the cheaper option:**

1. **A third external consumer appears** that is not `abbey`, `abbey-bot`, or
   `AbbeyBot`. Four vendored copies is the practical ceiling for
   digest-equality bookkeeping.
2. **A second contract major is in flight** while the first is still being
   migrated. Dual-major maintenance across four vendored copies inside a
   runtime repository will produce a mistake.
3. **`abi` and `wdbx` diverge on contract ownership**, that is, the contracts
   directory acquires a change that is not driven by ABI. Ownership ambiguity
   inside a runtime repository is worse than a separate repository.
4. **A non-Donald contributor needs to propose a schema change** without commit
   access to the ABI runtime. The contract review surface should not require
   runtime write access.
5. **Generated bindings become worth publishing**, that is, the hand-written
   `Codable` and `serde` types become a real maintenance burden across four
   copies rather than a modest one.

Until a trigger fires, the answer is a directory. When one fires, the migration
is mechanical: move `abi/contracts/`, keep the same digest, keep the same
vendored-copy discipline, and add a release tag. Nothing in this design depends
on the contracts living in `abi` specifically.

## 9. Error taxonomy

### 9.1 Shape

**Proposed.** Every error frame carries:

- `code`: a stable lower-snake-case string from the closed set in 9.2. Never a
  free-form message match, never a numeric code that requires a table.
- `message`: fixed, non-leaking, chosen from a closed set. `abi-mcp`'s
  `ToolError::message` is the model: whatever the internal variant, the
  client-facing string is one of a fixed list and exposes no internal
  identifier, no path, no key, and no content.
- `retryable`: `never`, `after_backoff`, or `after_authorization_change`. This
  is what an adapter actually branches on. Deriving retryability from the code
  string in four places is exactly the kind of duplicated cognitive behavior
  this program exists to prevent.
- `degradation`: absent, or one of the section 10 modes, so an adapter can tell
  "this failed" from "this is running degraded and you must disclose it."
- `correlation_id`, and `receipt_ref` when a partial effect may exist.

### 9.2 The closed code set

Derived from constitution section 10's typed error list, one code per listed
condition plus the transport-level conditions this program adds.

| Code | Source | `retryable` |
| --- | --- | --- |
| `authorization_denied` | constitution section 10 | `after_authorization_change` |
| `approval_required` | constitution section 10 | `after_authorization_change` |
| `approval_expired` | constitution section 10 | `after_authorization_change` |
| `capability_unsupported` | constitution section 10 | `never` |
| `capability_revoked` | constitution section 10 | `never` |
| `schema_stale` | constitution section 10 | `never` |
| `schema_incompatible` | constitution section 10 | `never` |
| `provider_unavailable` | constitution section 10 | `after_backoff` |
| `provider_unqualified` | constitution section 10 | `never` |
| `memory_unavailable` | constitution section 10 | `after_backoff` |
| `memory_corrupt` | constitution section 10 | `never` |
| `memory_migration_blocked` | constitution section 10 | `never` |
| `cancelled` | constitution section 10 | `never` |
| `cancellation_raced` | section 5.3 | `never` |
| `consent_epoch_closed` | constitution section 10 | `never` |
| `resource_exhausted` | constitution section 10 | `after_backoff` |
| `deadline_exceeded` | constitution section 10 | `after_backoff` |
| `rate_limited` | constitution section 10 | `after_backoff` |
| `precondition_failed` | constitution section 10 | `never` |
| `postcondition_failed` | constitution section 10 | `never` |
| `rollback_complete` | constitution section 10 | `never` |
| `rollback_partial` | constitution section 10 | `never` |
| `rollback_failed` | constitution section 10 | `never` |
| `unauthenticated` | this program, section 6 | `never` |
| `unsupported_contract_version` | this program, section 7.2 | `never` |
| `corpus_digest_mismatch` | this program, section 7.2 | `never` |
| `idempotency_conflict` | this program, section 5.4 | `never` |
| `frame_too_large` | this program, section 3.3 | `never` |
| `malformed_frame` | this program, section 3.3 | `never` |
| `internal` | catch-all | `after_backoff` |

**Proposed invariant E1.** `internal` never carries a cause. Everything an
adapter is permitted to branch on has its own code. An `internal` that an
adapter needs to branch on is a missing code, and the fix is a
`contract_revision` bump, not a message-string match.

**Proposed invariant E2.** An error frame never contains message content, audio,
transcripts, prompts, generated responses, credentials, keys, values, vectors,
participant identities, or file paths. The same redaction discipline that
`MutationNotice` already enforces on the event path applies to the error path.

## 10. Degraded operation: the adapter's obligations

Constitution section 10 lists five degraded behaviors and decision 78 states
that API failure "degrades visibly and cannot weaken safety." Error codes do
not achieve that on their own: if ABI authorization is unreachable, the
**adapter** must deny consequential execution, and no message the API sends can
make it do so.

**Proposed.** These are conformance requirements on each adapter, each with a
named fixture in the corpus.

| Condition | Adapter obligation | Fixture |
| --- | --- | --- |
| `Authorize` unreachable or `internal` | **Deny** every consequential action. Do not fall back to a local permission check. Platform permission is a necessary fact, never sufficient authority. | `degraded/authorize-unreachable.json` |
| `Retrieve` returns `memory_unavailable` | Proceed only with tasks explicitly marked stateless-permitted, and **disclose** in the user-facing response that memory was unavailable. Never present a memoryless answer as a normal one. | `degraded/memory-unavailable.json` |
| No qualified model route | Use a deterministic path, or state that no safe route exists. Never fall back to a route that crosses a privacy boundary, gains tools, or lowers the evidence requirement. | `degraded/no-qualified-route.json` |
| Platform state changed between `Propose` and `Execute` | Revalidate, then **stop**. Do not apply a stale operation. | `degraded/stale-platform-state.json` |
| Rollback incomplete | Surface the receipt's per-step `completed` / `reverted` / `unresolved` enumeration. Do not summarize it as success or as failure. | `degraded/rollback-partial.json` |
| Connection to the host lost mid-epoch | Close the media epoch locally and immediately. The media gate does not depend on the API being reachable. | `degraded/host-lost-during-epoch.json` |

The last row is the one that most needs stating: `abbey-bot` owns the media
gate, and that gate must fail **closed** without consulting anything across the
API. Constitution section 10: "ABI or provider failure must not weaken the
media gate."

## 11. What belongs behind the API, and what stays adapter-local

### 11.1 Behind the API, canonical, never reimplemented locally

Constitution section 2 assigns exactly one owner per concern; this is that
table expressed as an API boundary.

- Persona definitions and selection contracts. Adapters render or request a
  persona; they do not define one.
- Typed authorization, capability grants, approval, revocation, delegation.
- Evidence semantics, retrieval policy, evidence dimensions, and SEA selection.
- Durable episode writes, retention classing, correction, supersession,
  contradiction, quarantine, deletion, and the claim ledger.
- Capability packages, their versions, risk classes, and promotion state.
- Command manifest compilation and validation.
- Model qualification manifests and routing decisions.
- Receipt minting and digest computation (invariant T2).

### 11.2 Adapter-local, and deliberately so

**Platform safety boundaries stay local.** This is not a compromise; it is the
correct placement, because these boundaries must hold when the API is
unreachable.

`abbey-bot` (Rust, Discord):

- Discord gateway connection, sharding, intents (`non_privileged()` today),
  rate-limit handling, and the REST client.
- Voice UDP, DAVE, transport AEAD, RTP/Opus, VAD, capture and playback.
- Consent epoch **enforcement** and the media gate. The API is told about
  epochs; it does not run them. Raw audio never crosses the API and never
  enters WDBX, per constitution section 10.
- Songbird lifecycle, barge-in detection, playback cancellation.
- `wyhash.rs`, `embedding.rs`, and the persona transcriptions pinned by golden
  tests. **Do not deduplicate these against ABI**; that reverses a documented,
  tested decision recorded in the crate itself.

`AbbeyBot` (Swift, macOS and Vapor):

- SwiftUI presentation, SwiftData local persistence, the `/dashboard` SPA.
- The human confirmation gate's **rendering and interaction**. The decision of
  whether confirmation is required is an API answer; the dialog is local.
- Keychain credential storage.
- Twitch IRC, EventSub HMAC-SHA256 verification, the replay guard, and Helix
  calls.
- The `/api/*` sync surface, its cursors, LWW rules, and tombstones. This is
  desktop-to-server sync of AbbeyBot's own local state and is not federation.
- Process lifecycle, `ABBEY_PACKAGE_ROLE`, toolchain selection.

`abbey` (Rust, runtime host):

- Process supervision, socket lifecycle, credential file management.
- Provider adapters, local model lifecycle, and packaging.
- OS control policy and audit (`src/os_control.rs`, its policy and audit
  siblings).
- Northbound-to-southbound proxying under invariant H1.

### 11.3 The line, stated as a test

If a behavior must remain correct while the Abbey API is unreachable, it is
adapter-local. If two adapters implementing it independently could produce
different authority, persona, evidence, or memory semantics, it is behind the
API. Anything satisfying both is behind the API for its **decision** and local
for its **enforcement**, which is exactly the consent-epoch arrangement.

## 12. Interval rules that constrain this program today

**Proposed, but forced by the constitution rather than chosen here.**

1. **`abbey-bot`'s JSON facts remain canonical until Program 4 completes.**
   Constitution section 5: its WDBX v1 rows "remain a semantic projection
   during that interval." Therefore any `ProposeWrite` originating from
   `abbey-bot` before Program 4 cutover must be marked `origin: projection` and
   must not be treated as a canonical episode. Failing to state this would let
   this document read as though `abbey-bot` can write canonical episodes today,
   contradicting section 5 and decision 77.
2. **No dual canonical writers.** Decision 77 and section 5. During migration,
   shadow-read, replay, compare, cut over one writer, retain rollback evidence.
   The API must therefore carry a `writer_role` discriminator so a shadow write
   is distinguishable on the wire, not only in the store.
3. **Digest and signature fields are optional and non-canonical** until the gap
   analysis section 6.4 findings are addressed. See invariant T1's closing note.
4. **`GuildSettings.learning_enabled` currently defaults to `true`** in
   `abbey-bot`, against constitution decision 31. That migration belongs to the
   guild-learning program, not this one, but the API must not make it harder:
   learning-state fields are read-and-report on this surface, and the API never
   sets a guild's learning state as a side effect of any other call.

## 13. Acceptance and evidence

Per constitution section 11, evidence at one level permits only that level's
claim, and no level auto-promotes the next.

| Level | What would establish it for this program |
| --- | --- |
| **C0** | This document, reviewed and approved by Donald, with the section 0.1 numbering conflict resolved. |
| **C1** | Schemas and fixtures exist; each of the four repositories has a passing conformance test asserting corpus digest equality plus encode and decode round-trips, including the invalid, boundary, unknown-field, and degraded fixtures. Privacy tests assert no content in error or event frames. |
| **C2** | Deterministic replay: a recorded frame sequence replays to equivalent decisions and equivalent cancellations across a restart, with pinned policy, schema, and fixture versions. |
| **C3** | Not applicable to a transport contract on its own. Offline evaluation belongs to the programs whose behavior the API carries. |
| **C4** | Proposal-only shadow operation: one adapter speaks `abbey.v1` for real traffic while its existing local path remains authoritative, and the two are compared. |
| **C5** | Bounded canary: one adapter, one guild, one low-risk reversible capability, fixed budget, monitoring, and a rehearsed rollback to the pre-API path. |
| **C6** | Donald witnesses one exact end-to-end outcome through the API on a named binary, adapter build, and contract revision. |
| **C7** | Sustained operation with drift bounds on version negotiation failures, corpus mismatches, and per-identity budget exhaustion. |

**Preregistered failure criteria**, to be fixed before any result is inspected:

- Any adapter reaching a consequential effect without an `Authorize` allow is a
  hard failure and blocks promotion at every level.
- Any content appearing in an error frame, an event frame, or a receipt is a
  hard failure.
- Any adapter computing a digest is a hard failure (invariant T2).
- Any degraded-operation fixture that an adapter passes by disclosing nothing
  is a failure even if it returns the right code.

**Rollback path.** Every adapter keeps its current local path until its own C5
completes. Rollback is disabling the `abbey.v1` client in that adapter's
configuration and returning to the pre-API path. No schema is deleted; the
corpus keeps every fixture permanently.

## 14. Open questions for Donald

1. **The section 0.1 numbering conflict.** Amend the constitution's Program 6/7
   assignment, or retitle this document? This is the only item that blocks C0.
2. **Strict corpus-digest enforcement.** Section 7.2 proposes warn-by-default
   and refuse-under-strict. Should the strict profile be the default for the
   local Mac topology, where all four adapters are updated by one person?
3. **`AbbeyBot`'s optional `ABBEY_API_TOKEN`.** Should the open-when-unset
   behavior on `/api/*` be closed as part of this program, or is it explicitly
   out of scope because that surface is AbbeyBot-local? This document assumes
   out of scope but flags it because an unauthenticated local surface beside an
   authenticated one is a standing hazard.
4. **Contract revision cadence.** Is a revision bump per additive change
   acceptable, or should revisions batch on a schedule?
5. **Where the reference implementation lands first.** This document does not
   choose. The natural first mover is `abbey`, because its daemon already has
   the envelope shape, the bearer discipline, the capability set, and the Unix
   transport, and because it is the host in the topology.

## 15. Summary of proposed invariants

| ID | Invariant |
| --- | --- |
| **T1** | The wire encoding is never a durable digest input. Digests name their own canonical encoding explicitly. |
| **T2** | Adapters carry and echo digests. They never compute or authoritatively verify them. |
| **H1** | The host is a policy-enforcing proxy. It may add authorization context; it may never remove, rewrite, or widen it. |
| **A1** | Effective authority is the deny-by-default intersection of channel-principal ceiling and subject-principal grants. |
| **V1** | A single wire shape is never reinterpreted. Changed meaning means a changed name. |
| **E1** | `internal` never carries a cause. A needed branch is a missing code. |
| **E2** | Error and event frames carry no content, identity, credential, or path. |
