# ABI Rust Rewrite — Port Plan

Goal: replace the entire Zig implementation of `abi` with Rust (nightly), and
delete every Zig source file, build script, and toolchain pin from the repo.

Branch: `rust-rewrite`.

## Scope (measured, not estimated)

Only git-tracked Zig counts. The ~2,100 `.zig` files under `.claude/worktrees/`
are git-ignored agent scratch worktrees and are not project source.

```
git ls-files '*.zig' | wc -l        # 302 files
git ls-files '*.zig' | xargs wc -l  # 61,500 LOC
```

| Area | Zig LOC | Notes |
|---|---:|---|
| `src/features/wdbx/` | 12,841 | vector store: HNSW, MVCC, WAL, REST, cluster |
| `src/features/ai/` | 7,764 | router, personas, constitution |
| `src/cli/` | 7,488 | registry + 25 handlers; **frozen surface** |
| `src/features/tui/` | 5,461 | dashboard / diagnostics render loop |
| `src/connectors/` | 4,633 | OpenAI, Anthropic, Grok, Discord, Twilio, HTTP |
| `src/foundation/` | 3,741 | env, creds, io, http, logger, json |
| `src/features/gpu/` | 3,437 | Metal/CUDA/Vulkan/WebGPU detect + CPU fallback |
| `src/mcp/` | 2,507 | JSON-RPC server; **frozen 12-tool contract** |
| `src/features/sea/` | 1,608 | sparse evidence attention learn loop |
| `src/core/` | 1,548 | config, scheduler, memory, registry, task |
| `src/plugins/` | 1,264 | 14 plugins, each `mod.zig` + `stub.zig` |
| `src/features/nn/` | 1,205 | small neural net |
| other `src/features/*` | ~2,990 | os_control, mobile, telemetry, shaders, accelerator, mlir, metrics, hash |
| root/tests/tools/examples | ~4,000 | `main.zig`, `root.zig`, contracts, benchmarks |

## Toolchain (a trap — read this)

`/opt/homebrew/bin/cargo` is Homebrew **stable** 1.97.1 and shadows rustup on
PATH. Because Homebrew ships real binaries rather than rustup shims,
`rust-toolchain.toml` is **silently ignored** — a nightly-only feature would
compile for whoever has the right PATH and fail everywhere else. This is the
same failure shape as the existing `brew-zig-shadows-zvm` note.

Every build must go through rustup explicitly:

```
rustup run nightly cargo …    # cargo 1.99.0-nightly / rustc 1.99.0-nightly
```

`tools/cargo.sh` wraps this; `tools/check.sh` is the replacement gate. Do not
call bare `cargo`.

## Sequencing: port additively, delete once at the end

**This reverses an earlier plan to delete each surface's Zig as it was ported.
Do not re-adopt that; it does not work here.** `build.zig` wires the whole
module graph through `src/root.zig`, and the test steps (`test-contracts`,
`test-feature-contracts`, `check-parity`) walk all of it, so deleting any one
directory breaks `zig build check` — and it stays broken until the last
directory goes. From that first deletion onward there would be no running Zig
implementation to compare against.

That oracle is worth more than a monotonically decreasing Zig count. So:

- **Steps 2–10 are purely additive.** The Zig tree stays intact and
  `./build.sh check` stays green, so behaviour can be diffed against a working
  implementation at any point and every commit leaves a shippable repo.
- **Step 11 deletes all Zig in one commit**, once `tools/check.sh` covers
  everything the Zig gate did — including the golden fixtures below.

### Golden fixtures — capture before deleting anything

Captured at commit `919dad8` while the Zig gate was green; see
`tests/golden/README.md`. They convert "the Rust CLI has the right command
names" into "the Rust CLI emits byte-identical output".

- `tests/golden/help.json` — the full 18 KB frozen CLI surface
- `tests/golden/help.txt`, `help-<command>.txt` × 13
- `tests/golden/completion.{bash,zsh,fish}`
- `tests/golden/mcp-initialize.json`, `mcp-tools-list.json` — the 12 tools with
  full input schemas, **in emitted order** (which is not declaration order:
  `wdbx_stats` comes after `plugin_list`)
- `tests/golden/wdbx-format.md` + synthetic `wdbx-sample.*` fixtures
- `tests/golden/mcp-tool-calls.jsonl` — all 12 tools with empty arguments; the
  nine validation error strings are contract too
- `tests/golden/mcp-tool-calls-args.jsonl` — the success paths
- `tests/golden/wdbx-db-verify.txt`, `wdbx-stats.txt`, `backends.txt`,
  `plugin-list.txt`, `scheduler-status.txt`, `mcp-scheduler-calls.jsonl`

- [x] **0. Workspace** — cargo workspace, nightly pin, wrapper scripts, gate.
- [x] **1. `abi-foundation`** — errors, env, time, temp_path, json, validation, logger, io, text, credentials (+file/keychain/secret/Windows ACL), http, system, plugin_manifest. 155 tests.
- [x] **1b. Golden fixtures** — captured while the Zig gate was green.
- [x] **2. `abi-core`** — config, registry, task, scheduler, memory. Concurrency decided: **one-shot**, tasks run synchronously on the caller's thread (`mode=one-shot`, `running=0` at rest). Golden-tested against the captured `scheduler_stats` output. 79 tests.
- [x] **3a. `abi-connectors` core** — connector types, URL/auth (HTTPS enforcement + host-boundary check), payload builders + byte-exact local synthesis, SSE parsing (both dialects), `Transport` trait with `ureq` live impl + `RecordingTransport`, clients for OpenAI/Anthropic/Grok/Discord/Twilio. 75 tests.
- [ ] **3b. `abi-connectors` remainder** — Discord gateway + WS client (`discord_gateway.zig` 483, `discord_ws_client.zig` 228, `discord_routing.zig` 126), Twilio relay (`twilio_relay.zig` 554), the local bridge (`local_bridge.zig` 236) and the Apple `FoundationModels` shim (`fm.zig` 217). These need a WebSocket client and a Swift FFI shim; ~1.8k Zig LOC.
- [x] **4a. `abi-wdbx` on-disk format** — records (all 6 types), both hash encodings, manifest, checkpoint load, chain verification. **Verified against the user's real 301-epoch store**: 327 blocks, chain verifies from genesis, 32-dim vectors. 56 tests.
- [x] **4b. WDBX checkpoint salvage** — descending newest-valid recovery
  skips missing/corrupt active epochs without merging full checkpoints or
  duplicating the block chain; total corruption remains a loud error.
- [x] **4c. WDBX checkpoint writer** — all six record types, invariant
  validation, SHA-256 trailer verification, atomic segment/manifest publication,
  and multi-epoch round trips.
- [x] **4d. WDBX WAL/recovery core** — CRC32-framed append, bounded verify,
  torn-tail handling, absolute vector continuity, deterministic block replay,
  epoch-gated checkpoint merge/stale-WAL discard, and checkpoint reset.
- [x] **4e. WDBX algorithms/services** — HNSW index + storage, richer MVCC,
  cluster surfaces, compression/entropy/neural-compress,
  FHE + crypto_he demos, spatial 3-D octree, temporal graph, multiway engine,
  ANS, retrieval, remote compute. A deterministic exact cosine index provides
  the correctness oracle, and the layered HNSW graph/storage/search core with
  rollback journaling is now ported. A durable store facade now joins recovery,
  WAL-backed mutations, checkpoints and HNSW search. Manifest-authoritative
  retain-latest compaction and reset are also ported. The deterministic
  in-process Raft-style election/replication/failover core and its bounded TCP
  RequestVote/AppendEntries transport are ported, including fixed-work
  shared-secret authentication, peer allowlisting/reload, fail-closed
  non-loopback binds, and real loopback quorum tests. The loopback-only REST
  surface is ported with its five routes, hybrid temporal/causal/persona
  re-ranking, optional fixed-work bearer authentication, 64 KiB request bound,
  failed-auth-aware token bucket, and real-TCP tests. Compute backend selection,
  runtime CPU feature detection, nightly portable-SIMD DOT with ragged-tail
  handling, truthful ANE presence metadata, deterministic accelerator fallback,
  and the bounded loopback remote-DOT reference transport are also ported. The
  3-D spatial index now includes Euclidean/Manhattan/cosine distance, borrowed
  payload results, lazy octree rebuild, radius and k-nearest queries, and
  threshold/distribution tests against a linear oracle. Per-vector affine
  8-bit quantization and exact order-0 canonical Huffman coding are ported too;
  Huffman retains the 256-byte code-length table and stored-mode fallback, so it
  never expands incompressible payloads. This does not claim learned/SOTA
  compression, native CUDA/Vulkan/ANE execution, or production remote TPU
  dispatch. The deterministic in-process tanh/linear autoencoder, single-key
  additive WyHash-masked aggregation demo, and arbitrary-precision DGHV
  add/multiply demo are now ported with the same 126-bit secret, 20-bit noise,
  and tested depth-3 reference constants. They are explicitly not production
  encryption, multi-key HE, bootstrapped FHE, security-audited, or SOTA learned
  compression. Self-contained order-0 rANS plus order-1 previous-byte residual
  coding is ported with stored fallback, explicit corrupt-mode/truncation errors,
  and deterministic blobs. Persona-weighted, persona-isolated, and semantic/3-D
  spatial hybrid retrieval are also ported with borrowed store views and the
  oracle's saturating 8x candidate over-fetch. The append-only MVCC audit chain
  now preserves deterministic sequencing/hashing, stable shared blocks, frozen
  snapshots, concurrent append safety, and strict integrity verification. The
  bounded multiway core now ports exact byte-string rules, overlapping matches,
  deterministic breadth-first expansion, state deduplication with full event
  multiplicity, atomic hard caps, cancellation/deadline handling, and structural
  metrics. Canonical JSON/config/export hashes, token-lineage causal edges, DOT,
  resume decoding, content-addressed WDBX persistence, and latest/config-hash
  retrieval are ported too; an opt-in integration test proves the representative
  Rust canonical export is byte-for-byte identical to the restored Zig oracle.
  Durable recovery now also recognizes a legacy Zig single-file checkpoint when
  no active segment manifest exists, so the next Rust checkpoint can migrate it
  into the segmented layout without orphaning its records.
  TLS cert/key environment loading and accessibility validation are ported too,
  while preserving the disclosed-partial boundary: native TLS termination is
  not linked and non-loopback production hardening is not claimed.
- [ ] **5. `abi-ai` + `abi-sea` + `abi-nn`**
  - [x] **5a. `abi-ai` core + `ai_run`.** Identity contracts, the keyword
    router, incremental persona generation, and constitutional governance. The
    crate is **pure** — no WDBX dependency, no I/O, fully deterministic — which
    is exactly why `ai_run` is byte-reproducible. Attached to MCP and verified
    two ways: the captured fixture matches byte-for-byte through full dispatch,
    and a **60-input differential run against the live Zig binary has zero
    mismatches** (neutral prior, every keyword class, prefix-stem vs suffix
    false positives, explicit persona addresses, punctuation trimming, unicode,
    near-tie mixtures). 46 crate tests.
    - Two fidelity traps found and fixed, both load-bearing rather than
      theoretical. (1) Zig's `std.ascii.whitespace` includes vertical tab
      (0x0B); Rust's `is_ascii_whitespace` does not, so `"Aviva\x0bgo"` would
      have routed to Abbey instead of Aviva — confirmed against the Zig binary
      in both directions. (2) Routing must accumulate in `f32`, matching Zig's
      order and precision; widening to `f64` would silently change the outcome
      at near-ties.
    - `AuditResult.timestamp` is dropped: no ported caller reads it, and a
      wall-clock field inside an otherwise pure, comparable result would make
      the type non-deterministic for no benefit.
  - **The `ai_*` fixtures are a weaker oracle than the plan assumed.** Only
    `ai_run`'s captured line is store-independent and reproducible.
    `ai_complete`, `ai_learn`, and `ai_train` embed live store counters
    (`total_vectors`, `query_vector_id`, a SHA-256 `block_id`) captured at one
    moment — and the capture itself advanced the store, which is why successive
    fixture lines show `total_blocks` 328, 329, 330. Those three are **shape and
    field-order references, not equality targets.** What must be asserted for
    them instead: the persona substring byte-for-byte, the field names/order/
    formatting (`audit_escore` to three decimals), and the counter *arithmetic*
    (`response_vector_id == query_vector_id + 1`, `metadata_key ==
    "completion:{query_vector_id}"`, `total_*` advancing by exactly the reported
    delta) against a seeded temporary store. That is a stronger contract than a
    frozen line, because it holds at every store state.
  - **Store safety for the remaining work.** `ai_complete`/`ai_learn`/`ai_train`
    write to the user's real store under `~/.abi/` when a path resolves. All
    step-5 testing must use a scratch `DurableStore` (parameterised, never via
    process env), and the real store's content digest must be re-verified before
    each commit — this is the one failure here that git cannot undo. Content-only
    SHA-1 of every file under `~/.abi/` for this session (5b closeout):
    `39363c5aab63f23bdaa74ec813ff8b678926b07d`. Tests never opened that path;
    the WAL remains a header-only `base_epoch=306` record with no data frames.
  - [x] **5b. `ai_complete`.** Scope is now measured rather than guessed:
    - **The modulator is *not* on this path.** `ai_complete` calls
      `completeWithScheduler` → `completeWithStore` → `complete()`, which is the
      pure `analyzeSentiment` + `selectBestProfile` pair already ported in 5a.
      `AdaptiveModulator` is only reached through `completeAdaptive` /
      `completeWithStoreAdaptive`, i.e. the SEA path. So persona selection for
      `ai_complete` is deterministic and store-independent; only the counters and
      `block_id` are store-derived. (An earlier note here assumed otherwise.)
    - **Wyhash ported faithfully** as `abi_foundation::wyhash`, verified against
      188 `(seed, len, hash)` triples emitted by the pinned Zig toolchain
      (`crates/abi-foundation/tests/wyhash_zig_refs.txt`). The Rust `wyhash` 0.5
      crate is deliberately not used for embeddings — measured divergence is
      total (seed 3 / `"hel"`: Zig `10846395113768030678` vs crate
      `490820195397404894`). The same trap remains, harmlessly, in
      `abi-wdbx`'s `crypto_he::mask()` (self-consistent ephemeral ciphertexts).
    - Ported: `textEmbedding` / `responseEmbedding`, the model catalog, pure
      `complete` with hard safety-veto substitution, `completionMetadataJson` /
      `completion:<id>` (UTF-8-safe escaping — iterating by `char`, not by
      byte-as-char), and the MCP persistence tail (`put_vector` × 2 + metadata
      KV + audit block) against a scratch `DurableStore`. Store resolution is
      parameterised so tests never touch process env or `~/.abi/`. When no
      persist path resolves, the tool reports `persisted=false` with an explicit
      `wdbx_status` (in-memory `DurableStore` is not yet ported). Attached to
      MCP `ai_complete`.
  - [x] **5c. `abi-sea` + `ai_learn`.** New `abi-sea` crate: memory taxonomy,
    query-plan keyword inference, eight-signal scorer + budgeted selection,
    evidence recall (semantic + exact_recall lexical blend, authority forced to
    `inferred` for generic store metadata), prompt augmentation with the 4 KiB
    preamble cap, and the learn loop (adaptive complete + independent weight
    reload/save under `modulator:weights`). Pure `AdaptiveModulator` lives in
    `abi-ai`; store I/O stays in SEA. MCP `ai_learn` is attached with the same
    scratch-store / no-env test discipline as `ai_complete`. 17 crate tests +
    MCP report-line tests.
  - [x] **5d. `ai_train` (MCP path).** Profile validation, dataset inspection
    (text/csv/jsonl), confined-path helpers, profile embedding → store vectors +
    `agent:{profile}:training` KV + audit block. **`backend=cpu`** is disclosed
    rather than Zig's `gpu-metal` (no Rust GPU linked). The optional
    PointNeuralNetwork autoencoder weight-write is not ported yet — the message
    says `model weights unchanged` when no net is trained. Full `abi-nn` char-LM
    demo remains open under step 6/CLI `nn`.
- [ ] **6. `abi-gpu` + small features** — gpu, accelerator, shaders, mlir,
  hash, metrics, telemetry, mobile, os_control.
  - The bounded process-wide telemetry counter table and Prometheus text
    rendering are ported as `abi-telemetry`.
  - [x] **6a. `abi-gpu` detection + MCP `gpu_status`.** Declared seven-backend
    capability table, preferred backend (Metal on macOS / simulated elsewhere),
    and the MCP wire line. **Native kernels are not linked** — `accelerated`
    is always `false` and the message discloses vectorized CPU fallback. Shape-
    checked in the golden fixture path; not byte-equal to Zig's Metal-linked
    message.
  - [x] **6b. `abi-nn` + CLI `nn train|sample`.** Hand-backprop char-LM demo
    (embed → hidden → softmax, SGD/Adam). Loss-decrease and greedy-sample
    property tests pass. JSONL field extract + CLI wiring. Demo-grade only;
    checkpoint persist format still open.
- [ ] **7. `abi-tui`**
  - [x] **7a. One-shot dashboard / `tui` / `--tui`.** Stacked digest with all
    five panes (System, Plugins, WDBX Storage, Scheduler, Memory), `--list-panes`,
    `--json`, `--compact`, `--pane`, `--plain`, `--once`. Collects live plugin
    registry (16), one-shot scheduler probe (completed=2), MemoryTracker, and
    honest `abi-gpu` status. Interactive raw-mode refresh is **not** linked —
    every invocation is one-shot with an explicit footer note.
- [x] **8a. `abi-cli` contract model** — frozen 13-command metadata,
  top-level help, shortcut resolution, and argument-free command help are
  golden-tested. This does **not** claim handler or full typed-help parity.
- [ ] **8b. `abi-cli` executable** — typed/raw dispatch, all command handlers,
  full `help.json` / `help-*.txt`, and `completion.*` parity.
  - A real nightly-Rust `abi` binary now owns process streams/exit codes,
    top-level help/JSON/completions, suggestions, shortcuts, and explicit
    not-yet-ported failures.
  - WDBX `db`, `block`, stats/empty query modes, cluster status/demo, compute
    info, secure demo, bounded `simulate`, and the captured HNSW benchmark are
    attached. `simulate` includes config/rules files, hard bounds,
    cancellation, canonical JSON/DOT, resume from JSON or WDBX, and WDBX
    persistence. Help, dry-run, JSON, DOT, and stable summary bytes match Zig;
    two-way Zig/Rust JSON and checkpoint resume produces one identical
    canonical export. The benchmark preserves the oracle workload and report
    shape without making cross-run performance claims.
  - `scheduler status` is attached with byte-exact one-shot scheduler,
    serialized task execution, MemoryTracker, and telemetry output. Local
    `auth status` and `auth logout` are attached through the Rust credential
    backend; interactive `auth signin` remains open.
  - `backends` is attached with Rust build identity, explicit per-feature
    migration status, CPU SIMD selection, and claim-honest native accelerator
    fallback disclosure.
  - [x] **8c. CLI complete / train / nn / agent / dashboard.** Local
    `complete` (+`--learn`), `train`, `nn train|sample`, one-shot
    `dashboard`/`tui`, and `agent plan|multi|train|os|spawn|browser` are
    attached. Interactive `agent tui`, complete `--live`/`--neural`, and
    `twilio` remain open.
  - Still open: interactive TUI raw-mode, REST/cluster listener CLI, interactive
    auth signin, Twilio simulation, workspace-tree `file_context` on agent plan.
- [x] **9a. `abi-mcp` protocol + stdio transport** — JSON-RPC envelope,
  structural pre-check (size/depth/object-root), the frozen 12-tool table
  (schemas pre-parsed so property order is preserved), declarative field
  validation, and the byte-by-byte stdio line transport (overlong lines
  dropped with a `-32700` before they grow past `MAX_REQUEST_SIZE`, matching
  Zig's accumulate/clear behavior exactly, including the double-response edge
  case that produces). `initialize` and `tools/list` are golden-tested
  byte-for-byte against `mcp-initialize.json` / `mcp-tools-list.json`, order
  included. `tools/call` dispatch is golden-tested against every one of the 9
  validation-error lines in `mcp-tool-calls.jsonl`.
  - Wired to real backends: `scheduler_stats`/`scheduler_info` (via
    `abi-core`, golden-matched — the MCP variant never submits a probe task,
    unlike the CLI's `scheduler status`, so it stays all-zero at rest) and
    `connector_test` for `openai`/`anthropic`/`discord`/`grok` (via
    `abi-connectors`' already-ported local synthesis; `openai` is
    golden-matched byte-for-byte, including Anthropic's MCP-specific
    `max_tokens=256` versus the connector's own default of 4096).
  - Wired after 5a–5d/10: all four AI tools (`ai_run`, `ai_complete`,
    `ai_learn`, `ai_train`), plus `plugin_list`/`plugin_run`.
  - Wired: `wdbx_query` (persona prototype seed + hybrid re-rank via
    `hybrid_search_with_persona`, fixed `now_ms=1000` matching Zig),
    `gpu_status` (honest no-kernel disclosure via `abi-gpu`).
  - Honestly stubbed (`NotYetPorted`, after validation still runs):
    `connector_test` for `twilio` only — depends on `twilio_relay.zig`'s
    conversation builder from step 3b.
  - `wdbx_stats` reads the real durable store (env resolution — `ABI_WDBX_PATH`,
    `ABI_WDBX_PERSIST`, `XDG_DATA_HOME`, `HOME` fallback — ported and unit
    tested standalone) but **discloses `backend=cpu`** rather than Zig's
    linked `metal`, since no Rust GPU backend is linked; excluded from the
    golden byte-match for that one field, everything else about it is real.
  - State is intentionally simpler than Zig's: no shared long-lived
    scheduler/session (each call opens fresh) since no ported tool mutates
    the store yet and the double-checked-atomic lifecycle only pays for
    itself once one does.
  - Still open (step 9b): the HTTP/SSE transport, and every tool currently
    stubbed above once its backing feature lands.
- [x] **10. `abi-plugins`** — all sixteen bundled plugins, the plugin manager,
  and both listing surfaces. `abi plugin list | run` and the MCP `plugin_list` /
  `plugin_run` tools are attached and byte-verified against the live Zig binary,
  not just the fixtures: `plugin list` diffs identically, and all three MCP calls
  (list, a `__cmd__:` run, an unknown-name error) are byte-identical including
  `plugin_run`'s "Internal error" — Zig's `errorMessage` had no arm for
  `error.PluginNotFound`, so it fell through to `else`.
  - The two listings differ **by contract**, and both are golden-tested: MCP
    emits declaration order (`telemetry-exporter` third), while the CLI renders
    the registry alphabetically, because Zig built that list from a generated
    file that walked the plugin directory.
  - **mod/stub parity is preserved, not dropped.** Both `mod.rs` and `stub.rs`
    implement a `Plugin` trait, so a missing item is a compile error, and
    `assert_plugin_parity!` adds a `const` check that the four metadata constants
    agree. That is strictly stronger than `tools/check_parity.zig`, which the
    plan already recorded as a deliberate drop.
  - Two disclosed deviations. (1) `entry_point` is `mod.rs`, not `mod.zig` — that
    field names a file that will not exist after step 11, so the golden
    assertions rewrite exactly that token and match every other byte. (2)
    `load_bundled` reads a compiled-in table instead of resolving 16
    repo-root-relative paths, so the frozen 16-plugin listing no longer depends
    on the process's working directory (Zig silently emitted `count=0` when run
    from elsewhere). A test parses all 16 on-disk manifests and asserts they
    match the compiled-in table field by field, so the two cannot drift.
  - `abi-foundation`'s manifest validator gained `commands` /
    `context_providers`, reproducing Zig's per-field mix of strict rules (a
    non-object entry or absent/empty `name` fails) and lenient ones (a
    non-string `summary` becomes `""`; a non-array `aliases` is ignored).
  - Fixed in passing: `abi scheduler status` renders the whole process-global
    telemetry table, so its golden test raced any test recording an event. Reads
    and writes of that table now take a shared lock (`reset()` alone was not
    enough, since Cargo runs a crate's tests as threads in one process).
- [ ] **11. Zig teardown, in one commit** — `src/**/*.zig`, `build.zig`, `build.zig.zon`, `build.sh`, `.zigversion`, `zig-out/`, `zig-cache/`, `.zig-cache/`, `tools/*.zig`, `tests/**/*.zig`, `examples/**`, `.gitattributes` Zig rules, `.github/workflows` calling `./build.sh`.
- [ ] **12. Docs + memory** — `CLAUDE.md`, `AGENTS.md`, `GEMINI.md` together (they must not drift); `README.md`, `CHANGELOG.md`, `docs/**`; the `abi/` row in `~/CLAUDE.md`; delete the now-false `zig-pin-path` and `brew-zig-shadows-zvm` memories.

## Frozen contracts the Rust side must satisfy

These separate "done" from "it compiles".

**MCP: exactly 12 tools, no more, no fewer** (`src/mcp/handlers.zig`):
`ai_run`, `ai_complete`, `ai_learn`, `ai_train`, `wdbx_query`, `wdbx_stats`,
`scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`,
`plugin_list`, `plugin_run`. The contract test ports with the server.

**CLI: 13 top-level commands, metadata verbatim** (`src/cli/usage.zig`):
`help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`,
`tui`, `dashboard`, `wdbx`, `scheduler`, `nn`. Plus: the `--tui` →
`tui` shortcut rewrite in `main.zig`, `help-json` output, and bash/zsh/fish
completion output. Name/usage/summary are deliberately isolated in `usage.zig`
so help renders without the rest of the graph — keep that separation.

**WDBX on-disk format.** `~/.abi/` holds live `wdbx.seg.*.jsonl`, a manifest,
and an index. The Rust store must read existing segments, or ship a documented
migration. Silently orphaning the user's local state is not acceptable.

**`std.Io` has no Rust analog.** Zig 0.17 threads an explicit `std.Io` through
`main` → dispatch → every handler. This is the one part that is not mechanical
translation: each module needs a decision. Default is blocking `std` I/O with
an injected writer/reader pair for testability; async only where a listener
genuinely needs it (WDBX REST, MCP HTTP).

**mod/stub parity** (`tools/check_parity.zig`, 14 plugins × `mod.zig` +
`stub.zig`) is a Zig-shaped invariant: a feature and its no-op stub must expose
identical APIs. In Rust the same guarantee comes from a trait plus a
compile-time impl check, so the parity *script* is dropped and the invariant is
preserved as a test. Recorded here so the drop is deliberate, not silent.

## Gate

`./build.sh check` (build + tests + lint + parity) is replaced by
`tools/check.sh`: `cargo fmt --check`, `cargo clippy -D warnings`,
`cargo build --workspace`, `cargo test --workspace`, contract tests. GitHub
Actions on this repo is billing-locked, so the gate is local-only and must be
run before every commit.
