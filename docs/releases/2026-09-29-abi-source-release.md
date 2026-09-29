# ABI Rust source release — 2026-09-29

## Scope and preservation

Donald selected `donaldfilimon/abi` and requested adaptation to ABI's existing
code. This release hardens that Rust framework. It does not import the separate
Abbey Bot product or merge the unrelated historical `donald-filimon/abi` PR.
Canonical Discord skeleton PR #14 and the Rust rewrite are already merged.

The baseline is `77e6b652a4e08809b67ec609ffdc28e64313f46a`. It contains no tracked
`.zig` or `build.zig` files. Historical rewrite records, fixtures, and Git history
remain intact. No additional Zig deletion or archive migration is necessary;
no frozen fixture, persistence format, sibling repository, model default,
public command, or provider authorization boundary changes in this release.

## Acceptance matrix

“Current” below means the named local source boundary is exercised by tests.
It does not imply deployment or live service acceptance.

| Area from the original plan | ABI implementation and evidence | Boundary |
| --- | --- | --- |
| Conversation and Discord | `abi-ai` deterministic routing, frozen CLI/MCP goldens, `abi-connectors` gateway/routing/WebSocket tests and local TLS peers. New tests cover redacted gateway configuration and cleanup on every gateway I/O failure. | Current local connector skeleton. Bot DM/guild memory, mentions/slash commands, Serenity lifecycle, and private bot help are outside ABI's existing product. |
| Providers and tools | `abi-agent-runtime` cancellation/budget contracts; `abi-agent-host/tests/contracts.rs` schema validation, policy/audit, bounded continuation, failures and terminal ordering; connector SSE malformed-frame and sink-error tests. A fabricated result now rejects the buffered provider turn before any tool executes. | Current injected-provider behavior; no cloud authorization, default-model change, real provider qualification, or claim of preempting an uncooperative provider. |
| Memory and persistence | CLI/MCP scratch-store completion/training tests, SEA evidence/retrieval tests, gateway episode reopen/signature/memory-edge tests. Native WDBX Abbey corpus and Python/Rust episode commitment tests are required by the gate. | Existing formats and opt-in episode policy remain unchanged. No production store is opened or bot-specific guild isolation claimed. |
| Approvals and scheduling | Agent-host denial/confirmation tests, semantic-change contracts, worker signed admission/cancellation/replay/deadline/quota/result-recovery tests, frozen one-shot scheduler status golden. | Current ABI foundations. Durable bot message delivery, quiet hours, music permissions, and exactly-once network delivery are not ABI features. |
| Images and audio | `abi-ai/src/multimodal.rs` deterministic synthetic image/video/audio/voice tests; model-runtime scratch Tokenizers/Safetensors fixtures with CPU and macOS Metal checks. | Local analysis and tiny-model infrastructure, not image generation/OCR service qualification, live voice/music, or human audio acceptance. |
| Managed operation | MCP frame/error recovery, bounded HTTP/SSE admission and shutdown tests; browser-studio slow-peer shutdown tests; gateway authentication and worker transport configuration contracts. | Local adapters and library lifecycle. No installed service, bot installer/rollback, listener deployment, or separate-host cluster acceptance. |
| Other transports | Existing Twilio local TLS and provider tests remain in the workspace gate. | ABI does not acquire the separate bot's Telegram/Slack adapters in this adaptation. |
| Compatibility and security | Frozen 14-command and 12-tool goldens, 113-artifact Abbey contract corpus, required dependency scan, source-size/format/clippy/rustdoc gates. | Existing reviewed exceptions remain unchanged. Independent security/cryptographic review and DAST remain external acceptance. |

## Changes and regression evidence

- `GatewayConfig` debug formatting redacts its token, including through
  `Gateway` debug output. Public fields and Identify behavior are preserved.
- `Gateway::run` invokes transport close after both successful and failed runs.
  The TLS adapter shuts down its socket immediately rather than waiting for the
  caller to drop it. Local tests cover invalid Hello/authentication, seven I/O
  failure positions, and peer EOF while the transport remains alive.
- The agent host rejects provider-supplied tool results while buffering the
  turn. The reproducer previously executed one tool before returning an error;
  it now executes and authorizes none.
- All dependency-resolving commands in `tools/check.sh` use `--locked`.
  Native WDBX contract conformance, the existing security policy, and release
  builds of `abi`/`abi-mcp` are mandatory stages.
- The benchmark runs the executable from `CARGO_TARGET_DIR`, with quoted paths.
  Scratch tests prove absolute/relative paths with spaces work and a missing
  selected build cannot silently use an old default binary.
- Self-hosted CI allocates a new temporary build directory for each run and
  cleans up that directory afterward. Executed workflow-step tests prove fresh
  allocation, failure without export, and confinement of cleanup; existing
  checkout/trust tests remain.

## Reproduction and evidence

Run from the canonical ABI checkout with the required sibling WDBX checkout:

```bash
CARGO_TARGET_DIR="$(mktemp -d "${TMPDIR:-/tmp}/abi-release-XXXXXX")" || exit 1
test -n "$CARGO_TARGET_DIR" && test -d "$CARGO_TARGET_DIR" || exit 1
export CARGO_TARGET_DIR
ABI_WDBX_PATH=:memory: ./tools/check.sh
```

Do not globally set `ABI_WDBX_PERSIST=0` for the suite: tests that deliberately
select temporary durable stores must be able to write them. The first baseline
attempt used that flag and failed two persistence tests; the corrected baseline
passed 906 Rust tests, with zero ignored. Its benchmark ran the old default
target, which is the reproduced gate defect fixed here.

The workflow pins WDBX to `9fee98ff5ccb92fa86a2ed44f93abd65e7e181ae`.
The local sibling had later unrelated commits, but its consumed crates,
manifests, lockfile, contract corpus, and episode oracle matched that pin.
CI checks out the exact revision. The Rust pin is `nightly-2026-09-01`;
the observed compiler is `rustc 1.100.0-nightly (0dfb098f3 2026-08-31)`.

The dependency policy retains only `RUSTSEC-2025-0141` and `RUSTSEC-2024-0436`,
the previously reviewed unmaintained `bincode` and `paste` exceptions in optional
TFHE-rs transitives. The baseline
`cargo-audit 0.22.2` scan passed without adding exceptions. The required final
gate repeats that scan against the locked graph and the current advisory DB.

Local source acceptance passed the complete revised gate: 143 Python policy
tests and 924 Rust tests, zero ignored Rust tests, required native WDBX and
cross-language conformance, security policy, format, clippy, build, Metal feature
tests, benchmark, rustdoc, and optimized `abi`/`abi-mcp` builds. Mintlify and
actionlint passed separately. Five new Rust regressions reproduced the runtime
defects before their fixes; five new Python tests cover gate and CI identity.

Release receipts must contain the clean source SHA, lockfile hash, WDBX pin,
toolchain, artifact checksums, test results, and a successful CI URL whose
`headSha` equals published `main`. Copy the macOS FoundationModels shim and
Metal dot library beside the binaries when retaining artifacts. Verify the
copied executables' loader dependencies, signatures, and frozen CLI help.
A release build and checksums establish
artifact identity; no cross-host byte-for-byte reproducibility is asserted.
Repository completion evidence belongs in `tasks/goals.md`; post-publication
CI receipts accompany the delivered artifacts to avoid a self-referential SHA.

## Operator acceptance remains separate

This is a source release. No installed artifact or service is replaced,
started, or restarted; no Discord command is registered or live message sent.
Real providers, model quality, live Discord isolation, deployment readiness,
human-witnessed voice, Linux/Windows runtime acceptance, CUDA/Vulkan execution,
verified ANE placement, separate-host operation, and independent audits remain
unperformed. CUDA compilation is an explicit skip when `nvcc` is absent.
Billing-refused hosted CodeQL and omitted hosted lanes are not passing evidence.
