# In-Process Neural Voice (`--neural` flag) — Design

Status: Draft, pending user review.
Scope: Sub-project A of the "neural/ggml in-process sampler" todo item
(`tasks/todo.md`, Candidate next slices, priority 1). Sub-project B (real
GGUF/transformer inference in pure Zig) is explicitly out of scope and
queued for its own future brainstorming cycle.

## Problem

`tasks/todo.md` lists an open, non-blocked, non-non-goal item: an
in-process neural sampler, distinct from the existing `local_bridge.zig`
(which already dispatches to an *external* user-run inference server over
HTTP) and distinct from `incremental.zig`'s template-based persona
generator (real incremental emission, but not neural). There is already a
real, working pure-Zig neural network module at `src/features/nn/`
(manual backprop, real forward pass, real greedy sampling) at toy
character-level scale, currently only exercised via the `abi nn train` /
`abi nn sample` demo commands — it is not wired into live completion.

## Goal

Wire the existing char-LM into the live `abi complete` path as an honest,
small, in-process neural voice option, gated behind a `--neural` flag.
Quality will be limited (character-level, greedy decode, tiny corpus) —
this is explicitly a real-but-small slice, not a claim of LLM-quality
output.

## Architecture

- **Training (offline, one-time, not part of `./build.sh check`):** a
  script/step trains the existing char-LM (`nn.trainOnText` /
  `nn.trainOnJsonl`) on a corpus assembled from the repo's own docs
  (`AGENTS.md`, `README.md`, persona description text for Abbey/Aviva/Abi).
  The resulting `Model` is serialized and committed as a small binary/data
  file, e.g. `assets/nn/persona-checkpoint.bin`.
- **Loading (runtime, CLI startup):** a loader in `src/features/nn/` (or a
  thin wrapper module) deserializes the bundled checkpoint when `--neural`
  is passed. No training happens at runtime.
- **Generation/streaming:** a new `incremental.zig` `StreamMode` variant
  (`.neural`) — or a small parallel module — drives `nn.sample()` in a
  loop, buffering 4-8 sampled characters per `stream_callback` invocation
  (avoids per-character callback overhead), capped at a **300-character**
  max output length (no natural stop token exists in this toy model, so a
  fixed cap is required).
- **CLI wiring:** `abi complete --neural "<input>"` bypasses both the
  template router and the HTTP local-bridge, selecting this path
  directly. `--neural` and `--model` are **mutually exclusive** — passing
  both is a CLI error, not a silent override, since `--neural` is a mode
  flag rather than a routable model id.

## Honesty / claims discipline

Per `AGENTS.md`'s claims-discipline section, help text and any response
metadata for this path must state plainly: *"in-process character-level
demo model, trained on ABI's own docs — not a production LLM, not
comparable in quality to the template or live completion paths."* This
must never be positioned as a general-purpose assistant mode.

## Error handling

If the bundled checkpoint is missing, corrupt, or fails to deserialize,
fail loudly with a distinct error (e.g. `error.NeuralCheckpointMissing`).
Never silently fall back to the template router without telling the user
— per repo convention, no silent `catch {}` in inference paths.

## Testing

- CLI smoke test: `abi complete --neural "..."` produces non-empty output
  and exits 0.
- Unit test: checkpoint loader round-trip (serialize/deserialize
  fidelity).
- Unit test: chunked-streaming wrapper — matches the existing
  `incremental.zig` test pattern of asserting callback-received segments
  reconstruct the full output exactly.
- CLI error-path test: `--neural --model X` together produces the
  mutual-exclusivity error, not silent override.

## Explicitly out of scope for this slice

Tokenization beyond raw chars/bytes; any pretrained-weight loading; GGUF
format; attention/transformer architecture; quality comparable to a real
LLM. These belong to Sub-project B (real GGUF/transformer inference in
pure Zig), a separate multi-month initiative requiring its own
decomposition and design cycle — not folded into this spec.

## Open questions for implementation planning

- Exact corpus assembly: which specific files/sections feed the training
  corpus, and what's the resulting corpus size (affects training time and
  output coherence).
- Whether the bundled checkpoint format needs versioning (e.g. to detect
  a stale checkpoint if `Model`'s serialization shape changes later).
