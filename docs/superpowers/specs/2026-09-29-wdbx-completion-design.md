# Completing WDBX

Status: **design only, not implemented.** Written 2026-09-29. This document orders the remaining evidence-half gaps and names one first sub-project. It authorizes no code, no claim-level change, and no push. Sub-project 1 is specified in `2026-09-29-spec-member-erasure.md` and planned, as future work, in `../plans/2026-09-29-plan-member-erasure.md`.

## Context

WDBX's own ledger says the codebase implements most of the structural half and little of the evidence half (`wdbx/docs/claims.md`, `wdbx/README.md` "Status: honest"). Program 4 is that evidence half (`2026-08-22-spec-canonical-wdbx-episodes.md`).

What is already true, and must not be re-described as missing or promoted:

- C0 and C1: the v3 episode store, golden fixtures, and detached Ed25519 signatures over episode digests. That signature is not COSE, has no general key rotation or revocation, and is not verified by a second language.
- C2 is partial for canonical encoding only: a Python witness re-derives receipt digests. It is not a signature verifier.
- C3, C4, C5, and C6 are not claimed. C7 is partial and incomplete.
- Quarantine, contradiction, and resolution exist as memory-edge episodes. Retrieval does not weigh them. Ranking remains semantic, temporal, causal, and persona affinity multiplied into one score.
- Gap-analysis §6.9 (retention, cryptographic erasure, redacted derivatives, revocation edges, auditable garbage collection) is not implemented in the store.

Donald's 2026-09-29 decisions, recorded in `~/tasks/goals.md` under Federate and copied into the erasure spec §12, put provable member erasure ahead of retrieval-time staleness. The same-day revision that recommended keyed commitments as the end state is rejected. The end state is encrypted fact text inside WDBX.

## Approaches

**A. Keyed commitments.** Leave fact text in abbey-bot and store an HMAC under a member key in WDBX. Revision 1 of the erasure spec recommended this. Donald rejected it on 2026-09-29: keyed commitments are not the end state, because the plaintext would remain canonical outside WDBX and the unsalted hashes already on disk would stay guessable. Not used.

**B. One Program 4 change.** Land erasure, COSE envelopes, evidence-weighted retrieval, `task_regime` / `regime_posterior`, staleness weighting, and C3–C7 evidence in a single implementation. That is larger than one reviewable sub-project, and shipping any of it would pressure claim rows that are still unproven. Not used.

**C. Ordered sub-projects, erasure first.** Specify and later build provable member erasure alone. Name every other evidence-half gap in one deferral list. Do not raise C3–C7, do not call COSE or evidence-weighted retrieval Current, and do not treat this design as shipped. This is the design.

## Design

### Sub-project 1, the only one this design specifies

Provable member erasure, as `2026-09-29-spec-member-erasure.md` revision 3. Binding end state:

- Member fact text moves into WDBX, encrypted per member. The gateway holds the data keys, wrapped by a macOS Keychain master. abbey-bot never sees raw keys. **Keyed commitments are rejected for the end state.**
- The host operator may erase any member directly, with a signed receipt. A member may erase their own memories, and a guild owner, administrator, or manager may erase a member in that guild.
- After erasure the Keychain master key rotates automatically, at most hourly. The item is this-device-only. Donald further required after-first-unlock access; the exact attribute is measured on macOS 27.2 before any receipt mentions backups.
- Erasure clears all channel summaries in that scope.
- Each fact has its own key wrapped by the member key, so forgetting one fact destroys that fact's key.
- New byte fields are base64 in ledger lines, and the ledger cap for this work is 256 MiB.
- A scope switches to WDBX automatically after 7 days of shadow reads with zero mismatches. Rollback is operator-only through the abi CLI. An environment kill switch disables auto-switchover. After that rollback, this revision does not auto-switch the scope again.

Erasure appends a tombstone and destroys keys. It does not rewrite encrypted records, so digests, signatures, and replay stay valid. Holds are signed records and block erasure until released. Nothing in that spec is implemented. §12 approval is empty.

Abbey's in-process WDBX memory stays the v2 durable store behind `--features wdbx`. It is not a second canonical member-fact store. abbey-bot keeps canonical facts in `abbey-state.json` and the `wdbx.seg.0.jsonl` projection, and its `Cargo.toml` does not take a path or git dependency on abi or wdbx. abi and abbey keep sibling `path = "../wdbx/..."` dependencies.

### Ordered remaining gaps

1. **Provable member erasure** (this sub-project). Encrypted payloads, tombstones, holds, migration, hourly this-device-only rotation, summary clearing.
2. Redacted derivative blocks that link to an original without overwriting it.
3. Auditable garbage collection of unreferenced high-rate traces.
4. Retrieval-time staleness weighting.
5. Evidence-weighted retrieval that returns trust dimensions separately, instead of one multiplicative score.
6. `task_regime` and `regime_posterior` on the episode.
7. COSE envelopes and a cross-language signature verifier (finishing C2 for signatures, not claiming it now).
8. C3 live provider / Discord evidence.
9. C4 hosted service / federation evidence.
10. C5 production deployment evidence.
11. C6 operator-witnessed exact outcomes. The erasure spec's live acceptance is a later claim, not part of this design's implementation.
12. C7 reconstructible experiment manifest.

Only item 1 has a spec and a plan. Items 2–12 are deferred.

### Explicit deferrals

This is the only deferral list for the completion design. It matches erasure-spec §11.3.

- Future-learning opt-out (erasure does not itself stop later learning).
- A dedicated DM `guild_ref` assignment scheme beyond one key per `(guild_ref, member)`.
- A resume command that would let a rolled-back scope auto-switch again.
- Redacted derivative blocks.
- Auditable garbage collection of high-rate traces.
- Retrieval-time staleness weighting.
- Evidence-weighted retrieval.
- `task_regime` and `regime_posterior`.
- COSE envelopes and cross-language signature verification.
- Claim levels C3, C4, C5, C6, and C7.
- Hosted federation.

### Claim boundary

No claim level rises because of this document. Erasure, payload custody, COSE, and evidence-weighted retrieval are not Current. The completion design is not implemented in abi, abbey, or abbey-bot.

## Placeholder scan

In-scope sections of this file and of `2026-09-29-spec-member-erasure.md` contain no unfinished-marker tokens and no phrase that leaves a choice open. Deferred items are named in the list above and in erasure-spec §11.3. §11.2 of the erasure spec records draft rules Donald did not separately decide; they are rules, not placeholders. Implementation still waits on an empty §12 approval line.

## Self-review

The architecture matches the erasure spec: ciphertext in the signed episode, keys outside the ledger, tombstone append, hourly rotation, summaries cleared, keyed commitments rejected. Abbey is a consumer of the v2 store, not a second canonical fact store. abbey-bot remains a transcribing adapter. Items 2–12 are out of the writing plan on purpose. No section contradicts §12 of the erasure spec.
