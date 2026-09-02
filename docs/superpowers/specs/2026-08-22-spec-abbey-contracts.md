# Abbey contracts

Status: **proposed.** Written 2026-08-22.

> **Constitutional mapping.** This document specifies **Program 1, Abbey
> contracts**, in the ratified constitution's section 13. It is the one program
> whose number is the same under both numberings, because the earlier
> conversational list had no equivalent and this spec was written directly
> against section 13.

Section 13 defines the scope: "Publish principals, scopes, capabilities,
policies, consent, events, episodes, receipts, errors, claims, and compatibility
rules with cross-language fixtures. **No runtime behavior changes.**"

That last sentence is the whole discipline. This program ships a vocabulary and
its fixtures. It changes no decision any component makes.

## Why this comes first

The application-federation spec states the division outright: "Program 1 owns
the schemas; this program owns the wire." Every other program references types
this one has not yet named. Writing them down first is what stops five programs
from inventing five spellings of `principal`.

## Current, verified by reading source

Measured across the four repositories on 2026-08-22. The honest summary is that
**most of this vocabulary does not exist yet**, and the parts that do exist are
component-local rather than shared.

| Concept | Current |
| --- | --- |
| Principals | **Absent as a concept.** The word appears only incidentally (`abi-models/src/registry.rs`, `abi-model-runtime/src/provider.rs`). There is no type naming who is acting. |
| Scopes | **Absent.** `abi-agent-runtime/src/policy.rs` mentions the word; authorization is keyed on `ToolEffect`, not a scope. |
| Capabilities | **Absent as a contract.** `abi-gpu` and `abi-cli/src/backends.rs` use "capability" for hardware feature detection, an unrelated sense. |
| Policies | **Present, component-local.** `ExecutionPolicy` with `DenyAllPolicy` (documented safe default) and `EffectScopedPolicy` (documented as trusting the tool author's declaration, and selected by nothing in production). |
| Consent | **Present, adapter-local and real.** `abbey-bot` has a consent-epoch model in its voice path. Not expressed as a shared type. |
| Events | **Present, component-local.** `ModelEvent` in `abi-agent-host`; `WatchMutations` streams `MutationEvent` in `abi-wdbx-gateway`. |
| Episodes | **Absent. Zero occurrences of the word.** `V2AuditBlock` (`v2/types.rs:116`) is the nearest thing and carries 8 fields, which the gap analysis measures against a specified ~28. |
| Receipts | **Absent. Zero occurrences.** |
| Errors | **Present, component-local.** `GatewayError`, `HostError`, `ImportError`, tonic `Status` codes. No shared taxonomy. |
| Claims | **Present and the strongest existing piece.** `abbey/src/claims.rs`: a `Status` enum, `CLAIMS`, `CLAIMS_SCHEMA_VERSION = 1`, and a `validate_registry()` gate. Its five states predate the constitution's seven. |
| Compatibility rules | **Absent.** No versioning policy exists across components. |

One mechanism already exists and is proven, and this program generalizes it
rather than inventing a second one: the cross-implementation conformance fixture
checked into two repositories and asserted from both sides
(`abbey-bot/tests/fixtures/wdbx_v1_conformance.seg.jsonl` and
`../wdbx/crates/abi-wdbx/tests/golden/abbey-bot-projection.seg.jsonl`), each pinned
by its own test and both pinned to LF by `.gitattributes` after a real
CRLF-on-Windows failure.

## Proposed

### 1. One normative corpus, N vendored copies

Schemas live in **`abi/contracts/abbey/v1/`** as JSON Schema, because that is
readable by Rust and Swift without a code generator and adds no build-graph
dependency to either. Each consumer vendors a copy and asserts **digest
equality** against the normative one in its own test suite.

This is the existing pairwise fixture pattern promoted to a hub. It is chosen
over a shared crate or package for a reason that is not preference: `abbey-bot`
pins stable Rust 1.97.1 while the substrate needs nightly `portable_simd`, and
`AbbeyBot` is Swift. **No single toolchain compiles all consumers**, so a shared
compiled artifact is unavailable, and a shared *file* with an asserted digest is
the strongest mechanism that actually works.

A shared schema repository is **not** created now. The federation spec's triggers
govern that decision; until one fires, a directory in `abi` plus digest
assertions is less machinery for the same guarantee.

### 2. The eleven types, and what each must answer

Each schema answers one question and nothing else.

- **Principal.** Who is acting. Distinguishes an adapter identity (the Rust bot,
  the Swift server) from a subject identity (a Discord user) from a delegation
  chain. Two principals per request, not one, because "the bot did it on behalf
  of a user" is the normal case and collapsing it loses the audit.
- **Scope.** What a principal may reach, expressed as a tenant-qualified
  namespace. The tenant is the guild; guild isolation is the correctness
  boundary and guild-plus-user the privacy boundary.
- **Capability.** A typed action with declared inputs, effects, preconditions,
  and postconditions. Distinct from the hardware sense already in `abi-gpu`; the
  schema name is `capability.schema.json` and the hardware sense keeps its
  in-crate name.
- **Policy.** A decision function's inputs and outputs, so a decision can be
  replayed and audited without re-running the policy.
- **Consent.** An epoch with a subject, a scope, a grant time, and an
  expiry. Lifted from `abbey-bot`'s working voice implementation rather than
  designed fresh, because that one has been exercised.
- **Event.** Something that happened, with a monotonic ordering key and an
  acquisition time distinct from a receipt time.
- **Episode.** The durable record. This schema is **owned by Program 4** and is
  referenced here, not defined here, so the two cannot disagree.
- **Receipt.** Proof that an authorized action was attempted, carrying the
  request digest, the deciding policy, the principal pair, the outcome, and the
  rollback plan if one was captured. Entirely new.
- **Error.** A closed taxonomy with a stable code, a `retryable` flag, and a
  degradation hint. Closed, so an adapter can exhaustively match; a new code is a
  compatibility event.
- **Claim.** The evidence ledger entry, extending `abbey/src/claims.rs` rather
  than replacing it: bump `CLAIMS_SCHEMA_VERSION`, add the ladder level, and
  reconcile the five states with section 11's seven.
- **Compatibility rule.** How each of the above may change without breaking a
  consumer.

### 3. Compatibility policy

Three axes, and mixing them is what makes versioning arguments unresolvable:
**schema version** (the shape), **policy version** (the rules), and **contract
revision** (the negotiated set).

Unknown-field handling is deliberately **mixed**, not uniform:

- **Strict reject** on grants, consent, approvals, and anything carrying a
  digest. A field you do not understand in an authorization decision is a
  security event, not forward compatibility.
- **Tolerant round-trip** on content and telemetry, preserving unknown fields
  verbatim so an older adapter does not silently strip a newer one's data. This
  is the same rule `abbey-bot`'s WDBX projection already follows for unmodelled
  record types, and it is why that projection round-trips a real abi block.

`abbey/src/daemon/protocol.rs` is currently `deny_unknown_fields` on every
envelope. That is legitimate under the constitution's "round-trip or fail", but
it forecloses additive evolution, so moving its event bodies to the tolerant rule
is a deliberate `contract_revision` change rather than an incidental edit.

### 4. Fixtures

Every schema ships at least one **valid** and one **invalid** fixture. The
invalid one matters more: it proves the validator rejects, and a schema nobody
can fail is not a contract.

Each consumer asserts, in its own suite and its own toolchain:

1. the vendored copy's digest equals the normative copy's,
2. every valid fixture parses,
3. every invalid fixture is rejected.

Fixture files are pinned to LF via `.gitattributes` in every repository. This is
not hypothetical: a CRLF rewrite on a Windows checkout already broke the existing
conformance fixture once.

## What this program must not do

No runtime behavior change. No component may start enforcing a schema it did not
enforce before, and no decision may change. Enforcement is Program 2's
(authorization) and Program 4's (episodes). A schema that silently begins
rejecting traffic is a behavior change wearing a contract's clothes.

## Verification

- Every schema has a valid and an invalid fixture, and the invalid one is
  actually rejected.
- Each consumer's suite asserts digest equality against the normative corpus, so
  divergence fails a test rather than surfacing in production.
- `abi/tools/check.sh`, `abbey/check.sh`, `abbey-bot/check.sh`, and
  `wdbx`'s gate all green with **unchanged test counts for existing tests**,
  which is the evidence that no behavior moved.

## Honest residual

Nothing here is implemented. Seven of the eleven concepts do not exist in any
form today, and two of them (`receipt`, `episode`) have zero occurrences of even
the word across all four repositories. The claims vocabulary reconciliation is
the one piece with real existing code behind it, and it is also the one that
touches a working gate, so it carries the most risk of an accidental behavior
change in a program whose defining constraint is causing none.
