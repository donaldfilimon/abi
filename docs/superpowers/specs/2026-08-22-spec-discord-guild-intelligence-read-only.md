# Program 3: Read-only Discord guild intelligence

Status: **approved; initial closed synthetic C1 foundation implemented, full
Program 3 acceptance incomplete**. Date: 2026-08-22; evidence updated
2026-08-23.

This focused design extracts Program 3 (`discord-guild-intelligence`) from the
approved combined guild-intelligence/execution design. The constitution and
the federation reconciliation remain authoritative. This document narrows
scope; it does not amend normative intent.

## Outcome and evidence boundary

Program 3 turns a bounded, metadata-only guild observation into a deterministic
five-view twin, permission findings, operator-selectable alternatives, and a
non-executable desired-state plan with rollback preview. Its first slice is a
local synthetic recording adapter in Abbey `app_core`.

The slice initially claims **C1 local contract evidence only**: compiled
contracts, deterministic synthetic replay, negative fixture-authorization and
schema tests, and a static exclusion guard. Its replay mechanism is intended to
support a later complete C2 gate, but this partial fixture is not labeled as
full Program 3 C2. It does not establish Discord connectivity, production
fitness, participant consent, provider state, durable persistence, or any live
behavior. Those claims require separately authorized evidence.

The initial foundation is Abbey commit
`68a5c8530767883a14111954dbbb6b3bf6835414`, merged by
[`donaldfilimon/abbey#92`](https://github.com/donaldfilimon/abbey/pull/92) as
`edf0b029f23d38eea8a7076e8dcb17b9d77a2551`. Its exact-head and post-merge
macOS production gates passed. That evidence covers closed replay,
normalization, five views, watermarks and coverage, lower-authority/schema/
reference rejection, explicit selection, rollback metadata, redacted status,
and a static read-only exclusion guard. It does not yet cover explicit
administrator-acceptance and hard-limit vectors, exhaustive permission and
unknown-target vectors, plan precondition/postcondition fields, or a complete
C2 replay gate.

## Hard boundary

The slice MUST:

- accept only recordings that claim owner or administrator authority, reject a
  recorded owner whose opaque reference differs from the recorded guild owner,
  and label that basis as synthetic input rather than P2/live authorization;
- retain structural and permission metadata, opaque references, timestamps,
  schema versions, confidence, privacy classification, coverage, and
  watermarks;
- represent structure, authority, workflow, goal, and health as distinct twin
  views and keep facts, inferences, and operator goals distinguishable;
- calculate permissions deterministically, including administrator override,
  owner authority, base-role union, and ordered channel overwrites;
- emit typed findings, at least two substantive alternatives plus do-nothing,
  require explicit option selection, and produce only a desired-state plan;
- include a rollback preview derived from the observed before-state;
- emit a fixed, concise, redacted status suitable for projection; and
- replay the same closed synthetic recording byte-for-byte deterministically.

The slice MUST NOT contain Serenity, Discord REST/Gateway I/O, network clients,
message bodies, message history, member enumeration, credentials, raw tokens,
audio, transcripts, filesystem paths, WDBX access, durable stores, dynamic tool
registration, commands, approval engines, executors, actuators, or write verbs.
Program 5 exclusively owns previewed Discord effects, approval, execution,
verification, compensation, and receipts.

## Closed observation contract

An observation has an explicit schema version, synthetic-fixture marker,
observation timestamp, opaque guild/operator references, operator authority,
guild-wide role metadata, channel metadata, bot-self membership, active-thread
metadata, and coverage. Unknown fields fail closed. All collections have hard
limits, references are bounded opaque strings, and free-form Discord names or
content are not retained.

Coverage records the surfaces requested and observed, counts visible and known
objects, marks unavailable surfaces explicitly, and states that content and
member enumeration are excluded. A watermark binds each object to its source,
observation time, schema version, privacy class, and stable digest. Input order
is normalized before analysis.

## Five-view twin

1. **Structure** contains normalized roles, channels, parent relations, and
   active-thread relations.
2. **Authority** contains owner/admin status, bot-self role position, base
   permissions, and calculated per-channel effective permissions.
3. **Workflow** records only metadata-derived workflow observations; absence is
   explicit and never inferred from message content.
4. **Goal** contains no guessed guild goal. It records the operator's explicit
   selected alternative only after selection.
5. **Health** contains typed findings, coverage limitations, contradictions,
   confidence, and staleness.

Every assertion carries source, observed-at, confidence, staleness,
contradiction state, privacy class, schema version, and digest. Derived facts
identify their deterministic rule.

## Analysis and planning

Permission calculation uses Discord-compatible bit arithmetic over closed
metadata, but never calls Discord. Findings are stable codes with severity,
affected opaque references, evidence digests, and deterministic explanation
codes. Alternative ordering and identifiers are stable. At minimum the engine
offers a least-privilege alternative, a focused structural/overwrite repair,
and do-nothing. No alternative is selected implicitly.

An explicit operator selection produces a desired-state plan containing its
source observation digest, preconditions, typed desired-state deltas, expected
postconditions, and a rollback preview restoring the observed state. The plan
is data only: there is no execute/apply/approve/compensate method or transport.

## Redacted status

Status uses a fixed schema and fixed vocabulary. It may report schema version,
evidence level, read-only mode, source kind, freshness, coverage counts,
finding/alternative counts, whether an option was selected, whether a plan was
produced, and exclusion flags. It never includes guild/user/channel names,
opaque references, findings prose, content, or credentials.

## Acceptance evidence

| Check | Required local evidence | Explicitly not proved |
| --- | --- | --- |
| C1 schema | compile, closed-deserialization and limit tests | Discord compatibility |
| C1 fixture authorization | recorded owner/admin accept; lower authority and owner-reference mismatch reject | P2 grant or real operator identity |
| C1 permissions | deterministic unit vectors and unknown-target behavior | live Discord state |
| C2 replay | two synthetic replays serialize identically | network/provider behavior |
| C2 planning | alternatives, explicit selection, rollback preview tests | executable change |
| Boundary | static guard plus full Abbey gate | absence in unrelated programs |

Production, canary, or live Discord validation is a later, separately
authorized activity and must be reported separately from this local evidence.
