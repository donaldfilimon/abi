# Program 1 Abbey Contract Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task and superpowers:test-driven-development for every production change. The corpus is a data contract, not permission to activate production federation or live Discord.

**Goal:** Deliver the canonical, language-neutral Program 1 `abbey-contracts` v1 corpus in ABI with closed schemas, synthetic fixtures, deterministic digests, an independent Rust verifier, and strict privacy/compatibility gates.

**Architecture:** Raw UTF-8 JSON schemas and fixtures live under `contracts/abbey/` and have no Cargo dependency. A standard-library Python tool validates the manifest, file digests, fixture taxonomy, JSON duplicate members, reference locality, and privacy sentinels. A small nightly Rust crate independently computes the aggregate digest and validates/decode-classifies the same artifacts. ABI's full gate runs both implementations before compiling the workspace.

**Tech Stack:** JSON Schema Draft 2020-12, Python 3 standard library plus existing environment validation, Rust nightly 2024, `serde_json`, `jsonschema`, `sha2`, RFC 8785-compatible bounded canonical JSON profile.

**Spec:** `docs/superpowers/specs/2026-08-22-federation-reconciliation-and-abbey-contracts.md`.

## Execution record

This table is the authoritative execution status. The unchecked procedural
steps below preserve the planned RED/GREEN sequence; they are not an unfinished
work ledger.

| Task | Status | Evidence |
| --- | --- | --- |
| 1. Corpus and tooling scaffold | Complete | `ae5fa0b`; observed import and incomplete-manifest REDs before GREEN |
| 2. Identity, tenancy, and scope | Complete | `de44718`; schema and delegation behavior REDs before GREEN |
| 3. Authorization and approval | Complete | `17002bb`; prohibited-grant, self-approval, and degradation REDs before GREEN |
| 4. Execution and receipts | Complete | `7654483`; lifecycle, cancellation, idempotency, and privacy REDs before GREEN |
| 5. Voice consent evidence | Complete | `9832462`; consent-transition and operator-report REDs before GREEN |
| 6. WDBX and learning boundaries | Complete | `2d7085f`; evidence, retention, default-off, and QUIET REDs before GREEN |
| 7. Independent Rust verifier | Complete | `ba520ef`; missing-crate RED before independent digest/schema/fixture GREEN |
| 8. Compatibility and closeout | Complete | This closeout commit; companion ownership headers reconciled manually, mutation checks and full strict gate passed |

The planned prose header-consistency test was intentionally not added because
the execution instruction prohibited prose-grep and change-detector tests.
Header ownership was reviewed directly; corpus acceptance remains enforced by
behavioral schema, fixture, digest, privacy, and mutation tests.

## Global Constraints

- Every authority-bearing object has `additionalProperties: false`; tolerant content/event metadata uses one explicit `extensions` map and never consults it for authority.
- Every string, array, map, and encoded object is bounded before retention.
- Fixtures contain synthetic opaque IDs and counters only—no real messages, prompts, transcripts, audio, participant/user/channel/guild identifiers, credentials, paths, vectors, or WDBX payloads.
- JSON transport bytes are not WDBX canonical episode commitments.
- `Prohibited` is decodable but ungrantable. `Unset` learning is disabled. Missing local HTTP auth fails closed.
- A mismatched corpus disables consequential work; a developer diagnostic mode may only read and report mismatch.
- Program 1 can reach only C1 in this implementation. It cannot claim live authorization, production federation, installed-artifact qualification, or consented Discord evidence.
- Each test is observed failing for the expected reason before its implementation is added.

---

## Task 1: Scaffold the data-only corpus and tooling contract

**Files:**
- Create: `contracts/abbey/README.md`
- Create: `contracts/abbey/compatibility.md`
- Create: `contracts/abbey/manifest.json`
- Create: `tools/abbey_contracts.py`
- Create: `tools/tests/test_abbey_contracts.py`
- Modify: `tools/check.sh`

**Interfaces:**
- `load_json_strict(path: Path) -> object` rejects duplicate members, non-finite numbers, invalid UTF-8, and unbounded files.
- `discover_artifacts(root: Path) -> tuple[Path, ...]` returns normalized, sorted relative paths and rejects symlinks/path escape.
- `verify_manifest(root: Path) -> VerificationReport` verifies per-file and aggregate commitments without rewriting files.
- CLI: `python3 tools/abbey_contracts.py verify contracts/abbey`.

- [ ] **Step 1: Write failing empty-corpus and traversal tests**

```python
class CorpusBoundaryTests(unittest.TestCase):
    def test_corpus_has_no_symlink_or_path_escape(self):
        with self.assertRaisesRegex(ContractError, "symlink is forbidden"):
            discover_artifacts(self.fixture("symlink-corpus"))

    def test_manifest_lists_every_normative_artifact_once(self):
        report = verify_manifest(Path("contracts/abbey"))
        self.assertEqual(report.unlisted, ())
        self.assertEqual(report.missing, ())
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tools.tests.test_abbey_contracts.CorpusBoundaryTests -v`

Expected: import failure because `tools.abbey_contracts` does not exist.

- [ ] **Step 3: Implement strict loading, normalized discovery, and an intentionally incomplete manifest**

Use only explicit path roots, `Path.relative_to`, `lstat`, and JSON `object_pairs_hook` duplicate rejection. Reject files over 1 MiB and the aggregate corpus over 16 MiB. Never follow symlinks. The first `manifest.json` deliberately has an empty artifact list so the listing test fails on `README.md` and `compatibility.md`.

- [ ] **Step 4: Verify the intended manifest RED**

Run: `python3 -m unittest tools.tests.test_abbey_contracts.CorpusBoundaryTests -v`

Expected: import succeeds and the completeness test fails with stable unlisted relative paths.

- [ ] **Step 5: Implement manifest generation as a review-only command**

Add `build-manifest` that writes only when `--write` is explicitly present. Each artifact row has `path`, `bytes`, `media_type`, optional `schema_id`, and lower-hex `sha256`. The manifest has `contract_major`, `contract_revision`, `algorithm`, `redaction_profile`, `artifacts`, and `aggregate_digest`.

Aggregate input is:

```text
abbey-contract-corpus-v1\0
<path UTF-8>\0<decimal byte length>\0<lower SHA-256>\n
```

for lexicographically sorted artifacts. During self-commitment, parse the manifest, replace only `aggregate_digest` with 64 ASCII zeroes, serialize with the repository's fixed JSON formatting, and include those bytes. The independent verifier vector in Task 7 ensures this algorithm is not self-consistently wrong.

- [ ] **Step 6: Add verification to ABI's gate**

Insert before Cargo stages:

```bash
python3 -m unittest discover -s tools/tests -p 'test_*.py' -v
python3 tools/abbey_contracts.py verify contracts/abbey
```

- [ ] **Step 7: Verify GREEN and commit**

Run:

```bash
python3 tools/abbey_contracts.py build-manifest contracts/abbey --write
python3 -m unittest tools.tests.test_abbey_contracts.CorpusBoundaryTests -v
python3 tools/abbey_contracts.py verify contracts/abbey
git diff --check
```

Commit:

```bash
git add contracts/abbey tools/abbey_contracts.py tools/tests/test_abbey_contracts.py tools/check.sh
git commit -m "feat(contracts): scaffold Abbey v1 corpus"
```

## Task 2: Common identity, tenancy, and scope schemas

**Files:**
- Create: `contracts/abbey/v1/schemas/common/definitions.schema.json`
- Create: `contracts/abbey/v1/schemas/identity/principal.schema.json`
- Create: `contracts/abbey/v1/schemas/identity/delegation-chain.schema.json`
- Create: `contracts/abbey/v1/schemas/authorization/scope.schema.json`
- Create: `contracts/abbey/v1/fixtures/{valid,invalid,boundary,privacy}/identity-*.json`
- Modify: `tools/tests/test_abbey_contracts.py`

**Schemas:**
- Opaque IDs: `^[a-z][a-z0-9_-]{0,63}$`; digest IDs: `^sha256:[0-9a-f]{64}$`.
- RFC 3339 UTC instants with `Z`; finite duration milliseconds as decimal strings bounded to 86,400,000.
- `Principal` separates channel/workload from subject and uses a closed `kind` enum.
- `PlatformScope` is a `oneOf` tagged union; no nullable guild and no wildcard.

- [ ] **Step 1: Write failing schema compilation and taxonomy tests**

Require every schema to use Draft 2020-12, an absolute `https://abbey.local/contracts/...` ID, local-only `$ref`, closed objects, and declared `x-abbey-data-class`, `x-abbey-max-bytes`, and `x-abbey-unknown-fields` metadata.

Run: `python3 -m unittest tools.tests.test_abbey_contracts.SchemaContractTests -v`

Expected: missing schemas and classifications fail.

- [ ] **Step 2: Implement common and identity schemas**

Reject blank IDs, raw numeric Discord snowflakes, display names, credentials, filesystem-looking strings, cycles, repeated delegation IDs, non-narrowing scope, absent expiry, and more than eight delegation hops.

- [ ] **Step 3: Add valid, invalid, boundary, and privacy fixtures**

Each fixture envelope contains exactly `case_id`, `schema`, `expect`, and `document`; invalid fixtures declare a closed `reason_code`, not an expected free-form validator message.

- [ ] **Step 4: Verify and commit**

Run:

```bash
python3 tools/abbey_contracts.py build-manifest contracts/abbey --write
python3 -m unittest tools.tests.test_abbey_contracts -v
python3 tools/abbey_contracts.py verify contracts/abbey
```

Commit: `feat(contracts): define identity and tenancy boundaries`.

## Task 3: Capability, grant, policy, approval, and closed errors

**Files:**
- Create: `contracts/abbey/v1/schemas/capability/package.schema.json`
- Create: `contracts/abbey/v1/schemas/authorization/grant.schema.json`
- Create: `contracts/abbey/v1/schemas/authorization/policy-decision.schema.json`
- Create: `contracts/abbey/v1/schemas/authorization/approval.schema.json`
- Create: `contracts/abbey/v1/schemas/error/error.schema.json`
- Create: `contracts/abbey/v1/fixtures/{valid,invalid,boundary,privacy,degraded}/authorization-*.json`
- Modify: `tools/tests/test_abbey_contracts.py`

- [ ] **Step 1: Write failing invariant tests**

Assert `RiskClass` ordering is fixed, `Prohibited` cannot validate as a grant, approvals bind exact request/capability/grant/policy/scope/effect digests, approvals are single-use and expiring, and error payloads admit only closed reason codes plus bounded correlation/version metadata.

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tools.tests.test_abbey_contracts.AuthorizationInvariantTests -v`

Expected: missing schema/fixture failure.

- [ ] **Step 3: Implement closed schemas and cross-document semantic checks**

JSON Schema handles wire shape; `tools/abbey_contracts.py` performs finite semantic invariants that Draft 2020-12 cannot express, keyed by schema ID. It must return only reason codes and artifact paths, never offending values.

- [ ] **Step 4: Add hostile fixtures**

Cover unknown authority fields, wildcard scope, anonymous consequential grant, prohibited risk, self-approval, expiry omission, mismatched digest, over-bound requirements, embedded free-form cause, and dependency degradation that never increases authority.

- [ ] **Step 5: Verify, rebuild manifest, and commit**

Run full Python contract tests and verifier. Commit: `feat(contracts): close authorization and approval envelopes`.

## Task 4: Request, response, event, cancellation, proposals, and receipts

**Files:**
- Create: `contracts/abbey/v1/schemas/cognition/request.schema.json`
- Create: `contracts/abbey/v1/schemas/cognition/response.schema.json`
- Create: `contracts/abbey/v1/schemas/event/metadata-event.schema.json`
- Create: `contracts/abbey/v1/schemas/event/cancellation.schema.json`
- Create: `contracts/abbey/v1/schemas/capability/recommendation.schema.json`
- Create: `contracts/abbey/v1/schemas/capability/action-proposal.schema.json`
- Create: `contracts/abbey/v1/schemas/capability/execution-request.schema.json`
- Create: `contracts/abbey/v1/schemas/receipt/outcome-receipt.schema.json`
- Create: `contracts/abbey/v1/fixtures/{valid,invalid,boundary,unknown-field,privacy,cancellation,degraded}/execution-*.json`
- Modify: `tools/tests/test_abbey_contracts.py`

- [ ] **Step 1: Write failing terminality and redaction tests**

Assert exactly one terminal response state, ordered metadata events, first-class cancellation reference, idempotency for effects/durable writes, exact proposal digest on execution, bounded step counts, and no content-bearing receipt fields.

- [ ] **Step 2: Implement schemas and semantic checks**

Authority objects reject unknown fields. Metadata events may preserve only an `extensions` object whose values are bounded JSON scalars; extension values are never copied into policy/grant/receipt decisions.

- [ ] **Step 3: Add cancellation race fixtures**

Cover cancellation-before-start, cancellation-during-provider, cancellation-during-actuator with `outcome_unresolved`, deadline expiry, partial rollback, and stale/mismatched cancellation reference.

- [ ] **Step 4: Verify and commit**

Rebuild manifest, run the full Python suite and verifier, and commit `feat(contracts): define execution and receipt lifecycle`.

## Task 5: Consent epoch and operator-verification report

**Files:**
- Create: `contracts/abbey/v1/schemas/consent/epoch.schema.json`
- Create: `contracts/abbey/v1/schemas/consent/transition.schema.json`
- Create: `contracts/abbey/v1/schemas/consent/barge-in.schema.json`
- Create: `contracts/abbey/v1/schemas/consent/operator-verification-report.schema.json`
- Create: `contracts/abbey/v1/fixtures/{valid,invalid,boundary,privacy,cancellation,degraded}/consent-*.json`
- Modify: `tools/tests/test_abbey_contracts.py`

- [ ] **Step 1: Write failing consent state-machine tests**

Valid transitions are only `Closed -> PendingAttestation -> Open -> Closing -> Closed`. Opening requires authorized manager, all-current-participant consent, exact participant-set digest, positive aggregate count, and current timestamps. Participant changes, unidentified participants, authorization/attestation loss, connection loss, and stop close the epoch and cancel epoch-bound stages. Barge-in cancels playback/stale downstream work but does not close consent.

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tools.tests.test_abbey_contracts.ConsentContractTests -v`

Expected: missing consent artifacts.

- [ ] **Step 3: Implement schemas and transition validation**

The verification report exposes only build/revision, authorization result, epoch/participant-change/stage/cancellation/pause/resume/leave counters, bounded durations, terminal status, and evidence classification `local_test|installed_artifact|live_discord`. It has no identity, guild, channel, message, audio, transcript, response, credential, or path field.

- [ ] **Step 4: Add the complete synthetic operator flow fixture**

One valid fixture observes authorization, initial epoch open, decoded receive, STT completion, synthesis completion, playback completion, barge-in cancellation, participant-change close/pause, newly authorized and consented epoch resume, and final leave. Its evidence classification is `local_test`, never `live_discord`.

- [ ] **Step 5: Verify privacy and commit**

Add sentinel traversal proving the report and errors do not contain real-content classes. Rebuild/verify manifest and commit `feat(contracts): specify consented voice verification evidence`.

## Task 6: Episode, evidence, claim, retention, and learning schemas

**Files:**
- Create: `contracts/abbey/v1/schemas/episode/proposal.schema.json`
- Create: `contracts/abbey/v1/schemas/episode/evidence.schema.json`
- Create: `contracts/abbey/v1/schemas/episode/claim.schema.json`
- Create: `contracts/abbey/v1/schemas/episode/tombstone.schema.json`
- Create: `contracts/abbey/v1/schemas/learning/guild-learning-policy.schema.json`
- Create: `contracts/abbey/v1/schemas/learning/promotion-candidate.schema.json`
- Create: `contracts/abbey/v1/fixtures/{valid,invalid,boundary,privacy,degraded}/episode-*.json`
- Create: `contracts/abbey/v1/fixtures/{valid,invalid,boundary,privacy,degraded}/learning-*.json`
- Modify: `tools/tests/test_abbey_contracts.py`

- [ ] **Step 1: Write failing ownership and learning tests**

Assert adapters submit proposals but never claim a canonical episode digest; integrity/provenance/semantic truth remain separate; tombstones and correction/contradiction/supersession/quarantine links are typed; `Unset` and `ExplicitDisabled` deny adaptive update; `ABBEY_QUIET` overrides unsolicited action; learning messages cannot carry grants, approvals, command registration, safety-policy mutation, or direct platform writes.

- [ ] **Step 2: Implement closed schemas**

Episode payload is a bounded reference or deliberately empty redacted summary, never embedded content. Claims carry separate class, capability state, evidence level, permitted conclusion, version/environment, evidence/contradiction references, expiry, and rollback condition.

- [ ] **Step 3: Add MandatoryIncident minimization fixtures**

Prove mandatory classification bypasses discretionary utility only while still requiring scope, minimization, redaction, hold, retention, and deletion-key validation.

- [ ] **Step 4: Verify and commit**

Rebuild/verify manifest and commit `feat(contracts): define WDBX and learning boundaries`.

## Task 7: Independent Rust verifier and canonical digest vectors

**Files:**
- Create: `crates/abi-contracts/Cargo.toml`
- Create: `crates/abi-contracts/src/lib.rs`
- Create: `crates/abi-contracts/tests/corpus.rs`
- Create: `contracts/abbey/v1/fixtures/valid/corpus-digest-vector.json`
- Create: `contracts/abbey/v1/fixtures/valid/jcs-vector.json`
- Create: `contracts/abbey/v1/fixtures/invalid/jcs-duplicate-member.json`
- Create: `contracts/abbey/v1/fixtures/boundary/jcs-number-domain.json`
- Modify: `Cargo.toml`

**Interfaces:**
- `Corpus::open(path: impl AsRef<Path>) -> Result<Corpus, ContractError>`.
- `Corpus::verify() -> Result<VerifiedCorpus, ContractError>`.
- `canonicalize_jcs(schema_family: &str, major: u32, value: &Value) -> Result<Vec<u8>, ContractError>`.
- `VerifiedCorpus::validate_fixture(path: &Path) -> FixtureOutcome`.

- [ ] **Step 1: Write failing independent-digest tests**

The Rust test reads raw manifest bytes and artifacts, independently recomputes every SHA-256 and aggregate digest, and compares the frozen vector. It must not call Python or reuse generated output.

- [ ] **Step 2: Verify RED**

Run: `./tools/cargo.sh test -p abi-contracts --test corpus independent_digest -- --exact`

Expected: package/test absence.

- [ ] **Step 3: Implement bounded Rust corpus loader**

Reject absolute/backslash/parent paths, symlinks, duplicate paths, files over 1 MiB, corpus over 16 MiB, digest mismatch, unknown manifest keys, duplicate JSON members, external `$ref`, and schema compilation errors. Error display contains only closed code and normalized corpus-relative path.

- [ ] **Step 4: Implement `abbey-jcs-v1` vectors**

Use explicit domain prefix `abbey-jcs-v1:<schema-family>:<major>\0`. Reject numbers outside the documented safe integer/finite decimal domain before canonicalization. Pin Unicode, key-order, escaping, duplicate-member, negative-zero, and boundary cases. Do not treat ordinary `serde_json::to_vec` as canonical evidence.

- [ ] **Step 5: Run Rust and Python implementations together**

Run:

```bash
python3 tools/abbey_contracts.py build-manifest contracts/abbey --write
python3 tools/abbey_contracts.py verify contracts/abbey
./tools/cargo.sh test -p abi-contracts --all-targets
```

Expected: both implementations agree on every per-file and aggregate digest and all fixtures.

- [ ] **Step 6: Commit**

Commit `feat(contracts): add independent Rust corpus verifier`.

## Task 8: Compatibility policy, companion headers, and strict closeout

**Files:**
- Modify: `contracts/abbey/README.md`
- Modify: `contracts/abbey/compatibility.md`
- Modify: `docs/superpowers/specs/2026-08-22-spec-capability-authorization-kernel.md`
- Modify: `docs/superpowers/specs/2026-08-22-spec-discord-guild-intelligence-and-execution.md`
- Modify: `docs/superpowers/specs/2026-08-22-spec-canonical-wdbx-episodes.md`
- Modify: `docs/superpowers/specs/2026-08-22-spec-guild-world-model-and-arbiter.md`
- Modify: `docs/superpowers/specs/2026-08-22-spec-application-federation.md`
- Modify: `docs/superpowers/specs/2026-08-22-spec-learning-evaluation-promotion.md`
- Modify: `docs/superpowers/specs/2026-08-22-federation-reconciliation-and-abbey-contracts.md`
- Modify: `docs/superpowers/plans/2026-08-22-program-1-abbey-contract-corpus.md`

- [ ] **Step 1: Write failing header-consistency test**

Add a Python test that requires each companion spec to cite its constitutional slug/number without renumbering, identify Program 1 as `abbey-contracts`, and describe learning/evaluation/promotion as cross-cutting rather than Program 8.

- [ ] **Step 2: Verify RED**

Run the focused header test and observe exact stale documents.

- [ ] **Step 3: Reconcile headers and document compatibility**

Document breaking-major versus additive-revision policy, tolerant versus strict unknown-field policy, digest mismatch refusal, vendoring rules, developer read-only degradation, extraction triggers, and rollback to the last qualified corpus. Do not rewrite substantive P2-P7 designs.

- [ ] **Step 4: Run the complete fresh ABI gate**

Run: `./tools/check.sh`

Expected: Python workflow/corpus/header tests, corpus verifier, format, clippy `-D warnings`, build, all workspace tests, available model feature checks, benchmark guard, and warning-denied docs all pass.

- [ ] **Step 5: Verify artifact/privacy completeness**

Run:

```bash
python3 tools/abbey_contracts.py verify contracts/abbey
rg -n '\b(TBD|TODO)\b|implement later|fill in' contracts/abbey
git diff --check
git status --short --branch
```

Expected: no unresolved implementation markers, no unmanifested artifacts, and only intended branch changes.

- [ ] **Step 6: Commit closeout evidence**

Commit `docs(contracts): qualify Program 1 source corpus` with exact C1 evidence. Explicitly retain C2-C7, production, and live Discord as unperformed.
