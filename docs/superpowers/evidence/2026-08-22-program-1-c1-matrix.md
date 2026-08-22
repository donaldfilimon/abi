# Program 1 C1 conformance matrix

Date: 2026-08-22

This is a redacted source-and-contract evidence report. It contains repository
and immutable revision identities, test counts, and closed evidence states. It
does not contain local filesystem paths, credentials, Discord identifiers,
message content, prompts, responses, transcripts, audio, WDBX values, or
participant identity.

## Canonical corpus identity

- Source repository: `https://github.com/donaldfilimon/abi`
- Qualified source revision: `348754bdaaf59a40fbb858380f925e0aba95a23b`
- Contract major/revision: `1` / `1`
- Aggregate digest: `72e241e34967df318376bf68f4a0e2db13f5ebf17d1a219709731f1f470dbe8e`
- Manifest-listed artifacts: 81
- Manifest-listed artifact bytes: 88,328
- Schemas: 27
- Fixtures: 52

ABI's deterministic vendoring tool completed a read-only `--check` against
each of the four consumer trees. Every consumer contained exactly the 81
manifest-listed files with byte equality and the same aggregate digest. No
consumer tree was rewritten during this comparison.

## Exact-revision matrix

| Repository | Exact tested head | Native toolchain | Focused evidence | Full local gate | Hosted exact-head/default-branch state |
| --- | --- | --- | --- | --- | --- |
| `donaldfilimon/abi` | `348754bdaaf59a40fbb858380f925e0aba95a23b` | Rust nightly `1.100.0-nightly (f7d782a3b 2026-08-19)` plus Python | 12/12 deterministic-vendoring behavior tests; authoritative corpus tests and an independent Rust verifier | `./tools/check.sh` passed at the exact source head | [PR #803](https://github.com/donaldfilimon/abi/pull/803) merged as `a3b9bfd1980eeff021816146c72d7a65c7d5aadf`; exact-head self-hosted CI, Windows ACL, CodeQL, GitGuardian, review, and preview checks passed. Main CI and dependency scan passed. Pages failed on Liquid parsing of a literal GitHub-expression example; the follow-up closeout branch contains the raw-block fix and has not yet established hosted Pages success. |
| `donaldfilimon/wdbx` | `b3777fbad43a9dee2d8aa4d9612f88ac33d82d2e` | Rust nightly `1.100.0-nightly (c656540d6 2026-08-21)` | 9/9 native WDBX-owned episode-family conformance and mutation tests | format and clippy passed; 569/569 workspace tests passed | [PR #1](https://github.com/donaldfilimon/wdbx/pull/1) merged as `4963bdc5f5d591492b510fa539df218efc9b8c6c`; exact-head hosted gate and review passed. The first main run hit the pre-existing transient `WriterBusy` lock-retry failure; rerun attempt 2 passed format, clippy, and the complete test job. |
| `donaldfilimon/abbey` | `e4418b5b99c502f5b2fbfdc58e835f92b171c16c` | Rust nightly `1.100.0-nightly (e71c0f1e3 2026-08-18)` | 3/3 corpus integration tests plus the C1 claim test; all 52 declared fixtures matched | `./check.sh` passed default, WDBX, personal, accelerator, clippy, rustdoc, claim/docs, Metal parity, install, and rollback smoke | [PR #89](https://github.com/donaldfilimon/abbey/pull/89) merged as `43317c7b49a9e9e915d68129e4b308926e300b84`. Its macOS adjunct failed because daemon fixtures inherited the gate's ambient ABI provider and gained extra model authority. Local follow-up `ac446fd81a061606c57ed2333c7ce2b0affd28be` reproduced all three affected cases, isolates the test environment, and passed the exact-head full gate; it has no post-fix hosted run. |
| `donaldfilimon/abbey-bot` | `c5a45a6e3bfd33cc2e9cd3283fbf72d55fac34d2` | stable Rust `1.97.1 (8bab26f4f 2026-07-14)` | 7/7 Python byte/privacy guard tests and 7/7 native stable-Rust tests; all 52 fixtures matched; synthetic operator report classified `local_test` | `./check.sh` passed; 640 Rust tests passed, 0 failed, 2 ignored; clippy and locked release build passed | [PR #38](https://github.com/donaldfilimon/abbey-bot/pull/38) merged as `6cba85659946165ae110b996c0c57960775b200c`; Ubuntu and macOS passed, while Windows converted a manifest-bound README to CRLF and failed its byte length. Local follow-up `3d499f3bc14d607b3a3ab8d2db9b18f1047ca3a9` adds an autocrlf checkout regression and byte-preserving attributes, then passed the exact-head full gate; it has no post-fix hosted Windows run. The already-merged live-voice verifier has separate earlier provider evidence and is not evidence for this conformance head. |
| `donaldfilimon/AbbeyBot` | `f92704f99d514f6a60463c8813b2935fbe9e56cb` | Apple Swift `6.5-dev (1f6442f1d1fcbb7)` | 9/9 native corpus tests; all 52 fixtures matched; exact-pin, duplicate-member, byte, and recomputed-identity mutations failed closed | `Scripts/verify-all.sh` passed static/security, package graphs, web 17/17, all Swift targets, signed-app launch, server smoke, and CLI smoke | Local branch only; no exact-head hosted run, push, PR, or merge is claimed. Docker/Postgres was unavailable; the designed SQLite fallback passed. |

## Native behavior boundary

- ABI is the corpus authority and deterministic byte-vendoring source.
- WDBX validates only the episode, evidence, claim, tombstone/retention, and
  canonicalization boundary it owns. It does not open or mutate a durable
  store during conformance.
- Nightly Abbey validates the complete corpus but exposes no production
  authorization, consent, execution, or memory path from this task.
- Stable Rust Abbey bot validates the complete corpus in test-only code. The
  synthetic operator flow is `local_test` evidence and is not wired into live
  Discord execution.
- Swift AbbeyBot validates the complete corpus and exact pin with native Swift
  code, including duplicate-member and recomputed-identity rejection. Its
  fail-closed local HTTP API authentication is a separate stacked commit below
  the conformance head.

The approved corpus has no positive `abbey-cbor-episode-v1` golden fixture or
canonical CBOR bytes. WDBX therefore proves the available negative boundary:
transport JSON and adapter projections are not canonical durable episode
commitments. Positive canonical-CBOR decoding remains unqualified; no protocol
or golden bytes were invented to fill that gap.

## Evidence boundary and explicit non-claims

This matrix establishes C1 source/contract conformance only. It does not
establish installed-artifact qualification, production federation, production
deployment, a real capability grant or approval, a provider or Discord effect,
a durable WDBX episode write, or C2-C7 promotion.

No production action or participant-consented live Discord session was
performed for this matrix. The live Abbey voice verifier still requires a
separately authorized session with current-participant consent and current
manager authorization. Local tests, hosted CI, installed artifacts, provider
delivery, and witnessed live Discord operation remain separate evidence rows.
