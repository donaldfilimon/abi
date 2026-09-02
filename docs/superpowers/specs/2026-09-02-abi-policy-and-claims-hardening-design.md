# ABI Policy and Claims Hardening Design

**Approved:** 2026-09-02

**Repository:** `donaldfilimon/abi`

**Delivery:** canonical `main`, normal push, exact-head CI and Pages verification

## Goal

Close the current CI-policy, sibling-WDBX, public-claims, published-site, and
agent-guidance gaps without changing ABI's Rust public APIs, wire formats,
frozen CLI/MCP catalogs, or WDBX source.

## Invariants

- Every `actions/checkout` step has a step-local `with` mapping containing
  `persist-credentials: false`; a missing mapping or key fails policy.
- CI sibling checkout requirements are derived from the workspace manifest and
  use immutable revisions at their canonical sibling paths.
- The full-FHE helper invokes the sibling WDBX manifest and fails clearly when
  that manifest is absent.
- Public documentation describes the required sibling WDBX layout and uses
  source-derived corpus evidence. Capability wording stays within current
  source, test, runtime, and hosted evidence.
- Published-site references cannot use protocol-relative URLs or escape the
  site root through normalized relative traversal.
- Cross-CLI skill copies are generated mirrors of their canonical source and
  are never edited independently.

## Five delivery cycles

1. Harden the CI contract and its missing-`with`, missing-key, truthy, quoted,
   and multiple-checkout regressions.
2. Route full-FHE checks through `../wdbx/Cargo.toml` without modifying WDBX.
3. Reconcile WDBX paths, public capability wording, and the current contract
   corpus count across documentation and the task ledger.
4. Validate the exact Pages artifact, local-reference containment, and
   benchmark input before rendering; exercise it in a local browser.
5. Compact canonical agent guidance and verify cross-CLI skill mirror parity.

## Verification and publication

Run focused Python policy tests, the in-memory `abi-mcp` suite, the real
full-FHE helper, full Python discovery, and `./tools/check.sh`. Review the
complete base-to-head diff, push `main` without force, wait for exact-head CI,
then verify the deployed Pages artifact independently from local source gates.
