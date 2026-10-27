# Restore the dependency-security baseline

**Goal:** Remove the observed mandatory gate failure on Tokio without suppressing advisories or changing supported contracts.
**Architecture:** ABI retains its existing runtime and exact WDBX sibling cohort; a narrow lockfile update selects a patch that resolves both observed Tokio advisories.
**Spec:** Approved product modernization milestone 2, compatible contracts and reproducible baseline. Baseline mandatory scanner rejected `tokio 1.53.1` for `RUSTSEC-2026-0386`.
**Tech stack:** Rust 1.99.1, repository `tools/cargo.sh`, Cargo lockfile and required security scanner.

## Global Constraints

- Preserve original checkouts and all source work. Use this isolated task checkout and its pinned WDBX sibling.
- No advisory suppression, disabling scanners, relaxing locked gates, dependency-source substitution or unrelated broad update.
- Preserve existing CLI/MCP fixtures, permission boundaries, API behavior and repository instructions.
- A dependency scan pass is not an implementation, platform or production qualification claim.
- Use repository wrappers, stdin at EOF and isolated test stores/build output. Never use live stores.

## Task 1: Update the affected Tokio pin and verify the cohort

**Owned files:** `Cargo.lock` only, plus an optional additive `docs/verification/2026-10-25-dependency-cohort.md` receipt if all statements have exact evidence. Do not edit source code, manifests, other repositories, frozen receipts or security exceptions. Report subsequent unrelated baseline failures for a separate task.

**Affected requirements:** ABI C01 and C42 (partial contributions; no full-row qualification).

**Baseline:** ABI snapshot `3fea36fd9ae51704693cb740e2e10f1a2766ab9f`; WDBX snapshot `6372def9afbccfb3d6e72b00cb22904cef76618f` lives at required adjacent `../wdbx`. Abbey and WDBX Studio already use Tokio `1.54.0`; Bot uses patched `1.53.3`. The local current advisory says patched `>= 1.53.3` or the separate `1.51.6` line. The attempted cohort target `1.54.0` was independently rejected for `RUSTSEC-2026-0384` and yanked status; the current advisory requires `>= 1.54.1`. The reviewed execution ruling selects `1.54.1` instead. Preserve the failed attempt as evidence. Keep existing accepted TFHE exception scope unchanged.

**Steps:**

- [ ] Read AGENTS and applicable dependency/claims instructions. Read the advisory and root gate baseline evidence supplied by the controller.
- [ ] Update precisely with `./tools/cargo.sh update -p tokio --precise 1.54.1`. Inspect lockfile diff; only Tokio and strictly necessary dependency resolution may change. No broad dependency upgrade.
- [ ] Run the required dependency scan and confirm this advisory no longer fails. A network/advisory database error is not a passing scan.
- [ ] Run `./tools/check.sh < /dev/null` with isolated `CARGO_TARGET_DIR`, `ABI_WDBX_PATH=:memory:` and `ABI_WDBX_PERSIST` unset. Persistence-specific tests own isolated scratch stores; globally disabling persistence would invalidate their audit assertions. Use `env -u ABI_WDBX_PERSIST` for this gate. Preserve any subsequent failure with its exit/stage instead of changing unrelated source or claiming full green.
- [ ] Check the final lockfile identity and `git diff --check`, self-review, commit owned changes and report exact commands/results. No test is needed that merely asserts the lockfile version; the real scanner/build/gate is the behavioral validation.

**Rollback:** Revert the task commit to restore the old reproducible lockfile; the previously observed security gate failure then remains. No live service or store is touched.
