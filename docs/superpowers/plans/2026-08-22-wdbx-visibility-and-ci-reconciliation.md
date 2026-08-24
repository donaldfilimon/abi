# WDBX Visibility and CI Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Every source change follows superpowers:test-driven-development. Do not treat a provider setting, local gate, hosted run, production deployment, or live Discord observation as interchangeable evidence.

**Goal:** Make the canonical WDBX substrate publicly readable, remove obsolete secret-only dependency checkouts, pin the current immutable substrate revision, and restore executable trusted and fork-safe CI for ABI and `abbey`.

**Architecture:** Repository visibility changes once at the GitHub provider boundary. ABI and `abbey` continue consuming WDBX only through sibling path dependencies, while CI checks out the public repository at an immutable revision into the required sibling path. Static workflow contract tests fail before YAML changes, and each repository's documented strict gate plus observed GitHub jobs qualify only the exact committed head.

**Tech Stack:** GitHub CLI/API, GitHub Actions YAML, Python 3 `unittest` workflow guards, Rust nightly gates, immutable Git revisions.

**Spec:** `docs/superpowers/specs/2026-08-22-federation-reconciliation-and-abbey-contracts.md`, especially sections 8 and 11.

## Global Constraints

- The visibility target is exactly `donaldfilimon/wdbx`, which current provider evidence identifies as the Rust substrate after rename commit `f42b9789eabcf89f952df0a160a7b6837c5acb57`. The former unrelated Workers repository is now `donaldfilimon/wdbx-workers-legacy` and remains untouched.
- Before mutation, verify local WDBX `main` equals `origin/main`, is clean, and inspect provider visibility and default branch.
- Public visibility authorizes source readability only. It does not publish credentials, runtime data, user content, audio, transcripts, or WDBX stores.
- Keep ABI, `abbey`, and WDBX as sibling path dependencies. Do not mix path and Git sources for the same crate identity.
- Pin WDBX to `f42b9789eabcf89f952df0a160a7b6837c5acb57` only after verifying that commit is current public `main` and anonymously readable.
- Never expose a self-hosted runner to fork code. Trusted self-hosted jobs retain exact same-repository event guards; forks use GitHub-hosted runners.
- Do not delete repository secrets as part of this plan. Remove references made obsolete by public checkout, then report unused secret cleanup separately.
- Production deployment and live Discord are outside this plan and remain separately authorized.

---

## Task 1: Capture pre-change authority and make WDBX public

**Files:**
- Modify: GitHub repository setting for `donaldfilimon/wdbx`
- Modify: `/Users/donaldfilimon/dev/active/wdbx/AGENTS.md`
- Modify: `/Users/donaldfilimon/dev/active/wdbx/README.md`

**Evidence:**
- Produces a before/after provider record containing repository identity, visibility, default branch, and current immutable head only.
- Produces anonymous HTTPS read evidence for the exact target revision.

- [x] **Step 1: Verify the exact local and provider target**

Run:

```bash
git -C /Users/donaldfilimon/dev/active/wdbx status --short --branch
git -C /Users/donaldfilimon/dev/active/wdbx rev-parse HEAD origin/main
gh repo view donaldfilimon/wdbx --json nameWithOwner,visibility,defaultBranchRef,url
gh repo view donaldfilimon/wdbx-workers-legacy --json nameWithOwner,visibility,url
```

Expected: the substrate is clean at `f42b9789eabcf89f952df0a160a7b6837c5acb57`, provider visibility is `PRIVATE`, and the renamed legacy Workers repository is still distinct and public.

- [x] **Step 2: Change visibility at the provider boundary**

Run:

```bash
gh repo edit donaldfilimon/wdbx --visibility public --accept-visibility-change-consequences
```

Expected: command succeeds for the exact target.

- [x] **Step 3: Verify public readability independently**

Run:

```bash
gh repo view donaldfilimon/wdbx --json nameWithOwner,visibility,defaultBranchRef,url
env -u GH_TOKEN -u GITHUB_TOKEN git ls-remote https://github.com/donaldfilimon/wdbx.git f42b9789eabcf89f952df0a160a7b6837c5acb57
```

Expected: `visibility` is `PUBLIC`; anonymous `ls-remote` returns the exact commit.

- [x] **Step 4: Update WDBX documentation**

State that the GitHub source repository is public as of 2026-08-22, stores and runtime data remain private/operator-owned, and the unrelated `donaldfilimon/wdbx-workers-legacy` repository remains untouched.

The test-quality review rejected a prose-grep test here: human documentation is
not executable behavior. Provider visibility, anonymous Git readability, and
the real WDBX gate are the observable evidence.

Run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets
cargo test --workspace
git diff --check
```

Expected: metadata check and WDBX strict gate pass.

- [x] **Step 5: Commit on a WDBX feature branch**

```bash
git add AGENTS.md README.md
git commit -m "docs(wdbx): record public source boundary"
```

## Task 2: Add ABI workflow contract tests

**Files:**
- Create: `tools/tests/test_ci_contract.py`
- Modify: `tools/check.sh`

**Interfaces:**
- Produces `load_ci_text() -> str` and assertions for public checkout, immutable pin, credential absence, trusted runner isolation, fork-hosted execution, and Windows Bash declarations.

- [ ] **Step 1: Write the failing tests**

```python
class PublicWdbxWorkflowTests(unittest.TestCase):
    def test_wdbx_checkout_is_public_and_credential_free(self):
        text = load_ci_text()
        self.assertNotIn("WDBX_CHECKOUT_TOKEN", text)
        self.assertNotIn("Require the substrate credential", text)
        github_expression = "token: $" + chr(123) * 2 + " secrets."
        self.assertNotIn(github_expression, text)
        self.assertEqual(text.count("repository: donaldfilimon/wdbx"), 3)
        self.assertIn("WDBX_REVISION: f42b9789eabcf89f952df0a160a7b6837c5acb57", text)

    def test_untrusted_pull_requests_never_reach_self_hosted(self):
        text = load_ci_text()
        self.assertIn("github.event.pull_request.head.repo.full_name == github.repository", text)
        self.assertIn("github.event.pull_request.head.repo.full_name != github.repository", text)
        self.assertIn("runs-on: macos-latest", text)
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tools.tests.test_ci_contract -v`

Expected: failures identify the secret references, stale revision, and credential preflights.

- [ ] **Step 3: Add the test to the strict gate**

Insert a `workflow contract tests` step in `tools/check.sh` before Cargo compilation:

```bash
python3 -m unittest discover -s tools/tests -p 'test_*.py' -v
```

Run: `./tools/check.sh`

Expected: the gate stops at the new failing workflow contract before the expensive Rust stages.

- [ ] **Step 4: Commit the red test separately**

```bash
git add tools/tests/test_ci_contract.py tools/check.sh
git commit -m "test(ci): require public pinned WDBX checkout"
```

## Task 3: Reconcile ABI CI with public WDBX

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/self-hosted-runner.md`
- Modify: `docs/superpowers/specs/2026-08-22-federation-reconciliation-and-abbey-contracts.md`

- [ ] **Step 1: Remove obsolete credential logic**

For all three jobs, retain the sibling checkout but remove the secret preflight, private-repository prose, and `token:` input. Set `WDBX_REVISION` to `f42b9789eabcf89f952df0a160a7b6837c5acb57`. Preserve `path: wdbx`, trusted self-hosted `if:` gates, `permissions: contents: read`, fork-hosted routing, and explicit Bash on every multi-line Bash step.

- [ ] **Step 2: Verify the focused contract GREEN**

Run: `python3 -m unittest tools.tests.test_ci_contract -v`

Expected: all workflow contract tests pass.

- [ ] **Step 3: Run ABI's complete fresh gate**

Run: `./tools/check.sh`

Expected: format, workflow tests, clippy with warnings denied, build, workspace tests, platform feature checks available on the host, benchmark guard, and warning-denied docs pass. CUDA may be explicitly unavailable only when `nvcc` is absent.

- [ ] **Step 4: Record the approved/current evidence state**

Update the design status without claiming hosted success before a hosted run exists. Mark WDBX visibility and local ABI source gate with exact dates/revisions; retain GitHub Actions as `unverified` until Task 6.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml .github/self-hosted-runner.md \
  docs/superpowers/specs/2026-08-22-federation-reconciliation-and-abbey-contracts.md
git commit -m "ci: consume public pinned WDBX substrate"
```

## Task 4: Add `abbey` workflow guards in an isolated worktree

**Files:**
- Create: isolated worktree from `donaldfilimon/abbey` `origin/main`
- Modify: `tools/tests/test_workflow_guards.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `AGENTS.md`
- Modify: `src/claims.rs`

**Interfaces:**
- Tests trusted self-hosted isolation, public WDBX checkout without a secret, exact WDBX revision, and an executing GitHub-hosted fork/untrusted job.

- [ ] **Step 1: Create a clean feature worktree**

Use `superpowers:using-git-worktrees`. Create branch `cursor/public-wdbx-ci-20260822` from current `origin/main`; do not modify `/Users/donaldfilimon/dev/active/abbey`, which contains concurrent work.

- [ ] **Step 2: Write failing public-WDBX and execution-path tests**

Extend `tools/tests/test_workflow_guards.py` to assert:

```python
self.assertNotIn("WDBX_CHECKOUT_TOKEN", workflow)
self.assertNotIn("token: ${{ secrets.", workflow)
self.assertIn("WDBX_REVISION: f42b9789eabcf89f952df0a160a7b6837c5acb57", workflow)
self.assertIn("github.event.pull_request.head.repo.full_name != github.repository", workflow)
self.assertRegex(workflow, r"runs-on: (macos|ubuntu)-latest")
```

Run: `python3 -m unittest tools.tests.test_workflow_guards -v`

Expected: failures on current secret/private/pin and absent hosted execution path.

- [ ] **Step 3: Implement the safe workflow topology**

Retain trusted self-hosted jobs behind same-repository checks. Add or enable one GitHub-hosted fork/untrusted job that checks out `abbey` and public WDBX as siblings, then runs the honest portable subset or complete gate supported by that runner. Remove WDBX token preflights and secret checkout inputs. Do not schedule forks on self-hosted labels.

- [ ] **Step 4: Reconcile claims**

Update the `ci-self-hosted-linux-proof` claim only to the exact observed status. If macOS becomes executing, say so; Linux stays blocked unless an actual Linux runner/job completes. Regenerate generated claim tables with `python3 tools/check_claims_sync.py --write`.

- [ ] **Step 5: Verify focused and full gates**

Run:

```bash
python3 -m unittest tools.tests.test_workflow_guards -v
./check.sh
git diff --check
```

Expected: workflow guards and all four Abbey feature-mode gates pass; unsupported soft cross targets remain explicitly skipped, never counted as proof.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci.yml tools/tests/test_workflow_guards.py \
  AGENTS.md CLAUDE.md docs/claims.md src/claims.rs
git commit -m "ci: run Abbey against public pinned WDBX"
```

## Task 5: Configure only evidenced trusted-runner enablement

**Files:**
- Modify: GitHub repository variable only if the corresponding runner is observed online with matching labels.

- [ ] **Step 1: Inspect runner and variable state**

Run:

```bash
gh api repos/donaldfilimon/abbey/actions/runners
gh variable list --repo donaldfilimon/abbey
```

Expected: capture runner names, labels, and online/busy status without tokens.

- [ ] **Step 2: Enable only a matching runner**

If an online macOS ARM64 runner satisfies the workflow labels, run:

```bash
gh variable set ABBEY_MACOS_ARM64_RUNNER --body enabled --repo donaldfilimon/abbey
```

Do not set `ABBEY_LINUX_ARM64_RUNNER` without an observed matching online runner.

- [ ] **Step 3: Verify provider state**

Run: `gh variable list --repo donaldfilimon/abbey`

Expected: only evidenced enablement is present. If runner API access is unavailable, record the provider blocker and rely on the hosted-safe path instead.

## Task 6: Publish branches and observe exact-head CI

**Files:**
- No additional source files unless a hosted failure exposes a source defect.

- [ ] **Step 1: Rebase each branch on current upstream**

Fetch and rebase only the isolated clean branches. Rerun each repository's full gate after any rebase.

- [ ] **Step 2: Push feature branches and open reviewable PRs**

Use `superpowers:finishing-a-development-branch`. Push WDBX, ABI, and `abbey` branches; create PRs with explicit local-gate, provider-visibility, hosted-CI, production, and live-Discord evidence sections.

- [ ] **Step 3: Observe the exact commit jobs**

Run `gh run list`, `gh run view`, and `gh pr checks` for each exact head SHA. A startup failure or zero-job workflow is a provider failure, not a local test failure and not green evidence.

- [ ] **Step 4: Merge only after required review and green gates**

After merging, verify default-branch ancestry and observe the new main-branch workflow. Do not infer main success from the PR head.

- [ ] **Step 5: Closeout evidence**

Report separately:

- WDBX provider visibility and anonymous readability;
- local WDBX, ABI, and `abbey` strict gates;
- PR/default-branch hosted CI at exact SHAs;
- production deployment: not performed unless separately authorized;
- participant-consented live Discord: not performed and not part of this plan.
