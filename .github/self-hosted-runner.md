# Self-hosted GitHub Actions runner (macOS arm64)

Ops and security notes for the repository self-hosted runner used by `.github/workflows/ci.yml` and `.github/workflows/benchmarks-gh-pages.yml`.

The donaldfilimon account's GitHub Actions billing is locked, so GitHub-hosted jobs are refused at dispatch (they fail in about two seconds without a runner). Self-hosted jobs still run, so every trusted job that can run on macOS arm64 uses this runner.

## Which jobs run where

| Workflow | Job (check name) | Runner | Why |
|----------|------------------|--------|-----|
| `ci.yml` | `check` (`check (self-hosted)`) | self-hosted `abi` | Primary gate; same-repo events only. |
| `ci.yml` | `check-hosted` (`check (GitHub-hosted, fork PRs)`) | `macos-latest` | Fork PR code must never run on this machine. |
| `ci.yml` | `windows-acl` (`windows credential ACL`) | `windows-latest` | Proves Windows DACL behaviour; needs Windows. Blocked while billing is locked. |
| `benchmarks-gh-pages.yml` | `deploy` | self-hosted `abi` | Pages publish; runs only on push to `main` and `workflow_dispatch` (no PR trigger). |
| `dependency-scan.yml` | `scan` | `ubuntu-latest` | `ossf/scorecard-action` is a Docker action and runs only on Linux. Blocked while billing is locked. |

## Current registration

| Field | Value |
|-------|--------|
| Host labels | `self-hosted`, `macOS`, `ARM64`, `abi` |
| Typical install dir | `~/actions-runner` |
| LaunchAgent | `actions.runner.donaldfilimon-abi.<hostname>` |
| Repo UI | [Settings → Actions → Runners](https://github.com/donaldfilimon/abi/settings/actions/runners) |

Jobs that need this machine use:

```yaml
runs-on: [self-hosted, macOS, ARM64, abi]
```

## Security model (public repository)

Self-hosted runners on **public** repos are dangerous: a workflow that reaches `runs-on: self-hosted` can execute arbitrary PR code on the host.

Hardening applied here:

1. **Job-level trust gate** — self-hosted jobs run only when:
   - `github.repository == 'donaldfilimon/abi'`, and
   - event is `push` / `workflow_dispatch`, or a `pull_request` whose
     `head.repo.full_name == github.repository` (same-repo PR only).
2. **Fork PRs** use GitHub-hosted `macos-latest` jobs only (`check-hosted`). The Pages workflow has no `pull_request` trigger, so its self-hosted `deploy` job is gated to `push` and `workflow_dispatch` only and needs no hosted fallback.
3. **Least-privilege token** — workflow `permissions: contents: read`; the Pages workflow adds only `pages: write` and `id-token: write`, which deployment requires. Every checkout uses `persist-credentials: false`.
4. **Nightly Rust via rustup** — self-hosted jobs install/ensure `nightly` with `rustfmt`/`clippy`/`rust-src` and build only through `./tools/cargo.sh` / `./tools/check.sh` (Homebrew’s stable `cargo` shadows rustup and must not be used bare).

Still recommended on the host and in GitHub settings:

- Prefer a **dedicated machine or user account** for the runner (not your daily desktop) when possible.
- In repo **Settings → Actions → General**:
  - Require approval for first-time contributors (and preferably all outside collaborators).
  - Restrict Actions permissions; do not grant write where unnecessary.
- Do **not** store production secrets on the runner host beyond what CI needs.
- Treat `_work` checkouts as untrusted after any job; do not reuse artifacts casually on the desktop.
- Never register the same runner against untrusted org/repos.

Making the repository **private** remains the strongest fix; this project stays public, so the gates above are mandatory.

## Toolchain on the runner host

CI expects a rustup **nightly** toolchain matching [`rust-toolchain.toml`](../rust-toolchain.toml), plus the usual macOS SDK:

```bash
rustup show
rustup run nightly rustc --version
./tools/cargo.sh --version   # never bare `cargo` — Homebrew shadows rustup
xcode-select -p              # needed for Apple-framework / Foundation Models paths when those features are on
```

The Pages `deploy` job additionally needs:

- **GNU tar as `gtar`** (`brew install gnu-tar`). On macOS, `actions/upload-pages-artifact` archives with `gtar` so it can use `--hard-dereference`; stock macOS ships only bsdtar. The job's first check step fails with a clear error if `gtar` is missing from the runner PATH.
- **Homebrew's bin directory on the runner PATH** (`/opt/homebrew/bin`). The runner records PATH in `~/actions-runner/.path` at configure time; edit it or re-run `./config.sh` after installing Homebrew, then restart the service.
- **A current runner version.** `actions/deploy-pages` runs on Node 24 and `actions/configure-pages` on Node 20; the runner ships its own Node runtimes, so keep the runner auto-updating rather than pinning an old release.

No other host tooling is needed for Pages: it publishes `site/` as-is with no build step, authenticates through the job's OIDC token (`id-token: write`), and deploys through the `github-pages` environment.

After changing the Rust toolchain or PATH, restart the service:

```bash
cd ~/actions-runner
./svc.sh stop && ./svc.sh start
./svc.sh status
```

## Install / reconfigure

Use the helper (requires `gh` auth with `repo` scope):

```bash
./tools/github-runner/setup-macos-arm64.sh
```

Or follow GitHub’s UI at  
https://github.com/donaldfilimon/abi/settings/actions/runners/new?arch=arm64  
(Settings → Actions → Runners → New self-hosted runner, macOS, ARM64), add the custom label `abi`, then install the service with `./svc.sh install && ./svc.sh start`.

A runner is registered to exactly one repository. If this Mac also serves another repository (for example `gama` or `mlai-website-app`), give each one its own runner in its own directory, such as `~/actions-runner` for `abi` and `~/actions-runner-gama` for `gama`, each with its own custom label equal to the lowercase repository name and its own LaunchAgent. Never point one runner directory at two repositories.

Until a runner with the `abi` label is online, the self-hosted jobs wait in the queue.

## Day-2 operations

```bash
cd ~/actions-runner
./svc.sh status
./svc.sh stop
./svc.sh start
./svc.sh uninstall   # remove LaunchAgent only; does not unregister

# Unregister from GitHub (needs a removal token from the UI or API)
./config.sh remove --token <REMOVE_TOKEN>
```

Logs:

- `~/Library/Logs/actions.runner.donaldfilimon-abi.*/`
- `~/actions-runner/_diag/`

## Local host hardening checklist

Run:

```bash
./tools/github-runner/harden-macos.sh
```

That script is read-only by default; it prints a checklist and optional steps (LaunchAgent confirm, PATH, disk free space, no world-writable runner dir).

## Workflow author rules

- Never add `runs-on: self-hosted` (or the `abi` label set) without the same-repo `if:` gate used in `ci.yml`.
- Prefer `permissions:` minimal scopes on every workflow.
- Do not use `pull_request_target` with checkout of the PR head on self-hosted runners.
- Keep fork coverage on `macos-latest` (or another GitHub-hosted label).
