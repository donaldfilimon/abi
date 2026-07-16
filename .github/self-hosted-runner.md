# Self-hosted GitHub Actions runner (macOS arm64)

Ops and security notes for the repository self-hosted runner used by `.github/workflows/ci.yml`.

## Current registration

| Field | Value |
|-------|--------|
| Host labels | `self-hosted`, `macOS`, `ARM64`, `abi` |
| Typical install dir | `~/actions-runner` |
| LaunchAgent | `actions.runner.donaldfilimon-abi.<hostname>` |
| Repo UI | [Settings → Actions → Runners](https://github.com/donaldfilimon/abi/settings/actions/runners) |

CI jobs that need this machine use:

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
2. **Fork PRs** use GitHub-hosted `macos-latest` jobs only (`check-hosted`, `cross-smoke-hosted`).
3. **Least-privilege token** — workflow `permissions: contents: read`.
4. **Pinned Zig from host PATH** — self-hosted jobs refuse to run if `zig version` ≠ `.zigversion` pin (no silent toolchain drift).

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

CI expects the pin in `.zigversion` / workflow `ZIG_VERSION` on `PATH` (e.g. via `zvm` / `zigup`).

```bash
zig version   # must match .zigversion
xcode-select -p   # needed for Apple-framework / Foundation Models paths when those features are on
```

After changing Zig or PATH, restart the service:

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
then install the service with `./svc.sh install && ./svc.sh start`.

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
