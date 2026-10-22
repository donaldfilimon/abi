#!/usr/bin/env bash
# Read-only hardening checklist for the abi self-hosted runner host.
# Exit 0 if no critical failures; non-zero if something blocks safe CI use.
set -euo pipefail

INSTALL_DIR="${GITHUB_RUNNER_DIR:-${HOME}/dev/ci/runners/abi}"
REPO="${GITHUB_RUNNER_REPO:-donaldfilimon/abi}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

failures=0
warns=0

ok() { printf '  [ok]  %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; warns=$((warns + 1)); }
fail() { printf '  [FAIL] %s\n' "$*"; failures=$((failures + 1)); }

echo "==> Host"
echo "    $(uname -s) $(uname -m) — $(sw_vers -productName) $(sw_vers -productVersion)"

if [[ "$(uname -m)" != "arm64" ]]; then
  fail "host is not arm64 (CI labels expect ARM64)"
else
  ok "architecture arm64"
fi

echo "==> Rust toolchain"
# Check the toolchain the repository actually pins, not a floating channel.
# CI installs the exact channel from rust-toolchain.toml, so the host must
# have that one available or the gate cannot run.
# `|| true` keeps a missing/unreadable file on the diagnostic path instead of
# aborting the whole checklist under `set -e` before the other checks run.
TOOLCHAIN_CHANNEL="$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' \
  "${REPO_ROOT}/rust-toolchain.toml" 2>/dev/null | head -1 || true)"
if [[ -z "$TOOLCHAIN_CHANNEL" ]]; then
  fail "cannot read the pinned channel from rust-toolchain.toml"
elif ! command -v rustup >/dev/null 2>&1; then
  fail "rustup not on PATH (CI installs/uses ${TOOLCHAIN_CHANNEL} via rustup)"
elif ! rustup run "$TOOLCHAIN_CHANNEL" rustc --version >/dev/null 2>&1; then
  fail "${TOOLCHAIN_CHANNEL} toolchain missing — run: rustup toolchain install ${TOOLCHAIN_CHANNEL} --profile minimal && rustup component add --toolchain ${TOOLCHAIN_CHANNEL} rustfmt clippy rust-src"
else
  ok "$(rustup run "$TOOLCHAIN_CHANNEL" rustc --version 2>/dev/null | head -1)"
fi
if [[ ! -x "${REPO_ROOT}/tools/cargo.sh" ]]; then
  fail "tools/cargo.sh missing or not executable"
else
  ok "tools/cargo.sh present (use instead of bare cargo — Homebrew shadows rustup)"
fi

echo "==> Runner install (${INSTALL_DIR})"
if [[ ! -d "${INSTALL_DIR}" ]]; then
  fail "install dir missing — run tools/github-runner/setup-macos-arm64.sh"
else
  ok "install dir exists"
  if [[ -f "${INSTALL_DIR}/.runner" ]]; then
    ok "runner configured (.runner present)"
  else
    fail "runner not configured (no .runner)"
  fi
  # World-writable install dir is unsafe
  if [[ -n "$(find "${INSTALL_DIR}" -maxdepth 0 -perm -0002 2>/dev/null)" ]]; then
    fail "install dir is world-writable"
  else
    ok "install dir not world-writable"
  fi
fi

echo "==> LaunchAgent / service"
if [[ -x "${INSTALL_DIR}/svc.sh" ]]; then
  status_out=""
  if status_out="$(cd "${INSTALL_DIR}" && ./svc.sh status 2>&1)"; then
    if grep -q 'Started:' <<<"${status_out}"; then
      ok "svc.sh reports Started"
    else
      warn "svc.sh status did not show Started"
      printf '%s\n' "${status_out}"
    fi
  else
    warn "svc.sh status failed (runner may be run via ./run.sh instead)"
  fi
else
  warn "svc.sh missing"
fi

echo "==> Disk"
# Need headroom for Cargo/target caches + check. A missing install dir makes
# `df` fail; under `set -e` that aborted the whole checklist before the
# summary, so measure the nearest existing ancestor instead.
DISK_PATH="${INSTALL_DIR:-${HOME}}"
while [[ -n "$DISK_PATH" && ! -d "$DISK_PATH" ]]; do
  DISK_PATH="$(dirname "$DISK_PATH")"
  [[ "$DISK_PATH" == "/" || "$DISK_PATH" == "." ]] && break
done
avail_kb="$(df -k "$DISK_PATH" 2>/dev/null | awk 'NR==2 {print $4}')"
if [[ -z "$avail_kb" ]]; then
  warn "could not measure free space for ${INSTALL_DIR:-${HOME}}"
else
  avail_gb=$((avail_kb / 1024 / 1024))
  if (( avail_gb < 15 )); then
    fail "only ~${avail_gb}GiB free; recommend ≥15GiB for check + Cargo target/"
  else
    ok "~${avail_gb}GiB free"
  fi
fi

echo "==> GitHub visibility reminder"
if command -v gh >/dev/null 2>&1; then
  vis="$(gh api "repos/${REPO}" --jq .visibility 2>/dev/null || echo unknown)"
  if [[ "${vis}" == "public" ]]; then
    warn "repo is public — keep job-level same-repo gates; prefer dedicated host"
  elif [[ "${vis}" == "private" ]]; then
    ok "repo is private"
  else
    warn "could not determine repo visibility (${vis})"
  fi
  online="$(gh api "repos/${REPO}/actions/runners" --jq '[.runners[]|select(.status=="online")]|length' 2>/dev/null || echo 0)"
  if [[ "${online}" == "0" ]]; then
    fail "no online self-hosted runners registered for ${REPO}"
  else
    ok "${online} online runner(s) on GitHub"
  fi
else
  warn "gh not installed; skip remote checks"
fi

echo "==> Manual GitHub settings (confirm in UI)"
echo "    - Actions → General: require approval for outside collaborators"
echo "    - Actions → General: restrict GITHUB_TOKEN permissions (read on workflow is set in ci.yml)"
echo "    - Do not disable the same-repo if: gates in .github/workflows/ci.yml"
echo "    - Prefer not storing extra secrets on this host"
echo "    - Docs: .github/self-hosted-runner.md"

echo
if (( failures > 0 )); then
  echo "RESULT: ${failures} failure(s), ${warns} warning(s)"
  exit 1
fi
echo "RESULT: ok (${warns} warning(s))"
exit 0
