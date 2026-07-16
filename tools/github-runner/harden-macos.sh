#!/usr/bin/env bash
# Read-only hardening checklist for the abi self-hosted runner host.
# Exit 0 if no critical failures; non-zero if something blocks safe CI use.
set -euo pipefail

INSTALL_DIR="${GITHUB_RUNNER_DIR:-${HOME}/actions-runner}"
EXPECTED_ZIG="${EXPECTED_ZIG:-}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -z "${EXPECTED_ZIG}" && -f "${REPO_ROOT}/.zigversion" ]]; then
  EXPECTED_ZIG="$(tr -d '[:space:]' <"${REPO_ROOT}/.zigversion")"
fi

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

echo "==> Zig toolchain"
if ! command -v zig >/dev/null 2>&1; then
  fail "zig not on PATH"
else
  actual="$(zig version)"
  ok "zig on PATH: ${actual}"
  if [[ -n "${EXPECTED_ZIG}" && "${actual}" != "${EXPECTED_ZIG}" ]]; then
    fail "zig version ${actual} != pin ${EXPECTED_ZIG}"
  elif [[ -n "${EXPECTED_ZIG}" ]]; then
    ok "matches pin ${EXPECTED_ZIG}"
  fi
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
  if (cd "${INSTALL_DIR}" && ./svc.sh status) >/tmp/abi-runner-svc.status 2>&1; then
    if grep -q 'Started:' /tmp/abi-runner-svc.status 2>/dev/null; then
      ok "svc.sh reports Started"
    else
      warn "svc.sh status did not show Started — see /tmp/abi-runner-svc.status"
      cat /tmp/abi-runner-svc.status || true
    fi
  else
    warn "svc.sh status failed (runner may be run via ./run.sh instead)"
  fi
else
  warn "svc.sh missing"
fi

echo "==> Disk"
# Need headroom for Zig caches + cross-smoke
avail_kb="$(df -k "${INSTALL_DIR:-${HOME}}" | awk 'NR==2 {print $4}')"
avail_gb=$((avail_kb / 1024 / 1024))
if (( avail_gb < 15 )); then
  fail "only ~${avail_gb}GiB free; recommend ≥15GiB for check + cross-smoke"
else
  ok "~${avail_gb}GiB free"
fi

echo "==> GitHub visibility reminder"
if command -v gh >/dev/null 2>&1; then
  vis="$(gh api repos/donaldfilimon/abi --jq .visibility 2>/dev/null || echo unknown)"
  if [[ "${vis}" == "public" ]]; then
    warn "repo is public — keep job-level same-repo gates; prefer dedicated host"
  elif [[ "${vis}" == "private" ]]; then
    ok "repo is private"
  else
    warn "could not determine repo visibility (${vis})"
  fi
  online="$(gh api repos/donaldfilimon/abi/actions/runners --jq '[.runners[]|select(.status=="online")]|length' 2>/dev/null || echo 0)"
  if [[ "${online}" == "0" ]]; then
    fail "no online self-hosted runners registered for donaldfilimon/abi"
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
