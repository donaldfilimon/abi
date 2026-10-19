#!/usr/bin/env bash
# Install a new macOS arm64 GitHub Actions runner for donaldfilimon/abi.
# Requires: curl, tar, gh (repo scope), macOS arm64.
set -euo pipefail
umask 077

REPO="${GITHUB_RUNNER_REPO:-donaldfilimon/abi}"
INSTALL_DIR="${GITHUB_RUNNER_DIR:-${HOME}/dev/ci/runners/abi}"
RUNNER_NAME="${GITHUB_RUNNER_NAME:-$(hostname -s)-arm64}"
LABELS="${GITHUB_RUNNER_LABELS:-self-hosted,macOS,ARM64,abi}"
WORK_DIR="${GITHUB_RUNNER_WORK:-_work}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this script is for macOS only" >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "error: expected arm64 host, got $(uname -m)" >&2
  exit 1
fi
if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI required (https://cli.github.com/)" >&2
  exit 1
fi

mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

if [[ -f .runner || -f .service ]]; then
  echo "error: existing runner/service in ${INSTALL_DIR}; use ~/dev/ci/fleet.py health or guarded restart, or explicitly drain/uninstall/remove before reconfiguration" >&2
  exit 1
else
  echo "==> Resolving latest actions/runner osx-arm64 release"
  TAG="$(gh api repos/actions/runner/releases/latest --jq .tag_name)"
  VERSION="${TAG#v}"
  ASSET="actions-runner-osx-arm64-${VERSION}.tar.gz"
  URL="https://github.com/actions/runner/releases/download/${TAG}/${ASSET}"
  echo "==> Downloading ${URL}"
  curl -fsSL -o "${ASSET}" "${URL}"
  tar xzf "${ASSET}"
  rm -f "${ASSET}"
fi

if [[ ! -x ./config.sh ]]; then
  echo "error: config.sh missing in ${INSTALL_DIR}; remove the directory and re-run" >&2
  exit 1
fi

echo "==> Requesting registration token for ${REPO}"
TOKEN="$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq .token)"

echo "==> Configuring runner name=${RUNNER_NAME} labels=${LABELS}"
ACTIONS_RUNNER_INPUT_TOKEN="${TOKEN}" ./config.sh \
  --url "https://github.com/${REPO}" \
  --unattended \
  --name "${RUNNER_NAME}" \
  --labels "${LABELS}" \
  --work "${WORK_DIR}"
unset TOKEN

chmod 700 "${INSTALL_DIR}"
chmod 600 .runner .credentials .credentials_rsaparams .env .path

if [[ "${GITHUB_RUNNER_INSTALL_SERVICE:-1}" == "1" ]]; then
  if [[ -x ./svc.sh ]]; then
    if [[ -z "${GITHUB_ACTIONS_RUNNER_SERVICE_TEMPLATE:-}" && -f "${HOME}/dev/ci/launchd-runner.plist.template" ]]; then
      export GITHUB_ACTIONS_RUNNER_SERVICE_TEMPLATE="${HOME}/dev/ci/launchd-runner.plist.template"
    fi
    ./svc.sh install
    ./svc.sh start
    ./svc.sh status || true
  else
    echo "warn: svc.sh not present; start manually with ./run.sh" >&2
  fi
else
  echo "Skipping service install (GITHUB_RUNNER_INSTALL_SERVICE=0). Start with: cd ${INSTALL_DIR} && ./run.sh"
fi

echo "==> Done. Confirm online:"
echo "    gh api repos/${REPO}/actions/runners --jq '.runners[] | {name,status,labels:[.labels[].name]}'"
echo "Security notes: .github/self-hosted-runner.md"
