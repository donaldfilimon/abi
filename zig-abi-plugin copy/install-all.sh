#!/usr/bin/env bash
set -euo pipefail

# Install zig-abi-plugin into common plugin roots (opencode, claude code, codex, etc.)

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION=""
if [[ -f "$SRC_DIR/.mcp.json" ]]; then
  VERSION=$(grep -Eo '"version"\s*:\s*"[^"]*"' "$SRC_DIR/.mcp.json" 2>/dev/null | head -1 | sed -E 's/.*\"([^\"]+)\".*/\1/')
fi
VERSION="${VERSION:-unknown}"

echo "Installing zig-abi-plugin (version ${VERSION}) from ${SRC_DIR}"

ROOTS=(
  "$HOME/.opencode/plugins"
  "$HOME/Code/OpenCode/plugins"
  "/usr/local/share/opencode/plugins"
  "$HOME/.claude/plugins"
  "$HOME/Code/ClaudeCode/plugins"
  "/usr/local/share/claude/plugins"
  "$HOME/.codex/plugins"
  "$HOME/Code/Codex/plugins"
  "/usr/local/share/codex/plugins"
)

INSTALL_COUNT=0
for ROOT in "${ROOTS[@]}"; do
  if [[ -d "$ROOT" ]]; then
    DEST="${ROOT}/zig-abi-plugin"
    mkdir -p "$ROOT"
    if [[ -d "$DEST" ]]; then
      PREV_V=""
      if [[ -f "${DEST}/.INSTALL_VERSION" ]]; then
        PREV_V=$(cat "${DEST}/.INSTALL_VERSION")
      fi
      if [[ "$PREV_V" == "$VERSION" ]]; then
        echo "Already up-to-date at ${DEST} (version ${VERSION})"
        continue
      fi
      echo "Updating install at ${DEST} (was ${PREV_V:-unknown}, now ${VERSION})"
      rm -rf "$DEST"
    fi
    cp -a "$SRC_DIR" "$DEST"
    echo "$VERSION" > "${DEST}/.INSTALL_VERSION"
    echo "Installed to ${DEST}"
    INSTALL_COUNT=$((INSTALL_COUNT+1))
  fi
done

if (( INSTALL_COUNT == 0 )); then
  echo "No suitable install roots found. Searched ${#ROOTS[@]} candidates."
  echo "Set environment variables OPENCODE_PLUGINS_ROOT, CLAUDE_CODE_PLUGINS_ROOT, CODEX_PLUGINS_ROOT to direct installation."
fi

echo "Install complete."
