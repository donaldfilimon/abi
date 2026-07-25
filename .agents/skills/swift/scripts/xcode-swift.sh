#!/usr/bin/env bash
# Force Xcode default toolchain for Swift (ignore swiftly / TOOLCHAINS).
set -euo pipefail
unset TOOLCHAINS || true
exec /usr/bin/xcrun --toolchain default swift "$@"
