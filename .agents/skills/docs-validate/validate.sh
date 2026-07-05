#!/usr/bin/env bash
# docs-validate — validate the Mintlify docs site (docs/docs.json + docs/**/*.mdx).
#
# This is the one surface `./build.sh check` and CI do NOT cover: broken docs.json
# or .mdx pages only surface on push (Mintlify builds via its GitHub app). Run this
# as a pre-push gate after touching docs/. Requires network (npx fetches mint).
set -uo pipefail

SCRIPT_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")

cd "$(git rev-parse --show-toplevel 2>/dev/null)" 2>/dev/null || {
  echo "RESULT: FAIL (run from inside the abi repo)"; exit 1; }

# Mintlify config lives under docs/.
if [ -f docs/docs.json ]; then cd docs
elif [ -f docs.json ]; then :
else echo "RESULT: FAIL (no docs.json found — is this the docs site?)"; exit 1; fi

command -v npx >/dev/null 2>&1 || { echo "RESULT: FAIL (npx not on PATH — install Node/npm)"; exit 1; }

# mintlify requires an LTS node and hard-fails on node 25+. Detect early and give
# an actionable message instead of a confusing validator error (this is an env
# issue, not a docs problem).
node_major="$(node -v 2>/dev/null | sed 's/^v//; s/[.].*//')"
if [ -n "$node_major" ] && [ "$node_major" -ge 25 ] 2>/dev/null; then
  fallback_node_bin="${ABI_DOCS_VALIDATE_NODE_BIN:-$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin}"
  fallback_node="$fallback_node_bin/node"
  if [ "${ABI_DOCS_VALIDATE_NODE_FALLBACK:-0}" != "1" ] && [ -x "$fallback_node" ]; then
    fallback_major="$("$fallback_node" -v 2>/dev/null | sed 's/^v//; s/[.].*//')"
    if [ -n "$fallback_major" ] && [ "$fallback_major" -lt 25 ] 2>/dev/null; then
      echo "node $(node -v) unsupported; retrying with bundled node $("$fallback_node" -v)."
      PATH="$fallback_node_bin:$PATH" ABI_DOCS_VALIDATE_NODE_FALLBACK=1 exec "$SCRIPT_PATH" "$@"
    fi
  fi
  echo "RESULT: SKIP (node $(node -v) unsupported — mintlify needs an LTS node, fails on 25+)."
  echo "  Select an LTS node (nvm/fnm) and re-run; this is not a docs error."
  exit 3
fi

echo "[1/2] mint validate (config + pages) in $(pwd) ..."
if npx --yes mint@latest validate; then
  vfail=0
else
  vfail=1
fi

# Extra guard: catch the .md->.mdx link rot that bit PR #651 — internal links or
# nav entries pointing at .md files that no longer exist after the Mintlify cutover.
echo "[2/2] scanning for stale .md references in nav/pages ..."
stale="$(grep -rInE '"[^"]+\.md"|\]\([^)]+\.md[)#]' docs.json ./*.mdx 2>/dev/null | grep -v '\.mdx' || true)"
if [ -n "$stale" ]; then
  echo "  ! possible stale .md references (verify these still resolve):"
  printf '%s\n' "$stale" | sed 's/^/    /'
fi

if [ "$vfail" -eq 0 ]; then
  echo "RESULT: PASS — Mintlify config + pages valid"
  exit 0
fi
echo "RESULT: FAIL — see 'mint validate' output above"
exit 1
