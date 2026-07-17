---
name: docs-validate
description: Validate the Mintlify docs site (docs/docs.json + docs/**/*.mdx) via `npx mint@latest validate`, plus a stale-.md-reference scan. Local pre-push gate; CI job docs (mint validate) runs the same script on push/PR. Not part of ./build.sh check. User-invocable only.
disable-model-invocation: true
---

# docs-validate — Mintlify docs gate

Driver: **`.agents/skills/docs-validate/validate.sh`** (paths relative to repo root).
Runs `npx mint@latest validate` against `docs/docs.json` + `docs/**/*.mdx`, then
scans for stale `.md` references (the `.md`→`.mdx` link rot fixed in #651). Evidence
is the `RESULT:` line.

## Run (user path)
```bash
.agents/skills/docs-validate/validate.sh
```
Prints `RESULT: PASS` (exit 0) or `RESULT: FAIL` (exit 1) with the validator output.

## Gates
| Gate | Covers docs? |
|------|----------------|
| `./build.sh check` | No (Zig primary gate stays separate) |
| CI job **docs (mint validate)** (Node 22) | Yes — runs this script on push/PR |
| Local `validate.sh` | Yes — run after editing `docs/` before push |

Mintlify also builds the hosted site via its GitHub app. This skill is
`disable-model-invocation: true` (user-only) because it shells out to `npx`
(network + package fetch).

## Gotchas
- ⚠️ **mintlify needs an LTS node — it hard-fails on node 25+.** On a too-new node
  (e.g. this repo's dev host runs node 26) `mint validate` refuses to run. The driver
  detects this and prints `RESULT: SKIP` (exit 3) with a hint to select an LTS node
  via `nvm`/`fnm` — that's an environment issue, not a docs error.
- **Needs network** — `npx mint@latest validate` fetches the `mint` package on first
  run. Offline → it fails at fetch (a tooling/env failure, not a docs failure).
- **Mintlify config is `docs/docs.json`** — the driver `cd`s into `docs/`; run it
  from anywhere in the repo.
- The stale-`.md` scan is a heuristic (nav/link entries ending in `.md`); treat hits
  as "verify these resolve," not hard failures — the `mint validate` result is authoritative.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `npx not on PATH` | Install Node/npm; `mint` runs via `npx`. |
| fails at fetch / offline | Environment has no network — retry where npx can reach the registry. |
| `no docs/docs.json found` | You're not in the abi repo, or the docs site moved. |
| validation errors | Fix the reported `docs/docs.json`/`.mdx` issue; re-run. Preview locally with `npx mint@latest dev`. |
