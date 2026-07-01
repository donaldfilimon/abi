---
name: docs-validate
description: Validate the Mintlify docs site (docs/docs.json + docs/**/*.mdx) via `npx mint@latest validate`, plus a stale-.md-reference scan. Use as a pre-push gate after editing docs/ — this is the one surface ./build.sh check and CI do NOT cover, so broken config/pages otherwise only surface on push. User-invocable only.
disable-model-invocation: true
---

# docs-validate — pre-push gate for the Mintlify docs site

Driver: **`.claude/skills/docs-validate/validate.sh`** (paths relative to repo root).
Runs `npx mint@latest validate` against `docs/docs.json` + `docs/**/*.mdx`, then
scans for stale `.md` references (the `.md`→`.mdx` link rot fixed in #651). Evidence
is the `RESULT:` line.

## Run (user path)
```bash
.claude/skills/docs-validate/validate.sh
```
Prints `RESULT: PASS` (exit 0) or `RESULT: FAIL` (exit 1) with the validator output.

## Why this exists
`./build.sh check` and `.github/workflows/ci.yml` do **not** validate the docs site —
Mintlify builds/hosts via its GitHub app on push, so a broken `docs.json` or `.mdx`
page is only caught *after* it lands. This skill closes that gap as a local pre-push
check. It is `disable-model-invocation: true` (user-only) because it's a validation
action that shells out to `npx` (network + package fetch).

## Gotchas
- ⚠️ **mintlify needs an LTS node — it hard-fails on node 25+.** On a too-new node
  (e.g. this repo's dev host runs node 26) `mint validate` refuses to run. The driver
  detects this and prints `RESULT: SKIP` (exit 3) with a hint to select an LTS node
  via `nvm`/`fnm` — that's an environment issue, not a docs error.
- **Needs network** — `npx mint@latest validate` fetches the `mint` package on first
  run. Offline → it fails at fetch (a tooling/env failure, not a docs failure).
- **`docs.json` lives under `docs/`** — the driver `cd`s there; run it from anywhere
  in the repo.
- **Not wired into `./build.sh` or CI on purpose** — this is a manual gate. Run it
  after any change under `docs/` and before pushing docs edits.
- The stale-`.md` scan is a heuristic (nav/link entries ending in `.md`); treat hits
  as "verify these resolve," not hard failures — the `mint validate` result is authoritative.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `npx not on PATH` | Install Node/npm; `mint` runs via `npx`. |
| fails at fetch / offline | Environment has no network — retry where npx can reach the registry. |
| `no docs.json found` | You're not in the abi repo, or the docs site moved. |
| validation errors | Fix the reported `docs.json`/`.mdx` issue; re-run. Preview locally with `npx mint@latest dev`. |
