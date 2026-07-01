---
name: auth-localcheck
description: Build the abi CLI and check the credential/auth surface without storing, deleting, or transmitting anything — `auth status` (provider table) and `auth signin` usage banner. Use to smoke-test the auth command wiring after touching credential code. Never writes or deletes credentials and never hits the network.
---

# auth-localcheck — drive abi's auth surface (non-destructive)

Driver: **`.claude/skills/auth-localcheck/auth.sh`** (paths relative to repo root).
Builds the CLI and exercises only the safe auth surfaces. Evidence is the
`RESULT:` line. **No creds written/deleted, no network.**

## Run (agent path)
```bash
.claude/skills/auth-localcheck/auth.sh
```
- `auth status` → asserts `Authentication Status:`, `OpenAI:`, `Anthropic:`, `Twilio:`.
- `auth signin` (no service) → asserts the `usage: abi auth signin` banner
  (validates the subcommand is wired; **stores nothing**).

Prints `RESULT: PASS` (exit 0) or a FAIL count.

Verified this session: **PASS** on Zig master `0.17.0-dev.1099` — status lists all
five providers as `not configured`; bare `signin` prints its usage banner.

## Gotchas
- ⚠️ **`auth logout` is destructive** — it deletes any stored credentials. The
  driver deliberately does NOT run it, and does NOT run a real `auth signin
  <svc>` (which would store creds). Do those manually when you actually intend
  to change credential state.
- Overlap note: `connector-localcheck` also touches `auth status`; this skill is
  the auth-lifecycle-focused check (status + signin wiring) and stays credential-safe.
- Real remote calls (`complete --live`, live connector transport) require
  `auth signin <svc>` first; that path is intentionally out of scope here.
- For a source-level audit of credential validation + the live/local boundary,
  use the `connector-validator` subagent.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Check `zig version` (see `/zig-pin`), then `./build.sh check`. |
| `Authentication Status:` missing | Handler grammar drift — check the `auth` path in `src/cli/handlers/` and `src/foundation` credentials. |
