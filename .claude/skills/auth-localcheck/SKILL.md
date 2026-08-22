---
name: auth-localcheck
description: Build the abi CLI and check the credential/auth surface without storing, deleting, or transmitting anything — `auth status` (provider table) and `auth signin` usage banner. Use to smoke-test the auth command wiring after touching credential code. Never writes or deletes credentials and never hits the network.
---

# auth-localcheck — drive abi's auth surface (non-destructive)

Driver: **`.agents/skills/auth-localcheck/auth.sh`** (paths relative to repo root).
Builds the CLI and exercises only the safe auth surfaces. Evidence is the
`RESULT:` line. **No creds written/deleted, no network.**

## Run (agent path)
```bash
.agents/skills/auth-localcheck/auth.sh
```
- `auth status` → asserts `Authentication Status:`, `OpenAI:`, `Anthropic:`, `Twilio:`.
- `auth signin` (no service) → asserts the `usage: abi auth signin` banner
  (validates the subcommand is wired; **stores nothing**).

Prints `RESULT: PASS` (exit 0) or a FAIL count.

Current Rust driver: builds `target/debug/abi`, checks the provider status
table, and proves bare `signin` reaches the usage-only path without writing.

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
| `build` FAIL | Check nightly via `./tools/cargo.sh --version`, then `./tools/check.sh`. |
| `Authentication Status:` missing | Handler grammar drift — check `crates/abi-cli/src/auth.rs` and `../wdbx/crates/abi-foundation/src/credentials/`. |
