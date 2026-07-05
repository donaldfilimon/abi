---
name: connector-localcheck
description: Build the abi CLI and exercise the connector surfaces that run fully locally — the Twilio ConversationRelay simulation and the auth-status report. Use to smoke-test connectors without credentials or network. Does NOT touch the live transport or any remote provider.
---

# connector-localcheck — drive abi's local connector surfaces

Driver: **`.claude/skills/connector-localcheck/connectors.sh`** (paths relative to repo root).
Read-only-effect CLI check — evidence is the `RESULT:` line. **No creds, no network.**

## Run (agent path)
```bash
.claude/skills/connector-localcheck/connectors.sh                      # default utterance
.claude/skills/connector-localcheck/connectors.sh "cancel my account"  # custom caller text
```
Builds the CLI, runs `abi twilio simulate "<utterance>"` (asserts
`Twilio ConversationRelay simulation`, `response:`, `escalation:`) and
`abi auth status` (asserts `Authentication Status:`). Prints `RESULT: PASS`
(exit 0) or a FAIL count.

Historical verification: **PASS** on Zig master `0.17.0-dev.1099` — local Twilio
sim responds + reports `escalation: false`; auth status lists all providers.

## Gotchas
- ⚠️ **Local only by design.** This exercises the deterministic local connector
  paths. The remote/`.live` transport (real OpenAI/Anthropic/Discord/Twilio
  calls) needs stored credentials and is deliberately NOT driven here — use
  `abi auth signin <svc>` + `abi complete --live` for that, manually.
- For a source-level audit of credential validation and the live/local boundary,
  use the `connector-validator` subagent.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | `/zig-build-doctor` or `./build.sh check`. |
| missing `Twilio ConversationRelay simulation` | grammar drift — check `src/cli/handlers` twilio path + `src/connectors/twilio.zig`. |
