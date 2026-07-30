---
name: plugin-runtime-tester
description: Build the abi CLI, list the generated plugin registry, and execute bundled plugins through `plugin run` to confirm each dispatches its real run() (not PluginNotFound or a generic ack). Use when adding/changing plugins or verifying the registry + run-dispatch wiring.
---

# plugin-runtime-tester — verify the plugin registry + run dispatch

Driver: **`.agents/skills/plugin-runtime-tester/plugins.sh`** (paths relative to repo root).
Read-only CLI capture — evidence is the `RESULT:` line + per-plugin run output.

## Run (agent path)
```bash
.agents/skills/plugin-runtime-tester/plugins.sh
```
Builds the CLI, runs `abi plugin list`, executes a sample of plugins via
`abi plugin run <name> probe` (asserting each returns its `event (bytes=…)`
line), and checks an unknown name errors with `PluginNotFound`. Prints
`RESULT: PASS` (exit 0) or a FAIL count.

Historical verification: **PASS** — `Installed Plugins (16):`; `ai-plugin event
(bytes=5)`, `hash-plugin …`, etc., on Zig master `0.17.0-dev.1099`.

## Gotchas
- **Registering ≠ enabling.** `plugin list` reads the generated registry (shows
  every manifest); `plugin run` only works if the plugin is BOTH loaded in
  `crates/abi-cli/src/plugin.rs` AND dispatched in `crates/abi-plugins/src/manager.rs`.
  A plugin in the list but not both dispatch sites → `PluginNotFound` or a generic
  contract-ack. The `plugin-system-reviewer` subagent audits this.
- `plugin run` reads manifests from `crates/abi-plugins/plugins/<name>/` at runtime, so run it
  from the repo root (the driver `cd`s there).
- Adding a plugin requires bumping the count in `crates/abi-plugins/tests/`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | `./tools/check.sh`. |
| a plugin → `PluginNotFound` | missing `loadBundledPlugin` + dispatch branch — see `plugin-system-reviewer`. |
