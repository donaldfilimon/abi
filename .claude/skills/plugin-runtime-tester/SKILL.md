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

Current Rust verification: the driver enumerates all 16 bundled fixtures,
dispatches every one through `plugin run`, and checks an unknown name fails.

## Gotchas
- **Registering ≠ enabling.** `plugin list` reads the generated registry (shows
  every manifest); `plugin run` must also resolve through
  `crates/abi-plugins/src/manager.rs`. A list-only fixture fails the runtime
  portion of this driver.
- `plugin run` reads manifests from `crates/abi-plugins/plugins/<name>/` at runtime, so run it
  from the repo root (the driver `cd`s there).
- Adding a plugin requires bumping the count in `crates/abi-plugins/tests/`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | `./tools/check.sh`. |
| a plugin → `PluginNotFound` | inspect the generated fixture inventory and `crates/abi-plugins/src/manager.rs` dispatch. |
