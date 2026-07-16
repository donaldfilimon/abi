# zig-newest-skills — gotchas

Battle scars for the master-toolchain driver. Keep SKILL.md lean; load this when
debugging zvm, pin revert, or false re-downloads.

## Old nightlies are not re-downloadable

`zvm` serves the **current** master from ziglang.org. Once a pin like
`0.17.0-dev.1252+…` is superseded, `zvm install <old-pin>` often fails with
`unsupported Zig version`. Installing master can reclaim or replace local
`~/.zvm/<pin>` state.

**Implications:**

- `--revert` may fail after a master install if the pin directory is gone.
- The driver prints a clear FATAL with options: stay on master, bump
  `.zigversion`, or restore a private tarball into `~/.zvm/<pin>`.
- For reproducibility after a successful master validation, bump the pin
  deliberately (`.zigversion` + CI + instruction files) rather than relying
  on re-fetch of the old hash.

## Never grep `zvm list` for presence

`zvm list` is ANSI-colored (e.g. `\x1b[32mmaster`). A word-boundary grep like
`grep '\bmaster\b'` can fail because the `m` in `[32m` abuts `master`.

**Correct check (as in the driver):**

```bash
[ -x "${ZVM_PATH:-$HOME/.zvm}/master/zig" ]
```

Grepping list output causes needless ~50MB+ re-downloads every run.

## `build.sh` does not enforce the pin

`./build.sh` / `tools/build.sh` invoke whatever `zig` is on `PATH` and print
`Using Zig: …`. Selecting master **must** be done via `zvm use master` (or
equivalent) before the build. There is no in-repo toolchain switcher.

## `check-parity` is not a feature-graph compile

`zig build check-parity` is a std-only, host-target line scanner for mod/stub
public names. It can succeed when the full feature graph would fail. The driver
runs it **first** to separate “toolchain runs our tools” from “CLI/MCP feature
graph compiles under this std.”

## No macOS `timeout(1)`

Do not wrap gates in GNU `timeout`; it is absent on stock macOS. Rely on the
caller’s tool timeout (agent shell) instead.

## Smoke path ports

`--smoke` should prefer `.agents/skills/run-abi/smoke.sh`, falling back to
`.claude/skills/run-abi/smoke.sh` when the mirror layout is used. Both must stay
executable and hermetic (`ABI_WDBX_PERSIST=0` etc. inside the harness).

## Historical note (do not treat as current)

Past sessions validated intermediate masters (e.g. pin `1252` → master `1275`)
with zero source changes. That is **historical**, not a promise for current
master vs the live `.zigversion` pin. Always re-run the driver after pin or
`std` churn.
