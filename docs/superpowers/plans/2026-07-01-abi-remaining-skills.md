# abi Remaining Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the remaining meaningful `.claude/skills/` for abi — the interactive TUI/dashboard and the long-running WDBX servers — the surfaces the existing 17 skills do not yet cover.

**Architecture:** Each skill is a directory under `.claude/skills/<name>/` with a `SKILL.md` (man page) and an executable driver script in the established house style. The two new surfaces need driver shapes the existing skills don't use: a **tmux pty wrapper** for the interactive dashboard, and a **background-launch + `curl` poll** for the REST/cluster servers. Every driver must be *run to green this session* before its SKILL.md is written.

**Tech Stack:** Zig 0.17 CLI (`zig-out/bin/abi`), bash drivers, `tmux` (pty for TUI), `curl` (server probing).

## Global Constraints

- Toolchain: builds with the `zig` on PATH; project pins `0.17.0-dev.978+a078d55a2`, also compiles on master `0.17.0-dev.1099`. Do not switch the pin.
- House driver style (copy verbatim from `.claude/skills/wdbx-bench/bench.sh`): `set -uo pipefail`; resolve `REPO_ROOT` from `${BASH_SOURCE[0]}` via `../../..`; `cd "$REPO_ROOT"`; `ABI="$REPO_ROOT/zig-out/bin/abi"`; `say()` + a marker-check helper; `fail` counter; final `RESULT: PASS`/`RESULT: FAIL — N check(s).`; `exit "$fail"`.
- Build step in every driver: `./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }`.
- Skill dir name == frontmatter `name:` == slash command. Descriptions must contain the verbs an agent types (run/start/build/smoke-test/screenshot).
- Non-destructive: never store/delete real credentials, never bind non-loopback, always kill launched servers/sessions on exit, use scratch paths under `zig-out/`.
- After adding a skill, append it to the CLAUDE.md "Project Skills" list (keep it accurate).
- Commit locally per skill with the standard `Co-Authored-By: Claude Opus 4.8 (1M context)` + `Claude-Session:` trailers. Do not push unless asked.

---

### Task 1: `run-tui` — drive the interactive diagnostics dashboard via tmux

The one surface with no headless driver anywhere (run-abi explicitly punts). `abi tui`/`dashboard`/`--tui` require a TTY: with piped stdin, `tcgetattr` fails with `errno 19` and it stack-traces (`src/features/tui/mod.zig` → `dashboard.zig`). A pty via tmux is the only way to drive it programmatically.

**Files:**
- Create: `.claude/skills/run-tui/tui.sh`
- Create: `.claude/skills/run-tui/SKILL.md`
- Modify: `CLAUDE.md` (Project Skills list)

**Interfaces:**
- Consumes: `zig-out/bin/abi` (built by the driver).
- Produces: nothing importable — a driver + man page.

- [ ] **Step 1: Discover the real render markers and quit key.**
  Read `src/features/tui/dashboard.zig` and `src/features/tui/mod.zig` to find (a) the stable header/label strings the dashboard prints on first paint, and (b) the key that quits (look for the input-handling switch — e.g. `'q'`). You will assert on a header string you can *see*, not one you guessed.

- [ ] **Step 2: Prove the pty launch by hand first.**
  Run:
  ```bash
  ./build.sh cli >/dev/null 2>&1
  tmux kill-session -t abi-tui 2>/dev/null
  tmux new-session -d -s abi-tui -x 200 -y 50 './zig-out/bin/abi tui'
  sleep 2
  tmux capture-pane -pt abi-tui | sed -n '1,20p'
  tmux send-keys -t abi-tui q
  sleep 1
  tmux kill-session -t abi-tui 2>/dev/null
  ```
  Expected: the captured pane shows the dashboard (NOT a stack trace / `errno 19`). Record one stable line from the pane as `DASH_MARKER`.

- [ ] **Step 3: Write the driver `tui.sh`.**
  House style + tmux. Skeleton (fill `DASH_MARKER` with the string observed in Step 2):
  ```bash
  #!/usr/bin/env bash
  # run-tui driver: build the abi CLI and drive the interactive diagnostics
  # dashboard through a tmux pty (abi tui needs a real terminal). Captures the
  # rendered pane, asserts the dashboard painted, sends the quit key, tears down.
  set -uo pipefail
  SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
  cd "$REPO_ROOT"
  ABI="$REPO_ROOT/zig-out/bin/abi"
  SESSION="abi-tui-skill"
  DASH_MARKER="<stable header line observed in Step 2>"
  fail=0
  say() { printf '\n=== %s ===\n' "$*"; }
  command -v tmux >/dev/null || { echo "[FAIL] tmux not installed (brew install tmux)"; exit 1; }
  say "build cli"
  ./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
  [ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }
  say "launch abi tui under tmux pty"
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" -x 200 -y 50 "$ABI tui"
  sleep 2
  pane=$(tmux capture-pane -pt "$SESSION" 2>/dev/null || true)
  printf '%s\n' "$pane"
  printf '%s' "$pane" | grep -qF -- "$DASH_MARKER" && echo "[ok] dashboard painted" \
      || { echo "[FAIL] dashboard did not paint (missing: $DASH_MARKER)"; fail=$((fail+1)); }
  printf '%s' "$pane" | grep -qiE "errno 19|panic|tcgetattr" && { echo "[FAIL] tty error in pane"; fail=$((fail+1)); } || echo "[ok] no tty error"
  say "quit + teardown"
  tmux send-keys -t "$SESSION" q 2>/dev/null || true
  sleep 1
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  echo "[ok] session torn down"
  say "summary"; echo "failed checks: $fail"
  [ "$fail" -eq 0 ] && echo "RESULT: PASS — TUI dashboard painted under pty." || echo "RESULT: FAIL — $fail check(s)."
  exit "$fail"
  ```

- [ ] **Step 4: Run the driver and confirm PASS.**
  Run: `chmod +x .claude/skills/run-tui/tui.sh && .claude/skills/run-tui/tui.sh`
  Expected: `RESULT: PASS — TUI dashboard painted under pty.` and exit 0. If it shows `errno 19`, the pty isn't being allocated — confirm you launched via `tmux new-session` (not a pipe) and that `-x/-y` give a large enough pane.

- [ ] **Step 5: Write `SKILL.md`.**
  Frontmatter `name: run-tui`; description with verbs: "run/launch/screenshot the abi diagnostics TUI/dashboard under a tmux pty." Body: intro (interactive, pty-driven), Run (agent path) pointing at `tui.sh`, the verified-this-session PASS line, Gotchas (needs `tmux`; `abi tui` stack-traces without a TTY — that's expected; `dashboard`/`--tui` are the same surface), Troubleshooting (tmux missing → `brew install tmux`; blank pane → increase `sleep`/pane size). Cross-ref the `tui-navigation-guide` subagent for source-level questions.

- [ ] **Step 6: Update CLAUDE.md and commit.**
  Append `` `run-tui` (interactive dashboard via tmux pty) `` to the Project Skills list.
  ```bash
  git add .claude/skills/run-tui CLAUDE.md
  git commit -m "feat(skills): add run-tui (tmux pty driver for the diagnostics dashboard)"
  ```

---

### Task 2: `wdbx-api-serve` — drive the WDBX REST server (background + curl)

`abi wdbx api serve [port]` starts a loopback REST listener honoring `ABI_WDBX_REST_TOKEN` (bearer). Server-type driver: background-launch, poll until up, hit an endpoint, verify token enforcement, kill.

**Files:**
- Create: `.claude/skills/wdbx-api-serve/serve.sh`
- Create: `.claude/skills/wdbx-api-serve/SKILL.md`
- Modify: `CLAUDE.md` (Project Skills list)

**Interfaces:**
- Consumes: `zig-out/bin/abi`.
- Produces: a driver + man page.

- [ ] **Step 1: Discover the REST surface.**
  Read `src/features/wdbx/rest.zig` (and `net_line.zig`) to find the exact route path(s) and a response marker (e.g. a JSON field), the default port, and how `ABI_WDBX_REST_TOKEN` gates requests (expected 401/unauthorized without the bearer). Record the endpoint path as `ROUTE` and an expected body marker as `BODY_MARKER`.

- [ ] **Step 2: Prove the launch by hand.**
  ```bash
  ./build.sh cli >/dev/null 2>&1
  PORT=8091
  ./zig-out/bin/abi wdbx api serve $PORT & SRV=$!
  for i in $(seq 1 20); do curl -sf "http://127.0.0.1:$PORT<ROUTE>" && break; sleep 0.3; done
  echo; curl -s -o /dev/null -w "no-token http=%{http_code}\n" "http://127.0.0.1:$PORT<ROUTE>"
  kill $SRV 2>/dev/null
  ```
  Expected: the endpoint returns the `BODY_MARKER`. Note the no-token HTTP code (to decide token-enforcement assertion in Step 3).

- [ ] **Step 3: Write the driver `serve.sh`** in house style: build; pick a scratch port (default `8091`, overridable `$1`); launch `abi wdbx api serve "$PORT" &` capturing the PID; poll `curl -sf` up to ~6s; assert the `ROUTE` returns `BODY_MARKER`; then export `ABI_WDBX_REST_TOKEN=probe-token`, restart the server, and assert an unauthenticated request is rejected while a `-H "Authorization: Bearer probe-token"` request succeeds; always `kill "$PID"` in an `EXIT` trap; print the `RESULT:` line. (Bind loopback only; never expose externally.)

- [ ] **Step 4: Run the driver and confirm PASS.**
  Run: `chmod +x .claude/skills/wdbx-api-serve/serve.sh && .claude/skills/wdbx-api-serve/serve.sh`
  Expected: `RESULT: PASS` and exit 0, with the server killed (no lingering process: `pgrep -f 'abi wdbx api serve'` returns nothing).

- [ ] **Step 5: Write `SKILL.md`** — verbs: "run/start/serve/smoke-test the WDBX REST API." Gotchas: loopback-only, `ABI_WDBX_REST_TOKEN` bearer scheme (not a TLS substitute), the driver kills the server on exit, choose a free port. Cross-ref `wdbx-explorer` subagent.

- [ ] **Step 6: Update CLAUDE.md and commit** (`feat(skills): add wdbx-api-serve (REST server smoke)`).

---

### Task 3 (optional): `wdbx-cluster-serve` — networked cluster node

`abi wdbx cluster serve <port> [node] [host]` starts a networked cluster node (distinct from the in-process `cluster demo` that `cluster-demo-guide` covers). Only build this if Task 2's server pattern generalizes cleanly.

**Files:** `.claude/skills/wdbx-cluster-serve/cluster-serve.sh`, `.claude/skills/wdbx-cluster-serve/SKILL.md`, `CLAUDE.md`.

- [ ] **Step 1:** Read `src/features/wdbx/` cluster serve path for the listen marker + readiness signal.
- [ ] **Step 2:** Prove: launch `abi wdbx cluster serve 8092 &`, poll for the listen marker on stdout/stderr or a port probe (`nc -z 127.0.0.1 8092`), then kill.
- [ ] **Step 3:** Driver in house style, background-launch + readiness poll + assert listen marker + `EXIT`-trap kill.
- [ ] **Step 4:** Run → `RESULT: PASS`, no lingering process.
- [ ] **Step 5:** SKILL.md (verbs: run/start/serve the WDBX cluster node; loopback-only; killed on exit).
- [ ] **Step 6:** Update CLAUDE.md, commit.

---

## Surfaces intentionally NOT given a skill (with reason)

- `abi help` — trivial usage print; no behavior to drive.
- `abi train "<x>"` — verified to be a minimal completion-style action ("Abi action: … minimal overhead"); already representative-covered by `complete-base`/`agent-plan-train`.
- `abi wdbx compute info` / `wdbx gpu info` — already driven by `backend-diagnostics/diag.sh`.
- `abi twilio simulate` / `abi auth status` — covered by `connector-localcheck` and `auth-localcheck`.
- `abi complete --live` / on-device `apple-fm` — require stored credentials / Apple-Intelligence hardware; not headless-safe. Documented in `complete-base` Gotchas as a manual path.
- Individual MCP tools (`ai_run`, `ai_complete`, `ai_train`, `wdbx_query`, `scheduler_stats`) — the JSON-RPC surface is already exercised end-to-end by `run-abi/smoke.sh` (initialize + tools/list + tools/call).

## Verification (end-to-end)

After each task, its own driver printing `RESULT: PASS` (exit 0) is the gate. After all tasks:
```bash
for s in run-tui/tui.sh wdbx-api-serve/serve.sh wdbx-cluster-serve/cluster-serve.sh; do
  [ -f ".claude/skills/$s" ] && { echo "== $s =="; .claude/skills/$s; echo "exit=$?"; }
done
pgrep -f 'abi wdbx (api|cluster) serve' && echo "LEAK: server still running" || echo "clean: no lingering servers"
grep -c 'run-tui\|wdbx-api-serve\|wdbx-cluster-serve' CLAUDE.md   # docs updated
```
Expect every driver `RESULT: PASS`, no lingering servers, and CLAUDE.md referencing the new skills.
