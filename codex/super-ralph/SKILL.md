---
name: super-ralph
description: Run the ABI Ralph iterative loop (native Zig). One-shot power: abi ralph super. Full control: init, run, improve, skills, gate. Multi-Ralph via ralph_multi API.
---

# Super Ralph (Codex)

Run the **Ralph** iterative agent loop from the ABI framework. Install once; then invoke Ralph from any project that has the ABI CLI on PATH.

---

## Install (one-time)

From the ABI repo root:

```bash
./scripts/install_super_ralph_codex.sh
```

Or manually:

```bash
mkdir -p "$HOME/.codex/skills/super-ralph"
cp -r codex/super-ralph/* "$HOME/.codex/skills/super-ralph"
```

**Verify:** `abi ralph status` (should show workspace or prompt to run `abi ralph init`).

---

## One-shot power: `abi ralph super`

Best for autonomous multi-step tasks with optional quality gate:

```bash
abi ralph super --task "Your goal here"
abi ralph super --task "..." --auto-skill    # Run + extract lesson into Abbey memory
abi ralph super --task "..." --gate          # Run then run quality gate on report JSON
abi ralph super --task "..." -i 20           # Override max iterations
```

If no Ralph workspace exists, `super` runs `abi ralph init` first, then runs the loop.

---

## Full control

| Action | Command |
|--------|---------|
| Create workspace | `abi ralph init` |
| Run from PROMPT.md | `abi ralph run` |
| Run with inline task | `abi ralph run --task "goal"` |
| Run + store lesson | `abi ralph run --task "..." --auto-skill` |
| Self-improvement pass | `abi ralph improve` |
| List / add / clear skills | `abi ralph skills`, `abi ralph skills add "lesson"`, `abi ralph skills clear` |
| Status | `abi ralph status` |
| Quality gate | `abi ralph gate` (input/output paths via `--in`, `--out`) |
| **Multi-agent (parallel)** | `abi ralph multi -t "g1" -t "g2"` (Zig threads + lock-free bus) |

Requires ABI built and on PATH (e.g. `zig build run -- ralph help`).

---

## Power workflows

1. **Improve then run** — `abi ralph improve` (stores a lesson), then `abi ralph run --task "..."` so the new skill is in context.
2. **Gate then commit** — `abi ralph super --task "..." --gate`; only proceed if gate passes (exit 0).
3. **Chained runs** — Run once, edit PROMPT.md, run again; skills from the first run apply to the second.

---

## Multi-Ralph (Zig, fast multithreading)

- **CLI:** `abi ralph multi -t "goal1" -t "goal2"` — runs N Ralph agents in parallel (ThreadPool + lock-free RalphBus).
- **Zig API:** `ralph_multi` (RalphBus, RalphMessage) and `ralph_swarm` (ParallelRalphContext, parallelRalphWorker). Use for handoff, skill_share, task_result between agents.

---

## References

- ABI repo: `tools/cli/commands/ralph.zig`, `src/features/ai/abbey/ralph_multi.zig`
- CLAUDE.md (in repo): Working outside the Ralph loop, Super Ralph power use
