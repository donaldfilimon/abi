# Global Agents Registry

This file documents the personas active within the ABI Framework.

1. **Abbey (`ABBEY.md`):** High-level orchestrator.
2. **Aviva (`AVIVA.md`):** Dense executor.
3. **Claude (`CLAUDE.md`):** External workflow manager.
4. **Gemini (`GEMINI.md`):** Migration specialist CLI.

For underlying structural layout rules, consult `ABI.md`.

## Global Multi-CLI Consensus

### 0. Always Run Parallel Consensus

• For every non-trivial task, run `/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh --mode <code|vision> --timeout-sec 120 --prompt-file <file> --out-dir <dir>` before implementation.
• Treat `run_tricli_consensus.sh` as the compatibility wrapper around `run_consensus.py`.
• Use `--mode code` for normal tasks and `--mode vision` for image/screenshot-heavy tasks.
• Keep `--timeout-sec 120` per tool.
• Treat this as mandatory best-effort: continue with surviving tools if one fails.
• Use Codex as final arbiter when outputs disagree.

## Workflow Orchestration

### 1. Plan Node Default

• Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
• If something goes sideways, STOP and re-plan immediately - don't keep pushing
• Use plan mode for verification steps, not just building
• Write detailed specs upfront to reduce ambiguity
• Present plan for approval before implementation on high-stakes changes

### 2. Subagent Strategy

• Use subagents liberally to keep main context window clean
• Offload research, exploration, and parallel analysis to subagents
• For complex problems, throw more compute at it via subagents
• One task per subagent for focused execution
• Aggregate and synthesize subagent results before proceeding

### 3. Self-Improvement Loop

• After ANY correction from the user: update `tasks/lessons.md` with the pattern
• Write rules for yourself that prevent the same mistake
• Ruthlessly iterate on these lessons until mistake rate drops
• Review lessons at session start for relevant project
• Patterns to capture: root causes, not just symptoms

### 4. Verification Before Done

• Never mark a task complete without proving it works
• Diff behavior between main and your changes when relevant
• Ask yourself: "Would a staff engineer approve this?"
• Run tests, check logs, demonstrate correctness
• For UI changes: verify visually; for API changes: test the endpoint

### 5. Demand Elegance (Balanced)

• For non-trivial changes: pause and ask "is there a more elegant way?"
• If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
• Skip this for simple, obvious fixes - don't over-engineer
• Challenge your own work before presenting it
• Simplicity is the ultimate sophistication

### 6. Autonomous Bug Fixing

• When given a bug report: just fix it. Don't ask for hand-holding
• Point at logs, errors, failing tests - then resolve them
• Zero context switching required from the user
• Go fix failing CI tests without being told how
• Investigate root cause; fix the disease, not the symptom

---

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

---

## Core Principles

| Principle | Description |
|-----------|-------------|
| **Simplicity First** | Make every change as simple as possible. Minimal code impact. |
| **No Laziness** | Find root causes. No temporary fixes. Senior developer standards. |
| **Minimal Impact** | Changes should only touch what's necessary. Avoid introducing bugs. |
| **Review Lessons** | Review `lessons.md` at session start for the relevant project. |

> **Note**: AI responses may include mistakes. Always verify critical changes.
