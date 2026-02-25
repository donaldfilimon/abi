# AGENTS.md

Instructions for AI assistants (Claude, Codex, Cursor, Gemini) working in this repository.

## Workflow Orchestration

### 1. Plan Node Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
- Present plan for approval before implementation on high-stakes changes

### 2. Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution
- Aggregate and synthesize subagent results before proceeding

### 3. Self-Improvement Loop

- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project
- Patterns to capture: root causes, not just symptoms

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness
- For UI changes: verify visually; for API changes: test the endpoint

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Simplicity is the ultimate sophistication

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Investigate root cause; fix the disease, not the symptom

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

---

## Code Style (Zig 0.16)

- **Formatting**: 4 spaces, no tabs; lines ~100 chars; `zig fmt .` on every save
- **Naming**: `PascalCase` types/enums, `camelCase` functions/vars, `snake_case.zig` files
- **Imports**: Explicit only — never `usingnamespace`
- **Collections**: Prefer `std.ArrayListUnmanaged(T) = .empty` — allocator per-call
- **Logging**: `std.log.*` in library code; `std.debug.print` only in CLI/TUI display
- **CLI output**: Always `utils.output.printError`/`printInfo` for user-facing messages
- **Error cleanup**: `errdefer`, not `defer`, when returning allocated values
- **File footer**: End every source file with `test { std.testing.refAllDecls(@This()); }`

## Testing

- `zig build test --summary all` — main suite (1290 pass, 6 skip)
- `zig build feature-tests --summary all` — feature inline tests (2360 pass, 5 skip)
- `zig test src/path/to/file.zig` — single file
- Initialize test structs with `std.mem.zeroes(T)`, never `= undefined`
- Skip hardware-gated tests with `error.SkipZigTest`
- Use `test { _ = @import(...); }` for discovery — NOT `comptime { }`

## Stubs

- Every `mod.zig` has a matching `stub.zig` — always update both
- Stub functions: discard params with `_`, return `error.FeatureDisabled`
- Use `StubContext(ConfigT)` from `src/core/stub_context.zig`

## TUI Development

- Panel vtable: `render`, `tick`, `handleEvent`, `name`, `shortcutHint`, `deinit`
- Use `term.moveTo(row, col)` for positioning
- File I/O in panels: `std.c.fopen`/`std.c.fread` (no I/O backend needed)
- Call `flush()` once per frame — `Terminal.write()` buffers internally

## Commits and PRs

- Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `ci:`
- Keep commits scoped; avoid mixing refactors with behavior changes
- PRs need: clear summary, linked issue, passing `zig build full-check`
