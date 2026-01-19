# Ralph Prompt Planning (Iterative Agent Loop)
> **Codebase Status:** Synced with repository as of 2026-01-18.

> **Date:** January 17, 2026
> **Status:** Implemented ✅
> **Target:** Zig 0.16 / Abbey Framework

## Objective

Implement the **Ralph Pattern** (Recursive Agent Loop for Poly-step Heuristics) within the Abbey framework. This feature enables an agent to enter a self-correcting, iterative loop to complete complex tasks (refactoring, large-scale code generation, rigorous testing) that exceed a single inference pass.

The core concept is to intercept the agent's "stop" signal and re-prompt it with updated context and a directive to verify/continue its work until a strict completion condition is met.

## Core Components

### 1. The "Ralph" Persona
A specialized persona variant optimized for endurance and attention to detail.

- **Role:** Tireless Worker / Refactor Specialist
- **Traits:** Thorough, iterative, self-critical, non-conversational (in loop).
- **Location:** `src/features/ai/prompts/personas.zig` (add `.ralph` variant)

### 2. Prompt Architecture

The Ralph loop requires a specific sequence of prompts:

#### A. Initiation Prompt (System + User Task)
Sets the stage for a multi-step operation.
*   "You are Ralph. Your goal is to X. Do not stop until X is verified complete."

#### B. The "Keep Going" Prompt (Loop Injection)
Injected when the agent attempts to finish or pause.
*   *Template:* "You have completed step {N}. Review your work against criteria {C}. If incomplete, continue to step {N+1}. If complete, verify again."
*   *Dynamic Context:* Must include a summary of changes made in the last step.

#### C. The "Stop Hook" Prompt (Verification)
Triggered when the agent claims completion.
*   *Template:* "You state the task is done. Please output a JSON summary of verification tests run. If no tests were run, resume and write tests."

### 3. State Management (Zig 0.16)

The loop state must be managed efficiently using `std.ArrayListUnmanaged` and explicit allocators.

```zig
const RalphState = struct {
    iteration: usize,
    max_iterations: usize,
    changes_log: std.ArrayListUnmanaged([]const u8),
    last_context_hash: u64,
    // ...
};
```

### 4. Integration Points

*   **Engine:** `src/features/ai/abbey/engine.zig` - Add `runRalphLoop()` method.
*   **Prompts:** `src/features/ai/prompts/ralph.zig` (New file) - Store specific prompt templates.
*   **CLI:** `tools/cli/commands/agent.zig` - Add `agent ralph --task "..."` command.

## Implementation Plan

### Phase 1: Prompt Design (Complete)
- [x] Define the `ralph` persona in `personas.zig`.
- [x] Create `src/features/ai/prompts/ralph.zig` with format strings for loop injections.
- [x] Design the "Critic" prompt that evaluates completion.

### Phase 2: Engine Support
- [ ] Implement `RalphLoop` struct in `engine.zig`.
- [ ] Add support for "Stop Hooks" (intercepting `[DONE]` token).
- [ ] Implement context window sliding/summarization for long loops.

### Phase 3: Tooling
- [ ] Add CLI command `zig build run -- agent ralph ...`.
- [ ] Add TUI visualizer for loop progress.

## Prompt Drafts

### System Prompt (Draft)
```text
You are Ralph, an iterative engineering agent.
You do not aim for speed; you aim for precision and completeness.
When given a task, break it down into atomic steps.
Execute one step at a time.
After each step, assess if the entire task is complete.
If not, proceed to the next step.
NEVER summarize or chat. Output WORK only.
```

### Loop Injection (Draft)
```text
[SYSTEM]
Iteration {i} complete.
Files modified: {files}.
Errors detected: {errors}.
Task status: IN_PROGRESS.
Instruction: Proceed to the next logical step. Fix any errors found.
```

## Next Steps

1.  Create `src/features/ai/prompts/ralph.zig`.
2.  Update `src/features/ai/prompts/personas.zig`.
3.  Prototype the loop in `src/features/ai/abbey/engine.zig`.
