# AI Scheduler/Memory Integration + Advanced Stateful Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deeply integrate the core Scheduler and MemoryTracker inside `features/ai` for real training and completion workloads, add comprehensive tests for the new paths, and scaffold one additional advanced stateful feature with submodules using the modern validated process.

**Architecture:** Treat Scheduler as the cooperative task submission mechanism for AI work (training jobs, completion pipelines). Use MemoryTracker for visibility into allocation hotspots inside AI paths. New advanced feature will follow the exact mod/stub + types + stateful Context + submodules pattern proven with `metrics`.

**Tech Stack:** Zig 0.17, existing `core/scheduler.zig`, `core/memory.zig`, `features/ai`, `./build.sh` gates, mod/stub discipline.

**Status note (2026-06-09):** The AI scheduler/helper work, production call-site wiring, and test expansion have since landed in the broader workspace. The "advanced stateful feature" track materialized as the default-on `telemetry` feature; the active WDBX follow-through now lives in `tasks/todo.md` and `docs/spec/wdbx-north-star.md`.

---

## PR Plan

### PR 1: Core AI Scheduler & Memory Integration Helpers (mod + stub parity)

**Description:** Add the internal submission and instrumentation helpers inside `features/ai` so that training and completion paths can submit work to a Scheduler and be tracked by a MemoryTracker. This is the foundational surface — pure API addition with full mod/stub parity.

**Files/components affected:**
- Create/Modify: `src/features/ai/mod.zig`, `src/features/ai/pipeline.zig`
- Modify: `src/features/ai/stub.zig` (must mirror all new symbols)
- Modify: `src/features/ai/mod.zig` (add new public or re-exported helpers)
- Tests: `src/features/ai/mod.zig` (unit tests for the new helpers)

- [ ] **Step 1: Add Scheduler submission helper in AI**
   ```zig
   pub fn submitTrainingTask(sched: *Scheduler, profile: Profile, dataset: Dataset, artifact_dir: []const u8) !u64 {
       // wrap the existing training work as a high-priority Task
   }
   ```
   (Similar helper for completion paths.)

- [ ] **Step 2: Add optional MemoryTracker instrumentation points**
   Instrument hot embedding/vector paths inside AI with an optional `*MemoryTracker`.

- [ ] **Step 3: Ensure full mod + stub parity**
   Every new symbol in `mod.zig` must have an identical declaration in `stub.zig` (stub returns `error.FeatureDisabled` or no-op versions).

- [ ] **Step 4: Add basic unit tests for the new helpers**
   Test submission, stats visibility, and disabled behavior via stub.

- [ ] **Step 5: Run full gate**
   ```bash
   ./build.sh check
   zig build check-parity
   ```

- [ ] **Step 6: Commit**
   ```bash
   git add src/features/ai/
   git commit -m "feat(ai): add scheduler submission + memory tracking helpers (mod+stub)"
   ```

### PR 2: Wire real AI training & completion call sites to use Scheduler

**Description:** Update the actual call sites in CLI handlers and MCP so that training and completion work is submitted through the new helpers when a scheduler context is available. Preserve all existing public signatures.

**Files/components affected:**
- Modify: `src/cli/handlers/agent.zig`, `src/cli/handlers/train.zig` (and any other AI entry points)
- Modify: `src/mcp/handlers.zig`
- Modify: `src/features/ai/mod.zig` (internal wiring)
- Tests: integration tests that exercise the paths

- [ ] **Step 1: Pass scheduler context from CLI entry points**
   In `agent train` and similar handlers, obtain the long-lived Scheduler (owned at dispatch or main level) and pass it into the AI calls.

- [ ] **Step 2: Update MCP tool handlers**
   Do the same for `ai_train` and `ai_complete` / `ai_run` paths.

- [ ] **Step 3: Internal dispatch in AI**
   Make `train*` and `complete*` functions check for a provided scheduler and call `submitTrainingTask` / equivalent instead of direct execution when available.

- [ ] **Step 4: Keep full backward compatibility**
   All public APIs (`train`, `complete`, etc.) must continue to work exactly as before when no scheduler is supplied.

- [ ] **Step 5: Run gates and manual smoke**
   ```bash
   ./build.sh check
   zig-out/bin/abi agent train ...   # verify it uses scheduler when available
   ```

- [ ] **Step 6: Commit**
   ```bash
   git add src/cli/handlers/ src/mcp/handlers.zig src/features/ai/
   git commit -m "feat: wire AI training/completion through scheduler (CLI + MCP)"
   ```

### PR 3: Comprehensive tests for the new AI scheduler/memory paths

**Description:** Add the missing unit, integration, and contract tests that prove the integration works and remains stable under feature flags.

**Files/components affected:**
- Create/Modify: `src/features/ai/mod.zig` (more unit tests)
- Create: new integration test(s) under `tests/` or `src/integration_tests.zig`
- Modify: `tests/contracts/feature_modules.zig`, `tests/contracts/surface.zig`, `tests/contracts/mcp_tools.zig`

- [ ] **Step 1: Unit tests for submission + stats**
   Add tests that create a real Scheduler, submit AI-flavored work, and assert stats increase.

- [ ] **Step 2: Integration test exercising real paths**
   End-to-end test that triggers training/completion through the normal CLI/MCP surfaces and observes both scheduler counts and MemoryTracker usage.

- [ ] **Step 3: Contract test updates**
   Update feature modules contract to cover the new AI symbols when the feature is on/off. Add any new MCP tools or CLI surfaces if exposed.

- [ ] **Step 4: Feature-off matrix verification**
   ```bash
   ./build.sh check
   zig build test-feature-contracts -Dfeat-ai=false
   ```

- [ ] **Step 5: Commit**
   ```bash
   git add tests/ src/features/ai/
   git commit -m "test: comprehensive coverage for AI scheduler/memory integration"
   ```

### PR 4: Scaffold one new advanced stateful feature with submodules (implemented as `telemetry`)

**Description:** Using the exact modern scaffolding process validated with `hash` and `metrics`, add one additional advanced feature that is stateful (has init/deinit owning state), contains at least one submodule that requires mod/stub parity, and demonstrates cross-import wiring (e.g. into scheduler or AI).

**Files/components affected:**
- Create: `src/features/<new-feature>/` (mod.zig, stub.zig, types.zig + at least one submodule)
- Modify: `build.zig`, `src/features/mod.zig`, `tools/check_feature_stubs.sh`
- Modify: `tests/contracts/feature_modules.zig`
- Docs: `docs/spec/abi-refactor-design.md`, `tasks/roadmap-next.md`

- [ ] **Step 1: Choose and name the feature**
   (Example candidates: evaluation harness, telemetry/metrics extension, or another observability-focused stateful module. Confirm with user if needed.)

- [ ] **Step 2: Create the feature using the modern checklist**
   - types.zig with shared Error set
   - stateful Context (init/deinit)
   - At least one submodule with its own mod/stub pair
   - Full mod + stub parity
   - isEnabled() correctly gated

- [ ] **Step 3: Wire the feature**
   - Add to `build.zig` (default on or off as appropriate)
   - Add to `src/features/mod.zig`
   - Add to `tools/check_feature_stubs.sh`
   - Optional light cross-import (e.g. AI or scheduler can use it for observability)

- [ ] **Step 4: Add contract coverage**
   Update `tests/contracts/feature_modules.zig`

- [ ] **Step 5: Full verification**
   ```bash
   ./build.sh check
   zig build check-parity
   zig build test-feature-contracts -Dfeat-<new-feature>=false
   ```

- [ ] **Step 6: Update documentation**
   Refresh design doc and roadmap.

- [ ] **Step 7: Commit**
   ```bash
   git add src/features/<new-feature>/ build.zig src/features/mod.zig ...
   git commit -m "feat: add new advanced stateful feature <name> with submodules"
   ```

---

## Verification Gates (apply after every PR)

```bash
./build.sh check
zig build check-parity
zig build test-feature-contracts -Dfeat-ai=false
zig build test-feature-contracts -Dfeat-<new-feature>=false   # when applicable
```

**Success signals:**
- Real scheduler submission + MemoryTracker usage visible and tested inside AI training/completion.
- New advanced feature passes full on/off matrix and parity.
- No regressions in existing public AI, WDBX, CLI, or MCP surfaces.
- All changes follow AGENTS.md rules (./build.sh on macOS, mod/stub sync, relative imports, no hand-edits to generated files).

**Manual smoke:**
- `zig-out/bin/abi agent train ...`
- `zig-out/bin/abi-mcp` + relevant tools
- `zig build test -- --test-filter "ai.*scheduler|MemoryTracker"`

---

*Plan created from the detailed implementation plan in the session plan.md (2026-05-27). Ready for subagent-driven execution or /execute-plan.*
