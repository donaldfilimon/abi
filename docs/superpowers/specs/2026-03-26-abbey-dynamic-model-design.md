# Abbey Dynamic Model — Prompt Pipeline DSL with WDBX Universal Persistence

**Date:** 2026-03-26
**Status:** Implemented

## Problem

The Abbey-Aviva-Abi multi-profile pipeline is wired together procedurally in `router.zig` — each step (routing, modulation, generation, validation, storage) is hard-coded in sequence. This makes it difficult to compose custom pipelines, reuse steps independently, or audit the execution trace. Additionally, modulation state lives only in memory and is lost on process restart.

## Solution

A composable prompt pipeline DSL where each operation is chainable and every step execution is recorded as a WDBX `ConversationBlock` with cryptographic integrity. WDBX becomes the universal persistence layer across modulation, routing, and pipeline execution.

## DSL Syntax

```zig
var builder = abi.ai.pipeline.chain(allocator, "session-123");
var p = builder
    .withChain(&wdbx_chain)
    .retrieve(.wdbx, .{ .k = 5 })          // Pull recent context
    .template("Given {context}, respond: {input}")  // Render prompt
    .route(.adaptive)                        // Profile routing
    .modulate()                              // EMA preference adjustment
    .generate(.{ .mode = .streaming })       // LLM inference
    .validate(.constitution)                 // 6-principle check
    .store(.wdbx)                            // Persist to block chain
    .build();
const result = try p.run("Hello Abbey!");
```

## Architecture

### Pipeline Core (`src/features/ai/pipeline/`)

- **types.zig** — `StepKind` enum (10 variants), `StepConfig` tagged union, `PipelineResult`
- **context.zig** — `PipelineContext` mutable state (fragments, prompt, response, routing, block IDs, metadata)
- **builder.zig** — `PipelineBuilder` chainable API accumulating steps via `ArrayListUnmanaged(Step)`
- **executor.zig** — `Pipeline.run()` iterates steps, records WDBX blocks per step
- **persistence.zig** — `PipelineBlockAdapter` converts step state to `BlockConfig`; `ModulationPersistence` serializes preferences to embeddings
- **mod.zig / stub.zig** — Feature module + API-compatible stub (gated on `feat_reasoning`)

### Step Implementations (`steps/`)

| Step | Wraps | Sets on Context |
|------|-------|-----------------|
| retrieve | `BlockChain.traverseBackward()` | `context_fragments` |
| template | `{variable}` interpolation | `rendered_prompt` |
| route | Heuristic keyword matching | `routing_weights`, `primary_profile` |
| modulate | EMA preference bias | Adjusted `routing_weights` |
| generate | Demo mode (prod: `ClientWrapper`) | `generated_response` |
| validate | Safety pattern scan (prod: `Constitution`) | `validation_passed` |
| store | `BlockChain.addBlock()` | `block_ids` |
| transform | User-provided `fn([]const u8, Allocator) ![]const u8` | `generated_response` |
| filter | User-provided `fn(*const anyopaque) bool` | Early halt |
| reason | Chain-of-thought trace | `metadata["reasoning"]`, `metadata["confidence"]` |

### WDBX Universal Persistence

- **BlockConfig extended** with `pipeline_step: PipelineStepTag`, `pipeline_id: ?u64`, `step_index: ?u16`
- **AdaptiveModulator** gains `attachWdbx()` — write-behind persistence after each `recordInteraction()`
- **MultiProfileRouter** gains `routeAndExecutePipeline()` — full pipeline equivalent of `routeAndExecute()`
- **ConversationMemory** gains `getChainMut()` for pipeline builder attachment

### Design Decisions

- **Runtime builder**: Template strings are runtime data, so steps use `ArrayListUnmanaged` not comptime arrays
- **Write-behind modulation**: In-memory HashMap stays as hot path; WDBX writes are append-only durability
- **Backward compatible**: All existing methods unchanged; pipeline is opt-in via new methods
- **String ownership**: All strings in `PipelineContext` are `allocator.dupe()`-ed; `deinit()` frees all

## Files

### Created (18)
- `src/features/ai/pipeline/{types,context,builder,executor,persistence,mod,stub}.zig`
- `src/features/ai/pipeline/steps/{retrieve,template,route,modulate,generate,validate,store,transform,filter,reason}.zig`
- `test/integration/pipeline_test.zig`, `src/pipeline_mod_test.zig`, `test/pipeline_mod.zig`

### Modified (7)
- `src/core/database/block_chain.zig` — `PipelineStepTag` + 3 fields on `BlockConfig`
- `src/features/ai/{mod,stub}.zig` — Register pipeline sub-feature
- `src/features/ai/modulation.zig` — WDBX write-behind
- `src/features/ai/profile/router.zig` — `routeAndExecutePipeline()`
- `src/features/ai/profile/memory.zig` — `getChainMut()`
- `build/validation.zig` — `pipeline-tests` lane

## Verification

- 3497/3501 tests pass (4 skipped)
- 3744/3748 parity tests pass
- `zig build pipeline-tests` focused lane works
- `zig build check-parity` clean
