---
name: pipeline-dsl
description: Use when working with the Abbey Dynamic Model pipeline DSL — building pipelines, adding steps, wiring WDBX persistence, modifying pipeline context, or debugging pipeline execution. Trigger when user mentions "pipeline", "chain", "PipelineBuilder", "PipelineContext", "pipeline step", "WDBX persistence", or edits files under src/features/ai/pipeline/.
---

# Abbey Dynamic Model — Pipeline DSL

Composable prompt pipeline for the ABI multi-profile AI system. Each step is a typed operation; every execution is recorded as a WDBX block with cryptographic integrity.

## Quick Reference

```zig
const pipeline = @import("abi").ai.pipeline; // or @import("pipeline/mod.zig") from within src/

var builder = pipeline.chain(allocator, "session-123");
var p = builder
    .withChain(&wdbx_chain)           // attach WDBX for persistence
    .retrieve(.wdbx, .{ .k = 5 })    // pull recent context
    .template("Given {context}, respond to: {input}")
    .route(.adaptive)                  // profile routing
    .modulate()                        // EMA preference adjustment
    .generate(.{})                     // LLM inference
    .validate(.constitution)           // 6-principle check
    .store(.wdbx)                      // persist everything
    .build();
defer p.deinit();

const result = try p.run("Hello Abbey!");
defer result.deinit();
```

## Key Files

| File | Purpose |
|------|---------|
| `src/features/ai/pipeline/mod.zig` | Entry point, re-exports all types |
| `src/features/ai/pipeline/stub.zig` | API-compatible no-ops (feat_reasoning=false) |
| `src/features/ai/pipeline/types.zig` | StepKind, StepConfig, PipelineResult, per-step configs |
| `src/features/ai/pipeline/context.zig` | PipelineContext — mutable state between steps |
| `src/features/ai/pipeline/builder.zig` | PipelineBuilder — chainable DSL |
| `src/features/ai/pipeline/executor.zig` | Pipeline — runs steps, records WDBX blocks |
| `src/features/ai/pipeline/persistence.zig` | PipelineBlockAdapter, ModulationPersistence |
| `src/features/ai/pipeline/steps/` | 10 step implementations |

## Feature Gating

Gated on `feat_reasoning` (same as abbey/constitution). In `src/features/ai/mod.zig`:
```zig
pub const pipeline = if (build_options.feat_reasoning)
    @import("pipeline/mod.zig")
else
    @import("pipeline/stub.zig");
```

When `feat_reasoning=false`, the stub compiles but `run()` returns `error.FeatureDisabled`.

## Step Types (10 total)

| Step | Kind | Config Struct | Purpose |
|------|------|---------------|---------|
| retrieve | `.retrieve` | `RetrieveConfig` | Pull k recent WDBX blocks with recency decay |
| template | `.template` | `TemplateConfig` | `{variable}` interpolation (input, context, profile) |
| route | `.route` | `RouteConfig` | Heuristic/adaptive profile routing → abbey/aviva/abi |
| modulate | `.modulate` | `ModulateConfig` | EMA preference adjustment of routing weights |
| generate | `.generate` | `GenerateConfig` | LLM inference (real client or demo echo) |
| validate | `.validate` | `ValidateConfig` | Constitution 6-principle check |
| store | `.store` | `StoreConfig` | Persist full pipeline state to WDBX block chain |
| transform | `.transform` | `TransformConfig` | User callback `fn([]const u8, Allocator) ![]const u8` |
| filter | `.filter` | `FilterConfig` | User predicate — halt or continue pipeline |
| reason | `.reason` | `ReasonConfig` | Chain-of-thought with confidence output |

## Critical Patterns

### String Ownership

**All strings in PipelineContext are heap-allocated.** Never store string literals directly.

```zig
// WRONG — will crash on deinit
pctx.generated_response = "hello";

// CORRECT — dupe to heap
pctx.generated_response = try allocator.dupe(u8, "hello");
```

`PipelineContext.deinit()` frees: `input`, `session_id`, all `context_fragments`, `rendered_prompt`, `generated_response`, all metadata keys/values.

### Pipeline Owns session_id

`Pipeline.init()` dupes `session_id` — safe to deinit the builder before running:
```zig
var builder = pipeline.chain(allocator, "session-123");
var p = builder.build();
builder.deinit(); // safe — pipeline has its own copy
defer p.deinit();
const result = try p.run("input");
```

### Stack-Local Embedding Safety

When passing `[4]f32` embedding arrays to `BlockConfig`, the array MUST live in the same stack frame as `addBlock()`:
```zig
// WRONG — dangling pointer after function returns
fn makeConfig() BlockConfig {
    var embedding = hashEmbedding(text);
    return .{ .query_embedding = &embedding }; // dangling!
}

// CORRECT — same stack frame
var embedding = hashEmbedding(text);
const config = BlockConfig{ .query_embedding = &embedding };
const block_id = try chain.addBlock(config); // embedding still alive
```

### CompletionResponse Cleanup

When using real LLM client in generate step, always deinit the response:
```zig
if (wrapper.complete(request)) |resp_val| {
    var resp = resp_val;
    defer resp.deinit(pctx.allocator); // frees content, model, tool_calls
    const duped = try pctx.allocator.dupe(u8, resp.content);
    pctx.generated_response = duped;
}
```

### Metadata Error Propagation

`setMetadata` propagates `fetchPut` errors — don't swallow OOM:
```zig
// CORRECT — uses try, errdefer guards activate on OOM
try pctx.setMetadata("key", "value");
```

### Timer Usage (Zig 0.17)

`std.time.timestamp()` and `std.time.nanoTimestamp()` are removed. Use:
```zig
const foundation = @import("../../../foundation/mod.zig");

// Elapsed time measurement
var timer = foundation.time.Timer.start() catch null;
// ... work ...
const elapsed_ms: u64 = if (timer) |*t| t.read() / std.time.ns_per_ms else 0;

// Wall-clock timestamps
const now = foundation.time.nowSeconds();    // i64
const now_ms = foundation.time.unixMs();     // i64

// Unique IDs
const unique = foundation.time.getUniqueId(); // u64
```

## Adding a New Step

1. Create `src/features/ai/pipeline/steps/<name>.zig`:
   ```zig
   const std = @import("std");
   const types = @import("../types.zig");
   const ctx_mod = @import("../context.zig");
   const PipelineContext = ctx_mod.PipelineContext;

   pub fn execute(pctx: *PipelineContext, cfg: types.YourConfig) !void {
       // Read from pctx, write results back to pctx
   }
   ```

2. Add config struct to `types.zig`:
   ```zig
   pub const YourConfig = struct {
       // fields with defaults
   };
   ```

3. Add variant to `StepKind` enum and `StepConfig` union in `types.zig`.

4. Wire in `executor.zig` `executeStep()` switch.

5. Add builder method in `builder.zig`.

6. Add matching no-op in `stub.zig` PipelineBuilder.

7. Run `zig build check-parity` to verify stub alignment.

## Testing

```bash
zig build pipeline-tests          # Focused pipeline test lane
zig build test --summary all -- --test-filter "pipeline"  # Filter by name
```

Test files:
- `src/features/ai/pipeline/pipeline_test.zig` — unit tests
- `test/integration/pipeline_test.zig` — integration tests via `@import("abi")`

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| String literal in PipelineContext | SIGSEGV on deinit | `allocator.dupe()` all strings |
| Missing `resp.deinit()` in generate | Memory leak per LLM call | `defer resp.deinit(pctx.allocator)` |
| Embedding in separate function | Dangling pointer to stack | Keep in same frame as `addBlock` |
| `std.time.timestamp()` | Compile error in Zig 0.17 | Use `foundation.time.nowSeconds()` |
| Swallowing `fetchPut` error | Silent memory leak | Use `try` to propagate |
| Missing stub `withChain` | Compile error when feat disabled | Add no-op to stub PipelineBuilder |
