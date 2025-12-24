# Repository Guidelines for Agentic Coding Agents

## Project Structure
```
src/
├─ compute/           # High-performance compute runtime
│  ├─ runtime/         # Engine, workloads, configuration
│  ├─ concurrency/     # Lock-free data structures (Chase-Lev, ShardedMap)
│  ├─ memory/          # Allocators, buffer management
│  ├─ gpu/             # GPU backends (feature-gated)
│  ├─ network/         # Distributed compute (feature-gated)
│  └─ profiling/       # Metrics collection (feature-gated)
├─ core/               # Core infrastructure
├─ features/           # AI, GPU, database, web features
├─ framework/          # Orchestration, config, runtime
└─ shared/             # Logging, platform, utilities
```

## Build, Test, Lint Commands

| Command | Description |
|---------|-------------|
| `zig build` | Build project |
| `zig build test` | Run all tests |
| `zig test src/compute/runtime/engine.zig` | Run tests for specific file |
| `zig test --test-filter="engine init"` | Run tests matching pattern |
| `zig fmt .` | Format all source files |
| `zig fmt --check .` | Check formatting without modifying |

### Feature Flags
| Flag | Default |
|------|---------|
| `-Denable-gpu` | `true` |
| `-Denable-network` | `false` |
| `-Denable-profiling` | `false` |
| `-Denable-ai` | `true` |
| `-Denable-web` | `true` |
| `-Denable-database` | `true` |

**Example:** `zig build test -Denable-profiling=true -Denable-network=true`

## Coding Style

### Formatting
- **Indentation:** 4 spaces, no tabs
- **Line Length:** Under 100 characters when possible
- **Encoding:** UTF-8, LF line endings
- Run `zig fmt .` before committing

### Naming Conventions
- **Types:** PascalCase (`EngineConfig`, `WorkloadVTable`)
- **Functions/Variables:** snake_case (`init_engine`, `execute_task`)
- **Constants:** SCREAMING_SNAKE_CASE (`MAX_BUFFER_SIZE`, `EMPTY`)
- **Modules:** snake_case (`mod.zig`, `engine.zig`)

### Imports
```zig
const std = @import("std");
const build_options = @import("build_options");
const concurrency = @import("../concurrency/mod.zig");
const workload = @import("workload.zig");
```

**Rules:**
- Group std imports first, then internal
- Avoid `usingnamespace` - use explicit imports
- Prefer qualified access (`std.mem.Allocator`) over aliases

### Feature Gating Pattern
```zig
const ProfilingModule = if (build_options.enable_profiling)
    struct {
        pub const MetricsCollector = @import("../profiling/mod.zig").MetricsCollector;
    }
else
    struct {
        pub const MetricsCollector = void;
    };

// In struct field:
metrics_collector: if (build_options.enable_profiling) ?*ProfilingModule.MetricsCollector else void,
```

### Error Handling
```zig
pub fn riskyOperation(allocator: std.mem.Allocator) !Result {
    errdefer {
        // Cleanup on error
    }

    const resource = try allocateResource(allocator);
    errdefer deallocateResource(resource);

    return result;
}
```

**Rules:**
- Use `!T` for fallible operations
- Use specific error enums over generic errors
- Prefer explicit error returns over panics
- Use `defer`/`errdefer` for cleanup

### Memory Management
**Critical Pattern - Compute Engine:**
- **Stable allocator** (`Engine.allocator`): Long-lived allocations (results, work items)
- **Worker arenas** (`Worker.arena`): Per-thread scratch memory, reset after each task
- **Results allocated from stable allocator, NOT worker arena** (prevents dangling pointers)

```zig
// Correct: result from stable allocator
const result_ptr = try item.vtable.exec(item.user, &ctx, engine.allocator);

// Worker arena reset after task execution
defer _ = worker.arena.reset(.retain_capacity);
```

**Rules:**
- Always pass allocators explicitly
- Document ownership semantics in function docs
- Use arena allocators for scoped operations
- Reset arenas, never destroy mid-session

## Zig 0.16 Specific APIs

**Use these:**
- `cmpxchgStrong` (NOT `compareAndSwap`)
- `std.time.Timer` for timing
- `std.atomic.spinLoopHint()` for spinning
- `std.Thread.spawn` for thread creation

**Avoid these (deprecated/not available in 0.16):**
- `compareAndSwap` (use `cmpxchgStrong`)
- `std.Thread.sleep` (use spin loops)
- `std.atomic.fence` (use acquire/release ordering)

## Testing

### Test Organization
- Co-locate tests near source: `src/compute/runtime/engine_test.zig`
- Use `mod.zig` to export test modules if needed

### Test Patterns
```zig
test "engine initialization succeeds" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var engine = try compute.Engine.init(gpa.allocator(), config);
    defer engine.deinit();

    try std.testing.expect(engine.running.load(.acquire));
}
```

### Running Tests
```bash
# Run all tests
zig build test

# Run specific module tests
zig test src/compute/runtime/engine.zig

# Run tests matching pattern
zig test --test-filter="engine init"
```

## Module Conventions

- Use `mod.zig` as re-export surface for submodules
- Feature-gated optional modules: `if (build_options.enable_feature) @import(...) else struct { pub const Type = void; }`
- Keep modules focused and single-purpose
- Avoid circular imports between feature modules
- Shared helpers belong in `src/shared/`

## Pre-Commit Checklist
1. `zig build` - ensure compilation
2. `zig build test` - all tests pass
3. `zig fmt --check .` - formatting correct
4. Test feature flags if modified

## Development Patterns

### Worker Thread Pattern
```zig
const Worker = struct {
    id: u32,
    local_deque: ChaseLevDeque,
    arena: std.heap.ArenaAllocator,
    thread: ?std.Thread,
    engine: *Engine,
};

fn workerMain(worker: *Worker) void {
    while (worker.engine.running.load(.acquire)) {
        var task_id = worker.local_deque.popBottom();
        if (task_id == null) task_id = trySteal(worker);
        if (task_id) |id| executeTask(worker, id) catch |err| { /* handle */ };
    }
}
```

### Work-Stealing Protocol
- Workers pop from **bottom** of local deque
- Workers steal from **top** of other workers' deques
- Use `ChaseLevDeque` for lock-free work distribution

## Commit Guidelines
- Format: `<type>(<scope>): <summary>`
- Types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`
- Example: `feat(compute): add GPU work-stealing scheduler`
