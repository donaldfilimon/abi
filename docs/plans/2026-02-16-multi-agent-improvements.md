# Multi-Agent Module Improvements

## Context

The `src/features/ai/multi_agent/` module (1,305 LOC, 4 files) provides a `Coordinator`
that orchestrates multiple `agents.Agent` instances across execution strategies (sequential,
parallel, pipeline, adaptive) with aggregation strategies (concatenate, vote, select_best,
merge, first_success).

**Problems identified:**
1. **Thread join bug** — `executeParallel` has broken join logic that can skip joining spawned threads or attempt to join unspawned ones
2. **No timeout enforcement** — `agent_timeout_ms` config exists but is never applied
3. **No resilience** — No retry or circuit-breaker; agent failures silently produce error strings
4. **CLI is simulated** — `tools/cli/commands/multi_agent.zig` never registers real agents
5. **No integration tests** — Only inline unit tests exist
6. **Raw thread usage** — Uses `std.Thread.spawn` directly instead of the existing `runtime.ThreadPool`

**Reusable components found:**
- `src/services/runtime/scheduling/thread_pool.zig` — Work-stealing pool, <1us dispatch
- `src/services/runtime/concurrency/channel.zig` — Lock-free MPMC Vyukov channel
- `src/services/shared/utils/retry.zig` — `RetryConfig` + `calculateDelay()` (already in AgentConfig)
- `src/features/ai/streaming/circuit_breaker.zig` — Lock-free atomic circuit breaker
- `src/services/shared/utils/structured_error.zig` — Fixed-size error Context

---

## Phase 1: Fix Thread Join Bug + Timeout Enforcement

**Files:** `src/features/ai/multi_agent/mod.zig`

### Task 1.1: Fix `executeParallel` thread join logic

The current code tracks a `spawned` counter but joins using an unrelated condition:
```zig
// BUG: checks response/success instead of tracking which threads were spawned
for (0..agent_count) |i| {
    if (thread_results[i].response != null or thread_results[i].success) {
        if (i < spawned) {
            threads[i].join();
        }
    }
}
```

**Fix:** Track spawn success per-thread with a boolean array:
```zig
const spawned = self.allocator.alloc(bool, agent_count) catch return Error.ExecutionFailed;
defer self.allocator.free(spawned);
@memset(spawned, false);

// In spawn loop:
spawned[i] = true;  // only set on successful spawn

// In join loop — unconditionally join all spawned threads:
for (0..agent_count) |i| {
    if (spawned[i]) threads[i].join();
}
```

### Task 1.2: Add timeout enforcement to agent execution

Add a deadline check in `runAgentThread`. Since Zig doesn't have thread cancellation,
use a cooperative timeout: check elapsed time after `process()` returns and mark as
timed out if exceeded.

For truly blocking agents, wrap with a timeout using `std.Thread.Futex` or simply
record the duration and mark timeout in the result:

```zig
const ThreadResult = struct {
    response: ?[]u8 = null,
    success: bool = false,
    duration_ns: u64 = 0,
    timed_out: bool = false,  // NEW
};
```

In `runAgentThread`, check duration against timeout after `process()` returns:
```zig
const dur = if (timer) |*t| t.read() else 0;
const timed_out = timeout_ns > 0 and dur > timeout_ns;
result.* = .{
    .response = if (timed_out) null else response,
    .success = !timed_out,
    .duration_ns = dur,
    .timed_out = timed_out,
};
if (timed_out and response.len > 0) allocator.free(response);
```

Pass `timeout_ns` to `runAgentThread` from `config.agent_timeout_ms * 1_000_000`.

### Task 1.3: Add `timed_out` to `AgentResult`

```zig
pub const AgentResult = struct {
    agent_index: usize,
    response: []u8,
    success: bool,
    duration_ns: u64,
    timed_out: bool = false,  // NEW
};
```

**Verify:** `zig build test --summary all && zig build feature-tests --summary all`

---

## Phase 2: Per-Agent Circuit Breaker + Retry

**Files:** `src/features/ai/multi_agent/mod.zig`

### Task 2.1: Add per-agent circuit breaker tracking

Import the streaming circuit breaker:
```zig
const CircuitBreaker = @import("../../ai/streaming/circuit_breaker.zig").CircuitBreaker;
```

Wait — features can't cross-import other features directly. Instead, add a simple
inline circuit breaker struct to mod.zig (following the streaming CB's pattern but
simplified for the coordinator's needs):

```zig
const AgentHealth = struct {
    consecutive_failures: u32 = 0,
    failure_threshold: u32 = 5,
    total_successes: u64 = 0,
    total_failures: u64 = 0,
    is_open: bool = false,

    fn recordSuccess(self: *AgentHealth) void {
        self.consecutive_failures = 0;
        self.total_successes += 1;
        self.is_open = false;
    }

    fn recordFailure(self: *AgentHealth) void {
        self.consecutive_failures += 1;
        self.total_failures += 1;
        if (self.consecutive_failures >= self.failure_threshold) {
            self.is_open = true;
        }
    }

    fn canAttempt(self: *const AgentHealth) bool {
        return !self.is_open;
    }

    fn successRate(self: *const AgentHealth) f64 {
        const total = self.total_successes + self.total_failures;
        if (total == 0) return 1.0;
        return @as(f64, @floatFromInt(self.total_successes)) / @as(f64, @floatFromInt(total));
    }
};
```

Add `health: std.ArrayListUnmanaged(AgentHealth)` to `Coordinator`. Initialize in
`register()`, update in execution methods after each agent completes.

### Task 2.2: Add retry wrapper using existing RetryConfig

Import retry utilities:
```zig
const retry = @import("../../../services/shared/utils/retry.zig");
```

Add `retry_config: retry.RetryConfig = .{}` to `CoordinatorConfig`.

In `executeSequential` and the thread function, wrap `ag.process()` in a retry loop:
```zig
fn processWithRetry(ag: *agents.Agent, task: []const u8, allocator: std.mem.Allocator, cfg: retry.RetryConfig) ![]u8 {
    var attempt: u32 = 0;
    while (true) {
        const result = ag.process(task, allocator) catch |err| {
            attempt += 1;
            if (attempt >= cfg.max_retries) return err;
            const delay = retry.calculateDelay(cfg, attempt);
            std.time.sleep(delay * std.time.ns_per_ms);
            continue;
        };
        return result;
    }
}
```

### Task 2.3: Skip unhealthy agents in execution

In `executeSequential`/`executeParallel`, check `health.items[i].canAttempt()` before
running. If circuit is open, record a skip result:
```zig
if (!self.health.items[i].canAttempt()) {
    self.results.append(self.allocator, .{
        .agent_index = i,
        .response = self.allocator.dupe(u8, "[Skipped: circuit open]") catch return Error.ExecutionFailed,
        .success = false,
        .duration_ns = 0,
    }) catch return Error.ExecutionFailed;
    continue;
}
```

**Verify:** `zig build test --summary all && zig build feature-tests --summary all`

---

## Phase 3: ThreadPool + Agent Channel Messaging

**Files:** `src/features/ai/multi_agent/mod.zig`, `src/features/ai/multi_agent/messaging.zig`

### Task 3.1: Replace raw `std.Thread` with ThreadPool in `executeParallel`

**Decision: Keep raw `std.Thread` for now.** The ThreadPool uses a fixed task frame
(128 bytes) which can't easily capture the agent pointer + task slice + result pointer
+ allocator needed for `runAgentThread`. The current std.Thread approach is correct
once the join bug is fixed. Switching to ThreadPool would require refactoring the
task capture model — not worth the complexity for this module's use case.

### Task 3.2: Add inter-agent message channel to messaging.zig

Add an `AgentMailbox` type using the existing channel infrastructure concept (but
implemented inline to avoid cross-feature imports):

```zig
pub const AgentMessage = struct {
    from_agent: usize,
    to_agent: usize,
    content: []const u8,
    timestamp: u64,
};

pub const AgentMailbox = struct {
    allocator: std.mem.Allocator,
    inbox: std.ArrayListUnmanaged(AgentMessage) = .{},
    mutex: sync.Mutex = .{},

    pub fn init(allocator: std.mem.Allocator) AgentMailbox { ... }
    pub fn deinit(self: *AgentMailbox) void { ... }
    pub fn send(self: *AgentMailbox, msg: AgentMessage) !void { ... }
    pub fn receive(self: *AgentMailbox) ?AgentMessage { ... }
    pub fn pendingCount(self: *const AgentMailbox) usize { ... }
};
```

Add `mailboxes: std.ArrayListUnmanaged(AgentMailbox)` to Coordinator,
initialize per-agent in `register()`.

### Task 3.3: Wire mailboxes into pipeline execution

In `executePipeline`, instead of passing raw text between agents, send via mailbox:
```zig
// After agent i produces output, send to agent i+1's mailbox
if (i + 1 < self.agents.items.len) {
    try self.mailboxes.items[i + 1].send(.{
        .from_agent = i,
        .to_agent = i + 1,
        .content = response,
        .timestamp = ... ,
    });
}
```

**Verify:** `zig build test --summary all && zig build feature-tests --summary all`

---

## Phase 4: CLI Wiring + Integration Tests

**Files:** `tools/cli/commands/multi_agent.zig`, `src/services/tests/multi_agent_test.zig`

### Task 4.1: Wire CLI `run` command to create real echo agents

In `runWorkflow()`, instead of simulating, create actual agents with `.echo` backend:
```zig
// Create agents based on workflow template
const template = findTemplate(workflow_name);
for (template.agents) |agent_name| {
    var agent = try Agent.init(allocator, .{
        .name = agent_name,
        .backend = .echo,  // Safe default, no API keys needed
    });
    try coord.register(&agent);
}
const result = try coord.runTask(task_description.?);
```

This makes `abi multi-agent run --task "test"` actually exercise the coordinator.

### Task 4.2: Add integration test file

Create `src/services/tests/multi_agent_test.zig`:
- Test coordinator with 3 echo agents, sequential execution
- Test coordinator with 3 echo agents, parallel execution
- Test pipeline execution (output chains)
- Test aggregation strategies with known inputs
- Test circuit breaker behavior (force failures, verify skip)
- Test timeout behavior

Register in the test discovery chain.

### Task 4.3: Add stress test

Add to the integration test file:
- 10 agents, parallel execution, 100 tasks
- Verify no memory leaks (all allocations freed)
- Verify all threads joined (no hangs)

**Verify:** `zig build test --summary all && zig build feature-tests --summary all`

---

## Phase 5: Stub Parity + Final Verification

**Files:** `src/features/ai/multi_agent/stub.zig`

### Task 5.1: Update stub.zig to match new public API

Add any new public types/fields to stub.zig:
- `AgentHealth` struct (with no-op methods)
- `AgentMessage` and `AgentMailbox` types
- `timed_out` field on `AgentResult`
- `retry_config` field on `CoordinatorConfig`
- Any new methods on `Coordinator`

### Task 5.2: Run full verification gate

```bash
zig fmt .
zig build test --summary all           # Must maintain 1252+ pass
zig build feature-tests --summary all  # Must maintain 1512+ pass
zig build validate-flags               # All flag combos compile
zig build -Denable-ai=false            # Stub compiles cleanly
zig build cli-tests                    # CLI smoke tests pass
```

---

## Verification

After all phases:
1. `zig build full-check` — format + tests + feature tests + flag validation + CLI smoke
2. `zig build -Denable-ai=false` — confirm stub path compiles
3. `zig build run -- multi-agent run --task "hello"` — confirm CLI exercises real coordinator
4. Manual review: thread join logic, timeout enforcement, circuit breaker transitions
