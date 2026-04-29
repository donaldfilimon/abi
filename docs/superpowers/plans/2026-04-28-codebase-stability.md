# Codebase Stability & Core Logic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve codebase stability by fixing the ABA problem in lock-free primitives, completing the OpenAI streaming handler, and resolving baseline test failures.

**Architecture:** Implement Tagged Pointers for atomic safety, complete the OpenAI endpoint handler using existing streaming helpers, and audit the test suite for a clean "green" state.

**Tech Stack:** Zig 0.17, Lock-free atomics, SSE (Server-Sent Events).

---

### Task 1: Fix ABA in `LockFreeStack`

**Files:**
- Modify: `src/runtime/concurrency/lockfree.zig`
- Test: `src/runtime/concurrency/lockfree.zig` (inline tests)

- [ ] **Step 1: Create a stress test to expose ABA**
```zig
test "LockFreeStack ABA stress test" {
    const stack = LockFreeStack(u32).init();
    // Create high contention with node reuse...
}
```
- [ ] **Step 2: Run test to verify failure (or risk)**
Run: `zig test src/runtime/concurrency/lockfree.zig`
- [ ] **Step 3: Implement TaggedPointer and update Stack**
```zig
const TaggedPointer = struct {
    ptr: ?*Node,
    tag: usize,
};
// ... update head to std.atomic.Value(TaggedPointer)
```
- [ ] **Step 4: Verify fix with stress test**
- [ ] **Step 5: Commit**
```bash
git add src/runtime/concurrency/lockfree.zig
git commit -m "fix(concurrency): mitigate ABA problem in LockFreeStack using tagged pointers"
```

### Task 2: Complete OpenAI Streaming Handler

**Files:**
- Modify: `src/features/ai/streaming/server/openai.zig`

- [ ] **Step 1: Implement handler logic**
```zig
pub fn handleOpenAIChatCompletions(...) {
    const body = try routing.readRequestBody(server.allocator, request);
    const chat_req = try formats.openai.ChatCompletionRequest.parse(server.allocator, body);
    if (chat_req.stream) {
        return streamOpenAIResponse(server, conn_ctx, chat_req);
    } else {
        return nonStreamingOpenAIResponse(server, request, chat_req);
    }
}
```
- [ ] **Step 2: Verify with manual integration test**
- [ ] **Step 3: Commit**
```bash
git add src/features/ai/streaming/server/openai.zig
git commit -m "feat(ai): complete OpenAI-compatible streaming handler"
```

### Task 3: Resolve Test Baseline

- [ ] **Step 1: Audit skipped and "known" failing tests**
Run: `./build.sh test --summary all`
- [ ] **Step 2: Fix 2 inference connector failures**
- [ ] **Step 3: Fix 1 auth integration failure**
- [ ] **Step 4: Verify 0 failures**
- [ ] **Step 5: Commit**
```bash
git commit -m "test: resolve pre-existing failures in inference and auth"
```
