# Inference Connector Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate inference engine bridge with LLM connectors and remove automatic echo mode fallbacks.

**Architecture:** Refactor `src/inference/engine/backends.zig` to remove pseudo-echo fallbacks in the connector path. Ensure environment variables are audited using the standard `ABI_` prefix.

**Tech Stack:** Zig 0.17, ABI foundation (async HTTP client), LLM connectors.

---

### Task 1: Remove Echo Fallbacks from generateConnector

**Files:**
- Modify: `src/inference/engine/backends.zig`

- [ ] **Step 1: Remove echo warning and catch block in generateConnector**

```zig
// BEFORE
pub fn generateConnector(self: anytype, request: scheduler_mod.Request) !types.Result {
    std.log.warn("inference: connector backend using echo mode for model '{s}' — connector bridge not yet wired to external providers", .{self.config.model_id});
    // ...
    const response_text = dispatchToConnector(self.allocator, self.config.model_id, request.prompt) catch |err| blk: {
        std.log.debug("Connector dispatch failed ({s}), using echo fallback", .{@errorName(err)});
        break :blk try std.fmt.allocPrint(
            self.allocator,
            "[echo/{s}] Processing: {s}",
            .{ self.config.model_id, request.prompt[0..@min(request.prompt.len, 200)] },
        );
    };
    // ...
}

// AFTER
pub fn generateConnector(self: anytype, request: scheduler_mod.Request) !types.Result {
    const start = time_mod.timestampNs();

    const cache_ok = try self.kv_cache.allocate(request.id, request.max_tokens);
    if (!cache_ok) {
        return types.makeErrorResult(request.id, "Error: insufficient KV cache capacity");
    }
    defer self.kv_cache.free(request.id);

    // No longer catches error; real failure is propagated
    const response_text = try dispatchToConnector(self.allocator, self.config.model_id, request.prompt);
    errdefer self.allocator.free(response_text);
    // ...
}
```

- [ ] **Step 2: Commit**

```bash
git add src/inference/engine/backends.zig
git commit -m "refactor(inference): remove automatic echo fallback in generateConnector"
```

### Task 2: Remove Echo Fallbacks from Connector Dispatchers

**Files:**
- Modify: `src/inference/engine/backends.zig`

- [ ] **Step 1: Remove echoFallback usage in callOpenAICompatible**

```zig
// BEFORE
    var http = async_http.AsyncHttpClient.init(allocator) catch {
        std.log.debug("async_http init failed, using echo fallback", .{});
        return echoFallback(allocator, model_name, prompt);
    };

// AFTER
    var http = try async_http.AsyncHttpClient.init(allocator);
```

- [ ] **Step 2: Remove echoFallback usage in callAnthropicNative**

```zig
// BEFORE
    var client = anthropic.Client.init(allocator, config) catch {
        std.log.debug("Anthropic client init failed, using echo fallback", .{});
        const model_name = model_override orelse "claude-3-5-sonnet-20241022";
        return echoFallback(allocator, model_name, prompt);
    };

// AFTER
    var client = try anthropic.Client.init(allocator, config);
```

- [ ] **Step 3: Commit**

```bash
git add src/inference/engine/backends.zig
git commit -m "refactor(inference): remove echoFallback from specific connector call paths"
```

### Task 3: Audit Environment Configuration

**Files:**
- Verify: `src/connectors/env.zig`
- Verify: `src/connectors/openai.zig`
- Verify: `src/connectors/anthropic.zig`

- [ ] **Step 1: Ensure all connectors use ABI-prefixed env vars primarily**
(Already confirmed in research, but double check if any were missed).

- [ ] **Step 2: Update tests if any connector was missing prefix support**

### Task 4: Final Verification

- [ ] **Step 1: Run inference engine tests**

Run: `./build.sh test --summary all -- --test-filter "inference.engine.backends"`
Expected: Tests should pass (or fail with expected `MissingApiKey` instead of silently succeeding with echo).

- [ ] **Step 2: Clean up unused echoFallback**
If `echoFallback` is no longer used anywhere, remove it.

- [ ] **Step 3: Commit final changes**

```bash
git add .
git commit -m "chore(inference): complete connector bridge integration and echo mode removal"
```
