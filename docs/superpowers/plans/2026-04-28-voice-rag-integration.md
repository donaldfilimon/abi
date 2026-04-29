# Voice Gateway and RAG Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement voice gateway connection/disconnection and memory-augmented inference (RAG) using WDBX.

**Architecture:** 
1. `AbbeyDiscordBot` in `discord.zig` will be updated to manage a Voice Gateway connection, handling connection/disconnection.
2. `AbbeyEngine.process` in `engine.zig` will be updated to retrieve relevant context from the database (WDBX) using memory-augmented retrieval and inject it into the LLM context.
3. Inference performance (timing) will be logged to the `AbbeyEngine` internal statistics.

**Tech Stack:** Zig 0.17, WDBX Database, Discord Gateway (internal).

---

### Task 1: Update AbbeyDiscordBot for Voice Connection Management

**Files:**
- Modify: `abi/src/features/ai/abbey/discord.zig`
- Test: `abi/src/features/ai/abbey/discord.zig` (existing test block)

- [ ] **Step 1: Add Voice Gateway state to `AbbeyDiscordBot`**

Add `voice_gateway: ?*VoiceGateway = null` to `AbbeyDiscordBot` struct.

- [ ] **Step 2: Add connection methods to `AbbeyDiscordBot`**

```zig
pub fn connectVoice(self: *Self, guild_id: []const u8, channel_id: []const u8) !void {
    // Logic to initiate connection
    log.info("Connecting voice gateway to guild {s} channel {s}", .{guild_id, channel_id});
}

pub fn disconnectVoice(self: *Self) void {
    log.info("Disconnecting voice gateway", .{});
}
```

- [ ] **Step 3: Update `deinit` to stop voice gateway**

- [ ] **Step 4: Run tests**
Run: `./build.sh test`

- [ ] **Step 5: Commit**

---

### Task 2: Implement Memory-Augmented Inference (RAG) in AbbeyEngine

**Files:**
- Modify: `abi/src/features/ai/abbey/engine.zig`
- Test: `abi/src/features/ai/abbey/engine.zig` (existing test block)

- [ ] **Step 1: Update `processManual` to inject RAG context**

```zig
// In step 7 of processManual:
// Use self.memory to retrieve relevant context for the current user input.
// Inject it into the prompt.
var hybrid_context = try self.memory.getHybridContext(
    context_embedding, // Already computed or retrieved
    self.config.memory.max_context_tokens,
    5,
);
defer hybrid_context.deinit(self.allocator);
// The existing engine already uses hybrid_context; ensure WDBX is queried here.
```

- [ ] **Step 2: Add performance monitoring for inference**

```zig
// Log timing in engine stats.
self.updateAverageResponseTime(response_time);
log.info("Inference completed in {d}ms with context injection", .{response_time});
```

- [ ] **Step 3: Run tests**
Run: `./build.sh test`

- [ ] **Step 4: Commit**
