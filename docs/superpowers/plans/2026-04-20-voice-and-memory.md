# Voice Gateway and Memory-Augmented Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement voice gateway connectivity in the Discord connector and enable WDBX memory retrieval within the Abbey Engine pipeline.

**Architecture:**
1.  **Discord Connector:** Add `VoiceGatewayBridge` and event hooks in `discord.zig`.
2.  **Abbey Engine:** Modify `engine.zig` to hook into `wdbx` memory retrieval at the beginning of the `processManual` pipeline.
3.  **Verification:** Implement integration tests in `abi/src/features/ai/abbey/test_wdbx_integration.zig` and verify functionality.

**Tech Stack:** Zig 0.17, WDBX Database, Discord Gateway (ABI-native)

---

### Task 1: Update Discord DiscordBotConfig and Gateway Types

**Files:**
- Modify: `abi/src/features/ai/abbey/discord.zig:63-95`

- [ ] **Step 1: Update `DiscordBotConfig` and `GatewayBridge` types**
Add `voice_enabled` to config and `voice_gateway` types.

```zig
pub const DiscordBotConfig = struct {
    // ... existing
    voice_enabled: bool = false,
};

pub const VoiceEvent = enum {
    CONNECT,
    DISCONNECT,
    SPEAKING,
};
```

---

### Task 2: Implement Voice Gateway Handlers in Discord Bridge

**Files:**
- Modify: `abi/src/features/ai/abbey/discord.zig`

- [ ] **Step 1: Add handlers to `GatewayBridge`**

Add internal state for voice and handler hooks.

```zig
// In GatewayBridge
voice_ready: bool = false,

fn onVoiceServerUpdate(ctx: ?*anyopaque, payload: []const u8) void {
    // ... implementation
}
```

---

### Task 3: Integrate WDBX Memory Retrieval in Abbey Engine

**Files:**
- Modify: `abi/src/features/ai/abbey/engine.zig:200-240`

- [ ] **Step 1: Update `processManual` to inject Memory**

Retrieve context from `wdbx` and prepend it to the context window.

```zig
// In processManual
// 7. Build context
// Prepend memory
const memory_context = try self.memory.retrieveRecent(self.allocator, 1024);
defer self.allocator.free(memory_context);
// ... inject into context ...
```

---

### Task 4: Integration Testing

**Files:**
- Modify: `abi/src/features/ai/abbey/test_wdbx_integration.zig`

- [ ] **Step 1: Implement Test Case**

```zig
test "memory-augmented inference pipeline" {
    // 1. Init engine
    // 2. Store knowledge
    // 3. Process query
    // 4. Assert memory is prepended
}
```

---

### Execution Instructions

1. Use `superpowers:subagent-driven-development` to execute tasks 1-4.
2. Review outputs after each task.
3. Run `zig build check-parity` and `zig build check` after all tasks complete.
