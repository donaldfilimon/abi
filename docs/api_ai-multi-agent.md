---
title: "api_ai-multi-agent"
tags: []
---
# Multi‑Agent Coordination API

The **Multi‑Agent Coordinator** lives in `src/ai/multi_agent/mod.zig` and provides a
simple orchestrator that can run a collection of `agents.Agent` instances on a
shared textual task. It is gated by the **AI feature flag** – when `enable_ai` is
disabled the stub in `src/ai/multi_agent/stub.zig` is used and all operations
return `error.AgentDisabled`.

## Public Types

* `Coordinator` – holds an `ArrayListUnmanaged(*agents.Agent)` and exposes:
  * `init(allocator)` – create a new coordinator.
  * `deinit()` – free internal resources.
  * `register(agent_ptr)` – add an existing `Agent` instance.
  * `runTask(task)` – invoke `handle(task)` on each registered agent and
    concatenate the results.
* `Error` – `AgentDisabled` (when AI is off) and `NoAgents` (no agents registered).

## Usage Example (AI enabled)

```zig
const std = @import("std");
const ai = @import("../src/ai/mod.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    // Create coordinator
    var coord = try ai.MultiAgentCoordinator.init(alloc);
    defer coord.deinit();

    // Register agents (example stub agent)
    var myAgent = try ai.createAgent(alloc, "example");
    defer myAgent.deinit();
    try coord.register(&myAgent);

    const result = try coord.runTask("Summarize this text");
    std.debug.print("Result:\n{s}\n", .{result});
}
```

When the AI feature is disabled the same import will resolve to the stub which
mirrors the API surface but always returns `error.AgentDisabled`.

## CLI Integration

The command `abi multi-agent info` prints whether the AI feature is enabled and
how many agents are currently registered (zero in the default example). See the
source at `tools/cli/commands/multi_agent.zig` for the implementation details.


