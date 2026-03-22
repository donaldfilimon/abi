---
title: network API
purpose: Generated API reference for network
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2934+47d2e5de9
---

# network

> Network Module

Distributed compute network with node discovery, Raft consensus,
and distributed task coordination.

## Features
- Node registry and discovery
- Raft consensus for leader election
- Task scheduling and load balancing
- Connection pooling and retry logic
- Circuit breakers for fault tolerance
- Rate limiting

## Usage

```zig
const network = @import("network");

// Initialize the network module
try network.init(allocator);
defer network.deinit();

// Get the node registry
const registry = try network.defaultRegistry();
try registry.register("node-a", "127.0.0.1:9000");
```

**Source:** [`src/features/network/mod.zig`](../../src/features/network/mod.zig)

**Build flag:** `-Dfeat_network=true`

---

## API

### <a id="pub-const-context"></a>`pub const Context`

<sup>**const**</sup> | [source](../../src/features/network/mod.zig#L333)

Network context for Framework integration.

### <a id="pub-fn-connect-self-context-void"></a>`pub fn connect(self: *Context) !void`

<sup>**fn**</sup> | [source](../../src/features/network/mod.zig#L362)

Connect to the network.

### <a id="pub-fn-disconnect-self-context-void"></a>`pub fn disconnect(self: *Context) void`

<sup>**fn**</sup> | [source](../../src/features/network/mod.zig#L370)

Disconnect from the network.

### <a id="pub-fn-getstate-self-context-state"></a>`pub fn getState(self: *Context) State`

<sup>**fn**</sup> | [source](../../src/features/network/mod.zig#L375)

Get current state.

### <a id="pub-fn-discoverpeers-self-context-nodeinfo"></a>`pub fn discoverPeers(self: *Context) ![]NodeInfo`

<sup>**fn**</sup> | [source](../../src/features/network/mod.zig#L380)

Discover peers.

### <a id="pub-fn-sendtask-self-context-node-id-const-u8-task-anytype-void"></a>`pub fn sendTask(self: *Context, node_id: []const u8, task: anytype) !void`

<sup>**fn**</sup> | [source](../../src/features/network/mod.zig#L388)

Send a task to a remote node.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence while replacing the toolchain.
