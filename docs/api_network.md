# network API Reference

**Source:** `src/network/mod.zig`

 Network Module
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
 const network = @import("network/mod.zig");
 // Initialize the network module
 try network.init(allocator);
 defer network.deinit();
 // Get the node registry
 const registry = try network.defaultRegistry();
 try registry.register("node-a", "127.0.0.1:9000");
 ```
### `pub const Context`

 Network context for Framework integration.

### `pub fn connect(self: *Context) !void`

 Connect to the network.

### `pub fn disconnect(self: *Context) void`

 Disconnect from the network.

### `pub fn getState(self: *Context) State`

 Get current state.

### `pub fn discoverPeers(self: *Context) ![]NodeInfo`

 Discover peers.

### `pub fn sendTask(self: *Context, node_id: []const u8, task: anytype) !void`

 Send a task to a remote node.

