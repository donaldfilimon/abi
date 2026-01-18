# database API Reference

**Source:** `src/database/mod.zig`

 Database feature facade and convenience helpers.
### `pub const Context`

 Database Context for Framework integration.
 Wraps the database functionality to provide a consistent interface with other modules.

### `pub fn getHandle(self: *Context) !*DatabaseHandle`

 Get or create the database handle.

### `pub fn openDatabase(self: *Context, name: []const u8) !DatabaseHandle`

 Open a database at the configured path.

### `pub fn insertVector(self: *Context, id: u64, vector: []const f32, metadata: ?[]const u8) !void`

 Insert a vector into the database.

### `pub fn searchVectors(self: *Context, query: []const f32, top_k: usize) ![]SearchResult`

 Search for similar vectors.

### `pub fn getStats(self: *Context) !Stats`

 Get database statistics.

### `pub fn optimize(self: *Context) !void`

 Optimize the database index.

