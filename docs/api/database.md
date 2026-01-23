# database API Reference

> Vector database (WDBX with HNSW/IVF-PQ)

**Source:** [`src/database/mod.zig`](../../src/database/mod.zig)

---

Database feature facade and convenience helpers.

---

## API

### `pub const Context`

<sup>**type**</sup>

Database Context for Framework integration.
Wraps the database functionality to provide a consistent interface with other modules.

### `pub fn getHandle(self: *Context) !*DatabaseHandle`

<sup>**fn**</sup>

Get or create the database handle.

### `pub fn openDatabase(self: *Context, name: []const u8) !DatabaseHandle`

<sup>**fn**</sup>

Open a database at the configured path.

### `pub fn insertVector(self: *Context, id: u64, vector: []const f32, metadata: ?[]const u8) !void`

<sup>**fn**</sup>

Insert a vector into the database.

### `pub fn searchVectors(self: *Context, query: []const f32, top_k: usize) ![]SearchResult`

<sup>**fn**</sup>

Search for similar vectors.

### `pub fn getStats(self: *Context) !Stats`

<sup>**fn**</sup>

Get database statistics.

### `pub fn optimize(self: *Context) !void`

<sup>**fn**</sup>

Optimize the database index.

---

*Generated automatically by `zig build gendocs`*
