# WDBX-AI Database Quickstart

## Installation

Add to your `build.zig`:
```zig
const database_mod = b.addModule("database", .{
    .source_file = b.path("src/database.zig"),
});
```

## Basic Usage

### 1. Create Database
```zig
const database = @import("database");

var db = try database.Db.open("vectors.wdbx", true);
defer db.close();
try db.init(384); // 384-dimensional vectors
```

### 2. Add Vectors
```zig
var embedding = try allocator.alloc(f32, 384);
defer allocator.free(embedding);
// Fill with your data...
const row_id = try db.addEmbedding(embedding);

// Batch insertion
var batch = try allocator.alloc([]f32, 100);
// Fill batch...
const indices = try db.addEmbeddingsBatch(batch);
defer allocator.free(indices);
```

### 3. Search
```zig
var query = try allocator.alloc(f32, 384);
// Fill query vector...
const results = try db.search(query, 10, allocator);
defer allocator.free(results);

for (results) |result| {
    std.debug.print("Index: {}, Score: {d}\n", .{result.index, result.score});
}
```

## Performance Tips

- Use batch operations for multiple insertions
- Choose appropriate vector dimensions
- Monitor memory usage with `db.getStats()`
- Use SIMD optimizations (automatic for 16+ dimensions)

## Error Handling

```zig
const result = db.addEmbedding(&embedding) catch |err| {
    switch (err) {
        error.DimensionMismatch => return error.InvalidInput,
        error.InvalidState => return error.InvalidState,
        else => return err,
    }
};
```

## Example: Document Search

```zig
const DocumentStore = struct {
    db: *database.Db,
    
    pub fn addDocument(self: *DocumentStore, id: []const u8, embedding: []const f32) !void {
        _ = try self.db.addEmbedding(embedding);
    }
    
    pub fn search(self: *DocumentStore, query: []const f32, top_k: usize) ![]database.Db.Result {
        return try self.db.search(query, top_k, self.allocator);
    }
};
```

## Testing

```zig
test "basic operations" {
    var db = try database.Db.open("test.wdbx", true);
    defer db.close();
    
    try db.init(64);
    try testing.expectEqual(@as(u16, 64), db.getDimension());
}
```

## Next Steps

- Read the full API reference
- Explore advanced features
- Build real applications
- Contribute improvements
