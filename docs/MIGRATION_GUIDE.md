# WDBX Migration Guide

## Migrating from v1.x to v2.0 (Refactored)

This guide helps you migrate from the original WDBX implementation to the refactored modular architecture.

## Overview of Changes

### Structural Changes

1. **Modular Architecture**: Code is now organized into clear modules (core, api, utils)
2. **Unified Entry Point**: Single `main.zig` instead of multiple entry files
3. **Consistent Interfaces**: Standardized APIs across all components
4. **Improved Error Handling**: Comprehensive error types with context

### API Changes

#### CLI Changes

**Old:**
```bash
./wdbx add "1.0,2.0,3.0,4.0"
./wdbx query "1.1,2.1,3.1,4.1"
```

**New:**
```bash
./wdbx cli add --db vectors.wdbx --vector "1.0,2.0,3.0,4.0"
./wdbx cli search --db vectors.wdbx --query "1.1,2.1,3.1,4.1" --k 5
```

#### Programmatic API Changes

**Old:**
```zig
const wdbx = @import("wdbx.zig");
var db = try wdbx.Db.open("vectors.wdbx", true);
try db.init(384);
```

**New:**
```zig
const core = @import("core");
var db = try core.Database.open(allocator, "vectors.wdbx", true);
try db.init(.{
    .dimensions = 384,
    .index_type = .hnsw,
    .distance_metric = .euclidean,
});
```

### Data Format Changes

The file format has been updated for better performance and features:
- Version 2 format with enhanced header
- Improved index serialization
- Metadata support

**Migration Tool:**
```bash
./wdbx migrate --input old.wdbx --output new.wdbx
```

## Step-by-Step Migration

### Step 1: Update Imports

Replace old imports with new module structure:

```zig
// Old
const database = @import("database.zig");
const wdbx = @import("wdbx.zig");

// New
const core = @import("core");
const api = @import("api");
```

### Step 2: Update Database Initialization

**Old:**
```zig
var db = try database.Db.open("vectors.wdbx", true);
defer db.close();
try db.init(128);
```

**New:**
```zig
const db = try core.Database.open(allocator, "vectors.wdbx", true);
defer db.close();
try db.init(.{
    .dimensions = 128,
    .index_type = .hnsw,
    .distance_metric = .euclidean,
    .enable_simd = true,
});
```

### Step 3: Update Vector Operations

**Old:**
```zig
const id = try db.addEmbedding(&vector);
const results = try db.search(&query, 10, allocator);
```

**New:**
```zig
const id = try db.addVector(&vector, null); // Optional metadata
const results = try db.search(&query, 10, allocator);
defer allocator.free(results); // Always free results
```

### Step 4: Update Error Handling

**Old:**
```zig
db.addEmbedding(&vector) catch |err| {
    std.debug.print("Error: {}\n", .{err});
};
```

**New:**
```zig
db.addVector(&vector, null) catch |err| {
    switch (err) {
        error.DimensionMismatch => {
            // Handle dimension error
        },
        error.DatabaseNotInitialized => {
            // Handle initialization error
        },
        else => return err,
    }
};
```

### Step 5: Update Server Code

**Old:**
```zig
const server = @import("wdbx_http_server.zig");
try server.start(8080);
```

**New:**
```zig
var server = try api.HttpServer.init(allocator, .{
    .db_path = "vectors.wdbx",
    .host = "127.0.0.1",
    .port = 8080,
    .api_config = .{
        .enable_auth = true,
        .enable_rate_limit = true,
    },
});
defer server.deinit();
try server.run();
```

## Configuration Migration

### Old Configuration (CLI arguments)
```bash
./wdbx http 8080
```

### New Configuration (Structured)
```zig
const config = .{
    .db_config = .{
        .dimensions = 384,
        .index_type = .hnsw,
        .hnsw_m = 16,
        .hnsw_ef_construction = 200,
    },
    .api_config = .{
        .enable_auth = true,
        .rate_limit_rpm = 1000,
    },
};
```

## Feature Mapping

| Old Feature | New Feature | Notes |
|------------|------------|-------|
| `Db` struct | `core.Database` | Enhanced with configuration |
| `addEmbedding` | `addVector` | Now supports metadata |
| `search` | `search` | Same interface, better performance |
| Multiple main files | Single entry point | Use commands: `serve`, `cli`, etc. |
| Basic error types | Comprehensive `ErrorSet` | With context tracking |
| Manual SIMD | Automatic SIMD | Runtime detection |

## Breaking Changes

1. **File Format**: v2 format not backward compatible (use migration tool)
2. **Import Paths**: All imports must use new module structure
3. **Configuration**: Now uses structured config instead of positional args
4. **Error Types**: New error names and error handling patterns
5. **Memory Management**: Explicit allocator passing required

## Performance Improvements

The refactored version includes several performance improvements:

- **SIMD Optimization**: Automatic detection and use
- **Better Memory Layout**: Cache-friendly structures
- **Optimized Indexes**: Improved HNSW implementation
- **Parallel Operations**: Multi-threaded index building

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   error: unable to find 'wdbx.zig'
   ```
   Solution: Update to use module imports (`@import("core")`)

2. **Initialization Errors**
   ```
   error: expected type 'u32', found 'DatabaseConfig'
   ```
   Solution: Use configuration struct instead of individual parameters

3. **Memory Errors**
   ```
   error: OutOfMemory
   ```
   Solution: Ensure proper allocator is passed to all functions

### Getting Help

- Check the [Architecture Guide](ARCHITECTURE.md)
- Review [API Documentation](API_REFERENCE.md)
- Submit issues on GitHub

## Example Migration

Here's a complete example of migrating a simple application:

### Old Code
```zig
const std = @import("std");
const wdbx = @import("wdbx.zig");

pub fn main() !void {
    var db = try wdbx.Db.open("vectors.wdbx", true);
    defer db.close();
    
    try db.init(128);
    
    const vector = [_]f32{1.0} ** 128;
    const id = try db.addEmbedding(&vector);
    
    const results = try db.search(&vector, 10, std.heap.page_allocator);
    defer std.heap.page_allocator.free(results);
    
    std.debug.print("Found {} results\n", .{results.len});
}
```

### New Code
```zig
const std = @import("std");
const core = @import("core");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const db = try core.Database.open(allocator, "vectors.wdbx", true);
    defer db.close();
    
    try db.init(.{
        .dimensions = 128,
        .index_type = .hnsw,
        .distance_metric = .euclidean,
    });
    
    const vector = [_]f32{1.0} ** 128;
    const id = try db.addVector(&vector, null);
    
    const results = try db.search(&vector, 10, allocator);
    defer allocator.free(results);
    
    std.debug.print("Found {} results\n", .{results.len});
}
```

## Next Steps

1. Review the new [Architecture](ARCHITECTURE.md)
2. Update your build configuration
3. Run the migration tool on existing databases
4. Update your application code
5. Test thoroughly
6. Deploy with confidence!