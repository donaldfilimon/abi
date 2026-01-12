# Troubleshooting

Common issues and solutions when working with the ABI Framework.

---

## Build Issues

### Feature Disabled Errors

**Symptom**: Getting errors like `error.AiDisabled`, `error.GpuDisabled`, or `error.NetworkDisabled` at runtime.

**Cause**: The feature was disabled at compile time.

**Solution**: Rebuild with the feature enabled:

```bash
# Enable specific features
zig build -Denable-ai=true
zig build -Denable-gpu=true
zig build -Denable-network=true
zig build -Denable-database=true

# Enable all features
zig build -Denable-ai=true -Denable-gpu=true -Denable-database=true -Denable-network=true -Denable-profiling=true
```

---

### GPU Backend Conflicts

**Symptom**: Warning about CUDA and Vulkan conflicts, or GPU operations failing.

**Cause**: Multiple GPU backends enabled that may conflict.

**Solution**: Enable only one GPU backend:

```bash
# Use only CUDA
zig build -Dgpu-cuda=true -Dgpu-vulkan=false -Dgpu-metal=false

# Use only Vulkan (default)
zig build -Dgpu-vulkan=true -Dgpu-cuda=false

# Use CPU fallback (stdgpu)
zig build -Dgpu-stdgpu=true -Dgpu-cuda=false -Dgpu-vulkan=false
```

---

### WASM Build Missing Features

**Symptom**: Features unavailable when building for WASM.

**Cause**: Certain features are automatically disabled for WASM targets due to platform limitations.

**Expected behavior**: These features are disabled in WASM builds:
- `database` - No `std.Io.Threaded` support
- `network` - No socket support
- `gpu` - Native GPU backends unavailable

**Solution**: This is by design. Use the JavaScript/browser equivalents for these features in WASM builds.

---

### Slow Builds

**Symptom**: Build takes longer than expected.

**Solutions**:

1. Clear the build cache:
   ```bash
   rm -rf .zig-cache
   zig build
   ```

2. Reduce parallel jobs:
   ```bash
   zig build -j 2
   ```

3. Disable unused features:
   ```bash
   zig build -Denable-network=false -Denable-profiling=false
   ```

4. Check disk space - Zig needs space for incremental compilation.

---

### Test Filter Not Working

**Symptom**: `zig build test --test-filter "pattern"` doesn't filter tests.

**Cause**: `--test-filter` must be passed directly to `zig test`, not through `zig build test`.

**Solution**:

```bash
# Correct - filter tests directly
zig test src/tests/mod.zig --test-filter "pattern"

# Does NOT work with zig build
zig build test --test-filter "pattern"  # Won't filter
```

---

## Runtime Issues

### Out of Memory

**Symptom**: `error.OutOfMemory` during operation.

**Solutions**:

1. Use arena allocators for temporary allocations:
   ```zig
   var arena = std.heap.ArenaAllocator.init(allocator);
   defer arena.deinit();
   const temp_allocator = arena.allocator();
   ```

2. Free memory promptly with `defer`:
   ```zig
   const data = try allocator.alloc(u8, size);
   defer allocator.free(data);
   ```

3. Reduce batch sizes for database operations:
   ```zig
   const config = batch.BatchConfig{
       .batch_size = 500,  // Reduce from default 1000
   };
   ```

4. Limit GPU memory:
   ```zig
   var gpu = try abi.Gpu.init(allocator, .{
       .max_memory_bytes = 512 * 1024 * 1024,  // 512MB limit
   });
   ```

---

### Timeout Errors

**Symptom**: `EngineError.Timeout` when waiting for task results.

**Cause**: Task taking longer than specified timeout.

**Solutions**:

1. Increase timeout:
   ```zig
   // Wait longer (5 seconds instead of 1)
   const result = try abi.compute.runTask(&engine, u32, myTask, 5000);

   // Wait indefinitely
   const result = try abi.compute.runTask(&engine, u32, myTask, null);
   ```

2. Check for blocking operations in task code.

3. Verify worker threads are running:
   ```zig
   const summary = engine.getMetricsSummary();
   std.debug.print("Active workers: {d}\n", .{summary.active_workers});
   ```

---

### Connection Refused (Network)

**Symptom**: `error.NodeUnreachable` or connection failures.

**Solutions**:

1. Check if the target service is running:
   ```bash
   # For Ollama
   curl http://127.0.0.1:11434/api/version

   # For local scheduler
   curl http://127.0.0.1:8081/health
   ```

2. Verify environment variables:
   ```bash
   echo $ABI_OLLAMA_HOST
   echo $ABI_LOCAL_SCHEDULER_URL
   ```

3. Check firewall rules for the port.

4. Use circuit breaker for graceful degradation:
   ```zig
   var breaker = try abi.network.CircuitBreaker.init(allocator, .{
       .failure_threshold = 5,
       .reset_timeout_ms = 30000,
   });
   ```

---

### GPU Not Detected

**Symptom**: GPU operations fall back to CPU simulation.

**Solutions**:

1. Check available backends:
   ```bash
   zig build run -- gpu backends
   zig build run -- gpu devices
   ```

2. Verify GPU drivers are installed.

3. For CUDA, ensure NVIDIA drivers and toolkit are installed.

4. For Vulkan, install Vulkan SDK and runtime.

5. Try explicit backend selection:
   ```zig
   var gpu = try abi.Gpu.init(allocator, .{
       .preferred_backend = .vulkan,
       .allow_fallback = false,  // Fail instead of falling back
   });
   ```

---

### Database Path Validation Error

**Symptom**: `PathValidationError` on backup/restore.

**Cause**: Path contains unsafe characters or traversal sequences.

**Solution**: Use safe filenames without path separators:

```zig
// Good - simple filename
try db.backup("snapshot_2025.db");
try db.restore("snapshot_2025.db");

// Bad - path traversal (will fail)
try db.restore("../../../secret.txt");  // PathValidationError
try db.restore("/etc/passwd");           // PathValidationError
try db.restore("C:\\Windows\\file.db");  // PathValidationError
```

All backup/restore operations are restricted to the `backups/` directory.

---

## API Issues

### Zig 0.16 I/O Migration

**Symptom**: Code using `std.fs.cwd()` or `std.io.AnyReader` doesn't compile.

**Cause**: Zig 0.16 uses the new `std.Io` API.

**Solution**: Update to new patterns:

```zig
// OLD (Zig 0.15)
const file = try std.fs.cwd().openFile(path, .{});

// NEW (Zig 0.16)
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty,
});
defer io_backend.deinit();
const io = io_backend.io();

const content = try std.Io.Dir.cwd().readFileAlloc(
    io, path, allocator, .limited(10 * 1024 * 1024)
);
defer allocator.free(content);
```

See [Zig 0.16 Migration Guide](migration/zig-0.16-migration.md) for full details.

---

### HTTP Server Interface Access

**Symptom**: Compilation error when initializing `std.http.Server`.

**Cause**: Need to use `.interface` field for reader/writer.

**Solution**:

```zig
// Correct pattern for Zig 0.16
var connection_reader = stream.reader(io, &recv_buffer);
var connection_writer = stream.writer(io, &send_buffer);
var server: std.http.Server = .init(
    &connection_reader.interface,  // Use .interface
    &connection_writer.interface,
);
```

---

### Format Specifier Errors

**Symptom**: Compilation warnings about `@tagName()` or `@errorName()`.

**Cause**: Zig 0.16 prefers `{t}` format specifier.

**Solution**:

```zig
// OLD
std.debug.print("Error: {s}", .{@errorName(err)});
std.debug.print("State: {s}", .{@tagName(state)});

// NEW (Zig 0.16)
std.debug.print("Error: {t}", .{err});
std.debug.print("State: {t}", .{state});
```

---

## Performance Issues

### High CPU Usage

**Symptom**: CPU usage higher than expected.

**Solutions**:

1. Reduce worker thread count:
   ```zig
   var engine = try abi.compute.createEngine(allocator, .{
       .worker_count = 2,  // Reduce from default
   });
   ```

2. Add backoff in busy loops:
   ```zig
   const backoff = abi.compute.Backoff.init();
   while (condition) {
       backoff.wait();
   }
   ```

3. Use profiling to find bottlenecks:
   ```zig
   var timer = try std.time.Timer.start();
   // operation
   const elapsed = timer.read();
   ```

---

### Memory Leaks

**Symptom**: Memory usage grows over time.

**Solutions**:

1. Ensure every allocation has corresponding deallocation:
   ```zig
   const data = try allocator.alloc(u8, size);
   defer allocator.free(data);  // Always free
   ```

2. Use `errdefer` for cleanup on error:
   ```zig
   const resource = try allocator.create(Resource);
   errdefer allocator.destroy(resource);
   try resource.init();  // If this fails, resource is freed
   ```

3. Check for circular references in data structures.

4. Use `std.heap.GeneralPurposeAllocator` in debug builds:
   ```zig
   var gpa = std.heap.GeneralPurposeAllocator(.{
       .stack_trace_frames = 10,
   }){};
   defer {
       const check = gpa.deinit();
       if (check == .leak) @panic("Memory leak detected");
   }
   ```

---

### Slow Database Queries

**Symptom**: Vector search taking too long.

**Solutions**:

1. Optimize HNSW parameters:
   ```zig
   const config = wdbx.HnswConfig{
       .m = 16,              // Connections per node
       .ef_construction = 200,  // Index build quality
       .ef_search = 50,      // Search quality (lower = faster)
   };
   ```

2. Use batch operations:
   ```zig
   const result = try db.batchInsert(records, .{
       .batch_size = 1000,
       .parallel_workers = 4,
   });
   ```

3. Pre-filter with metadata:
   ```zig
   const results = try db.searchVectors(query, 10, .{
       .filter = filter.Filter.init().field("category").eq(.{ .string = "tech" }),
       .filter_strategy = .pre_filter,
   });
   ```

---

## Getting Help

If your issue isn't covered here:

1. Check the [GitHub Issues](https://github.com/donaldfilimon/abi/issues)
2. Review the relevant feature documentation
3. Run with debug logging enabled
4. Check the [Performance Baseline](PERFORMANCE_BASELINE.md) for expected metrics

---

## See Also

- [Framework](framework.md) - Configuration options
- [Monitoring](monitoring.md) - Debugging with metrics
- [Zig 0.16 Migration](migration/zig-0.16-migration.md) - API changes
- [Performance Baseline](PERFORMANCE_BASELINE.md) - Expected performance
