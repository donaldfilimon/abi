---
title: "Troubleshooting"
tags: [troubleshooting, help, debugging]
---
# Troubleshooting
> **Codebase Status:** Synced with repository as of 2026-01-30.

<p align="center">
  <img src="https://img.shields.io/badge/Guide-Troubleshooting-red?style=for-the-badge" alt="Troubleshooting"/>
  <img src="https://img.shields.io/badge/Topics-Build%7CRuntime%7CGPU-blue?style=for-the-badge" alt="Topics"/>
</p>

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
   const result = try abi.runtime.runTask(&engine, u32, myTask, 5000);

   // Wait indefinitely
   const result = try abi.runtime.runTask(&engine, u32, myTask, null);
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

6. **Use diagnostics** (2026.01): `const diag = gpu_mod.DiagnosticsInfo.collect(allocator); if (diag.is_degraded) { ... }`

---

### GPU Operations Failing / Degradation (2026.01)

**Silent failures**: Use `error_handling.ErrorContext.init(.backend_error, backend, "msg")` to capture context.

**No CPU fallback**: Use `failover.FailoverManager.init(allocator)` with `.setDegradationMode(.automatic)`.

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

### Database Health Issues (2026.01)

**Slow/inconsistent results**: Use `const diag = db.diagnostics(); if (!diag.isHealthy()) { try db.rebuildNormCache(); }`

**Search performance**: Enable `.cache_norms = true`, use `db.searchBatch()` for multiple queries.

---

### AI Agent API Errors (2026.01)

**Debugging**: Use `agent.ErrorContext.apiError(err, .openai, endpoint, status, model)` then `ctx.log()`.

**Common errors**: `ApiKeyMissing` (set env var), `RateLimitExceeded` (increase backoff), `Timeout` (check network), `ConnectionRefused` (start Ollama), `ModelNotFound` (download model).

**Rate limiting**: Use `ErrorContext.retryError()` with exponential backoff.

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

See [CLAUDE.md](../CLAUDE.md) for Zig 0.16 patterns and examples.

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
   var engine = try abi.runtime.createEngine(allocator, .{
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

## Debugging with GDB/LLDB

### Source-Level Debugging

ABI Framework supports debugging with standard debuggers (GDB, LLDB). To enable source-level debugging:

**Build with Debug Info**:
```bash
# Debug build (default)
zig build -Doptimize=Debug

# Release with debug info
zig build -Doptimize=ReleaseSafe
```

**Using GDB**:
```bash
# Debug the CLI
gdb ./zig-out/bin/abi

# Set breakpoints
(gdb) break src/compute/runtime/engine.zig:150
(gdb) run -- db stats

# Step through code
(gdb) next
(gdb) step
(gdb) continue

# Print variables
(gdb) print worker_count
(gdb) info locals
```

**Using LLDB** (recommended on macOS):
```bash
# Debug the CLI
lldb ./zig-out/bin/abi

# Set breakpoints
(lldb) breakpoint set --file engine.zig --line 150
(lldb) run -- db stats

# Step through code
(lldb) next
(lldb) step
(lldb) continue

# Print variables
(lldb) frame variable
(lldb) print worker_count
```

**Common Debug Commands**:

| GDB | LLDB | Description |
|-----|------|-------------|
| `break file:line` | `b file:line` | Set breakpoint |
| `run` | `run` | Start program |
| `continue` | `continue` | Continue execution |
| `next` | `next` | Step over |
| `step` | `step` | Step into |
| `print var` | `print var` | Print variable |
| `backtrace` | `bt` | Show call stack |
| `info threads` | `thread list` | List threads |

### Debugging Memory Issues

Use the built-in `TrackingAllocator` for leak detection:

```zig
const tracking = @import("src/shared/utils/memory/tracking.zig");

var tracker = tracking.TrackingAllocator.init(std.heap.page_allocator, .{});
defer {
    if (tracker.detectLeaks()) {
        tracker.dumpLeaks(std.io.getStdErr().writer()) catch {};
    }
    tracker.deinit();
}
const allocator = tracker.allocator();
// Use allocator for operations...
```

### Debugging GPU Operations

Enable GPU profiling for timing and memory transfer analysis:

```zig
const gpu_profiling = @import("src/gpu/profiling.zig");

var profiler = gpu_profiling.Profiler.init(allocator);
defer profiler.deinit(allocator);

profiler.enable();
// Run GPU operations...
gpu_profiling.formatSummary(&profiler, std.io.getStdErr().writer()) catch {};
```

### Debugging Task Execution

Use the compute profiler to track task metrics:

```zig
const profiling = @import("src/compute/profiling/mod.zig");

var collector = try profiling.initCollector(allocator, .{}, worker_count);
defer collector.deinit();

// After running tasks...
const summary = collector.getSummary();
std.debug.print("Total tasks: {d}, Avg time: {d}ns\n", .{
    summary.total_tasks,
    summary.avg_execution_ns,
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
- [Developer Guide](../CLAUDE.md) - Zig 0.16 patterns and API conventions
- [Performance Baseline](PERFORMANCE_BASELINE.md) - Expected performance
