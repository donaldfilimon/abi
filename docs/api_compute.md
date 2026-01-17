# compute API Reference

**Source:** `src/compute/runtime/mod.zig`

### `pub fn runWorkload(engine_instance: *DistributedComputeEngine, comptime ResultType: type, work: anytype, timeout_ms: u64) !ResultType`

 Alias for runTask() - runs a workload and waits for the result
 @param engine_instance The compute engine instance
 @param ResultType The expected result type
 @param work The workload/task to execute
 @param timeout_ms Timeout in milliseconds (0=immediate check, null=wait indefinitely)
 @return The workload result

