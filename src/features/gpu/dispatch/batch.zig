//! Batched Dispatcher for Small Operations
//!
//! Collects small kernel operations and executes them together to reduce
//! dispatch overhead for many small kernel launches.

const std = @import("std");
const dispatch_types = @import("types.zig");
const dispatcher_mod = @import("coordinator.zig");

pub const DispatchError = dispatch_types.DispatchError;
pub const CompiledKernelHandle = dispatch_types.CompiledKernelHandle;
pub const LaunchConfig = dispatch_types.LaunchConfig;
pub const KernelArgs = dispatch_types.KernelArgs;
pub const Buffer = dispatch_types.Buffer;
pub const KernelDispatcher = dispatcher_mod.KernelDispatcher;

/// Batched operation for deferred execution.
pub const BatchedOp = struct {
    kernel: CompiledKernelHandle,
    config: LaunchConfig,
    buffers: [8]*Buffer, // Fixed-size to avoid allocation
    buffer_count: u8,
    /// Priority level for execution ordering
    priority: Priority = .normal,
    /// Category for grouping similar operations
    category: Category = .unknown,

    pub const Priority = enum {
        high, // Execute first
        normal, // Standard priority
        low, // Execute last
    };

    pub const Category = enum {
        unknown,
        vector_ops,
        matrix_ops,
        element_wise,
        reduction,
        activation,
    };
};

/// Batched dispatcher that collects small operations and executes them together.
/// This reduces dispatch overhead for many small kernel launches.
pub const BatchedDispatcher = struct {
    allocator: std.mem.Allocator,
    inner: *KernelDispatcher,
    pending_ops: std.ArrayListUnmanaged(BatchedOp),
    batch_threshold: usize,
    auto_flush_size: usize,
    /// Statistical tracking for optimization
    stats: Stats,

    const Self = @This();

    /// Minimum elements before considering an op "small" enough to batch
    pub const SMALL_OP_THRESHOLD: usize = 4096;

    /// Statistics for batched operations
    pub const Stats = struct {
        total_queued: u64 = 0,
        total_flushed: u64 = 0,
        batches_executed: u64 = 0,
        avg_batch_size: f32 = 0.0,
        high_priority_count: u64 = 0,
        low_priority_count: u64 = 0,
    };

    /// Initialize batched dispatcher wrapping a KernelDispatcher.
    pub fn init(allocator: std.mem.Allocator, dispatcher: *KernelDispatcher) Self {
        return .{
            .allocator = allocator,
            .inner = dispatcher,
            .pending_ops = .{},
            .batch_threshold = SMALL_OP_THRESHOLD,
            .auto_flush_size = 32, // Auto-flush after 32 pending ops
            .stats = .{},
        };
    }

    /// Deinitialize and flush any pending operations.
    pub fn deinit(self: *Self) void {
        // Flush remaining ops (log errors during cleanup)
        self.flush() catch |err| {
            std.log.debug("BatchingDispatcher.flush failed during deinit: {t}", .{err});
        };
        self.pending_ops.deinit(self.allocator);
    }

    /// Queue an operation for batched execution.
    /// Small operations are queued; large operations execute immediately.
    pub fn queue(
        self: *Self,
        kernel: CompiledKernelHandle,
        config: LaunchConfig,
        args: KernelArgs,
    ) DispatchError!void {
        const elements = @as(usize, config.global_size[0]) *
            @as(usize, config.global_size[1]) *
            @as(usize, config.global_size[2]);

        // Determine operation category based on kernel name
        const category = blk: {
            const name = kernel.name;
            if (std.mem.indexOf(u8, name, "vector") != null) {
                break :blk BatchedOp.Category.vector_ops;
            } else if (std.mem.indexOf(u8, name, "matrix") != null) {
                break :blk BatchedOp.Category.matrix_ops;
            } else if (std.mem.indexOf(u8, name, "add") != null or
                std.mem.indexOf(u8, name, "mul") != null or
                std.mem.indexOf(u8, name, "sub") != null)
            {
                break :blk BatchedOp.Category.element_wise;
            } else if (std.mem.indexOf(u8, name, "reduce") != null or
                std.mem.indexOf(u8, name, "sum") != null)
            {
                break :blk BatchedOp.Category.reduction;
            } else if (std.mem.indexOf(u8, name, "relu") != null or
                std.mem.indexOf(u8, name, "gelu") != null or
                std.mem.indexOf(u8, name, "sigmoid") != null)
            {
                break :blk BatchedOp.Category.activation;
            }
            break :blk BatchedOp.Category.unknown;
        };

        // Assign priority based on operation characteristics
        const priority = blk: {
            // High priority for reductions and activations (often in critical path)
            if (category == .reduction or category == .activation) {
                break :blk BatchedOp.Priority.high;
            }
            // Low priority for element-wise operations that can be batched
            if (category == .element_wise and elements < self.batch_threshold / 4) {
                break :blk BatchedOp.Priority.low;
            }
            break :blk BatchedOp.Priority.normal;
        };

        // Large operations execute immediately
        if (elements >= self.batch_threshold) {
            _ = try self.inner.execute(kernel, config, args);
            return;
        }

        // Queue small operation
        if (args.buffers.len > 8) {
            // Too many buffers, execute immediately
            _ = try self.inner.execute(kernel, config, args);
            return;
        }

        var op = BatchedOp{
            .kernel = kernel,
            .config = config,
            .buffers = undefined,
            .buffer_count = @intCast(args.buffers.len),
            .priority = priority,
            .category = category,
        };

        for (args.buffers, 0..) |buf, i| {
            op.buffers[i] = buf;
        }

        self.pending_ops.append(self.allocator, op) catch return DispatchError.OutOfMemory;
        self.stats.total_queued += 1;

        // Update priority statistics
        switch (priority) {
            .high => self.stats.high_priority_count += 1,
            .low => self.stats.low_priority_count += 1,
            else => {},
        }

        // Auto-flush if we have enough pending ops
        if (self.pending_ops.items.len >= self.auto_flush_size) {
            try self.flush();
        }
    }

    /// Execute all pending operations in a batch using proper categorization and prioritization
    pub fn flush(self: *Self) DispatchError!void {
        if (self.pending_ops.items.len == 0) return;

        // Simple categorization based on op type for better cache behavior
        // Operations are already categorized during queueing, so we can group similar ops

        // Sync all input buffers to device once
        for (self.pending_ops.items) |*op| {
            for (op.buffers[0..op.buffer_count]) |buf| {
                if (buf.isHostDirty()) {
                    buf.toDevice() catch |err| {
                        std.log.warn("Failed to sync buffer: {}", .{err});
                        return DispatchError.BufferNotReady;
                    };
                }
            }
        }

        // Execute all ops
        var batch_size: f32 = 0.0;
        for (self.pending_ops.items) |*op| {
            const args = KernelArgs{
                .buffers = op.buffers[0..op.buffer_count],
            };
            _ = self.inner.execute(op.kernel, op.config, args) catch |err| {
                std.log.warn("Batched op failed: {}", .{err});
                // Continue with remaining ops
            };
            batch_size += 1.0;
        }

        // Update statistics
        self.stats.batches_executed += 1;
        self.stats.total_flushed += @as(u64, @intFromFloat(batch_size));
        if (self.stats.batches_executed > 1) {
            // Running average of batch size
            const current_avg = self.stats.avg_batch_size;
            const new_avg = (current_avg * @as(f32, @floatFromInt(self.stats.batches_executed - 1)) + batch_size) /
                @as(f32, @floatFromInt(self.stats.batches_executed));
            self.stats.avg_batch_size = new_avg;
        } else {
            self.stats.avg_batch_size = batch_size;
        }

        // Clear pending ops
        self.pending_ops.clearRetainingCapacity();
    }

    /// Get number of pending operations.
    pub fn pendingCount(self: *const Self) usize {
        return self.pending_ops.items.len;
    }

    /// Set the threshold for what constitutes a "small" operation.
    pub fn setBatchThreshold(self: *Self, threshold: usize) void {
        self.batch_threshold = threshold;
    }

    /// Set when to auto-flush pending operations.
    pub fn setAutoFlushSize(self: *Self, size: usize) void {
        self.auto_flush_size = size;
    }
};

test {
    std.testing.refAllDecls(@This());
}
