//! Vulkan command buffer pooling for efficient command submission.
//!
//! Provides command buffer allocation, recycling, and pool management
//! to minimize overhead from command buffer creation/destruction.

const std = @import("std");
const time = @import("../../../shared/utils/time.zig");
const types = @import("vulkan_types.zig");
const init = @import("vulkan_init.zig");

pub const VulkanError = types.VulkanError;

/// Command buffer state.
pub const CommandBufferState = enum {
    /// Buffer is available for use.
    available,
    /// Buffer is being recorded.
    recording,
    /// Buffer has been submitted.
    submitted,
    /// Buffer execution is complete.
    completed,
};

/// Managed command buffer with state tracking.
pub const ManagedCommandBuffer = struct {
    buffer: types.VkCommandBuffer,
    state: CommandBufferState,
    fence: ?types.VkFence,
    last_used: i64,
    reset_count: u64,

    pub fn isReady(self: *const ManagedCommandBuffer) bool {
        return self.state == .available or self.state == .completed;
    }
};

/// Command pool configuration.
pub const CommandPoolConfig = struct {
    /// Queue family index for the pool.
    queue_family: u32 = 0,
    /// Initial number of command buffers to pre-allocate.
    initial_buffer_count: u32 = 8,
    /// Maximum number of command buffers in pool.
    max_buffer_count: u32 = 64,
    /// Enable command buffer reset.
    enable_reset: bool = true,
    /// Enable transient allocations (short-lived buffers).
    transient: bool = false,
};

/// Command buffer pool for efficient allocation and reuse.
pub const CommandPool = struct {
    allocator: std.mem.Allocator,
    config: CommandPoolConfig,
    pool: types.VkCommandPool,
    buffers: std.ArrayListUnmanaged(ManagedCommandBuffer),
    available_buffers: std.ArrayListUnmanaged(usize),
    mutex: std.Thread.Mutex,
    allocation_count: u64,
    reset_count: u64,

    /// Initialize a command pool.
    pub fn init(allocator: std.mem.Allocator, config: CommandPoolConfig) !CommandPool {
        if (!init.vulkan_initialized or init.vulkan_context == null) {
            return VulkanError.InitializationFailed;
        }

        const ctx = &init.vulkan_context.?;

        // Create command pool
        var flags: u32 = 0;
        if (config.enable_reset) {
            flags |= 0x2; // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        }
        if (config.transient) {
            flags |= 0x1; // VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
        }

        const pool_create_info = types.VkCommandPoolCreateInfo{
            .flags = flags,
            .queueFamilyIndex = config.queue_family,
        };

        const create_pool_fn = init.vkCreateCommandPool orelse return VulkanError.InitializationFailed;
        var pool: types.VkCommandPool = undefined;
        const result = create_pool_fn(ctx.device, &pool_create_info, null, &pool);
        if (result != .success) {
            return VulkanError.InitializationFailed;
        }

        var cmd_pool = CommandPool{
            .allocator = allocator,
            .config = config,
            .pool = pool,
            .buffers = .{},
            .available_buffers = .{},
            .mutex = .{},
            .allocation_count = 0,
            .reset_count = 0,
        };

        // Pre-allocate initial command buffers
        try cmd_pool.allocateBuffers(config.initial_buffer_count);

        return cmd_pool;
    }

    /// Deinitialize the command pool.
    pub fn deinit(self: *CommandPool) void {
        if (!init.vulkan_initialized or init.vulkan_context == null) {
            return;
        }

        const ctx = &init.vulkan_context.?;

        // Destroy fences
        if (init.vkDestroyFence) |destroy_fence_fn| {
            for (self.buffers.items) |*managed| {
                if (managed.fence) |fence| {
                    destroy_fence_fn(ctx.device, fence, null);
                }
            }
        }

        // Destroy command pool (this frees all command buffers)
        if (init.vkDestroyCommandPool) |destroy_fn| {
            destroy_fn(ctx.device, self.pool, null);
        }

        self.buffers.deinit(self.allocator);
        self.available_buffers.deinit(self.allocator);
        self.* = undefined;
    }

    /// Acquire a command buffer from the pool.
    pub fn acquire(self: *CommandPool) !types.VkCommandBuffer {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Try to reuse an available buffer
        if (self.available_buffers.items.len > 0) {
            const idx = self.available_buffers.pop();
            const managed = &self.buffers.items[idx];

            // Reset if needed
            if (managed.state == .completed and self.config.enable_reset) {
                try self.resetBuffer(managed);
            }

            managed.state = .recording;
            managed.last_used = time.nowMilliseconds();
            return managed.buffer;
        }

        // Allocate new buffer if under limit
        if (self.buffers.items.len < self.config.max_buffer_count) {
            try self.allocateBuffers(1);
            const idx = self.buffers.items.len - 1;
            const managed = &self.buffers.items[idx];
            managed.state = .recording;
            managed.last_used = time.nowMilliseconds();
            return managed.buffer;
        }

        // Wait for oldest buffer to complete
        try self.waitForOldest();
        return self.acquire();
    }

    /// Release a command buffer back to the pool.
    pub fn release(self: *CommandPool, buffer: types.VkCommandBuffer) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Find the buffer
        for (self.buffers.items, 0..) |*managed, i| {
            if (managed.buffer == buffer) {
                managed.state = .completed;
                try self.available_buffers.append(self.allocator, i);
                return;
            }
        }

        return VulkanError.InvalidHandle;
    }

    /// Submit a command buffer and track its completion.
    pub fn submit(
        self: *CommandPool,
        buffer: types.VkCommandBuffer,
        queue: types.VkQueue,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!init.vulkan_initialized or init.vulkan_context == null) {
            return VulkanError.InitializationFailed;
        }

        const ctx = &init.vulkan_context.?;

        // Find the buffer
        for (self.buffers.items) |*managed| {
            if (managed.buffer == buffer) {
                // Create fence if needed
                if (managed.fence == null) {
                    const fence_create_info = types.VkFenceCreateInfo{
                        .flags = 0,
                    };

                    const create_fence_fn = init.vkCreateFence orelse return VulkanError.InitializationFailed;
                    var fence: types.VkFence = undefined;
                    const fence_result = create_fence_fn(ctx.device, &fence_create_info, null, &fence);
                    if (fence_result != .success) {
                        return VulkanError.InitializationFailed;
                    }
                    managed.fence = fence;
                }

                // Submit to queue
                const submit_info = types.VkSubmitInfo{
                    .commandBufferCount = 1,
                    .pCommandBuffers = &buffer,
                };

                const queue_submit_fn = init.vkQueueSubmit orelse return VulkanError.InitializationFailed;
                const submit_result = queue_submit_fn(queue, 1, &submit_info, managed.fence.?);
                if (submit_result != .success) {
                    return VulkanError.DeviceLost;
                }

                managed.state = .submitted;
                return;
            }
        }

        return VulkanError.InvalidHandle;
    }

    /// Wait for all submitted command buffers to complete.
    pub fn waitAll(self: *CommandPool) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!init.vulkan_initialized or init.vulkan_context == null) {
            return;
        }

        const ctx = &init.vulkan_context.?;
        const wait_fn = init.vkWaitForFences orelse return;

        for (self.buffers.items) |*managed| {
            if (managed.state == .submitted) {
                if (managed.fence) |fence| {
                    const wait_result = wait_fn(ctx.device, 1, &fence, 1, std.math.maxInt(u64));
                    if (wait_result != .success) {
                        return VulkanError.SynchronizationFailed;
                    }
                    managed.state = .completed;

                    // Reset fence
                    if (init.vkResetFences) |reset_fn| {
                        _ = reset_fn(ctx.device, 1, &fence);
                    }
                }
            }
        }
    }

    /// Reset all command buffers in the pool.
    pub fn resetAll(self: *CommandPool) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!init.vulkan_initialized or init.vulkan_context == null) {
            return VulkanError.InitializationFailed;
        }

        const ctx = &init.vulkan_context.?;

        // Reset entire pool at once
        const reset_pool_fn = init.vkResetCommandPool orelse return VulkanError.InitializationFailed;
        const result = reset_pool_fn(ctx.device, self.pool, 0);
        if (result != .success) {
            return VulkanError.InitializationFailed;
        }

        // Update state
        for (self.buffers.items) |*managed| {
            managed.state = .available;
            managed.reset_count += 1;
        }

        self.reset_count += 1;

        // Rebuild available list
        self.available_buffers.clearRetainingCapacity();
        for (0..self.buffers.items.len) |i| {
            try self.available_buffers.append(self.allocator, i);
        }
    }

    /// Get pool statistics.
    pub fn getStats(self: *const CommandPool) PoolStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        var available: usize = 0;
        var recording: usize = 0;
        var submitted: usize = 0;
        var completed: usize = 0;

        for (self.buffers.items) |managed| {
            switch (managed.state) {
                .available => available += 1,
                .recording => recording += 1,
                .submitted => submitted += 1,
                .completed => completed += 1,
            }
        }

        return .{
            .total_buffers = self.buffers.items.len,
            .available_buffers = available,
            .recording_buffers = recording,
            .submitted_buffers = submitted,
            .completed_buffers = completed,
            .allocation_count = self.allocation_count,
            .reset_count = self.reset_count,
        };
    }

    // Internal helpers
    fn allocateBuffers(self: *CommandPool, count: u32) !void {
        if (!init.vulkan_initialized or init.vulkan_context == null) {
            return VulkanError.InitializationFailed;
        }

        const ctx = &init.vulkan_context.?;

        const alloc_info = types.VkCommandBufferAllocateInfo{
            .commandPool = self.pool,
            .level = 0, // VK_COMMAND_BUFFER_LEVEL_PRIMARY
            .commandBufferCount = count,
        };

        const buffers = try self.allocator.alloc(types.VkCommandBuffer, count);
        defer self.allocator.free(buffers);

        const allocate_fn = init.vkAllocateCommandBuffers orelse return VulkanError.InitializationFailed;
        const result = allocate_fn(ctx.device, &alloc_info, buffers.ptr);
        if (result != .success) {
            return VulkanError.InitializationFailed;
        }

        for (buffers) |buffer| {
            const managed = ManagedCommandBuffer{
                .buffer = buffer,
                .state = .available,
                .fence = null,
                .last_used = time.nowMilliseconds(),
                .reset_count = 0,
            };
            try self.buffers.append(self.allocator, managed);
            try self.available_buffers.append(self.allocator, self.buffers.items.len - 1);
            self.allocation_count += 1;
        }
    }

    fn resetBuffer(self: *CommandPool, managed: *ManagedCommandBuffer) !void {
        if (!init.vulkan_initialized or init.vulkan_context == null) {
            return VulkanError.InitializationFailed;
        }

        const reset_fn = init.vkResetCommandBuffer orelse return VulkanError.InitializationFailed;
        const result = reset_fn(managed.buffer, 0);
        if (result != .success) {
            return VulkanError.InitializationFailed;
        }

        managed.reset_count += 1;
        managed.state = .available;
    }

    fn waitForOldest(self: *CommandPool) !void {
        if (!init.vulkan_initialized or init.vulkan_context == null) {
            return VulkanError.InitializationFailed;
        }

        const ctx = &init.vulkan_context.?;
        const wait_fn = init.vkWaitForFences orelse return VulkanError.InitializationFailed;

        var oldest_idx: ?usize = null;
        var oldest_time: i64 = std.math.maxInt(i64);

        for (self.buffers.items, 0..) |*managed, i| {
            if (managed.state == .submitted and managed.last_used < oldest_time) {
                oldest_time = managed.last_used;
                oldest_idx = i;
            }
        }

        if (oldest_idx) |idx| {
            const managed = &self.buffers.items[idx];
            if (managed.fence) |fence| {
                const result = wait_fn(ctx.device, 1, &fence, 1, std.math.maxInt(u64));
                if (result != .success) {
                    return VulkanError.SynchronizationFailed;
                }
                managed.state = .completed;

                // Reset fence
                if (init.vkResetFences) |reset_fn| {
                    _ = reset_fn(ctx.device, 1, &fence);
                }
            }
        }
    }
};

/// Pool statistics.
pub const PoolStats = struct {
    total_buffers: usize,
    available_buffers: usize,
    recording_buffers: usize,
    submitted_buffers: usize,
    completed_buffers: usize,
    allocation_count: u64,
    reset_count: u64,
};
