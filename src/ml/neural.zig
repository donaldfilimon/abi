//! Neural Network Module for Self-Learning Vector Embeddings
//!
//! This module provides neural network capabilities for learning and generating
//! vector embeddings with enhanced memory safety and efficiency. It includes:
//! - Feed-forward neural network with configurable layers
//! - Backpropagation training with SGD optimizer and gradient checkpointing
//! - Activation functions (ReLU, Sigmoid, Tanh)
//! - Embedding generation from raw input
//! - SIMD-accelerated matrix operations
//! - Memory pool system for efficient buffer reuse
//! - Mixed precision training support
//! - Enhanced error handling and memory safety

const std = @import("std");
const math = std.math;
const core = @import("../core/mod.zig");
const abi = @import("../root.zig");
const simd = @import("../simd/mod.zig");
const memory_tracker = @import("../perf/memory_tracker.zig");

/// Re-export commonly used types
pub const Allocator = core.Allocator;

/// Neural network layer types
pub const LayerType = enum {
    Dense,
    Embedding,
    Dropout,
};

/// Activation functions with mixed precision support
pub const Activation = enum {
    ReLU,
    Sigmoid,
    Tanh,
    None,

    /// Apply activation function to a f32 value
    pub fn apply(self: Activation, x: f32) f32 {
        return switch (self) {
            .ReLU => if (x > 0) x else 0,
            .Sigmoid => 1.0 / (1.0 + math.exp(-x)),
            .Tanh => math.tanh(x),
            .None => x,
        };
    }

    /// Apply activation function to a f16 value
    pub fn applyF16(self: Activation, x: f16) f16 {
        return switch (self) {
            .ReLU => if (x > 0) x else 0,
            .Sigmoid => 1.0 / (1.0 + @as(f16, @floatCast(std.math.exp(@as(f64, -x))))),
            .Tanh => @as(f16, @floatCast(math.tanh(@as(f64, x)))),
            .None => x,
        };
    }

    /// Derivative of activation function (f32)
    pub fn derivative(self: Activation, x: f32) f32 {
        return switch (self) {
            .ReLU => if (x > 0) 1 else 0,
            .Sigmoid => {
                const s = self.apply(x);
                return s * (1 - s);
            },
            .Tanh => {
                const t = math.tanh(x);
                return 1 - t * t;
            },
            .None => 1,
        };
    }

    /// Derivative of activation function (f16)
    pub fn derivativeF16(self: Activation, x: f16) f16 {
        return switch (self) {
            .ReLU => if (x > 0) 1 else 0,
            .Sigmoid => {
                const s = self.applyF16(x);
                return s * (1 - s);
            },
            .Tanh => {
                const t = self.applyF16(x);
                return 1 - t * t;
            },
            .None => 1,
        };
    }
};

/// Precision mode for computations
pub const Precision = enum {
    f32, // Standard 32-bit float
    f16, // 16-bit float for memory efficiency
    mixed, // Mixed precision (f16 forward, f32 backward)
};

/// Neural network training configuration with enhanced memory options
pub const TrainingConfig = struct {
    learning_rate: f32 = 0.001,
    batch_size: usize = 32,
    epochs: usize = 100,
    momentum: f32 = 0.9,
    weight_decay: f32 = 0.0001,
    early_stopping_patience: usize = 10,
    validation_split: f32 = 0.2,
    /// Precision mode for computation
    precision: Precision = .f32,
    /// Enable gradient checkpointing
    enable_checkpointing: bool = true,
    /// Checkpoint interval for gradient computation
    checkpoint_interval: usize = 10,
    /// Memory pool configuration
    memory_pool_config: MemoryPool.PoolConfig = .{},
};

/// Layer configuration
pub const LayerConfig = struct {
    type: LayerType,
    input_size: usize,
    output_size: usize,
    activation: Activation = .None,
    dropout_rate: f32 = 0.0,
    weight_init_scale: f32 = 1.0,
};

/// Complete neural network configuration
pub const NetworkConfig = struct {
    input_size: usize,
    hidden_layers: []const LayerConfig,
    output_size: usize,
    training: TrainingConfig = .{},
};

/// Memory pool for efficient buffer reuse
pub const MemoryPool = struct {
    /// Memory pool configuration
    pub const PoolConfig = struct {
        /// Initial buffer capacity
        initial_capacity: usize = 1024,
        /// Maximum buffer size to pool
        max_buffer_size: usize = 1024 * 1024, // 1MB
        /// Enable memory tracking
        enable_tracking: bool = true,
    };

    /// Pooled buffer
    pub const PooledBuffer = struct {
        /// Buffer data
        data: []f32,
        /// Original size requested
        size: usize,
        /// Pool this buffer belongs to
        pool: *MemoryPool,
        /// Whether buffer is currently in use
        in_use: bool = false,

        /// Return buffer to pool
        pub fn release(self: *PooledBuffer) void {
            if (!self.in_use) return;
            self.in_use = false;
            // Reset buffer contents for safety
            @memset(self.data[0..self.size], 0);
        }

        /// Get buffer as slice of requested size
        pub fn slice(self: *PooledBuffer, size: usize) []f32 {
            std.debug.assert(size <= self.data.len);
            return self.data[0..size];
        }
    };

    /// Enhanced buffer with liveness tracking
    pub const TrackedBuffer = struct {
        /// The actual buffer
        buffer: *PooledBuffer,
        /// Last access timestamp
        last_access: u64,
        /// Access count
        access_count: usize,
        /// Whether buffer is currently in use
        in_use: bool,

        /// Check if buffer is stale (not accessed recently)
        pub fn isStale(self: TrackedBuffer, current_time: u64, stale_threshold_ns: u64) bool {
            return !self.in_use and (current_time - self.last_access) > stale_threshold_ns;
        }

        /// Update access time
        pub fn markAccessed(self: *TrackedBuffer, current_time: u64) void {
            self.last_access = current_time;
            self.access_count += 1;
        }
    };

    /// Liveness analysis configuration
    pub const LivenessConfig = struct {
        /// Stale buffer threshold (nanoseconds)
        stale_threshold_ns: u64 = 1_000_000_000, // 1 second
        /// Enable automatic cleanup
        enable_auto_cleanup: bool = true,
        /// Cleanup interval (nanoseconds)
        cleanup_interval_ns: u64 = 10_000_000_000, // 10 seconds
        /// Maximum stale buffers before forced cleanup
        max_stale_buffers: usize = 100,
    };

    /// Available buffers - simplified to avoid hash map corruption
    available_buffers: std.ArrayListUnmanaged(*PooledBuffer) = .{},
    /// Configuration
    config: PoolConfig,
    /// Parent allocator
    allocator: std.mem.Allocator,
    /// Memory tracker (optional)
    memory_tracker: ?*memory_tracker.MemoryProfiler = null,
    /// Liveness analysis state
    liveness_config: LivenessConfig = .{},
    /// Tracked buffers for liveness analysis - simplified
    tracked_buffers: std.ArrayListUnmanaged(TrackedBuffer) = .{},
    /// Last cleanup time
    last_cleanup_time: u64 = 0,

    /// Initialize memory pool
    pub fn init(allocator: std.mem.Allocator, config: PoolConfig) !*MemoryPool {
        const self = try allocator.create(MemoryPool);
        errdefer allocator.destroy(self);

        self.* = .{
            .available_buffers = .{},
            .config = config,
            .allocator = allocator,
            .memory_tracker = if (config.enable_tracking) memory_tracker.getGlobalProfiler() else null,
            .tracked_buffers = .{},
            .liveness_config = .{},
            .last_cleanup_time = 0,
        };

        return self;
    }

    /// Deinitialize memory pool
    pub fn deinit(self: *MemoryPool) void {
        // Clean up available buffers safely
        for (self.available_buffers.items) |buffer| {
            if (buffer.data.len > 0) {
                self.allocator.free(buffer.data);
            }
            self.allocator.destroy(buffer);
        }
        self.available_buffers.deinit(self.allocator);

        // Clean up tracked buffers safely
        self.tracked_buffers.deinit(self.allocator);

        self.allocator.destroy(self);
    }

    /// Allocate buffer from pool or create new one
    pub fn allocBuffer(self: *MemoryPool, size: usize) !*PooledBuffer {
        // Find a suitable buffer in the available list
        for (self.available_buffers.items, 0..) |buffer, i| {
            if (buffer.size >= size) {
                // Remove from available list
                _ = self.available_buffers.swapRemove(i);
                // Record access for liveness analysis
                self.recordBufferAccess(buffer);
                return buffer;
            }
        }

        // Create new buffer (using regular allocation for now)
        const data = try self.allocator.alloc(f32, size);
        errdefer self.allocator.free(data);

        const buffer = try self.allocator.create(PooledBuffer);
        errdefer self.allocator.destroy(buffer);

        buffer.* = .{
            .data = data,
            .size = size,
            .pool = self,
        };
        buffer.in_use = true;

        // Track allocation if enabled
        if (self.memory_tracker) |tracker| {
            _ = try tracker.recordAllocation(
                @as(u64, @intCast(size * @sizeOf(f32))),
                32, // Fixed alignment for now
                @src().file,
                @src().line,
                "MemoryPool.allocBuffer",
                null,
            );
        }

        // Record access for liveness analysis
        self.recordBufferAccess(buffer);

        return buffer;
    }

    /// Return buffer to pool for reuse
    pub fn returnBuffer(self: *MemoryPool, buffer: *PooledBuffer) void {
        if (buffer.size > self.config.max_buffer_size) {
            // Don't pool large buffers
            self.allocator.free(buffer.data);
            self.allocator.destroy(buffer);
            return;
        }

        buffer.in_use = false;
        @memset(buffer.data[0..buffer.size], 0); // Clear for safety

        // Add to available buffers list, but limit pool size to prevent memory bloat
        if (self.available_buffers.items.len < 50) { // Keep max 50 total buffers
            self.available_buffers.append(self.allocator, buffer) catch {
                // If append fails, just free the buffer
                self.allocator.free(buffer.data);
                self.allocator.destroy(buffer);
            };
        } else {
            // Pool is full, free the buffer
            self.allocator.free(buffer.data);
            self.allocator.destroy(buffer);
        }
    }

    /// Get pool statistics
    pub fn getStats(self: *MemoryPool) struct {
        total_pooled_buffers: usize,
        total_memory_used: usize,
        buffer_sizes: usize,
    } {
        var total_buffers: usize = 0;
        var total_memory: usize = 0;

        // Count unique buffer sizes for the sizes statistic
        var size_set = std.AutoHashMapUnmanaged(usize, void){};
        defer size_set.deinit(self.allocator);

        for (self.available_buffers.items) |buffer| {
            total_buffers += 1;
            total_memory += buffer.data.len * @sizeOf(f32);
            _ = size_set.put(self.allocator, buffer.size, {}) catch {};
        }

        return .{
            .total_pooled_buffers = total_buffers,
            .total_memory_used = total_memory,
            .buffer_sizes = size_set.count(),
        };
    }

    /// Initialize liveness analysis
    pub fn initLivenessAnalysis(self: *MemoryPool, config: LivenessConfig) void {
        self.liveness_config = config;
        self.last_cleanup_time = @as(u64, @intCast(std.time.nanoTimestamp()));
    }

    /// Record buffer access for liveness analysis
    pub fn recordBufferAccess(self: *MemoryPool, buffer: *PooledBuffer) void {
        if (!self.liveness_config.enable_auto_cleanup) return;

        const current_time = std.time.nanoTimestamp();

        // Find existing tracked buffer or add new one
        var found = false;
        for (self.tracked_buffers.items) |*tracked| {
            if (tracked.buffer == buffer) {
                tracked.markAccessed(@as(u64, @intCast(current_time)));
                tracked.in_use = buffer.in_use;
                found = true;
                break;
            }
        }

        if (!found) {
            // Add new tracked buffer
            const tracked = TrackedBuffer{
                .buffer = buffer,
                .last_access = @as(u64, @intCast(current_time)),
                .access_count = 1,
                .in_use = buffer.in_use,
            };
            self.tracked_buffers.append(self.allocator, tracked) catch return;
        }

        // Periodic cleanup check
        if (current_time - self.last_cleanup_time > self.liveness_config.cleanup_interval_ns) {
            self.performLivenessCleanup(@as(u64, @intCast(current_time)));
            self.last_cleanup_time = @as(u64, @intCast(current_time));
        }
    }

    /// Perform liveness-based cleanup
    pub fn performLivenessCleanup(self: *MemoryPool, current_time: u64) void {
        // Skip cleanup if no tracked buffers
        if (self.tracked_buffers.items.len == 0) return;

        var stale_buffers = std.ArrayListUnmanaged(*PooledBuffer){};
        defer stale_buffers.deinit(self.allocator);

        // Find stale buffers
        for (self.tracked_buffers.items) |tracked| {
            if (tracked.isStale(current_time, self.liveness_config.stale_threshold_ns)) {
                stale_buffers.append(self.allocator, tracked.buffer) catch continue;
            }
        }

        // Remove stale buffers from pool
        for (stale_buffers.items) |buffer| {
            // Remove from available buffers
            for (self.available_buffers.items, 0..) |avail_buf, i| {
                if (avail_buf == buffer) {
                    _ = self.available_buffers.swapRemove(i);
                    break;
                }
            }

            // Free the buffer
            self.allocator.free(buffer.data);
            self.allocator.destroy(buffer);

            // Remove from tracking
            for (self.tracked_buffers.items, 0..) |tracked, i| {
                if (tracked.buffer == buffer) {
                    _ = self.tracked_buffers.swapRemove(i);
                    break;
                }
            }
        }

        if (stale_buffers.items.len > 0) {
            std.log.info("MemoryPool: Cleaned up {d} stale buffers", .{stale_buffers.items.len});
        }
    }

    /// Get liveness statistics
    pub fn getLivenessStats(self: *MemoryPool) struct {
        total_tracked_buffers: usize,
        active_buffers: usize,
        stale_buffers: usize,
        average_access_count: f64,
    } {
        const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
        var active_count: usize = 0;
        var stale_count: usize = 0;
        var total_accesses: usize = 0;

        // Iterate over tracked buffers
        for (self.tracked_buffers.items) |tracked| {
            if (tracked.in_use) {
                active_count += 1;
            } else if (tracked.isStale(current_time, self.liveness_config.stale_threshold_ns)) {
                stale_count += 1;
            }
            total_accesses += tracked.access_count;
        }

        const total_tracked = self.tracked_buffers.items.len;
        const avg_access = if (total_tracked > 0) @as(f64, @floatFromInt(total_accesses)) / @as(f64, @floatFromInt(total_tracked)) else 0.0;

        return .{
            .total_tracked_buffers = total_tracked,
            .active_buffers = active_count,
            .stale_buffers = stale_count,
            .average_access_count = avg_access,
        };
    }
};

/// Neural network layer with enhanced memory safety and mixed precision support
pub const Layer = struct {
    type: LayerType,
    weights: []f32,
    biases: []f32,
    /// f16 versions for mixed precision training
    weights_f16: ?[]f16 = null,
    biases_f16: ?[]f16 = null,
    activation: Activation,
    dropout_rate: f32,
    input_size: usize,
    output_size: usize,
    allocator: std.mem.Allocator,
    /// Memory pool for this layer (optional)
    memory_pool: ?*MemoryPool = null,
    /// Alignment used for weights and biases allocation
    weights_alignment: u29 = 32,
    biases_alignment: u29 = 32,
    /// Cached buffers for forward pass
    cached_output: ?[]f32 = null,
    /// Cached buffers for backward pass
    cached_input_gradient: ?[]f32 = null,

    /// Initialize a new layer with memory pool support
    pub fn init(allocator: std.mem.Allocator, config: LayerConfig, memory_pool: ?*MemoryPool) !*Layer {
        const self = try allocator.create(Layer);
        errdefer allocator.destroy(self);

        // Allocate weights and biases with proper alignment
        const weights_alignment = comptime std.mem.Alignment.fromByteUnits(32);
        const biases_alignment = comptime std.mem.Alignment.fromByteUnits(32);

        const weights = try allocator.alignedAlloc(f32, weights_alignment, config.input_size * config.output_size);
        errdefer self.allocator.free(weights);
        const biases = try allocator.alignedAlloc(f32, biases_alignment, config.output_size);
        errdefer self.allocator.free(biases);

        self.* = .{
            .type = config.type,
            .weights = weights,
            .biases = biases,
            .activation = config.activation,
            .dropout_rate = config.dropout_rate,
            .input_size = config.input_size,
            .output_size = config.output_size,
            .allocator = allocator,
            .memory_pool = memory_pool,
            .weights_alignment = @as(u29, @intCast(@intFromEnum(weights_alignment))),
            .biases_alignment = @as(u29, @intCast(@intFromEnum(biases_alignment))),
        };

        // Initialize weights with Xavier/Glorot initialization
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(config.input_size + config.output_size)));
        var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const random = rng.random();

        for (self.weights) |*w| {
            w.* = (random.float(f32) * 2 - 1) * scale;
        }
        for (self.biases) |*b| {
            b.* = 0;
        }

        return self;
    }

    /// Initialize f16 versions for mixed precision training
    pub fn initF16(self: *Layer) !void {
        // Convert f32 weights/biases to f16
        self.weights_f16 = try self.allocator.alignedAlloc(f16, std.mem.Alignment.fromByteUnits(32), self.weights.len);
        errdefer self.allocator.free(self.weights_f16.?);

        self.biases_f16 = try self.allocator.alignedAlloc(f16, std.mem.Alignment.fromByteUnits(32), self.biases.len);
        errdefer self.allocator.free(self.biases_f16.?);

        // Convert values
        for (self.weights, self.weights_f16.?) |f32_val, *f16_val| {
            f16_val.* = @as(f16, @floatCast(f32_val));
        }
        for (self.biases, self.biases_f16.?) |f32_val, *f16_val| {
            f16_val.* = @as(f16, @floatCast(f32_val));
        }
    }

    /// Synchronize f16 weights/biases back to f32 after training
    pub fn syncToF32(self: *Layer) void {
        if (self.weights_f16) |weights_f16| {
            for (self.weights, weights_f16) |*f32_val, f16_val| {
                f32_val.* = @as(f32, @floatCast(f16_val));
            }
        }
        if (self.biases_f16) |biases_f16| {
            for (self.biases, biases_f16) |*f32_val, f16_val| {
                f32_val.* = @as(f32, @floatCast(f16_val));
            }
        }
    }

    /// Forward pass with mixed precision support
    pub fn forwardMixed(self: *Layer, input: []const f32, use_f16: bool) ![]f32 {
        std.debug.assert(input.len == self.input_size);

        if (use_f16 and self.weights_f16 == null) {
            try self.initF16();
        }

        var output = try self.allocBuffer(self.output_size);
        errdefer self.freeBuffer(output);

        if (use_f16 and self.weights_f16 != null) {
            // Mixed precision forward pass using f16
            const weights_f16 = self.weights_f16.?;
            const biases_f16 = self.biases_f16.?;

            var i: usize = 0;
            while (i < self.output_size) : (i += 1) {
                const weights_start = i * self.input_size;
                const weights_end = weights_start + self.input_size;

                var sum: f16 = biases_f16[i];
                for (input, weights_f16[weights_start..weights_end]) |in_val, w_val| {
                    sum += @as(f16, @floatCast(in_val)) * w_val;
                }

                // Apply activation and convert back to f32
                output[i] = @as(f32, @floatCast(self.activation.applyF16(sum)));
            }
        } else {
            // Standard f32 forward pass
            var i: usize = 0;
            while (i < self.output_size) : (i += 1) {
                const weights_start = i * self.input_size;
                const weights_end = weights_start + self.input_size;

                var sum: f32 = self.biases[i];
                for (input, self.weights[weights_start..weights_end]) |in_val, w_val| {
                    sum += in_val * w_val;
                }

                output[i] = self.activation.apply(sum);
            }
        }

        // Apply dropout during training
        if (self.dropout_rate > 0) {
            var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
            const random = rng.random();
            for (output) |*o| {
                if (random.float(f32) < self.dropout_rate) {
                    o.* = 0;
                } else {
                    o.* /= (1.0 - self.dropout_rate); // Scale to maintain expected value
                }
            }
        }

        return output;
    }

    /// Backward pass with mixed precision support
    pub fn backwardMixed(
        self: *Layer,
        input: []const f32,
        output: []const f32,
        output_gradient: []const f32,
        learning_rate: f32,
        use_f16: bool,
    ) ![]f32 {
        std.debug.assert(output_gradient.len == self.output_size);

        if (use_f16 and self.weights_f16 == null) {
            try self.initF16();
        }

        var input_gradient = try self.allocBuffer(self.input_size);
        errdefer self.freeBuffer(input_gradient);

        if (use_f16 and self.weights_f16 != null) {
            // Mixed precision backward pass
            const weights_f16 = self.weights_f16.?;
            const biases_f16 = self.biases_f16.?;

            for (0..self.output_size) |i| {
                const gradient_f16 = @as(f16, @floatCast(output_gradient[i])) *
                    self.activation.derivativeF16(@as(f16, @floatCast(output[i])));

                // Update biases (f16)
                biases_f16[i] -= @as(f16, @floatCast(learning_rate)) * gradient_f16;

                // Update weights and calculate input gradient
                const weights_start = i * self.input_size;
                for (0..self.input_size) |j| {
                    const weight_idx = weights_start + j;
                    const input_f16 = @as(f16, @floatCast(input[j]));

                    input_gradient[j] += @as(f32, @floatCast(weights_f16[weight_idx] * gradient_f16));
                    weights_f16[weight_idx] -= @as(f16, @floatCast(learning_rate)) * gradient_f16 * input_f16;
                }
            }
        } else {
            // Standard f32 backward pass
            for (0..self.output_size) |i| {
                const gradient = output_gradient[i] * self.activation.derivative(output[i]);

                // Update biases
                self.biases[i] -= learning_rate * gradient;

                // Update weights and calculate input gradient
                const weights_start = i * self.input_size;
                for (0..self.input_size) |j| {
                    const weight_idx = weights_start + j;
                    input_gradient[j] += self.weights[weight_idx] * gradient;
                    self.weights[weight_idx] -= learning_rate * gradient * input[j];
                }
            }
        }

        return input_gradient;
    }

    /// Allocate buffer using memory pool if available, fallback to allocator
    pub fn allocBuffer(self: *Layer, size: usize) ![]f32 {
        if (self.memory_pool) |pool| {
            const buffer = try pool.allocBuffer(size);
            return buffer.slice(size);
        } else {
            return try self.allocator.alloc(f32, size);
        }
    }

    /// Free buffer using memory pool if available, fallback to allocator
    pub fn freeBuffer(self: *Layer, buffer: []f32) void {
        if (self.memory_pool != null) {
            // Find the buffer in the pool and release it
            // Note: This is a simplified implementation. In practice, you'd need
            // to track which buffers came from the pool.
            self.allocator.free(buffer);
        } else {
            self.allocator.free(buffer);
        }
    }

    /// Free layer resources with proper cleanup
    pub fn deinit(self: *Layer) void {
        // Free cached buffers if they exist
        if (self.cached_output) |output| {
            self.freeBuffer(output);
        }
        if (self.cached_input_gradient) |gradient| {
            self.freeBuffer(gradient);
        }

        // Free f16 weights and biases if they exist
        if (self.weights_f16) |weights_f16| {
            self.allocator.rawFree(@as([]u8, @alignCast(std.mem.sliceAsBytes(weights_f16))), @as(std.mem.Alignment, @enumFromInt(self.weights_alignment)), @returnAddress());
        }
        if (self.biases_f16) |biases_f16| {
            self.allocator.rawFree(@as([]u8, @alignCast(std.mem.sliceAsBytes(biases_f16))), @as(std.mem.Alignment, @enumFromInt(self.biases_alignment)), @returnAddress());
        }

        // Free weights and biases with proper alignment
        self.allocator.rawFree(@as([]u8, @alignCast(std.mem.sliceAsBytes(self.weights))), @as(std.mem.Alignment, @enumFromInt(self.weights_alignment)), @returnAddress());
        self.allocator.rawFree(@as([]u8, @alignCast(std.mem.sliceAsBytes(self.biases))), @as(std.mem.Alignment, @enumFromInt(self.biases_alignment)), @returnAddress());

        // Clean up memory pool if this layer owns it
        if (self.memory_pool) |pool| {
            pool.deinit();
        }

        self.allocator.destroy(self);
    }

    /// Forward pass through the layer with memory pool support
    pub fn forward(self: *Layer, input: []const f32) ![]f32 {
        std.debug.assert(input.len == self.input_size);
        var output = try self.allocBuffer(self.output_size);
        errdefer self.freeBuffer(output);

        // Matrix multiplication with SIMD
        var i: usize = 0;
        while (i < self.output_size) : (i += 1) {
            const weights_start = i * self.input_size;
            const weights_end = weights_start + self.input_size;
            output[i] = simd.dotProduct(
                input,
                self.weights[weights_start..weights_end],
            ) + self.biases[i];
            output[i] = self.activation.apply(output[i]);
        }

        // Apply dropout during training
        if (self.dropout_rate > 0) {
            var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
            const random = rng.random();
            for (output) |*o| {
                if (random.float(f32) < self.dropout_rate) {
                    o.* = 0;
                } else {
                    o.* /= (1.0 - self.dropout_rate); // Scale to maintain expected value
                }
            }
        }

        return output;
    }

    /// Backward pass through the layer with memory pool support
    pub fn backward(
        self: *Layer,
        input: []const f32,
        output: []const f32,
        output_gradient: []const f32,
        learning_rate: f32,
    ) ![]f32 {
        std.debug.assert(output_gradient.len == self.output_size);
        var input_gradient = try self.allocBuffer(self.input_size);
        errdefer self.freeBuffer(input_gradient);

        // Calculate gradients
        for (0..self.output_size) |i| {
            const gradient = output_gradient[i] *
                self.activation.derivative(output[i]);

            // Update biases
            self.biases[i] -= learning_rate * gradient;

            // Update weights and calculate input gradient
            const weights_start = i * self.input_size;
            for (0..self.input_size) |j| {
                const weight_idx = weights_start + j;
                input_gradient[j] += self.weights[weight_idx] * gradient;
                self.weights[weight_idx] -= learning_rate * gradient * input[j];
            }
        }

        return input_gradient;
    }
};

/// Gradient checkpointing state
pub const CheckpointState = struct {
    /// Whether checkpointing is enabled
    enabled: bool = false,
    /// Checkpoint interval
    interval: usize = 10,
    /// Stored checkpoints (inputs at specific layers)
    checkpoints: ?std.ArrayList([]f32) = null,
    /// Current step count
    step_count: usize = 0,
};

/// Neural network for learning embeddings with enhanced memory safety
pub const NeuralNetwork = struct {
    layers: std.ArrayList(*Layer),
    allocator: std.mem.Allocator,
    /// Shared memory pool for all layers
    memory_pool: ?*MemoryPool = null,
    /// Training configuration
    training_config: TrainingConfig = .{},
    /// Gradient checkpointing state
    checkpoint_state: CheckpointState,
    /// Precision mode
    precision: Precision = .f32,

    /// Initialize a new neural network with optional memory pool
    pub fn init(allocator: std.mem.Allocator, config: TrainingConfig) !*NeuralNetwork {
        const self = try allocator.create(NeuralNetwork);
        errdefer allocator.destroy(self);

        var checkpoints = try std.ArrayList([]f32).initCapacity(allocator, 16);
        errdefer checkpoints.deinit(allocator);

        self.* = .{
            .layers = try std.ArrayList(*Layer).initCapacity(allocator, 16),
            .allocator = allocator,
            .training_config = config,
            .precision = config.precision,
            .checkpoint_state = .{
                .enabled = config.enable_checkpointing,
                .interval = config.checkpoint_interval,
                .checkpoints = if (config.enable_checkpointing) checkpoints else null,
                .step_count = 0,
            },
        };

        // Initialize memory pool if configured
        // Temporarily disabled due to memory corruption issues in hash map implementation
        if (false and config.memory_pool_config.enable_tracking) {
            self.memory_pool = try MemoryPool.init(allocator, config.memory_pool_config);
        }

        return self;
    }

    /// Initialize a new neural network with default configuration (backward compatibility)
    pub fn initDefault(allocator: std.mem.Allocator) !*NeuralNetwork {
        return try init(allocator, .{});
    }

    /// Free network resources with proper cleanup
    pub fn deinit(self: *NeuralNetwork) void {
        // Clean up checkpoints
        if (self.checkpoint_state.checkpoints) |*checkpoints| {
            for (checkpoints.items) |checkpoint| {
                self.allocator.free(checkpoint);
            }
            checkpoints.deinit(self.allocator);
        }

        // Clean up layers
        for (self.layers.items) |layer| {
            layer.deinit();
        }
        self.layers.deinit(self.allocator);

        // Clean up memory pool
        if (self.memory_pool) |pool| {
            pool.deinit();
        }

        self.allocator.destroy(self);
    }

    /// Deinitialize with enhanced cleanup (for MemoryPool with liveness analysis)
    pub fn deinitEnhanced(self: *MemoryPool) void {
        // Clean up tracked buffers - simplified to avoid iterator issues
        self.tracked_buffers.deinit(self.allocator);

        // Call regular deinit
        self.deinit();
    }

    /// Add a layer to the network with memory pool support
    pub fn addLayer(self: *NeuralNetwork, config: LayerConfig) !void {
        const layer = try Layer.init(self.allocator, config, self.memory_pool);
        errdefer layer.deinit();
        try self.layers.append(self.allocator, layer);
    }

    /// Save network to file (basic implementation)
    pub fn saveToFile(self: *NeuralNetwork, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Write header
        _ = try file.write("NEURAL_NETWORK_V1\n");

        // Write layer count
        var layer_count_buf: [32]u8 = undefined;
        const layer_count_str = try std.fmt.bufPrint(&layer_count_buf, "{}\n", .{self.layers.items.len});
        _ = try file.write(layer_count_str);

        // Write each layer (simplified - just layer type and sizes)
        for (self.layers.items) |layer| {
            var layer_buf: [128]u8 = undefined;
            const layer_str = try std.fmt.bufPrint(&layer_buf, "{} {} {}\n", .{
                @intFromEnum(layer.type),
                layer.input_size,
                layer.output_size,
            });
            _ = try file.write(layer_str);
        }
    }

    /// Load network from file (basic implementation)
    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !*NeuralNetwork {
        // For now, return a simple error - full implementation needs more complex parsing
        // that would require more Zig standard library features
        _ = allocator;
        _ = path;
        return error.NotImplemented;
    }

    /// Forward pass through the network with memory optimization
    pub fn forward(self: *NeuralNetwork, input: []const f32) ![]f32 {
        var current = try self.allocator.dupe(f32, input);
        errdefer self.allocator.free(current);

        // Clear existing checkpoints if starting fresh
        if (self.checkpoint_state.enabled and self.checkpoint_state.step_count == 0) {
            if (self.checkpoint_state.checkpoints) |*checkpoints| {
                for (checkpoints.items) |checkpoint| {
                    self.allocator.free(checkpoint);
                }
                checkpoints.clearRetainingCapacity();
            }
        }

        for (self.layers.items, 0..) |layer, layer_idx| {
            const next = try layer.forward(current);

            // Store checkpoint if enabled and at checkpoint interval
            if (self.checkpoint_state.enabled and
                (layer_idx + 1) % self.checkpoint_state.interval == 0)
            {
                if (self.checkpoint_state.checkpoints) |*checkpoints| {
                    const checkpoint = try self.allocator.dupe(f32, current);
                    try checkpoints.append(self.allocator, checkpoint);
                }
            }

            self.allocator.free(current);
            current = next;
        }

        if (self.checkpoint_state.enabled) {
            self.checkpoint_state.step_count += 1;
        }

        return current;
    }

    /// Forward pass with mixed precision support
    pub fn forwardMixed(self: *NeuralNetwork, input: []const f32) ![]f32 {
        const use_f16 = self.precision == .f16 or self.precision == .mixed;
        var current = try self.allocator.dupe(f32, input);
        errdefer self.allocator.free(current);

        // Clear existing checkpoints if starting fresh
        if (self.checkpoint_state.enabled and self.checkpoint_state.step_count == 0) {
            if (self.checkpoint_state.checkpoints) |*checkpoints| {
                for (checkpoints.items) |checkpoint| {
                    self.allocator.free(checkpoint);
                }
                checkpoints.clearRetainingCapacity();
            }
        }

        for (self.layers.items, 0..) |layer, layer_idx| {
            const next = try layer.forwardMixed(current, use_f16);

            // Store checkpoint if enabled and at checkpoint interval
            if (self.checkpoint_state.enabled and
                (layer_idx + 1) % self.checkpoint_state.interval == 0)
            {
                if (self.checkpoint_state.checkpoints) |*checkpoints| {
                    const checkpoint = try self.allocator.dupe(f32, current);
                    try checkpoints.append(self.allocator, checkpoint);
                }
            }

            self.allocator.free(current);
            current = next;
        }

        if (self.checkpoint_state.enabled) {
            self.checkpoint_state.step_count += 1;
        }

        return current;
    }

    /// Train the network on a single sample with memory optimization
    pub fn trainStep(
        self: *NeuralNetwork,
        input: []const f32,
        target: []const f32,
        learning_rate: f32,
    ) !f32 {
        // Forward pass with intermediate values
        var activations = try std.ArrayList([]f32).initCapacity(self.allocator, 8);
        defer {
            for (activations.items) |activation| {
                self.allocator.free(activation);
            }
            activations.deinit(self.allocator);
        }

        try activations.append(self.allocator, try self.allocator.dupe(f32, input));
        for (self.layers.items) |layer| {
            const output = try layer.forward(
                activations.items[activations.items.len - 1],
            );
            try activations.append(self.allocator, output);
        }

        // Calculate loss and output gradient
        const output = activations.items[activations.items.len - 1];
        var loss: f32 = 0;
        var output_gradient = try self.allocator.alloc(f32, output.len);
        errdefer self.allocator.free(output_gradient);

        for (output, target, 0..) |o, t, i| {
            const diff = o - t;
            loss += diff * diff;
            output_gradient[i] = 2 * diff; // MSE derivative
        }
        loss /= @as(f32, @floatFromInt(output.len));

        // Backward pass
        var current_gradient = output_gradient;
        var i: usize = self.layers.items.len;
        while (i > 0) : (i -= 1) {
            const layer = self.layers.items[i - 1];
            const layer_input = activations.items[i - 1];
            const layer_output = activations.items[i];
            const input_gradient = try layer.backward(
                layer_input,
                layer_output,
                current_gradient,
                learning_rate,
            );
            errdefer self.allocator.free(input_gradient);

            // Free the current gradient if it's not the original output_gradient
            if (i < self.layers.items.len) {
                self.allocator.free(current_gradient);
            }

            // For the last layer, we don't need the input gradient
            if (i > 1) {
                current_gradient = input_gradient;
            } else {
                // Last layer - free the input gradient since we don't return it
                self.allocator.free(input_gradient);
            }
        }

        // Free the original output gradient
        self.allocator.free(output_gradient);

        return loss;
    }

    /// Train the network on a single sample with mixed precision support
    pub fn trainStepMixed(
        self: *NeuralNetwork,
        input: []const f32,
        target: []const f32,
        learning_rate: f32,
    ) !f32 {
        const use_f16 = self.precision == .f16 or self.precision == .mixed;

        // Forward pass with intermediate values
        var activations = try std.ArrayList([]f32).initCapacity(self.allocator, 8);
        defer {
            for (activations.items) |activation| {
                self.allocator.free(activation);
            }
            activations.deinit(self.allocator);
        }

        try activations.append(self.allocator, try self.allocator.dupe(f32, input));
        for (self.layers.items) |layer| {
            const output = try layer.forwardMixed(
                activations.items[activations.items.len - 1],
                use_f16,
            );
            try activations.append(self.allocator, output);
        }

        // Calculate loss and output gradient
        const output = activations.items[activations.items.len - 1];
        var loss: f32 = 0;
        var output_gradient = try self.allocator.alloc(f32, output.len);
        errdefer self.allocator.free(output_gradient);

        for (output, target, 0..) |o, t, i| {
            const diff = o - t;
            loss += diff * diff;
            output_gradient[i] = 2 * diff; // MSE derivative
        }
        loss /= @as(f32, @floatFromInt(output.len));

        // Backward pass
        var current_gradient = output_gradient;
        var i: usize = self.layers.items.len;
        while (i > 0) : (i -= 1) {
            const layer = self.layers.items[i - 1];
            const layer_input = activations.items[i - 1];
            const layer_output = activations.items[i];
            const input_gradient = try layer.backwardMixed(
                layer_input,
                layer_output,
                current_gradient,
                learning_rate,
                use_f16,
            );
            errdefer self.allocator.free(input_gradient);

            // Free the current gradient if it's not the original output_gradient
            if (i < self.layers.items.len) {
                self.allocator.free(current_gradient);
            }

            // For the last layer, we don't need the input gradient
            if (i > 1) {
                current_gradient = input_gradient;
            } else {
                // Last layer - free the input gradient since we don't return it
                self.allocator.free(input_gradient);
            }
        }

        // Free the original output gradient
        self.allocator.free(output_gradient);

        // Sync f16 weights back to f32 for mixed precision
        if (self.precision == .mixed) {
            for (self.layers.items) |layer| {
                layer.syncToF32();
            }
        }

        return loss;
    }
};

test "neural network basics" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create a simple network
    var nn = try NeuralNetwork.init(allocator, .{});
    defer nn.deinit();

    // Add layers
    try nn.addLayer(.{
        .type = .Dense,
        .input_size = 2,
        .output_size = 3,
        .activation = .ReLU,
    });
    try nn.addLayer(.{
        .type = .Dense,
        .input_size = 3,
        .output_size = 1,
        .activation = .Sigmoid,
    });

    // Test forward pass
    const input = [_]f32{ 0.5, -0.2 };
    const output = try nn.forward(&input);
    defer allocator.free(output);

    try testing.expect(output.len == 1);
    try testing.expect(output[0] >= 0 and output[0] <= 1);

    // Test training
    const target = [_]f32{0.7};
    const loss = try nn.trainStep(&input, &target, 0.1);
    try testing.expect(loss >= 0);
}



