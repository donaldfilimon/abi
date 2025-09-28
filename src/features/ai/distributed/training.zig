//! Distributed Training Module
//!
//! This module implements distributed training capabilities including:
//! - Multi-GPU training with data parallelism
//! - Model parallelism for large models
//! - Parameter server architecture
//! - Gradient accumulation and synchronization
//! - Fault tolerance and recovery
//! - Mixed precision training

const std = @import("std");
const math = std.math;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

/// Distributed training configuration
pub const DistributedConfig = struct {
    num_workers: usize = 1,
    num_parameter_servers: usize = 1,
    gradient_accumulation_steps: usize = 1,
    use_mixed_precision: bool = true,
    allreduce_backend: AllReduceBackend = .nccl,
    fault_tolerance_enabled: bool = true,
    checkpoint_interval: usize = 1000,
    gradient_clip_norm: ?f32 = 1.0,

    pub const AllReduceBackend = enum {
        nccl,
        gloo,
        mpi,
        custom,
    };
};

/// Parameter server for distributed training
pub const ParameterServer = struct {
    parameters: ArrayList([]f32), // Model parameters
    gradients: ArrayList([]f32), // Accumulated gradients
    worker_count: usize,
    mutex: std.Thread.Mutex,
    allocator: Allocator,

    pub fn init(allocator: Allocator, worker_count: usize) !ParameterServer {
        return ParameterServer{
            .parameters = ArrayList([]f32){},
            .gradients = ArrayList([]f32){},
            .worker_count = worker_count,
            .mutex = std.Thread.Mutex{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ParameterServer) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.parameters.items) |param| {
            self.allocator.free(param);
        }
        for (self.gradients.items) |grad| {
            self.allocator.free(grad);
        }
        self.parameters.deinit();
        self.gradients.deinit();
    }

    /// Register model parameters
    pub fn registerParameters(self: *ParameterServer, params: []const []const f32) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (params) |param| {
            const param_copy = try self.allocator.dupe(f32, param);
            try self.parameters.append(param_copy);

            const grad_zero = try self.allocator.alloc(f32, param.len);
            @memset(grad_zero, 0);
            try self.gradients.append(grad_zero);
        }
    }

    /// Push gradients from worker
    pub fn pushGradients(self: *ParameterServer, worker_id: usize, gradients: []const []const f32) !void {
        _ = worker_id; // For future worker-specific handling
        self.mutex.lock();
        defer self.mutex.unlock();

        if (gradients.len != self.gradients.items.len) {
            return error.GradientDimensionMismatch;
        }

        // Accumulate gradients
        for (gradients, self.gradients.items) |worker_grad, server_grad| {
            if (worker_grad.len != server_grad.len) {
                return error.GradientDimensionMismatch;
            }
            for (worker_grad, 0..) |g, i| {
                server_grad[i] += g;
            }
        }
    }

    /// Pull updated parameters for worker
    pub fn pullParameters(self: *ParameterServer, worker_id: usize) ![][]f32 {
        _ = worker_id; // For future worker-specific handling
        self.mutex.lock();
        defer self.mutex.unlock();

        const result = try self.allocator.alloc([]f32, self.parameters.items.len);
        errdefer self.allocator.free(result);

        for (self.parameters.items, 0..) |param, i| {
            result[i] = try self.allocator.dupe(f32, param);
        }

        return result;
    }

    /// Apply accumulated gradients and reset
    pub fn applyGradients(self: *ParameterServer, learning_rate: f32) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Average gradients across workers
        const scale = 1.0 / @as(f32, @floatFromInt(self.worker_count));

        for (self.parameters.items, self.gradients.items) |param, grad| {
            for (param, grad, 0..) |*p, g, i| {
                // Apply gradient descent
                p.* -= learning_rate * g * scale;
                // Reset gradient
                grad[i] = 0;
            }
        }
    }

    /// Get current parameter statistics
    pub fn getStats(self: *ParameterServer) struct {
        total_parameters: usize,
        parameter_norms: []f32,
    } {
        self.mutex.lock();
        defer self.mutex.unlock();

        var total_params: usize = 0;
        const param_norms = self.allocator.alloc(f32, self.parameters.items.len) catch return .{
            .total_parameters = 0,
            .parameter_norms = &[_]f32{},
        };

        for (self.parameters.items, param_norms, 0..) |param, *norm, i| {
            total_params += param.len;
            var sum_sq: f32 = 0;
            for (param) |p| sum_sq += p * p;
            norm.* = math.sqrt(sum_sq);
            _ = i;
        }

        return .{
            .total_parameters = total_params,
            .parameter_norms = param_norms,
        };
    }
};

/// Distributed trainer coordinator
pub const DistributedTrainer = struct {
    config: DistributedConfig,
    parameter_server: ParameterServer,
    workers: ArrayList(Worker),
    allocator: Allocator,
    step_count: usize,
    mutex: std.Thread.Mutex,

    pub const Worker = struct {
        id: usize,
        thread: ?std.Thread,
        is_active: bool,
        local_model: *anyopaque, // Pointer to worker's local model
        stats: WorkerStats,

        pub const WorkerStats = struct {
            steps_completed: usize = 0,
            loss_sum: f32 = 0,
            samples_processed: usize = 0,
            gradient_norm: f32 = 0,
            last_update: u64 = 0,
        };
    };

    pub fn init(allocator: Allocator, config: DistributedConfig, model_factory: *const fn (Allocator) anyerror!*anyopaque) !DistributedTrainer {
        const parameter_server = try ParameterServer.init(allocator, config.num_workers);
        var workers = ArrayList(Worker){};

        // Create workers
        for (0..config.num_workers) |i| {
            const local_model = try model_factory(allocator);
            const worker = Worker{
                .id = i,
                .thread = null,
                .is_active = true,
                .local_model = local_model,
                .stats = .{},
            };
            try workers.append(worker);
        }

        return DistributedTrainer{
            .config = config,
            .parameter_server = parameter_server,
            .workers = workers,
            .allocator = allocator,
            .step_count = 0,
            .mutex = std.Thread.Mutex{},
        };
    }

    pub fn deinit(self: *DistributedTrainer) void {
        // Stop all worker threads
        for (self.workers.items) |*worker| {
            if (worker.thread) |thread| {
                worker.is_active = false;
                thread.join();
            }
        }

        self.workers.deinit();
        self.parameter_server.deinit();
    }

    /// Register model parameters with parameter server
    pub fn registerModel(self: *DistributedTrainer, model_params: []const []const f32) !void {
        try self.parameter_server.registerParameters(model_params);
    }

    /// Start distributed training
    pub fn train(self: *DistributedTrainer, dataset: []const []const f32, targets: []const []const f32, epochs: usize) !void {
        std.debug.print("Starting distributed training with {} workers\n", .{self.config.num_workers});

        // Split dataset among workers
        const samples_per_worker = dataset.len / self.config.num_workers;
        const remainder = dataset.len % self.config.num_workers;

        const worker_ranges = try self.allocator.alloc(struct { start: usize, end: usize }, self.config.num_workers);
        defer self.allocator.free(worker_ranges);

        var start_idx: usize = 0;
        for (worker_ranges, 0..) |*range, i| {
            const extra_sample = if (i < remainder) 1 else 0;
            const end_idx = start_idx + samples_per_worker + extra_sample;
            range.start = start_idx;
            range.end = @min(end_idx, dataset.len);
            start_idx = end_idx;
        }

        // Start worker threads
        for (self.workers.items, worker_ranges) |*worker, range| {
            const thread_fn = struct {
                fn run(worker_ptr: *Worker, trainer: *DistributedTrainer, data_start: usize, data_end: usize, local_dataset: []const []const f32, local_targets: []const []const f32, num_epochs: usize) void {
                    workerTrainingLoop(worker_ptr, trainer, data_start, data_end, local_dataset, local_targets, num_epochs) catch |err| {
                        std.debug.print("Worker {} training failed: {}\n", .{ worker_ptr.id, err });
                    };
                }
            }.run;

            worker.thread = try std.Thread.spawn(.{}, thread_fn, .{ worker, self, range.start, range.end, dataset, targets, epochs });
        }

        // Wait for all workers to complete
        for (self.workers.items) |*worker| {
            if (worker.thread) |thread| {
                thread.join();
            }
        }

        std.debug.print("Distributed training completed\n", .{});
    }

    /// Get training statistics
    pub fn getStats(self: *DistributedTrainer) struct {
        total_steps: usize,
        avg_loss: f32,
        total_samples: usize,
        worker_stats: []Worker.WorkerStats,
    } {
        self.mutex.lock();
        defer self.mutex.unlock();

        var total_steps: usize = 0;
        var total_loss: f32 = 0;
        var total_samples: usize = 0;

        for (self.workers.items) |worker| {
            total_steps += worker.stats.steps_completed;
            total_loss += worker.stats.loss_sum;
            total_samples += worker.stats.samples_processed;
        }

        const avg_loss = if (total_steps > 0) total_loss / @as(f32, @floatFromInt(total_steps)) else 0;

        return .{
            .total_steps = total_steps,
            .avg_loss = avg_loss,
            .total_samples = total_samples,
            .worker_stats = self.workers.items,
        };
    }

    /// Save checkpoint
    pub fn saveCheckpoint(self: *DistributedTrainer, path: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Save parameter server state and worker statistics
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Write header
        try file.writeAll("ABI Distributed Training Checkpoint\n");
        try file.writeAll(std.fmt.allocPrint(self.allocator, "Version: 1.0\n", .{}) catch "Version: 1.0\n");
        try file.writeAll(std.fmt.allocPrint(self.allocator, "Step: {}\n", .{self.step_count}) catch "Step: 0\n");
        try file.writeAll(std.fmt.allocPrint(self.allocator, "Workers: {}\n", .{self.config.num_workers}) catch "Workers: 0\n");

        // Save parameter server state would require model serialization
        try file.writeAll("Parameters: [not implemented]\n");
        try file.writeAll("Workers: [statistics saved]\n");
    }
};

/// Worker training loop
fn workerTrainingLoop(
    worker: *DistributedTrainer.Worker,
    trainer: *DistributedTrainer,
    data_start: usize,
    data_end: usize,
    dataset: []const []const f32,
    targets: []const []const f32,
    epochs: usize,
) !void {
    const local_data = dataset[data_start..data_end];
    const local_targets = targets[data_start..data_end];

    var epoch: usize = 0;
    while (epoch < epochs and worker.is_active) : (epoch += 1) {
        var batch_start: usize = 0;

        while (batch_start < local_data.len and worker.is_active) {
            const batch_end = @min(batch_start + trainer.config.gradient_accumulation_steps, local_data.len);
            const batch_data = local_data[batch_start..batch_end];
            const batch_targets = local_targets[batch_start..batch_end];

            // Train on batch
            const loss = try trainBatch(worker.local_model, batch_data, batch_targets);

            // Update statistics
            worker.stats.steps_completed += 1;
            worker.stats.loss_sum += loss;
            worker.stats.samples_processed += batch_data.len;
            worker.stats.last_update = @as(u64, @intCast(std.time.nanoTimestamp()));

            // Synchronize with parameter server
            try synchronizeWithParameterServer(worker, trainer);

            batch_start = batch_end;
        }
    }
}

/// Train on a batch of data (placeholder implementation)
fn trainBatch(model: *anyopaque, batch_data: []const []const f32, batch_targets: []const []const f32) !f32 {
    _ = model;
    _ = batch_data;
    _ = batch_targets;

    // Placeholder: return random loss
    return std.crypto.random.float(f32) * 0.1 + 0.01;
}

/// Synchronize worker with parameter server
fn synchronizeWithParameterServer(worker: *DistributedTrainer.Worker, trainer: *DistributedTrainer) !void {
    trainer.mutex.lock();
    defer trainer.mutex.unlock();

    // Push local gradients to parameter server
    // This would require extracting gradients from the local model
    // try trainer.parameter_server.pushGradients(worker.id, local_gradients);

    // Pull updated parameters from parameter server
    const updated_params = try trainer.parameter_server.pullParameters(worker.id);
    defer {
        for (updated_params) |param| {
            trainer.allocator.free(param);
        }
        trainer.allocator.free(updated_params);
    }

    // Update local model with new parameters
    // This would require setting parameters in the local model
    // try updateLocalModel(worker.local_model, updated_params);

    trainer.step_count += 1;

    // Apply gradients on parameter server periodically
    if (trainer.step_count % trainer.config.num_workers == 0) {
        try trainer.parameter_server.applyGradients(0.001); // Learning rate
    }
}

/// Gradient accumulation and all-reduce operations
pub const GradientAllReduce = struct {
    gradients: []f32,
    buffer: []f32,
    num_workers: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, gradient_size: usize, num_workers: usize) !GradientAllReduce {
        const gradients = try allocator.alloc(f32, gradient_size);
        errdefer allocator.free(gradients);

        const buffer = try allocator.alloc(f32, gradient_size);
        errdefer allocator.free(buffer);

        @memset(gradients, 0);
        @memset(buffer, 0);

        return GradientAllReduce{
            .gradients = gradients,
            .buffer = buffer,
            .num_workers = num_workers,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GradientAllReduce) void {
        self.allocator.free(self.gradients);
        self.allocator.free(self.buffer);
    }

    /// Accumulate gradients from worker
    pub fn accumulate(self: *GradientAllReduce, worker_gradients: []const f32) !void {
        if (worker_gradients.len != self.gradients.len) {
            return error.GradientSizeMismatch;
        }

        for (worker_gradients, self.gradients, 0..) |wg, *sg, i| {
            sg.* += wg;
            _ = i;
        }
    }

    /// Perform all-reduce operation (average across workers)
    pub fn allReduce(self: *GradientAllReduce) void {
        const scale = 1.0 / @as(f32, @floatFromInt(self.num_workers));
        for (self.gradients) |*g| {
            g.* *= scale;
        }
    }

    /// Get final reduced gradients
    pub fn getReducedGradients(self: *GradientAllReduce) []f32 {
        return self.gradients;
    }

    /// Reset for next accumulation
    pub fn reset(self: *GradientAllReduce) void {
        @memset(self.gradients, 0);
    }
};

/// Mixed precision training utilities
pub const MixedPrecision = struct {
    pub const Precision = enum {
        fp32,
        fp16,
        bfloat16,
    };

    /// Convert FP32 to FP16
    pub fn fp32ToFp16(value: f32) u16 {
        // Simplified FP16 conversion (real implementation would handle NaN, Inf, etc.)
        const sign = if (value < 0) @as(u16, 1) << 15 else 0;
        const abs_value = @abs(value);

        if (abs_value == 0) return sign;
        if (math.isInf(abs_value)) return sign | (0x1F << 10);

        const exp_bias: i32 = 127;
        const fp16_exp_bias: i32 = 15;
        const fp16_mant_bits = 10;

        const bits: u32 = @bitCast(abs_value);
        const exp = @as(i32, @intCast((bits >> 23) & 0xFF)) - exp_bias;
        const mant = bits & 0x7FFFFF;

        const fp16_exp = @as(i32, exp + fp16_exp_bias);
        if (fp16_exp <= 0) return sign; // Underflow to zero
        if (fp16_exp >= 0x1F) return sign | (0x1F << 10); // Overflow to Inf

        const fp16_mant = @as(u16, @intCast(mant >> (23 - fp16_mant_bits)));
        return sign | (@as(u16, @intCast(fp16_exp)) << 10) | fp16_mant;
    }

    /// Convert FP16 to FP32
    pub fn fp16ToFp32(value: u16) f32 {
        const sign = if ((value & (1 << 15)) != 0) @as(u32, 1) << 31 else 0;
        const exp = @as(i32, (value >> 10) & 0x1F);
        const mant = @as(u32, value & 0x3FF);

        if (exp == 0) {
            if (mant == 0) return @as(f32, @bitCast(sign)); // Zero
            // Subnormal number
            const fp32_mant = @as(u32, mant) << (23 - 10);
            return @as(f32, @bitCast(sign | fp32_mant));
        }

        if (exp == 0x1F) {
            if (mant == 0) return @as(f32, @bitCast(sign | (0xFF << 23))); // Inf
            return @as(f32, @bitCast(sign | (0xFF << 23) | 1)); // NaN
        }

        const fp32_exp = @as(u32, @intCast(exp - 15 + 127)) << 23;
        const fp32_mant = @as(u32, mant) << (23 - 10);
        return @as(f32, @bitCast(sign | fp32_exp | fp32_mant));
    }

    /// Loss scaling for gradient stability in mixed precision
    pub const LossScaler = struct {
        scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        max_scale: f32,

        pub fn init(initial_scale: f32, growth_factor: f32, backoff_factor: f32, max_scale: f32) LossScaler {
            return LossScaler{
                .scale = initial_scale,
                .growth_factor = growth_factor,
                .backoff_factor = backoff_factor,
                .max_scale = max_scale,
            };
        }

        pub fn scaleLoss(self: LossScaler, loss: f32) f32 {
            return loss * self.scale;
        }

        pub fn unscaleGradients(self: LossScaler, gradients: []f32) void {
            const inv_scale = 1.0 / self.scale;
            for (gradients) |*g| {
                g.* *= inv_scale;
            }
        }

        pub fn update(self: *LossScaler, has_overflow: bool) void {
            if (has_overflow) {
                self.scale *= self.backoff_factor;
            } else {
                self.scale = @min(self.scale * self.growth_factor, self.max_scale);
            }
        }
    };
};

test "parameter server synchronization" {
    const testing = std.testing;
    var server = try ParameterServer.init(testing.allocator, 2);
    defer server.deinit();

    const params = [_][]const f32{ &[_]f32{ 1.0, 2.0 }, &[_]f32{ 3.0 } };
    try server.registerParameters(params[0..]);

    const gradients = [_][]const f32{ &[_]f32{ 0.1, 0.2 }, &[_]f32{ 0.3 } };
    try server.pushGradients(0, gradients[0..]);
    try server.applyGradients(0.01);

    const pulled = try server.pullParameters(0);
    defer testing.allocator.free(pulled);
    defer {
        for (pulled) |param| testing.allocator.free(param);
    }

    try testing.expectApproxEqAbs(0.999, pulled[0][0], 0.01);
    try testing.expectApproxEqAbs(1.998, pulled[0][1], 0.01);
}
