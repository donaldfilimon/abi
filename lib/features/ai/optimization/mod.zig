//! Training Optimization Module
//!
//! This module provides gradient-based optimization algorithms for training neural networks.
//! All optimizers follow a common interface and support standard hyperparameters.
//!
//! Supported Algorithms:
//! - **SGD**: Stochastic Gradient Descent with optional momentum and weight decay
//! - **Adam**: Adaptive Moment Estimation with bias correction
//! - **RMSProp**: Root Mean Square Propagation for adaptive learning rates
//!
//! All optimizers implement memory-efficient updates and proper resource management.

const std = @import("std");

/// Optimization algorithm types
pub const OptimizerType = enum {
    sgd,
    adam,
    rmsprop,
};

/// Common optimizer configuration
pub const OptimizerConfig = union(OptimizerType) {
    sgd: SGDConfig,
    adam: AdamConfig,
    rmsprop: RMSPropConfig,

    pub const SGDConfig = struct {
        learning_rate: f32 = 0.01,
        momentum: f32 = 0.0,
        weight_decay: f32 = 0.0,
    };

    pub const AdamConfig = struct {
        learning_rate: f32 = 0.001,
        beta1: f32 = 0.9,
        beta2: f32 = 0.999,
        epsilon: f32 = 1e-8,
        weight_decay: f32 = 0.0,
    };

    pub const RMSPropConfig = struct {
        learning_rate: f32 = 0.001,
        alpha: f32 = 0.99,
        epsilon: f32 = 1e-8,
        weight_decay: f32 = 0.0,
    };
};

/// Generic optimizer interface
pub const Optimizer = struct {
    const Self = @This();

    /// Function pointer for update step
    updateFn: *const fn (*anyopaque, []f32, []const f32) void,
    /// Context pointer (points to concrete optimizer)
    context: *anyopaque,
    /// Deinitialization function
    deinitFn: *const fn (*anyopaque) void,

    pub fn step(self: *Self, parameters: []f32, gradients: []const f32) void {
        self.updateFn(self.context, parameters, gradients);
    }

    pub fn deinit(self: *Self) void {
        self.deinitFn(self.context);
    }
};

/// Stochastic Gradient Descent optimizer
pub const SGD = struct {
    const Self = @This();

    config: OptimizerConfig.SGDConfig,
    velocities: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: OptimizerConfig.SGDConfig, param_count: usize) !SGD {
        const velocities = try allocator.alloc(f32, param_count);
        @memset(velocities, 0.0);

        return SGD{
            .config = config,
            .velocities = velocities,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.velocities);
    }

    pub fn step(self: *Self, parameters: []f32, gradients: []const f32) void {
        for (parameters, gradients, self.velocities) |*param, grad, *vel| {
            // v = momentum * v - learning_rate * (grad + weight_decay * param)
            vel.* = self.config.momentum * vel.* -
                self.config.learning_rate * (grad + self.config.weight_decay * param.*);
            // param = param + v
            param.* += vel.*;
        }
    }

    pub fn asOptimizer(self: *Self) Optimizer {
        return Optimizer{
            .updateFn = updateFn,
            .context = self,
            .deinitFn = deinitFn,
        };
    }

    fn updateFn(context: *anyopaque, parameters: []f32, gradients: []const f32) void {
        const self: *Self = @ptrCast(@alignCast(context));
        self.step(parameters, gradients);
    }

    fn deinitFn(context: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(context));
        self.deinit();
    }
};

/// Adam optimizer
pub const Adam = struct {
    const Self = @This();

    config: OptimizerConfig.AdamConfig,
    m: []f32, // First moment
    v: []f32, // Second moment
    t: u32, // Timestep
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: OptimizerConfig.AdamConfig, param_count: usize) !Adam {
        const m = try allocator.alloc(f32, param_count);
        errdefer allocator.free(m);
        @memset(m, 0.0);

        const v = try allocator.alloc(f32, param_count);
        errdefer allocator.free(v);
        @memset(v, 0.0);

        return Adam{
            .config = config,
            .m = m,
            .v = v,
            .t = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.m);
        self.allocator.free(self.v);
    }

    pub fn step(self: *Self, parameters: []f32, gradients: []const f32) void {
        self.t += 1;
        const t_f32 = @as(f32, @floatFromInt(self.t));

        for (parameters, gradients, self.m, self.v) |*param, grad, *m, *v| {
            // m = beta1 * m + (1 - beta1) * grad
            m.* = self.config.beta1 * m.* + (1 - self.config.beta1) * grad;
            // v = beta2 * v + (1 - beta2) * grad^2
            v.* = self.config.beta2 * v.* + (1 - self.config.beta2) * grad * grad;

            // Bias correction
            const m_hat = m.* / (1 - std.math.pow(f32, self.config.beta1, t_f32));
            const v_hat = v.* / (1 - std.math.pow(f32, self.config.beta2, t_f32));

            // param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
            const update = self.config.learning_rate * m_hat / (@sqrt(v_hat) + self.config.epsilon);
            param.* -= update + self.config.weight_decay * param.*;
        }
    }

    pub fn asOptimizer(self: *Self) Optimizer {
        return Optimizer{
            .updateFn = updateFn,
            .context = self,
            .deinitFn = deinitFn,
        };
    }

    fn updateFn(context: *anyopaque, parameters: []f32, gradients: []const f32) void {
        const self: *Self = @ptrCast(@alignCast(context));
        self.step(parameters, gradients);
    }

    fn deinitFn(context: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(context));
        self.deinit();
    }
};

/// RMSProp optimizer
pub const RMSProp = struct {
    const Self = @This();

    config: OptimizerConfig.RMSPropConfig,
    cache: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: OptimizerConfig.RMSPropConfig, param_count: usize) !RMSProp {
        const cache = try allocator.alloc(f32, param_count);
        @memset(cache, 0.0);

        return RMSProp{
            .config = config,
            .cache = cache,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.cache);
    }

    pub fn step(self: *Self, parameters: []f32, gradients: []const f32) void {
        for (parameters, gradients, self.cache) |*param, grad, *cache| {
            // cache = alpha * cache + (1 - alpha) * grad^2
            cache.* = self.config.alpha * cache.* + (1 - self.config.alpha) * grad * grad;

            // param = param - lr * grad / (sqrt(cache) + epsilon)
            const update = self.config.learning_rate * grad / (@sqrt(cache.*) + self.config.epsilon);
            param.* -= update + self.config.weight_decay * param.*;
        }
    }

    pub fn asOptimizer(self: *Self) Optimizer {
        return Optimizer{
            .updateFn = updateFn,
            .context = self,
            .deinitFn = deinitFn,
        };
    }

    fn updateFn(context: *anyopaque, parameters: []f32, gradients: []const f32) void {
        const self: *Self = @ptrCast(@alignCast(context));
        self.step(parameters, gradients);
    }

    fn deinitFn(context: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(context));
        self.deinit();
    }
};

/// Create an optimizer instance using the provided configuration.
/// This factory function allocates and initializes the appropriate optimizer
/// based on the configuration type.
///
/// # Parameters
/// - `allocator`: Memory allocator for optimizer state
/// - `config`: Configuration specifying optimizer type and hyperparameters
/// - `param_count`: Number of parameters to optimize
///
/// # Returns
/// Initialized optimizer ready for use
///
/// # Errors
/// - `OutOfMemory`: If memory allocation fails for optimizer state
///
/// # Example
/// ```zig
/// const config = OptimizerConfig{ .adam = .{
///     .learning_rate = 0.001,
///     .beta1 = 0.9,
///     .beta2 = 0.999,
/// }};
/// var optimizer = try createOptimizer(allocator, config, 1000);
/// defer optimizer.deinit();
/// ```
///
/// # Memory Management
/// The returned optimizer owns its internal state and must be deinitialized
/// with `deinit()` when no longer needed.
pub fn createOptimizer(allocator: std.mem.Allocator, config: OptimizerConfig, param_count: usize) !Optimizer {
    return switch (config) {
        .sgd => |sgd_config| {
            var sgd = try SGD.init(allocator, sgd_config, param_count);
            return sgd.asOptimizer();
        },
        .adam => |adam_config| {
            var adam = try Adam.init(allocator, adam_config, param_count);
            return adam.asOptimizer();
        },
        .rmsprop => |rms_config| {
            var rms = try RMSProp.init(allocator, rms_config, param_count);
            return rms.asOptimizer();
        },
    };
}

test "SGD optimizer basic functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var sgd = try SGD.init(allocator, .{ .learning_rate = 0.1 }, 3);
    defer sgd.deinit();

    var params = [_]f32{ 1.0, 2.0, 3.0 };
    const grads = [_]f32{ 0.1, 0.2, 0.3 };

    sgd.update(&params, &grads);

    // Parameters should be updated
    try testing.expect(params[0] < 1.0);
    try testing.expect(params[1] < 2.0);
    try testing.expect(params[2] < 3.0);
}

test "Adam optimizer initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var adam = try Adam.init(allocator, .{}, 5);
    defer adam.deinit();

    try testing.expectEqual(@as(usize, 5), adam.m.len);
    try testing.expectEqual(@as(usize, 5), adam.v.len);
    try testing.expectEqual(@as(u32, 0), adam.t);
}

test "Adam optimizer step update" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var adam = try Adam.init(allocator, .{
        .learning_rate = 0.1,
        .beta1 = 0.9,
        .beta2 = 0.999,
    }, 3);
    defer adam.deinit();

    var params = [_]f32{ 1.0, 2.0, 3.0 };
    const grads = [_]f32{ 0.1, 0.2, 0.3 };

    adam.step(&params, &grads);

    // Parameters should be updated (Adam adapts learning rate)
    try testing.expect(params[0] != 1.0);
    try testing.expect(params[1] != 2.0);
    try testing.expect(params[2] != 3.0);
    try testing.expectEqual(@as(u32, 1), adam.t);
}

test "RMSProp optimizer functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var rms = try RMSProp.init(allocator, .{
        .learning_rate = 0.1,
        .alpha = 0.9,
    }, 2);
    defer rms.deinit();

    var params = [_]f32{ 1.0, 2.0 };
    const grads = [_]f32{ 0.5, 0.3 };

    rms.step(&params, &grads);

    // Parameters should be updated
    try testing.expect(params[0] < 1.0);
    try testing.expect(params[1] < 2.0);
}

test "createOptimizer factory function" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test SGD creation
    const sgd_config = OptimizerConfig{ .sgd = .{ .learning_rate = 0.01 } };
    var sgd_opt = try createOptimizer(allocator, sgd_config, 10);
    defer sgd_opt.deinit();

    // Test Adam creation
    const adam_config = OptimizerConfig{ .adam = .{ .learning_rate = 0.001 } };
    var adam_opt = try createOptimizer(allocator, adam_config, 10);
    defer adam_opt.deinit();

    // Test RMSProp creation
    const rms_config = OptimizerConfig{ .rmsprop = .{ .learning_rate = 0.01 } };
    var rms_opt = try createOptimizer(allocator, rms_config, 10);
    defer rms_opt.deinit();

    // Test that they can perform updates
    var params = [_]f32{ 1.0, 2.0 };
    const grads = [_]f32{ 0.1, 0.2 };

    sgd_opt.update(&params, &grads);
    try testing.expect(params[0] < 1.0);
}
