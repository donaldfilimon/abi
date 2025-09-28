const std = @import("std");

/// Enumeration of supported optimizers for the AI stack.
pub const OptimizerType = enum {
    sgd,
    momentum_sgd,
    nesterov_sgd,
    adam,
    adamw,
    adamax,
    nadam,
    rmsprop,
    adagrad,
    adadelta,
    adambound,
    radam,
    lookahead,
    lamb,
};

/// Scheduler strategies applied to the optimizer learning rate.
pub const SchedulerType = enum {
    constant,
    step_decay,
    exponential_decay,
    polynomial_decay,
    cosine_annealing,
    cosine_annealing_warm_restarts,
    reduce_on_plateau,
    cyclic,
    one_cycle,
};

/// Scheduler configuration that can be shared across modules.
pub const SchedulerConfig = struct {
    kind: SchedulerType = .constant,
    decay_rate: f32 = 0.1,
    decay_steps: usize = 1000,
    warmup_steps: usize = 0,
    minimum_learning_rate: f32 = 0.0,
};

/// Complete optimizer configuration bundle.
pub const OptimizerConfig = struct {
    optimizer: OptimizerType = .adam,
    learning_rate: f32 = 0.001,
    weight_decay: f32 = 0.0001,
    use_momentum: bool = true,
    momentum: f32 = 0.9,
    nesterov: bool = false,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
    amsgrad: bool = false,
    scheduler: SchedulerConfig = .{},

    pub fn withLearningRate(self: OptimizerConfig, lr: f32) OptimizerConfig {
        var updated = self;
        updated.learning_rate = lr;
        return updated;
    }
};

/// Interface-based abstraction for optimizer behaviour.
pub const OptimizerOps = struct {
    apply_gradients: fn (ctx: *anyopaque, params: []f32, grads: []const f32) anyerror!void,
    update_learning_rate: fn (ctx: *anyopaque, step: usize) f32,
};

const StatelessContext = struct {
    learning_rate: f32,
    weight_decay: f32,
};

/// Lightweight stateless optimizer implementation useful for tests and fallbacks.
pub fn createStatelessOps(config: OptimizerConfig) struct {
    ops: OptimizerOps,
    context: StatelessContext,
} {
    const context = StatelessContext{
        .learning_rate = config.learning_rate,
        .weight_decay = config.weight_decay,
    };

    const ops = OptimizerOps{
        .apply_gradients = struct {
            fn apply(ctx: *anyopaque, params: []f32, grads: []const f32) !void {
                const c: *StatelessContext = @ptrCast(@alignCast(ctx));
                for (params, grads) |*param, grad| {
                    param.* -= c.learning_rate * grad + c.weight_decay * param.*;
                }
            }
        }.apply,
        .update_learning_rate = struct {
            fn update(ctx: *anyopaque, step: usize) f32 {
                const c: *StatelessContext = @ptrCast(@alignCast(ctx));
                const decay = std.math.pow(f32, 0.95, @as(f32, @floatFromInt(step)));
                return c.learning_rate * decay;
            }
        }.update,
    };

    return .{ .ops = ops, .context = context };
}

test "optimizer config adjustments" {
    const cfg = OptimizerConfig{};
    const updated = cfg.withLearningRate(0.01);
    try std.testing.expectApproxEqAbs(0.01, updated.learning_rate, 1e-6);
    try std.testing.expect(updated.optimizer == .adam);
}

test "stateless optimizer ops update" {
    const config = OptimizerConfig{ .learning_rate = 0.1, .weight_decay = 0.05 };
    var bundle = createStatelessOps(config);

    var params = [_]f32{ 1.0, 2.0 };
    const grads = [_]f32{ 0.5, 1.0 };
    try bundle.ops.apply_gradients(@ptrCast(&bundle.context), params[0..], grads[0..]);
    try std.testing.expect(params[0] < 1.0);
    try std.testing.expect(params[1] < 2.0);

    const lr = bundle.ops.update_learning_rate(@ptrCast(&bundle.context), 5);
    try std.testing.expect(lr < config.learning_rate);
}
