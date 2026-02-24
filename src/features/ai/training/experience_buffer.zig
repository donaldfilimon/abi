//! Experience Replay Buffer
//!
//! Priority experience replay buffer with importance sampling
//! for the self-learning system.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const learning_types = @import("learning_types.zig");

pub const LearningExperience = learning_types.LearningExperience;

/// Priority experience replay buffer with importance sampling
pub const ExperienceBuffer = struct {
    allocator: std.mem.Allocator,
    experiences: std.ArrayListUnmanaged(LearningExperience),
    priorities: std.ArrayListUnmanaged(f32),
    capacity: usize,
    total_priority: f64,
    alpha: f32, // Priority exponent
    beta: f32, // Importance sampling exponent
    beta_increment: f32,
    next_id: u64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, capacity: usize) Self {
        return .{
            .allocator = allocator,
            .experiences = .{},
            .priorities = .{},
            .capacity = capacity,
            .total_priority = 0,
            .alpha = 0.6,
            .beta = 0.4,
            .beta_increment = 0.001,
            .next_id = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.experiences.items) |*exp| {
            exp.deinit(self.allocator);
        }
        self.experiences.deinit(self.allocator);
        self.priorities.deinit(self.allocator);
    }

    /// Add experience with initial priority
    pub fn add(self: *Self, experience: LearningExperience) !void {
        const max_priority: f32 = if (self.priorities.items.len > 0) blk: {
            var max: f32 = 0;
            for (self.priorities.items) |p| {
                if (p > max) max = p;
            }
            break :blk max;
        } else 1.0;

        if (self.experiences.items.len >= self.capacity) {
            // Remove lowest priority experience
            var min_idx: usize = 0;
            var min_priority: f32 = self.priorities.items[0];
            for (self.priorities.items, 0..) |p, i| {
                if (p < min_priority) {
                    min_priority = p;
                    min_idx = i;
                }
            }
            self.total_priority -= std.math.pow(f64, min_priority, self.alpha);
            self.experiences.items[min_idx].deinit(self.allocator);
            _ = self.experiences.orderedRemove(min_idx);
            _ = self.priorities.orderedRemove(min_idx);
        }

        var exp_copy = experience;
        exp_copy.id = self.next_id;
        self.next_id += 1;

        try self.experiences.append(self.allocator, exp_copy);
        try self.priorities.append(self.allocator, max_priority);
        self.total_priority += std.math.pow(f64, max_priority, self.alpha);
    }

    /// Sample a batch with priority weighting
    pub fn sample(self: *Self, batch_size: usize) !SampledBatch {
        if (self.experiences.items.len < batch_size) {
            return error.InsufficientExperiences;
        }

        var indices = try self.allocator.alloc(usize, batch_size);
        errdefer self.allocator.free(indices);
        var weights = try self.allocator.alloc(f32, batch_size);
        errdefer self.allocator.free(weights);

        const n = self.experiences.items.len;
        const segment = self.total_priority / @as(f64, @floatFromInt(batch_size));

        // Get time-based seed
        const seed = blk: {
            var timer = time.Timer.start() catch break :blk @as(u64, 0);
            break :blk timer.read();
        };
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();

        for (0..batch_size) |i| {
            const lower = segment * @as(f64, @floatFromInt(i));
            const upper = segment * @as(f64, @floatFromInt(i + 1));
            const sample_val = lower + random.float(f64) * (upper - lower);

            var cumsum: f64 = 0;
            var idx: usize = 0;
            for (self.priorities.items, 0..) |p, j| {
                cumsum += std.math.pow(f64, p, self.alpha);
                if (cumsum >= sample_val) {
                    idx = j;
                    break;
                }
            }
            indices[i] = idx;

            // Importance sampling weight
            const prob = std.math.pow(f64, self.priorities.items[idx], self.alpha) / self.total_priority;
            const weight = std.math.pow(f32, @floatCast(1.0 / (@as(f64, @floatFromInt(n)) * prob)), self.beta);
            weights[i] = weight;
        }

        // Normalize weights
        var max_weight: f32 = 0;
        for (weights) |w| {
            if (w > max_weight) max_weight = w;
        }
        if (max_weight > 0) {
            for (weights) |*w| {
                w.* /= max_weight;
            }
        }

        // Increment beta towards 1
        self.beta = @min(1.0, self.beta + self.beta_increment);

        return .{
            .indices = indices,
            .weights = weights,
            .experiences = self.experiences.items,
            .allocator = self.allocator,
        };
    }

    /// Update priorities after training
    pub fn updatePriorities(self: *Self, indices: []const usize, td_errors: []const f32) void {
        const epsilon: f32 = 1e-6;
        for (indices, td_errors) |idx, td_err| {
            const old_priority = self.priorities.items[idx];
            const new_priority = @abs(td_err) + epsilon;
            self.total_priority -= std.math.pow(f64, old_priority, self.alpha);
            self.priorities.items[idx] = new_priority;
            self.total_priority += std.math.pow(f64, new_priority, self.alpha);
        }
    }

    pub fn len(self: *const Self) usize {
        return self.experiences.items.len;
    }
};

pub const SampledBatch = struct {
    indices: []usize,
    weights: []f32,
    experiences: []LearningExperience,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *SampledBatch) void {
        self.allocator.free(self.indices);
        self.allocator.free(self.weights);
    }
};

test {
    std.testing.refAllDecls(@This());
}
