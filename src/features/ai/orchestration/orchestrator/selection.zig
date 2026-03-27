const std = @import("std");
const utils = @import("../../../../foundation/mod.zig").utils;
const types = @import("../types.zig");

fn unixMs() i64 {
    return utils.unixMs();
}

pub fn route(
    self: anytype,
    prompt: []const u8,
    task_type: ?types.TaskType,
) types.OrchestrationError!types.RouteResult {
    self.mutex.lock();
    defer self.mutex.unlock();

    var available = std.ArrayListUnmanaged(*types.ModelEntry).empty;
    defer available.deinit(self.allocator);

    var it = self.models.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.isAvailable()) {
            available.append(self.allocator, entry.value_ptr) catch
                return types.OrchestrationError.OutOfMemory;
        }
    }

    if (available.items.len == 0) {
        return types.OrchestrationError.NoModelsAvailable;
    }

    const selected = selectModel(self, available.items, task_type);

    return types.RouteResult{
        .model_id = selected.config.id,
        .model_name = selected.config.model_name,
        .backend = selected.config.backend,
        .prompt = prompt,
    };
}

pub fn selectModel(
    self: anytype,
    available: []*types.ModelEntry,
    task_type: ?types.TaskType,
) *types.ModelEntry {
    return switch (self.config.strategy) {
        .round_robin => selectRoundRobin(self, available),
        .least_loaded => selectLeastLoaded(self, available),
        .task_based => selectByTask(self, available, task_type),
        .weighted => selectWeighted(self, available),
        .priority => selectByPriority(self, available),
        .cost_optimized => selectByCost(self, available),
        .latency_optimized => selectByLatency(self, available),
    };
}

fn selectRoundRobin(self: anytype, available: []*types.ModelEntry) *types.ModelEntry {
    const index = self.round_robin_index % available.len;
    self.round_robin_index += 1;
    return available[index];
}

fn selectLeastLoaded(self: anytype, available: []*types.ModelEntry) *types.ModelEntry {
    var min_load: f64 = 1.0;
    var selected: *types.ModelEntry = available[0];

    for (available) |model| {
        const load = model.loadFactor(self.config.max_concurrent_requests);
        if (load < min_load) {
            min_load = load;
            selected = model;
        }
    }

    return selected;
}

fn selectByTask(
    self: anytype,
    available: []*types.ModelEntry,
    task_type: ?types.TaskType,
) *types.ModelEntry {
    if (task_type == null) {
        return selectRoundRobin(self, available);
    }

    var best: ?*types.ModelEntry = null;
    var best_score: f64 = 0.0;

    for (available) |model| {
        if (modelSupportsTask(self, model, task_type.?)) {
            const score = model.successRate() * (1.0 - model.loadFactor(self.config.max_concurrent_requests));
            if (score > best_score) {
                best_score = score;
                best = model;
            }
        }
    }

    return best orelse selectRoundRobin(self, available);
}

fn selectWeighted(self: anytype, available: []*types.ModelEntry) *types.ModelEntry {
    _ = self;

    var total_weight: f64 = 0.0;
    for (available) |model| {
        total_weight += model.config.weight;
    }

    var prng = std.Random.DefaultPrng.init(@as(u64, @bitCast(unixMs())));
    const rand_val = prng.random().float(f64) * total_weight;
    var cumulative: f64 = 0.0;
    for (available) |model| {
        cumulative += model.config.weight;
        if (cumulative >= rand_val) {
            return model;
        }
    }

    return available[available.len - 1];
}

fn selectByPriority(self: anytype, available: []*types.ModelEntry) *types.ModelEntry {
    _ = self;
    var best: *types.ModelEntry = available[0];
    var best_priority: u32 = available[0].config.priority;

    for (available) |model| {
        if (model.config.priority < best_priority) {
            best_priority = model.config.priority;
            best = model;
        }
    }

    return best;
}

fn selectByCost(self: anytype, available: []*types.ModelEntry) *types.ModelEntry {
    _ = self;
    var cheapest: *types.ModelEntry = available[0];
    var min_cost: f32 = available[0].config.cost_per_1k_tokens;

    for (available) |model| {
        if (model.config.cost_per_1k_tokens < min_cost) {
            min_cost = model.config.cost_per_1k_tokens;
            cheapest = model;
        }
    }

    return cheapest;
}

fn selectByLatency(self: anytype, available: []*types.ModelEntry) *types.ModelEntry {
    _ = self;
    var fastest: *types.ModelEntry = available[0];
    var min_latency: f64 = available[0].avgLatencyMs();

    for (available) |model| {
        const latency = model.avgLatencyMs();
        if (latency < min_latency or (min_latency == 0.0 and latency == 0.0)) {
            min_latency = latency;
            fastest = model;
        }
    }

    return fastest;
}

pub fn modelSupportsTask(
    self: anytype,
    model: *types.ModelEntry,
    task_type: types.TaskType,
) bool {
    _ = self;
    const capability = taskToCapability(task_type);
    for (model.config.capabilities) |cap| {
        if (cap == capability) return true;
    }
    return model.config.capabilities.len == 0;
}

pub fn taskToCapability(task_type: types.TaskType) types.Capability {
    return switch (task_type) {
        .reasoning => .reasoning,
        .coding => .coding,
        .creative => .creative,
        .analysis => .analysis,
        .summarization => .summarization,
        .translation => .translation,
        .math => .math,
        .general => .reasoning,
    };
}

test {
    std.testing.refAllDecls(@This());
}
