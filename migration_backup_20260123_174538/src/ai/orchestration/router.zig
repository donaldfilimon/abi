//! Model Routing Module
//!
//! Provides intelligent routing of requests to appropriate models based on
//! various strategies including task type, load balancing, and model capabilities.

const std = @import("std");
const mod = @import("mod.zig");

// ============================================================================
// Types
// ============================================================================

/// Strategy for routing requests to models.
pub const RoutingStrategy = enum {
    /// Distribute requests evenly across all models.
    round_robin,
    /// Route to model with fewest active requests.
    least_loaded,
    /// Route based on task type and model capabilities.
    task_based,
    /// Route based on configured weights.
    weighted,
    /// Route to highest priority available model.
    priority,
    /// Route to cheapest available model.
    cost_optimized,
    /// Route to model with lowest average latency.
    latency_optimized,

    pub fn toString(self: RoutingStrategy) []const u8 {
        return @tagName(self);
    }
};

/// Task type for task-based routing.
pub const TaskType = enum {
    /// Complex reasoning tasks.
    reasoning,
    /// Code generation and analysis.
    coding,
    /// Creative writing and content generation.
    creative,
    /// Data analysis and interpretation.
    analysis,
    /// Text summarization.
    summarization,
    /// Language translation.
    translation,
    /// Mathematical computations.
    math,
    /// General-purpose tasks.
    general,

    pub fn toString(self: TaskType) []const u8 {
        return @tagName(self);
    }

    /// Detect task type from prompt content (heuristic-based).
    pub fn detect(prompt: []const u8) TaskType {
        const lower = blk: {
            var buf: [512]u8 = undefined;
            const len = @min(prompt.len, buf.len);
            for (prompt[0..len], 0..) |c, i| {
                buf[i] = std.ascii.toLower(c);
            }
            break :blk buf[0..len];
        };

        // Code-related keywords
        if (std.mem.indexOf(u8, lower, "code") != null or
            std.mem.indexOf(u8, lower, "function") != null or
            std.mem.indexOf(u8, lower, "implement") != null or
            std.mem.indexOf(u8, lower, "debug") != null or
            std.mem.indexOf(u8, lower, "program") != null)
        {
            return .coding;
        }

        // Math-related keywords
        if (std.mem.indexOf(u8, lower, "calculate") != null or
            std.mem.indexOf(u8, lower, "solve") != null or
            std.mem.indexOf(u8, lower, "equation") != null or
            std.mem.indexOf(u8, lower, "math") != null)
        {
            return .math;
        }

        // Summarization keywords
        if (std.mem.indexOf(u8, lower, "summarize") != null or
            std.mem.indexOf(u8, lower, "summary") != null or
            std.mem.indexOf(u8, lower, "brief") != null or
            std.mem.indexOf(u8, lower, "tldr") != null)
        {
            return .summarization;
        }

        // Translation keywords
        if (std.mem.indexOf(u8, lower, "translate") != null or
            std.mem.indexOf(u8, lower, "translation") != null)
        {
            return .translation;
        }

        // Creative keywords
        if (std.mem.indexOf(u8, lower, "write") != null or
            std.mem.indexOf(u8, lower, "story") != null or
            std.mem.indexOf(u8, lower, "creative") != null or
            std.mem.indexOf(u8, lower, "poem") != null)
        {
            return .creative;
        }

        // Analysis keywords
        if (std.mem.indexOf(u8, lower, "analyze") != null or
            std.mem.indexOf(u8, lower, "analysis") != null or
            std.mem.indexOf(u8, lower, "examine") != null or
            std.mem.indexOf(u8, lower, "evaluate") != null)
        {
            return .analysis;
        }

        // Reasoning keywords
        if (std.mem.indexOf(u8, lower, "reason") != null or
            std.mem.indexOf(u8, lower, "explain") != null or
            std.mem.indexOf(u8, lower, "why") != null or
            std.mem.indexOf(u8, lower, "think") != null)
        {
            return .reasoning;
        }

        return .general;
    }
};

/// Result of routing decision.
pub const RouteResult = struct {
    /// ID of the selected model.
    model_id: []const u8,
    /// Name of the model.
    model_name: []const u8,
    /// Backend type.
    backend: mod.ModelBackend,
    /// Original prompt.
    prompt: []const u8,
    /// Detected or specified task type.
    task_type: ?TaskType = null,
    /// Confidence in the routing decision (0.0 to 1.0).
    confidence: f64 = 1.0,
    /// Reason for the routing decision.
    reason: ?[]const u8 = null,
};

/// Routing criteria for filtering models.
pub const RoutingCriteria = struct {
    /// Required task type (if any).
    task_type: ?TaskType = null,
    /// Required capabilities.
    required_capabilities: []const mod.Capability = &.{},
    /// Maximum acceptable latency in ms.
    max_latency_ms: ?u64 = null,
    /// Maximum acceptable cost per 1K tokens.
    max_cost_per_1k: ?f32 = null,
    /// Minimum required success rate.
    min_success_rate: ?f64 = null,
    /// Preferred backends.
    preferred_backends: []const mod.ModelBackend = &.{},
    /// Excluded model IDs.
    excluded_models: []const []const u8 = &.{},
};

// ============================================================================
// Router
// ============================================================================

/// Model router that selects the best model for a given request.
pub const Router = struct {
    allocator: std.mem.Allocator,
    default_strategy: RoutingStrategy,
    task_model_preferences: std.AutoHashMapUnmanaged(TaskType, []const u8),

    pub fn init(allocator: std.mem.Allocator, strategy: RoutingStrategy) Router {
        return .{
            .allocator = allocator,
            .default_strategy = strategy,
            .task_model_preferences = .{},
        };
    }

    pub fn deinit(self: *Router) void {
        self.task_model_preferences.deinit(self.allocator);
    }

    /// Set a preferred model for a specific task type.
    pub fn setTaskPreference(self: *Router, task: TaskType, model_id: []const u8) !void {
        try self.task_model_preferences.put(self.allocator, task, model_id);
    }

    /// Get preferred model for a task type.
    pub fn getTaskPreference(self: *Router, task: TaskType) ?[]const u8 {
        return self.task_model_preferences.get(task);
    }

    /// Clear task preferences.
    pub fn clearTaskPreferences(self: *Router) void {
        self.task_model_preferences.clearRetainingCapacity();
    }

    /// Score a model for a given criteria.
    pub fn scoreModel(
        _: *Router,
        model: *const mod.ModelEntry,
        criteria: RoutingCriteria,
    ) f64 {
        var score: f64 = 1.0;

        // Check exclusions
        for (criteria.excluded_models) |excluded| {
            if (std.mem.eql(u8, model.config.id, excluded)) {
                return 0.0;
            }
        }

        // Check latency constraint
        if (criteria.max_latency_ms) |max_lat| {
            const avg_lat = model.avgLatencyMs();
            if (avg_lat > @as(f64, @floatFromInt(max_lat))) {
                return 0.0;
            }
            // Bonus for lower latency
            score *= 1.0 - (avg_lat / @as(f64, @floatFromInt(max_lat)));
        }

        // Check cost constraint
        if (criteria.max_cost_per_1k) |max_cost| {
            if (model.config.cost_per_1k_tokens > max_cost) {
                return 0.0;
            }
            // Bonus for lower cost
            score *= 1.0 - (model.config.cost_per_1k_tokens / max_cost);
        }

        // Check success rate constraint
        if (criteria.min_success_rate) |min_rate| {
            const success_rate = model.successRate();
            if (success_rate < min_rate) {
                return 0.0;
            }
            score *= success_rate;
        }

        // Bonus for preferred backends
        if (criteria.preferred_backends.len > 0) {
            for (criteria.preferred_backends) |pref| {
                if (model.config.backend == pref) {
                    score *= 1.2;
                    break;
                }
            }
        }

        // Check required capabilities
        if (criteria.required_capabilities.len > 0) {
            var has_all = true;
            for (criteria.required_capabilities) |req_cap| {
                var found = false;
                for (model.config.capabilities) |cap| {
                    if (cap == req_cap) {
                        found = true;
                        break;
                    }
                }
                if (!found and model.config.capabilities.len > 0) {
                    has_all = false;
                    break;
                }
            }
            if (!has_all) {
                score *= 0.5; // Penalize but don't exclude
            }
        }

        // Factor in model weight
        score *= model.config.weight;

        // Factor in priority (inverse - lower priority number = higher score)
        score *= 1.0 / @as(f64, @floatFromInt(@max(model.config.priority, 1)));

        return @max(score, 0.0);
    }

    /// Select the best model from a list based on criteria.
    pub fn selectBest(
        self: *Router,
        models: []const *mod.ModelEntry,
        criteria: RoutingCriteria,
    ) ?*const mod.ModelEntry {
        if (models.len == 0) return null;

        var best: ?*const mod.ModelEntry = null;
        var best_score: f64 = 0.0;

        for (models) |model| {
            const score = self.scoreModel(model, criteria);
            if (score > best_score) {
                best_score = score;
                best = model;
            }
        }

        return best;
    }
};

// ============================================================================
// Routing Rules
// ============================================================================

/// Rule-based routing configuration.
pub const RoutingRule = struct {
    /// Rule name for identification.
    name: []const u8,
    /// Condition to match.
    condition: Condition,
    /// Action to take when condition matches.
    action: Action,
    /// Rule priority (higher = evaluated first).
    priority: u32 = 0,
    /// Whether rule is enabled.
    enabled: bool = true,

    pub const Condition = union(enum) {
        /// Match specific task type.
        task_type: TaskType,
        /// Match prompt containing text.
        prompt_contains: []const u8,
        /// Match prompt length range.
        prompt_length: struct { min: usize, max: usize },
        /// Always match.
        always: void,
        /// Match during time window (hour of day).
        time_window: struct { start_hour: u8, end_hour: u8 },
    };

    pub const Action = union(enum) {
        /// Route to specific model.
        route_to: []const u8,
        /// Use specific strategy.
        use_strategy: RoutingStrategy,
        /// Exclude specific models.
        exclude: []const []const u8,
        /// Require specific capabilities.
        require_capabilities: []const mod.Capability,
    };

    /// Check if this rule matches the given context.
    pub fn matches(self: RoutingRule, prompt: []const u8, task: ?TaskType) bool {
        if (!self.enabled) return false;

        return switch (self.condition) {
            .task_type => |tt| task != null and task.? == tt,
            .prompt_contains => |text| std.mem.indexOf(u8, prompt, text) != null,
            .prompt_length => |range| prompt.len >= range.min and prompt.len <= range.max,
            .always => true,
            .time_window => |window| blk: {
                const hour: u8 = @intCast(@mod(@divTrunc(std.time.timestamp(), 3600), 24));
                break :blk hour >= window.start_hour and hour < window.end_hour;
            },
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "task type detection" {
    try std.testing.expectEqual(TaskType.coding, TaskType.detect("Write a function to sort an array"));
    try std.testing.expectEqual(TaskType.math, TaskType.detect("Calculate the derivative of x^2"));
    try std.testing.expectEqual(TaskType.summarization, TaskType.detect("Summarize this article"));
    try std.testing.expectEqual(TaskType.translation, TaskType.detect("Translate this to French"));
    try std.testing.expectEqual(TaskType.creative, TaskType.detect("Write a short story about space"));
    try std.testing.expectEqual(TaskType.general, TaskType.detect("Hello, how are you?"));
}

test "router initialization" {
    var router_inst = Router.init(std.testing.allocator, .round_robin);
    defer router_inst.deinit();

    try std.testing.expectEqual(RoutingStrategy.round_robin, router_inst.default_strategy);
}

test "task preferences" {
    var router_inst = Router.init(std.testing.allocator, .task_based);
    defer router_inst.deinit();

    try router_inst.setTaskPreference(.coding, "codellama");

    const pref = router_inst.getTaskPreference(.coding);
    try std.testing.expect(pref != null);
    try std.testing.expectEqualStrings("codellama", pref.?);

    try std.testing.expect(router_inst.getTaskPreference(.math) == null);
}

test "routing rule matching" {
    const rule = RoutingRule{
        .name = "code-to-codellama",
        .condition = .{ .task_type = .coding },
        .action = .{ .route_to = "codellama" },
    };

    try std.testing.expect(rule.matches("test", .coding));
    try std.testing.expect(!rule.matches("test", .math));
    try std.testing.expect(!rule.matches("test", null));
}

test "prompt contains rule" {
    const rule = RoutingRule{
        .name = "python-code",
        .condition = .{ .prompt_contains = "python" },
        .action = .{ .route_to = "codellama" },
    };

    try std.testing.expect(rule.matches("Write Python code for sorting", null));
    try std.testing.expect(!rule.matches("Write JavaScript code for sorting", null));
}
