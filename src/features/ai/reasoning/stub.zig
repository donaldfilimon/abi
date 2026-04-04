//! Reasoning Stub Module

const std = @import("std");
const core_types = @import("../core/types.zig");
const ai_config = @import("../../core/config/ai.zig");

// Import canonical types from core (shared with mod/engine)
pub const ConfidenceLevel = core_types.ConfidenceLevel;
pub const Confidence = core_types.Confidence;

pub const ReasoningConfig = ai_config.AiConfig.ReasoningConfig;

// Import canonical StepType from shared types (prevents drift with mod/engine)
pub const StepType = @import("types.zig").StepType;

pub const ReasoningStep = struct {
    step_type: StepType,
    description: []const u8,
    confidence: Confidence,
    timestamp_ns: i128 = 0,
    duration_ns: u64 = 0,

    pub fn format(self: ReasoningStep, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator, "{s}: {s}", .{
            self.step_type.toString(),
            self.description,
        });
    }
};

pub const ReasoningChain = struct {
    allocator: std.mem.Allocator,
    query: []const u8,
    steps: std.ArrayListUnmanaged(ReasoningStep) = .empty,
    start_time_ns: i128 = 0,
    finalized: bool = false,
    overall_confidence: ?Confidence = null,

    pub fn init(allocator: std.mem.Allocator, query: []const u8) ReasoningChain {
        return .{
            .allocator = allocator,
            .query = query,
        };
    }

    pub fn deinit(self: *ReasoningChain) void {
        self.steps.deinit(self.allocator);
    }

    pub fn addStep(
        self: *ReasoningChain,
        step_type: StepType,
        description: []const u8,
        confidence: Confidence,
    ) !void {
        try self.steps.append(self.allocator, .{
            .step_type = step_type,
            .description = description,
            .confidence = confidence,
        });
    }

    pub fn finalize(self: *ReasoningChain) !void {
        self.finalized = true;
        self.overall_confidence = self.getConfidence();
    }

    pub fn getOverallConfidence(self: *ReasoningChain) Confidence {
        if (!self.finalized) {
            self.finalize() catch |err|
                std.log.warn("failed to finalize reasoning chain: {}", .{err});
        }
        return self.getConfidence();
    }

    pub fn getConfidence(self: *const ReasoningChain) Confidence {
        return self.overall_confidence orelse .{
            .level = .unknown,
            .score = 0.0,
            .reasoning = "Reasoning is disabled",
        };
    }

    pub fn stepCount(self: *const ReasoningChain) usize {
        return self.steps.items.len;
    }

    pub fn researchTriggered(self: *const ReasoningChain) bool {
        for (self.steps.items) |step| {
            if (step.step_type == .research) return true;
        }
        return false;
    }

    pub fn getSummary(self: *ReasoningChain, allocator: std.mem.Allocator) ![]u8 {
        _ = self;
        return allocator.dupe(u8, "Reasoning is disabled");
    }
};

// Sub-module stub matching mod.zig's `pub const engine = @import("engine.zig")`
pub const engine = struct {
    pub const ConfidenceLevel = core_types.ConfidenceLevel;
    pub const Confidence = core_types.Confidence;
};

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: ReasoningConfig) !*Context {
        return error.FeatureDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn createChain(_: *Context, query: []const u8) ReasoningChain {
        return ReasoningChain.init(std.heap.page_allocator, query);
    }
};

pub fn isEnabled() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
