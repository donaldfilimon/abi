//! Unified Assistant Facade — Massive Framework Completion
//!
//! Ties together all core ABI features:
//! - Multi-profile routing (Abbey-Aviva-Abi)
//! - Internal API policy (model resolution)
//! - WDBX block-chained memory
//! - Adaptive EMA modulation
//! - Self-learning and feedback loops
//! - Post-generation Constitution checks
//!
//! This is the canonical entry point for high-level UI/CLI integrations.

const std = @import("std");
const ai_types = @import("../types.zig");
const types = @import("types.zig");
const registry_mod = @import("registry.zig");
const router_mod = @import("router.zig");
const modulation_mod = @import("../modulation.zig");
const constitution_mod = @import("../constitution/mod.zig");
const learning_mod = @import("../learning.zig");
const policy_mod = @import("../internal_api_policy.zig");
const build_options = @import("build_options");

pub const AssistantConfig = struct {
    session_id: []const u8 = "default",
    policy: policy_mod.Config = .{},
    routing: types.RoutingConfig = .{},
    multi_profile: registry_mod.MultiProfileConfig = .{},
    enable_learning: bool = true,
    enable_constitution: bool = true,
};

pub const AssistantResponse = struct {
    response: types.ProfileResponse,
    decision: types.RoutingDecision,
    latency_ms: f32,
    wdbx_block_id: ?u64,

    pub fn deinit(self: *AssistantResponse) void {
        var resp = self.response;
        resp.deinit();
    }
};

pub const Assistant = struct {
    allocator: std.mem.Allocator,
    registry: registry_mod.ProfileRegistry,
    router: router_mod.MultiProfileRouter,
    learning: ?learning_mod.LearningRuntime = null,
    modulator: ?*modulation_mod.AdaptiveModulator = null,
    config: AssistantConfig,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: AssistantConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        var registry = registry_mod.ProfileRegistry.init(allocator, config.multi_profile);
        var registry_needs_deinit = true;
        errdefer if (registry_needs_deinit) registry.deinit();
        try registry.initAll();

        self.* = .{
            .allocator = allocator,
            .registry = registry,
            .router = undefined,
            .learning = null,
            .modulator = null,
            .config = config,
        };
        registry_needs_deinit = false;
        errdefer self.registry.deinit();

        self.router = router_mod.MultiProfileRouter.init(allocator, &self.registry, config.routing);
        errdefer self.router.deinit();

        self.router.attachMemory(config.session_id);

        if (config.enable_constitution) {
            self.router.attachConstitution(constitution_mod.Constitution.init());
        }

        if (build_options.feat_ai) {
            self.modulator = try modulation_mod.AdaptiveModulator.init(allocator, .{});
            errdefer if (self.modulator) |m| m.deinit();
            self.router.attachModulator(self.modulator.?);
        }

        if (config.enable_learning) {
            self.learning = try learning_mod.LearningRuntime.init(allocator);
            errdefer if (self.learning) |*l| l.deinit();
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.learning) |*l| l.deinit();
        if (self.modulator) |m| m.deinit();
        self.router.deinit();
        self.registry.deinit();
        self.allocator.destroy(self);
    }

    pub fn process(self: *Self, input: []const u8) !AssistantResponse {
        const start_time = @import("../../../foundation/mod.zig").time.unixMs();

        // 1. Analyze and Route
        const decision = self.router.route(input);
        const request = ai_types.ProfileRequest{ .content = input };

        // 2. Execute based on decision
        const response = try self.router.execute(decision, request);

        // 3. Post-generation: validate against Constitution
        var validated_response = response;
        if (self.router.constitution) |c| {
            if (!c.isCompliant(response.content)) {
                response.allocator.free(response.content);

                const safe_msg = "I cannot provide this response as it may violate safety guidelines.";
                validated_response = types.ProfileResponse{
                    .profile = .abi,
                    .content = try self.allocator.dupe(u8, safe_msg),
                    .confidence = 1.0,
                    .allocator = self.allocator,
                };
            }
        }

        const model_resolution = try policy_mod.resolveModel(
            &.{ "abi/profile-router", "ollama/abbeycode", "llama_cpp/qwen2.5" },
            self.config.policy,
        );

        const end_time = @import("../../../foundation/mod.zig").time.unixMs();
        const latency = @as(f32, @floatFromInt(end_time - start_time));

        var block_id: ?u64 = null;
        if (self.router.memory) |*mem| {
            block_id = try mem.recordInteraction(decision, input, validated_response, null);
        }

        // Record interaction in learning runtime for telemetry
        if (self.learning) |*l| {
            try l.recordInteraction(.{
                .prompt = input,
                .response = validated_response.content,
                .profile = validated_response.profile.name(),
                .latency_ms = latency,
                .selected_model = model_resolution.selected_model,
                .wdbx_block_id = block_id,
                .route_reason = decision.reason,
            });
        }

        return AssistantResponse{
            .response = validated_response,
            .decision = decision,
            .latency_ms = latency,
            .wdbx_block_id = block_id,
        };
    }

    pub fn recordFeedback(self: *Self, kind: learning_mod.FeedbackKind, note: ?[]const u8) !void {
        if (self.learning) |*l| {
            try l.recordFeedback(kind, note);
        }

        // Also update the modulator if active
        if (self.modulator) |m| {
            const was_positive = kind == .positive;
            // Assuming Abbey for simple CLI feedback
            try m.recordInteraction(self.config.session_id, .abbey, was_positive);
        }
    }
};

test "assistant init and process" {
    if (!build_options.feat_ai) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var assistant = try Assistant.init(allocator, .{
        .session_id = "test-assistant",
    });
    defer assistant.deinit();

    const response = try assistant.process("Hello, are you connected?");
    defer {
        var r = response;
        r.deinit();
    }

    try std.testing.expect(response.response.content.len > 0);
    try std.testing.expect(response.latency_ms > 0);
}
